"""Run one W61 B x D assignment through the original 60 s route, M1 and W44 cap actor."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import run_s1_go2_r12
import run_s1_w59_research


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):
            h.update(chunk)
    return h.hexdigest()


def main(args):
    root=Path(__file__).resolve().parents[1]
    shared=args.source.resolve()
    run_id='tm-s1-go2-local-readiness-r12'
    parent=f'w59-b3-{args.case.lower()}-original-e60-a01'
    raw=shared/'outputs/terrain_memory/simulation'/run_id
    old=shared/'results/terrain_memory/simulation'/run_id/'w59/attempts'/(parent+'.json')
    receipt=json.loads(old.read_text());command=receipt['command']
    inherited=raw/'jobs'/parent/'execution-contract.json'
    contract=json.loads(inherited.read_text())
    output=args.out.resolve()
    attempt=f'{args.series}-{args.case.lower()}-{args.gait}-{args.wbc}'
    style_seed=int(args.gait[5:]) if args.gait.startswith('style') else None
    decoder_seed=int(args.gait) if args.gait in ('6111','6112','6113') else None
    fitted=root/f'orbit-seed-{decoder_seed}.json' if decoder_seed else root/'phase-basis.json'
    if style_seed:
        fitted=root/f'style-seed-{style_seed}.json'
    gait='orbit' if decoder_seed else 'cpg' if style_seed else args.gait
    small=output/'episodes'; large=output/'streams'
    for path in (small,large,output/'raw'):
        path.mkdir(parents=True,exist_ok=True)
    parent_ck=Path(command[command.index('--checkpoint')+1])
    identity=json.loads((root/'pair-runtime.json').read_text())
    contract.update(run_id='w61-bd-m1-e60-v1', current_task='W61_BD_M1_E60',
        protocol_revision='W61_FROZEN_B3_BD_E60_V1',w57_raw_root=str(output/'raw'),
        outputs={'large_root':str(large),'compact_root':str(small)},w61=dict(case=args.case,
            gait_role=gait,role=args.wbc,orbit_decoder_seed=decoder_seed,style_cpg_seed=style_seed,artifact=str(fitted),
            observation_window_s=args.window_s,initial_condition=args.condition,source_identity=identity,
            original_attempt=parent,original_actor_sha256=sha(parent_ck),
            original_contract_sha256=sha(inherited),frozen_original_qualification=False))
    if args.window_s:
        contract.update(run_id='w61-gait-m1-first-traverse-v1',current_task='W61_GAIT_M1_FIRST_TRAVERSE',
                        protocol_revision='W61_FIXED_M1_FIRST_OUTBOUND16_V1')
    contract['w57']['task_mode']='w61_paired'
    # Configure the executable using the standard W59 loader first. Only the versioned
    # gait/WBC treatment changes. Case actor, normalizer, material, MPC and plant keep their owning code.
    configure_original=run_s1_w59_research.configure_episode
    def configure_pair(c,spec):
        configure_original(c,spec)
        c['w54_backend_source']='scripts/w61_b3_components.py'
        c['w54_f']['w61']=spec['w61'];c['r11_stage']['qualification_usable']=False
        added=['w61_b3_components.py','w61_controlled_orbits.py','w61_support_bridge.py',
               'w61_wbc_internal_model.py','w61_natural_wbc.py','w61_b3_pair.py']
        if decoder_seed or style_seed:
            added.append('w61_orbit_decoder.py')
        if style_seed:
            added.append('w61_style_orbit.py')
        if args.window_s:
            c['r11_stage'].update(level='W61_FIRST_OUTBOUND16',task_kind='first_outbound_observation',
                                 duration_s=args.window_s,qualification_usable=False)
            c['task']['maximum_episode_s']=args.window_s
        if args.wbc=='reduced':
            added.append('w61_equality_wbc.py')
        c['w54_execution_sources']+=['scripts/'+name for name in added]
        c['w54_execution_source_names']+=added
    run_s1_w59_research.configure_episode=configure_pair
    if args.window_s:
        import s1_go2_r11_protocol
        def evaluate_window(result,c,joint,arrays):
            # A complete observation and a successful traverse are distinct.
            full=result['simulation_s']>=args.window_s-1.e-7
            far=result['max_x_m']>=2.40
            error=result['reason']=='execution_error'
            domain=(result['reason'] or '').startswith(('body_fall','invalid_contact','material_domain_exit'))
            loaded=bool(joint.ground.ever_loaded.any())
            traversed=bool(full and far and loaded and not error and not domain)
            status=('technical_interruption' if error else 'physical_failure' if domain else
                    'first_outbound_reached' if traversed else 'first_outbound_incomplete')
            result.update(success=traversed,stage_status=status,qualification_usable=False,
                original_mission_success=None,sustained_execution_success=None,evaluation_stage=c['r11_stage'],
                case_key=c['case_key'],case_identity=c['case_identity'],qualification=dict(usable=False,
                role='W61 new first-outbound 16s observation with a 2.40m body crossing and actual material loading'),
                stage_progress=dict(full_observation=bool(full),body_crossed_2p40m=bool(far),material_loaded=loaded))
            return result
        s1_go2_r11_protocol.evaluate_stage=evaluate_window
    job=output/'jobs'/attempt;job.mkdir(parents=True,exist_ok=False)
    path=job/'contract.json';path.write_text(json.dumps(contract,indent=2)+'\n')
    a=SimpleNamespace(attempt=attempt,contract=str(path),group='PANN_PPO_S5',seed=receipt['seed'],
        build=receipt['controller'],workers=1,level='E60',condition=args.condition,assets=shared/'work/assets',
        checkpoint=str(parent_ck),window_start_unix=time.time(),deadline_unix=None,policy_output=large/attempt)
    if a.build!=contract['reuse']['active_controller_build']:
        raise ValueError('Source actor and controller build disagree')
    print(json.dumps(dict(attempt=attempt,source=identity,actor_sha256=contract['w61']['original_actor_sha256'],
                          pid_command=sys.argv)),flush=True)
    res=run_s1_go2_r12.evaluate(a,contract)
    print(json.dumps({k:res.get(k) for k in ('attempt','reason','simulation_s','stage_status','success','wall_s')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--case',choices=['A','B'],required=True)
    p.add_argument('--gait',choices=['fixed','library','cpg','orbit','6111','6112','6113','style6111','style6112','style6113'],required=True)
    p.add_argument('--wbc',choices=['fixed','hybrid','reduced'],required=True)
    p.add_argument('--series',default='w61-bd-v1')
    p.add_argument('--window-s',type=float,choices=[16.])
    p.add_argument('--condition',choices=['nominal','yaw_plus5','yaw_minus5'],default='nominal')
    p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    main(p.parse_args())
