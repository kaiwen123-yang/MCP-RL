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
    small=output/'episodes'; large=output/'streams'
    for path in (small,large,output/'raw'):
        path.mkdir(parents=True,exist_ok=True)
    parent_ck=Path(command[command.index('--checkpoint')+1])
    identity=json.loads((root/'pair-runtime.json').read_text())
    contract.update(run_id='w61-bd-m1-e60-v1', current_task='W61_BD_M1_E60',
        protocol_revision='W61_FROZEN_B3_BD_E60_V1',w57_raw_root=str(output/'raw'),
        outputs={'large_root':str(large),'compact_root':str(small)},w61=dict(case=args.case,
            gait_role=args.gait,role=args.wbc,artifact=str(root/'phase-basis.json'),source_identity=identity,
            original_attempt=parent,original_actor_sha256=sha(parent_ck),
            original_contract_sha256=sha(inherited),frozen_original_qualification=False))
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
        if args.wbc=='reduced':
            added.append('w61_equality_wbc.py')
        c['w54_execution_sources']+=['scripts/'+name for name in added]
        c['w54_execution_source_names']+=added
    run_s1_w59_research.configure_episode=configure_pair
    job=output/'jobs'/attempt;job.mkdir(parents=True,exist_ok=False)
    path=job/'contract.json';path.write_text(json.dumps(contract,indent=2)+'\n')
    a=SimpleNamespace(attempt=attempt,contract=str(path),group='PANN_PPO_S5',seed=receipt['seed'],
        build=receipt['controller'],workers=1,level='E60',condition='nominal',assets=shared/'work/assets',
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
    p.add_argument('--gait',choices=['fixed','orbit'],required=True)
    p.add_argument('--wbc',choices=['fixed','hybrid','reduced'],required=True)
    p.add_argument('--series',default='w61-bd-v1')
    p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    main(p.parse_args())
