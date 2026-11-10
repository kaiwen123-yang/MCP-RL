#!/usr/bin/env python3
"""W44's four retained real problems, shared reconstruction and bounded solvers.

No plant episode, PPO step, fresh seed, or historical trajectory replay. The
frozen r33 binary's R32/S/N/P modes use identical recovered state and policy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from mujoco_mpc.agent import Agent
from s1_go2_r29_records import R29Records

ROOT = Path(__file__).resolve().parents[1]
REL = Path('outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
SMALL = ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w44-mechanisms'
OUT = ROOT/REL/'w44-mechanisms'
CASES = [('A221','a',1,'035658',221),('A1517','a',1,'035658',1517),
         ('B221','b',2,'035700',221),('B1517','b',2,'035700',1517)]
MISSING = ['Private Quadruped task transition/filter history and phase accumulator were not recorded.',
           'Original solver qacc warmstart and integration-substep contact solve memory were not recorded.',
           'Warm controller call history is replaced by the recovered full input policy and a common fresh task context.']


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def file_record(path):
    return dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path))


def cost_definitions(model_path,weights):
    definitions=[];offset=0
    for sensor in ET.parse(model_path).findall('.//sensor/user'):
        name=sensor.attrib['name'];dim=int(sensor.attrib['dim'])
        user=[float(x) for x in sensor.attrib['user'].split()]
        definitions.append(dict(name=name,start=offset,end=offset+dim,norm=int(user[0]),
            norm_parameter=user[4] if len(user)>4 else None,weight=weights[name]))
        offset+=dim
    return definitions


def cost_terms(residual,definitions):
    result={}
    for term in definitions:
        q=residual[:,term['start']:term['end']];p=term['norm_parameter']
        if term['norm']==0:v=.5*np.sum(q*q,axis=1)
        elif term['norm']==2:v=np.sqrt(np.sum(q*q,axis=1)+p*p)-p
        elif term['norm']==6:v=np.sum(np.sqrt(q*q+p*p)-p,axis=1)
        else:raise ValueError(f'Undefined norm {term}')
        result[term['name']]=float(np.mean(term['weight']*v))
    return result


def prepare_case(case):
    label,suffix,seed,stamp,call=case
    source=ROOT/REL/f'job-20261110T{stamp}Z-w43-{suffix}-eval-PANN_PPO_S5-s{seed}-E30'
    dest=OUT/label;dest.mkdir(parents=True,exist_ok=True)
    search_path=source/f'r32-search-call{call:06d}.npz'
    with np.load(search_path) as search:
        header=search['search_header'];origin=float(header[3]);pid=int(header[12])
        original_search={k:search[k].copy() for k in search.files}
    policy_path=source/f'accepted-reference-gain-H-plan{pid:06d}.npz'
    with np.load(policy_path) as z:policy={k:z[k].copy() for k in z.files}
    with np.load(source/'trajectory.npz') as tr:
        i=int(np.argmin(abs(tr['time']-origin)))
        state=dict(time=float(tr['time'][i]),qpos=tr['qpos'][i],qvel=tr['qvel'][i],act=np.zeros(0),
            mocap_pos=tr['mocap_pos'][i],mocap_quat=tr['mocap_quat'][i],
            userdata=tr['support_reference_userdata'][i])
        readback={k:tr[k][i] for k in ('q_target','actual_grf','contact','feet','body_target',
            'heading_target','command_velocity','phase','phase_replan','unclipped_pd_request')}
    if abs(state['time']-origin)>1e-8:raise ValueError('No exact saved physical input node')
    resolved_path=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'/f'{source.name}-resolved.json'
    resolved=json.loads(resolved_path.read_text())
    settings=[x for x in resolved['applied_settings'] if x['time_s']<=origin+1e-10][-1]
    # Snapshot stores the actual deployed feedback reference, not the accepted
    # rollout used to judge it. Preserve original complete policy arrays.
    arrays=[policy['header']]
    for key in ('reference_states','reference_actions','reference_times','reference_residual',
                'K','H','base_Qu','base_Qxu','base_active'):
        arrays.append(policy[key].ravel())
    np.concatenate(arrays).astype('<f8').tofile(dest/'input-policy.bin')
    np.savez_compressed(dest/'input-state.npz',**state,**{'readback_'+k:v for k,v in readback.items()})
    original_call=next(json.loads(line) for line in (source/'planner-calls.jsonl').open() if json.loads(line)['call_id']==call)
    model_path=source/'model/G1-project-go2.xml'
    record=dict(fixture=label,stream=suffix,source_call=call,origin_s=origin,
        input_policy_id=pid,source_index=i,settings=settings,
        sources=[file_record(p) for p in (search_path,policy_path,resolved_path,model_path)],
        state_source=str((source/'trajectory.npz').relative_to(ROOT)),
        input_policy=file_record(dest/'input-policy.bin'),input_state=file_record(dest/'input-state.npz'),
        model_path=str(model_path.relative_to(ROOT)),private_history_missing=MISSING,
        reconstruction='Fresh identical task transition context, exact saved state/time/mocap/userdata and complete input policy. Not an exact historical replay.',
        native_baseline='R32 mode in the same frozen r33 executable and reconstruction; candidate retention contract shared across modes.',
        recovered_policy_fields=list(policy),cost_definitions=cost_definitions(model_path,settings['cost_weights']),
        historical_input_reference_costs=cost_terms(policy['reference_residual'],cost_definitions(model_path,settings['cost_weights'])),
        historical_search_header=header.tolist(),original_call=original_call,
        original_physical_readback={k:np.asarray(v).tolist() for k,v in readback.items()},
        state_vs_saved_nominal_max_abs=float(np.max(abs(np.r_[state['qpos'],state['qvel']]-original_search['nominal_states'][0]))),
        position_goal=state['mocap_pos'].tolist(),current_local_baseline_required=True)
    write(dest/'input.json',record)
    return record


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def execute_case(label,mechanism,workers):
    dest=OUT/label;fixture=json.loads((dest/'input.json').read_text())
    run=dest/mechanism;run.mkdir(exist_ok=False)
    frozen=ROOT/'work/r11-frozen/S5-r33'
    model=mujoco.MjModel.from_xml_path(str(ROOT/fixture['model_path']))
    with np.load(dest/'input-state.npz') as saved:
        state={k:saved[k].copy() for k in ('time','qpos','qvel','act','mocap_pos','mocap_quat','userdata')}
    state['time']=float(state['time'])
    env=dict(os.environ,MJPC_TASKS_DIR=str(frozen/'runtime-tasks'),LD_LIBRARY_PATH=str(frozen/'lib'),
        MJPC_R33_INPUT_POLICY=str(dest/'input-policy.bin'),MJPC_R33_MECHANISM=mechanism,
        MJPC_R29_SNAPSHOT=str(run/'r29-current-plan.bin'),
        MJPC_R29_CALL_RECORD=str(run/'planner-calls.jsonl'),
        MJPC_R27_PLAN_RECORD=str(run/'planner-records.jsonl'),
        MJPC_R27_ACTION_RECORD=str(run/'feedback-actions.jsonl'))
    started=time.monotonic();record=dict(fixture=label,mechanism=mechanism,workers=workers,
        status='running',physical_episodes=0,source_call=fixture['source_call'],origin_s=fixture['origin_s'],
        shared_policy_sha256=fixture['input_policy']['sha256'],shared_state_sha256=fixture['input_state']['sha256'])
    write(run/'result.json',record)
    error=None
    try:
        with (run/'agent-server.log').open('w') as log,Agent(task_id='Quadruped Flat',model=model,
            server_binary_path=frozen/'agent_server',extra_flags=[f'--mjpc_workers={workers}'],
            subprocess_kwargs=dict(stdout=log,stderr=log,env=env),run_init=False) as agent:
            agent.init(task_id='Quadruped Flat',model=model,send_as='xml')
            agent.set_mode('Quadruped')
            agent.set_task_parameters({'Gait switch':'Manual','Gait':fixture['settings']['task_parameters']['Gait']})
            agent.set_state(**state)
            agent.set_task_parameters(fixture['settings']['task_parameters'])
            agent.set_cost_weights(fixture['settings']['cost_weights'])
            agent.set_state(**state)
            record['readback_parameters']=agent.get_task_parameters()
            record['readback_weights']=agent.get_cost_weights()
            record['readback_mode']=agent.get_mode()
            begin=time.monotonic();agent.planner_step();record['planner_rpc_wall_s']=time.monotonic()-begin
            record['installed_action_at_origin']=np.asarray(agent.get_action(time=state['time'])).tolist()
            records=R29Records(run,model);records.controller='S5-r33'
            records.plan(state['time'])
            records.event(state['time'],'saved_input_comparison',fixture=label,mechanism=mechanism)
            record['event_manifest']=records.finish(state['time'],'saved_input_completed_no_plant_step')
            record['status']='completed'
    except Exception as exc:
        error=exc;record.update(status='technical_failure',error=repr(exc),traceback=traceback.format_exc())
    record['total_wall_s']=time.monotonic()-started
    if (run/'planner-calls.jsonl').exists():record['calls']=read_jsonl(run/'planner-calls.jsonl')
    if (run/'planner-records.jsonl').exists():record['accepted_records']=read_jsonl(run/'planner-records.jsonl')
    write(run/'result.json',record)
    print(json.dumps({k:record[k] for k in ('fixture','mechanism','status','total_wall_s')}),flush=True)
    if error:raise error
    return record


def quantiles(values):
    values=np.asarray(values,dtype=float).ravel()
    return dict(zip(('min','median','max'),np.percentile(values,[0,50,100]).tolist()))



def policy_hash(arrays,feedback_scale,representation):
    digest=hashlib.sha256()
    for key in ('reference_states','reference_actions','reference_times','K','H','Qu','Qxu','active'):
        digest.update(np.asarray(arrays[key],dtype='<f8').tobytes())
    digest.update(np.asarray([feedback_scale,representation],dtype='<f8').tobytes())
    return digest.hexdigest()


def take_policy_arrays(raw,cursor,T,ns,nu,nx,nr):
    arrays={}
    for prefix in ('','reference_'):
        for key,shape in [('states',(T,ns)),('actions',(T-1,nu)),('times',(T,)),('residual',(T,nr))]:
            count=int(np.prod(shape));arrays[prefix+key]=raw[cursor:cursor+count].reshape(shape);cursor+=count
    for key,shape in [('K',(T,nu,nx)),('H',(T,nu,nu)),('Qu',(T,nu)),('Qxu',(T,nx,nu)),('active',(T,nu))]:
        count=int(np.prod(shape));arrays[key]=raw[cursor:cursor+count].reshape(shape);cursor+=count
    return arrays,cursor


def parse_candidates(path):
    raw=np.fromfile(path,dtype='<f8');cursor=0;result=[]
    while cursor<len(raw):
        h=raw[cursor:cursor+19];cursor+=19
        if h[0]!=33:raise ValueError('Invalid r33 candidate record schema')
        T,ns,nu,nx,nr=map(int,h[5:10])
        arrays,cursor=take_policy_arrays(raw,cursor,T,ns,nu,nx,nr)
        for field in ('prefix','feet'):
            count=int(raw[cursor]);cursor+=1;arrays[field]=raw[cursor:cursor+count];cursor+=count
        meta=dict(candidate_record=len(result),call_id=int(h[1]),kind=int(h[2]),slot=int(h[3]),
            alpha=float(h[4]),horizon=T,feedback_scale=float(h[10]),native_J=float(h[11]),
            trunk_violation=float(h[12]),failure=bool(h[13]),metric_valid=bool(h[14]),
            origin_s=float(h[15]),mu=float(h[16]),representation=int(h[17]),input_policy_id=int(h[18]))
        result.append(dict(metadata=meta,arrays=arrays))
    return result


def analyze_case(label,mechanism):
    dest=OUT/label;fixture=json.loads((dest/'input.json').read_text());run=dest/mechanism
    result=json.loads((run/'result.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(ROOT/fixture['model_path']))
    candidates=parse_candidates(run/'r29-current-plan.bin.r33-candidates')
    summaries=[]
    for candidate in candidates:
        meta=candidate['metadata']
        if meta['failure'] or not np.isfinite([meta['native_J'],meta['trunk_violation']]).all():
            summary={**meta,'failed_trajectory_not_interpreted_as_motion':True}
            for key in ('native_J','trunk_violation'):
                if not np.isfinite(summary[key]):summary[key]=None
        else:summary=candidate_summary(candidate,fixture,model)
        summaries.append(summary)
    raw=np.fromfile(run/'r29-current-plan.bin',dtype='<f8');h=raw[:16]
    arrays,_=take_policy_arrays(raw,16,*map(int,h[3:8]))
    selected_hash=policy_hash(arrays,float(h[12]),int(h[13]))
    selected=[c for c in summaries if c.get('policy_sha256')==selected_hash]
    if not selected:raise ValueError('Installed policy has no identical complete native-scored candidate')
    call=result['calls'][-1]
    selected=[c for c in selected if c['native_J']==call['deployed_current_cost'] and c['trunk_violation']==call['deployed_current_trunk_violation']]
    if not selected:raise ValueError('Installed policy score differs from matching native-scored candidate')
    chosen=selected[0]
    result.update(candidates=summaries,selected=chosen,selected_policy_sha256=selected_hash,
        matching_scored_candidate_records=[c['candidate_record'] for c in selected],
        full_policy_native_installation_identity_verified=True,
        cost_recomputation_max_abs_error=max(abs(c.get('native_J_recompute_error',0)) for c in summaries))
    scale=run/'r29-current-plan.bin.r33-scale'
    if scale.exists():
        raw=np.fromfile(scale,dtype='<f8');D=raw[6:].reshape(int(raw[2]),int(raw[3]))
        result['scale']=dict(source_mean=float(raw[4]),degenerate_floor=float(raw[5]),
            D=quantiles(D),D_sha256=hashlib.sha256(D.astype('<f8').tobytes()).hexdigest(),
            units='Native trajectory cost / radian^2; mu dimensionless')
    result['raw_candidate_file']=file_record(run/'r29-current-plan.bin.r33-candidates')
    write(run/'analysis.json',result)
    return result


def summarize(cases,mechanisms):
    results=[analyze_case(c[0],m) for c in cases for m in mechanisms]
    contexts={};scale_pairs={}
    for case in cases:
        records=[r for r in results if r['fixture']==case[0]]
        ref=records[0]['calls'][-1]
        contexts[case[0]]={key:all(r['calls'][-1][key]==ref[key] for r in records)
            for key in ('task_weights','task_parameters','gait_context')}
        if not all(contexts[case[0]].values()):raise ValueError('Methods did not share the same reconstructed task context')
        by_mode={r['mechanism']:r for r in records}
        if 'S' in by_mode and 'N' in by_mode:
            scale_pairs[case[0]]=by_mode['S']['scale']['D_sha256']==by_mode['N']['scale']['D_sha256']
            if not scale_pairs[case[0]]:raise ValueError('S and N did not share identical D')
    report=dict(schema='w44_finite_mechanism_comparison_v1',status='completed',physical_episodes=0,
        historical_replay=False,private_history_missing=MISSING,same_task_context=contexts,
        same_S_N_D=scale_pairs,methods=mechanisms,results=results,
        comparison_scope='One original saved problem at each of four actual W43 E30 inputs. One successful local improvement; fixed attempt/alpha/P32 budgets. No motion qualification from local prediction.',
        budget_note='P32 candidate slots are not worker count or equal compute to S/N; native/FD/nominal calls and walltime are separately reported.')
    write(SMALL/'summary.json',report)
    print(json.dumps([dict(fixture=r['fixture'],mechanism=r['mechanism'],source=r['calls'][-1]['candidate_source'],
        J=r['selected']['native_J'],v=r['selected']['trunk_violation'],dx=r['selected']['predicted_dx_m'],
        feet=r['selected']['feet']['maximum_bottom_per_foot_m'],wall_s=r['total_wall_s']) for r in results],indent=2))
    return report


def candidate_summary(candidate,fixture,model):
    arrays=candidate['arrays'];meta=candidate['metadata']
    states=arrays['states'];actions=arrays['actions'];nr=len(actions)
    joints=model.actuator_trnid[:,0]
    raw_pd=60*(actions-states[:nr,model.jnt_qposadr[joints]])-5*states[:nr,model.nq+model.jnt_dofadr[joints]]
    terms=cost_terms(arrays['residual'],fixture['cost_definitions'])
    policy_digest=policy_hash(arrays,meta['feedback_scale'],meta['representation'])
    record=dict(**meta,cost_components=terms,cost_recomputed=sum(terms.values()),
        native_J_recompute_error=sum(terms.values())-meta['native_J'],
        predicted_dx_m=float(states[-1,0]-states[0,0]),
        predicted_dz_m=float(states[-1,2]-states[0,2]),
        vx_m_s=quantiles(states[:,model.nq]),
        target_rad=quantiles(actions),raw_PD_request_Nm=quantiles(raw_pd),
        raw_PD_outside_actuator_force_range_count=int(np.sum((raw_pd<model.actuator_forcerange[:,0])|(raw_pd>model.actuator_forcerange[:,1]))),
        policy_sha256=policy_digest,
        K_abs_max=float(np.max(abs(arrays['K']))),
        H_diagonal=quantiles(np.diagonal(arrays['H'][:-1],axis1=1,axis2=2)))
    feet=arrays['feet'].reshape(-1,29)
    if len(feet):
        ff=feet[:,1:].reshape(-1,4,7)
        forces=ff[:,:,4]+ff[:,:,5]
        record['feet']=dict(order=['FL','FR','RL','RR'],samples=len(feet),
            time_first_s=float(feet[0,0]),time_last_s=float(feet[-1,0]),
            bottom_height_m=quantiles(ff[:,:,3]),
            maximum_bottom_per_foot_m=np.max(ff[:,:,3],axis=0).tolist(),
            native_Fz_N=quantiles(ff[:,:,4]),proxy_Fz_N=quantiles(ff[:,:,5]),
            total_predicted_Fz_N=quantiles(forces),
            minimum_force_per_foot_N=np.min(forces,axis=0).tolist(),
            native_zero_contact_nodes_per_foot=np.sum(ff[:,:,6]==0,axis=0).tolist(),
            proxy_positive_nodes_per_foot=np.sum(ff[:,:,5]>0,axis=0).tolist(),
            geometric_bottom_positive_nodes_per_foot=np.sum(ff[:,:,3]>0,axis=0).tolist(),
            zero_native_contact_and_nonpositive_proxy_nodes_per_foot=np.sum((ff[:,:,6]==0)&(ff[:,:,5]<=0),axis=0).tolist(),
            semantics='Native rollout pre-integration/terminal observations; proxy forces are planner material proxy, not real TM02 measurements.')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--summarize',action='store_true')
    parser.add_argument('--case',choices=[c[0] for c in CASES],action='append')
    parser.add_argument('--mechanism',choices=['R32','S','N','P'],action='append')
    parser.add_argument('--workers',type=int,default=10)
    args=parser.parse_args()
    cases=[c for c in CASES if not args.case or c[0] in args.case]
    if args.prepare:
        records=[prepare_case(c) for c in cases]
        write(SMALL/'inputs.json',dict(schema='w44_same_snapshot_v1',physical_episodes=0,
            private_history_missing=MISSING,fixtures=records))
        print(json.dumps(dict(prepared=[x['fixture'] for x in records]),indent=2))
    if args.run:
        for case in cases:
            for mechanism in args.mechanism or ['R32','S','N','P']:
                execute_case(case[0],mechanism,args.workers)
    if args.summarize:summarize(cases,args.mechanism or ['R32','S','N','P'])

if __name__=='__main__':main()
