#!/usr/bin/env python3
"""Export saved E30 event windows; kinematics only, never integrate/replay physics."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco

PHASES=['initial_stand','outbound','turn','return','terminal_stand','complete','far_hold','home_hold','turn_home']

def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()] if path.exists() else []

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',',':'), allow_nan=False)+'\n')

def export_episode(episode_dir, destination, result):
    p, out = Path(episode_dir), Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    missing=[]; paths=[]
    required=['trajectory.npz','feedback-actions.jsonl','planner-records.jsonl','model.mjb']
    missing=[name for name in required if not (p/name).exists()]
    if missing:
        summary={'episode':p.name,'status':'unavailable_saved_sources','missing':missing,'new_physics':False}
        path=out/'event-summary.json';dump(path,summary)
        return dict(paths=[str(path)], summary=summary, missing=missing)
    traj=dict(np.load(p/'trajectory.npz'))
    times=traj['time']; terminal=float(result.get('simulation_s',times[-1]))
    outbound=np.flatnonzero(traj['phase']==1)
    start=float(times[outbound[0]]) if len(outbound) else None
    actions=read_jsonl(p/'feedback-actions.jsonl');plans=read_jsonl(p/'planner-records.jsonl')
    material=read_jsonl(p/'material-events.jsonl')
    physical_parts=[np.load(f)['physical'] for f in sorted(p.glob('physical-*.npz'))]
    physical=np.concatenate(physical_parts) if physical_parts else np.zeros((0,61))
    if not physical_parts: missing.append('physical-*.npz: per-substep values unavailable')
    model=mujoco.MjModel.from_binary_path(str(p/'model.mjb'));data=mujoco.MjData(model)
    floor=model.geom('floor').id; trunk=model.body('trunk').id
    geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g]==trunk and
           ((model.geom_contype[g]&model.geom_conaffinity[floor]) or (model.geom_contype[floor]&model.geom_conaffinity[g]))]
    qadr=model.jnt_qposadr[model.actuator_trnid[:,0]];vadr=model.jnt_dofadr[model.actuator_trnid[:,0]]
    urange=model.actuator_ctrlrange; frange=model.actuator_forcerange
    def geometry(state):
        data.qpos[:]=np.asarray(state)[:model.nq]
        mujoco.mj_kinematics(model,data)
        return [{'geom':g,'margin_m':float(max(model.geom_margin[floor],model.geom_margin[g])),
                 'gap_minus_margin_m':float(mujoco.mj_geomDistance(model,data,floor,g,1,None)-max(model.geom_margin[floor],model.geom_margin[g]))} for g in geoms]
    # The RPC protobuf time is float32. Join its exact quantization to the
    # double control clock, and require the same float32 stored target.
    valid=np.isclose(times,traj['diagnostic_time'],rtol=0,atol=1e-10)&(times>0)
    bytime={float(np.float32(times[i])):int(i) for i in np.flatnonzero(valid)}
    executed={}; excluded=0
    for a in actions:
        i=bytime.get(a['time'])
        if i is not None and np.array_equal(np.asarray(a['action_after_clip'],dtype=np.float32),traj['q_target'][i]): executed[i]=a
        else: excluded+=1
    def derived(i,a):
        q=traj['qpos'][i,qadr];v=traj['qvel'][i,vadr]
        nom=np.clip(a['nominal_unclipped'],urange[:,0],urange[:,1]);actual=traj['q_target'][i]
        raw_nom=60*(nom-q)-5*v;raw_exec=60*(actual-q)-5*v
        sat_nom=(raw_nom<frange[:,0])|(raw_nom>frange[:,1]);sat_exec=(raw_exec<frange[:,0])|(raw_exec>frange[:,1])
        return dict(control_time_s=float(times[i]),rpc_time_s=a['time'],plan_id=a['plan_id'],
            nominal_raw_pd_Nm=raw_nom.tolist(),nominal_clipped_pd_Nm=np.clip(raw_nom,frange[:,0],frange[:,1]).tolist(),
            executed_raw_pd_recomputed_Nm=raw_exec.tolist(),executed_clipped_pd_Nm=np.clip(raw_exec,frange[:,0],frange[:,1]).tolist(),
            nominal_saturated_actuators=np.flatnonzero(sat_nom).tolist(),executed_saturated_actuators=np.flatnonzero(sat_exec).tolist(),
            feedback_newly_saturated_actuators=np.flatnonzero(sat_exec&~sat_nom).tolist(),
            tangent_q_error_norm=float(np.linalg.norm(a['state_delta'][:model.nv])),
            tangent_v_error_norm=float(np.linalg.norm(a['state_delta'][model.nv:2*model.nv])),
            max_abs_raw_K_delta_x_rad=float(np.max(np.abs(a['raw_K_delta_x']))),
            max_target_clip_rad=float(np.max(np.abs(np.asarray(a['action_before_clip'])-a['action_after_clip']))),
            action_equation_max_error=float(np.max(np.abs(np.asarray(a['nominal_unclipped'])+a['feedback_scaling']*np.asarray(a['raw_K_delta_x'])-a['action_before_clip']))),
            actual_trunk_geometry=geometry(traj['qpos'][i]))
    events={}
    if start is not None:
        for name, predicate in [('target_clipping',lambda a:np.any(np.asarray(a['action_before_clip'])!=a['action_after_clip'])),
                                ('large_raw_K_delta_x',lambda a:np.max(np.abs(a['raw_K_delta_x']))>.5)]:
            match=next(((i,a) for i,a in sorted(executed.items()) if times[i]>=start and predicate(a)),None)
            events[name]=dict(time_s=float(times[match[0]]),source='executed RPC joined to control trajectory',control=derived(*match)) if match else None
        if len(physical):
            raw=physical[:,49:61];mask=(physical[:,0]>=start)
            for name,where in [('raw_PD_saturation',np.any((raw<frange[:,0])|(raw>frange[:,1]),axis=1)),
                               ('nonfoot_contact',physical[:,48]>0),('nonfoot_load',physical[:,47]>0)]:
                ix=np.flatnonzero(mask&where)
                if len(ix):
                    row=physical[ix[0]];i=int(np.searchsorted(times,row[0]+1e-9)-1)
                    events[name]=dict(time_s=float(row[0]),post_integration_time_s=float(row[1]),
                        physical_row=row.tolist(),control=(derived(i,executed[i]) if i in executed else None))
                else:events[name]=None
        else:
            events.update(raw_PD_saturation=None,nonfoot_contact=None,nonfoot_load=None)
        load=next((e for e in material if e['type']=='load' and e['time_s']>=start),None)
        events['material_load']=dict(time_s=load['time_s'],event=load) if load else None
    events['terminal']=dict(time_s=terminal,reason=result.get('reason'))
    accepted_violations=[];plan_geometry={}
    for plan in plans:
        nodes=[]
        for n in plan.get('nodes',[]):
            g=geometry(n['accepted_state']);worst=min(g,key=lambda x:x['gap_minus_margin_m'])
            nodes.append(dict(node=n['node'],time_s=n['t'],accepted=g,
                feedback_reference=geometry(n['feedback_reference_state']),linearization=geometry(n['linearization_state'])))
            if worst['gap_minus_margin_m']<0:
                accepted_violations.append(dict(origin_s=plan['origin'],plan_id=plan['plan_id'],node=n['node'],time_s=n['t'],**worst))
        plan_geometry[plan['plan_id']]=nodes
    if accepted_violations:
        first=accepted_violations[0];events['first_observed_accepted_trunk_violation']=dict(time_s=first['origin_s'],evidence=first)
    intervals=[]
    for ev in events.values():
        if ev is not None:intervals.append([max(0.,ev['time_s']-.1),min(terminal,ev['time_s']+.2)])
    intervals.append([max(0.,terminal-.1),terminal]);merged=[]
    for lo,hi in sorted(intervals):
        if merged and lo<=merged[-1][1]+1e-9:merged[-1][1]=max(hi,merged[-1][1])
        else:merged.append([lo,hi])
    nonfoot=[]
    for w,(lo,hi) in enumerate(merged):
        ti=np.flatnonzero((times>=lo-1e-9)&(times<=hi+1e-9));pi=np.flatnonzero((physical[:,0]>=lo-1e-9)&(physical[:,0]<=hi+1e-9))
        aa=[dict(control_index=int(i),control_time_s=float(times[i]),record=executed[i],derived=derived(i,executed[i])) for i in ti if i in executed]
        ids={a['record']['plan_id'] for a in aa};pp=[dict(p,geometry=plan_geometry[p['plan_id']]) for p in plans if p['plan_id'] in ids or lo<=p['origin']<=hi]
        # Each source is independently named so required small evidence cannot
        # be hidden by a generic raw-file or 2MB filter.
        for label,value in [('control',dict(columns={k:v[ti].tolist() for k,v in traj.items()},actions=aa)),
                            ('physical',dict(columns=result.get('physical_columns'),rows=physical[pi].tolist())),
                            ('plans',dict(records=pp)),
                            ('material',dict(events=[e for e in material if lo<=e['time_s']<=hi]))]:
            path=out/f'window-{w:02d}-{label}.json';dump(path,dict(episode=p.name,window_s=[lo,hi],**value));paths.append(str(path))
        if len(pi):
            rows=physical[pi];nf=np.maximum(rows[:,47],0);total=np.maximum(rows[:,44],0)
            nonfoot.append(dict(window_s=[lo,hi],positive_nonfoot_impulse_Ns=float(np.sum(nf*(rows[:,1]-rows[:,0]))),
                positive_total_impulse_Ns=float(np.sum(total*(rows[:,1]-rows[:,0]))),
                nonfoot_impulse_fraction=float(np.sum(nf)/np.sum(total)) if np.sum(total)>0 else None,
                maximum_nonfoot_Fz_N=float(np.max(nf)),maximum_total_Fz_N=float(np.max(total))))
    summary=dict(episode=p.name,source_directory=str(p),selection='First events after Stand->outbound; [-.10,+.20] merged plus terminal .10; accepted violating plan included.',
        source_model='Saved episode model.mjb; mj_kinematics + original mj_geomDistance only',new_physics=False,
        outbound_time_s=start,events=events,windows_s=merged,selected_duration_s=sum(hi-lo for lo,hi in merged),
        accepted_violations=accepted_violations,accepted_coverage='Only stored node0 and node10(+20ms); no full accepted horizon or between-node claim.',
        prediction_npz_role='feedback reference; never substituted for accepted trajectory',
        clocks=dict(control='trajectory time float64; action RPC time float32 quantization',physical='force-evaluation time precedes post-integration time',
                    excluded_rpc_rows=excluded,unique_executed_controls=len(executed),mismatched_terminal_rows=np.flatnonzero(~np.isclose(times,traj['diagnostic_time'],rtol=0,atol=1e-10)).tolist()),
        foot_order=dict(runtime=['FL','FR','RL','RR'],risk=['FL','RL','FR','RR'],risk_to_runtime=[0,2,1,3]),
        actuator_names=[model.actuator(i).name for i in range(model.nu)],ctrlrange=urange.tolist(),forcerange=frange.tolist(),kp=60,kd=5,
        original_protected_trunk_geoms=geoms,nonfoot_support_windows=nonfoot,
        missing=missing+['Actual qpos/qvel are 500Hz, not per 0.2ms; nominal-at-actual-state comparison uses stored control states only.',
                         'No actual contact forces are synthesized from stored accepted/reference states.'],
        terminal_result={k:result[k] for k in ['attempt','controller_build','reason','simulation_s','stage_status','success','legal_route','source_commit','execution_fingerprint'] if k in result})
    path=out/'event-summary.json';dump(path,summary);paths.append(str(path))
    return dict(paths=paths,summary=summary,missing=summary['missing'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('episode_dir',type=Path);ap.add_argument('destination',type=Path);ap.add_argument('result',type=Path);args=ap.parse_args()
    exported=export_episode(args.episode_dir,args.destination,json.loads(args.result.read_text()))
    print(json.dumps(dict(files=len(exported['paths']),summary=str(args.destination/'event-summary.json'),events={k:(v['time_s'] if v else None) for k,v in exported['summary'].get('events',{}).items()})))
