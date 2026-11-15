#!/usr/bin/env python3
"""W50 A: analyze retained W49 arrays and clocks; no solver or integration."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REL = Path('outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
RESULTS = ROOT / 'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
LEGS = ['FL', 'FR', 'RL', 'RR']
MASS = 15.206408
LIMITS = np.tile([23.7, 23.7, 45.43], 4)
FIELDS = ['reference_times', 'reference_states', 'reference_actions', 'K', 'H']
FIXTURES = {'a': [237, 328], 'b': [236, 625]}
WINDOWS = [(2.002, 4.3), (4.3, 5.2), (5.2, 7.0), (7.0, 30.0)]


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def arr(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def enum_bits(value):
    return int(np.asarray([value], dtype='<f8').view('<i8')[0])


def gait(t, context, parameters):
    # Exact locked native GetPhase / FootStep formulas, reordered FL FR RL RR.
    t = np.atleast_1d(np.asarray(t, dtype=float))
    phase = context[8] + (t-context[9])*context[10]
    amplitude, duty = parameters[3], parameters[4]
    offsets = np.array([0., .5, .5, 0.])*2*np.pi
    angle = (np.fmod(phase[:, None]+np.pi-offsets, 2*np.pi)-np.pi)*.5/(1-duty)
    swing_duration = (1-duty)*2*np.pi/context[10]
    swing_t = (angle+np.pi/2)/np.pi*swing_duration
    rise_duration = swing_duration-.20
    rising = swing_t < rise_duration
    u = np.where(rising, swing_t/rise_duration, (swing_t-rise_duration)/.20)
    smooth = u*u*u*(10+u*(-15+6*u))
    step = np.where(rising, amplitude*smooth, amplitude*(1-smooth))
    velocity = np.where(rising, amplitude/rise_duration, -amplitude/.20)*30*u*u*(1-u)*(1-u)
    active = (angle > -np.pi/2) & (angle < np.pi/2)
    return phase, np.where(active, step, 0.), np.where(active, velocity, 0.)


def longest(mask, dt):
    runs = np.diff(np.r_[False, mask, False].astype(int))
    begin, end = np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)
    return float(np.max(end-begin)*dt) if len(begin) else 0.


def head_target(packet, tau):
    p = np.asarray(packet)
    b = min(tau, abs(p[6]-p[5])/.5)
    integral = p[5]*b + .5*np.sign(p[6]-p[5])*.5*b*b + p[6]*(tau-b)
    return [float(p[2]+p[4]*min(abs(p[3]-p[2]), p[5]*.25+integral)+p[8]), float(p[10]+p[9]), float(p[11])]


def analyze(stream, seed):
    attempt = f'job-20261113T145020Z-w49-{stream}-eval-PANN_PPO_S5-s{seed}-E30-attempt01'
    src = ROOT/REL/attempt
    calls = rows(src/'planner-calls.jsonl')
    boundaries = rows(src/'w49-planner-boundaries.jsonl')
    held = rows(src/'w49-held-clock.jsonl')
    manifest = json.loads((src/'w49-event-manifest.json').read_text())
    result = json.loads((RESULTS/(attempt+'.json')).read_text())
    trajectory = arr(src/'trajectory.npz')
    physical = arr(src/'physical-000.npz')['physical']
    cols = {name:i for i,name in enumerate(result['physical_columns'])}
    physical_t = physical[:,cols['force_evaluation_time_s']]
    physical_dt = physical[:,cols['post_integration_time_s']]-physical_t
    fz = physical[:,[cols[f'grf_{leg}_z'] for leg in LEGS]]
    nonfoot = physical[:,cols['nonfoot_grf_z']]
    request = physical[:,[cols[f'unclipped_pd_request_{i}'] for i in range(12)]]
    loaded = fz > 1.
    moving = [c for c in calls if enum_bits(c['gait_context'][7]) == 2]
    start = moving[0]
    gctx, gparams = start['gait_context'], start['task_parameters']
    phase, step, step_velocity = gait(physical_t, gctx, gparams)
    groups = {k: {} for k in FIELDS}
    policy_rows = []
    by_id = {x['call_record']['call_id']:x for x in manifest['event_calls']}
    snapshots = {}
    for row in manifest['event_calls']:
        cid, origin = row['call_record']['call_id'], row['boundary']['origin_s']
        policy = arr(src/row['objects']['policy']['file'])
        hashes = {k:digest(policy[k]) for k in FIELDS}
        for k,h in hashes.items():
            groups[k].setdefault(h, []).append([cid, origin])
        delta = policy['delta_values']
        compact = dict(call_id=cid, origin_s=origin, installed_plan_id=row['boundary']['installed_plan_id'],
                       policy_file=row['objects']['policy']['file'], reference_start_s=float(policy['reference_times'][0]),
                       reference_end_s=float(policy['reference_times'][-1]),
                       query_past_reference_s=max(0., origin-float(policy['reference_times'][-1])),
                       feedback_scale=float(policy['semantics'][5]), metric_regularization=float(policy['semantics'][1]),
                       delta_knots=len(policy['delta_times']), delta_max_abs_rad=float(abs(delta).max()) if delta.size else 0.,
                       exact_array_sha256=hashes)
        policy_rows.append(compact)
        if cid in FIXTURES[stream] or cid in [1, 120, 121, 122, 1521]:
            snapshots[cid]=(row, policy)
    group_summary={k:[dict(sha256=h,count=len(x),first_call=x[0][0],first_origin_s=x[0][1],last_call=x[-1][0],last_origin_s=x[-1][1]) for h,x in g.items()] for k,g in groups.items()}
    summary = dict(stream=stream, source_attempt=attempt, source_directory=str(src),
        new_native_solves=0, new_plant_substeps=0, new_predictor_substeps=0, new_PPO_samples=0, new_PPO_updates=0,
        observed_counts=dict(all_calls=len(calls),all_boundaries=len(boundaries),held_rows=len(held),physical_rows=len(physical),saved_event_calls=len(manifest['event_calls'])),
        clocks=dict(trajectory_time='state and same-time mj_forward force samples, 2ms grid',
                    physical_force_time=result['physical_time_note'],physical_step_min_s=float(physical_dt.min()),physical_step_max_s=float(physical_dt.max())),
        foot_order=dict(native=['FL','HL','FR','HR'],trajectory_and_physical=LEGS,native_to_trajectory=[0,2,1,3]),
        private_gait=dict(first_trot_call=start['call_id'],first_trot_origin_s=start['origin'],
            current_gait_raw=gctx[7],current_gait_decoded=enum_bits(gctx[7]),raw_encoding='double bits reinterpreted as int64 by locked native source',
            nominal_cadence_hz=gparams[2],amplitude_m=gparams[3],duty_ratio=gparams[4],
            phase_start=gctx[8],phase_start_time_s=gctx[9],phase_velocity_rad_s=gctx[10],
            unique_moving_parameters=len({tuple(c['task_parameters']) for c in moving}),
            moving_calls=len(moving),phase_at_final_call_rad=gait(calls[-1]['origin'],gctx,gparams)[0][0],
            conclusion='Trot is active and phase advances; unchanged phase origin does not mean phase is frozen.'),
        installed_policy_array_groups=group_summary,
        feedback_lifecycle_conclusion='Every retained installed policy has byte-identical reference states/actions/times, K and H to first valid call 1 at t=0; reference end .35s. Source generation remains initial Stand despite changing installed IDs, scale and delta.',
        lifecycle_scope='Exact array equality over saved event objects only; all-call old-reference use also checked through actual held query interpolation below.',
        saved_policy_lifecycle=policy_rows, diagnostic_events=manifest['events'])
    windows=[]
    tt=trajectory['time']
    for lo,hi in WINDOWS:
        pm=(physical_t>=lo-1e-9)&(physical_t<hi-1e-9)
        tm=(tt>=lo-1e-9)&(tt<hi-1e-9)
        q=trajectory['qpos'][tm];vel=trajectory['qvel'][tm]
        pstep=step[pm];pl=loaded[pm];swing=pstep>0
        target_phase,target_step,target_vel=gait(tt[tm],gctx,gparams)
        clearance=trajectory['feet'][tm,:,2]-.022
        dt=float(np.median(physical_dt[pm]))
        patterns,counts=np.unique(pl.astype(int),axis=0,return_counts=True)
        windows.append(dict(start_s=lo,end_exclusive_s=hi,physical_samples=int(pm.sum()),state_samples=int(tm.sum()),
            displacement_x_m=float(q[-1,0]-q[0,0]),min_z_m=float(q[:,2].min()),max_z_m=float(q[:,2].max()),
            min_vz_m_s=float(vel[:,2].min()),max_vz_m_s=float(vel[:,2].max()),
            mean_foot_fz_N=fz[pm].mean(0).tolist(),mean_nonfoot_fz_N=float(nonfoot[pm].mean()),
            mean_total_vertical_support_over_weight=float((fz[pm].sum(1)+nonfoot[pm]).mean()/(MASS*9.81)),
            loaded_gt1N_fraction=pl.mean(0).tolist(),nonfoot_gt1N_fraction=float((nonfoot[pm]>1).mean()),
            max_continuous_loaded_s=[longest(pl[:,i],dt) for i in range(4)],
            max_continuous_unloaded_s=[longest(~pl[:,i],dt) for i in range(4)],
            off_to_on_load_transitions=((~pl[:-1])&pl[1:]).sum(0).tolist(),
            on_to_off_load_transitions=(pl[:-1]&(~pl[1:])).sum(0).tolist(),
            predicted_swing_fraction=swing.mean(0).tolist(),
            actual_loaded_during_target_swing_fraction=[float(pl[:,i][swing[:,i]].mean()) for i in range(4)],
            actual_loaded_during_target_stance_fraction=[float(pl[:,i][~swing[:,i]].mean()) for i in range(4)],
            actual_clearance_min_m=clearance.min(0).tolist(),actual_clearance_max_m=clearance.max(0).tolist(),
            target_height_rmse_m=np.sqrt(((clearance-target_step)**2).mean(0)).tolist(),
            stance_foot_speed_rms_m_s=[float(np.sqrt((trajectory['feet_velocity'][tm,:,0:3][target_step[:,i]==0,i]**2).sum(1).mean())) for i in range(4)],
            actual_held_pd_saturation_any_fraction=float(np.any(abs(request[pm])>LIMITS+1e-10,axis=1).mean()),
            load_patterns=[dict(loaded_FL_FR_RL_RR=p.tolist(),fraction=float(n/pm.sum())) for p,n in zip(patterns,counts)]))
    summary['support_windows']=windows
    samples=[]
    for wanted in [2.002,2.1,2.25,2.5,3.,4.,4.3,4.32,4.4,4.5,4.62,4.7,5.,5.2,6.14,7.,8.,12.08,20.,30.]:
        k=int(np.argmin(abs(tt-wanted)));now=float(tt[k]);ph,sh,sv=gait(now,gctx,gparams)
        samples.append(dict(time_s=now,requested_time_s=wanted,qpos_xyz=trajectory['qpos'][k,:3].tolist(),
            base_vxyz=trajectory['qvel'][k,:3].tolist(),rpy=trajectory['rpy'][k].tolist(),
            true_fz_N=trajectory['actual_grf'][k,:,2].tolist(),nonfoot_fz_N=float(trajectory['nonfoot_ground_force'][k,2]),
            feet_z=trajectory['feet'][k,:,2].tolist(),feet_vxyz=trajectory['feet_velocity'][k].tolist(),
            phase_rad=float(ph[0]),target_foot_clearance_m=sh[0].tolist(),target_foot_vz_m_s=sv[0].tolist(),
            command_vx=float(trajectory['command_velocity'][k,0]),
            saturated_joints=np.flatnonzero(abs(trajectory['unclipped_pd_request'][k])>LIMITS+1e-10).tolist()))
    summary['state_support_samples']=samples
    matched=[x for x in boundaries if x['previous_prediction']['status']=='available']
    summary['recorded_same_policy_prediction_errors']=dict(pairs=len(matched),
        note='Saved previous actually applied policy, matched next actual boundary time only; not a future replanned-trajectory comparison.',
        max_abs={k:float(max(np.max(abs(np.asarray(x['previous_prediction'][k]))) for x in matched)) for k in
          ['base_position_m','base_orientation_rad','joint_position_rad','base_linear_velocity_m_s','base_angular_velocity_rad_s','joint_velocity_rad_s']})
    fixtures=[]
    for cid in FIXTURES[stream]:
        row,policy=snapshots[cid];context=arr(src/row['objects']['context']['file']);accepted=arr(src/row['objects']['accepted']['file']);p0=arr(src/row['objects']['p0']['file'])
        b=row['boundary'];c=row['call_record'];origin=b['origin_s'];t=accepted['accepted_times'];xs=accepted['accepted_states'];res=accepted['accepted_residual']
        nominal=[d for d in c['r34_decisions'] if d['kind']==0 and d['replaced']]
        phase,heights,vzs=gait(t,context['private_gait'],context['task_parameters'])
        # Native Gait residual is actual foot z minus (.022 + target), always active in moving r37.
        predicted_z=res[:,7:11][:,[0,2,1,3]]+.022+heights
        next_boundary=next((z for z in boundaries if z['call_id']>cid and z['previous_prediction'].get('source_call')==cid),None)
        fixtures.append(dict(label=stream.upper()+str(cid),call_id=cid,origin_s=origin,source_directory=str(src),
            objects={k:dict(path=str(src/v['file']),**v) for k,v in row['objects'].items()},
            P0=b['P0'],installed_plan_id=b['installed_plan_id'],problem_epoch=b['problem_epoch'],
            actual_z_m=float(b['actual_state'][2]),private_gait=context['private_gait'].tolist(),
            task_parameters=context['task_parameters'].tolist(),userdata84_96=context['native_userdata'][84:96].tolist(),
            r26_head_target_t0_m=head_target(context['native_userdata'][84:96],0),
            r26_head_target_terminal_m=head_target(context['native_userdata'][84:96],.35),
            mission_body_target=b['mission_actor']['body_target'],
            reference_start_end_s=policy['reference_times'][[0,-1]].tolist(),
            reference_expired_s=origin-float(policy['reference_times'][-1]),
            P0_feedback_scale=float(p0['semantics'][5]),installed_feedback_scale=float(policy['semantics'][5]),
            historical_B_decision=nominal[-1] if nominal else None,
            historical_B_full_rollout='not separately retained; do not equate P0 or installed accepted to B; W50 nominal screen generates current B',
            current_solve=b['current_solve'],
            accepted_own_prediction=dict(start_end_s=t[[0,-1]].tolist(),min_z_m=float(xs[:,2].min()),
                terminal_z_m=float(xs[-1,2]),terminal_dx_m=float(xs[-1,0]-xs[0,0]),
                predicted_foot_z_min_m=predicted_z.min(0).tolist(),predicted_foot_z_max_m=predicted_z.max(0).tolist(),
                prediction_fz=None,prediction_fz_reason='No accepted force array retained; gait residual recovers feet height only.',
                no_new_integration=True),
            next_same_policy_error=next_boundary['previous_prediction'] if next_boundary else None,
            query_held=next(h for h in held if abs(h['time_s']-origin)<=1e-9),
            external_restore_scope=b['mission_actor']['external_snapshot_scope'],
            historical_exact_plant_replay=False))
    summary['fixed_problems']=fixtures
    # Actual executed query clock: join the explicit applied_hold IDs, then read their native interpolation.
    applied={r['query_id'] for r in rows(src/'action-execution.jsonl') if r['query_kind']=='applied_hold'}
    actual_queries=[]
    with (src/'feedback-actions.jsonl').open() as f:
        for line in f:
            z=json.loads(line)
            if z['query_id'] in applied:
                actual_queries.append(z)
    past=[z for z in actual_queries if z['time']>.35+1e-9]
    first=next((z for z in actual_queries if z['time']>.35+1e-9),None)
    summary['actual_held_reference_query']=dict(count=len(actual_queries),after_reference_end=len(past),
        unique_interpolation_nodes_after_end=sorted({tuple(z['interpolation_nodes']) for z in past}),
        first_after_end={k:first[k] for k in ['query_id','time','plan_id','interpolation_nodes','interpolation_fraction','feedback_scaling']} if first else None,
        last={k:actual_queries[-1][k] for k in ['query_id','time','plan_id','interpolation_nodes','interpolation_fraction','feedback_scaling']},
        final_query_past_reference_end_s=float(actual_queries[-1]['time']-.35),
        metric_source_plan_id_is_installation_not_generation=True)
    summary['conclusions']=[
        'The private gait reaches Trot at 2.002s and advances at 2Hz; no evidence here of a Stand-to-Trot or phase-clock execution failure.',
        'Feedback reference, K and H remain exactly those of initial Stand in all retained event policies. Real held queries after .35s use the old end node.',
        'Before 4.3s, target swing phases coexist predominantly with actual >1N foot loading. Small forward movement and repeated later support collapse do not demonstrate stable Trot.',
        'Changing complete installation IDs does not refresh reference/K/H. This supports the scoped W50 refresh experiment, without proving it is the sole cause of collapse.',
        'r26 native Position target is the local head reference from userdata84:96, not mission x=3.',
        'No new dynamics or arbitrary historical plant restore performed.'
    ]
    return summary


def main():
    out=RESULTS/'w50';out.mkdir(exist_ok=True)
    result=dict(schema='w50_existing_w49_evidence_v1',status='completed',
        plan_commit='9ba97283c13119ec197429b75fce6b139a506a33',
        scope='All saved W49 calls/boundaries/held/physical arrays plus 231 retained event policies; source formulas only, no new dynamics.',
        deadline=None,wall_deadline_enabled=False,
        thresholds='1N is an existing diagnostic load threshold, not a new stop or qualification rule.',
        resources=dict(cpu_affinity=__import__('os').sched_getaffinity(0),threads=1),
        work=dict(new_native_solves=0,new_native_full_rollouts=0,new_plant_substeps=0,new_predictor_substeps=0,new_gpu_world_steps=0,new_PPO_samples=0,new_PPO_updates=0),
        streams=[analyze('a',1),analyze('b',2)])
    result['resources']['cpu_affinity']=sorted(result['resources']['cpu_affinity'])
    path=out/'task-a.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(path),bytes=path.stat().st_size,streams=[dict(stream=s['stream'],counts=s['observed_counts'],reference_query=s['actual_held_reference_query']) for s in result['streams']]),indent=2))


if __name__=='__main__':
    main()
