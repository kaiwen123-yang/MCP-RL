"""Paired M0-B/M1-B short-revisit outcomes, separate from the frozen M1 matrix."""
import argparse
import hashlib
import json

import numpy as np

from analyze_s1_w57_episodes import episode
from publish_s1_go2_w54 import write
from summarize_s1_w57 import RAW, R, read


def metrics(data):
    return np.quantile(data,[0.,.5,.95,1.]).tolist() if len(data) else None


def mechanics(row):
    path=R/'mechanics'/(row['attempt']+'.json')
    if not path.exists():
        write(path,episode(row))
    return read(path)


def revisit_distribution(path):
    accepted=[]
    for line in path.open():
        v=json.loads(line)
        if v['condition_met'] and v['exact_overlap_m2']>0.:
            accepted.append(v)
    groups={}
    for cycle in sorted({r['cycle'] for r in accepted}):
        rows=[r for r in accepted if r['cycle']==cycle]
        unique={(r['source_id'],r['release_id'],r['target_id']):r for r in rows}
        group=list(unique.values())
        groups[str(cycle)]=dict(unique_event_chains=len(group),unique_world_cells=len({tuple(r['key']) for r in group}),
             recontact_gap_s_min_p50_p95_max=metrics([r['gap_s'] for r in group]),
             overlap_m2_min_p50_p95_max=metrics([r['exact_overlap_m2'] for r in group]),
             absolute_preload_q11_min_p50_p95_max=metrics([abs(r['preload_q'][0]) for r in group]),
             actual_source_target_initial_cell_force_N_min_p50_p95_max=
                  [metrics([r[k] for r in group]) for k in ('source_N','target_N')])
    return dict(all_accepted_cell_rows=len(accepted),unique_event_chains=len({
        (r['source_id'],r['release_id'],r['target_id']) for r in accepted}),cycles=groups)


def first_actual_load(folder):
    with (folder/'material-events.jsonl').open() as stream:
        for line in stream:
            event=json.loads(line)
            if event['type']=='load':
                return event['time_s']
    return None


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def trajectories(folder):
    fields=('time','qpos','qvel','rpy','actual_grf','feet_velocity','q_target','actuator_force',
            'command_velocity','requested_speed_cap','phase')
    with np.load(folder/'trajectory.npz') as f:
        return {k:f[k] for k in fields}


def common_prefix(traces, starts):
    recorded=[np.rint(d['time']/.0002).astype(np.int64) for d in traces]
    ticks=np.intersect1d(*recorded)
    ticks=ticks[ticks%10==0]
    t=ticks*.0002
    if not np.all(np.diff(ticks)==10):
        raise ValueError('Original closed-loop trajectory has a gap in common time.')
    idx=[np.searchsorted(clock,ticks) for clock in recorded]
    before=t<min(v for v in starts if v is not None) if any(v is not None for v in starts) else np.ones(len(t),bool)
    pre={};first={}
    for name,slice_,unit in [('qpos',slice(None,3),'base_xyz_m'),('rpy',slice(None),'rpy_rad'),
                             ('qpos',slice(7,None),'joint_rad'),('qvel',slice(6,None),'joint_rad_s'),
                             ('q_target',slice(None),'mapped_target_rad'),('actuator_force',slice(None),'actuator_Nm'),
                             ('actual_grf',slice(None),'foot_force_N'),('requested_speed_cap',None,'requested_speed_cap_m_s')]:
        v0,v1=(d[name][i] for d,i in zip(traces,idx))
        if slice_ is not None:
            v0,v1=v0[:,slice_],v1[:,slice_]
        difference=v0-v1
        if name=='rpy':
            difference=np.arctan2(np.sin(difference),np.cos(difference))
        diff=np.abs(difference).reshape(len(t),-1).max(axis=1)
        pre[unit]=float(diff[before].max(initial=0.))
        at=np.flatnonzero(diff>0)
        first[unit]=dict(first_nonzero_difference_time_s=float(t[at[0]]) if len(at) else None,
                         maximum_absolute_difference=float(diff.max(initial=0.)))
    data=[]
    for d,i in zip(traces,idx):
        dt=np.diff(d['time'][i],append=d['time'][i[-1]])
        force=d['actual_grf'][i]
        loaded=force[:,:,2]>1.
        vel=d['feet_velocity'][i]
        work=d['actuator_force'][i]*d['qvel'][i,6:]
        path=float(np.linalg.norm(np.diff(d['qpos'][i,:2],axis=0),axis=1).sum())
        data.append(dict(joint_positive_work_J=float(np.sum(dt[:,None]*np.maximum(work,0.))),
                    joint_absolute_work_J=float(np.sum(dt[:,None]*np.abs(work))),
                    base_xy_path_m=path,loaded_foot_center_tangential_path_m=float(
                       ((np.linalg.norm(vel[:,:,:2],axis=2)*loaded)*dt[:,None]).sum()),
                    peak_abs_roll_pitch_deg=float(np.rad2deg(np.abs(d['rpy'][i,:2])).max()),
                    minimum_body_z_m=float(d['qpos'][i,2].min())))
    return dict(clock_s=[float(t[0]),float(t[-1])],control_samples=len(t),first_actual_material_load_s=starts,
                virgin_common_control_rows=int(before.sum()),pre_material_max_absolute_difference=pre,
                full_prefix_differences=first,observed={'M0':data[0],'M1':data[1]},
                M1_minus_M0={k:data[1][k]-data[0][k] for k in data[0]})


def reference_onsets(folders):
    streams=[]
    for d in folders:
        rows={}
        for line in (d/'w55-f/normal-execution.jsonl').open():
            r=json.loads(line)
            rows[round(r['consume_s']/.002),r['leg']]=r
        streams.append(rows)
    keys=sorted(streams[0].keys() & streams[1].keys())
    first_reference=None
    controls_matched=0
    largest=0.
    for time_leg in keys:
        a,b=[v[time_leg] for v in streams]
        if max(abs(a[k]-b[k]) for k in ('measured_N','target_N'))>1e-8:
            continue
        if abs(a['owned_area_m2']-b['owned_area_m2'])>1e-14:
            continue
        controls_matched+=1
        increments=[v['measured_delta_m']-v['target_delta_m'] for v in (a,b)]
        difference=increments[1]-increments[0]
        largest=max(largest,abs(difference))
        if abs(difference)>5.21e-17 and first_reference is None:
            first_reference=dict(time_s=.002*time_leg[0],leg=time_leg[1],foot_observations_N=[a['measured_N'],b['measured_N']],
                target_N=[a['target_N'],b['target_N']],normal_increment_m=increments,difference_m=difference)
    return dict(common_foot_and_time_rows=len(keys),matched_force_and_area_rows=controls_matched,
                max_increment_difference_at_matched_force_and_area_m=largest,
                first_difference_above_paired_r60_numeric_resolution=first_reference,
                scope='Matched measured force, requested force, and owned area at a common leg/time. Entry-level coincidence does not establish equal prior trajectories after divergence.')


def one_pair(ids):
    rows=[read(R/'attempts'/(x+'.json')) for x in ids]
    if [r['material'] for r in rows]!=['M0','M1'] or any(
       r['arm']!='B' or r['task_mode']!='short' or r['settings']['planner']!='P0'
       or r['settings']['timing_mode']!='zero' or r['old_actor_updates']!=0 for r in rows):
        raise ValueError('Expected the fixed P0/R0 short pair, ordered M0 then M1.')
    if rows[0]['case']!=rows[1]['case'] or rows[0]['seed']!=rows[1]['seed']:
        raise ValueError('Legacy actor cases differ.')
    folders=[RAW/x for x in ids]
    m=[mechanics(r) for r in rows]
    fixed_fields=('schema','cycles','target_strip_x_m','target_strip_y_m','center_park_x_m',
                  'body_reversal_x_m','free_target_hold_s','short_gap_upper_s','observation_s')
    common={k:rows[0][k]==rows[1][k] for k in ('checkpoint_sha256','normalization_sha256','retained_resume_sha256')}
    common['physical_task_fields']=all(rows[0]['settings']['mechanism'][k]==rows[1]['settings']['mechanism'][k] for k in fixed_fields)
    common['saved_plant_binary_bitwise_equal']=sha(folders[0]/'model.mjb')==sha(folders[1]/'model.mjb')
    src=[r['active_sources'] for r in rows]
    same_key_changes={k:[src[0][k],src[1][k]] for k in sorted(src[0].keys() & src[1].keys()) if src[0][k]!=src[1][k]}
    full={}
    for r,mech,d in zip(rows,m,folders):
        full[r['material']]=dict(attempt=r['attempt'],status=r['status'],reason=r.get('reason'),
            physical_s=r.get('simulation_s'),source_commit=r['source_commit'],controller=r['controller'],
            original_qualification_usable=False,confirmed_cycles=mech['mechanism']['confirmed_cycle_count'],
            completed_traversals=len(mech['mechanism']['traversals']),home_stable=mech['mechanism']['home_stable'],
            accepted_cell_event_rows=mech['mechanism']['positive_overlap_event_rows'],
            gap_s_min_median_max=mech['mechanism']['gap_min_median_max_s'],
            max_principal_strain=mech['max_principal_strain'],peak_Fz_by_leg_N=mech['peak_Fz_by_leg_N'],
            total_task_slip_integral_s=mech['original_slip_integral_s'],
            total_task_indentation_integral_s=mech['indent_integral_s'],
            peak_nonfoot_force_N=mech['peak_nonfoot_force_N'],observed=mech['groups'],
            distinct_revisit_pairs=revisit_distribution(d/'w57-revisit-pairs.jsonl'))
    prefix=common_prefix([trajectories(d) for d in folders],[first_actual_load(d) for d in folders])
    return dict(case=rows[0]['case'],old_seed=rows[0]['seed'],frozen_inputs_equivalence=common,
            common_runtime_source_digests_changed=same_key_changes,sources_only_in_M1=sorted(src[1].keys()-src[0].keys()),
            matched_physical_prefix=prefix,measured_WBC_increments=reference_onsets(folders),
            whole_episodes=full,delta_confirmed_cycles=full['M1']['confirmed_cycles']-full['M0']['confirmed_cycles'],
            delta_home_stable=int(full['M1']['home_stable'])-int(full['M0']['home_stable']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempts',nargs=4,metavar='attempt')
    args=p.parse_args()
    pairs=[one_pair(args.attempts[:2]),one_pair(args.attempts[2:])]
    if [x['case'] for x in pairs]!=['A','B']:
        raise ValueError('Expected A then B.')
    write(R/'m/closed-loop-short.json',dict(physical_distinct_episodes=4,extra_M0_diagnostic_episodes=2,
        selected_M1_primary_matrix_episodes_reused_for_the_contrast=2,new_physics_in_this_analysis=0,
        source_attempts=args.attempts,scope='Two same-case independent closed-loop contrasts of whole M estimators on the declared short revisit task, each from virgin state with frozen W44 speed actor and P0/R0; no original-route qualification awarded, no new learning. Compare work over common exposure and reported task accomplishment.',
        cases=pairs))
    print(json.dumps({k:[p[k] for p in pairs] for k in ('case','delta_confirmed_cycles','delta_home_stable')}))


if __name__=='__main__':
    main()
