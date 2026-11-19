"""Describe the original F task from the logged control, material and force clocks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = 'tm-s1-go2-local-readiness-r12'
SMALL = ROOT/'results/terrain_memory/simulation'/RUN/'w55/f'
LARGE = ROOT/'outputs/terrain_memory/simulation'/RUN


def rms(values):
    return np.sqrt(np.mean(np.square(values),axis=0)).tolist() if len(values) else None


def point(t, samples, mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return None
    i = indices[0]
    return dict(time_s=float(t[i]),xyz_m=samples[i,:3].tolist())


def analyze(attempt):
    out = LARGE/attempt
    result = json.loads((SMALL/(attempt+'.json')).read_text())
    recording = result['w54_f_recording']
    resolved = json.loads((SMALL/(attempt+'-resolved.json')).read_text())
    with np.load(out/'trajectory.npz') as z:
        t, pos, angles, phases = z['time'],z['qpos'][:,:3],z['rpy'],z['phase']
        velocities = z['feet_velocity']
        feet = z['feet']
    folder = out/'w55-f'
    raw_root = result['physical_columns']
    force_columns = [raw_root.index('total_grf_'+axis) for axis in 'xyz']
    control_keys = ('time','mode','force','measured_grf','active_wbc_contact','feet_position_reference',
                    'feet_velocity_reference','foot_center_acceleration_achieved','body_acceleration_achieved',
                    'body_acceleration_reference','tau_total','requested_pd','torque_lower','torque_upper')
    parts = {key:[] for key in control_keys}
    for file in sorted(folder.glob('control-*.npz')):
        with np.load(file) as controls:
            for key in control_keys:
                parts[key].append(controls[key])
    controls = {key:np.concatenate(v) for key,v in parts.items()}
    ct = controls['time']
    if np.max(np.abs(ct-t[:len(ct)])) > 1e-9:
        raise ValueError('Trajectory and controller clocks disagree')
    material_columns = result['material']['foot_columns']
    force_ids = [material_columns.index(f'force_{k}_z') for k in range(4)]
    first_load = None
    loaded_peak = 0.
    for path in sorted(out.glob('material-[0-9]*.npz')):
        with np.load(path) as z:
            a = z['foot']
            if not len(a):
                continue
            actual = a[:,1]<=result['simulation_s']+1e-10
            fz = a[:,force_ids]
            loaded = np.flatnonzero((fz>1).any(axis=1)&actual)
            if len(loaded) and first_load is None:
                first_load = float(a[loaded[0],0])
            if actual.any():
                loaded_peak = max(loaded_peak,float(fz[actual].max()))
    contact_events=[json.loads(line) for line in (folder/'contact-events.jsonl').read_text().splitlines()]
    liftoffs=[d for d in contact_events if d['kind']=='liftoff_committed']
    touchdowns=[d for d in contact_events if d['kind']=='touchdown_search']
    material_events=[json.loads(line) for line in (out/'material-events.jsonl').read_text().splitlines()]
    # Average forces over each applied 2ms hold. A partial terminal hold is omitted.
    sums=np.zeros((len(ct),3))
    counts=np.zeros(len(ct),int)
    peak_nonfoot=0.
    for path in sorted(out.glob('physical-*.npz')):
        with np.load(path) as z:
            a=z['physical']
            hold=np.rint(a[:,0]/.0002).astype(int)//10
            np.add.at(sums,hold,a[:,force_columns])
            np.add.at(counts,hold,1)
            ids=[raw_root.index('nonfoot_grf_'+ax) for ax in 'xyz']
            peak_nonfoot=max(peak_nonfoot,float(np.linalg.norm(a[:,ids],axis=1).max()))
    full=(counts==10)
    actual_acc=sums[full]/(10*float(resolved['mass_kg']))+[0.,0.,-9.81]
    mask=ct[full]>=2.
    revisits=[e for e in material_events if e['type']=='load' and e.get('cross_leg') is True]
    duration=float(t[-1])
    windows=[]
    anchor=2.
    for lo,hi in [(0.,anchor), (anchor,first_load if first_load else duration),
                  (first_load if first_load else duration, duration)]:
        use=(t>=lo-1e-9)&(t<=hi+1e-9)
        if not use.any() or hi<=lo:
            continue
        x=pos[use]
        windows.append(dict(start_s=lo,end_s=hi,dx_m=float(x[-1,0]-x[0,0]),
                            z_range_m=[float(x[:,2].min()),float(x[:,2].max())],
                            max_abs_roll_pitch_rad=abs(angles[use,:2]).max(0).tolist()))
    swing=(controls['mode']==0) & (ct[:,None]>=2.-1e-9)
    swing_p=controls['feet_position_reference']-feet[:len(ct)]
    swing_v=controls['feet_velocity_reference']-velocities[:len(ct)]
    actual_foot_acc=np.diff(velocities,axis=0)/np.diff(t)[:,None,None]
    pairs=min(len(actual_foot_acc),len(ct))
    foot_acc_res=(controls['foot_center_acceleration_achieved'][:pairs]-actual_foot_acc[:pairs])
    last=liftoffs[-1] if liftoffs else None
    delays=[]
    for event in contact_events:
        if event['kind']!='actual_touchdown' or event['prior_mode'] not in ('search','swing'):
            continue
        matches=[e for e in liftoffs if e['time_s']<=event['time_s'] and event['legs'][0] in e['legs']]
        if matches:
            delays.append(event['time_s']-matches[-1]['fixed_touchdown_s'])
    first_by_height=point(t,pos,(t>=2.)&(pos[:,2]<.28))
    analysis=dict(attempt=attempt,build=recording['config']['identity'],result=result['stage_status'],
        reason=result['reason'],passed=result['success'],simulation_s=duration,physical_samples=result['physical_samples'],
        control_solves=len(ct),MPC_replans=recording['mpc_calls'],drift_samples=sum(c['samples'] for c in recording['drift_chunks']),
        final_xyz_m=pos[-1].tolist(),max_x_m=float(pos[:,0].max()),min_z_m=float(pos[:,2].min()),
        windows=windows,first_z_below_028_after_stand=first_by_height,
        material=dict(first_accepted_force_above_1N_s=first_load,peak_accepted_foot_normal_N=loaded_peak,
                      max_abs_strain=result['material']['max_abs_principal_strain'],
                      domain_exit=result['material'].get('domain_exit'),loaded_cells=result['material']['loaded_ever_cells'],
                      event_field_counts=dict(Counter(e.get('kind',e.get('type','')) for e in material_events)),
                      cross_leg_reloads=len(revisits),first_cross_leg_reload_s=revisits[0]['time_s'] if revisits else None,
                      distinct_cross_leg_reloaded_world_keys=len({tuple(e['world_key']) for e in revisits}),
                      legal_route=result['legal_route'],original_stage_progress=result['stage_progress']),
        gait=dict(liftoffs=len(liftoffs),first_liftoff=liftoffs[0] if liftoffs else None,last_liftoff=last,
                  actual_touchdowns=sum(e['kind']=='actual_touchdown' for e in contact_events),
                  alternating_diagonals=all(a['legs']!=b['legs'] for a,b in zip(liftoffs,liftoffs[1:])),
                  touchdown_searches=len(touchdowns),
                  search_gap_at_scheduled_end_max_m=max((e['remaining_gap_m'] for e in touchdowns),default=None),
                  touchdown_minus_planned_end_ms=(1000*np.percentile(delays,[0,50,95,100])).tolist() if delays else None,
                  last_liftoff_to_terminal_s=duration-last['time_s'] if last else None,
                  swing_center_position_error_xyz_RMS_m=rms(swing_p[swing]),swing_center_velocity_error_xyz_RMS_m_s=rms(swing_v[swing]),
                  predicted_minus_measured_swing_acceleration_xyz_RMS_m_s2=rms(foot_acc_res[swing[:pairs]]),
                  observed_minus_WBC_COM_accel_xyz_RMS_m_s2=rms((actual_acc-controls['body_acceleration_achieved'][full])[mask]),
                  peak_nonfoot_ground_force_N=peak_nonfoot,
                  max_PD_mapping_error_Nm=float(abs(controls['tau_total']-controls['requested_pd']).max())),
        wall_s=result['wall_s'],timing=recording['timing'],physical_interval=[result['physical_started_unix_s'],result['physical_ended_unix_s']],
        new_PPO_samples=0,new_PPO_updates=0,donor=0,material_memory_causal_comparison=False)
    target=SMALL/(attempt+'-analysis.json')
    target.write_text(json.dumps(analysis,indent=2)+'\n')
    return analysis


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--attempt',required=True)
    args=p.parse_args()
    row=analyze(args.attempt)
    print(json.dumps({k:row[k] for k in ('attempt','result','reason','simulation_s','max_x_m','min_z_m')}))
