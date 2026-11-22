"""Align the four archived E600 executions, original WBC gates and physical contact clocks."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import resource
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REL = Path('terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
R12 = ROOT/'results'/REL
LEGS = ('FL', 'FR', 'RL', 'RR')
REV = dict(r55='MCP-RL-w55-f', r60='MCP-RL-w56-r60')


def quant(values):
    values = np.asarray(values, dtype=float)
    return (dict(zip(('min', 'p05', 'p50', 'p95', 'max'),
                     np.percentile(values, [0,5,50,95,100]).tolist())) if values.size else None)


def sections(mask):
    cuts = np.r_[0, 1+np.flatnonzero(mask[1:] != mask[:-1]), len(mask)]
    return [(a,b) for a,b in zip(cuts[:-1], cuts[1:]) if mask[a]]


def run(case, rev):
    wall = time.perf_counter()
    root = next((ROOT.parent/REV[rev]/'outputs'/REL).glob(f'*-{case}-{rev}-*-E600-attempt01'))
    result = next(row for row in json.loads((R12/'w56/phase.json').read_text())['cases']
                  if row.get('attempt') == root.name)
    cols = json.loads((ROOT/result['source']).read_text())['physical_columns']
    review = R12/'reviews/W56-review'
    name = f'f-{case}-{rev}-e600'
    dropouts = json.loads((review/'evidence-force-dropouts.json').read_text())[name]
    with np.load(root/'trajectory.npz') as data:
        t, phase, q, v, feet, foot_v, fz, torque, rpy = (
            data['time'], data['phase'], data['qpos'], data['qvel'], data['feet'],
            data['feet_velocity'], data['actual_grf'][:,:,2], data['actuator_force'], data['rpy'])
    raw = root/'w55-f'
    cfg = json.loads((raw/'manifest.json').read_text())
    mass = cfg['model_mass_kg']
    physics_events = [json.loads(x) for x in (raw/'contact-events.jsonl').open()]
    interior = (q[:,0]>=.8) & (q[:,0]<=2.2)
    travel = np.isin(phase, (1,3))
    groups = dict(straight_rigid=travel & ~interior, straight_strip=travel & interior,
                  turns=np.isin(phase,(2,8)), endpoint_hold=np.isin(phase,(4,5,6,7)),
                  initial_stand=phase==0)
    dt = np.diff(t, append=t[-1])
    travel_distance = np.r_[np.linalg.norm(np.diff(q[:,:2],axis=0),axis=1), 0.]
    bearing = fz > 1.
    power = torque * v[:,6:]
    limits = np.array([23.7,23.7,45.43]*4)
    summaries = {}
    # A root segment defines the terrain classification. Each foot need not be on that substrate.
    for group, mask in groups.items():
        if not mask.any():
            continue
        distance = float(travel_distance[mask].sum())
        pos_work = float((np.maximum(power[mask],0)*dt[mask,None]).sum())
        abs_work = float((np.abs(power[mask])*dt[mask,None]).sum())
        moving = group.startswith('straight') or group == 'turns'
        summaries[group] = dict(
            samples=int(mask.sum()),duration_s=float(dt[mask].sum()),distance_xy_m=distance,
            duty_Fz_over_1N=(dt[mask,None]*bearing[mask]).sum(axis=0).tolist(),
            body_height_m=quant(q[mask,2]),roll_pitch_abs_deg=quant(np.rad2deg(np.abs(rpy[mask,:2]))),
            torque_max_fraction_per_joint=(np.abs(torque[mask])/limits).max(axis=0).tolist(),
            positive_mechanical_work_J=pos_work,absolute_mechanical_work_J=abs_work,
            positive_mechanical_CoT=pos_work/(mass*9.81*distance) if moving and distance else None,
            absolute_mechanical_CoT=abs_work/(mass*9.81*distance) if moving and distance else None,
            loaded_center_xy_path_m=((np.linalg.norm(foot_v[mask,:,:2],axis=2)*bearing[mask])*dt[mask,None]).sum(0).tolist(),
            legs={})
        summaries[group]['duty_Fz_over_1N'] = (np.asarray(summaries[group]['duty_Fz_over_1N']) / dt[mask].sum()).tolist()
        for li, leg in enumerate(LEGS):
            periods,td_v,td_peaks,impulses,lo,td=[],[],[],[],[],[]
            events = [e for e in physics_events if leg in e['legs'] and e['kind']=='liftoff_committed']
            for event in events:
                it = int(np.searchsorted(t,event['time_s']-1.e-8))
                if it >= len(t) or not mask[it]:
                    continue
                prev = events[events.index(event)-1] if events[0] is not event else None
                if prev and event['time_s'] - prev['time_s'] < 1.5:
                    i0 = np.searchsorted(t, prev['time_s']-1.e-8)
                    if np.all(mask[i0:it+1]):
                        periods.append(event['time_s']-prev['time_s'])
                j = it
                while j < len(t) and bearing[j,li] and t[j] < event['fixed_apex_s']:
                    j += 1
                if j >= len(t) or bearing[j,li]:
                    continue
                lo.append(float(t[j]-event['time_s']))
                k = np.searchsorted(t, event['fixed_apex_s'])
                while k < len(t) and not bearing[k,li] and t[k] < event['fixed_touchdown_s']+.3:
                    k += 1
                if k>=len(t) or not bearing[k,li] or not mask[k]:
                    continue
                td.append(float(t[k]-event['fixed_touchdown_s']))
                td_v.append(float(foot_v[k,li,2]))
                stop = np.searchsorted(t,t[k]+.020-1.e-9)
                td_peaks.append(float(fz[k:stop,li].max()))
                impulses.append(float((fz[k:stop,li]*dt[k:stop]).sum()))
            summaries[group]['legs'][leg] = dict(
                periods=len(periods),period_s=quant(periods),period_CV=float(np.std(periods)/np.mean(periods)) if periods else None,
                force_off_minus_committed_liftoff_s=quant(lo),first_force_on_minus_committed_touchdown_s=quant(td),
                touchdown_observed_vz_m_s=quant(td_v),first20ms_normal_peak_N=quant(td_peaks),
                first20ms_normal_impulse_Ns=quant(impulses))
    # Contact-point slip of a loaded rolling sphere, distinct from center displacement.
    tkeys = ('time','rolling_contact_velocity','measured_grf')
    rolling = {g:np.zeros(4) for g in groups}
    for file in sorted(raw.glob('control-*.npz')):
        with np.load(file) as z:
            tx, rv, force = (z[k] for k in tkeys)
        ids=np.clip(np.searchsorted(t,tx-1.e-8),0,len(t)-1)
        val=np.linalg.norm(rv[:,:,:2],axis=2)*(force[:,:,2]>1)*dt[ids,None]
        for g,mask in groups.items():
            rolling[g]+=val[mask[ids]].sum(0)
    for g in summaries:
        summaries[g]['loaded_rolling_point_xy_slip_distance_m']=rolling[g].tolist()

    @lru_cache(maxsize=2)
    def phys(file):
        with np.load(root/file) as z:
            return z['physical']

    @lru_cache(maxsize=4)
    def control(chunk):
        keys=('time','event_mode','mode','confirmed_contact','load_bearing_contact','planned_contact',
              'measured_grf','mpc_force','redistributed_force_target','force','feet_position_reference',
              'feet_acceleration_reference','foot_center_acceleration_achieved','transfer_fraction','mapped_u',
              'tau_total','requested_pd','contact_confirmation_count','transfer_support_pair')
        with np.load(raw/f'control-{chunk:05d}.npz') as z:
            return {k:z[k] for k in keys}

    normal_path=raw/'normal-execution.jsonl'
    keys={(LEGS.index(d['leg']),int(round(x/.002))) for d in dropouts
          for x in np.arange(d['start_s']-.012,d['start_s']+.005,.002)}
    normals={}
    if normal_path.exists():
        for line in normal_path.open():
            rec=json.loads(line)
            key=(rec['leg'],int(round(rec['consume_s']/.002)))
            if key in keys:
                normals[key]=rec
    incidents=[]
    for d in dropouts:
        g=LEGS.index(d['leg']);p=phys(d['file']);t0=d['start_s'];ts=p[:,0]
        j=int(np.searchsorted(ts,t0-1.e-9))
        if j < 1:
            continue
        fy=p[:,cols.index(f'grf_{LEGS[g]}_z')]
        a=j-1
        while a>0 and fy[a-1]>1:
            a-=1
        first=float(ts[a]);lo=int(np.searchsorted(ts,t0-.024));early=p[lo:j+1]
        ic=np.searchsorted(ts,t0-.12)
        vz=p[:,cols.index(f'foot_velocity_{LEGS[g]}_z_m_s')]
        relevant = [e for e in physics_events if LEGS[g] in e['legs'] and t0-.36<=e['time_s']<=d['end_s']+.015]
        liftoffs=[e for e in relevant if e['kind']=='liftoff_committed' and e['time_s']<=first]
        liftoff=liftoffs[-1] if liftoffs else None
        frames=[]
        for k in range(max(0, int(np.floor((max(t0-.018,first-.004))/.002+1.e-7))),
                       int(np.ceil((t0+.010)/.002))+1):
            c=control(k//500+1);i=k%500
            if i>=len(c['time']):continue
            capture=int(np.clip(np.searchsorted(ts,c['time'][i]-1e-8),0,len(ts)-1))
            joint=slice(g*3,3*g+3)
            frames.append(dict(t_s=float(c['time'][i]),event_mode=str(c['event_mode'][i,g]),wbc_mode=int(c['mode'][i,g]),
                confirmed=bool(c['confirmed_contact'][i,g]),bearing=bool(c['load_bearing_contact'][i,g]),
                planned_stance=bool(c['planned_contact'][i,g]),confirmation_ticks=int(c['contact_confirmation_count'][i,g]),
                measured_Fz_N=float(c['measured_grf'][i,g,2]),mpc_Fz_N=float(c['mpc_force'][i,g,2]),
                redistributed_Fz_N=float(c['redistributed_force_target'][i,g,2]),wbc_Fz_N=float(c['force'][i,g,2]),
                transfer_fraction=float(c['transfer_fraction'][i]),transfer_support_pair=c['transfer_support_pair'][i].tolist(),
                sole_ref_z_m=float(c['feet_position_reference'][i,g,2]-.022),sole_z_m=float(p[capture,cols.index(f'foot_position_{LEGS[g]}_z_m')]-.022),
                measured_vz_m_s=float(vz[capture]),target_az_m_s2=float(c['feet_acceleration_reference'][i,g,2]),
                wbc_az_m_s2=float(c['foot_center_acceleration_achieved'][i,g,2]),
                normal_inverse=normals.get((g,k)),mapped_u_rad=c['mapped_u'][i,joint].tolist(),
                wbc_torque_Nm=c['tau_total'][i,joint].tolist(),
                actual_pd_Nm=[float(p[capture,cols.index(f'actuator_force_{idx}')]) for idx in range(g*3,3*g+3)]))
        contact_rows=[x for x in frames if first-.00201<=x['t_s']<=t0+1e-9]
        modes=[x['event_mode'] for x in contact_rows]
        label=('early_contact_still_swing' if liftoff and first < liftoff['fixed_touchdown_s'] and 'swing' in modes
               else 'confirmed_preceding_loss' if any(x['confirmed'] for x in contact_rows) else 'unconfirmed_search_or_other')
        incidents.append(dict(**d,case=root.name,first_preceding_force_on_s=first,
            preceding_positive_force_s=t0-first,physical_impact_peak_N=float(np.max(fy[a:j])),
            impact_impulse_Ns=float(np.sum(fy[a:j]*np.diff(ts[a:j+1]))),vertical_velocity_at_first_contact_m_s=float(vz[a]),
            vertical_velocity_at_loss_m_s=float(vz[j]),committed_touchdown_s=liftoff['fixed_touchdown_s'] if liftoff else None,
            onset_minus_planned_s=first-liftoff['fixed_touchdown_s'] if liftoff else None,
            preceding_force_window_within_record=bool(a>0),classification=label,events=relevant,control=frames))
    dest=R12/'w57/p';dest.mkdir(exist_ok=True,parents=True)
    summary=dict(attempt=root.name,version=rev,case=case,physical_original_read=True,source_result=result['source'],
        semantics=dict(terrain_group='Trunk x .8 to 2.2 defines straight_strip. A leg can cross the edge separately.',
          touchdown='First 500Hz Fz>1N after committed apex, offset from committed touchdown. Force impulse uses next 20ms.',
          slip='Integrated horizontal speed of the loaded instantaneous rolling point (WBC) and sole center separately, Fz>1 N.',
          work='500Hz measured actuator torque times observed joint velocity, per-joint positive/absolute work. CoT divides by mg and accumulated planar travel distance.',
          dropout='All archived W56-review >40ms small force-off events in its specified 5kHz windows. Not all kinds of loss in 600s.'),
        mass_kg=mass,groups=summaries,dropouts=len(incidents),classes=dict(Counter(x['classification'] for x in incidents)),
        core_s=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        wall_s=time.perf_counter()-wall)
    with (dest/f'gait-{name}.json').open('w') as f:json.dump(summary,f,indent=2,allow_nan=False)
    import gzip
    with gzip.open(dest/f'gait-{name}-incidents.jsonl.gz','wt') as f:
        for row in incidents:f.write(json.dumps(row,allow_nan=False)+'\n')
    print(name,summary['classes'],summary['wall_s'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=['a','b']);p.add_argument('revision',choices=['r55','r60'])
    args=p.parse_args();run(args.case,args.revision)
