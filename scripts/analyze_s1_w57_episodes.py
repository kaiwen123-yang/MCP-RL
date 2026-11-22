"""Paired W57 mechanical and material outcomes by task, actor case and condition."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from publish_s1_go2_w54 import write
from summarize_s1_w57 import ROOT, R, RAW, read


def quant(a):
    a=np.asarray(a)
    return np.quantile(a,[0,.5,.95,1]).tolist() if a.size else None


def episode(row):
    attempt=row['attempt']
    out=RAW/attempt
    result=read(R/'episodes'/(attempt+'.json'))
    manifest=read(out/'w55-f/manifest.json')
    with np.load(out/'trajectory.npz') as d:
        t=d['time'];q=d['qpos'];v=d['qvel'];force=d['actual_grf'];vel=d['feet_velocity'];phase=d['phase']
        torque=d['actuator_force'];angles=d['rpy']
    dt=np.diff(t,append=t[-1])
    distance=np.r_[np.linalg.norm(np.diff(q[:,:2],axis=0),axis=1),0.]
    mass_kg=manifest['model_mass_kg']
    fz=force[:,:,2];loaded=fz>1.;moving=np.isin(phase,[1,2,3,8])
    in_strip=(q[:,0]>=.8)&(q[:,0]<=2.2)&(np.abs(q[:,1])<=.6)
    groups={}
    for name,mask in [('all',np.ones(len(t),bool)),('motion',moving),('material_travel',moving & in_strip),
                      ('turn',np.isin(phase,[2,8])),('standing',~moving)]:
        if not mask.any():continue
        interval=float(dt[mask].sum());length=float(distance[mask].sum())
        power=torque[mask]*v[mask,6:]
        pos_work=float((dt[mask,None]*np.maximum(power,0.)).sum())
        abs_work=float((dt[mask,None]*np.abs(power)).sum())
        loaded_path=((np.linalg.norm(vel[mask,:,:2],axis=2)*loaded[mask])*dt[mask,None]).sum(axis=0)
        groups[name]=dict(duration_s=interval,base_xy_path_m=length,positive_joint_work_J=pos_work,
             absolute_joint_work_J=abs_work,positive_mechanical_CoT=(pos_work/(mass_kg*9.81*length)
             if name in ('motion','material_travel') and length>0 else None),
             loaded_foot_center_tangential_path_by_leg_m=loaded_path.tolist(),
             duty_Fz_gt_1N_by_leg=(dt[mask,None]*loaded[mask]).sum(0).tolist() if interval==0 else
                                 ((dt[mask,None]*loaded[mask]).sum(0)/interval).tolist(),
             diagonal_only_s=float((dt[mask]*(
                 ((loaded[mask]==[True,False,False,True]).all(axis=1)) |
                 ((loaded[mask]==[False,True,True,False]).all(axis=1)))).sum()),
             height_min_m=float(q[mask,2].min()),max_abs_roll_pitch_deg=float(np.rad2deg(np.abs(angles[mask,:2])).max()))
    columns=result['physical_columns']
    iz=[columns.index(f'grf_{leg}_z') for leg in ('FL','FR','RL','RR')]
    nonfoot=[columns.index(f'nonfoot_grf_{axis}') for axis in 'xyz']
    peak=np.zeros(4);nonfoot_peak=0.
    for f in sorted(out.glob('physical-*.npz')):
        with np.load(f) as z:phys=z['physical']
        if len(phys):
            peak=np.maximum(peak,phys[:,iz].max(axis=0))
            nonfoot_peak=max(nonfoot_peak,float(np.linalg.norm(phys[:,nonfoot],axis=1).max()))
    material=result['material']
    plpath=out/'w55-f/material-candidates.jsonl'
    records=[]
    if plpath.exists():
        records=[json.loads(l) for l in plpath.open()]
    winners=Counter(x.get('selected','unavailable') for x in records)
    applied=out/'w55-f/material-execution.jsonl'
    res=dict(attempt=attempt,case=row['case'],arm=row['arm'],task_mode=row['task_mode'],
        material=row['material'],controller=row['controller'],level=row['level'],phase_status=row['status'],
        reason=row.get('reason'),completed_original=bool(row['task_mode']=='original' and result['success']),
        observed_physical_s=float(t[-1]),first_home_stable_s=result.get('first_completion_s'),
        mass_kg=mass_kg,groups=groups,peak_Fz_by_leg_N=peak.tolist(),peak_nonfoot_force_N=nonfoot_peak,
        original_slip_integral_s=material.get('integral_slip_sq_s'),indent_integral_s=material.get('integral_indentation_sq_s'),
        max_principal_strain=material.get('max_principal_strain',material.get('max_abs_principal_strain')),
        strain_fields={k:v for k,v in material.items() if 'strain' in k and isinstance(v,(int,float))},
        planner_winners=dict(winners),candidate_rows=len(records),
        mapped_input_rows=sum(1 for _ in applied.open()) if applied.exists() else 0,
        candidate_wall_s_min_median_p95_max=quant([x['candidate_wall_s'] for x in records if 'candidate_wall_s' in x]),
        timing=result.get('w57_timing'),mechanism=result.get('mechanism'),
        measurement='500 Hz positive and absolute joint mechanical work; CoT uses moving path length. 5 kHz true contact-force peaks. Slip integral is the original dimensionless squared-speed integrand integrated over seconds.')
    if result.get('mechanism'):
        with (out/'w57-revisit-pairs.jsonl').open() as f:
            pairs=[json.loads(line) for line in f]
        matches=[p for p in pairs if p['condition_met'] and p['exact_overlap_m2']>0.]
        release=np.array([np.linalg.norm(np.asarray(p['release_q'])*[1.,np.sqrt(2)]) for p in matches])
        preload=np.array([np.linalg.norm(np.asarray(p['preload_q'])*[1.,np.sqrt(2)]) for p in matches])
        substantial=release>1.e-8
        res['revisit_measurements']=dict(
            matched_unique_plant_load_events=len({p['target_id'] for p in matches}),
            world_cell_count=len({tuple(p['key']) for p in matches}),
            footprint_overlap_mm2=quant([p['exact_overlap_m2']*1.e6 for p in matches]),
            source_load_to_target_reload_s=quant([p['target_s']-p['source_s'] for p in matches]),
            actual_free_release_to_reload_s=quant([p['gap_s'] for p in matches]),
            measured_target_normal_by_cell_N=quant([p['target_N'] for p in matches]),
            preloading_q_norm=quant(preload),nonzero_release_norm_threshold=1.e-8,
            preload_to_release_norm_fraction=quant(preload[substantial]/release[substantial]),
            scope='Per matching cell load, with positive continuous footprint overlap. A traversal contributes at most one cycle; cells and events are correlated.')
    return res


def compare(allrows,observer):
    summaries=[]
    for row in allrows:
        if row['task_mode']=='training' or 'ended_unix_s' not in row:continue
        f=R/'mechanics'/(row['attempt']+'.json')
        if not f.exists() and (R/'episodes'/(row['attempt']+'.json')).exists():
            write(f,episode(row))
        if f.exists():summaries.append(read(f))
    table=[]
    for case in ('A','B'):
        for mode in ('short','recovered'):
            rows={r['arm']:r for r in summaries if r['case']==case and r['task_mode']==mode and r['material']==observer}
            if not rows:continue
            values={}
            for arm,r in rows.items():
                m=r['mechanism']
                values[arm]=dict(verified_cycles=m['confirmed_cycle_count'],at_home_stable=int(m['home_stable']),
                    slip_integral_s=r['original_slip_integral_s'],motion_positive_CoT=r['groups'].get('motion',{}).get('positive_mechanical_CoT'),
                    peak_Fz_N=max(r['peak_Fz_by_leg_N']),max_principal_strain=r['max_principal_strain'])
            names=('B','P','R','PR')
            interaction={}
            if all(a in rows for a in names):
                for metric in values['B']:
                    if any(values[a][metric] is None for a in names):continue
                    # Cumulative work/slip and peaks over different observation lengths
                    # describe the observed prefix and cannot establish an efficiency benefit.
                    if metric not in ('verified_cycles','at_home_stable') and any(
                       rows[a]['phase_status']!='passed' for a in names):continue
                    interaction[metric]=(values['PR'][metric]-values['P'][metric])-(values['R'][metric]-values['B'][metric])
            table.append(dict(case=case,condition=mode,attempts={a:r['attempt'] for a,r in rows.items()},
                              observed=values,interaction_PR_minus_P_minus_R_plus_B=interaction))
    write(R/'paired-mechanisms.json',dict(observer=observer,cases=table,
         scope='Separate original qualifications; two nominal W44 cases, one timing learning seed. Descriptive matched contrasts only. Blind is a separately trained P0 head with the same action permission, reward and sample budget.'))
    return table


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--material',required=True,choices=('M0','M1'))
    args=parser.parse_args()
    result=compare([read(p) for p in sorted((R/'attempts').glob('*.json'))],args.material)
    print(json.dumps(result),flush=True)
