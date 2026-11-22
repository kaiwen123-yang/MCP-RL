"""Original-task B/P pairs on the same observer, actor case, physical step and horizon."""
import argparse
import json
from pathlib import Path
import numpy as np

from analyze_s1_w57_planner_outcomes import ROOT,REL


def run(baseline,planned):
    r=ROOT/'results'/REL/'w57';large=ROOT/'outputs'/REL
    labels=('B','P');snaps={};info={};trajectory={}
    for label,attempt in zip(labels,(baseline,planned)):
        receipt=json.loads((r/'attempts'/(attempt+'.json')).read_text())
        info[label]=receipt
        result=json.loads((r/'episodes'/(attempt+'.json')).read_text())
        with np.load(large/attempt/'trajectory.npz') as d:
            trajectory[label]={k:d[k] for k in ('time','q_target','qpos','actual_grf','actuator_force','qvel','phase')}
        z=trajectory[label];t=z['time'];dt=np.diff(t,append=t[-1]);v=z['qvel'];torque=z['actuator_force']
        distance=np.r_[np.linalg.norm(np.diff(z['qpos'][:,:2],axis=0),axis=1),0.]
        m=np.isin(z['phase'],(1,2,3,8));weight=json.loads((large/attempt/'w55-f/manifest.json').read_text())['model_mass_kg']*9.81
        power=torque*v[:,6:];pos=float((dt[:,None]*np.maximum(power,0)).sum());ab=float((dt[:,None]*np.abs(power)).sum())
        passages=result['legal_route']['passages'];strips=[x for x in passages if x['completed']]
        snaps[label]=dict(attempt=attempt,level=receipt['level'],success=result['success'],physical_end_s=float(t[-1]),
          progress=result['legal_route']['progress'],legal_outbound=result['legal_route']['legal_outbound_traversals'],
          legal_return=result['legal_route']['legal_return_traversals'],
          completed_roundtrips=result['legal_route']['completed_roundtrips'],
          first_outbound_entry_s=passages[0]['entry_s'] if passages else None,
          first_outbound_exit_s=passages[0]['exit_s'] if passages else None,passages=passages,
          slip_integral_s=result['material']['integral_slip_sq_s'],
          indentation_integral_s=result['material']['integral_indentation_sq_s'],
          peak_principal_strain=result['material']['max_abs_principal_strain'],positive_joint_work_J=pos,
          absolute_joint_work_J=ab,root_xy_path_m=float(distance.sum()),root_xy_path_in_motion_m=float(distance[m].sum()),
          motion_positive_mechanical_CoT=float((dt[m,None]*np.maximum(power[m],0)).sum()/(weight*distance[m].sum())),
          motion_absolute_mechanical_CoT=float((dt[m,None]*np.abs(power[m])).sum()/(weight*distance[m].sum())))
    if any(info['B'][k]!=info['P'][k] for k in ('case','level','material','checkpoint_sha256','normalization_sha256','seed')):
        raise ValueError('Mismatched original-task pair')
    tt=trajectory['P']['time'];tb=trajectory['B']['time'];n=min(len(tt),len(tb))
    maxima={}
    for k in ('q_target','qpos','actual_grf','actuator_force'):
        a=trajectory['B'][k][:n];b=trajectory['P'][k][:n]
        err=np.abs(a-b).reshape(n,-1).max(1);diff=np.flatnonzero(err>0)
        maxima[k]=dict(first_difference_s=float(tt[diff[0]]) if len(diff) else None,
            first_max_abs_difference=float(err[diff[0]]) if len(diff) else None,
            preceding_all_bitwise_equal=bool(len(diff) and np.array_equal(a[:diff[0]],b[:diff[0]])),
            whole_prefix_peak=float(err.max()))
    gained={}
    same_completed=all(s['success'] for s in snaps.values()) and info['P']['level']==info['B']['level']
    for k in ('progress','first_outbound_entry_s','first_outbound_exit_s','slip_integral_s','indentation_integral_s',
              'peak_principal_strain','positive_joint_work_J','absolute_joint_work_J','motion_positive_mechanical_CoT',
              'motion_absolute_mechanical_CoT','root_xy_path_m'):
        a=snaps['B'][k];b=snaps['P'][k]
        if a is not None and b is not None:
            gained[k]=dict(P_minus_B=float(b-a),percent_relative_to_B=float(100*(b-a)/a) if a else None)
    res=dict(case=info['P']['case'],estimator=info['P']['material'],level=info['P']['level'],arms=snaps,
       same_end_and_both_success=same_completed,shared_trajectory_rows=n,max_trajectory_clock_difference_s=float(np.abs(tt[:n]-tb[:n]).max()),
       first_closed_loop_difference=maxima,paired_differences=gained,
       scope='Descriptive deterministic pair on the original task. Slip is the original normalized velocity-squared integral and mechanical CoT divides positive joint work by mg times XY path in motion. The two controllers may travel different distances in the same time budget.')
    dest=r/'p/physical'/(planned+'-paired-B.json');dest.parent.mkdir(exist_ok=True,parents=True)
    dest.write_text(json.dumps(res,indent=2,allow_nan=False)+'\n');return res


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('planned')
    a=p.parse_args();answer=run(a.baseline,a.planned)
    print(json.dumps(answer['paired_differences']))
