"""Read-only checks on published, saved W55 arrays; no model stepping or solves."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import argparse
_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--originals', type=Path, required=True)
_parser.add_argument('--output', type=Path, required=True)
_args = _parser.parse_args()
BASE = _args.originals
OUT = _args.output


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def maxabs(x):
    return float(np.max(np.abs(x))) if np.asarray(x).size else 0.


def exceed(x, bounds):
    return float(max(0., np.max(bounds[:, 0] - x), np.max(x - bounds[:, 1])))


def read_all(directory, pattern):
    files = sorted(directory.glob(pattern))
    rows = []
    for path in files:
        with np.load(path) as z:
            rows.append({key: z[key] for key in z.files})
    keys = set.intersection(*(set(row) for row in rows))
    return files, {key: np.concatenate([row[key] for row in rows]) for key in keys}


result = dict(scope='Only downloaded selected control/drift/MPC files; full physical arrays are independently audited by the evidence review.',
              new_physics_steps=0, new_native_rollouts=0, new_QP_solves=0, cases={})
for label in ('f-a-r53-e60', 'f-b-r53-e60', 'f-a-r55-e60', 'f-b-r55-e60'):
    root = next((BASE / label).glob('job-*'))
    root = root if (root / 'w55-f').exists() else next(p for p in (BASE / label).glob('job-*') if (p / 'w55-f').exists())
    folder = root / 'w55-f'
    model = ET.parse(root / 'model/G1-project-go2.xml').getroot()
    actuators = list(model.find('actuator'))
    ctrl_bounds = np.array([[float(x) for x in a.get('ctrlrange').split()] for a in actuators])
    torque_bounds = np.array([[float(x) for x in a.get('forcerange').split()] for a in actuators])
    files, z = read_all(folder, 'control-*.npz')
    item = dict(control_files=len(files), samples=len(z['time']), time_min_s=float(z['time'].min()),
                time_max_s=float(z['time'].max()), all_solver_statuses=np.unique(z['solver_status']).tolist(),
                mapped_input_excess_rad=exceed(z['mapped_u'], ctrl_bounds),
                torque_start_excess_Nm=exceed(z['tau_start'], torque_bounds),
                torque_mean_excess_Nm=exceed(z['tau_mean'], torque_bounds),
                predicted_hold_torque_excess_Nm=exceed(z['predicted_hold_torque'], torque_bounds),
                pd_mapping_max_error_Nm=maxabs(60*(z['mapped_u']-z['q_joint'])-5*z['v_joint']-z['tau_start']),
                floating_dynamics_max_abs_residual=maxabs(z['dynamics_residual']),
                max_solver_primal_residual=float(z['solver_primal_residual'].max()),
                force_unilateral_violation_N=float(max(0., -z['force'][:, :, 2].min())),
                force_inner_pyramid_violation_N=float(max(0., (np.abs(z['force'][:, :, :2])-(.5/np.sqrt(2))*z['force'][:, :, 2, None]).max())),
                force_circular_cone_violation_N=float(max(0., (np.linalg.norm(z['force'][:, :, :2], axis=2)-.5*z['force'][:, :, 2]).max())),
                inactive_contact_force_max_abs_N=maxabs(z['force'][~z['active_wbc_contact'].astype(bool)]))
    common=[]
    for path in files:
        drift=folder/path.name.replace('control-', 'drift-')
        if not drift.exists():
            continue
        with np.load(path) as c, np.load(drift) as d:
            count=len(c['time'])
            if not np.array_equal(c['time'], d['time'][:count]):
                raise ValueError('Saved control/drift timestamps differ')
            # The failing final r53 tick computed drift before the WBC failed;
            # it has no successfully returned control row and is not installed.
            dv={k:d[k][:count] for k in d.files}
            selected=np.where(c['foot_task_is_contact'][:, :, None],dv['rolling_corrected'],dv['foot_corrected'])
            common.append(dict(samples=len(c['time']),
                               runtime_foot_drift_error=maxabs(c['feet_Jdotv']-selected),
                               point_cross_error=maxabs(dv['foot_point_term']-np.cross(c['foot_angular_velocity'],c['foot_velocity'])),
                               foot_correction_error=maxabs(dv['foot_corrected']-dv['foot_raw']-dv['foot_point_term']),
                               rolling_correction_error=maxabs(dv['rolling_corrected']-dv['rolling_raw']-dv['foot_point_term']),
                               body_COM_correction_error=maxabs(dv['body_COM_corrected']-dv['body_COM_raw']-dv['body_COM_point_term'])))
    item['drift_runtime_recalculation']={k:sum(r[k] for r in common) if k=='samples' else max(r[k] for r in common) for k in common[0]}
    mpcs=[]
    for path in sorted(folder.glob('mpc-*.npz')):
        with np.load(path) as m:
            if 'moment_linearization_com' not in m:
                continue
            dt=float(m['times'][1]-m['times'][0]);x0=m['x0'];n=len(m['contact'])
            centers=x0[3:6]+(m['times'][:-1]-m['times'][0])[:,None]*x0[9:12]
            _,pitch,yaw=x0[:3];c,s=np.cos(yaw),np.sin(yaw)
            E=np.array([[c/np.cos(pitch),s/np.cos(pitch),0.],[-s,c,0.],[c*np.tan(pitch),s*np.tan(pitch),1.]])
            aerr=berr=cerr=0.
            balancing=np.zeros((n,3));balancing[:,2]=m['contact'].any(axis=1)*float(m['mass'])*9.81
            for k,center in enumerate(centers):
                I=m['inertia'];omega=x0[6:9]
                A=np.eye(12);A[:3,6:9]=dt*E;A[3:6,9:12]=dt*np.eye(3)
                G=np.concatenate([np.linalg.solve(I,skew(p-center)) for p in m['feet'][k]],axis=1)
                L=np.tile(np.eye(3)/float(m['mass']),(1,4))
                B=np.zeros((12,12));B[:3]=.5*dt*dt*E@G;B[3:6]=.5*dt*dt*L;B[6:9]=dt*G;B[9:12]=dt*L
                alpha=-np.linalg.solve(I,np.cross(omega,I@omega))
                aff=np.r_[.5*dt*dt*E@alpha,[0.,0.,-.5*9.81*dt*dt],dt*alpha,[0.,0.,-9.81*dt]]
                derivative=np.linalg.solve(I,skew(balancing[k]));at=.5*dt*dt*(E@derivative);wt=dt*derivative
                A[:3,3:6]+=at;A[6:9,3:6]+=wt;aff[:3]-=at@center;aff[6:9]-=wt@center
                aerr=max(aerr,maxabs(A-m['stage_A'][k]));berr=max(berr,maxabs(B-m['stage_B'][k]));cerr=max(cerr,maxabs(aff-m['stage_c'][k]))
            dynamics=m['states'][1:]-np.einsum('nij,nj->ni',m['stage_A'],m['states'][:-1])-np.einsum('nij,nj->ni',m['stage_B'],m['forces'].reshape(n,12))-m['stage_c']
            mpcs.append(dict(file=path.name,physical_time_s=float(m['times'][0]),phase=str(m['phase']),
                             center_formula_max_error_m=maxabs(centers-m['moment_linearization_com']),
                             balancing_formula_max_error_N=maxabs(balancing-m['moment_linearization_net_force']),
                             stage_A_max_error=aerr,stage_B_max_error=berr,stage_c_max_error=cerr,
                             saved_dynamics_equality_max_error=maxabs(dynamics),
                             x0_max_error=maxabs(m['states'][0]-m['x0'])))
    item['r55_hold_mpc_recalculation']=mpcs
    result['cases'][label]=item
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
