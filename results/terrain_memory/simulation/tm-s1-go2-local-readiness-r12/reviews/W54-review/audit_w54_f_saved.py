"""Read-only W54 control/physical saved-array analysis. No simulator or solver."""
from pathlib import Path
import argparse
import json
import numpy as np

P=argparse.ArgumentParser(description=__doc__)
P.add_argument('--originals',type=Path,required=True)
P.add_argument('--output',type=Path,required=True)
a=P.parse_args()
BASE=a.originals
OUT=a.output
OUT.mkdir(parents=True,exist_ok=True)
LEGS=['FL','FR','RL','RR']
JOBS={52:'job-20261117T054859Z-w54-f-a-r52-eval-PANN_PPO_S5-s1-E30-attempt01',51:'job-20261117T053419Z-w54-f-a-r51-eval-PANN_PPO_S5-s1-E30-attempt01',50:'job-20261117T051942Z-w54-f-a-r50-eval-PANN_PPO_S5-s1-E30-attempt01'}
def rms(a):return float(np.sqrt(np.mean(np.asarray(a)**2)))
def normrms(a):return float(np.sqrt(np.mean(np.sum(np.asarray(a)**2,axis=-1))))
def read(v):
 r=BASE/JOBS[v];zs=[np.load(p) for p in sorted((r/'w54-f').glob('control-*.npz'))]
 c={k:np.concatenate([z[k] for z in zs]) for k in zs[0].files}
 tr=dict(np.load(r/'trajectory.npz'));p=np.concatenate([np.load(f)['physical'] for f in sorted(r.glob('physical-*.npz'))])
 assert np.max(abs(c['time']-tr['time'][:len(c['time'])]))<1e-8
 return r,c,tr,p
def audit(v):
 r,c,tr,p=read(v);n=len(c['time']);t=c['time'];nt=min(n,len(tr['time'])-1,len(p)//10)
 step=np.diff(tr['time'])[:nt]
 actual_qacc=np.diff(tr['qvel'],axis=0)[:nt]/step[:,None]
 actual_facc=np.diff(tr['feet_velocity'],axis=0)[:nt]/step[:,None,None]
 ph=p[:nt*10].reshape(nt,10,-1)
 assert np.max(abs(ph[:,0,0]-t[:nt]))<1e-8
 tau=ph[:,:,2:14].mean(1);f=ph[:,:,30:42].reshape(nt,10,4,3).mean(1)
 events=[json.loads(x) for x in (r/'w54-f/contact-events.jsonl').read_text().splitlines()]
 correction=np.cross(c['foot_angular_velocity'],c['foot_velocity'])
 out={'version':v,'control_count':n,'physics_steps':len(p),'swings':[],'windows':[],'closure':[],
      'jacdot_bug':{'official_fix':'https://github.com/google-deepmind/mujoco/commit/de48f4178f971d945fc135afb42699b8ca6bec8e',
        'fixed_release':'3.3.1; F frozen library is 3.2.3',
        'missing_term_contraction':'omega_body_world cross v_point_world',
        'boundary':'Saved-output algebra only; no corrected QP/action/physics run; derivatives of recorded velocities approximate hold-average accelerations.',
        'windows':[], 'kinematic_identity_samples':[]}}
 for ev in events:
  if ev['kind']!='liftoff_committed':continue
  a=ev['time_s'];b=ev['fixed_touchdown_s'];ii=np.flatnonzero((t[:nt]>=a-1e-8)&(t[:nt]<b-1e-8));end=int(np.argmin(abs(t-b)))
  for leg in ev['legs']:
   j=LEGS.index(leg);td=next((e['time_s'] for e in events if e['kind']=='actual_touchdown' and leg in e['legs'] and e['time_s']>a),None)
   out['swings'].append({'leg':leg,'liftoff':a,'planned_touchdown':b,'confirmed_touchdown':td,'confirmed_delay_s':None if td is None else td-b,'end_gap_m':float(tr['feet'][end,j,2]-.022),'end_vertical_velocity_m_s':float(tr['feet_velocity'][end,j,2]),'foot_tracking_rms_xyz_m':np.sqrt(np.mean((tr['feet'][ii,j]-c['feet_position_reference'][ii,j])**2,axis=0)).tolist(),'predicted_foot_task_error_rms':rms(c['feet_acceleration_residual'][ii,j]),'actual_minus_WBC_foot_acc_rms':rms(actual_facc[ii,j]-c['foot_center_acceleration_achieved'][ii,j]),'actual_minus_WBC_foot_acc_mean_xyz':np.mean(actual_facc[ii,j]-c['foot_center_acceleration_achieved'][ii,j],axis=0).tolist(),'actual_minus_mean_tau_rms':rms(tau[ii,3*j:3*j+3]-c['tau_mean'][ii,3*j:3*j+3]),'dry_model_error_rms':rms(c['friction_compensation'][ii,3*j:3*j+3]+c['actual_friction_constraint'][ii,3*j:3*j+3]),'body_start_end_z':tr['qpos'][[ii[0],end],2].tolist()})
 for a,b in [(0,2),(2.002,2.254),(2.254,3.268),(3.268,4.106),(4.106,4.68),(4.68,4.88),(4.88,5.325),(5.325,5.6)]:
  ii=np.flatnonzero((t[:nt]>=a)&(t[:nt]<b))
  if not len(ii):continue
  out['windows'].append({'window':[a,b],'samples':len(ii),'mean_root_vx':float(tr['qvel'][ii,0].mean()),'min_root_z':float(tr['qpos'][ii,2].min()),'foot_qp_task_residual_rms':rms(c['feet_acceleration_residual'][ii]),'body_qp_task_residual_rms':rms(c['body_acceleration_residual'][ii]),'actual_minus_WBC_qacc_rms':rms(actual_qacc[ii]-c['qacc'][ii]),'actual_minus_WBC_foot_acc_rms':rms(actual_facc[ii]-c['foot_center_acceleration_achieved'][ii]),'actual_minus_mean_tau_rms':rms(tau[ii]-c['tau_mean'][ii]),'dry_model_error_rms':rms(c['friction_compensation'][ii]+c['actual_friction_constraint'][ii]),'actual_minus_qp_force_rms':rms(f[ii]-c['force'][ii])})
 for a,b in [(2.052,2.252),(2.254,4.68),(4.68,4.88)]:
  ii=np.flatnonzero((t[:nt]>=a-1e-8)&(t[:nt]<b-1e-8));sw=c['mode'][ii]==0
  before=(actual_facc[ii]-c['foot_center_acceleration_achieved'][ii])[sw];after=before-correction[ii][sw]
  out['jacdot_bug']['windows'].append({'window':[a,b],'swing_foot_ticks':int(sw.sum()),'old_prediction_RMS_xyz':rms(before),'corrected_old_prediction_RMS_xyz':rms(after),'old_prediction_mean_vertical_error':float(before[:,2].mean()),'corrected_old_prediction_mean_vertical_error':float(after[:,2].mean())})
 for path in sorted((r/'w54-f').glob('mpc-*.npz')):
  z=np.load(path);time=float(z['times'][0]);i=int(np.argmin(abs(t-time)))
  if abs(t[i]-time)>1e-8 or i>=nt:continue
  M=z['M'];JF=z['J_contact_force'].reshape(12,18);dqa=actual_qacc[i]-c['qacc'][i]
  dtau=np.r_[np.zeros(6),tau[i]-c['tau_mean'][i]]
  df=JF.T@(f[i]-c['force'][i]).ravel()
  dry=np.r_[np.zeros(6),c['friction_compensation'][i]+c['actual_friction_constraint'][i]]
  force_acc=np.linalg.solve(M,df);tau_acc=np.linalg.solve(M,dtau);dry_acc=np.linalg.solve(M,dry)
  rem=dqa-force_acc-tau_acc-dry_acc
  if 2.002<=time<4.88:
   kerr=actual_facc[i]-c['foot_center_acceleration_achieved'][i]-np.einsum('lij,j->li',z['J'],dqa)
   for leg in range(4):
    out['jacdot_bug']['kinematic_identity_samples'].append({'time':time,'leg':LEGS[leg],'swing':bool(c['mode'][i,leg]==0),'original_kinematic_error':kerr[leg].tolist(),'missing_term':correction[i,leg].tolist(),'corrected_kinematic_error':(kerr-correction[i])[leg].tolist()})
  out['closure'].append({'time':time,'qacc_delta':dqa.tolist(),'force_delta_acc':force_acc.tolist(),'tau_delta_acc':tau_acc.tolist(),'dry_delta_acc':dry_acc.tolist(),'remainder_acc':rem.tolist(),'generalized_residual':(M@rem).tolist(),'feet_center_acc_delta_from_force':(z['J'].reshape(12,18)@force_acc).reshape(4,3).tolist(),'feet_center_acc_delta_from_tau':(z['J'].reshape(12,18)@tau_acc).reshape(4,3).tolist(),'feet_center_acc_delta_from_dry':(z['J'].reshape(12,18)@dry_acc).reshape(4,3).tolist(),'feet_center_acc_remainder':(z['J'].reshape(12,18)@rem).reshape(4,3).tolist()})
 kis=out['jacdot_bug']['kinematic_identity_samples'];sw=[s for s in kis if s['swing']]
 out['jacdot_bug']['kinematic_identity_swing_summary']={'samples':len(sw),'before_RMS_xyz':rms([s['original_kinematic_error'] for s in sw]),'after_RMS_xyz':rms([s['corrected_kinematic_error'] for s in sw])}
 return out
if __name__=='__main__':
 out={str(v):audit(v) for v in (52,51)}
 (OUT/'f-saved-execution-audit.json').write_text(json.dumps(out,indent=2)+'\n')
 for v,d in out.items():
  print('version',v)
  print(json.dumps(d['windows'],indent=2))
  print(json.dumps(d['swings'],indent=2))
