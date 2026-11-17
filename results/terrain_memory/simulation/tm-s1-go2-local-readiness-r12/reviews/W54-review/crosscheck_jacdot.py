"""Independent saved-velocity kinematic identity check; no model calls or new finite-difference rollouts."""
from pathlib import Path
import argparse,csv,json,numpy as np
P=argparse.ArgumentParser(description=__doc__);P.add_argument('--originals',type=Path,required=True);P.add_argument('--output',type=Path,required=True);args=P.parse_args();O=args.originals;args.output.mkdir(parents=True,exist_ok=True);out={}
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
for rev in [51,52]:
 p=next(O.glob('*-r'+str(rev)+'-*'));tr=load(p/'trajectory.npz');ph=load(p/'physical-000.npz')['physical'];chunks=[load(f) for f in sorted((p/'w54-f').glob('control-*.npz'))];control={k:np.concatenate([v[k] for v in chunks]) for k in chunks[0]};tt=control['time'];velocity=ph[::10,73:85].reshape(-1,4,3);assert np.max(abs(velocity-tr['feet_velocity'][:len(velocity)]))==0
 methods={'forward':[],'central':[]}
 for f in sorted((p/'w54-f').glob('mpc-*.npz')):
  q=load(f);t=float(q['times'][0]);i=int(np.argmin(abs(tt-t)))
  if not (2.002<=t<4.88 and i>0 and i+1<len(velocity)):continue
  assert abs(tt[i]-t)<1e-8
  bias=control['foot_center_acceleration_achieved'][i]-np.einsum('lij,j->li',q['J'],control['qacc'][i]);missing=np.cross(control['foot_angular_velocity'][i],control['foot_velocity'][i]);swing=control['mode'][i]==0
  for method,left in [('forward',i),('central',i-1)]:
   right=i+1;step=tr['time'][right]-tr['time'][left];qa=(tr['qvel'][right]-tr['qvel'][left])/step;fa=(velocity[right]-velocity[left])/step
   error=fa-np.einsum('lij,j->li',q['J'],qa)-bias
   for leg in np.flatnonzero(swing):methods[method].append(dict(time_s=t,leg_index=int(leg),before=error[leg].tolist(),missing=missing[leg].tolist(),after=(error[leg]-missing[leg]).tolist()))
 result={'samples':methods,'physical_velocity_matches_500Hz_trajectory_exactly':True,'scope':'2.002 <= t < 4.88; swing-foot records at saved MPC J times; no corrected control or new physical run.'}
 for k,v in methods.items():
  before=np.array([x['before'] for x in v]);after=np.array([x['after'] for x in v]);miss=np.array([x['missing'] for x in v]);result[k]={'swing_foot_times':len(v),'before_RMS_all_xyz':float(np.sqrt(np.mean(before**2))),'after_RMS_all_xyz':float(np.sqrt(np.mean(after**2))),'before_mean_vertical':float(before[:,2].mean()),'after_mean_vertical':float(after[:,2].mean()),'corr_missing_to_error':float(np.corrcoef(miss.ravel(),before.ravel())[0,1])}
 out[str(rev)]=result;print(rev,result['forward'],result['central'])
(args.output/'jacdot-independent-crosscheck.json').write_text(json.dumps(out,indent=2)+'\n')
(args.output/'jacdot-summary.json').write_text(json.dumps({k:{x:v[x] for x in v if x!='samples'} for k,v in out.items()},indent=2)+'\n')
with (args.output/'jacdot-r52-central.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['physical_time_s','leg_index','before_x_m_s2','before_y_m_s2','before_z_m_s2','missing_x_m_s2','missing_y_m_s2','missing_z_m_s2','after_x_m_s2','after_y_m_s2','after_z_m_s2'])
 for v in out['52']['samples']['central']:w.writerow([v['time_s'],v['leg_index']]+v['before']+v['missing']+v['after'])
