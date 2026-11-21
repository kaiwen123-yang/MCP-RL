from pathlib import Path
import argparse,numpy as np,json
P=argparse.ArgumentParser();P.add_argument('--repo',type=Path,required=True);P.add_argument('--originals',type=Path,required=True);P.add_argument('--output',type=Path,required=True);a=P.parse_args();w=a.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w56';LEGS=['FL','FR','RL','RR'];out={};sources={r['attempt']:a.repo/r['source']for r in json.loads((w/'phase.json').read_text())['cases']if r.get('source')}
for batch in ['f-a-r55-e600','f-b-r55-e600','f-a-r60-e600','f-b-r60-e600']:
 raw=next((a.originals/batch).glob('job-*/trajectory.npz')).parent;r=json.loads(sources[raw.name].read_text());cols={k:i for i,k in enumerate(r['physical_columns'])}
 with np.load(raw/'trajectory.npz')as z:ts=z['time'];ph=z['phase']
 rows=[]
 for file in raw.glob('physical-*.npz'):
  with np.load(file)as z:x=z['physical']
  t=x[:,0];phase=ph[np.clip(np.searchsorted(ts,t+1e-9)-1,0,len(ts)-1)];fz=x[:,[cols['grf_'+l+'_z']for l in LEGS]];pc=x[:,[cols['contact_'+l]for l in LEGS]]>0;ps=x[:,[cols['foot_position_'+l+'_'+ax+'_m']for l in LEGS for ax in 'xyz']].reshape(-1,4,3)
  for li,l in enumerate(LEGS):
   loaded=fz[:,li]>1;cut=np.r_[0,np.flatnonzero(loaded[1:]!=loaded[:-1])+1,len(t)]
   for s,e in zip(cut[:-1],cut[1:]):
    if loaded[s]or e>=len(t)or phase[s]not in[1,3]or np.any(phase[s:e]!=phase[s]):continue
    duration=t[e]-t[s];apex=np.max(ps[s:e,li,2]-.022)
    if duration>.05 and apex>.005:continue
    if duration<.04:continue
    rows.append(dict(file=file.name,leg=l,start_s=float(t[s]),end_s=float(t[e]),duration_s=float(duration),bottom_height_min_max_m=[float((ps[s:e,li,2]-.022).min()),float(apex)],contact_fraction=float(pc[s:e,li].mean()),Fz_min_max_N=[float(fz[s:e,li].min()),float(fz[s:e,li].max())],support_count_min=int((fz[s:e]>1).sum(1).min()),foot_xy_displacement_m=float(np.linalg.norm(ps[e,li,:2]-ps[s,li,:2])),foot_center_start_xyz=ps[s,li].tolist(),material_interior=bool(.8158113883008419<ps[s,li,0]<2.1841886116991583 and abs(ps[s,li,1])<.5841886116991581)))
 out[batch]=sorted(rows,key=lambda r:r['duration_s'],reverse=True)
 print(batch,len(rows),out[batch][:5])
a.output.write_text(json.dumps(out,indent=2)+'\n')
