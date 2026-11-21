"""Offline independent arithmetic on four full E600 saved trajectories, no simulation.
Contact/support statistics use logged contact or Fz>1 N, never a modified evaluator.
Stride extraction qualifies high-clearance swings by requiring an airborne episode
longer than 50 ms whose foot bottom clears the original horizontal plane by 5 mm.
Every other force-off episode is retained in separate statistics, including rebounds.
"""
from pathlib import Path
import argparse,json,numpy as np,collections,gzip,csv,xml.etree.ElementTree as ET
P=argparse.ArgumentParser();P.add_argument('--repo',type=Path,required=True);P.add_argument('--originals',type=Path,required=True);P.add_argument('--output',type=Path,required=True);a=P.parse_args()
BASE=a.repo/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12';phasejson=json.loads((BASE/'w56/phase.json').read_text());RESULTS={x.get('attempt'):a.repo/x['source'] for x in phasejson['cases'] if x.get('source')};LEGS=['FL','FR','RL','RR'];PHASES=['initial_stand','outbound','turn','return','terminal_stand','complete','far_hold','home_hold','turn_home'];RADIUS=.022

def native(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 raise TypeError(str(type(x)))
def quant(x):
 x=np.array(x);return dict(zip(['min','p05','p50','p95','max'],np.percentile(x,[0,5,50,95,100]).tolist())) if len(x) else None
def segments(x):
 cut=np.r_[0,np.flatnonzero(x[1:]!=x[:-1])+1,len(x)];return list(zip(cut[:-1],cut[1:]))
def metrics(d,mask):
 fz=d['actual_grf'][:,:,2];q=d['qpos'];v=d['qvel'];rp=d['rpy'];b=fz>1.;c=d['contact'];n=b[mask].sum(1);m=(b[mask]*(2**np.arange(4))).sum(1);cm=(c[mask]*(2**np.arange(4))).sum(1);fcvel=np.linalg.norm(d['feet_velocity'][mask,:,:2],axis=2);bearing=np.maximum(fz[mask],0.)
 return dict(samples=int(mask.sum()),interval_weighted_observed_s=float(np.sum(np.r_[np.diff(d['time']),0.][mask])),height_m=quant(q[mask,2]),max_abs_roll_pitch_deg=float(abs(rp[mask,:2]).max()*180/np.pi),linear_speed_m_s=quant(np.linalg.norm(v[mask,:3],axis=1)),world_x_velocity_m_s=quant(v[mask,0]),
  support_count_fraction={str(i):float(np.mean(n==i)) for i in range(5)},force_support_mask_fraction={str(k):v/len(m)for k,v in sorted(collections.Counter(m).items())},logged_contact_mask_fraction={str(k):v/len(cm)for k,v in sorted(collections.Counter(cm).items())},force_support_fraction_per_foot=b[mask].mean(0),logged_contact_fraction_per_foot=c[mask].mean(0),
  force_weighted_foot_center_xy_speed_m_s=float((fcvel*bearing).sum()/bearing.sum()),foot_center_xy_speed_when_Fz_gt1_m_s=quant(fcvel[b[mask]]),max_abs_y_m=float(abs(q[mask,1]).max()),nonfoot_contact_samples=int((d['nonfoot_ground_contacts'][mask]>0).sum()))
def analyze(f):
 result=json.loads(RESULTS[f.parent.name].read_text())
 model=f.parent/'model/G1-project-go2.xml'
 if model.exists():
  xml=ET.parse(model);radii=[float(xml.find('.//geom[@name="'+leg+'"]').get('size').split()[0])for leg in LEGS];assert radii==[RADIUS]*4
 with np.load(f,allow_pickle=False) as z:d={k:z[k] for k in ['time','qpos','qvel','rpy','phase','contact','feet','feet_velocity','actual_grf','nonfoot_ground_contacts','torque','actuator_force','command_velocity','feedback_s','planner_s']}
 t=d['time'];ph=d['phase'];q=d['qpos'];p=d['feet'];fz=d['actual_grf'][:,:,2];v=d['qvel'];rp=d['rpy'];segs=segments(ph);passages=[];lap=[]
 for i,(s,e) in enumerate(segs):
  if i and ph[s-1]in(4,7) and ph[s]in(5,8):lap.append(float(t[s]))
  if ph[s]not in (1,3):continue
  outbound=ph[s]==1;en=np.flatnonzero(q[s:e,0]>=.8 if outbound else q[s:e,0]<=2.2);en=s+int(en[0])if len(en)else None;ex=None
  if en is not None:
   ex=np.flatnonzero(q[en:e,0]>=2.2 if outbound else q[en:e,0]<=.8);ex=en+int(ex[0])if len(ex)else None
  ids=np.arange(en,ex+1 if ex is not None else e)if en is not None else np.array([],int);feet=p[ids];h=np.sqrt(.001)/2
  inside=(feet[:,:,0]>=.8+h)&(feet[:,:,0]<=2.2-h)&(abs(feet[:,:,1])<=.6-h);loaded=bool(np.any(inside&(fz[ids]>0)));side=ids[abs(q[ids,1])>.6]
  passages.append(dict(direction='outbound'if outbound else'return',entry_s=float(t[en])if en is not None else None,exit_s=float(t[ex])if ex is not None else None,material_loaded=loaded,lateral_exit_s=float(t[side[0]])if len(side)else None,completed=bool(en is not None and ex is not None and loaded and not len(side))))
 # Original five independent terminal criteria.
 goodby=dict(contact=d['contact'].all(1),position=np.linalg.norm(q[:,:2],axis=1)<.15,linear_speed=np.linalg.norm(v[:,:3],axis=1)<.05,angular_speed=np.linalg.norm(v[:,3:6],axis=1)<.1,roll_pitch=abs(rp[:,:2]).max(1)<.15);good=np.logical_and.reduce(list(goodby.values()));hold=ph==5
 holdstat=dict(start_s=float(t[hold][0]),duration_s=float(t[-1]-t[hold][0]),samples=int(hold.sum()),violation_samples=int((~good[hold]).sum()),criterion_pass_fraction={k:float(x[hold].mean())for k,x in goodby.items()},max_origin_distance_m=float(np.linalg.norm(q[hold,:2],axis=1).max()),max_linear_speed_m_s=float(np.linalg.norm(v[hold,:3],axis=1).max()),max_angular_speed_rad_s=float(np.linalg.norm(v[hold,3:6],axis=1).max()))
 travel=np.isin(ph,[1,3]);turn=np.isin(ph,[2,8]);strides={};touch=[]
 for l in range(4):
  b=fz[:,l]>1;off=[];small=[]
  for s,e in segments(b):
   if b[s]or e>=len(t) or not travel[s]or not travel[e-1]or np.any(ph[s:e]!=ph[s]):continue
   rec=dict(start_s=float(t[s]),end_s=float(t[e]),duration_s=float(t[e]-t[s]),peak_bottom_height_m=float((p[s:e,l,2]-RADIUS).max()),phase=int(ph[s]),td_vz_m_s=float(d['feet_velocity'][e,l,2]),td_xy_speed_m_s=float(np.linalg.norm(d['feet_velocity'][e,l,:2])))
   if rec['duration_s']>.05 and rec['peak_bottom_height_m']>.005:off.append(rec);touch.append(dict(leg=LEGS[l],**rec))
   else:small.append(rec)
  periods=[x['start_s']-pr['start_s']for pr,x in zip(off,off[1:]) if pr['phase']==x['phase']and x['start_s']-pr['start_s']<1.5]
  strides[LEGS[l]]=dict(qualified_swing_count=len(off),all_force_off_episode_count=len(off)+len(small),non_qualifying_off_episode_count=len(small),non_qualifying_off_duration_s=quant([x['duration_s']for x in small]),non_qualifying_off_apex_height_m=quant([x['peak_bottom_height_m']for x in small]),force_bearing_fraction=float(b[travel].mean()),swing_duration_s=quant([x['duration_s']for x in off]),liftoff_period_s=quant(periods),reciprocal_median_period_Hz=float(1/np.median(periods)),swing_apex_bottom_height_m=quant([x['peak_bottom_height_m']for x in off]),td_center_vz_500Hz_m_s=quant([x['td_vz_m_s']for x in off]))
 official=result['legal_route']['passages'];agree=len(official)==len(passages) and all(all(r[k]==p[k]for k in p)for p,r in zip(passages,official))
 out=dict(attempt=f.parent.name,source_result=str(RESULTS[f.parent.name].relative_to(a.repo)),trajectory_sha_source='sources.json SHA256 and CRC verified by evidence_fetch_originals.py',samples=len(t),end_s=float(t[-1]),passages=passages,passages_exactly_match_evaluator=agree,roundtrip_completion_times_s=lap,roundtrips=len(lap),hold_after_completion=holdstat,whole=metrics(d,np.ones(len(t),bool)),travel=metrics(d,travel),turn=metrics(d,turn),stride_method='Fz>1N off episode >50ms and apex foot center z-0.022>5mm, entirely inside one outbound/return segment; all other off episodes counted separately, not control gates',strides=strides,
  feedback_ms=quant(d['feedback_s']*1000))
 eventpath=a.output.parent/(f.parent.parent.name+'-swing-events.csv.gz')
 with gzip.open(eventpath,'wt')as h:
  writer=csv.DictWriter(h,fieldnames=list(touch[0]));writer.writeheader();writer.writerows(touch)
 out['qualified_swing_events']=eventpath.name
 # First, first-return, and final-return physical chunks only; no extrapolation to all 3e6 substeps.
 phys=[];cols=result['physical_columns']
 for pf in sorted(f.parent.glob('physical-*.npz')):
  with np.load(pf,allow_pickle=False)as z:x=z['physical']
  cdict={k:i for i,k in enumerate(cols)}
  sel=[cdict['torque_'+str(i)]for i in range(12)];tau=x[:,sel];lim=np.array([23.7,23.7,45.43]*4);nf=x[:,cdict['nonfoot_ground_contacts']]>0
  pv=x[:,[cdict[f'foot_velocity_{l}_{ax}_m_s']for l in LEGS for ax in 'xyz']].reshape(-1,4,3)
  fx=x[:,[cdict[f'grf_{l}_{ax}']for l in LEGS for ax in 'xyz']].reshape(-1,4,3)
  pos=x[:,[cdict[f'foot_position_{l}_{ax}_m']for l in LEGS for ax in 'xyz']].reshape(-1,4,3)
  tc=x[:,0];ti=np.clip(np.searchsorted(t,tc+1e-9)-1,0,len(t)-1);pc=ph[ti];trav=np.isin(pc,[1,3]);contact=x[:,[cdict[f'contact_{l}']for l in LEGS]]>0
  bearing=fx[:,:,2]>1.;bn=bearing.sum(1);cn=contact.sum(1);touchstats={}
  for leg in range(4):
   raw=[];swing=[]
   for ss,ee in segments(bearing[:,leg]):
    if bearing[ss,leg]or ee>=len(tc)or not trav[ss]or not trav[ee-1]or np.any(pc[ss:ee]!=pc[ss]):continue
    item=dict(duration_s=float(tc[ee]-tc[ss]),peak_bottom_height_m=float((pos[ss:ee,leg,2]-RADIUS).max()),td_time_s=float(tc[ee]),td_vz_m_s=float(pv[ee,leg,2]),td_force_N=float(fx[ee,leg,2]))
    raw.append(item)
    if item['duration_s']>.05 and item['peak_bottom_height_m']>.005:swing.append(item)
   short=[r for r in raw if not(r['duration_s']>.05 and r['peak_bottom_height_m']>.005)]
   touchstats[LEGS[leg]]=dict(all_force_off_episode_count=len(raw),qualified_swing_count=len(swing),non_qualifying_off_count=len(short),all_force_off_duration_s=quant([r['duration_s']for r in raw]),non_qualifying_off_duration_s=quant([r['duration_s']for r in short]),qualified_swing_td_vz_m_s=quant([r['td_vz_m_s']for r in swing]),worst_downward_touchdown=min(swing,key=lambda r:r['td_vz_m_s'])if swing else None)
  phys.append(dict(file=pf.name,samples=len(x),time_start_s=float(tc[0]),time_end_s=float(tc[-1]),post_integration_end_s=float(x[-1,1]),max_torque_limit_excess_Nm=float(np.maximum(abs(tau)-lim,0).max()),nonfoot_contact_samples=int(nf.sum()),foot_velocity_columns_available=True,travel_samples=int(trav.sum()),travel_support_counts_Fz_gt1={str(i):int(np.sum(bn[trav]==i))for i in range(5)},travel_support_counts_logged_contact={str(i):int(np.sum(cn[trav]==i))for i in range(5)},travel_total_vertical_force_N=quant(fx[trav,:,2].sum(1)),touchdowns=touchstats))
 out['physical_subset']=phys
 return out
res={}
for f in sorted(a.originals.glob('*/job-*/trajectory.npz')):
 r=analyze(f);res[f.parent.parent.name]=r;print(f.parent.parent.name,json.dumps({k:r[k]for k in ['samples','end_s','passages_exactly_match_evaluator','roundtrips','hold_after_completion']}),flush=True)
output={'source_commit':'3590646bc15a59903c9561381368d985963dfe9e','scope':'Saved recordings only; no robot/solver/training was run','foot_radius_m':RADIUS,'radius_provenance':'Four published original model XMLs SHA/CRC verified; foot sphere size .022 parsed for all legs','cases':res};a.output.write_text(json.dumps(output,default=native,indent=2)+'\n')
