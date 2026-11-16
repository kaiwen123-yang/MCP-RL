#!/usr/bin/env python3
"""Read the complete saved W52 diagnostics; no physical or optimizer execution."""
import json,time
from collections import Counter
from pathlib import Path
import numpy as np
import mujoco
from publish_s1_go2_w52 import ROOT,SMALL,LARGE,write
W=SMALL/'w52'
def jsonl(p):return [json.loads(x) for x in p.read_text().splitlines()]
def first(t,mask):
 ids=np.flatnonzero(mask);return float(t[ids[0]]) if len(ids) else None
def bouts(t,mask):
 edge=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int)))
 return [dict(begin_s=float(t[a]),end_s=float(t[b]) if b<len(t) else float(t[-1]),
   duration_s=float((t[b] if b<len(t) else t[-1])-t[a])) for a,b in zip(edge[::2],edge[1::2])]
def analyze(arm,summary):
 out=LARGE/summary['attempt']
 original=json.loads((SMALL/(summary['attempt']+'.json')).read_text())
 with np.load(out/'trajectory.npz') as f:tr={k:f[k] for k in f.files}
 model=mujoco.MjModel.from_binary_path(str(out/'model.mjb'))
 calls=jsonl(out/'planner-calls.jsonl');bycall={c['call_id']:c for c in calls}
 seeds=jsonl(out/'w51-seeds.jsonl');motion=[s for s in seeds if 'motion' in s]
 t=tr['time'];fz=tr['actual_grf'][:,:,2];loads=fz>1;clear=tr['feet'][:,:,2]-.022
 def wins(lo,hi):
  ids=np.flatnonzero((t>=lo-1e-9)&(t<=hi+1e-9))
  if not len(ids):return dict(requested=[lo,hi],status='not_reached')
  ts=t[ids];dt=np.diff(ts);weight=dt.sum();ll=loads[ids];mask=ll.astype(int)@np.array([1,2,4,8])
  frac=lambda x:float(np.sum(x[:-1]*dt)/weight) if weight else None
  result=dict(requested=[lo,hi],actual=[float(ts[0]),float(ts[-1])],
   dx_m=float(tr['qpos'][ids[-1],0]-tr['qpos'][ids[0],0]),
   command_forward_displacement_m=float(np.sum(tr['command_velocity'][ids[:-1],0]*dt)),
   min_z_m=float(tr['qpos'][ids,2].min()),max_abs_roll_pitch_rad=np.max(abs(tr['rpy'][ids,:2]),axis=0).tolist(),
   four_foot_fraction=frac(mask==15),no_foot_fraction=frac(mask==0),
   diagonal_mask_6_fraction=frac(mask==6),diagonal_mask_9_fraction=frac(mask==9),
   nonfoot_load_fraction=frac(tr['nonfoot_ground_force'][ids,2]>1),
   loaded_transitions=np.count_nonzero(np.diff(ll.astype(int),axis=0),axis=0).tolist(),
   clearance_max_mm=(clear[ids].max(axis=0)*1000).tolist(),unloaded_bouts=[])
  for i in range(4):
   runs=bouts(ts,~ll[:,i]);dur=np.array([x['duration_s'] for x in runs])
   result['unloaded_bouts'].append(dict(foot=['FL','FR','RL','RR'][i],count=len(runs),
    duration_ms_quantiles=(1000*np.quantile(dur,[0,.5,.9,1])).tolist() if len(dur) else None,
    longest=max(runs,key=lambda x:x['duration_s']) if runs else None))
  return result
 windows=[wins(0,2),wins(2.002,4.3),wins(4.3,float(t[-1]))]
 physical=np.concatenate([np.load(p)['physical'] for p in sorted(out.glob('physical-*.npz'))])
 col={n:i for i,n in enumerate(original['physical_columns'])};pt=physical[:,col['force_evaluation_time_s']]
 pd=physical[:,[col[f'unclipped_pd_request_{i}'] for i in range(12)]]
 limits=np.max(abs(model.actuator_forcerange),axis=1);pdex=np.max(abs(pd)-limits,axis=1)
 af=physical[:,[col[f'actuator_force_{i}'] for i in range(12)]]
 action={x['query_id']:x for x in jsonl(out/'feedback-actions.jsonl')}
 executed=[x for x in jsonl(out/'action-execution.jsonl') if x['query_kind']=='applied_hold']
 fa=[action[x['query_id']] for x in executed];ht=np.array([x['time_s'] for x in executed])
 hard=np.array([np.max(abs(np.array(x['action_before_clip'])-x['center_before_projection']))>1e-12 for x in fa])
 qp=np.array([x['projection_changed'] for x in fa],dtype=bool)
 total=np.array([np.max(abs(np.array(x['action_before_clip'])-x['action_after_clip']))>1e-12 for x in fa])
 kval=np.array([np.max(abs(np.array(x['raw_K_delta_x']))) for x in fa])
 expired=np.array([tt>x['reference_end']+1e-9 for tt,x in zip(ht,fa)])
 events=dict(first_z_below_028_s=first(t,tr['qpos'][:,2]<.28),
  first_z_below_025_s=first(t,tr['qpos'][:,2]<.25),first_hard_clip_s=first(ht,hard),
  hard_clip_hold_count=int(hard.sum()),first_QP_change_s=first(ht,qp),QP_change_hold_count=int(qp.sum()),
  any_target_rewrite_hold_count=int(total.sum()),K_delta_max_rad=float(kval.max()),
  K_delta_max_time_s=float(ht[kval.argmax()]),first_PD_excess_001Nm_s=first(pt,pdex>.01),
  PD_excess_001Nm_substeps=int((pdex>.01).sum()),PD_excess_1Nm_substeps=int((pdex>1).sum()),
  unclipped_PD_max_excess_Nm=float(pdex.max()),actual_actuator_max_limit_ratio=float(np.max(abs(af)/limits)),
  first_nonfoot_1N_s=first(pt,physical[:,col['nonfoot_grf_z']]>1),
  max_nonfoot_Fz_N=float(physical[:,col['nonfoot_grf_z']].max()),
  truly_expired_held_queries=int(expired.sum()),first_expired_hold_s=first(ht,expired),
  held_queries=len(fa),last_policy_generation_s=calls[-1]['policy_generation_time'],
  last_reference=[calls[-1]['reference_start'],calls[-1]['reference_end']])
 material=[]
 for p in sorted(out.glob('material-[0-9]*.npz')):
  with np.load(p) as z:material.append(dict(file=p.name,cells_shape=list(z['cells'].shape)))
 with np.load(out/'material-final.npz') as z:
  final_material={k:list(z[k].shape) for k in ['keys','loaded','ever_loaded']}
 # Read all saved S-native trajectories and all candidate headers at their epochs.
 native=[]
 for s in motion:
  meta=s['motion']
  if not meta['S_valid']:continue
  cid=s['call_id'];path=out/f'r33-candidates-call{cid:06d}.npz'
  record=dict(call_id=cid,time_s=s['time_s'],B_J=meta['B_J'],chosen=bycall[cid]['candidate_source'],
    retained_candidate_file=path.name)
  if not path.exists():record['status']='unavailable_saved_candidate';native.append(record);continue
  with np.load(path) as z:
   headers=[(k[:-6],z[k]) for k in z.files if k.endswith('header')]
   prefix,h=next((k,h) for k,h in headers if int(h[2])==4)
   states=z[prefix+'accepted_states'];feet=z[prefix+'foot_rows'].reshape(-1,29)
   ff=feet[:,1:].reshape(-1,4,7);fl=(ff[:,:,4]+ff[:,:,5])>1
   with np.load(out/s['objects']['IK']['file']) as k:raw=k['kinematics_payload']
   n=int(raw[3]);target=raw[5+n:5+n+12*n].reshape(n,4,3)[:,[0,2,1,3]]
   record.update(status='read',S_J=float(h[11]),S_minus_B_J=float(h[11]-meta['B_J']),
    native_dx_m=float(states[-1,0]-states[0,0]),native_z_min_m=float(states[:,2].min()),
    native_four_foot_fraction=float(fl.all(axis=1).mean()),
    native_foot_max_clearance_mm=(ff[:,:,3].max(axis=0)*1000).tolist(),
    target_max_clearance_mm=((target[:,:,2]-.022).max(axis=0)*1000).tolist(),
    max_native_target_foot_error_mm=(np.max(abs(ff[:n,:,:3]-target),axis=(0,2))*1000).tolist(),
    max_under_target_vertical_mm=(np.max(target[:,:,2]-ff[:n,:,2],axis=0)*1000).tolist(),
    candidates=[])
   decisions=bycall[cid]['r34_decisions']
   for i,(pr,hd) in enumerate(headers):
    st=z[pr+'accepted_states'];ft=z[pr+'foot_rows'].reshape(-1,29)[:,1:].reshape(-1,4,7)
    ld=(ft[:,:,4]+ft[:,:,5])>1
    record['candidates'].append(dict(source=decisions[i]['source'],kind=int(hd[2]),slot=int(hd[3]),alpha=float(hd[4]),
      J=float(hd[11]),risk=float(hd[12]),failure=bool(hd[13]),dx_m=float(st[-1,0]-st[0,0]),
      min_z_m=float(st[:,2].min()),four_foot_fraction=float(ld.all(axis=1).mean()),
      max_clearance_mm=(ft[:,:,3].max(axis=0)*1000).tolist()))
  native.append(record)
 manifest=json.loads((out/'w50-event-manifest.json').read_text())['event_calls']
 retained={x['call_record']['call_id'] for x in manifest}
 installed={c['call_id'] for c in calls if c['policy_updated']}
 selected_coverage=dict(installed_calls=len(installed),retained_complete_calls=len(retained),
  missing_complete_selected_calls=sorted(installed-retained),
  missing_selected_candidate_rings=sorted(c for c in installed if not (out/f'r33-candidates-call{c:06d}.npz').exists()),
  directly_installed_S=[c['call_id'] for c in calls if c['candidate_source']=='motion_seed_S' and c['policy_updated']])
 resource=jsonl(ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/jobs'/('job-20261116T110800Z-w52-'+arm.lower())/'resources.jsonl')
 job=json.loads((W/(arm+'.json')).read_text())
 resource_stats=dict(**job['stages'][0]['resource_usage'],samples=len(resource),
  sampled_process_tree_peak_rss_bytes=max(x['live_rss_bytes'] for x in resource),
  sampled_card_peak_used_MiB=max(float(x['GPU_observation'].split(',')[1]) for x in resource if x['GPU_observation']),
  sampled_card_peak_utilization_percent=max(float(x['GPU_observation'].split(',')[0]) for x in resource if x['GPU_observation']))
 source_count=Counter(c['candidate_source'].split('homotopy')[0]+'homotopy' if 'homotopy' in c['candidate_source'] else c['candidate_source'] for c in calls if c['policy_updated'])
 result=dict(scope='Full actual trajectory, all physical rows, all applied queries, all active seed metadata; all retained valid-S candidate epochs. No new physics.',
  full_trajectory_rows=len(t),physical_substeps=len(physical),windows=windows,events=events,
  command_readback_max_error=float(np.max(abs(tr['support_reference_userdata'][:,78:81]-tr['command_velocity']))),
  original_invalid_contact=dict(pair=[0,13],geom0=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,0),
   geom13=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,13),body13=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[13]))),
  final_rpy_rad=tr['rpy'][-1].tolist(),material_chunks=material,material_final_shapes=final_material,
  installed_sources=dict(source_count),selected_object_coverage=selected_coverage,all_valid_S_native=native,resources=resource_stats,
  video=dict(decoded_frames=172 if arm=='D2' else 200,all_frames_viewed=True,
   scope='Every encoded frame viewed chronologically in 30-frame close-up sheets; timing/force conclusions use original physical logs, not container duration'),
  selection='not_selected_no_sustained_normal_support_progression')
 if arm=='D3':
  branches=jsonl(out/'w52-branches.jsonl')
  result['dual_search']=dict(dual_calls=sum(len(x['branches'])==2 for x in branches),
   chosen_sources=dict(Counter(x['selected_source'].split('homotopy')[0]+'homotopy' if 'homotopy' in x['selected_source'] else x['selected_source'] for x in branches)),
   branch_backward_failures={name:sum(not b['first_valid'] for x in branches for b in x['branches'] if b['source']==name) for name in ['B','S']})
 return result,tr
def plot(traces):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap
 from PIL import Image
 plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axs=plt.subplots(5,2,figsize=(9.2,10.5),sharex=True,layout='constrained')
 blue='#0072B2';orange='#D55E00'
 xmin=min(x['qpos'][:,0].min() for x in traces.values())*1000-5
 xmax=max(x['qpos'][:,0].max() for x in traces.values())*1000+5
 zmax=max(x['qpos'][:,2].max() for x in traces.values())+.005
 for col,(arm,tr) in enumerate(traces.items()):
  t=tr['time'];q=tr['qpos'];load=tr['actual_grf'][:,:,2]>1
  axs[0,col].plot(t,q[:,0]*1000,color=blue,lw=1.4);axs[0,col].axhline(0,color='.6',lw=.7)
  axs[0,col].set(ylabel='Body x (mm)',title=f'{arm}: original termination at {t[-1]:.4f} s',ylim=(xmin,xmax))
  axs[1,col].plot(t,q[:,2],color=blue,lw=1.4);axs[1,col].set(ylabel='Body z (m)',ylim=(.20,zmax))
  axs[2,col].plot(t,np.rad2deg(tr['rpy'][:,0]),color=blue,label='Roll',lw=1.2)
  axs[2,col].plot(t,np.rad2deg(tr['rpy'][:,1]),color=orange,ls='--',label='Pitch',lw=1.2)
  axs[2,col].set(ylabel='Attitude (deg)',ylim=(-52,52));axs[2,col].legend(loc='upper center',frameon=False,ncol=2)
  # Rectangles use actual nonuniform terminal intervals; no extension to t=8.
  axs[3,col].pcolormesh(t,np.arange(5)-.5,load[:-1].T,cmap=ListedColormap(['white',blue]),vmin=0,vmax=1,shading='flat',rasterized=True)
  axs[3,col].set(yticks=np.arange(4),yticklabels=['FL','FR','RL','RR'],ylabel='Loaded: Fz > 1 N',ylim=(3.5,-.5))
  original=json.loads((W/(arm+'.json')).read_text())
  attempt=original['stages'][0]['attempt'];physical=np.load(LARGE/attempt/'physical-000.npz')['physical']
  cols=json.loads((SMALL/(attempt+'.json')).read_text())['physical_columns']
  axs[4,col].plot(physical[:,0],physical[:,cols.index('nonfoot_grf_z')],color=orange,lw=1.2)
  axs[4,col].set(ylabel='Non-foot Fz (N)',xlabel='Physical time (s)',ylim=(-10,920))
  for row in range(5):
   ax=axs[row,col];ax.set_xlim(0,8);ax.axvline(2,color='.5',ls=':',lw=.8);ax.axvline(4.3,color='.5',ls=':',lw=.8)
   ax.axvspan(t[-1],8,color='.92',zorder=-5);ax.spines[['top','right']].set_visible(False)
   ax.text(.02,.96,chr(97+2*row+col),transform=ax.transAxes,fontweight='bold',va='top',bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=1))
   if row!=3:ax.grid(axis='y',alpha=.15)
 fig.suptitle('W52: contact changes without sustained forward progression',fontsize=12)
 p=W/'full-trajectory.png';fig.savefig(p,dpi=180);fig.savefig(W/'full-trajectory.svg')
 Image.open(p).convert('L').save(LARGE/'w52-analysis/full-trajectory-grayscale.png')
 # Check exported text stays inside the actual canvas; no runtime/control tests.
 fig.canvas.draw();box=fig.bbox;renderer=fig.canvas.get_renderer()
 clipped=[]
 for txt in fig.findobj(matplotlib.text.Text):
  if not txt.get_visible() or not txt.get_text():continue
  b=txt.get_window_extent(renderer)
  if b.width and b.height and (b.x0<box.x0-1 or b.x1>box.x1+1 or b.y0<box.y0-1 or b.y1>box.y1+1):clipped.append(txt.get_text())
 print('figure_text_outside_canvas',clipped,flush=True);plt.close(fig)
def main():
 paired=json.loads((W/'paired-results.json').read_text());traces={}
 for arm,summary in paired['arms'].items():
  result,tr=analyze(arm,summary);summary['full_trajectory_analysis']=result;traces[arm]=tr
  p=W/'original-evidence'/(summary['attempt']+'.json');write(p,summary)
  print(arm,json.dumps(dict(events=result['events'],nativeS=[{k:x[k] for k in ['call_id','time_s','S_minus_B_J','native_dx_m','native_four_foot_fraction','native_foot_max_clearance_mm','target_max_clearance_mm']} for x in result['all_valid_S_native'] if x['status']=='read']),ensure_ascii=False),flush=True)
 paired.update(selection=None,selection_status='completed_negative_no_controller_selected',
  decision=dict(original_E30_starts=0,reason='Both original D8 runs ended early with floor-trunk invalid_contact, no sustained normal forward support progression.',
   D3_longer_survival_is_not_qualification=True,formal_chain='unrun; not eight failed experiments',new_physics_this_analysis=0))
 write(W/'paired-results.json',paired);plot(traces)
if __name__=='__main__':main()
