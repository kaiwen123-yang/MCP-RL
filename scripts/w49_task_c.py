#!/usr/bin/env python3
"""W49 one held/two-world reset experiment: F1 / S1+partial / S0 restore, <=60 steps."""
import dataclasses,hashlib,importlib.metadata,inspect,json,os,time,traceback
from pathlib import Path
import numpy as np,mujoco,warp as wp,mujoco_warp as mjw
from s1_go2_w41_warp_backend import filter_contact,material_force
ROOT=Path(__file__).resolve().parents[1]
REL=Path('outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
CPU=ROOT/REL/'w46-mechanisms/B158'
GPU=Path('/root/autodl-tmp/MCP-RL')/REL/'gpu/w46'
OUT=ROOT/REL/'w49/task-c'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w49/task-c.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def h(a):
 return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def walk(obj,prefix=''):
 out={}
 for f in dataclasses.fields(obj):
  value=getattr(obj,f.name);name=prefix+f.name
  if isinstance(value,wp.array):out[name]=value
  elif dataclasses.is_dataclass(value):out.update(walk(value,name+'.'))
 return out
def diff(a,b):
 if a.shape!=b.shape:return dict(shape_mismatch=[list(a.shape),list(b.shape)])
 eq=np.equal(a,b)|(np.isnan(a)&np.isnan(b)) if a.dtype.kind=='f' else np.equal(a,b)
 idx=np.argwhere(~eq)
 ans=dict(bitwise_equal=h(a)==h(b),first_index=idx[0].tolist() if len(idx) else None)
 if a.size and a.dtype.kind in 'fiu':
  finite=np.isfinite(a)&np.isfinite(b)
  ans['max_abs']=float(np.max(abs(a[finite].astype(float)-b[finite].astype(float)))) if np.any(finite) else None
  if a.dtype.kind=='f' and a.dtype.itemsize in (4,8):
   signed='i'+str(a.dtype.itemsize);rawa=a.view(signed).astype(object);rawb=b.view(signed).astype(object)
   maximum=1<<(a.dtype.itemsize*8-1)
   aa=np.where(rawa<0,-maximum-rawa,rawa);bb=np.where(rawb<0,-maximum-rawb,rawb)
   ans['max_ULP']=int(np.max(abs(aa[finite]-bb[finite]))) if np.any(finite) else None
 return ans
def main():
 begin=time.time();OUT.mkdir(parents=True,exist_ok=False)
 r=dict(status='initializing',pid=os.getpid(),started_unix_s=begin,deadline=None,wall_deadline_enabled=False,
  affinity=sorted(os.sched_getaffinity(0)),gpu_world_steps=0,warmup_world_steps=0,native_physics=0,
  native_solves=0,FD=0,new_PPO=0,equivalent_native_ranking=False,full_native_cost_implemented_on_gpu=False,
  original_task_qualified=False,dispatch='direct synchronized mjw.step in every arm; no external CUDA capture',
  conditions=[],initialization_scope='canonical declared cold-start, not historical complete plant restoration')
 write(SMALL,r)
 try:
  sources=[CPU/'input-state.npz',CPU/'input.json',CPU/'original-baseline.bin',CPU/'original-policy.bin',GPU/'paired-states-actions.npz',GPU/'B158-model.xml']
  r['sources']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sources]
  def policy_arrays(p):
   z=np.fromfile(p,dtype='<f8');hh=z[:16];T,ns,nu,nx,nr=map(int,hh[3:8]);i=16;vals=[]
   for shape in [(T,ns),(T-1,nu),(T,),(T,nr),(T,nu,nx),(T,nu,nu)]:
    n=int(np.prod(shape));vals.append(z[i:i+n].copy());i+=n
   return vals,hh[[12,13]]
  aa,ah=policy_arrays(CPU/'original-baseline.bin');bb,bh=policy_arrays(CPU/'original-policy.bin')
  assert all(np.array_equal(a,b) for a,b in zip(aa,bb)) and np.array_equal(ah,bh)
  z=np.load(GPU/'paired-states-actions.npz');held0=z['gpu_actions'][0,18].copy();held1=z['gpu_actions'][0,21].copy()
  r['old_held_comparison']=diff(held0,held1);held=held0.astype(np.float32)
  r['held_fp64']=held0.tolist();r['held_fp32']=held.tolist()
  r['optional_native_action_batch']='not_started; conditional batch remains separate and limited to 8 identities'
  m=mujoco.MjModel.from_xml_path(str(GPU/'B158-model.xml'))
  m.opt.timestep=.0002;m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  m.jnt_solimp[:,0]=0;m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
  st=np.load(CPU/'input-state.npz');canonical=mujoco.MjData(m);mujoco.mj_resetData(m,canonical)
  canonical.qpos[:]=st['qpos'];canonical.qvel[:]=st['qvel'];canonical.userdata[:]=st['userdata'];canonical.time=float(st['time']);canonical.ctrl[:]=held
  for name in ['act','mocap_pos','mocap_quat']:
   if name in st:np.copyto(getattr(canonical,name),st[name])
  integration=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION))
  mujoco.mj_getState(m,canonical,integration,mujoco.mjtState.mjSTATE_INTEGRATION)
  np.savez_compressed(OUT/'canonical-integration.npz',integration=integration,held_fp64=held0,held_fp32=held)
  r['canonical_fields']={}
  for name in ['act','mocap_pos','mocap_quat','eq_active','qfrc_applied','xfrc_applied','userdata','qacc_warmstart','plugin_state','history']:
   if hasattr(canonical,name):
    val=np.asarray(getattr(canonical,name))
    r['canonical_fields'][name]=dict(shape=list(val.shape),sha256=h(val),source='saved' if name in st else 'mj_resetData default')
   else:r['canonical_fields'][name]={'status':'not_exposed_by_this_version'}
  r['versions']={k:importlib.metadata.version(k) for k in ['mujoco','mujoco-warp','warp-lang']}
  r['sources_code']={str(p):sha(p) for p in [Path(inspect.getsourcefile(mjw.put_data)),Path(inspect.getsourcefile(filter_contact.func)) if hasattr(filter_contact,'func') else ROOT/'scripts/s1_go2_w41_warp_backend.py',ROOT/'scripts/s1_go2_w41_warp_backend.py']}
  r['options']={k:np.asarray(getattr(m.opt,k)).tolist() for k in ['timestep','integrator','solver','cone','jacobian','iterations','ls_iterations','tolerance','disableflags','enableflags']}
  r['physical_model_arrays']={k:h(getattr(m,k)) for k in ['geom_solimp','pair_solimp','jnt_solimp','body_mass','actuator_gainprm','actuator_biasprm','actuator_forcerange','geom_friction']}
  wm=mjw.put_model(m);wm.opt.tolerance.fill_(float(m.opt.tolerance))
  d=mjw.put_data(m,canonical,nworld=2,nconmax=128,njmax=768)
  feet=wp.array([m.geom(n).id for n in ['FL','RL','FR','RR']],dtype=int)
  stiff=float(m.numeric('r10_local_contact').data[0]);floor=float(m.numeric('s1_flat_ground_height').data[0])
  def cf(wm,d):wp.launch(filter_contact,dim=d.naconmax,inputs=[d.contact.geom,d.contact.worldid,d.nacon,d.geom_xpos,feet],outputs=[d.contact.dist])
  def pf(wm,d):wp.launch(material_force,dim=(2,18),inputs=[d.geom_xpos,wm.geom_size,wm.geom_bodyid,feet,d.userdata,d.qvel,wm.body_parentid,wm.body_rootid,wm.dof_bodyid,wm.body_isdofancestor,d.subtree_com,d.cdof,stiff,floor],outputs=[d.qfrc_passive])
  wm.callback.contactfilter=cf;wm.callback.passive=pf
  arrays=walk(d)
  def snap():wp.synchronize();return {k:v.numpy().copy() for k,v in arrays.items()}
  def restore(x):
   for name,v in x.items():wp.copy(arrays[name],wp.array(v,dtype=arrays[name].dtype,device=arrays[name].device))
   wp.synchronize()
  S0=snap();S1=None
  r['device_initial_fields']={k:dict(shape=list(v.shape),dtype=str(v.dtype),bytes=v.nbytes,sha256=h(v)) for k,v in S0.items()}
  r['snapshot_coverage']='All exposed dataclass Data arrays including contact/constraint workspace; immutable scalar capacities retained in the same Data object. Driver/allocator internal state is not exposed. No outer capture.'
  r['persistent_input_audit']='step calls forward before integration; qpos/qvel/act/time/userdata/mocap/eq_active/applied forces are explicit inputs; disabled warmstart; all exposed solver scratch is snapshotted, uninitialized output padding is not compared as physical state.'
  required=['qpos','qvel','act','time','userdata','mocap_pos','mocap_quat','eq_active','qfrc_applied','xfrc_applied','ctrl','qacc_warmstart']
  r['initial_input_comparison']={k:diff(S0[k][0],S0[k][1]) for k in required if k in S0 and S0[k].ndim and S0[k].shape[0]==2}
  assert all(v['bitwise_equal'] for v in r['initial_input_comparison'].values()),'Required device inputs differ before substep0'
  assert np.array_equal(S0['ctrl'],np.tile(held,(2,1)))
  r['host_device_ctrl_equal']=True;r['device_tolerance']=wm.opt.tolerance.numpy().tolist()
  r['status']='running';write(SMALL,r)
  records={};timing=[]
  physical=['qpos','qvel','act','time','ctrl','qacc','qacc_warmstart','qfrc_actuator','actuator_force','qfrc_passive','qfrc_constraint','qfrc_applied','xfrc_applied','solver_niter','nefc','ncon']
  for condition in ['F1','L','F2']:
   if condition=='F1':restore(S0)
   elif condition=='L':
    restore(S1)
    for name in ['qpos','qvel','userdata','time','ctrl']:
     wp.copy(arrays[name],wp.array(S0[name],dtype=arrays[name].dtype))
    d.qacc_warmstart.zero_();wp.synchronize()
   else:restore(S0)
   current=snap();condition_row=dict(condition=condition,first_difference=None,initial_vs_S0={k:diff(current[k],S0[k]) for k in required if k in current},substeps=[])
   if condition!='L':assert all(h(current[k])==h(S0[k]) for k in S0),'Complete exposed snapshot restore changed bytes'
   records[condition]=[current]
   for j in range(10):
    wp.synchronize();tick=time.perf_counter()
    mjw.step(wm,d);wp.synchronize();elapsed=time.perf_counter()-tick
    r['gpu_world_steps']+=2;current=snap();records[condition].append(current)
    if condition=='F1' and j==0:S1={k:v.copy() for k,v in current.items()}
    pair={k:diff(current[k][0],current[k][1]) for k in physical if k in current and current[k].ndim and current[k].shape[0]==2}
    first=next((k for k in physical if k in pair and not pair[k]['bitwise_equal']),None)
    if first and condition_row['first_difference'] is None:condition_row['first_difference']=dict(substep=j+1,field=first,detail=pair[first])
    contact={}
    if 'nacon' in current:
     n=int(current['nacon'].flat[0]);assert n<=d.naconmax,'contact capacity overflow'
     ids=current['contact.worldid'][:n]
     for w in range(2):
      indices=np.flatnonzero(ids==w);items=[]
      for i in indices:
       items.append({k:current['contact.'+k][i].tolist() for k in ['geom','dist','frame','efc_address'] if 'contact.'+k in current})
      contact[str(w)]=sorted(items,key=lambda x:(x.get('geom',[]),x.get('dist',0)))
    step=dict(substep=j+1,physics_after_origin_s=(j+1)*.0002,world_comparison=pair,contacts=contact,step_wall_s=elapsed)
    condition_row['substeps'].append(step)
    write(OUT/(condition+'-progress.json'),condition_row);write(SMALL,r)
   r['conditions'].append(condition_row);write(SMALL,r)
  comparisons={}
  for condition in ['L','F2']:
   comparisons[condition+'_vs_F1']=[dict(substep=j,fields={k:diff(records[condition][j][k],records['F1'][j][k]) for k in physical if k in S0}) for j in range(11)]
  r['same_world_reset_comparisons']=comparisons
  raw={}
  for cond,series in records.items():
   for j,x in enumerate(series):
    for name,value in x.items():raw[f'{cond}_{j:02d}_{name.replace(".","_")}']=value
  np.savez_compressed(OUT/'all-exposed-substeps.npz',**raw)
  r.update(status='completed',ended_unix_s=time.time(),warmup_world_steps=0,
   timing_scope='synchronized first F1 dispatch includes JIT compilation; subsequent dispatch times recorded separately, not throughput',
   unavailable=['driver/allocator hidden state','unexposed physical solver KKT residual'],
   artifact=dict(path=str(OUT/'all-exposed-substeps.npz'),bytes=(OUT/'all-exposed-substeps.npz').stat().st_size,sha256=sha(OUT/'all-exposed-substeps.npz')))
 except Exception:
  r.update(status='technical_failure',traceback=traceback.format_exc());raise
 finally:
  r['wall_s']=time.time()-begin;write(SMALL,r)
if __name__=='__main__':main()
