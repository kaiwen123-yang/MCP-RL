#!/usr/bin/env python3
"""W46 same full-policy mixed FP64 action/FP32 dynamics; finite 24 scenarios."""
import ctypes,json,os,time,traceback,xml.etree.ElementTree as ET,sys
from pathlib import Path
import numpy as np,mujoco,warp as wp,mujoco_warp as mjw
from s1_go2_w41_warp_backend import filter_contact,material_force
ROOT=Path(__file__).resolve().parents[1]
CPU=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46-mechanisms'
OUT=Path('/root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w46')
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46'
sys.path.insert(0,str(ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review'))
from policy_arrays import raw_policy,interp
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def ma(a):return float(np.max(abs(a)))
@wp.kernel
def feedback(x:wp.array2d[wp.float64],r:wp.array2d[wp.float64],ff:wp.array2d[wp.float64],K:wp.array3d[wp.float64],
 scale:wp.array[wp.float64],bounds:wp.array2d[wp.float64],out:wp.array2d[wp.float64]):
 w,a=wp.tid()
 qw=r[w,3]*x[w,3]+r[w,4]*x[w,4]+r[w,5]*x[w,5]+r[w,6]*x[w,6]
 qx=r[w,3]*x[w,4]-r[w,4]*x[w,3]-r[w,5]*x[w,6]+r[w,6]*x[w,5]
 qy=r[w,3]*x[w,5]+r[w,4]*x[w,6]-r[w,5]*x[w,3]-r[w,6]*x[w,4]
 qz=r[w,3]*x[w,6]-r[w,4]*x[w,5]+r[w,5]*x[w,4]-r[w,6]*x[w,3]
 norm=wp.sqrt(qw*qw+qx*qx+qy*qy+qz*qz);qw=qw/norm;qx=qx/norm;qy=qy/norm;qz=qz/norm
 sn=wp.sqrt(qx*qx+qy*qy+qz*qz);ang=wp.float64(2.0)*wp.atan2(sn,qw)
 if ang>wp.float64(3.141592653589793):ang=ang-wp.float64(6.283185307179586)
 factor=wp.float64(2.0)
 if sn>wp.float64(1.e-15):factor=ang/sn
 val=wp.float64(0.0)
 for j in range(36):
  dx=wp.float64(0.0)
  if j<3:dx=x[w,j]-r[w,j]
  elif j==3:dx=qx*factor
  elif j==4:dx=qy*factor
  elif j==5:dx=qz*factor
  elif j<18:dx=x[w,j+1]-r[w,j+1]
  else:dx=x[w,j+1]-r[w,j+1]
  val=val+K[w,a,j]*dx
 out[w,a]=wp.clamp(ff[w,a]+scale[w]*val,bounds[a,0],bounds[a,1])
@wp.kernel
def risks(x:wp.array2d[wp.vec3],mat:wp.array2d[wp.mat33],sizes:wp.array2d[wp.vec3],kinds:wp.array[int],
 trunk:wp.array[int],margins:wp.array[float],floor:float,risk:wp.array[float]):
 w=wp.tid();worst=float(0.0)
 for k in range(trunk.shape[0]):
  g=trunk[k];size=sizes[0,g];r=mat[w,g];extent=float(0.0)
  if kinds[g]==6:extent=wp.abs(r[2,0])*size[0]+wp.abs(r[2,1])*size[1]+wp.abs(r[2,2])*size[2]
  elif kinds[g]==3:extent=size[0]+wp.abs(r[2,2])*size[1]
  elif kinds[g]==2:extent=size[0]
  worst=wp.max(worst,margins[g]-(x[w,g][2]-extent-floor))
 risk[w]=wp.max(risk[w],worst)
class Bridge:
 def __init__(self,path):
  self.lib=ctypes.CDLL(str(path),mode=os.RTLD_LOCAL|os.RTLD_DEEPBIND)
  P=ctypes.c_void_p;D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
  self.lib.w46_create.argtypes=[ctypes.c_char_p,ctypes.c_char_p];self.lib.w46_create.restype=P
  self.lib.w46_reset.argtypes=[P,D,D,ctypes.c_double];self.lib.w46_diff.argtypes=[P,D,D,D]
  self.lib.w46_project.argtypes=[P,D,D,D,D];self.lib.w46_hold.argtypes=[P,D,D,D];self.lib.w46_destroy.argtypes=[P]
 def create(self,path):
  e=ctypes.create_string_buffer(2048);p=self.lib.w46_create(str(path).encode(),e)
  if not p:raise ValueError(e.value)
  return p
 def project(self,p,state,H,u):
  u=np.ascontiguousarray(u);info=np.empty(16)
  self.lib.w46_project(p,np.ascontiguousarray(state),np.ascontiguousarray(H),u,info)
  return u,info
def read_policy(path):
 z=np.fromfile(path,dtype='<f8');h=z[:16];T,ns,nu,nx,nr=map(int,h[3:8]);i=16;d={}
 for name,shape in [('states',(T,ns)),('actions',(T-1,nu)),('times',(T,)),('residual',(T,nr)),('K',(T,nu,nx)),('H',(T,nu,nu))]:
  n=int(np.prod(shape));d[name]=z[i:i+n].reshape(shape);i+=n
 d['scale']=h[12];d['representation']=int(h[13]);return d
def main():
 start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=False)
 report=dict(status='running',pid=os.getpid(),deadline_unix_s=None,wall_deadline_enabled=False,started_unix_s=time.time(),
  host_affinity=sorted(os.sched_getaffinity(0)),scenario_count=24,new_PPO_samples=0,new_PPO_updates=0,
  action_execution='GPU FP64 interpolation inputs/quaternion tangent/K product; exact frozen native r29-qp2 H projection on host. Complete mixed execution, not all-device H-QP.',
  precision='FP64 actions explicitly cast when writing FP32 MJWarp ctrl; GPU dynamics FP32 versus native3.2.3 FP64.',
  full_native_cost_implemented_on_gpu=False,equivalent_native_ranking=False,world_steps=0,native_world_steps=0,warmup_world_steps=0)
 write(SMALL/'gpu.json',report)
 try:
  bridge=Bridge(ROOT/'work/w46-implementation/native-bridge.so')
  worlds=[];policies=[];states=[];times=[];userdatas=[];tags=[];models=[];expected=[]
  for label in ['A148','A153','B155','B158']:
   f=json.loads((CPU/label/'input.json').read_text());st=dict(np.load(CPU/label/'input-state.npz'));delta=np.load(CPU/label/'perturbation.npz')['delta']
   tree=ET.parse(ROOT/f['original_model_path']);tree.getroot().remove(tree.find('sensor'))
   tree.find('option').set('timestep','.0002');tree.find('option').set('integrator','implicitfast')
   xml=OUT/(label+'-model.xml');tree.write(xml,encoding='unicode');m=mujoco.MjModel.from_xml_path(str(xml))
   m.jnt_solimp[:,0]=0;m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_WARMSTART);models.append(m)
   for name in ['original-baseline.bin','original-policy.bin']:
    pol=read_policy(CPU/label/name)
    if pol['representation']!=1:raise ValueError('Finite action implementation requires actual linear policy representation')
    for sign in [0,1,-1]:
     q=st['qpos'].copy();mujoco.mj_integratePos(m,q,delta[:18],float(sign));x=np.r_[q,st['qvel']+sign*delta[18:]]
     p=bridge.create(xml);bridge.lib.w46_reset(p,x,np.ascontiguousarray(st['userdata']),float(st['time']))
     worlds.append(p);policies.append(pol);states.append(x);times.append(float(st['time']));userdatas.append(st['userdata'])
     tags.append(dict(label=label,policy=name,sign=sign))
  for m2 in models[1:]:
   for key in ['geom_solimp','pair_solimp','jnt_solimp','actuator_gainprm','actuator_biasprm','actuator_forcerange','body_mass','geom_friction']:
    if not np.array_equal(getattr(models[0],key),getattr(m2,key)):raise ValueError('Fixture physical models differ: '+key)
  m=models[0];states=np.asarray(states);initial=states.copy();times=np.asarray(times);userdatas=np.asarray(userdatas)
  bounds=wp.array(m.actuator_ctrlrange,dtype=wp.float64);scale=wp.array(np.asarray([p['scale'] for p in policies]),dtype=wp.float64)
  def action(current,ts):
   ff=[];ref=[];gain=[];metrics=[];cpu=[]
   for x,t,p in zip(current,ts,policies):
    u,fb,dx,k=raw_policy(t,x,p['times'],p['states'],p['actions'],p['K'],p['scale'])
    r=interp(t,p['times'],p['states']);r[3:7]/=np.linalg.norm(r[3:7])
    ff.append(u);ref.append(r);gain.append(k);metrics.append(interp(t,p['times'],p['H'][:-1]))
    cpu.append(np.clip(u+fb,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]))
   dev=wp.zeros((24,12),dtype=wp.float64)
   wp.launch(feedback,dim=(24,12),inputs=[wp.array(current,dtype=wp.float64),wp.array(np.asarray(ref),dtype=wp.float64),
    wp.array(np.asarray(ff),dtype=wp.float64),wp.array(np.asarray(gain),dtype=wp.float64),scale,bounds,dev])
   raw=dev.numpy();ug=[];un=[];ig=[];inn=[]
   for i in range(24):
    u,q=bridge.project(worlds[i],current[i],metrics[i],raw[i]);v,r=bridge.project(worlds[i],current[i],metrics[i],cpu[i])
    ug.append(u);un.append(v);ig.append(q);inn.append(r)
   return np.asarray(ug),np.asarray(un),np.asarray(ig),np.asarray(inn),ma(raw-np.asarray(cpu))
  a,b,ai,bi,rawerr=action(states,times);wp.synchronize()
  algebra=dict(raw_center_max_error_rad=rawerr,held_action_max_error_rad=ma(a-b),
    pd_equivalent_error_Nm=60*ma(a-b),active_equal=bool(np.array_equal(ai[:,4:],bi[:,4:])),
    all_QP_converged=bool(np.all(ai[:,0]==1)&np.all(bi[:,0]==1)),max_kkt_scaled=float(max(ai[:,2].max(),bi[:,2].max())))
  report['same_state_action']=algebra
  np.savez_compressed(OUT/'action-algebra.npz',states=states,gpu_action=a,native_action=b,gpu_info=ai,native_info=bi)
  # Exact Euclidean joint-reference recentering counterexample: raw feedback changes, held action does not.
  recenter=[]
  for i,(x,t,p) in enumerate(zip(states,times,policies)):
   shift=np.load(CPU/tags[i]['label']/'perturbation.npz')['delta'][6:18]
   q=dict(p);q['states']=p['states'].copy();q['states'][:,7:19]+=shift
   q['actions']=p['actions']+p['scale']*np.einsum('tij,j->ti',p['K'][:-1,:,6:18],shift)
   u,fb,_,_=raw_policy(t,x,p['times'],p['states'],p['actions'],p['K'],p['scale'])
   v,fb2,_,_=raw_policy(t,x,q['times'],q['states'],q['actions'],q['K'],q['scale'])
   H=interp(t,p['times'],p['H'][:-1])
   u,inf=bridge.project(worlds[i],x,H,np.clip(u+fb,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]))
   v,inf2=bridge.project(worlds[i],x,H,np.clip(v+fb2,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]))
   recenter.append(dict(**tags[i],held_action_difference_rad=ma(u-v),raw_feedback_difference_rad=ma(fb-fb2)))
  report['reference_representation']=recenter
  report['action_setup_s']=time.perf_counter()-start;write(SMALL/'gpu.json',report)
  if algebra['held_action_max_error_rad']>1e-9 or not algebra['active_equal'] or not algebra['all_QP_converged']:
   report.update(status='finite_action_semantics_not_equivalent',full_350ms='not_started');return
  wm=mjw.put_model(m);wm.opt.tolerance.fill_(float(m.opt.tolerance))
  d=mjw.make_data(m,nworld=24,nconmax=128,njmax=768)
  feet=wp.array([m.geom(n).id for n in ['FL','RL','FR','RR']],dtype=int)
  stiff=float(m.numeric('r10_local_contact').data[0]);floor=float(m.numeric('s1_flat_ground_height').data[0])
  def cf(wm,d):wp.launch(filter_contact,dim=d.naconmax,inputs=[d.contact.geom,d.contact.worldid,d.nacon,d.geom_xpos,feet],outputs=[d.contact.dist])
  def pf(wm,d):wp.launch(material_force,dim=(24,18),inputs=[d.geom_xpos,wm.geom_size,wm.geom_bodyid,feet,d.userdata,d.qvel,wm.body_parentid,wm.body_rootid,wm.dof_bodyid,wm.body_isdofancestor,d.subtree_com,d.cdof,stiff,floor],outputs=[d.qfrc_passive])
  wm.callback.contactfilter=cf;wm.callback.passive=pf
  trunk=m.body('trunk').id;floorid=m.geom('floor').id
  ids=[g for g in range(m.ngeom) if m.geom_bodyid[g]==trunk and ((m.geom_contype[g]&m.geom_conaffinity[floorid]) or (m.geom_contype[floorid]&m.geom_conaffinity[g]))]
  trunkdev=wp.array(ids,dtype=int);kind=wp.array(m.geom_type,dtype=int);margins=wp.array(np.maximum(m.geom_margin,m.geom_margin[floorid]).astype(np.float32),dtype=float)
  risk=wp.zeros(24,dtype=float)
  def load():
   for name,v in [('qpos',initial[:,:19]),('qvel',initial[:,19:]),('userdata',userdatas),('time',times),('ctrl',a.astype(np.float32))]:
    wp.copy(getattr(d,name),wp.array(v,dtype=getattr(d,name).dtype))
   d.qacc_warmstart.zero_();risk.zero_()
  def step():
   mjw.step(wm,d);wp.launch(risks,dim=24,inputs=[d.geom_xpos,d.geom_xmat,wm.geom_size,kind,trunkdev,margins,floor,risk])
  t=time.perf_counter();load();step();wp.synchronize()
  with wp.ScopedCapture() as cap:step()
  wp.synchronize();report['compile_capture_s']=time.perf_counter()-t;report['warmup_world_steps']=48
  load();wp.synchronize();native=initial.copy();gpu=initial.copy()
  alln=[native.copy()];allg=[gpu.copy()];allun=[];allug=[];nrisk=np.zeros(24)
  disagree=[];maptime=0.;devtime=0.;native_time=0.;download_time=0.
  for k in range(175):
   t=time.perf_counter();ug,_,ig,_,_=action(gpu,times+.002*k)
   _,un,_,inn,_=action(native,times+.002*k);maptime+=time.perf_counter()-t
   if not np.all(ig[:,0]) or not np.all(inn[:,0]):
    disagree.append(dict(node=k,reason='QP_failure'));report['stop_reason']='same_complete_policy_projection_failure';break
   allun.append(un.copy());allug.append(ug.copy())
   wp.copy(d.ctrl,wp.array(ug.astype(np.float32),dtype=float))
   t=time.perf_counter()
   for j in range(10):wp.capture_launch(cap.graph)
   wp.synchronize();devtime+=time.perf_counter()-t
   t=time.perf_counter();gpu=np.c_[d.qpos.numpy(),d.qvel.numpy()].astype(np.float64);grisk=risk.numpy().astype(float);download_time+=time.perf_counter()-t
   t=time.perf_counter()
   for i in range(24):
    rec=np.empty((10,13));bridge.lib.w46_hold(worlds[i],np.ascontiguousarray(un[i]),native[i],rec);nrisk[i]=max(nrisk[i],rec[:,12].max())
   native_time+=time.perf_counter()-t;alln.append(native.copy());allg.append(gpu.copy())
   report['world_steps']+=240;report['native_world_steps']+=240
   if k==9:
    mismatch=(grisk>0)!=(nrisk>0)
    short=dict(state_component_max_error=ma(gpu-native),joint_position_max_error_rad=ma(gpu[:,7:19]-native[:,7:19]),
     velocity_max_error=ma(gpu[:,19:]-native[:,19:]),held_command_max_error_rad=ma(np.asarray(allug)-np.asarray(allun)),
     trunk_constraint_mismatches=np.flatnonzero(mismatch).tolist(),native_trunk=nrisk.tolist(),gpu_trunk=grisk.tolist(),
     action_map_verified=algebra,completed_s=.02)
    report['short20ms']=short;write(SMALL/'gpu.json',report)
    if np.any(mismatch):
     report.update(status='finite_constraint_disagreement',full_350ms='not_started');break
    report['full_350ms']='same_running_batch_continuation_no_restart'
  else:report.update(status='completed_finite_full_policy_execution',full_350ms='completed')
  report.update(timing=dict(action_mapping_and_host_projection_s=maptime,device_advance_wall_s=devtime,
    native_confirmation_s=native_time,download_s=download_time),constraint_disagreements=disagree,
    final_state_component_max_error=ma(gpu-native),final_native_trunk=nrisk.tolist(),final_gpu_trunk=grisk.tolist(),
    throughput_scope='Host coupled complete policy diagnostic with exact native H projection, not resident all-device MPC or fair optimized speed benchmark.')
  np.savez_compressed(OUT/'paired-states-actions.npz',native_states=np.asarray(alln),gpu_states=np.asarray(allg),
    native_actions=np.asarray(allun),gpu_actions=np.asarray(allug),native_trunk=nrisk,gpu_trunk=grisk)
  write(OUT/'scenarios.json',tags)
 except Exception as e:report.update(status='technical_failure',first_error=repr(e),traceback=traceback.format_exc())
 finally:
  report.update(ended_unix_s=time.time(),wall_s=time.perf_counter()-start)
  write(SMALL/'gpu.json',report);write(OUT/'result.json',report)
if __name__=='__main__':main()
