#!/usr/bin/env python3
"""W45 finite resident GPU proposal; native full policy remains the judge."""
import hashlib,json,os,subprocess,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco,warp as wp,mujoco_warp as mjw
from s1_go2_w41_warp_backend import filter_contact,material_force
ROOT=Path(__file__).resolve().parents[1]
CPU=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w45-mechanisms'
OUT=Path('/root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w45')
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w45'
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
@wp.kernel
def controls(index:wp.array[int],actions:wp.array3d[float],q:wp.array2d[float],v:wp.array2d[float],
 bounds:wp.array2d[float],force:wp.array2d[float],u:wp.array2d[float]):
 w,j=wp.tid();raw=actions[w,index[0],j]
 lo=wp.max(bounds[j,0],q[w,7+j]+(force[j,0]+5.0*v[w,6+j])/60.0)
 hi=wp.min(bounds[j,1],q[w,7+j]+(force[j,1]+5.0*v[w,6+j])/60.0)
 # Axis projection only; no K or H. This is explicitly approximate proposal.
 u[w,j]=wp.clamp(raw,lo,hi)
@wp.kernel
def save_node(index:wp.array[int],q:wp.array2d[float],v:wp.array2d[float],nodes:wp.array3d[float],offset:int):
 w,j=wp.tid();k=index[0]+offset
 if j<19:nodes[k,w,j]=q[w,j]
 else:nodes[k,w,j]=v[w,j-19]
@wp.kernel
def advance(index:wp.array[int]):
 index[0]+=1
@wp.kernel
def statistics(x:wp.array2d[wp.vec3],mat:wp.array2d[wp.mat33],sizes:wp.array2d[wp.vec3],
 kinds:wp.array[int],trunk:wp.array[int],margins:wp.array[float],floor:float,
 q:wp.array2d[float],v:wp.array2d[float],u:wp.array2d[float],force:wp.array2d[float],
 nefc:wp.array[int],nacon:wp.array[int],risk:wp.array[float],torque:wp.array[float],capacity:wp.array[int]):
 w=wp.tid();worst=float(0.0)
 for k in range(trunk.shape[0]):
  g=trunk[k];size=sizes[0,g];r=mat[w,g];extent=float(0.0)
  if kinds[g]==6:extent=wp.abs(r[2,0])*size[0]+wp.abs(r[2,1])*size[1]+wp.abs(r[2,2])*size[2]
  elif kinds[g]==3:extent=size[0]+wp.abs(r[2,2])*size[1]
  elif kinds[g]==2:extent=size[0]
  worst=wp.max(worst,margins[g]-(x[w,g][2]-extent-floor))
 risk[w]=wp.max(risk[w],worst)
 peak=float(0.0)
 for j in range(12):
  raw=60.0*(u[w,j]-q[w,7+j])-5.0*v[w,6+j]
  peak=wp.max(peak,wp.max(raw-force[j,1],force[j,0]-raw))
 torque[w]=wp.max(torque[w],peak)
 wp.atomic_max(capacity,0,nacon[0]);wp.atomic_max(capacity,1,nefc[w])
def candidates(label,dest):
 import struct
 h0=np.fromfile(CPU/label/'input-policy.bin',dtype='<f8')[:16]
 raw_base=np.fromfile(CPU/label/'baseline-policy.bin',dtype='<f8')
 raw_output=np.fromfile(CPU/label/'input-policy.bin',dtype='<f8')
 T,ns,nu,nx,nr=map(int,h0[3:8]);begin=16+T*ns
 times=raw_base[begin+(T-1)*nu:begin+(T-1)*nu+T]
 values=[raw_output.copy(),raw_base.copy()]
 for j in range(30):
  z=raw_base.copy();actions=z[begin:begin+(T-1)*nu].reshape(T-1,nu)
  group=(j//2)//3;onset=.08*((j//2)%3);sign=1 if j%2==0 else -1
  s=(times[:-1]-float(h0[2])-onset)/.16;active=(s>0)&(s<1)
  for leg in range(4):
   if group==leg or (group==4 and leg in (0,3)):
    actions[active,3*leg]+=sign*.04*np.sin(2*np.pi*s[active])
    actions[active,3*leg+1]+=sign*.04*np.sin(np.pi*s[active])
    actions[active,3*leg+2]-=sign*.08*np.sin(np.pi*s[active])
  values.append(z)
 rows=[];acts=[]
 for i,z in enumerate(values):
  path=dest/f'policy{i:02d}.bin';z.astype('<f8').tofile(path)
  rows.append(dict(slot=i,file=path.name,sha256=hashlib.sha256(z.astype('<f8').tobytes()).hexdigest(),
    identity='saved_output' if i==0 else 'saved_best_nominal_or_explicit_duplicate' if i==1 else f'P_pair_{(i-2)//2}_sign_{1 if i%2==0 else -1}'))
  acts.append(z[begin:begin+(T-1)*nu].reshape(T-1,nu))
 write(dest/'candidates.json',rows)
 reference=raw_base[16:16+T*ns].reshape(T,ns)
 return np.asarray(acts,dtype=np.float32),reference
def one(label):
 begin=time.perf_counter();dest=OUT/label;dest.mkdir(parents=True,exist_ok=False)
 f=json.loads((CPU/label/'input.json').read_text());acts,reference=candidates(label,dest)
 state=dict(np.load(CPU/label/'input-state.npz'))
 tree=ET.parse(ROOT/f['model_path']);root=tree.getroot();root.remove(root.find('sensor'))
 root.find('option').set('timestep','.0002');root.find('option').set('integrator','implicitfast')
 xml=dest/'gpu-model.xml';tree.write(xml,encoding='unicode')
 m=mujoco.MjModel.from_xml_path(str(xml))
 # r33 native geom/pair contact, only inherited joint-limit smoothing.
 m.jnt_solimp[:,0]=0;m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
 wm=mjw.put_model(m);wm.opt.tolerance.fill_(float(m.opt.tolerance))
 d=mjw.make_data(m,nworld=32,nconmax=128,njmax=768)
 feet=wp.array([m.geom(n).id for n in ['FL','RL','FR','RR']],dtype=int)
 stiff=float(m.numeric('r10_local_contact').data[0]);floor=float(m.numeric('s1_flat_ground_height').data[0])
 def cf(wm,d):wp.launch(filter_contact,dim=d.naconmax,inputs=[d.contact.geom,d.contact.worldid,d.nacon,d.geom_xpos,feet],outputs=[d.contact.dist])
 def pf(wm,d):wp.launch(material_force,dim=(32,18),inputs=[d.geom_xpos,wm.geom_size,wm.geom_bodyid,feet,d.userdata,d.qvel,wm.body_parentid,wm.body_rootid,wm.dof_bodyid,wm.body_isdofancestor,d.subtree_com,d.cdof,stiff,floor],outputs=[d.qfrc_passive])
 wm.callback.contactfilter=cf;wm.callback.passive=pf
 trunk=m.body('trunk').id;floorid=m.geom('floor').id
 ids=[g for g in range(m.ngeom) if m.geom_bodyid[g]==trunk and
      ((m.geom_contype[g]&m.geom_conaffinity[floorid]) or (m.geom_contype[floorid]&m.geom_conaffinity[g]))]
 if any(m.geom_type[g] not in [2,3,6] for g in ids):raise ValueError('Unsupported trunk geometry in declared proposal')
 actions=wp.array(acts,dtype=float);idx=wp.zeros(1,dtype=int);nodes=wp.zeros((176,32,37),dtype=float)
 force=wp.array(m.actuator_forcerange.astype(np.float32),dtype=float);bounds=wp.array(m.actuator_ctrlrange.astype(np.float32),dtype=float)
 trunkdev=wp.array(ids,dtype=int);kinds=wp.array(m.geom_type,dtype=int)
 margins=wp.array(np.maximum(m.geom_margin,m.geom_margin[floorid]).astype(np.float32),dtype=float)
 risk=wp.zeros(32,dtype=float);torque=wp.zeros(32,dtype=float);capacity=wp.zeros(2,dtype=int)
 initial={}
 for name,array in [('qpos',np.tile(state['qpos'],(32,1))),('qvel',np.tile(state['qvel'],(32,1))),
   ('ctrl',acts[:,0,:]),('userdata',np.tile(state['userdata'],(32,1))),('time',np.full(32,state['time'])),
   ('mocap_pos',np.tile(state['mocap_pos'],(32,1,1))),('mocap_quat',np.tile(state['mocap_quat'],(32,1,1)))]:
  initial[name]=wp.array(array,dtype=getattr(d,name).dtype)
 def load():
  for name,array in initial.items():wp.copy(getattr(d,name),array)
  d.qacc_warmstart.zero_();idx.zero_();risk.zero_();torque.zero_();capacity.zero_()
 def hold():
  wp.launch(controls,dim=(32,12),inputs=[idx,actions,d.qpos,d.qvel,bounds,force,d.ctrl])
  for _ in range(10):
   mjw.step(wm,d)
   wp.launch(statistics,dim=32,inputs=[d.geom_xpos,d.geom_xmat,wm.geom_size,kinds,trunkdev,margins,floor,
      d.qpos,d.qvel,d.ctrl,force,d.nefc,d.nacon,risk,torque,capacity])
  wp.launch(save_node,dim=(32,37),inputs=[idx,d.qpos,d.qvel,nodes,1])
  wp.launch(advance,dim=1,inputs=[idx])
 setup=time.perf_counter()-begin;compile_start=time.perf_counter()
 load();hold();wp.synchronize()
 with wp.ScopedCapture() as cap:hold()
 compile_s=time.perf_counter()-compile_start
 upload_start=time.perf_counter();load()
 wp.launch(save_node,dim=(32,37),inputs=[idx,d.qpos,d.qvel,nodes,0]);wp.synchronize()
 upload_s=time.perf_counter()-upload_start
 device_start=wp.Event(enable_timing=True);device_end=wp.Event(enable_timing=True)
 wp.record_event(device_start);start=time.perf_counter()
 for _ in range(175):wp.capture_launch(cap.graph)
 wp.record_event(device_end);wp.synchronize()
 advance_s=time.perf_counter()-start;device_ms=wp.get_event_elapsed_time(device_start,device_end)
 download=time.perf_counter()
 states=nodes.numpy();v=risk.numpy().astype(float);excess=torque.numpy();peaks=capacity.numpy()
 download_s=time.perf_counter()-download
 np.savez_compressed(dest/'gpu-rollout.npz',states=states,trunk_violation=v,hold_torque_excess=excess,capacity=peaks,actions=acts)
 if peaks[0]>d.naconmax or peaks[1]>d.njmax:raise RuntimeError('GPU contact/constraint capacity overflow')
 if not np.isfinite(states).all():raise RuntimeError('Nonfinite GPU trajectory')
 # Deliberately approximate score: dimensionless tracking of saved best nominal
 # reference, using fixed observation scales, not the original native objective.
 scales=np.r_[[3,3,.3],[1]*4,[1]*12,[.25]*3,[1]*3,[10]*12]
 proxy=np.mean(np.sum(((states-reference[:,None,:])/scales)**2,axis=2),axis=0)
 ranking=sorted(range(32),key=lambda i:(v[i],proxy[i],i));top=ranking[:4]
 order=top+[i for i in [0,1] if i not in top]+[i for i in range(32) if i not in set(top+[0,1])]
 ranking_record=dict(ranking=ranking,top4=top,confirmation_order=order,trunk_violation=v.tolist(),proxy_score=proxy.tolist(),
  semantics='approximate_proposal_forward; no K/H, axis projection instead of H projection; FP32 MuJoCo4; reference tracking proxy is not native J',
  timing=dict(setup_candidate_allocation_s=setup,compile_and_capture_s=compile_s,batch_upload_s=upload_s,
  advance_wall_s=advance_s,device_ms=device_ms,download_s=download_s,preconfirmation_s=time.perf_counter()-begin),
  world_steps=56000,capacity_peak=peaks.tolist(),capacity_limits=[d.naconmax,d.njmax],
  source_model_sha256=hashlib.sha256((ROOT/f['model_path']).read_bytes()).hexdigest(),native_geom_pair_solimp_preserved=True,
  material_proxy='Frozen causal support anchors, existing contactfilter/passive proxy; not full TM02 historical material evolution.',
  synchronization='One timed upload and terminal download per full1750step evaluation; warm compilation load accounted separately.',
  warm_compile_extra_world_steps=320)
 write(dest/'gpu-ranking.json',ranking_record)
 native_start=time.perf_counter()
 command=[str(ROOT/'work/envs/d57/bin/python'),'-u',str(ROOT/'scripts/s1_go2_w45_native_batch.py'),'--label',label,'--directory',str(dest)]
 env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'work/upstreams/mujoco_mpc/python'))
 with (dest/'native.log').open('w') as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 native=json.loads((dest/'native-results.json').read_text());rows={r['slot']:r for r in native['rows']}
 def key(i):return (bool(rows[i]['failure']),rows[i]['trunk_violation'],rows[i]['native_J'],i)
 native_order=sorted(range(32),key=key);chosen=min(top,key=key);best=native_order[0]
 recall=len(set(top)&set(native_order[:4]))/4
 false=[i for i in range(32) if v[i]==0 and (rows[i]['failure'] or rows[i]['trunk_violation']>0)]
 native_startup=max(0,time.perf_counter()-native_start-native['total_native_s'])
 result=dict(label=label,status='completed',gpu=ranking_record,native=native,native_top4=native_order[:4],
  top4_recall=recall,native_best_recalled=best in top,false_feasible_slots=false,
  constraint_misses=dict(trunk_or_failed_native=len(false),denominator=32,
    note='GPU has no full H projection; native failure is retained, not labeled feasible.'),
  chosen_slot=chosen,native_best_slot=best,regret_v=rows[chosen]['trunk_violation']-rows[best]['trunk_violation'],
  regret_J=rows[chosen]['native_J']-rows[best]['native_J'] if rows[chosen]['trunk_violation']==rows[best]['trunk_violation'] else None,
  native_baselines=[rows[0],rows[1]],end_to_end_top4_confirmation_s=ranking_record['timing']['preconfirmation_s']+native_startup+native['top4_and_baseline_confirmation_s'],
  native32_total_s=native['total_native_s']+native_startup,all_research_wall_s=time.perf_counter()-begin)
 write(dest/'result.json',result);return result
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 report=dict(status='running',started_unix_s=time.time(),pid=os.getpid(),deadline_unix_s=None,host_affinity=sorted(os.sched_getaffinity(0)),
  finite_problem_count=4,results=[],new_PPO_samples=0,new_PPO_updates=0)
 write(SMALL/'gpu.json',report)
 try:
  for label in ['A158','A163','B153','B173']:
   report['current_problem']=label;write(SMALL/'gpu.json',report)
   report['results'].append(one(label));write(SMALL/'gpu.json',report)
  report['status']='completed'
 except Exception as e:
  report.update(status='technical_failure',first_error=repr(e),traceback=traceback.format_exc())
 finally:
  report['ended_unix_s']=time.time();write(SMALL/'gpu.json',report);write(OUT/'result.json',report)
 from publish_s1_go2_w45 import compact
 compact([SMALL/'gpu.json'],'Record W45 finite GPU outcome before original delivery')
 # Publication is serialized with the CPU tree; only server academic proxy.
 command=[str(ROOT/'work/envs/d57/bin/python'),'scripts/publish_s1_go2_w45.py','--name','gpu','--source',str(OUT)]
 subprocess.run(command,cwd=ROOT,check=True)
if __name__=='__main__':main()
