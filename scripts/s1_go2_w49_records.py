"""W49 observation-only pre-call alignment and bounded, hash-deduplicated events."""
import hashlib,json,os,time,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
from s1_go2_r29_records import R29Records
def digest_arrays(arrays):
 h=hashlib.sha256()
 for name,a in sorted(arrays.items()):
  a=np.asarray(a);h.update(name.encode());h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
 return h.hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def write(path,x):
 path.write_text(json.dumps(plain(x),indent=2,allow_nan=False)+'\n')
def policy_fields(T,ns,nu,nx,nr):
 return [('reference_states',(T,ns)),('reference_actions',(T-1,nu)),('reference_times',(T,)),
  ('reference_residual',(T,nr)),('K',(T,nu,nx)),('H',(T,nu,nu)),
  ('base_Qu',(T,nu)),('base_Qxu',(T,nx,nu)),('base_active',(T,nu))]
class W49Records(R29Records):
 def __init__(self,out,model,config,fingerprint):
  super().__init__(out,model)
  self.controller='S5-r37';self.config=config;self.fingerprint=fingerprint
  self.cost_sensors=ET.parse(out/'model/G1-project-go2.xml').findall('.//sensor/user')
  self.call_offset=0;self.call_ring={};self.current=None;self.deployed=None;self.previous_action=None;self.previous_torque=None
  self.windows=[];self.event_calls={};self.object_manifest={};self.pending=None;self.prepared=False;self.calls=0
  self.object_dir=out/'w49-objects';self.object_dir.mkdir()
  self.alignment=(out/'w49-planner-boundaries.jsonl').open('w')
  self.execution=(out/'w49-held-clock.jsonl').open('w')
  self.schema=dict(schema='w49_pre_call_prediction_events_v1',actual_capture='before agent.planner_step',
   native_pre_context='before OptimizePolicy mutates epoch or policy; after explicit offline import for finite fixtures',
   prediction='previous actually applied complete policy accepted rollout, never reference_states or new solve',
   time_match='nearest recorded node only if abs(node_time-origin)<=1e-9 seconds; no interpolation',
   missing='unavailable with reason; no fabricated state/error',
   position_units=dict(translation='m',orientation='rad',joints='rad'),
   velocity_units=dict(linear='m/s',angular='rad/s',joints='rad/s'),
   object_hash='SHA256 over sorted array name,shape,dtype,bytes; immutable NPZ separately hashed',
   pre_window_s=.1,post_window_s=.2,thresholds_are_diagnostic_only=True,
   native_transport='w49-precall.bin and r29-current-plan.bin are overwritten transport, event objects retained',
   controller_build='S5-r37',original_config=config['r36_state_difference'],delta_policy=config['w49_delta_policy'])
  write(out/'w49-recording-schema.json',self.schema)
 def before_call(self,t,data,mission,joint):
  actual=np.r_[data.qpos,data.qvel,data.act].copy()
  meta=dict(time_s=float(t),mission_phase=mission.phase,command_velocity=np.asarray(mission.v).tolist(),
   command_omega=np.asarray(mission.w).tolist(),body_target=np.asarray(mission.body_target).tolist(),
   heading=float(mission.heading),cap=float(joint.cap) if joint is not None else None,
   execution_fingerprint=self.fingerprint,case_key=self.config.get('case_key'),
   task_configuration_sha256=hashlib.sha256(json.dumps(plain(mission.task),sort_keys=True).encode()).hexdigest(),
   task_mode='Quadruped',risk=0.,solver_budget=dict(mechanism='K',strict=True,constant=True,max_backward_attempts=5,alpha_slots=10))
  arrays=dict(actual_state=actual,actual_mocap_pos=data.mocap_pos.copy(),actual_mocap_quat=data.mocap_quat.copy(),
   actual_userdata=data.userdata.copy(),actual_ctrl=data.ctrl.copy())
  if joint is not None and getattr(joint,'pending',None) is not None:
   arrays['actor_observation']=joint.pending['obs'].detach().cpu().numpy().copy()
   meta.update(actor_tick=int(joint.last_actor_tick),actor_observation_sha256=digest_arrays({'observation':arrays['actor_observation']}),
    observer_time_s=float(joint.observer.time) if joint.observer is not None else None,
    actor_checkpoint=self.config['r12_contract']['w49_fixed_policy']['checkpoint'],
    actor_checkpoint_sha256=self.config['r12_contract']['w49_fixed_policy']['checkpoint_sha256'])
  prev=dict(status='unavailable',reason='no_previously_applied_policy')
  if self.deployed is not None:
   d=self.deployed;times=d['accepted']['accepted_times'];node=int(np.argmin(abs(times-t)))
   prev.update(source_call=d['call']['call_id'],source_plan=d['call']['installed_plan_id'],
    source_epoch=d['call']['problem_epoch'],policy_sha256=d['policy_hash'],accepted_sha256=d['accepted_hash'])
   if abs(float(times[node])-t)<=1e-9:
    x=d['accepted']['accepted_states'][node];dq=np.empty(self.model.nv)
    mujoco.mj_differentiatePos(self.model,dq,1.,x[:self.model.nq],actual[:self.model.nq])
    # This is a diagnostic error; honor the same exact identity convention.
    for j in range(self.model.njnt):
     typ=self.model.jnt_type[j]
     if typ not in (mujoco.mjtJoint.mjJNT_FREE,mujoco.mjtJoint.mjJNT_BALL):continue
     q=int(self.model.jnt_qposadr[j])+(3 if typ==mujoco.mjtJoint.mjJNT_FREE else 0)
     v=int(self.model.jnt_dofadr[j])+(3 if typ==mujoco.mjtJoint.mjJNT_FREE else 0)
     if np.array_equal(x[q:q+4],actual[q:q+4]) or np.array_equal(x[q:q+4],-actual[q:q+4]):dq[v:v+3]=0
    dv=actual[self.model.nq:self.model.nq+self.model.nv]-x[self.model.nq:self.model.nq+self.model.nv]
    prev.update(status='available',reason=None,node=node,node_time_s=float(times[node]),predicted_state=x.tolist(),
     position_tangent=dq.tolist(),velocity_difference=dv.tolist(),
     base_position_m=dq[:3].tolist(),base_orientation_rad=dq[3:6].tolist(),joint_position_rad=dq[6:].tolist(),
     base_linear_velocity_m_s=dv[:3].tolist(),base_angular_velocity_rad_s=dv[3:6].tolist(),joint_velocity_rad_s=dv[6:].tolist())
   else:prev['reason']='no_matching_recorded_node_or_beyond_horizon'
  integration=np.empty(mujoco.mj_stateSize(self.model,mujoco.mjtState.mjSTATE_INTEGRATION))
  mujoco.mj_getState(self.model,data,integration,mujoco.mjtState.mjSTATE_INTEGRATION)
  arrays['plant_mjSTATE_INTEGRATION']=integration
  arrays['plant_geom_contype']=self.model.geom_contype.copy()
  arrays['plant_geom_conaffinity']=self.model.geom_conaffinity.copy()
  metadata={};unavailable=[]
  def capture(prefix,obj,skip):
   for key,value in vars(obj).items():
    if key in skip:continue
    name=prefix+'__'+key
    if isinstance(value,np.ndarray):arrays[name]=value.copy()
    elif value is None or isinstance(value,(str,bool,int,float,np.generic)):
     metadata[name]=plain(value)
    elif isinstance(value,(list,tuple,dict,set)):
     try:
      encoded=json.dumps(plain(value) if not isinstance(value,set) else list(value),sort_keys=True)
      metadata[name]=json.loads(encoded)
     except (TypeError,ValueError):unavailable.append(name)
    else:unavailable.append(name)
  capture('mission',mission,set())
  if joint is not None:
   capture('material',joint,{'root','contract','model','data','native_model','native_data','out','events_file','failures_file','records','cells','chunks','stream','agent','pending','ground','observer'})
   capture('ground',joint.ground,{'law','events','rate_hist','ids'})
   arrays['ground__world_keys']=np.asarray(list(joint.ground.ids.keys()),dtype=np.int64).reshape(-1,2)
   arrays['ground__world_values']=np.asarray(list(joint.ground.ids.values()),dtype=np.int64)
   if joint.observer is not None:capture('observer',joint.observer,{'law','network','model'})
  arrays['external_metadata_utf8']=np.frombuffer(json.dumps(metadata,sort_keys=True).encode(),dtype=np.uint8).copy()
  scope=dict(integration_enum=int(mujoco.mjtState.mjSTATE_INTEGRATION),integration_size=len(integration),
    numeric_arrays_captured=True,external_unserialized_fields=unavailable,
    limitations='Native optimizer caches, actor tensors/RNG and arbitrary Python objects are not an automatic mid-episode resume; fixed checkpoint/resume remain separate.')
  self.pending=dict(meta=meta,arrays=arrays,previous_prediction=prev,external_scope=scope)
 def read_pre(self):
  raw=np.fromfile(self.out/'w49-precall.bin',dtype='<f8');h=raw[:20];i=20
  ctx={'native_header':h.copy()}
  for name in ['native_state','native_mocap','native_userdata','private_gait','task_parameters','task_weights','norm_parameters','norms']:
   n=int(raw[i]);i+=1;ctx[name]=raw[i:i+n].copy();i+=n
  T,ns,nu,nx,nr=map(int,h[3:8]);p0={}
  for name,shape in policy_fields(T,ns,nu,nx,nr):
   n=int(np.prod(shape));p0[name]=raw[i:i+n].reshape(shape).copy();i+=n
  p0['semantics']=h[10:17].copy()
  ext=raw[i:i+6].copy();i+=6;n=int(ext[2])
  assert ext[0]==49
  p0['delta_header']=ext;p0['delta_times']=raw[i:i+n].copy();i+=n
  p0['delta_values']=raw[i:i+n*nu].reshape(n,nu).copy();i+=n*nu
  assert i==len(raw),'W49 precall schema mismatch'
  return h,ctx,p0
 def read_post(self):
  p=self.out/'r29-current-plan.bin'
  with p.open('rb') as f:
   h=np.fromfile(f,dtype='<f8',count=16);T,ns,nu,nx,nr=map(int,h[3:8]);accepted={};policy={}
   for name,shape in [('accepted_states',(T,ns)),('accepted_actions',(T-1,nu)),('accepted_times',(T,)),('accepted_residual',(T,nr))]:
    accepted[name]=np.fromfile(f,dtype='<f8',count=int(np.prod(shape))).reshape(shape)
   for name,shape in policy_fields(T,ns,nu,nx,nr):
    policy[name]=np.fromfile(f,dtype='<f8',count=int(np.prod(shape))).reshape(shape)
   ext=np.fromfile(f,dtype='<f8',count=6);n=int(ext[2]);assert ext[0]==49
   policy['delta_header']=ext;policy['delta_times']=np.fromfile(f,dtype='<f8',count=n)
   policy['delta_values']=np.fromfile(f,dtype='<f8',count=n*nu).reshape(n,nu)
  policy['semantics']=np.array([ext[1],h[8],h[9],h[10],h[11],h[12],h[13]])
  return h,policy,accepted
 def plan(self,t):
  h,ctx,p0=self.read_pre();post,policy,accepted=self.read_post()
  with (self.out/'planner-calls.jsonl').open() as f:
   f.seek(self.call_offset);rows=[json.loads(x) for x in f];self.call_offset=f.tell()
  call=rows[-1]
  assert int(h[1])==call['call_id'] and int(post[1])==call['installed_plan_id']
  pending=self.pending
  assert pending is not None and float(h[2])==pending['meta']['time_s']
  assert np.array_equal(ctx['native_state'],pending['arrays']['actual_state'])
  ctx.update(pending['arrays'])
  pending['meta']['external_snapshot_scope']=pending['external_scope']
  refpath=self.out/f"r29-current-plan.bin.r35-reference-{call['r35_reference_epoch']}.bin"
  reference={'r35_reference':np.fromfile(refpath,dtype='<f8')}
  ph=digest_arrays(policy);ah=digest_arrays(accepted);p0h=digest_arrays(p0);ch=digest_arrays(ctx);rh=digest_arrays(reference)
  compact=dict(schema=self.schema['schema'],call_id=call['call_id'],origin_s=float(t),actual_state=pending['arrays']['actual_state'].tolist(),
   actual_context_sha256=ch,P0=dict(plan_id=int(h[9]),generation_epoch=int(h[17]),generation_call=int(h[18]),
   generation_time_s=float(h[19]),policy_sha256=p0h,metric_valid=bool(h[10])),
   previous_prediction=pending['previous_prediction'],installed_policy_sha256=ph,accepted_rollout_sha256=ah,
   installed_plan_id=call['installed_plan_id'],problem_epoch=call['problem_epoch'],
   r35_reference_sha256=rh,r35_reference_id=call['r35_reference_id'],
   reference_file=refpath.name,mission_actor=pending['meta'],
   current_solve=dict(J=call['deployed_current_cost'],chosen_expected=call['chosen_expected'],chosen_actual=call['chosen_actual'],
    risk=call['deployed_current_trunk_violation'],candidate_source=call['candidate_source'],stop_reason=call['stop_reason'],return_reason=call['return_reason']))
  # Small all-call cost decomposition, without retaining every large candidate.
  residual=accepted['accepted_residual'];jref=float(.5*np.sum(residual[:,-12:]**2)/len(residual))
  compact['current_solve'].update(Jref=jref,J0=call['deployed_current_cost']-jref)
  parts={};offset=0
  for sensor,weight in zip(self.cost_sensors,call['task_weights']):
   dim=int(sensor.attrib['dim']);usr=[float(x) for x in sensor.attrib['user'].split()];norm=int(usr[0])
   q=residual[:,offset:offset+dim];offset+=dim
   if norm==0:value=.5*np.sum(q*q,axis=1)
   elif norm==2:value=np.sqrt(np.sum(q*q,axis=1)+usr[4]**2)-usr[4]
   elif norm==6:value=np.sum(np.sqrt(q*q+usr[4]**2)-usr[4],axis=1)
   else:raise ValueError('Unrecorded native norm in W49 decomposition')
   parts[sensor.attrib['name']]=float(weight*np.mean(value))
  compact['current_solve'].update(cost_components=parts,cost_recompute_error=sum(parts.values())-call['deployed_current_cost'])

  self.alignment.write(json.dumps(compact,allow_nan=False)+'\n');self.alignment.flush()
  row=dict(origin=float(t),last_ref=float(t),call=call,compact=compact,policy=policy,accepted=accepted,
   p0=p0,context=ctx,reference=reference,policy_hash=ph,accepted_hash=ah,p0_hash=p0h,context_hash=ch,reference_hash=rh)
  self.current=row;self.call_ring[call['call_id']]=row;self.calls+=1
  self.ring[call['installed_plan_id']]={'origin':float(t),'last_ref':float(t),'data':{'H':policy['H']}}
  if t<=self.until:self.save_call(row)
  planner=self.out/'planner-records.jsonl'
  if planner.exists():
   with planner.open() as stream:
    stream.seek(self.plan_offset);plans=[json.loads(line) for line in stream];self.plan_offset=stream.tell()
   for plan in plans:
    if 'acceptance_mode' in plan and plan['acceptance_mode']!='feasible_to_feasible_cost':
     self.event(t,'trunk_restoration',mode=plan['acceptance_mode'],plan_id=plan['plan_id'])
    elif plan.get('accepted_trunk_violation',0)>0:
     self.event(t,'trunk_restoration',mode='scored_archive_restoration',candidate_source=plan['candidate_source'],plan_id=plan['plan_id'])
  self.prune(t)
 def save_object(self,identity,arrays):
  if identity in self.object_manifest:return self.object_manifest[identity]
  p=self.object_dir/(identity+'.npz');np.savez_compressed(p,**arrays)
  with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  rec=dict(file=str(p.relative_to(self.out)),content_sha256=identity,file_sha256=h,bytes=p.stat().st_size)
  self.object_manifest[identity]=rec;return rec
 def save_call(self,row):
  cid=row['call']['call_id']
  if cid in self.event_calls:return
  objects={}
  for role,key in [('policy','policy_hash'),('accepted','accepted_hash'),('p0','p0_hash'),('context','context_hash'),('reference','reference_hash')]:
   objects[role]=self.save_object(row[key],row[role])
  self.event_calls[cid]=dict(boundary=row['compact'],call_record=row['call'],objects=objects)
  write(self.out/'w49-event-manifest.json',dict(schema=self.schema['schema'],events=self.events,merged_windows=self.windows,
   calls=list(self.event_calls.values()),objects=list(self.object_manifest.values()),actual_bytes=sum(v['bytes'] for v in self.object_manifest.values())))
 def event(self,t,kind,first=False,**extra):
  if first and kind in self.seen:return
  self.seen.add(kind);start=max(0.,float(t)-.1);end=float(t)+.2
  self.events.append(dict(time_s=float(t),kind=kind,start_s=start,end_requested_s=end,**extra))
  if self.windows and start<=self.windows[-1]['end_requested_s']:
   self.windows[-1]['end_requested_s']=max(end,self.windows[-1]['end_requested_s']);self.windows[-1]['kinds'].append(kind)
  else:self.windows.append(dict(start_s=start,end_requested_s=end,kinds=[kind]))
  self.until=max(self.until,end)
  for row in self.call_ring.values():
   if row['origin']>=start or row['last_ref']>=start:self.save_call(row)
  if self.deployed is not None:self.save_call(self.deployed)
  if self.current is not None:self.save_call(self.current)
 def query(self,t,kind,action=None,qpos=None,qvel=None):
  result=super().query(t,kind,action,qpos,qvel)
  if kind=='applied_hold':
   if self.previous_action is not None:
    delta=float(np.max(abs(np.asarray(action)-self.previous_action)))
    if delta>1.:self.event(t,'first_actual_held_jump_over_1rad',first=True,jump_rad=delta,previous_time_s=float(t)-.002)
   self.previous_action=np.asarray(action).copy()
   self.deployed=self.current
   if self.deployed:self.deployed['last_ref']=float(t)
  return result
 def held(self,t,action,torque,phase,data):
  torque=np.asarray(torque).copy();knees=np.arange(2,12,3)
  if self.previous_torque is not None:
   reverse=(abs(torque[knees])>=45.43-1e-8)&(abs(self.previous_torque[knees])>=45.43-1e-8)&(torque[knees]*self.previous_torque[knees]<0)
   if reverse.any():self.event(t,'first_actual_knee_limit_reversal',first=True,joints=knees[reverse].tolist(),torque_Nm=torque.tolist(),previous_torque_Nm=self.previous_torque.tolist())
  self.previous_torque=torque
  self.execution.write(json.dumps(dict(time_s=float(t),phase=phase,plan_id=self.current['call']['installed_plan_id'],
   call_id=self.current['call']['call_id'],policy_sha256=self.current['policy_hash'],qpos=data.qpos.tolist(),
   held_target=np.asarray(action).tolist(),actuator_force_Nm=torque.tolist()))+'\n');self.execution.flush()
  if not self.prepared:
   # Required same-attempt readback after normal planning and before first held integration.
   with (self.out/'w49-planner-boundaries.jsonl').open() as f:first=json.loads(next(f))
   assert first['P0']['policy_sha256'] and first['actual_context_sha256'] and first['accepted_rollout_sha256']
   assert 'plant_mjSTATE_INTEGRATION' in self.current['context'] and 'delta_header' in self.current['policy']
   assert 'ground__world_keys' in self.current['context']
   assert first['previous_prediction']['status'] in ('available','unavailable')
   self.event(t,'first_normal_plan_before_first_held_physics',first=True)
   write(self.out/'w49-first-record-readback.json',dict(status='verified_before_first_held_physics',
    physical_time_s=float(data.time),call_id=self.current['call']['call_id'],first_boundary_call=first['call_id'],
    deadline_unix_s=None,wall_deadline_enabled=False,physics_advanced_by_recorder=0,
    schema=self.schema['schema'],controller='S5-r37',artifact=self.config['r12_contract']['w49_fixed_policy']['controller_binary_sha256'],
    initial_prediction_status=first['previous_prediction']['status'],checked_unix_s=time.time(),
    event_manifest='w49-event-manifest.json',objects=len(self.object_manifest)))
   self.prepared=True
 def prune(self,t):
  keep={x['call']['call_id'] for x in (self.current,self.deployed) if x is not None}
  for cid,row in list(self.call_ring.items()):
   if cid not in keep and row['origin']<t-.12 and row['last_ref']<t-.12:del self.call_ring[cid]
  for pid,row in list(self.ring.items()):
   if row['last_ref']<t-.12:del self.ring[pid]
 def finish(self,t,reason):
  self.event(t,'terminal',reason=reason)
  for w in self.windows:w['end_available_s']=min(w['end_requested_s'],t)
  for e in self.events:e['end_available_s']=min(e['end_requested_s'],t)
  result=dict(schema=self.schema['schema'],controller=self.controller,calls=self.calls,events=self.events,
   merged_windows=self.windows,event_calls=list(self.event_calls.values()),objects=list(self.object_manifest.values()),
   actual_bytes=sum(v['bytes'] for v in self.object_manifest.values()),pre_window_s=.1,post_window_s=.2,
   terminal_s=float(t),plans=[],complete_context_policy_manifest='w49-event-manifest.json',readback_verified=self.prepared,
   all_call_alignment='w49-planner-boundaries.jsonl',held_clock='w49-held-clock.jsonl',all_candidate_derivatives_retained=False)
  write(self.out/'w49-event-manifest.json',result);write(self.out/'r29-event-manifest.json',result)
  self.alignment.close();self.actions.close();self.execution.close();return result
