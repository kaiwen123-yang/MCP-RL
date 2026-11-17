"""Record restored N's actual calls/holds; native files contain full policies.

Native candidates are never recomputed to fill a recording gap. Each policy file
contains the reference state/action, complete K, action improvement and the
accepted trajectory. Ordinary rejected full arrays are intentionally absent.
"""
import json,time
from pathlib import Path
import numpy as np

def read_policy(path):
    """Read the native little-endian record without native/model computation."""
    with Path(path).open('rb') as f:
        h=np.fromfile(f,dtype='<i8',count=10);info=np.fromfile(f,dtype='<f8',count=4)
        magic,version,call,H,nx,ndx,nu,nr,ntrace,representation=map(int,h)
        if (magic,version)!=(53,1):raise ValueError('W53 N policy record version')
        dims=[('reference_times',(H,)),('reference_states',(H,nx)),('reference_actions',(H,nu)),
              ('feedback_gain',(H,nu,ndx)),('action_improvement',(H,nu)),('accepted_times',(H,)),
              ('accepted_states',(H,nx)),('accepted_actions',(H,nu)),('accepted_residual',(H,nr)),
              ('accepted_costs',(H,)),('accepted_trace',(H,ntrace,3))]
        out={k:np.fromfile(f,dtype='<f8',count=int(np.prod(shape))).reshape(shape) for k,shape in dims}
    out.update(call_id=call,time_s=info[0],feedback_scaling=info[1],reference_J=info[2],accepted_J=info[3],representation=representation)
    return out

class W53NRecords:
    def __init__(self,out,model,c,fingerprint):
        self.out=Path(out);self.native=self.out/'w53-n-native';self.native.mkdir(exist_ok=True)
        self.events=(self.out/'w53-n-events.jsonl').open('w',buffering=1)
        self.holds=(self.out/'w53-n-held.jsonl').open('w',buffering=1)
        self.context=(self.out/'w53-n-context.jsonl').open('w',buffering=1)
        self.failure=None;self.call_id=0;self.policy_call_id=0;self.last_native=0;self.calls=[];self.seen=set()
        self.model=model;self.fingerprint=fingerprint;self.controller='S5-r42';self.start=time.perf_counter()
        self.io_wall=0.;self.io_cpu=0.;self.control_samples=0;self.query_count=0
        self.qadr=model.jnt_qposadr[model.actuator_trnid[:,0]];self.vadr=model.jnt_dofadr[model.actuator_trnid[:,0]]
    def _write(self,f,v):
        w=time.perf_counter();cpu=time.process_time()
        f.write(json.dumps(v,separators=(',',':'),allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')
        self.io_wall+=time.perf_counter()-w;self.io_cpu+=time.process_time()-cpu
    def before_call(self,t,data,mission,joint):
        self._write(self.context,dict(time_s=t,call_id=self.call_id+1,qpos=data.qpos.copy(),qvel=data.qvel.copy(),
          mocap_pos=data.mocap_pos.copy(),mocap_quat=data.mocap_quat.copy(),userdata=data.userdata.copy(),
          mission_phase=mission.phase,command_velocity=mission.v.copy(),command_omega=mission.w.copy(),
          cap=float(joint.cap) if joint is not None else None))
    def plan(self,t):
        path=self.native/'calls.jsonl'
        with path.open() as f:
            f.seek(self.last_native)
            rows=[json.loads(x) for x in f if x.strip()];self.last_native=f.tell()
        self.calls.extend(rows)
        row=rows[-1];self.call_id=row['call_id'];self.policy_call_id=row['policy_call_id']
    def query(self,t,kind,action=None,qpos=None,qvel=None):
        self.query_count+=1
        if kind=='startup_probe':self._write(self.events,dict(time_s=t,kind=kind,call_id=self.call_id,policy_call_id=self.policy_call_id))
    def progress(self,t,qpos,mission_v,contact):
        if qpos[2]<.25:self.event(t,'first_base_below_0.25m',first=True,base_z_m=float(qpos[2]))
        (self.out/'progress.json').write_text(json.dumps(dict(time_s=t,call_id=self.call_id,policy_call_id=self.policy_call_id,
          qpos=qpos.tolist(),command_velocity=mission_v.tolist(),contact=contact.tolist(),controller='S5-r42',
          deadline=None,wall_deadline_enabled=False))+'\n') if self.control_samples%250==0 else None
    def held(self,t,action,actual_torque,phase,data):
        self.control_samples+=1
        self._write(self.holds,dict(time_s=t,call_id=self.call_id,policy_call_id=self.policy_call_id,phase=phase,
          action=action,qpos=data.qpos.copy(),qvel=data.qvel.copy(),actuator_force=actual_torque.copy(),
          requested_PD=60*(action-data.qpos[self.qadr])-5*data.qvel[self.vadr]))
    def event(self,t,kind,first=False,**fields):
        if first and kind in self.seen:return
        self.seen.add(kind);self._write(self.events,dict(time_s=t,kind=kind,call_id=self.call_id,policy_call_id=self.policy_call_id,**fields))
        if kind in ('phase_transition','first_material_load','first_base_below_0.25m','contact_mode_mismatch'):
            (self.native/'full-ring.request').write_text(kind+'\n')
    def finish(self,t,reason):
        begin=time.perf_counter();cpu=time.process_time()
        self.event(t,'finish',reason=reason)
        for f in (self.events,self.holds,self.context):f.flush();f.close()
        files=list(self.native.glob('*.bin'))
        result=dict(controller='S5-r42',calls=self.call_id,installed_policies=sum(x['installed'] for x in self.calls),
          actual_held_samples=self.control_samples,query_count=self.query_count,
          full_rollouts=sum(x['full_rollouts'] for x in self.calls),transition_fd_calls=sum(x['transition_fd_calls'] for x in self.calls),
          risk_fd_calls=0,native_record_files=len(files),native_record_bytes=sum(p.stat().st_size for p in files),
          full_candidate_calls=[x['call_id'] for x in self.calls if x['full_candidates']],
          native_stage_wall_s={k:sum(x['wall_s'][k] for x in self.calls) for k in ('nominal','derivative','cost','backward','rollouts','policy_update')},
          native_stage_cpu_s=np.sum([x['cpu_s'] for x in self.calls],axis=0).tolist() if self.calls else [],
          native_serialization_write_wall_s=sum(x['record_wall_s'] for x in self.calls),
          native_serialization_write_cpu_s=sum(x['record_cpu_s'] for x in self.calls),
          python_record_write_wall_s=self.io_wall,python_record_write_cpu_s=self.io_cpu,
          final_flush_wall_s=time.perf_counter()-begin,final_flush_cpu_s=time.process_time()-cpu,
          rejected_full_array_boundary='Only key-event calls saved; cannot infer normal rejected trajectories for other calls.',
          installed_full_policy_and_actual_accepted_trajectory=True,failure=self.failure)
        (self.out/'w53-n-recording.json').write_text(json.dumps(result,indent=2)+'\n');return result
