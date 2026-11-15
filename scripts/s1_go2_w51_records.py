"""W51 actual-path recording; inherited W50 binary schema, explicit B/S/Q roles."""
import json
from pathlib import Path
import numpy as np
from s1_go2_w50_records import W50Records,digest_arrays,write
from w50_policy_io import read_policy
class W51Records(W50Records):
 def __init__(self,out,model,config,fingerprint):
  super().__init__(out,model,config,fingerprint)
  self.seed_log=(out/'w51-seeds.jsonl').open('w');self.seed_records=[]
  self.schema.update(experiment='W51',arm=self.fixed_policy['arm'],qualification_usable=config['r11_stage']['level']!='D8',
    motion_seed=config.get('w51_motion_seed'),candidate_source='r33 original same-problem metadata')
  write(out/'w51-recording-schema.json',self.schema)
 def plan(self,t):
  super().plan(t)
  self.r33_plan(t);self.search_plan(t)
  call=self.current['call'];cid=call['call_id']
  record=dict(call_id=cid,time_s=float(t),arm=self.fixed_policy['arm'],controller=self.controller,
    installed_source=call['candidate_source'],installed_policy_sha256=self.current['policy_hash'],
    S_status=call.get('w51_seed_status','D0_refresh_only'),Q_is_S=call.get('w51_Q_is_S',False),
    native_seed_rollouts=call.get('w51_seed_rollouts',0),expected=None,ratio=None)
  path=self.out/'w51-motion.json'
  if path.exists():
   meta=json.loads(path.read_text())
   if meta['call_id']==cid:
    record['motion']=meta;record['objects']={}
    for role in ('B','S','Q'):
     p=self.out/f'w51-motion-{role}.bin'
     if p.exists() and (role!='S' or meta['native_seed_rollouts']):
      _,obj=read_policy(p);identity=digest_arrays(obj);record['objects'][role]=self.save_object(identity,obj)
    p=self.out/'w51-motion-kinematics.bin'
    obj={'kinematics_payload':np.fromfile(p,dtype='<f8')};identity=digest_arrays(obj)
    record['objects']['IK']=self.save_object(identity,obj)
  self.seed_log.write(json.dumps(record,allow_nan=False)+'\n');self.seed_log.flush()
  self.seed_records.append(record)
  if t<=self.until:
   if cid in self.r33_ring:self.save_r33(cid)
   if cid in self.search_ring:self.save_search(cid)
  for ring in (self.r33_ring,self.search_ring):
   for key,row in list(ring.items()):
    if row['origin']<t-.12:del ring[key]
 def save_call(self,row):
  super().save_call(row)
  cid=row['call']['call_id']
  if cid in self.r33_ring:self.save_r33(cid)
  if cid in self.search_ring:self.save_search(cid)
 def finish(self,t,reason):
  result=super().finish(t,reason)
  self.seed_log.close()
  result.update(experiment='W51',arm=self.fixed_policy['arm'],
   candidates=list(self.r33_saved.values()),searches=list(self.search_saved.values()),
   seeds='w51-seeds.jsonl',qualification_usable=self.config['r11_stage']['level']!='D8')
  write(self.out/'w51-event-manifest.json',result)
  return result
