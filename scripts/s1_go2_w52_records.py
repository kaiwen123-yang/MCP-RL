"""W52 same-call chosen candidates, branch state and command readback; no rollout."""
import json
import numpy as np
from s1_go2_w51_records import W51Records
from s1_go2_w50_records import digest_arrays,write
class W52Records(W51Records):
 def __init__(self,*args):
  super().__init__(*args)
  self.last_geometry=None;self.last_source=None;self.branch_records=[]
  self.schema.update(experiment='W52',geometry=self.config['w52_geometry'],search=self.config['w52_search'],
    foot_orders=dict(trajectory=['FL','FR','RL','RR'],motion=['FL','RL','FR','RR'],permutation=[0,2,1,3]))
  write(self.out/'w52-recording-schema.json',self.schema)
 def plan(self,t):
  super().plan(t) # This reads the current r33/search ring before the code below.
  row=self.current;call=row['call'];cid=call['call_id'];seed=self.seed_records[-1]
  geometry=seed.get('motion',{}).get('geometry_reason')
  source=call['candidate_source']
  if geometry is not None and geometry!=self.last_geometry:
   self.event(t,'geometry_availability_change',reason=geometry)
   self.last_geometry=geometry
  # Save every actually installed complete object and same-call candidate ring.
  # No extra native evaluation is performed to obtain these records.
  if call['policy_updated'] or source!=self.last_source:
   self.save_call(row)
   if cid in self.r33_ring:self.save_r33(cid)
   if cid in self.search_ring:self.save_search(cid)
  self.last_source=source
  path=self.out/'w51-motion-branches.json'
  if path.exists():
   branches=json.loads(path.read_text())
   if branches['call_id']==cid:
    branches['objects']={}
    if call['policy_updated'] or t<=self.until:
     for b in branches.get('branches',[]):
      role=b['source']
      for suffix in ('backward-'+role+'.bin','centers-'+role+'.bin'):
       p=self.out/('w51-motion-'+suffix)
       if b['first_valid'] and p.exists():
        obj={'payload':np.fromfile(p,dtype='<f8')}
        branches['objects'][suffix]=self.save_object(digest_arrays(obj),obj)
      p=self.out/('r29-current-plan.bin.search-'+role)
      if p.exists():
       obj={'payload':np.fromfile(p,dtype='<f8')}
       if int(obj['payload'][1])==cid:
        branches['objects']['search-'+role]=self.save_object(digest_arrays(obj),obj)
    self.branch_records.append(branches)
    with (self.out/'w52-branches.jsonl').open('a') as f:f.write(json.dumps(branches,allow_nan=False)+'\n')
  ctx=row['context'];mission=row['compact']['mission_actor']
  expected=np.r_[mission['command_velocity'],mission['command_omega']]
  actual=ctx['native_userdata'][78:84]
  if cid==1 or (np.any(expected) and 'moving_command_readback' not in self.seen):
   name='first_normal_command_readback' if cid==1 else 'moving_command_readback'
   self.seen.add(name)
   write(self.out/(name+'.json'),dict(call_id=cid,time_s=float(t),controller=self.controller,
     expected_world_command=expected.tolist(),native_userdata78_84=actual.tolist(),
     max_abs_difference=float(np.max(abs(actual-expected))),physics_added=0,
     artifact_sha256=self.fixed_policy['controller_binary_sha256']))
 def finish(self,t,reason):
  result=super().finish(t,reason)
  result.update(experiment='W52',branch_calls=len(self.branch_records),branches='w52-branches.jsonl',
    geometry='w51-seeds.jsonl',chosen_policy_and_rollout='w50-event-manifest.json',
    qualification_usable=self.config['r11_stage']['level']!='D8')
  write(self.out/'w52-event-manifest.json',result);return result
