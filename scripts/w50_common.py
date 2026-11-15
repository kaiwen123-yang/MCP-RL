"""W50 fixed W49 inputs; all originals stay on D57."""
import json, os
from pathlib import Path
import numpy as np
from w49_common import write, sha, arrays, fields, save_policy
from compare_s1_go2_w44_saved import cost_definitions, parse_candidates, cost_terms
ROOT=Path(__file__).resolve().parents[1]
REL=Path('outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
LARGE=ROOT/REL/'w50'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w50'
BUILD=ROOT/'work/r11-frozen/S5-r38'
LABELS={'A237':('a',1,237,4.32),'B236':('b',2,236,4.30),'A328':('a',1,328,6.14),'B625':('b',2,625,12.08)}
def source(label):
 suffix,seed,_,_=LABELS[label]
 return ROOT/REL/f'job-20261113T145020Z-w49-{suffix}-eval-PANN_PPO_S5-s{seed}-E30-attempt01'
def prepare():
 records=[]
 for label,(suffix,seed,cid,expected_t) in LABELS.items():
  d=source(label);manifest=json.loads((d/'w49-event-manifest.json').read_text())
  r=next(x for x in manifest['event_calls'] if x['call_record']['call_id']==cid)
  origin=r['boundary']['origin_s']
  if abs(origin-expected_t)>1e-8:raise ValueError('Fixed W49 call/time mismatch')
  dest=LARGE/'inputs'/label;dest.mkdir(parents=True,exist_ok=False)
  objects={};ident={}
  for role,desc in r['objects'].items():
   path=d/desc['file'];objects[role]=arrays(path);ident[role]=dict(path=str(path),**desc)
  ctx=objects['context']
  st=dict(time=origin,qpos=ctx['native_state'][:19],qvel=ctx['native_state'][19:37],act=np.zeros(0),
   mocap_pos=ctx['actual_mocap_pos'],mocap_quat=ctx['actual_mocap_quat'],userdata=ctx['native_userdata'])
  np.savez_compressed(dest/'state.npz',**st)
  save_policy(dest/'installed.bin',objects['policy'],r['boundary']['installed_plan_id'],origin)
  save_policy(dest/'P0.bin',objects['p0'],r['boundary']['P0']['plan_id'],origin)
  ctx['private_gait'].astype('<f8').tofile(dest/'context.bin')
  objects['reference']['r35_reference'].astype('<f8').tofile(dest/'reference.bin')
  resolved=json.loads((ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'/(d.name+'-resolved.json')).read_text())
  settings=[x for x in resolved['applied_settings'] if x['time_s']<=origin+1e-10][-1]
  model=d/'model/G1-project-go2.xml'
  f=dict(label=label,source=str(d),source_call=cid,source_plan=r['boundary']['installed_plan_id'],origin_s=origin,
   model_path=str(model),settings=settings,source_objects=ident,boundary=r['boundary'],call=r['call_record'],
   cost_definitions=cost_definitions(model,settings['cost_weights']),historical_exact_replay=False,
   reconstruction='Same saved W49 input P0/native state/private gait and same-epoch r35 target; explicit native predictor context, not arbitrary plant/actor resume.',
   baseline='Current nominal screening selects full B; P0 and prior installed output are distinct.',
   missing_historical_fields=r['boundary']['mission_actor']['external_snapshot_scope']['external_unserialized_fields'],
   integration_scope='Saved integration and partial external state exist but are not represented as complete plant replay.',
   same_epoch_reference=True,identity_schema=49)
  write(dest/'input.json',f);records.append(f)
 write(SMALL/'inputs.json',dict(status='prepared',fixtures=records,new_physics=0,new_solves=0))
 return records

def env(out,label,policy='P0.bin',**extra):
 inp=LARGE/'inputs'/label
 v=dict(os.environ,MJPC_TASKS_DIR=str(BUILD/'runtime-tasks'),LD_LIBRARY_PATH=str(BUILD/'lib'),
  MJPC_R33_INPUT_POLICY=str(inp/policy),MJPC_R33_MECHANISM='H',MJPC_R34_ACCEPTANCE='strict',
  MJPC_R34_CONTEXT=str(inp/'context.bin'),MJPC_R35_REFERENCE=str(inp/'reference.bin'),MJPC_R35_BRANCH='constant',
  MJPC_R29_SNAPSHOT=str(out/'snapshot.bin'),MJPC_R29_CALL_RECORD=str(out/'planner-calls.jsonl'),
  MJPC_R27_PLAN_RECORD=str(out/'planner-records.jsonl'),MJPC_R27_ACTION_RECORD=str(out/'feedback-actions.jsonl'))
 v.update(extra);return v

if __name__=='__main__':prepare()
