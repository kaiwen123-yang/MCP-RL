"""Shared W49 complete-policy I/O and new-event source identities."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from compare_s1_go2_w44_saved import cost_definitions,parse_candidates
ROOT=Path(__file__).resolve().parents[1]
REL=Path('outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
LARGE=ROOT/REL/'w49'
SMALL=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w49'
BUILD=ROOT/'work/r11-frozen/S5-r37'
LABELS={'A213':('a',1,213),'A363':('a',1,363),'A364':('a',1,364),'B176':('b',2,176),'B328':('b',2,328)}
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arrays(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def fields(T,ns,nu,nx,nr):
 return [('reference_states',(T,ns)),('reference_actions',(T-1,nu)),('reference_times',(T,)),('reference_residual',(T,nr)),
 ('K',(T,nu,nx)),('H',(T,nu,nu)),('base_Qu',(T,nu)),('base_Qxu',(T,nx,nu)),('base_active',(T,nu))]
def save_policy(p,a,plan,origin):
 T,ns=a['reference_states'].shape;nu=a['reference_actions'].shape[1];nx=a['K'].shape[2];nr=a['reference_residual'].shape[1]
 valid,mu,typ,nom,alpha,scale,rep=a['semantics']
 h=[49,plan,float(a['reference_times'][0]),T,ns,nu,nx,nr,mu,typ,nom,alpha,scale,rep,96,0]
 ext=a.get('delta_header',np.array([49,valid,0,0,0,0]))
 np.concatenate([h]+[a[k].ravel() for k,_ in fields(T,ns,nu,nx,nr)]+[ext,a.get('delta_times',np.array([])),a.get('delta_values',np.array([])).ravel()]).astype('<f8').tofile(p)
def read_policy(p):
 z=np.fromfile(p,dtype='<f8');h=z[:16];a={};i=16;T,ns,nu,nx,nr=map(int,h[3:8])
 for k,shape in fields(T,ns,nu,nx,nr):
  n=int(np.prod(shape));a[k]=z[i:i+n].reshape(shape).copy();i+=n
 a['delta_header']=z[i:i+6].copy();i+=6;n=int(a['delta_header'][2])
 a['delta_times']=z[i:i+n].copy();i+=n;a['delta_values']=z[i:i+n*nu].reshape(n,nu).copy();i+=n*nu
 a['semantics']=np.r_[a['delta_header'][1],h[8:14]]
 assert i==len(z) and h[0]==49 and a['delta_header'][0]==49
 return h,a
def source(label):
 suffix,seed,cid=LABELS[label]
 return ROOT/REL/f'job-20261113T090632Z-w48-{suffix}-eval-PANN_PPO_S5-s{seed}-E30-attempt01'
def prepare():
 for label,(suffix,seed,cid) in LABELS.items():
  d=source(label);manifest=json.loads((d/'w48-event-manifest.json').read_text())
  r=next(x for x in manifest['event_calls'] if x['call_record']['call_id']==cid)
  dest=LARGE/'inputs'/label;dest.mkdir(parents=True,exist_ok=False)
  objects={};ident={}
  for role,desc in r['objects'].items():
   p=d/desc['file'];assert p.stat().st_size==desc['bytes'] and sha(p)==desc['file_sha256']
   objects[role]=arrays(p);ident[role]=dict(path=str(p),**desc)
  ctx=objects['context'];origin=r['boundary']['origin_s']
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
   model_path=str(model),model_sha256=sha(model),settings=settings,source_objects=ident,boundary=r['boundary'],call=r['call_record'],
   cost_definitions=cost_definitions(model,settings['cost_weights']),historical_exact_replay=False,
   missing_historical_fields=['complete integration/solver state','external material/callback state'],
   identity_schema=49,policy_validity_source='W48 explicitly saved semantics, not legacy import default')
  write(dest/'input.json',f)
  print(json.dumps(dict(label=label,origin=origin,P0=r['boundary']['P0']['plan_id'],installed=r['boundary']['installed_plan_id'])),flush=True)
def env(out,label,policy='P0.bin',**extra):
 inp=LARGE/'inputs'/label
 v=dict(os.environ,MJPC_TASKS_DIR=str(BUILD/'runtime-tasks'),LD_LIBRARY_PATH=str(BUILD/'lib'),
  MJPC_R33_INPUT_POLICY=str(inp/policy),MJPC_R33_MECHANISM='N',MJPC_R34_ACCEPTANCE='strict',
  MJPC_R34_CONTEXT=str(inp/'context.bin'),MJPC_R35_REFERENCE=str(inp/'reference.bin'),MJPC_R35_BRANCH='constant',
  MJPC_R29_SNAPSHOT=str(out/'snapshot.bin'),MJPC_R29_CALL_RECORD=str(out/'planner-calls.jsonl'),
  MJPC_R27_PLAN_RECORD=str(out/'planner-records.jsonl'),MJPC_R27_ACTION_RECORD=str(out/'feedback-actions.jsonl'))
 v.update(extra);return v
if __name__=='__main__':prepare()
