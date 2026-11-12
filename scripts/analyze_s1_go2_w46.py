from pathlib import Path
import sys,json,numpy as np
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'scripts'),str(root/'work/upstreams/mujoco_mpc/python')]
from compare_s1_go2_w44_saved import parse_candidates,cost_terms
small=root/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46';out=root/'outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46-mechanisms'
checks=[]
for label in ['A148','A153','B155','B158']:
 f=json.loads((out/label/'input.json').read_text())
 for branch in ['none','constant','load']:
  run=out/label/branch;ref=np.fromfile(run/'r29-current-plan.bin.r35-reference-1.bin',dtype='<f8');ubar=ref[8:2108].reshape(175,12);R=ref[-2100:].reshape(175,12)
  cand=parse_candidates(run/'r29-current-plan.bin.r33-candidates');err=0;costerr=0;term=0
  for c in cand:
   a=c['arrays'];res=np.sqrt(R)*(a['actions']-ubar)
   err=max(err,float(abs(res-a['residual'][:-1,-12:]).max()))
   term=max(term,float(abs(a['residual'][-1,-12:]).max()))
   costerr=max(costerr,abs(sum(cost_terms(a['residual'],f['cost_definitions']).values())-c['metadata']['native_J']))
  a=max(cand,key=lambda c:float(np.sum(c['arrays']['residual'][:,-12:]**2)))['arrays']['actions'];delta=a-ubar;h=1e-6
  cuerr=float(abs((.5*R*(delta+h)**2/176-.5*R*(delta-h)**2/176)/(2*h)-R*delta/176).max())
  checks.append(dict(label=label,branch=branch,scored_candidates=len(cand),native_residual_reconstruction_max=err,terminal_max=term,native_cost_recompute_max=costerr,central_cost_Cu_max_error=cuerr,Cu='R*(u-ubar)/176',Cuu='R/176',double_projection_Jacobian=False))
(small/'numerical-consistency.json').write_text(json.dumps(checks,indent=2)+'\n')
g=Path('/root/autodl-tmp/MCP-RL/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w46')
z=np.load(g/'paired-states-actions.npz');tags=json.loads((g/'scenarios.json').read_text());rep=json.loads((small/'gpu.json').read_text())
validation=[]
for i,tag in enumerate(tags):
 if tag['policy']=='original-policy.bin' and tag['sign']==0:
  old=np.load(out/tag['label']/'old-r34-equivalence/selected.npz')['states']
  validation.append(dict(label=tag['label'],native_bridge_vs_original_agent_states_max=float(abs(z['native_states'][:,i]-old).max()),first20ms_max=float(abs(z['native_states'][:11,i]-old[:11]).max())))
from scipy.spatial.transform import Rotation
short_hard={}
for key in ['native_states','gpu_states']:
 s=z[key][:11];rp=Rotation.from_quat(s[:,:,[4,5,6,3]].reshape(-1,4)).as_euler('xyz')[:,:2]
 short_hard[key]=dict(body_fall=bool((s[:,:,2]<.12).any() or (abs(rp)>1.2).any()),maximum_abs_roll_pitch_rad=float(abs(rp).max()),minimum_z=float(s[:,:,2].min()))
rep['native_bridge_identity_check']=validation;rep['short20ms']['body_hard_condition_readback']=short_hard
rep['first_divergence']={}
for k in range(1,11):
 diff=abs(z['native_states'][k]-z['gpu_states'][k]);rep['first_divergence'][str(k)]=dict(time_after_origin_s=.002*k,position_max=float(diff[:,:19].max()),velocity_max=float(diff[:,19:].max()),action_difference_rad=float(abs(z['native_actions'][k-1]-z['gpu_actions'][k-1]).max()))
rep['interpretation']='Same FP64 state/action algebra agrees; engine-generated state differences are then amplified by the unchanged complete feedback and H projection. Native3.2.3 versus MuJoCo4/MJWarp FP32 contact/engine effects are not separately identified. No native-equivalent ranking or online acceleration claim.'
(small/'gpu.json').write_text(json.dumps(rep,indent=2)+'\n');(g/'result.json').write_text(json.dumps(rep,indent=2)+'\n')
print(json.dumps({'residual_max':max(x['native_residual_reconstruction_max'] for x in checks),'cost_error':max(x['native_cost_recompute_max'] for x in checks),'derivative_error':max(x['central_cost_Cu_max_error'] for x in checks),'native_bridge':validation,'short_hard':short_hard},indent=2))
