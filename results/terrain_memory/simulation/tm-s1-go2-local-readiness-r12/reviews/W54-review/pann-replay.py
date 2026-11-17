"""Read-only constitutive replay; no MuJoCo, robot simulation, training or repo edits."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,pathlib,json,time,hashlib,inspect,argparse,csv,subprocess
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=pathlib.Path,required=True)
parser.add_argument('--originals',type=pathlib.Path,required=True)
parser.add_argument('--weights',type=pathlib.Path,required=True)
parser.add_argument('--output',type=pathlib.Path,required=True)
parser.add_argument('--cases',nargs='+',choices=['r48','r52'],default=['r52','r48'])
parser.add_argument('--save-arrays',action='store_true')
parser.add_argument('--reuse-completed',action='store_true',help='Explicitly reuse compatible completed case summaries in the output folder.')
args=parser.parse_args()
ROOT=args.repo.resolve();OUT=args.output.resolve();RAW=args.originals.resolve();OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'src'))
import jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
from tm02.s1_material import ReferenceColumns,LearnedColumns
from tm02.reference import IntegrationFailure
src=inspect.getsource(LearnedColumns)
assert "device = jax.devices('gpu')[0]" in src
src=src.replace("device = jax.devices('gpu')[0]","device = jax.devices('cpu')[0]")
namespace={'np':np,'IntegrationFailure':IntegrationFailure,'__package__':'tm02'}
exec(src,namespace)
weights=args.weights.resolve();law=namespace['LearnedColumns'](weights)
params=json.load(open(ROOT/'experiments/tm02/r2-multihistory.json'))['reference_material']['parameters']
ref=ReferenceColumns(params)
N=8;zero=jnp.zeros(N);false=jnp.zeros(N,bool)
def call_law(ep,qp,normal,dt,fixed):
 # The constitutive columns are independent. Sequential batch=1 avoids a
 # heterogeneous CPU-vmap/XLA stall reproduced on the saved N48 step.
 # Same source, same residuals, same Newton/backtracking limits and start.
 values=[np.empty_like(ep),np.empty_like(qp),np.empty_like(ep),np.empty(len(normal))]
 for i in range(len(normal)):
  virgin=(not fixed[i] and normal[i]==0 and not np.any(ep[i]) and not np.any(qp[i]))
  cache_key=float(dt)
  if virgin and cache_key in virgin_cache:out=virgin_cache[cache_key]
  else:
   out=tuple(np.asarray(x) for x in law.solve(ep[i:i+1],qp[i:i+1],normal[i:i+1],dt,fixed[i:i+1]))
   if virgin:virgin_cache[cache_key]=tuple(x.copy()for x in out)
  for target,value in zip(values,out):target[i]=value[0]
 return tuple(values)
virgin_cache={}
def rollout(normals,mask,areas,dt):
 # Python sequencing keeps a deployed-style residual gate at every increment.
 # Constitutive kernels are the unchanged jitted LearnedColumns.solve on CPU.
 ep=np.zeros((N,2));qp=ep.copy();output=[[] for _ in range(7)]
 for step_no,(normal,geometric,area) in enumerate(zip(normals,mask,areas)):
  fe,fq,fs,fr=call_law(ep,qp,np.zeros(N),dt,np.zeros(N,bool))
  candidate=geometric & (normal<=fe[:,0]);en=np.where(candidate,normal,0.)
  ce,cq,cs,cr=call_law(ep,qp,en,dt,candidate)
  active=candidate & (cs[:,0]<0)
  e=np.where(active[:,None],ce,fe);q=np.where(active[:,None],cq,fq);s=np.where(active[:,None],cs,fs)
  pressure=np.where(active,-s[:,0]*25000.,0.);residual=np.maximum(fr,np.where(candidate,cr,0.))
  if not np.isfinite(residual).all() or residual.max()>1e-7:
   raise RuntimeError(f'Deployed constitutive failure at replay increment {step_no}: residual={residual.max()}')
  for target,value in zip(output,(e,q,s,pressure,pressure*area,residual,active)):target.append(value.copy())
  ep,qp=e,q
  if step_no%100==0:print(json.dumps({'learned_increment':step_no,'of':len(normals),'max_residual_MPa':float(residual.max())}),flush=True)
 return tuple(np.asarray(v)for v in output)
def reference_replay(normals,mask,areas,dt):
 ep=np.zeros((N,2));qp=ep.copy();output=[[] for _ in range(7)]
 for normal,geometric,area in zip(normals,mask,areas):
  fe,fq,fs,fr=ref.step(ep,qp,np.zeros(N),dt,np.zeros(N,bool));candidate=geometric&(normal<=fe[:,0]);idx=np.flatnonzero(candidate)
  e,q,s=fe.copy(),fq.copy(),fs.copy();residual=fr.copy();active=np.zeros(N,bool)
  if len(idx):
   ce,cq,cs,cr=ref.step(ep[idx],qp[idx],normal[idx],dt,np.ones(len(idx),bool));comp=cs[:,0]<0;ii=idx[comp]
   e[ii]=ce[comp];q[ii]=cq[comp];s[ii]=cs[comp];residual[idx]=np.maximum(residual[idx],cr);active[ii]=True
  pressure=np.where(active,-s[:,0]*25000.,0.);vals=(e,q,s,pressure,pressure*area,residual,active)
  for target,value in zip(output,vals):target.append(value.copy())
  ep,qp=e,q
 return tuple(np.asarray(v)for v in output)
def geometry(foot,keys):
 bottom=foot[:,6:18].reshape(-1,4,3);half=np.sqrt(.001)/2;cell=.02
 interior=(bottom[:,:,0]>=.8+half)&(bottom[:,:,0]<=2.2-half)&(np.abs(bottom[:,:,1])<=.6-half)
 normals=np.zeros((len(foot),N));mask=np.zeros((len(foot),N),bool);areas=np.zeros((len(foot),N));owners=np.full((len(foot),N),-1)
 for k,key in enumerate(keys):
  lo=np.maximum(bottom[:,:,:2]-half,np.asarray(key)*cell);hi=np.minimum(bottom[:,:,:2]+half,(np.asarray(key)+1)*cell)
  aa=np.prod(np.maximum(hi-lo,0),axis=2);aa=np.where(interior,aa,0)
  bz=np.where(aa>0,bottom[:,:,2],np.inf);owner=bz.argmin(axis=1);valid=np.isfinite(bz.min(axis=1));rows=np.arange(len(foot))
  mask[:,k]=valid;normals[:,k]=np.where(valid,bottom[rows,owner,2]/.05,0.);areas[:,k]=np.where(valid,aa[rows,owner],0.);owners[:,k]=np.where(valid,owner,-1)
 return normals,mask,areas,owners
def metrics(a,b,keys):
 rows=[]
 for i,key in enumerate(keys):
  loaded=a[6][:,i]|b[6][:,i];unloaded=~loaded
  row={'key':key,'strain_rmse':float(np.sqrt(np.mean((a[0][:,i]-b[0][:,i])**2))),'q_rmse':float(np.sqrt(np.mean((a[1][:,i]-b[1][:,i])**2))),'stress11_rmse_MPa':float(np.sqrt(np.mean((a[2][:,i,0]-b[2][:,i,0])**2))),'stress11_abs_max_MPa':float(np.max(np.abs(a[2][:,i,0]-b[2][:,i,0]))),'stress11_peak_reference_MPa':float(np.max(np.abs(b[2][:,i,0]))),'normal_force_rmse_N':float(np.sqrt(np.mean((a[4][:,i]-b[4][:,i])**2))),'normal_force_abs_max_N':float(np.max(np.abs(a[4][:,i]-b[4][:,i]))),'normal_force_peak_reference_N':float(np.max(b[4][:,i])),'loaded_state_mismatches':int(np.sum(a[6][:,i]!=b[6][:,i])),'unloaded_recovery_abs_max_um':float(np.max(np.abs(a[0][unloaded,i,0]-b[0][unloaded,i,0]))*.05*1e6) if unloaded.any() else None}
  row['stress11_rmse_over_reference_peak']=row['stress11_rmse_MPa']/max(row['stress11_peak_reference_MPa'],1e-15)
  row['loaded_union_samples']=int(loaded.sum())
  row['stress11_loaded_rmse_MPa']=float(np.sqrt(np.mean((a[2][loaded,i,0]-b[2][loaded,i,0])**2))) if loaded.any() else None
  row['normal_force_loaded_rmse_N']=float(np.sqrt(np.mean((a[4][loaded,i]-b[4][loaded,i])**2))) if loaded.any() else None
  row['unloaded_reference_peak_compression_um']=float(np.max(-b[0][unloaded,i,0])*.05*1e6) if unloaded.any() else None
  row['stress11_peak_error_over_reference_peak']=row['stress11_abs_max_MPa']/max(row['stress11_peak_reference_MPa'],1e-15)
  rows.append(row)
 return rows
summary={'scope':'Offline autonomous constitutive integration on saved measured foot kinematics; never robot physics, control counterfactual, training, or reference-q injection.','repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'jax_version':jax.__version__,'device':'CPU float64; same LearnedColumns source except gpu device selection; independent columns dispatched batch=1 to avoid saved heterogeneous CPU-vmap stall; exact virgin free solution cached per dt','sample_selection':'Preselected by original reference physical extrema/first-load/cross-leg-revisit; not by model error.','cases':{}}
selection={'r52':[[41,4],[42,4],[43,3]],'r48':[[42,-9],[60,7],[61,8],[57,7],[51,6],[67,10]]}
if args.reuse_completed and (OUT/'pann-replay-summary.json').exists():
 previous=json.load(open(OUT/'pann-replay-summary.json'))
 if previous.get('weights_sha256')==summary['weights_sha256']:summary['cases'].update(previous['cases'])
for case,keys in selection.items():
 if case not in args.cases:continue
 started=time.time();d=next(RAW.glob('*-'+case+'-*'));files=sorted(d.glob('material-[0-9]*.npz'));foot=np.concatenate([np.load(f)['foot']for f in files]);cells=np.concatenate([np.load(f)['cells']for f in files]);normals,mask,areas,owners=geometry(foot,keys)
 first=int(np.flatnonzero(mask.any(axis=1))[0]//10*10);foot=foot[first:];normals=normals[first:];mask=mask[first:];areas=areas[first:];owners=owners[first:]
 allvals={};case_summary={'job':d.name,'keys':keys,'first_replay_boundary_s':float(foot[0,0]),'last_replay_material_s':float(foot[-1,1]),'n_fine_steps':len(foot),'initial_state':'virgin e=q=0 for every selected independent world cell before its first footprint overlap','selected_original_cells_saved':int(sum(np.all(cells[:,1:3]==k,axis=1).sum() for k in keys))}
 for label,stride in [('observer2ms',10),('plant02ms',1)]:
  if stride==1:ii=np.arange(len(foot));times=foot[ii,1];dt=.0002
  else:ii=np.flatnonzero(np.isclose(foot[:,0]/.002,np.rint(foot[:,0]/.002),atol=1e-5,rtol=0));times=foot[ii,0];dt=.002
  nr,ma,aa=normals[ii],mask[ii],areas[ii]
  print(json.dumps({'stage':'reference_replay','case':case,'clock':label,'steps':len(ii)}),flush=True)
  rv=reference_replay(nr,ma,aa,dt)
  print(json.dumps({'stage':'learned_replay','case':case,'clock':label,'reference_residual_max':float(rv[5].max())}),flush=True)
  before=time.time();pv=tuple(np.asarray(x) for x in rollout(nr,ma,aa,dt));elapsed=time.time()-before
  print(json.dumps({'stage':'learned_complete','case':case,'clock':label,'wall_s':elapsed,'residual_max':float(pv[5].max())}),flush=True)
  if not np.isfinite(pv[5]).all() or pv[5].max()>1e-7:raise RuntimeError('PANN Newton residual exceeds deployed acceptance')
  allvals[label]=(times,rv,pv,ii)
  case_summary[label]={'steps':len(ii),'dt_s':dt,'first_output_time_s':float(times[0]),'last_output_time_s':float(times[-1]),'pann_wall_s_including_compile':elapsed,'reference_residual_max_MPa':float(rv[5].max()),'pann_residual_max_MPa':float(pv[5].max()),'pann_vs_reference_same_boundary_and_dt':metrics(pv,rv,keys)}
  if args.save_arrays:np.savez_compressed(OUT/('pann-replay-'+case+'-'+label+'.npz'),time=times,keys=np.asarray(keys),normal=nr,geometric=ma,areas=aa,owner=owners[ii],reference_e=rv[0],reference_q=rv[1],reference_stress=rv[2],reference_pressure=rv[3],reference_normal_force=rv[4],reference_residual=rv[5],reference_loaded=rv[6],pann_e=pv[0],pann_q=pv[1],pann_stress=pv[2],pann_pressure=pv[3],pann_normal_force=pv[4],pann_residual=pv[5],pann_loaded=pv[6])
 tf,rf,pf,idxf=allvals['plant02ms'];to,ro,po,idxo=allvals['observer2ms']
 original_compare=[]
 tick_to_row={round(float(t)/.0002):i for i,t in enumerate(tf)}
 for j,key in enumerate(keys):
  saved=cells[np.all(cells[:,1:3]==key,axis=1)];ind=np.array([tick_to_row[round(float(t)/.0002)] for t in saved[:,0]]);pred=np.c_[rf[0][ind,j],rf[1][ind,j],rf[2][ind,j],rf[6][ind,j],rf[4][ind,j]];truth=np.c_[saved[:,3:9],saved[:,9],saved[:,12]]
  original_compare.append({'key':key,'saved_rows':len(saved),'max_abs_error_by_e11_e22_q11_q22_sigma11_sigma22_loaded_normalN':np.max(np.abs(pred-truth),axis=0).tolist()})
 case_summary['reference_fine_vs_published_cells']=original_compare
 alignment=np.max([x['max_abs_error_by_e11_e22_q11_q22_sigma11_sigma22_loaded_normalN'] for x in original_compare],axis=0)
 if alignment[:4].max()>1e-9 or alignment[4:6].max()>1e-6 or alignment[6]!=0 or alignment[7]>1e-6:raise RuntimeError('Reference replay does not align with published original states; no model accuracy verdict.')
 valid=np.array([round(float(t)/.0002) in tick_to_row for t in to]);match=np.array([tick_to_row[round(float(t)/.0002)]for t in to[valid]])
 case_summary['reference_observer2ms_vs_actual_fine_at_same_clock']=metrics(tuple(x[valid]for x in ro),tuple(x[match]for x in rf),keys)
 case_summary['pann_observer2ms_vs_actual_fine_at_same_clock']=metrics(tuple(x[valid]for x in po),tuple(x[match]for x in rf),keys)
 case_summary['clock_note']='plant02ms uses original pre-integration foot boundary and material output t+0.2ms; observer2ms uses measured endpoint foot at t for preceding 2ms interval, exactly the online observer sampling rule. Thus coarse-vs-fine includes endpoint sampling/within-window boundary loss and integrator dt, not a pure dt-only convergence test.'
 case_summary['cpu_dispatch']='Independent batch=1 columns with unchanged CPU kernel; exact virgin free solution cached per dt'
 case_summary['terminal_scope']='Fine replay includes all published .2ms force boundaries. Coarse uses regular 2ms measured boundary samples only; finish-only final observer update is excluded, so this is not a final live-observer dump.'
 case_summary['wall_s']=time.time()-started;summary['cases'][case]=case_summary
 (OUT/'pann-replay-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({'case_done':case,'wall_s':case_summary['wall_s'],'saved':str(OUT/'pann-replay-summary.json')}),flush=True)
rows=[]
for case,c in summary['cases'].items():
 for j,key in enumerate(c['keys']):
  row={'case':case,'ix':key[0],'iy':key[1]}
  for clock in ['plant02ms','observer2ms']:
   for name,value in c[clock]['pann_vs_reference_same_boundary_and_dt'][j].items():
    if name!='key':row[clock+'_'+name]=value
  for name,value in c['reference_observer2ms_vs_actual_fine_at_same_clock'][j].items():
   if name!='key':row['sampling_only_'+name]=value
  original=c['reference_fine_vs_published_cells'][j]
  row['reference_original_alignment_max_e_q']=max(original['max_abs_error_by_e11_e22_q11_q22_sigma11_sigma22_loaded_normalN'][:4])
  row['reference_original_alignment_max_stress_MPa']=max(original['max_abs_error_by_e11_e22_q11_q22_sigma11_sigma22_loaded_normalN'][4:6])
  row['reference_original_alignment_max_normal_N']=original['max_abs_error_by_e11_e22_q11_q22_sigma11_sigma22_loaded_normalN'][7]
  rows.append(row)
with (OUT/'pann-replay-cells.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary.update(robot_physics_steps=0,training_updates=0,valid_world_columns=sum(len(c['keys'])for c in summary['cases'].values()),effective_column_increments_excluding_dummy_padding=sum(len(c['keys'])*sum(c[q]['steps']for q in ['plant02ms','observer2ms'])for c in summary['cases'].values()))
(OUT/'pann-replay-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
short={'scope':summary['scope'],'weights_sha256':summary['weights_sha256'],'reference_original_validation_first':{},'cases':{},'robot_physics_steps':0,'training_updates':0,'world_columns':summary['valid_world_columns'],'effective_column_increments':summary['effective_column_increments_excluding_dummy_padding'],'terminal_scope':'Regular 2ms boundary samples only; final finish-only observer updates omitted. Full .2ms plant history included.'}
for case,c in summary['cases'].items():
 errors=np.max([x['max_abs_error_by_e11_e22_q11_q22_sigma11_sigma22_loaded_normalN']for x in c['reference_fine_vs_published_cells']],axis=0)
 short['reference_original_validation_first'][case]={'max_e_q':float(errors[:4].max()),'max_stress_MPa':float(errors[4:6].max()),'loaded_disagreements':float(errors[6]),'max_force_N':float(errors[7])}
 short['cases'][case]={}
 for clock in ['plant02ms','observer2ms']:
  rr=c[clock]['pann_vs_reference_same_boundary_and_dt']
  short['cases'][case][clock]={k:max(r[k]for r in rr)for k in ['stress11_abs_max_MPa','stress11_peak_error_over_reference_peak','normal_force_abs_max_N','unloaded_recovery_abs_max_um','stress11_loaded_rmse_MPa']}
  short['cases'][case][clock]['max_solver_residual_MPa']=c[clock]['pann_residual_max_MPa']
 short['cases'][case]['sampling_only_max_force_error_N']=max(x['normal_force_abs_max_N']for x in c['reference_observer2ms_vs_actual_fine_at_same_clock'])
(OUT/'pann-replay-short.json').write_text(json.dumps(short,indent=2)+'\n')
print('DONE',flush=True)
