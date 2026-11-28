"""Solve real archived Stable F QPs with CPU OSQP and finite sparse JAX Riccati/ADMM."""
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import jax
import numpy as np
import osqp
from scipy import sparse

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from s1_w54_f_r52_backend import assemble_force_horizon
from w61.mpc.fixed_qp import solve_batch


def cpu(inputs,cfg):
    start=time.perf_counter()
    qp=assemble_force_horizon(inputs,cfg)
    solver=osqp.OSQP()
    solver.setup(P=sparse.triu(qp['P'],format='csc'),q=qp['q'],A=qp['A'],l=qp['lower'],u=qp['upper'],
                 verbose=False,eps_abs=cfg['qp']['eps_abs'],eps_rel=cfg['qp']['eps_rel'],max_iter=cfg['qp']['max_iter'],
                 polishing=True,adaptive_rho=True,adaptive_rho_interval=50,check_termination=25,warm_starting=True)
    sol=solver.solve(raise_error=False)
    if sol.info.status_val not in (1,2):
        raise RuntimeError(sol.info.status)
    nx=inputs['reference'].size
    return sol.x[:nx].reshape(-1,12),sol.x[nx:].reshape(-1,4,3),time.perf_counter()-start


def objective(x,f,p):
    e=(x-p['ref'])**2
    df=np.diff(np.concatenate((p['prev'][None],f)),axis=0)
    return float(((e.sum(0)+e[-1])*p['qw']).sum()+p['R']*(f**2).sum()+p['D']*(df**2).sum())


def main():
    seedroot=ROOT.parent/'MCP-RL-w61-gait/work/w61/gait/r55-dev0/w55-f'
    cases=[seedroot/f'mpc-{i:06}.npz' for i in (15,38,72,112,155,205,237)]
    src=ROOT.parent/'MCP-RL-w56-r60/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12'
    # Archived A and B fixed motion. Different initial states remain separate cases.
    for side in ('a','b'):
        path=next(src.glob(f'*-{side}-r60-*-E30-attempt01'))/'w55-f'
        cases.extend(path/f'mpc-{i:06}.npz' for i in (32,120,310))
    inputs=[]
    refs=[]
    rows=[]
    for f in cases:
        with np.load(f) as data:
            d={k:data[k] for k in ('x0','reference','last_force','contact','feet','force_normal_upper','stage_A','stage_B','stage_c')}
            was=data['forces']
        name='s1-w55-f-r55.json' if seedroot in f.parents else 's1-w56-f-r60.json'
        cfg=json.loads((ROOT/'experiments/terrain_memory'/name).read_text())
        p=dict(A=d['stage_A'],B=d['stage_B'],c=d['stage_c'],x0=d['x0'],ref=d['reference'],prev=d['last_force'],
               qw=np.array(cfg['mpc']['state_weights']),R=cfg['mpc']['force_regularization'],D=cfg['mpc']['force_difference_weight'],
               mu=cfg['force_pyramid_slope'],contact=d['contact'],cap=d['force_normal_upper'])
        x,force,wall=cpu(d,cfg)
        rows.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),cpu_OSQP_s=wall,
                         archived_force_max_delta_N=float(np.max(abs(was-force))),reference_objective=objective(x,force,p),iterations={}))
        inputs.append(p)
        refs.append(force)
    start=time.perf_counter()
    proc=time.process_time()
    host={k:np.stack([p[k] for p in inputs]) for k in inputs[0]}
    budgets=(4,16,64,256)
    for n in budgets:
        # First call includes compilation and transfers; repeat synchronization
        # reports warmed end-to-end host input to host solution on this shared GPU.
        times=[]
        for _ in range(4):
            t=time.perf_counter()
            x,force,primal,dual=map(np.asarray,solve_batch(host,n,.001))
            times.append(time.perf_counter()-t)
        for i,p in enumerate(inputs):
            z=force[i]
            off=np.abs(z*(~p['contact'])[:,:,None]).max()
            cone=np.maximum(np.abs(z[:,:,:2])-p['mu']*z[:,:,2,None],0).max()
            row=rows[i]
            row['iterations'][str(n)]=dict(first_force_L2_N=float(np.linalg.norm(z[0]-refs[i][0])),
                sequence_rms_N=float(np.sqrt(np.mean((z-refs[i])**2))),
                objective_delta=objective(x[i],z,p)-row['reference_objective'],
                maximum_returned_constraint_violation_N=float(max(off,cone,np.maximum(-z[:,:,2],0).max(),
                    np.maximum(z[:,:,2]-p['cap'],0).max())),ADMM_primal_inf_N=float(primal[i]),ADMM_dual_inf=float(dual[i]))
        for row in rows:
            row['iterations'][str(n)]['batch_compile_first_and_three_warmed_s']=times
    out=ROOT/'results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/mpc/w61-c-fixed-qp-v1'
    out.mkdir(parents=True,exist_ok=True)
    record=dict(runtime=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),fixed_iterations=budgets,rho=.001,
        device=str(jax.devices()[0]),jax=jax.__version__,python=platform.python_version(),cases=rows,wall_s=time.perf_counter()-start,
        cpu_core_s=time.process_time()-proc,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        notes='Historical original objectives and hard constraints. Times are shared D57 host-to-device-to-host batch; no execution on NX.',
        new_robot_samples=0,deadline=None,wall_deadline_enabled=False,kill_on_conversation_end=False)
    (out/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('completed',len(rows),budgets,flush=True)


if __name__=='__main__':
    main()
