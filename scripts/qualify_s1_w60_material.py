"""Compare actual r58 A/B measured packets and formal WBC calls to CPU r60."""
import hashlib
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from qualify_s1_w56_material import solve, PARAMETERS, RUN
from s1_w56_f_r58_backend import FController as R58
from s1_w56_f_r60_backend import FController as R60
from s1_w56_window import WindowObserver
from s1_w60_cpu_observer import CPUWindowObserver, LIBRARY, ROOT


RAW=ROOT.parent/'MCP-RL-w56-material'/'outputs'/RUN
OUT=ROOT/'results'/RUN/'w56/c'
CFG_PATH=Path('experiments/terrain_memory/s1-w56-f-r60.json')
WEIGHT_PATH=Path('work/assets/B_random1000--ideal-potentials.npz')
CFG=json.loads((ROOT/CFG_PATH).read_text())
DIFF=lambda a,b:float(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0.))


def checksum(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source(case):
    raw,=RAW.glob(f'job-*-f-{case}-r58-eval*-E30-attempt01')
    j={'a':210,'b':230}[case]
    with np.load(raw/'w55-f/control-00006.npz') as f:row={k:f[k][j] for k in f.files}
    with np.load(raw/'w55-f'/f"mpc-{int(row['plan_call']):06}.npz") as f:p=dict(f)
    with np.load(raw/'material-0005.npz') as f:foot=f['foot']
    # Include the complete virgin prefix physically measured before this chunk.
    prefix=[]
    for idx in range(5):
        with np.load(raw/f'material-{idx:04}.npz') as f:prefix.append(f['foot'][:,:30])
    t=row['time'];prefix.append(foot[foot[:,1]<=t+1e-9,:30])
    all_rows=np.concatenate(prefix)
    return raw,row,p,all_rows


def run_path(case,device,prior_law=None):
    raw,row,p,rows=source(case)
    path=ROOT/WEIGHT_PATH
    observer=(CPUWindowObserver if device=='cpu' else WindowObserver)(PARAMETERS,path,law=prior_law)
    g=observer.ground
    if device=='jax':
        original=g.step
        measured=[];owned=[]
        def capture(feet,v,dt,t,log):
            forces,indent,maps=original(feet,v,dt,t,log)
            measured.append(forces[:,2])
            owner={}
            for leg,(ids,aa) in enumerate(maps):
                for i,a in zip(ids,aa):
                    if i not in owner or feet[leg,2]<feet[owner[i][0],2]:owner[i]=(leg,a)
            totals=np.zeros(4)
            for leg,a in owner.values():totals[leg]+=a
            owned.append(totals)
            return forces,indent,maps
        g.step=capture
    begin=time.perf_counter();host=time.process_time()
    outputs=[];active_wall=0.;active_packets=0
    for j in range(0,len(rows),10):
        packet=rows[j:j+10]
        for r in packet:observer.completed(r[0],r[1],r[6:18].reshape(4,3),r[18:30].reshape(4,3))
        b=time.perf_counter();observer.observe(packet[-1,1],None,None)
        if len(g.ids):
            active_wall+=time.perf_counter()-b;active_packets+=1
        if len(g.ids):
            # Selected archived sections are short; retain native units for all rows.
            fq=(observer.last_forces.copy() if device=='cpu' else np.stack(measured[-10:]))
            aa=(observer.last_areas.copy() if device=='cpu' else np.stack(owned[-10:]))
            outputs.append((observer.time,g.e.copy(),g.q.copy(),g.stress.copy(),fq,aa,tuple(g.ids)))
    elapsed=time.perf_counter()-begin;cpus=time.process_time()-host
    xy=p['foot_center_plan'][int(abs(p['times']-(row['time']+.2)).argmin()),:,:2]
    b=time.perf_counter();recovery,_=observer.recovery(xy);query_wall=time.perf_counter()-b
    soft=row['contact_point_source']==2
    model=mujoco.MjModel.from_binary_path(str(raw/'model.mjb'));data=mujoco.MjData(model)
    b=time.perf_counter()
    ans=solve(R60 if device=='cpu' else R58,model,data,p,row,soft,observer,rows[-1,6:18].reshape(4,3),float(row['time']))
    wbc_wall=time.perf_counter()-b
    return dict(run=raw.name,physical_t=float(row['time']),active_packets=active_packets,world_cells=len(g.e),
                packet_count=len(rows)//10,max_column_residual_MPa=float(g.max_residual),
                observer_wall_s=elapsed,active_packet_wall_s=active_wall,observer_cpu_core_s=cpus,
                recovery_wall_s=query_wall,formal_WBC_reconstruction_and_solve_wall_s=wbc_wall,
                archive_mapped_u_abs_rad=DIFF(row['mapped_u'],ans['action']),
                archive_reference_abs_m=DIFF(row['feet_position_reference'],ans['reference']),
                archive_wbc_force_abs_N=DIFF(row['force'],ans['force']),
                qr=ans['primal'],qd=ans['dual']),outputs,recovery,ans,observer


def main():
    # solve() temporarily creates a record directory underneath this path.
    (ROOT/'work/w56-c').mkdir(exist_ok=True)
    import qualify_s1_w56_material as original
    original.CFG=CFG
    begun=time.time();pairs={}
    for case in ('a','b'):
        pairs[case]=run_path(case,'cpu')
    modules_cpu=sorted(x for x in sys.modules if x in ('jax','jaxlib'))
    if modules_cpu:raise RuntimeError(f'CPU-only deployed path unexpectedly imported {modules_cpu}')
    results={};law=None
    for case,cpu in pairs.items():
        jax=run_path(case,'jax',law)
        law=jax[-1].ground.law
        diff=np.zeros(5)
        for a,b in zip(cpu[1],jax[1],strict=True):
            if a[0]!=b[0] or a[-1]!=b[-1]:raise ValueError('Published world keys or time differed')
            diff=np.maximum(diff,[DIFF(x,y) for x,y in zip(a[1:-1],b[1:-1])])
        coordinate,action,force=(DIFF(cpu[3][k],jax[3][k]) for k in ('reference','action','force'))
        recovery=DIFF(cpu[2],jax[2])
        results[case]=dict(r60_CPU=cpu[0],r58_JAX=jax[0],max_e11_e22_or_q11_q22_absolute=diff[:2].tolist(),
                           max_two_axis_stress_MPa=float(diff[2]),max_owned_normal_force_N=float(diff[3]),
                           max_area_m2=float(diff[4]),recovery_m=recovery,formal_ref_m=coordinate,
                           formal_mapped_input_rad=action,formal_solved_force_N=force)
        print(case,results[case],flush=True)
        if not (max(diff[:2])<1e-8 and diff[2]<1e-6 and diff[3]<1e-5 and diff[4]<1e-16
            and recovery<1e-10 and coordinate<1e-10 and action<1e-8 and force<1e-5
            and cpu[0]['archive_mapped_u_abs_rad']<1e-8 and cpu[0]['max_column_residual_MPa']<=1e-7):
            raise ValueError('Saved causal-window or formal WBC numerical discrepancy')
    sources=set(CFG['material_execution_sources']+CFG['material_compiled_artifacts']+
                [str(CFG_PATH),str(WEIGHT_PATH),CFG['backend_source']]+['scripts/'+n for n in CFG['inherited_execution_sources']])
    result=dict(schema='s1_r60_cpu_window_full_WBC_gate_v1',ready_for_original_E30=True,
                source_sha256={k:checksum(ROOT/k) for k in sorted(sources)},start_unix_s=begun,end_unix_s=time.time(),
                jax_modules_after_cpu_observer_query_inverse_wbc=modules_cpu,
                cpu_native_library=str(LIBRARY),cases=results,
                scope='r58 A/B E30 archive from virgin state to first archived all-confirmed soft MPC replanning point; every closed window matched including both stresses, measured footprint and all four normal forces. CPU and JAX share WBC and hard limits; independent CPU builds must earn original E30 first.')
    path=OUT/'r60-interface-qualification.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)


if __name__=='__main__':main()
