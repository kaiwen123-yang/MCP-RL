#!/usr/bin/env python3
"""Numerical evidence for the R8 held-input dynamics on retained states."""
import json
from pathlib import Path
import numpy as np
import mujoco
from run_s1_go2_wb_baseline import index_file
ROOT=Path(__file__).resolve().parents[1]

def main():
    c=json.loads((ROOT/'experiments/terrain_memory/s1-go2-wb-support.json').read_text())
    source=ROOT/'outputs/terrain_memory/simulation/tm-s1-go2-wb-equilibrium-r7/g1-nsh-nominal-03'
    m=mujoco.MjModel.from_binary_path(str(source/'model.mjb'))
    m.opt.timestep=.001;m.opt.integrator=3
    m.jnt_solimp[:,0]=0;m.geom_solimp[:,0]=0;m.pair_solimp[:,0]=0
    with np.load(source/'trajectory.npz') as z:
        times=z['time'];q=z['qpos'];v=z['qvel'];u=z['q_target']
    nx=2*m.nv+m.na;eps=1e-6;records=[]
    for t in [0.,1.,2.,34.,40.,50.]:
        i=int(np.searchsorted(times,t-1e-8));d=mujoco.MjData(m)
        d.time=times[i];d.qpos[:]=q[i];d.qvel[:]=v[i];d.ctrl[:]=u[i]
        initial=mujoco.MjData(m);mujoco.mj_copyData(initial,m,d)
        a1=np.zeros((nx,nx));b1=np.zeros((nx,m.nu));a2=a1.copy();b2=b1.copy()
        mujoco.mjd_transitionFD(m,d,eps,False,a1,b1,None,None)
        mujoco.mj_step(m,d)
        mujoco.mjd_transitionFD(m,d,eps,False,a2,b2,None,None)
        composed_a=a2@a1;composed_b=a2@b1+b2
        def advance(delta_x=None,delta_u=None):
            x=mujoco.MjData(m);mujoco.mj_copyData(x,m,initial)
            if delta_x is not None:
                mujoco.mj_integratePos(m,x.qpos,delta_x[:m.nv],1.)
                x.qvel[:]+=delta_x[m.nv:2*m.nv]
            if delta_u is not None:x.ctrl[:]+=delta_u
            mujoco.mj_step(m,x);mujoco.mj_step(m,x)
            return x
        nominal=advance();full_a=np.zeros_like(a1);full_b=np.zeros_like(b1)
        def difference(x):
            position=np.zeros(m.nv);mujoco.mj_differentiatePos(m,position,eps,nominal.qpos,x.qpos)
            return np.r_[position,(x.qvel-nominal.qvel)/eps]
        for j in range(nx):
            dx=np.zeros(nx);dx[j]=eps;full_a[:,j]=difference(advance(delta_x=dx))
        for j in range(m.nu):
            du=np.zeros(m.nu);du[j]=eps;full_b[:,j]=difference(advance(delta_u=du))
        records.append(dict(saved_time_s=float(times[i]),A_max_abs_difference=float(abs(full_a-composed_a).max()),
            B_max_abs_difference=float(abs(full_b-composed_b).max()),
            A_relative_frobenius=float(np.linalg.norm(full_a-composed_a)/np.linalg.norm(full_a)),
            B_relative_frobenius=float(np.linalg.norm(full_b-composed_b)/np.linalg.norm(full_b)),
            held_transition_dt_s=float(nominal.time-initial.time),finite_difference_eps=eps))
    with index_file(ROOT/c['outputs']['root'],c) as idx:
        idx['held_dynamics_calculation']=dict(records=records,method='A2*A1 and A2*B1+B2 versus forward differences of two mj_step calls under one fixed u, MuJoCo position tangent coordinates.',
            plant_starts=0,model='R7 N_SH prediction settings; retained project plant geometry, bounds and refsafe.',
            cost_semantics='C/D from first substep, first-substep residual preserved in both C++ rollout paths; terminal residual from final forward at node175.')
    print(json.dumps(records),flush=True)
if __name__=='__main__':main()
