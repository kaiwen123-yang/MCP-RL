"""Recover first applied interval from actual A/E60 standing horizons."""
import argparse
import json
from pathlib import Path
import numpy as np
import osqp
from s1_w55_f_r55_backend import standing_stages, assemble_force_horizon, assemble_centroidal

ROOT = Path(__file__).resolve().parents[1]
RUN = 'tm-s1-go2-local-readiness-r12'
CASE = 'job-20261119T041610Z-w55-f-a-r53-eval-PANN_PPO_S5-s1-E60-attempt01'


def solve(horizon,config):
    problem=assemble_force_horizon(horizon,config)
    qp=osqp.OSQP()
    options={k:config['qp'][k] for k in ('eps_abs','eps_rel','max_iter','polishing','adaptive_rho_interval')}
    qp.setup(P=problem['P'],q=problem['q'],A=problem['A'],l=problem['lower'],u=problem['upper'],verbose=False,**options)
    result=qp.solve(raise_error=False)
    if result.info.status_val not in (1,2):
        raise ValueError(result.info.status)
    n=len(horizon['contact'])
    first=(n+1)*12
    return result.x[first:].reshape(n,4,3),result.x[:first].reshape(n+1,12)


def check():
    root=ROOT/'outputs/terrain_memory/simulation'/RUN/CASE/'w55-f'
    entries=[json.loads(row) for row in (root/'mpc-solves.jsonl').read_text().splitlines()]
    config=json.loads((ROOT/'experiments/terrain_memory/s1-w55-f-r55.json').read_text())
    answers=[]
    for target in (38.5,40.178,41.5):
        e=min(entries,key=lambda v:abs(v['time_s']-target))
        with np.load(root/e['file']) as z:
            h={k:z[k] for k in ('x0','reference','last_force','contact','feet','force_normal_upper','stage_A','stage_B','stage_c')}
            dt=config['mpc']['dt_s'];slope=config['force_pyramid_slope']
            x0=h['x0'];mass=float(z['mass']);inertia=z['inertia']
            fixed,centers,balancing=standing_stages(mass,inertia,x0,h['feet'][:-1],h['contact'],z['times'],dt,slope)
            geometry=[assemble_centroidal(mass,inertia,c,p,x0[:3],x0[6:9],dt,slope) for c,p in zip(centers,h['feet'])]
            row=dict(saved_file=e['file'],t=e['time_s'],COM_x_m=float(x0[3]),first_future_goal_x_m=float(h['reference'][1,3]),
                     original_first_future_moment_center_x_m=float(h['reference'][1,3]),
                     corrected_first_future_moment_center_x_m=float(centers[1,0]),measured_vx_m_s=float(x0[9]),comparisons={})
            for label,stages in (('original',None),('continuous_geometry_only',geometry),('S5-r55',fixed)):
                inputs=h.copy()
                if stages is not None:
                    inputs.update({k:np.stack([s[m] for s in stages]) for k,m in
                                   (('stage_A','A'),('stage_B','B'),('stage_c','c'))})
                f,x=solve(inputs,config)
                ax=(f[0].sum(0)/mass+[0.,0.,-9.81])[0]
                cone=np.linalg.norm(f[:,:,:2],axis=-1)-config['friction_coefficient']*f[:,:,2]
                row['comparisons'][label]=dict(first_ax_m_s2=float(ax),first_Fx_N=float(f[0,:,0].sum()),
                                              terminal_x_m=float(x[-1,3]),terminal_vx_m_s=float(x[-1,9]),
                                              circular_cone_violation_N=float(np.maximum(cone,0.).max()))
                if label=='original':
                    row['original_max_force_reproduction_error_N']=float(abs(z['forces']-f).max())
            answers.append(row)
    return dict(build='S5-r55',source_attempt=CASE,checks=answers,new_CPU_MPC_horizon_solves=9,
                new_physics_steps=0,new_WBC_solves=0,new_rollouts=0,deadline=None,wall_deadline_enabled=False)


if __name__=='__main__':
    outcome=check()
    d=ROOT/'results/terrain_memory/simulation'/RUN/'w55/f/r55-saved-hold.json'
    d.write_text(json.dumps(outcome,indent=2)+'\n')
    for row in outcome['checks']:
        print(row['t'],row['original_max_force_reproduction_error_N'],row['comparisons']['original']['first_ax_m_s2'],
              row['comparisons']['S5-r55']['first_ax_m_s2'])
        if row['original_max_force_reproduction_error_N']>1e-5:
            raise ValueError('Saved horizon could not be reconstructed at its frozen tolerance')
