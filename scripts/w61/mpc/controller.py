"""Execute prospectively selected diagonal transfers through the original force QP and WBC."""
import json
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from s1_w54_f_r52_backend import STAND_PHASES,assemble_force_horizon
from .problem import build_problem,DT,H,MODES
from .solver import search


class TopologyMPC:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.planner_mode='P1'
        self.c_last_event=None
        self.c_solution=None
        self.c_count=0
        self.c_refine_count=0
        self.c_warm=np.zeros((2,16))
        self.c_dir=Path(self.cfg['w61']['out'])
        self.c_method=self.cfg['w61']['role']
        self.c_cfg=self.cfg['w61']['mpc_transfer']
        self.c_commands=(self.c_dir/'c-plans.jsonl').open('a')

    def _update_schedule(self,t,data,mq,mission,soft,material_info,observed,grf):
        # Close the decision as soon as a transfer starts; an airborne foot never
        # becomes an alternative liftoff. Only simultaneous diagonals are
        # executable by the frozen r55 allocator; staggered modes stay offline.
        mark=self.last_liftoff if self.last_liftoff is not None else -1.
        if (mission.phase not in STAND_PHASES and mark!=self.c_last_event and
            np.all(self.event_mode=='stance') and np.all(self.confirmed & observed)):
            self.c_last_event=mark
            x,ref=self._body_reference(t,mq,mission)
            obs=dict(x0=x,reference=ref,feet=mq['contact_force_point'][None],contact=np.array(observed)[None],
                     h=mq['h'],inertia=mq['inertia'],mass=self.mass,J_contact_force=mq['Jforce'],
                     torque_lower=self.model.actuator_forcerange[:,0],torque_upper=self.model.actuator_forcerange[:,1],
                     last_force=self.last_force)
            problem=build_problem(obs,float(np.linalg.norm(mission.v[:2])),np.ones(4),self.c_cfg['belief'],0.)
            sol=search(problem,self.c_method,self.c_cfg['seed']+self.c_count,self.c_warm,
                       candidates=self.c_cfg['samples'],iterations=self.c_cfg['rounds'],mode_ids=(0,1))
            self.c_warm=sol['per_mode']
            self.next_pair=np.asarray(((0,3),(1,2))[sol['mode']])
            self.c_solution=(float(t),sol['force_reference'])
            self.c_count+=1
            self.c_commands.write(json.dumps(dict(event=self.c_count,physical_s=float(t),mode=MODES[sol['mode']],
                per_mode_cost=sol['per_mode_cost'].tolist(),latent=sol['z'].tolist(),candidates=sol['evaluated_candidates'],
                prediction_torque_residual=sol['maximum_torque_violation_Nm'],search_wall_s=sol['search_wall_s'],
                synchronized_batch_times=sol['transfer_rollout_refinement_sync_s']))+'\n')
            self.c_commands.flush()
        return super()._update_schedule(t,data,mq,mission,soft,material_info,observed,grf)

    def select_force_plan(self,t,mission,mq,times,foot_plan,contact,inputs,stages,states,forces,status):
        if self.c_solution is None or t-self.c_solution[0]>=H*DT:
            return stages,states,forces,status
        issued,preference=self.c_solution
        age=times[:-1]-issued
        valid=(age>=-1.e-9)&(age<H*DT)
        ix=np.clip(np.floor(age/DT+1e-8).astype(int),0,H-1)
        nx=states.size
        qp=assemble_force_horizon(inputs,self.cfg)
        weight=self.c_cfg['force_tracking_weight']*np.repeat(valid,12)
        qp['P']=qp['P']+sparse.diags(np.r_[np.zeros(nx),2*weight],format='csc')
        qp['q'][nx:]-=2*weight*preference[ix].ravel()
        start=time.perf_counter()
        z,info=self._solve(qp['P'],qp['q'],qp['A'],qp['lower'],qp['upper'],'mpc',t,
                           warm=np.r_[states.ravel(),forces.ravel()])
        self.c_refine_count+=1
        info['topology_preference_solve_s']=time.perf_counter()-start
        info['first_F_difference_N']=float(np.linalg.norm(z[nx:nx+12]-forces[0].ravel()))
        return stages,z[:nx].reshape(states.shape),z[nx:].reshape(forces.shape),info

    def at_rollout_end(self,**kwargs):
        self.c_commands.close()
        (self.c_dir/'c-activity.json').write_text(json.dumps(dict(plans=self.c_count,force_QP_projected=self.c_refine_count))+'\n')
