"""S5-r55 standing centroidal moment linearization at observed robot motion."""
import json
import time
from pathlib import Path

import numpy as np

from s1_w54_f_r52_backend import assemble_centroidal, assemble_force_horizon, euler_world_rate, skew
from s1_w55_f_r53_backend import FController as R53Controller

ENDPOINT_HOLDS = ('terminal_stand', 'complete', 'far_hold', 'home_hold')


def standing_stages(mass, inertia, x0, feet, support, times, dt, friction):
    """Linearize sum((foot-COM) x force) in both COM and support force.

    The expansion point continues the observed COM with its measured velocity.
    Its supported force is the balancing vertical weight, zero in flight. The
    goal remains solely a state cost. At fixed nominal force the body-position
    derivative is inv(I) @ skew(total_force).
    """
    centers = x0[3:6] + (times[:-1]-times[0])[:,None]*x0[9:12]
    balancing = np.zeros((len(centers),3))
    balancing[:,2] = support.any(axis=1) * (mass*9.81)
    euler = euler_world_rate(x0[:3])
    stages = []
    for center, points, net_force in zip(centers,feet,balancing):
        stage = assemble_centroidal(mass,inertia,center,points,x0[:3],x0[6:9],dt,friction)
        derivative = np.linalg.solve(inertia,skew(net_force))
        angle_term = .5*dt**2 * (euler @ derivative)
        omega_term = dt * derivative
        stage['A'][:3,3:6] += angle_term
        stage['A'][6:9,3:6] += omega_term
        stage['c'][:3] -= angle_term @ center
        stage['c'][6:9] -= omega_term @ center
        stages.append(stage)
    return stages,centers,balancing


class FController(R53Controller):
    def __init__(self, model, data, feet, base, out, config=None):
        if config is None:
            config_path = Path(__file__).resolve().parents[1]/'experiments/terrain_memory/s1-w55-f-r55.json'
            config = json.loads(config_path.read_text())
        super().__init__(model,data,feet,base,out,config)
        self.manifest.update(endpoint_moment_linearization=config['revision']['endpoint_moment_linearization'])
        self._write_manifest()

    def replan(self,t,data,mq,mission):
        # Walking and original initial stand keep the r53 execution exactly.
        if mission.phase not in ENDPOINT_HOLDS:
            return super().replan(t,data,mq,mission)
        w,c=time.perf_counter(),time.process_time()
        n=self.cfg['mpc']['intervals'];dt=self.cfg['mpc']['dt_s']
        x0,ref=self._body_reference(t,mq,mission)
        times=t+np.arange(n+1)*dt
        contact,foot_plan=self._future_contact_feet(t,times,mq,mission)
        force_foot_plan=foot_plan.copy();force_foot_plan[:,:,2]-=self.radius[None,:]
        force_foot_plan[0]=mq['contact_force_point']
        stages,centers,balancing=standing_stages(
            self.mass,mq['inertia'],x0,force_foot_plan[:-1],contact[:-1],times,dt,self.cfg['force_pyramid_slope'])
        inputs=dict(moment_linearization_com=centers,moment_linearization_net_force=balancing,
            phase=np.array(mission.phase),x0=x0,reference=ref,last_force=self.last_force.copy(),contact=contact[:-1],
            feet=force_foot_plan,force_normal_upper=np.full((n,4),np.inf),
            stage_A=np.array([x['A'] for x in stages]),stage_B=np.array([x['B'] for x in stages]),
            stage_c=np.array([x['c'] for x in stages]))
        qp=assemble_force_horizon(inputs,self.cfg)
        self._time('matrix_assembly',w,c);w,c=time.perf_counter(),time.process_time()
        z,status=self._solve(qp['P'],qp['q'],qp['A'],qp['lower'],qp['upper'],'mpc',t)
        self._time('mpc_solve',w,c)
        nx=(n+1)*12;states=z[:nx].reshape(n+1,12);forces=z[nx:].reshape(n,4,3)
        self.last_plan=dict(time=t,times=times,state=states,reference=ref,forces=forces,
            feet=foot_plan,contact=contact[:-1],blocks=stages[0],status=status)
        self.latest_plan=dict(times=times,query_feet=foot_plan.copy(),source='F causal event horizon; predicted touchdown is not future observation')
        self.plan_count+=1;w,c=time.perf_counter(),time.process_time()
        path=self.record_dir/f'mpc-{self.plan_count:06d}.npz'
        np.savez_compressed(path,times=times,states=states,state_reference=ref,forces=forces,
            contact_schedule=contact[:-1],inertia=mq['inertia'],com=mq['com'],
            mass=np.array(self.mass),M=mq['M'],h=mq['h'],J=mq['J'],
            foot_center_plan=foot_plan,J_contact_force=mq['Jforce'],J_rolling=mq['Jroll'],
            contact_point_source=mq['contact_point_source'],qpos=data.qpos.copy(),qvel=data.qvel.copy(),
            event_mode=self.event_mode.copy(),confirmed_contact=self.confirmed.copy(),
            load_bearing_contact=self.load_bearing.copy(),stance_anchor=self.anchor.copy(),
            event_number=np.array(self.event_number),**inputs)
        with (self.record_dir/'mpc-solves.jsonl').open('a') as f:
            f.write(json.dumps(dict(call=self.plan_count,time_s=t,file=path.name,**status))+'\n')
        self.manifest['full_mpc_files'].append(dict(call=self.plan_count,time_s=t,path=path.name,size=path.stat().st_size))
        self._time('record_serialization_io',w,c)
