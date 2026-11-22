"""Force planning with private causal world-cell histories and the unchanged centroidal cost."""
import ctypes as ct
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np

from s1_w54_f_r52_backend import (ControllerNumericalFailure, assemble_force_horizon,
                                  euler_world_rate, skew)
from s1_w60_cpu_observer import ARRAY, INDEX
from tm02.s1_contact import StanceReferenceInfeasible


@lru_cache(maxsize=1)
def native():
    path = Path(__file__).resolve().parents[1]/'work/w57-p/candidate.so'
    lib = ct.CDLL(str(path))
    lib.w57_candidate.argtypes = [ARRAY,ct.c_int,ct.c_int,ARRAY,INDEX,ARRAY,ARRAY,ARRAY,
                                  ARRAY,ARRAY,ct.c_int,ARRAY]
    return lib


def footprint(g, xy):
    """Public world geometry, including virgin cells, with no creation on the observer."""
    if not g.interior(xy):
        return {}
    lo, hi = xy - g.half, xy + g.half
    a, b = np.floor(lo/g.cell).astype(int), np.floor(hi/g.cell).astype(int)
    out = {}
    for ix in range(a[0], b[0]+1):
        dx = min(hi[0], (ix+1)*g.cell) - max(lo[0], ix*g.cell)
        for iy in range(a[1], b[1]+1):
            dy = min(hi[1], (iy+1)*g.cell) - max(lo[1], iy*g.cell)
            if dx > 0 and dy > 0:
                out[(ix,iy)] = dx*dy
    return out


def protected_footprints(g, bottom, measured):
    if bottom is None:
        return set()
    maps = [set(footprint(g, p[:2])) for p in bottom]
    current = set(np.flatnonzero(measured))
    current.update(int(i) for i in g.last_owner[g.loaded] if i>=0)
    protected = set().union(*(maps[i] for i in current))
    # Include complete footprints transitively for any current shared ownership group.
    for _ in range(len(maps)):
        connected = set().union(*(m for m in maps if m & protected))
        if connected == protected:
            break
        protected = connected
    return protected


class Forecast:
    def __init__(self, g, times, intervals, xy, contact, published_bottom, measured, node_xy=None):
        self.g = g
        self.times = np.array(times)
        self.dt = np.ascontiguousarray(np.diff(intervals))
        self.segment = np.searchsorted(times[1:], intervals[:-1]+1.e-10)
        self.xy = np.array(xy)
        self.node_xy = self.xy if node_xy is None else np.asarray(node_xy)
        self.contact = np.array(contact)
        shape = (len(xy), 4)
        self.maps = [[footprint(g, p) if use else {} for p,use in zip(points, on)]
                     for points,on in zip(xy, contact)]
        self.keys = sorted({k for maps in self.maps for m in maps for k in m})
        rows = {key:i for i,key in enumerate(self.keys)}
        self.area = np.zeros((*shape,len(rows)))
        for j, maps in enumerate(self.maps):
            for leg, m in enumerate(maps):
                for key, value in m.items():
                    self.area[j,leg,rows[key]] = value
        self.active = self.area.sum(axis=2)>0.
        self.initial = np.zeros((len(rows),6))
        protected = protected_footprints(g, published_bottom, measured)
        self.protected = np.array([k in protected for k in self.keys],dtype=bool)
        self.reset = np.zeros(len(rows),dtype=bool)
        for i,key in enumerate(self.keys):
            if key in g.ids:
                src=g.ids[key]
                self.initial[i]=np.r_[g.e[src],g.q[src],g.stress[src]]
                self.reset[i]=g.ever_loaded[src] and not self.protected[i]
        self.published_indices = np.array([g.ids.get(k,-1) for k in self.keys],int)

    def propagate(self, forces, h, history='full', tangents=False):
        subdivisions = np.ascontiguousarray(np.maximum(1, np.ceil(self.dt/h-1.e-8)), dtype=np.int32)
        # QP's tiny open-contact/negative residuals use the r60 unilateral WBC convention.
        loads = np.ascontiguousarray(np.maximum(forces[self.segment,:,2],0.))
        x = self.initial.copy()
        if history == 'private_reset':
            x[self.reset] = 0.
        elif history != 'full':
            raise ValueError(f'Unexpected history intervention {history}')
        answer = np.zeros((len(self.dt),4,4))
        states = np.empty((len(self.dt),len(self.keys),6))
        detail = np.zeros(3)
        code = native().w57_candidate(self.g.law.model,len(self.keys),len(self.dt),self.dt,
                                      subdivisions,self.area,loads,x,answer,states,int(tangents),detail)
        if code:
            labels={1:'constitutive_solve',2:'strain_domain',3:'requested_load_domain_or_inverse',
                    4:'load_residual',5:'no_consistent_footprint_ownership'}
            raise StanceReferenceInfeasible(dict(reason=labels[code],segment=int(detail[0]),
                                                subdivision=int(detail[1]),dt_s=float(detail[2])))
        return dict(z=answer[:,:,0], mean_z=answer[:,:,1], tangent=answer[:,:,2],
                    inverse_residual=answer[:,:,3], states=states,steps=int(subdivisions.sum()))

    def linearization(self, inputs, inertia, prediction):
        points=inputs['feet'].copy()
        B=inputs['stage_B'].copy()
        active_dt=self.active*self.dt[:,None]
        duration=np.zeros((len(B),4)); zsum=np.zeros_like(duration)
        np.add.at(duration,self.segment,active_dt)
        np.add.at(zsum,self.segment,active_dt*prediction['mean_z'])
        for j,dt in enumerate(np.diff(self.times)):
            for leg in np.flatnonzero(inputs['contact'][j]):
                if duration[j,leg]==0.:
                    if self.g.interior(self.node_xy[j,leg]):
                        B[j,:,3*leg:3*leg+3]=0.
                    continue
                fraction=min(1.,duration[j,leg]/dt)
                # The first moment arm is the measured reaction point, before the new input.
                z=points[j,leg,2] if j==0 else zsum[j,leg]/duration[j,leg]
                dz=z-points[j,leg,2];points[j,leg,2]=z
                cols=slice(3*leg,3*leg+3)
                G=np.linalg.solve(inertia,skew((0.,0.,dz)))
                B[j,:3,cols]+=.5*dt*dt*(euler_world_rate(inputs['x0'][:3])@G)
                B[j,6:9,cols]+=dt*G
                B[j,:,cols]*=fraction
        return dict(inputs, stage_B=B, feet=points)


def original_cost(inputs, cfg, force):
    state = np.empty_like(inputs['reference'])
    state[0]=inputs['x0']
    for k, (a,b,c) in enumerate(zip(inputs['stage_A'],inputs['stage_B'],inputs['stage_c'])):
        state[k+1]=a@state[k]+b@force[k].ravel()+c
    err=(state-inputs['reference'])**2
    err[-1]*=2.
    terms=np.array([(err*np.asarray(cfg['mpc']['state_weights'])).sum(),
                    cfg['mpc']['force_regularization']*(force**2).sum(),
                    cfg['mpc']['force_difference_weight']*(np.diff(force,axis=0,prepend=inputs['last_force'][None])**2).sum()])
    return float(terms.sum()), terms, state


def rank(forecast, inputs, inertia, cfg, nominal, solve, h, history, tangents=False):
    a=forecast.propagate(nominal,h,history,tangents)
    linear=forecast.linearization(inputs,inertia,a)
    cost0,terms0,x0=original_cost(linear,cfg,nominal)
    try:
        corrected_z,status=solve(assemble_force_horizon(linear,cfg))
        nx=inputs['reference'].size
        corrected=corrected_z[nx:].reshape(nominal.shape)
        # Both scores use the same force-to-material-to-centroidal evaluator.
        b=forecast.propagate(corrected,h,history,tangents)
    except (StanceReferenceInfeasible, ControllerNumericalFailure) as error:
        # Nominal has already passed the same evaluator and remains the sole eligible plan.
        return dict(cost=[cost0,None],winner=0,predictions=[a,None],
                    corrected_rejection=getattr(error,'detail',str(error)),terms=[terms0,None],
                    scored_states=[x0,None])
    own=forecast.linearization(inputs,inertia,b)
    cost1,terms1,x1=original_cost(own,cfg,corrected)
    winner=int(cost1<cost0-1.e-12*max(1.,abs(cost0),abs(cost1)))
    return dict(cost=[cost0,cost1],terms=[terms0,terms1],scored_states=[x0,x1],
                winner=winner,linear=linear,corrected=corrected,
                qp_states=corrected_z[:nx].reshape(inputs['reference'].shape),status=status,
                predictions=[a,b])


def select_online(controller,t,mission,mq,times,foot_plan,contact,inputs,stages,states,forces,status):
    g=controller.current_material_info['material_observer'].ground
    if not any(g.interior(p) for p in foot_plan[:,:,:2].reshape(-1,2)):
        return stages,states,forces,status
    obs = controller.current_material_info['material_observer']
    if abs(obs.time-t)>1.e-9:
        raise ValueError(f'Candidate consumed publication {obs.time} at {t}')
    begin, cpu=time.perf_counter(),time.process_time()
    events=controller._forecast_events(t,float(times[-1]),mq,mission)
    breaks=sorted({float(x) for leg in events for a,b,_ in leg for x in (a,b)
                   if times[0]<x<times[-1] and np.min(np.abs(times-x))>1.e-8})
    grid=np.sort(np.r_[times,breaks])
    # A kinematic interval beginning exactly on a contact switch uses its new mode.
    mids=.5*(grid[:-1]+grid[1:])
    available, feet=controller._future_contact_feet(t,mids,mq,mission)
    forecast=Forecast(g,times,grid,feet[:,:,:2],available,obs.published_bottom,controller.measured,foot_plan[:-1,:,:2])
    if not len(forecast.keys):
        return stages,states,forces,status
    def solve(qp):
        return controller._solve(qp['P'],qp['q'],qp['A'],qp['lower'],qp['upper'],'w57-material-mpc',t)
    h=controller.cfg['w57']['candidate_substep_s']
    record=dict(call=controller.plan_count+1,time_s=float(t),observer_publication=int(obs.windows),
                observer_time_s=float(obs.time),current_control_event=int(controller.event_number),
                history=controller.planner_history,private_substep_max_s=h,force_point_clock='interval mean; measured point at k=0',
                requested_handover_s=controller.next_liftoff,source='nominal_r60_and_one_material_relinearization')
    try:
        ranking=rank(forecast,inputs,mq['inertia'],controller.cfg,forces,solve,h,controller.planner_history)
    except StanceReferenceInfeasible as exc:
        # Identical hard constitutive evaluator is mandatory even for the original candidate.
        # A failed common rollout is reported as a controller outcome, never accepted silently.
        record.update(rejection=exc.detail,selected=None)
        with (controller.record_dir/'material-candidates.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        raise ControllerNumericalFailure(f'W57 material forecast at {t}: {exc.detail}') from exc
    winner=ranking['winner']
    if 'corrected_rejection' in ranking:
        record.update(selected='nominal',original_objective=ranking['cost'],
                      corrected_rejection=ranking['corrected_rejection'],num_world_cells=len(forecast.keys),
                      detached_history=int(forecast.reset.sum()),candidate_wall_s=time.perf_counter()-begin,
                      candidate_cpu_s=time.process_time()-cpu)
        with (controller.record_dir/'material-candidates.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        controller.w57_new_plan_call=controller.plan_count+1
        return stages,states,forces,status
    a,b=ranking['predictions']
    selector=int(np.argmax(np.linalg.norm(ranking['corrected'][0]-forces[0],axis=1)))
    record.update(selected=('nominal','material')[winner],original_objective=ranking['cost'],
                  objective_terms=[term.tolist() for term in ranking['terms']],
                  first_force_N=[forces[0].tolist(),ranking['corrected'][0].tolist()],
                  max_first_force_difference_N=float(np.max(np.abs(ranking['corrected'][0]-forces[0]))),
                  max_normal_depth_m=float(np.max(np.abs(a['z']))),num_world_cells=len(forecast.keys),
                  present_protected=int(forecast.protected.sum()),detached_history=int(forecast.reset.sum()),
                  candidate_first_difference_leg=selector,private_steps=a['steps']+b['steps'],
                  candidate_wall_s=time.perf_counter()-begin,candidate_cpu_s=time.process_time()-cpu)
    name=f'material-plan-{controller.plan_count+1:06d}.npz'
    record['data']=name
    np.savez_compressed(controller.record_dir/name,segment_grid_s=grid,xy=forecast.xy,active=forecast.active,
                        world_keys=np.array(forecast.keys),initial=forecast.initial,protected=forecast.protected,
                        detached_history=forecast.reset,footprint_area_m2=forecast.area,nominal_force=forces,
                        candidate_force=ranking['corrected'],full_sole_z_m=a['z'],candidate_sole_z_m=b['z'],
                        selected_common_states=ranking['scored_states'][winner],
                        selected_cell_states=ranking['predictions'][winner]['states'])
    with (controller.record_dir/'material-candidates.jsonl').open('a') as stream:
        stream.write(json.dumps(record,allow_nan=False)+'\n')
    controller.w57_new_plan_call=controller.plan_count+1
    if not winner:
        return stages,states,forces,status
    inputs.update(ranking['linear'])
    installed=[dict(stage,B=inputs['stage_B'][i]) for i,stage in enumerate(stages)]
    return installed,ranking['qp_states'],ranking['corrected'],ranking['status']
