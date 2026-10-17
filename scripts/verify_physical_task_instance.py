"""Closed-form math verification of one declared recovery/passage/docking task.

This is not a robot environment or a trained policy. No sampled rollout,
training batch, network, optimizer, simulator, or checkpoint is used. Constant
acceleration segments are integrated as exact polynomials and checked against
independent integral moments. The same-class TO is solved by monotonicity/KKT,
not by a gain sweep or a fitted controller parameter.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'notes/physical-task-instance-2026-10-17'
CONFIG = {'a':1., 'v0':1., 'b':1., 'L':4., 'gate_x':2.}
TOL = 1e-12  # Algebraic floating-point identity check, not a physical tolerance.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_time(r, *, a, v0, L):
    return v0/a*np.sqrt((1-r)/(1+r))+2*np.sqrt(L/a-(v0/a)**2*r/(2*(1+r)))


def task_derivative(r, *, a, v0, L):
    q=L/a-(v0/a)**2*r/(2*(1+r))
    return -v0/a*np.sqrt((1-r)/(1+r))/(1-r*r)-(v0/a)**2/(2*(1+r)**2*np.sqrt(q))


def feasible_allocation_limit(a, v0, b, L):
    if b < v0*v0/(2*a):
        raise ValueError('Even the declared lateral-first witness exceeds clearance; no claim of global infeasibility.')
    clearance=np.sqrt(1-(v0*v0/(2*a*b))**2)
    stopping=2*a*L/(v0*v0+2*a*L)
    return min(clearance,stopping), {'clearance_bound':clearance,'stopping_bound':stopping}


def scalar_extrema(p, v, u, duration):
    times=[0.,duration]
    if u != 0 and 0 < -v/u < duration:
        times.append(-v/u)
    values=[p+v*t+.5*u*t*t for t in times]
    return min(values),max(values)


def gate_crossing(p, v, u, duration, gate):
    if u == 0:
        roots=[] if v == 0 else [(gate-p)/v]
    else:
        roots=np.roots([.5*u,v,p-gate])
    real=[float(np.real(t)) for t in roots if abs(np.imag(t))<TOL and -TOL<=np.real(t)<=duration+TOL]
    return min(real) if real else None


def polynomial_witness(r, sign, cfg):
    a,v0,b,L,gate=[cfg[k] for k in ('a','v0','b','L','gate_x')]
    s=np.sqrt(1-r*r)
    recovery=v0/(a*s)
    x1=.5*a*r*recovery**2
    vx1=a*r*recovery
    peak_v=np.sqrt(a*(L-x1)+.5*vx1**2)
    accelerate=(peak_v-vx1)/a
    brake=peak_v/a
    assert accelerate>=-TOL
    phases=[('observed_disturbance_recovery',recovery,np.array([a*r,-sign*a*s])),
            ('forward_before_stopping_surface',accelerate,np.array([a,0.])),
            ('brake_on_stopping_surface',brake,np.array([-a,0.]))]
    p=np.zeros(2);v=np.array([0.,sign*v0]);total=0.;effort=0.;segments=[];gate_events=[]
    global_low=np.zeros(2);global_high=np.zeros(2)
    for label,duration,u in phases:
        # Exact integration of a declared constant-input polynomial, no time-step loop.
        next_p=p+duration*v+.5*duration**2*u
        next_v=v+duration*u
        low,high=zip(*(scalar_extrema(p[j],v[j],u[j],duration) for j in range(2)))
        global_low=np.minimum(global_low,low);global_high=np.maximum(global_high,high)
        cross=gate_crossing(p[0],v[0],u[0],duration,gate)
        if cross is not None and all(abs(total+cross-event['t'])>TOL for event in gate_events):
            gate_events.append({'t':total+cross,'y':float(p[1]+v[1]*cross+.5*u[1]*cross**2),
                                'vx':float(v[0]+u[0]*cross),'vy':float(v[1]+u[1]*cross)})
        segments.append({'phase':label,'start_time':total,'duration':float(duration),
                         'p_start':p.tolist(),'v_start':v.tolist(),'u':u.tolist(),
                         'p_end':next_p.tolist(),'v_end':next_v.tolist(),
                         'position_min':list(low),'position_max':list(high)})
        total+=duration;effort+=float(u@u)*duration;p,v=next_p,next_v
    # Independent full-interval input moments, without composing state transitions.
    moment_v=np.array([0.,sign*v0])
    moment_p=moment_v*total
    for segment in segments:
        u=np.array(segment['u']);dt=segment['duration'];start=segment['start_time']
        moment_v=moment_v+u*dt
        moment_p=moment_p+u*dt*(total-start-.5*dt)
    identity=max(np.max(np.abs(moment_p-p)),np.max(np.abs(moment_v-v)))
    max_force=max(np.linalg.norm(u) for _,_,u in phases)
    occupied=max(abs(global_low[1]),abs(global_high[1]))
    stopping_surface=L-segments[2]['p_start'][0]-segments[2]['v_start'][0]**2/(2*a)
    pass_checks={'terminal_x':abs(p[0]-L)<=TOL,'terminal_velocity':np.linalg.norm(v)<=TOL,
                 'whole_path_clearance':occupied<=b+TOL,'input_ball':max_force<=a+TOL,
                 'gate_crossed':len(gate_events)>0,'nonnegative_forward_path':global_low[0]>=-TOL,
                 'moment_identity':identity<=TOL,'time_identity':abs(total-task_time(r,a=a,v0=v0,L=L))<=TOL,
                 'stopping_surface_identity':abs(stopping_surface)<=TOL}
    pass_checks={key:bool(value) for key,value in pass_checks.items()}
    assert all(pass_checks.values())
    return {'r':float(r),'observed_impulse_sign':sign,'duration':float(total),'recovery_duration':float(recovery),
            'occupied_lateral_clearance':float(occupied),'squared_acceleration_integral':float(effort),
            'resource_units':'acceleration^2 * time; NOT measured electrical energy or actuator torque loss',
            'terminal_position':p.tolist(),'terminal_velocity':v.tolist(),'gate_events':gate_events,
            'max_input_norm':float(max_force),'moment_identity_error':float(identity),
            'checks':pass_checks,'segments':segments}


def packet_credit(r, *, a, v0, L):
    """One exact phase response and a finite downstream value covector."""
    s=np.sqrt(1-r*r);tau=v0/(a*s)
    x=.5*v0*v0/a*r/(1-r*r);vx=v0*r/s
    response=np.array([tau,x,vx,v0*v0/(2*a*s)])  # time, x, vx, |y|
    jac=np.array([v0/a*r/s**3,.5*v0*v0/a*(1+r*r)/(1-r*r)**2,
                  v0/s**3,.5*v0*v0/a*r/s**3])
    q=(L-x)/a+vx*vx/(2*a*a)
    covector=np.array([1.,-1/(a*np.sqrt(q)),vx/(a*a*np.sqrt(q))-1/a,0.])
    gradient=float(covector@jac)
    complex_step=float(np.imag(task_time(r+1e-20j,a=a,v0=v0,L=L))/1e-20)
    direct=float(task_derivative(r,a=a,v0=v0,L=L))
    assert abs(gradient-direct)<TOL and abs(complex_step-direct)<TOL
    return {'anchor_r':r,'response_order':['recovery_time','x_end','vx_end','abs_y_end'],
            'response':response.tolist(),'response_derivative':jac.tolist(),
            'downstream_task_covector':covector.tolist(),'contract_covector_clearance':[0.,0.,0.,1.],
            'contract_derivative_clearance':float(jac[3]),'local_recovery_only_derivative':float(jac[0]),
            'complete_task_derivative_by_packet':gradient,'complete_task_direct_derivative':direct,
            'complex_step_identity_derivative':complex_step,
            'interpretation':'Model/physical-task credit in one declared finite feedback coordinate; not a PPO batch or a robot NN gradient.'}


def main():
    start=time.perf_counter();cfg=CONFIG.copy()
    protocol=OUT/'protocol.md';protocol_before=sha(protocol)
    r,bounds=feasible_allocation_limit(**{k:cfg[k] for k in ('a','v0','b','L')})
    cases={}
    for label,allocation in [('frozen_lateral_first',0.),('geometry_derived_joint_feedback',r)]:
        cases[label]=[polynomial_witness(allocation,sign,cfg) for sign in (-1,1)]
    old=cases['frozen_lateral_first'][0];new=cases['geometry_derived_joint_feedback'][0]
    gradient=task_derivative(r,**{k:cfg[k] for k in ('a','v0','L')})
    # T'(r)<0 on the proven feasible interval; the standard constrained TO
    # therefore chooses the same active endpoint and has multiplier -T'(r).
    to_r=min(bounds.values());to_t=task_time(to_r,**{k:cfg[k] for k in ('a','v0','L')})
    assert abs(to_r-r)<=TOL and abs(to_t-new['duration'])<=TOL
    credit=packet_credit(0.,**{k:cfg[k] for k in ('a','v0','L')})
    assert sha(protocol)==protocol_before
    report={'date':'2026-10-17','status':'completed_analytic_verification',
            'protocol_sha256':protocol_before,'script_sha256':sha(__file__),
            'config':cfg,'units':'Explicit normalized mathematical construction, not robot measurements',
            'assumptions':['Known exact planar acceleration model with closed input ball',
                           'Initial transverse impulse already observed; no subsequent disturbance',
                           'Free lateral endpoint within clearance band, NOT y=0 docking',
                           'Continuous ideal feedback switching; no claims for unmeasured robot sampling/actuator dynamics'],
            'allocation_limits':bounds,'cases':cases,
            'comparison':{'frozen_with_exact_model_and_optimal_suffix_time':old['duration'],
                          'finite_feedback_change_time':new['duration'],
                          'time_reduction_fraction':1-new['duration']/old['duration'],
                          'clearance_occupancy_change':new['occupied_lateral_clearance']-old['occupied_lateral_clearance'],
                          'same_class_standard_TO_allocation':float(to_r),'same_class_standard_TO_time':float(to_t),
                          'same_class_time_difference':float(to_t-new['duration']),
                          'same_class_TO_upper_bound_multiplier':float(-gradient),
                          'model_only_optimism_if_old_behavior_is_mislabeled_as_new':old['duration']-new['duration'],
                          'same_class_algorithm_advantage_established':False,
                          'general_time_varying_free_control_optimum_computed':False},
            'credit_packet':credit,
            'budget':{'signed_policy_cases':4,'exact_polynomial_segment_maps':12,
                      'independent_terminal_moment_checks':4,'packet_gradient_checks':1,
                      'complex_analytic_expression_checks':1,'gain_sweep_points':0,
                      'numerical_optimizer_calls':0,'robot_environment_rollouts':0,'RL_batches':0,
                      'actor_weight_updates':0,'checkpoint_reads':0,'checkpoint_writes':0,
                      'wall_seconds':time.perf_counter()-start},
            'decision':'Finite causal feedback can trade clearance for time in this analytic task, but standard same-class TO gives the same solution and credit. No independent learning/algorithm advantage or robot capability established.'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'analytic-results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('comparison','credit_packet','budget')},indent=2))


if __name__=='__main__':
    main()
