"""Continuous R7 mission and evidence derived from the actual 500 Hz record."""
import numpy as np


def wrap(x):
    return np.arctan2(np.sin(x),np.cos(x))


def stable_at(pos,velocity,ori,omega,standing,endpoint,c):
    return bool(standing and np.linalg.norm(pos[:2]-endpoint)<c['goal_position_radius_m'] and
        np.linalg.norm(velocity)<c['terminal_speed_max_m_s'] and
        np.linalg.norm(omega)<.1 and np.max(np.abs(ori[:2]))<.15)


class ContinuousMission:
    """Five uninterrupted round trips, with actual stable stops at both ends."""
    def __init__(self,task,protocol):
        self.task,self.protocol=task,protocol
        self.phase='initial_stand'; self.stable_since=None; self.final_stable_s=0.
        self.v=np.zeros(3); self.w=np.zeros(3); self.body_target=np.array([0.,0.,.27])
        self.heading=0.; self.turn_target=0.; self.laps=0
        self.transitions=[]; self.arrivals=[]; self.arrival_time=None

    def update(self,t,dt,pos,velocity,ori,omega,standing):
        c=self.task; old=self.phase
        far=np.array(c['turnaround_xy_m']); home=np.array(c['return_xy_m'])
        endpoint=far if self.phase in ('outbound','far_hold','turn') else home
        distance=np.linalg.norm(endpoint-pos[:2])
        if self.phase=='initial_stand' and t>=c['initial_stand_s']:
            self.phase='outbound'
        elif self.phase=='outbound' and distance<c['goal_position_radius_m'] and np.linalg.norm(velocity[:2])<.06:
            self.phase='far_hold'; self.arrival_time=t; self.stable_since=None
            self.body_target[:2]=far
        elif self.phase=='return' and distance<.06 and np.linalg.norm(velocity[:2])<.05:
            self.phase='home_hold'; self.arrival_time=t; self.stable_since=None
            self.body_target[:2]=home
        elif self.phase in ('far_hold','home_hold','complete'):
            good=stable_at(pos,velocity,ori,omega,standing,endpoint,c)
            self.stable_since=(t if self.stable_since is None else self.stable_since) if good else None
            streak=t-self.stable_since if self.stable_since is not None else 0.
            if self.phase=='complete':
                self.final_stable_s=streak
            elif streak>=c['terminal_stand_s']:
                self.arrivals.append(dict(endpoint='far' if self.phase=='far_hold' else 'home',
                    lap=self.laps+1, arrival_s=self.arrival_time, stable_completion_s=t,
                    stopping_s=t-self.arrival_time, position_m=pos.copy(), heading_rad=float(ori[2])))
                if self.phase=='far_hold':
                    self.turn_target=2*self.laps*np.pi+np.pi; self.phase='turn'
                else:
                    self.laps+=1
                    if self.laps==self.protocol['full_round_trips']:
                        self.phase='complete'; self.final_stable_s=streak
                    else:
                        self.turn_target=2*self.laps*np.pi; self.phase='turn_home'
                if self.phase!='complete': self.stable_since=None
        elif self.phase in ('turn','turn_home') and abs(wrap(self.turn_target-ori[2]))<c['turn_heading_tolerance_rad']:
            self.phase='return' if self.phase=='turn' else 'outbound'
        if old!=self.phase or not self.transitions:
            self.transitions.append(dict(time_s=float(t),phase=self.phase,lap=self.laps,
                position_m=pos.copy(),velocity_m_s=velocity.copy(),rpy_rad=ori.copy(),
                omega_rad_s=omega.copy(),all_feet_contact=bool(standing)))
        desired=np.zeros(3); yaw_rate=0.
        if self.phase in ('outbound','return'):
            endpoint=far if self.phase=='outbound' else home
            delta=endpoint-pos[:2]
            desired[:2]=delta*min(.8,c['command_speed_m_s']/max(np.linalg.norm(delta),1e-12))
            yaw_goal=2*self.laps*np.pi+(np.pi if self.phase=='return' else 0.)
            yaw_rate=np.clip(1.5*wrap(yaw_goal-ori[2]),-.5,.5)
            self.heading=min(yaw_goal,self.heading+c['turn_yaw_rate_rad_s']*dt)
        elif self.phase in ('turn','turn_home'):
            endpoint=far if self.phase=='turn' else home
            desired[:2]=np.clip(.8*(endpoint-pos[:2]),-.1,.1)
            yaw_rate=min(c['turn_yaw_rate_rad_s'],1.5*((self.turn_target-ori[2])%(2*np.pi)))
        self.v+=np.clip(desired-self.v,-.5*dt,.5*dt)
        self.w[2]+=np.clip(yaw_rate-self.w[2],-dt,dt)
        if self.phase in ('initial_stand','far_hold','home_hold','complete'):
            endpoint=far if self.phase=='far_hold' else home
            self.v[:]=np.r_[np.clip(.8*(endpoint-pos[:2]),-.1,.1),0.]
            self.w[:]=0.
        if self.phase in ('outbound','return'):
            self.body_target[0]=np.clip(self.body_target[0]+self.v[0]*dt,0.,3.)
        if self.phase in ('turn','turn_home'):
            self.heading=min(self.turn_target,self.heading+max(0.,self.w[2])*dt)
        return self.body_target.copy(),self.heading


def hold_metrics(a,first_completion,duration,mission):
    t=a['time']; q=a['qpos']; v=a['qvel']; r=a['rpy']
    checks=dict(contact=a['contact'].all(axis=1),position=np.linalg.norm(q[:,:2],axis=1)<.15,
        linear_speed=np.linalg.norm(v[:,:3],axis=1)<.05,
        angular_speed=np.linalg.norm(v[:,3:6],axis=1)<.1,
        roll_pitch=np.max(np.abs(r[:,:2]),axis=1)<.15)
    good=np.logical_and.reduce(list(checks.values()))
    start=first_completion if first_completion is not None else (duration-30 if mission=='long' else None)
    if start is None:
        terminal=np.flatnonzero(a['phase']==4)
        start=float(t[terminal[0]]) if len(terminal) else None
    selected=(t>=start-1e-8) if start is not None else np.zeros(len(t),dtype=bool)
    times=t[selected]; valid=good[selected]
    if not len(times):
        return dict(scope='terminal_hold_not_observed',start_s=start,observed_samples=0,
            observed_through_s=None,full_observation=False,all_criteria_fraction=None,
            endpoint_position_violation_samples=None,violation_samples=None,interruptions=None,
            longest_stable_s=None,final_stable_s=None,criterion_pass_fraction={k:None for k in checks},
            max_speed_m_s=None,max_omega_rad_s=None,max_origin_distance_m=None,stable_intervals_s=[])
    intervals=[]; since=None
    for j,(tm,ok) in enumerate(zip(times,valid)):
        if ok and since is None: since=float(tm)
        if since is not None and (not ok or j==len(times)-1):
            end=float(tm if ok else times[j-1]); intervals.append([since,end]); since=None
    lengths=[b-e for e,b in intervals]
    final_stable=(times[-1]-intervals[-1][0]) if len(times) and valid[-1] else 0.
    return dict(start_s=float(start),observed_samples=int(selected.sum()),
        observed_through_s=float(t[-1]),full_observation=bool(t[-1]>=duration-1e-7),
        all_criteria_fraction=float(valid.mean()) if len(valid) else None,
        endpoint_position_violation_samples=int((~checks['position'][selected]).sum()),
        violation_samples=int((~valid).sum()),interruptions=int(np.sum(valid[:-1]&~valid[1:])),
        longest_stable_s=float(max(lengths,default=0.)),final_stable_s=float(final_stable),
        criterion_pass_fraction={k:float(x[selected].mean()) if len(valid) else None for k,x in checks.items()},
        max_speed_m_s=float(np.linalg.norm(v[selected,:3],axis=1).max()) if len(valid) else None,
        max_omega_rad_s=float(np.linalg.norm(v[selected,3:6],axis=1).max()) if len(valid) else None,
        max_origin_distance_m=float(np.linalg.norm(q[selected,:2],axis=1).max()) if len(valid) else None,
        stable_intervals_s=intervals)
