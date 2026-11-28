"""Two event-committed Go2 orbit charts, with a shared C2 actuator-facing reference."""
import json
from pathlib import Path

import numpy as np
from s1_w54_f_r52_backend import STAND_PHASES, curve, quintic

CHARTS={'trot':((1,2),(0,3)), 'crawl':((1,),(2,),(0,),(3,))}
ZERO=np.zeros(3)


def segment(t,p,v,landing,duration,height,normal):
    apex=.5*duration
    mid=.5*(p+landing);mid[2]=max(p[2],landing[2])+height
    vmid=(landing-p)/duration;vmid[2]=0.
    return dict(start=t,end=t+duration,apex=t+apex,landing=landing.copy(),
        rise=quintic(p,v,ZERO,mid,vmid,ZERO,apex),
        fall=quintic(mid,vmid,ZERO,landing,np.array([0.,0.,normal]),ZERO,duration-apex))


def state_at(s,t):
    u=np.clip(t,s['start'],s['end'])
    if u<=s['apex']:
        return curve(s['rise'],u-s['start'])
    return curve(s['fall'],u-s['apex'])


class ControlledOrbit:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        spec=self.cfg['w61']
        self.gait_role=spec.get('gait_role',spec['role'])
        self.chart='trot';self.index=0;self.segment_count=0;self.bridge_requests=0;self.bridges=0
        self.section_action=0.;self.chart_value=0.
        self.crossing=np.zeros(4);self.touch_time=np.full(4,-np.inf)
        self.material_cells={};self.material_updated=np.full(4,-np.inf)
        self.local_material=np.zeros(4);self.latest_grf=np.zeros((4,3))
        self.orbit_log=[];self.command=None

    def set_orbit(self,mode,trajectory,duration):
        """External decision is applied at the next handover, leaving airborne feet committed."""
        if mode not in CHARTS:
            raise ValueError(mode)
        foot=np.asarray(trajectory,dtype=float).reshape(3)
        self.command=(mode,np.clip(foot,[-.022,-.015,-.008],[.022,.015,.012]),
                      float(np.clip(duration,.16,.24)))

    def _update_schedule(self,t,data,mq,mission,soft,material_info,observed,grf):
        self.latest_grf=grf
        for leg in np.flatnonzero(observed & (grf[:,2]>12.)):
            z=max(0.,self.radius[leg]-mq['feet'][leg,2]);load=grf[leg,2]
            compliance=np.clip((z/load-8.e-7)/7.e-6,0.,1.)
            self.local_material[leg]+=.035*(compliance-self.local_material[leg])
            xy=tuple(np.floor(mq['feet'][leg,:2]/.035).astype(int))
            self.material_cells[xy]=(self.local_material[leg],t,int(leg))
            self.material_updated[leg]=t
        return super()._update_schedule(t,data,mq,mission,soft,material_info,observed,grf)

    def _new_handover_request(self,t,mission,mq,material_info,grf):
        # The inherited measured support/force handover remains the sole permission to lift.
        # Chart changes are requested at an event; the first step of the new chart waits
        # for all committed swings to return to stance and for the new support to bear load.
        previous=self.chart
        if self.command is not None:
            requested=self.command[0]
        elif self.segment_count<4 or self.segment_count>=8:
            requested='trot'
        else:
            requested='crawl'
        if requested!=previous:
            self.bridge_requests+=1
            self.chart=requested;self.index=0
        elif self.segment_count:
            self.index=(self.index+1)%len(CHARTS[self.chart])
        self.next_pair=np.asarray(CHARTS[self.chart][self.index])
        duration=self._duration()
        swing_interval=duration+(.041 if self.chart=='trot' else .026)
        if self.last_liftoff is not None:
            self.next_liftoff=self.last_liftoff+swing_interval
        super()._new_handover_request(t,mission,mq,material_info,grf)
        normal=float(np.max(self.local_material[self.next_pair]))
        self.cfg['contact_events']['load_transfer_ramp_s']=.05+(.012*normal if self.gait_role in ('material','orbit') else 0.)
        self.chart_value=float(self.chart=='crawl')

    def _duration(self):
        if self.command is not None:
            return self.command[2]
        nominal=.20 if self.chart=='trot' else .166
        if self.gait_role in ('feedback','material','orbit'):
            return float(np.clip(nominal+.32*np.mean(self.crossing),.155,.225))
        return nominal

    def _coordinates(self,t,leg,mq,mission,p):
        duration=self._duration();crawl=self.chart=='crawl'
        velocity=mission.v[:2]
        target=self.model_data_root.copy()+mq['R']@self.nominal_feet_body[leg]
        stance=(.45 if crawl else .30)
        target[:2]+=(duration+.5*stance)*velocity
        target[2]=self.radius[leg]
        height=.026 if crawl else .033
        normal=-.012 if crawl else -.020
        if self.gait_role=='cpg':
            theta=2*np.pi*(t-self.walk_origin)/(.76 if crawl else .50)
            height+=.004*np.cos(theta);target[1]+=.003*np.sin(theta)
        if self.gait_role in ('feedback','material','orbit'):
            target[:2]+=.055*(mq['com_velocity'][:2]-velocity)
            side=1 if leg in (0,2) else -1
            target[1]+=.022*mq['rpy'][0]
            target[0]+=.020*mq['rpy'][1]
            height+=.18*side*np.clip(self.crossing[leg],-.025,.025)
        if self.gait_role in ('material','orbit'):
            cell=tuple(np.floor(target[:2]/.035).astype(int))
            cached=self.material_cells.get(cell)
            belief=self.local_material[leg]
            if self.gait_role=='orbit' and cached is not None:
                memory,stamp,owner=cached
                elapsed=max(0.,t-stamp)
                belief=.5*belief+.5*memory*np.exp(-elapsed/1.2)
            normal*=1-.65*belief;height+=.003*belief
            target[0]-=.004*belief*np.sign(velocity[0])
        if self.command is not None:
            target+=self.command[1]*[1.,1.,0.]
            height+=self.command[1][2]
        return target,duration,float(height),float(normal)

    def _start_swing(self,t,leg,data,mq,mission,soft,material_info):
        p=mq['feet'][leg];v=mq['foot_velocity'][leg]
        target,duration,height,normal=self._coordinates(t,leg,mq,mission,p)
        self.swing[leg]=segment(t,p,v,target,duration,height,normal)

    def _swing_reference(self,t,leg):
        s=self.swing[leg]
        return state_at(s,t) if s is not None else (self.anchor[leg].copy(),ZERO.copy(),ZERO.copy())

    def _event(self,t,kind,legs,**extra):
        if kind=='actual_touchdown':
            i=int(legs[0]);s=self.swing[i]
            if s is not None and self.touch_time[i]<s['start']:
                self.crossing[i]=.65*self.crossing[i]+.35*np.clip(t-s['end'],-.045,.045)
                self.touch_time[i]=t
                extra.update(orbit_timing_error_s=t-s['end'],orbit_committed_t=s['end'])
        elif kind=='liftoff_committed':
            self.segment_count+=1
            s=self.swing[int(legs[0])]
            extra.update(fixed_apex_s=s['apex'],fixed_touchdown_s=s['end'],orbit_chart=self.chart,
                         orbit_phase_error_s=self.crossing.tolist(),material_belief=self.local_material.tolist())
            if self.orbit_log and self.orbit_log[-1]['chart']!=self.chart:
                self.bridges+=1
            self.orbit_log.append(dict(time=t,legs=list(map(int,legs)),chart=self.chart,
                duration=s['end']-t,landing=s['landing'].tolist(),bridge_count=self.bridges))
        super()._event(t,kind,legs,**extra)

    def _forecast_events(self,t,end_time,mq,mission):
        events=[[] for _ in range(4)]
        available=np.full(4,t); contact_cfg=self.cfg['contact_events']
        for leg,s in enumerate(self.swing):
            if self.event_mode[leg]=='swing':
                events[leg].append((s['start'],s['end'],s))
                available[leg]=s['end']+contact_cfg['touch_confirm_s']
            elif not self.confirmed[leg]:
                target=s['landing'] if s else self.initial_landing[leg]
                gap=max(0.,mq['feet'][leg,2]-target[2])
                available[leg]=t+gap/contact_cfg['search_speed_m_s']+contact_cfg['touch_confirm_s']
                events[leg].append((t,available[leg],dict(search=True,landing=target.copy())))
        if not self.moving or self.next_liftoff is None:
            return events
        due=max(self.next_liftoff,float(available.max())+contact_cfg['load_transfer_ramp_s'])
        if np.all(self.confirmed) and self.transfer_start is not None:
            due=max(self.next_liftoff,self.transfer_start+contact_cfg['load_transfer_ramp_s'])
        index=self.index;duration=self._duration()
        while due<=end_time+1.e-10:
            for leg in CHARTS[self.chart][index]:
                p=events[leg][-1][2]['landing'] if events[leg] else mq['feet'][leg]
                target,duration,height,normal=self._coordinates(t,leg,mq,mission,p)
                target[:2]+=(due-t)*mission.v[:2]
                s=segment(due,p,ZERO,target,duration,height,normal)
                events[leg].append((due,s['end'],s))
            due+=duration+(.041 if self.chart=='trot' else .026)
            index=(index+1)%len(CHARTS[self.chart])
        return events

    def _future_contact_feet(self,t,times,mq,mission):
        events=self._forecast_events(t,float(times[-1]),mq,mission)
        contact=np.tile(self.measured&(self.event_mode!='swing'),(len(times),1))
        feet=np.tile(mq['feet'],(len(times),1,1))
        for i,leg_events in enumerate(events):
            for k,tk in enumerate(times):
                if abs(tk-t)<1.e-10:
                    continue
                for start,end,s in leg_events:
                    if tk<start:
                        break
                    if tk<end:
                        contact[k,i]=False
                        if s.get('search'):
                            feet[k,i,2]-=(tk-t)*self.cfg['contact_events']['search_speed_m_s']
                        else:
                            feet[k,i]=state_at(s,tk)[0]
                        break
                    feet[k,i]=s['landing'];contact[k,i]=True
        # W59 nominal B3 has zero force-preference; external candidate authority is handled in the new orbit.
        self.w59_horizon_mask=np.zeros(len(times)-1,bool)
        return contact,feet

    def _body_reference(self,t,mq,mission):
        state,ref=super()._body_reference(t,mq,mission)
        if self.walk_origin is None or mission.phase in STAND_PHASES:
            return state,ref
        phase=2*np.pi*(t+np.arange(len(ref))*self.cfg['mpc']['dt_s']-self.walk_origin)/(.76 if self.chart=='crawl' else .5)
        amplitude=.0015 if self.chart=='crawl' else .0008
        rate=2*np.pi/(.76 if self.chart=='crawl' else .5)
        ref[:,5]+=amplitude*(1-np.cos(phase));ref[:,11]+=amplitude*rate*np.sin(phase)
        if self.chart=='crawl':
            ref[:,0]+=.004*np.sin(phase);ref[:,6]+=.004*rate*np.cos(phase)
        return state,ref

    def flush(self):
        super().flush()
        out=self.out/'controlled-orbit.json'
        out.write_text(json.dumps(dict(role=self.gait_role,issued_segments=self.segment_count,
            transition_requests=self.bridge_requests,executed_transitions=self.bridges,
            measured_cells=len(self.material_cells),events=self.orbit_log),indent=2)+'\n')
