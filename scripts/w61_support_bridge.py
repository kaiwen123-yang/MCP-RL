"""Capture-point bridge from the diagonal orbit into a single-swing support chart."""
import json
import numpy as np
from w61_controlled_orbits import ControlledOrbit, CHARTS
from s1_w54_f_r52_backend import STAND_PHASES


def barycentric(triangle,point):
    E=np.vstack((triangle.T,np.ones(3)))
    return np.linalg.solve(E,np.r_[point,1.])


class SupportBridge(ControlledOrbit):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.pending=None;self.rejected=[];self.lock_diagonal=False
        self.margin_min=1.;self.bridge_shift=np.zeros(2)
        self.bridge_alignment=None;self.last_orbit_legs=None

    def _new_handover_request(self,t,mission,mq,material_info,grf):
        previous=self.chart
        if self.lock_diagonal:
            self.set_orbit('trot',[0.,0.,0.],.20)
        super()._new_handover_request(t,mission,mq,material_info,grf)
        if self.chart=='crawl':
            support=[i for i in range(4) if i not in self.next_pair]
            vertices=mq['feet'][support,:2].copy()
            omega=np.sqrt(9.81/max(.12,mq['com'][2]))
            capture=mq['com'][:2]+mq['com_velocity'][:2]/omega
            coords=barycentric(vertices,capture)
            inside=np.maximum(coords,.18);inside/=inside.sum()
            goal=inside@vertices
            self.bridge_alignment=(t,goal-capture)
            self.pending=(t+(.20 if previous=='trot' else .08),vertices,
                          t+(.50 if previous=='trot' else .40))
        else:
            self.pending=None;self.bridge_alignment=None;self.bridge_shift[:]=0.

    def _update_schedule(self,t,data,mq,mission,soft,material_info,observed,grf):
        if self.pending is not None:
            earliest,vertices,last=self.pending
            omega=np.sqrt(9.81/max(.12,mq['com'][2]))
            margin=float(barycentric(vertices,mq['com'][:2]+mq['com_velocity'][:2]/omega).min())
            self.margin_min=min(self.margin_min,margin)
            if t>=earliest and margin>=.035:
                self.pending=None
            elif t>=last and np.all(self.event_mode=='stance'):
                # Resume the alternating diagonal subsequent to the last real liftoff.
                # Retain the original force and residual release gate.
                self.rejected.append(dict(time=t,margin=margin,candidate=list(map(int,self.next_pair))))
                self.lock_diagonal=True;self.chart='trot'
                self.index=1 if self.last_orbit_legs==(1,2) else 0
                self.next_pair=np.asarray(CHARTS['trot'][self.index])
                self.transfer_start=t;self.next_liftoff=t+self.cfg['contact_events']['load_transfer_ramp_s']
                self.pending=None;self.bridge_alignment=None
            else:
                self.next_liftoff=max(self.next_liftoff,t+.002)
        return super()._update_schedule(t,data,mq,mission,soft,material_info,observed,grf)

    def _body_reference(self,t,mq,mission):
        state,ref=super()._body_reference(t,mq,mission)
        if self.bridge_alignment is not None and mission.phase not in STAND_PHASES:
            begin,shift=self.bridge_alignment
            T=.20
            times=t+np.arange(len(ref))*self.cfg['mpc']['dt_s']
            u=np.clip((times-begin)/T,0.,1.)
            smooth=u**3*(10.-15.*u+6.*u*u)
            slope=30./T*u*u*(1.-u)**2
            ref[:,3:5]+=smooth[:,None]*shift
            ref[:,9:11]+=slope[:,None]*shift
        return state,ref

    def _event(self,t,kind,legs,**extra):
        if kind=='liftoff_committed':
            self.last_orbit_legs=tuple(map(int,legs))
        super()._event(t,kind,legs,**extra)

    def flush(self):
        super().flush()
        (self.out/'support-bridges.json').write_text(json.dumps(dict(rejected=self.rejected,
            minimum_barycentric_capture_weight=self.margin_min,executed=self.bridges),indent=2)+'\n')
