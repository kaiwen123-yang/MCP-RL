"""Periodic contact-phase plus aperiodic task disturbances inside the same WBC QP."""
import argparse
import json
from pathlib import Path

import numpy as np
from w61_natural_wbc import NaturalWBCMixin


def harmonics(theta):
    return np.stack((np.sin(theta),np.cos(theta),np.sin(2*theta),np.cos(2*theta)),axis=-1)


def fit(run,out):
    trace=np.load(run/'trajectory.npz')
    controls=[np.load(p) for p in sorted((run/'w55-f').glob('control-*.npz'))]
    accelerations=np.concatenate([a['qacc'][:,:6] for a in controls])
    with (run/'w55-f/contact-events.jsonl').open() as stream:
        events=[json.loads(line) for line in stream]
    starts=[e['time_s'] for e in events if e['kind']=='liftoff_committed' and e['legs']==['FR','RL']]
    t=trace['time'][:-1]
    dv=np.diff(trace['qvel'][:,:6],axis=0)/np.diff(trace['time'])[:,None]
    residual=dv-accelerations[:-1]
    parts=[]; targets=[]
    for begin,end in zip(starts,starts[1:]):
        mask=(t>=begin)&(t<end)&np.all(trace['contact'][1:]==trace['contact'][:-1],axis=1)
        parts.append(harmonics(2*np.pi*(t[mask]-begin)/(end-begin)))
        targets.append(residual[mask])
    X=np.concatenate(parts);y=np.concatenate(targets)
    med=np.median(y,axis=0); bound=np.maximum(np.percentile(np.abs(y-med),90,axis=0),.2)
    H=np.linalg.solve(X.T@X+.03*len(X)*np.eye(4), X.T@np.clip(y-med,-bound,bound))
    U,s,V=np.linalg.svd(H/bound,full_matrices=False)
    periodic=((U[:,:2]*s[:2])@V[:2])*bound
    out.write_text(json.dumps(dict(design_run=run.name,design_result=json.loads((run/'result.json').read_text())['runner'],
        form='two-harmonic generalized trunk acceleration residual, rank-two periodic basis', design_intervals=len(starts)-1,
        used_samples=len(X),bound=bound.tolist(),periodic=periodic.tolist(),singular_values=s.tolist()),indent=2)+'\n')


class WBCObserver:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        spec=self.cfg['w61'];model=json.loads(Path(spec['artifact']).read_text())
        self.obs_bound=np.asarray(model['bound']);self.periodic=np.asarray(model['periodic'])
        self.use_phase=spec['role']=='hybrid'
        self.tracking_gain=spec.get('tracking_adaptation_s_inv',0.) if self.use_phase else 0.
        self.obs_bias=np.zeros(6);self.obs_amp=np.zeros(6)
        self.obs_cycle=None;self.obs_period=.52;self.obs_last_event=0
        self.obs_prev=None;self.obs_mq=None;self.obs_data=None;self.obs_applied=[]

    def wbc(self,t,data,mq,mission,scheduled,observed,grf,soft,material_info):
        for e in self.events[self.obs_last_event:]:
            if e['kind']=='liftoff_committed' and e['legs']==['FR','RL']:
                if self.obs_cycle is not None:
                    self.obs_period=.65*self.obs_period+.35*(t-self.obs_cycle)
                self.obs_cycle=t
        self.obs_last_event=len(self.events);self.obs_data=data;self.obs_mq=mq
        self.obs_mode=tuple(scheduled&observed)
        return super().wbc(t,data,mq,mission,scheduled,observed,grf,soft,material_info)

    def _solve(self,P,q,A,lo,hi,kind,t,warm=None):
        if kind!='wbc':
            return super()._solve(P,q,A,lo,hi,kind,t,warm)
        mq=self.obs_mq;v=self.obs_data.qvel[:6].copy()
        phase=np.zeros(6) if self.obs_cycle is None else harmonics(2*np.pi*(t-self.obs_cycle)/self.obs_period)@self.periodic
        if self.obs_prev is not None:
            prev_v,prev_acc,prev_t,prev_mode,prev_phase=self.obs_prev
            if self.obs_mode==prev_mode:
                elapsed=t-prev_t
                innovation=np.clip((v-prev_v)/elapsed-prev_acc,-self.obs_bound,self.obs_bound)
                estimate=self.obs_bias+(self.obs_amp*prev_phase if self.use_phase else 0.)
                error=innovation-estimate
                if self.tracking_gain:
                    plan=self.last_plan
                    u=np.clip((t-plan['time'])/self.cfg['mpc']['dt_s'],0.,1.)
                    xd=(1-u)*plan['state'][0]+u*plan['state'][1]
                    tracking=np.r_[mq['com_velocity']-xd[9:12],mq['R'].T@(mq['omega']-xd[6:9])]
                    # Persistent signed velocity error provides an additional bounded adaptation drive.
                    error+=np.clip(self.tracking_gain*tracking,-self.obs_bound,self.obs_bound)
                self.obs_bias=np.clip(self.obs_bias+(1-np.exp(-elapsed/.12))*error,-self.obs_bound,self.obs_bound)
                if self.use_phase:
                    gain=prev_phase/(.2*self.obs_bound**2+prev_phase**2)
                    self.obs_amp=np.clip(self.obs_amp+(1-np.exp(-elapsed/.35))*gain*error,0.,1.5)
        predicted=self.obs_bias+(self.obs_amp*phase if self.use_phase else 0.)
        # Base angular velocities are body-local, and the WBC task uses world angular acceleration.
        # The translational free-base disturbance is a common translation at COM.
        correction=np.r_[predicted[:3],mq['R']@predicted[3:]]
        cfg=self.cfg['wbc']
        J=np.vstack((mq['Jcom'],mq['Jr']));W=np.repeat([cfg['body_weight'],cfg['orientation_weight']],3)
        q=q.copy();q[:self.model.nv]+=2.*J.T@(W*correction)
        x,info=super()._solve(P,q,A,lo,hi,kind,t,warm)
        self.obs_prev=v,x[:6].copy(),t,self.obs_mode,phase
        self.obs_applied.append(np.r_[self.obs_bias,self.obs_amp*phase])
        return x,info

    def w61_d_stats(self):
        parent=getattr(super(),'w61_d_stats',None)
        evidence=parent() if parent else {}
        values=np.asarray(self.obs_applied)
        evidence['observer']=dict(solved=len(values),periodic=self.use_phase,
            mean_abs_bias=np.mean(np.abs(values[:,:6]),axis=0).tolist() if len(values) else [],
            mean_abs_periodic=np.mean(np.abs(values[:,6:]),axis=0).tolist() if len(values) else [])
        return evidence


class HybridWBC(WBCObserver,NaturalWBCMixin):
    pass


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();fit(a.run,a.out)
