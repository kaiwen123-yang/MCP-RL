"""Matched learned style-conditioned oscillator on the same trajectories and support bridge."""
import argparse
import json
from pathlib import Path

import numpy as np
from w61_orbit_decoder import extract, features, Y, SCALE
from w61_support_bridge import SupportBridge

LEGS=['FL','FR','RL','RR']


def style_map(raw):
    v=np.atleast_2d(raw)
    front,side,crawl=v[:,6],v[:,7],v[:,8]
    leg=np.where(front>.5,0,2)+side.astype(int)
    diag=np.array([.5,0.,0.,.5])[leg]
    single=np.array([.50,0.,.25,.75])[leg]
    phase=2*np.pi*np.where(crawl>.5,single,diag)
    return np.column_stack((v[:,0],v[:,1],front,side,crawl,np.sin(phase),np.cos(phase),
                            np.cos(2*phase),v[:,0]*np.sin(phase),v[:,0]*np.cos(phase),v[:,0]**2))


def train(args):
    train=[extract(p) for p in args.train];valid=[extract(p) for p in args.validation]
    x=style_map(np.concatenate([a for a,_ in train]));target=np.concatenate([b for _,b in train])/SCALE
    args.out.mkdir(parents=True,exist_ok=True)
    for seed in (6111,6112,6113):
        rng=np.random.Generator(np.random.PCG64(seed))
        W=rng.normal(0,.45,size=(x.shape[1],20));b=rng.uniform(-1,1,size=20)
        phi=features(x,W,b);reg=np.diag(np.r_[0.,np.full(phi.shape[1]-1,.012*len(phi))])
        beta=np.linalg.solve(phi.T@phi+reg,phi.T@target)
        for _ in range(4):
            weights=1./np.maximum(1.,np.abs(phi@beta-target)/.25)
            for i in range(target.shape[1]):
                weighted=phi*weights[:,i,None]
                beta[:,i]=np.linalg.solve(phi.T@weighted+reg,weighted.T@target[:,i])
        val={}
        for raw,(a,y) in zip(args.validation,valid):
            val[raw.name]=dict(n=len(a),MAE=np.abs(features(style_map(a),W,b)@beta*SCALE-y).mean(0).tolist())
        obj=dict(id='w61-phase-and-style-oscillator-v1',seed=seed,W=W.tolist(),b=b.tolist(),beta=beta.tolist(),
            targets=Y,train=[p.name for p in args.train],training_distinct_strides=len(x),validation=val,
            feature_scope='Commanded velocity, leg identity, chart and periodic harmonics; same support-bridge physics and event release gates as proposed arm. No actual transverse state or shared material memory in the decoder.',
            comparison='two executable modes; matched-random-feature decoder, dimensions, ridge penalty, 4 IRLS passes. This is a style-conditioned CPG baseline, not a reproduction of all eleven published gaits.')
        (args.out/f'seed-{seed}.json').write_text(json.dumps(obj,indent=2)+'\n')


class StyleOscillator(SupportBridge):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        spec=json.loads(Path(self.cfg['w61']['artifact']).read_text())
        self.style_W=np.array(spec['W']);self.style_b=np.array(spec['b']);self.style_beta=np.array(spec['beta'])
        self.style_ramp=.05;self.style_heave=.0012

    def _predict(self,mission,i):
        x=np.zeros(11);x[:2]=mission.v[:2]/[.22,.06]
        x[6:9]=[float(i<2),float(i%2),float(self.chart=='crawl')]
        return (features(style_map(x),self.style_W,self.style_b)@self.style_beta)[0]*SCALE

    def _coordinates(self,t,leg,mq,mission,p):
        # Baseline gets the same contact bridge and time-parametrized C2 action dimensions.
        target,duration,height,normal=super()._coordinates(t,leg,mq,mission,p)
        coef=self._predict(mission,leg)
        residual=mq['R']@np.r_[np.clip(coef[:2],[-.018,-.014],[.018,.014]),0.]
        target[:2]+=.45*residual[:2]
        return (target,float(.7*duration+.3*np.clip(coef[4],.155,.235)),
            float(.55*height+.45*np.clip(coef[2],.015,.05)),
            float(.65*normal+.35*np.clip(coef[3],-.11,0.)))

    def _start_swing(self,t,leg,data,mq,mission,soft,material_info):
        super()._start_swing(t,leg,data,mq,mission,soft,material_info)
        coeff=self._predict(mission,leg)
        self.style_ramp=.05+.2*np.clip(coeff[5]-.022,-.025,.03)
        self.style_heave=.25*np.clip(coeff[6],.002,.012)

    def _new_handover_request(self,t,mission,mq,material_info,grf):
        super()._new_handover_request(t,mission,mq,material_info,grf)
        self.cfg['contact_events']['load_transfer_ramp_s']=self.style_ramp

    def _body_reference(self,t,mq,mission):
        state,ref=super()._body_reference(t,mq,mission)
        if self.walk_origin is not None and self.moving:
            omega=2*np.pi/(.76 if self.chart=='crawl' else .50)
            phase=(t+np.arange(len(ref))*self.cfg['mpc']['dt_s']-self.walk_origin)*omega
            amplitude=self.style_heave-(.0015 if self.chart=='crawl' else .0008)
            ref[:,5]+=amplitude*(1.-np.cos(phase));ref[:,11]+=amplitude*omega*np.sin(phase)
        return state,ref


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',type=Path,nargs='+',required=True)
    p.add_argument('--validation',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    train(p.parse_args())
