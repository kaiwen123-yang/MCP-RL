"""Learn a periodic two-arc trajectory family from measured whole-route execution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from w61_support_bridge import SupportBridge

LEGS=['FL','FR','RL','RR']
Y=['touchdown_body_x_residual_m','touchdown_body_y_residual_m','swing_apex_m',
   'touchdown_velocity_m_s','observed_flight_s','time_to_45N_s','body_excursion_m']
SCALE=np.array([.03,.02,.035,.10,.20,.04,.008])
X_SCALE=np.array([.22,.06,.08,.06,.10,.10,1.,1.,1.,1.,1.])


def inputs(v,cmd,rpy,i,crawl,material,historical):
    return np.array([cmd[0],cmd[1],v[0]-cmd[0],v[1]-cmd[1],rpy[0],rpy[1],float(i<2),float(i%2),
                     float(crawl),material,historical])/X_SCALE


def extract(directory):
    with (directory/'w55-f/contact-events.jsonl').open() as f:
        events=[json.loads(line) for line in f]
    with np.load(directory/'trajectory.npz') as tr:
        legacy='feet' in tr
        t=tr['time'];q=tr['qpos'];vel=tr['qvel'];feet=tr['feet' if legacy else 'foot_xyz']
        force=tr['actual_grf' if legacy else 'force'];vref=tr['command_velocity' if legacy else 'requested_velocity']
        rpy=tr['rpy' if legacy else 'body_rpy']
    rotation=Rotation.from_quat(q[:,[4,5,6,3]])
    nominal=rotation[0].inv().apply(feet[0]-q[0,:3])
    # Track only prior sampled loaded cells. Labels for the landing are never inputs.
    memory={};material=np.zeros(4);last_index=0;active={};X=[];Y_true=[]
    for e in events:
        k=min(int(np.searchsorted(t,e['time_s']-1.e-9)),len(t)-1)
        if e['kind']=='liftoff_committed':
            for j in range(last_index,k+1):
                for i in np.flatnonzero(force[j,:,2]>12.):
                    z=max(0.,.022-feet[j,i,2]);load=force[j,i,2]
                    c=np.clip((z/load-8.e-7)/7.e-6,0.,1.)
                    material[i]+=.035*(c-material[i])
                    memory[tuple(np.floor(feet[j,i,:2]/.035).astype(int))]=(material[i],t[j],i)
            last_index=k+1
            for name in e['legs']:
                i=LEGS.index(name);crawl=len(e['legs'])==1
                cmd=vref[k,:2];v=vel[k,:2]
                duration=.166 if crawl else .20; stance=.45 if crawl else .30
                target=q[k,:3]+rotation[k].apply(nominal[i])
                target[:2]+=(duration+.5*stance)*cmd
                previous=memory.get(tuple(np.floor(target[:2]/.035).astype(int)))
                hist=0. if previous is None or previous[2]==i else previous[0]*np.exp(-(t[k]-previous[1])/1.2)
                active[i]=(k,target,inputs(v,cmd,rpy[k],i,crawl,material[i],hist))
        elif e['kind']=='actual_touchdown':
            i=LEGS.index(e['legs'][0])
            if i not in active:continue
            start,nom,x=active.pop(i)
            if k-start<3 or k+30>=len(t):continue
            delta=feet[k,i]-nom
            delta=rotation[start].inv().apply(np.r_[delta[:2],0.])[:2]
            peak=feet[start:k+1,i,2].max()-max(feet[start,i,2],feet[k,i,2])
            collision_vel=(feet[k,i,2]-feet[k-2,i,2])/(t[k]-t[k-2])
            loaded=np.flatnonzero(force[k:k+30,i,2]>=45.)
            load_delay=(loaded[0]*.002 if len(loaded) else .060)
            X.append(x)
            Y_true.append([*delta,peak,collision_vel,t[k]-t[start],load_delay,
                           np.ptp(q[start:k+1,2])])
    return np.asarray(X),np.asarray(Y_true)


def features(A,W,b):
    return np.column_stack((np.ones(len(A)),A,np.tanh(A@W+b)))


def train(args):
    datasets={p.name:extract(p) for p in [*args.train,*args.validation]}
    design=args.train
    A=np.concatenate([datasets[p.name][0] for p in design]);Y0=np.concatenate([datasets[p.name][1] for p in design])
    source={p.name:{'events_sha256':hashlib.sha256((p/'w55-f/contact-events.jsonl').read_bytes()).hexdigest(),
                    'stride_samples':len(datasets[p.name][0])} for p in [*args.train,*args.validation]}
    args.out.mkdir(exist_ok=True,parents=True)
    for seed in [6111,6112,6113]:
        rng=np.random.Generator(np.random.PCG64(seed));W=rng.normal(0,.45,size=(A.shape[1],20));b=rng.uniform(-1.,1.,size=20)
        Phi=features(A,W,b);Y_norm=Y0/SCALE
        penalty=np.diag(np.r_[0.,np.full(Phi.shape[1]-1,.012*len(A))])
        beta=np.linalg.solve(Phi.T@Phi+penalty,Phi.T@Y_norm)
        for _ in range(4):
            weight=1./np.maximum(1.,np.abs(Phi@beta-Y_norm)/.25)
            for col in range(len(Y)):
                weighted=Phi*weight[:,col,None]
                beta[:,col]=np.linalg.solve(Phi.T@weighted+penalty,weighted.T@Y_norm[:,col])
        validation={}
        for p in args.validation:
            x,y=datasets[p.name];prediction=(features(x,W,b)@beta)*SCALE
            validation[p.name]=dict(count=len(y),MAE=np.abs(prediction-y).mean(axis=0).tolist(),
                constant_training_median_MAE=np.abs(np.median(Y0,axis=0)-y).mean(axis=0).tolist())
        obj=dict(seed=seed,features=X_SCALE.tolist(),targets=Y,scales=SCALE.tolist(),
                 W=W.tolist(),b=b.tolist(),beta=beta.tolist(),sources=source,
                 validation=validation,training_distinct_strides=len(A),train_routes=[p.name for p in design],
                 validation_routes=[p.name for p in args.validation],optimizer='ridge random-feature map, 4 Huber IRLS iterations per output')
        (args.out/f'seed-{seed}.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
        print(seed,len(A),{k:v['count'] for k,v in validation.items()},flush=True)


class LearnedOrbit(SupportBridge):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        fitted=json.loads(Path(self.cfg['w61']['artifact']).read_text())
        self.decode_W=np.asarray(fitted['W']);self.decode_b=np.asarray(fitted['b'])
        self.decode_beta=np.asarray(fitted['beta'])
        self.learned_heave=.0012;self.learned_load_s=.05

    def _coordinates(self,t,leg,mq,mission,p):
        target,duration,height,normal=super()._coordinates(t,leg,mq,mission,p)
        xy=tuple(np.floor(target[:2]/.035).astype(int));cached=self.material_cells.get(xy)
        past=0. if cached is None or cached[2]==leg else cached[0]*np.exp(-max(0.,t-cached[1])/1.2)
        condition=inputs(mq['com_velocity'][:2],mission.v[:2],mq['rpy'],leg,self.chart=='crawl',
                         self.local_material[leg],past)
        coeff=(features(condition[None,:],self.decode_W,self.decode_b)@self.decode_beta)[0]*SCALE
        residual=mq['R']@np.r_[np.clip(coeff[:2],[-.018,-.014],[.018,.014]),0.]
        target[:2]+=.45*residual[:2]
        height=.55*height+.45*np.clip(coeff[2],.015,.05)
        normal=.65*normal+.35*np.clip(coeff[3],-.11,0.)
        duration=.70*duration+.30*np.clip(coeff[4],.155,.235)
        return target,float(duration),float(height),float(normal)

    def _start_swing(self,t,leg,data,mq,mission,soft,material_info):
        super()._start_swing(t,leg,data,mq,mission,soft,material_info)
        target=self.swing[leg]['landing']
        cached=self.material_cells.get(tuple(np.floor(target[:2]/.035).astype(int)))
        past=0. if cached is None or cached[2]==leg else cached[0]*np.exp(-max(0.,t-cached[1])/1.2)
        x=inputs(mq['com_velocity'][:2],mission.v[:2],mq['rpy'],leg,self.chart=='crawl',self.local_material[leg],past)
        response=(features(x[None,:],self.decode_W,self.decode_b)@self.decode_beta)[0]*SCALE
        self.learned_load_s=.05+.2*np.clip(response[5]-.022,-.025,.03)
        self.learned_heave=.25*np.clip(response[6],.002,.012)

    def _event(self,t,kind,legs,**extra):
        if kind=='liftoff_committed':
            extra['per_leg_committed_apex_s']=[self.swing[int(i)]['apex'] for i in legs]
            extra['per_leg_committed_touch_s']=[self.swing[int(i)]['end'] for i in legs]
        super()._event(t,kind,legs,**extra)

    def _new_handover_request(self,t,mission,mq,material_info,grf):
        super()._new_handover_request(t,mission,mq,material_info,grf)
        self.cfg['contact_events']['load_transfer_ramp_s']=self.learned_load_s

    def _body_reference(self,t,mq,mission):
        state,ref=super()._body_reference(t,mq,mission)
        if self.walk_origin is not None and self.moving:
            omega=2*np.pi/(.76 if self.chart=='crawl' else .50)
            phase=(t+np.arange(len(ref))*self.cfg['mpc']['dt_s']-self.walk_origin)*omega
            a=self.learned_heave-(.0015 if self.chart=='crawl' else .0008)
            ref[:,5]+=a*(1.-np.cos(phase));ref[:,11]+=a*omega*np.sin(phase)
        return state,ref


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',type=Path,nargs='+',required=True)
    p.add_argument('--validation',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    train(p.parse_args())
