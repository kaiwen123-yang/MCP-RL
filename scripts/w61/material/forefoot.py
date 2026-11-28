"""Independent fixed-controller causal material comparators in the frozen F forefoot plant."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import lsq_linear
from w61.joint.heads import MaterialHead,NA,NC
from w61.joint.plant import rollout
from s1_w60_cpu_observer import CPUColumns

WEIGHTS=Path('/root/autodl-tmp/MCP-RL/work/assets/B_random1000--ideal-potentials.npz')


class Embedding(MaterialHead):
    def __init__(self,checkpoint,seed):
        import torch
        from tm02.w61_material import Family
        f=torch.load(checkpoint,map_location='cpu',weights_only=True)
        cfg=f['config']
        net=Family(cfg,'embedding_response')
        self.name='embedding_response_'+str(seed)
        net.load_state_dict(f['state_dicts'][self.name])
        self.base=net.base.detach().numpy().astype(float)
        self.affine=net.hidden.weight.detach().numpy().astype(float)
        self.bias=net.hidden.bias.detach().numpy().astype(float)
        self.readout=net.readout.detach().numpy().astype(float)
        self.scale=np.array(cfg['stiffness_scale_MPa'])
        self.tau=np.array(cfg['tau_s'])
        self.strain_scale=cfg['strain_scale']

    def coefficients(self,code):
        return self.scale*np.logaddexp(0,self.base+np.tanh(self.affine@code+self.bias)@self.readout)


class OrdinaryKV:
    def adapt(self,theta,support,steps):
        bottom=np.stack([r['bottom'] for r in support])
        normal=np.stack([r['fz'] for r in support])
        mask=np.stack([r['soft'] for r in support])
        dt=np.asarray([r['dt'] for r in support])[:,None]
        rate=-np.diff(bottom,axis=0,prepend=bottom[:1])/dt
        design=np.stack((np.maximum(-bottom,0.),rate),axis=-1)
        split=2*len(support)//3
        obs=mask[:split]&(bottom[:split]<0)
        v=lsq_linear(design[:split][obs],normal[:split][obs],bounds=(0,np.inf),method='bvls').x
        pred=np.where(bottom<0,np.maximum(design@v,0.),0.)
        err=(pred-normal)[split:][mask[split:]]
        return v,float(v[0]),float(np.sqrt(np.mean(err**2)))


class FrozenPANN:
    def __init__(self):
        self.law=CPUColumns(WEIGHTS)

    def adapt(self,theta,support,steps):
        # Each column has zero lateral stress. After separation it evolves under zero
        # normal stress, and force uses its measured overlap with the current sole.
        index={}; frames=[];half=np.sqrt(.001)/2
        for row in support:
            keys=[];legs=[];areas=[]
            for leg in np.flatnonzero(row['soft']):
                lo=row['xy'][leg]-half;hi=row['xy'][leg]+half
                for ix in range(int(np.floor(lo[0]/.02)),int(np.floor(hi[0]/.02))+1):
                    for iy in range(int(np.floor(lo[1]/.02)),int(np.floor(hi[1]/.02))+1):
                        corner=np.array([ix,iy])*.02
                        area=np.maximum(0.,np.minimum(hi,corner+.02)-np.maximum(lo,corner)).prod()
                        if area>0:
                            keys.append(index.setdefault((ix,iy),len(index)));legs.append(leg);areas.append(area)
            frames.append((np.array(keys,int),np.array(legs,int),np.array(areas)))
        e=np.zeros((len(index),2));q=np.zeros_like(e)
        pred=[]
        for row,(keys,legs,area) in zip(support,frames):
            d=row['dt']
            free_e,free_q,_,_=self.law.step(e,q,np.zeros(len(e)),d,np.zeros(len(e),bool))
            free=free_e[keys,0]
            normal=row['bottom'][legs]/.05
            touched=normal<=free
            k,leg,aa=keys[touched],legs[touched],area[touched]
            load=np.zeros(4)
            if len(k):
                be,bq,s,_=self.law.step(e[k],q[k],normal[touched],d,np.ones(len(k),bool))
                pressure=np.maximum(0.,-25000*s[:,0])
                compressed=pressure>0
                free_e[k[compressed]]=be[compressed];free_q[k[compressed]]=bq[compressed]
                load=np.bincount(leg,weights=aa*pressure,minlength=4)
            e,q=free_e,free_q
            pred.append(load)
        split=2*len(support)//3
        mask=np.stack([r['soft'] for r in support])[split:]
        error=(np.asarray(pred)-np.stack([r['fz'] for r in support]))[split:][mask]
        tiny=np.zeros((1,2));strain=-1.e-6
        _,_,s,_=self.law.step(tiny,tiny,np.array([strain]),.0002,np.ones(1,bool))
        stiffness=float(25000*.001*s[0,0]/(.05*strain))
        return np.zeros(0),stiffness,float(np.sqrt(np.mean(error**2)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('spec',type=Path)
    p.add_argument('case',type=int)
    p.add_argument('method')
    p.add_argument('seed',type=int)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    cfg=json.loads(a.spec.read_text())
    methods={'lowrank_response':lambda cp,ignored:MaterialHead(cp,a.seed),
             'embedding_response':lambda cp,ignored:Embedding(cp,a.seed),
             'ordinary_KV':lambda *_:OrdinaryKV(),'frozen_PANN':lambda *_:FrozenPANN()}
    result=rollout(a.out,np.zeros(NA+NC),cfg['cases'][a.case],cfg['matched_query_rng']+a.case,
                   arm='frozen_C',head_factory=methods[a.method])
    print(a.method,a.seed,a.case,result['physical_s'],result['reason'],flush=True)


if __name__=='__main__':
    main()
