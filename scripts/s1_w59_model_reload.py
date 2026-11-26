"""One validated FP64 first-reload correction to the shared learned GRF forecast."""
import numpy as np
import torch

from s1_w59_model_fields import STATE,INTERVAL,DT
from s1_w59_model_cells import footprint
from s1_w59_model_recovery import responses


class PublishedMaterial:
    """Causal source states from the actual current observer, indexed by world ID."""
    def __init__(self,record):
        rows=record['rows'];self.rows={int(r[0]):r for r in rows}
        self.coeff=record['coefficients'];self.t=record['t'];self.law=record['law']
        self.free_cache={}

    def m1_at(self,key,t):
        # Only cells that an actual candidate reaches need a free continuation.
        clock=round((t-self.t)/.0002)
        ck=(key,clock)
        if ck not in self.free_cache:
            from score_s1_w57_material import ROOT
            import ctypes as ct
            fn=ct.CDLL(str(ROOT/'work/w57-m/forecast_cpu.so')).w57_forecast_cpu
            pointer=ct.c_void_p
            fn.argtypes=[pointer,pointer,pointer,pointer,ct.c_int,ct.c_double,ct.c_int,pointer];fn.restype=None
            initial=np.ascontiguousarray(self.rows[key][11:15],np.float64)
            boundary=np.zeros(1);length=np.array([max(clock,0)],np.int32);out=np.empty(6)
            fn(self.law.model.ctypes.data,initial.ctypes.data,boundary.ctypes.data,length.ctypes.data,1,.0002,1,out.ctypes.data)
            self.free_cache[ck]=out[:4].copy()
        return self.free_cache[ck]

    def correction(self,nxt,old,new,previously_loaded):
        valid=(old[...,2]<0)&(old[...,4]>0)&(new[...,2]>=0)&(old[...,3]!=new[...,2])
        count=valid.sum(-1)
        # No oracle label: each predicted foot and its overlapping area are
        # determined by that candidate's predicted trajectory.
        valid=valid&~previously_loaded
        selection=valid.nonzero().detach().cpu().numpy()
        if not len(selection):
            return nxt.new_zeros((len(nxt),4)),count,count.new_zeros(len(nxt))
        owner=new[...,2].detach().cpu().numpy()
        xyz=nxt[:,STATE['feet']].reshape(-1,4,3)
        positions=xyz.detach().cpu().numpy();clock=nxt[:,STATE['time']].detach().cpu().numpy()[:,0]
        choices={}
        for b,k in selection:
            r=self.rows.get(int(k))
            if r is None:continue
            l=int(owner[b,k])
            if l==int(r[1]):continue
            keys,areas=footprint(*positions[b,l,:2]);where=np.flatnonzero(keys==k)
            if not len(where):continue
            area=float(areas[where[0]]*.001);oldchoice=choices.get((b,l))
            if oldchoice is None or area>oldchoice[1]:choices[b,l]=(int(k),area)
        if not choices:return nxt.new_zeros((len(nxt),4)),count,count.new_zeros(len(nxt))
        batches=[];legs=[];x1=[];x3=[];areas=[]
        p=self.coeff
        for (b,l),(k,area) in choices.items():
            r=self.rows[k];t=float(clock[b])
            x1.append(self.m1_at(k,t))
            release=r[3:7]+p['release_prior_blend']*(r[7:11]-r[3:7])
            x3.append(release*np.exp(-p['positive_recovery_rate_s_inv']*max(t-r[2],0.)))
            batches.append(b);legs.append(l);areas.append(area)
        z=positions[batches,legs,2].astype(float)-.022
        x1=np.array(x1);x3=np.array(x3);areas=np.array(areas)
        def evaluate(pos):
            return p['contact_gain']*responses(self.law,x3,pos,areas)-responses(self.law,x1,pos,areas)
        v=evaluate(z);jac=(evaluate(z+1e-8)-evaluate(z-1e-8))/2e-8
        # Preserve the verified within-mode vertical derivative. Source lookup,
        # support identity and force onset remain discrete.
        it=torch.as_tensor(batches,device=nxt.device);il=torch.as_tensor(legs,device=nxt.device)
        height=xyz[it,il,2]
        values=nxt.new_tensor(v)+(height-height.detach())*nxt.new_tensor(jac)
        delta=nxt.new_zeros((len(nxt),4)).index_put((it,il),values)
        n_valid=count.new_zeros(len(nxt)).index_add(0,it,torch.ones_like(it))
        return delta,count,n_valid
