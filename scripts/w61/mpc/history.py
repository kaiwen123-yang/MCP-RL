"""Ablate causal cross-foot memory and common posterior scenarios inside a real MPC-controlled Go2."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
FROZEN=Path('/root/autodl-tmp/MCP-RL-w59')
sys.path.insert(0,str(FROZEN/'scripts'))
from s1_w57_material import W57MaterialObserver
from s1_w56_f_r60_backend import FController as StableF
from w61.mpc.controller import TopologyMPC
from w61.mpc.problem import build_problem


class PrivateBank(W57MaterialObserver):
    """Keep an independent M1 grid for each leg, and the ordinary common M1 for WBC."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.private=[W57MaterialObserver(*args,**kwargs) for _ in range(4)]

    def completed(self,start,end,bottom,velocity,measured_force):
        super().completed(start,end,bottom,velocity,measured_force)
        for leg,observer in enumerate(self.private):
            positions=bottom.copy();forces=np.zeros_like(measured_force)
            # Suppress the other legs' measurements by placing them outside the modeled strip.
            positions[np.arange(4)!=leg,0]=-2.
            forces[leg]=measured_force[leg]
            observer.completed(start,end,positions,velocity,forces)

    def observe(self,t,bottom,velocity):
        super().observe(t,bottom,velocity)
        for observer in self.private:observer.observe(t,bottom,velocity)


class MemoryController(TopologyMPC,StableF):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        self.memory_log=[]
        self.fit_rows=[]
        self.last_fit_tick=-1

    def _update_schedule(self,t,data,mq,mission,soft,material_info,observed,grf):
        # 20-ms support observations with the actual measured force and bottom.
        if round(t/.002)%10==0:
            pub=material_info['force_boundary_bottom']
            legmask=np.asarray(soft) & observed & (grf[:,2]>1.)
            if np.any(legmask):
                self.fit_rows.extend(zip(np.maximum(-pub[legmask,2],0.),grf[legmask,2]))
        return super()._update_schedule(t,data,mq,mission,soft,material_info,observed,grf)

    def construct_problem(self,obs,mission,info):
        observer=info['material_observer'];arm=self.c_cfg['history']
        k0=1227272.7272727273
        stiffness=np.asarray([.70,1.,1.30])*k0
        prob=np.array([0.,1.,0.])
        if arm=='posterior' and self.fit_rows:
            depth,fz=np.asarray(self.fit_rows).T
            # The width is a force model discrepancy in N. Average over the causal
            # window, not a product of falsely independent adjacent observations.
            err=(depth[:,None]*stiffness[None,:]-fz[:,None])/30.
            logits=np.log([.25,.50,.25])-.5*np.mean(err**2,axis=0)
            prob=np.exp(logits-logits.max());prob/=prob.sum()
        lookup={}
        def memory(leg,key):
            lookup[leg,key]=observer.history(key)
            ground=observer.private[leg].ground if arm=='private' else observer.ground
            idx=ground.ids.get(key)
            if idx is None or (arm=='virgin' and not ground.loaded[idx]):return 0.
            return max(0.,float(-ground.h*ground.e[idx,0]))
        yaw=float(obs['x0'][2])
        v=float(np.dot(mission.v[:2],[np.cos(yaw),np.sin(yaw)]))
        p=build_problem(obs,v,np.ones(4),prob,0.,memory=memory,stiffness=stiffness,compliant=(True,True,True))
        cross=sum(bool(v['ever_loaded'] and not v['loaded'] and v['last_owner']!=leg) for (leg,key),v in lookup.items())
        self.memory_log.append(dict(phase=mission.phase,speed=v,posterior=prob.tolist(),support_rows=len(self.fit_rows),
                    prospective_world_foot_queries=len(lookup),foreign_released_queries=cross,
                    max_initial_depth_m=float(p['initial_delta'].max())))
        return p

    def at_rollout_end(self,**kw):
        super().at_rollout_end(**kw)
        (self.c_dir/'memory.json').write_text(json.dumps(self.memory_log,indent=2)+'\n')


def main():
    from w61.material.live_observer import run
    p=argparse.ArgumentParser();p.add_argument('spec',type=Path);p.add_argument('case')
    p.add_argument('arm');p.add_argument('destination',type=Path);a=p.parse_args()
    spec=json.loads(a.spec.read_text())
    def factory(model,data,feet,trunk,out,cfg):
        cfg['w61']=dict(role='hybrid',out=str(out),mpc_transfer={**spec['transfer'],'history':a.arm})
        return MemoryController(model,data,feet,trunk,out,cfg)
    run(a.destination,'cpu',a.case,a.spec,controller_factory=factory,
        observer_factory=PrivateBank if a.arm=='private' else W57MaterialObserver)


if __name__=='__main__': main()
