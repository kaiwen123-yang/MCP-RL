"""Current-state WBC linearization, in physical units, for the W59 short model horizon."""
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from s1_w59_model_fields import STATE, BG

LEGS=('FL','FR','RL','RR')


class RigidLinearization:
    """M^-1[-h,S^T,J^T] at an observed or predicted configuration."""
    def __init__(self,binary):
        self.model=mujoco.MjModel.from_binary_path(str(binary))
        self.data=mujoco.MjData(self.model)
        m=self.model
        self.gids=[m.geom(leg).id for leg in LEGS]
        self.floor=m.geom('floor').id
        self.jac=np.empty((3,18))
        self.full=np.empty((18,18))
        self.unit=np.eye(18)[:,6:]
        self.rhs=np.zeros((18,25))
        self.rhs[:,1:13]=self.unit
        self.radii=m.geom_size[self.gids,0].copy()

    def __call__(self,state):
        m,d=self.model,self.data
        d.qpos[:]=state[STATE['qpos']];d.qvel[:]=state[STATE['qvel']]
        mujoco.mj_forward(m,d)
        mujoco.mj_fullM(m,self.full,d.qM)
        self.rhs[:,0]=d.qfrc_passive-d.qfrc_bias
        for leg,gid in enumerate(self.gids):
            point=d.geom_xpos[gid].copy();point[2]-=self.radii[leg]
            half=np.sqrt(.001)/2
            soft=.8+half<=point[0]<=2.2-half and abs(point[1])<=.6-half
            if not soft:
                for c in d.contact:
                    if (int(c.geom1)==gid and int(c.geom2)==self.floor) or (int(c.geom2)==gid and int(c.geom1)==self.floor):
                        point=c.pos;break
            mujoco.mj_jac(m,d,self.jac,None,point,int(m.geom_bodyid[gid]))
            self.rhs[:,13+leg*3:16+leg*3]=self.jac.T
        return np.linalg.solve(self.full,self.rhs)

def main():
    from s1_w59_model_bank import WORK,OUT,write
    rows=json.loads((OUT/'source-inventory.json').read_text())['attempts']
    (WORK/'rigid').mkdir(exist_ok=True,parents=True);done=[];start=time.time()
    for row in rows:
        if not row['trajectory']:continue
        p=Path(row['root']);ep=row['episode'];out=WORK/'rigid'/(ep+'.npy')
        if not out.exists():
            statefile=WORK/'bank'/(ep+'.npz')
            while not statefile.is_file():time.sleep(2)
            # Reuse this recording's model binary (identical geometry alone is insufficient).
            if not (p/'model.mjb').exists():
                done.append(dict(episode=ep,status='source_model_binary_missing'));continue
            engine=RigidLinearization(p/'model.mjb')
            with np.load(statefile) as f: states=f['state']
            src=np.empty((len(states),18,25),np.float32)
            for i,state in enumerate(states):src[i]=engine(state)
            with out.with_suffix('.tmp').open('wb') as f:np.save(f,src)
            out.with_suffix('.tmp').replace(out)
        done.append(dict(episode=ep,status='complete',boundaries=int(np.load(out,mmap_mode='r').shape[0])))
        print(ep,done[-1],flush=True)
    write(OUT/'rigid.json',dict(episodes=done,elapsed_s=time.time()-start,operator='observed-state M^-1[-(bias-passive),S^T,J^T] with actual TM02 application point / current native contact point',**BG))

if __name__=='__main__':main()
