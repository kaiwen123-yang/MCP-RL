"""W54 deterministic pre-contact action map shared with every native rollout/FD."""
import ctypes,json
from pathlib import Path
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
class PrecontactE:
    def __init__(self,model):
        self.model=model
        self.library=ctypes.CDLL(str(ROOT/'work/r11-frozen/S5-r44/lib/w54_e_core.so'))
        self.fn=self.library.w54_shape
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.fn.argtypes=[ctypes.c_int]+[ptr]*11
        self.fn.restype=ctypes.c_int
        self.qadr=model.jnt_qposadr[model.actuator_trnid[:,0]]
        self.vadr=model.jnt_dofadr[model.actuator_trnid[:,0]]
        self.feet=[model.geom(n).id for n in ('FL','FR','RL','RR')]
        self.floor=float(model.numeric('s1_flat_ground_height').data[0])
    def apply(self,data,raw):
        m=self.model
        mujoco.mj_forward(m,data)  # same current-state preparation as native E
        M=np.empty((m.nv,m.nv));mujoco.mj_fullM(m,M,data.qM)
        H=np.empty((4,m.nu));gap=np.empty(4);vz=np.empty(4);material=np.empty(4)
        for i,g in enumerate(self.feet):
            jp=np.empty((3,m.nv));jr=np.empty((3,m.nv));mujoco.mj_jacGeom(m,data,jp,jr,g)
            sol=np.linalg.solve(M,jp[2])
            H[i]=sol[self.vadr]*m.actuator_gear[:,0]
            p=data.geom_xpos[g];r=m.geom_size[g,0]
            gap[i]=p[2]-r-self.floor;vz[i]=jp[2]@data.qvel
            material[i]=float(.8-r<=p[0]<=2.2+r and abs(p[1])<=.6+r)
        raw=np.ascontiguousarray(raw,dtype=np.float64);out=np.empty(m.nu);diag=np.empty((4,8))
        valid=self.fn(m.nu,np.ascontiguousarray(data.qpos[self.qadr]),np.ascontiguousarray(data.qvel[self.vadr]),
                raw,np.ascontiguousarray(m.actuator_ctrlrange),np.ascontiguousarray(m.actuator_forcerange),
                H,gap,vz,material,out,diag)
        if not valid:raise RuntimeError("W54 E original torque/input intersection empty")
        return out,diag
