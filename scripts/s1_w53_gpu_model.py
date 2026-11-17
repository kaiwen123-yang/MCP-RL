"""FP64 batched rigid Go2 kinematics and composite inertia for offline W53 C.

This module never steps a plant or computes finite differences. The model
parameters are copied from the exact MjModel used by F, not a robot surrogate.
The supported joint tree is the frozen free-base Go2 with 12 revolute joints.
"""
import numpy as np
import torch


def quaternion_matrix(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3,3))


def skew(v):
    z = torch.zeros_like(v[..., 0])
    x, y, zz = v.unbind(-1)
    return torch.stack((z, -zz, y, zz, z, -x, -y, x, z), -1).reshape(v.shape[:-1]+(3,3))


class Go2BatchModel:
    def __init__(self, model, foot_geom_ids, device='cuda'):
        self.device = device
        self.model = model
        self.feet = list(map(int, foot_geom_ids))
        self.nv = int(model.nv)
        self.root = int(model.jnt_bodyid[0])
        self.body_ids = []
        for b in range(1, model.nbody):
            p = b
            while p and p != self.root:
                p = int(model.body_parentid[p])
            if p == self.root:
                self.body_ids.append(b)
        self.mass = float(sum(model.body_mass[b] for b in self.body_ids))
        self.parameters = {}
        for name in ('body_pos', 'body_quat', 'body_ipos', 'body_iquat',
                     'body_inertia', 'body_mass', 'jnt_axis', 'jnt_pos',
                     'qpos0', 'geom_pos'):
            self.parameters[name] = torch.tensor(np.asarray(getattr(model, name)), dtype=torch.float64, device=device)
        self.fixed_rotation = quaternion_matrix(self.parameters['body_quat'])
        self.inertia_rotation = quaternion_matrix(self.parameters['body_iquat'])
        self.eye = torch.eye(3, dtype=torch.float64, device=device)
        self.ancestors = []
        for g in self.feet:
            b = int(model.geom_bodyid[g]); joints = []
            while b and b != self.root:
                for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b]+model.body_jntnum[b])):
                    joints.append(j)
                b = int(model.body_parentid[b])
            self.ancestors.append(joints)
        # These are model-family requirements, not state validity gates/fallbacks.
        if model.nq != 19 or model.nv != 18 or int(model.jnt_type[0]) != 0 or np.any(model.jnt_type[1:] != 3):
            raise ValueError('W53 C supports the frozen nq19/nv18 free-base/revolute Go2 model only')

    def evaluate(self, qpos):
        m, p = self.model, self.parameters
        n = qpos.shape[0]
        zero = torch.zeros((n,3), dtype=qpos.dtype, device=qpos.device)
        origin = {0: zero, self.root: qpos[:, :3]}
        rotation = {0: self.eye.expand(n,3,3), self.root: quaternion_matrix(qpos[:, 3:7])}
        axes, anchors = {}, {}
        for b in self.body_ids:
            if b == self.root:
                continue
            parent = int(m.body_parentid[b])
            R = rotation[parent] @ self.fixed_rotation[b]
            pos = origin[parent] + (rotation[parent] @ p['body_pos'][b].unsqueeze(-1)).squeeze(-1)
            for j in range(int(m.body_jntadr[b]), int(m.body_jntadr[b]+m.body_jntnum[b])):
                axis = p['jnt_axis'][j]
                axes[j] = (R @ axis.unsqueeze(-1)).squeeze(-1)
                anchors[j] = pos + (R @ p['jnt_pos'][j].unsqueeze(-1)).squeeze(-1)
                angle = qpos[:, int(m.jnt_qposadr[j])] - p['qpos0'][int(m.jnt_qposadr[j])]
                K = skew(axis)
                dR = self.eye + torch.sin(angle)[:,None,None]*K + (1-torch.cos(angle))[:,None,None]*(K@K)
                R = R @ dR
                pos = anchors[j] - (R @ p['jnt_pos'][j].unsqueeze(-1)).squeeze(-1)
            origin[b], rotation[b] = pos, R
        body_com = torch.stack([origin[b] + (rotation[b] @ p['body_ipos'][b].unsqueeze(-1)).squeeze(-1) for b in self.body_ids], 1)
        mass = p['body_mass'][self.body_ids]
        com = (body_com * mass[None,:,None]).sum(1) / self.mass
        inertia = torch.zeros((n,3,3), dtype=qpos.dtype, device=qpos.device)
        for k,b in enumerate(self.body_ids):
            Ri = rotation[b] @ self.inertia_rotation[b]
            d = body_com[:, k] - com
            inertia = inertia + (Ri * p['body_inertia'][b][None,None,:]) @ Ri.transpose(-1,-2)
            inertia = inertia + mass[k]*(d.square().sum(-1)[:,None,None]*self.eye-d[:,:,None]*d[:,None,:])
        feet, jacobians = [], []
        for g,joints in zip(self.feet,self.ancestors):
            b = int(m.geom_bodyid[g])
            foot = origin[b] + (rotation[b] @ p['geom_pos'][g].unsqueeze(-1)).squeeze(-1)
            J = torch.zeros((n,3,self.nv), dtype=qpos.dtype, device=qpos.device)
            J[:,:,:3] = self.eye
            for k in range(3):
                J[:,:,3+k] = torch.linalg.cross(rotation[self.root][:,:,k], foot-origin[self.root], dim=-1)
            for j in joints:
                J[:,:,int(m.jnt_dofadr[j])] = torch.linalg.cross(axes[j],foot-anchors[j],dim=-1)
            feet.append(foot);jacobians.append(J)
        return {'com':com,'inertia':inertia,'feet':torch.stack(feet,1),
                'jacobian':torch.stack(jacobians,1),'root_rotation':rotation[self.root]}


def assemble_centroidal(mass, inertia, com, feet, jacobian, h_joint,
                        q_joint, v_joint, torque_range, ctrl_range, rotation, omega,
                        dt=.025, friction=.5):
    """Batched FP64 form of F's actual seven centroidal stage blocks."""
    n=len(com);device=com.device;dtype=com.dtype
    pitch=torch.asin(-rotation[:,2,0]);yaw=torch.atan2(rotation[:,1,0],rotation[:,0,0])
    cy,sy=torch.cos(yaw),torch.sin(yaw);cp=torch.cos(pitch);tp=torch.tan(pitch)
    z=torch.zeros_like(cp);one=torch.ones_like(cp)
    E=torch.stack((cy/cp,sy/cp,z,-sy,cy,z,cy*tp,sy*tp,one),-1).reshape(n,3,3)
    G=torch.linalg.solve(inertia[:,None,:,:],skew(feet-com[:,None,:])).permute(0,2,1,3).reshape(n,3,12)
    eye=torch.eye(3,dtype=dtype,device=device)
    L=eye.repeat(1,4)/mass
    A=torch.eye(12,dtype=dtype,device=device).expand(n,12,12).clone()
    A[:,:3,6:9]=dt*E;A[:,3:6,9:12]=dt*eye
    B=torch.zeros((n,12,12),dtype=dtype,device=device)
    B[:,:3]=.5*dt*dt*(E@G);B[:,3:6]=.5*dt*dt*L
    B[:,6:9]=dt*G;B[:,9:12]=dt*L
    alpha=-torch.linalg.solve(inertia,torch.linalg.cross(omega,(inertia@omega[:,:,None]).squeeze(-1),dim=-1))
    c=torch.zeros((n,12),dtype=dtype,device=device)
    c[:,:3]=.5*dt*dt*(E@alpha[:,:,None]).squeeze(-1);c[:,5]=-.5*9.81*dt*dt
    c[:,6:9]=dt*alpha;c[:,11]=-9.81*dt
    F=torch.zeros((16,12),dtype=dtype,device=device)
    local=torch.tensor([[1.,0.,-friction],[-1.,0.,-friction],[0.,1.,-friction],[0.,-1.,-friction]],dtype=dtype,device=device)
    for leg in range(4):F[4*leg:4*leg+4,3*leg:3*leg+3]=local
    T=-jacobian[:,:,:,6:].reshape(n,12,12).transpose(-1,-2)
    lo=torch.maximum(torque_range[:,0],60.*(ctrl_range[:,0]-q_joint)-5.*v_joint)
    hi=torch.minimum(torque_range[:,1],60.*(ctrl_range[:,1]-q_joint)-5.*v_joint)
    return dict(A=A,B=B,c=c,friction_C=F.expand(n,16,12),torque_C=T,torque_l=lo-h_joint,torque_u=hi-h_joint)
