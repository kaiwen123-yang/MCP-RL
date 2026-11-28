"""Finite implicit actuator update, configuration-dependent dynamics and causal FK closure."""
import mujoco
import numpy as np
import torch
from s1_w59_model import insert,quaternion_step
from s1_w59_model_fields import DT,STATE
from s1_w59_model_rigid import RigidLinearization


class DiscreteExecution:
    def __init__(self,variant,binary):
        self.variant=variant
        self.engine=RigidLinearization(binary) if variant in ('configuration','kinematics') else None
        self.torque_defect_max=0.
        self.steps=0
        self.singular=[]

    def operators(self,x,origin,active):
        if self.engine is None:
            return origin
        state=x.detach().cpu().numpy()
        ok=active.cpu().numpy() & np.isfinite(state).all(axis=1)
        matrices=origin.detach().cpu().numpy().copy()
        for i in np.flatnonzero(ok):
            try:
                matrices[i]=self.engine(state[i])
            except np.linalg.LinAlgError:
                # An undefined operator terminates this origin; preserve it in
                # the denominator and retain the other independent trajectories.
                matrices[i].fill(np.nan)
                self.singular.append((int(i),self.steps+1))
        return origin.new_tensor(matrices)

    def fk(self,state,active):
        arr=state.detach().cpu().numpy().copy()
        engine=self.engine
        model,data=engine.model,engine.data
        jac=np.empty((3,model.nv))
        finite=active.cpu().numpy() & np.isfinite(arr[:,STATE['qpos']]).all(axis=1) & np.isfinite(arr[:,STATE['qvel']]).all(axis=1)
        for i in np.flatnonzero(finite):
            data.qpos[:]=arr[i,STATE['qpos']]
            data.qvel[:]=arr[i,STATE['qvel']]
            mujoco.mj_kinematics(model,data)
            mujoco.mj_comPos(model,data)
            velocity=[]
            for g in engine.gids:
                mujoco.mj_jacGeom(model,data,jac,None,g)
                velocity.append(jac @ data.qvel)
            arr[i,STATE['feet']]=data.geom_xpos[engine.gids].ravel()
            arr[i,STATE['foot_velocity']]=np.asarray(velocity).ravel()
        return state.new_tensor(arr)

    def __call__(self,member,x,action,original,active):
        from s1_w59_model_fields import INTERVAL,NS
        rigid=self.operators(x,original,active)
        result,y=member(x,action,rigid)
        q=x[:,STATE['qpos']]
        v=x[:,STATE['qvel']]
        held=.5*(x[:,STATE['held_command']]+result[:,STATE['held_command']])
        mean=.5*(x[:,STATE['measured_force']]+result[:,STATE['measured_force']]).reshape(-1,4,3)
        force=torch.cat((mean[:,:,:2],y[:,INTERVAL['load_mean']].clamp_min(0).unsqueeze(-1)),-1)
        net_in=torch.cat(((x-member.mean_s)/member.scale_s,(action-member.a_mean)/member.a_span),-1)
        dv=member.net(net_in)[:,:NS][:,STATE['qvel']]*member.delta_s[STATE['qvel']]
        # Keep the frozen 50k velocity residual unchanged for the integrator ablation.
        free=v+dv+DT*(rigid[:,:,0]+(rigid[:,:,13:] @ force.flatten(1)[:,:,None]).squeeze(-1))
        drive=rigid[:,:,1:13]
        gain=5.+.5*DT*60.
        rhs=60.*(held-q[:,7:]-.5*DT*v[:,6:])-gain*free[:,6:]
        eye=torch.eye(12,device=x.device,dtype=x.dtype)
        mat=eye+gain*DT*drive[:,6:,:]
        # Symmetric box QP from applying the actual 60/5 affine command at the
        # predicted endpoint velocity and midpoint configuration. Fixed-work
        # projected accelerated iterations also handle torque saturation.
        lmax=mat.abs().sum(-1).amax(-1,keepdim=True).clamp_min(1.)
        rate=1./lmax
        momentum=(lmax.sqrt()-1)/(lmax.sqrt()+1)
        tau=rhs.clamp(-member.t_limit,member.t_limit)
        z=tau
        for _ in range(32):
            update=(z-rate*((mat@z[:,:,None]).squeeze(-1)-rhs)).clamp(-member.t_limit,member.t_limit)
            z=update+momentum*(update-tau)
            tau=update
        defect=(tau-(tau+rhs-(mat@tau[:,:,None]).squeeze(-1)).clamp(-member.t_limit,member.t_limit)).abs()
        usable=defect[active]
        if usable.numel():
            valid=usable[torch.isfinite(usable)]
            if valid.numel():
                self.torque_defect_max=max(self.torque_defect_max,float(valid.max().item()))
        self.steps+=1
        vel=free+DT*(drive @ tau[:,:,None]).squeeze(-1)
        midpoint=.5*(v+vel)
        pos=torch.cat((q[:,:3]+DT*midpoint[:,:3],quaternion_step(q[:,3:7],midpoint[:,3:6]),q[:,7:]+DT*midpoint[:,6:]),-1)
        result=insert(insert(result,STATE['qvel'],vel),STATE['qpos'],pos)
        if self.variant=='kinematics':
            result=self.fk(result,active)
        return result,y
