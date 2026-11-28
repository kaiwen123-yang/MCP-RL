"""A causal execution transition with learned contact-force, held-command and actuator residuals only."""
import torch
from torch import nn
import torch.nn.functional as F
from s1_w59_model import U_MIN,U_MAX,FMAX,quaternion_step,insert
from s1_w59_model_fields import DT,STATE


def features(x,a):
    q=x[:,STATE['qpos']]
    feet=x[:,STATE['feet']].reshape(-1,4,3)
    # Body translations are removed; known geographic footprint validity and the
    # last observed release/history are retained as separate quantities.
    return torch.cat((q[:,2:],x[:,STATE['qvel']],(feet-q[:,:3,None].transpose(1,2)).flatten(1),
        x[:,STATE['foot_velocity']],x[:,STATE['measured_force']],x[:,STATE['held_command']],
        x[:,STATE['command']],x[:,STATE['memory']][:,:16],x[:,STATE['memory_valid']][:,:4],a),-1)


class Response(nn.Module):
    def __init__(self,mean,scale):
        super().__init__()
        self.register_buffer('mean',mean)
        self.register_buffer('scale',scale)
        self.register_buffer('upper',torch.tensor(U_MAX))
        self.register_buffer('lower',torch.tensor(U_MIN))
        self.register_buffer('limit',torch.tensor(FMAX))
        self.net=nn.Sequential(nn.Linear(len(mean),128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,48))

    def forward(self,x,action,rigid,learned=True):
        q=x[:,STATE['qpos']]
        v=x[:,STATE['qvel']]
        held0=x[:,STATE['held_command']]
        f0=x[:,STATE['measured_force']].reshape(-1,4,3)
        if learned:
            response=self.net((features(x,action)-self.mean)/self.scale)
            dcommand,dmean,dend,actuator=response.split(12,-1)
            held=(held0+.4*dcommand.tanh()).clamp(self.lower,self.upper)
            def reaction(d):
                d=d.reshape(-1,4,3)
                # A smooth learned change of normal load about the measured
                # contact; the estimator's tangential execution force stays in the cone.
                z=4.*F.softplus((f0[:,:,2]+70.*d[:,:,2])/4.)
                slope=.5/2**.5
                xy=torch.clamp(f0[:,:,:2]+30*d[:,:,:2],-slope*z[:,:,None],slope*z[:,:,None])
                return torch.cat((xy,z[:,:,None]),-1)
            mean_force=reaction(dmean);end_force=reaction(dend)
            drive_residual=.35*self.limit*torch.tanh(actuator)
        else:
            held=held0;mean_force=end_force=f0;drive_residual=torch.zeros_like(held)
        free=v+DT*(rigid[:,:,0]+(rigid[:,:,13:] @ mean_force.flatten(1)[:,:,None]).squeeze(-1))
        B=rigid[:,:,1:13]
        scale=5.+.5*DT*60.
        rhs=60.*(.5*(held+held0)-q[:,7:]-.5*DT*v[:,6:])-scale*free[:,6:]+drive_residual
        mat=torch.eye(12,device=x.device)+scale*DT*B[:,6:,:]
        lipschitz=mat.abs().sum(-1).amax(-1,keepdim=True).clamp_min(1.)
        momentum=(lipschitz.sqrt()-1)/(lipschitz.sqrt()+1)
        tau=rhs.clamp(-self.limit,self.limit);point=tau
        for _ in range(12):
            new=(point-((mat@point[:,:,None]).squeeze(-1)-rhs)/lipschitz).clamp(-self.limit,self.limit)
            point=new+momentum*(new-tau);tau=new
        vel=free+DT*(B@tau[:,:,None]).squeeze(-1)
        mid=.5*(v+vel)
        pos=torch.cat((q[:,:3]+DT*mid[:,:3],quaternion_step(q[:,3:7],mid[:,3:6]),q[:,7:]+DT*mid[:,6:]),-1)
        y=insert(insert(x,STATE['qpos'],pos),STATE['qvel'],vel)
        y=insert(insert(y,STATE['held_command'],held),STATE['measured_force'],end_force.flatten(1))
        return insert(y,STATE['time'],x[:,STATE['time']]+DT),mean_force,tau
