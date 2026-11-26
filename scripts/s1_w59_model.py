"""Action-conditioned execution world models, with a held request on the 20 ms clock."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch import nn
from s1_w59_model_fields import STATE, INTERVAL, NS, NI, DT, MASS, pack_state

EVENTS=('touch','rebound','bearing_on','request','handover','hard_failure','truncated','complete','hold_broken')
BINARY=tuple(i for k in EVENTS for i in range(INTERVAL[k].start,INTERVAL[k].stop))
CONT=tuple(i for i in range(NI) if i not in BINARY)
U_MIN=(-1.0472,-1.5708,-2.7227)*4
U_MAX=(1.0472,3.4907,-.83776)*4
FMAX=(23.7,23.7,45.43)*4


def insert(x,where,value):
    return torch.cat((x[...,:where.start],value,x[...,where.stop:]),dim=-1)


def quaternion_step(q,omega):
    w,v=q[...,:1],q[...,1:]
    dq=torch.cat((-(v*omega).sum(dim=-1,keepdim=True),w*omega+torch.linalg.cross(v,omega)),dim=-1)
    return nn.functional.normalize(q+.5*DT*dq,dim=-1)


def critic_features(state):
    return torch.cat((state[...,STATE['time']]/60,state[...,STATE['progress']],state[...,STATE['phase']],
        state[...,STATE['measured_force']].unflatten(-1,(4,3))[...,2]/(MASS*9.81/4),state[...,STATE['hold']]/2),dim=-1)


def addressed(state):
    center=state[...,STATE['feet']].reshape(-1,4,3)
    targets=state[...,STATE['target_xy']].reshape(-1,4,2)
    points=torch.cat((center[:,:,:2],targets),dim=1)
    half=(.001**.5)/2;lo=points-half;hi=points+half
    offsets=points.new_tensor([[i,j] for i in range(3) for j in range(3)])
    indices=(lo/.02).floor().unsqueeze(2)+offsets
    left=indices*.02
    overlap=(torch.minimum(hi.unsqueeze(2),left+.02)-torch.maximum(lo.unsqueeze(2),left)).clamp_min(0).prod(dim=-1)/.001
    valid=(points[:,:,0]>=.8+half)&(points[:,:,0]<=2.2-half)&(points[:,:,1].abs()<=.6-half)
    valid=torch.cat((valid[:,:4],valid[:,4:]&(state[...,STATE['target_valid']]>.5)),dim=1)
    weights=overlap*valid.unsqueeze(-1)
    ids=((indices[...,0]-40)*60+indices[...,1]+30).long().clamp(0,4199)
    return ids,weights


def query_world(state,world):
    ids,w=addressed(state)
    batch=torch.arange(len(state),device=state.device)[:,None,None]
    info=world[batch,ids];loaded=info[...,2]>=0;released=(info[...,4]>0)&~loaded
    t=state[...,STATE['time']].reshape(-1,1,1)
    fraction=(w*released).sum(-1)
    memory=torch.stack(((w*loaded).sum(-1),(w*released*info[...,0]).sum(-1),
        (w*released*(t-info[...,1]).clamp_min(0)).sum(-1)/fraction.clamp_min(1e-12),fraction),dim=-1)
    state=insert(state,STATE['memory'],memory.flatten(1))
    return insert(state,STATE['memory_valid'],w.sum(-1))


def private_world(state,world):
    ids,w=addressed(state)
    batch=len(state);protected=state.new_zeros(batch,4200);target=protected.clone()
    loaded=(state[...,STATE['measured_force']].reshape(batch,4,3)[:,:,2]>1).to(w.dtype)
    protected.scatter_add_(1,ids[:,:4].flatten(1),(w[:,:4]*loaded[:,:,None]).flatten(1))
    target.scatter_add_(1,ids[:,4:].flatten(1),w[:,4:].flatten(1))
    reset=(target>0)&(protected==0)&(world[...,2]<0)
    virgin=world.new_tensor([0,-1,-1,-1,0]).view(1,1,5)
    return torch.where(reset.unsqueeze(-1),virgin,world)


def private_state(state,world=None):
    if world is None:
        raise ValueError('Private footprint ownership requires the causal world table')
    return query_world(state,private_world(state,world))


def advance_world(state,world,average_force=None):
    ids,w=addressed(state);ids=ids[:,:4];w=w[:,:4]
    batch=len(state);feet=state[...,STATE['feet']].reshape(batch,4,3)
    normals=(state[...,STATE['measured_force']].reshape(batch,4,3)[:,:,2] if average_force is None else average_force)
    active=(normals>1)[:,:,None]&(w>0)
    # Lowest observed bottom owns a shared cell. All candidates have independent tables.
    depth=(.022-feet[:,:,2]).clamp_min(0)
    zgap=torch.where(active,feet[:,:,2,None].expand_as(w),torch.full_like(w,1e4))
    minz=torch.full_like(world[:,:,0],1e4).scatter_reduce(1,ids.flatten(1),zgap.flatten(1),reduce='amin')
    owner=torch.arange(4,device=w.device,dtype=w.dtype)[None,:,None].expand_as(w)
    owner=torch.where(active&(zgap<=minz.gather(1,ids.flatten(1)).reshape_as(w)),owner,torch.full_like(owner,4))
    new_owner=world.new_full((batch,4200),4).scatter_reduce(1,ids.flatten(1),owner.flatten(1),reduce='amin')
    loaded=new_owner<4;idx=new_owner.long().clamp(max=3)
    new_depth=depth.gather(1,idx)
    t=state[:,STATE['time']]
    return torch.stack((torch.where(loaded,new_depth,world[...,0]),
        torch.where((world[...,2]>=0)&~loaded,t,world[...,1]),torch.where(loaded,new_owner,-1.),
        torch.where(loaded,new_owner,world[...,3]),torch.where(loaded,1.,world[...,4])),dim=-1)


class ExecutionMember(nn.Module):
    def __init__(self,kind,stats,width=128):
        super().__init__()
        self.kind=kind;self.width=width
        self.net=nn.Sequential(nn.Linear(NS+3,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,NS+NI))
        for name,value in stats.items():self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        self.register_buffer('u_min',torch.tensor(U_MIN))
        self.register_buffer('u_max',torch.tensor(U_MAX))
        self.register_buffer('t_limit',torch.tensor(FMAX))
        self.register_buffer('a_mean',torch.tensor([0,.05,0.]))
        self.register_buffer('a_span',torch.tensor([.05,.03,.10]))
        self.register_buffer('cont_indices',torch.tensor(CONT,dtype=torch.long))
        self.register_buffer('binary_indices',torch.tensor(BINARY,dtype=torch.long))
        self.register_buffer('event_mask',torch.tensor([i in BINARY for i in range(NI)]))
        self.register_buffer('copy_mask',torch.tensor([any(STATE[k].start<=i<STATE[k].stop for k in
            ('version','request_valid','control_valid','route_valid')) for i in range(NS)]))

    def forward(self,x,action,rigid):
        u=(action-self.a_mean)/self.a_span
        hidden=self.net(torch.cat(((x-self.mean_s)/self.scale_s,u),dim=-1))
        sn,yn=hidden.split([NS,NI],dim=-1)
        y=torch.where(self.event_mask,torch.sigmoid(yn+self.event_bias),yn*self.scale_y+self.mean_y)
        result=(self.mean_s+sn*self.scale_s if self.kind=='blackbox' else x+sn*self.delta_s)
        if self.kind=='structured':
            q=x[...,STATE['qpos']];v=x[...,STATE['qvel']]
            held=torch.maximum(torch.minimum(result[...,STATE['held_command']],self.u_max),self.u_min)
            tau=(60*(.5*(held+x[...,STATE['held_command']])-q[...,7:])-5*v[...,6:])
            tau=torch.maximum(torch.minimum(tau,self.t_limit),-self.t_limit)
            mean=(.5*(x[...,STATE['measured_force']]+result[...,STATE['measured_force']])).unflatten(-1,(4,3))
            mean=torch.cat((mean[...,:2],y[...,INTERVAL['load_mean']].clamp_min(0).unsqueeze(-1)),dim=-1)
            wrench=torch.cat((torch.ones_like(tau[...,:1]),tau,mean.flatten(-2)),dim=-1)
            acc=(rigid*wrench.unsqueeze(-2)).sum(-1)
            vel=v+DT*acc+sn[...,STATE['qvel']]*self.delta_s[STATE['qvel']]
            mid=.5*(v+vel)
            position=torch.cat((q[...,:3]+DT*mid[...,:3],quaternion_step(q[...,3:7],mid[...,3:6]),
                q[...,7:]+DT*mid[...,6:]),dim=-1)
            result=insert(result,STATE['qvel'],vel)
            result=insert(result,STATE['qpos'],position)
            result=insert(result,STATE['held_command'],held)
            feet=x[...,STATE['feet']]+DT*.5*(x[...,STATE['foot_velocity']]+result[...,STATE['foot_velocity']])
            result=insert(result,STATE['feet'],feet+sn[...,STATE['feet']]*self.delta_s[STATE['feet']])
            solved=result[...,STATE['force_solved']].unflatten(-1,(4,3))
            fz=solved[...,2:].clamp_min(0);margin=fz*.5/(2**.5)
            solved=torch.cat((torch.maximum(torch.minimum(solved[...,:2],margin),-margin),fz),dim=-1)
            result=insert(result,STATE['force_solved'],solved.flatten(-2))
        result=torch.where(self.copy_mask,x,result)
        result=insert(result,STATE['time'],x[...,STATE['time']]+DT)
        if hasattr(self,'action_coefficients'):
            from s1_w59_model_calibration import design
            correction=design(x,action)@self.action_coefficients
            delta,dy=correction.split((NS,NI),dim=-1)
            result=result+delta
            # Keep zero residual exactly identical to the original member.
            # Only the proposed increment is limited by its available margin.
            q=result[...,STATE['held_command']]
            before=q-delta[...,STATE['held_command']]
            result=insert(result,STATE['held_command'],before+torch.minimum(torch.maximum(delta[...,STATE['held_command']],torch.minimum(self.u_min-before,torch.zeros_like(q))),torch.maximum(self.u_max-before,torch.zeros_like(q))))
            y=torch.where(self.event_mask,(y+dy).clamp(0,1),y+dy)
        return result,y

    def install_action_coefficients(self,coefficients):
        self.register_buffer('action_coefficients',coefficients.to(self.mean_s),persistent=False)

    def save(self,path,optimizer=None,metadata=None):
        target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix('.pending')
        torch.save(dict(kind=self.kind,width=self.width,model=self.state_dict(),stats={k:getattr(self,k).cpu() for k in
            ('mean_s','scale_s','delta_s','mean_y','scale_y','event_bias')},optimizer=optimizer.state_dict() if optimizer else None,
            metadata=metadata or {},action_coefficients=getattr(self,'action_coefficients',None)),tmp)
        tmp.replace(target)

    @classmethod
    def load(cls,path,device='cpu'):
        c=torch.load(path,map_location=device,weights_only=False)
        model=cls(c['kind'],c['stats'],c['width']).to(device)
        model.load_state_dict(c['model']);model.metadata=c.get('metadata',{})
        if c.get('action_coefficients') is not None:model.install_action_coefficients(c['action_coefficients'])
        return model


@lru_cache(maxsize=1)
def _engine():
    from s1_w59_model_rigid import RigidLinearization
    from s1_w59_model_bank import ROOT,REL
    base=ROOT.parent/'MCP-RL-w58'/'outputs'/REL/'w58-g0-m1-p2-full-a-original-e30-a01/model.mjb'
    return RigidLinearization(base)


def local_dynamics(states):
    a=states.detach().cpu().numpy().reshape(-1,NS)
    return states.new_tensor(np.stack([_engine()(x) for x in a])).reshape(*states.shape[:-1],18,25)


class ExecutionEnsemble(nn.Module):
    def __init__(self,members):
        super().__init__();self.members=nn.ModuleList(members);self.device=next(members[0].parameters()).device

    @classmethod
    def load(cls,checkpoints,device='cuda'):
        return cls([ExecutionMember.load(p,device).eval() for p in checkpoints])

    def step(self,state,action,rigid=None):
        if rigid is None:rigid=local_dynamics(state)
        pairs=[m(state,action,rigid) for m in self.members]
        return tuple(torch.stack([p[i] for p in pairs]).mean(0) for i in (0,1))

    def bind_material_publication(self,record):
        from s1_w59_model_reload import PublishedMaterial
        self.material_publication=PublishedMaterial(record)

    def _material(self,nxt,y,old_world,rigid,material,seen):
        new_world=advance_world(nxt,old_world,y[...,INTERVAL['load_mean']])
        renewed=(old_world[...,2]<0)&(old_world[...,4]>0)&(new_world[...,2]>=0)&(old_world[...,3]!=new_world[...,2])
        count=renewed.sum(-1);branch=(renewed&seen).sum(-1)
        applied=torch.zeros_like(count)
        if material=='M3':
            if not hasattr(self,'material_publication'):
                raise ValueError('M3 requires paired M0/M1 physical release publications')
            delta,_,applied=self.material_publication.correction(nxt,old_world,new_world,seen)
            normal=y[...,INTERVAL['load_mean']]
            delta=(normal+delta).clamp_min(0)-normal.clamp_min(0)
            y=insert(y,INTERVAL['load_mean'],normal+delta)
            y=insert(y,INTERVAL['load_peak'],torch.maximum(y[...,INTERVAL['load_peak']],(normal+delta).clamp_min(0)))
            f=nxt[...,STATE['measured_force']].unflatten(-1,(4,3))
            nxt=insert(nxt,STATE['measured_force'],torch.cat((f[...,:2],(f[...,2]+delta).clamp_min(0)[...,None]),-1).flatten(-2))
            impulse=(rigid[...,15::3]*delta[:,None,:]).sum(-1)*DT
            nxt=insert(nxt,STATE['qvel'],nxt[...,STATE['qvel']]+impulse)
            q=nxt[...,STATE['qpos']]
            nxt=insert(nxt,STATE['qpos'],torch.cat((q[...,:3]+.5*DT*impulse[...,:3],q[...,3:7],q[...,7:]+.5*DT*impulse[...,6:]),-1))
            new_world=advance_world(nxt,old_world,y[...,INTERVAL['load_mean']])
        elif material!='M1':raise ValueError(material)
        return query_world(nxt,new_world),y,new_world,count,branch,applied

    def step_with_world(self,state,action,world,*,material='M1',private=False,rigid=None):
        if private:world=private_world(state,world)
        if rigid is None:rigid=local_dynamics(state)
        nxt,y=self.step(query_world(state,world),action,rigid)
        # At a one-step policy call the supplied world is the current source;
        # for multi-step M3 pass a live source at a true decision, then use
        # predict_candidates or retain the publication through the rollout.
        seen=(world[...,2]>=0)
        return self._material(nxt,y,world,rigid,material,seen)[:3]

    def predict_candidates(self,state,requests,steps=32,*,world,material='M1',private=False,successor='stop',successor_request=None,successor_policy=None):
        if successor not in ('stop','nominal','fixed','policy'):raise ValueError(successor)
        if successor=='fixed' and successor_request is None:raise ValueError('fixed successor request required')
        if successor=='policy' and successor_policy is None:raise ValueError('source policy callback required')
        state=torch.as_tensor(state,dtype=torch.float32,device=self.device).reshape(-1,NS)
        actions=torch.as_tensor(requests,dtype=torch.float32,device=self.device)
        world=torch.as_tensor(world,dtype=torch.float32,device=self.device).reshape(-1,4200,5)
        n=len(actions);base=local_dynamics(state);state=state.expand(n,-1);world=world.expand(n,-1,-1)
        rigid=base.expand(n,-1,-1)
        if private:world=private_world(state,world)
        returns=[]
        for member in self.members:
            s=query_world(state,world);memory=world;ss=[s];yy=[];vv=[];events=[];ends=[];cross=[];branch=[];corrected=[]
            held=actions;nominal=actions.new_tensor([0.,.05,0.]).expand_as(held)
            seen=torch.zeros_like(memory[...,0],dtype=torch.bool)
            valid=(s[...,STATE['phase']][:,5]<.5)
            for _ in range(steps):
                nxt,y=member(s,held,rigid)
                nxt,y,advanced,c,b,a=self._material(nxt,y,memory,rigid,material,seen)
                seen=seen|((advanced[...,2]>=0)&valid[:,None])
                memory=torch.where(valid[:,None,None],advanced,memory)
                s=torch.where(valid[:,None],nxt,s)
                request=(y[...,INTERVAL['request']].squeeze(-1)>=.5)&valid
                done=((y[...,INTERVAL['hard_failure']].squeeze(-1)>=.5)|(y[...,INTERVAL['complete']].squeeze(-1)>=.5))&valid
                ss.append(s);yy.append(torch.where(valid[:,None],y,0.));vv.append(valid)
                events.append(request);ends.append(done);cross.append(c*valid);branch.append(b*valid);corrected.append(a*valid)
                valid=valid&~done
                if successor=='nominal':held=torch.where(request[:,None],nominal,held)
                elif successor=='fixed':
                    fixed=torch.as_tensor(successor_request,device=held.device,dtype=held.dtype).expand_as(held)
                    held=torch.where(request[:,None],fixed,held)
                elif successor=='policy':
                    issued=(request&~done).nonzero(as_tuple=True)[0]
                    if len(issued):
                        with torch.no_grad():replacement=successor_policy(s[issued].detach(),memory[issued].detach())
                        if isinstance(replacement,tuple):replacement=replacement[0]
                        held=held.index_copy(0,issued,replacement.detach())
                else:valid=valid&~request
            returns.append(tuple(torch.stack(v,dim=1) for v in (ss,yy,vv,events,ends,cross,branch,corrected)))
        xs,ys,used,issued,done,cross,branch,corrected=[torch.stack([r[j] for r in returns]) for j in range(8)]
        return dict(trajectory=xs,interval=ys,valid_steps=used,request_event_mask=issued,done=done,
            cross_reload_cells=cross,cross_reload_from_branch=branch,corrected_published_reload=corrected,
            spread=ys.std(dim=0,unbiased=False))
