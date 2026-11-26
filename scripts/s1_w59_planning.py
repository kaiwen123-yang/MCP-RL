"""P3 event-layer search over conditional true-request continuations in the execution WM."""
import numpy as np
import torch

from s1_w59_f_backend import LOWER, NOMINAL, UPPER
from s1_w59_model_fields import DT, INTERVAL, MASS, PHASES, STATE, pack_state
from s1_w59_model_cells import footprint, query_cells


def prepare_private(state,world):
    """Remove released target cells, protecting the whole measured current load footprint."""
    original_tensor=torch.is_tensor(state)
    x=state.detach().cpu().numpy().copy() if original_tensor else np.asarray(state).copy()
    table=world.detach().cpu().numpy().copy() if torch.is_tensor(world) else np.asarray(world).copy()
    feet=x[STATE['feet']].reshape(4,3)
    forces=x[STATE['measured_force']].reshape(4,3)
    xy=x[STATE['target_xy']].reshape(4,2)
    valid=x[STATE['target_valid']]>0
    protected=set()
    for pos,force in zip(feet,forces):
        if force[2]>1.:
            ids,_=footprint(*pos[:2])
            protected.update(int(i) for i in ids if i>=0)
    affected=set()
    for pos in xy[valid]:
        ids,_=footprint(*pos)
        affected.update(int(i) for i in ids if i>=0 and int(i) not in protected
                        and table[i,2]<0 and table[i,4]>0)
    for i in affected:
        table[i]=[0.,-1.,-1.,-1.,0.]
    m,mask=query_cells(table,float(x[STATE['time']][0]),np.vstack((feet[:,:2],xy)))
    m[4:]*=valid[:,None];mask[4:]*=valid
    x[STATE['memory']]=m.ravel()
    x[STATE['memory_valid']]=mask
    return x,table,len(affected)


def controller_state(controller,t,mission,mq,grf):
    data=controller.w59_data
    joint=controller.w59_joint
    targets=np.zeros((4,2)); arrival=np.zeros(4); valid=np.zeros(4)
    for entry in controller.plan_publication:
        if entry['valid']:
            i=entry['leg'];targets[i]=entry['xy'];arrival[i]=entry['target_time_s']-t;valid[i]=1.
    memory,mvalid=controller.causal_cells.query(t,mq['feet'],targets,valid)
    incoming=np.ones(4);incoming[controller.next_pair]=0.
    phase=np.zeros(len(PHASES));phase[PHASES.index(mission.phase)]=1.
    hold=getattr(mission,'stable_since',None)
    ref=getattr(controller,'w59_force_ref',None)
    return pack_state(qpos=data.qpos,qvel=data.qvel,feet=mq['feet'],foot_velocity=mq['foot_velocity'],
        measured_force=grf,held_command=data.ctrl,force_reference=ref if ref is not None else np.zeros((4,3)),
        force_solved=controller.last_force,confirmed=controller.confirmed,bearing=controller.load_bearing,
        transfer=controller.transfer_fraction,incoming=incoming,request_age=0.,nominal_due_in=controller.next_liftoff-t,
        active=1.,target_xy=targets,target_in=arrival,target_valid=valid,phase=phase,command=mission.v,
        goal=mission.body_target,hold=0. if hold is None else max(0.,t-hold),progress=joint.legal_route.progress,
        laps=joint.legal_route.completed_laps,time=t,version=[0,0,0,1],memory=memory,memory_valid=mvalid,
        request_valid=1.,control_valid=float(ref is not None),route_valid=1.)


def signed_neighbours(device,dtype):
    nominal=torch.as_tensor(NOMINAL,device=device,dtype=dtype)
    low=torch.as_tensor(LOWER,device=device,dtype=dtype)
    high=torch.as_tensor(UPPER,device=device,dtype=dtype)
    points=nominal.repeat(7,1)
    for j in range(3):
        points[1+2*j,j]=low[j]
        points[2+2*j,j]=high[j]
    return low,nominal,high,points


def components(interval):
    """All terms have a matching measurement in the true 5 kHz/500 Hz records."""
    legal=interval[...,INTERVAL['progress_delta']].squeeze(-1)
    slip=interval[...,INTERVAL['slip_m']].clamp_min(0).sum(-1)
    work=interval[...,INTERVAL['work_J']].squeeze(-1).clamp_min(0)
    failure=interval[...,INTERVAL['hard_failure']].squeeze(-1).clamp(0,1)
    return -5.*legal + DT/60 + (work/(MASS*9.81)+slip)/6., failure


def value(parts,substeps):
    flowing,failed=parts
    flow=flowing[...,:substeps].sum(-1)
    hazard=1.-(1.-failed[...,:substeps]).prod(-1)
    return (flow+10.*hazard).mean(0)


def prediction(wm,state,requests,world,material,private,substeps):
    result=wm.predict_candidates(state,requests,steps=substeps,world=world,material=material,
                                 private=private,successor='nominal')
    valid=result['valid_steps'].to(result['interval'].dtype)
    flowing,hazard=components(result['interval'])
    crossings=result.get('cross_reload_cells')
    own=result.get('cross_reload_from_branch')
    evidence=dict(predicted_next_legal_requests=((result['interval'][...,INTERVAL['request']].squeeze(-1)>=.5)*valid).sum(-1),
        cross_cell_count=(crossings*valid).sum(-1) if crossings is not None else None,
        cross_from_branch=(own*valid).sum(-1) if own is not None else None)
    from s1_w59_forward_domain import candidate_mask
    allowed=candidate_mask(result,getattr(wm,'normal_envelope_N',float('inf')))
    return (flowing*valid,hazard*valid),evidence,allowed


@torch.no_grad()
def select_one(wm,state,world,private,material,substeps):
    device,dtype=state.device,state.dtype
    low,nom,high,axis=signed_neighbours(device,dtype)
    if private:
        masked,world,deleted=prepare_private(state,world)
        state=torch.as_tensor(masked,device=device,dtype=dtype)
    else:deleted=0
    # A deterministic 3D Sobol sequence gives the same fixed candidate budgets to all arms.
    points=torch.quasirandom.SobolEngine(3,scramble=False).draw(129)[1:].to(state)
    c1=torch.cat((axis, low+(high-low)*points[:57]),0)
    first,events1,legal1=prediction(wm,state,c1,world,material,private,substeps)
    rank=value(first,substeps)
    allowed1=legal1&torch.isfinite(rank)
    if not allowed1.any():raise FloatingPointError('P3 generation 1 has no physically supported finite candidate')
    valid_ids=torch.where(allowed1)[0]
    elite=c1[valid_ids[torch.argsort(rank[valid_ids],stable=True)[:8]]]
    center=elite.mean(0)
    sig=elite.std(0,unbiased=False).clamp_min(.02*(high-low))
    z=1.4142135623730951*torch.erfinv(2*points[64:126].clamp(.001,.999)-1)
    c2=torch.cat((nom[None],elite[:1],(center+sig*z).clamp(low,high)),0)
    second,events2,legal2=prediction(wm,state,c2,world,material,private,substeps)
    common=substeps
    candidates=torch.cat((c1,c2))
    scores=torch.cat((value(first,common),value(second,common)))
    allowed=torch.cat((legal1,legal2))&torch.isfinite(scores)
    if not allowed.any():raise FloatingPointError('P3 has no physically supported finite candidate')
    winner=int(scores.masked_fill(~allowed,torch.inf).argmin())
    if allowed[0]:
        eps=8*torch.finfo(dtype).eps*max(1.,abs(float(scores[0])))
        if float(scores[0]-scores[winner])<=eps:winner=0
    coverage={}
    for name,v in events1.items():
        if v is not None:
            values=torch.cat((v,events2[name]),dim=1)
            coverage[name]=dict(nominal=values[:,0].cpu().tolist(),
                selected=values[:,winner].cpu().tolist(),nonzero_candidates_per_member=(values>0).sum(1).cpu().tolist())
    return candidates[winner],dict(coverage=coverage,successor_rule='nominal at each subsequent predicted legal request',
        count=128,search_iterations=2,common_substeps=common,
        common_s=float(common*DT),winner=winner,selected_score=float(scores[winner]),
        nominal_score=float(scores[0]) if allowed[0] else None,removed_released_cells=deleted,
        rejected_domain_or_nonfinite=int((~allowed).sum()),predicted_costs=[v if ok else None for v,ok in zip(scores.cpu().tolist(),allowed.cpu().tolist())],requests=candidates.cpu().tolist())


@torch.no_grad()
def select_batch(wm,states,worlds,*,private=False,material='M1',substeps=None):
    substeps=int(substeps if substeps is not None else getattr(wm,'planning_substeps',32))
    choices=[];logs=[]
    for state,world in zip(states,worlds,strict=True):
        choice,record=select_one(wm,state,world,private,material,substeps)
        choices.append(choice);logs.append(record)
    return torch.stack(choices),logs


class EventPlanner:
    def __init__(self,settings):
        from s1_w59_model import ExecutionEnsemble
        self.device=torch.device('cuda')
        self.wm=ExecutionEnsemble.load(settings['model_paths'],self.device)
        self.wm.planning_substeps=settings.get('planning_substeps',32)
        self.wm.normal_envelope_N=settings['normal_envelope_N']
        self.material=settings['material']
        self.private=settings['arm']=='P3-private'

    def choose(self,controller,t,mission,mq,material_info,grf):
        if self.material=='M3':
            self.wm.bind_material_publication(material_info['material_observer'].planning_publication())
        state=controller_state(controller,t,mission,mq,grf)
        world=controller.causal_cells.table
        x=torch.as_tensor(state,device=self.device)
        decision,records=select_batch(self.wm,x[None],[world],private=self.private,material=self.material)
        if self.private:
            actor_state,_,_=prepare_private(state,world)
        else:actor_state=state
        predicted=decision[0].cpu().numpy()
        action=np.clip(predicted.astype(np.float64),LOWER,UPPER)
        # Use the registered FP64 physical values on exact nominal/end-point choices.
        for reference in (LOWER,NOMINAL,UPPER):
            same=(predicted==reference.astype(predicted.dtype))
            action[same]=reference[same]
        return actor_state,action,records[0]
