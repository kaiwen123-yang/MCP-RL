"""Fixed physical-time P3 forecasts with per-candidate nominal future requests and real reloading evidence."""
import numpy as np
import torch
from s1_w59_model_fields import DT, INTERVAL
from s1_w59_planning import components, value, select_batch
from functools import lru_cache

NODES=(1,5,17,18,32,65,100)
SIGNED=('due_minus','due_plus','ramp_minus','ramp_plus','rho_minus','rho_plus')


@lru_cache(maxsize=2)
def frozen_policy(paths,device):
    from s1_w59_model import ExecutionEnsemble
    return ExecutionEnsemble.load(paths,device)


def successor_settings(source,wm,publication=None):
    arm=source.get('source_arm','B3')
    if arm=='B3':return dict(successor='nominal')
    if arm=='Fixed-action':return dict(successor='fixed',successor_request=[0.,.060,.05])
    if arm not in ('P3-full','P3-private'):raise ValueError('Unregistered source event policy')
    settings=source['source_settings']
    controller=frozen_policy(tuple(settings['model_paths']),str(wm.device))
    if settings['material']=='M3':controller.bind_material_publication(publication)
    def replanner(states,worlds):
        return select_batch(controller,states,worlds,private=(arm=='P3-private'),material=settings['material'],
                            substeps=settings['planning_substeps'])[0]
    return dict(successor='policy',successor_policy=replanner)


def horizon_comparisons(state,world,records,wm,reg,material,source,publication=None):
    labels=list(reg['signed'])
    x=torch.tensor(state,device=wm.device)
    acts=torch.tensor([reg['signed'][s] for s in labels],device=x.device,dtype=x.dtype)
    successor=successor_settings(source,wm,publication)
    with torch.no_grad():
        result=wm.predict_candidates(x,acts,steps=max(NODES),world=world,material=material,
                                     private=False,**successor)
        masks=result['valid_steps'].to(x.dtype)
        flow,hazard=components(result['interval'])
        parts=(flow*masks,hazard*masks)
    nominal=acts[:1].detach().clone().requires_grad_(True)
    path=wm.predict_candidates(x,nominal,steps=max(NODES),world=world,material=material,
                               private=False,**successor)
    nm=path['valid_steps'].to(x.dtype)
    flow,fail=components(path['interval'])
    derivatives=(flow*nm,fail*nm)
    out={}
    for n in NODES:
        p=dict(zip(labels,value(parts,n).cpu().tolist()))
        g=torch.autograd.grad(value(derivatives,n).sum(),nominal,retain_graph=(n<NODES[-1]))[0][0]
        derivative_ok=bool(torch.isfinite(g).all());g=g.detach().cpu().numpy()
        # A natural hard terminal absorbs. A technical short branch is excluded earlier as unobserved.
        true={k:float(v['cost'][:n].sum())+10.*(v['measured']['hard_failure'] and v['measured']['time_s']<=n*DT+1e-8)
              for k,v in records.items()}
        floor=abs(true['nominal1']-true['nominal2'])
        mean=.5*(true['nominal1']+true['nominal2'])
        rows=[]
        for s in SIGNED:
            real=true[s]-mean;delta=p[s]-p['nominal1']
            signed_action=np.array(reg['signed'][s])-np.array(reg['signed']['nominal1'])
            jac=float(g@signed_action) if derivative_ok else None
            resolved=abs(real)>max(floor,1e-12)
            hard=records[s]['measured']['hard_failure']
            rows.append(dict(signed=s,observed=real,predicted=delta,linearized=jac,resolvable=resolved,
                rank_correct=bool(resolved and delta*real>0.),
                gradient_correct=bool(resolved and derivative_ok and jac*real>0.),
                hard_failure=hard,excess_hard=hard and not (
                    records['nominal1']['measured']['hard_failure'] or records['nominal2']['measured']['hard_failure'])))
        best=min(['nominal1',*SIGNED],key=p.get)
        actual_coverage={}
        for s,v in records.items():
            events=[e for e in v['measured']['physical_cross_foot_reload'] if e['offset_s']<=n*DT+1e-8]
            actual_coverage[s]=dict(cells=len(events),unique_world_keys=len({tuple(e['world_key']) for e in events}),
                source_in_branch=sum(e['source_in_branch'] for e in events),area_sum_m2=sum(e['area_m2'] for e in events))
        forecast_coverage={}
        for k in ('cross_reload_cells','cross_reload_from_branch','request_event_mask'):
            if k in result:
                vals=(result[k][...,:n]*masks[...,:n]).sum(-1)
                forecast_coverage[k]={s:vals[:,j].cpu().tolist() for j,s in enumerate(labels)}
        out[str(n)]=dict(common_s=n*DT,nominal_replay_floor=floor,nominal_gradient=g.tolist() if derivative_ok else None,
            gradient_finite=derivative_ok, directions=rows,best=best,best_real_gain=mean-true[best],
            best_new_hard_fail=records[best]['measured']['hard_failure'] and
                not (records['nominal1']['measured']['hard_failure'] or records['nominal2']['measured']['hard_failure']),
            actual_coverage=actual_coverage,predicted_coverage=forecast_coverage,
            successor=source.get('source_command_after_first',[0.,.05,0.]))
    return out
