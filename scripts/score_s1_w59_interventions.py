"""Grade W59 real legal-request effects, ensemble ordering and local gradients in the same window."""
import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from s1_w59_contact import contact_velocity_xy
from s1_w59_model_fields import DT, INTERVAL, MASS, PHASES, pack_state
from s1_w59_planning import components, value
from run_s1_w59_interventions import ROOT, REL, SMALL, LARGE, job_name

SIGNED=('due_minus','due_plus','ramp_minus','ramp_plus','rho_minus','rho_plus')


def lines(p):
    with p.open() as f:return [json.loads(v) for v in f if v.strip()]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def starting_state(raw,reg,snapshot):
    request=next(v for v in lines(raw/'w55-f/handover-requests.jsonl') if v['stage']=='request')
    if request['decision_id']!=reg['request_id'] or abs(request['sample_time_s']-reg['request_s'])>1e-9:
        raise ValueError('Physical branch started at a different legal request')
    with np.load(sorted((raw/'w55-f').glob('control-*.npz'))[0]) as z:
        grf=z['w59_measured_grf'][0]
        confirmed=z['confirmed_contact'][0];bearing=z['load_bearing_contact'][0];transfer=z['transfer_fraction'][0]
    with np.load(raw/'trajectory.npz') as z:
        q=z['qpos'][0];v=z['qvel'][0];feet=z['feet'][0];vel=z['feet_velocity'][0]
        phase=int(z['phase'][0]);command=z['command_velocity'][0];goal=z['body_target'][0]
    base=Path(reg['snapshot']).parents[1]
    with np.load(base/'trajectory.npz') as s:
        i=int(snapshot['step']); held=s['q_target'][i-1].copy()
    tm=request['sample_time_s']
    incoming=np.zeros(4);incoming[request['incoming_pair']]=1.
    xy=np.zeros((4,2));arrival=np.zeros(4);mask=np.zeros(4)
    for row in request['plan_publication']:
        if row['valid']:
            j=row['leg'];xy[j]=row['xy'];arrival[j]=row['target_time_s']-tm;mask[j]=1.
    ctrl=snapshot['controller']
    cells=ctrl['causal_cells']
    cells.observe(tm,feet,grf)
    mem,mv=cells.query(tm,feet,xy,mask)
    prev=snapshot['joint']['legal_route']
    route=pickle.loads(pickle.dumps(prev))
    route.update(tm,SimpleNamespace(phase=PHASES[phase],laps=snapshot['mission'].get('laps',0)),q[:2],feet,grf)
    phase_vec=np.eye(len(PHASES))[phase]
    state=pack_state(qpos=q,qvel=v,feet=feet,foot_velocity=vel,measured_force=grf,held_command=held,
        force_reference=ctrl['w59_force_ref'],force_solved=ctrl['last_force'],confirmed=confirmed,bearing=bearing,
        transfer=transfer,incoming=incoming,request_age=0.,nominal_due_in=request['nominal_due_s']-tm,
        active=1.,target_xy=xy,target_in=arrival,target_valid=mask,phase=phase_vec,command=command,
        goal=goal,hold=0.,progress=route.progress,laps=route.completed_laps,time=tm,version=[0,0,0,1],
        memory=mem,memory_valid=mv,request_valid=1.,control_valid=1.,route_valid=1.)
    return state,cells.table


def measured(raw,result,snapshot):
    with np.load(raw/'trajectory.npz') as z:
        t=z['time'];q=z['qpos'];v=z['qvel'];tau=z['actuator_force'];feet=z['feet']
        rpy=z['rpy'];phase=z['phase']
    with np.load(raw/'physical-000.npz') as z:plant=z['physical']
    foot=np.concatenate([np.load(p)['foot'] for p in sorted(raw.glob('material-[0-9]*.npz'))])
    # Failed unapplied material candidates can produce an additional material row.
    n=len(plant);foot=foot[:n]
    if len(foot)!=n or not np.allclose(foot[:,0],plant[:,0],atol=1e-8,rtol=0.):
        raise ValueError('Measured material and force clocks disagree')
    vel=plant[:,73:85].reshape(-1,4,3)
    slip=contact_velocity_xy(foot,vel,snapshot['controller']['radius'])
    fz=plant[:,32:42:3]
    load=fz>1.
    slip_rate=np.linalg.norm(slip,axis=2)*load
    control=[]
    prev=snapshot['joint']['legal_route']
    route=pickle.loads(pickle.dumps(prev))
    force_chunks=[];solved_chunks=[];ref_chunks=[];request_chunks=[];allowed_chunks=[]
    for p in sorted((raw/'w55-f').glob('control-*.npz')):
        with np.load(p) as z:
            force_chunks.append(z['w59_measured_grf']);solved_chunks.append(z['force'])
            ref_chunks.append(z['w59_reference_offset']);request_chunks.append(z['w59_request_id'])
            allowed_chunks.append(z['w59_preference_active'])
    forces=np.concatenate(force_chunks);solved=np.concatenate(solved_chunks)
    offsets=np.concatenate(ref_chunks);ids=np.concatenate(request_chunks);allowed=np.concatenate(allowed_chunks)
    mission=SimpleNamespace(laps=snapshot['mission'].get('laps',0))
    for j,f in enumerate(forces):
        mission.phase=PHASES[int(phase[j])]
        route.update(t[j],mission,q[j,:2],feet[j],f)
        control.append(route.progress)
    n20=min(len(plant)//100,(len(control)-1)//10)
    cost=np.zeros(n20)
    cprog=np.array(control)
    for k in range(n20):
        sl=slice(10*k,10*k+10)
        delta=cprog[10*k+10]-cprog[10*k]
        work=np.maximum(tau[sl]*v[sl,6:],0.).sum()*.002
        slip_m=slip_rate[100*k:100*k+100].sum()*.0002
        cost[k]=-5.*delta+DT/60+(work/(MASS*9.81)+slip_m)/6.
    events=lines(raw/'w55-f/handover-requests.jsonl')
    current=next(e for e in events if e['stage']=='request')
    nxt=next((e['sample_time_s'] for e in events if e['stage']=='request' and e['decision_id']>current['decision_id']),t[-1])
    n_before=max(0,int(np.floor((nxt-t[0])/DT+1e-7)))
    motion=dict(time_s=float(t[-1]-t[0]),applied_request=current['macro_action'],
        original_request_s=current['sample_time_s'],next_request_s=nxt,
        max_torque_Nm=float(abs(plant[:,2:14]).max()),min_body_z_m=float(q[:,2].min()),
        max_abs_roll_pitch_rad=float(abs(rpy[:,:2]).max()),slip_m=slip_rate.sum(axis=0).tolist(),
        slip_integration_dt_s=.0002,peak_Fz_N=fz.max(axis=0).tolist(),
        raw_positive_work_J=float((np.maximum(tau[:-1]*v[:-1,6:],0.)*np.diff(t)[:,None]).sum()),
        stage=result['stage_status'],hard_failure=result['stage_status']=='physical_failure',
        physical_rows=len(plant),legal_progress_delta=float(control[-1]-control[0]))
    motion['slip_m']=(slip_rate.sum(axis=0)*.0002).tolist()
    span=(ids==current['decision_id']);both=span&allowed
    pair=np.asarray(current['incoming_pair'],int)
    executed=next((e for e in events if e['stage']=='executed' and e['decision_id']==current['decision_id']),None)
    motion['execution']=dict(incoming=pair.tolist(),eligible_WBC_ticks=int(both.sum()),
        total_initial_request_WBC_ticks=int(span.sum()),requested_rho=float(current['macro_action'][2]),
        peak_reference_offset_N=float(abs(offsets[span,:,:]).max()) if span.any() else None,
        max_reference_net_normal_N=float(abs(offsets[span,:,2].sum(axis=1)).max()) if span.any() else None,
        eligible_solved_mean_Fz_N=solved[both][:,pair,2].mean(axis=0).tolist() if both.any() else None,
        eligible_measured_mean_Fz_N=forces[both][:,pair,2].mean(axis=0).tolist() if both.any() else None,
        accepted_s=executed['actual_liftoff_s'] if executed else None,
        gate_blocks=executed['gate_blocks'] if executed else None)

    observed=lines(raw/'material-events.jsonl')
    initiated={e['event_id'] for e in observed if e['type'] in ('load','revisit') and e['time_s']>=t[0]-1e-8}
    motion['physical_cross_foot_reload']=[dict(offset_s=float(e['time_s']-t[0]),
        event_id=e['event_id'],source_event_id=e['previous_load_event_id'],world_key=e['world_key'],
        gap_s=e['unloaded_gap_s'],area_m2=e['area_m2'],source_leg=e['previous_leg'],reload_leg=e['leg'],
        source_in_branch=e['previous_load_event_id'] in initiated)
        for e in observed if e['type']=='revisit' and e['cross_leg'] and e['area_m2']>0.
        and e['previous_load_event_id']>=0 and e['time_s']>=t[0]-1e-8]

    return cost,min(n20,n_before),motion


def replay_against_source(raw,source):
    with np.load(raw/'trajectory.npz') as branch,np.load(source/'trajectory.npz') as original:
        t=branch['time'];origin=original['time'];start=int(np.rint(t[0]/.002))
        n=min(len(t),len(origin)-start);window=slice(start,start+n)
        if n!=len(t) or not np.allclose(t,origin[window],atol=1e-8,rtol=0.):
            raise ValueError('Comparison requires coincident unperturbed physical times')
        return {k:float(np.max(abs(branch[k]-original[k][window]))) for k in
            ('qpos','qvel','actual_grf','q_target','actuator_force')}


def physical_rows(reg,role):
    for state in reg['states'][role]:
        records={}
        snap_path=Path(state['snapshot'])
        if sha(snap_path)!=state['sha256']:raise ValueError('Snapshot checksum differs')
        snapshot=pickle.loads(gzip.decompress(snap_path.read_bytes()))
        for signed,request in reg['signed'].items():
            job=job_name(state,role,signed)
            p=SMALL/'episodes'/(job+'.json')
            if not p.exists():
                records[signed]={'unrun':True};continue
            result=json.loads(p.read_text()); raw=LARGE/job
            if result.get('stage_status')=='technical_interruption' or not result.get('branch',{}).get('applied'):
                records[signed]={'technical':True,'stage_status':result.get('stage_status'),'reason':result.get('reason')}
                continue
            costs,count,motion=measured(raw,result,snapshot)
            if signed.startswith('nominal'):
                motion['original_reference_max_abs_error']=replay_against_source(raw,snap_path.parents[1])
                init=state.get('source_request',[0.,.05,0.])
                motion['source_first_request_matched']=bool(np.allclose(init,[0.,.05,0.],atol=1e-9,rtol=0))
                motion['source_comparison_scope']='full replay' if motion['source_first_request_matched'] else 'matched prefix, changed first request'
            records[signed]=dict(cost=costs,valid_steps=count,measured=motion)
        yield state,records,snapshot


def fit_state(state,records,snapshot,wm,reg,material,role):
    signed=list(reg['signed'])
    source=job_name(state,role,'nominal1')
    x,world=starting_state(LARGE/source,reg=state,snapshot=snapshot)
    x=torch.tensor(x,device='cuda')
    actions=torch.tensor([reg['signed'][s] for s in signed],device=x.device,dtype=x.dtype)
    with torch.no_grad():
        pred=wm.predict_candidates(x,actions,steps=32,world=world,material=material,private=False)
        n=int(pred['valid_steps'].to(torch.int32).cumprod(-1).sum(-1).min().item())
        n=min(n,min(r['valid_steps'] for r in records.values()))
        vals=value(components(pred['interval']),n).cpu().tolist()
    if n:
        u=actions[:1].detach().clone().requires_grad_(True)
        differentiable=wm.predict_candidates(x,u,steps=n,world=world,material=material,private=False)
        objective=value(components(differentiable['interval']),n).sum()
        derivative,=torch.autograd.grad(objective,u)
        grad=derivative[0].detach().cpu().numpy()
    else:grad=np.zeros(3)
    actual={s:float(records[s]['cost'][:n].sum()) for s in signed};predicted=dict(zip(signed,vals))
    mu=.5*(actual['nominal1']+actual['nominal2'])
    floor=abs(actual['nominal2']-actual['nominal1'])
    comparisons=[]
    for s in SIGNED:
        dr=actual[s]-mu; dp=predicted[s]-predicted['nominal1']
        dg=float(grad@(np.asarray(reg['signed'][s])-np.asarray(reg['signed']['nominal1'])))
        resolvable=n>0 and abs(dr)>max(floor,1e-12)
        comparisons.append(dict(signed=s,observed=dr,predicted=dp,linearized=dg,resolvable=resolvable,
            rank_correct=bool(resolvable and dp*dr>0),gradient_correct=bool(resolvable and dg*dr>0),
            hard_failure=records[s]['measured']['hard_failure'],
            excess_hard=records[s]['measured']['hard_failure'] and not (records['nominal1']['measured']['hard_failure'] or records['nominal2']['measured']['hard_failure'])))
    best=min(['nominal1',*SIGNED],key=predicted.get)
    return dict(common_s=n*DT,nominal_replay_floor=floor,utility_units='dimensionless cost; lower is better',
        directions=comparisons,nominal_gradient=grad.tolist(),best=best,
        best_real_gain=mu-actual[best],best_new_hard_fail=records[best]['measured']['hard_failure'] and
            not (records['nominal1']['measured']['hard_failure'] or records['nominal2']['measured']['hard_failure']))


def summarize(rows):
    complete=[r for r in rows if 'comparison' in r]
    if not complete:return dict(denominator_states=len(rows),evaluated_states=0,admitted=False)
    cmp=[r['comparison'] for r in complete]
    rates=np.array([sum(d['gradient_correct'] for d in x['directions'])/6 for x in cmp])
    ranks=np.array([sum(d['rank_correct'] for d in x['directions'])/6 for x in cmp])
    resolvable=sum(sum(d['resolvable'] for d in x['directions']) for x in cmp)
    random=np.random.default_rng(5901)
    draw=random.integers(0,len(cmp),(10000,len(cmp)))
    lower=float(np.quantile(rates[draw].mean(1),.025))
    gain=np.array([x['best_real_gain']-x['nominal_replay_floor'] for x in cmp])
    excess=sum(x['best_new_hard_fail'] for x in cmp)
    all_hard=sum(sum(d['hard_failure'] for d in x['directions']) for x in cmp)
    extra_hard=sum(sum(d['excess_hard'] for d in x['directions']) for x in cmp)
    return dict(denominator_states=len(rows),evaluated_states=len(cmp),denominator_directions=len(rows)*6,
        resolved=resolvable,unresolved=len(cmp)*6-resolvable,gradient_correct=int(round(rates.sum()*6)),
        gradient_all=float(rates.sum()/len(rows)),gradient_resolved=float(rates.sum()*6/resolvable) if resolvable else None,
        rank_all=float(ranks.sum()/len(rows)),gradient_state_bootstrap_lower95=lower,
        gain_above_replay_floor_mean=float(gain.mean()),winner_excess_hard=excess,directional_hard_failures=all_hard,directional_excess_hard_failures=extra_hard,
        admitted=len(cmp)==len(rows) and lower>.5 and gain.mean()>0 and excess==0 and extra_hard==0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--role',choices=('development','sealed_admission'),required=True)
    p.add_argument('--wm-checkpoint',action='append',default=[]);p.add_argument('--material',choices=('M1','M3'),default='M1')
    p.add_argument('--wm-recipe',type=Path)
    p.add_argument('--model-kind',choices=('structured','blackbox'),default='structured')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.role=='sealed_admission' and args.wm_checkpoint and not args.wm_recipe:
        p.error('A sealed model comparison requires its previously frozen physical forecast horizon')
    fixed=32 if not args.wm_recipe else json.loads(args.wm_recipe.read_text())['planning_substeps']
    from s1_w59_action_horizons import NODES,horizon_comparisons
    if fixed not in NODES:raise ValueError('The frozen planning horizon is not in the registered comparison nodes')
    reg=json.loads(args.registry.read_text());wm=None
    if args.wm_checkpoint:
        from s1_w59_model import ExecutionEnsemble
        wm=ExecutionEnsemble.load(args.wm_checkpoint,'cuda')
    pubs={}
    if wm is not None and (args.material=='M3' or any(s.get('source_material')=='M3' for s in reg['states'][args.role])):
        from s1_w59_action_publications import source_publications
        pubs=source_publications(reg['states'][args.role])
    from s1_w59_forward_domain import historical
    if args.wm_recipe:
        original=json.loads(args.wm_recipe.read_text());normal_max=original['observed_normal_max_N']
    else:
        # Fixed once from the complete enrolled B3/Fixed originals used for method development.
        normal_max=0.
        for arm in ('b3','fixed-action'):
            for case in ('a','b'):
                d=LARGE/f'w59-{arm}-{case}-original-e30-a02'
                with np.load(d/'physical-000.npz') as f:
                    normal_max=max(normal_max,float(f['physical'][:,32:42:3].max()))
    allowed=historical(SMALL/'model',args.model_kind,NODES,normal_max)
    out=[]
    for state,records,snap in physical_rows(reg,args.role):
        row={'state_id':state['snapshot_id'],'stratum':state['stratum'],'records':{}}
        for s,r in records.items():
            row['records'][s]={k:v for k,v in r.items() if k!='cost'}
            if 'cost' in r:row['records'][s]['interval_cost']=r['cost'][:32].tolist()
        if all('cost' in r for r in records.values()):
            common=min(r['valid_steps'] for r in records.values())
            full=min(len(r['cost']) for r in records.values())
            actual={k:float(r['cost'][:common].sum()) for k,r in records.items()}
            full_cost={k:float(r['cost'][:full].sum())+10.*r['measured']['hard_failure'] for k,r in records.items()}
            row['physical_effects']=dict(common_first_request_s=common*DT,common_full_s=full*DT,
                first_request_cost=actual,full_cost=full_cost,
                first_request_nominal_repeat_difference=abs(actual['nominal1']-actual['nominal2']),
                full_nominal_repeat_difference=abs(full_cost['nominal1']-full_cost['nominal2']),
                full_delta_from_nominal={k:full_cost[k]-.5*(full_cost['nominal1']+full_cost['nominal2']) for k in SIGNED})
        if wm and all('cost' in r for r in records.values()):
            pub=pubs.get(state['snapshot_id'])
            if args.material=='M3':wm.bind_material_publication(pub)
            # The actual pending interval is still reported in physical_effects.
            source=job_name(state,args.role,'nominal1')
            # Re-read the saved table because starting_state advances the completed causal observation once.
            clean=pickle.loads(gzip.decompress(Path(state['snapshot']).read_bytes()))
            x,table=starting_state(LARGE/source,state,clean)
            row['horizons']=horizon_comparisons(x,table,records,wm,reg,args.material,state,pub,allowed=allowed,normal_max=normal_max)
            if row['horizons'][str(fixed)].get('usable'):row['comparison']=row['horizons'][str(fixed)]
        out.append(row)
    summary=summarize(out)
    measured_records=[r for state in out for r in state['records'].values() if 'measured' in r]
    replays=[r['measured']['original_reference_max_abs_error'] for r in measured_records
             if 'original_reference_max_abs_error' in r['measured'] and r['measured']['source_first_request_matched']]
    summary.update(physical_states_complete=sum('physical_effects' in r for r in out),
        registered_branches=len(out)*8,measured_branches=len(measured_records),
        observed_mj_steps=sum(r['measured']['physical_rows'] for r in measured_records),
        actual_hard_failures=sum(r['measured']['hard_failure'] for r in measured_records),
        replay_vs_unperturbed_max_abs={k:max(r[k] for r in replays) for k in replays[0]} if replays else None)
    summary['by_stratum']={k:summarize([r for r in out if r['stratum']==k]) for k in sorted({r['stratum'] for r in out})}
    by_horizon={}
    for n in NODES:
        selected=[]
        for v in out:
            item={k:w for k,w in v.items() if k!='comparison'}
            measure=v.get('horizons',{}).get(str(n),{})
            if measure.get('usable'):item['comparison']=measure
            selected.append(item)
        graded=summarize(selected)
        finite=[dict(v,comparison=dict(v['comparison'],directions=[
            dict(d,gradient_correct=d['rank_correct']) for d in v['comparison']['directions']]))
            if 'comparison' in v else v for v in selected]
        rank=summarize(finite)
        observed=[v['comparison']['actual_coverage'] for v in selected if 'comparison' in v]
        graded.update(historical_forward_domain=allowed[str(n)],ordering_eligible=rank['admitted'] and allowed[str(n)]['pass_gate'],rank_bootstrap_lower95=rank.get('gradient_state_bootstrap_lower95'),
            states_with_cross_foot_reload=sum(any(v['cells']>0 for v in a.values()) for a in observed),
            states_with_candidate_source_then_cross_foot_reload=sum(any(v['source_in_branch']>0 for v in a.values()) for a in observed))
        by_horizon[str(n)]=graded
    result=dict(registry_sha256=sha(args.registry),role=args.role,
        model_kind=args.model_kind,observed_normal_max_N=normal_max,frozen_planning_substeps=fixed,fixed_horizon_comparisons=by_horizon,
        terminal_continuation='after the first absolute perturbation, use the source B3/Fixed/P3 policy at its next model-predicted legal events',checkpoints=[dict(path=str(p),sha256=sha(Path(p))) for p in args.wm_checkpoint],
        bootstrap_unit='initial legal request, 10000 resamples; states within a source case/episode share history',
        source_episodes=sorted(set(v['snapshot_id'].split('/')[0] for v in reg['states'][args.role])),
        admission_scope='sealed_only' if args.role=='sealed_admission' else 'development_only',
        summary=summary,states=out)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(args.output)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
