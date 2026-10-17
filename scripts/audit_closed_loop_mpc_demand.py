"""Fixed-pool execution-aware MPC audit and independent task costates.

All checkpoints are frozen. Existing reference CEM pools are reconstructed on
saved states, then rescored with the actual native actor/plant. No training,
parameter update, future disturbance preview or new reward is introduced.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from response_ppo.env import EnvironmentConfig, ResponseServo
from response_codesign.splates_bridge import build_splates_planner
from response_ppo.closed_loop_prediction import (NativeContext, native_step, reference_step,
    predict_macro_commands, planner_coordinates_to_execution, prefix_violation_summary)

OUT=ROOT/'notes/closed-loop-interface-2026-10-17'
WORK=ROOT/'work/closed-loop-interface-2026-10-17'
REPLAY=ROOT/'work/neural-demand-audit-2026-10-17/replays'
SEEDS=(17,29,43)
ARMS=('joint_support','fixed_support','joint_penalty','fixed_penalty')
QUERIES=(0,60,120,180)
KEYS=('mean.0.weight','mean.0.bias','mean.2.weight','mean.2.bias','mean.4.weight','mean.4.bias')
torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def text_sha(path): return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def plain(v):
    if isinstance(v,torch.Tensor): return v.detach().cpu().numpy().tolist()
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    raise TypeError(type(v).__name__)


def mean(p,o):
    h=torch.tanh(F.linear(o,p[:1088].reshape(64,17),p[1088:1152]))
    h=torch.tanh(F.linear(h,p[1152:5248].reshape(64,64),p[5248:5312]))
    return F.linear(h,p[5312:5376].reshape(1,64),p[5376:5377]).squeeze(-1)


def known_goals(seed,cfg):
    generator=torch.Generator(device='cpu').manual_seed(seed*10000+5000)
    torch.randn(16,3,generator=generator)  # Same original initialization draw.
    n=(256+cfg.horizon*cfg.command_steps)//cfg.target_steps+3
    magnitude=.15+.15*torch.rand(16,n,generator=generator)
    sign=torch.where(torch.rand(16,n,generator=generator)<.5,-1.,1.)
    # Do not read or generate the following hidden disturbance-phase draw here.
    return magnitude*sign


def preview(goals,t,cfg):
    return goals[:,(torch.arange(t,t+cfg.horizon*cfg.command_steps)//cfg.target_steps)]


def choose(cost,valid):
    finite=torch.isfinite(cost)&valid
    any_valid=finite.any(-1)
    idx=torch.where(finite,cost,torch.inf).argmin(-1)
    return idx,any_valid


def branch_signature(pred):
    s=pred['steps']
    fields=[s['interval_feasible'].long(),s['reference_clip_active'].long(),
            torch.sign(s['reference_raw_input']).long()*s['reference_clip_active'].long(),
            torch.where(s['fallback_used'],s['fallback_candidate_index'], -1)]
    for key in ('support_lower_candidate_index','support_upper_candidate_index'):
        if key in s: fields.append(s[key].long())
    return torch.stack(fields,-1).detach()


def exact_native_shooting(p,x,z,commands,design,goals,ctx,support):
    """Independent forward using the historical env methods, without env.step/MPC."""
    env=object.__new__(ResponseServo)
    env.config=ctx.cfg; env.P=ctx.P; env.B=ctx.B; env.E=ctx.E
    env.batch=1; env.raw_design=design[None]; env.support=support
    env.x=x[None].clone(); env.z=z[None].clone()
    cost=torch.zeros((),dtype=p.dtype)
    with torch.no_grad():
        for j in range(len(goals)):
            env.command=commands[j:j+1]
            obs=env.observation()
            znext,_,interval=env.interval()
            u_norm=torch.tanh(mean(p,obs))
            midpoint=(interval.lo+interval.hi)/2
            half=(interval.hi-interval.lo)/2
            u=torch.where(interval.feasible,midpoint+half*u_norm,env.fallback(znext)) if support else ctx.cfg.input_bound*u_norm
            env.x=env.f0(env.x)+env.B*u[:,None]  # Declared nominal w=0.
            env.z=znext
            cost+=(env.x[0,0]-goals[j]).square()+.005*u[0].square()+.01*env.x[0,1].square()
    return cost/len(goals)


def costate_gradient(p,pred,commands,design,goals,tick,ctx,support):
    """Independent local VJPs + analytic stage derivatives, not full graph backprop."""
    L=len(goals)
    lx=torch.zeros(3,dtype=p.dtype); lz=torch.zeros_like(lx)
    gradient=torch.zeros_like(p)
    contributions=[]
    B=ctx.B.to(p)
    for j in range(L-1,-1,-1):
        y=torch.cat((pred['x'][j,0,0],pred['z'][j,0,0])).detach().requires_grad_()
        pars=p.detach().requires_grad_()
        result=native_step(lambda o:mean(pars,o),y[:3],y[3:],commands[j],design,
                           goals[j],tick+j,ctx,support=support)
        u=result['action']
        up,uy=torch.autograd.grad(u,(pars,y))
        xp=pred['x'][j+1,0,0].detach()
        q=torch.stack((2*(xp[0]-goals[j]),.02*xp[1],torch.zeros((),dtype=p.dtype)))
        beta=lx+q
        xi=.01*u.detach()+B@beta
        contribution=up*xi
        gradient+=contribution
        contributions.append(float(torch.linalg.vector_norm(contribution)))
        h=ctx.cfg.dt
        oldx=y[:3].detach()
        A=torch.zeros((3,3),dtype=p.dtype)
        A[0,0]=1; A[0,1]=h
        A[1,0]=-h*torch.cos(oldx[0]); A[1,1]=1-.2*h-.2*h*oldx[1].abs(); A[1,2]=h
        A[2,2]=1-h/ctx.cfg.tau
        rz=torch.autograd.functional.jacobian(
            lambda state:reference_step(state,commands[j],design,ctx)[0], y[3:].detach())
        lx=A.T@beta+uy[:3]*xi
        lz=rz.T@lz+uy[3:]*xi
    return gradient/L,list(reversed(contributions))


def gradient_audit(p,x,z,candidate,design,goals,tick,ctx,support,pool_cost,distinct_gap):
    execution=planner_coordinates_to_execution(candidate)
    command_low=execution.repeat_interleave(ctx.cfg.command_steps,dim=0)
    def predict(parameters):
        return predict_macro_commands(lambda o:mean(parameters,o),x[None],z[None],
            execution[None,None],design[None],goals[None],tick,ctx,support=support)
    pars=p.detach().requires_grad_()
    started=time.perf_counter(); pred=predict(pars)
    cost=pred['task_cost_mean'].squeeze()
    g=torch.autograd.grad(cost,pars)[0].detach()
    direct_seconds=time.perf_counter()-started
    base_signature=branch_signature(pred)
    started=time.perf_counter()
    adj,contributions=costate_gradient(p,pred,command_low,design,goals,tick,ctx,support)
    adjoint_seconds=time.perf_counter()-started
    shooting=exact_native_shooting(p,x,z,command_low,design,goals,ctx,support)
    norm=float(torch.linalg.vector_norm(g))
    direction=-g/norm if norm else torch.zeros_like(g)
    checks=[]
    for h in (1e-4,1e-5):
        with torch.no_grad():
            plus=predict(p+h*direction); minus=predict(p-h*direction)
        jp=float(plus['task_cost_mean'].squeeze()); jm=float(minus['task_cost_mean'].squeeze())
        fd=(jp-jm)/(2*h); target=float(g@direction)
        same=bool(torch.equal(base_signature,branch_signature(plus)) and torch.equal(base_signature,branch_signature(minus)))
        relative=abs(fd-target)/max(abs(target),1e-300)
        def margins(value): return {k:plain(v.squeeze()) for k,v in prefix_violation_summary(value).items()}
        checks.append({'h':h,'base_cost':float(cost.detach()),'negative_gradient_probe_cost':jp,
                       'opposite_probe_cost':jm,'finite_difference':fd,'analytic_directional_derivative':target,
                       'relative_error':relative,'same_piecewise_branch':same,
                       'smooth_FD_pass':bool(same and relative<=1e-3),
                       'negative_probe_margins':margins(plus),'opposite_probe_margins':margins(minus)})
    difference=float(torch.linalg.vector_norm(g-adj))
    s=pred['steps']
    result={'status':'computed','tick':tick,'environment_index':0,'prediction_mode':'nominal_zero_support_W_unchanged',
            'cost':float(cost.detach()),'pool_cost':float(pool_cost),'pool_cost_difference':float(cost.detach())-float(pool_cost),
            'historical_native_shooting_cost':float(shooting),'shooting_cost_difference':float(shooting-cost.detach()),
            'distinct_command_winner_gap':float(distinct_gap),'gradient_norm':norm,
            'adjoint_gradient_norm':float(torch.linalg.vector_norm(adj)),
            'direct_vs_costate_relative_error':difference/max(norm,1e-300),
            'direct_vs_costate_max_error':float((g-adj).abs().max()),
            'adjoint_pass':bool(difference/max(norm,1e-300)<=1e-8),
            'task_gradient_resolves':bool(norm>1e-12),
            'direct_gradient_seconds':direct_seconds,'adjoint_seconds':adjoint_seconds,
            'local_parameter_contribution_norms_unaveraged':contributions,
            'selected_margins':{k:plain(v.squeeze()) for k,v in prefix_violation_summary(pred).items()},
            'selected_fallback_low_steps':int(s['fallback_used'].sum()),
            'selected_nonregular_feasible_low_steps':int((s['interval_feasible']&~s['support_regular_feasible']).sum()),
            'fixed_direction_checks':checks,'checkpoint_updated':False,
            'same_information_comparison':'Direct model-based PO/BPTT and independent constrained-policy shooting costate evaluate exactly the same fixed-command nominal task; equality is expected, not superiority.',
            'mean_action_preservation_claim':False,
            'budget':{'forward_gradient_model_transitions':len(goals),'costate_local_model_steps':len(goals),
                      'reference_state_jacobian_steps':len(goals),'independent_shooting_steps':len(goals),
                      'finite_difference_model_transitions':4*len(goals),'parameter_probe_vectors':4,
                      'training_updates':0,'policy_updates':0,'checkpoint_writes':0}}
    return result,{'gradient':g.numpy(),'costate_gradient':adj.numpy(),'direction':direction.numpy(),
                  'commands_execution':execution.numpy(),'x':pred['x'].detach().numpy(),'z':pred['z'].detach().numpy()}


def audit_case(seed,arm):
    started=time.perf_counter()
    folder=ROOT/f'work/ppo-mechanism-2026-10-16/formal/seedrun-{seed}/seed-{seed}'
    checkpoint=folder/f'{arm}.pt'; checkpoint_hash=sha(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    run=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    cfg=EnvironmentConfig(**run['environment'])
    p=torch.cat([saved['actor'][k].reshape(-1) for k in KEYS]).double()
    replay=REPLAY/f'{seed}-{arm}.npz'
    with np.load(replay,allow_pickle=False) as d:
        x,z,design,oldcommand=[torch.tensor(d[k]) for k in ('x','z','raw_design','command')]
        P,B,E=[torch.tensor(d[k]) for k in ('P','B','E')]
        oldtarget=torch.tensor(d['target'])
    goals=known_goals(seed,cfg)
    assert torch.equal(goals[:,torch.arange(256)//cfg.target_steps].T,oldtarget)
    original=object.__new__(ResponseServo)
    original.config=cfg; original.P=P; original.B=B; original.E=E; original.goals=goals
    captured=[]
    def reference_callback(state,commands,tick):
        if int(tick) in QUERIES: captured.append(commands.detach().clone())
        return original.sequence_objective(state,commands,tick)
    planner=build_splates_planner(reference_callback,horizon=cfg.horizon,command_dim=2,
                                 command_bound=.45,n_samples=cfg.candidates,n_iterations=cfg.iterations,
                                 device='cpu')
    ctx=NativeContext(cfg,P.double(),B.double(),E.double())
    support=arm.endswith('support')
    rows=[]; model_steps=0; command_error=0.; gradient_result=None
    for tick in range(0,max(QUERIES)+1,cfg.command_steps):
        captured.clear()
        result=planner.plan(torch.cat((z[tick],design[tick]),-1),tick,seed=(seed*10000+5000)*10000+tick)
        execution_first=planner_coordinates_to_execution(result.action)
        err=float((execution_first-oldcommand[tick]).abs().max()); command_error=max(command_error,err)
        if err!=0: raise RuntimeError(f'Reference planner reconstruction differs: {seed}/{arm}/{tick}/{err}')
        if tick not in QUERIES: continue
        pool=torch.cat(captured,dim=1).double()
        task_preview=preview(goals,tick,cfg).double()
        with torch.no_grad():
            pred=predict_macro_commands(lambda obs:mean(p,obs),x[tick].double(),z[tick].double(),
                planner_coordinates_to_execution(pool),design[tick].double(),task_preview,tick,ctx,support=support)
        margins=prefix_violation_summary(pred)
        actual,reference=pred['task_cost_mean'],pred['reference_task_cost_mean']
        reference_valid=margins['reference_domain_margin_max']<=0
        actual_index,actual_ok=choose(actual,reference_valid)
        old_index,old_ok=choose(reference,reference_valid)
        full_valid=(reference_valid & ~(pred['steps']['actual_domain_violation'].bool().any(0))
                    & ~(pred['steps']['tube_violation'].bool().any(0)) & (margins['input_margin_max']<=0))
        full_new,full_new_ok=choose(actual,full_valid); full_old,full_old_ok=choose(reference,full_valid)
        selected=[]
        for env in range(16):
            ni,oi=int(actual_index[env]),int(old_index[env])
            valid=bool(actual_ok[env] and old_ok[env])
            identical=(pool[env]==pool[env,ni]).all(dim=(-1,-2))
            competitor=torch.where(reference_valid[env]&~identical,actual[env],torch.inf).min()
            gap=float(competitor-actual[env,ni]) if torch.isfinite(competitor) else None
            row={'environment':env,'both_reference_gate_feasible':valid,'reference_winner_index':oi,
                 'actual_winner_index':ni,'distinct_command_changed':bool(valid and not torch.equal(pool[env,ni],pool[env,oi])),
                 'reference_winner_reference_cost':float(reference[env,oi]),
                 'actual_winner_reference_cost':float(reference[env,ni]),
                 'reference_winner_actual_cost':float(actual[env,oi]),
                 'actual_winner_actual_cost':float(actual[env,ni]),'actual_winner_distinct_gap':gap,
                 'reference_prediction_absolute_error_at_reference_winner':float((reference[env,oi]-actual[env,oi]).abs()),
                 'reference_winner_margins':{k:plain(v[env,oi]) for k,v in margins.items()},
                 'actual_winner_margins':{k:plain(v[env,ni]) for k,v in margins.items()},
                 'common_full_physical_screen':{'exists':bool(full_new_ok[env] and full_old_ok[env]),
                    'new_index':int(full_new[env]),'reference_index':int(full_old[env]),
                    'new_actual_cost':float(actual[env,full_new[env]]),
                    'reference_actual_cost':float(actual[env,full_old[env]])},
                 'same_actual_cost_array_argmin_index':int(np.argmin(np.where(reference_valid[env].numpy(),actual[env].numpy(),np.inf)))}
            selected.append(row)
        case_dir=WORK/'pools'; case_dir.mkdir(parents=True,exist_ok=True)
        path=case_dir/f'{seed}-{arm}-tick{tick}.npz'
        np.savez_compressed(path,commands_planner=pool.numpy(),actual_cost=actual.numpy(),reference_cost=reference.numpy(),
                            **{key:value.numpy() for key,value in margins.items()})
        rows.append({'tick':tick,'candidate_count_per_environment':pool.shape[1],
                     'pool_relative_path':path.relative_to(ROOT).as_posix(),'pool_byte_sha256':sha(path),'selections':selected})
        model_steps+=pool.shape[0]*pool.shape[1]*cfg.horizon*cfg.command_steps
        if tick==60:
            if bool(actual_ok[0]):
                chosen=int(actual_index[0]); gap=selected[0]['actual_winner_distinct_gap']
                gradient_result,raw=gradient_audit(p,x[tick,0].double(),z[tick,0].double(),pool[0,chosen],
                    design[tick,0].double(),task_preview[0],tick,ctx,support,actual[0,chosen],float('inf') if gap is None else gap)
                np.savez_compressed(WORK/f'gradient-{seed}-{arm}.npz',**raw)
            else:
                gradient_result={'status':'not_run_no_reference_gate_feasible_candidate','tick':60,'environment_index':0}
    assert sha(checkpoint)==checkpoint_hash
    return {'seed':seed,'arm':arm,'checkpoint_byte_sha256':checkpoint_hash,'checkpoint_unchanged':True,
            'replay_byte_sha256':sha(replay),'reference_first_command_reconstruction_max_error':command_error,
            'queries':rows,'task_gradient':gradient_result,
            'budget':{'reference_pool_generation':planner.objective_counts(),
                      'reference_planner_instances':planner.n_plans*16,
                      'reference_prediction_microsteps':planner.objective_counts()['candidate_macro_steps']*cfg.command_steps,
                      'actual_common_pool_prediction_microsteps':model_steps,
                      'gradient_computation':gradient_result.get('budget',{}),
                      'new_environment_transitions_in_this_script':0,'training_updates':0,'policy_updates':0,'checkpoint_writes':0},
            'wall_seconds':time.perf_counter()-started}


def main():
    WORK.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    cases=[]
    for seed in SEEDS:
        for arm in ARMS:
            case=audit_case(seed,arm); cases.append(case)
            g=case['task_gradient']; selections=[s for q in case['queries'] for s in q['selections']]
            print(json.dumps({'seed':seed,'arm':arm,'queries':len(selections),
                              'changed_distinct_commands':sum(s['distinct_command_changed'] for s in selections),
                              'gradient_status':g['status'],'gradient_norm':g.get('gradient_norm'),
                              'adjoint_error':g.get('direct_vs_costate_relative_error'),
                              'FD':[c['relative_error'] for c in g.get('fixed_direction_checks',[])],
                              'wall_seconds':case['wall_seconds']},ensure_ascii=False),flush=True)
            report={'date':'2026-10-17','status':'completed' if len(cases)==12 else 'running',
                    'protocol_content_sha256':text_sha(OUT/'protocol.md'),
                    'script_content_sha256':text_sha(Path(__file__)),
                    'prediction_source_content_sha256':text_sha(ROOT/'src/response_ppo/closed_loop_prediction.py'),
                    'cases':cases,'scope':'Same original nominal task and frozen actors; fixed-pool audit and equal-objective derivative verification, no training or unique learning advantage claim'}
            (OUT/'pool-and-gradient-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=plain)+'\n',encoding='utf-8')


if __name__=='__main__': main()
