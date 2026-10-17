"""Bounded continuous-command feasibility audit of all42 unresolved queries.

Only command coordinates are optimized. Actors/designs stay frozen. Finding a
feasible point is a witness; exhausting SLSQP starts is UNKNOWN, not a proof.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
import scipy
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from response_ppo.env import EnvironmentConfig,ResponseServo
from response_ppo.realization import map_normalized_action
from response_ppo.closed_loop_prediction import NativeContext,predict_macro_commands,planner_coordinates_to_execution,prefix_violation_summary

OUT=ROOT/'notes/stage-feasibility-2026-10-17'
WORK=ROOT/'work/stage-feasibility-2026-10-17/continuous'
AUTHORITY=OUT/'one-step-authority.json'
SELECTOR=OUT/'selector-audit.json'
PROTOCOL=OUT/'continuous-protocol.md'
PROTOCOL_HASH='0e48c8e68061ba15eb8582f1dfbcc297f63c540b8ca2b275fdf567f9ad123a3a'
KEYS_HASH='f861c91df6b4e06ba03022ae4fc3e47650360bd8b6cda7933ce2f631b5609c28'
PARAMETER_KEYS=('mean.0.weight','mean.0.bias','mean.2.weight','mean.2.bias','mean.4.weight','mean.4.bias')
MARGIN_KEYS=('reference_domain_margin_max','actual_domain_margin_max','input_margin_max','tube_margin_max')
LIMITS=np.array([0.,1e-7,0.,1e-8])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def mean(p,obs):
    h=torch.tanh(F.linear(obs,p[:1088].reshape(64,17),p[1088:1152]))
    h=torch.tanh(F.linear(h,p[1152:5248].reshape(64,64),p[5248:5312]))
    return F.linear(h,p[5312:5376].reshape(1,64),p[5376:5377]).squeeze(-1)


def known_goals(seed,cfg):
    generator=torch.Generator(device='cpu').manual_seed(seed*10000+5000)
    torch.randn(16,3,generator=generator,dtype=torch.float32)
    n=(256+cfg.horizon*cfg.command_steps)//cfg.target_steps+3
    magnitude=.15+.15*torch.rand(16,n,generator=generator,dtype=torch.float32)
    sign=torch.where(torch.rand(16,n,generator=generator,dtype=torch.float32)<.5,-1.,1.)
    return (magnitude*sign).double()


class FeasibleFound(Exception):
    def __init__(self,witness):self.witness=witness


class NonfiniteDerivative(Exception):pass


class Evaluator:
    def __init__(self,key):
        self.key=key
        s,self.arm,t,e=key.split('/');self.seed,self.tick,self.env=int(s),int(t),int(e)
        base=ROOT/f'work/ppo-mechanism-2026-10-16/formal/seedrun-{self.seed}/seed-{self.seed}'
        self.checkpoint=base/f'{self.arm}.pt';self.checkpoint_hash=sha(self.checkpoint)
        cp=torch.load(self.checkpoint,map_location='cpu',weights_only=True)
        self.p=torch.cat([cp['actor'][k].reshape(-1) for k in PARAMETER_KEYS]).double()
        assert not self.p.requires_grad
        setting=json.loads((base/'result.json').read_text(encoding='utf-8'))
        self.cfg=EnvironmentConfig(**setting['environment'])
        self.replay=ROOT/f'work/neural-demand-audit-2026-10-17/replays/{self.seed}-{self.arm}.npz'
        with np.load(self.replay,allow_pickle=False) as z:
            self.x=torch.tensor(z['x'][self.tick,self.env:self.env+1],dtype=torch.float64)
            self.z=torch.tensor(z['z'][self.tick,self.env:self.env+1],dtype=torch.float64)
            self.design=torch.tensor(z['raw_design'][self.tick,self.env:self.env+1],dtype=torch.float64)
            self.P,self.B,self.E=[torch.tensor(z[k],dtype=torch.float64) for k in ('P','B','E')]
            old_targets=torch.tensor(z['target'],dtype=torch.float64)
        goals=known_goals(self.seed,self.cfg)
        expected=torch.stack([goals[:,i//self.cfg.target_steps] for i in range(256)])
        assert torch.equal(expected,old_targets)
        self.goals=goals[self.env:self.env+1,torch.arange(self.tick,self.tick+80)//self.cfg.target_steps]
        self.ctx=NativeContext(self.cfg,self.P,self.B,self.E)
        self.support=self.arm.endswith('support')
        self.cache_x=None;self.cache=None;self.jacobian=None
        self.forward_calls=0;self.jacobian_calls=0;self.native_checks=0
        self.best=None;self.validation_failures=[];self.start_number=None
        self.last_progress=time.perf_counter()

    def evaluate(self,x):
        x=np.asarray(x,dtype=np.float64)
        if self.cache_x is not None and np.array_equal(x,self.cache_x):return self.cache
        self.forward_calls+=1
        c=torch.tensor(x.copy(),dtype=torch.float64,requires_grad=True)
        pred=predict_macro_commands(lambda obs:mean(self.p,obs),self.x,self.z,
             planner_coordinates_to_execution(c.reshape(1,1,8,2)),self.design,self.goals,self.tick,self.ctx,support=self.support)
        margins=prefix_violation_summary(pred)
        m=torch.stack([margins[k].reshape(()) for k in MARGIN_KEYS])
        outputs=torch.stack((m[3]/self.cfg.tube_budget,-m[0],1e-7-m[1],-m[2]))
        numbers=m.detach().numpy();cost=float(pred['task_cost_mean'].detach().reshape(()))
        finite=bool(np.isfinite(numbers).all() and np.isfinite(cost))
        other_ok=bool(finite and np.all(numbers[:3]<=LIMITS[:3]))
        point={'command_planner':x.reshape(8,2).tolist(),'margins':dict(zip(MARGIN_KEYS,numbers.tolist())),
               'actual_cost':cost,'reference_cost':float(pred['reference_task_cost_mean'].detach().reshape(())),
               'other_gate_feasible':other_ok,'full_gate_feasible':bool(other_ok and numbers[3]<=1e-8),
               'command_bounds_feasible':bool(np.all(np.abs(x)<=.45)),
               'support_empty_any':bool(margins['support_empty_any'].reshape(())),
               'actual_one_step_excess_max':float(margins['actual_one_step_excess_max'].detach().reshape(()))}
        if other_ok and (self.best is None or numbers[3]<self.best['margins']['tube_margin_max']):
            self.best={**point,'start_number':self.start_number,'forward_call':self.forward_calls}
        self.cache_x=x.copy();self.jacobian=None
        self.cache={'c':c,'prediction':pred,'outputs':outputs,'point':point}
        if point['full_gate_feasible'] and point['command_bounds_feasible']:
            check=self.independent_check(c.detach().reshape(8,2),pred)
            if check['passed']:
                raise FeasibleFound({**point,'independent_check':check,'start_number':self.start_number,
                                     'forward_call':self.forward_calls})
            self.validation_failures.append({'start_number':self.start_number,'forward_call':self.forward_calls,'check':check})
        if time.perf_counter()-self.last_progress>20:
            self.last_progress=time.perf_counter()
            print(json.dumps({'phase':'continuous_progress','key':self.key,'start':self.start_number,
                              'forward_calls':self.forward_calls,'jacobian_calls':self.jacobian_calls,
                              'best_eligible_tube_margin':None if self.best is None else self.best['margins']['tube_margin_max']}),flush=True)
        return self.cache

    def derivatives(self,x):
        data=self.evaluate(x)
        if self.jacobian is None:
            self.jacobian_calls+=1
            gradient=torch.autograd.grad(data['outputs'],data['c'],grad_outputs=torch.eye(4,dtype=torch.float64),
                                         is_grads_batched=True,retain_graph=False)[0]
            self.jacobian=gradient.detach().numpy()
            if not np.isfinite(self.jacobian).all():raise NonfiniteDerivative('Nonfinite native command derivative; no masking or retuning')
        return self.jacobian

    def objective(self,x):return float(self.evaluate(x)['outputs'][0].detach())
    def objective_jac(self,x):return self.derivatives(x)[0]
    def constraints(self,x):return self.evaluate(x)['outputs'][1:].detach().numpy()
    def constraint_jac(self,x):return self.derivatives(x)[1:]

    @torch.no_grad()
    def independent_check(self,commands,prediction):
        self.native_checks+=1
        env=object.__new__(ResponseServo)
        env.config,env.P,env.B,env.E=self.cfg,self.P,self.B,self.E
        env.batch,env.support=1,self.support
        env.x,env.z,env.raw_design=self.x.clone(),self.z.clone(),self.design
        xs,zs,actions,tasks=[env.x.clone()],[env.z.clone()],[],[]
        margins=[];ref_cost=[]
        for low in range(80):
            command=commands[low//10]
            env.command=torch.stack((command[0],1.+command[1]*(.2/.45)))[None]
            obs=env.observation();normal=torch.tanh(mean(self.p,obs))
            znext,uref,interval=env.interval();fallback=env.fallback(znext)
            mapped=map_normalized_action(normal,interval)
            action=torch.where(interval.feasible,mapped,fallback) if self.support else self.cfg.input_bound*normal
            xnext=env.f0(env.x)+self.B*action[:,None]
            error=xnext-znext;value=torch.einsum('bi,ij,bj->b',error,self.P,error)
            goal=self.goals[:,low]
            task=(xnext[:,0]-goal).square()+.005*action.square()+.01*xnext[:,1].square()
            rc=(znext[:,0]-goal).square()+.005*uref.square()+.01*znext[:,1].square()
            margins.append(torch.stack(((znext.abs()-znext.new_tensor(self.cfg.reference_bounds)).amax(),
                                        (xnext.abs()-xnext.new_tensor(self.cfg.actual_bounds)).amax(),
                                        action.abs().amax()-self.cfg.input_bound,
                                        value.amax()-self.cfg.tube_budget)))
            env.x,env.z=xnext,znext;xs.append(xnext.clone());zs.append(znext.clone());actions.append(action.clone());tasks.append(task.clone());ref_cost.append(rc)
        xstack=torch.stack(xs)[:,0];zstack=torch.stack(zs)[:,0]
        astack=torch.stack(actions)[:,0];tstack=torch.stack(tasks)[:,0]
        mm=torch.stack(margins).amax(0).numpy()
        errors={'x':float((xstack-prediction['x'][:,0,0]).abs().max()),
                'z':float((zstack-prediction['z'][:,0,0]).abs().max()),
                'action':float((astack-prediction['steps']['action'][:,0,0]).abs().max()),
                'task':float((tstack-prediction['steps']['task'][:,0,0]).abs().max()),
                'actual_cost':abs(float(tstack.mean())-float(prediction['task_cost_mean'].reshape(())))}
        gate=bool(np.all(np.isfinite(mm)) and np.all(mm<=LIMITS) and bool((commands.abs()<=.45).all()))
        passed=gate and max(errors.values())<=1e-10
        path=WORK/(self.key.replace('/','-')+f'-native-check-{self.native_checks}.npz')
        np.savez_compressed(path,commands_planner=commands.numpy(),x=xstack.numpy(),z=zstack.numpy(),
                            action=astack.numpy(),task=tstack.numpy(),goals=self.goals.numpy(),
                            margins_by_step=torch.stack(margins).numpy())
        return {'passed':passed,'gate_passed_independently':gate,'margins':dict(zip(MARGIN_KEYS,mm.tolist())),
                'actual_cost':float(tstack.mean()),'reference_cost':float(torch.stack(ref_cost).mean()),
                'max_absolute_errors':errors,'trace_path':path.relative_to(ROOT).as_posix(),'trace_sha256':sha(path),
                'forward_identity':'Original ResponseServo methods, no constructor/step/planner, full80 prefixes',
                'model_transitions':80}


def run_query(key,selector_row):
    torch.set_num_threads(1)
    started=time.perf_counter();cpu_start=time.process_time()
    evaluator=Evaluator(key)
    pool_path=ROOT/selector_row['pool_path']
    with np.load(pool_path,allow_pickle=False) as p:
        commands=p['commands_planner'][evaluator.env].copy()
    first=selector_row['best_tube_diagnostic']['index'];second=selector_row['primary_actual']['index']
    starts=[('old_min_tube_with_other_gates',commands[first].reshape(-1)),
            ('old_actual_primary_winner',commands[second].reshape(-1)),('zero_commands',np.zeros(16))]
    attempts=[];witness=None
    print(json.dumps({'phase':'continuous_start','key':key}),flush=True)
    for number,(label,x0) in enumerate(starts,1):
        evaluator.start_number=number;evaluator.cache_x=None;evaluator.cache=None;evaluator.jacobian=None
        f_before,j_before,n_before=evaluator.forward_calls,evaluator.jacobian_calls,evaluator.native_checks
        attempt={'number':number,'label':label,'initial_commands_planner':x0.reshape(8,2).tolist()}
        try:
            result=minimize(evaluator.objective,x0,method='SLSQP',jac=evaluator.objective_jac,
                            bounds=[(-.45,.45)]*16,
                            constraints={'type':'ineq','fun':evaluator.constraints,'jac':evaluator.constraint_jac},
                            options={'maxiter':100,'ftol':1e-10,'disp':False})
            final=evaluator.evaluate(result.x)['point']
            attempt.update(status='solver_returned_without_verified_feasible_witness',solver_success=bool(result.success),
                           solver_status=int(result.status),message=str(result.message),nit=int(result.nit),
                           nfev=int(result.nfev),njev=int(result.njev),final_point=final)
        except FeasibleFound as found:
            witness=found.witness
            attempt.update(status='verified_feasible_point_early_stop',witness=witness)
        except Exception as error:
            attempt.update(status='start_error_unknown',error=f'{type(error).__name__}: {error}')
        attempt.update(forward_calls=evaluator.forward_calls-f_before,
                       jacobian_calls=evaluator.jacobian_calls-j_before,native_checks=evaluator.native_checks-n_before)
        attempts.append(attempt)
        print(json.dumps({'phase':'continuous_attempt_end','key':key,'start':number,'status':attempt['status'],
                          'forward_calls':attempt['forward_calls'],'jacobian_calls':attempt['jacobian_calls']}),flush=True)
        if witness is not None:break
    unchanged=sha(evaluator.checkpoint)==evaluator.checkpoint_hash
    assert unchanged
    result={'key':key,'seed':evaluator.seed,'arm':evaluator.arm,'tick':evaluator.tick,'environment':evaluator.env,
            'status':'FEASIBLE_WITNESS' if witness is not None else 'UNKNOWN_WITHIN_FROZEN_BUDGET',
            'witness':witness,'best_other_gate_feasible_point':evaluator.best,'attempts':attempts,
            'independent_validation_failures':evaluator.validation_failures,
            'checkpoint_sha256':evaluator.checkpoint_hash,'checkpoint_unchanged':unchanged,
            'replay_sha256':sha(evaluator.replay),'source_pool_sha256':sha(pool_path),
            'wall_seconds':time.perf_counter()-started,'process_cpu_seconds':time.process_time()-cpu_start,
            'budget':{'command_optimizer_starts':len(attempts),'forward_prediction_calls':evaluator.forward_calls,
                      'forward_model_transitions':80*evaluator.forward_calls,
                      'command_jacobian_calls':evaluator.jacobian_calls,'reverse_output_rhs_per_call':4,
                      'reverse_step_rhs_equivalents':80*4*evaluator.jacobian_calls,
                      'independent_native_checks':evaluator.native_checks,'independent_model_transitions':80*evaluator.native_checks,
                      'environment_step_calls':0,'training_updates':0,'actor_parameter_updates':0,'checkpoint_writes':0},
            'interpretation':'A success is a verified continuous-command existence witness; a failure is UNKNOWN, not global infeasibility. Actor and design are fixed.'}
    path=WORK/(key.replace('/','-')+'-result.json');path.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=json_value)+'\n',encoding='utf-8')
    return result


def write_report(results,keys,started,finished):
    order={k:i for i,k in enumerate(keys)};results=sorted(results,key=lambda r:order[r['key']])
    successful=[r for r in results if r['status']=='FEASIBLE_WITNESS']
    unknown=[r for r in results if r['status']!='FEASIBLE_WITNESS']
    counts={key:sum(r.get('budget',{}).get(key,0) for r in results)
            for key in ('command_optimizer_starts','forward_prediction_calls','forward_model_transitions','command_jacobian_calls',
                        'reverse_step_rhs_equivalents','independent_native_checks','independent_model_transitions',
                        'environment_step_calls','training_updates','actor_parameter_updates','checkpoint_writes')}
    result={'date':'2026-10-17','status':'completed' if finished else 'running','protocol_sha256':sha(PROTOCOL),
            'script_sha256':sha(__file__),'authority_source_sha256':sha(AUTHORITY),'selector_source_sha256':sha(SELECTOR),
            'target_keys':keys,'target_count':len(keys),'completed_count':len(results),'feasible_count':len(successful),'unknown_count':len(unknown),
            'pending_keys':[k for k in keys if k not in {r['key'] for r in results}],
            'cases':results,'budget':counts,'workers':4,'torch_threads_per_worker':1,
            'main_elapsed_wall_seconds':time.perf_counter()-started,
            'sum_query_process_cpu_seconds':sum(r.get('process_cpu_seconds',0) for r in results),
            'sum_query_wall_seconds':sum(r.get('wall_seconds',0) for r in results),
            'method':'SLSQP pure tube feasibility objective normalized by existing2e-4 budget; original reference/actual/input constraints, bounds+-0.45,3 fixed starts,maxiter100,ftol1e-10',
            'information':'Nominal w0, original supportW=.04, original visible goal preview only; no hidden future phase',
            'scope':'All42 unresolved finite-pool queries; no checkpoint/actor/design update, no new reward, no claim of equal budget to old pool or globally sufficient search',
            'success_rule':'Any point satisfying all original gates, independently rechecked by original native80-step methods with<=1e-10 forward difference; stop remaining starts',
            'failure_rule':'UNKNOWN within the frozen local solver budget, never global no-solution',
            'earlier_debug_attempt':'continuous-debug.json: aborted serialization-only bug; extra CPU and model-count lower bounds are separate and must be included in whole-turn accounting',
            'runtime':{'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__}}
    (OUT/'continuous-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=json_value)+'\n',encoding='utf-8')
    table='\n'.join(f"| {r['key']} | {r['status']} | {len(r.get('attempts',[]))} | {r.get('budget',{}).get('forward_prediction_calls',0)} | {r.get('wall_seconds',0):.2f} |" for r in results)
    text=f"""# 固定反馈下的连续命令可行见证

日期：2026-10-17。状态`{result['status']}`。[事前协议](continuous-protocol.md)、[逐query证据](continuous-results.json)、[源码](../../scripts/audit_continuous_command_feasibility.py)。

全42个名义一步控制权界尚未解释的旧空pool查询固定不变，已完成{len(results)}个，独立核验可行{len(successful)}个，冻结预算内UNKNOWN {len(unknown)}个，待完成{len(keys)-len(results)}个。未完成不能写成失败或成功。

仅优化原8×2个宏命令坐标，actor/design不变。目标是降低原tube margin寻找可行点，另三项原gate保持；这不是新RL奖励。固定三starts为旧pre-tube可行候选中的min-tube、旧actual主赢家、zero；SLSQP maxiter100/ftol1e−10，不因结果改设置。成功由原始ResponseServo方法对全部80拍及原cost/四gate独立核验；找到即停止后续starts，不追求task cost最优。

| 固定query | 结论 | 已尝试starts | forward预测调用 | query墙钟秒 |
|---|---|---:|---:|---:|
{table}

成功只说明相应旧34候选pool漏掉了连续命令可行方案，否定“该pool失败已证明固定反馈无能”。UNKNOWN保留局部优化预算、分段非光滑和控制可达性的不确定，不能当全局不可达。旧140个失败中已由一步界解释的98个不在这42项优化内；原分母保持，不只汇报成功。

当前累计forward模型转移{counts['forward_model_transitions']}、4输出反向step/RHS等效计数{counts['reverse_step_rhs_equivalents']}、独立核验模型转移{counts['independent_model_transitions']}。4workers×torch1；总墙钟{result['main_elapsed_wall_seconds']:.2f}s，query进程CPU合计{result['sum_query_process_cpu_seconds']:.2f}s。连续优化预算大于旧有限pool，不宣称同预算优势。环境step/训练/actor更新/checkpoint写入均0。

首次尝试因NumPy bool_报告序列化错误中止，科学设置未改。其额外CPU与模型计数下界见[调试账本](continuous-debug.json)，在途精确计数不完整；不能把本次修复后成功运行的预算当成本轮全部开销。该错误不作为科学UNKNOWN结果。

这仍是旧名义tube合同的分析，不是完整狗/臂阶段入口验证；成功没有说明策略学出新能力，失败也没有证明学习不可替代。
"""
    (OUT/'continuous-results.md').write_text(text,encoding='utf-8')


def main():
    assert sha(PROTOCOL)==PROTOCOL_HASH
    authority=json.loads(AUTHORITY.read_text(encoding='utf-8'));keys=authority['all']['unresolved_keys']
    assert len(keys)==42 and hashlib.sha256('\n'.join(keys).encode()).hexdigest()==KEYS_HASH
    selector=json.loads(SELECTOR.read_text(encoding='utf-8'))
    selected={f"{r['seed']}/{r['arm']}/{r['tick']}/{r['environment']}":r for r in selector['queries']}
    assert all(not selected[k]['common_gate_exists'] for k in keys)
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();results=[];write_report(results,keys,started,False)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(run_query,k,selected[k]):k for k in keys}
        pending=set(futures)
        while pending:
            done,pending=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                key=futures[future]
                try:result=future.result()
                except Exception as error:
                    result={'key':key,'status':'UNKNOWN_WORKER_ERROR','error':f'{type(error).__name__}: {error}'}
                results.append(result);write_report(results,keys,started,False)
                print(json.dumps({'phase':'continuous_query_done','key':key,'status':result['status'],
                                  'completed':len(results),'target':len(keys)}),flush=True)
            if not done:
                print(json.dumps({'phase':'continuous_main_progress','completed':len(results),'pending':len(pending),
                                  'elapsed_seconds':time.perf_counter()-started}),flush=True)
    write_report(results,keys,started,True)
    print(json.dumps({'phase':'continuous_finished','completed':len(results),
                      'feasible':sum(r['status']=='FEASIBLE_WITNESS' for r in results),
                      'unknown':sum(r['status']!='FEASIBLE_WITNESS' for r in results),
                      'wall_seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':main()
