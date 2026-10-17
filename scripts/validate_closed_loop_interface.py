"""Frozen identity checks, then fixed-actor execution-aware MPC evaluations.

Identity uses explicitly labelled historical disturbances. New planner calls
use nominal w=0 while support W remains .04. No training or checkpoint write.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from response_ppo.env import EnvironmentConfig
from response_ppo.ppo_backend import GaussianActor, PPOConfig
from response_ppo.closed_loop_prediction import NativeContext, native_step, predict_timeline
from response_ppo.execution_aware_env import ExecutionAwareServo

OUT = ROOT/'notes/closed-loop-interface-2026-10-17'
WORK = ROOT/'work/closed-loop-interface-2026-10-17/actual-evaluations'
PROTOCOL = OUT/'protocol.md'
PROTOCOL_HASH = '59a479700f73ce6b1cbc42d4fc387b4bc70f28b964dd454dd5ef1b829ac906a8'
SEEDS = (17,29,43)
ARMS = ('joint_support','fixed_support','joint_penalty','fixed_penalty')
PRODUCTION = ('src/response_ppo/env.py','src/response_ppo/pilot.py','src/response_ppo/ppo_backend.py',
              'src/response_ppo/realization.py','src/response_codesign/splates_bridge.py','src/response_codesign/dynamics.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def load_case(seed,arm):
    base=ROOT/f'work/ppo-mechanism-2026-10-16/formal/seedrun-{seed}/seed-{seed}'
    checkpoint=base/f'{arm}.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    setting=json.loads((base/'result.json').read_text(encoding='utf-8'))
    actor=GaussianActor(PPOConfig(**setting['ppo']));actor.load_state_dict(saved['actor']);actor.eval()
    replay=ROOT/f'work/neural-demand-audit-2026-10-17/replays/{seed}-{arm}.npz'
    with np.load(replay,allow_pickle=False) as z:
        data={k:torch.from_numpy(z[k].copy()) for k in z.files}
    cfg=EnvironmentConfig(**setting['environment'])
    context=NativeContext(cfg,data['P'],data['B'],data['E'])
    return actor,saved,cfg,context,data,checkpoint,replay


def historical_external(seed,cfg,steps=256,batch=16):
    """Recreate only original initialization draws, not a physical trajectory."""
    gen=torch.Generator(device='cpu').manual_seed(seed*10000+5000)
    initial=.015*torch.randn(batch,3,generator=gen)
    count=(steps+cfg.horizon*cfg.command_steps)//cfg.target_steps+3
    magnitude=.15+.15*torch.rand(batch,count,generator=gen)
    sign=torch.where(torch.rand(batch,count,generator=gen)<.5,-1.,1.)
    goals=magnitude*sign
    phase=2*np.pi*torch.rand(batch,generator=gen)
    disturbance=torch.stack([cfg.disturbance_bound*(.7*torch.sin(3.1*(j*cfg.dt)+phase)
                                                  +.3*torch.cos(7.3*(j*cfg.dt)+2*phase))
                             for j in range(steps)])
    targets=torch.stack([goals[:,j//cfg.target_steps] for j in range(steps)])
    return initial,targets,disturbance


def difference(a,b):
    equal=torch.equal(a,b)
    matching_nonfinite=(~torch.isfinite(a)) & (~torch.isfinite(b)) & (a==b)
    finite=torch.isfinite(a)&torch.isfinite(b)
    bad=(~finite)&(~matching_nonfinite)
    maximum=float('inf') if bool(bad.any()) else (float((a[finite]-b[finite]).abs().max()) if bool(finite.any()) else 0.)
    return {'exact':equal,'max_abs':maximum}


def identity_case(seed,arm):
    actor,saved,cfg,ctx,data,checkpoint,replay=load_case(seed,arm)
    checkpoint_hash=sha(checkpoint)
    initial,targets,disturbance=historical_external(seed,cfg)
    assert torch.equal(initial,data['x'][0]) and torch.equal(targets,data['target'])
    next_x=torch.cat((data['x'][1:],data['terminal_x'][None]),dim=0)
    next_z=torch.cat((data['z'][1:],data['terminal_z'][None]),dim=0)
    metrics=('task','total','V','excess','empty','action','reference_input','support_width','position',
             'reference_position','actual_domain_violation','reference_domain_violation','tube_violation','target','geometric_bound')
    checks={key:{'exact':True,'max_abs':0.} for key in (*metrics,'obs','mean','x_next','z_next','interval_lo','interval_hi')}
    started=time.perf_counter()
    with torch.no_grad():
        for tick in range(256):
            result=native_step(actor.mean,data['x'][tick],data['z'][tick],data['command'][tick],
                               data['raw_design'][tick],targets[tick],tick,ctx,
                               support=arm.endswith('support'),disturbance=disturbance[tick])
            expected={key:data[key][tick] for key in metrics}
            expected.update(obs=data['obs'][tick],mean=data['mean'][tick,:,0],x_next=next_x[tick],
                            z_next=next_z[tick],interval_lo=data['interval_lo'][tick],interval_hi=data['interval_hi'][tick])
            for key,wanted in expected.items():
                check=difference(result[key],wanted)
                checks[key]['exact'] &= check['exact']
                checks[key]['max_abs']=max(checks[key]['max_abs'],check['max_abs'])
        prediction=predict_timeline(actor.mean,data['x'][0],data['z'][0],
                                    data['command'][:80].transpose(0,1),data['raw_design'][0],
                                    targets[:80].transpose(0,1),0,ctx,support=arm.endswith('support'),
                                    disturbance_timeline=disturbance[:80].transpose(0,1),
                                    scenario_label='historical-disturbance-identity-only-not-planning-preview')
    expected_x=torch.cat((data['x'][:1],next_x[:80]),dim=0)
    expected_z=torch.cat((data['z'][:1],next_z[:80]),dim=0)
    multi={'x':difference(prediction['x'],expected_x),'z':difference(prediction['z'],expected_z)}
    multi.update({key:difference(prediction['steps'][key],data[key][:80]) for key in metrics})
    one_max=max(v['max_abs'] for v in checks.values());multi_max=max(v['max_abs'] for v in multi.values())
    assert sha(checkpoint)==checkpoint_hash
    return {'seed':seed,'arm':arm,'one_step':checks,'one_step_max':one_max,
            'timeline80':multi,'timeline80_max':multi_max,'passed':one_max<=1e-6 and multi_max<=1e-5,
            'checkpoint_sha256':checkpoint_hash,'replay_sha256':sha(replay),'wall_seconds':time.perf_counter()-started,
            'budget':{'one_step_model_transitions':256*16,'historical_timeline_model_transitions':80*16,
                      'identity_actor_observations':(256+80)*16,'new_execution_simulation_transitions':0,
                      'training_updates':0},
            'disturbance_semantics':'Explicit reconstruction of historical disturbance for identity only; nominal MPC w remains0 and supportW=.04'}


def summary(arrays):
    return {'task_cost':float(arrays['task'].mean()),'total_cost':float(arrays['total'].mean()),
            'actual_domain_violation_rate':float(arrays['actual_domain_violation'].mean()),
            'reference_domain_violation_rate':float(arrays['reference_domain_violation'].mean()),
            'tube_violation_rate':float(arrays['tube_violation'].mean()),
            'empty_support_rate':float(arrays['empty'].mean()),
            'one_step_violation_rate':float((arrays['excess']>1e-8).float().mean()),
            'max_V':float(arrays['V'].max()),'max_excess':float(arrays['excess'].max()),
            'maximum_input':float(arrays['action'].abs().max())}


def evaluate_case(seed,arm):
    actor,saved,cfg,_,old,checkpoint,_=load_case(seed,arm)
    checkpoint_before=sha(checkpoint)
    started=time.perf_counter()
    with torch.no_grad():
        env=ExecutionAwareServo(cfg,saved['design_mean'][None].expand(16,-1),
                                seed*10000+5000,256,arm.endswith('support'),actor.mean)
        assert torch.equal(env.x,old['x'][0]) and torch.equal(env.z,old['z'][0])
        obs=env.observation();records={};metrics={}
        for tick in range(256):
            mean=actor.mean(obs)
            for key,value in {'x':env.x,'z':env.z,'command':env.command,'obs':obs,'mean':mean}.items():
                records.setdefault(key,[]).append(value.detach().clone())
            next_obs,_,step=env.step(torch.tanh(mean[:,0]))
            for key,value in step.items():metrics.setdefault(key,[]).append(value.detach().clone())
            obs=next_obs
    wall=time.perf_counter()-started
    arrays={key:torch.stack(value) for key,value in metrics.items()}
    assert torch.equal(arrays['target'],old['target'])
    full={key:torch.stack(value).numpy() for key,value in records.items()}
    full.update({key:value.numpy() for key,value in arrays.items()})
    full.update(terminal_x=env.x.numpy(),terminal_z=env.z.numpy(),P=env.P.numpy(),B=env.B.numpy(),E=env.E.numpy())
    full['plan_ticks']=np.array([p['tick'] for p in env.plan_diagnostics],dtype=np.int64)
    full['plan_commands']=torch.stack([p['selected_sequence_planner_coordinates'] for p in env.plan_diagnostics]).numpy()
    full['plan_reference_gate_feasible']=torch.stack([p['selected_reference_gate_feasible'] for p in env.plan_diagnostics]).numpy()
    full['plan_selected_found']=torch.stack([p['selected_found_in_evaluated_pool'] for p in env.plan_diagnostics]).numpy()
    diagnostic_keys=env.plan_diagnostics[0]['selected_diagnostics']
    for key in diagnostic_keys:
        full['plan_'+key]=torch.stack([p['selected_diagnostics'][key] for p in env.plan_diagnostics]).numpy()
    destination=WORK/f'{seed}-{arm}.npz';np.savez_compressed(destination,**full)
    counts=env.planner.objective_counts()
    assert env.execution_prediction_low_steps==counts['candidate_macro_steps']*cfg.command_steps
    unchanged=checkpoint_before==sha(checkpoint) and all(torch.equal(value,saved['actor'][key]) for key,value in actor.state_dict().items())
    assert unchanged
    old_summary,new_summary=summary(old),summary(arrays)
    return {'seed':seed,'arm':arm,'status':'completed','old_reference_planner':old_summary,
            'new_actual_planner':new_summary,
            'task_cost_change_new_minus_old':new_summary['task_cost']-old_summary['task_cost'],
            'task_relative_change_percent':100*(new_summary['task_cost']/old_summary['task_cost']-1),
            'initial_x_z_and_goals_matched':True,'same_hidden_disturbance_initialization':True,
            'checkpoint_unchanged':unchanged,'checkpoint_sha256':checkpoint_before,
            'output_path':destination.relative_to(ROOT).as_posix(),'output_sha256':sha(destination),
            'selected_candidates_found_fraction':float(torch.tensor(full['plan_selected_found']).float().mean()),
            'selected_reference_gate_feasible_fraction':float(torch.tensor(full['plan_reference_gate_feasible']).float().mean()),
            'selected_diagnostic_summary':{key:float(torch.stack([p['selected_diagnostics'][key] for p in env.plan_diagnostics]).float().max())
                                           for key in diagnostic_keys if key.endswith('_max')},
            'wall_seconds':wall,'budget':{'new_execution_simulation_transitions':4096,
                                         'planner_instances':env.planner_instances,'planner_batch_calls':env.planner.n_plans,
                                         **counts,'prediction_low_steps':env.execution_prediction_low_steps,
                                         'prediction_actor_observations':env.execution_prediction_low_steps,
                                         'execution_actor_observations':4096,'training_updates':0,'actor_updates':0,
                                         'gradient_optimizer_steps':0,'checkpoint_writes':0},
            'interpretation':'Same frozen actor; new counterfactual simulation measures prediction/planning effects, not learned capability gains',
            'gate':'Primary CEM gate remains reference-domain only; actual/input/tube/support reported diagnostically'}


def write_outputs(identities,evaluations,status,production_before):
    all_pass=len(identities)==12 and all(v['passed'] for v in identities)
    report={'date':'2026-10-17','status':status,'identity_passed_all12':all_pass,
            'protocol_sha256':sha(PROTOCOL),'script_sha256':sha(__file__),
            'protocol_content_sha256':text_sha(PROTOCOL),'script_content_sha256':text_sha(__file__),
            'implementation_sha256':{p:sha(ROOT/p) for p in ('src/response_ppo/closed_loop_prediction.py','src/response_ppo/execution_aware_env.py')},
            'production_before':production_before,'production_after':{p:sha(ROOT/p) for p in PRODUCTION},
            'identity_cases':identities,'actual_planner_cases':evaluations,
            'identity_budget':{key:sum(v['budget'][key] for v in identities) for key in identities[0]['budget']} if identities else {},
            'actual_planner_budget':{key:sum(v['budget'][key] for v in evaluations) for key in evaluations[0]['budget']} if evaluations else {},
            'boundary':'Historical scenario is used only for identity. Actual planner predicts w0 with supportW=.04, unchanged reference primary gate; no training, weight update or learning-benefit claim.'}
    (OUT/'validation-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    identity_rows='\n'.join(f"| {v['seed']} | {v['arm']} | {v['one_step_max']:.3g} | {v['timeline80_max']:.3g} | {v['passed']} |" for v in identities)
    eval_rows='\n'.join(f"| {v['seed']} | {v['arm']} | {v['old_reference_planner']['task_cost']:.10g} | {v['new_actual_planner']['task_cost']:.10g} | {v['task_relative_change_percent']:.5g}% | {v['new_actual_planner']['actual_domain_violation_rate']:.5g} | {v['new_actual_planner']['tube_violation_rate']:.5g} | {v['wall_seconds']:.2f} |" for v in evaluations)
    if evaluations:
        old_mean=sum(v['old_reference_planner']['task_cost'] for v in evaluations)/len(evaluations)
        new_mean=sum(v['new_actual_planner']['task_cost'] for v in evaluations)/len(evaluations)
        worse_tube=[v for v in evaluations if v['new_actual_planner']['tube_violation_rate']>v['old_reference_planner']['tube_violation_rate']]
        tradeoff='；'.join(f"{v['seed']}/{v['arm']}：{v['old_reference_planner']['tube_violation_rate']:.8g}→{v['new_actual_planner']['tube_violation_rate']:.8g}" for v in worse_tube) or '尚无已完成case的tube违反率上升'
        aggregate=f"已完成{len(evaluations)}组的简单平均task cost为{old_mean:.12g}→{new_mean:.12g}（{100*(new_mean/old_mean-1):.6g}%）。这只描述固定评估集合的规划变化，不是学习收益或统计泛化结论。tube违反率上升的case为：{tradeoff}。其他约束及空集必须一并查看，不能将成本下降写成全面改善。"
    else:
        aggregate='尚未执行actual-planner评估，不汇总任务效果。'
    text=f"""# 实际闭环接口核验与固定actor重规划评估

日期：2026-10-17。状态`{status}`。[协议](protocol.md)、[逐项数值](validation-results.json)、[实现说明](interface-implementation.md)、[脚本](../../scripts/validate_closed_loop_interface.py)。全部12组identity通过：`{all_pass}`。

一步identity使用全部256×16个旧评估原生锚点；80步timeline使用原最初80个命令和明确标注的历史扰动，仅用于核验，不给新planner未来扰动预览。参考z连续，实际任务/原total与全部原metric分开核对。一次映射门槛1e−6、80步门槛1e−5先冻结。

| seed | arm | 一步最大误差 | 80步最大误差 | 通过 |
|---:|---|---:|---:|---|
{identity_rows}

只有全部identity通过后才运行新adapter。全12组原first-short的seed/actor/design/初态/任务/实际扰动条件不变；新planner读当前x/z，名义预测w=0，但supportW仍为.04。它是**新的反事实仿真**，不是旧记录重放。主可行门槛仍为原reference域，实际域/输入/tube/support均保留诊断。

| seed | arm | 旧planner任务cost | 新actual planner任务cost | 相对变化 | 新actual域违反率 | 新tube违反率 | 新仿真墙钟秒 |
|---:|---|---:|---:|---:|---:|---:|---:|
{eval_rows}

{aggregate}

计数按已完成case累计，完整账本见JSON：identity离线模型步`{sum(v['budget']['one_step_model_transitions']+v['budget']['historical_timeline_model_transitions'] for v in identities)}`；新执行仿真转移`{sum(v['budget']['new_execution_simulation_transitions'] for v in evaluations)}`；候选预测微步`{sum(v['budget']['prediction_low_steps'] for v in evaluations)}`。预测每个候选微步现在包含真实网络与support/fallback计算，同候选数不等于同计算成本。

所有原checkpoint与生产源码哈希单列；没有训练、参数更新、梯度优化器步或checkpoint写入。任务变化只归于相同冻结actor下的预测/规划改变，不称RL学出新能力，也不证明优于同信息同目标直接优化。逐组原/new全部指标与selected-plan诊断均保留在JSON。实际case墙钟合计`{sum(v['wall_seconds'] for v in evaluations):.6g}s`；这是本轮运行记录，可能与其它分析并行，不作独占实时率或公平加速比benchmark。
"""
    (OUT/'validation-results.md').write_text(text,encoding='utf-8')


def main():
    assert text_sha(PROTOCOL)==PROTOCOL_HASH
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32)
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    before={p:sha(ROOT/p) for p in PRODUCTION};identities=[];evaluations=[]
    for seed in SEEDS:
        for arm in ARMS:
            result=identity_case(seed,arm);identities.append(result)
            write_outputs(identities,evaluations,'identity_running',before)
            print(json.dumps({'phase':'identity','seed':seed,'arm':arm,'passed':result['passed'],
                              'one_step_max':result['one_step_max'],'timeline80_max':result['timeline80_max']}),flush=True)
    if not all(v['passed'] for v in identities):
        write_outputs(identities,evaluations,'identity_failed_actual_evaluation_not_run',before)
        return
    for seed in SEEDS:
        for arm in ARMS:
            result=evaluate_case(seed,arm);evaluations.append(result)
            write_outputs(identities,evaluations,'actual_evaluation_running',before)
            print(json.dumps({'phase':'actual_eval','seed':seed,'arm':arm,'old_cost':result['old_reference_planner']['task_cost'],
                              'new_cost':result['new_actual_planner']['task_cost'],'wall_seconds':result['wall_seconds'],
                              'prediction_steps':result['budget']['prediction_low_steps']}),flush=True)
    assert before=={p:sha(ROOT/p) for p in PRODUCTION}
    write_outputs(identities,evaluations,'completed',before)
    print(json.dumps({'completed_cases':len(evaluations),'new_execution_simulation_transitions':sum(v['budget']['new_execution_simulation_transitions'] for v in evaluations),'training_updates':0}),flush=True)


if __name__=='__main__':main()
