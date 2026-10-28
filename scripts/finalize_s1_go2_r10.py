#!/usr/bin/env python3
"""Close the actual bounded R10 run in its existing plan and single index."""
import csv,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN='tm-s1-go2-soft-landing-r10'
SMALL=ROOT/'results/terrain_memory/simulation'/RUN


def main():
    index=json.loads((SMALL/'index.json').read_text());s=index['actual_summary']
    if s['completed_original_missions']:
        raise RuntimeError('A completed original mission requires the authorized joint-learning continuation; do not close as blocked.')
    reason='Material nominal full mission not established before the 12-start development allowance ended.'
    cases=[]
    policies=[('FIXED_S4',None)]+[(group,seed) for group in ('PPO_RAW_S4','PANN_PPO_S4') for seed in range(3)]
    for duration in (60,600):
        for group,seed in policies:
            for condition in ('nominal','yaw_plus5','yaw_minus5'):
                cases.append(dict(group=group,policy_seed=seed,condition=condition,terrain='tm02_anchor',
                    duration_s=duration,status='not_executed',reason=reason))
    for duration in (1800,3600):
        for group,seed in [('FIXED_S4',None),('PPO_RAW_S4',0),('PANN_PPO_S4',0)]:
            cases.append(dict(group=group,policy_seed=seed,condition='nominal',terrain='tm02_anchor',
                duration_s=duration,status='not_executed',reason='Same-policy core 600 s and preceding extension prerequisites not established.'))
    cases.append(dict(group='FIXED_S4',policy_seed=None,condition='nominal',terrain='rigid',duration_s=60,
        status='not_executed',reason='No material-qualified frozen S4 was established; the historical rigid R8 matrix was not reopened.'))
    with (SMALL/'evaluation-manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
    index.update(status='blocked_material_execution_development_allowance_exhausted',
        training_entry_satisfied=False,training_transitions=0,ppo_updates=0,trained_checkpoints=0,
        evaluation_manifest='evaluation-manifest.csv',evaluation_started=0,
        training_streams=[dict(group=g,seed=seed,status='not_executed',transitions=0,
            updates=0,target_transitions=32768,reason=reason) for g,seed in policies if seed is not None],
        named_formulations=['G','G+N','G+N+C'],
        executed_builds=['G','GN','GNC','GNC-r2','GNC-r3'],
        build_revision_scope='GNC-r2 adds an observed local affine intercept; GNC-r3 completes N with material-support body height. All versions and failures retained.',
        comparison_scope='Development records with different controller revisions and worker allocations, not paired policy-seed performance estimates.',
        wall_clock_limit_exhausted=False,development_allowance_exhausted=True,
        hardware_runs=0,nx_realtime_measured=False,
        outcome_distinctions=dict(stable_joint_execution='not_established',
            pann_changes_effective_actions='not_evaluated',task_improvement='not_evaluated'))
    empty=[a for a in index['assets'] if a['size']==0]
    index['empty_original_files']=empty
    index['assets']=[a for a in index['assets'] if a['size']>0]
    index['empty_file_policy']='Empty original logs remain on the execution host and are listed here; no nonexistent data are fabricated or uploaded.'
    (SMALL/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    config_path=ROOT/'experiments/terrain_memory/s1-go2-soft-landing-r10.json'
    config=json.loads(config_path.read_text());config['status']=index['status']
    config['actual_execution']=dict(index=str((SMALL/'index.json').relative_to(ROOT)),
        development_attempts=12,initialization_failures=3,physical_missions=9,
        complete_original_missions=0,integrated_simulation_s=s['integrated_simulation_s'],
        integrated_physical_steps=s['integrated_physical_steps'],ppo_streams_started=0,
        ppo_transitions=0,pann_weight_updates=0,formal_60s_started=0,core_600s_started=0,
        optional_1800s_started=0,optional_3600s_started=0,
        observer_replay='Existing R9 causal-record replay only; 4169 state updates and 17 recovery queries, no closed-loop or training claim',
        status=index['status'],development_allowance_exhausted=True,wall_clock_limit_exhausted=False)
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    prefix=f'''**R10 实际结果（2026-10-28）：开发额度12/12已结束，材料完整任务0/9，联合PPO未启动。** 12次启动包括3次零物理步初始化失败和9次真实Go2材料任务；累计{ s['integrated_simulation_s']:.4f}秒、{s['integrated_physical_steps']}个0.2ms物理步。G、G＋N与G＋N＋C均保留完整记录，C的局部力参考和支撑高度修正逐版保存。最终0.215m/s版本通过去程和远端转向，在返程32.5672秒触发原材料域；仍未完成原任务。

冻结PANN的独立观察器完成既有R9记录的4169次状态推进与17次条件恢复查询，权重SHA256与原B ideal seed0一致；这不是闭环成功。标准PPO接口已准备但未执行，六流训练、PPO更新、新策略权重、21条60秒与21条600秒正式评价均为0。三模块稳定联合执行尚未成立，PANN改变动作及改善任务均未评价。真实材料、Go2物理、PD/力矩、任务阈值未放宽；未启动硬件，未测NX实时。

48小时墙钟上限未耗尽，开发启动额度已耗尽。剩余49个预定评价case保留清单与未执行原因，不自动刷新额度。当前已无R10仿真或训练作业；`tmux`保留执行日志入口。

[唯一运行索引](results/terrain_memory/simulation/{RUN}/index.json) · [全部启动与结果](results/terrain_memory/simulation/{RUN}/episodes.csv) · [轨迹与阈值图](results/terrain_memory/simulation/{RUN}/development-trajectories.png) · [未执行评价清单](results/terrain_memory/simulation/{RUN}/evaluation-manifest.csv) · [完整数据与负例视频](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/{RUN})

'''
    readme=ROOT/'README.md';text=readme.read_text();first,rest=text.split('\n',1)
    readme.write_text(first+'\n\n'+prefix+'以下R9及更早内容保留其历史身份。\n'+rest)
    plan=ROOT/'docs/TERRAIN_MEMORY_PLAN.md';text=plan.read_text()
    text=text.replace('状态：**planned_not_executed**','状态：**blocked_material_execution_development_allowance_exhausted**',1)
    point=text.index('## 1.')
    plan_prefix=prefix.replace('(results/','(../results/')
    text=text[:point]+'## 本轮实际执行结果\n\n'+plan_prefix+'以下保留本轮执行时的协议；预算不重新授权。\n\n'+text[point:]
    plan.write_text(text)
    agents=ROOT/'AGENTS.md';text=agents.read_text();first,rest=text.split('\n',1)
    agents.write_text(first+'\n\n'+
        '**R10已结束本轮开发：12/12启动（3次零步初始化失败＋9次物理任务），完整材料任务0/9；PPO训练0。** 最后0.215m/s在返程32.5672秒材料域退出。不得重新起钟、重开额度或启动未通过前置条件的联合训练。实际结果见[唯一索引](results/terrain_memory/simulation/'+RUN+'/index.json)。\n\n'+
        '以下为本轮冻结执行约定，保留历史协议身份；当前状态以以上实际结果为准。\n'+rest)
    copies=[('/root/autodl-tmp/MCP-RL-r10','G'),('/root/autodl-tmp/MCP-RL-r10-gn','GN'),
        ('/root/autodl-tmp/MCP-RL-r10-gnc','GNC'),('/root/autodl-tmp/MCP-RL-r10-gnc-r2','GNC-r2'),
        ('/root/autodl-tmp/MCP-RL-r10-gnc-r3','GNC-r3')]
    for source,build in copies:
        name='experiments/terrain_memory/s1-go2-r10-'+build+'.patch'
        shutil.copy2(Path(source)/name,ROOT/name)
    notes=f'''R10 bounded development ended with 12/12 starts: 3 zero-physics initialization failures and 9 actual Go2/TM02 missions. Complete original material missions: 0/9.

The final G+N+C implementation at cap 0.215 m/s crossed the outbound strip and turned at the far endpoint, then exited the unchanged material strain domain on return at 32.5672 s. Other full recorded attempts ended in material-domain exit, body fall or non-foot contact. None is a completed round trip.

Actual integration: {s['integrated_simulation_s']:.4f} simulated seconds / {s['integrated_physical_steps']} physical steps. The original plant, PD/torque bounds, material law, 0.02 strain limit and completion criteria remain unchanged. Three named formulations and all five compiled revisions are retained; no universal impossibility claim is made.

Independent frozen-PANN replay: 4169 causal observer state updates and 17 recovery queries on an existing R9 recording. PANN weight updates: 0. PPO streams, sampled training transitions, PPO updates and new policies: 0. The 21 original and 21 core 600 s evaluations, rigid regression and optional long cases were not executed because the prerequisite was not established; all 49 cases are listed in evaluation-manifest.csv.

Joint execution is not established; PANN action dependence and task improvement were not evaluated. The development allowance is exhausted, not the 48-hour wall-clock ceiling. No hardware or NX realtime claim.

Assets are original execution-host arrays, model/source identities, full negative videos and causal replay outputs. Every asset is below 2 GiB. Uploads use AutoDL academic acceleration directly from the execution host. Verification uses URL/size/SHA256 metadata, with no asset redownload or user-PC archive.

Index: https://github.com/kaiwen123-yang/MCP-RL/blob/main/results/terrain_memory/simulation/{RUN}/index.json
'''
    (ROOT/'work/r10-release-notes.md').write_text(notes)
    print(json.dumps(dict(assets=len(index['assets']),bytes=sum(a['size'] for a in index['assets']),cases=len(cases),summary=s)),flush=True)


if __name__=='__main__':main()
