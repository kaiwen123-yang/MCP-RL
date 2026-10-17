"""Re-select existing saved MPC pools under the already declared common gate.

No predictor, environment, actor, optimizer or checkpoint is loaded. All
candidate sequences, costs and margins are old saved arrays. A failed finite
pool has no returned corrected plan; it is not a global infeasibility claim.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'notes/closed-loop-interface-2026-10-17/pool-and-gradient-results.json'
OUT = ROOT/'notes/stage-feasibility-2026-10-17'
SEEDS = (17,29,43)
ARMS = ('joint_support','fixed_support','joint_penalty','fixed_penalty')
TICKS = (0,60,120,180)
MARGINS = ('reference_domain_margin_max','actual_domain_margin_max','input_margin_max','tube_margin_max',
           'actual_one_step_excess_max','support_worst_case_excess_max')
THRESHOLDS = {'reference_domain_margin_max':0.,'actual_domain_margin_max':1e-7,
              'input_margin_max':0.,'tube_margin_max':1e-8}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(cost, valid):
    valid=valid&np.isfinite(cost)
    return int(np.argmin(np.where(valid,cost,np.inf))) if np.any(valid) else None


def chosen_record(arrays, env, index):
    if index is None:return None
    return {'index':index,'actual_cost':float(arrays['actual_cost'][env,index]),
            'reference_cost':float(arrays['reference_cost'][env,index]),
            'margins':{key:float(arrays[key][env,index]) for key in MARGINS},
            'support_empty_any':bool(arrays['support_empty_any'][env,index]),
            'fallback_used_any':bool(arrays['fallback_used_any'][env,index]),
            'passes_common_gate':all(arrays[key][env,index]<=limit for key,limit in THRESHOLDS.items()),
            'tube_violation':bool(arrays['tube_margin_max'][env,index]>1e-8)}


def values_summary(values):
    if not values:return {'count':0}
    a=np.asarray(values,dtype=np.float64)
    return {'count':len(values),'mean':float(a.mean()),'median':float(np.median(a)),
            'minimum':float(a.min()),'maximum':float(a.max()),
            'q25':float(np.quantile(a,.25)),'q75':float(np.quantile(a,.75))}


def aggregate(rows):
    feasible=[r for r in rows if r['common_gate_exists']]
    empty=[r for r in rows if not r['common_gate_exists']]
    output={'queries':len(rows),'common_gate_feasible':len(feasible),'empty_common_gate_pool':len(empty),
            'empty_pool_initial_tube_outside':sum(r['initial_tube_violation_recorded'] for r in empty),
            'empty_pool_initial_tube_inside':sum(not r['initial_tube_violation_recorded'] for r in empty),
            'feasible_pool_initial_tube_outside':sum(r['initial_tube_violation_recorded'] for r in feasible),
            'feasible_pool_initial_tube_inside':sum(not r['initial_tube_violation_recorded'] for r in feasible),
            'primary_actual_gate_violation_with_feasible':sum(not r['primary_actual']['passes_common_gate'] for r in feasible),
            'primary_actual_tube_violation_with_feasible':sum(r['primary_actual']['tube_violation'] for r in feasible),
            'primary_reference_tube_violation_with_feasible':sum(r['primary_reference']['tube_violation'] for r in feasible),
            'corrected_actual_common_gate_violations':sum(not r['corrected_actual']['passes_common_gate'] for r in feasible),
            'corrected_reference_common_gate_violations':sum(not r['corrected_reference']['passes_common_gate'] for r in feasible),
            'corrected_actual_distinct_command_changes':sum(r['corrected_actual_distinct_command_changed'] for r in feasible),
            'corrected_reference_distinct_command_changes':sum(r['corrected_reference_distinct_command_changed'] for r in feasible),
            'failure_cause_counts':dict(Counter(r['failure_cause'] for r in empty)),
            'corrected_actual_support_empty_diagnostic':sum(r['corrected_actual']['support_empty_any'] for r in feasible),
            'corrected_actual_one_step_excess_diagnostic':sum(r['corrected_actual']['margins']['actual_one_step_excess_max']>1e-8 for r in feasible)}
    old=[r['primary_actual']['actual_cost'] for r in feasible]
    corrected=[r['corrected_actual']['actual_cost'] for r in feasible]
    ref_corrected=[r['corrected_reference']['actual_cost'] for r in feasible]
    output['costs_on_same_feasible_query_subset']={
        'primary_actual':values_summary(old),'corrected_actual':values_summary(corrected),
        'corrected_reference_evaluated_actual':values_summary(ref_corrected),
        'corrected_minus_primary_actual':values_summary([b-a for a,b in zip(old,corrected)]),
        'corrected_actual_minus_corrected_reference_actual':values_summary([a-b for a,b in zip(corrected,ref_corrected)]),
        'relative_change_of_subset_means_percent':100*(np.mean(corrected)/np.mean(old)-1) if old and np.mean(old)!=0 else None}
    output['failed_pool_min_tube_margin'] = values_summary([r['best_tube_diagnostic']['tube_margin'] for r in empty if r['best_tube_diagnostic'] is not None])
    return output


def main():
    source=json.loads(SOURCE.read_text(encoding='utf-8'))
    assert {(c['seed'],c['arm']) for c in source['cases']}=={(s,a) for s in SEEDS for a in ARMS}
    rows=[];pool_records=[];metric_sources=[];consistency=[];candidate_total=0
    for seed in SEEDS:
        for arm in ARMS:
            case=next(c for c in source['cases'] if c['seed']==seed and c['arm']==arm)
            metric_path=ROOT/f'work/ppo-mechanism-2026-10-16/formal/seedrun-{seed}/seed-{seed}/{arm}-short-trace.npz'
            with np.load(metric_path,allow_pickle=False) as archive:
                recorded_V=archive['V'].copy();recorded_tube=archive['tube_violation'].copy()
            metric_sources.append({'path':metric_path.relative_to(ROOT).as_posix(),'sha256':sha(metric_path),
                                   'fields_used':['V','tube_violation'],'shape':list(recorded_V.shape),
                                   'semantics':'Stored poststep metric; query tick>0 uses prior tick. At tick0 x=z so V0=0, although x itself is not zero.'})
            assert {q['tick'] for q in case['queries']}==set(TICKS)
            for query in sorted(case['queries'],key=lambda q:q['tick']):
                path=ROOT/query['pool_relative_path'];digest=sha(path)
                assert digest==query['pool_byte_sha256']
                with np.load(path,allow_pickle=False) as archive:
                    arrays={key:archive[key].copy() for key in archive.files}
                shape=arrays['actual_cost'].shape
                assert shape[0]==16
                candidate_total+=int(np.prod(shape))
                reference=arrays['reference_domain_margin_max']<=0
                actual_domain=arrays['actual_domain_margin_max']<=1e-7
                inputs=arrays['input_margin_max']<=0
                tubes=arrays['tube_margin_max']<=1e-8
                finite=np.isfinite(arrays['actual_cost'])&np.isfinite(arrays['reference_cost'])
                common=reference&actual_domain&inputs&tubes&finite
                pool_records.append({'path':query['pool_relative_path'],'sha256':digest,
                                     'shape':list(shape),'candidate_count':int(np.prod(shape)),
                                     'nonfinite_cost_pairs':int((~finite).sum()),
                                     'threshold_distances':{key:float(np.min(np.abs(arrays[key]-limit))) for key,limit in THRESHOLDS.items()}})
                for env in range(16):
                    old_index=select(arrays['reference_cost'][env],reference[env])
                    primary_index=select(arrays['actual_cost'][env],reference[env])
                    new_index=select(arrays['actual_cost'][env],common[env])
                    corrected_ref_index=select(arrays['reference_cost'][env],common[env])
                    assert old_index is not None and primary_index is not None
                    old_record=query['selections'][env]
                    existing=old_record['common_full_physical_screen']
                    exists=new_index is not None
                    agrees=(old_index==old_record['reference_winner_index'] and primary_index==old_record['actual_winner_index']
                            and exists==existing['exists'] and (not exists or
                            (new_index==existing['new_index'] and corrected_ref_index==existing['reference_index'])))
                    consistency.append(agrees)
                    stage_masks=[finite[env],finite[env]&reference[env],
                                 finite[env]&reference[env]&actual_domain[env],
                                 finite[env]&reference[env]&actual_domain[env]&inputs[env],common[env]]
                    names=['finite_costs','reference_domain','actual_domain','input','tube']
                    failure=None if exists else next(name for name,mask in zip(names,stage_masks) if not np.any(mask))
                    pre_tube=stage_masks[-2]
                    best_tube_index=select(arrays['tube_margin_max'][env],pre_tube)
                    best_tube=None if best_tube_index is None else {
                        'index':best_tube_index,'tube_margin':float(arrays['tube_margin_max'][env,best_tube_index]),
                        'excess_above_original_tolerance':float(arrays['tube_margin_max'][env,best_tube_index]-1e-8),
                        'actual_cost':float(arrays['actual_cost'][env,best_tube_index]),
                        'meaning':'Diagnostic minimum tube margin within finite/reference/actual/input gates; not a selected feasible fallback'}
                    row={'seed':seed,'arm':arm,'tick':query['tick'],'environment':env,
                         'initial_V_from_record':0. if query['tick']==0 else float(recorded_V[query['tick']-1,env]),
                         'initial_tube_violation_recorded':False if query['tick']==0 else bool(recorded_tube[query['tick']-1,env]),
                         'pool_path':query['pool_relative_path'],'candidate_count':shape[1],
                         'counts_after_gate':{name:int(mask.sum()) for name,mask in zip(names,stage_masks)},
                         'common_gate_exists':exists,'failure_cause':failure,
                         'primary_reference':chosen_record(arrays,env,old_index),
                         'primary_actual':chosen_record(arrays,env,primary_index),
                         'corrected_reference':chosen_record(arrays,env,corrected_ref_index),
                         'corrected_actual':chosen_record(arrays,env,new_index),
                         'best_tube_diagnostic':best_tube,
                         'corrected_actual_distinct_command_changed':bool(exists and not np.array_equal(arrays['commands_planner'][env,new_index],arrays['commands_planner'][env,primary_index])),
                         'corrected_reference_distinct_command_changed':bool(exists and not np.array_equal(arrays['commands_planner'][env,corrected_ref_index],arrays['commands_planner'][env,old_index])),
                         'matches_prior_saved_secondary_selection':agrees}
                    if exists:
                        assert row['corrected_actual']['passes_common_gate'] and row['corrected_reference']['passes_common_gate']
                        assert row['corrected_actual']['actual_cost']>=row['primary_actual']['actual_cost']-1e-14
                        assert row['corrected_actual']['actual_cost']<=row['corrected_reference']['actual_cost']+1e-14
                    rows.append(row)
    assert len(rows)==768 and all(consistency)
    groups={'all':aggregate(rows),
            'support':aggregate([r for r in rows if r['arm'].endswith('support')]),
            'penalty':aggregate([r for r in rows if r['arm'].endswith('penalty')])}
    by_case=[{'seed':seed,'arm':arm,**aggregate([r for r in rows if r['seed']==seed and r['arm']==arm])}
             for seed in SEEDS for arm in ARMS]
    by_tick=[{'tick':tick,'family':family,**aggregate([r for r in rows if r['tick']==tick and r['arm'].endswith(family)])}
             for tick in TICKS for family in ('support','penalty')]
    assert groups['penalty']['empty_common_gate_pool']==140
    assert groups['penalty']['primary_actual_tube_violation_with_feasible']==163
    result={'date':'2026-10-17','scope':'Re-selection of all existing saved fixed candidate pools; no new candidates, predictions, env runs or policy changes',
            'source':{'path':SOURCE.relative_to(ROOT).as_posix(),'sha256':sha(SOURCE)},
            'script_sha256':sha(__file__),'thresholds_unchanged':THRESHOLDS,
            'gate_definition':'finite actual/reference costs AND original reference gate AND actual domain original1e-7 tolerance AND hardware input margin<=0 AND original tube1e-8 tolerance',
            'excluded_from_gate':'support_empty, fallback, one-step excess and worst-case excess remain separately reported; this is the already-declared nominal common gate, not a new complete robust or stage-entry contract',
            'prior_secondary_selection_matches':sum(consistency),'pools':pool_records,'historical_metric_sources':metric_sources,'groups':groups,
            'by_checkpoint':by_case,'by_tick':by_tick,'queries':rows,
            'fixed_support_all_nonempty':all(r['common_gate_exists'] for r in rows if r['arm']=='fixed_support'),
            'failed_query_ids':[{key:r[key] for key in ('seed','arm','tick','environment','failure_cause')} for r in rows if not r['common_gate_exists']],
            'budget':{'source_pool_files':len(pool_records),'source_old_metric_files':len(metric_sources),'fixed_queries':len(rows),'saved_candidate_records_inspected':candidate_total,
                      'new_candidate_sequences':0,'new_model_transitions':0,'new_environment_transitions':0,
                      'new_actor_calls':0,'training_updates':0,'checkpoint_reads':0,'checkpoint_writes':0},
            'interpretation':['163 feasible penalty queries demonstrate a selector omission under the declared nominal gate, not a proved feedback-capability deficit.',
                              'The140 empty penalty pools only show failure of these saved finite candidate sets; they do not prove continuous-command or physical infeasibility.',
                              'The corrected selector pays the measured task-cost cost of honoring the existing gate; no closed-loop performance or training improvement is evaluated.',
                              'All methods use identical candidate predictions and the same gate. Actual-objective selection is definitionally no worse in actual cost than reference-objective selection within that common set.',
                              '105 empty pools start outside the tube,35 start inside; two initially outside queries still have feasible candidates. Initial violation alone does not prove lack of recovery.',
                              'The native tube is an existing diagnostic requirement, not yet the complete long-stage entry contract required by the robot research question.']}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'selector-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_report(result)
    print(json.dumps({'groups':groups,'fixed_support_all_nonempty':result['fixed_support_all_nonempty'],
                      'prior_selection_matches':sum(consistency),'budget':result['budget']},ensure_ascii=False))


def write_report(r):
    groups=r['groups'];penalty=groups['penalty'];allrows=r['queries']
    checkpoint_rows='\n'.join(f"| {v['seed']} | {v['arm']} | {v['queries']} | {v['empty_common_gate_pool']} | {v['primary_actual_tube_violation_with_feasible']} | {v['corrected_actual_common_gate_violations']} |" for v in r['by_checkpoint'])
    tick_rows='\n'.join(f"| {v['tick']} | {v['family']} | {v['queries']} | {v['empty_common_gate_pool']} | {v['primary_actual_tube_violation_with_feasible']} |" for v in r['by_tick'])
    costs=penalty['costs_on_same_feasible_query_subset']
    text=f"""# 已存候选池中的选择器遗漏与正确共同gate重选

日期：2026-10-17。[复算脚本](../../scripts/audit_stage_selector.py)、[全部768查询证据](selector-audit.json)。本轮只读48个旧NPZ及已存成本/margin，**没有生成命令候选、调用模型或环境、训练或修改权重**。

## 1. 独立复核与明确分层

- support的384查询全部有共同gate可行候选，包括全部3个fixed_support checkpoint的192查询；没有空pool。
- penalty的384查询中140个空pool（joint_penalty68/192、fixed_penalty72/192）。其余244个已有可行候选，但reference-only主gate的actual-cost赢家有163个tube超界；旧reference-cost赢家为156个。
- 本脚本从NPZ独立重建掩码与argmin，768查询的原primary索引、secondary是否可行以及可行时两个secondary索引全部与旧JSON一致。

**163个问题首先是选择器未要求执行已知门槛，不是固定反馈能力不足的证据。140个空pool只证明这些有限保存候选未满足条件，不证明连续命令规划无解，更不证明物理不可达。**

## 2. gate完全沿用，未放宽门槛

共同筛选是`reference_margin≤0`、`actual_domain_margin≤1e−7`、`input_margin≤0`、`tube_margin≤1e−8`，另要求两类成本有限。两个容差来自此前原native violation标志，不是本次调整。

support空集、fallback及一步/最坏耗散excess仍作诊断，没有偷偷加入或删除新的条件。因此这里只称“既有名义共同gate”，不是全扰动鲁棒证书，也不是已经定义好的长程机器人阶段入口合同。空pool的corrected selection为null，没有把index0或最小违反候选伪装成成功fallback。

在同一gate内分别最小化actual成本和reference成本，所有候选及预测保持相同。正确actual选择对声明gate的违反数为0；这由筛选定义保证，不是学习或新控制算法效果。

| seed | checkpoint arm | 查询 | 空pool | 有可行候选却选tube违反 | 正确actual选择违反 |
|---:|---|---:|---:|---:|---:|
{checkpoint_rows}

## 3. 遵守既有条件的任务成本代价

仅在**相同244个已有可行候选的penalty查询**上比较，不把140个失败删除后冒充全体表现：

- reference-only gate的actual赢家平均actual cost：`{costs['primary_actual']['mean']:.12g}`。
- 正确共同gate的actual赢家平均actual cost：`{costs['corrected_actual']['mean']:.12g}`。
- 同一共同gate的reference赢家、按actual模型评价的平均cost：`{costs['corrected_reference_evaluated_actual']['mean']:.12g}`。
- 正确actual相对原primary的配对平均成本增加：`{costs['corrected_minus_primary_actual']['mean']:.12g}`；均值相对变化`{costs['relative_change_of_subset_means_percent']:.6g}%`。

这是满足已声明条件的有限池代价取舍；不把约束纠正称为无代价收益。corrected actual成本不高于corrected reference的actual成本，是同池argmin定义的结果。没有执行新计划，因此不能报告新的闭环成功率、恢复能力或完成时间。

正确actual选择后，仍有`{penalty['corrected_actual_support_empty_diagnostic']}`个可行penalty查询出现support-empty诊断、`{penalty['corrected_actual_one_step_excess_diagnostic']}`个出现一步excess诊断；这些不属于当前四项gate，必须保留而非改称完整鲁棒可行。

## 4. 所有失败的结构，不选代表checkpoint

按全部固定tick/组别分解：

| tick | family | 查询 | 空pool | 有可行候选却选tube违反 |
|---:|---|---:|---:|---:|
{tick_rows}

140个空pool的首个失效gate统计为`{json.dumps(penalty['failure_cause_counts'],ensure_ascii=False)}`。JSON为每个query保留finite→reference→actual-domain→input→tube的候选数，和已通过前四层候选中的最小tube margin；后者只定位有限池缺口，不作为有效计划返回。

另只读取原保存metric的V与tube标志，对规划起点分层：140个空pool中**105个进入规划时已经tube外，35个仍在tube内**。tick0没有空pool。另有**2个初始tube外却有可行候选**的查询，因此初始违反不能直接推出不可恢复。tick>0使用已存poststep`V[tick−1]`及其原tube标志；tick0因为x=z而误差V=0，并非物理x=0。没有新状态预测。

这揭示了合同语义需要分开：当前筛选要求每个未来prefix立即满足旧tube；长程任务中的恢复阶段是否允许一段过渡时间，应在完整stage合同中预先定义。不能因初始超界就自动放宽门槛，也不能把全部105个例子直接归为固定反馈的物理极限。

所有失败ID都公开列出，不按结果挑某个checkpoint。接下来若做连续命令或更充分规划，应在预声明相同合同下使用这些失败全集或明确抽样规则；只能由新增规划/可达性证据进一步区分预算不足与固定反馈限制。

## 5. 资源和本轮结论

读取{r['budget']['source_pool_files']}个旧pool文件、{r['budget']['source_old_metric_files']}个旧metric文件、{r['budget']['fixed_queries']}查询、{r['budget']['saved_candidate_records_inspected']}条已保存候选记录；新增candidate、模型转移、环境转移、actor调用、训练、checkpoint读写均为0。pool NPZ hash与原记录核对，旧metric来源hash另列，未改任何原数组。

本轮可排除“所有违反都来自固定反馈无能”的解释：已知可行候选存在时，正确选择能直接修复163个penalty tube误选。剩余140个空pool仍是有限搜索失败。此时尚无证据把它们命名为必须由RL创造的新能力，也没有把旧servo门槛包装为狗/臂长程任务验证。
"""
    (OUT/'selector-audit.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
