# Go2 公平对照证据索引

协议及运行源码预先固定于 `6481466`；18 条主对照 + 4 条模型诊断均完成，没有失败、超时、重跑或按结果调参。旧基线六条结果与源码原样保留。本次不是训练实验。

- [结论、完整表格与研究边界](../../docs/go2-fair-comparison-2026-10-19.md)
- [运行前协议](PROTOCOL.md)与[机器可读配置](../../configs/go2_fair_comparison.json)
- [逐次指标、配对效应、跨阶段状态与图](analysis/README.md)
- [数据与冻结条件核验](analysis/data_and_contract_audit.json)
- [同快照恢复核验](snapshot-check.json)
- [本地保留的66份原始文件校验清单](local-raw-manifest.json)
- [交叉复核](independent-review.md)
- [复现命令和数据频率](../../src/go2_frozen/README.md)
- [保留的旧六条结果](../go2-frozen-2026-10-19/INDEX.md)

## 全部运行

每个目录均含 `summary.json`、`trajectory_10hz.csv`、`decisions.json`、`stage_transitions.json`、`prediction_errors.csv`。名称按实际因素编码；不是随机种子重复。

| 条件 | 跟随器 | 跟踪 MPC | 任务 MPC |
|---|---|---|---|
| standard nominal | [结果](runs/primary_standard_follower_nominal/summary.json) | [结果](runs/primary_standard_tracking_nominal/summary.json) | [结果](runs/primary_standard_mission_nominal/summary.json) |
| standard narrow | [结果](runs/primary_standard_follower_narrow/summary.json) | [结果](runs/primary_standard_tracking_narrow/summary.json) | [结果](runs/primary_standard_mission_narrow/summary.json) |
| standard wide | [结果](runs/primary_standard_follower_wide/summary.json) | [结果](runs/primary_standard_tracking_wide/summary.json) | [结果](runs/primary_standard_mission_wide/summary.json) |
| fast nominal | [结果](runs/primary_fast_follower_nominal/summary.json) | [结果](runs/primary_fast_tracking_nominal/summary.json) | [结果](runs/primary_fast_mission_nominal/summary.json) |
| fast narrow | [结果](runs/primary_fast_follower_narrow/summary.json) | [结果](runs/primary_fast_tracking_narrow/summary.json) | [结果](runs/primary_fast_mission_narrow/summary.json) |
| fast wide | [结果](runs/primary_fast_follower_wide/summary.json) | [结果](runs/primary_fast_tracking_wide/summary.json) | [结果](runs/primary_fast_mission_wide/summary.json) |

| 原 nominal 场景、fast 档、有限六候选 | 名义预测排序 | 物理分叉排序 |
|---|---|---|
| tracking | [结果](runs/bank_fast_tracking_nominal_nominal/summary.json) | [结果](runs/bank_fast_tracking_physics_nominal/summary.json) |
| mission | [结果](runs/bank_fast_mission_nominal_nominal/summary.json) | [结果](runs/bank_fast_mission_physics_nominal/summary.json) |

## 保存范围

全部实际 50 Hz 身体、关节、接触与动作记录、场景文件及准备记录留在工作树 `work/go2-fair-runs/<相同运行名>/`，未删除、未公开原始权重。公开轨迹降采样为 10 Hz，但全部高层决策、阶段出口与评价汇总保留。physics 决策另包含每个候选每 0.2 s 的六维边界状态、raw/applied 命令、两目标分数和可行性；没有保存每条分叉完整 50 Hz 关节轨迹或二进制快照。

物理预测诊断拥有完整状态和额外计算，不能作为等算力在线强基线；所有可行性错误与完整任务结果一起保留。主对照没有新增场景或外部扰动。没有新训练、PINN、硬件操作或 PR 合并。
