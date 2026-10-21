# Go2 公平条件对照：保存结果的分析

本目录只分析已保存的22条评价，不启动机器人仿真、不拟合模型、不训练策略。每格一次确定性运行；三个条件不是独立重复，不进行显著性检验。

| suite | case | speed | controller/objective | predictor | success | elapsed/s | path RMS/m | work proxy/J |
|---|---|---|---|---|---|---:|---:|---:|
| primary | nominal | standard | follower | nominal | True | 18.04 | 0.0690 | 749.0 |
| primary | nominal | standard | tracking | nominal | True | 17.86 | 0.0544 | 769.4 |
| primary | nominal | standard | mission | nominal | True | 17.70 | 0.0807 | 785.6 |
| primary | narrow | standard | follower | nominal | True | 17.66 | 0.0709 | 747.3 |
| primary | narrow | standard | tracking | nominal | True | 17.42 | 0.0609 | 736.9 |
| primary | narrow | standard | mission | nominal | True | 17.04 | 0.0886 | 740.5 |
| primary | wide | standard | follower | nominal | True | 17.62 | 0.0763 | 746.8 |
| primary | wide | standard | tracking | nominal | True | 18.26 | 0.0617 | 766.8 |
| primary | wide | standard | mission | nominal | True | 17.54 | 0.0879 | 763.4 |
| primary | nominal | fast | follower | nominal | True | 11.30 | 0.0927 | 736.2 |
| primary | nominal | fast | tracking | nominal | True | 11.66 | 0.0871 | 745.4 |
| primary | nominal | fast | mission | nominal | True | 11.48 | 0.0964 | 742.3 |
| primary | narrow | fast | follower | nominal | True | 11.44 | 0.0930 | 723.2 |
| primary | narrow | fast | tracking | nominal | True | 11.50 | 0.0886 | 724.8 |
| primary | narrow | fast | mission | nominal | True | 11.50 | 0.0867 | 740.8 |
| primary | wide | fast | follower | nominal | True | 11.48 | 0.0940 | 729.2 |
| primary | wide | fast | tracking | nominal | True | 11.64 | 0.0871 | 724.8 |
| primary | wide | fast | mission | nominal | True | 11.42 | 0.0952 | 726.4 |
| bank | nominal | fast | tracking | nominal | True | 11.48 | 0.0828 | 736.3 |
| bank | nominal | fast | tracking | physics | True | 12.30 | 0.0787 | 758.1 |
| bank | nominal | fast | mission | nominal | True | 11.28 | 0.0870 | 725.7 |
| bank | nominal | fast | mission | physics | True | 11.68 | 0.0816 | 758.6 |

## 本轮条件内读数

- standard 下目标tracking→mission的完整时间变化，按nominal/narrow/wide顺序为 -0.16/-0.38/-0.72 s。负号代表mission更快。路径与功代价另列，不能只看时间宣布全面优势。
- fast 下目标tracking→mission的完整时间变化，按nominal/narrow/wide顺序为 -0.18/+0.00/-0.22 s。负号代表mission更快。路径与功代价另列，不能只看时间宣布全面优势。
- 候选库tracking：nominal→physics时间 11.48→12.30 s；路径RMS 0.08278→0.07871 m；机械功代理 736.29→758.09 J。physics额外 1,488,000 物理步/74,400 策略推理。
- 候选库mission：nominal→physics时间 11.28→11.68 s；路径RMS 0.08697→0.08164 m；机械功代理 725.75→758.63 J。physics额外 1,416,000 物理步/70,800 策略推理。
- 历史到本轮同时改变了多项控制条件；用历史旧MPC大幅慢于旧follower的差距直接证明模型瓶颈是不成立的。当前物理分叉能复现选中首步，并不保证内部代理目标对完整时间的排序正确。

## 解释边界与文件

- `all_22_metrics.csv/json` 保留全部成功、失败、超时和执行异常。失败的elapsed是已运行时间，不是完成时间；`completion_time_s`为空，图中用×标识。任务未完成的时间对比为UNKNOWN。
- `stage_times_and_exits.csv` 保存阶段时间和瞬时出口状态；未完成阶段明确列UNCOMPLETED。停止仍是平面位置、速度和朝向误差保持阈值1s，不代表全身完全静止。
- `primary_paired_contrasts.csv`：同speed同case的tracking→mission目标表达对照、follower→mission对照和同mode跨speed条件对照。差值是right−left，不作总体统计推断。
- `historical_condition_changes.csv`：与旧六条结果的描述性比较。命令模长governor及跟随器切向前视增益共同改变，不能计算唯一的独立因果贡献百分比。
- `bank_2x2_effects_and_interaction.csv`：固定候选构造下objective×predictor对完整任务的条件效应。interaction=(mission physics−mission nominal)−(tracking physics−tracking nominal)。它不是跨objective内部损失的差值，也不是统计ANOVA结论。
- `candidate_scores_and_ranks.csv`：每快照六候选分别在tracking/mission下的名义分数与已测物理分数、可行性与排名。不同objective的score量纲与权重不同，不能直接比较数值大小。
- `counterfactual_ranking_and_regret.csv`/`counterfactual_ranking_summary.csv`：仅在physics组访问的实际状态上计算同objective、同快照、同候选库regret。nominal组未测物理分叉留空；名义选择物理不可行时只报告可行性错误，不把标量差称可行regret。
- 各物理分叉共享raw序列与governor规则；governor根据各预测/物理状态限速，因此名义与物理的最终applied命令不保证逐值相同。不同组状态随闭环分歧，所以后续候选数值也会改变；不是跨组始终共享完全相同的序列。
- 精确physics拥有完整模拟器/策略历史特权且追加大量物理步；预算、推理次数与时延单列。它同时改变轨迹预测和可行性判断（名义身体轮廓 versus 实际全机器人接触/跌倒），本轮没有再把这两项拆开归因。可能排序改变却不缩短完整任务，须按完整结果判断。即使有收益也不称PINN收益或等算力在线优越性。
- 所有候选物理不可行时，固定实现退回最低代理score并只执行首0.2s，随后重规划。完整4s分叉不可行不意味着实际下一0.2s已经失败；全程成功也不能反推所有候选预测安全或已有硬安全保证。
- `data_and_contract_audit.json` 验证原TaskSpec/实体场景/hash、冻结权重、实际和分叉计数、评分重算、选中物理分叉首步与真实下一状态、命令边界模长约束，以及所有输入列/JSON字段有限值。
- 输入文件hash同时保留`sha256`（本地原始文件字节）和`sha256_lf`（先将CRLF规范化为LF）。Git可能规范化文本换行；公开仓库跨机器核验JSON/CSV时使用`sha256_lf`，本地原始字节存档使用`sha256`。运行源码与分析脚本hash同样采用LF规范化口径。
- CEM计数只含候选评分。primary MPC每次另40名义步，bank每次另100名义步加单列120 bank步；runner日志预测每次另1名义步。实际求解时间包含控制器内全部计算。
- 物理时钟在求解时暂停。低于0.2s只说明本机计算耗时；physics超过周期同样如实报告，不冒称真实墙钟部署。机械功代理不是电池电能。

图1为18条同条件配对点与路径/功取舍；灰线仅连接同一个case的三种控制器，不连接不同case。图2为固定候选库2×2结果及特权计算，规划时延使用明确log轴。图3为nominal-fast三种模式速度与偏航连续轨迹，S/T/D/Stop为直行/转弯/过门/停止，灰色不是置信区间。全部图以单条原始运行表示，不画均值柱或显著性。

```powershell
work/venv/Scripts/python.exe scripts/analyze_go2_fair.py
```

可选`--skill-scripts`启用本地SciPilot布局审计；数据和成图不依赖该路径。
