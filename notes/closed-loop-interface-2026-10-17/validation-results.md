# 实际闭环接口核验与固定actor重规划评估

日期：2026-10-17。状态`completed`。[协议](protocol.md)、[逐项数值](validation-results.json)、[实现说明](interface-implementation.md)、[脚本](../../scripts/validate_closed_loop_interface.py)。全部12组identity通过：`True`。

一步identity使用全部256×16个旧评估原生锚点；80步timeline使用原最初80个命令和明确标注的历史扰动，仅用于核验，不给新planner未来扰动预览。参考z连续，实际任务/原total与全部原metric分开核对。一次映射门槛1e−6、80步门槛1e−5先冻结。

| seed | arm | 一步最大误差 | 80步最大误差 | 通过 |
|---:|---|---:|---:|---|
| 17 | joint_support | 0 | 0 | True |
| 17 | fixed_support | 0 | 0 | True |
| 17 | joint_penalty | 0 | 0 | True |
| 17 | fixed_penalty | 0 | 0 | True |
| 29 | joint_support | 0 | 0 | True |
| 29 | fixed_support | 0 | 0 | True |
| 29 | joint_penalty | 0 | 0 | True |
| 29 | fixed_penalty | 0 | 0 | True |
| 43 | joint_support | 0 | 0 | True |
| 43 | fixed_support | 0 | 0 | True |
| 43 | joint_penalty | 0 | 0 | True |
| 43 | fixed_penalty | 0 | 0 | True |

只有全部identity通过后才运行新adapter。全12组原first-short的seed/actor/design/初态/任务/实际扰动条件不变；新planner读当前x/z，名义预测w=0，但supportW仍为.04。它是**新的反事实仿真**，不是旧记录重放。主可行门槛仍为原reference域，实际域/输入/tube/support均保留诊断。

| seed | arm | 旧planner任务cost | 新actual planner任务cost | 相对变化 | 新actual域违反率 | 新tube违反率 | 新仿真墙钟秒 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 17 | joint_support | 0.01508273557 | 0.01485097967 | -1.5366% | 0 | 0 | 10.22 |
| 17 | fixed_support | 0.01512715407 | 0.0148641374 | -1.7387% | 0 | 0 | 10.65 |
| 17 | joint_penalty | 0.01488621905 | 0.01486470085 | -0.14455% | 0 | 0.27026 | 10.50 |
| 17 | fixed_penalty | 0.01502486132 | 0.01489890739 | -0.8383% | 0 | 0.1958 | 10.22 |
| 29 | joint_support | 0.01646774635 | 0.01646204107 | -0.034645% | 0 | 0 | 10.50 |
| 29 | fixed_support | 0.01653545909 | 0.01643782295 | -0.59047% | 0 | 0 | 10.81 |
| 29 | joint_penalty | 0.01752008684 | 0.01672627777 | -4.5309% | 0 | 0.55615 | 11.06 |
| 29 | fixed_penalty | 0.01799480803 | 0.01678631268 | -6.7158% | 0 | 0.55981 | 10.69 |
| 43 | joint_support | 0.01603035815 | 0.01585896313 | -1.0692% | 0 | 0 | 10.65 |
| 43 | fixed_support | 0.01600213721 | 0.01595135964 | -0.31732% | 0 | 0 | 10.20 |
| 43 | joint_penalty | 0.01643930748 | 0.01609471999 | -2.0961% | 0 | 0.30151 | 10.55 |
| 43 | fixed_penalty | 0.01659421995 | 0.01633714326 | -1.5492% | 0 | 0.44385 | 10.29 |

已完成12组的简单平均task cost为0.016142091093→0.0158444471502（-1.8439%）。这只描述固定评估集合的规划变化，不是学习收益或统计泛化结论。tube违反率上升的case为：17/joint_penalty：0.24291992→0.27026367；29/joint_penalty：0.55493164→0.55615234；43/joint_penalty：0.22973633→0.30151367；43/fixed_penalty：0.33081055→0.44384766。其他约束及空集必须一并查看，不能将成本下降写成全面改善。

计数按已完成case累计，完整账本见JSON：identity离线模型步`64512`；新执行仿真转移`49152`；候选预测微步`13578240`。预测每个候选微步现在包含真实网络与support/fallback计算，同候选数不等于同计算成本。

所有原checkpoint与生产源码哈希单列；没有训练、参数更新、梯度优化器步或checkpoint写入。任务变化只归于相同冻结actor下的预测/规划改变，不称RL学出新能力，也不证明优于同信息同目标直接优化。逐组原/new全部指标与selected-plan诊断均保留在JSON。实际case墙钟合计`126.337s`；这是本轮运行记录，可能与其它分析并行，不作独占实时率或公平加速比benchmark。
