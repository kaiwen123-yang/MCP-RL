# 既有Go2轨迹与检查点的离线归因

本轮回答“完整任务为何在终点变慢、KL为何几乎一步即停，以及这批负结果能否归罪于伴随机制”。不训练、不拟合、不生成新物理轨迹、不改场景或参数，PR保持未合并。

先读[综合科学结论](../../docs/go2-adjoint-diagnosis-2026-10-20.md)，原固定小试的结果仍保留在[原报告](../../docs/go2-adjoint-rerun-results-2026-10-20.md)。

| 证据链 | 输出 | 复算脚本 |
|---|---|---|
| 实际过门、位置入区、速度/朝向越界、保持重置及命令执行 | [终端摘要](terminal/summary.json)、[逐episode](terminal/episode_terminal_metrics.csv)、[逐次保持](terminal/hold_attempts_and_resets.csv)、[实际时序图](terminal/figures/terminal_hold_chain.png) | [保存轨迹分析](../../scripts/diagnose_go2_terminal_saved.py) |
| 原规划重现、同状态同候选池g/V替换、完成后评分 | [解释](planner/README.md)、[摘要](planner/summary.json)、[原目标分解](planner/terminal_score_decomposition.csv) | [只读神经规划重放](../../scripts/diagnose_go2_planner_saved.py) |
| 原B状态上前后策略的解析KL与实际停止逻辑 | [解释](kl/README.md)、[逐轮KL](kl/round_kl.csv)、[首轮同B比较](kl/paired_arms.csv) | [保存策略诊断](../../scripts/diagnose_go2_kl_saved.py) |
| 保存V与确定性实际后缀return | [摘要](value/summary.json)、[区域入口](value/region_entry_values.csv)、[逐episode](value/episodes.csv) | [保存值诊断](../../scripts/diagnose_go2_value_saved.py) |

原始批次、检查点、全部50Hz轨迹、服务器环境/进程记录仍在忽略目录`work/server-adjoint-20261020-r2/preserved/`，不公开私人环境状态。各诊断目录保存读入文件的相对路径和SHA；生产算法与原配置没有修改。

离线重放只用于确定旧组件如何计算预测、评分和命令。它没有提供换组件后的真实物理任务结果，不能据此报告新的成功率、完成时间改善或模块因果百分比。独立训练种子和重复开发条件的身份也保持不变。
