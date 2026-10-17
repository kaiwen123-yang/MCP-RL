# 固定反馈下的连续命令可行见证

日期：2026-10-17。状态`completed`。[事前协议](continuous-protocol.md)、[逐query证据](continuous-results.json)、[源码](../../scripts/audit_continuous_command_feasibility.py)。

全42个名义一步控制权界尚未解释的旧空pool查询固定不变，已完成42个，独立核验可行26个，冻结预算内UNKNOWN 16个，待完成0个。未完成不能写成失败或成功。

仅优化原8×2个宏命令坐标，actor/design不变。目标是降低原tube margin寻找可行点，另三项原gate保持；这不是新RL奖励。固定三starts为旧pre-tube可行候选中的min-tube、旧actual主赢家、zero；SLSQP maxiter100/ftol1e−10，不因结果改设置。成功由原始ResponseServo方法对全部80拍及原cost/四gate独立核验；找到即停止后续starts，不追求task cost最优。

| 固定query | 结论 | 已尝试starts | forward预测调用 | query墙钟秒 |
|---|---|---:|---:|---:|
| 17/joint_penalty/60/10 | FEASIBLE_WITNESS | 1 | 3 | 1.38 |
| 17/joint_penalty/60/12 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 33 | 8.62 |
| 17/joint_penalty/60/15 | FEASIBLE_WITNESS | 1 | 11 | 2.70 |
| 17/joint_penalty/120/2 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 37 | 10.06 |
| 17/joint_penalty/120/7 | FEASIBLE_WITNESS | 3 | 21 | 5.63 |
| 17/joint_penalty/120/8 | FEASIBLE_WITNESS | 1 | 6 | 1.27 |
| 17/joint_penalty/120/9 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 184 | 35.96 |
| 17/joint_penalty/180/11 | FEASIBLE_WITNESS | 3 | 17 | 4.62 |
| 17/joint_penalty/180/12 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 94 | 18.81 |
| 17/joint_penalty/180/13 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 28 | 7.44 |
| 17/fixed_penalty/60/0 | FEASIBLE_WITNESS | 1 | 2 | 0.52 |
| 17/fixed_penalty/60/15 | FEASIBLE_WITNESS | 1 | 2 | 0.51 |
| 17/fixed_penalty/120/7 | FEASIBLE_WITNESS | 1 | 7 | 1.45 |
| 17/fixed_penalty/120/13 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 13 | 3.35 |
| 17/fixed_penalty/180/8 | FEASIBLE_WITNESS | 1 | 2 | 0.58 |
| 17/fixed_penalty/180/12 | FEASIBLE_WITNESS | 3 | 17 | 4.42 |
| 29/joint_penalty/120/2 | FEASIBLE_WITNESS | 2 | 14 | 2.80 |
| 29/joint_penalty/120/7 | FEASIBLE_WITNESS | 1 | 2 | 0.57 |
| 29/joint_penalty/120/13 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 21 | 5.06 |
| 29/joint_penalty/180/1 | FEASIBLE_WITNESS | 1 | 3 | 0.87 |
| 29/joint_penalty/180/8 | FEASIBLE_WITNESS | 1 | 20 | 3.82 |
| 29/joint_penalty/180/10 | FEASIBLE_WITNESS | 1 | 5 | 1.13 |
| 29/fixed_penalty/60/1 | FEASIBLE_WITNESS | 1 | 2 | 0.50 |
| 29/fixed_penalty/60/2 | FEASIBLE_WITNESS | 1 | 12 | 2.48 |
| 29/fixed_penalty/60/4 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 313 | 58.08 |
| 29/fixed_penalty/120/2 | FEASIBLE_WITNESS | 1 | 11 | 2.27 |
| 29/fixed_penalty/180/8 | FEASIBLE_WITNESS | 1 | 9 | 1.87 |
| 29/fixed_penalty/180/10 | FEASIBLE_WITNESS | 1 | 2 | 0.51 |
| 43/joint_penalty/60/7 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 769 | 135.64 |
| 43/joint_penalty/60/15 | FEASIBLE_WITNESS | 1 | 2 | 0.51 |
| 43/joint_penalty/120/3 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 709 | 137.67 |
| 43/joint_penalty/120/5 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 226 | 46.31 |
| 43/joint_penalty/120/9 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 20 | 6.11 |
| 43/joint_penalty/180/0 | FEASIBLE_WITNESS | 1 | 2 | 0.57 |
| 43/joint_penalty/180/11 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 226 | 44.49 |
| 43/fixed_penalty/60/4 | FEASIBLE_WITNESS | 1 | 2 | 0.55 |
| 43/fixed_penalty/120/5 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 27 | 7.14 |
| 43/fixed_penalty/120/9 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 22 | 5.66 |
| 43/fixed_penalty/120/13 | FEASIBLE_WITNESS | 1 | 13 | 2.73 |
| 43/fixed_penalty/180/0 | FEASIBLE_WITNESS | 1 | 4 | 0.81 |
| 43/fixed_penalty/180/11 | UNKNOWN_WITHIN_FROZEN_BUDGET | 3 | 23 | 5.98 |
| 43/fixed_penalty/180/12 | FEASIBLE_WITNESS | 1 | 2 | 0.51 |

成功只说明相应旧34候选pool漏掉了连续命令可行方案，否定“该pool失败已证明固定反馈无能”。UNKNOWN保留局部优化预算、分段非光滑和控制可达性的不确定，不能当全局不可达。旧140个失败中已由一步界解释的98个不在这42项优化内；原分母保持，不只汇报成功。

当前累计forward模型转移235040、4输出反向step/RHS等效计数314240、独立核验模型转移2080。4workers×torch1；总墙钟173.64s，query进程CPU合计574.41s。连续优化预算大于旧有限pool，不宣称同预算优势。环境step/训练/actor更新/checkpoint写入均0。

首次尝试因NumPy bool_报告序列化错误中止，科学设置未改。其额外CPU与模型计数下界见[调试账本](continuous-debug.json)，在途精确计数不完整；不能把本次修复后成功运行的预算当成本轮全部开销。该错误不作为科学UNKNOWN结果。

这仍是旧名义tube合同的分析，不是完整狗/臂阶段入口验证；成功没有说明策略学出新能力，失败也没有证明学习不可替代。
