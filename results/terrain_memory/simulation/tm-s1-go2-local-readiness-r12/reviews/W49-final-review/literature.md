# W50一手文献：滚动参考、可行支撑和反馈更新

2026-11-15核对。本文区分论文事实与针对S1的研究推断；没有复现这些代码，也不将论文性能、定理或期刊分区直接当本项目保证。四篇相对于本轮检索的已有文献清单是新增主参考；不声称团队从未阅读过。

## 1. 最直接的证据：上一解要随当前问题推进

| 论文 | 已核一手事实 | 对当前S1的启发（本项目推断） |
|---|---|---|
| **BiConMP**，T-RO 39(2):905–922，2023 | §III-E基于当前实际状态重规划全身轨迹，接触计划随时间覆盖[t,t+T]；并研究horizon和频率影响。 | 同问题内冻结一个候选便于比较，跨问题永久冻结reference/K/H则是另一件事。应检查真实参考的起止时刻与来源。 |
| **CAFE-MPC**，T-RO 41:837–856，2025 | §VI-C把上一MPC解前移并追加新变量；遇新冲击阶段时，直接复制旧末状态可能产生大动力学缺陷，改用roll-out初始化。还处理policy lag。 | 优先修实际滚动时域与反馈生命周期；不能只把generation_time改新。不能照搬其整套控制器或把论文求解时间限额变成本项目后台截止。 |
| **Bilevel Optimization for Real-Time Control with Application to Locomotion Gait Generation**，CDC2024 | §III-A将触地/离地时刻作为受约束的上层变量；过去时刻退出优化，未来时刻随horizon补入，并限制正在摆动阶段的瞬间改变。 | 若正常支撑推进确实需要时序权限，scalar cap不能自动提供；先证实缺哪种权限，再作一个确定性对照，不直接扩PPO。 |
| **Layered Nonlinear MPC for Robust Stabilization of Hybrid Systems**，ACC2025 | 将混合模式计划、较快固定模式MPC和低层跟踪分开；理论依赖可行性/终端等条件，并讨论重规划速度与跟踪误差的关系。 | 有50Hz/500Hz层次时钟不等于稳定性证明。要同时确认参考可执行、相位正确、足部承重成立，并辨别预测和实际执行。 |

### 主文献、发表信息与代码

- BiConMP：[正文v2，§III-E](https://arxiv.org/html/2201.07601v2#S3.SS5)、[作者机构发表记录](https://portal.fis.tum.de/en/publications/biconmp-a-nonlinear-model-predictive-control-framework-for-whole-/)、[DOI](https://doi.org/10.1109/TRO.2022.3228390)、[官方实现](https://github.com/machines-in-motion/biconvex_mpc)。官方实现依赖Pinocchio/Crocoddyl；本轮未安装或运行。
- CAFE-MPC：[正文v3，§VI-C](https://arxiv.org/html/2403.03995v3#S6.SS3)、[ROAM正式发表目录](https://roamlab.nd.edu/publications/)、[DOI](https://doi.org/10.1109/TRO.2024.3504132)、[官方实现](https://github.com/ROAM-Lab-ND/CAFE-MPC)。主仓库构建后仍需要相应仿真和全身控制器，并非可直接替换本项目的单一程序。
- Bilevel：[正文v1，§III-A](https://arxiv.org/html/2409.12366v1#S3.SS1)、[作者CDC2024页面](https://www.zacholkin.com/research/bilevel-gait-gen/)、[作者实现](https://github.com/Zolkin1/bilevel-gait-gen)。本轮只核文献与公开实现入口，不声称复现。
- Layered MPC：[正文v1](https://arxiv.org/html/2503.12810v1)、[作者ACC2025页面](https://www.zacholkin.com/research/layered_mpc_hybrid_robustness/)、[示例代码](https://github.com/Zolkin1/hybrid-mpc)。作者明确代码为重建论文示例，不是通用库；不能当现成Go2/TM02稳定控制器。

## 2. 从论文到W50，要保留的推断边界

论文支持维护当前状态、时域、未来接触与可执行参考；**没有哪篇证明W50同alpha的raw-center/反馈/H混合公式必然成功**。该方案是本项目为区分旧N整包换反馈与W49永久冻结而提出的有限假说，必须经原native实际评分、真实接口和原任务判断。

用户补充的视频显示长期低伏挪动。结合本仓库状态，可优先检查Stand→outbound时的实际gait相位、reference是否仍来自初始站立、足部承重是否完成交换。仅增加生存时长或挪动距离，没有回答正常步态问题；这项判断来自本项目视频/日志，而非借用外部论文结论。

当前.35s horizon在2Hz下是.7周期。这只提示可能需要检查下一次承重和末端继续性，不证明不足一个周期就必然失稳。若前进候选已可行且原J更优却找不到，先改搜索/初始化；若可行候选被原目标持续排后，再研究单项目标/参考；若接触交换本身做不到，再研究权限或终端延续。不同horizon不能直接拿未统一口径的总J作优劣证据。

W50先处理生命周期，不同时移植BiConMP、CAFE-MPC、分层接触时序或新RL接口。先把正常承重推进与原完整任务做出来，再检验跨足/返程材料记忆的因果收益。这一顺序来自项目当前证据与可证伪性要求。
