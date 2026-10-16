# 近期高水平控制与学习文献专题

**选文范围更正：**用户指出本批没有充分覆盖2026，这是成立的筛选缺口。该批适合作为近五年理论基础专题，不能据此称2026最新控制方向已被系统比较。后续新核到的2026条目列于[独立候选记录](../../literature/supplements/control-2026-screening-2026-10-16.json)，其状态仅为出版与方法定位核验，未并入下面的已读数量。

日期2026-10-16。本轮依用户要求选读10篇，九篇正式卷期为2022—2025，另有一篇已录用、分配2026年11月未来卷期但精确online日期待核的前沿作者稿。覆盖Nature、Science Robotics、IEEE TAC、Automatica和IEEE T-RO。期刊选择依用户点名与主题相关性；2025中科院大类一区/Top没有取得官方复核，不能以JCR Q1替代。主100篇v2库及MR编号不变。

实际阅读为指定PDF共200页正文及内嵌方法/附录，关键公式图表另核原页；包含对以前局部读过工作的补齐，不称10篇此前未知论文。下载/逐页通读/证明独立核验/复现分别记录，未把全文通读称所有定理认证。[完整版本与读取清单](reading-manifest.json)。

| 编号 | 论文与正式出处 | 指定稿页数 | 这轮最有用的思想 |
|---|---|---:|---|
| EXT-ACT-T01 | [Robust adaptive MPC using control contraction metrics](theory-T01-ccm-rampc.md)，Automatica 2023 | 15 | 状态与输入相关的误差管束，反馈和模型误差一起进入收紧 |
| EXT-ACT-T02 | [Stabilizing reinforcement learning control: A modular framework for optimizing over all stable behavior](theory-T02-stable-behavior.md)，Automatica 2024 | 31 | 限制可学习行为族，优化器可以保留；不能把TD3确定策略结论直接转给随机PPO |
| EXT-ACT-T03 | [Recurrent Equilibrium Networks](theory-T03-ren.md)，TAC 2024 | 16 | 稳定参数化仍须加实际系统correctness；稳定模型不等于正确模型 |
| EXT-ACT-T04 | [Synthesizing neural network controllers with closed-loop dissipativity guarantees](theory-T04-dissipative-ppo.md)，Automatica已录用，未来Nov2026卷期 | 17 | 已有PPO＋耗散硬检查/投影；饱和和离散化仍是需核前提 |
| EXT-ACF01 | [Champion-level drone racing using deep reinforcement learning](frontier-01-swift.md)，Nature 2023 | 14 | 任务目标与系统辨识/低层执行动态的连接；不用轨迹拟合代替真实任务 |
| EXT-ACF02 | [DTC: Deep Tracking Control](frontier-02-dtc.md)，Science Robotics 2024 | 18 | 参考不断迁就失败策略会造成lazy解；更新频率与规划成本要一起研究 |
| EXT-ACF03 | [Learning robust perceptive locomotion for quadrupedal robots in the wild](frontier-03-perceptive-locomotion.md)，Science Robotics 2022 | 22 | 记忆和接触反馈补足状态；长时实测不等于长期规划保证 |
| EXT-ACF04 | [Safe Control With Learned Certificates](frontier-04-neural-certificates.md)，T-RO 2023综述 | 20 | 动态可行参考与误差收敛须分别成立；联合学策略/证书不是新概念 |
| EXT-ACR01 | [Integration of Adaptive Control and Reinforcement Learning for Real-Time Control and Learning](root-01-adaptive-rl.md)，TAC 2023 | 39 | 稳定实现参考与真实任务最优可能冲突；作者明确提出参考设计缺口 |
| EXT-ACR02 | [Necessary and Sufficient Conditions for Data-Driven Model Reference Control](root-02-informative-reference.md)，TAC 2025 | 8 | 数据能否支持参考要求，不等于能否唯一辨识；近似匹配另需稳定条件 |

## 对固定研究主线的收敛判断

共同设计闭环响应与实际执行策略仍有研究意义，但“PPO＋控制证书”“稳定网络”“模型参考控制＋学习”“规划提供训练目标”均有直接先例。最值得继承的关系是：**零误差时参考真实可实现，偏差出现时能受反馈控制，输入资源支持这些纠偏，并且这种关系在能力接续和策略变化时仍有效。**

本轮将先用已知低维离散系统把这条关系算清楚，再用成熟PPO检验任务驱动响应更新是否比固定响应分布更好。耗散输入支持映射可作为保证不被PPO任意更新破坏的结构基线，但其可行集合目前不依赖策略权重，因此不能称为学习扩张能力集合。针对这一点已有[直接近邻与机制审查](../ppo-mechanism-2026-10-16/mechanism-neighbor-audit.md)。

长程与并行启发要分别落实：共同度量或切换接续关系服务误差传播；低维响应和宏周期规划服务计算；成熟PPO服务可重复学习。它们各自是已有工具，只有完整关系带来公平任务收益，才可讨论方法增量。[理论组综合](theory-synthesis.md)，[机器人机制组综合](frontier-synthesis.md)。
