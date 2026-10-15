# 近期优先阅读路线

2026-10-13全文阅读后的专题入口：[研究候选与问题驱动阅读路线](research-mainline-2026-10-13.md)。下文保留原全局路线；实际已读范围见[逐篇覆盖](../notes/fulltext-2026-10-13/INDEX.md)，当前为100篇指定主版本文字通读，证明独立核验与复现尚未完成。

以2024—2026年正式论文为主要入口，先建立方法与理论认识，再开展通用仿真。经典在遇到具体理论问题时查阅，不要求先沿年代补完所有早期工作。

主清单的90篇近期论文按正式发表年份统计；最早已核公开年份和版本关系另行记录。阅读期刊扩展稿时，要识别其沿用的基础和新增部分。下面是**计划中的阅读顺序**，不是已完成精读的清单。

## 第一组：最新综述与问题地图

先读 **MR004 · Synthesis of model predictive control and reinforcement learning: Survey and classification（2026）**，重点看分类、方法接口与讨论，再用 **MR157 · A view on learning robust goal-conditioned value functions: Interplay between RL and MPC（2025）** 对照全局价值学习和局部规划的分工。

本组应能回答：MPC在训练和部署中分别做什么；模型、代价、价值与优化器学习怎样区分；理论保证针对哪一个闭环和哪一种误差。

## 第二组：模型、策略、价值与规划如何协同

| 顺序 | 文献 | 第一遍重点 |
|---|---|---|
| 1 | MR026 · TD-MPC2: Scalable, Robust World Models for Continuous Control（2024） | 学习模型、策略、价值与在线规划的完整循环 |
| 2 | MR122 · Bootstrapped Model Predictive Control，BMPC（2025） | 策略向MPC学习后怎样反向帮助规划，价值目标由谁产生 |
| 3 | MR119 · TD-M(PC)²: Improving Temporal Difference MPC Through Policy Constraint（2026） | 规划探索策略与价值评价策略不一致怎样影响估计与动作 |

**TD-MPC2与TD-M(PC)²是不同论文。** 阅读时保留题名和编号。需要理解早期设计时，再查经典 **MR022 · Temporal Difference Learning for Model Predictive Control（2022）**。

本组先画出“谁收集数据—谁产生目标—谁提出动作—谁最终执行”的流程。再核对模型或策略更新后，其他模块是否仍在评价同一种行为；不要只按最终平均回报判断机制。

## 第三组：学习优化过程、可微结构与实际计算

| 顺序 | 文献 | 第一遍重点 |
|---|---|---|
| 1 | MR077 · Deep Model Predictive Optimization（2024） | RL学习优化更新与移位；采样数量和实际计算时间的区别 |
| 2 | MR163 · DEQ-MPC: Deep Equilibrium Model Predictive Control（2025） | 网络与求解器联合固定点，梯度、热启动与迭代怎样相互影响 |
| 3 | MR087 · Differentiable Model Predictive Control on the GPU（2026） | 前向求解与反向梯度怎样利用问题结构及并行计算 |
| 4，可选 | MR154 · Policy Optimization for Unknown Systems using Differentiable MPC；MR168 · Computationally efficient Gauss–Newton reinforcement learning for model predictive control（均2026） | 错模型和策略更新成本分别怎样影响学习 |

本组应能区分“学习控制策略”“学习MPC参数”“学习求解更新”。遇到隐式微分和MPC层表示时，查经典 **MR081 · Differentiable MPC（2018）**；遇到采样规划原理时，查 **MR066 · Information-Theoretic MPC（2018）**。

## 第四组：终端价值、约束与安全条件

从 **MR029 · AC4MPC（2026正式卷期）** 区分actor初值和critic终端项的贡献，再读 **MR167 · Learning soft constrained MPC value functions（2024）** 与 **MR156 · A characterization method of terminal ingredients for nonlinear MPC using value-based reinforcement learning（2026）**，比较价值近似如何影响短时域性能、可行域与理论条件。

随后读 **MR159 · Safety Filtering While Training（2025）**、**MR170 · Dynamic Model Predictive Shielding（2024）**，理解安全机制参与训练后，学习目标、探索和最终策略行为怎样改变。可用 **MR162 · Stability Mechanisms for Predictive Safety Filters（2024）** 补充“安全”与“稳定”的区别。

需要回查时，使用经典MR005的MPC稳定性框架、MR008的参数化MPC与RL联系、MR054的预测安全过滤。经典提供理解条件的工具，不要求每读一篇新文献都先推导完整历史谱系。

## 每篇保留的最小阅读记录

| 问题 | 应记录内容 |
|---|---|
| 要解决什么 | 主要困难及原有方法边界 |
| 学什么、优化什么 | 参数接口、学习目标、优化变量与时域 |
| 训练与部署 | 数据来源、更新顺序、求解器角色及执行动作 |
| 理论条件 | 精确或有限求解、模型误差、价值误差、可行性与稳定性假设 |
| 证据到哪里 | 对照、消融以及实际数据和计算成本 |
| 版本与增量 | 最早已核公开时间，期刊扩展继承和新增了什么 |
| 限制与展望 | 作者明确内容、阅读者推断与待验证假设分开 |

每组读完后，先复述机制并提出一个能区分原因的仿真问题，再决定深入哪些证明和实现。暂不指定最终平台、网络或求解器，也不急于宣布创新点。

## 已有笔记的使用方式

[MR004综述笔记](../notes/MR004-survey.md)、[MR029初读](../notes/MR029-ac4mpc.md)、[MR008经典初读](../notes/MR008-economic-nmpc.md)、[MR022／MR026对比笔记](../notes/MR022-MR026-tdmpc.md)可直接使用，其状态仍为初读。新增优先论文需要继续逐篇形成笔记；已有100篇书目不表示已经精读100篇。
