# 从论文提取可实施的控制关系

核查：2026-11-16。下面区分原文结论和本项目推论；不是论文成功率向 Go2/TM02 的转移证明。W53 只实施 N/F 两个物理控制族，其他文献用于确定结构、接口和后续边界。

## 1. 最直接的恢复参照：MuJoCo 全身 iLQR

**Whole-Body Model-Predictive Control of Legged Robots with MuJoCo**，Zhang 等，2025，实际阅读 arXiv v2（2025-10-18）；项目页仍按预印本介绍，不擅自补会议/期刊标签。

[论文](https://arxiv.org/html/2503.04613v2) · [作者项目页](https://johnzhang3.github.io/mujoco_ilqr/) · [部署代码](https://github.com/johnzhang3/mujoco_mpc_deploy)

论文将 joint PD 放入预测模型、优化关节目标，用完整状态轨迹及 TV-LQR 反馈执行；在 Go1/Go2 等硬件演示步行。§IV 讨论接触求解、单边有限差分与导数开销；§V 描述约 50 Hz 规划、300 Hz 反馈、.35 s 预测，以及用于这些实验的 20 核桌面 CPU。其具体频率/模型不同于本项目，不是我们的实时证明。

**项目推论：** 这个控制族具备真实行走先例，仓库 R8 又有自己的 9+3 成功记录。N 应恢复完整运动语义，不能只换一个步高或复制论文的 impratio。当前与 R8 的模型、成本和执行差异见 [恢复矩阵](baseline-and-literature.md)。后者的刚地成功不提供当前材料资格。

## 2. F 的直接工程组织：接触力 MPC + WBIC/WBC

**Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control**，Kim、Di Carlo 等，2019；核查公开预印本。

[全文](https://arxiv.org/pdf/1909.06586) · [MIT 官方 Cheetah-Software](https://github.com/mit-biomimetics/Cheetah-Software)

论文 §III 由 MPC 规划接触力，§IV 的全身控制协调反力与浮基加速度，再得到完整关节命令。官方实现把同一 gait 的 contact table、摆足 p/v/a、MPC 反力接入 WBC，输出 tauFeedForward、qDes、qdDes。不是“IK 得到位置，再让原静站偏置自动完成动态支撑”。

**项目推论：** F 采用这一分工，适配同一 Go2 模型、原 60/5 PD 与原限制。原文 Mini-Cheetah 的 gains、摩擦或力上限不能直接搬入本项目；PD 接口换算和对完整总力矩的约束是我们的模型推导。材料法向压入/回弹还需与原 TM02 相容，论文不提供现成保证。

## 3. 实际接触反馈与完整执行层

**Perceptive Locomotion through Nonlinear Model Predictive Control**，Grandia 等，IEEE TRO 2023；所读方法全文为 2022 arXiv 公开稿。

[全文](https://arxiv.org/pdf/2208.08373) · [作者机构出版记录](https://www.research-collection.ethz.ch/entities/publication/ed499b50-44d5-4bd7-95a3-438fbc64a940) · [官方 OCS2](https://github.com/leggedrobotics/ocs2)

§VI 把 MPC 的接触力和关节速度轨迹经反馈及 WBC 转成力矩，并处理计划与观测接触不一致。高频 WBC 的全身动力学任务用 QP 协调，规划日程与真实接触不是同一个变量。

**项目推论：** F 的 swing 零规划力不能用于覆盖 plant 的实测力；必须记录计划接触、真实接触、参考、WBC 力和 actuator 输出。刚地 stance 零速度不能无修改当作可压缩地面的真实运动条件。早触地/晚触地行为要在首轮任务前明确，不在失败后临时生成一组新恢复策略。

## 4. 同一优化问题内协调运动与受力

**Whole-Body Inverse Dynamics MPC for Legged Loco-Manipulation**，Molnar 等，RA-L 11(1), 898–905，2026；实际阅读 2025 arXiv v1。

[正文](https://arxiv.org/html/2511.19709v1) · [作者项目页](https://lukasmolnar.github.io/wb-mpc-locoman/) · [官方代码](https://github.com/lukasmolnar/wb-mpc-locoman)

原文式(5)以全身逆动力学约束关节力矩、广义加速度和接触力；式(6)把 stance 速度/摩擦和 swing 零反力/竖直运动写入同一计划。B2+Z1 上的约 80 Hz 是其具体系统结果，论文还采用特定离散与力矩约束设计。

**项目推论：** 这为“运动几何、浮基动力学和力同时一致”提供另一种成熟组织方式。W53 不同时再实现完整 Fatrop 全身 MPC，而采用范围更明确的 force MPC + WBC；也不复制论文只对前两节点限制力矩等取舍来放宽本项目规则。

## 5. 采样方法也需要完整动态输入表示

**Reference-Free Sampling-Based Model Predictive Control**，Schramm 等，实际阅读 2025 arXiv v1；作者单位机器人页列于 ICRA 2026。

[正文](https://arxiv.org/html/2511.19204v1) · [Inria Go2 项目记录](https://inria-paris-robotics-lab.github.io/Robots/Go2.html)

方法同时采样关节位置与速度控制点，以 Hermite 样条形成连续目标，再由 PD 产生力矩；保存并执行已经 rollout 的最佳轨迹。作者在 Go2 上展示基于 CPU 的实时控制，所用仿真器、低层频率和任务与本项目不同。

**项目推论：** 不能只因为 D3 更多搜索仍失败就推断所有采样/优化都无效；候选表示能否表达实际运动很关键。但再加 MPPI 会产生第三个物理控制族，增加对模型与动作语义的适配成本，本轮不启动它。它也不证明必须用 GPU 才能走。

## 6. GPU 应迁移真实模型计算

**CusADi: A GPU Parallelization Framework for Symbolic Expressions and Optimal Control**，Jeon 等，实际阅读 2024 arXiv v1。

[正文](https://arxiv.org/html/2408.09662v1) · [官方实现](https://github.com/se-hwan/cusadi)

CusADi 将符号函数生成为批量 CUDA 计算，能涵盖动力学量与优化装配。原文明确讨论 host/device 数据搬运对收益的影响；官方示例支持 double，但并不是完整 Go2/TM02 MPC。

**项目推论：** W53 C 针对 F 真正需要的 Go2 模型与约束矩阵，采用既有状态有限遍历、CPU/GPU 同模型对比及包含传输的计时。CPU 的接触积分/FD 不会因为 GPU 显存大而自动消失。批量路径可有价值，单个小 QP 也可能没有 GPU 延迟收益，两种结果都应真实报告。

## 7. 与本轮实际决策的映射

| 实证或方法启发 | 本轮动作 | 尚未解决的边界 |
|---|---|---|
| R8 与作者 MuJoCo 控制已行走 | N 恢复完整运动内核，并接当前任务 | 材料未知 predictor；不能继承旧资格 |
| MPC 反力与 WBC 全身执行 | F 替换旧几何输入增量控制族 | 新模型/接口工程；软材料与接触切换 |
| 接触反馈和统一动力学约束 | 保留真实接触反馈、完整 q/v/tau 命令、原物理限制 | 硬日程不自动等于实际卸载 |
| 丰富动态搜索表示 | 停止把更多旧局部方向作为默认补救 | MPPI 不在本轮实施范围 |
| GPU 批量模型函数 | C 服务 F，不复做 residual→J | 不代表 GPU physics、整 MPC 或实时 |

公式和项目判断来自原仓库与本次分析，不应把本文解释成任何论文为 W53 成功背书。完整参数、任务次数及后续阶段条件统一见 `docs/S1_W53_STABILITY_PLAN.md`。
