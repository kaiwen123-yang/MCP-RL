# W56 文献复核：先保留 r55 资格链，再使材料历史进入决策

审查日期：2026-11-20。本文记录 W55 审查的文献依据；W56 计划另见 docs/S1_W56_STABILITY_PLAN.md。项目事实以逐 job 原件为准。本审查未启动物理、训练或修改仓库。

## 1. 适用前提与结论

主审已经确认：F/r55 的 A、B 各自原 E30、E60、E300 均通过；两条 E600 在最新快照仍运行。r53 的终点保持错误是围绕未来参考 COM 计算当前接触力矩臂，r55 进行了局部几何修复。N 仍停滞；本轮材料候选在快速诊断改善、两个独立 holdout 恢复恶化，因此没有替换原 B ideal。

因此，文献不支持现在撤掉 r55、重写控制族或用残差 RL 补一个已经修复的几何错误。当前最有用的三项借鉴是：

1. 保留成功控制基线，用完整 MPC 数值链和实际工作负载评估 GPU 收益。
2. 把材料历史与候选预测、动作选择的连接做实；材料预测误差改善本身不等于任务收益。
3. 清楚区分学习的三种位置：动力学模型修正、MPC 高层参考/接触调度、MPC 之后的执行残差。它们需要不同状态、动作和约束合同，不能把论文标题里的 MPC+RL 当成相同方案。

下列“本项目借鉴”均为本审查的推论和实验设计，不是论文已经证明的本项目结论。

## 2. 四篇优先文献

### A. Residual MPC: Blending Reinforcement Learning With GPU-Parallelized Model Predictive Control

- 作者：Se Hwan Jeon、Ho Jae Lee、Seungwoo Hong、Sangbae Kim。
- 日期：arXiv v1 2025-10-14；IEEE T-RO 正式记录 2026-08-07，卷 42，3469–3488。
- 原文：<https://arxiv.org/html/2510.12717v1>
- 正式记录：<https://ieeexplore.ieee.org/abstract/document/11644391/media>
- 作者页：<https://hojae-io.github.io/>
- 官方公开组件：<https://github.com/se-hwan/cusadi>。这只是 CusADi 组件；本次没有确认完整 Residual MPC 复现仓库。
- 阅读范围：arXiv HTML 的模型、SQP/ADMM、GPU、融合接口、结果及限制；正式记录和作者页。未运行代码，未逐帧看作者视频。

**文献事实。** 采用 floating-base kinodynamic MPC，一次 SQP、Ruiz 缩放、KKT 分解复用和 ADMM；函数/导数由 CusADi 上 GPU，分解和回代使用 cuDSS。策略最终选用 joint-action/torque-blending。1000 环境的 MPC 一步墙钟约 0.5 s；100 Hz 是控制频率，不能据此称批处理在 10 ms 完成。MIT Humanoid 有前向、横向、转向硬件展示，机载 CPU OSQP 为 100 Hz；作者明确硬件范围有限，并发现接触仿真差异。它没有证明任意残差均维持原 MPC 硬约束。

**本项目借鉴。** 优先复用已完成的 force-horizon GPU 工作，在 r55 新产生的起步、跨材料、回家制动、终点保持状态上检查完整数值链的一致性；不要再拿旧 16 状态重复算成新成果。算子顺序必须包含状态/材料更新、矩阵构建、缩放、求解、执行映射及最终约束检查。GPU batch 吞吐、单条在线延迟、完整实验墙钟要分别测。残差学习可以是日后独立方法比较，但不能在运行中的 E600 热加入或继承资格。

### B. RL-Augmented MPC for Non-Gaited Legged and Hybrid Locomotion

- 作者：Andrea Patrizi、Carlo Rizzardo、Arturo Laurenzi、Francesco Ruscelli、Luca Rossini、Nikos G. Tsagarakis。
- 日期：arXiv v1 2026-03-11；作者官方代码列 IEEE RA-L 2026，DOI 10.1109/LRA.2026.3675839；本次未独立核到正式上线日。
- 原文：<https://arxiv.org/html/2603.10878v1>
- 官方代码：<https://github.com/AndrePatri/AugMPC>，README 还链接 MPCHive、EigenIPC、AugMPCEnvs 与模型。
- 阅读范围：摘要、控制与接触调度、动作/观测、episode、并行实现、结果和结论；官方仓库 README。未运行或逐文件审计代码。

**文献事实。** 高层 RL 输出 twist 和接触起飞触发，MPC 执行；观测含 MPC 接触计划、收敛/约束健康指标与动作历史。并行是 CPU MPC 子进程、共享内存和 GPU 仿真/训练协同。Centauro 有真实零样本迁移。文中不施加关节力矩界，不等式以 quadratic barriers 近似；强持续外扰仍列为后续系统研究，因此不能直接继承其安全/长期鲁棒结论。

**本项目借鉴。** 可参考显式 learning-control 接口：材料状态应该对应有物理作用的决策量，同时提供当前支持、计划相位和可行性信息。本项目现有 2 Hz cap 不能自动等同该论文的接触调度动作空间。先测 cap 对材料状态的真实敏感度，明确新接口能改变什么，再选择低维受限参数。不能为照搬这篇论文而移除总力矩界或改原判据。其 CPU/GPU 混合并行也说明 96 GB 显存不等于所有低维控制计算都应搬上 GPU。

### C. Adaptive Legged Locomotion via Online Learning for Model Predictive Control

- 作者：Hongyu Zhou、Xiaoyu Zhang、Vasileios Tzoumas。
- 日期：arXiv 初稿 2025-10-17，v2 2025-11-30；v2 和作者页确认 RA-L 于 2025-11-25 接收。
- 当前原文：<https://arxiv.org/html/2510.15626v2>
- 作者记录：<https://hongyu-zhou.github.io/>
- 官方代码：<https://github.com/UM-iRaL/Adaptive-Legged-Locomotion>。README 明确这是 Gazebo 实验代码，不能称全部 MuJoCo/硬件复现包。
- 阅读范围：v1 模型、在线算法、实验失败与计算讨论；v2 摘要、版本/接收信息及官方链接。v2 正文未逐段与 v1 对比。HTML 公式字体有乱码，未据乱码复述定理常数或公式细节。

**文献事实。** 用随机 Fourier 特征和在线最小二乘更新预测动力学残差，再送入 MPC；系统观测与模型更新逐周期互相作用。验证是 Gazebo 和 MuJoCo 仿真。论文明确突发落足变化、意外接触和极端载荷/变摩擦条件下仍会失败；动态 regret 结果不能解释为每时刻硬安全或材料迟滞模型已被解决。

**本项目借鉴。** 最相关的是学到的信息确实参与未来状态预测。可为现有 PANN 建立因果的逐地块状态更新和候选独立演化，让“预测更准确”能够影响选择；不必照搬 RFF 重学一个已经有物理结构的材料模型。先在原日志上比较 learned-history、屏蔽历史和 oracle-history 的预测/选动作差异。oracle 只用于诊断上限，不能作为部署输入。只有证明控制接口使用了这些信息，才有理由为这一用途增加训练。

### D. Whole-Body Inverse Dynamics MPC for Legged Loco-Manipulation

- 作者：Lukas Molnar、Jin Cheng、Gabriele Fadini、Dongho Kang、Fatemeh Zargarbashi、Stelian Coros。
- 日期：arXiv v1 2025-11-24；IEEE RA-L 11(1):898–905，2026-01；DOI 10.1109/LRA.2025.3636005。
- 原文：<https://arxiv.org/html/2511.19709v1>
- 项目：<https://lukasmolnar.github.io/wb-mpc-locoman/>
- 官方代码：<https://github.com/lukasmolnar/wb-mpc-locoman>
- 阅读范围：HTML 方法、warm start、边界、硬件结果及限制；项目和仓库首页。未运行代码。

**文献事实。** 全身逆动力学 MPC 利用 Fatrop 的阶段稀疏结构，在 B2+Z1 上以 80 Hz MPC、500 Hz 插值执行。接触序列改变时重建接触力初猜，防止旧解接触模式不匹配；作者也修正了模型 COM 偏差。关节力矩界只施加于前两节点。论文明确模型、接触和时延失配，不能将其结果理解成整预测域严格力矩可行。

**本项目借鉴。** 这支持优先核实力矩臂坐标、实际 COM、真实接触模式与 warm start 的一致性。r55 已修得有效结果，当前应冻结它。未来若重新出现特定接触模式的数值失效，先保存并重放该模式的原始矩阵和猜测，隔离几何/模式/初始化因素，再讨论换求解器。

## 3. 备用数值来源，不作为本轮更换 solver 的理由

**ProxQP: an Efficient and Versatile Quadratic Programming Solver for Real-Time Robotics Applications and Beyond**，Antoine Bambade 等，T-RO 2025-06，DOI 10.1109/TRO.2025.3577107。

- 作者页：<https://scaron.info/publications/tro-2025.html>
- 项目：<https://simple-robotics.github.io/publications/proxqp/>
- 官方代码：<https://github.com/Simple-Robotics/proxsuite>
- 阅读范围：作者摘要、项目与仓库介绍；HAL 全文遇到验证页面，未阅读全文，不能称已复核其全部理论。

若未来出现真实新 QP 故障，可作为保存矩阵的离线交叉求解器。作者所说 closest feasible QP 不表示原来的全部硬约束自动被满足。独立检查必须在原单位、原界上进行。

OSQP 官方状态文档：<https://osqp.org/docs/interfaces/status_values.html>；设置文档：<https://osqp.org/docs/interfaces/solver_settings.html>。`maximum iterations reached` 与 `primal infeasible` 是不同状态；`solved inaccurate` 条件采用请求容差的 10 倍。不能将它们都统计为同一种不可行根因。保持任务没有墙钟截止，也不意味着一个 QP 必须无限迭代；两种时间/终止语义应分开记录。

## 4. 可以立刻设计、不会打断原 E600 的两项工作

### 4.1 材料历史是否对决策有用途

从 r55 已有全程原件建立按世界地块、事件和方向分层的清单：首次加载、快速释放、跨足重访、返程重访、终点附近。沿各自真实历史重建状态，明确 0.2 ms plant 与 2 ms observer 的边界及可观测输入。

离线反事实只消融控制器可见信息；不把真实 plant 的历史清零。相同冻结控制状态、候选集合和代价下，比较 learned-history、memory-ablated、oracle-history 的预测和动作选择，同时报告约束残差、动作变化及动作值差。如果当前 cap 几乎不变，应明确这是接口弱使用，不能以预测误差降低冒充记忆收益。

下一阶段必须另设“候选会改变材料未来”的检验：每条候选都独立 clone 材料状态并沿自身载荷更新，不能让候选共享可变状态，不能用真实内部 q teacher forcing。用屏蔽历史仍可获得的信息构造公平基线；若不同历史当前可观察状态无法匹配，说明未完成因果隔离。完整闭环对照须等冻结资格链的结论并由新实验身份执行，不在 E600 中途改变 observer/PANN/控制接口。

### 4.2 完整计算路径的 GPU 迁移或批量收益

从合格 r55 自然状态抽取实际任务相关批次，覆盖去程、返程和终点，不重做 W54 已结束清单。先 profile：CPU 物理、材料状态更新、MPC/WBC 构建、求解、执行映射、日志，各自占多少时间。

把 GPU 结果与独立 CPU 复算对齐：同一输入、物理单位、接触模式、初猜、摩擦集合、总力矩界及输出语义。原界可行性和获选动作/代价必须单独比较。若 WBC/执行映射仍在 CPU，要明确目前只覆盖 force-horizon；不能称完整控制已 GPU 化。

报告 batch 吞吐、完整批次墙钟、单实例延迟分位数、传输/编译/热计算、显存峰值和两条物理链实际资源重叠。批次时间除以环境数只能表示摊销吞吐成本；不能代表在线 deadline。GPU 有用工作可以与 A/B E600 并行，但 GPU 满显存本身不是目标。

## 5. 检索边界

已额外筛读 Dual-MPC Footstep Planning for Robust Quadruped Locomotion（arXiv:2511.07921，2025-11-11）的公式和 GO1 硬件实验；它将落点与 GRF 联合用于角动量控制，但仍是近似 SRB，且加快步频的基线在部分跟踪指标更好。本轮不把它扩成第三控制族。

检索到 DR-MPC（arXiv:2609.20035，2026-09）及 CacheMPC（arXiv:2606.28300，2026-06）原始摘要；全文入口在本环境不可读，未验证完整方法与代码，因此不作为本轮关键依据。对新论文的摘要，不能把“relaxed feasibility”误写成满足原物理硬约束，也不能把求解器的时间预算误写成后台任务完成截止。

