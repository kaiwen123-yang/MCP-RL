# W45 后的控制机制评审：完整名义策略、接触附近反馈与刚地迁移

日期：2026-11-12。审查仓库基点 `3e3f69c440d6a33c14131563e5f485d88da3a803`，实际控制器 `S5-r34`。本文只做静态源码、既有结果及一手文献分析；没有运行新的物理实验，也没有替换原 Go2/TM02 任务。最新原件定量因果链以同轮 control review 为准。

## 1. 直接判断

W45 已消除 W44 中可辨认的 alpha=0 平手换反馈触发，但 A/B 原 E30 仍分别在 13.072/5.582 s 跌倒。A 曾实际加载材料，B 未加载；合法贯穿/返程仍未建立。因此继续修同一个 alpha=0 问题或扩大 PPO 样本，不是本轮最有根据的主方向。

最新原件已经把问题缩小得更具体：**现有代价确实愿意用较强控制、较大反馈和更多接触切换，交换短时高度及前进的恢复。这个偏好在 native 候选里就已发生，不能只归咎于实际执行误差。** A call153 的正 alpha=1 使 native J 从 .116834480 降到 .089095071：MaterialSupportHeight 下降 .059589644，而 Effort 增加 .030527000；Upright、Gait 也变坏。native 在 2.648 s 已预测全足零法向力，并在 2.650 s 出现约 2.287 rad 有效反馈；实际分别在 2.6464 s 和 2.650 s 出现对应现象，后者约 3.123 rad。B 的 2.68 s nominal 同样自己预测稍后的零承载，2.762 s native/actual 大反馈约为 .73615/.74044 rad。

与此同时，这个动作在当前 .35 s horizon 内并非没有价值：A 新策略把最低 body z 从 .248456 提到 .272044 m，把 dx 从 −.019971 改善为 +.015222 m；末端从 (.248456 m, −.247233 m/s) 变成 (.292285 m, +.259592 m/s)。因此“把高度项去掉”“只把反馈永久压小”都没有得到现有数据支持。还需判断这些短时恢复对真实尺度扰动是否稳定，以及 horizon 末端的状态是否有可信后续。

当前最值得检验的是**所选完整策略的恢复能力与接触附近闭环适用范围**。代价下降、QP 收敛、native FD 正确都不足以证明这件事。也不能把 2.65 s 的大反馈直接断言为 13.072 s 跌倒的唯一原因；中间还有多次重新规划。下一轮应同时保留“名义动作已不可信”和“名义短时改善但反馈局部性太强”两个可否定的分支。

建议先在 W45 已存的早期事件做一个有限反事实，随后只选择一种最有证据的机制进入新的 A/B 固定策略原任务。当前不建议同时换 solver、动作空间、接触模型和任务 gait，也不建议直接重写成 IDTO 或 FDDP。

## 2. R8/S3 成功控制器与 W45/r34 的实质差别

### 2.1 可直接复核的材料

- `experiments/terrain_memory/s1-go2-wb-support.json`
- `experiments/terrain_memory/s1-go2-wb-support-upstream.patch`
- `results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/project-s3-single-05-resolved.json`
- 同目录 `project-s3-single-05.json`、`project-s3-long-06.json` 和 formal-v1 各条件结果。
- `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261112T103825Z-w45-a-eval-PANN_PPO_S5-s1-E30-attempt01-resolved.json`
- `experiments/terrain_memory/s1-go2-r11-S5-r34.json`。
- 本轮按保留 r25 源码反向 r25 得到 r24，再按 r26/r27/r28/r29/qp1/qp2/r30/r31/r32/r33/r34 顺序重构的源码；所有 patch 无 reject。重构 provenance 由 control review 保存。

R8 是原项目 G1 刚地实验，不是原作者较宽松物理模型的结果。原作者 G0 另有质量和无界 actuator force 差别，不能与项目 S3 混写。

### 2.2 已逐字段比较的共同物理配置

R8 S3 nominal 与 W45-A resolved 的以下字段完全相等：质量 15.206408 kg、MuJoCo 3.2.3、初始 qpos/qvel、关节名称与物理范围、actuator gear/gain/bias/类型、target 范围、force 范围及 limited 开关、head offset，以及 plant .2 ms / feedback 2 ms / planning 20 ms。PD 都是 60/5；每腿原力矩范围都是 23.7/23.7/45.43 Nm。

这支持一个有限但重要的结论：**已有结果证明原硬 Go2 与原力矩界并非刚地不可行。** 它不证明材料任务可行，也不证明当前 r34 仍保留了 R8 的闭环稳定性。

### 2.3 控制器对照

| 层次 | R8/S3 实际执行 | W45/r34 实际执行 | 含义 |
|---|---|---|---|
| 任务物理 | 原 G1 刚地；去程、转向、返回、终点连续稳定 | 同一机器人增加原 TM02 世界固定材料带与真实应变域 .02 | 刚地成功不能直接当作材料资格 |
| 规划时域 | .35 s，175 个 2 ms 控制区间 | 相同 | 不能把短时域单独认定为新增根因 |
| 每控制区间预测积分 | 2×1 ms implicitfast | 10×.2 ms implicitfast，coldstart | 当前更细；细化并未自动带来稳定 |
| 规划接触/导数 | differentiable=1 的平滑规划前向与 FD | 原 native geom/pair 接触前向；先 soft residual GN 度量，再 native-contact FD 方向 | 当前不是原作者默认或 R8 的优化问题 |
| 移动 gait | 2 Hz、duty .45、amplitude .03 m | 2 Hz、duty .45、amplitude .005 m，分段 C2 缓降 | 3 cm 与 5 mm 只是完整差异之一；没有证据能单独用换脚高解决 |
| 移动 Upright/Height | 原作者移动目标，Stand 用兼容参考 | 移动时全程有一阶 roll/pitch 目标、额外 absolute body-height；名义 stance 足也有高度代价 | 当前代价的局部曲率与偏好已改变 |
| 移动前进参考 | 原任务 .25 m/s 目标调度 | PPO 2 Hz cap .18–.25 m/s，r26 因果头部参考 packet、.15 m lead、.5 m/s² 加速 | 当前不仅多了材料扰动，还换了目标生成 |
| 风险 | 原物理终止与原模型 | 附加落地/缩进/材料域/撞击/躯干风险，r27 C2 空间激活与十子步峰值 | 这些项必要性要按物理和数值作用分别检查 |
| 控制约束与反馈 | 原 target bounds 与 actuator force limits | r29 H 投影、r30 移动 PD bounds Riccati、r32 epoch lifecycle、r33 D 尺度、r34 完整策略比较 | 不能用“同样是 iLQR”认定是同一基线 |
| Stand 参考 | 原生静态平衡全 q/u、脚位/脚速、相容 body/head 参考 | 仍继承 | 不是尚未实现的修复 |

### 2.4 R8 真正解决了什么，哪些不能再重复提案

R8 C/H/S1/S2 的保留结果都是 60 s task_timeout，而非启动后立即 body_fall；S3 nominal 原任务于 36.378 s 首次完成，最终观测至 60 s；名义长任务观测至 300 s。配置和索引记录 S3 原任务 9/9 与持续 300 s 3/3。

S3 并非只增加抬脚高度。它在 Stand entry 构造原生可行的整套参考：世界脚位、零脚速、body/head 姿态、关节姿态、重力平衡控制输入。静止时 `u_ref=q_ref+tau_ref/60`，并非在执行端额外叠加重力力矩。S3 把 Stand 输入参考尺度从 .1 rad 改到 .02 rad，同时保留 Effort 权重 .08，提高该项输入曲率 25 倍。S2 曾增加独立承载力目标，结果更差，S3 将其权重设为 0。

这些 Stand 机制在 W45 的 XML/readback 与源码中仍存在；当前早期问题发生在 Trot。因而“再做静态平衡”“再加 PD”“再加 TV-LQR”“再给脚加位置代价”都不是本轮的新修复。

当前移动 Effort 仍为 `.02*actuator_force`，Stand 才是 `(ctrl-u_ref)/.02`。Balance 仍使用四足几何中心，非实际承载足的力加权支撑域。即使 solver 的 `D`、PD bounds 和 projection 都成立，也没有自动产生一个在接触模式改变后仍可信的反馈策略。

### 2.5 双 Height 目标、支撑几何与已有撤回项

现有原 `Height` 为 torso 惯性 COM 的 z 减四只脚几何中心 z 的平均，再减 .27 m；移动时仍用 .27，不沿用 Stand 的相容高度参考。`MaterialSupportHeight` 则为 `(base body z−.29020336039643685)/.04`，权重 1、二次范数。r14 起它在全部 moving Quadruped 上生效，而不是只在材料内生效。该项来自 position-only 高度恢复，移动时没有相同用途的垂向速度目标。

在记录的 native Stand 平衡，base z≈.290204、原相对 Height≈.262842 m，所以 Stand→Trot 时已经有约 7.158 mm 的参考变化；摆腿还会改变四脚平均高度。绝对高度项的小误差曲率为 625/m²，而原 Height 的 smoothabs/.04 约为 25/m²。这是两套参照不完全相容的可复核证据。它不是“收腿一定降低 Height”或“这一差值导致全部失败”的证据：收腿会提高平均脚高，对原 Height 的影响随误差符号而变；A153 的绝对高度改善也真实发生。

`Balance` 中的支撑中心仍是四足几何中心；`r27` 只把材料接近区的 LandingVelocity/ImpactMargin 激活改为 C2，并没有把 Balance 改成实际承载足、没有构造一条全身可行的走路参考。`r25_stable_support` 曾加入 phase 请求载荷、实际力/力矩和落脚代价，但随后撤回；r26 从 r24 分叉，当前 r34 没有这些项。不能把重新加入它们称为未经尝试的新修复。

尤其不能简单关掉全局绝对高度：r14 的原始证据是 r12/r13 在脚退出材料带后同时关闭 absolute height/first-order Upright，随后 nominal 允许 body 下落到 .175499 m。当前全局激活修的是那个漏洞，撤销会让旧问题复活。

A153 新策略的无足法向力节点为 68/176，基准为 30/176；最长连续区段分别 27/30 个采样点。B155/B158 分别为 56/74 个节点。这里包含 terminal knot，节点数不能直接冒称 5000 Hz 真值下的精确持续时间；短 flight 本来是允许的。这个差异可用于检查支撑交换与后续恢复，不能变成新增的物理失败条件。

## 3. 三篇直接相关的一手文献

以下介绍各文献中与本轮有关的一小部分，不把作者实验效果外推成本项目的保证。之前已经审查的 MuJoCo WB、ROM-guided、LinearWalking、RF-MPC、FS-MPC、DIAL 不在这里重复铺表。

### 3.1 Hybrid iLQR Model Predictive Control for Contact Implicit Stabilization on Legged Robots

Nathan J. Kong、Chuanzheng Li、George Council、Aaron M. Johnson。IEEE Transactions on Robotics 39(6):4712–4727，2023；DOI `10.1109/TRO.2023.3308773`。

- [原论文 PDF](https://arxiv.org/pdf/2207.04591)
- [IEEE 正式出版](https://ieeexplore.ieee.org/document/10252162/)
- [CMU 作者组出版记录](https://www.cmu.edu/me/robomechanicslab/publications.html)
- [作者公开算法代码](https://github.com/Kongx231/ilqr)

**原文机制（§III-C/D、§IV-A/B）：** saltation 处理事件时刻扰动，参考延拓处理提前/延迟接触。但是作者明确区分 event-driven 和 time-stepping：多步形成接触时，离散模式不够明确，不能直接使用该延拓法。其四足 time-stepping 路径按每腿法向承载份额设置输入偏差权重，低承载腿提高输入改变量成本；并保留旧解有效部分、将完整参考始终作为候选。论文中并行 line search 允许各候选完整计算。

关键公式为 `w_j=Fz_j/sum(Fz)`、`R_j=Rmax-w_j*(Rmax-Rmin)`（Eq.38–39）。四足均分承载时 w=.25，两腿均分时 w=.5；原文并非用 mg/支撑数归一化。原文未提供可直接照抄的 Rmin/Rmax 数值和本项目关节目标输入单位。该代价是 `(u−u_ref)^T R (u−u_ref)`，不是相邻时刻的 `u_k−u_{k−1}`，也不是只加到 Quu 的优化正则。

**关键前提（§VI-B）：** 作者的完整 walking reference 来自未受扰动时稳定运行的 Convex MPC。该方法围绕已经可行的走路轨迹处理扰动，并非保证一个已经不稳定的 incumbent 只靠 R 就能变稳定。本项目现在只有 native Stand 全参考，不存在已经核验过的完整 moving reference；把前者写成后者会高估已有能力。

**源码核对：** `matlab/hybrid-ilqr/h_ilqr.m` 的 `create_hybrid_collars`、`forward_pass` 和 saltation 分支存在；实际代码在 mode mismatch 时选取对应参考、输入与 K，而不是继续用错误模式的时间索引。该仓库是公开的基础算法与示例；本次没有确认完整 Unitree A1/Isaac Gym 部署实现公开，不能称为可直接移植的 MuJoCo 补丁。

**本项目映射（有条件的建议）：** 承载相关输入偏差代价是比“加盐跃矩阵”更小的候选，但不能先验宣告它解决当前失稳，或绕过有限对照直接进入长任务。现有 r27 调的是材料空间激活，r30 解的是瞬时 actuator 可行盒，r33 调的是 soft residual GN 度量，它们都不等于承载控制权处理；但没有可信 moving nominal 时，追踪旧输入可能锁住坏轨迹。只有有限对照支持名义本身可恢复、主要困难是低承载腿不合适的反馈控制权，才值得推进。其 reference、承载权重和 native 代价必须一致，不能只在部署端缩 K。

### 3.2 Contact-Implicit Model Predictive Control: Controlling Diverse Quadruped Motions Without Pre-Planned Contact Modes or Trajectories

Gijeong Kim、Dongyun Kang、Joon-Ha Kim、Seungwoo Hong、Hae-Won Park。IJRR 44(3):486–510，2025；online 2024-10-01；DOI `10.1177/02783649241273645`。

- [论文 v2](https://arxiv.org/html/2312.08961v2)
- [正式出版](https://journals.sagepub.com/doi/abs/10.1177/02783649241273645)
- [KAIST 出版记录](https://pure.kaist.ac.kr/en/publications/contact-implicit-model-predictive-control-controlling-diverse-qua/)

**原文机制（§4、§6.1–6.3）：** forward 使用严格接触，backward 使用松弛互补梯度；FDDP 同时移位保留状态/控制轨迹，逐步收缩初态 discrepancy 的 dynamics gap。作者用转向对照说明：无 plant 偏差时单射击可行；加入模拟器后，微小脚高差引起提前丢接触，原本用于承载的力矩加速自由腿，随后轨迹发散。初始化从求解过的站立轨迹开始。其执行层使用规划状态/速度与前馈力矩 PD；作者说明 iLQR gains 未超过该 PD，原因是接触变化容易使局部增益失效。

**实现边界：** 3D 实现基于 Crocoddyl Box-FDDP/Pinocchio，45 kg HOUND，40 Hz MPC/25 ms knot、.5 s horizon；论文不是 Go2/TM02 复现。本次未确认完整 HOUND 官方实现公开。本文方法中不完全收敛的 gap 有时保留至下一次重规划，这与本项目完整 native rollout 的安装要求有差别。

**本项目映射（我们的建议）：** 首先检查当前 shifted nominal 是不是从新状态重新滚动后出现接触级联，以及该问题与部署反馈崩坏谁先发生。若有此证据，再考虑保留完整 x/u 的可行性驱动搜索。不能只保存旧 x 数组就宣称实现 FDDP；需要 dynamics gap、backward 递推和接纳一起修改，且安装前仍须原物理可执行验证。已有 Stand equilibrium 和 PD 是起点，不能作为新进展重复计数。

### 3.3 Inverse Dynamics Trajectory Optimization for Contact-Implicit Model Predictive Control

Vince Kurtz、Alejandro Castro、Aykut Özgün Önol、Hai Lin。IJRR，online 2025-05-30；正式卷期 45(1):23–40，2026-01；DOI `10.1177/02783649251344635`。这里校正仅称“IJRR2025”容易产生的卷期混淆。

- [论文 v3](https://arxiv.org/html/2309.01813v3)
- [正式出版](https://journals.sagepub.com/doi/abs/10.1177/02783649251344635)
- [官方项目](https://idto.github.io/)
- [官方代码](https://github.com/ToyotaResearchInstitute/idto)

**原文机制（§4–5、§6.4、§8）：** 用位置轨迹作为变量，逆动力学求控制，使用缩放后的 Gauss–Newton dogleg trust region；柔顺/平滑规划接触与更严格执行模型分开。它能绕开部分 forward shooting 接触局部极小，并允许时域并行计算。四足 60 Hz 是 Mini Cheetah 仿真，>100 Hz 硬件是双臂。欠驱动力约束只在收敛时成立，未收敛结果可能含虚构浮基力，不能直接算原物理可行。

**源码核对：** 读取官方 `optimizer/trajectory_optimizer.cc`（本次 main tree `de0629c7811aa9b330e56c4385629005b09495f0`）与 `optimizer/solver_parameters.h`；`CalcTrustRatio` 用实际/预测 merit 改善，`CalcScaleFactors` 与 dogleg 使用缩放约束。还有小改善时人为返回 .5 ratio 的分支；本项目不能照抄该分支来撤销 r34 完整策略平手保留，因为二者控制策略对象与接纳目的不同。

**本项目映射（我们的建议）：** 可作为以后离线生成可信启动/短步参考的工具，不是当前最小在线修复。r33 的曲率缩放只涉及 Hessian 正则，不能等同 IDTO 的位置变量、信赖域和欠驱动可行性。任何外部生成的 nominal 都需回到原 Go2/TM02、PD、力矩及非足接触规则复核；不能用平滑模型的“成功”替代真实域 .02 的资格。

## 4. 下一轮有限实验应该区分什么

**当前唯一执行范围以 [W46 稳定性主计划](../../../../../../docs/S1_W46_STABILITY_PLAN.md) 为准。** 以下说明已经冻结的有限比较及其解释边界，不另行授权实验。FDDP、在线鲁棒评分、Euler/implicitfast 切换和完整 moving reference 生成都是以后可能单独立项的条件分支，不能在本轮顺带执行。以下内容均不是已完成测试。

### 4.1 已冻结的四问题、12 次求解、36 次诊断

使用四个 W45 保存上下文：A call148/plan136@2.54 s、A call153/plan140@2.64 s、B call155/plan141@2.68 s、B call158/plan144@2.74 s。A148 仅是前 20 ms 相对平稳的参照，其整个 .35 s 不能称为平稳。每个问题在共同重建上下文和共同冻结参考下，分别执行 r34、trace-matched constant、load 三种有限求解，共 12 次；这不是另加一组“两套旧策略”的 CPU 实验。

每个分支求得的完整策略只作原初态、记录量级正偏差、反向偏差三次 350 ms 闭环诊断，共 `4×3×3=36` 次。诊断固定该次求解的 reference、K/H、scale 和共同 ubar/F/R，只按 2 ms 执行反馈、H 投影和保持输入，**不在窗内每 20 ms 重新优化或创建新 epoch**。

这 36 次统一使用当前 **native predictor、implicitfast、每个 2 ms hold 内 10×.2 ms、原冷启动与冻结材料 proxy**，不新增 Euler 真值重演或改变积分器。所记录的是该 native rollout 中实际施加的投影后目标、预测 PD 力矩、支撑与状态，不能称为新 constant/load 版本的 actual plant 真值。W45 已保存 actual 只能对应其原部署策略；新版本的原 Go2/TM02 真值要由后续原 E30 给出。

缺少完整历史 task/private gait/solver 内态时，必须标为共同重建上下文对照，不冒称严格历史重放。偏差优先来自问题形成前已记录的实际状态减对应 native 预测的切空间误差；事后才可恢复的偏差只能按主计划标注为离线敏感性诊断，不能冒称因果线上输入。不为得到漂亮结论调整幅值或扫参数。

第三 GPU 任务与上述 36 次分开：仅对这四个问题的原完整 baseline/原选中策略及原初态/±偏差做最多 `4×2×3=24` 个旧策略 scenario，按主计划依次检查动作代数、最初 20 ms、条件满足后最多一次完整 350 ms 批次。它不增加 CPU 求解分支，也不把 MJWarp 输出冒充 native 或 plant 真值。

### 4.2 结果的解释与以后可能单独立项的方向

下面是文献对负结果的解释线索，**不是本轮额外任务、追加测试或可现场选择的补丁菜单**。本轮只实施 §4.3 的输入参考代价有限比较；不支持时按主计划完成负结果，不顺带运行下表方向。

| 以后出现的证据条件 | 可单独评估的方向 | 必须保留的边界 |
|---|---|---|
| 名义尚可，但记录量级扰动使策略明显失去恢复能力 | 在线完整候选的有限闭环邻域评分 | 本轮 36 次离线诊断不等于已经实现线上鲁棒接纳，也不证明全状态鲁棒域 |
| 新状态下重新初始化 nominal 后，接触级联先于部署失稳 | 可行性驱动的状态/控制 warm start，必要时研究 FDDP | 要明确 dynamics gap 收缩和最终 native 可执行性；不能只保留旧 x 数组就称为 FDDP |
| 今后单独的完整策略对照证明数值执行映射差异主导 | 预测/FD 与执行积分语义对齐 | 本轮主诊断不切换 Euler；不能仅凭积分器名字不同认定根因 |
| incumbent 与新候选都不可靠，局部求解持续找不到可信动作 | 完整可执行 moving nominal 或短步参考生成 | 刚地参考不能代替 TM02 资格，不得硬锁未来接触或安装含虚构浮基力的轨迹 |

这些方向都需要本轮完成后另行评估。当前论文没有提供本项目的非线性稳定性保证；有限诊断的价值是区分名义改善与记录量级误差下的恢复，不能假定 native 从未看见大反馈，也不能把选回 incumbent 写成安全保证。

### 4.3 本轮收敛后的 r35 有限候选：输入参考偏差代价

主计划已将下一步收敛为 §4.1 所述的 r34、trace-matched constant、load 三种版本、12 次有限求解和 36 次固定 native predictor 诊断。以下只是这一候选机制的数学说明，不再增加参数网格、另一套 gait、在线鲁棒接纳或全新 solver。该实验是验证一个控制假设，不是原 E30 的替代资格。

对每个 current epoch，在更新前用共同原上下文重滚 incumbent 的**完整策略**，取其实际部署语义下的关节目标 `ubar[k]` 与预测承载 `Fz_j[k]` 作为共同 anchor；这一对数组在该 epoch 内冻结。它不是已证明稳定的 walking reference。三个版本都保留原目标各项与原物理约束，分别增加以下新项或不增加：

`Jadd = (1/(2N)) * sum_{k=0}^{174} (u[k]−ubar[k])^T R[k] (u[k]−ubar[k])`，其中 `N=176`；终端没有新增控制项，成本只归约一次。

Load 版本按论文份额定义 w；无承载时 w=0 并取 Rmax 是本实验需明确声明的工程分支，不冒称论文原值。Constant 对照每个 knot 用完整 12 维矩阵的 `Rconst[k]=trace(Rload[k])/12 * I12`，这样总输入曲率完全匹配；如果 constant 同样好，证据支持的是输入参考代价，不能把效果归功于承载分配。不能让每个候选自己重设 ubar/R，否则代价对象不同且可被归零。

输入是 rad 的关节目标，不是论文的 Nm 力矩。新增 R 的单位是代价/rad²。原移动 Effort `.08 * .5 * (.02*tau)^2` 经未饱和 PD 对输入的斜率 60 对应未除 horizon 的局部输入曲率 `R0=.08*(.02*60)^2=.1152/rad²`，除以 176 后为约 `.000654545`；饱和/active bounds 处仍应按实际映射检查，不能把该标度误称每处的实际曲率。这可以提供工程标度，而不是原论文给定的 Rmin/Rmax；具体数值由主计划事先冻结。若直接照搬 Stand 的 `.02 rad` 参考，输入曲率会变成 `.08/.02²=200/rad²`，约为当前移动标度的 1736 倍，可能把移动锁住。

新增项必须同时进入 forward/native 总代价、残差导数、Riccati/部署度量和完整策略接纳。对冻结 reference/R、独立节点输入坐标，归约后的直接导数为 `Cu+=R*(u−ubar)/N`、`Cuu+=R/N`；若实现已在阶段导数外统一归约，不得再除一次。若实际动作映射由 active bounds/projection 耦合状态，仍须经过既有对应链式关系，不能一面评分 projected action、一面忽略其映射。仅在 Quu 加 scheduled D 属于优化正则，并不意味着 native 真 J 对该动作付出了代价。

判别结果需要并列报告原 J 各项、新 Jadd、该 native rollout 实际施加的 qtarget/预测力矩、支撑恢复、身体状态、原路线进度，以及误差正负方向下的改变；新版本的 actual plant 结果此时尚未产生。不能只看 K 小了或新 J 降了。若两个新增版本都仅把机器人驻留、保留了坏 incumbent、令原高度/进度持续恶化，或三版本都无法恢复，就应否定本轮固定实现；不得改 success 口径、继续扩 R 网格，或重复同类失败 E30 来制造样本数量。

只有有限证据显示有实际恢复改善，才冻结一个版本，在两个原 update57 策略上并行启动原 E30 及已定义的条件阶梯。它仍是新控制器资格实验，不是已有稳定性保证；若需要完整可行 walking reference，R8 成功轨迹的提取、相位/世界坐标对齐、状态与控制相容检查、材料在线适应都属于新的实质模块，不是复制一个 gait 参数就能完成。

## 5. 为什么并行应围绕具体工作，而不是占满显存

这三篇方法没有支持“显存闲着就把物理样本开到最大”的结论。25 核适合在隔离状态/模型下并行有限候选或时间节点导数；硬接触 CPU solver、JAX observer、GPU approximate proposal 的耗时性质不同。

W45 已显示 GPU proposal 的排序/原生可行性筛选仍不可靠。它不能越过 CPU native 裁决进入正式控制闭环。前述有限比较支持一个冻结分支后，两条 CPU 固定策略原任务按主计划并行；第三任务只做已声明的最多 24 个旧完整策略 scenario 的执行一致性诊断，不增加第三条 PPO 流。后台没有固定墙钟 kill；有限求解/问题数量和原物理失败依旧终止。GitHub 已核验交付后不额外在用户电脑归档。

## 6. 本次阅读边界

已核 R8 配置/实际 resolved 与成功/失败摘要；W45 实际 resolved、r34 spec；r34 重构的 quadruped residual、planner D/候选接纳、Riccati active-set 和 policy feedback/projection 源码。已深读上述三篇的相关方法/实现/限制，读取 HiLQR 基础代码和 IDTO solver 核心。没有运行论文代码，没有声称复现其硬件，没有读取所有历史原件每一个字节。

伴随的 `compare_r8_w45_resolved.py` 已只读运行，生成 `r8-w45-resolved-comparison.json`：19 组机器人/时钟字段逐值相等；记录两个输入 SHA256，列出 prediction numerics、首次 Trot 参数/权重、Stand reference 与核心源码 hash 的实际差异，并从 12 份 S3 结果重新汇总 single 9/9、long 3/3。MuJoCo 版本号相同只表示记录标签相同，不等于共享库或 ABI hash 相同；权重相同也不代表残差语义相同。脚本不运行 simulator、不连接服务器。

仓库前轮已经确认的原 MuJoCo 默认 PD60/5、dq_target=0、.35 s horizon、forward FD、TV-LQR 都应保留为已实现事实。下一步的研究价值应来自可信 nominal 与接触附近闭环有效性的新增证据，而非再次列出这些已有部件。
