# W44 后续机制选择：论文与作者实现核对

日期：2026-11-12。对应审查基线：`69d5a6ff2aa9803fddc8ac3a78a906008ea526c6`，控制器 `S5-r33`、W44。本文记录已实际读取的原文、官方代码和可迁移机制，不把论文实验成功当作本项目成功，也不把后续计划写成已执行结果。

## 1. 对 W45 的直接结论

**W45 首先修复完整策略的接纳语义，再评价原任务。文献支持把名义轨迹和反馈作为一个控制策略审查，但没有证明本项目拟定的接纳规则具有稳定性保证。**

W44 原始记录的审查已把问题收窄到具体事件：A 在 2.84 s 的 `alpha=0` 候选只有约 `8e-15` 的名义成本改善，却引入了从 scale 0 到 scale 1 的新反馈；B 在 3.04 s 的 plan 161/call 173 出现同类候选，名义改善约 `2.17e-12`，随后 3.052 s 出现很大的真实状态反馈。这里需要区别浮点级名义变化与实际策略变化：即使 feedforward 步长为零，替换 reference、K、投影度量 H 或 feedback scale，仍可能改变偏离名义轨迹时的控制。

当前审查已核对实际 Kδx、native/native successor 及 hold/clip 语义，未找到足以解释该事件的明显接口接线错误。因此本轮不把“重新统一接口”“加 PD”“加 TV-LQR”作为尚未落实的修复，也不继续用 PPO 扩样绕过这个已定位问题。

W45 的优先机制是：在实质平手时保留完整 baseline；`alpha=0` 不因名义浮点下降而替换 K/H/scale。先在四个既有问题上做有限反事实和偏差诊断，再使用 A/B 各自固定 update 57 策略评价原任务。该规则保护的是完整 incumbent，不等于禁止所有 `alpha=0` 改变，也不等于证明新策略在所有状态下更差。反事实结果和新的实际资格必须决定它是否有用。

“可信名义步态/动作先验”和“有限闭环采样”保留为接纳修复不足时的下一阶段方向。它们不应和当前最小修复同时堆入一个版本。

## 2. 原 MuJoCo 论文及官方 Go2 实现

### 2.1 文献与代码身份

**Whole-Body Model-Predictive Control of Legged Robots with MuJoCo**，John Z. Zhang 等；2025 年 arXiv 初稿，2025-10-18 v2。作者 publication 页面列为 ICRA 2026；早期项目页的 under-review 状态有滞后。

- [论文 v2](https://arxiv.org/html/2503.04613v2)
- [作者 publication 页面](https://johnzhang3.github.io/publications/)
- [官方项目](https://johnzhang3.github.io/mujoco_ilqr/)
- [官方部署仓库](https://github.com/johnzhang3/mujoco_mpc_deploy)
- [作者 MuJoCo MPC 的 Go2 分支](https://github.com/johnzhang3/mujoco_mpc/tree/go2)

官方代码确实公开；部署仓库仍注明 work in progress，机器人通信、定位与模型分支需要实际配置。公开源码可以用来核对实现，不能据此宣称本项目已复现硬件结果。

### 2.2 对本项目最重要的原文事实

§III-A Eq.(3) 使用 forward finite difference；§III-B Eq.(4)–(11) 定义 residual-norm 成本、Gauss–Newton 曲率和动力学 A/B，并返回名义轨迹及 TV-LQR。§IV-A 把 `impratio` 从 1 调到 100，处理接触滑移和动作抖动。§IV-B 将关节 PD 包含在动力学中，优化关节角目标。§IV-D 报告 centered FD 在其 locomotion 仿真中失败；负高度扰动进入接触的解释被作者明确列为假说。§V-A 默认 horizon 为 .35 s、动力学100 Hz、iLQR约50 Hz、TV-LQR约300 Hz。§VI 承认接触模式探索、初值和长时域单次射击的局限。

这些事实不支持把 `.35 s` 或零关节目标速度单独当作 W44 根因，也不支持因为反馈有益就接纳任何新 K。作者允许的 native 接触穿透经验不能用于放宽本项目 TM02 应变域。

### 2.3 原作者源码核对表

以下行号对应本次读取的官方公开分支，blob 标识列在表后。它们描述源码默认行为，不代替论文当年硬件运行时的 GUI 参数记录。

| 项目 | 已核实现 | 对本项目的含义 |
|---|---|---|
| 关节目标与 PD | `go2.xml` L13：gain 60、bias `0 -60 -5`；`go2interface.py` L56–57、L145–150：Kp60/Kd5、q=action、dq=0、tau=0 | 当前已有相同类型低层接口；不能把零目标速度本身当错误 |
| 时间 | `task_flat.xml` L10–11：horizon .35、dt .01；部署发送线程 `go2interface.py` L123 为2 ms | 当前500 Hz反馈、2 ms节点和10×.2 ms内部步已有明确实现；更细不自动保证更稳 |
| 接触参数 | `go2.xml` L4：elliptic、impratio100；L39–40：foot radius .022、solimp `.015 1 .022`、condim6、friction `.8 .02 .01` | 原作者模型和项目 G1/TM02 不是同一物理任务，不能复制其参数来放松真实模型 |
| 实际规划接触 | `agent.cc` L156–164 默认为 iLQG 开启 differentiable；L295–308 在规划前调用 MakeDifferentiable；`utilities.cc` L59–74 把 joint/geom/pair 的 solimp[0]置0 | XML 的 d0=.015 不是默认规划期间的 d0；作者默认 forward 与 FD 使用平滑规划模型，不能描述成 r33 的 native/native |
| 积分器 | `agent.cc` L97–98 默认继承 `model->opt.integrator`；本次读取的 task/go2/common XML 无显式 planner integrator 覆盖 | 论文未固定具体积分器，不把作者当年的实际设置断言为 Euler 或 implicitfast |
| 真正 trot 预设 | `quadruped.h` L88–96：duty .45、cadence2 Hz、amplitude .03 m、Balance .2/Upright1/Height1；Transition 会应用预设 | XML 初始 amplitude .06 不等于实际切换后的 trot 值；不能只比较初始 XML |
| Height | `quadruped.cc` L75–98：trunk 惯性中心z−四足几何中心平均z−.27 m | 不是 base 绝对高度；同名权重不能替代残差语义核对 |
| Gait | `quadruped.cc` L147–154：只在 swing 时惩罚脚高误差，stance 时该残差为0 | 抬脚残差不等于实际承重或完整可行 x/u 轨迹 |
| Balance | `quadruped.cc` L158–164：capture point−全部四足平均XY；绘图段注明 support polygon 未用于 cost | 论文 Table II 用 support polygon 概括，代码并非基于实际承重足的支撑域约束 |
| Effort/Posture | `quadruped.cc` L167–174：Effort=.02×actual actuator_force，Posture为关节角差；`quadruped.h` L133–134：abduction再×2 | 不是关节目标幅值成本；改变单位后不能直接照搬权重 |
| 范数 | `task_flat.xml` L132–138；`norm.h` L25–35、`norm.cc` 的 Norm | Height/Gait为smooth-abs，Position为向量smooth-L2；平滑参数与residual同单位，不是先除尺度再平方 |

范数具体为：type6 是逐元素 `Σ(sqrt(r_i²+p²)−p)`，type2 是 `sqrt(rᵀr+p²)−p`，type0 是 `.5 rᵀr`。残差、范数和单位共同决定局部曲率，不能只核对 weight 数字。

官方源码链接与读取到的 blob：

- [go2interface.py](https://github.com/johnzhang3/mujoco_mpc_deploy/blob/main/src/go2interface.py)：`e3b843b4b8885760f8d08406f8a324d8ed0e0aee`。
- [task_flat.xml](https://github.com/johnzhang3/mujoco_mpc/blob/go2/mjpc/tasks/quadruped/task_flat.xml)：`9917b3f1f5896036b74f3f92724e10a67d7034b7`。
- [go2.xml](https://github.com/johnzhang3/mujoco_mpc/blob/go2/mjpc/tasks/quadruped/go2.xml)：`110a8fdd27d2e33c454d863866b6a68ccf4f8577`。
- [quadruped.cc](https://github.com/johnzhang3/mujoco_mpc/blob/go2/mjpc/tasks/quadruped/quadruped.cc)：`a10767cab974c1e1998b3479d1b271bf69cd7575`。
- [quadruped.h](https://github.com/johnzhang3/mujoco_mpc/blob/go2/mjpc/tasks/quadruped/quadruped.h)：`6911ae5bf912e5d5dc0f98d24d42a6cb2706f772`。
- [agent.cc](https://github.com/johnzhang3/mujoco_mpc/blob/go2/mjpc/agent.cc)、[utilities.cc](https://github.com/johnzhang3/mujoco_mpc/blob/go2/mjpc/utilities.cc)、[model_derivatives.cc](https://github.com/johnzhang3/mujoco_mpc/blob/go2/mjpc/planners/model_derivatives.cc)。
- [norm.cc](https://github.com/johnzhang3/mujoco_mpc/blob/go2/mjpc/norm.cc)、[norm.h](https://github.com/johnzhang3/mujoco_mpc/blob/go2/mjpc/norm.h)。

## 3. 五篇补充论文：迁移机制及限制

### 3.1 Reduced-Order Model Guided Contact-Implicit Model Predictive Control for Humanoid Locomotion

Sergio Esteban、Vince Kurtz、Adrian Ghansah、Aaron Ames；2025年 arXiv:2502.15630。本次依据已读取的预印本，不另行认定会议版本。

- [论文](https://arxiv.org/html/2502.15630v1)
- [作者项目](https://rom-cimpc.github.io/)

§III-B Eq.(11)–(13) 用 HLIP/IK 构造 q、qdot 参考；§III-C Eq.(14)–(15) 根据目标和实测速度混合站立/行走参考；§III-E Eq.(18) 结合 feedforward torque 与 PD。§IV-D 表明 CIMPC 单独可能拖脚，但 combined 方法在速度成本太低时也可能停着。实验为 Achilles 仿真；§V 说明参考在扰动下也可能给错脚位。

**可迁移：**若 W45 接纳修复后仍有名义动作问题，可考虑完整启动 x/u 参考或 warm start，让局部优化从可执行动作开始。**不可直接迁移：**二足 HLIP 不能原样用于 Go2；一个参考不保证稳定，也不能把接触模式锁死后宣称仍在检验材料适应。

官方项目只确认论文入口，本次未确认整套 ROM-guided 实现的官方代码。底层 [IDTO](https://github.com/ToyotaResearchInstitute/idto) 公开不等于此整篇实现公开。

### 3.2 The Surprising Effectiveness of Linear Models for Whole-Body Model-Predictive Control

Arun Bishop 等，Humanoids2025。

- [论文](https://arxiv.org/pdf/2509.17884)
- [会议 DOI](https://doi.org/10.1109/Humanoids65713.2025.11203045)
- [作者项目](https://linearwalking.github.io/)
- [官方代码](https://github.com/LinearWalking/LinearWalking)

§III-A Eq.(7) 包含低层 PD；§III-B Eq.(8) 使用固定站立点线性化与 backward Euler；§III-C 规定接触日程并约束 stance 足位置、摩擦与力矩；§III-D Eq.(11) 跟踪完整状态、控制和接触力参考。§IV 离线生成可行步态，在线 OSQP；Go2 使用 .2s horizon、.01s knot、约500Hz闭环。§V 的边界包括大扰动和 QP infeasibility。

**可迁移：**完整可行 nominal、约束和高速反馈能支持短 horizon；这反驳“延长 .35s 就一定能解决”的简化判断。**不可直接迁移：**固定接触模式和线性模型不能替代 TM02 历史物理。公开仓库有 MuJoCo notebook、模型和预生成 H5 数据；本次未确认完整 Julia 参考生成链公开。

### 3.3 Reference-Free Sampling-Based Model Predictive Control

Schramm、Fabre、Perrin-Gilbert、Carpentier；2025年 arXiv:2511.19204，IEEE 与作者主页确认 ICRA2026。此文不是2023年题名近似的另一篇 Reference-Free MPC。

- [论文](https://arxiv.org/html/2511.19204v1)
- [IEEE 会议版本](https://ieeexplore.ieee.org/abstract/document/11696779)
- [Pierre Fabre 作者主页](https://pierfabre.github.io/)
- [作者论文页](https://fabinsch.github.io/publication/2025-09-15-mppi)

§III-A Eq.(1)–(2) 用 Hermite 联合表示 q/qdot 并限制变化速率；§III-C Eq.(10) 用 q/qdot 目标 PD；§III-D 执行实际最优 rollout，均值只作下一轮搜索中心；§III-E 处理真实计算延迟。§IV-B 用约 .9s horizon 发现动作。§IV-C walking ablation 为 Hermite10/10、无velocity cubic6/10、quadratic0/10；best 与 nominal 执行在 walking 都是10/10。

**可迁移：**有限、平滑、可执行的动作候选，以及对真正部署候选的验证。**边界：**最优仿真样本不保证真实安全；这里没有验证 W45 的 K/H/scale 平手保留规则。若新增 qdot 动作，必须同时修改 forward/FD/feedback/actuator，不能仅改变执行端。文中约27%的数字是新解未胜过旧移位解的频率，不是少27%跌倒。未确认公开官方实现，不能承诺直接 clone 复现。

### 3.4 Hybrid Feedback Sampling for Sample-Efficient Model Predictive Control

Chaoyi Pan、Zeji Yi、John Z. Zhang、Zachary Manchester、Guannan Qu、Guanya Shi；2026-08-19 arXiv:2608.19443。作者 publication 页面仍列 under review。

- [论文](https://arxiv.org/pdf/2608.19443.pdf)
- [作者 publication 页面](https://johnzhang3.github.io/publications/)
- [Chaoyi Pan 作者主页](https://panchaoyi.com/)

§IV-A Thm.1 分析开环不稳定给纯 MPPI 采样带来的困难；§IV-B Eq.(4) 在反馈闭环内采样并使用 Riccati 曲率塑形噪声；§IV-C Eq.(6) 混合局部反馈和全局 proposal，避免仅追随坏局部解。§V-D 实验采用8局部+24全局样本。线性理论不能直接推出非线性材料机器人稳定；§V-C 仍指出估计/接触延迟瓶颈。

**可迁移：**若完整策略接纳修复后仍陷坏接触局部解，可复用已验证 K，在候选 rollout 内探索少量结构化轨迹。**边界：**它不证明当前任意 K 都值得安装，也不提供 W45 自定义接纳规则的稳定性定理；32样本不是本项目算力保证。未确认公开官方代码，属于后续机制参考。

### 3.5 Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing

Haoru Xue、Chaoyi Pan、Zeji Yi、Guannan Qu、Guanya Shi；2024年 arXiv:2409.15610，ICRA2025。

- [论文](https://arxiv.org/html/2409.15610v1)
- [作者项目](https://lecar-lab.github.io/dial-mpc/)
- [官方代码](https://github.com/LeCAR-Lab/dial-mpc)

§III-C Algorithm1/Eq.(7) 使用迭代和时域双重退火，近期动作低噪声、远期保留探索。§IV 用2048样本、.4s/20节点；Appendix A-B 仍有 trot 脚高参考。Appendix A-A Eq.(8) 给 torque command 加 `.65×joint velocity` 阻尼，模型与执行都包含；planner50Hz、命令200Hz，并为加速关闭非足碰撞。

**可迁移：**把探索限制在平滑、时序合理的候选族，而非任意全关节白噪声。**边界：**不能关闭本项目非足接触、改真实阻尼或在候选之间共用材料历史。官方代码确实公开仿真和异步 Go2 硬件接口，依赖 CUDA/JAX；整套 GPU 移植不是当前策略接纳问题的最小修复。

## 4. 机制排序及本次明确不做的推论

| 优先次序 | 本项目问题 | 采用的机制 | 证据边界 |
|---|---|---|---|
| W45 当前 | 浮点级名义改善伴随完整反馈策略替换 | 平手保留完整 baseline；统一审查 reference/K/H/scale；四个既有问题有限反事实 | 来自 W44 记录及代码分析；论文只提供名义/反馈与部署候选应一致的背景，不提供稳定性证明 |
| W45 验证 | 修复是否改善真实闭环和原任务资格 | A/B 各自固定 update57，原任务及其分级评价 | 不用新增 PPO 更新掩盖控制器变化；新结果决定是否继续 |
| 若仍存在坏 nominal | 低层局部搜索起点不能形成可信支撑交换 | 可行启动 x/u 参考或 warm start，保留现有在线优化 | ROM-guided/LinearWalking 支持方向；具体 Go2 参数须由本项目证据决定 |
| 若可信局部起点仍陷坏 basin | 接触模式局部搜索不足 | 一个有限的平滑闭环候选族，再用原生模型评价 | RF/FS/DIAL 提供机制，不等于应同时换 planner、动作空间和GPU后端 |

读记录时另有两个必要区分。第一，raw `Kδx` 不等于实际施加的动作，后者还受当前 target/torque 约束与投影影响。第二，trot duty=.45、2Hz、两对腿相差半周期，本身允许短时全足 flight；瞬时无足接触不能独立定义失稳，需与私有相位、预测接触、持续时间和机体状态对齐。不能把这两种指标直接写成根因证据。

## 5. 保留材料记忆科学主线

本轮收窄到策略接纳，是为了得到可解释的闭环基线。最终目标仍是原 Go2/TM02 完整任务，以及材料历史预测是否改善实际接触决策；不是只让机器人完成几秒启动。

后续控制器必须保留世界固定材料带、真实应变域、力矩和接触规则、完整足迹带内实际加载，以及 E30→E60→E300→E600 的原资格。新版本不能继承旧版本的任务通过结论。完整 nominal 可以作为初始化或软参考，不能把未来动作固定到失去材料适应空间。

待闭环基线可信后，再检验外层 cap 的实际权限与因果材料特征是否改变决策；RAW/PANN 比较需要匹配控制器、训练预算和任务。观察器保持因果，候选材料历史独立推进，不向 policy 注入真实内部材料状态。这样才能回答长期问题：相同瞬时观测、不同材料历史时，记忆预测能否带来可重复的闭环收益。

## 6. 阅读覆盖及未完成事项

本次已读上述六篇的相关方法、实现、实验与限制原文；核对原作者 Go2 部署接口、模型/task XML、quadruped residual/transition/header、norm 和 Agent/MakeDifferentiable 实现；用官方论文、会议或作者页面确认身份与状态。未运行各论文代码，未把源码存在等同于复现成功。对 ROM-guided、RF-MPC、FS-MPC，表述为“本次未确认公开官方实现”，不推断全网绝不存在。

本文是机制评审与来源记录。W45 接纳改动、反事实及原任务评价均需由对应提交和实际结果单独记录。
