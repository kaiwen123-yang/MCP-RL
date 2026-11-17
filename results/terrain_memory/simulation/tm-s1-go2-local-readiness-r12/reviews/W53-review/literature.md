# W54 文献与官方实现复核：把“能走”推进到“可持续、接触可执行”

审阅日期：2026-11-17。任务基线：项目最新结果 `c01ed82d241b559e4d6c798f06bb0483c622b422`。本报告只研究论文与官方源码，没有运行物理实验、训练、安装控制框架，也没有修改项目仓库。

## 1. 对下一轮最有影响的结论

**主线应保留 W53-N 已恢复的行走能力，集中修正首次材料落足的冲击与执行层适配；F 线应成组修复已定位的接触计划、实际承载和未来执行可行性问题。现在把全部控制栈换成 AugMPC 或 Quadruped-PyMPC，收益不如修复已经有直接证据的故障。** GPU 第三线可以升级为已有状态上的完整 SRB 力序列求解、选择和独立复算，不应继续只做若干大矩阵吞吐量演示，也不应未经适配就启动数百万步新 RL 训练。

本报告采用主审与控制审查员传来的 W53 结果作为项目事实，不把它们冒充本报告独立重算的结论：N 的最大 x 为 0.723845 m，6.4656 s 因材料主应变 0.020178 超过 0.02 结束；FL 在 6.4640 s 首次落到新地块时法向速度约 −0.9558 m/s，随后 8 个 0.2 ms 子步压入约 0.99445 mm、承受约 1194.56 N，下一候选压入约 1.00891 mm。F 真实落足晚于计划 74–114 ms，50 ms 固定重叠不足，并有当前静态执行界复制到全部未来节点、未实际接触即冻结 stance anchor、删除未接触计划力后改变机身加速度参考、绝对时钟继续卸载其他支撑、未来落足回退旧 anchor 等明确代码问题。最终项目报告应以原始日志与项目代码为这些事实的引用。

由这些证据得出的工程判断是：

1. N 不是还在寻找一个完全不会走路的策略。一次已恢复的行走在材料边界处暴露了具体的高速落足问题，值得保留其基线能力。
2. 1.6 ms 内发生的冲击不能只靠低频上层标量或触地后的模型更新处理。应先改变触地前下降速度/轨迹与执行层响应，再处理接触后的承载跟踪。具体限速值必须由项目执行与材料模型确定，本报告不凭论文给出未经验证的数值。
3. F 的主要问题不要求先训练另一个 RL。必须使计划接触、实测接触、可承载接触分开表示，并让交接规则、脚位预测、力分配和 WBC 使用一致的状态。仅把 50 ms 改成更长固定重叠无法解决反馈缺失。
4. 文献可以提供模块接口和验证方法，不能替当前项目宣布“稳定性保证”。所有下述论文均未验证项目的 TM02 材料历史、子步应变域及原始执行器合同。

## 2. 用户点名的 AugMPC：有价值，但不是现有标量策略的即插即用替代品

主论文：[RL-Augmented MPC for Non-Gaited Legged and Hybrid Locomotion, arXiv:2603.10878v1](https://arxiv.org/html/2603.10878v1)。官方入口：[AugMPC](https://github.com/AndrePatri/AugMPC)、[IBRIDO](https://github.com/AndrePatri/IBRIDO)、[AugMPCEnvs](https://github.com/AndrePatri/AugMPCEnvs)、[MPCHive](https://github.com/AndrePatri/MPCHive)、[模型目录](https://huggingface.co/AndrePatri/AugMPCModels/tree/main)。已读论文模型、接触注入、执行、训练、实验和局限章节，以及下列源码路径；没有下载运行权重或完成 Go2 移植。

### 2.1 论文架构和必须保留的边界

论文用 SAC 决定基座六维运动参考及各足的飞行阶段注入；下层全刚体逆动力学 MPC 规划状态、加速度与接触力，再提供关节参考和前馈。接触切换在未来节点注入，为 RTI 求解预留时间。GPU 支持矢量化仿真与学习，CPU 并行 MPC。实验平台不是项目当前 Go2/TM02 组合。论文明确未施加关节力矩限制，并讨论 action repeat 不合适时可能学成站立局部最优。因此其结果不能作为放宽项目力矩界、直接载入现有权重或跳过下层稳定性的理由。上述均是论文层面的事实；下面的更细接口来自独立官方源码审查。

### 2.2 官方当前源码到底提供什么

源码固定版本：

| 仓库 | 本次源码 pin | 已检查范围 |
|---|---|---|
| AugMPC | `d695aeecd61bed5f19b333b41005d7edb72096b3` | README；训练入口；接触参考和 gait manager；MPC q/v/a/effort 输出；关节阻抗执行 |
| AugMPCEnvs | `cac164c8eba1e060ea506659bd90371834e8cf86` | README；twist tracking 的动作、接触标志、MPC 观测、共享内存交互 |
| MPCHive | `9791f8b6215dbb044431cceaa95c4745aa2aadb9` | README；控制器集群与共享内存边界、进程/资源管理接口 |

**算法身份。** `aug_mpc/scripts/launch_train_env.py:326–345` 分别选择 HybridSAC、SAC、PPO；直接入口未传 `--sac` / `--hybrid_sac` 时走 PPO。论文使用 SAC；仓库支持 PPO，并不意味着用户现有 PPO 的观测、动作和 checkpoint 可以直接延续。源码存在 `training_algs/sac/sac.py` 与 `training_algs/ppo/ppo.py`，要记录真正启动参数与权重 schema，不能只记录仓库名。

**动作权限。** `AugMPCEnvs/aug_mpc_envs/training_envs/twist_tracking_env.py:439–461,710–748` 将策略动作转为逐足 contact flags，再写入 MPC 共享参考；同文件 `820–823` 还读取 MPC 接触力/接触位置作为观测。它具有改变接触阶段的权限，远超过只调一个全局参数。若本项目决定走这条路线，需要新的 action/observation/version 合同与独立训练，而不是把现有低维策略换个模型文件名。

**接触注入。** `AugMPC/aug_mpc/controllers/rhc/horizon_based/hybrid_quad_rhc_refs.py:165–233` 处理 contact flags；`gait_manager.py:342–490` 在固定未来位置添加飞行阶段，并设置足轨迹参考，`451–468` 可添加后续 touchdown phase。这里的未来注入是优化预见机制，不能自动视为“已经检测到实际落足，才允许另一腿离地”。项目 F 的迟落足交接必须另有实际承载闭环。

**规划到执行。** `hybrid_quad_rhc.py:652–678` 输出关节速度、加速度并通过 `eval_efforts_on_node()` 获取前馈力矩；`1313–1334` 返回 q/v/a 解。`utils/jnt_imp_control_base.py:392–429` 的直接力矩分支显式组成

\[
\tau=\tau_{ff}+K_p(q_{ref}-q)+K_d(v_{ref}-v),
\]

再对完整结果施加 limiter 后送执行器。对项目最有用的是这条完整合同，而不是套用论文的增益。把其他控制器的完整总力矩送入原有带 PD 的位置执行器时，必须按原合同做逆映射，避免重复叠加反馈。此处描述的是未滤波显式覆盖分支，不宣称所有配置分支都等价。

**并行。** MPCHive 的基本资源是 MPC 控制进程/任务与共享内存；其状态、参考、预测和 health/status 结构是可借鉴接口。AugMPCEnvs README 描述的 Isaac Sim 矢量环境与 XBot2/MuJoCo 接口不是项目已有材料仿真的 GPU 移植。96 GB 显存不会自动使 CPU 上的全刚体优化或 TM02 材料积分变成 GPU 计算；25 核如何映射多少 MPC/仿真实例仍需在本机测量。

### 2.3 本轮可借与后置事项

本轮可借：显式接触日程接口；足轨迹的完整 p/v/a 参考；未来接触变化提前通知求解器；上一解平移 warm start；记录 MPC 求解/约束健康状态；保持前馈与反馈总力矩合同；以每个任务自己的状态、日志和种子隔离并行。

后置：完整 AugMPC 训练栈、Go2 全刚体模型/初始姿态/关节顺序/接触 frame 接入、原执行器限制、TM02 接触/地块记忆观测、材料环境矢量化、全新 RL 动作和奖励。等下层在目标材料上产生可重复的行走和接触数据之后，再评估“RL 管步态/落足/参考、MPC 管动力学”的长期路线，价值会比现在启动更多失败样本训练高。

## 3. 与 N/F 具体问题最匹配的新论文

以下不是论文数量清单。每篇只保留能改变下一步实现或解释边界的内容。

### 3.1 冲击建模：IRCMPC，RA-L 2026

[Impact-Aware Robust Convex Model Predictive Control for Quadruped Locomotion on Uncertain Terrain](https://doi.org/10.1109/LRA.2026.3692086)，IEEE RA-L 11(7):7820–7827，2026-05-11 发表。官方摘要/引言可检索；直接打开 IEEE 全文时遇到验证页面，本次未读到全文，也未找到经确认的官方代码。

论文把地形冲击表示为 SRB 有界不确定性，用动量构造逐步冲击上界，并用松弛因子降低不必要的保守性；稳健 min–max MPC 化为 QP，在 Go2 仿真和硬件进行比较。这说明“落足冲击应进入预测问题而非仅在失败后调权重”是有直接近期研究支撑的方向。**不能从摘要推导其具体动量公式，更不能声称它约束了 TM02 应变或消除了首次冲击峰值。**

对本项目的建议是工程推论：先在现有 N/F 中记录并约束触地前法向速度与接触力增长，保留扰动/冲量项的接口；暂不按摘要重造一整套所谓 IRCMPC。机身稳健跟踪与足下材料局部峰值是不同指标，单纯提高机身抗扰并不必然保护材料。

### 3.2 接触执行：MPC + 自适应阻抗/导纳，2026，有源码

[Terrain-Adaptive Quadrupedal Locomotion Using Model Predictive Control With Impedance Control and SINDy Model Enhancement](https://onlinelibrary.wiley.com/doi/full/10.1002/rnc.70700)，2026-08-14；[作者代码](https://github.com/pjmcconnellogue/Terrain-Adaptive-Quadrupedal-Locomotion)。已读全文核心 §§2–5 与局限/结论，源码 pin 为 `3af92ce668c579c093bc269dc3bfd68996ce59fa`。

论文将 MPC 期望力与高频接触反馈结合；四态接触依据实测力切换阻抗/导纳，SINDy 只离线校正角动力学。验证为 PAWS/PyBullet 仿真，整体级联没有另行证明全局稳定性。**§5.4 明确说明首次触地仍有无法立即补偿的力尖峰，导纳改进主要发生在尖峰之后。** 所以它支持接触后的执行适配，但不能替代 N 的触地前减冲击。

独立源码证据与可迁移机制：

| 官方文件 `main.py` | 实际机制 | 本项目可用的部分/不能照搬的部分 |
|---|---|---|
| 279–295 | 足位置/速度误差变为纠正力，再经 Jᵀ 变力矩 | 可借真实执行闭环，必须核对世界/机身坐标、符号与总力矩限制 |
| 375–419 | 实测力与期望总力比较；接触进入/退出具有不同阈值 | 可借接触滞回与轻触/承载区分；零期望力及噪声场景需项目定义，不能直接复制比例阈值 |
| 424–459 | 接触状态驱动 anchor 变化、导纳积分和限幅 | 可借有界积分与接触事件重置；原代码 50 mm offset 与当前工况约 1 mm 的材料有效位移尺度不相容 |
| 467–492 | MPC 前馈力矩加阻抗反馈后输出完整力矩 | 可借完整执行合同；不允许只约束前馈、漏掉反馈的饱和 |
| 55–56、436、683–704 | MPC 更新与子步执行分开 | 可借不同时间尺度；不能承诺其仿真频率就是本项目响应能力 |

当前代码的 anchor 更新/进入 solid 时重置与论文固定初始 anchor 的描述不完全相同，而且这段逻辑仍由 scheduled stance 进入，不能声称作者已替本项目实现迟触地支撑门。应借小模块的机制，不整体搬运另一套仿真和控制栈。

### 3.3 模型残差：RK-MPC，2026 预印本

[RK-MPC: Residual Koopman Model Predictive Control for Quadruped Locomotion in Offroad Environments](https://arxiv.org/html/2604.04221v2)，[作者项目页](https://sriram-2502.github.io/rk-mpc/)。已读模型、数据和实验段；2026-04 预印本，不冒称已发表期刊。作者项目页代码按钮目前写“coming soon”。

它在名义 SRB 上学习残差，仍用 QP MPC；数据来自已经能够运动的名义控制器，含多种地形/速度条件，并非仅靠短失败前缀。对本项目的价值是把“稳定后仍有系统性模型偏差”作为可学习对象。当前 N 材料暴露很短、首落足就触域，首先需要执行/接触修正和足够的数据覆盖。角/速度预测误差下降不等于材料峰值受控；本轮不把训练 Koopman 模型列为必须先完成的重任务。

### 3.4 GPU 求解：OT-MPC，2026 预印本

[Sampling-Based Control via Entropy-Regularized Optimal Transport](https://arxiv.org/html/2605.02147v1)，[作者项目页](https://acdslab.github.io/ot-mpc/)。已读算法、Go2 场景与计算开销。OT 的有用点是保留多个控制候选模式，避免把不同有效模式平均成差动作。论文的 Go2 仿真用 MuJoCo rollout；报告的约 0.2 ms 是特定 Sinkhorn/重心计算开销，不能当成完整控制步包含物理 rollout 的时延，更不能作为项目 GPU 加速比。

当前 Quadruped-PyMPC 已提供 SRB/JAX 上的 OT-MPC 实现，可作为第三线完整优化的候选。这是该官方仓库的适配实现，不宣称它与论文所有动力学/调参/计算预算相同。当前主要失败并非已证实是采样模式平均导致，因此不应把换 OT 当成本轮主修复。

### 3.5 对“稳定性有保证”的补充边界

[Adaptive Non-Linear Centroidal MPC With Stability Guarantees for Robust Locomotion of Legged Robots](https://arxiv.org/html/2409.01144v2)（RA-L 2025；[官方代码](https://github.com/ami-iit/paper_elobaid_2024_stable-centroidal-mpc)）在特定扰动和可行性假设下组织自适应/稳定性约束。已核查其假设与实验，不把连续或常值扰动结论推广成任意短时冲击、材料域或离散执行的保证。它适合在名义行走成立后考虑系统性误差，与当前“让落足真正可执行”的工作互补。

## 4. Quadruped-PyMPC：可以复用求解器，不能假设已解决 F 的接触问题

官方：[iit-DLSLab/Quadruped-PyMPC](https://github.com/iit-DLSLab/Quadruped-PyMPC)。源码 pin：`5831ef2647bfaa3092a4d12ad3c4ff48989c3fb6`。已读 README、Go2 配置、MuJoCo simulation 入口、wb interface、swing controller、early stance detector、wrapper 与 JAX 主要求解路径。没有安装或运行，不能声称本机复现了 README 视频。

### 4.1 Go2 支持真实存在，但执行合同不同

`quadruped_pympc/config.py` 的默认机器人为 Go2，并提供质量/惯量、步态和足轨迹参数。`interfaces/wb_interface.py:395–423` 对 stance 使用腿 Jacobian 转换 GRF，对 swing 使用 Cartesian 误差、腿质量矩阵、Jacobian 和 bias 项。`455–478` 还给出 IK 位置/速度参考。其接口名含 wb，不等于它已经是一套与项目当前 floating-base WBC QP 等价的控制器。

`quadruped_pympc_wrapper.py:222–228` 的关节 PD 追加项是注释，`270` 返回总 torque。配置中的关节 PD 数值不代表该仿真链路实际重复加了一遍 PD。项目必须对照原执行器把总力矩正确送进去；例如在原合同为 `tau=60*(u-q)-5*v` 的情况下，总力矩到位置输入的代数映射为 `u=q+(tau_des+5*v)/60`，仍需满足原有输入/力矩界。这是合同推导，不是已完成移植。

### 4.2 接触和落足：最不能误判的地方

`interfaces/wb_interface.py:215–243` 由 periodic gait 产生 contact sequence，第一列给 current_contact；liftoff/touchdown 更新依赖这些计划边缘。`383–387` 调用可选早触地检测。`helpers/early_stance_detector.py:24–25` 对 sampling 模式明确关闭该功能；其余分支主要依据轨迹误差/几何接触修正早碰撞，未实现本项目所需的“迟触地未达到承载条件，就不卸载剩余支撑”的完整交接逻辑。`helpers/swing_trajectory_controller.py:119–164` 的 swing time/touchdown 检查也依赖计划接触序列。

因此，不能以“官方 Go2 能走”推出“移植其调度器会自动消除当前 74–114 ms 延迟”。当前 F 的几个可定位逻辑错误应一次成组修；官方接口可用于交叉检查脚轨迹/力的语义，但全栈移植会同时引入模型、frame、关节序、执行器、初始化和 TM02 环境适配工作。

### 4.3 JAX 真实计算范围

`controllers/sampling/centroidal_nmpc_jax.py:96–125,208–210` 支持 random sampling、MPPI、CEM-MPPI、OT-MPC，并 JIT/vmap 整段候选 rollout。`282–330` 投影单边法向力和摩擦约束；`332–513` 完成完整 horizon 动力学累计代价；`552–573` 返回首步 GRF；`1081–1180` 是完整 OT 更新、选择与首步输出。这比单独跑矩阵乘法具有明确的求解任务。

但必须同时看到限制：

* `centroidal_model_jax.py:93–174` 为 SRB 状态、外给接触掩码与固定足位置的 Euler 积分，非 TM02 材料仿真，也不含完整关节动力学。
* `centroidal_nmpc_jax.py:471–501` 的输入正则项在当前代码被三引号注释，当前 rollout 代价主要是状态误差。不能把该默认目标当成当前项目带力变化/执行限制的同一问题。
* contact mask 仍是外部给定；约束投影不能让一只悬空足产生实际承载。
* 对于 GPU 声明，必须检查实际设备放置、JIT 首次编译与 warm solve、同步后的总时延。代码存在 CPU fallback；源码可用不等于本机 CUDA 接口已经验证。

## 5. 建议的双物理任务 + 完整 GPU 求解任务

### N：保留已行走基线，完成材料落足适配

把触地前末端下降轨迹、法向速度、执行力矩变化和接触后的承载跟踪放在同一修改包内。保留已恢复的前进能力，避免为了抑制冲击又退化为不抬腿/不前进。输出应同时展示：首次材料触地前后足位/足速、实测/目标力、总执行力矩及饱和、主应变和材料状态、持续前进和完整任务终点。只增加 run 存活时间却未真正进入材料或行走，不算解决。

主审补充的历史证据表明，旧 R10 已尝试近地 0.05 m/s 软惩罚、局部弹簧及材料支撑高度，r3 曾运行到 32.567 s 后在回程失败。因此本建议不把这些旧项再加一遍当成新方案；重点是把触地前下降条件变为真实有效的约束或经过执行跟踪验证的轨迹条件，并将触地检测、anchor 更新、承载建立和下一次卸载纳入同一事件逻辑。软代价里的目标速度与真实足速不一致时，必须沿优化输出—执行器—足运动定位偏差，而不能仅提高惩罚权重或降低名义数字。

工程上可采用“先减小落足动量，再用可执行的局部反馈保持承载”的层次。不是简单把足轨迹末端速度标成零就宣布成功；必须检查实际足速与原执行器输出，保证触地事件到来时实际状态符合设计。相邻地块的落足高度和材料有效域应使用项目已有允许观测/估计，不引入只有 evaluator 才知道的隐藏真值。

### F：一次修复一致性链路，避免再拆成五轮

合并处理未来节点对应的状态/约束、未来足位递推、实际落足 anchor 更新、实测支撑决定的相位交接，以及 WBC 的参考/力分配一致性。对 swing，不能要求不存在的接触力补偿当前静态 bias；对未来 stance，不能沿用当前 q/v 的执行可行界而不说明近似。更不能删掉未接触计划力后把机身目标加速度改成自由下落，再把数值失败归咎于“优化器不稳定”。具体实施以控制审查员给出的项目行号和证据为准。

### C/GPU：完整求解和独立裁决，服务 N/F

可以直接采用 PyMPC 的 JAX rollout/优化骨架，但任务边界应设为：

1. 输入已保存的实际状态、足位置、可用接触/候选相位、质量惯量和参考；明示缺失字段与近似，不杜撰完整 q/v。
2. 在同一已定义 SRB 问题上执行全 horizon 动力学、接触/摩擦/力约束、目标及力变化处理；得到整条候选力序列和首步动作。
3. 用 CPU 独立复算每个节点的目标值与可行性，标出“SRB 可行但原执行器/TM02 尚未验证”的边界。若拿不到完整未来关节状态，不宣称完整 WBC 可行性。
4. 在相同输入、相同候选/计算预算下比较现有 CPU 求解与 GPU 方案，给出同步后的 end-to-end 时延、显存、编译开销、解质量与有效候选比例。输出能解释 N/F 的输入，而不是仅证明 GPU 忙。
5. 只在完整离线候选确实有增益且合同一致后，再由后续在线验证把它接入物理主线。本轮不要再新增第三个重物理控制家族与两条主线竞争资源。

若当前 CPU QP 很小且能稳定满足控制步，强行搬 GPU 可能不划算。GPU 的真实价值更可能来自多状态、多候选接触时序、长 horizon 的并行完整求解；应以质量/时延实测决定，不以占满 96 GB 作为成功指标。该建议不需要为后台设统一墙钟完成时间；应以任务完成、有效失败证据和资源异常条件管理，日志/结果推送 GitHub 即完成交付，不添加用户电脑归档步骤。

## 6. 一手来源与阅读范围

| 来源 | 原始链接 | 阅读边界 |
|---|---|---|
| AugMPC 论文 | https://arxiv.org/html/2603.10878v1 | 模型、接触、训练、实验、限制；未在本机复现 |
| AugMPC 官方实现 | https://github.com/AndrePatri/AugMPC | 固定 pin 的动作/接触/执行源码 |
| IBRIDO | https://github.com/AndrePatri/IBRIDO | 官方架构入口 |
| PyMPC | https://github.com/iit-DLSLab/Quadruped-PyMPC | 固定 pin 的 Go2 执行链、接触表与 JAX 优化 |
| IRCMPC | https://doi.org/10.1109/LRA.2026.3692086 | 官方摘要/引言；未取得全文或经确认的代码 |
| MPC+阻抗/SINDy | https://onlinelibrary.wiley.com/doi/full/10.1002/rnc.70700 | 核心全文，特别 §5.4 首次峰值限制 |
| 上述作者代码 | https://github.com/pjmcconnellogue/Terrain-Adaptive-Quadrupedal-Locomotion | 固定 pin 的 main.py 控制链 |
| RK-MPC | https://arxiv.org/html/2604.04221v2 | 方法、数据、实验；预印本 |
| RK 作者项目 | https://sriram-2502.github.io/rk-mpc/ | 代码入口本次仍为 coming soon |
| OT-MPC | https://arxiv.org/html/2605.02147v1 | 方法、Go2 和计算开销；预印本 |
| OT 作者项目 | https://acdslab.github.io/ot-mpc/ | 作者补充资料 |
| 自适应稳定 MPC | https://arxiv.org/html/2409.01144v2 | 假设、方法与实验边界 |

固定版本源码链接可由以下前缀加表中路径定位：

* `https://github.com/AndrePatri/AugMPC/blob/d695aeecd61bed5f19b333b41005d7edb72096b3/`
* `https://github.com/AndrePatri/AugMPCEnvs/blob/cac164c8eba1e060ea506659bd90371834e8cf86/`
* `https://github.com/AndrePatri/MPCHive/blob/9791f8b6215dbb044431cceaa95c4745aa2aadb9/`
* `https://github.com/iit-DLSLab/Quadruped-PyMPC/blob/5831ef2647bfaa3092a4d12ad3c4ff48989c3fb6/`
* `https://github.com/pjmcconnellogue/Terrain-Adaptive-Quadrupedal-Locomotion/blob/3af92ce668c579c093bc269dc3bfd68996ce59fa/`

源码固定版本用于本轮审查。本报告没有把“克隆了源码”描述为“逐字读取了整个仓库”，也没有把阅读论文/代码描述为实验复现。
