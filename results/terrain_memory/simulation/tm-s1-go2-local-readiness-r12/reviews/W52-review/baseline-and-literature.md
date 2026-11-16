# W53：恢复已验证运动内核与接触一致动力学方案

只读审查日期：2026-11-16。仓库 `/workspace/scratch/f55a41043871/MCP-RL`，HEAD `5e036c6918ae72bb795c4b1b5be08c1e109ade32`。本报告未修改仓库，未编译，未启动任何物理 episode。文中的下一轮方案是审查建议，不能当作已实施或已通过资格。

## 1. 结论

**R8/S3 的正常运动确实有可追溯的源码、配置和完整任务证据，值得恢复为独立运动基线。它不能被表述成当前 TM02 材料任务已经合格的控制器。** 成功记录对应原 Go2、同一 PD 60/5 与限力、同一 500 Hz 回授和 50 Hz 重规划。旧运动内核与当前 r40/r41 的差异不是只有 30 mm 对 5 mm 步高：预测接触与积分、运动目标、输入参考、投影与候选接受链均已改变。

**并行的结构性方案应集中成一个“接触状态—力—运动共同生成”的控制族。** 足端几何可以是这个控制器的输入，但不能继续只把 qIK 差分叠到承重中心上，然后假定卸载和身体支撑会自行出现。可执行参照是 SRBD MPC + WBIC/WBC：规划支撑力，按照同一接触集合生成摆腿位置/速度/加速度，再用全身动力学得到关节力矩与关节参考。该方案在本项目还没有实现，更没有 TM02 成功保证。

R8 恢复和动力学方案回答不同问题：前者尽快建立正常运动的实证基线；后者改变目前几何、支撑与力缺乏协调的结构。不要把两者混为一次只改步高的小消融，也不要用不同目标函数的 J 数值直接宣称优劣。

## 2. R8/S3 的成功到底是什么

### 2.1 可核查的实际任务结果

原 R8/S3 刚地单次任务 9/9 完成，每次完整观察 60 s；条件为 nominal、yaw ±5°、vx ±0.05、vy ±0.05、wz ±0.1。目标是出发走至约 3 m、转向、回家并稳定站立，而非仅保持不跌倒。nominal 首次完成为 36.378 s；单次任务全部完成时间约 36.292–44.136 s。

连续 300 s 的 nominal、yaw ±5° 三个任务全部完成 5 个往返，首次完成各为 214.356、216.564、218.454 s，之后仍观察至 300 s。三条轨迹最大 x 约 3.021–3.028 m，最低机身高度约 0.26560、0.26804、0.27319 m。nominal 单任务末段 40–60 s 四足持续接触，速度极小。因此旧成功确实含持续行走、掉头、回家与停稳。

证据：

- `results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/index.json` 的 `formal_coverage`、`formal_cohorts`、`campaign_outcome`、`project_long_last30_native_support`。
- `results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/project-s3-single-05.json`。
- `results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/project-s3-long-06.json`。
- 同目录 `project-s3-formal-v1-single-*.json`、`project-s3-formal-v1-long-yaw_plus5.json`、`project-s3-formal-v1-long-yaw_minus5.json`。
- `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review/r8-w45-resolved-comparison.json` 的 `r8_s3_results` 汇集了逐条文件、哈希和结果。

这是刚地的资格。R8 没有 TM02、没有 PANN/PPO 材料收益、没有 NX 实时或硬件资格。R8 共 19 次物理启动中包含负结果与开发阶段，不能把 9/9 与 3/3 误写成全部 19 次均成功。R8 也不是当前控制修订名中的 `S5-r8`。

### 2.2 原 S3 修复只在 Stand 生效

三个上游补丁实际保留作者运动逻辑。`s1-go2-wb-support-upstream.patch` 的支撑判据是 Quadruped、Stand 且 userdata 支撑参考激活；运动阶段重新使用作者的 actuator-force effort 与 home posture。S3 的修改用于让回家后的站立目标彼此相容，不是把四足站立参考锁在整段运动中。

`scripts/s1_go2_support_reference.py` 先求可达足端与支撑，再用 27 变量 native `mj_forward` 平衡求解：18 维零加速度、8 维足端 xy 与 1 维高度条件。它得到一致的 joint pose、contact force、actuator torque 和 PD 输入中心，零速度下满足 `u_ref=q_ref+tau_ref/60`。mg/4 只属于初值思想，不是强制承重分配。参考构造不推进或重置真实 plant。

S3 把 Stand 的输入误差尺度从 0.1 rad 改为 0.02 rad，原 effort 权重仍为 0.08，产生 25 倍二次曲率；StandSupportForce 权重为零，S2 加强承重跟踪的负结果被保留。运动中的 effort 仍是 `0.02 * actuator_force`，不是这个 Stand 输入误差。这个激活边界是恢复时不可省略的内容。

## 3. 确实可以恢复到什么程度

### 3.1 源码和构建证据

| 项目 | 已核查内容 | 证据边界 |
|---|---|---|
| 原项目冻结提交 | `fb7f924f2d519582a127d15ff2e1e2cb2cd5bc9e` | 原 Git 内容可读；发布证据另见 `e23d7e419020e6cf702e2dd2c1109125ffb923a3` |
| 作者上游 | `johnzhang3/mujoco_mpc@e00c47a5adb9856af2e0f24231bb3a60d5be23c4` | 本次通过 GitHub 连接器实际读了该提交的 policy.cc 与 quadruped.cc |
| 依赖 | MuJoCo `088079eff0450e32b98ee743141780ed68307506`，Menagerie `d82891f8b6c576f9f6a9df308f3a1155ebedbc40` | 锁定记录，不是本次重新编译证明 |
| 必要补丁 | `s1-go2-wb-startup-upstream.patch` → `s1-go2-wb-equilibrium-upstream.patch` → `s1-go2-wb-support-upstream.patch` | 这三份补丁从原冻结提交至当前没有 diff |
| 支撑求解 | `scripts/s1_go2_support_reference.py` | 从原冻结提交至当前没有 diff |
| 入口与模型适配 | `scripts/run_s1_go2_wb_baseline.py`、`scripts/s1_go2_wb_model.py` | 分别有 97、29 行的 diff 统计；必须取旧提交版本或明确审查后续改动，不能假定当前同名文件就是旧冻结版本 |
| 配置 | `experiments/terrain_memory/s1-go2-wb-support.json` | 写明 pin、顺序、执行时钟、S3 与任务协议 |
| 原 S3 可执行哈希 | `1a586ca2879b5fed37d58cbadb12e3d2f4afd601dd468fe8f1c8ae0f78a21af3` | 记录来自 resolved 与 index；本次没有取得或启动该二进制 |
| 原构建 | `cmake --build work/terrain-memory/upstreams/mujoco_mpc-go2/build --target agent_server -j 2`，`-fno-strict-aliasing`，日志 `work/r8-build-s3.log` | 历史构建已成功；未验证当前环境直接重建成功或二进制逐字节复现 |

上游 policy 的真实行为是：插值 action、state、feedback gain；计算 MuJoCo state difference；加入完整 `K δx`；再限制到 actuator ctrlrange。不是只跟踪开环轨迹。R8 Python 入口以 `nominal_action=False, averaging_duration=0` 获取每 2 ms 动作并直接写 `data.ctrl`，然后由原 affine PD 和物理限力执行。

精确上游文件：

- [R8 锁定 iLQG policy.cc](https://github.com/johnzhang3/mujoco_mpc/blob/e00c47a5adb9856af2e0f24231bb3a60d5be23c4/mjpc/planners/ilqg/policy.cc)
- [R8 锁定 quadruped.cc](https://github.com/johnzhang3/mujoco_mpc/blob/e00c47a5adb9856af2e0f24231bb3a60d5be23c4/mjpc/tasks/quadruped/quadruped.cc)

以上两文件通过官方 GitHub 工具读取成功；普通网页抓取返回 DisabledError。不能把失败的网页引用当全文读取成功。旧 Release 资产列表未见 agent_server 或完整源码包；因此准确说法是“恢复所需源码、pin、补丁与构建证据齐备”，不是“现成旧二进制已在现场可直接开跑”。

### 3.2 与 W52 r40/r41 的实际差异

本次把 R8 nominal resolved 与 W52 D2 resolved 直接比较，19 个机器人与时钟字段全部相等：质量 15.206408 kg、完整初态、关节名与范围、actuator 类型/gear/60 与 −60/−5 参数/ctrlrange/forcerange、head offset、plant 0.2 ms、反馈 2 ms、规划 20 ms。刚地 native contact 也相同：solref `[.002,1]`、solimp `[.99,.999,.001,.5,2]`、margin 0、friction `[.5,.005,0]`。这支持“旧的正常运动与当前使用同一基础机器人和执行器”，不支持“整个控制问题完全相同”。

| 组成 | R8/S3 | W52 r40/r41 | 恢复含义 |
|---|---|---|---|
| Go2 与执行器 | 同一 15.206408 kg、PD 60/5、hip/thigh 23.7 Nm、calf 45.43 Nm | 相同 | 无需换机器人或放宽物理限力 |
| 运动 gait | Trot，2 Hz，duty .45，amplitude .03 m | 2 Hz、.45 保留，amplitude .005 m | .03 是旧配置之一；不能孤立声称它就是失败根因 |
| 预测积分 | .001 s ×2，节点 .002 s，176 节点/.35 s；原 differentiable predictor | .0002 s ×10；实际 native-contact direction 与 soft GN metric 等后续角色 | 必须按实际 forward/derivative 合同恢复；不能只看继承的 agent_differentiable 元数据 |
| 运动 residual | 作者 swing 足高度、姿态、身体目标、balance、effort 等；stance 不加冻结足点约束 | moving-body-support、moving-stance-height、InputReference、landing/material/impact/trunk 等 | 原权重相同不代表 residual 定义与激活相同 |
| 参考 | 支撑 reference 仅 Stand，运动回作者 home/force effort | 另有 r35 输入中心、运动支撑与后续完整 policy 补丁 | 不要保留当前冻结承重中心却命名为恢复 R8 |
| 策略执行 | 原 full-state iLQG feedback + ctrlrange clamp → 原 PD | H/QP 瞬时 PD 可行域、严格 native 接受、reference/K/H 协同、D2/D3 seeds 等 | 若选择恢复内核，应整组恢复；若保留某层，明确叫适配变体 |
| 初始化 | t=0 的 20 次规划，不推进 plant，初始 2 s Stand | 有后续 coldstart/初始化和完整候选链 | 不能把旧已走轨迹当新 donor 或拼接初态 |
| 环境 | 仅刚地 | TM02 原任务与材料力、observer、记忆链 | R8 不含当前材料 native proxy；这是最主要的资格边界 |
| 高层输入 | 原任务速度 .25 m/s 与转向 .5 rad/s | 冻结 PPO 给 cap，高层命令按当前任务传递 | 保留当前高层输入就已是新适配基线，应同输入公平评估 |

直接比较文件为：

- `results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/project-s3-single-05-resolved.json`
- `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261116T110800Z-w52-d2-eval-PANN_PPO_S5-s1-D8-attempt01-resolved.json`

当前 r40 的额外 numerics 包括 r10/r11 的接触与运动支持、r12 substep/trunk、r26 impact/reference、r35 InputReference；对应新项如 ImpactMargin/MaterialDomainMargin/TrunkClearance 权重 176、MaterialSupportHeight 1、LandingIndentation .1、LandingVelocity .02、InputReference 1。这些共同改变了问题。不能从差异表单独证明哪一项是根因。

## 4. 适合下一轮的恢复执行范围

建议明确区分两种名称和用途：

1. **R8 原冻结刚地内核**：代码与控制语义的恢复参照。历史 9+3 已经提供实证，不必为了证明记得这段历史重新跑完整扰动矩阵。
2. **R8 内核接当前任务的适配基线**：真实执行器、plant、原路线、材料加载、观察长度、违规/失败条件、记录与冻结高层命令保持当前任务；运动内核成组恢复。它是新基线，从新的原 E30 建立资格。

适配的关键是把 plant/任务外壳与优化器控制语义分开。恢复旧 optimizer 时，不需要把材料世界改成刚地，也不需要清空材料状态；但必须承认旧 predictor 不认识 TM02，此时它只构成材料未知的运动基线。若继续把当前材料 proxy、全部新运动成本和当前 QP 执行层揉进去，已经是第三种控制器，失去了恢复旧可行运动的明确比较意义。

新增工作应落在必要的桥接与一次真实任务上：使用实际当前初态与冻结命令、正确 foot/body/joint/clock 映射、同样的物理 actuator/材料、清楚的新版本身份、实际 policy 与原件记录。完成这些后，可以在新范围中直接安排一次原 E30，正常走到终点或原失败条件。没有必要先叠加 D8、若干短 smoke 和 9+3 旧矩阵。E30 成功只使该配置、该 case 进入既定下一阶段；失败也不能借旧刚地成绩获得材料资格。是否采用这个新范围由主审方案明确，而不是把 W52 已冻结的禁止重跑规则暗中改名绕过。

如果保留 PPO cap 与 PANN observer，却不把 PANN 材料量用于这个旧 predictor，就应明说：高层冻结输入/记录链保留，但该运动基线并未检验材料记忆收益。比较应以正常支撑交换、身体推进、原任务/材料约束等物理结果为主；不能比较两个不同 residual 体系的 J 总值。

## 5. 力与运动协调：只保留一个直接相关的替代族

### 5.1 原始文献与实际公开代码

**Donghyun Kim, Jared Di Carlo, Benjamin Katz, Gerardo Bledt, Sangbae Kim. “Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control.” 2019，arXiv:1909.06586。** 本次核查的是公开预印本，未获得足够依据把它标成特定会议或期刊正式论文。不要凭印象加 RA-L/ICRA 发表标签。

论文 §III 用预定接触序列求支撑力，非接触足不占反力决策变量；式 (11) 给摩擦/正法向约束。§IV 先求位置/速度/加速度任务，式 (25) 联合调整 MPC 反力和浮基加速度，式 (26) 从全身动力学得到关节力矩。论文 §V 的 joint gain 是 3/.3（外展关节阻尼 1），且明确命令力矩可能超出电机能力后被截断；它没有证明本项目 60/5、TM02 或当前严格执行链的可行性。[原文](https://arxiv.org/pdf/1909.06586)

[论文全文](https://arxiv.org/pdf/1909.06586)；[MIT 官方 Cheetah-Software](https://github.com/mit-biomimetics/Cheetah-Software)。

本次实际读取的官方实现进一步明确了接口关系：

| 官方文件 | 对本任务有用的实现事实 |
|---|---|
| [ConvexMPCLocomotion.cpp](https://github.com/mit-biomimetics/Cheetah-Software/blob/master/user/MIT_Controller/Controllers/convexMPC/ConvexMPCLocomotion.cpp) | 同一 gait 产生接触表与摆足 p/v/a；MPC 反力送 `Fr_des`。未使用 WBC 时也给 stance force feedforward |
| [LocomotionCtrl.cpp](https://github.com/mit-biomimetics/Cheetah-Software/blob/master/user/MIT_Controller/Controllers/WBC_Ctrl/LocomotionCtrl/LocomotionCtrl.cpp) | 每条腿按 contact_state 进入带 `Fr_des` 的 contact list，或进入摆足 p/v/a task |
| [WBIC.cpp](https://github.com/mit-biomimetics/Cheetah-Software/blob/master/user/MIT_Controller/Controllers/WBC/WBIC/WBIC.cpp) | 用接触 Jacobian、全身 M/C/g、反力约束和 QP 形成 torque，不是几何 IK 后靠 joint PD 猜承重 |
| [WBC_Ctrl.cpp](https://github.com/mit-biomimetics/Cheetah-Software/blob/master/user/MIT_Controller/Controllers/WBC_Ctrl/WBC_Ctrl.cpp) | 同时输出 `tauFeedForward`、`qDes`、`qdDes`；这些构成完整命令，不能只拷贝 qDes |
| [SingleContact.cpp](https://github.com/mit-biomimetics/Cheetah-Software/blob/master/user/MIT_Controller/Controllers/WBC_Ctrl/ContactSet/SingleContact.cpp) | 具体包含摩擦锥近似与法向上下界；默认 μ=.4 和 Fz 上限 1500 是该实现参数，不应移植为本项目参数 |

公开代码已确认可读。它是 Mini-Cheetah 的实现，不是现成 Go2/TM02 插件，模型、关节正负号、腿排序、摩擦、执行器接口与限力均必须适配。

### 5.2 对本项目的推论：完整命令可以映射到原 PD 接口

以下是根据本仓库 actuator 映射的代数推导，**不是论文对本项目的实验结论**。当前未截断的 12 维执行器命令满足

\[
\tau_{raw}=60(u-q)-5\dot q.
\]

如果全身方案构造的完整命令是

\[
\tau_d=\tau_{ff}+60(q_d-q)+5(\dot q_d-\dot q),
\]

则相同的原接口可以用

\[
\boxed{u=q_d+\frac{\tau_{ff}+5\dot q_d}{60}}
\]

表达它。这与 R8 静态平衡的 `q_ref+tau_ref/60` 相衔接，但多了实际动态力矩和目标速度；**不能把旧承重输入中心继续留下，再添加同一支撑力的前馈**。ctrlrange、力矩截断、当前 QP 投影会破坏未截断恒等式，因此实际 executed control、raw PD 和限力仍须按原物理路径记录。若采用不同 joint gains，需重新推导状态相关补偿，不能假装上述式子仍然成立。

在 MuJoCo 上构建这个动力学命令时，可从同一模型的 M、bias、passive、足端 Jacobian 建立

\[
S_f(M\ddot q+h-J_c^T f)=0,\qquad
\tau_j=S_a(M\ddot q+h-J_c^T f),
\]

其中 h 应一致包含本模型的 bias/passive 约定。支撑足的力需满足原摩擦和法向限制；摆动足的规划反力为零。接触转换、摆腿惯性和身体加速度同时参与求解，不能用“抬起的一足删掉，再把 mg 平均给另外三足”代替整个运动控制器。

### 5.3 建议的具体边界

首先构造完整的模型一致候选，明确产生 q/v/τ、同一接触集合和对应反馈；不要沿用与新参考不相容的旧 K/H，然后把新结果只称为运动 seed。可以选择在现 MPC 内将这些量作为新一致名义轨迹，或建立一个明确独立的 SRBD MPC + WBC 运动内核。两条实现中一次只选一条，不能无限叠加。

最小的“支撑一致前馈”适合检查旧 PD 中心与摆动卸载是否冲突，但它仍需动态接触和浮基平衡。它不能因满足某一瞬间静态平衡就被宣称为完整稳定步态。完整 SRBD MPC + WBC 的优点是分工和公开实现清楚；成本是新的机器人模型/求解器/执行器接口工程，不能描述成几个参数即可接通。

TM02 还有额外边界：刚地 WBC 中 stance 零速度/零加速度假设不直接等价于可压缩材料中的真实支撑。材料可以压入、回弹，实际接触不能由 gait 开关强行置零。控制器的 stance 约束或目标应与材料预测、因果接触估计一致，并允许必要的法向运动；真实 plant 接触力与原 strain/非法接触条件必须继续决定结果。论文与官方代码均未提供这部分现成证明。

因此下一轮的有限选择是：**一个完整恢复的 R8 运动基线，加一个明确的接触一致动力学控制方案**。二者并行实施/审查可以减少继续在失效内核上逐个几何修补的依赖；实际使用资源、实验次数和阶段推进须由同一新计划明确。论文不是“换了就成功”的证据，旧成功也不是新材料任务的资格。

## 6. 本次覆盖与限制

已实读 R8 配置、三补丁关键控制段、入口执行段、模型适配与原支撑求解；检查冻结提交到当前的上述文件 diff；核对 formal 逐条记录与已存在的 W45 比较；另直接比较 R8 nominal/W52 D2 的 19 个机器人/时钟字段与 native contact。实际读取了锁定作者 fork 的 iLQG policy 与 quadruped 源码，及 MIT 官方 WBIC/WBC/Contact 实现。没有复算整条旧轨迹、下载大 Release、编译或运行新任务。论文方法看了原文 §III/IV/V，未扩展成文献清单。

供主审重新打开后引用的网页：`turn29view0` / `turn30view0` 均为 arXiv1909.06586 全文。官方代码内容由 GitHub fetch_file 获取，网页检索对若干路径 DisabledError；主审可用同一连接器核取，而不应引用错误页。上游源码内容只用于确认控制语义，不声称本次完成构建兼容性验收。


当前实际实施范围以仓库 `docs/S1_W53_STABILITY_PLAN.md` 为准；本文保留独立审查的证据与建议形成过程。
