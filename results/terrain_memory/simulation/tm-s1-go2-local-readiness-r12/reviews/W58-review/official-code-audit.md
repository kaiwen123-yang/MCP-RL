# 官方 SuperMimic、物理世界模型与 MCP-RL 融合审阅

日期：2026-11-26（上海）。本笔记仅依据官方论文、官方源码及所列论文作者来源。没有使用用户个人 SuperMimic 复现仓库、权重、训练结果或归档记录。

## 1. 核验版本与阅读范围

- 官方代码：<https://github.com/HybridRobotics/supermimic>，本次只读克隆 commit `5fb92297ab4fe3d86629d9f0c635334791bb0d5f`（2026-11-21，最新提交说明为 body velocity 使用 anchor heading）。代码许可 Apache-2.0；不是任意第三方复现。
- 最新论文：<https://arxiv.org/abs/2610.28480>；实际下载并阅读全文 v3 PDF：<https://arxiv.org/pdf/2610.28480v3>，2026-11-19 修订；CoRL 2026 Oral。HTML 访问失败，PDF 可读。
- 官方项目页：<https://supermimic.github.io/>。
- 源码实读：`models.py`、`representations.py` 的核心表示/积分、`learners/wm_learner.py`、`learners/policy_learner.py`、`learners/learner_utils.py`、`envs/world_model_env.py`、`runners/supermimic_runner.py` 主要训练/收集路径、`runners/run_config.py`、`tasks/task_spec.py`、`tasks/env_cfgs.py`、`sim_context.py`、`scripts/train.py`、`pyproject.toml`、README。没有运行训练、没有安装其环境、没有验证其论文效率。
- 近邻全文重点读：DynaMimic v1 PDF；PWM v3 方法章节及官方摘要；DMO 官方论文摘要/项目页；Coupled Local and Global World Models 官方方法页；RWM-U 最新官方摘要。Trust What Repeats 由 root 独立全文审阅，本笔记只标明必须纳入创新边界，详见 root 的研究记录。

## 2. 官方 SuperMimic 的可迁移核心

核心是三段交替循环：当前策略在 MuJoCo Warp 收集真实模拟器转移；世界模型从 replay 多步拟合；固定世界模型参数，以短 imagined rollout 的任务回报及末端 critic 训练策略。反向传播经过 learned world model，**不经过 MuJoCo**。世界模型不是永久冻结，只有 actor 更新期间冻结。

官方针对 Unitree G1 全身参考动作跟踪；actor 输出 29 个关节目标的归一化参数，低层 PD 执行。我们是 Go2 上层时间/落脚请求、MPC、WBC 和硬接触门共同构成的闭环，二者 action 的含义、执行状态和数据生成机制不同。可迁移的是学习框架、结构设计及实现经验，不能迁移 G1 actor/WM 权重或直接替换当前控制栈。

### 2.1 状态与结构

源码 GenCoord 保留根姿态辅助状态，将网络状态组织为高度、机体坐标下重力/线速度/角速度、关节位置和速度。网络输出根线速度、角速度及关节速度的增量；随后积分根姿态和关节角，保证预测的姿态与速度一致。当前源码根据 simulator decimation 使用 `(d+1)/(2d)` 修正姿态积分中的速度增量权重，体现了学习模型必须匹配前向离散化。

模型输入还包括执行后的 action 和 episode context。`sim_context.py` 从模拟器读摩擦、COM 偏移、encoder bias；这些是训练时已知的随机化参数，不是由机器人感知推断的未知材料状态。将我们未观测的真实材料隐变量直接放入在线输入，会引入额外信息，不能称公平迁移。

### 2.2 世界模型学习

当前 `WorldModelLearnerCfg`：10 个 batch，每批 2048 个长度 8 的序列；**代码实际是 8 个状态帧、7 次自回归预测并监督**。论文概述写 `k=8`，移植配置必须记录“多少状态帧、多少转移、多少实际秒”，不能含混地把 horizon=8 当成八步转移。

损失对 integrated root pose、joint position/velocity 按各组 nominal error 加权；不是单一步预测后每一步都用真值重新校正。多步训练使模型面对自身误差累积。已有分析表达式的 reward、observation、termination 无需另学一个黑盒头。

### 2.3 策略与 critic

默认 actor-critic learner imagined horizon=32、gamma=.99、lambda=.95。固定 WM 参数但保留对输入和 action 的导数，逐步累积 discounted survival-weighted reward，加末端 critic。critic 在这批 imagined transitions 上拟合 GAE/λ-return，target detach，并限制最大可达回报；不是通过 actor loss 反过来修改环境模型以夸大奖励。

论文的控制周期是 20 ms，故 32 步是约 .64 s；不能把这个数原样搬到一次跨脚事件约 .6 s 的宏动作层，后者将变成约 20 s 的开环想象。我们应该先明确模型时间尺度，并通过短连续时间 rollout + 事件/回访 terminal value 覆盖较慢的材料恢复。

官方软 termination 是为可微训练构造的预测概率。当前代码实际按 reference height/tilt error 的硬规则作 sigmoid 近似，softness=.05。**这不是允许真实执行器违反 hard gate 或把失败后的虚构轨迹当真**。

### 2.4 实现与论文有差异，必须 pin recipe

| 项目 | 论文 v3 | 本次官方 main 源码 |
|---|---|---|
| actor 输入 | 10 帧带噪 proprio | 默认 `actor_obs=rep_state, history_length=1`；支持显式 proprio/history |
| 优化器 | 表 A.5 写 Adam | hidden Linear 权重 Muon，其余 fused capturable Adam |
| 数值精度 | 项目页吞吐测量注明 TF32 | Dynamics/policy forward 使用 CUDA bfloat16 autocast |
| update/data | .4 | 默认 replay_ratio=16 与 4096/(10×2048×8) 相乘，等价 .4 次 joint update/collection step |
| 收集动作噪声 | N(0,.1²) | 已读 runner 直接将 actor action 传 env，未在此路径发现该显式噪声 |
| 训练调度 | 250k cosine 到 .03×，320k 总迭代 | 当前默认 max_iterations=1,000,000；已读 runner 未发现同一 cosine recipe |
| 多接口学习 | velocity RL encoder 独立任务回报，decoder 参数 stop-grad | 当前 task/runner 注释及 world-model env 路径为各接口共用 motion-tracking reward |

因此计划应写“采用官方的某个训练机制并按我们的闭环重建”，不能写“clone 官方仓库即得到论文实验配方”。源码没有为我们提供已校准的 ensemble uncertainty 或梯度正确性证书。

官方论文速度来自同一 H100 上的大批量 G1 学习及短模型 rollout。我们的 MuJoCo .2 ms、WBC 2 ms、MPC .35 s、材料服务及高层动作执行开销完全不同；不能把 1000× GPU-hour 或 10–62× rollout 速度写成项目收益预期。

## 3. 最近工作已覆盖什么，项目还可做什么

### DynaMimic（2026-11-24，预印本）

官方：<https://arxiv.org/abs/2611.32944>，<https://arxiv.org/pdf/2611.32944v1>。这是最直接的近邻，不能遗漏。

每个 20 ms 控制步网络预测 15 个候选支撑 link 在四个 5 ms 小段的 wrench 与一个 35D residual generalized force；normal force 非负、切向力在摩擦锥内。解析 PD、被动力、Jacobian 与 Lagrangian dynamics 用两个 10 ms 半隐式积分步生成状态。训练使用 8 状态/7 预测的 state+wrench+residual-regularization；32 步 imagined BPTT。地形图在预测 pose 改变后更新，训练时有已知 episode 参数。

启示：**“解析刚体动力学 + 学接触残差 + SuperMimic”已是直接已有方法**。其真实部署只运行 policy，没有在线 MPC；其 terrain 是几何/接触，不含我们持续的材料恢复世界状态、跨脚回访的历史闭环。因此可借鉴 force-level residual 及物理输出约束，但核心研究应定位到材料记忆如何通过可执行动作产生长期收益。

论文有受控 ablation（link input、forward dynamics、contact supervision），并非仅模型 MSE。其 world-model MSE 用各方法自身策略访问的轨迹，不能当共同同分布模型优劣的完全证明。训练集 motion coverage 也不能替代新材料、速度与历史上的泛化验证。

### Trust What Repeats（2026-11-20）

<https://arxiv.org/abs/2611.27648>。由 root 全文审阅：直接围绕 SuperMimic 梯度重复性、真实模拟器方向检验、逐 rollout cap。因此“可信梯度”“clip 梯度”“用真实 rollout 验证”单独也不再足以构成原创主张。我们的候选差异应落到**执行门导致的动作失效、世界格点历史跨脚传递与事件时间尺度下的可辨识决策梯度**，并需要实际实验建立。

### DMO（CoRL 2025）

<https://proceedings.mlr.press/v305/amigo25a.html>，<https://machines-in-motion.github.io/DMO/>。

真实 simulator 前向生成轨迹，learned dynamics 提供反向，critic 也能用真实轨迹学习，已有 Go2 硬件任务。适合我们模型开环长预测尚不可信时作为过渡参照，但每次更新仍依赖新 simulator rollout；它不会自动解决我们的采集吞吐瓶颈。

### PWM（ICLR 2025）

<https://arxiv.org/html/2407.02466v3>，<https://proceedings.iclr.cc/paper_files/paper/2025/hash/240ea1741b205ea295721d55184ac43b-Abstract-Conference.html>。

方法章节通过接触与混沌例子说明，预测 MSE、梯度方差、最终优化收益是不同标准；较平滑模型可能更易优化。这里能借鉴“模型用于决策，应检验优化后真实收益”。不能推导出可以随意把硬门平滑掉或用更低 MSE 掩盖实际动作未兑现。

### RWM-U（2026 修订版本）

<https://arxiv.org/abs/2504.16680>，v3 为 2026-01-08；作者项目页 <https://sites.google.com/view/uncertainty-aware-rwm/home>。

自回归世界模型的不确定性和多步传播配合 uncertainty-penalized PPO，说明保留 PPO 也能用 WM 扩大数据复用。故“WM+PPO”应列作基线，而不直接宣称新联合算法。

### Coupled Local and Global World Models（CoRL 2026）

<https://coupled-global-local-wm-rl.pages.dev/>，arXiv:2602.06219。作者使用高保真 global 模型前向、轻量 local RSSM 反向；因此“两个世界模型融合/局部模型供梯度”也有直接近邻。官网展示 policy exploiting model hallucination 及真机数据修补。对本项目的启示是：模型奖励上升后必须回到真实执行器验证，并收集它实际犯错的区域；不需要为追逐该论文而增加视觉生成模型。

## 4. 对 W59 主线的技术建议

建议名称只作内部工作描述：**具有材料记忆、预测实际控制执行结果的结构化世界模型**。其组成应为同一个受数据约束的 transition operator，不是三套互不相容的预测器。

状态至少分成机器人状态 x、控制器内部状态 c、材料格点 belief b，以及任务状态 g。c 包括接触 phase、pending timing request、已确认 bearing/release、current reference 和可能影响下一步的 MPC/WBC 缓存。若忽略 c，同一 x、b、a 可能产生不同实际动作，模型难以成为 Markov 预测器。controller version/参数必须作为数据上下文记录；同一数据池内改了控制器却不标版本，会把控制器变化错误学成材料响应。

### M3：材料物理记忆 + 可观测 belief

保持材料本构/恢复方程为骨架，用完成接触后的测量更新低维、可辨识 belief 及不确定度。网络补预测残差和未知参数，不宣称从单 foot force 反演所有隐变量。按真实 world-cell ID 存储、回访和释放时间演化；不得用未来真值“恢复”输入。

损失重点不只是当前 force：加入 release→reload 时点的承载、接触高度/压陷、预测区间校准与整 episode holdout。材料物理贡献与执行误差应分开有头可检验，否则网络可把 WBC touchdown 偏差假装成材料记忆。

### P3：action-conditioned execution WM + 同一候选算子

输入是提议给原控制器的高层 action，输出必须包含实际发生的 touchdown/support/release 时刻、已执行的接触负载、滑移/工作/姿态极值及下一状态。不能只对给定候选力作材料 rollout，再默认那个力一定被 MPC/WBC 实现。

可以由解析 centroidal/刚体更新承担已知部分，由学习模块预测实际接触力偏差和事件时间。约束应施加在总预测 force/torque，而不是仅对 residual 正则化；比较 DynaMimic 的 unilateral/friction 约束与我们现有执行限幅。先用 frozen predictor 排序候选或 CEM，真实执行选择后的动作，记录 model rank 与真实 rank；这不需要对整个 MPC QP 和其离散选择求导。

保留 full/private/fixed-wait 的等候选集合、等动作权限和同一 objective。若历史并不能改变实际动作/回报，学习模型不能创造这个物理效应。此时可推进预测/步态效率方向，但不要以大网络掩盖材料记忆假设失败。

### R3：先真实任务 critic，再条件式短 imagination

先以真实 executed transitions 拟合任务 critic，并明确 reward、failure 与 truncate 的区别。原任务到家后的 hold、时间预算与已完成任务状态都应进入值函数。把“episode 数据截断”学成失败概率、把回家后等待当无穷正回报、或用软存活权重赦免已经触发原失败门，都会产生错误 credit。

初期先做 WM-enhanced critic / candidate policy；模型策略梯度通过验证后再开短 BPTT。与 PPO 比较时固定相同实际采集预算、相同高层 action 权限、同一初始策略与奖励。不要把 PPO 的 score gradient 与不同目标的模型 gradient 任意加权后称理论成立的联合算法；先明确并分别报告每项训练信号。

### 联合算法的关键接口

同一个 predictor 服务 P3 候选评估、R3 imagined return 和 M3 的 material loss；每次 actor/planner 更新时冻结 world-model 参数。真实新数据回来后再拟合 WM，更新标版本的 checkpoint。这样“共享一个预测世界”是可操作接口，但是否具有原创性仍要靠材料跨脚记忆与执行受限决策实验。

真正希望验证的是：相同几何和观测、不同材料历史导致不同可执行接触负载与时机；共享模型让 planner 和 policy 利用这个差异；回访后的真实收益显著超出模型误差、离散化和随机波动。当前 W58 给出的 readiness 误差大于 full/private 差异几个数量级，因此 W59 必须正面处理信号尺度，而非直接大规模 R3。

## 5. 不可微闭环、梯度与必要准入

原 MPC active set、候选 argmin、脚接触切换、support/release hard gate、action clipping、termination 都可能不可微。一个 torch MLP 可以有梯度，并不意味着它对真实闭环的 action sensitivity 正确。

建议首先做**已保存完整状态的配对真实执行扰动**。对相同 x/c/b 和同一随机状态，执行 a±δ；同时比较 predicted action→actual action、event time、物理指标及任务回报。应报告请求有效率、饱和/被门挡住比例和跨模式比例。被门完全挡住且结果一致的动作区间应预测接近零影响，而不是产生虚构的连续梯度。

连续模式内部可检验方向导数、不同 δ 下的稳定性；跨接触模式的离散选择先用候选/分支评估。若将门平滑成训练 surrogate，必须标为偏置近似，并用真实结果排序/更新方向检验，不能说梯度已经穿过原硬门。

模型训练不应只有归一化 state MSE：

1. 开环多时距（以秒计）state/force/load 误差与事件时间误差，尤其 touchdown→rebound→bearing 和 release→reload。
2. 真实 hard gate 的命中/未命中、执行延迟及误放行概率；稀有事件不要被平稳帧数量淹没。
3. held-out episode 的 uncertainty coverage，并对未见 history/速度条件单列。
4. 同输入 full/private 或相同状态不同 action 的真实决策排序；误差应与要利用的最小效应量比较。
5. 最终仍由原 E30/E60/E600 稳定性与任务效率验收；WM reward 和预测 MSE 不能替代真实执行。

梯度校验本身已有 Trust What Repeats 等近邻；项目潜在贡献应是**因执行门、材料记忆和异步恢复而产生的决策可用性标准与相应算法**，不能只改名包装。

## 6. 官方代码可复用清单

直接借鉴设计：autoregressive WM learner、actor update 内冻结 WM、短 H+critic、有 episode 边界的 replay、状态/动作归一化、噪声分离、动作保存为实际执行分布、GPU batch 编译。

需要重写：机器人表示与维度、世界格点材料状态、MPC/WBC/controller state、动作接口、现实执行时长、hard gate 对应的事件预测、任务 reward/termination、真实数据导出与 replay schema、异步 actor/model version 管理。

暂不应直接搬入：G1 权重、BONES 数据、mocap 目标 encoder、多接口 velocity reward 配方、50 Hz/32 step 未换算的 horizon、已知 simulator μ 作为在线材料估计、官方吞吐倍数。Python3.13 + sibling mjlab_research 的完整环境也不应直接覆盖稳定 MCP-RL 环境。

并行最有价值的分工：既有数据的 M3/WM 对照拟合（GPU）、完整状态 action 干预及 G0 步态证据（CPU）、P3 候选执行闭环（CPU）；R3 原型/梯度计算可与数据收集并行，但大规模训练必须使用当轮已校验模型与动作权限。六任务可以是六个互补工作流，不等于六个重型仿真同时争 CPU，也不等于96GB显存必须占满。

## 7. 最重要的判断

可以加入官方 SuperMimic 思想，而且它与当前“昂贵真实闭环采集、GPU 很空闲”的结构相匹配：用少量可信真实转移训练能批量运行的模型，再复用这些转移改善决策。但是，SuperMimic 解决的是学习效率；它不会让原本无效的 timing request 获得权限，也不会凭空放大微弱材料历史效应。

W59 应并行建立三条实质证据：材料记忆对可执行动作有意义、world model 的短期预测和局部行动效果可相信、利用它的控制/学习比相同预算的原算法更好。只有这三条连接起来，才构成项目自己的联合算法结果。
