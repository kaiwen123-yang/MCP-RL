# 学习模块：可信闭环梯度、PPO 更新与独立研究假设

研究日期：2026-11-26。本文是设计与文献核验，不代表实现或实验已经完成。未运行训练、未修改 active W59。本文只承接仓库已有的官方 SuperMimic 等阅读；不使用用户 SuperMimic 复现记录作证据。新增论文记录见 `learning_sources.json`。

## 1. 结论与当前工程边界

**可以把学习模块做成独立研究方向，但“PPO + 世界模型梯度 + 可信度权重”本身已不足以成为核心贡献。** AGPO 已混合模拟器梯度与 Q 梯度；AHAC 已根据接触刚性调整解析时域；SAPO 已实现最大熵解析 actor-critic；MB-MIX 已结合动力学 Sobolev 训练和多时域梯度；MFPG-PPO 已直接把高、低保真控制变量并入 PPO；Open-DiffLoco/JAVE 已监督 critic Jacobian；AC-MPC 与 SmoothedTurboMPC 已分别展示 PPO 和 SHAC 端到端训练可微 MPC actor。[L02–L09]

我们有价值的问题更加具体：**当机器人实际执行的是有约束、有内部状态、有延迟、会切换接触模式的完整控制栈时，什么导数能改善真实闭环目标，怎样同时降低梯度偏差和方差，而不会污染 PPO 的概率模型？**

仓库已有 true-time discount/GAE、raw action/logp、Gaussian KL 检查与 rollback，属于必须保留的正确性基础，不应包装为新贡献。当前 Diff actor imagination 更新不等于“PPO + 模型校正”；真实与想象目标共训同一 critic 也不自动保证目标一致。已有方向审计没有支持当前模型梯度大范围放开，不能把低预测误差或少数稳定行走视频解释为梯度可信。

两条独立候选路线：

1. **研究假设 A：执行与接触事件条件下的多保真梯度校正。** 要解决的是相关但有偏的低保真梯度如何帮助高保真 PPO；核心衡量是同样高保真预算和总算力下的真实目标改善，尤其关注分布错配、事件错位和零均值校正。
2. **研究假设 B：带执行器状态的完整闭环伴随校准。** 要解决的是动力学值预测、控制器求解和 critic 标量都准确时，为什么 actor 导数仍错；核心在完整反馈链与事件导数的一致性，而不是再加一项普通 Jacobian loss。

两者是可检验研究假设，并非“已证实首创”或“必然够 T-RO”。A 必须与 MFPG-PPO 正面对比；B 必须与 JAVE、MB-MIX、AGPO/AHAC 正面对比。

## 2. 新增 12 篇文献：已读内容与实机边界

阅读顺序考虑用户的期刊优先级，技术取舍按直接相关性。本文没有核验 2025 中科院分区，因此不标注任何未核验的“一区 TOP”。Nature Conference 上展示不等于 Nature 期刊论文。

| 编号 | 论文与核验状态 | 本次实际阅读与应借鉴之处 | 实验与边界 |
|---|---|---|---|
| L01 | **DreamerV3**, *Mastering diverse control tasks through world models*, Nature，2025-04-02 | 正式版 learning algorithm、critic/actor、实验及方法；RSSM、分布式价值表示、稳健量纲处理。正式版对连续与离散动作都用 **REINFORCE**，不能泛称“穿过世界模型的解析 actor 梯度”。 | 多领域基准，不构成四足硬接触、实时 WBC 或机器人安全证明。 |
| L02 | **AGPO**, ICML 2024 | §3–5、算法与理论假设；以批次 action-node 梯度方差混合 simulation gradient 和 Q-gradient，逐步累计可信度，低值时截断；Q 仍可能有系统偏差。 | Brax 等控制基准；不是硬件部署结果。它不是简单的 PPO score-gradient 与解析梯度混合。 |
| L03 | **AHAC**, ICML 2024 | §3–5、接触误差分析、并行版本与消融；单环境接触截断与最终同步批处理版本有区别；最终算法以接触 Jacobian 约束调整时域。 | 5 类 DFlex locomotion、10 seeds；同一时域可能偏好特定步态周期。未验证真实机器人。 |
| L04 | **SAPO**, ICLR 2025 Spotlight；arXiv v2 2025-02-27 | §3–6、设计选择与消融；短时域回报含熵，soft critic、状态相关方差、双 critic。训练需可微奖励与模拟器，温度调节不能修正错误动力学导数。 | 刚体与软材料 6 类仿真、10 seeds；HandReorient 高回报尚不等于连续完成目标重定向。无实机。 |
| L05 | **MB-MIX**, AAAI 2025；arXiv v1 2025-03-03 | §3–6；混合多个截断时域，监督状态/动作动力学 Jacobian。论文明确理论尚未覆盖 learned-model 情况。 | DiffRL/Brax/DaXBench；Bruce 是导入 DiffRL 的机器人模型，所读结果未提供真实 Bruce 上部署证据。 |
| L06 | **MFPG**, TMLR 02/2026；arXiv v4 2026-02-12 | §4–5、控制变量 Lemma 1、耦合采样和假设；同均值 LF paired/independent 统计量可校正 HF 方差。有限样本拟合系数不自动继承理想无偏结论。 | 仿真 robotics benchmarks；与下一行新 PPO 扩展不同。 |
| L07 | **MFPG-PPO**, arXiv v1 2026-10-01，预印本 | §3–6、App A.2；HF clipped surrogate + coupled LF − independent LF mean；三 critic、匹配 GAE 时域、相关性与方差门控。LF 两组重置分布不同会引入偏差，作者明确承认。 | H1/ANYmal-D 仿真；Franka 抓取 3 seeds，每 seed 80 更新×4 回合；已有 LF warm start。四足结果不是实机。 |
| L08 | **AC-MPC**, IEEE T-RO，在线 2025-12，卷 42 | §III.E–H、实验/探索消融；高斯均值就是 diffMPC 输出，PPO 可合法通过 KKT 更新 cost map；并用 MPC 预测扩展价值训练。 | 真实无人机 racing。论文所用 solver 支持输入约束而非一般状态约束；Gaussian 探索样本也不能由均值可行性推出可行。 |
| L09 | **SmoothedTurboMPC**, arXiv:2611.09937v1，预印本 | §II–V、KKT/IFT、SQP-ADMM、SHAC pipeline；平滑互补条件，保留 rollout 求解结果后单次反传，避免 PPO 多 epoch 反复解 MPC。 | 真实 Lexus 赛车、4 次所列 trials；无足式接触评估。PDF页眉日期 2026-10-13 与编号月份不一致，保留版本并标记元数据冲突。 |
| L10 | **Open-DiffLoco / JAVE**, arXiv v1 2026-08-03，预印本 | §II.D–V、App I–II；监督 value Jacobian；明确当前冻结动作省略闭环 actor-feedback 项，L2 one-step model 也不能保证 Jacobian。 | Go2 部署；训练对比与部署配置不同，PPO reward/reference 不完全相同。改善主要在早期稳定性，未证明明确最终性能优势。 |
| L11 | **FastTD3**, arXiv v3 2025-06-01，技术报告 | §2–3、设计消融与部署；并行 rollout、大 batch、distributional critic 和低有效 UTD。是必须认真调优的效率 baseline。 | Booster T1 sim-to-real；自然步态依赖 reward。其大 batch/高吞吐不能直接照搬到当前 CPU 仿真瓶颈。 |
| L12 | **TD-MPC2**, ICLR 2024；arXiv v2 2024-03-21 | §3–5、规划与消融；控制目标驱动 latent model、SimNorm、Q ensemble、最大熵 policy prior，实际规划为 **MPPI 零阶采样**。 | 104 continuous-control tasks；该文不提供四足材料适应与毫秒级 WBC 实机保证。 |

### 2.1 对创新范围的直接约束

- “监督模型导数”必须超越 MB-MIX；“监督 value 导数”必须超越 JAVE。
- “短时域/自适应时域”必须区分 SHAC、AHAC 和 MB-MIX 已解决的部分。
- “可信度混合”必须超越 AGPO 的方差权重、MFPG-PPO 的相关性门控，以及之前已读的 Trust What Repeats。
- “可微 MPC + PPO”已由 AC-MPC 直接实现；“GPU 平滑 MPC + SHAC”已有 L09。
- 自然步态、恢复能力、OOD 稳定性、总训练时间和板载尾延迟需要分别测量，不能用单个 reward 代替。

## 3. 三种梯度究竟是什么：不能简单相加

令扩展状态为

\[
z_t=(x_t,b_t,h_t^{act},h_t^{obs},c_t),
\]

包含机器人状态、材料信念、执行器与观测历史、MPC warm start/WBC 积分器等执行状态。策略产生离散步态/接触选项 \(m_t\) 和连续请求 \(a_t\)。实际过程为

\[
(m_t,a_t)\sim\pi_\theta(\cdot\mid o_t,b_t,c_t),\quad
v_t=E_\psi(z_t,m_t,a_t),\quad
(\tau_t,h_{t+1}^{act})=A_\xi(z_t,v_t),\quad
x_{t+1}=F_{phys}(x_t,\tau_t,b_t,m_t)+R_\phi(z_t,\tau_t,m_t).
\]

这里 \(E\) 是 MPC/WBC 与投影/限幅的执行映射，\(A\) 是因果执行器模型，\(F_{phys}+R\) 是明确约定的物理预测与剩余误差。

**PPO 的真实数据 score gradient** 使用 \(\nabla_\theta\log\pi_\theta(a,m\mid o)\hat A^{HF}\)。它不要求环境可微，但 critic/GAE、裁剪、多 epoch 等已经使实际 PPO 不等于严格无偏全回报梯度；应把“对高保真 PPO surrogate 不额外引入模型偏差”与“对真实无限时域目标无偏”分开表述。

**控制器梯度** 是优化解的灵敏度，例如 \(\partial E/\partial a\)、\(\partial E/\partial\psi\)，可由 KKT 隐式微分或求解迭代展开得到。

**动力学/世界模型梯度** 是 \(\partial F/\partial x\)、\(\partial F/\partial\tau\) 等局部 Jacobian。用它与控制器梯度、执行器梯度、policy Jacobian 相乘，才能形成同一条路径的 actor 梯度。

例如一段链式贡献为

\[
\frac{dJ}{d\theta}\supset
\frac{dJ}{dx_{t+1}}
\frac{\partial F}{\partial\tau_t}
\frac{\partial A}{\partial v_t}
\frac{\partial E}{\partial a_t}
\frac{\partial a_t}{\partial\theta}.
\]

因此不能把同一链上的“控制器梯度”和“动力学梯度”分别当两个完整 policy gradient 再相加。若世界模型已预测 requested-action 到 next-state 的整个闭环，再额外乘一次 MPC/WBC/actuator Jacobian，也会重复计数。物理与残差的局部导数只有在 \(F=F_{phys}+R\) 的清晰分解下才按加法合成。

“可信模型梯度”不是第四种独立数学梯度，而是对某一估计器的可靠范围、偏差和方差进行证据约束。

### 3.1 NeuralActuator 接口：posterior 与 counterfactual 不同

主代理已核验新加入 NeuralActuator。对学习设计最关键的是：如果一个模型输入未来实测 current/telemetry，它可以帮助解释已执行轨迹，但不是候选 command 的因果生成器。不能在未来电流保持原记录的情况下改变候选动作，再把得到的导数叫真实执行器动作 Jacobian。

需要两条接口：

- **观察/辨识分支**：只用截至当前的测量更新 \(b_t,h_t^{act}\)，为训练提供 posterior、置信区间和监督。
- **因果预测分支**：未来电流/有效 effort 必须由候选 command、已知历史与 latent actuator state 预测，再输出力矩和 next actuator state。

若主力矩头已经吸收接触效应，不能直接与 PANN 的接触力叠加；要重新定义残差对象或只取可辨识的执行器子模型。缺乏这种分解时，模型可作为辅助 critic/状态估计器，但暂不授予 actor 的动作导数权限。

## 4. 第一阶段可落地更新规则

### 4.1 先保留完整 HF PPO，再添加可审计的辅助量

对每次 rollout 冻结 actor-old、模型、critic-target、执行器模型、normalizer、trust calibrator 的版本。episode 分割必须保存完整扩展状态与物理 dt。路径分支冻结参数时保留对输入的导数。

最先实现的对照应是：HF-only PPO、HF PPO+固定权重 pathwise、AGPO-style variance gating、MFPG-PPO-style score control variate、下述 conditional pathwise control variate。每个对照必须使用相同 action authority/reward/终止条件。

对高保真起始条件 \(C\) 和共享噪声 \(\epsilon\)，定义：

\[
g_H(C,\epsilon)=\nabla_\theta\ell^{HF}_{PPO},\qquad
Y(C,\epsilon)=g^{model}_{path}(C,\epsilon).
\]

使用同一个 \(C\) 的独立噪声 \(\epsilon'_1,\ldots,\epsilon'_K\) 估计 \(\mu_Y(C)\)。构造

\[
\widehat g_{CV}=g_H+B(C)\left(Y(C,\epsilon)-\widehat\mu_Y(C)\right).
\]

这里的零均值结论首先只讨论**固定当前参数、冻结模型/critic/normalizer、一次fresh batch的梯度估计**；不讨论整个PPO训练算法的无偏性。\(B\)及启用门控必须由此前独立校准数据确定，不能由本批未来事件或校正效果选择。

若 \(\widehat\mu_Y\) 是同一起点下 \(K\) 个相互独立且与paired样本独立的LF样本均值，条件协方差满足

\[
\operatorname{Cov}(Y-\widehat\mu_Y\mid C)=(1+1/K)\Sigma_{Y,Y}(C).
\]

相应的最优线性系数（加ridge时是正则化选择）为：

\[
B^*_K(C)=-\Sigma_{H,Y}(C)\left((1+1/K)\Sigma_{Y,Y}(C)+\lambda I\right)^{-1}.
\]

先前不含 \(1+1/K\) 的公式仅对应 \(K\to\infty\) 或均值已知的理想情况。若独立均值样本共享噪声、跨起点重用或不是同分布，上式也不能直接使用，需计算实际校正项协方差。

维度太高时在 action/actor-last-layer 或固定低维参数方向计算，而非存储完整参数协方差。

**这个式子的价值与限制必须一起交代：**在这些条件下，校正项的条件均值为零，即使 \(Y\) 是有偏模型导数，也只保留当前 \(g_H\) 估计器已有的期望，而不会因该校正额外改变它。**这不使PPO/GAE/critic无偏，也不保证有限样本拟合系数、最终Adam步、全局norm clipping或KL回滚后更新无偏。** 它只能降低 within-start-state 方差，不能凭空消掉高保真起点的 between-state 方差；有限 K 还会带来均值估计噪声。控制变量的一般思想已有 MFPG，不能把这个恒等式单独宣称新方法。[L06–L07]

**需要研究的部分**是：如何按接触/控制器事件与真实梯度分量建立有效相关结构，如何估计块系数与额外仿真预算，并在独立校准表明事件错位或低相关时退回 HF PPO；同时守住条件同分布，避免复刻 MFPG-PPO 自由 LF 起点带来的偏差。起点已知mode可进入 \(C\)；本批未来触地、终止匹配、校正后方差等不能直接用于选择门控或系数。若要利用LF未来事件，应把事件函数纳入 \(Y\)，并对paired与independent两组执行完全相同的规则；不能只筛paired样本。

**首版只允许fresh HF batch的一次CV梯度更新。** 多epoch复用旧batch后，当前 \(\theta\) 已依赖本批paired噪声；即便两组LF重算同一 \(\theta\)，旧paired噪声的条件分布也不必等于fresh independent噪声。因此不能继承上述精确零均值论证。旧batch多epoch版本只能作为另有偏差评估的实验近似，不能声称无偏。cross-fitting必须按独立episode/采样单位划分；随机打散相关时间步不足以建立独立性。

在 \(\theta=\theta_{old}\) 的首次更新，PPO ratio为1、clipping尚不起限制作用。因此首个严格CV原型应称“单次score actor-critic更新，保留PPO概率/KL基础设施”，不是已经完成的多epochPPO创新。标准HF-PPO保留为强基线；若在其完整更新后追加CV辅助更新，需另采fresh HF batch并把成本计入比较。

### 4.2 真正想直接利用模型上升方向时：明确承认有偏，并受 HF 约束

零均值控制变量只减少噪声，不提供任意额外的模型方向。若模型需要贡献一个系统性更新，可另设

\[
g_{mix}=g_{CV}+\alpha g_{path},\quad
\Delta\theta=\operatorname{Update}(g_{mix}).
\]

在独立高保真审计 batch 的小低维子空间中，根据执行闭环有限差分判断 \(\Delta\theta\) 的实际回报方向；若置信下界不足，缩小 \(\alpha\) 或回退。模型分支必须通过以下全部检查：方向一致、幅值比例、事件序列稳定、近邻尺度一致性、板载/训练求解器近似一致性。

这部分是受约束的有偏梯度辅助，不具有无条件 policy-improvement 证明。KL 小只限制策略分布变化，不能证明动力学模型正确，也不能证明不会跌倒。真实审计若仍为H0，则禁用直接pathwise项及pathwise-CV对actor的贡献；只可继续离线审计、模型/critic拟合与已授权的HF-score学习。不能借“CV均值为零”绕过H0。

## 5. 可信度与偏差怎么估计

### 5.1 四种误差分开审计

| 误差对象 | 观察量 | 失败后的处理 |
|---|---|---|
| 状态/接触预测 | 多时域 state error、接触事件 precision/recall、力矩/冲量误差 | 缩短 rollout；不要以单步 MSE 给整条梯度授权 |
| 局部导数 | 因果完整闭环的 \(Jv\) 与 paired central finite-difference；步长扫描 | 同模式区间训练 directional Sobolev；错导数关闭而非只截幅 |
| 事件/active-set 导数 | contact onset、触地时刻、摩擦锥/力矩饱和活动集变化；KKT residual/condition | 同事件路径可用局部 KKT；切换时采用事件敏感性/统计 score，禁止生套单一光滑 Jacobian |
| 回报/critic 导数 | 真实短时域 return difference 对比模型方向；critic \(v^\top\nabla V\) 误差 | 真实 critic 保持HF目标；辅助 critic 限权、独立训练 |

方向探针必须按物理单位标准化：例如相同幅度的 rad、N·m、s、摩擦参数不能直接比较未经归一化的 L2。对相同起点使用 common random numbers，但重复随机种子估计接触噪声；FD 的两边如果处在不同 mode，要标成“事件变化”，不能作为普通平滑导数真值。

置信度不能只来自 ensemble disagreement。多个错误模型可以一致；梯度重复也可以重复犯错。按未参与训练的 material/task/command 区间评估导数校准曲线，给出“通过校准的覆盖率”，同时报告未解析方向比例。

### 5.2 局部导数错误怎么处理

1. 模型内的数值梯度与自身 finite-difference 不一致：先修 autograd、detach、求解器梯度或步长，不能让 RL 掩盖 bug。
2. 模型内一致、对 HF 错：属于模型导数偏差；用 matched counterfactual probes 校准，或只把该模型用作零均值辅助变量。
3. 硬事件导致导数不存在/不稳定：把 mode probability 放到 score branch；连续段才走 pathwise。可审计的事件时间/saltation 只能在 transversality 与 mode 模型成立时使用。
4. solver 未收敛但用“精确 KKT”反传：它并非当前前向算法的精确导数。选择对固定迭代算法反传，或提高收敛质量并核验其近似误差，不能混称。
5. “梯度很小”不代表可信。限幅和投影可能真正形成零灵敏度；错误 backward 也可能错误归零。应区分两者并评估是否浪费采样。

## 6. 研究假设 B：完整闭环 value-Jacobian 与执行器校准

对固定旧策略定义因果完整闭环转移

\[
T_\theta(z,\epsilon)=F\big(z,A(z,E(z,\pi_\theta(o(z),\epsilon)))\big).
\]

同一平滑 mode 内，价值 Bellman 的状态导数目标为

\[
\nabla_z V(z)=\mathbb E_\epsilon\left[
\nabla_z r_{cl}(z,\epsilon)+\gamma_t\left(\partial_z T_\theta\right)^\top\nabla V(T_\theta)
\right],
\]

其中 \(\partial_zT\) 必须含 policy feedback、MPC/WBC、执行器与历史更新；若 duration 可学习，\(\gamma_t=\exp(-\beta\Delta t)\) 的时间依赖也要处理。这里不重新把已有 true-time GAE 当贡献。

JAVE 已指出冻结 action 会遗漏 actor-feedback，且 MSE 不约束 model Jacobian；MB-MIX 已用 simulator Jacobian 做 Sobolev 监督。[L05,L10] 因此我们必须进一步做：

- **执行闭环的梯度目标**，不是裸 physical transition 的 Jacobian。
- **可观测历史/信念空间的定义**，避免假设 critic observation 一定能重建 actor history。
- **局部 directional target 与事件跨越目标分离**：前者在同 mode 内校准，后者以成对真实回报/事件统计约束。
- **奖励价值尺度与导数分别训练**：真实 reward 标量用 \(V_H\)；用于解析终端的 \(V_D\) 使用监督的局部导数但其 value anchors 仍来自 HF，且要审计导数正则导致的标量偏差。
- **因果 actuator prediction 单独校准**，延迟缓冲、command history 与 hidden effort state 必须随候选动作滚动；后验测量模型只提供监督，不可充当前向反事实模型。

可实现目标：

\[
\mathcal L_{V_D}=\mathcal L_{HF\ value}
+\lambda_J\mathbb E\left[\left(v^\top\nabla V_D-\widehat D_v^{HF}\right)^2\right]
+\lambda_B\mathcal L_{closed\ loop\ Bellman\ gradient},
\]

只在可信 mode 内启用第三项；所有 targets stop-gradient，保留 actor 更新时对 \(V_D\) 输入的导数。这个目标没有自动的全局收敛保证：部分可观测状态、非光滑模式、带偏模型及 critic 固定点都要给出限定条件。

独立研究命题：**即使预测 MSE 和 value MSE 相近，闭环导数校准能否显著减少 actor 更新的真实目标反向率，从而提高多任务样本效率与 sim-to-real 鲁棒性？** 如果只在当前材料地图单一任务有效，应降为系统组件结果，而非泛化 RL 算法贡献。

## 7. On-policy 合法性：两种架构不能混淆

### A. 当前风格：上游请求动作 sampled，控制栈属于环境

\[
a^{raw}\sim\pi_\theta,\quad u=E_\psi(a^{raw},z).
\]

PPO 用 raw likelihood 是合法的，映射可以不可逆。前提是更新期间环境/执行规则与采样时一致。若把 \(\psi\) 悄悄改掉，old raw likelihood ratio 无法修正改变后的环境；这种外循环必须重采样，或明确将 \(\psi\) 变成联合随机动作/策略结构并重建概率模型。

### B. AC-MPC 风格：solver 本身属于随机 actor

\[
u^{raw}\sim\mathcal N\big(E_\psi(z,Q_\theta(z),p_\theta(z)),\Sigma_\theta(z)\big).
\]

这时 PPO 可以通过 solver 更新 cost map/\(\psi\)，因为 new/old likelihood 都是在同一个实际 sampled-action 随机变量上计算，并确实依赖 solver 输出。代价是多 epoch 重算 MPC，且要区分采样后固定的 physical limits 与可学习 policy layer。不能把架构 A 的“冻结 E”要求泛化为架构 B 不能学习 controller。[L08–L09]

对混合动作：

\[
\log\pi(m,a\mid z)=\log p(m\mid z)+\log p(a\mid m,z),
\]

\[
D_{KL}(\pi_{old}\Vert\pi_{new})=
D_{KL}(p_{old}(m)\Vert p_{new}(m))+
\mathbb E_{m\sim p_{old}}D_{KL}(p_{old}(a\mid m)\Vert p_{new}(a\mid m)).
\]

这属于概率正确性，不是创新。Gumbel/straight-through 只是梯度估计近似，不能代替分类概率与联合 KL，也不能把新模式在旧策略没有支持的样本通过普通 ratio 补出来。

## 8. 真实与想象 critic 如何分离

建议至少明确三个角色，是否共享 encoder 由梯度污染消融决定：

- **\(V_H\)**：仅 HF 数据价值目标，用于 PPO advantage 与现实性能判断。
- **\(V_M\)**：模型世界自己的 value，用于 imagined return / LF 统计，不得把模型奖励混成 \(V_H\) 真值。
- **\(V_D\)**：在 HF value anchors 上额外校准闭环导数，专用于 pathwise terminal adjoint，可先与 \(V_H\) 同初始化但独立优化。

MFPG-PPO 的 coupled 与 uncoupled LF 若分布不同，还用不同 LF critics；如我们保证同 conditional distribution，可以先共享 LF critic，但必须检查 truncation/GAE/terminal 定义完全一致。[L07] 共享 trunk 若接收 imagined gradients，仍可能改变真实 critic；“head 分开”不足以声称不污染。

## 9. 决定性消融与不应省略的基线

| 研究问题 | 必要比较 | 判定失败的证据 |
|---|---|---|
| A 的校正是否有价值 | HF-PPO、naive add、AGPO权重、MFPG-PPO、条件 score CV、条件 pathwise CV | equal-HF 与 equal-wall-clock 两种预算下均无收益；模型错配后比HF-only更易崩 |
| 事件结构是否是贡献 | step covariance、episode covariance、contact-event covariance；同样探针预算 | 改进仅来自更多数据或更大batch；事件对齐不如普通episode统计 |
| 去偏是否成立 | same-conditional LF vs free-reset LF；cross-fit vs same-batch fit；有限K扫描 | 只报variance下降，真实gradient bias显著升高；所谓零均值校正长期偏离0 |
| B 是否超越JAVE/MB-MIX | MSE-only、state/action Sobolev、frozen-action JAVE、完整feedback、加入actuator-history、event-aware | 必须依赖privileged未来电流才能提升；删掉反馈/执行器导数性能不变 |
| 训练快是否真的快 | tuned PPO、FastTD3、SAPO/SHAC；统一reward/action/range/simulator资源 | 仅样本少，GPU小时更大；导数probe成本被排除 |
| 跨任务泛化 | 硬地、滑地、弹性地、材料突变、延迟/饱和/质量变化；留出整个条件组合 | 只在训练材料或同一命令段有效 |

主要结果报告：success/fall rate、跟踪误差、恢复时间、步态对称/足滑/接触时序、能耗、约束违例、实际执行命令变化、HF样本数、总GPU/CPU小时、probe和solver反传成本，以及实机policy/MPC/WBC各频率与P95/P99尾延迟。每类性能都给 seed/episode 不确定性，不用“最好的一条视频”替代统计。

## 10. 并行工作的具体安排

可以同时推进，但先区分“并行研究/代码”与“同一GPU六份训练”。25核CPU和96GB VRAM不是6×16GB自动可行的证明，CPU仿真吞吐、GPU SM、图编译和solver factorization都可能竞争。

推荐学习线独立的三个工作单元，与材料/步态/MPC线共享明确接口：

1. **概率与梯度审计单元**：冻结现有baseline，实现闭环 causal-state replay、raw likelihood、joint KL与probe registry，只输出报告，不训练新actor。
2. **研究A单元**：实现HF/LF paired+independent采样、条件CV、cross-fitting与预算记录；先小型可知真梯度任务，再当前机器人。
3. **研究B单元**：实现闭环Jv/value directional supervision，做完整feedback/actuator-state消融；其模型可供A使用，但结果独立评价。

heavy runs 由实测 profiling 决定2或3任务并行；其余任务做数据预处理、代码核验、日志与paper reading。训练终止按算法步数/收敛/异常指标，**不设置后台按墙钟到点 kill**。本轮不启动训练；保留active W59。

### 10.1 最低可行实验，先回答一个明确问题

第一轮不需要同时训练十个大模型。按三个门槛推进：

1. **冻结 actor 的离线导数实验**：从既有评估轨迹按稳态支撑、摆腿、触地与饱和四类各取若干完整扩展状态；以16个起点、4个归一化方向、2个扰动尺度、3组共享噪声、正负两侧为一个可管理的起始设计。一次最长5步rollout可以复用前缀得到1/3/5步结果。完整记录实际执行量、future current是否因候选动作改变、mode/active-set与return。先回答“主要错误来自模型、执行器、求解器还是critic”，不更新actor。
2. **估计器实验**：用相同HF批构造HF-only、naive pathwise、variance-gated、score CV与conditional pathwise CV；在独立大样本HF参考梯度的固定低维投影上测偏差、方差、方向正确率与成本。先用可知动力学的小控制任务验证零均值与cross-fitting，再用机器人数据；有限差分不能解析的事件方向单独统计。
3. **单次/短链更新试验**：只选在门槛2有优势的两种估计器与HF-PPO比较；每次记录更新前后独立HF评估，3 seeds用于筛选而非论文最终统计。条件是完整概率比、critic角色和执行器因果接口已通过审计。只有确实降低误导更新率且改善equal-budget性能，才扩展到多材料/步态泛化和实机。

若第一门槛表明主要误差来自非因果current输入或执行映射遗漏，应先修模型接口；继续叠加PPO目标只会把结构性错误藏进超参数。

## 11. 新颖性与证据诚实性清单

- 本文新增12篇，至少 AGPO/AHAC/SAPO/MB-MIX/MFPG-PPO/AC-MPC/SmoothedTurboMPC/JAVE 读到方法和实验，并对关键偏差/假设查阅附录。
- 未把题目/摘要读取当全文深读；未声称每篇每页都已审核。
- source ledger 区分 paper body、代码仓库定位、commit pin 与实现阅读，pin HEAD 不等于审核代码。
- 额外代码核验：JAVE当前commit `35256b6c8e243d658d6379ff8d17cde8e01d8211` 的 `compute_gradient_bellman_targets` 明确 `actions_sg = stop_gradient(actions_batch)`；LDM训练仅平方残差。AC-MPC当前commit `c59e53aec11c1fffa8b69d99b0ee7879ba7ccb28` 的 policy 文件完整阅读，MPC输出送入Gaussian mean，`distr_identity=True`；其实现细节与论文架构叙述并非完全一致，不能把当前HEAD当原论文release。
- MFPG-PPO代码在官方项目页仍标 coming soon；不能借旧 MFPG 仓库声称已审新算法实现。
- 本文没有复现论文数值。所有候选创新仍需先查更近邻、实现与可证伪实验；不以“组合了更多模块”替代贡献。

## 参考入口

1. L01: https://www.nature.com/articles/s41586-025-08744-2
2. L02: https://proceedings.mlr.press/v235/gao24m.html
3. L03: https://proceedings.mlr.press/v235/georgiev24a.html
4. L04: https://arxiv.org/html/2412.12089v2 ; https://rewarped.github.io/
5. L05: https://arxiv.org/html/2503.01178v1 ; https://ojs.aaai.org/index.php/AAAI/article/view/34419
6. L06: https://arxiv.org/pdf/2503.05696v4 ; https://xinjie-liu.github.io/mfpg-rl/
7. L07: https://arxiv.org/pdf/2610.02505v1 ; https://xinjie-liu.github.io/mfpg-ppo/
8. L08: https://rpg.ifi.uzh.ch/docs/TRO25_ACMPC_Romero.pdf
9. L09: https://arxiv.org/pdf/2611.09937v1
10. L10: https://arxiv.org/pdf/2608.02069v1 ; https://diffloco.martin-opat.com/
11. L11: https://arxiv.org/pdf/2505.22642v3 ; https://younggyo.me/fast_td3/
12. L12: https://proceedings.iclr.cc/paper_files/paper/2024/file/cf73d57b6dcda32b293df7c2d5341f49-Paper-Conference.pdf
