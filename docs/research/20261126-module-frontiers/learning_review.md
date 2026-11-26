# 最终审查：NeuralActuator → MPC/WBC → RL 的因果接口与 conditional CV

日期：2026-11-26。只审查已有方案与已读来源，不新增检索、不修改项目仓库、不运行训练。对应主专题 `learning.md` 的公式和适用范围已同步修正。这里的 HF 默认指项目高保真仿真；没有物理机器人证据时，不称为实机数据。

## 1. 审查结论

方案可以继续作为实现设计，但必须同时满足以下边界：

1. **Observer 与 generative actuator 分工严格。** 前者解释截至当前的测量，后者预测候选未来命令；未来实测电流不能进入反事实预测。
2. **执行器与接触力归属唯一。** 吸收接触效应的等效力矩不能再次叠加 PANN 接触力。
3. **完整闭环只求一次链式导数。** 物理、执行器、MPC/WBC 可以因子化，但包含它们的整体 WM 不再重复乘这些 Jacobian。
4. **H0 对 actor 的解析动力学路径是硬关闭。** 直接 pathwise 与 pathwise-CV 都关闭；允许离线计算诊断值，不据此改变 actor。
5. **Conditional CV 仅在明确条件下保持当前 HF 梯度估计器的期望。** 不使整个 PPO、GAE、有限样本拟合系数、Adam 或被筛选的最终更新无偏。
6. **首版 CV 限定 fresh batch、一次梯度更新。** 多 epoch 复用旧噪声后，不能继承固定参数下的条件零均值论证。

“可以实施”的结论不意味着当前模型已经通过上述门槛。若真实审计为 H0，当前正确结果仍是保留 HF baseline 并修模型/接口。

## 2. 建议的接口契约

避免把 actuator latent 与 policy action 都记为 `a_t`，建议使用以下命名。

### 2.1 真实时间线

\[
\underbrace{y_{\le t},u^{cmd}_{<t}}_{\text{已发生的信息}}
\xrightarrow{O_\eta}
(b_t^{mat},h_t^{act},P_t)
\xrightarrow{\pi_\theta,E_\psi}
u_t^{cmd}
\xrightarrow{\text{真实执行}}
y_{t+1}.
\]

`live telemetry at t` 的时间戳必须明确：当前电流通常反映此前执行的命令，不能因为字段同名就认为它由当前尚未施加的候选命令产生。Observer 输入包括命令、测量各自时间戳、采样 dt、延迟缓冲；当前观测在控制决策开始前可用才算因果。

| 接口 | 输入 | 输出 | 不允许的输入/推论 |
|---|---|---|---|
| `observe(history_le_t, live_y_t)` | 已发生的命令、关节/基座测量、电流、电压、温度、时间戳 | actuator/material posterior及其不确定性 | 当前候选动作作用后的测量；把不可辨识材料与执行器误差强行给成确定值 |
| `actuate_predict(h_act, candidate_command, x, voltage, temperature, dt)` | 当前状态、因果历史/latent、候选命令、当前外生量或其预测 | 候选力矩/有效effort、next actuator latent、可选current预测及不确定性 | 回放未来真实current；更改command但保留依赖旧command的future effort |
| `contact_predict(x, b_mat, mode, memory, dt)` | 候选状态、材料belief、接触模式及世界坐标记忆 | 接触力/冲量、材料状态/记忆演化 | 将执行器头已吸收的接触项再次相加 |
| `closed_loop_step(z, policy_noise, candidate_policy)` | 完整状态与已声明的策略/控制器 | next-state、reward、事件、求解诊断 | 隐藏调用未来真实telemetry；跨候选共享可变solver缓冲 |

未来电压和温度若未建模，在 1/3/5 步短预测内可以显式采用“保持当前值”的近似，但必须作为模型假设记录；不能使用原始轨迹未来测量提高预测评分。应单独比较 teacher-forced prediction 与 command-conditioned autoregressive rollout。

### 2.2 想象中的 belief 是 prior prediction

首版 imagined rollout 只推进 actuator/material belief 的预测分布。没有新真实测量时，不执行真实 posterior correction，也不能无依据缩小不确定性。

**这会带来一个必须承认的近似：**真实机器人后续收到传感器信息会重新估计 belief，纯预测 imagined rollout 没有这条未来信息反馈。因此完整计算图的导数，不自动等于真实部分可观测闭环的导数。冻结actor的HF审计应允许真实执行栈按真实新测量更新observer；两者差异必须记入模型偏差。未来若加入生成式传感器与虚拟观测更新，应作为独立扩展验证，不能把真实未来记录当替代品。

### 2.3 力与力矩只分配一次

可采用清晰的浮动基方程：

\[
M(q)\dot v+h(q,v)=S^\top\tau_{act}+J_c(q)^\top\lambda_{contact}+d_{unmodeled}.
\]

其中 `actuate_predict` 只负责 actuator output；PANN只负责被定义为contact的分量；WM残差只负责剩余误差。若现有NeuralActuator头预测的是包含接触/其他负载的等效广义力，而不是已分离的 \(\tau_{act}\)，先将它用于observer/监督或重新设计分解，不能直接插入右端作为独立执行器力矩。

这一分解存在辨识难度：相似的关节运动误差可能来自接触、执行器、惯性或延迟。联合模型必须保留相关不确定性并通过干预/激励检验，而不是仅凭loss下降宣称分别识别了材料与电机参数。

## 3. 梯度只计算一条真实声明的路径

### 因子化模型

\[
a^{req}\rightarrow E_{MPC/WBC}\rightarrow u^{cmd}
\rightarrow A_{causal}\rightarrow\tau
\rightarrow F_{body/contact}\rightarrow x'.
\]

对一次路径贡献：

\[
\partial_\theta J=
\partial_{x'}J\,\partial_\tau F\,\partial_u A\,
\partial_a E\,\partial_\theta a+\text{其他正确的反馈项}.
\]

状态反馈、history、belief prior、warm start、WBC积分器的导数同样属于完整图。不能只相乘一次动作前馈链就称包含全部闭环反馈。

### 整体闭环模型

若 \(F_{full}(z,a^{req})\) 的训练目标和前向已经含控制器、actuator、contact，则直接求 \(\partial_aF_{full}\)。**不得再乘 \(\partial E\)、\(\partial A\) 或额外接触Jacobian。**

审计必须区分：同mode局部导数、硬事件时间变化、active-set变化、solver未收敛。硬前向/平滑反向是surrogate gradient；它可以有用，但不是硬闭环的精确导数。

## 4. Conditional CV：原报告必须修正的四件事

### 4.1 固定什么，才有条件零均值

令 \(\mathcal F\) 包含：当前固定 \(\theta\)、冻结的模型/observer/critic/normalizer、已完成独立校准的系数/门控，以及未来rollout噪声产生之前的起点 \(C\)。记

\[
g_H=g_H(C,\epsilon),\quad Y_0=Y(C,\epsilon),\quad
\bar Y_K=\frac1K\sum_{k=1}^K Y(C,\epsilon'_k).
\]

其中 paired LF 与 independent LF 在 \(\mathcal F\) 条件下使用相同算法、相同初始扩展状态分布、相同时域/终止定义；独立均值样本与paired噪声独立。

若 \(B\) 和启用门控是 \(\mathcal F\)-可测量，即已经固定，且相关矩存在，则

\[
\mathbb E[B(Y_0-\bar Y_K)\mid\mathcal F]=0,
\]

\[
\mathbb E[g_{CV}\mid\mathcal F]=\mathbb E[g_H\mid\mathcal F],\qquad
g_{CV}=g_H+B(Y_0-\bar Y_K).
\]

这是**梯度估计器的期望保持条件**。它不要求LF模型导数等于HF真导数；相关性决定能否降低方差。可是它也不移除 \(g_H\) 已有的GAE/critic/截断/裁剪目标误差。

独立估计的有限K均值本身可以在这些条件下保持零均值；有限K并不必然引入偏差。真正容易破坏条件的是模型、系数、门控或均值目标对当前paired噪声的自适应依赖。

### 4.2 有限K的最优系数

在同一 \(C\) 下，若每个LF样本协方差为 \(\Sigma_{YY}\)，则

\[
\operatorname{Cov}(Y_0-\bar Y_K\mid C)
=(1+1/K)\Sigma_{YY}.
\]

因而不带正则的最优线性系数为

\[
B^*_K=-\Sigma_{HY}\big((1+1/K)\Sigma_{YY}\big)^{-1}.
\]

秩不足时用广义逆，实际估计可使用ridge，但ridge版本不再称无约束精确最优。原报告不含 \(1+1/K\) 的式子只适用于 \(K\to\infty\) 或已知均值。

若LF均值复用paired样本、K组噪声并不独立，或批次共享随机性，需要用实际 \(D=Y_0-\bar Y\) 的 \(\Sigma_{HD}\) 和 \(\Sigma_{DD}\)，不能照抄上式。

### 4.3 有限样本拟合系数和选择门控

\(\widehat B\) 从当前 \((g_H,Y_0)\) 同批拟合时，一般有

\[
\mathbb E[\widehat B(Y_0-\bar Y)]\ne0.
\]

对 \(\widehat B\) 做stop-gradient，只是禁止autograd继续求导，**并不消除统计依赖**。EMA也不是独立性的证明。

首版采用此前独立episode校准的 \(B\)/gate，在当前fresh batch完全冻结。若用cross-fitting，应按独立采样单位划分，并防止跨fold共享未来轨迹、target训练或数据相关normalizer；随机打散同一episode的时间步不够。

本批未来接触事件、HF/LF是否同时终止、当前校正后方差下降与否，不能直接作为选择 \(B\)/gate 的条件。起点已有mode可以进入 \(C\)。LF未来事件若有价值，可统一定义为辅助变量的一部分，例如 \(Y'=\mathbf1\{\text{LF触地}\}Y\)，然后两组LF都估计该 \(Y'\) 的同一条件均值。不能只删除paired一侧的“坏事件”。

固定时域加absorbing-terminal统计比事后只保留“HF/LF都存活”更容易守住同分布条件。

### 4.4 PPO多epoch与最终优化步

第一步后 \(\theta_j\) 已依赖本批噪声 \(\epsilon\)。此时旧paired噪声在给定 \(\theta_j\) 后的分布不必等于fresh independent noise。因此“paired与independent都用最新 \(\theta_j\) 重算”是必要的实现一致性，但**不足以恢复严格条件零均值**。

首版只做固定 \(\theta\) 的估计器实验，以及fresh HF batch的一次CV更新。多epoch版本只能标成另有偏差评估的研究近似。若完整PPO更新后追加一次严格CV辅助更新，应重新采集该当前策略的fresh HF batch，并将额外采样计入成本。

首次更新在 \(\theta=\theta_{old}\) 上，ratio=1，PPO clipping尚不发挥限制作用。因此应称“单次score actor-critic更新＋PPO概率/KL基础设施”，不能说已经得到完整多epoch PPO创新。

此外，即便 \(g_{CV}\) 与 \(g_H\) 均值相同，经gradient norm clipping、Adam预条件、KL回滚或HF接受筛选后，最终 \(\Delta\theta\) 的期望也未必相同。报告应把梯度统计与优化后真实性能分开。

## 5. 两种PPO架构的合法性保留

| 架构 | 随机动作与概率比 | 同batch可更新的量 |
|---|---|---|
| A：请求动作在控制器之前采样 | `a_raw ~ pi_theta`；保存真实raw sample与old logp；下游E是环境执行映射 | actor参数；E/observer若作为环境的一部分则冻结。修改E需重采样或重新定义联合策略 |
| B：MPC作为策略的一部分 | `u_raw ~ N(mu_solver(theta, psi, saved_state), Sigma)`；新旧概率都依赖solver均值 | θ、ψ可经正确policy Jacobian更新；必须重算新均值，保留旧采样概率 |

两种架构的概率变量都不能换成不可逆限幅后的动作后还沿用原Gaussian logp。架构B评估同一个样本时，solver的warm start和内部状态必须来自该样本所记录的上下文；不能因minibatch排列或全局可变缓存而改变“策略函数”。

H0在这里指 **WM/动力学/执行器pathwise horizon为0**。它关闭所有这类解析actor贡献，包括pathwise CV。未来架构B的正常HF-score PPO可能需要对MPC策略均值本身求导，那属于另一个独立的policy-Jacobian正确性门槛，不应假借H0授权为动态模型反传。首个最低可行实验采用架构A并冻结E，避免两种门槛混淆。

## 6. 最小算法伪码

以下是实现规格，不是本轮已运行的程序。`H_trust`只是解析梯度允许的预测步数，**不是后台进程完成时间**。

```text
Inputs:
    frozen baseline actor pi_old
    declared action architecture A (initial implementation)
    independent calibration set D_cal
    disjoint HF audit states D_audit with complete reset snapshots
    causal observer O, actuator A, contact/PANN P, residual WM R
    HF critic V_H, derivative critic V_D, normalization snapshot

0. Seal versions and causal state schema
    freeze parameters of O, A, P, R, E, V_D and normalizers
    preserve input derivatives in the analytic graph
    assert candidate rollout reads no measured telemetry from time > t
    assert actuator/contact/residual force components are assigned once

1. Frozen-actor causal audit
    for start C in D_audit:
        restore x, material memory, observer history, actuator delay state,
                MPC warm start, WBC state, clocks and noise streams
        for paired direction, perturbation scale, repeated noise:
            run HF full execution for +epsilon and -epsilon
            run candidate-conditioned model with predicted future effort
            collect 1/3/5-step prefixes, physical dt, contact and active sets
            compare prediction, Jv, executed-command and return directions
    H_trust = largest tested prefix passing preregistered independent checks
    if no prefix passes: H_trust = 0

2. H0 branch
    if H_trust == 0:
        allow offline model/critic diagnostics and already-authorized HF PPO
        forbid model-pathwise actor update AND pathwise-CV actor contribution
        publish causal/gradient failure report; STOP analytic branch

3. Fit a conditional auxiliary estimator (offline, independent of update batch)
    use D_cal to fit block coefficient B_K and pre-action-context gate
    include finite-K mean variance in coefficient choice
    freeze B_K, gate, projection basis and all Y-defining parameters
    do not fit/gate on the next batch's future events or variance result

4. Fresh single-update CV experiment
    theta = pi_old.parameters
    collect a fresh HF batch under this fixed theta and fixed execution E
    save raw actions, old logp, base policy noise, complete starting states
    build HF advantages from frozen HF value targets
    compute g_H at this fixed theta

    for each starting condition C:
        Y_pair = model_path_gradient(C, shared stored policy noise, H_trust)
        Y_ind[1:K] = same function from same C with independent fresh noise
        use identical reward/termination/event/statistic rules in both groups
        correction[C] = frozen_gate(C) * B_K(C) * (Y_pair - mean(Y_ind))

    g_CV = g_H + aggregate(correction)
    record raw gradient bias/variance diagnostics BEFORE nonlinear optimizer
    take at most ONE candidate actor optimizer step for this fresh batch
    apply existing correct joint-KL/rollback and independent HF evaluation
    report accepted-update statistics separately from gradient statistics
    discard this batch for any claimed strictly zero-mean CV actor reuse

5. After actor decision
    train O/A/P/R and V_H/V_D only in their declared data/target roles
    update calibration only for a later independent actor batch
    version all components and repeat audit after material model changes

6. Resource scheduling
    benchmark isolated and actual co-running combinations
    allocate by throughput, peak memory, CPU demand and P95/P99 latency
    do not infer six simultaneous trainings from 96GB capacity alone
    use algorithmic steps/convergence/error gates, no wall-clock kill deadline
```

如果想保留标准多epochPPO，首轮就让它作为单独baseline完整运行；不要为了给CV贴PPO标签把上述严格一次更新限制隐藏起来。后续可以研究近似多epoch版本，但新增偏差与吞吐收益应是论文需要回答的问题。

## 7. 最小报告字段

- `snapshot_sha`：actor、observer、actuator、PANN、WM、controller、normalizer各版本。
- `action_semantics`：architecture A/B、raw/logp定义、限幅位置、mode probability。
- `causal_inputs`：各输入时间戳、未来telemetry访问次数应为0、current为measured或predicted。
- `force_partition`：actuator/contact/residual各输出定义、量纲与坐标系。
- `audit`：1/3/5步有效覆盖率、resolved方向比例、方向正确率、幅值误差、event变化比例、未收敛KKT比例。
- `cv`：K、实际independent样本数、B拟合数据来源、gate使用的时间范围、paired/independent统计定义、均值差及其区间。
- `optimizer`：raw g统计、clipping/Adam后step、joint KL、回滚、独立HF更新前后性能分别记录。
- `compute`：HF/LF/probe/solver-backward成本分开；实际并发组合的总吞吐与尾延迟。

这些字段让“因果完整”“未额外引入模型偏差”“实时”和“H0关闭”成为可审查的实现属性，而不是文稿形容词。

## 8. 依据与阅读范围

文献依据沿用 `learning_sources.json`：MFPG/MFPG-PPO的控制变量与条件起点方差讨论；AC-MPC的solver-inside-policy结构；JAVE的冻结动作与critic Jacobian偏差；SmoothedTurboMPC的KKT/求解收敛边界。本文有限K系数、固定参数与自适应多epoch区别、因果接口和伪码是本次推导/设计，不作为论文作者已经给出的保证。
