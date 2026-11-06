# W39 原件复核后的 r29 数学与接口审查

审查基线：`07b0ee3e15c9f10e9b1a6a1de3997bec36f39556`。审查对象是尚未实施、尚未启动的 W40 方案。本文没有运行物理、训练或 GPU 实验，也没有修改运行代码。数值反例只使用普通代数，见同目录 `r29-algebra-examples.py`。

## 结论：撤回上一版中心 QP 的等价性承诺，缩小为一种明确的策略投影

上一版公式

\[
\min_\eta \tfrac12\eta^TH\eta+(sQ_{ux}\delta x)^T\eta,
\quad l(x)-u_{ff}\leq\eta\leq h(x)-u_{ff}
\]

**一般不能恢复当前源码的反馈策略**。原因不仅是矩阵插值：当前 `K` 已经包含原目标角 BoxQP 的活跃集合，公式却丢掉了原 `Q_u` 所确定的活跃乘子和自由集合。即使没有新 PD 约束、没有模型误差，它也可能改变已选 line-search 的原反馈。不能继续把它写成“只将 PD 实现域加入原反馈，其他自动等价”。

本轮建议实施的单一机制改为：

\[
\begin{aligned}
c(x,t;s)&=\operatorname{clip}_{[a,b]}
 \left(u_{ff}(t)+sK_{old}(t)\delta x\right),\\
u^*(x,t;s)&=\arg\min_{u\in\mathcal U(x)}
 \tfrac12(u-c)^TH(t)(u-c),\\
\mathcal U(x)&=[a,b]\cap[l_\tau(x),h_\tau(x)].
\end{aligned}
\]

这里先完整保留旧策略及旧目标角裁剪，`H` 采用产生对应策略的原非对角正则化 `Quu_reg`。当 `c` 已在 PD 未饱和区间内，直接原样返回 `c`，因此包括旧活跃行、原 `alpha`、原插值、原目标角 clip 在内的行为均保持。当 PD 约束介入时，非对角度量才对十二轴重新分配。这个精确的“未介入时保持旧策略”性质，比上一版的广义等价性承诺更适合当前单机制实验。

应称它为**状态相关的策略投影**。它不是已经完成的状态约束 DDP、没有证明递归可行，也不能保证十个子步和真实 TM02 的稳定性。本轮保留原 backward/value 递推作为产生基策略的方法，明确记录投影对其局部模型的偏离；不要只替换一个 `K`，却声称完成了约束最优控制的完整推导。

| 启动前必须改的合同 | 精确处理 |
|---|---|
| 新 QP 的中心和线性项 | 用旧目标角裁剪后的 `c`；QP 的线性项为 `-H*c`，不再把 `s*Q_ux*delta_x` 当作旧反馈的普遍等价式 |
| 原目标角活跃集 | 保留原 `K_old`，不从完整 `H^-1*Q_ux` 重建或释放旧活跃行 |
| action-candidate 的反馈缩放 | 明确 `s=1`；只有 nominal-feedback 搜索使用其搜索到的缩放值 |
| H 来源和生命周期 | 保存本次成功 backward 实际使用的完整 `Quu_reg`；与策略一起复制，不能用更新后的 `mu` 重建 |
| seed policy 没有 H | 初始 nominal/startup 读数显式标记 `legacy_seed_no_metric`，仍走原 seed 路径；第一次真实持有动作前必须已经安装成功 backward 的 metric，不默默用单位阵 |
| 不适用的旧 expected decrease | `ratio=null` 并进入明确的正则化分支，不能用 `surprise=0` 伪装成预测失败 |
| 等界或空交集恢复 | 固定轴先消元；当前依赖版本的 `mju_boxQP` 连 `lower==upper` 都拒绝 |
| 名义动作与执行动作查询 | read-only probe 不改 QP warm-start、上次执行动作或物理记录；null-state nominal 查询不假装拥有当前 PD 状态 |
| GPU 分支 | 对齐修订后的策略投影，不移植已撤回的 `Q_ux` 中心公式；预测 A/B 仍是固定持有目标的物理导数 |

## 1. 实际源码的数学语义

以仓库保留的 `s1-go2-r25-source` 为源码底本，叠加实际 r26、r27、r28 patch；这些 patch 没有重写下面的 `RiccatiStep`。r26 早期 `.inc` 草稿不作为现行实现。

### 1.1 `Quu`、`Quu_reg`、`Qxu_reg` 不是同一个对象

`backward_pass.cc:90–153` 先生成原局部二次项：

\[
Q_u=c_u+B^TV_x',\quad Q_{xu}=c_{xu}+A^TV_{xx}'B,
\quad Q_{uu}=c_{uu}+B^TV_{xx}'B.
\]

默认 control regularization 对应 `H=Q_uu+mu*I`。上游固定版本的 `settings.h` 默认 `regularization_type=0`；当前可见配置没有另行设定 `ilqg_regularization_type`。真正实施仍应把实际 type 与 `regularization_used` 一同记录，不能仅凭字段名解释。

对于 value/state-control 类型，源码计算了正则化的 `Qxu_reg`，但求 K 的两个分支实际都读**未正则化的 `Qxut`**，随后 `Qxu_reg` 又被当作临时乘法缓冲覆盖（`backward_pass.cc:176–191,207–212,235–247`）。所以：

- 在当前默认 control regularization 中，`Qxu_reg=Qxut`，这不是已经证明的 W39 失败原因。
- 如果换正则化类型或导出矩阵，必须区分“计算过的正则化交叉项”和“实际用于 K 的交叉项”。不能借本轮顺手改变该历史语义。
- `backward_pass.Quu` 保存原二次项，**不是**自动带 `mu` 的矩阵；`boxqp.R` 是自由集合上的 Cholesky 结果，**不是**完整十二维 H。
- 成本导数已有 `weights[i]/T` 归一化（`cost_derivatives.cc:133–154`）；不额外除以时域、乘控制周期，或用重新归一化的 H 搭配原 `mu`。

### 1.2 当前 K 来自原目标角 BoxQP

源码解的是

\[
k=\arg\min_d\;\tfrac12d^THd+Q_u^Td,
\quad a-\bar u\leq d\leq b-\bar u.
\]

若原目标角活跃集合为 A，自由集合为 F，则源码为

\[
K_A=0,\qquad K_F=-H_{FF}^{-1}(Q_{ux})_F.
\]

见 `backward_pass.cc:155–196`。只有原问题全部轴自由时，才有完整的 `K=-H^-1 Q_ux`。不能因为 H 是十二维，就认为旧 K 也是完整十二维无约束解。

### 1.3 alpha 和 feedback scaling 属于两个不同搜索

`planner.cc:647–651` 先把 `alpha*k` 加入各候选的动作数组，形成候选前馈 `u_ff`。随后 `ActionRollouts` 在 `:662–688` 使用

\[
\operatorname{clip}_{[a,b]}(u_{ff}+K\delta x),
\]

其反馈缩放实际为 **1**，而不是 `alpha`，也不是此时成员变量中残留的 `feedback_scaling`。

另一个 `FeedbackRollouts` 才在 `planner.cc:713–720` 设置并搜索 `feedback_scaling`。`CopyFrom` 原本不复制该标量；安装策略后显式设置为 1。若把 action-candidate 简单改成调用原 `Action()`，却没有重置/显式传入缩放，就可能错误沿用 nominal-feedback 搜索中的缩放网格，新增一层反馈缩放错误。

r28 保留生成赢家的候选 policy/reference，不把 accepted rollout 重新锚定为 reference。这一修正应保留。accepted trajectory、nominal linearization trajectory、feedback reference 分别承担不同角色，不可以为“看起来一致”而强行覆盖。

## 2. 上一版公式的具体反例

### 2.1 没有新 PD 约束也会改变旧反馈

一维取 `H=1, Q_u=-2, Q_ux=1, bar_u=0`，原目标界 `[-1,1]`。原 backward 解为 `k=1`，上界严格活跃，`K_old=0`。

当 `alpha=.5, delta_x=.1`，旧候选动作是 `.5`。上一版中心 QP 以 `.5` 为中心，其自由解 `eta=-.1`，输出 `.4`。没有任何新 PD 约束介入，策略已经变了。

当 `alpha=1`，旧候选动作仍为 `1`。上一版中心 QP 给出 `.9`；而带原 `Q_u` 的原二次问题在该小扰动下仍严格取上界 `1`。原因是把中心移到 `u_ff` 后，删掉 `Q_u+H*k` 就同时删掉了非零活跃乘子。在 `delta_x=0` 时新 QP 的约束乘子全为零，不能声称继承原活跃集合的局部反馈。

这个反例只是否定通用公式，不是声称已经重建了未保存的 W39 H 或判定某个实际动作必然由此生成。r29 尚未实施，不能把它倒叙为 W39 的物理失败根因。

### 2.2 分别插值 H、C、K 仍然不相容

两节点取 `H=(1,4), C=(-1,-8)`，则 `K=(1,2)`。半点旧线性插值 `K=1.5`，但 `-interp(H)^-1 interp(C)=1.8`。保持旧 K 为中心的度量投影没有这个矛盾：H 决定约束介入时的距离度量，旧插值 K 决定未介入时的原策略，不再冒称二者满足完整原 `Q_ux` 恒等式。

### 2.3 非对角投影确实可以改变实际十二轴分配

取

\[
H=\begin{bmatrix}2&1\\1&2\end{bmatrix},\quad c=(1,0),
\]

新约束只有 `u_1<=.2`，第二轴范围足够宽。逐轴 clip 得 `(.2,0)`，度量投影得 `(.2,.4)`；对应目标函数分别 `.64` 与 `.48`。投影会主动改变尚未饱和的第二轴，因此和原独立 force clipping 有实质区别。

反过来，若用对角 H，单时刻逐轴投影经原 affine PD 后可以和原 force clip 产生相同的瞬时力矩。即使瞬时相同，保持目标不同也可能在后续子步分离。不要把“瞬时相同”扩展为“整个 2ms 保持完全相同”，也不要把非对角重新分配写成必然提高稳定性。

## 3. 建议实施的策略投影合同

原 actuator gain/bias 是 `60` 和 `[0,-60,-5]`，原 gear 按模型保留。本 Go2 原 motor 为 unit gear；关节位置、速度必须按 actuator→joint 映射读取。完整状态存储为 19 维 qpos 加 18 维 qvel，共 37 维；扰动为 36 维切空间，不能将二者的索引直接混用。

\[
l_{\tau,i}(x)=q_i+\frac{\tau_{min,i}+5\dot q_i}{60},\qquad
h_{\tau,i}(x)=q_i+\frac{\tau_{max,i}+5\dot q_i}{60}.
\]

与原 target range `[a_i,b_i]` 取交后解

\[
\min_u\;\tfrac12u^THu-(Hc)^Tu.
\]

`c` 必须是旧目标角 clamp **之后**的值。若直接以未裁剪的 `u_ff+sK*delta_x` 为中心，即使 PD 没介入，非对角投影也会改变旧目标角 clipping 的行为，失去本轮所需的干预边界。

当所有 `c_i` 在交集内，直接返回 `c`，不经数值求解器再近似还原。nominal 自己若已经请求过大原始 PD 力矩，即使 `delta_x=0`，投影也应介入；不能把 `delta_x=0` 当成无条件旁路。

物理仍执行原 2ms 目标保持及十个 0.2ms PD/force clipping；每个子步不得重解 QP、重选目标或偷偷变成 5kHz policy。新限制是当前采样状态下的未饱和目标区间，**不是**原来物理没有力矩上限，也不是全保持区间的可行性证书。

一个需要明确记录的剩余数值风险是：投影可能把名义目标恰好放到 PD 饱和折点。原 one-sided `u+epsilon` 差分在上界与下界选取的分支不同，不能保证给出双侧可用的控制权限。当前回合不同时更换差分方式或平滑物理；把 FD 时的原始 PD、饱和分支、投影活跃界随关键 snapshot 保存，GPU也先复现这一原差分语义，而不是悄悄用 AD 的另一侧导数“修好”它。

### 活跃边界导数怎样记录

在时间固定、H 冻结且活跃集合局部固定时，设 `G=dc/d(delta_x)`，它已经包含原 K、实际 s 和旧 target clamp 的局部导数。对新 QP 的活跃轴：

\[
\left(K_{eff}\right)_A=B_A.
\]

原目标界活跃时该行 `B_A=0`；PD 边界活跃时

\[
B_i=e_{q_i}^T+\tfrac5{60}e_{v_i}^T.
\]

自由轴为

\[
\left(K_{eff}\right)_F
=G_F-H_{FF}^{-1}H_{FA}(B_A-G_A).
\]

这与上一版直接套 `-H_FF^-1(s*C_F+H_FA*B_A)` 的语义不同。若旧 target clip 未激活，则 `G=sK_old`；也可以写成使用 `C_eff=-H*K_old` 的形式，但 **`C_eff` 不是原 Q_ux**。新旧字段必须不同名。

角/PD 两个界重合、投影开始/退出、活跃轴切换时不存在唯一的普通导数。记录实际活跃来源与切换状态，不能把一侧局部 Jacobian 当作全局光滑增益。本文建议该 Jacobian用于解释和诊断，本轮不反向替换原 backward 的 K。

## 4. 为什么本轮不“顺手”半改 backward/value

若实施完整状态相关约束 DDP，至少要在 `RiccatiStep` 增加 nominal state、状态相关边界、当前 KKT 活跃乘子/自由集，重新定义 nominal 改进 `k` 和其状态导数 `K`，并一起传播：

\[
\begin{aligned}
V_x&=Q_x+K^T(Q_{uu}k+Q_u)+Q_{xu}k,\\
V_{xx}&=Q_{xx}+K^TQ_{uu}K+Q_{xu}K+K^TQ_{ux}.
\end{aligned}
\]

这正是现有源码完整形式（`backward_pass.cc:218–248`），但其输入必须是同一个明确定义的局部控制律。当前固定目标界的 `K_A=0` 不能直接沿用到 PD 边界。另一方面，把部署时新状态的投影 Jacobian 写回已完成的 backward，也不能补上过去所有节点的 Bellman 递推。

还必须解决 alpha<1 时旧活跃轴可能在区间内部的问题：保留原前馈 `alpha*k`、沿用原非零活跃乘子、又声称在原固定 box 上重新求解同一局部最优 QP，这三项一般不能同时成立。完整改法需要重新规定候选同伦/线搜索，不能靠给部署 QP 多加一次 `Q_u` 解决。

因此本轮合理边界是：原 iLQG 产生基策略，全部候选与最终执行共用新投影，真实 nonlinear rollout 继续评估新策略。它是可实施的局部策略修订；原 backward 仍是近似优化器，不能据此宣称已解完整受约束 DDP。该范围足够具体，可以直接实现进入已授权 W40，无需再开泛泛算法研究轮。

## 5. 空交集、等界与 native BoxQP 接口

### 5.1 确定性的恢复规则

原 target box 是硬界。逐轴取最小原始 PD 力矩违反：

- 若 `l_tau>b`，固定该轴为 `b`；
- 若 `h_tau<a`，固定该轴为 `a`；
- 其余轴保留正常交集。

这对应先在原 target box 内最小化 PD 原始请求对原 torque interval 的逐轴距离，再在这些最小违反解中最小化 H 距离。因为 affine PD 单调且 kp 为正，空交集轴的最近端点唯一，不需要任意 slack 惩罚权重。受影响轴的物理力矩仍由原 force clip 限制。

这种恢复**没有把空交集变成可行**。日志须分别记录空交集轴、最小原始力矩违反 Nm、实际 target、实际饱和力矩和 `restoration=true`。非空但只剩一个点的交集是 `touching_singleton`，应与空交集分开。

### 5.2 等界不能直接交给本项目的 `mju_boxQP`

已直接核对固定 MuJoCo dependency commit `088079eff0450e32b98ee743141780ed68307506` 的 `src/engine/engine_util_solve.c`：`mju_boxQPoption` 对 `lower[i]>=upper[i]` 调用 `mjERROR`。所以等界轴和恢复后固定轴都必须先消元。

在 `f=.5*u'Hu+g'u` 中，若固定轴为 A，其余为 F，自由子问题是

\[
\min_{u_F}\;\tfrac12u_F^TH_{FF}u_F+
 (g_F+H_{FA}u_A)^Tu_F.
\]

必须保留 `H_FA*u_A`。同目录代数例子中，`H=[[2,1],[1,2]]`、中心零、第一轴固定 1，第二轴正确为 `-.5`，漏交叉项会错误得 0。所有轴固定时直接构造结果，不调用 `n=0` 的 BoxQP。

该版本返回非负数表示自由轴数；`MAX_ITER`、`MAX_LS_ITER` 也可能非负返回，不能将 `nfree>=0` 对外宣称为严格 KKT 收敛。应记录候选 primal violation、自由梯度、活跃轴的对偶符号/驻点残差。实际求解失败必须显式保留失败状态；不能静默换单位阵或把一个与声明机制不同的动作记为正常投影。若工程上需要保留原 clip 作为失败 fallback，应事先定义并明确记录该退化，不能算作新 QP 正常完成。

## 6. 最小实现接线：矩阵、插值、启动与三个调用入口

### 6.1 H 与 policy 同生同灭

在成功 `RiccatiStep` 内、scratch 被复用前，导出当前节点完整 `Quu_reg`、原 `Q_u/Q_xu`（仅用于解释）、原自由集合和 `regularization_used/type`。H 是每个成功 backward 的策略数据，不是全局 planner 可变工作区的指针。

新数据必须进入 `Allocate/Reset/CopyFrom`、候选拷贝、winner 安装、`previous_policy`、horizon 尾端处理。上一计划/上一候选的 H 不可与新 K 拼接；不使用正则化更新后的 mu；不只保存压缩后的 `boxqp.R`。失败 backward 的部分节点不能成为已安装 snapshot。

投影需要完整 H 为正定。原 constrained backward 只对当时自由子块做 Cholesky，不能把“旧 BoxQP 成功”当作所有情况下完整 H 都正定的证明。如果完整 H 无法建立有效正定 metric，就在原 backward 正则化循环内处理本次失败，使重新生成的 k/K/H 使用同一个 mu；不要在每次部署时单独随意加对角项。

### 6.2 初始 policy 尚无 H

`Reset` 时只有原 repeated home/seed action 和零 K，没有成功 backward 的 H。第一次 `NominalTrajectory/FeedbackRollouts` 位于首次 backward **之前**，startup 还会反复读取 action。不能因此访问未分配矩阵或默默塞 `I`。

精确约定：`metric_valid=false, status=legacy_seed_no_metric`。这只用于第一次求导之前的 nominal seed rollout，以及明确的 read-only startup/nominal probe，保持原 seed 路径。第一轮成功 backward 产生全部节点 H 后才设置 valid。正常既有 startup 的规划迭代数不增加；在真正向 plant 持有动作并推进物理之前，已安装策略必须有有效 H。如果原全部 startup backward 都失败，则记录控制器初始化技术失败，不能用 r28 seed 继续采样却声称 r29 正常工作。区分 probe 与 applied action 可随同一动作响应/日志元数据返回，不需要增加独立准入实验。

### 6.3 保留原插值、单独定义正定 metric 插值

当前 policy representation 为线性。`u_ff`、reference state、K 继续按原逻辑查询，reference 四元数仍归一化、`StateDiff` 仍使用 36 维切空间。H 使用同一动作时间网格上的线性凸组合，端点按原动作 horizon 截取规则，不单独外推。

正定矩阵的凸组合保持正定；一般 cubic 插值不保证这一点。本轮不顺手开启 cubic，也不独立插值 H/C 再重建 K。时间处于节点时，离散候选与时间查询应取同一个 snapshot 节点；节点间按原规则查询 center 和 metric。它们不用满足旧 `-H^-1 C` 恒等式，因为新机制已明确是保留旧 K 的 metric projection。

### 6.4 共用一个纯 evaluator，而非只改部署 `Action`

| 入口 | center 的来源 | 明确的 s | 使用新投影 |
|---|---|---:|---|
| `FeedbackRollouts` | 旧已安装 policy 的前馈/reference/K/H，经 query time 插值 | nominal-feedback 搜索值 | 是；初始 seed 是上节显式例外 |
| `ActionRollouts` | 本候选已加一次 `alpha*k` 的前馈及本次 backward K/H，按 index | 1 | 是 |
| 实际部署 | 选中并安装的同一候选 reference/前馈/K/H，经 query time 插值 | 安装值，现行为 1 | 是 |
| `nominal_action=True` 读数 | 原名义前馈的 target-clipped 读数 | 不以 null state 求 PD 边界 | 保留其原读数语义，另报 at-actual-state nominal 投影时须显式给 state |

计划级字段分别记录 `nominal_feedback_scale`、`action_step_alpha`、`installed_feedback_scaling`；不要用同一个 `feedback_scaling` 字段同时代表三者。新机制下 `nominal_action=True` 的无状态读数，也不等于有实际 rollout state 的 `FeedbackRollouts(s=0)`：后者仍须考虑 nominal 自身的 PD 不可实现。

候选之间不要共享一个会被修改的 QP scratch 或 warm-start。每次 evaluator 从传入 snapshot/state 计算动作，诊断 read 不改变下一次动作；不会因 Python先读 actual、后读 nominal 就改变求解路径。每个 rollout 的投影次数、最大更改、恢复/失败状态随候选记录，使最终选中动作与候选评估可关联。

## 7. expected decrease 与正则化不能留一个未定义分支

原 `dV` 来自旧 k/K 和原二次近似。新增投影介入之后，真实候选的控制增量已不是原 `alpha*k+K*delta_x`，旧 scalar `expected` 通常不能当作新策略的准确预测下降。

建议本轮采用无需新参数的明确规则：

1. 选中候选全程新投影未改变旧输出，且满足 r28 原来的可行比较条件时，保留旧 ratio/正则化处理。它仍是原本意义下的局部模型比值。
2. 若新投影修改选中候选或进入恢复，记录 `cost_ratio_valid=false, cost_expected=null, cost_surprise=null, reason=actuator_projection_changed_policy`。
3. 前后 trunk 都可行但 ratio 因上述原因不适用时：实际 cost 降低就原样保持 `regularization` **及 `regularization_rate`**，不调用 `ScaleRegularization(1)`；cost 不降则沿用原增大规则。不要把“没有可解释 ratio”编码为 `surprise=0`，否则 r28 的 `surprise<.1` 会将所有这种正常介入都当作坏预测反复增加 mu。
4. 前后任一个 trunk 不可行时，保留 r28 原 `violation_improvement` 分支；不将 actuator-restoration 和 trunk-restoration 混成一个量或改变原 `(v,J)` 排序。`restored_to_feasible` 日志分类修正独立完成。

这是一项明确而有限的优化器处理约定，并不是新的下降证明。后续若要恢复可用 ratio，应计算包含实际投影局部控制律的线性/二次预测 rollout；不能只把原 dV 改名或用实际下降同时充当预测分母。

## 8. 多次 action 查询：当前可确认的副作用与新日志要求

`scripts/s1_go2_r11_runtime.py:483–500` 在启动阶段每次规划后都会读取反馈动作；`:521–539` 在每个实际 tick 先读取 actual，再读取 nominal，最终把 earlier actual 数组写入 Python plant 的 `data.ctrl`。因此 `feedback-actions.jsonl` 中的 RPC 读数并非每一条都意味着执行过物理；startup 同时刻多 plan 读数要保留其身份。

原 r27 只对有 state 的 recorded action 写日志，null-state nominal 不写。固定上游 `grpc_agent_util.cc:199–228` 的 `averaging_duration=0` 路径没有推进物理，nominal 明确传 null state。仓库 `agent_service.cc:179–193` 仍会把每次 GetAction 返回值写入服务器内部 `data_->ctrl`；所以 actual 后的 nominal 请求会覆盖服务器这个 ctrl 缓冲。它没有覆盖 Python 随后真实执行的 actual 数组，不能据此直接宣称 W39 实际执行了 nominal。

新 QP 不能引入更多有意义的查询副作用：read-only nominal/probe 不更新上次执行 active set、warm-start、held target 或物理 action-id。最稳妥是同一次 actual evaluation 随结果记录原 `u_ff`、旧 `raw_K_delta_x`、旧 target-clipped center `c`、新投影 `u*`、实际 s、实际使用 snapshot、bounds、status；无需为每个诊断再调用一次会改状态的控制器。

建议新增 `query_kind`/`applied_action_id`（或等价已知上下文）、`metric_valid/source_plan_id`、`projection_changed`、`center_before_projection`、`projected_delta`、原 nominal 和最终 raw-PD，以及每个轴的 active bound source。旧 raw-K、原目标裁剪量与新 QP 修正量分开；不能把所有修正都写进名为 `raw_K_delta_x` 的旧字段。实际 plant记录才决定哪一条读数进入十子步持有。

当前GetAction响应为float32，须分开QP数学解和实际持有值。分别记录RPC前double投影解u_star与实际float32往返后held target u_held、delta_u及同一采样状态下两者的PD/约束残差。QP状态按u_star的primal/KKT判定，执行统计仍用u_held；若u_held=double(float32(u_star))且delta_tau=60*delta_u能解释边界越出，单列serialization_boundary_crossing，不当求解失败，也不从原严格饱和计数中删除。此解释只限持有开始同状态，后续子步饱和另记；不改RPC协议/物理阈值。

完整 H/K/reference snapshot 每 plan 保存一次，动作行引用 snapshot/插值节点和权重。H 的完整时域记录是下一版新证据；旧 W39 未保存的 H 不能通过“原件已上传”推断为存在。

## 9. GPU 第三任务的具体校准边界

用户已授权本轮第三 GPU MPC 适配与测试，旧“只允许两个计算槽/不做 benchmark”文字不再阻止必要工作。该任务可以和两条主线并行；以下是具体数值对齐对象，不是要求先再做一轮泛泛调研。

| 层次 | CPU 现有语义 | GPU 必须匹配/明确区分 |
|---|---|---|
| 预测模型 | `Agent::PlanIteration` 设置 prediction timestep/integrator，调用 `MakeDifferentiable`，按已有开关禁 warmstart | 不能把原 plant MJB 直接 put_model 后当作同一预测模型 |
| native contact | `utilities.cc:59–74` 对 joint/geom/explicit pair 的 `solimp[0]` 都置零；规划后恢复 | 只改 geom、漏 joint/pair 或用原刚地参数，会混入模型变化 |
| 控制持有 | 2ms target、十个 0.2ms 子步，原 affine PD 与 force limits | 不能一大步 2ms，不能每个子步重新投影或更新目标 |
| 积分器/精度 | 当前模型生成器选 implicitfast；实际 dependency 固定，CPU double FD | 先匹配原版本/设置和 FP64；新 backend/version 与 FP32 的差异单列，不把它们都叫性能优化 |
| 物理 A/B | 固定 target 的物理 `F(x,u)` 导数，36 维切空间，one-sided FD epsilon 1e-6 | 不对包着 QP 的闭环 `F(x,policy(x))` 直接 AD 后当作原 A/B；那会混入 `B*K_eff` |
| 十子步合成 | `P_0=I,Q_0=0; P_{j+1}=A_j P_j; Q_{j+1}=A_j Q_j+B_j` | 对应同一 held u、同一基准子步；不能把独立逐步导数简单求和 |
| 路径风险 | r26 当前几何 direct-u 导数为零，活动子步的 `r_x*P/r_x*Q` 回链；最早最大值 tie | 任意 max 的 AD、后积分传感器、另一个 tie 规则都不自动等价原 FD/活动分支 |
| 自定义预测接触 | `R10ContactFilter/Force`、四冻结世界坐标 patch、因果 userdata、残差 callbacks | C++ callback 不会随 MJB 自动成为 JAX/Warp 代码；不得额外给真实 TM02 隐状态 |
| 新策略层 | 本文的旧 target-clipped center + H metric projection | 不让 GPU 分支继续实现撤回的 Q_ux 中心公式；candidate/full-planner 对比用同一 r29 版本 |

当前 r26 实际 patch 见 `s1-go2-r11-S5-r26.patch:273–337`，而不是旧 `.inc` 草稿。GPU 可以先保留 CPU backward/BoxQP，用批量预测和 FD 获取收益，不需要第一步把全部 Riccati 迁移。

这里“固定 target 的导数”具体是：求导点的 u 取 nominal rollout 实际保存的目标动作，可能已经由新投影选出；进入物理 F 的有限差分时，把 u 当作独立输入并按原定义持有。不能在每一个状态扰动上再次调用投影，不能把开环 A/B 和闭环 Jacobian 混用。

已存状态的 GPU 校准应明确固定输入（同一 state、同一 target、同一 userdata、同一预测参数）再比较一步/十步及 A/B/C/D。此前 accepted+20ms 与 actual 的差异包含 closed-loop action 分离，不能直接拿它当“同输入模型残差”的 GPU容差目标。若旧时域 H 未保存，为 GPU 测试重算的新 H 应标记为新校准结果，不冒充 W39 原始记录。

GPU 后端会改变数值实现，所以需报告状态/导数误差、活动子步/约束集变化和完整 planner 耗时，不只内核毫秒。主线保留原 CPU 真 TM02；GPU 未完成某层就报告其实际层次，允许继续工程推进，不冒称已经等价或稳定。

## 10. 实施顺序与证据限度

在启动任何 W40 任务前，先将主计划、JSON、AGENTS、启动提示词中的旧中心 QP/完整约束反馈承诺替换为本审查的策略投影合同，并同步第三 GPU 分支。正常实现接线按 H 快照→共享 evaluator→三个调用入口→ratio/空区间/初始 seed→日志做完，再使用既有构建和已授权联合任务流程；不新增主线物理准入轮。

数值与源码证据足以说明上一版方案需要纠正，**不足以证明新的投影必然解决 W39 跌倒**。它检验的是“原局部反馈在执行器饱和时缺少状态相关多轴再分配”这一具体机制。真实接触预测、闭环输入分离、材料域退出和 RL暴露不足仍须由各自原始证据判断，不能因采用 QP 就一起宣称修好。

### 源码及一手接口依据

- 仓库固定源码：[backward_pass.cc](https://github.com/kaiwen123-yang/MCP-RL/blob/07b0ee3e15c9f10e9b1a6a1de3997bec36f39556/experiments/terrain_memory/s1-go2-r25-source/mjpc/planners/ilqg/backward_pass.cc)，关键 `90–248`。
- 候选、缩放与复制：[planner.cc](https://github.com/kaiwen123-yang/MCP-RL/blob/07b0ee3e15c9f10e9b1a6a1de3997bec36f39556/experiments/terrain_memory/s1-go2-r25-source/mjpc/planners/ilqg/planner.cc)，关键 `168–224,434–535,637–732`；[policy.cc](https://github.com/kaiwen123-yang/MCP-RL/blob/07b0ee3e15c9f10e9b1a6a1de3997bec36f39556/experiments/terrain_memory/s1-go2-r25-source/mjpc/planners/ilqg/policy.cc)，关键 `53–179`。
- 当前增量：[r26 patch](https://github.com/kaiwen123-yang/MCP-RL/blob/07b0ee3e15c9f10e9b1a6a1de3997bec36f39556/experiments/terrain_memory/s1-go2-r11-S5-r26.patch)、[r27 patch](https://github.com/kaiwen123-yang/MCP-RL/blob/07b0ee3e15c9f10e9b1a6a1de3997bec36f39556/experiments/terrain_memory/s1-go2-r11-S5-r27.patch)、[r28 patch](https://github.com/kaiwen123-yang/MCP-RL/blob/07b0ee3e15c9f10e9b1a6a1de3997bec36f39556/experiments/terrain_memory/s1-go2-r11-S5-r28.patch)。
- 原始 PD 建模：[s1_go2_r11_model.py](https://github.com/kaiwen123-yang/MCP-RL/blob/07b0ee3e15c9f10e9b1a6a1de3997bec36f39556/scripts/s1_go2_r11_model.py)，关键 `91–105,170–177`；实际调用：[s1_go2_r11_runtime.py](https://github.com/kaiwen123-yang/MCP-RL/blob/07b0ee3e15c9f10e9b1a6a1de3997bec36f39556/scripts/s1_go2_r11_runtime.py)，关键 `483–578`。
- 预测模型：[agent.cc](https://github.com/kaiwen123-yang/MCP-RL/blob/07b0ee3e15c9f10e9b1a6a1de3997bec36f39556/experiments/terrain_memory/s1-go2-r25-source/mjpc/agent.cc)，关键 `286–365`；[utilities.cc](https://github.com/kaiwen123-yang/MCP-RL/blob/07b0ee3e15c9f10e9b1a6a1de3997bec36f39556/experiments/terrain_memory/s1-go2-r25-source/mjpc/utilities.cc)，关键 `59–74`。
- 固定上游版本：[settings.h](https://github.com/johnzhang3/mujoco_mpc/blob/e00c47a5adb9856af2e0f24231bb3a60d5be23c4/mjpc/planners/ilqg/settings.h)；[grpc_agent_util.cc](https://github.com/johnzhang3/mujoco_mpc/blob/e00c47a5adb9856af2e0f24231bb3a60d5be23c4/mjpc/grpc/grpc_agent_util.cc)，关键 `199–228`；本仓库 [agent_service.cc](https://github.com/kaiwen123-yang/MCP-RL/blob/07b0ee3e15c9f10e9b1a6a1de3997bec36f39556/experiments/terrain_memory/s1-go2-r25-source/mjpc/grpc/agent_service.cc)，关键 `161–193`。
- 固定 MuJoCo 依赖：[engine_util_solve.c](https://github.com/google-deepmind/mujoco/blob/088079eff0450e32b98ee743141780ed68307506/src/engine/engine_util_solve.c)，`mju_boxQPoption` 的 bounds 检查、自由子块分解、终止状态与返回值。该外部一手源码已直接读取，并非依据最新版本的接口印象推断旧版本。
