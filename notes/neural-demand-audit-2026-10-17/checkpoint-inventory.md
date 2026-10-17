# 真实神经策略、原生接口与已存轨迹盘点

日期：2026-10-17（Asia/Shanghai）。本盘点只读取已有checkpoint/trace及源码；另对一个旧非PPO策略做已存状态上的前向身份核对。未仿真、训练、计算参数SVD或改变权重。逐checkpoint SHA256、张量形状/dtype、全部trace字段见[checkpoint-inventory.json](checkpoint-inventory.json)。

## 1. 当前最可靠的路线

**优先真实PPO原生对象。**正式seed17/29/43、四arm共有12份可读取的真实训练后PPO checkpoint。它们属于原生三状态单执行器servo，不可直接接到后来的六状态双执行器探针。当前已有PPO NPZ没有完整原生观测，因此要在完整真实评估状态上计算反馈混合导数，需先按原seed与最终checkpoint做一次明确记账的确定性评估重放，逐项核对旧NPZ，再保存这次重放的完整观测。

该重放是**重新仿真原测试**，不是读取到了原本不存在的完整观测，更不是恢复了旧随机训练batch。盘点阶段未执行重放；其后按照[独立冻结协议](replay-protocol.md)已完成全部12组，结果见第6节。

另有2026-10-15的18份真实神经checkpoint与完整状态trace可以直接核算，但该训练器不是PPO。它们只用于明确哪些资料现成可用，不替代本轮选定的真实PPO路线。

## 2. 真实PPO网络与执行接口

12份正式checkpoint均包含`actor,critic,design_mean`。实际张量全部为float32且有限；同一结构：

```text
actor mean: Linear(17,64) -> Tanh -> Linear(64,64) -> Tanh -> Linear(64,1)
actor log_std: 1 learned scalar
actor total: 5378 parameters = 5377 mean parameters + 1 log_std
critic: Linear(17,64) -> Tanh -> Linear(64,64) -> Tanh -> Linear(64,1)
critic total: 5377 parameters
design_mean: 4 raw logits
```

结构来源为[`ppo_backend.py`](../../src/response_ppo/ppo_backend.py)。完整12组源码hash与正式manifest匹配，mean网络末层是线性输出。确定性mean-controller的动作导数对`log_std`恒为零，不能将其5378个参数全部当作5378个确定性反馈自由度。

原生17维观测定义于[`env.py`](../../src/response_ppo/env.py)：

| Python切片 | 意义 |
|---|---|
| `[0:3]` | 物理`x=[p,v,a]`除以`[1,1,1.2]` |
| `[3:6]` | 连续参考`z=[p_ref,v_ref,a_ref]`同尺度归一化 |
| `[6:9]` | `x−z`同尺度归一化 |
| `[9]` | 位置命令`c/.45` |
| `[10]` | 响应速度命令`(eta−1)/.2` |
| `[11:15]` | `2*sigmoid(raw_design)−1` |
| `[15:17]` | 当前支持区间lo/hi除以1.2；不可行时输入0 |

这里的3维z是参考状态，**不是第二个物理执行器的3个状态**。虽然增广`[x,z]`有6个数，也不能等同六状态双执行器探针。

实际动作必须按arm区分：

- `*_penalty`：确定性评估动作`u=1.2*tanh(mu(obs))`。
- `*_support`：可行时`u=midpoint+halfwidth*tanh(mu(obs))`；支持为空时执行独立于actor的minimax fallback。

因此support组的`Dtheta u`含halfwidth因子，`Dtheta Dx u`还含支持区间、归一化和重复误差特征的状态导数。不可只计算网络对17维obs的偏导，就宣称得到了物理状态反馈导数。支持边界的min/max/root分支、空区间和单点区间需要明确；fallback时actor参数不影响实际动作。

原生物理对象为离散Euler单输入servo：

\[
x^+=\begin{bmatrix}p+h v\\v+h(a-\sin p-.2v-.1v|v|)\\(1-h/\tau)a\end{bmatrix}
+\begin{bmatrix}0\\0\\h/\tau\end{bmatrix}u
+\begin{bmatrix}0\\h\\0\end{bmatrix}w,
\quad h=.02,\ \tau=.12.
\]

其运动与控制接口不同于后续RK4双执行器对象。微步物理`Dx`可以按固定z/command/design定义；若分析跨步闭环，则必须把连续z的推进及宏命令更新一起声明。

## 3. 已存数据缺口与可重放条件

正式24个短/长NPZ只含15项float32 metric数组，短`(256,16)`、长`(1024,16)`：`task,total,V,excess,empty,action,reference_input,support_width,position,reference_position,actual_domain_violation,reference_domain_violation,tube_violation,target,geometric_bound`。

缺少完整x/z、command、obs、潜动作、old logprob、value/next_value及advantage。最终checkpoint与汇总history不能补回原训练batch。完整字段曾在训练时驻留内存，但保存逻辑只写确定性评估metric，见[`pilot.py`](../../src/response_ppo/pilot.py)。

原first-short评估可按下列原条件固定重放：

- `eval_seed=seed*10000+5000`、16环境、**恰好256步**、最终checkpoint中的design_mean、各arm原support/penalty映射。
- fresh原生环境与planner，每次重新开始warm-start；float32、CPU单线程。
- CEM每次规划在fork_rng中使用`eval_seed*10000+t`；环境另有自己的固定generator。网络初始化后载入原state_dict，不影响这些固定随机源。
- 不能取long评估的前256步代替short，因为初始化goal数量随steps变化，继而改变随机数消耗及phase。
- 仅实例化`GaussianActor`并载入权重；不需要`PPOBackend`、optimizer或critic。

旧日志每run耗时约1.69—2.08秒，仅为历史计时参考。每run原预算为4096个仿真转移、416次planner实例（26×16）、78次目标批调用、14144条候选序列、113152个候选宏步（1131520个预测微步）。若冻结全部12组一次短测，新增重放仿真转移应记49152，planner实例4992，候选预测微步13578240。不能仍写“0仿真转移”。

重放额外记录obs、x/z、command、raw_design、支持区间及分支、znext/uref、latent mean及next fields后，逐15字段核对旧NPZ。只有重放核对成功，才把这些完整字段称作匹配已存评估的重建观测；仍不能称为旧随机PPO训练样本。

## 4. 原生MPC本身的需求边界

当前`_prepare_command`以`[z,raw_design]`调用planner，`sequence_objective`只用解析`response(z,c,eta,raw_design)`。它不读取actor θ，也不读取实际x；z持续独立推进，规划边界不重置为x。

因此在固定design下，当前原生MPC成本对神经反馈参数θ的直接导数为零。它可以对command/design产生任务需求，但要将新反馈Jacobian能力纳入需求，必须新增并说明θ相关闭环响应/鲁棒读出接口。不能把独立附加的诱导增益梯度称为原MPC已经具有的非零信用。

这项边界不妨碍验证真实神经策略可实现哪些反馈方向；它说明“真实网络可实现性”和“原MPC提出对应需求”是两个必须分别检查的缺口。

## 5. 旧非PPO网络：仅作现成材料备查

2026-10-15正式seed11/23/37、六arm共18份checkpoint，网络为12→32(Tanh)→32(Tanh)→1，共1505个float32参数。特征只由物理x和当前command构造；实际动作`1.2*tanh(actor(features))`，网络承担完整物理动作。训练为已有score/proximal小试，不是PPO。

每份trace含`state(100,8,3)`、`action(100,8,1)`、`reference(100,8,3)`、`command(100,8,1)`、`disturbance(100,8)`、`initial_state(8,3)`。state是转移后状态；转移前状态可直接用`[initial_state; state[:-1]]`对齐，不需要积分。

只对seed11/joint进行了一次800个已存观测的前向身份核对，没有仿真：最大动作误差`1.7881393432617188e−7`、均值误差`8.970018505749522e−9`；第一样本完全一致。checkpoint hash为`f4d77dd0d9d2b3f3b075f5236e97a3da6fbe73eb249b9a6627b5dda0825b6aed`，trace hash为`add16cce79e818337d587c66f1cc49cebb112eb251cc6d011c607d5afeb1103e`。这只核实资料对应关系，不证明训练收益或与原PPO同类。

所有checkpoint均用`torch.load(...,weights_only=True,map_location="cpu")`读取；NPZ使用`allow_pickle=False`。原始权重/轨迹仍在忽略目录；本次只公开原创盘点、路径、hash及必要结构信息。当前盘点阶段没有参数导数、SVD、仿真重放或权重修改。

## 6. 后续固定重放已完成，完整原生锚点现已可用

按冻结协议执行全部12组first-short评估，所有256步×16环境×15metric与旧NPZ**数值和字节均精确一致，最大差0**。全部checkpoint执行前后字节与actor张量不变。新增完整观测位于`work/neural-demand-audit-2026-10-17/replays/{seed}-{arm}.npz`，每组字段/shape/hash见[重放证据](replay-results.json)。

新增`obs(256,16,17)`、`x/z/znext(256,16,3)`、`command(256,16,2)`、`raw_design(256,16,4)`、`mean(256,16,1)`以及支持区间、分支、discriminant、P/B/E和末态。它们是本轮重放后匹配旧评估的重建原生记录，可用于真实神经策略的状态/参数导数；不是原先已经保存的观测，不是随机训练batch。

这一步实际增加49152个确定性仿真转移、4992次planner实例、169728条候选序列、13578240个预测微步；12组仿真墙钟合计17.583秒。训练更新、权重更新、梯度优化器步和checkpoint写入均为0。三状态PPO与六状态双执行器不兼容、旧随机batch缺失及原native MPC不依赖actor参数的边界均保持。
