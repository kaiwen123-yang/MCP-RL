# 既有数据可用性审计与固定片段

日期：2026-10-16（Asia/Shanghai）。本次只读取既有文件并做代数计算，没有训练、新物理转移、MPC求解、参数扫描或重建旧随机轨迹。逐文件SHA256、实际字段及形状见[data-audit.json](data-audit.json)。

## 结论与适用边界

**未保存完整PPO训练批次。** 当前文件不能支持“在实际旧PPO批次上重新计算score／advantage／单批策略更新”的声称。可以用已存经典反馈轨迹中的固定状态、实际动作与执行器状态检查算法的控制关系；这必须称作“固定历史数据上的代数决策见证”，不能称作PPO训练批次重放或已实现的反馈学习收益。

正式PPO数据实际位于`work/ppo-mechanism-2026-10-16/formal/seedrun-{17,29,43}/`；`work/outputs/local`不存在。三个manifest均为`completed`，各记录40次更新、16个环境、每批256步。三个manifest记录的六份源码SHA256均与本次读取时的相应源码一致。

## 实际文件与内容

| 文件 | 实际读取范围 | 能支持的检查 | 缺口 |
|---|---|---|---|
| 正式PPO `*-short-trace.npz` / `*-long-trace.npz` | 3 seed × 4 arm × 2长度，共24文件；短数组`(256,16)`，长数组`(1024,16)`，均float32 | 确定性评估成本、实际动作、位置及约束统计 | 不是训练数据；无完整观测、原始高斯动作、旧logprob、value/next_value、advantage与逐批版本 |
| 正式PPO `*.pt` | 12份最终checkpoint，均安全读取 | actor/critic结构与最终design值 | 无逐批旧策略、逐批旧critic、优化器状态或训练buffer |
| 正式PPO `*.json` | 每arm完整40项history以及result/manifest | 已发生的更新汇总、四维design score gradient及版本追溯 | 汇总不能恢复逐样本actor score或PPO方向 |
| `work/capability-probe-2026-10-16/study.mat` | `evaluation(3,4)`、`probes(2,3)`、`designRuns(18,2)` | 包含完整状态和执行器状态，可作不重新仿真的控制关系检查 | 全部由经典反馈生成，非PPO批次 |

NPZ的15项字段为：`task,total,V,excess,empty,action,reference_input,support_width,position,reference_position,actual_domain_violation,reference_domain_violation,tube_violation,target,geometric_bound`。即使有`total`也不能把评估成本当作原训练batch的reward和advantage。

PT只包含`actor,critic,design_mean`。actor为17→64→64→1的MLP加一个`log_std`，共5378参数；critic为17→64→64→1，共5377参数；`design_mean`形状`(4,)`。读取这些权重不等于取得生成训练数据时的旧策略。

MAT各轨迹包含`theta,cfg,t,x,raw,u,command,disturbance,stageCost,metrics`。通常形状为`x(301,6)`、`raw/u(300,2)`、`command/disturbance/stageCost(300)`、`t(301)`。六状态依次为位置、内部角度、前进速度、角速度、执行器1状态、执行器2状态。

## 保存逻辑证据

- [`pilot.py`](../../src/response_ppo/pilot.py)第38—45行只在训练时构造`obs/raw_action/reward/terminated/truncated/log_prob/value/next_value`；第76行组成内存`Rollout`，第112行交给`backend.update`，之后未保存该对象。
- 同文件第133—138行对`training=False`的确定性评估保存metric arrays；第142行仅保存最终actor、critic和design均值。第124—127行仅保留每次更新汇总。
- [`env.py`](../../src/response_ppo/env.py)第201—209行给出的metric字段，与实际NPZ一致。
- [`runCapabilityStudy.m`](../../matlab/capability/runCapabilityStudy.m)第93行保存上述MAT结构；文件明确说明为经典反馈且无RL学习。

以后恢复训练时若需可重算的单批见证，应额外持久化实际观测/必要历史、采样原始动作、执行动作、old logprob、reward/value/next_value、终止/截断标记、策略与响应模型版本、命令段边界与必要响应/资源统计。**本次没有为了补缺启动采集、训练或新rollout。**

## 本次固定的数据片段

按本轮算法见证的事先选择，固定`evaluation{1,1}`（case3、linear_current_mpc）第一次扰动开始的五步段：Python索引45—49、MATLAB控制时刻46—50，状态端点为`t=0.90 → 1.00 s`。未扫描其它片段来选择有利结果。该段高层命令均为`0.7352628471147254`，扰动均为`−1.5`。

定义实际输入共模`f=(u1+u2)/2`、差模`g=(u1−u2)/2`以及执行器差状态`a=(x5−x6)/2`。本段已存数据：

| t / s | f | g | a（动作前） | 差模输入余量 |
|---:|---:|---:|---:|---:|
| 0.90 | 0.454393681503122 | 0 | 0 | 1.345606318496878 |
| 0.92 | 0.453984389375153 | 0.045828642702433 | 0 | 1.346015610624847 |
| 0.94 | 0.450317054012284 | 0.093168574784509 | 0.008307323142642 | 1.349682945987716 |
| 0.96 | 0.444016771248746 | 0.141756728109025 | 0.023690057566077 | 1.355983228751254 |
| 0.98 | 0.435624921141859 | 0.191355996086388 | 0.045091912972195 | 1.364375078858141 |

记录端点`a(1.00)=0.07160509186953326`，执行器限额`U=1.8`，`tau1=tau2=0.1`。

已保存轨迹由每低层采样4个RK4子步产生，采样`dt=0.02`、子步`h=0.005`。因此复核数字记录使用

\[
\alpha_{RK}=\left(1-z+\frac{z^2}{2}-\frac{z^3}{6}+\frac{z^4}{24}\right)^4,
\quad z=h/\tau,
\quad \beta_j=(1-\alpha_{RK})\alpha_{RK}^{K-1-j}.
\]

这里`alpha_RK=0.8187307619695058`，不是连续系统精确ZOH的`exp(-dt/tau)=0.8187307530779818`。`beta=[0.08144952248877015,0.09948267033821773,0.12150840662064073,0.14841070137433823,0.18126923803049422]`。

直接用记录的实际输入计算：

\[
a_K-\alpha_{RK}^K a_0-\sum_j\beta_jg_j=-1.11\times10^{-16}.
\]

这验证已记录执行器差状态与对应离散实现关系一致，**不是新轨迹或新策略的预测验证**。在相同实际共模`f_j`下，输入关系`|f_j|+|g_j|≤U`给出本段正端点上界`0.8560636965345728`；这个上界只是有限段输入资源可实现性，不是全任务最优值，也未考虑未知扰动下的因果策略实现限制。

## 读取安全与资源账本

NPZ使用`numpy.load(..., allow_pickle=False)`；checkpoint使用`torch.load(..., map_location="cpu", weights_only=True)`；MAT使用`scipy.io.loadmat(..., simplify_cells=True)`。未加载任意pickle执行代码。

`study.mat` SHA256：`076c3e11416c8c87dc3fe59a10940f97cefbb5c99476789e089a0fe1e99a0a01`。

本次审计新增物理转移0、MPC调用0、训练更新0、优化器步0、参数扫描候选0。原始MAT/NPZ/PT仍留在忽略目录；公开材料仅包含必要数值片段、原始文件hash和原创审计。

## 公开重算与原文件复核分开

为使仓库读者不依赖忽略目录中的MAT，已公开[source-segment.json](source-segment.json)：仅含固定五步的6行状态、5行实际输入、时间、命令、扰动、四项执行器配置，以及来源路径/hash和选段规则。该文件SHA256为`42e6c3623d808de882277c6e22c4cf8f30b38de4c54430105a130301c97d2bca`。

[`joint_update_witness.py`](../../scripts/joint_update_witness.py)默认`--source excerpt`，直接从公开片段重算，本次excerpt模式的所有科学数值与此前原MAT模式逐字段完全一致。此模式明确记录`original_mat_verified_this_run=false`，不能称为再次核验原文件。

可访问原MAT时使用`--source mat`，脚本核验原始SHA256、逐字段确认提取与公开片段完全一致，再重算。同一读取入口已单独验证原MAT与片段一致；该读取核验不执行物理仿真或学习。JSON分别保存原文件验证状态与公开片段hash，避免把可复算性和原始来源再次核验混为一谈。
