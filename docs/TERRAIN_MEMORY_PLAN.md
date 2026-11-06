# S1：W39 / r28 完整复核与 W40 稳定性、GPU MPC 三任务推进计划

日期：2026-11-06（北京时间）。**本次提交完成分析与下一轮授权计划；没有在 D57 实现、构建或启动 r29 / GPU 后端。** W39 数值分析固定于 [e51acd5](https://github.com/kaiwen123-yang/MCP-RL/commit/e51acd535445f617bd0486650e67df4bcc116481)；后续上传状态核对至 7150a6b，保留其服务器上传脚本、索引与 manifest 更新。W38/W39 原启动计划及执行历史保存在 [原主计划](https://github.com/kaiwen123-yang/MCP-RL/blob/e51acd535445f617bd0486650e67df4bcc116481/docs/TERRAIN_MEMORY_PLAN.md)，不改写为未执行。

## 当前交接（执行后只更新这6行）

- 当前任务：W39 / S5-r28 两流训练和末端评价已结束；下一任务 W40_STABILITY_WITH_PARALLEL_GPU_MPC 已获用户授权、尚未实现/启动，计划为两条 r29 联合主线加一条独立 GPU MPC 适配测试任务。
- w39-a：job-20261105T032354Z-w39-a 已结束；新增1024/4至8593/33，tail0、excluded145，最后ep284；自身update33 E30在10.476s body_fall，E60阻塞，紧凑Git已发布。
- w39-b：job-20261105T032354Z-w39-b 已结束；新增1024/4至8593/33，tail0、excluded145，最后ep280；自身update33 E30在17.592s body_fall，E60阻塞，紧凑Git已发布。
- 任务能力：W39训练62回合＝59次自然失败＋3次截断；58加载、30机身入条带、26进入且内部足加载；合法去程/返程/往返均0。下一主修订是状态相关执行器约束进入局部反馈及候选同一路径，尚无稳定通过证据。
- 交付：W39紧凑Git与原始诊断均已完成；北京时间2026-11-06 17:25逐项核对GitHub Release，A492/492、B484/484，合计976项/7,680,574,905字节，missing/size/state差异均0；仅补传B一项未成功文件，未重传成功资产。远端W40计划已保留合并，原截止未重置，自动删除0、电脑归档0。
- 续接位置：autodl-d57:/root/autodl-tmp/MCP-RL；先读本计划4/5/6节、W39-review和docs/S1_W40_START_PROMPT.md。主线各续自身W39完整resume，GPU独立目录/版本/时钟；先查现有会话，已启动的实际作业优先，不重复启动。

## 1. 目标、已经完成的工作与本次阅读范围

长期问题是：**同一世界坐标地块被前足、后足及返程反复加载时，怎样利用可恢复材料的持久状态，让未来支撑需求约束当前加载、落点和重访时间？** 最终需要完整 Go2/TM02 合法往返、记忆因果收益、泛化、NX 推理和实机；近期论文方向与更高层期刊目标不变。原始科学路线见[冻结计划](https://github.com/kaiwen123-yang/MCP-RL/blob/4d0af25c7bcea620c7e8c9a4f5b103c4956a3fda/docs/TERRAIN_MEMORY_PLAN.md)。

当前链路为冻结因果 PANN 观察器 → PPO 的2Hz标量速度cap → 全身MuJoCo iLQG → 时变局部反馈 → PD → Go2/真实TM02。冻结的是PANN参数，地块历史仍在更新；PPO训练Actor/Critic。MPC候选尚未传播完整PANN/TM02历史，cap也没有独立的落脚、支撑分配或重访时序权限。

| 层次 | 已有结果 | 当前边界 |
|---|---|---|
| R1 | 固定2.4Hz与adaptive控制对照、全部正负结果 | 自适应全面优势未成立 |
| R2 | 三臂60模型、600工况/700历史评价；B ideal供观察器 | noisy恢复/重访仍有负结果；辨识精度不等于控制收益 |
| S0/早期S1 | 世界地块持久历史、相容映射与局部决策 | 尚不能代替自由机器人通行 |
| R8刚地 | S3九任务9/9、独立连续300s三任务3/3 | 不转移为当前材料/r29资格 |
| 当前完整联合工程 | 独立谱系采样、PPO更新、冻结末次策略评价、紧凑自动交付均实际运转 | 稳定合法材料通行仍未建立 |
| 论文核心 | 对象与科学假设明确 | 跨腿/返程记忆改变动作并产生可重复收益尚待验证 |

本次逐文件读取e51快照的5303个tracked文件，共544,772,170 bytes；4969个可UTF-8解码，334个二进制/非UTF-8。**字节扫描不是声称语义理解了每个二进制字节。** 数值审计完整解析1000份W38/W39文件、66,008,495 bytes；覆盖两轮全部训练回合、四条E30、学习小表、配置及交付清单。人工/并行审查重点是实际r26/r27/r28补丁、直接执行/学习/交付链和项目历史。没有把尚在服务器或Release队列中的原NPZ当作已读。

本次证据：

- [统计、逐回合表及数值审计](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-review/results-audit.json)，[复算脚本](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-review/analyze_results.py)。
- [控制与首发时序审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-review/code-audit.md)。
- [文献方法/实验/局限记录](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-review/literature.json)，[逐文件阅读边界](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-review/read-coverage.json)。

## 2. W39到底推进到了哪里

### 2.1 进入材料的机会增加，失败转移，通行仍为零

| 指标 | W38 / r27 | W39 / r28 |
|---|---:|---:|
| 两流独特新增transitions | 2048 | 2048 |
| 实际训练回合 | 70 | 62（A33/B29） |
| 自然物理/材料失败 | 68 | 59（A32/B27） |
| body_fall | 27 | 41（A21/B20） |
| 躯干geom13非法接触 | 40 | 4（A0/B4） |
| material_domain_exit | 1 | 14（A11/B3） |
| 预算达到截断 | 2 | 2 |
| 60s外部截断 | 0 | 1 |
| 实际材料加载回合 | 53 | 58 |
| 机身进入条带 | 15 | 30 |
| 进入且内部足加载 | 8 | 26 |
| 自然回合存活中位数 | 14.5119s | 15.0140s |
| 自然回合平均最远x | .736514m | .808886m |
| 最大x | .975877m | 1.135616m |
| 合法去程/返程/往返、到远端/开始返回 | 全0 | 全0 |

r28没有“完全没用”：部分可行性选择生效，躯干接触终止明显减少。但它没有使机器人在原任务上稳定下来。跨轮同时存在控制版本、策略继续训练和随机轨迹变化，以上差异不是隔离的因果消融。

W39有58个自然失败已加载材料；首次load到终止中位1.3808s，9次≤0.5s，A ep260约1.2ms即材料域退出。14次材料域退出的最大主应变约.020091–.024801，越过当前2%有效域；不能将其直接称为真实材料破裂或Newton数值发散。59次自然失败全部12轴都曾达到原扭矩上限，但峰值不提供持续时间和首因。

B ep272的60s `external_session_truncation` 是重要反例：maxx=.641864m、未加载材料、未入条带、未到远端；reward约−.909438。它说明存活但停滞可以占据大量样本，不能按“60秒没倒”宣布S1通过。A ep284和B ep280因1024样本目标达到分别在7.5/6s截断，也不算自然失败。

### 2.2 学习已经实际发生，但没有连续稳定改善证据

每支从7569/29新增1024/4到8593/33＝8448已训练＋145历史excluded，tail0。两条策略独立；实验树唯一采样为6545＋4×1024＝10641，其中唯一已训练10496，**不是一条10641样本或两支更新数相加的模型**。

剔除全部截断后，A前16→后16自然回合的存活中位15.036→14.752s、平均maxx .813946→.792920m；B前13→后13（略去中间一个自然回合）为16.7052→15.014s、.807577→.824474m。没有一致单向改善。

2048条真实transition逐行复算reward，最大误差1.776e−15；物理终止bootstrap全0，外部截断保留next_value；每支四个行为版本各256条。保存的gamma=.99、lambda=.95下，8次更新的GAE均值纯算术复算与float32日志最大差1.128e−6，行为log-prob自检最大误差9.537e−7。未发现当前证据内的奖励算术、跨回合bootstrap或行为版本混用错误；未重新运行网络或假称读过未上传完整obs。

总reward从−701.918变为−611.547，其中W39物理失败惩罚−590。固定样本预算下少9次失败、一个长停滞回合就会显著改变总和；不能把这个总reward改善解释为学会通行。本轮先保持W38奖励不变，避免同时改控制和学习目标。

当前动作使用的PANN非零行为A30/B56，共86/2048＝4.1992%（W38为2.7832%）；返程暴露仍0。58个加载回合中27个回合的全部2Hz当前PANN特征都为0。末次E30已有同状态清零记录：A21点动作差全0；B36点仅1点非零，最大cap差1.061335e−6m/s。这只说明当前评价中贡献极弱，不是完整重训因果对照，也不证明材料记忆没有科学价值。

### 2.3 末次策略资格与吞吐

| 本分支update33固定策略 | A | B |
|---|---:|---:|
| E30物理终止 | body_fall | body_fall |
| 仿真时间 | 10.476s | 17.592s |
| 最大x | .663673m | .755814m |
| 机身进入x≥.8 | 否 | 否 |
| E60 | blocked_by_predecessor | blocked_by_predecessor |

末次评价使用自身checkpoint008593、策略均值和冻结归一化，没有PPO更新。训练最好前缀、旧刚地结果、60s停滞均不替代此资格。原30→60→300→600阶梯保留，当前仍不安排300/600。

W39共享collection墙钟14.4560h，2048独特新样本＝141.671条/h；W38为14.0583h、145.679条/h，约下降2.75%。两轮都用原2×8；不是2×12/14的新实测。

W39全部训练回合wall之和103051.831s：普通planning占87.4371%、启动planning2.0593%，合计89.4964%；50453次普通规划平均1.785931s。physics占6.1988%，feedback .5298%，状态/RPC .9219%，其余2.8532%。material计时已经包含在physics中。这里是同步离线仿真，等待规划时物理不推进；慢会损害研究吞吐，不能直接当作这些跌倒的实时延迟根因，也没有实现墙钟50Hz MPC。

## 3. 为什么仍不稳定：本次能确认到哪一层

### 3.1 躯干几何可行只是整个任务的一部分

r28在有限候选中按原trunk违反量v、原代价J排序，保留无可行候选时的restoration。已上传E30窗口A51/B70个去重计划的accepted违反量均为0；A有两次nominal不可行→accepted可行，B窗口的十候选始终都trunk可行。这证明选择机制真实生效，也证明零trunk违反量并不足以排除后来的body_fall。

一个明确的**日志语义错误**是：r28只按accepted的v是否>0设置`acceptance_mode`，把nominal不可行→accepted可行也写成`feasible_cost`，实际正则化用的却是可行性分支、`cost_ratio_valid=false`。下一版记录前后状态并区分`restored_to_feasible`；历史原值保留。这不是已经证明的跌倒根因。

### 3.2 首发异常早于真实材料受载

| E30事件（Stand→outbound之后，原仿真时钟） | A | B |
|---|---:|---:|
| 首次raw-PD饱和 | 9.224s | 14.734s |
| 首次大raw Kδx（描述阈值>.5rad） | 9.224s，1.168303rad | 15.044s，1.295255rad |
| 首次目标角裁剪 | 9.244s，.444501rad | 15.044s，.422234rad |
| 首次真实材料load | 9.850s | 15.3816s |
| 首次非足端承载 | 10.0912s | 16.1296s |
| body_fall | 10.476s | 17.592s |

A首次大反馈时nominal十二轴不饱和，反馈新引入2轴饱和；首次目标裁剪时新引入4轴。B14.734s nominal自身已饱和，所以只限制K会漏问题；15.044s nominal不饱和而反馈新增6轴饱和。两次大反馈时base约.290m、姿态仍近水平；之后数毫秒roll角速度才大幅变化。

直接比较accepted的+20ms与同刻actual，误差并非仅仅混淆reference名称：A9.22→9.24s速度向量误差范数1.685，主要FR hip/thigh；B15.04→15.06s为2.899，主要FL hip/thigh，而base高度误差分别仅.367/.112mm。更早的关节速度分歧已可见。范数混合线速度与角速度，仅用于同构向量诊断；具体关节差见控制审查。

证据支持的工作假设是：**接触邻域关节速度预测偏差，经局部反馈放大，改变执行器饱和/接触状态，随后破坏姿态和支撑。** 还不能从短窗唯一确定哪个native contact参数、导数或接触模式是首个原因。首+2/+4ms的accepted状态/接触力及完整K当时未记录，不能通过补传创造。

### 3.3 已经排查、不能再当作已证bug的解释

实际r26的`R26RiskJacobian`不是旧`.inc`草稿：风险当前直接u导数为0有几何依据，后续u影响通过10子步P/Q链传播。qpos使用37维存储、36维切空间；足序与时钟映射已有明确记录。未发现足以归因本轮的四元数、足序、风险直接D遗漏或plan/K配对错误。

plant与预测都含kp60/kd5和原力矩限幅，不能说“预测漏了限幅”。但是现有BoxQP只约束目标角ctrlrange，固定活跃集K与末端逐轴clip并不在实际状态下重新分配受限的十二轴修正。accepted与feedback reference允许不同；将reference强改为accepted而不变动作/K，反而改变刚刚评估的策略。r28已修候选0的这类重锚风险。

预测的`MakeDifferentiable`接触设置、warmstart规则与真实plant不同；材料候选使用四个冻结因果patch，并未传播完整恢复历史。这些是明确的模型近似。因首发发生在首次材料load前，本轮先处理执行反馈及native接触邻域一致性；不能把真实TM02或PANN统一判为首因。

## 4. 下一主任务 W40 / S5-r29：先修可执行的局部反馈

### 4.1 一个主要控制机制，直接进入完整联合实验

本轮选定的优先干预是**状态相关执行器约束进入局部反馈与候选同一路径**。这是待实现、待验证的修订，不是稳定性保证。开始时用约30分钟读取D57已有原件：A9.12–9.30s、B14.63–15.10s，及A ep260首load后domain-exit；核对执行时间、plan、真实接触/PD与已存reference。不等待Release，不重跑旧物理。如果查出更早的确定接口错误，先修该错误并写明对本选择的替代；不同时叠加多种机制，也不因缺失从未记录的K无限延长分析。

原PD为 `tau=clip(60*(u−q)−5*qdot, 原tau范围)`。在当前状态定义目标角与瞬时原扭矩实现域的交集：

```text
l_i(x) = max(original_target_min_i, q_i + (tau_min_i + 5*qdot_i)/60)
h_i(x) = min(original_target_max_i, q_i + (tau_max_i + 5*qdot_i)/60)
```

仅逐轴投影到该区间可能得到与原force clip完全相同的瞬时力矩，不能拿它当新稳定性机制。实现应使用对应计划的非对角正则化`H=Q_uu`与`C=Q_ux`，以已选line-search前馈`u_ff`为中心，保留候选实际feedback scaling `s`，重解12维局部QP：

```text
min_eta  .5 eta' H eta + (s C delta_x)' eta
s.t.     l(x)−u_ff <= eta <= h(x)−u_ff
u = u_ff + eta
```

控制查询时刻必须定义唯一一致的H(t)、C(t)与无约束K(t)，不能分别插值不相容的H/C/旧K再声称等价，因为−interp(H)⁻¹interp(C)一般不等于interp(−H⁻¹C)。矩阵采用产生该策略时的regularization_used，不能用更新后的正则化拼接；候选与部署共用相同时间插值定义。若以旧插值K为无约束目标，须明确对应QP线性项与原Q_ux的区别。

数学边界详见[控制审查第7节](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-review/code-audit.md#7-下一版单一主要机制状态相关的执行器约束反馈)：无约束极限、nominal自身不可实现、状态相关活跃边界导数、QP与原K/regularization兼容必须一起处理。原目标界的导数为0；PD边界为`e_q+(5/60)e_v`，自由集合需含`H_FA B_A`交叉项。不能在部署时重新加入一次完整`Q_u`、取消已接受的α；约束投影使旧expected decrease失效时重算或明确ratio不可用。

候选生成、nominal反馈搜索、最终部署调用同一策略；原2ms目标保持、十个0.2ms PD/力矩clip、r26 A/B/P/Q路径和真实物理不变。瞬时QP可实现不等于整个2ms hold或TM02可行。区间空时显式记录不可行，按事先声明的最小违反restoration和原物理饱和处理，不冻结成成功、不放宽扭矩、不改stand来隐藏失败。

保留r28 trunk(v,J)、r27 C2门、因果packet、冻结PANN、658维观察、2Hz cap、W38奖励、原Go2/TM02 .02域/路线/终止/评价。r25旧载荷分配和MovingSupport/SupportPlacement不恢复。本轮不同时改奖励、步态、PANN训练或整个预测物理。若执行问题修后同类偏差仍在，依据新完整窗口推进native接触/支撑预测一致性，才考虑多重射击或结构化动作权限。

### 4.2 两个主分支与计数

| 新stream | 唯一父stream | 父完整resume（相对large_root） | 新目录 |
|---|---|---|---|
| w40-a / seed1 | w39-a | training/w39-a/PANN_PPO_S5-seed1/resume-state-ep00284.pt | training/w40-a/PANN_PPO_S5-seed1 |
| w40-b / seed2 | w39-b | training/w39-b/PANN_PPO_S5-seed2/resume-state-ep00280.pt | training/w40-b/PANN_PPO_S5-seed2 |

恢复各自Actor/Critic、optimizer、RNG、归一化、8593/33、tail0与excluded145；保留祖先fork_identity并新增phase_lineage。不得回退共同ep216、重seed、合并分支、覆盖W39目录或把checkpoint权重文件当完整resume。

每流新增1024/4，目标9617/37＝9472已训练＋145excluded，tail0；首新batch8594–8849形成update34。两流都达标时再新增2048，样本树唯一计数12689，仍不是单一模型。自然编号A从285、B从281继续，保留stream/job命名。

当前R12活动契约仍记录已执行W39；[W40计划JSON](../experiments/terrain_memory/s1-go2-w40-plan.json)是待实现计划，**不能仅改名字直接启动旧r28**。Codex需完成r29正常源码/构建，以及parent_map、training directory、r29日志开关、训练/评价/finisher消费者、W40 clock、实际身份和紧凑交付接线。只在实现后把主R12激活为W40；历史job冻结契约不改。

两主线默认各12 MPC workers；各14已授权。GPU第三任务并行期间优先2×12。现有workers就是C++求解器内部工作池；不只是外层Python进程。Torch1/BLAS1、原JAX共享与精度保持。25vCPU还承载控制/RPC/材料/I/O，不能把2×14再加CPU密集GPU参考任务当作无代价并行。

### 4.3 预算、评价和实际推进标准

交互≤4h；完成必要实现、正常构建、源码配置推送后，直接启动两条完整PANN→PPO→MPC→Go2/TM02联合续训，看到真实样本/日志推进即可交接，不等待整轮结束。无主线纯MPC、E30准入、smoke/mock、恢复演习、无关测试套件或参数网格。

两主线首次实际W40启动建立新共享28h绝对训练时钟；第二流和普通resume不重置。各流训练结束后自身≤4h，只对本phase本分支实际完成的新update（>33）固定策略E30，原同case合格才E60；零新update不拿父权重替代。原known_r25停止谓词保留，普通物理失败保存后自然续采，不新增失败次数上限。紧凑交付≤6h兼容上限，完成即结束；不自动加时或串接r30。

进展同时报告：原stage_pass与合法去程/返程/往返；材料加载后存活、进入且正足载、材料域退出；nominal自身饱和与反馈新增饱和的区别；accepted→actual关节速度误差；QP空交集/restoration；候选trunk恢复。仅K变小、loss下降、更新更多、站住60s或GPU更快，都不等于稳定通行。

## 5. 第三任务：本轮实际做 GPU MPC 适配与测试

### 5.1 授权、隔离和资源

用户2026-11-06明确允许在原双并行之外**增加一个专门做GPU MPC测试的并行任务**。此授权覆盖旧文档的“两计算槽/不做GPU benchmark”限制；本轮允许必要数值一致性、性能和接入测试。第三任务不是第三条合并PPO训练分支，也不占用主线正式评价资格。

任务名`w40-gpu`，独立tmux、输出目录、版本、依赖记录与GPU计划身份。默认1–2个CPU host线程，必要编译最多2个job；CPU参考计算串行/有界，不再开一组12/14线程。主线2×12与第三任务允许适度超配；用实际训练吞吐和普通资源快照观察争用。显存以正常启动时实测空闲量为依据分配，保持主线PANN/PPO余量；不把“填满96GB”设为成功标准。

优先复用可兼容依赖；必要GPU包可放独立环境并锁版本，不升级或替换两主线已经使用的MuJoCo3.2.3 ABI/环境。第三任务与主线并行做代码、适配及GPU计算，Git/index短写串行。第三任务开始时设独立≤28h绝对上限并在测试完成时提前结束；普通resume不延长，不自动启动新训练campaign。

### 5.2 先搬真正耗时的批量部分

W39规划占89.4964%回合墙钟，支持现在投入GPU工程。仍需把已有内部六项计时导出，区分nominal、model derivatives、cost derivatives、backward、rollouts、policy update；报告单位、RPC/传输、JIT及整步时间。不能在没测内部构成时认定Riccati一定是首要瓶颈。

优先原12维PD目标问题的批量动力学/有限差分与候选rollout；参考MJX-JAX/MPX的JAX批处理与时间并行。现有.35s horizon、2ms节点含176状态节点/175转移；36维切空间＋12动作的前向差分每个转移最多49组含基准状态，按节点×扰动方向批量，比只有10条候选更有并行空间。保留r26十子步前缀传播与峰值时刻定义；terminal残差单独处理。buffers留GPU，整批进出，避免每个0.2ms子步CPU/GPU往返。

当前首选是保留FP64与既有差分语义的MJX-JAX原型；若已安装版本不兼容，写清接口并在隔离环境固定可用版本。MJWarp可用于独立rollout吞吐候选，但当前官方说明其主要为FP32、无自动微分、重吞吐而非单例延迟；不能把原1e−6 FD直接搬过去就称数值等价。不要为速度缩小接触/模型或降低精度后宣称是同一个后端。只有明确瓶颈需要且原型证据支持，才在后续独立身份采用MPX式multiple-shooting SQP/parallel scan；本轮不同时展开多个GPU求解器移植。

### 5.3 用真实已存事件完成有界测试

从实际W39保存状态固定选择：刚地正常、A首大反馈/裁剪、B nominal饱和与首大反馈、首次材料加载、首材料域退出。GPU端的预测只使用当时因果可得的patch/packet；真实TM02隐藏状态仅可用于离线plant真值对照，不可进入控制器。

GPU必须逐项匹配：Go2模型与原碰撞几何、37存储/36切空间、四元数和足序、12个目标而非torque/GRF、原kp/kd/目标/力矩范围、2ms保持内10次PD更新、冻结四patch接触代理/禁用对应native foot contact、原C2与风险峰值/导数、因果参考packet、trunk(v,J)/restoration、r29约束策略（接r29时）。Python/C++自定义callback不会因device=cuda自动执行在GPU；必须显式实现并测对应语义。

第三任务按依赖推进：同状态一步及十子步预测 → A/B及风险C/D → 固定候选整段输出/排序 → 完整一次planner调用的端到端时间。小规模固定重复用于测真实耗时，包含编译/首次调用与稳态耗时的区别、同步、H2D/D2H、内存及p50/p95；不开展无关模型、worker或超参数大网格。输出原始差异，容差按变量单位、原FD步长/模型精度预先写明，不用放宽物理通过阈值掩盖不一致。

若首选后端及同问题求解器已经在这轮实现并通过上述针对性对齐，可在第三任务用**同一冻结W39策略、原CPU真实plant**做独立标记的短闭环集成测试，最长原E30时长；训练权重与主线资格不变。其目的是检查GPU求解器输出接入，不能自动升级E60/300/600，不在正在运行的A/B中热替换。若只完成动力学或导数batch，就准确交付该层的代码和结果，不冒称完整GPU MPC或生成伪造的E30成绩。

必要的受限测试是本轮授权工作，不再被旧“一律无benchmark”条款阻断；同时也不让A/B等待全部GPU移植完成。完成后给出实际结论：兼容且有益、兼容但开销大、具体模型/精度/接口不支持，或尚完成到哪个层次。新推送的GPU计划不等于GPU已部署。

## 6. 记录与交付：小结果先走，诊断后台，用户自己备份大文件

先推代码/配置/全部逐回合JSON/更新与末次评价/结论，再继续研究；已请求诊断由现有prepare/run/queue后台上传。不要为了补传旧W39再跑旧finisher或重跑物理。最新上传脚本允许每队列1–5个上传worker，它是网络传输并发，与本计划“两主线＋一GPU计算任务”不是同一个计数。7150a6b已记录当前总并发5文件(A3/B2)、服务器独立Clash以及短窗吞吐；保留这次实际配置和原deadline，不将每队列可选上限误当现有并发。

每个回合完整原feedback-actions、planner-records（按实际记录范围）、reference-packets、trajectory及prediction、必要模型/配置/材料事件、该phase全部2Hz记录/新rollout原NPZ仍属于用户已请求证据。E30及每流各失败族首例的physical/material数组优先；额外关键case可显式追加。不因raw/NPZ/大于2MB自动排除，不用摘要替换已请求原件。

**补齐新版本真正需要的accepted/reference/K记录，同时控制体积。** 在policy buffer复用前复制该次完整accepted状态/动作/时间/代价与实际部署reference/nominal/K/scaling；r29另含H/C、活跃集/边界、QP状态与原PD量。一个plan只存一次冻结H/C、参考K、regularization_used与插值定义，不在500Hz动作行重复完整矩阵。在线QP的有效Jacobian随实际状态/活跃集改变，不能把参考K当成全部实际反馈；每次500Hz动作保存δx、s、plan及插值节点/权重、即时上下界、QP解/活跃集/状态、原始与最终动作、PD请求和边界导数所需身份，以便重建有效Jacobian。采用无损二进制/压缩；关键事件前.10s/后.20s、末端.10s和restoration窗口用内存环缓存保留完整horizon，重叠窗合并；普通时段保留全回合原标量/动作记录。manifest明确“哪些plan有完整horizon”，不能写成整回合全部plan都有完整K。此选择符合只上传有分析价值部分的要求，避免在2ms节点×50Hz重规划下把完整K重复写成几十GB JSON。事件触发至少包含nominal首次原PD饱和、反馈新增饱和、max|raw Kδx|>.5rad、首次目标裁剪、首次材料load、domain exit、非足承载、QP空交集/失败/restoration、trunk restoration与终止；只是记录触发，不改变物理终止，也不再受旧x区间日志门控限制。前窗最早动作所引用plan的origin即使早于窗口下界，也保留完整快照。开始/终止附近如实记录覆盖，终止时截断后窗，不为凑数据继续物理。不得先把既有原件删掉来腾空间。

旧r27/r28完整accepted/K从未记录，无法补传；prediction是反馈参考，不是accepted。尚未上传、文件缺失、未记录、合法空文件分别标记。2ms节点完整也不等于0.2ms接触子步全覆盖；关键窗的10子步PD/接触/材料记录保留其真实覆盖。

大checkpoint、完整resume和全长视频默认留D57，用户定期导出存储盘备份后自行删除。**Codex不在用户电脑另做归档；不自动删除服务器研究输出或已上传副本。** 紧凑push不等于大数据备份完成；上传进程exit0不等于manifest每项完成。Git普通重试/整合远端更新，不强推，不覆盖同时上传的状态。

## 7. 新文献如何帮助这次决策

本次核心8项：5篇新加入项目的控制论文、TM43补读、MPX明确复用、MPCGPU补充2026作者实现。不是把所有项目都称作2026新论文。具体方法、实验与局限见[完整阅读记录](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-review/literature.json)。

| 来源 | 本次有用的机制 | 迁移边界 |
|---|---|---|
| [Adaptive Non-linear Centroidal MPC with Stability Guarantees](https://iit-dlslab.github.io/papers/elobaid25ral.pdf)，RA-L2025 | 同时约束平移稳定、角动量内部动态、总前馈＋反馈力 | 质心/GRF控制；递归可行等假设，关节执行性仍是限制，不给我们全身PD/TM02直接稳定保证 |
| [Modular Residual Learning](https://arxiv.org/html/2507.18138v1)，RA-L2025 | 动力学估计与落脚/相位修正分别学习 | Go1实机；更多动作权限，不等同2Hz cap，也没有本项目恢复记忆 |
| [Dual-MPC Footstep Planning](https://arxiv.org/html/2511.07921v1)，2025预印本，实验室列2026期刊 | 落脚与GRF共同调节角动量 | 简化模型、近似量冻结；部分草地指标快基线更好，不支持盲加第二优化器 |
| [Contact-Implicit MPC / CI-FDDP](https://arxiv.org/html/2312.08961v2)，IJRR2025 | 硬接触rollout与松弛导数分开，多重射击减少warm-start接触连锁偏差 | HOUND实机，力矩规划；错误高度仍失败、动力学defect需报告，不是Go2材料即插即用 |
| [Feedback-MPPI](https://discovery.ucl.ac.uk/id/eprint/10218402/1/RA_L_FeedbackMPPI_Final.pdf)，RA-L2026 | 明确指出局部反馈不跟踪active-set跳变、常值约束指示不进入增益 | 四足为Aliengo仿真，实机是四旋翼；借约束/反馈一致性，不建议本轮换MPPI |
| [Adaptive Legged Locomotion via Online Learning for MPC](https://arxiv.org/html/2510.15626v1)，2025预印本，TM43补读 | 在线残差适应动力学偏差；其突发接触失败案例提示我们应单独诊断接触模式错误 | 仿真证据；突发落脚/意外接触仍失败，regret不等于不跌倒 |
| [MPX](https://arxiv.org/html/2506.07823v3)，RA-L2026，复核已有来源 | multiple-shooting SQP、parallel scan与JAX批处理 | 论文WB控制24维含力/力矩，单次迭代计时；短horizon可能慢于CPU |
| [MPCGPU](https://a2r-lab.org/MPCGPU/)，ICRA2024/作者2026-10实现 | block-tridiagonal PCG与并行线性求解 | 固定基座iiwa、更新实验存在deadline overrun/未开关节限制，内核微秒不是完整Go2 MPC速度 |

另筛选了2026 ABC-DDP、Meccanica可变形地形接触感知、CUDA MPC；仅取得摘要/公告的部分明确标注，不用其headline速度或理论替代全文证据。主线动作与模型要由我们真实事件决定，不能为了追新同时堆八种方法。

长期进入下一科学阶段的顺序仍是：可稳定合法通过材料 → 因果地块状态进入候选预测 → 赋予可解释的落脚/加载/重访时序权限 → 跨前后足与返程的记忆对照及长时终端价值 → 泛化/NX/实机。GPU工程从本轮并行推进，为这些实验提供吞吐，不替代其能力验证。

## 8. Codex启动入口

使用 [S1_W40_START_PROMPT.md](S1_W40_START_PROMPT.md)；机器可读计划为 [s1-go2-w40-plan.json](../experiments/terrain_memory/s1-go2-w40-plan.json)。先核对实际作业再按本轮授权推进：W39历史已完成，W40两主线与GPU第三任务均在实现/构建/真实启动后才更新为running。本次仅完成仓库审查和计划发布。
