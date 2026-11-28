# W59 最终模型、控制与资源审查，以及 W61 首批算法任务

审查日期：2026-11-28。固定仓库快照：c8378beae4be371eb4f104782b75e5d295b6f59a。本文是只读静态审查，没有运行新的训练、MuJoCo rollout、论文复现或实机实验。W59 视频与总体步态由另一审查线处理；本文不代替视频结论。机器可读指标、完整来源路径与论文阅读范围见同目录 evidence_model_control.json。

## 1. 结论与证据边界

**W59 应作为固定的实验底座与负结果基线保留，W61 可以立即开始 A/C/D 的新代码开发。** 不需要继续给 W59 叠加架构变更，也不能把 W59 描述成已经完成可微世界模型、可靠模型规划与模型梯度联合学习。现有闭环已经能进行相当长的受约束运动；核心缺口是材料机制任务的有效作用、可用于决策的预测模型、跨状态可靠的控制与学习收益，及墙钟效率/实机实时性的差距。

最终 outcome 状态是 earned_chain_terminal，结束于 2026-11-28 07:25:58 +08:00；learning_scope=partial、matched_four_arms=false。index 登记 400 次物理尝试、active=[]。终止原因中 264 次 stage_horizon_reached 是分支/窗口到达预设仿真长度，不能统称失败；129 次 stage_passed，5 次 execution_error，2 次 storage_enospc。最后三类必须分开解释。来源：[outcome](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/outcome.json)、[index](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/index.json)。

这里有三个特别容易误读、会直接影响下一轮设计的事实：

1. 六个执行世界模型均已训练至 50,000 更新，但已发表的长时域 rollout 表以及动作准入采用 10,000 更新模型。不能把 10k 的结构模型发散写成“最终 50k 结构模型已经证明发散”。
2. 六路并发的总吞吐实际比四路增加约 65% 和 57%，不是只有 1–2%；1–2% 是原有四路自身的吞吐变化。
3. WBC 已有完整的受约束浮动基逆动力学、滚动接触和执行器保持边界约束。下一步如果只是“加入 M、J、浮动基动力学”，实际上没有新增算法。

## 2. 材料模块：M3 做到了什么

M3 在两个 A 录制的 35+32=67 个事件行上，仅拟合三个系数：

| 参数 | 最终值 | 实际作用 |
|---|---:|---|
| release_prior_blend | 0.0899047771366178 | 释放状态在 M1 与 M0 的估计之间混合 |
| positive_recovery_rate_s_inv | 0.99735344928719 | 释放后指数恢复率 |
| contact_gain | 1.054129997868099 | 被选择的跨足重载接触修正 |

模型只对已经观测、释放、且由另一只脚查询的世界单元生效；一次 footprint 查询选择一个释放来源单元，其余受载/非跨足路径维持 M1。拟合目标是脚底合力，真实单元应变与真实单元力仅用于诊断，不是可用观测。它是有用的局部恢复候选，但不是材料族自适应、在线材料参数辨识或 PANN 本构结构改造。

两个 B 开发录制的跨足整足力 MAE 分别由 0.151963737 降到 0.070017099 N，以及由 0.179222025 降到 0.103126512 N，约下降 53.9% 和 42.5%。然而短时域整体 MAE 分别由 0.119286998 升到 0.123795538 N、由 0.122770705 升到 0.124673784 N，说明增益有明确条件，不能只报最有利的跨足子集。最终 offline_candidate_accepted=true、accepted_physical=false；没有获得可以替换冻结 M1 的闭环物理资格。

来源：[recovery.json](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/model/recovery.json)、[M3 实现](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/scripts/s1_w59_model_material.py)。

**已确认的问题**是材料改造范围小、跨足修正收益不等于闭环收益，以及可观测输入不足以任意识别每个材料单元的所有参数。**尚未确认的原因**包括恢复模型阶次不足、状态误差与材料参数误差混淆、激励覆盖不足；需要用特定加载/卸载/驻留实验区分，而不是先增加大量自由参数。

## 3. 执行世界模型、P3 与 R3 的最终状态

### 3.1 训练完成度与评价版本

训练数据银行处理 241 个来源，有 76 个可用窗口 episode、634,800 个 20 ms 区间。现有训练已经使用 7 步自回归窗口（140 ms），不是纯单步 teacher forcing。三个种子的 blackbox 与 structured 均完成 50k 更新。

| 模型 | seed | 10k 归一化验证损失 | 50k 归一化验证损失 |
|---|---:|---:|---:|
| blackbox | 5901 | 0.170305 | 0.129154 |
| blackbox | 5902 | 0.173674 | 0.129883 |
| blackbox | 5903 | 0.173852 | 0.127717 |
| structured | 5901 | 1.154949 | 0.127685 |
| structured | 5902 | 1.290814 | 0.134433 |
| structured | 5903 | 1.022526 | 0.128062 |

50k 的结构模型短窗口验证损失已接近 blackbox；这值得重新做版本一致的评估。归一化损失本身不等于力误差、事件精度、动作排序或稳定性。

已经存在的 10k 长 rollout：每个模型 547 个 origins。结构模型 0.35 s 的平均 Fz MAE 达到约 14,946 / 51,513 / 958 N，0.64 s 达到约 1.34e11 / 8.19e11 / 5.77e9 N；2 s 有限输出仅 215 / 365 / 263 个 origins。blackbox 三种子在这些时域仍全有限，2 s 平均 Fz MAE 约 25.97 / 35.40 / 25.46 N。这个对照证明“加物理结构”在当时的实现与训练节点上没有自动获得好预测；**它没有评价 50k 的长 rollout**。

来源：[最终训练完成表](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/model/fit-completion.json)、[模型汇总](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/outcome.json)、[模型 recipe](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/model/recipe.json)。

### 3.2 动作已具备物理影响，但模型尚不能可靠选择动作

72 个注册动作方向全部能观测，60 个超过重复基线并未触发失败。动作空间确实有作用，不能把后续失败全部归因于“动作完全被门控抹掉”。

冻结模型只支持 1 个预测子步，即 20 ms。ranking_admitted=false、r3_gradient_admitted=false、r3_gradient_test_substeps=0。后者表示没有合格的梯度时域被选中，不表示测得“梯度准确率为 0”。

开发时域检查进一步说明方向预测问题：

| 时域 | 可分辨方向 | 正确/可分辨 | 全部 72 方向中的正确比例 | forward 支持 |
|---|---:|---:|---:|---|
| 20 ms | 29 | 9/29，31.0% | 12.5% | 是 |
| 100 ms | 36 | 28/36，77.8% | 38.9% | 否 |
| 340 ms | 60 | 26/60，43.3% | 36.1% | 否 |

100 ms 的条件比例看似高，但不满足前向支持，不能单独引用它准入模型梯度。密封动作评估 72 方向中仍有 46 未分辨。最终只是允许真实 PPO；Diff-full / Diff-private / Diff-fixedWM 均未运行，imagined transitions 均为 0。

现有 P3 在三个宏动作上执行两轮各 64 候选的 Sobol/高斯精英搜索，不是完整落足、接触拓扑和力轨迹的 GPU MPC。扩大候选数不能补救预测排序失真。

来源：[冻结 recipe](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/action/model-recipe-freeze.json)、[R3 时域检查](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/action/r3-development-horizons.json)、[规划代码](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/scripts/s1_w59_planning.py)。

### 3.3 静态源码提供的原因假设

ExecutionMember 的结构分支在 20 ms 步长内用当前刚体算子和预测的执行器/接触力推进速度，再叠加网络残差。刚体算子 M^-1[-h,S^T,J^T] 从最初观察状态在 CPU/MuJoCo 计算，转为 detached 张量；训练窗口和候选滚动没有逐步更新该算子。关节位置与速度、足部位置与速度、部分力与控制内部状态分别预测，可能逐步失去几何与执行一致性。

应把下列项设计成可隔离消融，而非已证实原因：

- 初始冻结的 M/J/偏置无法反映预测构型变化。
- 显式 20 ms 近似与高增益执行器耦合可能放大误差；必须先测离散谱/局部响应，不能只凭经验断言。
- 独立的足位预测与 q 的 FK 可能不一致；独立接触力/模式预测可能与动力学状态不一致。
- 140 ms 训练窗口与 0.64–2 s 使用范围不同，状态分布逐步偏离。
- 接触/所有权切换的离散操作使可微路径只在局部模式内有效，不能直接当跨模式全局梯度。

来源：[模型转移](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/scripts/s1_w59_model.py)、[刚体算子](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/scripts/s1_w59_model_rigid.py)、[训练实现](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/scripts/s1_w59_model_train.py)。

## 4. 当前 MPC/WBC 实际已有的内容

W59 继承 W57→r60→r58→r55/r53→r52。底座 MPC 是 12 维单刚体/质心近似状态与四足力的有限时域 QP，使用固定/事件预测接触，状态目标、力正则、力差分、摩擦内方锥及法向力界；r55 对终端站立补了 COM 与支撑力的力矩线性化。W59 改动主要是 due_offset、transfer_ramp、pair_load_preference 三个宏参数，以及力正则参考与 WBC 分配目标的偏置。

WBC 优化变量为 18 维广义加速度和 12 维接触力；包含：

- 浮动基六行硬动力学 M qddot − J^T f + h = 0。
- COM、姿态、摆腿、支撑、关节姿态及力跟踪加权任务。
- Jdot v、真实滚动接触点及滚动接触 Jacobian。
- 原始力矩、位置输入范围、2 ms hold 内十个 0.2 ms 边界的局部预测力矩约束、关节可达边界。
- 历史材料模型产生软法向位置参考；干摩擦模型按参考方向进入。
- 输出总力矩再通过 u=q+(tau+5v)/60 映射到原有位置执行器；并非另加一个无约束力矩通道。

因此下一步的 WBC 贡献应改变任务惯量、误差动力学、模式内的扰动内模、材料接触能量交换或实时稀疏求解结构；不能把已有的完整 M/h/J 计算重新命名为创新。上述 hold 安全近似也不是全局非线性安全保证。

源码：[r52 完整控制器](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/scripts/s1_w54_f_r52_backend.py)、[r55 终端线性化](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/scripts/s1_w55_f_r55_backend.py)、[r58 材料法向执行参考](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/scripts/s1_w56_f_r58_backend.py)、[W59 宏动作](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/scripts/s1_w59_f_backend.py)。

## 5. 资源与实时性：应怎样加速

### 5.1 瓶颈在物理采样链

记录的一次 E60：60 s 物理时间需要 1422.8057 s 墙钟，observer 553.6838 s、physical loop 480.1124 s、feedback 357.8219 s。physical loop 内部材料耗时 363.0387 s 是嵌套项，不能与前三项再次相加。最近三个 256 样本 PPO 优化批次仅 0.0631–0.1027 s，而批次之间为 1416.56–1431.99 s。就这些记录而言，优化器用时占比不到 0.01%，把 PPO 网络再放大并不会使真实采样更快。

来源：[gpu-time-20261127.json](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/dispatch/gpu-time-20261127.json)。

### 5.2 六路并发有实测收益

同四个 incumbent PID、相同 240 s 墙钟窗口插值实际物理时钟：

| 配置 | 总仿真秒/墙钟秒 | 相对四路总吞吐 |
|---|---:|---:|
| 原四路 | 0.085634926 | — |
| 六路，第一窗 | 0.141323277 | +65.029951% |
| 六路，后一窗 | 0.134743940 | +57.346944% |

原四路自身变化仅 +1.899505% 和 +1.137098%。这恰好说明新增两路带来了有效总产出，且原四路没有明显变慢。不过接触阶段和其他项目工作负载未完全控制，不能断言任意任务混合都能线性扩展。对应窗口其他项目约使用 10 个 CPU 核，最小可用 RAM 约 3.59 GiB，GPU 利用率是整卡共享指标。96 GB 显存并不等于 96 GB 空闲显存，更不等于 RAM 充足。

来源：[mechanism-overlap.json](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/dispatch/mechanism-overlap.json)。

### 5.3 GPU 混合观察器已经值得移入 W61，但未证明完整闭环加速

同输入的 A/B 完整离线 observer 回放，各 30,000 个发布窗口。合计 CPU 1083.327315 s、hybrid 666.204121 s，1.626119 倍。2,880 个 actor 查询高度、查询时间/footprint，以及逾 1.04 亿个 material cell tick 状态比较没有改变；最大力差约 4.07e-9 N。

它 new_physics_steps=0、new_training_events=0，生产 5902 observer 仍为 CPU。完整 E60 的 1.18378/1.15879 倍是投影，不是测得的闭环结果。CPU median 19.94 ms→hybrid 6.82 ms，但 p95 37.54→35.93 ms；尾延迟仍然高，无法据此宣称 500 Hz 实时。

小 batch 明显不适合 GPU：≤32 个活跃列时 CPU/GPU 速度比仅约 0.17–0.19，GPU 反而慢 5–6 倍；129+ 列约 1.85–1.94 倍加速。应在 W61 建立按活跃规模和设备队列选择 CPU/GPU 的实现，并以端到端闭环测量决定。不能把这些离线类别直接当成所有机器通用的固定阈值。

CPU/GPU 等价也不表示 observer 的物理事件标签等同于 plant：例如 A 的 cross-foot revisit 52/52 中精确 tick 对齐 47、前次 load lineage 对齐 37；B 为 60/60 中精确 tick 45、lineage 35。可在新算法中改善观测建模，不能用GPU等价掩盖状态估计误差。

来源：[M1 hybrid overall](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/dispatch/m1-hybrid-20261127/overall.json)。

## 6. 核心论文：读到的方法、可借鉴处和不可直接搬用处

下列正文阅读覆盖方法、实验和限制，不只题目摘要；没有重新跑论文代码。分区信息未重新取得官方 2025 中科院分区表，因此不臆造一区/二区 Top 标签。

### 6.1 HyperNODE，npj Computational Materials 2026

[官方论文](https://www.nature.com/articles/s41524-026-02027-8)，DOI 10.1038/s41524-026-02027-8。读 Methods 方程 9–19、材料场插值、分阶段初始化、强形式平衡/牵引损失、合成与打印试件实验和 Discussion。其材料坐标经 hypernetwork 产生局部 NODE 本构权重，以结构保证能量相关条件；利用全场位移及边界力反演异质超弹性响应。关键区别是 NODE 的积分深度不等于实际恢复时间，也没有直接解决我们脚底稀疏合力观测下的黏弹/历史状态辨识。

可借鉴条件化的本构族与结构约束；我们需新增的研究问题是用有限可用观测分离材料身份与加载记忆。官方仓库 [node_inv](https://github.com/tajtac/node_inv/blob/85d551a4dd8dc2777fdb18fa1aa1a8886a6ee132/utils_hyperelasticity.py) 已静态读 common/sample 权重、NODE/RK 和应力计算；未进行复现。

### 6.2 HANO，2025 作者正文

[正文](https://arxiv.org/html/2506.10352v2)。读第 3 节结构/训练，第 4 节对照，第 5.4 节历史长度与第 6 节限制。用可观测应变/应力历史窗口与应变增量预测路径依赖响应，采用逐步降低 teacher forcing、逐步增加输入噪声来训练自由滚动；RNN 对照使用匹配输入与预算。历史长度实验依赖具体材料，不能据某实验的平台值给所有材料指定“六步足够”。

对本项目有价值的是可辨识历史条件和滚动训练协议；物理能量/耗散约束仍应由我们的 PANN 结构承担。不能直接把 HANO 写成已有普适热力学保证的在线材料估计器。

### 6.3 Residual MPC，TRO 2026

[正式出版信息](https://ieeexplore.ieee.org/abstract/document/11644391/media)，DOI 10.1109/TRO.2026.3721669；实际精读为[作者 2025v1 正文](https://arxiv.org/html/2510.12717v1) III–V、算法 1–2、表 I–II、硬件段落。方法是 12 节点浮动基 kinodynamic MPC，单次 SQP、25 次 ADMM，一次 KKT 分解复用回代；CusADi 生成周边符号计算，cuDSS 批量分解；RNEA 前馈＋PD 与关节空间残差进行力矩混合。不是完整关节力矩受限的全身优化，正文明确省略部分限制。

100 Hz 是控制离散频率；千环境单步墙钟约 0.5 s，完整 RL 训练比纯 RL 慢约 8 倍。硬件 OSQP 100 Hz 属真实部署，范围有限。公开 CusADi 核心静态读取没有 PyTorch backward，也不是已核实完整 Residual MPC 训练栈。应借鉴稀疏求解/前馈/训练并行，进一步研究材料与接触约束下可行残差，不能仅重复“MPC+PPO”。

### 6.4 VR-MPPI，RSS 2026

[官方 proceedings](https://roboticsproceedings.org/rss22/p108.html)，[全文](https://arxiv.org/pdf/2602.03639)。读 II–IV、算法 1、二次提议分布、重要性修正、曲率近似及实验表 II。将代价拆为二次近似和残差，以二次结构调整高斯均值/协方差，再按残差重新加权；支持精确 Hessian、Gauss–Newton、BFGS 或随机平滑。收益在有限候选下提高采样效率。

接触实验曲率构造约 1412 ms、候选求值约 706 ms，不能用它证明四足实时性。我们的直接对照需计入额外平滑样本、曲率开销和 CPU/GPU 转移时间；并保留多接触模式，避免单一局部高斯丢失拓扑多模态。

### 6.5 Natural Task-Space Decoupling，TRO 2026

[DLR 官方全文](https://elib.dlr.de/221903/1/main_final_with_CCBY.pdf)，DOI 10.1109/TRO.2025.3647777。读 II–IV、VI–VIII，包括假设、任务惯量、Coriolis 一致性、稳定/被动性条件与 Rollin Justin 实验。它保留各任务的自然惯量块，处理跨任务动力学耦合，减少对强逆动力学补偿和高反馈增益的依赖。

证明要求全驱动和局部可逆任务映射；不能直接用于欠驱动浮动基与软接触切换。我们的可研究改造是在当前接触模式的可实现子空间内设计部分任务解耦及材料法向顺应，而非照搬全驱动律。论文对无扰动实机优势、摩擦/模型误差和力观测误差也有边界说明。

可选强对照：[Sampling-based MPC Using Trust Regions](https://arxiv.org/pdf/2605.07801v2)，当前元数据为 CDC 2026 accepted。已读 KL/entropy 约束的双变量求解、密度修正权重、Gaussian 投影、确定性高斯采样/缓存、实验时延。可与 CEM/VR-MPPI 公平比较，但不能只把旧精英更新换名称，也不具有本文任务的现成实时性证明。

## 7. W61 首批任务：需要真正改变算法的 A/C/D

这些是建议设计，不是已经实现或证明的创新。六线开发可并行，首轮先冻结模块接口、比较单模块；在相同基线之上得到可解释的结果后，再做材料+MPC，以及步态+WBC 的两组联合，最后整体联合。无需等待所有模块都达到最终论文水平。

### A0：补齐版本一致的世界模型证据，和 A1 同时开始

从冻结的 50k 权重重新运行原有 20/100/350/640/1300/2000 ms rollout、事件响应与方向检查，所有 checkpoint hash、数据 split、起点、控制输入一致。补测写入 W61，不修改 W59 表或覆写旧结果。优先复用已有输入/状态；若需要物理分支，另登记新尝试。它是一项证据补齐任务，不应占据整轮研究，也不能用测试集选择模型或超参数。

### A1：条件 PANN + 可辨识历史状态适应

把材料身份 c 和当前加载历史 z 分开：c 控制正参数/本构权重，z 描述可恢复/不可恢复历史。采用带正参数、耗散/能量结构的条件本构族，并从有限的接触力、压陷、速度与时间历史进行有界在线更新。材料身份更新可以低频，状态更新保持高频。第一轮只允许少量可辨识参数或低维 latent，不对每个单元任意反演完整材料场。

训练集明确包含独立材料族和加载路径；开发与验证按材料和整条历史分离。既比较新材料，也比较已见材料的新加载速度/卸载停留/跨足重访。把旧 W59 单材料固定任务保留为回归集，不通过修改旧地形让新方法更容易产生收益。

直接对照：冻结 M1/PANN、M3 三参数修正、同观测条件的普通参数估计、条件黑箱模型、条件 PANN；oracle 材料参数只做诊断上界并独立标记，不泄漏给部署 actor。评价整段力响应、恢复时间、切线响应、自由滚动、辨识不确定性，以及固定控制器下的闭环收益。

### A2：执行世界模型改为一致的离散物理转移

第一版实质修改不是扩 MLP：将显式 stiff-PD/刚体推进改为半隐式/隐式离散，刚体算子依赖预测构型，足位由 FK(q) 产生，学习的是有物理位置的残差/力/执行器误差，不再把每个冗余状态都独立生成。材料内态和接触状态按同一因果时间推进，模式切换显式记录。

可以先对冻结构型下的线性 PD 子问题做解析稳定离散，再引入构型更新。应比较现有结构转移、仅隐式执行器、再加构型更新、再加几何一致性四个递进版本；控制训练预算与数据不变，测局部响应/长滚动与动作方向，而不只追逐归一化 loss。不能在没检查梯度经过哪些 CPU/离散操作时声称端到端可微。

### C1：有限步稀疏动力学 MPC + GPU 批量候选

先在固定接触计划和同一权限下，实现具有缩放、warm start、结构复用和残差记录的有限迭代 QP/SQP。再将 B 线提供的有限接触拓扑候选、落足/时长候选和材料场景在 GPU 上并行评价。接触序列保持分支，不用一个高斯去平均互斥拓扑。优先把每个分支的完整约束问题解好，再探索 VR-MPPI 的二次提议和 trust-region 分布更新。

最初可用已知刚体模型与冻结 M1 运行，不应等待新世界模型准入才开始开发。A 输出材料后验后，再加入接触法向响应/恢复成本/场景风险；W59 仅 20 ms 支持的执行 WM 不应被直接延长作高质量大范围规划依据。

直接对照：原 CPU QP、相同权限的标准 warm-start SQP/ADMM、MPPI/CEM、VR-MPPI 或 TRMPC、拟议拓扑批量方法。先固定接触候选库比较求解器，再固定求解器比较拓扑/材料贡献，避免把动作空间扩张误当算法收益。报告相同墙钟预算与相同候选预算两组结果，包含所有分解、曲率、转移、通信、约束残差、超期与控制失效率。

训练吞吐与单机器人部署需分别验收。GPU 千实例的每步耗时、单位有效样本吞吐，和单实例控制循环 p50/p95/p99 是不同指标。规划控制的固定有限计算预算不是给后台任务设置完成时间。

### D0：先对真实 WBC 失败的保存状态做数值定位

index 可确认唯一运行中 execution_error 是 seed 5901 的 B/E60/0023，发生在 36.774 s；其余四个 execution_error 是 0.002 s 初始化。已读取[该 attempt 原始 receipt](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/attempts/w59-r3-5901-ppo-full-fallback-b-e60-0023.json)，其 runtime commit 为 4ad3a5f28b99ad9ccc84455915fdffa9dfcb57af，traceback 明确为 ControllerNumericalFailure: wbc: maximum iterations reached。尚未进行该保存状态的精确 QP 反事实重放；这不能直接证明物理不可行、矩阵缩放问题，或增加迭代就能解决。

从失败前状态恢复相同 QP，记录矩阵缩放、活动约束、原始/对偶残差、迭代轨迹、warm-start 可行性和接触切换。分清“可行问题未在预算内收敛”“条件恶化”“本身约束不可行”。不要直接增加最大迭代数并声称新控制理论，也不要降低原约束来制造成功。数值定位可和 D1 新架构开发并行。

### D1：模式内可实现的任务惯量与混合内模 WBC

保留现有 M/h/J 和原执行器权限，在支撑模式允许的可实现子空间设计任务惯量和耦合关系。法向软接触采用有材料状态的顺应/耗散任务，切向接触继续服从摩擦/滚动边界。扰动内模至少区分周期性步态误差与非周期载荷/执行器偏差，并避免把材料误差、接触冲击和电机偏差都吸收到一个自由残差。

第一版可用相位基函数的有界内部模型估计重复误差，慢参数更新；瞬时可行投影仍执行硬动力学与执行器约束。随后比较相位内模、普通 disturbance observer/ESO 与无内模。内模输出作用在明确的任务/广义力通道，由同一可行优化器实现，避免绕过原位置输入接口。

对照：原受约束逆动力学 QP、同约束的标准扰动观测补偿/顺应控制、任务惯量改造、任务惯量+混合内模。固定步态、MPC和测量条件，分别施加载荷/摩擦/延迟/执行器偏差，再与 B 的新步态组合。测姿态/速度跟踪、接触冲击、滑移、力矩饱和、输入平滑性、QP失败和完整回路延迟。理论目标应写成模式内局部稳定/ISS/耗散及切换条件，尚无依据写全局鲁棒保证。

## 8. 并行安排与交付

- 六条研发线可立即工作；共享冻结 baseline、测量接口和变化清单，开发分支/输出目录独立。
- CPU 物理队列可在资源允许时使用已验证有收益的六槽，不把“最多2–3任务”当永久限制。同时安排材料 GPU batch、离线模型诊断和求解器批量性能工作，避免所有队列都争同一 CPU/磁盘瓶颈。
- 首批优先把 A0 和 A1、C1、D0 和 D1 同时变成能运行、能比较的实现；日后先两模块联合再全栈联合。每次模块合并保留单模块消融路径。
- 根据可用 RAM、scratch 空间、GPU队列和其他项目实际负载分配 lease；不杀其他项目，不为聊天结束或预设墙钟时刻结束后台训练。
- GitHub 作为结果与计划交付入口；成功推送后不再安排用户电脑上的重复归档。一个独立 daily progress 文件按日期追加：运行/提交/模型hash、假设、变化、结果、失败、已解决问题、未解决问题与下一步；每天不能只有一句“完成训练”。
