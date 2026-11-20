# W55 控制与原任务资格审查

控制审查基点：`8c4768eae35a84b12aa8d44dbf00da20877c894a`；原 W55 计划：`c8e0b38d8256305b9e5a14907829980c3f1b0054`。后续交付更新先到 `a15019260087b7b6936aa46b696d733a2a2db924`，再到 `0eb39b13932c18d8e52ff305a3c35477a4510ab7`，补齐 B/E300 独立 runner/result/analysis/resolved 及交付信息；没有新控制代码或新完成资格阶段。以下 B/E300 证据边界与 E600 快照已随最新交付更新。

本审查阅读控制实现、运行入口、资格检查及保存证据；对下载到审查环境的原始控制、drift、MPC 数组做了独立数值复算。**新增机器人物理、native rollout、QP 重解和训练均为 0。** 全轨迹与视频的独立审查另见本轮证据报告；以下明确区分本次独立复算、执行端既有诊断和推断。

## 1. 现在应把 F/r55 当作已建立的稳定基线

当前已经超出“连稳定行走都没有”的状态。F/r55 在两个自己的冻结 case 上，各自通过 E30、E60、E300，合计六个已结束阶段全部通过；E300 各完成 5 次合法材料往返。A、B 的 E600 在审查快照里已经运行，但尚没有完成结果。这支持**保持 r55 不变、续接原 A/B E600、随后推进完整受控对照**，不支持再以全面改写 F 步行内核作为下一轮主任务。[S1][S2]

这里的“稳定”限于已实际测试的 Go2/TM02、nominal 初态、两份 W44 冻结策略和现有物理边界。两个 PPO 父策略并不是大量随机扰动样本；阶梯的多个阶段有重叠任务前缀。不能把六次通过当成独立同分布试验，再据此报一个高置信度可靠率。真实系统实时性、扰动鲁棒性和材料记忆的因果收益仍是不同的未完成问题。

### 1.1 重建后的原 attempt 账本

| 原身份 | E30 | E60 | E300 | E600 |
|---|---|---|---|---|
| F/r53 A，seed1 | 通过 | 42.676 s 控制器中断 | 未运行 | 未运行 |
| F/r53 B，seed2 | 通过 | 41.254 s 控制器中断 | 未运行 | 未运行 |
| F/r55 A，seed1 | 通过 | 通过，1 次合法往返 | 通过，5 次合法往返 | running 快照 |
| F/r55 B，seed2 | 通过 | 通过，1 次合法往返 | 独立结果通过，5 次合法往返 | running 快照 |
| N/r54 A，seed1 | 完整 30 s，未贯穿材料 | 未获资格 | 未获资格 | 未获资格 |
| N/r56 A，seed1 | 完整 30 s，未贯穿材料 | 未获资格 | 未获资格 | 未获资格 |

因此已结束原任务是 **12 次：8 通过、2 技术中断、2 任务条件未满足**；共 **983.9300000148 s、4,919,650 个实际物理步**。E30 是 6 次、4 次通过，E60 是 4 次、2 次通过，E300 是 2 次、2 次通过。另有 2 个运行快照，不能算已通过、失败或已完成。[S1][S7]

`w55/index.json` 的 11 次、E30 5 次漏掉了已经结束的 N/r56。`current.json`、AGENTS 的部分叙述也滞后于 phase 中的新结果。N/r56 的 phase 行缺少时长等数值，但其原 result 和 `N-A-r56/E30-actual-analysis.json` 明确记录完整 30 s、150000 步、`success=false`。下一轮先修入口一致性，避免旧摘要继续驱动错误研发优先级。[S1][S7]

**最新交付已补齐 B/E300 独立结果，独立复核范围仍需区分。** A/r55 E300 的原数组已独立复算。B/r55 E300 在 `e5ff147` 已补齐独立 runner/result/analysis/resolved，资格、case key 与前序关系不变；Release 共82项资产中包含B视频和20个ZIP。主审已核B视频的SHA/size、完整解码与16个抽样帧。本次没有独立复算B的原数组，因此B在232.696 s完成第5圈、随后稳定67.304 s是已核独立结果中的执行端分析值，不再是仅phase声明，也不能写成已独立数组复算。[S2][S13]

最新北京时间2026-11-20 16:33:31的phase快照中，A/B E600的物理时间分别为153.502/113.502 s，仍为running，未获得E600完成资格。它们是当时的进度快照，不能作为之后服务器仍在运行的实时证明。[S13]

## 2. F/r53 修复是真的，但首次成功没有掩盖下一处缺陷

### 2.1 Jdotv 修复覆盖了真实执行调用链

`s1_w55_f_r53_backend.py` 继承 r52；在固定 MuJoCo 3.2.3 环境中，对平移 drift 补上 `omega_body_world × v_point_world`。构造器限制版本为 323，避免未来升级后重复补项。四脚使用各自足刚体角速度与球心世界速度；全机器人 COM 是逐连杆质心分别修正后按质量平均；rolling 在球心修正上加一次原有固定世界向下半径的角导数项；base 和 foot 的角 Jacobian drift 不加此项。[S3]

这几处区分有实际意义：把全身 COM 当作一个刚体、拿 base 角速度去修总 COM，或把球底 rolling 当成附着在足上的同一个物质点，都可能再次引入错误。当前源码没有这些混淆。r52 的 WBC 依次读取修正后的 `Jcomdotv`、`Jdotv`、`Jrolldotv`，不是只修了诊断输出。

执行端 `r53-same-state.json` 的八个 r51/r52 保存状态，使用实际 r53 helper 和随 q 移动的计算点做运动学方向差分：修后球心最大误差约 `1.96e-10 m/s²`，COM `2.78e-11`，rolling `1.97e-10`。这份既有验证不含动力学推进。[S3]

本轮又从四组 E60 原件选取的 **27,967 个成功返回控制 tick** 独立检查：实际 WBC 所用 foot/rolling drift 与修正记录误差为 0；`omega × v` 复算误差为 0；加法组合残差不超过 `4.44e-16`。r53 最后失败 tick 已计算 drift，但没有成功返回控制，因此只对时间一致的已返回控制行比较，不把失败 tick 冒充已执行控制。复算程序与数值在 `control-raw-check.py/json`。

### 2.2 r53 的 E60 失败发生在回家后的站立段

两份 r53 策略均已到远端、转向、穿过材料返回。A 在 40.178 s 完成原往返及回家稳定条件，随后进入 complete；记录的后续 hold 从这个时刻计时，又稳定 1.180 s 后漂移，42.676 s WBC 返回 maximum iterations。B 在 38.424 s 进入 terminal_stand，尚未完成原回家稳定计数，41.254 s 中断。A 的后续 hold 只有 1.180 s，不等于 A 从未完成此前原任务要求的回家稳定；两者的计时起点不同。[S4]

必须区分“最终退出原因”和“最早错误”。日志最后报 OSQP/WBC 迭代上限，不能据此先把 solver 迭代次数翻倍。保存的 MPC 已在更早的站立时刻给出方向错误的水平合力：机身尚在原点正前方，目标在原点，规划反而向前推。两个 r53/r55 同 seed 原件在 terminal_stand 之前的 qpos 逐位相同，首次差异恰在该阶段后的 2 ms 控制步。这把新旧行为差异定位到了站立分支，支持该处修复具有直接因果意义。独立轨迹复算详见本轮 `arrays-recomputed.json`。

**正式标签是技术中断，不等于此前没有物理失稳。** 证据线对全5 kHz原件的独立复算发现，r53 A/B末2 s机身最低约0.2041/0.2042 m，分别在42.5978/41.1578 s出现首次非足接触，非足合力峰值约503.9/430.1 N。原允许地面接触集合包含四腿hip子树的身体；聚合非足接触数组没有逐contact body ID，因此不能直接判成禁止的躯干触地。可确认的是“终点漂移、明显下沉、已有非足承重，最后因WBC数值错误退出”；不能把它描述成纯粹求解器报错但身体一直正常。r55相匹配时窗未出现这些非足接触。

### 2.3 r55 修的是质心力矩臂的展开位置与一阶耦合

r52/r53 的站立 MPC 使用 `reference[k,3:6]` 作为未来力矩臂中心。进入站立后，reference 很快指向终点，而真实 COM 仍与终点相差数厘米。目标被提前当成动力学里的实际位置，导致模型错误解释接触力对姿态的作用。[S5]

真实力矩是 `sum((p_i-c) × f_i)`。在 `cbar` 与合力 `Fbar` 附近，COM 一阶项为 `Fbar × (c-cbar)`，因为 `-delta_c × Fbar = Fbar × delta_c`。r55 用当前 COM 加测得速度作预测展开点，把这一项同时加入角速度与 Euler 角的 A/c；原终点仍在状态代价中，原足点、摩擦约束、权重和求解器设置保持不变。修复仅作用于 `terminal_stand/complete/far_hold/home_hold`；步行及 initial_stand 走原 r53 分支。[S5]

执行端从同一 A/r53 E60 原件保存的三个 MPC 时刻做过有限重解：

| 时刻 | 实测 COM x | r53 第一节点水平加速度 | r55 同输入修正后 | 含义 |
|---|---:|---:|---:|---|
| 38.500 s | 0.055389 m | +0.030783 m/s² | −0.280262 m/s² | 从离开目标变为向目标收敛 |
| 40.178 s | 0.060823 m | +0.122298 m/s² | −0.226180 m/s² | 修正原站立继续向前推 |
| 41.500 s | 0.080811 m | +0.293965 m/s² | −0.220874 m/s² | 原漂移反馈已加重 |

原力复现误差为 0；这份既有检查包含 9 次 CPU MPC 重解，不含新物理/WBC/native rollout。它是执行端 W55 已完成证据，本次没有重复这些求解。[S5]

本次对 r55 A/B 原 E60 共 **30 个保存的 hold MPC** 独立重建 A/B/c：展开 COM 与支撑重力基准误差为 0；矩阵最大差约 `1.3e-12`，保存预测状态的动力学等式残差不超过 `6.2e-11`。这确认记录中的实际矩阵确实含有声明的修复，而非只有配置文字改变。

### 2.4 r55 的闭环结果越过了原失稳点

| 原任务 | 完成往返时刻 | 后续连续稳定观察 | 终点位置违约 |
|---|---:|---:|---:|
| A/r55 E60 | 40.176 s | 19.824 s，至 60 s | 0 |
| B/r55 E60 | 40.428 s | 19.572 s，至 60 s | 0 |
| A/r55 E300 | 231.942 s 完成第 5 圈 | 68.058 s，至 300 s | 0 |
| B/r55 E300（独立result中的执行端分析，未独立数组复算） | 232.696 s 完成第 5 圈 | 67.304 s，至 300 s | 0 |

A/B E300正式记录分别给出最小机身高度 `0.283007 m`、最大 x 接近 3 m、5 次去程与5次返程合法贯穿、0 次侧向绕出。A 材料峰值主应变 `0.00322918`，B `0.00340561`，都无已记录 domain exit。这与“靠站着活满时长通过”不符；他们完成了规定路线。A已独立数组复算；B为独立结果/分析与视频核验。[S1][S2][S4][S13]

## 3. 原标准没有被 r55 放宽；有限精度也要如实报告

审查原计划与当前源码差异后，`s1_go2_r11_protocol.py` 未改变，SHA256 仍为 `d424d29f604cfc93f6a19ce9fb56ca66ef559383b6f9b602bf2fd78692a2e30e`。E30 要完整 30 s、到远端并合法贯穿；E60 要完整 60 s 与合法往返；E300/E600 保留 5/10 圈、270/540 s 路线期限与末端稳定要求。R12 的双向材料资格仍额外检查合法去程和返程数。[S6]

结果 `qualification.protocol_revision` 写 v2，而运行合同 formal protocol 写 v3，来源是旧全局 CONTRACT 标签；实际函数仍执行 `bidirectional_material_roundtrips_v1`。这是元数据命名不统一，不能单凭 v2 字符串断言资格退回旧规则。后续应清理新入口的显示，但不能修改已有身份合同来伪造一致性。[S6]

A/r55 各已结束阶段 case key 都是 `0fcd6e1f...3b8f05`，B 是 `8b0c61dc...c07b`；各自 checkpoint、normalizer、controller、原 nominal 初态与 virgin 地面、workers=1 保持一致。A/B checkpoint 哈希确实不同。每个 E30 从自己的 t=0 启动；r55 也没有继承 r53 资格。r55 与 r53 的 E30 数组相同有代码原因：E30 没到达被修改的终点 hold 分支，且其自己的 E30 已独立完整执行。[S1][S3][S5][S6]

F/r55 和 r53 配置的行为差别仅是上述修订；原 MPC/WBC 权重、摩擦内接方锥、PD60/5、原力矩/位置输入/关节范围、保持区间预测约束、求解容差均未改。`force_normal_upper=inf` 在 r52 就存在，不是 r55 临时删限制；真实关节力矩等硬界仍在 WBC，实际 plant 原限幅顺序也保留。[S3][S5]

独立保存数组检查中，r55 A/B 各 8501 个选取 tick 的位置输入、起始力矩、平均力矩和十子步预测力矩超界均为 0。精确 PD 映射残差最大约 `1.6e-14 Nm`。不过全选段含启动等状态，计划力的有限精度残差并非数学零：圆锥最大微残差约 `2.63e-5/4.06e-5 N`；r53 临终位置输入还出现 `1.48e-6/2.84e-6 rad` 微超界。应报告这些数值尺度，不能把 solver 的 solved 标签当作无限精度可行证书，也不应把这些微残差与数厘米的 COM 漂移混为同一主因。本次只检查已下载的控制区段，全 5 kHz 实际输入/力矩由证据报告单独复核。

## 4. N/r56 有真实稳定改善，但仍未形成能持续过材料的动作

### 4.1 r54/r56 已经不再是 W54 原来的自由体刹脚模型

当前 `w55_n/support.cc` 已用共用支撑响应的 Schur 消元和联合力矩投影；硬地支撑接触使用 rolling 几何，材料足使用冻结因果状态的法向力与切线；加上支撑反力单边性、摩擦、位置输入、力矩及保持区间约束。触地前制动不允许降低原可行基线的身体竖直加速度或支撑法向力。native 内还加入身体接触/姿态/关节/材料域和竖直制动可行检查，以及通过同排序的 support_recovery 候选。[S8]

实际执行、native、FD 使用同一执行映射和来源明确的冻结 userdata 包；材料场从独立 observer e/stress 构造，不读真 plant 隐藏 q。附加状态不是只藏在 Python 的私有变量。虽然接触响应和短期冲击仍是局部模型近似，但不能继续把 W54 的“自由体逐脚制动、没有身体可行检查”当作当前代码状态。[S8]

| 指标 | N/r54 A | N/r56 A |
|---|---:|---:|
| 原完整观察 | 30 s | 30 s |
| 最远 x | 0.788521 m | 0.894534 m |
| 终态 x | 0.779948 m | 0.887126 m |
| 最小机身 z | 0.276509 m | 0.283114 m |
| 身体进入材料西边界 | 未进入 | 10.692 s |
| 身体穿出东边界 x=2.2 m | 未完成 | 未完成 |
| 材料峰值应变 | 0.011473 | 0.006959 |
| 可行恢复候选/实际安装 | 1520 / 0 | 1521 / 1 |
| 可行推进候选/实际安装 | 无此生成器 | 1475 / 106 |

r56 更早推进、身体更稳、能小幅进入材料，改善是真实的。但最远仅进入 0.8 m 西边界后约 9.45 cm，距离贯穿还很远。稳定停在边界附近不满足任务。这两次应记为任务未完成，不应记成 body_fall，也不应写成“成功稳定过材料”。[S7]

### 4.2 r56 修掉一个实证的数值映射缺陷

r54 保存的 RR calf 边界例子中，力矩反算位置输入高出原上界约 `1.0805e-7 rad`。r56 在原位置区间内生成精确可表示命令，重建实际 PD 力矩，并重新检查相同联合约束；不是放宽上界。该修改消除了该保存反例中的超界，并保持 Python/native 执行相同。它解释了为何 r56 候选 code4 型问题减少，但本身不产生完整前进步态。[S9]

### 4.3 当前存在短推进前缀与停步候选的相互限制

`progress.inc` 只在每次新计划的前 20 ms 请求推进，其后 330 ms 又沿 shifted policy。候选保留非承重腿输入、原可行基线的 roll/pitch 加速度与竖直支撑余量，最后仍过原 350 ms 预测、E 和同选择器。设计是有边界的，但 20 ms 的小冲量不能保证整个 horizon 存在前进的支撑序列。[S8][S10]

最明确保存现场是 r56 call809、15.76 s：

- 真正安装的 action_search slot0，全 horizon 四足承重，`J=0.4371103`，预测 x **倒退 37.735 mm**。
- supported_progress slot11 的预测和停步 shifted 候选几乎一致，倒退 **37.769 mm**，没有形成有效新推进。
- 正向位移较大的 slot4/7/8 分别先触发 code7；这不是单纯把有进展的候选权重调高就能解决，因为其中存在真实支撑不可行。
- 保存同输入私有对照把推进请求由20 ms延长到整个350 ms，倒退从35.805 mm改善为6.500 mm，但仍然没有前进。该私有路径没有原任务资格，不能拿它报“修复通过”。[S10]

同时，两个 N 版本每秒窗口中20 ms x预测与实际 RMS误差约 `18–71 μm`，很小；多数停滞后的已安装350 ms路径本身预测倒退。这支持优先检查全 horizon 候选生成、支撑交接、原目标分项与排序，而非先假定所有问题来自巨大的一步预测接线误差。20 ms 的位移预测准确仍不能证明350 ms接触可行性准确，更不能证明材料完整本构预测已接入。[S7]

### 4.4 数值假不可行确实存在，修掉它仍不够

r56 的支撑半空间投影是有限120轮坐标迭代后检查最大违约。保存的 call809 slot8 在15.993 s被code7拒绝，但对**相同归一化不等式**的既有 HiGHS 复核找到可行力矩见证，最大违约仅 `2.11e-15`；这是一处真实数值假阴性，不是合理的保守拒绝。[S11]

但不能由这个单点推导“只换 solver 就能过”。slot4/7 即使去掉额外 arrest 行，支撑原约束也不可行；slot8 连续四个保存1 ms子步出现假拒后，15.997 s的下一子步又有独立真不可行。执行端只核过有限尾部，未证明整个正向候选可用。正确下一步是保持原不等式、修可行性判定与候选生成两部分，保留真假不可行的不同原因。[S11]

## 5. 下一轮可一次完成的合理技术工作包

### 5.1 主线：冻结 F/r55，把稳定成果兑现为完整资格

先核现存 A/B E600 的 job/tmux/PID 与物理时钟。存在有效原进程就继续，不从t=0重启，不热更PANN、PPO、控制代码、workers或原合同。600 s和540 s仍是原物理任务时长/路线期限，不是后台墙钟超时。输出已有 E600 最终结果与完整原件；B/E300已发布的独立结果与资产直接复用，不重复物理或上传。若 E600 出现自然失败，再由该新轨迹最早偏差决定最小修复；现在没有证据支持提前重写 F。[S12][S13]

### 5.2 第二线：完成一个有统一机制的 N 新身份

下述内容应作为一个成组修复，而非拆成三轮无关参数尝试：

1. **修数值可行性与精确映射。** 复用 slot8 的保存见证检查相同约束，稳定地寻找可行投影或给明确失败理由；映射到原位置输入后用重构 PD 力矩再验原行。不能调松摩擦/力矩/材料域阈值吞掉假拒，也不能把有限迭代未收敛一律命名物理不可行。保留 slot4/7及slot8后续真不可行作为反例。
2. **生成完整350 ms可实现推进与支撑交接。** 当前20 ms前缀可保留为对照，但新候选需要在其自己的状态/接触演化下持续可实现推进。保存试验已说明“只延长冲量”不够；候选还必须具备可行的卸载、摆动、再承载过程，不能靠推四个始终不动的脚制造进展。选一个有机理依据的生成机制，不开参数网格或引入第三控制族。
3. **保留原J及同选择器，并公开分项。** 在相同当前状态、同horizon、同E与原检查下比较 shifted、恢复、推进和优化候选，记录每个原代价项、预计进展、支撑序列、首个约束及实际安装来源。先解释为何可行进展候选不存在或未被选中。不能强行装入未检查候选，也不能通过后验加成功奖励改变原资格语义。

最小验证先用已有 call809、call856 和确定的映射边界样例验证具体风险，保持保存诊断与新原任务分账；必要同输入/导数一致检查可在执行端进行，完成后直接用新身份、原 W44 A/seed1、原 checkpoint/normalizer、virgin 地面、t=0、workers=10进入原 E30，不另加无关机器人 smoke 阶梯。若通过且正常运动，依自己新身份建立后续资格；不得继承 N/r56 或 F/r55 资格。若发现新的明确首因，可在已授权范围内同轮继续修订；不以固定失败次数给整轮设停机点。[S8–S12]

N 的新原任务要等待重物理槽释放。F A/B E600仍同时有效运行时，N可做保存状态/构建工作，不硬开第三套机器人模拟。它是独立研发线，不应阻塞已达E300的F主线。

### 5.3 第三线：利用新稳定往返原件做材料/记忆工作

当前F资格链保持原权重。新稳定往返第一次提供了重复加载、跨足和返程数据，可用GPU做分组保留集、自由积分、观察器采样与决策敏感性分析；这比再拿W54的失败前缀反复解释更有增量。若要证明记忆有用，要在稳定基线上安排保持物理材料相同的“保留世界地块历史/去除控制可见历史”等成对对照，先验证干预确实改变控制接口，再开展已声明的原任务比较；不能把有重访、PANN调用或路线成功直接称为记忆收益。

两条正式物理任务加一条材料GPU任务可以并行。CPU核数与GPU显存是配额，不是科学目标；F的小CPU QP不会因96GB显存自动加速。所有新后台不设墙钟完成deadline，交付直接从服务器增量推Git/Release；无需用户电脑二次归档，已有引用原件保留。[S12]

## 6. 本审查没有证明的内容

- E600尚无完成结果；B/E300独立结果、发布资产与视频已核，B原数组尚未在本次审查中独立复算。
- 当前F可靠性仅覆盖两份冻结策略和已跑nominal任务，没有广泛扰动覆盖。
- A/B E300反馈中位耗时约4.21/3.92 ms，p95约51.41/50.65 ms，2 ms逻辑周期有大量超时；仿真物理时钟可稳定推进，不代表实现已能500 Hz实时部署。[S1]
- 材料模型精度、observer采样、材料状态是否进入动作以及控制基础稳定性是不同问题。当前N未完成任务不能直接归因于“PANN冻结太早”；当前F使用原冻结模型已能完成五圈，也不能反过来证明PANN误差对所有情形都无关。
- 保存局部模型/有限矩阵验证不是全状态安全证明。本轮没有运行新动力学来补强这些结论。

## 来源索引

下面的路径均锚定审查结果提交，避免后续 main 推进把旧证据替换。另有本轮独立复算 `control-raw-check.py/json`、`control-summary.json` 和证据线 `arrays-recomputed.json`。

- [S1：W55 phase、全部分支实际状态](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/phase.json)
- [S2：F/r55 A E300 原结果](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/f/job-20261120T035811Z-w55-f-a-r55-eval-PANN_PPO_S5-s1-E300-attempt01.json)
- [S3：r53真实调用链](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/scripts/s1_w55_f_r53_backend.py)；[r53保存状态核对](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/f/r53-same-state.json)
- [S4：F/r53 A E60 原失败与hold](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/f/job-20261119T041610Z-w55-f-a-r53-eval-PANN_PPO_S5-s1-E60-attempt01.json)；[F/r55 A E60 原结果](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/f/job-20261120T035811Z-w55-f-a-r55-eval-PANN_PPO_S5-s1-E60-attempt01.json)
- [S5：r55实际站立修复](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/scripts/s1_w55_f_r55_backend.py)；[三个保存horizon证据](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/f/r55-saved-hold.json)
- [S6：原评价与同case前序检查](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/scripts/s1_go2_r11_protocol.py)；[自己的原任务启动](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/scripts/run_s1_w55_f.py)
- [S7：N/r54完整实际分析](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/n/N-A-r54/E30-actual-analysis.json)；[N/r56完整实际分析](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/n/N-A-r56/E30-actual-analysis.json)
- [S8：N支撑投影和native检查](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/experiments/terrain_memory/w55_n/support.cc)；[r56推进候选](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/experiments/terrain_memory/w55_n_r56/progress.inc)；[r56执行与因果userdata](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/scripts/s1_go2_w56_n.py)
- [S9：r56映射修复](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/scripts/w56_n_prepare.py)；[原映射反例](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/n/r56-initial-propulsion-counterexample.json)
- [S10：r56 call809前缀及候选证据](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/n/r56-material-entry-prefix-evidence.json)
- [S11：r56同约束可行见证及后续真不可行](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/n/r56-native-arrest-witness.json)
- [S12：原W55计划的任务、并行、无墙钟deadline与直接交付规则](https://github.com/kaiwen123-yang/MCP-RL/blob/8c4768eae35a84b12aa8d44dbf00da20877c894a/docs/S1_W55_STABILITY_PLAN.md)
- [S13：最新补交的B/r55 E300独立结果](https://github.com/kaiwen123-yang/MCP-RL/blob/0eb39b13932c18d8e52ff305a3c35477a4510ab7/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/f/job-20261120T042344Z-w55-f-b-r55-eval-PANN_PPO_S5-s2-E300-attempt01.json)；[B/E300执行端分析](https://github.com/kaiwen123-yang/MCP-RL/blob/0eb39b13932c18d8e52ff305a3c35477a4510ab7/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/f/job-20261120T042344Z-w55-f-b-r55-eval-PANN_PPO_S5-s2-E300-attempt01-analysis.json)；[最新phase与E600快照](https://github.com/kaiwen123-yang/MCP-RL/blob/0eb39b13932c18d8e52ff305a3c35477a4510ab7/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/phase.json)
