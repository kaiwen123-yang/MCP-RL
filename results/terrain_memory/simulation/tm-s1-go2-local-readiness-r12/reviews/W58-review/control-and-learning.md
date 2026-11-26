# 2026-11-26 W57/W58 控制、材料、规划及资源审查

审查基准：`kaiwen123-yang/MCP-RL@ef500f31c7a96d94f83c9afb5c7a880864b447dc`。只读仓库与已推送证据，没有连接 D57、启动实验或改动仓库。下文 `R12` 为 `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12`。视频由另一审查分支处理，本笔记不声称已亲看视频。

## 1. 当前状态必须重新定位

`R12/w58/index.json` 在 2026-11-26 12:35:49 +08 已为 `finished`，12:49 观察时 `running=[]`，56 次尝试均终止，`selected_gait=G0`、`selected_material=M1`、`joint_eligible=false`、`joint_distinct={}`。`current.json` 中 W58 的 isolated-development 字样落后于它，不能据此说 W58 还在等待 W57。

56 次终止不等于 56 次成功：逐份 `w58/attempts/*.json` 重算为 21 次原资格 passed、32 次正式机制完整观察（4 passed、28 mechanism_criteria_unmet）、3 次 technical_interruption。53 个自然终局合计 8,226 模拟秒、41,130,000 个 0.2 ms 物理子步、292,338.596 CPU 秒。三次技术中断的末次观测为 87.002、58.002、12.002 模拟秒，不计入上述完整总数。原因是监测读取正在写入的 live.json 后 dispatcher/worker 丢失；旧尝试与原件保留，同冻结身份 a02 从初态重取，不能把它们称为物理摔倒或原步续算。

W58 发布完成回执：61 bundles、335 assets、121,557,127,453 bytes，远端 size/SHA256 核对完成。`w58/pool.json` 最后仍残留一个接近终局的 active 条目，时间早于 finished；不能把该残留当当前仍在运行。

W57 已从 11-23 尚在收尾推进至全部结束：full/blind 各 8192 distinct、32 updates，总 16384/64；B/P/R/PR/独立 blind 原 A/B E30/E60 均有资格，PR 自己又通过 A/B E300/E600。这里 PR 是 P1 加在 P0 上训练后冻结的 R，不是 P1/R 联合 on-policy 训练。正式旧机制 16/16 完整执行、0 passed，另 matched blind 4/4、0 passed。所有 v1 机制稳定到家但精确跨足几何条件没齐，收益随 case/条件方向不同，不能据此宣布记忆或联合协同已成立。

关键依据：`w58/index.json`、`w58/outcome.json`、`w58/attempts`、`w58/development-matrix.json`、`w58/validation-matrix.json`、`w57/index.json`、`w57/paired-mechanisms.json`、`docs/S1_DAILY_WORKLOG.md`。

## 2. G1：确实改善一项步态缺陷，但没有成为更好的整体步态

实际代码 `scripts/s1_w58_gait.py`：末摆动 5 kHz 首次触地锁存；下一次 2 ms 控制时接管；对末段过快下落设置连续 clearance correction；通过 quintic C1 offset 连接 swing、search、承载参考，保留原实际触地/承载确认、负载转移、力矩约束和物理状态，不靠直接改速度或把单脉冲宣称为承载。

G1 自己的 A/B E30/E60 全部通过。`g/selection.json` 的同case/同阶段比较：

| 比较 | 首触地至承载平均离力时间变化 | 原 slip_s_per_m 变化 | 正机械 CoT 变化 |
|---|---:|---:|---:|
| A E30 | −66.68% | +12.89% | +1.43% |
| A E60 | −63.74% | +18.94% | +0.324% |
| B E30 | −66.37% | +12.56% | +0.330% |
| B E60 | −65.90% | +9.93% | −0.261% |

平均离力由约 11.9–12.6 ms 降至 4.0–4.5 ms，bearing coverage 没恶化、low-support fraction 保持 0。四组滑移都更差，故采用 G0 是冻结条件的正确结果。不能说 G1 失败到不能走，也不能只展示触地更顺就推广为总体改进。

进一步研究价值：连续接管确有作用，代价可能在接管位置锚定、切向运动及接触法向/切向任务耦合；下一轮应分离法向消反弹与切向接触点/滚动约束，优化真实接触滑移和冲量，不只 off-time。球足中心位移与接触点滑移不可混作同一量：G1 比较的是原 `slip_s_per_m`，W58 机制表还列 `loaded_center_slip_per_m`，定义、单位均需保持。

## 3. M2：改善跨足误差，却没有解决恢复动力学根因

`scripts/s1_w58_material.py` 的 M2 是两条从 virgin 出发的因果路径。M1 在受载脚上估计单个 shared creep-duration 方向；M0 仅按相同测量边界传播。只有已经释放、两路都未受载的格，在发布时将 e/q/stress 换成 M0；当前承载格保持 M1，融合发布不反喂任一路内部历史。材料参数与网络权重全部冻结。它不是新训练世界模型，也不是高维状态全可观测滤波器。

两条 retained 轨迹结果：

| 轨迹 | 跨足 release→reload cell-force MAE M1→M2 | preload depth MAE M1→M2 | 非 release 20.2 ms whole-foot force MAE M1→M2 |
|---|---:|---:|---:|
| full B 0023 | 0.120394→0.081272 N | 0.5551→0.3786 µm | 0.122771→0.134948 N |
| blind A 0027 | 0.104945→0.071781 N | 0.4695→0.3274 µm | 0.117722→0.127346 N |

跨足力误差分别降约 32.5%、31.6%，但两条的短时预测损失均超过预先允许的 `M1 + 10%×|M0−M1|`，所以不晋升，保留 M1。新 M2 robot episodes=0、new PPO=0。离线 GPU 预测实际有 33075 paths/9649602 column-steps，不是毫无工作。

更深结果在 `m2/development-free-propagation.json`：从真实 release 状态离线起步，学习的自由演化仍给后续接触力带来平均 +0.128366 N（A）/+0.113645 N（B）分量。M0 的释放初态误差分量是 −0.041565/−0.037769 N，恰好相互抵消；35/35、31/31 个事件抵消，M1 只有 10/35、6/31。把初态估得更接近真值，反而去掉了这种抵消，所以只看最终 reload MAE 会误判模型。

下一轮应分别做 release-state、free-recovery law、reload contact inversion 三项误差分解与修正。优先发展带已知耗散/恢复结构的低维状态/残差模型，用真值作离线训练监督或诊断可以，正式在线输入仍只能用因果可观测量。不能把真 plant q 引进控制器，也不能用最终 MAE 一个数决定本构“更真实”。校准区间同域留出覆盖也并非全部达到95%，更不能宣称OOD可信。

依据：`experiments/terrain_memory/w58/material.json`、`w58/m/selection.json`、`w58/m2/errors-and-coverage.json`、`w58/m2/development-free-propagation.json`、`scripts/s1_w58_material.py`。

## 4. P2：已实装，但当前自由度比“动作条件世界模型规划”窄得多

四臂 B2/P2-full/P2-private/Fixed-wait 共用 public within-node contact availability 对原 force MPC 的修正，原代价与硬约束保持。该共同部分与旧 P1 的“历史/高度”项应分开归因。

P2 在新承诺 incoming pair 时读取已承诺 XY、落脚钟、当前/上次 MPC 力、最新材料发布。三个候选只有 **0、2、6 ms 等待**，共享同一个既定载荷场景：incoming foot 3 N 维持2ms，再按 solved pair threshold 每足40%体重维持6ms，旧支撑腿按当前/上次力分余重。它推一次 full/private 材料响应，估算 touch/bearing ready，再以 `d² + max(0,ready−nominal−d)²` 排序。候选间没有各自不同的完整 future load/foothold/contact trajectory，也没有多步优化接触序列；`support_forecast` 的代码和spec明确承认是 prescribed causal load scenario。

选出的 delay 修改 request due，仍经过原 observed support、bearing、load-transfer、release 等硬门。不是发送请求就自动执行。private 对照只重置已释放的相关私有预测列，保护当前脚全部 footprint 和 shared owners；因此结论是该干预下的增量材料历史作用，不是整系统全部无记忆。

### 新数据给出的判定

八个 P2-full 正式机制回合（development/validation × A/B × short/recovered）合计：3817 requests、3592 material requests、3413 次非空 private reset，**winner switches=0、observed gate switches=0**。full/private 在全部两种split、两case、两history的 moving slip、CoT、home-time 完全相同；history-action evidence 全 false。

同状态 full/private ready 最大差各回合 0.192–0.744 ms，7/8 回合最大值仍低于局部 uncertainty floor 0.451770 ms。唯一 A-development-short 最大0.744ms也未翻转候选。开发 A-short full/private 相比 B2/Fixed 出现小变化，但 full=private、返家慢30ms，不能归因历史，也不满足≤2ms慢行界。

先前保存状态校准：101次request、40次材料request、17次非空reset、0winner flips；full/private 全40次都选2ms。实际 ready 误差 median40ms/p95=104ms，full/private差median/p95均0、max0.139ms，步长减半仅1.57ns。故主导误差在机器人触地/确认/实际执行过程，已经不是积分精度问题。

`development-decision.json`：own E60与paired exposure为true，但 actual-short-target 同状态有效动作false，超过private/fixed的short收益false，所以新joint没获准。该停止不是时间耗尽，不应解除科学门直接跑49k样本。

### v2几何解决了一部分问题

新任务是12前向FR→RR目标、11反向转场，共23次穿越；相较v1不再要求无重叠的反向步完成相同机制。development 和新平移0.28m validation目标格不相交；short168s、recovered288s是物理观察窗，不是后台墙钟deadline。全部32条都完成窗口和返家。

| split/history | A真实目标覆盖 | B真实目标覆盖 |
|---|---:|---:|
| development short | B2/Fixed10，full/private9 | 四臂10 |
| development recovered | 四臂7 | 四臂5 |
| validation short | 四臂10 | 四臂12 |
| validation recovered | 四臂6 | 四臂5 |

总机制通过4/32，全部在validation B-short，四个臂都通过。它证明一个条件下稳定准确重访可实现，不能证明P2创新收益。几何曝光和效应可辨识仍须分开改进。

依据：`scripts/s1_w58_p2.py`、`experiments/terrain_memory/s1-w58-p2.json`、`scripts/score_s1_w58_p2.py`、`w58/p2/saved-support-calibration.json`、`w58/mechanics/*.json`、`w58/development-decision.json`、两矩阵与v2spec。

## 5. R与联合：代码存在，W58新联合实验没有执行

`run_s1_w58_joint.py` 已实现5801/5802/5803三seed、full/private成对初始化、两个各128条collector组成256distinct batch、原子占两槽、真实dt信用、节点2048/4096/8192、blind匹配full最终节点、自然尾段。启动时显式读取 `development-decision.json` 的 `joint_eligible`；现在为false，所以没有任何新head、seed outcomes、长时joint attempts。不要把“runner已写”当“训练已做”。

已验证长期可走的是旧r55/r60 baseline与W57 PR A/B E600；W58 B2/full/private/fixed当前仅自己的A/B E60及完整新机制路线。不应把新接口资格自动继承为E600，更不应将已有PR的稳定性当新世界模型的成果。

## 6. 可微动力学+世界模型如何接到当前系统：针对当前证据的建议

以下为设计推断，官方SuperMimic论文/源码细节由另一分支核验；不采用用户另一个复现工程的任何结论。

最有价值的可微模型边界是 **宏动作经现有MPC/WBC/硬门/保持器后的实际闭环事件演化**。状态应包含机器人运动、局部材料隐状态及不确定性、event mode、确认与承载计数、transfer fraction、上一force-plan、PD held-action等；如果模型只吃材料高度与request delay，会把主导的40–104ms执行误差继续遗漏。

建议结构：解析可微/可校验的material constitutive backbone + 可观测历史编码/低维残差 + 事件执行残差world model。输出真实下一事件的时间、触地/承载/卸载事件、状态、实际执行宏动作或动作影响、任务cost与不确定性。先预测事实，再用经过验证的局部梯度做短想象优化，远期由critic补充。

关键约束：

1. 将 raw requested action、经过硬门的实际执行、映射关节控制/力矩及延时全部记入训练。被门完全屏蔽的方向不应在模型里产生虚假的姿态/材料收益。
2. 对离散触地/释放模式，用事件模型或固定模式内导数；跨模式附近报告不确定性。不能把soft gate训练梯度当真实硬门导数，更不能移除真实硬门来“使梯度好看”。
3. 在完全相同的可恢复仿真状态上做真实物理±扰动，检验macroaction对next event/state/cost的方向、排序、幅度；不能只对surrogate自身finite difference验证autograd。若无法恢复material/solver/controller全部状态，应明确改用成对virgin rollout设计，不冒充same-state。
4. 离线旧W57±50ms数据覆盖P0，不自动覆盖新P2或新负载/落脚动作；数据必须带controller identity，在相同执行接口上补足有限、预注册物理采集。高频相邻片段不随机分到训练和留出，按完整episode/路线/历史分割。
5. 梯度可信域、模型误差与动作效应尺度一起验收。当前数十ms ready误差远大于0/2/6ms候选与亚ms历史信号，在此之前进行大规模BPTT只会更快优化模型偏差。
6. M3补自由恢复闭式结构/误差；P3让候选真的改变未来承载、foot placement或handover轨迹（先按预声明窄范围研究），R3通过模型学有真实权限的宏动作；再用完整真实rollout校正actor/critic。分模块与联合的对照必须持同动作权限，加入仅模型预测不用梯度、仅解析模型、memoryblind、matchedfixedwait等，才能归因每项贡献。

下一轮关键假设应从“再等一点就能利用历史”改成“何种真实可执行动作能产生超过模型误差与门控分辨率的历史收益”。必要时用新的已预注册机械任务/材料区间取得辨识度，但旧任务阴性保留，不能看新算法成绩后改几何/材料来挑胜例。

## 7. 六槽已经做到，算力问题是执行路径与任务结构

W58 物理源码4f7e006aa34f，P2 FP64 equal-columns在调用线程同步运行、native workers=1。初始3槽，实际采用同一租约池3→4→6。

混personal/scratch的4流虽然总吞吐升约43%，同scratch类每流只剩83–84%，连续两窗不合90%条件，按规则自然drain到3而不杀在途。后来全scratch重新标定：11-25 22:43:26–22:47:41六物理流共0.143470 sim s/wall s，四流0.097128、三流0.064371。六对四raw+47.7%、normalized work+46.3%、continuing streams保95.2%，因此保留六槽。随后11-26 09:20–09:24一窗每流0.891低于0.90，记为一次bad，未满连续两窗，尾段自然结束。

该六流窗口约6.02 CPU cores，projectGPU峰值4236MiB、RSS13.27GiB。分配3CPU affinity/slot不代表实际占满3核；串行物理/材料/控制器工作不能靠显存大自动并行。最新tail资源还出现其他项目占8.56CPU/大量GPU，后续仍需共享容量核验。日志里每回合候选P95可达约100–240ms、反馈约80–97ms，远高于逻辑2ms控制周期；这些是离线慢仿真，不能声称满足实机实时闭环。

合理下一轮并行：六个真实physics streams（或三组原子双collector）共享全局池；辅助核运行数据质量/视频/闭环计分；GPU并行批量material/执行world-model fit与短想象rollout。在有足够新任务价值前，不用重复相同无效候选来填CPU/GPU。每条后继用证据和样本节点退出，不设置墙钟杀停期限；性能240s观察窗不是后台deadline。GitHub直交与逐日日志已实现，继续沿用。

资源依据：`w58/dispatch/evaluation-capacity.json`、`evaluation-scratch-capacity.json`、`evaluation-throughput-readback.json`、`w58/pool.json`、`scripts/s1_w58_pool.py`、`run_s1_w58_joint.py`。

## 8. W59 接口建议：只扩大载荷与时序，不先动 XY

这部分是对父代理新要求的独立推敲。建议第一版结构化动作 `u=(δ_due, T_ramp, ρ_pair)`，分别是交接due残差、一次请求的载荷转移时长、两条真实incoming support腿的承载偏好。先不改变xy、摆动端点、腿顺序或原W44速度actor。`ρ`是力分配偏好，不能标成“实际承载比例”，实际值由全身动力学/接触/力矩约束决定。

### 具体代码接点

| 作用 | 当前接点 | 建议新增内容 |
|---|---|---|
| 每事件只发一次宏动作 | `s1_w57_f_backend.FController._new_handover_request` | 新P3/R3请求结构、唯一ID、nominal/P/requested/applied分账；提交后冻结至下次请求 |
| due与ramp实际执行 | `s1_w54_f_r52_backend.FController._update_schedule` | 一个request-specific profile hook，由实际确认时刻开始推进连续ramp；原ready条件逐项保持 |
| 计划时钟 | 同文件 `_forecast_events` / `_future_contact_feet` | 调用同一profile，不再一个地方50ms、另一个地方可变T；已有committed swing仍优先 |
| 实际载荷分配偏好 | 同文件 `_redistribute`，随后WBC的force_target任务 | 只在真实active contact上施加pair偏好；返回原required wrench，全部WBC约束保持 |
| MPC候选 | `select_force_plan` | 每个候选重建其自己的support profile、载荷预测和材料复制状态；不能继续共用P2固定3N→40%场景 |
| 材料法向执行 | `s1_w56_f_r58_backend.soft_reference` → `tm02.s1_material_prediction.normal_reference` | M3因果published state进入同一目标力求逆；保留时间一致、shared ownership与域拒绝 |
| 物理输入 | r52 `wbc`末尾 `u=q+(τ_start+5qdot)/60` | 原PD60/5映射一次，不再加residual torque或另加position residual |

当前 `_redistribute` 的权重是outgoing `(1−α)`、incoming `(1+α)`，并在`transfer_start!=None and all(active)`时起作用。可在incoming两腿添加正的对称偏好 `(1+ρ,1−ρ)`，保持其他值；它经过加权wrench分配与WBC后不保证得到同名比例，必须记录请求、target、solved、实测四级force/ratio。更规范的实现也可把该偏好作为小型受约束force-reference QP的目标，但不要绕过原WBC hard rows，也不要把同一required wrench再额外加到扭矩上。

动作窗口可先预注册窄诊断集，例如due±10ms（留在既有±50ms权限内）、ramp40/50/60ms、ρ±0.1，以零中心和单轴±探针识别敏感度；这些是建议的新合同参数，尚未验证。只有归因用开发数据证明可执行、域内、方向稳定后才冻结首个学习动作边界。不要看到胜负后挑范围，也不要因为所有动作被吞就降低bearing/release阈值。若持续无效，报告这项自由度没有权限并换研究假设。

### 硬门和不可微处明确列出

实际r52的touch>=3N、release<1N、连续6ms touch、4ms release、每足10%mg连续6ms bearing、pair measured>=65%mg、pair solved>=80%mg、released pair<=20%mg、body residual<=1.5、angular<=4、所有腿stance、α到1、due时钟共同决定liftoff。`active=scheduled & observed`也是离散选择。材料层还有footprint世界格创建、最低sole owner排序、单边接触max、根求解与域拒绝；QP也有active-set变更。因此不能把整个真实闭环宣称全局可微。

新R3可用learned execution surrogate 的路径导数，但必须如实命名为模型梯度；可在固定模式内校验，模式选择保留枚举/采样或显式事件分支，不用straight-through硬门冒充真实物理导数。真实动作有效性同时看actual gate时间和ramp期实际force/torque：同一liftoff tick不代表ρ/ramp动作无作用。但仅request不同、预测值不同也不够；需要连续两个2ms hold的送达力矩/受载轨迹差超过本身ripple与重构误差，并能绑定真正后继加载。

新actor接管宏接口后，旧W57一维timing agent必须停用；旧W44 speed cap可作为共同冻结背景保留。组合只进行一次 `P3_base + bounded_R3_residual`，投影/裁剪一次并保存前后值。现83D/1D checkpoint不能直接当新三维策略继续训；新spec/schema、独立初始化、optimizer和normalization都应有身份。

## 9. M3/执行世界模型/R3的一个可执行分解

### M3

先按三个误差来源分别建立损失与保留集：release状态误差、从正确source态开始的自由恢复误差、正确reload态下的contact inverse误差。可保持现有能量势结构，以正半定mobility `D=L Lᵀ` 或保持凸性的耗散势修正自由演化，要求无载自由能不增、零外力边界一致、原应变域内、法向切线和压力约束成立。不能在force输出任意加神经残差而破坏能量/耗散后再只测总体MAE。

在线隐状态编码器只能从完成的脚运动、测力和事件历史推断，附不确定性；真plant e/q可作事前声明的离线监督/分解，不能作为部署输入。低维可辨识latent可优先于不可观测的逐格全状态点估计。保留M1承载精度门，并在释放后恢复机制上独立过门。

### 执行世界模型 E

E预测“已知机器人动力学+新宏动作经闭环执行的残差”：接触/承载/卸载时刻、actual load profile、状态后继、能耗/滑移与不确定性。材料预测由M3传播同一candidate实际预测载荷，不能E预测一套力而M3吃另一套P2固定场景。输入必须包含旧plan、transfer phase、确认计数、held PD输入和mode；当前ready的主要40–104ms误差才有可能被模型解释。

先用已存日志做事实拟合和整episode留出，再补新接口的有限受控物理探针。每条旧数据保留controller/action schema，不把W57/P0的数据假定为新P3闭环；全系统状态重载若缺controller/solver/material内部状态，paired virgin episodes就是合适替代，但不可标same-state。

### R3短想象学习

actor输出宏动作，冻结M3/E参数但保留对输入的梯度，短展开2–4个真实事件（初始建议；不照搬32步），末端接实轨迹校准的V。M3/E只用事实预测损失更新，不允许跟actor一起改模型参数使虚拟回报变好；交替拟合后，必须重新做模型有效域/梯度和真实闭环检查。模型误差、接触分支熵或OOD超过已注册界则不采用该梯度，回退共同MPC/PPO更新而不放松物理门。

梯度验收对同一个宏接口在真实plant做有物理分辨率的±动作、比较next-event/force/cost方向和候选排序；只检查autograd对自家forward的数值导数不够。离散门附近允许真实导数为0或不存在，不能靠改成soft gate得到一张漂亮非零梯度图。有限扰动可以检验有意义幅度上的决策价值，不冒称精确无穷小物理导数。

## 10. 必须继承10-20伴随负结果的教训

已读取 `docs/go2-adjoint-diagnosis-2026-10-20.md` 及 planner/KL/terminal详细诊断：

1. **终止吸收与任务代价一致。** 原实现complete后仍传播状态、每步计固定时间和几何代价，25/25终端候选V虽已清零仍错排，9/25拒绝自己预测更早完成的候选。W59想象器遇真正任务terminal时，所有后续运行代价、终端余程与bootstrap一致吸收；如果任务定义要求到家后继续保持观察，那个保持是任务状态的一部分，不能在first-home提前terminal。自然E60/168/288观察截断是truncation，可bootstrap但切断递推；物理failure是真termination；技术异常不能伪造terminal后继。
2. **宏端点不能认证连续条件。** 旧200ms端点漏20ms内部速度峰，案例16.80s宏判完成、真实26.44s。W59要预测/重建区间内2ms确认/保持计数及0.2ms力事件，至少输出区间违例、first-passage与reset事件；不可只拟合宏终点低MSE就宣称touch/bearing连续门准确。终局验收使用原细粒度task判据。
3. **真实接受的策略更新量。** 24/24旧轮联合12D KL超0.02，22轮仅1step；pre-step检查不能撤销刚越界的更新，末尾KL也不是触发KL。R3若继续Gaussian策略，应在固定audit states上做post-update解析joint KL，预先规定回滚/缩步，rollback包括actor和独立optimizer状态；记录accepted/rejected steps、触发KL、整batch KL。新3D联合KL量纲与旧12D阈值不能直接混淆。采样保存raw Gaussian、tanh后宏动作、投影后动作与实际执行，不能用裁剪动作算原Gaussian logp。
4. **优化器独立和actor/model时点。** 旧Adam共享污染已有明确无效批次；新全/盲、seed和ablation必须deep-copy独立状态，记录共同初始化哈希。critic在随机采集与确定部署、更新前后分布的差异要单独校准，不能把被mask成0的V强行解释为具体行为首因。

## 11. 六条并行工作线及必要对照

工作线与物理槽分开定义，六个工作主题不等于同时启动六个互不协调的dispatcher。

| 线 | 可并行工作 | 结束证据 |
|---|---|---|
| L1 M3 | release/free/reload误差分解、受约束恢复模型，GPU成批fit | held分项误差、耗散/域/承载精度均通过 |
| L2 E | 闭环执行数据集、事件与实际load的world model | 整episode held error、区间门语义、误差/效应比 |
| L3 P3 | 同接口动作权限、候选独立载荷传播、匹配blind/fixed | A/B真实连续动作、实际重访绑定、数值与物理扰动相符 |
| L4 R3 | 新schema、短BPTT、真实dt、terminal/KL处理、model冻结 | 无新物理也可先完成数学/离线证据；物理采集在L2/3门后 |
| L5 G/metrics | G1切向代价归因、真实接触滑移/滚动区分、几何命中原因 | G2若尝试，独立资格且不可混入主比较中途；保留G0主线 |
| L6 evidence/throughput | 完整source/normalization/attempt账本、视频/逐日日志、native/GPU批量profile与发布 | 6槽实际吞吐、无其他项目冲击、已核远端原件 |

物理阶段一可用6槽同时跑3组paired probes/qualifications；每个pair同初态/同随机流，仅一个宏参数不同，两case按预声明块覆盖。后续如R3门通过，6槽可原子供3个双collector campaign；GPU成批模型fit/想象与CPU plant保持独立预算和同一资源余量规则，不能两个队列各自认为有六槽。已在途动作不因review结束或性能观察窗结束而kill。

最低归因对照应共享新动作权限和相同几何：

- **B执行接口零动作**与原B2核对；另保留强简单fixed宏动作（在开发集事前冻结）以分离generic waiting/load shift。
- **M1 vs M3**：同一P3、同一宏动作/受控输入比较，明确是状态/恢复改进。
- **P3-full vs P3-private**：同候选、同execution model、同预算，仅已释放查询历史干预；current loaded/shared cells保持。若E或actor latent含历史，相应blind必须一致处理，不能只遮旧12维预测而泄漏新latent。
- **PPO vs short-BPTT**：同M3/E/P3接口、同真实样本上限、同初始策略；再有“用同模型做候选规划但不用actor模型梯度”对照，避免把多模型计算收益都算BPTT创新。
- **联合全模型与逐模块移除**：通过前序门后才在冻结协议中检验，不在同一持出集反复选winner。先等物理样本对齐，另外报告GPU模型步/墙钟/总算力；imagined steps不冒充distinct real samples。

完整2×2×2或更多组合不是第一天全部长训的理由。先靠各线的窄可归因诊断决定哪几个模块具有可观测作用，再固定联合比较、三独立seed、matched blind预算和独立验证。已知W58阴性不会被新设计替换。
