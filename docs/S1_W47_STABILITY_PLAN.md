# W47稳定性计划：搜索路径一致性与可执行移动候选

<!-- W47_EXECUTION_BEGIN -->
> 执行更新（2026-11-13）：W47 原位分析、B 来源核验与 C 全量复算已完成；隔离诊断构建已完成，A 的 8 条恒等修复配对验证待启动。 唯一 R8 原件在 D57 不可用，最新 W46 A/B 所需完整策略与同时间预测缺失；B 未启动 donor solve，未把历史 fallback 当最新双流覆盖。1944 次原调用中有 1601 次正步安装（1598 次 J 下降、3 次风险优先恢复）与 343 次内部停滞。N/r34 strict 保留，新 PPO 与新原 plant 回合均为 0；本轮诊断构建不冒称已部署 S5-r36。以下冻结计划和 W46 审查按其形成时点阅读，当前执行状态以索引为准。 [W47 实际运行索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w47/phase.json)。
<!-- W47_EXECUTION_END -->

更新：2026-11-13。审查结果基点：`15ee620b8a909246a12baf999c136c2be299655b`。本计划为 **planned_not_implemented**；r36尚未实现、构建或启动。本次审查新增服务器任务、原plant实验、MPC求解和PPO均0。将本文件交给有D57访问能力的Codex后，可自主完成以下有限工作及满足条件后的原任务，不需再次询问已授权的构建、实验和GitHub发布。

[W46最终审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W46-final-review/README.md) · [控制证据](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W46-final-review/control-report.md) · [GPU证据](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W46-final-review/gpu-report.md) · [文献](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W46-final-review/literature-and-next-direction.md) · [机器交接](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W46-final-review/handoff.json) · [启动提示词](S1_W47_START_PROMPT.md)

## 1. 本轮要回答的问题

W46/r35 constant已自然结束。A E30 26.432s躯干触地，B 11.638s侧翻；合法去程/返程/往返0，E60/300/600 blocked，新PPO0/0，各仍14737/update57。96个Release资产已经完成交付，不能因旧“运行中/尚未实施”文字重启W46，也不重跑W45/G5。

四个保存上下文的12次solve中，60次backward全部完成，540个正alpha native候选没有模拟/QP失败或trunk violation，却全部J更差。A153的局部变化来自旧nominal倍率重选，constant/load选中相同完整策略；不是产生了新下降方向。B155/B158的alpha0也没有重现nominal成本。因此，本轮核心是：

1. **零步与方向是否在同一实际执行问题上成立？** 找到首处分叉，修有证据的执行错误或确认局部敏感性/导数问题。
2. **当前模型里是否缺少一个有用的移动输入种子？** 固定一个已有成功R8来源，在当前模型、原投影与共同目标下检验。
3. **实际全程搜索与GPU诊断究竟说明什么？** 服务器原位读取已有记录，补全1944次真实调用分布和GPU测量语义，不再只看汇总标签。

A/B两条科学线可并行准备；正式比较共享冻结输入与执行语义。Task C独立，不阻塞CPU。不是继续扫输入惩罚、更多训练或同轮重写FDDP。

## 2. 开场与共同基线

### 2.1 只从真实状态续接

默认SSH `autodl-d57`，CPU树`/root/autodl-tmp/MCP-RL-w42`，GPU树`/root/autodl-tmp/MCP-RL`；环境`work/envs/d57`、`work/envs/w41-mjwarp`。先只读核当前main、未提交修改、cgroup/affinity、RAM/GPU/磁盘、tmux/PID/job/物理clock。存在有效W47时续接该任务，不能重复开同一工作。W46终结结果保留；不用旧launch快照替代final phase。

服务器原件/checkpoint/resume/build均原位只读复用。当前审查的批量原件读取被自动审批按“额外电脑下载/归档”拒绝，不能换URL/connector重试；下轮直接在D57已有文件上核SHA、读数组、产出紧凑诊断。若指定原件缺失，先核已有manifest和服务器路径，然后记录确切缺项；不要在用户电脑收集，也不改为重跑已完成任务来补原件。

### 2.2 固定问题，不隐式更换控制任务

共同起点是S5-r35 constant、N机制、r34 strict及已部署二进制SHA `a32287b0821d9a77687575250dcb2379e370dbc84264fae80a4091027db7345e`。旧r35作为只读baseline；有实现修改后才创建真实r36 spec/patch/build，不预先写“已构建”。

保持原Go2/TM02、质量15.206408kg、PD60/5、每腿力矩23.7/23.7/45.43Nm、2ms held目标/10×.2ms物理、.35s horizon、2Hz/.45/.005 C2步态、原高度项、r28 trunk/r27 C2/r26 packet、PANN/658D/2Hz cap/W38奖励和任务判据。r14全局支持修复不撤销；短腾空不成为新失败；不以raw Kδ大小代替实际动作与恢复。

每个有限问题冻结当前x0/time、mocap/userdata、R32 private gait state、当前材料proxy、完整incumbent及其K/H/scale、模型/编译/时钟身份和一次共同r35 ubar/F/R/hash。所有baseline、donor及新候选使用同一个J0+Jref。参考从原current incumbent生成，不能在另一个进程导入donor后自动改成donor自己的参考。

缺失原历史pre-call私有状态时，命名为共同重建context；不能称exact historical replay。所有动态推进计入实际native rollout/FD数量，不把“8次solve”写成仅8次积分，也不把内部候选拒绝计作任务失败。

## 3. Task A：零端点、真实搜索方向与必要修复

### 3.1 先使用已有完整数组

固定四个已声明W45保存上下文：A148/plan136@2.54、A153/140@2.64、B155/141@2.68、B158/144@2.74；首先分析W46 constant，none/load现有数组用于检查结论是否依赖参考惩罚，不重跑12个solve。

读取服务器已有`r29-current-plan.bin.r33-candidates`及对应完整policy/input/context/reference，按来源manifest核size/SHA。对每个已有attempt的best nominal、alpha0、记录中最好正alpha做逐节点比较；完整采样数据可回答时不新增rollout。保留旧完整policy对象与递推用actual nominal轨迹，不能把二者混为同一个反馈表示。

最低必需输出：

- 唯一candidate identity、reference hash、origin/time grid；u_ff、alpha*k、新旧K/H/scale和已执行accepted u分开。
- 第一处状态切空间差、raw/ctrl-clipped/projected/实际held u差，首次超出同输入重复计算噪声包络的节点及其前后值。
- 连续ActionImpl与节点ActionIndex的选点/插值、四元数自差与归一化、瞬时PD/ctrl边界、活动面、QP KKT与投影是否改变中心。
- 首分叉前后接触/法向力、每项J0与Jref；沿现有alpha网格的真实ΔJ及`−alpha*(dV0+alpha*dV1)`。dV未保存时明确缺项，仅在必要的有限补录中新增。
- 所有完整policy/alpha候选采用实际native模型和原H-QP；仅第3.2节已标记的post-QP固定输入隔离例外。模型已禁warmstart，不预设qacc历史是原因。

数学检查：如果初态、f、时标、实际输入、投影与动态可行nominal完全一致，alpha0即便更换K也应由x=bar_x、u=bar_u逐节点再生。源码里的“alpha只缩放前馈”不能解释所有端点差。先定位第一处分叉，再讨论接触/反馈对微差的放大。

### 3.2 缺字段时才补有限诊断

必要时每fixture最多四条350ms固定策略native rollout：旧完整policy重复；accepted-u开环重放；同一actual x/u中心上的新K/H alpha0；已有最好正alpha。最多16条，不重新优化，不在窗内每20ms重规划。每条从同一重建初态/模型开始，完整记录首分叉，复用已有字段不再生成重复大数组。旧完整policy用原ActionImpl；alpha0/正alpha用原ActionIndex＋RolloutDiscrete，保存候选已包含alpha*k时不得二次添加。现有R34_SCORE_ONLY/run_native走ActionImpl，不能直接代替搜索路径；如需比较同一候选的部署ActionImpl，应作为16条预算内有标签的隔离项。

accepted-u开环隔离逐节点直接重放保存的post-QP held u，保留原native动力学与PD/actuator clamp；H-QP只shadow核边界不修改此条固定输入。若重新投影改变u，该条必须改名为零K投影策略，不能据它独自判断f/初态不一致。这仅是离线隔离，正式policy及alpha候选仍执行原H-QP。

开环也分叉，先查输入/模型/完整初态/时标；仅新K/H分叉，继续分离活动面、索引与反馈敏感性。只有需要定位且原数组不足时，才在这16条预算内替换一条为“新K旧H”或“旧K新H”隔离；不能额外展开所有组合。

若发现明确实现错误，修最小可定位链路，例如同节点时间/插值/动作生成或参考表示不一致。修前证据保留，修后对四fixture各一次old/corrected配对，最多再8条固定rollout。不能用放宽r34、重新允许alpha0安装、任意裁K/目标角或修改物理来掩盖问题。

**零端点恢复不等于有下降方向，更不等于行走稳定。** 若端点自洽而真实方向仍不降，交付有效导数/接触活动面或局部恢复性问题的具体证据，不自动改成任意K插值或FDDP。Task A最多24条新增固定rollout，已有数组足够则为0；修复循环不能变成无界参数搜索。

## 4. Task B：一个固定R8移动输入种子的有限对照

### 4.1 先固定来源和四个问题

固定来源候选为已验证R8/S3的`project-s3-formal-v1-single-yaw_plus5-trajectory.npz`，size28100505、SHA256 `079f57e1ebad260c9cdac842dc089012e29cb46992ab64d2c31cd9d380b61625`。按R8索引找到服务器原位文件并核验。不存在可用原件或关键相位字段不足则明确`donor_unavailable`，不能改挑其他seed直到成功。

四fixture为B155/B158，另加W46 A/B各一个最新早期事件context：按已记录500Hz膝关节首次±45.43Nm反转对应时刻（A6.860s、B6.150s），取该命令之前/当时最后一个身份和字段齐全的已部署planning context。事件只用于固定取样，不成为新失败阈值；用实际日志确认call/plan ID，不凭时刻算出假ID。manifest保存source_round/source_case/source_time、事件时间、距事件时差及该policy当时实际部署的调用关系；不能无界向前找旧context仍称对应当前事件。

若一个新context缺完整input/gait/reference，分别按预声明fallback A148/A153替代，记录缺项与替代原因，不按候选表现挑选。fallback均保留真实W45-A来源，只用于历史机制诊断；不得重标为最新W46-A/B，也不满足第6节最新双流覆盖门槛。先把fixture manifest、哈希与选取规则推Git，再看donor结果。

新W46 context的±偏差使用前一份已保存accepted预测在同一origin的状态，与实际x0作MuJoCo位置切空间差和速度差，保存预测源文件SHA/plan/node/time及delta。没有时间匹配预测则标perturbation_unavailable，不拿W45 delta或零向量凑24条诊断；缺失的最新流恢复证据不能自动过第6节。

Task A与Task B的原位分析、donor映射可以并行；如果A需要修控制执行语义，B的正式solve等共享修复基线冻结后才执行，避免用不同问题混比或重复8solve。

### 4.2 输入种子的精确定义

R8有实际500Hz x/u，但没有完整K/H。其trajectory.phase是PHASES.index(mission.phase)任务阶段枚举，**不是native gait phase**。用已保存resolved.applied_settings、对应版本GetPhase初值/重置和cadence切换逻辑确定性重建R8 gait phase（已见0s Stand/Cadence1、2.002s Trot/Cadence2，最终以该源记录为准），与当前R32 private gait state核对。

唯一选段规则是当前episode elapsed对应R8最近2ms原节点、等距取较早节点；记录时间取整差与由phase_velocity=2π*cadence导出的相位取整误差。要求整个350ms来源范围存在、任务移动阶段相同、重建相位差在该取整与FP64算术误差包络内；不匹配则source_unmatched，不平移时间/拉伸/搜另一个相位片段来凑条件。两个时钟不能仅因同是2Hz就假定相同。

只对donor机器人参考做xy平移、yaw刚体锚定，四元数正确复合，速度按实际自由关节坐标约定处理。z、关节构型、速度剩余差逐项报告。世界材料/ground/goal/observer/history始终用当前context，不随donor搬动。R8的3cm足轨迹与当前.005m目标差异显式列出，不将旧3cm参数写回当前控制任务。

donor实际q_target是输入seed；初始化K=0，H取同一current incumbent在当前时间网格上的有效SPD metric，仍走原Clamp/H-QP，metric_valid保持真实有效，不借关闭metric绕开PD边界。seed中继承H、零初始化的Q数组须注明来源，不能冒称R8已求解的Riccati对象。

从当前真实x0由当前native predictor完整重滚，再进行正常N backward/搜索。当前single shooting会覆盖donor x，因此旧x只作相位、初差和动态defect诊断；本分支叫“移动执行输入种子”，不能叫已实现独立状态—输入热启动/FDDP。

### 4.3 预算、评分与结果

每fixture调用第4.4节同一在线候选入口：先current家庭一次，若明确停滞且source合法才donor家庭一次，各用现有有限solve预算，最多8solve；未触发/无源就如实少跑，不为凑数另开donor求解。保留current incumbent、current best nominal、原donor及全部新候选完整对象，在**同一J**下作联合裁决，不因新进程遗漏baseline。原r35共同参考固定，原J0/Jref和总J同时给出。

每家庭选中完整policy，从原初态及记录尺度±偏差各一次350ms fixed-policy诊断，最多24条；窗内只2ms反馈/投影/保持，不20ms重新优化。这不是原plant新E30，也不是精确历史重放。预算中已有相同identity、同context的结果可复用，避免重复。

要论证donor单步缺口，须计算 d[k+1] = f_current(T*x_R8[k], projected_u_R8[k]) ⊖ T*x_R8[k+1]：每个旧机器人状态分别在当前绝对时间、冻结材料proxy/原PD/H-QP下作一个2ms转移，再与变换后的下一源状态作流形差。不能拿当前single-shooting自行生成的x算缺口（它按构造接近0），也不能用初态差或长链偏移替代。若原件没有这些转移，额外最多4×175=700个held单步评估、7000个.2ms native子步，单独计数，非新plant episode；初差、逐节点动力学缺口与长链误差分列单位，不混成一个状态分量最大值。缺这项证据就不能据表格宣称已识别单射击破坏了良好状态轨迹。

逐项给出：投影后Δu/实际PD力矩、实际前移与reference要求的前移/比例、足接触与原trunk/跌倒条件、最低高度和末端恢复状态、J0各项/Jref、动态缺口与初差。原生20ms前缀力矩与完整350ms力矩分开；不把毫米级驻留、短窗更温和或较低总成本直接当作有用行走。

| 观测 | 结论与本轮动作 |
|---|---|
| 当前模型中donor可执行、有前移/恢复收益且同J接纳 | 支持初始化/候选不足；可进入第6节资格判断 |
| donor在J0/运动上有益，只因Jref被拒绝 | 记录参考代价冲突；不在本轮给donor免费参考或扫R |
| donor更稳却J0更高 | 列出冲突分项，作为下一单机制目标研究依据 |
| 单步尚接近，但长链重滚破坏有用x/contact | 单独论证状态保留/FDDP/MS/SQP；本轮不自动重写 |
| 一开始的输入/接触就在当前模型严重不匹配 | 当前问题需生成可行参考，旧成功不能直接迁移 |
| nominal有益、±偏差失效/没有持续前移 | 反馈恢复尚未建立，不凭seed宣称稳定 |
| 两路线均无有用候选 | 有限负结果结束，不重复已知E30、不换seed凑成功 |

### 4.4 冻结的在线规则：仅停滞时追加一个donor家庭

若Task B支持部署，本轮唯一允许的在线方案是：**每个去程/返程moving problem epoch先执行原current家庭N solve；内部stop_reason为native_search_stagnation、且第4.2节固定source映射有效时，最多再执行一次donor家庭N solve。** Stand、转向、终点保持不导入移动seed；current已得到正步下降或source缺失/越界/阶段相位不匹配时，直接保留current完整结果。

在current solve之前保存本epoch输入完整incumbent P0，冻结x0/physical time/mocap/userdata/private gait/material context、由P0生成的ubar/F/R，以及donor借用的P0 H。current的结果仅暂存，不能把其新选择另当P0。两个solve期间不推进plant、不更新observer。第二家庭使用相同目标、同初态与独立初始化的有限数值工作区，不能继承第一家庭的stopped/lambda递增/archive/cache分数，也不能重冻为donor参考。触发只读本epoch已经完成N搜索的内部stop_reason，不读顶层return_reason/chosen_alpha，也不沿用前epoch stopped。每个家庭的正则化起点/最大attempt与有限比较一致；immutable问题身份共享，family/候选generation/评分身份单独记录。保存current incumbent/current best完整对象，再用原r34真实(v,J)准则和数值容差联合裁决，平手保留current baseline，donor不会自动覆盖。family的暂存plan/generation ID分命名空间；联合赢家才分配实际plan ID并向执行器安装一次，不把内部current暂存policy_installed记成已实际执行。禁止把获胜rollout覆盖原candidate反馈reference后再安装。

将该逻辑实现为可复用的内部入口，离线fixture和原E30调用同一个入口；原在线过程保持存活，不在每个20ms周期启动新agent_server/外部进程。不得用一次MJPC_R33_INPUT_POLICY进程启动导入来冒充这个规则。每epoch最多两个有限solve，失败后不在同epoch循环；新增call/rollout/FD/time、current solve数、eligible/triggered/source匹配/跳过、donor solve数及获选数全记。source窗口合法只依据固定来源与当前已知mission/gait，不读取未来真实plant状态。elapsed固定为回合物理起点后的仿真时间，不切换成阶段elapsed。若所有声明context都不匹配，记donor_unavailable_for_declared_contexts，不能称donor机制已验证无效。

这可能使停滞epoch的native优化量接近两倍，不宣称提速。只有它带来有用移动/恢复才值得原任务检验；两CPU流各10worker的额度不因两个家庭叠加而变20worker/流。若仅Task A正确性修复已满足移动证据，则可部署不含donor的修复版本；二者不能不加区分地混称donor收益。

## 5. Task C：实际调用统计、GPU记录修正与可选单hold分层

### 5.1 零新physics的主体

在服务器原位读取W46 A/B已有完整planner-calls（1342/602，合计1944，包含启动调用），按Stand/moving、r35 enabled/disabled、case/time统计candidate_source、stop/return reason、正alpha接纳、nominal倍率与incumbent保留、J变化、rollout/FD/QP与记录量。将内部候选拒绝、物理失败、技术错误、blocked分别计数。对首次大目标/力矩事件回溯对应完整policy，不用12个fixture替代全程统计。

原位核验GPU已有paired/action/scenario输入，恢复场景和关节索引、raw/clip/held口径、base/姿态切空间/关节/速度不同单位的误差。原件没存的raw device输出、.2ms接触或KKT不能补造；按公式离线复算须标明重算。

修四项诊断：

1. raw比较前显式copy；native投影传独立copy，分别保存raw、clipped、held，不让原位别名覆盖投影前量。
2. 明列decision_time、hold_start/end、post_state_time；+10ms命令产生+12ms状态，不混写时标。
3. 删除“trunk都0就自动满足扩窗”的隐式结论；历史350ms保留为有限负诊断，不重写成已过原门。新任务只按本节单hold范围运行。
4. 区分实际warmup、CUDA graph capture记录和capture_launch执行；W46 compact所写48保留原记录，追加更正为源码/API推导24实际+24记录，主循环42000不变。

GPU原8policy/24scenario来自W45保存的baseline/selected，不是r35 constant/load新selected。按完整执行数组/scale/初态/时间/model identity去重，名义24与有效18分别报告。现同state动作层对照是Warp公式/Python公式+共享native H-QP，不是24场景逐个调用完整native Policy::Action；已有bridge全链补证只覆盖4个selected nominal。

### 5.2 可选且严格有限的分层

只有原位复算仍留下明确的首hold差异、现场已有接口足够且资源可用，才执行一个身份齐全的场景：同初态/held输入/proxy下，native3.2.3、CPU MuJoCo4 FP64、MJWarp FP32各推进2ms。**新增真实物理子步总上限30，含任何实际预热；capture记录不算执行。** 所需额外预热/接口准备使上限内无法比较，则如实结束，不扩窗或扩大场景。

记录compiled-model差异、每.2ms状态/控制cast、接触对/距离/法向力、可取得的solver读数。接口不支持关键读数则明确不可识别。CPU4→MJWarp同时改变backend和精度，仍不能称纯FP32因果。无新24-world/350ms，无GPU完整J/risk则无等价排序；不重跑G5或reference proxy。

## 6. 何时进入原A/B任务

下列材料齐备后Codex自行判断并继续，不为重复许可停下：Task A有明确一致性结果/修复；同一最终执行语义下Task B及必要原基线比较完成；两个流的**最新W46代表context**都有可解释的移动/恢复收益，且各自有时间匹配的±扰动证据，不暴露新的原硬失败；历史fallback或缺失/零扰动不满足此覆盖。在线候选生成、触发及联合接纳必须与有限入口相同；结果不只是旧scale重选而仍驻留，也不只有proxy变好。

若证据只支持正确性修复、不支持移动收益，如实停在有限结果；如果donor只是一次静态种子有效，不能假装receding-horizon已经验证。进入原任务后才能检验重复重规划是否保住收益。不得保证E30会通过。

选择至多一个主控制候选，两流同版。A的必要正确性修复可成为共享baseline，但B相对该baseline的增量效果必须单列；不得把复合改变全归因于donor。冻结真实S5-r36 spec/patch、build/源SHA、接口和记录，再推Git。所有prepare/runtime/policy/records白名单、contract active_controller_build/controller_artifact/selected_mechanism、CLI/env和binary readback一致，明确N/r34 strict，防止r33默认S回退。--build不自动修改合同。

A固定W44-A seed1的checkpoint-014737.pt，B固定W44-B seed2同名checkpoint；路径见第9节。各Actor mean/normalization/RNG/case按原身份，PANN网络固定，但observer/material memory继续按真实观测因果更新；新PPO0/0、计数14737/57，不自动执行未来15249/59。

使用`scripts/run_s1_go2_r12.py`单次evaluation入口，明确`--attempt --evaluation-start-unix --checkpoint --level --seed --workers --build --contract --condition nominal`；不用要求本phase新训练的`--job-dir`。新建W47 phase/job，不覆盖W46。先推实现/spec/有限证据/真实phase，再启动并确认两条物理clock及记录增长。

| 级别 | 原资格 |
|---|---|
| E30 | 满30s，合法实际材料去程，满足原3m目标的到达容差并进入原转向条件（不新增x≥3.0硬要求）；不强加完整返回 |
| E60 | 独立满60s，至少一次合法往返，回家连续稳定2s |
| E300 | 独立连续5合法往返，270s内路线完成，观察满300s及原稳定条件 |
| E600 | 独立连续10合法往返，540s内路线完成，观察满600s及原稳定条件 |

每case自身通过前级才进后级；A通过可继续，B失败不阻塞A，但不能借A替B越级。失败结束当前attempt，后级blocked；不换seed、截取长轨迹前缀或拼接资格。完整足迹实际Fz>0才计材料穿越；原主应变max(abs(principal strain))≤.02、非足接触/跌倒/路线不放宽。Actor checkpoint不能恢复中途plant/材料/MPC/任务完整状态；技术中断若确需续跑，另建独立attempt并保留原interrupted记录。

## 7. 25CPU/96GB GPU的并行使用

W46记录25CPU、120GiB RAM、97887MiB GPU且有两项无关GPU任务；不是当前实时可用量。开场实测资源，不动无关进程。按cgroup允许的CPU编号映射：

| 工作 | CPU额度 | 实际安排 |
|---|---:|---|
| A诊断/有限solve，后为原A |10|全进程树隔离；Torch/BLAS/OMP各1 |
| B种子/有限solve，后为原B |10|独立输入/输出与可写目录；同样单线程库 |
| C原位统计/可选GPU |2总host|含native确认；不得额外占一组10worker |
| 系统/主控/发布 |约3|留I/O、日志和串行Git写余量 |

build-j6先于双重计算负载完成。共享static assets、环境、父输出只读；可写upstream/build不能经共享可写软链。两个solve同时最多20CPU，不在原双E30之外再挂两组solve；A/B依赖相同冻结context/reference时先完成冻结再排队并行。

CPU A/B中Actor/PANN也用GPU；继承`XLA_PYTHON_CLIENT_PREALLOCATE=false`、`JAX_ENABLE_X64=true`。按现有双流运行中的实际GPU峰值和剩余显存给C准入，不能假装96GB全空闲或凭空固定三等分。不单独增加无意义benchmark，只随真实任务记录CPU/显存、planner分项、I/O和物理进度。

W46 GPU约17s且提前于E30完成，没有持续三负载证据。优先双任务的有效产出；第三有限工作自然结束就释放资源，不为显存占用增开PPO/重复不等价诊断。GPU吞吐只有在完整动作/目标质量相同、host代价计全时才有加速意义。

## 8. 后台、记录、发布与结束条件

**所有新后台deadline=null、wall_deadline_enabled=false，省略--deadline-unix。无固定墙钟完成时刻、无到点kill。** 等待/轮询timeout不等于任务deadline；不要把工具等待上限传给后台。原物理30/60/300/600s、有限数值工作量和网络单次timeout保留。按有限任务完成、原结果、明确技术失败或用户停止结束，无无限重试。

存活进程需要暂停时按精确PID处理，可恢复原clock；自然完成的物理失败/task-unmet保留原判定，技术中断单列。不能宽前缀kill、改坏失败结果或新开同配置“直到成功”。

记录每阶段真实commit/build/spec、fixture/source/current参考、是否修复、真实new rollout/FD/physics/PPO、选择依据、job/PID/clock、每级判定。记录设计保留全程紧凑标量、完整500Hz轨迹及关键事件前后高率/完整策略/context；对已生成原件不事后删减，避免在设计上多份重复数组。必要缺项标明never_generated/blocked/interrupted，不造空壳。

紧凑Git先发，授权/请求的已生成原件由D57增量Release直传，核commit/URL/size/digest；已上传资产先查询现存身份，响应丢失按源/远端核验调和，不重复上传相同字节。Git/index短写串行合并最新main，不force push、不覆盖并发结果。

**GitHub核验后不在用户电脑额外下载、打包、镜像或归档；服务器checkpoint/resume、原件、视频、上传副本不自动删除。** D57上传仅用已授权服务器学术代理，禁止电脑VPN SSH转发，不自动切电脑/Clash/直连；凭据不进日志或Git。

最终报告分别说明：计划/实现/运行/控制资格/材料记忆证据/交付。无有用移动候选则以明确有限负结果结束，并给出证据支持的下一机制；不是再次以“更多排查”代替一个完成的工作包。

## 9. 固定学习身份与长期后续

A checkpoint：`outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w44-a/PANN_PPO_S5-seed1/checkpoint-014737.pt`，完整resume `resume-state-ep00428`；B对应`w44-b/PANN_PPO_S5-seed2/checkpoint-014737.pt`，resume `ep00417`。以目录内实际文件后缀为准，保留actor/critic/optimizer/norm/RNG/fork/lineage；tail0、excluded145。共享祖先不能重复记样本。

通过原资格链后冻结控制基线，再另立学习与科学验证phase：增加真正跨足/返程曝光，在同控制器/信息/计算预算下比较RAW/PANN、冻结或打乱历史；让候选各自传播未来材料历史，检查cap之外必要动作权限；之后才推进条件泛化和Go2/NX。W47不自动新增PPO、扩大训练seed、改材料或运行硬件。
