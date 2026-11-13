# 地面材料记忆主计划：W46未通过，W47先恢复有用的移动搜索

<!-- W47_EXECUTION_BEGIN -->
> 执行更新（2026-11-13）：W47 原位分析、B 来源核验与 C 全量复算已完成；隔离诊断构建已完成，A 的 8 条恒等修复配对验证待启动。 唯一 R8 原件在 D57 不可用，最新 W46 A/B 所需完整策略与同时间预测缺失；B 未启动 donor solve，未把历史 fallback 当最新双流覆盖。1944 次原调用中有 1601 次正步安装（1598 次 J 下降、3 次风险优先恢复）与 343 次内部停滞。N/r34 strict 保留，新 PPO 与新原 plant 回合均为 0；本轮诊断构建不冒称已部署 S5-r36。以下冻结计划和 W46 审查按其形成时点阅读，当前执行状态以索引为准。 [W47 实际运行索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w47/phase.json)。
<!-- W47_EXECUTION_END -->

更新2026-11-13。结果基点`15ee620b8a909246a12baf999c136c2be299655b`。**W46/r35 constant已自然结束并交付；W47/r36仍为planned_not_implemented。** 本次是审查和下一计划发布，未启动服务器任务、原plant实验、MPC求解或PPO。

[W46最终审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W46-final-review/README.md) · [W47详细计划](S1_W47_STABILITY_PLAN.md) · [完整启动提示词](S1_W47_START_PROMPT.md) · [执行规则](../AGENTS.md) · [机器交接](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W46-final-review/handoff.json)

## 1. 长期目标与实际成熟度

研究同一世界地块在前足、后足、返程重复加载后的可恢复材料状态，使未来支撑需求影响当前加载、落点与重访；最终建立原Go2/TM02合法完整往返中的因果记忆收益、条件泛化与Go2/NX实机证据。

真实联合链已存在：冻结因果PANN材料观察器→658D PPO观察→2Hz标量cap(.18–.25m/s)→逻辑50Hz全身MPC→500Hz时变目标→PD60/5→5000Hz原物理。PPO曾真实更新；评价中权重固定而observer/history继续更新。逻辑频率不是墙钟实时。

当前仍处在材料任务稳定控制基线建立阶段。候选尚未沿各自未来动作完整传播世界材料内部历史，cap也不能直接控制足时/落点/卸载；稳定、有效历史曝光、候选内记忆、因果对照和泛化是相依而不同的任务。

| 层次 | 已有证据 | 尚缺 |
|---|---|---|
| 早期材料/学习 | R1/R2/S0及正负实验保留 | 不能把所有早期方向改写为成功 |
| 刚地 | R8/S3原任务9/9，连续300s 3/3 | 当前r35/TM02资格 |
| 材料 | W34合法去程3次、W35 1次，有返程进展 | 稳定合法完整往返 |
| 联合学习 | A/B各14737/update57，完整恢复链 | 有效材料历史决策数据和因果收益 |
| 数值控制 | r34严格接纳保留；r35参考/成本/导数一致性有证据 | 新方向下降、稳定移动与恢复 |
| GPU | 完整旧策略动作层和有限配对实验 | 同质量完整成本排序、端到端收益 |

## 2. W46真实结果

| 流 | 固定策略 | 原E30结果 | 材料/路线 |
|---|---|---|---|
| A | W44-A seed1/update57 |26.432s floor–trunk contact；wall65.59min；max x .626506m|无实际材料加载；合法去/返/往返0|
| B | W44-B seed2/update57 |11.638s body_fall；wall40.42min；max x .396364m|有材料加载，应变.00649112<.02；合法去/返/往返0|

E30通过0/2，E60/300/600六项blocked。W45对应13.072/5.582s；W46时长增加，但A前进下降且反复退步。B明显下沉/侧倾先于500Hz首次完整足材料受载。相邻2ms目标最大变化1.53536/1.46864rad，相邻力矩反转90.86Nm仍存在，不等于实际越界施加该力矩。

A/B各固定自身checkpoint-014737，完整resume ep00428/ep00417，tail0、excluded145，新PPO0/0。共同祖先6545/update25，PANN唯一22929，另RAW256，R12 distinct23185；不把两14737简单相加。

全程1944calls启用N/r34 strict、新alpha0安装0为仓库记录；本次没有重新逐个读取所有实际call原件。96资产/19,983,994,745B在三处Release完成，远端metadata已复核；交付完成不代表任务成功。

## 3. 目前最需要处理的具体问题

四个保存context×三个分支共12solve，60次backward全部完成；540个正alpha候选没有rollout/QP失败、restoration或trunk violation，却全部J增加。最小增加.002914～.105996，比较容差仅约4.2e−11～6.2e−11，不能通过放宽r34制造成功。

A153在none→constant/load时只把旧nominal scale .00237137改为0，其余三处最终policy不变。350ms目标/力矩跳变减小，但前移仍仅毫米级；前20ms力矩并未相同程度改善，统计窗口必须区分。constant/load最终同policy，权重和Jref确实不同，载荷调度的信息收益没有成立。

r35惩罚u对共同参考ubar的偏离，不惩罚相邻u变化；坏incumbent跟随自身仍可Jref=0。none分支也停滞，不能把原因全部归给r35。B155/B158 alpha0的J差明显，须定位首分叉。标准iLQR在同一动态可行nominal上应由alpha0再生轨迹；新K/H离轨映射不同并不是端点跳变的充分解释。

当前源码/轨迹支持先查搜索一致性与可执行移动候选；尚未证明高度项、FP32、材料应变、PPO不足或后台时限是唯一原因。保留原物理、步态、判据和已验证修复。

## 4. W47有限工作与决策

| 工作 | 具体内容 | 新增计算上限 |
|---|---|---|
| A | 原位candidate首分叉/方向分析；只修已证执行错误 | 缺字段才16条固定native rollout；有修复再最多8条配对验证 |
| B | 一个固定R8输入种子、当前H-QP/真实x0/共同J；四fixture两家庭 |8次有限solve＋24条350ms诊断；另≤700 donor held单步评估 |
| C | 原位1944calls聚合；GPU raw别名/时标/门槛/计数修正 |主体0physics；可选单场景三引擎首hold总≤30真实子步含预热 |

donor在线只在去/返moving的current N搜索停滞且固定源合法时追加一个家庭，每epoch最多两solve、独立数值状态、共同参考/联合裁决；有限与在线同入口。R8 mission phase须与重建gait phase分开；最新W46上下文或匹配扰动缺失时，历史fallback仅诊断，不满足原E30准入。A/B的原位分析与输入准备并行；A若需执行语义修复，正式B solve等共享baseline冻结。使用current incumbent生成同一ubar/F/R，不给donor自己的免费参考；保留所有已确认baseline联合比较。R8旧x仅作相位/初差/defect诊断，当前single shooting会重滚覆盖，不能称已实现FDDP。

R8的3cm步态/旧predictor与当前.005m/10×.2ms native不同，固定一个来源与片段规则，不改当前任务、不搜有利seed。有用候选被提出、重滚毁坏运动、J0/Jref排斥运动分别导向不同研究，不把所有问题都归成“该换FDDP”。新增Box-FDDP2022、MS-DDP2023和结构化SQP2025的一手依据与适配边界见审查文献。

只有对两个流代表context都有明确的移动/恢复与扰动证据，才选一个真实r36进入一次原A/B E30；每case自身通过再E60→E300→E600。只有数值修复/更温和但驻留则有限负结果结束，不重开已知配置，不扩网格、不加PPO。新实现/spec/有限证据/phase先推Git，启动后核实际clock。

## 5. 原资格与并行工作流

原E30满30s合法实际材料去程及远端/转向；E60满60s一合法往返、回家稳定2s；E300/600按270/540s内5/10合法往返并观察满时长及原稳定条件。Go2/TM02质量、PD/力矩、绝对主应变.02、非足接触/跌倒与路线不放宽。完整足迹实际受载才计材料，不把驻留、成本下降或程序exit0当成功。

默认D57两CPU任务各10额度、第三host总2、约3给系统/主控/发布；数值库各1线程、完整进程树隔离、build-j6先完成、可写upstream/build隔离。现场核可用25CPU/96GB GPU与无关进程；PANN/Actor共卡，关闭XLA预占、JAXx64保留，显存按实测剩余/峰值准入。GPU支线没有有用新问题就结束，不为填满显存另开PPO。

全部新后台deadline=null、wall_deadline_enabled=false、无固定墙钟完成时刻/到点kill；工具等待timeout不变成任务deadline。物理阶段时长、有限工作量、技术失败/用户停止与单次网络timeout保留，无无限重试。存活进程可精确暂停恢复；仅Actor不能恢复完整物理中途状态，不拼接资格。

紧凑Git先发、原件从D57增量Release直传核size/digest；短写合并最新main，不强推/覆盖并发结果。GitHub核验后不在电脑额外下载、打包、镜像或归档，服务器原件/checkpoint/resume/视频不自动删除。D57只用授权学术代理，不自动切电脑VPN/Clash/直连。下轮原件诊断在服务器现有文件原位完成。

## 6. 稳定之后的研究路径

冻结真正通过原资格的控制基线后，另立学习phase；未来15249/59不是W47自动预算。增加跨足/返程有效历史曝光；在相同控制器/信息/计算预算下做RAW/PANN与冻结/打乱历史因果对照；补足候选内材料历史传播及所需动作权限；再研究条件泛化、轻量化、Go2/NX。

本次库存读全部8,611个tracked文件、1,021,007,506B，47个变化文件完整结构检查；关键源码重建、A/B轨迹和部分GPU输入有独立核验。字节覆盖、compact聚合和原数组复算的区别及审批造成的限制见审查，不能承诺每个历史Release已重放。

## 历史入口

[W43审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W43-final-review/README.md) · [W44审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W44-final-review/README.md) · [W45审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review/README.md) · [W46历史计划](S1_W46_STABILITY_PLAN.md) · [W46 final phase](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/phase.json)

历史文件里的“当前/尚未实施/继续运行”、旧固定截止和电脑归档只属于形成时点。当前执行统一本页、W47计划与AGENTS，不据旧文重启已结束任务。
