# 地面材料记忆主计划：W49已结束，W50刷新滚动反馈参考

更新2026-11-15（Asia/Shanghai）。结果基点`de6606dc8363e8ab1c816fa682cace752a06318a`。**W49/S5-r37/K原A/B都观察满30s，但E30 0/2、合法往返0；低姿态/非足支撑与前进失败仍在。W50 planned_not_implemented。** 本次为仓库/文献/交付审查，无新服务器任务、native solve、plant/GPU积分或PPO。

[W49最终审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/README.md) · [W50详细计划](S1_W50_STABILITY_PLAN.md) · [启动提示词](S1_W50_START_PROMPT.md) · [AGENTS](../AGENTS.md) · [机器交接](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/handoff.json) · [W49实际phase](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w49/phase.json)

## 1. 长期目标与真实研究阶段

核心问题：同一世界地块被前足、后足和返程重复加载时，利用材料持久状态，使未来支撑需求影响当前加载、落点和重访。首先建立原Go2/TM02稳定合法完整往返，再建立材料历史的因果决策收益，随后研究泛化、轻量化和Go2/NX实机。

实际链：因果PANN→658D PPO观察→2Hz scalar cap（.18–.25m/s）→逻辑50Hz全身MPC→500Hz时变关节目标→PD60/5→5000Hz原物理。PPO确实曾学习，当前评价固定权重而observer/材料历史继续因果更新。逻辑频率不是墙钟实时。

当前候选仍未沿每条未来动作完整传播世界材料内态；cap不能直接改变足点、支撑时长或快速关节恢复。因此稳定原控制、有效跨足/返程曝光、候选内材料历史、RAW/PANN/冻结或打乱历史因果对照、泛化是相关但不同的研究任务。只生存或原地驻留不能替代合法完整路线；也不能因训练过PPO就宣称记忆收益成立。

| 阶段 | 应保留的事实 | 资格边界 |
|---|---|---|
| 早期R1/R2/S0 | 材料辨识、学习正负结果均保留 | 不把早期方向全部写成成功 |
| R8/S3刚地 | 原任务9/9、连续300s 3/3 | 不转移为当前r37材料资格 |
| R9/R10材料 | 完整任务0/8、0/9，R10曾进入返程 | 完整材料任务未成立 |
| R11 |23真实启动，有效E30/E60均0|失效/旧资格不复活|
| W34/W35 |合法去程分别3与1，有返程进展|稳定合法完整往返仍0|
| W44学习 |A/B各14737/update57、完整resume|同祖先数据不重复算独立样本|
| W45/r34 |A13.072s/B5.582s跌倒|strict接纳不等于稳定|
| W46/r35 constant |A26.432s触地、B11.638s跌倒|仍无原E30资格|
| W47/W48/r36 |修复self/antipode与alpha0；原A/B轨迹与W46相同|数学修复成立，行为/资格未改善|
| W49/r37 K |4困难问题均有改善；两原E30均30s自然结束|低位/非足支撑/前进失败，E30 0/2|
| 当前总体 |联合系统、接口与事件证据更完整|完整稳定材料任务、因果记忆收益、NX实时未建立|

A/B共享祖先6545/update25；PANN唯一转移22929，另RAW256，R12 distinct23185。各自W44 checkpoint-014737、resume ep00428/ep00417、tail0/excluded145保留。W45–W49新增PPO0/0，W50也0/0；15249/59不是本轮自动训练预算。

## 2. 最新W49原任务应怎样理解

| E30 | A | B |
|---|---:|---:|
| 结束 |30s material_traversal_not_completed|30s material_traversal_not_completed|
| 墙钟 |101.68min|79.22min|
| 最大body x |.349667m|1.020530m|
| 合法路线比例 |5.81745%|17.01414%|
| 最低/最终z |.123298/.138718m|.130666/.142818m|
| 首非足载荷 |4.6232s|5.0216s|
| 足端首材料载荷 |无|12.1866s|
| 机体west entry |无|20.064s|
| 合法去/返/往返 |0/0/0|0/0/0|
| 后级 |E60/300/600未运行|E60/300/600未运行|

原body_fall为z<.12m或|roll/pitch|>1.2rad；模型允许腿部部分几何接地。A最低仅高出高度终止线3.30mm，所以没有提前终止并不等于正常承重。A6.238s命令不前进2s事件已有四足contact=true，真实位移不足1cm；B4.760s已trunk_restoration。A没到材料就出问题，B异常也早于首材料加载，不能把根因全部归给材料/PANN。

A末vx约.0000198m/s、命令.2128；B末vx.000544m/s、命令.2179。A最大前进量较W48反而下降，B增加。两个固定seed不提供泛化稳定性结论。两次独立30s不可拼接成一个60s成功。

[结果审查与独立聚合](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/results/review.md)列出真实事件切片及未嵌入Git的整秒边界，不对缺失时间点插值。

用户随后提供的B视频已查看：[分段审查/关键帧](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/video-review.md)。4.4–5s反复下蹲/顶起，7–8s后持续低伏、腿折叠/外伸；后半贴地挪动，不能称正常步态。W50主目标前移预防起步足部承重/步态失效，低位恢复次级。

## 3. W49解决了什么，哪些判断应停止重复

- Task B：4次完整N solve后36个新K候选，25个strict改善，四事件各有赢家；最优J下降3.1761/18.6728/1.9388/33.5849%。108个原N正候选仅3eligible，工作量不等，不能直接当算法成功率统计。
- 4×20ms真实RPC部署400predictor子步；metric/QP与单次FP32 response一致，只证明短窗接口和对象正确，不等于恢复或原任务资格。
- Task A：41实际状态节点、82query，native/历史和RPC/actual-held一致；时间量化最大动作差约.003407rad，限幅后PD差约.2044Nm。W39/W40精度旧线索已量化，不再用极小改动重复旧原轨迹。
- 8×20ms诊断800子步为共同冷启动predictor，缺历史外部全态的A3明确不可用。不能据此将全部误差归某单一物理参数。
- 有限native rollouts196、transitionFD15408、riskFD3618；原A/B合计native rollouts59666、transitionFD11579112、riskFD3048862、plant子步300000；这些全量计数来自执行端提交汇总。本次只独立读了有限数据和231事件call。
- 两次query-only失败涉及service ABI/导入ready，一次deployment Init涉及先载sensor-free模型影响XML序列化；均physics0，已修复。两原E30自然满30s后未过，后六级blocked。不要混成八次跌倒或当作deadline问题。
- 正式binary8637e3d118c686831a2acb54ef48271de6092c6095f56c9456f7289607d734e5；有限binary另有身份和修复说明。build的旧N/not_started字段不能覆盖最终runtime/原结果。

## 4. 本轮最关键的新机制问题

r37 online K在有效bootstrap后复制旧完整B，只叠加当前k曲线；nominal只调整旧feedback scale；reference/K/H持续继承。原reference时间窗过去后，查询沿旧末端插值，虽然新的delta和完整策略ID不断产生。

这是源码确定的生命周期限制，已读事件与之相符。实际K/H数组、reference末时刻和过期时长还需在D57现有NPZ验证；约29.65s是当前源码推断，不能写成逐数组实测或唯一失败原因。`policy_age_s`、`generation_time`、`metric_source_plan_id`实际反映安装身份，不足以表示反馈来源。

A/B各1521calls，正步安装802/1393且全部同问题J下降；末支撑高度项仍占97.2686%/97.1754%。所以控制器仍在大支撑缺口的状态附近找到局部小改进。`internal_stagnation=0`只是没命中旧枚举，A尾部仍backward_failed。

r26每次以actual x锚定局部head reference，.215m/s匀速示例前视约.05375→.129m。mission x=3不等于native Position误差；小Position项不能直接证明该增大权重。先看滚动参考/反馈是否真正更新，再用可执行候选区分搜索、目标排序、支撑交换权限和预测—执行差。

## 5. W50主计划与原评价推进

唯一主候选：在当前native rollout重新建立reference，旧完整policy的raw center记为c_B、当前切空间Jacobian记为L_B；用同一个原alpha构造u_ff=c_B+alpha*k_N，K=(1-alpha)L_B+alpha*K_N，H=(1-alpha)H_B+alpha*H_N。最终scale1、旧delta已吸入center再清空；alpha0直接deep-copy旧B，不rebase，不假称正alpha全局恒等。首有效backward同时capture k/K/H/μ/active，不能首k配末K/H。原风险/J/roundoff/strict与赢家原对象部署保持。

具体公式、坐标和来源、配置/ABI接入以[W50详细计划](S1_W50_STABILITY_PLAN.md)为准。该候选是待证的局部重参数化，不冒称论文保证或已经修好。

A零新积分从2.002s Stand→outbound读现有原件，核真实private gait/phase、脚参考、Fz与接触交换，与B编码并行。四个W49例：A237@4.32、B236@4.30、A328@6.14、B625@12.08；各一次首有效方向、9候选。四例含nominal/必要baseline≤80full rollout；最多两条40msRPC部署在20ms各正常再规划一次，≤2extra solve/40rollout/400子步；FD与零物理坐标求值另列。只用原一次性导入后正常online入口，不让query-only和旧time/gait残留。

以A237/B236早期例为首要依据，至少一例有原strict认可的实质动作/足部承重或有效前进变化，或已证早期步态错误被修复，不能仅凭晚期低伏挪动更快，实际入口/生命周期/原约束正确后，可自主冻结同一个正式版本并行原A/B各一次E30。不要求诊断先完成全程或两流都恢复，也不等GPU/donor。没有实质改动则交有限负结果，不重跑相同原轨迹。

每case E30通过才E60→E300→E600；原30s材料去程/远端到达/完整观察，60s合法往返与回家稳定2s，300/600分别270/540s内5/10合法往返和完整观察。失败只blocked本case后级；固定seed/权重/版本，不追成功重跑或拼接资格。无新增PPO、成本/步态/horizon网格或整个新求解器移植。

## 6. 记录能力、GPU与并行

W49保存integration303维、geom masks及部分外部数值态，但未完整序列化history/legal_route/部分对象引用、native caches、actor/RNG，也没有通用restore。它明显改善了诊断能力，仍非任意历史plant中途恢复。缺字段明示，不填零、不另开原episode补记录。

GPU F1/L/F2完成60world-step、warmup0，完整可见状态恢复也在首.2ms有浮点差。首步ctrl/actuator/passive/外力一致，contact排序字段十步一致，constraint/qacc首步不同。已有1873543B全暴露NPZ可零新积分沿constraint/solver/reduction定位；不反复重测相同2ms或把bitwise差自动认定不可用。full native cost未实现，GPU仍不正式排序。

W49原A/B实际并行79.412min；C只有3.51s且在原A/B前结束，不能称96GB持续满用。继续实测25CPU/96GB余量，10+10+2+3分栏、线程库1、build-j6先完；两CPU重求解之外不挂第三重native任务。C独立读旧数据/源码，必要有具体修复后仅一次≤20GPU step含预热；可做有目的的现有矩阵GPU批分析。GPU proposal/native评分是后续架构方向，不为占显存添加无效PPO或新控制器。

## 7. 工作流、交付与阅读边界

**新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix，无固定墙钟完成时刻、无到点kill。** 有限工作量、原物理阶段、技术失败/用户停止和单次网络timeout保留；工具等待不成为任务期限，没有无限重试。对话结束不杀后台，不把running改completed。

紧凑Git先推，必要原件/视频D57增量Release直传，核commit/URL/size/digest，回执丢失先查远端。Git短写串行合并最新main、不强推。**GitHub核验后不在用户电脑额外下载、复制、打包、镜像或归档；服务器原件/checkpoint/resume/视频不自动删除。** 只用已授权服务器网络，不自动借电脑VPN/Clash/SSH转发。

本次live核16资产1,658,430,730B：W49有限4ZIP、A3、B3、四视频、W48两技术ZIP均uploaded。A交付恢复已完成；六个有Git预期size/digest的资产匹配，十个W49ZIP本次未下载复算，仅核API与manifest/收据。[核验范围](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/remote-verification.json)

基点8817tracked files、1,055,255,013B。继承旧全仓库存与历史审查，本轮64变化49,031,786B完整读流/41JSON解析/23文本读取，三线与主审深入核代码和关键证据；231个Git事件call不等于外部3042全日志。没有重下载NPZ/视频或运行服务器实验；用户直接提供的B视频已分段/加密抽帧查看，不宣称每byte或全部帧语义已理解。只做有目的的分析和必要验证，不追加全仓hash/mock/smoke/导入形状等防御性测试。

## 8. 文献和历史入口

[本轮文献](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/literature.md)新增BiConMP T-RO2023、CAFE-MPC T-RO2025、Bilevel CDC2024、Layered MPC ACC2025的一手原文/作者代码，重点是滚动时域、参考更新、可行接触和执行反馈，不将外部成功/理论套成本项目保证。

[W43审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W43-final-review/README.md) · [W44审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W44-final-review/README.md) · [W45审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review/README.md) · [W46审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W46-final-review/README.md) · [W47审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W47-final-review/README.md) · [W48审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W48-final-review/README.md) · [W49历史计划](S1_W49_STABILITY_PLAN.md)

历史“当前/待实施/继续运行”、旧固定截止和归档规则仅属于形成时点；最新执行以本页、W50计划和AGENTS为准。W49已结束，不按旧文字重启。
