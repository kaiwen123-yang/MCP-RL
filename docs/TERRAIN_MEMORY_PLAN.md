# 地面材料记忆主计划：W50有限负结果已审查，W51从起步成对辨识

更新2026-11-15（Asia/Shanghai）。结果基点`09bd55e0cb70244938ad82db7dd202a31ac70f77`。W50/S5-r38/H已实现并发布，原A/B E30未启动。**W51 planned_not_implemented**；本次审查已修复公共runtime两处缩进错误，没有新native solve、物理/GPU积分或PPO。

[W50最终审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W50-final-review/README.md) · [W51详细计划](S1_W51_STABILITY_PLAN.md) · [启动提示词](S1_W51_START_PROMPT.md) · [AGENTS](../AGENTS.md) · [机器交接](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W50-final-review/handoff.json) · [W50实际phase](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w50/phase.json)

## W50实际结果及新结论

四个保存问题76full rollout、36正候选10strict改善；加两次midpoint replan共116full rollout、114candidate records、23112transition FD、4746risk FD。另400bridge子步已消耗，same-SONAME错误库造成ABI失配，native物理等价有效数0；loader改dlmopen后没有新物理验证。四例独立native评分仍有效，两个midpoint输入只可作已记录代数对照。原八阶段blocked/unrun，不是八次失败；新增PPO0/0、GPU物理0。

A237原J下降1.335%，最低z仅提高.149mm、.35s位移−4.297→−18.141mm；B236下降.01144%，最低z仅提高.016mm、位移−18.833→−19.024mm。A328有所抬高仍倒退，B625有9个正alpha候选但均未strict改善。原件存在向前但仍塌低且成本更高的被拒候选，不能说J已错拒正常可行步态。

W50 Task A把旧参考冻结从源码假说变为实测：231个留存policy均继承initial Stand的reference/K/H，参考末端.35s；两流各14825/15001条held查询过期，最终29.65s。private Trot在2.002s正常切换且2Hz相位推进，但早期目标swing仍88–94%受载，四足共载约86%、PD饱和0。W50四例均从4.3s以后取点，尚未检验r38从正常起步连续刷新能否预防。

## 1. 长期目标与真实研究阶段

核心问题：同一世界地块被前足、后足和返程重复加载时，利用材料持久状态，使未来支撑需求影响当前加载、落点和重访。首先建立原Go2/TM02稳定合法完整往返，再建立材料历史的因果决策收益，随后研究泛化、轻量化和Go2/NX实机。

实际链：因果PANN→658D PPO观察→2Hz scalar cap（.18–.25m/s）→逻辑50Hz全身MPC→500Hz时变关节目标→PD60/5→5000Hz原物理。PPO确实曾学习，当前评价固定权重而observer/材料历史继续因果更新。逻辑频率不是墙钟实时。

当前候选仍未沿每条未来动作完整传播世界材料内态；cap不能直接改变足点、支撑时长或快速关节恢复。因此稳定原控制、有效跨足/返程曝光、候选内材料历史、RAW/PANN/冻结或打乱历史因果对照、泛化是相关但不同的研究任务。只生存或原地驻留不能替代合法完整路线；也不能因训练过PPO就宣称记忆收益成立。

| 阶段 | 应保留的事实 | 资格边界 |
|---|---|---|
| 早期R1/R2/S0 | 材料辨识、学习正负结果均保留 | 不把早期方向全部写成成功 |
| R8/S3刚地 | 原任务9/9、连续300s 3/3 | 不转移为当前r38或拟r39材料资格 |
| R9/R10材料 | 完整任务0/8、0/9，R10曾进入返程 | 完整材料任务未成立 |
| R11 |23真实启动，有效E30/E60均0|失效/旧资格不复活|
| W34/W35 |合法去程分别3与1，有返程进展|稳定合法完整往返仍0|
| W44学习 |A/B各14737/update57、完整resume|同祖先数据不重复算独立样本|
| W45/r34 |A13.072s/B5.582s跌倒|strict接纳不等于稳定|
| W46/r35 constant |A26.432s触地、B11.638s跌倒|仍无原E30资格|
| W47/W48/r36 |修复self/antipode与alpha0；原A/B轨迹与W46相同|数学修复成立，行为/资格未改善|
| W49/r37 K |4困难问题均有改善；两原E30均30s自然结束|低位/非足支撑/前进失败，E30 0/2|
| W50/r38 H |4保存问题10/36strict改善；原任务未运行|有限预测仍坍低/后退，bridge物理证据无效|
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

用户随后提供的B视频已查看：[分段审查/关键帧](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/video-review.md)。4.4–5s反复下蹲/顶起，7–8s后持续低伏、腿折叠/外伸；后半贴地挪动，不能称正常步态。W51主目标前移预防起步足部承重/步态失效，低位恢复次级；W50新分析还确认B在5.2–7s曾暂时恢复较高身体且无非足载荷，不能把5s后写成持续趴地。

## 3. W49解决了什么，哪些判断应停止重复

- Task B：4次完整N solve后36个新K候选，25个strict改善，四事件各有赢家；最优J下降3.1761/18.6728/1.9388/33.5849%。108个原N正候选仅3eligible，工作量不等，不能直接当算法成功率统计。
- 4×20ms真实RPC部署400predictor子步；metric/QP与单次FP32 response一致，只证明短窗接口和对象正确，不等于恢复或原任务资格。
- Task A：41实际状态节点、82query，native/历史和RPC/actual-held一致；时间量化最大动作差约.003407rad，限幅后PD差约.2044Nm。W39/W40精度旧线索已量化，不再用极小改动重复旧原轨迹。
- 8×20ms诊断800子步为共同冷启动predictor，缺历史外部全态的A3明确不可用。不能据此将全部误差归某单一物理参数。
- 有限native rollouts196、transitionFD15408、riskFD3618；原A/B合计native rollouts59666、transitionFD11579112、riskFD3048862、plant子步300000；这些全量计数来自执行端提交汇总。W49最终审查当时只独立读了有限数据和231事件call；本轮另实际读取W50发布的完整输入/候选原件。
- 两次query-only失败涉及service ABI/导入ready，一次deployment Init涉及先载sensor-free模型影响XML序列化；均physics0，已修复。两原E30自然满30s后未过，后六级blocked。不要混成八次跌倒或当作deadline问题。
- 正式binary8637e3d118c686831a2acb54ef48271de6092c6095f56c9456f7289607d734e5；有限binary另有身份和修复说明。build的旧N/not_started字段不能覆盖最终runtime/原结果。

## 4. 根因层次：已有证据和仍需辨识的部分

**已证生命周期问题。** r37有效bootstrap后复制旧B、叠加delta/调整scale，reference/K/H一直旧。W50 Task A全量原查询和231事件数组确认reference=[0,.35]，每流98.8267%查询夹在末动作节点174；安装ID/generation_time不能反映反馈实际来源。本次直接读到的四个W50输入完整对象也保留该旧参考。

**实际接触没有跟上目标步态。** 2.002–4.3s A/B四足同时>1N占85.718%/86.736%，目标swing承重约87.811%–94.113%；这早于首非足载荷及材料加载，不能全归材料记忆或PD饱和。B7–30s非足平均45.084N，约体重30.2%，非足>1N占88.447%，视频后期低伏有力日志支持。1N用于描述，不新增任务终止规则。

**r38构造基本接通，但接触模式探索未证解决。** 原bin独立复算raw center≤4.44e−16rad、切空间Jacobian≤2.22e−10、K/H混合逐数组吻合、安装对象吻合。末控制持值与状态插值有很小非严格等价，不足以解释厘米级倒退。新K大不自动代表错误；B625大alpha的总J约152万由ImpactMargin主导，不是NaN。原strict下降是真实局部改善，但不保证正常步态或长任务进展。

**两类技术错误单独处理。** bridge同SONAME不同mjData布局导致actuator字段垃圾，不能用正确FP32 RPC cast掩盖错误物理；下一轮正常plant/独立server入口绕开该bridge，不重开40ms。公共runtime两处dump缩进错误会使整个模块无法解析，本次已修、仅确认AST语法；W50有限脚本不经过它，四例评分不因此作废。

原r26的head reference仍是actual附近局部目标，mission x=3不是其直接残差。没有证据因Position项小就应盲目加权，也不能把W49 25/36和W50 10/36不同问题的候选比例当连续稳定性曲线。接下来区分：从起步刷新效果、运动候选缺失、目标排序及同对象预测—执行偏差。

## 5. W51：一个成对闭环实验，再决定原阶梯

详细实现与执行边界以[W51计划](S1_W51_STABILITY_PLAN.md)为准。D0/D1均使用W44 A/seed1 checkpoint-014737、同归一化/确定性actor、原evaluation t=0完整初始化，各一次连续8s；前2s Stand、随后原outbound。D0=r38/H refresh-only，D1=r38+唯一相位感知完整运动候选S作替代线性化起点（拟r39），不改变物理/成本/步态/horizon/PPO。

需要真实D8入口及evaluate_stage诊断提前返回，沿用正常plant/独立native Agent/RPC/hold与PolicyStream(training=False)。qualification_usable=false；不能用training_duration=8、外部contract覆盖或wall timeout代替，不从不完整历史plant快照拼装。保留原提前物理终止，完整视频覆盖实际运行时段，不补齐或为满8s重跑。

D1每epoch保留原R35Freeze和十nominal得完整全局B。一个S复制B并追加delta_q=q_IK−q_B_joint，raw center保持PD承重偏置/旧feedback/delta。足路径沿原绝对相位、2Hz/.45/C2/.005和速度命令：stance开始世界锚点跨replan保持，当前swing touchdown固定，剩余段从实际足位置/速度连续接续；身体/触地点/原关节限位和原support几何按计划，不加新步高或候选网格。

S仅一native rollout；有效时即使J(S)>J(B)也可作Q，但最终安装必须原strict优于全局B。仅围绕Q一个first-valid backward、原9alpha，沿Q做r38当前时域及K/H转移；不先B9再S9。alpha0保留B，positive趋零不声称等于B，expected/ratio=null。S无效回B，无有用候选仍保留B，不强制刷新年龄。

D0正常epoch通常≤20full rollout、D1≤21，双8s约842calls、最多80000plant子步、约17261full rollout量级，FD另计，实际提前结束更少。它不是W50的120rollout微测试，也非严格同计算预算。比较0–2、2.002–4.3、4.3–8s真实卸载/支撑交换、足滑移/间隙、z/vz/姿态、非足承重与前进，而非只看J或没触发body_fall。

D0已有效且D1无增量时优先简单r38；D1有实质改善时选同一冻结D1；两臂都差或只有贴地挪动则交对照负结果，按候选/排序/执行分辨下一项改变，不现场叠加新成本、步态、horizon、PPO和新求解器。

选定一个正式机制后原A seed1/B seed2各一次E30并行，从原完整evaluation初态开始，明确独立attempt。D8不作为资格或拼接E30。每case原E30过才E60→E300→E600：30s合法材料去程/3m远端原容差/完整观察，60s合法往返和回家稳定2s，300/600为270/540s内5/10合法往返并完整观察。失败只blocked本case后级，不换seed追成功。W44各14737/update57、tail0/excluded145全学习状态保留，新PPO0/0、donor0。

## 6. GPU与并行的实际边界

W50零新GPU物理；用旧F1/L/F2全暴露记录规范化活动DOF/约束/contact后，首步smooth dynamics与约束输入一致，首个可观察差异为contact efc_force→qacc；完整contact字段27/30边界相同，第10步也有分歧。仍未锁定唯一kernel，canonical sort stub不是修复，不重复相同探针或以逐位相同作总门。

W51 C默认零新GPU积分，用已有114候选/113维残差/22项定义实现FP64 GPU成本归约，并逐项列ResidualFn的state/kinematics/context/接触/solver依赖。76条保存问题与38条错误bridge midpoint分开，后者仅代数对照；risk另需trunk_violation/failure元数据，只同问题/同epoch比较。残差→J不等于完整native成本移植，可实现有数据支持的纯运动学/代数部分，缺字段明确列出，不填零或新开原episode补历史档。GPU不正式安装、不阻塞CPU，不为占显存重复batch/PPO/吞吐轮。

W50有限A/B仅重叠11.5159s，C为.8841s CPU分析，无持续利用率证据；W49原A/B79.412min重叠是上一轮事实。实测25CPU/96GB余量按10+10+2+3，线程库1，先-j6构建，隔离进程树/可写build，尊重其他项目。不外挂第三重native求解；C完成自然结束，硬件容量不是无效任务的理由。

W49 integration303维及部分外部状态仍不等于通用完整plant恢复；W44学习resume也不是历史plant checkpoint。当前从原evaluation初态开始，保留引用原件，不增加无用恢复框架或重跑旧档采集。

## 7. 工作流、交付与阅读边界

**新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix，无固定墙钟完成时刻/到点kill。** 8s诊断与30/60/300/600s是物理任务定义；原物理终止/技术错误/用户停止、有限工作量与单次网络timeout保留，无无限重试。对话结束不杀后台，running不写completed。

紧凑Git先推，必要原件和视频D57直传增量Release，核commit/URL/size/digest；回执丢失先查远端。Git短写串行合并最新main、不强推。**GitHub核验后不在用户电脑额外下载、复制、打包、镜像或归档。** 服务器仍被引用的科研输入/checkpoint/resume/视频保留，不自动删除。只用已授权服务器网络，不自动借电脑VPN/Clash/SSH转发。

本次继承历史审查/库存，完整读流40个变化文件40,087,699B并解析16JSON；三并行审查线核控制、实际结果/GPU和论文。W50三个已上传ZIP共464,340,186B，API size/digest与Git manifest匹配，本次实际下载、提取254成员569,226,459B并解码关键完整输入/候选原数组；没有把大型原件复制进Git或另在用户电脑归档。没有额外全量SHA重扫，字节读流/解压不等于每数字已语义理解。

231policy与W49全量物理的结论明确归属执行端Task A；本次直接核四个W50输入完整对象、六组114candidate records、所选policy与接口原NPZ，并复看已有用户B视频分段/关键帧。没有新W50完整视频，没有下载所有历史Release或声称逐帧看完所有历史视频。[阅读清单](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W50-final-review/read-coverage.json) · [本轮远端核验](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W50-final-review/remote-verification.json)

只做有信息量的分析和必要验证，不新增全仓hash/mock/shape/import/smoke或防御性测试轮。本次实质代码修复只有两处runtime缩进，r39/D8/GPU成本迁移尚待W51实施。

## 8. 文献和历史入口

[本轮文献](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W50-final-review/literature.md)读取MuJoCo全身MPC、HOUND接触隐式MPC、HiLQR和DIAL-MPC的一手原文及可核实作者入口。借鉴接触探索、初始轨迹和实际模式对齐；不把论文成功或GPU吞吐套作本项目资格。原动力学FD已是单边，raw-center零物理Jacobian不是同一问题；.35s曾有相关成功，不能仅因小于一步态周期就盲目延长。

[W43审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W43-final-review/README.md) · [W44审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W44-final-review/README.md) · [W45审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review/README.md) · [W46审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W46-final-review/README.md) · [W47审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W47-final-review/README.md) · [W48审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W48-final-review/README.md) · [W49审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/README.md) · [W50历史计划](S1_W50_STABILITY_PLAN.md)

历史“当前/待实施/继续运行”、旧固定截止和归档规则仅属于形成时点；最新执行以本页、W51计划和AGENTS为准。W50已完成，不按旧文字重启。
