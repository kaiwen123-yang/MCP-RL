# 地面材料记忆主计划：W45完成审查，W46先检验移动控制稳定性

## W46 actual implementation and finite evidence — 2026-11-12

S5-r35 is now built from reviewed main a34bc45393da1e552df135f856df6f98c762c51f. The 12 declared finite solves and 36 fixed-policy 350ms native diagnostics are complete; no PPO or original-task physics was added by that comparison. The selected original-task branch is **constant**, with mechanism N and r34 strict acceptance retained. Load and constant selected identical complete policies in all four contexts, so no load-information benefit is supported.

A153 provides limited useful evidence: nominal target jump 1.08418→0.56936 rad and torque jump 65.3074→34.2848 Nm, similarly under both recorded perturbations. Minimum height decreases 1.152mm and J0 rises .0002356; terminal vertical recovery remains positive and small forward displacement increases. The other three contexts are unchanged; B158 still has a bad high-jump incumbent. This supports one original A/B evaluation, not a stability claim.

The common-reference residual reconstruction error is zero; all four disabled-penalty native state/action/residual comparisons are exactly equal to r34. Candidate cost recomputation error is at most 4.45e-16, held-u residual FD error at most 7.96e-11, direct state C zero. Actual evidence, frozen binary/spec identity and the new phase are under [w46](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/). At the 2026-11-12 launch, both original evaluation-only E30 ladders were observed running in tmux s1-w46-a/s1-w46-b with advancing physical clocks (phase.json is the live authority); each keeps its own W44 checkpoint014737/update57, new PPO0/0, no wall deadline.

The independent GPU study has completed 24 old-policy scenarios and 42000 world-steps including the 20ms prefix, plus 48 warmup steps. It uses GPU FP64 feedback algebra and the exact old native H-QP on host, then an explicit cast to MJWarp FP32 dynamics. Same-state held-action error is 2.49e-16rad, but 20ms cross-engine command differences reach 1.501rad; no equivalent ranking or online acceleration is established. Original plant Euler is not re-simulated; the paired dynamics are native implicitfast and MJWarp.

Implementation: [spec](../experiments/terrain_memory/s1-go2-r11-S5-r35.json), [finite selection](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/mechanism-evidence.json), [phase](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/phase.json), [GPU evidence](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/gpu.json). The W45 facts and original bounded protocol below remain historical provenance and operating constraints.


更新：2026-11-12。最新结果基点`3e3f69c440d6a33c14131563e5f485d88da3a803`。**当前实际控制版本r34/W45已自然结束、原任务未通过；下一轮r35/W46为planned_not_implemented。** 本次是源码/已发布原件审查与离线复算，没有启动服务器任务、physics或PPO。

[W45完整审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review/README.md) · [W46详细实施计划](S1_W46_STABILITY_PLAN.md) · [完整启动提示词](S1_W46_START_PROMPT.md) · [执行规则](../AGENTS.md) · [机器交接](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review/handoff.json)

## 1. 长期研究目标与当前成熟度

研究同一世界固定地块在前足、后足及返程重复加载后的可恢复材料状态，让未来支撑需求影响当前加载、落点和重访，最终建立原Go2/TM02合法完整往返中的因果记忆收益、条件泛化与Go2/NX实机证据。

真实链路已建立：冻结因果PANN材料观察器→658D PPO观察→2Hz标量cap(.18–.25m/s)→逻辑50Hz全身MPC→500Hz时变反馈目标角→PD60/5→5000Hz原Go2/TM02物理。PPO曾真实更新；固定评价中网络不更新，observer/material memory仍按观测因果变化。逻辑频率不是墙钟实时资格。

当前候选尚未随各自未来动作完整传播世界材料内部历史，主要使用当前锚点与瞬时接触代理。cap的动作权限也不直接控制足时/落点/卸载。控制稳定、历史曝光、候选内因果记忆、匹配对照与泛化是相依但不同的任务，不能把联合代码跑通当作研究闭环完成。

| 阶段 | 已有事实 | 尚缺 |
|---|---|---|
| 材料与早期方法 | R1/R2/S0、复现与学习正负证据均保留 | 早期方法并非全部成功，不能改写旧结论 |
| 刚地 | R8/S3原硬Go2任务9/9、连续300s 3/3 | 当前r34控制器/材料任务资格 |
| 材料 | W34合法去程3次、W35 1次，有历史返程进展 | 稳定合法完整往返 |
| 联合学习 | A/B各14737/update57，完整恢复链保留 | 充分有效的记忆决策数据与因果收益 |
| W45数值控制 | r34接纳修复有效，新alpha0安装0 | 稳定的完整移动策略与实际任务成功 |
| GPU | 有完整原件和有限对照 | 完整策略语义一致性、同质量端到端收益 |

## 2. W45实际结果与交接点

| 流 | 最后job /固定策略 | 原E30 | 后级 |
|---|---|---|---|
| A | job-20261112T103825Z-w45-a / W44-A update57 | 13.072s body_fall；max x .82517m；极少量材料受力，未贯穿 | E60/300/600 blocked |
| B | job-20261112T103825Z-w45-b / W44-B update57 | 5.582s body_fall；max x .29728m；无材料受力 | E60/300/600 blocked |
| G5 | g5-20261110T033858Z / W40-B update37、r30 | 满30s耗19.366h，最大前移.0000242m，task criteria未满足 | 最终原件已发布且本轮核验，不重跑 |

两W45流合法去/返/往返均0，新增PPO0/0、样本与更新仍14737/57。固定权重为`outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w44-a/PANN_PPO_S5-seed1/checkpoint-014737.pt`及w44-b/seed2同名文件；完整resume分别ep00428/ep00417，tail0、excluded145。共有6545/update25祖先，PANN唯一总量22929，另RAW256，R12 distinct23185，不能把两14737简单相加。

974次planner call全启用r34，新的alpha0安装0；四问题完整candidate/installed数组0差、J复算至机器精度，实际QP failure/restoration0。原物理侧翻不是后台到点终止。A在12.830s已有明显侧倾，13.048s才首次材料受力；39份PANN置零cap对照全部差0。当前首先是完整移动控制不足，尚不是有效的材料记忆收益检验。

G5旧Git入口指向早期技术attempt，本次追加[最终审查指针](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w41-gpu/continuation-final-audit.json)并保留旧结果。111个最终源文件及整包SHA已核验，不再把已发布文件列为服务器待补事项。

## 3. 失稳证据与下一机制选择

A2.650s的native投影后目标相邻2ms变化1.41880rad，actual1.16403；B2.764s为1.23629/1.23587rad。A某膝力矩在正负45.43Nm之间反转。完整native策略自己就含剧烈命令，不能全归因于plant模型误差或alpha0。raw Kδ有表示依赖，不能只裁K或通过重定中心把其数值变小就称成功。

A call153的正alpha真实改善J：高度项减少.05959、Effort增加.03053，其他项有退化；同时最低高度与前移改善。当前目标愿意接受这个权衡。不能无证据删除全局高度项，r14曾靠它修复离开材料带后nominal下落；不能把短暂合法flight改为新失败。现在缺的是在接触变化与真实扰动下可持续的完整移动参考和控制。

一手文献提供两个先后方向：HiLQR在time-stepping下用承载份额调度参考输入代价，但它有稳定walking reference前提；HOUND FDDP同时保持状态/输入并收缩动态缺口，适用于单射击初始化与接触级联问题。当前不把整套solver替换、GPU引擎与RL同时修改；先检验一个有限代价机制，负结果再明确转向可执行moving nominal/可行性热启动。

## 4. W46冻结计划

唯一主机制：在r34上给移动阶段加入**实际held qtarget相对共同冻结输入参考**的代价。每个epoch从当前状态native重评分输入incumbent，冻结u_bar/F_bar/R/context；全部候选按同一新目标重评分。新项进入native residual/cost、held-u导数、Riccati/K/H与archive/接纳，不能仅改Quu正则。当前incumbent未证明稳定，必须检验是否锁住坏动作。

固定四个W45上下文A148/136@2.54、A153/140@2.64、B155/141@2.68、B158/144@2.74，比较r34/constant/load，共12次有限求解。constant匹配load每节点平均强度；R起点从当前Effort/PD单位确定，不扩参数网格。每输出策略从原初态及记录尺度正负偏差各350ms固定native闭环，合计36诊断，不在窗内重规划，不把新variant预测称为actual truth。

选择一个有用分支，两流同版；比较共同J0、实际命令/力矩、恢复/进度、扰动及原硬条件。如果只是驻留、锁坏参考或没有有用变化，交付mechanism_not_supported，不叠加别的架构或重复已知无用E30。若支持，冻结实际r35 spec/build/runtime/records，核白名单/机制N/合同/二进制身份后，A/B固定自身update57直接原E30。

同case前级通过才E60→E300→E600。E30满30s、合法实际材料去程与原远端/转向；E60完整1往返、回家稳定2s；E300/600原270/540s路线期限内5/10往返并完整观察。原Go2/TM02绝对主应变.02、PD/力矩、非足接触/跌倒与路线判据不变。没有新增PPO，PANN权重固定而观察状态继续因果更新。

第三任务独立检验同一完整策略GPU执行，最多24个旧策略scenario：同state action map→20ms→满足语义和原约束判断条件才一次350ms。最多42000完整world-step，短窗/预热另计；明确FP64动作层与FP32 dynamics边界。停止旧reference-proxy top4研究，不重跑G5，不阻塞CPU，不扩大成新仿真引擎项目。

## 5. 并行、后台和交付

默认D57 CPU树`/root/autodl-tmp/MCP-RL-w42`，GPU树`/root/autodl-tmp/MCP-RL`，SSH`autodl-d57`。历史25vCPU/120GB RAM/96GB GPU，开场核现场资源与有效任务。A/B各10workers、线程库各1，GPUhost总2CPU含native确认，约3CPU余量。有限native比较最多两个10CPU进程，结束后才双E30；build-j6先完成，可写upstream/build隔离。PANN/Actor共卡，保留JAX FP64且关闭预占，不以显存满作为目标。

新后台deadline=null，执行层wall_deadline_enabled=false，无固定墙钟完成时刻、无到点kill；原物理时长、有限数值预算与网络timeout保留。存活进程可精确暂停恢复；未完成的技术退出保留interrupted，已自然结束的物理失败/任务未满足保留原结果。仅Actor无法恢复完整plant中途状态，必要续跑另建独立attempt，不拼接资格。

实际实现/spec/有限证据/真实phase先推Git，再启动并确认物理时钟和记录增长；结果紧凑Git先发、请求原件增量Release核size/digest，短写串行合并并发main，不强推。GitHub核验后不在用户电脑另下载、打包或归档，不自动删除服务器checkpoint/resume/原件/视频。D57上传仅已授权学术代理，禁止电脑VPN SSH转发。具体执行参数与保存要求以W46详细计划为准。

## 6. 稳定之后再推进记忆研究

通过原资格链后冻结控制基线，另立学习phase；各+512/2至15249/59只是未来候选预算，W46不自动执行。随后增加真实跨足/返程历史曝光，在同控制器/信息/计算预算下做RAW/PANN/冻结或打乱历史因果对照，补足候选内材料历史传播和必要动作权限，再推进条件泛化与Go2/NX。

本次全面库存覆盖8,548个tracked文件、约1.016GB字节，关键原件与源码有独立哈希和数值重算。字节/结构完整读取不等于逐字节语义理解或全部历史Release重放；具体范围在[审查目录](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review/README.md)。原任务未通过仍如实报告；计划发布不等于机制已验证。

## 历史入口

[W43审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W43-final-review/README.md) · [W44审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W44-final-review/README.md) · [W45历史实施计划](S1_W45_STABILITY_PLAN.md) · [W45实际phase](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w45/phase.json)

历史文件中的“当前/尚未实施/继续运行”、旧截止或归档规则只适用于形成时点。下一轮统一按本页、W46详细计划与AGENTS执行，不根据陈旧文字重启已结束配置。
