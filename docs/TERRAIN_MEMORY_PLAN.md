# 地面材料记忆主计划：W48结论冻结，W49检验执行与恢复

执行更新（2026-11-13）：D57已实施S5-r37，四个新事件完成4次N solve及36个k引导既有P候选；四事件均有原strict可接纳且动作不同的策略，4×20ms真实RPC部署完成400子步。C完成60 world-step、预热0。原E30尚未启动，后续状态以[W49实际索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w49/phase.json)为准。计划首段“planned”仅为发布时状态。

更新2026-11-13（Asia/Shanghai）。结果基点`61e16975fa6b38dc8e2d31779c2bd82f1aaeb8b6`。**W48正式r36与原评价已完成；W49 planned_not_implemented。** 本次是仓库审查与计划发布，无新服务器任务、native solve、plant、PPO或GPU物理。

[W48最终审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W48-final-review/README.md) · [W49详细计划](S1_W49_STABILITY_PLAN.md) · [启动提示词](S1_W49_START_PROMPT.md) · [执行规则](../AGENTS.md) · [机器交接](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W48-final-review/handoff.json)

## 1. 长期目标与真实成熟度

研究同一世界地块经过前足、后足和返程重复加载后的持久材料状态，使未来支撑需求影响当前加载、落点与重访。最终在原Go2/TM02完整合法往返中建立因果记忆收益，再研究条件泛化、轻量化、Go2/NX实机。

实际链：因果PANN→658D PPO观察→2Hz scalar cap（.18–.25m/s）→逻辑50Hz全身MPC→500Hz时变关节目标→PD60/5→5000Hz原物理。逻辑频率不是墙钟实时。PPO曾真实更新，当前评价权重冻结而observer/材料历史继续因果更新。

当前候选尚未沿每条未来动作完整传播世界材料内态，cap不能直接控制足点、支撑时长或快速关节恢复。稳定原控制、有效跨足/返程曝光、候选内材料历史、因果对照、泛化是相依但不同的研究任务。首先稳定有必要；只站住更久或减少前进不替代原合法任务。

| 阶段 | 保留事实 | 不应外推 |
|---|---|---|
| 早期R1/R2/S0 | 材料辨识、学习正负结果保留 | 不能把早期方向全部写成成功 |
| R8/S3刚地 | 原任务9/9、连续300s 3/3 | 不是当前r36/TM02资格 |
| R9/R10材料 | 完整任务0/8、0/9，R10曾进返程 | 完整材料任务未成立 |
| R11 | 23真实启动，有效E30/E60均0 | 过期或无效资格不复活 |
| W34/W35 | 合法去程分别3与1，有返程进展 | 稳定合法完整往返仍0 |
| W44学习 | A/B各14737/update57、完整resume | 同祖先数据不能重复算独立样本 |
| W45/r34 | A13.072s/B5.582s跌倒 | strict接纳不等于稳定 |
| W46/r35 constant | A26.432s触地、B11.638s跌倒 | 存活更长不等于路线改善 |
| W47 | 定位四元数自差，修复四fixture零端点 | 无新优化或原任务 |
| W48/r36 | 正式修复部署，8solve/4部署及原A/B完成 | 实际轨迹与W46相同，资格无改善 |
| 当前科学资格 | 联合系统运行、数值与事件证据更清楚 | 完整稳定材料任务、记忆因果收益、NX实时未建立 |

A/B共享祖先6545/update25；PANN唯一转移22929，另RAW256，R12 distinct23185。各自W44 checkpoint-014737、resume ep00428/ep00417、tail0/excluded145保留。W45–W48新增PPO0/0，未来15249/59不是W49自动训练预算。

## 2. 最新W48原任务与负结果

| 原E30 | A | B |
|---|---:|---:|
| 正式控制器 | S5-r36 | S5-r36 |
| 结束 |26.432s floor–trunk contact|11.638s body_fall|
| 墙钟 |54.43min|33.63min|
| 最大body x |.626506m|.396364m|
| 实际材料加载 |无|有，主应变最大.00649112＜.02|
| 合法去/返/往返 |0/0/0|0/0/0|
| E30及后级 |失败，后三级blocked|失败，后三级blocked|

A反复退步；B在8.464s已z＜.2m、8.572s已绝对roll＞.5rad，早于首完整足材料加载9.122s。这些为诊断描述，不新增硬门。相邻2ms held最大跳变仍1.535358/1.468642rad，膝极限反转37/5次；90.86Nm是±45.43Nm相邻差，不是越界力矩。

[原位比较](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w48/original-comparison.json)报告两轮time/qpos/qvel/held/torque/GRF/contact/cap/phase逐元素相同，1944calls的选择/alpha/plan相同；chosen_actual A123/B38次不同。报告全数组相同来源是执行端比较，本次未重下载原NPZ；独立核验277事件call相同选择/alpha/plan，支持内部评分改变而实际行为未变。

正式binary SHA `b05c8da8ca74551dc73ea0272329234217d7160731e5cea8e33101bba3241338`与实际进程一致，N/r34 strict/constant正确。不是未部署修复，也不是deadline造成自然物理失败。

## 3. W48真正解决了什么

20组alpha0精确恢复nominal；self/antipode为0，真实微小旋转保留。修复数学上成立，但alpha0本已不安装。四fixture240个old/corrected候选policy数组配对相同，20backward关键数值相同，四个赢家相同；每侧180正候选均预测下降/真实J上升。98个正候选J变化，最大4.6861e−8，远未扭转选择。

四部署核对为nominal/incumbent实际scale的native ActionImpl→ActionImpl，全部精确。没有接受的正步跨ActionIndex→ActionImpl，也没有RPC返回动作参与积分。不能据此排除真实客户端到plant路径问题。

原1944calls中1601正步安装（1598J下降、3风险优先），343停滞。B下沉/侧倾前一秒各50/50正步下降。因此必须区分当前问题的代价下降、策略真实执行、支撑恢复和完整路线，旧困难fixture不能代表全程。

四次零plant启动失败因build替换继承overrides丢17键，r12_substep_path启用却缺r12_trunk_clearance，引起native空指针。当前合并配置已修，binary不变；不重复有限实验。崩溃前部分native积分未知，不能写所有计算0。

## 4. 新记录与尚未闭合的问题

1904非启动边界匹配上一实际部署accepted预测，startup40 unavailable合理。本次独立核237可用事件配对、277事件call与995对象的算术/身份/交付SHA；233条20ms、另2条2ms/2条18ms，不全部叫20ms。完整全调用JSONL仍在D57/Release。

新关键事件A3.846/B3.106首held大跳、A6.860/B6.150首膝反转均有完整policy/accepted/P0/context/reference。首大跳均晚于规划边界6ms。现有数据足以进入机制研究，不再将旧缺快照作为重新跑同配置整轮的理由。

两流0.02s站立已出现小非零预测误差；全程最大关节位置差A/B .0492763/.0164553rad，速度差7.15560/4.30498rad/s。这是闭环预测—执行差，不能直接当纯模型残差。需要分清计划本身已不利于恢复，还是执行偏离后反馈使状态恶化。

显式GetAction请求time为float，SetState time为double，响应action为float。r36工具源码核SHA一致；省略time可用刚同步的native double时钟。**W39已量化时间舍入，W40已有动作FP32；不是新发现，也不是已证唯一根因。** W49先量化新事件作用，作用小就结束该假说，继续恢复线。

W48Records.plan缺父类trunk_restoration触发，下一版补回；已有关键事件不因此失效。完整策略/当前context不等于全部历史MjData、外部callback与optimizer缓存，有限重建必须准确标范围。

## 5. W49冻结工作包

全部详细判据与身份见[W49计划](S1_W49_STABILITY_PLAN.md)。

- CPU A：利用现有实际held链，0积分同态time/action分解；必要四事件×三条20ms，≤1200子步、0solve/FD。相同输入模型对照与同策略执行对照分开，缺hidden/外部全态不伪造，不重开W48。
- CPU B：既有accepted/J分项/接触分析后，≤4新问题N solve，首有效k方向各9正alpha＝≤36候选，必要baseline≤4条。用r33 P既有冻结反馈架构替换手工方向为新问题k，明确双时间网格与完整策略表示，保持原strict；dV/ratio不可用为null，不调mu/rate。不是重新命名旧P或none/load/constant。
- 合并：只选一个有证据的机制；≤4条20ms真实接口部署核对。至少一个代表问题真实可接纳且行为不同，或修正实际影响动作的已证错误，正常入口/约束正确后，自主A/B各一次E30，过则原阶梯。不能要求先整程成功才能实验；也不因极小旧精度变化重复原轨迹。
- C：B158同held双world F1完整初始化、L一步后旧部分reset、F2完整恢复，核心60 GPU world-step，必要预热≤2；条件触发≤8唯一身份native动作对照0physics。不重跑24×350ms，不授予native排序等价。
- 旧零plant技术A/B各一ZIP按原manifest在D57续传；R8已知单科学输入可独立恢复，本轮donor0，均不阻塞主线。

## 6. 并行与最新工作流

W48两原进程实际重叠2030.4135s，约33.84min，双任务已发生；C只有约1.935秒旧数据分析，无新增GPU分配，不能称96GB充分利用。

W49现场核25CPU/96GB实际余量：两CPU线各10，C总host2，约3系统/主控/发布；线程库1、全进程树隔离、可写build隔离，build-j6先完成。原A/B之外不再挂第三重native solve。GPU按实际余量/峰值准入，关闭XLA预占、JAXx64保留，不为占卡新增无效PPO。

**所有新后台deadline=null、wall_deadline_enabled=false，无固定墙钟完成时刻、无到点kill。** 保留有限工作量、原物理阶段、技术失败/用户停止和单次网络timeout；工具等待不是任务期限，无无限重试。对话结束不杀后台、不把running改completed。

紧凑Git先推，必要已生成原件/视频D57增量Release直传核commit/URL/size/digest；回执丢失先查远端，Git短写串行合并最新main、不强推覆盖。**GitHub核验后不在用户电脑额外下载、打包、镜像或归档；服务器原件/checkpoint/resume/视频不自动删除。** 必要服务器单输入恢复不改变该规则。只用已授权服务器网络，不自动借电脑VPN/Clash/SSH转发或未授权直连。

## 7. 文献、交付、覆盖和历史入口

本轮新增Grandia T-RO2023实际触地响应、Elobaid RA-L2025稳定性/角动量约束、Nguyen RA-L2025数据/预测/执行时序、RLOC T-RO2022动作权限；原文/官方代码与适用边界见[文献核对](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W48-final-review/literature.md)。不把外部论文成功率、理论或另一机器人接口直接等同本项目。

本次实时核科学13资产、1,775,245,315B；旧技术两tag404、尚待续传，按hash后续已覆盖坏模型MJB，不能说全部技术原件丢失。R8下载EOF/0bytes是输入可用性问题。全部审查文件不重复保存到用户电脑。

结果基点8750tracked文件、1,044,013,899B完整读流/哈希库存；88变化文件56,762,626B结构检查无错误。两个离线审查脚本实际复算，关键代码/事件/候选/清单已读；字节库存不等于每byte语义理解或全部Release重放。本次没有下载Release payload、登录服务器或新执行物理。

[W43审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W43-final-review/README.md) · [W44审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W44-final-review/README.md) · [W45审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review/README.md) · [W46审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W46-final-review/README.md) · [W47审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W47-final-review/README.md) · [W48历史计划](S1_W48_STABILITY_PLAN.md) · [W48实际phase](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w48/phase.json)

历史“当前/待实施/继续运行”、固定截止和归档规则只属于形成时点。当前执行统一本页、W49详细计划和AGENTS，不重启已结束轮次。
