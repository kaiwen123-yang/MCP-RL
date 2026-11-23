# W57 日志、原始记录与视频独立复核

初始审查基点：`e2d38923a32b0ea4d622001e963789203529de0f`；收尾状态已刷新至 `fd4ede8d46572d638b736d4538f4d9946326265d`，快照时间2026-11-23 19:20:56 Asia/Shanghai。已下载原始数组与视频的身份仍按初始基点的delivery核验，两次提交间这些资产不变。更新补齐发布回执并推进运行中的head计数，物理资格没有变化。以下结论针对已推送的仓库快照；没有登录 D57，没有启动训练或仿真，没有修改任何实验结果。W57 根目录为 `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w57`，下文路径以此为相对根目录。

## 1. 当前可以怎样定义 S1

**已有可用的长时行走 baseline；W57 的增强控制器、训练头和材料记忆机制仍处于验证阶段。** 两者要分开。

历史基线 `S5-r55` 与 `S5-r60` 的 A/B 四条 E600 均在 `../w56/phase.json` 中正式通过，每条 3,000,000 个物理步、600 秒，完成 10 次合法往返、无侧向逃逸。r60 A/B 首次完成时间分别 466.054 / 469.820 秒。这足以支持“已有指定场景内能持续完成任务的基线”。这些是两个固定案例/种子的证明，不等于跨场景、跨种子、扰动与硬件泛化已经证明。

W57 的实际计数由 `index.json` 的 attempt 行独立求和：

| 指标 | 已归档事实 | 解释 |
|---|---:|---|
| 真正物理开始 | 128 | 不是 128 个完整独立资格评估 |
| passed / mechanism_criteria_unmet / running | 124 / 2 / 2 | running 是快照状态 |
| original / short / training | 3 / 2 / 123 | 正式 original 只有 3 条 |
| training passed / running | 121 / 2 | rollout pass 不等于最终冻结策略通过 E60 |
| 完成物理步 | 37,750,000 | 约 7,550 秒物理时间 |
| 进程 CPU | 183,727.384 秒 | 约 51.035 core-hours |
| full head | 8192 已提交 transition、32 updates、finished | 训练预算完成，仍需冻结评估 |
| blind head | 7424 已提交、29 updates，加 A2/B4 部分样本 | 7430 是 distinct observation 总数，6 条尚未进入已提交更新 |

full+blind 已进优化器的不同 transition 总数是 **15,616**；把两个未完成 rollout 的 6 条加入，才是 **15,622**。初始19:01快照是blind7168/28updates加78partial，执行中的状态随后正常前进；两次统计不能混用。旧 W44 actor 的 update 仍为 0，新训练对象是独立 timing head。不能把这写成旧 actor 已通过新一轮 PPO 更新。

当前 original 的三条是 M1-B A E30、M1-B B E30、M1-P A E30。`dispatch/frozen-evaluations.json` 仍等待 final R node；`paired-mechanisms.json` 的 `cases=[]`。因此不应把训练 E60、full head 预算完成、或者 M0 的旧机制尝试写成“最终 M1 B/P/R/PR 与 full/blind 全部比较完成”。

## 2. 步态：站得住、走得完，但软区触地质量仍不理想

`p/gait-f-a-r60-e600.json` 与 B 同名文件把直行刚性区、直行材料区、转弯及静止分开，避免端点站立稀释问题。地形分组基于 trunk x∈[0.8,2.2]，并不声称同一时刻每只脚都处在同一种地面。

| r60 E600 指标 | 刚性直行 | 材料直行 |
|---|---:|---:|
| 步周期中位数 | 0.526–0.528 s | 0.628–0.632 s |
| 步周期 CV | 约 2.5–3.1% | 约 5.8–6.4% |
| FR 首次受力相对计划 touchdown | 约 0 ms | 约 −8 ms |
| FL 同一指标 | 约 0 ms | 约 −4 ms |
| 后足同一指标 | 约 0–2 ms | 约 +14 ms |
| FR 首次触地后 20ms 内峰值的中位数 | 约 43 N | A 103.319 N、B 106.015 N |
| 正机械 CoT | A 0.434946、B 0.435612 | A 0.431272、B 0.431947 |

这里的步周期是同一条腿相邻已提交 liftoff 之间的时间，材料区约 1.59 Hz，与混合全任务后得到约 1.85 Hz 并不矛盾。受力占空比材料区 FR 约 0.548，而 FL 约 0.610、后足约 0.630–0.644，左右/前后不完全对称。

不能从一个正机械 CoT 数值宣称电池耗能低；该数值只积分正关节机械功。也不能把 loaded foot-center path 直接写成真正接触点滑移。旧 E600 gait 文件另有 `loaded_rolling_point_xy_slip_distance_m`，这是已受力球形足的瞬时滚动接触点水平速度积分；材料区 r60 A 为 [0.5045, 0.3558, 0.3587, 0.2776] m，B 为 [0.5090, 0.3633, 0.3631, 0.2815] m，需按相同暴露时长/距离比较。

## 3. 前足再离力已经得到更具体的事件归因

W57 在四条旧 E600 原始记录上做了全量 5kHz 扫描，总计 **12,000,000 行物理记录**。本次重新读取其四份逐事件 `impact-sequence-*.jsonl.gz` 并独立聚合，共 **1047 个选定事件**：r55 A253/B260，r60 A267/B267；FR691、FL302、RR53、RL1，其中材料区1020、刚性区27。

选择范围必须一起报告：闭合的 Fz≤1N 区间至少40ms；若时长>50ms且足底最高>5mm则视为明显高摆腿而排除；初末未闭合区间排除。这不是“所有可能的失接触都没有问题”的证明。

最初单次力脉冲分类为1021早触地反弹、26未决；`resolve_s1_w57_impact_sequences.py` 从已提交 swing apex 重建完整冲击序列后，26个一物理步阈值再穿越也归入相同机制，得到：

- **1047 / 1047 是 early_swing_contact_rebound；选定事件中 confirmed-bearing loss = 0。**
- 冲击发生时 WBC 仍处于 swing，planned stance 为 false，支撑力目标近零。
- 力初次出现比已提交 touchdown 早4.4–13.2ms，中位早8.8ms。
- 初次冲击垂直速度中位约−0.0822m/s，离力时已反向向上，中位约+0.0767m/s。
- 离力持续时间中位76.4ms，最长144.8ms；最高离地中位1.059mm，最大2.511mm。
- 初次冲击峰值中位103.0N，95分位164.9N。

因此应把“前足支撑中又失载”的笼统描述更新为更精确的链条：**摆腿末端提早碰地 → 短促冲击与反弹 → 原计划尚未到 touchdown / 支撑尚未确认 → 重新搜索接触并建立承载。** 已定位到这个可测量的事件机制，但尚未证明更深一层的唯一成因究竟是轨迹末端跟踪误差、材料面预测、执行延迟、接触模式切换还是其组合。

这意味着下一步优先处理触地位置/速度与接触模式的一致性，比仅增加 stance normal gain 更有依据。任何候选改动均应保存同一事件对齐的 reference/actual z、vz、Fz、确认计数、planned-contact、求解力与实际力；不应通过改Fz阈值或重新定义失败来消除问题。

## 4. 对本轮新 P/A/E30 原始数据的独立复算

从 W57 Release 下载 `w57-m1-p-a-original-e30-a01-originals-001.zip`，315,561,014 bytes，与 pinned delivery 的 SHA256 完整一致：

`74061506a1f6e42efc469a705a3a3976ba36a023f32cb90bc075dac1f86e9915`

只解出需要的原始 trajectory、physical、contact-events、manifest 和相关 control chunks。controller 身份为 `S5-w57-p-56b53ddc5b73`，不是旧 r60。独立脚本和完整复算结果在 `recompute_p_a.py` 与 `p-a-e30-independent-metrics.json`。

150000×85条5kHz物理数据全部有限，时间间隔最大误差4.66×10⁻¹⁶s；15001条500Hz轨迹的最小高度0.283007m、最大|roll/pitch|1.370460°、非足接触和非足受力均为0；全部实际关节力矩的最大限额比例54.26%。材料直行8.986s、2.04823m，正机械CoT0.429143；周期FL/RR0.626s、FR/RL0.631s，CV5.21–5.80%；刚性直行周期0.526s、CV1.77–2.17%。这些结果与已发布 mechanics 的同名可比字段吻合。

用同一个物理选择定义独立找到 **20次小 force-off：FL6、FR11、RR3**。逐条连接原始WBC后，20次均为尚未建立承载的早触地反弹。因此 **新P执行仍有同类问题，不能以“新版本跑通了”视为已经修好**。

一个可直接复核的例子：FR在9.6518s开始冲击，比计划touchdown9.664s早12.2ms，峰值167.490N；垂直速度从−0.135604变为+0.121586m/s；9.6554–9.7578s处于Fz≤1N，持续102.4ms，足底反弹到1.793667mm。

| 控制时刻 | mode / planned | 接触确认 / 承载 | 实测与控制响应 |
|---|---|---|---|
| 9.652s | swing / false | false / false | 触碰计数1，WBC法向目标近0 |
| 9.654s | swing / false | false / false | 计数2，观测162.112N，仍不建支撑目标 |
| 9.656s | swing / false | false / false | 实测归零，确认计数清零 |
| 9.666s | search / true | false / false | 搜索速度目标−0.03m/s |
| 9.758s | search / true | false / false | 再受力5.436N，WBC开始给33.936N |
| 9.764s | stance / true | true / false | touch已确认 |
| 9.770s | stance / true | true / true | 承载才建立 |

原始5kHz物理记录与500Hz轨迹的取值位置不同：前者在`before_physics`材料力应用及`mj_step`后写下force-evaluation/post-integration两种时钟；后者在进入这批substeps前记录。不能简单把相邻行Fz强行视为相同时点测量；本次峰值和离力扫描均只用5kHz原始力。

## 5. 新视频真正看了哪些，能支持什么

下载并逐个核验SHA256与大小的7条完整视频：M1-B A E30、M1-P A E30、M0-B A/B short100s、R-full A ep1/A ep10/B ep9。全部1280×720、30fps；E30各901帧、short各3001帧、E60各1801帧。manifest含各资产URL、大小、SHA和ffprobe元数据，位于 `media-sources-and-timecodes.json`。

实际视觉审查范围：每片12个跨全时长的时间码采样帧，共84帧；再检查B/P/R ep10材料区9.0–9.6s共21帧、P/A9.4–9.9s连续16帧及一张原始全画幅，合计122张提取帧（部分重复时刻）。这不是声称逐帧观看了7条视频的全部视频帧。

观察到的宏观行为与记录相符：

- B/P E30开头站立，随后穿过材料区、远端转向、开始回程；最后29.9s仍在材料区回程，尚未回家。两者外观近似，未显示肉眼可确认的大幅步态改善。
- 两条short任务到约80s已回到起点并稳定站立，99s仍保持站姿。没有支持“因100秒截止来不及返回”的迹象。
- 三条R-full E60训练录像约42s已回家，48/54/59.9s为站立。E60后段站立不能作为长时间持续步行的证据。
- 密集截图支持躯干姿态较稳定、摆腿仍在交替；30fps和原镜头尺寸无法直接判定4–13ms提前碰撞与1–2mm反弹，必须依赖原始日志。该限制也是为何不以视频观感取代受力/接触事件分析。

**本次看到的训练视频仍是早中期。** 初始快照核验过的receipt视频清单是full A到ep10、B到ep9；19:20快照补齐了full B ep10的第二个原始ZIP和视频，未继续下载这一新增视频。未取得最终8192/32update冻结head的独立视频，也未取得blind最终视频。因此没有把ep10片段当成最终训练质量证明，更没有用旧r60视频冒充新P/R。

直接逐一读取两次提交的delivery目录：e2共有79个asset entries、17,660,813,035 bytes，与当时index一致；但完整`status=verified`的bundle receipt只覆盖78资产。第79个是full B ep10的第一个ZIP，已记录asset digest，但该bundle还没有完整闭合。这个asset级计数与完整bundle级计数的差异属于发布进行中的快照，不能据此断言文件丢失或数据损坏。最终 `index.json` 已计 **81资产、17,874,794,470 bytes**，full B ep10 receipt为verified、3资产齐全；新增的是`originals-002.zip`（199,254,496B）和`full.mp4`（14,726,939B）。主审另外核实发布元数据EOF读取路径已修复；这不改变本次已核验原始文件的SHA身份。

供主审直接查看的图像：

- `B-P-R-material-cycle-comparison.jpg`
- `P-A-E30-9.4-9.9-contact-detail.jpg`
- `P-A-E30-FR-rebound-9.65s.png` 与同名PDF（原始数据绘图）
- 七份 `*-sheet.jpg` 对应全时长抽样。

## 6. short机制未达标的主因是几何重访，不能只加时长

两条M0-B short都完成12次物理traversal并归位，但A只确认周期[1,3,5,7,9,11]，B确认[1,3,5,7,9,10,11]。12次来回/穿行不等于12次符合材料记忆定义的cross-foot连续面积重叠再加载。

| 指标 | A short | B short |
|---|---:|---:|
| 12次traversal完成 | 64.904s | 65.828s |
| 首次稳定回家 | 79.094s | 79.994s |
| 正式confirmed cycles | 6/12 | 7/12 |
| 正连续面积交叠事件 | 44 | 32 |
| 涉及world cells | 23 | 21 |
| 单个world cell最多跨多少confirmed cycles | 3 | 3 |
| release→reload gap中位数 | 1.2184s | 1.3069s |
| 重载前残余q范数/释放时范数中位数 | 0.2829 | 0.2583 |

`m/revisit-geometry-*.json` 重构原始事件ID和精确footprint，确认集合与正式计分0分歧，previous_load/ownership/clock无错配。A的所有反向traversal没有共同loaded world key，纵向footprint边缘间距15.883–37.528mm，中位25.582mm。B只有第10反向周期真正成功，其他反向大多有间隙。忽略周期规则勉强跨前一traversal配对，会引入约4–22s的时间间隔，仍不能获得合格short重访证据。

`s1_w57_mechanism.py` 在12次traversal后主动归位，而正式criterion要求12个confirmed cycles。因此只把100秒改为更长并不会自动多走、也不会让错开的footprints重合。应先定义期望的footprint/路线几何与正反方向、保留原始任务单独资格，然后在机制任务内把可复现的同一world-cell重访设计清楚；不能把不重叠的接触算成记忆成功。

## 7. P对B的收益仍是混合结果

`p/physical/w57-m1-p-a-original-e30-a01-paired-B.json` 的单个确定性配对：P相对B slip积分−8.925%，indentation积分−4.185%，正功−0.849%；同时路径−1.887%、任务进度−1.616%、峰应变+2.788%，按移动距离归一化的正机械CoT **+1.059%**。不能只摘正功下降来宣称节能，也不能从一个A/E30确定性配对外推稳定优势。

两者E30均没有完成完整往返：outbound=1、return=0、completed_roundtrips=0。mechanics内`completed_original=true`在这里是原E30阶段执行完成的字段，不是完整往返/回家已完成。

## 8. 并行资源核算

最新 `index.json` 资源观测覆盖114633.066s，两个物理槽都在前进103362.811s，比例 **90.168%**。这支持双任务并行确实已经工作，不能再笼统说一直单路。另一个110947.674s是已完成任务生命周期重叠，不能与按计数器前进的同一比例混用。初始19:01快照的同一比例为90.263%，不是不同实验结论。

W57进程GPU显存峰值约2130MiB，主机总峰值31165MiB；主机GPU利用率均值4.7505%，包含其他项目，不能算作本项目独占利用率。这套CPU MuJoCo/OSQP/FP64主路径不会因为显存96GB就自动转为GPU训练吞吐。合理方向是保持两个物理槽，第三槽执行材料候选批量、原始事件分析、视频/数据核验及训练更新；用实际core-hours、物理step/s、候选预测/s、I/O和显存峰值评估第三物理槽是否有净收益。不能为了GPU占用而改物理模型或扩大非必要并发。

## 9. 给下一轮的证据门槛

1. 先收束/识别正在运行的blind rollout与head状态，再冻结final full/blind身份；不可重复开同样训练或把partial当作更新样本（19:20快照为6条）。
2. 保留r60 A/B E600为已证明的baseline，新的M1/P/R/PR各自按身份取得资格，不继承旧通过记录。
3. 接触线的候选修复对准early-touchdown速度/轨迹/模式一致性；同时评估冲击峰、提前时间、反弹时长/高度、支持占空比、period CV、真实接触点滑移和跟踪误差。
4. 机制线先让预定cross-foot重访在几何上确实发生，再比较short与recovered残余状态；记录每个独立world-cell的重复次数，避免把多cell相关事件当独立样本。
5. P/R收益报告使用同距离/同任务完成条件、最终冻结head、相同初始材料与seed的配对，并同时报告进度、CoT、slip/strain和安全指标。
6. 导出最终full/blind冻结视频；视频叠加真实接触状态、foot heights、command/actual speed、phase、case/head身份，使用同视角及关键事件慢放。不同版本不能共用宣传性“已稳定”视频。

## 附：证据层级与复现产物

- `evidence-derived.json`：attempt计数、逐事件归档重聚合、机制几何审计摘录、独立新P原始复算与视频manifest索引。机制几何是引用仓库已执行的精确复核；本轮未重放其全部原始材料场。
- `p-a-e30-independent-metrics.json`：本轮直接读取新P原始物理数组的结果；20个事件都有时刻、速度、力、原始控制状态。
- 对1047个旧E600事件，本轮是逐事件原始分析产物重新聚合和算法审阅，**未重新下载四条600秒全部物理数组**；原仓库产物声明全12M行分析。新P E30的150k物理数组则在本轮直接扫描。
- 7条视频及1份ZIP均SHA256核验；下载总量435,915,926 bytes，仅作为本次分析的临时证据，不在用户电脑重复归档，也没有做任何原始数据删除。


## 最小复现包说明

本目录只纳入报告、脚本、JSON和三张关键图；完整ZIP、原视频与七张全时长概览图没有复制到git。`README.md`给出下载固定SHA资产、重算新P原始数据和重新生成图像的命令。原始物理数组计数为150000行，轨迹为15001行，二者采样频率和首末边界不同；没有150001行的说法。
