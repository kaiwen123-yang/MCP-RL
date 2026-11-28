# W59 步态、机制机会与后续 B 线只读审查

审查快照：`c8378beae4be371eb4f104782b75e5d295b6f59a`。审查日期：2026-11-28。本文未修改仓库、未训练、未推送。配套精确数据：`evidence_gait.json`；E600 原始收据摘录：`e600_receipt_summary.json`。

## 1. 可以冻结什么，不能把什么合并称为“稳定基线”

W59 已终态，400 attempts，active=[]。作为下一轮研究的回归基线可以冻结；没有必要继续修改 W59 来追求一个笼统的“更稳定”。但必须冻结多个有明确身份的参照：

|控制器身份|A 原任务资格|B 原任务资格|解释|
|---|---:|---:|---|
|名义 B3|E60|E60|对照基线，不能把5901的长测成绩转记给B3|
|Fixed-action|E60|E60|固定事件动作对照|
|PPO-full fallback 5901|E60|E600|唯一取得600s完整长窗的冻结候选；长窗限定B|
|PPO-full fallback 5902|E60|E60|第二个冻结种子，但未形成第二个E600复现|
|5903|无成功冻结头|无成功冻结头|不能宣称三种子训练均成功|

原因是 index.physical.qualified 按底层 arm B3 聚合，PPO 外层也运行于 B3，会把不同controller混在同一臂名下。应使用 attempt/controller/checkpoint/source_commit 的完整身份。5901 frozen checkpoint 为 `a870c1550f3dae64016f2a9f1014951b638793978f6b2f4260c9dfb5a16a90e2`，5902 为 `0073814da158722078464fdd728bb0e2585dc2d8cc472042cc521169b7665e9c`。

完整性来源：[outcome.json](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/outcome.json)，[index.json](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/index.json)。

## 2. E600 不是600秒持续行走或600秒持续PPO加动作：原始收据已核

episode：`w59-ppo-full-fallback-5901-b-original-e600-a01`；controller：`S5-w59-research-a524ade8e770798e46`；运行source_commit：`4ad3a5f28b99ad9ccc84455915fdffa9dfcb57af`。

从Release下载 `originals-021.zip`（19,168,406字节），本地完整SHA256等于delivery登记的 `ed33470316fbf4eefe013b39245a183cd6476dcc04417901379b65ab7f41c1d2`。直接读取receipt、learner-result、actor-events，而非从时长倒推：

- 初始事件动作开始于 **2.002 s**。
- 最后一趟 home_hold 开始于 **469.408 s**。
- **first_completion_s=471.44400001340324 s**，complete 的 lap=10；10次合法去程和10次合法回程均完成。
- actor-events 共1534条。末条request1534的start=469.228 s、end=471.444 s，task_complete=true、terminated=true、end_kind=success。其余1533条为ongoing；没有471.444之后的新W59事件动作记录。
- learner-result task_absorbed=true；完成之后 **128.555999979 s** 是底层继续运行的终端保持观察。该段all_criteria_fraction=1，0 interruptions，0 violation samples。
- 顶层 final_stable_s=130.556 s 是含完成前两秒的连续稳定时长；并非完成timestamp。用600减130.556会把实际完成时间错算2秒。
- hold最大线速度0.00458543 m/s、角速度0.00387355 rad/s、距起点0.01745574 m；这些是终端保持指标，不是运动段速度/扰动恢复指标。
- 全路径62.43849 m；平移路径59.29211 m；moving positive CoT=.43399911；moving actual-contact-point slip/m=.02877294。最小机身高.28300733 m；最大绝对roll/pitch1.556954°。
- 模拟物理步3,000,000，sim599.99999999 s；wall15,158.78 s。此长测只支持该冻结组合、B工况、既定路线的长窗完成与终端保持，不能外推实机实时性或材料分布外鲁棒性。

原始证据：[已核ZIP](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-w59/w59-ppo-full-fallback-5901-b-original-e600-a01-originals-021.zip)，内部 `receipt/...e600-a01.json`、`episode/w59-learner-result.json`、`episode/w59-actor-events.jsonl`。

## 3. 步态是否改善：稳定与质量必须分别说

这些指标来自新的5kHz实际接触点滑移算法；不要拿 `gait/summary.json` 的W57/G1旧分析冒充W59总表。原任务 E60 同case对照如下：

|身份|A moving CoT|A slip/m|B moving CoT|B slip/m|
|---|---:|---:|---:|---:|
|B3|.43174644|.02872500|.43383532|.02942937|
|Fixed-action|.42403497|.02738725|.42892602|.02882168|
|5901|.43415698|.02925733|.43036732|.02892178|
|5902|.42796126|.02794072|.43318606|.03035826|

|候选相对同case B3|滑移相对变化|CoT相对变化|
|---|---:|---:|
|5901 A E60|+1.8532%|+0.5583%|
|5901 B E60|−1.7248%|−0.7994%|
|5902 A E60|−2.7303%|−0.8767%|
|5902 B E60|+3.1563%|−0.1497%|

结果没有支持“PPO已在所有材料情景显著改善步态”。5901A甚至两个成本都恶化，因而没有取得该case长测资格；这与摔倒不同。5902某case没有长测也不能自动解释为长测失败，发布展示种子规则优先5901。

相位统计提供了更具体的研究问题。E600 5901B：

|实际阶段|已承诺leg swings|周期中位数|周期CV范围|首次撞击后、建立承重前累计off均值|首次20ms峰值力p95|
|---|---:|---:|---:|---:|---:|
|硬地直行|1310|约.528 s|.02715–.02798|5.5414 ms|82.6096 N|
|材料直行|844|.628–.631 s|.06697–.06776|26.20095 ms|157.21086 N|
|转弯|912|.538–.540 s|.00699–.00802|.3077 ms|73.3438 N|

这些swing的bearing_fraction均1，运动段fewer_than_two_supports_s=0。材料段并未表现为全面失稳，但相对于硬地，落地至建立承重间的累计无力间隙更长、早期冲击尾部更大、周期变异更高。材料段峰值最大227.09488 N。下一轮有价值的目标是改善接触恢复、冲击、真实步态协调和效率，而不是只证明能走。

统计边界：

1. off_s 是first impact后直到bearing established之间，Fz判定低于阈值的累计区间；不是整段腾空时间，也不是已稳定支撑之后的全部失联时长。
2. bearing_fraction=1 表示最终建立承重，并不保证第一次触地就连续承重。
3. 约3cm的swing peak clearance基本继承名义轨迹；这不是已学得自适应抬脚高度。
4. E600平移占空比FL=.60016、FR=.56427、RL=.61994、RR=.62347；前左右差3.5885个百分点。仅凭它不能定性病态、不对称设计错误或“漂亮步态”。
5. cycle统计排除了相位切换周期；要证明转换品质必须单独统计切换，而不能仅看稳定段CV。
6. `gait/phases.motion_positive_CoT`包含转弯，`gait/new.moving_positive_CoT`只含phase1/3平移；两者不可混用。原地转弯XY位移近零，其CoT可很高，不能据此称转弯能耗增加8–10倍，应另报J/turn和J/rad。
7. CoT为正机械功，不是电池能量；接触滑移采用静止水平面/球形足的特定重建，不可原样推广坡面、移动地面或任意足形。

来源：[新接触指标](https://github.com/kaiwen123-yang/MCP-RL/tree/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/gait/new)，[相位指标](https://github.com/kaiwen123-yang/MCP-RL/tree/c8378beae4be371eb4f104782b75e5d295b6f59a/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/gait/phases)，[指标实现](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/scripts/s1_w59_contact.py)。

## 4. 机制窗口失败到底说明什么

24个short/recovered窗口都跑到完整观察终点，body_traversals=23，home_stable=true；全部status为mechanism_criteria_unmet，是“12个指定目标未全通过”，不是24次摔倒。

源码固定：development cell[69,−9]，heldout[95,−9]，格宽2cm。每个窗口12个奇数forward traversal目标。筛选分层：

1. source：该固定cell、FR leg1、source发生在相应forward traversal内。
2. pair：该source之后该cell**紧接的一次load**来自RR leg3，且target仍处于同一traversal。不能跳过一个同足重载，去挑后面更有利的RR事件。
3. physical match：source释放、正自由间隔、真实足印交集面积>0。
4. short：0<gap<3s；recovered：gap≥6s，并且任务层另有连续6秒清空证明。
5. episode：观察窗完整、回家稳站、12/12都达成。

|分母|达成|无指定FR source|有source无合格RR pair|已有pair但未过物理/时间筛选|body未完成|
|---|---:|---:|---:|---:|---:|
|全部24窗口，288目标|44|168|61|15|0|
|heldout16窗口，192目标|36|106|36|14|0|
|其中学习8窗口，96目标|22|54|14|6|0|

所有recovered目标continuous_clear_s≥6。因而“没有等满6秒”不能解释当前这些失败。更突出的实际瓶颈是指定脚位/入射步态相位没有产生足够的合格source/pair。

尤其：**24窗口的F01全部未达成**；22个没有FR source，另外2个是development-recovered B3 A/B有source无pair；heldout16个F01全部无source。也就是说本轮12/12整窗通过门在每次第一目标后就已不可能达成。此事实需要正视，但不能事后删除F01或降低W59标准来制造成功。

机会并非整个环境不存在：24窗口的完整cell ledger合计396,574 source loads、38,646 positive cross-foot overlap；E600原路线本身也有17,638 source loads、1,461 positive cross-foot overlap。因此应表述为“固定cell、指定FR→RR且立即重访、同次forward及时间条件下覆盖稀疏”，而不是“模型完全没有见过历史接触”。

尚未证明的细分必须保留：61个source_no_pair里面，到底多少是same-leg抢先重载、其他足、RR太晚或从未来到；15个pair_no_match里多少是零交集、间隔不符或release问题。compact targets不足以拆解这些，必须读取对应原始 `w59-source-load-opportunities.jsonl.gz`，不能猜比例。此轮没有把全部原始ledger从数十个大ZIP取出。

heldout逐case结果与相对B3成本：

|候选/case/mode|达成/12|B3达成/12|slip变化|CoT变化|
|---|---:|---:|---:|---:|
|5901 A short|2|2|−.3993%|−.0412%|
|5901 A recovered|4|1|−.5999%|+.1749%|
|5901 B short|4|6|−.4625%|−.1965%|
|5901 B recovered|3|1|+.6978%|+.1708%|
|5902 A short|3|2|+.9015%|−.1854%|
|5902 A recovered|3|1|−.5261%|+.2138%|
|5902 B short|3|6|−1.7415%|−.4904%|
|5902 B recovered|0|1|−.2313%|+.2163%|

按48个heldout目标聚合：B3=10/48，Fixed-action=4/48，5901=13/48，5902=9/48。它们不是48个独立随机样本，也不构成对新材料的泛化统计。5902B recovered的0/12可直接分解为10个无source+2个无pair，不能直接判定材料恢复预测完全失效。

来源：[score_targets与cell_chains完整定义](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/scripts/s1_w59_geometry.py)，[目标协议](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/experiments/terrain_memory/s1-w59-footprints.json)，各attempt的mechanism.targets见配套JSON，保留全部288个目标。

## 5. 视频证据可用范围与时钟

本agent负责证据索引，未自行声称逐视频观看。父agent已报告实际解码、核hash并观察：5901/5902 A E60各六片段各24帧、B3 A E60六片段24帧、5901B E600十六时点及三段十二帧序列，总计124个视觉样本。最终报告应写“抽样视频/关键帧核查”，而非整段600秒每帧观看。

- [E600完整视频](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-w59/w59-ppo-full-fallback-5901-b-original-e600-a01-full.mp4)，181,841,351B。
- [5901 A六事件片段ZIP](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-w59/w59-ppo-full-fallback-5901-a-original-e60-a01-gait-excerpts-originals-001.zip)，1,383,622B。
- [5902 A六事件片段ZIP](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-w59/w59-ppo-full-fallback-5902-a-original-e60-a01-gait-excerpts-originals-001.zip)，1,389,944B。

clip选择start/material-entry/straight/turn/crossfoot-reload/complete-home-hold各2秒，起点约event−.5s；manifest记录的是源片30fps索引，父agent测得输出mp425fps、306frame、12.24s，不能把二者直接当物理日志对齐故障。源时钟与clip时钟应通过manifest和脚本映射。

E600 t130仍在outbound与10往返长任务一致，469.408进入home_hold，471.444才complete。视觉上468–470秒的静止/足部调整段与日志吻合。视频适合观察体态、足序、触地回弹、侧向偏置；毫秒级off与冲击力需日志量化，不能凭低帧率视频否认或夸大。

## 6. 近期官方正文带来的设计启示与已有工作的边界

### 6.1 最新强步态参照：Learning optimal gaits and styles（首版2026-11-19；实际所读v2为2026-11-21）

[官方论文](https://arxiv.org/abs/2611.25958)，[PDF](https://arxiv.org/pdf/2611.25958)。本次从官方不带版本号的PDF地址取得文件；随后核对本地PDF第一页内部标记为 arXiv:2611.25958v2 [cs.RO] 21 Nov 2026，文件9,344,095字节、SHA256为 be7e37e4aa46bb239c715bde8090b21d3492f453ddd3b9bc7a7c7a6b7f893c6d。所读范围为Methods、相关限制和补充参数，不声称全文每字逐读；未核官方代码。状态是预印本，不应写Nature/Science已接收。

核心是两阶段：先在11种**人为给定耦合矩阵**间随机切换，用同一PPO学身体高度、净空、足前后偏置、4振幅和4频率；11维动作33Hz，CPG/IK/PD1kHz。再冻结低层，训练按性能标准选择11种步态的策略。它已经覆盖“多步态+样式+地形选择”，只做这种组合不能算本项目独占创新。机会在于实际接触而非仅振荡器相位、耗散材料历史、三维脚位与过渡轨道。非成功率指标只在成功机器人上计算，不能复制这个评估口径而忽略失败代价。

### 6.2 LoComposition (v2,2026-10)

[官方作者页](https://www.loukaskordos.de/publications/locomposition/)，[全文](https://arxiv.org/pdf/2606.15896v2)。本次读方法、消融结果与限制。它把任务跟踪、执行器/姿态操作限制、机械功、地形感知分开；CaT处理限制，功率惩罚逐步加大，无固定air-time/contact-count/clearance先验，50Hz输出关节位置偏移。

应借鉴目标分工和匹配消融，而不是再堆奖励。其主要策略在所有测试速度仍是trot，未展示速度诱导的离散步态转换；CaT也不是运行时安全证明。它为“直接从约束和能耗生成运动”提供强外部对照，同时提醒固定接触数量/抬脚模板可能限制适应性。

### 6.3 PROMO (2026-10预印本)

[官方全文](https://arxiv.org/pdf/2610.01260v1)，[作者项目](https://amrmousa.com/promo/)。本次读III方法和IV评估。它将跟踪/稳定/效率作为三个语义目标value head，步态时序/接触平衡等作为固定prior；preference条件actor与history encoder，privileged critic训练。actor梯度不穿过history latent/velocity estimator；仍用PPO/GAE，加很小的对称KL动作多样性项。

它证明偏好接口可作为部署控制量，但没有提供可信动力学梯度；报告中各评估rollout偏好固定，不等于完成了动态材料变化下的闭环偏好切换证明。本项目可研究“材料风险与支撑质量驱动的可解释性能折中”，必须同时检验速度任务、能耗和接触风险，避免把冲击减少的慢走策略包装为全面提升。

### 6.4 NMM (IJRR online2025 / volume2026)

[期刊DOI](https://journals.sagepub.com/doi/10.1177/02783649251347305)，[官方开放PDF](https://pure.tudelft.nl/ws/portalfiles/portal/249500525/calzolari-et-al-2025-exciting-families-of-passive-gaits-in-an-elastic-quadruped-via-natural-motion-manifold-control.pdf)。本次读保守轨道族、横向坐标、线性化、输入零空间、稳定条件和实机部分。

可借鉴的是轨道族：Poincaré截面上的固定点曲线按能量参数化，横向稳定与能量注入分开。理论要求相应输入秩/可控性与局部条件；3kg BERT串联弹性机构主要矢状面、跑步机上约.14/.20m/s。Go2浮动基、塑性地面与碰撞耗散不满足直接照搬条件。本项目应构造“材料条件的受控周期轨道族”，验证局部返回映射/扰动衰减，而不能把它称为已证明的被动自然运动流形。

### 6.5 必须正视的近邻：Terramechanics-Aware Adaptive Quadrupedal Locomotion (RA-L2026)

[IEEE官方页](https://ieeexplore.ieee.org/abstract/document/11685272/)，DOI10.1109/LRA.2026.3732903。官方元数据/摘要已核：塑性屈服的足地模型、非线性沉陷/剪切饱和/承载损失，基于接触历史估计摩擦与可变形性并匹配步态，含IsaacSim、MPM Newton及Go2沙地验证。**全文和官方代码尚未取得，不能冒充已读公式或排除它含有某功能。**

这足以否定“加入材料自适应+步态”本身就是未有先例的说法。本文对它仅做近邻提醒；完整算法差异/创新权利要求需要在合法可得全文与代码后继续核。2025中科院分区未在本次验证，不能杜撰具体分区。

## 7. 建议 B 线成为真正步态方法研究，而非W59三维事件参数再调一次

### 7.1 一个可检验的核心问题

**在具有接触历史、恢复时间和承载变化的地面上，怎样联合改变脚位、轨道形状、支撑转换与步态节律，使真实接触的冲击与承载建立可控，同时保持任务速度和机械效率？**

这与“改变due时间”不同，也比笼统“学自适应步态”更能对应W59暴露的故障。创新是候选假设，不是已有结果。初期把MPC/WBC和PANN接口冻结，分别证明步态层的作用；它们自身的大改在其他分支开展，联合阶段再接入经过独立验证的版本。

### 7.2 改变的对象和接口

- 轨道变量不仅频率：三维落脚位置、摆腿样条控制点、触地法向速度、支撑法向力建立曲线、机身高度/姿态轨道、相位间耦合与接触转换时间。
- 状态包含真实GRF/接触置信度、滑移、机身与关节状态、材料参数及历史状态的估计分布。未知材料时用信念/不确定性，不能直接把仿真真值当部署输入。
- 以2–3个可实现接触模式图和连续轨道族开始，设计支撑切换的守卫条件及轨道衔接；保留新策略拒绝/退回冻结轨道的接口。模式图只做初始求解结构，不能把“任何时候至少两足”写成所有未来动态步态不可改的研究公理。
- 材料条件作用于轨道和支撑转换本身；仅给policy加一个latent、仍固定足轨迹和名义接触序列，不足以证明这一创新。
- 分离沿轨道进度与偏离轨道的误差反馈；对局部受控返回映射做经验线性化，报告扰动后逐周期衰减和不确定性。若不能满足秩/接触可微条件，报告经验结果，别借NMM定理做保证。
- 实机目标需要闭环耗时、p95/p99和截止时间错失记录；逻辑500Hz/1kHz并不等于墙钟达到该频率。控制预算由部署栈和目标硬件实际profiling决定，不在此编造可达速率。

### 7.3 必须分开的比较

内部因果比较保持同MPC/WBC/模型/动作范围/训练预算：冻结W59；只开脚位/样条；只开真实接触相位反馈；两者结合；加材料状态；加历史状态；加轨道族转换。可合并前几项为小型分阶段消融，避免六个大改同进一条主线。

强外部比较至少包括11-gait/style CPG思想的同接口改造、固定/可切换轨道库、以及LoComposition式约束+能耗学习的系统级参照。后者是关节位置策略，底层不同，应标为整栈比较；不能把整栈优势反推为某个本模块的因果贡献。PROMO式偏好条件作为任务性能折中参照，不当动力学模型梯度参照。

### 7.4 先修可检验性，再花大训练预算

- 完整保留W59旧12目标任务及失败结果用于回归。
- 新协议先用冻结控制器和离线运动学/动态可行域检查F01及后续目标，从预先划定的物理可达cell、入射相位、速度范围抽取实例。禁止看候选结果后移动cell。
- 同时报“无source机会率”“有source但无RR配对率”“满足几何/时间条件后的控制成功率”“端到端任务成功率”，不能只报最有利的条件指标。
- 同一footprint历史比较需要重置/保存材料状态，固定当前状态但改变过去加载、跨足共享与private限制；否则材料历史收益与脚位覆盖收益混淆。
- heldout要包含未见材料参数组合/恢复时间、加载顺序、路径、速度、扰动，而不止把一个cell东移26格。
- 种子和场景作为独立统计单位；同一回合12targets只作嵌套观测。实验停止按预先登记的数据/收敛/质量条件，遵从用户不设后台完成时间的要求。

### 7.5 第一批可交付、可并行的B线任务

1. **诊断与协议任务**：导出288目标逐级漏斗、F01入射phase/足位置、材料与硬地的接触周期、冲击分布、实际足序与转换轨迹；补读15个pair_no_match对应ledger。输出新旧协议分开的覆盖可行域，不改W59分数。
2. **轨道与控制任务**：实现全足轨迹/脚位/接触建立曲线接口、两个模式图、真实接触相位反馈、局部轨道误差与退回机制；先在相同冻结低层上做有限物理可行实验。不是直接上六份PPO大训练。
3. **学习与对照任务**：搭建CPG/style、受控轨道族、偏好条件和材料历史的匹配训练接口；数据/身份/度量确定后再放行成批训练，比较样本效率和真实仿真成本。

这三项分别可由独立worktree/责任文件推进，但共享协议、版本化接口和精确artifact身份。CPU/GPU槽数量由根审查中已核的吞吐/内存/IO共同决定；此步态审查不替代并发资源测量，也不保留“六槽只多1–2%吞吐”这一已被其他代理纠正的旧表述。

## 8. 最终对外口径建议

“W59已形成可冻结的仿真回归基线；5901B完成10次合法往返，并在471.444s完成后继续通过128.556s稳站观察，总物理窗600s。PPO步态收益尚不一致，固定足印机制的主要问题是指定source/pair覆盖不足，全部窗口第一个目标未达成。可以立即开展独立研究分支，主线优先拓展脚位/轨道/真实接触反馈及材料条件的轨道族，而不是继续无限迭代W59。实机实时、广泛泛化、材料历史收益与联合算法优越性尚待专门实验。”
