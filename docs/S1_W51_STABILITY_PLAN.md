# S1 W51：从起步辨识反馈刷新与步态初始轨迹的作用

命令传递定向修正（提交 `463e91b`）：首个 D1 在真实入口发现 userdata[78:84] 未填充，原运动候选使用了零世界速度；已于实际 3.120s 停止其独立 server，并保留 execution_error/技术无效记录及全部已保存工作量。D0 原进程连续保留。修正仅使 W51 将原 mission.v/mission.w 写入既有命令槽，不启用额外 twist 成本；修正后 D1 从原 t=0 作一次有效配对尝试，详见 phase.json 的当前 jobs 和 technical_attempts。当前 tmux 为 `s1-w51-pair-command-repair`。此技术重启不作为新的物理失败或成功样本，也没有重跑 D0。

D0 实际终止结果已到达：原 body_fall 于 5.814s 触发，末 z=0.1189336264m，最大 x=0.0957089087m；实际计数为 29,070 plant 子步、311 完成 planner 调用、5,929 full rollout。2.002–4.3s 前进量为 -0.020903054m、目标 swing 仍受载约 71.18–80.19%，未形成稳定支撑推进。修正后的 D1 仍 running；截至已核验的 2.408s，原 mission 命令与 native userdata 命令一致，S 已真实作为 Q（包括 J(S)>J(B) 的搜索），未据此提前选择版本或启动原 E30。

## 本计划的实际实施状态（2026-11-15）
实现提交 `df68de8`；r39 完整重编 62 个相关单元及 service/server。首次编译因一行 if 的 misleading-indentation 警告被 -Werror 拒绝，修正后通过；未在该失败构建中执行物理。D0/r38 与 D1/r39 的 job 为 `job-20261115T112019Z-w51-d0`、`job-20261115T112019Z-w51-d1`，在 D57 `s1-w51-pair` 持续运行，无墙钟 deadline。实际状态以 [phase.json](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w51/phase.json) 为准。
运动状态由正常 Agent SetState/Transition 每 2ms 观察更新，stance 世界锚点及本次 swing touchdown 跨 replan 保存；IK 使用原 home 几何、原关节限位和上一节点连续分支。剩余相位采用唯一五次位置/速度端点连接，保留原 C2/5mm 意图。IK 数值迭代最多 40 次、单次关节更新最大 .2rad、末端容差 1e-5m；这些是求逆数值设置，不增加物理/落点/步高网格。
用户允许 tmux 稳定后结束本次对话。后台自动等两臂实际结束，生成逐窗统计、完整实际时段视频并从 D57 发布原件。后台状态随后为 `paired_D8_ended_pending_scientific_selection`，科学选择须读取实际承重和视频证据；不新增任意自动步态阈值。若有推进依据，使用 `python scripts/run_s1_go2_w51.py --select D0` 或 `--select D1` 启动该冻结版本原 A/B 各一次阶梯；两臂都差则记录成对负结果，不启动 E30。不得重复启动既有 D8。
C 已完成一次 114×176×113 的 FP64 CUDA 归约和 22 项覆盖记录：最大总 J 误差 5.820766091346741e-11，六组同问题内原 risk/roundoff 比较差异 0。76 条保存问题与 38 条错误 bridge 中点严格区分，新增物理/PPO 为 0。纯 state 代数及 Stand 非激活分支另记真实误差，MaterialSupportHeight 最大 9.009910922230802e-6；没有声称完整 residual/native cost 移植。见 [task-c.json](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w51/task-c.json) 及其已核验 Release。



更新：2026-11-15（Asia/Shanghai）。结果基点：`09bd55e0cb70244938ad82db7dd202a31ac70f77`；W50计划提交：`9ba97283c13119ec197429b75fce6b139a506a33`。

**计划形成时状态：planned_not_implemented。本次审查修复了公共runtime两处缩进错误；W51诊断入口、运动候选、服务器构建与实验尚未执行。** 没有新native solve、物理/GPU积分或PPO。W50的四例/两条短部署已经结束，不重开其预算。

[W50最终审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W50-final-review/README.md) · [视频与日志](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W50-final-review/video-review.md) · [启动提示词](S1_W51_START_PROMPT.md) · [机器交接](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W50-final-review/handoff.json) · [项目记忆](TERRAIN_MEMORY_PLAN.md)。

## 1. 为什么改变实验起点

W50/S5-r38/H在四个已保存W49问题上实施了当前时域重参考及同alpha反馈/度量转移：76个full rollout、36个正候选、10个strict改善。A237成本降1.335%，最低z仅提高.149mm，位移−4.297→−18.141mm；B236成本降.01144%，最低z仅提高.016mm，位移−18.833→−19.024mm。晚期A328抬高但倒退，B625有9个正alpha候选但均未strict改善。它没有产生支持原E30的正常承重/前进证据。

原始候选重算支持这些结果，所选三个赢家都倒退；另有向前但同样坍低、成本更高的被拒候选。不能说原J已经错拒了正常可行步态，不能用W49/W50不同fixture的25/36与10/36推断性能变化。源码和原完整数组没有显示重复alpha、旋转符号、K/H混配或主要网格错误。

W50 Task A带来更重要的分辨：private Trot在2.002s生效并按2Hz推进，但2.002–4.3s目标swing中实际足仍受载约88–94%，四足共同受载约86%，两流PD饱和均0。231个保留policy的reference/K/H确实来自初始Stand [.0,.35]，两流各14825次实际held查询使用过期末节点。可见坍低之前，足部卸载/支撑交换就没有形成。

**4.30s不是正常起步点，已经比Trot切换晚约2.3s。** W50只能说明在这些受困状态单次更新不足以恢复，未检验r38从t=0连续刷新能否预防。W51用同初态成对闭环回答两个问题：

1. 持续使用r38本身，能否避免旧参考造成的早期失效？
2. 在相同目标/物理下，为优化提供一个有支撑/摆动运动意图的初始轨迹，是否有额外作用？

长期目标仍是原Go2/TM02稳定合法完整往返及跨足/返程材料记忆的因果收益。当前先解决有效承重和前进，不新增PPO、不增加cap动作权限、不把短诊断当记忆收益。

## 2. 已修复和仍有边界的技术问题

本次直接修复`scripts/s1_go2_r11_runtime.py`两处`dump(...resolved)`缩进：W50更新后该文件在line499报`IndentationError`，另有同类line541；修复后AST可解析。它不影响W50独立有限脚本，但会阻断公共原任务入口。只作了针对这两处问题的语法确认，没有import模块、启动Agent或积分。

W50两条40ms部署的ctypes bridge重用了Python wheel同SONAME `libmujoco.so.3.2.3`，而编译头与所需frozen native库的`mjData`布局不同。原记录里actuator字段出现denormal垃圾；400子步不具备native物理等价性。修复后的`dlmopen`仅验证库映射，没有正确库的新物理部署。四个有限solve在独立native server中使用frozen库，不能因bridge失效而全部作废；两个再规划的输入状态受错误bridge污染，物理行为解释无效。

**W51直接使用原plant+独立native Agent+实际RPC/query/hold入口。** 不使用该ctypes bridge，不再安排两条40ms作为准入。正常plant本身仍使用原授权Python物理环境，不把两个库句柄/对象混传。第一次实际D8中记录真实server加载库、qpos/qvel/actuator/足力/非足力与策略来源；这是实际实验记录，不另起哈希、mock或冒烟轮。

## 3. 实验设计：两个完整初态、同seed、各8秒

| 诊断臂 | 控制机制 | 初态与权重 | 目的 |
|---|---|---|---|
| D0 | 已实现S5-r38/H，refresh-only | W44 A的seed1、checkpoint-014737；原evaluation从t=0初始化 | 测早期持续刷新的预防效果 |
| D1 | r38基础上增加一个相位感知运动候选S作为替代线性化起点，拟S5-r39 | 与D0相同checkpoint、归一化、seed和原evaluation初态 | 测运动初始轨迹的增量 |

每臂只一次t=0→8s连续科学诊断，前2s原Stand，随后原outbound任务。环境、actor确定性均值、observer因果更新、物理、成本、gait与求解器其他设置一致，独立工作目录/进程/可写build。D0/D1不等于正式A/B，二者都用seed1；正式B仍seed2。

使用原evaluation初始化，保留W44 actor/critic/optimizer/norm/RNG/祖先文件。不能把`resume-state-ep00428`当作完整plant中途恢复；不能在2.002s从不完整历史snapshot拼装世界。8s覆盖移动后的约12个名义步态周期，足以观察此前4.3s及7s的异常阶段，但不等于完整任务资格或泛化验证。

### 必须实现真实的D8入口

当前`run_s1_go2_r12.py --level`只支持E30/E60/E300/E600，`configuration()`会生成固定物理时长。因此需增加明确的`diagnostic`/D8路径，沿用原runtime、Joint/PolicyMaterial及`PolicyStream(training=False)`，把科学时长明确设为8s，结果写`qualification_usable=false`与独立诊断状态。不能只改外部contract时长后假称生效，不能用墙钟timeout截断E30，也不能用`training_duration=8`（会切换训练/long mission语义），不走仍带旧campaign时限的R11 CLI。

D8不得写入原E30通过记录或作为E60前置；正常终止写`diagnostic_window_completed`，原身体跌倒/非法接触/材料域等物理终止仍有效，技术中断另列。原短腾空许可不变。不新增“非足碰一次即停”“始终固定对角支撑”等硬规则。

同时在`s1_go2_r11_protocol.evaluate_stage()`增加明确diagnostic分支并提前返回；当前未知level会落入E300/E600的else并读取`target_circuits`，只改CLI/configuration不足以接通D8。诊断结果保留原物理失败与实际时长，独立于正式资格索引。

## 4. 唯一新增机制：一个运动候选S提供不同线性化起点

### 4.1 原B不可丢失

每epoch沿原路径做同epoch R35参考冻结和十个nominal screening，得到完整最优B及其原native score。**全局比较对象始终保留B。** 新候选S不重置R35参考，不改变同问题目标/权重，不能靠把较差S叫作baseline而制造下降。

令`c_B(x,t)`为B的完整pre-clamp/pre-QP控制中心，包含原feedback scale、旧delta及承重PD偏置；`q_B,j(t)`为B当前native rollout的12维关节位置。用下述相位足路径生成限位内`q_IK,j(t)`，构造

\[
\delta q_S(t)=q_{IK,j}(t)-q_{B,j}(t),\qquad c_S(x,t)=c_B(x,t)+\delta q_S(t).
\]

S是B的完整copy，追加这一条时变delta，保留原反馈和度量。不能以裸IK关节角覆盖raw center，也不能用已clip/QP的动作当raw center。只生成一个确定性的S，不做步高/速度/相位/权重/beta网格；不增加随机采样族。

### 4.2 足部运动意图与坐标

所有脚序按native与plant明确映射。保留2Hz、duty .45、原C2、5mm幅值和绝对phase；身体与足路径从当前实测状态连续接续，不每20ms从phase零开始。Nominal stance的足xy保持当前世界锚点，剩余swing向原下一touchdown运动。已有摆动从当前实测足状态接续，不重新开始一整段swing。

stance锚点在该足本次stance开始时从实际世界xy建立，整个stance和跨replan保持，不能每20ms重取滑动后的当前位置。当前swing的touchdown点在swing开始或首次接入时固定，剩余段从实际足状态连续接续；后续速度变化用于下一摆动，不让当前端点反复追逐。记录必要的足序、phase、锚点及当前touchdown来源，重复同一时刻的startup/求解不重置它们。

身体xy候选沿原当前世界速度命令推进（原PPO cap .18–.25m/s），方向/转向沿原mission；起点与actual base一致，目标高度沿原support/body目标。使用base而非未去offset的head位置，避免把r26的head前视与身体位移重复相加。一个固定的触地点构造为

\[
p_{td,i,xy}=c_{des,xy}(t_{td})+R_z(\psi_{des})r_{nom,i,xy}+\tfrac12T_{stance}v_{cmd,xy},\quad T_{stance}=.45/2=.225s.
\]

`r_nom`从原正常站姿几何取，不新增足距调参；`c_des`按当前状态及原命令连续推进。它是本项目候选设计，非已证稳定控制律。足z使用原地面/材料support参考、足半径和原phase/C2，不混淆足中心与足底间隙，不把额外5cm等步高偷偷并入。正常Stand/terminal hold时不加运动S。

原Gait成本的C2参考保持不变；实际脚位置可能不等于该参考，因此seed不能在第一节点跳到参考后声称连续。用原剩余swing时间和唯一确定性的连续连接，从实际足位置/速度接到既定touchdown，并保留原5mm参考的运动意图，不新增可调步高或多连接方案网格。种子轨迹不要求在接入瞬间严格等于另一个不同的C2参考。

运动学解沿时间连续分支并受原关节限制，优先使用已有机器人几何/kinematics。IK/坐标计算不做物理积分；若既定目标无解，记录该epoch不可用并回B，不搜索额外步幅/关节分支网格。IK路径本身不证明动力学可行，后续native rollout裁决实际受力、限幅、碰撞与任务成本。

### 4.3 只保留一个方向族，区分搜索起点与接纳基线

对完整S只做一次原native rollout。S有效时，即使`J(S)>J(B)`也可作局部搜索起点Q；暂时较高的候选成本不等于允许部署较差策略。若S无效，Q=B。只围绕Q执行原一次first-valid backward流程及九个正alpha，**不先跑B的九个再跑S的九个**。

沿Q当前native轨迹按r38方式计算raw center及当前MuJoCo切空间Jacobian，首个有效backward同时捕获k/K/H/μ/active；用同alpha做当前时域前馈/反馈/H转移。原176状态/175自由动作网格、末节点规则、实际scale1与delta吸收、K/H双来源、严格native评分保持。positive分支是Q附近局部构造；alpha0仍直接返回原全局B的deep-copy，不声称Q=S时positive趋零等于B。

将有效S及全部新正候选按原R34风险/J/roundoff规则与**原B和全局当前最优者**比较；只比S好却比B差的解不能安装。若S本身严格更好可保留为有效候选；其旧reference/K/H来源必须如实标记，不能因为新增delta就称已刷新。无更好者保留B，不为刷新年龄强制安装劣解。

候选期望下降/ratio继续null，不能借Q局部dV给混合策略编造接纳比；最终策略对象必须就是完整实际评分对象。保留原无有效backward返回/复用语义，不无限重试。

### 4.4 计算量差异必须诚实记录

D0正常新epoch通常最多`R35 freeze1 + nominal10 + positive9 = 20`个full rollout；D1至多增加一个S，因此21。已有same-epoch冻结可复用，真实计数可能较少。每臂只有一个方向族，FD按实际原规则单独计数；没有额外整套N或多初值搜索。

两条8s包含约842个planner调用（每臂约20次startup、400次普通replan及一次phase切换；以实际计数为准）、合计最多80,000个.2ms plant子步。保守full-rollout规模约17,261，不包含FD内积分；不是W50的120rollout量级。额外S会增加D1工作量，不宣称严格同计算预算或吞吐胜利。记录solve/rollout/FD/CPU墙钟/GPU工作分别的实际数值。

## 5. 根据实际步态选择版本，再进入原任务

预先比较同窗口0–2、2.002–4.3、4.3–8s：实际安装/参考覆盖，足部swing卸载与stance受力、支撑足世界切向速度、足间隙、z/vz/姿态、非足受载、前进/倒退与原成本分项。视频必须与真实物理时间和力日志对齐。允许短腾空，不把1N诊断阈值变成新资格。

判断顺序：

1. 若D0已有正常承重/前进且D1没有实质增益，优先选简单的r38；不把从早期刷新本身的收益归给S。
2. 若D1改善卸载/支撑交换、维持身体与推进且满足原约束，选择D1同一冻结实现。报告相较D0的整体增量，不能把后续reference变化单独宣称为独立因果贡献。
3. 若只有成本下降或趴低挪动，或两臂都无法正常支撑推进，停止本轮原阶梯，交付对照负结果。此时按原数组分辨：候选生成不足、目标排序、或预测—执行偏差；不现场叠加新步高/权重/horizon/PPO。

不要求D8提前完成3m材料去程，不要求两个诊断臂都成功，不以微小数值改善作实质进步。诊断选择不是新任务资格；它给出是否有理由投入原任务的证据。

选定后固定**一个正式控制机制**，原A=seed1、B=seed2各从原evaluation初态一次E30，可并行。D8是预声明配对诊断，不能拼接或冒充E30；如需重新初始化进入正式E30，明确区分attempt，不以不完整checkpoint声称无缝接续。原A/B仍W44各14737 transitions/update57，checkpoint-014737，resume ep00428/ep00417完整保留，tail0/excluded145，新PPO0/0、donor0。

- E30：30s内原合法材料去程、远端3m原容差，并完整观察30s；转向/返程继续记录。
- E60：原合法完整往返、回家连续稳定2s，完整观察60s。
- E300/E600：原270/540s内5/10次合法往返，完整观察300/600s。
- 各case前级通过才后级；失败只blocked本case后级，另一case继续。不换seed、不追成功重复、不拼接资格，技术中断与物理/任务失败分开。

保留Go2质量15.206408kg、PD60/5、每腿23.7/23.7/45.43Nm、TM02主应变.02、条带x[.8,2.2]/|y|≤.6、原任务/接触/跌倒条件、.35s horizon、2ms hold/10×.2ms子步。无新增训练、无实机操作。

## 6. 第三线：把GPU工作推进到成本覆盖

W50已经用旧记录将首个可观察差异缩到约束求解输出：活动约束输入和接触首步一致，contact efc_force/qacc开始分歧。全面contact字段在10步中有9步一致，第10步也出现差异；旧“10步全一致”只适用于此前有限字段。仍未锁定唯一kernel，不能把stub排序flag当作已实现修复，也不继续重复相同F1/L/F2。

C与D0/D1独立，**本轮默认零新GPU物理步**。使用W50已有114条候选记录及同问题上下文，为GPU成本迁移建立可复用结果：

1. 实现并保存FP64批量native成本归约：从已有113维残差、22项定义/norm/weights计算每节点分项和总J；原风险排序另读`trunk_violation`、`failure`等候选元数据，不能从残差单独恢复。只在各自同问题/同epoch组内比较，不能把六组混排；与CPU原件逐项对照，复用已有预算，不引入宽松GPU接纳。114条中76条是保存问题候选，38条来自错误bridge状态的midpoint；后38条仅作代数与已记录比较逻辑对照，不用于正确物理、轨迹可行性或控制收益结论。
2. 将22项ResidualFn逐项标成可由保存state/action/kinematics/context计算、依赖接触/solver数据、或缺数据；可独立实现纯运动学/代数项的GPU批量版本，保留真实误差与未覆盖项。
3. 输入仅有native残差时，只能称“成本归约已覆盖”，不能称“GPU完整native cost已移植”。若从state重建残差需要缺失数据，标缺并提供精确采集字段，不伪造零值，不新开原episode补旧档。
4. 不生成新的物理轨迹，不用重复batch填显存，不额外跑吞吐轮或PPO。后续GPU提案/CPU native裁决建立在该覆盖结果上，本轮GPU不决定正式安装或阻塞CPU。

C可自然结束；25CPU/96GB不意味着三条任务必须全程占满。该支线的目标是减少未来迁移完整目标函数的工作，而非维持无效显存占用。

## 7. 资源、正常入口和构建

先核最新main、未提交改动、服务器已有W51 job/tmux/PID/真实physical clock；已存在有效同身份任务则续接，不重复启动、不宽前缀kill、不动其他项目。默认CPU树`/root/autodl-tmp/MCP-RL-w42`、GPU树`/root/autodl-tmp/MCP-RL`，环境`work/envs/d57`/`work/envs/w41-mjwarp`，以现场有效路径为准。

实测25vCPU配额及96GB实际余量：D0用10核，D1用10核，C用2核，3核留系统/主控/发布；根据现有其他任务调整实际affinity。线程库1，记录实际全进程树资源，不能以affinity宣称独占或利用率。两CPU主线不再加第三个重native solve。

build最多-j6先完成，再运行两个重求解。D1如改policy/header，完整重编依赖service和server，不能复用ABI不符archive；从r38完整配置继承后overlay，并更新prepare/spec/patch/runtime/policy/recorder/执行合同的实际接入。D0使用r38原native行为，公共runtime缩进修复两臂共享。不要让字符串全局替换再次破坏语法或只更新标签漏掉真实分支。

只做与这次改动及真实入口直接相关的必要验证，不新增全仓hash、mock、shape/import/smoke或无信息增量测试轮。D8就是实际闭环实验，记录正常server/RPC/plant路径；不为恢复旧桥接功能另跑准入任务。

## 8. 记录与交付

每臂输出完整真实轨迹和实际运行时段视频：正常完成覆盖8s，原物理提前终止则保留真实终止时刻，不补齐、不为凑满时长重复。另保存设置、初始状态/权重身份、每次原B/S/Q、score与最终安装来源、reference/K/H来源、实际足Fz/非足Fz/足世界速度、预测支撑、PD/投影、原J分项、工作量及失败原因。保留首Trot及最早异常前后的完整策略与候选，不仅存末态或索引。

视频按已保存真实轨迹渲染，无新物理。紧凑逐窗统计、关键同问题候选与直接证据提交Git；必要大原件/视频从服务器增量Release直传。每个产物注明D0/D1或正式A/B、实际控制器、seed、物理时段，避免再次只有“跑满”和总J。

**新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix；不设墙钟完成时刻、不到点kill、不因对话结束杀后台。** 8s诊断窗与30/60/300/600s评价是物理任务定义；保留技术错误退出、原物理终止、用户停止和单次网络timeout，无无限重试。运行中如实标running，未发生的不填completed。

Git短写串行合并最新main，不强推覆盖；核验真实commit/资产后结束，不在用户电脑额外下载、复制、打包、镜像或归档。服务器科研原件、checkpoint/resume/视频与仍被引用输入保留，不在本轮审查中自动删除。W50三ZIP已经发布，无需重复上传或重跑。

最终交付：实际实现、两个诊断臂的步态/承重/前进对照、选定或未选定理由、原阶梯最高真实结果、技术错误及未运行原因、CPU/GPU实际工作与远端位置。W51的任何理论提案、计划或短诊断都不写成完整任务成功。

## 9. 文献如何约束方向

[本轮四篇一手文献](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W50-final-review/literature.md)涵盖MuJoCo全身MPC、HOUND接触隐式MPC、HiLQR、DIAL-MPC。它们提示接触探索、初始轨迹与足部运动意图重要；不支持把小成本下降当正常步态，也不支持仅凭.35s小于一周期就判时域错误。

动力学FD当前已为单边FP64，W50 raw-center中心差分是零物理局部映射，不因论文对动力学中心差分的经验重新开该支线。HOUND防滑项在全足静止时也为零，不能单独复制就保证迈步。DIAL使用不同动力学/接触设置，其GPU吞吐不移植为当前原任务资格。本轮S只改变候选初始化；目标结构、接触模型、FDDP或新采样控制器留待本轮证据决定下一项改变。
