# W49稳定性计划：从真实失稳事件分离执行误差与恢复能力

执行更新（2026-11-13）：D57已实施S5-r37，四个新事件完成4次N solve及36个k引导既有P候选；四事件均有原strict可接纳且动作不同的策略，4×20ms真实RPC部署完成400子步。C完成60 world-step、预热0。原A/B E30已在tmux运行，资格尚未产生，后续状态以[W49实际索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w49/phase.json)为准。计划首段“planned”仅为发布时状态。

日期：2026-11-13（Asia/Shanghai）。审查基点：`61e16975fa6b38dc8e2d31779c2bd82f1aaeb8b6`。**W48已完成；W49为planned_not_implemented。** 本次审查/发布新增服务器任务、native solve、plant、PPO和GPU physics均0。用户要求的双/三任务并行、无固定后台截止、GitHub核验后无电脑归档在本计划持续生效。

[W48最终审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W48-final-review/README.md) · [项目记忆](TERRAIN_MEMORY_PLAN.md) · [启动提示词](S1_W49_START_PROMPT.md) · [机器交接](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W48-final-review/handoff.json)

## 1. 本轮必须回答的问题

W48正式r36确实部署，identity修复有效，但四个旧困难问题的赢家未变，原A/B物理轨迹与W46逐元素相同。原E30仍0/2，合法去/返/往返0；不能再次以同配置重复原任务期待随机成功。另一方面，原1944调用有1601正步安装，不能再用旧四fixture推出全程没有下降。

W49的主问题是：**真实执行在哪一步偏离预测；预测可信时，优化所选策略为什么不能维持支撑并继续前进？** 两条CPU线并行，一条定位实际链路，一条测试一个有明确行为差异的恢复候选。第三线解决GPU首hold初始化疑点和交付尾项。

不是预先认定float时间、材料、K大、代价权重或PPO不足为唯一根因。W39已量化显式时间舍入，W40已有动作FP32记录；W48补齐的新事件允许重新判断其在当前控制器中的量级，但这个旧假说不能占掉整轮。

## 2. 开场、基线与冻结身份

默认执行端D57：`autodl-d57`；CPU树`/root/autodl-tmp/MCP-RL-w42`，GPU树`/root/autodl-tmp/MCP-RL`；环境`work/envs/d57`和`work/envs/w41-mjwarp`。先核最新main、未提交修改、有效tmux/PID/job、实际物理clock/记录增长、CPU affinity/cgroup、RAM/GPU/磁盘和无关作业。有效W49存在就按身份续接，不重复启动，不重开W45–W48或G5，不宽前缀kill，不动SuperMimic等无关进程。

基线S5-r36 binary SHA256：`b05c8da8ca74551dc73ea0272329234217d7160731e5cea8e33101bba3241338`。保留exact self/antipode修复、N/r34 strict/constant。诊断build和正式build分别命名，不能仅换字符串冒充新控制器。

保留Go2/TM02质量15.206408kg，PD60/5，每腿23.7/23.7/45.43Nm，2ms held与10×0.2ms子步，.35s horizon，2Hz/.45/.005 C2步态，r14/r26/r27/r28机制，原材料主应变.02、非足接触/跌倒、原路线和资格。PANN/658D/2Hz cap/W38奖励不变。短腾空合法；raw Kδ与诊断.2m/.5rad不成为新硬阈值。无PPO、seed/权重/步态网格和完整新求解器移植。

从**完整冻结r36配置**复制再施加显式差异，不用新`configuration_overrides`替换继承表。对生成模型所需numeric、source/model/library/protobuf描述符与实际进程环境一次读回，直接防止W48已发生的17键遗漏；不另开原plant测试配置。native评分、部署路径和接口代码版本分别记录。

## 3. 已有数据与四个新事件

原位读取W48现有`w48-planner-boundaries.jsonl`、`w48-held-clock.jsonl`、`feedback-actions.jsonl`、`action-execution.jsonl`、原500Hz trajectory、必要physics流及事件对象。按已提交source清单核size/hash。优先D57已有原件，不下载整批历史Release，不重跑原任务找已经存在的对象。

| 事件 | 主要问题起点 | 原事件 | 用途 |
|---|---|---|---|
| A首held大跳 | call213@3.84，plan178 | 3.846s | 边界后6ms的命令/状态分叉 |
| B首held大跳 | call176@3.10，plan161 | 3.106s | incumbent部署后6ms分叉 |
| A首膝反转 | call363@6.84，邻接call364@6.86，plan305 | 6.860s | 明确反转在重规划前后哪个事件 |
| B首膝反转 | call328@6.14，plan267 | 6.150s | 20ms内反馈/投影与承重变化 |

Task A第三项用6.84区间并检查6.86查询，覆盖反转前因；Task B可用call364完整问题分析恢复。两者不能偷换为同一历史solve。事件对象保存了policy、accepted、P0、context和同epoch reference；保留旧incumbent时使用真正部署的plan。0.02s站立首误差可作为已有日志对照，默认不为其新增第五组物理实验。

先关联完整时序：planner/问题epoch→SetState→native querytime→reference插值→Kδ→ctrl clamp→H-QP→RPC响应→actual held→10个物理子步→下一状态/接触。相同时间的probe、nominal查询和applied_hold分开。记录old/new `q_target`、实际PD/限幅、首差组件、单位和发生先后，不能把规划边界误差移到6ms后的事件时刻。

## 4. CPU Task A：真实执行链分解

### 4.1 先做零积分同态查询

对上述四事件已有实际held状态，在固定完整policy/context上对比：

- 原显式float请求时间路径；
- 同一次SetState之后省略GetAction time、使用native double state.time的路径；
- native double动作、RPC float响应、实际held三者的对应关系。

将querytime误差、参考插值差、raw/clip/H-QP/active-set差、响应舍入和`60×Δu`对应的未限幅PD差分列。实际力矩仍按PD60/5及原限幅计算，不能把60Δu当真实越界力矩。使用相同原state、相同完整policy、相同H，不加状态噪声/K缩放/容差扫描。输出前后数值和在事件实际首差中的量级，不仅报告是否不等于0。

省略time无需改proto，但必须核每条查询最近SetState的时间确为当前时间。返回动作float精度保持不变，避免一次混两个干预。该比较0积分、0FD、0solve；不能单凭float字段存在宣布根因。

### 4.2 必要时做≤12条20ms诊断

四事件各最多三个变体，总≤12条20ms、≤1200个0.2ms子步，0优化solve/FD。先看首2ms，明确问题已经回答时不为耗满额度扩窗。

| 变体 | held来源 | 模型/起点 | 要识别的差别 |
|---|---|---|---|
| A1 | 保存的实际held序列 | predictor、显式共同重建状态 | 相同输入下的预测结果 |
| A2 | 同一policy、原真实查询/响应路径闭环生成 | 同predictor、同重建状态/context | 策略执行与保存输入之间的贡献 |
| A3 | 同一保存held序列 | plant模型、可核的共同重建与外部状态 | 相同输入下的模型贡献 |

time-only若需要积分，只能替换已不再必要的一个变体，并预先写清比较对象；不再增加独立长窗。新控制候选的短部署见第6节另列额度。

**完整历史MjData没有全部保存。** qpos/qvel/act不是完整数值状态。两对象须明确相同的初始化/coldstart、mocap/userdata、外力、callback和模型设置，保存可用的integration状态与solver/接触痕迹。共同重建试验只能解释该条件下的差，不能冒充历史bitwise replay，也不能因此排除未保存warmstart影响。若A3需要的材料/外部callback状态不能忠实构造，标unavailable并取消A3，不填零、不因此重跑整个W48。零积分动作查询仍可完成。

首分叉归类为“输入/查询先不同”“同输入模型先不同”“已保存字段不足”或“尚无法唯一定位”。时间作用小则在同一W49结束该假说，不把它升级为下一次完整原回合的唯一理由。

## 5. CPU Task B：计划与恢复，不再只检查零端点

### 5.1 用现有accepted horizon先判断计划是否本身不利于恢复

四个新问题固定实际P0/x0/time/private gait/material proxy、norm/weights、同epoch constant ubar/F/R。读取已保存accepted全.35s：前移/侧移、最低高度、姿态、终端速度、动作跳变、风险及J0/Jref分项；结合逐held真实接触/GRF。未保存的预测接触需要模型forward重建时明确标注，不能用脚高或gait phase冒充真实受载。

同一策略在预测状态和对应实际状态的动作比较用原生求值。超过下一replan之后，实际轨迹不再是这个固定策略的闭环结果，不能拿.35s预测与多次换策略的实际段宣称同策略误差。

输出机制判断：计划本身已驻留/下沉/缺支撑；或计划尚可、实际误差经反馈/投影后恶化；或参考/模型接入错误；或证据仍不足。MaterialSupportHeight分项大、K大或J下降都只是线索。不得先选一个喜欢的权重再解释数据。

### 5.2 一个有限候选家族：冻结baseline反馈，检验feedforward增量

若事件显示新反馈/投影整包更新可能破坏方向，或当前同问题新反馈候选继续预测下降/实际恶化，允许测试唯一候选家族：**保持baseline完整反馈动作映射，只加本次方向的feedforward增量。** 这检验新K/H/reference整体替换的影响，不裁K、不加任意平滑阈值。

历史r33 `R33Structured`的P32已保留best nominal的reference/K/H/scale并扰动feedforward；本轮不把冻结反馈架构冒称新发现。新的有限问题是：在四个W48真实事件上，用同问题N backward算出的k替代旧P的15对手工正弦方向。r35 none/load/constant改变的是任务输入参考ubar/R，与此候选方向维度不同。只做这一个k-guided P变体，不重开旧P方向网格。

最多四个新事件各一次完整N solve，≤4 solve；先保存实际baseline与本次被线性化的nominal、完整方向和问题身份。保留完整N是为了得到同问题的新K/H候选与冻结反馈候选的直接比较；其内部原候选rollout按原最多60/solve、合计≤240另列，FD另列，不能把后面的36当总工作量。每问题只选一个预声明backward方向，使用原已有九个正alpha，≤36额外固定rollout。选方向规则须在看新家族分数前冻结：取本次首个有效backward方向；没有有效方向则该问题不补凑。求解内部仍保留原预算，不增加R/seed/正则化网格。

baseline B必须是本次`NominalTrajectory`选出的`r33_nominal_policy`，已经在本次x0/time/context与冻结r35目标上评分，其比较成本来自对应`r33_nominal_trajectory`；P0仅保留为installed incumbent身份，不默认为B。冻结B完整旧reference states/times、K、H、feedback_scaling、representation和metric来源，H须`metric_valid`，无效时该分支不可部署，不能借用N新H。

源码中B保留旧策略参考时标，而N的k来自当前绝对2ms网格`r32_nominal.times`，两者不能按index直接相加。k取首个有效backward后、任何alpha修改actions前的`action_improvement`；原N是`u += alpha*k`，不得再取负号。用horizon−1个动作时标，176状态对应175动作，不将终端复制行当新自由节点。候选定义为：在原ActionImpl插值得到的B前馈＋原scaled反馈之后、原Clamp和R29Project之前，加`alpha * k_N(t)`。K/reference/H仍分别按B原时标插值，k按本次独立绝对时标和原动作插值类型求值，末端按原ActionImpl夹持；有效horizon和边界行为在schema保存，不超范围任意外推。alpha=0直接使用完整B与已有评分，0新增rollout。

当前schema32/33不能表示这第二时间网格。须给完整policy增加独立delta times/values/alpha，并同步扩展serialize/import、CopyFrom、Reset、hash和部署求值，保持旧策略默认无delta兼容；不得只加未归档runtime offset。新policy连同两个时标的对象必须可独立恢复。不能以regrid旧reference/K/H冒充完全冻结，也不能把accepted rollout覆盖反馈reference。zero-delta时整个动作映射应保持B，在已有实际state/time节点0积分核验即可，不重跑旧零端点物理诊断。

在线生命周期必须一并明确：B可能已带旧delta，新proposal是`delta_B(t)+alpha*k_N(t)`，不能覆盖旧增量；先物化为实际存储曲线再评分、hash、序列化和部署。若存累计已缩放delta值，运行时只加一次插值结果，alpha仅作本次增量provenance，不能二次乘alpha。可按分段线性union-knot合并，但不假称浮点合并天然逐位等价；alpha0直接保留B而不合并。限定曲线到当前可用horizon并保留必要边界语义，避免无界递归base-policy链或历史节点无限增长。metric_valid、metric_regularization/type及H必须真实序列化往返，不能沿用旧import无条件metric_valid=true来掩盖缺失度量。

下一次`NominalTrajectory/FeedbackRollouts`复制完整delta；nominal反馈scale只缩放Kδ、不缩放前馈delta；`R35Freeze`须经同一ActionImpl包含delta。当N工作对象的trajectory被本次native rollout替换用于线性化时，清除**该N工作对象**旧delta，因为新nominal actual actions已经包含它；此前保存的B仍完整保留delta。新冻结家族只走ActionImpl/Rollout。ActionIndex显式拒绝带delta对象，除非实现了正确的独立绝对query-time参数；不能拿旧reference.times[index]冒充当前时间。

所有候选用同一x0/model/context、同epoch ubar/F/R、原native连续ActionImpl部署语义与相同风险规则评分；与该baseline在同路径的成本比较。baseline已有同路径证据可复用，确需补算每问题最多一条，≤4额外baseline rollout。原N方向的dV不自动适用于该不同反馈家族；预期下降/ratio不可用必须为null，不伪造ratio，不据此调整damping。沿用P既有null语义及r34 `R34Better`：metric有效、有限性、风险顺序、真实成本降幅/roundoff预算和alpha0不安装保持。当前源码中dV用于ratio/regularization，并非真实严格接纳的必要条件；无有效成本ratio时保持mu/rate，原trunk-restoration的violation-progress规则仍单列保留，不能暗中放宽strict或一刀禁止全部原正则化行为。

这是本项目待检验proposal，不冒称标准DDP更新或文献已验证方案；none/load/constant是输入参考分支，不能和此反馈冻结机制混为同一维度。它是否与历史已有实验重复，执行前用源码/归档逐项核identity，不因新命名重跑旧机制。

如果A已证一个实际影响控制的执行/参考接入错误，可优先最小修复该错误并代替这个候选家族；不在同一正式版本同时改时间、反馈、目标、步态和PPO。若accepted计划本身已恶化且本家族也无改进，报告负结果与明确证据，不能盲目提高高度权重或换多套算法追成功。

## 6. 合并判断、正式实现与原任务

两CPU线的结论合并后，只选**一个**新控制机制、一个正式版本。若需要r37，创建真实spec/patch/build/源码与库SHA，接入prepare/runtime/policy/records/evaluation合同和所有build分支；`--build`命名不能代替接入。有限工具的offline INPUT_POLICY/CONTEXT/REFERENCE、SCORE_ONLY等环境不得进入正常原任务。

候选若进入在线求解，必须在原有限候选预算中清楚分配/替换对应proposal，标明实际候选数和FD成本；不能暗中叠加大候选池然后宣称同预算。优先替换对应正alpha家族，保留原incumbent/nominal回退和strict规则。完整策略部署必须与有限评分同义，不能评价冻结反馈却安装整包新K/H。

对拟部署策略从相同新事件问题做≤4条额外20ms真实执行链核对（≤400子步）；保存原始native动作、RPC返回和实际held，验证本来就应相同的量，并量化剩余舍入差。不要求bitwise消除所有物理误差，也不把短窗通过当任务资格。

满足以下可检验条件后自主进入原A/B，不再次请求许可：正常入口/原约束/配置与build正确；至少一个新代表问题形成按原规则实际可接纳且行为不同的策略，或证实并修正一个实际影响动作的执行错误；短部署无未解决实现错误；变化来源明确。**不要求两个流先局部恢复成功，不要求先完成全程，不等donor、不等GPU，也不要求缺失历史全状态。**

若只有已知的极小时间舍入变化，所有赢家/主要事件行为仍不变，或拟议策略无法通过原接纳，不重复原A/B。此时交付有明确因果界限的有限负结果和下一机制判断；不能把未启动写成失败或把本计划写成控制已修好。

### 原评价

先推有限结果、spec/patch/真实build、记录schema和实际phase，再A/B各一次evaluation-only原E30。A/B固定各自W44 seed1/2的`checkpoint-014737.pt`、完整resume ep00428/ep00417，actor/critic/optimizer/norm/RNG/fork/lineage保留，tail0/excluded145。权重固定、observer/material history因果更新；新PPO0/0，不自动执行15249/59。

用`scripts/run_s1_go2_r12.py`单次evaluation入口，显式attempt/evaluation-start-unix/checkpoint/level/seed/workers/build/contract/condition nominal，不用要求新训练的job-dir。两个流同版，不混控制机制，不补同配置r36基线；W48已有同seed/checkpoint对照。

| 阶段 | 原资格 |
|---|---|
| E30 | 独立满30s，合法实际材料去程、原3m目标容差及转向；不新增x≥3.0硬门 |
| E60 | 本case前级通过后，独立满60s，一次合法往返、回家连续稳定2s |
| E300 | 本case前级通过后，独立5合法往返、270s路线期限、完整观察300s |
| E600 | 本case前级通过后，独立10合法往返、540s路线期限、完整观察600s |

某case真实失败就结束并阻塞本case后级，另一case可继续。不要换seed、重复同配置追成功、拼接轨迹或把前缀当独立资格。技术中断保留独立attempt；仅Actor不能恢复完整plant/材料/observer/MPC/任务，不伪造中途续接。

继续W48每call预测配对与事件去重，补回父类`trunk_restoration`事件。拟议新原回合顺带保存可恢复`mjSTATE_INTEGRATION`与外部callback/material/任务状态，注明当前API覆盖范围；不再额外开原回合只为记录。该attempt首次正常规划后、首个held推进前检查实际字段写出；startup无上一预测允许unavailable。

## 7. 第三线Task C：GPU首hold辨识与发布尾项

### 7.1 小实验解决一个具体缺口

W48重复组已确认首2ms分叉，不再重复24world×350ms，也不再只读取相同极值。历史W46 GPU实现先真实warmup step，再部分load qpos/qvel/userdata/time/ctrl并清qacc_warmstart；其余Data初始化状态尚未逐项核定。这是可检验线索，不是已证GPU错误根因。

选一个已有明确重复身份的B158 nominal双world（历史索引18/21），共同held只cast一次为FP32后复制同一bytes，不再闭环求新动作。F1从合法完整初始化S0运行首hold；L恢复F1第一个.2ms子步后的S1，再只执行旧load的部分字段恢复；F2完整恢复同一S0再次运行首hold。不能把推进10步后的残留当一步warmup。三个条件彼此独立，每hold10个.2ms子步，**核心≤60 GPU world-step，必要预热≤2 world-step另计**。snapshot须包含API可写的数值状态、外力/act/userdata/mocap、warmstart及接触/solver相关初始化；若API不支持完整恢复则标部分覆盖，不能假称full restore。

各子步同步后读回q/v/act/time、qacc/qacc_warmstart、输入、外力、接触pair/distance/normal force、solver迭代/残差或可取得KKT字段；缺字段据实记录。固定MuJoCo/Warp/JAX版本、模型/编译选项、精度、solver/积分器/迭代设置、world映射和初始hash。FP64动作计算不等于GPU物理引擎FP64。

对照要同时报告组内两world差和同world复位前后差，第一差在哪个子步/字段。所有测时先明确同步，编译/预热/初始化/传输/物理分开；不把首hold称吞吐benchmark。若fresh和restore一致而旧partial不同，支持该输入下reset路径解释；若fresh仍不同但可重复，指向可重复world/layout或未对齐隐藏状态；同world F1/F2不一致则核恢复覆盖、持久缓冲和非确定性。不直接宣布纯浮点或随机性。必要输入在子步0即不等时停止积分，记初始化技术问题；辅助solver residual/KKT未暴露不自动取消实验。

仅当初始held不等，或共享首分叉状态缺真实native `Policy::Action`与GPU分阶段对照，允许一次≤8个去重身份、0physics动作批次；不再仅比较Python公式与GPU公式。保留`equivalent_native_ranking=false`、`full_native_cost_implemented_on_gpu=false`、`original_task_qualified=false`。详见[Task C实现细则](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W48-final-review/w49-task-c-design.md)。

此实验只解决同GPU重复一致性，不授予native等价资格。GPU结果不参与正式native候选排序。主机C总2CPU；当前资源不允许则先完成CPU主线，记录未运行原因，不抢占无关作业。

### 7.2 交付与R8

科学finite/A/B的13资产已核uploaded。旧技术`w48/delivery/w48-a`和`w48-b`各一ZIP未发布，A6/B7非空源尚未被后续hash覆盖。只在D57按既有manifest续传，先查远端是否已存在，核URL/size/digest后更新状态；不重打包、不重实验，不阻塞控制主线。

R8单个科学输入如本地仍缺，可独立尝试恢复已知asset681937433，28,100,505B，SHA`079f57e1ebad260c9cdac842dc089012e29cb46992ab64d2c31cd9d380b61625`，核schema/time/phase。本轮不加donor solve、不批量下载、不重跑R8；EOF或0bytes是输入可用性，不是donor无效。服务器只用已有授权网络路径，不借用户电脑VPN/Clash/SSH转发或未授权直连，不把凭据写入Git/日志。

## 8. 并行、工作量与后台

| 时段 | CPU A额度 | CPU B额度 | C额度 | 系统/主控/发布 |
|---|---:|---:|---:|---:|
| 有限机制工作 | 10 | 10 | 总host2，GPU按实测余量 | 约3 |
| 原A/B获准后 | A原任务10 | B原任务10 | 轻分析/上传或独立小GPU检查 | 约3 |

先核实际可分配25核/96GB及其它作业。全进程树限制，线程库各1；build-j6先完成，不与两条重型solve争核。可写upstream/build和job输出隔离，共享原件只读。两个原回合之外不再挂第三个重型native solve。GPU关闭XLA预分配、保留JAXx64，显存准入以实测峰值和其它进程余量为准，不以96GB“必须占满”倒推任务。

本轮上界：A≤12×20ms/1200子步；B≤4新问题N solve＋36新候选＋必要4baseline固定rollout；部署核对≤4×20ms/400子步；C≤60+2 GPU world-step。求解内部rollout、transition/risk FD另按实际逐项计数，不把固定rollout上界当全部物理成本。没有有效方向/必要变体则少跑，不凑额度。正式阶段至多一个新版本，两流各一次E30；各自通过才后级。

**所有新后台`deadline=null`、`wall_deadline_enabled=false`，省略`deadline-unix`，无固定墙钟完成时刻或到点kill。** 有限样本/子步/solve预算与原物理时长不是墙钟deadline；单次网络timeout和工具等待timeout不传播为后台期限。无无限重试。对话结束不杀后台，running不改completed；按真实clock/日志增长检查状态，必要操作只针对精确PID。

## 9. 输出、完成与不重复归档

在W49独立路径保存身份、原始输入manifest、同态动作差、短窗各臂、候选策略/J分项/风险、GPU逐子步首差、真实计数和未运行原因。紧凑Git先推，已生成必要原件/视频由D57增量Release直传。回执丢失先核远端，不重复上传；Git/index短写串行、基于最新main合并，不强推覆盖并发。

完成报告分列：已修技术错误、已证/未证机制、实际改变动作、原阶段资格、真实PPO/FD/rollout/plant/GPU工作量、资源使用与缺项。原任务未通过明确未通过，不以文件齐全或数值一致冒充稳定成功。

**GitHub核验commit/URL/size/digest后，不在用户电脑额外下载、打包、镜像或归档；服务器原件/checkpoint/resume/视频及上传副本不自动删除。**
