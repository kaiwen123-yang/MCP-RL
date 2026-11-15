# S1 W50：刷新滚动反馈参考，恢复支撑与前进

更新：2026-11-15（Asia/Shanghai）。审查结果基点：`de6606dc8363e8ab1c816fa682cace752a06318a`；上轮计划：`b142001624cfea5877c57e63f45e0f29c9dc4735`。

**状态：implemented_finite_negative；原A/B未运行。** W50/S5-r38/H 已实现并完成有限工作：四个W49问题76次full rollout，36个正候选中10个strict改善；两个早期赢家仍下沉/后退，未达到原E30推进依据。另两条40ms消耗400子步和2次再规划40rollout，但bridge同SONAME库复用造成ABI失配，修复loader后未重跑。原A/B E30均未运行，后级blocked；新增PPO0/0，新增GPU积分0。

入口：[W49 最终审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/README.md) · [启动提示词](S1_W50_START_PROMPT.md) · [项目记忆](TERRAIN_MEMORY_PLAN.md) · [执行规则](../AGENTS.md) · [机器交接](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/handoff.json)。

## 实际执行与结束

[W50实际索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w50/phase.json)、[四例结果](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w50/task-b.json)、[完整策略构造](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w50/policy-construction.json)、[短部署技术限制](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w50/deployment.json)是本轮实际证据。总计6次solve、116full rollout、23112 transition FD、4746 risk FD、76650次零物理raw-center求值；400部署子步与FD另列。正式控制器build已完成62个native/service编译单元。36个四例正候选中10个strict改善，早期赢家未形成有用承重/前进改善，原任务各级blocked。没有额外变体或重跑。

以下为本轮冻结设计及解释，状态不得覆盖上述实际结果。发布完成后只保留服务器原件，任何新科学执行需要新指令。

## 1. 要解决的实际问题

W49 取得了真实的行为改变：旧 r36 的 A26.432 s 躯干触地、B11.638 s 跌倒，变成两条 r37 都自然观察满 30 s。然而 A/B 最低 z 仅 .123298/.130666 m，末 z .138718/.142818 m，非足受载早在 4.6232/5.0216 s 出现。A 最大 x .349667 m，B1.020530 m；材料合法穿越、到达远端、合法往返均未完成。A 全程没有材料加载，B 足端首次加载12.1866 s，机体西侧入条带20.064 s。不能把持续30 s称作稳定行走。

原指标不变：body_fall 为 z<.12 m 或绝对 roll/pitch>1.2 rad；允许腿部部分几何接地，所以非足诊断与 invalid_contact 不相同。不增加或降低终止阈值以制造成功。

主要新线索来自实际源码：

1. r37 的 K 路径在首个有效 backward 后，复制 `w49_base=r33_nominal_policy`，只添加本次 k 前馈曲线。其 reference states/actions/times、K/H 没有随新问题安装更新；nominal 分支也只改变 feedback scale。
2. 初始有效 N bootstrap 之后，这个分支持续继承旧参考，查询超过旧网格时沿末端插值。完整策略安装 ID/generation_time 递增，并不等于反馈参考和 H 生成时刻变新。`metric_source_plan_id` 原来直接复用安装 ID，也不能作为 H 已更新的证据。
3. 已读 Git 事件子集符合该生命周期；实际 reference 末时刻、K/H 数组来源和过期时长必须从 D57 留存 NPZ 读出。不要把源码推导出的约29.65 s过期写成已经读到的实测。
4. A/B 各1521次调用，其中802/1393次正步安装均是同问题真实 J 下降；末支撑高度项仍占 J 的97.27%/97.18%。所以不是简单“优化器不下降”。小 Position 项也不能证明加权不足：r26 实际追踪的是局部 head reference，mission 内 x=3不是直接残差目标。

**W50 主问题：保留旧完整策略作比较基线，在当前实际时域上刷新参考与反馈，能否避免或恢复低姿态，并让前进发生？** 本轮不同时修改成本、步态、horizon、PPO或物理。

### 用户补充视频后的优先级修正

[视频逐段审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/video-review.md)确认B在约4.4–5s反复深蹲/顶起，7–8s后长期低伏、腿部折叠/外伸，后半呈贴地挪动，不能称正常步态。主目标前移为起步后维持有效足部承重、避免最早步态失效并推进；低姿态恢复是补充。存活、位移、步态质量分开报告，不能再用跑满30s概括稳定性进步。

## 2. 长期目标和本轮边界

研究目标仍是同一世界地块跨前足、后足和返程的持久材料状态，让未来支撑需求影响当前加载、落点与重访。原链为因果PANN→658D观察→2Hz scalar cap(.18–.25m/s)→逻辑50Hz全身MPC→500Hz时变关节目标→PD60/5→5000Hz真实物理。

当前可以推进有辨别力的控制实验；不把新增PPO样本、长时间驻留、旧刚地资格或短部署当作完整材料任务证据。完整稳定路线之后，才推进跨足/返程曝光、RAW/PANN和冻结/打乱历史对照、候选内材料状态传播与必要动作权限，再做泛化/Go2/NX。

原Go2/TM02质量15.206408 kg、PD60/5、每腿23.7/23.7/45.43 Nm、材料主应变.02、条带x[.8,2.2]/|y|≤.6、原接触/跌倒/合法路线、2Hz/.45/.005 C2 gait、.35s horizon、2ms hold/10×.2ms子步都保留。短腾空不是新失败条件。native风险优先/真实J/roundoff预算和严格接纳保持原定义。

本轮新PPO=0 samples/0 updates；donor=0。R8恢复、GPU路径、旧论文复现不作为CPU主线的前置条件。结束 W48 identity-only、旧时间FP32、旧四fixture原样重跑的支线；这些已有结论。

## 3. 先读取现有原件，不再用原任务补档

执行端先核最新main、本地修改、已有W50 job/tmux/PID/真实physical clock。已存在有效同身份工作就续接；W49已结束，不重开W45–W49或G5、不宽前缀kill、不动无关作业。默认CPU树 `/root/autodl-tmp/MCP-RL-w42`，GPU树 `/root/autodl-tmp/MCP-RL`，环境 `work/envs/d57` 与 `work/envs/w41-mjwarp`，以现场有效路径为准。

优先在D57读取已留存对象；需要时仅按manifest恢复必要单输入，使用已授权服务器网络。没有理由在用户电脑下载全套ZIP或借VPN/Clash/SSH转发。

CPU A的首段是零新增积分的数据分析，与CPU B编码并行。读取全量已有boundary/call/held记录及下列四个W49问题，不再使用W48 A213/B176等问题作为新证据：

| 目的 | W49流 / call | 实际时刻 | 已有身体状态 |
|---|---|---:|---|
| 下沉早期 | A /237 |4.32 s|z≈.290605 m；随后4.4 s明显下降|
| 下沉早期 | B /236 |4.30 s|z≈.267324 m|
| 低姿态恢复 | A /328 |6.14 s|z≈.141877 m；接近非进展事件|
| 加载前恢复 | B /625 |12.08 s|z≈.131105 m；足端首材料加载之前|

call/time必须由对应实际attempt与对象核对；浮点时刻按原记录，不凭文件名猜。四例是有限问题集，不是四个新增原episode。必要输入缺失时只标该例不可用，不换大量新fixture补凑。

每例关联已有P0、安装policy/accepted、reference、r35同epoch目标、任务参数/weights/gait/context、实际held。历史best nominal B的完整rollout未必另存；先读其decision来源/scale，新的B由本轮已预算的10个nominal screen得到，不能因缺这份历史大数组阻塞B线。**P0不默认等于B，accepted不默认等于feedback reference。** 直接列出reference start/end、query是否越界、K/H生成来源、安装来源、nominal scale、delta范围；用数值对象与时刻区分，不新建全仓哈希门禁。

A的读取窗口从已有2.002s Stand→outbound切换开始，核实际private gait/phase、nominal feet reference及脚顺序、相位是否持续演进、旧reference是否仍来自initial Stand，以及真实Fz/接触时长/摆动落脚和支撑交换。配置Gait=Trot不等于真实步态成立；.005m抬脚小也不能解释长期低伏。历史reference/K/H若已刷新则据实否定冻结假说，按读到的具体问题调整唯一修复，不机械套预设。

同时追踪首次下沉前后的：身体高度/垂直速度/姿态、真实足与非足承重、预测承重、PD饱和/投影、实际前进、完整J分项、下一20ms同策略预测—执行误差。r26的精确userdata84:96必须读取；不要把mission.body_target当native Position目标。每个已保存accepted的.35s轨迹可用于说明其自身是否计划恢复，不能与之后多次重规划的实轨直接配成固定策略误差。

W49新记录仅有integration303维及部分外部态，缺若干history/legal_route/对象引用、native缓存、actor/RNG，且没有通用restore函数。本轮可以共同重建明确的native predictor；不能无依据宣称历史plant exact replay或中途无缝恢复。所有缺失如实列明，不填零伪造。

## 4. 唯一主候选：当前时域重参考，并随同一个alpha更新反馈

### 4.1 为什么不是直接恢复旧N

W49有限实验中，首个有效k配旧完整反馈能在4个问题上改善；其中旧完整N的正候选仍可能全部差于baseline。旧N在任意正alpha下整包换成新K/H，和仅缩小前馈步长不同。因此本轮既不永久冻结，也不盲目把所有正步整包换成新反馈。

拟定一个待验证的候选族：**将B在当前预测轨迹上的完整原始控制中心重新表示，再用同一个原alpha同时转移前馈、反馈和H。** 这不是已证定理或论文原样移植；四个实际问题的native评分及后续原任务决定它是否有用。

### 4.2 数学和对象语义

设B为当前问题原路径已经评分的完整baseline。`c_B(x,t)`是其加完旧delta、但尚未Clamp/R29Project的原始控制中心；已包含B的实际feedback scale。`H_B(t)`为B按自己的时标插值的有效投影度量。

在B本次native rollout的当前网格上，以状态 `xbar_i` 为新参考。176个状态到t0+.35s，175个自由动作的网格只到t0+.348s；终端行沿原规则，不另造terminal k。

- `b_i = c_B(xbar_i,t_i)`；不得把已clip/QP的实际actions冒充raw center。
- `L_B,i = ∂c_B(xbar_i⊕ξ,t_i)/∂ξ | ξ=0`，坐标为当前MuJoCo切空间。
- 从同一次首个有效N backward同时捕获 `k_N,i / K_N,i / H_N,i`、linearization轨迹、μ、rate、active与问题身份。
- 对原九个正alpha，构造 `u_ff,i=b_i+alpha*k_N,i`，`K_alpha,i=(1-alpha)*L_B,i+alpha*K_N,i`，`H_alpha,i=(1-alpha)*H_B(t_i)+alpha*H_N,i`。
- 完整candidate以当前`xbar/t`作reference，`feedback_scaling=1`，representation沿用原linear，旧delta已吸入b，清空该candidate的delta；query时只进行一次原Clamp和H-QP。末状态与最后K/H重复行沿用原176状态/175动作节点语义，不新增控制节点。base_Qu/base_Qxu/active只保留其实际backward来源，不冒充混合策略的新导数。

`L_B`必须吸收B已安装scale；`K_N`按原ActionRollouts实际scale=1捕获，不能再乘旧scale/alpha。旋转坐标转换须正确：使用native StateDiff与MuJoCo位置切空间积分，解析链式变换或固定原数值尺度的零物理中心差分均可；不能直接把旧参考下的K数组当作新参考下Jacobian。只计算必要旋转映射/矩阵，记录额外零积分查询数，不伪装为零工作量、不做epsilon网格。

**alpha=0直接返回B的deep copy，不做rebase、不换参考/K/H、不清其delta，也不新增零点rollout或安装。** 对alpha>0，重参考仅局部一阶保持旧反馈行为，不宣称全状态空间恒等或alpha趋零全局连续。重新评分正是为了检验这项近似。对于非有限/无效metric候选按既有无效路径拒绝，不放宽约束。

`H_alpha`若两端有效SPD，凸组合可保持该性质；实际仍使用原metric有效性与QP路径。混合H没有单一“生成μ”，需记录两端来源/μ与alpha，不用安装ID冒充metric来源。完整数组和来源由安装对象随CopyFrom/Reset/Save/Import保存；不再用不可回放的临时offset或仅更新generation_time。

### 4.3 捕获时序与严格评分

在**首个有效backward、任何alpha写入候选之前**一次性冻结k/K/H/μ/active。W49的部分有限问题先跑完N的多个失败方向才追加最初k，末μ可为1024而k来自μ1；W50不能从末工作对象拼出不属于同一次backward的字段。

有限与线上使用同一个提案入口：本次nominal→首有效方向→九个新候选→原native严格评分。有限模式不再先完整N搜索后额外追加新家族。新dV不适用于该重参数化时expected/ratio=null，不使用旧N预测下降比调μ/rate；继承既有K的风险进展更新语义。没有有效方向按既有失败/保留baseline语义返回，不无限重试。

每个正候选经过实际连续ActionImpl→原10子步native predictor→原完整J和风险→R34Better，与同epoch B比较。排序对象、短部署对象、正式安装对象必须为同一个完整policy；不能一边score新反馈一边部署旧反馈。原alpha0不安装、ties保留旧解规则不变。

线上每个新state/time/reference epoch都尝试当前时域的候选；更新成功才安装，失败则保留原有效baseline并如实报告reference年龄。**不得为刷新年龄强制安装更差策略。** 当前epoch重新锚定目标是已有r26语义；本轮不改其参数或route定义。报告正常支撑推进与躯干垂直振荡/非足承重，但不新增硬对角接触门、高度阈值或非足一碰即停规则；原短腾空和资格保留。

## 5. 有限工作量、针对性部署与转入原任务

没有墙钟截止；以下约束是科学比较的有限工作量，不是到点kill。

| 工作 | 最大新增量 | 含义 |
|---|---:|---|
| A：原件/生命周期/接触/成本分析 |0 physics /0 solve|充分利用现有原件，不另开原任务补记录|
| B：四W49问题首有效方向 |4 solve|各保持原有限backward规则，不先跑完整N候选搜索|
| 新重参考候选 |4×9=36个.35s rollout|原alpha表，一个参数，无alpha×beta网格|
| B必要独立baseline确认 |≤4个.35s rollout|已有同路径baseline可复用，不重复积分|
| 有限solve总rollout |≤80|同epoch导入r35参考时：nominal10+新9+必要baseline1，四例总数；FD另列|
| 跨epoch实际RPC短部署 |≤2条40ms、≤400个.2ms子步|A/B各取最早可接纳候选；不足两条不补凑|
| 短部署中+20ms再规划 |≤2 solve、≤40个full rollout|新epoch通常r35 freeze1+nominal10+新9；独立列FD|
| C：旧GPU数据与源码分析 |默认0新GPU world-step|先读已有all-exposed-substeps，不重复F1/L/F2|
| C有具体修复时的最小区分探针 |≤20 GPU world-step|一个旧状态、同held双world×10步；只在明确代码变化后做一次，预热从中计，不新增吞吐benchmark|
| PPO / donor /旧原回合重跑 |0 /0 /0|本轮主线不需要|

上述最大120个full native rollout不包含400个部署子步，也不包含FD内的积分，必须按实际语义另计；不能把120当全部物理工作。原件同epochr35输入缺失时，不用新冻结目标冒充旧问题：标明共同重建的新身份，并在同样总量内比较。每例19+I_freeze+I_extra_baseline≤20；需新Freeze的一例复用nominal screening的B score，不另加baseline，因此四例仍≤80。不要追加变体。

两条40ms部署是native predictor上的实际RPC/query/hold/重规划路径诊断，材料proxy范围明确，不称历史plant回放。起点直接安装对应完整winner，20ms处再规划一次以检查跨epoch生命周期；当前没有运行中修改这些env的RPC setter；须实现明确的一次性诊断安装入口，首次安装后消费并禁用QUERY_ONLY、INPUT_POLICY、R34_CONTEXT、R35_REFERENCE，或使用第二个Agent以当前state/gait/context与完整policy建立正常规划。不能每次OptimizePolicy又导回旧private gait，旧`MJPC_R35_REFERENCE`也不可带入新时刻。新epoch用正常R35Freeze生成参考。homotopy赢家安装时记录当前reference与混合K/H来源；nominal-scale赢家或retained B仍如实保留旧来源，不能从policy_updated推断刷新。

**推进依据是实质改动，而非要求诊断先完成整条路线。** 以A237/B236两个早期代表问题为首要依据：至少一个有原strict可接纳、动作行为改变且减少异常下沉/改善足部承重或有效推进的候选；或A发现并修复确实影响早期步态的具体执行错误。不能仅凭A328/B625晚期低伏挪得更快作为新的原回合推进理由。完整策略真实接口、原约束和跨epoch来源正确后，可自主冻结一个正式版本，A/B各一次原E30。不能只凭极小roundoff、metadata变化、总J稍降却明显加重坍低，就重复原任务。

如果四例均无有用可接纳候选，或问题只改名字而行为不变，交付有限负结果及候选为何被拒绝。下一方向以证据区分：可行前进候选在原J下更好但未找到→搜索/初值；可行候选被目标排序持续压过→单项参考/成本结构；预测支撑交换本身不可行→受约束接触/运动参考；参考先失效→先修生命周期。不得为求原任务成功临时叠加成本、horizon、步态或训练。

## 6. 构建、原评价及结果记录

在独立可写build树按实际r37完整配置继承后overlay。新增policy/class字段会改变ABI，必须正常重编所有依赖该header的service静态库和server，不复用W49曾失配的旧archive。先序列化原带sensor模型，再载入诊断用sensor-free predictor，保持W49已修复的saveLastXML顺序。

更新真实spec/patch/build及prepare/runtime/policy/records/执行合同入口，确保正常在线模式。只做与此次变化直接相关的验证；不新增全仓hash、mock、shape/import/smoke或没有信息增量的防御性测试轮。构建身份/版本与发布核对是必要交付证据，不升级成反复卡住实验的门禁。

正式版本只一个主控制机制，预计S5-r38；不能A/B各用不同机制。A/B仍各自W44 seed1/2、checkpoint-014737、resume ep00428/ep00417，完整actor/critic/optimizer/norm/RNG/祖先保留，各14737 transitions/update57、tail0/excluded145。评价冻结权重但observer/history因果更新；新PPO0/0。使用单次evaluation入口，不用自动启动新训练的job-dir入口。

每个case独立执行：

- E30：30s内完成原合法材料去程、到达3m远端原容差并完整观察；转向/返程继续记录。仅存活30s不通过。
- E60：原完整合法往返及回家连续稳定2s，观察满60s。
- E300/E600：分别270/540s内5/10合法往返，完整观察300/600s。
- 本case前一级通过才后一级；失败结束本case后级blocked，另一case继续。固定seed、同版本；无追成功重跑、拼接资格或把blocked当运行失败。技术中断单列attempt，未具备完整恢复时不冒充原连续回合接续。

记录沿用W49事件去重，并增加与本机制直接相关的字段：reference生成/末时刻、K与H来源、old/new μ、alpha、reference是否覆盖query、真实安装时间和reference年龄、当前完整策略与accepted各自身份。把安装ID、反馈来源和投影度量来源分开。完整risk/J、实际前进/承重/姿态、same-time预测误差、失败原因、实际计数保留。

修正汇总语义：`internal_stagnation=0`只表示旧枚举计数为零。新增报告no-improving-candidate、incumbent-retained、backward-failed、有效方向但无更新，以及真实任务非进展；保留旧字段供历史解释，不篡改W49结果。build配置、有限诊断和最终runtime身份各自标时点，最终phase汇总不再留下N/not_started冒充当前状态。

## 7. 双CPU主线与第三GPU线

现场复核25CPU配额和96GB显存实际余量，不仅看宿主CPU数。CPU A 0–9，CPU B10–19，C20–21，22–24留主控/发布/系统；根据现有任务调整实际cpuset，保持总量。BLAS/OMP等线程库1、全进程树归属明确。先完成最多-j6构建，不与两条10-worker重求解抢配额。

A读取/分析与B实现可并行；四个独立有限问题可分给两CPU线。两线合成同一正式版本后，A/B原任务并行，不再外挂第三个重native solve。W49已实证两原进程重叠79.412min；此分配可复用，但要记录实际重叠和CPU/GPU利用，不能由affinity宣称满载。

C先读取W49现有 `all-exposed-substeps.npz`（1,873,543B，3×11边界）和锁定MuJoCo-Warp4.0.0/Warp1.19.0源码：按active contact/nefc/world有效布局比较，排除padding，沿constraint assembly/solver/reduction/callback写集定位首差。首步ctrl/actuator/passive/外力一致，排序后contact在10步均一致，而constraint/qacc在首步开始不同，已足够缩小范围；不再重复完整同样F1/L/F2。

只有定位到一个具体可修复路径，才用表中≤20step一次区分探针。不能因有1 ULP差就认定工程不可用，也不要求逐位一致作为总门；当前full native cost没有移植，GPU仍不决定正式排序。

有可批处理的现有K/H矩阵和状态时，C可用GPU完成零物理的Jacobian映射、谱/条件性和候选中心批分析，为A/B提供证据；只做有明确输出的问题，不以占满96GB为目标。后续GPU近似生成proposal、CPU原native最终评分是可行的架构方向；本轮不额外并入一个新采样控制器或PPO训练。C不阻塞CPU，完成后自然结束，不为保持“三任务”空转。

## 8. 后台和交付约定

**所有新后台deadline=null、wall_deadline_enabled=false；省略deadline-unix；不设固定墙钟完成时刻，不到点kill，不因对话结束停止后台。** 保留上面的有限工作量、原评价物理阶段、技术错误/用户停止及单次网络timeout；没有无限重试。工具等待时长不是后台任务期限。运行中如实写running，不提前写completed。

紧凑源码/配置/计划/结果先推GitHub，必要原件/视频由D57增量Release直传；回执丢失先查远端，Git短写串行合并最新main，不强推覆盖。W49本轮10ZIP、4视频和W48两技术ZIP均已存在，A后处理恢复已完成，无需再续传原件或实验。

**GitHub提交与资产核验后，不在用户电脑额外下载、复制、打包、镜像或归档。服务器原件、checkpoint/resume、视频及上传副本不自动删除。** 不把保留服务器科研输入与电脑归档混为一谈。结束交付给真实commit、资产位置、未过/blocked原因和准确续接点。

## 9. 论文怎样支撑本轮决策

[本轮一手文献](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/literature.md)新增BiConMP、CAFE-MPC、Bilevel gait generation、Layered MPC。它们共同提示要随实际状态和时间维护可执行参考及未来接触；CAFE-MPC对shift/append和impact阶段初始化尤其具体。它们没有验证本项目的homotopy公式，也没有证明旧反馈冻结就是唯一原因。

先检验生命周期这一直接源码问题。只有候选证据指向目标、支撑交换或未来可行性，才考虑一个新的参考/接触权限/终端延续方案；不同时搬入整套外部控制器。当前.35s在2Hz下为.7周期，只能提出时域假说，不能据此宣布horizon一定不足。
