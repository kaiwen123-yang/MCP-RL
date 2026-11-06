# S1：W38 / r27复核与W39接触执行修订计划

## 当前交接（执行后只更新这6行）

- 当前任务：W39_CONTACT_EXECUTION_REPAIR_PARALLEL_JOINT / S5-r28两流训练与末端评价已结束；训练各新增1024/4至8593/33、tail0，原8 workers及预算保持，无重启或重跑。
- w39-a：job-20261105T032354Z-w39-a，training_ended；新增1024/4，累计8593/33，tail=0、excluded=145；评价=completed，紧凑Git=published。
- w39-b：job-20261105T032354Z-w39-b，training_ended；新增1024/4，累计8593/33，tail=0、excluded=145；评价=completed，紧凑Git=published。
- 任务能力：A末次update33固定策略E30在10.476s跌倒，B在17.592s跌倒；两流E60均blocked_by_predecessor；训练与评价合法去程/返程/往返均0，尚未形成合格通行基线。
- 交付：45a0069/b5d408d已发布紧凑结果；原始诊断A492/B484项约7.68GB，按用户授权切到服务器独立Clash、总并发5文件(A3/B2)。实际短窗合计吞吐约6.79MB/s，尚待manifest全量确认；原截止北京时间2026-11-06 22:35:33不变，3个未完成远端占位项改名保留，自动删除0。
- 续接位置：autodl-d57:/root/autodl-tmp/MCP-RL；diag-job-20261105T032354Z-w39-a/b仅传保存原件，训练/评价均已结束。各jobs/<job>/diagnostics/evidence.json记录上传状态，checkpoint-008593及完整resume、原始数组留D57供用户备份后手动删除。

## 1. 长期目标、当前阶段与阅读边界

长期问题仍是：**同一个世界坐标地块被前足、后足及返程反复踩踏时，怎样利用可恢复材料的持久状态，让未来支撑需求约束当前加载、落点和重访时间？** 原始路线见[冻结计划](https://github.com/kaiwen123-yang/MCP-RL/blob/4d0af25c7bcea620c7e8c9a4f5b103c4956a3fda/docs/TERRAIN_MEMORY_PLAN.md)。先建立完整Go2与受控TM02材料闭环，再证明记忆的因果收益、泛化、NX推理和实机；IROS为近期目标，T-RO为更高目标。

现有链路是冻结因果PANN观察器 → PPO的2Hz标量速度cap → 全身MuJoCo iLQG → TV反馈 → PD → Go2/真实TM02 → 观测。冻结的是PANN参数，世界地块状态仍在更新；PPO训练Actor/Critic。MPC候选尚未传播完整PANN/TM02历史，标量cap也不是跨足加载分配或终端重访价值。

| 层次 | 已形成的证据 | 尚未完成 |
|---|---|---|
| R1控制基础 | 全部正负结果，固定2.4Hz与adaptive强对照 | 自适应全面优势未成立，不重凑成功率 |
| R2材料辨识 | 三臂60模型、600工况/700历史评价；B ideal供当前观察器 | noisy恢复/重访仍有负结果；预测准确不能替代控制收益 |
| S0/早期S1 | 世界地块持久历史、相容映射与局部决策证据 | 不能替代自由机器人完整通行 |
| R8刚地 | S3九原任务9/9；独立连续300s任务3/3 | 不迁移为材料或r27资格 |
| 当前S1联合工程 | 双分支完整采样、更新、独立末端评价与紧凑发布均实际运行 | 稳定合规材料通行仍未建立 |
| 论文核心 | 研究对象与基础链路清楚 | 跨腿/返程记忆改变动作并产生收益、候选历史传播、终端价值、NX/实机 |

**W39已在9544cfe实现候选躯干可行性优先，并由84f5867记录两流真实采样启动；这次补充只扩展交付和读取已有数据，不重启或热改该实验。** 当前处于“稳定合法联合基线尚未形成”的工程阶段，不是已经进入记忆收益大规模验证。

上一轮全面复核固定依据为[5e00beca](https://github.com/kaiwen123-yang/MCP-RL/commit/5e00beca0896bb5517bab8c1dcc441303e5c2098)。逐文件字节盘点覆盖4511个tracked文件、454382972 bytes，其中334个不能按UTF-8文本阅读；这不是声称语义理解了每个二进制字节。逐条分析全部70个W38新训练JSON、2条E30、2048条cap记录、8次学习更新，阅读实际r27/r26补丁、直接执行/学习/交付源码和历史主线，并检索新的原始论文。当时W38原始rollout/密集数组与planner/action JSONL未进入Git，未声称已读其内容。随后本次新增[61fd022](https://github.com/kaiwen123-yang/MCP-RL/commit/61fd022af3ff8ce1d2a88f42e17639311788e42f)的真实学习小表和E30时序窗，初步复审见5.3；完整原始数据补传与抽取窗口的阅读范围分别记录。

证据文件：[W38统计与逐回合表](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W38.json)、[复算脚本](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/analyze_W38.py)、[控制审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W38-code-audit.md)、[论文阅读记录](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W38-literature.json)。

## 2. W38实际推进了多少

### 2.1 两流完成采样目标，任务能力仍未过关

| 指标 | w38-a | w38-b | 本轮合计/解释 |
|---|---:|---:|---|
| 自然编号217–251的训练回合 | 35 | 35 | 70个不同stream回合 |
| 新transitions / updates | 1024 / 4 | 1024 / 4 | 2048条独特新数据；8个分支更新事件 |
| 跌倒 | 13 | 14 | 27 |
| invalid_contact | 20 | 20 | 40，全部原始reason为trunk geom13 |
| 材料域退出 | 1 | 0 | 1 |
| 目标达到截断 | 1 | 1 | 2，不计自然失败 |
| 真实材料加载回合 | 27 | 26 | 53 |
| 机身进入x≥.8 | 11 | 4 | 15 |
| 进入后有条带内部足端正Fz | 5 | 3 | 8 |
| 首程侧出 | 0 | 1 | 1 |
| 最大x | .975877m | .934916m | 均未到1m |
| 合规去程 / 返程 / 整圈 | 0 / 0 / 0 | 0 / 0 / 0 | 均0 |
| 到远端 / 开始返程 | 0 / 0 | 0 / 0 | 均0 |

两流每条都为7569/29＝7424条已用于完成更新＋145条历史excluded，tail0。W38共同父历史6545只计一次，实验树独特历史为6545+1024+1024=8593；**不存在一条8593/33策略**。PANN真实训练回合累计214+70=284；历史ep13/14零物理步ABI失败另存，RAW的256/1独立。

53个材料加载回合中40个机身从未进入条带；“足尖触到材料”与“机器人在材料上稳定承载通行”相距很远。A的ep219/222虽机身进入，却没有材料加载，因此两种指标也不能互相替代。68次自然失败都至少有一个关节曾达到原力矩上限，66次12个关节均曾达到；这是最大值证据，不是持续饱和时长或首次致因。

与历史相比，W35/r24有73/73加载、72机身进入、最大x3.00755m、1次合规去程；W36/r25为0加载且退行；W37/r26恢复到50/72加载、7进入、最大x.968767m。W38增加了入口尝试数量，但仍未恢复r24曾到达远端的能力。材料域退出减少同时伴随探索深度缩短，不能宣布材料冲击问题已解决。

### 2.2 八次分支更新不等于稳定学习改善

排除各ep251目标截断，比较每流前17与后17个自然回合：

| 指标 | A前→后 | B前→后 |
|---|---|---|
| 平均最大x | .762206→.740864m | .744334→.698654m |
| 存活中位数 | 15.406→14.0054s | 13.6302→14.2994s |
| 机身进入回合 | 5→6 | 4→0 |

小样本且来自同父策略，不能宣称统计显著恶化；但没有证据支持已经学会稳定通行。loss A约.225→.135、B约.224→.158，只说明当前优化目标拟合变化。两流cap均值约.21445/.21470m/s，范围约.18328–.24445/.18257–.24307，均未卡在[.18,.25]端点；行为log-prob复算误差最大约7.15e−7，没有发现PPO数值爆炸。

服务器按已存rollout汇总的PANN非零恢复特征为22/1024与35/1024，共57/2048＝2.7832%；上一轮没有原rollout；本次61fd022已补12维小表并独立计数，确认相同22/35结果，仍未读取服务器原NPZ。返程暴露为0。两条E30的同状态PANN特征清零诊断均28/28点cap完全相同；它不能区分原特征为零与策略不敏感，更不是重训反事实。当前尚无记忆收益证据。2Hz标量cap既没有足够动作权限直接修复毫秒级接触，也没有遇到足够材料/返程状态去学习长期协同。

真实终止结算修复已生效：68次自然失败末端legal progress全为0；67次撤回正临时credit，另一次此前侧出已经归零。撤回总进度7.75852，对应奖励38.79260。两次外部目标截断语义保留。按已保存cap、积分项和终止结果重构的68条未折扣自然回合reward约−10.454到−10.203，失败惩罚总−680；这是上一轮重构口径；61fd022实际transition补齐后，逐回合reward复算差不超过1.78e−15。未声称读到未保存的完整GAE过程。不能与旧奖励版本直接比较总reward。

### 2.3 独立末端评价与真实吞吐

| 末次本分支checkpoint-007569 | A | B |
|---|---:|---:|
| E30结束物理时间 | 13.5634s | 13.5208s |
| 最大x | .606748m | .792777m |
| 最终base高度 | .203995m | .193117m |
| 原终止reason | invalid_contact:[[0,13,0,1]] | 相同 |
| 已加载材料cell | 28 | 46 |
| 机身进入x≥.8 / 合规去程 | 否 / 0 | 否 / 0 |
| E60 | 前序阻塞 | 前序阻塞 |

评价使用固定策略均值、冻结归一化、无PPO更新，并未拿采样时的最好回合代替末次策略。E300/600未执行。

两流在北京时间2026-11-04 18:47:30.611开始共享训练时钟。共同训练完成耗时14.05833h，2048新样本对应145.67873条/h；W37单流111.593条/h，工程吞吐约提高30.5%。两流重叠约13.96091h，占共同训练墙钟99.31%；含两条最终E30至全部结束约14.44149h。新物理总时长1008.138s。

规划占两流回合墙钟约88.10%，50444次规划平均约1.7536s/次。这里是同步离线仿真：规划期间物理时钟不推进，不能把1.75s直接称作机器人真实控制延迟；同时，这也**没有实现墙钟50Hz实时MPC**。当前没有CPU/GPU利用率实测，不能根据显存容量推断还有几倍加速。

## 3. 失败原因：已确认、尚待区分及排除项

### 3.1 已确认修复正确，但作用范围有限

r27的C2门同时进入LandingVelocity、ImpactMargin的候选代价、完整乘积有限差分与活跃子步选择；真实材料mask、力、.02域和终止没有乘门。它平滑的是人为接近区的激活，不能消除真实接触切换、正部函数或max子步切换。

真实终止credit在最终transition形成之前结清。两项修复都实际构建并用于W38，不能继续当成“还没实现”的下轮工作。它们不足以解决整个闭环，但失败本身也不能证明修复无效。

### 3.2 不能再用“漏了力矩限制”或“K0/K∞接错”解释

plant、候选rollout、十子步A/B与实际执行均使用kp60/kd5、原关节目标范围和原23.7/45.43Nm限幅。BoxQP显式盒约束是位置目标范围；状态相关力矩饱和与接触主要通过动力学/FD隐含进入。这仍可能造成局部模型有效域不足，但不是已经证实的预测漏限幅。

把目标投影到瞬时非饱和PD区间，在区间非空时产生的当前瞬时力矩与原force clipping相同。后续2ms保持过程可能不同，不能把新增末端clip或低通直接叫作降冲击修复。

实际即时全足刚度K0=1227272.7273N/m，平衡K∞=375000N/m。loaded_target已经用mg/(2K0)约60.775µm；活跃LandingIndentation只对超过mg/(2K∞)约198.900µm的深度施加单侧软惩罚，不是强制达到此深度。两者差别由原契约明确保留，历史r1还尝试过改成K0。真实材料离散切线依赖时间步、历史和有效加载面积，线性K0只是声明的近似；本次未证明常数接错，不重开旧失败改法。

### 3.3 当前必须区分“计划本身有问题”和“计划被执行反馈破坏”

**候选本身不可行的可能性：**实际任务允许腿部几何接触，躯干接触才终止。现有直接风险行主要覆盖四足和躯干，没有直接约束膝/胫/大腿承担支撑；高度、姿态、平均足位置只是软目标。因此优化器存在通过腿部支撑减轻足端代价的自由度，有限躯干代价也未等价于不可违反的任务终止。源码证明这条路径存在，尚未证明r27 accepted实际选中了它。

**局部反馈改变接触状态的可能性：**仿射反馈依赖局部线性化，接触或饱和活动集合变化后，原K可能不再提供可实现的修正。r27确实增加了plan id、accepted/reference身份、Kδx和截断前后动作日志，其事件窗口现已由61fd022补入Git，具体数值和不同分支的先后见5.3。r26旧9.378s等时间仍不能移植成r27事实。

**模型近似可能参与：**预测使用即时线性接触代理，真实TM02携带可恢复内态。只有对齐事件表明首次分离发生在真实加载后，才能把相应模型接口作为这轮首要修订；不能仅凭“软地面”就宣布是本构根因。

## 4. W39已启动：已选机制、继承路径与续接边界

### 4.1 已有事件读取与后续只读补充

两条E30数值窗已补入Git；先读5.3及对应numeric-evidence。需查完整源文件时使用[控制审查第5节](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W38-code-audit.md)的实际目录。开场≤5行，先查已运行W39两流，不重复启动。按第5节后台补传W38原件并做只读分析，不重导出已成功的小表，不热改当前控制器或延长训练时钟。

扫描原记录，定位Stand→outbound之后首次目标截断、首次raw-PD饱和、首次大Kδx、首次实际非足承载、首次真实材料load及原终止。每事件保留前.10s/后.20s，合并重叠窗，再保留末端.10s；通常每条E30约一至两秒关键数据。0.5rad仅作与历史对照的描述阈值，不新增物理停止标准。

必须对齐实际q/qdot、PD请求/实际力矩、执行目标、完整action record、plan id/起点/参考packet、现有accepted/reference/linearization切片、K/C/D、winner、regularization、真实足/非足Fz和材料事件。分别计算nominal在实际状态是否已经饱和，以及feedback新引入的变化。剔除未实际执行的time0 RPC请求和末行时间错位；risk脚序FL/RL/FR/RR与runtime脚序FL/FR/RL/RR要映射。

现有accepted只存node0与+20ms：发现违法足以证明该点违法，两端安全却不能证明十子步或整个.35s horizon安全。prediction NPZ是feedback reference，不能冒充完整accepted。纯运动学几何计算可以读取已存状态，不能伪造未存接触力或补出实际trajectory。确需训练事件作对照时，每流取最早匹配自然回合；不挑最好回合，不重跑旧物理。

### 4.2 启动时的单机制选择与当前已实现范围

以下是原启动时采用的选择规则。**9544cfe现已选择并实现第2项：原躯干几何上的候选可行性优先及无可行候选时的restoration；84f5867记录两流已启动。** 该修订是一次待实验验证的机制，不等于已解决首次失稳。当前任务不得再改回selected_mechanism=null、重跑实现流程或叠加另一项控制修改。

1. 若先发现真实的时间、参考packet、winner身份或导数不一致，优先修该具体接口；用既有数据定位，不凭日志名称猜测。
2. 若accepted自身已预测进入原任务的躯干终止几何，优先修候选接受的可行性语义。对每候选在原十子步及terminal、原受保护躯干几何上计算v=max[原XML margin−几何间距,0]。有v=0候选时只在其中按原任务代价J选择；没有时显式记prediction_infeasible/restoration，按(v,J)字典序选择已有候选。复用原几何/导数，不新增任意惩罚权重或改变真实终止。winner、accepted rollout、候选policy、feedback reference、improvement/surprise与regularization接受逻辑必须一致，不能只改显示的winner或冻结旧动作。这只是模型内有限候选可行性，不保证真实物理安全。
3. 若现有可观察候选切片未违规，而执行feedback先改变了接触/执行器活动集合，则按实际触发接口推导局部反馈可实现域修订。状态相关PD约束的导数和整段保持过程要与候选/执行一致；不预设K clip、简单缩放或低通。不能把“未在两个节点发现违规”写成“全horizon已安全”。
4. 若首次分离明确发生在真实材料加载后，则只修对应的预测/真实接口或材料近似；不输入真实隐藏q/e给PANN，不放宽.02域，不直接替换整套模型。

这是有限的证据选择，不是四项一起叠加或参数网格。如果已有数据确实不足以确定可实施修订，交付实际提取数值与精确缺失接口；不得将未改r27改名r28再盲跑28h，也不生成另一个泛泛计划轮次。

保留r27 C2门、r26因果packet/十子步A/B/TV与PD、658维观察、冻结PANN、2Hz cap、W38终止奖励及全部真实物理标准。r25 MovingSupport/SupportPlacement、载荷分配和r18/r19旧失败项不恢复。必要代码完成、正常构建且源码/配置推送后，直接完整PANN→PPO→MPC→Go2/TM02训练；不增加纯MPC、孤立模块、E30准入、mock/smoke、测试套件、恢复演习或吞吐benchmark。

### 4.3 两个已有分支各自继续，隔离新phase文件

| 新stream | 唯一父分支 | 父恢复点（相对large_root） | 新training目录 |
|---|---|---|---|
| w39-a / seed1 | w38-a | training/PANN_PPO_S5-seed1/resume-state-ep00251.pt | training/w39-a/PANN_PPO_S5-seed1 |
| w39-b / seed2 | w38-b | training/PANN_PPO_S5-seed2/resume-state-ep00251.pt | training/w39-b/PANN_PPO_S5-seed2 |

父权重和resume仍在服务器，未默认上传Release。各自恢复Actor/Critic、optimizer、RNG、归一化及7569/29；tail0、excluded145保持一次。不再fork/reseed，不回退共同ep216，不合并optimizer或数据。祖先fork_identity保留为W38谱系，新增phase_lineage记录W39当前身份；新phase校验不能把祖先stream当成当前stream，也不能删除原身份检查绕过迁移。

9544cfe已接通每流parent map、W39独立training目录、新phase_lineage及train/eval/finish消费者，并保留各父ep251与原恢复检查。84f5867的两job启动快照已提供真实source/build/clock和采样证据；不把这些已完成接线再作为新任务重做。后续resume只沿本phase实际作业继续，不重新fork/reseed，不删除旧W38时钟。

首次实际W39启动已创建w39-training-clock.json，共同28h截止为2026-11-06T07:23:54.713909Z（北京时间15:23:54.713909）；第二流启动和普通resume不重置。每流目标新增1024条/4更新，达到8593/33＝8448已训练＋145历史excluded、tail0，末次checkpoint-008593。当前新控制版本不变更W38奖励。首个新batch为7570–7825，形成本分支update30。

两流都达标时W39共新增2048，实验树独特历史为10641；它仍不是一个10641条或37更新的单策略。新策略资格只来自其本phase本分支实际新update。

### 4.4 资源、预算和停止/评价语义

继续D57现有25vCPU、96GB级GPU、120GB内存与依赖。**当前已启动的W39两流冻结为各8 MPC workers**；现有进程、execution-contract、启动事实和普通resume保持原值，不为worker调整重启或重置时钟。Torch1、BLAS1及已执行的GPU/JAX共享和精度配置保持。

**后续尚未启动的新job默认两流各12 MPC workers；用户已授权各14，无需再次询问。** starter的`--workers {12,14}`只在新job创建时选择；省略即用12，选定值写入该job冻结契约和环境记录，并同步其后续E30/合格E60评价。当前W39的既有评价仍按自己的冻结契约执行，普通resume不接受worker覆盖。24或28个MPC workers与控制/RPC/材料/JAX/I/O共享25vCPU；14允许适度超配，不保证线性加速。该项只调整执行并行度，不自动改变候选数、迭代数、horizon、算法、物理或奖励。

用户报告当前约12核占用，本次没有采集新的利用率或CPU亲和性/cgroup证据，不能据此断言CPU被硬限制为12核。用正常训练已有的planning耗时、每流增量transitions/自身collection墙钟、两流独特新增transitions/共同训练墙钟，以及普通CPU快照判断12或14的实际收益；共同训练墙钟截至后完成的collection，不混入末端评价，也不重复计入继承样本。可在正常启动时保存一次普通CPU/GPU利用率与RSS/VRAM快照，不新增benchmark、worker网格、准入测试或监控平台。训练中独立分支并行，训练或评价每流最多占一个计算槽；紧凑finish/Git串行且复用index锁，状态同时保留两流。

W38在2×8布局下观测到合计训练吞吐比W37高约30.5%；两轮控制版本、奖励和轨迹不同，不能将全部增益隔离归因于并行布局。这条历史事实保持原样，也不作为2×12或2×14已经提速的证据；本次不增加第三流。

2026-11-05用户进一步询问求解器内部并行与GPU MPC，已完成[可行性审查与工程路线](S1_GPU_MPC_FEASIBILITY.md)。现有worker就是内部C++工作池；W38全部70个训练回合普通/启动规划合计占回合墙钟和90.445%，支持优先研究求解器加速。下一未启动版本可研究GPU批量预测/有限差分，保持当前12维PD目标、十子步、预测接触与r26/r27/r28语义；MPX、GATO、TurboMPC等作为明确的架构参考。先利用已有内部计时，不新增独立benchmark。该文档不代表GPU后端已实现或部署，不修改当前W39；旧“本轮不重写GPU planner”不构成永久禁止GPU研究。

交互≤4h，从首次活动起算；实现、必要正常构建、代码配置推送后看到真实样本与普通日志推进即可交接离开，不等完整更新。两流共享28h训练绝对截止；各流训练结束后独立≤4h评价。紧凑交付沿用≤6h兼容上限，完成即结束，不等大传输，不自动加时或串接下一控制版本。

完整保留R12 learning.known_r25_startup_regression已实现的原谓词：自然终止结果保存后首次符合既有退行签名即停止该分支、留部分结果、跳过末端评价并发布。不得改阈值，也不新增失败次数/attempt cap。其他普通物理失败保存后自然续采，每256条新数据正常更新并继续。

正常结束只选该stream/phase/controller/reward最后实际完成的新update（updates>29），固定策略均值/归一化执行nominal E30；原同case资格通过才独立E60。零新update不拿父策略凑评价。保持原30→60→300→600阶梯、原远端3m/完整往返/路线和时长；本phase只安排E30及合格E60，不自动跑300/600或21-case矩阵。stage_pass与合规贯穿/往返分开报告。

## 5. 已请求的原始诊断证据：先推小记录，再后台传完整分析数据

2026-11-05用户补充授权：**原始反馈修正、优化器选中轨迹、参考轨迹以及判断问题所需的数组，一并上传。** 这些数据从本次起属于已请求的诊断证据，不再因“raw/NPZ/超过2MB”自动排除。大checkpoint、完整resume和全长视频仍默认留服务器；分析确需其他数组时可继续扩展明确清单，不必再为常规诊断请求一次批准。

### 5.1 每个实际回合都要交付的控制诊断

| 类别 | 已有文件与内容 | 发布方式 |
|---|---|---|
| 原始反馈修正 | feedback-actions.jsonl：state_delta、原始Kδx、feedback scaling、nominal、截断前后动作、plan id/时间 | 完整原记录，允许无损gzip；Release |
| 优化器接受结果 | planner-records.jsonl：winner、cost、regularization、accepted/reference/linearization角色及已有数值 | 完整原记录；缺失的历史full horizon明确not_recorded |
| 路线参考packet | reference-packets.json及字段定义 | 完整文件；它不是优化器状态轨迹 |
| 完整反馈参考轨迹 | 全部prediction-*.npz | 原NPZ、原精度，不能改名为accepted |
| 真实执行轨迹 | trajectory.npz、startup.npz | 原NPZ；区分实际执行、初始化RPC及末行时钟 |
| 必要模型与解释信息 | model/G1-project-go2.xml、model.mjb、已有result/resolved/身份记录；prediction-model.xml存在时附上 | Git小记录＋Release模型文件；逐attempt保留实际source commit |
| 材料事件 | material-events.jsonl、material-failures.jsonl、cap-decisions.json与已有摘要 | 完整记录；合法空文件记present_empty |
| 学习证据 | 本phase全部2Hz transitions及新rollout-*.npz，包含原obs/next_obs/value/next_value/log-prob/reward/终止标志 | 小记录/索引先Git，原rollout可Release；不上传整个模型权重替代数据 |

密集physical-*.npz和material-*.npz/material-final.npz优先覆盖最终评价和每流各原物理终止原因的首个自然失败；遇到其余关键case，通过显式attempt继续补传整文件。保留原采样率、列和数值精度，不用降采样图、最大值或事件摘要替代原始证据。短事件窗仍有助于快速阅读，但完整已选源文件同时可取。

现有r27只保存条件位置窗内accepted/K的node0与+20ms，**没有完整accepted rollout或完整K horizon可补传**。文件存在但没上传、文件遗失、当时从未记录、合法空记录必须分别标注。不得把reference换名、插值或重跑的结果当作当时选中轨迹。

### 5.2 当前记录边界与下一未启动版本的完整轨迹记录

在正常policy update、原buffer复用前保存完整accepted状态/动作/时间/残差/cost，以及实际安装的feedback reference状态/nominal/K/scaling/插值方式，附winner、plan id、更新起点、原模型和维度。linearization单独标注，不冒充accepted导数；现有风险子步峰值及其来源可随原记录保留。完整K每次policy update存一次，避免在每个500Hz动作查询重复整段矩阵。

记录扩展不重跑候选，不修改控制选择、代价、导数、限幅、真实物理和奖励。完整2ms节点轨迹也不等于完整0.2ms子步接触过程，覆盖范围必须写清。9544cfe已运行的r28仍只保留旧两节点accepted/K及新增候选标量，完整accepted/K扩展留给下一明确未启动版本的正常实现/构建；不重建或中断当前两流，不宣称已存在。

9544cfe已修训练episode实际路径，并在训练/评价2Hz日志写入实际value/next_value、12维PANN及action_change分项；这些不再是待重做任务。旧W38评价缺失值仍不能事后推理冒充原值。本次另修runtime仅接受S5-r27的诊断环境开关，使S5-r28尚未启动的新评价Agent也能写已有planner/action日志；这是Python记录接线，不改C++算法或重启正在跑的训练。该开关仍只提供当前二进制已有的两节点粒度，完整accepted/K并未因此出现。本会话未在D57部署这个补充。

### 5.3 本次W38补传与复审顺序

范围由两个已完成job的attempts确定：70训练＋2条E30，共72个真实attempt。优先两条最终E30及每流第一个自然失败ep217的完整控制/物理/材料文件；随后补齐A的ep218（invalid_contact）、ep226（material_domain_exit）及B的ep218（body_fall）的密集数组，并完成全部其余回合的核心反馈、规划、参考和执行轨迹。其他必要case可显式追加。两条E30及两个ep217已声明的trajectory＋physical＋prediction合计289556325 bytes，全部72回合这三类合计5143218280 bytes；这是原result manifest的大小，不包含JSONL/material/model，也不是本次服务器现存实测。

**上一轮统计和源码结论保留；61fd022新增的小表和事件窗已在本次读取，完整原始数据仍待补传。** D57端可以直接读取本地原文件分析，同时后台补传；不必等所有Release文件完成才做W39目录/恢复/记录接线等工作。当前W39已经实际运行，保留9544cfe所选机制；不热改正在执行的控制器、重启重复作业或重置deadline。补传和只读分析并行进行，结论用于解释本轮及后续明确版本。

新增证据见[A数值索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/jobs/job-20261104T104730Z-w38-a/numeric-evidence/index.json)和[B数值索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/jobs/job-20261104T104730Z-w38-b/numeric-evidence/index.json)。每流1024条训练记录均与保存的PANN/value数组匹配，序号6546–7569连续；各989普通＋34真实terminated＋1目标truncated，34次终止bootstrap均0，目标截断保留next_value。当前12维PANN非零分别22/35条，合计57/2048＝2.7832%，返程样本0；53个加载过材料的回合不能当成53个充分学习到记忆的回合。每条E30虽有28条真实标量transition，但value/next_value及当前/next PANN均未原始保存，导出诚实留null，不能事后造值。action_change分项缺字段是原metrics记录缺口，原奖励已包含该项，逐回合奖励与原公式复算一致。

两条E30事件窗口分别覆盖0.6288s和1.4048s，不能称为完整回合轨迹。以下时间均为原模拟时间，原始来源为各event-windows/<E30>/event-summary.json，控制与物理行保存在同目录window文件：

| 事件 | A / seed1 | B / seed2 |
|---|---:|---:|
| 首次raw-PD饱和 | 6.9900s | 11.2220s |
| 首次abs(raw Kδx)>0.5rad | 13.3380s，最大5.6157rad | 11.3260s，最大0.9309rad |
| 首次目标截断 | 13.3380s | 12.0740s |
| 首次材料load | 13.3346s | 12.5276s |
| 首次非足正载荷 | 13.4446s | 13.0992s |
| 原躯干接触终止 | 13.5634s | 13.5208s |

两流首次raw-PD饱和时，同一保存的实际状态下，单独nominal也已饱和，反馈没有新增饱和关节。A材料load后约3.4ms出现大反馈/目标截断，但在13.300s机身已以−0.58033m/s下降，13.334s已达−0.88297m/s，早于13.3346s的首次材料load。B则在首次材料load之前已经出现大反馈与截断。因此材料接触不是所有早期异常的共同起点，最大Kδx也不足以定位首次失稳或选定K裁剪。B首次非足承载前对应控制步的raw Kδx仅0.00667rad、无目标截断；其12.9992–13.2992s窗口非足正冲量约占总正冲量60.77%，说明接触承载问题不能仅以瞬间控制峰值解释。

两条已存accepted的+20ms节点确实出现原trunk geom13负间隙：A计划699起点13.5600s，节点13.5800s间隙减margin为−0.002360m；B计划697起点13.5200s，节点13.5400s为−0.002977m。它们分别只早于真实终止3.4ms和0.8ms，而且节点预测时刻在终止之后。这证明末端存在被接受的不可行节点，支持审查候选可行性与restoration；不能据此断言它就是更早失稳的首因，或已有修订就一定能恢复任务。导出窗口内A315行、B705行已执行动作的nominal+scaling·rawKδx−before_clip最大绝对误差均为0，只排除这1020行的该种代数接线错误，不能排除所有实现问题。A计划687已存节点accepted与reference相同，B计划587的两种轨迹则有真实差别；角色差异本身不证明错接。当前4.2的候选可行性机制已经启动；以上证据用于审查其能解决的范围及解释后续结果，仍不伪造缺失的accepted完整horizon，也不临时叠加反馈裁剪。

本次会话没有D57连接配置，SSH别名解析失败，未取得或上传W38完整原文件；已通过Git读取上述新补传窗口，并推送交付工具、入口接线和修订要求。必须分别记录“紧凑数值已补传/已读”与“完整原始诊断Release仍pending”，不能把工具推送当作完整数据上传完成。

### 5.4 可直接执行的补传入口与后台流程

使用新入口scripts/publish_s1_go2_r12_diagnostics.py：

- prepare：从已完成job建立有来源和覆盖说明的诊断清单/无损传输文件，不进行网络上传。
- run：先推小元数据和真实pending，再复用现有显式manifest Release上传器，按本job资产逐项确认并推回状态。
- queue：建立独立tmux会话运行run；不等待网络，不占训练GPU槽位。每个job自身防止重复上传；不同job可并行，Git/index写入仍串行。

W38补传直接使用该入口，**不能重跑旧finish来补文件**：旧W38 finisher会重写当期交接和r27状态，可能覆盖W39。新正常finisher则先完成compact Git、释放其锁，再queue诊断上传。每个诊断任务首次建立独立≤6h绝对传输截止，普通resume不延长，不使用已经结束的W38训练截止，也不改变W39训练预算。

在D57项目环境中执行：

```bash
python scripts/publish_s1_go2_r12_diagnostics.py queue --job-dir outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261104T104730Z-w38-a --hours 6
python scripts/publish_s1_go2_r12_diagnostics.py queue --job-dir outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261104T104730Z-w38-b --hours 6
```

需要某个额外完整物理/材料case时，在对应job命令追加--attempt及其原attempt名称。使用已保存文件，不重采样、不渲染新视频。当前job/输出目录以既有实际记录为准；原始执行source commit与新上传工具commit分别记录。正在跑的W39不提前按完成job排队；其正常训练/评价结束后的finisher负责排队，历史W38现在可单独补传。

小元数据与manifest保存至compact_root/jobs/<job>/diagnostics/。原始证据放GitHub Release；体积大时无损压缩或按顺序分块，保存原路径、字节数、采样/时钟/脚序、分块顺序、URL及uploaded/pending/missing/not_recorded状态。复用现有name/size/upload-state核对，不新增全量hash认证、重下载或电脑归档。上传器exit0不等于全部完成，以本manifest逐项状态为准。

### 5.5 存储与历史规则

先推代码/配置/全部逐回合JSON/学习评价/结论，再后台上传上述**已请求**原始证据并继续研究。必须上传的诊断有pending就如实保留；未选大checkpoint/全视频才属于retained_local/not_requested。两类状态分别记录，不能把整轮一概写“raw不上传”。

所有原件和有效resume继续保留服务器，由用户定期导出到自己的存储盘、备份后自行统一删除。Codex不自动删除研究输出或上传副本，不另做用户电脑归档。r26已执行的旧Release历史不重做；本次扩展是新授权，不改写W38此前按旧紧凑规则完成交付的事实。

## 6. 本轮新增论文：机制、阅读范围与迁移边界

截至2026-11-05，在既有TM01–34基础上新增TM35–43九条不同论文；其中TM42由此前补充来源转入正式编号，不能重复算作新发现；另更新TM31/32刊发和阅读身份，不计作新论文。下面七条已阅读相关方法主体/实验段，另外两条明确限于摘要或部分文本。完整来源、版本与阅读范围见[文献清单](../literature/terrain_memory_sources.json)及W38-literature.json。

| ID / 论文 | 新的启发 | 当前如何使用、不能怎样使用 |
|---|---|---|
| TM35 [Planning to Learn / A-SLS](https://arxiv.org/abs/2610.20925v1)，2026-10预印本 | 把估计器、参数不确定性和未来学习纳入闭环可达集；GPU结构化鲁棒MPC | 长期让材料状态及信息改善真正改变候选规划。其四足实验估计关节阻尼，不是本项目地块恢复记忆；不能照搬保证 |
| TM36 [Shooting for Contact / DSMS](https://arxiv.org/abs/2608.03116v2)，2026-10更新预印本 | 细MuJoCo积分＋粗多重射击、动力学连续性缺陷与约束；实际力矩/输入变化代价 | 启发先考虑可实现轨迹与执行一致；主要是运动重定向，未证明可直接替换当前在线MuJoCo/iLQG规划器 |
| TM37 [πⁿ MPC](https://arxiv.org/abs/2606.05687v1)，2026-06预印本 | horizon和多环境维并行的ADMM MPC；时间对应MPC参考用于RL | 为真正GPU利用提供后端方向。其训练期MPC引导与部署纯RL、质心模型/动作不同，不能把现有MuJoCo进程改cuda即获同速 |
| TM38 [Contact Trust Region](https://arxiv.org/html/2505.02291v4)，2025预印本、IJRR2026 | 对局部状态/动作更新施加接触力锥约束；R-CTR去掉原可行约束、保留对偶力锥 | 支持检查反馈有效域；是准动态操作，不是Go2材料安全证明，也不是事后K裁剪 |
| TM39 [Guard Saltation NMPC](https://arxiv.org/html/2403.14159v2)，IROS2024 | 显式传播接触时刻/guard不确定性与协方差，形成约束收紧 | 解释“名义梯度平滑”仍不足以覆盖触地事件；有预定模式/横截等假设，不可直接套在当前柔顺多足模型 |
| TM40 [Fast Contact-Implicit MPC](https://arxiv.org/abs/2107.05616)，T-RO2024 | 在局部模型中保留接触互补结构与接触模式变化能力 | 提示避免只在单一接触模式下线性化。Go1示例点足力输入/低层调谐与本项目全身PD不同，不是即插即用替换 |
| TM41 [Convergent iLQR](https://arxiv.org/html/2304.00346v3)，ICRA2024 | 优化扰动在连续段和冲击间的放大，而非只最小化名义代价 | 为“低cost不等于可跟踪”提供直接依据；平面仿真与高阶导数实现，不照搬新χ项或声称χ总<1 |
| TM42 [Terramechanics-Aware Adaptive Quadrupedal Locomotion](https://doi.org/10.1109/LRA.2026.3732903)，RA-L2026 | 塑性虚拟地面＋接触历史估计摩擦/变形后适配步态，含Go2砂地实验 | 本轮只取得官方摘要；支持地形物理与动作适配共同建模，不等同恢复材料的世界坐标持久记忆 |
| TM43 [Adaptive Legged Locomotion via Online Learning for Model Predictive Control](https://arxiv.org/html/2510.15626v1)，2025-10预印本 | 在线残差模型/随机Fourier特征与MPC | 只读到摘要、引言及方法提纲；不声称完整证明/实验已核验，不据此本轮实现 |

这些论文形成几条不同的解决路径：约束候选本身、限制反馈有效域、显式处理接触切换、让学习状态进入预测、改写真正可并行的求解器。它们支持逐层定位本项目实际缺口，不支持把所有方法同时堆进r28，也没有哪一篇直接证明我们的可恢复材料往返任务已经可解。

## 7. 后续研究方向与阶段判断

当前先让已启动W39按冻结机制与预算继续，并后台补齐完整诊断和小型数值证据。新的只读分析用于解释本轮结果和后续明确版本，不在运行中更换机制。判断进展依次看真实承载/入口存活、合法去程、远端/返程覆盖、合法完整往返与同case评价；更新数、loss或脚尖加载不能单独宣布通过。

稳定基线出现后，再选择能表达研究问题的结构：让因果地块状态和不确定性进入候选接触预测；让策略对落点、加载或重访时序拥有可解释权限；为超过.35s horizon的恢复/返程建立终端价值或上层计划。相同世界cell的跨腿、返程因果对照才验证“记忆”收益，不用仅有即时PANN清零图替代重训/闭环对照。

GPU推进应沿可并行模型与优化结构设计。πⁿ MPC、A-SLS和DSMS提供不同参考，但当前CPU MuJoCo/iLQG不能通过占满96GB显存自动加速。现在保留已执行并记录吞吐的两流布局；待明确需要的接触/材料模型后，再单列后端改写，不在本轮同时替换物理、控制、动作和学习。

原R1/R2负结果、codesign/PPO/adjoint不成功路径、r25退行和r24不合法裸往返全部保留；不重新命名为成功。当前可以推进实验，也有足够具体的下一步，但还没有稳定联合基线、记忆收益、实时部署或实机结果。

## 8. 当前Codex续接与补传提示词

```text
继续MPC+RL，先读GitHub main的AGENTS.md、主计划顶部6行及4/5节、当前R12契约和最新两流job/index。9544cfe已实现并构建S5-r28，84f5867已记录job-20261105T032354Z-w39-a/b真实采样；这是已运行任务，不能改回planned、重新选择机制或重复启动。开场最多5行后直接工作，先核对D57现有s1-w39会话和两流进程，保留未提交修改。

本次新增任务是W38完整原始诊断后台补传与只读分析。安全同步本次交付补充，不替换正在运行的二进制/冻结契约、不重建C++、不热改训练进程、不重置共同截止2026-11-06T07:23:54.713909Z。按主计划5.4分别对job-20261104T104730Z-w38-a/b运行publish_s1_go2_r12_diagnostics.py queue；不要重跑旧finisher。确认queue真实状态和稳定tmux后即可继续研究，不等待全量网络传输。

每个W38完成回合都上传完整feedback-actions.jsonl、planner-records.jsonl、reference-packets.json、trajectory.npz及全部prediction-*.npz，附已有startup、模型/配置/结果与材料事件。两条E30、A217/A218/A226、B217/B218的完整physical/material数组优先；全部本phase新rollout原NPZ和2Hz记录也上传。需要其他case时按同job显式追加--attempt；不因raw/NPZ/大于2MB静默过滤，不用摘要取代已请求原件。Git保存manifest、来源、分块顺序、URL和真实pending/missing/uploaded状态；原数组精度不变。

61fd022已补两流各1024条学习/PANN/value和两E30短窗，9544cfe还补了reference_packet对应信息，先读这些，不重做成功导出。A在材料load前已下降，B大反馈和截断先于load；末端accepted违规距终止很近，不能直接当早期唯一首因。当前r28选择的是原躯干margin候选可行性/restoration，冻结运行；进一步完整时序分析用于解释本轮和下一明确版本，不加K裁剪/低通或重跑旧物理。

r28训练episode路径和2Hz value/PANN/action_change日志已在9544cfe修复。本次runtime补充仅让尚未启动的新r28评价进程开启现有planner/action日志，部署后不需要重启训练或重建C++；若评价已经启动则保留其现状。现有r27/r28完整accepted/K horizon从未记录，旧节点仅0/+20ms；prediction是反馈参考，不准改名或伪造完整accepted。完整记录扩展列入下一未启动版本，当前两流继续。

W39保持各自父seed1/2 ep251谱系与optimizer/RNG/归一化、training/w39-a/b目录、各8workers/Torch1/BLAS1、既有28h绝对训练期限及各自≤4h末端评价。沿正常作业和实际resume续接，目标各8593/33；仅本phase新update固定策略E30，合格才E60。原known_r25停止谓词、普通失败续采和全部物理/材料/奖励约束不变，不新增失败次数上限、准入测试或benchmark。

后续尚未启动的新job默认两流各12 MPC workers，也已授权各14，无需再次询问；starter --workers {12,14}只用于新job，选定值写入该job契约并同步后续评价。当前W39冻结8及其评价不改，不为worker调整重启、重置或重复启动。保持两计算槽、Torch1/BLAS1及原GPU/JAX配置；28个MPC workers共享25vCPU允许适度超配，不能据此自动改候选数、迭代数或算法。用正常日志已有planning耗时、独特新增transitions/实际训练墙钟及普通CPU快照判断收益，不加benchmark、网格或新准入。用户报告约12核占用不是本会话实测，也不证明CPU亲和性或cgroup限制。

正常finisher先成功推代码/配置/所有逐回合JSON/学习评价/结论，再释放compact锁并后台queue已请求诊断；两流上传可并行，Git/index短写串行。未选大checkpoint/完整resume/全视频继续留D57，用户导出备份后手动统一删除；所有原件和已上传副本都不自动删除，不另做电脑归档。收尾记录实际补传与只读分析状态，保留两流当前进度，不把工具推送或队列已启动写成数据已全部上传。

求解器/GPU加速研究另读docs/S1_GPU_MPC_FEASIBILITY.md及其中工程提示词。其结论来自固定提交的W38全回合计时、源码和一手文献；当前尚无GPU后端实测。可在现有训练继续期间研究适配，不在W39进程内热换求解器或动力学。
```
