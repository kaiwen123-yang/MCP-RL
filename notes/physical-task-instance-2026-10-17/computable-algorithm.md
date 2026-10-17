# 有限预算的PPO、后段需求与配对响应更新

日期：2026-10-17。承接`b5d5697`复审及本轮物理任务实例。**这是可实现的训练阶段计算规格，当前未执行PPO、actor/model优化、新batch采集或机器人rollout。**本轮未取得可用于所定义狗/臂任务的已核验checkpoint或已校准动力学/响应模型；旧servo随机PPO batch仍缺失。下述数值预算是预声明设计例，不是实测性能。

本稿选择的身份是：**标准随机PPO提案＋确定性部署任务的有限模型化约束步骤＋配对响应辨识/蒸馏**。没有声称新求解器、独有梯度或已优于同信息鲁棒PO/策略受限TO。

## 1. 明确选定随机性和目标

训练使用`a~Normal(mu_theta(o),sigma_theta(o))`及实际固定执行映射`u=A(o,a)`探索；部署能力只对`ubar_theta(o)=A(o,mu_theta(o))`定义。std保持正值，但**不对Gaussian全部噪声声称能力集合扩大**。固定执行映射、支持、传感权限和所有后段上下文向各比较方法相同开放。

PPO仍用原完整物理任务的实际reward、固定GAE和潜动作ratio。确定性步骤优化的是**同一物理完成时间/资源目标的mean-controller预测值**，不是随机期望、不是把mean value塞进随机GAE，也不是另造奖励权重。二者通过共享mean参数联系，统计对象保持分开。

全任务成本`J_task`与完成/失败规则由任务合同给定。若本轮解析实例仅比较完成时间，则`J_task=T`，不添加能量权重；若以后硬件任务另报告资源指标，应先声明约束或多目标规则，不在看结果后加权。违反合同仍为独立g，不藏进PPO reward或改名为训练成功。

## 2. 不用精确K作为oracle：合同和有限后段值

任务文件必须提供可执行函数：

```text
observe(history) -> policy observation and permitted stage context
stage_event(s, clock, task) -> current stage / completed / failed
g_path(s,u,task), g_entry(s,task), g_finish(s,task) -> signed margins
task_increment(s,u,clock,task) -> original time/resource cost
admissible_commands(s,task) -> common command permissions
```

正margin表示违反；尺度来自净空、允许力/加速度、到达容差和真实时间预算。物理任务中的净空例为`|y|-b`，输入例为`||u||2-a`；到达/停稳容差必须来自任务和测量要求。没有机器人数据时这些硬件容差标为未定，不能拿旧P/tube或随意level set填上。

后段对象采用**有限可计算价值**，不使用未知全局`K2`：在入口s，为当前mean策略评估Q个已声明的合法因果宏命令树，每个树最多H个低层步。树节点只根据当前可用观测/阶段事件/时钟选择边，不能根据未来扰动scenario编号选择动作；观察历史尚相同时命令共享。共同库应包含任务允许的等待、减速和既有可行方案。有限库遗漏方案是规划预算问题，不是物理不可达。

对模型成员e和明确场景w，记树q的预测后段成本为`J2(q;s,theta,psi_e,w)`。有限值为

\[
\widehat V_{2,k}^{B}(s)=\min_{q\in\mathcal Q_k(s)}
\sum_{e,w}\omega_{ew}J_2(q;s,\bar\pi_{\theta_k},\widehat F_{\psi_e}),
\tag{C1}
\]

只在满足本次明确的有限场景合同且完成任务的候选中选取；没有完成候选则返回`UNKNOWN_CONTINUATION`，不给它假造一个精确K或“无解”标签。达到预测截断但未完成的样本不能以零尾值当成功。`omega`在设计中冻结；均匀场景均值只是该有限场景目标，不自动等于实际扰动概率期望。约束对全部已声明成员/场景分别检查。

当前解析实例可以直接使用已知单向加速—制动后段值。在`D=L-x1>=vx1²/(2a), vx1>=0`的适用域内：

\[
V_2(D,v_x)=\frac{2\sqrt{aD+v_x^2/2}-v_x}{a}.
\tag{C2}
\]

这只是该已知物理任务/控制类的解析值；域外不据此宣布机器人不可恢复。一般机器人阶段改用(C1)及有限拟合值，不能继续引用(C2)当机器人oracle。

## 3. 当前数据能给什么标签

每个未来真实执行batch保存：`behavior_actor_hash, action_map_hash, planner_version, model_version, task_contract_hash, observation/history, permitted context, raw sampled action, actual action, old logprob, reward, value/next_value, done/truncated, physical measurements, command/clock, stage/event`。

模型先分成两层：

1. **执行动力学F_psi：**真实执行记录给`(s_t,u_t,s_(t+1))`或可观测历史预测标签。机器人中的u是实际发送且经过软件限幅的当前控制接口命令，例如关节位置/速度目标；未测电机力矩或笛卡尔力不能作为已知输入。实际伺服、延迟和必要观测历史属于F_psi的对象。随机策略数据可拟合该policy-independent转移。旧策略数据只有在硬件/执行映射/信息状态一致且覆盖有效时才可继续使用。缺观测分量、时钟错误、未记录执行映射或环境已改变时对应标签无效，不靠假设补全。二维norm球只属于解析例，不给机器人actor增加力控制权限；具体权限见[机器人任务合同](robot-task-contracts.md)。
2. **当前mean闭环G_(theta,psi)：**由同一实际mean actor、观测与执行映射、F_psi组合得到。它不是任意优化的理想响应头。紧凑宏模型G_phi若用于在线提速，其标签来自这个组合模型的展开，必须标为**模型生成标签**；不能把它叫新theta的真实执行数据。

机器人没有精确状态或物理导数时，必须用已记录的测量/历史模型表示，明确可辨识性和域。模型ensemble差异只是诊断，不自动成为置信包络；小预测误差也不自动证明控制方向/状态Jacobian正确。无对应模型或有效标签时，额外model-based actor步骤返回`UNSUPPORTED`，不是用oracle代替。

PPO的随机GAE只由原batch实际reward和其冻结critic bootstrap形成。deterministic后段值、模型场景、newtheta生成轨迹不冒充该batch的old logprob或advantage。任务终止/截断使用事前规定的有限reward、done及bootstrap规则；不能把失败写成无穷return再求梯度。模型路径的失败、截断、UNKNOWN用状态标志与实际有限prefix成本保存，不填零后段value，也不对无穷成本求导。缺有效完成后段的packet不进入性能value梯度；这减少了可支持需求，不能偷偷用成功样本均值代表全部入口。

## 4. 需求选择：有限M个packet，明确它改变目标

先从batch中的实际阶段入口按`episode_id, stage_id, time`固定顺序选最多M0个状态；同一episode/stage只取首次预声明入口。下面权重中的M0指实际选入数量，上限设计为32；不足时不补造/复制入口，预算按实计。任务preview来自允许的任务文件，不能偷看未来外扰。按(C1)对每个入口使用相同Q、E、S、H预算，保留成本、g、是否完成和不确定性状态。

packet为

```text
D_i = (s_i, task/context, held-command clock, chosen causal tree,
       finite scenario/model IDs, V2 estimate and source,
       currently relevant physical facets and margins,
       task/constraint covectors, version hashes, support/calibration status)
```

最多保留每个入口2个合同facet：按归一到物理容差后的最大违反/最小正余量排序，tie按固定facet编号；性能cost covector另保留。所有约束松弛时，性能需求仍来自V2，不凭空产生约束dual。

压缩规则固定为：按共同任务合同ID和stage划分离散context（不把每个连续状态值另当context），每个context至少选一个代表；有已知违反的context先处理，代表取归一g最大者，tie按入口顺序。context数超过M则返回`COVERAGE_BUDGET_EXCEEDED/NO_MODEL_STEP`，不偷偷跨任务合并。余下slots以同物理尺度下的covector方向作确定性最远方向选择，tie按原入口顺序，直到M或全部入口已选。将M0个入口按同stage/context及最近方向分配到所选代表，记录每个cluster的权重`w_i=n_i/M0`。零/不可信covector保留“没有可用信用”状态，不靠挑梯度大来制造需求。

**这不是无损压缩，也不是无偏抽样。**用代表替换成员改变有限经验目标及覆盖，可能遗漏稀有关键入口。必须保存原M0入口/分配/权重/遗漏的facet，并在有限接受检查中重评全部M0，而不是只检查选中M。模型不支持、搜索未解、物理外界已排除或已知可行但误选的请求分别标记；它们不能都当成“必须学习”标签。

## 5. 一轮实际更新的确定步骤

以下是**以后获准训练时**的一轮伪代码；本轮不执行任何theta/psi/phi更新。

整体先后关系固定如下；特别是本轮F拟合只发生在需求形成之前：

```text
input: true random-policy batch B_k, task contract, previous accepted deployment pair
freeze old log-prob/GAE/returns and train/holdout episode split
fit F_psi from real transition labels with the fixed fitting budget
F_supported = check its declared empirical domain
freeze psi and model/scenario IDs for this whole round
theta_P, nu_P = fixed-budget PPO proposal with per-epoch KL rollback
if not F_supported: return TRAINING_PROPOSAL_ONLY and retain the old deployed pair
D_all = finite preview on M0 recorded entrances under theta_P and frozen F_psi
D = deterministic M-packet selection; retain every D_all record and status
if no supported completed continuation remains: return NO_SUPPORTED_DEMAND, retain deployed pair
require supported task-time derivatives and unique active max branches, else NO_MODEL_STEP
form current-branch task/aggregate-contract derivatives and empirical behavior metric
solve one bounded small-dimensional local subproblem
for alpha in [1, 1/2, 1/4, 1/8]:
    recompute full finite value and every contract on all M0 entrances
    recompute old-batch surrogate and both empirical KL tests
    accept first passing mean proposal; otherwise do not publish new capability
generate versioned mean-response labels using the same frozen F_psi and candidate actor
fit compact G_phi with fixed steps; perform independent model holdout checks
validate candidate and old deployment on the same predeclared mean-controller case set
publish EMPIRICAL_CASESET pair only if the finite release conditions pass
otherwise retain old deployed pair; log every proposal, rejection and spent operation
```

第6节的F拟合是上述同一次拟合，不是接受后再训练F。若psi、场景、合同、执行接口或mean参数在接受后改变，旧的接受结果失效；不得直接发布，必须另计预算重新形成需求和复核。相同的冻结psi贯穿所有需求、回溯、G标签、模型留出检查与该轮验证中的预测对照。

### 5.1 随机PPO提案

冻结`B_k`的old logprob、GAE、return、命令和任务语义。使用成熟PPO的

\[
L_{clip}(\theta)=\frac1N\sum_t\min\{r_t\widehat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t\},
\quad r_t=e^{\log\pi_\theta(a_t|o_t)-\log\pi_{\theta_k}(a_t|o_t)}.
\tag{C3}
\]

固定4 epochs、固定minibatch顺序/随机seed，原critic拟合随机任务return，mean/std按PPO更新；得到本地提案`theta_P,nu_P`。正方差Gaussian使用潜动作密度；执行映射存在singleton/fallback时不能称潜密度等于物理动作密度。对KL采用显式检查，clip不是KL保证。

每epoch前保存actor/critic/optimizer状态；epoch后在原batch计算经验Gaussian KL。超过预声明`delta_P`时恢复该epoch前完整状态并结束PPO块，不额外补跑epoch；已经花掉的更新仍计账。设计例取`delta_P=.015`，为后续mean块留出独立检查空间，但KL数值不能用简单相加替代最后的合成KL重算。

PPO步骤改进的是随机任务surrogate，不保证确定性mean表现。`theta_P`只是提案，尚未发布可用能力。后续mean步骤不更改std和critic，不重新生成GAE。

参数块按函数依赖定义：`theta_mu`包含会影响mean的全部共享主干；“std-only”仅指不会改变mean函数的专有参数。若网络共享特征，不能把主干更新误记成std-only变化，或据此跳过G版本刷新。

### 5.2 在theta_P上形成模型化mean任务步骤

冻结已校准的物理模型psi；在theta_P下重新生成第4节有限需求，避免用theta_k的后段能力偷偷代替theta_P。对选中M个packet，采用原任务成本的有限mean预测目标

\[
J_D(\theta_\mu)=\sum_iw_i\sum_{e,w}\omega_{ew}
J_{task}(\bar\pi_{\theta_\mu},\widehat F_{\psi_e},\chi_i,w).
\tag{C4}
\]

所选因果树、事件身份及候选分支在求导时stop-gradient；实际状态、mean actor和执行映射全链求导，**事件发生时刻不能同时冻结**。该导数是当前活动分支的模型surrogate导数，不是CEM KKT乘子或全局可达值导数。出现切换时用有限前向检查，不伪称光滑。

完成时间信用采用一个明确的可微时间估计：模型提供预先定义、经过数据核验的连续段内插值`s(t,theta)`，对唯一活跃完成事件`h(s(T,theta),T)=0`作定位；事件身份局部固定而时间T参与求导。横截条件`h_s dot_s+h_t != 0`成立时，

\[
D_\theta T=-\frac{h_s(D_\theta s)_{t=T}+h_\theta}{h_s\dot s+h_t}.
\tag{C4a}
\]

固定保持时间可随后加回，多个先后事件须对实际事件链求导。这是连续模型/插值的任务时间估计，不冒充离散完成拍数的精确梯度；有限回溯和发布仍重评任务文件原定义的事件与时间。数值设计例选定用已算两端状态在有效局部坐标内作线性dense-output，离散模式/任务编号不插值；每步一次构造，不增加动力学forward。该近似必须先核验，不适用姿态/接触段不能强行线性插值。每path最多8个任务事件、每事件固定最多8次标量定位评价，分辨率/有效域不足则拒绝请求，不继续求根直到成功。若将来改高阶dense-output并增加动力学调用，要重新计账。

若只有离散完成拍数且停止梯度，T通常局部恒定，C5不能凭空产生非零时间信用。线性dense-output在采样结点通常不是C1，本版要求事件根处在同一插值段内部；根落在结点且左右隐式导数不一致时也拒绝，不用某一侧分母非零冒充唯一导数。未核验dense-output/时间导数、事件相切、并列完成分支、接触重置导数缺失或预算内定位失败时，返回`UNSUPPORTED_TIME_CREDIT/NO_MODEL_STEP`；本版不靠straight-through、未声明的新reward或免费的精确value代替。解析例的C2是已证明的光滑后段值，所以可以直接取得−1.25；它不自动为机器人提供可用时间导数。

共享actor若同时改变前后段，必须包含后段的直接参数导数：

\[
\nabla_\theta J=
\nabla_\theta J_{pre}
+(D_sV_2)D_\theta s_{entry}
+(\partial_\theta V_2)_{s_{entry}}.
\tag{C5}
\]

只保存入口covector却省略最后一项，不能声称得到了共享策略的完整梯度。实现可用VJP，不必存完整Jacobian；packet可压缩任务方向，但不凭空免去后段计算。

每个所选packet至多保留两个物理facet。在局部子问题中先聚合为`G_(i,l)=max_(e,w,t) [g_l(s_t,u_t)+m_(i,l,e,w,t)]`，其中t覆盖该facet适用的全部prefix/入口/完成时点，m是已声明经验误差余量。因此G至多2M行，而不是把2MES条场景约束误算成2M行。余量来源/域/留出样本数一起保存；有限残差上界仅是经验范围，不是全局证书。

本版只在聚合最大分支唯一且可微时用其梯度；严格并列或在预声明导数数值精度内无法分辨时返回`NONUNIQUE_ACTIVE_MAX/NO_MODEL_STEP`。因为真实方向导数是所有活跃分支`max_b A_b d`，不能任取一行就声称精确线性化。该保守拒绝可能显著降低更新接受率，是本算法的限制；若未来纳入全部活跃行，必须另改维数与反向预算。本版至多2M行的A只描述当前可微局部问题，不保证所有场景的非线性g。实际回溯仍复核全部场景、时点和所有合同facet，包括未进入局部QP的facet。算`g_D=grad J_D`、A及PPO surrogate梯度`g_S`，std冻结。

采用一个**小维约束近端步骤**，不把未知argmin当oracle：

1. 从每个batch样本在theta_P的潜动作mean score构造`Fisher_diag=(1/N) sum_t score_t⊙score_t`，再取正对角行为度量`D=diag(max(Fisher_diag,1e−6))`。旧行为样本下的这个经验二阶矩不等于theta_P分布下的精确Fisher；batch平均梯度平方也不等于逐样本平方的均值。只用mean参数，它依赖参数化，不称真实物理可达集合。
2. 对`D^(−1/2)[g_D,g_S,A^T]`做QR/SVD，保留数值独立列U；令`d=D^(−1/2)U z`。M=8、聚合后每packet至多2facet时，z维数最多18。若没有数值截断，`D^(1/2)d`在该列空间外的分量不改变任何线性任务/约束，却只增加近端项和trust范数，因此删掉它精确保留**这个局部线性/二次子问题**的解。它不证明真实非线性策略问题的无损降维。数值秩阈值固定为`max(matrix_shape)*machine_epsilon*largest_singular_value`；记录丢弃奇异值，非零截断时只称受限数值提案。
3. 用固定SLSQP预算求**凸局部子问题的数值提案**：最小化`g_D^T d + ||z||²/(2 alpha_D)`，约束`G+A d<=0`、`g_S^T d>=0`、`||z||²<=2 delta_D`。固定`maxiter=100,ftol=1e−10`；检查返回约束，不以solver success替代可行性。不收敛/线性约束在trust域内不可行则返回`NO_MODEL_STEP`，不加新reward或加大预算追结果。
4. 仅按固定`alpha∈{1,1/2,1/4,1/8}`回溯。对每个提案重新计算全部M0入口的Q树/E模型/S场景，允许真实因果树分支按新状态改变；重新算后段值，不沿用旧theta的K/V。
5. 只接受同时满足：全部M0入口的本次有限合同、**未压缩M0等权完整前向任务值**不高于theta_P、原batch PPO surrogate不低于theta_P、mean步骤经验KL≤delta_D以及合成更新经验KL≤delta_total的首个提案。压缩J_D只提出方向，不能用其自身评分掩盖被代表入口的退化。比较误差按事前数值容差记录，不用事后放宽物理g。全失败保留theta_P作为训练提案，并明确mean能力未通过；在线部署仍保留旧已接受配对，不能将NO_MODEL_STEP包装成成功。

若theta_P已远离可行区域，这个小trust域可能无法恢复合同。该算法没有声称每次都能找到可行更新；普通PPO探索可继续产生训练数据，但生产能力标签/模型发布必须经过后述独立检查。不得为初始不可行虚构一个已经可用的安全策略。

这一步是标准确定性任务约束优化，在PPO提案上另作有限mean步骤；它改变了训练算法，**不是未修改PPO自然拥有的性质**。所有g松弛时仍可有g_D，其来源是后段任务value而非λ。条件完全相同时，同信息model-based PO/策略受限TO可以得到同一梯度和局部步骤。

## 6. A4的有限实现及模型版本

1. F_psi用真实执行transition训练，按episode划分训练/留出，不能相邻帧随机切分伪装独立校准；记录策略、任务、执行映射及硬件版本。模型拟合用真实state/action标签，不拟合“希望得到”的响应。此项对应第5节整体伪代码开头，之后直到本轮结束都冻结psi；这里没有第二次F更新。
2. 对通过模型内有限接受的theta_candidate，使用同一F_psi ensemble的当前策略展开，保存`Y=(macro endpoint, prefix g maxima, actual task increments, completion/timeout flag, actuator/history continuation state)`，同时标明`label_origin=model_generated`。**复用最后一次已通过的完整回溯所保存、版本完全匹配的展开**，不在预算外再免费生成一轮teacher；若没有这样的记录，需另计完整生成预算，当前设计返回`MISSING_CURRENT_LABELS`。G的一条标签只对应一个实际宏命令保持段，包含该段输入/时钟；不能用320步整后段端点冒充10步宏转移。设计例每宏段10低层步，从各320步path切出至多32个宏标签，任务完成后不伪造后续标签。
3. 每个紧凑G_phi,e只拟合相应F_psi,e下当前mean策略的Y，固定训练步数；性能value头拟合(C1)的有限值和完成状态。censored/UNKNOWN后段不填零value，也不标“不可达”。当前theta改变后旧macro标签不作为当前theta真实标签，只能作有版本的预训练/回放，并重新校准。
4. 最终候选另在预声明的mean-controller执行环境验证：使用完整任务/同信息MPC、场景和独立episode，不只是模型rollout或训练数据误差。设计例固定16个可复现案例，列出初始条件、合法可观测外扰、任务、接口和最长400步；不得看结果换案例。候选必须在每个案例完成全部事件并通过所有真实任务g；对旧部署同案例已完成者，候选的原任务成本逐例不得更差（只允许事前数值误差容差）。旧部署失败/截断的案例按完成状态分列，不能给它填无穷成本做梯度或平均值；候选通过可以记录该案例由失败到完成，但不虚报一个有限时间改进值。任何候选未完成、违约、信息/接口不一致或可比较成本退化均拒绝这次能力发布。

   通过后的标签仅为`EMPIRICAL_CASESET`，限定这16个案例及相同版本，不是全域或概率证书，16例零失败不能宣称95%可靠。若将来要概率风险结论，须另定抽样分布、独立性、置信水平和样本预算，不能沿用本标签。默认新旧策略各执行16×400上限，共12,800执行transition；只有任务、接口、初始条件/扰动、旧actor及其MPC/model版本全部固定、旧原始执行记录完整可核时才可复用baseline缓存，此时本轮新增可降至6,400，但必须明确缓存来源和历次总成本。实际硬件不能精确复现案例时需预先设计配对/随机实验和容差，当前规格不替它假设可复现。未做此项只能返回`MODEL_ONLY_PROPOSAL`。
5. 发布元组`(theta_mean_hash,phi_hash,psi_hash,task/action-map/context schema,calibration domain,validation status)`。随机std变化本身不改变mean-controller模型，但mean参数、执行映射、任务条件或硬件变化会使对应宏响应版本失效。策略与模型共同更新不等于必须同一optimizer；对应关系必须可查。

当前没有机器人F/G或mean验证数据，以上发布条件未满足。本阶段只有算法规格和解析信用例，不能标记新的可部署能力。

## 7. 有限验证如何进入MPC

在线MPC只查询匹配当前mean策略版本的紧凑G_phi/S_phi或直接组合模型，输入包含当前真实可得状态/历史、命令、保持时钟、任务与后段上下文。候选按同一任务成本及完整阶段g选择，允许合法等待/减速；模型不支持的入口返回UNSUPPORTED，不外推为全局能力。

每个查询返回：预测endpoint/后段value、prefix/资源margin、有限场景通过状态、校准/版本标签。有限模型/场景通过只能叫该范围内的经验预测可用；没有有效外包与场景覆盖时不叫鲁棒认证。无候选时使用任务事先定义且确有适用条件的降级或报告无计划，不能把零命令默认为普适安全。

这是有限能力估计，不精确重算全局K。若只修正phi而theta未变，收益归为模型/规划；若theta改变而实际mean验证没有改善，也不能把新模型预测说成能力增长。

## 8. 本轮物理packet的具体输入输出

解析任务中初始横向速度±v0已经合法观测，输入满足真实二维加速度norm预算。参数r是解析反馈分配族的描述，**不是原神经网络参数，也不是PPO采样动作**。

在归一化`a=1,v0=1,L=4,b=1`、r=0处：

\[
Y=(\tau,x_1,v_{x1},|y_1|)=(1,0,0,0.5),\quad
\frac{dY}{dr}=(0,0.5,1,0),
\]
\[
D_Y(\tau+V_2)=(1,-0.5,-1,0),\quad
\frac{dT}{dr}=D_YT\frac{dY}{dr}=-1.25.
\tag{C6}
\]

只追横向恢复时间有`tau'=0`；这不等于完整任务PPO的真实梯度为0。净空margin为−.5、约束松弛，λ=0；非零信用由后段完成时间给出。

旧恢复后状态Y如上，后段先加速2，制动点x=2，绝对开始制动t=3，总时间5。任务导出的有限分配`r=sqrt(3)/2`后，`Y=(2,sqrt(3),sqrt(3),1)`；后段加速约`.2090697947`，制动点约`2.1160254`，绝对开始制动约`2.2090698`，总时间`4.15019039690625`。模型准确描述真实反馈改变，MPC因此改变后段切换时刻；不是独立把模型标签改乐观。

同一反馈类、信息、norm预算和任务下的标准TO给出相同解/信用。这个packet证明“后段value能指出当前局部指标遗漏的任务取舍”，不证明独立算法优势，不证明RL已学会这段控制，也不替代机器人checkpoint验证。

**有限变化反证已经核算，不能只凭旧packet接受候选。**若把r=0处的旧covector直接用于上述有限Y变化，得到

\[
T_{linear}=5+(1,-.5,-1,0)\cdot
[(2,\sqrt3,\sqrt3,1)-(1,0,0,.5)]
=3.4019237886466844,
\]

而完整后段值给出`T=4.15019039690625`，旧线性cache过度乐观`0.7482666082595659`。新点的costate约为`(1,−.51516634,−.10770572,0)`，确已变化。见[纯信用模块核算](credit-packet-results.json)与[`control_task_credit.py`](../../src/control_task_credit.py)。

因此(C6)的anchor链式梯度正确，不推出跨有限变化的cache有效。第5节回溯必须重新计算完整后段value/约束与实际因果分支，并把这部分预算列入；不能只看线性surrogate下降就通过。这里没有按`.7482666`选择新容差来让cache过关，也不声称另一个标准TO缺少同样刷新步骤。

## 9. 一轮预算：符号与预声明设计例

记真实执行batch N、PPO epochs U_P、物理模型ensemble E、候选入口M0、保留需求M、候选因果树Q、场景S、展开H、回溯L、独立模型留出入口M_v。每条model path都按H计，不把模型反事实当免费数据。

| 项目 | 上界/计数式 | 设计例（未运行） |
|---|---|---:|
| 随机PPO执行数据 | N | 64env×256=16,384 transition |
| PPO actor/critic minibatch更新 | U_P ceil(N/b_P) | 4×16=64，b_P=1024 |
| F_psi拟合 | E U_F个更新；E U_F b_F标签重用 | E=3,U_F=32,b_F=1024：96更新/98,304标签使用 |
| F_psi真实标签留出检查 | E N_Fval一步模型forward | N_Fval≤4096：至多12,288 model步；不进入F拟合 |
| M0入口的有限后段规划 | M0 Q E S H | 32×8×3×4×320=983,040 model步 |
| packet covector重算 | M0 E S H forward，最多3 RHS反向 | 122,880 forward；368,640 step×RHS |
| 选中M的mean参数信用 | M E S H forward，最多3 RHS反向 | 8×3×4×320=30,720 forward；92,160 step×RHS |
| 一次小维QP | 维数≤2M+2，固定100iter上限 | ≤18维，1次solver调用 |
| 固定4次回溯的全部需求重核 | L M0 Q E S H | 4×983,040=3,932,160 model步 |
| 独立模型留出任务检查 | M_v Q E S H | 16×8×3×4×320=491,520 model步 |
| 紧凑G_phi拟合 | E U_G b_G模型生成标签重用 | 3×16×1024=49,152标签使用；不计为真实数据 |
| 紧凑G_phi留出预测比较 | M_v Q E S ceil(H/m)个宏响应预测 | m=10：16×8×3×4×32=49,152 macro-model调用；对应上行全部留出宏片段 |
| 确定性mean实际验证 | 2 N_val episode×最大步数，旧/新同案例 | 2×16×400=12,800执行transition；合格baseline缓存可降本轮新增至6,400 |
| 一次在线MPC | Q_online E S H_macro紧凑调用 | 64×3×4×6=4,608 macro-model调用 |

设计例每轮额外模型forward上界合计**5,572,608微步**（原展开5,560,320＋F真实留出检查12,288），另**460,800反向step×RHS**；随机与配对mean验证执行最多**29,184 transition**（16,384＋12,800）。F的4096条留出数据属于N之内：预先划出16个独立环境流，整段episode仅属训练或留出，实际不足时不补造标签；机器人只有一个流时另按完整episode划分并报告实际数，不能默认独立。满足上述baseline缓存条件时，本轮新增可为22,784，但历史baseline成本仍保留。F/G拟合、PPO优化、QP和在线MPC不混入微步数后消失。mean真实验证只对一个最终候选执行，不因模型回溯反复产生未计账的真实分支。无可接受候选也照记已花预算。

完成时间的dense-output与事件查询另列：上述展开共17,376条至多320步的path，至多5,560,320次线性段内输出构造；每path至多8事件、每事件8次定位，故至多1,112,064次额外事件谓词评价。若每步检查K_g个合同量，基础path合同评价另为至多5,560,320 K_g次。几何碰撞/事件检查不是免费，耗时取决于真实几何复杂度；这些调用不能混称动力学forward或已测FLOPs。当前物理任务还没有经验证的机器人dense-output或时间导数，相关模型步骤仍不可执行。

实际采集与mean验证中还会调用MPC，必须另记`N_plan*Q_online*E*S*H_macro`个紧凑模型调用，不包含在上述微步数里。设计例若事前固定每10低层步规划一次，则采集最多`64*ceil(256/10)=1664`次、两组mean验证最多`2*16*ceil(400/10)=1280`次，共2944次；上界为**13,565,952个宏模型调用**，再加G全部留出宏片段比较49,152次。这里使用旧/新已配对紧凑模型；若改用完整F展开，必须按实际每宏段低层步数重新计数，不能沿用便宜宏调用账。事件触发额外重规划必须预先给上限并占用同一N_plan预算，超额不计为本设计成功。

额外actor计算单独列账，不能由“≤18维QP”隐去。P为mean网络参数数目、d≤18：逐样本经验Fisher需要N个score梯度贡献，设计例16,384个，可向量化/流式但工作量按`O(NP)`累积；这不是一次batch平均梯度的平方。g_S至少再需一次全B前向/反向。每次回溯需要全B的候选surrogate与两种KL，可共享该候选forward，至多`L N=65,536`个候选actor样本forward；theta_k和theta_P的分布参数分别最多N个forward并缓存。4个PPO epoch末KL另最多`4N=65,536`个样本forward（训练minibatch工作已在表中）。QR/SVD额外`O(Pd²)`运算和`O(Pd)`内存，Fisher缓存/流式累积至少`O(P)`内存。完整可微模型展开中的actor forward/backward已随各model path计入调用账，但其真实wall/显存须另测，不能把一微步等同一个固定硬件时间。

模型QP设计例采用`alpha_D=1,delta_D=.005,delta_total=.02`、4个固定回溯比例；数值尺度与优化精度为预声明求解设置，不是物理g的新门槛，不按当前解析例结果调节。对物理time使用秒这一固定单位。PPO clip=.2、学习率3e−4、minibatch1024、4epochs；它们只是未来实验初始设计，所有比较相同，当前未运行。

物理参数、扰动分布、任务g和场景概率仍必须由实际任务/测量确定；表中S=4只表示算力预算，不假设4条场景已经覆盖真实世界。若任务不能在H内完成，须事前改完整预算或明确UNKNOWN，不能把截断后段当零成本。存储主要为真实batch `O(N*d_record)`、teacher标签 `O(M0QESH*d_label)`的流式/片段缓存及反向激活；可按scenario分块VJP减少峰值内存，但不减少总模型工作量或自动降低统计误差。

## 10. 对照和可证伪项

1. **同目标/同信息的直接鲁棒PO或策略受限TO：**获得同B、F/G、g、因果树、场景、mode与参数化，对同一选中D计算梯度/步骤；相同是正确结果。不能把标准QP、VJP、PPO clip或value bootstrap称创新。
2. **同预算不压缩需求：**在同总模型/交互预算下分配更多入口、较少每入口树/场景，公开覆盖差异；最终在同独立完整任务分布比较。由于压缩权重改变提案目标，不能直接把梯度不同称更正确。另保留相同D的直接full-unroll/VJP实现作为数值等价检查。M/M0=.25只缩减选中需求的某些导数调用，不等于整轮4倍加速；teacher生成及全部M0有限接受仍然计账。
3. **同结构普通PPO＋充分当前模型/正确MPC：**若它解释全部收益，停止共同学习增量主张。模型生成标签与真实数据预算向它公平开放。
4. **mean/stochastic分开：**报告随机训练任务回报、确定性完整任务表现、合同违反和模型误差；mean改善而随机退化/反向情况都保留。没有mean独立验证不发布mean能力结论。
5. **范围停止：**模型导数/后段值未获支持、有限接受全拒绝、共享θ后段改变破坏接续、或同信息强方法达到相同有限资源取舍时，不靠加训练量/改权重/换平台继续包装。

当前最多能提出需求packet压缩、版本对应和紧凑规划表示的有限预算研究假设。它们的独立价值**尚未证明**；当前物理解析例反而显示同信息标准TO完全等价。机器人主平台未定，机器人actor/F/G与实证数据均未具备，本轮没有执行上述任何训练或发布步骤。
