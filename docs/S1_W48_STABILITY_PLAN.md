# W48 稳定性计划：部署已证数值修复，验证真实闭环并补齐关键记录

<!-- W48_EXECUTION_BEGIN -->
> 2026-11-13 执行更新：正式 S5-r36 已隔离构建并完成无积分 StateDiff 核验；8 次 old/corrected N solve 与 4 条实际 scale 的 350 ms 部署核对完成。四个困难 fixture 正方向仍无接纳，四条部署与评分轨迹完全一致；不外推全程不会下降。新 PPO 0/0，原 A/B 正在按计划准备/推进，实际状态和物理 clock 以 [W48 索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w48/phase.json) 为准；下方 planned 描述保留为已发布计划基线。
<!-- W48_EXECUTION_END -->

更新：2026-11-13（Asia/Shanghai）。审查结果基点：`6e9075f48718509d48f51d08d0a8f62a34edfb41`。状态：**planned_not_implemented**。本次发布是审查与下一计划，新增服务器任务、native solve、plant、PPO、GPU physics 均为0；正式S5-r36尚未构建或部署。

[W47最终审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W47-final-review/README.md) · [控制审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W47-final-review/control-review.md) · [调用与GPU](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W47-final-review/calls-gpu-review.md) · [文献](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W47-final-review/literature.md) · [启动提示词](S1_W48_START_PROMPT.md)

## 1. 本轮问题与计划调整

W47完成八条old/corrected固定alpha0诊断，找到四元数自差的浮点首扰动。corrected四例states/actions/times/residual/J精确再生nominal，但没有重算正方向、部署r36或运行新原任务。原W46全程1944calls有1601次正步安装（1598次J下降、3次风险优先恢复）；B明显下沉/侧倾前一秒各50次调用全部J下降。

**W48将已证修复接到真实搜索和闭环任务，回答它是否改善实际行动。** 修复正确、零端点自洽、正步下降、抗扰恢复、完整任务通过分别报告。

上一W47计划把最小修复的原任务评价与唯一R8 donor、最新W46事件完整context/预测误差及预先移动收益耦合过紧。数据缺失后，能产生新证据的实验也无法进行。本计划替代这些准入条件：

- donor是否可用/有效，不再阻塞correctness-only r36的有限求解或一次原A/B E30。
- 最新W46缺失策略/预测不能补造，也不再要求新实验预先拥有；随新回合同步记录。
- 旧困难fixture正步仍不下降、nominal前移仍小，是有效有限结果；实现、身份、原约束和入口正确且无新技术错误后，仍进行一次预声明的A/B闭环因果评价。
- 不保证E30通过。原成功、跌倒、材料、路线判据继续生效；真实失败是本轮结果，不换seed或重复同配置直到成功。

## 2. 开场状态与共同问题

默认D57：SSH `autodl-d57`；CPU树 `/root/autodl-tmp/MCP-RL-w42`，GPU树 `/root/autodl-tmp/MCP-RL`，环境 `work/envs/d57`、`work/envs/w41-mjwarp`。先核main、未提交修改、有效tmux/PID/job/物理clock、cgroup/affinity、RAM/GPU/磁盘。有效W48存在则续接，不重复启动、覆盖他人工作或重启W45/W46/W47/G5。

基线仍为S5-r35 constant、N/r34 strict，冻结binary SHA256：`a32287b0821d9a77687575250dcb2379e370dbc84264fae80a4091027db7345e`。W47的两个隔离诊断binary不能更名冒充正式r36。

保留Go2/TM02质量15.206408kg、PD60/5、每腿23.7/23.7/45.43Nm、2ms held/10×0.2ms物理、0.35s horizon、2Hz/.45/.005 C2步态、r14支持修复及r26/r27/r28机制、原高度项、PANN/658D/2Hz cap、W38奖励和原任务定义。短腾空合法；raw Kδ及描述用0.2m/0.5rad不成为新失败条件。本轮不加PPO、步态/成本网格、完整FDDP/SQP重写或硬件任务。

## 3. Task A：正式最小修复与有限真实搜索

### 3.1 正式构建与接入

控制行为的唯一预声明改变为 `experiments/terrain_memory/w47-state-difference-identity.patch`：MJPC `utilities::StateDiff`在原`mj_differentiatePos`后，对free/ball quaternion**逐分量完全相同或精确反号**的旋转差置零。不同真实微小旋转继续使用原计算；不加邻域容差，不裁K/输入，不改接触、r34数值容差或alpha0安装规则。

这不修改MuJoCo引擎`mjd_transitionFD`，不能宣称动力学雅可比已修好。W47 indexed score-only补丁只是离线工具。正式原任务不得继承`MJPC_R34_SCORE_ONLY`、`MJPC_W47_INDEXED_SCORE`或离线INPUT_POLICY/CONTEXT/REFERENCE文件环境；正常r35参考仍按原算法从当前incumbent生成。

建立真实`s1-go2-r11-S5-r36.json/.patch`及源码/构建/库SHA，核实际读回。至少检查：

- `prepare_s1_go2_r12_support.py`的身份与配置传递。
- `s1_go2_r11_runtime.py`的N、strict、constant及记录分支；尤其现有`controller_build == 'S5-r35'`不能漏r36。
- `s1_go2_r11_policy.py`的full-plan、检查点与lineage。
- `s1_go2_r29_records.py`的candidate、r35分项、事件和manifest白名单。
- evaluation contract的active_controller_build/controller_artifact/selected_mechanism/input_reference_branch，CLI、实际环境和`/proc/PID/exe`一致。

记录改进同版接入但不改变动作、任务时间或求解调用，控制补丁和记录补丁分列。不能把--build字符串当接入完成。

对新编译真实StateDiff做一次**不积分**代数检查：四个已有观察四元数的self、精确antipode、真实非零小旋转，位置/速度/act差保留。小旋转由原模型切空间积分产生并报告大小、单位和非零结果，不做巨量随机网格；不重跑W47已经完成的八条零步配对。

### 3.2 四个可用问题的修前后N求解

固定A148/plan136@2.54、A153/140@2.64、B155/141@2.68、B158/144@2.74，来源均为W45保存问题与W46重建context。D57 `w46-mechanisms`已有完整原件，按manifest核size/SHA；不称exact历史pre-call replay。

每问题固定同一P0、x0/time、mocap/userdata、private gait、模型、constant ubar/F/R及原求解预算。两变体用同一参考文件/hash，不能各自生成不同免费参考。保留全部incumbent/nominal/候选完整策略与实际评分轨迹，不用赢家rollout覆盖反馈reference。

优先复用身份逐项可核的W46 constant old结果；如同入口、参数或生命周期身份不足，才每fixture补一个old N solve。每fixture一个corrected N solve，**最多4 old＋4 corrected＝8 solve**。每solve保持原有限backward/alpha/正则化预算，不增加none/load/R/seed网格。两组各10CPU并行，四fixture排队完成。

必记：每attempt dV/预测下降、真实ΔJ及J0/Jref分项、风险、投影/恢复、chosen alpha、stop/return reason、赢家完整identity、raw与held动作、前移/高度/姿态/接触，以及全部rollout/transition FD/risk FD数量。每次问题内成本比较有效，跨epoch不可把不同参考下J或ΔJ相加当闭环进展。

### 3.3 四条实际部署路径核对

每fixture对corrected最终完整policy按正式`ActionImpl`的真实scale、插值/归一化/Clamp/H-QP，从同一x0做一次350ms fixed-policy诊断，**最多4条，7000个0.2ms子步**；窗内不重规划。与该solve对同一策略保存的实际评分rollout比较。

正步scale=1原搜索为`ActionIndex/RolloutDiscrete`；nominal/incumbent须对照其实际评分路径，不能让scale≠1旧策略强用indexed scale=1。alpha已在feedforward中，不二次添加。保存首次raw/clip/projected/held与状态差。ActionImpl归一化可能使表示不逐位相等，这只是待核边界，不预设为第二个bug。

路径差异据实记录。可定位的错误调用、对象/单位错配、配置遗漏或不合法数值属于先修的技术问题；真实非零误差敏感性、物理失稳或仍无正步本身，不取消第5节闭环评价。不展开任意K插值/平滑/epsilon搜索。

## 4. 随新任务补齐预测—执行记录

W46缺件来自事件选择：约0.12s内存环、前0.1/后0.2s窗口，没有首次实际held目标大跳变或实际膝反转触发。A6.86/B6.15s所需策略/上次预测从未保存，完整上传也无法恢复。新记录随原回合生成，不重跑W46。

### 4.1 每次规划边界的小型记录

在`agent.planner_step()`覆盖旧计划前关联：

- actual x0/time、实际incumbent P0的generation/call/plan/hash，当前mission/cap/目标/model/context身份。
- 上一个**实际部署策略的accepted rollout**在当前origin对应的预测状态与source plan/call/node/time/hash。
- MuJoCo位置切空间与速度误差；位置m、姿态rad、关节rad、线/角速度分列。
- `accepted_states`是被评分实际rollout，`reference_states`是反馈中心；不能互换，不能取本次新解冒充上次预测。缺匹配节点或已超horizon标unavailable，不填零或借旧W45 delta。
- 同call的J0/Jref/风险、预期与实际下降、候选来源、内层stop及顶层return；原全程小型记录继续保留。

各origin保存紧凑状态对和身份，不要求全程保存所有大候选/导数。采用实际同时间节点；确需插值时记录规则与误差，不能插出假的exact历史状态。

### 4.2 完整事件包

保留原事件，新增每回合首次：相邻2ms**实际held目标差>1rad**、膝关节实际力矩在±45.43Nm限幅端反转。力矩取runtime `mj_forward`后同一时刻真实值。阈值仅选窗口，不改任务终止/物理限幅。阶段切换和terminal也要能关联完整部署对象。

使用前0.1/后0.2s，重叠合并。包内关联pre-solve x0/act/time/mocap/userdata、R32 private gait、当前材料proxy/参数、P0、最终完整部署policy（reference x/u/time/K/H/scale/representation/metric）、accepted rollout、上一预测、同epoch r35 ubar/F/R/hash、任务与观察/Actor身份。仍被使用的P0不能因旧origin误丢；私有状态在原程序相应时点直接记录，不事后猜测。

完整策略与参考按不可变identity/hash去重，transport buffer不是历史。不逐call复制所有候选与完整导数；若固定事件需要该call的nominal/alpha0/最好正步，可保留这些对象，不扩到所有calls。不要先制造重复几十GB，再事后删除原件。

最小执行策略约0.88MB、每call两个37维FP64状态约592B只是未压缩设计估算；实际大小在运行中计量。原500Hz轨迹、有效物理/路线信息及已经生成的旧原件保留。

记录代码、schema和输出路径就绪即可启动原E30进程。在该attempt自身首次正常规划后、首个held物理推进前，核新字段实际写出和身份可读回，不要求启动前已有runtime记录，不额外开plant做记录测试。首次没有上一预测可合法标unavailable；字段/来源身份必须存在，不能填零伪造。关键记录未启用是技术交付缺陷，保留该attempt技术状态并修复；不能完整跑完才发现recorder没有启用。

## 5. Task B：各一次修复版原A/B E30，按原阶梯继续

### 5.1 启动条件与自主推进

Task A有真实构建与有限结果，N/r34 strict/constant/原物理任务参数读回正确，正常求解/部署入口、记录schema和输出路径已接入，且无未解决技术错误后，**自主启动A/B各一次r36 evaluation-only E30**。实际记录在同一attempt首次正常规划后、首个held物理推进前核验；缺关键字段按技术错误处理并保留attempt，不要求启动前已有新plant记录。不为重复许可暂停，不等donor，不要求旧困难fixture先证明完整运动收益。

有限结果、spec/patch/build/source SHA、记录schema、合同与真实W48 phase先推Git。已有W46同seed/checkpoint作历史对照，不增跑r35 E30；这不是跨版本统计显著性结论。一版主控制器，两流同版，不混donor、成本/步态改变或新训练。

A固定`outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w44-a/PANN_PPO_S5-seed1/checkpoint-014737.pt`；B对应`w44-b/PANN_PPO_S5-seed2/checkpoint-014737.pt`。保留完整resume ep00428/ep00417（后缀按实际文件）、actor/critic/optimizer/norm/RNG/fork/lineage，tail0、excluded145，各14737/update57。权重固定，PANN observer/材料memory按真实观测因果更新；新增PPO0/0，不自动执行未来15249/59。

使用`scripts/run_s1_go2_r12.py`单次evaluation入口，明确attempt/evaluation-start-unix/checkpoint/level/seed/workers/build/contract/condition nominal；不用要求本phase新训练的job-dir。创建W48新job/attempt，不覆盖旧轮。启动后核两条真实clock、轨迹和planner记录增长，tmux存在不等于实际运行。

### 5.2 原资格保持

| 级别 | 原资格与推进 |
|---|---|
| E30 | 独立观察满30s，合法实际材料去程，满足原3m目标到达容差并进入原转向；不新增x≥3.0硬门，也不强加完整返回 |
| E60 | 同case前级通过后，独立满60s，至少一次合法往返，回家连续稳定2s |
| E300 | 同case前级通过后，独立连续5合法往返、270s内完成路线，完整观察300s及原稳定条件 |
| E600 | 同case前级通过后，独立连续10合法往返、540s内完成路线，完整观察600s及原稳定条件 |

每case自己过前级才进后级；A通过可继续，B失败不阻塞A，也不借A替B越级。一次真实失败结束attempt、后级blocked；不换seed、不重复同配置追成功、不截长轨迹前缀或拼接资格。完整足迹实际Fz>0、主应变max(abs(principal strain))≤.02、原非足接触/跌倒/路线判据保留。

技术中断时Actor不能恢复中途plant/材料/observer/MPC/任务全状态；保留interrupted，必要技术重启另建独立attempt并记录diff/原因，不能拼接资格或把自然失败改成技术失败重跑。

### 5.3 比较与完成条件

W46与新A/B分别报告实际时长/终止、合法去返往返、材料加载、前进回退、最低高度/姿态、首大held跳变/限幅反转、正步/停滞分布、预测—执行误差、J0分项/Jref/风险和规划/反馈/I/O墙钟。风险优先J上升单列，避免极值主导均值；成本下降率不替代稳定/路线完成率。

若仍失败，以首个有完整证据的偏差区分执行路径/模型、真实误差下反馈恢复、目标冲突或阶段衔接。无证据保留unknown；本轮不自动再换算法。成功也只报告实际通过的级别，不提前宣称材料记忆收益。

## 6. Task C：独立补证、必要输入恢复与有限GPU工作

C可与构建/CPU工作并行准备，host总额度2，不成为A/B准入，也不在两条10CPU原任务之外再开重型native solve。

### 6.1 现有GPU重复身份逐时差

W47纠错数据的6组重复场景，声明policy×初态×origin×model相同、native极值一致，GPU极值略异。先在D57现有paired/scenario数组上按完整身份分组，计算重复组逐时状态/命令差、首分叉与不同单位极值，核隐藏初始化/world映射/接触和solver字段可识别性。未保存字段明示缺失，不预设纯FP32或已证不确定性。

主体0新增physics。历史数据仍是W45旧策略，不是r36等价排序。`scripts/s1_go2_w46_gpu.py`仍向W46写输出，**不得直接重启**；如复用组件须W48独立输出根和identity，旧结果只读。

本轮r36完整policy生成且接口足够时，可在新入口做**最多一次同状态FP64动作批次**：四个fixture的old/corrected策略×原态及已有可验证±扰动，名义≤24组，缺扰动不补零，身份去重；加入严格self/antipode自差规则，raw/ctrl-clipped/H-QP-held独立copy。仅计算动作，0动态积分，不重跑20/350ms。若只有GPU公式＋host H-QP，明确范围，不称逐场景完整native Policy::Action等价；编译/传输/host桥接都计时。接口不足则完成已有数组结论，不为显存占用扩大任务。

分别记录`policy_generation_build`和`action_evaluation_build`及实际StateDiff规则。old生成的policy若也经共同corrected mapper执行，标为“旧/新策略在共同r36动作语义下的比较”，不能据此声称旧/新原生路径等价。self/antipode判断必须作用于与native一致的实际插值、归一化后的反馈reference，不能改在未插值原数组上判断，也不引入epsilon。

### 6.2 仅恢复一个确知R8科学输入

实时Release元数据已核：`tm-s1-go2-wb-support-r8`中asset681937433，`project-s3-formal-v1-single-yaw_plus5-trajectory.npz`，28,100,505B，SHA256 `079f57e1ebad260c9cdac842dc089012e29cb46992ab64d2c31cd9d380b61625`，state uploaded。

D57确无文件时，可沿既有授权服务器网络路径从[确切资产URL](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-wb-support-r8/project-s3-formal-v1-single-yaw_plus5-trajectory.npz)只恢复这个输入到独立只读科学输入目录，核size/SHA/NPZ schema。这是服务器实验输入，不是发布后向用户电脑追加归档。不要下载整批Release、重跑R8或绕过此前电脑批量读取审批拒绝。

只核时间范围、500Hz x/u、任务phase与源resolved cadence/GetPhase重建gait phase是否足够；本轮不执行donor solve或在线多家庭，不与identity-only评价混改。连接不可用则明示支线受阻，A/B继续。D57仅用既有授权学术代理，不自动借电脑VPN/SSH转发、Clash或未授权直连，凭据不进Git/日志。

## 7. 并行资源与后台

W47预检25CPU、120GiB RAM、97887MiB GPU，另有无关SuperMimic占34318MiB，仅是历史快照。现场实测，不动无关进程。

| 工作 | CPU额度 | 安排 |
|---|---:|---|
| 有限问题组1，后为原A |10|独立进程树/可写目录，Torch/BLAS/OMP各1 |
| 有限问题组2，后为原B |10|同冻结实现并行，各case独立推进 |
| C原位分析/可选动作GPU |2总host|含host确认，结束即释放 |
| 主控/I/O/发布/系统 |约3|Git短写串行，保留余量 |

build-j6先完成再双重计算；共享环境/资产只读，可写upstream/build不经共享可写软链。两原任务运行时不再加两组10worker有限solve。XLA预分配false、JAX x64保留；GPU按A/B实测峰值和剩余准入，不假定96GB独占/硬分三等份，不用无效PPO或重复长窗填利用率。

**所有新后台deadline=null、wall_deadline_enabled=false，省略--deadline-unix；无固定墙钟完成时刻，无到点kill。** 等待/轮询timeout不变成任务期限。保留物理30/60/300/600s、有限工作量和单次网络timeout；按工作完成、原判定、明确技术失败或用户停止结束，无无限重试。窗口/对话结束不能杀后台或写成已完成。

原阶梯仍运行时推真实running、PID/job/clock及已产结果，完成后更新final并核进程。暂停只对精确PID，不宽前缀kill。

## 8. 交付与长期路线

紧凑Git优先，要求的已生成原件由D57增量Release直传，核commit/URL/size/digest。回执丢失先查已有资产调和，不重复上传。Git/index短写串行合并最新main，不强推、覆盖并发。实现、有限结果、启动、自然终结、交付分别记录。

**GitHub核验后不在用户电脑额外下载、打包、镜像或归档；服务器checkpoint/resume、原件、视频和上传副本不自动删除。** 必要服务器输入恢复不改变这项要求。不存在/未生成/blocked/技术中断/物理失败分开，不造空壳。

最终给实际commit/spec/build/sourceSHA、控制及记录变化、正步/部署路径结果、A/B每级资格、首偏差证据、新rollout/FD/physics/PPO和资源计量、C结论、Git/Release和真实续接点。没有完整任务通过就明确未通过。

长期仍为稳定原材料任务→跨足/返程有效历史曝光→匹配RAW/PANN及冻结/打乱历史对照→候选内材料历史传播与必要动作权限→泛化和Go2/NX。W48先回答最小修复的真实闭环价值。
