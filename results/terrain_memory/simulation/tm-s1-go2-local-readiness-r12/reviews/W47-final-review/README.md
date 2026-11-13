# W47 最终审查：数值缺陷已定位，正式闭环尚未验证

审查日期：2026-11-13（Asia/Shanghai）。结果基点：`6e9075f48718509d48f51d08d0a8f62a34edfb41`。前次计划：`8f254aee8401ffca821cb360e7c33a01038b9a74`；最近原任务结果：W46/r35，`15ee620b8a909246a12baf999c136c2be299655b`。

**结论：可以继续实验，下一轮应将已经验证的最小 StateDiff 修复接入正式 r36，有限核正步与真实部署，再进行 A/B 各一次原 E30。S1 仍未形成稳定合法材料往返，不能把 W47 的诊断完成当成原任务成功。**

W47 找到真实可复现的四元数自差数值缺陷，并消除了四个保存问题的零端点分叉。新增全程调用表同时修正了上一轮过度集中于四个困难 fixture 的解释：1,944 次原调用有 1,601 次正步安装，并非全程搜索停滞。下一步必须同时判断搜索下降、执行一致性与真实支撑恢复。

[W48详细计划](../../../../../../docs/S1_W48_STABILITY_PLAN.md) · [完整启动提示词](../../../../../../docs/S1_W48_START_PROMPT.md) · [机器交接](handoff.json) · [控制核验](control-review.md) · [调用/GPU核验](calls-gpu-review.md) · [一手文献](literature.md) · [审查验证](validation.json)

本次仅仓库/源码/提交数值的审查与计划发布：新服务器任务、native优化、plant、PPO、GPU物理均0。W48为`planned_not_implemented`，没有创建正式r36构建，也没有把历史tmux状态当现场存活检查。

## 1. S1现在到了什么程度

长期问题仍是：同一世界地块经过前足、后足、返程加载后，如何利用持久材料状态，让未来支撑需求影响当前加载、落点和重访。成功标准包含合法完整路线、稳定恢复和因果记忆收益，之后才是条件泛化与Go2/NX。

真实联合链已存在：冻结因果PANN→658D Actor观察→2Hz速度cap（.18–.25m/s）→逻辑50Hz全身MPC→500Hz时变关节目标→PD60/5→5000Hz原物理。逻辑频率不是墙钟实时；权重冻结评价中，observer/history仍因果更新。

| 层次 | 已经成立 | 当前边界 |
|---|---|---|
| 刚地执行 | R8/S3项目9/9原任务、300s 3/3 | 不是当前r35/TM02资格 |
| 材料联合运行 | W34合法去程3次、W35 1次，已有返程进展 | 稳定合法完整往返未建立 |
| PPO与恢复链 | A/B各14737转移、update57、完整checkpoint/resume | W45/46/47新PPO0；更多样本不是当前证据指向的首要修复 |
| 数值接纳 | r34 strict保持；大量原调用有合法正步 | 正步下降不保证支撑、恢复或路线 |
| W47修复 | exact quaternion self/antipode零差，四对固定轨迹精确回到nominal | 正方向未重算，正式r36未部署，原闭环0 |
| GPU | 动作与既有paired结果有纠错和更清楚边界 | 尚无完整native同质量排序/端到端加速证据 |
| 材料记忆研究 | PANN提供因果历史特征 | 候选尚未沿各自未来动作完整传播材料内态；cap不能直接分配足时/落点/卸载；因果收益未证 |

A/B共有祖先6545/update25，PANN唯一转移22929，另RAW256，R12 distinct23185。不能将两条14737简单相加当独立训练样本。完整历史与长期路线见[主计划](../../../../../../docs/TERRAIN_MEMORY_PLAN.md)。

## 2. W47实际做了什么，没有做什么

[W47 phase](../../w47/phase.json)记录`completed_finite_work_and_delivery`；后台`s1-w47-finite`自然完成，pane exit0，结束核验没有有效W47计算/发布进程。`deadline=null`，没有墙钟截止杀任务。

| 工作量 | W47新增 |
|---|---:|
| 固定策略350ms native rollout |8（4 old＋4 corrected）|
| held区间 |1400|
| 0.2ms native物理子步 |14000|
| 独立诊断时长之和 |2.8s，不是连续原任务|
| native optimizer solve / transition FD |0 / 0|
| donor held / donor solve |0 / 0|
| 原plant episode / 新PPO样本与更新 |0 / 0与0|
| GPU physics / GPU新分配 |0 / 0|
| 隔离构建 / paired墙钟 |63.294s / 35.256s|

Task C在D57原位处理43个文件、122,624,900B，约2.49s，复算调用与已有GPU数组。Task B完成可用性负结果，没有实施donor干预。预设三组affinity不等于持续三项重负载实际并行。

新增原件ZIP已发布[W47 Release](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-local-readiness-r12-w47-identity-part01)，96,738,357B，远端digest与manifest一致。本审查重新核的是元数据，没有下载payload；见[Release库存](release-inventory.json)。

## 3. 已确认的错误：相同姿态被注入极小非零反馈

### 3.1 因果证据

四fixture×五次backward的20个constant alpha0首分叉都满足：实际state与reference逐元素相同，reference动作与nominal实际u一致，时间一致，首个H-QP为内点恒等。MuJoCo3.2.3四元数切空间自差仍出现约`1.69e−20～2.98e−19`的非零分量；K把该数当真实误差，解释了首个输入差。

第一attempt配对结果：

| 保存问题 | 旧alpha0相对nominal的ΔJ | 旧全窗最大命令差/rad | 修复ΔJ | 修复350ms前移/mm |
|---|---:|---:|---:|---:|
| A148 |−9.6054e−10|9.1686e−5|0|−2.742|
| A153 |−6.0345e−7|8.1738e−5|0|+1.494|
| B155 |+.0234713|1.026335|0|+1.081|
| B158 |+.0181906|1.108094|0|+.354|

old诊断四处精确再现存档alpha0；corrected四处states/actions/times/residual/J精确再现nominal。这不是笼统猜测“浮点误差”：首扰动、相同输入、最小补丁和结果恢复已经组成有限证据链。另一方面，毫米级前移和A148后退说明nominal本身并不等于稳定移动。

来源：[saved-array-summary](../../w47/task-a/saved-array-summary.json)、[paired-evidence](../../w47/task-a/paired-evidence.json)、[诊断构建](../../w47/task-a/diagnostic-build.json)；算术和补丁复核见[control-audit](control-audit.json)。

### 3.2 修复边界

[identity补丁](../../../../../../experiments/terrain_memory/w47-state-difference-identity.patch)在MJPC `StateDiff`调用原`mj_differentiatePos`后，仅对free/ball quaternion逐分量完全相同或精确反号的旋转差清零。没有epsilon邻域、K裁剪、真实小旋转抹平，也没有改变位置/速度/act差。

该补丁不修改引擎`mjd_transitionFD`。不能据此说动力学雅可比、真实微扰恢复或接触问题都已解决。冻结r35二进制仍是`a32287b0821d9a77687575250dcb2379e370dbc84264fae80a4091027db7345e`，W47只有两个隔离诊断binary。

另一个[诊断补丁](../../../../../../experiments/terrain_memory/w47-fixed-candidate-diagnostic.patch)让保存候选走真实`ActionIndex/RolloutDiscrete`。配对只覆盖该scale=1路径；正式`ActionImpl`还涉及插值、归一化与实际scale。下一轮需核同一完整policy的实际评分/部署路径，不能将scale≠1的nominal强塞indexed scale1，也不能二次添加已写入feedforward的alpha。

### 3.3 尚未回答的方向问题

旧四fixture三分支共540个正候选全部预测下降、实际J增加；公式核验无矛盾，none分支也失败。W47没有重新求解，所以目前既不能说修复已经解决正方向，也不能说修复无效。应该用现存完整问题有限重算，而不是继续重复八条零端点或扫参数。

## 4. 全程证据：下降经常发生，失稳仍然发生

独立解析[完整1,944行CSV](../../w47/task-c/planner-calls.csv)，其中40次为启动调用：

| 范围 | 调用 | 正步安装 | 同问题J下降 | 风险优先J上升 | 内部停滞 |
|---|---:|---:|---:|---:|---:|
| 全部 |1944|1601|1598|3|343|
| A moving |1222|1066|1063|3|156|
| B moving |482|317|317|0|165|
| 两流moving |1704|1383|1380|3|321|
| 两流Stand |240|218|218|0|22|
| 去掉启动 |1904|1561|1558|3|343|

全程正步安装82.36%，moving81.16%。343次内部停滞仍有137次安装nominal，206次保留incumbent；内层stop与顶层return不是同一字段。3次J上升是晚期风险优先降低trunk violation，不能当一般成本下降，也不能据极端J计算全程均值。

**B明显下沉前一秒的50次调用，和明显侧倾前一秒的50次调用，都全部安装了降低本次J的正步，停滞为0。** 对应描述性阈值是z<.2m@8.464s、|roll|>.5rad@8.572s；不是新增任务失败阈值。B首膝反转6.15s前也有48/50正步。

局部困难确实存在：A最长连续moving停滞36次（3.12–3.82s），B40次（3.08–3.86s）；A首>1rad held跳变前一秒41/50停滞。要同时保留早期停滞和后期持续下降时失稳的证据。

因此，“全程没有下降所以跌倒”已不成立。应检查同问题优化收益怎样转化为真实下一步状态、支撑/恢复，以及重规划后参考和目标怎样变化。不同epoch的J不在共同基准，不能把1,598次下降相加当任务进展。详见[完整调用审查](calls-gpu-review.md)。

## 5. 原任务为什么仍没有完成

最近实际原任务仍为W46：

| 指标 | A | B |
|---|---:|---:|
| 原E30时长/终止 |26.432s，floor–trunk contact|11.638s，body_fall|
| 墙钟 |65.59min|40.42min|
| 最大body x |.626506m|.396364m|
| 实际材料加载 |无，足最远x .795019<.8|有，首次完整足受载9.122s|
| 合法去/返/往返 |0/0/0|0/0/0|
| 相邻2ms held目标最大变化 |1.535358rad|1.468642rad|
| 膝±45.43Nm完整反转次数 |37|5|

E30通过0/2；后六个阶段blocked，不是六次额外失败。W47全部原任务阶段未启动，也不能累计成八次新跌倒。W45对应A13.072s/B5.582s；W46活得更久但A前进更少且回退，不能仅按存活时间判定更稳定。

B在首材料受载前已经明显下沉/侧倾，首次受载时约z=.1455m、roll=−.6426rad；最大材料主应变.00649112低于.02。A未实际材料受载。因此当前失败首先要求解释移动与支撑控制，不能归结为材料模型或PPO训练量不足。

90.86Nm是相邻时刻从+45.43到−45.43的变化，不是实际施加90.86Nm或力矩限幅失效。raw反馈巨大也不是实际关节角。同步仿真的墙钟慢主要降低实验吞吐，不能直接当作物理计算延迟致跌倒。

可以确认与尚待确认的原因分开：

| 类型 | 已有证据 | 尚不能下的结论 |
|---|---|---|
| 数值一致性 | 已找到并修复保存问题的自差首扰动 | 它是所有原任务失败的唯一原因 |
| 部署与恢复 | 大held跳变、膝反转、下降仍下沉 | 未有完整同origin预测可归因哪一层 |
| 目标/模型 | 同问题J下降与原路线未完成同时存在 | 高度项、参考惩罚或某个参数单独负责 |
| 数据记录 | 关键事件未触发完整保存 | 用户漏推已生成文件 |
| 实验流程 | 缺donor/缺历史快照阻塞了最小修复闭环评价 | donor算法已经失败 |
| 计算资源 | native规划占主要墙钟；GPU仍不等价 | 多占显存本身会改善控制 |

## 6. 为什么上传齐全仍无法重建关键失败

[B fixture manifest](../../w47/task-b-fixture-manifest.json)检查A50/B34个交付ZIP的成员，A6.86s plan305与B6.15s plan267及上一预测没有保存。现有recorder约.12s环、事件前.1/后.2s，触发按nominal饱和/raw Kδ/clip/restoration等，不包括首次实际held跳变/实际膝反转。数据未生成，完整上传也无法补回。

已有A3.846s plan178、B3.106s plan161完整policy可提供线索，但没有全部pre-call私有context，仍不是exact replay。无需指责本轮推送，也不应事后补造K/H、gait或预测。

W48记录每call紧凑actual/previous accepted prediction配对和身份；完整context/policy用固定事件窗口、重叠合并、hash去重。`accepted_states`与反馈`reference_states`分开。在覆盖旧计划前保存P0和上一实际部署策略的预测，缺节点标unavailable。初始无上一预测合法，不填零。首实际held>1rad、首膝完整反转、阶段/terminal关联完整对象；阈值只选诊断。

该记录随同一次原E30生成。在同一attempt正常首次规划后、首个held物理推进前核写出，不先另开plant或要求实验启动前已有runtime记录。

R8也未全局丢失。[source manifest](../../w47/task-b-source-manifest.json)证明D57本地不可用、下载0；[实时Release元数据](r8-release-metadata.json)仍有唯一28,100,505B的yaw_plus5轨迹。W48允许仅在D57通过既有授权路径恢复该科学输入并核SHA/schema/time/phase；不下载整批、不在用户电脑归档，本轮不做donor solve，也不阻塞r36。

## 7. GPU应怎样继续，才能贡献有效证据

W47纠正raw投影别名、动作/末态时标、扩窗解释及capture计数。已有24场景只有18种有效组合、6种完整策略，源是W45旧策略。

初始同state held差约2.49e−16rad；首2ms关节位置差约1.46e−5rad；+10ms下一hold差可1.50136rad；350ms最大held差1.75833rad、关节位置差.130953rad。784.29rad是离线把各自已分叉状态带入raw公式得到，不是实际关节转动或实际输出。

另有6组声明执行身份相同的场景：native极值相同，GPU min-z/max-roll略异。尚未核所有隐藏初始化/接触workspace，不能直接断言纯FP32或GPU不确定性。下一步原位读现有数组算重复组首分叉，0新physics。

可选一次同状态FP64动作批次名义≤24组、缺扰动不凑；区分policy生成build与动作计算build，正确使用插值归一化后的reference，raw/clip/held独立。范围若只有GPU公式＋host H-QP则如实说明，不冒称完整native Policy::Action。历史`s1_go2_w46_gpu.py`仍硬编码W46输出，禁止直接重启覆盖；新入口写W48。

## 8. 文献给出的方向与适用边界

本次追加六篇一手文献，完整出处与适配限制见[literature.md](literature.md)。最直接的四项启示：

1. **ICRA2025，Shirai等：** 平滑接触模型的线性反馈可能预测现实接触不能提供的恢复方向。研究是双臂操作，不能据此断定Go2同因；它提示要测真实误差下支撑/活动面与恢复，不能只看nominal下降。[论文](https://arxiv.org/pdf/2411.06542)
2. **IJRR2026，Suh等Contact Trust Region：** 局部模型可信范围结合接触力可行性。其准动态模型与Go2动态行走存在关键差别，当前只能作为新增证据支持后的候选机制，不直接移植。[全文](https://arxiv.org/html/2505.02291v2) · [出版社](https://journals.sagepub.com/doi/10.1177/02783649251398875)
3. **RA-L2024，Khazoom等：** 分离优化精度、模型/离散误差与闭环效果；更精确子问题不自动抵消执行误差。本项目先测预测—执行链，r34标准不放宽。[全文](https://arxiv.org/html/2407.10789v2)
4. **MuJoCo全身MPC2025，Zhang等：** PD在模型内、全身规划与反馈分频，反馈和差分选择须实际验证。作者任务/离散方式与定制175 held问题不同，不能只看方法名判等价。[全文](https://arxiv.org/html/2503.04613v2)

这些是研究方向的依据，具体用于本项目的判断属于推论。W48保持一个控制变化；是否后续研究模型/执行、接触可信区域、目标冲突、阶段衔接或donor，由新完整证据决定，不同时混改。

## 9. W48的可执行推进与资源安排

上一计划将correctness修复的评价与donor、新近历史context和预先移动收益耦合过紧。这次明确调整：这些缺项不再阻塞一次最小修复原任务评价。原物理、成功、失败、材料和路线标准不变。

| 顺序 | 工作 | 真实上限/输出 |
|---|---|---|
| A1 | 正式r36接入：identity控制补丁＋非行为记录 | spec/源码/构建/库/合同/实际exe读回；一次无积分代数检查 |
| A2 | 四个现有constant/N问题的修复后搜索 |4 corrected，必要时≤4 old，总≤8solve；无网格 |
| A3 | 四个corrected最终完整policy真实ActionImpl |≤4×350ms，7000物理子步；同策略实际评分对照 |
| B | A/B各一次r36原E30 |原W44各14737/update57、seed1/2，新PPO0/0；已有W46历史对照 |
| 后级 | 同case通过才E60→E300→E600 |真实失败后级blocked，另一case可独立继续 |
| C | GPU已有重复身份逐时分析、一个R8输入恢复 |主体0physics；可选一次≤24动作批次0积分；donor solve0 |

两CPU流各10额度，C总host2，约3留系统/主控/发布，数值库线程各1；build-j6先于双重负载，可写build隔离。有限求解结束后额度转为原A/B，不额外挂两组10worker。W47预检GPU97887MiB且无关SuperMimic占34318MiB只是历史快照；现场实测峰值和剩余，XLA预分配false、JAXx64保留，不动无关进程。

所有新后台`deadline=null`、`wall_deadline_enabled=false`、省略deadline-unix，无固定墙钟完成时刻/到点kill。保留原物理时长、有限工作量、技术失败和用户停止；工具等待timeout不是任务期限。Git/Release由D57增量直传核commit/URL/size/digest后，不在用户电脑额外下载、打包、镜像或归档；服务器原件/checkpoint/resume/视频不自动删除。

## 10. 本次阅读与复算覆盖

[仓库库存](repository-inventory.json)：结果基点8,657个tracked文件、1,025,161,586B全部完成字节读取/哈希库存；相对前次计划34个变化文件、41,749,875B，JSON/JSONL解析错误0。该覆盖不等于声称逐字节语义理解或重放每个历史Release。

语义复核集中于项目记忆/历史入口、W47全量变化、两控制补丁及实际build、完整四fixture提交证据、全1,944调用CSV、runtime/policy/recorder/GPU及最新原任务记录。两份stdlib审查程序复算540正候选、20个首分叉、8个paired、全调用分层和GPU重复身份，并验证两补丁应用后的源码SHA。

本审查没有重新下载W47/GPU Release原数组，没有登录D57重放。W47执行者在D57完成的原数组重算、本审查的提交JSON/CSV独立核验、实时Release元数据三者分别标注。新原实验为0，所有实际新控制结果留给W48执行。
