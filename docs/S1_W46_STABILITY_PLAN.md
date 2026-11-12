# W46：检验承载相关的控制代价，推进原任务稳定性

## W46 actual implementation and finite evidence — 2026-11-12

S5-r35 is now built from reviewed main a34bc45393da1e552df135f856df6f98c762c51f. The 12 declared finite solves and 36 fixed-policy 350ms native diagnostics are complete; no PPO or original-task physics was added by that comparison. The selected original-task branch is **constant**, with mechanism N and r34 strict acceptance retained. Load and constant selected identical complete policies in all four contexts, so no load-information benefit is supported.

A153 provides limited useful evidence: nominal target jump 1.08418→0.56936 rad and torque jump 65.3074→34.2848 Nm, similarly under both recorded perturbations. Minimum height decreases 1.152mm and J0 rises .0002356; terminal vertical recovery remains positive and small forward displacement increases. The other three contexts are unchanged; B158 still has a bad high-jump incumbent. This supports one original A/B evaluation, not a stability claim.

The common-reference residual reconstruction error is zero; all four disabled-penalty native state/action/residual comparisons are exactly equal to r34. Candidate cost recomputation error is at most 4.45e-16, held-u residual FD error at most 7.96e-11, direct state C zero. Actual evidence, frozen binary/spec identity and the new phase are under [w46](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/). At the 2026-11-12 launch, both original evaluation-only E30 ladders were observed running in tmux s1-w46-a/s1-w46-b with advancing physical clocks (phase.json is the live authority); each keeps its own W44 checkpoint014737/update57, new PPO0/0, no wall deadline.

The independent GPU study has completed 24 old-policy scenarios and 42000 world-steps including the 20ms prefix, plus 48 warmup steps. It uses GPU FP64 feedback algebra and the exact old native H-QP on host, then an explicit cast to MJWarp FP32 dynamics. Same-state held-action error is 2.49e-16rad, but 20ms cross-engine command differences reach 1.501rad; no equivalent ranking or online acceleration is established. Original plant Euler is not re-simulated; the paired dynamics are native implicitfast and MJWarp.

Implementation: [spec](../experiments/terrain_memory/s1-go2-r11-S5-r35.json), [finite selection](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/mechanism-evidence.json), [phase](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/phase.json), [GPU evidence](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/gpu.json). The W45 facts and original bounded protocol below remain historical provenance and operating constraints.


日期：2026-11-12。结果基点：`3e3f69c440d6a33c14131563e5f485d88da3a803`。依据：[W45最终原件审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review/README.md)。

**本文件是下一轮实施计划，尚未实现或启动r35/W46。当前实际运行版本仍为已结束的r34/W45。** 本次审查没有启动服务器、物理仿真或PPO。用户已要求下一轮继续解决稳定运行、尽可能双/三任务并行、后台无固定墙钟完成时间、GitHub核验后不在电脑另做归档。

## 1. 本轮结论与需要回答的问题

W45固定各自update57，新增PPO0/0；A/B原E30分别13.072/5.582s侧翻，合法去程/返程/往返均0，后续E60/300/600全部blocked。674/300次planner调用中，新alpha0策略安装均0。因此r34接纳修复有效，但不是完整稳定性方案。

A在12.830s已出现|roll|>.5rad（描述性指标，不是新失败阈值），13.048s才首次实际材料受力，13.072s侧翻；末段5000Hz材料记录只有18个正受力子步，累计对应3.6ms。B全程未加载材料。39次PANN置零动作对照的cap差均为0，当前失败不能直接解释为材料记忆训练不足。

新的控制证据指向**名义策略的实际控制变化及其在接触附近的恢复能力**：

- A call153/plan140@2.64s为有意义的alpha=1改善，原J从.11683448降至.08909507。MaterialSupportHeight降低.05958964，Effort增加.03052700，Upright/Gait变坏；但候选同时改善了最低高度与前移，不能把它简单说成错误高度项或毫无价值的动作。
- 在native模型本身，A的投影后目标相邻2ms可变化1.41880rad；实际约1.16403rad，之后某膝关节力矩反向变化90.86Nm。B@2.764s的native/actual目标变化约1.23629/1.23587rad。瞬时力矩限幅与QP可行并没有限制这种连续命令行为。
- native也预测了早期无足承载片段，B的前10ms状态/动作与实际很接近。仅修Euler/implicitfast差异不能解释全部事件。短暂腾空在原步态中允许，不增设“无接触即失败”。
- 大Kδx具有参考表示依赖性；等价的仿射重定中心可改变该数值而不改变命令。优化/验收依据应是完整策略的**投影后qtarget、真实PD力矩、支撑、姿态和任务进度**，不是K的大小或raw feedback是否变小。

本轮问题是：在当前完整native求解和原物理条件下，加入明确的输入参考偏离代价，并按各腿预测法向承载份额分配该代价，能否减少过激的实际命令，同时保留恢复高度与前移的能力，进而改善原任务？份额不是绝对承载能力：总力很小时也可能出现某腿w接近1。这是待验证的假设，不是已找到唯一根因或稳定性保证。

## 2. 唯一主机制：完整执行输入相对共同参考的代价

拟实施`S5-r35`，保留r34接纳规则和原控制/物理链。只改变移动阶段的输入代价；不同时调整高度权重、目标高度、步态、horizon、PPO、PD或真实接触参数。Stand沿用现有native平衡参考。

### 2.1 共同参考与承载安排

每个真实problem epoch开始时，先用当前实际状态/时间重评分输入incumbent的完整native闭环策略。由这一次rollout记录每节点的**实际投影后目标角**`u_bar[k]`及每腿法向支撑力`F_bar[k,j]`，形成当轮共同参考。native力与材料proxy力遵循现有模型的唯一合并口径，不能重复计数。

这份参考的时间、状态/context、完整策略身份、u_bar、F_bar、R及内容hash在该epoch内冻结。所有incumbent、nominal、alpha候选使用同一份参考，重新按同一新目标评分；参考不能随候选变化，也不能让某个候选把自己的动作当作参考从而使新增成本为零。参考或安排变化时必须更新R32问题身份并清除不再有效的分数/数值状态，不能跨epoch比较旧成本。

**当前incumbent不是已证明稳定的行走参考。** HiLQR论文用稳定Convex MPC生成完整walking reference；本仓库的R8提供成功控制结果和Stand平衡参考，尚没有可直接替换的稳定moving reference。本轮是受论文启发的局部机制检验，不宣称复制了论文的完整稳定控制器。如果共同参考本身震荡、失去恢复能力，新项可能保留坏动作；有限比较必须正面检查此风险。

### 2.2 三个有限比较分支

记`J0`为r34原完整native目标，`N=176`为当前成本归约节点数，175个控制节点增加：

\[
J_{ref}=\frac{1}{2N}\sum_{k=0}^{174}\sum_{j=1}^{4}
(u_{k,j}-\bar u_{k,j})^T R_{k,j}(u_{k,j}-\bar u_{k,j}),\qquad J=J_0+J_{ref}.
\]

终端无新增控制项，不能重复除以176。这里u是每腿3维**完整策略实际投影后目标角**，不是candidate内部的feedforward或Kδx；该代价是相对共同参考，不是相邻`u[k]-u[k-1]`的精确控制速率代价。报告真实相邻变化以检验效果，不能偷换定义。

对`F_bar^+=max(F_bar,0)`，若总力大于0，`w[k,j]=F_bar^+[k,j]/sum_j F_bar^+[k,j]`；全足无载时四个w均0。该分支只定义代价，不增加物理接触阈值。

| 分支 | 新代价 | 用途 |
|---|---|---|
| r34原版 | `J_ref=0` | 当前方法的共同上下文基准 |
| constant | 各腿同`R_bar[k]` | 隔离新增参考代价本身的作用 |
| load | `R[k,j]=R_max-w[k,j]*(R_max-R_min)` | 检验承载调度的额外作用 |

constant必须匹配load在同节点的矩阵trace/平均强度：`R_bar[k]=(sum_j R[k,j])/4`，不能拿更强的统一惩罚误当承载信息的收益。若`R_min=0`、`R_max=R0 I3`，有载时constant为`.75 R0 I3`，全无载为`R0 I3`。

首次有限检验固定`R_min=0`、`R_max=R0 I3`；标度取当前moving Effort的未饱和目标角曲率`R0=.08*(.02*60)^2=.1152 cost/rad²`。这是来自现有成本/PD单位的**工程起点，不是论文给出的最佳超参数**；既有执行器裁切会改变局部实际曲率，也不因此宣称此标度最优。禁止在负结果后自动扫描倍率、改horizon、删高度项或叠加别的机制。Stand的`.02rad`强正则不是moving可直接照搬的R。

### 2.3 实现必须贯通的地方

新项必须真实进入完整native成本、残差/导数、Riccati K/H、约束投影相关的局部模型、候选archive、nominal/BestRollout、Iteration接纳与记录。只向Quu加`mu D`属于改变搜索度量，不能代替新目标；只给alpha候选打新分、incumbent/nominal仍用旧J同样不成立。

DP/Riccati优化变量仍为实际held qtarget `u=data.ctrl`，线性化点沿用best native accepted_actions。参考与R冻结时，该项的阶段导数是`c_u=R*(u-u_bar)`、`c_uu=R`、`c_x=c_xu=c_xx=0`，由现有成本引擎统一归约一次；可用`sqrt(R)*(data.ctrl-u_bar)`残差实现。真实候选先把内部中心v经原Clamp/H-QP得到u，再以u评分。沿用r30状态相关PD边界及其活动面/自由面耦合，不在独立held-u成本FD上再乘一次投影Jacobian；也不能改用未投影v作FD变量却套u的导数。若以v作为优化变量，就必须对成本和动力学共同链式求投影导数，属于本轮不采用的额外架构改动。对新项做必要的解析/有限差分数值核对，完整策略绑定与r34关闭新项的等价性核对即可；不添加一串无关回归、mock或独立smoke任务。

冻结实际spec/patch/build/prepare/runtime/recording后正常构建，写入实际二进制和源hash。记录原J0每项、新J_ref、合计J、参考身份和每腿R；跨不同分支比较必须用共同原J0和物理结果，不能直接比较不同定义的合计J来宣称改善。

## 3. 四个现存问题：12次有限求解，36次闭环诊断

使用本轮实际W45原件，不再复用W44那四个旧问题代替当前首异常。

| 流/用途 | call / output plan | origin |
|---|---|---:|
| A，前20ms较平稳参照 |148 /136|2.54s|
| A，有效正alpha且实际命令剧烈变化 |153 /140|2.64s|
| B，最早nominal支撑变化附近 |155 /141|2.68s|
| B，native本身有剧烈目标变化 |158 /144|2.74s|

A148仅前20ms相对温和，其整个350ms native轨迹的最大目标变化仍约1.50591rad，不能把它当已证明稳定的完整nominal。每问题3个分支各执行一次现有有限求解预算，共12次。保存输出策略不等于保存了历史调用全部输入、warmstart和私有状态；采用共同重建上下文，明确导入的是哪个原输出/候选，缺失字段列出。不能称精确历史重放。共同参考的来源按2.1预先声明，在同问题各分支相同，不借知道未来失败来选有利参考。

每个分支所得完整策略在同一模型/目标下作3次350ms闭环诊断：原初态、同量级正偏差、反向偏差，共4×3×3=36次。固定这一次求解产生的完整reference/K/H/scale和共同u_bar/F_bar/R，350ms内只按2ms周期反馈/投影/保持，**不再每20ms重新优化或创建epoch**。统一使用当前native predictor：同一模型、implicitfast、10×.2ms、原coldstart和冻结材料proxy；不是新variant的原plant真值。偏差优先取该问题形成前已经记录的最近实际状态减对应native预测的切空间误差；记录来源时刻/策略/量纲，正确进行MuJoCo四元数积分与切空间转换。若只能恢复事后偏差，可作为明确标注的离线敏感性诊断，不能把它用于线上决策或称因果可部署输入。

统计实际新增native rollouts、FD子调用、缓存、编译和墙钟；12次“求解”不等于12次rollout。没有新PPO样本，也不把36个诊断当36个原物理任务回合。

必须同时检查：

- 原硬条件、trunk violation和原J0分项；原目标高度恢复、前移和末端状态有没有被牺牲。
- native rollout实际施加的投影后目标变化、该rollout的PD力矩变化/反向、饱和占比、H投影/KKT、姿态、垂向运动、接触时序；全部指标保留单位。已记录W45 actual只与原部署策略对应，新variant的原plant真值由后续E30给出。
- 完整闭环对记录量级偏差的敏感性。raw Kδ变化只用于解释，不能单独做成功阈值。
- current incumbent是否本来就不可靠；新增成本是否只把机器人冻结在旧策略附近或变为驻留。

### 分支选择与有限负结果

在看原任务E30之前写出选择结论。load若相对trace匹配constant没有额外证据，就不能声称承载调度起效；constant若已经提供更好的实际命令/恢复权衡，可如实选constant。选择只保留一个明确分支进入下一步，两流使用同一分支。

有益变化应体现在实际命令与恢复能力，且不引入新的原硬条件违规；同一原J0、进度、姿态及扰动结果并列，保留所有权衡，不用不同J总值或单个更小反馈数掩盖退化。有限比较的改善仍不是长任务稳定性证明，必须由原任务检验。

若两个新分支均只驻留、锁死坏参考或没有可辨识的有用改变，记录`mechanism_not_supported`并完成这轮有限负结果；不要对已知无变化/更差配置反复开E30，也不要现场扩网格补到成功。此时下一研究问题应转向**完整可执行moving nominal的生成/状态—输入热启动与恢复能力**，单独立项评估FDDP/可行参考；不在本冻结轮次同时改架构。用户会得到具体失败证据与下一决策点，而不是“GPU任务仍活着”的占用报告。

## 4. 双CPU：选定版本直接进入原Go2/TM02任务

有限比较支持一个新分支且实现核对完成后，推送源码/spec/有限证据与真实phase声明，再启动A/B。不另加一串站立、刚地、旧QP回放准入；原E30就是本轮控制变化的实际实验。

| 流 | 固定策略 | 完整学习恢复点（保留） |
|---|---|---|
| A | `outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w44-a/PANN_PPO_S5-seed1/checkpoint-014737.pt` | 同目录`resume-state-ep00428.pt` |
| B | `outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w44-b/PANN_PPO_S5-seed2/checkpoint-014737.pt` | 同目录`resume-state-ep00417.pt` |

两流保持各自Actor mean、归一化、seed、case与checkpoint身份；PANN因果观察和cap链继续运行，PPO不更新、PANN网络权重固定，observer/material memory继续按实际观测因果更新。计数保持14737/57、tail0、excluded145；共有祖先不重复计数，不能称从零独立训练种子。W46不自动执行+512/+2。

使用现有`run_s1_go2_r12.py`单次evaluation入口包装新phase，明确传`--attempt --evaluation-start-unix --checkpoint --level --seed --workers --build --contract --condition nominal`，不能沿用会要求本phase新增训练的`--job-dir`分支。W45调度脚本的硬编码phase/build需新建或明确扩展，不能覆盖W45目录/结果后称续接。合同的`reuse.active_controller_build/controller_artifact/selected_mechanism`与CLI、PolicyStream和实际二进制一致，`--build`不自动修合同。实际合同以JSON布尔false关闭`execution.background_job.wall_deadline_enabled`，不传`--deadline-unix`。

当前代码有只列到S5-r34的build白名单；实施r35时必须同步prepare、runtime、policy、records及相关分支，特别是`MJPC_R33_MECHANISM=N`、r34 strict与R29/R27记录。r33 C++默认机制是S，漏传环境可能静默回S并缺记录。核真实readback/首调用身份与冻结spec；这是必要的启动正确性检查，不另开物理准入批次。

每流各自E30通过才同case独立E60→E300→E600：

- **E30**：满30s，原远端到达/进入转向，原x[.8,2.2]、|body y|≤.6合法去程，穿越期间至少一个完整足迹位于原材料带内，且该足实际Fz>0。E30不要求完成返程。
- **E60**：原独立60s评价，至少1次合法去程、返程和往返，回家原连续稳定2s，完整观察。
- **E300/E600**：至少5/10次合法去返/往返，于270/540s内完成，保留原远端/回家稳定、完整300/600s观察、`endpoint_position_violation_samples==0`和最终稳定≥2s等现有条件。不能把长测专属字段加成E60新规则。

初始站立2s、原质量15.206408kg、力矩[23.7,23.7,45.43]Nm/腿、PD60/5、`max(abs(principal_strain))≤.02`、非足接触、base z<.12或|roll/pitch|>1.2失败、路线与其他原判据保持。侧绕不算合法贯穿；满时长/站住/较低成本不能替代任务资格。每次失败结束该attempt，后续blocked不是实验失败次数再加三次。不要更换seed反复到成功。

两流彼此独立推进；较快失败的流交付原件后释放资源，不启动另一条同配置重复跑来填满CPU。正在运行的另一流可在不改变其冻结worker/case条件的前提下继续；不热改控制器。

## 5. 第三GPU任务：同一完整策略的执行一致性

旧G5最终包已找到并核验，见[最终审查入口](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w41-gpu/continuation-final-audit.json)。不重新跑G5，也不再安排上服务器寻找这份已发布文件。

停止W45的reference-tracking proxy top4方案。它未执行K/H/原H投影，两个问题漏掉native最优；增加槽数或继续重复不能修正评估对象不同的问题。

第三任务独立使用第3节四个W45问题。每问题取原完整baseline与原选中策略，加原初态/±同尺度记录误差，共最多24个scenario，CPU主任务不等它完成。绑定已保存/共同重建的完整身份，缺失历史pre-call incumbent不补造，重复身份可明示并缓存。

1. **同状态的动作代数**：在已保存state上核对wxyz四元数切空间、时标插值、K/scale、原H度量QP、活动面/KKT、序列化和实际held目标。可在这一独立代数层用FP64实现；不推进physics，先识别表示/实现差异。
2. **同策略最初20ms**：保持相同native predictor模型/积分器/冷启动身份，逐个2ms hold内推进10×.2ms。分别记录actual plant Euler、native predictor implicitfast与MJWarp的差异，不把三者混成GPU误差。
3. **最多一次完整350ms批次**：只有完整执行语义已实现且短窗差异可解释、不改变原约束判断，才扩同批24scenario至350ms，最多42000 world-step。完整原native目标/风险若未实现，明确没有等价排序能力；仍由CPU完整native裁决。

42000只计完整350ms批次；独立20ms阶段至多另有2400 world-step。若复用完整运行态续接，则20ms是350ms前缀；若从头重算则另计。预热、CPU native确认与重复身份缓存分别记录，24scenario不是24个原任务回合。

MJWarp dynamics采用FP32，native默认FP64；FP64动作层转入FP32 dynamics时记录cast位置，不能承诺改一个dtype得到等价FP64引擎。GPU浮点/原子操作差异可能被接触放大，应记录第一处分歧与量纲。若现有接口下语义或约束判断仍不一致，交付有限诊断并自然结束，不扩大为重写仿真引擎、不放宽原truth、不重复proxy排序。

编译、准备、上传、设备推进、回传、CPU确认、共卡竞争与总耗时分别记。若以后做筛选，最终选择必须包括全部已确认baseline；不能确认了baseline却只在top4中选赢家。本支线的产出是同策略执行证据，不预先宣称在线MPC加速。

## 6. 25CPU / 96GB：实际并行与隔离

历史容量25vCPU/120GB RAM/96GB GPU，开场核当前cgroup/affinity、实际内存/GPU进程、磁盘、tmux/PID、未提交修改及任务目录。默认D57：`ssh autodl-d57`；CPU树`/root/autodl-tmp/MCP-RL-w42`，GPU树`/root/autodl-tmp/MCP-RL`；环境`work/envs/d57`与`work/envs/w41-mjwarp`。

| 工作 | 资源 |
|---|---|
| CPU A |10 workers；Torch/BLAS/OMP各1；整个进程树明确affinity|
| CPU B |10 workers；Torch/BLAS/OMP各1；与A隔离|
| GPU一致性支线 |GPU + 总host额度2CPU，含native确认|
| 主控/观察/发布/系统 |约3CPU余量，按现场实际资源调整|

前述有限native比较使用A/B两份各10CPU的亲和额度分问题并行，同时最多两个求解进程；GPU的2CPU host预算另计。有限比较结束前不叠开两条全负载E30；构建`-j6`在双流重负载前完成。继承现有`autodl_d57_env.sh`：`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`XLA_PYTHON_CLIENT_PREALLOCATE=false`、`JAX_ENABLE_X64=true`。所有可写upstream/build隔离，prepare不能经共享可写软链影响另一个任务；环境、静态资产与已完成父输出可只读复用。

两“CPU流”的PANN FP64 JAX与Actor也可能共用GPU。W45原planning占总wall约77–79%，PANN observer+query约0.004–0.804%；当前主瓶颈不是PANN小查询。显存占满不是成果标准；第三有限任务完成后继续双任务是合理安排。不为利用率增加第三个10worker CPU/PPO流或让已失败配置循环运行。

## 7. 后台生命周期与发布

**所有新后台deadline=null，无固定墙钟完成时刻，无到点kill。** 按有限工作完成、原物理评价自然结束、明确技术失败或用户停止结束。原30/60/300/600物理秒、有限求解预算和单次网络timeout保留；“无固定后台截止”不等于无限重试，也不取消原任务的物理完成要求。

每个phase/job/clock只声明一次。存活进程可精确PID暂停恢复；尚未完成且因暂停/技术原因退出的attempt保留interrupted，已自然结束的成功、物理失败或任务未满足保持原结果。没有完整plant/材料/observer/MPC/任务状态时，仅Actor不能中途恢复物理过程，若需要续跑则新独立attempt从原初态开始，不覆盖或拼接连续资格。不得宽前缀kill其他有效任务。

先推实际实现/spec/有限证据/真实phase，再启动并确认两条物理时间、执行记录持续增长；tmux活着不足以证明推进。保存共同参考/权重及完整策略身份、J0/J_ref分项、alpha/scale、高精度接纳、actual/预测首20ms、投影后目标/真实力矩变化、关键5000Hz子步、材料加载和合法路线。关键大数组按事件保存一次并引用身份，不复制冗余；历史证据保留。

紧凑Git先发布，请求原件增量Release后发布并核size/digest；Git/index短写串行合并最新main，不强推、不覆盖并发数据。尚未生成的文件标明never-generated，不制造空壳。计划、已构建、已运行、任务资格与数据交付分别报告。

**GitHub核验后不在用户电脑另下载、打包、镜像或归档。服务器原件、checkpoint/resume、视频和上传副本不自动删除。** 单独数据迁移/清理不属此任务。D57上传只用已授权服务器学术代理；禁止用户电脑VPN SSH转发，不自动切电脑/Clash/直连；凭据不进日志或Git。

## 8. 稳定之后的长期路线

控制版本若通过原资格链，冻结完整控制基线再单独声明学习phase。未来各+512/+2到15249/59只是后续候选预算，W46不自动启动。随后需要真实跨足/返程曝光、候选内独立传播材料历史、检查cap动作权限是否足够、同控制器/信息/预算下RAW/PANN/冻结或打乱历史的因果对照，再推进条件泛化与Go2/NX。

目前MPC候选仍主要使用瞬时接触代理，没有完整候选材料历史；稳定行走修复不能自动成为材料记忆创新。此次首要目标是把原Go2/TM02任务跑稳，使后续记忆研究有有效数据与可信对照。

## 9. 论文依据与迁移边界

- **Hybrid iLQR Model Predictive Control for Contact Implicit Stabilization on Legged Robots**，TRO2023，§IV-A/B、VI-B：[论文](https://arxiv.org/pdf/2207.04591)。time-stepping承载力调度参考输入代价，不是照搬event-driven reference extension；其完整walking reference前提必须保留。r35的参考/标度属于本项目待检验选择。
- **Contact-Implicit Model Predictive Control: Controlling Diverse Quadruped Motions Without Pre-Planned Contact Modes or Trajectories**，§6：[论文](https://arxiv.org/html/2312.08961v2#S6)。轻微接触差异能破坏单射击初始化，FDDP继承状态和输入并缩减动态缺口；这是若可信nominal本身不足时的后续方向，不作为当前已验证因果或本轮顺带改写。
- [官方MJWarp文档](https://mujoco.readthedocs.io/en/4.0.0/mjwarp/index.html)：吞吐/延迟、FP32与native差异、四元数及容量语义。W45的近似proxy负结果不能外推为GPU不能做MPC。

更完整的原论文、R8差异与限定见[本轮文献/控制基线审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review/literature-and-baseline.md)。
