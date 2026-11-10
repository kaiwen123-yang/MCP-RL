# W42 CPU 实验计划：原生接触前向预测与平滑局部导数

> 2026-11-10：本文件保留历史身份，当前执行入口为 [W43计划](S1_W43_STABILITY_PLAN.md) 和 [W43完整启动提示词](S1_W43_START_PROMPT.md)。W42及原GPU已结束；GPU只按新计划从失败阶段续接，不重复已完成目标。

状态：**W42两流训练、末次评价及请求清单交付已完成；E30均未通过。以下保留原执行协议，不重复启动。**

[W41完整审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/README.md) · [控制原件复核](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/controller-review.md) · [论文机制](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/literature-mechanisms.md) · [完整启动提示词](S1_W42_START_PROMPT.md)

## 1. 本轮要解决什么

W41 两支各新增2048条/8更新，到11665/45、tail0，130回合没有结果级技术中断；127次物理失败，合法材料去程/返程/往返均0。QP不再中断采样是进展，但当前控制没有形成持续移动和可重复接触转换。两个E30在材料前已有显著欠跟踪；B长存活含倒退，不能当任务接近完成。

新原件进一步排除了单纯的“投影吞动作/风险墙”：A590停滞计划整段无活跃动作边界、Impact/Domain/Landing成本严格0，Position约占53.35%。9个计划的J重算与记录相差不超过1.43e-14，r30自由行Riccati关系缩放残差约2.74e-15；没有证据要求重写已经一致的QP公式。

更直接的问题发生在接触预测。A635、B712、B713的首个0.2ms来自相同记录起点，目标最大差分别为5.71e-8、5.95e-8、5.15e-6rad；当前平滑规划模型预测的刚地足端反力约0.007–0.013N，而实际记录约114–247N，首个2ms后关节速度已有约0.39–0.44rad/s偏差。这是同状态、近乎同目标的已交付记录比较，尚未新跑严格相同输入回放。它发生在材料加载之前，随后时变反馈放大偏差。详见控制报告及独立[接触交叉核验](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/contact-crosscheck.md)。它建立了需要处理的forward失配；solimp、coldstart等配置的各自贡献仍待分离，也没有证明全部跌倒的唯一原因。

**本轮唯一方法变更：让nominal/候选/成本和风险评分使用原生接触前向预测；在该原生名义轨迹上使用零阶锚定的平滑代理A/B。** 不再把这次工作定义为又一个QP数值补丁，不并行叠加奖励、步态、动作维度、PANN或材料物理改动。

## 2. 为什么不直接重复 r17

旧r17恢复过geom/pair的原生solimp d0，dev19在15.3628s因不合法接触失败，最大x约1.03166m。该负结果说明“恢复d0就必然稳定”不成立；它不证明现在量化到的前向接触失配可以忽略。

W42明确区分两个模型角色：

| 角色 | r30现状 | W42/S5-r31拟实现 |
|---|---|---|
| 前向名义/候选滚动 | MakeDifferentiable后的接触参数 | 恢复原生geom/pair接触阻抗的前向模型 |
| 求导模型 | 同一平滑模型 | 独立平滑模型，仅提供局部A/B |
| 求导的线性化点 | 平滑轨迹 | 每个原生名义子步的实际状态 |
| 候选J/躯干v/路径峰 | 平滑轨迹上计算 | 原生前向轨迹上计算 |
| backward k/K/value | r30完整递推 | 保留该递推，输入明确标为代理A/B |
| 实际plant/PD/终止 | 原Go2/TM02 | 原样保留 |

这是对前向保真与接触搜索方向的分工实验。不是复现DiffContact的解析摩擦锥梯度，也不是将原生MuJoCo接触等同于完整TM02真实材料。旧r17失败、代理导数偏差及现有TV反馈风险均须保留在结论中。

HOUND CI-DDP（IJRR2025）、DiffMJX（2026版）和DiffContact（2026-10初稿、11-02修订）分别提供了前向与反向分工的相关依据。它们并不保证我们沿用的d0=0代理或由此得到的K能够稳定Go2；HOUND实际反馈层也不同。[文献与边界](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/literature-mechanisms.md)

## 3. 实现契约

### 3.1 两个冻结模型与同一个原生名义轨迹

1. 新建独立 `S5-r31`，based_on=`S5-r30`，声明任务 `W42_CONTACT_FORWARD_JOINT`。保留r30、qp2及运行中GPU的原产物。
2. 在MakeDifferentiable修改之前保存原geom/pair solimp。前向模型恢复这些原接触参数；joint-limit solimp保留现行平滑设置。其余模型、原2ms保持/十个0.2ms子步、当前规划积分器、contact filter、因果userdata、材料proxy和力矩限幅保持，单独记录仍与plant不同的integrator/warmstart/proxy上下文。
3. 为求导建立独立的平滑模型与每worker独立data。**不得在共享mjModel上让不同线程临时切换solimp。** 两模型的几何、控制和因果状态身份明确；时间、control、userdata、callback与residual快照必须来自同一规划快照。
4. nominal、旧warm start的重评估、所有反馈候选/动作线搜索、accepted与反馈reference的前向状态、成本和风险全部来自新的前向模型。不能只修改最终候选评分，也不能只有第一个子步原生、后九步仍沿平滑轨迹。

### 3.2 零阶锚定、平滑对平滑差分

设一个子步的原生前向为`f_n`、平滑模型为`f_s`，原生名义点为`(xbar_j, ubar)`，原生后继为`xbar_(j+1)=f_n(xbar_j,ubar)`。局部模型以原生后继为零点：

```text
delta_x_(j+1) = A_s,j * delta_x_j + B_s,j * delta_u
A_s,j, B_s,j = smooth-model derivatives evaluated at (xbar_j, ubar)
```

可用`f_n(bar) + f_s(x,u) - f_s(bar)`理解其零阶锚定；实际实现应使用MuJoCo的位置/姿态切空间与原状态差工具，不能直接相减四元数再相加。

单侧有限差分必须是 **`f_s(perturbed)-f_s(base)`**。不能用`f_s(perturbed)-f_n(base)`，否则原生与平滑模型的零阶差会被除以1e-6，制造假A/B。保持原FP64、1e-6单侧扰动设置。本次改变的是模型分工，不以调epsilon掩盖问题。

十个子步分别在各自原生名义状态上算代理A/B，按原固定目标链复合：`P_(j+1)=A_s,j P_j`、`Q_(j+1)=A_s,j Q_j+B_s,j`。C/D和每个风险分量的活跃峰必须沿原生轨迹重新计算，并使用对应原生峰状态与代理P/Q；不能继续用旧平滑轨迹的峰、接触或残差，也不能复用soft FD留下的传感器值。

### 3.3 完整优化与部署身份

保留qp2原容差、瞬时目标/PD移动边界及r30完整k/K/Vx/Vxx/dV。代理导数下的K、H及预测下降属于这个局部模型，不能标为真实plant精确Jacobian或稳定性证明。候选接受继续看原生前向的实际`(v,J)`，保留真实cost decrease、预测下降、surprise、正则化变化与无改进原因。

记录accepted与用于部署的reference/nominal/K/H、alpha/feedback scale及其模型身份。不要通过部署后额外clip K、任意缩反馈、放宽QP容差或替换失败动作掩盖负结果。若原生候选持续不下降，就报告代理搜索方向在相应接触模式无效；该事实应指导下一版本，而非无限自动重试。

### 3.4 只做与新变更直接相关的检查

使用本次已读A590、A635、B712、B713真实输入，核对原生与平滑模型角色、同一x/u的首0.2ms/完整2ms接触力和状态、零阶锚定、soft-soft差分、原生峰选择及候选J。借实际搜索方向测原生held-map方向误差与实际cost变化；不要求代理A/B冒充原生真导数，也不能仅凭代理FD自洽宣布控制正确。

这些是新算法接线与已量化缺陷的针对性验证。正常构建后直接进入完整PANN→PPO→MPC联合任务；不新增纯MPC/刚地/E30准入、smoke/mock、恢复演习、全库hash认证、参数网格或重复旧12例QP回放。

## 4. 两个 CPU 流与正在运行的 GPU

### 4.1 先隔离运行目录

默认服务器 `autodl-d57`，原根目录 `/root/autodl-tmp/MCP-RL`，环境 `work/envs/d57`。**W41 GPU的bridge会在后续阶段重新读取根r12配置并延迟导入模块。** 因此仅承诺“不杀GPU进程”不够；新CPU代码/config不能覆盖它还要读取的旧文件。

在现场确认现有PID/tmux/job与未提交研究修改后，创建独立W42 worktree/checkout（建议 `/root/autodl-tmp/MCP-RL-w42`）。复用现有虚拟环境、已核对的依赖和静态资产；仅对明确的只读依赖/父点建立引用，新build、job、clock、log和输出有自己的写入位置。不要将整个`work`或活动`outputs`共用成可写目录，不复制大原件做电脑归档。

`prepare_s1_go2_r12_support.py`会重写`work/terrain-memory/upstreams/mujoco_mpc-go2`源码、CMake及其build依赖目录。该**可写upstream checkout和build也必须独立**，不能为了复用依赖把整目录软链到运行中GPU的工作树。只读依赖与现有venv可引用，不能混淆依赖复用与共享可变构建目录。

新目录可把两个已完成W41父训练目录按原相对路径只读引用，以满足现有路径记录和parent检查；保留真实来源。原GPU根目录继续使用原代码、r30产物及配置。Git提交/发布整合并发远端变更，不把新代码checkout进活动GPU目录。

### 4.2 明确父点与工作量

| 新流 | 自身W41父点 | 初始状态 | 预声明新增 | 目标 |
|---|---|---|---|---|
| w42-a | `training/w41-a/PANN_PPO_S5-seed1/resume-state-ep00387.pt` | 11665/45，tail0 | 2048条/8更新 | 13713/53 |
| w42-b | `training/w41-b/PANN_PPO_S5-seed2/resume-state-ep00375.pt` | 11665/45，tail0 | 2048条/8更新 | 13713/53 |

表中训练路径前缀为 `outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/`。实际现场若有更晚有效同流父点，以真实状态为准并重算目标，不回退。完整继承actor/critic/optimizer/normalization/RNG、fork_identity、phase_lineage和excluded145；两个策略不互换，不重新fork/reseed。新phase各自有目录、时钟、parent_map和产物边界。

两流采用同一个r31机制，分别复核自身续训结果。它们是两个历史分支的工程重复，不是同父同随机数的严格因果对照，也不是两个从零训练seed。本轮不额外重跑已完成的r30训练；将来做严格机制消融时另声明同父与相同评价条件。

### 4.3 资源安排

第三个任务就是用户已经在跑的GPU工作，不再新开第三条CPU解算流。先读取真实CPU负载、可用内存、GPU进程/显存峰值与本任务占用；不能把96GB全部算空闲。

新CPU流建议先按 **10+10 MPC workers** 留出主控、GPU host及上传空间；若现场GPU host负载低且可用CPU支持，可在启动前声明11+11或12+12。Torch/BLAS各1，原生C++构建最多`-j6`并安排在两流重负载之前，不修改运行中的GPU/XLA配置。启动器现仅允许12/14，需在W42目录为新的已声明workers补齐合法范围，普通resume仍保留原冻结workers。

以两流合计独特样本/h、真实规划耗时和资源峰值评价效率。W41约167.4独特样本/h，规划调用平均约1.382s/逻辑20ms；这不是实时能力。新增模型分工也可能增加开销，要记录而不以显存占用充当进展。当前不热接未完成GPU后端。

## 5. 评价与成功/负结果口径

每流完成新目标后，用自身声明目标的末次新update做固定策略E30（当前父点下为update53）；按原同case前级要求进入E60，之后300/600仍按原序列。E60/E300/E600保留 `bidirectional_material_roundtrips_v1`，三个阶段分别要求合法去程、合法返程及完整往返三项均达到1/5/10次，并满足所有原条件。不能只按raw圈数、存活时长或奖励评分。

本轮同时回答三个不同问题：

1. **预测缺陷有没有减少：**同起点/同保持动作接触力与2ms状态偏差，首个20ms计划/执行偏离；区别原生前向误差与有意代理导数误差。
2. **可执行移动有没有恢复：**正速度指令下净推进、四足驻留、摆动/支撑转换、首次nonprogress、反馈修正、触地速度和材料加载到失稳的时间。无故障但原地站住不算改善主目标。
3. **任务能力有没有形成：**合法出带/远端/返程/往返、原终止原因及固定策略评价。只有真实结果可以提升资格。

若forward偏差降低但仍驻留，再由原生候选的分项成本/实际相位判断参考或结构化初值是否必要；若接触模式搜索失败，再考虑更合适的接触代理梯度。若底层稳定后需要学习纠偏，再另立一类显式残差动作权限，保持零残差恢复旧行为；这些不与W42第一机制叠加，也不自动更改奖励。

## 6. 记录与交付

继续保留全部逐回合结果、4096条新transition/value、16更新、末次固定评价及已有原始日志/数组要求。新增记录两模型身份、原生/代理导数角色、实际子步线性化点、成本分解、原生方向误差和投影/反馈身份。保留现有stdout/stderr及MuJoCo warning计数；已完成W41若服务器仍有日志可补交，缺失就注明，不补跑旧物理。

新phase生成时按真实父点生成`controller_version_boundary`和`checkpoint_selection`，不继续携带W40/33/8593陈旧文字。历史W41结果保持原样；它们的实际checkpoint选择正确，不把元数据陈旧写成旧策略误用。

先推实现/spec、父点/计划与启动身份，再运行；结束先推紧凑结果，所请求原件继续后台Release并在Git记录完整状态。复用现有无损分包/上传能力，不重复重新打包已发布W41 24ZIP。队列已提交与远端已发布分开报告，合并并发提交，不强推。

**所有新/续接后台不设固定墙钟完成时间。** 以2048/8样本更新目标、声明评价完成、明确技术失败或用户停止结束；deadline=null。保留数值迭代guard、单次网络请求timeout及物理30/60/300/600s；不增加泛泛失败次数上限，不吞错或无限自动重启。

**GitHub推送核验后，不在用户电脑另做归档；不自动删除服务器checkpoint/resume/原件或上传副本。** 启动交接应给真实job/PID、当前transition增长、实际artifact、版本与Git提交，不将计划写成已完成实验。

## 本轮执行位置与冻结配置

实际工作树 /root/autodl-tmp/MCP-RL-w42，基线4e1eb97。活动GPU根 /root/autodl-tmp/MCP-RL 保持原HEAD和文件；只读引用venv、静态资产、r30父产物与已完成W41父训练目录。可写upstream及全部build依赖为新树独立副本。新CPU固定10+10 workers、Torch/BLAS1、构建-j6；使用现场可用academic独立联网。父点A ep387/B ep375，均11665/45、tail0、excluded145；目标各13713/53。新时钟为w42-training-clock.json，deadline=null。declare_s1_go2_w42.py由真实父点生成版本边界，启动器进一步冻结每流checkpoint选择；不修改历史W41记录。

S5-r31已在独立工作树完成实现、正常构建和四个指定保存输入核验。首子步Fz误差约浮点精度；2ms关节速度最大误差A590/A635/B712/B713由0.34467/0.43351/0.38575/0.43711降为0.000756/0.009661/0.008869/0.006646 rad/s。原生held-map方向相对L2误差仍为3.044/2.165/0.575/0.986，不能称精确梯度。两个A输入新C++快照的soft A/B重算一致，十子步P/Q误差不超过1.78e-15，原生峰位置/值一致，J重算误差不超过1.39e-17。B712/B713在未恢复历史warm-start/私有phase的保存状态规划中backward_failed，保留原记录、不重试；这不是新任务回合或正式评价。

原件与紧凑证据：results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w42-contact-forward/result.json；服务器原件在outputs同run的w42-contact-forward目录。四个输入不是泛化验证，原plant积分器/solver历史及材料proxy近似保持；下一项是已授权真实联合训练。

实际启动：w42-a job-20261108T154719Z-w42-a / PID654367；w42-b job-20261108T154727Z-w42-b / PID655195。23:50:26各11668/45、tail3，目标各13713/53；源码4dcc1bb先推送后运行，网络academic、workers10、deadline=null。本次交接不等待训练完成；后续保持现有tmux，如进程停止再用同job最新有效断点普通resume，不能再次new-phase。

W42两条CPU已完成各自2048条/8更新，分别到13713/53、tail0、excluded145。两条末次update53 E30均运行满30s，以material_traversal_not_completed未通过；E60 blocked_by_predecessor。36个训练回合为34个60s采集边界和2个样本目标截断，结果级技术错误0；材料加载与合法去程/返程/往返均0，最大base x约0.025mm。

2026-11-10交付：W42-A/B当前请求清单各829项已在远端按名称、uploaded状态和大小核对，无遗漏。A两项上传中断的starter资产保留，完整内容用新名称补传；清单已指向新资产。GPU旧29,337,276字节包和新增95,663,446字节包均已发布，新增包SHA256与远端digest一致。原件、检查点及上传副本保留D57；不在电脑归档，不自动删除。
