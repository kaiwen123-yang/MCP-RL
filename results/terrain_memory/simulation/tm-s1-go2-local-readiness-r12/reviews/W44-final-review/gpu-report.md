# W44 GPU 证据审查与 W45 并行支线

审查日期：2026-11-12。结果基点为 `69d5a6ff2aa9803fddc8ac3a78a906008ea526c6`。本次重新读取和复算已生成原件，没有运行新的 GPU 或机器人仿真。

本报告区分四类证据：**G1/G2 是完整保存数组的复算；G5 最终状态是仓库现场完成观察；同步、拷贝和重复 world 数是源码计数；瓶颈的时间占比与下一条 GPU 支线的收益尚未测量。** 后文四问题批量 proposal 任务是待实施计划，不能写成已完成加速。

以下 `R12` 表示仓库相对路径 `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12`；代码路径也均相对仓库根。

## 1. 本轮研究选择

W45 主任务是稳定性隔离：**A/B 各自固定 W44 的 update57 Actor，在原 Go2/TM02 任务评价控制修正，本轮不追加 PPO 样本或更新。** 第三条任务为四个 W44 已保存问题上的 GPU 完整 horizon 候选提案筛选，以及 CPU native 完整策略验证。它是有限离线研究，不新增 PPO 谱系、不复跑 G5，也不成为 CPU 开跑前的准入链。统一执行范围见 [W45稳定性计划](../../../../../../docs/S1_W45_STABILITY_PLAN.md)。

研究问题是：在本项目原状态、成本、限制和完整 CPU 部署策略下，GPU 的廉价近似筛选能否找到有价值的候选，并减少得到 native 确认候选的总计算时间？如果排序或总耗时不支持线上使用，就完成证据与结论，保持为独立支线。不能将它预先称为 CPU 等价迁移、控制成功或已获加速。

这一选择直接针对现有 GPU 的分片与往返结构，也避免继续用不等价、对接触敏感的 FP32 FD 替换当前 N 方向。真正 GPU 原生全求解器依然可以是长期研究方向，但本轮不整体迁移到新框架。

## 2. 已经有的 GPU 证据

### 2.1 G1/G2 原件复算

独立读取两个已在工作区的原件包，重新执行仓库纯数组审查程序：

- `w41-new-evidence-01.zip`：29,337,276 bytes，SHA256 `af68f148c5f1c010532361eccbf03b0e9b05123df15c7ed9243483fb04668ec0`。
- `w41-g2-20261108T034922Z-evidence-01.zip`：95,663,446 bytes，SHA256 `d4a4e0aee63565d729781483a84d6273f0ed5d20da1dccb51fc3bc46cb9abd40`。
- 语义审查读取 246 个有关成员，共 122,965,986 解压字节；未运行物理模拟。计算结果见 [gpu-recomputed.json](gpu-recomputed.json)，所用程序为 [W42原件审查程序](../W42-final-review/audit-gpu-evidence.py)。JSON 中 `baseline_commit` 保留被审历史证据基点，不代表本次报告的最新仓库 HEAD。

G1 的 32 个局部事件只有 **5 个满足原阈值、27 个不满足**。最大 position error 为 `4.4400358545360064e-7`，velocity error 为 `0.0022200179268274667`，A/B 导数元素误差最大为 `2439.6287350643142`。不同状态/导数分量单位不同，不能将这些值合并为一个有统一物理单位的误差。保存 CPU/GPU 数组可以准确重构所报误差；小位移误差没有建立导数等价。

G2 的 176 个节点原件均存在且 finite：每节点 49 个 FD 输入、10 个子步，合计 **8,624 个逻辑输入、86,240 个 world-step**。状态数组 `[10,49,37]`，导数数组 `[10,36,48]`。176 是状态探针数；原控制 horizon 有 175 个 action，末节点复用 action174。

| G2 字段 | 复算数值 | 能解释的范围 |
|---|---:|---|
| 176 次 `resident.request` wall 合计 | 152,237.1021484565 s / 42.28808393 h | 输入转换、同步计算；后续完整取回/压缩写盘在该计时之外 |
| 单节点中位数 | 899.395182568 s | 当前节点分片实现的实际耗时 |
| 单节点 p95 | 1,182.541324886 s | 节点耗时分布，不是尾延迟 SLA |
| 首节点 | 637.620929420 s | 不能把所有后续耗时归因于首次编译 |
| 剩余175节点平均 | 866.282749823 s | 尚无 per-kernel profile |
| 固定 contact / constraint 槽 | 1,007 / 3,045 | 实际负接触距离只有3–4项，非零约束力21–24项 |

`scripts/s1_go2_w41_gpu.py:20–25,65–79,188–207` 显示：三个 jit 函数在 resident 初始化时建立；G2 外层 `for i in range(176)` 依次请求并同步，只有节点内49路输入批量。**没有证据表明每个节点都重新 JIT**，也没有把176节点做成同时 resident batch。现有 G2 没有配对 native 同 horizon 的方向一致率、力一致率和整 MPC 加速率，不能从176/176完成推导它们。

固定 MJX3.2.3 elliptic Newton 的 `cm=diag(...)` / `J.T @ cm @ J` 与该槽数形成明确的性能嫌疑：49路 `3045×3045` FP64 矩阵的概念存储为3,634,633,800 bytes，约 **3.385 GiB**；保存 JAX peak 为3,723,891,200 bytes，约3.468 GiB。源码/形状吻合是结构证据，不是设备 profile 对42小时的百分比分解。`solver_niter=0` 在该版本是未回写计数，不能解释成求解器零次迭代。证据：`R12/reviews/W42-final-review/gpu-review.md:48–58` 及 `gpu-audit.json`。

### 2.2 G4/G5 哪些是历史故障、哪些是完成观察

旧 G4 native/MJX 都缺 `--evaluation-start-unix`，CLI exit2，MJX requests0，没有物理评价。旧 result 的 G3=`completed_actual_outer_loop` 因而不支持“整链完成”。旧 G5 将设备 tolerance array 替换成 Python float，第一步 kernel ABI 失败，fixture0。

修复沿革应保留：`ec5982126028bd71be42648de36eaa6d71c40919` 修正 G2 scan 的 int32/int64 contact 索引；G2随后实际完成176节点。`06206f4` 修正共同评价参数、以 `.fill_(1e-8)` 保留 tolerance array 身份、完成状态聚合与明确 G4/G5 continuation。`8ba7fd0` 为随后已核实发布路由。修复不能擦掉首次错误，也不能用首次错误覆盖后续完成。

最新 G5 continuation `g5-20261110T033858Z` 的**仓库现场观察**是：模拟约30 s，wall `69717.86702171527 s` / **19.36607417 h**，`material_traversal_not_completed`，无实际材料加载、合法往返0，未重启。原完整观察在 commit `18161ff` 的 `index.w44_live_observation.GPU`；当前 `AGENTS.md:15`、`docs/TERRAIN_MEMORY_PLAN.md:17` 保留。当前 `gpu/w41-gpu/{result,mjwarp-result}.json` 仍是前次失败文件，W44 completion 替换 live object 时未保留 GPU 子对象。

本次查看的最新 HEAD 没有新的 G5 final trajectory/完整最终 result。收集既有 continuation 的最终 `mjwarp-result.json`、物理 result、resolved contract、完整日志、request count、first10 request arrays、trajectory、planner/action记录是交接工作；**不重新运行以补齐**。若服务器当前也不存在，按具体路径报告缺失。19.37h记录支持“该隔离评价很慢且未通过材料任务”，不支持精确分配其中 GPU kernel、CPU controller、PANN、编译与暂停所占比例。

GPU continuation 仍使用 **W40-B / update37 / r30 冻结合同**。它不是 W44 update57/r33、不是当前 N 控制器，更不是第三条新学习流。原 CPU Go2/TM02 plant 没有迁移。

## 3. 19小时链路中能从源码确认的开销

`scripts/s1_go2_w41_warp_backend.py:52–101,148–172` 和 `experiments/terrain_memory/s1-go2-r11-S5-r30-gpu-mjx.patch:24–44`：

1. 每个2 ms hold各自打开 Unix socket、发送153个 float64、等待返回并关闭；服务器逐个处理连接。没有把一个完整 horizon 的候选或独立节点作为一次请求。
2. Warp data固定49 worlds。**forward模式也计算49份相同轨迹，只使用world0**，每2ms为490次world-step而有效名义物理步只有10。49倍是重复工作数，不是已测49倍加速空间。
3. 稳态 FD请求初始load一次，10个子步再各load一次，共11次；每load对qpos/qvel/ctrl/userdata/time/mocap_pos/mocap_quat七个array新建 `wp.array` 并 `wp.copy`，故77次分配/拷贝。每子步显式 `wp.synchronize()`，下载qpos和qvel，共10次显式同步、20次 `.numpy()`；48路切空间差分在CPU完成，共480次 `mj_differentiatePos`。这些是代码计数，不是性能采样结果。
4. 每load的153×49个FP32约29.3KiB，payload本身不大；不能仅凭PCIe峰值带宽论证这条链很快。分配、序列启动、同步和CPU回调值得计时区分。
5. 只 capture 一个 MuJoCo子步。初次warm step/graph建构只出现一次；无法把每次请求视为重新编译。
6. iLQG递推、native残差/风险/几何、候选排序、反馈投影及物理plant都留在CPU。仅把2ms动力学切到GPU，并不等于整个MPC在GPU。
7. 旧 Warp模型主动 `geom/jnt/pair_solimp[:,0]=0`，FP32且使用局部材料proxy；当前r33 N/native状态与原soft模型存在明确身份差异。调大epsilon不能消除全部跨引擎/模型不一致。

因此，“碎片化交换+重复forward worlds+未批量horizon”是确定的结构；其wall占比、GPU kernel热点、并卡竞争和升级后的实际speedup均仍待计时。不能把“显存没满”当首要故障原因。

## 4. 第三任务的完整可执行范围

### 4.1 与CPU分析共享四个W44保存问题

四份完整 accepted/reference/K/H 原件均已取回并检查数组 shape、time 和策略身份；B161 的完整候选文件也已取得，远端 SHA 比较匹配。它们是各 call 的**输出策略**，不能冒充原 call 输入。按各 plan 起点与对应 task/gait 参数共同重建四个问题，并在同一重建上下文重算 native 基准。缺失的私有历史字段列出，不将新建上下文写成历史精确 replay。控制事件和完整策略身份详见 [控制报告](solver-report.md)。

| 问题 | 固定评价attempt | call / plan | plan origin s |
|---|---|---|---:|
| A早期 | `job-20261110T143358Z-w44-a-eval-PANN_PPO_S5-s1-E30` | 158 /151 | 2.739999999999731 |
| A首次异常附近 | 同上 | 163 /156 | 2.839999999999720 |
| B零反馈倍率反例 | `job-20261110T143400Z-w44-b-eval-PANN_PPO_S5-s2-E30` | 153 /143 | 2.639999999999742 |
| B实际反馈异常 | 同上 | 173 /161 | 3.039999999999698 |

原件路径为 `outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/<上述attempt>/accepted-reference-gain-H-plan000151.npz` 等相应 plan 文件；准确 `source_path` 及 Release 条目保存在 [A诊断清单](../../jobs/job-20261110T143358Z-w44-a/diagnostics/evidence.json) 和 [B诊断清单](../../jobs/job-20261110T143400Z-w44-b/diagnostics/evidence.json)。各文件有 accepted `[176,37]`、actions `[175,12]`、reference `[176,37]`、K `[176,12,36]`、H `[176,12,12]`、userdata96、0.35s times。每次源合同/文件路径/plan身份必须来自清单，不能按文件名猜模型。

B143 的 installed feedback scaling 为0，2.664s附近 raw Kδx 虽大但未施加，是必要反例；B161 的 chosen alpha 为0、installed feedback scaling 为1，是真正首次有效 Kδx 大幅施加的策略。A156 同样 chosen alpha0、installed scaling1。alpha 与反馈倍率必须分别保留；不能把 alpha0 解释为恢复旧完整策略。

### 4.2 一组固定候选，两个不同角色的评价器

每问题声明32候选槽：两个有来源的完整基线策略身份，以及同一个已评分基准上15组正负结构扰动。若两个基线相同则保留两个槽身份并去重计算，不能编造一个不存在的历史incumbent。已有r33 P生成器的5组腿组合×3 onset×正负，onset0/.08/.16 s、duration .16 s、关节扰动 .04/.04/-.08 rad，可作为本次固定proposal family；不要扩成幅值/worker/参数网格。源：`experiments/terrain_memory/s1-go2-r11-S5-r33.patch:227–264`。若CPU主改动要求另一组候选，先同步相同有限清单，避免各用不同候选宣称排序一致。

CPU验证策略对象须绑定同一actions/times/reference/K/H/feedback_scaling、同一目标与阶段、同一原硬限制和原生成本 `(v,J)`。生成后的候选先保留完整身份；均值/插值/再优化后是新对象，不能沿用旧分数。重算 native 全部32槽（去重数单列）作为本次离线真值，验证GPU top4与基线，缓存同对象native结果；研究期不另作在线部署。

为避免把预先算好的native真值当作零成本确认，计时顺序固定为：先GPU生成排名，再在2-host-CPU队列优先确认top4及必要基线，记实际确认完成时刻；随后完成剩余native候选以建立本次32槽真值。两段实际native工作成本、离线研究总成本分别报告。若用缓存，只有本次已计费的完全相同对象可以复用；历史预计算命中单列，不能混入“新问题端到端加速”。

GPU只承担**近似proposal排名**。如果暂未完整移植 `ActionImpl` 的反馈、H度量QP/PD约束，明确记录 `evaluation_semantics=approximate_proposal_forward`；GPU用可复核的前馈序列/冻结proxy推进，CPU再用绑定K/H/投影的完整策略裁决。不能称两边是同一个闭环策略，也不能将GPU的低约束违反率当native可行。若移植了device反馈投影，也仍需跨引擎误差实测，不能只改状态名为equivalent。

新研究模型从这四个W44问题的XML/参数构建，保留task/constraint/solver参数身份；不要默默沿用G5 `MakeDifferentiable/solimp=0` 的r30合同。FP32与碰撞实现、tolerance表示、材料proxy、private phase缺口分别记录。PANN观测/真实TM02物理状态仍由原链负责；不把完整历史材料law未经验证替换为GPU弹簧。

### 4.3 GPU执行形状

- 初始32个**独立候选world**，在一个问题内共用model；顺序处理四个问题。不要把1750个因果时间步误当独立world并行。
- 完整350ms horizon =175个2ms控制段，每段10个0.2ms物理步；每候选1750步，每问题32×1750=56,000 world-step，四问题共**224,000 world-step（推算工作量，不是wall预测）**。不要另加49倍FD世界，本支线不计算A/B导数。
- 所有候选初态与动作数组预分配并一次上传。GPU内部连续更新state，device读取每段action，在同一stream顺序推进。capture可复用10步graph并排队175段（每段device更新控制索引），或等价完整horizon图；本轮不强求巨型graph。中间无每0.2ms `.numpy()`，末尾才同步并取回完整状态/约束摘要、排名和所需全数组。
- 每问题保存32候选完整176节点状态及关键1750子步峰值，不保存每个kernel冗余副本。已有全量记录不删。1750×32×37个FP32状态约8.29MB仅是状态输出量，不是引擎显存需求；model/contact/solver/graph另算。
- capacity从已有 `nconmax=128,njmax=768` 起步并记录实际峰值/overflow，避免一开始改接触容量与算法两个因素；callback维度改为实际32world而非硬编码49。任何overflow为技术不成立，不能忽略接触或伪造native通过。官方API说明nconmax是跨world总接触分配的均值语义，njmax是每world硬容量；不能相互类比。
- 不预分配96GB，不为提高显存利用率扩到上万world。32候选是声明算法量，若当前可用内存不足以共存，则保留两CPU任务，第三任务等待可用资源或按相同清单分片；不是缩短物理时间或改评价标准。

### 4.4 真正需要的指标与结束条件

每问题保留32候选的GPU proxy分数、CPU native `(v,J)`、原约束/失败原因及完整对象身份，给出：

1. GPU top4是否含native最优或等价并列最优、native可行候选召回数；平分规则明确。
2. 约束假阴性数：GPU判断可行而native无效；按足/躯干接触、QP/投影失败等原限制分别列出，分母是本批实际比较候选，不能外推总体概率。
3. top4中native最优相对32槽native最优的lexicographic regret：先比较违反量v，同v再比较J，不用任意加权标量掩盖可行性差异；相对incumbent的改善也列出。
4. 独立计时 `compile_and_capture_s`、proposal生成/上传、device推进、下载、CPU确认、**从收到问题到拿到CPU确认候选的总时间**。完整32槽native真值成本单列，不能从筛选总时间中暗藏它，也不能拿只测kernel时间对照CPU全流程。
5. native32基准须使用同样的CPU host预算2及同样候选/上下文；此有限研究不设worker sweep。同时保留32槽实算/缓存/无效数。
6. 并卡期间记录 A/B 原任务评价的 simulation-second / active wall、planning wall 分布、PANN observation/query wall、固定 Actor 推理 wall，以及 GPU 进程内存/峰值和普通利用率快照。本轮 PPO update 数为0，不以新 transition 吞吐作为交付目标。若以后另立学习 phase，再记录每 invocation 新样本和 update wall。比较相邻正常运行区段时明确姿态/事件分布不同，不能把变化全归 GPU 任务。

四问题全部完成或出现明确技术错误并保存first-error/input后自然结束。没有人为wall截止、没有周期超时kill、没有固定时间结束后称成功。若测得GPU排名漏掉native好候选、假可行严重或端到端不快，给出有限负结果并结束该支线；不再自动追加G5 E30、环境升级矩阵或大batch搜索。CPU正常研究继续。

## 5. 并行资源与进程约束

目前只有历史容量记录25 vCPU、120GiB RAM、96GB class RTX PRO6000；它们不等于当前空闲。执行端一次读取当前cgroup/affinity、内存可用量、GPU每进程占用、tmux/PID、disk和既有job。保留其他任务，不能按宽名字kill、也不因旧PID列表认定现在忙闲。

默认分配是**CPU A10 workers、B10 workers、GPU第三任务总host affinity2 CPUs，余3用于主控/系统/发布余量**。实际cpuset编号由当前允许集合选取；线程pool规模不等于进程总线程数，须让整进程亲和性与环境限制共同控制总预算。Torch/BLAS/OMP1，build -j6在双流重负载前做。第三任务native32验证也包含在其2 CPU额度内，不另开大型pool。不要将过时“2×12优先”再叠第三任务照搬为默认。

两条所谓CPU流并非GPU无关：

- `src/tm02/s1_material.py:72–118` 明确选择 `jax.devices('gpu')[0]`，PANN LearnedColumns是FP64 GPU jit/vmap Newton；固定128行block，`step`取回numpy（122–144）。
- `scripts/s1_go2_r10_observer.py:17–50` 每实际2ms观察，recovery查询有250次2ms constitutive step。它不等于PANN重新训练，但有真实共享GPU工作。
- r12资源 `torch_device=cuda`，固定 Actor 推理仍使用 GPU，历史 PPO 更新也共享 GPU；本轮不做新的 PPO 更新。保留 `XLA_PYTHON_CLIENT_PREALLOCATE=false`、`JAX_ENABLE_X64=true`，不为支线更换主流数值设置。

因此资源调度目标是有价值的联合吞吐。第三任务是可分片完成的4个工作单元，CPU固定策略评价不等待它的研究结果；若共卡实测压力大，优先维持两个CPU评价/PANN任务。没有证据可预先承诺96GB可容纳任意batch或某倍加速。

历史训练速度只可作为量级参考：W44 A单invocation512新样本/44,543.58375s，即12.3732h、41.3797新样本/h。W44 B最后一次resume仅186新样本/19,339.48920s，即5.3721h、34.6235/h；不能用512除该尾段时间。这些是历史采样阶段的测量，**不用于预测 W45 固定策略评价时长，更不是 wall 上限**。A的最终固定评价包含用户暂停，也不能直接作为纯计算基准。

## 6. 固定策略身份、未来续接点与统计边界

### 6.1 当前固定策略与保留的完整续接点

| 流 | 完整继续学习文件（仓库相对路径） | 固定评价policy文件 |
|---|---|---|
| A | `outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w44-a/PANN_PPO_S5-seed1/resume-state-ep00428.pt` | 同目录 `checkpoint-014737.pt` |
| B | `outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w44-b/PANN_PPO_S5-seed2/resume-state-ep00417.pt` | 同目录 `checkpoint-014737.pt` |

每流计数14737/57、tail0、excluded145。W45 分别载入自己的 `checkpoint-014737.pt` 固定评价，Actor不继续更新，学习计数保持不变。未来若另立学习phase，须用各自resume恢复actor/critic/optimizer/RNG/normalization、fork_identity与phase_lineage；不能假定固定评价policy文件含完整继续状态。这些文件主要保留在服务器，W44 diagnostics Release声明checkpoint bulk不在请求范围，不能因未见Release就误称丢失。

**本轮不执行 +512/+2。** 仅保留未来另立学习phase时的计数参考：若届时声明每流增加512条、2次更新，则目标是15249/59，第一批14738–14993/update58，第二批14994–15249/update59。该参考不是W45运行目标、扩样授权或自动触发条件。今后恢复时仍须保留excluded145、不重新fork/reseed、不混两支权重和tail，并显式声明新phase。

A/B共享W37的6545/25祖先，随后各自延续，不是从零独立种子。当前两支各14737不能相加说训练29474独立样本：唯一PANN谱系为6545+16384=22929条，其中145历史排除；另RAW独立256才是全R12 distinct23185。W45固定策略评价不新增这项训练总量；评价记录不能混入学习样本。

### 6.2 W44 physical、truncation与18条未生成记录

- A17个实际训练回合：14 body_fall、2 invalid trunk-contact、1 sample-goal truncation；材料加载9/17。
- B18个实际训练回合：16 body_fall、1用户pause truncation、1 sample-goal truncation；材料加载6/18。
- 合计35个实际训练回合=32 physical failure+3 valid truncation。最终固定E30两次body_fall另列，不能加到训练回合分母。
- 另有2个零物理步startup attempt：A `...ep00411`、B `...ep00399`。**18不是18次setup失败**：`diagnostics/evidence.json.unavailable_before_plant_start` 各9条，是这两个attempt没生成的feedback、planner、reference、trajectory、calls、prediction、result、physical、material源记录/模式。每条physical_steps0，data_substitutedfalse；按“2个技术中断、18条不可生成记录”报告。

两流最终E30分别11.934/12.742s body_fall，材料真实加载均0，合法去程/返程/往返均0。A base x曾越过.8投影边界也不能计足接触材料。W44训练PANN当前非零A6/512、B3/512；两次固定评价recovery全零。未建立材料历史收益、跨足复用或往返收益。下一轮首先需要区分刚地启动到首偏离的控制一致性、实际换支撑/接触和失败，再讨论材料记忆有效性。

W45每流直接使用自身update57固定E30；合格才按原同case E60/E300/E600次序。E60/300/600合法去程、返程与完整往返分别至少1/5/10，并满足原条件；30s未倒也不自动等于E30材料任务通过。固定评价发生物理失败就保留该评价终止与后级blocked，不转成继续采样循环。明确技术错误、用户暂停和物理失败分列，保留首个技术错误，不吞错重试到看似成功；旧训练专用r25停止开关不能污染本轮固定评价语义。

首失败指标要绑定时间轴：plan origin与输出plan身份、first command/projection/PD异常、首预测—实际差异、首真实lift/touchdown、首材料load、最终body_fall分别记录；先后发生不是单独的因果证明。action alpha、installed feedback scaling、raw K和实际applied Kdx不可混同。满时长/运动量/材料加载/合法穿越/训练目标/数据交付分列，避免继续一个status涵盖所有层次。

## 7. 交付与长期方向边界

CPU source/spec/patch/prepare/runtime/recording先冻结一种控制机制，再正常构建并启动两个已声明的固定策略评价job；不创建新的PPO学习phase。第三GPU任务用独立可写目录与已有隔离环境，不共享会被prepare改写的build树。原件留服务器，新增紧凑数值与脚本进Git，已请求生成原件增量Release，串行合并index的新字段，避免覆盖另一任务。不存在的物理源文件标明never-generated，不用空文件补齐；无PC下载镜像、无自动删除。

可参考MPX作者关于跨horizon/环境并行的实现思路，但其论文当前仅显式处理动力学约束，其余等式/不等式依赖barrier/penalty；不能直接继承为本项目硬PD/QP可行性。它的gait/reference工具也不意味着保留本项目phase、PANN历史与评估定义。长期GPU全MPC应单列模型/约束/政策语义对齐工作；本轮四问题支线回答一个更窄且可验证的问题。

## 8. 已核查的外部一手来源

以下均是2026-11-12浏览所得。实施仍固定本项目已有版本；最新文档用于解释原则，不能推定旧版本具有其全部新增字段。

- MuJoCo官方MJWarp概览与性能说明：<https://mujoco.readthedocs.io/en/latest/mjwarp/>。明确高吞吐和单步低延迟的不同目标、graph与容量考虑。
- MuJoCo4.0.0 make_data API：<https://mujoco.readthedocs.io/en/4.0.0/mjwarp/api.html>。nworld/nconmax/naconmax/njmax语义。
- NVIDIA Warp1.19.0并发文档：<https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/concurrency.html>。host同步、device events、graph及其顺序要求。
- MPX作者仓库：<https://github.com/iit-DLSLab/mpx>。
- MPX作者论文：<https://openreview.net/pdf?id=grAZBtvQJB>，尤其结论的显式约束限制。本文未借其硬件实验推算本项目speedup。

仓库证据以所列相对路径与commit定位。G1/G2复算见 [gpu-recomputed.json](gpu-recomputed.json)；W34–W44逐回合聚合见 [history-recomputed.json](history-recomputed.json)；当前W44计数、完整trajectory及学习记录见 [CPU报告](cpu-report.md) 和 [recomputed-summary.json](recomputed-summary.json)。
