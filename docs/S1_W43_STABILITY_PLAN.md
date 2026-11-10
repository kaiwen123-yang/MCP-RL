# W43：优化器状态修正、双流联合实验与 GPU 失败阶段续接

日期：2026-11-10。审查基点：`c4b3f9adc698b15de5c94ef952e2884f3abfec62`。

**2026-11-10 实施状态：S5-r32 已实现并正常构建，根 r12 已声明 W43 的真实双父点与每流 512/2 目标。** 两支保存输入的 epoch/reset/停止状态已经观测：t0 第12次用完5次尝试后锁定，13–20次不再增长阻尼；2.000/2.002秒的新问题从1/1开始。稀疏 outbound 未找到 native 下降，A4秒无可部署候选，均保留，不称任务通过。[针对性证据](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w43-optimizer-inputs/result.json)。首版候选索引接线错误与修复前产物也保留。CPU联合训练按以下原定目标执行；GPU g5-20261110T033858Z 已完成6fixture/18FD请求并进入独立E30物理；两CPU各10workers的W43作业已启动并出现有效transition增长。

## 1. 决策与研究目标

首要问题是恢复**有效且可解释的运动优化**。不再用“进程结束”“没有跌倒”“有 PPO 更新”代替稳定材料通行。长期目标不变：原 Go2 在 TM02 材料上合法去程、返程和连续往返，并证明同一世界地块跨足/返程的恢复历史改善加载、落点及重访时间，最后再推进泛化、Go2 实机和 NX。

W42/S5-r31 已真实实现 native geom/pair 前向与独立平滑代理导数，两个 CPU 流各完成 +2048/8，到 13713/53，tail0、excluded145。两次末次 E30 满 30 秒但材料加载/合法通行全零，E60 未运行。当前最明确的数值机制发生在接触材料之前：两流 t=0 的第12–20次初始化规划无 native 成本下降，λ/rate 累乘到 λ=1e6；全部后续1501次非零时刻规划继承它。四个完整计划中 Qu 非零而 k 约1.5e-10rad、K约2e-7，指令无法形成有用运动。此证据支持修正生命周期，不证明修正后必然稳定行走。

W41 的127次物理失败和 W42 的36次驻留训练是不同失败形态。旧 qp2、完整约束 Riccati、r31 前向接线成果保留；不重复恢复 W40、重跑 W41 或原目标 W42。

## 2. CPU 单一新机制：S5-r32

基于 S5-r31。实现对象是优化问题身份、阻尼/增长率和停止/接受状态的生命周期；不同时改物理、奖励、PANN、步态或动作维度。

### 2.1 分开保存三个对象

1. **可复用的轨迹初值**：保留当前 warm trajectory、参考和时间平移逻辑，作为下一问题的初值。
2. **当前问题的候选及部署策略**：native 重新计算当前问题的 incumbent 和候选，记录策略身份、状态/参考版本、生成时间、年龄、K/reference 所属者。保留 incumbent 不等于证明其安全；不得用旧问题的成本标签假称当前已验证。
3. **当前问题的数值状态**：λ、`regularization_rate`、本次失败/停滞/停止状态独立管理。新问题首次求解从项目原始 λ=1、rate=1 开始；原 min/max、数值迭代 guard 保留。同一问题的内部重试允许调整阻尼，不每次 backward 都重置。

新问题由真实规划状态/时间与因果参考或目标版本决定。t=0 相同状态/相同参考的20次 startup 属于同一问题；2.002秒 Stand→outbound 即使因同一时刻重规划，也必须识别为新目标问题。每次真正新的在线 MPC 状态也应建立新问题身份。身份标记在 native planner/服务快照中实现，不为了一个标志随意增加 nuserdata=96 或改动既有 GPU 协议。目标/权重/步态改变、恢复或 Reset 时明确生命周期；worker 不共享可写状态。

这是本项目提出的实验改动，**不是声称官方 MJPC 每次 OptimizePolicy 自动重置阻尼**。官方实现只有 Allocate/Reset 清理完整状态，Crocoddyl 的求解入口则将 warm trajectory 与 init_reg 分开；相关源码与新论文见 [文献机制报告](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W42-final-review/literature-mechanisms.md)。

### 2.2 native 接受、重试与停滞

保持 r28 的 trunk 可行性和 native `(v,J)` 排序、qp2 容差、r30 完整 k/K/Vx/Vxx/dV 递推、r31 原生 forward/soft derivative 分工。不能以重置 λ 为由绕过 native 接受、强装未验证的大动作或删除约束。

backward 与 native 线搜索的失败处理必须在同一明确数值预算内：复用现有数值迭代 guard，必要的同名义点阻尼重试计入这一预算，不另叠一个无限循环。无下降和技术求解失败要区分；预算耗尽就明确报告。只有 incumbent 在当前问题的 native 评分有限、身份有效且满足既有部署规则时才可复用；评分失败或无可部署候选时保留明确失败，走既有错误/终止处理，不标记 policy_installed、不伪造成本、不新增隐式 fallback。新物理问题不继承上一个问题已经饱和的 rate。不能为了减少失败计数吞掉数值错误。

先执行当前问题允许的有限重试；确认停滞或预算耗尽后，锁定该 problem epoch 的停止状态。剩余相同 t=0 startup 调用不得重新获得一套重试预算或继续增加阻尼；只有新 epoch 才清除停止状态并重置 λ/rate。将停止状态写为 `native_search_stagnation` 等真实原因；它不是“已证明收敛”。只有未缩放梯度/可行驻点残差及原生下降支持时，才使用更强的收敛表述。alpha=0 后的 `expected=1e-16` 下界、巨大 λ 造成的 tiny k，都不能单独作收敛判据。浮点成本相等与真正下降分开记录，不靠单一全局经验阈值掩盖代理偏差。

每个新问题从1开始也可能丢掉接触切换时有用的阻尼经验，增加重试或暴露非下降代理方向；本轮专门测量这一风险，不把 reset 当作稳定性证明。若每个新问题只有一次成功的Riccati迭代，每次reset为1也意味着在线常以1求解，成功后降到0.5未必会用于同一问题的下一迭代；须记录实际λ_used、同问题迭代次数与动作量，不能假定reset就获得理想步长。

### 2.3 必要记录

在已有 planner/矩阵原件中加入：problem epoch 及变更原因；λ/rate before、used、after；未乘 alpha 的 dV；未正则化 Qu 与按现有边界计算的可行驻点残差；k/K；native 候选成本、实际下降与代理下降；接受/停滞/backward failure/复用 incumbent 的独立状态；部署策略身份/年龄。保留原10候选、176状态节点/175控制区间、accepted/reference/K/H/Q、十子步残差、物理动作与回报。

不要直接在旧保存 H 上减去1e6再求逆，当作完成了低阻尼 Riccati；必须真的重新递推。也不要只记录 `policy_installed=true`，因为 W42 全部3042次调用都安装策略却没有有效运动。

## 3. 核验范围：直接服务修正，不新建准入长链

只取本轮已有 t=0 初始化序列、2.002秒首次 outbound 与约4秒保存问题，保留相同状态、参考、控制、模型与可恢复的 warm context，检验：

- 同一初始化问题不再因机械重复把数值状态污染到下一问题；进入新问题 λ/rate 的实际取值与声明一致。
- 新问题内的 backward/native 接受与失败状态一致；Qu、k/K、实际下降可解释，不是日志字段改名。
- 物理前向/soft FD/十子步/任务约束保持 r31 分工，候选与部署 reference/K 身份一致。

若历史 warm context 无法精确恢复，明确 fresh 初始化或重建序列的范围，不冒称严格等价复放。接线错误修好后正常构建，直接进入下面的真实联合实验。没有必要再做旧12例 QP、刚地/纯 MPC/E30 准入、smoke/mock、恢复演习、全库 hash 认证、worker 网格或重复旧文献基准。

## 4. W43 双流真实联合实验

当前父点位于 D57 CPU 树 `/root/autodl-tmp/MCP-RL-w42`。下面路径相对于该树：

| 新流 | 自身父点 | 当前累计 | 本轮新增 | 本轮目标 |
|---|---|---:|---:|---:|
| w43-a / seed1 | `outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w42-a/PANN_PPO_S5-seed1/resume-state-ep00405.pt` | 13713 / 53 | 512 / 2 | 14225 / 55 |
| w43-b / seed2 | `outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w42-b/PANN_PPO_S5-seed2/resume-state-ep00393.pt` | 13713 / 53 | 512 / 2 | 14225 / 55 |

`512/2` 是本轮完整科学样本目标，不是测试完自动扩2048的门槛。既往每流2048样本约23小时且完全驻留，先使用足够完成两次 PPO 更新的真实实验判断是否恢复动作响应。达到目标后执行末次固定评价并交付；不因负结果自动追加同类样本或临时叠新机制。若现场已存在更晚有效 W43 点，按本轮冻结任务继续，不能再用 new-phase 重复初始化。

完整继承各自 actor、critic、optimizer、normalization、RNG、fork_identity、phase_lineage 与 excluded145；不互换、不重新 fork、不 reseed、不丢合法 tail。两流有共同祖先，是各自历史分支的工程复核，不能宣称两个从零独立 seed 或严格同父随机因果对照。W42 新增4096可相加；两支各13713不能合成一个27426样本策略。

实现后新建独立 r32 spec/patch、正常构建产物和 W43 phase 声明。按真实调用链补齐 artifact/逻辑版本、prepare/build、runtime/记录、恢复/发布与启动器支持；不要对仍写 W42 的根配置直接执行 `--new-phase`。新输出用 w43-a/b，父目录只读，原结果不改。当前根配置在本次审查中不提前换成未运行的 r32。

保持原 Go2、TM02 应变域.02、PD60/5、力矩/接触/跌倒/路线、r28 trunk、r27 C2、r26因果 packet/十子步/TV、冻结 PANN、658D、2Hz 标量 cap 和 W38 奖励；撤回的 r25 项不复活。

## 5. GPU 第三路：修复后从 G5 续接

原 GPU 工作已结束：G1仅5/32输入对齐，G2完成176/176节点，耗时152237.10秒；G4两个子进程缺 `--evaluation-start-unix`，物理未开始且 MJX request=0；G5把 device array 的 `opt.tolerance` 覆盖成 Python float，首次 kernel 调用失败。旧 `completed_declared_gpu_stages` 不是闭环成功。详见 [GPU 审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W42-final-review/gpu-review.md)。

本次代码已补齐评价时钟参数；Warp 使用原 device array 的 `fill_(native_tolerance)`，保留原声明1e-8的 FP32 表示，不能直接删赋值退回 `put_model` 默认1e-6；结果汇总区分技术失败、物理开始和成功。新增 G4/G5 显式续接、唯一 invocation 输出、冻结契约和新增证据发布。验证范围见 `gpu-fix-validation.json`，没有宣称 CUDA kernel 已通过。

推荐首次实际执行（仅在已结束的原 GPU 树核对并同步修复后）：

```bash
cd /root/autodl-tmp/MCP-RL
source scripts/autodl_d57_env.sh
python scripts/start_s1_go2_w41_gpu.py --resume-from G5
```

此命令跳过已完成 G1/G2 和原 stock MJX G4；保留旧失败与发布资产。G5 在独立 MuJoCo4.0.0/Warp1.19.0 环境完成原声明的6个实际 fixture、3个 FD 尺度和独立闭环阶段，具有新的 FP32 predictor 身份。仍使用冻结的 W40-B update37、r30 native 任务契约，不拿当前 W42/W43 根配置替换旧对照。新 invocation 文件不能覆盖旧日志、XML、NPZ 或失败结果。总体报告保留旧G4技术失败，本次退出码按 active_technical_failures 判断所选阶段；G5成功也不能抹掉旧失败，旧失败也不应触发已完成阶段重跑。分别读取本次G5状态、active_technical_failures与完整history。

当前 MJX 每节点实际中位899秒；首节点并非最慢，不能只归咎首次 JIT。它是每节点49个扰动输入的批量，而不是176节点整体常驻批量；现有显存峰不等于整卡可用量。不要默认再付出多日成本跑同一 stock MJX 全闭环。后续吞吐改造应定位固定接触/约束布局、resident 多节点/候选批量和 host/device 同步，保持原物理与数值身份。测有效节点/候选每秒与端到端耗时，不以占满96GB作为目标。

若 G5 的原声明阶段再次出现新技术错误，保存首次完整输入/版本/traceback，修明确接线问题；不自动切一串后端、降低物理要求或无限重启。GPU 是独立研究第三路，不能热替换两条 r32 CPU 主流。

## 6. 并行与运行规则

先核对现场 tmux/PID/job/clock、git 状态、最新有效断点及实际 CPU/RAM/VRAM 空闲量；25核/120GB/96GB是记录总容量，不是本次实测。保留其他 GPU 训练。旧 W42 与 W41 GPU 已结束不代表现场没有新任务。

默认 CPU A/B 各10 workers，Torch/BLAS各1，GPU host沿用原2核预算；C++ build `-j6` 在双CPU重负载前完成。可在新phase启动前依据现场明确改11/12，不给正在运行的普通 resume 热改冻结workers。第三路用GPU恢复/批量规划或轻量分析，不再加一条12workers CPU训练挤满25核。用新transition/墙钟、planner耗时、GPU批吞吐及实际内存峰判断分配。

CPU可复用已结束且干净的 `/root/autodl-tmp/MCP-RL-w42`，保留冻结r31产物和父输出；现场若有活动作业或未提交研究修改，则创建隔离 W43 树。GPU使用自己的 `/root/autodl-tmp/MCP-RL` 和隔离环境。新 CPU 可写 upstream/build 不得与运行中的 GPU 共享；venv、静态资产、已完成父目录可只读复用。没有必要为改名复制整库原件或归档到用户电脑。

**所有新建/续接后台 `deadline=null`。** 以样本/更新目标、声明评价完成、明确技术失败或用户停止结束；不设置4h/28h等固定墙钟期限。物理采集边界60秒、评价30/60/300/600秒、数值求解 guard、单次网络 timeout、实时性测量的逻辑deadline各有原义，不能一并删除。无墙钟期限不等于忽略错误或无限重试。

## 7. 评价与判断方向

每支达到14225/55后用自己的末次新update固定E30。原同case通过才E60，之后300/600仍按原顺序；`bidirectional_material_roundtrips_v1` 保留：E60/E300/E600的合法去程、合法返程、完整往返三项分别均≥1/5/10，同时满足全部原条件。原地站满、raw圈数或奖励改善都不能代替任务通过。

按四层报告结果：

1. **运行完整性**：技术错误、警告、父点继承、更新/尾段、实际评价身份。
2. **求解有效性**：初始化 λ/rate 轨迹、新问题reset、未缩放梯度、native下降、有效 k/K 和目标角响应；有无重新饱和与发生时刻。
3. **实际运动及任务**：正指令下净位移/速度、四足驻留与支撑转换、材料接近/加载、合法出带/远端/返程/往返、终止原因；这些是不同级别，不混作一个成功率。
4. **科学证据**：只有真实接触/重访后才讨论恢复状态与PANN因果作用；当前PPO只控制cap，记忆在候选内完整传播和动作权限扩展仍需独立阶段。

若 r32 仍无 native 下降且动作停滞，利用本轮已保存未缩放 dV/Qu 和方向数据区分代理偏差、局部候选缺失及目标/步态结构，不自动继续同机制大采样。若恢复运动但进入材料后失稳，按首个真实失稳链检查完整保持段执行、加载/接触转换与材料模型，不回头把所有失败归QP。

只有生命周期修正后仍显示接触探索不足，才另立**低维关节目标样条候选 → native完整评分 → 保留实际赢家 → 可选局部精化**实验。借鉴 Whole-Body MPPI、DIAL-MPC、Reference-Free MPC、Global CI-MPC/MTP 和 VR-MPPI 的明确机制；均值/精化输出必须重新rollout，局部失败不能覆盖已评分赢家，不直接把含1e6阻尼的H逆当采样协方差。不是本轮同时切换为完整新控制器。

## 8. 推送、原件与交接

先推实现/spec、针对性数值证据、真实父点和冻结新phase，再启动；看到两流真实transition增长和GPU本次阶段进入后交接。结束先紧凑Git，原件后台Release；CPU/GPU Git及index短写串行整合，核对远端并发更新，不强推、不覆盖新结果。只传新产物，不重复原已发布包；queued/published分开报告。

GitHub核验后**不在用户电脑另做归档，不自动删除服务器checkpoint/resume/原件、视频或上传副本**。保留历史失败、首次技术输入、真实动作与所有已请求诊断。最终给实际工作树、源码/产物、job/PID/样本/更新/tail、评价、GPU独立阶段、Git/Release状态与准确续接位置；实现、检查、物理运行和任务能力分别说明。
