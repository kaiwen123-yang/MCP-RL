# W49 原任务、记录、GPU、并行与交付独立审查

审查基准：`de6606dc8363e8ab1c816fa682cace752a06318a`，对照 W49 计划基准 `b142001624cfea5877c57e63f45e0f29c9dc4735`。审查只读仓库，不登录 D57、不执行物理/GPU积分、不重启任务、不改仓库。`aggregate_w49.py` 使用标准库重新聚合 Git 内 JSON，完整输出为 `analysis.json`。

## 1. 最重要结论

**W49 两流均完成原始 E30 的 30 s 物理时间，且都没有触发提前终止；两流仍然 0/2 合格。新问题已经从 W48 的较早物理终止变成更清楚的“低姿态、非足支撑、低前进量或停滞，完成不了材料穿越”。** 不能说 W49 还是两次摔倒，也不能说已经稳定行走。

| 指标 | A / seed 1 | B / seed 2 |
|---|---:|---:|
| 原始物理时长 | 30.000 s | 30.000 s |
| 墙钟运行时间 | 6,101.02 s / 101.68 min | 4,753.37 s / 79.22 min |
| 原始资格结果 | task_criteria_unmet | task_criteria_unmet |
| 原因 | material_traversal_not_completed | material_traversal_not_completed |
| 最大 x | 0.349667 m | 1.020530 m |
| 最终 x | 0.348269 m | 1.020070 m |
| 合法路线进度 | 5.81745% | 17.01414% |
| 合法往返 / outbound / return | 0 / 0 / 0 | 0 / 0 / 0 |
| 初始 z | 0.290209 m | 0.290209 m |
| 最低 z | 0.123298 m | 0.130666 m |
| 最终 z | 0.138718 m | 0.142818 m |
| 最终 vx | 0.00001983 m/s | 0.00054401 m/s |
| 最终上层命令 vx | 0.212763 m/s | 0.217916 m/s |
| 净前进 / 27.998 s 平移阶段 | 0.01247 m/s | 0.03646 m/s |
| 首次非足载荷 > 1 N | 4.6232 s | 5.0216 s |
| 首次材料足端加载 | 没有 | 12.1866 s |
| 机体 legal-route west_entry | 没有 | 20.064 s |
| 最大材料主应变 | 0 | 0.006666 |
| 新 PPO samples / updates | 0 / 0 | 0 / 0 |

数据来源：原始结果 `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261113T145020Z-w49-{a,b}-eval-PANN_PPO_S5-s{1,2}-E30-attempt01.json`，同名 `-resolved.json`，以及 `w49/original-evidence/` 中两份汇总。净前进速度按阶段起点和终点计算，不是完整轨迹上速度的绝对值均值。

`s1_go2_r11_runtime.py` 的 body_fall 条件仍为 roll/pitch 超过 1.2 rad 或 z < 0.12 m。A 的最低 z 仅比高度终止线高 3.30 mm，B 高 10.67 mm。模型允许四腿 hip/thigh/calf/foot 与地面接触，所以“非足载荷”与“invalid_contact 提前终止”不是同一条件。这解释了为什么结束标签没有摔倒，但低姿态、腿部接触和难以前进仍然严重。此处不凭摘要认定究竟是哪条腿、哪块几何在支撑；完整物理流在外部原件。

对照 W48：A 在 26.432 s 因 invalid_contact 提前结束，最大 x 为 0.626506 m；B 在 11.638 s body_fall，最大 x 为 0.396364 m。W49 延长了两流的不中断时长，B 前进量增加，A 最大前进量反而下降。两个固定种子的事实支持“失败形态改变，取得局部进步”，尚不支持跨扰动稳定性或完整任务能力。E60/E300/E600 全部被 E30 资格阻断，不能将没有启动计为这些级别的物理失败。

## 2. 运动、命令、接触与代价证据

### 可直接从 Git 核验的时间切片

Git 内两份 resolved 文件只嵌入 A 116 / B 115 个事件窗口 call；全量 boundary JSONL 为每流 1,521 个 call，在服务器/Release 原件中。不能把事件窗口之间缺失的时间点插值成观测。

| 目标时间 | A | B |
|---|---|---|
| 0 s | x≈0，z=.290209，vx=0；初始站立 | 相同 |
| 2 s | x=-.000778，z=.290298，vx=.000069；仍是 phase 转换前边界 | 相同 |
| 4 s | 精确边界未嵌入 Git | 精确边界未嵌入 Git |
| 6 s | 精确边界未嵌入 Git | 精确边界未嵌入 Git |
| 10 s | 精确边界未嵌入 Git | 精确边界未嵌入 Git |
| 20 s | 精确边界未嵌入 Git | 精确边界未嵌入 Git；路线摘要另有 20.064 s west_entry |
| 30 s | x=.348269，z=.138718，vx=.0000198；命令 vx=.212763 | x=1.020070，z=.142818，vx=.000544；命令 vx=.217916 |

可用的邻近实际事件边界更有诊断价值，以下不伪装成整秒切片：

| 实际时间 | A：x / z / vx / vz (m, m/s) | B：x / z / vx / vz (m, m/s) |
|---|---|---|
| 4.400 s | .073747 / .268791 / .100428 / -.653390 | .067919 / .235130 / .145071 / -.325012 |
| 4.700 s | .061782 / .150726 / .001062 / -.046592 | .109826 / .291600 / .084065 / .668641 |
| 6.200 s | .062347 / .141776 / -.001530 / -.008087 | 无该边界 |
| 7.300 s | .060348 / .132436 / .024527 / .046635 | 无该边界 |
| 7.600 s | .077426 / .301537 / .180934 / .577005 | 无该边界 |
| 12.200 s | 无该边界 | .606950 / .131243 / -.002856 / .000082 |
| 29.900 s | .348269 / .138719 / -.000019 / .000002 | 1.019874 / .141468 / .039257 / -.041562 |

这些点证明较早就有明显高度下沉/上下起伏，不能以终点存活或一个风险标量为零替代稳定步行评价。

### 诊断事件按实际顺序

A：2.002 s 开始 outbound；4.422 s 首次 nominal PD 饱和；4.6232 s 首次非足载荷；6.238 s 命令不前进持续约 2 s（dx=.009870 m，command vx=.219536 m/s，四足接触标志均 true）；7.260 s 首次 target clip；7.420 s feedback-added saturation；30 s stage_horizon_reached。

B：2.002 s outbound；4.398 s nominal PD 饱和；4.508 s target clip；4.542 s feedback-added saturation；4.760 s `trunk_restoration`，mode=`scored_archive_restoration`，candidate=`k_guided_8`；5.0216 s 非足载荷；12.1866 s 首次材料载荷；30 s stage_horizon_reached。

两份 W49 事件清单没有 `first_actual_held_jump_over_1rad` 或 `first_actual_knee_limit_reversal`。源码在每次 applied_hold/held 都监测这些阈值，因此与 W48 的大跳变证据相比，是有价值的局部改善信号；仍不能从“没超过这两个阈值”推导闭环稳定。A、B 峰值力矩仍到达多个关节的力矩限幅，最大关节范围超出分别 .013865 / .014529 rad。

### 末次原任务 call 的代价与前进能力

| 指标，call 1521 / t≈30 s | A | B |
|---|---:|---:|
| J | 7.372798 | 6.940759 |
| MaterialSupportHeight | 7.171418，97.2686% | 6.744711，97.1754% |
| Height | .122728 | .119502 |
| Posture | .054147 | .059435 |
| Position | .007195，.0976% | .007648，.1102% |
| InputReference | 0 | .000004364 |
| candidate | incumbent | k_guided_4 |
| return / stop | incumbent_retained / backward_failed | policy_installed / w49_single_first_direction_finished |
| policy age | .22 s | 0 |

两流低姿态仍有巨大支撑高度代价，但大量局部下降和新安装没有转化成任务推进，值得追查“低位是否可恢复、控制搜索方向及实际导数是否正确、当前完整反馈律是否仍适用于该姿态”。A 尾段已经出现 backward_failed；B 则仍能持续下降安装却不完成材料穿越。

**不要据小 Position 项直接推断接错目标或应该增大 Position 权重。** boundary 中 mission.body_target 最终 x=3 是上层 mission 终点。active r26 reference 在每次 replan 重锚当前实际位置，配置为 horizon .35 s、maximum lead .15 m、maximum speed .25 m/s、acceleration .5 m/s²、userdata 84:96。native Position 使用局部 head reference，不能拿 x=3−actual_x 当作其实际 residual。精确 userdata packet 在 context NPZ 外部原件，本轮没有下载。局部前进参考缺少完整路线保证可作为设计限制，但单独的小 Position 数值不能建立因果。

## 3. 工作量、失败计数和实时性

两流各 1,521 native planner calls，其中 20 次 startup、1,501 次 normal planning。提交汇总给出的 positive alpha/k_guided installs 为 A 802、B 1,393，全部 chosen_actual>0，risk-priority cost-increase 为 0。两流合计：59,666 native rollouts、56,864 candidate records、6,012 derivative passes、11,579,112 transition FD calls、3,048,862 risk geometry FD calls、3,288 backward attempts、300,000 plant substeps。

这些是已提交摘要对全量外部日志的统计，本轮只独立读到 Git 事件窗口子集，不冒充全量重算。`positive_installs` 只包括来源 alpha_/k_guided_ 且 chosen_alpha>0，不是所有 policy_updated 的同义词。

**`internal_stagnation=0` 也不表示“没有优化停滞”。** `w49_run_summary.py` 只匹配 stop_reason=`native_search_stagnation`；r37 大多数 call 的 stop_reason 是 `w49_single_first_direction_finished`，A 尾段另有 `backward_failed`。必须同时审查实际 return、incumbent retention、search attempt、真实前进量，不能把枚举名变化当进展。

有限任务与原实验应分开：Task A 查询 82 次，积分 0；A1/A2 重建 8 个 20 ms rollout、800 子步；Task B 4 次 N solve、36 新 guided rollouts，总 native rollouts 196；部署 4×20 ms、400 子步；GPU 3×2×10=60 world-steps，warmup=0；全部新 PPO=0。

本轮还有 3 次技术尝试，不能计为额外“机器人摔倒”：

- 两次 Task A 查询启动失败，均在首次 PlannerStep RPC 处 connection reset，native solve/FD/physics/PPO 均 0。phase 记录修复为重新编译因 iLQGPlanner ABI 改动而失配的 service static archive，并设置导入查询 policy ready。
- 一次 deployment 初始化卡住，physics=0，精确 runner SIGINT 后保留 attempt；原因是 bridge 在 Python RPC model serialization 之前载入 sensor-free XML，改变了 saveLastXML 所依赖的全局 last XML。修复为先序列化原带 sensor model，再载入独立 predictor。不是后台墙钟截止杀死。

慢速同步模拟确实在跑：A planning p50=4,457.65 ms，B=2,527.99 ms；两流 1,501/1,501 次 planning 均超过 logical 20 ms。因为 `synchronous_slow_simulation=true`，这些 logical deadline misses 不会把墙钟延迟直接混进仿真物理时间；它们说明 CPU 成本高、离硬件实时尚远，不足以说明 wall deadline 导致运动失败。长时间停滞期间继续花多秒求解一帧值得优化，但应先修复有效搜索与恢复能力。

## 4. Task A 已澄清什么，尚未澄清什么

5 个查询 fixture / 41 个实际 state nodes / 82 次查询中，native query 对历史 native action 最大误差全部 0；原 applied held target 对 explicit-time RPC 全部 0；完整 policy import/export 精确一致。

时间量化最大约 2.3652e-7 s，五个 fixture 的最大 projected/action 差分别 .002180、.0000582、.0002639、0、.0034071 rad；最大限幅后 PD 差 .204427 Nm。A364 的 raw 差 .0001738 rad 在投影后直接变为 0。这些结果把“大动作跳变来自 action transport 没有正确执行”的简单解释排除了，时间精度不宜继续占据主线。它们不是对任意新状态、任意闭环非线性放大机制的普遍证明。

A1/A2 重建使用同一个明确 canonical cold-start predictor，8×20ms；A1 与历史植物从第一步就有差异。A213/A363 的反馈分歧明显（held 最大 .131865/.095200 rad），B176 两臂完全一致，B328 很小。历史完整 integration/external callback state 缺失，A3 明确 unavailable。因此能说“重建 predictor 与历史 plant 的差异会被反馈放大”，不能进一步只归因为某个物理模型参数或仅隐藏状态。W50 不应重复跑同一旧 fixture 希望缺失的历史状态自然出现。

## 5. 新记录能力与恢复缺口

`s1_go2_w49_records.py` 的改进是实质性的：pre-call actual_state、当前 native context、P0 完整策略、previous actually applied policy 的 accepted rollout、当前 solve/真实 held clock 分开；同物理时刻误差只在原节点时间差≤1e-9 s 时计算，不插值；每个事件保存前 .1 s、后 .2 s；大对象按内容去重；首次 held physics 前做同 attempt readback。

新记录还保存 mjSTATE_INTEGRATION（此模型 303 个 double，enum8191）、plant geom masks、mission/material/ground/observer 的数值数组及可 JSON 化元数据。A、B 各有 1,501 个可对齐 prediction pair，20 次 startup 不可用。最大同时间位姿误差为 A .0001943 m / .0009823 rad，B .0002346 m / .0015104 rad；关节速度误差可分别到 1.5884 / .6327 rad/s。小位置误差不能与稳定性同义，误差本身也不能建立失败原因。

**这还不是可任意恢复的闭环 checkpoint。** 实际 scope 明列未序列化 `material__history`、`material__legal_route`、`material__fk_data`、`material__mission`、`observer__ground`；native optimizer caches、actor tensors/RNG、任意 Python 对象也没有自动恢复。类里只有 capture/write，没有对应 round-trip restore。历史材料数组/世界坐标 ids 的保存改善了短窗 actual plant 复原前景，但应先从原件构造明确的恢复函数，并用原保存 state/下一步输出对齐核定其范围；不要从文件存在直接承诺中途接续原 episode。

其中 first-readback 仅验证关键字段存在、对象写出和当前 schema，不是一次完整 restore→replay 的物理等价试验。Git 嵌入 A 116、B 115 个事件 call，外部对象分别 394 / 403 个、146,547,287 / 164,681,966 B。全量边界和 held JSONL以及 NPZ payload 必须在下一轮直接于 D57 已留存原件读取。

## 6. GPU F1 / L / F2：结论应收紧但不重复同一试验

本轮锁定 MuJoCo 4.0.0、MuJoCo-Warp 4.0.0、Warp 1.19.0，FP32 device，2 个 world，dt=.0002 s，warmstart关闭，原 tolerance1e-8 的 FP32 表示，所有条件直接同步 `mjw.step`，没有外层 CUDA capture。固定旧 B158 held action，两个历史来源 action 精确一致；host/device ctrl 精确一致；S0 所有声明持久输入相同。

- F1：完整 S0 恢复后运行；两个 world 第 1 子步已 qpos 差 1.16415e-10（1 ULP）、qvel 差 2.38419e-7、qacc 差 .00109863。
- L：恢复明确的 F1 首步 S1，再只恢复 qpos/qvel/userdata/time/ctrl 并清 warmstart；第 1 子步 qvel 差 2.98023e-8、qacc 差 .00007629。
- F2：恢复全部暴露 Data 数组 S0（脚本验证所有数组 bytes），第 1 子步 qacc 差 .00012207；qpos/qvel 此时尚相同，但随后不同。

F2 vs F1 的相同 world 在第 1 子步也不同：qpos 最大 1.16415e-10，qvel 2.38419e-7，qacc .00115967。由此不能把轻量 reset 不全视为唯一解释，也不能宣称全状态重置已解决 GPU 重复性。

所有 3 条件首步 `ctrl`、`qfrc_actuator`、`actuator_force`、`qfrc_passive`、外部施力均两 world 位级相同，`qfrc_constraint` 和 qacc 已不同；排序后的 contact geom/dist/frame/efc_address 在各条件完整 10 步都相同。这将离线定位重点指向约束求解/归约及其写入链，而不是继续反复猜 fixed held action 不同。它不等于已经定位具体 race 或库 bug。自定义 `material_force` 的 `(world,dof)` 各有独立写目标，当前源码未见直接两 world 同写一元素的简单错误。

**W50 的并行 C 可以先零新 GPU 积分。** 原 `all-exposed-substeps.npz` 仅 1,873,543 B，已经保存 3×11 个状态边界与全部暴露数组。直接在 D57 留存原件上按实际 ncon/nefc、worldid 和有效布局规范化，找 step1 第一个不同的中间量及依赖链；读取已经锁版本的安装源码，检查 constraint assembly、solver/reduction、island/capacity、callback写集与重置覆盖。排除未初始化 padding，不能把垃圾区差异当物理差异。形成一个具体源路径假设后再决定是否有必要做一个新的、能区分假设的最小探针。

这是比再重跑同一 2 ms 表格更有信息量的第三任务。所有暴露状态相同不含 driver/allocator hidden state；当前不可能从摘要证明其严格等价。另一方面，约1e-9的 qpos 位级差也不自动等于工程不可用；若未来接受近似 GPU 候选生成，应先建立 native 完整成本复核与误差界/排名间隔，而不是无限追逐字节一致。当前 `full_native_cost_implemented_on_gpu=false`、`equivalent_native_ranking=false`，不能让 GPU 主导正式排序。相对 W46 24-world/外层capture等条件不同，不能把本轮范围外的差异称为旧方案已修复。

## 7. 实际并行与资源

原始 A 和 B 几乎同时启动，评估 PID 的实际重叠为 **4,764.730 s / 79.412 min**；分别 affinity0–9 与10–19、10 native workers。CPU20–21给C/交付，22–24留系统，CPU quota25，内存 cgroup128849018880 B。这是真双任务，不只是计划。

有限阶段不应包装成三条科学任务同时常驻：B 运行1794580474.895–0503.928；C于1794580684.384–0687.890，成功 A query于0851.571–0872.075，A-model于1291.641–1305.691；原A/B1515.5后启动。GPU C仅3.51s/60worldsteps，不能由affinity和硬件容量宣称96GB显存高利用率。正式PPO frozen模型推理用cuda也不表示训练或大量GPU负载。

建议继续10+10+2+3分栏，A/B放两种互补的、同一问题身份下的控制验证或固定种子原任务，C承担旧GPU数据/代码因果定位。CPU第三栏同时进行只读分析或交付不会构成额外物理风险；不要在两个10-worker求解器之外叠加第三个10-worker作业。必须等待依赖成立的部分顺序执行，不能为了“三并行”跳过前置证据。编译仍最多-j6，避免与两条求解同时抢25核。没有可检验GPU工作时，空闲优于无新信息的重复测量。

## 8. 交付和旧状态

Git 内三份最新原件delivery收据都为published，upload_exit_code=0、finish_exit_code=0：finite 4个archive/362个source，A3个archive/1964source，B3个archive/1974source；source未压缩总量2,132,720,687 B。A明确是同一已结束任务的postprocessing-only恢复，new physics=0。已有四个review MP4的catalog及size/digest远端校验收据，两个W49视频各30.034s，另两个是W48对照。原视频内容未在本子任务中下载观看；catalog不是视频视觉检验。

`phase.json` 的顶层自然结束和最终 stream results 应优先于历史 `publication=running` / `continuation=...` 等观察快照。`task-b.json` 的 pending interface、`build.json` 的 selected_mechanism=N、qualification=original not started、deployment_scope=N是旧阶段/继承字段；实际 runtime与最终phase为K、S5-r37、binary8637e3...。应在W50汇总中清理状态来源和阶段含义，不能据这些旧字段重复开跑原实验。远端Release现在是否仍存在、字节/digest是否对应，交由root独立live核验；本报告不冒充已读取所有外部原件。

目前遵守了不设实验墙钟截止、服务器原件保留、GitHub交付、computer_download=false。网络 fetch/push 的短超时是单次网络操作失败边界，不是强行结束后台实验的完成时间；应保持实验生命周期和网络传输状态解耦。

## 9. W50优先方向

1. **先用已完成原任务证据定位首次高度坍低，不再以旧fixture或30s存活作为全部目标。** 在D57原件对齐A约4.3–4.8s、B约4.3–5.2s及新非足载荷/恢复窗口的actual state、held target/torque、complete policy、accepted rollout、支撑高度/接触代价与导数；确认r26 actual packet。把任务进展、低位恢复与数值cost descent分列。
2. **基于主控制审查的完整策略更新问题进行一个明确机制修复和有限鉴别。** 此子任务的记录支持“lower cost仍不能恢复/走动”；不单凭cost比例调权、不扩大PPO。修复选择须由完整feedback/reference/action及搜索实现证据支持。短窗诊断通过后再执行A/B固定权重原E30，失败如实停在当前资格层，不硬开E60/600。
3. **补足可恢复范围的工程闭环。** 明确定义“短窗固定held真实plant复原”与“完整MPC+observer+actor闭环resume”的不同状态要求；先证明前者，再决定是否需要后者。禁止对历史缺失字段用默认零伪造精确replay。
4. **第三栏推进GPU旧数据离线归因。** 使用本轮已有首步所有数组，不重复F1/L/F2试验；仅在具体代码路径假设明确时开小探针。GPU完整native代价和独立native复核未具备前，不承诺正式加速排名。
5. **交付与资源规则保持。** 无后台完成墙钟截止；固定预算是科学试验边界，不能等同无止境实验；已有自然完成轨迹不重跑；推送GitHub和可用清单后不向用户电脑重复归档。记录质量修复和有限科学验证均须保持已有成功/失败证据，不通过改变标签淡化失败。

以上是子任务证据与建议，最终W50机制、预算和启动提示词由root结合控制源码审查及论文依据定稿。
