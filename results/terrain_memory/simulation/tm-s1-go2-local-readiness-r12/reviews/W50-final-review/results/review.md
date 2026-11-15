# W50 结果、原件、GPU 与下一轮入口审查

审查基线：`09bd55e0cb70244938ad82db7dd202a31ac70f77`。本报告只做读取、原件解码及算术复算；新增 physics、native solve、FD、PPO、GPU world step 均为零。`analysis.json` 保存全部复算结果；`aggregate_w50.py` 可重复运行。

## 1. 本轮实际到了哪里

**W50 完成了一轮有证据的有限负结果研究，尚未开始新的完整任务验证。** `phase.json` 的状态为 `completed_finite_negative_published`。S5-r38 已构建、四个保存问题完成 native 候选评估、两次短接口检查执行且被正确标记为库不匹配，原件已发 GitHub Release。新的 A/B E30 尝试数为 **0**，E60/E300/E600 也均未运行。八个原任务阶段是 `blocked`，不能记成八次摔倒、八次运行失败，或 W50 的 E30 成功率 0/2。

W50 没有新原任务视频。`sources.json` 的 254 个原件中没有视频扩展名；`run_s1_go2_w50.py` 只会为实际原任务尝试渲染视频，而 `phase.json` 明确记录原任务未开始。当前可以与完整 30 s 日志对齐的仍是 W49：

- A：`job-20261113T145020Z-w49-a-eval-PANN_PPO_S5-s1-E30-attempt01`。
- B：`job-20261113T145020Z-w49-b-eval-PANN_PPO_S5-s2-E30-attempt01`。

本子报告负责精确日志与时刻，视频画面由主审查报告核对；这里不把 W49 画面称作 W50 新结果。

证据：[phase.json](../../../w50/phase.json)、[task-a.json](../../../w50/task-a.json)、[sources.json](../../../w50/delivery/finite/sources.json)、`scripts/run_s1_go2_w50.py`。

## 2. 独立复算的工作量与有限结果

按 `task-b.events` 计数一次，不能再把 `streams` 中的重复记录相加；再加入两个 deployment midpoint 的实际调用。

| 工作 | 本轮实际值 | 解释 |
|---|---:|---|
| 保存问题 native solve | 4 | A237、B236、A328、B625 |
| 保存问题完整 rollout | 76 | 每个 10 nominal + 9 正候选 |
| 20 ms midpoint native replan | 2 | 输入状态来自后来确认 ABI 无效的 bridge |
| midpoint 完整 rollout | 40 | 每个 1 R35 freeze + 10 nominal + 9 正候选 |
| 合计完整 native rollout | **116** | 与 phase 逐项一致 |
| transition FD | **23,112** | 不能混称完整 rollout |
| risk geometry FD | **4,746** | 同上 |
| derivative pass / backward attempt | **12 / 6** | 六次求解的记录 |
| candidate records | **114** | 76 + 38；额外两个 freeze 不在 candidate 文件中 |
| raw-center 零物理计算 | **76,650** | 6 × 12,775；不是 physics steps |
| bridge 物理子步 | **400** | 已执行，native 物理等价有效数为 0 |
| 新原任务 / PPO / GPU steps | **0 / 0 / 0** | 不把 GPU 旧日志分析算作新运行 |

直接读取三个 Release ZIP 中的六个 `snapshot.bin.r33-candidates`，解码全部 **114** 条候选，用归档 `inputs/*/input.json` 中的权重和 norm 对原始 residual 重算 J。普通候选误差约 `1e-15`，全体最大绝对误差 `2.3283064365386963e-10`，发生于 J 达 152 万的 B625 候选。114 条均 `failure=false`，状态/动作数值有限。这支持“求解结果确实如此”，不支持把不合格候选一律叫程序崩溃。

| 保存边界 | 当前 baseline J → 选中 J | J 降幅 | 最低 z：baseline → winner (m) | dx：baseline → winner (m) | 严格改善正候选 |
|---|---|---:|---|---|---:|
| A237，4.32 s | 2.48199081 → 2.44885351 | 1.33511% | 0.15308029 → 0.15322887 | −0.00429678 → −0.01814068 | 4/9 |
| B236，4.30 s | 2.87175363 → 2.87142512 | 0.0114393% | 0.14091890 → 0.14093525 | −0.01883277 → −0.01902412 | 2/9 |
| A328，6.14 s | 7.24939863 → 7.09672200 | 2.10606% | 0.13775498 → 0.14138284 | −0.00024841 → −0.02334256 | 4/9 |
| B625，12.08 s | 8.10979846 → 保留 nominal_8 | 0 | baseline 0.13055266 | baseline −0.00158548 | 0/9 |

36 个正候选中 10 个严格降 J、0 个数值 failure；**三个选中 winner 都后退，且仍处于下沉/低姿态预测。** A237 最低高度仅增加 0.14858 mm；B236 仅增加 0.016348 mm。J 的有效下降远超过浮点比较余量，不能说是舍入噪声，但也不能据此宣称恢复稳定。

原始记录补充了汇总表没有展示的反例：A237 的大 alpha 候选确实能向前 17.5 / 35.6 mm，B236 的 alpha=1 能向前 25.9 mm，然而终端 z 分别只有 0.1531 / 0.1489 / 0.1382 m，且 J 更高。准确结论是“被选 winner 都后退；未选候选中有前进但仍坍低的结果”，不是“所有候选都后退”。

B625 大 alpha 的巨大 J 也有明确原件原因：alpha=1 总 J 为 1,523,733.13279，`ImpactMargin` 占 1,523,721.76338；slot 3–8 的主导项都是 `ImpactMargin`。这不是 NaN 或 solver crash。现有方向离开局部低姿态邻域后引入昂贵冲击，而小步长又无法得到当前 J 的严格改善。

这些是从 W49 边界重建的当前问题，不是精确历史重放，更不是从 t=0 运行 r38 的闭环实验。W49 与 W50 的 fixture/机制不相同，不宜把两个版本的候选合格比例直接当成稳定性退步曲线。W50 对 4.3 s 及更晚状态的负结果，也没有回答“从 2.002 s 开始预防坍塌是否有效”。

证据：[task-b.json](../../../w50/task-b.json)、[deployment.json](../../../w50/deployment.json)、`task-b/{A237,B236,A328,B625}/snapshot.bin.r33-candidates`、`deployment/{A237,B236}/snapshot.bin.r33-candidates`、[analysis.json](analysis.json) 的 `original_payload_review`。

## 3. 视频所见异常有日志支持：相位在走，实际卸载没有跟上

Task A 是对 W49 已有原任务的完整新分析，不是 W50 新运行。每条流读取 1,521 次调用/边界、15,001 条 held action、150,000 条物理记录，并保留 A 116 / B 115 个完整策略事件。

两条流均在 call 121、t≈2.002 s 切换 Trot；private phase 按 2 Hz 推进，幅度 0.005 m、duty 0.45，没有简单“仍在 Stand / gait phase 不动”的执行错误。但 231 个留存完整策略事件的 reference states/actions/times、K、H 均与 t=0 初始 Stand 参考相同，参考终点一直是 0.35 s。每流 14,825 / 15,001 条实际 held 查询已越过终点，全部夹在 `[174,174]`，最后查询超过参考终点 **29.65 s**。安装了新 plan ID 并不等于生成了新参考。

| 原任务时间窗 | A | B | 判断边界 |
|---|---|---|---|
| 2.002–4.3 s | 前移 0.06134 m，四足同时负载 85.72% | 前移 0.05497 m，四足同时负载 86.74% | 已在移动阶段，实际卸载模式仍不像目标 Trot |
| 同窗目标 swing 时仍负载 | 每足 88.09–94.11% | 每足 87.81–93.53% | 相位目标不是实际接触时序 |
| 同窗 PD 饱和比例 | 0 | 0 | 不能首先归咎于长期饱和 |
| 5.2–7 s | z 0.1350–0.1465 m，非足支撑平均 55.14 N、超过 1 N 的比例 100% | 暂时恢复并前移 0.2879 m | B 短时好转不构成整段稳定步态 |
| 7–30 s | 非足支撑超过 1 N 的比例 91.64% | z 0.1307–0.2124 m，非足支撑平均 45.08 N、比例 88.45% | 依靠躯干等非足部位的持续支撑，不能当作正常行走 |

这里 1 N 是原分析采用的接触诊断阈值，不是新增停机门槛；反复跨过它也不能作为真实步数。物理支撑序列是 5 kHz 的接触力；candidate 的 176 个 foot 样本取每个节点第一原生子步，不是同一采样口径。candidate 未保留非足支撑，不能由其缺失字段反推非足支撑为零。

B 视频可重点对齐：4.3 s 高度 0.2673 m 且向下速度 −0.5033 m/s；4.62 s 总足竖向力约 852.72 N；4.7 s 蹿到 0.2916 m、足力为零；7 s 再次下沉；8 s 高度 0.1427 m 且非足支撑 24.76 N；12.08 s 已低至 0.1311 m；30 s 仍只有 0.1428 m。单个时刻足力为零可以是飞行，不单独判失败；持续的低姿态、非足支撑和目标 swing 无法卸载共同支持异常步态结论。

因此当前最有价值的因果试验，是在坍塌之前从同一正常初态运行，区分“新参考真正刷新后是否已足够”和“还需要能够进入目标接触节奏的 phase-aware 种子”。这仍是待验证解释，不是已证明全部根因。

证据：[task-a.json](../../../w50/task-a.json) 中 `private_gait`、`installed_policy_array_groups`、`saved_policy_lifecycle`、`actual_held_reference_query`、`support_windows`、`state_support_samples`；[analysis.json](analysis.json) 的 `old_original_support`。

## 4. 两类实际工程错误必须分开

### 4.1 40 ms bridge 的同 SONAME / ABI 不匹配

需要的 frozen MuJoCo 库 SHA 为 `235066cab0a2038050593b153e0eb10c90da546c56a3dd8ebeb8a6bd9f38dbd6`，实际绑定 Python wheel 库 SHA 为 `78f5455cbcbc2b4c1452f09a096aaa1d79893d63bbeb0bad93f610ddde19d264`。两者名义版本都是 3.2.3，但 native bridge 使用的 `mjData` 布局不等价。Python 提前加载 wheel 后，原 `RTLD_DEEPBIND` 没隔离同名库。

这有原始 NPZ 直接证据：A/B 每臂 200 / 200 子步的 actuator 数组都含 denormal 值，每臂 1,800 个字段；首行从 `3.81959242374e-313` 等开始。两臂 held action 与 native double 单次 FP32 cast 则位级相等，最大 cast 舍入约 `1.2e-7 rad`。**RPC cast 正确与物理 ABI 正确是不同命题。** 两条 40 ms 轨迹的 actuator、support、native 物理等价结论不能使用。

`scripts/w50_deployment.py` 已真正改为 `dlmopen(-1, ..., RTLD_NOW|RTLD_LOCAL)` 的独立 ELF namespace；记录只验证了修复后的动态库解析，没有修复后的积分证据。四个保存问题由独立 `agent_server` 子进程、显式 `LD_LIBRARY_PATH=S5-r38/lib` 运行，没有证据表明它们受 Python bridge 库污染。两个 midpoint native solve 本身仍是独立 native server，但其初态由错误 bridge 演化而来，故不能赋予历史状态/物理等价含义。

若下一轮使用正常 Python plant → RPC → 独立 native server 入口，不经过 ctypes bridge，最小必要验证应嵌入真正的早期闭环诊断，记录实际进程/库来源和动作链；没有必要强制再跑晚态 40 ms 作为准入条件。

### 4.2 本次审查发现的原任务入口语法错误

基线提交 `scripts/s1_go2_r11_runtime.py` 的两处新增 `dump(...resolved)` 在原 498 / 541 行被意外退格。静态 `ast.parse` 在原 line 499 报 `IndentationError: unexpected indent`。这会在导入阶段阻断下一次正常原任务入口，未影响已经独立执行的 W50 finite 脚本。主审查已经只修正这两行缩进，修后 AST 可解析，未 import runtime、未执行物理。本次修复不能冒称改善了已观测行走结果，但必须随下一计划一并纳入。

证据：[deployment.json](../../../w50/deployment.json)、原始 `deployment/*/real-interface-40ms.npz`、`scripts/w50_deployment.py`、`scripts/w50_common.py`、本次 `scripts/s1_go2_r11_runtime.py` 差异。

## 5. GPU 支线的真实进展

Task C 使用 W49 的 1,873,543 字节 NPZ 作 CPU 离线分析：3 个条件、33 个状态边界、30 对 step，耗时 **0.8841 s**，`GPU_context_created=false`，没有新的 GPU steps、warmup 或 kernel probe。

按约束类型、物理 DOF、接触 geom pair 和接触分量重新匹配后，第一子步三条件的 canonical `J/pos/margin/D/vel/aref/frictionloss`、smooth 输入和接触数据均相同。原先一部分数组差异确是 row allocation / padding，而不是物理问题差异。第一处可观察差异在 contact `efc_force`，继而 `qfrc_constraint/qacc`，在积分之前；F1 首步 force 最大差约 `6.64e-4 N`，L/F2 约 `1.53e-5 N`。

同一个 world 的 F1/F2 也可在同输入下产生不同 force，所以仅有 friction 行重排不能充分解释。原子分配/求和顺序是合理候选，但 boundary dump 不能定位第一个 kernel，也不能证明唯一 race。已安装 `_sort_constraints` 为 stub，单改开关不是已证成的修复。

须更正一个容易沿用的旧概括：本轮更完整字段比较，三个条件都只有前 9 / 10 步接触值相同，第 10 步不同，合计 **27 / 30**，不是“十步全部接触完全一致”。末段 qpos 最大差约 `1.86e-9`，qvel 最高约 `1.43e-6`；这些 FP32 微小差异本身不证明 GPU 工程不可用。当前真正尚未补齐的是 full native J 与等价候选排序，GPU 还不能替代 CPU 主结论。

本轮未找到可隔离修补的唯一原因，条件式 ≤20 step probe 没执行。C 不阻塞 CPU 稳定性诊断，也没有理由为了占用 96 GB 显存强行重复毫无判别力的运行。

证据：[task-c.json](../../../w50/task-c.json)、`scripts/w50_gpu_analysis.py`、[analysis.json](analysis.json) 的 `gpu`。

## 6. 并行与交付没有丢失，但不能夸大利用率

资源配置为 25 CPU，A 0–9、B 10–19、C 20–21、系统/发布 22–24；thread libraries 为 1，build jobs 为 6，保留无关训练。A/B finite 各约 12.7548 / 11.5159 s，实际重叠 **11.5159 s**。C 的 0.8841 s 发生在 finite 之前，不是三任务长时间同时占用；CPU affinity 也不是独占预留或实测满载。无 CPU/GPU 利用率时间序列，不能宣称跑满硬件。不要拿 W49 的长时并行时长描述本轮 W50。

phase start 到 finite 完成约 787.17 s，到 publication 完成约 1,027.17 s；这些是事后测量，不是后台截止时间。构建文件继承的旧 startup/build 字段不是本轮新增失败或工作量。

Release `tm-s1-go2-local-readiness-r12-w50-finite-part01` 已 live 核实为非 draft，3 个 uploaded ZIP 总 **464,340,186 bytes**，source manifest 为 **254 文件 / 569,226,459 bytes**。主审查下载 ZIP 并提供原件映射，本子审查直接读取候选与部署 NPZ，没有把索引代替原始数值。无需再做个人电脑归档，服务器原件保留。本报告也没有追加全文件 hash 门禁。

证据：[publication.json](../../../w50/publication.json)、[delivery.json](../../../w50/delivery/finite/delivery.json)、[manifest.json](../../../w50/delivery/finite/manifest.json)、[Release](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-local-readiness-r12-w50-finite-part01)。

## 7. W51 两个 8 s 配对诊断的入口与预算约束

主审查提出的同 seed 1、同完整初态、r38 reference-refresh 对照与唯一 phase-aware 种子处理各 8 s，是针对当前证据缺口的合理下一步。须落实以下边界，否则容易得到“看似跑了 8 s，实际上换了任务”的结果：

1. `run_s1_go2_r12.py --level` 只有 E30/E60/E300/E600，`configuration()` 会重建 stage duration。仅修改外层 duration 或墙钟 timeout 不会产生合格诊断入口。需要明确 D8/diagnostic 分支及独立结果标签，`qualification_usable=false`。
2. 不用 `training_duration=8` 绕过枚举：那会把 `kind` 切成 `continuing_acquisition`，继而 `arguments()` 走 long mission。也不用仍带旧 campaign 墙钟约束的 R11 CLI 直接启动。
3. 两臂均从 t=0 正常初始化，保留原 2 s 初始站立、真实 plant/RPC/独立 native server、原物理失效条件、冻结权重与 `PolicyStream(training=False)`；不从 4.3 s 保存快照起跑。初始随机状态相同，后续观察/动作按各自因果状态演化，不能强制回放未来相同控制命令。
4. 8 s 是预先定义的物理观察长度，不是后台墙钟完成期限。按现有 50 Hz replanning、20 startup calls 和额外 phase-transition call，预计每臂约 **421 次 planner call**、**40,000 plant 子步**，双臂约 842 calls / 80,000 子步；早停时更少。内部可能产生数千完整 rollout 和大量 FD，不能当作本轮 116 rollout 的近似工作量。事后按原生 counters 结算，不能用过短 wall timeout 截断。
5. 对照重点是从 2.002 s 开始实际参考新鲜度、摆动足卸载、足离地、躯干高度/非足支撑、前向进度以及与视频一致性，不能只比较总 J 或碰到低姿态前的几帧。使用相同时间窗和相同录制设置。8 s 诊断不能替代原 A/B E30，不能解锁 E60/E300/E600。
6. 两臂固定各 10 CPU，第三条用 2 CPU 做日志/视频/既有 GPU 证据分析，留 3 CPU 系统与发布。GPU 物理实验只在有独立可判别问题时启动，不影响两条 CPU 主线。诊断确定唯一版本后再跑一次完整 A/B 原任务，避免无证据地增加种子、超参或阶段。

下一步应优先解决并验证稳定正常入口和早期接触步态，而非立即扩大 PPO 或继续以晚态局部降 J 代替稳定性推进。两处语法修复和对 bridge 的正确隔离是工程必要项；真正的科学进展仍需要从正常初态获得稳定步态证据，并最终通过原完整任务。
