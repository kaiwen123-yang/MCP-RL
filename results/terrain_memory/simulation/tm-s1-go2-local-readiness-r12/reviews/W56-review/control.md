# W56 控制、材料接口与下一步科学问题审查

审查源码：`3590646bc15a59903c9561381368d985963dfe9e`，相对上次计划提交 `6da12efc2cb85dcc99e53243dca489c9966e4969`。本文件由代码、已发布结果和已发布 exposure CSV 生成；没有新机器人物理、native rollout、QP 求解、材料求解或 PPO。本文数值除明确标注纯表格复算者外，是执行端原记录，不冒称全部独立重算。

## 1. 当前应当冻结两种不同用途的结果

**r55 是已有完整原任务资格的稳定参考；r60 是已有同样资格、材料正常执行接口已经接通的工作版本。** 两者 A/B 都完成各自 E600。下一轮不再将“找到能走的控制器”当成尚未解决的主问题，也不能将 r60 的完成当成材料记忆收益已证。

`w56/f/F-{A,B}-r60/E600-paired.json` 使用各自同 actor/normalizer 与 r55 对照，执行端记录的十圈完成时刻为：A 467.094→466.054 s，B 470.302→469.820 s。按原表纯算，r60 的全程归一化滑移代价相对 r55 分别 **+0.3174%、+1.8161%**，材料峰值应变 **+3.2661%、+4.4671%**。E300 滑移分别 +1.6611%、+2.1090%，峰值应变 +7.5902%、+4.4671%。两个版本均成功、完成时间略有不同，没有一致的材料性能改善。

这里不能用单次确定性闭环分叉后的几毫米、几毫秒或千分之几的累计差异建立统计可靠性。A/B 是两份冻结 actor case，不是两个充分覆盖扰动、土壤、随机初态的独立分布样本。原任务成功饱和后，反复跑相同 E600 主要增加成本，不能自动补上因果机制。

## 2. r57：真正执行了显式 12 维特征消融，但范围很窄

`scripts/s1_go2_r11_policy.py:418–446` 的 `actor_observation` 在原始观察进入同一 frozen actor 之前清零末 12 维；`install_cap` 将选中动作写入 `mission.task['command_speed_m_s']`，另一个模式仅作影子反事实。`w56/f/F-{A,B}-r57/E60-paired.json` 给出两例均 120/120 cap readback 匹配、误差 0，双方 E60 仍通过。

与 r55 配对：A 滑移 +5.4867%、B +2.8908%；峰值应变 A +3.1333%、B −1.8171%。A 完成稍快 0.092 s，B 稍慢 0.244 s。它证明显式特征有实际接线和确定性轨迹影响；目前没有充分证据说其提升了任务成功率或普遍降低应变。它只删除了恢复预测这 12 维，保留 20×27 的近期力/运动历史、其余 106 维状态及固定 actor。因此不可声称“所有记忆都删除了”，也不可由两个成功例断言 PANN 无用。

## 3. r58/r60 的真实数据流与正确性范围

### 3.1 因果窗口

`scripts/s1_go2_r11_runtime.py:799–804` 在每个完成的 0.2 ms 物理子步后提交其左边界足位置、速度及 start/end 时戳。`scripts/s1_w56_window.py` 只发布完整 10 子步包；发布后 observer time 等于消费时刻。r60 复用同一封包语义，在 CPU C++ FP64 内完成相同本构求解。

`scripts/s1_w60_cpu_observer.py:103–125` 在实际有限脚掌面积上找全部 world cells，以最低 sole z、相等时原 leg 顺序分配唯一 owner，不重复计算同 cell 的反力；cell=.02 m、sole area=.001 m² 对应最多 3×3 邻域。`observe:127–164` 将本次覆盖 cells 与仍在恢复的 cells 合并，逐 0.2 ms 推进；自由恢复只有在完整四维状态逐位不变后才停算，再接触可唤醒。

这不是材料参数在线重训。材料势函数、归一化、域和权重保持；PANN 估计状态因果更新，真实 plant 仍是 ReferenceColumns。新 PPO samples/updates/donor=0。

### 3.2 法向执行参考确实进入 WBC

`src/tm02/s1_material_prediction.py:23–67` 在同一只读当前快照、同面积/ownership、同 dt=.002 s 下反演目标与实测法向力：

\[
z_{ref}=z_{now}+\delta(F_{meas},\hat h)-\delta(F_{target},\hat h).
\]

反演使用整个共同 sole 的压力积分，保留非拉力条件、原 ±.02 主应变域；零力差严格给零参考增量。不会把绝对形变量再次加到已经下沉的观测脚高。`scripts/s1_w56_f_r58_backend.py:17–35` 检查 publication 时钟，记录两次 inverse 的位移/力残差，再返回真实 foot z reference；`s1_w54_f_r52_backend.py:608–617` 将其代入软支撑 normal acceleration task，进入同一个有原力矩、mapped input、关节可达和 friction 界的 WBC。

`w56/c/r60-interface-qualification.json` 的执行端保存输入核验覆盖 A/B 从 virgin 至首个全确认软支撑重规划点：A 5.42 s、B 5.46 s。全部已封包列、两轴应力、footprint 面积和反力一致；CPU/JAX 对照最大 e/q 差约 1.46e−14、应力差 2.15e−11 MPa、反力差 5.52e−10 N、实际参考差 5.20e−17 m、mapped input 差 4.44e−16 rad。这些是有限前缀数值证据，不等价于整个 E600 CPU/GPU 逐值相同；r60 是独立新身份并已完成自己的原资格阶梯。

### 3.3 仍未接入材料规划

`rollout_candidate`（`src/tm02/s1_material_prediction.py:128–148`）已实现每候选从观测快照 clone、预建未来 footprint 并集、完整受载和自由恢复推进；`candidate_step:70–125` 处理共同位移边界、重叠 owner、一致提交和力残差。

但其在线调用尚未进入 F 的 `replan`。r60 继承 r55 的 force MPC：`scripts/s1_w55_f_r55_backend.py:49–80` 以及父类 `s1_w54_f_r52_backend.py:473–505` 仍使用原接触表/foot plan/质心模型，`force_normal_upper` 为 infinity，原材料候选目前仅在离线 qualification 调用。r60 的新语义是**材料历史参与当前法向执行参考**，不能写成材料历史已经改变未来 force/path 的 MPC 优化。

脚落点也是现存限制：父类 `_start_swing:278–282` 仍用固定 k 的局部 rest 估计法向落点，未来 `_future_contact_feet` 在新落点处用 radius。新材料预测不能只写日志，必须进入确实消费的规划矩阵或 foot reference 才算新增控制权。

## 4. 目前收益弱的两个可量化原因

### 4.1 同一状态下的差分参考可能抵消主要历史偏置

若在当前工作区间 \(\delta(F,h)\approx b(h)+F/k(h)\)，则 normal reference 中两次 inverse 的差为 \((F_{meas}-F_{target})/k(h)\)，历史导致的自由表面/残余压入偏置 \(b(h)\) 被相消。这个相消本身避免绝对压入重复叠加，是合理锚定；但它也解释为何“接上 PANN”未必带来可见动作变化。

已有 `w56/c/interface-qualification.json` 的首个软接触保存点 5.420 s 提供具体证据：

| 保存输入比较 | 最大变化 |
| --- | ---: |
| r58 vs r55 soft reference | 1.11917e−7 m |
| r58 history vs 全 virgin 模型 soft reference | 8.85888e−11 m |
| 同一 history vs virgin 的 mapped input | 5.89135e−11 rad |
| 同一 350 ms 材料候选绝对位移 | 6.68139e−7 m |

这是一个早期加载点，不能外推所有重访；它要求下一轮以完整事件族核查 absolute delta、incremental delta、tangent、首动作变化与真实执行误差，不以“数值不为零”当科学效应。

### 4.2 跨圈已经几乎完全恢复；真正剩余信号集中同圈跨足

本次独立读入 `w56/exposure/A-E300-query-cells.csv.gz` 与 B 表的全部记录（源身份为 r55 A/B，不是 r60 查询），筛选 `exact_query_next_touch_overlap_m2>0`，按执行端已有 `same_world_cell_history` 分类。可复现脚本 `recompute_history_signal.py` 只读 CSV/JSON，不导入任何模型。完整结果在 `history-signal-recomputed.json`。

| 真实下次触地有面积重叠的查询 cell 类别 | A/B 查询时 \(|q_{11}|\) 中位数 | A/B 2ms 自由恢复深度中位数 |
| --- | --- | --- |
| 同圈跨足 | 2.63565e−4 / 2.50270e−4 | 9.11345 / 8.65472 μm |
| 跨圈跨足 | 1.14501e−11 / 1.14392e−11 | 3.85841e−13 / 3.85445e−13 m |
| 跨圈同足 | 1.14550e−11 / 1.14528e−11 | 3.85991e−13 / 3.85920e−13 m |

同圈跨足的真实 plant 下一次受载前 \(|q_{11}|\) 中位约 2.34146e−4 / 2.32607e−4；跨圈两类该中位数约 1e−21，接近完全恢复。跨圈跨足仍有个别较大极值，完整表保留，不以中位数断言每个 cell 都严格为零。统计粒度是匹配查询-cell 行，非独立机器人样本、非所有加载事件；重复查询可能多次提到同一后续 load，故同时记录 unique next-load IDs。

因此数千个 revisit event 不能直接支持“跨圈记忆收益”。原路线一圈约 47 s，而此模型局部恢复秒级；将原任务从五圈加到十圈并不能自动放大残留历史。先研究已存在的同圈跨足/近时重访更有辨识力。

另一个预测范围限制：F horizon 仅 .35 s，小于标称同足 .5 s 周期。它可以读已有历史，不能直接完整优化“本次加载对下次同足加载”的整个因果链。若要研究主动跨步协同，应另外明确较长材料预测层或新的任务/目标，不能将 350 ms 一次 rollout 叫作跨步联合规划。

## 5. r58 挂起：已绕开，不应伪造精确根因

`w56/c/r58-e300-stall.json`、`r58-e300-recovery-outcome.json` 区分了强线索和未知：A 最后成功 MPC 是 119.700 s，B 是 78.140 s；两个 GPU context 有负载，但长期无物理/MPC推进。CPU 的真 plant/force MPC/WBC 在这条路径之外；最强候选是 JAX learned-column free/mixed solve 或设备结果回读。A 前 37 个已保存计划当前足都在材料带外，free recovery 路径尤其可疑；B 最后记录的两个 normal inverses 已返回。

互斥 peer GPU、Lock/Unlock 以及完整 Checkpoint/Restore 均未恢复原物理时钟。精确主机 PC、设备 kernel 与驱动层根因没有采到；不能写成已经证明 GPU 并发争用、某个 Newton 无限循环、材料权重坏掉或者后台截止导致。原 context 成功返回 CUDA API 状态不代表业务状态推进。用户授权后停止了这两个确切子进程，保留原件；不是后台达到墙钟上限自动杀任务。

r60 用 CPU FP64 小批材料求解消除了这条在线 JAX 依赖，保留数学内容，A/B 实际 E600 全通过。当前应保持 CPU 在线，GPU 服务独立大批候选/材料计算。没有充分理由为利用显存把已验证稳定执行链搬回同一路径。新的 r60 `CallMarker`/SIGUSR1 是取证，不是自动 watchdog/deadline。

## 6. N/r59 仍未完成任务，但这已不再阻塞主线

`w56/n/N-A-r59/E30-actual-analysis.json`：30 s 原任务自然结束、150000 子步，x 最大 1.03983494 m、末 x1.01368234 m，材料西边界进入7.256 s，未到东边界2.2 m；min z=.27212765 m、max strain=.00747034，原 E30 失败原因为 material_traversal_not_completed。不是 r58 GPU 挂起，也不是新r60不稳定。

新 QR/dual active-set 投影在旧保存的15.993–15.996 s四个可行矩阵上消除了假不可行，15.997 s 的真不可行仍被拒；同输入 call809 handoff 从后退35.83 mm改善为前进9.84 mm，call856仍后退5.23 mm。实际完整任务里 handoff 提案1521个、可行1519、安装0；修好了一个局部候选不等于选中了可行推进解。

`E30-native-objective.json` 已有三个时刻原成本分解。5.660 s，handoff 预测前进80.50 mm，比被选候选约21 mm大，但原 J=.062409 高于已选 .038139；handoff 的 Upright .019698、Height .005238、Posture .010139 明显较大。它不是“选错数值最小J”的直接证据，也不能强装。全部晚期停滞时刻的分项未由这三个点覆盖，不应把其最初根因唯一确定成某个权重。

既然 F 已有稳定接口，建议 N 暂保留诊断和成果，不再与材料科学主实验争两条物理槽。若未来明确选择非步态约束的运动创新，应独立恢复，而不是为了继续消耗25核不断修同一非主线。

## 7. 可直接实现的下一轮干预定义

### 7.1 先明确要检验哪一种记忆

推荐第一项为 **MPC 对已卸载地块跨接触记忆的使用**，而非一口气声称删除一切材料历史。做在每个规划候选的私有副本：

1. 从同一已发布因果 observer 快照复制候选；真 plant 和主 observer 永不重置。
2. 完整历史臂保留全部估计 e/q/stress。
3. 消融臂保留当前实测接触/受载足完整几何 footprint 及共享 ownership 组内全部 cells 的完整元组和实际测力/足边界（不仅是 loaded=True）；对当前真正脱离这些足印组、过去曾受载、此次候选 footprint 会查询的 cells，把私有元组整体设为 virgin `(e,q,stress)=(0,0,0)`，表示模型假定这些格已完全恢复。不能只清 q 还保留不相容 e/应力。
4. 两臂从此同样按候选完整受载/恢复时钟演化；horizon中不反复清零。当前 actor、即时测量、r60 WBC normal reference保留；变化只作用于未来规划所读取的旧地块记忆。
5. 每个candidate记录 reset world keys、哪些属于当前接触完整足印组所以保留、publication time、未来实际查询的面积及独立状态变化；主状态不可被candidate污染。

当前 CPU observer 没有更新 `ever_loaded` / `last_owner` / `last_release` 元数据（旧 observer 也以 log=False 工作），**不能直接读取现成 ever_loaded 当真值**。应在 causal packet 中依据自身估计加载/owner transitions维护一个专用history mask和event id；记录期间的字典/复制/稀疏恢复语义必须同时进入候选状态合同。可用真 plant events离线检查，但不能把真q或真historymask喂进控制器。

这个干预有物理解释：无接触时控制器不再保留此前形变/耗散状态，未来按完全恢复模型预测。它不改变真实地面，不改变当前受载条件。实验只能归因“使用卸载跨接触记忆”而非当前接触内记忆。若在 r60 仅当前normal reference 上实施同样干预，作用通常会很弱，不能因此否认未来规划通道。

### 7.2 最小材料规划接口

沿用 r60 标称 gait、xy落点与接触表、原MPC代价和硬界。先取原 force-QP 解作为候选，完整14节点传播材料，得到 loaded sole z、切线、force residual和domain状态。将材料预报的normal位置进入 force moment arm / 线性化模型，做一轮更新QP产生第二候选。两者都在相同完整材料+质心预测模型下重算原 state/force/force-difference 代价及原约束，再选；不能拿旧线性模型目标和新物理模型目标直接比较。

实际行动仍经原完整 WBC。比较影子目标、真正获选force、执行reference、mapped input和下一个物理段；原方案也必须按同一新预测评估。不能任意提高材料惩罚、降低力上限、强行挑历史状态小的候选来制造差异。若未来确实要优化能耗/沉陷/跨步资源，公开新增研究目标并预声明权重含义，不能称仍是原J。

这里还有现实限制：本次实际沉陷不足约0.2mm，仅 moment-arm 几何通道可能很弱。保存完整事件族若显示同输入候选排序和首动作均无辨识变化，应报告该模型/任务/通道的限制。不能在没有信息收益时自动堆两条E600。

### 7.3 独立短重访任务

原r55/r60任务资格冻结，另建明确的补充研究任务，不修改原plant、材料参数、0.2ms物理、PD/WBC硬界和标称2Hz/.6/3cm。可以从virgin原起点走入材料中部，在安全几何范围维持实测交替抬落足与小幅平移/原地交替踩踏，形成实际0.5s量级同足重访；对照等待/间隔依据已有模型恢复尺度预声明。必须实测卸载、下一脚落到同cell的面积与残留状态，静站持续承载不是重访。

实现时不要为了新任务添加 phase 维数导致658D frozen actor语义变化。F是否迈步来自 `mission.phase not in STAND_PHASES`；可用独立 task_mode 与既有观察phase承载任务逻辑。`PolicyMaterial.observation` 当前目标硬编码3/0，新增任务应显式从任务读目标，保留原任务默认值并记录新观察语义。局部原地阶段若不消费PPO速度cap，要明确它是材料控制机理任务，而非PPO联合收益；之后回到真实平移任务检验迁移。

如果补充任务改变路线/阶段或物理观察长度，使用独立qualification标签，不能把短任务通过继承为原E60/E600。任务实例可以增加有限重复以建立效应方向，先预声明条件/重复规则，不因某次负结果任意换seed或材料软度。

## 8. 本轮还应清理的说明性遗漏

- r60配置 `wbc.material` 字符串仍描述旧固定k模型，顶层 `material_reference` 和实际代码才是新语义。后续新身份可以纠正说明字段，但不能改原合同SHA冒充原case没变。
- 当前 `normal-execution.jsonl` 有publication、两次delta和总owner面积，却没有每次引用的worldkeys/e/q原快照；未来因果干预应直接记录足够紧凑、可还原的观测状态/keys，避免又以稀疏保存点代全程。
- r58挂起精确根因未知。记录执行端实际 CPU/material/query/MPC/WBC/IO时间边界，将GPU batch吞吐与单机器人延迟分开；不再凭96GB容量默认在线GPU更快。
- 本次不更改控制器或开展新实验，只为下一轮给出证据和实现范围。最终研究结果允许为“此固定材料/任务的历史效应不足”，这是有价值的排除结论，不需用重训或放松原约束强迫产生正结果。
