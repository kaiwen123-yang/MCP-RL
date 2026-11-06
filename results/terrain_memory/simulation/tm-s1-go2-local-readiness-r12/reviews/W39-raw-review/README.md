# W39 原始数据补齐后的二次审查与 W40 启动前更正

日期：2026-11-06。数据与代码基准为 `07b0ee3e15c9f10e9b1a6a1de3997bec36f39556`。本次下载、解压、解析 GitHub Release 原件，做保存数据之间的对齐、源代码审查和纯代数复算；**没有访问 D57、重跑物理、重训或启动 W40/GPU。** 本报告修订上一轮 `W39-review` 的证据范围和 r29 数学建议，历史结果不改写。

## 1. 结论与本次真正改变的判断

S1 仍处于“完整联合工程能运行，但原任务上的稳定通行未建立”的阶段。W39 两流各到 8593 transitions / update33，62 个训练回合、59 次自然失败、合法去程/返程/往返全 0；末次 E30 仍为 A 10.476 s、B 17.592 s 跌倒。补传没有产生新的通过成绩，却显著提高了失败解释的分辨率。

本次需要在 W40 启动前更正六件事：

1. **撤回上一版 r29 的 Qux 中心 QP 公式。** 其“无新增约束时恢复原反馈”的断言在原目标角活跃集/line-search 条件下不成立。改为保留旧输出的非对角 H 度量投影，明确属于策略修订，不冒称完整 constrained DDP。
2. **accepted→actual 的 +20 ms 差是闭环轨迹分歧。** 候选与真实系统随后反馈所见状态不同，输入也随之不同；不能把差值直接叫同输入开环模型误差，更不能把选中窗口里的第一次差值当全回合首因。
3. **第一次材料加载前已经存在更严重的反馈异常。** B 在 15.264 s 的 raw Kδx 峰值 48.914271 rad 位于旧窗口间隙；首次 load 是 15.3816 s。它不是实际执行的 48.9 rad 关节目标。
4. **PANN 大量为零有可核查的空间与采样原因。** 1962 条零输出中，1820 条四个未来查询足全在材料外，140 条查询没有先前真实加载地块，余下 2 条是 2 ms 观察端点之间的短负载。没有证据支持“PANN 接线断了，所以应马上重训”。
5. **材料域退出主要暴露了毫秒级落足/载荷问题。** 已读稠密原件显示真实力与材料输出一致、Newton 残差正常；越域候选行没有进入物理积分。2 Hz cap 来不及在一次 1.2 ms 加载失败中收到新观测再反应。
6. **B272 的 60 s 是控制退化后的异常局部平衡。** 前进指令、cap 和规划一直存在，后 40 s 只前进 2.69 mm，右后腿长期卡在两个关节目标上界并离地。减少跌倒或饱和次数不足以证明修好了通行。

因此可以继续 W40，但应使用此次更正后的控制数学、指标与 GPU 重放合同。继续两条主线及第三 GPU 工程任务；不增加 PANN/奖励/步态/物理模型等第二个主要改动，也不原样再跑一轮 r28。

## 2. 原件覆盖与证据边界

服务器已在北京时间 2026-11-06 17:25 对 Release 清单核对完成：A 492/492，B 484/484，共 976 个资产；传输文件 7,680,574,905 bytes，原始源文件 8,754,284,739 bytes。这两个口径不同，前者含 gzip 后大小。

本次完整读取除 62 份训练 MJB 外的 914 个资产：所有 64 回合的反馈、planner、reference、trajectory、prediction、配置与材料事件，8 个 PPO rollout、selected transitions、全部已选 physical/material，以及两份 E30 的 MJB；另外 62 个训练 MJB 只列清单，不重复下载。所选传输大小 4,998,268,409 bytes，原始源文件 6,071,978,243 bytes。914/914 已完成下载和解析，包含332个NPZ、613080行JSONL，解压源大小全部一致、解析错误0；1133个非有限数全部为下述last_release哨兵。逐文件状态、数组形状/非有限值分类以 [read-coverage.json](read-coverage.json) 为准。解析/逐数组读取不等于对每个数字作了因果解释。

重点语义复核覆盖两份完整 E30、全部训练反馈/计划与执行轨迹、2048×658 的 observation/next_observation、全材料事件，以及 A254/B256 的 0.2 ms 施力序列。`material-final.last_release` 中的 NaN 是源程序定义的“未曾卸载”哨兵，不能和状态积分发散混为一谈。

**上传完成不等于当年记录了所有量。** r28 完整 accepted/K/H horizon 当时没有保存，prediction NPZ 是 feedback reference；A260 不在当时 dense 展开集合中，没有它的完整 physical/material 数组可以补传。现有原件足以核对其逐材料事件和 500 Hz 轨迹，不能据此虚构亚毫秒未记录状态。GPU 也不能拿未保存的旧求解器内部状态声称精确重放 W39 整个 planner。

## 3. 控制链：排除接线猜测，保留真实的不稳定证据

详见 [control-audit.md](control-audit.md)、[control-e30-analysis.json](control-e30-analysis.json)、[control-all-attempts.json](control-all-attempts.json)。

两 E30 分别对齐 5238 / 8796 个真实执行控制时刻；剔除 startup 重复请求与终端复制诊断。执行动作与 trajectory 的 float32 值一致；`u_before=u_nom+sKδx` 与目标范围裁剪等式精确成立。使用完整参考轨迹还原全部 14034 个执行时刻，nominal 差为 0，切空间 state_delta 最大差不超过 3.25e−15。已保存 accepted 节点按同 reference/K/u_ff 重算动作误差约 1.33e−15 / 3.45e−13；全部带节点计划的 origin q/v 与实际相同。参考 packet 与 userdata 对应通道一致，没有引用未来 actor 决策。

这排除了当前原件能检验范围内的动作接错、reference 身份错、漏做目标 clip、packet 错位等解释。它不证明接触模型、局部导数或反馈增益适合真实系统。

| 事件 | A E30 | B E30 |
|---|---:|---:|
| 首次原始 PD 请求饱和 | 9.224 s | 14.734 s |
| 首次 raw Kδx > 0.5 rad | 9.224 s，1.168303 rad | 15.044 s，1.295255 rad |
| 真实材料首 load | 9.850 s | 15.3816 s |
| load 前大反馈控制时刻数 | 8 | 22 |
| 新发现的 load 前大峰 | 见全量控制表 | 15.264 s，48.914271 rad |
| 全回合 raw Kδx 最大值 | 9.922 s，5.851179 rad | 15.744 s，86.680943 rad |
| 首次非足承载 | 10.0912 s | 16.1296 s |
| 原规则 body_fall | 10.476 s | 17.592 s |

B 的 48.9 rad 位于旧窗口 15.144 与 15.282 s 之间。86.7 rad 则在真实 load 之后、非足承载之前。两者都是裁剪前反馈量，不能解释为真实角度或直接据此更改物理阈值。

完整计划还显示：两 E30 的 544 / 900 次计划全部 accepted trunk violation=0，但 B 在 16.58/16.66 s 另有两次 nominal 不可行→accepted 可行恢复；这些晚于非足承载，不是早期根因。全训练存在 accepted 仍不可行的恢复阶段，不能把两 E30 的结论推广所有训练计划。全62训练中，60回合出现大rawK和执行PD饱和；58个loaded回合中57个两事件均早于load，唯一例外A256。全训练已记录51685个计划，51451个前后trunk可行、165个恢复至可行、69个accepted仍不可行（18回合）。这些是跨回合关联证据，不能证明每个失败都只有同一个原因。

训练实际有51693次planner调用，少于它的51685条新计划记录并非上传缺失。8次均为B276的无新policy调用（13.48s、13.52–13.64s）；第一处mu变化符合五次backward失败，余7次无法由旧日志唯一分类。末plan695原点13.50s持续用到13.650s，逻辑年龄150ms，13.652s跌倒。首大反馈11.524s、首load12.9166s在前，所以这是晚期退化，不改写早期首因。下一版必须记录每次调用完成/提前return及policy复用。

旧 `acceptance_mode` 将恢复到可行命名为 feasible_cost 的日志问题，下一版应修；旧原件保留，以前后 violation 重新分类。

轨迹误差比旧窗口更早存在：A 7.72→7.74 s 已有闭环速度差，B 13.64→13.66 s 已有 FR thigh +1.82997、FR calf −1.18023 rad/s 的差。此前系统仍能继续运行，所以“发现误差”不是充分失稳条件。已保存 K 节点也出现增益显著增大，最大元素常在 base-z 列、单位 rad/m；不能与速度列混用单位比较。爆发当时的完整 K 未存，不能用 node0/node10 冒充 node2 的 K 来编造分项归因。

当前最稳妥的工作假设是：**持续的接触/跟踪偏差与过强局部反馈、执行器活跃集变化相互作用，破坏支撑。** 这比“某个接触参数已经被证明出错”更符合原件。下一轮记录应区分闭环分歧、原 PD 不可实现请求、反馈新增饱和、实际施力和合法前进。

## 4. 材料与学习：原始数组补出的解释

详见 [material-learning-review.md](material-learning-review.md) 与 [material-learning-audit.json](material-learning-audit.json)。

### 4.1 PPO 的数据交接没有出现所怀疑的错误

8 个 rollout 的全部 2048 条 obs/next_obs 有限，1986 条同回合相邻交接逐元素相等；action、logp、value、终止/截断标志、PANN 与紧凑表一致，reward 只有保存为 float32 的舍入差，最大约 4.73e−7。查询 mask 与几何一致，没有材料外非零 PANN。查询所用计划的逻辑仿真年龄通常 20 ms，B 最大 40 ms；这不是墙钟规划延迟，也不能替代每个控制tick的复用policy年龄诊断。

| 当前 PANN 特征类别 | 数量 / 2048 | 解释 |
|---|---:|---|
| 四个未来查询足全部在材料外 | 1820，88.8672% | 本来就没有材料查询暴露 |
| 至少一个查询在材料内，但没有先前真实加载格 | 140，6.8359% | 尚未获得重访记忆机会 |
| 有先前真实加载格，但 2 ms observer 漏过短负载 | 2，0.0977% | A269 的 1.4 ms、B258 的 0.4 ms 脉冲 |
| PANN 非零 | 86，4.1992% | 实际已进入 actor 的非零记忆特征 |

后两例用真实 material events 与 2 ms 端点几何交叉验证，隐藏材料状态仅用于离线诊断。A269 的目标空间格在前一观察端点尚未被足底覆盖，下一端点已抬离；B258 同理。这是当前观察采样的真实限制，不证明 PANN 接线断开，也不授权用隐藏真值替代观察器输入。此次仅加曝光/短脉冲诊断，不同时改变 658 维输入或训练 PANN。

actor 接收的是固定物理尺度归一化后的向量，没有 RunningMeanStd/LayerNorm；GRF 历史特征峰为 A 86.525、B 56.356，PANN 峰约 .0443/.0980。冲击特征尺度值得后续跟踪，但没有读取行为权重做前向重放，不能据此断言 Tanh 已饱和或它就是失稳首因。本轮保持原输入与奖励，避免和控制修订混杂。

### 4.2 高速触地会在控制器获得新机会前越过有效域

TM02 当前厚度 0.05 m、压缩有效域 .02，在单轴压入情形对应约 1 mm。该约束是本任务已冻结的材料有效域；不是允许随意放宽的成功标准，也不是材料物理破裂的直接证据。

A260 首 load 的 material-state 事件时间为14.3636s，physical终止为14.3648s，报告差为1.2ms；对应首载位移boundary为14.3634s，所以压入约.126→1.019mm的同类boundary间隔是1.4ms。足在触地前已经位于材料内部。加载前向下速度达到约 .842 m/s。此例不支持“刚跨软硬边缘时携带已有穿透切换”的解释；2 Hz actor 上次决策在 14.0 s，下次应为 14.5 s，来不及在该次 load 后响应。需要的是触地之前的速度/支撑预测和可执行动作。

| 稠密原件 | 最后一次相关触地速度 | 压入变化 | 边界时钟间隔 | 最后实际法向力 | 被拒绝的越域候选力 |
|---|---:|---:|---:|---:|---:|
| A254 / FR | −1.959 m/s | .377→1.028 mm | .4 ms | 897.90 N | 1251.10 N |
| B256 / FL | −1.410 m/s | .167→1.007 mm | .8 ms | 1054.76 N | 1220.65 N |

两份原件共 183489 条物理施力样本与软足材料输出逐元素一致；各有一条额外 material 越域候选没有进入物理积分。material_state、边界求解和实际物理终点有 0.2 ms 的记录语义差，应按 schema 对齐；不能把被拒候选力当成已经施加到机器人上的力。全部14份原始失败记录的最大应变都精确等于该加载足的−bottom_z/.05，Newton最大残差≤1e−8；B268虽峰值qdot来自新小格，但整足压入1.04185mm，并非假越域。现有证据不支持“失败多是 Newton 无法收敛”的解释。

### 4.3 B272：存活与通行必须分开

B272 在 9.966 s 出现首次大反馈和目标裁剪，10–20 s 经历大量饱和，roll 一度到 1.16472 rad、base z 降到 .15793 m。随后进入异常局部平衡：20→60 s 的 x 从 .288839 仅到 .291531 m，y≈−.348 m；右后腿抬高，RR_thigh / RR_calf 目标几乎一直在上界。

后 40 s 仍有 2000 次规划、active reference、3 m endpoint、cap均值 .197493 m/s、命令vx均值 .195894 m/s和非零 line-search 步长；候选 trunk 可行、代价稳定，PD 饱和又变少。故它既不是 planner 停止，也不是 PPO 选零速，而是原控制目标/有限视野下能停留的非通行状态。W40 必须同时看合法进展、载荷/支撑、姿态、饱和和恢复，不能把“更少大 K”单独当成稳定结论。

## 5. r29 的启动前数学更正

完整推导、反例与实现合同见 [r29-math-audit.md](r29-math-audit.md)；纯代数复算为 [r29-algebra-examples.py](r29-algebra-examples.py)。

旧 BoxQP 在固定目标界上的 K 行会为零；一般不能写成全维 `K=-H^-1 C`。上一版在已选 `u_ff` 周围只加 `s C δx` 的目标，丢失了原活跃集乘子与前馈/line-search 关系。因此，即使新增 PD 界没有激活，也可能改变旧控制器；不应作为默认实现。

此次明确采用策略投影：

```text
c(x,t) = 原策略的 target_clip(u_ff(t) + s K_old(t) delta_x)
l(x)   = max(original_target_min, q + (tau_min + 5 qdot)/60)
h(x)   = min(original_target_max, q + (tau_max + 5 qdot)/60)
u*     = argmin_{l(x)<=u<=h(x)} 0.5 (u-c)' H(t) (u-c)
```

H 为产生策略时的正定、非对角正则化 Q_uu 度量。原输出 c 已在新域内时严格返回 c，因此保留原 BoxQP 活跃集、alpha、目标 clip 与插值行为。只有新增 PD 界实际介入时才重分配轴间修正。它不是完整状态约束 DDP，也不在本轮半改 backward/value 递推。对角投影的瞬时扭矩可能与旧硬 clip 等价，不能伪装成上述非对角重分配。

必须一并落实：候选/nominal 搜索/部署共用纯求值路径；action candidate 的 s 显式为 1，nominal-feedback 才取搜索 s；H 随 policy 一起分配/复制/重采样，首次求解前身份明确；原 BoxQP 不支持等界，固定轴需消元并保留交叉线性项；区间空明确最小违反恢复；不能用返回 nfree≥0 代替 KKT 检查；旧 predicted decrease 不再适用时标 unavailable，不能把 surprise=0 送进原分支无限升正则化。完整细节以数学审查及主计划为准。

新增记录还区分double投影解与float32 RPC后实际持有值：同状态下序列化造成的微小PD边界越出单列，QP KKT按double解判断，真实饱和仍按实际目标保留。后续十子步因状态演化的饱和不能归给舍入；不改RPC或物理阈值。

这仍是有证据支持的下一项控制实验，不是已证明有效的修复。两 E30 实际状态的目标范围与瞬时 PD 域都有非空交集；空区间是需要正确处理的实现边界，并非本次已观察的首发原因。瞬时可实现也不保证十个子步、足端冲击、原材料域或闭环稳定。

## 6. GPU 并行、记录和交付更正

[gpu-replay-audit.json](gpu-replay-audit.json) 已检查两 E30 的模型、完整 96D userdata、mocap、时间、任务参数及 176 节点参考。现有原件可以建立同状态 CPU/GPU 预测测试；旧完整 policy/H/K 和 task 内部相位未保存，不能宣称精确复现旧整次 planner。重建时显式声明相同任务相位/transition 历史和初始 policy/warm start；模型二进制本身不安装自定义回调，也不自动施加预测端 MakeDifferentiable/coldstart。

第三任务继续独立并行，主线各 12 workers，GPU host 1–2 线程；各 14 已授权但不额外开启又一组大 CPU 参考线程。CPU/GPU 必须比较同一版本，r29 指此次修订后的 H 投影，不是已撤回的 Qux 公式。先同状态一步/十子步、导数和风险，再候选与整次调用；若仅完成 batch 层就交付该层，不冒称完整 GPU MPC。没有原件可恢复的字段由新 fixture 显式初始化，不能为了补旧数据重跑原 W39。

新事件记录保留完整 accepted/reference/K/H 与 live 投影活跃集、KKT 残差、插值/scale 身份；每 plan 保存一次，不重复写满 500 Hz JSON。补每次planner调用的return reason/是否更新policy/复用年龄；补记录负载边界时钟、候选失败是否已施力、PANN 查询区/历史命中与 observer 漏短脉冲、命令存在却无合法进展等诊断；不改变 actor 输入、物理终止或任务标准。

W39 上传已完成，不再要求执行者等它、重启旧队列或重跑旧 finisher。W40 延续紧凑结果先推、已请求原件后台传；大 checkpoint/完整 resume/视频留 D57 由用户备份并手动删除，不自动删除服务器研究输出，不在用户电脑另做归档。最新执行入口是 [主计划](../../../../../../docs/TERRAIN_MEMORY_PLAN.md)、[W40 JSON](../../../../../../experiments/terrain_memory/s1-go2-w40-plan.json) 与 [启动提示词](../../../../../../docs/S1_W40_START_PROMPT.md)。


## 7. 保存数据复算入口

以下仅供需要复核数值时使用，**不是 W40 启动前的新门槛**。原件按 `read-coverage.json` 的 Release URL 下载到同一目录，保留资产文件名；不重跑物理、不要求下载那62份训练MJB。

```bash
python prepare_raw_index.py --raw /path/to/w39-raw
python scan_raw_assets.py --index /path/to/w39-raw/raw-index.json --output read-coverage.json
python control-analysis.py --raw /path/to/w39-raw
python control-analysis.py --raw /path/to/w39-raw --all-attempts
python control-stall-analysis.py --raw /path/to/w39-raw
python material-learning-audit.py --raw-dir /path/to/w39-raw
python gpu-replay-audit.py --index /path/to/w39-raw/raw-index.json --output gpu-replay-audit.json
python r29-algebra-examples.py
```

从本审查目录执行。读取代码只需 Python、NumPy、SciPy；不需要调用 MuJoCo、训练网络或 GPU。`prepare_raw_index.py` 只从仓库两份 evidence 清单建立原件映射，不下载、不删除。原始记录不被本审查覆盖，分析输出文件与历史原件分开。
