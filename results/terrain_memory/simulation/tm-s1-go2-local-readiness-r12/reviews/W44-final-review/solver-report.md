# W44 控制原件审查：近同名义轨迹为何安装了不同的离轨反馈

审查日期：2026-11-12。源码基点：`69d5a6ff2aa9803fddc8ac3a78a906008ea526c6`。范围：两流最后一次固定 update57 E30，以及其首段事件的完整候选、安装策略、动作查询和物理子步。没有重新运行控制器、物理回合或训练。

## 1. 结论及证据边界

**最有直接证据、也最适合作为下一轮单一修正的机制，是完整策略的接纳与保留。** A、B 都出现：nominal feedback search 选出的反馈倍率很小或为零；action search 的 alpha=0 候选把 reference 移到该 nominal 的 accepted trajectory，却使用新 Riccati K/H，并把反馈倍率设为 1。两条名义 rollout 几乎相同，极小的成本差通过严格 `<` 接纳了很不同的离轨策略。真实状态一旦偏离 reference，实际反馈和 H 投影产生大动作。

这不是已经证明的“未评分 K 偷换”。两个关键事件的完整候选确实进行了 native rollout；候选到安装的 13 类数组逐元素完全一致。缺口在于：沿 reference 的近零误差没有检验新 K 的离轨行为，而数值接近的名义分数足以更换整个反馈对象。

目前能证明这条链发生在两流第一次大幅反馈/投影事件中，不能证明它解释了之后每次接触异常、所有跌倒，或修复后必然完成 E30。B 在该事件之前已经出现 2.9282 s 的 hold 内力矩饱和；两流接触线性化和 plant/predictor 数值差异仍然存在。**下一轮应检验这个明确的接纳机制，不把它写成全部失稳的唯一已证原因。**

## 2. 原件、身份与复算范围

主轨迹为：

| 流 | 最后 E30 attempt | 轨迹节点 | planner calls | 结果 |
|---|---|---:|---:|---|
| A | `job-20261110T143358Z-w44-a-eval-PANN_PPO_S5-s1-E30` | 5,968 | 617 | 11.934 s body_fall |
| B | `job-20261110T143400Z-w44-b-eval-PANN_PPO_S5-s2-E30` | 6,372 | 658 | 12.742 s body_fall |

这 4 份 trajectory/planner-calls 原件共 **14,015,068 字节**，此前已下载并核对远端 SHA256。本次另外精确选择 **21 份控制原件，共 194,211,521 字节（约 185.21 MiB）**：两流完整 feedback-actions、planner-records、action-execution、physical-000、生成模型 XML；7 份完整 accepted/reference/gain/H 计划；A call163、B call173、B call158 各自全部 20 候选；B 的事件索引。21 份均核对 GitHub Release 资产名、大小、远端 SHA256 与实际下载内容一致。最后增加 B call158，专门回答下文早期腾空是否也在 native nominal 中。

另有已存在的 A 事件索引（压缩后 4,342 字节）仅用于定位文件，不是下面数组计算的输入，也不计入上述 21 份重新校验范围。这里没有声称逐字节重读 156.36 GiB 全部 Release。

逐文件 source_path、URL、Release asset ID、压缩/原始大小和哈希见 [control-originals.json](control-originals.json)。全部定量结果及逐次 alpha0 事件见 [control-summary.json](control-summary.json)。[recompute_control.py](recompute_control.py) 仅使用 Python/NumPy 对现有数组做计算，先核验 25 份输入的本地 SHA256，不导入 MuJoCo、不重新求解、不联网：

```bash
python recompute_control.py \
  --originals-dir /path/to/21-control-originals \
  --raw-dir /path/to/4-trajectory-and-call-originals \
  --output /path/to/recomputed-control-summary.json
```

使用 `control-originals.json` 的 `local_name` 保存输入文件即可。gzip 文件保持压缩；不要把 `.jsonl.gz` 当作已经解压的 `.jsonl`。

动作诊断必须按 `query_id` 与 action-execution 中的 `applied_hold` 连接。如此得到 A 5,967、B 6,371 个真正施加的 2 ms hold；记录的 `u_held` 与轨迹 `q_target` 全部精确一致。单看所有 RPC query 或 raw Kδx 会把未施加的诊断误当成动作。

## 3. 两个关键接纳事件

| 项目 | A call163 / plan156 | B call173 / plan161 |
|---|---:|---:|
| 安装时刻 | 2.84 s | 3.04 s |
| 最佳 nominal 候选编号 | 009 | 004 |
| 被选 alpha 候选编号 | 019 | 019 |
| alpha | 0 | 0 |
| nominal feedback scale | 0 | 0.013335214321633242 |
| 安装 feedback scale | 1 | 1 |
| nominal native J | 0.13569430369560503 | 0.6186566767713665 |
| selected native J | 0.13569430369559699 | 0.618656676769194 |
| J 改善 | 8.049116928532385e−15 | 2.1724844145865063e−12 |
| nominal / selected trunk violation | 0 / 0 | 0 / 0 |
| 最好正 alpha 候选 J | 0.15211369857519758 | 0.6383769704334482 |
| 两 accepted states 最大差 | 4.1542319584308984e−10 | 4.444193568708954e−9 |
| 两 accepted actions 最大差 | 9.049173910202768e−11 | 1.9032362352788823e−9 |
| 两 K 数组最大绝对差 | 113057.75430996447 | 234799.86915993787 |
| 两 H 数组最大绝对差 | 367.87896933370945 | 94.24058211371435 |

这两个 archive 中，selected 的 reference states/actions/times/residual **恰等于** best nominal 的 accepted 对应数组，差值全部为零；所有正 alpha 候选分数都比该 nominal 差。selected 与真正安装的 accepted/reference 的 8 类数组，加 K、H、base_Qu、base_Qxu、base_active，共 13 类数组也全部差值为零。

因此，“alpha=0”只说明新前馈步为零，不等于完整策略没有变化。reference、K、H、反馈倍率和活动约束数据可以同时变化。新 reference 本身是一次已经积分出来的轨迹，从相同初态再 rollout 时，沿轨迹误差非常小，新的大 K 在这个 nominal score 中几乎没有被激发。这个事实解释了为什么“完整评分身份正确”仍不足以授权数值近似平手时更换离轨反馈。

### 3.1 A：从微小预测偏差到第一次大反馈

plan156 的起点状态与真实状态完全一致；起点实际 target 与 accepted target 最大差约 6e−8 rad，来自动作传输精度。后续同一 plan 的关键节点如下。位置偏差列为 saved accepted qpos 与实际 qpos 的最大分量差，速度偏差为 qvel 最大分量差；包含平移和关节量，不能把整列统一当成弧度。

| 时刻 | 位置最大差 | 速度最大差 | 实际 scale·Kδx 最大值 | H 投影修正最大值 |
|---|---:|---:|---:|---:|
| 2.840 | 0 | 0 | 近零 | 0 |
| 2.842 | 约 2.07e−5 | 约 0.01508 | 约 0.00218 rad | 0 |
| 2.848 | 约 4.86e−5 | 约 0.08451 | 约 0.05782 rad | 0 |
| 2.850 | 约 1.99e−4 | 约 0.08176 | 约 0.07475 rad | 0 |
| **2.852** | **约 6.17e−4** | **约 0.47701** | **0.7875087603 rad** | **0.5054608813 rad** |

2.852 s 的最终 held target 相对当前 nominal 最大偏差 0.5427596569 rad，较前一个实际 target 最大跳变 0.5255513266 rad。在**同一个真实状态**，把该节点 nominal target 直接放入 PD，其力矩并未越界。因此，这次大修正不是当时 nominal 已经超力矩而不得不做的恢复；来源在反馈中心和 H 度量投影。

首个物理 0.2 ms 子步的 hold 内超力矩发生在 **2.8532 s**，随后实际执行器按原 forcerange 正确裁切。到 2.860 s，accepted 与实际的关节位置最大差约 0.0110 rad、速度最大差约 1.821 rad/s。

对 2.852 s 使用记录的插值节点/比例，将保存的 K 乘以同一条日志的 36 维 state_delta，可复得 raw Kδx，最大差 2.78e−17。最大 FL thigh 修正有明显的多项大数相消：例如 FL hip 位置差约 0.0006167 乘约 −1014.46 给出 −0.6256；该关节速度差约 0.4770 乘约 −0.9498 给出 −0.4531；机体旋转差约 −0.0004140 乘约 −1011.73 又给出 +0.4189。这里支持局部反馈对微小离轨敏感，**没有发现 K 符号或状态顺序装错的证据**。

### 3.2 B：首个大冲击在 3.044 s，早于 raw Kδx 超过 0.5 的节点

不能把 2.664 s 的 raw Kδx≈1.23014 rad 当作实际冲击：它属于 plan143，feedback scale=0，未施加该项。

plan161 在 3.04 s 安装后，起点状态同样精确一致。第一段 2 ms 后，位置最大差约 9.67e−6、速度最大差约 0.01036，实际反馈约 0.004853 rad。**3.044 s 已出现真正的大反馈/投影动作**：

- 实际日志 state_delta 的位置/速度部分最大值为 1.067107606e−4、0.1865988920；saved accepted 与实际 qpos/qvel 的对应差为 1.065573495e−4、0.1866199334。两者小差来自查询时刻插值，不可混称同一量。
- 有效反馈最大值 **0.4996758023 rad**，H 投影修正 **0.9076175298 rad**。
- held target 相对 accepted target 最大差 **0.9107163596 rad**，相对前一个实际 target 跳变 **0.8464540243 rad**。
- 在这一真实状态，nominal target 的 PD 力矩仍完全在界内。
- 3.046 s，速度预测差已达约 2.765 rad/s；3.052 s，有效反馈进一步达 **5.1440307956 rad**，3.054 s 达约 8.8300 rad。

H 投影在相关关节之间耦合，某分量的投影改变量大于原 raw feedback 最大值并不自动表示 QP 解错。此处 QP 仍满足所记录的一阶最优性和原瞬时 bounds；问题是哪个完整反馈中心/度量被接纳并施加。

B 在 **2.9282 s** 已经出现较早的 hold 内饱和，先于 plan161。因此这条共享机制是明确的早期放大事件，不应擦掉 B 已有的其他控制/接触问题。

## 4. 全部最终 E30 调用中的 alpha0 规模

计数按 `candidate_source` 为 `alpha_*` 且确实新安装区分，排除 nominal/incumbent 策略继承的旧 `chosen_alpha` 字段。

| 项目 | A | B |
|---|---:|---:|
| 所有 planner calls | 617 | 658 |
| 字段 chosen_alpha=0 的调用（含其他来源） | 126 | 108 |
| **新 alpha0 完整策略安装** | **123** | **90** |
| 其中 nominal scale<1，安装改为1 | 111 | 88 |
| 其中 nominal scale=0，安装改为1 | 36 | 27 |
| 与 nominal trunk violation 相同 | 123 | 90 |
| J 改善 ≤1e−12 的新 alpha0 安装 | 75 | 29 |
| J 改善 ≤1e−10 的新 alpha0 安装 | 101 | 42 |
| J 改善中位数 | 2.2328e−13 | 3.6885e−10 |
| J 改善最大值 | 0.0002669584 | 0.0174470867 |
| 实际有效反馈 >0.5 rad 的 hold 数 | 183 | 201 |
| 其中由新 alpha0 策略施加 | 29 | 24 |

表中 1e−12、1e−10 和 0.5 rad **只是透明的描述分箱，不是推荐的接纳阈值、安全阈值或失稳定义**。成本改善还记录了相对值和 nominal 处 ULP 数，见 JSON 的 `zero_alpha`。

不能把所有 alpha0 改善都归为直接求和舍入：部分差异明显更大，native 接触、reference 重新居中、投影和反馈传播都可能放大初始数值差。下一版本需要定义 no-step 的完整策略身份，并从实际累计误差/同输入重评分确定分数分辨率，不能从上述分箱倒推出一个拍脑袋容差。

## 5. 已检查并排除的简单解释

### 5.1 N 确实上线，当前 FD 为单侧

两份最终 planner-calls 全部记录 `r33_mechanism=N`、`derivative_passes=2`、`P_candidate_slots=0`。不是只改了计划名，或线上仍采用 P proposal。现有 runtime 声明 `fd_tolerance=1e-6, fd_mode=0`，即 forward FD；没有发现当前线上使用 centered FD 的证据。

N 的模型导数副本由 `mj_copyModel` 建立，跳过 `MakeDifferentiable`；每个 native 子步用 `mj_copyData` 复制到每 worker 的独立 derivative data。七份保存计划内 `r31_native_soft_substeps` 的 native successor 与 derivative-copy successor 全部逐元素差 0。历史字段名中的 `soft` 不能据此解释为当前 N 仍采用软接触导数。源码位置：r33.patch 48–50、68、426–435；r31.patch 137–180。

K 的 state tangent 顺序为位置切空间 18 维与速度状态 18 维，即 base xyz、旋转切向量、12 关节，再 base 线/角速度和 12 关节速度。实际数组后 18 项是 **qvel**；复算按 `state_delta[18:]` 的速度部分处理。PD 仍是 `60*(q_target-q)-5*qdot`，目标速度零；这点本身不是相对原作者接口的失配。

### 5.2 hold 和 force clamp 在预测端已经存在

native accepted rollout 以同一 q_target 保持 10 个 0.2 ms 子步；FD 把这 10 次转移复合到 2 ms；物理端也用相同 held-target 合同。七份 selected plan 的原始 prefix 可以重建 `raw_PD` 和实际 actuator force，均符合原 `forcerange` 裁切，首个 hold 的最大残差为浮点量级（约 2e−14 Nm）。B161 的保存 native prefix 自身还出现约 **0.4662 Nm** 的 hold 内 raw PD 超界，执行器同样正确裁切。因此不能说“预测遗漏了整个 hold 的力矩裁切”。

物理端全子步结果：

| 项目 | A | B |
|---|---:|---:|
| 首次 raw PD 超 forcerange（判定余量1e−4 Nm） | 2.8532 s | 2.9282 s |
| 超界子步比例 | 约 2.398% | 约 2.031% |
| 最大 raw PD 超界幅度 | 67.0512 Nm | 51.7010 Nm |

这些是裁切前请求超界幅度，不是声称执行器实际输出超出了硬限幅。只看 500 Hz endpoint 会遗漏一部分 hold 内行为；但“瞬时 QP 不保证整个 hold 内始终不裁切”已是当前机制的边界，不能单独包装成新的根治发现。

### 5.3 没有 QP 求解失败或明显序列化故障

两流实际施加 hold 中，QP failed=0、restoration=0、primal residual最大为0；scaled KKT residual 最大 A≈6.90e−9、B≈6.26e−9，均在记录的 1e−7 口径内。动作序列化对应最大力矩差 A≈1.43e−5、B≈7.15e−6 Nm，不足以解释这次半弧度量级的 target 变化。

### 5.4 plant 与 predictor 仍不是同一个数值系统

两个生成 plant XML 均为 elliptic cone、impratio=100、timestep=.0002；plant option 没有 integrator override，使用 Euler。planner 配置保留 `agent_integrator=3`（implicitfast）及 prediction coldstart；其 warmstart disable 也随 derivative model copy 继承。native/derivative 的零扰动一致，不等于 plant/native 一致。

A156、B161 的首个物理子步与预测 raw PD 差约 3e−6 Nm，与目标序列化量级相符；同一个 held target 下，到第二个 0.2 ms 子步，差已经分别约 0.0140、0.0213 Nm，第一段 2 ms 后出现前述非零速度预测差。差异在后续大反馈前就存在，能作为触发离轨的具体观察；**没有通过反事实分别分离 Euler/implicitfast、warmstart、接触模式和传输的份额**，因此不能直接断言其中某一个独自造成最终跌倒。

完整历史 plant `qacc_warmstart`/solver 私有状态并未随所有快照完整保存。虽 `mj_copyData` 会复制当前 derivative 所需 data，候选状态序列本身不是全部历史 plant state。新旧方案的共同上下文比较必须明确缺失字段，不能号称历史精确 replay。

### 5.5 首次异常不需要材料加载才能发生

两次最后 E30 都没有实际材料加载；上述早期事件仍在刚地部分，不是已证 PANN 预测错误或材料历史恢复失败。候选内部没有完整 TM02 历史演化依然是长期缺口，但不能拿它解释本次首先发生的刚地反馈事件。

同样，单个“全足 contact=false”节点不是跌倒证明。**当前冻结声明和实际 readback 都是 Manual Trot、2 Hz、duty=.45、Amplitude=.005 m**。依据是两份当前最后 E30 的 resolved 中 `r12_configuration.task.move_and_turn_settings`、`r11_contact_timing`，以及 `applied_settings[1].declared_parameters` 与 `task_parameters` 一致；上述运动段实际 planner call 的 task_parameters 和私有 phase_velocity=4π 也吻合。见 [A 当前 resolved](../../job-20261110T143358Z-w44-a-eval-PANN_PPO_S5-s1-E30-resolved.json)、[B 当前 resolved](../../job-20261110T143400Z-w44-b-eval-PANN_PPO_S5-s2-E30-resolved.json)。这不是把作者默认值当成实际值，也不是 R11 旧 dev07 的“声明1Hz/.65却被默认值覆盖”事件再现。

2 Hz/.45 对角步态理论上允许约25 ms的双对摆动重叠，但这不足以解释本次较早长腾空。500 Hz记录首先显示A2.666–2.720、B2.750–2.804全无足接触，均54 ms；5000 Hz原件进一步定位实际无足接触且GRF全0的长段为 **A2.6646–2.7200（55.4 ms）**、**B2.7484–2.8032（54.8 ms）**。取这两段内首个500 Hz样本，用当时私有 `phase_start + (time-phase_start_time)*phase_velocity` 重建，相位分别约2.07345/3.12903 rad，都是 **FL/RR处于步态定义的stance段**，而非理论双对swing重叠；当时实际base vz分别约+.239851/+.212037 m/s。步态定义的stance不是当前控制器的实际接触硬约束，不能把这条偏离直接当成协议违规或最终跌倒证明。

**B的这段早期腾空已明确存在于native nominal。** call158、输出plan146、origin2.74的完整候选显示：best nominal candidate006（feedback scale=.0749894209）与alpha0 selected candidate019（scale=1）都预测2.750–2.800 native四足contact/Fz全0。在2.750时，predicted basez=.2927394648、vz=.212016968，实际分别为.2927396574、.212036680。这里的初期预测执行吻合很好；不能把这个更早支撑行为归因于3.044之后的大反馈失配。该call也有alpha0的微小接纳，但即使保留其best nominal，也保留了这段名义腾空，因而接纳修正并不承诺消除它。native foot rows记录有一个0.2 ms子步的几何/力时标语义，后续物理每20 ms也会重规划；不能把整段开放预测当成固定policy的历史闭环重演。

A在第一长腾空附近的call154候选archive未出现在该attempt的精确文件清单中；其候选archive从call159/2.76开始、完整plan从151/2.74开始，均晚于第一长腾空。本次不以这些事后文件冒充A该时刻的预测接触证据。此覆盖限制保留在复算输出。

因此，早期名义支撑/腾空行为是接纳修正之外的明确未解项。本报告不把首次全无足接触时间直接定义为第一失稳，也不把高频 contact bool 切换算作有效步态。

## 6. W45 的最小合理修正与有限反事实

建议采用计划中的**完整策略接纳与平手保留**，保留 native N、尺度 D、约束 Riccati、原成本/步态/物理/任务。具体契约应包含：

1. 对当前 epoch 重新 native 评分的 incumbent/nominal，维护完整基线身份。不能用旧 epoch 分数，不能只保留 actions 而重置 K/H/scale。
2. alpha=0 默认是 no-step/完整基线保留。新的纯反馈策略若要成为独立改进对象，需要单独的离轨依据；本轮不靠名义微小下降隐式授权。
3. 数值平手预算由 FP64 成本累计量级/项数和同输入重评分误差给出。原 v 优先级和物理硬判据不放宽；不能把一个绝对1e−6当成经验成功开关。
4. 规则必须同时用于 **nominal/BestRollout 的基线选择、r33 archive/R33Consider** 与继承 **r32 Iteration 的 descent、accepted_native_descent、重试停止/regularization 状态**。只修最终 archive 会留下“最后保留了基线，内部却记作成功并停止”的不一致；不同选择层也不能把两套 baseline 身份混用。
5. 正 alpha 仍允许真正可分辨的 native 改进和正常 TV 反馈。这个修正不是全局关闭 K，也不是对 K 或动作加未经验证的安全限幅。

主要代码落点：[r33.patch](../../../../../../experiments/terrain_memory/s1-go2-r11-S5-r33.patch) 的 R33Consider（191–205）、nominal archive（378–382）、ActionRollouts 里强制 feedback_scaling=1（516–517）；[r32.patch](../../../../../../experiments/terrain_memory/s1-go2-r11-S5-r32.patch) 的 strict native descent（359–365）和 accepted 状态记录（402–403）。当前完整候选身份机制本身应保留。

共同上下文使用四份现有完整输出：A call158/plan151/2.74 s、A call163/plan156/2.84 s、B call153/plan143/2.64 s、B call173/plan161/3.04 s。它们分别覆盖事前参考、A异常、raw未施加反例、B异常。先用已有两份完整候选 archive 对选择规则做确定性复核，再对旧/新规则使用同一重建上下文及有限扰动诊断。观察是否消除了 alpha0 的完整反馈重置、以及离轨时的真实动作/投影放大；不能只看 nominal J。

本次没有执行这种反事实，也没有建立“改后已稳定”的结果。固定各流自己的 update57 Actor/normalization、暂不新增 PPO 样本，是隔离这一个控制变化的合理安排。真正效果仍需原任务 E30→条件 E60/E300/E600；若失败，保留剩余最早异常，再决定下一主机制。B 的早先饱和、接触/积分差异，以及正 alpha 下的大 K 仍可能使此修正不足。
