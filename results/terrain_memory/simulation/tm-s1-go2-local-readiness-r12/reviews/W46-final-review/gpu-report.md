# W47 前置审查：W46 GPU 一致性支线与并行资源

执行规则以[最终W47计划](../../../../../../docs/S1_W47_STABILITY_PLAN.md)为准；本独立报告的建议由主审查整合，不能另开一套并列实验。

审查基点：`15ee620b8a909246a12baf999c136c2be299655b`；日期：2026-11-13。

## 结论

W46 GPU 支线完成了有价值的一步：真正执行了保存策略的 reference、K、scale 和 H，并调用原 native H-QP，已经超过 W45 的 reference-tracking proxy。但是它仍没有得到可以替代 native 求解器的模型一致性、完整成本排序或在线加速。现有结果支持“同一初态上的动作代数基本一致，进入两种物理引擎后迅速分离，随后闭环命令差被放大”。它没有把这条链条中的版本、模型编译、FP32、初态/控制量 cast 和接触求解因素分别识别出来，也没有证明它们是 W46 A/B 原任务最终跌倒的原因。

本次明确发现四项需要修正的实验记录/协议问题：

1. `raw_center_max_error_rad` 在 native 投影原位修改数组以后才计算，实际上重复测了投影后命令差，不能当独立的投影前代数核验。
2. `first_divergence` 把某个 hold 的末端状态与该 hold 起点决定的命令放在同一时间标签下。最大 1.50136rad 命令差发生在 **+10ms 决策、+10～12ms 保持区间**，不能写成 +12ms 才开始发生。
3. 20ms 后是否继续 350ms 的代码门槛只有 trunk 布尔判断是否不同；计划要求的“短窗差异可解释、不改变原约束判断”没有完整实现。24 个 trunk 判断全为零，这并不能证明跨引擎约束分类已经可靠。
4. `warmup_world_steps=48` 把一次 CUDA 图 capture 记录也算成物理执行。按标准 Warp/CUDA 语义，该源码实际执行24预热world-step，另24仅被capture记录；主循环42000的计数不受影响。

此外，已有 8 份策略二进制的逐数组复算发现：**B155 与 B158 各自的 baseline/selected 执行策略完全相同，实际是 6 个不同策略、18 个不同策略×初态组合，被执行为 24 个 scenario。** 42000 world-step 是实际工作量，不能理解为 24 个独立策略样本。该重复可以在未来有限核验中按内容身份去重。

下一轮优先让 CPU 双任务解决实际控制稳定性；GPU 第三任务只承担下面列出的有限、可判定的问题。没有必要为了显存占用再开一次同样的 24-world/350ms 批次。W46 的有限 GPU 任务在 A/B 原任务启动前约 406.620s 已经结束，已有事实也不支持“整轮一直三任务并行”。

## 1. 本次究竟核对了什么

本报告使用三类证据，刻意区分它们的证明力。

| 证据 | 本次处理 | 能支持什么 |
|---|---|---|
| 已提交源码、计划、`w46/gpu.json`、`phase.json`、A/B 紧凑结果及 resolved | 阅读、逐文件 SHA256、算术复算 | 执行路径、门槛、工作量和已报告时间的口径 |
| 工作区此前已经存在的 19 个原件：8 policy bin、4 input.json、4 input-state.npz、3 perturbation.npz | 13,368,310 bytes 全部 size/SHA256 对上 Release 来源清单；纯 NumPy 复算 | 完整策略数组身份、20 个可恢复初态的 raw center 与 PD/ctrl 边界余量 |
| GPU `paired-states-actions.npz`、`action-algebra.npz` | **本次没有获得** | 对应轨迹差、active/KKT 和 native bridge 对比数值只能引用已发布 compact，不能称本次独立数组复算 |

对 Release 原件的落盘读取与随后纯内存读取均被自动审批拒绝；后者给出的理由仍是可能下载并处理用户不希望再下载/归档的原件。收到拒绝后停止新原件获取，未换工具或 URL 重试。只继续使用原先已存在的数据。缺少的是 B158 的 perturbation 与 GPU 配对数组，不能用来源清单和资产 digest 核验冒充数组内容复算。

可重算文件：[recompute_w46_gpu_offline.py](recompute_w46_gpu_offline.py)。它没有网络导入或网络回退分支，不调用 MuJoCo、Warp、native library，不产生新 physics/PPO。结果为 [gpu-offline-summary.json](gpu-offline-summary.json)，包括 17 个本地源码/compact 文件 SHA 和 19 个原件 SHA/来源 member。运行方式：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python recompute_w46_gpu_offline.py \
  --repo /path/to/MCP-RL --existing /path/to/existing-w46-policy-layout \
  --output gpu-offline-summary.json
```

服务器原位复算无需建立上述审查布局：可以使用`--policy-root /root/autodl-tmp/MCP-RL-w42/outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46-mechanisms`，脚本只枚举四label各五个白名单文件，必需文件缺失明确报错，perturbation按真实存在状态核SHA并统计。不会递归吞入其它文件、要求复制原件或硬凑19件/20初态。AGENTS上下文哈希固定读取审查基点Git blob，计算输入记录实际文件SHA。本报告的19件/20初态是本轮实际范围；服务器若有完整20件，可输出24个初态。该脚本没有实现paired trajectory或1944calls全程聚合，它们属于下轮Task C。

这些是审查派生文件；原件不再另复制进报告或 Git。源码证据入口：[GPU 实现](https://github.com/kaiwen123-yang/MCP-RL/blob/15ee620b8a909246a12baf999c136c2be299655b/scripts/s1_go2_w46_gpu.py)、[native bridge](https://github.com/kaiwen123-yang/MCP-RL/blob/15ee620b8a909246a12baf999c136c2be299655b/scripts/s1_go2_w46_native_bridge.cc)、[GPU compact](https://github.com/kaiwen123-yang/MCP-RL/blob/15ee620b8a909246a12baf999c136c2be299655b/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/gpu.json)、[W46 计划](https://github.com/kaiwen123-yang/MCP-RL/blob/15ee620b8a909246a12baf999c136c2be299655b/docs/S1_W46_STABILITY_PLAN.md)。

## 2. 是否真的是同一策略、模型和执行条件

### 2.1 策略范围已经比 W45 明确，但 native 对照仍有范围

**本节全部策略都是 W45 保存的 `original-baseline.bin` / `original-policy.bin`，不是 r35 finite 比较新产生的 constant/load selected。** 文件虽放在 `w46-mechanisms`，GPU代码特意读取 original 前缀的旧策略；因此下表中的 A148/A153 selected scale=1 不能与控制主线 r35 selected 的 scale混用，也不能把这组GPU轨迹当成已部署r35 constant的GPU验证。

GPU Python 程序从原 binary 读取 reference states/actions/times、K、H、scale，要求 representation=1。GPU kernel 用 FP64 计算 wxyz 四元数切空间和 Kδ，再按原 ctrlrange 裁切；两条路径都调用同一 `R29Project`。H-QP 是原 native host 执行，不能写成全部在 GPU 上执行。

`same_state_action` 的 CPU 一侧是 Python `raw_policy` 加 native H-QP；没有直接调用完整 native `Policy::Action`。另有后处理，把四个 selected/nominal 的 native bridge rollout 与 original-agent rollout 对比：20ms 状态分量最大差依次为 6.25e-14、8.77e-11、2.57e-13、1.70e-13；350ms 最坏为 A153 的 2.32e-6。它支持这四个名义场景与既有 native 链接近，不能扩成 24 个场景所有时刻与原生 policy 二进制严格等价。本次未获得这组轨迹数组，以上是 compact 已报告值。

逐原件数组比较得到：

| Context | baseline scale | selected scale | baseline 与 selected 的执行数组 |
|---|---:|---:|---|
| A148 | .001 | 1 | 不同 |
| A153 | .001 | 1 | 不同 |
| B155 | .001 | .001 | states/actions/times/K/H 全相同 |
| B158 | .0133352143 | .0133352143 | states/actions/times/K/H 全相同 |

B155/B158 的 residual、base_Qu、base_Qxu、base_active 也完全相同；文件 SHA 不同并不意味着执行策略不同。去重依据应是执行所需完整数组与 scale、初态/时间/模型身份，不能只比较文件名、header plan id 或整文件 SHA。

### 2.2 能确认的模型条件

四个输入记录指向同一 original model SHA。程序从 original XML 去掉 sensor，将 predictor 的 dt 设为 .0002、integrator 设为 implicitfast；Python 与 native bridge 都禁用 warmstart、将 `jnt_solimp[:,0]` 设为 0。两侧每个 2ms hold 都推进 10 个 .2ms 子步。它保留 ctrlrange、PD60/5、每腿力矩界 [23.7,23.7,45.43]Nm，使用固定 userdata 的材料 proxy/contact callback。

W46 **只导入** W41 的接触/材料 kernel；未调用旧 `WarpPredictor` 类里把 geom/pair solimp 置零的初始化。不能把旧 G5 配置问题直接套到 W46。W46 源码对四份 Python 模型的部分 compiled arrays 作相等检查，但没有逐数组比较 native3.2.3 与 MuJoCo4 编译后的整个模型。

### 2.3 尚未隔离的差别

| 层 | native 对照 | GPU 对照 | 当前证明边界 |
|---|---|---|---|
| 动力学实现 | MuJoCo3.2.3、FP64 | MuJoCo4/MJWarp、FP32 | 版本与精度同时改变，不能单归因 FP32 |
| policy 外部时间 | FP64 origin+2ms×k | 同一 FP64 policy 时间 | 不等于设备内部动态时间字段逐 bit 相同 |
| 初始状态 | 从同一 FP64 reconstructed state reset | 写入 FP32 buffer | 保存的 initial 数组在 cast 前；它不证明实际设备初态逐 bit 相同 |
| held control | FP64 native ctrl | 显式 cast 为 FP32 ctrl | 动作代数一致与物理输入逐 bit 一致是两回事 |
| 原 plant | 原 E30 为 Euler、真实观测/材料链 | 本 GPU 诊断没有原 plant | 不能把此诊断写成 original-task 真值或完整历史 replay |
| 材料历史 | 固定 proxy userdata | 固定 proxy userdata | 没有完整候选内材料历史推进 |
| 初始私有状态 | reset/coldstart | reset/coldstart | 原始 plant qacc_warmstart、接触 solver memory 等并未历史恢复 |

这不是声称物理配置已经发现错误；它说明实验设计目前不能唯一定位 engine 差异的原因。官方 [MJWarp 文档](https://mujoco.readthedocs.io/en/4.0.0/mjwarp/index.html)也区分批吞吐和低延迟，并说明 GPU 与 native 数值实现差别。需要对本仓库原件作分层验证，不能以官方差异说明替代自己的首处分歧证据。

## 3. raw center 记录错误及本次可完成的独立代数核验

`Bridge.project` 使用 `np.ascontiguousarray(u)`。传入的 `raw[i]`、`cpu[i]` 本来就是连续 float64 数组，这个调用可以返回同一内存；随后 C++ `R29Project(...,u)` 原位更新它。`action()` 最后才计算 `ma(raw-np.asarray(cpu))`，所以该数已经经过投影。

最小修复是显式保留 `raw_before = raw.copy()` 和 `cpu_before = np.asarray(cpu).copy()`，投影时传 `u.copy()`，分别保存 raw、ctrl-clipped center、held 三个阶段的差和 active/KKT。这个诊断变量修复不应改变实际控制链。必须同时记录出现 active projection 的状态，否则全都在 box 内的起点不能验证活动面切换时的执行一致性。

本次仅用已经存在的原件完成了 20 个初态的纯 CPU 数值复算：A148/A153/B155 各 2 策略×3 扰动，加 B158 两个 nominal。使用保存的 quaternion 切空间扰动、同一插值与 K/scale，另按 GPU kernel 的逐标量 FP64 运算重写一份代数，二者 raw center 最大差为 **1.11e-16rad**。这验证代数公式，不等于重新运行 Warp 编译器。

这 20 个 ctrl-clipped center 全部严格位于原 R29 的 PD+ctrl box 内，离最近边界至少 **.323743rad**；因此 native R29 对这些初态会走 `inside` 分支，不需要改变 center。这里的 alias 测量错误没有把这 20 个初态的真实投影差掩盖掉。B158 的四个 ± 扰动场景没有完整既有输入，本次不能补证；不能将 20/24 写成 24/24。

另一个纯代数分离：20 个可恢复初态写入 FP32 再读回时，最大状态分量舍入约 **5.57e-8**；只改变这一初态精度、保持同一 FP64 policy 公式，最大 raw center 差 **1.97e-5rad**；初始 ctrl 本身 FP32 cast 最大差 **5.59e-8rad**。这些只说明初始 cast 的量级，不能由它们预测接触动力学后 10ms 的误差，更不能把 1.501rad 归咎于一个显式 cast。

投影边界公式和 `inside` 分支见 [r29 patch](https://github.com/kaiwen123-yang/MCP-RL/blob/15ee620b8a909246a12baf999c136c2be299655b/experiments/terrain_memory/s1-go2-r11-S5-r29.patch)，后续 qp1/qp2 修复没有取消该分支。20 行原始 center、边界余量、策略 SHA 与比较数值都在本次派生 JSON 中，未保存新 rollout。

## 4. 1.501rad 究竟是什么、第一处分歧在哪里

已发布 compact 显示同一初始 FP64 state 上的 held action 差 2.489e-16rad，按 P=60 换算为 1.494e-14Nm。推进第一个 2ms hold 后，状态已经出现差别；下一次反馈才产生可见的目标角差。这与“物理执行分离后，闭环反馈对不同 state 作不同命令”一致；不能据此确定究竟是接触算法、版本、精度、模型编译或哪个约束活动面首先改变。

以下状态列为源文件的跨所有场景、所有 qpos/qvel 分量最大值，存在 m、四元数无量纲、rad 或 m/s、rad/s 混合单位，不能一概标成 rad。命令列才明确是 rad。行内状态对应 hold 末端，命令对应 hold 起点。

| 命令决策相对时间 | hold 末端 | qpos 分量最大差 | qvel 分量最大差 | 该 hold 最大目标角差 rad |
|---:|---:|---:|---:|---:|
| 0ms | 2ms | 1.462e-5 | .026638 | 2.489e-16 |
| 2ms | 4ms | 2.885e-5 | .041979 | .007666 |
| 4ms | 6ms | .0001181 | .112033 | .102852 |
| 6ms | 8ms | .0002901 | .201576 | .040212 |
| 8ms | 10ms | .0017583 | 1.837714 | .746669 |
| 10ms | 12ms | .0052783 | 4.076386 | **1.501362** |
| 12ms | 14ms | .0124124 | 3.891910 | 1.058696 |
| 14ms | 16ms | .0175507 | 3.212296 | .560880 |
| 16ms | 18ms | .0209913 | 3.255677 | .365575 |
| 18ms | 20ms | .0235766 | 2.671217 | .308350 |

20ms 端点关节角最大差为 .0235766rad；**1.50136rad 是前十个 hold 的最大命令差，不是 20ms 关节姿态差**。350ms 末态混合分量最大差降为 .844949，也不代表途中差更小或已重新等价。

仅有每 2ms 末状态和 held action，不能找回第一个 .2ms 子步的接触对、接触距离、法向力、solver residual、活动面及精确误差所在关节。现有 compact 的最大值也没有 scenario/joint 索引。因此本次只能把首个可观测分歧定位到第一个 2ms hold 内；不能写出已经找到“某足某次碰撞”或“某 K 元素”是首因。

## 5. 20ms 到 350ms 的准入确实比声明弱

W46 计划要求：完整执行语义已实现、短窗差异可解释且不改变原约束判断，才续接完整 350ms。源码在 k=9 做的是 `(gpu_trunk>0)!=(native_trunk>0)`；存在任何差别就停止，否则继续。它没有把已记录的 1.501rad 命令差、.02358rad 关节角差交给“解释完成”的显式判断，也没有在这个决策点检查全部原 hard conditions。

`body_hard_condition_readback` 来自后处理；它不是当时扩窗之前的 gate。`constraint_disagreements` 在循环中主要记录 QP failure，其字段名不能被理解为已经核完完整原约束集合。

24 个短窗 native/gpu trunk 都为 0，后处理 body_fall 都为 false。它们说明这些早期场景没有触发相应布尔失效，但没有测试任何 trunk 阳性或接近真实失败边界的分类。trunk geometric risk、body_fall、完整 native cost、材料/路线资格也不是同一个判据。

因此 350ms 结果应作为“门槛不足情况下取得的有限负诊断”保留，不能补写成准入条件已经满足。报告本身明确 `full_native_cost_implemented_on_gpu=false`、`equivalent_native_ranking=false`，这两条应继续保留。问题在研究协议的执行覆盖，不是已证明它导致 A/B 跌倒。

## 6. 工作量和性能：host 部分不可忽略

### 6.1 工作量复核

实际 for-loop 为 175 holds×24 worlds×10 子步，即 GPU **42000 world-step**；native 对照另有 **42000 world-step**。20ms 使用同一运行态续接，是前 10 个 hold，已经包含在 42000 中，不能再多加 2400。

**预热计数需改口径。** 源程序先在 capture 外执行一次 `load(); step(); synchronize()`，这会推进24个world。随后 `with wp.ScopedCapture(): step()` 只是记录CUDA图；退出后调用synchronize并重新load，并没有在主循环前执行一次capture_launch。NVIDIA [Warp图内存文档](https://nvidia.github.io/warp/latest/user_guide/execution_and_performance/memory_management.html)与[维护者解释](https://github.com/NVIDIA/warp/discussions/565)都明确capture不执行CUDA操作，须launch才执行。因此按该源码和API语义，应分别记 **24 executed warmup world-step、24 capture-recorded world-step**；源compact直接写48 warmup不能当48次真实物理推进。本次未执行任何GPU程序；这个修正来自源码/API语义，不是新增实测。

每个 hold 调用两次 `action()`，一次对 GPU state、一次对 native state；每次 `action()` 都对 24 个 world 同时做 GPU center 和 Python center 的 native H-QP。于是：

- 主循环 native H-QP 调用为 175×2×24×2＝**16800 次**。
- 初始同状态核验 48 次，recenter 核验另 48 次，总计 **16896 次**。
- GPU feedback kernel 共 1+175×2＝**351 次**。
- native `w46_hold` 为 175×24＝**4200 次**，每次内部 10 子步。

第二次 `action(native)` 仍做 GPU kernel 与两侧 projection，最后主要取 CPU 输出。这是诊断对照开销；不能直接等同于未来优化后的线上实现成本，也不能把它从本次实际消耗中删掉。

### 6.2 已报告时间的算术分解

| 部分 | 时间 s | 解释 |
|---|---:|---|
| action/setup | 6.337175 | 初始化与代数/recenter 等工作混合 |
| compile/capture | 1.050583 | 预热、编译/capture 部分 |
| 175 hold 的 action mapping+host H-QP | 2.705160 | 包括两侧动作实现、CPU循环、投影与同步，未单独计纯 H-QP |
| device advance wall | 3.827242 | 1750 次批 step launch 与 synchronize 的墙钟 |
| native confirmation | 2.775936 | 24 native worlds 串行 bridge 保持推进 |
| 状态/risk download | .025926 | 显式该段计时 |
| 其余 | .246669 | 控制上传等未分项和记录开销 |
| 总计 | **16.968692** | 一次有限诊断 |

主循环这些分项合计 9.334265s；每个“24 worlds 一起推进 2ms”的 hold 平均 **53.339ms 墙钟**。其中 action mapping+host H-QP 为15.458ms、device advance 为21.870ms、native confirmation 为15.862ms。mapping+native confirmation 共占循环 **58.72%**，setup+compile 占全程 **43.54%**。

这些计时可以说明当前流程是 host 耦合诊断，但不能独立求出“纯 H-QP 吃了多少 CPU 秒”、设备利用率、独占卡的吞吐、与共卡任务的竞争损失，或未来完整 MPC 的速度。源记录没有分别记这些量。每2ms hold都需要回传状态、host projection和下一次上传，因此它也不是 resident all-device horizon。

## 7. 双 CPU 的真实负载与 W47 资源安排

W46 phase 记录 cgroup 25CPU、120GiB RAM、GPU97887MiB；当时有两项无关 GPU 任务，未被触碰。A/B 各10 workers，GPU host额度2，OMP/OpenBLAS/MKL各1；JAX启用x64且关闭client preallocation。[phase](https://github.com/kaiwen123-yang/MCP-RL/blob/15ee620b8a909246a12baf999c136c2be299655b/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/phase.json)

| W46 原任务 | A | B |
|---|---:|---:|
| 物理时长 s |26.432|11.638|
| 墙钟 s |3935.575|2425.399|
| 每物理秒所需墙钟秒 |148.894|208.403|
| planning 占 wall |81.72%|77.25%|
| startup+planning 占 wall |83.03%|79.41%|
| planning p50 ms |2069.57|2596.95|
| planning p95 ms |4849.39|5444.61|
| feedback p50 ms |1.110|1.061|
| feedback p95 ms |1.965|1.875|

计时来源：[A原任务结果](https://github.com/kaiwen123-yang/MCP-RL/blob/15ee620b8a909246a12baf999c136c2be299655b/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261112T133729Z-w46-a-eval-PANN_PPO_S5-s1-E30-attempt01.json)、[B原任务结果](https://github.com/kaiwen123-yang/MCP-RL/blob/15ee620b8a909246a12baf999c136c2be299655b/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261112T133729Z-w46-b-eval-PANN_PPO_S5-s2-E30-attempt01.json)。

planning 的 1322/582 次 logical deadline misses 是相对逻辑20ms目标的性能记录；不是后台到点 kill，也不能直接当新增实验失败次数。原任务是同步仿真，不能只看 miss 数就说真实物理时钟中每次都失控。physics 的最大单段16.17s/19.30s墙钟则需要结合记录/I/O分项才可归因，现有紧凑数据不能证明是 GPU 竞争造成。

GPU task 的 ended_unix 与两个 E30 的 started_unix 相减，GPU 提前406.620s已结束，所以这轮并没有测“两个原任务+GPU重负载”的稳定并行性能。25核和大显存不能免除 CPU planning、host synchronization、记录 I/O、共卡 JAX 的资源竞争。

W47 建议按实际 cgroup/affinity映射以下额度，先完成双 CPU 的有限稳定性工作和资源基准记录，再决定第三支线准入：

| 工作 | CPU/线程 | GPU/显存规则 |
|---|---|---|
| CPU A |10CPU额度，完整进程树亲和；Torch/BLAS/OMP各1|只运行本流既定Actor/PANN；preallocate=false，记录本流实测显存峰值|
| CPU B |独立10CPU额度，其余同A|同上，不复制超额缓存|
| 第三有限审查 |最多2CPU总host额度，含native确认|先做已有数组离线核验；确有可区分的GPU问题时，按剩余显存与A/B实测峰值准入|
| 系统/主控/发布 |约3CPU余量|保持日志/I/O余量，既有无关GPU进程不动|

已有记录没有每流显存峰值，不能凭空把96GB等分成三份或认定全卡空闲。开场读取实际 nvidia-smi/cgroup，用现有双流运行过程中记录到的峰值和可用余量约束第三任务；如果第三任务使必要native工作明显受阻，完成/停止这个有限诊断后释放资源，不调低原任务物理标准填满卡。构建 -j6 在双重负载之前做完；发布短写串行，不与两个求解任务争抢全CPU。

全部后台 `deadline=null`、`wall_deadline_enabled=false`，不设置墙钟完成时刻、不以等待时间到达杀进程。保留明确的有限问题数量、原物理任务时长、技术错误终止与单次I/O timeout。有限完成、明确失败或用户停止之后自然结束，不能将“无 wall kill”变成无限重复运行。GitHub核验交付之后不增加用户电脑归档步骤。

## 8. 第三任务下一轮最有价值的有限内容

**第一步只用服务器已有原件，零新增 physics：**修正raw/held别名和时间标签；在原位读取 `action-algebra.npz`、`paired-states-actions.npz`、四个输入/扰动与策略，逐件核SHA；恢复24场景索引和关节索引，分别计算原始center、投影前后变化、关节/base/旋转切空间误差。将有效18种策略×初态身份标清，核完整已保存350ms中的body条件；不能把数组没有记录的子步接触/active/KKT重新编造出来。compact报告修正后交付即可，不必占用GPU。

旧 `action-algebra.npz` 存的是held目标与projection info，没有另存投影前device raw。用保存state/policy按公式重算的raw center必须标为离线重算，不能冒充当时GPU kernel的原始输出。修正诊断代码也不会使历史错误字段自动变成有效测量；历史字段应明确弃用或限定，后续若执行零physics动作代数核验再记录三个阶段。

**若仍需要新执行，下一问题应当是首个2ms hold的一项分层核验，而不是再次350ms扩窗：**从现有数组选定已观测首分歧最大且身份齐全的单一场景，冻结同一初态、held命令、proxy与模型；先核 native3.2.3 和 CPU MuJoCo4 FP64 的 compiled-model/单hold差异，再核 CPU MuJoCo4 与 MJWarp FP32 的同一hold。记录每.2ms子步的接触对/距离/法向力、solver residual/计数、实际模型参数、状态与控制cast。若现有接口不能提供这些数据，就将该原因标为不可识别并结束，而不是扩大场景数。

三引擎各一次2ms、各10子步，**总额最多30个真实physics子步**，不加第二场景、第二hold或额外真实step预热；capture记录与真实launch分账。如果接口离不开额外physics预热、不能还原相同初态/控制与关键readback，就记录技术限制并停止，不能绕过有限预算。三引擎分层只能把3.2→4的版本变化与4CPU FP64→MJWarp FP32的backend/精度组合分开，后者仍不能单独证明“纯FP32”因果；单hold也不能验证完整350ms排序或原任务稳定。

此有限核验只有在W47计划明确纳入、资源准入且需要区分这些假设时才执行；本次没有启动它。只有确实定位误差来源、确认原判据覆盖以后，才另议是否值得把完整 cost/risk移到GPU。当前缺少这两项，用GPU直接筛选/排序可能改变优化对象，不应代替native裁决。

这条工作与CPU控制稳定性研究可以并行，但它不应成为CPU主线等待的依赖，更不能作为已找到原任务终局跌倒原因的替代品。GPU空闲时停止已经完成的有限任务，是资源留给下一项有解释力工作的安排。

