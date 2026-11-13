# W48 最终审查：修复已部署，原轨迹不变；W49转向真实执行与恢复

审查日期：2026-11-13（Asia/Shanghai）。结果基点：`61e16975fa6b38dc8e2d31779c2bd82f1aaeb8b6`；上一审查/计划基点：`068d8388feba972dd0e8cbd5bbbb1a7fd3ee71bf`。本次是读取、独立算术/源码审查和计划发布，没有登录D57，没有运行新native优化、plant、PPO或GPU物理。

**S1可以继续实验，但仍在建立稳定移动和原材料任务基线的阶段。W48完成了正式r36、原A/B评价与关键记录；没有提高原任务资格。下一轮不应再次重复同配置r36，也不应把增加PPO、放宽判据或占满显存当成稳定化方案。**

[W49详细计划](../../../../../../docs/S1_W49_STABILITY_PLAN.md) · [Codex启动提示词](../../../../../../docs/S1_W49_START_PROMPT.md) · [机器交接](handoff.json) · [控制审查](control-review.md) · [记录/GPU/交付审查](records-review.md) · [新增一手文献](literature.md)

## 1. 现在到了什么程度

长期目标是在同一世界材料地块经过前足、后足和返程重复加载时，利用持久材料状态改善当前加载、落点与重访，并以完整合法任务中的RAW/PANN、冻结/打乱历史对照证明因果收益。稳定控制、有效历史曝光、候选内材料状态传播、记忆收益、泛化与Go2/NX是不同的里程碑。

当前实际链为因果PANN观察→658D PPO观察→2Hz标量速度cap→逻辑50Hz全身MPC→500Hz关节位置目标→PD60/5→5000Hz原物理。PPO曾真实训练；当前评价中权重固定，observer/material history继续因果更新。cap没有直接决定落点、支撑时长或关节恢复的权限，候选尚未沿各自未来动作完整传播世界材料历史。不能把“联合系统可以执行”当作“材料记忆方法已经验证”。

| 里程碑 | 当前证据 | 结论 |
|---|---|---|
| 历史刚地控制 | R8/S3原任务9/9、300s 3/3 | 有价值的历史基线，不能代替当前r36/TM02资格 |
| 材料路线进展 | W34/W35有合法去程与返程进展 | 稳定合法完整往返尚未建立 |
| 联合学习链 | W44 A/B各14737转移、57更新，完整resume保留 | 学习链能运行，不代表动作权限足够 |
| 数值正确性 | W47定位四元数自差；W48正式r36部署并验证 | 一个具体数值缺陷已修复 |
| 当前原任务 | W48 A/B E30均失败，合法去/返/往返均0 | 仍未形成稳定材料任务基线 |
| 问题可观测性 | 1904上一accepted预测配对、277事件call、995对象 | 可以从现有证据定位实际失稳链 |
| GPU替代评分 | 旧重复组首2ms即分叉；W48无新增GPU physics | 尚未形成可信native替代 |
| 记忆因果收益与实机 | 匹配因果对照、泛化、NX实时尚未建立 | 不宜提前进入论文主结果扩量 |

历史依据与谱系见[上一主审查](../W47-final-review/README.md)和[项目记忆](../../../../../../docs/TERRAIN_MEMORY_PLAN.md)。PANN唯一转移22929，另RAW256，R12 distinct23185；不能将A/B共同祖先6545/update25重复算作独立数据。W45–W48新增PPO均0。

## 2. 最新原任务：确实执行了新实验，结果与W46相同

| W48原E30 | A / seed1 | B / seed2 |
|---|---:|---:|
| 正式controller | S5-r36 | S5-r36 |
| 物理结束 | 26.432s，floor–trunk contact | 11.638s，body_fall |
| 墙钟 | 3265.949s，54.43min | 2017.731s，33.63min |
| 最大body x | .626506m | .396364m |
| 实际材料加载 | 无 | 有，最大主应变.00649112＜.02 |
| 合法去/返/往返 | 0/0/0 | 0/0/0 |
| E30资格 | 失败 | 失败 |
| E60/E300/E600 | 各自blocked | 各自blocked |
| 权重及新学习 | W44 checkpoint-014737，新增0/0 | 同左，各自独立resume |

不是“8项都失败”：两项真实物理失败、六项前序阻塞，后六项没有运行。A没有实际材料受载；B在8.464s已低于诊断高度.2m、8.572s已侧倾超过诊断角.5rad，均早于首完整足材料加载9.122s。因而不能把材料超限当成这两条失败的统一解释。这两个诊断阈值没有变成新终止标准。90.86Nm是相邻±45.43Nm极限反转的差，不是实际超力矩。

执行者在D57对两轮原数组进行比较，提交的[original-comparison.json](../../w48/original-comparison.json)显示：A13217行、B5820行的time、qpos、qvel、q_target、torque、actual_grf、contact、cap、phase逐元素相同，最大差0。1944次调用的选择来源、alpha、plan、stop、rollout/FD数量也相同。文件字节hash可因墙钟字段不同而不同，不意味着物理不同，也没有证据支持复制旧轨迹。

本审查没有重新下载两轮原NPZ，因此将上述全数组结论明确标为**执行者提交的原位比较**。本次独立核对的277个事件call，选择/alpha/plan和工作量均与W46全调用CSV一致；其中15个事件call的`chosen_actual`不同。全程提交汇总有A123/B38次该评分变化。这支持“内部评分发生变化，但最终选择和实际运动不变”，而不是“修复没有部署”。

来源：[W48 phase](../../w48/phase.json)、[控制复算](control-audit.json)、[记录复算](records-audit.json)。

## 3. 为什么修了一个明确错误，还是原样失败

正式r36 binary SHA256为`b05c8da8ca74551dc73ea0272329234217d7160731e5cea8e33101bba3241338`。四个corrected有限问题与A/B实际`/proc/PID/exe`相符，N/r34 strict/constant正确，原任务没有离线score/import环境。正式补丁重建后的utilities与planner源码SHA也与build一致。

identity修复解决的是严格self/antipode的四元数自差注入：20组alpha0成本都恢复精确nominal。**但r34本来就不允许alpha0作为新步安装。修复被排除的零端点，并不会自动生成可安装、能恢复支撑的新方向。**

| 四个旧困难问题 | 修复后赢家 | 最小正步真实ΔJ | 赢家是否改变 |
|---|---|---:|---|
| A148 | nominal_3 | +.0337678 | 否 |
| A153 | nominal_9 | +.0450258 | 否 |
| B155 | incumbent | +.0774750 | 否 |
| B158 | incumbent | +.1059959 | 否 |

修前/修后每侧180个正候选，均预测下降、实际J上升；240个候选的序列化policy identity全部相同，20组backward的关键数值相同。98个正候选成本有变化，最大仅4.6861e−8，没能扭转赢家。保存数组hash相同不代表StateDiff可执行函数相同；这里的结论是实际选择不变。

四条350ms部署核对均精确一致，但赢家全是nominal/incumbent，核的是**native ActionImpl→同native ActionImpl、实际scale**。没有覆盖被接受正步的ActionIndex→ActionImpl，也没有把RPC返回动作放进积分。`w48_finite.py`是在固定rollout生成之后查询RPC。因此它们不能证明真实客户端、服务端、held输入和plant全链一致。

同时也不能反过来说“优化全程都找不到下降方向”。原1944调用中1601次正步安装：1598次按同问题规则J下降、3次风险优先；B明显下沉/侧倾前一秒各50次调用均正步下降。**代价下降与任务可恢复性之间还有未闭合的关系。** 不同epoch的参考和起点变化，也不允许把所有ΔJ直接累加成闭环收益。

## 4. “失败次数多”应分开计算

### 4.1 四次启动失败是已修复的配置继承错误

初版`w48_build.py`用新的`configuration_overrides`替换完整继承表，丢了17个键，其中包括`r12_trunk_clearance`。模型仍启用`r12_substep_path`；`Trajectory::UpdateTrunkViolation`直接索引缺失custom numeric，实际栈在`NoisyRollout/Rollout/FeedbackRollouts`启动路径SIGSEGV。

当前代码已改为合并继承配置并恢复17键；native binary没有因此改变。有限实验使用完整模型，不需重复8次solve。四个attempt为A3/B1，原plant步数0；崩溃前可能已做部分native预测积分，精确量未知，不能写所有计算0。CRLF/LF构建准备问题另属构建阶段，没有形成原任务资格。

下一轮针对这一真实回归做一次生成模型必需字段、配置来源和差异核验，避免又在完整原任务启动时发现同类缺键。不是新增一组plant“冒烟任务”。

### 4.2 两次正式物理失败不是后台到点结束

W48原任务没有固定墙钟deadline，结束原因为实际触地/跌倒。取消墙钟kill是正确工作流，但它不会自动修好已经自然发生的物理失稳。仿真采用同步逻辑时钟，墙钟慢主要影响吞吐，不能直接当成控制状态陈旧的证据。

### 4.3 候选拒绝、任务失败、数据缺失和交付故障不是一个分母

一个solve中的多个候选拒绝不是多次机器人跌倒；被阻塞的E60/E300/E600不是失败回合；R8下载EOF不是donor机制失败；Release上传TLS不是控制失败。旧四fixture是有意保存的困难问题，不能代表所有在线调用。反复混用这些口径，会让“失败很多”既难定位，也无法判断一轮修复是否有效。

## 5. 新记录使下一轮可以问更具体的问题

1904个非启动边界用的是上一个**实际部署策略的accepted rollout**在当前origin的预测，不是本次新解、不是真实误差应为零的反馈reference。各20个startup没有先前实际部署，unavailable合法。每次2ms实际held更新后才更新deployed身份，覆盖旧计划前保存P0与context。

本审查独立核验277个事件call、995个完整对象的size/SHA与交付清单；其中237条可用配对的source、P0、plan、节点时间、平移/关节差和速度算术一致。223条上一source也在提交子集中可交叉核验，14条位于窗口前边界；没有宣称独立重读全部1904原JSONL。233条是20ms配对，另有2条2ms、2条18ms阶段配对。

| 事件 | 真正事件时刻 | 要检查的此前规划区间 | 已有证据 |
|---|---:|---|---|
| A首次实际held跳变＞1rad | 3.846s | call213@3.84至3.86 | 完整policy、accepted、P0、context、reference |
| B首次实际held跳变＞1rad | 3.106s | call176@3.10至3.12 | 同上 |
| A首次膝极限反转 | 6.860s | call363@6.84及call364@6.86 | 同上；检查反转落在重规划前后哪个事件 |
| B首次膝极限反转 | 6.150s | call328@6.14至6.16 | 同上 |

两流首次大跳均发生在规划边界6ms后。A3.84边界最大关节误差仅8.53e−6rad、关节速度差.0010774rad/s；B3.10分别.000150774rad/.0283081rad/s。不能将50Hz边界的误差冒充6ms后的误差，必须按每个held输入、接触和状态排序。

第一次非零预测误差更早：两流均0.02s，仍在站立。全程误差最大分量A/B关节位置为.0492763/.0164553rad，关节速度为7.15560/4.30498rad/s。这里是闭环预测—执行差，混合了反馈命令、模型和初始化因素，不能直接称动力学残差并喂给学习器。

还需要判别两种机制：

- accepted计划预期健康恢复，实际动作/状态先偏离：优先修执行、模型或真实误差下的反馈恢复。
- accepted计划本身已下沉、停滞或失去支撑，却仍降低J：优先核目标冲突、接触时序与可用候选。

A3.84的J约2.18749，其中MaterialSupportHeight约1.86129；B3.10的J约.136767，其中该项.078966。分项大只是可检验线索，不等于该权重就是错误，也不能因为实际未进入材料就断言未来horizon不该有材料目标。

记录的一个小遗漏是`W48Records.plan`覆盖父类后未继承`trunk_restoration`事件解析。新关键事件已齐，不阻塞分析；下次接入恢复该事件。完整固定策略可复核不等于保存了整个历史optimizer生命周期或所有MjData数值状态，重建试验要标注coldstart/缺失字段。

## 6. RPC时间：应受控消除，不应再当成新发现的唯一根因

State的time/qpos/qvel为double，GetActionRequest.time为optional float，响应action为float。runtime每2ms先SetState(double)，随后显式GetAction(time=t)，可能让策略查询时间再次舍入。核定r36实际grpc源码SHA为`94b7328afb705bb226ee4dd306defce1a3f09972c4432da46bef8ee4c033e5c1`；缺省time使用刚同步的native state.time。

**W39已经识别并量化显式时间误差；W40已有动作FP32记录。** 旧首大反馈时，时间量化造成的参考速度差远小于真实参考速度差。这反对仅凭float字段就宣布它足以解释跌倒；旧策略也不能完全替代新r36事件定量。

最小可做干预是每次确已SetState后省略显式time，保持响应动作精度不变。先同state/policy/context零积分对比double与float查询，分开reference插值、Kδ、H-QP、响应舍入和实际held，再做有限短窗。若效果小或未改变事件机制，就明确排除主要解释力，并在同一W49继续恢复线，不能耗掉整轮后只提出再看精度。

作者接口的`get_action()`也省略time，但其硬件/UI是异步循环，不能整体照搬。细节及历史量级见[控制审查](control-review.md)、[文献与官方实现](literature.md)。仅将收到的float转为float64不能恢复已经丢失的动作精度；不应无证据重写整套protobuf。

## 7. 计算资源：双任务已经实现，GPU尚需可信用途

W48 A/B正式native affinity为0–9与10–19。两原evaluation进程实际重叠2030.4135s，即33.84min；这是实际双任务并行。A/B墙钟较W46更短，但同机负载未控制，不能把差值当加速benchmark。

| 已完成工作 | W48数量 |
|---|---:|
| 有限solve / 部署固定rollout | 8 / 4 |
| 有限native rollout | 484 |
| 原任务native调用 / rollout | 1944 / 57104 |
| 两部分已完成transition FD | 7519104 |
| 两部分已完成risk geometry FD | 1447990 |
| 原plant子步 | 190350，合计38.07物理秒 |
| 有限显式rollout子步 | 847000，另有FD内部积分 |
| 新PPO / donor solve / GPU physics | 0 / 0 / 0 |

上述是本轮新执行工作，数值与旧轮相同不代表重复计入历史；启动崩溃前未完成的native工作另列unknown，不混入精确合计。

GPU六组同声明身份重复world，native全窗精确相同，GPU在首2ms状态已不同，后续held差最大.074646rad。它不是历史CPU/GPU全窗1.758rad误差。隐藏初始化、warmstart、接触workspace尚未完整对齐，不能唯一归因FP32，也不能把这个GPU评分器直接接管正式native排序。W48 C约1.935秒数据分析，未新增GPU分配或物理，不能称持续三任务满载。

W49采用两CPU线各10、C主机总2、系统/主控/发布约3的额度。GPU做小范围首hold初始化对照或同状态动作计算，不重新跑24world×350ms；正式可信后再谈批量候选、导数或学习吞吐。有效计算比显存占用比例更重要。

## 8. 交付状态：主体已到GitHub，仍有技术尾项

本次实时查询三项科学Release，共13个uploaded资产、1,775,245,315B，size/digest与提交清单匹配。finite8个、A3个、B2个；两段视频在ZIP内。A之前TLS上传失败已续传完成。本次只读元数据，没有重新下载payload。

但`w48/delivery/w48-a`、`w48/delivery/w48-b`两份旧零plant技术job清单仍publication_pending，对应tag实时404。按hash，后续包已覆盖A14/B13个源，包括坏模型MJB；尚有A6/B7个非空源未覆盖，B含1,672,688B pre-call。下一执行者仅在D57按原manifest续传已有ZIP，不重跑、不重打包、不阻塞科学主线。

R8这轮单文件下载unexpected EOF、0 bytes，仍是输入网络可用性问题；没有donor求解结果。它不应继续成为稳定化的总前置门。来源：[实时Release核验](release-inventory.json)、[交付独立审查](records-review.md)。

## 9. 下一轮的决策顺序与长期推进

W49用既有W48事件并行分离真实执行链和支撑恢复，完成一个有机制证据的控制改动后再进行一次原A/B。time-only假说如果解释力不足，要转向接触恢复、目标冲突或候选可行性；不再重复旧四fixture、identity-only原任务或不受控调参网格。具体有限预算、候选选择和原评价入口见[W49计划](../../../../../../docs/S1_W49_STABILITY_PLAN.md)。

稳定后依次推进：原E30/E60→独立E300/E600→足够跨足/返程历史曝光→匹配RAW/PANN及冻结/打乱历史→候选内材料状态与必要高层动作权限→泛化与实机。先求稳定是对的，但“站住更久、走得更慢”不能替代合法原路线；恢复能力和前进能力必须一起报告。

**全部新后台不设固定墙钟完成时间/到点kill；保留有限实验工作量和原物理阶段。GitHub核验后不在用户电脑额外归档，服务器原件/checkpoint/resume/视频不自动删除。**

## 10. 阅读覆盖、验证和限制

对结果基点8750个tracked文件、1,044,013,899B逐文件完整读流与SHA256库存；相对上一计划88个变更文件、56,762,626B完整结构检查无错误。关键控制源码、配置继承、8solve、候选/方向、原调用与事件、GPU重复组和交付清单分别审查。字节覆盖不等于声称对每个byte作了人工语义理解，更不等于重放全部历史Release。

两个标准库复算脚本已实际运行；本次源补丁重建及事件算术验证不调用MuJoCo物理。详见[库存](repository-inventory.json)、[控制脚本](audit_w48_control.py)、[记录脚本](audit_w48_records.py)、[验证记录](validation.json)。在冻结结果checkout上可运行：

```bash
python results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W48-final-review/audit_w48_control.py --repo . --output /tmp/w48-control-audit.json
python results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W48-final-review/audit_w48_records.py --repo . --output /tmp/w48-records-audit.json
```

控制脚本可选`--r35-source`及`--rpc-source`校验重建源码SHA；未提供时只复核已提交数值/身份，不能声称执行了该可选源码检查。没有把未重读的远程原数组说成独立复现，也没有把计划写成已经实施。
