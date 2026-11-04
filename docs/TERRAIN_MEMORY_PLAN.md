# S1：r26结果复核与r27入口平滑双流联合计划

## 当前交接（执行后只更新这6行）

- 当前任务：W38_SMOOTH_APPROACH_PARALLEL_JOINT / S5-r27 已在D57最新main上完成实际源码与双流接线，正常构建完成（158.78s），状态implemented_built_pending_launch；尚无r27物理结果。
- 已完成W37：r26新增2048条/8更新，累计6545/25＝6400已训练＋145历史excluded＋0当前尾；最新resume-state-ep00216.pt、checkpoint-006545.pt，父历史仅计一次。
- 真实能力：72回合中36跌倒、34躯干非足接触、1材料域退出、1达到目标截断；50回合材料加载、7回合机身入条带，合规去程/返程/整圈均0；最终E30在14.3096s失败，E60阻断。
- 已修改：LandingVelocity/ImpactMargin共用75.25mm带内完整C2 g*r；物理终止先结清未完成贯穿正credit；现有planner/action链记录accepted/reference、winner/regularization及原始Kδx/限幅动作。
- 执行：源码/config先推，正常构建后同父ep216启动w38-a/b；各8 workers、Torch1/BLAS1、独立optimizer/buffer/RNG/job/phase/deadline文件；共同28h截止，各自E30→条件E60≤4h，compact finish/Git串行。
- 交付边界：r26的73视频/Release历史保持，94b43d0已完成紧凑记录和两项补传，pending0；从W38执行compact-first，不自动传大权重/视频/数组、不做PC归档、不删服务器输出，用户备份后自行统一删除。

## 1. 长期目标与当前实际阶段

长期问题仍是：**同一个世界坐标地块被前足、后足及返程反复踩踏时，怎样利用可恢复材料的持久状态，让未来支撑需求约束当前加载、落点和重访时间？** 原始路线保存在[2026-10-21冻结计划](https://github.com/kaiwen123-yang/MCP-RL/blob/4d0af25c7bcea620c7e8c9a4f5b103c4956a3fda/docs/TERRAIN_MEMORY_PLAN.md)。受控黏弹材料和完整Go2闭环是起点，随后证明记忆的因果收益、泛化、NX推理和实机；IROS是近期目标，T-RO是更高目标。

当前真实运行的链路是：冻结因果PANN观察器 → 标准PPO标量速度cap → 全身MuJoCo iLQG → TV全状态反馈 → PD → Go2/真实TM02 → 观测。PANN参数冻结、地块状态持续更新；PPO训练Actor/Critic。MPC候选尚不传播完整PANN/TM02历史，2Hz cap也不是终端重访价值或跨足加载协同算法。

| 层次 | 已有证据 | 当前缺口 |
|---|---|---|
| R1控制基础 | 保留全部三轮正负结果，固定2.4Hz与adaptive强对照 | 未证明自适应全面优于固定频率，不重凑复现成功率 |
| R2材料辨识 | 三臂60模型、600工况/700历史评价；B ideal已供当前观察器使用 | noisy恢复/重访仍有负结果；预测准确不等于机器人收益 |
| S0与早期S1 | 世界地块持久历史、力/位移相容映射和局部决策证据 | 不能替代自由机器人完整通行 |
| R8刚地 | 项目S3九原任务9/9、独立连续300s任务3/3 | 不代表材料任务或当前版本已合格 |
| 材料联合基线 | r24曾有合规去程；r26恢复向前、材料暴露与连续8次更新 | 当前稳定合规往返仍为0；最终E30未过 |
| 论文主张 | 完整工程链已接通 | 跨腿/返程记忆带来的动作改变和收益、候选材料传播、终端重访价值、NX/实机均待验证 |

**可以继续推进实验，但不建议原样再跑一轮r26。W38只修当前有证据的入口残差机制，随后直接双流完整joint训练。** 不把“代码能运行、更新次数增加”解释为“稳定材料通行或论文贡献已经成立”。

物理/学习复算固定依据为[3d285451](https://github.com/kaiwen123-yang/MCP-RL/commit/3d2854510862ec9822dd4a1eee88daf4434c03d8)；发布前另核对[最新交付94b43d0](https://github.com/kaiwen123-yang/MCP-RL/commit/94b43d0bda02fecddc7734fc4afa00ac713d8bbe)，其中新增紧凑记录和两项Release补传，没有新增或重跑物理/学习。读取了全仓目录/身份与历史主线、全部72个W37训练JSON和resolved配置、8次学习记录/rollout及2048条transition、最终E30和必要的原采样率轨迹/物理/预测/材料事件、实际r26补丁及执行/学习/交付源码。并行复核历史、控制因果和论文；未逐个下载所有大权重/视频，也不声称语义阅读了每一个二进制字节。精确复算、来源和局限见[W37证据](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W37.json)及[E30证据](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W37-e30.json)。

## 2. r26推进了多少

### 2.1 数量、轨迹深度与真实能力

| 阶段 | 控制器 | 回合 | 新样本/更新 | PANN累计样本/更新 | 实际结果 |
|---|---|---:|---:|---:|---|
| W3 | r21 | 12 | 256/1 | 256/1 | 完整学习链接通 |
| W34 | r24 | 30 | 1024/4 | 1280/5 | 3次合规去程，0合规往返 |
| W35 | r24 | 73 | 2048/8 | 3328/13 | 1次合规去程，3次裸往返均不合法；最终E30材料域退出 |
| W36 | r25 | 27 | 1169/4 | 4497/17，尾145 | 全部未触材料，整体退行；评价取消 |
| W37 | r26 | 72 | 2048/8 | 6545/25，尾0、excluded145 | 恢复向前与材料加载，0合规贯穿；最终E30躯干触地 |

PANN累计214个真实训练回合；ep13/14的零物理步ABI失败单独保留。RAW历史256/1独立，不混入PANN。W37新增编号4498–6545连续、无重复，更新18–25每批256条、每次16个optimizer steps；历史4353–4497的145条r25尾段没有进入新更新。末回合ep216在2.5仿真秒达到目标，是外部截断；`budget_truncated`分类不代表耗尽28h，真实停止原因是`target_reached`。update25的训练后能力由独立E30检验，不拿ep216的短前缀代表训练后性能。

| 指标 | W35/r24 | W36/r25 | W37/r26 |
|---|---:|---:|---:|
| 材料加载回合 | 73/73 | 0/27 | 50/72 |
| 机身进入x≥.8条带 | 72 | 0 | 7 |
| 最远x | 3.00755m | .00002625m | .968767m |
| 合规去程 | 1 | 0 | 0 |
| 到远端/开始返程 | 7/7 | 0/0 | 0/0 |
| 跌倒/非足/材料域退出 | 18/12/40 | 22/4/0 | 36/34/1 |
| 原始往返/合规往返 | 3/0 | 0/0 | 0/0 |

r26的50个材料加载回合有43个机身从未入条带；七个机身进入回合中，只有四个在进入后记录到条带内部足端正Fz。材料域退出40→1主要同时伴随探索路段缩短，不能据此宣称冲击/本构域问题已经解决。全部34次非足终止都是geom13躯干；71个自然回合都至少有一个关节曾触力矩限，66个回合的12个关节均曾触限，**最大值证据不等于持续饱和**。

### 2.2 八次更新没有显示明确的任务改善

排除ep216目标截断后，前36与后35自然回合的平均最大x为.685299→.698812m，仅增加1.35cm；存活中位数14.4829→12.8246s，跌倒13/36→23/35。单流不能做显著性或独立seed结论，但不支持“明显学会了通行”。

| 消耗批次的更新 | 新批PANN非零/256 | 平均cap(m/s) | 平均reward | loss |
|---:|---:|---:|---:|---:|
| 18 | 0 | .213687 | −.30551 | .20549 |
| 19 | 5 | .213561 | −.34436 | .23838 |
| 20 | 29 | .211868 | −.26764 | .21184 |
| 21 | 4 | .212964 | −.37932 | .20459 |
| 22 | 10 | .213704 | −.34611 | .18271 |
| 23 | 4 | .214124 | −.34160 | .15632 |
| 24 | 9 | .212071 | −.38224 | .13004 |
| 25 | 1 | .214526 | −.34193 | .16330 |

行为log-prob重算误差≤9.54e−7，approx KL约.00629–.01505，观察为有限值，没有发现PPO数值爆炸或批次错配。loss下降可以来自更好地拟合重复失败分布，不能替代任务指标。各批cap均值几乎不动，整体范围.186047–.243760，未卡在[.18,.25]边界。

PANN非零恢复特征仅62/2048＝3.03%，分布于16回合，返程观察为0。E30已有同状态特征清零诊断的29点中27点cap完全相同，另两点差约2.61e−9和3.52e−6m/s。这是本轨迹即时动作依赖很小的证据，**不是重训对照或完整反事实rollout**；不能宣布PANN普遍无用，也不能宣称已有记忆收益。W35与W37奖励版本不同，总reward不直接横向排名。

### 2.3 最终评价与成本

最终checkpoint-006545固定策略均值、冻结归一化、无PPO更新。E30于14.3096s以`invalid_contact:[[0,13,0,1]]`终止，最终base=(.746252,.669964,.227356)m；机身未到x=.8，但已越过y=.6。`side_exits=0`只是尚未进入passage统计区，不能解释为无侧偏。E60正确标记`blocked_by_predecessor`，E300/600未运行。

W37训练北京时间2026-11-03 15:56:23至11-04 10:17:32，18.35234h，111.59新样本/h；逐回合墙钟合计18.22724h，物理1006.7796s。规划占回合墙钟84.18%，平均每次1.09651s；PANN观察/查询约.97%。W35训练22.61575h、90.56样本/h；r26总体较快还包含材料工作量减少，不能全部归因于实现提速。W36真实训练22.41426h，后处理启动延迟不计入训练。恢复的累计`wall_s`跨停机，不作本轮吞吐分母。

## 3. 失败原因：确证、强嫌疑和未解决边界

### 3.1 确证缺陷：物理终止没有结清临时进度奖励

`LegalRoute`声明未完成贯穿的正临时credit撤回，现有代码在phase切换、侧出等运行分支结算；`PolicyMaterial.finish()`在outbound物理失败时没有关闭该passage。71个自然失败中70个保留正进度，64个机身根本未入条带；transition中进度增量之和与结果一致，问题影响实际reward。

自然失败残留正进度奖励合计+38.34037，71次失败惩罚合计−710；71条自然回合return仍全为负（约−10.249至−9.494）。**必须修，但量级与时序不足以把它当作主要物理致败原因。** 物理失败须在最后transition形成前结清尚未完成贯穿的正临时credit；正常进行中的密集反馈、合法已完成贯穿计数和外部采集截断要保留真实语义。

### 3.2 当前优先因果候选：接近材料时，人为残差硬开关与局部反馈耦合

最终E30的原采样记录给出以下顺序：

| 仿真时刻 | 已保存事实 | 能说明什么 |
|---:|---|---|
| 9.126s | FL中心越过approach左边界x=.7405613883m | LandingVelocity与ImpactMargin的空间布尔开关由0突然激活 |
| 9.136s | executed qtarget相对clipped nominal首次超过.5rad | 大反馈修正出现在越界后10ms |
| 9.378s | 修正达2.632905rad，多个目标到ctrlrange端点 | 四足中心仍在真实材料区外，尚未发生真实材料加载 |
| 9.924s | 首足中心进入完整足印材料区 | 晚于前述大修正 |
| 9.9864s | 非足接触开始出现 | 此时可能是允许的腿部接触，不是14.3096s躯干终止 |
| 10.6004s | material-events首次真实加载 | 2ms轨迹对应10.602s；真正软材料接触更晚 |
| 14.3096s | geom13躯干触地终止 | 前面已出现横移和足端支撑失常 |

10–11s真实总Fz均值155.52N，其中足端22.73N、非足132.79N。不能把“足端力减少”误读成总地面承载消失；实际是允许的腿部接触承担了大部分支撑。依据见[原采样摘要图](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W37-e30-approach.svg)。

实际C++在x<.8+.0158113883−.07525时直接返回0；越界后立刻启用已有落足速度项及按.2mm尺度归一化的ImpactMargin。9.0s保存的反馈参考轨迹中，9.04s一个FL节点距这条边界仅.319微米，而FD扰动为1微米。该参考节点距人工边界的距离与FD尺度相当，支持差商受模式切换影响的嫌疑；但当前使用forward FD，接近边界本身不能证明实际扰动跨门，实际C/D与K也未保存。

9.378s对齐到执行feedback reference时，base高度差约.234mm、最大关节位置差.00319rad，但最大关节速度差仍有.7905rad/s。不能只用位置误差小反推K异常；原始K未保存，2.6329rad也不是未限幅Kδx。完整状态差在E30证据JSON中保留。

这仍是**首要待验证机制，不是已证明的唯一根因**：尚未做相同初态的单因素完整闭环比较，也不能把一条E30的时序推广成71次失败逐条已证明。空间硬开关存在是确定事实；它经FD/反馈造成失稳的因果权重须由r27完整joint新结果检验。r26的共同参考已消除旧1m累计lead，本次预测horizon内最大lead约.13365m；当前证据不支持再重做同一个参考修复或再把head偏置当成新发现。

### 3.3 记录语义边界与真正未解决的问题

上游planner保存的`policy.trajectory`可能是feedback reference，而非此次line search最终accepted rollout。winner非0时，现有NPZ不足以完整恢复两者身份。重居中可能保持反馈动作等价，**不能由此直接宣判“选中A却执行B”**，也不能把参考轨迹的躯干几何穿透直接叫作accepted候选违法。r26留在仓库的planner-record inc尚未接入实际补丁。

下轮需在同一plan记录winner/regularization/选中代价、accepted rollout与feedback reference的身份、原始Kδx和限幅后qtarget，并只保留解释入口事件必需的节点切片。原全频率数组留服务器；不新建监控/回放平台。现有qtarget−clipped nominal是**实际执行修正**，不是原始未限幅Kδx，不能混称。

更深的边界仍存在：MuJoCo平滑接触与真实TM02有模型差异；ImpactMargin是几何/速度软风险代理，不是本构有效域硬约束；PD/TV反馈可以放大局部预测误差。50mm厚、.02域限约对应1mm竖向压缩尺度，而r24真实足端下降曾到约1.41m/s。2Hz cap只有[.18,.25]m/s纵向权限，不能直接命令毫秒级接触力、横向纠偏、脚高或触地时序。当前应先消除人为非光滑入口；若此后仍有同类大反馈或材料冲击，再由实际新证据决定接触可行性、带宽或一个明确的物理动作通道。

## 4. W38 / S5-r27具体执行计划

### 4.1 单一控制修订：同一个C2空间激活

以实际r26构建为父版，不回r25/ep144。保留共同参考packet、head偏置、加速度界、r24运动骨架、十子步A/B、原forward FD、MakeDifferentiable、TV反馈和PD。r26参考公式及端点/转向例外保存在[实际父版计划第4.2节](https://github.com/kaiwen123-yang/MCP-RL/blob/3d2854510862ec9822dd4a1eee88daf4434c03d8/docs/TERRAIN_MEMORY_PLAN.md)，此轮沿用。新版本[S5-r27计划规格](../experiments/terrain_memory/s1-go2-r11-S5-r27.json)只是待实现规格，不能写成已经构建或运行。

令完整足印范围`xmin=.8+h, xmax=2.2-h, ymax=.6-h`，`h=.015811388300841896m`；沿用`b=.07525m`，不新增带宽网格。定义：

```text
S(u) = clip(u,0,1)^3 * (10 - 15*clip(u,0,1) + 6*clip(u,0,1)^2)
g(x,y) = S((x-xmin+b)/b) * S((xmax+b-x)/b) * S((ymax+b-|y|)/b)
LandingVelocity_new = g * LandingVelocity_without_the_spatial_boolean
ImpactMargin_new = g * ImpactMargin_without_the_spatial_boolean
```

完整材料区内g=1、提前区外g=0、其间C2过渡。原垂向条件、落足目标、风险onset/scale/weight不变。这一修改会改变接近带内的代价形状，不只是数值导数修补。两项共用helper，不能一个残差平滑、另一个仍硬切，也不能只把Jacobian做小而保留跳变cost。

Impact每个0.2ms pre-integration状态先算完整`g*r`，再在2ms内取逐行最早最大值；FD对完整乘积求导，经已有P/Q前缀复合。终端同式，LandingVelocity仍用节点起点完整C/D。MaterialDomainMargin、TrunkClearance和十子步不删。**只改优化代价激活；真实材料foot判定、Ground/接触力切换、应变域.02和真实终止不乘g，不软化物理边界。** 峰值切换等其他非光滑仍存在，不能宣称全局光滑或真实可行性已保证。

此外，完成第3.1节终止关账与第3.3节已有记录链的必要接线。正常源码检查/构建后直接完整联合采样，不另开pure MPC、单模块、E30准入、恢复演习、参数扫描或吞吐benchmark。不要凭此轮猜测任意裁剪K、关TV反馈、加输出低通、改PD、重新添加MovingSupport或六维twist残差。

### 4.2 两条完整联合流并行

| 项目 | w38-a | w38-b |
|---|---|---|
| group /目录seed标识 | PANN_PPO_S5 / 1 | PANN_PPO_S5 / 2 |
| 共同父状态 | ep216，6545/25，tail0，excluded145 | 同一父状态 |
| 首次恢复后探索RNG | 2026110401 | 2026110402 |
| 新样本/新更新目标 | 1024/4 | 1024/4 |
| 分支目标计数 | 7569/29＝7424 trained＋145 excluded | 同左，独立checkpoint |
| solver/Torch/BLAS | 8/1/1 | 8/1/1 |
| 控制、材料世界、PANN状态、PPO optimizer/buffer | 全部私有 | 全部私有 |

这是**同父权重的两个工程探索分支**，不是从头独立训练的多seed统计证据。恢复Actor/Critic/optimizer/归一化/历史后，在首次fork明确分叉NumPy/Torch/CUDA采样RNG；以后普通resume保留本分支自己的RNG和合法尾段，不再次reseed。当前父tail为0，无需再次隔离145条；excluded历史原样继承一次。无完整环境快照时另起真实物理会话，不伪造连续successor。

两流各达到目标时，新采集一共2048条、新更新一共8次，但**不存在一份融合后的“8593/33策略”**。跨分支唯一历史样本数可写6545+1024+1024=8593；模型能力和更新计数必须分别报告。保留同样256样本、4epochs/minibatch64、2Hz cap与658维观测，更新不强制重置物理回合。

直接复用[父ep216](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-job-20261103T075623Z-part03/job-20261103T075623Z__training__PANN_PPO_S5-seed0__resume-state-ep00216.pt)，服务器已有时不重复下载；不下载全历史。共同初值不是PANN与RAW公平对比，本轮不新增RAW分支。

### 4.3 并行接线与算力边界

现有入口不是双流即开即用，必要改动直接落在原脚本：

1. `start_s1_go2_r12_tmux.py`不再只取`active_streams[0]`；显式选stream、job/phase/session加分支身份，冻结各自execution-contract。同秒job命名也不能碰撞。
2. `train_s1_go2_r12.py`/评价入口允许已声明seed标识，训练目录使用独立group-seed；末端评价按job/stream/controller/reward/checkpoint匹配，不借另一流或旧版资格。
3. 移除共享`small/training-deadline.json`通信，使用各job的deadline文件并传给PolicyMaterial；两流计算私有buffer、计数、observer与Agent，使用已有自动RPC端口。
4. 在**checkpoint恢复之后**执行仅一次的fork RNG；在stream-identity/resume记下父身份、fork seed、controller/reward边界，不能以启动前seed代替。
5. runtime/index已有flock可复用；finish的索引更新也用同锁，紧凑汇总和Git提交/推送串行。父tmux协调两个既有训练→评价子链和完成后的compact finish即可，不建立调度平台。提交明确job拥有的小文件，交接同时报告两流，不能一条流结束覆盖另一条的在途状态。

16个solver workers以外给控制/RPC/JAX/I/O留余量；这不是严格总线程数承诺。保留GPU PANN/torch cuda、FP64和`XLA_PYTHON_CLIENT_PREALLOCATE=false`。依据实际任务自然记录两流新增样本之和/共同墙钟、每流planning time；不承诺两倍速度，不另开worker网格。当前CPU iLQG占84.18%，PANN观察/查询不到1%；GPU空闲显存不是缺一个CUDA开关。MJX或GPU规划重写留待真正吞吐瓶颈研究，不作为W38前置条件。

两份独立trainer不能共写optimizer、拼异步旧策略样本或冒充一个并行PPO learner。若后续改同步双环境collector，须统一behavior版本、独立GAE边界、单learner更新后广播；每环境128×2会改变GAE截断长度，这是另一次明确算法协议变更。

### 4.4 预算、停止与末端评价

交互≤4h，从Codex首次工作计时；源码/config先普通提交推送，正常构建后启动并见到真实采样/普通日志前进即可离开，不等待首个完整update。两训练流共用**首条实际启动起28h的绝对截止**，第二条较晚启动不重置预算；各自达到1024/4即结束，可提前进入自己的末端评价。每流评价自其训练结束起≤4h，保持每流8workers，因此另一流仍训练时不形成额外第三个计算槽。两个flow中的slot始终只运行训练或评价之一。

普通物理失败保存原结果后自然续采。保留现有`known_r25_startup_regression`窄判据：自然fall/nonfoot、仅outbound、无材料/无后续phase、max前进≤原.15m容差且final退行<−.15m时停止该流；r26入口失败不误套成该判据。还在目标/截止/技术中断时停止；不新增失败次数或attempt cap，不自动延时、串联r28或重跑r26。

正常结束仅用**本分支r27/reward新phase最后实际完成的新update**做固定均值/冻结归一化nominal E30，同case合格才E60。无新update如实unavailable，known-r25停流则skip。E30仍要求原连续30s、远端3m与合法去程条件，不要求30s完整往返；E60仍按原完整任务。stage_pass、legal贯穿/整圈和资格分别报告，不把原始圈数当合规。E300/E600不自动启动。

完成或部分结束立即紧凑交付，沿用≤6h兼容上限但push完成即结束；不因大文件传输等待。若实现未完成，在4h内推具体未完代码/状态，不把旧r26改标签启动为r27。

## 5. 从下一轮执行紧凑交付；r26历史保持

用户2026-11-04明确：**r26最后保持老样子，下一轮再执行新的GitHub规则。** 保留r26已完成的73段渲染与Release历史。推送计划前又读取了最新94b43d0：全部72训练回合和1次E30的紧凑结果/配置/学习记录已补齐，原ep176 XML与ep185 material-0011.npz两项Release失败也已补传并按名称/大小/uploaded状态确认；当前`pending_assets=[]`、compact发布`published`、删除0。历史2267/2269与push失败保存在`historical_bulk_publication`，不是当前未完成状态。不要为旧记录再次重训、重渲染或补全量归档。

通用`records/training__PANN_PPO_S5-seed0__learning.jsonl`等仍是早期256/1，不能当本轮数据。真实W37记录此前在[Release part03](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-local-readiness-r12-job-20261103T075623Z-part03)的stream-records，本次先用这些记录复算；最新提交已把collection/progress/stopped和八次学习记录放进[同job的training目录](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/jobs/job-20261103T075623Z/training/)，新增小学习记录与所读Release数据一致。W38继续复用该按job/stream命名的入口，不重复补这项已完成工作；密集reference-packets.json仍留原始数据，不进默认compact Git。

**W38默认顺序：服务器完成计算 → 普通Git推源码、配置、全部逐回合JSON、学习/评价小记录和结论 → 继续研究。** 大checkpoint、resume权重、全长视频、rollout及原始轨迹/预测/物理/材料数组留服务器，不自动全量渲染或排队上传。仅具体诊断所需的时间窗、矩阵节点、短片或选定权重按需沿用原Release入口；不为视频重跑物理。

精简种类，不挑成功回合：全部真实失败、零步、技术中断、截断、缺失原因都进紧凑证据。JSON/JSONL后缀不自动意味着小文件；密集原始数据留本地，逐update及任务关键事件摘要进Git。每job写清自身目录、全部episode目录、`training/<group>-seed<seed>`、末次有效resume和update位置。

用户定期导出到自己的存储盘，**备份后由用户统一删除**。Codex不自动删除服务器研究输出，不在用户电脑额外归档，不把compact push成功当作原始数据已备份。旧冻结cleanup授权不能覆盖新规则。已运行的冻结计算/评价契约保留身份；新的finish按当前交付策略处理，计算不因交付更改而重启。Git失败保留全部输出、普通续推，不强推/重训/删除。

本次是仓库分析与计划发布，未SSH、未部署r27、未查现场磁盘。原仓库顶部记录新版finish已同步D57，这是历史执行记录；下一次实际启动须以服务器文件和真实job状态为准。

## 6. 论文借鉴：具体机制与迁移边界

| 一手来源 | 与当前失败有关的机制 | W38实际采用与边界 |
|---|---|---|
| [TM20 MuJoCo全身MPC v3 §III/IV/VI](https://arxiv.org/html/2503.04613v3) | PD纳入预测，前向FD，局部TV反馈；iLQR依赖局部可微模型，接触模式和串行递推有限制 | 保留骨架，修人为代价硬开关；impratio100已经采用，不重复当新方案；论文不证明本TM02任务可用 |
| [TM33 平滑动力学上的线性反馈，§VII](https://arxiv.org/html/2411.06542) | 单边接触使对称线性模型可能要求现实无法执行的推/拉；平滑不自动使LQR可靠 | 解释为何需要区分参考/accepted轨迹和实际反馈；研究为准动态双臂操作，不直接移植四足参数或关闭反馈 |
| [STANCE §IV/VI-C](https://iit-dlslab.github.io/papers/fahmi19tro.pdf)与[SBIP作者方法](https://ddebenedittis.github.io/soft_bilinear_inverted_pendulum_astro/) | 力、压陷、接触运动联立；规划和跟踪同时考虑柔顺，内环带宽影响软地稳定性 | 恢复有效通行后继续解决加载/卸载相容；不抄HyQ增益、恢复不相容MovingSupport或称已实现其WBC |
| [TM34 Frequency-Aware MPC，RA-L2019 §II](https://www.researchgate.net/publication/327621320_Frequency-Aware_Model_Predictive_Control) | 把频率整形写入增广状态的最优控制，规划预先考虑有限变化能力，而非输出后加滤波 | 若r27仍有真实高频执行/载荷不相容，这是一条有机制依据的后续方向；W38不并行引入该大改，不照搬ANYmal参数或声称整体稳定性证明 |
| [TM27结构化参考 v1](https://arxiv.org/abs/2610.24779) | 可独立trot的nominal骨架；复制完整controller–simulator loop并行训练，给予body/force/foothold等权限 | 支持完整joint并行和后续有证据的动作权限设计；作者RL替代reference generator，100Hz/25维/256环境不是现成iLQG移植；仅平地仿真、ICRA2027审稿中 |
| [MuJoCo3.2.3 MJX](https://mujoco.readthedocs.io/en/3.2.3/mjx.html)、[Tianshou并行说明0.5.0](https://tianshou.org/en/v0.5.0/tutorials/cheatsheet.html) | GPU批量后端与CPU C++不同；环境隔离、同步/异步和seed语义需明确 | 本轮使用两个独立CPU MPC完整流，共享GPU推理；不声称切cuda就迁移规划器。依赖实际Tianshou0.5.1，说明文档不是精确同版源码 |

阅读范围和未复现边界记入[文献清单](../literature/terrain_memory_sources.json)，不新增论文全文复现或阅读准入阶段。

## 7. 下一阶段的科学判断

W38的核心问题是：**消除材料接近区的人为残差突变后，完整控制链能否稳定进入条带，使联合学习接触到真正的加载、后足重访和返程数据？** 两分支比较全部回合的入口反馈峰值、关节触限持续占比、足/非足实际Fz、落足速度、机身入条带、合规贯穿/返程与末端固定策略结果，不能只看loss和最好一回合。

如果入口反馈异常减小但仍在材料域/落足失败，下一具体机制是预测—执行的载荷/压陷/触地加速度与带宽相容；若仍在入口大反馈，则用已补齐的accepted/reference/原始Kδx记录检查局部模型与可执行反馈域。不要自动给旧现象换权重、增加同类残差或无限堆样本。只有实际证据表明cap权限不足，再选一个有物理含义的有界reference动作，不同轮同时扩维、改奖励、扫步态。

形成可用合规基线后，才用同执行器/奖励/硬件/worker条件的fixed、RAW、PANN，多seed和重训消融，区分在线估计、跨足共享、返程保持和普通历史。随后将持久材料状态进入候选传播与终端重访价值，使未来支撑确实影响当前加载与落点。长时E300/600、真实恢复样件、NX实时和Go2硬件按各自资格与预算推进，不能提前写成完成。

## 8. Codex启动词

```text
请执行GitHub最新main中的W38_SMOOTH_APPROACH_PARALLEL_JOINT。先读AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md顶部6行及第4/5节、当前R12契约和S5-r27计划规格；复用autodl-d57:/root/autodl-tmp/MCP-RL、work/envs/d57及既有依赖/联网。开场最多5行后直接实施。先看同流tmux与最新真实状态，已有W38作业则接续，不重复启动或重置deadline。

W37/r26已完成：新增2048/8，累计6545/25，历史excluded145、当前tail0，父resume-state-ep00216.pt，末次checkpoint-006545.pt。r26最后按老样子，保留其结果与渲染/Release历史，不补跑；两项旧上传失败已在94b43d0补传，不再重做归档。r27只是计划，必须真实修改和构建，不能给r26改名。

本轮控制只做计划4.1的共同C2 approach activation：LandingVelocity与ImpactMargin使用原75.25mm band、同一g，候选cost、完整乘积FD、十个pre-integration子步峰值和终端一致；保留原垂向条件、系数、材料物理边界、.02域、r26参考packet/头偏置、十子步A/B、MakeDifferentiable/TV反馈/PD。另修物理终止时未完成贯穿的临时正奖励关账，正确区分terminated与外部truncated；补已有记录中的accepted rollout、feedback reference、winner/regularization及原始Kδx身份。不要任意裁剪K、关反馈、加输出滤波、恢复MovingSupport或开权重网格。

同时在现有start/train/PolicyStream/evaluate/finish入口完成必要的双流接线：w38-a/b同父ep216恢复Actor/Critic/optimizer/归一化及6545/25，tail0，excluded145仅继承；首次恢复之后分别设置探索RNG2026110401/2026110402，后续resume保留各自RNG/tail。group均PANN_PPO_S5，目录seed标识1/2，记录是同父工程分支。每流私有job/phase、目录、deadline、PANN世界、buffer、optimizer与Agent；索引复用现有锁，compact finish/Git串行，交接同时列两流。

源码/config先普通提交推GitHub，正常构建后直接两条完整PANN→PPO→MPC→Go2/TM02联合训练。各8 MPC workers、Torch1/BLAS1，GPU和FP64设置沿用；各新增1024条/4次更新到各自7569/29。不合并optimizer/异步样本，不把两份父6545重复计数或声称得到一份8593/33策略。每256条更新后继续真实回合，普通物理失败自然续采。保留已有known_r25退行停止条件；不新增attempt/失败次数cap、pure-MPC或E30准入、测试套件、恢复演习、benchmark和监控平台。

交互≤4h，见真实采样和正常日志即可离开。两流共同使用首条启动起28h绝对训练截止，较晚启动不重置；各自训练结束后，使用本分支本phase最后实际新update固定均值/冻结归一化做nominal E30，同case合格才E60，各自≤4h、仍8workers；无新update或已知退行如实unavailable/skip，不借旧/另一流权重，不自动E300/600或r28。记录实际吞吐，不承诺跑满或加速倍数。

从本轮开始：计算完成就先推实际源码/config、全部逐回合JSON与失败/零步/截断、按job/stream命名的collection/progress/stopped/学习小记录、独立评价与结论，再继续研究。大checkpoint、全长视频、rollout和原始数组默认留服务器，不自动全量渲染/上传；诊断需要再选择既有切片，不为视频重跑。我定期导出到自己的存储盘并在备份后统一删除；不要自动删服务器研究输出，不在我电脑另做归档。写清job/episode/training及有效resume位置。紧凑Git失败保留结果普通续推，不强推、不重训、不删除。

最终只更新正常job/index、主计划6行和实际配置状态；分别报告两流训练、合规任务能力、评价、compact Git交付与大数据本地保留。未完成的实现如实写具体代码位置，不能用新计划轮或旧版启动冒充本轮执行。
```
