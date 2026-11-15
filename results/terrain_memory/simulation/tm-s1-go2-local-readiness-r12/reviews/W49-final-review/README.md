# W49 最终审查：30秒观察完成，稳定行走与完整材料任务仍未成立

审查日期：2026-11-15（Asia/Shanghai）。结果基点 `de6606dc8363e8ab1c816fa682cace752a06318a`；相对W49计划 `b142001624cfea5877c57e63f45e0f29c9dc4735` 审查19个新增提交、64个变化文件。**本次审查没有登录D57、启动服务器作业、运行native solve、物理/GPU积分或PPO。W50为计划，不是已完成实验。**

[W50详细计划](../../../../../../docs/S1_W50_STABILITY_PLAN.md) · [启动提示词](../../../../../../docs/S1_W50_START_PROMPT.md) · [机器交接](handoff.json) · [当前项目记忆](../../../../../../docs/TERRAIN_MEMORY_PLAN.md)

## 一、结论与成熟度

S1已经形成可运行的PANN—PPO cap—全身MPC—真实材料闭环，W49也确实改变了控制行为。两条原任务都自然观察满30 s，是对W48提前终止的局部改善。但低姿态、非足支撑和任务非进展仍在；当前完整材料任务资格、因果记忆收益及NX实时均未建立。

| 原E30指标 | A / seed1 | B / seed2 |
|---|---:|---:|
| 正式控制器 |S5-r37 / K|S5-r37 / K|
| 物理观察 |30 s|30 s|
| 资格 |未通过|未通过|
| 最终原因 |material_traversal_not_completed|material_traversal_not_completed|
| 墙钟 |101.68 min|79.22 min|
| 最大body x |.349667 m|1.020530 m|
| 合法路线进度 |5.81745%|17.01414%|
| 合法去/返/往返 |0/0/0|0/0/0|
| 最低body z |.123298 m|.130666 m|
| 最终body z |.138718 m|.142818 m|
| 首次非足载荷>1N |4.6232 s|5.0216 s|
| 首次足端材料加载 |无|12.1866 s|
| 机体西侧入条带 |无|20.064 s|
| 最终vx |.00001983 m/s|.00054401 m/s|
| 最终命令vx |.212763 m/s|.217916 m/s|
| 后续E60/300/600 |blocked，未运行|blocked，未运行|

[原始phase](../../w49/phase.json)、[A原结果](../../job-20261113T145020Z-w49-a-eval-PANN_PPO_S5-s1-E30-attempt01.json)、[B原结果](../../job-20261113T145020Z-w49-b-eval-PANN_PPO_S5-s2-E30-attempt01.json)、[独立结果聚合](results/analysis.json)支持上述数值。两个独立30秒不能拼成一个连续60秒任务。

初始z约.290209m。原body_fall高度线为.12m，A最低只高3.30mm；允许腿部部分几何接地，因此非足载荷不必触发invalid_contact。这解释“未提前终止”与“不稳定行走”可同时存在，不是本轮改了资格。A的最大前进量还低于W48的.626506m，B则高于W48的.396364m；不能把本轮写成全面提升或完全无进步。

### 用户视频补充：确实不是正常步态

用户随后提供的W49 B视频已作全程分段和早晚加密抽帧。约4.4–5s反复下蹲/顶起，7–8s后长期低伏、腿部折叠/外伸，后半呈贴地挪动。数值与画面相互支持，不能把它概括为稳定性已提高。[视频审查与关键帧](video-review.md)

W50优先预防起步后足部承重/步态失效，低姿态恢复作为补充；A从2.002s站立转outbound开始核实际gait/phase和足端受力。只有后期低伏挪得更快不成为新原回合的充分推进依据，原任务判据与短腾空许可保持。

## 二、失败为什么仍多：需要分清五类事情

1. **原任务不合格：2条。** 本轮都是自然观察满30s后路线不满足，不是两次早期摔倒、不是程序崩溃、不是墙钟到点kill。
2. **后级被阻断：6级。** 没有运行，不计为六次物理失败。
3. **技术尝试：3次。** 两次query-only启动因service ABI/导入ready修复后成功；一次deployment Init因模型XML序列化顺序中断后修复；都为physics0，和原任务分开。
4. **候选被严格规则拒绝：正常优化记录。** 不能把每个拒绝candidate当一次完整实验失败，也不能把每个J下降当一次任务成功。
5. **发布恢复：零新物理。** A恢复的是已完成实验的原件上传，现在已完成，不需要再启动实验。

W49原A/B各1521次planner调用；正步安装802/1393，均记录同问题J下降。汇总的 `internal_stagnation=0` 只统计旧枚举 `native_search_stagnation`；r37主要返回 `w49_single_first_direction_finished`，A末段还有backward_failed。不能据字段零值断言没有停滞。全量计数来自已提交执行端汇总；本次独立复算的是Git内231个事件调用及有限结果，非重新读取外部3042行全日志。

两原任务合计59,666 native rollout、6,012 derivative pass、11,579,112 transition FD、3,048,862 risk FD、300,000 plant子步。两流1501次普通planning都超过logical20ms，属于同步慢仿真；计算时物理暂停，墙钟慢不能直接解释仿真失稳，但说明目前远非实机实时。[详细工作量与计数语义](results/review.md)

## 三、真正的新进展与尚未闭合的机制

### 3.1 有限机制和接口已经有价值

W49四个保存问题的36个K候选中25个通过原strict改善，四事件均有行为改变；最佳J降幅3.1761%、18.6728%、1.9388%、33.5849%。原N正候选共108个仅3个eligible；两组工作量不同，不是公平总体成功率比较。四个K赢家下降远超roundoff预算，不能归为容差假象。

4×20ms真实RPC部署完成400 predictor子步，metric/QP有效；Task A 41个状态节点82次查询，native-vs历史及RPC-vs实际held精确一致。时间量化最大动作差约.003407rad、限幅后PD差约.2044Nm，这条旧精度支线没有解释当前低位驻留的充分证据。8×20ms共同重建不是完整历史plant回放。

有限binary与正式binary不同，build记录包含service ABI和导入cost metadata修复；正式r37 binary为 `8637e3d118c686831a2acb54ef48271de6092c6095f56c9456f7289607d734e5`，原进程身份正确。有限先完整N后加首k；线上在首有效方向后直接K返回。证据与语义详见[控制审查](control/review.md)、[独立算术](control/analysis.json)。

### 3.2 首要嫌疑是跨问题参考与反馈一直继承

源码确定：有效bootstrap后，K候选复制旧完整base，只更新delta；nominal只缩放反馈。当前backward产生的新K/H没有被安装，旧reference时标过期后仍端点夹持。新安装ID、generation_time、metric_source_plan_id不能证明这些数组变新。

这可解释为什么“每20ms有新优化和新动作”不等于“对当前状态有新反馈”。但是当前完整NPZ仍在D57/Release，本次未下载；约29.65s过龄是源码推断，必须用现有原件核实，不能当作已测唯一根因。检查现有对象即可完成这一步，无需新原任务。

W50主候选是在当前native rollout上重参考，以同一个原alpha转移raw前馈、切空间反馈和H，alpha0仍精确保留旧B。候选真实native评分、原strict裁决、原对象部署。这是待证局部重参数化方案；详细公式、来源和工作量见[W50计划](../../../../../../docs/S1_W50_STABILITY_PLAN.md)，不是承诺一次修好。

### 3.3 下沉和目标排序需要与生命周期一起看

A/B末MaterialSupportHeight占总J的97.2686%/97.1754%，accepted horizon本身仍有约.1515/.1469m的RMS支撑高度误差。A晚期backward_failed，B仍下降安装却无法前进。不是高度成本没有写入，而是当前可找到/可部署的候选没有恢复承重。

r26每次以实际x锚定短局部head reference；.215m/s匀速示例的起点前视.05375m、.35s末前视.129m。上层mission x=3不是native Position直接追踪的3m误差。小Position项支持检查长程进度与局部参考的关系，不能单独推出“增大Position权重”。A没有材料加载就下沉，B也在首材料加载前发生非足承重，材料模型或PANN不是唯一可能解释。

## 四、下一轮如何产生有用的新证据

两CPU线并行：A读取W49原件并分清生命周期/早期下沉/恢复；B实现一个重参考反馈候选。固定W49 A237@4.32、B236@4.30、A328@6.14、B625@12.08四例，每例只到首有效backward、9正alpha；不重复旧四fixture。

有限四solve加必要baseline≤80full rollout；最多两条40ms实际RPC部署，在20ms各做一次真实新epoch再规划，≤2额外solve/40rollout/400predictor子步。FD和零物理坐标导数另计。以早期A237/B236为首要依据，至少一个有原strict认可的实质动作/正常承重推进变化，或修复已证早期步态接入错误，且真实入口正确，即可冻结一个版本并行原A/B各一次E30；不用先证明整个任务成功。失败如实保留，不换seed或叠加多个模块追成功。

成本/参考结构、支撑交换权限、horizon/终端延续是条件后续方向。先根据同状态候选的可行性、排序与实际执行判断哪一层需要改；本轮不同时换求解器、步态、奖励和PPO。

## 五、GPU与并行的真实状态

W49两原任务实际重叠4,764.730s，即79.412min，双CPU运行已经实现。C仅3.51s/60world-step且在原A/B前完成；不能说三路持续满载或96GB已充分利用。

F1/L/F2首个.2ms步都出现浮点差，完整可见状态恢复F2也一样，不能只归因轻量reset。首步ctrl/actuator/passive/外力一致、排序后contact在10步均一致而constraint/qacc在首步开始不同。已有1,873,543B全暴露子步NPZ足够下一步零新积分定位约束装配/求解归约/写集；不需要再重复同一个2ms表格。

微小位级差本身不证明GPU工程不可用。当前full native J尚未移植，不能让GPU正式排序；可做旧状态批量数学诊断，后续研究GPU proposal、CPU native裁决。继续10+10+2+3CPU分配，build-j6先完成。C不阻塞主线，不为显存占满添加无效训练。[GPU与并行详情](results/review.md)

## 六、交付、范围和可信度

本次live Release API核对16资产共1,658,430,730B：W49有限4ZIP、A3ZIP、B3ZIP、W49/W48四MP4，以及W48两旧技术ZIP，均存在且uploaded、Release非draft。四视频和两旧技术ZIP还可与Git中expected size/digest比较；十W49ZIP的Git manifest没有压缩size/digest，本次只确认API元数据、名称和published收据，未下载重算payload。[远端核对](remote-verification.json)

W49 A的上传恢复已成功，不再写待续。旧build/phase中的N/not_started/old live pid属于历史快照；最终原结果、最终runtime与已核实交付优先。

背景无固定墙钟截止、无到点kill；有限工作量和原物理资格保留。GitHub核验后不在用户电脑额外归档，不自动删除服务器原件/checkpoint/resume/视频。

结果基点8817tracked files，Git blob大小合计1,055,255,013B；继承前轮全仓库存与历史审查，本轮64变更共49,031,786B全部读流，41JSON完整解析、23文本读取。关键源码、有限结果、两原任务、231事件调用和GPU由三条独立审查与主审复核。字节读取不等于理解每个数字；没有重新下载完整NPZ/视频或复放全部Release；用户后来直接提供的B视频已按上述范围查看。[阅读范围](read-coverage.json)

复算命令在仓库根执行，不产生新实验：

```bash
python results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/control/recompute.py --repo .
python results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/results/aggregate_w49.py --repo .
```

[一手文献与边界](literature.md) · [W48历史审查](../W48-final-review/README.md) · [W49历史计划](../../../../../../docs/S1_W49_STABILITY_PLAN.md)
