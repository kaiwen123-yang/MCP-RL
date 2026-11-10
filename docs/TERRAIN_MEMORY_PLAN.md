# 地面材料记忆主计划：W42驻留审查与W43优化器修正

**2026-11-10 已在 D57 实现 S5-r32，已完成保存输入核验并启动 W43 双流；旧 W42 结果保留。** GPU 已以 g5-20261110T033858Z 从 G5 续接，6 个 fixture × 3 个 FD 尺度完成，独立 E30 已进入真实物理（观测planner origin0.10s），尚无任务通过结论。

[W42完整审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W42-final-review/README.md) · [CPU原件报告](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W42-final-review/cpu-report.md) · [GPU报告](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W42-final-review/gpu-review.md) · [W43实施计划](S1_W43_STABILITY_PLAN.md) · [完整启动提示词](S1_W43_START_PROMPT.md) · [执行规则](../AGENTS.md)

## 当前交接

本次实际核对（2026-11-10T15:57:05+08:00）：W43-A/B仍运行，分别新增348 / 342条，各完成1次新更新到update54。四个已完成60秒采集回合均无技术中断，λ重置有效，但去程段约后退28mm、四足全程接触、材料加载和合法去返/往返全零。G5同一invocation仍在独立E30，最近检查物理5.64秒，303次planner均返回policy_installed，当前日志未见新技术异常。最新数字和源路径见index.w43_progress_observation；继续现有作业至原目标，末次评价尚未开始。

- 当前任务：W43_OPTIMIZER_LIFECYCLE_JOINT / S5-r32；后台按各自样本/更新目标结束，无固定墙钟完成期限；原物理时长与技术失败停止保留。
- w43-a：job-20261110T035658Z-w43-a，training_ended；新增512/2，累计14225/55，tail=0、excluded=145；评价=completed，紧凑Git=published。
- w43-b：job-20261110T035700Z-w43-b，training_ended；新增512/2，累计14225/55，tail=0、excluded=145；评价=completed，紧凑Git=published。
- 任务能力：w43-a合规去程/返程/整圈=0/0/0，材料加载回合=0；w43-b合规去程/返程/整圈=0/0/0，材料加载回合=0。每流仅本phase最后实际新update评价E30，同case合格才E60；训练更新不等于合规任务能力。
- 交付：两流compact finish/Git串行，使用原index锁；全部逐回合JSON、2Hz学习转移、12维PANN/value小表与E30关键事件窗先Git；已请求的原始反馈/规划/参考轨迹及必要数组由diagnostics入口后台Release，未选权重/全视频留D57，无自动删除。
- 续接位置：autodl-d57:/root/autodl-tmp/MCP-RL-w42；各job、training目录与有效resume见正常job/index的两流状态；分别继承自身父分支，optimizer/RNG/归一化不混合。

| 工作流 | 最新完成状态 | 下一起点 |
|---|---|---|
| W42-A / seed1 | job-20261108T154719Z-w42-a；13713/53，tail0，excluded145，ep405；E30满30秒未完成材料通行 | 自身ep405，W43-A各+512/2，到14225/55 |
| W42-B / seed2 | job-20261108T154727Z-w42-b；13713/53，tail0，excluded145，ep393；E30满30秒未完成材料通行 | 自身ep393，W43-B各+512/2，到14225/55 |
| W41 GPU | G1 5/32对齐；G2完成176状态探针；G4启动缺参、G5容差类型错误 | 原GPU树从G5续接，冻结W40-B/r30契约，复用G1/G2 |

CPU现有树为 autodl-d57:/root/autodl-tmp/MCP-RL-w42，GPU为 /root/autodl-tmp/MCP-RL。先核实现场，不能拿历史PID代替当前检查。已结束且干净的CPU树可复用，不为改名复制整库；有活动任务/未提交修改才隔离新树。GPU与新CPU可写源码/build不共享，venv和静态资产可以只读复用。

W42共36训练回合：34个60秒采集边界、2个4秒目标截断，无结果级技术中断或自然物理终止，但材料加载、合法去返/往返均零；最大base x仅0.0245873mm。两次末次E30无材料通行，E60未执行。**零跌倒不等于稳定行走。** 每流新增2048/8真实运行，PPO cap不同却几乎同轨迹；两流累计不能相加成一个独立策略。

2026-11-10原件交付已完成：CPU请求清单各829项；GPU旧29,337,276与新增95,663,446字节包已发布。A旧starter资产保留，完整新名称已入清单。原件/检查点/上传副本保留服务器，不在电脑归档、不自动删除。

## 长期目标与当前层级

研究问题是：**同一世界地块被前足、后足和返程重复加载时，如何用可恢复材料的持久状态，让未来支撑需求影响当前加载、落点与重访时间？** 最终目标为Go2＋NX上的合法完整往返、因果记忆收益、泛化及实机证据。G0/G1是Go2配置名称。

当前冻结因果PANN参数、持续更新地块历史 → PPO Actor/Critic学习2Hz标量速度cap → 全身MuJoCo iLQG → 时变反馈 → 目标角PD → 原Go2/TM02。联合采样/更新/恢复/固定评价/发布已经运行。MPC候选仍未传播完整材料恢复历史，cap无法独立决定足时、落点与载荷；W42完全未加载材料，recovery特征为零，不能证明记忆收益。

历史外部复现、材料辨识、S0、R8刚地与R12联合链结论保留；r24曾有合法去程，R10曾有返程进展。当前r31没有稳定材料通行；旧成绩不能替代当前末次策略原任务资格。

## 为什么这次优先修优化器，而不继续同样大采样

W41有127/130训练回合物理失败；W42变成全程驻留，两者不能混称同一技术报错。qp2和r30完整递推已运行；r31的同输入native接触前向明显改善，但代理方向相对L2仍0.575–3.044，不能称真实接触梯度已对齐。

完整E30揭示更早发生的数值链：t=0的20次startup中，前11次改善站立目标；第12–20次无native下降，λ/rate持续累乘至1e6。两支随后全部1501次非零时刻规划沿用上限，2.002秒真实切outbound也没有清理。四个完整计划的175控制节点Qu约1.5e-4，但H几乎由1e6I主导，k约1.5e-10rad、K约2e-7。所有3042次planner均安装策略，不代表有用运动。

因此先将warm trajectory与每个MPC问题的λ/rate/停止状态分开管理；保留当前native评分和候选/部署身份。相同startup epoch有限重试后停止，新状态/目标epoch才重置。重置1也有动作尺度和重试风险，需要真实干预，不能提前宣称恢复稳定。详见[CPU证据](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W42-final-review/cpu-report.md)。

## 下一轮安排

1. 读取现场，正常实现S5-r32、针对本轮已保存t0/首次outbound输入核验并构建，先推实现/真实父点与新phase。保持物理、奖励、步态、PANN及动作维度；不新增旧QP/纯MPC/刚地/E30准入或smoke/mock长链。
2. 两CPU新w43-a/b分别继承自身W42 ep405/393，完整学习状态和excluded145；每流+512/2到14225/55，是本轮完整目标，不默认追加2048。默认10+10workers，Torch/BLAS各1，按现场余量启动前调整。
3. GPU修复实际缺参和设备数组错误，按已有脚本 --resume-from G5。保持冻结W40-B update37/r30，复用G1/G2，新FP32后端独立实验，不热替CPU。旧MJX 42.288小时的节点计算不重复，不默认补跑相同多日全闭环。
4. 每流末次update55固定E30，前级通过才E60、300/600。合法去程/返程/完整往返三项在E60/300/600分别均≥1/5/10及全部原条件；不以站满或raw圈数过关。
5. 新后台无固定墙钟截止；原样本目标、物理时长、数值guard及网络timeout保留。结束Git先紧凑、原件Release，整合并发远端更新，不强推、不自动删原件、不做电脑归档。

若数值生命周期修正后仍有有效梯度却缺接触探索，再另立低维样条候选/native评分/实际赢家/局部精化机制，不本轮临时全改。新论文包括JMLR2025、Whole-Body MPPI、DIAL-MPC、Reference-Free MPC、Global CI-MPC、MTP与RSS2026 VR-MPPI，具体接口与迁移边界见[文献报告](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W42-final-review/literature-mechanisms.md)。

## 证据、历史与不确定性

- [W42完整当前审查及覆盖](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W42-final-review/README.md)：7770 tracked文件字节清单不冒称每份历史原件语义全读；本轮CPU紧凑记录、完整E30与GPU证据包实际覆盖单列。
- [W41 CPU审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/README.md)、[历史比较](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/history-comparison.md)、[W40完整审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/README.md)。
- [W42原计划](S1_W42_CPU_PLAN.md)、[原启动词](S1_W42_START_PROMPT.md)、[W41 GPU原计划](S1_W41_GPU_PLAN.md)仅作历史，不再视为当前启动指令。
- [原始科学路线](https://github.com/kaiwen123-yang/MCP-RL/blob/4d0af25c7bcea620c7e8c9a4f5b103c4956a3fda/docs/TERRAIN_MEMORY_PLAN.md)。

当前已完成r32正常构建及保存输入干预，CUDA的6fixture/18FD请求已运行。W43联合训练与GPU固定评价依实际job/index推进；稳定材料往返和记忆因果收益仍未建立。
