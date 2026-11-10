# 地面材料记忆主计划：W41 CPU审查与W42接触前向实验

W42两条CPU已完成各自2048条/8更新，分别到13713/53、tail0、excluded145。两条末次update53 E30均运行满30s，以material_traversal_not_completed未通过；E60 blocked_by_predecessor。36个训练回合为34个60s采集边界和2个样本目标截断，结果级技术错误0；材料加载与合法去程/返程/往返均0，最大base x约0.025mm。

W41 GPU于2026-11-10 06:07自行结束，G2完成176/176节点；G4 native/MJX E30均因缺少evaluation-start-unix参数而未进入物理运行，G5因opt_tolerance标量/数组类型错误技术中断。结果与失败原件已发布；不是闭环资格通过。本次只补发布，没有重跑GPU。
[W41完整审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/README.md) · [控制原件复核](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/controller-review.md) · [W42详细计划](S1_W42_CPU_PLAN.md) · [完整启动提示词](S1_W42_START_PROMPT.md) · [执行规则](../AGENTS.md)

## 当前交接

- 当前任务：W42_CONTACT_FORWARD_JOINT / S5-r31 已完成并结束，未获得材料任务资格；不要重复原目标训练或评价。
- w42-a：job-20261108T154719Z-w42-a，13713/53、tail0、excluded145，最后ep405；末次E30满30s未完成材料通行，E60未运行。
- w42-b：job-20261108T154727Z-w42-b，13713/53、tail0、excluded145，最后ep393；末次E30满30s未完成材料通行，E60未运行。
- 任务能力：两支均原地驻留，材料加载及合法去程/返程/往返0/0/0；不以长存活、更新完成或前向误差降低宣称稳定通行。
- 交付：CPU紧凑结果和两份829项原件清单已核验；GPU新增证据包已发布并保留G4/G5技术失败，详见现有运行index和各delivery/evidence.json。
- 位置：autodl-d57:/root/autodl-tmp/MCP-RL-w42为CPU源码与结果；GPU原件仍在/root/autodl-tmp/MCP-RL。全部研究原件/检查点保留，用户自行备份和清理。

## 长期目标与当前层级

研究问题：**同一世界坐标地块被前足、后足及返程反复加载时，怎样利用可恢复材料的持久状态，让未来支撑需求约束当前加载、落点和重访时间？** 最终平台Go2＋NX，逐步形成完整合法往返、因果记忆收益、泛化和实机证据。G0/G1是Go2配置名称。

当前实现为冻结因果PANN参数、持续更新地块历史 → PPO Actor/Critic学习2Hz标量速度cap → 全身MuJoCo iLQG → 时变局部反馈 → 目标角PD → 原Go2/TM02。联合采样、学习、断点继承、末次固定评价与交付真实运行；MPC候选仍未传播完整材料恢复历史，cap也不能独立决定足时、落点与载荷。

历史外部复现、材料辨识、S0局部决策、R8刚地成功与R12联合训练保持原结论。r24曾有4次合法去程，R10曾进入返程；不能说历史从未有进展。当前r30没有合法去程，完整合法材料往返仍未形成。新资格来自当前末次策略原任务评价，不能借用旧刚地或训练最佳前缀。

## 为什么现在换这一处机制

W41不再因QP错误中断，不能继续把物理失败统称技术报错。真实9份plan的J重算误差约1.42e-14、r30自由行Riccati缩放残差约2.73e-15；两条材料前停滞窗口投影完全不改动作，仍有正指令而实际仅约0.01m/s。A590的风险成本为零、Position占53.35%，简单“风险墙/投影吞动作”解释不足。

同起始状态、近乎同目标的三个材料前记录中，平滑规划器刚地Fz约0.007–0.013N，实际约114–247N，2ms后关节速度偏差约0.39–0.44rad/s。独立核对时间、坐标、材料来源与末状态后差异仍成立。它是前向模型缺口，尚未分别隔离solimp/coldstart贡献，也不能解释全部127次失败。

下一轮只分离原生前向与平滑代理导数，保留物理、PANN、动作维度、奖励和步态。旧r17恢复阻抗仍失败，本轮通过明确两个模型角色、原生轨迹上的求导点、零阶锚定和实际候选接受规则区别于该旧开关。相关HOUND、DiffMJX、DiffContact及后续参考/残差机制见[论文整理](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/literature-mechanisms.md)。

## 接下来怎么执行

1. 精确核对现场；独立新CPU工作树，保护GPU将来还会读取的代码/配置和可写构建源。复用venv/只读资产，不重下载全库或归档到电脑。
2. 实现S5-r31、针对既有真实输入核对新模型/求导接线并正常构建。有限差分必须soft-minus-soft，在每个原生名义子步求导；C/D和风险峰来自native轨迹。保留qp2与r30完整递推，不额外clip K或放宽任务。
3. 推送实现/spec及真实父点后，A/B各自新phase+2048/8，双CPU与现有GPU并行。启动并看到真实transition增长后交接，不只给计划。不新增纯MPC/刚地/E30准入、smoke/mock、参数网格或旧QP回放。
4. 每支末次新update固定E30，原同case通过才E60；E60/E300/E600分别要求合法去程、合法返程、完整往返三项均≥1/5/10及所有原条件。原物理时长保留，后台无固定墙钟期限。
5. 紧凑结果先Git，原件继续后台Release；合并并发GPU提交，不强推。核验后不在用户电脑另归档、不自动删除服务器checkpoint/resume/原件。

W42配置已按真实W41父点声明；尚未将声明视为已采样。细节以[W42计划](S1_W42_CPU_PLAN.md)与[启动提示词](S1_W42_START_PROMPT.md)为准。

## 证据与历史

- [W41 CPU总审查与覆盖范围](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/README.md)、[历史分母对照](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/history-comparison.md)、[独立接触核验](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/contact-crosscheck.md)。
- [W40最终审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/README.md)、[完整科学历史](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/history-review.md)、[qp2/恢复/合法返程修复](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/technical-independent-verification.md)。
- [W41 CPU原计划](S1_W41_STABILITY_PLAN.md)、[仍在运行的GPU任务原计划](S1_W41_GPU_PLAN.md)保留历史身份；旧W41启动词不再作为新CPU入口。
- [原始科学路线](https://github.com/kaiwen123-yang/MCP-RL/blob/4d0af25c7bcea620c7e8c9a4f5b103c4956a3fda/docs/TERRAIN_MEMORY_PLAN.md)。

## W42已执行证据

S5-r31已在独立工作树完成实现、正常构建和四个指定保存输入核验。首子步Fz误差约浮点精度；2ms关节速度最大误差A590/A635/B712/B713由0.34467/0.43351/0.38575/0.43711降为0.000756/0.009661/0.008869/0.006646 rad/s。原生held-map方向相对L2误差仍为3.044/2.165/0.575/0.986，不能称精确梯度。两个A输入新C++快照的soft A/B重算一致，十子步P/Q误差不超过1.78e-15，原生峰位置/值一致，J重算误差不超过1.39e-17。B712/B713在未恢复历史warm-start/私有phase的保存状态规划中backward_failed，保留原记录、不重试；这不是新任务回合或正式评价。

启动源码4dcc1bb已推送并核对远端；两流均观测到11665→11666→11668。指定输入原件及r31二进制已经发布并按远端size/digest核验：https://github.com/kaiwen123-yang/MCP-RL/releases/tag/w42-contact-forward-20261108 。本轮训练原件继续保留D57，结束后的自动紧凑Git及Release尚未开始。GPU PID585559保留，G2已记录60节点；其他GPU PID599069/599252继续运行。精确job/clock/progress与最新持久断点见运行index的w42_startup；当前样本tail为活动内存状态，不能把它误称已经写入初始resume文件。

## 2026-11-10交付收口

2026-11-10交付：W42-A/B当前请求清单各829项已在远端按名称、uploaded状态和大小核对，无遗漏。A两项上传中断的starter资产保留，完整内容用新名称补传；清单已指向新资产。GPU旧29,337,276字节包和新增95,663,446字节包均已发布，新增包SHA256与远端digest一致。原件、检查点及上传副本保留D57；不在电脑归档，不自动删除。
