# 地面材料记忆主计划：W41 CPU审查与W42接触前向实验

审查基线：[9cc42d5](https://github.com/kaiwen123-yang/MCP-RL/commit/9cc42d5a9189a2bfd7e1e6cd99b9300045a16724)，2026-11-08。**两条W41 CPU联合任务已完成，工程采样/更新/末次评价能够收口；稳定材料通行仍未建立。GPU按用户说明仍在运行，保持现有任务。** 下一轮聚焦原生接触前向与平滑局部导数分工；S5-r31已完成独立实现/正常构建/指定保存输入核验；两条W42联合训练已启动并确认真实transition增长。

[W41完整审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/README.md) · [控制原件复核](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/controller-review.md) · [W42详细计划](S1_W42_CPU_PLAN.md) · [完整启动提示词](S1_W42_START_PROMPT.md) · [执行规则](../AGENTS.md)

## 当前交接

- 当前任务：W42_CONTACT_FORWARD_JOINT / S5-r31；后台按各自样本/更新目标结束，无固定墙钟完成期限；原物理时长与技术失败停止保留。
- w42-a：job-20261108T154719Z-w42-a，running；PID 654367 / MPC 654376，ep388；23:50:26快照新增3/0，累计11668/45，tail=3、excluded=145；评价pending_training。
- w42-b：job-20261108T154727Z-w42-b，running；PID 655195 / MPC 655203，ep376；23:50:26快照新增3/0，累计11668/45，tail=3、excluded=145；评价pending_training。
- 任务能力：w42-a尚无最终任务汇总；w42-b尚无最终任务汇总。每流仅本phase最后实际新update评价E30，同case合格才E60；训练更新不等于合规任务能力。
- 交付：两流compact finish/Git串行，使用原index锁；全部逐回合JSON、2Hz学习转移、12维PANN/value小表与E30关键事件窗先Git；已请求的原始反馈/规划/参考轨迹及必要数组由diagnostics入口后台Release，未选权重/全视频留D57，无自动删除。
- 续接位置：autodl-d57:/root/autodl-tmp/MCP-RL-w42；各job、training目录与有效resume见正常job/index的两流状态；分别继承自身父分支，optimizer/RNG/归一化不混合。

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
