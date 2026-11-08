# 地面材料记忆主计划：W41 CPU审查与W42接触前向实验

审查基线：[9cc42d5](https://github.com/kaiwen123-yang/MCP-RL/commit/9cc42d5a9189a2bfd7e1e6cd99b9300045a16724)，2026-11-08。**两条W41 CPU联合任务已完成，工程采样/更新/末次评价能够收口；稳定材料通行仍未建立。GPU按用户说明仍在运行，保持现有任务。** 下一轮聚焦原生接触前向与平滑局部导数分工；S5-r31尚未实现/构建/运行。

[W41完整审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/README.md) · [控制原件复核](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/controller-review.md) · [W42详细计划](S1_W42_CPU_PLAN.md) · [完整启动提示词](S1_W42_START_PROMPT.md) · [执行规则](../AGENTS.md)

## 当前交接

- 已完成任务：W41_MATCHED_BOUND_RICCATI_JOINT / S5-r30。A为`job-20261107T100149Z-w41-a`，B为`job-20261107T093906Z-w41-b`，各新增2048/8，各自累计11665/45、tail0、excluded145；末episode分别387/375。
- 任务能力：130回合中127物理失败、3合法截断、0结果级技术中断；合法材料去程/返程/往返均0。末次固定E30在14.100/28.408s跌倒，E60因前级未通过未执行。自然失败中位存活14.422s，最远训练base x1.0573m，尚未穿出东界2.2m。
- 已交付：两个CPU完整紧凑结果，及W41共24ZIP/12608源文件原件，Release `w41-raw-24zip-20261108T115134Z`。GPU当前进度以自身现场记录为准，本审查不把未完成结果记为最终成绩。
- 当前新工作：W42_CONTACT_FORWARD_JOINT / S5-r31计划已明确，尚未实现/构建/运行。nominal/candidate/cost用原生geom/pair接触，独立平滑A/B沿原生轨迹零阶锚定；其数学、验证范围和风险详见W42计划。
- 下一父点：A `training/w41-a/PANN_PPO_S5-seed1/resume-state-ep00387.pt`，B `training/w41-b/PANN_PPO_S5-seed2/resume-state-ep00375.pt`；路径前缀`outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/`。每支+2048/8，当前目标13713/53；完整状态与excluded145保留，更晚有效同流父点需据实重算。
- 执行位置：D57活动GPU根`/root/autodl-tmp/MCP-RL`不改；新CPU建议独立`/root/autodl-tmp/MCP-RL-w42`，包括可写upstream/build。新CPU建议10+10 workers，GPU作为第三任务继续。后台deadline=null；推GitHub核验后不在电脑归档，不自动删服务器原件。

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

W42尚未运行，因此当前R12运行JSON保留真实W41身份，不提前把计划写成实际状态。细节以[W42计划](S1_W42_CPU_PLAN.md)与[启动提示词](S1_W42_START_PROMPT.md)为准。

## 证据与历史

- [W41 CPU总审查与覆盖范围](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/README.md)、[历史分母对照](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/history-comparison.md)、[独立接触核验](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/contact-crosscheck.md)。
- [W40最终审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/README.md)、[完整科学历史](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/history-review.md)、[qp2/恢复/合法返程修复](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/technical-independent-verification.md)。
- [W41 CPU原计划](S1_W41_STABILITY_PLAN.md)、[仍在运行的GPU任务原计划](S1_W41_GPU_PLAN.md)保留历史身份；旧W41启动词不再作为新CPU入口。
- [原始科学路线](https://github.com/kaiwen123-yang/MCP-RL/blob/4d0af25c7bcea620c7e8c9a4f5b103c4956a3fda/docs/TERRAIN_MEMORY_PLAN.md)。
