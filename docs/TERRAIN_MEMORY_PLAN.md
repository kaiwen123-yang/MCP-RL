# 地面材料记忆主计划：W40 最终审查与 W41 稳定执行

日期：2026-11-07，北京时间。当前结论：完整联合学习已经实际运行，稳定合法材料通行尚未建立；最新技术故障有独立 qp2 数值修复，GPU 支线尚未对齐。先完成 W40 A 的同任务续接，并行核对真实饱和保持动力学、实施有证据的单一控制修复，再推进 W41 两条完整联合续训。

[W40 最终审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/README.md) · [完整历史评估](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/history-review.md) · [W41 详细计划](S1_W41_STABILITY_PLAN.md) · [GPU扩展实验](S1_W41_GPU_PLAN.md) · [可直接使用的启动提示词](S1_W41_START_PROMPT.md) · [执行规则](../AGENTS.md)

## 当前交接（执行后只更新这6行）

- 当前任务：W40_STABILITY_WITH_PARALLEL_GPU_MPC / S5-r29；后台按各自样本/更新目标结束，无固定墙钟完成期限；原物理时长与技术失败停止保留。
- w40-a：job-20261106T103028Z-w40-a，launched；新增915/3，累计9508/36，tail=147、excluded=145；评价=pending_training，紧凑Git=pending_computation。
- w40-b：job-20261106T103028Z-w40-b，training_ended；新增1024/4，累计9617/37，tail=0、excluded=145；评价=completed，紧凑Git=published。
- 任务能力：w40-a尚无最终任务汇总；w40-b合规去程/返程/整圈=0/0/0，材料加载回合=26。每流仅本phase最后实际新update评价E30，同case合格才E60；训练更新不等于合规任务能力。
- 交付：两流compact finish/Git串行，使用原index锁；全部逐回合JSON、2Hz学习转移、12维PANN/value小表与E30关键事件窗先Git；已请求的原始反馈/规划/参考轨迹及必要数组由diagnostics入口后台Release，未选权重/全视频留D57，无自动删除。
- 续接位置：autodl-d57:/root/autodl-tmp/MCP-RL；各job、training目录与有效resume见正常job/index的两流状态；分别继承自身父分支，optimizer/RNG/归一化不混合。

## 长期目标与当前层级

研究问题是：**同一世界坐标地块被前足、后足及返程反复加载时，怎样利用可恢复材料的持久状态，让未来支撑需求约束当前加载、落点和重访时间？** 最终平台 Go2＋NX，逐步形成完整合法材料往返、因果记忆收益、泛化和实机证据。G0/G1 是 Go2 配置名称。

当前实现为冻结因果 PANN 参数、持续更新地块历史 → PPO Actor/Critic 学习 2Hz 标量速度 cap → 全身 MuJoCo iLQG → 时变局部反馈 → 目标角 PD → 原 Go2/TM02。采样/更新/末次固定策略评价/自动交付均真实运行；MPC 候选尚未传播完整材料恢复历史，cap 还不能独立决定足时、落点与载荷。

历史已完成的外部复现、材料辨识、S0 局部决策、R8 刚地成功和 R12 联合训练都保留有效结论及边界。r24 曾有合法去程，不能说历史从未做到；完整合法往返仍未形成。新的主线资格必须来自当前版本末次策略在原任务上的评价，不能借用旧刚地或训练最好前缀。

## 接下来怎么执行

1. 构建 qp2，普通 resume W40 A，保留147合法tail和全部学习状态，补完原目标后E30→有条件E60；B已完成不重跑。
2. 与A并行核对同输入饱和actuator/十子步/A-B与有限差分一致性，据证据实施单一r30；整段未裁剪PD可行投影只是候选，不预认定为根因修复。保持物理、PANN、动作维度、奖励和步态；必要针对性数值检查后直接完整联合实验，不再拆成层层防御性准入。
3. 在各自tail0父点新建 W41 两条 phase，每支建议新增2048条/8更新；默认2×12 workers，第三GPU任务按G1–G5扩展真实状态、驻留批量、候选/整planner和独立闭环实验，见GPU专项计划；保留另一项在跑的GPU训练。
4. 所有后续后台取消预设墙钟完成期限；保留物理评价时长、数值迭代与单次网络超时。GitHub交付核验后不在用户电脑再归档，服务器原件由用户备份后自行删除。

具体数学、实施边界、终止/资源/交付语义以 [W41计划](S1_W41_STABILITY_PLAN.md) 和 [启动提示词](S1_W41_START_PROMPT.md) 为准。当前 R12 配置保留 W40实际谱系，待现场真实父点确认并实现r30后再声明W41，不预写虚假运行状态。

## 可追溯证据

- [本次总审查与全部数据入口](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/README.md)：W40回合、原始执行、学习、GPU、源码阅读覆盖与论文。
- [具体技术故障及修复](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/technical-error.md)：真实QP、零步长循环、原容差回放。
- [独立技术复核与新增修复](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/technical-independent-verification.md)：真实12例重跑、恢复文件保护、新长阶段合法返程判据与无截止调用链。
- [W39原始审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-raw-review/README.md)：既有前馈/反馈、材料/学习与r29数学纠偏。
- [W40最初退出审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-stop-review/README.md)：qp1修复的前两次问题，与这次qp2的新故障区分。
- [本次替换前的完整主计划](https://github.com/kaiwen123-yang/MCP-RL/blob/c438f5463659ab7c8d3a7f5eced251639e14b2ef/docs/TERRAIN_MEMORY_PLAN.md)：保留在Git历史，旧截止时间/断点不作为当前执行指令。
- [原始科学路线](https://github.com/kaiwen123-yang/MCP-RL/blob/4d0af25c7bcea620c7e8c9a4f5b103c4956a3fda/docs/TERRAIN_MEMORY_PLAN.md)：长期材料记忆与跨步协同问题的来源。
