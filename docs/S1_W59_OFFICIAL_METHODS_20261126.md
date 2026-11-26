# W59 官方 SuperMimic 与近期方法：迁移点和创新边界

核验日期：2026-11-26，Asia/Shanghai。只使用官方论文、作者源码及作者项目页。用户个人SuperMimic复现不作为本项目依据。

## 1. 官方版本和实际方法

- [SuperMimic论文v3](https://arxiv.org/pdf/2610.28480v3)：2026-11-19修订，CoRL2026论文。
- [官方源码固定版本](https://github.com/HybridRobotics/supermimic/tree/5fb92297ab4fe3d86629d9f0c635334791bb0d5f)：2026-11-21，Apache-2.0；本次没有安装或运行它的训练环境。
- [作者项目页](https://supermimic.github.io/)。

官方使用当前策略的模拟器转移训练一个共享可微世界模型，再在短想象轨迹上以回报和critic更新策略。梯度经过学习模型，不经过MuJoCo。它的任务是G1参考动作跟踪；本项目的动作是经MPC/WBC及硬门执行的高层请求，因此必须重建状态、动作、模型输出与奖励接口。不能转用G1权重或把论文训练速度当作Go2/TM02预期收益。

## 2. 源码可落实的迁移点

以下路径相对于官方固定commit的 `src/supermimic/`；文件名/类名用于复核，不代表已复制进MCP-RL。

| 官方实现 | 读到的关键机制 | W59迁移决策 |
|---|---|---|
| `representations.py` / GenCoord | 预测速度增量，再按匹配离散化积分状态 | 已知运动学/动力学承担可确定部分，材料和执行残差补未知部分 |
| `learners/wm_learner.py` | 多步自回归拟合；配置8指8状态帧、7次受监督预测 | 配置同时记录状态数、转移数、物理秒，特别监督接触与重载 |
| `learners/policy_learner.py` | actor更新时固定WM参数、保留输入导数；短rollout加critic | WM+真实任务critic，先验证实际动作方向，再开BPTT |
| `envs/world_model_env.py` | 想象状态生成观测、奖励与终止 | 原任务保持计时、hard failure和truncation明确建模，不照抄软终止 |
| `runners/supermimic_runner.py` | 当前策略收集、回放和模型/策略更新循环 | 保留真实D57采集，GPU上复用数据；控制器版本与macro request加进replay |
| `learners/learner_utils.py` | 当前代码含Muon/Adam与CUDA autocast | 不为了复现默认配置覆盖稳定环境；微小材料差先用FP64/FP32参考 |

官方32×20ms约0.64s，不能直接解释为32个跨足宏事件。我们在模型子步中持有当前请求，合法事件才重新发动作。

当前main与论文表格也不完全等价：默认`rep_state/history_length=1`与论文部署的proprio历史不同；优化器/精度/训练迭代默认也有变化。必须pin具体recipe，不以“clone成功”宣称完整复现论文设置。源码实读范围及需重写接口见[官方代码专项记录](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W58-review/official-code-audit.md)。

## 3. 最近邻：每篇具体改变什么决策

| 论文与核验状态 | 方法要点 | 对项目的约束/用途 |
|---|---|---|
| [DynaMimic](https://arxiv.org/html/2611.32944v1)，2026-11-24预印本；全文 | 学接触wrench和广义力残差，接解析Lagrangian动力学 | “物理动力学＋接触残差＋SuperMimic”已有直接近邻；应比较持久材料恢复、跨足记忆与执行门 |
| [Trust What Repeats](https://arxiv.org/html/2611.27648v1)，2026-11-20预印本；全文 | 检验模型策略梯度的重复性与模拟器兑现，按rollout限制极大梯度 | 预测MSE不是梯度证书；可信梯度/clip本身也不是原创 |
| [DiffContact](https://diffcontact.github.io/)，[论文](https://arxiv.org/abs/2610.35852)，2026；作者方法页/摘要 | 保持硬接触前向，仅在反向构造平滑接触敏感度 | 可学习其区分前向物理与反向近似的做法；其刚性接触推导不能直接当TM02材料导数 |
| [LocoWM](https://arxiv.org/abs/2609.39179)，2026-09-30预印本；摘要/官方项目与代码 | 动作条件未来状态预测，分阶段训练残差适配器 | 证明“稳定基线＋预测＋残差”已有；我们需说明跨足物理记忆与MPC权限差异 |
| [Coupled Local and Global World Models](https://coupled-global-local-wm-rl.pages.dev/)，CoRL2026；作者方法页 | 高保真前向与轻量局部反向分离，修补模型被策略利用的错误 | “两模型融合”并非空白；当前项目不需要因此增加视频生成模型 |
| [RWM-U](https://arxiv.org/html/2504.16680v3)，v3为2026-01-08；全文方法 | ensemble不确定性沿自回归传播，与惩罚式模型PPO结合 | WM+PPO必须有合理比较；ensemble一致仍不等于误差校准 |
| [DMO](https://proceedings.mlr.press/v305/amigo25a.html)，CoRL2025；官方摘要/作者页 | 真实模拟器前向、学习模型反向，真实数据critic | 只有局部导数可信、长前向漂移主导时才适合作过渡，不保证解决局部错误梯度 |
| [PWM](https://arxiv.org/html/2407.02466v3)，ICLR2025；方法章节 | 模型平滑性、预测误差和策略优化收益不等价 | 用真实优化收益评价模型，不能只比拟合误差 |
| [Learning Deployable Locomotion Control via Differentiable Simulation](https://proceedings.mlr.press/v305/schwarke25a.html)，CoRL2025；官方摘要 | 为四足接触提供可学习导数并做实机迁移 | 接触梯度本身有成熟近邻；不以软化真实接触换容易训练 |

近邻工作含不同机器人、任务与算力，不能把各篇提升百分比跨任务拼在一起。预印本与正式会议状态分别注明；未查验分区，不给期刊档次承诺。

## 4. 项目可以提出但尚未证明的贡献

1. **模型贡献：**可恢复world-cell材料记忆与控制器内部记忆组成可观测belief，区分材料恢复误差和执行接触误差；同一个预测算子面向跨足长期回访。
2. **规划贡献：**每个候选自己的加载/恢复轨迹影响另一足未来承载；收益经硬门后的实际动作证实，而非只看优化器内代价。
3. **学习贡献：**以真实动作兑现和接触事件为校准依据，选择可用于策略更新的短想象；与同权限PPO比较真实数据效率。
4. **联合贡献：**M/P/R共享状态、模型和版本，交替采集/拟合/优化，比较固定WM、私有历史及冻结R，证明超过模块拼接。

这些都必须有新实验支撑，不能只靠架构命名。W58的负结果要求我们先处理真实动作和模型尺度，任何论文都不能替项目补出尚未观察到的材料历史效应。详细执行、数据分割、预算和对照在[W59计划](S1_W59_RESEARCH_PLAN.md)。
