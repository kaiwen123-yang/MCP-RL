# 地面材料记忆主计划：W43 最终驻留审查与 W44 运动优化

## W44 当前执行位置（2026-11-10）

S5-r33 已实施、正常构建并完成四问题有限比较，已冻结主机制 **N**：与 S 使用同一曲率度量 D，改用原生接触 FD 方向，完整重算约束 Riccati并由native评分决定安装。**W44双流已声明，尚待本轮推送核验后启动。**

两个4.00s问题中，N以更低原生成本给出+2.864/+2.620mm的0.35s预测前移；S仍后退。N晚段仍后退且不优于S，所有四个N计划均四足承载。这仅支持进入真实双流验证，不构成摆步、闭环起步、材料通行或记忆收益。

四例参数/权重/步态上下文一致，S/N的D逐字节一致，16安装策略完整身份与实际评分匹配。有效16调用有480次native rollout、113.092s；另16次重建selection索引错误造成的无效上下文调用（490次rollout、111.994s）完整保留并排除，旧物理回合重跑0。

两流分别继承自身W43 ep00410/ep00398，各14225/55、tail0、excluded145，目标14737/57（+512/+2），workers10+10，deadline=null。CPU使用/root/autodl-tmp/MCP-RL-w42私有可写source/build；现有GPU继续原作业，不重启、不热改。新W44网络声明direct，因为本次GitHub API的Clash/academic失败、direct成功；只改变新作业shell。

[四问题完整证据](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w44-mechanisms/summary.json) · [r33规格](../experiments/terrain_memory/s1-go2-r11-S5-r33.json) · [运行索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/index.json)

> 以下W43审查和W44原始授权保留形成时语境；其中“尚未实施/尚未声明”不再代表最新状态。以本页上方和运行索引为准。

**2026-11-10：W43/S5-r32 两条 CPU 已完成训练和末次固定评价。数值生命周期修复有效，起步与材料通行仍未建立。W44/S5-r33 是下一轮待实施方案；现有 GPU 继续运行。**

[W43 最终原件审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W43-final-review/README.md) · [CPU 训练与评价](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W43-final-review/cpu-report.md) · [求解器与成本](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W43-final-review/solver-report.md) · [本轮文献](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W43-final-review/literature-mechanisms.md) · [W44 实施计划](S1_W44_CPU_PLAN.md) · [完整启动提示词](S1_W44_START_PROMPT.md) · [执行规则](../AGENTS.md)

## 当前交接

审查基点为 `6a94c94c8e21e3b50fcca40ab4363165dee2987e`，2026-11-10 北京时间 20:52:55 的最终 CPU 交付。`1334f76` 和索引 `w43_progress_observation` 记录的是 update54 中途观察；它们的 live 数字和 GPU 时间点保留为历史，不再描述 CPU 最终状态。

| 工作流 | 已完成的真实状态 | 下一起点 |
|---|---|---|
| W43-A / seed1 | `job-20261110T035658Z-w43-a`；各自 +512/2 到 14225/55，tail0、excluded145，ep00410；update55 E30 满30秒但材料通行失败 | 自身 ep00410；W44 各 +512/2 到 14737/57 |
| W43-B / seed2 | `job-20261110T035700Z-w43-b`；各自 +512/2 到 14225/55，tail0、excluded145，ep00398；update55 E30 同样失败 | 自身 ep00398；W44 各 +512/2 到 14737/57 |
| 现有 GPU | 用户最新确认仍进行；历史 g5 invocation 与冻结 W40-B/update37/r30 契约保留 | 继续自己的现有作业；读取现场后交接，不重启 G5、不热换环境 |

W43 共 10 个训练回合：8 个 60 秒采集边界、2 个约 16 秒样本目标截断，无结果级技术中断或自然物理终止。两流各自末次 update55 E30 `error=null`，`stage_status=task_criteria_unmet`，原因 `material_traversal_not_completed`；E60 `blocked_by_predecessor`。训练样本达标、评价正常结束与任务合格分别报告。

两次最终 E30 出发段平均命令 vx 为 +0.22178/+0.20627 m/s，实际净位移为 −13.2415/−13.2178 mm；四足持续承载、无抬落足，最大足心 x 约 .19216 m，仍未到材料起点 .8 m。材料加载、世界地块写入及合法去程/返程/往返全零。**当前获得的是驻留运行，没有获得稳定行走。**

实际核对了两份 Release 的 391+388=779 项、5,357,128,793 字节。CPU 审查下载并读取 52 项，包括 12 完整轨迹、28,850 次 planner calls、24 个材料日志、4 个 rollout，另核对 Git 的 1024 条新增 transition 及对应 PANN/value；技术审查实际读取 46 项、34 个完整 accepted/reference/H/K 事件矩阵，部分资产重叠。并未语义复算全部 5.36 GB、全部物理子步或未发布非赢家轨迹。覆盖与哈希逐项列在审查目录，不能把资产清单核验等同全量重算。

CPU 现有树为 `autodl-d57:/root/autodl-tmp/MCP-RL-w42`，GPU 为 `/root/autodl-tmp/MCP-RL`。先核实现场；CPU 已结束且干净可复用，GPU 可写 upstream/build 与新 CPU 隔离。根 r12 当前仍声明 W43，**下一轮必须实施 r33 并显式声明 W44，不能直接用旧配置新开同名任务。**

## 长期目标与当前层级

研究问题是：**同一世界地块被前足、后足和返程重复加载时，如何用可恢复材料的持久状态，让未来支撑需求影响当前加载、落点与重访时间？** 最终目标为 Go2＋NX 上的合法完整往返、因果记忆收益、泛化及实机证据。G0/G1 是 Go2 配置名称。

当前冻结因果 PANN 参数、持续地块历史接口 → PPO Actor/Critic 学习 2 Hz 标量速度 cap → 全身 MuJoCo MPC → 时变反馈 → 目标角 PD → 原 Go2/TM02 的联合采样、更新、恢复、固定评价、发布链已经运行。本轮 4 个 rollout、1024 条新样本核对有限且一致，未见 PPO 数值崩溃；但 cap 只能在现有权限内改变速度请求，不能直接决定足时、落点、卸载和载荷。当前/next 的 12 维 PANN 恢复特征仍全零。

MPC 候选未完整传播材料恢复历史；W42/W43 又没有实际加载材料，无法检验记忆作用。先获得可靠的起步、材料通行，再构造真实接触历史、进行冻结/打乱/无记忆等因果对照，最后扩展落点/重访权限及实机。当前不能跳到“记忆方法有效”的论文结论。

历史外部复现、材料辨识、S0、R8 刚地与 R12 联合链结论保留；r24 曾有合法去程、R10 曾有返程进展。当前 r32 没有稳定材料通行，旧成绩不能替代当前末次策略原任务资格。

## 失败为何变化，下一步为何修运动优化

| 轮次 | 训练回合 | 技术中断 | 自然物理失败 | 其余结束 |
|---|---:|---:|---:|---|
| W40 | 64 | 3 | 60 | 1 样本目标截断 |
| W41 | 130 | 0 | 127 | 3 样本目标截断 |
| W42 | 36 | 0 | 0 | 34 个60秒采集边界 + 2 个目标截断 |
| W43 | 10 | 0 | 0 | 8 个60秒采集边界 + 2 个16秒目标截断 |

各轮控制器、父点和数据分布不同，这不是同分布成功率曲线。W40/W41 的中断/跌倒、W42/W43 的不前进、单次 planner 的 `native_search_stagnation` 是不同层面的失败。

W42 曾在 t=0 重复 startup 中将 λ/rate 累积到 λ=1e6，并让全部后续非零时刻继承；r32 已修复这个明确的数值生命周期错误。W43 全 28,850 次调用中，新 epoch 第一尝试均为 λ/rate=1/1；相同停止 epoch 不再领取重试预算。因此本轮不能继续把驻留归咎于旧上限继承。

新原件将问题定位得更具体：

1. **规划本身没有行走。** 34 个保存 accepted 计划全部预测后退、节点中无足部离地。32 个有真实未来数据的计划，首20ms base误差最高约2.30e-8m；另两个是30s终端未来计划，仅可检查安装映射。末次动作没有QP失败或投影修正，当前驻留窗口不支持“计划想走、执行破坏”的解释。
2. **绝对阻尼与运动曲率尺度不匹配值得优先验证。** λ=1占已存H对角约99.86%–99.93%；Stand/运动Effort直接控制曲率约差1736倍。不能据此承诺减λ就会走，但可以据此开展完整Riccati下的尺度化比较。
3. **proxy与native下降不一致仍存在。** 最终出发段accepted的所选α比率中位约.236，全部低于.5，accepted后λ只有1/2，没有.5。不能说成功后已经降阻尼被重置，也不能说更多相同迭代自然解决。
4. **局部成本改善没有形成换支撑。** 早期Position约占A成本66.7%，目标没被关闭；末端一次局部改善约97.6%来自MaterialSupportHeight，Gait改善约1.25e-9。该支撑项刚地运动时也启用，不是材料受载证据。缺少非赢家全轨迹，尚不能断言目标全局偏好驻留。
5. **候选保留是小的契约问题。** 最终出发段A3/B4次复用错过更便宜已算候选，最大改善不足1e-9。应修完整策略最佳候选保留，但不当作当前驻留主因。

精确字段、完整成本复算和适用边界见[求解器报告](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W43-final-review/solver-report.md)。

## W44 的具体安排

1. 用两流 E30 的四个已有真实保存问题：call221/4.00s、call1517/29.92s。共同上下文比较尺度匹配正则 S、同尺度 native 局部方向 N、32槽结构候选 P。完整重算约束 Riccati/native，逐例记录成本与运动。不做宽泛网格，不把缺失warm context冒充精确重放；必要时在共同重建快照补 r32 单次局部基准。
2. 只将一种得到证据支持的主机制实施为 r33。若有限对照仍无有效运动，形成一个有证据的控制/参考/目标修订再训练，不继续原样增加PPO。不要同时改物理、奖励、步态和动作权限。
3. 声明W44真实父点、phase、clock、源码/产物，先推送后运行。两流分别从自身W43 ep00410/00398、14225/55继续，各+512/2到14737/57，tail0/excluded145及完整学习状态保留。r33当前尚未实施，W44尚未运行。
4. 默认两CPU各10workers + 现有GPU，Torch/BLAS各1；按现场实际容量预留GPU host/系统/发布。完成构建后再双流重负载，不能将25核/96GB总容量当空闲量。保留GPU自己的冻结契约与可写目录，不重启G5。
5. 各自末次update57固定E30，前级合格才同case E60/300/600；合法去程、返程、完整往返三项在E60/300/600分别均≥1/5/10及全部原条件。关键窗口增加非赢家预测、真实成本和身份，继续完整请求原件发布。
6. 新后台无固定墙钟截止。样本目标、物理时长、数值预算与网络timeout保留。紧凑Git先发、原件Release后发；串行整合并发CPU/GPU远端更新，不强推、不重复旧包、不做电脑归档、不自动删除服务器断点和原件。

本轮新文献覆盖2025/2026的ROM-guided CI-MPC、IDTO、Reference-Free、GPC、MPX、FS-MPC、TBM，以及HiLQR/控制受限DDP和官方LM文档。借鉴落在尺度、接触方向、结构参考、真实候选保留和批量计算上；论文中的低层控制/动作接口不同，不直接移植成绩。详见[迁移边界](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W43-final-review/literature-mechanisms.md)。

## 历史证据入口

- [W42完整审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W42-final-review/README.md)：含当时tracked字节清单及CPU/GPU实际读取范围，不是本轮全仓库逐字语义审阅声明。
- [W41审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/README.md)、[历史比较](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/history-comparison.md)、[W40审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/README.md)。
- [W43原计划](S1_W43_STABILITY_PLAN.md)、[原启动词](S1_W43_START_PROMPT.md)、[W42原计划](S1_W42_CPU_PLAN.md)、[W41 GPU原计划](S1_W41_GPU_PLAN.md)保留历史身份，不再作为重复启动指令。
- [原始科学路线](https://github.com/kaiwen123-yang/MCP-RL/blob/4d0af25c7bcea620c7e8c9a4f5b103c4956a3fda/docs/TERRAIN_MEMORY_PLAN.md)。
