> **历史文件，W44已于2026-11-12完成。** 以下保留形成时原文，不再作为新启动指令。当前请使用[W45计划](S1_W45_STABILITY_PLAN.md)和[W45启动提示词](S1_W45_START_PROMPT.md)。

# W44 完整 Codex 启动提示词

## W44 当前执行位置（2026-11-10）

S5-r33 已实施、正常构建并完成四问题有限比较，已冻结主机制 **N**：与 S 使用同一曲率度量 D，改用原生接触 FD 方向，完整重算约束 Riccati并由native评分决定安装。**W44双流已在原job/clock下普通续接运行，已观察真实transition持续增长。不要再次使用--new-phase。**
首次两个启动均在0物理步因artifact身份丢失17项父配置而中断，错误及gdb依据已保留。完整父配置继承现已恢复，同一二进制未改变；修正身份和失败原件见Release补充包。每流当前进展与磁盘续接点不同，实时计数见运行索引w44_live_observation；update57固定评价尚待训练目标完成。


两个4.00s问题中，N以更低原生成本给出+2.864/+2.620mm的0.35s预测前移；S仍后退。N晚段仍后退且不优于S，所有四个N计划均四足承载。这仅支持进入真实双流验证，不构成摆步、闭环起步、材料通行或记忆收益。

四例参数/权重/步态上下文一致，S/N的D逐字节一致，16安装策略完整身份与实际评分匹配。有效16调用有480次native rollout、113.092s；另16次重建selection索引错误造成的无效上下文调用（490次rollout、111.994s）完整保留并排除，旧物理回合重跑0。

两流分别继承自身W43 ep00410/ep00398，各14225/55、tail0、excluded145，目标14737/57（+512/+2），workers10+10，deadline=null。CPU使用/root/autodl-tmp/MCP-RL-w42私有可写source/build；现有GPU继续原作业，不重启、不热改。新W44网络声明direct，因为本次GitHub API的Clash/academic失败、direct成功；只改变新作业shell。

[四问题完整证据](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w44-mechanisms/summary.json) · [r33规格](../experiments/terrain_memory/s1-go2-r11-S5-r33.json) · [运行索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/index.json)

> 以下W43审查和W44原始授权保留形成时语境；其中“尚未实施/尚未声明”不再代表最新状态。以本页上方和运行索引为准。

日期：2026-11-10（北京时间）。与 [W44 实施计划](S1_W44_CPU_PLAN.md) 配套。当前仓库只交付 W43 结果审查和 W44 方案；**S5-r33 尚未实现，W44 尚未声明或启动**。以下整段用于交给可连接 D57 的 Codex，授权其完成具体实施、针对性比较、推送及真实双流运行。

---

请读取 `kaiwen123-yang/MCP-RL` 最新 main、`AGENTS.md`、`docs/TERRAIN_MEMORY_PLAN.md`、`docs/S1_W44_CPU_PLAN.md`，以及 `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W43-final-review/` 的 README、cpu-report、solver-report、literature-mechanisms 和两份 summary。然后直接完成 W44 的控制器实施、针对性原因判别、代码与身份推送、双 CPU 联合实验及交付；不要只复述计划，也不要重新开启一串与当前问题无关的检查。

## 一、真实起点

W43/S5-r32 两 CPU 已结束，基点 `6a94c94c8e21e3b50fcca40ab4363165dee2987e`：

- A：`job-20261110T035658Z-w43-a`，末 ep00410；B：`job-20261110T035700Z-w43-b`，末 ep00398。
- 两流各从 13713/53 新增 512/2，到 14225/55，tail0、excluded145。
- 10 训练回合为 8 个 60 秒采集边界 + 2 个 16 秒样本目标截断，结果级技术中断和自然物理失败均为 0。
- 各自末次 update55 固定 E30 满 30 秒，却都 `material_traversal_not_completed`，E60 blocked；出发段后退约 13.24/13.22 mm，四足全程承载，没有抬落足，材料加载/合法去返/往返全零。
- 28,850 次完整 planner calls 证明 r32 新 epoch λ/rate=1/1 重置有效；不能继续把本轮失败解释为沿用 W42 的 λ=1e6。
- 全部 34 个保存完整计划本身预测后退；其中 32 个有真实未来 20 ms，base 误差最高约 2.30e-8 m。末次评价无 QP 失败/投影修正。当前优先修运动优化，不能无证据重做 PD/QP。
- 已存 H 中 λ=1 占对角约 99.86%–99.93%，Stand/运动 Effort 直接曲率约差 1736 倍。实际 chosen-alpha native/proxy 改进比中位约 .236，全部低于 .5，accepted 后从未产生 λ=.5。末端局部改善约 97.6% 来自 `MaterialSupportHeight`，并不代表材料受载。以上是机制依据，不是“降低 λ 必然起步”的证明。
- W43 请求原件实际已发布 779 项，不重复下载或重传全包。只取当前判别需要的原件；分清全部资产发布核验与实际复算范围。
- `1334f76` / `w43_progress_observation` 是中途快照，不再当最终 CPU 状态。根 r12 配置当前仍为已结束的 W43，不能直接用旧配置 `--new-phase`。

GPU 用户确认仍运行。默认现场 `ssh autodl-d57`，CPU 树 `/root/autodl-tmp/MCP-RL-w42`，GPU 树 `/root/autodl-tmp/MCP-RL`。先精确读取 tmux/PID/job/clock、当前 GPU、工作树未提交修改与最新有效父点。已有有效 W44 就接续自己的 job，不重复启动；不要重启 G5、热换 GPU 后端、改变其冻结 W40-B/update37/r30 契约或停止其他任务。

## 二、先完成有界原因判别，再冻结一个主改变

使用两流最终 E30 已发布的 `r32-search-call000221.npz`（4.00 s）与 `r32-search-call001517.npz`（29.92 s），共四个真实问题，连接对应 incumbent/accepted-reference/K/H/Q/模型与实际 readback。晚段 1517 输入 A plan1380 / B plan1378 的完整矩阵已发布。首次 outbound 2.002 s 缺完整上下文，不冒称有精确输入。

恢复可获得的 state/time/userdata、任务参考/权重、策略与步态上下文。缺失私有历史明确记录；若无法完整还原，所有方法从同一个重建快照开始，并补一次当前 r32 局部基准，不拿历史不同上下文数值做因果对照。这不需要重跑旧物理回合。

比较三种方向：

1. **S，优先方案：曲率尺度匹配的正则。** 保留 r31 soft derivative/native forward，按 `H_reg=Q_uu+mu*D` 等有量纲依据的形式定义正则。D 为 Hessian 单位的正对角尺度，mu 无量纲；spec 明确来源、均值归一化、退化方向下界和单位。不要把绝对下界重新设为 1，也不要只随手换一个 λ 常数。
2. **N，导数对照：native 局部方向。** 与 S 共用同一 D、下界、初始 mu、目标和尝试预算，只将保存问题上的局部方向换成 native 接触 FD。不要随 N 导数再换一套正则尺度，避免混淆原因。
3. **P，结构候选对照。** 一个 32 候选槽批次：incumbent/best nominal 两个基线身份 + 15 组正负、时间有结构的目标角扰动，针对卸载—摆足—落足。使用同一已评分 reference/K/H/feedback-scale 为基底，只改声明的前馈时间结构；native rollout 与安装用完全相同的策略。不能给开环好看的轨迹临时附上另一套反馈。

S/N 每次最多 5 个 regularization attempt（首次加最多四次重试），每次成功 backward 最多 10 个 alpha rollout；nominal 反馈倍率筛选的 10 次 rollout 与 FD 子调用另计。P 的 32 槽与 S/N 并非等计算预算，去重、缓存、新 rollout、walltime 分别记录，不把 32 当 worker 数。不扩成 λ/权重/worker 网格，不设墙钟强停。

完整重算 r30 的约束 Riccati、k/K、Vx/Vxx/dV、活动约束及实际部署 H。不能减旧 H 的 λ 再求逆冒充新 Riccati；不能降低 native 接受要求、把 proxy 下降当真实下降，或人为缩放 surprise 触发降阻尼。最初保持每问题一次成功改进，避免同时改变正则尺度和额外精修预算。

每个问题逐例记录原 `(trunk violation, native J)`、所有成本分项、预测 base Δx/vx、足高/法向力/离接触、动作/PD请求、chosen-alpha expected/actual、k/K/H/D/mu、来源与耗时。

据此只选择一种主机制进入 S5-r33：S 若有真实原生成效则优先；若 N 或 P 明确提供更好的可部署运动方向，选择对应机制并写清 spec。四个问题的微小正位移不能称闭环起步，选择仅支持进入真实验证。若有限候选仍驻留或前进候选更贵，交付分项负结果，形成一个有证据的控制/参考/目标修订，再训练；不要原样扩大 PPO 或仅把 r32 改名。

所有方法保留当前问题中已真实评分的最优完整策略，包含 incumbent、最佳 nominal、搜索候选。策略的 actions/times/reference/K/H/feedback-scale 必须与评分一致；任何均值、插值、精修结果重新 native 评分。无可部署候选明确报错，不吞错或安装未评分旧策略。现有遗漏改善不足 1e-9 的候选保留问题单列为契约修复，不能当主要运动收益。

## 三、实现、父点和完整训练目标

补全 r33 spec/patch、build/prepare、artifact/runtime、记录和发布支持；正常构建并完成上述直接相关比较。不要改写旧 W43 job、评价和 r32 产物。保留原 Go2/TM02 .02 域、PD60/5、力矩/接触/跌倒/路线、r28 trunk、r27 C2、r26 packet/十子步/TV、冻结 PANN、658D 观察、PPO 2 Hz 标量 cap 和 W38 奖励。长期落点/载荷权限另立阶段，不同时改一串物理和学习条件。

创建 W44 phase 声明脚本与新 clock/训练目录，冻结一个 selected_mechanism，两流同一控制器。各自真实父点为：

```text
A: outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w43-a/PANN_PPO_S5-seed1/resume-state-ep00410.pt
B: outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w43-b/PANN_PPO_S5-seed2/resume-state-ep00398.pt
```

继承完整 actor/critic/optimizer/normalization/RNG、fork_identity、phase_lineage，起点各 14225/55、tail0、excluded145。**W44 各新增 512 条、2 次更新到 14737/57**：首批 14226..14481/update56，次批 14482..14737/update57。不混父点、不重新 seed、不丢 tail、不合并权重、不默认扩到 2048。修正新声明遗留 `phase_count_rule=+2048/+8` 文字，历史冻结 job 原样保留。

先推实现、spec、机制比较、真实父点与 phase 声明，再启动。下列为完成 r33 接线和 W44 声明后的启动形式，**不能在当前旧 W43 配置上直接执行**：

```bash
python scripts/prepare_s1_go2_r12_support.py --build S5-r33
# 创建并执行 W44 声明脚本；核对其读取的是上述两个真实父点。
python scripts/start_s1_go2_r12_tmux.py --new-phase --no-wall-deadline --stream w44-a --workers 10 --session s1-w44-a --network-mode clash
python scripts/start_s1_go2_r12_tmux.py --new-phase --no-wall-deadline --stream w44-b --workers 10 --session s1-w44-b --network-mode clash
```

实际启动必须在构建、比较、声明与推送已完成之后；上面 prepare 若已成功完成则不重复构建。已有同一 W44 job 时使用它自己的 job/resume/clock，不重复 `--new-phase`。两流结束后使用各自末次 update57 固定 E30；同 case 前级通过才 E60、300、600。合法去程、合法返程、完整往返三项在 E60/300/600 分别均须 ≥1/5/10，并满足原条件。训练 60 秒采集不是正式 E60 成功。

## 四、并行、持续运行和交付

默认 **CPU A 10 workers + CPU B 10 workers + 现有 GPU 作业**，Torch/BLAS 各 1，先以 `-j6` 完成构建再进入双 CPU 重负载。总容量 25 vCPU/120GB RAM/96GB GPU 不是当前空闲量；读取现场后再分配，给 GPU host、系统、发布留余量。新 phase 前可据真实可用量声明 11/12，但两流合计及其他负载不能超额；不热改活动 workers，不开第三条大 CPU 流。

CPU 原树已结束且干净可复用；活动任务或研究修改存在时隔离。GPU 可写 upstream/build/env 不共享，prepare 不得改到它的目录。venv、静态资产、完成父输出可只读复用。当前 GPU 不重跑 G1/G2/G5，不抢占其他训练；自然结束或明确技术失败后再按其原契约收集。

**全部新后台 deadline=null，无固定墙钟完成期限，无到点 kill。** 按样本/更新目标、评价完成、明确技术失败或用户停止结束。物理 30/60/300/600 秒、训练 60 秒采集、有限数值预算与单次网络 timeout 保留，不无限重启或吞错。

保留逐回合 compact/resolved、完整轨迹和 planner calls、2 Hz transition/PANN/value、rollout、学习更新、末次固定评价、首次错误输入、stdout/stderr/warning。新增尺度、chosen-alpha 高精度比例、成本分项和候选身份；关键 nonprogress/终止窗口保存非赢家候选状态/足部/动作与评分，使后续能区分“没找到运动”和“找到但没选”。不要只推索引，也不要每个 500 Hz 动作行重复所有大矩阵。

结束先推 compact 和学习数值，所请求原件后台 Release。Git/index 短写串行整合最新 CPU/GPU 提交，不强推、不覆盖并发结果；只发新增，不重传 W43 的 779 项。远端资产核验与实际下载复算范围分别报告。

**推送 GitHub 并核验后，不在我的电脑另下载、镜像、打包或归档；不自动删除服务器 checkpoint/resume、原件、视频及上传副本。** 最终交接给出实际树、source/artifact、选择的机制和依据、两流 PID/job/父点/样本/更新/tail、当前 GPU 状态、评价、Git/Release 和准确续接点。开场简短后直接工作；不要停在“已制定计划”，也不要将构建成功或站满时长写成稳定材料通行。
