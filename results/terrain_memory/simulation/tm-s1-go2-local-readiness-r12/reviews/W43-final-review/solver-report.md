# W43 最终 CPU 求解器原件复核

复核基准：`6a94c94c8e21e3b50fcca40ab4363165dee2987e`。对象为 `job-20261110T035658Z-w43-a`、`job-20261110T035700Z-w43-b` 的最终 update 55 / E30，控制器 `S5-r32`。本报告不启动物理实验、训练或 GPU 任务，不修改运行代码。

## 1. 结论与证据强度

**λ 重置已有效消除旧问题之间的大阻尼继承；本轮最终评价仍没有产生行走计划。** 这不只是实际机器人没有前进：已发布的 34 个完整 accepted 预测本身全部后退，全部记录节点的四足几何底部都仍在地面接触高度。能与后续真实数据比较的 32 个计划，在首 20 ms 内与实际执行高度一致。因此，至少这些真实窗口中，不能把失败解释为“规划已给出向前步态，但 PD/QP 把它破坏了”。

目前最值得直接处理的是**规划器的局部求解尺度和求解充分性**：新问题从 λ=1 开始，记录的正则项占 action Hessian 对角线约 99.86%–99.93%；成功接收通常就结束本次局部求解；native 实际下降只达到所选步长局部模型预测下降的大约四分之一。更关键的是，已选计划的改进主要是微小的身体支撑高度修正，没有建立足部离地与接触切换。

这些证据**尚不能唯一判定**“只降低 λ 就会走”“增加局部迭代就会走”或“目标函数必然更喜欢驻留”。下一步应对已保存的同一问题做小规模、可区分原因的 native 精确比较，然后只将一个有证据支持的机制放入双流实验。纯扩大 PPO 样本、单纯增加同方向 line search，或先大幅增加随机候选，都没有被当前原件直接支持。

## 2. 实际读取和复算范围

这不是把运行索引数字改写成原件复算。两个 Release 中实际用于本报告的 **46 个文件**，均逐项核对 GitHub 资产 `digest`、`size` 与下载文件 SHA-256、字节数；完整 URL、原 `source_path` 和哈希列于 [solver-summary.json](solver-summary.json) 的 `inputs`。

| 对象 | 本报告实际读取范围 |
|---|---|
| `planner-calls.jsonl.gz` | 两流末次 E30 各 1,521 行，包含失败、复用和接受调用 |
| `planner-records.jsonl.gz` | A 1,385 / B 1,383 个接受计划及所选 α、dV、实际 native 改进 |
| `trajectory.npz` | 两流各 15,001 个 2 ms 记录节点，供短期预测对齐 |
| `feedback-actions.jsonl.gz`、`action-execution.jsonl.gz` | 用 `query_id` 连接，只统计 `query_kind=applied_hold` 的实际安装记录 |
| `accepted-reference-gain-H-plan*.npz` | A 17 份、B 17 份完整事件窗口；每份 176 个状态/残差节点及 H/K 等原数组 |
| `G1-project-go2.xml` | 原运行模型 XML 中残差定义、norm 和足部运动学 |
| 仓库 `*-eval-*-E30-resolved.json` | 实际 outbound 阶段 `applied_settings[-1].cost_weights` |

原件发布映射见 [A manifest](../../jobs/job-20261110T035658Z-w43-a/diagnostics/manifest.json)、[B manifest](../../jobs/job-20261110T035700Z-w43-b/diagnostics/manifest.json) 及同目录 `evidence.json`。资产对应最终 `-eval-` 路径；没有把同名训练文件混入最终 E30。

**仍未具备的证据：**并非每一个在线调用都发布了完整状态、残差、H/K；发布的是选定事件窗。其他候选的总成本、约束违规和失败状态有日志，但没有所有非赢家的全时域状态/残差。因而不能从本报告推出“所有候选都完全没有抬脚”，也不能复算未发布非赢家的分项成本。未审查 checkpoint 内部内容，未执行新的反事实 native 求解；训练全轮与材料事件覆盖另见 [cpu-report.md](cpu-report.md) 和 [publication-audit.json](publication-audit.json)。

源码依据：仓库 [r25 source](../../../../../../experiments/terrain_memory/s1-go2-r25-source) 逆去 r25 后顺序应用 r26、r27、r28、r29、r29-qp1、r29-qp2、r30、r31、[r32 patch](../../../../../../experiments/terrain_memory/s1-go2-r11-S5-r32.patch)。未编译该环境。未改动的 norm 定义按仓库锁定作者版本 [e00c47a5 / norm.cc](https://github.com/johnzhang3/mujoco_mpc/blob/e00c47a5adb9856af2e0f24231bb3a60d5be23c4/mjpc/norm.cc) 和 [task.cc](https://github.com/johnzhang3/mujoco_mpc/blob/e00c47a5adb9856af2e0f24231bb3a60d5be23c4/mjpc/task.cc) 读取。

## 3. λ=1 的重置有效，但不等于当前数值尺度适合运动阶段

`R32BeginProblem` 在物理状态/时间或参考、目标、权重、步态改变时建立新 epoch 并把 `regularization`、`regularization_rate` 置为 1。`Iteration` 中的 regularization 重试循环在首次满足 native 接受条件后 `break`；这个重试循环不是“接受后重新线性化若干次”的循环。[r32 patch](../../../../../../experiments/terrain_memory/s1-go2-r11-S5-r32.patch)

最终 E30 出发阶段（`origin >= 2.002 s`）原件统计如下；完整值在 `solver-summary.json.streams.{a,b}`。

| 字段/计算 | A | B |
|---|---:|---:|
| `outbound_calls` | 1,401 | 1,401 |
| `outbound_accepted_plans` | 1,374 | 1,372 |
| 已接受调用的 backward attempt 数 | 全部 1 | 全部 1 |
| 接受后 `lambda_after=1` | 1,184 | 1,173 |
| 接受后 `lambda_after=2` | 190 | 199 |
| 接受后 `lambda_after=0.5` | 0 | 0 |
| 实际所选 `action_step_alpha=1` | 1,374 | 1,371 |
| 其他所选 α | 无 | 1 次 `0.1778279410038923` |
| 所选 α 的 native/proxy 改进比，中位数 | 0.2359596012 | 0.2353629939 |
| 该比率最大值 | 0.2951247544 | 0.2942996385 |

**不能描述为“每次成功后已经降到 0.5，却马上被下一 epoch 覆盖”。真实记录没有发生这种情况。** 现有更新规则要求 `surprise > 0.5` 才减小 λ；全部已接受的 outbound 比率低于 0.5，大部分保留 1，部分因低于 0.1 增大为 2。跨问题重置避免污染与同问题求解是否充分是两个问题。

比率按所选 α 重新计算：

`expected = -alpha * (dV_unscaled[0] + alpha * dV_unscaled[1]) + 1e-16`

`ratio = cost_improvement / expected`

重算结果逐条等于高精度 `proxy_expected_decrease` 所对应的比率。没有把 `planner-calls.search_attempts[].proxy_full_step_decrease` 当作 α≠1 时的接受分母。`cost_expected` 和 `cost_surprise` 原日志经过 `std::to_string` 只保留 6 位小数，不适合单独用来重算很小改进；所需高精度字段实际上已发布。

### 正则项相对曲率的量级

34 个已发布 accepted H 的非终端节点中，λ 均为 1；H 的特征值总体约在 `[1.0006547624, 1.0020238222]`。`lambda/H_ii` 在 `[0.9985541363, 0.9993456601]`，说明当前正则项主导该局部 action metric。

代码与实际配置还解释了一个明显的阶段尺度差：Stand 的 Effort 是 `(ctrl-reference)/0.02`；运动阶段 Effort 是 `0.02 * actuator_force`。按权重 0.08、176 节点均值以及本轮未饱和 PD 的 `kp=60`，只看 Effort 的直接局部 action 曲率：

| 阶段 | 直接 Effort 曲率示例 |
|---|---:|
| Stand | `0.08 / (0.02² × 176) = 1.1363636364` |
| 运动阶段 | `0.08 × (0.02 × 60)² / 176 = 0.0006545455` |

二者相差约 1,736 倍；这不是完整 Riccati 曲率，但运动阶段数值与 H 超出单位阵的最低量级一致。因此“同样重置到 1”在两个阶段意义非常不同。**不能把存储的 H 减掉 I 再求逆，冒充完成了另一 λ 下的 Riccati 递推**：值函数递推、反馈、约束活跃集和新 rollout 均会改变。较小尺度是否产生可靠前进，必须通过新的真实求解与 native 评分验证。

## 4. 已接受计划本身没有规划出行走

以下来自 `accepted_states` 首末节点，时域 0.35 s；正方向目标是向前。

| 流 / plan_id | origin (s) | 预测 Δx (mm) | 预测 Δz (mm) |
|---|---:|---:|---:|
| A / 91 | 3.98 | -0.141914 | -0.104208 |
| B / 90 | 3.98 | -0.138177 | -0.101028 |
| A / 1385 | 30.00 | -1.085135 | -1.018246 |
| B / 1383 | 30.00 | -1.081271 | -1.014380 |

34 个完整事件窗的 `predicted_dx_m` 全为负；所有记录节点的预测 vx 也全为负。所有足部几何底部距刚性平面的值约为 **−4.060 至 −3.363 μm**，没有出现目标约 5 mm 的摆动足离地。这里“无离地”严格指已保存节点的几何结果；没有把纯运动学计算称为所有积分子步的接触力重放。

运动学方法是从实际 XML 删除仅用于显示的 mesh 元素，保留关节、刚体、足部几何，执行 `mj_kinematics`。足部 geom 名为 `FL/FR/RL/RR`，而非 `_foot` 的 body 名。每流用 7 个真实 trajectory 节点交叉核对 `feet`，最大位置差 `5.55e-17 m`。分析环境 MuJoCo 4.0.0 **只用于运动学，没有以另一版本重新模拟项目物理**。

这些完整原件证明了“接受计划仍驻留/后退”，但不能证明当前十个 α 候选以外存在或不存在一个可行且更便宜的步态。line search 调整的是同一个局部方向的步长，数量更多并不等于接触模式更多。

## 5. 目标没有消失；native 局部改进主要没有转化为步态

用原 XML 的 residual 维度和 norm、实际 readback 权重，对 `accepted_residual` 与 `reference_residual` 复算全部成本。34 份 accepted 总成本与原 `header.accepted_cost` 最大差 `3.47e-17`；reference 总成本与 `feedback_reference_cost` 同样匹配。

规范为 `norm 0: 0.5*sum(r²)`，`norm 2: sqrt(sum(r²)+p²)-p`，`norm 6: sum(sqrt(r_i²+p²)-p)`，逐项乘实际权重，再对 176 个节点取算术均值。risk=0。完整 residual 索引、norm、参数和权重在 `solver-summary.json.streams.*.cost_definitions`，逐计划分项在 `plans[].accepted_components/reference_components/component_decrease`。

| native 分项 | A plan 91 accepted | A plan 1385 accepted | B plan 1383 accepted |
|---|---:|---:|---:|
| Position | 0.007804897922 | 0.007848209008 | 0.006936985856 |
| MaterialSupportHeight | 0.000010324830 | 0.028600466608 | 0.028481350807 |
| Height | 0.000667037037 | 0.003330504985 | 0.003322879458 |
| Effort | 0.002516117970 | 0.002260456328 | 0.002260716366 |
| Gait | 0.000692356827 | 0.000618188573 | 0.000618188550 |
| 总成本，包含其余各项 | 0.011698249379 | 0.043038121985 | 0.041999009353 |

Position 在早期约占 A 总成本的 66.72%，没有被禁用。晚期身体支撑高度误差占主要成本。A 最后一个计划比该次 reference 便宜 `0.0003252551222`，其中 `MaterialSupportHeight` 改善 `0.0003174249011`，约为净改善的 **97.59%**；B 对应为 **97.70%**。A/B 的 Gait 改善仅约 `1.25e-9 / 1.24e-9`。

不能把这些结果写成“优化主要为了减少力矩”。以上早晚代表窗口中，接受新计划实际上都增加了 Effort 成本：A 最后一步 `reference-Effort minus accepted-Effort = -0.00002593076027`。它付出了较多 effort，以减轻该次参考下的身体高度/姿态误差，却仍未找到足部卸载、前进的方向。跨时刻参考和身体状态在改变，因此“每次局部 J 降低”不能推出“闭环 J 一直下降”或“任务越来越接近完成”。

### `MaterialSupportHeight` 非零不代表材料已加载

重建源码 `QuadrupedFlat::ResidualFn::Residual` 的条件是：

```cpp
residual[counter++] = (material_support || r11_moving_body_support)
    ? (base_z - support_height[0]) / .04 : 0;
```

其中 `r11_moving_body_support` 对 Quadruped 的非 Stand 步态启用，**不要求足部位于材料域**。因此刚性起点也会出现此项，不能拿字段名称或非零 residual 证明材料承载。实际材料接触应看材料日志、足部位置、材料事件和加载记录，详见 [cpu-report.md](cpu-report.md)。同样，`support_active=0` 是 Stand 支撑协议没有开启，并不意味着所有运动期身体支撑项均关闭。

## 6. 最终评价没有重现 PD/QP 执行错误

两流末次 E30 各有 15,001 条 `applied_hold` 记录，其中含 30 s 终点的安装记录；不能将其全部说成有完整未来时间段的实际保持。按 `query_id` 连接反馈记录后，两流均为：

- `qp_converged=false`：0；`metric_valid=false`：0。
- `projection_changed`：0；`projection_delta` 最大绝对值：0。
- `PD_violation_Nm` 最大绝对值：0。

34 个保存计划的首目标与 `action-execution.held_target` 最大绝对差 `9.20e-8 rad`，符合记录/传输精度量级。**其中只有 32 个计划有后续真实 20 ms 可比较；两流 30 s 终点计划仅能检查安装目标，不能声称已验证其未来执行。**

32 个计划在 0–20 ms 的 11 个节点上，与真实 trajectory 的最大绝对差为：base xyz `2.30e-8 m`、关节角 `1.52e-7 rad`、qvel `1.31e-5`。因此这些窗口里的 native rollout 与执行管线相符。此结论不外推到尚未到达的材料接触、足部离地或高饱和状态。

## 7. nominal/incumbent 缺口需要收口，但不是本轮驻留的主要解释

`NominalTrajectory` 会评价不同反馈倍率，选 best nominal 进行局部优化；若局部尝试不能严格改善 nominal，r32 当前可能复用当前问题已精确评分的原 incumbent，而不是部署本次已评分的最优 nominal。这个语义缺口在后续多候选或多次 refinement 中值得统一解决。

但此次必须用最终 E30 全调用量化它。对出发阶段复用调用，比较 `incumbent_current_cost` 与本次已有 `search_attempts[].native_candidates` 中同等 trunk violation 的有限非失败候选：

| 流 | 复用调用 | 现有候选比 incumbent 更便宜的调用 | 最大遗漏改善 |
|---|---:|---:|---:|
| A | 27 | 3 | `3.7891e-10` |
| B | 29 | 4 | `9.5388e-10` |

因此，先前少量输入上发现的较大差距不能代替本轮在线统计；本轮最终评价中的这个缺口只有极小成本量级。**修复“所有已实际评分的有效候选保留最优”是合理的契约修复，不能包装成已经找到让 W43 行走的主因。**

实现时 best archive 应同时保存实际评分的参考轨迹、前馈、反馈增益/倍率、约束 metric、native rollout 和身份；不能换了反馈倍率仍沿用旧成本。均值候选或 refinement 结果须重新 native 评分；没有有限有效候选时显式报错，不能静默部署未评分旧策略。

## 8. 下一步：一个机制进入 W44，先用保存问题区分原因

优先级建议是 **求解尺度与 native 有效步，而非立即全面扩张采样**。现有证据最直接支持：固定 λ 的阶段尺度不匹配、局部方向的 native 成效偏弱、首次小下降即可结束；它还没有证明唯一根因。建议把下列工作作为一个明确的控制器实现决策，完成后直接进入同机制双流联合实验，而不是再安排一串只生成审计报告的工作。

1. **固定现有真实问题和成本。** 建议使用两流已发布的 `r32-search-call000221.npz`（4.00 s）与 `r32-search-call001517.npz`（29.92 s），共四个问题，连同模型、参考 packet 和原始状态。选择 1517 而不是首个晚期窗口 1516，是因为 1517 的输入 incumbent A1380/B1378 有完整已发布策略，1516 的 A1379/B1377 没有；首次 outbound 的 call121 也没有完整 search/策略快照，不能宣称为等价历史 replay。首先核对原 native 成本/身份能否复现；若内部 warm start 等状态未完整保存，明确重建差异，不把“相似状态”说成逐位同问题。冻结奖励、PPO 权重、物理、路线和材料定义。
2. **做小规模针对性比较。** 保留原 λ=1 单步作为基准；比较一项根据 action 曲率尺度确定的正则化/有限次数重新线性化方案，并用 native 局部方向与一个低维、有步态结构的候选族作为诊断对照。每种方向最终都交给同一 native rollout、真实执行约束与成本选择，不做宽泛 λ 网格或高维盲采样。较小 λ 的试验必须重算 Riccati、反馈与轨迹；不能从当前 H 代数改造冒充结果。
3. **根据可反证结果只选择一项主改变。** 如果尺度化/重新线性化已经产生更便宜、可部署、有向前预测和卸载/摆足迹象的 native 计划，就先做局部求解修复；如果 native 局部方向有效而 proxy 方向无效，优先修导数模型；如果局部方向仍锁在四足支撑、结构候选能在同成本下降下产生进展，才让结构候选成为主机制。如果可行前进候选持续比驻留更贵，才能开始有证据地修改任务成本/时域，而非增加同一目标下的候选数量。
4. **两 CPU 流用同一个选定机制。** 分别从自身完整父点继续，保留随机性与历史差异作工程重复；不要 A 改正则、B 改步态，然后把结果差异当成单因素结论。真实闭环输出同时记录足部离地/接触、前进、身体高度、材料加载和合法路线；仅 native J 下降不算起步成功。

这里的比较不是预先承诺三种机制同时上线，也不是给后台设完成时限。有限求解迭代/候选预算是算法定义，30 s 物理评价是任务定义；它们均不等于墙钟强制终止。现有 GPU 任务不动，CPU worker 数依据现场实际占用划分；两个求解流需为 GPU host、系统和发布预留核数，不能按总显存容量直接推断还能同时运行几个任务。

## 9. 重现本报告

[analyze_solver_raw.py](analyze_solver_raw.py) 读取 `solver-summary.json.inputs` 所列的 46 个发布资产和仓库 resolved 配置。把资产按 `local_name` 存入同一 `RAW_DIR`；[release-assets-snapshot.json](release-assets-snapshot.json) 是本次两流公开 Release API 的资产元数据映射（每个 job 对应 `assets[].name/size/digest/url`），作为独立发布核验输入，并非新的物理原件。分析需要 NumPy 和 MuJoCo；此处 MuJoCo 仅执行上述运动学运算。从本审查目录执行：

```bash
python analyze_solver_raw.py \
  --repo /path/to/MCP-RL \
  --raw /path/to/raw-originals \
  --release-metadata ./release-assets-snapshot.json \
  --output /path/to/solver-summary.json
```

程序断言 34 份成本重算匹配、所选 α 预期下降公式匹配、原件大小与 SHA-256 匹配，并输出全量逐计划和逐复用调用结果。它不启动 native dynamics、checkpoint 续训或后台任务。
