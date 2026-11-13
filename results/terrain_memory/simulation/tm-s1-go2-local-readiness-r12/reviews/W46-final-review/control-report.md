# W47 控制侧审查：W46/r35 的有限改善、搜索停滞与下一步辨识

执行规则以[最终W47计划](../../../../../../docs/S1_W47_STABILITY_PLAN.md)为准；本独立报告的建议由主审查整合，不能另开一套并列实验。

审查基点：`15ee620b8a909246a12baf999c136c2be299655b`。本轮没有运行 MuJoCo 物理、MPC 求解、服务器实验或 PPO，也该独立审查阶段没有修改控制实现；主审查把派生报告随计划发布。审查对象是已经完成的 W46，不是重复实施 W45/r34。

## 1. 核心结论

**r35 的新增代价实现已有较强的一致性证据，但它没有使原任务稳定运行。下一步应先区分“没有产生好候选”“重滚破坏运动种子”“当前目标拒绝有用运动”这三个问题；继续增加 R 扫描或直接大改 FDDP，都缺少充分的定位依据。**

这次最有价值的新整理结果不是“又失败了 540 次”，而是：四个保存问题、三个分支的 12 次有限求解中，60 次后向递推全部完成；540 个正 alpha 候选的 native rollout failure、trunk violation、QP failure 和 actuator restoration 全为 0，却全部比同次 nominal 的目标值更差。因此，这是一个明确的**方向/候选质量和实际下降问题**，不能继续归因为接纳容差太严、QP 未收敛或后台被定时杀死。

W46 原 A/B 的 E30 则确实是两次物理任务失败：26.432 s 地面与躯干接触，11.638 s 侧翻；完整任务资格 0/2。有限窗口里 A153 的命令改善没有延伸成稳定移动。新方向何以无效、原任务何以最终失稳之间仍有需要分离的环节，不能将单个早期事件当作十几秒后的充分原因。

### 证据等级

- 本审查直接阅读了 r25 完整相关源码、r26–r35 补丁、W46 比较/分析脚本与所有相关 compact JSON；随后补读根代理离线重建的完整 r35 planner/policy/trajectory，确认下述执行链与完整上下文一致。对 compact 中逐候选记录进行了独立聚合，输出 [control-compact-aggregation.json](control-compact-aggregation.json)。这部分可以直接本地重算。
- 12/36 的完整策略与轨迹 NPZ、全部实际 planner calls 没有在本代理重新读取。两次批量原件读取被自动审批拒绝：第一次落盘缓存、第二次约 335 MB 的 RAM 读取，都被判与用户“不再在电脑额外下载/归档”的要求冲突。本代理没有继续请求或改道重试。因此，成本、策略哈希、原始诊断数组一致性标作“仓库记录”，不冒称本代理从原始数组独立复算。
- 实际 A/B 轨迹的新增数字来自根代理已独立核验 SHA256/ZIP CRC 后，在 RAM 中复算的紧凑结果 [trajectory-a-summary.json](trajectory-a-summary.json)、[trajectory-b-summary.json](trajectory-b-summary.json)；原件未落盘。不能把这种轨迹复算扩展成全部 planner/candidate 原件均已复算。

## 2. r35 改了什么，哪些错误目前可以排除

主要来源：`experiments/terrain_memory/s1-go2-r11-S5-r35.patch`，特别是 116–171 行的 `R35Freeze`、194–198 行的调用顺序与 173–185 行的导数记录；`scripts/compare_s1_go2_w46_saved.py`；`w46/fixtures.json`、`numerical-consistency.json`。

### 共同参考与评分

每个新的物理/参考 epoch，`R32BeginProblem` 先建立问题身份，`R35Freeze` 随后从当前完整输入 incumbent 在当前状态重滚一次 native 轨迹，冻结投影后的目标 `u_bar`、法向力及 R。之后 nominal screening、正 alpha 候选、archive 使用同一新目标。参考在 worker 调度前写入，期间保持不变；并没有每个候选用自己的动作作参考而自动把新代价变成 0。

分支加入 R32 问题身份；参考 hash 包含 origin、dt、source policy、u、force、R、branch。每次真实物理时钟推进会形成新的问题，不能据源码把这次解释成“始终使用一个陈旧参考 epoch”。也没有发现“选中完整策略后只安装部分数组”在 r35 又被重新引入的直接证据。

新增项是：

\[
J_{\mathrm{ref}}=\frac{1}{2\,176}\sum_{k=0}^{174}(u_k-\bar u_k)^\top R_k(u_k-\bar u_k).
\]

变量是实际 held qtarget `u=data.ctrl`。冻结参考后，直接状态导数为 0，控制一阶、二阶项为 `R(u−ubar)/176`、`R/176`；terminal 新增项为 0。它随 residual/score/cost derivatives 进入递推，不是只改 LM 正则。投影仍在已有控制执行链中，未再次把投影 Jacobian 乘进 held-u 残差导数。

仓库记录：四个 disabled-penalty 对照的 state/action/旧 residual/J 差均 0；候选成本复算最大约 `4.44e−16`；held-u residual D 误差最大 `7.96e−11`，直接 C 为 0；共同参考差 0，trace 差约 `4.44e−16`。这些支持新增代价实现自洽，但不验证整个 native 动力学导数、完整闭环稳定性或长期任务目标正确。

### constant/load 相同不等于分支未生效

每腿权重是 `.1152*(1−w_leg)`。只要总法向力为正，各腿权重的均值恰好 `.0864`；全无载节点的均值是 `.1152`。因此这里的 constant 是**同节点各腿等强度、与 load 匹配 trace**，不是跨所有节点都固定为一个数。

在 A148 同一个完整策略上，constant 的 Jref 为 `.0071250364`，load 为 `.0072069613`；A153 分别 `.0304686379`、`.0303900888`。共同参考一致但新增成本不同，显示 load 权重不是简单被忽略或与 constant 别名。四个上下文的最终完整 policy hash 都满足 constant=load，是已有候选在两种加权下选择相同，不能据此宣称发现了载荷信息收益。

### 这项成本没有直接惩罚相邻输入跳变

`u_k−ubar_k` 与 `u_k−u_(k−1)` 是不同的量。只要候选跟随当前 incumbent 的剧烈动作，Jref 就可以为 0；即便 incumbent 相邻 2 ms 跳变超过 1 rad，也不会因为“参考自身很跳”被新项惩罚。B155/B158 选中 incumbent 的 Jref 正好为 0，是这一限制的实证。

它提供的是围绕当前输入序列的偏离代价，不会自动生成一个能前移、保持接触并恢复的运动参考。另一方面，none 分支也保留相同 B incumbent，因此尚不能把坏参考锁定全部归罪于新增惩罚；问题在加入惩罚以前已经存在。

## 3. 12 次求解到底做成了什么

来源：`w46/{A148,A153,B155,B158}.json` 的 `results[].call.r34_decisions`、`search_attempts` 与 `selected/diagnostics`。聚合脚本逐项计算，并保存输入文件 SHA256。

每个分支有 10 个 nominal 候选、5×10 个 line-search 候选；后者每次有 9 个正 alpha 和 1 个零 alpha。因此总计 720 个候选记录、540 个正步、60 个零步；另外 12 次参考重滚，使求解 rollout 总数为 732。transition FD 46224 次，risk geometry FD 7522 次。36 次固定策略完整 350 ms 诊断另计，8 次 disabled 等价对照另计，不能混称新的物理任务。

下表的 `min ΔJ` 是每次求解 45 个正步候选中最小的 `J_candidate−J_nominal`；全部严格大于 0。各行的正步候选数均为 45，eligible/replaced 均 0；lambda 使用顺序均 `[1, 2, 8, 64, 1024]`，五次 backward 均完成。

| 上下文 | 分支 | 最终来源 / scale | min ΔJ | 达到该值的 attempt / alpha | 五个 alpha0 的 ΔJ 范围 |
|---|---|---|---:|---|---:|
| A148 | none | nominal_3 / .0133352 | +.0173863907 | 5 / .00237137 | −6.15493e−7 ～ +8.81134e−8 |
| A148 | constant | nominal_3 / .0133352 | +.0337677667 | 3 / .001 | −9.60537e−10 ～ +1.67179e−7 |
| A148 | load | nominal_3 / .0133352 | +.0322383776 | 5 / .001 | −4.37531e−12 ～ +8.75985e−8 |
| A153 | none | nominal_1 / .00237137 | +.0029140583 | 5 / .001 | −2.33101e−11 ～ +6.86626e−10 |
| A153 | constant | nominal_9 / 0 | +.0450258404 | 1 / .0749894 | −6.03451e−7 ～ +5.32103e−8 |
| A153 | load | nominal_9 / 0 | +.0282995530 | 3 / .0133352 | −6.23201e−7 ～ +9.41366e−9 |
| B155 | none | incumbent / .001 | +.0543732799 | 3 / .0316228 | +.0130069 ～ +.0312558 |
| B155 | constant | incumbent / .001 | +.0774749609 | 5 / .0749894 | +.0218545 ～ +.0501948 |
| B155 | load | incumbent / .001 | +.0888539787 | 4 / .177828 | +.0189449 ～ +.0533015 |
| B158 | none | incumbent / .0133352 | +.0788072024 | 3 / .0749894 | +.00617747 ～ +.0170057 |
| B158 | constant | incumbent / .0133352 | +.1059959183 | 4 / .00237137 | +.0179571 ～ +.0297040 |
| B158 | load | incumbent / .0133352 | +.0945374013 | 4 / 1 | −1.33924e−7 ～ +.00688593 |

所有正步的 comparison tolerance 最大仅 `4.18e−11` 到 `6.22e−11`；最好的正步相对容差仍差约 `6.79e7` 到 `1.92e9` 倍。没有一条被拒方向只是“落在计算精度误差里”。所有 nominal_bound_violation 为 0；positive rollout failure、trunk violation、QP failure/restoration 为 0。候选中投影实际介入过，每条正步的介入节点数区间跨全表为 6–130，但投影介入不等于 QP 失败。

### 选中策略的实际变化

| 上下文 | none → constant 的改变 | 350 ms nominal 目标最大跳变 | 350 ms nominal 相邻 .2 ms 力矩变化 | 位移 / 最低高度 |
|---|---|---:|---:|---|
| A148 | 完整策略不变 | .816725 → .816725 rad | 49.0917 → 49.0917 Nm | dx −2.742 mm，最低 .280687 m |
| A153 | 旧完整 policy 的 feedback scale .00237137 → 0 | 1.084183 → .569357 rad | 65.3074 → 34.2848 Nm | dx .019 → 1.494 mm；最低 .279224 → .278072 m |
| B155 | incumbent 不变 | .513507 → .513507 rad | 30.5961 → 30.5961 Nm | dx 1.081 mm，最低 .227429 m |
| B158 | incumbent 不变 | 1.373791 → 1.373791 rad | 83.2045 → 83.2045 Nm | dx .354 mm，最低 .259185 m |

A153 的 J0 `.0499301747 → .0501657927`，略增 `.000235618`；末端 vz `.123461 → .135877 m/s` 仍为正，最低高度下降约 1.152 mm。它确有局部折中改善，但来源是旧 policy 的 nominal scale 重选，不是新 Riccati 方向产生了更好的运动轨迹。

**时间窗容易混淆。** `selected.torque_adjacent_substep_max_Nm` 对应求解默认只记录的 100 个子步，即前 20 ms；`diagnostics[].summary` 为 1750 个子步，即完整 350 ms。A153 前 20 ms 的力矩跳变反而 `11.2757→11.3239 Nm`，完整 350 ms 才是 `65.3074→34.2848 Nm`。目标 actions 数组则含完整 horizon，不能把同一个 summary 中不同覆盖范围的指标误认为都只看前 20 ms 或都看完整 350 ms。

这四个问题以保存的 W45 **输出策略**及重建上下文作为输入；原始 pre-call incumbent、完整私有 optimizer 生命周期、原 plant 数值内部状态并未完整恢复。脚本明确 `historical_replay=false`。这些结果能检验声明的固定上下文机制，不能表述成精确重跑了原历史调用，也不能直接推断真实 W46 1944 次调用全部停滞。

## 4. alpha0 端点为何值得先查，但现在不能直接判成单一 bug

源码路径：

- `s1-go2-r11-S5-r32.patch` 263–265：`NominalTrajectory` 用当前初态的 best native rollout 覆盖 `candidate_policy[0].trajectory`。不是简单沿用旧状态、不重新滚动的实现。
- 同补丁 282–291：每个正则化 attempt 都重置到同一 `r32_nominal`。
- `s1-go2-r11-S5-r29.patch` 557–567：`ActionIndex` 明确将新 K 作用于当前状态相对新 nominal 的切空间偏差，随后做原 Clamp/H-QP。
- `s1-go2-r11-S5-r33.patch` 512–520：action rollout 的 feedback scale 固定为 1，alpha 只缩放前馈步。
- `s1-go2-r11-S5-r34.patch` 90–108、174–185、255–266：alpha0 仍记录但不可作为新 action-search 策略安装，正步必须真正改善当前 nominal。

离线重建 r35 的完整对应位置为 `mjpc/planners/ilqg/planner.cc` 的 735–744（保留完整 old nominal 并把实际 rollout 置为递推中心）、1010–1019（逐 attempt 重置中心）、1141–1168（重新生成并评估每批方向）、1367–1402（alpha 前馈步及离散 rollout）；`policy.cc` 的 101–191（旧策略时间插值路径）与 272–282（新离散候选路径）。其中 alpha0 没有通过相同源码分支调用旧 policy；但同模型、同中心下再生 nominal 的条件仍须满足，不能把路径不同本身定性成程序错误。

原先完整 incumbent 的映射是旧 reference、旧 K/H、旧 scale 加上状态相关投影；新的候选近似为：

\[
u_k^{(\alpha)}=\Pi_{H_k^{new},x_k}\left[\bar u_k+\alpha k_k+K_k^{new}(x_k\ominus\bar x_k)\right].
\]

因此 alpha=0 只意味着前馈步为 0，**不意味着完整反馈映射等于旧 incumbent**。它仍换了 K/H，并以 accepted rollout 重新设定中心。离开 nominal 后，两个策略可以不同。这是源码确证的机制。

但在严格相同动力学/数值内部初态下，若 `x=xbar`、切空间自差为 0、ubar 在当前瞬时边界内，则 alpha0 理论上应沿 nominal 再生。B155 的 `.013～.0533`、B158 最高 `.029704` 的目标差不能只用“alpha0 换 K”一句话宣告已经解释清楚。它提示必须定位第一处分叉：时间索引/离散索引、quaternion 自差与插值、QP 中心和活动边界、执行输入微差、模型内部状态，再到接触分支及反馈放大。模型已启用 `mjDSBL_WARMSTART`，没有依据直接归因于 qacc warmstart 污染。

raw K 的绝对数值有表示和单位依赖，不能单独作为坏策略或新惩罚标准；应查看实际投影后的 `Δu`、接触/状态差及完整目标。这些有限结果也不支持放开 r34 容差或重新允许 alpha0 安装——那会重新引入已修复的“零前馈步替换完整反馈策略”问题。

## 5. 真实 A/B 的失稳边界

来源：`w46/phase.json`、`w46/w46-a.json`、`w46/w46-b.json`；根代理的原轨迹 RAM 复算摘要。

| 指标 | A | B |
|---|---:|---:|
| E30 实际物理结束 | 26.432 s，floor–trunk contact | 11.638 s，body_fall |
| wall 时间 | 3935.575 s | 2425.399 s |
| 最大 body x | .626506 m | .396364 m |
| 原合法去程/返程/往返 | 0 / 0 / 0 | 0 / 0 / 0 |
| 最大相邻 2 ms 目标变化 | 1.535358 rad @26.332 s | 1.468642 rad @7.752 s |
| 超过 .5 rad 的 hold 数 | 2125 | 661 |
| 超过 1 rad 的 hold 数 | 319 | 101 |
| 相邻 500 Hz 力矩变化达到约 90.86 Nm | 37 次，首次 6.860 s | 5 次，首次 6.150 s |
| z<.2 m 首次 | 8.868 s | 8.464 s |
| abs(roll)>.5 首次 | 11.098 s | 8.572 s |
| 完整足加载材料（500 Hz 观测） | 未观测到 | 首次 9.122 s |

`.5 rad` 姿态和 `.2 m` 高度只用于描述时间顺序，不是新增失败门槛；短腾空也不是新的失败。90.86 Nm 是膝关节 ±45.43 Nm 限值间的相邻反转幅度，不是实际施加了超出限值的 90.86 Nm 力矩。

A 的 x 在 9.5 s 为 .510775 m，20 s 退至 .286175 m，结束才到 .626506 m；11.5 s roll 约 .810 rad，12 s 又恢复接近水平。时长增加不能当作持续前进。A 全程未实际加载材料；它的失败不能主要归因于已经进入材料后历史记忆失效。

B 首次完整足材料加载时，body z 已为 `.145487 m`、roll 为 `−.642580 rad`；8.464/8.572 s 的明显高度/侧倾变化更早。这个顺序支持“明显失稳已经在完整足材料加载以前出现”，不证明此前绝无部分足接触/材料作用，也不证明某个早期命令是末端侧翻的唯一原因。材料最大应变约 `.006491<.02`，最终原因是 body_fall，不能写成材料域越界导致。

实际 1944 次 planner call 全为 N/r34 strict、r35 激活计数 1222/482、无新 alpha0 安装，这是仓库记录；本代理未从完整实际 call 原件重算其 source 分布、所有 search stop 分布、首大跳变对应 policy 和首次实际 QP 残差。报告应保留这个边界，尤其不能把有限 12 次全部 stagnation 外推为真实全部调用。

## 6. 建议 W47 的最小控制工作包

### 6.1 先在服务器原位利用已有数组定位，不先重开 E30

先读已存在的 `r29-current-plan.bin.r33-candidates` / 相应 NPZ、选中完整策略、input/context/reference 文件，按 file SHA 和 fixture 对齐。对四个已有上下文逐 attempt 比较：best nominal、alpha0、最优正 alpha。已有 candidate 保存全 horizon 的 x/u/reference/K/H，先纯数组/四元数代数查出第一处分叉，不需要重复所有 12 个求解。

至少输出：第一非零及超过本次重复计算数值噪声包络的 `x⊖xbar` 和 `u−ubar` 节点；时间索引/离散索引选点；new/old reference 与 feedback scale；Clamp 前后中心、H、瞬时上下界、QP 活动面；该节点之前/之后接触与法向力变化；`J0`、`Jref`、每项成本增量。用已有 dV 按每个 alpha 重新列出原预测 `−alpha*(dV0+alpha*dV1)` 与实际下降，并注明投影/接触分支变化后 ratio 的解释限制。

若已有记录无法定位，再开展有限、成对的原 native 诊断：旧完整 policy 重滚；accepted u 的纯开环重放；同一 accepted x/u 上的新 K/H alpha0；已记录最优正 alpha。每条从同一完整重建初态启动，保存首分叉前后高率节点。开环重放也分叉时，先查模型/状态/输入索引；只有新 K/H 路径放大时，再查反馈、投影及接触敏感性。旧上下文缺失的历史私有状态依然明确列出，不能写成历史 exact replay。

此门的输出是证实 bug、证实局部敏感性或记录不足三类之一，不预先把结论写成“必须调整 K”或“必须 FDDP”。不能靠给 alpha0 放行、放宽容差、加任意动作 clipping 来让 gate 通过。

### 6.2 一个固定 donor，检验是否缺少可用运动候选

建议在当前 single-shooting 框架先做一个有限 donor 对照。donor 是**来自已验证 R8 硬地移动轨迹的动作/运动种子**，不是已经证明适用于 r35/TM02 的控制器。固定一个成功运行、一个片段和选段规则，核关节顺序、单位、500 Hz 时间、PD/模型差异与原步态相位。允许声明的 xy/yaw 坐标锚定与相位对齐；不换原 plant、质量、PD、摩擦/接触、任务阈值或 PANN，也不使用未来 W46 失败信息挑有利片段。

每个待测 current x0 上比较两个输入家庭：当前完整 incumbent 与固定 donor seed。donor 可用 K=0 作为初始化，H 沿用**同一当前 incumbent 在对齐时间网格上的有效 H**，投影仍调用原状态相关 H-QP；所有 seed 字段注明来源，不把继承 H 或零初始化的 Q 数组假称 donor 的已求解 Riccati 结果。之后仍走当前 N 的完整 backward/rollout/strict 接纳。

**必须固定同一参考问题。** 每个 fixture 先由当前 incumbent 生成一次共同 `ubar/F/R/context`，两个家庭都使用这份 reference 文件。否则 donor 一成为输入，r35 自动把它设为新自参考，目标定义也跟着变，便不能把差异解释成“仅换初始化”。始终保留已评分 current incumbent、current best nominal 与原 donor 的完整对象，用同一个 J 对 union 作最终比较；不得因为给 donor 单独开了新进程就忘记 current baseline。

**必须说明会保留什么。** 现 `NominalTrajectory` 会从当前 x0 重新推进 donor u，随后以实际 rollout 覆盖 nominal x/u。K=0 时 donor 的状态轨迹几乎不参与动作计算，当前模型可立即把它滚到别的接触模式。这项最小实现检验“执行输入种子能否在当前模型中生存并改善候选”，不等于完成了真正的 state-control/multiple-shooting warm start。donor x/contact phase 要作为对照保存，并记录重滚前后差异。

建议固定最多四个 fixture：两条已有坏 incumbent（B155/B158）加 A/B 各一个 W46 已保存的早期实际大命令上下文；按预先声明的原件可得性和时间规则选择，不按谁更好看选择。每个 fixture 两个家庭各一次原有限预算 solve，最多 8 solves；各输出在原状态和已记录尺度 ± 偏差下三次 350 ms fixed-policy 诊断，最多 24 次。后续不扩 donor 网格、R 网格或 seed 重试。

### 6.3 用结果选择下一机制

| 观测 | 能支持的判断 | 下一动作 |
|---|---|---|
| donor 在当前模型重滚就失去预期接触/运动，甚至不满足原约束 | 给旧成功输入不等于给当前问题可执行 nominal | 查动态缺口/接触与状态初始化；有证据后再单独评估保留 x/u 的可行性热启动或 FDDP |
| donor 当前模型内有更稳的前移且同一 J 更低，现有搜索此前找不到 | 缺少有用候选或局部搜索吸引域是主要障碍之一 | 可将固定 seed family 纳入正式实现，在原 strict 规则下验证部署 |
| donor 更稳前移但 J0 更高 | 当前任务成本与所需行为存在冲突的实证 | 先按分项定位冲突，再另立唯一目标修改，不用更多 PPO 掩盖 |
| J0 更好但加 Jref 后被拒绝 | 当前自参考惩罚阻碍该有用改变 | 明确这是参考/惩罚的证据，后续单独消融，不把它混入 donor 主对照 |
| nominal 好、±偏差坏或一次重规划后运动消失 | 只有静态种子有效，恢复/闭环保持尚未建立 | 查完整策略反馈与 receding-horizon 更新，不能直接宣布移动稳定 |
| 全部固定家庭都无效 | 本有限候选集未提供改进 | 有边界地结束，保留失败，不继续重复已知 E30 或“调到能过” |

只有形成明确有用的同定义候选和恢复证据，才冻结一个构建、一个分支，进行一次 A/B 原 E30；依旧原关卡逐级推进。较长存活、低成本或低 raw K 都不替代合法去程/返程与原稳定条件。

### 6.4 并行与工作流

两个 CPU 流各 10 workers，第三独立工作可以做现有原件聚合、方向一致性与 GPU 完整动作语义审查；不要为显存占用率加第三 PPO 或继续明知不等价的 rollout 排名。若第三项需要 native确认，保持其 host 约 2 CPU，并给主控/发布留余量。依赖 fixture/reference 冻结的诊断必须等共同输入冻结后再并行，不能同时改共享模型/引用目录。

后台无固定墙钟结束时间；有限样本/求解预算、物理 E30/E60/E300/E600、技术失败和人工停止仍是结束条件。最终先推紧凑证据、实现与 phase，再上传授权的原件增量 Release 并核验；GitHub 核验后不在用户电脑归档，不自动清理服务器原件。当前 W46 已完成，不能以“续接”覆盖其失败结果。

## 7. 本审查交付与复现

文件：[aggregate_w46_control.py](aggregate_w46_control.py)、[control-compact-aggregation.json](control-compact-aggregation.json)。脚本只用 Python 标准库，读取本地 Git 中四个 context JSON、mechanism/numerical/phase 和 A/B job JSON；聚合数值与文件 SHA 可复核，JSON 明确 `raw_array_recomputation_performed=false`。

```bash
python aggregate_w46_control.py --repo /path/to/MCP-RL --output control-compact-aggregation.json
```

已实际执行成功，核对 12 solves /36 diagnostics /732 rollouts /46224 transition FD /7522 risk FD 均与原记录一致。没有将被拒绝、未完成的批量原件读取脚本或空 provenance 作为验证产物。下一轮应将“方向为何不下降”与“能否注入一个在当前模型里有用的运动种子”分别回答清楚，才能判断下一步是修实现、改初始化，还是改任务目标/求解架构。
