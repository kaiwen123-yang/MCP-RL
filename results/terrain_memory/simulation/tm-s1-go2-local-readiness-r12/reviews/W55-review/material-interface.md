# W56 材料接口审查与最小闭环推进

2026-11-20。只读源码审查；没有修改仓库，没有运行物理或 QP。审查针对 r55 的实际继承链。父审已确认 r55 A/B 原 E30、E60、E300 均通过，E600 在途；材料新候选两个独立 holdout 退步，仍保留原 B ideal。

## 1. 明确结论

当前存在持续的世界地块材料状态，但它**没有进入 F/r55 的力 MPC 候选转移**。PANN 的主要控制通道仍是 12 个零载恢复特征 → 冻结 658D actor → 2 Hz 标量速度 cap。F 收到的是当前实测接触力/足端状态和一个固定刚度，不是 PANN 的每地块 e/q。

因此，后续应同时完成两件不同的事：实际闭环执行一次同 actor 的 12 维特征屏蔽，回答既有接口贡献；实现有世界地块身份、因果封包、候选独立材料演化的接口，在存在动作/可行性影响的条件下，以新身份直接开始原 E30。前者不能证明完整材料记忆的价值，后者不能只靠完成代码或离线残差变小宣称任务收益。

## 2. 已核实的数据链

| 环节 | 实际代码与行为 | 含义 |
| --- | --- | --- |
| 真 plant | `scripts/s1_go2_joint_material.py: JointMaterial.__init__/before_physics` 建立不带 weights 的 `Ground`，使用解析 `ReferenceColumns`；每 0.2 ms 按实际足底边界推进 | 真物理不是被冻结 PANN 驱动 |
| 学习材料 | `scripts/s1_go2_r10_observer.py: MaterialObserver.observe` 按当前 t 与上次时间差调用自己的 `Ground.step(..., log=False)` | 权重冻结，但按 world-cell 存储的 e/q 每次更新；不是材料状态也冻结 |
| 世界身份 | `src/tm02/s1_contact.py: Ground.footprint` 以 `(floor(x/.02), floor(y/.02))` 分配永久 key，并按相交面积分摊 0.001 m² 足底 | 同一世界地块可跨腿、跨步、返程复用状态；换腿不清零 |
| 恢复特征 | `MaterialObserver.recovery` 从当前 observer clone，对查询足底并集做 250 次 2 ms 零载恢复，取第 1/100/250 步 | 三个预测时距为 0.002/0.2/0.5 s，四脚合计 12 维；这不是受候选载荷驱动的未来 |
| 查询落点 | `scripts/s1_go2_r11_policy.py: PolicyMaterial.observation` 从已有 `latest_plan` 找最接近 `t+.2` 的节点，读取 `query_feet` 的 XY | 查的是旧已完成计划的预计位置，不保证就是下一次真实着地位置，也不保证仍有未恢复记忆 |
| 658D 输入 | `PolicyMaterial.observation`：前 106 维当前/任务/查询信息，`106:646` 是 20×27 近期历史，`646:658` 是 PANN 恢复特征 | 12 维屏蔽只屏蔽显式 PANN 特征，不屏蔽全部历史 |
| 近期历史 | `PolicyMaterial.control` 每 0.05 s 保存足底高度、法向力、水平足速、接触、机身速度、已用 cap；20 帧 | 即使 PANN 12 维全零，actor 仍看到约 1 s 的运动与力历史 |
| actor/action | `PolicyMaterial.control` 每 0.5 s 调用固定 actor；`PPOBackend.act` 用 `tanh(mean)`；`cap=.215+.035*action` | cap 范围为 0.18–0.25 m/s；critic 值在此记录，不是 F 的终端价值函数 |
| 既有零特征对照 | `PolicyMaterial.control` 的 `zero=obs.clone(); zero[:,-12:]=0` 只另算 cap 写 `pann-action-dependence.json` | 它没有真正执行 alternate 动作，不是闭环消融 |
| 到 F 的入口 | `scripts/s1_go2_r11_runtime.py` F 分支构造 `material_info`：足底力作用边界、GRF、足速、配置常数刚度 | 没有 observer、世界 key、e/q、恢复或候选加载序列 |
| F 接收 | `scripts/s1_w54_f_r52_backend.py: FController.update`，r55 继承此路径 | 材料信息用于当前执行；PANN 只可能经 cap 间接改变 `mission.v` |
| 力 MPC | R52 `replan/assemble_force_horizon` 与 r55 `replan` 均令 `force_normal_upper=inf`；14×25 ms；只含 12D SRB 状态及力 | 无候选材料状态、加载恢复转移、状态依赖柔顺代价或材料可行域 |
| 材料 WBC | R52 `wbc` 对软接触用 `p_des=p_now+(F_now-F_des)/k`，`k=1227272.7272727273 N/m`，法向速度取实测值 | 这是当前力反馈的固定刚度桥接，不是材料历史预测 |
| 摆动/未来落足 | R52 `_start_swing` 仅在当前 soft 时以 `p_now+F_now/k` 修局部着陆高；`_future_contact_feet` 的新预测目标高度设为足半径 | 没有针对目标 world-cell 的加载后恢复预测 |

r55 只在终点保持阶段改了 COM 力矩线性化；其他运动、材料 WBC 和 observer/actor 链继承 R53/R52。新材料工作不得覆盖该已验证几何修复。

## 3. 先确认原任务到底存在多大的记忆机会

**这是有待实测的机会假设，不能先宣布没有收益。** 主审已观察 r55 峰应变量约 0.00323，对 50 mm 厚材料约是 161.5 μm 位移；actor 的已查询恢复多为 μm 量级，现有 cap 影响很小。历史约 1 s 的局部恢复时间尺度也提示，一圈约 46 s 后复踩原地块未必还保有显著残余。这个局部时间尺度不能当作所有非线性历史的统一指数衰减定律。较短的前后腿复踩、连续部分覆盖与释放间隔仍可能有机会。

必须把“发生过重访”“控制器查了这块地”“查询时还剩状态”“状态能改变下一动作”分开统计。使用 r55 已有原件，建立一张事件级表，而不是重新做旧九地块或旧 cap 探针：

- 每次 actor 查询、下一实际着地分别记录 world key、足底重叠面积、腿号、去/返程、lap、首次/同腿/跨腿/跨 lap 类型。
- 对每个相交地块记录最后一次加载/释放到**查询时**的年龄、最后腿号、当时 observer e/q、零载剩余压入及目标加载下的增量压入。真 plant 内态只进入离线误差/机会标签。
- 查询机会分母必须是所有 query，而不是只筛非零值；同时给按相交面积加权的分布，避免一只脚触及 4 个格子被当 4 次独立机会。
- 记录查询 XY 与后来真正着地 XY 的世界 key 重合率、查询提前量与落地时材料剩余状态；不能用“地图有状态”替代“决策窗口命中”。
- 报告原物理分辨率下的幅值/误差和实际动作差，不因结果小就放大变形、软化 plant、改变材料厚度或绕路增加重访。

有一个记录边界：`PolicyMaterial.observation` 的 `query-exposure.jsonl` 分支仅列 r29–r41，未包括 r55；evaluation 的 `PolicyStream.transition` 保存恢复/cap/value，但不会像 training 那样写完整 rollout obs。应以已存每次 MPC `times/foot_center_plan` 和调用顺序重建 r55 查询，并用恢复/cap日志核对。actor 先于本 tick 的 F replan，不能误用同 tick 后完成的新计划。若原件不足，明确缺项；新身份时增加查询包与状态 hash，不回填成过去已记录数据。

## 4. 同固定 actor 消融：应实际执行，但限定预算

新增显式合同字段，例如 `material_feature_mode=full|zero_last12`，在 `PolicyMaterial.observation/control` 的最终 actor 输入处实施。保留 observer 正常更新与查询，另存 `obs_full` 和 `obs_actor`、两者 hash、屏蔽索引。固定同一个 W44 actor、normalization、case、seed、r55 控制、plant、路线及初态，唯一动作信息变化是最后 12 维置零。

**不要直接把历史 `PPO_RAW_S5` 另一套 checkpoint 拿来当这一对照。** RAW 可以研究不同训练路径的系统表现，但不能隔离 PANN 12 维在同一 actor 上的因果作用。A/B 也各有自己的 checkpoint，效果应分别与各自 full 基线配对。

建议在原 E600 A/B 自然结束后，使用新身份做 A/B 各一次同路线闭环消融。为了不产生四级阶梯的乘法膨胀，可将其预声明为最多 60 s 的原路线机制实验，报告实际完整往返/稳定及全程物理，**不把诊断身份称作通过原 E60 资格、不借 r55 的 E30 资格**。若 W56 最终坚持所有新身份必须先原 E30，则各做一次原 E30，通过后只追加必要的 E60 来覆盖返程；不自动扩展 E300/E600。两者由新计划在启动前固定其一。

这轮实际闭环可以得出“既有 12 维通道贡献小”的有价值结果；若差异小，不重新训练 actor 或反复改变屏蔽方式寻找显著性。它不能推出所有材料记忆无价值，因为近期 540 维历史仍存在，且 F 没有使用受载材料预测。

## 5. W56 同轮实现的最小新接口

### 5.1 复用已完成因果封包，接到在线消费位置

已有 `scripts/s1_w55_c_observer.py: observe_window`，不是待重新发明的算法。它接收连续十段 0.2 ms 的左端足底位置/速度和起止时刻，只在最后一段结束后发布 2 ms 状态；既有 `w55/c/causal-observer-window.json` 已完成有限部署函数与离线历史检查。

新实现应把每个**已经完成**的物理子步边界放入只读环形窗口，在下个 500 Hz 控制时刻消费该闭合窗口，替代 endpoint observer 更新；不能既消费窗口又再调用 endpoint `observe(t,...)` 导致双推进。不得传真 e/q 或未来子步。需要把有用的 `observe_window` 核心与依赖旧 C 文件路径的报告入口分离，但不改历史审计输出。

当前模拟器能提供 0.2 ms 足端边界，这是仿真内可观察接口；不能无条件声称实体 Go2 已提供同频同精度测量。所有使用新封包的完整闭环都属于新身份，不能加入现存 E600。

### 5.2 候选独立材料传播 API

建议新增独立模块 `src/tm02/s1_material_prediction.py`，由 F 新版本调用，接口至少包含：

`snapshot(t_obs, world_keys, e_hat, q_hat, geometry, law_hash, observation_clock)`

`rollout_candidate(snapshot, candidate_id, times, foot_xy, contact_schedule, normal_force_or_common_sole_boundary)`

返回每个节点/地块的自有 e/q、加载/释放/恢复、预测足底高度/反力、材料域余量、共同足底逆解残差、事件、输入/输出 hash。过去已知的 world-cell 状态与未来新增 virgin cell 分开标记。法向载荷与足底位移不能同时任意指定：以力为输入时必须先解相容的共同足底位移，再提交这一边界下的新 e/q；接触切换的零载区间也要积分。

**两个现有函数不能误用：**

1. `Ground.force_inverse` 内部 clone，只返回位移/残差；不把新 e/q 提交到外层。因此重复 inverse 不是“沿候选历史演化”。
2. `Ground.clone` 会重置 `prediction_steps/prediction_last`。若源本来就是带 lazy future 恢复的 branch，再 clone 会丢失未访问地块的待传播时间。当前 actor 从实时 observer clone 不受这点影响；新未来候选不能嵌套照抄。

最小可核实实现是：每候选从同一观测快照只 clone 一次，先建该候选预测域所有 footprint 的 world-key 并集，按相同时间分段推进全部并集列，未受载列也做零载恢复。这样避免 lazy 分支再次 clone 的歧义，也适合 GPU 按候选×地块批处理。若多只脚同时覆盖同 cell，保留 `Ground.step` 的同位移边界/较低足底归属规则；不能各腿独立求逆后把同一地块反力算两次。

重用 `Ground.footprint/step` 和 `LearnedColumns.step` 的材料与面积语义；不能把旧 `s1_planner.py` 的旧控制器整体接回 r55。旧代码有独立 clone/forecast 的结构参考，但带不同接触、静态力矩代理和执行语义，不继承资格。

### 5.3 最小真实执行改进及其范围

首个低风险、直接影响实际动作的位置是 R52 继承的 `FController.wbc` 材料法向目标，而不是再向 actor 增添一份报告。保持当前原接触事件、目标载荷、支撑、XY、摆动周期和所有 WBC 约束；用自身因果材料状态预测的力—位移增量替换固定 `1/k`：

`p_z_des = p_z_now - [delta_hat(F_target, z_hat) - delta_hat(F_measured, z_hat)]`。

两次逆解从同一只读状态、同一 dt 和同一 footprint 开始，均不改变真实 observer。这样在目标力等于实测力时不凭模型绝对高度误差平移足端。对每个 20 ms 计划仍应沿整条已有法向力序列独立演化并记录未来参考，当前 2 ms 的执行目标保持时序一致。

这一步的准确命名是**材料历史参与法向执行参考**；它仍不等于力 MPC 已联合优化材料历史。若机会表显示该增量差异小于已有观测/数值误差，不能为了得到动作变化放大增益或材料变形。

若机会和同输入验证表明未来受载状态会改变落足高度、原材料域余量或实际可选的力路径，则在同轮进一步实现**一个固定的材料耦合规划版本**：从同一快照生成 r55 原力序列和一次材料几何/边界修正后的力序列，各候选独立传播；使用完全相同的原任务评价/物理界，保留 r55 的终点 COM 线性化。禁止以当前静态力矩代理复制到未来，也不能单纯把 `force_normal_upper` 换成任意常数就声称完整材料 MPC。选中序列的材料参考、力矩臂和实际 WBC 法向执行必须一致，候选只通过预测检查也不构成真 plant 硬安全保证。

## 6. 因果对照与启动决策

材料新接口至少提供 `full_world_history` 与 `virgin_unloaded_revisit_cells` 两种预测模式。后者只在私有候选中对未来将复踩的、当前未受载的历史地块使用 virgin 假设；当前已受载地块、实时观测、真实 plant、actor、控制实现与候选集合保持一致。不要每 tick 全地图清零，避免把当前接触状态破坏也归为“长期记忆贡献”。原 observer `Ground.step(log=False)` 不维护 `ever_loaded/last_owner` 这类记录元数据，需从可观测足迹/受载事件建立控制侧历史标签，不能偷用 plant 真内态决定消融。

按主审决定，推荐本轮这样推进：

1. 当前 A/B r55 E600 继续，自然完成；C 并行完成 r55 机会表、因果封包接线和候选 API 实现，复用旧验证，不重复旧全矩阵。
2. 对新接口做必要的同输入验证：只读状态不被污染；零候选/重排候选结果相同；候选加载不同确实产生不同未来 e/q；先加载后他腿复踩沿同 world-cell；连续接触/释放时序不丢窗；共同足底反算与前算一致；原物理界、底/中心坐标与 r55 保持一致。这些是具体实现风险检查，不是新的物理任务。
3. 若原任务存在尚未恢复的可命中机会且新接口有可解释的动作或可行性影响，**同轮用一个冻结新身份启动原 E30**，不要停在方案。通过后仅对最有信息的 full/消融 pair 继续必要 E60；如 A 无影响或失败，不自动扩展成多模型×两 seed×四级阶梯。
4. 保持原 B ideal，本轮不因新接口而自动采用已经 holdout 退步的新权重；PANN 模型更新、observer 时钟更新、控制接口更新分别登记，避免混合效果归因。
5. 若查询机会在原任务中确实稀少，报告“该任务对材料记忆价值的辨别力有限”，保留实测分母与上限，不声称整个方法无用，也不为造收益改变原 plant。本轮仍应交付可运行的因果预测接口和已有接口的有限闭环消融，而不是只新增报告。

## 7. 文件落点清单

| 修改责任（未来 W56 实现） | 现有落点 | 建议变更边界 |
| --- | --- | --- |
| 显式 12 维消融/查询日志 | `scripts/s1_go2_r11_policy.py: PolicyMaterial.observation/control` | 不用 RAW checkpoint 替代；新身份字段与输入 hash |
| 已完成窗口发布 | `scripts/s1_w55_c_observer.py: observe_window`；runtime 子步边界 | 复用已验证函数语义，移除报告路径耦合，单次消费 |
| 独立预测 | `src/tm02/s1_contact.py` / `s1_material.py` + 新模块 | clone/commit 明确；共同足底与面积一致；不改真 plant |
| F 接收只读状态 | `scripts/s1_go2_r11_runtime.py` F 分支 | `material_info` 扩成显式带时间/哈希的 observer packet |
| 法向执行参考 | 新 F backend，继承 r55 的 R52 `wbc` 点 | 替换固定 k 增量；所有原硬界保留；新身份 |
| 真正候选材料规划 | 新 F backend 的 `replan` / `assemble_force_horizon` 输入 | 先独立传播，再有限同范围几何/材料边界修正；不回退 r55 终点几何 |
| 试验登记 | 新 config / launcher / per-job 合同 | 原 E600 不变；每变化新身份；一个冻结版本，有限闭环预算 |

