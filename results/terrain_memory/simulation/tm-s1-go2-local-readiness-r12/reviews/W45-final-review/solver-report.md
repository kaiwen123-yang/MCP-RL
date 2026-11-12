# W45 / r34 控制链审计：接纳修复有效，但 native 完整策略仍规划了剧烈执行动作

审计对象：`kaiwen123-yang/MCP-RL`，提交 `3e3f69c440d6a33c14131563e5f485d88da3a803`。本报告只做已保存原件的读取、逐字节校验、源码审查及 NumPy 数组复算；没有启动仿真、训练、服务器任务，也没有修改运行控制器。

## 1. 结论

**W45 排除了上一轮一个真实的接纳错误，尚未获得稳定的原任务控制能力。** 全部 A674 / B300 次 planner call 都启用了 r34，新的 alpha=0 策略安装均为 0。两次独立原 E30 仍分别在 13.072 s / 5.582 s body_fall。不能把本轮继续失败归结为“alpha0 修复没真正生效”。[最终状态](https://github.com/kaiwen123-yang/MCP-RL/blob/3e3f69c440d6a33c14131563e5f485d88da3a803/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w45/phase.json)

**更关键的新证据是：部分剧烈动作已存在于被选中的 native 完整闭环策略中。** 它们不是全部由真实 plant 偏离一条平稳预测轨迹后才生成。A 在 2.650 s 的 native 相邻 2 ms 已执行目标跳变为 1.41880 rad，实际为 1.16403 rad；B 在 2.764 s 对应 1.23629 / 1.23587 rad。B 的实际扭矩相邻变化达到 85.6056 Nm。对齐积分器、warmstart、通信精度值得有限鉴别，但单独校正它们不足以解释并消除这些 native 自己已经规划的动作。

**大 `Kδx` 本身不是一个不依赖表示方式的失败指标。** 当前完整 policy 有自己的线性化参考；native rollout 离开该参考时，其反馈项本来就可很大。等价重定中心可以改变“前馈/反馈”的分解。真正应改善的是同一完整策略输出的投影后关节目标、PD 扭矩变化、接触与支撑恢复，以及原物理任务结果。后文给出的相邻目标、扭矩、native–actual 差异不靠 `Kδx` 的分解方式成立。

下一轮最有根据的一个控制方向，是**将接触承载信息引入对共同输入参考偏离的实际 MPC 代价，使该项同时进入 native 评分、导数、Riccati 的前馈/反馈与 H**。这是一项待证机制，不是已经证明的修复。只给 `Quu` 加 scheduled D、只压 raw K、或给接纳层加额外阈值，都没有解决本轮已观察到的完整名义策略质量问题。当前也没有一条已证明稳定的完整移动参考，因此必须明确处理“围绕坏 incumbent 加惩罚反而锁住坏策略”的反例。

## 2. 覆盖范围与复算完整性

读取并校验了 W45 A/B 的 37 个控制原件，合计 511,521,756 bytes；包括全量 planner calls、feedback/action-execution、5000 Hz physical、prediction、关键完整 policies 与 candidates。另使用主审计已校验的两个完整 trajectory（9,727,108 bytes）。随后补取了 4 个实际构建源码原件。下载采用发布 ZIP 的明确 byte range，**逐成员核验仓库 source manifest 的 size 与 SHA256**；本控制审计没有把部分 ZIP 下载冒称整包本地 hash 校验。完整来源、归档 member、URL、大小和 SHA 均在 [control-originals.json](control-originals.json)。[A 原件索引](https://github.com/kaiwen123-yang/MCP-RL/blob/3e3f69c440d6a33c14131563e5f485d88da3a803/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w45/delivery/w45-a/sources.json)；[B 原件索引](https://github.com/kaiwen123-yang/MCP-RL/blob/3e3f69c440d6a33c14131563e5f485d88da3a803/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w45/delivery/w45-b/sources.json)

四个有限上下文为 A148 / plan136 @2.54 s、A153 / plan140 @2.64 s、B155 / plan141 @2.68 s、B158 / plan144 @2.74 s。每个选中 candidate 与实际安装 policy 的 13 组数组——accepted/reference 的 state/action/time/residual、K、H、Qu、Qxu、active set——最大差均为 **0**。按当时 task weights 与 XML norm 定义逐项重算 native J，最大误差 **5.55e-17**。因此以下不是仅看日志标签推断。

复算脚本 `recompute_w45_control.py` 与 `policy_arrays.py` 不导入 MuJoCo；运行前自动验证 18 个直接依赖控制原件及 2 个 trajectory 的大小和 SHA。输出 `control-summary.json`，只保留必要数值，不复制大数组。

源码从 r25 快照反向还原 r24，再按真实继承关系应用 r26→r34。22 个关键文件中初次 18 个与 build SHA 一致；其余实际源码已从 W45 mechanism release 补取并匹配 build SHA。三个差异仅为 CRLF/LF，另一个是 r29_projection.h 尾部被相同 include guard 排除的旧定义。planner、policy、backward pass、model derivatives、quadruped 等关键实现均有实际构建字节证据。[构建记录](https://github.com/kaiwen123-yang/MCP-RL/blob/3e3f69c440d6a33c14131563e5f485d88da3a803/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w45/build.json)

这不等于语义读懂全部历史大原件的每个 byte，也不构成完整物理历史重放。记录没有保存全部 qacc warmstart、求解器内部历史和材料内部状态；本轮没有用不完整状态伪造历史复现。

## 3. 全量 call 与实际执行统计

| 指标 | A | B |
|---|---:|---:|
| planner calls | 674 | 300 |
| positive-alpha source | 583 | 235 |
| nominal-scale source | 46 | 32 |
| incumbent retained | 45 | 33 |
| 新 alpha0 安装 | 0 | 0 |
| 明确拒绝的 alpha0 candidate 记录 | 1,127 | 589 |
| native 完整候选评分数 | 18,010 | 8,890 |
| native_search_stagnation calls | 90 | 65 |
| backward_failed calls | 1 | 0 |
| 实际 applied_hold | 6,536 | 2,791 |
| effective feedback >0.5 rad 的 hold | 144 | 82 |
| projection >0.5 rad 的 hold | 78 | 39 |
| applied QP failure / restoration | 0 / 0 | 0 / 0 |
| 最大 scaled KKT residual | 6.41e-9 | 5.30e-9 |
| 最大 primal residual | 0 | 0 |

0.5 rad 是描述性统计分箱，不是新增任务失败条件。planner 的候选拒绝、stagnation 与物理 episode 失败不是同一种计数。W45 的物理结果是两次 E30 失败；不能把 26,900 个候选评分或所有 line search 拒绝当作 26,900 次物理失败。

首个持有期间未裁剪 PD 请求超限发生在 A2.8962 / B2.8344 s，比下文首个剧烈动作晚。整个 physical 记录的未裁剪请求超限比例为 2.243% / 1.806%；这不等于实际 actuator torque 越过物理力矩限制。native 和实际都继续实施原 force clamp。未发现本轮由 QP 解失败、恢复分支、坐标符号错误或未评分 K 偷换导致首个事件的证据。

## 4. 首异常的 native–actual 对齐

### A：2.64 s 的有意义正步已规划了高频目标变化

A153 的 incumbent/native best nominal 为 candidate000，scale=.001；选中 candidate018，alpha=1、scale=1，当前 native J 从 .11683448045 降至 .08909507075，收益远大于约 3.9e-11 的算术比较预算。四元数 StateDiff 与同一时间插值重建实际记录误差在机器精度范围内。

| 物理时间 | native effective feedback rad | actual effective feedback rad | native 相邻目标变化 rad | actual 相邻目标变化 rad |
|---|---:|---:|---:|---:|
| 2.642 | .18555 | .19641 | .17371 | .18457 |
| 2.646 | .46555 | .42804 | .66457 | .63349 |
| 2.648 | .58843 | .46256 | .80213 | .53613 |
| 2.650 | 2.28746 | 3.12302 | 1.41880 | 1.16403 |

初始 native/actual 状态一致，首 held target 差仅 1.30e-5 rad。第一段 2 ms 后最大关节位置差 2.62e-6 rad、最大速度分量差 .0100；到了 10 ms 位置差 .0005215 rad、速度分量差 .60898；20 ms 位置差 .0048441 rad、速度分量差 .56471。完整记录显示小模型/数值差异随后被闭环放大，同时 native 自己也已经安排了上述大动作。

native 在 2.648 s 已预测全足无接触、native/proxy Fz=0；实际首小片段为 2.6464–2.6492 s，并在 2.6506–2.6562 s 再次出现。native 不是在该区间预测稳固承载而 actual 无故掉了支撑。但 **短暂腾空本来可以属于合法 trot，不能独立判失败，更不能据此添加“四足必须始终着地”的新门槛**。

更直接的执行问题是 A2.650 的实际相邻扭矩变化 74.0108 Nm，2.652 的某膝关节跨越正负限值，变化 90.86 Nm。这些是投影后已施加命令造成的变化，不能通过重命名前馈/反馈消除。

### B：名义完整策略的剧烈输出更接近实际

B155 @2.68 s 选择 nominal_0/candidate001，scale=.001，仍保留从 2.54 s 开始的内部 policy reference。其 native 已在 2.694 / 2.696 s 预测全足无接触与零支持力；实际首次 500 Hz 全足无接触也在 2.694 s。第一段 2 ms 后 q/v 最大差约 7.91e-6 / .00642，native/actual 目标差只有 2.15e-6 rad；10 ms 后相应约 4.86e-5 / .01789，目标差仍只有 1.05e-5 rad。此处实际基本在执行所选 native 的动作。

B158 @2.74 s 选 nominal_3/candidate004，scale=.0133352，J 从 .31972310615 降至 .12416392145。到 2.762 s，同一已安装 policy 仍在执行：native feedback=.736150 rad，actual=.740444；native 已施加目标相对于内部前馈偏移 .865493 rad，actual=.872367 rad。相邻目标跳变是 .864029 / .870424 rad；下一节点 2.764 s 则是 1.236289 / 1.235869 rad。2.762 s 的 actual/native 状态误差却只有 q≈.0001126、v≈.00946。

这组证据尤其不支持“把 Euler 换成 implicitfast 就会让这条名义策略平顺”的确定性断言。模型一致性仍值得校正，但要与 nominal 执行质量分开衡量。

首个大幅动作到最终 body_fall 之间还有多次重规划；A 又继续了约十秒。本报告没有证明某一个早期节点是最终跌倒的充分原因。

## 5. 为什么 native 仍会选中这些动作

A153 的逐项成本解释了选择，而不是显示一个比较器错误：

| 成本项 | best nominal | selected alpha1 | 变化方向 |
|---|---:|---:|---|
| MaterialSupportHeight | .08556680 | .02597716 | 明显改善 |
| Height | .00424858 | .00240363 | 改善 |
| Position | .00838650 | .00679512 | 改善 |
| Effort | .00484759 | .03537459 | 显著增加 |
| Upright | .01039093 | .01311177 | 恶化 |
| Gait | .00131338 | .00357952 | 恶化 |
| 总 J | .11683448 | .08909507 | 当前目标确实下降 |

selected 在 .35 s 内的前进预测从 −19.97 mm 变为 +15.22 mm，最低 base z 从 .24846 m 改善到 .27204 m。这是一个有真实局部收益的候选，并非应当简单改回“完全不动”。问题在于当前目标允许用更大的执行动作和更多接触转换换取这些收益，尚未形成可持续的完整原任务行为。

B155 的总 J≈.25278，其中 MaterialSupportHeight≈.20049。B158 的多个姿态和高度项也随选择改善，而剧烈目标变化仍出现在它的完整 native rollout 中。仅仅检查“原目标下降”和“当前 trunk violation=0”无法保证长期稳定。

当前绝对高度目标来自 native stand 的 .29020336 m；moving 时还保留相对平均四足中心的 .27 m 高度项。两个参考的几何关系、缺少稳定移动参考、短预测窗口与未证明的终端可恢复性，都是后续应保持可见的限制。但 r14 曾为避免离开材料带就关闭支撑目标而将其全局启用；不能据本轮分项直接撤销绝对支撑项、降低原物理门槛或把短 flight 全禁掉。

## 6. 完整 policy 表示是否有重定中心 bug

审查结论：**没有证据表明本轮是遗漏了下一轮线性化的重新定中心。** `NominalTrajectory` 会把当前最佳 native rollout 的 states/actions 送入下一次局部导数与 Riccati。alpha rollout 则围绕该线性化轨迹构成一个完整 feedback law；最终部署保持被 native 评分的整个 policy，包含原 reference、K、H 与 scale。

在欧氏固定时刻，若 `u=u_ff+sK(x−xref)`，选择新中心 xbar 并令 `u_ff'=u_ff+sK(xbar−xref)`，可保持未投影 action law 不变，而让 xbar 处的反馈项变为零。这里还有四元数切空间、K 与 reference 的时间插值、Clamp 和 H 投影；仅把 post-projection accepted_actions 配 same K 放回 reference，并不是这个等价变换。

因此，大 native `Kδx` 应作为定位信息；**不能把它的下降直接当作新控制器成功，也不能让 r35 的新代价取决于每个 candidate 自己随意选择的内部 reference。** 新项必须相对于同一外部冻结输入参考，作用于完整 native 执行的目标或其明确 PD 等价量。评估重点是投影后目标/扭矩变化、支撑/进度与实际 E30。

## 7. 建议的单一机制与最小有限对照

建议保留 r34 的接纳语义，以一个新的**承载调度输入参考代价**做主机制。HiLQR 的 time-stepping 四足实例使用 load fraction `w_j=Fz_j/ΣFz`，及 `R_j=Rmax−w_j(Rmax−Rmin)`；它针对 `(u−u_ref)`，其原始输入是 torque，而且参考来自已有稳定控制器。因此不能把论文的“加入 R”当成当前已有稳定参考的证据，不能把 torque 单位直接搬到关节目标。

实施必须满足下列关系：

1. 每个 problem epoch 从同一个明确基线生成输入参考和每节点载荷 schedule，全部 candidate 共享冻结字节。全足无承载时明确 w=0；足序到 12 个关节按模型名字映射。不能每个 candidate 自己改 reference 后互相比不同的目标。
2. 将新项作为真实 native cost，并贯通 residual、C/D、cost derivatives、Riccati、完整 baseline/nominal/alpha/archive 评分与日志。仅改 `Quu` 的 LM 正则不会改变旧 native J 的偏好。
3. 避免循环定义：先取得当前 epoch 的完整 native 基线轨迹，再冻结 schedule；此前已按旧 J 归档的 baseline/nominal 必须在新口径下重算或明确重新计分。不能让 alpha 使用新 J、incumbent 使用旧 J。
4. qtarget 的单位可通过 `Δτ=kp·Δqtarget` 映射；kp=60。当前 moving Effort 的单位曲率基准为 `.08·(.02·60)^2=.1152/rad²`，再除176得到现 D 量级。这只标定单位，不能证明任何 Rmin/Rmax 比值最优。论文没有一组可直接照搬的 R 数值。
5. constant 对照匹配 load-scheduled R 的每节点 trace 平均；否则效果可能只来自总体惩罚加重。新旧 J 定义不同，跨 variant 应比较原目标分项、输入参考新项、执行动作与支撑/进度，不直接比较一个总 J 数值。

有限比较仅用四个已保存上下文与三个清晰 variant：现 r34、共同输入参考的常量代价、相同平均强度的承载调度代价。它能区分“新增输入项”与“承载信息”的作用，不需要参数网格。A148 只是前20 ms 较平稳对照；其完整 .35 s 内 native 最大目标跳变仍有 1.50591 rad，**不能称为稳定的整段移动参考**。

必须主动查的反例是：如果新项围绕已经不稳的 incumbent，而 incumbent 对新项恰为零，则可能锁死坏 nominal；如果仅 raw feedback 变小但 post-projection target/torque 不变，则只是分解变化；如果控制变平顺但退回长期无前进，也没有解决原任务。出现这些情况，应如实记机制不支持，不以新增 PPO、更多同配置 E30 或更宽 sweep 补样本。

若有限结果支持真实执行动作与名义支撑/进度改善，随后两 CPU 流应直接使用冻结 update57 的两个原任务 E30；后续 E60/E300/E600 继续以真实前级通过为条件。第三 GPU 任务必须对这一具体方向提供可回用的独立证据，而不是拿丢掉 K/H 的 reference-only 排名冒称在线完整 policy 评估。无固定后台 wall deadline；数值有限预算与原物理观察阶段仍须明确。

## 8. 仍不能从当前证据宣称的事情

不能宣称单个 K 值、一次短 flight 或一个同源低成本 rollout 已证明终局因果；不能宣称 Euler/warmstart 的贡献已经分离；不能宣称承载输入代价会自动得到可行 gait；不能宣称当前候选完整模拟了未来 TM02 世界材料记忆；也不能把一次 E30 的少量材料接触视为合法穿越或往返。

当前最稳妥的推进方式是：承认接纳层已修好，利用本轮完整 policy 原件把研究问题推进到**什么名义输入与接触执行真正值得优化和保持**，然后用两条实际原任务结果评价这一项机制，而不是继续用局部评分下降替代任务完成。

## 复算

```bash
python recompute_w45_control.py \
  --originals-dir /path/to/control-originals \
  --trajectories-dir /path/to/trajectories \
  --provenance control-originals.json \
  --output control-summary.json
```

控制原件目录保持 `w45-a/`、`w45-b/` 子目录；trajectory 文件名为 `w45-a-trajectory.npz`、`w45-b-trajectory.npz`。脚本依赖 Python 3 与 NumPy，只进行校验和数组运算。
