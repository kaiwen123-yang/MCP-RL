# W49 控制实现、有限证据与 W50 最小机制审查

审查基点：`de6606dc8363e8ab1c816fa682cace752a06318a`；上一计划：`b142001624cfea5877c57e63f45e0f29c9dc4735`。

本审查只读仓库源码和已提交 JSON，并做派生算术。没有访问 D57，没有下载大原件，没有新增 MuJoCo/Warp 物理、native solve、transition FD、PPO 或测试。`recompute.py` 可复算本审查 `analysis.json`。为核对调用链，仅在 scratch 重建了 r35→r36→r37 五个 native 源文件，重建仅供本次阅读，不作为交付依赖；正式源码证据是仓库中的r35/r36/r37 patch及w49_native_source.py。审查时未修改实验实现。

## 1. 控制侧结论

W49 是一个有真实动作差异、局部 native 严格改善、原任务轨迹也已改变的新机制。不能再将它描述为没有推进，也不能将两次完成 30 秒描述为稳定行走成功。最新原评价仍为 E30 0/2，材料去程/完整任务均未完成；低姿态、非足支撑与不前进仍存在。

源码中没有发现把两时间网格按 index 错加、把 delta 再乘一次 alpha、用 accepted rollout 冒充 frozen reference、或用 N 的 dV 冒充新家族 ratio 的直接错误。W49 的有限验证本身支持该实现的短窗动作和部署语义。

**值得优先推进的新问题是反馈策略生命周期。** 在线 K 从首个有效 baseline 开始，每次只更新独立前馈 delta；当前 backward 虽然计算了新的 K/H，却不安装。nominal 筛选只能改变旧 K 的 scalar，不能刷新旧 reference/K/H。这样的继承关系可无限持续，尽管每次安装都获得新的 plan ID、generation time。旧参考网格过期以后，ActionImpl 对旧参考、K/H 使用端点夹持，新 delta 仍在当前网格上滚动。这是源码确定的机制边界；它对当前低姿态失败的因果贡献尚未经过独立干预证明。

## 2. W49 有限结果的实际含义

来源：`w49/task-b.json`、`w49/deployment.json`，相对路径均位于 `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/`。

| W48 保存问题，用于 W49 有限构造 | baseline J | 最佳 K J | 降幅 | alpha | 合格 K / 9 | 原 N 正候选合格 / 实算正候选 |
|---|---:|---:|---:|---:|---:|---:|
| A213，3.84 s | 2.2199178750 | 2.1494116493 | 3.1761% | 0.4216965 | 9 / 9 | 1 / 9 |
| B176，3.10 s | 0.1367666493 | 0.1112284560 | 18.6728% | 1 | 8 / 9 | 0 / 45 |
| A364，6.86 s | 0.1703863504 | 0.1670828227 | 1.9388% | 0.4216965 | 3 / 9 | 0 / 45 |
| B328，6.14 s | 0.2872743823 | 0.1907935175 | 33.5849% | 1 | 5 / 9 | 2 / 9 |

36 个 K 中 25 个按原 strict 规则改善；108 个原 N 正候选仅 3 个 eligible。不是等算力大样本比较，也不是成功率估计，但足以说明冻结反馈并取首 N 方向的构造确实绕过了这四例中的部分问题。

四例选中 K 的风险均为 0，cost 下降远高于各自约 1e-10 的求和 roundoff budget。其最大动作差分别为 .281521、.373590、.964038、1.133265 rad；后两例动作差不小，不能把“严格 cost 下降”自动解释为更平滑或更安全的全过程。

有限 Task B 实际先执行完整 N，再追加首个有效方向的 9 个 K：N 总 rollout 为 20/60/60/20，K 为 9/9/9/9，总计 196 native rollout、8 derivative pass、15,408 transition FD、3,618 risk geometry FD、12 backward attempts。B176/A364 的首 backward 在 mu=1 已有效，但 N 搜索全部无下降，继续到 mu=1024 后才调用 W49Guided；因此 guided JSON 的 `mu_before=1024` 不能当作该 k 的生成 mu。首次方向身份必须来自 first-capture/attempt 记录。

线上 K 的有效 baseline 路径则在首有效 backward 后立即做 9 个 K 并返回，跳过这些 N 正候选。有限与线上评分、动作家族相同，搜索工作量和后续 damping 历史不同。应据实描述这一界限。

部署为 4×20 ms，400 predictor 子步，0 新 solve/FD，四者 import/export 完整、metric/QP 有效、RPC 返回等于单次 FP32 cast 的 native action。最大响应舍入 1.18542e-7 rad。四个短窗的 x 增量仅 .000204、.000455、.000709、.002529 m。它们证明保存对象经实际接口可执行；不证明持续支撑、材料穿越或恢复成功。

这些不是历史 bitwise plant replay。输入明确 `historical_exact_solve=false`；历史完整数值/外部材料状态并未全部存在。部署使用共同声明的 native predictor 冷启动和接口，不应据此排除历史隐藏状态的影响。

## 3. delta 与严格接纳的源码审查

关键源码：`scripts/w49_native_source.py` 与 `experiments/terrain_memory/s1-go2-r11-S5-r37.patch`；重建源码便于核读，但提交文件是正式证据。

1. `W49Capture` 在首个有效 backward 后、任何 ActionRollouts 的 alpha 改写前保存 k；时间取当前 `candidate_policy[0].trajectory.times[0:horizon-1]`，基线取本次已评分 `r33_nominal_policy`，不是默认 P0。
2. `AddDelta` 将旧 delta 和本次 `alpha*k` 在当前动作域的 union knots 物化为一条已缩放线性曲线。alpha=0 直接 return，不重采样。ActionImpl 先求旧名义前馈和 scaled feedback，再加 delta，再走原 Clamp 和 H-QP；运行时不再乘 delta_alpha。
3. CopyFrom、Reset、完整 binary Save/ReadDelta、pre/post observation transport 均携带/重置 delta；H metric_valid 进入 schema49。ActionIndex 显式拒绝带 delta 策略，不会使用错误的参考 index 来求新时间网格。
4. NominalTrajectory 把实际 rollout actions 用作 N 线性化工作对象时清空该工作对象旧 delta，避免把已包含 delta 的实际 action 再加一次；先保存的 B 仍保留旧完整 delta。R35Freeze 经 ActionImpl，因此新输入 reference 也包括 delta。
5. `R33Consider(kind=1)` 要求正 alpha、metric 有效、有限、candidate 不失败，且相对本次 best nominal 在原 trunk violation/J 次序严格改善，才有资格替换 archive。原 trunk 次序没有被放宽，equal-risk 的 cost 差仍必须超过双方 R34Budget 之和。
6. 新 K source 使用 `k_guided_`，日志 `chosen_expected`、ratio 为 null。W49Guided 保留 mu/rate，只独立保留原 violation-progress regularization 规则。不存在直接拿新 family 的 cost 下降除以原 N dV 的操作。

delta 的端点常量延续是原计划声明的 interpolation/clamp 语义，不是突然发现的时间精度错误。新的重点是更新策略参考和反馈，不是再次改变 float/double query time。

## 4. 永久冻结与观测证据边界

重建源码定位：

- `planner.cc` 716：NominalTrajectory；787：清线性化工作 delta。
- 1065：每个 backward attempt 恢复 current nominal 并清工作 delta。
- 1188–1198：首次有效方向 capture；K + valid B 进入 W49Guided 后立即返回。
- 1555：W49Capture 保存 `w49_base=r33_nominal_policy`。
- 1570：W49Guided 的每个候选 `candidate_policy[i]=w49_base`，然后只 AddDelta。
- `policy.cc` 292：完整 CopyFrom；320/325：DeltaAction/AddDelta；原 ActionImpl 的 reference/K/H 随 reference grid 插值，区间外夹到端点。

这是归纳不变量：valid bootstrap 后，K winner 仅改变 delta，nominal winner 仅改变 feedback_scaling，incumbent 完全保留；三个路径都不生成新 reference/K/H。在没有 Reset、offline import、或再次 invalid bootstrap 的普通 online 路径上，最初 K/H/reference 持续继承。R32BeginProblem 重置搜索 epoch/mu，不重置或刷新 policy 的这些数组。

已提交事件日志 A 116 个、B 115 个调用：两流首 call 均为 t=0 的 `alpha_0_8` bootstrap，call2 为 K；保留的后续事件仅见 K、nominal、incumbent，所有保留的 metric 均 valid。与这个不变量一致。源码与初始 horizon .35 s 推出原 reference_end 约 .35 s，末次调用潜在端点超龄约 29.65 s；**此数值是源逻辑推断，不是读过 NPZ 后的数组实测。**

完整 policy NPZ 不在当前仓库文本中，只提供了对象 manifest；本次没有下载它们。故不能声称已逐数组证明 K/H 在全 1521 个调用逐位相同，也不能提供全程实际 feedback/reference age 曲线。服务器上的既有少量 startup、首异常、末期 policy 对象即可补上这点，不需要新增物理。

`policy_age_s` / `P0.generation_time` 是完整策略最近安装的年龄。A 末期 .22 s、B 末期 0 s，不是反馈数组年龄。另外 ActionImpl 的 `metric_source_plan_id` 直接输出当前 `record_plan_id`；K 安装递增 ID 而 H 不变，故该标签也不是 H 真正来源。W50 应分别记录安装身份、reference origin/end、K/H 来源与混合 provenance；不增加泛化哈希门禁。

## 5. 支撑成本很大、却没有前进：目前能证明什么

末事件 A 的 J=7.372798、MaterialSupportHeight=7.171418，占约 97.27%；B 的 J=6.940759、Support=6.744711，占约 97.18%。MaterialSupportHeight 定义为 0.5×mean(((z−.29020336)/.04)^2)，所以对应的 horizon RMS 高度误差约 .1515/.1469 m。实际末高度 .138718/.142818 m 与大幅低姿态问题一致。这说明优化接受的计划仍有很大的支撑缺口，不是高度成本消失了，也不是仅仅评分容差造成的假下降。

与此同时 Position 成本只有约 .0072/.00765，这不能直接证明“目标接错”。`r26_reference.h` 的 R26HeadReference 覆盖全局 mocap goal，每次 replan 以实际 base x 为锚生成局部参考；.15 m 最大 lead/.25 m/s 最大 speed/.35 s horizon 给出 .25 s lead time。在 .215 m/s 简化匀速例子中，当次 lead 为 .05375 m，horizon 末约 .129 m，再加入 .3 m head offset。任务层的 body_target=3 m 不等于 native Position residual 直接用 3 m 误差。

这个局部参考方案避免过远参考，但它没有独立保证全局进度。配合过期反馈、接触/受力约束和非光滑局部方向，可能持续在低姿态大代价附近找到一点改善。这里只能提出机制联系，尚不能宣称已证明某一个权重、K 或局部参考就是唯一失败原因。没有证据支持先提高 Position/height 权重或恢复 PPO 就会解决。

## 6. W50 一个明确、有限的新机制

建议唯一主候选：**当前 horizon 重参考，并用同一个既有 alpha 转移前馈、反馈和投影 metric。** 不同时改变 r26 目标、代价权重、步态、PPO、原任务边界或时间精度。

每个新的真实 state/time/reference epoch：先按现有方法 native 评分得到完整 B 和其当前 rollout X_B；正常完成 soft-D + native 方向两 pass，最多原 5 次 backward，仅到首个有效 backward 就保存 k_N、K_N、H_N、mu、active 和其 current grid，随后只构造 9 个原 positive-alpha 槽。不要像 W49 finite 先跑完整 N 再拿末次工作对象；不能混用首 k 和末 K/H。

### 定义

令 `c_B(t,x)` 是 B 原 native ActionImpl 中 **Clamp/H-QP 之前** 的完整 center：旧 u_ref + s_B K_B StateDiff(old_ref,x) + 已缩放 delta_B。

在当前 rollout 节点 x_j=X_B(t_j) 上，以 MuJoCo 当前状态切空间 eta 定义：

\[
c_j=c_B(t_j,x_j),\qquad
L_{B,j}=\left.\frac{\partial c_B(t_j,x_j\oplus\eta)}{\partial\eta}\right|_{\eta=0}.
\]

L_B 已包含 installed s_B。可通过 native raw-center 的零物理 central FD 求得：qpos 用 mj_integratePos 的 tangent 扰动，qvel/act 按对应坐标扰动；沿用声明的 fd_tolerance，不做 epsilon 扫描。它不是 mjd_transitionFD，也不推进 physics；单列 raw-center evaluation 数，避免把额外工作藏起来。Go2 nx=36、175 动作节点，中心差分最多 12,600 次 raw-center 求值，加 175 个中心本值。不能直接将旧 quaternion reference 坐标中的 K_B 当作新参考坐标中的 K。

对于每个 alpha>0：

\[
\begin{aligned}
x^{\rm ref}_{\alpha,j}&=x_j,\\
u^{\rm ff}_{\alpha,j}&=c_j+\alpha k_{N,j},\\
K_{\alpha,j}&=(1-\alpha)L_{B,j}+\alpha K_{N,j},\\
H_{\alpha,j}&=(1-\alpha)H_B(t_j)+\alpha H_{N,j}.
\end{aligned}
\]

存成一个普通完整策略：当前 t_j 网格、current x_ref、上述 u_ff/K/H，`feedback_scaling=1`，旧 delta 已在 c_j 内吸收，新 delta 清空。N 原正步 ActionRollouts 的 K_N 实际尺度就是 1；不能再乘 B.nominal_feedback_scale，也不能将最终 K 再整体乘 alpha。末状态按已有规则，175 动作自由节点，最后一个 K/H 重复行不变成新控制节点。

**alpha=0 不是上述构造的一个重参考策略，而是直接 deep-copy 原完整 B、沿用其原评分。** 由于 StateDiff/姿态图和重参考是局部近似，正 alpha 分支不能声称与 B 全局恒等，也不能声称 alpha 趋零时每个任意状态的动作都严格趋于原 B。新的候选必须按实际对象完整 native 评分；比较基线始终是未改动 B。

候选全部使用原连续 ActionImpl/Rollout 的实际线性插值、Clamp、H-QP 与同一 x0/context/r35 目标，不用 ActionIndex 冒充部署。原 native 严格风险/J次序裁决，expected/ratio 仍 null，不用新的家族比值调整 damping。赢家原对象安装、序列化，下一次作为完整 B 使用，不维持无限增长的 base-policy 对象链。

H_B 与 H_N 均来自有效 metric，凸混合保持 SPD 的数学条件成立；仍由原 ActionImpl 的正常投影行为决定该候选是否失败。混合 H 没有单一 mu：必须保存 baseline source、当前 N source、alpha 和 first-valid mu，不能把 N 的 mu 冒充最终 H 的唯一正则化来源。base_Qu/base_Qxu/active 属于原 backward 证据，并非混合策略全局有效的新导数；要在 schema 中注明，不应伪造等价关系。

这确实不同于旧 N 对任意小 alpha 都整包换入 K_N/H_N 的行为，同时解除 W49 永久不刷新参考的限制。它是一项待验证的局部重参数化假说，不是已证最优控制算法。

## 7. 四个 W49 新事件与合理工作量

| 用途 | W49 event call | origin | 实际高度 | 当前 J | 理由 |
|---|---|---:|---:|---:|---|
| 早期支撑丢失 | A237 | 4.32 s | .290605 m | 2.481991 | 实际高度尚正常，预测 Support 已 2.372188 |
| 早期支撑丢失 | B236 | 4.30 s | .267324 m | 2.764252 | 已开始下沉，预测 Support 2.666313 |
| 低姿态恢复 | A328 | 6.14 s | .141877 m | 7.248768 | command-nonprogress 事件窗口 |
| 低姿态恢复 | B625 | 12.08 s | .131105 m | 8.096923 | first-material-load 前窗口 |

这些均来自 W49 原件，而非 W48 A213/B176/A364/B328；resolved 内都有 policy/accepted/P0/context/reference 五对象身份。精确对象路径和已有外部状态覆盖边界在 analysis.json。A251@4.60/B271@5.00 虽在首非足接触之前，但高度已到 .168343/.198047 m，不能再称尚未下降的初始事件；若改选，应明示目标是晚期干预。

四问题各一次只到首 valid backward 的 solve，各 9 个新候选；必要时每问题只补一个同路径 baseline。若使用保存的同 epoch r35_reference，R35Freeze 不新增 rollout，正常 nominal10+candidate9=19；必要 baseline 后每事件最多20，即四例≤80 full rollouts，其中36为新正候选。Iteration 内 r32_nominal 是拷贝，不额外生成 nominal rollout。transition/risk FD 与零物理 raw-center FD 分列；不能把80当作全部积分成本。没有首有效方向就少做，不凑满。

接口部署建议两条40ms，A/B各选最早 eligible 新策略：起始直接装该 winner，20ms 时只做一次正常重规划，继续到40ms。共≤400 predictor 子步、≤2额外 solve。每次新epoch一般 R35Freeze1+nominal10+candidate9=20 full rollouts，两次≤40；所以四fixture+两次跨epoch部署 solve 的已声明 full-rollout 合计上界≤120，FD另计。

这一部署针对 reference 真正刷新和下一次完整对象是否能继续使用。它仍是 native predictor 的共同声明 context/因果材料 proxy 条件，不冒称完整 plant 历史回放，也不是任务资格。

实现必须关闭一次性 query-only 安装模式后进入真正 online planner；W49 QUERY_ONLY 环境会每次直接 return。也不能把原保存 `MJPC_R35_REFERENCE` 文件带入20ms新 epoch，其 header 时间不同会报错。新epoch正常生成自己的 r35 输入 reference，保留相同 constant 分支；若用新 Agent 进程切换，明确传递完整 policy 和当前状态/gait/context，不宣称实现了任意全状态恢复。

有限评估同时报告 native J、J_support/J_position、风险、预测最小高度/姿态、真正受载脚与非足支撑、同一20/40ms动作变化。至少一个代表事件出现原 strict 可接纳且有动作/支撑实质变化、部署没有未解决实现错误后，可按原计划自动进入同版本原 A/B 各一次 E30；不要求有限问题提前全程成功。若全部仍无改善，交付这个确定机制的负结果，不扩成多权重/多beta网格。

## 8. build/runtime 与失败分类

W49 finite binary 为 `08fb4d…`，部署与原任务正式 binary 为 `8637e3…`。`build.json.finite_build_difference` 明确解释 service ABI relink 与 imported reference cost metadata 修复；optimizer/candidate/action implementation 声明未变。应保留两者身份，不说整个 finite 原样运行在正式 executable 上。最终 build 全部 native translation units 与 service archive 一起重编译，必要，因为 iLQGPlanner/policy 结构布局发生变化。

`s1_go2_r11_runtime.py` 接入 r37、机制 K、r34 strict、constant 与 W49 recorder，并阻止离线 INPUT_POLICY/CONTEXT/REFERENCE/QUERY_ONLY/FINITE_GUIDED 等标志进入正式原评价。PolicyStream 和 material exposure 的 build 分支也接入 r37。W49Recorder 保存新的完整 delta/context；其新增 integration/外部数值数组是更好的事后证据，但仍显式排除 native optimizer cache、actor tensors/RNG 等任意对象恢复，不等于 full resume。

本轮 2 次 query-only 技术失败及 1 次 deployment Init 停滞发生在物理前，已修复；它们应与 2 次原任务 E30 不合格分开计数。后者均是自然完成30s后资格不满足，不是后台墙钟截止截断。build/phase 内沿袭的历史 qualification/selection 字段有过时文字，不能用它们覆盖最终原结果。

下一轮应继续 10+10+2 CPU 并行（约3核系统/主控），build-j6 与两条重型 solve 分时；GPU 只承担独立有新信息的小工作或 proposal 生成，最终原 native 评分裁决。不能以占满96GB显存为任务质量指标，也不再为 GPU 增开一轮无关等价门禁。固定后台 deadline 保持 null；GitHub核验后不做用户电脑额外归档。
