# W41 控制器、真实接触与代价复核

基线：`9cc42d5a9189a2bfd7e1e6cd99b9300045a16724`，2026-11-08。范围为已经结束的两条 CPU W41；不操作或评价未结束的 GPU。本文是已交付记录的离线审查，没有运行新物理、重放旧 12 例 QP、改运行源码或部署新控制器。

**结论：r30 的约束 Riccati 确实已经实现，保存的 K/H 满足其声明的局部方程。两条末次评价在材料前已经严重欠跟踪，但该段实时投影没有改变任何部署动作，A 停滞 plan 的风险项也为零。更直接的问题是：从完全相同的真实初始状态、近乎相同的第一保持目标出发，预测器的原生刚地支撑力接近零，实际支撑力却在百牛量级；首个 2 ms 内已经产生可观速度偏差，随后较大的 K 把真实偏差放大成弧度量级的目标修正。**

这支持下一轮优先纠正 forward 的刚地接触保真度，同时显式处理平滑导数的用途。它不证明这一改动足以稳定穿越，也不能把旧 r17 原生阻抗开关直接改名再跑。

数值出处：[可复算脚本](controller-input-audit.py)、[完整小型结果](controller-input-audit.json)。脚本使用九份完整 `accepted-reference-gain-H` NPZ、两个 E30 的 trajectory/physical/实际执行动作日志、实际 XML 与 resolved 配置；输入字节数与 SHA256 同列在 JSON。原件来源与读取范围见 [raw-plan-read-coverage.json](raw-plan-read-coverage.json)、[raw-read-coverage.json](raw-read-coverage.json)、[raw-metadata-read-coverage.json](raw-metadata-read-coverage.json)。

## 1. r30 改了什么，什么没有改变

实际产物是 `S5-r30` / `W41_MATCHED_BOUND_RICCATI_JOINT`，基于 r29-qp2；不能再把 r30 写成待实施。重建 r24→r26→r27→r28→r29→qp1→qp2→r30 源码后，quadruped、agent、trajectory、model_derivatives、ilqg planner/policy、gRPC 文件与 E30 resolved 中的 author source SHA256 相符。[r30 规格与补丁](../../../../../../experiments/terrain_memory/s1-go2-r11-S5-r30.json)、[实现](../../../../../../experiments/terrain_memory/s1-go2-r11-S5-r30.patch)

在名义节点，r30 用成功 backward 得到的同一个 `H=Quu_reg` 解约束增量问题：

\[
\min_k \tfrac12 k^\top Hk+Q_u^\top k,
\quad u+k\in [u_{min},u_{max}]\cap
\left[q+\frac{\tau_{min}+5\dot q}{60},
q+\frac{\tau_{max}+5\dot q}{60}\right].
\]

它向 H 度量投影传入未预裁剪的绝对中心 `u-H^-1 Qu`。活动 PD 边界的反馈行是 `e_q+(5/60)e_qdot`，固定目标边界/恢复行是零；自由行解 `H_FF K_F=-Qux_reg_F-H_FA K_A`。随后 Vx、Vxx 与预测改善使用同一 k/K，保留完整交叉项。并非只在旧 K 后再夹一下。

九份真实 plan 的非终端节点复算：自由行 `H K+Qux` 最大缩放残差 **2.733e-15**，活动行与上述解析行最大差 **0**，所有保存的 H 为正定。这里核对的是发布记录及当前方程，不是重新跑 QP，也不是声称全局控制已稳定。

前向仍使用 r29 原动作语义：原目标范围裁剪 `u_ref+alpha*k+K delta_x`，再按当前状态的动态盒做 H 度量投影；candidate 与部署共用该映射，部署反馈倍率为 1。保存的 reference 保留生成策略的参考轨迹，没有被 accepted trajectory 偷换。这个局部 Riccati 与非线性前向在活跃集改变、目标预裁剪和较大状态误差下不可能自动成为全局精确模型。

原始 plant 仍是 PD60/5、原关节目标范围、原力矩限幅、0.2 ms 物理步长、每 10 子步保持一个目标；r30 的瞬时无饱和盒不承诺后续九个子步的未裁剪 PD 请求永不超限。实际 actuator clamp 仍有效。没有在本轮证据中发现漏 clamp、旧 K 冒充新 K、参考时间错接或已经重现的 qp2 技术故障。

## 2. 已有 same-input 动力学检查能证明到哪一步

W41 发布的 [same-input 结果](../../w41-dynamics/result.json) 使用 32 个 W40 来源状态，核对目标裁剪、真实 actuator map、局部与组合十子步有限差分。它记录的力矩映射最大误差为约 2.84e-14；若干 whole-hold 有限扰动偏差随 epsilon 变小而减小。[有限差分诊断](../../w41-dynamics/hold-fd-diagnosis.json)

这些结果支持“同一预测器内的接线与组合公式未发现确定错误”，**不能推出这个预测器等于 physical plant**。检查本身已经区分 prediction context 与物理隐藏材料状态。尤其当前 `mjd_transitionFD` 使用 1e-6 有限扰动，一些敏感状态在这个尺度仍有明显近似误差；既不能把它说成精确真导数，也不应将随 epsilon 收敛的差异直接判为代码接线错误。

本次进一步使用 W41 原始前向记录，检查的是预测器与实际运行之间的零阶响应，不是重复上述导数检查。

## 3. 参考与步态：没有看到错误 yaw，但移动意图没有被持续实现

实际配置在 2.002 s 从 Stand 切换到 Trot，cadence=2 Hz、duty ratio=0.45、amplitude=0.005 m；两个 E30 均保持 outbound、heading=0，没有错误进入返程或临时换反方向。`Walk speed=0` 是未启用原自动 walk 命令的配置值，实际前进来自 r26 reference packet，不能据此断言“没有下发速度”。

r26 使用同一因果 packet 和解析时间函数给 C++ 与 Python 提供位置/heading 参考；前向加速上限 0.5 m/s²，约 0.25 s 的 lead 对 0.2 m/s 指令产生约 0.05 m 的领先参考。packet 的 anchor velocity 是有界命令速度，不是错误地冒充实测速度。当前 moving 模式没有另一个激活的 body velocity residual；`StandLinearVelocity` 等在非 stand-equilibrium 条件下为零。这是既有目标定义，不是本次新发现的接线错误。[参考实现](../../../../../../scripts/s1_go2_r26_reference.py)、[r26 补丁](../../../../../../experiments/terrain_memory/s1-go2-r11-S5-r26.patch)

| 两个 E30 的 9–11 s，尚未材料加载 | A | B |
|---|---:|---:|
| 平均前进指令，m/s | 0.22182 | 0.20810 |
| 净推进速度，m/s | 0.01447 | 0.00980 |
| 四足同时接触比例 | 94.8% | 88.7% |
| 最前足 x 最大值，m | 0.72852 | 0.55519 |
| 最大绝对 yaw，rad | 0.00847 | 0.00650 |
| 真正执行的 feedback query 数 | 1000 | 1000 |
| projection_changed / 活跃边界数 | 0 / 0 | 0 / 0 |
| QP 未收敛数 | 0 | 0 |

因此，**早期停滞不能归因为实时投影持续吞掉推进指令，也不能归因为错误 yaw**。Trot 的高度代价是软参考，命令为 Trot 不意味着实际必须按两足交替离地；四足接触比例已经说明实际步态和期望移动能力之间存在差距，但仅凭接触比例不能辨别局部最小、预测失配与有限视野各自贡献。

A 的最前足后来接近风险 gate 起点，最初看起来像边界障碍；B 在远离 gate 的 x≈0.555 m 已经出现同类停滞。进一步的完整成本分解排除了“风险墙主导早期停滞”的简单解释。

## 4. 精确代价分解：停滞期与失稳前是不同问题

按实际 XML 的 norm 类型/参数、实际 applied weights 和 176 节点残差重算。使用原 norm0 平方、norm2 平滑 L2、norm6 分量平滑绝对值，不能只比较权重数字。九份 J 与保存 accepted cost 的最大误差为 **1.421e-14**。[固定版本 norm 源码](https://github.com/johnzhang3/mujoco_mpc/blob/e00c47a5adb9856af2e0f24231bb3a60d5be23c4/mjpc/norm.cc)

A plan590、11.38 s：J=0.0146093031，预测 0.35 s 后 base x 减少 **0.011019 m**，整个 horizon 的 active rows 为零。

| 项 | 平均成本 | 占 J |
|---|---:|---:|
| Position | 0.00779428 | 53.35% |
| Posture | 0.00205981 | 14.10% |
| Effort | 0.00152224 | 10.42% |
| Gait | 0.00108197 | 7.41% |
| Balance | 0.00077689 | 5.32% |
| Height | 0.00069264 | 4.74% |
| MaterialSupportHeight | 0.00066032 | 4.52% |
| ImpactMargin / MaterialDomainMargin / LandingVelocity | 0 / 0 / 0 | 0 |

TrunkClearance 约为浮点平滑正部尾项量级。`MaterialSupportHeight` 是程序中的残差名称，并不表示实际已经进入材料；它也参与当前的高度参考。这份 plan 说明“正向参考存在，但优化所接受的短期轨迹不持续推进”，不能单凭一个 accepted plan 宣称已经数学证明了站立局部极小，也不能仅凭 Position 最大就认为移动奖励充足。

之后的代价迅速变化：

| Plan / 时刻 | J | 最有区分度的组成 |
|---|---:|---|
| A634 / 12.26 s | 0.045245 | SupportHeight 61.30%，Position 13.64%，Impact 4.26% |
| A635 / 12.28 s | 0.454587 | Impact 89.52%；最大 stage 在 +4 ms |
| B712 / 13.82 s | 0.010653 | Position 46.60%，Effort 17.40%，SupportHeight 12.36%，Gait 7.13% |
| B713 / 13.84 s | 0.012630 | Position 37.65%，Effort 22.47%，Impact 16.57% |
| B737 / 14.32 s | 65.552326 | Impact 99.973%；最大 stage 在 +2 ms |

这支持把材料前欠跟踪、接触预测偏差、临近加载时的高曲率风险与后续跌倒分开。按接触时序或距地高度关掉风险项会改变控制器的风险目标；不改变原物理条件，但目前没有证据表明它是首选修复。单纯加长 horizon 或加大 gait/velocity 代价也仍属待检验假设。

## 5. 首个子步的刚地支撑预测与实际差异

`R29RecordedStep` 在 `mj_step` 前保存 qpos/qvel，之后保存该次力求值的 contact force、actuator force 与 post-state；physical 同时保存 force-evaluation clock 与 post-integration clock。本次用同一 force-evaluation time 对齐，未把返回后的 qpos 与下一步力错配。所列接触均为水平 floor 平面与球形足端，floor 在 contact geom0，contact-frame normal 对应世界 +z。physical 第 44 列本义是总世界 Fz；表中五个起点非足力和非足接触计数均为零，因而此处才等于足部 Fz。完整记录的十子步列表和 contact pair 在复算 JSON 中。[独立时序、坐标与材料来源交叉核对](contact-crosscheck.md)

以下均在首次实际材料加载前。第一行的预测 pre-state 与实际 trajectory state 逐项完全相同；候选与实际第一保持目标因 RPC float 时间/动作序列化有很小差异，故准确说法是**实际同状态、近乎同目标的原始记录比较**，不是已经新跑了逐位相同输入 replay。

| 事件 | 第一目标最大差，rad | 预测原生地面足力 Fz，N | 实际足端总 Fz，N | 首个 2 ms 后关节速度最大误差，rad/s |
|---|---:|---:|---:|---:|
| A590 / 11.38 s | 2.26e-6 | 0.00935 | 166.050 | 0.34468 |
| A634 / 12.26 s | 3.17e-6 | 0.00832 | 141.175 | 0.71922 |
| A635 / 12.28 s | 5.71e-8 | 0.00716 | 127.541 | 0.43351 |
| B712 / 13.82 s | 5.95e-8 | 0.00719 | 113.940 | 0.38575 |
| B713 / 13.84 s | 5.15e-6 | 0.01340 | 246.877 | 0.43712 |

A635 首个 actuator force 差仅约 **3.43e-6 Nm**，B712 约 **3.57e-6 Nm**。因此这些百牛级接触差异不是另一个大动作先被发下去的结果。A635 的第一 2 ms 中，预测支撑从约 0.007 N 增至 1.65 N，而物理值始终在百牛量级；此时还没有下一保持段的反馈输入混淆。**2 ms 之后实际反馈目标已经分化，不再把其后状态差称为相同输入差。**

源码中有直接解释：`Agent::PlanIteration` 对整个规划使用 `MakeDifferentiable`，把 joint/geom/pair 的 `solimp[0]` 置零，规划完成才恢复；实际地面和足端 XML 的 `solimp[0]=0.99`。因真实刚地接触穿透很浅，在同一实际支撑状态下，平滑预测器给出的支撑很小。原生材料代理仍另有其刚度与因果锚点近似；不能把这里的刚地证据推广成“所有 TM02 误差已经定位完”。[现行模型生成](../../../../../../scripts/s1_go2_r11_model.py)、[既有 r17 原理记录](../../../../../../experiments/terrain_memory/s1-go2-r11-S5-r17.json)

这证明当前 forward 已有明显零阶不匹配，并有明确、可检查的阻抗配置机制。旧 warmstart/coldstart、求解容差等 context 也可能贡献少量差异，尚未以单独开关实验量化每项份额；本文不把百牛差异的每一位数字都归给单一参数。

## 6. 大反馈确实放大了偏差，但“缩 K”不是充分修复

两支首次真正执行的 `max|K delta_x|>1 rad` 分别是 A 12.284 s/plan635 和 B 13.844 s/plan713：1.43836 与 1.17824 rad。用保存的 K、正确插值节点和实际 `state_delta` 复算，最大误差分别只有 1.11e-16 与 2.78e-17；不是日志误读或 K 来源错接。

此时 joint-position 偏差仅约 0.0014 rad，joint-velocity 偏差约 0.56 rad/s；base 平移偏差约 0.0001 m。多个状态块共同贡献大目标修正。B713 整个 horizon 的 backward active rows 仍为零，却已经有 K 最大元素约 1658.54；A635 为 2269.05。B737 更高，约 102359.65，H 条件数约 2.73e6。H 正定与 QP 收敛不意味着反馈对真实模型误差稳健。

这些量支持“前向失配被高反馈放大”的链条；尚不能由两条轨迹证明它是 127 次物理失败的唯一根因。只增大正则化或在部署端缩 K 可以减弱症状，也可能让已经严重欠推进的机器人更接近站立。下一轮优先解决已量化的 forward 接触失配，避免把它再次改写为另一个纯 QP/限幅问题。

## 7. 旧 r17 的失败怎样约束下一步

旧 [r17 补丁](../../../../../../experiments/terrain_memory/s1-go2-r11-S5-r17.patch) 在 `MakeDifferentiable` 后统一恢复 geom/pair d0，保留 joint-limit smoothing；当时 forward 与求导都随之使用恢复后的接触。其 [dev19 E30](../../../tm-s1-go2-material-continuation-r11/dev19-S5-r17-E30.json) 在 15.3628 s 以 invalid_contact 失败，max x=1.03166 m，error=null。它早于 r26 十子步/因果 reference、r28 与本轮 r30，版本与控制链都不同。

旧结果反证的是“单独把原生阻抗恢复就足够稳定”。它没有证明在当前 r30 中保留明显错误的 forward 是必要的，也不能证明所有具有原生 forward 的算法都不可行。因此，不重复 r17 的单一开关；若下一轮采用原生 forward，必须让求导与前向的分工、零阶一致性、误差度量和接受规则成为真正新增且可审计的算法内容。

## 8. 首选 W42 机制：原生接触前向、沿原生轨迹的锚定平滑局部模型

这是建议，尚未实现、构建或运行。其目标是同时保留两点：候选轨迹的刚地支撑使用原生接触；局部搜索仍有显式、受控的平滑导数。它不是原 plant 的精确 Jacobian，也不是对 r30 的稳定性证明。

定义 `f_n` 为保留原生 geom/pair 接触参数的 0.2 ms predictor step；原材料代理、PD/实际限幅、时步、参考、成本保持。它并非完整真 TM02。`f_s` 为只用于求导的现有 d0=0 平滑模型。沿 native 名义子步 `(xbar_j, ubar)`，局部 surrogate 用流形上的零阶锚定：

\[
g_j(\delta x,\delta u)=f_n(\bar x_j,\bar u)
\oplus\left[f_s(\bar x_j\oplus\delta x,\bar u+\delta u)
\ominus f_s(\bar x_j,\bar u)\right].
\]

`oplus/ominus` 使用 MuJoCo 状态切空间，姿态按 quaternion 差分/积分处理，不能直接减四元数。于是 `g_j(0,0)=f_n(xbar_j,ubar)`，局部 A/B 明确来自 **soft-minus-soft**。以 native 的下一状态作为下一个求导点，十子步组成 `P_{j+1}=A_s,j P_j`、`Q_{j+1}=A_s,j Q_j+B_s,j`，P0=I/Q0=0。不能只把起点换成 native，随后仍沿 soft 轨迹滚十步；也不能用 soft-perturbed 减 native-baseline 再除 epsilon，那会把零阶差除成伪导数。

实施边界：

1. 所有 nominal/candidate/accepted forward、风险峰选择、r28 `(v,J)` 排序与实际接受下降均用 native 预测轨迹。阶段残差 C/D 在对应 native 状态和正确风险活跃子步求取，再通过声明的 surrogate prefix 传播；不得沿用旧 soft trajectory 的峰或残差。
2. 使用独立冻结的 derivative model 与各 worker 独立 mjData。不能在多个 worker 共用的 mjModel 上临时切换 solimp。joint-limit smoothing 是否维持原值须单独明确，不把“原生刚地接触”写成所有原 plant 机制均完全一致。
3. r30 的 k/K/V 仍共同使用同一 surrogate A/B 和同一 H，动态盒仍基于当前名义状态。没有部署后再夹 K、改奖励、改 gait/动作维度或风险门控的附加机制。
4. 控制器版本必须明确标注 approximate Jacobian。原 expected-cost ratio 只能视作 surrogate 预测质量指标，不能声称已变成 native cost 的精确 Taylor 比率。实际 native `(v,J)` 的下降是接受依据。

### 可证伪的检查与对照

必要的小型数值检查只针对已经保存的关键输入，直接服务于这个新算法：A635/B712 是第一目标差极小的刚地样本，A590 是停滞样本，另保留一个真实接触/材料过渡样本。比较新 native forward 与记录的旧 soft forward、真实第一保持响应；量化力、状态与目标误差。验证锚定 surrogate 的十子步组合公式，但同时单独报告 **native held-map 的真实方向有限差分与 surrogate 方向的偏差、接触分支、epsilon 和实际 native 成本下降**。代理自己和自己相等不能冒充真实梯度正确。

训练比较遵循主计划：A/B 分别继续自身 W41 的空 tail 父点，各 +2048/8；二者实施同一新机制，以自身 W41 记录作历史参照。这样可以检验两个已有学习谱系是否均改善，但不是随机化同父 r30/r31 对照，不宣称严格因果效应；不额外复制一轮 r30 训练。若需要机制级匹配对照，先使用上述相同 x/u 的离线前向/方向检查，避免再占两条完整采样流。

如果 native first-hold 偏差没有明显缩小，说明实际实现或当前失配解释不成立，应报告而非立即叠约束。如果该偏差缩小但持续移动/合法通行仍不改善，说明 forward 保真是已修复问题却不是充分条件；届时再根据真实接受轨迹判断局部极小、步态目标和有限视野，不能凭计划提前宣布解决。

## 9. 续训与末次策略选择

本轮两支确实各新增 2048/8，到 11665/45、tail=0。A 从自身 W40 ep318 父点、B 从自身 ep314 父点继续；loader 保留 actor、critic、optimizer、RNG、normalization 和各自 lineage。末次评价实际使用各自 `checkpoint-011665.pt`，不是误用早期策略。[恢复入口](../../../../../../scripts/continue_s1_go2_w41.py)、[评价选择实现](../../../../../../scripts/run_s1_go2_r12.py)

`evaluation.json` 内有残留 W40/33/8593 的 checkpoint-selection 说明文字；执行端实际验证的是当前 phase、last_completed_update、当前 job 起始样本/更新阈值、artifact/lineage 与 tail=0。它是需要下一版本改清楚的元数据说明，不能由此反推这次使用了错误 checkpoint，更不应改写已经发布的历史结果。

两流的完整技术/训练分母、自然失败类别、planner 调用状态见 [总审查](README.md) 与 [cpu-progress.json](cpu-progress.json)。当前控制审查未发现一个新的确定性运行源码 bug 足以直接宣布修复完成；发现的是已量化的控制模型缺口，以及针对该缺口有区别、可评估的新算法实验。
