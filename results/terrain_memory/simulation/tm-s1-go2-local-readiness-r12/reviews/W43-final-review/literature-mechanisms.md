# W43 文献复核：先恢复可解释的起步，再扩大训练

检索日期：2026-11-10。仓库阅读基点：`6a94c94c8e21e3b50fcca40ab4363165dee2987e`。本文件是针对 W43 后续机制的文献分析，不是新增实验结果，也不声称逐项复算了全部 Release 原件。

项目依据为 [W43 计划](../../../../../../docs/S1_W43_STABILITY_PLAN.md)、[S5-r32 patch](../../../../../../experiments/terrain_memory/s1-go2-r11-S5-r32.patch)、[运行入口](../../../../../../scripts/s1_go2_r11_runtime.py)和本次 [CPU 原件复核摘要](cpu-summary.json)。后者明确列出实际读取范围：10 个训练回合、2 个末次 E30、12 份 trajectory/planner-call 原件及新 transition/rollout 数组；`aggregate` 给出 `all_bad_reset_calls=0`、`all_liftoffs=0`、`all_touchdowns=0`、`all_material_events=0`。这些数据支持“数值状态重置已生效，但没有建立接触切换与材料任务能力”，不支持“继续降低 λ 或增加同类 PPO 样本就能起步”。完整发布范围另见 [publication-audit.json](publication-audit.json)。

## 1. 用最新实证重新排序文献建议

**第一优先是目标、曲率与正则化尺度一致，以及原生改善的实际含义。** 最新 [solver 原件复核](solver-report.md)及其 [结构化统计](solver-summary.json)表明：全部 34 个已发布完整 accepted 计划的原生预测均后退，脚部几何均未离地；最初深入检查的六个运动计划中，预测 `vx` 全负。A 的 30 秒计划中，`MaterialSupportHeight` 占成本约 66.45%，该次成本改善约 97.6% 来自这一项，`Gait` 改善仅约 `1.25e-9`。这一名称带有 Material 的高度项在刚地也启用，不能用它的下降证明材料已经加载。

深入比较的运动阶段矩阵 `H` 几乎由 `λI` 构成，λ=1 的贡献约 99.86%–99.94%；Stand 与 moving 的 Effort 直接曲率因单位/归一化差约 1736 倍。因此“λ 已从 1e6 重置为 1”不能推出“正则化已处于合适尺度”。两条 E30 outbound 分别有 1374/1372 次 accepted，但已选 alpha 的 surprise 全部小于 0.5，中位数约 0.236；这些 outbound accepted 的 `λ_after` 只有 1 或 2，没有 0.5。**单纯重复相同在线迭代并不会因此自动获得更小 λ。** 这些结果支持先审查残差单位、原生/代理曲率及可信改善；既不支持盲目继续降低 λ，也不支持把当前优化的高度改善当作运动进展。[solver 复核](solver-report.md)

**第二优先是以有结构的模式候选，针对性地区分搜索不到动作与目标不愿选择动作。** IDTO 和 ROM-guided CI-MPC 解释了接触局部优化为何需要跨接触模式的方向或参考。下一机制可以提出卸载、摆足、落足候选，再由本项目原生模型与原成本裁决。若推进候选在同一目标下更优却没有被找到，才支持扩大探索；若推进候选总被姿态、动作或其他代价压过，则须定位目标权衡。更多采样本身无法保证消除后一问题。

**候选赢家保留是必要的完整性修正，但目前证据不支持把它列为不走的主要瓶颈。** 最终两条 E30 outbound 的 27/29 次 incumbent 复用中，只有 3/4 次存在已算出的更好候选，最大 `J` 差分别约 `3.79e-10/9.54e-10`，量级极小。GPC 和 Reference-Free 的最佳真实候选语义仍应借鉴，但不能用这一缺口解释全部驻留。

**预测与执行不一致也不再是当前最优先的解释。** 34 个已发布 full plan 中，32 个可做 20 毫秒对照；最大位置误差约 `2.3e-8 m`，关节位置误差约 `1.52e-7 rad`。实际执行几乎准确跟随“本来就后退、没有足离地”的预测。这个结论限于已对照的原件与时间段，不能推广成接触切换、材料加载阶段也必然准确。[solver 复核及范围](solver-report.md)

以上量化引用同轮 solver 审查，不声称本文又独立复算了一遍全部原件。它们把优先级收紧为：尺度与目标 → 原生可信下降 → 模式候选对照；候选身份及反馈契约作为完整性要求保留。无需同时更换物理模型、PPO 权限、步态参考、预测时域和 GPU 后端。

## 2. 八篇近期主源与一篇经典方法源

下表有六项相对于 W41/W42 文献清单新增的方向、两项针对当前失败形态重新深入的方向，以及一篇直接解释局部曲率和正则化的经典论文。DIAL 和 Whole-Body MPPI 的既有阅读不再重复展开。发表信息以论文、出版社和作者页面为准；“未核到完整代码”只描述本次读取范围。

| 论文及状态 | 本次读取与代码 | 对当前问题最有用的机制 | 必须保留的限制 |
|---|---|---|---|
| **Generative Models From and For Sampling-Based MPC: A Bootstrapped Approach For Adaptive Contact-Rich Manipulation**，Brudermüller 等；RA-L 11(3), 2026，DOI `10.1109/LRA.2026.3655193`。新增。 | [原文 §IV-B/C/D、Algorithm 1](https://arxiv.org/html/2510.14643v1)；[Oxford 发表记录](https://ora.ox.ac.uk/objects/uuid%3A1a21941e-8b2e-46f9-a75e-4ead1e3228a1)；[作者页](https://www.robo.guru/research.html)。作者页本次仍标 website/code coming soon。 | 生成模型提出整段控制序列，和在线 CEM 候选合并评分；执行真实最低成本候选，而非未评估的精英均值；下一次均值从最佳候选平移。 | Spot 实验已有冻结的低层行走策略，高层规划躯干/机械臂；不是从静止接触自行学出步态。学习 proposal 依赖已有有效任务数据，并有分布外限制。W43 驻留数据不能替代运动示范。 |
| **Reference-Free Sampling-Based Model Predictive Control**，Schramm 等；ICRA 2026，DOI `10.1109/ICRA57385.2026.11696779`。重新深入最新 v3。 | [原文 §III、Algorithm 1](https://arxiv.org/html/2511.19204v3)；[IEEE 发表页](https://ieeexplore.ieee.org/abstract/document/11696779)；[作者页面](https://pierfabre.github.io/)。本次未核到完整算法代码仓库。 | 对关节位置和速度节点联合采样，以 Hermite 插值获得时间结构；MPPI 名义序列与实际最优候选分别保存，执行后者。 | 使用 Simple 仿真器；Go2 硬件有外部运动捕捉。其输出包含期望位置与速度，本项目当前 PD 目标契约不同。可借用结构和候选保留，不能声称直接复制便等价，更不能转移其安全表述。 |
| **Hybrid Feedback Sampling for Sample-Efficient Model Predictive Control**，Pan 等；arXiv `2608.19443v1`，2026-08-19；[作者页](https://johnzhang3.github.io/publications/)列 Under review。新增。 | [完整论文 §IV-C/D/E、Algorithm 2](https://arxiv.org/pdf/2608.19443.pdf)；本次论文/作者页未核到完整代码链接。 | 混合 iLQR 或学习反馈内的局部采样与较宽的全局采样；反馈减少发散，全局部分减少只停在反馈局部模式的风险。 | 线性二次问题的协方差结论不能直接推广成 Go2/TM02 接触保证。论文的均值更新也不免除本项目对最终策略再次原生评分的责任。当前驻留 `K` 不自动成为有效运动稳定器。 |
| **Reduced-Order Model Guided Contact-Implicit Model Predictive Control for Humanoid Locomotion**，Esteban 等；ICRA 2025，DOI `10.1109/ICRA55743.2025.11127605`。重新深入 W41 已列方向。 | [原文 §III-B/D](https://arxiv.org/html/2502.15630v1)；[作者项目页](https://rom-cimpc.github.io/)。未核到完整 ROM-CIMPC 发布仓库，IDTO 代码不能冒称是同一系统。 | HLIP 生成名义步态，经 IK 形成全状态参考，再由 contact-implicit 优化适应接触；站立和行走参考混合，帮助优化器获得有结构的动作方向。 | 它不仅改初值，还改变参考跟踪目标。论文主要为 Achilles 人形仿真；双足 HLIP 不能直接移植 Go2。参考可以暂时不满足动力学，需优化修正，不能把 IK 轨迹当物理可行证据。 |
| **Inverse Dynamics Trajectory Optimization for Contact-Implicit Model Predictive Control**，Kurtz 等；IJRR，2025-05-30 在线发表，DOI `10.1177/02783649251344635`。新增深入。 | [完整原文 §7.1.2、§8](https://arxiv.org/html/2309.01813v3)；[作者项目](https://idto.github.io/)；[Toyota Research Institute 代码](https://github.com/ToyotaResearchInstitute/idto)。 | 直接优化构型轨迹，以逆动力学和柔顺接触建立稀疏最小二乘问题；原文展示前向小力扰动无法抬起静止物体时，构型方向更容易表达接触切换。 | 仍是局部优化；未收敛或罚约束残差可产生非真实的虚拟作用，接触平滑也会改变行为。不能把 Drake 接触模型/逆动力学改写当作当前原生 MuJoCo 控制器的局部修补。 |
| **Hybrid iLQR Model Predictive Control for Contact Implicit Stabilization on Legged Robots**，Kong 等；T-RO 39(6):4712–4727，2023，DOI `10.1109/TRO.2023.3308773`。新增。 | [全文，重点模式错配、reference extension 与并行 rollout](https://arxiv.org/pdf/2207.04591)；[IEEE](https://ieeexplore.ieee.org/document/10252162/)；[作者示例代码](https://github.com/Kongx231/ilqr)。后者不能当作完整 A1 硬件流水线。 | 混合动力学的 saltation matrix、模式相容的参考延拓和反馈，处理实际接触时序与参考不同的问题；并行候选中保留参考轨迹。 | 本项目是固定步长 MuJoCo 接触，不具有现成的显式 guard/reset 模型；saltation 不能直接贴进现有 soft FD。它支持检查反馈/参考模式一致性，不支持无条件删除反馈或照搬硬件成功率。 |
| **The Trajectory Bundle Method: Unifying Sequential-Convex Programming and Sampling-Based Trajectory Optimization**，Tracy 等；arXiv `2509.26575`，2025；[作者页](https://johnzhang3.github.io/publications/)列 ICRA 2026。新增。 | [原文 §III/IV、§VII limitations](https://arxiv.org/html/2509.26575v1)。本次原文/作者页未核到完整代码链接。 | 采样动力学、代价和约束，以局部线性插值代替导数构造顺序凸问题；涵盖单次射击采样和多次射击约束优化，可利用并行原生计算。 | 文中实验是双积分、四旋翼、cartpole、赛车，不是 Go2 接触。采样分布选择、可靠最优性度量和鲁棒收敛率仍有限制；插值满足约束不等于真实模型执行满足约束。适合作为中期方法支线。 |
| **Primal-Dual iLQR for GPU-Accelerated Learning and Control in Legged Robots**，Amatucci 等；RA-L 11(1), 2026，在线发表 2025-11-14，DOI `10.1109/LRA.2025.3632610`。新增。 | [作者完整论文 §II-F](https://iit-dlslab.github.io/papers/amatucci26ral.pdf)；[作者 MPX 代码](https://github.com/iit-DLSLab/mpx)。 | 多次射击 primal-dual KKT，用 associative scan 实现时间维并行；JAX 支持环境批量化。论文在线控制也采用每周期一次迭代及平移初值。 | 单次在线迭代本身不是失败证据；需看实际方向、下降和闭环。其不等式处理包括松弛障碍，不等同本项目硬约束。当前瓶颈若仍在动力学与 rollout，单独加速 backward 不能带来同等总收益。 |
| **Control-Limited Differential Dynamic Programming**，Tassa、Mansard、Todorov；ICRA 2014。经典机制源。 | [作者原文 §II-B/D](https://roboti.us/lab/papers/TassaICRA14.pdf)。本次用于核对局部二次模型、正则化和 line search，不要求替换现有实现。 | `k=-Q_uu^{-1}Q_u` 表明方向大小依赖曲率；最小二乘残差的 Jacobian 决定 Gauss–Newton 曲率，正则化及 alpha 按 LM 启发式调整。 | 原文未给本项目 λ 数值或残差单位的保证。其控制约束和仿真结论不证明本项目的 trunk/PD/材料稳定性。 |

## 3. 直接映射到当前求解器

### 3.1 先把正则化与残差尺度放到同一坐标下解释

考虑固定局部二次模型的无约束方向，或固定活动集下的自由方向：

\[
k=-(H_0+\lambda I)^{-1}g.
\]

若仅把整个目标乘上正数 `c`，同一局部模型变为 `g'=cg, H_0'=cH_0`；但仍固定同一个 λ，则

\[
k'=-(cH_0+\lambda I)^{-1}cg
   =-(H_0+\lambda I/c)^{-1}g.
\]

这说明绝对数值“λ=1”没有脱离成本尺度的强弱含义。这个局部代数例子不是本项目完整约束 Riccati 的替代复算，更不能直接从已保存的 H 减掉 λ 后求逆来宣布新控制器有效。[Tassa 等原文](https://roboti.us/lab/papers/TassaICRA14.pdf)

工程上可参照 [Ceres 官方 LM 说明](https://ceres-solver.googlesource.com/ceres-solver/%2B/master/docs/source/nnls_solving.rst)：它将局部残差项与 `||Dδ||²` 的正则项放在同一最小二乘问题中，D 通常取 `JᵀJ` 对角的平方根，并用真实与预测改善的吻合程度控制信赖程度。这个来源只支持“尺度化度量与真实改善需要共同考虑”的原则，**不证明应直接套用 Ceres，也不证明把 D 换成对角曲率就能让本项目行走**。

本项目应据此追踪三种不同尺度：各残差的单位与归一化、原生/代理二次模型的曲率、用于约束和策略投影的度量。Stand/moving 的 Effort 曲率差需要解释清楚是任务设计还是坐标变化，不能只凭数值不同宣布错误；但相同 λ 跨这些尺度使用时，效果确实不等价。`MaterialSupportHeight` 的改善占比很高也不自动证明必须删掉该项，应比较其是否在当前阶段压过推进/步态目标，以及有用推进候选在原生排序中受到什么惩罚。[solver 原件复核](solver-report.md)

当前 surprise 偏低意味着代理预测的改善不能完整兑现，不能用“accepted 次数很多”忽略它，也不能靠强制降低 λ 来跳过它。后续数值机制要同时保留未正则曲率、实际使用的度量、原生/预测改善和约束结果；是否改残差归一化、正则尺度、参考，必须根据对应证据选择并分开声明。

### 3.2 保留“评估的策略就是执行的策略”这一不变量

记完整候选为

\[
p=(U,X,K,\text{reference},H,\text{projection contract},\text{problem identity}).
\]

这不是建议新增所有同名字段，而是要求现有对象的归属保持一致。对当前物理状态和同一个目标版本，将有效 incumbent、best nominal、局部搜索候选以及未来采样候选纳入同一集合。最终选择必须遵守现有 trunk 容差和原生 `(v,J)` 比较器：

\[
p_* = \operatorname*{argmin}^{\text{现有比较规则}}_{p\in\mathcal E_t}
       \bigl(v_{\rm native}(p),J_{\rm native}(p)\bigr).
\]

其中 `E_t` 只包含本次真实 rollout 有限、身份有效且符合现有部署规则的对象。若局部步未优于 best nominal，不应推导为“只能退回一个更差 incumbent”；也不能把只在候选 rollout 中使用的反馈倍率换掉后保留原成本标签。当前 r32 的 `NominalTrajectory/BestRollout`、`nominal_winner`、`incumbent_valid`、接受后的 `winner` 和 `reused_current_native_scored_incumbent` 分支应围绕这一语义检查。[r32 实现](../../../../../../experiments/terrain_memory/s1-go2-r11-S5-r32.patch)

GPC/Reference-Free 提供的是设计依据；本轮原件已把这一缺口的量级限定为上述极小成本差，不能提升为主要运动根因。保证这一不变量是为了避免后续更丰富候选被错误丢弃，成本改善仍不能替代实际前进、抬足和材料加载。

### 3.3 用同一原生目标区分三种停滞

下表是本项目提出的判别方式，不是任何一篇论文已经替本项目完成的实验。

| 同状态、同模型、同成本下的观察 | 支持的解释 | 后续动作 |
|---|---|---|
| 有真实卸载/摆足/推进的候选满足约束且优于驻留，现有搜索未选中 | 候选探索或候选保存不足 | 使用低维、时间相关、接触结构明确的 proposal；局部精修后仍与原候选比较，不把本轮极小赢家差当成已找到推进候选 |
| 有推进候选，但原生总成本始终更差 | 当前目标/参考权衡可能偏好驻留；也可能候选动作本身质量差 | 保存各项成本与约束变化，区分动作代价、姿态/高度、速度/步态等贡献；据此提出单独的参考或目标版本，不能仅增加样本数 |
| 原生预测推进，实际执行仍停留或方向不同 | 候选与部署契约、状态快照、反馈、PD/饱和/保持过程可能不一致 | 先解决模型与执行对齐，不能把预测进展计入任务成功 |

第三行是未来遇到不同证据时的判别分支。本轮 32 个短时对照高度一致，且 34 个已保存完整计划原生预测本来就不前进，因此当前优先检查前两行及尺度问题，而非假设执行接线出错。非赢家完整轨迹尚未发布，有限已选轨迹不能证明目标的全局最优偏好驻留。

结构化候选可以包含少量卸载—摆动—落足模式与时间相关关节节点，而不是对 175 个区间独立加入白噪声。候选要通过当前动作权限、投影、PD 及十子步物理进入原生 rollout；脚端几何高度上升不等于已经卸载，接触计数/法向力与位移应共同解释。该设计借鉴 ROM-guided 的动作结构和 IDTO 的模式方向，**不要求先把双足 HLIP、外部低层步态控制器或另一接触模型接入主线**。

必须把两种实验分开标记：

1. **只改候选初值或 proposal**：保留原成本、参考和物理，可直接比较候选排序。
2. **引入新的全状态/摆足参考**：目标版本已改变，要声明配置与代价分项，不能把跨目标的 `J` 大小当作改进证据。

当前 0.35 秒预测时域只覆盖 2 Hz 步态的 0.7 个周期，是需要结合保存轨迹考虑的限制；但单凭这一比例不能认定时域就是根因。不能在同一轮同时增时域、改参考、提权重并扩动作权限后仍声称验证了单一机制。[W43 冻结约定](../../../../../../docs/S1_W43_STABILITY_PLAN.md)

### 3.4 混合反馈候选与局部精修，不把噪声当运动能力

FS-MPC 的 Eq. (6) 给出局部反馈分布与全局分布的混合。映射到本项目时，可以先把它当 proposal 设计，而不复制其理论保证：

\[
q(p)=\rho q_{\rm feedback}(p)+(1-\rho)q_{\rm structured}(p).
\]

这里把第二部分换成结构化动作是本项目的建议，区别于原论文的各向同性全局噪声。所有候选仍需原生裁决；采样温度也不是 r32 的 Levenberg 阻尼 `λ`，两者不能混用命名或调参解释。HiLQR 的参考延拓进一步提示：如果候选进入不同接触模式，继承的反馈参考应有明确语义；否则局部反馈可能把摆足动作拉回驻留轨迹。

顺序应是：提出候选 → 按实际执行契约原生 rollout → 从真实候选选择初值 → 使用当前 soft derivative/native forward 局部精修 → 将精修结果与原候选一起重新选择。不能因为做了精修就丢掉较好的初始候选；不能在实际 rollout 尚无卸载或推进时称“跳出了局部最优”。这不要求给现有活动 GPU 作业热换求解器。

## 4. 当前不适合优先做的事情

- **继续只扩大标量 cap 的 PPO 样本。** 当前 PPO 权限是 2 Hz 标量速度 cap，不能直接选择抬哪只脚、摆足高度或接触时序。没有运动响应时，训练增长不能证明低层具备动作可控性。GPC 的高层规划成功依赖已有低层能力，恰好提醒这一前提。[项目动作契约](../../../../../../docs/S1_W43_STABILITY_PLAN.md)
- **直接用驻留样本训练生成器。** GPC 的提议学习需要覆盖有用控制序列；先获得真实有用运动，再考虑用学习提议减少搜索成本。旧历史可用于诊断，但不能把不同控制器、任务与成本的动作直接混成等价标签。
- **为了增加 GPU 占用整体重写求解器。** TBM/MPX 是可行的中期方向，但其动力学、约束与当前 r31/r32 原生语义不相同。当前活动 GPU 阶段应按冻结契约继续。后续扩大 GPU 工作应优先服务真实节点、导数、候选轨迹和完整策略排名；测有效吞吐与闭环延迟，不能以显存占用代替结果。
- **把原生前向和平滑导数差异视作已经解决。** 新论文说明跨接触方向的表达很关键，并不能证明当前代理梯度正确。需结合现有方向预测/原生下降、接触模式和部署轨迹，解释每次有效或无效更新；同时避免退回把代理物理当任务真值的旧错误。

## 5. 对 W44 的文献决策

结合最新原件，优先顺序是：**解释并处理残差/曲率/正则尺度与原生可信改善 → 用接触结构候选在同目标下区分搜索不足和目标偏好 → 据成本及接触证据决定是否改参考或进一步精修**。已评估赢家保留是必要完整性，当前极小缺口不能解释全部不走；更多采样是有针对性的候选机制，不能替代尺度和目标分析。八篇近期论文、一篇经典方法论文及官方数值实现说明提供可检验的机制与边界，没有一个来源替本项目证明了 TM02 稳定往返。

新一轮仍可保留双 CPU 流与既有 GPU 支线，但并行应避免同时引入无法区分的控制变化；现场资源分配、父点、真实样本目标和发布范围由 W44 主计划明确。后台不设置固定墙钟截止；数值求解有限迭代、物理评价时长及明确技术失败保持各自语义。GitHub 交付完成后无需另做用户电脑归档。
