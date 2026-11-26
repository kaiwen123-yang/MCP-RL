# 学习算法 v3：从 PPO 梯度修补走向三个完整的策略改进算法

日期：2026-11-26。状态：研究设计；未训练，未更改活动控制源码。本文新增 12 个直接近邻，不重复计算 v2 中的 SAPO、AHAC、AGPO、MFPG、MFPG-PPO、AC-MPC、TD-MPC2、DreamerV3、FastTD3、Open-DiffLoco，也不把已有物理时间 GAE、raw action/logp、KL 回滚算作创新。

最新项目状态以主报告核验的远端 `2f1d0fcb` 为准：W59 seed5901 已发生真实 PPO fallback actor 更新，imagined 为 0，首轮配对评估正收益主要集中于一对样本；这支持继续真实 PPO 实验，不代表解析模型梯度已可信，H0 仍保留。

**建议主线是 A：把步态事件、持续时间、连续控制和执行映射一起纳入策略更新；B 是针对实机鲁棒性的独立算法方向；C 是允许 actor 与 MPC/WBC 同时从历史数据学习的另一种完整范式。** 三者修改的对象分别是策略梯度及更新几何、风险与稳定性目标、数据使用方式及 Bellman/策略改进算子。它们都不以 NeuralActuator 为前提。

这三个设计是待验证研究假设，不是已经成立的原创性结论。新论文数量不能替代最近邻比较；每个设计必须在一个具体未解决问题上，比最近邻增加可独立检验的机制，而不是把更多现有模块放入同一系统。

## 1. 本轮文献及阅读边界

`learning_sources.json` 保存正式来源、PDF、版本、阅读范围、源码 pin 和实机边界。下面的“方法深读”指已阅读全文方法章节及指定实验/消融，不表示逐页核验全部附录证明。来源编号 B01–B12 仅为本报告索引。

| 来源 | 正式状态/日期 | 本轮阅读 | 对项目最直接的价值 | 实机边界 |
|---|---|---|---|---|
| B01 Simple Policy Optimization，SPO | ICML 2025，7 月 | §3–6、主要比较；官方 loss 源码 | PPO clip 可直接替换为有恢复梯度的二次 ratio 目标；是必须加入的便宜强对照 | 所读实验为 MuJoCo/Atari |
| B02 Absolute Policy Optimization，APO/PAPO | ICML 2024，7 月 | §2–5，理论目标与实用变体 | 优化性能下概率界；风险目标可以改变整个策略改进过程 | 连续控制/Atari/GUARD，未据此认定实机 |
| B03 Total Propagation X，TPX | ICML 2023，7 月 | §3–6、误差/偏差讨论 | 在中间随机节点逐坐标混合 LR/RP，比全参数粗暴加权更可扩展 | DreamerV1/视觉控制模拟 |
| B04 Hybrid/Mixed Gradient Policy Optimization，HPO/MGPO | arXiv v1 2026-05-14；官方 NeurIPS 2026 文稿采用 MGPO 名称 | §2–5、算法伪码、主要消融；源码结构定位 | 连续动作改变未来离散选择概率，必须存在交叉信用项 | 库存与切换 LQR；没有腿足实机验证 |
| B05 Rectified Robust Policy Optimization，RRPO | TMLR，2025-08 | §3–6，包括更正后的误差界 | 稳健约束 RL 不普遍具有强对偶；可采用 primal rectification | 网格与 Robust HalfCheetah；不是实机安全证书 |
| B06 ContractionPPO | arXiv 2026-03-20；Caltech 列 RA-L to appear | §III–V；渲染核对定理；核心官方源码 | PPO+收缩学习已有直接腿足近邻；正文和源码必须分清 | Go1 平台/倒立/扰动试验；理论存在下述公式不一致 |
| B07 CARL | T-RO Early Access，2026-08-24 | §IV–V 方法与条件、实验配置 | 学习度量生成器，冻结度量后用 PPO 学习收缩相关回报 | 有模拟与真实机器人路径跟踪；不等于混合接触步态证书 |
| B08 Dream-MPC | ICML 2026，7 月 | §4–5、推理时间与消融 | 用少量策略候选的梯度细化代替大规模采样；依赖好的 proposal | 24 个模拟任务；4090 时间不能等同机器人板载时间 |
| B09 Reusing Trajectories in Policy Gradients Enables Fast Convergence，RT-PG | ICML 2026，7 月 | §3–7、估计器与偏差条件、实验设置 | 轨迹复用有 target/cross-time bias；有偏修正也可以有理论分析 | CartPole/Swimmer/HalfCheetah，非腿足实机 |
| B10 Bootstrapped Model Predictive Control，BMPC | ICLR 2025 | §4–5、lazy reanalyze、官方 actor/critic 源码 | MPC→策略模仿→更强 MPC 的闭环已经存在；不能再作为独立新意 | DMC/HumanoidBench 等模拟 |
| B11 CrossQ | ICLR 2024 | 问题、方法设计和实验结论，非证明深审 | 低 UTD 的现代离策略对照；算力效率不必靠更多 critic 更新 | 所读论文为连续控制基准；引言引用的实机不算其自身实验 |
| B12 Risk-Aware RL with Bandit-Based Adaptation for Quadrupedal Locomotion | ICRA 2026；所读 arXiv v1 2025-10-16 | §III–VIII 的方法、训练、硬件和限制 | CVaR-PPO + 在线风险选择已有 Go2 近邻；过保守策略会不前进 | Go2 平地/斜坡/隐藏软障碍；bandit 目标不直接保证安全 |

### 1.1 新文献改变了哪些判断

**PPO 并非只能调 clip。** SPO 使用

\[
L_{\rm SPO}=\mathbb E[\rho A-|A|(\rho-1)^2/(2\epsilon)]
\]

使越过期望 ratio 边界的样本仍产生恢复梯度。其逐样本最优 ratio 与范围对齐，不意味着共享神经网络、多 epoch、估计优势下存在全局单调改进保证。APO 则改成均值与方差共同决定的性能下概率界；实用 PAPO 使用平均 KL 阈值提前停止，不能把精确理论目标的结论原样赋给近似实现。[B01,B02]

**混合梯度可以深入计算图，而不是只有一个全局开关。** TPX 在中间分布参数节点估计 LR/RP 的逐坐标协方差组合，减少取得每样本全网络梯度的开销。它明确讨论 Dreamer 实现带来的偏差；同一批数据拟合的混合系数也不自动产生有限样本无偏估计。HPO/MGPO 进一步说明：固定采样离散模式后，连续参数仍会改变后续状态，从而改变未来离散模式的概率；只做连续 BPTT 加离散 PPO 会漏掉交叉项。[B03,B04]

**风险与稳定性不能仅作额外 reward 权重。** RRPO 给出稳健约束问题非零对偶间隙的例子；其理论需要充分准确的 robust critic、覆盖条件，并保留不确定集合直径造成的 \(O(c)\) 误差项。CARL 通过度量相关回报及 freeze-and-learn 学习策略；其最优策略、收缩参考策略存在性和统计校准条件，不能转化为“有限训练 PPO 已保证稳定”。[B05,B07]

**腿足风险学习已有明确的未解决问题。** B12 的 Go2 实验展示了风险水平与步态/推进性的冲突：非常保守的策略在坡或隐藏障碍前会迈不开步；按平均回报选 policy 的 bandit 也不直接优化执行风险。我们应分别约束摔倒/力矩/滑移尾风险与任务推进，不能用“平均 reward 更高”替代这些指标。[B12]

**规划蒸馏和 GPU 梯度 MPC 都已有完整近邻。** BMPC 使用 planner distribution 的模仿、模型 TD 与 lazy reanalyze；其实际 TD horizon 常取 1，反映短模型不适合直接长 rollout。Dream-MPC 从策略产生少量候选，结合动作复用和 ensemble 不确定性做梯度优化；在 BMPC proposal 下更好，在 TD-MPC2 下不能持续胜过 MPPI。减少模型前向次数并不按同样倍数减少墙钟时间。[B08,B10]

**离策略不是 PPO ratio 补一下就完成。** RT-PG 区分历史依赖目标参数与跨时间权重造成的偏差，并用有界 power-mean 权重及集中界分析；不能把其结果理解成任意 replay/多 epoch PPO 无偏。CrossQ 则说明 UTD=1 配合合适 critic 也可取得强结果，提醒我们同时比较环境样本、梯度步和真实墙钟时间。[B09,B11]

## 2. 统一符号与不能混淆的三种导数

\(z\) 是用于决策的完整上下文，包括状态估计、相位、必要历史、控制器 warm state 和观测器状态；\(a\) 是策略采样的原始请求；\(u=E_\psi(z,a)\) 是 MPC/WBC 实际输出的低层命令；\(F_\phi\) 是可选生成模型，\(Q_R,V_R\) 是真实交互数据训练的 critic。

“控制器梯度”是 \(\partial E_\psi/\partial a\) 或 \(\partial E_\psi/\partial\psi\)；“动力学梯度”是 \(\partial F/\partial u,\partial F/\partial z\)；完整 pathwise actor 梯度把它们相乘。若 \(F\) 已直接从原始请求预测整个闭环状态转移，就不再额外乘一遍 MPC/WBC/执行器 Jacobian。若 \(Q_R(z,u)\) 已近似未来回报，其 \(\partial Q_R/\partial u\) 也不是还需再乘完整未来模型 Jacobian 的中间量。

真实 score 梯度与完整 pathwise 梯度可以是同一目标的两个估计器；控制器 Jacobian 与动力学 Jacobian 通常只是这条完整导数的不同因子。所谓可信度是估计误差条件，不是第三个独立梯度。

下面三个算法保留这一点，但它不再是研究主张本身。

## 3. 候选 A：事件时间下的混合策略自然梯度

### 3.1 要解决的问题与独立主张

现有逐帧 PPO 通常把接触模式、步频、落足/力目标混成一个固定频率动作。不同步态对“一个动作”覆盖的物理时间不同；相同 raw action 更新经过 MPC/WBC 后，有时被约束消掉，有时造成很大的执行变化。我们提出同时改变 **决策时标、信用分配、信赖域几何**。

待检验主张 A：在同一奖励和同一可行控制器下，使用事件级策略、正确的模式/持续时间信用和执行变化约束，可以比 hybrid PPO、HPO+TRPO 更快学到可切换步态，同时减少无效 raw-action 更新和切换后的姿态损失。这个联合主张必须分别做消融；仅实现 HPO 的交叉项不算我们的贡献。

### 3.2 策略与目标

在事件 \(k\) 采样

\[
m_k\sim p_\varphi(m\mid z_k),\quad
(v_k,\Delta_k)\sim p_\kappa(v,\Delta\mid z_k,m_k),
\]

其中 \(m\) 为有限接触/步态选项，\(v\) 是连续落足、力、姿态或控制器参考，\(\Delta\in[\Delta_{\min},\Delta_{\max}]\) 为请求持续时间。允许选择的模式需由物理可行性规则定义。保存的是原始采样及其密度；环境可以因安全或真实触地提前结束选项。

\[
\bar r_k=\int_0^{d_k}e^{-\beta t}r(x(t),u(t))dt,\quad
\gamma_k=e^{-\beta d_k},\quad
J=\mathbb E\sum_k e^{-\beta T_k}\bar r_k.
\]

\(d_k\) 是实际执行时间，\(T_k=\sum_{j<k}d_j\)。事件边界 TD 为 \(\bar r_k+\gamma_k V_R(z_{k+1})-V_R(z_k)\)。熵若加入目标，应按持续时间定义其单位，避免以更频繁决策免费获得更多探索奖励。\(\Delta_{\min}>0\) 也避免无限快切换。

### 3.3 完整梯度估计

纯真实数据版本使用 joint score：

\[
g_H=\sum_k e^{-\beta T_k}\nabla\log[p_\varphi(m_k|z_k)p_\kappa(v_k,\Delta_k|z_k,m_k)]\hat A_{R,k}.
\]

这里通常存在 critic/GAE 近似偏差，不能称完整 PPO 无偏。它不要求动力学可微，因而在当前 H0 状态也可开展。

模型版本在固定模式采样和噪声下对连续轨迹求导；**对连续参数还要保留**

\[
g_{\kappa,\rm cross}=\sum_k e^{-\beta T_k}
  \nabla_\kappa\log p_\varphi(m_k\mid z_k(\kappa))\,\operatorname{sg}(\hat A_{M,k}).
\]

这不是对采样整数求导，而是对未来离散概率所依赖的状态求导。对于可微的选项终点，\(\gamma_k V(z_{k+1})\) 对持续时间的导数还包含 \(-\beta\gamma_k V\,\partial d_k/\partial\kappa\)。实际接触提前终止时必须处理 guard/reset 事件及事件时刻灵敏度；不能把固定模式 BPTT 当成正确接触导数。[B04]

可选融合 \(g=(1-\rho)g_H+\rho g_M\)。在理想化 \(g_H\) 无偏、模型偏差 \(b\) 固定条件下，最小均方误差的标量系数为

\[
\rho^*=\operatorname{clip}_{[0,1]}
\frac{\operatorname{tr}(\Sigma_H-\Sigma_{HM})}
{\operatorname{tr}(\Sigma_H+\Sigma_M-\Sigma_{HM}-\Sigma_{MH})+\|b\|^2}.
\]

这是已有估计思想的组件，不是 A 的独立创新。实践用历史独立评估片段估计风险上界；同批拟合 \(\rho\)、critic 误差和 learned-model bias 都需记录。H0 时 \(\rho=0\)，也不借 pathwise control variate 绕开 H0。

### 3.4 更新规则：由执行变化定义的信赖域

令 \(F\) 为 **完整混合动作分布**的 Fisher，\(J_E\) 为在固定上下文下，策略均值经当前控制器变成实际低层命令的 Jacobian。定义

\[
G_E=\mathbb E[J_E^\top W_uJ_E],\qquad
H=F+\lambda_EG_E+\lambda_0 I.
\]

执行映射导数可只涉及确定性的 MPC/WBC 求解器，不要求环境模型梯度。求解器 active-set 改变时局部导数不可靠，要用实际重算的命令变化检验这条近似；它不是物理状态响应的证书。

用 Hessian/Fisher-vector products 近似求解

\[
\max_\delta\ g^\top\delta-\frac{1}{2\eta}\delta^\top H\delta,
\quad
\tfrac12\delta^\top F\delta\le\epsilon_{\rm KL},\quad
\tfrac12\delta^\top G_E\delta\le\epsilon_u.
\]

对候选步长重算完整 joint KL、低层命令变化和 surrogate；不满足则缩步。离散模式概率变化和条件连续分布变化都必须计入 KL，不能只检查高斯头。执行几何的目的，是限制真正高增益的方向，同时识别被控制器投影消掉的无效更新。

### 3.5 内外循环、critic 与模型

1. 冻结控制器版本、normalizer 和行为策略，收集真实事件轨迹；记录实际持续时间、模式、原始采样、old logp 与执行结果。
2. 用真实事件 return/TD 更新 \(V_R\)；可选模型单独拟合真实转移，\(V_M\) 只接模型数据。actor 更新期间两者冻结。
3. 计算真实 score、可选完整 model gradient、Fisher/执行映射向量积，完成一次 constrained natural step。
4. 独立真实评估决定保留，下一批再采样。初版一次 fresh-batch 更新；不声称旧 batch 多 epoch 恢复无偏。

**最小实现**：先只新增有限模式+持续时间头、事件 critic、Fisher 更新；再添加 \(G_E\)，最后在模型导数通过审核后增加混合梯度。每一阶段都是可运行的完整策略更新，而不是等待所有模型完成。

**成本**：基础 score 与 value 网络，外加约定数量的 Fisher-vector products 和控制器命令重算；不建立全参数稠密 Hessian。完整 BPTT 另有 \(O(BH)\) 模型状态保存成本。部署仍是一个小 actor 加原有控制器，训练中二阶向量积不进入每个控制周期。

**理论范围**：精确半马尔可夫 Bellman 方程和混合策略梯度可单独推导；局部二次信赖域不是全局单调改善保证。采样 KL 不是全状态上界；learned model、partial observability、碰撞 grazing 与神经网络投影都会引入额外误差。

**决定性对照**：同动作空间 hybrid PPO、SPO、TRPO、HPO+TRPO；依次移除事件时间、交叉项、执行几何；统一 reward/网络规模/模型访问权限。必须验证收益不是来自“给我们的算法额外步态动作而不给 baseline”。

## 4. 候选 B：相位相容的尾风险与稳定性约束策略改进

### 4.1 要解决的问题与独立主张

“不摔倒”可能靠几乎不走实现；一般收缩条件又可能错误地要求周期步态不同相位的轨迹在绝对坐标中重合。B 将 **任务推进、坏事件尾风险、步态横向稳定性**分别定义，直接改变优化问题和 critic。

待检验主张 B：在相同名义速度与训练随机化下，对横向误差及接触转换实行相位相容的策略改进，比普通 contraction reward 或 return-CVaR 能同时提高推进率、降低滑移/摔倒尾风险，并保持允许的步态节律变化。要证明的是这种“能走且能恢复”的约束设计，而不是“PPO 加 Lyapunov”这一已有组合。[B06,B07,B12]

### 4.2 目标与数据定义

以固定物理时间窗口定义成本 \(C_j\)，分别包含摔倒、执行饱和、滑移和冲击，失败后需保留终止成本/吸收定义，避免提前失败导致看似更小的累计成本。推进 \(P\) 单独衡量速度跟踪、位移与任务完成。

\[
\max_\pi \min_{\omega\in\Omega} J_R^\omega(\pi),\qquad
\max_\omega\operatorname{CVaR}_{\alpha}(C_j^\omega)\le b_j,
\qquad\min_\omega J_P^\omega\ge p_{\min}.
\]

这里 \(\alpha\) 是**上尾成本的分位水平**，例如 0.9；与 B12 下尾回报的 tail mass 参数不是同一记号。\(\omega\) 为整段实验保持一致的地面/质量/延迟等情境，不能在每一步随意拼接成不存在的材料。有限情境集合的 robust 性不是所有未知环境的保证。

\[
\operatorname{CVaR}_{\alpha}(C)=\min_\eta
\left\{\eta+\frac{1}{1-\alpha}\mathbb E[(C-\eta)_+]\right\}.
\]

真实回报 critic 与每类真实成本分布 critic 分开训练。平均 value head 不能提供尾部分布；需要 quantile/distributional cost head，并用真实完整窗口校准尾部。大量相邻 timestep 不是同样数量独立 tail samples。

### 4.3 相位相容的稳定性目标

在去除步态相位自由度和允许的整体平移之后，明确构造横向误差坐标 \(e_\perp\)。其变分矩阵必须包含坐标基随相位变化产生的项，不能简单写一个投影 \(PAP\) 就省略 \(\dot P\)。对模式 \(m\) 的候选条件为

\[
A_{\perp,m}^{\top}M_m+M_mA_{\perp,m}+\dot M_m+2\lambda_mM_m\preceq0.
\]

接触切换必须另有 reset/saltation 条件

\[
\Xi_{mn}^{\top}M_n(x^+)\Xi_{mn}\preceq \kappa_{mn}M_m(x^-),
\qquad
\sum_{\rm cycle}\log\kappa_{mn}-2\sum_m\lambda_m d_m<0.
\]

允许单次触地短暂放大，但完整步态周期净收缩。对自主、非 grazing 的 guard，盐跃矩阵为
\(\Xi=DR+(f^+-DRf^-)n^\top/(n^\top f^-)\)；显式时间依赖要增加相应项或扩充时钟状态。浮动基、MPC/WBC 内部状态、执行延迟均需进入所声称的闭环状态。

这一条件是拟研究的混合系统证书目标；通过有限样本惩罚并不意味着已验证所有状态。\(M\) 用严格正定参数化、上下谱界和尺度归一化，防止靠把度量缩小来作弊。当前 H0 可以先训练真实尾风险/推进目标；解析接触稳定性 actor loss 需等相应局部导数与模式转换审核通过。

### 4.4 策略更新与内外循环

在固定 actor/metric 上，用独立真实窗口估计各情境的奖励、成本 CVaR、推进约束和恢复表现。理论上 CVaR 的 score 梯度在固定最优分位阈值、常规可微条件下为

\[
g_{C_j}=\frac1{1-\alpha}\mathbb E[(C_j-\eta_j)_+\sum_t\nabla\log\pi(a_t|z_t)].
\]

阈值处有原子时采用次梯度；估计分位数、bootstrapping 和同批自适应引入有限样本误差。

策略步骤采用 **primal rectification**：若约束显著违反，选择归一化违反量最大的约束，优先改善该约束；可行时提升最差情境任务回报。用局部线性约束和 Fisher 信赖域计算步长；多个约束局部不相容时，解最小最大违反量恢复问题并报告不可行，而不是让拉格朗日乘子无限增大。维护经过独立检验的最佳可行策略。RRPO 的理论不能直接套用，因为这里使用连续状态、CVaR 和学习度量；它提供算法选择的理由。[B05]

外层先拟合并冻结 metric/成本分布 critic，内层更新 actor，随后重新采样评估；不要 actor 与 metric 同时降低惩罚却缺少独立验证。环境模型只供候选稳定性导数/情境构造，不能把 imagined 成本统计当真实 CVaR。

**最小实现**：在相同 PPO actor 上先做固定窗口的真实成本分布 critic、推进下界与 primal 风险更新；再接入相位横向误差，最后研究跨接触周期条件。初版仍具有完整的非 clip actor 改进规则。

**成本与实时性**：风险统计与 metric 网络主要在训练端；若部署只用 actor/原控制器，无需把矩阵特征值计算放入高频循环。为足够尾部样本增加的真实/高保真交互量需要计入预算。实机在线只在已验证策略/控制器范围内更新风险状态，不以未经验证的模型证书批准危险探索。

**决定性对照**：DR-PPO、APO/PAPO、B12 风险 PPO、ContractionPPO 的实际 reward 机制、CARL 风格 frozen-metric reward、仅 tail constraint、仅 cycle constraint；同时报告名义速度、任务完成率、cost-CVaR、恢复时间、周期横向放大率。若只是牺牲前进变得更稳，B 的主要假设失败。

## 5. 候选 C：在执行命令空间进行离策略联合策略迭代

### 5.1 要解决的问题与独立主张

当前“策略先采样请求，MPC/WBC 再执行”的架构中，更新控制器后继续用旧请求轨迹做 PPO，会改变请求到下一状态的转移律。只重算 raw-action ratio 无法修正这种环境变化。C 主动离开 PPO clip 范式：以 **实际低层命令对应的物理转移**构造真实 replay critic，再在当前控制器内部对 actor 和 MPC 参数一起求导。

待检验主张 C：对执行命令及控制器状态作正确的数据重构后，联合训练能复用旧 MPC/WBC 版本的真实数据，减少重新收集全部样本的代价；同时用模型规划产生的改进分布加速困难步态学习。独立贡献应落在“跨控制器版本的 Bellman 一致性与联合改进”，而不是 MPC 蒸馏、SAC、implicit differentiation 任一已知组件。[B08–B11；v2 AC-MPC]

### 5.2 状态与回放为何必须重新定义

记录 servo 级转移 \((z_t,u_t,r_t,z_{t+1},c_t)\)，其中 \(u_t\) 是真正发给低层执行接口的命令，**不是估计出来的实测力矩标签代替候选输入**。\(z_t\) 包含使转移可预测的延迟/历史/控制器记忆；实际改变了 PD 增益或执行配置时，\(c_t\) 必须进入条件。

单个高层动作期间有多次反馈控制，不能把一个平均力矩当成该整段的完整动作。最低可行版本按固定 servo step 训练；高层策略改进通过完整低层闭环积分回到事件决策。如果部分可观测，用历史/recurrent critic，并承认近似状态不是严格 Markov。

在这些条件下，控制器版本变化主要改变数据采样策略，而不会改变给定 \((z,u,c)\) 的物理转移。这才是复用依据；它不是说任何旧数据、任何机器人参数都能自动兼容。

**这里复用的是物理转移，不是旧版控制器的整个增强状态转移。** MPC 的 warm start、WBC 积分器和 recurrent actor 状态若随参数改变，不能直接把旧 \(h_{t+1}\) 当成当前控制器的下一状态。最低可行版采用无内部记忆或统一初始化的执行映射；扩展版把 \(h\) 与物理 \(z\) 分开，从保存的原始历史重建当前 \(h\)，再用已知当前递推式生成 \(h'\) 计算 Bellman target。无法重建，或内部状态更新还依赖未保存的原始请求时，该段数据不能宣称具有跨版本一致性。真实执行配置改变的转移也必须条件化或分开建模。

### 5.3 真实 Bellman critic 与 actor/MPC 联合更新

以当前 actor 与当前执行映射定义下一动作

\[
a'\sim\pi_\theta(\cdot|z'),\quad u'=E_\psi(z',a'),\quad
y_R=r+\gamma\mathbb E[Q_R^-(z',u',c')-\alpha\log\pi_\theta(a'|z')].
\]

用真实 replay 拟合 \(Q_R(z,u,c)\)。这是 off-policy one-step Bellman learning，不要求把历史行为动作的 logp 冒充当前 on-policy PPO；仍要求覆盖、critic 准确性和稳定的训练设置。熵是请求策略的正则项；当执行映射非可逆时，它不是执行命令分布的熵。

actor/controller 改进目标可写为

\[
\mathcal J_{\rm real}(\theta,\psi)=\mathbb E_{z\sim D,\epsilon}
[Q_R(z,E_\psi(z,a_\theta(z,\epsilon)),c)-\alpha\log\pi_\theta(a_\theta|z)]
-\lambda_{\rm prox}\mathbb E_z D_{\rm KL}(\pi_{\rm old}\|\pi_\theta).
\]

\[
\nabla_\theta Q_R=Q_{R,u}E_{\psi,a}\,a_{\theta,\theta},
\qquad
\nabla_\psi Q_R=Q_{R,u}E_{\psi,\psi}.
\]

这里的 Q 梯度由真实数据 critic 提供，优化器梯度来自当前 solver 的 unroll/KKT；不再另外加一个同义的“动力学梯度”。离散模式用条件枚举（模式数量少时）或 score 梯度；连续条件动作重参数化。若有 projection、饱和、active-set 变化，需采用有效局部导数或不穿越该点的更新，不能宣称 KKT 处处光滑。

控制器成本参数与 actor 在训练端交替更新，且每次只用固定版本 critic/目标网络；避免 actor、critic、model、solver 同一微批全部追逐彼此变化。\(Q_R\) 的近似动作导数也要独立校准，这不是把真实数据 critic 视为无误。

### 5.4 模型规划作为可选的策略改进算子

用真实数据训练 \(F_\phi(z,u,c)\)、奖励和可选模型 critic \(Q_M\)。模型只负责反事实候选，真实 replay critic 保持独立目标。以当前策略和各步态模式为 proposal，生成短候选序列，再用采样或梯度规划。没有可靠动力学梯度时，采样规划仍可运行。

令经不确定性惩罚和可行性筛选的模型候选评分为 \(\hat Q_H(z,a)\)，构造改进分布

\[
q^*(a|z)\propto \pi_{\rm old}(a|z)\exp[\hat Q_H(z,a)/\eta],
\]

并用 \(D_{\rm KL}(q^*\|\pi_\theta)\) 投影到 actor，和上面的真实 critic 改进项交替执行。有限候选只近似这一分布；对离散模式保持显式分量，避免把两种互斥接触计划压成一个不可行高斯均值。\(q^*\) 是冻结 teacher，不能对其生成过程和 actor 拟合重复反传。

这种 mirror policy improvement/teacher projection 本身有 MPO/专家迭代祖先，BMPC 已实现重要近邻；我们的额外研究对象是 **执行命令 replay 下联合变化的 controller、模式分量保留和真实 critic 对规划的校正**。若只是用固定 BMPC teacher 蒸馏，应归为 baseline。

可选实质扩展是拟合真实 Bellman 残差

\[
e_V(z,u)=\mathbb E_P[r+\gamma V_R(z')]-\mathbb E_{F_\phi}[\hat r+\gamma V_R(\hat z')],
\]

用它校正短模型评价，并按真实 critic 与 planner 的改善分歧分配 reanalyze。**只有在固定 V、条件期望准确时，\(T_{F}V+e_V=T_PV\) 是一阶 Bellman 等式；它不消除多步模型状态分布误差，也不是有限数据无偏保证。** 是否比单独 Q ensemble、BMPC fixed-interval reanalyze 更有效，必须作为独立消融。

### 5.5 完整内外循环、成本与理论界限

1. 当前 actor+controller 在高保真环境采样；保存低层实际命令和完整条件，形成版本明确的 replay。
2. 只用真实 replay 更新 \(Q_R\)，另行拟合 model/模型 critic；target network/分布归一化固定后再做策略改进。
3. 用当前 \(E_\psi\) 重新计算 replay 状态的候选命令；交替更新 actor 与 MPC 参数。把成本参数硬约束在正定/物理允许范围内。
4. 可选后台短规划，得到保持模式分量的 teacher；在限定 reanalyze 预算内监督 actor。
5. 独立真实评估近期 actor/controller 对；必要时回到已验证参数，不把 planner 预测 improvement 当部署许可。

**成本**：一阶 Q 更新通常比全模型 BPTT 便宜；额外 solver 隐式求导、规划 teacher 和 replay 存储必须实测。报告 UTD、每个真实物理秒的梯度步、每个采样状态的 solver 调用次数、重规划比例与显存峰值。部署可保留小 actor+快速 MPC/WBC，也可专门比较蒸馏后的 actor；两种架构的延迟与鲁棒性应分别报告。

**理论范围**：在完整 Markov 状态、同一物理低层转移、精确 critic、足够覆盖等理想条件下，旧控制器数据可以用于当前策略 Bellman 评估；神经网络 critic 的离策略收敛和函数逼近误差没有自动保证。规划 soft improvement 的逐状态性质不等于近似投影后的全局单调改善。

**决定性对照**：SAC/CrossQ、FastTD3、TD-MPC2、BMPC、Dream-MPC、AC-MPC；请求空间 replay 与执行空间 replay、固定控制器与联合更新、无 teacher/单高斯 teacher/显式模式 teacher、固定重分析与分歧分配、无 Bellman 残差校正。数据版本一致性和 actor+MPC 联合更新必须真正比“冻结控制器后训练”的方案多解决一个问题。

## 6. 如何选主线并并行，而不是把所有部件一次塞进去

| 研究臂 | 近期可做的完整学习问题 | 依赖模型导数吗 | 优先产出 | 最大风险 |
|---|---|---|---|---|
| A | 事件级混合策略、自然梯度/执行几何 | 基础版不依赖 | 步态切换质量、样本效率、无效动作率 | 模式/持续时间动作设计本身带来优势，掩盖更新算法贡献 |
| B | 真实 cost-tail critic、推进下界、primal 风险改进 | 基础版不依赖；混合收缩扩展依赖 | 风险–推进 Pareto、扰动恢复 | 更保守却不前进；样本不够估尾部 |
| C | 执行空间 off-policy critic、actor/controller 联合改进 | Q/solver 导数；模型规划可用采样 | 数据复用效率、联合更新收益、时间成本 | 缺少 Markov 状态或覆盖；solver 导数/critic 梯度偏差 |

建议先 A 做主学习算法，B 与 C 各形成可比较的独立分支；不要宣称必须把三种算法相加才是最终方案。三条分支共享训练/验证情境和评估脚本，但共享数据需保留行为策略、控制器版本与真实/模型来源。

最低有效对照先包含现有 PPO、SPO 和现代离策略基线，避免只和一个未充分调优的老 PPO 比。单种子用于排除实现错误和资源不可行，算法主张需多种子、同样环境预算和同样墙钟预算分别评价；不固定一个可能挤占 GPU 的任务数。

25 核/96 GB 可同时进行模型拟合、控制器导数审计、独立策略训练/评估，但是否运行 2、3 或 6 个重训练，取决于**同一组合下**测得的吞吐、显存峰值与 p95/p99 控制延迟。资源利用率高而训练每样本慢，不是效率提升。训练按物理步数、episode 数、有效评估与可恢复 checkpoint 推进，不设置随意的墙钟 kill deadline。

### 一次研究迭代的最小报告字段

记录完成的真实物理交互时长、并行环境数、有效事件数、gradient/solver 调用数、每模块训练耗时、吞吐、显存峰值、推理尾延迟；同时记录推进/失败/滑移/冲击/执行饱和、接触模式和持续时间分布。新增算法的每个 claimed gain 必须有等预算 baseline 与机制消融。

每日工作记录应分别写“新实现”“发现/排除的问题”“数据与代码版本”“实际结果”“未成立的主张”“下一项决定性实验”，避免把完成实现写成完成实机验证。

## 7. 附录：源码与理论审计中的关键边界

### 7.1 ContractionPPO：借鉴训练思想，不照搬其所读定理

所读 arXiv PDF 第 5 页已渲染核验：残差定义/分解包含 \(+\alpha I\)，式 (13) 和证明末尾却采用 \(-\alpha\) 上界；并写出一般不成立的 \(\lambda_{\max}(S)=\|S\|_2\)。谱范数非负，不能据其被负上界约束推出所述结论。式 (13) 的 \(m_{\min}^{-1/2}\) 与式 (17) 的 \(m_{\min}^{-1}\) 也不一致。因此当前文本的 Theorem 1 不可直接用作本项目证书。这是可定位的公式问题，不等于否定硬件实验。[B06]

官方 pin `e63ed83516e46a2630ad203115d069d83de98f4b` 中，`contraction_metric.py` 对三角因子对角使用 softplus+最小值，实际处理了严格正定条件。`contraction_ppo.py` 将采集的状态/动作 detach，单独更新 metric；在 PPO 总 loss 中直接加 contraction loss 的语句被注释。其 `process_env_step` 把 detached contraction reward 加入真实 rollout reward，所以 actor 通过 score/优势学习该奖励。它不等价于“对完整浮动基闭环 Jacobian 的 contraction loss 直接联合反传”。我们只做静态源码审计，没有运行该代码。[B06]

### 7.2 BMPC：KL 方向与实际代码

论文 Eq. (6)/(8) 写 expert→policy KL；官方 pin `d5dabc6cb58d463829571548c47711dff074f5af` 的 `bmpc.py:update_pi` 传入 `(actions_dist, expert_action_dist)`，`common/math.py:kl_div` 按该顺序计算 `kl_divergence(normal1, normal2)`，实际是 policy→expert。两种方向对多模态/方差学习不等价；复现实验应明确采用哪一版，不能以论文公式和代码混用来解释结果。[B10]

### 7.3 NeuralActuator 只是一种可选生成/观测接口

上述算法可使用刚体模型、任务相关 latent model、PANN 或普通历史状态模型。若另外接入 NeuralActuator：posterior observer 只看至当前的历史与实测 telemetry；candidate action-conditioned generative model 才能预测未来命令的效果。未来记录电流不能进入 imagined 候选；整条模型若已包含执行器和控制器，不重复乘它们的 Jacobian。去掉未来 posterior correction 的 imagination 是预测模型，不自动等于完整真实 Bayes 闭环导数。

## 8. 主要来源入口

- B01：https://proceedings.mlr.press/v267/xie25m.html
- B02：https://proceedings.mlr.press/v235/zhao24i.html
- B03：https://proceedings.mlr.press/v202/parmas23a.html
- B04：https://arxiv.org/abs/2605.14297 ；官方 NeurIPS 文稿 https://nips.cc/media/neurips-2026/Slides/151120_LYhDiJP.pdf
- B05：https://openreview.net/forum?id=7l63xwAgAW ；https://arxiv.org/abs/2508.17448
- B06：https://contractionppo.github.io/ ；https://arxiv.org/abs/2603.19632
- B07：https://ieeexplore.ieee.org/abstract/document/11664333 ；https://arxiv.org/abs/2506.15700
- B08：https://proceedings.mlr.press/v306/spieler26a.html
- B09：https://proceedings.mlr.press/v306/montenegro26a.html
- B10：https://proceedings.iclr.cc/paper_files/paper/2025/hash/8636419dea1aa9fbd25fc4248e702da4-Abstract-Conference.html
- B11：https://proceedings.iclr.cc/paper_files/paper/2024/hash/f381114cf5aba4e45552869863deaaa7-Abstract-Conference.html
- B12：https://ieeexplore.ieee.org/abstract/document/11695854 ；https://arxiv.org/abs/2510.14338
