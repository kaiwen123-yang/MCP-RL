# 规划续接方向与策略梯度控制变量：一手近邻核查

日期：2026-10-20。对应[候选主方法](../../docs/go2-adjoint-learning-method-2026-10-20.md)。

本轮检索围绕“事实动力学与规划余程给出动作方向，用带score补偿的控制变量辅助完整任务PPO”。核查了下列一手论文的方法、公式、相关证明或讨论段落，并区分正式年份、首稿年份、作者主张与本项目判断。未完成各论文代码复现，也不以本次定向核查更新主文献库或宣称全面查新。

## 1. 直接判定

**控制变量估计器的基本原理已有明确前人。** Q-Prop已覆盖线性化critic加score补偿，且明确讨论模型构造的critic；IPG已把相邻方法置于统一的偏差—方差框架。2026年的MFPG及MFPG-PPO进一步处理低保真模型、PPO和机器人学习。因此，“MPC/模型方向＋PPO”“补偿后不额外偏移期望”“模型帮助少样本学习”不能独立作为新原理。

本项目可检验的具体取舍是：当前实际反馈的微步事实模型、规划已选续接命令和实际余程价值，能否形成比一般critic更有用的动作方向，从而改善多阶段任务的有限数据学习；其短展开与解析均值补偿是否比额外模型rollout更省成本。这是候选问题，不是已确认空白。

## 2. Q-Prop：估计器骨架与模型方向已有先例

**Shixiang Gu, Timothy Lillicrap, Zoubin Ghahramani, Richard E. Turner, Sergey Levine. _Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic_. ICLR 2017；arXiv首稿2016-11-07。**

一手来源：[原文PDF](https://arxiv.org/pdf/1611.02247)、[首稿与版本记录](https://arxiv.org/abs/1611.02247)、[作者机构正式发表记录](https://research.google/pubs/q-prop-sample-efficient-policy-gradient-with-an-off-policy-critic/)。

本轮核查范围：策略梯度控制变量构造、线性化critic、附录估计器推导及模型型critic讨论。

作者方法：用critic在策略均值处的一阶展开构造动作相关控制变量；从采样策略梯度中减去该项，再加回可计算的期望梯度。原文附录还明确列出model-based critic作为可选形式。

本项目判断：固定方向\(d\)与解析均值补偿直接继承这一类恒等式。规划costate经事实动力学构造\(d\)是其信息来源的具体选择，不能仅因方向来自MPC就宣布新的策略梯度机制。

## 3. IPG：补偿与直接模型梯度混合应分开

**Shixiang Gu, Timothy Lillicrap, Zoubin Ghahramani, Richard E. Turner, Bernhard Schölkopf, Sergey Levine. _Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning_. NeurIPS 2017；arXiv首稿2017-06-01。**

一手来源：[正式论文页面](https://papers.nips.cc/paper_files/paper/2017/hash/a1d7311f2a312426d710e1c617fcbc8c-Abstract.html)、[正式PDF](https://papers.nips.cc/paper/6974-interpolated-policy-gradient-merging-on-policy-and-off-policy-gradient-estimation-for-deep-reinforcement-learning.pdf)、[版本记录](https://arxiv.org/abs/1706.00387)。

本轮核查范围：§3的估计器、控制变量与on/off-policy混合关系，§4的偏差分析入口。

作者方法：统一分析采样策略梯度与critic梯度的插值；改变状态采样分布、以不准确critic直接替代任务梯度会产生需要分析的偏差。

本项目判断：当前候选只增加期望为零的修正，不直接优化模型虚拟回报。不应一面使用有意偏置的直接模型梯度，一面引用零均值控制变量的结论。普通同信息模型辅助策略优化仍是有力对照。

## 4. Mirage：无偏公式不保证实现无偏或实际有益

**George Tucker, Surya Bhupatiraju, Shixiang Gu, Richard Turner, Zoubin Ghahramani, Sergey Levine. _The Mirage of Action-Dependent Baselines in Reinforcement Learning_. ICML 2018，PMLR 80:5015–5024；arXiv首稿2018-02-27。**

一手来源：[正式页面](https://proceedings.mlr.press/v80/tucker18a.html)、[正式PDF](https://proceedings.mlr.press/v80/tucker18a/tucker18a.pdf)、[公开版本记录](https://arxiv.org/abs/1802.10031)。

本轮核查范围：方差分解、§4.1优势归一化及原实现偏差分析。

作者发现：在所测试域中，学得的动作相关baseline未显著超过状态baseline；若只归一化残差学习信号而不同比处理补偿项，会偏离理论公式。论文没有证明所有动作相关控制变量都无效。

本项目落实：不对\(\widehat A-\alpha b\)另作归一化，不把残差送进PPO clip后继续使用未裁剪补偿。采用“原PPO项＋独立零均值修正”，并将真实任务效果与局部方差统计分开。

## 5. MFPG：2026正式近邻，首稿属于2025

**Xinjie Liu, Cyrus Neary, Kushagra Gupta, Wesley A. Suttle, Christian Ellis, Ufuk Topcu, David Fridovich-Keil. _A Multi-Fidelity Control Variate Approach for Policy Gradient Estimation_. TMLR 2026；arXiv首稿2025-03-07，核查公开版v4为2026-02-12。**

一手来源：[v4原文](https://arxiv.org/html/2503.05696v4)、[版本记录](https://arxiv.org/abs/2503.05696)、[作者发表记录](https://xinjie-liu.github.io/)、[正式评审入口](https://openreview.net/forum?id=zAo0L7Dcqt)。

本轮OpenReview入口返回浏览器验证页，因此机制核查使用作者的arXiv v4；正式发表身份另由作者记录及后续论文引用交叉确认，没有声称读到了受阻页面的完整评审。

本轮核查范围：摘要、§4的多保真控制变量构造、理论结论及版本记录。

作者方法：将少量目标环境数据与大量低保真数据组合，以相关低保真统计量及其均值构造控制变量；给出特定假设下REINFORCE的理论分析。

本项目判断：通过近似模型降低真实数据需求已经有近期直接工作。当前候选不需要额外大批低保真rollout来估计线性控制变量的均值，但它只覆盖一个局部动作函数；这种计算取舍需要实际比较，不能由公式自动宣布优越。

## 6. MFPG-PPO：2026年10月的直接后续

**Xinjie Liu, Ruihan Zhao, Anirban Chaudhuri, Cyrus Neary, Ufuk Topcu, David Fridovich-Keil. _Multi-Fidelity Policy Gradients Stabilize Data-Scarce Reinforcement Learning_. arXiv:2610.02505v1，2026-10-01公开；本次核查时为预印本。**

一手来源：[原文](https://arxiv.org/html/2610.02505v1)、[版本记录](https://arxiv.org/abs/2610.02505)、[作者项目页](https://xinjie-liu.github.io/mfpg-ppo/)。

本轮核查范围：§3、§4.1的PPO统计量、配对数据、GAE与系数构造，§5相关消融，§6局限。没有执行其代码。

作者方法：将高保真、配对低保真、独立低保真三批PPO统计量组合；处理轨迹相关性、多epoch更新和估计方差膨胀，并报告并行运动控制及真实Franka学习。作者明确指出，配对轨迹重同步而独立轨迹自由演化可能造成修正均值不一致，带来偏差。

本项目判断：PPO、GPU并行、机器人与控制变量的组合已有非常直接的近邻。当前方法以条件动作均值解析补偿，避免该特定跨轨迹均值估计，但不处理它所覆盖的全部轨迹噪声。这不是现阶段可以声称全面胜出的关系。

作者明确未来方向包括改善相关性、处理重同步的偏差取舍和降低低保真环境构造成本；“用任务规划续接构造局部方向”是本项目推断的候选，不能写成作者原话。

## 7. TrajCV：局部动作控制变量的长程边界

**Ching-An Cheng, Xinyan Yan, Byron Boots. _Trajectory-wise Control Variates for Variance Reduction in Policy Gradient Methods_. CoRL论文，PMLR 100:1379–1394，正式卷记录2020。最早公开稿年份本轮未独立核定。**

一手来源：[正式页面](https://proceedings.mlr.press/v100/cheng20a.html)、[正式PDF](https://proceedings.mlr.press/v100/cheng20a/cheng20a.pdf)。

本轮核查范围：§3的方差分解、§4的轨迹控制变量动机与构造。

作者方法：利用跨时间相关性减少仅靠state-action控制变量未消除的未来轨迹方差。

本项目判断：本文方向虽然包含余程critic，修正本身仍主要围绕当前动作。不能因为任务有四阶段，就宣称该局部修正解决了长程信用分配。完整轨迹学习速度与阶段接续表现需要实际证据。

## 8. Costate-focused RL与本项目的继承关系

**Bita Behrouzi, Xuefei Liu, Douglas Tweed. _Costate-focused models for reinforcement learning_. arXiv:1711.05817；首稿2017-11-15。本文不据本次检索补写未经核定的正式刊会身份。**

一手来源：[作者公开稿记录](https://arxiv.org/abs/1711.05817)。

本轮核查到的作者主张：用costate改进策略，并使模型更关注与任务相关的动力学特征。

本项目判断：costate与动力学导数相乘来构造策略更新方向已有直接基础。候选的区别应落在实际多阶段任务、共同事实模型、仅依赖采样前信息的方向和PPO外补偿的具体作用，不在于第一次使用伴随变量。实现于B批采集后、任何参数更新或本批拟合前用旧网络批量重构方向；这不表示方向可以利用本次动作或未来真实结果。事实模型仍以实际转移为标签；本轮不加入可自由降低模型自评成本的目标。

## 9. 当前新意边界与必要对照

目前可以定位为：**规划续接信息构造的局部动作控制变量，在相同事实建模与实际任务学习体系中的机制候选。**

不能单独作为新意的元素包括：线性化Q函数、模型辅助动作梯度、伴随反传、补偿保持条件期望、PPO与控制变量结合、任务条件策略、模型随策略更新。

如果候选在统一Go2体系中显示完整任务或样本效率改善，接下来的直接比较应回答两件事：

1. 相同补偿形式下，规划构造方向是否优于同数据神经Q critic方向；否则增益可能主要来自已知Q-Prop机制。
2. 相比MFPG类利用更多低保真数据的方式，局部事实模型与解析补偿是否在完整数据/计算账本下提供有价值的取舍。

这些比较是解释未来正结果的必要步骤，不是本轮新增训练授权，也不是在实现当前三组之前追加庞大的资格检查。第一版固定\(\alpha=0.25\)，不宣称最优系数、保证降方差或已成立的论文创新。

当前实现的普通PPO与候选组均使用\(\gamma=1\)的实际Monte Carlo后缀return，优势不归一化、未采用GAE；只有未完成的采集截断接旧critic。候选方向使用不超过一个宏周期的旧均值闭环与带宏周期相位的同一个实际return critic。f只拟合B+C微步事实，g只拟合更新后确定性C的完整宏周期，V只拟合B的实际return。上述实现选择与近邻论文各自的GAE、多保真数据和归一化设置应分别记录，不能把它们表述为已有方法的完整复现。

本轮允许原nominal的一条无参数更新整程接线检查；其执行状态与数值以证据报告为准。接线通过不等于训练收益，本近邻文档不预报候选效果。
