# 控制理论结合点：有边界的谱系检索与本轮取舍

日期：2026-10-15。固定问题：在输入受限、存在扰动和复杂执行动态的机器人上，共同设计对任务有用、实际可实现且适合在线优化的闭环动力学与反馈策略。结构固定为MPC规划、RL执行。本报告不改变科学问题、不增加训练任务、不把控制理论工具本身申报创新。

## 1. 结论与阅读边界

本轮按11类主要控制理论谱系检索相关一次来源，结合前一轮已读MR-ARL、反馈线性化RL、控制抽象等材料；这是针对明确变量的覆盖检索，不能声称遍历了全部控制理论。关键新增发现是：**以闭环响应为设计变量、保证响应可实现、共同优化反馈与误差管，都已经有成熟理论甚至直接非线性工作。** 这没有否定问题价值，但决定了论文必须落实到输入和执行器限制下现有方法未解决的具体困难与同等条件的真实收益。

本轮只把三项实际带入原型与算法：

1. **动态扩展与backstepping：**保留执行器状态，构造可调且充分整定的经典执行控制器，提供非线性响应的可实现起点与强对照。
2. **增量ISS及多速率实现关系：**把段内误差、执行器误差、真实输入和命令接续放进同一个证明对象，再由它给MPC收紧约束。
3. **状态—输入成对响应与约束联合设计：**借鉴SLS的因果可实现性思想，联合优化任务响应与实现控制器；不把整个SLS求解器或一套额外安全QP机械叠入结构。

优化工具可以是约束分块优化、原始—对偶或可微MPC。它们解决如何计算，不能替代这三个对象的定义。passivity、CBF、HJ reachability、dual control暂作为替代路线或核对工具；没有证据证明本轮需要同时部署它们。

### 本轮读取层级

- **正文方法/定理局部读取，非全文：**非线性SLS[7] §II–III（含可实现性约束、FIR构造、向量系统限制与部分反馈线性化）；约束SLS[8] §II、§III定理2–3；收缩RAMPC[4] §2、§3.1–3.2关键条件、§3.5定理12与证明起始；多速率MPC/CBF[10] §III性质1–4、定理1/Remark1、§IV及定理2；Differentiable MPC[11] §4.2–4.3；CBF综述[6] 定义与定理2/Remark5；近期Koopman收缩MPC[15] Assumption1、Proposition1–2、Theorem1及相关证明与算法描述。
- **摘要/引言/作者明确限制层面筛选：**动态面控制[3]官方摘要；RG[5]官方作者机构摘要/导论；energy shaping[9]作者稿摘要；dual control[12]正式论文导论；HJ[13]作者稿摘要；近期DC管MPC[14]摘要/引言与问题定位；反馈线性化RL[1]、MR-ARL[2]本轮重新核对作者页面摘要，深读依据沿用前轮记录。
- 本轮没有读取上述所有论文的全部附录，也没有独立复核全部证明；新发现的2026预印本没有被称为正式发表论文，没有占用主100篇经典名额。

## 2. 各谱系能提供什么，以及不能替我们证明什么

表中的“本研究判断”是阅读者推断；它不是作者写出的未来工作。

| 谱系 | 作者结果/明确适用条件 | 对本任务的帮助与已有覆盖 | 本研究判断与本轮取舍 |
|---|---|---|---|
| 模型参考、自适应与反馈线性化 | 反馈线性化RL[1]学习线性化控制器；MR-ARL[2]已有变化参考模型与实际反馈同步适应 | “RL实现可规划参考模型”“参考模型也优化”均已有先例 | 可继承匹配/实现条件。不能忽略相对阶、内部动态和饱和；当前难点应落在受限执行与任务共同设计，而非换一个可学习参考模型名称 |
| Backstepping / dynamic surface | DSC[3]针对strict-feedback系统，用滤波减少反复求导；官方摘要明确非Lipschitz情形为半全局界 | 执行器一阶动态可以作为真实附加子系统，递推构造控制而不是假设瞬时动作到位 | 本轮优先backstepping作经典基线；若使用动态面滤波，必须把滤波状态和误差计入，不能滤掉后仍声称精确实现原参考 |
| Anti-windup / reference governor | RG[5]修改已稳定闭环的参考以满足约束；已有不精确优化的command governor | 饱和处理与可执行命令管理已有丰富方法，不能用只clip的弱PID作为唯一经典对照 | 基线应有anti-windup或相同约束命令管理。若MPC已负责约束，不为“创新点”重复叠RG；证明仍须使用实际限幅后的控制 |
| ISS / small-gain / contraction | 收缩RAMPC[4]在有界不确定性、约束域内CCM及终端条件下给出管递推与递归可行 | 连接执行误差、模型误差与任务规划。非线性响应模型与输入误差管可共同分析 | 本轮先用低维显式增量Lyapunov/ISS关系；神经CCM和在线测地线不是当前必需。条件只在声明域内有效 |
| Tube / robust MPC | [4]已有状态/输入相关管传播和参数集合更新；2026 DC工作[14]已共同优化管几何与顶点控制 | 鲁棒约束满足与递归可行是可继承的理论底座 | “更小管”“学习反馈+管”不能单独申报创新。需要展示共同响应设计怎样改变任务允许范围，而非只增加管的自由度 |
| Passivity / energy shaping | IDA-PBC通过匹配方程赋予期望能量与耗散；神经IDA-PBC[9]已学习匹配条件 | 对机械系统保留有利自然动态、避免过度抵消、给能量解释很有帮助 | 若最终任务是接触交互/操作，值得进入下一阶段；当前无证据必须把能量函数引入首个执行器原型。稳定能量设计不自动处理硬输入限幅 |
| Reachability / CBF / viability | CBF[6]保证条件满足时的前向不变；HJ[13]对非线性与有界扰动直接处理但维数成本高 | 可用于离线核对可实现域或证明其保守性，避免只在采样点上观察“不失败” | 本轮可把低维可达性作为独立检查；CBF若改变最终执行动作，必须把它计入同一策略响应，不能证书和执行器各用不同动作 |
| SLS / Youla式响应参数化 | 非线性SLS[7]明确给因果闭环映射的实现约束，部分抵消跨时间分配；约束SLS[8]已处理状态/输入限制与饱和 | 直接回应“共同优化什么”：必须成对描述状态响应和实现它的输入响应 | 采用这个原则，不要求把机器人化成全局多项式SLS。原文向量构造明确直接控制每个状态且输入可能无界，不能直接覆盖带执行器的欠驱动机器人 |
| Adaptive / dual control | MR-ARL[2]自适应参考；dual control[12]同时考虑动作对状态与未来信息的作用，完整Bayesian问题不可解而作近似 | 可区分“策略真正变强”与“只是辨识更准确” | 本轮固定已知原型与扰动范围，先排除学习信息差异，不引入主动探索新变量。将来在线未知环境适应才有充分理由引入dual control |
| Hybrid / multirate | [10]已有低频MPC、高频非线性控制的段内与重置约束分析；作者Remark1明确理论低层连续更新 | 高层段末相同不等于段内安全；命令切换不能把误差清零 | 本轮必须保持真实执行器状态与误差接续，离散控制需加入采样/保持误差。多速率结构本身已有先例 |
| Optimization / bilevel / differentiable control | [11]固定点KKT求导已有直接基础；作者§4.3明确未收敛时直接套固定点导数可能错误 | 提供响应参数影响MPC再影响真实策略的计算路径 | 采用完整实际映射梯度或经过核验的零阶更新；有限迭代MPC不能假装求到精确固定点。联合优化不是自动全局最优 |

## 3. 三项结合如何具体进入本轮原型

以下是本研究的设计建议/推导要求，不是声称从某一篇原文得到新的定理。

### 3.1 用动态扩展构造真实的经典实现器

一个合适的分析对象包含机械状态与执行器状态，例如

\[
\dot q=v,\qquad \dot v=f(q,v)+b(q)a+d,
\qquad \tau\dot a=-a+\operatorname{sat}_{U}(u).
\]

其中a是实际执行器输出，u是命令，二者不能混同。若另有限速，则执行器方程需要真实限速项；若策略只看到带噪状态，经典对照须相同。为了推导可控，第一原型可先在b有固定符号且远离零的紧域开展，不把该条件外推到全机器人。

根据期望响应g构造虚拟执行器目标a_d，再递推设计u。可实现性取决于a_d和它的变化率所需的u是否落入限幅范围。即使输出响应看似简单，快速变化也可能需要超出执行器带宽或输入容量的动作。这是动态扩展给共同设计的实质约束。

必须保留两种经典对照：充分整定固定响应的实现器；相同策略族、参数数目及调参预算下，允许经典方法按完整任务直接优化。后一种若达到相同最优结果，应如实得出“在该原型中RL无额外必要性”，仍可保留解析原型作为机制说明。

### 3.2 用增量ISS与多速率关系把实现性变成可验命题

令e包含输出/速度与参考的误差，s=a-a_d为执行器误差。可先寻找V(e,s)，而非只约束段末位置误差。一个足够明确的目标形式为

\[
\dot V\le -\lambda V+\sigma_d\bar d^2+\sigma_\delta\bar\delta^2,
\qquad |u_{\rm actual}|\le U,
\]

其中delta必须包含确实存在的模型、估计、采样保持和近似实现误差，不能直接设成零。若只有互联子系统ISS界，则检查增益环条件而非凭“各层稳定”推出整体稳定。

参考命令每T秒改变时，要满足实际重置/接续映射

\[
V(e^+,s^+)\le \chi V(e^-,s^-)+b_{\rm switch}(c^+,c^-).
\]

连续段收缩与切换放大共同决定可接续性。若采用固定参数并持续保存参考状态，可减少不必要重置；如果按新命令重建a_d，s会随之改变，这个变化必须进入界。简单地每段把参考重置成真实输出，再把误差写为零，会隐藏执行器记忆和未来动作需求。

进入MPC的不是独立选取的误差惩罚，而是同一关系推得的状态和输入裕度。稳态界V≤b/λ只有在相关域不变、输入条件持续成立时有效。仿真没有越界不等于这些前提已证明。

### 3.3 用成对响应约束任务共同设计

响应模型应至少表达候选命令下的任务状态轨迹，以及实现它所需的输入/执行器资源；它们不能分别朝乐观方向自由拟合。借鉴SLS，可把候选命令、已发生扰动和初始内部状态到实际状态/输入的因果映射作为一致对象。

低维原型可由显式反馈直接诱导该对象，联合设计参数p=(eta,theta)满足

\[
\min_p J_{\rm physical}(p),\quad
\text{s.t. }\mathcal R_p\text{非空、段内可保持、跨段可接续，且实际输入受限}.
\]

这里eta改变期望任务响应，theta改变实际执行反馈；二者只在关系R_p允许时成对被使用。先允许一般约束优化解这个小问题，再判断机器人阶段为什么需要RL来实现复杂反馈。解析阶段不必为了名字而强行使用RL。

理论上的最小应达结论是：在声明域、采样率、扰动集合与可行初态内，成对实现关系保持；MPC若使用对应收紧和终端条件则具有相应递归可行性；模型任务收益与真实任务收益之间有实际可计算的差界。若算法只保证局部驻点，不许写全局最优共同设计或单调真实改进。

## 4. 直接影响创新性的三个新发现

### 非线性SLS是需要正面对照的理论近邻

[7]的状态—输入闭环映射实现条件，以及跨时间安排非线性抵消，已经覆盖“设计闭环响应并节约输入”的宽泛主张。作者明确把向量构造建立在每个状态直接受控、输入可能无界之上，并限制时不变映射。本任务的执行器状态、硬输入与高低频信息约束提供了具体不同条件，但**不能仅列出这些条件就宣布创新**；需要新可解算法和可信收益。仅把全状态直接作用的u换成网络输出，而不处理执行器内部动态，不算突破该边界。

### 非线性管的共同优化已经很强

2026-09的[14]已在差凸结构下联合优化管几何与顶点控制；[15]又将Koopman近似误差、收缩和约束收紧连在一起。因此“我们的管随任务变化”“同时学习控制器与误差界”必须进一步说明已有管方法未解决的内容。当前更有意义的对象是**目标响应本身可设计、受实际执行与段内输入共同约束**，其方法与计算收益仍须证明。

### 多速率保证不能从连续低层直接搬到Simulink

[10]的Remark1明确指出低层连续更新假设，尽管仿真以高频离散实现。我们的Simulink模型若使用固定步长和零阶保持，需要研究该实现，而不是用连续证明为任意步长担保。应区分积分器数值误差与控制器采样保持误差。前者可以用求解器收敛检查降低，后者是真实控制问题的一部分。

## 5. 公平比较与创新判别

- 固定输出坐标和物理目标；不能eta变化后仍用同一个命令能量惩罚，而把命令缩放当物理节能。
- 相同观测、执行频率、扰动可见时间、预览和执行器模型。不能让拟方法低层即时看到扰动，而让经典反馈只能开环保持。
- 固定响应基线必须在同一训练任务分布上充分整定，包括合理anti-windup或命令可行管理。
- 同参数数量/同反馈族的完整任务直接优化是强对照。若共同设计只等价于该问题的换元，收益应为零；若优化速度更好，应报告为优化效率而非能力集合增大。
- 增加动态记忆、候选响应数或策略容量时，对照获得相同容量，否则不能排除额外自由度贡献。
- 若理论域极小、误差界比任务收益大很多，不能仅凭“有定理”宣布有价值。须同时报告可实现域大小、计算开销和真实任务表现。
- 仿真胜过未整定基线不能证明新方法；经典反馈达到同样表现也不否定科学问题，但说明该算例不足以证明RL或新耦合的必要性。

## 6. 一次来源及版本

以下只作为方法理论补充，不自动加入主文献库。链接所指预印本年份与正式发表年分开；未核正式记录不推测。

1. Westenbroek et al., *Feedback Linearization for Unknown Systems via Reinforcement Learning*, first arXiv 2019；正式题名版本为*Feedback Linearization for Uncertain Systems via Reinforcement Learning*, ICRA 2020。 [作者稿](https://arxiv.org/abs/1910.13272)；[作者机构PDF](https://people.eecs.berkeley.edu/~sastry/pubs/Pdfs%20of%202019/WestenbroekFeedback2019.pdf)。本轮摘要核对；深读见前轮笔记。
2. Borghesi, Bosso, Notarstefano, *MR-ARL: Model Reference Adaptive Reinforcement Learning for Robustly Stable On-Policy Data-Driven LQR*, first arXiv 2024。[作者稿](https://arxiv.org/abs/2402.14483)。正式版本关系沿用前轮核验，本轮不宣称重读正式版。
3. Swaroop et al., *Dynamic Surface Control for a Class of Nonlinear Systems*, IEEE TAC 45(10), 1893–1899, 2000；前身ACC 1997已存在，本轮未精读会议版。[官方记录及摘要](https://ieeexplore.ieee.org/document/880994/)；DOI:10.1109/TAC.2000.880994。
4. Sasfi, Zeilinger, Köhler, *Robust Adaptive MPC using Control Contraction Metrics*, first arXiv 2022；Automatica 155 (2023) 111169。[已接受版本v2正文](https://arxiv.org/html/2209.11713v2)；[正式DOI](https://doi.org/10.1016/j.automatica.2023.111169)。
5. Kolmanovsky, Garone, Di Cairano, *Reference and Command Governors: A Tutorial on Their Theory and Automotive Applications*, 2014作者机构报告。[MERL原文](https://www.merl.com/publications/docs/TR2014-119.pdf)。并查Garone, Kolmanovsky, *Command Governors with Inexact Optimization and without Invariance*, first arXiv 2021。[作者稿](https://arxiv.org/abs/2111.10234)。
6. Ames et al., *Control Barrier Functions: Theory and Applications*, 2019。[作者稿正文](https://arxiv.org/html/1903.11199v1)。本轮只核定义/定理2与相关备注。
7. Conger, Li, Mazumdar, Brunton, *Nonlinear System Level Synthesis for Polynomial Dynamical Systems*, arXiv 2022。[v2正文](https://arxiv.org/html/2205.02187v2)。§II-B可实现性条件、§III构造与限制是本轮主要新增理论邻居。
8. Chen, Anderson, *System Level Synthesis with State and Input Constraints*, arXiv 2019。[v2正文](https://arxiv.org/html/1903.07174v2)。本轮未独立核正式刊会。
9. Sanchez-Escalonilla, Reyes-Baez, Jayawardhana, *Total Energy Shaping with Neural Interconnection and Damping Assignment—Passivity Based Control*, first arXiv 2021。[作者稿](https://arxiv.org/abs/2112.12999)。仅局部筛选，不称全文精读。
10. Rosolia, Ames, *Multi-Rate Control Design Leveraging Control Barrier Functions and Model Predictive Control Policies*, first arXiv 2020。[v4正文](https://arxiv.org/html/2004.01761v4)。本轮未独立核正式刊会。
11. Amos et al., *Differentiable MPC for End-to-end Planning and Control*, NeurIPS 2018。[正式论文](https://papers.nips.cc/paper/2018/file/ba6d843eb4251a4526ce65d1807a9309-Paper.pdf)；[作者稿正文](https://arxiv.org/html/1810.13400v2)。
12. Klenske, Hennig, *Dual Control for Approximate Bayesian Reinforcement Learning*, first arXiv 2015；JMLR 17 (2016), article127。[正式全文](https://jmlr.csail.mit.edu/papers/volume17/15-162/15-162.pdf)。本轮读取导论/问题定位，未称全文。
13. Bansal et al., *Hamilton-Jacobi Reachability: A Brief Overview and Recent Advances*, 2017。[作者稿](https://arxiv.org/abs/1709.07523)。本轮为摘要筛选。
14. Badalamenti et al., *Configuration-Constrained Tube MPC for Difference-of-Convex Nonlinear Systems*, arXiv:2609.07809，2026-09预印本；作者页标明submitted to Automatica，不能视为录用。[作者稿正文](https://arxiv.org/html/2609.07809v1)。本轮读摘要/引言，尚未完整核验凸传播证明。
15. *Contraction-Based Model Predictive Control using Bilinear Koopman Realizations with Proportional Error Bounds*, arXiv:2609.34844，2026-09预印本。[作者稿正文](https://arxiv.org/html/2609.34844v1)。本轮核Assumption1、Proposition1–2、Theorem1与相关算法，不宣称全文。

## 7. 本轮研究决策

可以继续推进，不需要推翻已确认的主线。控制理论提供的是把问题做实的工具及强基线，不是等待拼装的模块清单。当前最有判别力的结果应是：在同一真实执行器与输入限制下，说明哪些任务响应能被实现；说明联合设计比充分整定的固定响应何时有收益；再用同自由度的直接经典优化排除换元和容量差异。若最后这一项达到同样结果，就把它作为解析最优参照，研究重点转向复杂反馈下实现该共同设计的学习方法和效率，而不篡改比较结论。
