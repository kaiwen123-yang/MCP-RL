# 2026 TAC / Automatica 定向补查：控制如何参与学习

日期：2026-10-16（Asia/Shanghai）。这是针对既定共同设计问题的定向检索及关键段核读，不是全年穷尽综述，也不是下面10篇全部全文精读。三篇指定论文的全文范围另见本目录逐篇笔记。

## 检索与判定方法

先核项目当前状态、既有全文笔记及前序对话，再检索出版社、作者机构库、arXiv版本历史。TAC另通过Crossref的0018-9286期刊接口查询2025—2026记录，关键词为`reinforcement learning policy model predictive neural`，返回相关性排序的100条；随后按卷71、2026正式记录及与当前问题的联系筛选。该100条是检索返回量，不是新增阅读量，不等于完整全年目录。DBLP整卷入口受大小/反机器人限制，没有以该入口宣称完成全卷检查。

以下10项保留：TAC五项、Automatica五项；其中MR-ARL和耗散神经控制已有项目阅读，本次只核书目/相关边界，不计新增精读。Stable Koopman属于指定三篇，未重复算入这10项。正式年份、最早已核公开记录、实际阅读版本分开；Crossref `created`只表示DOI登记时间，不当成online发表日。主100篇v2库和MR编号不改。结构化记录见[补充登记](../../literature/supplements/control-frontier-2026-10-16.json)。

## TAC：五项直接机制或理论近邻

| 文献 | 正式记录与版本 | 本次读取范围 | 对主线的作用 |
|---|---|---|---|
| Neural Port-Hamiltonian Models for Nonlinear Distributed Control: An Unconstrained Parameterization Approach | TAC 71(9), 6100–6115, 2026；[DOI](https://doi.org/10.1109/TAC.2026.3683628)；同题arXiv始于2024-11-15，文内说明继承较短先行版 | [arXiv v2](https://arxiv.org/html/2411.10096v2) §II–V、定理3/4及相关训练说明定向核读；未核正式版全文 | 控制理论进入反馈网络的结构，学习仍选择实际动态行为 |
| Combined Learning of Linear Parameter-Varying Models and Robust Control Invariant Sets | TAC 2026已登记，1–14；卷期/精确online日待核；[DOI](https://doi.org/10.1109/TAC.2026.3705297)；arXiv始于2024-11-27 | [v2](https://arxiv.org/html/2411.18166v2) §2、§3.2–3.4、算法1、§4.4；未核正式版全文 | 模型准确度与下游控制可行性可联合优化，不能再宣称“控制导向模型”本身新颖 |
| Safe Learning Control with Optimality and Stability Guarantees | TAC Early Access 2026-06-25；[DOI](https://doi.org/10.1109/TAC.2026.3707508)；同题arXiv始于2025-01-26 | [v2](https://arxiv.org/html/2501.15373v2) 问题定位、§IV-B式(30)–(34)、引理4及其证明；未全文核验其他定理 | 已有安全/任务梯度冲突的定量处理；简单梯度投影不能成为我们的新增贡献 |
| Bridging Model Reference Adaptive Control and Data Informativity | TAC 2026已登记，1–16；卷期/精确online日待核；[DOI](https://doi.org/10.1109/TAC.2026.3724422)；arXiv始于2025-02-28 | [v3](https://arxiv.org/html/2502.21091v3) §IV的引理3、命题1、定理1及§V的在线数据条件；未全文证明核验 | 数据能否支撑特定响应，实现所需信息可能少于完整辨识 |
| MR-ARL: Model Reference Adaptive Reinforcement Learning for Robustly Stable On-Policy Data-Driven LQR | TAC 71(2), 1129–1144, 2026；[DOI](https://doi.org/10.1109/TAC.2025.3611155)；已有CDC2023前身和2024扩展稿 | 复核[作者机构记录](https://cris.unibo.it/handle/11585/1042541)并沿用项目已读24页v1；正式TAC全文本次未取得 | 已有变化参考、同步反馈、学习系统稳定性；不能以“参考也学习”宣称创新 |

**神经port-Hamiltonian的启示与边界。** 式(9)把控制器写成有内部状态的能量/阻尼互连。定理3通过阻尼与耦合矩阵的关系限制控制器输入输出增益；定理4另要求Hamiltonian Hessian一致正定且有界。学习仍决定能量形状和反馈行为。真实闭环保证还需被控对象的被动性或小增益条件，离散实现需保持耗散性质；加饱和、随机探索和任意变能力命令均须重新检查。本文用BPTT求有限时域控制目标，不能直接称为PPO算法。[作者原文](https://arxiv.org/html/2411.10096v2)

**qLPV联合学习的启示与边界。** 式(29)将可行不变集内的跟踪目标引入模型学习，式(30)兼顾拟合与控制目标。模型/不确定集合必须真正覆盖对象误差；原文承认扰动膨胀系数难以预先保证、初始RCI可能不可行、多面体维数扩展困难。它学习对象模型及控制可行域，没有学习一个改变对象闭环响应的RL执行器。我们的区别不能只是增添一项控制损失，而应体现策略改变后可实现行为随之改变。[关键原文](https://arxiv.org/html/2411.18166v2)

**安全梯度与数据可实现性。** Safe Learning明确分析安全动作与最优动作的梯度夹角；相应性能结论使用理想价值函数等前提，不能直接继承给PPO近似critic。Informativity论文则在离散线性、匹配方程框架给出数据像空间包含条件；它不是任意神经闭环的有限数据证书。这两条分别约束我们“只改梯度方向”和“采样残差小就声称实现”的表述。[安全梯度原文](https://arxiv.org/html/2501.15373v2)、[信息性原文](https://arxiv.org/html/2502.21091v3)

## Automatica：五项直接机制或理论近邻

| 文献 | 正式记录与版本 | 本次读取范围 | 对主线的作用 |
|---|---|---|---|
| Stability-certified on-policy data-driven LQR via recursive learning and policy gradient | 191, 113101, September2026；[DOI](https://doi.org/10.1016/j.automatica.2026.113101)；2024-03-08扩展稿，正式文首页注明CDC2023前身 | [v3](https://arxiv.org/html/2403.05367v3) §2.2、算法1、§3的激励/局部条件、§4证明结构；核正式首页，未通读正式全文 | 将学习与物理控制作为同一个动态互连分析 |
| Reinforcement learning for control with probabilistic stability guarantee: A finite-sample approach | 188, 112964, June2026；[DOI](https://doi.org/10.1016/j.automatica.2026.112964)；本次核到2026 arXiv，另有高度相近ICLR2021投稿，继承/增量待核 | [v1](https://arxiv.org/html/2603.00043v1) 假设1–3、引理2/3.2、定理3.4及证明主要步骤；不是完整定理认证 | 直接改变策略学习，且严格区分独立轨迹数与每轨迹长度 |
| Policy iteration with discounted cost: Near-optimality bounds and stability guarantees | 192, 113158, October2026；[DOI](https://doi.org/10.1016/j.automatica.2026.113158)；继承CDC2024/2024-03-29稿 | [出版社](https://www.sciencedirect.com/science/article/pii/S0005109826003420) 摘要、引言、条件与结论定位；未阅读全文 | 折扣、有限迭代性能和稳定性不能混为一谈；长程任务理论近邻 |
| MAKO: Meta-Adaptive Koopman Operators for learning-based model predictive control of parametrically uncertain nonlinear systems | 186, 112827, April2026；[DOI](https://doi.org/10.1016/j.automatica.2026.112827)；最早公开年份待核 | [出版社](https://doi.org/10.1016/j.automatica.2026.112827) 摘要/引言/贡献定位；未阅读全文 | 变化对象上的快速模型适应，不等于RL学习新反馈能力 |
| Synthesizing neural network controllers with closed-loop dissipativity guarantees | 193, 113185，分配November2026未来卷期；[DOI](https://doi.org/10.1016/j.automatica.2026.113185)；已上线记录可见，精确online日待核 | 本次复核出版社；此前17页作者稿阅读见[旧专题](../advanced-control-2026-10-16/theory-T04-dissipative-ppo.md) | 已有PPO、神经控制器、硬耗散条件及参数投影；不能重复包装 |

**Relearn LQR。** 未知A/B的在线递归辨识与反馈增益策略梯度同步进行。稳定性对象包含物理系统、辨识和优化状态，使用激励、足够小步长和局部/平均系统分析。其学习真实改变反馈增益，提供了“理论约束学习过程”的实例；但LQR、激励和局部条件不能直接扩成接触机器人PPO保证。[作者原文](https://arxiv.org/html/2403.05367v3)

**有限样本稳定RL。** 定理依赖稳态分布、初始覆盖、混合速率和Lyapunov双边界；独立单位是整条轨迹，不能将同一轨迹的MT个相关步都当独立样本。原稿式(25)与式(27)/证明中所需正裕量存在系数需核对的问题；本轮不直接采用其置信公式，也不把与裁剪状态范数有关的结论升级为全域真实二阶矩保证。学习到的策略/证书再使用同一训练数据验收还需处理数据依赖。另发现[ICLR2021投稿](https://openreview.net/pdf?id=QfEssgaXpm)已有近似题名、有限轨迹与REINFORCE机制，在版本继承查明前，**不能宣称该方法首次提出于2026**。[2026作者稿](https://arxiv.org/html/2603.00043v1)

**折扣PI与MAKO。** 前者明确研究精确PI，近似执行留作未来工作；后者调整预测模型适应参数变化。它们分别为长期性能分析和模型适应提供参照，均未在本次已核范围内提供我们要求的“任务塑造实际反馈能力，再进入同一响应规划”的完整机制。[PI出版社原文](https://www.sciencedirect.com/science/article/pii/S0005109826003420)、[MAKO出版社原文](https://doi.org/10.1016/j.automatica.2026.112827)

## 排除与未覆盖

- 搜到的JAS、TASE、IFAC Journal of Systems and Control、Computers & Chemical Engineering条目不冒充TAC/Automatica。
- 博弈编码、图学习、纯优化器加速、与当前反馈能力无直接联系的应用论文不为凑2026篇数纳入重点。
- 对尚无卷期的TAC条目保留2026正式登记和online日未知；不伪造卷号页码。Automatica未来11月卷期单列。
- 这次没有为补充条目重新核官方中科院2025大类分区；没有向主库追加经典名额，没有用新卷期重写方法首次年份。
- 未阅读全文的条目不能用于断言“作者没有做某事”；表中缺口只限本次可核范围，机制定稿前仍需对最贴近的原文逐项补齐。

## 对主线的实际收敛

本次文献支持把控制理论用于**可学习反馈结构、任务相关响应的实现关系、学习更新与物理执行的耦合分析**。它们不能单独证明RL必要，也不能保证任何结构化神经策略优于充分整定的经典反馈。

下一步最关键的数学对象是：策略改变实际段内响应的方向，是否能实现规划提出的响应改进；纠偏和输入资源是否仍允许该改进；完整任务是否因此获益。纯稳定参数化、单独拟合更准的模型、奖励增加一项Lyapunov残差，都已有先例。本轮机制判断见后续[共同设计报告](../../docs/control-informed-coupling-2026-10-16.md)。
