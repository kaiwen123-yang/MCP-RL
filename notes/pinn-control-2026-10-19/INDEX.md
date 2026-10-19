# PINN 与模型控制和强化学习运动控制 入门调研 2022—2026

检索日期：2026-10-19（Asia/Shanghai）。范围：用户要求的两条线——PINN 与传统模型控制、PINN 与强化学习运动控制。按 Nature 系、Science 系、2025 年中科院升级版大类一区 Top、一区、二区 Top、二区、相关领域顶会组织。**这是独立专题的首轮筛选与关键段核查，不改变 MPC+RL 主库、MR 编号或当前研究主线，也不代表完成全文精读、证明核验或复现。**

本次登记 **14 篇主候选 + 9 篇综述、理论和概念边界补充**。[结构化元数据和逐篇阅读范围](../../literature/supplements/pinn-control-2026-10-19.json)保存精确题名、作者、DOI、版本和状态。部分 RL 期刊论文只核到出版社摘要/引言，必须保留“作者称 PINN、具体残差待核”的身份。

## 先理解两条线在解决什么问题

传统控制一侧至少有三种不同用法：

1. **预测模型**：用 PINN 近似受控动力学流映射，再由 MPC 求动作。代表为 PINC、RAMP-Net 和 T-RO 软体机器人工作。
2. **控制律求解**：把最优性方程、HJB 或 IDA-PBC 匹配方程转成神经网络训练问题。代表为 PDE 最优控制、Robust Neural IDA-PBC、AAAI 2026 策略迭代。
3. **证书学习**：学习 Lyapunov 函数并进一步验证，帮助判断稳定性或吸引域；不必直接生成动作。

RL 一侧也需要区分：

- **模型里放物理**：PINN 学环境动力学，给离线 RL 提供合成轨迹或改善 sim-to-real。
- **价值学习里放方程**：HJB/Eikonal 残差进入 critic/value 的训练损失。
- **结构、奖励或课程里放物理**：port-Hamiltonian 策略、模型控制器残差、物理课程学习。这是广义 physics-informed RL，不能仅凭名称当作经典 PINN。

本专题把明确微分方程残差训练的网络与广义物理结构网络分别记录。这个区分是为了理解机制，不是否定结构式物理学习的价值。

## 按要求的刊会层级查看

| 层级 | 本轮结果与处理 |
|---|---|
| Nature 系 | Scientific Reports 2024 有直接 PINN 最优控制工作；Nature Communications 2022 的 AI Pontryagin、Scientific Reports 2023 鱼形机器人 RL 属相关但不同机制。Scientific Reports 是 Nature Portfolio 期刊，2025 大类为3区非Top，保留刊名与实际层级。 |
| Science 系 | 本轮未确认 Science、Science Advances、Science Robotics 中满足这两类严格交集的直接方法论文。不等于不存在。Science Partner Journals 不当作 Science 子刊；泛学习动力学综述不计入直接方法数。 |
| 中科院一区 Top | T-RO 软机器人 PINN-MPC；JIII 机械臂 PICRL；ESWA 的 PIRL 综述。T-RO 多机器人 RL 另列广义结构方法。 |
| 中科院一区 | 2025 规则将大类一区期刊列为 Top，因此本口径下不另造“一区非Top”组。 |
| 中科院二区 Top | Journal of Computational Physics 的 PDE 最优控制；Ocean Engineering 的两篇 PINN+RL AUV 工作。 |
| 中科院二区 | Automatica、Neurocomputing、RA-L、Ain Shams Engineering Journal 的候选。 |
| 相关领域顶会主会 | ICRA 2023 RAMP-Net；NeurIPS 2025 Eik-HIQL；AAAI 2026 PINN策略迭代。NeurIPS 2022 损失理论及 ICLR 2024 广义物理RL另列。HJBPPO 是 ICML Workshop，不当作主会。 |

分区一律使用**2025年3月升级版大类**，具体数值来自公开转录、待官方库复核。2025 规则“大类1区期刊为Top期刊”见[丽水学院科研处转载的发布说明](https://kyc.lsu.edu.cn/2025/0327/c2834a353049/page.htm)。期刊分区用于执行检索优先级，不代替方法质量或个人阅读顺序。

## 第一类 PINN 与传统模型控制

### PINN-A01 Nature Portfolio 中的直接最优控制例子

**Jaemin Seo. Solving real-world optimization tasks using physics-informed neural computing. Scientific Reports, 2024, 14:202.**  
[原刊与 DOI 10.1038/s41598-023-49977-3](https://www.nature.com/articles/s41598-023-49977-3)

把动力学残差和任务目标合成优化问题，展示摆起、最短时间路径和航天器轨迹求解。适合直观看懂“物理方程约束下同时学状态与控制”。它主要是离线开环轨迹优化，不能称为 PINN-MPC 或 PINN+RL。正式发表2024，更早首稿未知。2025分区：综合性期刊3区非Top，因 Nature Portfolio 归属单列。

### PINN-A02 一区 Top 机器人闭环主读

**Habich et al. Generalizable and Fast Surrogates: Model Predictive Control of Articulated Soft Robots Using Physics-Informed Neural Networks. IEEE Transactions on Robotics, 42:619–636, 2026.**  
[正式机构记录与 DOI 10.1109/TRO.2025.3631818](https://research.uni-hannover.de/de/publications/generalizable-and-fast-surrogates-model-predictive-control-of-art/) · [作者全文](https://arxiv.org/html/2502.01916v3)

ODE残差训练的代理模型进入 NMPC，在气动关节软体机器人上进行实机闭环验证。正刊摘要为6组动态实验、47 Hz MPC、最高约467倍预测加速。值得看它如何把普通PINN改成可处理不同初值、动作和参数的模型。

离线超参搜索成本很高，参数需已知或另估计；快速预测不等于训练便宜或已经解决稳定性。首稿2025-02-04，online 2025-11-12，**正式卷年2026**，不能混用旧摘要中的实验数量。2025大类：计算机科学1区Top。

### PINN-A03 二区 Top 最优控制基础

**Mowlavi & Nabi. Optimal control of PDEs using physics-informed neural networks. Journal of Computational Physics, 473:111731, 2023.**  
[作者研究机构页面](https://www.merl.com/publications/TR2022-143) · [DOI 10.1016/j.jcp.2022.111731](https://doi.org/10.1016/j.jcp.2022.111731)

在已知PDE下联合训练状态与控制，和传统伴随优化比较；涉及流体等分布参数数值问题。重点看控制目标、物理损失和边界条件如何权衡。属于离线最优控制，非机器人实时闭环。**2021已有首稿、2022在线、2023卷年**。2025大类：物理与天体物理2区Top，不能以数学物理小类1区替代大类。

### PINN-A04 二区 PINN-MPC 的清楚入门

**Antonelo et al. Physics-informed neural nets for control of dynamical systems. Neurocomputing, 579:127419, 2024.**  
[原刊与 DOI 10.1016/j.neucom.2024.127419](https://www.sciencedirect.com/science/article/pii/S0925231224001905) · [2021初稿](https://arxiv.org/abs/2104.02556)

PINC把当前状态、控制输入、局部时间一起输入PINN，短段串接为长时预测，供MPC优化。验证Van der Pol、四水箱和电潜泵系统。它解释了普通PINN为何不能直接拿来做任意初值和动作下的控制。免真实标签仍需要已知物理方程。正式2024但**2021已公开**。2025大类：计算机科学2区非Top。

### PINN-A05 二区 物理控制理论如何进入网络

**Sanchez-Escalonilla Plaza, Zoboli & Jayawardhana. Robust Neural IDA-PBC: Passivity-based stabilization under approximations. Automatica, 193:113225, 2026.**  
[原刊与 DOI 10.1016/j.automatica.2026.113225](https://www.sciencedirect.com/science/article/pii/S0005109826004097)

PINN求解能量整形/无源控制的匹配与耗散条件，研究近似残差下的稳定性条件。这条线中，PINN帮助求经典控制设计里的困难方程。正刊验证为12状态欠驱动四旋翼仿真，不能写成实飞。2024已有初稿；当前已在线、编入**2026年11月卷期**。定理未独立逐步验算。2025大类：计算机科学2区非Top。

### PINN-A06 二区 负载变化下的实机MPC

**Li et al. PINN-Based Predictive Control Combined With Unknown Payload Identification for Robots With Prismatic Quasi-Direct-Drives. IEEE Robotics and Automation Letters, 10(11):11275–11282, 2025.**  
[DOI 10.1109/LRA.2025.3606361](https://doi.org/10.1109/LRA.2025.3606361) · [作者实验室发表记录](https://amdt.hkust-gz.edu.cn/publications/?search=%22tag%3A+motor-design%22)

将在线负载辨识与PINN预测模型结合，支撑棱柱准直驱机器人NMPC实机跟踪。适合看“已有方程参数会变化时，物理先验怎样更新”。online 2025-09-04；与题名不同的arXiv2503.06995可能有前身关系，尚未锁定，不双计。2025大类：计算机科学2区非Top。

### PINN-A07 顶会 ICRA 四旋翼例子

**Sanyal & Roy. RAMP-Net: A Robust Adaptive MPC for Quadrotors via Physics-informed Neural Network. ICRA 2023:1019–1025.**  
[IEEE与 DOI 10.1109/ICRA48891.2023.10161410](https://ieeexplore.ieee.org/document/10161410) · [2022首稿](https://arxiv.org/abs/2209.09025)

用ODE损失约束名义动力学，用数据拟合残余扰动，再供MPC预测。是很直接的“PINN+传统控制”例子。验证为四旋翼轨迹跟踪**仿真**；题名robust不替代已核的严格鲁棒控制保证。首稿2022，正式ICRA2023。

### PINN-A08 顶会 AAAI 随机最优控制

**Kim et al. Physics-Informed Approach for Exploratory Hamilton–Jacobi–Bellman Equations via Policy Iterations. AAAI 2026, 40(27):22609–22616.**  
[正式论文与 DOI 10.1609/aaai.v40i27.39421](https://ojs.aaai.org/index.php/AAAI/article/view/39421)

固定策略后用PINN求线性化评估PDE，再作soft策略改进；分析迭代、策略网络和PDE残差三类误差。有5D/10D LQR、摆和cartpole数值实验。它是已知模型下的随机最优控制，与试错式RL有联系，但不当作PINN+机器人RL实机证据。首稿2025，正式2026。

## 第二类 PINN 与强化学习运动控制

### PINN-B01 一区 Top 连续时间机械臂RL

**Wang et al. Physics-informed continuous-time reinforcement learning with data-driven approach for robotic arm manipulation. Journal of Industrial Information Integration, 49:101008, 2026.**  
[原刊与 DOI 10.1016/j.jii.2025.101008](https://www.sciencedirect.com/science/article/pii/S2452414X25002316)

PICRL把半离散HJB数据损失与力矩一致性物理损失结合，研究7自由度机械臂控制。PandaGym中比较连续和离散时间RL。适合看物理信息直接参与策略/价值学习，而不只是提供环境模型。**目前核到仿真和出版社引言，完整公式/定理待核；没有实机证据。**正式卷年2026，更早首稿未知。2025大类：计算机科学1区Top。

### PINN-B02 二区 Top PINN模型帮助离线RL

**Li et al. Motion control of autonomous underwater vehicle based on physics-informed offline reinforcement learning. Ocean Engineering, 313:119432, 2024.**  
[原刊与 DOI 10.1016/j.oceaneng.2024.119432](https://www.sciencedirect.com/science/article/pii/S0029801824027707)

PICOPO用PINN学AUV动力学，在模型中做保守离线策略优化，再部署；作者报告2000样本。验证包含仿真和**室内水池**AUV实测。实测只有姿态/深度信息、2Hz，不能扩大为海试或完整三维轨迹实测。PINN具体残差公式尚未核到。正式2024，更早首稿未知。2025大类：工程技术2区Top。

### PINN-B03 二区 Top 离线到在线的延伸

**Li et al. Hybrid offline-online reinforcement learning for AUV trajectory tracking: Physics-informed policy optimization with sample efficiency. Ocean Engineering, 356:125412, 2026.**  
[原刊与 DOI 10.1016/j.oceaneng.2026.125412](https://www.sciencedirect.com/science/article/pii/S0029801826012461)

PINN集成产生合成经验，离线保守策略优化后进行受约束在线微调。作者摘要报告400样本、仿真和field tests。**实测环境、平台、完整损失与理论条件尚未核；不能据摘要把400与前作2000解释为同条件五倍提升。**正式2026，更早首稿未知。2025大类：工程技术2区Top。

### PINN-B04 二区 双足sim-to-real

**Liu et al. A Lightweight Physics-Informed Neural Network for Sim-to-Real of Biped Robot. IEEE Robotics and Automation Letters, 11(1):386–393, 2026.**  
[IEEE原始记录与 DOI 10.1109/LRA.2025.3632659](https://ieeexplore.ieee.org/abstract/document/11248833)

BRUCE双足采用Isaac Gym中训练的RL步行策略，并以Euler–Lagrange/LSTM动力学预测器生成参考状态，结合前馈力矩和反馈执行；作者报告嵌入式1kHz模型预测和实机。**作者明确称PINN，但本轮未确认是显式ODE残差式还是主要结构式网络，故作为机制待核候选。**online2025-11-14，卷年2026，更早首稿未知。2025大类：计算机科学2区非Top。

### PINN-B05 二区 DDPG机械臂实机候选

**Fareh et al. Physics-informed reward shaped reinforcement learning control of a robot manipulator. Ain Shams Engineering Journal, 16(10):103595, 2025.**  
[原刊与 DOI 10.1016/j.asej.2025.103595](https://www.sciencedirect.com/science/article/pii/S2090447925003363)

PI-DDPG结合PINN、Lyapunov奖励塑形与自适应探索噪声；作者摘要报告4自由度机械臂仿真与实机。**PINN在网络/损失中的具体位置待全文核查；Lyapunov奖励本身不能替代闭环稳定性证明。**正式2025，更早首稿未知。2025大类：工程技术2区非Top。

### PINN-B06 顶会 最清楚的价值网络残差式结合

**Giammarino, Ni & Qureshi. Physics-informed Value Learner for Offline Goal-Conditioned Reinforcement Learning. NeurIPS 2025 主会.**  
[正式主会与 DOI 10.52202/085713-4599](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c92373e12f4b4993c5ec8e7f603264c9-Abstract-Conference.html) · [正式PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/c92373e12f4b4993c5ec8e7f603264c9-Paper-Conference.pdf)

Eik-HIQL在HIQL价值训练中加入Eikonal PDE梯度残差。正式全文已核31个OGBench任务、10seed；大迷宫导航和轨迹拼接收益明显，但部分antsoccer、cube交互任务退化。

Eikonal与HJB精确等价依赖特殊动力学等条件，对一般机器人是几何先验。全部为仿真；正式Table2采用训练中最佳评估，不能写成最终策略评估。首稿2025-09-08，正式2025。

## 相关但要分清机制的补充

| 编号与论文 | 刊会/年份 | 为什么值得看，以及不能混称什么 |
|---|---|---|
| PINN-R01 [A survey on physics informed reinforcement learning: Review and open problems](https://www.sciencedirect.com/science/article/pii/S0957417425017865) | ESWA 2025；一区Top；2023已有首稿 | 建立物理先验进入RL流程的分类地图；覆盖广义PIRL，不是每篇都使用PINN。 |
| PINN-C01 [AI Pontryagin or how artificial neural networks learn to control dynamical systems](https://www.nature.com/articles/s41467-021-27590-0) | Nature Communications 2022 | neural ODE/可微编程生成控制；与PINN最优控制相邻，本轮不作严格残差PINN代表。 |
| PINN-C02 [Physics-informed reinforcement learning for motion control of a fish-like swimming robot](https://www.nature.com/articles/s41598-023-36399-4) | Scientific Reports 2023；3区非Top | 低保真物理代理+课程/迁移+DPG；游动仿真，无PINN残差证据。 |
| PINN-C03 [Physics-Informed Multiagent Reinforcement Learning for Distributed Multirobot Problems](https://doi.org/10.1109/TRO.2025.3582836) | T-RO 2025；一区Top | port-Hamiltonian结构+self-attention+SAC，Robotarium实测；属于结构式物理RL，非已核残差PINN。最早arXiv日期2023-12-30。 |
| PINN-C04 [Physics-Regulated Deep Reinforcement Learning: Invariant Embeddings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1018ac24d7555bfab7b183afbd845d93-Abstract-Conference.html) | ICLR 2024主会；初稿2023 | 模型控制器+DDPG残差、安全奖励与网络结构编辑，含A1实机；不是常规PINN，安全结论有条件。 |
| PINN-D01 [Is L2 Physics-Informed Loss Always Suitable for Training Physics-Informed Neural Network?](https://papers.nips.cc/paper_files/paper/2022/file/374050dc3f211267bd6bf0ea24eae184-Paper-Conference.pdf) | NeurIPS 2022主会 | 特定HJB类别中小L2残差未必保证解正确；适合理解PINN理论风险，不能泛化为所有PINN无效。 |
| PINN-D02 [Physics-informed neural network Lyapunov functions: PDE characterization, learning, and verification](https://doi.org/10.1016/j.automatica.2025.112193) | Automatica 2025；二区；初稿2023 | PINN学Lyapunov/Zubov函数，再SMT验证；证书与动作策略是不同对象，不自动保证所有PINN-MPC稳定。 |
| PINN-D03 [Bridging Physics-Informed Neural Networks with Reinforcement Learning: Hamilton-Jacobi-Bellman Proximal Policy Optimization (HJBPPO)](https://arxiv.org/abs/2302.00237) | ICML 2023 Workshop，**非主会** | HJB残差正则PPO critic，未知动力学有限差分估计；MuJoCo仿真。机制直接，但刊会等级必须如实标注。 |
| PINN-D04 [A comparison of strategies to embed physics-informed neural networks in nonlinear model predictive control formulations solved via direct transcription](https://www.sciencedirect.com/science/article/pii/S0098135425001097) | Computers & Chemical Engineering 2025；本轮未核分区 | 对比代理模型进入NMPC优化器的方式；神经网络前向快不保证优化问题整体更快，适合作为计算成本对照。 |

### 候选之外的排除记录

- [Physics-Informed Neural Networks to Model and Control Robots: A Theoretical and Experimental Investigation](https://doi.org/10.1002/aisy.202300385)，Advanced Intelligent Systems 2024：高相关，但核心为Lagrangian/Hamiltonian结构扩展，且2025大类3区非Top，不放用户一区/二区主组。
- [An explicit model predictive control framework based on physics-informed neural networks](https://doi.org/10.1016/j.jprocont.2026.103634)，Journal of Process Control 2026：直接学习显式近似控制器，有软约束/水箱仿真；该刊2025大类3区非Top，留作后续拓展。
- [QuietWalk](https://arxiv.org/html/2604.23702v1)：2026预印本，PINN估计接触力辅助RL奖励，尚未核正式录用，不占正式顶会/期刊席位。
- [Neural Policy Iteration 的2026更新版](https://arxiv.org/abs/2508.01718)：题名与2025初稿已有变化，聚合站的ICLR身份未独立确认，本轮不登记为ICLR正式论文。不要与已核AAAI的2508.01720混为一篇。
- 2021正式发表的physics-informed Dyna-style RL工作超出时间要求，未纳入。

## 2025分区来源对照

下表是**公开转录**，待官方分区库复核；不是JCR四分位，也不采用2026新锐表。

| 期刊 | 大类/分区/Top | 核验来源 |
|---|---|---|
| IEEE Transactions on Robotics | 计算机科学1区Top | [2025表](https://www.iikx.com/sci/technology/13051.html) |
| Journal of Industrial Information Integration | 计算机科学1区Top | [2025表](https://www.xueshu.com.cn/sci/49258.html) |
| Expert Systems with Applications | 计算机科学1区Top | [2025表](https://journalcompass.com/journals/expert-systems-with-applications-09574174.html) |
| Journal of Computational Physics | 物理与天体物理2区Top | [2025表](https://hy.iikx.com/sci/physics/13992.html) |
| Ocean Engineering | 工程技术2区Top | [2025表](https://www.zazhi.com.cn/sci/218708.html) |
| Automatica | 计算机科学2区非Top | [2025表](https://www.sci365.cn/journals/automatica-00051098/) |
| Neurocomputing | 计算机科学2区非Top | [2025表](https://www.sci365.cn/journals/neurocomputing-09252312/) |
| IEEE Robotics and Automation Letters | 计算机科学2区非Top | [2025表](https://www.sci365.cn/journals/ieee-robotics-and-automation-letters-23773766/) |
| Ain Shams Engineering Journal | 工程技术2区非Top | [2025表](https://www.iikx.com/sci/technology/33890.html) |
| Scientific Reports | 综合性期刊3区非Top | [2025表](https://journalcompass.com/journals/scientific-reports-20452322.html) |
| Journal of Process Control | 计算机科学3区非Top | [2025表](https://www.iikx.com/sci/technology/14703.html) |

## 适合先了解的阅读顺序

建议先读六篇形成概念地图，而不是按刊物名气逐篇读完：

1. **PINC（A04）**：理解PINN如何成为受控系统的预测器。
2. **T-RO软体机器人MPC（A02）**：看预测器如何落到真实机器人闭环，以及在线收益与离线成本。
3. **Robust Neural IDA-PBC（A05）**：看PINN如何参与经典控制律设计，和只做动力学拟合有什么不同。
4. **Eik-HIQL（B06）**：看PDE残差直接进入价值学习；同时阅读交互任务的负结果。
5. **PICOPO（B02）**：看PINN模型、合成数据、离线策略优化与实测的因果链。
6. **PICRL（B01）**：看连续时间RL如何组织HJB与物理一致性；目前需继续取得完整方法。

ESWA综述R01可在前后穿插，建立广义PIRL分类。RA-L双足B04很贴近运动控制，但先核清PINN形式再决定深入优先级。

## 后续问答 研究定位与机器狗机械臂先例

同日补核：PINN更适合定位为物理信息学习/数值求解技术，通常嵌入MPC、非线性反馈或RL；本轮判断其为活跃交叉研究分支，不声称已形成与MPC等并列的成熟通用机器人控制体系。该判断不是产业份额统计。[足式MPC综述](https://doi.org/10.1080/01691864.2023.2168134)、[机器人PINN综述预印本](https://doi.org/10.2139/ssrn.5125543)提供分类背景。

- **Kirin四足实机**：[Physics-informed Neural Network Predictive Control for Quadruped Locomotion，arXiv2503.06995v1](https://arxiv.org/html/2503.06995v1)，2025预印本。核到Eq23–24的拟合/物理残差和IV节实机；主实验为50kg负载前向0.2m/s小跑。辨识阶段才是垂直升降。不是只有仿真或静态姿态测试，也不外推为复杂地形、高速跳跃能力。与RA-L A06的精确版本关系仍待原始来源确认，不计为第二篇独立正式论文。
- **Franka Panda实机**：[Liu等2024作者全文](https://arxiv.org/html/2305.05375v3)，Sec6.2/Fig17–20确认关节周期轨迹与末端螺旋轨迹跟踪；Eq3–8确认Lagrangian/Hamiltonian结构扩展及耗散、驱动建模。作为广义结构式物理学习先例，不改称经典配点残差PINN。
- **Unitree A1实机**：[Phy-DRL ICLR2024正式全文](https://proceedings.iclr.cc/paper_files/paper/2024/file/1018ac24d7555bfab7b183afbd845d93-Paper-Conference.pdf)，Sec7.2、Appendix L、致谢部署说明及演示链接支持实机先例；是物理结构/模型控制器加RL，不是严格残差PINN。
- **4DOF机械臂**：B05出版社摘要明确仿真和实机，但具体PINN残差仍未核；B01的PandaGym结果仍是仿真，不混同Franka实机。

上述为既有候选及版本线索的问答补核，不改变23项登记数、14篇主候选数量或阅读状态。

## 同日轻量创新空间筛查

用户仅要求“大概看看”，本轮采用问题、失败分析、简单替代三种视角做定向查新，不扩成完整选题或实验计划。结论是仍有研究空间，但没有确认任何候选为文献空白；不以“PINN＋MPC/RL＋新平台”作为独立贡献，不改变已有主线，也不启动训练。

最重要的新近邻是[Physics-informed Goal-Conditioned Reinforcement Learning under Hybrid Contact Dynamics](https://arxiv.org/abs/2605.30503)，2026-05-28预印本。作者已经处理Eikonal价值正则在接触任务中的失配，提出可控方向正则与分层表示，并有UR5e抓放实测。作者明确留下力/执行器约束、顺应性与阻抗适应等问题。因此“接触感知PINN/RL”或“给接触正则补实机”不能作为空白主张。[全文方法、实机与局限](https://arxiv.org/html/2605.30503v1)已做关键段核查，未复现或独立验算。

| 候选问题（阅读者推断） | 可能有价值的具体推进 | 必须面对的强替代与停止条件 |
|---|---|---|
| 未知刚度/摩擦下，自由运动—接触—脱离如何限制真实力峰值并完成操作 | 物理约束进入学习/执行，使力限制与任务完成同时改善；不能仅加一个安全过滤器就归功于PINN | 阻抗/导纳＋在线辨识、自适应接触隐式MPC、残差RL＋相同安全层。若简单方法已消除问题或收益全来自安全层，不支持PINN机制 |
| 物理先验失准后，如何用有限交互恢复有效控制 | 让可用物理与失配项的判断真正影响动作/学习，并解释适应代价与任务后果 | 参数辨识、名义模型＋小MLP/GP残差、同预算MBRL/PPO。只调物理loss权重或加ensemble不够 |
| 哪些预测误差会改变动作选择，如何有针对性地修正 | 在可解释失配下研究动作排序/敏感度/任务后果，同时保持模型对真实数据的校准 | 普通辨识、任务导向模型、解析策略梯度与同信息PO/TO。与已有任务加权/共同学习高度相邻，具体机制未明确前暂缓；不能以PINN重新包装旧负结果 |

已核近邻进一步压缩这些候选的泛化表述：

- [Adaptive Contact-Implicit Model Predictive Control with Online Residual Learning](https://arxiv.org/abs/2310.09893)，ICRA2024、初稿2023：接触互补模型在线适应与Franka实机，未知接触＋在线残差＋MPC已有先例。
- [Fast Contact-Implicit Model Predictive Control](https://doi.org/10.1109/TRO.2024.3351554)，T-RO2024、初稿2021：接触模式/时刻/力优化及四足实机，非PINN强对照。
- [Physics-informed adaptive Kalman filter for contact force estimation in industrial robots considering model uncertainty](https://doi.org/10.1016/j.rcim.2026.103292)，RCIM2026：PINN动力学/摩擦与接触力观测，不能把“估计接触力”当空白。
- [Correcting model misspecification in physics-informed neural networks (PINNs)](https://doi.org/10.1016/j.jcp.2024.112918)，JCP2024、初稿2023：残差DNN修正错误物理并表达不确定性，非机器人控制收益的自动证明。
- [Where-to-Learn](https://arxiv.org/abs/2603.27317)，RA-L2026，DOI10.1109/LRA.2026.3678143：解析模型梯度引导PPO探索与机器人验证，属于广义物理学习，不能将“模型提供任务信号”泛称新机制。
- [Co-Learning Port-Hamiltonian Systems and Optimal Energy-Shaping Control](https://arxiv.org/abs/2604.26172)，2026预印本：策略感知数据采集、模型与控制器交替共同学习已有先例。

以上为增量查新来源，未占原23条入门书目名额。当前判断只支持继续核第一/第二类具体问题；第三类更贴近既有MPC规划/RL执行科学问题，但尚未形成新机制，保留已有负结果。既有方法的存在不否定后续问题价值，但需要具体方法差异与相同数据、信息和总计算预算下的完整任务证据。

## 检索执行范围

采用出版社/IEEE/正式会议论文集、作者机构库、arXiv版本记录与Crossref书目交叉核对，查询PINN+MPC、optimal control、passivity、Lyapunov、offline RL、AUV、biped、HJB/Eikonal等组合，并单独查Nature/Science域和顶会记录。没有宣称进行WoS/Scopus穷尽系统检索；学术检索专用MCP本会话未挂载，使用可用网页工具与公开原始来源。

2022—2026采用正式刊会年份；最早已核公开年份与online时间另列。未知首稿保留未知。摘要证据、方法关键段、已取得全文与完成全篇精读不是同一状态。无新训练、仿真、实机或算法修改；本轮提出的是阅读顺序，不是确定创新方向。

