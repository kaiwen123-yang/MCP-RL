# 新增来源账本：六条算法研究主线

日期：2026-11-26，Asia/Shanghai。新增指相对上一版45条专题来源账本，不代表此前完全未见过题目。Science Raibo此前仅摘要，本轮首次方法深读。

| 专题 | 来源数 | 主要方法及相关实验深读 | 限定/定向阅读 |
|---|---:|---:|---:|
| 材料 | 10 | 6 | 4 |
| 步态 | 11 | 7 | 4 |
| 学习 | 12 | 9 | 3 |
| MPC/WBC | 11 | 9 | 2 |
| 本轮合计 | 44 | 31 | 13 |

上一版45条与本版44条合计89条记录。31篇深读不等于全部附录逐字核验；13篇限定阅读中包含仅摘要/元数据条目。官方源码核验限列明文件和版本，全部未运行复现。2025中科院分区未取得可靠原表，未自行标注。

五模块各三条研究路线和两个联合方向是我们的设计，不是已获验证的原创性或性能结论。审阅修正见 [design_review.md](design_review.md)。

## 材料

方法与实验分析见 [material.md](material.md)；版本、章节、代码与边界见 [material_sources.json](material_sources.json)。

| ID | 原始来源 | 发表/阅读边界 |
|---|---|---|
| BM01 | [Fully data-driven inverse characterization of heterogeneous materials with hyper-network neural ODEs](https://www.nature.com/articles/s41524-026-02027-8) | npj Computational Materials 12, 165；主要方法/理论及实验深读 |
| BM02 | [Beyond empirical models: Discovering constitutive laws in solids with graph-based equation discovery](https://www.science.org/doi/10.1126/sciadv.aec0989) | Science Advances 12(37), eaec0989；指定章节方法/实验阅读 |
| BM03 | [HyperCAN: Hypernetwork-driven deep parameterized constitutive models for metamaterials](https://www.sciencedirect.com/science/article/pii/S2352431624001238) | Extreme Mechanics Letters 72, 102243；主要方法/理论及实验深读 |
| BM04 | [Microstructure-Conditioned Surrogate Models for Graded Multiscale Optimization of Mycelium Composites](https://arxiv.org/abs/2607.13688) | 2026 preprint; journal acceptance not verified；指定章节方法/实验阅读 |
| BM05 | [History-aware neural operator: Robust data-driven constitutive modeling of path-dependent materials](https://www.sciencedirect.com/science/article/pii/S0045782525006309) | Computer Methods in Applied Mechanics and Engineering 447, 118358；主要方法/理论及实验深读 |
| BM06 | [Learning Memory And Material Dependent Constitutive Laws](https://smai-jcm.centre-mersenne.org/articles/10.5802/smai-jcm.148/) | The SMAI Journal of computational mathematics 12, 219-267；主要方法/理论及实验深读 |
| BM07 | [Monotone peridynamic neural operator for nonlinear material modeling with conditionally unique solutions](https://www.sciencedirect.com/science/article/pii/S0045782526000666) | Computer Methods in Applied Mechanics and Engineering 453, 118792；主要方法/理论及实验深读 |
| BM08 | [Automatically Differentiable Model Updating (ADiMU): Conventional, hybrid, and neural network material model discovery including history-dependency](https://www.sciencedirect.com/science/article/pii/S0022509625003825) | Journal of the Mechanics and Physics of Solids 206, Part B, 106408；指定章节方法/实验阅读 |
| BM09 | [Thermodynamically consistent neural constitutive models for elastoplasticity via eigen-representations](https://arxiv.org/abs/2611.13162) | 2026 preprint; journal acceptance not verified；主要方法/理论及实验深读 |
| BM10 | [Learning the Constitutive Behavior of Materials via Neural Operators and Causal Attention: Case Studies in Plasticity and Damage](https://arxiv.org/abs/2609.02194) | 2026 preprint; journal acceptance not verified；指定章节方法/实验阅读 |

## 步态

方法与实验分析见 [gait.md](gait.md)；版本、章节、代码与边界见 [gait_sources.json](gait_sources.json)。

| ID | 原始来源 | 发表/阅读边界 |
|---|---|---|
| G3-01 | [High-speed control and navigation for quadrupedal robots on complex and discrete terrain](https://doi.org/10.1126/scirobotics.ads6192) | Science Robotics 10(102): eads6192；major_method_and_experiment_deep_read |
| G3-02 | [Exciting families of passive gaits in an elastic quadruped via natural motion manifold control](https://journals.sagepub.com/doi/10.1177/02783649251347305) | The International Journal of Robotics Research 45(2):233–258；major_method_and_experiment_deep_read |
| G3-03 | [Motion Priors Reimagined: Adapting Flat-Terrain Skills for Complex Quadruped Mobility](https://proceedings.mlr.press/v305/zhang25j.html) | CoRL 2025; PMLR 305:3762–3777；major_method_and_experiment_deep_read |
| G3-04 | [ManyQuadrupeds: Learning a Single Locomotion Policy for Diverse Quadruped Robots](https://arxiv.org/abs/2310.10486) | ICRA 2024；major_method_and_experiment_deep_read |
| G3-05 | [Learning Quadruped Walking from Seconds of Demonstration](https://arxiv.org/abs/2603.06961) | ICRA 2026；major_method_and_experiment_deep_read |
| G3-06 | [PROMO: Preference-conditioned Multi-Objective Reinforcement Learning for Quadrupedal Robots](https://arxiv.org/abs/2610.01260) | arXiv; author project lists HART Workshop at TAROS 2026；major_method_and_experiment_deep_read |
| G3-07 | [LoComposition: Terrain-Adaptive Energy-Efficient Quadruped Locomotion without Gait Priors](https://arxiv.org/abs/2606.15896) | 作者页面标 CoRL 2026；读取 arXiv v2；major_method_and_experiment_deep_read |
| G3-08 | [SG-CPG: Severity-Gated Central Pattern Generators for Adaptive Quadruped Locomotion under Continuous Actuator Degradation](https://arxiv.org/abs/2609.25687) | arXiv；targeted_method_and_experiment_read |
| G3-09 | [Learning Fault-Tolerant Locomotion with Adaptive Gait Timing](https://arxiv.org/abs/2608.07328) | arXiv；targeted_method_and_experiment_read |
| G3-10 | [Versatile Loco-Manipulation through Flexible Interlimb Coordination](https://proceedings.mlr.press/v305/zhu25a.html) | CoRL 2025; PMLR 305:610–632；targeted_method_and_experiment_read |
| G3-11 | [GRoQ-LoCO: Generalist and Robot-agnostic Quadruped Locomotion Control using Offline Datasets](https://arxiv.org/html/2505.10973v3) | arXiv；targeted_method_and_experiment_read |

## MPC/WBC

方法与实验分析见 [control.md](control.md)；版本、章节、代码与边界见 [control_sources.json](control_sources.json)。

| ID | 原始来源 | 发表/阅读边界 |
|---|---|---|
| C3-01 | [FLAIR: Getting robots back on track by reconstituting control in unexpected situations with online learning](https://www.nature.com/articles/s41467-026-70256-y) | Nature Communications 17:3715, 2026；主要方法/实验与限制深读 |
| C3-02 | [A general soft robotic controller inspired by neuronal structural and plastic synapses that adapts to diverse arms, tasks, and perturbations](https://pmc.ncbi.nlm.nih.gov/articles/PMC12778046/) | Science Advances 12(2):eaea3712, 2026；主要方法/实验与限制深读 |
| C3-03 | [Reference-Free Sampling-Based Model Predictive Control](https://arxiv.org/html/2511.19204v2) | ICRA 2026; arXiv 2511.19204v2；主要方法/实验与限制深读 |
| C3-04 | [Variance-Reduced Model Predictive Path Integral via Quadratic Model Approximation](https://roboticsproceedings.org/rss22/p108.html) | RSS 2026, DOI 10.15607/RSS.2026.XXII.108；主要方法/实验与限制深读 |
| C3-05 | [Sampling-based Model Predictive Control Using Trust Regions](https://arxiv.org/abs/2605.07801) | arXiv 2605.07801v2, September 2026, preprint；主要方法/实验与限制深读 |
| C3-06 | [LLA-MPPI: Rapidly Adaptive Whole-body Control of Legged Robots with GPU-Accelerated Parallel Simulations](https://arxiv.org/abs/2610.10465) | arXiv 2610.10465v1, October 2026, preprint；主要方法/实验与限制深读 |
| C3-07 | [Beyond Pure Sampling: Hybrid Optimization Mechanisms for Non-Convex Model Predictive Control](https://arxiv.org/abs/2606.00737) | arXiv 2606.00737v1, May 2026, preprint；主要方法/实验与限制深读 |
| C3-08 | [Robot Tracking Control With Natural Task-Space Decoupling](https://elib.dlr.de/221903/) | IEEE TRO, online 2025, volume42:734-749, DOI10.1109/TRO.2025.3647777；主要方法/实验与限制深读 |
| C3-09 | [A Novel Safety-Aware Energy Tank Formulation Based on Control Barrier Functions](https://elib.dlr.de/208540/) | IEEE RA-L9(6):5206-5213, 2024, DOI10.1109/LRA.2024.3389556；主要方法/实验与限制深读 |
| C3-10 | [Task-space tracking of robot manipulators via internal model principle approach](https://research.rug.nl/en/publications/task-space-tracking-of-robot-manipulators-via-internal-model-prin/) | Automatica174:112104, 2025, DOI10.1016/j.automatica.2024.112104；仅摘要与元数据；全文403未读 |
| C3-11 | [Singularity-Robust Prioritized Whole-Body Tracking and Interaction Control With Smooth Task Transitions](https://elib.dlr.de/204082/) | ICRA2024, DOI10.1109/ICRA57147.2024.10610961；引言与方法概要定向阅读 |

## 学习

方法与实验分析见 [learning.md](learning.md)；版本、章节、代码与边界见 [learning_sources.json](learning_sources.json)。

| ID | 原始来源 | 发表/阅读边界 |
|---|---|---|
| B01 | [Simple Policy Optimization](https://proceedings.mlr.press/v267/xie25m.html) | ICML 2025 / PMLR 267；主要方法与实验深读 |
| B02 | [Absolute Policy Optimization: Enhancing Lower Probability Bound of Performance with High Confidence](https://proceedings.mlr.press/v235/zhao24i.html) | ICML 2024 / PMLR 235；主要方法与实验深读 |
| B03 | [Model-based Reinforcement Learning with Scalable Composite Policy Gradient Estimators](https://proceedings.mlr.press/v202/parmas23a.html) | ICML 2023 / PMLR 202；主要方法与实验深读 |
| B04 | [Policy Optimization in Hybrid Discrete-Continuous Action Spaces via Mixed Gradients](https://arxiv.org/abs/2605.14297) | arXiv v1; official NeurIPS 2026 manuscript available；主要方法与实验深读 |
| B05 | [Rectified Robust Policy Optimization for Model-Uncertain Constrained Reinforcement Learning without Strong Duality](https://openreview.net/forum?id=7l63xwAgAW) | Transactions on Machine Learning Research；主要方法与实验深读 |
| B06 | [ContractionPPO: Certified Reinforcement Learning via Differentiable Contraction Layers](https://contractionppo.github.io/) | arXiv; Caltech publication list says RA-L to appear 2026；主要方法与实验深读 |
| B07 | [Contraction-Aware Reinforcement Learning for Nonlinear Control with Statistical Robustness](https://ieeexplore.ieee.org/abstract/document/11664333) | IEEE Transactions on Robotics, Early Access；限定方法/实验阅读；见JSON |
| B08 | [Dream-MPC: Gradient-Based Model Predictive Control with Latent Imagination](https://proceedings.mlr.press/v306/spieler26a.html) | ICML 2026 / PMLR 306；主要方法与实验深读 |
| B09 | [Reusing Trajectories in Policy Gradients Enables Fast Convergence](https://proceedings.mlr.press/v306/montenegro26a.html) | ICML 2026 / PMLR 306；限定方法/实验阅读；见JSON |
| B10 | [Bootstrapped Model Predictive Control](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8636419dea1aa9fbd25fc4248e702da4-Abstract-Conference.html) | ICLR 2025；主要方法与实验深读 |
| B11 | [CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f381114cf5aba4e45552869863deaaa7-Abstract-Conference.html) | ICLR 2024 Spotlight；限定方法/实验阅读；见JSON |
| B12 | [Risk-Aware Reinforcement Learning with Bandit-Based Adaptation for Quadrupedal Locomotion](https://ieeexplore.ieee.org/abstract/document/11695854) | ICRA 2026; publisher and author lab confirm；主要方法与实验深读 |

## 复核方式与限制

阅读来源使用期刊/会议、作者arXiv、机构存档和官方代码。预印本与正式发表分开；作者自报接收如未查正式proceedings，明确标注。不得把库名、模板页眉或投稿目标当录用证据。

报告中的原始PDF摘要用于标识本轮所读版本，PDF/代码浅克隆未随报告上传。缺少全文的Automatica内模条目只用于近邻定位，不承载公式级结论。控制/联合的公式修正和研究假设边界另见审阅记录。

来源数量不是创新性指标。每条候选都需要与最接近的方法在相同数据、模型、动作权限、算力和部署约束下比较。
