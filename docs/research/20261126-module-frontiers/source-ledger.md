# 前沿来源总表与阅读边界（2026-11-26）

本目录登记 **45 篇来源记录**：材料 10、步态 12、学习 12、控制 10、NeuralActuator 1。包含新增与加深/保留阅读，不等于 45 篇新增全文，不包含单列的纯代码入口。论文与会议版本分开登记；未取得全文的记录明确标记。

材料为 6 篇方法/实验详读、1 篇部分全文、3 篇官方摘要/预览；步态为 8 篇方法/实验重点详读、3 篇定向阅读、1 篇元数据/摘要；学习各篇范围分别记于 JSON，不将部分实验阅读称为完整补充材料审阅。控制含直接前沿与必要历史近邻，C10 仅摘要/预览。

阅读重点是方法、假设、实现和对本项目的适用边界。没有声称阅读每一个字节、完整观看所有视频或复现这些论文。正文的拟议机制不是文献已证明的结论。

按 Nature/Science 系优先检索，但不为凑刊物层级加入无关工作。上一轮 source-ledger 保留 BGS、DeepTransition、Custom Thermodynamics、DTC、STANCE、原 PANN、官方 SuperMimic 等。2025 中科院分区未独立核验的条目不标 Top；正式状态和预印本状态不混用。

## 来源索引

| ID | 原始来源 | 正式/读取状态 | 实际范围 | 元数据 |
|---|---|---|---|---|
| MAT01 | [A multi-fidelity Bayesian neural operator for mechanics of spinodal metamaterial](https://www.nature.com/articles/s41524-026-02112-y) | npj Computational Materials | fulltext_methods_and_experiments | [material_sources.json](material_sources.json) |
| MAT02 | [Material-embedding Physics-Augmented Neural Networks: A first application to constitutive law parameterization](https://www.sciencedirect.com/science/article/pii/S0045782525004608) | Computer Methods in Applied Mechanics and Engineering | official_abstract_only | [material_sources.json](material_sources.json) |
| MAT03 | [Neural operators solve inverse problems for constitutive model discovery](https://www.sciencedirect.com/science/article/pii/S0045782526007085) | Computer Methods in Applied Mechanics and Engineering | fulltext_methods_and_experiments_plus_selected_code | [material_sources.json](material_sources.json) |
| MAT04 | [Discovering uncertainty: Bayesian constitutive artificial neural networks](https://www.sciencedirect.com/science/article/pii/S0045782524007710) | Computer Methods in Applied Mechanics and Engineering | fulltext_methods_and_experiments | [material_sources.json](material_sources.json) |
| MAT05 | [Discovering uncertainty: Gaussian constitutive neural networks with correlated weights](https://doi.org/10.1007/s00466-025-02660-y) | Computational Mechanics | fulltext_methods_and_experiments_plus_selected_code | [material_sources.json](material_sources.json) |
| MAT06 | [Adaptive Material Fingerprinting for the fast discovery of polyconvex feature combinations in isotropic and anisotropic hyperelasticity](https://www.sciencedirect.com/science/article/pii/S0020722526001321) | International Journal of Engineering Science | fulltext_methods_and_experiments | [material_sources.json](material_sources.json) |
| MAT07 | [Towards Rapid Constitutive Model Discovery from Multi-Modal Data: Physics Augmented Finite Element Model Updating (paFEMU)](https://arxiv.org/abs/2604.07746) | arXiv preprint | fulltext_methods_and_experiments | [material_sources.json](material_sources.json) |
| MAT08 | [A dual-stage constitutive modeling framework based on finite strain data-driven identification and physics-augmented neural networks](https://www.sciencedirect.com/science/article/pii/S0045782525005614) | Computer Methods in Applied Mechanics and Engineering | partial_fulltext_methods_and_experiment_setup | [material_sources.json](material_sources.json) |
| MAT09 | [A material-embedded physically recurrent neural network framework for surrogate-assisted multiscale modeling of composite structures](https://www.sciencedirect.com/science/article/abs/pii/S0263822326007841) | Composite Structures | official_abstract_and_section_previews | [material_sources.json](material_sources.json) |
| MAT10 | [Unsupervised learning of history-dependent constitutive material laws with thermodynamically-consistent neural networks in the modified Constitutive Relation Error framework](https://www.sciencedirect.com/science/article/pii/S0045782524002238) | Computer Methods in Applied Mechanics and Engineering | official_abstract_and_method_experiment_previews | [material_sources.json](material_sources.json) |
| G01 | [Learning diverse natural behaviors for enhancing the agility of quadrupedal robots](https://www.nature.com/articles/s41467-026-72475-9) | Nature Communications | deep_methods_and_experiments | [gait_sources.json](gait_sources.json) |
| G02 | [Contact Planning for Multi-Legged Robots Under Constraints Through Parallel MCTS](https://doi.org/10.1109/TRO.2025.3619054) | IEEE Transactions on Robotics | metadata_and_abstract_only | [gait_sources.json](gait_sources.json) |
| G03 | [CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion](https://ieeexplore.ieee.org/document/9932888/) | IEEE Robotics and Automation Letters | targeted_methods_and_experiments | [gait_sources.json](gait_sources.json) |
| G04 | [Learning Free Gait Transition for Quadruped Robots via Phase-Guided Controller](https://arxiv.org/abs/2201.00206) | IEEE Robotics and Automation Letters | targeted_methods_and_experiments | [gait_sources.json](gait_sources.json) |
| G05 | [Online Optimization of Central Pattern Generators for Quadruped Locomotion](https://arxiv.org/html/2410.16417v1) | IEEE/RSJ IROS 2024 | deep_methods_and_experiments | [gait_sources.json](gait_sources.json) |
| G06 | [Non-Gaited Legged Locomotion with Monte-Carlo Tree Search and Supervised Learning](https://arxiv.org/html/2408.07508v2) | IEEE Robotics and Automation Letters | deep_methods_and_experiments | [gait_sources.json](gait_sources.json) |
| G07 | [Gaitor: Learning a Unified Representation Across Gaits for Real-World Quadruped Locomotion](https://proceedings.mlr.press/v270/mitchell25a.html) | CoRL2024；PMLR2025 | deep_methods_and_experiments | [gait_sources.json](gait_sources.json) |
| G08 | [AllGaits: Learning All Quadruped Gaits and Transitions](https://arxiv.org/html/2411.04787v1) | ICRA2025 | deep_methods_and_experiments | [gait_sources.json](gait_sources.json) |
| G09 | [Learning Feasible Transitions for Efficient Contact Planning](https://proceedings.mlr.press/v283/akizhanov25a.html) | L4DC2025 | deep_methods_and_experiments | [gait_sources.json](gait_sources.json) |
| G10 | [RL-Augmented MPC for Non-Gaited Legged and Hybrid Locomotion](https://arxiv.org/html/2603.10878v1) | arXiv preprint | deep_methods_and_experiments | [gait_sources.json](gait_sources.json) |
| G11 | [Learning optimal gaits and styles for terrain-adaptive quadruped locomotion](https://arxiv.org/abs/2611.25958) | arXiv preprint | deep_methods_and_experiments | [gait_sources.json](gait_sources.json) |
| G12 | [Discovery of skill-switching criteria for learning agile quadruped locomotion](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1697159/full) | Frontiers in Robotics and AI | targeted_methods_and_experiments | [gait_sources.json](gait_sources.json) |
| L01 | [Mastering diverse control tasks through world models](https://www.nature.com/articles/s41586-025-08744-2) | Nature 640, 647–653 | full-method and selected experiment read | [learning_sources.json](learning_sources.json) |
| L02 | [Adaptive-Gradient Policy Optimization: Enhancing Policy Learning in Non-Smooth Differentiable Simulations](https://proceedings.mlr.press/v235/gao24m.html) | ICML 2024, PMLR 235:14844–14858 | deep method/experiment read | [learning_sources.json](learning_sources.json) |
| L03 | [Adaptive Horizon Actor-Critic for Policy Learning in Contact-Rich Differentiable Simulation](https://proceedings.mlr.press/v235/georgiev24a.html) | ICML 2024, PMLR 235:15418–15437 | deep method/experiment read | [learning_sources.json](learning_sources.json) |
| L04 | [Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation](https://arxiv.org/html/2412.12089v2) | ICLR 2025 Spotlight | deep method/experiment read | [learning_sources.json](learning_sources.json) |
| L05 | [Differentiable Information Enhanced Model-Based Reinforcement Learning](https://arxiv.org/html/2503.01178v1) | AAAI 2025 | deep method/experiment read | [learning_sources.json](learning_sources.json) |
| L06 | [A Multi-Fidelity Control Variate Approach for Policy Gradient Estimation](https://arxiv.org/pdf/2503.05696v4) | Transactions on Machine Learning Research | method deep read; experiment targeted read | [learning_sources.json](learning_sources.json) |
| L07 | [Multi-Fidelity Policy Gradients Stabilize Data-Scarce Reinforcement Learning](https://arxiv.org/abs/2610.02505) | arXiv preprint | deep method/experiment/critical appendix read | [learning_sources.json](learning_sources.json) |
| L08 | [Actor-Critic Model Predictive Control: Differentiable Optimization meets Reinforcement Learning for Agile Flight](https://rpg.ifi.uzh.ch/docs/TRO25_ACMPC_Romero.pdf) | IEEE Transactions on Robotics, Vol.42, pp.673–692; DOI 10.1109/TRO.2025.3644945 | deep method/selected experiment plus code read | [learning_sources.json](learning_sources.json) |
| L09 | [Efficient Learning of Differentiable MPC Policies on the GPU via Smoothed ADMM and First-Order Reinforcement Learning](https://arxiv.org/pdf/2611.09937v1) | arXiv preprint | deep method/experiment read | [learning_sources.json](learning_sources.json) |
| L10 | [Open-DiffLoco: Open-Source Differentiable Learning for Deployable Blind Quadruped Locomotion](https://arxiv.org/pdf/2608.02069v1) | arXiv preprint | deep method/experiment/appendix plus targeted code read | [learning_sources.json](learning_sources.json) |
| L11 | [FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control](https://arxiv.org/pdf/2505.22642v3) | arXiv technical report | method and selected experiment read | [learning_sources.json](learning_sources.json) |
| L12 | [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://proceedings.iclr.cc/paper_files/paper/2024/file/cf73d57b6dcda32b293df7c2d5341f49-Paper-Conference.pdf) | ICLR 2024 | method and selected experiment read | [learning_sources.json](learning_sources.json) |
| N01 | [NeuralActuator: Neural Actuation Modeling for Robot Dynamics and External Force Perception](https://arxiv.org/html/2607.11734v2) | RSS 2026 | methods, experiments, ablations, appendix; selected official code | [control_sources.json](control_sources.json) |
| C01 | [Planning to Learn: Real-Time Robust Adaptive Control on the GPU with Guaranteed Future Learning](https://arxiv.org/abs/2610.20925) | 2026 preprint | IV-A to IV-C; assumptions; selected experiments | [control_sources.json](control_sources.json) |
| C02 | [μMPC: Microsecond-Scale Nonlinear and World Model Control from Edge to GPU via Koopman-Linearized Precompilation](https://arxiv.org/html/2611.34597v1) | 2026 preprint | Koopman projection, condensed QP, precompilation, solver, experiments and limits | [control_sources.json](control_sources.json) |
| C03 | [Primal-Dual iLQR for GPU-Accelerated Learning and Control in Legged Robots](https://iit-dlslab.github.io/papers/amatucci26ral.pdf) | IEEE RA-L 11(1), 2026 | II-IV; SQP, associative scan, line search, inequalities, hardware and benchmarks; prior work revisited | [control_sources.json](control_sources.json) |
| C04 | [Real-Time Whole-Body Control of Legged Robots with Model-Predictive Path Integral Control](https://arxiv.org/abs/2409.10469) | ICRA 2025 | method and implementation; selected experiments; external localization boundary | [control_sources.json](control_sources.json) |
| C05 | [Robust Whole-Body Motion Control of Legged Robots](https://arxiv.org/abs/1703.02326) | IROS 2017 | contact projection, IMC design, model and robustness assumptions; targeted read | [control_sources.json](control_sources.json) |
| C06 | [A Whole-Body Disturbance Rejection Control Framework for Dynamic Motions in Legged Robots](https://arxiv.org/html/2502.10761v3) | IEEE RA-L 2025 | methods, estimator assumptions, force redistribution and experiments | [control_sources.json](control_sources.json) |
| C07 | [A Three-Level Whole-Body Disturbance Rejection Control Framework for Dynamic Motions in Legged Robots](https://arxiv.org/html/2508.13531v2) | 2025 preprint version read | MH-ESO, IV signal processing, V robust MPC, implementation and selected experiments | [control_sources.json](control_sources.json) |
| C08 | [Experimental Comparison of Whole-Body Control Formulations for Humanoid Robots in Task Acceleration and Task Force Spaces](https://arxiv.org/abs/2507.18502) | IROS 2025 | ID/PB formulations, task mappings and selected experiment conditions; targeted read | [control_sources.json](control_sources.json) |
| C09 | [Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response](https://arxiv.org/html/2312.11460v3) | ICLR 2024 | history encoder, HIO supervision, PPO pipeline, deployment and experiments | [control_sources.json](control_sources.json) |
| C10 | [Whole-body disturbance rejection control for quadruped robots based on variable bandwidth nonlinear extended state observer](https://www.sciencedirect.com/science/article/pii/S0967066126004107) | Control Engineering Practice 176, 107167, 2026 | official abstract and section preview only; NOT full-method read | [control_sources.json](control_sources.json) |

## 官方代码审阅

- NeuralActuator 固定 SHA 541a93aac4368e73883e86a51f31a7add6bde059；阅读 transformer、diffsim 训练器、相关配置和推理入口。没有将作者机械臂权重用于本项目训练。
- 自然技能论文官方 quadrupedal-agility 固定 SHA 15d16ea99bc23e5d401b0b0e3b88a8edcc28f0ed；阅读 BBC discriminator/latent 与 TSC 混合分布关键代码。
- 材料专题的两套官方代码与学习专题 AC-MPC/JAVE 的固定版本和实际文件范围见相应 JSON；只 pin 仓库和实际读代码分开记录。
- 本目录不包含原始论文 PDF、整个第三方代码仓库或大数据。JSON 中的本地下载路径仅记录本轮临时阅读产物，不代表这些文件已推送；可依据原始 URL 和 SHA256 复核。

## 审查后修正

1. 浮动基的关节等效力矩不能替代未驱动六个方程的接触 wrench；双计分析限定作用空间。
2. 材料 nuisance 含未知历史初态；带先验曲率不等于数据可辨识性。
3. 想象预测不准使用未来真实遥测；生成的假想观测可用于未来 belief-space 研究，但不能冒充已得证据。
4. Conditional CV 的有限 K 系数、fresh-batch 条件与数据相关门控限制已同步到学习报告；没有总体 PPO 无偏结论。
5. 软接触法向相容、执行器可实现性与异步计划复检均已纳入总方案。

## 交付导航

- [NeuralActuator 专项](../../S1_NEURALACTUATOR_REVIEW_20261126.md)
- [五模块与联合算法计划](../../S1_FIVE_MODULE_FRONTIER_PLAN_20261126.md)
- [Codex 启动提示词](../../S1_FIVE_MODULE_START_PROMPT_20261126.md)
- [材料](material.md)、[步态](gait.md)、[控制](control.md)、[学习](learning.md)
- [材料理论复核](material_review.md)、[步态/WBC 接口复核](gait_wbc_review.md)、[学习因果接口复核](learning_review.md)
