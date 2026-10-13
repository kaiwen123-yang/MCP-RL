# 100 篇方法与理论文献索引

快照日期：2026-10-13。完整元数据见 [papers.json](papers.json)，导入文件见 [papers.ris](papers.ris)，筛选口径见 [说明](selection-method.md)。

“初读”表示已有原创导读，未表示已复现或证明已完全复核；其余条目处于书目与相关性核验阶段。

## 01 综述与统一框架 · 4 篇

### MR001 · Learning-Based Model Predictive Control: Toward Safe Learning in Control

**2020 · Annual Review of Control, Robotics, and Autonomous Systems**

梳理模型学习、性能学习和MPC安全保证的联系。

[正式论文入口](https://doi.org/10.1146/annurev-control-090419-075625) · [书目核验](https://api.crossref.org/works/10.1146/annurev-control-090419-075625) · [公开全文入口](https://www.annualreviews.org/doi/pdf/10.1146/annurev-control-090419-075625)

阅读状态：书目与相关性已核验，待初读。

### MR002 · Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning

**2022 · Annual Review of Control, Robotics, and Autonomous Systems**

厘清机器人学习控制中的安全假设与方法边界。

[正式论文入口](https://doi.org/10.1146/annurev-control-042920-020211) · [书目核验](https://api.crossref.org/works/10.1146/annurev-control-042920-020211) · [公开全文入口](https://arxiv.org/pdf/2108.06266)

阅读状态：书目与相关性已核验，待初读。

### MR003 · A review of learning-based dynamics models for robotic manipulation

**2025 · Science Robotics**

综述操作任务中的学习动力学、表示和规划接口，为接触模型选择及不确定性提供背景；不是单一MPC+RL算法。

[正式论文入口](https://doi.org/10.1126/scirobotics.adt1497) · [书目核验](https://api.crossref.org/works/10.1126%2Fscirobotics.adt1497) · [公开全文入口](https://albertboai.com/assets/pdf/2025_scirobotics.adt1497.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR004 · Synthesis of model predictive control and reinforcement learning: Survey and classification

**2026 · Annual Reviews in Control**

建立MPC与RL融合机制的完整分类，作为全库导读。

[正式论文入口](https://doi.org/10.1016/j.arcontrol.2026.101045) · [书目核验](https://api.crossref.org/works/10.1016/j.arcontrol.2026.101045) · [公开全文入口](https://publications.syscop.de/Reiter2026.pdf) · [初读笔记](../notes/MR004-survey.md)

阅读状态：初读，未复现。

## 02 MPC与RL的理论联系 · 10 篇

### MR005 · Constrained model predictive control: Stability and optimality

**2000 · Automatica**

理解终端代价、终端集合与闭环稳定性条件。

[正式论文入口](https://doi.org/10.1016/s0005-1098(99)00214-9) · [书目核验](https://api.crossref.org/works/10.1016/s0005-1098%2899%2900214-9)

阅读状态：书目与相关性已核验，待初读。

### MR006 · On Average Performance and Stability of Economic Model Predictive Control

**2012 · IEEE Transactions on Automatic Control**

支持非跟踪型任务代价与长期性能的理论分析。

[正式论文入口](https://doi.org/10.1109/tac.2011.2179349) · [书目核验](https://api.crossref.org/works/10.1109/tac.2011.2179349) · [公开全文入口](https://citeseerx.ist.psu.edu/document?doi=94ba96db7a322a59d731b5cd70b3d819d8bfe760&repid=rep1&type=pdf)

阅读状态：书目与相关性已核验，待初读。

### MR007 · Learning Model Predictive Control for Iterative Tasks. A Data-Driven Control Framework

**2018 · IEEE Transactions on Automatic Control**

利用已执行轨迹构建可行集合与价值近似来持续改进控制。

[正式论文入口](https://doi.org/10.1109/tac.2017.2753460) · [书目核验](https://api.crossref.org/works/10.1109/tac.2017.2753460) · [公开全文入口](https://arxiv.org/pdf/1609.01387)

阅读状态：书目与相关性已核验，待初读。

### MR008 · Data-Driven Economic NMPC Using Reinforcement Learning

**2020 · IEEE Transactions on Automatic Control**

建立经济NMPC作为RL策略与价值近似器的理论联系。

[正式论文入口](https://doi.org/10.1109/tac.2019.2913768) · [书目核验](https://api.crossref.org/works/10.1109/tac.2019.2913768) · [公开全文入口](https://arxiv.org/pdf/1904.04152) · [初读笔记](../notes/MR008-economic-nmpc.md)

阅读状态：初读，未复现。

### MR009 · Reinforcement Learning based on MPC and the Stochastic Policy Gradient Method

**2021 · 2021 American Control Conference (ACC)**

研究MPC参数化策略的随机策略梯度学习。

[正式论文入口](https://doi.org/10.23919/acc50511.2021.9482765) · [书目核验](https://api.crossref.org/works/10.23919/acc50511.2021.9482765) · [公开全文入口](https://iris.imtlucca.it/retrieve/dd396638-3d8d-561a-e053-d805fe0a1ba5/ExploAndPolicyGradient_Stochastic_2.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR010 · A new dissipativity condition for asymptotic stability of discounted economic MPC

**2022 · Automatica**

分析折扣回报与经济MPC稳定性之间的关系。

[正式论文入口](https://doi.org/10.1016/j.automatica.2022.110287) · [书目核验](https://api.crossref.org/works/10.1016/j.automatica.2022.110287) · [公开全文入口](https://arxiv.org/pdf/2106.09377)

阅读状态：书目与相关性已核验，待初读。

### MR011 · Economic MPC of Markov Decision Processes: Dissipativity in undiscounted infinite-horizon optimal control

**2022 · Automatica**

连接MDP、耗散性与无折扣经济MPC。

[正式论文入口](https://doi.org/10.1016/j.automatica.2022.110602) · [书目核验](https://api.crossref.org/works/10.1016/j.automatica.2022.110602) · [公开全文入口](https://arxiv.org/pdf/2104.10997)

阅读状态：书目与相关性已核验，待初读。

### MR012 · Robust Learning Model-Predictive Control for Linear Systems Performing Iterative Tasks

**2022 · IEEE Transactions on Automatic Control**

将迭代任务中的数据学习与鲁棒可行性结合。

[正式论文入口](https://doi.org/10.1109/tac.2021.3083559) · [书目核验](https://api.crossref.org/works/10.1109/tac.2021.3083559) · [公开全文入口](https://arxiv.org/pdf/1911.09234)

阅读状态：书目与相关性已核验，待初读。

### MR013 · Equivalence of Optimality Criteria for Markov Decision Process and Model Predictive Control

**2024 · IEEE Transactions on Automatic Control**

研究MDP与MPC最优性准则的等价条件，支持融合目标的正确设计。

[正式论文入口](https://doi.org/10.1109/tac.2023.3277309) · [书目核验](https://api.crossref.org/works/10.1109/tac.2023.3277309) · [公开全文入口](https://arxiv.org/pdf/2210.04302)

阅读状态：书目与相关性已核验，待初读。

### MR014 · Regret Analysis of Learning-Based MPC With Partially Unknown Cost Function

**2024 · IEEE Transactions on Automatic Control**

研究MPC中未知代价的探索利用与有限时域比较基准下的遗憾。

[正式论文入口](https://doi.org/10.1109/tac.2023.3328827) · [书目核验](https://api.crossref.org/works/10.1109/tac.2023.3328827) · [公开全文入口](https://arxiv.org/pdf/2108.02307)

阅读状态：书目与相关性已核验，待初读。

## 03 策略与长期价值辅助规划 · 16 篇

### MR015 · Guided Policy Search via Approximate Mirror Descent

**2016 · NeurIPS 2016**

以近似镜像下降解释局部优化教师和全局策略之间的投影误差及有限步改进。

[正式论文入口](https://proceedings.neurips.cc/paper/2016/hash/a00e5eb0973d24649a4a920fc53d9564-Abstract.html) · [书目核验](https://proceedings.neurips.cc/paper/2016/hash/a00e5eb0973d24649a4a920fc53d9564-Abstract.html) · [公开全文入口](https://proceedings.neurips.cc/paper/2016/file/a00e5eb0973d24649a4a920fc53d9564-Paper.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR016 · Learning from the hindsight plan - Episodic MPC improvement

**2017 · ICRA 2017**

HIMPC离线计算长时域hindsight计划并学习重塑代价，让实时短时域MPC继承长期决策信息。

[正式论文入口](https://doi.org/10.1109/ICRA.2017.7989043) · [书目核验](https://cris.technion.ac.il/en/publications/learning-from-the-hindsight-plan-episodic-mpc-improvement/) · [公开全文入口](https://arxiv.org/pdf/1609.09001)

阅读状态：书目与相关性已核验，待初读。

### MR017 · Plan Online, Learn Offline: Efficient Learning and Exploration via Model-Based Control

**2019 · ICLR 2019**

POLO研究局部轨迹优化与全局价值学习的互补及价值误差、规划时域和探索的联系。

[正式论文入口](https://openreview.net/forum?id=Byey7n05FQ) · [书目核验](https://openreview.net/pdf/8279c1e102c8ffb21dad858784822e8bc003feed.pdf) · [公开全文入口](https://arxiv.org/pdf/1811.01848)

阅读状态：书目与相关性已核验，待初读。

### MR018 · Exploring Model-based Planning with Policy Networks

**2020 · ICLR 2020**

POPLIN用策略初始化动作或在策略参数空间在线搜索，并考察从规划中蒸馏可独立执行的策略。

[正式论文入口](https://openreview.net/forum?id=H1exf64KwH) · [书目核验](https://openreview.net/forum?id=H1exf64KwH) · [公开全文入口](https://arxiv.org/pdf/1906.08649)

阅读状态：书目与相关性已核验，待初读。

### MR019 · Practical Reinforcement Learning For MPC: Learning from sparse objectives in under an hour on a real robot

**2020 · Proceedings of the 2nd Conference on Learning for Dynamics and Control (L4DC), PMLR 120**

从稀疏高层目标学习MPC价值函数，并验证真实机器人学习效率。

[正式论文入口](https://proceedings.mlr.press/v120/karnchanachari20a.html) · [书目核验](https://proceedings.mlr.press/v120/karnchanachari20a.html) · [公开全文入口](https://proceedings.mlr.press/v120/karnchanachari20a/karnchanachari20a.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR020 · Blending MPC & Value Function Approximation for Efficient Reinforcement Learning

**2021 · International Conference on Learning Representations (ICLR 2021)**

通过混合预测与学习价值平衡模型偏差和价值近似误差。

[正式论文入口](https://openreview.net/forum?id=RqCC_00Bg7V) · [书目核验](https://openreview.net/pdf?id=RqCC_00Bg7V) · [公开全文入口](https://arxiv.org/pdf/2012.05909)

阅读状态：书目与相关性已核验，待初读。

### MR021 · Learning Off-Policy with Online Planning

**2022 · CoRL 2021; PMLR 164 (2022)**

LOOP用离策略价值辅助在线规划，并针对规划器与Actor行为分歧提出Actor正则化控制。

[正式论文入口](https://proceedings.mlr.press/v164/sikchi22a.html) · [书目核验](https://proceedings.mlr.press/v164/sikchi22a.html) · [公开全文入口](https://proceedings.mlr.press/v164/sikchi22a/sikchi22a.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR022 · Temporal Difference Learning for Model Predictive Control

**2022 · ICML 2022**

TD-MPC将短时域潜空间规划、终端价值和策略先验联合训练，直接连接TD学习与MPC。

[正式论文入口](https://proceedings.mlr.press/v162/hansen22a.html) · [书目核验](https://proceedings.mlr.press/v162/hansen22a.html) · [公开全文入口](https://proceedings.mlr.press/v162/hansen22a/hansen22a.pdf) · [初读笔记](../notes/MR022-MR026-tdmpc.md)

阅读状态：初读，未复现。

### MR023 · CACTO: Continuous Actor-Critic With Trajectory Optimization—Towards Global Optimality

**2023 · IEEE Robotics and Automation Letters**

研究actor-critic与轨迹优化相互改进而非简单级联。

[正式论文入口](https://doi.org/10.1109/lra.2023.3266985) · [书目核验](https://api.crossref.org/works/10.1109/lra.2023.3266985) · [公开全文入口](https://arxiv.org/pdf/2211.06625)

阅读状态：书目与相关性已核验，待初读。

### MR024 · CACTO-SL: Using Sobolev learning to improve continuous actor-critic with trajectory optimization

**2024 · Proceedings of the 6th Annual Learning for Dynamics & Control Conference (L4DC), PMLR 242**

用Sobolev学习改进actor-critic与轨迹优化之间的知识传递。

[正式论文入口](https://proceedings.mlr.press/v242/alboni24a.html) · [书目核验](https://proceedings.mlr.press/v242/alboni24a.html) · [公开全文入口](https://proceedings.mlr.press/v242/alboni24a/alboni24a.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR025 · Reinforcement Learning-Based Model Predictive Control for Discrete-Time Systems

**2024 · IEEE Transactions on Neural Networks and Learning Systems**

以MPC生成策略、RL评价并学习终端价值，分析收敛、可行性和稳定性。

[正式论文入口](https://doi.org/10.1109/tnnls.2023.3273590) · [书目核验](https://api.crossref.org/works/10.1109/tnnls.2023.3273590)

阅读状态：书目与相关性已核验，待初读。

### MR026 · TD-MPC2: Scalable, Robust World Models for Continuous Control

**2024 · ICLR 2024**

扩展TD-MPC的价值学习与模型训练稳定性，可用于研究学习模型规划的规模和多任务泛化。

[正式论文入口](https://openreview.net/forum?id=Oxh5CstDJU) · [书目核验](https://openreview.net/forum?id=Oxh5CstDJU) · [公开全文入口](https://arxiv.org/pdf/2310.16828) · [初读笔记](../notes/MR022-MR026-tdmpc.md)

阅读状态：初读，未复现。

### MR027 · Dynamic Non-Prehensile Object Transport via Model-Predictive Reinforcement Learning

**2025 · 2025 IEEE International Conference on Robotics and Automation (ICRA)**

研究预测控制与RL在非抓取动态搬运中的协作。

[正式论文入口](https://doi.org/10.1109/icra55743.2025.11127521) · [书目核验](https://api.crossref.org/works/10.1109/icra55743.2025.11127521) · [公开全文入口](https://arxiv.org/pdf/2412.00086)

阅读状态：书目与相关性已核验，待初读。

### MR028 · PIP-Loco: A Proprioceptive Infinite Horizon Planning Framework for Quadrupedal Robot Locomotion

**2025 · 2025 IEEE International Conference on Robotics and Automation (ICRA)**

结合本体感知规划、价值学习与控制以延长四足规划视野。

[正式论文入口](https://doi.org/10.1109/icra55743.2025.11128382) · [书目核验](https://api.crossref.org/works/10.1109/icra55743.2025.11128382) · [公开全文入口](https://arxiv.org/pdf/2409.09441)

阅读状态：书目与相关性已核验，待初读。

### MR029 · AC4MPC: Actor-Critic Reinforcement Learning for Guiding Model Predictive Control

**2026 · IEEE Transactions on Control Systems Technology**

研究actor初值和critic终端价值如何改善有限次NMPC求解与闭环性能。

[正式论文入口](https://doi.org/10.1109/tcst.2025.3620521) · [书目核验](https://api.crossref.org/works/10.1109/tcst.2025.3620521) · [公开全文入口](https://publications.syscop.de/Reiter2025a.pdf) · [初读笔记](../notes/MR029-ac4mpc.md)

阅读状态：初读，未复现。

### MR030 · Residual MPC: Blending Reinforcement Learning With GPU-Parallelized Model Predictive Control

**2026 · IEEE Transactions on Robotics**

GPU并行MPC与RL在力矩层残差融合，直接研究模型控制、策略补偿和训练吞吐的结合。

[正式论文入口](https://doi.org/10.1109/tro.2026.3721669) · [书目核验](https://api.crossref.org/works/10.1109%2FTRO.2026.3721669) · [公开全文入口](https://arxiv.org/pdf/2510.12717)

阅读状态：书目与相关性已核验，待初读。

## 04 学习模型与不确定性 · 21 篇

### MR031 · Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images

**2015 · NeurIPS 2015**

E2C将图像嵌入局部线性潜动力学空间，使最优控制可直接利用学习表示，是潜空间MPC的前序。

[正式论文入口](https://proceedings.neurips.cc/paper/2015/hash/a1afc58c6ca9540d057299ec3016d726-Abstract.html) · [书目核验](https://proceedings.neurips.cc/paper/2015/hash/a1afc58c6ca9540d057299ec3016d726-Abstract.html) · [公开全文入口](https://proceedings.neurips.cc/paper/2015/file/a1afc58c6ca9540d057299ec3016d726-Paper.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR032 · Value-Aware Loss Function for Model-based Reinforcement Learning

**2017 · AISTATS 2017**

根据价值函数结构学习动力学损失，说明预测误差最小与决策效果最佳并非同一个目标。

[正式论文入口](https://proceedings.mlr.press/v54/farahmand17a.html) · [书目核验](https://proceedings.mlr.press/v54/farahmand17a.html) · [公开全文入口](https://proceedings.mlr.press/v54/farahmand17a/farahmand17a.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR033 · Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models

**2018 · NeurIPS 2018**

PETS以概率动力学集成和轨迹采样连接模型不确定性与在线MPC，是学习模型规划的核心基线。

[正式论文入口](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html) · [书目核验](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html) · [公开全文入口](https://proceedings.neurips.cc/paper_files/paper/2018/file/3de568f8597b94bda53149c7d7f5958c-Paper.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR034 · Iterative Value-Aware Model Learning

**2018 · NeurIPS 2018**

利用近似价值迭代的结构简化value-aware模型学习，并分析模型与价值误差传播。

[正式论文入口](https://proceedings.neurips.cc/paper/2018/hash/7a2347d96752880e3d58d72e9813cc14-Abstract.html) · [书目核验](https://proceedings.neurips.cc/paper/2018/hash/7a2347d96752880e3d58d72e9813cc14-Abstract.html) · [公开全文入口](https://proceedings.neurips.cc/paper/2018/file/7a2347d96752880e3d58d72e9813cc14-Paper.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR035 · Universal Planning Networks: Learning Generalizable Representations for Visuomotor Control

**2018 · ICML 2018**

UPN通过可微规划训练潜表示，并将该表示用于新任务的RL奖励，连接规划结构与学习表征。

[正式论文入口](https://proceedings.mlr.press/v80/srinivas18b.html) · [书目核验](https://proceedings.mlr.press/v80/srinivas18b.html) · [公开全文入口](https://proceedings.mlr.press/v80/srinivas18b/srinivas18b.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR036 · Learning Latent Dynamics for Planning from Pixels

**2019 · ICML 2019**

PlaNet通过随机潜动力学和多步训练支持从像素在线规划，是潜空间MPC的重要前序。

[正式论文入口](https://proceedings.mlr.press/v97/hafner19a.html) · [书目核验](https://proceedings.mlr.press/v97/hafner19a.html) · [公开全文入口](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR037 · When to Trust Your Model: Model-Based Policy Optimization

**2019 · NeurIPS 2019**

MBPO以短模型rollout分析数据效率与模型偏差，是MPC+RL选择预测长度的理论对照，本身非在线MPC框架。

[正式论文入口](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html) · [书目核验](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html) · [公开全文入口](https://proceedings.neurips.cc/paper/2019/file/5faf461eff3099671ad63c6f3f094f7f-Paper.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR038 · Efficient Model-Based Reinforcement Learning through Optimistic Policy Search and Planning

**2020 · NeurIPS 2020**

H-UCRL将认知不确定性转为乐观规划的附加控制变量，研究模型不确定性如何指导探索而非仅保守避让。

[正式论文入口](https://proceedings.neurips.cc/paper/2020/hash/a36b598abb934e4528412e5a2127b931-Abstract.html) · [书目核验](https://proceedings.neurips.cc/paper/2020/hash/a36b598abb934e4528412e5a2127b931-Abstract.html) · [公开全文入口](https://proceedings.neurips.cc/paper/2020/file/a36b598abb934e4528412e5a2127b931-Paper.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR039 · Mastering Atari, Go, chess and shogi by planning with a learned model

**2020 · Nature**

MuZero将学习模型、价值函数与树搜索结合，支撑任务相关模型与在线规划分工；不是连续控制MPC。

[正式论文入口](https://doi.org/10.1038/s41586-020-03051-4) · [书目核验](https://api.crossref.org/works/10.1038%2Fs41586-020-03051-4) · [公开全文入口](https://arxiv.org/pdf/1911.08265)

阅读状态：书目与相关性已核验，待初读。

### MR040 · The Value Equivalence Principle for Model-Based Reinforcement Learning

**2020 · NeurIPS 2020**

用Bellman更新等价定义对规划足够的模型，为理解任务导向模型不必逐状态精确预测提供理论。

[正式论文入口](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html) · [书目核验](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html) · [公开全文入口](https://proceedings.neurips.cc/paper/2020/file/3bb585ea00014b0e3ebe4c6dd165a358-Paper.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR041 · Data-Driven MPC for Quadrotors

**2021 · IEEE Robotics and Automation Letters**

以高斯过程学习气动残差并嵌入实时MPC，显式研究模型精度与计算量取舍；属于学习MPC而非严格RL，是模型学习路线的重要强基线。

[正式论文入口](https://doi.org/10.1109/lra.2021.3061307) · [书目核验](https://api.crossref.org/works/10.1109%2FLRA.2021.3061307) · [公开全文入口](https://rpg.ifi.uzh.ch/docs/RAL21_Torrente.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR042 · Learning where to trust unreliable models in an unstructured world for deformable object manipulation

**2021 · Science Robotics**

学习动力学模型适用范围并据此选择可相信的规划区域，支持有限数据和模型失配条件下的接触操作；不是严格MPC+RL。

[正式论文入口](https://doi.org/10.1126/scirobotics.abd8170) · [书目核验](https://api.crossref.org/works/10.1126%2Fscirobotics.abd8170)

阅读状态：书目与相关性已核验，待初读。

### MR043 · Model-Based Reinforcement Learning via Latent-Space Collocation

**2021 · ICML 2021**

LatCo直接优化潜状态序列而非仅射击动作序列，针对长时域规划的信用分配与条件数问题。

[正式论文入口](https://proceedings.mlr.press/v139/rybkin21b.html) · [书目核验](https://proceedings.mlr.press/v139/rybkin21b.html) · [公开全文入口](https://proceedings.mlr.press/v139/rybkin21b/rybkin21b.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR044 · Online Learning of Unknown Dynamics for Model-Based Controllers in Legged Locomotion

**2021 · IEEE Robotics and Automation Letters**

在线学习局部时变残差补偿模型失配，含足式机器人及理论分析；属于监督模型学习而非RL，适合作为学习模型融合的强邻近基线。

[正式论文入口](https://doi.org/10.1109/lra.2021.3108510) · [书目核验](https://api.crossref.org/works/10.1109%2FLRA.2021.3108510) · [公开全文入口](https://hybrid-robotics.berkeley.edu/publications/RAL2021_OnlineLearningLocomotion.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR045 · Value Gradient weighted Model-Based Reinforcement Learning

**2022 · ICLR 2022**

VaGraM以价值梯度衡量状态预测误差对决策的影响，处理有限模型容量下value-aware损失的优化稳定性。

[正式论文入口](https://openreview.net/forum?id=4-D6CZkRXxI) · [书目核验](https://cvoelcker.de/publications/) · [公开全文入口](https://arxiv.org/pdf/2204.01464)

阅读状态：书目与相关性已核验，待初读。

### MR046 · Bayesian Multi-Task Learning MPC for Robotic Mobile Manipulation

**2023 · IEEE Robotics and Automation Letters**

跨任务学习误差先验并在线适应、嵌入MPC，支撑未知交互动力学和快速适应；属于学习MPC而非严格RL。

[正式论文入口](https://doi.org/10.1109/lra.2023.3264758) · [书目核验](https://api.crossref.org/works/10.1109%2FLRA.2023.3264758) · [公开全文入口](https://arxiv.org/pdf/2211.10270)

阅读状态：书目与相关性已核验，待初读。

### MR047 · Real-Time Neural MPC: Deep Learning Model Predictive Control for Quadrotors and Agile Robotic Platforms

**2023 · IEEE Robotics and Automation Letters**

将较大神经动力学模型高效嵌入在线优化，直接针对学习模型容量与实时求解的矛盾；不是RL。

[正式论文入口](https://doi.org/10.1109/lra.2023.3246839) · [书目核验](https://api.crossref.org/works/10.1109%2FLRA.2023.3246839) · [公开全文入口](https://rpg.ifi.uzh.ch/docs/RAL2023_Salzmann.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR048 · Learning Agile Locomotion and Adaptive Behaviors via RL-augmented MPC

**2024 · 2024 IEEE International Conference on Robotics and Automation (ICRA)**

通过RL增强MPC实现四足支撑与摆腿的适应控制。

[正式论文入口](https://doi.org/10.1109/icra57147.2024.10610453) · [书目核验](https://api.crossref.org/works/10.1109/icra57147.2024.10610453) · [公开全文入口](https://arxiv.org/pdf/2310.09442)

阅读状态：书目与相关性已核验，待初读。

### MR049 · Calibrated Value-Aware Model Learning with Probabilistic Environment Models

**2025 · ICML 2025**

分析常用价值感知损失的非校准问题并给出修正，适合研究潜模型与终端价值联合学习为何会失真。

[正式论文入口](https://proceedings.mlr.press/v267/voelcker25a.html) · [书目核验](https://proceedings.mlr.press/v267/voelcker25a.html) · [公开全文入口](https://arxiv.org/pdf/2505.22772)

阅读状态：书目与相关性已核验，待初读。

### MR050 · Mastering diverse control tasks through world models

**2025 · Nature**

Dreamer方法支撑：通过世界模型中的想象轨迹训练策略，说明规划成本可转入训练；不是部署时显式MPC。

[正式论文入口](https://doi.org/10.1038/s41586-025-08744-2) · [书目核验](https://api.crossref.org/works/10.1038%2Fs41586-025-08744-2) · [公开全文入口](https://www.nature.com/articles/s41586-025-08744-2.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR051 · Model-based reinforcement learning for ultrasound-driven autonomous microrobots

**2025 · Nature Machine Intelligence**

超声微机器人上的模型强化学习，支撑数据高效学习模型与控制的结合；平台邻近，不作为机械臂或四足证据。

[正式论文入口](https://doi.org/10.1038/s42256-025-01054-2) · [书目核验](https://api.crossref.org/works/10.1038%2Fs42256-025-01054-2) · [公开全文入口](https://www.nature.com/articles/s42256-025-01054-2.pdf)

阅读状态：书目与相关性已核验，待初读。

## 05 安全约束与稳定性 · 12 篇

### MR052 · Provably safe and robust learning-based model predictive control

**2013 · Automatica**

为学习模型与鲁棒MPC的组合提供早期安全理论。

[正式论文入口](https://doi.org/10.1016/j.automatica.2013.02.003) · [书目核验](https://api.crossref.org/works/10.1016/j.automatica.2013.02.003) · [公开全文入口](https://arxiv.org/pdf/1107.2487)

阅读状态：书目与相关性已核验，待初读。

### MR053 · Linear Model Predictive Safety Certification for Learning-Based Control

**2018 · 2018 IEEE Conference on Decision and Control (CDC)**

建立线性系统中学习动作的预测安全认证框架。

[正式论文入口](https://doi.org/10.1109/cdc.2018.8619829) · [书目核验](https://api.crossref.org/works/10.1109/cdc.2018.8619829) · [公开全文入口](https://arxiv.org/pdf/1803.08552)

阅读状态：书目与相关性已核验，待初读。

### MR054 · A predictive safety filter for learning-based control of constrained nonlinear dynamical systems

**2021 · Automatica**

为学习策略提供通过有限时域预测实现的约束过滤。

[正式论文入口](https://doi.org/10.1016/j.automatica.2021.109597) · [书目核验](https://api.crossref.org/works/10.1016/j.automatica.2021.109597) · [公开全文入口](https://arxiv.org/pdf/1812.05506)

阅读状态：书目与相关性已核验，待初读。

### MR055 · A Predictive Safety Filter for Learning-Based Racing Control

**2021 · IEEE Robotics and Automation Letters**

展示预测安全过滤器如何与学习控制器在快速系统上协作。

[正式论文入口](https://doi.org/10.1109/lra.2021.3097073) · [书目核验](https://api.crossref.org/works/10.1109/lra.2021.3097073) · [公开全文入口](https://arxiv.org/pdf/2102.11907)

阅读状态：书目与相关性已核验，待初读。

### MR056 · Safe Reinforcement Learning Using Robust MPC

**2021 · IEEE Transactions on Automatic Control**

以鲁棒MPC结构实现RL策略的约束保障和学习。

[正式论文入口](https://doi.org/10.1109/tac.2020.3024161) · [书目核验](https://api.crossref.org/works/10.1109/tac.2020.3024161) · [公开全文入口](https://arxiv.org/pdf/1906.04005)

阅读状态：书目与相关性已核验，待初读。

### MR057 · Learning for MPC with stability & safety guarantees

**2022 · Automatica**

讨论在线更新MPC参数时如何保持稳定性和安全性。

[正式论文入口](https://doi.org/10.1016/j.automatica.2022.110598) · [书目核验](https://api.crossref.org/works/10.1016/j.automatica.2022.110598) · [公开全文入口](https://arxiv.org/pdf/2012.07369)

阅读状态：书目与相关性已核验，待初读。

### MR058 · Probabilistic Model Predictive Safety Certification for Learning-Based Control

**2022 · IEEE Transactions on Automatic Control**

研究不确定模型下学习控制器的概率安全认证。

[正式论文入口](https://doi.org/10.1109/tac.2021.3049335) · [书目核验](https://api.crossref.org/works/10.1109/tac.2021.3049335) · [公开全文入口](https://arxiv.org/pdf/1906.10417)

阅读状态：书目与相关性已核验，待初读。

### MR059 · Safe Reinforcement Learning with Chance-constrained Model Predictive Control

**2022 · Proceedings of The 4th Annual Learning for Dynamics and Control Conference (L4DC), PMLR 168**

将机会约束MPC安全引导与策略梯度学习共同设计。

[正式论文入口](https://proceedings.mlr.press/v168/pfrommer22a.html) · [书目核验](https://proceedings.mlr.press/v168/pfrommer22a.html) · [公开全文入口](https://proceedings.mlr.press/v168/pfrommer22a/pfrommer22a.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR060 · Stability-constrained Markov Decision Processes using MPC

**2022 · Automatica**

用MPC结构将稳定性条件纳入MDP策略学习。

[正式论文入口](https://doi.org/10.1016/j.automatica.2022.110399) · [书目核验](https://api.crossref.org/works/10.1016/j.automatica.2022.110399) · [公开全文入口](https://arxiv.org/pdf/2102.01383)

阅读状态：书目与相关性已核验，待初读。

### MR061 · Predictive Control Barrier Functions: Enhanced Safety Mechanisms for Learning-Based Control

**2023 · IEEE Transactions on Automatic Control**

结合预测优化和屏障函数扩大安全学习控制的适用域。

[正式论文入口](https://doi.org/10.1109/tac.2022.3175628) · [书目核验](https://api.crossref.org/works/10.1109/tac.2022.3175628) · [公开全文入口](https://arxiv.org/pdf/2105.10241)

阅读状态：书目与相关性已核验，待初读。

### MR062 · Signal Temporal Logic Neural Predictive Control

**2023 · IEEE Robotics and Automation Letters**

训练神经控制器优化时序逻辑鲁棒度并以预测方式部署，提供替代在线重优化的快速约束控制参考；非标准actor-critic MPC。

[正式论文入口](https://doi.org/10.1109/lra.2023.3315536) · [书目核验](https://api.crossref.org/works/10.1109%2FLRA.2023.3315536) · [公开全文入口](https://arxiv.org/pdf/2309.05131)

阅读状态：书目与相关性已核验，待初读。

### MR063 · Deep Model Predictive Control With Stability Guarantees

**2025 · IEEE Transactions on Automatic Control**

结合神经模型双时间尺度适应与MPC，研究闭环稳定保证。

[正式论文入口](https://doi.org/10.1109/tac.2025.3550072) · [书目核验](https://api.crossref.org/works/10.1109/tac.2025.3550072) · [公开全文入口](https://arxiv.org/pdf/2104.07171)

阅读状态：书目与相关性已核验，待初读。

## 06 采样优化与实时计算 · 17 篇

### MR064 · Suboptimal model predictive control (feasibility implies stability)

**1999 · IEEE Transactions on Automatic Control**

提供不要求每次精确最优求解的稳定MPC基础。

[正式论文入口](https://doi.org/10.1109/9.751369) · [书目核验](https://api.crossref.org/works/10.1109/9.751369) · [公开全文入口](https://minds.wisconsin.edu/bitstream/1793/10898/1/file_1.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR065 · A Real-Time Iteration Scheme for Nonlinear Optimization in Optimal Feedback Control

**2005 · SIAM Journal on Control and Optimization**

提供每个控制周期只执行少量优化步骤的经典求解框架。

[正式论文入口](https://doi.org/10.1137/s0363012902400713) · [书目核验](https://api.crossref.org/works/10.1137/s0363012902400713) · [公开全文入口](https://lirias.kuleuven.be/retrieve/21380d1b-30f1-4f66-8f3f-f9c6f5b7f01c)

阅读状态：书目与相关性已核验，待初读。

### MR066 · Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving

**2018 · IEEE Transactions on Robotics**

信息论采样MPC/MPPI的理论与实现基础，支撑学习采样分布、价值函数和模型的融合；不是单独RL算法。

[正式论文入口](https://doi.org/10.1109/tro.2018.2865891) · [书目核验](https://api.crossref.org/works/10.1109%2FTRO.2018.2865891) · [公开全文入口](https://arxiv.org/pdf/1707.02342)

阅读状态：书目与相关性已核验，待初读。

### MR067 · An Online Learning Approach to Model Predictive Control

**2019 · Robotics: Science and Systems XV**

用动态镜像下降统一采样MPC更新，为学习优化提供方法基础。

[正式论文入口](https://doi.org/10.15607/rss.2019.xv.033) · [书目核验](https://api.crossref.org/works/10.15607/rss.2019.xv.033) · [公开全文入口](https://www.roboticsproceedings.org/rss15/p33.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR068 · Model-Predictive Control via Cross-Entropy and Gradient-Based Optimization

**2020 · L4DC 2020**

交替使用CEM与梯度更新优化学习动力学下的动作序列，为有限计算下搜索策略提供对照。

[正式论文入口](https://proceedings.mlr.press/v120/bharadhwaj20a.html) · [书目核验](https://proceedings.mlr.press/v120/bharadhwaj20a.html) · [公开全文入口](https://proceedings.mlr.press/v120/bharadhwaj20a/bharadhwaj20a.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR069 · Time-distributed optimization for real-time model predictive control: Stability, robustness, and constraint satisfaction

**2020 · Automatica**

分析实时有限优化步骤如何影响闭环稳定、鲁棒性与约束满足。

[正式论文入口](https://doi.org/10.1016/j.automatica.2020.108973) · [书目核验](https://api.crossref.org/works/10.1016/j.automatica.2020.108973) · [公开全文入口](https://arxiv.org/pdf/1903.02605)

阅读状态：书目与相关性已核验，待初读。

### MR070 · A Lyapunov function for the combined system-optimizer dynamics in inexact model predictive control

**2021 · Automatica**

分析有限次求解时系统与优化器联合动态的稳定性。

[正式论文入口](https://doi.org/10.1016/j.automatica.2021.109901) · [书目核验](https://api.crossref.org/works/10.1016/j.automatica.2021.109901) · [公开全文入口](https://publications.syscop.de/Zanelli2021b.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR071 · Robust Model Predictive Path Integral Control: Analysis and Performance Guarantees

**2021 · IEEE Robotics and Automation Letters**

给出鲁棒MPPI分析及性能保证，支撑学习采样控制器在扰动/模型误差下的设计；不把采样MPC误标成RL。

[正式论文入口](https://doi.org/10.1109/lra.2021.3057563) · [书目核验](https://api.crossref.org/works/10.1109%2FLRA.2021.3057563) · [公开全文入口](https://arxiv.org/pdf/2102.09027)

阅读状态：书目与相关性已核验，待初读。

### MR072 · Sample-efficient Cross-Entropy Method for Real-time Planning

**2021 · CoRL 2020; PMLR 155 (2021)**

iCEM以时间相关采样和记忆提升轨迹优化效率，是判断学习采样器增量价值的重要非学习基线。

[正式论文入口](https://proceedings.mlr.press/v155/pinneri21a.html) · [书目核验](https://proceedings.mlr.press/v155/pinneri21a.html) · [公开全文入口](https://proceedings.mlr.press/v155/pinneri21a/pinneri21a.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR073 · Learning to Optimize in Model Predictive Control

**2022 · ICRA 2022**

从高采样预算教师学习MPC的分布更新规则，区分学习求解过程与学习动力学的作用。

[正式论文入口](https://doi.org/10.1109/ICRA46639.2022.9812369) · [书目核验](https://doi.org/10.1109/ICRA46639.2022.9812369) · [公开全文入口](https://arxiv.org/pdf/2212.02603)

阅读状态：书目与相关性已核验，待初读。

### MR074 · Variational Inference MPC using Normalizing Flows and Out-of-Distribution Projection

**2022 · Robotics: Science and Systems (RSS) 2022**

FlowMPPI学习条件控制序列分布并对分布外环境表示作投影，处理规划先验失配。

[正式论文入口](https://www.roboticsproceedings.org/rss18/p027.pdf) · [书目核验](https://www.roboticsproceedings.org/rss18/p027.pdf) · [公开全文入口](https://www.roboticsproceedings.org/rss18/p027.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR075 · Learning Sampling Distributions for Model Predictive Control

**2023 · CoRL 2022; PMLR 205 (2023)**

用归一化流学习MPC的潜空间采样、分布更新与warm-start，研究少样本规划的效率。

[正式论文入口](https://proceedings.mlr.press/v205/sacks23a.html) · [书目核验](https://proceedings.mlr.press/v205/sacks23a.html) · [公开全文入口](https://proceedings.mlr.press/v205/sacks23a/sacks23a.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR076 · Optimization of the model predictive control meta-parameters through reinforcement learning

**2023 · Engineering Applications of Artificial Intelligence**

RL按状态联合选择何时重算MPC和其他元参数，并在两次求解之间使用线性反馈，直接研究控制性能与计算代价的协同优化；是预算感知融合的直接前驱，不能再把事件触发本身当成新创新。

[正式论文入口](https://doi.org/10.1016/j.engappai.2023.106211) · [书目核验](https://api.crossref.org/works/10.1016%2Fj.engappai.2023.106211) · [公开全文入口](https://arxiv.org/pdf/2111.04146)

阅读状态：书目与相关性已核验，待初读。

### MR077 · Deep Model Predictive Optimization

**2024 · ICRA 2024**

DMPO把MPC优化器及warm-start作为可学习决策过程，用RL改善有限计算预算下的优化。

[正式论文入口](https://doi.org/10.1109/ICRA57147.2024.10611492) · [书目核验](https://doi.org/10.1109/ICRA57147.2024.10611492) · [公开全文入口](https://arxiv.org/pdf/2310.04590)

阅读状态：书目与相关性已核验，待初读。

### MR078 · Learning a Generalizable Trajectory Sampling Distribution for Model Predictive Control

**2024 · IEEE Transactions on Robotics**

研究可泛化的轨迹采样先验如何帮助MPC，适合分析学习分布在新问题上的可靠性。

[正式论文入口](https://doi.org/10.1109/TRO.2024.3370026) · [书目核验](https://doi.org/10.1109/TRO.2024.3370026)

阅读状态：书目与相关性已核验，待初读。

### MR079 · RT-HCP: Dealing with Inference Delays and Sample Efficiency to Learn Directly on Robotic Platforms

**2025 · 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)**

结合物理先验、actor-critic与动作段预测控制处理真实推理延迟。

[正式论文入口](https://doi.org/10.1109/iros60139.2025.11247277) · [书目核验](https://api.crossref.org/works/10.1109/iros60139.2025.11247277) · [公开全文入口](https://arxiv.org/pdf/2509.06714)

阅读状态：书目与相关性已核验，待初读。

### MR080 · SIT-LMPC: Safe Information-Theoretic Learning Model Predictive Control for Iterative Tasks

**2026 · IEEE Robotics and Automation Letters**

安全信息论学习MPC将迭代经验与采样控制结合，为性能改进与安全可行性提供方法；学习并非必然神经RL。

[正式论文入口](https://doi.org/10.1109/lra.2025.3634881) · [书目核验](https://api.crossref.org/works/10.1109%2FLRA.2025.3634881) · [公开全文入口](https://arxiv.org/pdf/2602.16187)

阅读状态：书目与相关性已核验，待初读。

## 07 可微MPC与结构化策略 · 7 篇

### MR081 · Differentiable MPC for End-to-end Planning and Control

**2018 · Advances in Neural Information Processing Systems 31 (NeurIPS 2018)**

奠定将MPC作为可微策略层、端到端学习代价与动力学的基础。

[正式论文入口](https://papers.neurips.cc/paper_files/paper/2018/hash/ba6d843eb4251a4526ce65d1807a9309-Abstract.html) · [书目核验](https://papers.neurips.cc/paper_files/paper/2018/hash/ba6d843eb4251a4526ce65d1807a9309-Abstract.html) · [公开全文入口](https://papers.neurips.cc/paper_files/paper/2018/file/ba6d843eb4251a4526ce65d1807a9309-Paper.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR082 · Differentiable Convex Optimization Layers

**2019 · Advances in Neural Information Processing Systems 32 (NeurIPS 2019)**

提供可微凸优化层的通用方法，用于理解MPC层如何传播梯度。

[正式论文入口](https://papers.neurips.cc/paper_files/paper/2019/hash/9ce3c52fc54362e22053399d3181c638-Abstract.html) · [书目核验](https://papers.neurips.cc/paper_files/paper/2019/hash/9ce3c52fc54362e22053399d3181c638-Abstract.html) · [公开全文入口](https://papers.neurips.cc/paper_files/paper/2019/file/9ce3c52fc54362e22053399d3181c638-Paper.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR083 · Planning with Diffusion for Flexible Behavior Synthesis

**2022 · ICML 2022**

Diffuser用学习轨迹分布和奖励引导采样实现规划，提供将优化过程移入生成模型的相邻路线。

[正式论文入口](https://proceedings.mlr.press/v162/janner22a.html) · [书目核验](https://proceedings.mlr.press/v162/janner22a.html) · [公开全文入口](https://proceedings.mlr.press/v162/janner22a/janner22a.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR084 · Differentiable Robust Model Predictive Control

**2024 · Robotics: Science and Systems XX**

将鲁棒预测控制与端到端可微学习结合。

[正式论文入口](https://doi.org/10.15607/rss.2024.xx.003) · [书目核验](https://api.crossref.org/works/10.15607/rss.2024.xx.003) · [公开全文入口](https://www.roboticsproceedings.org/rss20/p003.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR085 · Learning Constrained Parametric Differentiable Predictive Control Policies With Guarantees

**2024 · IEEE Transactions on Systems, Man, and Cybernetics: Systems**

研究带约束和保证的可微预测策略学习。

[正式论文入口](https://doi.org/10.1109/tsmc.2024.3368026) · [书目核验](https://api.crossref.org/works/10.1109/tsmc.2024.3368026) · [公开全文入口](https://www.osti.gov/servlets/purl/2438488)

阅读状态：书目与相关性已核验，待初读。

### MR086 · Actor–Critic Model Predictive Control: Differentiable Optimization Meets Reinforcement Learning for Agile Flight

**2026 · IEEE Transactions on Robotics**

将可微MPC嵌入actor，研究学习代价、泛化与训练推理成本。

[正式论文入口](https://doi.org/10.1109/tro.2025.3644945) · [书目核验](https://api.crossref.org/works/10.1109/tro.2025.3644945) · [公开全文入口](https://arxiv.org/pdf/2306.09852v8)

阅读状态：书目与相关性已核验，待初读。

### MR087 · Differentiable Model Predictive Control on the GPU

**2026 · International Conference on Learning Representations (ICLR 2026)**

以GPU并行求解及隐式微分提高可微MPC训练效率，并用于强化学习控制。

[正式论文入口](https://openreview.net/forum?id=bFYfV6c9zu) · [书目核验](https://iclr.cc/virtual/2026/poster/10008630) · [公开全文入口](https://openreview.net/pdf?id=bFYfV6c9zu)

阅读状态：书目与相关性已核验，待初读。

## 08 规划教师与分层学习 · 13 篇

### MR088 · Guided Policy Search

**2013 · ICML 2013**

用轨迹优化生成引导样本训练全局策略，是优化教师与策略学习结合的基础框架。

[正式论文入口](https://proceedings.mlr.press/v28/levine13.html) · [书目核验](https://proceedings.mlr.press/v28/levine13.html) · [公开全文入口](https://proceedings.mlr.press/v28/levine13.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR089 · Learning Neural Network Policies with Guided Policy Search under Unknown Dynamics

**2014 · NeurIPS 2014**

迭代拟合局部动力学并优化轨迹，使未知动力学下的优化教师与神经策略协同学习。

[正式论文入口](https://proceedings.neurips.cc/paper_files/paper/2014/hash/c7c9344b5a3c0533e29fa69ce807cf08-Abstract.html) · [书目核验](https://proceedings.neurips.cc/paper_files/paper/2014/hash/c7c9344b5a3c0533e29fa69ce807cf08-Abstract.html) · [公开全文入口](https://proceedings.neurips.cc/paper_files/paper/2014/file/c7c9344b5a3c0533e29fa69ce807cf08-Paper.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR090 · Learning deep control policies for autonomous aerial vehicles with MPC-guided policy search

**2016 · ICRA 2016**

用在线MPC教师引导深策略，直接研究训练阶段优化与部署阶段网络控制的分工。

[正式论文入口](https://doi.org/10.1109/ICRA.2016.7487175) · [书目核验](https://doi.org/10.1109/ICRA.2016.7487175) · [公开全文入口](https://arxiv.org/pdf/1509.06791)

阅读状态：书目与相关性已核验，待初读。

### MR091 · Combining Model-Based and Model-Free Updates for Trajectory-Centric Reinforcement Learning

**2017 · ICML 2017**

将局部LQR模型更新与路径积分无模型改进结合，可接GPS，研究模型偏差下两类更新的互补。

[正式论文入口](https://proceedings.mlr.press/v70/chebotar17a.html) · [书目核验](https://proceedings.mlr.press/v70/chebotar17a.html) · [公开全文入口](https://proceedings.mlr.press/v70/chebotar17a/chebotar17a.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR092 · PLATO: Policy learning using adaptive trajectory optimization

**2017 · ICRA 2017**

利用自适应轨迹优化产生训练指导，研究在线教师如何覆盖学习策略所遇状态。

[正式论文入口](https://doi.org/10.1109/ICRA.2017.7989379) · [书目核验](https://doi.org/10.1109/ICRA.2017.7989379) · [公开全文入口](https://arxiv.org/pdf/1603.00622)

阅读状态：书目与相关性已核验，待初读。

### MR093 · Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning

**2018 · ICRA 2018**

先用神经动力学与MPC获得有效行为，再进行无模型微调，研究两种学习阶段的互补。

[正式论文入口](https://doi.org/10.1109/ICRA.2018.8463189) · [书目核验](https://doi.org/10.1109/ICRA.2018.8463189) · [公开全文入口](https://arxiv.org/pdf/1708.02596)

阅读状态：书目与相关性已核验，待初读。

### MR094 · See, feel, act: Hierarchical learning for complex manipulation skills with multisensory fusion

**2019 · Science Robotics**

从视触觉交互学习层级概率接触模型并用滚动规划选择动作，支撑接触抽象、主动信息与MPC接口；不是严格RL。

[正式论文入口](https://doi.org/10.1126/scirobotics.aav3123) · [书目核验](https://api.crossref.org/works/10.1126%2Fscirobotics.aav3123) · [公开全文入口](https://mcube.mit.edu/pdfs/2019-ScienceRob-Jenga-preprint.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR095 · MPC-Net: A First Principles Guided Policy Search

**2020 · IEEE Robotics and Automation Letters**

以控制Hamiltonian而非单纯动作拟合损失蒸馏MPC，连接最优性条件、约束与策略学习。

[正式论文入口](https://doi.org/10.1109/LRA.2020.2974653) · [书目核验](https://doi.org/10.1109/LRA.2020.2974653) · [公开全文入口](https://arxiv.org/pdf/1909.05197)

阅读状态：书目与相关性已核验，待初读。

### MR096 · Extracting Strong Policies for Robotics Tasks from Zero-Order Trajectory Optimizers

**2021 · ICLR 2021**

APEX联合轨迹优化与策略蒸馏，并自适应调整教师引导，防止优化器过早坍缩到弱策略。

[正式论文入口](https://openreview.net/pdf/1e36a9b55c2b184bab1395be47101e4beb882f41.pdf) · [书目核验](https://openreview.net/pdf/1e36a9b55c2b184bab1395be47101e4beb882f41.pdf) · [公开全文入口](https://openreview.net/pdf/1e36a9b55c2b184bab1395be47101e4beb882f41.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR097 · Learning high-speed flight in the wild

**2021 · Science Robotics**

从具有完整信息的规划专家蒸馏感知到轨迹策略，并由MPC跟踪；属于优化与模仿学习而非严格RL，是训练部署计算分工的重要邻近方法。

[正式论文入口](https://doi.org/10.1126/scirobotics.abg5810) · [书目核验](https://api.crossref.org/works/10.1126%2Fscirobotics.abg5810) · [公开全文入口](https://rpg.ifi.uzh.ch/docs/Loquercio21_Science.pdf)

阅读状态：书目与相关性已核验，待初读。

### MR098 · Model Predictive Actor-Critic: Accelerating Robot Skill Acquisition with Deep Reinforcement Learning

**2021 · ICRA 2021**

MoPAC以模型预测rollout引导Actor-Critic学习，研究规划利用与无模型探索如何共同减轻模型偏差。

[正式论文入口](https://doi.org/10.1109/ICRA48506.2021.9561298) · [书目核验](https://doi.org/10.1109/ICRA48506.2021.9561298) · [公开全文入口](https://arxiv.org/pdf/2103.13842)

阅读状态：书目与相关性已核验，待初读。

### MR099 · Reaching the limit in autonomous racing: Optimal control versus reinforcement learning

**2023 · Science Robotics**

直接分析最优控制与RL在无人机竞速中的目标函数、分层接口及模型误差差异；不是融合方法，但为为什么融合、怎样设置强基线提供关键证据。

[正式论文入口](https://doi.org/10.1126/scirobotics.adg1462) · [书目核验](https://api.crossref.org/works/10.1126%2Fscirobotics.adg1462) · [公开全文入口](https://arxiv.org/pdf/2310.10943)

阅读状态：书目与相关性已核验，待初读。

### MR100 · Precise aggressive aerial maneuvers with sensorimotor policies

**2026 · Science Robotics**

用模型轨迹规划提供初始化状态来缓解狭窄可行空间的RL探索问题，支撑训练侧优化引导；不是部署时在线MPC融合。

[正式论文入口](https://doi.org/10.1126/scirobotics.aeb0180) · [书目核验](https://api.crossref.org/works/10.1126%2Fscirobotics.aeb0180) · [公开全文入口](https://arxiv.org/pdf/2604.05828)

阅读状态：书目与相关性已核验，待初读。
