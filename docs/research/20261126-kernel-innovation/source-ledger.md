# 论文、官方代码与阅读覆盖证据表

2026-11-26，Asia/Shanghai。配套[主报告](../../S1_KERNEL_INNOVATION_REVIEW_20261126.md)。本表保存17篇论文的方法/实验层面阅读记录及3项有限范围补充。精读指从论文正文取得列明内容，不等于每条参考文献、全部附录证明或补充视频均已读完。未运行论文训练代码；所有性能数字是原作者在其任务/设备上的报告，不能外推本项目。

检索顺序为 Nature、Science、2025 CAS 高分区期刊及相关顶会。CAS 信息与正式出版状态独立核验；本轮没有取得完整2025原表。T-RO/CMAME、RA-L的高校公开分区交叉说明见主报告，其余分区不猜填。正式论文和实际阅读的作者版本有差异时明确登记。大型PDF通过本地下载后抽取正文阅读；没有把网页工具下载失败写成已经读到全文，也未把有版权的整篇PDF/全文再发布到本仓库。

## A. Nature、Science：优先机制来源

| ID / 工作 | 一手来源 | 实际阅读与实现认识 | 明确边界 |
|---|---|---|---|
| N1 Humphreys & Zhou，Learning to adapt through bio-inspired gait strategies for versatile quadruped locomotion，Nature Machine Intelligence 2025 | [正式页](https://www.nature.com/articles/s42256-025-01065-z)；[UCL存档](https://discovery.ucl.ac.uk/id/eprint/10232228/)；[成功读取的正式PDF存档](https://d-nb.info/1377065790/34)；[作者版](https://arxiv.org/html/2412.09440v3) | 正式版结果pp1143–1148、Methods pp1149–1151；双层PPO、状态反馈落点、步态过渡与接触/能耗目标；核对官方scheduler/YAML | 步态记忆并非持久材料地块记忆；论文与代码快照控制频率、一个过渡表达式有差异，未运行断言原因 |
| N2 Shafiee等，Viability leads to the emergence of gait transitions…，Nature Communications 2024 | [正式页](https://www.nature.com/articles/s41467-024-47443-w)；[已读作者预印本](https://arxiv.org/pdf/2306.07419)；[项目](https://miladshafiee.github.io/DeepTransition/) | 预印本v2 pp9、12–16、S1；PPO改变CPG幅值/频率/足端偏移，以任务存活解释步态转换 | 正式PDF访问失败，不能混用新旧实验组数；所读实机实验使用等价仿真的已知沟距；不是全机器人可行域定理 |
| N3 Chen等，Constructing custom thermodynamics using deep learning，Nature Computational Science 4,66–85,2024 | [正式页](https://www.nature.com/articles/s43588-023-00581-5)；[作者全文](https://arxiv.org/html/2308.04119v3) | Results/Discussion、低维坐标、Onsager结构、定理2条件、PCA残差编码/似然与敏感性；核对官方示例的PSD/反对称分解 | 微观聚合物轨迹训练比整足本体观测丰富；能量结论有噪声与外功条件，不是机器人稳定证明 |
| S1 Jenelten等，DTC: Deep Tracking Control，Science Robotics 2024 | [DOI](https://doi.org/10.1126/scirobotics.adh5401)；[作者全文v2](https://arxiv.org/html/2309.15462v2)；[作者页](https://juhe9842.github.io/publication/deep-tracking-control/) | Methods、实验/消融、Discussion与相关补充；优化落点参考训练PPO跟踪，按触地/扰动等重规划避免规划器不断迁就弱跟踪 | 使用固定trot；未核验完整训练代码；不能把其大规模并行训练速度当本项目可达性能 |

## B. 材料、MPC、WBC期刊：直接比较对象

| ID / 工作 | 一手来源 | 实际阅读与机制 | 明确边界 |
|---|---|---|---|
| J1 Grandia等，Perceptive Locomotion through Nonlinear Model Predictive Control，T-RO 2023 | [DOI](https://doi.org/10.1109/TRO.2023.3275384)；[全文](https://arxiv.org/pdf/2208.08373) | IV–VII：状态/约束、SQP/实时迭代、事件执行、实机与局限；已有早晚触反馈，联合身体与足端优化 | 外给步态、硬编码接触恢复、每足独立区域选择；作者100Hz/四核条件下的时延不能当机载Go2保证 |
| J2 Fahmi等，STANCE: Locomotion Adaptation over Soft Terrain，T-RO，作者接受稿2019 | [作者全文](https://iit-dlslab.github.io/papers/fahmi19tro.pdf)；[预印本身份](https://arxiv.org/abs/1904.12306) | III–V与实验/结论；形变QP、KV力关系、足加速度相容、刚度相关加载时间、在线估计 | HyQ使用Vicon辅助不漂移足位置；实机只估法向刚度；无线性以外持久跨足迟滞机制。本表年份按作者稿，不混充正式卷期 |
| J3 Rosenkranz等，Viscoelasticty with physics-augmented neural networks…，Computational Mechanics 74,1279–1301,2024 | [正式全文](https://link.springer.com/article/10.1007/s00466-024-02477-1)；[作者全文](https://arxiv.org/html/2401.14270v1) | §§2–4、§5方法比较、§6、Appendix A；能量/耗散、隐式演化、积分/FNN/LSTM三类训练 | 当前直接基线；训练辅助LSTM不等于部署在线世界模型；未确认原作者官方源码仓库 |
| J4 Kalina等，A physics-augmented neural network framework for finite strain incompressible viscoelasticity，CMAME455,118892,2026 | [出版页](https://www.sciencedirect.com/science/article/pii/S0045782526001659)；[作者全文v1](https://arxiv.org/html/2511.02959v1) | §2.4积分、§3势/门控/校准、§4实验/外推、§5；多Maxwell、门控稀疏化、保持结构的隐式指数积分 | 作者版2025、发表2026；不可压弹性体不等于土壤压密；未确认本篇官方模型实现，数据仓库不冒充算法源码 |
| J5 Raj等，A neural-network framework to learn history-dependent constitutive laws and identifiability of internal variables，JMPS217,106807,2026 | [出版页](https://www.sciencedirect.com/science/article/pii/S002250962600308X)；[作者全文](https://arxiv.org/html/2605.14179v1) | §§2–3及§4.3；凸耗散、应力训练、可辨识假设与反例；审阅官方模型文件 | 状态仅在强条件下辨识到等价变换；没有证明整足合力能重建网格真值；未逐行重演附录证明 |
| J6 Jeon等，Residual MPC: Blending Reinforcement Learning with GPU-Parallelized MPC，T-RO2026 | [出版信息](https://ieeexplore.ieee.org/abstract/document/11644391/media)；[实际已读作者v1](https://arxiv.org/html/2510.12717v1)；[DOI](https://doi.org/10.1109/TRO.2026.3721669) | v1 §§III–VI；MPC并行、残差混合、PPO、接触阶段分析与硬件 | 未逐页核对最终期刊版；训练并行步耗时与硬件100Hz不同；MPC残差本身已有 |

## C. 相关顶会与最新预印本

| ID / 工作 | 一手来源 | 实际阅读与机制 | 明确边界 |
|---|---|---|---|
| C1 Margolis & Agrawal，Walk These Ways，CoRL2022 | [全文](https://arxiv.org/pdf/2212.03238)；[官方项目](https://gmargo11.github.io/walk-these-ways/) | §3、§4.2–5；官方接触目标与奖励函数；一个策略覆盖可解释步态/姿态参数 | 行为选择仍需人/脚本；官方快照参数多于文稿，需同版本比较；是同动作权限的重要基线 |
| C2 Georgiev等，PWM，ICLR2025 | [正式proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/240ea1741b205ea295721d55184ac43b-Abstract-Conference.html)；[全文v3](https://arxiv.org/html/2407.02466v3) | §§2–4、式6–10、Appendix C/D；模型正则化、短时梯度与TD(lambda) | 每任务提取策略，非一个通用actor；log空间pseudo-reward改变优化量；未审源码 |
| C3 Amigo等，First Order Model-Based RL through Decoupled Backpropagation（DMO），CoRL2025 | [正式proceedings](https://proceedings.mlr.press/v305/amigo25a.html)；[全文v2](https://arxiv.org/html/2509.00215v2)；[官方项目](https://machines-in-motion.github.io/DMO/) | §3/4、Limitations、A.1.2–1.4；真实仿真前向、模型反向、真实critic；附录autograd实现 | 不修复错误局部Jacobian；需要可微reward；Go2短时任务不等于材料长时信用；未逐行审官方仓库 |
| C4 Liao等，官方SuperMimic，CoRL2026 | [v3全文](https://arxiv.org/pdf/2610.28480v3)；[官方项目](https://supermimic.github.io/)；[官方代码](https://github.com/HybridRobotics/supermimic) | §§2–4、6，附录A/B/C、算法A.1；三个关键官方文件全文；短想象梯度与想象critic | 原任务是G1动作跟踪；训练删除root-velocity push，不代表没有做扰动评估；不能把当前真实任务critic说成官方原法 |
| P1 DynaMimic，2026-11-24预印本 | [全文v1](https://arxiv.org/html/2611.32944v1) | §3.1–3.3、§4.1–4.3、表1–5、§5；接触wrench＋已知动力学，多率积分时重算构型项 | 仅部署actor；未审官方实现；其运动库/机器人验证不能代替材料跨足泛化 |
| P2 Trust What Repeats，2026-11-20预印本 | [全文v1](https://arxiv.org/html/2611.27648v1) | §§III–VII、实验/消融；重复梯度、真实方向检验、逐rollout cap | 作者自己的SuperMimic实现，不作为官方复现依据；无物理机器人外部梯度验证 |
| P3 Riemer等，Calibration of neural viscoelastic models via full-field data，2026-09预印本 | [全文](https://arxiv.org/pdf/2609.03645)；[身份页](https://arxiv.org/abs/2609.03645) | §§4–6；全场位移＋总反力的equilibrium gap、隐式约束、噪声和失败统计 | 需密集位移观测，远强于单足合力；部分初始化失败；不能把接受未收敛近似的做法照搬实时控制 |

## D. 有限范围补充：不计入17篇精读

| 工作 | 取得内容 | 使用范围 |
|---|---|---|
| Flaschel等，Convex neural networks learn generalized standard material models，JMPS200,106103,2025 | [出版页](https://www.sciencedirect.com/science/article/pii/S0022509625000791)与[官方GSMN代码](https://github.com/mflaschel/GSMN)；ETH全文403/429 | 官方代码已有零点规范化、凸势、隐式Newton与敏感度，足以确定这些技术非项目独创；不扩写未读实验 |
| Impact-Aware Robust Convex MPC，RA-L2026 | [官方DOI](https://doi.org/10.1109/LRA.2026.3692086)，摘要/出版信息 | Go2冲击鲁棒控制的待补全文对照；不使用未读推导支撑设计保证 |
| AllGaits，ICRA2025 | [EPFL记录](https://graphsearch.epfl.ch/en/publication/2f2ae3c0-26c7-4167-83c0-deca1db1287d)，摘要 | 检查步态自由切换近邻；不列为已读实现 |

## E. 官方源码固定版本

| 来源 | 固定SHA、已读文件 | 审计结论/限制 |
|---|---|---|
| [SuperMimic](https://github.com/HybridRobotics/supermimic/tree/5fb92297ab4fe3d86629d9f0c635334791bb0d5f) | `5fb92297ab4fe3d86629d9f0c635334791bb0d5f`；`src/supermimic/learners/policy_learner.py`、`wm_learner.py`、`envs/world_model_env.py`全文 | 冻结模型参数但保留输入导数；critic使用想象轨迹；horizon8是8帧7转移。没有安装运行或使用用户复现 |
| [BGS](https://github.com/ihcr/learning_to_adapt/tree/2066cbc6697f327d21a45035cd03b471b7f392fa) | `2066cbc6697f327d21a45035cd03b471b7f392fa`；`src/learning_to_adapt/gait_scheduler.cpp`、`configs/loco_bio_gs_unified.yaml` | 有状态反馈/相位混合；静态快照与论文频率及一个相位过渡比值不同，未运行判断原因 |
| [WTW](https://github.com/Improbable-AI/walk-these-ways/tree/0e7236bdc81ce855cbe3d70345a7899452bdeb1c) | `0e7236bdc81ce855cbe3d70345a7899452bdeb1c`；`go1_gym/envs/base/legged_robot.py`、`envs/rewards/corl_rewards.py`的接触/奖励相关函数 | 连续频率、相位与duration已存在；不把仅扩动作维度当新方法 |
| [Custom Thermodynamics](https://github.com/MLDS-NUS/DeepLearningCustomThermodynamics/tree/c068669704d37b22202f9318755a2de383342f63) | `c068669704d37b22202f9318755a2de383342f63`；`demo/demo_OnsagerNet.py`势与对称/反对称分解、RHS | demo外力网络在RHS乘零，不能因函数存在就称已验证任意驱动；归档实现未运行 |
| [Stable Material Model](https://github.com/m-raj/Learning-Stable-Material-Model/tree/31e22961e1b3878683c9d2bf41509a33f705250d) | `31e22961e1b3878683c9d2bf41509a33f705250d`；`3D_Polyconvex_TaylorData/neural_network_polyconvex.py` | 核对凸层/势/应力与状态步进；存在待核查静态变量命名问题，不能承诺下载即可复现 |
| [GSMN](https://github.com/mflaschel/GSMN/tree/53a7844aa880d024dd6147172bddb2dc833baabc) | `53a7844aa880d024dd6147172bddb2dc833baabc`；`gsmn/Potential_neural_network.py`、`gsmn/GSMN.py` | 凸网络、原点修正、二阶导/隐式Newton/敏感度；代码审阅不冒充论文全文阅读 |

## F. 本项目代码与证据覆盖

固定快照b58f333的主审阅包括：PANN势与训练损失、轴对称混合边界求解；M3候选拟合；W59三维请求hook；W57→r60→r58→r55/r52的相关继承路径；WBC接触与法向参考；WM状态推进/冻结动力学；原始策略密度、真实时间GAE、PPO/想象分支更新。结果身份用每日记录、W59 index及model/action/learning JSON核对。

本轮不是重新播放全部历史视频；关于步态的项目事实引用已发布W58视频/日志审查与W59统计。下一步新模型评测必须按触地、反弹、稳定承载等事件对齐新视频。没有以阅读索引或文件列表冒充每个原始字节已读。
