# W57 文献补充与实验映射（2026-11-22）

检索截至2026-11-22（Asia/Shanghai）。采用论文全文、出版社、作者/机构页面及作者仓库。优先Nature/Science家族，再按问题匹配阅读机器人期刊/顶会；没有核验2025中科院分区，不自行标“一区/Top”。当前稳定实现是接触力QP MPC＋WBC，以下建议据此提出。已读范围以表中为准，代码公开不等于在本项目复现完成。

## 1. 直接影响本轮设计的论文

| 论文、状态与来源 | 实际阅读范围 | 本项目采用的启示 | 适用边界 |
| --- | --- | --- | --- |
| **Learning to adapt through bio-inspired gait strategies for versatile quadruped locomotion**；Nature Machine Intelligence 7, 1141–1153，2025；[出版全文](https://www.nature.com/articles/s42256-025-01065-z)、[作者全文版本](https://arxiv.org/html/2412.09440v3)、[官方代码/数据](https://github.com/ihcr/learning_to_adapt) | 正文方法、步态指标、实验和代码可用性 | 联合评价接触时序、周期变异、身体稳定、能耗与力矩使用；W57据此补足步态质量表 | gait procedural memory不是被踩地块的可恢复材料内态；当前不因此改成多步态选择器 |
| **Learning quadrupedal locomotion on deformable terrain**；Science Robotics 8(74), eade2256，2023；[出版社](https://www.science.org/doi/10.1126/scirobotics.ade2256)、[作者机构记录](https://pure.kaist.ac.kr/en/publications/learning-quadrupedal-locomotion-on-deformable-terrain/) | 作者机构/实验室记录与官方摘要；本次出版社全文未取到 | 软地面模型与本体感知适应应一同验证，训练分布需含足地作用 | 颗粒介质与TM02黏弹材料不同；不由摘要推断其有/无完整持久地块记忆消融 |
| **Residual MPC: Blending Reinforcement Learning with GPU-Parallelized Model Predictive Control**；2025预印本，作者列IEEE T-RO 2026；[全文](https://arxiv.org/html/2510.12717v1)、[正式状态](https://seungwoo-hong.github.io/publications/) | 公开版本架构、SQP/ADMM、训练与性能/局限；出版状态独立核对 | 以可解释控制先验搭配具有真实权限的学习修正；完整闭环训练与批量求解共同考虑 | MIT Humanoid双足、kinodynamic MPC/力矩残差，并非当前Go2接口；CusADi开源不等于整套Residual MPC实验代码已确认公开 |
| **RL-Augmented MPC for Non-Gaited Legged and Hybrid Locomotion**；2026论文；[全文](https://arxiv.org/html/2603.10878v1)、[官方AugMPC](https://github.com/AndrePatri/AugMPC) | 接触调度、观察/动作、训练、并行软件及实验 | RL必须能改变实际接触请求；共享内存和独立MPC实例才支持批量采集 | 作者使用SAC和不同机器人/MPC，不能将其训练预算或数百实例吞吐搬成当前F的承诺；不删除本项目原力矩界 |
| **Load-bearing Assessment for Safe Locomotion of Quadruped Robots on Collapsing Terrain**；RA-L，2025；[全文](https://arxiv.org/html/2510.21369v1)、[作者PDF](https://iit-dlslab.github.io/papers/medeiros25ral.pdf) | 整步力包络、MPC与支撑/探测约束、真实Aliengo实验 | 接触处的未来力需求应与动作和其余足支撑一致；适合F规划层借鉴 | 慢速crawl/承载探测不同于恢复记忆协同；部分软约束有取舍，不是任意探测力都保证实现 |
| **Imitating and Finetuning Model Predictive Control for Robust and Symmetric Quadrupedal Locomotion**；RA-L 2023，公开版本2026更新；[全文](https://arxiv.org/html/2311.02304v3)、[作者项目](https://hyunyoungjung.github.io/projects/ifm/ifm.html) | 方法、运动质量、CoT/关节相图及实验 | 通过任务之外还要量化周期性、效率和对称性；直线与转向分开 | 部署是克隆后策略，不是保留在线MPC；借指标不意味切换本项目架构。PPI如采用需按原定义，不能给自定义相图距离改名 |
| **CusADi: A GPU Parallelization Framework for Symbolic Expressions and Optimal Control**；2024预印本/接收，RA-L 2025卷期；[全文](https://arxiv.org/html/2408.09662v1)、[官方代码](https://github.com/se-hwan/cusadi)、[MIT记录](https://biomimetics.mit.edu/publications/cusadi/) | 符号表达式批处理、数据传输与控制实例 | GPU适合独立材料历史/候选和批量动力学，记录端到端吞吐 | 单函数加速不能外推完整F的QP、WBC和真实材料plant；需要实际批量结构 |
| **DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning**；ICRA 2023；[全文](https://arxiv.org/html/2301.10602v2) | 短历史估计、PPO、种子对照与长距离真实验证 | 有必要区分短时本体历史适应与持久空间记忆，固定任务/训练预算做对照 | 短时latent不是同一地块恢复模型；本次未确认作者完整训练仓库，社区版本另标身份 |

## 2. 最新软地面近邻：已核官方摘要，尚不能替代全文比较

**Terramechanics-Aware Adaptive Quadrupedal Locomotion via Foot-Terrain Interaction Model**，RA-L 11(11):12552–12559，2026，2026-09-10在线，[IEEE原始记录](https://ieeexplore.ieee.org/abstract/document/11685272/)，DOI 10.1109/LRA.2026.3732903。官方摘要明确包含塑性屈服、沉陷、剪切饱和与承载衰减，以及从接触历史估计摩擦/可变形性；报告Isaac Sim、Newton MPM与Go2实机。它是本项目必须认真面对的近邻，不能把“接触历史＋软地面适应”当作空白。本次仅核官方完整摘要和元数据，没有据此断言其全部世界地块/恢复历史实现细节。

**MILD: Tractable Terrain Modeling for Learning Improved Bipedal Locomotion on Deformable Surfaces**，RA-L 11(2):1922–1929，2026，2025-12-17在线，[IEEE记录](https://ieeexplore.ieee.org/document/11302803/)、[HKU机构记录](https://hub.hku.hk/handle/10722/375084)，DOI 10.1109/LRA.2025.3645520。官方摘要包含离散接触建模、潜变量调制与本体估计，以及真实可变刚度表面验证。对象为双足；本次全文未取得，只引用官方摘要内容。arXiv在2026年8月上传不应被误作首次发表时间。

## 3. 从文献到本项目的具体决定

### 步态评价

借用已发表工作的评价维度，先从现有原件得到实际接触图、支撑比、步周期CV、触地/离地时刻误差、足端净空与触地速度、冲量/峰力、关节q–dq相图、身体姿态/高度波动、力矩使用/饱和和机械功。材料段、刚地段、转向、终点保持分别统计；地面适应可能需要不对称，不能将对称程度直接变成普适优劣标准。

机械CoT用合法行进路径长度归一，明确正功或绝对功，不使用往返末净位移，也不把机械功代理称电池能耗。现有滑移代价与新报告滑移距离分别保留原定义。

### 材料记忆的证据

上述文献已经覆盖软地面模型、近期历史适应、步态选择、承载探测与MPC+RL。待验证的主问题应更具体：**同一个世界地块因当前加载改变后，动作条件预测是否能让机器人为后足/后续重访留下更可用的支撑，并获得相同任务约束下的收益。**

首先固定执行链与动作权限，比较完整材料历史与候选私有历史干预，确认预测差异进入候选、首动作和真实接触。再比较有/无显式材料预测的同预算独立学习；保留合法当前观测。用短重访和充分恢复条件检验收益是否随残余历史改变，同时报告时间、路线、距离与负载，排除纯粹慢走和绕开材料造成的混淆。此处是研究设计推导，尚不是已验证成果，也不是“首次”的新颖性证明。

### 计算效率

Residual MPC的100Hz控制设置不等于成千环境的墙钟实时推进；其公开版本对GPU批量求解和端到端训练时延另有测量。AugMPC也依赖具体的CPU实例、共享内存与GPU仿真组织。当前F是不同实现，先让两条物理线和独立材料/候选批次持续产出，再按实际热点优化。96GB是可用容量，不能作为已实现吞吐或实时性的证据。

本补充不要求先换成某篇论文的整套控制器，也不把额外大规模复现变为本轮开始工作的前置条件。可直接执行的接口、比较与资源要求已进入[W57主计划](S1_W57_RESEARCH_PLAN.md)。
