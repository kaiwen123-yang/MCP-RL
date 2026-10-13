# 100 篇方法与理论文献索引 · v2

快照日期：2026-10-13。新版包括 **90 篇正式发表于 2024–2026 的近期论文 + 10 篇经典奠基例外**。2024 年 27 篇、2025 年 33 篇、2026 年 30 篇的统计采用正式刊会年份，不能理解为首次公开年份；已知首稿、在线版或扩展版信息列在各条目中。

结构化书目见 [papers.json](papers.json)，导入文件见 [papers.ris](papers.ris)，筛选口径见 [selection-method.md](selection-method.md)，本轮改版见 [修订说明](../docs/library-v2-revision.md)，旧版见 [v1 历史归档](archive/v1-2026-10-13/README.md)。

当前为 **5 篇初读、95 篇书目与相关性筛选**，尚无已完成复现的条目。初读也不等于全文精读或逐条证明复核。先理解方法与理论，再做聚焦仿真，最后选择机器人应用；创新包括推进已有方法尚未解决的重要问题。

以下八个主题是本仓库的导航目录，互有交叉，不是某篇综述的原始分类。论文既包括直接 MPC–RL 融合，也包括明确服务于该主线的模型学习、规划、优化和控制理论。各条“阅读边界”说明后续需核对的条件，不冒充对所有论文完成了审稿。

期刊的 2025 年中科院大类分区来自公开转录核对，待官方复核；会议不套用期刊分区。2026-10-13 已回读核验 Zotero 主库100条、v2选篇笔记100条、本地PDF覆盖89/100；历史库100条及原有笔记、附件保留。全文取得不等于精读；此处不表示云端附件同步已完成。

## 01 综述与统一框架 · 近期 3 篇

### MR003 · A review of learning-based dynamics models for robotic manipulation

**2025 · Science Robotics**

**选篇理由：** 综述操作任务中的学习动力学、表示和规划接口，为接触模型选择及不确定性提供背景；不是单一MPC+RL算法。

**阅读边界：** 综述与分类提供阅读地图，不能替代原始方法、定理假设与实验的核验；主题目录也不等于某篇综述的原始分类。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2025 年，不能据此断言无更早版本。 正式发表年份已核；是否有更早预印本尚未全面追溯。

[DOI：10.1126/scirobotics.adt1497](https://doi.org/10.1126/scirobotics.adt1497) · [正式论文入口](https://doi.org/10.1126/scirobotics.adt1497) · [书目核验](https://api.crossref.org/works/10.1126%2Fscirobotics.adt1497)

[公开全文来源（有本轮获取记录）](https://albertboai.com/assets/pdf/2025_scirobotics.adt1497.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR004 · Synthesis of model predictive control and reinforcement learning: Survey and classification

**2026 · Annual Reviews in Control**

**选篇理由：** 建立MPC与RL融合机制的完整分类，作为全库导读。

**阅读边界：** 综述与分类提供阅读地图，不能替代原始方法、定理假设与实验的核验；主题目录也不等于某篇综述的原始分类。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 2025预印本；2026正式综述

[DOI：10.1016/j.arcontrol.2026.101045](https://doi.org/10.1016/j.arcontrol.2026.101045) · [正式论文入口](https://doi.org/10.1016/j.arcontrol.2026.101045) · [书目核验](https://api.crossref.org/works/10.1016/j.arcontrol.2026.101045) · [早期版本依据](https://arxiv.org/abs/2502.02133)

[公开全文来源（有本轮获取记录）](https://publications.syscop.de/Reiter2026.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：初读，未复现 · [初读笔记](../notes/MR004-survey.md)。

### MR157 · A view on learning robust goal-conditioned value functions: Interplay between RL and MPC

**2025 · Annual Reviews in Control**

**选篇理由：** 把目标条件全局价值学习与局部场景MPC组合，讨论不确定性下的分工。

**阅读边界：** 综述与分类提供阅读地图，不能替代原始方法、定理假设与实验的核验；主题目录也不等于某篇综述的原始分类。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 已识别同题作者预印本arXiv:2502.06996，最早已核公开年份2025；正式刊会年份为2025。

[DOI：10.1016/j.arcontrol.2025.101027](https://doi.org/10.1016/j.arcontrol.2025.101027) · [正式论文入口](https://doi.org/10.1016/j.arcontrol.2025.101027) · [书目核验](https://api.crossref.org/works/10.1016/j.arcontrol.2025.101027) · [早期版本依据](https://arxiv.org/abs/2502.06996)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2502.06996)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

## 02 MPC与RL的理论联系 · 近期 4 篇

### MR118 · Simultaneous System Identification and Model Predictive Control With No Dynamic Regret

**2025 · IEEE Transactions on Robotics**

**选篇理由：** 随机傅里叶特征近似未知残差，在线最小二乘与MPC同步进行，对控制仿射RKHS未知动态给出相对知晓动态变化的控制器的动态遗憾界。

**阅读边界：** 理论结论须连同模型、约束、终端条件和优化求解假设阅读，不能直接推广为任意系统上的安全或性能保证。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 已核arXiv首稿公开于2024年（2407.04143）；正式发表年份2025。未把晚刊年份当作首次提出年份。

[DOI：10.1109/tro.2025.3576969](https://doi.org/10.1109/tro.2025.3576969) · [正式论文入口](https://doi.org/10.1109/tro.2025.3576969) · [书目核验](https://api.crossref.org/works/10.1109%2FTRO.2025.3576969) · [早期版本依据](https://arxiv.org/abs/2407.04143)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2407.04143)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR144 · Convergence guarantees for adaptive model predictive control with kinky inference

**2024 · 6th Annual Learning for Dynamics & Control Conference (L4DC), PMLR 242**

**选篇理由：** 研究kinky inference更新未知非线性模型时，鲁棒自适应MPC如何保证真实系统收敛。

**阅读边界：** 理论结论须连同模型、约束、终端条件和优化求解假设阅读，不能直接推广为任意系统上的安全或性能保证。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2024 年，不能据此断言无更早版本。 正式PMLR条目已核；尚未核实是否有更早的预印本或会议前稿。

[正式论文入口](https://proceedings.mlr.press/v242/zuliani24a.html) · [书目核验](https://proceedings.mlr.press/v242/zuliani24a.html)

[公开全文来源（有本轮获取记录）](https://proceedings.mlr.press/v242/zuliani24a/zuliani24a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR145 · Mapping back and forth between model predictive control and neural networks

**2024 · 6th Annual Learning for Dynamics & Control Conference (L4DC), PMLR 242**

**选篇理由：** 证明约束线性二次MPC的隐式网络表示并研究显式展开，为学习优化结构提供理论。

**阅读边界：** 理论结论须连同模型、约束、终端条件和优化求解假设阅读，不能直接推广为任意系统上的安全或性能保证。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 已识别同题作者预印本arXiv:2404.12030，最早已核公开年份2024；正式刊会年份为2024。

[正式论文入口](https://proceedings.mlr.press/v242/drummond24a.html) · [书目核验](https://proceedings.mlr.press/v242/drummond24a.html) · [早期版本依据](https://arxiv.org/abs/2404.12030)

[公开全文来源（有本轮获取记录）](https://proceedings.mlr.press/v242/drummond24a/drummond24a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR146 · Increasing information for model predictive control with semi-Markov decision processes

**2024 · 6th Annual Learning for Dynamics & Control Conference (L4DC), PMLR 242**

**选篇理由：** 通过时间抽象提高固定采样预算的信息量，针对学习MPC的探索效率而非应用换场景。

**阅读边界：** 理论结论须连同模型、约束、终端条件和优化求解假设阅读，不能直接推广为任意系统上的安全或性能保证。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 正式PMLR会议论文发表于2024，关联arXiv后补上传于2025；后补预印本日期不能作为首次公开日期。

[正式论文入口](https://proceedings.mlr.press/v242/hosseinkhan-boucher24a.html) · [书目核验](https://proceedings.mlr.press/v242/hosseinkhan-boucher24a.html) · [早期版本依据](https://proceedings.mlr.press/v242/hosseinkhan-boucher24a.html)

[公开全文来源（有本轮获取记录）](https://proceedings.mlr.press/v242/hosseinkhan-boucher24a/hosseinkhan-boucher24a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

## 03 策略与长期价值辅助规划 · 近期 17 篇

### MR024 · CACTO-SL: Using Sobolev learning to improve continuous actor-critic with trajectory optimization

**2024 · Proceedings of the 6th Annual Learning for Dynamics & Control Conference (L4DC), PMLR 242**

**选篇理由：** 用Sobolev学习改进actor-critic与轨迹优化之间的知识传递。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 已识别首稿或早期公开年份：2023。 arXiv首稿2023-12-17（2312.10666），正式L4DC2024；按正式发表年归类。

[正式论文入口](https://proceedings.mlr.press/v242/alboni24a.html) · [书目核验](https://proceedings.mlr.press/v242/alboni24a.html) · [早期版本依据](https://arxiv.org/abs/2312.10666)

[公开全文来源（有本轮获取记录）](https://proceedings.mlr.press/v242/alboni24a/alboni24a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR025 · Reinforcement Learning-Based Model Predictive Control for Discrete-Time Systems

**2024 · IEEE Transactions on Neural Networks and Learning Systems**

**选篇理由：** 以MPC生成策略、RL评价并学习终端价值，分析收敛、可行性和稳定性。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2024 年，不能据此断言无更早版本。 正式发表年份已核；是否有更早预印本尚未全面追溯。

[DOI：10.1109/tnnls.2023.3273590](https://doi.org/10.1109/tnnls.2023.3273590) · [正式论文入口](https://doi.org/10.1109/tnnls.2023.3273590) · [书目核验](https://api.crossref.org/works/10.1109/tnnls.2023.3273590)

全文入口：独立公开全文地址尚未确认；可从上述正式论文入口查询访问条件。

阅读状态：书目与相关性已核验，待初读。

### MR026 · TD-MPC2: Scalable, Robust World Models for Continuous Control

**2024 · ICLR 2024**

**选篇理由：** 扩展TD-MPC的价值学习与模型训练稳定性，可用于研究学习模型规划的规模和多任务泛化。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 已识别首稿或早期公开年份：2023。 2023-10预印本；ICLR2024正式发表

[正式论文入口](https://openreview.net/forum?id=Oxh5CstDJU) · [书目核验](https://openreview.net/forum?id=Oxh5CstDJU) · [早期版本依据](https://arxiv.org/abs/2310.16828)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2310.16828)。全文入口不代表已完成阅读或附件导入。

阅读状态：初读，未复现 · [初读笔记](../notes/MR022-MR026-tdmpc.md)。

### MR027 · Dynamic Non-Prehensile Object Transport via Model-Predictive Reinforcement Learning

**2025 · 2025 IEEE International Conference on Robotics and Automation (ICRA)**

**选篇理由：** 研究预测控制与RL在非抓取动态搬运中的协作。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 2024预印本；ICRA2025正式发表

[DOI：10.1109/icra55743.2025.11127521](https://doi.org/10.1109/icra55743.2025.11127521) · [正式论文入口](https://doi.org/10.1109/icra55743.2025.11127521) · [书目核验](https://api.crossref.org/works/10.1109/icra55743.2025.11127521) · [早期版本依据](https://arxiv.org/abs/2412.00086)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2412.00086)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR028 · PIP-Loco: A Proprioceptive Infinite Horizon Planning Framework for Quadrupedal Robot Locomotion

**2025 · 2025 IEEE International Conference on Robotics and Automation (ICRA)**

**选篇理由：** 结合本体感知规划、价值学习与控制以延长四足规划视野。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 2024预印本；ICRA2025正式发表

[DOI：10.1109/icra55743.2025.11128382](https://doi.org/10.1109/icra55743.2025.11128382) · [正式论文入口](https://doi.org/10.1109/icra55743.2025.11128382) · [书目核验](https://api.crossref.org/works/10.1109/icra55743.2025.11128382) · [早期版本依据](https://arxiv.org/abs/2409.09441)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2409.09441)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR029 · AC4MPC: Actor-Critic Reinforcement Learning for Guiding Model Predictive Control

**2026 · IEEE Transactions on Control Systems Technology**

**选篇理由：** 研究actor初值和critic终端价值如何改善有限次NMPC求解与闭环性能。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 2024预印本；2025在线发表；2026正式卷期，不能当作2026首次提出

[DOI：10.1109/tcst.2025.3620521](https://doi.org/10.1109/tcst.2025.3620521) · [正式论文入口](https://doi.org/10.1109/tcst.2025.3620521) · [书目核验](https://api.crossref.org/works/10.1109/tcst.2025.3620521) · [早期版本依据](https://arxiv.org/abs/2406.03995)

[公开全文来源（有本轮获取记录）](https://publications.syscop.de/Reiter2025a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：初读，未复现 · [初读笔记](../notes/MR029-ac4mpc.md)。

### MR030 · Residual MPC: Blending Reinforcement Learning With GPU-Parallelized Model Predictive Control

**2026 · IEEE Transactions on Robotics**

**选篇理由：** GPU并行MPC与RL在力矩层残差融合，直接研究模型控制、策略补偿和训练吞吐的结合。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 2025预印本；2026期刊版

[DOI：10.1109/tro.2026.3721669](https://doi.org/10.1109/tro.2026.3721669) · [正式论文入口](https://doi.org/10.1109/tro.2026.3721669) · [书目核验](https://api.crossref.org/works/10.1109%2FTRO.2026.3721669) · [早期版本依据](https://arxiv.org/abs/2510.12717)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2510.12717)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR105 · Neural critic learning with accelerated value iteration for nonlinear model predictive control

**2024 · Neural Networks**

**选篇理由：** 用带加速因子的价值迭代和单critic近似替代每个预测区间的优化求解，建立加速因子条件下的收敛/可容许性分析；重点是RL式价值迭代如何实现滚动优化，而不只是增加终端价值。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2024 年，不能据此断言无更早版本。 当前只核到出版社公开年份；待追溯更早预印本，非保证首次提出年份。

[DOI：10.1016/j.neunet.2024.106364](https://doi.org/10.1016/j.neunet.2024.106364) · [正式论文入口](https://doi.org/10.1016/j.neunet.2024.106364) · [书目核验](https://api.crossref.org/works/10.1016%2Fj.neunet.2024.106364)

全文入口：独立公开全文地址尚未确认；可从上述正式论文入口查询访问条件。

阅读状态：书目与相关性已核验，待初读。

### MR107 · Adaptive parameterized model predictive control based on reinforcement learning: A synthesis framework

**2024 · Engineering Applications of Artificial Intelligence**

**选篇理由：** RL在较慢时间尺度联合调整PMPC代价、状态反馈函数、模型与优化配置；通过减少决策变量及在线适应同时处理计算负担和模型失配。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 正式论文2024-07-24在线；原文引用先行Sun et al.工作，当前以完整synthesis framework呈现，具体扩展范围仍需比对。

[DOI：10.1016/j.engappai.2024.109009](https://doi.org/10.1016/j.engappai.2024.109009) · [正式论文入口](https://doi.org/10.1016/j.engappai.2024.109009) · [书目核验](https://api.crossref.org/works/10.1016%2Fj.engappai.2024.109009) · [早期版本依据](https://www.sciencedirect.com/science/article/pii/S0952197624011679)

[公开全文来源（有本轮获取记录）](https://repository.tudelft.nl/file/File_5f230ded-4f1a-456f-b3e4-4337cd8f90ec)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR119 · TD-M(PC)$^2$: Improving Temporal Difference MPC Through Policy Constraint

**2026 · L4DC 2026**

**选篇理由：** 从规划采样策略与价值评估策略的结构失配解释价值高估，并用软策略约束改进TD-MPC系列。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 arXiv首稿2025-02-05（2502.03550），正式L4DC2026；按正式发表年归类。

[正式论文入口](https://proceedings.mlr.press/v331/lin26a.html) · [书目核验](https://proceedings.mlr.press/v331/lin26a.html)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v331/main/assets/lin26a/lin26a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR120 · M^3PC: Test-time Model Predictive Control using Pretrained Masked Trajectory Model

**2025 · ICLR 2025**

**选篇理由：** 利用同一预训练掩码轨迹模型承担策略与动力学角色，在不追加参数训练的情况下用测试时MPC改进动作选择。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 已核arXiv首稿发表于2024年（https://arxiv.org/abs/2412.05675）；正式ICLR 2025主会。归入2024–2026池依据正式会议年，不将首稿年份改写成会议年。

[正式论文入口](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5d8201b37e165a131f5c6c6ad5230f58-Abstract-Conference.html) · [书目核验](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5d8201b37e165a131f5c6c6ad5230f58-Abstract-Conference.html)

[公开全文来源（有本轮获取记录）](https://proceedings.iclr.cc/paper_files/paper/2025/file/5d8201b37e165a131f5c6c6ad5230f58-Paper-Conference.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR122 · Bootstrapped Model Predictive Control

**2025 · ICLR 2025**

**选篇理由：** 让策略模仿MPC再反向引导MPC，结合模型TD更新和惰性重分析，解决独立Actor学习导致的价值不准。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2025 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[正式论文入口](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8636419dea1aa9fbd25fc4248e702da4-Abstract-Conference.html) · [书目核验](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8636419dea1aa9fbd25fc4248e702da4-Abstract-Conference.html)

[公开全文来源（有本轮获取记录）](https://proceedings.iclr.cc/paper_files/paper/2025/file/8636419dea1aa9fbd25fc4248e702da4-Paper-Conference.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR130 · Model-based Reinforcement Learning for Parameterized Action Spaces

**2024 · ICML 2024**

**选篇理由：** 学习离散—连续参数化动作条件动力学并修改MPPI，扩展MPC与RL组合的动作空间及误差分析。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2024 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[正式论文入口](https://proceedings.mlr.press/v235/zhang24r.html) · [书目核验](https://proceedings.mlr.press/v235/zhang24r.html)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v235/main/assets/zhang24r/zhang24r.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR136 · The Surprising Difficulty of Search in Model-Based Reinforcement Learning

**2026 · ICML 2026**

**选篇理由：** 分析准确模型下搜索仍损害控制的现象，将规划器引起的价值高估作为核心原因，并以价值集成改进搜索。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 已识别首稿或早期公开年份：2026。 arXiv首稿2026-01-29（2601.21306），正式ICML2026。

[正式论文入口](https://proceedings.mlr.press/v306/chang26a.html) · [书目核验](https://proceedings.mlr.press/v306/chang26a.html)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v306/main/assets/chang26a/chang26a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR142 · The Edge-of-Reach Problem in Offline Model-Based Reinforcement Learning

**2024 · NeurIPS 2024**

**选篇理由：** 揭示截断模型rollout末端缺乏价值监督可导致无误差模型下的价值高估，直接支撑短时域预测与价值衔接的分析。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2024 年，不能据此断言无更早版本。 NeurIPS2024主会；研究模型rollout与价值估计的理论衔接，不是独立MPC实现。

[DOI：10.52202/079017-2012](https://doi.org/10.52202/079017-2012) · [正式论文入口](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7314e20a73542bbfff25030d1185ce88-Abstract-Conference.html) · [书目核验](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7314e20a73542bbfff25030d1185ce88-Abstract-Conference.html)

[公开全文来源（有本轮获取记录）](https://proceedings.neurips.cc/paper_files/paper/2024/file/7314e20a73542bbfff25030d1185ce88-Paper-Conference.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR156 · A characterization method of terminal ingredients for nonlinear MPC using value-based reinforcement learning

**2026 · Automatica**

**选篇理由：** 利用基于价值的RL刻画非线性MPC终端条件，重点在吸引域与在线计算的联动。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2026 年，不能据此断言无更早版本。 正式出版元数据已核；最早公开时间仍待核对。

[DOI：10.1016/j.automatica.2025.112574](https://doi.org/10.1016/j.automatica.2025.112574) · [正式论文入口](https://doi.org/10.1016/j.automatica.2025.112574) · [书目核验](https://api.crossref.org/works/10.1016/j.automatica.2025.112574)

全文入口：独立公开全文地址尚未确认；可从上述正式论文入口查询访问条件。

阅读状态：书目与相关性已核验，待初读。

### MR167 · Learning soft constrained MPC value functions: Efficient MPC design and implementation providing stability and safety guarantees

**2024 · 6th Annual Learning for Dynamics & Control Conference (L4DC), PMLR 242**

**选篇理由：** 学习Lipschitz软约束MPC价值，结合约束收紧获得针对价值误差的ISS与安全保证。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 已核arXiv:2401.07780首次公开于2024；正式刊会年份2024。

[正式论文入口](https://proceedings.mlr.press/v242/chatzikiriakos24a.html) · [书目核验](https://proceedings.mlr.press/v242/chatzikiriakos24a.html) · [早期版本依据](https://arxiv.org/abs/2401.07780)

[公开全文来源（有本轮获取记录）](https://proceedings.mlr.press/v242/chatzikiriakos24a/chatzikiriakos24a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

## 04 学习模型与不确定性 · 近期 19 篇

### MR048 · Learning Agile Locomotion and Adaptive Behaviors via RL-augmented MPC

**2024 · 2024 IEEE International Conference on Robotics and Automation (ICRA)**

**选篇理由：** 通过RL增强MPC实现四足支撑与摆腿的适应控制。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 已识别首稿或早期公开年份：2023。 2023-10预印本；ICRA2024正式发表

[DOI：10.1109/icra57147.2024.10610453](https://doi.org/10.1109/icra57147.2024.10610453) · [正式论文入口](https://doi.org/10.1109/icra57147.2024.10610453) · [书目核验](https://api.crossref.org/works/10.1109/icra57147.2024.10610453) · [早期版本依据](https://arxiv.org/abs/2310.09442)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2310.09442)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR049 · Calibrated Value-Aware Model Learning with Probabilistic Environment Models

**2025 · ICML 2025**

**选篇理由：** 分析常用价值感知损失的非校准问题并给出修正，适合研究潜模型与终端价值联合学习为何会失真。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 2025预印本与ICML2025正式论文

[正式论文入口](https://proceedings.mlr.press/v267/voelcker25a.html) · [书目核验](https://proceedings.mlr.press/v267/voelcker25a.html) · [早期版本依据](https://arxiv.org/abs/2505.22772)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2505.22772)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR106 · Recurrent Convex Difference Neural Networks for Safety-Critical Model Predictive Control

**2025 · IEEE Robotics and Automation Letters**

**选篇理由：** 构造循环凸差神经模型服务安全关键MPC，研究学习模型表达能力与可优化结构的兼容。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2025 年，不能据此断言无更早版本。 当前只核到出版社公开年份；待追溯更早预印本，非保证首次提出年份。

[DOI：10.1109/lra.2025.3568315](https://doi.org/10.1109/lra.2025.3568315) · [正式论文入口](https://doi.org/10.1109/lra.2025.3568315) · [书目核验](https://api.crossref.org/works/10.1109%2FLRA.2025.3568315)

全文入口：独立公开全文地址尚未确认；可从上述正式论文入口查询访问条件。

阅读状态：书目与相关性已核验，待初读。

### MR108 · Lifelong Learning-Based MPC for Uncertain Multi-Task Manipulation

**2026 · IEEE Robotics and Automation Letters**

**选篇理由：** 以混合专家持续学习任务接触模型，依据预测误差与分布外接近程度选择专家和调整MPC柔顺性，针对新任务适应及遗忘。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2026 年，不能据此断言无更早版本。 当前只核到出版社公开年份；待追溯更早预印本，非保证首次提出年份。

[DOI：10.1109/lra.2026.3668586](https://doi.org/10.1109/lra.2026.3668586) · [正式论文入口](https://doi.org/10.1109/lra.2026.3668586) · [书目核验](https://api.crossref.org/works/10.1109%2FLRA.2026.3668586)

全文入口：独立公开全文地址尚未确认；可从上述正式论文入口查询访问条件。

阅读状态：书目与相关性已核验，待初读。

### MR109 · Variable-Frequency Model Learning and Predictive Control for Jumping Maneuvers on Legged Robots

**2025 · IEEE Robotics and Automation Letters**

**选篇理由：** 从跳跃数据学习模型并为接触/腾空阶段采用不同预测频率，研究短计算预算下覆盖长腾空时段和模型失配。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 已核arXiv首稿公开于2024年（2407.14749）；正式发表年份2025。未把晚刊年份当作首次提出年份。

[DOI：10.1109/lra.2024.3519864](https://doi.org/10.1109/lra.2024.3519864) · [正式论文入口](https://doi.org/10.1109/lra.2024.3519864) · [书目核验](https://api.crossref.org/works/10.1109%2FLRA.2024.3519864) · [早期版本依据](https://arxiv.org/abs/2407.14749)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2407.14749)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR110 · Adaptive Legged Locomotion via Online Learning for Model Predictive Control

**2026 · IEEE Robotics and Automation Letters**

**选篇理由：** 以随机傅里叶特征在线近似残差，在线最小二乘与MPC交互并给出动态遗憾分析；2025 SSI-MPC理论的足式扩展，研究时变负载/摩擦下的无离线训练适应。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 已核arXiv首稿公开于2025年（2510.15626）；正式发表年份2026。未把晚刊年份当作首次提出年份。

[DOI：10.1109/lra.2025.3644161](https://doi.org/10.1109/lra.2025.3644161) · [正式论文入口](https://doi.org/10.1109/lra.2025.3644161) · [书目核验](https://api.crossref.org/works/10.1109%2FLRA.2025.3644161) · [早期版本依据](https://arxiv.org/abs/2510.15626)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2510.15626)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR112 · Compound Learning-Based Model Predictive Control Approach for Ducted-Fan Aerial Vehicles

**2025 · IEEE Transactions on Neural Networks and Learning Systems**

**选篇理由：** 先学习名义动力学，再由在线RL生成MPC控制序列初值并滚动优化，同时讨论稳定性和递归可行性。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 正式卷期2025；作者机构记录核实2024年已作为in-press正式论文公开。未发现更早预印本。

[DOI：10.1109/tnnls.2024.3422401](https://doi.org/10.1109/tnnls.2024.3422401) · [正式论文入口](https://doi.org/10.1109/tnnls.2024.3422401) · [书目核验](https://api.crossref.org/works/10.1109%2FTNNLS.2024.3422401) · [早期版本依据](https://omega.polsl.pl/info/article/PSL82892c3c73be44528f23c931ec3ab723/)

全文入口：独立公开全文地址尚未确认；可从上述正式论文入口查询访问条件。

阅读状态：书目与相关性已核验，待初读。

### MR113 · Meta-Learning Enhanced Model Predictive Contouring Control for Agile and Precise Quadrotor Flight

**2025 · IEEE Transactions on Robotics**

**选篇理由：** 通过元学习提高气动模型对速度变化的适应能力，并服务高动态轮廓MPC，关注模型适应与轨迹优化的耦合。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2025 年，不能据此断言无更早版本。 当前只核到出版社公开年份；待追溯更早预印本，非保证首次提出年份。

[DOI：10.1109/tro.2025.3567491](https://doi.org/10.1109/tro.2025.3567491) · [正式论文入口](https://doi.org/10.1109/tro.2025.3567491) · [书目核验](https://api.crossref.org/works/10.1109%2FTRO.2025.3567491)

全文入口：独立公开全文地址尚未确认；可从上述正式论文入口查询访问条件。

阅读状态：书目与相关性已核验，待初读。

### MR116 · Controlling diverse robots by inferring Jacobian fields with deep networks

**2025 · Nature**

**选篇理由：** 从图像学习机器人表面的Jacobian场，通过优化把图像/三维目标运动转换为控制，并在MPC中闭环跟踪；支持感知到可控模型的接口，非严格RL。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2025 年，不能据此断言无更早版本。 当前只核到出版社公开年份；待追溯更早预印本，非保证首次提出年份。

[DOI：10.1038/s41586-025-09170-0](https://doi.org/10.1038/s41586-025-09170-0) · [正式论文入口](https://doi.org/10.1038/s41586-025-09170-0) · [书目核验](https://api.crossref.org/works/10.1038%2Fs41586-025-09170-0)

[公开全文来源（已获取并核验题名）](https://www.nature.com/articles/s41586-025-09170-0.pdf)。

阅读状态：书目与相关性已核验，待初读。

### MR121 · Discrete Codebook World Models for Continuous Control

**2025 · ICLR 2025**

**选篇理由：** 以随机离散码本潜状态配合决策时规划，直接比较不同潜表示对连续控制MPC的作用。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2025 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[正式论文入口](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1dd85d064697ec1a258cd2b8755b8c6d-Abstract-Conference.html) · [书目核验](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1dd85d064697ec1a258cd2b8755b8c6d-Abstract-Conference.html)

[公开全文来源（有本轮获取记录）](https://proceedings.iclr.cc/paper_files/paper/2025/file/1dd85d064697ec1a258cd2b8755b8c6d-Paper-Conference.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR123 · Bisimulation Metric for Model Predictive Control

**2025 · ICLR 2025**

**选篇理由：** 将双模拟度量用于编码器训练，研究任务相关表示对TD-MPC训练稳定性、抗噪性与计算成本的影响。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2025 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[正式论文入口](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ea0206fdf3afc2ff0578a230816a9e15-Abstract-Conference.html) · [书目核验](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ea0206fdf3afc2ff0578a230816a9e15-Abstract-Conference.html)

[公开全文来源（有本轮获取记录）](https://proceedings.iclr.cc/paper_files/paper/2025/file/ea0206fdf3afc2ff0578a230816a9e15-Paper-Conference.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR125 · COPlanner: Plan to Roll Out Conservatively but to Explore Optimistically for Model-Based RL

**2024 · ICLR 2024**

**选篇理由：** 用不确定性感知的策略引导MPC分别做保守模型rollout和乐观真实探索，改变模型数据与控制数据的生成机制。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 已识别首稿或早期公开年份：2023。 已核arXiv首稿发表于2023年（https://arxiv.org/abs/2310.07220）；正式ICLR 2024主会。归入2024–2026池依据正式会议年，不将首稿年份改写成会议年。

[正式论文入口](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4a32a646254d2e37fc74a38d65796552-Abstract-Conference.html) · [书目核验](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4a32a646254d2e37fc74a38d65796552-Abstract-Conference.html)

[公开全文来源（有本轮获取记录）](https://proceedings.iclr.cc/paper_files/paper/2024/file/4a32a646254d2e37fc74a38d65796552-Paper-Conference.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR126 · Continual Reinforcement Learning by Planning with Online World Models

**2025 · ICML 2025**

**选篇理由：** 在线增量学习世界动力学并用MPC处理任务变化，直接研究持续学习中的遗忘、模型更新与规划分工。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2025 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[正式论文入口](https://proceedings.mlr.press/v267/liu25p.html) · [书目核验](https://proceedings.mlr.press/v267/liu25p.html)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v267/main/assets/liu25p/liu25p.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR127 · DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

**2025 · ICML 2025**

**选篇理由：** 预测预训练视觉特征并在线优化动作序列，用目标特征定义任务，研究无奖励建模的视觉MPC。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 已核arXiv首稿发表于2024年（https://arxiv.org/abs/2411.04983）；正式ICML 2025主会。归入2024–2026池依据正式会议年，不将首稿年份改写成会议年。

[正式论文入口](https://proceedings.mlr.press/v267/zhou25t.html) · [书目核验](https://proceedings.mlr.press/v267/zhou25t.html)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v267/main/assets/zhou25t/zhou25t.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR129 · Trajectory World Models for Heterogeneous Environments

**2025 · ICML 2025**

**选篇理由：** 用可适应不同传感器与执行器维度的轨迹世界模型改善下游MPC，研究异构任务间模型迁移的实际控制收益。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2025 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[正式论文入口](https://proceedings.mlr.press/v267/yin25f.html) · [书目核验](https://proceedings.mlr.press/v267/yin25f.html)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v267/main/assets/yin25f/yin25f.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR133 · Learning Massively Multitask World Models for Continuous Control

**2026 · ICLR 2026**

**选篇理由：** Newt明确建立于TD-MPC2，以示范预训练、语言条件和多任务在线交互推进策略先验、价值与规划的协同学习。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2026 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[正式论文入口](https://proceedings.iclr.cc/paper_files/paper/2026/hash/092359ce5cf60a80e882378944bf1be4-Abstract-Conference.html) · [书目核验](https://proceedings.iclr.cc/paper_files/paper/2026/hash/092359ce5cf60a80e882378944bf1be4-Abstract-Conference.html)

[公开全文来源（有本轮获取记录）](https://proceedings.iclr.cc/paper_files/paper/2026/file/092359ce5cf60a80e882378944bf1be4-Paper-Conference.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR138 · Reflect-then-Plan: Offline Model-Based Planning through a Doubly Bayesian Lens

**2025 · ICML 2025**

**选篇理由：** 部署时依据新观测更新动力学信念并将不确定性边缘化到规划中，提高离线学习策略对动态变化的适应。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2025 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[正式论文入口](https://proceedings.mlr.press/v267/jeong25a.html) · [书目核验](https://proceedings.mlr.press/v267/jeong25a.html)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v267/main/assets/jeong25a/jeong25a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR155 · Choose Wisely: Data-driven Predictive Control for Nonlinear Systems Using Online Data Selection

**2026 · Annual Learning for Dynamics and Control Conference (L4DC), PMLR 331**

**选篇理由：** 按当前控制问题选择相关数据并在轨迹空间隐式线性化，降低非线性数据驱动MPC失配。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 已识别同题作者预印本arXiv:2503.18845，最早已核公开年份2025；正式刊会年份为2026。

[正式论文入口](https://proceedings.mlr.press/v331/naf26a.html) · [书目核验](https://proceedings.mlr.press/v331/naf26a.html) · [早期版本依据](https://arxiv.org/abs/2503.18845)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v331/main/assets/naf26a/naf26a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR158 · MAKO: Meta-Adaptive Koopman Operators for learning-based model predictive control of parametrically uncertain nonlinear systems

**2026 · Automatica**

**选篇理由：** 通过元学习Koopman模型在线适应未见参数，并分析模型适应与闭环收敛；不是严格RL。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 已识别同题作者预印本arXiv:2510.09042，最早已核公开年份2025；正式刊会年份为2026。

[DOI：10.1016/j.automatica.2026.112827](https://doi.org/10.1016/j.automatica.2026.112827) · [正式论文入口](https://doi.org/10.1016/j.automatica.2026.112827) · [书目核验](https://api.crossref.org/works/10.1016/j.automatica.2026.112827) · [早期版本依据](https://arxiv.org/abs/2510.09042)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2510.09042)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

## 05 安全约束与稳定性 · 近期 9 篇

### MR124 · SafeDreamer: Safe Reinforcement Learning with World Models

**2024 · ICLR 2024**

**选篇理由：** 在Dreamer世界模型规划中引入拉格朗日方法与安全成本，研究学习模型下的性能—约束权衡。

**阅读边界：** 须区分训练安全、部署安全、可行性与稳定性，并检查其模型误差和初始条件假设；不同保证不能相互替代。

**版本与首稿：** 已识别首稿或早期公开年份：2023。 已核arXiv首稿发表于2023年（https://arxiv.org/abs/2307.07176）；正式ICLR 2024主会。归入2024–2026池依据正式会议年，不将首稿年份改写成会议年。

[正式论文入口](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ece182f93af26c64187ba3f7dfd4309a-Abstract-Conference.html) · [书目核验](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ece182f93af26c64187ba3f7dfd4309a-Abstract-Conference.html)

[公开全文来源（有本轮获取记录）](https://proceedings.iclr.cc/paper_files/paper/2024/file/ece182f93af26c64187ba3f7dfd4309a-Paper-Conference.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR143 · Safe learning in nonlinear model predictive control

**2024 · 6th Annual Learning for Dynamics & Control Conference (L4DC), PMLR 242**

**选篇理由：** 把基函数参数在线适应、逐次凸化和管约束结合，给出可行性、稳定性及性能保证。

**阅读边界：** 须区分训练安全、部署安全、可行性与稳定性，并检查其模型误差和初始条件假设；不同保证不能相互替代。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2024 年，不能据此断言无更早版本。 正式PMLR条目已核；尚未核实是否有更早的预印本或会议前稿。

[正式论文入口](https://proceedings.mlr.press/v242/buerger24a.html) · [书目核验](https://proceedings.mlr.press/v242/buerger24a.html)

[公开全文来源（有本轮获取记录）](https://proceedings.mlr.press/v242/buerger24a/buerger24a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR147 · Learning for Layered Safety-Critical Control with Predictive Control Barrier Functions

**2025 · 7th Annual Learning for Dynamics & Control Conference (L4DC), PMLR 283**

**选篇理由：** 学习全阶rollout带来的鲁棒修正项，弥合降阶安全规划与全阶跟踪的模型差异；不是标准MPC求解器。

**阅读边界：** 须区分训练安全、部署安全、可行性与稳定性，并检查其模型误差和初始条件假设；不同保证不能相互替代。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 已识别同题作者预印本arXiv:2412.04658，最早已核公开年份2024；正式刊会年份为2025。

[正式论文入口](https://proceedings.mlr.press/v283/compton25a.html) · [书目核验](https://proceedings.mlr.press/v283/compton25a.html) · [早期版本依据](https://arxiv.org/abs/2412.04658)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v283/main/assets/compton25a/compton25a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR159 · Safety Filtering While Training: Improving the Performance and Sample Efficiency of Reinforcement Learning Agents

**2025 · IEEE Robotics and Automation Letters**

**选篇理由：** 研究安全过滤器进入训练过程后对RL性能、样本效率和控制抖动的影响。

**阅读边界：** 须区分训练安全、部署安全、可行性与稳定性，并检查其模型误差和初始条件假设；不同保证不能相互替代。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 已识别同题作者预印本arXiv:2410.11671，最早已核公开年份2024；正式刊会年份为2025。

[DOI：10.1109/lra.2024.3512374](https://doi.org/10.1109/lra.2024.3512374) · [正式论文入口](https://doi.org/10.1109/lra.2024.3512374) · [书目核验](https://api.crossref.org/works/10.1109/lra.2024.3512374) · [早期版本依据](https://arxiv.org/abs/2410.11671)

[公开全文来源（有本轮获取记录）](https://www.dynsyslab.org/wp-content/papercite-data/pdf/pizarro-ral25.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR161 · Shielded Planning Guided Data-Efficient and Safe Reinforcement Learning

**2025 · IEEE Transactions on Neural Networks and Learning Systems**

**选篇理由：** 以MPPI屏蔽规划指导安全探索和数据收集，联合策略优化及潜在动力学改进，分析训练安全与策略性能界。

**阅读边界：** 须区分训练安全、部署安全、可行性与稳定性，并检查其模型误差和初始条件假设；不同保证不能相互替代。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2025 年，不能据此断言无更早版本。 正式出版元数据已核；最早公开时间仍待核对。

[DOI：10.1109/tnnls.2024.3359031](https://doi.org/10.1109/tnnls.2024.3359031) · [正式论文入口](https://doi.org/10.1109/tnnls.2024.3359031) · [书目核验](https://api.crossref.org/works/10.1109/tnnls.2024.3359031)

[公开全文来源（有本轮获取记录）](http://staff.ustc.edu.cn/~zkan/Papers/Journals/%5B65%5D_2025TNNLS.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR162 · Stability Mechanisms for Predictive Safety Filters

**2024 · 2024 IEEE 63rd Conference on Decision and Control (CDC)**

**选篇理由：** 研究预测安全过滤器中促进收敛的稳定机制，区分安全和稳定。

**阅读边界：** 须区分训练安全、部署安全、可行性与稳定性，并检查其模型误差和初始条件假设；不同保证不能相互替代。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 已识别同题作者预印本arXiv:2404.05496，最早已核公开年份2024；正式刊会年份为2024。

[DOI：10.1109/cdc56724.2024.10886293](https://doi.org/10.1109/cdc56724.2024.10886293) · [正式论文入口](https://doi.org/10.1109/cdc56724.2024.10886293) · [书目核验](https://api.crossref.org/works/10.1109/cdc56724.2024.10886293) · [早期版本依据](https://arxiv.org/abs/2404.05496)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2404.05496)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR165 · Safety Beyond the Training Data: Robust Out-of-Distribution MPC via Conformalized System Level Synthesis

**2026 · Annual Learning for Dynamics and Control Conference (L4DC), PMLR 331**

**选篇理由：** 结合共形预测和系统级综合，将学习模型的状态动作相关误差界纳入鲁棒MPC。

**阅读边界：** 须区分训练安全、部署安全、可行性与稳定性，并检查其模型误差和初始条件假设；不同保证不能相互替代。

**版本与首稿：** 已识别首稿或早期公开年份：2026。 已识别同题作者预印本arXiv:2602.12047，最早已核公开年份2026；正式刊会年份为2026。

[正式论文入口](https://proceedings.mlr.press/v331/srinivasan26a.html) · [书目核验](https://proceedings.mlr.press/v331/srinivasan26a.html) · [早期版本依据](https://arxiv.org/abs/2602.12047)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v331/main/assets/srinivasan26a/srinivasan26a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR170 · Dynamic Model Predictive Shielding for Provably Safe Reinforcement Learning

**2024 · Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Main Conference Track**

**选篇理由：** 在安全屏蔽中动态规划兼顾回报的恢复动作，策略和规划器相互学习，并分析恢复遗憾。

**阅读边界：** 须区分训练安全、部署安全、可行性与稳定性，并检查其模型误差和初始条件假设；不同保证不能相互替代。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 已核arXiv:2405.13863首次公开于2024；正式刊会年份2024。

[DOI：10.52202/079017-3177](https://doi.org/10.52202/079017-3177) · [正式论文入口](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b589d92785e39486e978fa273d0dc343-Abstract-Conference.html) · [书目核验](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b589d92785e39486e978fa273d0dc343-Abstract-Conference.html) · [早期版本依据](https://arxiv.org/abs/2405.13863)

[公开全文来源（有本轮获取记录）](https://proceedings.neurips.cc/paper_files/paper/2024/file/b589d92785e39486e978fa273d0dc343-Paper-Conference.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR171 · Safe, Out-of-Distribution-Adaptive MPC with Conformalized Neural Network Ensembles

**2025 · 7th Annual Learning for Dynamics & Control Conference (L4DC), PMLR 283**

**选篇理由：** 用共形校准神经集成检测预测失配，并在分布外转入可达集控制，区分学习预测性能与最坏情形安全。

**阅读边界：** 须区分训练安全、部署安全、可行性与稳定性，并检查其模型误差和初始条件假设；不同保证不能相互替代。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 2024-06-04作者预印本；2025 L4DC正式会议。

[正式论文入口](https://proceedings.mlr.press/v283/contreras25a.html) · [书目核验](https://proceedings.mlr.press/v283/contreras25a.html) · [早期版本依据](https://arxiv.org/abs/2406.02436)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v283/main/assets/contreras25a/contreras25a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

## 06 采样优化与实时计算 · 近期 15 篇

### MR077 · Deep Model Predictive Optimization

**2024 · ICRA 2024**

**选篇理由：** DMPO把MPC优化器及warm-start作为可学习决策过程，用RL改善有限计算预算下的优化。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 已识别首稿或早期公开年份：2023。 2023-10预印本；ICRA2024正式发表

[DOI：10.1109/ICRA57147.2024.10611492](https://doi.org/10.1109/ICRA57147.2024.10611492) · [正式论文入口](https://doi.org/10.1109/ICRA57147.2024.10611492) · [书目核验](https://doi.org/10.1109/ICRA57147.2024.10611492) · [早期版本依据](https://arxiv.org/abs/2310.04590)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2310.04590)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR078 · Learning a Generalizable Trajectory Sampling Distribution for Model Predictive Control

**2024 · IEEE Transactions on Robotics**

**选篇理由：** 研究可泛化的轨迹采样先验如何帮助MPC，适合分析学习分布在新问题上的可靠性。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2024 年，不能据此断言无更早版本。 正式发表年份已核；是否有更早预印本尚未全面追溯。

[DOI：10.1109/TRO.2024.3370026](https://doi.org/10.1109/TRO.2024.3370026) · [正式论文入口](https://doi.org/10.1109/TRO.2024.3370026) · [书目核验](https://doi.org/10.1109/TRO.2024.3370026)

全文入口：独立公开全文地址尚未确认；可从上述正式论文入口查询访问条件。

阅读状态：书目与相关性已核验，待初读。

### MR079 · RT-HCP: Dealing with Inference Delays and Sample Efficiency to Learn Directly on Robotic Platforms

**2025 · 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)**

**选篇理由：** 结合物理先验、actor-critic与动作段预测控制处理真实推理延迟。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 2025预印本与IROS2025正式论文

[DOI：10.1109/iros60139.2025.11247277](https://doi.org/10.1109/iros60139.2025.11247277) · [正式论文入口](https://doi.org/10.1109/iros60139.2025.11247277) · [书目核验](https://api.crossref.org/works/10.1109/iros60139.2025.11247277) · [早期版本依据](https://arxiv.org/abs/2509.06714)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2509.06714)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR080 · SIT-LMPC: Safe Information-Theoretic Learning Model Predictive Control for Iterative Tasks

**2026 · IEEE Robotics and Automation Letters**

**选篇理由：** 安全信息论学习MPC将迭代经验与采样控制结合，为性能改进与安全可行性提供方法；学习并非必然神经RL。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2026 年，不能据此断言无更早版本。 正式发表年份已核；是否有更早预印本尚未全面追溯。

[DOI：10.1109/lra.2025.3634881](https://doi.org/10.1109/lra.2025.3634881) · [正式论文入口](https://doi.org/10.1109/lra.2025.3634881) · [书目核验](https://api.crossref.org/works/10.1109%2FLRA.2025.3634881)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2602.16187)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR111 · Primal-Dual iLQR for GPU-Accelerated Learning and Control in Legged Robots

**2026 · IEEE Robotics and Automation Letters**

**选篇理由：** primal-dual iLQR以并行扫描求解KKT系统，服务大批量机器人MPC与学习；是融合训练吞吐的优化基础，不把求解器本身称为RL。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 已核arXiv首稿公开于2025年（2506.07823）；正式发表年份2026。未把晚刊年份当作首次提出年份。

[DOI：10.1109/lra.2025.3632610](https://doi.org/10.1109/lra.2025.3632610) · [正式论文入口](https://doi.org/10.1109/lra.2025.3632610) · [书目核验](https://api.crossref.org/works/10.1109%2FLRA.2025.3632610) · [早期版本依据](https://arxiv.org/abs/2506.07823)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2506.07823)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR131 · Parallel Stochastic Gradient-Based Planning for World Models

**2026 · ICML 2026**

**选篇理由：** 以虚拟状态和软动力学约束并行优化学习模型中的长轨迹，直接针对CEM与梯度规划的效率及局部极值。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2026 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[正式论文入口](https://proceedings.mlr.press/v306/psenka26a.html) · [书目核验](https://proceedings.mlr.press/v306/psenka26a.html)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v306/main/assets/psenka26a/psenka26a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR132 · Sparse Imagination for Efficient Visual World Model Planning

**2026 · ICLR 2026**

**选篇理由：** 在视觉模型rollout时稀疏处理token以减少在线规划开销，并检验相同任务下效率和控制性能的权衡。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2026 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[正式论文入口](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a750d52284ff70c6d6bab8072c392d74-Abstract-Conference.html) · [书目核验](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a750d52284ff70c6d6bab8072c392d74-Abstract-Conference.html)

[公开全文来源（有本轮获取记录）](https://proceedings.iclr.cc/paper_files/paper/2026/file/a750d52284ff70c6d6bab8072c392d74-Paper-Conference.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR135 · Model-Based Diffusion Sampling for Predictive Control in Offline Decision Making

**2026 · ICML 2026**

**选篇理由：** MPDiffuser在采样中交替进行规划扩散与动力学扩散更新，纠正离线轨迹生成的动力学不可行性并保留任务意图。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2026 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[正式论文入口](https://proceedings.mlr.press/v306/balim26a.html) · [书目核验](https://proceedings.mlr.press/v306/balim26a.html)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v306/main/assets/balim26a/balim26a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR137 · Dream-MPC: Gradient-Based Model Predictive Control with Latent Imagination

**2026 · ICML 2026**

**选篇理由：** 从策略rollout生成少量候选，以梯度、不确定性正则和跨时刻迭代复用改进MPC，直接对照高成本采样规划。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2026 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[正式论文入口](https://proceedings.mlr.press/v306/spieler26a.html) · [书目核验](https://proceedings.mlr.press/v306/spieler26a.html)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v306/main/assets/spieler26a/spieler26a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR139 · DiffuserLite: Towards Real-time Diffusion Planning

**2024 · NeurIPS 2024**

**选篇理由：** 以粗到细的规划细化减少扩散轨迹中的冗余建模和采样成本，直接研究生成式规划的在线决策频率。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2024 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[DOI：10.52202/079017-3895](https://doi.org/10.52202/079017-3895) · [正式论文入口](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dd6a47bc0aad6f34aa5e77706d90cdc4-Abstract-Conference.html) · [书目核验](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dd6a47bc0aad6f34aa5e77706d90cdc4-Abstract-Conference.html)

[公开全文来源（有本轮获取记录）](https://proceedings.neurips.cc/paper_files/paper/2024/file/dd6a47bc0aad6f34aa5e77706d90cdc4-Paper-Conference.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR140 · Model-based Diffusion for Trajectory Optimization

**2024 · NeurIPS 2024**

**选篇理由：** 利用已知动力学显式构造扩散score求解轨迹优化，并可接入不同质量的示范，是学习先验与模型优化组合的重要对照。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2024 年，不能据此断言无更早版本。 NeurIPS2024主会；本身是可结合示范的模型优化方法，作为方法对照纳入，不冒充RL算法。

[DOI：10.52202/079017-1846](https://doi.org/10.52202/079017-1846) · [正式论文入口](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6a5c38a730b17f3827c09cf6b192be04-Abstract-Conference.html) · [书目核验](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6a5c38a730b17f3827c09cf6b192be04-Abstract-Conference.html)

[公开全文来源（有本轮获取记录）](https://proceedings.neurips.cc/paper_files/paper/2024/file/6a5c38a730b17f3827c09cf6b192be04-Paper-Conference.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR149 · Toward Near-Globally Optimal Nonlinear Model Predictive Control via Diffusion Models

**2025 · 7th Annual Learning for Dynamics & Control Conference (L4DC), PMLR 283**

**选篇理由：** 学习多模态局部最优控制序列分布，再通过在线采样选择近全局方案。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 已识别同题作者预印本arXiv:2412.08278，最早已核公开年份2024；正式刊会年份为2025。

[正式论文入口](https://proceedings.mlr.press/v283/huang25a.html) · [书目核验](https://proceedings.mlr.press/v283/huang25a.html) · [早期版本依据](https://arxiv.org/abs/2412.08278)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v283/main/assets/huang25a/huang25a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR150 · Diffusion Predictive Control with Constraints

**2025 · 7th Annual Learning for Dynamics & Control Conference (L4DC), PMLR 283**

**选篇理由：** 在去噪过程中引入模型投影和约束收紧，以处理训练中未见的状态及动作约束。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 已识别同题作者预印本arXiv:2412.09342，最早已核公开年份2024；正式刊会年份为2025。

[正式论文入口](https://proceedings.mlr.press/v283/romer25a.html) · [书目核验](https://proceedings.mlr.press/v283/romer25a.html) · [早期版本依据](https://arxiv.org/abs/2412.09342)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v283/main/assets/romer25a/romer25a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR153 · Warm-starting active-set solvers using graph neural networks

**2026 · Annual Learning for Dynamics and Control Conference (L4DC), PMLR 331**

**选篇理由：** 学习QP活跃约束以加速预测控制求解，考察跨问题维数的泛化。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 已识别同题作者预印本arXiv:2511.13174，最早已核公开年份2025；正式刊会年份为2026。

[正式论文入口](https://proceedings.mlr.press/v331/schmidtobreick26a.html) · [书目核验](https://proceedings.mlr.press/v331/schmidtobreick26a.html) · [早期版本依据](https://arxiv.org/abs/2511.13174)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v331/main/assets/schmidtobreick26a/schmidtobreick26a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR164 · Data-driven Acceleration of MPC with Guarantees

**2026 · Annual Learning for Dynamics and Control Conference (L4DC), PMLR 331**

**选篇理由：** 用离线MPC轨迹构造非参数贪婪策略，给出覆盖条件下的可行性与次优界。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 已核arXiv:2511.13588首次公开于2025；正式刊会年份2026。

[正式论文入口](https://proceedings.mlr.press/v331/castellano26a.html) · [书目核验](https://proceedings.mlr.press/v331/castellano26a.html) · [早期版本依据](https://arxiv.org/abs/2511.13588)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v331/main/assets/castellano26a/castellano26a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

## 07 可微MPC与结构化策略 · 近期 13 篇

### MR084 · Differentiable Robust Model Predictive Control

**2024 · Robotics: Science and Systems XX**

**选篇理由：** 将鲁棒预测控制与端到端可微学习结合。

**阅读边界：** 须确认梯度通过哪些模型与优化步骤、活动约束如何处理，以及学习后的控制性能和理论适用条件。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2024 年，不能据此断言无更早版本。 正式发表年份已核；是否有更早预印本尚未全面追溯。

[DOI：10.15607/rss.2024.xx.003](https://doi.org/10.15607/rss.2024.xx.003) · [正式论文入口](https://doi.org/10.15607/rss.2024.xx.003) · [书目核验](https://api.crossref.org/works/10.15607/rss.2024.xx.003)

[公开全文来源（有本轮获取记录）](https://www.roboticsproceedings.org/rss20/p003.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR086 · Actor–Critic Model Predictive Control: Differentiable Optimization Meets Reinforcement Learning for Agile Flight

**2026 · IEEE Transactions on Robotics**

**选篇理由：** 将可微MPC嵌入actor，研究学习代价、泛化与训练推理成本。

**阅读边界：** 须确认梯度通过哪些模型与优化步骤、活动约束如何处理，以及学习后的控制性能和理论适用条件。

**版本与首稿：** 已识别首稿或早期公开年份：2023。 2023预印本、ICRA2024会议版、2025在线期刊扩展稿、2026卷期；是近年扩展版本而非2026全新方法

[DOI：10.1109/tro.2025.3644945](https://doi.org/10.1109/tro.2025.3644945) · [正式论文入口](https://doi.org/10.1109/tro.2025.3644945) · [书目核验](https://api.crossref.org/works/10.1109/tro.2025.3644945) · [早期版本依据](https://arxiv.org/abs/2306.09852)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2306.09852v8)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR087 · Differentiable Model Predictive Control on the GPU

**2026 · International Conference on Learning Representations (ICLR 2026)**

**选篇理由：** 以GPU并行求解及隐式微分提高可微MPC训练效率，并用于强化学习控制。

**阅读边界：** 须确认梯度通过哪些模型与优化步骤、活动约束如何处理，以及学习后的控制性能和理论适用条件。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 2025预印本；ICLR2026正式主会

[正式论文入口](https://openreview.net/forum?id=bFYfV6c9zu) · [书目核验](https://iclr.cc/virtual/2026/poster/10008630) · [早期版本依据](https://arxiv.org/abs/2510.06179)

[全文入口（访问条件未核验）](https://openreview.net/pdf?id=bFYfV6c9zu)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR101 · LeTac-MPC: Learning Model Predictive Control for Tactile-Reactive Grasping

**2024 · IEEE Transactions on Robotics**

**选篇理由：** LeTac-MPC将触觉反馈学习与MPC结合，以抓取稳定性指导控制；用于研究感知表示如何进入可微优化及接触反应。

**阅读边界：** 须确认梯度通过哪些模型与优化步骤、活动约束如何处理，以及学习后的控制性能和理论适用条件。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 已核arXiv首稿公开于2024年（2403.04934）；正式发表年份2024。未把晚刊年份当作首次提出年份。

[DOI：10.1109/tro.2024.3463470](https://doi.org/10.1109/tro.2024.3463470) · [正式论文入口](https://doi.org/10.1109/tro.2024.3463470) · [书目核验](https://api.crossref.org/works/10.1109%2FTRO.2024.3463470) · [早期版本依据](https://arxiv.org/abs/2403.04934)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2403.04934)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR102 · DiffTune-MPC: Closed-Loop Learning for Model Predictive Control

**2024 · IEEE Robotics and Automation Letters**

**选篇理由：** DiffTune-MPC直接针对闭环表现调整MPC参数，区分通过控制器求导的参数学习与仅做模型辨识。

**阅读边界：** 须确认梯度通过哪些模型与优化步骤、活动约束如何处理，以及学习后的控制性能和理论适用条件。

**版本与首稿：** 已识别首稿或早期公开年份：2023。 已核arXiv首稿公开于2023年（2312.11384）；正式发表年份2024。未把晚刊年份当作首次提出年份。 本轮近三年方法优先清单不优先；若保留须明确旧稿来源。

[DOI：10.1109/lra.2024.3422836](https://doi.org/10.1109/lra.2024.3422836) · [正式论文入口](https://doi.org/10.1109/lra.2024.3422836) · [书目核验](https://api.crossref.org/works/10.1109%2FLRA.2024.3422836) · [早期版本依据](https://arxiv.org/abs/2312.11384)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2312.11384)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR141 · DiffTORI: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning

**2024 · NeurIPS 2024**

**选篇理由：** 把可微轨迹优化直接作为策略，并让策略梯度穿过优化器学习动力学与代价，针对模型学习和任务目标不一致。

**阅读边界：** 须确认梯度通过哪些模型与优化步骤、活动约束如何处理，以及学习后的控制性能和理论适用条件。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2024 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[DOI：10.52202/079017-3474](https://doi.org/10.52202/079017-3474) · [正式论文入口](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c592fc7e6207f82560ed45fece8d6937-Abstract-Conference.html) · [书目核验](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c592fc7e6207f82560ed45fece8d6937-Abstract-Conference.html)

[公开全文来源（有本轮获取记录）](https://proceedings.neurips.cc/paper_files/paper/2024/file/c592fc7e6207f82560ed45fece8d6937-Paper-Conference.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR148 · DeePC-Hunt: Data-enabled Predictive Control Hyperparameter Tuning via Differentiable Optimization

**2025 · 7th Annual Learning for Dynamics & Control Conference (L4DC), PMLR 283**

**选篇理由：** 直接对闭环DeePC性能反向传播，避免以开环拟合替代控制效果；不是严格RL。

**阅读边界：** 须确认梯度通过哪些模型与优化步骤、活动约束如何处理，以及学习后的控制性能和理论适用条件。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 已识别同题作者预印本arXiv:2412.06481，最早已核公开年份2024；正式刊会年份为2025。

[正式论文入口](https://proceedings.mlr.press/v283/cummins25b.html) · [书目核验](https://proceedings.mlr.press/v283/cummins25b.html) · [早期版本依据](https://arxiv.org/abs/2412.06481)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v283/main/assets/cummins25b/cummins25b.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR151 · Imperative MPC: An End-to-End Self-Supervised Learning with Differentiable MPC for UAV Attitude Control

**2025 · 7th Annual Learning for Dynamics & Control Conference (L4DC), PMLR 283**

**选篇理由：** 以双层优化联合学习惯性观测与MPC参数，分析端到端闭环目标接口；并非单纯重用MPC应用。

**阅读边界：** 须确认梯度通过哪些模型与优化步骤、活动约束如何处理，以及学习后的控制性能和理论适用条件。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 已识别同题作者预印本arXiv:2504.13088，最早已核公开年份2025；正式刊会年份为2025。

[正式论文入口](https://proceedings.mlr.press/v283/he25a.html) · [书目核验](https://proceedings.mlr.press/v283/he25a.html) · [早期版本依据](https://arxiv.org/abs/2504.13088)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v283/main/assets/he25a/he25a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR152 · Zero-Shot Function Encoder-Based Differentiable Predictive Control

**2026 · Annual Learning for Dynamics and Control Conference (L4DC), PMLR 331**

**选篇理由：** 联合函数编码器神经ODE与可微预测控制，跨参数系统部署显式策略而不重新求解MPC。

**阅读边界：** 须确认梯度通过哪些模型与优化步骤、活动约束如何处理，以及学习后的控制性能和理论适用条件。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 已识别同题作者预印本arXiv:2511.05757，最早已核公开年份2025；正式刊会年份为2026。

[正式论文入口](https://proceedings.mlr.press/v331/iqbal26a.html) · [书目核验](https://proceedings.mlr.press/v331/iqbal26a.html) · [早期版本依据](https://arxiv.org/abs/2511.05757)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v331/main/assets/iqbal26a/iqbal26a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR154 · Policy Optimization for Unknown Systems using Differentiable MPC

**2026 · Annual Learning for Dynamics and Control Conference (L4DC), PMLR 331**

**选篇理由：** 结合模型梯度与零阶无模型梯度，保留未知动力学下的策略优化收敛保证。

**阅读边界：** 须确认梯度通过哪些模型与优化步骤、活动约束如何处理，以及学习后的控制性能和理论适用条件。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2026 年，不能据此断言无更早版本。 正式PMLR条目已核；尚未核实是否有更早的预印本或会议前稿。

[正式论文入口](https://proceedings.mlr.press/v331/zuliani26a.html) · [书目核验](https://proceedings.mlr.press/v331/zuliani26a.html)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v331/main/assets/zuliani26a/zuliani26a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR160 · Differentiable Weights-Varying Nonlinear MPC via Gradient-Based Policy Learning: An Autonomous Vehicle Guidance Example

**2026 · IEEE Robotics and Automation Letters**

**选篇理由：** 通过求解器敏感度训练变化权重的MPC策略，处理多目标性能与在线调参。

**阅读边界：** 须确认梯度通过哪些模型与优化步骤、活动约束如何处理，以及学习后的控制性能和理论适用条件。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2026 年，不能据此断言无更早版本。 正式出版元数据已核；最早公开时间仍待核对。

[DOI：10.1109/lra.2026.3662644](https://doi.org/10.1109/lra.2026.3662644) · [正式论文入口](https://doi.org/10.1109/lra.2026.3662644) · [书目核验](https://api.crossref.org/works/10.1109/lra.2026.3662644)

全文入口：独立公开全文地址尚未确认；可从上述正式论文入口查询访问条件。

阅读状态：书目与相关性已核验，待初读。

### MR163 · DEQ-MPC : Deep Equilibrium Model Predictive Control

**2025 · Conference on Robot Learning (CoRL), PMLR 305**

**选篇理由：** 将网络输出和MPC必要条件统一为固定点，交替优化器迭代与网络推理以改善梯度和热启动。

**阅读边界：** 须确认梯度通过哪些模型与优化步骤、活动约束如何处理，以及学习后的控制性能和理论适用条件。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 2024 CoRL workshop有前身；当前条目为2025 CoRL正式主会论文，不把workshop重复计数。

[正式论文入口](https://proceedings.mlr.press/v305/gurumurthy25a.html) · [书目核验](https://proceedings.mlr.press/v305/gurumurthy25a.html) · [早期版本依据](https://cap.csail.mit.edu/sites/default/files/2025-09/Khai_CV_site.pdf)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v305/main/assets/gurumurthy25a/gurumurthy25a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR168 · Computationally efficient Gauss–Newton reinforcement learning for model predictive control

**2026 · Computers & Chemical Engineering**

**选篇理由：** 用Gauss–Newton近似避免二阶策略导数，并通过动量及信赖域提高MPC参数学习效率。

**阅读边界：** 须确认梯度通过哪些模型与优化步骤、活动约束如何处理，以及学习后的控制性能和理论适用条件。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 已识别同题作者预印本arXiv:2508.02441，最早已核公开年份2025；正式刊会年份为2026。

[DOI：10.1016/j.compchemeng.2026.109605](https://doi.org/10.1016/j.compchemeng.2026.109605) · [正式论文入口](https://doi.org/10.1016/j.compchemeng.2026.109605) · [书目核验](https://api.crossref.org/works/10.1016/j.compchemeng.2026.109605) · [早期版本依据](https://arxiv.org/abs/2508.02441)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2508.02441)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

## 08 规划教师与分层学习 · 近期 10 篇

### MR100 · Precise aggressive aerial maneuvers with sensorimotor policies

**2026 · Science Robotics**

**选篇理由：** 用模型轨迹规划提供初始化状态来缓解狭窄可行空间的RL探索问题，支撑训练侧优化引导；不是部署时在线MPC融合。

**阅读边界：** 须区分训练时规划教师与部署时在线规划，核对蒸馏、模仿或分层接口；模仿成功不自动继承教师的约束保证。

**版本与首稿：** 已识别首稿或早期公开年份：2026。 2026公开稿与Science Robotics论文

[DOI：10.1126/scirobotics.aeb0180](https://doi.org/10.1126/scirobotics.aeb0180) · [正式论文入口](https://doi.org/10.1126/scirobotics.aeb0180) · [书目核验](https://api.crossref.org/works/10.1126%2Fscirobotics.aeb0180) · [早期版本依据](https://arxiv.org/abs/2604.05828)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2604.05828)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR103 · Inverse Model Predictive Control: Learning Optimal Control Cost Functions for MPC

**2024 · IEEE Transactions on Industrial Informatics**

**选篇理由：** 从观测行为反推MPC代价函数，研究同一预测模型下学习控制目标的可辨识性与可实现性；不是直接奖励驱动RL。

**阅读边界：** 须区分训练时规划教师与部署时在线规划，核对蒸馏、模仿或分层接口；模仿成功不自动继承教师的约束保证。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2024 年，不能据此断言无更早版本。 当前只核到出版社公开年份；待追溯更早预印本，非保证首次提出年份。

[DOI：10.1109/tii.2024.3424238](https://doi.org/10.1109/tii.2024.3424238) · [正式论文入口](https://doi.org/10.1109/tii.2024.3424238) · [书目核验](https://api.crossref.org/works/10.1109%2FTII.2024.3424238)

全文入口：独立公开全文地址尚未确认；可从上述正式论文入口查询访问条件。

阅读状态：书目与相关性已核验，待初读。

### MR104 · Enhanced Imitation Learning of Model Predictive Control Through Importance Weighting

**2025 · IEEE Transactions on Industrial Electronics**

**选篇理由：** 用实际运行数据的核密度估计构建训练样本重要性权重，平衡均匀覆盖与常用状态精度，改进小网络对MPC的模仿；虽验证在变换器，机制直接关联控制策略蒸馏的数据分布。

**阅读边界：** 须区分训练时规划教师与部署时在线规划，核对蒸馏、模仿或分层接口；模仿成功不自动继承教师的约束保证。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 正式卷期2025；作者机构保存的IEEE稿显示2024年已发表，投稿2024-03、录用2024-08。未发现更早预印本；first_public_year为已核最早公开年。

[DOI：10.1109/tie.2024.3447734](https://doi.org/10.1109/tie.2024.3447734) · [正式论文入口](https://doi.org/10.1109/tie.2024.3447734) · [书目核验](https://api.crossref.org/works/10.1109%2FTIE.2024.3447734) · [早期版本依据](https://backend.orbit.dtu.dk/ws/portalfiles/portal/374845924/berat_Enhanced_Imitation_Learning_of_Model_Predictive_Control_Through_Importance_Weighting.pdf)

[公开全文来源（有本轮获取记录）](https://backend.orbit.dtu.dk/ws/portalfiles/portal/374845924/berat_Enhanced_Imitation_Learning_of_Model_Predictive_Control_Through_Importance_Weighting.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR114 · Toward Scalable Multirobot Control: Fast Policy Learning in Distributed MPC

**2025 · IEEE Transactions on Robotics**

**选篇理由：** 在每个预测区间内以前向递增的分布式actor-critic学习显式闭环策略，替代反复数值求解的分布式MPC并分析稳定性及规模迁移。

**阅读边界：** 须区分训练时规划教师与部署时在线规划，核对蒸馏、模仿或分层接口；模仿成功不自动继承教师的约束保证。

**版本与首稿：** 已识别首稿或早期公开年份：2024。 已核arXiv首稿公开于2024年（2412.19669）；正式发表年份2025。未把晚刊年份当作首次提出年份。 作者正文称有先行会议工作，当前期刊增加快速显式分布式策略、理论和规模实验；先行会议年份及增量仍需全文版本对照。

[DOI：10.1109/tro.2025.3531818](https://doi.org/10.1109/tro.2025.3531818) · [正式论文入口](https://doi.org/10.1109/tro.2025.3531818) · [书目核验](https://api.crossref.org/works/10.1109%2FTRO.2025.3531818) · [早期版本依据](https://arxiv.org/abs/2412.19669)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2412.19669)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR115 · Sample-Efficient Learning-Based Dynamic Environment Navigation With Transferring Experience From Optimization-Based Planner

**2024 · IEEE Robotics and Automation Letters**

**选篇理由：** 用可并行生成的离线MPC经验预热图网络DRL，缓解动态受约束导航的稀疏奖励和训练数据负担。

**阅读边界：** 须区分训练时规划教师与部署时在线规划，核对蒸馏、模仿或分层接口；模仿成功不自动继承教师的约束保证。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2024 年，不能据此断言无更早版本。 当前只核到出版社公开年份；待追溯更早预印本，非保证首次提出年份。

[DOI：10.1109/lra.2024.3412610](https://doi.org/10.1109/lra.2024.3412610) · [正式论文入口](https://doi.org/10.1109/lra.2024.3412610) · [书目核验](https://api.crossref.org/works/10.1109%2FLRA.2024.3412610)

全文入口：独立公开全文地址尚未确认；可从上述正式论文入口查询访问条件。

阅读状态：书目与相关性已核验，待初读。

### MR117 · Generative Predictive Control: Flow Matching Policies for Dynamic, Difficult-to-Demonstrate Tasks

**2026 · 2026 IEEE International Conference on Robotics and Automation (ICRA)**

**选篇理由：** 将采样预测控制与flow matching联系起来，用仿真优化而非人工示范训练生成式策略，并在部署热启动以保持时间一致性和高频反馈。

**阅读边界：** 须区分训练时规划教师与部署时在线规划，核对蒸馏、模仿或分层接口；模仿成功不自动继承教师的约束保证。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 已核arXiv首稿公开于2025年（2502.13406）；正式发表年份2026。未把晚刊年份当作首次提出年份。

[DOI：10.1109/icra57385.2026.11697338](https://doi.org/10.1109/icra57385.2026.11697338) · [正式论文入口](https://doi.org/10.1109/icra57385.2026.11697338) · [书目核验](https://api.crossref.org/works/10.1109%2FICRA57385.2026.11697338) · [早期版本依据](https://arxiv.org/abs/2502.13406)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2502.13406)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR128 · Reward-free World Models for Online Imitation Learning

**2025 · ICML 2025**

**选篇理由：** 通过逆soft-Q目标在Q—策略空间学习无显式奖励的潜模型，并在线规划控制，连接模仿学习与价值引导MPC。

**阅读边界：** 须区分训练时规划教师与部署时在线规划，核对蒸馏、模仿或分层接口；模仿成功不自动继承教师的约束保证。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2025 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[正式论文入口](https://proceedings.mlr.press/v267/li25af.html) · [书目核验](https://proceedings.mlr.press/v267/li25af.html)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v267/main/assets/li25af/li25af.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR134 · Model Predictive Adversarial Imitation Learning for Planning from Observation

**2026 · ICLR 2026**

**选篇理由：** 在仅观测示范下联合学习奖励与改进MPC规划代理，避免把逆强化学习与规划部署完全分离。

**阅读边界：** 须区分训练时规划教师与部署时在线规划，核对蒸馏、模仿或分层接口；模仿成功不自动继承教师的约束保证。

**版本与首稿：** 首稿年份尚未单独确认；目前已核公开记录最早为 2026 年，不能据此断言无更早版本。 已核对正式主会论文集；首个公开稿年份未单独核验。

[正式论文入口](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e59783120660eb4e847550d4584e74d6-Abstract-Conference.html) · [书目核验](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e59783120660eb4e847550d4584e74d6-Abstract-Conference.html)

[公开全文来源（有本轮获取记录）](https://proceedings.iclr.cc/paper_files/paper/2026/file/e59783120660eb4e847550d4584e74d6-Paper-Conference.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR166 · Learning to Plan, Planning to Learn: Adaptive Hierarchical RL-MPC for Sample-Efficient Decision Making

**2026 · Annual Learning for Dynamics and Control Conference (L4DC), PMLR 331**

**选篇理由：** RL动作引导MPPI采样，规划样本反过来更新价值，并按价值不确定性分配探索。

**阅读边界：** 须区分训练时规划教师与部署时在线规划，核对蒸馏、模仿或分层接口；模仿成功不自动继承教师的约束保证。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 已识别同题作者预印本arXiv:2512.17091，最早已核公开年份2025；正式刊会年份为2026。

[正式论文入口](https://proceedings.mlr.press/v331/hori26a.html) · [书目核验](https://proceedings.mlr.press/v331/hori26a.html) · [早期版本依据](https://arxiv.org/abs/2512.17091)

[公开全文来源（有本轮获取记录）](https://raw.githubusercontent.com/mlresearch/v331/main/assets/hori26a/hori26a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR169 · Learning the MPC objective function from human preferences

**2025 · 2025 IEEE 64th Conference on Decision and Control (CDC)**

**选篇理由：** 从人类偏好学习MPC目标函数，减少手工奖励及权重设计。

**阅读边界：** 须区分训练时规划教师与部署时在线规划，核对蒸馏、模仿或分层接口；模仿成功不自动继承教师的约束保证。

**版本与首稿：** 已识别首稿或早期公开年份：2025。 已识别同题作者预印本arXiv:2511.22502，最早已核公开年份2025；正式刊会年份为2025。

[DOI：10.1109/cdc57313.2025.11312675](https://doi.org/10.1109/cdc57313.2025.11312675) · [正式论文入口](https://doi.org/10.1109/cdc57313.2025.11312675) · [书目核验](https://api.crossref.org/works/10.1109/cdc57313.2025.11312675) · [早期版本依据](https://arxiv.org/abs/2511.22502)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/2511.22502)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

## 经典奠基例外 · 10 篇

以下论文单独计数，不占近期 90 篇的名额。每条均说明它对阅读近年方法的具体用途；保留经典不意味着预设研究方向。

### MR005 · Constrained model predictive control: Stability and optimality

**2000 · Automatica**

**选篇理由：** 理解终端代价、终端集合与闭环稳定性条件。

**经典保留理由：** Mayne等对约束MPC稳定性和最优性的经典系统化论述，用于理解融合理论中的终端条件；作为控制理论基准保留。

**阅读边界：** 理论结论须连同模型、约束、终端条件和优化求解假设阅读，不能直接推广为任意系统上的安全或性能保证。

**版本与首稿：** 首稿或早期公开年份未在本次全面追溯。 经典奠基例外，单独列出，不计入近三年90篇。

[DOI：10.1016/s0005-1098(99)00214-9](https://doi.org/10.1016/s0005-1098%2899%2900214-9) · [正式论文入口](https://doi.org/10.1016/s0005-1098%2899%2900214-9) · [书目核验](https://api.crossref.org/works/10.1016/s0005-1098%2899%2900214-9)

全文入口：独立公开全文地址尚未确认；可从上述正式论文入口查询访问条件。

阅读状态：书目与相关性已核验，待初读。

### MR007 · Learning Model Predictive Control for Iterative Tasks. A Data-Driven Control Framework

**2018 · IEEE Transactions on Automatic Control**

**选篇理由：** 利用已执行轨迹构建可行集合与价值近似来持续改进控制。

**经典保留理由：** Rosolia与Borrelli的Learning MPC建立从既有成功轨迹构造终端安全集和价值的迭代学习框架，是数据改善MPC性能的重要起点。

**阅读边界：** 理论结论须连同模型、约束、终端条件和优化求解假设阅读，不能直接推广为任意系统上的安全或性能保证。

**版本与首稿：** 首稿或早期公开年份未在本次全面追溯。 经典奠基例外，单独列出，不计入近三年90篇。

[DOI：10.1109/tac.2017.2753460](https://doi.org/10.1109/tac.2017.2753460) · [正式论文入口](https://doi.org/10.1109/tac.2017.2753460) · [书目核验](https://api.crossref.org/works/10.1109/tac.2017.2753460)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/1609.01387)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR008 · Data-Driven Economic NMPC Using Reinforcement Learning

**2020 · IEEE Transactions on Automatic Control**

**选篇理由：** 建立经济NMPC作为RL策略与价值近似器的理论联系。

**经典保留理由：** Gros与Zanon将参数化经济NMPC作为RL策略与价值函数近似器，奠定模型不准时直接学习闭环目标的理论路线。

**阅读边界：** 理论结论须连同模型、约束、终端条件和优化求解假设阅读，不能直接推广为任意系统上的安全或性能保证。

**版本与首稿：** 首稿或早期公开年份未在本次全面追溯。 经典奠基例外，单独列出，不计入近三年90篇。

[DOI：10.1109/tac.2019.2913768](https://doi.org/10.1109/tac.2019.2913768) · [正式论文入口](https://doi.org/10.1109/tac.2019.2913768) · [书目核验](https://api.crossref.org/works/10.1109/tac.2019.2913768)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/1904.04152)。全文入口不代表已完成阅读或附件导入。

阅读状态：初读，未复现 · [初读笔记](../notes/MR008-economic-nmpc.md)。

### MR022 · Temporal Difference Learning for Model Predictive Control

**2022 · ICML 2022**

**选篇理由：** TD-MPC将短时域潜空间规划、终端价值和策略先验联合训练，直接连接TD学习与MPC。

**经典保留理由：** TD-MPC是TD价值学习、潜在模型与短时域在线MPC联合训练的直接奠基工作，阅读近年的TD-MPC2及后续必须识别其原始机制。

**阅读边界：** 须辨明策略和价值进入规划的接口、训练目标及数据条件；有限任务收益不等于普遍优于独立MPC或RL。

**版本与首稿：** 首稿或早期公开年份未在本次全面追溯。 经典奠基例外，单独列出，不计入近三年90篇。

[正式论文入口](https://proceedings.mlr.press/v162/hansen22a.html) · [书目核验](https://proceedings.mlr.press/v162/hansen22a.html)

[公开全文来源（有本轮获取记录）](https://proceedings.mlr.press/v162/hansen22a/hansen22a.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：初读，未复现 · [初读笔记](../notes/MR022-MR026-tdmpc.md)。

### MR033 · Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models

**2018 · NeurIPS 2018**

**选篇理由：** PETS以概率动力学集成和轨迹采样连接模型不确定性与在线MPC，是学习模型规划的核心基线。

**经典保留理由：** PETS以概率动力学集成和轨迹采样建立高数据效率模型学习MPC基线，是现代不确定性模型规划的代表性起点。

**阅读边界：** 模型学习、表示或不确定性是与MPC相关的机制；须核对规划接口及分布变化条件，不能把预测精度直接等同于闭环收益。

**版本与首稿：** 首稿或早期公开年份未在本次全面追溯。 经典奠基例外，单独列出，不计入近三年90篇。

[正式论文入口](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html) · [书目核验](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)

[公开全文来源（有本轮获取记录）](https://proceedings.neurips.cc/paper_files/paper/2018/file/3de568f8597b94bda53149c7d7f5958c-Paper.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR054 · A predictive safety filter for learning-based control of constrained nonlinear dynamical systems

**2021 · Automatica**

**选篇理由：** 为学习策略提供通过有限时域预测实现的约束过滤。

**经典保留理由：** Wabersich与Zeilinger的非线性预测安全过滤框架为学习控制器的模型预测安全认证提供基础机制。

**阅读边界：** 须区分训练安全、部署安全、可行性与稳定性，并检查其模型误差和初始条件假设；不同保证不能相互替代。

**版本与首稿：** 首稿或早期公开年份未在本次全面追溯。 经典奠基例外，单独列出，不计入近三年90篇。

[DOI：10.1016/j.automatica.2021.109597](https://doi.org/10.1016/j.automatica.2021.109597) · [正式论文入口](https://doi.org/10.1016/j.automatica.2021.109597) · [书目核验](https://api.crossref.org/works/10.1016/j.automatica.2021.109597)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/1812.05506)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR066 · Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving

**2018 · IEEE Transactions on Robotics**

**选篇理由：** 信息论采样MPC/MPPI的理论与实现基础，支撑学习采样分布、价值函数和模型的融合；不是单独RL算法。

**经典保留理由：** 信息论MPC系统化推导采样式控制优化，是理解近年策略引导采样、MPPI和学习优化器的基础。

**阅读边界：** 须同时比较控制质量、计算预算与实时求解条件；优化器或采样效率提升本身不能证明端到端任务收益。

**版本与首稿：** 首稿或早期公开年份未在本次全面追溯。 经典奠基例外，单独列出，不计入近三年90篇。

[DOI：10.1109/tro.2018.2865891](https://doi.org/10.1109/tro.2018.2865891) · [正式论文入口](https://doi.org/10.1109/tro.2018.2865891) · [书目核验](https://api.crossref.org/works/10.1109%2FTRO.2018.2865891)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/1707.02342)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR081 · Differentiable MPC for End-to-end Planning and Control

**2018 · Advances in Neural Information Processing Systems 31 (NeurIPS 2018)**

**选篇理由：** 奠定将MPC作为可微策略层、端到端学习代价与动力学的基础。

**经典保留理由：** Amos等的可微MPC工作奠定将优化控制层嵌入端到端学习的核心路线。

**阅读边界：** 须确认梯度通过哪些模型与优化步骤、活动约束如何处理，以及学习后的控制性能和理论适用条件。

**版本与首稿：** 首稿或早期公开年份未在本次全面追溯。 经典奠基例外，单独列出，不计入近三年90篇。

[正式论文入口](https://papers.neurips.cc/paper_files/paper/2018/hash/ba6d843eb4251a4526ce65d1807a9309-Abstract.html) · [书目核验](https://papers.neurips.cc/paper_files/paper/2018/hash/ba6d843eb4251a4526ce65d1807a9309-Abstract.html)

[公开全文来源（有本轮获取记录）](https://papers.neurips.cc/paper_files/paper/2018/file/ba6d843eb4251a4526ce65d1807a9309-Paper.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR088 · Guided Policy Search

**2013 · ICML 2013**

**选篇理由：** 用轨迹优化生成引导样本训练全局策略，是优化教师与策略学习结合的基础框架。

**经典保留理由：** Guided Policy Search开创以轨迹优化指导策略学习的代表性路线，用于理解当代规划教师、蒸馏和策略优化。

**阅读边界：** 须区分训练时规划教师与部署时在线规划，核对蒸馏、模仿或分层接口；模仿成功不自动继承教师的约束保证。

**版本与首稿：** 首稿或早期公开年份未在本次全面追溯。 经典奠基例外，单独列出，不计入近三年90篇。

[正式论文入口](https://proceedings.mlr.press/v28/levine13.html) · [书目核验](https://proceedings.mlr.press/v28/levine13.html)

[公开全文来源（有本轮获取记录）](https://proceedings.mlr.press/v28/levine13.pdf)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。

### MR095 · MPC-Net: A First Principles Guided Policy Search

**2020 · IEEE Robotics and Automation Letters**

**选篇理由：** 以控制Hamiltonian而非单纯动作拟合损失蒸馏MPC，连接最优性条件、约束与策略学习。

**经典保留理由：** MPC-Net以第一性原理与最优控制结构指导策略搜索，是MPC经验向快速策略迁移的代表性基础工作。

**阅读边界：** 须区分训练时规划教师与部署时在线规划，核对蒸馏、模仿或分层接口；模仿成功不自动继承教师的约束保证。

**版本与首稿：** 首稿或早期公开年份未在本次全面追溯。 经典奠基例外，单独列出，不计入近三年90篇。

[DOI：10.1109/LRA.2020.2974653](https://doi.org/10.1109/LRA.2020.2974653) · [正式论文入口](https://doi.org/10.1109/LRA.2020.2974653) · [书目核验](https://doi.org/10.1109/LRA.2020.2974653)

[公开全文来源（有本轮获取记录）](https://arxiv.org/pdf/1909.05197)。全文入口不代表已完成阅读或附件导入。

阅读状态：书目与相关性已核验，待初读。
