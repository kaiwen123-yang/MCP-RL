# πMPC来源、版本与计算证据审计

日期：2026-10-15。本轮没有运行求解器、训练、仿真或性能测试。πMPC指定v2全文已读，CUDA MPC只读摘要、引言和相关方法定位。

## 1. 身份与版本

**πMPC: A Parallel-in-horizon and Construction-free NMPC Solver**，[arXiv记录](https://arxiv.org/abs/2601.14414)，v1于2026-01-20公开，当前v2于2026-05-20更新，共8页。v2作者为Liang Wu、Bo Yang、Junheng Li、Xu Yang、Yilin Mo、Yang Shi、Aaron D. Ames、Ján Drgoňa。部分搜索缓存和第三方条目仍列v1六作者，不能混用。

第一作者[主页](https://sites.google.com/view/liangdot/homepage)本轮仍将该稿列为IEEE TAC在审；[作者实验室页面](https://solaris-jhu.github.io/solaris-website/papers.html)标为Preprint。第三方存在ACC2026单页条目，但本轮未从正式刊会来源核清其与8页v2的关系。因此本报告把8页v2作为预印本，不将其称为已正式发表的ACC/TAC论文。

**CUDA MPC: A GPU-Native Solver for Model Predictive Control**，[arXiv记录](https://arxiv.org/abs/2608.03051)，Babak Akbari、Melissa Greeff，2026-08-04 v1，8页。本轮未确认正式刊会。p1–2明确在πMPC的horizon-parallel ADMM分裂上，进一步设计fused CUDA kernel、shared-memory存储和相邻block同步。

## 2. 本轮完整覆盖

- πMPC v2：p1–8全部文字，含全部25参考；原图核查p4、6、7、8，涵盖速度形式、主算法、全部实验图1–4、表I–II。未独立复现或重建全部引用收敛证明。
- CUDA MPC v1：仅p1–2全文，含摘要、动机、贡献与对πMPC的关系；未读其完整方法和实验，不把摘要性能数字当本轮已核实的实验结果。
- 原始PDF、逐页文字、页面图、hash在忽略目录 `work/pimpc-2026-10-15/source-and-cost/`。

πMPC v2 SHA256：`0945386fe7955b4ad291c3813d039eb0c16d403bb0bcb3ff0c79715516ef193f`。

CUDA MPC v1 SHA256：`03a7629f8a766ab3d8543469a04dc1962181b885bf846e3db0cc9ec018d1789d`。

## 3. 给用户的一句话意义

**πMPC把一条长预测计划的求解步骤改成可以沿时间段同时计算，为使用更长预测时域提供了一种求解工具。** 它没有学习机器人的新执行能力，也没有决定RL和MPC怎样共同学习。

## 4. 并行到底在哪一层

| 并行类型 | 并行的对象 | πMPC v2实际证据 |
|---|---|---|
| 预测时域并行 | 同一个优化问题中不同未来时刻的局部更新 | 本文核心：变量分裂后各时刻局部ADMM更新可并行 |
| 候选轨迹并行 | 多条控制候选分别rollout，如MPPI/CEM | 本文不是这一路线 |
| 环境并行 | 多个机器人仿真／多个独立MPC实例 | v2主实验没有大批RL环境；后续πⁿMPC另外实现 |
| 训练反向传播并行 | 多个batch或可微优化梯度 | v2结论仍将可微层列为未来工作 |

每轮中各预测时刻可同时工作，不意味着整个求解完全没有顺序：ADMM迭代仍依赖前一轮；残差归约、同步、projection及矩阵缓存也有成本。真实wall time取决于GPU资源、问题规模、迭代次数及数据移动，不能将“每轮horizon-parallel”读成“任意N总时间恒定”。

## 5. 原文算法边界

p2 Problem1是LTV MPC，可由在线线性化/RTI-NMPC得到；Remark1明确假设该子问题凸且可行。p3–4引入未来状态副本及`B_k u_k`副本，再用速度形式将输入并入状态，使子步由小矩阵运算和集合投影构成。

construction-free指直接利用阶段系统矩阵，避免显式组装整条时域的大QP矩阵；并非不需要动力学、线性化、局部矩阵逆/分解或约束projection。

p5明确承认长时域/高维问题可能因条件数和耦合而需要更多ADMM迭代。重启加速利用既有凸ADMM收敛结果；文中没有给一般NMPC全局最优、有限迭代物理硬安全、固定截止时间收敛或端到端WCET证书。

## 6. 成本：必须区分单次迭代与完整MPC

全部实验使用Julia、AMD Ryzen9 5900X和NVIDIA RTX3080桌面机（p6）。没有机器人实机或NX实测。

### AFTI-16（p6–7）

线性化不稳定飞机模型，N=5，采样0.05秒，200个MPC步。达到`10^-6`残差所需平均迭代次数：OSQP1326、πMPC2030、Riccati-ADMM2551、作者QP-ADMM5207。图2上限10000。

这说明πMPC的结构没有免费消除收敛迭代。该实验给迭代数，不应借其他问题的每轮耗时相乘后伪造它的完整求解时延。

### 随机系统（p7 Table I）

改变状态维数n=10…100、输入m=3…30、时域N=20…500。时间是**200次迭代的平均每迭代毫秒数**，不是解到共同精度的完整求解时间。

GPU πMPC在所列规模下约0.53–0.62ms/iteration。例`n=100,m=30,N=500`：GPU0.62ms、CPU πMPC8.14ms、OSQP99.36ms、Riccati425.69ms；作者一般QP-ADMM内存不足。小问题OSQP反而更快，例如`n=10,m=3,N=20`为0.02ms，GPU πMPC0.53ms。

不能把0.53ms写成0.53ms完成5000步MPC，也不能把表中比例当任意问题总加速比。不同实现的factorization/更新时间、稀疏结构和收敛精度都影响总体比较。

### 非线性CSTR（p7–8 Table II）

在线在当前状态线性化，Euler离散，采样间隔0.5分钟；N=5000；400个MPC步，每步固定1000次迭代。

| 统计 | OSQP总时间 | 其中QP构建时间 | πMPC解时间 |
|---|---:|---:|---:|
| 中位数 | 1.891s | 1.073s | 0.526s |
| 最大观测值 | 1.970s | 1.120s | 0.610s |

这是每MPC步秒数。作者用一致的闭环图说明这一设置下两方法表现接近；固定相同迭代数仍不等于精确相同最终残差/目标精度。采样30秒的过程控制实例不能直接证明50/100Hz机器人可部署。

## 7. 有无RL和学得响应

πMPC v2全文没有RL训练、learned response/world model或策略共同优化。论文是数值求解方法，结论把可微版本作为未来工作。

CUDA MPC所读p1–2同样定位于算法和CUDA执行设计：其核心增量是将整个ADMM循环放在fused kernel中，减少反复kernel launch、全局内存和host参与。本轮未读其完整实验；摘要中的965倍、长预测停车和swarm性能均保留为作者摘要声称，不能直接用于我们训练或NX运行预算。

## 8. 对当前研究的真实影响

用户提供这条文献意味着值得重新审视“训练中使用MPC的计算开销是否不可接受”。它提供更好的实现工具和明确结构，不单独证明某种RL＋MPC耦合新颖。

紧接的作者工作已经把这一工具用于大批RL环境：*Accelerating and Scaling MPC-Guided Reinforcement Learning for Humanoid Locomotion and Manipulation*，arXiv:2606.05687，完整阅读另见[继承工作审读](mpc-rl-followup.md)。因此不能把“给πMPC接上RL、用它同时算很多训练环境”本身当成未被做过的贡献。
