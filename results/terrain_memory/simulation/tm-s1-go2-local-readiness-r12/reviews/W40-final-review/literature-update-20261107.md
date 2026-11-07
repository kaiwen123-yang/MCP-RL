# S1 文献补充与方法适配审计（2026-11-07）

## 结论与使用边界

下一阶段值得优先借鉴的，不是继续给现有动作附加更复杂的投影，而是把三件事连起来：**预测器能识别真实接触变化；控制器能输出整个执行窗口内可实现的动作；学习模块拥有足够的动作权限来改变接触、负载或恢复策略。** GPU 的价值则需要落实为真实状态、真实候选动作和完整规划步骤上的正确性与吞吐结果。

本次精选 10 篇。第 1—5 篇的标题或标识符，在本次检索的 `literature/`、`docs/`、`notes/` 和 r12 `reviews/` 中未命中；这只是本地文本检索结论，不宣称此前从未接触过。第 6—10 篇属于原路线的补充核验和深化阅读。TurboMPC、MPX、GATO、WarpMPC、DIAL-MPC 已在既有 GPU 文档中出现，不能包装为首次发现。

论文结果不等于本仓库复现结果。下文逐项说明实际阅读范围；项目页、摘要、出版社元数据、全文关键章节是不同证据等级。本次没有连接 D57、运行 GPU 实验或复现作者代码，也没有据论文推断本项目已解决稳定通行。所有实验建议均是下一轮待验证方案。

## 一、精选 10 篇：机制、当前落点与边界

### 1. Learning to Adapt through Bio-Inspired Gait Strategies for Versatile Quadruped Locomotion

- **版本与发表：** Nature Machine Intelligence 7:1141–1153，2025；DOI `10.1038/s42256-025-01065-z`。arXiv `2412.09440v3`，2025-06-22，首稿 2024 年。
- **阅读范围：** 出版社元数据与方法检索；作者 arXiv v3 的框架、结果、方法中步态选择指标与补充恢复实验。未复现实验。
- **原理：** 高层选择步态，步态调度器提供程序化动作先验，低层策略修正非预期接触；使用能耗、力矩饱和、外部功和接触跟踪误差共同评价步态，辅助步态用于失稳恢复。
- **当前可执行借鉴：** 先把饱和持续时间、接触时序偏差、负载变化和恢复结果作为诊断输出；在新冻结阶段，让学习模块能影响有物理意义的步态/接触时序/局部恢复动作。不能期待低频单一 cap 自动获得这些权限。
- **边界：** 这里的 gait memory 是程序化步态记忆，不是本项目反复踩踏、随时间恢复的世界 patch 材料状态记忆；不应混淆贡献。当前有效训练尾段不据此改奖励。
- **来源与代码：** [出版社](https://www.nature.com/articles/s42256-025-01065-z)；[arXiv v3 全文](https://arxiv.org/html/2412.09440v3)；[作者代码](https://github.com/ihcr/learning_to_adapt)。代码入口已核实，未运行。

### 2. Agile Perceptive Multiskill Locomotion for Quadrupedal Robots in the Wild

- **版本与发表：** Science Robotics 11(116):eadz7397，2026-07；DOI `10.1126/scirobotics.adz7397`；arXiv `2607.13579`。
- **阅读范围：** 作者项目页、作者发表页和机构元数据；出版社全文访问受限，未把项目页视为已完整阅读论文。
- **原理：** APT-RL 用轨迹优化生成状态与控制数据，预训练 Transformer VAE 和步态力矩解码器，RL 输出潜变量与辅助动作，再进行感知蒸馏；真实 HOUND 展示多步态与动作切换。
- **当前可执行借鉴：** 将已经通过物理检查的 MPC 动作片段积累为可执行先验，单独检查恢复动作能否把状态带回可控制区间。以后可以研究先验加小残差，而非让每次训练从不可行动作附近重新探索。
- **边界：** 这是新的研究分支，涉及数据集、动作解码器、训练架构和感知；不是当前 QP 错误的修补方法，也不证明软地形历史记忆有效。未核实公开实现代码，不能写成可直接复现。
- **来源：** [作者项目与方法摘要](https://skillquadsr.github.io/)；[arXiv](https://arxiv.org/abs/2607.13579)；[作者发表页](https://seungwoo-hong.github.io/publications/)。

### 3. Learning Deployable Locomotion Control via Differentiable Simulation

- **版本与发表：** CoRL 2025，PMLR 305:3665–3684；arXiv `2404.02887v2`，2025-08-27，首稿 2024 年。
- **阅读范围：** PMLR 正式条目，arXiv v2 的接触求解、梯度平滑、训练和局限章节；未复现。
- **原理：** 在可微刚体模拟中处理硬接触与接触脉冲平滑，以解析梯度训练可部署控制；论文特别讨论离散接触、梯度不连续和有偏梯度问题。
- **当前可执行借鉴：** GPU 导数审计至少拆为三项：CPU/GPU 前向映射是否一致；同一 GPU 前向映射的有限差分与 AD 是否一致；这些梯度是否能正确预测候选动作的局部效果。三项不能只用一个矩阵最大误差概括。
- **边界：** 改平滑接触模型会改变所优化的动力学，不能称为原生 MuJoCo 等价修复。论文的模拟器与执行器简化也不能直接搬成 Go2 材料模型。
- **来源：** [PMLR](https://proceedings.mlr.press/v305/schwarke25a.html)；[arXiv v2 全文](https://arxiv.org/html/2404.02887v2)。本次未核实可运行作者代码入口。

### 4. The Surprising Effectiveness of Linear Models for Whole-Body Model-Predictive Control

- **版本与发表：** Humanoids 2025；DOI `10.1109/Humanoids65713.2025.11203045`；arXiv `2509.17884`。
- **阅读范围：** 官方项目、CMU 机构发表条目和作者仓库入口；未逐页阅读论文 PDF，未运行代码。
- **原理：** 使用全身动力学的线性时不变近似完成基本运动控制，避免在线非线性动力学求值与矩阵求逆；项目展示 Go2 扰动恢复、回到目标位置和跨越箱体等实验。
- **当前可执行借鉴：** 将简单而可执行的控制器作为独立比较对象：在相同物理、负载和执行窗口下，比较饱和、恢复与长时存活，而不是以模型复杂度判断方法优劣。
- **边界：** 该工作不证明线性模型适合本项目所有材料历史任务；应作为下一阶段独立对照，不能直接替换现有任务定义或拼接进未完成的训练谱系。
- **来源与代码：** [官方项目](https://linearwalking.github.io/)；[CMU 条目](https://publications.ri.cmu.edu/the-surprising-effectiveness-of-linear-models-for-whole-body-model-predictive-control)；[作者代码](https://github.com/LinearWalking/LinearWalking)。

### 5. Learning Quadruped Locomotion Using Differentiable Simulation

- **版本与发表：** CoRL 2024，正式论文集 PMLR 270:258–271 于 2025 年出版。应区分会议年份与论文集年份。
- **阅读范围：** PMLR 正式条目与方法摘要；没有将其标为全文推导已复核。
- **原理：** 用较简单、平滑且可微的模型提供训练梯度，并通过更准确的不可微模拟对齐机器人的前向状态，以缓和可微建模与真实接触的矛盾。
- **当前可执行借鉴：** 若原生接触导数在模式切换附近不适合作为优化方向，可以另设“代理梯度是否改善候选动作”的实验；同时保留真实前向模拟作评估。
- **边界：** 代理梯度是有意选择的近似，不能与“CPU/GPU 是同一个预测器”的一致性任务混用；采用它意味着新的算法版本和新的消融。
- **来源：** [PMLR 正式论文及 PDF 入口](https://proceedings.mlr.press/v270/song25a.html)。本次未核实作者代码。

### 6. Primal-Dual iLQR for GPU-Accelerated Learning and Control in Legged Robots（MPX）

- **版本与发表：** IEEE RA-L 11(1):1010–1017，2026；DOI `10.1109/LRA.2025.3632610`；arXiv `2506.07823v3`，2025-12-06。
- **阅读范围：** arXiv v3 的求解器、约束处理、计算性能与局限章节；作者仓库入口已核实，未复现。
- **原理：** 多重射击、原始对偶求解与关联扫描，把时间维度和多个环境的工作并行化。其不等式使用 relaxed barrier，不能笼统称所有约束严格满足。
- **当前可执行借鉴：** 先把单次 MPC 内的时间节点、有限差分扰动和候选动作批量化，不必为了用 GPU 新开大量 PPO 谱系；只有剖析显示 backward solve 是实际瓶颈时，再比较并行求解器。
- **边界：** 论文部分结果按一次迭代比较；短时域时 CPU 可能更快，传输和同步会抵消核函数加速。当前 12 维 PD 目标动作与论文中的控制变量不能直接等同。
- **来源与代码：** [版本与发表信息](https://arxiv.org/abs/2506.07823)；[v3 全文](https://arxiv.org/html/2506.07823v3)；[作者代码](https://github.com/iit-DLSLab/mpx)。

### 7. TurboMPC: Fast, Scalable, and Differentiable Model Predictive Control on the GPU

- **版本与发表：** arXiv `2606.24039v1`，2026-06-23；本次未核实正式会议或期刊接收，按预印本使用。
- **阅读范围：** 作者 PDF 的算法、实现与 limitations；官方仓库已核实，未复现。
- **原理：** 利用固定稀疏结构、GPU 直接求解及融合计算处理大量 MPC/QP，并通过 KKT 系统提供导数。
- **当前可执行借鉴：** 对真实 QP/候选批次做独立 benchmark，分开记录构造、分解、迭代、传输和同步；保留真实约束与收敛标准。
- **边界：** 直接分解有固定开销，支持的逐点 slab 约束并非一般接触互补约束；前向求解收敛与精度会影响隐式梯度。文中强调双精度，不能仅凭 96 GB 显存推断当前卡的 FP64 吞吐与实际加速。
- **来源与代码：** [arXiv](https://arxiv.org/abs/2606.24039)；[PDF](https://arxiv.org/pdf/2606.24039)；[Toyota Research Institute 代码](https://github.com/ToyotaResearchInstitute/turbompc)。

### 8. Residual MPC: Blending Reinforcement Learning with GPU-Parallelized Model Predictive Control

- **版本与发表：** 作者发表页确认 IEEE T-RO 2026；DOI `10.1109/TRO.2026.3721669`。arXiv `2510.12717v1` 首稿仍保留投稿时期说明，不能据旧说明否定后续正式发表。
- **阅读范围：** 作者发表页、arXiv 摘要及架构描述；本次未对全文推导逐式核验，也未核实可运行公开实现。
- **原理：** GPU 并行化 MPC 提供基线行为，RL 在低层力矩上给残差修正，使学习输出能直接影响身体与接触响应。
- **当前可执行借鉴：** 为未来 PANN/MPC 接口建立“输出到底改变什么物理量”的检查：速度 cap、接触时序、落脚点、负载分配和关节/力矩残差的权限不同。应先用可重放干预确认某个权限能改善失稳，再决定是否学习它。
- **边界：** 不能把该论文的力矩残差等同于当前 cap 调制；也不能从论文吞吐推导本仓库移植后速度。新动作接口属于新冻结阶段。
- **来源：** [arXiv](https://arxiv.org/abs/2510.12717)；[作者正式发表列表](https://seungwoo-hong.github.io/publications/)。

### 9. Terramechanics-Aware Adaptive Quadrupedal Locomotion via Foot-Terrain Interaction Model

- **版本与发表：** IEEE RA-L 11(11):12552–12559，2026；DOI `10.1109/LRA.2026.3732903`，在线发表 2026-09-10。
- **阅读范围：** IEEE 出版社元数据与摘要检索；出版社全文访问受限，本次没有把全文模型细节标为独立完整复核。
- **原理：** 通过 VG 足地作用模型刻画沉陷、剪切饱和与承载能力变化，用接触历史驱动地形适应和步态变化，并报告软地形仿真与 Go2 沙地实验。
- **当前可执行借鉴：** 材料辨识输出应能解释接触载荷/滑移/沉陷，再通过明确接口改变动作；优先验证“读到材料历史后改变负载或接触”这一因果链，而不只增加一个高层标签。
- **边界：** 该工作不能替本项目证明可恢复 patch 的记忆收益，也不能解释所有首次接触材料前就出现的饱和。未核实作者代码。
- **来源：** [IEEE 正式条目](https://ieeexplore.ieee.org/abstract/document/11685272/)；[DOI](https://doi.org/10.1109/LRA.2026.3732903)。

### 10. RK-MPC: Residual Koopman Model Predictive Control for Quadruped Locomotion in Offroad Environments

- **版本与发表：** arXiv `2604.04221v2`，2026-04-07；首稿 2026-04-05。本次未核实正式会议或期刊接收。仓库已引用 v1；固定版本引用可以保留，但不能称 v1 为最新版。
- **阅读范围：** 最新摘要/版本历史；v1 正文的模型、残差辨识、实验设计与讨论结构。未核实作者代码，项目视频页访问失败。
- **原理：** 在名义物理模型上学习低维 lifted residual，并保留凸 QP 控制结构；报告 Go1 越野实验。重点是补偿可识别的预测偏差，而不是由学习模块接管全部动力学。
- **当前可执行借鉴：** 用已保存真实轨迹构造 held-action 多步预测误差数据，先做按接触/负载/材料阶段分层的残差预测与留出验证；比较无残差、无历史残差、有历史残差，之后才考虑进入 MPC。
- **边界：** SRB 残差模型不等于 MuJoCo 全身接触模型；改善一步误差不保证长时闭环稳定，也不自动包含材料恢复的持久状态。
- **来源：** [最新版本信息](https://arxiv.org/abs/2604.04221)；[v1 全文](https://arxiv.org/html/2604.04221v1)。

## 二、优先落地的 8 个实验机制

以下是对本项目的建议，不是上述论文已经替本项目完成的实验。

| 优先级 | 机制与具体实验 | 应交付的证据 | 暂不作出的结论 |
|---|---|---|---|
| P0 | **真实执行窗口。** 对同一 held action 重放全部子步，记录原始 PD 请求、实际施加力矩、关节状态与接触变化。候选动作也按同一窗口验证。 | 饱和发生时刻、持续长度、幅度与接触/失稳的时间关系；修复前后同状态对照。 | 初始状态投影通过不代表后续子步可执行。 |
| P0 | **接触灵敏度。** 在真实状态比较 CPU-FD、GPU-FD、GPU-AD，并对接触开闭、饱和边界做正负方向扰动。 | 前向状态/接触差异与导数差异分开；方向导数能否预测实际代价变化。 | 模式边界不可微时，不能把所有差异直接归为程序 bug，也不能因此忽略前向模型错误。 |
| P0 | **完整候选选择。** 固定候选动作集、warm start、phase、残差与终端成本，比较完整 planner replay。 | 成本分项、可行性、候选排序、最终动作及退出原因。 | 单独 A/B 对齐不等于整个 MPC 等价；单独预测器速度也不等于训练收益。 |
| P1 | **可执行动作权限。** 在离线/短评估中分别干预步态时序、负载或小残差，先看能否改变失稳结局。 | 同初态、同扰动、不同物理干预的成对轨迹。 | 不先训练一个更大的策略来掩盖接口没有控制权。 |
| P1 | **恢复动作先验。** 从已通过物理验证的轨迹片段构建动作库或 warm start，针对常见失稳前状态重放。 | 可重复的恢复成功率、时间、饱和与接触指标；失败覆盖范围。 | 不把尚未稳定的失败片段直接当“专家先验”。 |
| P1 | **有历史/无历史的预测残差。** 对真实 held-action 轨迹做多步预测，按 patch、负载和接触阶段分层留出。 | 无残差、无历史残差、有历史残差的相同留出集结果与误差校准。 | 低预测误差不单独构成稳定通行或材料记忆因果收益。 |
| P1 | **强而简单的控制基线。** 在新冻结阶段准备结构较简单、动作权限明确的基线，使用同一物理任务。 | 稳定时长、恢复、合法通行、饱和、计算开销。 | 不用换任务、放宽物理约束或额外调参预算来宣称更优。 |
| P1 | **按实际工作批量化。** GPU 批处理 knots × perturbations × candidates，数据常驻设备，再评估是否值得迁移求解器。 | 编译、warm-up、核函数、传输、同步、端到端 p50/p95、峰值内存与吞吐。 | 不以显存占用越高越好，也不把作者大批次加速倍率移植为本机实测。 |

动作先验、接触时序和残差权限是独立研究变量，应有版本与对照。与此相对，已经复现的求解器实现错误应直接修复，并保持原约束和判据；无需靠新论文、换求解器或扩大容差来解释它。

## 三、GPU 支线应扩大为有产出的并行研究任务

最新用户授权支持更多 GPU 工作，不应继续用旧草稿“只跑少量 fixture 然后停止”的范围限制它。CPU 稳定性主线与 GPU 离线研究可同时推进。

1. **复用现有真实输入。** 先使用已经交付的 fixture/NPZ，避免重新制造无法对应的随机状态。区分预测器原版本与新控制器版本；不能把旧 r28 输入、新 r29/qp2 或拟议 r30 混为一次等价比较。
2. **扩充分层真实状态库。** 覆盖材料接触前、首次受载、保持动作初始与后续子步、饱和、接触切换和失稳前兆。历史状态必须能从已保存上下文重建；缺失状态不能编造，改在下一次运行增加捕获。上下文包括必要的 q/v、act、时间、mocap、task/phase、材料 patch 与 warm-start/solver 状态。
3. **完成前向和导数审计。** 对齐时间步、控制保持、精度、模型参数、接触与求解器设置。状态扰动使用相同的姿态切空间与差分步长。先分清不同前向映射与同一映射求导错误，再做 AD 或代理梯度实验。
4. **让数据常驻设备。** 根据实际空闲显存、内存和另一任务占用，自适应增加批量；以真实吞吐和尾延迟选择批量，不设置为了占满 96 GB 的目标。先试少量有代表性的批量档位，再根据结果扩展。
5. **推进到候选与完整规划。** 离线完成 phase/residual/cost、终端项、候选排序、动作输出和退出原因比较；不必等待 CPU 长时稳定才开始这些工作。只有想接入正式闭环训练时，才需要按新的冻结版本做匹配物理评估。
6. **有依据地比较替代方法。** MPX 适合检查时间并行求解；TurboMPC 适合独立 QP 批次；MJWarp 可作为前向批处理的候选。每项都应针对剖析发现的瓶颈，不同时重写整个栈。

可同时组织两个 CPU 任务和一个 GPU 任务；具体 worker 数量需根据真实 CPU、内存、显存与其他进程占用安排。初始可沿现有两条 12 worker 主线、GPU 主机 1—2 线程的结构测量争用，再在新阶段调整；这不是对当前机器空闲资源的现场确认。

后台任务不加墙钟完成期限。阶段以明确产物、必要样本数/更新数、技术失败或人工停止结束。数值迭代上限、物理评估时长和异常保护仍各有用途，不能把它们与人为墙钟终止混淆。推送 GitHub 后不再安排用户电脑上的额外归档。

### MuJoCo 后端证据边界

- [MJX 3.2.3 文档](https://mujoco.readthedocs.io/en/3.2.3/mjx.html)说明其优化目标偏向大量并行场景，单场景延迟可能较差。它与原生 MuJoCo 在碰撞实现等方面存在差异；这是需要检查的条件，**不是**已证明当前 base-z 误差的根因。
- [MJWarp 官方文档](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html)区分吞吐与单步延迟，说明其与原生 MuJoCo 的精度差异，并未提供可直接视为 MJX 等价替代的通用可微能力。引用 latest 文档时需记录实际安装版本，不能把当前文档特性倒推到冻结的 MuJoCo/MJX 3.2.3。

## 四、W40 原有 12 篇的版本与发表状态复核

本表更新证据状态，不覆盖原 `literature.json`。旧版本若为固定阅读对象仍可保留，应补充版本说明。

| W40 编号 | 核验结果及需修正点 | 一手来源 |
|---|---|---|
| L01 ProxQP | 正式 IEEE T-RO 2025，DOI `10.1109/TRO.2025.3577107`。不能只以 2023 预稿年份标记正式发表。closest-feasible 机制也不能静默改变本项目物理约束。 | [作者正式条目](https://scaron.info/publications/tro-2025.html) |
| L02 ABC-DDP | *Real-Time Control-Constrained DDP for Underactuated Balancing of Legged Robots*，arXiv `2608.18552v2`；作者页确认 RA-L 2026，DOI `10.1109/LRA.2026.3723262`。本次未核实 IEEE 最终卷页，全文未独立完整复核。 | [作者发表页](https://onesoulkang.github.io/)；[arXiv](https://arxiv.org/abs/2608.18552) |
| L03 DR-MPC | *DR-MPC: Fast and Feasible Dynamics-Relaxed Model-Predictive Control for Legged Locomotion*，arXiv `2609.20035v1`，2026-09-17；正文是 submitted to RA-L，不是已接收。其通过放松动力学/部分约束构造可行问题，不能把 QP 可行等同精确物理可行。已阅读 PDF 关键算法与限制。 | [作者 PDF](https://arxiv.org/pdf/2609.20035) |
| L04 CCMPC | 正式 RA-L 10(8):8388–8395，2025，DOI `10.1109/LRA.2025.3585315`。不确定性传播和摩擦裕量依赖模型假设与校准；不是通用不摔倒保证。 | [作者项目](https://cc-mpc.github.io/) |
| L05 Gait-Net | 正式 RSS 2025，DOI `10.15607/RSS.2025.XXI.069`。题为 *Gait-Net-augmented Implicit Kino-dynamic MPC for Dynamic Variable-frequency Humanoid Locomotion over Discrete Terrains*。 | [RSS 正式论文页](https://www.roboticsproceedings.org/rss21/p069.html) |
| L06 Differentiable Simulation | arXiv `2409.07107` 最新 v3（2026-09-21）题为 *Highly-Efficient Differentiable Simulation for Robotics*；更早版本题名含 End-to-End。需固定标题、作者和版本，不混用。正式 venue 本次未核实。 | [arXiv 最新元数据](https://arxiv.org/abs/2409.07107) |
| L07 ProxDDP | *ProxDDP: Proximal Constrained Trajectory Optimization*，正式 T-RO 41:2605–2624，2025，DOI `10.1109/TRO.2025.3554437`。可作为完整受约束轨迹优化参照，不是当前已定位实现错误的替代解释。 | [作者条目](https://simple-robotics.github.io/publications/proxddp-tro-2025/)；[IEEE](https://ieeexplore.ieee.org/abstract/document/10938351/) |
| L08 IPDDP2 / FilterDDP | `2504.08278v7`（2026-05-31）已更名 *Line-Search Filter Differential Dynamic Programming for Optimal Control with Nonlinear Equality Constraints*，方法 FilterDDP；ICRA 2026 接收有官方程序支持。旧 v2 的 *Interior Point Differential Dynamic Programming, Redux* 可按固定旧版本引用，但不能称最新版。已读 v7 的 filter/正则化与局限。 | [最新 arXiv](https://arxiv.org/abs/2504.08278)；[v7 全文](https://arxiv.org/html/2504.08278v7)；[ICRA 官方程序](https://ras.papercept.net/conferences/conferences/ICRA26/program/ICRA26_ContentListWeb_5.html) |
| L09 Parallel LQ | RSS 2024 正式题名为 *Parallel and Proximal Linear-Quadratic Methods for Real-Time Constrained Model-Predictive Control*，DOI `10.15607/RSS.2024.XX.002`；与较早 arXiv 题名不同。 | [RSS 正式论文页](https://www.roboticsproceedings.org/rss20/p002.html) |
| L10 MPC-guided RL | *Accelerating and Scaling MPC-Guided Reinforcement Learning for Humanoid Locomotion and Manipulation*，`2606.05687v1`，2026-06-04，本次仅核实预印本。MPC 用于训练期奖励/critic 指导，部署无需 MPC，不能当作当前在线全身 MPC 架构的直接成功证明。已阅读正文相关架构。 | [arXiv](https://arxiv.org/abs/2606.05687)；[全文](https://arxiv.org/html/2606.05687v1)；[作者项目](https://caltech-amber.github.io/mpc-guided-rl/)；[代码](https://github.com/junhengl/mpc-rl) |
| L11 Terrain-Adaptive SINDy | *Terrain-Adaptive Quadrupedal Locomotion Using Model Predictive Control With Impedance Control and SINDy Model Enhancement*，IJ Robust and Nonlinear Control 在线发表 2026-08-14，DOI `10.1002/rnc.70700`。现在已核实作者公开代码；旧“未核实代码”应更新。PyBullet 仿真证据不能写成 Go2 硬件验证。 | [出版社](https://onlinelibrary.wiley.com/doi/full/10.1002/rnc.70700)；[作者代码](https://github.com/pjmcconnellogue/Terrain-Adaptive-Quadrupedal-Locomotion) |
| L12 Control-Limited DDP | ICRA 2014，DOI `10.1109/ICRA.2014.6907001`。作为经典方法基础保留，不计为近期新文献；固定 box 约束与随状态变化、跨子步的 PD 可行域不同。 | [作者 PDF](https://roboti.us/lab/papers/TassaICRA14.pdf) |

## 五、研究判断

这些文献支持更明确的下一步：先让真实接触、真实动作保持和完整候选选择得到可核对的证据，同时用 GPU 扩大可重放测试；下一冻结阶段再评估具有实际物理权限的恢复/时序/残差接口，并检查材料历史是否提供独立收益。

更强的 GPU、更多训练或更复杂投影都不能单独保证稳定。可继续推进实验，关键是每条并行任务都回答一个可检验的问题，并留下能重放的输入、原始输出、版本、失败分类和对照。文献负责提出机制与边界，仓库中的真实闭环结果负责决定是否采用。
