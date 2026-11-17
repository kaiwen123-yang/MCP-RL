# W54 后的文献与官方实现核对

检索与阅读日期：2026-11-17（Asia/Shanghai）。以下按对当前故障的帮助排序；论文结论与本项目推论分开。最新论文不能替代原 Go2/TM02 的实际验证。

## 1. 最直接的依据是 MuJoCo 官方修复

[MuJoCo 3.3.1 changelog](https://mujoco.readthedocs.io/en/stable/changelog.html#version-3-3-1-apr-9-2025) 明确记录 `mj_jacDot` 曾遗漏计算点移动产生的项。对应 [官方提交 de48f4178f971d945fc135afb42699b8ca6bec8e](https://github.com/google-deepmind/mujoco/commit/de48f4178f971d945fc135afb42699b8ca6bec8e)，提交于 2025-03-11，随 2025-04-09 的 3.3.1 发布。已核对官方提交中的 `engine_support.c` 差分。

本项目 F 固定使用 3.2.3，足心和逐连杆质心的平移 `mj_jacDot` 调用正处于受影响路径。对当前用途，收缩后的缺项为同一世界坐标系中的 `omega_body × v_point`。角 Jacobian 不应加这一平移项，滚动球底须使用修正后的足心项，不能重复修。N 的 `W54NormalJdotV` 用运动学方向差分，不能套用同一补丁。[本项目控制审查](control.md)和[独立保存数组复算](jacdot-summary.json)提供具体调用与量化证据。

**对 W55 的决定：** 控制层局部补正确公式；保留原 3.2.3 plant。整包升级会同时改变其他物理/实现行为，妨碍此次归因。官方 [3.2.3 API 文档](https://mujoco.readthedocs.io/en/3.2.3/APIreference/APIfunctions.html#mj-jacdot)要求正确的 kinematics/comPos/comVel；当前 runtime 已先 `mj_forward`，没有把版本缺陷误判成漏调用阶段。

## 2. Kalina 等：有限应变黏弹 PANN，2026 年正式发表

**A physics-augmented neural network framework for finite strain incompressible viscoelasticity**，Computer Methods in Applied Mechanics and Engineering 455, 118892，2026-06-15。[出版社与 DOI](https://doi.org/10.1016/j.cma.2026.118892)；可阅读全文的 [2025-11-04 预印本](https://arxiv.org/html/2511.02959v1)。本次方法细节依据公开预印本，发表信息依据出版社记录。

论文 §3.2 明确：预测时网络权重固定，内部变量随每个增量积分。§3.3 按**整条加载历史**划分校准与测试集，训练包含内部演化的积分。物理结构限制和泛化精度是不同层面的要求。

**对本项目的推论：** 冻结权重本身不是错误；需要检查当前网络在真实加载速率、混合边界和释放/重访历史上的自由积分误差。训练与部署的积分方式、输入采样和边界条件必须分别核对。本项目是小应变、可压缩 TM02，不能照搬其不可压缩假设或用有限应变论文替本项目放宽原 `.02` 有效域。

## 3. De Benedittis 等：软接触同时进入规划与 WBC，2025

**Soft Bilinear Inverted Pendulum: A Model to Enable Locomotion With Soft Contacts**，IEEE TSMC: Systems 55(2), 1478–1491，2025；2024-12-04 在线发表。[DOI](https://doi.org/10.1109/TSMC.2024.3504342)、[作者方法与实验页](https://ddebenedittis.github.io/soft_bilinear_inverted_pendulum_astro/)、[官方代码](https://github.com/ddebenedittis/control_quadrupeds_soft_contacts)。

作者指出软接触使接触点移动、反力建立存在延迟，因而在 MPC 和全身跟踪控制两层都建模柔顺性；WBC 将动力学、执行器与接触可行性置于跟踪任务之前。作者页面给出多个四足仿真与 SOLO12 实验。

**对本项目的推论：** 若先修正确运动学后仍出现支撑力/变形/身体加速度不一致，应让因果接触模型贯穿 F 的规划与 WBC，或 N 的执行响应和预测。其线性黏弹接触不是 TM02 持久内态模型，不能直接替换本项目真实材料，也不提供本项目 1 mm 行程内的安全保证。先解决已证公式缺项，再用新轨迹决定是否需要这一层模型升级。

## 4. Patrizi 等：RL-Augmented MPC，2026-03

**RL-Augmented MPC for Non-Gaited Legged and Hybrid Locomotion**。[原文 arXiv:2603.10878v1](https://arxiv.org/html/2603.10878v1)、[官方 AugMPC](https://github.com/AndrePatri/AugMPC)。

高层动作同时给身体运动和逐足 flight 注入，底层显式处理接触日程与全身运动。作者使用 800 并行环境；论文也明确未施加关节力矩限制。策略频率不合适时会出现静止局部解。

**对本项目的推论：** 它提示长期应让材料信息影响落脚、接触时序或载荷分配，而不只影响一个速度 cap。但直接移植会改变当前动作空间、训练任务和约束；不能删除 Go2 原力矩界或照搬其并行数量。W55 先恢复低层稳定，保持当前 PPO 不变；后续的材料记忆学习再单独定义可表达这些决策的接口。

## 5. Zhang 等：Feasibility-aware WBC，2026-09

**Feasibility-aware whole-body control for terrain-adaptive humanoid locomotion**，Autonomous Robots 50, 42，2026-09-28。[全文与 DOI](https://link.springer.com/article/10.1007/s10514-026-10263-6)。

作者让本体感知、NMPC 与 WBC 共用地形描述，随估计坡度调整接触与跟踪优先级。它明确报告仅靠接触后感知不能预见台阶，以及陡坡上先打滑、后失去平衡的失效边界。

**对本项目的推论：** 应检查整个身体与支撑可行性，以及模型误差发生的先后顺序。足底未越过 `−1 mm` 不代表身体仍有可恢复支撑。该方法面向人形及坡面，不是 TM02 四足证明；不能将其坡度权重公式无依据搬入本项目。

## 6. FACET：力敏感目标，CoRL 2025

**FACET: Force-Adaptive Control via Impedance Reference Tracking for Legged Robots**，PMLR 305:705–720，2025。[官方会议论文页](https://proceedings.mlr.press/v305/xu25c.html)。本次核对官方摘要与元数据，PDF 入口未成功解析，不声称完整复现。

其学习目标让策略跟踪虚拟质量—弹簧—阻尼的响应，以显式处理外力下的顺应行为。**对本项目的推论：** 未来学习接口需要关心力与顺应性；当前训练一个 scalar cap 无法自动修复错误的足端加速度公式。W55 不另开这一套端到端控制族。

## 本轮采用顺序

先用官方正确运动学修 F；并行修 N 的支撑耦合响应与整身候选可行性；用真实材料路径检验冻结 PANN 与 observer。只有修后新证据需要时，再引入软接触一致建模或定向材料训练。论文提供方法依据，实际任务、隐状态因果性和原约束决定本项目是否成功。
