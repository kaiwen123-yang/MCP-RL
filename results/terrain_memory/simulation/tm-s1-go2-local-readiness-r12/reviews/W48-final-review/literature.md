# W49新增一手文献与作者接口核对

2026-11-13，基点`61e16975fa6b38dc8e2d31779c2bd82f1aaeb8b6`。本轮在W47已有论文之外增加以下四篇。读取原文并核对可获得的官方实现入口，没有复现论文，没有把论文结论当作本项目已建立的定理。正式发表信息与预印本版本分别保留；未查询或声称期刊分区。

## 1. 实际触地反馈与接触恢复

**Ruben Grandia等，Perceptive Locomotion Through Nonlinear Model-Predictive Control.** IEEE Transactions on Robotics, 39(5):3402–3421, 2023. DOI：[10.1109/TRO.2023.3275384](https://doi.org/10.1109/TRO.2023.3275384)。[原文](https://arxiv.org/pdf/2208.08373)，[机构记录](https://www.research-collection.ethz.ch/entities/publication/ed499b50-44d5-4bd7-95a3-438fbc64a940)，[官方OCS2](https://github.com/leggedrobotics/ocs2)。

论文§VI-A对早触地与晚触地提供不同响应：缺失计划接触时执行向下恢复，实际接触信息反馈规划；§VII-D仍指出外部固定gait对某些强扰动的局限。OCS2是官方优化工具箱，不等于整套论文硬件/感知系统已可直接复现。

**本项目的可检验推论：**利用W48逐held接触/GRF和accepted状态，区分计划stance、实际几何接触与真实承重。若实际承重与计划接触持续不符，或accepted计划本身未维持支撑，则有依据设计接触恢复；需要同时进入预测和部署，不能只在plant端加隐形补丁。当前Go2位置目标+PD+H-QP与ANYmal WBC接口不同，不照搬其硬件控制器。

## 2. 目标下降与稳定性条件需要分别验证

**Mohamed Elobaid等，Adaptive Non-linear Centroidal MPC with Stability Guarantees for Robust Locomotion of Legged Robots.** IEEE Robotics and Automation Letters, 10(3):2806–2813, 2025. DOI：[10.1109/LRA.2025.3536296](https://doi.org/10.1109/LRA.2025.3536296)。[作者最终PDF](https://iit-dlslab.github.io/papers/elobaid25ral.pdf)，[可读全文v2](https://arxiv.org/html/2409.01144v2)，[官方实现](https://github.com/ami-iit/paper_elobaid_2024_stable-centroidal-mpc)。

论文将自适应反馈、稳定性及角动量条件纳入质心MPC。理论有常值扰动、参考与递归可行性等条件，不是所有接触系统无条件稳定保证；官方仓库提供相关对照入口。

**本项目的可检验推论：**r34 strict只能说明当前问题接纳规则成立。W48大部分调用下降但仍跌倒，必须检查accepted计划中的高度、姿态、支撑、推进和各成本项；不能仅增加“J下降率”作为成功指标。当前full-body控制若以后引入稳定性约束，必须重新推导其状态、输入和可行性条件，不能把CLF名称当作已有证明。

## 3. 学习模型之前先对齐数据与控制语义

**Chuong Nguyen等，Variable-Frequency Model Learning and Predictive Control for Jumping Maneuvers on Legged Robots.** IEEE Robotics and Automation Letters, 10(2):1321–1328, 2025. DOI：[10.1109/LRA.2024.3519864](https://doi.org/10.1109/LRA.2024.3519864)。[全文v2](https://arxiv.org/html/2407.14749v2)，[作者PDF](https://erl.ucsd.edu/ref/Nguyen_VariableFrequencyMPC_RAL25.pdf)，[官方代码](https://github.com/DRCL-USC/Learning_MPC_Jumping)。

论文让采集、模型学习与MPC使用一致的分阶段时间步、horizon和接触日程；连续跳跃另有落地衔接阶段。它研究跳跃，不能据此断言当前行走必须延长.35s horizon。

**本项目的可检验推论：**不同查询时间、不同实际held动作下的状态差不是纯动力学残差。W49先区分“同输入、不同预测/plant模型”和“同策略、不同执行路径”；材料残差学习只在输入与状态语义明确后讨论。否则学习器会试图补偿接口或控制策略差，难以迁移到另一版控制器。

## 4. RL能否恢复受其动作权限约束

**Siddhant Gangapurwala等，RLOC: Terrain-Aware Legged Locomotion using Reinforcement Learning and Optimal Control.** IEEE Transactions on Robotics, 38(5):2908–2927, 2022. DOI：[10.1109/TRO.2022.3172469](https://doi.org/10.1109/TRO.2022.3172469)。[全文v3](https://arxiv.org/html/2012.03094v3)，[Oxford记录](https://ora.ox.ac.uk/objects/uuid%3A5c36ed3f-8d21-4087-801a-90f7bb3d4773)，[作者主页](https://gangapurwala.com/)。

RLOC把足点规划、关节力矩残差和高速关节恢复分为不同学习模块；恢复控制有直接关节目标权限。本轮未核到独立完整RLOC官方源码，不能将作者另一LFMC项目当作它的复现代码。

**本项目的推论：**当前2Hz scalar cap可以调节前进需求，但无法直接生成落点、支撑延长或快速关节恢复。增加同动作空间PPO样本不会新增这些权限。先稳定原控制链，再针对材料历史实际需要新增一种可解释动作，并以匹配对照评估；不把多模块大改混入W49。

## 5. 作者实现中的缺省GetAction时间

已通过GitHub读取官方`mujoco_mpc_deploy`的[go2interface.py](https://github.com/johnzhang3/mujoco_mpc_deploy/blob/main/src/go2interface.py)：`launch_mjpc_gui`先SetState(qpos/qvel)，随后GetAction不传time。所读blob为`e3b843b4b8885760f8d08406f8a324d8ed0e0aee`。这说明有可用API路径，不说明其异步硬件循环等同于本项目同步仿真。

固定上游commit `e00c47a5adb9856af2e0f24231bb3a60d5be23c4`：

| 文件 | 已核行为 |
|---|---|
| [agent.py](https://github.com/johnzhang3/mujoco_mpc/blob/e00c47a5adb9856af2e0f24231bb3a60d5be23c4/python/mujoco_mpc/agent.py) | get_action默认time=None |
| [agent.proto](https://github.com/johnzhang3/mujoco_mpc/blob/e00c47a5adb9856af2e0f24231bb3a60d5be23c4/mjpc/grpc/agent.proto) | State.time为double，请求time为optional float，响应动作为float |
| [grpc_agent_util.cc](https://github.com/johnzhang3/mujoco_mpc/blob/e00c47a5adb9856af2e0f24231bb3a60d5be23c4/mjpc/grpc/grpc_agent_util.cc) | 未传time则使用agent state.time，再求值策略 |

该工具源码加仓库Reset补丁得到r36实际源SHA `94b7328afb705bb226ee4dd306defce1a3f09972c4432da46bef8ee4c033e5c1`，与build一致。r36清单没有单列安装包protobuf描述符hash，下一执行端需读回，不冒称本次已检查D57包。

### 历史约束：不是W49新发现

[W39审查](../W39-raw-review/control-audit.md)已量化旧A/B请求时间舍入，最大约.473/.946微秒。首大反馈时，量化参考速度差范数约4.37e−6/9.22e−5，真实参考速度差约1.214/1.408；晚期敏感策略起点动作插值差曾约1.60e−4/3.76e−3rad。W40/r29–r34也明确讨论动作float32。

因此W49只先在新的实际事件上量化；省略显式time可单独消除请求时钟舍入，不改响应动作精度。若它不能解释主要分叉，报告负结果并继续恢复线。不要再次仅凭微小时间或零端点改动启动一整轮相同轨迹。

## 6. 文献如何进入本轮决策

这些文献共同支持把执行语义、接触恢复、优化目标和学习权限分开，但并未给出本项目的唯一故障答案。W49的两个CPU工作包先利用现有事件确定因果先后，再只测试一个控制机制。冻结反馈而改变feedforward的有限候选属于本项目待检验设计，不是声称上述论文已经验证了相同实现。

所有最终预算、条件和输出由[W49主计划](../../../../../../docs/S1_W49_STABILITY_PLAN.md)统一；本文不另加实验，不要求移植整套外部控制器，也不以文献成功率替代本项目原资格。
