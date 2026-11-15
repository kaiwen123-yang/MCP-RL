# W50 后的一手文献：从成本下降走向可执行步态

更新：2026-11-15（Asia/Shanghai）。结果基点：`09bd55e0cb70244938ad82db7dd202a31ac70f77`。本文件为文献与机制审查，没有运行新物理、求解、GPU 仿真或 PPO；下一轮实际执行合同以 W51 主计划为准。

## 1. 文献要回答的具体问题

[W50 Task A](../../w50/task-a.json)已经实测旧 reference/K/H 来自初始 Stand，且 private gait 在 2.002 s 切换到 Trot、相位继续前进；因此“参考冻结”和“相位没走”是不同问题。[Task B](../../w50/task-b.json)的当前时域候选虽然降低 J，早期两个 4.3 s 左右的问题仍坍低且倒退加重。W50 没有从 t=0 运行 r38 原闭环，因此它没有检验“从起步持续刷新能否预防失效”。[phase](../../w50/phase.json)

源码与日志还提示：moving gait 主要通过足 z 残差表达，目标摆动相位与实际承重长期共存。需要区分三种机制：缺少跨出当前接触状态的候选；可迈步候选被当前目标排序压过；候选预测与实际执行不一致。下面四篇文献直接关联这些问题，未重复上一轮的 BiConMP、CAFE-MPC、Bilevel gait 和 Layered MPC。

## 2. 四篇直接相关文献

### 2.1 Whole-Body Model-Predictive Control of Legged Robots with MuJoCo

John Z. Zhang 等，2025 预印本，arXiv:2503.04613；本次查阅作者页仍标注 under review，不把它写成已确认期刊论文。

- [论文全文](https://arxiv.org/html/2503.04613v1)
- [作者项目页](https://johnzhang3.github.io/mujoco_ilqr/)
- [官方部署接口](https://github.com/johnzhang3/mujoco_mpc_deploy)，其 README 说明 solver/task 在官方 MuJoCo MPC 中。

**论文事实。** §V 使用 Go1/Go2 实机，默认 horizon 0.35 s，规划约 50 Hz、时变反馈约 300 Hz。§VI 指出局部 iLQR 对接触模式探索与初始猜测敏感，步态参考通常通过任务成本提供；§IV-D 的动力学导数采用前向差分，并报告其中心差分在行走任务中失效。

**本项目含义。** 0.35 s 曾在相关平台成功，不能仅因它小于一个 2 Hz 周期就判定必然不足。W50 的动力学 FD 已是单边 FP64；raw-center Jacobian 的零物理中心差分不是同一件事，不重开这条已回答的支线。应核实际足部运动与成本所表达的运动意图。作者的接触/模型/任务和本项目不同，其成功不构成这里的资格。

### 2.2 Contact-Implicit Model Predictive Control: Controlling Diverse Quadruped Motions Without Pre-Planned Contact Modes or Trajectories

Gijeong Kim 等，IJRR 44(3):486–510，2025；2024 年在线发表，arXiv:2312.08961v2 为接受版本。

- [接受版全文](https://arxiv.org/html/2312.08961v2)
- [KAIST 正式出版记录](https://pure.kaist.ac.kr/en/publications/contact-implicit-model-predictive-control-controlling-diverse-qua/)
- DOI：[10.1177/02783649241273645](https://doi.org/10.1177/02783649241273645)

**论文事实。** §5.1.3 针对最小抬脚与擦地，使用

\[
l_f=c_f\sum_i \sigma(-30\phi_i)\lVert v_{i,t}\rVert^2.
\]

其中 \(\phi_i\) 是足间隙，\(v_{i,t}\) 是足切向速度；低间隙时移动代价较大。论文还用 air-time 与配对对称项引导步态，并用 FDDP 的状态—控制初始化应对单射击接触模式偏离。松弛梯度用于 backward，forward 保留其硬接触模型。

**本项目含义。** 全足静止也使上述 \(l_f=0\)，所以单独添加“防滑代价”不能保证抬脚，甚至可能进一步偏向不动。当前应先给原目标一个有明确支撑/摆动运动意图的候选，观察其实际接触和原 J 排序。FDDP、硬接触模型与新增多项成本是较大方法变化，不在同一轮全部搬入。没有在作者可核实入口确认完整 HOUND 复现代码，不把其他 ContactImplicitMPC 仓库当成本论文实现。

### 2.3 Hybrid iLQR Model Predictive Control for Contact Implicit Stabilization on Legged Robots

Nathan J. Kong 等，IEEE T-RO 39(6):4712–4727，2023。

- [全文](https://arxiv.org/html/2207.04591v2)
- [作者实验室出版记录](https://www.cmu.edu/me/robomechanicslab/publications.html)
- DOI：[10.1109/TRO.2023.3308773](https://doi.org/10.1109/TRO.2023.3308773)
- [作者实验室 Saltation-Tutorials](https://github.com/robomechanics/Saltation-Tutorials)：是基础教学示例，不能称完整四足 MPC 复现包。

**论文事实。** §III–IV 分别处理接触时序改变下的导数、参考和成本失配。事件驱动模型使用同模式参考延拓；论文明确指出时间步进接触存在中间状态，不能直接照搬该延拓，四足示例采用随实际接触力改变输入罚权的处理。其例子也显示局部梯度不一定会新增原 seed 中没有的事件。

**本项目含义。** “时钟相位正确”仍可能与实际接触不一致；参考、候选和受力应一起对齐。MuJoCo 的时间步进软接触不等于该文的事件驱动模型，不能直接加 saltation 或复制延拓公式就宣称修复。本轮先增加相位有结构的运动候选并观测真实承重，不新增接触硬门槛。

### 2.4 Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing

Haoru Xue 等，DIAL-MPC，ICRA 2025；arXiv:2409.15610。

- [论文全文](https://arxiv.org/html/2409.15610v1)
- [正式出版](https://ieeexplore.ieee.org/document/11127320/)
- [项目页](https://lecar-lab.github.io/dial-mpc/)
- [官方代码](https://github.com/LeCAR-Lab/dial-mpc)

**论文事实。** DIAL-MPC 用轨迹级与时域位置相关的退火采样兼顾探索和局部收敛。其论文实验采用 2048 个并行环境、0.4 s horizon、RTX 4090，并关闭足以外部位的接触；实现基于 JAX/Brax，直接输出力矩。

**本项目含义。** 可以借鉴“给搜索增加有信息的候选、将独立候选批处理”的思想。不能用这些吞吐或成功结果宣称当前 Go2 全碰撞/PD/TM02/native J 已能在 GPU 等价运行，也不能为了占满 96 GB 直接把原控制器整体替换成 DIAL。下一轮优先做已有 native 残差的只读批量覆盖与数值分析；未来 GPU proposal 仍需 CPU 原 native 最终裁决。

## 3. 对 W51 的机制辨识建议

以下是**本项目提出的实验设计**，不是上述论文原样算法或已经验证的结论。

### 同一早期起点，分清刷新自身收益与运动 seed 增益

优先用同一 A / seed1 的完整初始 checkpoint、相同 RNG 与环境，平行运行 t=0 到 8 s 两条诊断闭环：

1. D0：R38 当前时域刷新路径。
2. D1：以一个完整相位感知运动候选 S 作为替代线性化起点；其余目标、物理、步态和严格接纳规则保持。

两条 D8 都使用正常 plant/RPC 入口，不复用 W50 错库 bridge，也不再设置 40 ms 准入环节。D0 每 epoch 通常最多 20 个 full rollout；D1 多 1 个 S rollout，通常最多 21 个，仍只有同一条 first-valid backward 方向及九个正 alpha。总 rollout 工作量不完全相同，不能把它写成同预算吞吐胜利；FD、rollout、实际墙钟各自记录。

这样可以区别从起步运行 r38 本身的预防效果，与追加 seed 的增量。8 s 是物理观察窗，不是墙钟截止，也不算 E30 通过。两个 arm 都需报告实际安装及 reference/K/H 来源；若没有改善候选而保留旧对象，不能只根据配置声称参考已刷新。

若 R38 本身已建立有效步态，优先保留简单机制；若追加 seed 有增益，报告其整体增量，不把它引起的后续参考更新单独冒称已识别的独立因果因素。两者都失败时，已有结果仍能区分目标排序、候选生成或执行失配，不重复 W49 原轨迹。

### 一个完整运动候选作为线性化起点

每个 epoch 先保留原十个 nominal 的比较与最优完整 B。用 B 当前 rollout 的 12 维关节位置 \(q_{B,j}(t)\) 构造相位足路径，再得到限位内运动学目标 \(q_{\rm IK,j}(t)\)。S 复制完整 B，仅增加一个时变前馈曲线：

\[
\delta q_{\rm seed}(t)=q_{\rm IK,j}(t)-q_{B,j}(t),\qquad
c_S(x,t)=c_B(x,t)+\delta q_{\rm seed}(t).
\]

\(c_B\) 必须是完整旧策略的 pre-clamp/pre-QP raw center，保留其承重 PD 偏置、原反馈与既有 delta；不能用裸 IK 角替换它。用原 native predictor 对完整 S 做一次 rollout。S 有效时，可以在其轨迹附近执行原唯一 first-valid backward 和九个正 alpha；S 即使原 J 高于 B，也可以作为搜索起点，但不能因此安装它。S 无效时回到 B，不额外再跑一套方向或 alpha 家族。

全局严格比较对象始终是同问题的原 B，S 和所有后续正候选都经过原 J/risk/strict 裁决。相对 S 的下降不能冒充相对 B 的改善；没有更好的候选则保留 B。R38 的当前时域表示沿实际选中的线性化轨迹构造，并如实记录 K/H 的混合来源。这里只增加一个运动候选及替代初始化机制，不把 seed 与 Newton k 相加后再共同缩放，也不做 alpha×beta 网格。

足 xy 的基本几何为：支撑足锚定本次 stance 开始时的实际世界位置并跨 replan 保持；当前 swing 的 touchdown 在摆动开始或首次接入时固定，摆动足向与原速度命令一致的触地点运动。可用

\[
p_{{\rm td},i,xy}=c_{{\rm ref},xy}(t_{\rm td})
+R_z(\psi_{\rm ref})\,r_{{\rm nom},i,xy}
+\tfrac12 T_{\rm stance}v_{{\rm cmd},xy},
\quad T_{\rm stance}=0.45/2=0.225\ {\rm s}.
\]

该 touchdown 式是这里的 Raibert 型候选设计，不宣称出自上述四篇的同一公式。原绝对 gait clock、脚顺序、2 Hz/.45/C2/.005 参数保持。重规划接续当前摆动及原下一 touchdown，不能每 20 ms 从相位零重新起步；当前已在摆动的脚从实测足位置/速度用剩余 swing 时间确定性接续，不能让 seed 首节点跳到不同的 C2 参考；原 Gait 成本参考保持。足 z 使用原地面/支撑参考及足几何，不能把足中心高度当足底间隙。

运动学 IK 只提供候选，不证明动力学可行。报告实际摆动卸载、支撑脚切向移动、足间隙、身体高度及振荡、前进/倒退、非足承重、材料加载与原 J 分项，不新增“每刻固定对角接触”或“不得短腾空”的资格条件。5 mm 是否足以生成正常步态尚缺证据；若此构造在原幅值下失败，只能报告该方案失败，不能证明所有正常步态都不可能。

### 负结果怎样决定下一项改变

- 候选已产生有效卸载/支撑交换并在原 J 下更好，旧搜索找不到：支持候选生成或初始化方向。
- 候选确实更接近稳定前进，但原 J 系统性拒绝：才有具体依据选择一项足参考/目标结构修改，并保留原目标结果作对照。
- native 预测改善，真实同对象执行偏离：先定位执行/动力学一致性；W50 已知的动态库加载修复只解决该技术问题，不自动解释早期 native 坍低。
- 所有候选都不能形成有效支撑交换：检查候选运动学构造、原幅值下实际间隙与接触可达性，再选择一项变化；不同时叠加新成本、步态、horizon、PPO 和新求解器。

本轮优先解决正常承重与前进，随后才有条件开展跨足/返程材料记忆收益和 GPU 主规划资格。论文提供机制线索，不替代本项目实际原任务验证。
