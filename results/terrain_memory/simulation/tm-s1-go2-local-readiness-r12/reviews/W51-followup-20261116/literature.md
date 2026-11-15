# 六项原始文献：哪些经验适用于当前稳定运行问题

核查日期：2026-11-16（Asia/Shanghai）。这是对 [W51 原有五篇文献](../W51-review/literature.md)的补充。下文区分论文实际方法和本项目的推论；不将异机型、异任务或预印本结果当成本仓库资格，不将开源组件等同于完整复现。

## 1. 让摆动运动与承载力在同一问题里一致

**Whole-Body Inverse Dynamics MPC for Legged Loco-Manipulation**，Lukas Molnar 等。IEEE RA-L，2026 卷期（2025 在线）。[实际阅读正文：arXiv v1](https://arxiv.org/html/2511.19709v1) · [正式项目及出版信息](https://lukasmolnar.github.io/wb-mpc-locoman/) · [官方代码](https://github.com/lukasmolnar/wb-mpc-locoman)。

论文式(5–6)同时约束全身逆动力学、支撑足速度/摩擦锥、摆动足零接触力及竖直速度；§V-A2 修正 warm start 的接触相位错位，§VI-D 仍讨论实际接触延迟。

**本项目推论：** D2 的几何正确是必要环节，下一步必须同时观察 native 的卸载和 plant 支撑交换。若几何修好仍四足承重，再研究力与运动学协调。不能给 MuJoCo 的真实足力强行写零；迁移此优化器属于另一个控制变更。

## 2. 接触隐式优化仍依赖参考质量

**Fast Contact-Implicit Model Predictive Control**，Simon Le Cleac’h 等八位作者。IEEE TRO 40，1617–1629，2024，DOI 10.1109/TRO.2024.3351554。[正式作者机构全文](https://msl.stanford.edu/papers/le_cleach_fast_2024.pdf) · [官方实现](https://github.com/dojo-sim/ContactImplicitMPC.jl)。

§IV 在参考附近构造接触近似、保留互补关系并 warm start；§VI-A3 明确指出高质量参考重要，且其常规步行鲁棒性不及带 Raibert 落脚策略的凸 MPC。仓库旧版 BibTeX 不替代正式作者信息。

**本项目推论：** 先修已证实的参考错误有依据。D3 可检查额外初值方向的作用，但双起点没有全局最优或稳定保证；换成接触隐式也不会自动消除参考和初始化问题。

## 3. 全身采样 MPC 也使用有结构的运动参考

**Real-Time Whole-Body Control of Legged Robots with Model-Predictive Path Integral Control**，Juan Alvarez-Padilla 等。ICRA 2025，14721–14727。[实际阅读正文：2024 arXiv v1](https://arxiv.org/html/2409.10469v1) · [官方项目与发表信息](https://whole-body-mppi.github.io/) · [官方代码](https://github.com/jrapudg/RTWholeBodyMPPI)。

§III 使用低维关节样条和 CPU 并行 rollout；§IV-B 报告 walking 参考比 standing 参考更有利于步态鲁棒性。作者当时的 GPU/MJX 延迟结论是其具体实现和硬件的结果，不是普遍 GPU 上限。

**本项目推论：** 更有意义的控制候选与端到端时延值得优先关注。本轮先完成主 D2，D3 按既定计划可选；不能凭论文的硬件成功直接替换当前 iLQG，或照搬其采样数与 horizon。

## 4. RL 能解决什么，取决于它能直接改变什么

**Residual MPC: Blending Reinforcement Learning With GPU-Parallelized Model Predictive Control**，Se Hwan Jeon 等。IEEE TRO，2026；[作者出版记录](https://seungwoo-hong.github.io/publications/)。[实际阅读方法版本：2025 arXiv v1](https://arxiv.org/html/2510.12717v1)，不声称逐字核对最终刊版。

式(23–24)把策略关节残差经 PD 转成力矩，与 MPC 力矩混合；§III-D 使用 CusADi 和 cuDSS 批量求解，部署使用 CPU/OSQP。论文也区分样本效率和墙钟训练成本。

**本项目推论：** 当前 2 Hz scalar cap 没有同样的动作权限；这篇论文不能证明继续训练 cap 会学会逐足卸载。应在稳定基线后，再按失败模式判断动作空间是否够用。此次未核实完整官方 Residual MPC 训练仓库；CusADi 是相关组件，不是整篇实现。

## 5. 更低预测误差不保证更好闭环控制

**Learning Legged MPC with Smooth Neural Surrogates**，Samuel A. Moore、Easop Lee、Boyuan Chen，2026。[实际阅读正文：arXiv v1](https://arxiv.org/html/2601.12169v1) · [官方代码](https://github.com/generalroboticslab/SNS-MPC) · [项目页](https://generalroboticslab.com/SNS-MPC/)。

§IV-B 控制网络敏感度/曲率，§V-A 使用重尾似然，表 VII–VIII 区分单步误差与闭环效果。公开代码含 CPU MuJoCo 数据收集和 JAX 训练，不含完整真实机器人部署工具。

**本项目推论：** 未来若证据指向接触梯度质量，应分别审查状态预测、局部导数和闭环结果；当前没有证据要求更换动力学模型。实验室主页列 IROS 2026、详细项目仍保留 preprint 标签；本报告方法来源固定为 arXiv v1，未据旧标签断言“尚未接收”。

## 6. GPU 需要适合批量执行的完整计算结构

**CusADi: A GPU Parallelization Framework for Symbolic Expressions and Optimal Control**，Se Hwan Jeon 等。IEEE RA-L，2024 在线/2025 卷期，DOI 10.1109/LRA.2024.3512254。[实际阅读正文：arXiv v1](https://arxiv.org/html/2408.09662v1) · [作者机构出版记录](https://biomimetics.mit.edu/publications/cusadi/) · [官方代码](https://github.com/se-hwan/cusadi)。

§III 跨实例批量执行同一表达式；§V-A 式(9–10)以固定求解更新减少分支差异。其示例的模型和求解结构不等于任意全身接触仿真；公开代码核心是表达式生成框架。

**本项目推论：** W51 residual→J 归约正确，只证明一段计算。后续 GPU 后端需保留状态、接触、材料及求解器含义并衡量端到端收益；不在 W52 中重跑归约占卡，也不把缺项填零。

## 与当前实验的关系

这六项工作共同帮助区分：参考是否合理、候选是否动力学可行、预测与执行是否一致、学习策略是否有必要动作权限、加速是否覆盖真正瓶颈。它们不构成六个立即并行更换的方法。

当前先完成既定主 D2、可选 D3 和独立 C。D2 若已经产生持续正常支撑推进，就优先冻结简单实现进入原任务；若仍失败，按[主审的证据分流](README.md)定位后，再选择一个有证据支持的机制变更。材料记忆的因果贡献仍需稳定任务后的匹配对照。
