# 冻结Go2执行、连续任务与事实响应结构

这套代码已在服务器MuJoCo 3.3.7/CPU上实际运行。结果和边界见[主报告](../../docs/go2-frozen-run-2026-10-19.md)。没有硬件接口或自动调度。

后续的[紧凑事实模型比较](../../docs/go2-compact-response-2026-10-19.md)也已完成：`compact.py`提供26534参数的递推响应，`scripts/fit_go2_compact.py`只拟合模型；两种固定最终检查点在[artifacts](../../artifacts/go2-compact-2026-10-19/README.md)。普通单步和多步模型均保住3/3成功及低于200ms，但多步模型没有任务cost优势。策略保持冻结。

- `policy.py`：完整TorchScript estimator+actor、六帧历史、previous action、固定观测/动作缩放和关节顺序。
- `simulation.py`：实际5ms MuJoCo/PD、20ms策略，完整状态与历史复制、全部碰撞几何、力矩与机械功积分。预测与执行使用同一适配。
- `task.py`：连续路点/门/终点条件，真实时间戳停稳，明确剩余任务启发式。
- `planner.py`：相同预算CEM的理想速度模型与策略诱导物理预测。后者是名义物理充分信息对照，未学习紧凑g。
- `realized_learning.py`：事实预测损失、固定参考下实际响应代价、完整任务return、score/PPO目标。所有学习函数计算目标或梯度，不含优化器或自动训练。

## 实质结构变化

旧问题是固定策略时自由最小化预测器自评任务cost。新结构中，规划的响应必须来自当前反馈：

\[
R_\theta(h,c)=\operatorname{rollout}(F,\pi_\theta,h,c).
\]

本轮F由名义MuJoCo承担，保留整机状态/策略历史。未来紧凑g的事实目标为

\[
L_g=\mathbb{E}\|g_\phi(h,c)-Y_{\mathrm{executed}}\|_W^2,
\]

标签停止梯度；g不通过压低自己的预测任务cost获得更新。策略目标可以是

\[
\mathbb{E}_{\tau\sim\pi_\theta}
[J_{\mathrm{task}}(\tau)+\lambda D(Y_\tau,\operatorname{stopgrad}(g_\phi(h,c)))],
\]

其中实际响应Y与任务J均来自真实执行，参考版本固定，必要条件信息必须对actor/critic可见。代码用完整实际cost-to-go和真实采样动作的log probability计算score/PPO目标。确定性导出策略没有行为密度，本轮只接事实指标与实际回报，不伪造PPO样本。

与相同目标/数据/模型信息下的标准PO相同，不包装成新优化原理。与命令TO不同的可学习变量是反馈参数，但当前冻结反馈；命令MPC结果不属于反馈学习增益。紧凑响应模型拟合已经在后续轮次完成；随机Go2策略采集/critic/训练器与共同学习收益仍未运行，导出策略权重不等于可续训checkpoint。

## 使用

固定配置在`configs/go2_frozen_tasks.json`。资产采用主报告所列固定commit，权重和模型文件留在忽略目录。运行器`python scripts/run_go2_frozen.py --asset-root ... --mode tasks --output ...`只执行冻结仿真；输出目录需是新目录以保留先前证据。默认仅CPU、单线程、affinity6/7、nice10；换服务器需先按其资源情况明确调整运行配置，不能将其视为通用核心绑定要求。

直接机制测试：`python -m unittest discover -s tests -p 'test_go2_*.py'`。测试针对事实标签梯度、完整终端信用、标准PO等价关系及停稳时间戳边界，不代替真实任务运行。
