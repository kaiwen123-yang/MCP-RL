# 正式冻结策略的MATLAB与Simulink核验

日期：2026-10-15。检查对象是正式学习运行的seed 11、joint组、冻结评估第一个batch的第一个环境，共100个低层控制步。检查使用训练完成后的实际网络权重，没有重新训练或挑选更好的checkpoint。

**数值闭环一致性通过；该episode的两项经验实现条件没有全部通过。** 两者不能合并成一个“控制保证通过”结论。

## 实际运行了什么

对象为带非线性阻力、执行器滞后和物理限幅的三状态系统：

\[
\dot p=v,\qquad
\dot v=a-\sin p-0.2v-0.1v|v|+w,
\qquad
0.12\dot a=\operatorname{sat}_{1.2}(u)-a.
\]

控制周期0.02秒，每步用两个RK4子步。控制器为实际训练的12-32-32-1、1505参数MLP，输出完整动作`1.2*tanh(network(features))`。输入特征仅由真实状态与当前命令构成，不读取响应设计、虚拟参考、全局任务目标或真实扰动，也没有加入经典teacher残差。

Simulink模型使用原生Unit Delay保存真实物理状态，用两个离散Level-2 MATLAB S-Function分别运行冻结神经actor和非线性执行器更新。状态直接反馈到actor，动作每步由当前Simulink状态计算。From Workspace只提供记录的高层命令与外扰，未输入动作日志或覆盖真实状态。

确实执行了MATLAB R2025b的`sim`，不是只生成SLX或检查语法。MATLAB回放与Simulink共用两段MATLAB计算核，因此其零差主要验证调度、反馈接线与时序；Python到MATLAB的差异核对另外覆盖了网络、特征和物理更新的跨语言实现。

入口：[runLearningPilotReplay.m](../../matlab/learning_pilot/runLearningPilotReplay.m)。模型生成器：[buildLearningPilotModel.m](../../matlab/learning_pilot/buildLearningPilotModel.m)。原始权重、导出MAT及SLX留在忽略的工作目录，不作为公开研究代码提交。

## 正式结果

| 项目 | 最大绝对差 |
|---|---:|
| MATLAB—Python真实状态 | 1.1920929e-7 |
| MATLAB—Python实际动作 | 2.3841858e-7 |
| Simulink—Python真实状态 | 1.1920929e-7 |
| Simulink—Python实际动作 | 2.3841858e-7 |
| Simulink—MATLAB真实状态 | 0 |
| Simulink—MATLAB实际动作 | 0 |
| Simulink采样时间 | 0 |

计算采用single；预先声明的跨语言数值容差为5e-5。此容差不是鲁棒控制误差界。

## 两项实现指标另行核对

从真实状态每10步初始化一次响应预测，比较完整段内prefix。真实物理状态不重置。当前正式响应采用κ范围[3,14]和jerk限值±8，与冻结运行的源码配置一致。

跟踪指标使用更新后的真实状态与响应状态：

\[
q_1=\frac1N\sum_k[(p_{k+1}-r_{k+1})^2+0.05(v_{k+1}-s_{k+1})^2].
\]

输入指标使用该步更新前计算的参考物理输入：

\[
q_2=\frac1N\sum_k[u_k-u_{\rm ref}(z_k,c_k,\eta)]^2.
\]

[learningPilotRealizationCheck.m](../../matlab/learning_pilot/learningPilotRealizationCheck.m)独立重写了当前公式、响应RK4和macro重初始化。与Python相同导出轨迹的结果为：

| 指标 | MATLAB | Python | 差值 | 经验预算 | 判断 |
|---|---:|---:|---:|---:|---|
| q1 | 1.6314827e-5 | 1.6314842e-5 | 1.4552e-11 | 2e-4 | 此episode满足 |
| q2 | 0.106051959 | 0.106051944 | 1.4901e-8 | 0.03 | 此episode不满足 |

q2超预算表示**实际动作与响应模型所要求输入之间的均方不一致过大**，不是说实际饱和器输出超过了1.2。这个不一致不能因为状态跟踪指标较小而被忽略，也不能因为MATLAB和Simulink一致而被判作实现保证成立。

## 结论边界

- 已通过：冻结完整动作actor、真实执行器、非线性更新、采样时序的跨语言数值一致性，以及q1/q2公式一致性。
- 未通过：本episode的q2经验实现预算；数值一致性不改变这一事实。
- 未验证：全域可实现性、跨任务概率保证、真实机器人模型准确性或高层MPC的独立复现。高层命令由Python的真实规划日志提供，本次没有在MATLAB重新求解MPC。
- 这是一条冻结评估轨迹的实现检查，方法间性能判断仍使用完整三seed、五组的正式结果，不能把这一条轨迹当作全部评估样本。

不含私人绝对路径的结构化结果：[matlab-simulink-check.json](matlab-simulink-check.json)。其中分别记录`passed=true`（数值实现）和`realization_constraints_empirically_satisfied=false`（该episode的实现预算），并保留导出文件SHA256。
