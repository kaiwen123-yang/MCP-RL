# 学习策略的MATLAB与Simulink闭环核验

此模块读取一个已冻结的Python actor，在MATLAB及真实`sim`调用中重新计算状态反馈动作。它不训练策略，不替代策略为经典teacher，不回放Python动作，不复现高层MPC。

Simulink结构为：真实物理状态Unit Delay → 12-32-32-1神经actor → 非线性执行器RK4映射 → Unit Delay。两个离散Level-2 MATLAB S-Function在Simulink调度中运行，命令和未知扰动由From Workspace按0.02秒保持。实际动作由当前Simulink状态产生，物理状态没有被Python记录值覆盖。

导出MAT文件包含：

| 字段 | 形状/含义 |
|---|---|
| W1,b1 | 32×12、32×1 |
| W2,b2 | 32×32、32×1 |
| W3,b3 | 1×32、1×1 |
| initial_state | 3×1，第一个动作之前的真实[p,v,a] |
| command | N×1，通常N=100，每个低层步的高层命令 |
| disturbance | N×1，同一步内保持的真实外扰 |
| state_expected | N×3，每个动作执行后的真实状态 |
| action_expected | N×1，对应步骤的真实动作，仅用于误差比较 |

网络权重使用PyTorch Linear的`out_features × in_features`布局。计算使用single。对象保持当前pilot的`tau=.12,k=1,d=.2,drag=.1,U=1.2`；RK4每个0.02秒控制步使用两个子步。

运行：

```matlab
% 在仓库根目录运行；导出文件保留在忽略目录。
addpath(fullfile(pwd, 'matlab', 'learning_pilot'));
exportFile = fullfile(pwd, 'work', 'learning-pilot-2026-10-15', ...
    'formal', 'seedrun-11', 'seed-11', 'joint', 'matlab_export.mat');
result = runLearningPilotReplay(exportFile);
```

输出在数据文件旁的`matlab-simulink`目录，包括已保存的SLX模型、三端状态/动作误差JSON和MAT数据。源代码检查和实际运行分开；只有`result.passed=true`且工具运行成功，才表示本次导出轨迹的数值闭环核验通过。

数值容差5e-5用于single跨语言核验，不是理论鲁棒误差界。该实验继承高层命令序列，因此检验的是冻结神经反馈、执行器、采样和物理更新的一致性。它不证明未知机器人准确性、接触安全或高层重规划的独立复现。
