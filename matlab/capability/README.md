# MATLAB / Simulink反馈能力判别实验

本目录是**已知模型、经典反馈、无RL训练**的机制小试。研究目的、条件与结果见[报告](../../docs/capability-probe-report-2026-10-16.md)及[冻结协议](../../notes/capability-probe-2026-10-16/protocol.md)。它不替代SPlaTES/ReCo复现，不宣称新的RL算法。

本机已运行MATLAB R2025b和Simulink。MATLAB主研究不依赖额外优化工具箱，MPC为显式有限预算CEM。原生Simulink模型在运行时调用同一MPC函数。

## 从项目根目录重建与运行

```matlab
root = pwd; % 当前文件夹应为本项目根目录
addpath(fullfile(root,'matlab','capability'));
result = runCapabilityStudy(root);
checks = verifyCapabilitySimulink(root);
figures = plotCapabilityStudy(root);
```

主研究输出54条6秒轨迹的精简结果，完整MAT存到`work/capability-probe-2026-10-16/study.mat`。脚本会重写本专题的结果JSON/CSV，因此如需保留不同设置，请先使用独立输出目录或备份已有结果。当前配置和设计/评价任务固定于源码及协议，不能改后仍将结果称作同一冻结试验。

仅打开/运行一个完整在线模型：

```matlab
cfg = capabilityConfig(3);
theta = [6 1.5 8]; % 本轮设计集选定的扩展反馈；不是神经权重
[path,simIn,name] = buildCapabilityModel(string(root),cfg,theta);
open_system(name);
out = sim(simIn);
```

模型保存在忽略目录，构建源码公开可重建；已打开或同名模型不覆盖，不关闭用户已有模型。详见[Simulink接口](README-simulink.md)。

## 文件职责

| 文件 | 实际作用 |
|---|---|
| `capabilityConfig.m` / `capabilityReference.m` / `capabilityDisturbance.m` | 冻结对象、参考、设计与评价扰动；未来扰动不提供给MPC |
| `capabilityControlLaw.m` | 经典共模/差模反馈，返回限幅前需求 |
| `capabilityDerivative.m` / `capabilityIntegrate.m` | 含执行器记忆的物理对象与采样保持RK4 |
| `capabilityMPC.m` / `capabilityController.m` | 根据当前真实状态在线优化，包含采样点资源检查与有限预算失败记录 |
| `capabilitySimulate.m` / `runCapabilityStudy.m` | 完整闭环、设计选点、评价、固定命令探针和数学见证 |
| `buildCapabilityModel.m` / `capabilityControllerBlock.m` | 原生物理Simulink和实时离散控制接口 |
| `verifyCapabilitySimulink.m` | 两组选定反馈在MATLAB与完整在线Simulink上的对应比较 |
| `plotCapabilityStudy.m` | 从已保存轨迹出图，无重新训练或仿真 |

`raw`是限幅前输入，`u`是实际执行输入；两者不得混用。`q0`是非线性阻尼形状尺度，初态来自`cfg.s0`。`macroEvery`为预测命令块长度，`planEvery`为实际重规划间隔；高频对照只改变后者，不能宣称优化了所有高频未来命令。

无可行CEM候选时使用有限罚分排序，记录到`noFeasibleSample`，不保证恢复可行或闭环稳定。角度约束/峰值为0.02s采样点统计。`results.json`为本轮成功运行汇总；`resource-ledger.json`另外记录两次汇总阶段报错的计算开销，后者是本次执行记录，重跑时应按实际运行另行更新。
