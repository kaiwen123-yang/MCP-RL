# 在线MPC与实际反馈的Simulink模型

`buildCapabilityModel`根据同目录的配置、控制器和扰动函数构建模型。模型保存在忽略目录`work/capability-probe-2026-10-16/`，公开源码可重建；构建本身不表示已经仿真通过。该模型用于在明确对象上检查真实反馈、在线规划与输入资源，不在Simulink中训练RL。

```matlab
% 在项目根目录运行；仿真只调用本项目函数，不清空工作区或关闭其他模型。
root = string(pwd);
addpath(fullfile(root, 'matlab', 'capability'));
cfg = capabilityConfig();
theta = [2, 0.7, 0.2]; % 仅为接口示例，实际试验使用冻结协议中的参数
[modelPath, simIn, modelName] = buildCapabilityModel(root, cfg, theta);
out = sim(simIn);
open_system(modelName);
```

`buildCapabilityModel(root,cfg,theta)`返回模型文件路径、`Simulink.SimulationInput`和模型名。配置和策略参数同时存在模型workspace与`SimulationInput`中，不依赖base workspace。已有同名模型或文件时创建新名字，不覆盖或关闭用户模型。初值优先使用`cfg.s0`，否则为`zeros(6,1)`；`cfg.q0`是控制器非线性阻尼形状长度尺度，不是初始状态。模型使用固定步长`ode4`，积分步长为`cfg.integrationStep`，控制周期为`cfg.dt`；二者必须为整数倍关系。

顶层信号结构为：原生连续物理plant → 每`dt`采样真实状态 → 在线MPC与实际反馈 → 两执行器限幅 → 物理plant。Level-2 MATLAB S-Function逐拍调用：

```matlab
[raw, c, plan, info] = capabilityController(s, t, theta, cfg, plan);
```

`s=[p;q;v;r;a1;a2]`，`raw`为2维未限幅执行器输入，`c`为标量宏命令，`plan`为`H×1`未来命令。初始plan为零。控制器每`dt`使用当前Simulink状态，并在内部根据`cfg.planEvery`决定是否在线重规划；`cfg.macroEvery`另行定义预测命令块含有多少低层步。终点`t=cfg.T`只计算反馈，不再求解下一段规划。S函数仅调用控制器，用返回的`info.replanned`记录真实重规划次数；缓存同一采样时刻的计算结果，避免求解器反复调用Outputs时重复求解MPC。它没有持久全局变量，也不读取外部轨迹日志。

物理plant内部的六个原生Fcn与连续Integrator实现：

```text
pdot  = v
qdot  = r
vdot  = (a1+a2)/2 - dv*v
rdot  = (a1-a2)/2 - k*sin(q) - dr*r + w
a1dot = (-a1 + clip(raw1,-U,U))/tau1
a2dot = (-a2 + clip(raw2,-U,U))/tau2
```

限幅在顶层原生Saturation块中实现。扰动由`capabilityDisturbance(t,cfg)`在低层采样时刻计算并在相邻控制拍之间保持，与MATLAB对照端保持一致。扰动没有反馈给控制器输入；控制器只接收6维实际状态和预先固定的配置。`cfg.caseId`定义扰动波形，但是否在规划预测中使用波形，需要独立审查控制器实现；仅模型结构不能证明其没有使用未来信息。

`sim`的`SimulationOutput`包含以下timeseries，采样间隔均为`cfg.dt`：

| 字段 | 列及含义 |
|---|---|
| `stateLog` | `[p,q,v,r,a1,a2]`，采样时刻的真实物理状态 |
| `rawLog` | `[raw1,raw2]`，实际状态产生的未限幅输入 |
| `uLog` | `[u1,u2]`，限幅后进入执行器的输入 |
| `commandLog` | 在线宏规划当前执行命令`c` |
| `disturbanceLog` | 当前控制拍保持的扰动`w` |
| `controllerCountLog` | `[低层控制器求值累计次数,控制器报告的实际重规划累计次数]` |

日志包含`t=0`和`t=cfg.T`。最后一个时刻的控制器输出不再对应一个后续物理积分区间，因此与MATLAB逐步积分比较时，应区分`N+1`个状态和`N`个已执行动作。`out.tout`为求解器时间点，可能比日志采样更密，应使用各timeseries自身的`Time`对齐。控制器计数仅检查调用调度；重规划确实采用在线优化还需核对`capabilityController`源码和实际仿真结果。

该实现没有使用From Workspace播放高层命令、物理状态或执行器动作。与历史learning-pilot模型相比，本模型把高层求解也放入Simulink闭环。一次数值一致性通过，只能支持同一模型、采样、控制器和扰动下的一致性，不能自动支持RL能力提升或机器人实验结论。
