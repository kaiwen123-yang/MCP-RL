# 非线性物理模型的 Simulink 回放

`buildNonlinearPlantModel.m` 用原生 Simulink 块构建以下对象：

\[
\dot p=v,\qquad
\dot v=a-k\sin p-dv+w,\qquad
\tau\dot a=-a+\operatorname{sat}_{[-U,U]}(u).
\]

三个状态由独立 Integrator 块实现；非线性、输入限幅、执行器滞后和扰动在图中均为独立可检查的块。没有用 MATLAB Function 块包装整个积分器。

这个模型用于**独立重放给定控制序列与扰动、核验 MATLAB 积分实现**。它当前不含 MPC、RL 或联合优化器；回放一致并不证明控制器闭环性能，也不等于已经验证真实机器人模型。

## 接口与运行

```matlab
addpath(fullfile(pwd, 'matlab', 'nonlinear'));
plant = struct('k', 1.2, 'd', 0.25, 'tau', 0.15, ...
    'U', 2.0, 'h', 0.005, 'x0', [0.2; 0; 0], 'tStop', 2.0);
t = (0:plant.h:plant.tStop)';
uData = [t, 0.5 * ones(size(t))];
wData = [t, 0.1 * (t >= 0.75 & t < 1.25)];
outputDir = fullfile(pwd, 'work', 'nonlinear-2026-10-15', 'simulink');
[modelPath, simIn] = buildNonlinearPlantModel( ...
    outputDir, plant, uData, wData, true);
simOut = sim(simIn);
time = simOut.pLog.Time;
states = [simOut.pLog.Data, simOut.vLog.Data, simOut.aLog.Data];
```

生成文件为 `outputDir/codesign_nonlinear_plant.slx`。输出状态顺序是 `[p,v,a]`。另外记录 `uLog`（限幅前命令）、`uSatLog`（限幅后执行器目标）、`wLog`，都为 `timeseries`。

所有物理参数、初始条件与输入位于模型工作区的 `plantParams`、`uData`、`wData` 中，并同时写入返回的 `SimulationInput`。不会清理 base workspace、关闭其他模型、修改用户现有模型或更改全局 Simulink 配置。

首次构建要求同名模型未加载，输出文件不存在。后续试验应重用该模型，更新 `SimulationInput`，而不是重建或覆盖模型：

```matlab
simIn2 = simIn.setVariable('uData', nextUData, ...
    'Workspace', 'codesign_nonlinear_plant');
simIn2 = simIn2.setVariable('wData', nextWData, ...
    'Workspace', 'codesign_nonlinear_plant');
simIn2 = simIn2.setVariable('plantParams', nextPlant, ...
    'Workspace', 'codesign_nonlinear_plant');
simIn2 = simIn2.setModelParameter( ...
    'FixedStep', num2str(nextPlant.h, 17), ...
    'StopTime', num2str(nextPlant.tStop, 17));
simOut2 = sim(simIn2);
```

## 与 MATLAB RK4 比较时的时间约定

采用固定步长 `ode4`；两个 From Workspace 块的采样时间均为 `plantParams.h`，`Interpolate='off'`，末值保持。输入数据从 `t=0` 开始，变化时刻应位于步长网格。

第 `n` 个区间 `[t_n,t_{n+1})` 使用 `uData(n,2)`、`wData(n,2)`。独立 MATLAB RK4 的四级右端函数必须使用这一对保持值，不能在第四级提前读取下一个区间的命令。`tStop` 处记录的输入是该时刻的样本，终点状态则由前一个区间积分得到。

若把 From Workspace 设为连续采样，四级积分可能在输入跳变处采用不同采样约定；它与整步 ZOH 的 RK4 不再是同一个离散化。因此这里明确使用离散采样输入。

物理参数或输入日志错误不会被“自动修复”。至少核对 `uSatLog` 与手工限幅结果、状态时间网格，以及独立积分的逐状态最大绝对误差，再报告一致性。步长缩小只核验数值离散化，不证明被建模对象的真实性。

## 核对的官方文档

- [From Workspace：数据来源、采样、ZOH 与末值保持](https://www.mathworks.com/help/simulink/slref/fromworkspace.html)
- [模型工作区 assignin](https://www.mathworks.com/help/simulink/slref/simulink.modelworkspace.assignin.html)
- [SimulationInput.setVariable：指定模型工作区的参数覆盖](https://www.mathworks.com/help/simulink/slref/simulink.simulationinput.setvariable.html)
- [To Workspace：timeseries 输出](https://www.mathworks.com/help/simulink/slref/toworkspace.html)

文档访问日期为 2026-10-15。运行验证由主任务在用户 MATLAB 会话中执行；仅生成此构建脚本不等于已经完成仿真。

## 低层反馈闭环模型

`buildNonlinearClosedLoopModel.m` 额外提供真正的低层反馈闭环：Simulink 内的实际状态进入反馈律，采样后形成物理输入；不回放预先记录的低层控制量。

```matlab
cl = plant;
cl.x0 = [0.2; 0; 0; 0; 0; 0]; % [p,v,a,c,z1,z2]
cl.omega = 1.2; cl.zeta = 0.9;
cl.kp = 2; cl.kv = 3; cl.ka = 5; cl.rho = 1;
nuData = [t, 0.3 * (t < 1.0)];
[closedLoopPath, closedLoopIn] = buildNonlinearClosedLoopModel( ...
    outputDir, cl, nuData, wData, true);
closedLoopOut = sim(closedLoopIn);
```

该模型名为 `codesign_nonlinear_closed_loop`。其参考动态是

\[
\dot c=\nu,\quad\dot z_1=z_2,\quad
q=\dot z_2=-\omega^2(z_1-c)-2\zeta\omega z_2.
\]

每个原生 Fcn 块分别计算可检查的中间量：

\[
\begin{aligned}
e&=p-z_1,& y&=v-z_2,& b&=a-k\sin p-dv,\\
a_d&=k\sin p+dv+q-k_p e-k_v y,\\
\dot q&=-\omega^2(z_2-\nu)-2\zeta\omega q,\\
\dot a_{d,0}&=k\cos(p)v+(d-k_v)b+\dot q-k_p y+k_v q,\\
u^*&=a+\tau\bigl(\dot a_{d,0}-k_a(a-a_d)-\rho y\bigr).
\end{aligned}
\]

`Nominal desired actuator derivative` 使用名义加速度 `b`，不读取真实扰动 `w`，因此控制器没有扰动预知信息。控制器输出经过显式 Zero-Order Hold 和 Saturation，执行器再通过连续一阶动态作用到物理系统。采样间隙内参考状态继续积分，但实际控制输入保持不变。

高层命令斜率 `nuData` 可以来自 MATLAB MPC 运行日志。**这验证低层反馈闭环与真实 plant 的独立实现，并不表示 Simulink 内已经逐拍运行 MPC。** 更换扰动而保持 `nuData` 时，高层命令没有针对新扰动重新规划；不能将这种实验标为完整 MPC 闭环验证。

模型工作区参数为 `cl_k,cl_d,cl_tau,cl_U,cl_h,cl_x0,cl_tStop,cl_omega,cl_zeta,cl_kp,cl_kv,cl_ka,cl_rho`，输入为 `nuData,wData`。为兼容原生 Fcn 的标量参数语法而采用这些平铺变量，函数和返回的 `SimulationInput` 均由同一 `params` 结构写入。重新设定参数时，需要覆盖对应的 `cl_` 变量；改变步长或终止时间也同步设置 `FixedStep`、`StopTime`。

输出 `pLog,vLog,aLog,cLog,z1Log,z2Log,uRawLog,uSatLog,wLog,nuLog` 均为 `timeseries`。其中 `uRawLog` 是显式采样后的限幅前控制值，适于与 MATLAB 每步计算一次控制、整步保持的 RK4 实现逐点比较。

附加官方依据：[原生 Fcn 表达式及工作区标量参数](https://www.mathworks.com/help/simulink/slref/fcn.html)、[Zero-Order Hold](https://www.mathworks.com/help/simulink/slref/zeroorderhold.html)。
