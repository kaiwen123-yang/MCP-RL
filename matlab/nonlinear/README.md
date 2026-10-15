# 非线性共同设计解析原型

本目录提供MATLAB R2025b/Simulink原型，验证执行器动态、输入饱和、有界扰动、数字低层反馈及可实现参考规划。当前控制器为经典backstepping解析参照，没有RL训练。

在本目录作为MATLAB当前目录时：

```matlab
projectRoot = fileparts(fileparts(pwd));
outputDir = fullfile(projectRoot, "work", "nonlinear-prototype-2026-10-15");
report = runNonlinearPrototype(outputDir, false, false);
evidence = summarizeNonlinearEvidence(outputDir);
```

519组声明参数候选会按同一个证书和两个任务评价。`report.matchedClassAdvantage`比较同一评价表的共同最小化与嵌套固定响应最小化；其值应为0，不能当成两个独立优化器的性能实验。经典局部SQP可以复用已完成搜索：

```matlab
report = runNonlinearPrototype(outputDir, true, true);
```

SQP有明确70次评价预算，可能未收敛；结果单独记录，不替换已经完成全部核验的网格设计。全网格最优不是连续参数全局最优。

增益LMI的有理数复核，从仓库根目录运行：

```text
python matlab/nonlinear/verify_gain_certificates.py work/nonlinear-prototype-2026-10-15/evidence-summary.json work/nonlinear-prototype-2026-10-15/exact-gain-verification.json
```

原生Simulink模型由构建函数生成，低层真实反馈在Simulink运行，高层命令由对应MATLAB MPC日志提供。实际高层QP只使用内部参考状态，不能称为已实现实际扰动反馈下的机器人高层规划。详见[模型说明](README-simulink.md)及[完整研究报告](../../docs/nonlinear-prototype-report-2026-10-15.md)。

输出目录含JSON、MAT、SLX、PNG和FIG。已有模型会核对确切路径再复用；不会清空用户工作区、关闭所有图窗、运行外部训练或修改MATLAB系统设置。
