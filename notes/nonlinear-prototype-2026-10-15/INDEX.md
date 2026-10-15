# 非线性原型、控制理论结合与联合设计算法

日期：2026-10-15。围绕用户已确认的科学问题推进：共同设计对任务有用、实际能够实现、适合在线优化的闭环动力学与反馈策略，采用MPC规划、RL执行。当前阶段为理论、低维非线性解析与MATLAB/Simulink核验；不开展大规模RL训练。

## 研究文件

- [本轮研究与原型报告](../../docs/nonlinear-prototype-report-2026-10-15.md)：汇总实际执行结果、比较条件、失败边界及后续SPlaTES改造。运行状态与数值以该报告及对应结果文件为准；本索引不把源码存在当仿真成功。
- [控制理论谱系与取舍](control-theory-map.md)：11类主要谱系，选择动态扩展、增量ISS/多速率实现关系、成对响应共同设计三个结合；作者结果、明确条件与阅读者判断分开。
- [非线性可实现性与理论推导](nonlinear-proof.md)：含执行器动态、输入限幅、有界扰动的对象；不微分未知扰动的实现控制器；实现域、连续时间证书、离散保持与接续边界，以及共同设计退化为重参数化时的判断。
- [来源与阅读范围登记](../../literature/supplements/control-theory-connections-2026-10-15.json)：检索来源/理论谱系台账，不加入主100篇、不分配MR。

## MATLAB与Simulink入口

- [runNonlinearPrototype.m](../../matlab/nonlinear/runNonlinearPrototype.m)：本轮MATLAB原型与比较入口。应在MATLAB项目目录下运行；具体实测记录见本轮报告。
- [buildNonlinearPlantModel.m](../../matlab/nonlinear/buildNonlinearPlantModel.m)：原生Simulink物理对象，含非线性、输入限幅、执行器滞后与扰动；用于独立重放控制输入、交叉核对积分实现。
- [buildNonlinearClosedLoopModel.m](../../matlab/nonlinear/buildNonlinearClosedLoopModel.m)：真实低层采样反馈闭环，实际状态反馈形成控制命令；高层命令可使用外部日志，不能因此称Simulink中已运行完整MPC。
- [Simulink模型说明](../../matlab/nonlinear/README-simulink.md)：模型、变量接口、采样保持约定、工作区隔离与运行边界。
- [性能与增益分析](../../matlab/nonlinear/summarizeNonlinearEvidence.m)、[有理数增益复核](../../matlab/nonlinear/verify_gain_certificates.py)：耗散界、独立敏感性检查和六个有界实LMI的精确验证。
- [实际结果JSON](results.json)、[性能分析JSON](evidence.json)、[精确增益核验JSON](exact-gain-verification.json)：仅原创研究数据，不含本机私人路径。

## 计数与证据层级

控制理论报告列出15个编号参考项，对应**16篇来源记录**：参考项5包含RG教程与不精确优化command governor两篇，台账中分别为CT05、CT06。其中CT01、CT02分别复用已有CD01、CD02；其余14条是本次检索登记来源，**不是14篇新全文精读**。

本轮新增全文精读计数为**0**。其余阅读明确分为方法/定理局部读取、导论筛选或摘要筛选。既有CD01/CD02全文记录保留在[前轮台账](../../literature/supplements/codesign-decision-2026-10-15.json)，不重复计数、不把预印本更新日期当方法首次提出时间。2026新预印本没有被标为正式发表。

论文内容读取、独立数学推导、数值优化、Simulink一致性验证和机器人性能是不同证据层次。连续时间证书不能自动覆盖任意控制采样率；物理输入日志重放一致也不等于完整MPC闭环已验证。所有比较需保留相同物理目标、信息、执行频率、扰动可见时间与控制自由度。
