# 物理任务、有限反馈与可计算联合更新

日期：2026-10-17。接续`b5d5697`评审和`703bdf7`交接；第一阶段已推送`b10ee31`，整轮保持MPC规划/RL执行，不训练、不操作实机，不自动跟踪其它对话。

- [整轮主报告](../../docs/physical-task-coupling-instance-2026-10-17.md)
- [运行前冻结范围](protocol.md)
- [狗和臂的完整物理任务候选](robot-task-contracts.md)
- [确定性部署语义、Gaussian支持定理及因果反馈推导](risk-and-theory.md)
- [解析核验与同信息TO等价](analytic-verification.md)／[JSON](analytic-results.json)／[核算脚本](../../scripts/verify_physical_task_instance.py)
- [有限计算算法、伪代码、标签与总预算](computable-algorithm.md)
- [信用算术模块](../../src/control_task_credit.py)／[缓存反证脚本](../../scripts/verify_task_credit_packets.py)／[结果](credit-packet-results.json)
- [独立核查与修订边界](review-notes.md)

实际运行仅为明确归一化物理实例的闭式数学核验、保存结果的信用运算及预算整数复算。PPO/QP/model fitting与29,184执行transition等均属未来设计预算，本轮没有执行。不是训练后机器人策略、硬件成功率或计算效率胜出。

结论：有限反馈能在这个任务中用更多净空换取更短完整时间；同类标准TO完全同解，故不支持该例的独立算法优势。正确局部信用跨有限变化会显著乐观，必须重评后段。真实两平台的任务/执行数据、当前策略模型与有限预算学习收益仍缺证据。
