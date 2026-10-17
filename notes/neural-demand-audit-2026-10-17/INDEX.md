# 现有神经策略、实际MPC需求和同信息比较

日期：2026-10-17。接续`6de8bc6`；不训练、不改权重、不另造策略族。

- [总报告与继续/停止裁决](../../docs/neural-demand-verification-2026-10-17.md)
- [检查点盘点](checkpoint-inventory.md)／[JSON](checkpoint-inventory.json)
- [原评估重放协议](replay-protocol.md)／[结果](replay-results.md)／[JSON](replay-results.json)／[源码](../../scripts/replay_existing_ppo_audit.py)
- [神经方向冻结协议](neural-protocol.md)／[结果](neural-results.md)／[JSON](neural-results.json)／[源码](../../scripts/audit_existing_neural_feedback.py)
- [完整物理映射核验协议](physical-map-check-protocol.md)／[JSON](physical-map-check.json)／[源码](../../scripts/check_neural_physical_map.py)
- [实际MPC需求与同信息比较](mpc-demand-audit.md)
- [理论门槛及独立审查](acceptance-theory.md)

结论：12个真实PPO网络均存在128锚点上的局部值保持反馈方向，但所选方向均不保持其余3968条件；固定design时原native MPC不依赖actor，直接反馈需求为0。同原预测问题的标准方法也看到0导数，完整鲁棒任务性能比较未运行。当前应停止闭环联合学习已经成立/优越的主张，先解决实际任务—实际策略—规划预测对象的接口。

本阶段实际重放49,152个仿真转移及13,578,240个预测微步；不是零仿真读取。训练/策略更新/checkpoint写入为0，原权重与源码未改。
