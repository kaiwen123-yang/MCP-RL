# 实际执行闭环进入MPC：本轮交付索引

日期：2026-10-17。接续`e23a28a`及`8488a1c`；保持已确认的长程多阶段共同问题、MPC规划/RL执行和双平台暂定范围。

- [总报告、接口与三层验收](../../docs/execution-aware-mpc-interface-2026-10-17.md)
- [运行前冻结协议](protocol.md)／[独立协议审查](comparison-protocol-review.md)
- [接口实现说明](interface-implementation.md)／[纯functional预测](../../src/response_ppo/closed_loop_prediction.py)／[独立MPC adapter](../../src/response_ppo/execution_aware_env.py)
- [identity及固定actor实际重规划](validation-results.md)／[JSON](validation-results.json)／[脚本](../../scripts/validate_closed_loop_interface.py)
- [同池与原任务导数](pool-and-gradient-results.md)／[JSON](pool-and-gradient-results.json)／[脚本](../../scripts/audit_closed_loop_mpc_demand.py)
- [全池独立shooting补充协议](independent-shooting-protocol.md)／[结果](independent-shooting-results.md)／[JSON](independent-shooting-results.json)／[脚本](../../scripts/independent_native_shooting_comparison.py)
- [任务costate推导](theory.md)／[独立三层裁决](independent-verdict.md)

已完成：一步和80步历史映射精确一致；12个实际任务梯度非零且与标准BPTT/伴随一致；768同池查询113个不同命令选支改变；全部候选独立shooting得到相同成本/约束与赢家；12组冻结actor实际task成本下降，但penalty存在tube/total代价恶化。

当前只支持接口修复及有条件的预测/规划效果。没有训练、改变checkpoint、学习预算优势、连续TO性能胜出或机器人任务证据。新增反事实执行49,152仿真转移，所有离线模型和梯度计算另计；不称零仿真。原始权重、完整轨迹与候选池留work，公开必要数值与源码，复算仍需本地原材料。
