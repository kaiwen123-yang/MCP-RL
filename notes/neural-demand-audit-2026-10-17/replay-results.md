# 已有PPO短测的确定性重放结果

日期：2026-10-17（Asia/Shanghai）。[冻结协议](replay-protocol.md)、[逐字段证据](replay-results.json)、[脚本](../../scripts/replay_existing_ppo_audit.py)。当前状态：`completed`；已记录12/12组，完全一致通过12组。全部完成且通过：`True`。

**这是本轮重新执行原有固定评估的仿真，不是零仿真读取，也不是旧随机训练batch恢复。**原PPO网络、design、seed、16env、256step、原环境及原MPC全部保留；逐step核对旧NPZ的全部15项metric，首次不一致即停止该case。没有更换模型、调参数或根据结果选组。

| seed | arm | 状态 | 已执行step/env | 15metric最大差 | 仿真墙钟秒 |
|---:|---|---|---:|---:|---:|
| 17 | joint_support | exact_replay_pass | 256 | 0 | 1.557 |
| 17 | fixed_support | exact_replay_pass | 256 | 0 | 1.466 |
| 17 | joint_penalty | exact_replay_pass | 256 | 0 | 1.431 |
| 17 | fixed_penalty | exact_replay_pass | 256 | 0 | 1.430 |
| 29 | joint_support | exact_replay_pass | 256 | 0 | 1.437 |
| 29 | fixed_support | exact_replay_pass | 256 | 0 | 1.452 |
| 29 | joint_penalty | exact_replay_pass | 256 | 0 | 1.455 |
| 29 | fixed_penalty | exact_replay_pass | 256 | 0 | 1.527 |
| 43 | joint_support | exact_replay_pass | 256 | 0 | 1.468 |
| 43 | fixed_support | exact_replay_pass | 256 | 0 | 1.441 |
| 43 | joint_penalty | exact_replay_pass | 256 | 0 | 1.461 |
| 43 | fixed_penalty | exact_replay_pass | 256 | 0 | 1.458 |

完整重建观测保存在`work/neural-demand-audit-2026-10-17/replays/{seed}-{arm}.npz`。新增字段包含step前obs/x/z/command/raw_design、支持区间及discriminant、znext/uref和网络mean；原15metric保持原字段名。末态terminal_x/z可与step前状态移位组合得到next序列。每组实际shape、dtype、字节hash和规范化hash分别列于JSON。

本次执行时源文件字节hash与正式manifest完全匹配。脚本额外支持仅LF/CRLF换行不同的同一源文本，并分别记录字节匹配与内容等价，不允许其它源码改动。所有已经执行的checkpoint在前后字节一致，actor张量未变；仅实例化GaussianActor，不创建PPOBackend、optimizer或critic。物理状态由原离散servo重新推进，原MPC在线规划，所以实际仿真与预测成本必须记账。

本次累计实际预算：仿真转移`49152`，planner实例`4992`，候选序列`169728`，预测微步`13578240`；训练更新、actor权重更新、梯度优化器步及checkpoint写入均为0。结果中原始观察缺失的事实不变：本轮新记录不能被称作原本已保存的随机PPO数据或真实训练优势。
