# 已有真实PPO确定性短测重放：冻结协议

日期：2026-10-17（Asia/Shanghai）。本协议先于首次重放保存。用户当前要求核实真实神经策略与MPC接口，允许对已有评估作固定确定性重放；仍不训练、不调参数、不改checkpoint。

## 1. 固定对象与执行顺序

按seed`17,29,43`及每seed的`joint_support,fixed_support,joint_penalty,fixed_penalty`顺序，执行全部12组原first-short评估，不依据结果删组。

- checkpoint：`work/ppo-mechanism-2026-10-16/formal/seedrun-{seed}/seed-{seed}/{arm}.pt`。
- 对照NPZ：同目录`{arm}-short-trace.npz`。
- 原条件：`eval_seed=seed*10000+5000`、16env、256step、checkpoint保存的最终design_mean。
- 原`EnvironmentConfig`及arm动作映射不变；float32、CPU、`torch.set_num_threads(1)`。
- 每组新建原`ResponseServo`，从原seed初始化，planner warm-start重新开始。不得从长测截前缀，因为初始化随机数消耗与steps有关。
- 仅创建原`GaussianActor`并严格载入原state_dict，取均值执行；不创建PPOBackend、optimizer或critic，不调用任何learn/update入口。

原环境本地RNG及原SPlaTES CEM bridge的每次规划seed保留。所有state/action/模型计算均走原函数，不另造近似环境或新策略族。

## 2. 身份、哈希与读取

checkpoint使用`torch.load(...,weights_only=True,map_location="cpu")`，NPZ使用`allow_pickle=False`。记录原checkpoint/NPZ及相关源码的**当前字节SHA256**与**CRLF→LF规范化SHA256**，二者分列，不把换行变化隐藏为同一字节文件。源码当前字节hash与原manifest相符才直接继续；不符时报告并停止，不调整数值实现来追旧结果。

每组执行前后检查checkpoint字节hash不变，actor张量值不变。原始checkpoint、NPZ及源码不写入。结果只写本轮独占目录。

## 3. 每一步比较与新增记录

在每次step前额外记录：

```text
step_index, obs, x, z, command, raw_design,
interval_lo, interval_hi, interval_feasible, interval_discriminant,
znext, uref, mean
```

并记录原15项step metric：`task,total,V,excess,empty,action,reference_input,support_width,position,reference_position,actual_domain_violation,reference_domain_violation,tube_violation,target,geometric_bound`。保存终止时的`terminal_x,terminal_z`，必要时可由移位重建next状态，避免每步双存。

每step完成后立即与旧NPZ对应时间、全部16环境、全部15metric逐字段比较。通过标准是同dtype/shape及**数值逐元素完全相同**，不使用可放宽的误差阈值；同时报告字节一致性和最大绝对差。任何不一致立即停止该case，保存已执行前缀和差异，标为失败；不搜索seed、不调浮点精度或参数来掩盖差异。其它已冻结case可独立继续。

新完整观测文件是**本轮重新仿真并核对旧评估后获得的重建记录**，不是从旧NPZ直接读出的原观测，也不是旧随机训练batch。没有旧GAE/logprob/随机潜动作，因此仍不能重放真实历史PPO训练梯度。

## 4. 输出与资源账本

- 脚本：`scripts/replay_existing_ppo_audit.py`。
- 完整本地记录：`work/neural-demand-audit-2026-10-17/replays/{seed}-{arm}.npz`。
- 公开结果：本目录`replay-results.json/replay-results.md`，保留原路径/hash、字段、比较结果及实际计数。

原日志的每组预算为4096仿真转移、416planner实例、14144条候选序列和1131520预测微步。如果12组均完整重放，预期合计49152仿真转移、4992planner实例和13578240预测微步。最终按实际发生计数，失败case按已执行前缀计入；不将重放记作0仿真。

新增训练更新、optimizer.step、checkpoint修改、部署和调参均为0。该阶段只恢复可核对的真实网络原生评估观测，不开始神经Jacobian/SVD或新的反馈验证。
