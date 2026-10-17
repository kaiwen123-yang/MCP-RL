# 全候选独立原生射击法复算结果

日期：2026-10-17。状态：`completed`；全部通过：`True`。
[冻结补充协议](independent-shooting-protocol.md)、[逐项JSON](independent-shooting-results.json)、[独立脚本](../../scripts/independent_native_shooting_comparison.py)。

只用原ResponseServo基础方法重新积分保存的全部候选，不调用新closed_loop_prediction步进或预测函数；不调用env.step、MPC或训练器。目标、命令、原生支持、冻结mean actor、float64与w=0/supportW=.04均保持。原checkpoint和生产代码前后hash保留在JSON。

| seed | arm | actual成本最大绝对差 | 四个池均通过 | 墙钟秒 |
|---:|---|---:|---|---:|
| 17 | joint_support | 6.938893903907228e-17 | True | 0.687 |
| 17 | fixed_support | 6.938893903907228e-17 | True | 0.673 |
| 17 | joint_penalty | 1.1102230246251565e-16 | True | 0.690 |
| 17 | fixed_penalty | 8.326672684688674e-17 | True | 0.675 |
| 29 | joint_support | 8.326672684688674e-17 | True | 0.682 |
| 29 | fixed_support | 6.938893903907228e-17 | True | 0.685 |
| 29 | joint_penalty | 5.551115123125783e-17 | True | 0.680 |
| 29 | fixed_penalty | 6.938893903907228e-17 | True | 0.674 |
| 43 | joint_support | 8.326672684688674e-17 | True | 0.661 |
| 43 | fixed_support | 6.938893903907228e-17 | True | 0.675 |
| 43 | joint_penalty | 6.938893903907228e-17 | True | 0.678 |
| 43 | fixed_penalty | 8.326672684688674e-17 | True | 0.672 |

已完成池文件：`48`；query：`768`；actual不同命令赢家一致：`768`；reference赢家一致：`768`。原reference掩码全同：`True`；次级共同actual/tube/input掩码全同：`True`。次级池无可行候选仍为`140`，不证明连续问题不可行。

逐项数值最大差：

```json
{
  "actual_cost": 1.1102230246251565e-16,
  "reference_cost": 9.71445146547012e-17,
  "actual_domain_margin_max": 7.771561172376096e-16,
  "reference_domain_margin_max": 5.551115123125783e-16,
  "input_margin_max": 8.881784197001252e-16,
  "tube_margin_max": 1.2576745200831851e-17,
  "actual_one_step_excess_max": 3.469446951953614e-18,
  "support_worst_case_excess_max": 2.710505431213761e-18
}
```

预算：

```json
{
  "complete_candidate_trajectories": 26112,
  "offline_actual_model_transitions": 2088960,
  "actor_observations": 2088960,
  "original_observation_instances": 2088960,
  "original_interval_instances_including_observation": 4177920,
  "original_reference_response_instances": 4177920,
  "original_fallback_instances": 2088960,
  "environment_step_calls": 0,
  "mpc_calls": 0,
  "training_updates": 0,
  "optimizer_steps": 0,
  "checkpoint_writes": 0
}
```

总墙钟`8.169`秒。旧observation内部interval加手动interval重复计算几何，已计入调用数；不据此声称与新接口同墙钟预算或速度优劣。

这是在先前同数组argmin检查之后**追加**的完整独立候选积分证据。通过只支持保存的同目标成本、约束和固定池选支一致；不是新连续TO优化器、全局最优证明、RL学习或性能优势。

哈希口径：JSON原有`protocol_sha256`、`source_results_sha256`、`script_sha256`保留实际执行输入的字节SHA-256；另补`*_content_sha256`作为CRLF→LF规范化后的内容哈希。补元数据前核实三份文件字节均未变化；没有重新运行模型积分或修改已执行脚本。checkpoint及候选数组仍按二进制字节哈希核身份。
