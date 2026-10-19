# Go2事实紧凑预测：模型拟合与连续任务结果

证据状态：`COMPLETE_TWELVE_CASE_EVIDENCE`。已保存case数：12；预定12个；缺失：[]。

同一已知scene和三种已知条件的开发比较。模型拟合与冻结策略更新分别计账；本报告不声称新任务泛化。

## 模型拟合与策略冻结

拟合状态：`COMPLETED_CONFIGURED_MODEL_FIT`；模型优化步：1600；策略优化步：0；拟合墙钟：3.40638 s。
训练转移：900；验证转移：300；训练/验证episode哈希交集：[]。

| 模型 | 模型更新 | 策略更新 | 拟合wall s | 验证窗口 |
|---|---:|---:|---:|---:|
| id_one_step | 800 | 0 | 1.19418 | 300 |
| compact_multistep | 800 | 0 | 1.69375 | 300 |

| 模型 | 参数量 | checkpoint字节 | 初值与共同hash一致 | 末参数相对初值改变 |
|---|---:|---:|---|---|
| id_one_step | 26534 | 115338 | True | True |
| compact_multistep | 26534 | 115592 | True | True |

上述参数身份来自拟合记录的逐tensor哈希；评价加载的checkpoint哈希另外核对。

| 模型 | 验证口径 | 有效标签 | 位置RMSE m | yaw RMSE rad | body速度RMSE m/s | yaw-rate RMSE rad/s |
|---|---|---:|---:|---:|---:|---:|
| id_one_step | one_step | 1480 | 0.00791528 | 0.0146052 | 0.0476518 | 0.108438 |
| id_one_step | autoregressive_h5 | 1480 | 0.0300931 | 0.0546635 | 0.0668999 | 0.169417 |
| compact_multistep | one_step | 1480 | 0.00909999 | 0.0181216 | 0.0447407 | 0.107493 |
| compact_multistep | autoregressive_h5 | 1480 | 0.0181698 | 0.0351655 | 0.0491414 | 0.128265 |

验证集单步与五步递推误差、逐时域误差、模型hash和实际前反向计数保存在summary.json。单步和多步模型更新次数相同不代表计算量相同。

验证速度口径是body平面速度；下面在线selected-command误差是world平面速度，不能直接相减。

## 完整任务与200ms周期

| 方法 | 条件 | 实际成功 | 原因 | 任务秒 | 成本 | 位置RMSE m | 速度RMSE m/s | 总周期p95 s | 总周期max s | 超期数/宏步 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| id_one_step | nominal | True | success | 10 | 14.4861 | 0.0103054 | 0.0474167 | 0.0184361 | 0.0189115 | 0/50 |
| id_one_step | push_positive_y | True | success | 11 | 14.7827 | 0.0141462 | 0.0661462 | 0.0187592 | 0.0196089 | 0/55 |
| id_one_step | push_negative_y | True | success | 9.2 | 14.7665 | 0.012725 | 0.0583211 | 0.0186785 | 0.0191094 | 0/46 |
| compact_multistep | nominal | True | success | 9.6 | 15.2333 | 0.0115781 | 0.0486602 | 0.0188714 | 0.0189877 | 0/48 |
| compact_multistep | push_positive_y | True | success | 8.6 | 15.2013 | 0.0104416 | 0.0465754 | 0.0187543 | 0.0191726 | 0/43 |
| compact_multistep | push_negative_y | True | success | 8.8 | 15.2961 | 0.0108036 | 0.0524859 | 0.0187421 | 0.018958 | 0/44 |
| kinematic_mpc | nominal | True | success | 8.4 | 14.722 | 0.0317337 | 0.133831 | 0.0112943 | 0.0116113 | 0/42 |
| kinematic_mpc | push_positive_y | True | success | 8.6 | 14.7081 | 0.030362 | 0.129834 | 0.0113231 | 0.0116139 | 0/43 |
| kinematic_mpc | push_negative_y | True | success | 8.6 | 15.0577 | 0.0308257 | 0.126755 | 0.011311 | 0.0117145 | 0/43 |
| policy_rollout_mpc | nominal | True | success | 9 | 14.4156 | 0 | 0 | 0.63048 | 0.641601 | 45/45 |
| policy_rollout_mpc | push_positive_y | True | success | 9.6 | 14.5703 | 0.00246244 | 0.0194891 | 0.63722 | 0.659027 | 48/48 |
| policy_rollout_mpc | push_negative_y | True | success | 9 | 14.6429 | 0.00247775 | 0.0183309 | 0.65464 | 0.659286 | 45/45 |

只有所有已观察周期都≤0.2 s，才能说这次观测未超期；p95较低但最大值超期仍需报告。这不是硬实时或NX测量。

## 真实资源与终点

| 方法 | 条件 | ∫τ²dt | ∫\|τdq\|dt | 终点位置误差m | 终点yaw误差rad | 终点速度m/s | 严格保持秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| id_one_step | nominal | 4377.71 | 433.11 | 0.0169259 | 0.00578533 | 0.0410966 | 1.18 |
| id_one_step | push_positive_y | 4361.04 | 461.422 | 0.0235065 | 0.114449 | 0.00212822 | 1.16 |
| id_one_step | push_negative_y | 3874.85 | 407.864 | 0.0122248 | 0.0491672 | 0.0777772 | 1.18 |
| compact_multistep | nominal | 4008.03 | 431.376 | 0.0027402 | 0.0257602 | 0.0208442 | 1.18 |
| compact_multistep | push_positive_y | 3659.12 | 407.058 | 0.00600149 | 0.0339842 | 0.00379084 | 1.06 |
| compact_multistep | push_negative_y | 3664.47 | 408.485 | 0.00807799 | 0.0115701 | 0.012122 | 1 |
| kinematic_mpc | nominal | 3824.24 | 420.938 | 0.0194199 | 0.0958834 | 0.109981 | 1 |
| kinematic_mpc | push_positive_y | 3757.64 | 419.872 | 0.0165595 | 0.0285192 | 0.00921397 | 1.18 |
| kinematic_mpc | push_negative_y | 3725.96 | 413.421 | 0.00563518 | 0.0524824 | 0.0368355 | 1.08 |
| policy_rollout_mpc | nominal | 4086.1 | 409.155 | 0.00160927 | 0.0590299 | 0.000789297 | 1.18 |
| policy_rollout_mpc | push_positive_y | 4263.94 | 468.208 | 0.00290921 | 0.0262266 | 0.0125243 | 1.08 |
| policy_rollout_mpc | push_negative_y | 3977.45 | 418.4 | 0.00294741 | 0.0251065 | 0.000418782 | 1.12 |

## compact_multistep相对对照

| 对照 | 条件 | 双方成功 | 成本变化% | 时间变化% | p95周期变化% | 最大周期变化% |
|---|---|---|---:|---:|---:|---:|
| id_one_step | nominal | True | 5.15811 | -4 | 2.36127 | 0.403157 |
| kinematic_mpc | nominal | True | 3.4729 | 14.2857 | 67.0888 | 63.5282 |
| policy_rollout_mpc | nominal | True | 5.67273 | 6.66667 | -97.0068 | -97.0406 |
| id_one_step | push_positive_y | True | 2.83209 | -21.8182 | -0.0258551 | -2.22482 |
| kinematic_mpc | push_positive_y | True | 3.3532 | 0 | 65.6292 | 65.0831 |
| policy_rollout_mpc | push_positive_y | True | 4.33097 | -10.4167 | -97.0569 | -97.0908 |
| id_one_step | push_negative_y | True | 3.58612 | -4.34783 | 0.340531 | -0.792187 |
| kinematic_mpc | push_negative_y | True | 1.58267 | 2.32558 | 65.6983 | 61.8337 |
| policy_rollout_mpc | push_negative_y | True | 4.46057 | -2.22222 | -97.137 | -97.1245 |

早停失败或算力截断的累计成本仅描述已发生代价，不与成功完整任务排序。

## 评价调用计数

| 方法 | 条件 | 实际物理步 | 实际policy前向 | 预测物理步 | 预测policy前向 | compact encoder调用 | compact response步 |
|---|---|---:|---:|---:|---:|---:|---:|
| id_one_step | nominal | 2000 | 500 | 0 | 0 | 50 | 6000 |
| id_one_step | push_positive_y | 2200 | 550 | 0 | 0 | 55 | 6600 |
| id_one_step | push_negative_y | 1840 | 460 | 0 | 0 | 46 | 5520 |
| compact_multistep | nominal | 1920 | 480 | 0 | 0 | 48 | 5760 |
| compact_multistep | push_positive_y | 1720 | 430 | 0 | 0 | 43 | 5160 |
| compact_multistep | push_negative_y | 1760 | 440 | 0 | 0 | 44 | 5280 |
| kinematic_mpc | nominal | 1680 | 420 | 0 | 0 | 0 | 0 |
| kinematic_mpc | push_positive_y | 1720 | 430 | 0 | 0 | 0 | 0 |
| kinematic_mpc | push_negative_y | 1720 | 430 | 0 | 0 | 0 | 0 |
| policy_rollout_mpc | nominal | 1800 | 450 | 216000 | 54000 | 0 | 0 |
| policy_rollout_mpc | push_positive_y | 1920 | 480 | 230400 | 57600 | 0 | 0 |
| policy_rollout_mpc | push_negative_y | 1800 | 450 | 216000 | 54000 | 0 | 0 |

拟合产生的模型更新与这里的冻结预测调用分开；计数不是等价FLOPs。

## 本轮完整计算账

| 项目 | 实际记录/按固定启动规则换算 |
|---|---:|
| collection_episodes | 8 |
| collection_macro_transitions | 1200 |
| collection_physics_steps_including_settle | 51200 |
| collection_policy_calls_including_settle | 12800 |
| evaluation_completed_case_count | 12 |
| evaluation_task_physics_steps | 22080 |
| evaluation_task_policy_calls | 5520 |
| evaluation_reset_settling_physics_steps | 4800 |
| evaluation_reset_settling_policy_calls | 1200 |
| evaluation_prediction_physics_steps | 662400 |
| evaluation_prediction_policy_calls | 165600 |
| real_execution_physics_steps_including_all_settle | 78080 |
| all_physics_steps_including_prediction | 740480 |
| all_policy_calls_including_prediction | 185120 |
| compact_response_steps_during_evaluation | 34320 |
| compact_encoder_calls_during_evaluation | 286 |
| model_optimizer_steps | 1600 |
| policy_optimizer_steps | 0 |
| collection_wall_seconds | 6.43045 |
| fit_wall_seconds | 3.40638 |
| evaluation_wall_seconds | 93.8348 |
| sum_recorded_stage_wall_seconds | 103.672 |

与上一轮最终基线核对：6条case；任务和物理输出在1e-8内一致：True。旧耗时不复用；本轮耗时独立报告。

数据采集的每episode计数已包含其启动稳定；任务评价另计12次启动，避免漏算或重复相加。

## 证据边界

- 同一已知scene、route及三种已知条件的开发比较，不是未见任务、平台或地形泛化。
- 本轮允许拟合事实响应模型；冻结策略参数更新必须单独计为零，不能把模型拟合叫策略学习。
- 单步和多步模型同更新次数不等于同前反向计算或同墙钟；分别报告完整账本。
- 在线预测误差对应被选命令的下一宏步；不是全部候选的误差，也不是独立五步开环精度。
- 不同控制器的在线RMSE来自各自访问的状态/命令，不能直接当作同查询模型精度比较；共同验证窗口才提供匹配输入的精度对照。
- 未来外力不交给预测器，推扰宏步与普通宏步分别报告；预测更准不自动等于完整任务更好。
- 200ms判断使用每次完整控制周期的p95、最大值与超期比例；平均时间不能代替它们。
- 本机/服务器观察到的耗时不构成NX时延、实机控制频率或硬实时保证。
- 最终成功按真实MuJoCo轨迹、全身几何、接触及时间戳保持判断，不借用紧凑预测的未来接触或关节状态。
- 两个学习模型与kinematic使用同一航向矩形proxy；exact policy rollout仍拥有完整名义物理/几何，属于额外信息的特权对照。
- 学习模型共同零输出先验为当前yaw的Euler位移，旧kinematic使用midpoint yaw，两者不被混称为完全相同先验。
- 失败或算力截断全部保留；早停失败的小累计成本不用于宣称胜过成功任务。
- 只有少量确定性开发轨迹，不做统计显著性或总体鲁棒性结论。

数据一致性待核项：0。
