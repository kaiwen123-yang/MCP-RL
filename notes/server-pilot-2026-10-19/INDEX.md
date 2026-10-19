# 固定服务器实验与负结果证据

日期：2026-10-19。主结论见[科学结果报告](../../docs/server-pilot-results-2026-10-19.md)：codesign在三个训练seed下均劣于uniform，末轮成功0/24，成本高25.01%、分支耗时约高82.9%。完整固定预算已执行完毕，没有扩训、调参或重试。

| 文件 | 身份 |
|---|---|
| [report.md](report.md) | 全部逐seed/检查点结果与配对比较；失败轨迹保留 |
| [summary.json](summary.json) | 结构化任务、成本、原始粒度指标与运行账本 |
| [execution-summary.json](execution-summary.json) | 实际运行版本、资源限制、峰值内存、退出与范围 |
| [requirements-resolved.txt](requirements-resolved.txt) | 独立服务器环境实际依赖版本 |
| [mechanism/analysis.json](mechanism/analysis.json) | 36份既有评价快照的离线来源分析，无新环境采样 |
| [mechanism/across_training_seeds.csv](mechanism/across_training_seeds.csv) | 任务成本分解、预测误差与记录命令模型乐观差 |
| [mechanism/episode_metrics.csv](mechanism/episode_metrics.csv) | 每个评价条件的重算结果与原日志误差 |
| [mechanism/fixed_input_actor_comparisons.csv](mechanism/fixed_input_actor_comparisons.csv) | 共同记录观测上的动作映射差；非新物理rollout |

源码身份：算法基线38931af；实际执行89fa3c5，仅增加资源隔离和已有评价数据保存。配置字节/LF哈希为71a74d…6c32；机制分析还记录JSON规范化哈希913a2e…0a78，两者哈希序列化口径不同，配置内容已逐项一致核对。

原始结果包的SHA256为2139d9baea2de7faca21b6f4b59b29ba096972bd7b4f92910f0d521cfae65e56，服务器与本地一致。完整checkpoint、评价张量及运行日志保存在本地和服务器的忽略目录，不随公开提交上传。

聚合脚本：[任务汇总](../../scripts/summarize_task_response.py)、[已有评价来源分析](../../scripts/analyze_task_response_evidence.py)。所有来源分析只读已保存数据；新物理步和优化器更新均0。不把三个训练seed或24次固定条件评价写成总体统计证明。
