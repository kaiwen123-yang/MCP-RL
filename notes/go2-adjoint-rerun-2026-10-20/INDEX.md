# Go2修复后固定三组比较证据

状态：COMPLETED，比较有效，当前伴随候选无稳定增益。结论与解释见[完整报告](../../docs/go2-adjoint-rerun-results-2026-10-20.md)。

- [机器可读决策与合计成本](decision.json)，[进程退出与隔离环境](execution-summary.json)
- [末段接近与停稳的保存轨迹观察](terminal-phase-observations.json)
- [完整分析摘要](analysis/summary.json)，[输入文件与SHA](analysis/analysis_manifest.json)
- [固定末轮逐条件结果](analysis/final_evaluation.csv)，[逐组汇总](analysis/final_by_arm.csv)，[伴随与普通配对](analysis/paired_final_adjoint_vs_ordinary.csv)
- [全部评价](analysis/evaluation_rounds.csv)，[全部轮次](analysis/round_summary.csv)，[全部采集episode](analysis/all_collection_episodes.csv)，[按阶段汇总](analysis/all_collection_episodes_by_phase_arm.csv)
- [真实actor更新与KL停止](analysis/actor_update_history.csv)
- [逐阶段成本](analysis/cost_ledger.csv)，[分支成本](analysis/branch_costs.csv)，[公共准备](analysis/shared_common_costs.csv)
- [初始化独立核验](initialization-audit.json)，[九分支启动表](analysis/initialization_startups.csv)
- [同服务器原跟随器](follower-reference-summary.json)
- [按交互的曲线](figures/go2_task_vs_physics.png)，[按时间的曲线](figures/go2_task_vs_wall.png)，[图数据与视觉核验](figures/plot_manifest.json)
- [有限运行包装器](launch_fixed.py)，[参照与训练作业入口](run_job.py)
- [旧失效批次](../../docs/go2-adjoint-server-results-2026-10-20.md)，[输出异常及返工成本](../go2-adjoint-server-2026-10-20/execution-incident.md)

算法源固定为`c72ce059eeecff376bea0649523f40b78c682a87`，配置未改变。原始检查点、轨迹、进程与环境快照保存在忽略的`work/server-adjoint-20261020-r2/preserved/`，不公开私人状态。

仅复算JSON结果，不训练、不导入物理仿真、不加载pickle：

```powershell
python scripts/analyze_go2_adjoint_pilot.py --runs work/server-adjoint-20261020-r2/preserved/results --execution-dir work/server-adjoint-20261020-r2/preserved --output notes/go2-adjoint-rerun-2026-10-20/analysis
python scripts/plot_go2_adjoint_pilot.py --analysis notes/go2-adjoint-rerun-2026-10-20/analysis --output notes/go2-adjoint-rerun-2026-10-20/figures --reference notes/go2-adjoint-rerun-2026-10-20/follower-reference-summary.json
```

报告中的主结果固定为第四轮，不能以中间最好检查点替换。原跟随器为三个固定条件各一次，三个训练seed也不是新场景泛化样本。公共准备在项目总量只计一次，在“独立使用某方法”口径各分配一次，两种成本不能混加。
