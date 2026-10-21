# 冻结Go2紧凑事实模型证据

2026-10-19。状态：**固定数据采集、模型拟合、四组12条完整任务全部完成；策略更新0。** 两拟合组3/3成功，全部286个宏步低于200ms；多步预测误差改善没有形成全面任务代价优势。

- [主报告与科学判断](../../docs/go2-compact-response-2026-10-19.md)。
- [逐工况报告](report.md)、[CSV](cases.csv)、[结构化汇总](summary.json)、[逐段误差/停稳/耗时](mechanism.json)。
- [比较图PNG](compression-task-tradeoff.png)、[SVG](compression-task-tradeoff.svg)、[绘图核验](figure-check.json)。机器核对无缺字/越界，人工读图修正了底部说明与轴标签重叠。
- [两份原创响应检查点](../../artifacts/go2-compact-2026-10-19/README.md)，原始HIM策略和原始物理资产未公开复制。

采集代码commit `7ce75f114b44d175fa25b4a722a8bb40a918b23e`；拟合/评价代码commit `1da0c86`，执行源文件hash存fit-summary与本汇总。任务配置没有修改。

固定策略SHA256始终为`d9945f85a39944aaf241a67f7d2b25a15326a0222d0e503aeb05a7f188430509`。两个模型初始tensor hash同为`517f9338ec3e82a3a6aafc4b36bf90b4dbf0a42239d3b3d008c1f247f5f48952`，batch索引hash同为`620a1ff0392b0fafca1cb1a2a12655708371817506c93146e670ec31c1f8fcc1`。两组各800次更新，未按验证或任务表现选择检查点。

原始8份NPZ、训练日志/检查点、12条完整macro/trace、stdout、源代码archive、私有服务器快照保存在忽略目录`work/go2-compact-20261019/downloaded/`。完整压缩包`work/go2-compact-20261019/results.tar.gz`的SHA256为`aac845766a83351d3f51f58a2caa9324bac6f5defa10d81495a8c55570cea533`。不公开私有进程命令行。

```bash
python scripts/analyze_go2_compact.py --input work/go2-compact-20261019/downloaded --output notes/go2-compact-2026-10-19
python scripts/plot_go2_compact.py
```

离线分析不加载模型、不调用MuJoCo、不新增采样。完成判据用真实全身接触/几何与时间戳保持重算，不用预测器自评。已有三工况是开发任务，验证集为两条新命令序列，不称新平台/地形泛化。普通辨识已经能压缩计算，当前不能宣称共同学习必要或多步模型全面更优。
