# 冻结PPO试验的公开精简结果

生成日期：2026-10-16。三seed为17、29、43，四主组按事前协议完成；中点诊断另计。没有从开发seed选取正式结果。

复现汇总（不训练，不导入torch或Tianshou）：

```powershell
python scripts/summarize-ppo-mechanism.py
```

需要本地原始日志`work/ppo-mechanism-2026-10-16/formal/`与`midpoint/`。这些日志及权重未作为公开附件；JSON中的原始文件SHA256用于结果身份检查，不表示GitHub包含这些文件。公开训练源码、冻结配置和脚本可用于重新产生日志。

- `summary.json`：总体指标、逐seed比较、原始数据指纹、身份核验与总账本。
- `per-seed.csv`：24行（3seed×4方法×2长度），真实测试指标；计数求和、最大值取max。
- `paired-comparison.csv`：J_support对F_support配对差异及2×2差分交互。百分比正值表示J改善；交互负值表示设计更新在support下更有利，两者不是同一个指标。
- `learning-curves.csv`：480行全部训练批次，无平滑、无挑选；列出累计物理/候选模型步、PPO诊断和响应参数。
- `ledger.csv`：36行正式训练/短测/长测＋12行新增中点诊断。learned对照复用正式测试，没有重复计数。
- `config.json`：共同训练配置、解析系统、P矩阵、源文件指纹与动作映射解释。
- `midpoint-comparison.csv`：从中点原始batch重新汇总，与正式learned成本配对。
- `midpoint-summary.json`：先前生成的独立诊断记录，保留不覆盖；其中runner式均值字段不用于本版最大值/计数汇总。

聚合规则：cost与逐物理步rate按physical_steps加权；planner失败率按planner_instances加权；`max_*`取最大；事件和计算量求和。最终summary中的seed SD来自3个种子的成本均值，未声称显著性。原runner将所有数值字段平均的`evaluation.mean`不用于本版结果。

所有速率为0–1比例，只有名称含`percent`的列是百分数。`planner_low_level_model_steps`仅数规划候选预测，不包含全程序所有区间检查和参考执行调用。wall_seconds为并发CPU过程内观测时间，不能直接当无争用串行benchmark或GPU/NX性能。

长测持续20.48秒且参考不重置，但短测不是其同一随机任务前缀。该对象是已知离散输入仿射系统，支持由解析模型决定且不依赖θ；不代表机器人长程任务、完整SPlaTES复现或独立新算法贡献已经成立。
