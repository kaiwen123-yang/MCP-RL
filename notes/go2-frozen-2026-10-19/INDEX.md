# 两套 Go2 冻结基准的独立证据

本目录包含两项独立实验。robot_lab 与 HIM 使用不同策略、机器人模型、时钟和任务，结果不可拼接。历史原文中的“未合并”仅描述当时状态。

当前入口：robot_lab 使用 `src/go2_robot_lab` 和 `scripts/run_go2_robot_lab.py`；HIM 保留 `src/go2_frozen` 和 `scripts/run_go2_frozen.py`。历史记录的源码路径、提交和哈希保持原样。

## robot_lab：45维观测，1ms物理步

# Go2 冻结策略完整任务证据

日期：2026-10-19。仅实验分支，未合并主线。主报告：[完整任务与普通 MPC 基线](../../docs/go2-frozen-baseline-2026-10-19.md)。

- [预声明协议](PROTOCOL.md)：三场景、命令与成功条件、名义 MPC 的能力边界。
- [来源与文件哈希](assets.json)、[运行依赖](environment.txt)、[评价源码哈希](evaluation-source.json)。
- [执行账本](execution-ledger.json)：规定评价、适配过程与开发试跑分列；训练/硬件操作均 0。
- [适配与开发记录](preflight/)：首个 locomotion 适配与首次 nominal MPC 完整试跑；没有隐藏失败重试。
- [六条规定评价](runs/)：mpc/follower × nominal/narrow/wide；每条保留摘要、10 Hz 实际轨迹、决策序列、模型回放及阶段出口状态。
- [分析、表格与图](analysis/README.md)：逐条件结果与跨阶段响应；不做样本不足的显著性推断。
- [源码与证据交叉复核](independent-review.md)：包括冻结身份、阶段顺序、物理子步碰撞与平面停止口径。
- [MPC完整视频](media/mpc_nominal.mp4)、[参考跟随器完整视频](media/follower_nominal.mp4)：两条nominal实际评价回放及来源哈希。

第三方权重、模型副本、venv、完整 50 Hz 原始轨迹与其他视频在忽略目录，未上传 GitHub。所有公开数据与两条展示视频均为本次生成的合成机器人仿真证据，不是用户实机记录。


## HIM：六帧历史，独立任务与模型

# Go2冻结整机任务：证据索引

日期：2026-10-19。状态：**完成服务器冻结仿真与连续任务；没有训练、实机、自动检测。**

- 科学结论与完整协议：[主报告](../../docs/go2-frozen-run-2026-10-19.md)。
- 最终九案例：[report.md](report.md)、[cases.csv](cases.csv)、[summary.json](summary.json)。
- 同批预测误差、推扰/非推扰分段、完整过门/停稳、计算账：[mechanism.json](mechanism.json)。
- 原始九案例保留：[original-summary.json](original-summary.json)、[original-cases.csv](original-cases.csv)、[original-mechanism.json](original-mechanism.json)。原版计时只支持6条严格停稳，另3条合格样本跨度0.98秒；最终采用6条原结果＋3条时间戳修复结果。
- 图：[PNG](complete-task-comparison.png)、[SVG](complete-task-comparison.svg)，九条件全部绘出，无均值柱或统计显著性声明。[机器绘图核验](figure-check.json)无缺字/越界，人工读图核对了图例、点标记及标签。
- 服务器环境隔离结束记录：[final-isolation.json](final-isolation.json)。不公开原始进程命令行。

## 数据身份与复算

运行源码：`e72639f3083221bf862205446cf5a83143b22591`实现运行体系；`77c77a75c26f6cda96c38793394a8f794dd87711`修正MuJoCo状态复制API；`3d14590`将终端保持改为时间戳差并修复最终stdout数组序列化。未改变任务配置或冻结权重。

权重SHA256：`d9945f85a39944aaf241a67f7d2b25a15326a0222d0e503aeb05a7f188430509`。来源是此前固定的社区HIM Go2导出，非用户实机或原作者Go2原版性能复现。

原始trajectory、每20ms策略历史/动作、每宏段任务/资源、来源资产hash、运行源码archive、stdout与私有服务器快照保存在忽略目录`work/go2-frozen-20261019/downloaded/`。压缩包：

- `work/go2-frozen-20261019/results.tar.gz`：SHA256 `423c73bafb9206749d7d200bff5b15490d69dd9b09f36ff8e4cbe1c8e048e5b0`。
- `work/go2-frozen-20261019/holdfix-results.tar.gz`：SHA256 `ad4afba8865934a93ff7980f6f75a5a39e94ae238e0b24b85cd8dc261357ec08`。

仅三条停稳边界受影响的case重跑。与原轨迹同长度的完整trace字典逐项相同，三条均仅增加0.2秒并达到1.18秒严格合格时间跨度，未调整难度/权重/策略或搜索参数。所有额外执行计入总账。

```bash
python scripts/analyze_go2_frozen.py --replacement-root work/go2-frozen-20261019/downloaded/holdfix-waypoint --replacement-root work/go2-frozen-20261019/downloaded/holdfix-kinematic --replacement-root work/go2-frozen-20261019/downloaded/holdfix-policy-rollout
python scripts/plot_go2_frozen.py
```

分析脚本支持`--replacement-root`给出三个`holdfix-*`目录；它保留original证据，并按method/condition选取明确修复的记录。分析和绘图不加载策略、不调用MuJoCo、不新增仿真或训练。

## 当前可以与不可以声称的内容

真实整机基线和完整任务已完成；三组都成功，不能主张冻结策略缺少完成这条任务的能力。策略闭环预测进入实际MPC，真实轨迹进入事实损失与完整任务回报计算；仍没有策略学习、模型拟合、紧凑表示收益或同信息PO/TO增量证据。准确仿真预测组的零名义误差不等于实机/泛化精度，其全部138个选用宏步超过200ms周期。

物理接触按5ms执行计数，终点保持按50Hz记录样本首末时间跨度检查；不升级为采样间连续时间证明。任务资源不含启动稳定，但summary另给出所有尝试、启动、异常前额外启动与首次运动运行的总账。第一进程异常前CPU总时长没有最终落盘，保持未知。
