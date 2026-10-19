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
