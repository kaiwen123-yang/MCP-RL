# 本轮原创紧凑响应模型检查点

这两份是本项目本轮拟合得到的响应预测器，不是Go2低层动作策略。原始社区HIM策略与机器人模型仍在忽略目录；部署时必须使用[主报告](../../docs/go2-compact-response-2026-10-19.md)固定的完整执行链。

| 文件 | 参数 | 字节数 | SHA256 |
|---|---:|---:|---|
| [id_one_step.pt](id_one_step.pt) | 26534 | 115338 | `ea2ad7ed78ed4767433e948b40cac67a23ee37001569d05657159430c80c1097` |
| [compact_multistep.pt](compact_multistep.pt) | 26534 | 115592 | `36c39d00039b0398827c6946d3d541e22ec9a4eb62007259a54d31858a29202b` |

格式`go2-compact-factual-v1`，含state_dict、训练集归一化buffer、输入/输出合同、固定数据与源文件hash。读取使用`torch.load(..., weights_only=True)`；推理通过`go2_frozen.compact.CompactPredictor`。两者当前输入信息、容量、初始化和批次索引相同，监督时域不同。

```bash
python scripts/run_go2_frozen.py --asset-root /path/to/frozen-go2-assets --methods id_one_step compact_multistep kinematic_mpc policy_rollout_mpc --response-model-root artifacts/go2-compact-2026-10-19 --output /path/to/new-evaluation
```

这是显式运行入口，不自动启动。评价环境、完整任务配置和CPU运行限制见主报告。模型只预测平面pose/twist与latent，未认证任意策略、地形、接触、实机或NX；无自动继续学习。训练/验证原始数据及完整轨迹留在忽略目录，公开结果可从[证据索引](../../notes/go2-compact-2026-10-19/INDEX.md)追溯。
