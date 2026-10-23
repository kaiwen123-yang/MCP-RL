# 四足机器人地面记忆辨识与跨步协同控制

**最终研究问题（用户于 2026-10-21 确认）：机器狗在会被自己改变的软地面上，怎样决定当前加载、落点和重访时间，为后足及返程留下可用的支撑条件？**

核心机制：同一空间地块的材料状态跨腿、跨步、跨返程持续演化；后续支撑需求反过来约束当前踩踏。第一版从慢恢复黏弹性地层开始。

## 只从这两个入口执行

- [AGENTS.md：执行约定和禁止事项](AGENTS.md)
- [TERRAIN_MEMORY_PLAN.md：论文复现、完整仿真、实机的具体交付](docs/TERRAIN_MEMORY_PLAN.md)

主问题固定。已有文献库、复杂地形技能、旧共同学习源码和负结果均作为历史资料保留，不再决定当前执行顺序。

## 三个待实验验证的贡献

| 贡献 | 实际要解决的问题 |
|---|---|
| 跨足材料记忆辨识 | 从加载、卸载、无载恢复和不同腿重访中，恢复能够预测后继支撑的材料状态；PINN 约束材料演化。 |
| 后继支撑需求反传 | MPC 将后足需要的支撑转成当前加载、落点和时间的可行选择，并利用空间重访关系控制计算量。 |
| 远期重访价值学习 | RL 评价超出 MPC 时域的地面后果，在同预算下改善完整往返任务；与普通价值函数及长时域规划比较。 |

## 固定实施顺序

| 阶段 | 必须拿出的成果 |
|---|---|
| R1：四足 MPC 论文复现 | 按 Turrisi 等 IROS 2024 的实验对象和方法恢复跟踪、步频适应比较，生成对应图表、逐回合数据与视频；完成论文的 50 回合统计设置。 |
| R2：材料记忆方法重实现 | 按 Rosenkranz 等 2024 的参考本构及隐变量学习方法，生成未见加载历史的预测曲线和训练结果。 |
| S：本课题完整仿真 | 同一机器狗连续完成进入软地带、穿越、停留、转向、返程；保留地块状态，产出强基线、三项机制消融和完整任务视频。 |
| H：实机验证 | 材料标定 → 前后足重访 → 完整往返，使用仿真阶段确定的算法、指标与对照。 |

**先完成 R1 的论文方法及全部核心仿真实验，满足[进入 R2 的条件](docs/TERRAIN_MEMORY_PLAN.md#r1-完成范围与进入-r2-的条件)，再进入 R2。** R1 与 R2 均完成对应成果后进入主机制仿真。第二轮已有完整闭环结果，但现有实验边界与未支持的原文比较仍需收尾，R1 尚未完成。不能把软件启动、一次行走演示、若干短跑或跑满回合称为复现成功。原论文实机部分留到 H，未完成前不称“全文全部复现”。

## R1 收尾成果与复跑入口

本轮固定预算已于 2026-10-22 至 23 日完成：**103 个新回合、零训练**，包括固定 2.0/2.4 Hz 各 50 回合，以及空闲计算窗口内图 3 三方法各一条完整 60 秒轨迹。[收尾结果索引](results/terrain_memory/reproduction/pympc/r1-closeout-20261022/index.json)提供四组比较、机械正功、独占窗口计时、七段连续视频及全部原始数据下载。不要重跑已完成批次。

历史落足恢复 205 回合、时钟修正 101 回合均保留原始来源。四频率组的 60 秒完成数依次为固定 1.3 Hz 0/50、2.0 Hz 8/50、2.4 Hz 28/50、自适应 22/50。固定 2.4 Hz 纳入后续强 baseline；在线自适应优于该高固定频率的证据未建立。原图 4 的完整速度/推力协议及表 I 的时长、初始化、扰动生成与失败定义仍缺失，**R1 保持未完成，不进入 R2**。

| 本轮任务 | 新实验预算 | 配置 |
|---|---:|---|
| 固定 2.0 Hz，同一 adaptive 实现 | 50 个 seed × 最长 60 秒 | [r1-fixed-2p0.json](experiments/tm01/r1-fixed-2p0.json) |
| 固定 2.4 Hz，同一 adaptive 实现 | 50 个 seed × 最长 60 秒 | [r1-fixed-2p4.json](experiments/tm01/r1-fixed-2p4.json) |
| 图 3 独占计时，acados/naïve/MPPI | 各 seed 0 × 60 秒，共 3 回合 | [r1-fig3-isolated.json](experiments/tm01/r1-fig3-isolated.json) |

以下为独立复跑命令；`r1-closeout-20261022` 已完成，复跑须使用新的输出前缀：

```bash
bash scripts/run_tm01_r1_closeout.sh frequency NEW_RUN_PREFIX
```

独立复跑计时组仍需取得无竞争 CPU/GPU 计算负载的窗口；`isolated` 标签不会自动独占设备：

```bash
bash scripts/run_tm01_r1_closeout.sh timing NEW_RUN_PREFIX
```

两个入口共执行 103 个新回合。已完成结果的专用分析命令为 `python scripts/analyze_tm01_closeout.py --run r1-closeout-20261022` 和 `python scripts/analyze_tm01_closeout_timing.py --run r1-closeout-20261022-fig3-isolated`。四组比较引用的旧 100 回合不算新运行。实际计时的首次编译、稳态分位数和预算超时比例均在结果索引中；500 Hz 仿真采样率不代表实机控制达到 500 Hz。

原服务器环境已应用 `author.patch` 和一次 `foothold-2024.patch`，不要重复应用或重新安装环境。中断后只接续未完成回合；不要从头重跑整批。新环境才使用 `setup_tm01_r1.sh`。共享 GPU 的既有耗时保留为共享负载观测，不用于独占速度排名。

首次准备资料时使用：

```bash
python scripts/prepare_terrain_memory_sources.py
```

该脚本实际下载 [固定来源清单](literature/terrain_memory_sources.json) 中的 PDF，并取得固定提交的 Quadruped-PyMPC。下载位置为 `work/terrain-memory/`，来源与文件指纹写入该目录的 `sources.lock.json`。也可用 `--papers-only` 或 `--upstream-only` 分步取得。

原文方法对应关系和执行参数见实际结果索引及配置。原作者入口为 `simulation/simulation.py`；历史整组复跑入口 `scripts/run_tm01_r1.sh`、`scripts/run_tm01_r1_clock.sh` 和本轮 `scripts/run_tm01_r1_closeout.sh` 均保留。已有资料无需重复下载。

**当前作者代码默认包含原论文之后加入的方法和参数变化。R1 必须恢复论文中的 naïve sampling / MPPI / acados 比较，不能直接拿当前默认 OT-MPC 的结果冒充 2024 论文复现。** 机器人模型及依赖差异随实际结果说明。先完成 Aliengo 方法复现，Go2 作为后续移植。

## 成果与工作方式

每阶段交付可重跑命令、配置、数据、模型、图表和连续视频。禁止用防御性编程、单元/冒烟测试、审查报告和状态流水替代研究。允许并要求必要的训练、完整实验和明确故障的直接修复。

成果在实际生成后进入 `results/terrain_memory/`；大体积运行数据保存在 `outputs/terrain_memory/`。这些是约定的产物位置，不是已经完成的实验声明。

历史工作可通过 Git 历史、原 `src/`、`results/` 和 `notes/` 查阅；无需把旧任务的回归或百篇阅读当成本主线的门槛。
