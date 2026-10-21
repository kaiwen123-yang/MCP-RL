# Go2反馈学习与完整任务基准

更新：2026-10-21。当前伴随控制变量和四秒模型展开辅助学习已经退出主攻；保留历史实现与负结果。固定27.32米任务用作完整回归基准，尚未据此选出值得专门学习的新反馈能力。用户已授权整合PR底座；下一阶段转向复杂地形连续技能，当前不训练。见[执行边界](../../docs/RESEARCH_BOUNDARIES.md)。

已实现并实际运行过策略学习、真实响应再辨识、完整任务评价及后续不训练的MPC修复。旧文中的“学习未运行”“没有检查点”只描述10月20日实现交付时点，不能作为当前状态。

当前实现依赖独立的 [`go2_robot_lab`](../go2_robot_lab/README.md) 模块，使用 robot_lab 的45维单帧策略接口。main 原有 HIM 的六帧历史接口保留在 `go2_frozen`；两套策略、物理适配与结果分别保留。下方历史记录中的旧模块名和已保存结果的来源哈希维持原时点身份。

- 当前研究决定：[CURRENT_STATE](../../docs/CURRENT_STATE.md)。
- 固定任务与预算：[go2_continuation.json](../../configs/go2_continuation.json)。
- 现有检查点的完整评价入口：[validate_go2_mpc_repair.py](../../scripts/validate_go2_mpc_repair.py)，默认先验MPC；`--response-mode local_response`使用实测校正的局部学得响应。两种方式均不训练。
- 整程归因与验证结果：[go2-mpc-repair-2026-10-21.json](../../results/go2-mpc-repair-2026-10-21.json)。

当前最佳整程结果不依赖新增反馈学习；不把运行成功、预测修复或模型接口接通称为辅助机制的科研增量。历史训练配置保持原样，保证结果可追溯。

<details>
<summary>10月20日实现交付说明与原训练入口（历史记录）</summary>

# Go2 长程任务：普通 PPO 与规划续接伴随候选

状态：**实现已交付，学习未运行。** 原任务、机器人模型、公开策略、PD、场景与停止判据均从既有`go2_frozen`复用；不合并PR。候选的理论和区别见[主方法](../../docs/go2-adjoint-learning-method-2026-10-20.md)，比较方案见[实验协议](../../docs/go2-adjoint-experiment-2026-10-20.md)。

## 方法与入口

三组共享完整12关节actor、19维任务上下文、20ms动作模型f、200ms命令响应g、余程V和4s/256×3的MPC。g只拟合当前均值策略的实际完整宏转移；f只拟合实际动作转移；V只拟合实际回报。第三组额外计算旧反馈短展开的伴随方向，在原PPO目标外加入解析补偿的控制变量。

这是一份直接用PyTorch实现的共享clipped PPO后端，不声称复现Tianshou、SPlaTES或某篇已有算法的完整训练器。没有为了适配它改动既有冻结仿真的依赖环境；可使用PR已存在的`work/venv`和`requirements-go2-frozen.txt`。

| 文件 | 职责 |
|---|---|
| `state.py` | 48维物理状态、原阶段记忆、19维共同任务上下文 |
| `policy.py` | 原生复制公开actor；扩展输入新增列为零；真实高斯原始动作与密度 |
| `execution.py` | 学得动作进入原缩放、PD和物理步进，不复制另一个机器人环境 |
| `models.py` | 只接受事实监督的f/g，以及实际回报V；V另含宏相位 |
| `planner.py` | 用完整预测状态传播g，公共MPC及终端V；不微分CEM |
| `learning.py` | 连续整程return、短闭环伴随、原PPO与独立CV、事实拟合 |
| `collection.py` | 同一实物仿真采集、原始动作、终止/截断、完整宏标签、任务与计算记录 |
| `runner.py` | plan、无更新verify、后续显式train、三组共享初始化与版本循环 |

在本PR工作树根目录运行：

```powershell
# 只打印预算，不构造模型、采样或训练。
& work/venv/Scripts/python.exe scripts/run_go2_learning.py plan

# 模型/损失接线核验，全部权重不变，不运行物理。
& work/venv/Scripts/python.exe scripts/run_go2_learning.py verify --output work/go2-adjoint-check-fresh

# 可选：增加一条原nominal完整任务，无训练、无拟合。
& work/venv/Scripts/python.exe scripts/run_go2_learning.py verify --with-physics --output work/go2-adjoint-mission-fresh
```

输出目录必须全新。原始执行数据、派生策略检查点与优化器状态均留在忽略的`work/`。失败不会自动重采或改参数，异常写入`execution_failure.json`并停止该次调用。`verify`不调用任何优化器step；其中确定性数据只检查损失接口，不能用作PPO训练样本。

训练代码已实现，但本轮没有调用。以后明确授权固定实验后才执行：

```powershell
& work/venv/Scripts/python.exe scripts/run_go2_learning.py train --output work/go2-adjoint-pilot-fresh
```

`train`先用冻结actor取得共同模型/价值初始化，复制全部网络和优化器状态，然后分别执行三组。每轮顺序为随机行为采集 → 用旧网络和动作前状态/计划重构方向 → actor/V更新 → 新actor确定性校准 → f/g事实拟合 → 完整任务评价。模型不能先使用当前批后果拟合后，再反算这批控制变量方向。

当前没有已训练的新增f/g/V或任务actor检查点。零初始化f/V使初始伴随方向为零是如实的冷启动状态，不能以随机权重制造“机制已生效”的证据。

## 关键统计身份

- 真实奖励为每步负实际时间，失败再加70秒等效惩罚；原成功/时间/资源指标分别评价。
- 原35秒任务deadline属于真实失败终止；仅采集预算截断bootstrap。阶段切换不重置return。
- V拟合随机行为回报，在均值模型和固定命令续接中是近似余程代理，不是确定性重规划价值的精确导数。
- f/g/V、actor初始结构和信息三组相同；普通组拟合f是为匹配机制信息。计算比较另报其未使用的f拟合开销，不能据被强制承担辅助模型开销的普通组宣称优于最简PPO的总耗时。
- 每组随机采样步数相同，校准/评价episode数与上限相同；它们的实际长度不同。全部物理步含站立准备与校准另计，不自动等于同总算力。
- 7200秒在每轮开始前检查，包含初始评价；一整轮可能超出这个阈值，结果记录实际耗时，不称硬时间保证。
- 当前3个已知条件仅为开发比较，没有新场景或留出组合泛化结论。

本轮实际完成的无更新验证见[证据入口](../../notes/go2-adjoint-2026-10-20/INDEX.md)。没有学习收益、方差下降、实时部署或论文创新的已验证结果。

</details>
