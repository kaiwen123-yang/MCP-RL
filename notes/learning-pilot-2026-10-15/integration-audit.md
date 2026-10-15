# SPlaTES工程接入范围与运行证据

日期：2026-10-15。独立算法研究对象仍为受实现关系约束的响应—策略共同学习。本文将工程复用和原论文复现分开。

## 1. 实际使用哪些原代码

适配器：[src/response_codesign/splates_bridge.py](../../src/response_codesign/splates_bridge.py)。

SPlaTES实验仓库对应`90b3749abd318ff3817f14159e207d6aa6f3dc1d`；主要算法实现来自其官方LayeredRL依赖`37111d3538e18e4c6bcf1d98eef07407624c2e77`，仓库根许可为MIT，版权归Nico Gürtler。原始仓库位于忽略目录`work/four-paper-audit-2026-10-14/layeredrl-code/`，没有修改这些文件，也没有将其核心算法复制到我们目录。

| 原接口 | 本次接入方式 | 保留／替换边界 |
|---|---|---|
| `CEMPlanner.plan` | 实例化原类并调用原方法 | 保留原初始化、原CEM搜索、原动作序列返回 |
| `BlackboxPlanner.cost` | 原CEM规划时实际调用 | 保留原Predictor rollout和奖励聚合；关闭终端价值和随机终止 |
| `Predictor` | 我们的`ResponsePredictor`继承原基类 | 响应、任务成本与实现检查由独立算法定义；不执行原MI/TD-MPC2训练目标 |
| `CEM.optimize` | 原规划器实际调用 | 取`use_icem=False`，因此本轮是原CEM，不称原SPlaTES默认iCEM完整复现 |
| `Hierarchy.get_action/process_transition/learn` | `build_hierarchy`返回原Hierarchy实例 | 保留上下层命令传递、控制交还和学习调度 |
| `Level` | 自定义规划层和执行层继承原类 | 低层调用神经反馈策略，按声明时长交还高层；联合更新放在上层learn回调，避免重复更新 |

真实文件校验值：

| 文件 | SHA256 |
|---|---|
| `hierarchies/hierarchy.py` | `f7dc27c4a998d67d8ffc881f6e23511fcaab096f708db3c18ed71bd8ac2e57ed` |
| `levels/level.py` | `8c0ef40eeb76962010b5749167c02144760de732164d6e9c393d52ccc026674a` |
| `planners/cem_planner.py` | `1ab8181adc6d2785925713de063f0e558ff026021cfa2371228cc973fc13a9ab` |
| `predictors/predictor.py` | `022e8fe6bafdf8a37e5dbb0bd94c3038523dc291a6aea4f1b7adcd8c9cd4c7ea` |
| `optimizers/cem.py` | `1a16357c0239fcaaa484ed4e43213ec82a35e06cbda1e9746d2cd441a2411cd1` |

加载器用Python标准`importlib`将上述实际文件载入独立命名空间，跳过会导入整套TD-MPC2的包级`__init__`。导出的类均来自原文件；没有用替身CEM、替身Batch或替身Hierarchy冒充原仓库。运行记录保存类名及`__file__`，可核验真实执行来源。

上表保留当前Windows checkout原字节SHA。加载时先把CRLF规范为LF后核验核心内容，避免一个干净LF checkout仅因换行不同而无法运行；日志同时提供原字节SHA和`source_sha256_lf`。公开复现入口、固定依赖和安全bootstrap见[原型运行说明](../../src/response_codesign/README.md)。

## 2. 响应与实现条件怎样接入原规划器

外部提供统一回调：

```python
cost, margin = sequence_objective(state, candidate_commands, task)
# state: [B,D]
# candidate_commands: [B,S,H,A]
# cost、margin: [B,S]，margin <= 0 表示满足声明的检查
planner = build_splates_planner(sequence_objective, horizon=4, n_samples=64)
result = planner.plan(state, task, seed=seed)
```

`ResponsePredictor.rollout`调用真实响应 rollout/约束评价，将完整序列成本放在第一步负奖励中，其余奖励为零。在存活概率为1、终端价值关闭的配置下，原`BlackboxPlanner.cost`的奖励聚合精确返回同一序列目标。这是接口转换，不添加新模型、不伪造模型轨迹、不改变数学目标。

候选搜索中用违反量罚项引导CEM，最终从实际评估过的候选中选择任务成本最低的可行序列。每次规划均额外检查同样的移位初值与零命令。若没有可行候选，显式返回`feasible=False`；零命令不被默认为安全备用。候选检查是经验实现误差、解析充分条件还是其他证书，由实验的回调及记录明确区分。

本次原CEM是有限随机搜索，不能对其elite筛选反向传播然后声称获得KKT最优解梯度。联合学习采用独立算法声明的梯度估计／更新，完整控制器评价必须包括实际规划过程。

## 3. 原Hierarchy与独立学习器的连接

```python
hierarchy = build_hierarchy(
    planner=planner,
    policy=neural_policy,             # policy(obs, command) -> action
    env=actual_environment,
    task_provider=task_provider,
    command_steps=hold_steps,
    transition_callback=record_step,
    joint_update=perform_joint_update,
)
action = hierarchy.get_action(obs)
hierarchy.process_transition(obs_next, reward, terminated, truncated)
hierarchy.learn()
```

这个路径保持实际环境、策略和响应模型由独立实验提供。低层每个物理步执行反馈并记录状态、命令、输入和下一状态；上层按命令持续时间重新规划。原`Hierarchy.learn()`经过上层Level调用联合更新一次。当前适配器面向同步批次，不声明支持异步多环境重置、原TD-MPC2 replay buffer或原SPlaTES互信息损失。

## 4. 运行状态和身份边界

接口代码已经完成Python语法检查。实际依赖、原接口调用与小规模控制核验见下方运行证据，未运行的内容不计作完成。

- 可称：在SPlaTES官方LayeredRL工程接口上实现独立响应—策略算法的适配器。
- 不可称：已复现完整SPlaTES训练、已达到原论文性能、已超过原SPlaTES。
- 完整原版仍是后续强对照；当前小实验的固定响应／独立策略是定义明确的机制对照，不冒充原版SPlaTES、ReCo或MR-ARL实现。
- 依赖最小核心为NumPy、Torch；完整接口还需Gymnasium、Tianshou 0.5.1、TensorBoard、colorednoise和PyYAML。并没有运行外部仓库的安装脚本或完整训练入口。

## 5. 本轮执行证据

已经在项目`.venv`中实际执行接口核验，退出码0。可公开复现的入口为[scripts/check-pilot-integration.py](../../scripts/check-pilot-integration.py)。记录保存在忽略目录`work/learning-pilot-2026-10-15/integration-check.json`，Python使用Torch 2.7.1+cpu。

| 已执行的接口核验 | 结果 |
|---|---|
| 原CEMPlanner实例调用 | `CEMPlanner.plan`，源文件SHA256与上表一致 |
| 原成本聚合 | `BlackboxPlanner.cost`，SHA256=`4a2b390406b5c7d97eb49b55ddc273cabc72dff07f2f6a61a6b8ade519c5d170` |
| 原CEM优化 | `CEM.optimize`，两次迭代、每次24候选，另检查2条初值，共50条序列 |
| 原Predictor基类 | 构造真实子类，2次rollout调用；原MI/TD-MPC2学习没有执行 |
| 原Hierarchy单环境 | 6个物理步、3次规划、6次转移回调、1次联合更新回调 |
| 原Hierarchy同步批量 | 8环境×100物理步=800条转移，命令保持10步，共10次批量规划、1次联合更新回调 |
| 所有候选不可行 | 返回`feasible=False`，未伪造可行备用 |
| 混合可行性、2环境 | 1次初值评价＋1次12候选搜索＋1次全批次最终重评=3次回调；总30条候选、60个模型宏步 |

JSON同时保存六个实际原方法的`__qualname__`、`__module__`、`inspect.getsourcefile`及整个源文件SHA256：`CEMPlanner.plan`、`BlackboxPlanner.cost`、`CEM.optimize`、`Hierarchy.get_action`、`Hierarchy.process_transition`、`Hierarchy.learn`。因此这是原方法实际被实例化和调用的接口证据，而非仅给自定义函数取相同名称。

这项核验使用简单标量环境和固定网络，证明接口、时序与回调能运行；它不是小规模训练的性能结果。后续实际非线性学习运行应单独记录使用了哪些原接口，不能把这项smoke调用全部计入主实验。

候选统计现位于实际`sequence_objective`调用入口，包含`missing.any()`时额外的**全批次**重评。`plan.info`返回本次`objective_calls/candidate_sequences/candidate_macro_steps`增量；`planner.objective_counts()`返回生命周期累计。外部独立计数器与内部账本在混合可行性连续两次规划上完全一致：各30条序列，累计60条；没有只按缺失环境数或固定公式补算。8环境smoke共30次目标回调、4000条候选、12000个宏步。

## 6. 8环境、100低层步、10步命令的实际调用形式

`env`必须是同步`gymnasium.vector.VectorEnv`，提供`num_envs=8`、`single_observation_space`、`single_action_space`以及对应批量空间。动力学必须由实验环境真实执行；示例不会用预测状态覆盖真实状态。

```python
from response_codesign.splates_bridge import build_splates_planner, build_hierarchy
import torch

planner = build_splates_planner(
    sequence_objective,
    horizon=4, command_dim=1, command_bound=1.0,
    n_samples=24, n_iterations=2, device="cpu",
)

# task_provider接收当前有效的物理观测和有效环境mask；
# 若任务是批量张量，应返回与active_instances匹配的子批次。
hierarchy = build_hierarchy(
    planner=planner,
    policy=sample_execution_action,
    env=env,                                # actual VectorEnv, 8 instances
    task_provider=current_task,
    command_steps=10,
    transition_callback=record_actual_step,
    joint_update=perform_response_policy_update,
)
obs_np, info = env.reset(seed=seed)
obs = torch.as_tensor(obs_np, dtype=torch.float32)
planner.reset()
hierarchy.soft_reset()
for physical_step in range(100):
    action = hierarchy.get_action(obs)       # original Hierarchy method
    next_obs, reward, terminated, truncated, info = env.step(
        action.detach().cpu().numpy()
    )
    obs_next = torch.as_tensor(next_obs, dtype=torch.float32)
    hierarchy.process_transition(
        obs_next, reward, terminated, truncated
    )                                       # original Hierarchy method
    obs = obs_next
hierarchy.learn()                            # original method -> one joint update
```

`sample_execution_action`可在自己的轨迹记录中保留实际采样动作、预tanh变量、响应采样值等统计量；训练时依据真实记录计算声明的score estimator。CEM规划本身处于`torch.no_grad()`下，不能偷渡优化器梯度。更新回调由实验决定何时执行；上面每个同步episode调用一次，相应物理交互为800步。

这个同步原型应在全部环境同一时刻结束或reset的协议下使用。随机异步终止需要另外核验原Hierarchy的mask处理，本轮不把它列为已支持能力。
