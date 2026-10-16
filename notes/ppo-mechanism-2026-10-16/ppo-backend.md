# 成熟PPO底座接入与边界核验

日期：2026-10-16。用途：为同容量、同交互与更新预算的机制比较提供共同训练工具。本页不是新的算法贡献，也不报告机器人性能。

## 实际采用什么

使用已安装的 **Tianshou 0.5.1 `tianshou.policy.modelfree.ppo.PPOPolicy.learn`**。已经逐段检查本机安装的`PPOPolicy`、`A2CPolicy._compute_returns`、`BasePolicy.compute_episodic_return`及`value_mask`，以及[PPO原论文](https://arxiv.org/abs/1707.06347)的算法定位。固定版本的[官方源码](https://github.com/thu-ml/tianshou/blob/v0.5.1/tianshou/policy/modelfree/ppo.py)是源码复查入口。

`src/response_ppo/ppo_backend.py`不重新实现PPO目标。概率比裁剪、逐小批次优势标准化、价值MSE、熵项、梯度裁剪、多epoch反向传播与Adam step都在原版`PPOPolicy.learn`执行。适配器自身没有`loss.backward()`或`optimizer.step()`。

本地实现的是观测MLP、动作映射、时间优先rollout接口和张量GAE。未使用原版Collector、ReplayBuffer或`process_fn`；原因是原版返回计算会将rollout价值转到NumPy。张量GAE与原版回报计算按相同结束语义做了独立对照。这应描述为“使用原版PPO学习器的张量rollout适配”，不能描述为“原版完整训练流水线复现”。

默认设置：独立actor/critic，各两层64单元tanh；Adam 3e-4、eps=1e-5；gamma=.99、GAE lambda=.95；clip=.2；每批4 epoch；mini-batch256；value coefficient=.5；gradient norm=.5；初始Gaussian std=.35；默认entropy coefficient=0。没有承诺这些默认值已充分整定。实际比较必须冻结各组共同设置并报告网络参数与优化步数。

## 动作密度

先采样未限幅的潜变量动作`a~Normal(mu,sigma)`，执行`u=U*tanh(a)`。rollout保留原始`a`和它的Gaussian log probability；PPO在同一个固定原始动作上比较新旧概率。

对于固定、可逆的tanh映射，物理动作密度包含Jacobian，但这个Jacobian不依赖策略权重，在新旧log-ratio中相消。因此保留原始动作进行ratio计算合法。不能把`u`当作Gaussian采样再计算Gaussian log probability，也不能先clip采样然后假装仍是原密度。

适配器`action_log_prob`用稳定Jacobian公式提供物理动作密度，已检查两种log-ratio一致。上游熵仍是**潜变量Gaussian熵**，不是tanh后物理动作熵；默认熵权重0，避免模糊解释。

## 使用接口

```python
from response_ppo import PPOBackend, PPOConfig, Rollout

backend = PPOBackend(PPOConfig(obs_dim=observation_dimension, device="cpu"))
sample = backend.act(obs)  # obs [B,D]
next_obs, reward, terminated, truncated = environment.step(sample.action)
next_value = backend.value(next_obs)  # 必须在自动reset之前取真正末状态

# 收集T步；stack后除obs/raw_action之外各字段均为[T,B]
rollout = Rollout(
    obs=observations, raw_action=raw_actions, reward=rewards,
    terminated=terminated_flags, truncated=truncated_flags,
    log_prob=behavior_log_probs, value=behavior_values,
    next_value=successor_values,
)
metrics = backend.update(rollout)
```

`backend.act(obs, deterministic=True)`采用均值动作；也可以传入device匹配的`torch.Generator`隔离动作噪声流。`backend.actor`和`backend.critic`是普通PyTorch模块；`backend.provenance()`返回上游class、源码SHA256、参数量和配置。

采样到完成该轮PPO更新之间，不修改actor、能力定义或执行语义。适配器检查记录的behavior logprob与更新开始时actor是否一致，最大误差超过5e-5拒绝误用旧数据。该检查不能识别外部环境或能力含义被偷偷改动，因此模型版本仍由实验runner负责。

## 终止、截断与长rollout

GAE使用`delta = r + gamma*(1-terminated)*V(next) - V(current)`。递推在`terminated OR truncated`处停止。

- 真正任务终止：不bootstrap。
- 时间限制截断：保留真实末状态的bootstrap，但不把reset后新episode的优势串进来。
- 普通rollout收集边界：最后一个转移保留bootstrap，carry从0开始；不必伪造任务终止。

若使用自动reset环境，须提供其final observation计算`next_value`。直接使用reset后的观测会污染目标。有限任务成功应标`terminated`；仅为了截取较长任务训练数据而结束采样，不应标成真正terminal。

## 已执行的smoke

运行命令（仓库根PowerShell）：

```powershell
& '.venv/Scripts/python.exe' 'notes/ppo-mechanism-2026-10-16/check_ppo_backend.py'
```

新建CPU环境时可使用仓库`requirements-pilot.txt`固定依赖；该文件固定torch2.7.1+cpu、tianshou0.5.1等。smoke共4环境×16步=64个合成转移，3 epoch×4 mini-batch=12个**实际原版PPO优化步**，不是正式研究比较。

证据保存在[ppo-backend-smoke.json](ppo-backend-smoke.json)：

| 检查 | 实际结果 |
|---|---:|
| Tensor GAE与原版回报/优势最大差 | 1.43e-7 |
| 时间截断bootstrap误差 | 0 |
| 真终止bootstrap误差 | 1.49e-8 |
| tanh物理动作与raw动作log-ratio最大差 | 1.49e-8 |
| 更新前behavior logprob最大差 | 0 |
| actor参数变化L2 | 0.01961 |
| critic参数变化L2 | 0.05715 |

还以函数对象身份确认`backend.policy.learn.__func__ is PPOPolicy.learn`。上游`ppo.py`SHA256为`59e12218d75799872a306ce43d9385d230fa6097fe8a1f73f6ace297d6f4fd73`；本地适配器源码SHA256由smoke JSON记录，避免文档手工更新造成歧义。

## 并行能力边界

观测、动作、价值和GAE均保持在指定PyTorch device，环境维度批量计算；接口不要求逐环境Python Collector或在每步搬到NumPy。但是GAE仍有时间维循环，原版`Batch.split`有Python/CPU索引调度，原版learner每mini-batch会`.item()`取指标。不能宣称已经CUDA graph化或测得4096环境吞吐。

本机已验证的是CPU PyTorch，**未测CUDA、未测端到端GPU环境／规划器吞吐、未进行大规模训练**。机器人比较仍需明确实际模型/规划器更新成本，不能只统计PPO时间。PPO裁剪也不提供控制系统稳定或实现关系保证。
