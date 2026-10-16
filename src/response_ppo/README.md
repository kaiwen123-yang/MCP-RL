# 成熟PPO与响应实现关系原型

本目录调用Tianshou0.5.1原版PPOPolicy.learn。它是2026-10-16的受控研究原型，未复现完整SPlaTES或ReCo；[完整结果](../../docs/ppo-realization-report-2026-10-16.md)没有支持共同学习的实质收益。`response_codesign`保留上一轮原型，不覆盖。

使用仓库原有[依赖与固定上游准备流程](../response_codesign/README.md)，同一个`requirements-pilot.txt`即可。当前已验证Python3.12、Torch2.7.1+cpu、Tianshou0.5.1；GPU路径没有端到端验证。

在仓库根目录运行开发检查：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
& '.venv/Scripts/python.exe' -m response_ppo.pilot --output work/ppo-local-check --seeds 901 --updates 2 --test-batches 1
```

正式比较的完整参数见[冻结协议](../../notes/ppo-mechanism-2026-10-16/protocol.md)。复现时使用新输出目录，保留已有结果。下面顺序运行三个种子；原报告采用三个CPU进程并发，因此运行时间不应直接比较。

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$reproRoot = Join-Path 'work' ('ppo-repro-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
foreach ($trialSeed in @(17,29,43)) {
    & '.venv/Scripts/python.exe' -m response_ppo.pilot --output (Join-Path $reproRoot "seedrun-$trialSeed") --seeds $trialSeed --updates 40 --test-batches 4
    if ($LASTEXITCODE -ne 0) { throw "Pilot failed at seed $trialSeed" }
}
```

文件职责：

- `ppo_backend.py`：actor/critic、潜动作与张量GAE适配，真正损失/更新调用原版PPO；没有自写PPO代理损失。
- `realization.py`：输入仿射离散模型下的精确一步可行区间，空集明确返回失败。
- `env.py`：带执行器与非线性项的Euler离散系统、持续参考、紧凑MPC及真实计数。
- `pilot.py`：同容量4组采样/训练/冻结评估；保存每batch原始指标和source hash。
- `midpoint_diagnostic.py`：固定最终响应、零网络学习的解析区间中点强对照。
- `scripts/summarize-ppo-mechanism.py`：正式聚合直接读取每batch；计数求和、最大值取max，不把runner的`evaluation.mean`用于这些字段。

P、rho、delta、模型版本和能力定义在同轮PPO多epoch内固定，不能更改支持后继续用旧行为概率。当前φ是四参数响应设计，输入支持与θ无关；不要将结构提供的保证写成RL学得的能力。数学范围是声明离散对象/域/扰动，不自动覆盖旧RK4或真实机器人。

Python/MATLAB复算脚本与结果位于`notes/ppo-mechanism-2026-10-16/`。原始权重、轨迹、PDF、仿真缓存仅保存在忽略目录；公开结果足以审查本轮判断，但新机器须重新执行才能恢复本地权重。
