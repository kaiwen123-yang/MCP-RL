# 小规模响应—策略共同学习原型

本目录是独立的受实现关系约束的联合学习算法原型。真实调用SPlaTES官方LayeredRL的`CEMPlanner/Predictor/Hierarchy/Level`接口，采用**`use_icem=False`的CEM配置**。它不是SPlaTES完整MI/TD-MPC2训练复现，也不代表已超过SPlaTES、ReCo或MR-ARL。

## 环境和固定上游

当前实测平台为Windows、Python 3.12、Torch 2.7.1+cpu；没有GPU、大规模仿真或机器人依赖。精确Python包版本保存在[requirements-pilot.txt](../../requirements-pilot.txt)。其他操作系统尚未运行验证。

在本仓库根目录执行PowerShell：

```powershell
# Git与GitHub CLI需事先可用；若gh已认证，无需再次登录。
gh auth status --hostname github.com

# 只在固定work路径不存在时clone；已存在路径仅核验，不reset、不checkout、不删除。
./scripts/bootstrap-pilot.ps1

# 新建本项目虚拟环境，不修改系统Python。
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    py -3.12 -m venv .venv
}
.venv/Scripts/python.exe -m pip install -r requirements-pilot.txt
.venv/Scripts/python.exe -m pip check

# 可重复的真实接口核验；没有训练。
.venv/Scripts/python.exe scripts/check-pilot-integration.py
```

Bootstrap使用已经从本地真实origin和HEAD读取的[固定清单](../../scripts/pilot-upstreams.json)，通过已认证`gh repo clone`取得官方源码。只访问：

- `https://github.com/nicoguertler/SPlaTES.git`，提交`90b3749abd318ff3817f14159e207d6aa6f3dc1d`；
- `https://github.com/nicoguertler/layeredrl.git`，提交`37111d3538e18e4c6bcf1d98eef07407624c2e77`。

默认保存到`work/four-paper-audit-2026-10-14/{splates-code,layeredrl-code}`。已有目录的origin、HEAD或工作树状态不匹配时停止并保留原样，不自动修复。`-CheckOnly`只核验当前目录和认证，不clone、不fetch。不会运行上游安装脚本或上游训练入口。两个仓库的原MIT许可证随clone保留，适配器不复制其核心实现。

## 只运行一次小型学习检查

完成接口核验后，可以用独立输出目录执行很小的算法检查：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$pilotRun = Join-Path 'work' ('pilot-repro-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
.venv/Scripts/python.exe -m response_codesign.pilot `
    --seeds 11 --updates 2 --common-updates 2 --test-batches 1 --output $pilotRun
```

这个命令只检查当前多组学习流程能运行，不产生足以判断算法优势的统计证据。正式小规模对照的种子、预算、网络容量、组别和评估状态，以该次输出`specification.json`及对应报告为准。不要把接口smoke、开发性短运行和冻结评估混合统计，也不要把已存checkpoint自动当作优于强基线的证据。

## 规划和Hierarchy接口

```python
cost, margin = sequence_objective(state, commands, task)
# state [B,D], commands [B,S,H,A], cost/margin [B,S]
# margin <= 0 表示满足回调声明的检查；经验检查不等于全域证书。
planner = build_splates_planner(sequence_objective, horizon=5,
                               n_samples=24, n_iterations=2)
result = planner.plan(state, task, seed=seed)
```

原CEM负责候选搜索；独立响应模型、任务目标和实现条件负责评价。最终从实际检查过的候选中选择最低任务成本的可行序列；无可行候选时返回`feasible=False`，不假定零命令是安全备用。

完整8同步环境、100物理步、10步命令保持，以及独立联合更新回调的实际示例见[接入说明](../../notes/learning-pilot-2026-10-15/integration-audit.md)。本次未核验异步重置。神经策略不继承此前经典控制器的全域证书。

## 精确计算账本

每次`plan`返回的`info`包含本次真实回调入口累计增量：

- `objective_calls`：真实`sequence_objective`调用次数；
- `candidate_sequences`：跨批次的候选序列总数，即每次调用的`B*S`之和；
- `candidate_macro_steps`：候选模型宏步总数，即`B*S*H`之和；
- `evaluated_sequences_per_instance`：每环境候选数，包含额外全批次重评。

如果某一个环境没有可行候选，最终返回序列会对**整个批次**重评一次，账本按整个批次计数。实例生命周期累计值由`planner.objective_counts()`返回；`reset()`只重置warm start，不清空计算账本。

接口核验已经用外部独立计数器检查混合可行批次：`B=2,S=12,H=2,iterations=1`，初值2条、采样12条、最终重评1条，因此每次实际`3`次回调、`30`条批量候选、`60`个模型宏步。连续两次规划仍各返回30条增量，生命周期累计60条。数据集生成、辨识、参数选取和额外训练评估不由这些MPC字段自动覆盖，必须在实验层分别入账。

## 已执行与尚未执行

- 已执行：当前依赖`pip check`；两个已有源代码目录的`bootstrap -CheckOnly`；原方法来源和SHA核验；单环境、8环境以及混合可行性入口计数核验。
- 新clone分支仅完成脚本检查，本轮没有为测试而再clone一份源码，也没有重新安装干净虚拟环境；不能把当前环境可用称为已完成全新机器复现。
- 上游源码核验先将CRLF规范成LF后比较内容SHA，允许Windows和LF checkout的换行差异；运行日志同时记录原字节SHA与LF规范化SHA。
