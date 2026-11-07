# S1 求解器并行与 GPU MPC 可行性审查

## 2026-11-07 当前执行入口

W40 GPU已完成但未对齐，完整结果见[W40最终审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/README.md)。用户本次授权扩大GPU支线，当前以[W41 GPU扩展计划](S1_W41_GPU_PLAN.md)为准，包含批量动力学、完整规划和独立闭环；后台无固定完成时限。下面2026-11-05/06的任务快照、两槽安排、旧提示词与截止时间均是历史，不是新任务入口。

## 2026-11-06执行更新：第三并行任务已授权

**原件二次复核更正：** W39的976项原诊断已发布；新执行依据为[原始审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-raw-review/README.md)和[重放输入合同](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-raw-review/gpu-replay-audit.json)。r29现指保留旧target-clipped策略的非对角H投影，旧Qux中心公式撤回。GPU需同版本匹配，固定target的A/B不能混入policy/QP导数；成本/整planner重放需声明task phase和初始policy/warm-start，不能把未保存的旧K/H当作已上传。此更正不撤销第三并行测试授权。

**W39两流现已完成；本轮用户明确允许两条稳定性主线之外新增GPU MPC适配与测试。** 当前执行入口改为[主计划第5节](TERRAIN_MEMORY_PLAN.md#5-第三任务本轮实际做-gpu-mpc-适配与测试)、[W40计划](../experiments/terrain_memory/s1-go2-w40-plan.json)和[启动提示词](S1_W40_START_PROMPT.md)。下文2026-11-05计时、运行快照和旧“不另做benchmark/暂不增加第三任务”的安排是历史记录；后者不再阻止本轮明确授权的必要GPU数值、性能、接入测试。尚未实现/构建/运行GPU后端，本次只发布计划。

W39全62回合复算：普通planning87.4371%＋启动planning2.0593%＝89.4964%回合墙钟和，平均普通规划1.785931s；两流共享训练14.4560h、2048新样本＝141.671/h。见[新审计](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-review/results-audit.json)。这些是同步仿真耗时，不是实时控制延迟根因。

第三任务默认1–2CPU host线程，与两主线各12workers并行；需要的新依赖隔离固定版本。先做原12维PD目标问题的GPU批量预测/FD/rollout，优先FP64与原差分语义；报告真实同状态误差和整步计时，不能只有调研，也不能用缩减模型或内核微秒代表完整GPU MPC。最新官方[MuJoCo XLA](https://mujoco.readthedocs.io/en/latest/mjx.html)和[MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html)仍区分JAX自动微分与Warp吞吐/FP32限制；实际安装版本与主线MuJoCo3.2.3并不自动相同。

---

日期：2026-11-05。审查代码与结果固定在 [ccf4919](https://github.com/kaiwen123-yang/MCP-RL/tree/ccf49196d38be60156b62e83fdd3beabf0c6767a)。本文件记录源码审查、已保存数据复算、公开实现选型和后续工程建议；没有实现或部署 GPU MPC，没有新增物理实验，也没有改变 W39 的运行配置。

发布时接续了 [5265257](https://github.com/kaiwen123-yang/MCP-RL/commit/52652577205a7e5711f22209ea32c701f99eea7b) 的服务器交接：新 worker 政策已部署，当前 W39 仍各 8 workers；新快照为 A 新增 576/2、B 新增 577/2。该提交的主计划顶部六行完整保留。下文计时审查固定于 ccf4919，W39 第 2.2 节是当时的历史快照，不替代最新交接。

## 1. 决策

**GPU MPC 值得作为下一未启动版本的工程方向。** 已有证据表明规划调用占主要时间；增加 CPU worker 与研究 GPU 后端可以衔接推进，不必等待全部 S1 稳定性问题解决后才研究算力。

当前的 MPC workers 已经是每个求解器内部的 C++ 工作线程，不只是外层训练进程数。后续两条新任务各 12、可显式各 14 的策略已经在 ccf4919 接通。GPU 路线建议优先保留现有 iLQG 的问题定义，研究批量动力学、有限差分和候选 rollout；将 MPX 等 GPU 求解器作为下一层架构选项。普通 CUDA/PyTorch device 设置不会自动迁移当前 C++ MuJoCo 求解器。

本次用户对 GPU 的询问开启了这项研究，旧文档中“当前不重写 GPU planner”的安排不应解释成永久禁止 GPU 研究。当前 W39 继续原作业；GPU 后端落地应有自己的代码版本、配置身份和实际记录，不能在现有作业中热换。

## 2. 已保存数据：规划确实是主要耗时

### 2.1 完整 W38 训练统计

读取 seed1、seed2 的 ep217–251，共 70 个详细结果，包括失败和目标截断，未按结果筛选。文件均在 [R12 结果目录](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/)，例如 [A217](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/train-PANN_PPO_S5-s1-ep00217.json) 和 [B251](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/train-PANN_PPO_S5-s2-ep00251.json)。

分母为 70 个回合各自 wall_s 之和：100413.05750288814 秒。两流同时运行，这个和用于统计时间构成，不能当成两流共同训练墙钟。

| 计时项 | 回合累计秒数 | 占回合墙钟和 | 调用次数 | 平均每次毫秒 |
|---|---:|---:|---:|---:|
| 常规 planning | 88460.387848 | 88.096499% | 50444 | 1753.635474 |
| startup_planning | 2358.170090 | 2.348470% | 1400 | 1684.407207 |
| feedback | 555.481117 | 0.553196% | 504087 | 1.101955 |
| state_parameter_rpc | 953.470533 | 0.949548% | 504087 | 1.891480 |
| physics | 6067.005336 | 6.042048% | 504087 | 12.035631 |
| 五项未覆盖余项 | 2018.542579 | 2.010239% | — | — |

常规与启动规划合计 90.444968%。常规规划平均 1.753635 秒，50444 次均超过逻辑 20 ms。这里的 50 Hz 是同步仿真的逻辑规划频率，不能写成已达到墙钟实时控制。两条末端 E30 单独统计，普通规划也占其回合墙钟和的 87.938963%。

计时边界来自 [runtime](../scripts/s1_go2_r11_runtime.py)：planning 包住同步 planner_step 调用，含调用和等待；随后 best_trajectory 取回不在其中。feedback 包含实际/nominal action 的两次 RPC。physics 是每个 2 ms 控制 tick 内的十个 0.2 ms 子步，含材料处理、接触读取和记录，不能称为一次 mj_step 耗时。PANN/actor/PPO 的 joint.control 位于这些计时区间之外。最终部分文件压缩和 finisher 也不在这个回合 wall_s 内。

现有材料计时属于 physics 的内部子区间，不能重复相加。规划慢会限制研究吞吐，但同步仿真会等待规划返回；不能仅凭墙钟慢，把躯干接触或通行失败归因为实机式控制延迟。

### 2.2 W39 的证据范围

固定 ccf4919 的 index 已列 11 个完成回合，但其 result 指向的详细逐回合 JSON 尚未提交，不能复算 W39 的规划分项。索引中这些回合合计 wall_s=18096.051888、simulation_s=178.7098，约为 101.2594 倍墙钟/仿真时长；这只是已提交快照，不能套用 W38 的 88.10% 当成 W39 实测。

同一提交的顶层 job/index 比 jobs 子目录中启动时 collection 快照更新。当前状态应按主计划顶部六行与较新两流 job/index 读取，本报告不覆盖它们。

### 2.3 最小复算代码

在仓库根目录运行普通 Python 即可读取已有 JSON，不启动环境或仿真：

~~~python
import json
from pathlib import Path
root = Path('results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12')
rows = [json.loads((root / f'train-PANN_PPO_S5-s{s}-ep{e:05d}.json').read_text())
        for s in (1, 2) for e in range(217, 252)]
wall = sum(r['wall_s'] for r in rows)
for key in ('startup_planning', 'planning', 'feedback',
            'state_parameter_rpc', 'physics'):
    total = sum(r['timing'][key]['total_s'] for r in rows)
    count = sum(r['timing'][key]['count'] for r in rows)
    print(key, total, total / wall, total / count)
~~~

## 3. 求解器内部已经并行到哪里

| 层次/阶段 | 当前结构 | 下一步含义 |
|---|---|---|
| 两条 joint 训练 | 两个独立 job/策略/RNG/优化器 | 外层并行已经存在 |
| 每个 MPC | 显式 mjpc_workers 对应 AgentService 工作线程池 | 新任务 12/14 确实增加内部执行并行度 |
| 名义反馈候选、line search | 候选分别排任务，当前 10 条 | 多余线程不能制造更多候选；不为利用率修改候选数 |
| 动力学/代价导数 | 按时域节点排任务，当前网格 176 节点 | 很适合进一步研究批量执行 |
| 单条 rollout | 后一状态依赖前一状态 | GPU 可跨候选并行，不能直接打乱时间顺序 |
| 每个控制步 | 十个 0.2 ms PD/物理子步合成 2 ms | 子步状态与导数链有依赖 |
| iLQG backward/BoxQP | 当前按时域反向递推 | 时间并行需要改变线性系统求解组织方式 |
| 参数/动作 RPC、plant loop | Python 同步调用 | 不能由 GPU 求解速度直接替代 |

源码入口：[trainer](../scripts/train_s1_go2_r12.py)、[iLQG](../experiments/terrain_memory/s1-go2-r25-source/mjpc/planners/ilqg/planner.cc)、[节点导数](../experiments/terrain_memory/s1-go2-r25-source/mjpc/planners/model_derivatives.cc)，再叠加实际 [r26 patch](../experiments/terrain_memory/s1-go2-r11-S5-r26.patch)、[r27 patch](../experiments/terrain_memory/s1-go2-r11-S5-r27.patch)、[r28 patch](../experiments/terrain_memory/s1-go2-r11-S5-r28.patch)。r26-path-derivatives.inc 是较早草稿，不应拿其中的旧函数代替实际 r26 patch 的实现。

导数阶段中，名义节点已知后，不同节点及有限差分扰动可组织成批次；每个扰动内部仍执行正确的时间演化。GPU 并行工作量可以来自这些节点和扰动，不要求先新增数千条独立训练任务。

## 4. 保留现有控制问题时，GPU 要移植什么

优先候选为“CPU iLQG 外层＋GPU 批量预测/导数”，这是一条工程方案，尚未验证会更快。接口至少保留：

- Go2 的状态/四元数流形语义、12 维关节目标、原 PD/力矩限制；每个 2 ms 控制步仍执行十个 0.2 ms 子步。
- TM02 区域判断、原自定义 contactfilter/passive 力、材料 patch 的冻结参考与因果 packet。原有非足接触与躯干终止定义不能为了 GPU 速度删掉。
- 原代价残差、r27 C2 门、r26 原生子步 A/B 合成以及风险活动子步的导数回链。实际 R26RiskJacobian 的活动选择不能被“对整段 max 随手自动微分”静默替代。
- r28 原躯干 margin 的候选可行性、(v,J) 选择、restoration，以及 winner/reference/accepted 的一致语义。
- 时变反馈 K、参考轨迹、动作网格和原执行接口。两条独立流的模型输入、材料状态与策略谱系保持分离。

这里首先迁移的是现有 MPC 预测器：它使用四个冻结的世界坐标 patch、局部刚度/恢复参考与自定义接触力，并不推进真实 TM02 的全部内部材料状态。真实 Ground 可继续留在原执行环境，隐藏状态也不能为加速而额外提供给 MPC。依据是 [预测接触回调](../experiments/terrain_memory/s1-go2-r25-source/mjpc/tasks/quadruped/r10_contact.h) 与 [真实材料执行](../scripts/s1_go2_joint_material.py)。

具体策略是让模型和工作数组留在 GPU，整批计算后传回需要的 A/B/C/D、代价/可行性与候选结果，避免每个子步或矩阵都往返 CPU/GPU。首版可保留 CPU 上的 backward/BoxQP；若其实际占比随后变高，再研究时间并行的 KKT/Riccati 解法。

当前 C++ mj_step/mjd_transitionFD 和自定义回调不能通过 torch_device='cuda' 自动变成设备代码。MJX/Warp 提供的是可借用的 GPU 物理实现，不会自动编译本仓库的 C++ patch。

## 5. 最新一手实现如何借鉴

以下是功能和适配判断，不是将论文倍速外推给本项目。

| 工作 | 已确认能力 | 对本项目的用途与差异 |
|---|---|---|
| MPX / Primal-Dual iLQR，2025 预印本，2025-12 修订；RA-L 2026 卷期 | JAX、多重射击 SQP、并行 scan；官方提供 Go2/MJX 示例 | 最相关的足式 GPU 求解器参考。论文 WB 控制包含 GRF 与力矩，和当前 12 维 PD 目标不同；迁移意味着修改求解器/动力学接口 |
| GATO，ICRA 2026 | GPU 内外层批量 SQP、PCG，面向数十到数百个问题；机械臂实机 | 借鉴批量 KKT、线搜索、warm start；不等于已有 Go2/TM02 接触模型 |
| TurboMPC，2026-06 预印本 | GPU SQP＋ADMM，JAX/CUDA，可微与不等式约束，公开 RL 示例 | 对后续学习 MPC 参数很相关；论文仍依赖双精度，并讨论接触互补约束限制 |
| WarpMPC，2026-07 预印本 | 固定稀疏结构的批量 SQP/ADMM，Warp 展开 LDL 分解，MPC 数据生成 | 适合大批量结构相同的问题；两条当前任务不能直接套用大批量吞吐指标 |
| DIAL-MPC，2024 预印本 | GPU 采样 MPC，Go2 实机；MPPI 与退火 | 可作为算法替代方向，采用它会改变当前 iLQG；其 50Hz 实时配置关闭非足接触力，全接触爬箱另用非实时配置，不能直接套用其速度 |

一手论文与代码：

1. MPX：[论文](https://arxiv.org/html/2506.07823v3)；[代码](https://github.com/iit-DLSLab/mpx)。论文的时间对比只算一次优化迭代；WB 在短于 80 节点的例子中并不总胜过 CPU。其 GPU 可行性是直接证据，论文条件不能当成本项目速度承诺。
2. GATO：[论文](https://arxiv.org/html/2510.07625v2)；[项目](https://a2r-lab.org/GATO/)；[代码](https://github.com/A2R-Lab/GATO)。
3. TurboMPC：[论文](https://arxiv.org/html/2606.24039v1)；[代码](https://github.com/ToyotaResearchInstitute/turbompc)。
4. WarpMPC：[论文条目](https://arxiv.org/abs/2607.11603)；[作者代码与说明](https://github.com/hshose/warpmpc)。本次已直接核对作者 README，论文全文未成功取回，不将二手数字作为推荐依据。
5. DIAL-MPC：[论文](https://arxiv.org/html/2409.15610v1)；[作者项目与代码入口](https://lecar-lab.github.io/dial-mpc/)。

### 物理后端的选择

[官方 MJX 文档](https://mujoco.readthedocs.io/en/latest/mjx.html)说明 MJX-JAX 支持自动微分，适合批量执行；单场景不保证快。MJX-Warp 改善接触/约束吞吐，但目前不提供自动微分。

[官方 MJWarp 文档](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html)明确其面向高吞吐，单步延迟不保证优于 CPU；使用 float32，而常规 MuJoCo 默认 mjtNum 为 float64。因此，当前双精度有限差分与接触分支不能未经数值分析直接换成 float32。无自动微分不妨碍实现批量有限差分，但自定义材料/接触逻辑仍要移植。

选择 MJX-JAX、MJWarp 或自写 CUDA 应由所需动力学、精度、可批量程度和实际规划阶段耗时决定。96GB 显存提供容量余量，不是任意小矩阵/串行代码都会加速的依据。

## 6. 下一步工程顺序

1. **现有训练继续，CPU 新任务用 12/14 workers。** 当前 W39 不重启，不修改冻结契约。正常回合日志和已请求原始诊断继续按既有顺序推送。
2. **带出已有求解器阶段计时。** 先看已保存服务器普通日志；若未导出，在下一允许修改的正常版本把现有 nominal、model_derivative、cost_derivative、backward_pass、rollouts、policy_update 六个标量随 planner-records 保存。这些变量已经在 iLQG 中存在，单位为微秒。无需独立 benchmark 或新监控平台。
3. **优先实现批量预测/导数接口。** 从实际 r26/r27/r28 源码组合提取正确的单步、子步合成、风险行与候选输入输出；复用已有原始轨迹中的刚性地面、首次材料承载、躯干失效前状态做必要的数值对照。保留原物理语义，不通过减时域、降低子步、删除接触或换奖励取得表面倍速。
4. **进入正常完整 joint，判断端到端收益。** GPU 路径准备完成后在明确新版本中使用，继续保存原始反馈修正、选中/参考轨迹和必要数组。记录普通规划耗时、两流新增样本/共同墙钟、GPU 传输/JIT启动耗时及真实终止结果。时间与稳定性同时报告，不能用吞吐替代合法通行能力。
5. **再决定是否整体迁移求解器。** 如果 CPU backward/BoxQP 成为实际瓶颈，优先参考 MPX 的时间并行；需要显式丰富约束/可微学习时，继续评估 TurboMPC；需要成批多起点/场景优化时参考 GATO。每条路线都说明与当前 iLQG 问题的变化，不把 MPPI/SQP 更换记作原控制器单纯 CUDA 化。

研究和适配可以随当前训练并行推进；额外计算仍受现有硬件竞争和两计算槽安排约束。本报告不自动安排新 GPU 实验，也不自动修改依赖、精度或当前控制器。

## 7. 对速度的合理预期

用 W38 回合统计作条件算式：若普通＋启动规划占比 p=0.9044496821104134，且两类规划调用整体加速 s 倍、其余成本和工作量完全不变，则

S = 1 / ((1-p) + p/s)。

s=2、5、10 时，回合整体约为 1.826、3.617、5.376 倍。这是解释串行余项影响的算式，不是 GPU 实测预测。规划包含不可全搬上 GPU 的调用/等待；移植还可能增加同步与数据搬运。模型把回合内 PPO 等非规划成本视为不变，不预测轨迹或工作量变化及训练后交付成本。

实际目标先看“更少墙钟完成同样的有效 joint 更新”。实时部署能力和通行稳定性继续按自己的指标评估。

## 8. 后续 Codex 工程提示词

~~~text
继续 MPC+RL。先读取主计划当前交接，保留正在运行的 W39 作业与既有时钟。
读取 docs/S1_GPU_MPC_FEASIBILITY.md，研究求解器内部并行与 GPU 后端。

后续新 CPU job 默认两流各12 workers，显式各14已授权。
先利用现有普通日志里的六项 iLQG 内部计时；若尚未导出，下一允许修改的正常版本
把现有计时标量随 planner-records 输出，不新增独立benchmark、网格或准入轮。

GPU工程优先保持当前iLQG与原Go2/TM02/PD/十子步/r26导数/r27C2/r28(v,J)及反馈K语义，
研究批量动力学和有限差分接口；MPX、GATO、TurboMPC作为具名的求解器架构参考。
必须区分当前实现、拟移植内容和实际运行结果，不把更换控制模型当成等价提速。
必要数值对照复用已有原始事件状态；准备完成后用明确新版本正常joint衡量样本/墙钟与任务能力。
不热换W39，不改父谱系或原截止，不重跑物理来补录旧数据。

先推代码、配置、小记录和结论，已请求诊断后台上传。
不做电脑归档；未选大数据继续由用户导出备份后自行删除。
~~~
