# Codex：W40 A 数值续接与 W41 稳定联合执行

## 当前执行入口

先读 [当前交接](docs/TERRAIN_MEMORY_PLAN.md)、[W41 计划](docs/S1_W41_STABILITY_PLAN.md)、[GPU扩展计划](docs/S1_W41_GPU_PLAN.md)、[启动提示词](docs/S1_W41_START_PROMPT.md) 和 [W40 最终审查](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/README.md)。本文件及用户 2026-11-07 指令覆盖旧 W38–W40 执行期限和过时断点；历史报告与原运行条件保留。

默认执行端 AutoDL 西北 B 区 D57，SSH `autodl-d57`，项目 `/root/autodl-tmp/MCP-RL`，环境 `work/envs/d57`，激活 `source scripts/autodl_d57_env.sh`。记录容量为 RTX PRO 6000 96GB 级 GPU、25 vCPU、120GB RAM；现场先读实际资源，不能把总容量当空闲量。复用独立联网和持久 GitHub 登录，凭据不进仓库或日志。

## 已确认事实

- W40 A 最新交付为 ep314、9508/36、有效 tail147、excluded145，因 qp1 数值故障停止。原目标9617/37只差109条/一次更新；B已完成9617/37、tail0，末次E30在15.680s跌倒。合法去程/返程/往返均0。实际最新job/断点优先于文档快照。
- qp2 修复相同 QP 内部提前释放边界产生的零步长循环；12个真实C++输入回放通过原容差，未改H/中心/约束/物理。当前仅源码和回放已完成，尚未由此审查在D57构建或恢复。逻辑控制器仍S5-r29，产物独立S5-r29-qp2，based_on=qp1。
- 恢复保存已修复为保留来源checkpoint，实际新路径可能带restart后缀，以progress为准；新R12 E60/E300/E600增加已有合法材料去返/往返计数要求，旧结果不改写。独立复核报告明确区分真实QP/轨迹重评分与尚未完成的D57恢复。
- GPU原W40已完成batch predictor层，aligned=false，8例3例对齐，10份NPZ已交付；无整planner或正式资格。旧GPU入口含28h规则，现已退休，不再用它启动新任务或重跑旧JIT。
- PANN→PPO→MPC→PD→Go2/TM02联合采样与更新已运行。长期目标是同一地块跨足/返程恢复记忆改善加载/落脚/重访时序；稳定材料通行与因果记忆收益都未建立。G0/G1是Go2配置，不是换成人形G1。

## 执行规则

1. 开场最多5行后直接工作。精确核对tmux/PID/job/clock和未提交修改，已运行就不重复启动。不用前缀误匹配GPU，不kill/restart已完成B/GPU，不覆盖现场研究改动或同时上传提交。
2. 任务A正常构建独立qp2并普通resume自身最新W40断点，保留147合法tail、actor/critic/optimizer/RNG、fork_identity/phase_lineage/excluded145。此步不new-phase、不回退W39父点、不重新fork/reseed。已完成B不重跑。命令和未启动行政评价的保存语义见启动提示词。
3. 任务B与A并行：先用已观察事件的同一x/u核对真实饱和actuator map、2ms十子步及相同分支composed A/B与有限差分，在候选/nominal/部署保持纯因果一致。按具体证据实施一个r30修复；不预设全段未裁剪PD必须不饱和，计划第4.1–4.3节是可选约束设计。若采用，不以十个旧状态静态盒交集冒充解、不把原K当投影后Jacobian、不半改backward/value声称完整约束DDP。正常构建并推送后直接完整联合训练，不另设刚地/纯MPC/E30准入、smoke/mock、恢复演习、hash认证或参数网格。
4. 保持原Go2、TM02 .02域、PD60/5、力矩/接触/跌倒/路线、r28 trunk(v,J)、r27 C2、r26因果packet/十子步/TV、冻结PANN/658D/2Hz cap/W38奖励；不同时改步态、物理、奖励或动作维度。r25撤回项不复活，r17 native impedance/r18 body-twist旧失败不换名重试。
5. r30未实现前不能把当前契约改成已运行W41。W41新phase须以两支自身最新tail0父点分别继续，不混optimizer/RNG或从同一父点再fork。默认每支新增2048/8；若父点9617/37则目标11665/45，否则按实际父点加算。保存新目录/parent_map/时钟路径及artifact边界。末次新update固定E30，同case真通过才E60；300/600仅在前级通过后安排。新长阶段保留bidirectional_material_roundtrips_v1，合法去程/返程/往返分别达到1/5/10次，不能只按raw圈数评分。
6. **后台不设固定墙钟完成时间。** 用户最新要求覆盖旧交互4h、训练28h、评价4h、GPU28h。当前以样本/更新目标、声明的评价/fixture工作完成、明确技术失败或用户停止结束，活动deadline为null，旧值入历史。单次网络请求timeout、数值迭代guard及物理30/60/300/600s保留。无deadline不是无限重试/吞错；不新加泛泛失败次数上限。
7. 默认两主线各12 MPC workers，Torch/BLAS各1；普通resume保持自身冻结workers。第三GPU任务host1–2线程、编译最多2job，必要时降低其竞争，不能三个12workers流挤25核。2×14不是默认。按实际样本/h与资源快照判断效率，显存占用不代表研究产出。
8. GPU第三任务已获用户本次扩展授权，执行W41 GPU计划G1–G5：接触/导数定位、真实状态库扩展、176节点与候选的驻留分块计算、完整cost/phase/排序与CPU iLQG接入、独立E30/E60；有依据时只选一项MPX/TurboMPC/MJWarp替代对照。不再限为八例诊断；不重跑已完成W40或复用旧28h入口，新建无deadline任务。等价迁移保留原标准，新后端另立版本在原plant评价，不热换主线、不升级主环境。现场另一项GPU训练保持；按真实空闲量与峰值扩大microbatch，不把总96GB当全可用。
9. 保留完整accepted/reference/K/H、Q_u/Q_xu、c/投影delta/活跃集合/KKT、每子步hold残差、每次planner返回原因和policy age、学习tail/版本/祖先及首次技术故障输入。用事件窗与无损压缩，不静默过滤已请求大文件。旧记录不存在的内容不能伪造或重跑旧物理补成历史。
10. 先推源码/计划与紧凑结果，已请求原件后台续传；Git/index短写串行合并，不强推。queued不是published，compact完成不是全部raw交付。A修改共享clock不能覆盖已完成B自己的历史deadline/评价。
11. **推到GitHub并核验后，不在用户电脑另外归档。不自动删除服务器研究输出或上传副本。** 大checkpoint/完整resume/视频与原件留D57，用户备份后自行统一删除。真实部署、回放、采样、能力评价分别报告，不将计划或数值修复写成稳定通行成绩。
