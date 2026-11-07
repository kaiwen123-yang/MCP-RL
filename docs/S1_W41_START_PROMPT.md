# W41 启动提示词

以下整段可交给能连接 D57 的 Codex。当前仓库完成的是审查、qp2 数值修复、恢复文件保护、长阶段合法返程评价修复与无墙钟截止改造；r30 待实现，A 尚未由本次审查恢复。

```text
请继续 MCP-RL 项目的实际工作。开场最多5行后直接核对现场、实现、正常构建、推送并执行已授权任务，不再询问是否继续，不重新展开泛泛计划轮。

执行端：autodl-d57:/root/autodl-tmp/MCP-RL，环境work/envs/d57，source scripts/autodl_d57_env.sh。复用服务器独立联网和持久GitHub登录，凭据不进仓库/日志。仓库https://github.com/kaiwen123-yang/MCP-RL。

先读AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md、docs/S1_W41_STABILITY_PLAN.md、docs/S1_W41_GPU_PLAN.md，以及results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/下README.md、technical-error.md、technical-independent-verification.md、history-review.md、current-handoff.json。其余索引按具体问题查，不重新下载全库或重做12例QP回放。现场最新job/checkpoint优先于2026-11-07文档快照。

目标：先稳定原Go2/TM02上的完整PANN→PPO→MPC联合执行，再验证同一地块跨足/返程恢复记忆的因果收益。当前联合采样和学习已运行，但稳定合法材料往返尚未形成；不能只看loss、更新数或存活时间宣布成功。

任务A：立即准备qp2普通续接W40 A，与任务B并行。
先精确核对tmux、PID、job、clock、未提交修改和最新有效断点；已经在跑就继续观察，不重复启动。不要用s1-w40前缀误匹配GPU，不kill/restart已完成B/GPU，不覆盖研究修改或同时上传的提交。
当前A为ep314、9508/36、有效tail147、excluded145；原目标9617/37还差109条/一次更新。B已完成9617/37、tail0，末次E30在15.680s跌倒，原job不能再跑。qp2修的是相同QP的零步长活跃集循环，12个真实C++输入回放已通过原容差；独立artifact based_on=qp1，逻辑控制器仍S5-r29。
若qp2还未正常构建，执行：

python scripts/prepare_s1_go2_r12_support.py --build S5-r29-qp2

正确frozen产物已存在则复用，不覆盖旧r29/qp1。若现场仍是上述断点，执行：

python scripts/start_s1_go2_r12_tmux.py \
  --job-dir outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261106T103028Z-w40-a \
  --resume outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w40-a/PANN_PPO_S5-seed1/resume-state-ep00314.pt \
  --controller-artifact S5-r29-qp2 \
  --no-wall-deadline \
  --session s1-w40-a \
  --network-mode academic

有更晚同job有效断点就用最新，不回退。恢复保存已修成不覆盖来源文件；以progress实际路径为准，可能带restart时间后缀。A此步不--new-phase、不改workers，保留actor/critic/optimizer/RNG、147条tail、fork_identity、phase_lineage和excluded145。现场确认恢复计数和tail正确；本次审查没有加载实际.pt完成训练恢复。补齐原目标后，用自身末次新update37做固定E30，原同case合格才E60。原未启动的行政评价记录保存，不冒充物理评价。

任务B：并行定位保持动力学的一致性，据证据实现单一S5-r30，正常构建后直接完整联合训练。
按W41计划第4–5节，固定已有A314/B E30事件的同一x_0/u，在2ms十子步内核对真实tau=clip(60(u-q_j)-5v_j)、接触/饱和分支、候选/nominal/部署状态，并比对相同分支composed A/B与有限差分。不能把采用不同动作的accepted轨迹与闭环之差当作模型误差。发现具体映射/导数接线错误先修；若这些已一致，再明确选一个真实饱和动力学一致的优化/反馈方案，或匹配k/K/value的完整受约束递推。
实际actuator clamp一直生效，后续子步请求饱和与跌倒的时间关联不证明唯一因果；不预设“全段未裁剪PD必须不饱和”。计划4.1–4.3的保持段投影只是可选设计。若证据支持采用，不能用十个旧状态静态盒交集代替状态依赖约束，一般不等式QP不能冒充boxQP；原K不是投影后的Jacobian，物理不可行与数值错误分别报告。记录额外耗时，不能把投影本身当作完成长期目标。
只做解释已观察分歧所需的同输入检查，不加刚地/纯MPC/E30准入、smoke/mock、恢复演习、hash认证或参数网格，不叠多机制。候选/nominal/部署共享纯因果求值，不改正在跑A的frozen artifact。
保持原Go2、TM02 .02域、PD60/5、力矩/接触/跌倒/路线、2ms hold、r28 trunk、r27 C2、r26 packet/十子步/TV、PANN/658D/2Hz cap/W38奖励，不同时改步态/动作维度/物理。r25撤回项及r17 native impedance、r18 body-twist旧失败不盲目复活。
推送具体r30源码/规格与W41父点映射后直接启动已授权完整联合训练，看到真实transition增长再交接。默认A收口tail0后，w41-a/b各自继承自身最新W40完整checkpoint/optimizer/RNG，不重新fork/reseed。每流新增2048条/8更新；父点若9617/37则目标11665/45，现场更晚则按真实父点加算。先声明新phase/目录/时钟/parent_map，再--new-phase --no-wall-deadline；不能对仍指向W40的契约直接新启动。每支末次新update固定E30，同case通过才E60，300/600只在原前级通过后安排。
新R12 E60/E300/E600须使用已修复的bidirectional_material_roundtrips_v1，合法去程、返程和完整往返分别达到1/5/10次，并保留原条件观察与evaluator哈希。历史结果不重写，不能只按raw圈数宣布材料往返成功。

任务C：执行已授权的GPU扩展实验，不阻塞两条联合主线。
按docs/S1_W41_GPU_PLAN.md连续做G1–G5，不再只做8个fixture诊断。保留W40已完成但aligned=false的结果；原8例加约24–48个真实事件状态，保存逐子步接触/solver/state/A/B与受扰终点，定位首个差异。让模型、十子步循环、扰动常驻设备，按实际176节点×49输入分块，批量现有候选，补齐cost/private phase/候选排序并接CPU iLQG外层，随后独立完整栈E30→通过后E60。旧记录缺private state时用具名新fixture，不冒充历史重放。现有原始负结果无须重新算或重传。
若证据指向固定MJX接触/吞吐瓶颈，选一项MPX、TurboMPC或新MJWarp对照，明确新预测器/求解器身份，在原TM02 plant和任务上检验；不把它伪称等价CUDA。对等价迁移仍用原FP64/接触/差分标准。不要每个阶段都停下来问是否继续，也不要同时装一堆不相关算法。
现场先确认另一项GPU训练及CPU/显存/可用内存。两主线新phase默认各12 MPC workers、Torch/BLAS1；GPU host1–2、编译最多2jobs。按自身峰值与当前余量扩大microbatch，96GB不能全占，PREALLOCATE=false不是硬上限。复用编译缓存；不误停另一任务，不开第三条12workers PPO流。G3/G4有有效结果就继续，不把显存占用或作者速度当项目成绩。
新建w41-gpu入口、目录与版本，deadline=null；旧start_s1_go2_w40_gpu.py/s1_go2_w40_gpu.py含28h规则，只留历史不再启动。原GPU10份NPZ已交付，必要新计算按上述产出结束，不设置墙钟强停。

结束与交付：用户最新要求覆盖旧交互4h、训练28h、评价4h、GPU28h。所有新/续接后台不预设墙钟完成时间，不加timeout到钟强停、不因旧deadline拒绝合法续接。按样本/更新、物理评价、声明GPU阶段工作完成、明确技术失败或用户停止结束；单次网络请求timeout、数值迭代guard、30/60/300/600s物理评价保留。无deadline不是吞错或无限重启，新错误保存首次完整输入，不增加泛泛失败次数上限。A共享clock迁移不能覆盖已完成B自己的历史期限/评价。
保留完整accepted/reference/K/H、c/投影delta/KKT、十子步hold残差、每次planner返回/policy age、真实计数/tail/祖先及首次技术故障。先推源码/计划与紧凑结果，已请求原件后台续传；Git/index短写串行整合，不强推，区分queued与published。推GitHub并核验后不在用户电脑另外归档，不自动删除服务器checkpoint/resume/视频/原件或上传副本；用户备份后自行删除。
交接给出当前任务、各支最新job/episode/样本/更新/tail/技术状态、实际artifact、合法通行与E30/E60、Git/raw交付及续接位置。开发完成就启动授权任务，不再次只交一份计划；也不要把回放或本次源码工作写成已经取得新的物理能力。
```
