# W40 Codex启动提示词

此提示词用于执行已授权的下一轮工作。发布本文件时r29和GPU后端尚未实现、构建或启动；W39已完成。复制下方全文即可。

```text
继续MPC+RL项目，在D57执行W40稳定性修订＋GPU MPC并行工程。先同步GitHub main并保留未提交修改，读AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md顶部6行及4/5/6节、experiments/terrain_memory/s1-go2-w40-plan.json、W39-review/code-audit.md与results-audit.json；GPU再读docs/S1_GPU_MPC_FEASIBILITY.md，文献方法/边界见W39-review/literature.json。开场最多5行后直接工作。

先检查autodl-d57:/root/autodl-tmp/MCP-RL的实际tmux、job、上传队列。W39两流已完成，各8593/33、tail0、excluded145；A最后ep284，B最后ep280，末次E30在10.476/17.592s跌倒。不要重复W39，不把它改回running。若W40或w40-gpu已有实际作业，按其真实版本/目录/时钟续接，不重复启动。

用约30分钟直接读服务器已存A E30 9.12–9.30s、B E30 14.63–15.10s，以及A ep260首次load后domain-exit关键原件；同时让GPU适配代码工作开始。已有证据是关节速度预测偏差与大K/饱和先于首次材料load；不是已证K/reference接错、漏限幅或PPO奖励算错。不等待原始上传结束，不重跑旧物理。旧完整accepted/K当时未记录，不得拿prediction参考改名或重跑来冒充。

主修订S5-r29按主计划4.1实施：在原目标范围与状态相关PD扭矩实现域交集中重解12维局部反馈QP，使用同计划非对角H=Q_uu、C=Q_ux、实际feedback scaling s和已选line-search前馈；活跃PD边界导数不能当固定目标界置零，不能执行时再加一次完整Qu取消已选alpha。仅逐轴clip可能与原force clipping瞬时等价，不算有效实现。nominal自身不可实现也处理，区间空显式restoration。候选/nominal搜索/部署同一路径，原2ms保持和十个0.2ms PD子步照常，瞬时可实现不宣称全hold安全。若已有原件定位更早的确定接口错误，优先修该具体项并说明替代依据，不叠加多机制。修acceptance_mode日志语义，不改历史结果。

保留原Go2、TM02 .02域、所有扭矩/接触/跌倒/路线和30→60→300→600语义；保留r28(v,J)、r27 C2、r26因果packet/十子步A/B/TV、冻结PANN、658观察、2Hz cap和W38奖励。本轮不同时改步态、奖励、材料物理或动作维度，不复活r25旧载荷分配。补新版本关键事件窗完整accepted/reference/K/H/C与子步执行证据，每plan一次无损保存，不在500Hz行重复全K，准确标明覆盖范围。plan冻结H/C/参考K每次存一份，但逐动作实际QP活跃集/边界/解/plan与插值身份需保存，不能把参考K冒充在线有效K；事件前窗引用的更早plan也保留。H/C/K按同一时间插值定义，使用regularization_used。

将r29正常源码/构建与W40接线完成：parent_map、独立training目录、祖先fork_identity与新phase_lineage、runtime r29日志开关、训练/评价/finisher和W40共享时钟。当前R12仍是W39历史契约，不能仅改任务名字直接启动r28。w40-a恢复training/w39-a/PANN_PPO_S5-seed1/resume-state-ep00284.pt；w40-b恢复training/w39-b/PANN_PPO_S5-seed2/resume-state-ep00280.pt。各自恢复Actor/Critic、optimizer、RNG、归一化及8593/33、excluded145一次；新training/w40-a/b目录，不重新seed、不合并分支、不回共同父点。

两主线默认各12 MPC workers；各14已经授权，无需再问。第三任务并行时优先2×12，Torch1/BLAS1，保留主线JAX精度及共享设置。各新增1024条/4更新到9617/37，首新batch8594–8849为update34，每256新数据正常更新后继续。必要正常构建、代码配置推送后直接开始完整PANN→PPO→MPC→Go2/TM02联合训练；不加纯MPC/E30准入、smoke/mock、恢复演习、worker网格或无关测试。保留原known_r25停止谓词，普通失败保存后继续，不新加失败次数上限。

额外启动独立w40-gpu任务：用户已明确授权本轮第三并行和必要GPU数值/性能/接入测试，旧两计算槽/禁止benchmark不阻止这项工作。实际做批量动力学/FD与候选rollout适配，优先MJX-JAX/MPX可复用结构及FP64一致性，不能只提交另一篇调研。GPUhost默认1–2 CPU线程、必要编译最多2job；必要新依赖放隔离固定版本环境，不升级两主线MuJoCo ABI。正常启动记录一次实际CPU/GPU/显存情况，给主线PANN/PPO留余量，不以填满96GB为目标。

GPU用W39固定已存状态核对原Go2/12维PD目标/36维切空间/四元数脚序/十子步/原限幅/四patch预测接触/因果packet/风险C2与峰值导数；当前真实plant继续CPU TM02，不用隐藏真值喂预测器。先一步和十子步，再A/B与风险C/D，再固定候选和一次完整planner调用，明确CPU基线版本，不能混r28和r29。导出已有六项内部计时，记录JIT、传输、同步、p50/p95、误差、峰值内存和对两主线吞吐影响。MJWarp的FP32/无AD等差异要显式评估，不把1e-6 FD直接迁移即称等价。不要同时移植多套求解器。满足对应一致性后可在第三任务用冻结策略/原CPU plant做独立标记短闭环接入测试，最多原E30时长，不算主线资格、不自动升阶、不热换A/B。若只完成某个batch层，交付真实可运行代码与结果并说明未完成的接口，不伪称完整GPU MPC。

交互≤4h，真实样本与普通日志推进后即可交接。两主线首次W40真实启动建共享28h绝对训练期限；第二流/普通resume不重置。各自训练后≤4h，仅自身本phase新update>33固定策略E30，原同case合格才E60，零新update不拿父权重凑数，不自动E300/600。GPU首次实际开始另设≤28h绝对上限，完成即停，resume不延长；不自动串接r30或新增训练campaign。

W39已有诊断后台队列继续传保存原件，先核对实际manifest和原deadline，不能为补传重跑旧finisher或重置旧时钟。新轮先推代码/配置/全部逐回合JSON/学习评价/结论，再后台上传已请求反馈、规划、参考/执行轨迹、rollout和必要physical/material数组，并继续研究。Git/index短写串行，保留远端同时上传的提交，普通整合不强推。大checkpoint/完整resume/全视频默认留D57，由用户导出存储盘备份后自己删除；Codex不自动删服务器研究输出或已上传副本，不在用户电脑另做归档。交接只写真实实现/构建/启动/采样与上传状态，分开主线能力和GPU工程结果，不把计划push、queue启动或60s停滞当成成功。
```
