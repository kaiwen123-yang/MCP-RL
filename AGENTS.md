# Codex：S1先形成稳定、合规的完整联合基线

## 执行入口

使用AutoDL西北B区D57，SSH别名autodl-d57；项目/root/autodl-tmp/MCP-RL，环境work/envs/d57，激活source scripts/autodl_d57_env.sh。RTX PRO 6000 96GB级GPU、25 vCPU、120GB内存。复用现有依赖、服务器独立联网和持久GitHub登录；凭据不进入仓库/日志。

**当前任务：W39_CONTACT_EXECUTION_REPAIR_PARALLEL_JOINT / S5-r28；仅计划，尚未实现、构建或运行。** W38/r27两流已完成并发布。以[主计划顶部6行与第4/5节](docs/TERRAIN_MEMORY_PLAN.md)、[当前R12契约](experiments/terrain_memory/s1-go2-local-readiness-r12.json)、[W38统计](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W38.json)及[控制审查](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W38-code-audit.md)为准，启动词见主计划第8节。

1. 开场最多5行，随后直接工作；先查同流tmux和实际作业，有运行中任务就交接/续接，不重复启动。默认读当前交接、第4/5节、控制审查所需事件段及直接相关源码，不再全仓审查或另开计划轮。
2. W38两流各新增1024/4，累计各7569/29＝7424已训练＋145历史excluded，tail0，父恢复点各ep251。70个实际训练回合：27跌倒、40躯干geom13接触、1材料域退出、2目标截断；53加载、15机身进入、8进入后条带内部足载荷，合法去程/返程/往返0。两条末次策略E30在13.5634/13.5208s躯干接触失败，机身未进入x≥.8；各E60阻塞。不能把双流8个更新事件合成单模型8次更新，也不把训练目标完成当稳定能力。
3. 在同一≤4h交互内先读取两条既有E30关键数值窗，不新增物理：目标截断/raw-PD饱和、大Kδx、非足承载、材料load及终止；对齐时钟、脚序、plan id、accepted/reference/linearization、实际动作/力矩/Fz。accepted只存node0/+20ms，不能冒充全horizon。并行完成必要phase/交付接线，不先盲跑未改r27或再生成泛泛计划。
4. 按主计划4.2选择一项证据支持的控制修订。确证接口不一致先修接口；若accepted已预测原躯干终止，按原几何/margin做候选可行性优先、无可行候选显式restoration，并一致更新policy/acceptance/regularization；若feedback先改变活动集合，推导对应局部可实现域。selected_mechanism未确定前不得假装已实现。不得根据最大力矩预设K clip/低通，不把K0/K∞有意不同当作新bug，不叠加四项机制。
5. 保留原Go2、TM02 .02域、PD/力矩/接触/跌倒/路线和阶梯，保留r27 C2门、r26因果packet/十子步A/B/TV、冻结因果PANN、658维观测、2Hz标量cap与W38终止奖励。r25 MovingSupport/SupportPlacement/载荷分配及旧失败moving velocity项不恢复。必要实现、正常构建和代码配置推送后，直接完整PANN→PPO→MPC→Go2/TM02训练；不加单模块/纯MPC/E30准入、smoke/mock/测试套件、恢复演习、无关回归、参数网格、哈希认证或独立benchmark。
6. w39-a只继承w38-a seed1 ep251，w39-b只继承w38-b seed2 ep251，各恢复自己的Actor/Critic、optimizer、RNG与归一化，不重新fork/reseed、不回到ep216。历史excluded145只保留一次。新目录为large_root下training/w39-a/PANN_PPO_S5-seed1和training/w39-b/PANN_PPO_S5-seed2，父目录只读；接通每流parent map/new-phase边界/phase_lineage以及train/eval/finish路径，避免初始checkpoint覆盖父ep251。祖先fork_identity保留；当前phase身份另存，不能删原恢复保护绕过校验。
7. 每流新增1024条/4更新，到8593/33＝8448已训练＋145excluded、tail0；每256新数据更新后继续，首批7570–7825形成update30。两流各8 MPC workers、Torch1、BLAS1，共享原GPU/JAX设置。保持两计算槽，不凭显存容量改三流/worker网格；可在普通启动留一次真实利用率快照，记录实际吞吐。
8. 交互≤4h，看到真实样本与普通日志推进即可交接，不等首个完整更新。首次W39实际启动创建新的w39-training-clock.json，共享28h绝对训练截止；后续流启动/resume不重置，不删改旧W38时钟。各流训练后各自≤4h评价，只用本phase最后实际新update（>29）固定策略/归一化做nominal E30，同case合格才E60；零新update不拿父权重充当结果。E300/600不自动运行。保持原known_r25_startup_regression谓词，首次符合停止该流、跳过评价并发布；其余普通失败自然续采，不新增失败/attempt cap。
9. 两流计算并行、compact finish/Git串行且沿用index锁；保留两流交接，勿互相覆盖。紧凑发布包括全部逐回合JSON（失败/零步/中断/截断）、实际配置身份、2Hz transitions/reward/log-prob/term/trunc、12维PANN与value/next-value小表、关键事件数值片段、学习和独立评价/阻塞原因、结论。W38既有低频记录与两E30关键窗从保存文件一次性补推，不重跑物理。必需证据可按事件分文件，不能被通用大小过滤静默丢弃。
10. **先推紧凑证据，随后继续研究。** 大checkpoint、完整resume、视频和密集原数组默认留服务器，用户定期导出到自己的存储盘，备份后自行统一删除。Codex不自动删除研究输出（含已上传副本），不另做用户电脑归档，不默认全量渲染/上传，不为视频重跑物理。r26旧流程保持历史原样，不重复。现行交付授权优先于旧快照cleanup=true；紧凑push不是大数据备份完成。
11. 收尾只更新正常job/index和主计划6行，明确实际代码/构建/训练/评价/紧凑Git/本地大文件状态。Git失败保存结果后普通重试，不强推、不重训、不删除。GitHub计划已推送不等于D57部署已完成；本次审查无SSH、无新构建/实验。
