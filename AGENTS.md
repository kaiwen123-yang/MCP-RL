# Codex：S1先形成稳定、合规的完整联合基线

## 执行入口

使用AutoDL西北B区D57，SSH别名autodl-d57；项目/root/autodl-tmp/MCP-RL，环境work/envs/d57，激活source scripts/autodl_d57_env.sh。RTX PRO 6000 96GB级GPU、25 vCPU、120GB内存。复用现有依赖、服务器独立联网和持久GitHub登录；凭据不进入仓库/日志。

**当前任务：W39_CONTACT_EXECUTION_REPAIR_PARALLEL_JOINT / S5-r28已实现、构建并启动。** 以84f5867的两流启动job/index及[主计划顶部6行、第4/5节](docs/TERRAIN_MEMORY_PLAN.md)为准；当前补充为已请求原始诊断交付。61fd022已补W38数值窗，9544cfe为r28运行源码；不得将活跃任务改回planned或重新启动。

1. 开场最多5行，随后直接工作；先查同流tmux和实际作业，有运行中任务就交接/续接，不重复启动。默认读当前交接、第4/5节、控制审查所需事件段及直接相关源码，不再全仓审查或另开计划轮。
2. W38两流各新增1024/4，累计各7569/29＝7424已训练＋145历史excluded，tail0，父恢复点各ep251。70个实际训练回合：27跌倒、40躯干geom13接触、1材料域退出、2目标截断；53加载、15机身进入、8进入后条带内部足载荷，合法去程/返程/往返0。两条末次策略E30在13.5634/13.5208s躯干接触失败，机身未进入x≥.8；各E60阻塞。不能把双流8个更新事件合成单模型8次更新，也不把训练目标完成当稳定能力。
3. 按主计划5.4用diagnostics queue对两个已完成W38 job后台补传，不重跑旧finisher；与正在运行的W39并行。先读61fd022已补两流各1024条学习/PANN/value和两E30窗口，必要时读D57完整原件，不重复已成功导出。A在材料load前已下降，B大反馈先于load；末端accepted不可行不等于早期首因。对齐时钟、脚序、plan id与三种轨迹角色；不为补传重跑物理。
4. r28已选择original_trunk_margin_candidate_feasibility_with_explicit_restoration，有限候选按原躯干margin的(v,J)选择，并同步参考/接受/正则语义；不是已证明任务稳定。保持当前控制版本，不在运行中加K clip、低通、重调接触或叠加其他机制。后续只读诊断及新发现留给明确的下一未启动版本。
5. 保留原Go2、TM02 .02域、PD/力矩/接触/跌倒/路线和阶梯，保留r27 C2门、r26因果packet/十子步A/B/TV、冻结因果PANN、658维观察、2Hz cap与W38终止奖励。r25 MovingSupport/SupportPlacement/载荷分配不恢复。不新增纯MPC/E30准入、smoke/mock/测试套件、恢复演习、无关回归、参数网格、哈希认证或独立benchmark。
6. 已启动的job-20261105T032354Z-w39-a/b各继承自己W38 seed1/2 ep251，恢复各自Actor/Critic、optimizer、RNG、归一化；新training/w39-a/b目录、phase_lineage、episode实际路径已9544cfe接通。沿现有作业/实际resume续接，不重新fork/reseed、不覆盖父目录、不回ep216；excluded145仅保留一次，祖先fork身份不删。
7. 每流新增1024条/4更新，到8593/33＝8448已训练＋145excluded、tail0；每256新数据更新后继续，首批7570–7825形成update30。两流各8 MPC workers、Torch1、BLAS1，共享原GPU/JAX设置。保持两计算槽，不凭显存容量改三流/worker网格；可在普通启动留一次真实利用率快照，记录实际吞吐。
8. 交互≤4h，不等待完整训练；W39共享既有28h训练时钟，截止2026-11-06T07:23:54.713909Z，resume/补传不重置或延长。各流训练后自身≤4h评价，仅本phase末次实际新update(>29)固定策略E30，合格才E60；零新update不拿父权重替代，不自动E300/600。保留原known_r25停止谓词及普通失败自然续采，不新增失败次数上限。
9. 已授权每个完成回合完整feedback-actions/planner-records/reference-packets、trajectory/prediction及必要physical/material/rollout数组上传；小记录/manifest先Git，原件Release后台，优先E30及各失败族首例。r28训练现有诊断路径和2Hz value/PANN/action_change日志已接通；本次只扩runtime的r28评价日志开关，供尚未启动的新评价进程使用，不重启训练或重建C++。完整accepted/K仍未实现，留下一明确未启动版本；旧记录不伪造。所有失败/零步/中断/截断、真实配置身份和缺失字段都保留。
10. **先推紧凑证据，随后继续研究。** 已请求诊断由新入口自动补传；未选大checkpoint、完整resume与全视频默认留服务器，用户定期导出到自己的存储盘，备份后自行统一删除。Codex不自动删除研究输出（含已上传副本），不另做用户电脑归档，不默认全量渲染/上传，不为视频重跑物理。r26旧流程保持历史原样，不重复。现行交付授权优先于旧快照cleanup=true；紧凑push不是大数据备份完成。
11. 两流计算与每job原始补传可并行，compact finish/Git及index短写串行；保留两流交接。只更新正常job/index与主计划6行，区分运行实绩、紧凑Git和原始Release状态。Git失败普通重试，不强推、重训或删除。本会话仅在Git整合上述已启动事实，未连入D57执行补传；推送工具不等于部署或上传已完成。
