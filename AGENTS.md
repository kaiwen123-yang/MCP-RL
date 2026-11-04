# Codex：S1先形成稳定、合规的完整联合基线

## 执行入口

使用 AutoDL 西北B区 D57，SSH 别名 `autodl-d57`；项目 `/root/autodl-tmp/MCP-RL`，环境 `work/envs/d57`，激活 `source scripts/autodl_d57_env.sh`。RTX PRO 6000 96GB ×1、25 vCPU、120GB内存。复用现有环境、服务器独立联网和持久 GitHub 登录，不重新迁移。凭据不进入仓库或日志。

**当前任务：W38_SMOOTH_APPROACH_PARALLEL_JOINT / S5-r27；状态为 `implemented_pending_normal_build_not_run`。** W37/S5-r26已完成，父恢复点为本组ep216、6545条/25更新、tail0、历史excluded145。以[主计划顶部6行和第4节](docs/TERRAIN_MEMORY_PLAN.md)、[W37实际结果](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/jobs/job-20261103T075623Z.json)及[当前R12契约](experiments/terrain_memory/s1-go2-local-readiness-r12.json)为准；计划与实际实现/构建/启动分开记录。交付规则见第5节，启动词见第8节。

1. 开场最多5行，随后直接处理当前实现。先查同流tmux与最新实际结果；已有作业运行就交接进度，不重复启动或重置deadline。默认只读当前交接、第4节及直接相关源码，不全仓审查或另开计划轮次。
2. W37/S5-r26实际新增2048条/8更新，ep145–216共72个物理回合：36跌倒、34非足接触、1材料域退出、1达到采样目标截断；50回合实际加载材料、7回合机身投影进入条带、合法贯穿0。末端固定策略E30在14.3096s非足接触，E60被阻塞。累计6545/25中6400条用于已完成更新、145条历史excluded、tail0。W36/S5-r25的27回合、1169/4及4497/17父历史保留；不重复ep144迁移，不退回ep117，不重跑已关闭W35/W36/W37。
3. r27唯一控制主线：LandingVelocity与ImpactMargin共用C2空间门控g，原75.25mm approach band及物理边界不变；完整g*r残差在代价值、有限差分及子步峰值选择中保持同一定义。保留r26因果参考packet、十子步A/B、TV feedback与PD；公式和端点/转向例外见第4节。另补齐terminal物理失败的transition关账，分清accepted rollout与feedback-reference字段。不预设K裁剪、全局去平滑、频率滤波或新力残差，不恢复MovingSupport/SupportPlacement，不重新叠加r18/r19六维moving velocity项。
4. 保留原Go2、材料.02有效域、力矩/接触/跌倒/路线与评价定义；保持冻结因果PANN、PPO/658维观测及2Hz标量cap。完成上述修正与双流接线后直接完整联合训练，不设孤立模块、纯MPC或E30准入，不新增测试套件、恢复演习、无关回归、参数网格、哈希认证或独立吞吐benchmark。
5. W38启动两个独立完整PANN工程分支：stream_id为w38-a/w38-b，group均为PANN_PPO_S5，可用seed1/2区分目录，但不是从头训练的独立seed。各自从本组同一个ep216恢复Actor/Critic、optimizer、归一化及6545/25历史，保留excluded145、tail0；只在首次fork恢复后分别把探索RNG设为2026110401/2026110402。普通resume恢复本分支自己的RNG与合法tail，不重新fork。各流新增1024条/4更新，达到目标后各7569/29＝7424条已训练＋145条excluded、tail0；两流合计新增2048条，不能合并成同一策略8次更新，父历史不能重复计数。每256条新数据更新后继续，普通物理失败自然续采，不新增attempt cap。
6. 两流各用8 MPC workers、Torch1、BLAS1及既有GPU/PANN依赖，独立保存optimizer、buffer、PANN世界/历史、job身份、输出及deadline文件；训练和评价均维持每流8 workers。共用一个28h训练absolute deadline，从首条流启动起算，第二条启动或任何resume均不重置；各流训练结束后分别进入自身≤4h末端评价预算。交互≤4h，看到真实采样和普通日志正常即可离开，不等整次更新、不持续盯守。只做一次实际空间查看，源码/配置先提交再启动；记录真实任务吞吐，不承诺速度倍数或收敛。紧凑交付沿用≤6h上限、完成即结束，不等待渲染或大文件传输。
7. 复用既有tmux流水线，保留主计划第4.4节known_r25退化停止条件：首次符合即按原规则停止、保留部分结果并发布，不继续凑更新或自动E30，不加失败次数门禁。正常结束时，各流只用本分支本phase末次实际完成的新更新固定策略/归一化做nominal E30，同case合格才E60；本流零新更新不能拿父权重充当末端评价。stage_pass、合规贯穿/往返与阶梯资格分开记录；E300/600不自动启动。compact finish与Git串行执行并复用现有index锁，交接同时保留两流状态，不互相覆盖。
8. **先推紧凑证据，随后继续研究。** 自动推实际源码/配置、运行身份、全部逐回合结果JSON、学习与评价小记录、失败/零步/中断清单和结论；不挑成功回合。大checkpoint、全长视频、原始轨迹/数组默认留服务器，不自动全量渲染、排队上传或阻塞研究。具体诊断需要时再从既有数据选片/切片，按需使用现有Release入口；不为视频重跑物理。
9. **2026-11-04用户确认：r26保持已执行的旧流程；W38起按新规则，由用户定期导出到自己的存储盘，备份完成后自行统一删除。** Codex不自动删除服务器研究输出，不把GitHub紧凑交付视为大数据备份完成，也不另做用户电脑归档。继续保留有效resume/末次更新和原始证据；job目录、episode目录与共享training目录的位置写同job。旧冻结契约里的cleanup=true不能恢复旧行为；新finish只取当前契约的交付策略，训练/评价快照保持原样。GitHub推送不等于服务器已部署，交接须写清。
10. 收尾只更新正常结果与主计划6行交接，分别报告两流训练、评价、紧凑Git交付及大文件本地保留。W37既有73视频/Release保留；94b43d0已补齐紧凑记录和两个旧待传项，当前pending0，不重复补传或恢复旧全量流程。看合规贯穿/往返、侧出、真实承载/触地速度、材料域/跌倒/非足接触，不把更新次数或裸往返当成稳定任务能力。紧凑push失败保留记录、普通重试；不强推、不重训、不删除。
