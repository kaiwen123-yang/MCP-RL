# Codex：W41 CPU审查完成，执行W42接触前向实验

## 当前执行入口

先读[当前交接](docs/TERRAIN_MEMORY_PLAN.md)、[W42 CPU计划](docs/S1_W42_CPU_PLAN.md)、[完整启动提示词](docs/S1_W42_START_PROMPT.md)及[W41 CPU审查](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/README.md)。本文件和用户最新要求覆盖旧W40续接、W41启动、后台墙钟期限及电脑归档指令。历史结果/运行条件保留；W41 GPU按用户说明仍在运行，不重启或替换。

默认执行端AutoDL西北B区D57，SSH `autodl-d57`，活动GPU根目录`/root/autodl-tmp/MCP-RL`，环境`work/envs/d57`。记录容量25vCPU、120GB RAM、96GB GPU；先读现场空闲量。新CPU建议使用独立`/root/autodl-tmp/MCP-RL-w42`工作树，现有认证和只读依赖复用，凭据不进日志/仓库。

## 已确认事实

- W41-A/B均完成各自2048新样本/8更新，分别到11665/45，tail0、excluded145。A末ep387，B末ep375；末次固定E30分别14.100/28.408s跌倒，合法材料去程/返程/往返均0，E60未获前级资格。
- 130训练回合中127物理失败、3合法截断、0结果级技术中断。qp2与r30完整约束Riccati已真实运行，不再恢复W40，不重跑已完成W41。结果无error不等于所有内部warning为零。
- 原始输入复核显示刚地前向支撑明显失配：同起点近乎同目标的预测Fz约0.007–0.013N，实际114–247N；尚未新做严格同输入replay，solimp/coldstart贡献未分别量化。材料前欠推进期间动作投影未改变动作，九份plan的J/k/K复算一致。
- W42/S5-r31仍是待实现、待构建、待运行计划：原生geom/pair前向＋沿每个原生名义子步锚定的独立平滑代理A/B。不是已验证的物理修复，也不是只把旧r17开关换名。
- 长期目标是同地块跨足/返程恢复记忆改善加载、落点和重访。PANN→PPO→MPC→PD联合学习已运行；当前稳定材料通行、完整候选内恢复历史传播、记忆因果收益尚未建立。旧r24合法去程保留；G0/G1是Go2配置。

## 执行规则

1. 开场最多5行后直接工作。精确核对tmux/PID/job/clock、最新有效断点及未提交修改；已运行就继续，不重复启动，不按宽前缀kill，不覆盖研究修改或并发上传。
2. GPU后续会重新读取根r12配置并延迟import。新CPU必须独立工作树及可写源码/build/job/输出；不可直接改活动GPU根代码。`prepare_s1_go2_r12_support.py`会改写可写upstream checkout/CMake/build依赖，该目录也不能软链到GPU工作树。venv、明确只读依赖/资产/已完成父训练目录可以引用；不要共享整个work或outputs为可写目录。
3. 按W42计划实现单一r31。原geom/pair接触参数用于全部nominal/候选/accepted/reference、cost/风险及(v,J)接受；joint-limit smoothing维持现状。独立冻结平滑derivative model、每worker独立data，时间/control/userdata/callback快照一致，不在线程间临时切共享mjModel。
4. 每个原生0.2ms名义子步求soft A/B，零阶锚定原生后继；FD必须soft-perturbed减soft-base，使用MuJoCo切空间，不能减native-base或直接减四元数。十子步P/Q按同一固定目标链组合；C/D和风险峰沿native轨迹重算，不用soft FD残留sensor。r30完整k/K/value及qp2容差保留，代理导数/K不称真实plant精确Jacobian或稳定保证。
5. 保留原Go2、TM02 .02域、PD60/5、力矩/接触/跌倒/路线、r28 trunk、r27 C2、r26 packet/十子步/TV、冻结PANN/658D/2Hz cap/W38奖励。不叠奖励、步态、动作维度或物理变化；r25撤回项不复活。r17旧失败保留，用明确模型分工区别新实验，不宣传恢复阻抗必稳。
6. 对A590/A635/B712/B713及必要已有接触记录做直接服务新变更的同输入/锚定/方向/成本检查，正常构建后直接完整联合训练。不新增纯MPC/刚地/E30准入、smoke/mock、恢复演习、全库hash认证、参数网格或旧12例QP重复回放。明确接线错误要修；方法负结果记录，不无限自动调参吞错。
7. W42新phase分别继承自身W41最新有效父点；当前A ep387、B ep375，均11665/45、tail0。完整保留actor/critic/optimizer/normalization/RNG、fork_identity/phase_lineage/excluded145，不互换、不重新fork/reseed。每流新增2048/8，当前目标13713/53；更晚有效同流父点则重算。先声明parent_map/新目录/clock/产物，不能对仍是W41的配置直接--new-phase。
8. 新流建议10+10 MPC workers，Torch/BLAS各1，按真实空闲量可在启动前声明11+11或12+12；普通resume保留冻结workers。原生构建可-j6，在双流重负载前完成。正在运行GPU就是第三路任务，不开第三条12workers CPU，不重启GPU或热换后端。吞吐/资源峰值决定分配，显存占用不等于进展。
9. 每支用自己的末次新update固定E30；原同case通过才E60，300/600依原序列。`bidirectional_material_roundtrips_v1`保留：E60/E300/E600分别要求合法去程、合法返程、完整往返三项均≥1/5/10及所有原条件。训练更新、长存活和raw圈数均不能替代任务资格。
10. **所有新建/续接后台deadline=null，不设固定墙钟完成时间。** 以样本更新目标、声明评价完成、明确技术失败或用户停止结束。数值迭代guard、单次网络timeout和物理30/60/300/600s保留；不新增泛泛失败次数上限，不吞错或无限重启。
11. 保留已请求accepted/reference/K/H/Q、投影/KKT/十子步残差、planner返回、学习tail/祖先及首次技术故障；新增原生/代理模型身份、线性化点、方向误差和成本分解。正常保留stdout/stderr与MuJoCo warning计数。新phase元数据从真实父点生成，历史W41陈旧说明不改写成误用checkpoint。
12. 先推实现/spec/计划和父点，再运行，看到真实transition增长后交接。结束先紧凑Git，原件后台Release，Git/index短写串行整合并发CPU/GPU提交，不强推。queued与published分别报告；不重传已完整发布的W41 24ZIP。
13. **GitHub核验后不在用户电脑另归档，不自动删除服务器checkpoint/resume/原件/视频或上传副本。** 真实实现、数值验证、联合训练、固定评价、任务能力分别报告，不把计划或构建当通行成绩。
