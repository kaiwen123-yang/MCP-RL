# Codex：S1先稳定运行；W40两条联合主线＋一条GPU MPC任务

## 执行入口

使用AutoDL西北B区D57，SSH别名autodl-d57；项目/root/autodl-tmp/MCP-RL，环境work/envs/d57，激活source scripts/autodl_d57_env.sh。已记录RTX PRO6000 96GB级GPU、25vCPU、120GB内存；正常启动时读取实际资源，不把历史容量当空闲量。复用服务器独立联网及持久GitHub登录，凭据不进仓库/日志。

**当前事实：W39 / S5-r28两流已完成；下一任务W40_STABILITY_WITH_PARALLEL_GPU_MPC已授权但尚未实现/启动。** 以[主计划顶部6行及4/5/6节](docs/TERRAIN_MEMORY_PLAN.md)、[启动提示词](docs/S1_W40_START_PROMPT.md)、[W40计划JSON](experiments/terrain_memory/s1-go2-w40-plan.json)及最新实际job为准。不要按旧W39运行中提示重复启动。本次评审仅发布分析和计划，没有连接D57部署。

1. 开场最多5行后直接工作。先查同流tmux、实际job和未提交修改；若W40或GPU任务已经开始，续接实际作业，不重新fork/reseed或重复启动。默认读当前交接、上述章节、W39-review和直接执行源码；不再开启泛泛计划轮。
2. W39各新增1024/4至8593/33＝8448已训练＋145历史excluded，tail0；A最后ep284、Bep280。62训练回合含59自然失败（41跌倒、4躯干接触、14材料域退出）和3截断。58加载、30入条带、26进入且足载；合法单程/返程/往返0。最终自身update33 E30分别10.476/17.592s跌倒，E60阻塞。B有60s存活但未入材料，不能算通过。
3. 先用约30分钟读D57已存A9.12–9.30s、B14.63–15.10s及首材料域退出窗口，不等上传、不重跑旧物理。已知大K/目标裁剪早于首次真实材料load，且首发前已有accepted→actual关节速度误差；没有已证K/reference错配或漏限幅。旧完整accepted/K从未保存，不能伪造。若定位到更早确定接口bug，先修该具体项并写明替代证据，不叠加多机制。
4. r29主要修订按主计划4.1：原目标角与状态相关PD扭矩实现域进入12维局部反馈QP，保留非对角H/C、实际feedback scaling、已接受line-search前馈和活跃边界导数；候选和部署同一路径。不能把逐轴clip冒充新机制、执行时重复加Qu或将瞬时约束说成全hold安全。区间空显式restoration。修acceptance_mode日志分类，不改历史值。修订尚未实现，不得只把r28重命名就跑28h。
5. 保留原Go2、TM02 .02域、PD/力矩/接触/跌倒/路线和阶梯；保留r28 trunk(v,J)、r27 C2、r26因果packet/十子步A/B/TV、冻结因果PANN、658观察、2Hz cap、W38终止奖励。r25 MovingSupport/SupportPlacement等旧失败项不复活。本轮不同时改物理、奖励、步态、动作维度或PANN训练。
6. w40-a/b分别继承training/w39-a/PANN_PPO_S5-seed1/resume-state-ep00284.pt与training/w39-b/PANN_PPO_S5-seed2/resume-state-ep00280.pt，恢复各自Actor/Critic/optimizer/RNG/归一化；新training/w40-a/b目录，保留祖先fork_identity与新phase_lineage。各新增1024/4至9617/37、tail0、excluded145仅一次；首8594–8849成update34。实际接通parent_map、r29构建/日志、训练/评价/finisher后才激活主R12契约。
7. 两主线默认各12 MPC workers，各14已授权。GPU第三任务同时运行优先2×12；Torch1/BLAS1、主线原JAX精度/共享配置保留。starter选择同步job训练和评价workers；已有job普通resume保持冻结值。25vCPU还承担控制/RPC/材料/I/O，不能把线程数当有效加速或把用户报告的12核当本会话实测。
8. **用户已明确授权本轮额外第三条GPU MPC适配与测试任务，覆盖旧“两计算槽/不做GPU benchmark”限制。** 独立w40-gpu会话/目录/代码版本/时钟，默认1–2 host线程、编译最多2job；不新建第三PPO谱系。执行真实已存状态的预测/导数/候选/整步计时测试，必要时独立短闭环接入检查；不是只查论文。保持原12维PD、十子步、接触/风险/精度语义。需要新依赖用隔离环境固定版本，主线环境不升级。GPU原型未完成就诚实报告具体层次，不冒称已实现完整GPU MPC，不在主线运行中热换。
9. 交互≤4h，必要实现、正常构建、推送后直接启动完整联合训练，见真实样本推进即交接。两主线首次W40启动建立共享28h绝对期限；各自训练后≤4h，仅自身本phase新update>33的固定E30，原同case合格才E60。GPU首次启动另定≤28h绝对上限，完成即停；resume不延长任何deadline。保留原known_r25停止谓词与普通失败续采，不新增失败次数上限。除第三GPU任务明确需要的针对性数值/性能/接入测试外，不加主线纯MPC/E30准入、smoke/mock、恢复演习、无关测试套件、参数网格、hash认证或独立性能campaign。
10. 新版完整accepted/reference/K/H/C按主计划6节关键事件窗保存，每plan一次、无损二进制/压缩，不在500Hz动作行重复全矩阵；原全回合feedback/planner/reference/trajectory和已请求数组照常交付，manifest区分完整horizon覆盖和未记录。先推代码、配置、所有逐回合JSON、学习评价与结论，随后继续研究；原诊断后台queue，Git/index短写串行。当前W39上传队列只读保存原件，不重跑旧finisher、重训或重置传输时钟。
11. 大checkpoint/完整resume/全视频默认留D57，用户导出存储盘备份后手动统一删除。**不在用户电脑另做归档，不自动删除任何服务器研究输出或上传副本。** 已请求反馈/选中轨迹/参考轨迹/必要数组不能因raw或大于2MB静默过滤。compact push不是备份完成，queue启动也不是upload完成；普通整合远端提交，不强推，保留同时发生的上传更新。
