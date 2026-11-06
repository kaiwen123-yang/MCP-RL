# Codex：S1先稳定运行；W40两条联合主线＋一条GPU MPC任务

## 执行入口

使用AutoDL西北B区D57，SSH别名autodl-d57；项目/root/autodl-tmp/MCP-RL，环境work/envs/d57，激活source scripts/autodl_d57_env.sh。已记录RTX PRO6000 96GB级GPU、25vCPU、120GB内存；正常启动时读取实际资源，不把历史容量当空闲量。复用服务器独立联网及持久GitHub登录，凭据不进仓库/日志。

**当前事实：W39 / S5-r28两流已完成，976项原始诊断已发布；W40已授权、尚未实现/启动。** 本次原件复核已纠正上一版r29公式；以[主计划顶部6行及4/5/6节](docs/TERRAIN_MEMORY_PLAN.md)、[启动提示词](docs/S1_W40_START_PROMPT.md)、[W40计划v2](experiments/terrain_memory/s1-go2-w40-plan.json)、[新审查](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-raw-review/README.md)及实际job为准。没有连接D57部署，不重复旧W39或启动旧Qux中心方案。

1. 开场最多5行后直接工作。先查同流tmux、实际job和未提交修改；若W40或GPU任务已经开始，续接实际作业，不重新fork/reseed或重复启动。默认读当前交接、上述章节、W39-raw-review/README.md与r29-math-audit.md及直接执行源码；不再重复30分钟旧窗口提取或开启泛泛计划轮。
2. W39各新增1024/4至8593/33＝8448已训练＋145历史excluded，tail0；A最后ep284、Bep280。62训练回合含59自然失败（41跌倒、4躯干接触、14材料域退出）和3截断。58加载、30入条带、26进入且足载；合法单程/返程/往返0。最终自身update33 E30分别10.476/17.592s跌倒，E60阻塞。B有60s存活但未入材料，不能算通过。
3. 原件已确认58个loaded训练回合中57个先有大反馈/执行PD饱和；B E30在15.264s、load前有48.914rad原始反馈峰，实际动作受原clip限制。两E30全部14034执行动作/reference/packet身份一致；accepted→actual是闭环分歧，不能叫同输入开环残差。PANN零值主要是1820/2048 query全外域，另140未加载历史、2个短脉冲漏采。B272是有前进命令却进入异常局部平衡。旧完整accepted/K/H未存，A260无dense展开，不能伪造或重跑旧物理补齐。
4. **r29改为旧策略输出c=原target_clip(u_ff+s*K_old*δx)之后的非对角H度量投影**：min .5(u−c)'H(u−c)，约束为原target∩当前PD区间。c可行直接返回c；H取同次成功backward真正Quu_reg。旧Qux中心公式撤回，不半改backward/value或声称完整constrained DDP。候选/nominal搜索/部署共享纯求值；action candidate显式s=1，nominal搜索才用其s，alpha只进前馈一次。H生命周期、初始legacy_seed_no_metric、首次真实动作前metric有效、固定轴消元、空区间最小违反、KKT状态、ratio不可用时mu/rate处理按数学审查落实。保持2ms hold/十子步，不把瞬时域说成全hold安全；开环FD不能对扰动重新跑policy。修日志分类及每次planner调用返回原因/policy复用年龄，不改历史原值或把r28重命名。
5. 保留原Go2、TM02 .02域、PD/力矩/接触/跌倒/路线和阶梯；保留r28 trunk(v,J)、r27 C2、r26因果packet/十子步A/B/TV、冻结因果PANN、658观察、2Hz cap、W38终止奖励。r25 MovingSupport/SupportPlacement等旧失败项不复活。本轮不同时改物理、奖励、步态、动作维度或PANN训练。
6. w40-a/b分别继承training/w39-a/PANN_PPO_S5-seed1/resume-state-ep00284.pt与training/w39-b/PANN_PPO_S5-seed2/resume-state-ep00280.pt，恢复各自Actor/Critic/optimizer/RNG及现有固定归一化约定，不新加观测归一化器；新training/w40-a/b目录，保留祖先fork_identity与新phase_lineage。各新增1024/4至9617/37、tail0、excluded145仅一次；首8594–8849成update34。实际接通parent_map、r29构建/日志、训练/评价/finisher后才激活主R12契约。
7. 两主线默认各12 MPC workers，各14已授权。GPU第三任务同时运行优先2×12；Torch1/BLAS1、主线原JAX精度/共享配置保留。starter选择同步job训练和评价workers；已有job普通resume保持冻结值。25vCPU还承担控制/RPC/材料/I/O，不能把线程数当有效加速或把用户报告的12核当本会话实测。
8. **第三条GPU MPC适配与测试已获用户明确授权，覆盖旧“两计算槽/不做GPU benchmark”限制。** 独立w40-gpu会话/目录/版本/时钟，默认1–2 host线程、编译最多2job，不新建第三PPO谱系。实际做已存状态的预测/导数/候选/整步测试及必要短接入，不只查论文。同版本对齐：r28对r28或此次H投影r29对r29；保留原12维PD、十子步、MakeDifferentiable/coldstart/接触/风险/FD精度。现有q/v/mocap/96D userdata可建fixture，task phase和初始policy/warm-start须同一声明重建；未存旧K/H不能冒充历史整planner重放。依赖隔离固定版本，主线不升级、不热换，原型按真实完成层次报告。
9. 交互≤4h，必要实现、正常构建、推送后直接启动完整联合训练，见真实样本推进即交接。两主线首次W40启动建立共享28h绝对期限；各自训练后≤4h，仅自身本phase新update>33的固定E30，原同case合格才E60。GPU首次启动另定≤28h绝对上限，完成即停；resume不延长任何deadline。保留原known_r25停止谓词与普通失败续采，不新增失败次数上限。除第三GPU任务明确需要的针对性数值/性能/接入测试外，不加主线纯MPC/E30准入、smoke/mock、恢复演习、无关测试套件、参数网格、hash认证或独立性能campaign。
10. 新版完整accepted/reference/K/H及基策略Q_u/Q_xu诊断按主计划6节事件窗保存，每plan一次，无损压缩；动作行记旧clip后c、新投影delta、三种scale/alpha、实际动作身份、活跃界/KKT和插值，不把原K当在线有效Jacobian。每次planner调用均记return reason/policy_updated/reused age，防止只见成功计划。补query曝光、短脉冲、触地vz/压入及未施加候选力时钟诊断，不更改actor/物理。原完整feedback/planner/reference/trajectory及请求数组照常交付。先推紧凑结果再继续研究，新诊断后台传，Git/index短写串行；W39已976/976完成，不重启旧queue/finisher、重训或重置时钟。
11. 大checkpoint/完整resume/全视频默认留D57，用户导出存储盘备份后手动统一删除。**不在用户电脑另做归档，不自动删除任何服务器研究输出或上传副本。** 已请求反馈/选中轨迹/参考轨迹/必要数组不能因raw或大于2MB静默过滤。compact push不是备份完成，queue启动也不是upload完成；普通整合远端提交，不强推，保留同时发生的上传更新。
