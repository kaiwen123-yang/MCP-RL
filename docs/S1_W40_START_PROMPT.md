# W40 Codex 启动提示词：原始数据复核后更正版

本提示词替代上一版 Qux 中心 QP 方案。W39 原始诊断已全部上传；r29 / GPU 尚未实现、构建或启动。复制下方全文执行。已开始的实际作业优先核对，不重复启动或热改。

```text
继续MPC+RL项目，在D57执行原始数据复核后更正的W40：两条稳定性联合主线＋第三GPU MPC适配测试任务。先同步GitHub main并保留未提交修改，读AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md顶部6行及4/5/6节、experiments/terrain_memory/s1-go2-w40-plan.json，以及results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-raw-review/README.md、r29-math-audit.md；GPU同时读gpu-replay-audit.json和docs/S1_GPU_MPC_FEASIBILITY.md。开场最多5行后直接工作，不再开泛泛计划轮。

先查autodl-d57:/root/autodl-tmp/MCP-RL的tmux、实际job和未提交修改。W39已完成，两流各8593/33、tail0、excluded145；A最后ep284、B最后ep280；末次E30于10.476/17.592s跌倒。原诊断A492/B484共976项已发布，不能再要求等待旧上传、重启旧queue/finisher或重跑旧物理。若W40/w40-gpu已实际开始，按真实版本/目录/时钟续接，不重新fork/reseed或重复启动。

采用原始审查的新事实：58个loaded训练回合中57个先有大反馈/执行PD饱和；B E30在首次load前15.264s有48.914rad原始反馈峰。实际动作都经原clip，两E30的动作/reference/packet身份一致。accepted→actual是闭环轨迹分歧，不是同输入开环模型残差；不能把某个旧窗口第一次误差叫首因。PANN零值主要来自query空间暴露不足及2例2ms漏短脉冲，原始2048×658数据交接无错位；B272有持续命令却进入异常局部平衡。旧完整accepted/K/H与A260 dense没有保存，不可伪造或重跑来冒充。已完成原件复核，不再固定重复30分钟提取。

主修订S5-r29改成明确的策略投影：先完全按旧策略得到c=原target_clip(u_ff+s*K_old*delta_x)，再解min .5*(u-c)'H*(u-c)，约束为原target范围与当前q+(tau±+5*qdot)/60区间的交集。c可行直接原样返回。H使用同次成功backward实际的正定非对角Quu_reg；不是backward.Quu或自由子块Cholesky，也不由更新后的mu重建。上一版线性项s*Q_ux*delta_x的中心QP已撤回，不能继续实施。新机制不是完整constrained DDP，本轮不半改backward/value，不重建旧K，不额外加一次Qu或取消已选alpha。

按r29-math-audit落实H在Allocate/Reset/CopyFrom/候选/winner/previous_policy/端点的完整生命周期；metric SPD在原backward正则化循环内处理，使k/K/H同版本同mu。保留旧u/reference/K与四元数插值，H用同网格正定线性凸插值。三个入口共享纯evaluator：nominal FeedbackRollouts使用搜索s，ActionRollouts必须显式s=1，部署安装s=1；alpha只加在前馈一次。不能读取候选上残留的feedback_scaling当action rollout的s。日志分开nominal_feedback_scale、action_step_alpha、installed_feedback_scaling。

第一次backward前seed无H，明确legacy_seed_no_metric，只供原seed nominal rollout/read-only probe；不默默塞单位阵。既有startup迭代数不增加，首次实际持有动作推进物理前必须有有效metric；全部startup backward失败就记录初始化技术失败，不能r28 seed继续却标r29。null-state nominal读数保持旧语义；诊断probe不能改变QP warm-start、held target或下一次执行动作。

空交集在原target硬界内选最小原始PD违反端点，其余轴正常投影，明确restoration及违反Nm，原force clip保留。等界/固定轴先消元，保留H_FA*u_A交叉项；不要把lower>=upper传给现有mju_boxQP，所有轴固定时直接组装。记录primal/KKT残差及显式失败状态，不能把非负nfree当严格收敛或默默退化仍报成功。投影改变选中候选时旧expected/surprise置null；前后trunk可行且实际J下降则同时保持mu与regularization_rate，J不降沿原增大规则；trunk不可行仍按原violation分支。不能以surprise=0代表ratio不可用而不断升mu。

分别记录RPC前double投影解u_star与实际float32往返后held target u_held、delta_u及同一采样状态下两者的PD/约束残差。QP状态按u_star的primal/KKT判定，执行统计仍用u_held；若u_held=double(float32(u_star))且delta_tau=60*delta_u能解释边界越出，单列serialization_boundary_crossing，不当求解失败，也不从原严格饱和计数中删除。此解释只限持有开始同状态，后续子步饱和另记；不改RPC协议/物理阈值。

原2ms目标hold、10个.2ms PD/力矩clip保持，不每子步重解QP，不宣称瞬时区间保证全hold/材料/稳定。开环A/B在给定投影后的nominal u处对独立固定target的F(x,u)求导，状态扰动时不能再跑policy/QP混入B*Keff；保存原单边FD在PD折点的分支，不同时改差分或平滑物理。保留Go2、TM02 .02域、原限幅/接触/跌倒/路线与阶梯、r28(v,J)、r27 C2、r26因果packet/PQ、冻结PANN、658输入、2Hz cap及W38奖励，不叠加新步态/奖励/观察归一化/PANN训练/物理模型或r25旧载荷分配。

补新版本事件窗完整accepted/reference/K/H与子步证据，每plan一次无损保存；原基策略Qu/Qxu只作诊断，不混成投影线性项。动作行记query_kind/applied_action_id、metric来源、旧clip后center、新投影delta、原rawK、scale/alpha、插值、bounds/active/KKT/status/原PD，不能把参考K冒充投影有效Jacobian。每次planner调用必须记录call id、return reason、policy_updated、installed plan及reuse age；W39的8次无新plan都在B276，不是漏上传。按既有事件ring保留窗口前更早引用plan，终止后不补跑物理。补query曝光/短脉冲离线标签、负载边界/材料/physical时钟和候选力是否实际施加、非零命令下的非进展姿态诊断；不把真值喂actor。

接通r29正常源码/构建、parent_map、独立training目录、祖先fork_identity与新phase_lineage、runtime日志、训练/评价/finisher和W40共享时钟后再激活R12运行契约，不能只改名字启动r28。w40-a恢复training/w39-a/PANN_PPO_S5-seed1/resume-state-ep00284.pt；w40-b恢复training/w39-b/PANN_PPO_S5-seed2/resume-state-ep00280.pt。恢复各自Actor/Critic/optimizer/RNG及现有固定归一化约定、8593/33和excluded145一次，不重新seed、合并或回共同父点。各新增1024/4至9617/37，首8594–8849成update34，每256正常更新后继续。

两主线默认各12 MPC workers，各14已经授权。GPU并行时优先2×12，Torch1/BLAS1，主线JAX精度/共享配置保留。完成必要实现、正常构建和代码配置推送后，直接启动完整PANN→PPO→MPC→Go2/TM02续训，真实样本/日志推进即可交接；不加主线纯MPC/E30准入、smoke/mock、恢复演习、参数/worker网格或无关测试。保留原known_r25停止谓词，普通失败保存续采，不新增失败次数上限。

同时启动独立w40-gpu工程任务，用户已授权第三并行及必要GPU数值/性能/接入测试，旧两计算槽/不做benchmark限制不阻止它。实际做批量预测/有限差分/候选rollout，优先MJX-JAX/MPX可复用结构与FP64，不只提交调研。GPUhost默认1–2CPU线程、编译最多2job，不再开大CPU参考池；依赖隔离固定版本，不升级主线MuJoCo3.2.3 ABI。正常启动读实际资源，给PANN/PPO留显存，目标是吞吐而非填满96GB。

GPU从已存真实状态建明示fixture：q/v/time/target/mocap/96D userdata与applied settings齐全，但整planner还需同一task phase/transition历史与初始policy/warm-start；缺失旧K/H不能冒充历史复现。MJB不自动含自定义callback/预测MakeDifferentiable/coldstart。严格r28对r28或此次H投影r29对r29；先一步/十子步，再开环A/B与风险C/D、固定候选及整次planner。保留原模型/12维PD/36切空间/足序/精度/单边FD/风险最早峰值和原patch，真实CPU TM02不变，不喂隐藏材料真值。导出内部六项计时，报告JIT/传输/同步/稳态p50-p95/误差/内存与主线吞吐影响。必要且对齐后可独立冻结策略短闭环≤原E30，不算资格，不热换A/B，不自动升阶；只完成batch层就如实交付该层。

交互≤4h。两主线第一次W40实际启动建立共享28h绝对期限，第二流/resume不重置；各自训练后≤4h，只评价本phase本流新update>33固定E30，原同case合格才E60，零新update不用父权重凑数，不自动E300/600。GPU首次实际开始另设≤28h上限，完成即停，resume不延长，不串接r30。进展同时看合法去返/往返、支撑与材料load后存活、触地冲击、nonprogress、投影/KKT/复用policy等；少跌倒/少饱和或GPU快不代表S1通过。

新轮先推代码/配置/全部逐回合JSON/学习评价/结论，再后台上传已请求反馈/选中与参考/执行轨迹、新rollout及必要physical/material数组并继续研究。Git/index短写串行、保留远端并发更新、普通整合不强推。W39发布完成不再重启旧队列。大checkpoint/完整resume/全视频默认留D57，用户备份存储盘后自行删除；不自动删研究输出或上传副本，不在用户电脑另做归档。交接准确区分实现、构建、启动、真实采样、正式能力与GPU工程结果。
```
