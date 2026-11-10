# W43 完整启动提示词

2026-11-10。下文保留本轮原始启动授权；实际实施/运行状态见 [W43计划](S1_W43_STABILITY_PLAN.md) 和 [运行索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/index.json)。S5-r32现已实现/正常构建，真实W43父点和512/2目标已声明；已有W43作业只能按其冻结job普通resume，不能重复new-phase。GPU本轮invocation为g5-20261110T033858Z。

```text

请继续 MCP-RL 的实际研发与实验。开场最多5行后直接读取证据、核对现场、实现、正常构建、推送并启动本轮已授权工作，不再次只交一轮计划，不反复询问是否继续。
仓库：https://github.com/kaiwen123-yang/MCP-RL
服务器：autodl-d57（AutoDL西北B区D57）
CPU现有树：/root/autodl-tmp/MCP-RL-w42
GPU现有树：/root/autodl-tmp/MCP-RL
主环境：work/envs/d57，scripts/autodl_d57_env.sh
GPU隔离环境：work/envs/w41-mjwarp

一、读取最新事实，避免重复已结束任务

先读 AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md、docs/S1_W43_STABILITY_PLAN.md、本提示词，以及 results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W42-final-review/ 下的 README.md、cpu-report.md、gpu-review.md、literature-mechanisms.md。必要时读 cpu-planner-startup.json、cpu-full-plan-matrices.json、cpu-summary.json、gpu-audit.json 和 gpu-fix-validation.json。然后沿本轮真实源码调用链阅读，不重新下载全部历史Release或重做旧审计。

W42/S5-r31已经实现、构建并运行结束。A/B各新增2048条/8次更新，各累计13713/53，tail0、excluded145，末断点分别ep405/ep393。36训练回合是34个60秒采集边界和2个4秒样本目标截断，没有结果级技术中断或自然失败；但材料加载/合法去程/返程/往返全零，最大base x仅0.0245873mm。各自末次update53 E30完整30秒、material_traversal_not_completed，E60未执行。两支历史累计不能相加为独立策略样本量。

当前确实有PANN特征→PPO标量cap→MPC→时变反馈→PD联合学习，PANN参数冻结，本轮当前/next recovery特征均零。长期目标是原Go2/TM02上稳定合法材料往返及同地块跨足/返程恢复记忆的因果收益；当前任务能力、候选内完整材料历史传播和记忆收益未成立。旧r24合法去程保留，不用旧成绩替当前r31取得资格。

GPU原任务也已结束。G1仅5/32对齐，G2完成176/176但耗时152237秒；G4 native/MJX缺evaluation-start-unix，在CLI阶段exit2、无物理、MJX requests0；G5把opt.tolerance设备数组覆盖成float，首次kernel类型错误。代码修复已入仓库，实际解析器与真实MuJoCo4/Warp1.19 CPU-device类型检查通过；没有宣称CUDA或Go2闭环已通过。

二、核对现场和工作目录

精确检查tmux/PID/job/clock、CPU/GPU实际空闲量、git状态、未提交研究修改及有效checkpoint。已有本轮W43或G5续接就继续原作业，不重复启动，不按宽前缀kill，不覆盖并发结果。保留现场其他GPU训练。25vCPU/120GB RAM/96GB GPU是记录总容量，不是当前空闲量。

CPU可复用已结束且干净的 /root/autodl-tmp/MCP-RL-w42，保留r31产物和父点输出；若有活动作业/未提交修改，则建隔离W43树。GPU修复在自己的 /root/autodl-tmp/MCP-RL 应用，保留独立环境、冻结契约和旧原件。复用venv、静态资产及已完成父目录；CPU可写upstream/build不能与活动GPU共享。prepare_s1_go2_r12_support.py会重写可写upstream，不能将该目录或整个work/outputs共享为可写。环境脚本/PYTHONPATH指向正确树。复用服务器认证与独立网络，不输出凭据，不为改名复制整库到电脑。

三、实现S5-r32：优化器问题身份、λ/rate和停止状态

这是本轮CPU唯一新机制，based_on=S5-r31。已确认的链条：两流t=0第12–20次startup无native下降，λ/rate累乘到1e6；全部后续1501次非零时刻规划继承它。2.002秒outbound、正向cap和参考已生效。四个完整计划的175控制节点内Qu约1.5e-4，但H几乎为1e6I，k约1.5e-10rad，K约2e-7。全部3042次planner调用都安装策略仍几乎不动。不能继续仅归因PPO没训够、进入材料才失稳或QP一直失败。

1. 保留可复用warm trajectory/参考和时间平移；将当前问题候选/部署策略身份与数值状态分开。native当前问题重新评分incumbent，记录reference/K、生成时刻、问题版本及年龄，不拿旧成本标签冒称当前验证，也不把incumbent称为稳定性保证。

2. 新MPC问题首次求解将λ和regularization_rate同时恢复到原始1/1，保留原min/max和数值guard。同问题内部重试可调整阻尼，不每次backward重置。问题身份由真实规划状态/时间及因果参考、目标/权重/步态版本决定；t=0相同状态/参考的重复startup是同一epoch，新的物理状态或Stand→outbound是新epoch。用native planner/服务状态实现，不随意扩nuserdata96破坏GPU协议。恢复/Reset也要一致。

3. 保持native trunk可行性和(v,J)排序、qp2容差、r30完整k/K/Vx/Vxx/dV、r31原生前向与独立平滑代理。backward和native线搜索共用明确数值预算，必要的同名义点阻尼重试计入现有guard，不叠无限外循环，不绕过真实接受强装大动作。

4. 先完成当前问题允许的有限重试；确认停滞/预算耗尽后锁定该epoch停止状态，剩余相同t=0调用不得重获完整预算或继续增加λ/rate。新epoch才清停止状态并reset。写native search stagnation等真实原因，不冒称数学收敛。

5. 仅当当前问题incumbent评分有限、身份有效且符合既有部署规则时才复用；无可部署候选就保留明确失败，走原错误/终止处理，不标policy_installed、不伪造J、不加隐式fallback。不能为减少错误计数吞错。

6. 记录problem epoch/reset原因、λ/rate before/used/after、同问题迭代数、未乘alpha的dV、未正则化Qu、既有边界下可行驻点残差、k/K、native实际下降/代理下降、策略身份及接受/停滞/backward失败/复用的独立状态。alpha0后expected=1e-16是下界，不是真实未缩放梯度；高λ压小的k不能单独判收敛。

7. 每新问题从λ1开始可能丢掉有用阻尼经验；若每次仅一次成功Riccati，也可能在线始终以1求解。记录λ_used、重试和动作量检验风险，不把reset写成保证走路。不能仅在旧H减1e6求逆，冒充新的完整Riccati。

新增r32 spec/patch，沿实际build/prepare、artifact派发、runtime、记录、resume、评价和发布补齐支持，不机械全库替换历史版本。仓库目前尚无已运行r32，文件名字不等于实现。保留原Go2、TM02应变域.02、PD60/5、力矩/接触/跌倒/路线、r28 trunk、r27 C2、r26因果packet/十子步/TV、冻结PANN、658D、2Hz cap和W38奖励，r25撤回项不复活。

四、直接相关保存输入核验后完整联合实验

取W42已保存t=0序列、2.002秒首次outbound及约4秒输入，验证同epoch停止、新epochλ/rate、native接受、未缩放梯度与动作量、候选/部署身份。尽可能恢复相同warm context；缺失则明确fresh初始化或重建序列范围，不冒称等价复放。修明确接线错误，正常构建后直接联合训练。

不新增纯MPC/刚地/E30准入、smoke/mock、恢复演习、全库hash门禁、参数/worker网格，不重做旧12例QP，不无变化重跑W41/W42。r31前向改善与代理方向偏差分开记录；旧B712/B713 fresh-home replay的backward_failed不混入36训练的结果级错误。

五、两CPU流各完成512新样本、2新更新

当前父点相对于CPU树：
A：outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w42-a/PANN_PPO_S5-seed1/resume-state-ep00405.pt
B：outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w42-b/PANN_PPO_S5-seed2/resume-state-ep00393.pt
两者13713/53、tail0、excluded145。w43-a/b分别继承自身actor/critic/optimizer/normalization/RNG/fork_identity/phase_lineage，不互换、不重新fork/reseed、不丢合法tail。各+512/2，目标14225/55。这是完整本轮样本目标，不是随后自动追加2048的门槛。达到目标即末次评价和交付；仍驻留就记录负结果，不自动无限采样或临时叠机制。

先声明新phase/controller artifact/parent_map、新目录、clock及真实目标；根r12现仍是W42，不得未经声明直接--new-phase。已有本轮有效作业就按冻结身份resume。两个分支是既有工程分支，不宣传为从零独立seed或严格同父因果对照。

默认各10 MPC workers，Torch/BLAS各1，GPU第三路约2host核；C++ -j6在双流重负载前完成。若现场支持，可启动前明确11或12并同步冻结；普通resume不热改workers。用有效transition/墙钟、planner时间、GPU吞吐和实测内存峰评估，不以占满显存作成果，不开第三条12workers CPU。

完成r32实现、新phase声明、必要真实输入核验、正常构建并推送后，才使用以下命令形态：
python scripts/prepare_s1_go2_r12_support.py --build S5-r32
python scripts/start_s1_go2_r12_tmux.py --new-phase --no-wall-deadline --stream w43-a --workers 10 --session s1-w43-a --network-mode clash
python scripts/start_s1_go2_r12_tmux.py --new-phase --no-wall-deadline --stream w43-b --workers 10 --session s1-w43-b --network-mode clash

这些命令依赖本轮接线，不能对旧W42契约直接运行。有效构建复用，网络使用现场可用的服务器独立路径。两流看到真实transition增长后再交接，不能只报告tmux已创建。

六、GPU并行从G5续接，不重算G1/G2

确认旧GPU结束、修复同步且无活动同任务后，在原GPU树执行：
cd /root/autodl-tmp/MCP-RL
source scripts/autodl_d57_env.sh
python scripts/start_s1_go2_w41_gpu.py --resume-from G5

共用evaluation_command已补evaluation-start-unix；Warp在原device array内fill原声明1e-8，不退回put_model默认1e-6。类型验证仅在CPU设备完成，本次真实CUDA首次请求仍需观察。

保留原outputs/.../gpu/w41-gpu/native-evaluation-contract.json、W40-B checkpoint-009617/update37、S5-r30 native物理及任务对照，不能用W42/W43当前配置覆盖。G5是MuJoCo4/Warp1.19 FP32新预测器，不称等价迁移，不热接CPU主线。

恢复跳过G1/G2且不构造MJX Resident，新输出写continuations/g5-UTC，旧result/日志/NPZ/错误保留。执行原声明6个真实fixture、3个FD尺度与独立闭环阶段，区分first-call编译及steady耗时。结束只发布本invocation新增产物；总体保留旧G4失败，本次退出码由active_technical_failures决定，不能因历史失败重跑已完成部分。

原stock MJX G2总42.288小时、节点中位899秒，不默认再跑相同多日G4。后续GPU改造围绕真实吞吐/约束布局/host同步与批量候选，不为占显存扩大无效网格。新技术错保存首次输入/版本/traceback后修接线，不无限切后端/重试，不降低物理要求，保留其他GPU训练。

七、评价、结束与交付

各CPU达到14225/55后，用自己的末次新update固定E30；同case前级通过才E60，后续300/600按原序列。保留bidirectional_material_roundtrips_v1：E60/E300/E600合法去程、合法返程、完整往返三项分别均≥1/5/10，并满足全部原条件。长存活、raw圈数、奖励和更新完成不替代资格。

分别报告运行完整性、优化器有效性、正指令下位移/速度和支撑转换、材料接近/加载、合法去返/往返及终止原因。保留完整transition/value、rollout、更新、末次评价、首nonprogress、accepted/reference/K/H/Q、十子步/执行残差、stdout/stderr与MuJoCo warning。PANN恢复及跨足/返程收益需真实材料证据。

全部新建/续接后台deadline=null，不设4h/28h固定墙钟期限、不套到时强停。以样本/更新目标、声明评价完成、明确技术失败或用户停止结束。训练60秒采集边界、物理30/60/300/600秒、数值迭代guard、单次网络timeout与实时性测量逻辑deadline保留。无墙钟期限不等于吞错/无限重启，不新增泛泛失败次数上限。

先推实现/spec/针对性证据、真实父点、新phase和启动身份，再运行。结束先紧凑Git，已请求原件后台Release；Git/index短写串行整合CPU/GPU并发更新，核对远端，不强推、不覆盖他路结果。只传新产物，不重传已有原件；queued/published分开报告并核验远端commit和新增资产。

GitHub核验后，不在用户电脑另下载、镜像、压缩或归档。不自动删除服务器checkpoint/resume/原件、视频或上传副本，用户自行备份清理。

若生命周期修正后仍有有效梯度却缺接触探索，用本轮证据另立低维关节目标样条候选→native评分→保留实际赢家→可选局部精化，借鉴本轮七组新论文；不在本轮同时改reward/步态/动作权限或移植整套控制器。含1e6阻尼H的逆不能直接作探索协方差；均值/精化输出必须重新rollout。作者成绩不算本项目成绩。

最终交接给实际工作树、源码提交/r32产物、两流job/PID/父点/episode/样本/update/tail及真实增长、GPU本次invocation和阶段、评价、其他训练保留、Git/Release状态与准确续接位置。明确已实现/已检查/已运行/已通过的区别，不把计划或构建当稳定通行。持续完成已授权工作，不只停在建议。

```
