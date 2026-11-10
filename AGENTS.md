# Codex：W42已结束，按W43修正求解器并续接GPU失败阶段

## 当前入口与已确认事实

先读 [当前交接](docs/TERRAIN_MEMORY_PLAN.md)、[W43实施计划](docs/S1_W43_STABILITY_PLAN.md)、[完整启动提示词](docs/S1_W43_START_PROMPT.md)和[本轮审查](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W42-final-review/README.md)。本文件与用户最新要求覆盖旧W40/W41/W42启动、后台墙钟期限和电脑归档指令，历史结果身份保留。

- W42/S5-r31已实现、构建、训练及末次评价结束。A/B各+2048/8到13713/53，tail0/excluded145，末ep405/393。36训练=34个60秒采集边界+2个4秒目标截断，结果级技术错误和自然终止均0；材料加载/合法去返/往返0。两E30满30秒未完成材料通行，E60未运行。最大base x仅0.0245873mm，不能称稳定行走。
- 完整E30显示：两流t=0第12–20次startup无native下降，将λ/rate累乘到λ=1e6；全部后续1501次非零时刻规划仍用1e6。阶段/正向命令真实切换，Qu非零却被巨大阻尼压成约1.5e-10rad局部前馈。下一机制优先修数值状态生命周期，尚未证明reset后能完成任务。
- 原GPU已结束。G1仅5/32对齐，G2完成176状态探针（原175控制节点），约42.288小时；G4缺evaluation-start-unix、物理未开始；G5 opt.tolerance设备数组被覆盖float。两处接口与状态/续接/新增发布已修，真实CLI及CPU-device类型检查通过，CUDA/物理仍待验证。
- CPU S5-r32尚未实现/构建/运行，根r12仍保留W42。先实现与声明新phase，不能直接旧配置--new-phase。W43各+512/2到14225/55是本轮完整样本目标，不默认扩2048。
- 长期目标是原Go2/TM02上合法材料往返及同地块跨足/返程恢复历史的因果收益。冻结PANN→PPO 2Hz标量cap→全身MPC→TV→PD已真实运行；当前任务能力、完整候选历史传播和记忆收益未成立。旧r24合法去程保留，G0/G1是Go2配置。
- W42CPU请求原件各829项及GPU两个证据包已交付；保留服务器所有原件/断点，不重复旧目标或为补历史重跑。

## 现场与资源

默认AutoDL西北B区D57，SSH autodl-d57。CPU树 /root/autodl-tmp/MCP-RL-w42；GPU树 /root/autodl-tmp/MCP-RL。主环境 work/envs/d57；GPU隔离环境 work/envs/w41-mjwarp。记录总容量25vCPU/120GB RAM/96GB GPU，现场空闲量须实际读取。既有认证只复用，凭据不进日志/仓库。

1. 开场最多5行后直接工作。精确核对tmux/PID/job/clock、最新有效断点及git未提交修改；已有有效W43/G5任务就接续，不重复启动，不按宽前缀kill，不覆盖研究修改或并发上传。保留其他GPU训练。
2. CPU可复用已结束且干净的原CPU树，保留r31产物/父输出；若仍活动或有未提交研究修改，使用隔离W43树。GPU保留自己环境/冻结contract。新CPU可写upstream/build不能与活动GPU共享；prepare脚本会重写upstream/CMake依赖，不能软链该可写目录。venv/静态资产/完成父目录只读复用，不共享整个work/outputs为可写，不为改名归档整库。
3. 默认两CPU各10workers、Torch/BLAS各1，GPU约2host核；build-j6先于双流重负载。现场允许可新phase启动前声明11/12，普通resume保留冻结workers。不加第三条12workers CPU，不热改活动GPU/XLA/后端。以transition/墙钟、planner延迟和GPU有效批吞吐判断资源，不以占满显存为成果。

## 实施和实验约定

4. r32基于r31：warm trajectory、当前问题部署策略与λ/rate/停止状态分开。新真实状态/时间/参考或目标版本建立problem epoch，从λ1/rate1开始；相同t=0重复startup是同一epoch。内部重试不重置，问题身份不随意扩nuserdata96。明确Reset/恢复/目标变化及worker私有状态。
5. 保留native trunk(v,J)、qp2容差、r30完整k/K/value与r31 native forward/soft derivative。backward和native接受共用现有数值guard；当前epoch有限重试后停滞/耗尽就锁定停止状态，重复startup不重新领预算或增大阻尼。仅复用当前native评分有限、身份有效且符合原部署规则的incumbent；否则明确失败走既有处理，不伪装policy_installed、不加隐式fallback。
6. 记录epoch、λ/rate before/used/after、未乘alpha的dV、未正则化Qu/可行驻点残差、k/K、native下降/代理下降、策略身份/年龄及真实失败/停滞原因。alpha0后的1e-16或高阻尼tiny k不是收敛证明；不可减旧H的λ当新Riccati。每问题重置1的动作尺度和重试风险必须由新结果检验。
7. 只用W42实际t0序列、首次outbound2.002秒及约4秒输入做直接相关核验，明确warm context缺失范围。修接线、正常构建后直接真实联合训练；不新增旧QP回放、纯MPC/刚地/E30准入、smoke/mock、恢复演习、全库hash、参数/worker网格。方法负结果保留，不无限调参吞错。
8. 保留原Go2/TM02 .02域、PD60/5、力矩/接触/跌倒/路线、r28 trunk、r27 C2、r26 packet/十子步/TV、冻结PANN/658D/2Hz cap/W38奖励。r25撤回项不复活，不同时改物理/奖励/步态/动作维度。
9. w43-a/b继承各自W42 ep405/393，均13713/53、tail0、excluded145。完整actor/critic/optimizer/normalization/RNG/fork_identity/phase_lineage保留，不互换、重新fork/reseed或丢tail。先声明真实parent_map、r32产物、新目录/clock，当前各+512/2到14225/55；已有W43任务普通resume，不重复--new-phase。该目标是本轮完整实验，不自动追加2048。
10. 每流末次新update固定E30，同case前级通过才E60，300/600按原顺序。合法去程、合法返程、完整往返三项在E60/300/600分别均≥1/5/10，并满足全部原条件。存活/raw圈数/奖励/更新不能替代资格。

## GPU续接、结束和交付

11. 同步本次GPU修复并确认原任务已结束后，用原GPU树运行 python scripts/start_s1_go2_w41_gpu.py --resume-from G5。跳过已完成G1/G2，不构造旧MJX Resident；保留冻结native-evaluation-contract、W40-B update37/checkpoint009617与r30对照，不能用当前W43配置混比。新的FP32后端身份明确，不热替主线。新invocation路径保留旧错与原件。
12. G5执行原声明6fixture/3FD尺度及独立闭环；新CUDA/物理尚需实际验证。旧G4失败留总history，本次失败由active_technical_failures判定。原stock MJX每节点约899秒，不默认重做相同多日G4；后续批量工作针对真实计算瓶颈，不为占显存扩无效测试。
13. **所有新建/续接后台deadline=null，不设置固定墙钟完成期限。** 以样本/更新目标、声明评价完成、明确技术失败或用户停止结束。物理采集/评价时长、数值guard、单次网络timeout及实时性逻辑deadline保留。不吞错、不无限重启、不新增泛泛失败次数上限。
14. 先推实现/spec/针对性证据与真实父点，再运行；观测两CPU真实transition增长和GPU本次阶段后交接。保留既有全部请求原件、新数值状态、首次技术错误输入、stdout/stderr和warning。运行、学习、优化有效性、运动/材料资格与记忆收益分开报告。
15. 结束先紧凑Git，原件后台Release；Git/index短写串行合并CPU/GPU并发远端更新，不强推、不覆盖新结果。只上传新增产物，不重传已发布包；queued/published分开核验。
16. **GitHub核验后不在用户电脑另做归档，不自动删除服务器checkpoint/resume/原件、视频或上传副本。** 用户自行备份清理。最终给源码/实际产物、job/PID/父点/样本/update/tail、评价、GPU invocation、Git/Release及准确续接位置，不把计划或构建当稳定通行。
