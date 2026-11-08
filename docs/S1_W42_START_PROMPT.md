# W42 完整启动提示词

以下整段交给能够连接 D57 的 Codex。仓库当前交付的是 W41 两条已完成 CPU 任务的原始证据审查与 W42 计划；**S5-r31尚未实现、构建或运行**。GPU按用户说明仍在运行，现有GPU任务继续，不重新启动。本提示词替代旧W40/W41启动指令。

```text
请继续 MCP-RL 的实际研发和实验。开场最多5行后直接核对现场、实现、正常构建、推送并启动已授权任务，不再次只交一轮计划，也不反复询问是否继续。
仓库：https://github.com/kaiwen123-yang/MCP-RL 。服务器：autodl-d57。现有运行目录：/root/autodl-tmp/MCP-RL。现有环境：work/envs/d57，激活脚本 scripts/autodl_d57_env.sh。
W41两条CPU任务已经结束并完整交付；GPU仍在运行。本次推进CPU W42，不等待、停止、重启、替换或热改正在运行的GPU任务，也不干扰现场另一项GPU训练。

一、先读取当前交接与证据

阅读 AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md、docs/S1_W42_CPU_PLAN.md、docs/S1_W42_START_PROMPT.md，以及 results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W41-cpu-review/ 下的 README.md、controller-review.md、contact-crosscheck.md、history-comparison.md、literature-mechanisms.md。统计需要时读cpu-progress.json、controller-input-audit.json和raw-array-summary.json；代码按涉及调用链阅读。不重新下载全部历史Release，不重做已经完成的旧12例QP回放或泛化审计。

已确认事实：W41-A/B各新增2048条/8更新，各自累计11665条/45更新，tail=0、excluded=145。130训练回合为127物理失败、3合法截断，结果级技术中断0；合法材料去程/返程/往返均0。末次固定E30分别14.100s/28.408s跌倒。QP修复和r30完整约束Riccati已经真实运行，不能继续恢复W40或把r30当待实现。两个流不能合成一个23330样本策略，也不能把更长存活当成功。

长期目标仍是原Go2/TM02上稳定合法材料往返，并验证同一地块跨足/返程的可恢复历史怎样改善加载、落点和重访。当前PANN→2Hz标量cap PPO→全身MPC→TV反馈→PD联合链真实运行，但任务能力和记忆因果收益未建立。

二、保护现场并隔离新CPU工作目录

精确检查tmux/PID、当前CPU/GPU job、clock、git状态、有效checkpoint及未提交研究修改。已有W42有效任务就接续观察，不重复启动。不能按宽泛前缀杀进程，不能覆盖未提交工作。

运行中的W41 GPU会延迟导入bridge，并在后续阶段重新读取根r12配置。因此不要在 /root/autodl-tmp/MCP-RL 活动目录直接checkout/pull本轮CPU运行代码或改根配置。创建独立W42 worktree/checkout，建议 /root/autodl-tmp/MCP-RL-w42，从已发布审查之后的合适版本开始，保留并发远端提交。

复用原venv、已核对依赖及静态资产；新CPU的源码、build、job、clock、日志和输出独立。prepare_s1_go2_r12_support.py会重写work/terrain-memory/upstreams/mujoco_mpc-go2及其CMake/build依赖目录，这个可写upstream和build不能软链到活动GPU目录。只对明确只读的依赖/静态资产/已完成父训练目录建立引用，不把整个work或outputs共享为可写目录，不复制大原件到用户电脑归档。环境脚本/PYTHONPATH必须解析到新工作树运行代码，不能悄悄导入旧树模块。复用服务器独立联网和已有GitHub认证，不输出凭据。

三、实现一个明确的新机制：S5-r31 / W42_CONTACT_FORWARD_JOINT

本轮首选修正已量化的刚地接触前向缺口，分离原生前向与平滑局部导数，不同时叠奖励、步态、PANN或动作维度变更。

证据：A635/B712/B713起始qpos/qvel相同，首保持目标最大差约5.71e-8/5.95e-8/5.15e-6rad；当前平滑预测地面Fz约0.007–0.013N，实际约114–247N，首2ms关节速度差约0.39–0.44rad/s。这是已交付同状态、近乎同目标记录，不是已完成严格相同输入新回放；solimp/coldstart等贡献尚未分别量化。材料前9–11s已经严重欠推进，实时投影当时不改动作。九个计划J/k/K复算一致，不能再凭失败次数认定QP公式错误。

旧r17恢复geom/pair d0仍失败，负结果必须保留。W42新增内容是以下明确分工，不是把旧开关换名重试：

1. 新建独立S5-r31规格和patch，based_on=S5-r30。前向模型在MakeDifferentiable前保存、之后恢复原geom/pair接触solimp；joint-limit平滑保持现行设置。另建冻结的平滑derivative model，每worker独立mjData，不在共享mjModel上并发切换solimp。两模型的时间、control、userdata、callback/residual使用同一规划快照。
2. 所有nominal、旧warm-start重评、候选/线搜索、accepted和部署reference的前向状态、cost、trunk(v,J)和风险峰都使用原生接触前向。原材料proxy、积分器及coldstart等仍有近似，明确记录，不能称完整TM02真模型已对齐。
3. 每个0.2ms原生名义子步(xbar_j,ubar)上，从独立平滑模型求A_s/B_s；下一子步的求导点继续取原生名义后继。保持2ms固定目标、十子步、FP64、1e-6单侧FD。
4. 局部模型零阶锚定原生后继：delta_x_next=A_s*delta_x+B_s*delta_u。FD分子必须是soft(perturbed)-soft(base)，不能soft(perturbed)-native(base)再除epsilon。位置/姿态差使用现有MuJoCo切空间，不直接相减四元数。P_next=A_s*P，Q_next=A_s*Q+B_s，P0=I、Q0=0；不能首点native、后九步沿soft滚动。
5. C/D与各风险分量峰沿原生轨迹重新计算，使用对应原生峰状态及代理P/Q，不复用soft轨迹峰或FD残留sensor值。保留qp2原容差、瞬时PD移动边界及r30完整k/K/Vx/Vxx/dV。H/K明确属于近似局部模型，不宣传为plant精确Jacobian或稳定性保证。
6. 候选接受看原生前向实际(v,J)。记录实际下降、代理预测下降/比率、正则化与无改进原因，部署使用的reference/K与评估保持身份一致。不要额外clip K、随意缩反馈、放宽约束或加失败兜底动作掩盖负结果。

实现时检查build/spec、C++运行产物、Python artifact/逻辑版本派发、PANN query logging、训练恢复、评价与发布元数据中当前只接受r29/r30的明确分支，按实际调用链支持r31；不能只加JSON名字，也不要全库机械替换历史版本。当前W41配置和历史结果保持真实身份。修复新phase元数据生成，让controller_version_boundary/checkpoint_selection来自真实父点，不再残留W40/33/8593文字；历史W41实际末次checkpoint选择正确，不改写旧评价。

四、针对性核验后直接进入真实联合训练

只用已保存A590、A635、B712、B713输入及必要的已有真实接触过渡记录，验证模型分工、同一x/u的首0.2ms和完整2ms响应、锚定、soft-soft FD、十子步组合、原生风险峰与候选J。报告接触力/状态误差、实际接触分支及native held-map方向误差；代理FD与自身一致不能冒充真实模型对齐。不得把不同反馈动作后的长轨迹差称为同输入误差。

有明确新接线/运行错误就修好再继续。若方法接线正确但方向或物理收益不好，保留负结果，不用无限重试或临时叠多机制把它改成成功。正常构建后直接执行完整PANN→PPO→MPC联合实验，不新增纯MPC/刚地/E30准入、smoke/mock、恢复演习、全库hash认证或参数网格。

保持原Go2、TM02应变域.02、PD60/5、力矩/接触/跌倒/路线、r28 trunk规则、r27 C2、r26因果packet/十子步/TV、冻结PANN/658D观测/2Hz cap/W38奖励。r25撤回项不复活。不改变物理成功定义换取通过。

五、两CPU流分别继承自身W41父点

当前预期父点（前缀outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/）：
A：training/w41-a/PANN_PPO_S5-seed1/resume-state-ep00387.pt
B：training/w41-b/PANN_PPO_S5-seed2/resume-state-ep00375.pt
两者均11665/45、tail0、excluded145。若现场已有更晚有效同流父点，核实后按真实父点重算目标，不回退、不重复已完成W42。

先声明W42新phase、controller artifact、parent_map、新目录/时钟/输出；完整继承各自actor/critic/optimizer/normalization/RNG、fork_identity/phase_lineage/excluded145。每流新增2048条/8更新，当前父点下目标13713/53。不互换父点，不从一支重新fork两支，不reseed或丢合法tail。两流同一r31机制是已有分支的工程复核，不能宣传为同父随机化因果对照或两个从零seed。本轮不额外复制一轮r30训练。

现场资源为约25vCPU/120GB RAM/96GB GPU总容量，按实时可用量决策。新流建议10+10 MPC workers，Torch/BLAS各1，给现有GPU host/系统/上传留余量；若现场支持，可启动前声明11+11或12+12。启动器现只允许12/14，需为新phase补齐已声明workers范围；普通resume保持原冻结workers。原生C++构建可-j6，安排在双流重负载前。不要开第三条12workers CPU流，不修改运行中的GPU/XLA参数，不以占满显存替代吞吐。

完成上述代码、新phase和父点声明、正常构建并推送后，才在新工作树使用以下命令形态；这些不是对尚未改造的W41契约直接运行的命令：

python scripts/prepare_s1_go2_r12_support.py --build S5-r31

python scripts/start_s1_go2_r12_tmux.py \
  --new-phase --no-wall-deadline --stream w42-a --workers 10 \
  --session s1-w42-a --network-mode clash

python scripts/start_s1_go2_r12_tmux.py \
  --new-phase --no-wall-deadline --stream w42-b --workers 10 \
  --session s1-w42-b --network-mode clash

网络模式使用现场已可用的独立路径。若worker数按现场另行声明，同步命令与冻结配置。已有有效构建复用，不重复编译；新phase已启动时按自身真实job/checkpoint普通resume，不重复--new-phase。两个任务都看到真实transition增长后再交接，不只报告tmux已创建。

六、评价、结束与交付

每流达到自身新增2048/8目标后，以自己的末次新update固定评价E30；当前父点下是update53。同case原前级通过才E60，300/600按原序列。保留bidirectional_material_roundtrips_v1：E60、E300、E600分别要求合法去程、合法返程、完整往返三项均达到1、5、10次，并满足全部原条件。不能只按raw圈数、奖励或存活宣布成功。

分开报告：运行是否可靠；同输入前向偏差是否降低；正指令下实际推进与支撑转换是否恢复；合法出带/远端/返程/往返是否形成。保留成本分解、首nonprogress、四足驻留、材料加载到失稳、Kdelta/投影/饱和、完整accepted/reference/K/H/Q项、每子步残差和每次planner返回。保留完整transition/value、rollout、更新、末次评价及所请求原始记录，新增两模型身份、实际线性化点、native方向误差。stdout/stderr和MuJoCo warning计数随本轮正常采集；旧日志有就补交，无则注明，不重跑旧物理补历史。

所有新建或续接后台deadline=null，不设4h/28h等固定墙钟完成时间，不在外层套到时强停。以样本/更新目标、声明评价完成、明确技术失败或用户停止结束。物理30/60/300/600s、数值求解迭代guard和单次网络请求timeout仍保留。无deadline不等于吞错或无限重启；新技术错误保存首次完整输入，不增加泛泛失败次数上限。

先推实现/spec/计划/真实父点和启动身份，再运行；结束先推紧凑结果，已请求原件继续后台Release并记录完整交付状态。不重打包重传已经发布的W41 24ZIP。Git/index短写串行整合CPU/GPU并发远端更新，不强推、不覆盖GPU上传。queued与published分开，最终核对远端commit和raw状态。

GitHub推送核验后，不在用户电脑另做归档。不自动删除服务器checkpoint/resume/原件、视频或上传副本，用户备份后自行处理。

本轮若前向差缩小而仍不迈步，按新的原生轨迹证据另立参考/结构化初值方案；若接触搜索方向无效，再评估更合适的代理梯度；学习动作权限扩展另立阶段，不在本轮临时全改。论文机制和边界见literature-mechanisms.md，作者成绩不能当成本项目成绩。

最终交接必须给实际工作树、源码提交、S5-r31产物、两流job/PID/episode/样本/更新/tail、真实transition增长、评价/技术状态、正在运行GPU的保留情况、Git/raw交付与准确续接位置。区分本轮已执行和仍待验证事项，不把计划、构建成功或数值检查写成稳定合法通行。
```
