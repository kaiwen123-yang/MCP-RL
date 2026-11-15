# Codex执行入口：W50有限负结果已发布；W51从正常起步成对验证

命令传递定向修正（提交 `463e91b`）：首个 D1 在真实入口发现 userdata[78:84] 未填充，原运动候选使用了零世界速度；已于实际 3.120s 停止其独立 server，并保留 execution_error/技术无效记录及全部已保存工作量。D0 原进程连续保留。修正仅使 W51 将原 mission.v/mission.w 写入既有命令槽，不启用额外 twist 成本；修正后 D1 从原 t=0 作一次有效配对尝试，详见 phase.json 的当前 jobs 和 technical_attempts。当前 tmux 为 `s1-w51-pair-command-repair`。此技术重启不作为新的物理失败或成功样本，也没有重跑 D0。

D0 实际终止结果已到达：原 body_fall 于 5.814s 触发，末 z=0.1189336264m，最大 x=0.0957089087m；实际计数为 29,070 plant 子步、311 完成 planner 调用、5,929 full rollout。2.002–4.3s 前进量为 -0.020903054m、目标 swing 仍受载约 71.18–80.19%，未形成稳定支撑推进。修正后的 D1 仍 running；截至已核验的 2.408s，原 mission 命令与 native userdata 命令一致，S 已真实作为 Q（包括 J(S)>J(B) 的搜索），未据此提前选择版本或启动原 E30。

## W51 实际执行（2026-11-15）
D8 与 r39 已实施并构建，源码提交 `df68de8`。D57 `s1-w51-pair` 正在运行 D0/r38、D1/r39，两臂均为 W44 A seed1/checkpoint-014737、原 evaluation t=0 初态。运行索引：`results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w51/phase.json`。没有新增 PPO/donor 或原 E30 资格。
用户允许 tmux 稳定后结束对话。后台会完成实际 D8、逐窗统计、全时段视频与 D57 直传发布；版本选择待完整承重/推进证据，不另造数值阈值。续接先读 phase、D0/D1、paired-results 与 task-c；不得重开已启动的 D8。后文计划约束继续有效。



更新2026-11-15（Asia/Shanghai）。结果基点`09bd55e0cb70244938ad82db7dd202a31ac70f77`。**W50/S5-r38/H已实现并完成有限负结果；W51 计划形成时为 planned_not_implemented。** 本次审查实际修复了公共runtime两处缩进错误，未启动新物理、native solve、FD、GPU积分或PPO。

先读[项目记忆](docs/TERRAIN_MEMORY_PLAN.md)、[W50最终审查](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W50-final-review/README.md)、[W51详细计划](docs/S1_W51_STABILITY_PLAN.md)、[启动提示词](docs/S1_W51_START_PROMPT.md)、[机器交接](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W50-final-review/handoff.json)。算法、实际入口、工作量与判定以W51计划为准；旧计划/旧“当前”只按形成时点阅读，不重开W50预算。

## 已确认事实与工作重点

- W50四个保存问题76full rollout，36正候选10strict改善。A237/B236预测最低z只提高.149/.016mm，所选赢家仍倒退/坍低；晚态恢复不足不等于从t=0连续刷新无效。
- 加两次midpoint replan后合计116full rollout、114candidate records、23112transition FD、4746risk FD；400bridge子步已消耗，但同SONAME错误库造成ABI失配，native物理等价有效数0。四个独立native server保存问题评分仍有效。dlmopen修复仅核库映射，没有修后物理结果。
- W50原A/B E30均未运行，八个原阶段blocked/unrun，不是八次失败；没有新完整行走视频，新PPO0/0、GPU物理0。三ZIP原件已发布。
- 最新完整原任务仍W49/r37：A/B自然观察满30s，但资格0/2、合法往返0，最大x .349667/1.020530m、末z .138718/.142818m。用户B视频的下蹲/顶起、后期低伏及非足支撑有日志支持；5.2–7s曾短暂抬高，不能说5s后始终趴地。
- W50 Task A已实测231个留存policy的reference/K/H均来自t=0 Stand [.0,.35]，每流14825/15001条实际held查询过期，最后过龄29.65s；不再只写“源码假说”。private Trot于2.002s正常切换、2Hz相位正常，但2.002–4.3s目标swing仍约88–94%受载，四足共载约86%，PD饱和0。
- 原始数组复算未发现主要raw-center、切空间、K/H混配、重复alpha或安装对象错误。被拒候选有前进但仍坍低的例子，不能断言原J错拒了正常可行步态。先分清候选不足、目标排序与同对象执行偏差。
- 公共`s1_go2_r11_runtime.py`两处dump缩进已经修复，原line499的IndentationError会阻断公共入口；只确认AST可解析，不冒称服务器入口已跑通。

## W51执行顺序

1. 核最新main、本地未提交改动、D57现有W51 job/tmux/PID/真实physical clock与CPU/显存；有效同身份工作就续接，不重复启动、不宽前缀kill、不动其他项目。默认CPU树`/root/autodl-tmp/MCP-RL-w42`、GPU树`/root/autodl-tmp/MCP-RL`，环境`work/envs/d57`、`work/envs/w41-mjwarp`，以现场为准。
2. 两条主线均从原evaluation t=0完整初始化，使用同W44 A/seed1 checkpoint-014737与归一化：D0=r38/H refresh-only，D1=r38+唯一相位感知运动候选S作替代线性化起点（拟r39）。各一次连续8s，原2s Stand→outbound保留。D0/D1不是正式A/B；不从不完整W49快照恢复plant。
3. 实现真正D8诊断入口，沿用正常plant、独立native Agent、RPC/query/hold、PolicyStream(training=False)与因果observer。`qualification_usable=false`；同步修改`s1_go2_r11_protocol.evaluate_stage()`诊断分支提前返回，不能落入E300/E600 else。不能只改外部contract、用timeout截断E30或设training_duration=8。正常完成为diagnostic_window_completed，保留原物理提前终止。
4. 不使用W50 ctypes bridge，不重跑40ms准入。直接在真实D8记录实际server库来源、状态/动作/足力/非足力和策略来源。公共runtime修复两臂共享；控制器版本的真实入口、spec/patch/prepare/runtime/policy/recorder都要接通，不能只换标签。header/ABI变化须正常重编依赖service和server。
5. D1每epoch沿原R35Freeze与10nominal先得完整全局B及原score。S复制完整B，追加`delta_q=q_IK−q_B_joint`，raw center为`c_B+delta_q`，保留原PD承重偏置、旧feedback/scale/delta；不以裸IK或projected动作替代raw center。只生成一个确定性S，无额外步高、速度、相位、权重或beta网格。
6. 相位使用原2Hz/.45/C2/.005与绝对时钟；stance锚点从该stance开始时的实际世界xy建立并跨replan保持，当前swing touchdown在开始/首次接入时固定。剩余swing从实际足位置/速度连续接续，不每20ms重设锚/端点或phase。原Gait成本参考不变，种子首节点不能跳到不同参考。身体/触地点按计划原速度命令和几何构造，限位内连续IK，无解记录S不可用，不搜索新网格。
7. S只新增一次native rollout；有效时Q=S，即使J(S)>J(B)也可用于搜索，不能因此安装。无效Q=B。只围绕Q执行一个first-valid backward与原9正alpha，不先B9再S9。沿Q实现r38当前时域重参考/同alpha K/H转移，记录真实来源；首k/K/H/μ/active一致。原alpha0直接保留完整B，Q=S时不声称positive趋零等于B；expected/ratio=null。
8. S与后续候选均按原risk/J/roundoff/strict比较原B和全局最优；只比S好不算改善。无更好者保留B，不强制安装劣解刷新年龄。D0正常epoch通常≤20full rollout，D1≤21，FD另列；双D8约842calls、最多80000plant子步，full rollout约17261量级，非W50微测试预算、不宣称同总工作量。
9. 比较0–2、2.002–4.3、4.3–8s的实际swing卸载/stance受力、足世界速度/间隙、身体z/vz/姿态、非足支撑、前进及原J分项。保存全部实际时段视频；提前物理终止不补齐、不为凑8s重跑。1N是诊断口径，不新增资格阈值，不禁止原许可短腾空。
10. D0已有有效步态且D1无实质增益，优先简单r38；D1改善正常支撑推进则选同一冻结D1实现。仅J下降/低伏挪动或两臂都差，交成对负结果并分辨候选/排序/执行问题，不临时叠加成本、步态、horizon、PPO或新求解器。
11. 有推进依据后固定一个正式机制，原A seed1/B seed2各一次E30并行，逐case原E30→E60→E300→E600。D8不算资格，不拼接或冒充E30；正式任务从原完整evaluation初态开始，明确独立attempt。30s合法材料去程/到达3m原容差/完整观察，60s合法往返/回家稳定2s，300/600各270/540s内5/10往返并完整观察。失败只blocked本case后级，不换seed追成功。
12. W44 A/B各14737/update57、checkpoint-014737、resume ep00428/ep00417全actor/critic/optimizer/norm/RNG/祖先保留，tail0/excluded145；新增PPO0/0、donor0。训练resume不是完整plant快照。Go2/TM02/PD60/5/原力矩/主应变.02/路线/接触/跌倒/.35s horizon/2ms hold/.2ms物理保持，无实机操作。

## 第三线、资源与交付

13. C独立且不阻塞CPU，默认0新GPU物理。用已有114候选的113维残差、22项norm/weights做FP64 GPU成本归约和逐项覆盖；risk须保留trunk_violation/failure等元数据，只同问题/同epoch比较。76条保存问题可作原native评分证据；38条错bridge midpoint仅代数对照，不作正确物理/控制证据。列明可由state/kinematics/context重建项和接触/solver缺失项；残差→J不能冒称完整native成本迁移，不填零补缺，不新开原episode补历史档。
14. 不重复F1/L/F2，不把canonical sort stub当修复，不用bitexact作GPU总门。C可做有数据支持的纯运动学/代数项，不能正式安装排序；不为占显存重复batch、增PPO或吞吐轮。
15. 实测25CPU/96GB余量，D0/D1各10核、C2核、约3核系统/主控/发布，尊重其他项目配额；线程库1，进程树与可写build隔离。build最多-j6先完，不再外挂第三重native solve。记录真实工作量/耗时/可得利用率，affinity不等于满载。
16. 只做本改动与真实入口必要验证，不新增全仓hash、mock、shape/import/smoke或无信息增量防御性测试。D8本身是正式路径的科学诊断，已完成W50四例和两短部署不重开。
17. **所有新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix；无固定墙钟完成时间、无到点kill、不因对话结束杀后台。** 8s诊断和30/60/300/600s评价为物理定义，原物理失败/技术退出/用户停止/单次网络timeout及有限工作量保留，无无限重试。running如实保持running。
18. 紧凑Git先推，必要原件/完整视频D57增量Release直传；短写串行合并最新main，不强推，回执丢失先查远端。核commit/资产后结束，**不在用户电脑额外下载、复制、打包、镜像或归档**。服务器仍被引用的科研原件/checkpoint/resume/视频保留，不自动删除；只用已授权服务器网络，不自动借电脑VPN/Clash/SSH转发。

最终分开报告实现、正常步态/承重/前进、原资格、材料记忆收益；给真实工作量、技术/物理/任务失败及blocked/未运行、commit/资产和准确续接点。计划发布不等于实验成功。
