# W51 Codex启动提示词

在`kaiwen123-yang/MCP-RL`最新main自主实施W51并交付真实结果，不只复述计划。结果基点`09bd55e0cb70244938ad82db7dd202a31ac70f77`。先读AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md、docs/S1_W51_STABILITY_PLAN.md，以及results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W50-final-review/的README.md、handoff.json、video-review.md、control/review.md、results/review.md。详细对象语义、预算和判据以W51计划为准。

W50是finite negative：4保存问题76rollout，36正候选10strict改善，早期赢家仍塌低且倒退加重；没有原E30、没有新完整视频。另2次replan40rollout及400bridge子步已消耗，same-SONAME错误库使物理等价证据无效。原件已全部发布，不重开W50预算。W49视频确实无正常稳定步态；W50读原件确认2.002s已切Trot且phase正常，但早期目标swing仍88–94%受载、四足共载约86%、PD未饱和；reference/K/H仍来自初始Stand。4.3s保存问题已晚于Trot约2.3s，不能以晚态恢复失败否定从起步刷新。

本次审查已修公共s1_go2_r11_runtime.py两处dump缩进，原版line499有IndentationError，修后仅确认AST可解析；这不是已运行原任务。实际使用最新修复，勿被旧worktree覆盖。W51D8直接走正常plant+独立native server+RPC，不使用W50 ctypes bridge、不再重跑40ms准入。

先核最新main/本地工作、D57已有W51 job/tmux/PID/真实physical clock/CPU显存余量，有有效同身份工作就续接，不重复启动、不宽前缀kill、不影响其他项目。默认CPU树/root/autodl-tmp/MCP-RL-w42，GPU树/root/autodl-tmp/MCP-RL，环境work/envs/d57、work/envs/w41-mjwarp，以现场为准。25CPU按10+10+2+3，线程库1，build最多-j6先结束，隔离整个进程树。

主实验是同初始状态、同W44 A/seed1 checkpoint-014737、原evaluation t=0初始化的两臂，各一次连续8s：D0=r38/H refresh-only；D1=r38加唯一相位感知运动候选S作为替代线性化起点，拟r39。保留原2s Stand→outbound、actor确定性均值/归一化、observer因果更新、物理、成本、gait和horizon。resume-state不当plant checkpoint；D0/D1不是正式A/B。需实现明确D8诊断入口，沿用正常runtime/PolicyStream(training=False)，qualification_usable=false；当前--level仅E30/E60/E300/E600，不能仅改contract或用timeout截断，也不能设training_duration=8。

D8还须在s1_go2_r11_protocol.evaluate_stage增加diagnostic分支并提前返回，不能让未知level落入E300/E600资格else。

D1每epoch先按原时序R35Freeze和10nominal得全局B，保留B完整对象/score；构造一个S：沿原绝对phase和2Hz/.45/C2/.005，stance锚点在该stance开始时建立并跨replan保持，当前swing的touchdown开始时固定，剩余段从实际足位置/速度连续接续，不每20ms重设锚/端点或从phase零起步。原Gait成本C2不变，seed第一节点不能跳到不同参考；用原剩余swing时间作唯一确定性连接。目标沿原世界命令速度/方向。按计划固定触地点构造、原足几何/地面/身体目标求连续限位IK。delta_q=q_IK−q_B_joint；S复制完整B，raw center=c_B+delta_q，保留PD承重偏置/旧feedback/delta，不能用裸IK覆盖，不做额外步高/速度/beta网格。

S只做1次native rollout，有效则可作Q，即使J(S)>J(B)也允许在Q附近搜索但不等于可安装；无效Q=B。只围绕Q运行唯一first-valid backward和原9个正alpha，r38当前时域重参考及K/H转移沿Q，记录全部来源；不先跑B9个再跑S9个。S和所有新候选最终必须按原risk/J/roundoff/strict与原B及全局最优比较，不能只比S好就装。alpha0保留原B，positive趋零不声称等于B；expected/ratio=null。D0通常≤20、D1≤21full rollout/epoch，FD另计，工作量不完全相同。所有原约束保持，无有用候选保留B、不强制刷新。

D8约合计842planner调用、80000plant子步，full rollout可达约17261量级，FD另列；这是有明确对照的真实闭环，不是W50微测试预算。比较0–2、2.002–4.3、4.3–8s的实际reference、swing卸载/stance承重、足世界速度/间隙、身体z/vz/姿态、非足受载、实际前进与J分项，保存完整8s视频与时间对齐日志。不新增1N硬门槛或禁止短腾空。

若原物理提前终止，视频只覆盖真实运行时段，不补齐、不为满8s重复；调用/子步按实际计数。

若D0已正常推进且D1无实质增益，优先简单r38；若D1明显改善正常支撑推进，选同一冻结D1版本。只有J下降/贴地挪动或两臂都差则交成对负结果，按候选生成/目标排序/执行偏差解释，不临时叠加新权重、步高、horizon、PPO或求解器。

选定后冻结一个正式机制，原A seed1/B seed2各一次E30并行，继续原E30→E60→E300→E600前级资格。D8不冒充或拼接E30；正式评价按原完整初态开始。W44各14737/update57，checkpoint-014737和resume ep00428/ep00417全状态/祖先保留，tail0/excluded145，新PPO0/0、donor0。原Go2/TM02/PD/力矩/材料域/任务/失败判据/2Hz/.45/.005/.35s不变。某case失败只blocked其后级，另一case继续，不换seed追成功。

C独立且不阻塞CPU，默认0新GPU物理：使用已有114候选的113维native残差和22项定义，做FP64 GPU批量成本归约、原J分项/risk排序对照；逐项列state/kinematics/context能重建的ResidualFn与接触/solver/缺字段项，可实现有数据支持的纯运动学/代数部分。只从native残差算J不能称完整GPU native cost迁移。停止重复F1/L/F2，不以bitexact作总门，不为占显存重复batch或新增PPO/吞吐轮。

114条分为76条保存问题与38条错误bridge派生midpoint；后38条仅可作代数对照，不是正确物理/控制证据。risk需原trunk_violation/failure等元数据，候选只能同问题/同epoch组内排序，不混排六组。

正常重编受header/ABI影响的service/server，更新真正spec/patch/prepare/runtime/policy/recorder/合同入口，勿只换版本标签；只做必要验证，不新增hash/mock/shape/import/smoke或防御测试轮。原B、S、Q与最终安装分开记录。

全部新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix，无固定墙钟完成时间、不到点kill、不因对话结束杀后台。8s/30s等是物理任务定义，保留有限工作量、原物理失败、技术退出、用户停止及单次网络timeout，无无限重试。运行中如实写running。

源码/紧凑结果先推GitHub，大原件与完整视频从D57直传Release，Git短写串行合并最新main、不强推覆盖；核验真实commit/资产后不在我的电脑额外下载/复制/打包/镜像/归档，不自动删除服务器科研输入/checkpoint/resume/视频。最终报告实现、两臂正常步态/承重/前进对照、选择理由、原资格最高结果、错误与blocked、真实工作量、远端位置及续接点，不把计划或诊断写成完整任务成功。
