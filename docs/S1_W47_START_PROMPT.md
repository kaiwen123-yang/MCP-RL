# W47 Codex启动提示词

将下面整段交给具有D57与GitHub访问能力的Codex即可。它要求实施并完成已声明的有限工作，在条件成立时自主进入原任务；本文件发布本身不表示r36已构建或W47已启动。若已存在有效W47作业，按现场身份续接，不能重复启动。

```text
在 kaiwen123-yang/MCP-RL 继续W47。先同步并核对当前main、现场未提交修改及有效PID/tmux/job/clock，阅读AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md、docs/S1_W47_STABILITY_PLAN.md，以及results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W46-final-review/README.md、control-report.md、gpu-report.md、literature-and-next-direction.md、handoff.json。用户最新要求覆盖旧deadline和电脑归档规则。不要重启已经完成的W45/W46/G5。

结果基点15ee620b8a909246a12baf999c136c2be299655b。W46/r35 constant已自然结束：A E30 26.432s floor–trunk contact，B 11.638s body_fall；最大前移.626506/.396364m，合法去返/往返0，后级blocked。各固定自身W44 update57，14737/57，新PPO0/0。96个Release资产已交付。A没有实际材料加载；B明显下沉/侧倾在500Hz首完整足加载9.122s之前出现。不得把存活延长、文件交付或逻辑50Hz当作稳定或实时资格。

最重要的新证据：四个保存上下文的12次solve中60个backward完成，540个正alpha全部没有rollout/QP失败或trunk violation，却全部真实J增加。A153改善来自旧nominal feedback scale .00237137→0，而不是新正步方向；constant/load四处选中完整策略相同。B155/B158的alpha0也明显不再生nominal。保留r34 strict，不放大容差、不重新允许alpha0安装，不再扫R或增加PPO掩盖问题。

按W47计划并行推进三项：

A：先在服务器原位读取W46已存在完整candidate/policy/context/reference，核SHA，定位四个旧fixture的best nominal、alpha0与最好正alpha首分叉。核实际accepted u与内部center、时间ActionImpl/节点ActionIndex、四元数、旧scale/新K-H、PD/H-QP活动面及接触放大；输出实际ΔJ和已存dV的预测ΔJ。标准iLQR在相同动态可行nominal上alpha0本应再生，不能用“换K必然不同”替代定位。已有数组足够则0新rollout；缺字段最多16条fixed-policy诊断，有具体修复后最多再8条成对验证。旧policy用ActionImpl，alpha候选用原ActionIndex/RolloutDiscrete且不二次加alpha；post-QP固定u开环只做shadow边界检查，不重复投影改变输入，它只属离线隔离。修最小已证错误；无法定位则明确记录不足/局部敏感性，不盲目调K或同轮重写FDDP。

B：并行准备一个固定R8成功移动输入种子，来源、SHA与选段规则按计划先冻结。四fixture为B155/B158加W46 A/B首次膝力矩反转前对应的完整早期context；缺字段才按计划fallback，不按效果选点；fallback保留真实W45来源，不算最新双流覆盖。新context±偏差必须来自同origin前一accepted预测与实际状态的切空间差，缺预测不能拿零或旧delta凑数。R8提供500Hz实际x/u，不提供完整K/H。R8 phase是mission枚举，native gait phase须按源resolved cadence/GetPhase重建；只取同elapsed最近2ms节点（等距取早），阶段/相位/350ms范围不符就跳过，不搜索别段；elapsed固定回合物理起点，全部不匹配记干预不可用，不当机制失败。正确核时标、xy/yaw及速度坐标；不改当前.005m步态、材料世界或任务参数。seed K=0、H取当前incumbent有效metric，仍走原Clamp/H-QP；从真实current x0用当前native重滚，donor x仅作初差/defect诊断，不能冒称FDDP。两个家庭共用从原incumbent冻结的ubar/F/R/context，保留全部current baseline/current best/donor同J联合裁决。正式solve等A确定共同执行语义后进行，最多4×2=8solve、24条原态/记录尺度±扰动的350ms fixed-policy诊断，不扩网格。donor逐节点defect额外最多700 held transitions/7000个.2ms native子步，直接从变换后的R8状态计算，不能用本身动态可行的新rollout凑零缺口。在线唯一规则：每去/返moving epoch先current N solve，仅native_search_stagnation且source有效才追加至多一个donor N solve；current solve前冻结P0/x0/context/ref及借用H，两家庭期间不推进plant/observer；独立数值工作区，读本epoch内部stop_reason而非顶层return_reason/chosen_alpha，全部baseline联合r34裁决，实际只安装一次赢家。离线与E30同入口，不以启动时一次导入代替，不每20ms重开进程；工作量可能近两倍，不能称加速。J0/Jref、真实前移/参考比例、恢复、目标/力矩与硬条件分开报告；只变温和却驻留不算有用移动。

C：原位聚合实际1342/602、合计1944次planner call的source/stop/return/正步接纳/代价/工作量，不能把12个fixture停滞外推全程。原位核GPU已有paired/action数组，修raw投影别名、decision/hold-end时标、扩窗门槛表述和warmup/capture计数；原数据保留，修正摘要追加。GPU对象是W45旧baseline/selected，24名义scenario仅18种有效组合，并非r35新selected。没有完整native J/risk就不做等价排序，不重跑24world350ms/G5。仅在现有接口足够且需要分离首hold差异时，按计划可做单一场景三引擎各2ms、总≤30真实子步（含真实预热）的诊断；不够则报告不可识别，不扩展。

如果A/B形成同定义、对两个流最新W46代表context都有解释力的移动/恢复收益、真实时间匹配±扰动无新增原硬失败，且有限与在线入口一致，选择至多一个控制候选，冻结真实S5-r36 spec/patch/build/source SHA及runtime/records/contract一致身份，明确N/r34 strict。先推实现、有限证据和真实phase，再自主启动一次A/B evaluation-only原E30；每case自身通过才独立E60→E300→E600。若只有局部变温和、目标更低或零端点修复而没有移动收益，交付有限负结果，不重复已知E30。不得保证任务会通过，也不能只写计划就结束已获授权的实施工作。

A固定outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w44-a/PANN_PPO_S5-seed1/checkpoint-014737.pt；B对应w44-b/seed2。保留完整resume ep00428/ep00417、actor/critic/optimizer/norm/RNG/fork/lineage，tail0、excluded145。PANN权重固定、observer/material memory因果更新，新增PPO0/0；不自动做未来+512/2。用run_s1_go2_r12.py单次evaluation入口，显式attempt/evaluation-start/checkpoint/level/seed/workers/build/contract/condition nominal，不走要求本phase新增训练的job-dir。白名单、N、controller artifact、实际binary readback一致。

原Go2/TM02质量、PD60/5、每腿23.7/23.7/45.43Nm、主应变.02、非足接触/跌倒、合法路线、C2步态/.35s horizon不放宽。E30满30s合法实际材料去程、满足原远端到达容差及转向条件；E60满60s合法往返和回家稳定2s；E300/600在270/540s内5/10合法往返并完整观察。短腾空合法；body进入材料或满时长不能替代完整足迹受载/路线资格。blocked不是新失败；alpha候选不是物理回合；仅Actor不能恢复物理中途状态，不能拼接资格。

D57默认CPU树/root/autodl-tmp/MCP-RL-w42、GPU树/root/autodl-tmp/MCP-RL；SSH autodl-d57，环境work/envs/d57和work/envs/w41-mjwarp。现场核25CPU/96GB GPU实际余量及其他作业。A/B各10CPU，第三任务host总2含native确认，约3CPU留系统/主控/发布；线程库各1，全进程树affinity隔离，build-j6先完成，可写upstream/build隔离。XLA_PYTHON_CLIENT_PREALLOCATE=false、JAX_ENABLE_X64=true；按实际双流显存峰值给C准入，不假定全卡空闲，不为占满显存加第三PPO。第三有限任务结束后只保留双流即可。

所有新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix，无固定墙钟完成时刻、无到点kill。工具等待timeout不是任务deadline。保留物理阶段时长、有限数值预算和网络单次timeout；按工作完成、原结果、明确技术失败或用户停止结束，不无限重试。启动后确认真实物理clock/记录增长，不只看tmux。保留失败、技术中断和所有已生成原件，不宽前缀kill或自动清理。

本轮原件级检查在服务器已有文件上原位完成；不要重试此前被自动审批拒绝的电脑批量下载，也不补造未保存字段。紧凑Git先发，已生成且要求的原件由D57增量Release直传并核commit/URL/size/digest；短写合并最新main，不强推、不覆盖并发结果。服务器上传只用已授权学术代理，禁止电脑VPN SSH转发，凭据不进Git。GitHub核验后不在我的电脑额外下载、打包、镜像或归档；服务器checkpoint/resume/原件/视频/上传副本不自动删除。

最终给出实际commit/build/spec、A/B/C完成与限制、零端点/方向结论、donor对应的候选缺失/重滚失效/目标冲突判断、全部真实工作量、每级结果、GPU价值与资源计量、Git/Release链接和有效续接点。把计划、实现、运行、原任务资格、材料记忆收益及文件交付分别报告。
```
