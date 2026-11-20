# Codex 启动提示词：W56 完整资格、材料因果与有效并行

将下面完整内容交给执行端 Codex。先读取包含本文件的最新 main；已有原任务按实际进度续接。此文件是计划，不代表 W56 已执行或 E600 已通过。

~~~text
继续 kaiwen123-yang/MCP-RL 的 S1，执行 docs/S1_W56_STABILITY_PLAN.md。直接完成实现、必要验证、原任务、同轮有证据的修订和 GitHub 交付，不停在重新写计划。常规实现、独立工作区/构建、续接原后台和按计划新身份实验已授权，不重复询问。

先读根 AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md、docs/S1_W56_STABILITY_PLAN.md、results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/current.json，以及 reviews/W55-review/ 的 README.md、handoff.json、control.md、evidence.md、pann.md、material-interface.md、literature.md。审查结果基点 0eb39b13932c18d8e52ff305a3c35477a4510ab7；执行时再核最新 main、本地改动及服务器具体 job/tmux/PID/物理时钟/实际配额。已有同身份有效任务只续接，不重复启动，不覆盖用户改动，不宽前缀kill，不动其他项目。

当前事实：F/r55 A、B各自原E30/E60/E300通过，合计6个已结束阶段全通过。A/E300原数组核实231.942s完成第5圈并稳定到300s；B/E300独立result/analysis已补交，Release视频与20个ZIP也已发布；本审查核独立JSON/视频，未独立重算B数组。最新审查快照2026-11-20 16:33:31北京时间下，A/B E600分别到153.502/113.502物理秒，仍running，不代表现在实时状态或最终通过。A/B tmux分别s1-w55-f-a-r55-e600 / s1-w55-f-b-r55-e600，child PID快照500828/502596，workers=1。先查是否仍运行或已自然结束；先续接/收集已有任务，不为W56重新跑r55前序或热换模型。

W55正确台账是12次已结束、8通过、4非通过、2个E600在途；983.930模拟秒、4,919,650物理步。失败包括r53 A/B回家站立后WBC中断和N54/N56各一次未贯穿，不是r55当前六次仍大量失败。r53临终已经漂移、下沉和非足承重，max iterations不等于纯数值故障。r55修复了站立阶段错误COM力矩臂，原WBC硬界、MPC成本和评价没有放宽。不要重复旧r52 Jdotv探针或重写成功步行内核。

并行推进，最多两条重机器人物理加一条真实材料/效率计算线；谁就绪谁进入自己的原任务，不等总报告、上传或其他线全部完成。已有两条r55 E600优先完成原540s内10次合法材料往返及600s全观察，保留原PD60/5、2ms hold、5kHz plant、材料/接触/输入/力矩/摩擦硬界、checkpoint/normalizer/初态及各自资格链。新失效先保全最早偏差/原矩阵/初猜，在本轮修具体机制，新身份取得自己的原前序，不借资格、不原样重复负版本、不换seed追成功。需要更长阶段才逐级，不把所有消融自动铺成四级矩阵。

F后续做真正同actor的末12维消融：新增显式模式，只把送actor的658D观察末12维恢复特征置零，保留106:646的运动/力历史、同actor和normalizer、observer记录、真实plant历史与初态。实际执行反事实cap，不把现有影子pann-action-dependence当闭环。A/B分别配对，默认新身份各原E30→E60覆盖一次完整往返；只有多圈问题确需才扩E300/E600。原r55已完成同身份基线直接复用，不重跑。RAW另一套actor不是纯历史消融；末12维零也不等于删除所有隐式记忆。

C复用W55成果，同轮完成材料接线与有效计算。原PANN网络权重冻结但world-cell e/q持续更新；当前12恢复特征主要只能改2Hz scalar cap，F/r55力MPC没有候选材料状态。先从r55去程/跨足/返程原件建立query及实际复踩的world-cell、历史年龄、残余状态、面积匹配和动作影响。r55旧query-exposure guard未记录完整查询，按真实actor先于同tick replan顺序重建，不能伪称原先已直接保存完整obs。

复用s1_w55_c_observer.py的闭合10×0.2ms窗口，在新observer模式因果发布足边界/速度/时间戳，不读true q，不将端点插值当观测，不热改E600。候选独立材料传播先离线实现：每候选从同一观测快照clone一次，预建未来footprint并集，按自己的加载及未受载恢复推进e/q，原面积/ownership和共同sole boundary一致。Ground.force_inverse临时反演不提交q；嵌套clone会重置lazy未来记录，两个陷阱必须处理。

首次在线只接一个明确组合和控制落点：新F backend用同一只读状态、同footprint/面积/ownership/dt下delta(F_target)-delta(F_measured)的相容逆解，替代WBC固定1/k法向增量；z_ref=z_now−(delta_target−delta_measured)，零力差零偏移，不双算绝对变形。保留r55时序、落点、MPC成本/矩阵/摩擦与力界。准确称“历史相关法向执行参考”，不冒充完整材料耦合MPC。若窗口和参考同时变，效果归于组合，不扩全部组合矩阵。

只做改动必要的同输入/时序/clone/边界验证。保存状态上确认正式WBC参考和输出受模式影响、runtime消费已接通后，本轮就新身份同A/seed1直接原E30，成功后继续自己的资格及B，不等下次实现许可。只有影子候选变化不算在线接入。域外/数值失败显式记录，不静默裁切或切回旧参考；显式回退须进合同并单列。如果无可测作用，完成最小消融并报告机会/接口限制，不软化原plant造收益。

W55材料新候选仅两参数校准，两个独立holdout恢复变差，正式任务继续原B ideal，不直接采用。新的定向候选需r55决策敏感区间独立误差支持，分清采样、桥接和本构，保留完整事件族/episode保留及F1–F6迁移检查。新PPO samples/updates/donor仍0。oracle只作离线诊断，禁止下发。

N是次线，不能阻塞F。r56跑满30s，最远x=.894534m，仍未贯穿.8–2.2m材料。成组解决native候选半空间投影的数值假不可行/映射后原约束复验，以及完整350ms可实现推进与支撑交接候选；单延长20ms前缀仍倒退6.50mm，不够。slot8是275行可行见证，265行是slot4/7；稍后又真不可行。保留真不可行、原J和同选择器，记录分项/首约束/安装来源。warm-start等若影响输出须入可复制状态并在执行/native/FD一致。复用call809/856必要样例和正常构建，空槽就绪后新身份同A/seed1、virgin/t0、workers10直接原E30；新明确首因可同轮继续，不固定两次失败停机、不原样重跑。

25核预算：两F各≤4、C/离线/构建共享≤14、系统/其他≥3；F+N时4+10+8+3。数学线程1、构建总-j6计入配额，独立build/输出/进程树，F冻结workers1不改。当前材料GPU在已测9历史负载慢于单核CPU33.77倍，改四线程block/串行差分/重复分配/小批组织，批量独立地块和候选，不复制数据占96GB。优化只进新build，不能改在途E600动态import的文件。分报单实例延迟、batch吞吐、编译/传输/热段/核时/RSS/显存/真实重叠；吞吐除batch不是实时，未移植WBC/物理不能称全GPU。

全部新后台deadline=null、wall_deadline_enabled=false、kill_on_conversation_end=false，省略--deadline-unix，无固定墙钟截止、到点kill或对话结束kill。原物理长度/失败条件、数值或资源错误和用户停止有效，不复制历史W39的28h预算。保留W44完整checkpoint/resume及被引用服务器原件。

核对B300新交付回执并补齐N56任务原件；修finish_s1_w55_f_stage.py:navigation，只更新W55自身实际结果，不让旧job覆盖W56计划状态；从per-case正式结果生成统一派生汇总，同步AGENTS/README/项目记忆/current/next_authorized_plan，分开结束/通过/失败/在途/未运行/已发布，不改历史attempt事实。W56新结果独立w56/index/phase；已有r55 E600仍归W55。现有s1_w56_n*是W55的N/r56版本，不覆盖，取未占用新版本号。

紧凑代码与结果及时串行整合最新main推GitHub，不强推；必要原件和全视频D57直接增量Release，回执不明先核name/size/digest再续传。GitHub交付后不在我的电脑额外下载、复制、镜像、打包或归档，不自动借电脑VPN/Clash/SSH转发，不自动删被引用服务器原件。

最终分别交付：r55最高原资格及稳定范围；N首因/进展；材料估计/采样/接口/真实闭环增量作用；GPU真实收益；失败/未运行/在途清单；有效并行；GitHub提交与具体续接点。能完成的工作推进到结果，有任务仍运行就保留并准确交接，不编造最终成功或强行收尾。
~~~

