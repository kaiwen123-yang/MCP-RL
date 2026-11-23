> 2026-11-23状态更新：W57已实际执行，原文为形成时合同，继续约束现有冻结作业；旧“未开始”是历史状态。最新事实见[本轮审查](S1_W57_REVIEW_20261123.md)，续接和后继见[W58完整计划](S1_W58_RESEARCH_PLAN.md)及[新启动提示词](S1_W58_START_PROMPT.md)。不要按旧状态重复启动W57。

# Codex启动提示词：W57模块原型、受控学习与首次联合验证

请直接执行仓库最新main上的W57研究计划（含2026-11-22补充），目标是从已稳定的r55/r60基线推进到可归因的模块与联合改进。先读取 `AGENTS.md`、`docs/S1_W57_RESEARCH_PLAN.md`、`docs/S1_W57_REVIEW_20261122.md`、`docs/S1_W57_LITERATURE_UPDATE_20261122.md`、`results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/current.json`、`reviews/W56-review/README.md`及其control/evidence/module-innovation-roadmap，然后续写 `docs/S1_DAILY_WORKLOG.md`。路径不带根目录的review均位于上述R12目录。不要另开W58重复本轮规划。

原实验基点为3590646bc15a59903c9561381368d985963dfe9e，本次复核的原计划main为640be3d6ed74e2f1d40751e96218c6a689d82ea9。r55和r60 A/B均已通过原E600；r57 A/B通过E60；r58两条E300是计算停滞后授权终止；N59原E30未贯穿。当前稳定F是接触力QP MPC＋全关节逆动力学WBC＋原PD，不是旧N线全身iLQG。不要把旧失败率当当前F基线成功率，不原样重跑四条合格E600。先核最新main、已有修改和D57确切job/tmux/PID/物理时钟，已有有效作业则续接，不重复启动，不在运行中静默改其合同。

本轮可并行做三条逻辑线，最多两条重物理：

1. **M材料估计**：建立自身因果load/release事件标记，不能读log=False下未更新的ever_loaded，更不能在线读真plant q；以完整事件族/独立历史保留组比较原前推observer与物理一致的测力校正observer，区分模型、状态和权限误差。新PANN仅在保留数据支持时单独校准，不能因参数冻结就先换模型。
2. **P预测控制**：先将r55/r60约2mm、约100ms前足再离地与原计划/测力/交接/WBC对齐定位，同时补步周期CV、接触时序、冲量/滑移、机械功等质量指标；把材料candidate rollout真正接入未来接触/载荷预测和实际选择。先在保存事件上比较25ms候选单步与细分材料积分的误差，不能以inverse→forward力残差代替预测准确性。候选按同一完整模型、原目标与硬界比较。已审查r55 E300中精确足印交集>0的同圈跨足query-cell行，其query→next-load均在0.35s内；该结论不外推全部查询或r60。先接通当前候选，再按source-load→reload的真实后果决定低频事件层，不能混用两个时距。做私有候选卸载历史清零对照，不能重置真plant、主observer或当前接触足印完整状态。
3. **R学习接口**：旧W44 A/B速度actor保持deterministic eval，新建独立timing head，首动作仅下一对角支撑交接请求残差±50ms；原全部support gates保留，预测与实际调度同源。首条采样前固定独立schema/归一化，明确每个恢复量的查询足印、目标事件、发布时间、三个时距与自由恢复假设，区分材料外/virgin/充分恢复/未发布；需要额外mask输入时公开新schema，不能静默填零凑59D。事件dt reward/GAE、请求/接受/执行/mask全记录。不要把旧stream.training=True来训练新head。一个共享新head、一个学习seed、2collector各128同版本收集，首节点8×256=2048，4epochs/minibatch64；有效动作和mask单列，旧actor更新0与新head更新分列。

2048后按主计划四项条件判断同身份累计4096/8192：动作真有权限、训练任务有历史/后果信号、reward/GAE/账本有效、M/P/任务/schema/reward固定。不用保留机制评估挑checkpoint或追训；每档不等用户。若要回答新head是否利用材料预测，同轮从共同初始化独立训练同权限、同网络、同任务和同预算的显式材料预测盲head；保留合法当前接触和共同旧actor，不称整个系统无历史。full≤8192、blind匹配full，合计≤16384 distinct转移；两个collector在组间轮转，不能开四个常驻重物理。缺信号/缺权限先修机制，不盲目加训，也不把pilot当收敛证明。

实施时遵守计划中的四个关键合同：private history reset保护当前接触足的完整footprint和共享owner组，而不只是loaded cells；timing动作在walking_requested和真实liftoff完成pair切换后只采一次并锁存，P先给base、R加一次残差，不能沿旧hook读错pair；新PPO adapter使用action_bound=1及逐转移dt的[T=128,B=2] GAE，不能直接复用旧默认bound1.2/标量折扣；pilot固定选定M+P0+原路线E60，短重访作为冻结评估并报告分布变化，旧cap的action_change不冒称timing平滑项。

三个模块先各自给出对照；P/R先可用M0开发，不等M1胜出，采样前再冻结选定M。之后冻结同一个最终R1，按计划推进B/P/R/PR及短重访/充分恢复对照，同轮完成首批16个机制episode；独立blind训练触发时另4个同条件评价。新机制task与原资格分开，静站不是重访；原plant、材料和硬界不变。实际新控制版本分别取得必要E30→E60，包括更换observer后的B组；只有值得作为最终联合候选的版本继续自己的E300/E600，不能借旧资格。负结果允许不采用新模块，但须给出可核原因和本轮定向处理，不只交三份离线脚本后停下。

25核先按4+4+14+3共享上限组织，数学线程1、构建总-j6计入配额；线上保持CPU FP64，GPU用于独立真实材料历史/候选批次与学习。R双collector采样时占两物理槽，P转保存事件/候选工作，M继续独立批次；联合评价由两槽队列连续处理。记录真正CPU核时、端到端GPU吞吐和物理重叠，不能把affinity或占满96GB当有效工作。N59不占固定主物理槽。

所有新后台deadline=null、wall_deadline_enabled=false、kill_on_conversation_end=false，省略--deadline-unix；不设置墙钟结束时刻、到点kill或对话结束kill。原物理自然结束/失败条件与计划样本预算保留。按证据持续推进已授权工作，不因两次失败或写完一个小文件就结束，也不无依据原样重跑。

每天在docs/S1_DAILY_WORKLOG.md详细记录目标、假设、实际改动、job/attempt/时刻、实验与训练新增量、解决/未解问题、正负证据、资源、commit/Release及下一步；按Asia/Shanghai日期和唯一事件去重。GitHub直交，唯一整合者不强推；核远端commit及Release size/digest后结束，不在用户电脑二次归档。保留W44完整学习状态和所有被引用服务器原件，不自动删除，不借电脑VPN或SSH转发。

最终分别报告：工程修复、M/P/R独立收益、联合增益是否成立、最高原任务资格、机制任务结果、训练真实数量、步态问题、失败/中断/未运行、实际资源与GitHub链接。说明哪些仍是解释假设。W57计划已经授权上述工作，不把计划文件存在当作实现或实验完成。
