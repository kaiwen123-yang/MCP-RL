# Codex启动提示词：W57模块原型、受控学习与首次联合验证

请直接执行仓库最新main上的W57研究计划，目标是从已稳定的r55/r60基线推进到可归因的模块与联合改进。先读取 `AGENTS.md`、`docs/S1_W57_RESEARCH_PLAN.md`、`results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/current.json`、`reviews/W56-review/README.md`及其control/evidence/module-innovation-roadmap，然后续写 `docs/S1_DAILY_WORKLOG.md`。路径不带根目录的review均位于上述R12目录。

审查基点为3590646bc15a59903c9561381368d985963dfe9e。r55和r60 A/B均已通过原E600；r57 A/B通过E60；r58两条E300是计算停滞后授权终止；N59原E30未贯穿。不要把旧失败率当当前F基线成功率，不原样重跑四条合格E600。先核最新main、已有修改和D57确切job/tmux/PID/物理时钟，已有有效作业则续接，不重复启动。

本轮可并行做三条逻辑线，最多两条重物理：

1. **M材料估计**：建立自身因果load/release事件标记，不能读log=False下未更新的ever_loaded，更不能在线读真plant q；以完整事件族/独立历史保留组比较原前推observer与物理一致的测力校正observer，区分模型、状态和权限误差。新PANN仅在保留数据支持时单独校准，不能因参数冻结就先换模型。
2. **P预测控制**：先将r55/r60约2mm、约100ms前足再离地与原计划/测力/交接/WBC对齐定位；把已有材料candidate rollout真正接入未来接触/载荷预测和实际候选选择。候选按同一完整模型、原目标与硬界比较；若当前法向几何作用弱，同轮推进低频材料事件层覆盖真实load→reload延迟，不再靠重复长跑。做私有候选卸载历史清零对照，不能重置真plant、主observer或当前受载完整状态。
3. **R学习接口**：旧W44 A/B速度actor保持deterministic eval，新建独立timing head，首动作仅下一对角支撑交接请求残差±50ms；原全部support gates保留，预测与实际调度同源。独立59D schema/归一化、事件dt reward/GAE、请求/接受/执行/mask全记录。不要把旧stream.training=True来训练新head。按计划一个共享新head、一个学习seed、2collector各128同版本收集，8×256=2048 distinct on-policy总转移、4epochs/minibatch64；有效动作和mask单列，旧actor更新0与新head更新分列。pilot不冒称收敛。

实施时遵守计划中的四个关键合同：private history reset保护当前接触足的完整footprint和共享owner组，而不只是loaded cells；timing动作在walking_requested和真实liftoff完成pair切换后只采一次并锁存，P先给base、R加一次残差，不能沿旧hook读错pair；新PPO adapter使用action_bound=1及逐转移dt的[T=128,B=2] GAE，不能直接复用旧默认bound1.2/标量折扣；pilot固定选定M+P0+原路线E60，短重访作为冻结评估并报告分布变化，旧cap的action_change不冒称timing平滑项。

三个模块先各自给出对照，再冻结选定M与同一个R1，按计划推进B/P/R/PR及短重访/充分恢复对照，条件满足同轮完成首批16个机制episode。新机制task与原资格分开，静站不是重访；原plant、材料和硬界不变。实际新控制版本分别取得必要E30→E60，包括更换observer后的B组；只有值得作为最终联合候选的版本继续自己的E300/E600，不能借旧资格。负结果允许不采用新模块，但须给出可核原因和本轮定向处理，不只交三份离线脚本后停下。

25核先按4+4+14+3共享上限组织，数学线程1、构建总-j6计入配额；线上保持CPU FP64，GPU用于独立真实材料历史/候选批次与学习，不凑显存。记录真正CPU核时、端到端GPU吞吐和物理重叠。N59不占固定主物理槽。

所有新后台deadline=null、wall_deadline_enabled=false、kill_on_conversation_end=false，省略--deadline-unix；不设置墙钟结束时刻、到点kill或对话结束kill。原物理自然结束/失败条件与计划样本预算保留。按证据持续推进已授权工作，不因两次失败或写完一个小文件就结束，也不无依据原样重跑。

每天在docs/S1_DAILY_WORKLOG.md详细记录目标、假设、实际改动、job/attempt/时刻、实验与训练新增量、解决/未解问题、正负证据、资源、commit/Release及下一步；按Asia/Shanghai日期和唯一事件去重。GitHub直交，唯一整合者不强推；核远端commit及Release size/digest后结束，不在用户电脑二次归档。保留W44完整学习状态和所有被引用服务器原件，不自动删除，不借电脑VPN或SSH转发。

最终分别报告：工程修复、M/P/R独立收益、联合增益是否成立、最高原任务资格、机制任务结果、训练真实数量、步态问题、失败/中断/未运行、实际资源与GitHub链接。说明哪些仍是解释假设。W57计划已经授权上述工作，不把计划文件存在当作实现或实验完成。
