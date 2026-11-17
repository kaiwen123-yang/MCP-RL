# W53 Codex 启动提示词

**2026-11-17：此启动提示已执行。N/F各一条真实原E30均未通过，另两次零物理技术初始化已保留。W53本轮已结束，不重复下方启动命令；实际结果和续接入口见 [W53 phase](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w53/phase.json)。**

2026-11-16。W52 已完成负结果；本提示词用于下一轮 W53 的实际实施、实验与交付。发布此提示词时 W53 尚未运行。完整授权范围与条件见 [W53 计划](S1_W53_STABILITY_PLAN.md)。

```text
继续 kaiwen123-yang/MCP-RL 的 W53。先拉取并整合最新 main，读取 AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md、docs/S1_W53_STABILITY_PLAN.md，以及 results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W52-review/ 下的 README.md、control.md、evidence.md、baseline-and-literature.md、literature.md、handoff.json。按计划实际实施、运行、分析并推送，不停在建议，不再询问已经授权的启动/发布权限；若已有同身份有效任务，核对具体 job/tmux/PID/物理时钟后续接，不重复启动。

W52 D2/r40、D3/r41 已分别在5.6716/6.6278s因floor–trunk非法接触结束，材料加载0，原E30未启动。参考真正过期0。首有效S的5mm目标对应native FR/RL仍28.20/33.75N；保存数组精确证实S保留旧B的reference/K/H和支撑中心，只追加qIK差。不能继续把局部几何补丁当完整卸载控制，也不能把旧偏置或K直接删掉当作已证修复。不要重开已结束D2/D3、旧D8、旧GPU探针或已完成的C高度/残差归约。

并行实施两条完整运动路线。N：按计划锁定源码/pin/三补丁和R8/S3完整运动语义，恢复作者full-state iLQG、运动目标、.03m原步高与仅Stand启用的支撑参考，接当前原Go2/TM02 plant、路线、冻结PANN/PPO cap和记录。它是材料未知的新适配基线，旧刚地9/9和300s 3/3不继承资格；不能保留全部当前moving成本/H-QP/冻结承重中心却称恢复R8。已知执行正确性修复按计划明确列差异，不复活已证错误。

F：独立实现接触力MPC+WBC，统一身体、支撑力、摆足p/v/a与接触日程；使用同一Go2模型和原限位/力矩，完整输出q_d/v_d/tau_ff并正确映射原PD60/5。原接口u=q_d+(tau_ff+5*v_d)/60，约束映射后u和完整总PD力矩，避免重复支撑偏置/反馈；不再用旧B的K/H/J接管。F固定2Hz、duty .60、3cm、.20s摆动，按W53详细计划冻结起步/接触转换，明确软gait与硬接触日程区别；材料stance允许真实法向压入/回弹，不改实际接触标记，不用未来真值。一套参数在首个E30前冻结，不扫参追成功。

N/F各预声明一次原E30，同W44 A seed1 checkpoint-014737、归一化、确定actor、完整t=0和原2s Stand；真实spec/build/runtime/backend/命令readback/hold/记录/资格身份全部接通。先就绪者先运行，不等待另一线或C，不先加D8或smoke。原物理失败正常结束，技术未完成如实未运行。完整分析视频和轨迹；若N真正通过原E30且正常持续推进优先选N，若N的E30失败或已明确技术阻塞/本轮实现终止、未形成可用结果，而F通过上述条件，则选F；N正常构建或运行不算失败，技术未运行与物理失败分账。所选seed1这次身份完全一致的E30直接作E60前序，另一槽开所选W44 B seed2 E30；各case按原E30→E60→E300→E600条件推进，失败只阻塞自己的后级。两首轮都失败就交实际分阶段负结果，不新加第三控制族。任务标准不放宽，不借旧或他case资格，新PPO/donor=0。

C为不阻塞主线的第三项：服务F的FP64 GPU批量Go2运动学/动力学与实际MPC/WBC矩阵装配，只用已有W52完整状态一次有限遍历，和同冻结模型CPU参考比较完整所需输出、覆盖和含传输耗时。无新plant/native rollout/FD/PPO，无中途GPU在线替换；不重复已完成小归约占显存，确实无收益就如实结束。

现场25核余量按N最多10+F最多10+C/记录2+系统/其他项目3，线程库1，构建总上限先-j6，隔离可写build和进程树。随正常任务分解native/FD/backward/RPC/提取/序列化/压缩IO的wall与CPU，不能以affinity冒充满载。保留全程500Hz轨迹、5000Hz实际接触/力矩/双时钟、材料与全时段视频、每次实际安装的完整政策/参考/获选轨迹；按计划减少非赢家全ring重复保存，记录缺失边界，不重跑补档、不删历史原件。

所有新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix；无固定墙钟完成时间、无到点kill、对话结束不杀后台。原物理任务时长/失败条件、明确技术或资源错误、用户停止与有限科学工作量保留。tmux正常工作后可汇报真实running并结束对话，后台继续已授权内容。

代码和紧凑结果推GitHub，必要原件与完整视频D57直接增量上传Release；短写串行整合最新main、不强推，回执不明先核远端。远端commit/URL/size/digest核验后结束，不在我的电脑额外下载、复制、镜像、打包或归档。保留服务器被引用checkpoint/resume/原件，不自动借我的VPN/Clash/SSH转发。最终分开报告实现、正常行走、原任务资格、材料记忆收益、GPU覆盖、实际失败/blocked/未运行、资源与GitHub资产、下一续接点。
```
