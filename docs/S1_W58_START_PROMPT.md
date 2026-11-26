> 2026-11-26续接说明：W57/W58已结束，以下为已执行的历史合同。W58最终G0/M1，联合准入未通过，不能把条件计划写成已运行。当前新工作请读[W59计划](S1_W59_RESEARCH_PLAN.md)、[启动提示词](S1_W59_START_PROMPT.md)及[最新审查](S1_W58_REVIEW_20261126.md)。原结果和判据不改。

# Codex启动提示词：续接W57，成组推进W58

本提示词与[S1_W58_RESEARCH_PLAN.md](S1_W58_RESEARCH_PLAN.md)配套。先看现场，不根据审查快照重复启动已在运行或已完成任务。计划形成日2026-11-23，尚无W58执行成功声明。

## 可直接复制

~~~text
你在 kaiwen123-yang/MCP-RL 项目继续执行。先安全同步最新main，读取AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md、docs/S1_W57_REVIEW_20261123.md、docs/S1_W58_RESEARCH_PLAN.md、docs/S1_W57_LITERATURE_UPDATE_20261123.md、docs/S1_DAILY_WORKLOG.md，以及R12 current.json、w57/index.json和reviews/W57-review/handoff.json。R12为results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12。按该计划完成实装和已具条件的后继，不停在复述计划、一次探针或一个小修补。

先核D57实际工作区、tmux/PID、最后物理时钟、attempt、source/build/checkpoint、campaign final和两份dispatcher。2026-11-23 19:20:56审查快照中W57仍在执行：full8192/32updates完成；blind7424/29updates+6partial继续；新冻结原资格只有M1+B A/B E30和M1+P A E30。该数量不是现场实时值。已有有效作业就续接，不重启、不把训练E60当冻结资格、不用旧“W57未实施”段落覆盖当前事实。

第一阶段：保持W57当前冻结身份和双物理槽，让匹配blind、自然collector结尾、B/P/R/PR的16条机制与匹配blind4条、各自原资格、条件长任务及独立private-short-A后继按原合同自然完成。已有同身份结果引用即可，不重跑旧r55/r60 E600。PR是P0训练的冻结R与P1组合，不能叫联合on-policy训练。83D W57-R1-event83-v1及旧W44冻结不变，不能热换M/P、热加collector或重复计部分缓冲。完整报告最终checkpoint、冻结视频、正负结果、未运行及资格。

同时在独立工作树并行准备三线：
G：基于本轮5kHz证据修正末摆动提前触地、参考连续接管和承载建立；不把短碰撞当已承载、不重置真实p/v、不改PD或删硬界，直接取得新身份原E30/E60，并分地面/任务阶段量化冲击、反弹、真实接触表和步态质量。
M2：针对源释放→实际跨足重载的力/预载状态误差和不确定性，做可辨识低维状态校正与完整历史保留评估；20.2ms改善不能掩盖重载退化。不给observer真plant内态，不在旧训练中热换M。
P2：分离接触可用时长修正和私有材料历史作用；采用相同候选、落点/等待权限、目标和执行后端的full/private-blind，记录实际未来加载后果。已有2s事件层仍0赢家差，不能只加长horizon或加材料权重。历史动作差需超过相关数值/模型误差，收益需连同进度、等待和CoT评价。

W57旧机制6/12、7/12是足印几何未满足，保留原判。新v2任务先用baseline足印/独立几何确定真实source-release-cross-foot同格正面积交叠，冻结12个目标事件及短/恢复配置，所有组共享路线/权限并报告全部未达标事件，不按full胜负挑路线。先完成B2、P2-private、P2-full、Fixed-wait四臂×两历史×A/B共16条开发矩阵；若用于选方法就不能称独立保留验证。

只有真实曝光、历史动作价值和闭环收益迹象成立，才按计划同轮启动新的P2+timing联合学习。必须实现W58新runner/spec、可配置learning seed5801/5802/5803及P2/R各自的历史屏蔽范围；现W57固定P0/5701，复制目录不是多seed。每seed的full/private-blind同初始化、分别训练；每个seed由full的预声明诊断决定最终N，private-blind匹配同一N，不能按自身较差回报提前减预算；每head累计2048→4096→8192，不重复加节点，六head条件上限49152 distinct。保留真实dt的[128,2]GAE、policy-version及log-density合同。训练后的独立机制验证至少32条（学习两臂各3seed共24，基线/固定等待共8），开发集不可复用；各新身份/seed/case取得自己原E30/E60，按计划预指定主seed的条件长资格，不选保留成绩最好seed包装结果。机制阴性就量化原因/效应上限，不无依据扩训练。

W57所有后继及dispatcher自然结束后，统一迁移共享资源池，W58默认3物理槽，实测吞吐和CPU/RAM/GPU/I/O允许时扩4–6。双collector必须原子取得2槽并持有到两个collector及自然尾episode都结束，更新屏障不释放槽：3槽=1campaign+1独立评价，6槽才可3campaign。显式4个native材料worker计入配额，不能只限制OMP。25核不是独占权，先扣其他项目负载；每流physical_s/wall_s、总distinct/h、P95、CPU core-s、RSS、GPU与写队列都记录。GPU用于真实有用批次，不凑96GB占用；不要为测吞吐额外制造无关物理任务。

全程deadline=null、wall_deadline_enabled=false、kill_on_conversation_end=false，不设置墙钟完成时刻、不传deadline-unix、对话结束不kill；保留原模拟时长、原资格路线时限和物理/材料失败条件。并行开发后融合控制器必须先冻结统一身份，禁止边评边改。

每日按Asia/Shanghai和唯一事件ID详细追加docs/S1_DAILY_WORKLOG.md，记目标、修改与身份、实际物理/训练计数、解决与未解、正负证据、资源、commit/Release和续接。小代码/结果/协议及时推GitHub，完整必要原件/视频/checkpoint由D57增量直传Release并核size/digest；推送前合并最新main、不强推、不覆盖动态运行索引。GitHub核验后不在我的电脑下载、复制、打包、镜像或归档。此次不启动24路数据迁移、不删除原始数据、不影响其他项目。

已有授权且条件明确的工作直接连续做；实际前置不满足就记录具体证据并推进其余可做线，不在每个小步骤结束索要确认。只有缺失身份/权限会造成不可逆错误时才停对应动作。最终报告要区分工程资格、M预测、P历史决策、R学习、PR联合、实际并行和未运行，给GitHub commit/Release以及精确续接点；若后台仍正常运行就明确running，不虚报整轮完成。
~~~

## 执行者优先入口

- 当前事实：[W57审查](S1_W57_REVIEW_20261123.md)、[机器交接](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W57-review/handoff.json)。
- 具体阶段、数量、门和资源规则：[W58完整计划](S1_W58_RESEARCH_PLAN.md)。
- G原始诊断：[视频/原数组报告](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W57-review/evidence/video-and-raw-review.md)。
- M/P/R代码/并发边界：[独立审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W57-review/control-and-learning.md)。
- 日志：[唯一每日工作文件](S1_DAILY_WORKLOG.md)。
