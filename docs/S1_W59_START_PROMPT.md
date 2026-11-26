# W59 Codex 启动提示词

下方文本可整体交给在 D57/AutoDL 工作的 Codex。它授权按已推送的 W59 合同连续实现、训练、仿真、评价及 GitHub 交付；本文件本身不是这些任务已启动的记录。

```text
继续 kaiwen123-yang/MCP-RL 的 W59。先同步最新 main、读取根 AGENTS.md，并完整读：
docs/S1_W58_REVIEW_20261126.md
docs/S1_W59_OFFICIAL_METHODS_20261126.md
docs/S1_W59_RESEARCH_PLAN.md
experiments/terrain_memory/s1-w59-research-contract.json
results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W58-review/handoff.json
docs/S1_DAILY_WORKLOG.md
再核现场工作树、确切job/PID/tmux/物理时钟、当前资源与已迁移数据清单；有效同身份工作直接续接，不重复启动。

研究目标是可恢复世界地块材料记忆的跨足/跨步协同。仅依据官方SuperMimic论文v3和官方源码固定commit 5fb92297ab4fe3d86629d9f0c635334791bb0d5f借鉴方法；不要读取或使用我的SuperMimic复现记录、权重、数据或结果来设计本项目，不迁移G1权重。

继承事实：W57 full/blind各8192去重转移、32updates，PR A/B原E600已完成；PR是冻结组合不是新联合训练。W58 56attempts结束=53自然终局+3技术中断，最终G0/M1，joint训练0。G1降低off-time却恶化slip，M2恢复误差涉及抵消，P2 3413次非空history干预0赢家翻转。W58 validation已用于设计，不能再作W59未见数据。不要重复旧失败版本来凑进展。

按W59计划同时推进：
1. 从既有原件构建带controller/action/model版本的连续数据bank；三套模型对照为原M1解析、同输入黑盒WM、M3物理记忆+执行残差WM。分别处理release初态、free recovery law、reload solve，并学习真实touch/rebound/bearing、负载、滑移、功和区间极值。保留因果world-cell身份及不确定性，不在线读取材料真隐状态。
2. 实装唯一宏动作接口(due offset, transfer ramp, pair load preference)。同一最终request同时进入schedule/forecast/MPC/WBC；总请求统一投影、零残差还原P3。保留原材料/摩擦/力矩/关节/承载/释放/任务硬界，停用旧1D timing head避免重复叠加，不在PD后额外加torque。明确actor事件时钟与20ms模型/force MPC时钟。
3. 训练并校准action-conditioned执行WM；P3对每候选展开自己的加载历史。先候选排序，后条件式BPTT。24合法request状态预分12开发+12封存准入，各8分支，最多192分支/384模拟秒；中途快照不可热改已承诺动作。首次actor参数更新另按合同作独立真实before/after检验。
4. 同轮完成新身份原E30/E60、16开发机制及必要M1/M3配对。保留旧12目标阴性；原路线全机会分母与独立固定足印机制任务分开，另冻新holdout。真实接触点滑移与旧指标并列，不事后换指标撤销旧选择。
5. 条件具备则连续执行PPO-full、Diff-full、Diff-private、Diff-fixedWM四臂；seed5901先各2048，再按合同推进5902/5903与共同累计4096/8192。最多12heads、98304条实际用于优化的去重转移；尾回合/资格/评价/分支另计真实成本。每seed四臂匹配同N_s，数据/模型/optimizer隔离，想象样本不算真实交互。独立机制64条，若追加材料闭环泛化配对为68条，资格另计；主展示seed预定5901。

actor按封口128×2事件batch更新；WM拟合可后台进行，发布到collector等预定采集节点及两个自然尾回合结束，不阻塞仍未封口batch。真实任务critic区分成功保持、失败和truncation；保持/接触用原细粒度证据，不能宏端点漏峰。参数更新记录实际accepted steps、触发KL和回滚optimizer。路径梯度失信时保留真实PPO/候选搜索分支，但改标fallback身份，不把回退收益冒称纯BPTT。

资源按统一租约使用。六个工作流并行，物理默认3槽，当前存储/吞吐和其他项目余量支持则4–6；可复用W58六槽证据并在自然窗口确认。双collector campaign原子占2槽，六槽最多3campaign。25核中物理池上限18、辅助上限4、至少3留系统/其他；显式native workers另计，数学库线程1。GPU优先批量M3/WM训练、candidate evaluation和短imagination，按真实吞吐选择1或2 trainer，不为占满96GB显存制造无用计算。

不设置后台墙钟完成时间：deadline=null、wall_deadline_enabled=false、kill_on_conversation_end=false，不传deadline-unix，不因对话结束kill。原模拟时长、样本节点、物理/数值失败和用户停止仍有效。出现明确接线/记录错误本轮定向修复并建新身份，保留技术中断；没有新证据不得原样重跑已定负版本。独立分支继续推进，不只写框架或跑一个小样例就停。

只做当前改动必要的源码/保存数据核验与正式实验，不做全仓hash/mock/import/smoke或无信息参数网格。不要动其他项目，不启动原始数据迁移或删除。已迁移原件按实际清单读取，不假定旧路径存在。

每天按Asia/Shanghai在唯一docs/S1_DAILY_WORKLOG.md详细追加：目标与假设、改动和版本、实际新增物理/学习、问题与正负证据、资源与并发、GitHub回执和续接；用事件ID去重。最新代码、合同、紧凑结果及时串行整合main并非强推发布；必要原始日志、完整视频和checkpoint从D57直传GitHub Release并核size/digest。GitHub交付成功后不要在我的电脑额外下载、镜像、复制、打包或归档。

直到已具条件的模型、规划、学习、独立评价、原资格和GitHub交付链完成，或有明确且可核验的科学/资源阻塞。最终分别报告：实现了什么、最高真实资格、步态代价、材料历史因果效果、各head真实/想象/使用/尾样本、模型与梯度可靠性、实际并发和GPU用途、负结果/未运行原因、commit/Release与下一续接点。计划和模型内奖励不等于实验成功。
```
