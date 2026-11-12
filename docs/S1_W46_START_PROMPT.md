# W46 Codex启动提示词

以下内容可完整交给具有D57与仓库访问能力的Codex。它授权实施并运行计划中的有限新工作；发布本文件不代表r35已经构建或任务已经启动。

```text
在 kaiwen123-yang/MCP-RL 继续执行 W46。先同步并核对当前 main 与现场活动任务，完整读根 AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md、docs/S1_W46_STABILITY_PLAN.md，以及 results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review/README.md、solver-report.md、gpu-report.md、literature-and-baseline.md、handoff.json。以实际新结果和用户最新要求覆盖历史指令，不重启已完成的 W45/G5。

结果基点为3e3f69c440d6a33c14131563e5f485d88da3a803。W45/r34已结束：A/B固定各自update57的原E30为13.072/5.582s body_fall，合法去返/往返0，E60/300/600 blocked；974次planner调用新alpha0安装0，新增PPO0/0，各仍14737/57。保留r34，不重复把修alpha0或增加PPO当新主任务。

先理解原件：A明显侧倾早于首次材料加载，B全程未加载；39次PANN置零动作对照cap差全0。早期native本身已有剧烈的投影后目标/力矩变化，不能全归于plant误差。raw Kδ有参考表示依赖性，不能以它变小宣称稳定；短暂腾空原本允许，不设新硬失败阈值。A153原J下降确实改善短期高度和前进，不直接删除MaterialSupportHeight或撤回r14全局支持。

W46唯一主机制是待验证的完整执行输入参考代价，拟实现S5-r35。严格按计划第2节：每个problem epoch用当前输入incumbent的完整native rollout冻结共同u_bar、F_bar、R/context/hash；所有incumbent/nominal/alpha候选都按同一新目标重评分。共同参考不是已证明稳定的moving reference。新代价作用于完整策略实际投影后qtarget相对共同参考，不罚candidate内部raw Kδ，不冒充相邻输入速率代价。DP/Riccati变量保持held qtarget u=data.ctrl，不改成未投影中心v；沿用r30活动约束后向，不重复乘投影Jacobian。贯通native J、残差/导数、K/H、archive/Iteration与记录，不能只改Quu正则。

四个当前W45问题：A call148/plan136@2.54、A153/140@2.64、B155/141@2.68、B158/144@2.74。A148只在前20ms相对温和，其全350ms不是稳定参考。每问题r34、trace匹配constant、承载调度load各一次有限求解，共12次。Rmin=0、Rmax=.1152 cost/rad²为现有Effort的固定工程标度，不扫倍率；constant匹配每节点load平均强度。按预测法向承载份额w调度，不把相对份额称绝对承载能力。

每分支再做原初态及±记录切空间误差的350ms闭环诊断，共36次。固定该次完整策略及共同参考，350ms只运行2ms反馈/投影/保持，不每20ms重优化。统一当前native predictor模型、implicitfast、10×.2ms、coldstart/材料proxy；新variant的rollout不是原plant真值。缺失历史私有状态明列，称共同重建上下文，不伪称精确历史replay。原J分项、新代价、投影后目标/该rollout力矩、支撑恢复/前移/末端状态、硬条件及偏差敏感性并列，不能比较不同定义的总J吹收益。

完成必要实现/数值一致性核对，选择一个有实际恢复依据的分支，冻结spec/patch/源hash/二进制/记录结构。constant若更有依据可如实选择；load未超越trace匹配对照就不称承载信息收益。若两新分支均锁住坏参考、变驻留或无可辨有用改变，交付mechanism_not_supported有限负结果，不扩参数网格、不反复同配置E30、不现场叠FDDP/高度/步态/时域改动。

若有限比较支持新分支，先推实际代码/spec/有限证据和真实W46 phase，再同时启动A/B evaluation-only原Go2/TM02任务。A固定outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w44-a/PANN_PPO_S5-seed1/checkpoint-014737.pt；B固定对应w44-b/seed2文件。保留同目录resume-state-ep00428/00417及完整学习状态。Actor mean/归一化/seed/case各自固定，PANN因果观察和cap链继续，PPO不更新、PANN网络权重固定，observer/material memory继续因果更新；计数14737/57、tail0、excluded145，新增0/0，不自动执行未来+512/2。

使用scripts/run_s1_go2_r12.py单次evaluation入口，明确--attempt/--evaluation-start-unix/--checkpoint/--level/--seed/--workers/--build/--contract和--condition nominal；不走要求本phase新增训练的--job-dir分支。新建明确W46调度，不覆盖W45。同步contract active_controller_build/controller_artifact/selected_mechanism与CLI和实际binary。扩展现有只列到S5-r34的prepare/runtime/policy/records白名单，确保r35实际继承MJPC_R33_MECHANISM=N、r34 strict、R29/R27记录；r33默认S，不能漏传而静默换机制。核真实readback和首调用身份，不另开无关smoke批次。

A/B使用同一选定分支，各自原E30通过才同case独立E60→E300→E600。E30满30s、合法实际材料去程和远端/转向，不要求完整返回；E60原1次合法往返与回家连续稳定2s；E300/600按原5/10圈及270/540s完成、完整观察和原稳定条件。失败正常结束及blocked，不换seed凑成功，不把驻留/满时长/成本下降当成功。原质量、PD60/5、力矩、TM02 max(abs(principal_strain))≤.02域、非足接触/跌倒、路线、2Hz/.45/.005步态、.35s horizon及其他真实物理保持。

尽可能双CPU+独立GPU三任务并行，现场核25CPU/96GB GPU是否实际可用。A/B各10 workers、Torch/BLAS/OMP各1、GPU总host额度2含native确认，约3核给主控/系统/发布；整进程树affinity受控。build-j6在双流重负载前完成，有限问题和原E30不额外同时占四组CPU。SSH autodl-d57；CPU树/root/autodl-tmp/MCP-RL-w42，GPU树/root/autodl-tmp/MCP-RL；环境work/envs/d57、work/envs/w41-mjwarp。核已有PID/tmux/clock和未提交修改，隔离可写upstream/build，不经共享可写软链prepare另一任务。

第三GPU停止旧reference-proxy top4，改用四个当前W45问题的原完整baseline/selected×原初态/±偏差，最多24scenario。先同记录state的完整FP64 action map：切空间、插值、K/scale、H-QP/活动面/KKT、held命令；再配对native最初20ms，每hold10×.2ms。只有完整语义且约束判断一致，才做同批350ms最多42000 world-step；否则有限诊断结束。MJWarp dynamics是FP32，不能声称改dtype即等价FP64；没有原native完整成本就没有等价排序。第三任务不阻塞CPU，不为占满显存循环。旧G5最终包已在GitHub核验，30s/19.366h但驻留，直接读gpu/w41-gpu/continuation-final-audit.json，不重跑或再收集。

所有新后台deadline=null，执行层wall_deadline_enabled=false，省略--deadline-unix；不设固定墙钟完成时刻、到点kill或无限重试。原物理评价秒数、有限求解预算、网络单次timeout保留。按有限工作完成、原评价自然结束、明确技术失败或用户停止结束。存活进程可精确PID暂停恢复并保留clock；尚未完成且因暂停/技术原因退出者保留interrupted；自然完成/物理失败保持原结果。无完整plant/材料/observer/MPC/任务状态时仅Actor不能中途续接，需要续跑则新独立attempt从原初态开始，不覆盖或拼接。

启动后确认两条真实物理时间和记录持续增长，不只报告tmux存活。保留首技术错误、完整策略/共同参考身份、J分项、实际命令及关键5000Hz子步、材料受力和合法路线。紧凑Git先推，请求的已生成原件增量Release后推并核size/digest；短写串行合并最新main，不强推、不覆盖并发结果。D57上传只用已授权服务器学术代理，禁止电脑VPN SSH转发，不自动切电脑/Clash/直连，凭据不入Git。

GitHub核验后不在我的电脑额外下载、打包、镜像或归档；服务器原件/checkpoint/resume/视频/上传副本保留，不自动清理；已发布原件不重复上传。最终交接实际commit/build/spec、选择依据、job/PID/clock、每级结果、首异常变化与未解释问题、GPU价值/成本、Git/Release及有效续接点。把计划、实现、运行、控制资格、记忆收益和文件交付分开报告，原任务未通过就明确说未通过。
```

