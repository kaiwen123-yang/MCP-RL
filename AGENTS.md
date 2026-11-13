# Codex执行入口：W46已结束，W47检验搜索一致性与移动候选

<!-- W47_EXECUTION_BEGIN -->
> 执行更新（2026-11-13）：W47 有限计算已结束；配对验证、首分叉和工作量见运行索引。 唯一 R8 原件在 D57 不可用，最新 W46 A/B 所需完整策略与同时间预测缺失；B 未启动 donor solve，未把历史 fallback 当最新双流覆盖。1944 次原调用中有 1601 次正步安装（1598 次 J 下降、3 次风险优先恢复）与 343 次内部停滞。N/r34 strict 保留，新 PPO 与新原 plant 回合均为 0；本轮诊断构建不冒称已部署 S5-r36。以下冻结计划和 W46 审查按其形成时点阅读，当前执行状态以索引为准。 [W47 实际运行索引](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w47/phase.json)。
<!-- W47_EXECUTION_END -->

更新2026-11-13。先读[主计划](docs/TERRAIN_MEMORY_PLAN.md)、[W46最终审查](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W46-final-review/README.md)、[W47详细计划](docs/S1_W47_STABILITY_PLAN.md)、[完整启动提示词](docs/S1_W47_START_PROMPT.md)和[机器交接](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W46-final-review/handoff.json)。用户最新要求覆盖历史后台截止和电脑归档规定。

## 当前事实

- 结果基点`15ee620b8a909246a12baf999c136c2be299655b`。W46/r35 constant自然结束：A E30 26.432s floor–trunk contact，B11.638s body_fall；合法去/返/往返0，后级blocked，新PPO0/0，各14737/update57。96个Release资产已交付。不要重启W45/W46/G5。
- W47/r36为planned_not_implemented。本次审查没有启动新服务器任务、原plant实验、MPC求解或PPO。计划发布不是控制实现或资格通过。
- 四个保存上下文的12次solve中60个backward完成、540个正alpha候选全部技术成功/无trunk violation却J更差。A153只改变旧nominal倍率；constant/load四处同完整策略。B155/B158的alpha0成本不再生nominal。不能放松r34、重新允许alpha0安装或把有限结论外推实际1944calls。
- A全程无实际材料加载，B明显下沉/侧倾在500Hz完整足迹受载之前。原任务失败不是已证明由材料域、GPU、PPO不足、QP失败或后台deadline造成。低层稳定移动仍未建立，材料记忆因果收益仍未验证。
- 既有R8/S3刚地9/9、300s 3/3和W34/W35合法去程事实保留，不等于r35材料资格。长期仍研究同地块跨足/返程材料历史对当前加载、落点和重访的价值。

## W47实施与并行

1. 开场核当前main、未提交修改、有效PID/tmux/job/clock和实际资源；有效W47则接续，不能覆盖并发修改、重复启动或宽前缀kill。服务器已有原件原位读，不重试已被自动审批拒绝的电脑原件批量读取。
2. Task A先读四fixture现存candidate/policy/context/reference定位alpha0与真实方向首分叉；已有数组足够就不新跑。缺字段最多16条fixed-policy native诊断，具体修复后最多8条成对验证。核连续/离散动作、四元数、实际u、K/H/scale、活动面和接触；不预设“换K必然分叉”、qacc warmstart或FP32为原因。
3. Task B固定一个R8成功x/u来源和四fixture；B155/B158加W46 A/B首次力矩反转前对应context，字段不足才用计划fallback。核时标/phase/坐标；K=0可作seed，H来自当前incumbent有效metric，保留原H-QP。旧x仅诊断，当前native从真实x0重滚；这不是FDDP。共同r35 ubar/F/R必须来自原incumbent，两个家庭同J比较全部baseline/donor。最多8solve/24条350ms诊断，另最多700 donor单步转移/7000native子步，不能用重滚自产x替代donor defect。R8 mission phase不是gait phase，按源cadence/GetPhase重建，同elapsed确定性选段。在线仅current N搜索停滞且source合法时追加一个donor家庭，最多两solve/epoch、独立数值状态、同ref联合裁决；有限与在线同入口，不启动时一次导入冒充。不扩R/donor/seed网格。
4. A/B原位分析及种子准备并行；若A修执行语义，B正式solve等共同baseline冻结，避免混比。仅明确实现错误做最小修复；完整FDDP/MS/SQP、步态/高度/时域和PPO不在本轮同时修改。
5. Task C原位聚合1342/602真实calls并修GPU投影别名、动作/状态时标、扩窗口径及warmup/capture计数；GPU用的是W45旧策略，不是r35新selected。24名义场景仅18种有效组合。主体零新physics；需要且接口具备时按计划单场景三引擎首2ms，总≤30真实子步含预热。无24world350ms重跑、无GPU proxy排序、无G5重跑。
6. 仅当两个流最新W46代表context都有可解释的移动/恢复收益及各自同origin真实预测误差扰动证据，才选至多一个控制候选进入原E30；历史fallback不替代该覆盖；只有旧scale改变/驻留或数值正确性修复则有限负结果结束。Task C不阻塞CPU。不得把计划当结果或保证E30通过。

## 原任务与身份

7. 真正改实现后才冻结S5-r36 spec/patch/build/sourceSHA，更新prepare/runtime/policy/records白名单、合同、CLI/env和binary readback，明确N/r34 strict；r33默认S不得静默回退。--build不自动修合同。实现/spec/有限证据/真实phase先推Git，再启动确认两条clock/记录增长。
8. A/B固定各自W44 seed1/2的checkpoint-014737.pt、完整resume ep00428/ep00417、actor/critic/optimizer/norm/RNG/fork/lineage；tail0、excluded145，各14737/57，新PPO0/0。PANN网络固定，observer/material memory继续因果更新。不自动执行未来+512/2。
9. 使用run_s1_go2_r12.py单次evaluation入口，显式attempt/evaluation-start/checkpoint/level/seed/workers/build/contract/condition nominal；不用要求本phase新训练的job-dir。新W47 phase不能覆盖W46。每case通过E30才E60→E300→E600；失败后级blocked，不换seed或拼接资格。
10. 原Go2/TM02质量15.206408、PD60/5、每腿力矩23.7/23.7/45.43、主应变.02、非足接触/跌倒、合法路线、2Hz/.45/.005 C2步态/.35s horizon、r28/r27/r26、658D/2Hz cap/W38奖励保持。短腾空合法，raw Kδ和描述性.5rad/.2m不成为新硬判据。E30满30s合法材料去程及远端/转向；E60满60s一合法往返/稳定2s；E300/600按270/540s内5/10合法往返与完整观察。

## 资源、后台和交付

11. 默认D57：CPU树/root/autodl-tmp/MCP-RL-w42、GPU树/root/autodl-tmp/MCP-RL，SSH autodl-d57；环境work/envs/d57和work/envs/w41-mjwarp。现场核cgroup/CPU/RAM/GPU/磁盘和无关作业，不假定历史96GB全空闲。
12. 两CPU流各10额度，全进程树affinity隔离，Torch/BLAS/OMP各1；第三任务总host2含native确认，约3给系统/主控/发布。build-j6在双重负载前。共享资产只读，可写upstream/build隔离；不在两原E30之外再挂两组10worker solve。XLA预分配false、JAXx64true，显存按实际峰值/剩余准入；第三任务自然结束就释放，不为利用率开第三PPO。
13. **所有新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix；无固定墙钟完成时刻、无到点kill。** 工具等待timeout不能传成后台期限。保留原物理时长、有限数值工作量和单次网络timeout；工作完成/原结果/技术失败/用户停止才结束，不无限重试。
14. 存活进程可精确PID暂停/恢复；技术中断保留interrupted，自然物理失败/任务未满足保留原判定。无完整plant/材料/observer/MPC/任务状态时Actor不能恢复中途物理，必要续跑另建独立attempt。
15. 紧凑Git先发，要求的已生成原件从D57增量Release直传并核commit/URL/size/digest；响应丢失先查已有资产，不重复上传。Git/index短写串行合并最新main，不强推不覆盖并发结果。never_generated/blocked与真实失败分开，不造空文件。
16. **GitHub核验后不在用户电脑额外下载、打包、镜像或归档；服务器checkpoint/resume、原件、视频及上传副本不自动删除。** 上传仅已授权服务器学术代理，禁止电脑VPN SSH转发，不自动切电脑/Clash/直连；凭据不进日志/Git。

最终报告实际commit/build/spec、A/B/C结果及未识别问题、真实rollout/FD/physics/PPO、每级原评价、资源与GPU价值、Git/Release及有效续接点。原任务未通过则明确未通过；稳定之后另立有效历史曝光、候选内材料状态传播、匹配记忆对照、泛化和实机阶段。
