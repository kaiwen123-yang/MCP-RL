# Codex：W46原任务已结束并交付，E30两流均未通过

## W46 completed original results — verified 2026-11-13

**Final original-task result, verified 2026-11-13:** W46 has naturally ended; no W46 process remains. A E30 ended at 26.432s on floor–trunk contact (wall65.59min); B ended at 11.638s on body_fall (wall40.42min). Both retain their own update57, new PPO0/0. E30 passes0/2, legal outbound/return/roundtrips0, E60/300/600 blocked. A max x=.626506m and no actual material load; B max x=.396364m, actual material load and max strain=.00649112<.02. Compared with W45, both lasted longer but A made less forward progress. Actual target jumps still reach1.53536/1.46864rad and adjacent500Hz actuator force changes90.86Nm; the short-window improvement did not establish original-task stability or memory benefit. All1944 calls used N/r34 strict, new alpha0 installs0. Both videos were rendered. All96 requested assets are present in the three W46 Releases; one response-lost upload receipt was reconciled against the existing server/remote size and SHA256 without re-upload. The final phase/index is authoritative; launch snapshots below are historical. This bounded round is complete; no W45/G5 restart, parameter sweep, extra PPO or next architecture has been launched.



S5-r35 is now built from reviewed main a34bc45393da1e552df135f856df6f98c762c51f. The 12 declared finite solves and 36 fixed-policy 350ms native diagnostics are complete; no PPO or original-task physics was added by that comparison. The selected original-task branch is **constant**, with mechanism N and r34 strict acceptance retained. Load and constant selected identical complete policies in all four contexts, so no load-information benefit is supported.

A153 provides limited useful evidence: nominal target jump 1.08418→0.56936 rad and torque jump 65.3074→34.2848 Nm, similarly under both recorded perturbations. Minimum height decreases 1.152mm and J0 rises .0002356; terminal vertical recovery remains positive and small forward displacement increases. The other three contexts are unchanged; B158 still has a bad high-jump incumbent. This supports one original A/B evaluation, not a stability claim.

The common-reference residual reconstruction error is zero; all four disabled-penalty native state/action/residual comparisons are exactly equal to r34. Candidate cost recomputation error is at most 4.45e-16, held-u residual FD error at most 7.96e-11, direct state C zero. Actual evidence, frozen binary/spec identity and the new phase are under [w46](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/). At the 2026-11-12 launch, both original evaluation-only E30 ladders were observed running in tmux s1-w46-a/s1-w46-b with advancing physical clocks (phase.json is the live authority); each keeps its own W44 checkpoint014737/update57, new PPO0/0, no wall deadline.

The independent GPU study has completed 24 old-policy scenarios and 42000 world-steps including the 20ms prefix, plus 48 warmup steps. It uses GPU FP64 feedback algebra and the exact old native H-QP on host, then an explicit cast to MJWarp FP32 dynamics. Same-state held-action error is 2.49e-16rad, but 20ms cross-engine command differences reach 1.501rad; no equivalent ranking or online acceleration is established. Original plant Euler is not re-simulated; the paired dynamics are native implicitfast and MJWarp.

Implementation: [spec](experiments/terrain_memory/s1-go2-r11-S5-r35.json), [finite selection](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/mechanism-evidence.json), [phase](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/phase.json), [GPU evidence](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w46/gpu.json). The W45 facts and original bounded protocol below remain historical provenance and operating constraints.


## 当前事实与唯一入口（2026-11-12）

先读[主计划](docs/TERRAIN_MEMORY_PLAN.md)、[W45最终原件审查](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review/README.md)、[W46稳定性计划](docs/S1_W46_STABILITY_PLAN.md)、[完整启动提示词](docs/S1_W46_START_PROMPT.md)。审查基点`3e3f69c440d6a33c14131563e5f485d88da3a803`。用户最新要求和本文件覆盖历史后台截止、电脑归档及自动清理规则。

**W45/r34已自然结束且原任务未通过；W46/r35目前仅为计划，尚未实现、构建或启动。** 本次审查新增服务器任务/physics/PPO均0，不把审查发布当作新控制器部署。

- W45两流固定各自W44 update57，样本/更新仍14737/57，新PPO0/0。原E30 A13.072s、B5.582s因roll越过1.2rad body_fall，runner exit0/error null；并非墙钟到点kill。E60/300/600均blocked，合法去程/返程/往返0。
- A最多前移.8251715m，首次材料受力13.048s，仅末段18个5000Hz正受力子步；B最多.2972755m、无材料受力。39份PANN置零cap对照全部差0，尚未形成材料记忆动作收益。
- 全程674/300次planner call均r34 strict，新alpha0安装0。四上下文selected→installed完整13组数组0差，原J复算到机器精度；实际QP failure/restoration0。接纳修复真实生效，不能再当未完成项重复实施。
- 新证据在完整native策略本身：A2.650s投影后目标相邻2ms变化1.41880rad，actual1.16403rad；B2.764s为1.23629/1.23587rad。某膝力矩相邻变化90.86Nm是正负限值反转，不是实际越界施加90.86Nm。原目标真实下降但稳定移动未成立。
- raw Kδ有表示依赖，不单独作为新成本/成功条件；短暂腾空不构成新失败。Euler/implicitfast等差异尚未完全分离，不能说其是唯一原因，也不能把一次早期事件说成十秒后跌倒的充分原因。
- G5最终完整原件**已在GitHub Release找到并核验**，新增入口`results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/gpu/w41-gpu/continuation-final-audit.json`。冻结W40-B/update37/r30，30s耗19.366h，几乎驻留、无材料任务进度；保留旧技术失败attempt，不再上服务器补找已发布文件，不重跑G5。
- W45 GPU proxy四问题top4平均召回.25、2/4漏最优。缺K/H/H投影且目标不同，不重复扩大proxy筛选。最终选择应包含全部已确认baseline。

长期研究仍是同一世界地块跨足/返程恢复历史对加载、落点与重访的因果收益。当前候选没有完整传播各自未来材料历史；合法材料往返、匹配记忆对照、泛化与实机尚未建立。R8/S3原硬Go2刚地9/9、连续300s 3/3成立，不能转移成r34/TM02资格。

## W46控制机制与有限比较

1. 唯一主机制为移动阶段的共同输入参考代价，比较r34原版、constant、load三分支。每个真实epoch先按当前状态重评分输入incumbent完整native策略，冻结投影后u_bar、法向力份额/R/context/hash；所有候选使用同一参考和同一新J重评分。参考更新时更新问题身份、清理失效分数/数值状态。当前incumbent不是已证明稳定的moving reference，必须检查锁住坏轨迹/驻留的反例。
2. DP/Riccati变量保持实际held qtarget u=data.ctrl。冻结R/u_bar后，阶段c_u=R(u−u_bar)、c_uu=R、直接状态导数0，成本引擎仅归约一次/176，175控制节点、terminal新增0。候选v经原Clamp/H-QP得到u再评分；保留r30状态相关PD bound/活动面耦合，不重复乘投影Jacobian，不把v当u做FD。新项贯通native residual/score/derivatives/Riccati/K/H/archive/Iteration，不能仅改Quu LM正则。
3. load按R_j=Rmax−w_j(Rmax−Rmin)，w为非负法向力份额，全无载w0；constant匹配同节点平均强度。首次Rmin0、Rmax=.1152 I3，来自现Effort/PD单位，不是论文最优值。保留原高度/步态/horizon/PD/物理，不自动扩R网格、裁raw K或改solver架构。
4. 固定四个W45上下文：A148/136@2.54、A153/140@2.64、B155/141@2.68、B158/144@2.74。4×3共12次有限求解，各输出原初态/记录尺度正负偏差共36次350ms固定policy native诊断。implicitfast10×.2ms/coldstart/冻结材料proxy，窗内不20ms重规划；不是新variant原plant真值，也不冒称精确历史重放。A148仅前20ms较温和，整段并不稳定。
5. 比较共同原J0、实际目标/力矩变化、接触/姿态/恢复/前移及原硬条件，不直接比较不同定义总J。选择一个分支，两流一致；load须有超出trace匹配constant的证据。若都无用或退回驻留，交付mechanism_not_supported有限负结果，不重复已知更差E30，不现场追加FDDP/鲁棒评分/额外训练。后续另立完整moving nominal/可行性热启动问题。

## 原任务、启动身份与固定策略

6. 有支持证据才冻结S5-r35实际spec/patch/build/prepare/runtime/records，正常构建并记录真实hash。现有白名单只到r34，必须同步r35分支，显式MJPC_R33_MECHANISM=N、r34 strict、R29/R27记录；r33默认S不可静默回退。合同reuse.active_controller_build/controller_artifact/selected_mechanism、CLI、PolicyStream与binary readback一致，--build不自动修合同。
7. A/B固定自身`outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w44-a/PANN_PPO_S5-seed1/checkpoint-014737.pt`和w44-b/seed2同名checkpoint；完整resume为ep00428/ep00417，tail0、excluded145，actor/critic/optimizer/norm/RNG/fork/lineage保留。W46新增PPO0/0；PANN网络权重固定，但observer/material memory按观测继续因果更新。未来15249/59须另立学习phase。
8. 使用run_s1_go2_r12.py单次evaluation入口，明确--attempt --evaluation-start-unix --checkpoint --level --seed --workers --build --contract --condition nominal；不使用依赖本phase新增训练的--job-dir。新phase不能覆盖W45；合同execution.background_job.wall_deadline_enabled=false，后台deadline=null，不传--deadline-unix。
9. 每流原E30通过才同case独立E60→E300→E600。E30满30s且合法实际材料去程、远端/转向，不要求完整返程；E60满60s且完整合法往返/回家稳定2s；E300/600在270/540s内至少5/10合法往返且完整观察和原长测稳定条件。训练60s采集不是E60。失败结束当前attempt，后级blocked，不换seed补到成功。
10. 原Go2/TM02、mass15.206408kg、PD60/5、每腿23.7/23.7/45.43Nm、绝对主应变域.02、非足接触/跌倒/合法路线、声明步态、r28 trunk/r27 C2/r26 packet、冻结PANN/658D/2Hz cap/W38奖励不放宽。合法去程必须完整足迹实际Fz>0，不能仅看body进入；不能以驻留、较低成本或程序退出码代替资格。

## 两CPU与第三GPU并行

11. 默认D57，SSH autodl-d57；CPU树/root/autodl-tmp/MCP-RL-w42，GPU树/root/autodl-tmp/MCP-RL；环境work/envs/d57和work/envs/w41-mjwarp。先核现场cgroup/affinity、内存/GPU/磁盘、tmux/PID/job/clock与未提交修改。已有有效W46则接续，不重复启动，不覆盖并发修改或宽前缀kill。
12. 默认CPU A/B各10workers，全进程树affinity隔离，Torch/BLAS/OMP各1；GPU总host2CPU含native确认，约3CPU给系统/主控/发布。有限比较最多两个10CPU求解同时运行，结束后才双E30；build-j6在双流重负载前。继承XLA_PYTHON_CLIENT_PREALLOCATE=false、JAX_ENABLE_X64=true；PANN/Actor也共卡，不以占满96GB衡量成果。
13. 第三任务只做四个W45上下文、每个baseline/selected×原初态/±偏差，最多24scenario。先同state完整action map，再20ms配对推进；K/H/切空间/原投影等语义完整且约束判断一致才扩一次350ms。最多42000完整world-step，另计最多2400短窗、预热和缓存。FP64动作层转入FP32 MJWarp dynamics时明确cast，不能宣称单改dtype得到等价引擎。缺失历史incumbent不造；不等价就有限负结果结束，不变成G5/新引擎项目，不阻塞CPU。
14. 可写upstream/build隔离，prepare不经共享可写软链；环境/静态资产/已完成父输出只读复用。第三任务自然完成后两任务运行即可，不为利用率启动第三PPO或重复失败任务。

## 后台、记录与GitHub交付

15. **所有新后台无固定墙钟完成期限、无到点kill。** 保留物理30/60/300/600秒、有限数值预算和网络单次timeout；按有限工作完成、原任务结果、明确技术失败或用户停止结束，不无限重试。
16. 存活进程暂停可按精确PID恢复；未完成而技术退出的attempt为interrupted，已自然完成的成功/物理失败/task-unmet保留原结果。无完整plant/材料/observer/MPC/任务状态时Actor不能恢复物理中途，必要时另建独立attempt，不覆盖或拼接资格。
17. 先推实际实现/spec/有限证据/真实phase，再启动并确认两条物理时钟/记录增长。保留共同参考/R、完整policy身份、J0/J_ref、alpha/scale、首20ms状态与实际命令/力矩、高率关键子步、材料加载/合法路线；大数组按事件引用，不复制冗余，不删历史。准确区分计划、构建、运行、资格与交付。
18. 紧凑Git先发、请求原件增量Release后发，核commit/URL/size/digest。Git/index短写串行合并最新main，不强推不覆盖并发结果；never-generated保留明确身份，不造空壳。
19. **GitHub核验后不在用户电脑额外下载、打包、镜像或归档；不自动删除服务器checkpoint/resume、原件、视频或上传副本。** D57上传仅已授权服务器学术代理，禁止用户电脑VPN SSH转发，不自动切电脑/Clash/直连。凭据不进日志/Git。

最终报告实际commit/build/spec、选择分支、job/PID/clock、固定父策略、每级原评价、实际命令与恢复是否改善、剩余科学问题、GPU语义与成本、Git/Release及有效续接点。原任务未通过如实结束有限轮次，不以数值修复或数据发布代替稳定材料往返。
