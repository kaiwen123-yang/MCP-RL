# W50 Codex启动提示词

在MCP-RL仓库按最新main自主实施W50，完成后推送GitHub并给出真实状态。先读AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md、results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/README.md、handoff.json、docs/S1_W50_STABILITY_PLAN.md。详细公式、固定问题、预算和资格以W50计划为准；不要重复把本轮变成只做计划或状态报告。

结果基点de6606dc8363e8ab1c816fa682cace752a06318a。W49正式S5-r37/K原A/B均完成30s，但E30 0/2、材料穿越未完成、合法往返0；A/B最低z .123298/.130666、末z .138718/.142818，早已非足受载。A无材料加载，B首次足加载12.1866s、机体west-entry20.064s。当前是低姿态与前进失败，不把30s观察写成稳定行走。W49及旧技术原件和视频已发布，A上传恢复也已完成；不要重开旧原回合或重复续传。

先核最新main、未提交工作、D57当前tmux/PID/job/真实clock/资源。有效同身份W50继续，不重复启动；不宽前缀kill、不动无关作业。默认CPU树/root/autodl-tmp/MCP-RL-w42，GPU树/root/autodl-tmp/MCP-RL，环境work/envs/d57与work/envs/w41-mjwarp；以现场有效路径为准。核25CPU与96GB实际余量，10+10+2+3分配，BLAS/OMP等线程1，全进程树隔离，build最多-j6先完成。

用户补充B视频已确认4.4–5s反复深蹲/顶起、7–8s后长期低伏折腿/外伸、后半贴地挪动。首要目标是维持起步后的有效足部承重与前进、预防最早步态失效；低姿态恢复次级。读同目录video-review.md，不能以存活30s或低伏挪得更快称稳定。

并行A/B/C：

A在D57直接读取W49现有boundary/held/policy/accepted/P0/context/r35原件，零新增物理分析。从2.002s Stand→outbound切换开始核实际private gait/phase、脚顺序与足参考、Fz/接触交换/支撑脚速度/非足承重；不能只看配置Trot。确认reference起止时刻、K/H实际生成来源、query越过旧grid的时间与首下沉/非足承重/饱和/非进展。安装ID/generation_time/metric_source_plan_id不是反馈年龄。源码显示K只加delta而reference/K/H持续继承；约29.65s过期只是待实测推断。精确r26 userdata84:96必须读，mission target x=3不是native Position目标。新integration+部分外部态不等于完整plant checkpoint，缺失字段不填零、不假称历史exact replay。

B实现一个主候选：当前horizon重参考，并用同一个原alpha转移前馈/反馈/H，拟S5-r38。取当前best scored完整B，沿B本次native rollout current state/time网格求pre-clamp/pre-QP raw center c_B（含旧delta及installed scale）；在当前MuJoCo切空间求其局部Jacobian L_B，不能直接搬旧quaternion坐标K。只到首个有效N backward，在任何alpha改写前同时capture k_N/K_N/H_N/μ/active。每个原正alpha：u_ff=c_B+alpha*k_N，K=(1-alpha)L_B+alpha*K_N，H=(1-alpha)H_B+alpha*H_N；新reference为当前rollout、feedback_scaling=1、delta清零。K_N实际scale=1；L_B已含旧scale，不重复乘。alpha0直接deep-copy原B，不rebase、不新rollout、不安装；正alpha只有局部一阶等价，必须实际native评分，不宣称全局连续/恒等。混合H记录两端来源与alpha，不伪造单一μ或新Qu/Qxu。

固定W49四例：A237@4.32、B236@4.30、A328@6.14、B625@12.08；不是W48旧四fixture。四例各一次首有效方向solve、9正候选共36，保留原风险/J/roundoff/strict和赢家原对象安装，expected/ratio=null。finite与online同一入口，不先完整N再追加家族。原same-epoch r35缺失则共同重建身份明示；19+I_freeze+I_extra_baseline≤20，每例至多20、四例≤80full rollout，FD及零积分坐标查询另列。

最多两条40ms真实RPC predictor部署，A/B各取最早eligible候选；起点装winner，+20ms各正常replan一次，≤2extra solve/40full rollout/400子步。一次性导入后真正禁用QUERY_ONLY/INPUT_POLICY/R34_CONTEXT/R35_REFERENCE，避免新epoch再读旧time/gait；必要用明确的第二Agent接入，不说这是任意plant恢复。homotopy安装记录current reference/混合来源，nominal-scale或incumbent仍可旧来源，不以policy_updated推断刷新。不要求两流先都恢复或有限窗先完成全程。

以A237/B236早期例为首要依据，至少一例原strict认可实质动作/足部承重或有效前进改善（或修正已证早期步态接入错误），不能只靠晚期低伏挪动更快，且实际部署/来源正确，冻结同一个正式版本后自主并行原A/B各一次E30；只有roundoff或metadata变化不重跑，四例无用则交有限负结果与原因，不临时叠加新权重/步态/horizon/求解器。cost、接触权限与终端延续仅为后续条件方向。

正常重编所有受policy/header ABI变化影响的service库及server；保持先序列化原sensor模型再载入诊断predictor的修复。复制r37完整配置后overlay，更新真实spec/patch/build/prepare/runtime/policy/records/合同入口，禁止离线标志进入原任务。不新增全仓hash、mock/shape/import/smoke或无信息量防御测试；只验证这次改动和原任务所需链。

原A/B固定各自W44 seed1/2、checkpoint-014737、resume ep00428/ep00417，完整actor/critic/optimizer/norm/RNG/祖先保留，各14737/update57、tail0/excluded145；新PPO0/0，observer/history因果更新。使用单次evaluation入口，不用会新训练的job-dir。原Go2/TM02/PD/力矩/.02材料域/2Hz/.45/.005 gait/.35s horizon/原route与失败判据不变。

每case E30通过才E60→E300→E600：30s合法材料去程及远端原容差并完整观察，60s合法往返/回家稳定2s，300/600分别270/540s内5/10合法往返及完整观察。失败结束本case并blocked后级，另一case继续。不换seed、不追成功重复、不拼接资格。技术中断与物理/路线不合格分开。

C不阻塞CPU：先零新积分读旧all-exposed-substeps.npz（1873543B）与锁版本MuJoCo-Warp4.0.0/Warp1.19.0源码，按有效contact/nefc/world布局沿constraint/solver/reduction定位首差，排除padding；不再重复相同F1/L/F2。只有具体修复后才一次≤20GPU world-step区分探针，无额外预热/吞吐轮。微小非逐位相同不自动不可用；当前缺完整native cost，GPU不主排序。可批量做现有K/H/切空间映射等零物理GPU分析；不为占满96GB新增PPO或新采样控制器。

所有新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix，不设固定墙钟完成时间、不到点kill、不因对话结束杀后台；有限工作量、原物理阶段及单次网络timeout保留，无无限重试。继续记录实际运行中状态、原始失败和未运行原因。

紧凑源码/结果先推GitHub，必要原件和视频从D57增量Release直传，回执缺失先查远端，Git短写串行合并最新main，不强推覆盖。核验真实commit与资产后，不在用户电脑额外下载/复制/打包/镜像/归档；不自动删除服务器原件、checkpoint/resume/视频或上传副本。最终给实现、实际工作量、原资格、未完成原因、远端位置与准确续接点，不把计划/短诊断写成成功任务。
