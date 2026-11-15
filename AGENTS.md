# Codex执行入口：W50有限负结果已完成，原评价未启动

更新2026-11-15（Asia/Shanghai）。W50/S5-r38/H 已实现并完成有限工作：四个W49问题76次full rollout，36个正候选中10个strict改善；两个早期赢家仍下沉/后退，未达到原E30推进依据。另两条40ms消耗400子步和2次再规划40rollout，但bridge同SONAME库复用造成ABI失配，修复loader后未重跑。原A/B E30均未运行，后级blocked；新增PPO0/0，新增GPU积分0。

先读[项目记忆](docs/TERRAIN_MEMORY_PLAN.md)、[W49最终审查](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/README.md)、[W50详细计划](docs/S1_W50_STABILITY_PLAN.md)、[启动提示词](docs/S1_W50_START_PROMPT.md)、[机器交接](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/handoff.json)。公式/问题/预算/判据以W50计划为准，历史“当前”只按形成时点阅读。

## W50实际执行状态

实际结果与预算以[W50运行索引](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w50/phase.json)为准。当前仅完成紧凑结果和必要原件的GitHub发布/核验；原E30没有资格入口，不启动或重跑。两条短部署预算已消耗，不因loader修复刷新预算。以下W50约定保留为本轮协议与历史依据，不再把已完成四例重新启动。

## 已确认事实

- 原A/B最大x .349667/1.020530m，最低z .123298/.130666m，末z .138718/.142818m；首次非足受载4.6232/5.0216s。A无材料加载；B足端12.1866s、机体west entry20.064s。不能把30s观察或两个30s拼成稳定60s任务。
- W49四问题36个K候选中25个原strict改善，四事件均有行为改变，4×20ms真实RPC部署400子步。新PPO0/0。原任务变化真实，完整路线未过。
- 82次查询已确认native-vs历史、RPC-vs实际held一致；旧时间量化不再主攻。有限binary与正式binary身份不同且有ABI/metadata修复记录；正式binary8637e3d…正确运行。
- r37有效bootstrap后K只加delta、nominal只改scale，reference/K/H跨epoch持续继承；安装ID/age/metric_source_plan_id不能证明反馈已更新。精确reference_end与K/H实测须读现有NPZ，29.65s过期目前只是源码推断，不宣称唯一根因。
- 两次query-only失败、一次deployment Init中断均physics0；两原E30为自然结束后资格未满足，另六级未运行。internal_stagnation=0只匹配旧枚举，不代表没有停滞。
- W49所有原件与视频、W48两技术ZIP已发布，A上传恢复已完成。build/phase旧N/not_started/live字段按历史快照读，不据此重跑。

用户补充的W49 B视频已核：4.4–5s明显下蹲/顶起、7–8s后长期低伏、折腿/外伸并贴地挪动。首要目标是预防起步承重和步态先失效，低姿态恢复次级；[视频证据](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/video-review.md)。

## 执行顺序和唯一主机制

1. 核最新main/本地修改/D57有效tmux/PID/job/真实clock和资源，有同身份W50就续接，不重复启动、不重开W45–W49/G5、不宽前缀kill、不动无关任务。默认CPU树/root/autodl-tmp/MCP-RL-w42、GPU树/root/autodl-tmp/MCP-RL；环境work/envs/d57、work/envs/w41-mjwarp，以现场为准。
2. A读D57已有全量boundary/held/事件对象，零新积分从2.002s Stand→outbound核实际private gait/phase/足参考与真实接触交换，再分析生命周期、首次下沉/非足支撑、J分项、真实r26 userdata84:96及当前策略下20ms预测误差。B同时实现一个重参考/反馈机制；缺历史best nominal大数组不阻塞，当前B由已预算nominal screening产生。P0、B、accepted和feedback reference各自分清。
3. 固定W49 A237@4.32、B236@4.30、A328@6.14、B625@12.08，四例各一次到首有效backward即同时capture k/K/H/μ/active，不能先完整N再混首k与末K/H。不重用W48四fixture冒充新问题。
4. 新candidate基于B本次rollout当前网格：u_ff=c_B+alpha*k_N，K=(1-alpha)L_B+alpha*K_N，H=(1-alpha)H_B+alpha*H_N；c_B为含旧delta/scale的pre-clamp/pre-QP raw center，L_B为当前MuJoCo切空间Jacobian。不能用projected actions代替center，也不能直接搬旧四元数参考K。新scale=1，K_N实际scale=1，delta吸入center后清空；同一个原alpha，不设beta网格。
5. alpha0直接deep-copy旧完整B、原评分复用，不rebase/新rollout/安装。正alpha仅局部近似，实际连续ActionImpl/native完整风险/J/roundoff/strict评分；赢家原对象部署。expected/ratio=null，不借旧dV调μ；保留原风险进展语义。记录混合H双来源/alpha，不造单一μ或把旧Qu/Qxu当新导数。
6. 四例9候选共≤36，nominal与必要baseline总≤80full rollout：每例19+I_freeze+I_extra_baseline≤20；若缺同epochr35只作明示共同重建且新Freeze时复用已有B score。FD与零物理raw-center求值另列。
7. ≤2条40ms实际RPC native predictor部署，A/B各最早eligible，起点装winner、20ms各正常replan一次；≤400子步/≤2extra solve/≤40full rollout，有限合计≤6solve/120full rollout，FD另列。一次性导入后消费QUERY_ONLY/INPUT_POLICY/R34_CONTEXT/R35_REFERENCE或明确第二Agent当前state/gait接续，不能新epoch读旧time/私有gait。homotopy安装才更新对应来源，nominal-scale/retained仍可旧；不强制安装劣解刷新年龄。
8. 以早期A237/B236为首要依据，至少一个有原strict认可的实质动作/足部支撑或有效前进改善，或修复已证早期步态接入错误；不能只凭晚期低伏挪得更快，实际入口/原约束/生命周期正确，即可自主冻结一个正式版本后A/B各一次原E30。不要求先整程成功、两流都恢复或等待GPU/donor；只roundoff/metadata变化或四例全无用则交有限负结果，不重复旧轨迹追成功。
9. 完整r37配置复制后overlay；改policy/header ABI需正常重编service静态库/server。先序列化原sensor模型再载入诊断predictor。实现真实spec/patch/build/prepare/runtime/policy/records/合同入口，正式任务禁离线环境。只验证本改动必要链，不新增全仓hash/mock/shape/import/smoke或无信息量防御性测试。
10. A/B仍各自W44 seed1/2 checkpoint-014737、resume ep00428/ep00417，全actor/critic/optimizer/norm/RNG/祖先保留，各14737/update57、tail0/excluded145；新PPO0/0，observer/history因果更新。用单次evaluation入口，不用自动新增训练的job-dir。
11. 每case原E30通过才E60→E300→E600；30s合法材料去程/到达3m原容差且完整观察，60s合法往返/回家稳定2s，300/600各270/540s内5/10合法往返和完整观察。失败blocked后级，另一case继续；不换seed、不追成功重跑、不拼接资格，技术中断另列。
12. 保留Go2/TM02质量15.206408、PD60/5、每腿23.7/23.7/45.43Nm、主应变.02、原route/接触/跌倒、2Hz/.45/.005 C2 gait、.35s horizon。只有本主机制，无成本/步态/horizon/PPO/求解器多项叠加；候选内材料历史与动作扩权是后续条件方向。
13. 继续事件去重，分开记录reference/K/H生成来源与完整安装ID、实际前进/承重/高度、cost/risk、同策略预测—执行、各类stop和work。W49 integration+部分外部态不等于任意plant/actor/native恢复；缺失不填零、不新开原回合补历史档。

## 并行、后台和交付

14. 实测25CPU/96GB余量，A/B各10核、C2核、约3核系统/主控/发布；线程库1、全进程树隔离、可写build隔离，build-j6先结束。四独立问题可两线并行，最终同版本原A/B并行，不再外挂第三个重native solve。
15. C先零新积分读W49既有1873543B全暴露子步NPZ，按active布局排除padding，沿constraint/solver/reduction和锁版本源码定位。具体修复后最多一次20 GPU world-step含预热区分探针；不重跑同F1/L/F2，不追bitexact作为总门，缺完整native J时GPU不正式排序。可做已有矩阵的有目的GPU批分析，C不阻塞主线，不为占96GB加PPO。
16. **全部新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix，无固定墙钟完成时刻/到点kill。** 有限工作量、原物理阶段、技术失败/用户停止及单次网络timeout保留；工具等待不是任务期限，无无限重试。对话结束不杀后台，running不改completed。
17. 紧凑Git先推，必要原件/视频D57增量Release直传核commit/URL/size/digest；回执丢失先查远端，Git短写串行合并最新main、不强推。旧W49交付已完成，不重复实验/上传。
18. **GitHub核验后不在用户电脑额外下载、复制、打包、镜像或归档；服务器原件/checkpoint/resume/视频/上传副本不自动删除。** 只用已授权服务器网络，不自动借电脑VPN/Clash/SSH转发。

最终区分实现、动作改变、局部恢复、原资格和记忆收益；给真实工作量、失败/blocked/未运行原因、commit/资产和准确续接点。原任务未过明确未过，不把计划发布写成实验已成功。
