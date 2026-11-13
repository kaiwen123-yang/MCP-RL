# Codex执行入口：W47已完成；W48部署最小修复并验证原闭环

<!-- W48_EXECUTION_BEGIN -->
> 2026-11-13 完成核验：W48 正式 S5-r36 已完成有限实验及原 A/B E30。A 26.432 s 躯干触地，B 11.638 s body_fall，E30 0/2，后六级 blocked，当前无活动计算。与 W46 对照，A/B 的时间、qpos/qvel、实际目标、力矩、GRF、接触、cap 和 phase 逐元素相同，选中候选与 alpha 逐调用相同；最小数值修复未改善这两条原任务。1904 条非启动规划边界均有上一实际部署 accepted 预测配对，首次大 held 跳变与膝限幅反转完整事件已保存。有限 8 solve/4 部署核对、原件和两段原任务视频已核 Release size/digest；A 上传曾 TLS 失败，本次仅续传既有清单完成。新 PPO 0/0、新 GPU physics 0、donor solve 0。不自动重开任务。[W48 最终索引](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w48/phase.json)。下方已发布计划和旧 running 观察按其形成时点阅读。
<!-- W48_EXECUTION_END -->

更新2026-11-13（Asia/Shanghai）。结果基点`6e9075f48718509d48f51d08d0a8f62a34edfb41`。**W47有限工作与交付已完成；W48为planned_not_implemented，正式S5-r36尚未构建/部署。** 本次审查没有启动新服务器任务、native求解、plant、PPO或GPU物理。用户最新要求覆盖历史固定墙钟截止和电脑归档规定。

先读[项目记忆](docs/TERRAIN_MEMORY_PLAN.md)、[W47最终审查](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W47-final-review/README.md)、[W48详细计划](docs/S1_W48_STABILITY_PLAN.md)、[完整启动提示词](docs/S1_W48_START_PROMPT.md)、[机器交接](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W47-final-review/handoff.json)。历史计划只按形成时点阅读，不按旧“当前”重开已结束任务。

## 当前事实与本轮目标

- W47已定位四元数自差浮点首扰动；exact self/antipode StateDiff修复后四个保存问题alpha0精确再生nominal。8条350ms固定策略诊断＝1400held/14000个.2ms子步，独立时长合计2.8s；新optimizer solve/transition FD/donor/plant/PPO/GPU physics全0。该诊断不是正式r36或稳定资格。
- 最新原任务仍W46/r35 constant：A E30 26.432s floor–trunk contact，B11.638s body_fall；合法去/返/往返0，E30 0/2，后6项blocked。W47没有新增原任务失败。各14737/update57、resume ep00428/ep00417、tail0/excluded145，新PPO0/0。
- W47新全程CSV覆盖1944calls：1601正步安装（1598同问题J下降、3风险优先），343停滞。B明显下沉/侧倾前一秒各50/50正步下降。旧四fixture540正候选全拒只指这些旧困难问题，不能外推全程。
- A无实际材料加载，B明显下沉/侧倾先于首次完整足受载；当前需要稳定移动、支撑与恢复。未证材料、GPU、PPO不足、QP失败或deadline是唯一原因。
- W48只将已证最小修复接入正式搜索/部署，再各一次原A/B；同时补齐记录。donor缺件、最新W46历史context/预测缺失、旧fixture无明显前移，不再阻塞correctness-only评价。原成功/失败/材料/路线标准不放宽。
- 长期仍是稳定原材料任务、跨足/返程有效历史曝光、匹配RAW/PANN和冻结/打乱历史因果对照、候选内材料状态传播与必要动作权限、泛化及Go2/NX。

## W48实施与原任务

1. 核当前main、未提交修改、有效tmux/PID/job/实际物理clock及资源。有效W48存在则核身份续接，不重复启动、不覆盖并发、不重启W45/W46/W47/G5，不宽前缀kill。
2. 控制唯一变化是`w47-state-difference-identity.patch`：严格相同/精确反号free/ball quaternion自差为0，无epsilon邻域、K裁剪或真实小旋转归零。它不改MuJoCo引擎mjd_transitionFD。记录改进单列且不改控制/任务时钟。
3. 冻结真实r36 spec/patch/build/source/库SHA，接入prepare/runtime/policy/R29Records/evaluation contract，尤其原`controller_build == 'S5-r35'`分支与build白名单。N/r34 strict/constant及原参数需真实读回；--build字符串不能代替接入。正式任务禁带SCORE_ONLY、W47_INDEXED_SCORE和离线INPUT_POLICY/CONTEXT/REFERENCE，不能把诊断binary更名当r36。
4. 一次不积分StateDiff核self/antipode/真实小旋转及位置/速度/act保留。四个已有完整A148/A153/B155/B158固定同P0/x0/time/private context/model/constant ubar/F/R/预算，每例一个corrected N solve；身份可核复用old，否则每例最多补一个old，总≤8solve。无none/load/R/seed网格，不重复W47八条零端点。
5. 每例corrected最终完整policy按真实ActionImpl/scale/插值/归一化/H-QP做一次350ms部署诊断，总≤4条/7000子步，对照同策略实际评分轨迹。scale≠1 nominal不强用indexed scale1；alpha不二次加，accepted rollout不覆盖反馈reference。
6. 每次planner_step覆盖旧计划前保存actual/context/P0身份及上一实际部署策略accepted rollout同origin预测，切空间和速度误差分单位。无匹配节点标unavailable，不取本次新解、不填零/借旧delta。完整事件包保留pre-call state/act/time/mocap/userdata/private gait/material、完整P0和winner reference x/u/time/K/H/scale/metric、accepted rollout、上一预测、同epoch r35参考/任务/观察身份。首实际held>1rad、首膝±45.43Nm反转、阶段和terminal按前.1/后.2s保存，阈值仅选诊断；重叠合并/hash去重，不全call复制所有候选导数。
7. 构建/有限结果/正常入口/原约束/schema与输出路径正确、无未解决技术错误，先推spec/证据/真实phase，再自主启动A/B各一次r36 evaluation-only原E30，不等donor或旧fixture移动成功。记录在同一attempt首次正常规划后、首个held物理推进前核实际写出和身份；首次无上次预测合法unavailable，不要求启动前有runtime记录、不另开plant测试记录。缺关键字段保留技术失败并修复。
8. A/B固定各自W44 seed1/2 checkpoint-014737.pt及完整resume ep00428/ep00417，actor/critic/optimizer/norm/RNG/fork/lineage保留，各14737/57、tail0/excluded145，新PPO0/0；权重固定，observer/material history继续因果更新。不自动执行未来15249/59。用run_s1_go2_r12.py单次evaluation入口，显式attempt/evaluation-start-unix/checkpoint/level/seed/workers/build/contract/condition nominal，不用要求新训练的job-dir。
9. 每case过E30才E60→E300→E600。E30满30s合法材料去程及原3m目标容差/转向，不新增x≥3.0；E60满60s一合法往返/回家稳定2s；E300/600各270/540s内5/10合法往返及完整观察。失败结束该case、后级blocked，另一case可继续；不换seed、重复同配置追成功或拼接资格。已有W46作历史对照，不补旧r35 E30。
10. 保持Go2/TM02质量15.206408、PD60/5、每腿23.7/23.7/45.43Nm、主应变.02、非足接触/跌倒、合法路线、2Hz/.45/.005 C2步态/.35s horizon、r14/r26/r27/r28和658D/2Hz cap/W38奖励。短腾空合法，raw Kδ、描述用.2m/.5rad不成为硬判据。物理失败不改写为技术失败重跑；仅Actor不能恢复完整plant，技术中断独立attempt标记。

## 独立第三线、资源与交付

11. C原位分析现有GPU六组重复执行身份逐时差/首分叉/隐藏初始化可识别性，主体0新增physics。旧源是W45策略，不称r36排序。旧s1_go2_w46_gpu.py硬写W46，禁止直接重启；新入口独立W48输出。可选一次同状态FP64动作批次名义≤24、缺扰动不凑、去重，0积分；policy生成build与动作计算build分列，self/antipode基于native同一插值归一化reference，raw/clip/held独立。接口不足据实结束，不扩长窗。
12. R8 asset681937433仍uploaded：yaw_plus5 trajectory 28100505B，SHA079f57e1ebad260c9cdac842dc089012e29cb46992ab64d2c31cd9d380b61625。D57缺件可只恢复这一已知科学输入并核schema/time/phase，不下载整批、不重跑R8；本轮donor solve0/在线多家庭0，C不阻塞A/B。只用授权服务器学术代理，不借电脑VPN/SSH转发、Clash或未授权直连，凭据不进Git/日志。
13. 默认autodl-d57，CPU树/root/autodl-tmp/MCP-RL-w42、GPU树/root/autodl-tmp/MCP-RL，环境work/envs/d57、work/envs/w41-mjwarp。现场核25CPU/96GB GPU实际余量及无关作业，不动SuperMimic等无关进程。两个有限组/随后A/B各10CPU，全进程树隔离；C总host2，约3给系统/主控/发布。线程库各1，build-j6先完成，共享资产只读、可写upstream/build隔离。两原回合外不再挂重型solve。XLA预分配false、JAXx64true，显存按实际峰值/剩余准入，不为占用开新PPO。
14. **所有新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix，无固定墙钟完成时刻/到点kill。** 工具等待timeout不传成任务期限。保留原物理时长、有限工作量、技术失败/用户停止和单次网络timeout，无无限重试。对话结束不杀后台、不把running写completed；核真实clock/记录增长，存活进程只按精确PID暂停/恢复。
15. 紧凑Git优先、已生成所需原件D57增量Release直传，核commit/URL/size/digest；回执丢失先查已有资产，不重复上传。Git/index短写串行合并最新main、不强推或覆盖并发。未生成/blocked/中断/真实失败分开，不造空文件。
16. **GitHub核验后不在用户电脑额外下载、打包、镜像或归档；服务器checkpoint/resume、原件、视频与上传副本不自动删除。** 单个必要服务器科学输入恢复不改变该要求。

最终给实际commit/spec/build/源码SHA、正步/部署结果、每级原资格、首偏差证据、新rollout/FD/physics/PPO、资源与GPU价值、Git/Release和真实续接点。原任务未过明确未过，修复正确不等于稳定通过。
