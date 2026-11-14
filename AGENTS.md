# Codex执行入口：W48已完成；W49分离真实执行与恢复机制

更新2026-11-13（Asia/Shanghai）。审查基点`61e16975fa6b38dc8e2d31779c2bd82f1aaeb8b6`。**W48正式r36及原评价已完成；W49有限机制与短部署已执行，S5-r37/K原A/B均完成30s，但材料穿越未完成，E30 0/2，后六级blocked。** 最新实际身份见 results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w49/phase.json；以下预算继续生效。用户最新要求覆盖历史固定墙钟截止和电脑归档规定。

先读[项目记忆](docs/TERRAIN_MEMORY_PLAN.md)、[W48审查](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W48-final-review/README.md)、[W49详细计划](docs/S1_W49_STABILITY_PLAN.md)、[启动提示词](docs/S1_W49_START_PROMPT.md)、[机器交接](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W48-final-review/handoff.json)。预算、具体身份和判据以W49详细计划为准；历史“当前”只按形成时点阅读。

## 已确认状态

- r36 identity修复已部署；binary SHA `b05c8da8ca74551dc73ea0272329234217d7160731e5cea8e33101bba3241338`。旧四fixture零端点修好，但240候选policy配对、赢家相同，每侧180正候选仍真实J增加。结束旧fixture和identity-only原回合重复。
- W48原A26.432s躯干触地、B11.638s body_fall；物理轨迹/选择与W46相同。E30 0/2，后六级blocked，合法去/返/往返0。A无实际材料加载，B下沉/侧倾先于首完整足受载。无deadline造成这两次失败的证据。
- 原1944calls有1601正步安装，不能说全程没有下降。新增1904上一accepted预测配对、277事件call/995对象；不为已有事件再重跑W48补快照。
- 四零plant启动attempt是配置继承丢17键，已修；部分startup native工作未知，不写所有计算0。科学13资产已核，旧技术A/B各一个ZIP仍待按原manifest续传。
- 长期是稳定原材料任务→跨足/返程有效历史→RAW/PANN及冻结/打乱历史因果对照→候选内材料传播和必要动作权限→泛化/Go2/NX。当前2Hz scalar cap不能直接做足点/支撑/关节恢复。

## 实施顺序

1. 核最新main、未提交修改、D57有效tmux/PID/job/真实clock和资源；有效W49按身份续接，不重复启动、不重开W45–W48/G5、不宽前缀kill、不动无关作业。默认CPU树`/root/autodl-tmp/MCP-RL-w42`、GPU树`/root/autodl-tmp/MCP-RL`；环境`work/envs/d57`、`work/envs/w41-mjwarp`。
2. 完整r36配置复制后显式overlay，核生成模型必需numeric、source/model/library/protobuf与实际环境，防止重复丢配置。不另开plant测试配置，正式原任务禁离线score/import环境。
3. 使用W48新事件A213@3.84、B176@3.10、A363@6.84及邻接364@6.86、B328@6.14。A诊断用6.84区间覆盖首反转，B候选分析可用364完整问题，不能冒称同一历史solve。固定实际policy/accepted/P0/context/同epoch reference。
4. CPU A先0积分同态查询，分开SetState double、显式float querytime、native raw/clip/H-QP、RPC float和actual held。W39/W40已知精度，不冒称新根因。time=None仅单一对照，量级不足就结束支线。必要诊断≤12条20ms/1200子步、0solve/FD；同held predictor、同policy真实接口predictor、同held可核plant共同重建。缺外部全态取消对应臂，不填零、不称历史exact replay、不重开整轮。
5. CPU B读accepted全窗/J分项/真实接触，分清计划本身不良和反馈后恶化。≤4个新问题N solve，首有效backward方向各9正alpha＝≤36额外候选，必要同路径baseline≤4条。保留baseline反馈的架构r33 P已做；本轮仅用新问题k替代旧手工正弦方向。旧reference时标与k当前时标可能不同，禁止按index相加或以accepted覆盖reference；按W49计划扩完整策略表示并核部署。
6. 原dV对新家族无效时expected/ratio=null、保持mu/rate，沿用P既有语义及原r34风险/真实成本/roundoff预算/alpha0不安装，不放宽strict。none/load/constant是参考目标维度。A若已证实际执行/参考接入错误，可优先最小修复替代该家族；正式版本只一个控制机制。
7. 新策略≤4条20ms/400子步真实接口核对。正式版本需真实spec/patch/build及prepare/runtime/policy/records/合同接入，候选在原预算中分配/替换，实际成本明示。至少一个新代表问题有原规则可接纳且行为不同的策略，或修正实际影响动作的已证错误，且入口/原约束/短部署正确后，自主A/B各一次原E30；不要求先整程成功、两流都恢复、不等donor/GPU。仅极小舍入变化且主要行为未变，不重复原任务。
8. A/B固定各自W44 seed1/2 checkpoint-014737、resume ep00428/ep00417，完整actor/critic/optimizer/norm/RNG/lineage保留，各14737/57、tail0/excluded145；新PPO0/0，observer/history因果更新。用单次evaluation入口，明确attempt/evaluation-start-unix/checkpoint/level/seed/workers/build/contract/condition nominal，不用要求新训练的job-dir。
9. 每case E30通过才E60→E300→E600；保持原30s合法材料去程/3m容差转向，60s一合法往返/回家稳定2s，300/600各270/540s内5/10合法往返及完整观察。失败结束本case并blocked后级，另一case可继续；不换seed、追成功重跑、拼接资格。Actor不能恢复完整中途plant，技术中断另列attempt。
10. 保留Go2/TM02质量15.206408、PD60/5、每腿23.7/23.7/45.43Nm、主应变.02、原触地/跌倒/路线、2Hz/.45/.005 C2 gait、.35s horizon及原机制。短腾空合法，raw Kδ和描述.2m/.5rad不是新终止判据。无PPO、成本/步态/seed网格或整套新求解器移植。
11. 新原回合继续W48预测/事件去重，补trunk_restoration；顺带保存integration及外部callback/material全态，准确注明API覆盖，不另开原回合只补记录。同attempt首次正常规划后、首held推进前核写出；startup无上一预测可unavailable。

## 第三线与工作流

12. C一个B158 nominal双world，同FP32 held：F1完整S0、L恢复第一个.2ms后的S1再旧部分reset、F2完整恢复S0。60 GPU world-step＋必要预热≤2，总≤62。逐子步读回、同dispatch；辅助KKT缺失不自动阻塞，必要输入不等停积分。条件触发≤8唯一身份真实native动作对照、0physics。按[细则](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W48-final-review/w49-task-c-design.md)，不重跑24×350ms，不称native等价或正式排序。
13. 旧技术A/B ZIP只在D57按原manifest续传，先查远端，不重打包/实验，不阻塞主线。R8已知单输入可独立恢复核schema/time/phase，本轮donor0，不批量下载/重跑、不作为总门。仅用已授权服务器网络，不借电脑VPN/Clash/SSH转发或未授权直连。
14. 实测25CPU/96GB余量：两CPU线各10，C总host2，约3给系统/主控/发布；线程库1、全进程树隔离、可写build隔离，build-j6先完成。原A/B外不挂第三重native solve；XLA预分配false、JAXx64true，显存按实际余量/峰值准入，不为占卡加PPO。
15. **全部新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix，无固定墙钟完成时刻/到点kill。** 保留有限工作量、原物理阶段、技术失败/用户停止与单次网络timeout；工具等待不变任务期限，无无限重试。对话结束不杀后台，running不改completed。
16. 紧凑Git先推，已生成必要原件/视频D57增量Release直传，核commit/URL/size/digest；回执丢失先查远端。Git短写串行合并最新main、不强推覆盖。
17. **GitHub核验后不在用户电脑额外下载、打包、镜像或归档；服务器原件/checkpoint/resume/视频/上传副本不自动删除。**

最终区分实现正确、动作改变、有限恢复、原资格和记忆收益；给实际工作量、失败/blocked/未运行原因及真实续接点。原任务未过明确未过，不把计划发布写成实验已成功。
