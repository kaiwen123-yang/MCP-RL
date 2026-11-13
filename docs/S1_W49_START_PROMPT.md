# W49 Codex启动提示词

以下全文可直接交给有D57访问能力的Codex执行。当前发布仅完成审查和计划；W49尚未实施。预算与详细契约以本仓库`docs/S1_W49_STABILITY_PLAN.md`为准，不把提示词节选扩成新的实验网格。

```text
继续MPC+RL项目W49。先拉取并核对kaiwen123-yang/MCP-RL最新main，读取AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md、docs/S1_W49_STABILITY_PLAN.md，以及results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W48-final-review/全部审查与handoff.json。结果基点61e16975fa6b38dc8e2d31779c2bd82f1aaeb8b6；W48已结束，不是待实施。不要重开W45/W46/W47/W48/G5。

先核D57工作树、有效tmux/PID/job、真实物理clock/记录增长和25CPU/96GB实际余量；已有有效W49就按身份续接，不重复启动，不动无关作业。CPU树默认/root/autodl-tmp/MCP-RL-w42，GPU树/root/autodl-tmp/MCP-RL，环境work/envs/d57和work/envs/w41-mjwarp。用户已授权按本计划实现、有限实验、满足条件后原评价与GitHub交付，不因普通可逆实现选择再次停下来确认。

接受W48结论：正式r36 identity修复已部署，binary SHA b05c8da8ca74551dc73ea0272329234217d7160731e5cea8e33101bba3241338。四个旧fixture赢家相同，原A26.432s躯干触地、B11.638s body_fall，轨迹/动作/选择与W46相同，E30 0/2，后六级blocked。不要重复旧四fixture或仅identity/time极小舍入改动的同配置原回合。原1944calls有1601正步安装，不能说全程没下降。四零plant启动失败是继承配置丢17键，已修；从完整r36配置复制后显式overlay并核必需numeric，不能再替换继承表。

并行CPU A与B，各最多10CPU，全进程树隔离、线程库1，可写build隔离，build-j6先完成。C总host2，约3核给系统/主控/发布，原A/B之外不挂第三个重型native solve。GPU关闭XLA预占、保留JAXx64，按实际空余与峰值准入，不动其他项目，不为利用率加PPO。

CPU A：原位读取W48已有全调用、held/feedback/action-execution日志和事件完整对象。固定A213@3.84、B176@3.10、A363@6.84及邻接364@6.86、B328@6.14，按plan/epoch/hash关联真正部署policy、accepted、P0、context和同epoch reference。先0积分对齐native querytime→reference/Kdelta→Clamp/H-QP→RPC float→actual held，比较原显式time与SetState后省略time。W39/W40已发现时间/动作精度，不写成新根因；量级不足就结束该支线，继续B。必要最多12条20ms、1200个.2ms子步、0solve/FD：同保存held在predictor、同policy真实接口在同predictor、同保存held在可核plant共同重建。缺外部callback/完整MjData则取消相应臂并准确标范围，不填零，不称exact历史replay，不重跑W48补数据。

CPU B：用现有accepted全.35s、J0/Jref分项、姿态/高度/前进及真实接触/GRF，判断计划本身已不利恢复还是实际偏离后反馈恶化；不能把重规划之后的实际段称同固定policy轨迹。允许按主计划最多4个新事件N solve，首有效backward方向各9正alpha，最多36额外候选，必要同路径baseline最多4条。r33 P已保留旧反馈并扰动前馈，本轮只以同问题k替代旧手工正弦方向，不重复旧P网格。

baseline须是本次r33_nominal_policy，P0不默认为它。冻结B原reference/time/K/H/scale/representation，H须metric_valid；k使用同次N当前绝对2ms网格，不能按index加到旧时标。按主计划在B原反馈之后、Clamp/R29Project之前加alpha*k(t)，扩完整policy的第二时间网格、serialize/import/CopyFrom/Reset/hash与实际部署，不用未归档runtime offset或regrid冒充冻结。alpha0直接是B，不新增零点rollout。新家族原dV不可用则expected/ratio=null并保持mu/rate，沿用原P语义、r34 strict真实风险/J/roundoff和alpha0不安装，不放宽接纳。如果A先证实际影响动作的执行/参考接入错误，可优先最小修复替代该家族；正式版本只一个控制机制，不混成本/步态/时间/反馈/PPO多项。

选一个有证据的候选，真实spec/patch/build及prepare/runtime/policy/records/合同接入，候选在原预算内替换/分配，实际工作量明示。拟部署策略最多4条20ms真实接口核对，最多400子步。至少一个代表问题形成按原规则可接纳且实际行为不同的策略，或修正已证实际动作错误，入口/原约束/短部署无未解决技术错误后，先推有限结果与真实身份，再自主A/B各一次原E30。不要求先整程成功、两流先局部恢复，不等GPU/donor，不因缺历史全状态自我阻塞。若只是已知极小舍入变化且赢家/主要事件行为不变，就不重复原任务，交有因果界限的负结果。

A/B固定各自W44 seed1/2 checkpoint-014737、resume ep00428/ep00417，保留actor/critic/optimizer/norm/RNG/fork/lineage，14737/update57、tail0/excluded145，新PPO0/0；observer/material history因果更新，不自动训练15249/59。用run_s1_go2_r12.py单次evaluation入口，明确attempt/evaluation-start-unix/checkpoint/level/seed/workers/build/contract/condition nominal，不用要求新训练的job-dir。每case自己过E30才E60→E300→E600，原30/60/300/600资格、3m容差、合法材料路线、主应变.02、非足接触/跌倒、PD/力矩/步态/horizon均不放宽。失败结束本case并blocked后级，另一case可继续；不换seed追成功、不拼接资格，技术中断独立attempt。新原回合继续预测/事件去重、补trunk_restoration，并顺带保存integration及外部callback/material全态，不额外开原回合只补记录。

Task C依照w49-task-c-design.md做B158 nominal同FP32 held双world首2ms：F1完整S0；L恢复F1第一.2ms后S1再执行旧部分load；F2完整恢复S0。核心60 GPU world-step，必要预热≤2，总≤62，逐子步读回。必要输入子步0不等则停积分，辅助KKT/residual缺失不自动阻塞；graph/dispatch三臂一致，缺完整graph恢复可在实验前统一直接step并限定结论。条件触发≤8唯一身份真实native Policy::Action对照、0physics，不再Python公式自比，不重跑24×350ms，不声称native排序等价。C不阻塞CPU主线。

仅在D57按现有manifest续传W48旧技术w48-a/w48-b两个ZIP，先查远端，不重打包/实验、不阻塞主线。R8已知单科学输入可独立恢复核schema/time/phase，本轮donor0，不批量下载或重跑R8；EOF/0bytes不是donor机制失败。仅用已有授权服务器网络，不借电脑VPN/Clash/SSH转发或未授权直连。

所有新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix；不设固定墙钟完成时间或到点kill。保留原物理时长、有限工作量、技术错误/用户停止和单次网络timeout，不无限重试；工具等待timeout不能变成任务期限。对话结束不杀后台、running不改completed，按真实clock/日志增长报状态。

紧凑Git先推，已生成必要原件/视频由D57增量Release直传，核commit/URL/size/digest；回执丢失先查已有资产，Git/index短写串行合并最新main，不强推覆盖并发。GitHub核验后不在我的电脑额外下载、打包、镜像或归档；服务器原件/checkpoint/resume/视频/上传副本不自动删除。最终给出机制结论、真实动作变化、原资格、实际rollout/FD/plant/GPU/PPO账本、失败/blocked/未运行原因、Git/Release和真实续接点。任务未过就明确未过。
```
