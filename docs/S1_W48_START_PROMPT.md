# W48 Codex 启动提示词

整段交给具有D57和GitHub访问能力的Codex。完整边界以[W48计划](S1_W48_STABILITY_PLAN.md)为准；发布本文件不表示正式r36已构建或W48已启动。

```text
在 kaiwen123-yang/MCP-RL 执行 W48。先同步核当前main、未提交修改与有效PID/tmux/job/物理clock。阅读AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md、docs/S1_W48_STABILITY_PLAN.md，以及results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W47-final-review/下README.md、control-review.md、calls-gpu-review.md、literature.md、handoff.json。结果基点6e9075f48718509d48f51d08d0a8f62a34edfb41。按计划实施、完成有限求解并进入已声明原任务，不要只再输出计划。有效W48存在则核身份续接，不重启W45/W46/W47/G5。

W47已完成8条old/corrected固定alpha0配对、14000个.2ms子步；相同四元数自差的浮点首扰动已定位，exact self/antipode StateDiff修复后四例states/actions/times/residual/J精确再生nominal。它只修反馈自差，不修改MuJoCo mjd_transitionFD；正方向未重算、正式r36未部署、新plant/PPO/GPU physics均0。最近原任务仍是W46 A26.432s躯干触地、B11.638s跌倒，合法往返0。

采用新全程证据：1944calls有1601正步安装（1598同问题J下降、3风险优先恢复），343停滞；B明显下沉/侧倾前一秒50/50正步下降。旧四困难fixture的540正步预测下降却实际增加仍有效，但不能外推全程。研究修复对真实搜索/部署/恢复的作用，区分同问题下降与完整任务稳定。

解除旧W47耦合门槛：R8 donor缺失、最新W46历史context/预测缺失、旧fixture未有明显前移，不再阻止已证修复的一次A/B原评价。原成功、跌倒、材料、路线标准不放宽；失败据实结束，不换seed追成功。

Task A：
1. 从只读S5-r35 constant/N/r34 strict建立真实S5-r36，只加入w47-state-difference-identity.patch的控制修复和不改行为的记录。不给近邻姿态加容差，不裁K/动作，不改步态、成本、PD、材料或求解器。
2. 接入spec/patch/build/sourceSHA、prepare、runtime、policy、R29Records和evaluation contract白名单，特别核现有S5-r35单独分支，保证N/strict/constant不丢。进程exe、task参数和环境读回一致。W47诊断binary不能冒充r36；原任务不带SCORE_ONLY/INDEXED_SCORE及离线INPUT_POLICY/CONTEXT/REFERENCE。
3. 一次无积分StateDiff代数核验：self/精确反号为0、真实小旋转非零、位置/速度未改；不重复已完成八条零端点配对。
4. 用现存完整A148/A153/B155/B158，固定同P0/x0/time/private context/model/constant ubar/F/R和原预算。每fixture一个corrected N solve；old结果身份可核则复用，否则补一个old，总≤8solve。不扫none/load/R/seed。保存dV预测下降、真实ΔJ/J0/Jref/风险、完整赢家、stop/return及全部rollout/FD工作量。
5. 每fixture对corrected最终完整policy按真实ActionImpl/scale/插值/归一化/H-QP补一次350ms部署路径核对，总≤4条。对照其真实评分轨迹，不把scale≠1 nominal强套indexed scale1，不二次加alpha、不覆盖反馈reference。技术错误定位修最小链路；真实敏感性、物理失败或仍停滞据实记录。

新记录随任务接入：
每个planner_step前保存P0与actual x0/time/mission/cap/context身份，取实际部署上一策略accepted rollout在同origin的预测状态，保留plan/call/node/time/hash及位置切空间/速度差。accepted_states和反馈reference_states分开；不取本次新解，不补零，不借旧delta。
完整context/policy用事件环：原前.1/后.2s窗口，新增每回合首>1rad实际held目标跳变、首膝±45.43Nm实际反转，保留阶段与terminal。阈值只选诊断，不改失败。
事件包关联pre-call state/act/time/mocap/userdata/private gait/材料proxy、P0、赢家完整reference x/u/time/K/H/scale/metric、accepted rollout、上一预测、同epoch r35参考及观察/Actor身份。重叠合并，plan/hash去重，不逐call复制所有候选导数。schema/路径就绪即可启动原E30，在该attempt自身正常首条规划后、首个held物理推进前核字段写出和身份读回，不另开plant、不要求启动前已有runtime记录。首次无上一预测可标unavailable，不补零；关键字段缺失保留技术失败并修复。

Task B：
Task A有限结果与真实构建/入口/原约束、记录schema和输出路径接入正确，无未解决技术错误，先推Git再自主启动A/B各一次r36 evaluation-only原E30；在同一attempt首次正常规划立即核实际记录，缺关键字段按技术错误保留，不要求launch前已有新plant记录。旧困难fixture仍停滞不自动取消这项因果评价，donor不做前置。已有W46同seed/checkpoint作历史对照，不重跑r35 E30。
A固定W44-A seed1 checkpoint-014737.pt和完整resume ep00428；B固定W44-B seed2同名checkpoint和ep00417，路径按计划。各14737/update57，tail0/excluded145，新PPO0/0；权重冻结，observer/material memory因果更新。run_s1_go2_r12.py单次evaluation入口，显式attempt/evaluation-start/checkpoint/level/seed/workers/build/contract/condition nominal，不用要求新训练的job-dir。
每case自己过E30才独立E60→E300→E600；一次物理失败后级blocked，另一流可继续。原E30满30s合法材料去程与原远端容差/转向；E60满60s合法往返与回家稳定2s；E300/600按270/540s内5/10合法往返并完整观察。短腾空合法，不换seed、不拼接资格。仅Actor不能恢复完整plant，中断独立标记。

Task C独立、总host2CPU：
原位复算已有GPU paired数组的6组重复身份逐时差、首分叉与隐藏初始化可识别性，0新physics。历史源是W45旧策略，不能宣称r36等价/加速。旧s1_go2_w46_gpu.py仍写W46目录，禁止直接重启；新入口必须W48独立输出。
r36 policy产生且接口够用，可按计划最多一次同状态FP64动作批次（名义≤24，去重，缺扰动不凑），raw/clip/held独立，0动态积分；分别记录policy_generation_build/action_evaluation_build及StateDiff规则。共同corrected mapper不冒称old/corrected原生执行等价，self/antipode用native同一插值归一化后的reference，不加epsilon。不跑新20/350ms，不为利用率扩任务。
R8仍在GitHub uploaded：project-s3-formal-v1-single-yaw_plus5-trajectory.npz，28100505B，SHA079f57e1ebad260c9cdac842dc089012e29cb46992ab64d2c31cd9d380b61625。D57缺件可经既有授权服务器路径仅恢复这一科学输入，原位核schema/time/phase供后续；不下载整批、不在电脑归档、不重跑R8。本轮不加donor solve/在线多家庭，支线不可用不阻塞A/B。

默认autodl-d57，CPU树/root/autodl-tmp/MCP-RL-w42，GPU树/root/autodl-tmp/MCP-RL。现场核25CPU/96GB GPU实际余量和其他作业，不动无关SuperMimic。有限组/随后A/B各10CPU，C总host2，约3留系统/主控/发布；线程库各1，可写build隔离，build-j6先完成；XLA预分配false、JAXx64true。两原任务运行时不加重型solve；显存按实测峰值准入。

所有新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix，无固定墙钟完成时刻/到点kill。工具等待timeout不是任务期限。保留物理阶段、有限数值工作量和单次网络timeout，无无限重试。核真实clock/记录增长；对话结束不杀后台，不把running写成completed。

紧凑Git先推，要求原件由D57增量Release直传核commit/URL/size/digest；丢回执先查已有资产，不重传同字节。Git短写合并最新main，不强推/覆盖并发。GitHub核验后不在我的电脑额外下载、打包、镜像或归档；服务器checkpoint/resume/原件/视频不自动删除。服务器输入恢复只用授权学术代理，不借电脑VPN/SSH转发，凭据不进日志/Git。

最终给实际commit/build/spec、修复和记录变化、有限方向/真实部署结果、A/B各级原判定、首偏差证据、真实新增rollout/FD/physics/PPO、资源/GPU结论、Git/Release链接及实际续接点。若仍失败按新记录区分模型/执行、反馈恢复、目标或阶段问题，分别报告实现、运行、控制资格、材料记忆证据和交付。
```
