# 地面材料记忆主计划：W47定位数值缺陷，W48进入修复版原闭环

<!-- W48_EXECUTION_BEGIN -->
> 2026-11-13 执行更新：正式 S5-r36 已隔离构建并完成无积分 StateDiff 核验；8 次 old/corrected N solve 与 4 条实际 scale 的 350 ms 部署核对完成。四个困难 fixture 正方向仍无接纳，四条部署与评分轨迹完全一致；不外推全程不会下降。新 PPO 0/0。配置继承技术错误已修复，原 A/B E30 已正常启动并核首个 held 前记录；现有 E30 尚未通过。实际状态和物理 clock 以 [W48 索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w48/phase.json) 为准；下方 planned 描述保留为已发布计划基线。
<!-- W48_EXECUTION_END -->

更新2026-11-13（Asia/Shanghai）。结果基点`6e9075f48718509d48f51d08d0a8f62a34edfb41`。**W47有限工作与交付完成；W48 planned_not_implemented；正式S5-r36未构建、未部署。** 本次是仓库审查与计划发布，没有新服务器任务、native求解、plant、PPO或GPU物理。

[W47最终审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W47-final-review/README.md) · [W48详细计划](S1_W48_STABILITY_PLAN.md) · [启动提示词](S1_W48_START_PROMPT.md) · [执行规则](../AGENTS.md) · [机器交接](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W47-final-review/handoff.json)

## 1. 长期目标和真实成熟度

研究同一世界地块经过前足、后足、返程重复加载后的持久材料状态，使未来支撑需求影响当前加载、落点与重访。最终建立原Go2/TM02完整合法往返中的因果记忆收益，再研究条件泛化、轻量化及Go2/NX实机。

实际链为冻结因果PANN→658D PPO观察→2Hz标量cap（.18–.25m/s）→逻辑50Hz全身MPC→500Hz时变目标→PD60/5→5000Hz原物理。PPO曾真实更新，评价中权重冻结、observer/history继续因果更新；逻辑频率不是墙钟实时。当前候选尚未沿各自未来动作完整传播世界材料内态，cap不能直接控制足时/落点/卸载；稳定控制、有效历史曝光、候选内记忆、因果对照与泛化是相依但不同的任务。

| 阶段 | 保留的实际结果 | 不应外推 |
|---|---|---|
| 早期R1/R2/S0 | 材料辨识/学习正负实验均保留 | 不能把所有早期方向写成成功 |
| R8/S3刚地 | 项目9/9原任务、300s 3/3 | 不是当前r35/TM02资格 |
| R9/R10材料 | R9完整任务0/8；R10物理0/9，曾到返程 | 原材料整程未成立 |
| R11 | 23次物理启动，有效E30/E60均0 | 过期协议或无效资格不能复活 |
| W34/W35 | W34合法去程3，W35 1，有返程进展 | 稳定合法完整往返仍0 |
| W44联合学习 | A/B各14737/update57，有完整恢复链 | 不把同祖先样本重复当独立数据 |
| W45/r34 | A13.072s/B5.582s跌倒，严格接纳明确 | 搜索合法不等于任务稳定 |
| W46/r35 constant | A26.432s躯干触地、B11.638s跌倒 | 更长存活不等于更好移动 |
| W47 | alpha0四对精确复现修复、全程调用/GPU纠错 | 新优化/原plant/PPO/GPU物理0 |
| 当前研究资格 | 联合系统能运行，数值问题更清楚 | 稳定材料完整任务、记忆因果收益、NX实时尚未建立 |

A/B共有祖先6545/update25；PANN唯一转移22929，另RAW256，R12 distinct23185。A/B各自W44 checkpoint-014737.pt、resume ep00428/ep00417、tail0/excluded145；W45/W46/W47新增PPO0/0。未来15249/59不是W48自动预算。

## 2. 最近原任务仍是W46，W47没有新原失败

| W46原E30 | A | B |
|---|---:|---:|
| 结束 |26.432s floor–trunk contact|11.638s body_fall|
| 墙钟 |65.59min|40.42min|
| 最大body x |.626506m|.396364m|
| 实际材料加载 |无，足最远x .795019<.8|有；首次完整足受载9.122s|
| 最大材料主应变 |无实际受载|.00649112<.02|
| 合法去/返/往返 |0/0/0|0/0/0|
| 相邻2ms held最大跳变 |1.535358rad|1.468642rad|
| 膝±45.43Nm完整反转次数 |37|5|

E30通过0/2；后六项blocked，不是六次新增失败。A反复退步；B在8.464s已z<.2m、8.572s已|roll|>.5rad，均先于首实际材料受载。这两个数是诊断描述阈值，不新增硬失败门。90.86Nm是相邻极限反转差，不是实际越界力矩。

W46原件96资产/19,983,994,745B已交付。W47没有重启原任务，全八个原评价记录blocked。交付、数值诊断、原任务资格分别报告。

## 3. W47最重要的两个新增事实

### 3.1 找到明确的数值首扰动

四fixture×五backward的20个constant alpha0在首分叉时，actual state与reference逐元素相同、u/time一致、H-QP为内点；原四元数自差仍有约1.69e−20～2.98e−19旋转切空间舍入量，K放大后引入首个命令差。

old/corrected各四条350ms配对：old精确复现历史alpha0；corrected states/actions/times/residual/J均与nominal精确相同。B155/B158旧全窗命令差1.026335/1.108094rad，旧ΔJ +.0234713/+.0181906，修复后0。修复nominal前移仍−2.742/+1.494/+1.081/+.354mm，不能称稳定行走。

补丁只在MJPC StateDiff原差函数后，对free/ball quaternion严格self或精确antipode置零。无epsilon/K裁剪，不改真实旋转差；不改引擎mjd_transitionFD。W47 indexed诊断与实际ActionImpl插值/归一化/真实scale的边界尚需正式核对。冻结r35 binary SHA a32287b0821d9a77687575250dcb2379e370dbc84264fae80a4091027db7345e未变，两个诊断binary不是正式r36。

旧四fixture三分支540正候选仍全部预测下降/实际增加；W47没有重算正方向，不能把修复效果预先扩大或否定。

### 3.2 全程大部分调用有下降，失稳仍继续

W47新增全1944行CSV，独立复算：

| 范围 | 调用 | 正步安装 | 正步J下降 | 风险优先J上升 | 停滞 |
|---|---:|---:|---:|---:|---:|
| 全部 |1944|1601|1598|3|343|
| A moving |1222|1066|1063|3|156|
| B moving |482|317|317|0|165|
| 两流moving |1704|1383|1380|3|321|

全部含40启动调用；正步安装82.36%，moving81.16%。343内部停滞中137安装nominal、206留incumbent。B下沉前一秒50/50、侧倾前一秒50/50全部正步J下降；首膝反转前48/50。A早期首大held跳变前41/50停滞也仍成立。旧困难fixture不代表全部在线分布。

因此研究重心从“全程没有下降”修正为“正确的搜索与真实执行/支撑恢复是否一致”。跨epoch参考与问题改变，不能把J下降简单累计成闭环收益。三次晚期风险优先J大幅上升单列，不能用均值掩盖机制。

## 4. W47实际工作和推进障碍

W47状态completed_finite_work_and_delivery，后台自然结束、exit0，无墙钟deadline。8固定rollout、1400held、14000个.2ms子步，独立诊断合计2.8s；新optimizer/transitionFD/donor/plant/PPO/GPU physics均0。构建63.294s、paired35.256s；Task C在D57原位复算43文件122,624,900B约2.49s，未分配GPU。A/B/C目录与affinity不等于三项重负载持续并行。

上一计划将最小数值修复评价与donor、最新W46完整context/同origin预测及预先移动收益耦合过紧。执行后发现缺件，产生新证据的实验也无法启动。W48取消这组不必要的前置关系，保留原资格；真实失败仍是可接受的实验结果。

- R8不是全局丢失：D57本地未找到且下载0；GitHub asset681937433仍uploaded，yaw_plus5 trajectory 28,100,505B，SHA079f57e1ebad260c9cdac842dc089012e29cb46992ab64d2c31cd9d380b61625。只允许D57恢复这个科学输入并核schema/time/phase，不重跑R8、不批量下载，本轮donor solve0。
- 最新W46 A6.86s plan305/B6.15s plan267及上一预测未生成。事件环.12s/前.1后.2s缺首实际held跳变/力矩反转触发，检查全部已交付ZIP成员也没有。不是用户漏上传，不能从标量补造policy/context。
- 已有A3.846s plan178/B3.106s plan161完整policy不是完整pre-call context。新记录应随新原回合生成，不再用缺失旧数据阻塞最小修复评价。

## 5. W48冻结工作包

执行预算、路径、身份与判据以[W48详细计划](S1_W48_STABILITY_PLAN.md)为准。

1. 正式r36只加入exact-identity控制补丁及不改动作的记录，接入spec/build/源码/库、prepare/runtime/policy/records/contract及实际进程读回，保留N/r34 strict/constant和原物理/步态。一次无积分self/antipode/真实微旋转核验，不重复旧八条零点配对。
2. 四个现有完整fixture每个corrected N solve；old身份可核复用，否则每个最多补一个，≤8solve；每fixture最终完整policy真实ActionImpl一次350ms，≤4条/7000子步。用实际scale与实际评分来源，不强改indexed，不二次加alpha，不扫网格。
3. 每call在覆盖旧计划前记actual/P0/上一实际部署accepted rollout同origin预测，切空间/速度误差分单位；accepted与reference分开。完整事件包新增首held>1rad/首膝±45.43反转及阶段/terminal，前.1后.2s合并/hash去重，关键context/策略/同epoch参考齐全。缺预测标unavailable，不填零。
4. 构建/有限结果/原约束/正常入口/schema路径正确且无技术错误，先推Git，再自主A/B各一次修复版原E30。donor、缺历史context、旧fixture正步仍停滞不自动阻塞。实际记录在该attempt自身正常首次规划后、首个held物理推进前核，不要求launch前已有runtime记录、不另开plant。
5. 各14737/update57、W44 seed1/2原checkpoint/resume、tail0/excluded145，新PPO0/0；原W46作历史对照。E30各case通过才E60→E300→E600，失败后级blocked，不换seed追成功，不混成本/步态/新算法。

原E30满30s合法实际材料去程及原3m目标容差/转向；E60满60s合法往返和回家连续稳定2s；E300/600按270/540s内5/10合法往返并完整观察。短腾空合法，材料主应变.02、PD/力矩、非足接触/跌倒、合法路线和原高度项不放宽。仅Actor不能恢复中途完整plant，技术中断独立保留。

## 6. GPU第三线与资源

W47纠正raw别名、时标、扩窗解释和capture计数，来源仍W45旧策略，24名义场景18有效组合/6完整policy。初态held差约2.49e−16rad，首2ms关节差1.46e−5rad，+10ms命令差1.50136rad；350ms最大held差1.75833rad、关节差.130953rad。离线raw差784.29rad不是实际动作。

六组声明同执行identity重复场景native极值一致，GPU略异；隐藏初始化/接触workspace未全核，不能直接归FP32或不确定性。C原位分析现有逐时数组，0新physics；可选一次名义≤24同状态FP64动作批次，0积分，明确policy生成与动作计算build、原/修复映射差别。旧GPU driver硬写W46，不直接重启，所有新输出独立W48。接口不足不凑新长窗、不阻塞A/B。

默认D57两CPU流各10，第三host总2，约3给系统/主控/发布；线程库各1，全进程树隔离，可写build隔离，build-j6先完成。有限组结束后转原A/B，不额外挂重型solve。W47历史预检25CPU/120GiB RAM/97887MiB GPU，其他SuperMimic占34318MiB，不是当前余量。现场核实，不动无关进程；XLA预分配false、JAXx64保留，按实测显存峰值/余量准入，不为占卡新增PPO。

## 7. 文献与稳定之后的研究

新审查包含MuJoCo全身MPC2025、ICRA2025线性反馈与接触恢复、IJRR2026 Contact Trust Region、RA-L2024优化精度/模型误差、T-RO2024 contact-implicit MPC、IEEE Access2024 reference-free四足控制。书目信息、原文与适配限制见[一手文献](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W47-final-review/literature.md)。

核心启示是将nominal下降、真实预测—执行误差、接触恢复和路线资格分开。接触可信区域、参考自由度或多重射击只能由新机制证据决定，不同时混入identity-only W48。同步墙钟慢主要是吞吐问题，不直接归因物理实时延迟。

稳定原任务后另立学习phase：获得跨足/返程有效历史曝光；匹配控制器/信息/计算预算下RAW/PANN与冻结/打乱历史因果对照；研究候选内材料历史传播与必要动作权限；再扩条件泛化、轻量化、Go2/NX。控制修复与文件交付本身不是材料记忆创新。

## 8. 用户最新工作流与审查覆盖

**全部新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix，无固定墙钟完成时刻/到点kill。** 保留物理阶段、有限工作量、技术失败/用户停止、单次网络timeout，无无限重试；工具等待不是后台期限。运行未完推真实running/clock/PID，不因对话结束杀后台或改completed。

紧凑Git先推，已生成要求原件由D57增量Release直传核commit/URL/size/digest；回执丢失先查已有资产，Git短写串行合并最新main，不强推覆盖。**GitHub核验后不在用户电脑额外下载、打包、镜像或归档；服务器原件/checkpoint/resume/视频不自动删除。** 必要单文件服务器科学输入恢复单列；D57只用授权学术代理，不自动借电脑VPN/SSH转发/Clash/直连。

本次对结果基点8,657 tracked文件、1,025,161,586B完成字节/哈希库存，34变化文件41,749,875B解析无错误。关键源码、四fixture提交证据、全1944调用CSV与GPU纠错独立核验，Release只复核实时元数据；未下载原payload、未重放服务器实验。字节覆盖不等于声称逐byte语义理解或全部历史Release重放。

## 历史入口

[W43审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W43-final-review/README.md) · [W44审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W44-final-review/README.md) · [W45审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W45-final-review/README.md) · [W46审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W46-final-review/README.md) · [W47历史计划](S1_W47_STABILITY_PLAN.md) · [W47实际phase](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w47/phase.json)

历史“当前/待实施/继续运行”、固定截止和电脑归档只属于形成时点；当前执行统一本页、W48计划与AGENTS，不重启已结束轮次。
