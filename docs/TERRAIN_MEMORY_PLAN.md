# S1：r26联合作业接续与紧凑证据交付

## 当前交接（执行后只更新这6行）

- 当前任务：W37_CONTACT_CONSISTENT_JOINT / S5-r26 的 job-20261103T075623Z 已结束；本次合并远端紧凑交付规则与服务器实际结果，普通Git交付处理中。
- 训练：2026-11-04北京时间10:17:33达到目标，新增2048条/8更新，累计6545/25，旧r25尾145 excluded、新尾0；最后更新checkpoint-006545.pt，恢复resume-state-ep00216.pt。
- 任务表现：72个训练回合，36跌倒、34非足接触、1材料域退出、1目标达到截断；50回合材料加载，最大向前0.968767m，合规去程/返程/整圈均0，侧出1。
- 评价：最终权重nominal E30于14.3096s非足接触失败，最大向前0.746252m且有材料加载；E60前级阻断，E300/600未启动。评价流程结束不等于任务通过。
- 交付：保留已经执行的73视频渲染和Release历史；本次按最新规则先完成源码/配置、全部逐回合JSON、学习与评价小记录的普通Git推送；两个旧Release失败项不阻塞紧凑交付。
- 保留：新版finish已同步D57；不自动删除研究输出。job、episode、shared training和有效checkpoint均留服务器，由用户导出备份后统一删除；不另做PC归档，不重跑物理。

## 1. 原始目标与当前阶段

长期问题仍是：**同一个世界坐标地块被前足、后足及返程反复踩踏时，如何利用可恢复材料的持久状态，让未来支撑需求反过来约束当前加载、落点和重访时间？** 原始路线见[2026-10-21冻结计划](https://github.com/kaiwen123-yang/MCP-RL/blob/4d0af25c7bcea620c7e8c9a4f5b103c4956a3fda/docs/TERRAIN_MEMORY_PLAN.md)。先做受控黏弹材料与Go2完整闭环，再证明记忆收益、泛化并进入Go2＋NX；IROS是近期投稿目标，T-RO是更高目标。工程接线不等于已经完成论文贡献。

当前链路已经真实运行：冻结因果PANN观察器→标准PPO速度cap→全身MuJoCo iLQG→时变全状态反馈→PD→Go2/真实TM02→观测。PANN权重冻结，但世界地块状态演化；PPO训练Actor/Critic；MPC尚未在每个候选中传播完整PANN/TM02材料历史，也不是端到端学习。2Hz标量cap是联合基线，尚不是原计划的终端重访价值或跨步支撑协同算法。

| 层次 | 已达到 | 未达到 |
|---|---|---|
| 刚地基础 | R8项目S3九原任务9/9、独立连续300s任务3/3 | 不能替代材料任务或当前r25能力 |
| 材料任务 | r24出现合规去程及返程暴露 | 合规完整往返0，末端固定策略E30失败 |
| 联合学习 | W36结束4497/17；W37启动快照4501/17 | 新快照不是实时进度；无收敛、可靠成功率或PANN收益证据 |
| 当前版本 | r26已记录构建、迁移并启动实际采样 | 尚无本次查验的完整新回合结果，不能从启动推断修复成功 |
| 研究目标 | R1/R2/S0提供控制、材料辨识与决策基础 | 跨腿/返程记忆的因果收益、候选材料传播、终端重访价值、NX/实机仍待做 |

**可以推进实验，应推进修正后的完整联合版本。原样续跑r25缺乏价值，也不回到每修一个模块就另开资格测试。** 本次依据截至`718a9aff1a42eb491e57a5a237d8bf70ddfe314e`的内容；[该提交中的旧计划](https://github.com/kaiwen123-yang/MCP-RL/blob/718a9aff1a42eb491e57a5a237d8bf70ddfe314e/docs/TERRAIN_MEMORY_PLAN.md)保留历史，不回改旧结果。

## 2. 这一轮推进了多少

第2、3节保留基于`718a9aff`的W36诊断，第4节记录W37已实施的设计和仍适用的计算规则。`b57136e`已保存r26实际构建/采样交接，当前接续以顶部6行、同job实际记录和第8节为准；不重复执行初始迁移或启动。

### 2.1 学习数量与任务能力分开计

| 阶段 | 控制器 | 真实回合 | 新转移/更新 | PANN累计 | 任务结果 |
|---|---|---:|---:|---:|---|
| W3 PANN首轮 | r21 | 12 | 256/1 | 256/1 | 真实学习接通，没有整程成功 |
| W34 | r24 | 30 | 1024/4 | 1280/5 | 3次合规去程，0完整往返 |
| W35 | r24 | 73 | 2048/8 | 3328/13 | 1次合规去程；3次裸往返均曾侧出，合规往返0 |
| W35末端 | r24 | 1次独立E30 | 不训练 | 3328/13 | 12.5732s材料域退出，此前已侧出；E60未运行 |
| W36 | r25 | 27 | 1169/4，尾145 | 4497/17 | 22跌倒、4非足接触、1停止截断；无材料暴露 |

PANN真实训练回合共142；ep13/14为零物理步ABI失败，不计真实回合。RAW256/1独立，不能混入PANN累计。W34中r22/r23零步失败不是完成训练的新控制器。W36原目标5376/21未达到；进程退出码0不等于实验成功。

证据：[R12索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/index.json)、[W34](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/jobs/job-20261030T161045Z.json)、[W35](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/jobs/job-20261031T101803Z.json)、[W36及回归诊断](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/jobs/job-20261101T133954Z.json)。

### 2.2 权限、时钟与真实成本

Actor每0.5s输出`cap=.215+.035*tanh(a)`，范围[.18,.25]m/s，不直接控制脚力、关节目标、步态或力矩。658维观测＝106当前/任务/查询＋20×27历史＋12个PANN恢复特征。每256条更新，4 epochs、minibatch64；原PPO参数、归一化和冻结B ideal权重沿用。

真实物理步0.2ms；目标保持/反馈2ms；逻辑重规划20ms；预测窗口.35s、176节点，每个2ms节点含10个0.2ms子步。PD、力矩界不变。逻辑频率不代表墙钟实时。

| 阶段 | 回合内仿真秒 | 回合内墙钟合计 | 转移/小时 | 单次规划均值/规划占回合墙钟 |
|---|---:|---:|---:|---:|
| W34 r24 | 504.9738 | 10.6058h | 96.55 | 约1.001s / 66.3% |
| W35 r24 | 1006.5964 | 22.4911h | 91.06 | 约1.004s / 62.5% |
| W36 r25 | 579.5092 | 22.3659h | 52.27 | 约2.585s / 93.1% |

W35含回合外开销的invocation为22.6151h/2048条；包含停机、上传的累计wall不可替代。按r25这段速度2048条约39.2h，28h不保证跑满。明确的子步导数开销可在本次修正中消除重复工作；不开GPU重写、worker网格或六流扩容项目。

## 3. 为什么失败这么多

### 3.1 r25没有进入材料学习

27回合最大前进x均约`0.0000262493m`，末端x约−4.087至−.316m，材料world cells为0；1169份新增PANN恢复观测全0。条带位于x∈[.8,2.2]m、y∈[−.6,.6]m，机器人未到那里。合规去程、远端到达、返程暴露和合规往返均0。

这不证明PANN网络坏了；0侧出也不代表路线改善。历史`effective_translation`只统计outbound/return阶段时间，并非有效向前运动。PPO更新真实发生，但数据缺少材料与返程暴露；相同控制器继续积累样本无法填补这个问题。

### 3.2 首要假设：新增支撑软目标放大接触近似

r25新增MovingSupport（Fz和只由竖直力计算的roll/pitch力矩）及SupportPlacement，依赖同一支撑分配。这是软残差目标，**不是硬等式约束**。刚地力来自经过MakeDifferentiable的规划模型，`solimp[0]`从.99变0。r21/r24已有这层平滑；新问题是r25把其瞬时接触力直接用于承重目标和残差导数。

已存同一qpos/qvel/ctrl的零积分fresh-forward比较：2.002s时native总Fz154.128N、平滑模型.0166001N，相差约9285倍；2.006s时813.499N对.076865N。这不是不同时间的日志相减，也不是PANN材料误差。

同一首个Trot计划内，最大原始Kδx从2.004s的.181rad增至2.006s的.860rad、2.008s的2.631rad，实际力矩饱和关节达3、5个；相应支撑Jacobian陡增。这是**优先修正的因果候选**，尚未做单因素闭环消融，不能宣称唯一根因已证明。

另有确定错误：`s_i=exp(-step_i/.001)`、`availability=min(sum(s_i)/2,1)`。现2Hz、duty .45、5mm摆足下sum最小2.01347589，availability全周期恒1，包括约10%的名义飞行。因此所谓飞行减载没有生效。不能改成强制每时刻固定两脚/三脚支撑。

历史限制：[R8](../results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/index.json)S2新增支撑力跟踪曾变坏，S3兼容参考才成功；[R11](../results/terrain_memory/simulation/tm-s1-go2-material-continuation-r11/index.json)r17原生接触15.3628s失败，r18原生接触＋moving velocity残差5.2224s失败，r19平滑接触＋相同残差3.878s失败。全局关闭MakeDifferentiable或重新加速度惩罚不是直接的新解法。

### 3.3 lead削弱cap，但不是最早起因

r25在路线[0,3]截断后设置`target_x=base_x+direction*.15`。2.604s后饱和；5s时base约−.517m、目标−.367m，10s时base−1.119m、目标−.969m，PPO cap仍为正。饱和使cap对位置参考的直接作用消失，moving速度残差又为0。

反馈异常约2.006s已经出现，早于lead饱和。局部参考为负x不必然是方向写反，已后退机器人向前的小范围参考也可能为负；真正要修的是cap对未来参考的权限、起步平滑和端点一致性。全局参考永不后退、始终在[0,3]、距已退到−1m的机器人≤.15m，三者无法同时成立。第4.2节明确区分真实任务进度与局部参考。

### 3.4 动作接受链没有查出错配；子步完整C/D确有重复成本

已存状态用S5-r25-inspect重算，plant积分步0，输出动作与原记录最大差0；按`nominal+K*StateDiff+原目标限幅`重构约2.2e−16。没有证据支持“选中A却执行B”；policy reference不要求是自由滚动轨迹。此诊断不是新r26实验。

r24已有十子步A/B链式复合、节点起点C/D。r25在十子步都请求完整A/B/C/D再挑峰值风险行，产生重复sensor求导。尚无证据证明矩阵乘法顺序错。`718a9aff`诊断时r26 path两个inc尚未构建应用；后来W37已记录构建/采样，但启动仍不等于已证明修复效果或提速。

### 3.5 r24的材料问题仍然存在

W35末端E30于7.516s进入材料，10.848s侧出，12.5732s材料域退出；RR压入1.032317mm、主应变.02064634，Newton残差约1e−8、无solver error。`material_domain_exit`是本构有效域，不是侧向出条带。

50mm厚、.02域限对应约1mm竖向压缩尺度。r24已有5mm摆足、慢下降、支撑高度、材料余量和7个躯干净空残差，真实落足仍出现约−1.41m/s，远高于参考−.05m/s；机身参考曾超前约1.116m。目标已写入不代表能执行，软代价不保证子步路径可行。恢复向前后，还要处理这一真实承载/冲击问题；不能调软材料、放宽域限或修改终止制造成功。

### 3.6 流程扩大了重复损失

W36当时训练入口只在技术中断/预算结束停，对自然fall/nonfoot始终开下一回合，使同一刚地退行持续27回合。W37已按第4.5节接入针对现成证据的小范围phase停止分支。

早期序列化/ABI、重复导数节点、步态覆盖、approach band整数截断、合法路线奖励等均有修复历史，不重新列为新任务。旧README/计划正文/契约里的ep117和r25待实现入口本次一起纠正，旧job和结果不回改。

## 4. W37已实施设计与既有作业规则

### 4.1 r26实现设计与已记录状态

先查D57同流tmux与最新结果，有活动作业就接续，不重复启动。复用`work/envs/d57`、MuJoCo3.2.3及实际ABI依赖、原资产、20 MPC workers/Torch2/BLAS1。只查一次实际空间。本次审查未SSH、未改服务器环境、未启动新物理。

1. **撤销新冲突。** 删除r25 MovingSupport、SupportPlacement及其support_share/requested_load分配；moving loaded-height/软落足恢复明确r24定义。保留r24已有全身动力学、moving姿态/高度/stance、输入参考、Stand平衡、PD/力矩界和平滑接触/前向差分。不混用native力常量/导数与平滑A/B，不任意清零或截断TV反馈。
2. **统一前进参考。** 按第4.2节把同一因果参考包用于runtime和候选未来节点；原路线、phase、终点、合法进度不重置，不复装r18/r19的六维速度残差。
3. **保留路径风险并去重复求导。** ImpactMargin从r12_stable_support打包条件拆出，保持其原公式、.8mm onset/.2mm scale及weight176；MaterialDomainMargin/TrunkClearance沿用。第4.3节统一数值定义、峰值选择和active-substep导数。它仍是软风险/代理模型，不包装成真实TM02硬约束。
4. **保留已实现的学习接口。** r25合法路线奖励、冻结因果PANN、658维观测和PPO数值不改；只加第4.4节明确迁移、第4.5/4.6节停止与评价的必要接线。

W37已记录相关configuration/prepare/model/runtime、C++ task/trajectory/model_derivatives、policy/train/evaluate入口的实际实现、S5-r26构建与采样启动。构建及运行身份见顶部提交和job，以下公式保留为设计依据；不再因旧planned描述重复构建/迁移。仍须以实际运行身份判断结果，不能给旧版改名冒充新实现。

当前接续已有完整联合采样，无pure MPC、单模块或E30训练准入；不重建环境、不新增测试套件/恢复演习/无关矩阵/参数网格，不自动串联多个新候选。新的交付改动不改变本次计算定义。

### 4.2 同一加速度有限、cap有效的局部参考

原任务端点仍0/3m，phase、合法路线、圈数由真实运动触发；局部参考每次重规划可以重定位，但不重置任务进度或材料状态。

每20ms正常重规划（及已有phase-change replan）建立冻结packet：active、t0、真实base x_anchor、真实endpoint、local direction、v_anchor、cap、heading/world head offset。所有候选/FD/cost共享，不随候选未来x重建，不在callback里修改共享状态。选择cap在下一正常20ms重规划进入packet；分别记录actor→参考与参考→新优化动作延迟，不宣称2ms cap→新优化动作。

取H=.35s、L=.15m、v_max=.25m/s、T_lead=max(0,L/v_max−H)=.25s；加速度a=.5m/s²复用原mission。令d=sign(endpoint−x_anchor)、ell=abs(endpoint−x_anchor)，v_anchor为原mission纵向参考速度投影到d后截在[0,.25]。对τ∈[0,H]：

```text
v(s) = v_anchor + clip(cap - v_anchor, -a*s, a*s)
t_c = abs(cap - v_anchor)/a; b = min(τ, t_c)
I(τ) = v_anchor*b + 0.5*sign(cap-v_anchor)*a*b*b + cap*(τ-b)
X_ref(τ) = x_anchor + d*min(ell, v_anchor*T_lead + I(τ))
```

稳态还原`x_anchor+d*min(ell,cap*(T_lead+τ))`。整个horizon相对**本次真实anchor**最多前视.15m，消除跨plan累计1m lead；这不是未来真实tracking error永远≤.15m的保证。加速约束/端点截断激活时，cap灵敏度可暂为0，不人为制造非零权限。

边界必须一致：Stand→Trot、turn→return从现有mission速度状态投影后平滑衔接，状态为0时从0启动，不强制重置实际已有速度，也不能node0突然前跳4.5–6.25cm。越过端点时d指回真实endpoint，不能把负remaining截0后钉住错误anchor。Stand/far_hold/home_hold保留原S3支撑；turn/turn_home保留原moving gait与endpoint/heading，不切成S3 Stand分支；上述阶段的纵向移动packet inactive。新移动phase立即重建packet，不能沿用旧方向。y仍追踪原中心线、z保留原高度含义，不把侧偏也anchor化。

**Position residual追踪head site。** 上式先生成base参考，再用同一固定heading转换`p_head_ref=p_base_ref+Rz(heading)*[.3,0,0]`；去程加.3m、朝π返程减.3m。runtime body_target/mocap与C++ residual同packet计算，Stand已有support ref优先。不允许直接把base公式写成head目标造成30cm错位。局部参考暂在[0,3]外不改变真实端点，也不保证单靠reference必然前进。

### 4.3 子步风险值、峰值与导数保持同一个对象

三类几何风险由唯一共同helper定义，供Residual和geometry Jacobian复用。沿用每2ms hold内十个**pre-integration state**的逐行最大风险，tie取最早子步；终端节点独立计算。不要把mj_step返回的推进后qpos/qvel和其pre-integration sensordata混在一起。

对状态切空间和保持输入，保留十子步A/B：

```text
P_0=I, Q_0=0
P_(j+1)=A_j*P_j
Q_(j+1)=A_j*Q_j+B_j
对于风险行k：j_star=earliest_argmax_j rho_k(x[j])
C[k,:]=Jrho_k(x[j_star]) @ P[j_star]
D[k,:]=Jrho_k(x[j_star]) @ Q[j_star]
```

当前三类风险只依赖q/v，直接control项为0；以后若换力/加速度/control-dependent风险，需加其直接D项，不能照搬。常规完整C/D只在节点起点计算一次；仅在被选中的风险子步用必要kinematics/comPos/comVel与几何/速度导数（或所需行局部差分），不重复contact solver和全部sensor Jacobian。再覆盖相应风险行。

共同helper须保持原区域判断、足序、平滑/截断、单位和尺度；不抽稀子步、不把峰值改回起点、不减少十子步A/B。现有r26 inc只是实现起点；active branch在峰值切换处仍非光滑，不称全局精确梯度或真实材料可行性保证。按实际训练timing报告成本变化，不另开benchmark。

### 4.4 ep144与145条旧尾的显式迁移

只按需下载[resume-state-ep00144.pt](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-job-20261101T133954Z-part02/job-20261101T133954Z__training__PANN_PPO_S5-seed0__resume-state-ep00144.pt)到原`outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/PANN_PPO_S5-seed0/`。必要时才取[checkpoint-004352.pt](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-job-20261101T133954Z-part01/job-20261101T133954Z__training__PANN_PPO_S5-seed0__checkpoint-004352.pt)。不回ep117，不重跑r25凑batch，不回下载全历史。

W37初始迁移已恢复Actor/Critic/optimizer/RNG/归一化/历史计数和episode编号。没有完整世界快照时另起物理会话，旧last_observation/pending不能接成新会话successor。已接入的一次版本迁移保留旧checkpoint与4353–4497这145条出处、版本、未使用状态，仅在该边界清空active buffer。默认跨版保护仍保留；此后新r26同版resume保留自己的合法尾，不再次执行ep144迁移。

```text
历史采集4497 = 已用于更新4352 + r25旧尾145
新增2048/8，第一批新rollout4498–4753
目标累计6545 = 25*256已训练6400 + excluded旧尾145 + active tail0
目标末次更新 checkpoint-006545.pt
```

沿用`len(buffer)==256`更新，不要求累计samples整除256；checkpoint/collection/job保存`excluded_old_version_transitions=145`及迁移来源。目标不能误设6400，否则仅新增1903条、7更新＋111尾。该迁移为跨控制版本warm start，不是同一MDP不间断训练或干净r24/r25因果比较。

### 4.5 无人值守只停止已知的r25起步退行

正常自然回合结束，原result、episode、resume均保存后，下一回合前检查`known_r25_startup_regression`，仅使用已有JSON。须同时满足：

1. 自然终止reason以body_fall或invalid_contact开头；已进入outbound，没有far_hold/turn/return/home_hold等后续阶段；far_arrival_s/return_start_s明确为null，表示尚未到达。
2. material.loaded_ever_cells明确为0且stage_progress.actual_material_loaded明确为false。
3. initial_x取transitions[0].position_m[0]，max_x取max_x_m，final_x取final_qpos[0]；max_x−initial_x≤原goal位置容差.15m，final_x−initial_x<−.15m。

已有r24 ep15–117共103回合（含44自然fall/nonfoot）0命中；r25 ep118–143的26次自然失败全命中，ep144预算截断不命中。首次ep118就会在13.91sim-s、约33.7min墙钟结束。这是对已有记录的分析，无新物理实验。

第一次匹配就停本phase，是有依据的计算预算策略，表示**已知表现特征复现**，不证明唯一根因相同。字段缺失/数值坐标非有限记not_evaluable，不能默认0/false误停；未到达时间的明确null是合法值，不当作非有限错误。有限到达时间表示不匹配。没有连续N次或总尝试次数上限；普通材料失败、有明显前进、远端/返程和纯预算截断均不触发。

stopped/job保存触发attempt、原物理原因、三项x、加载字段、阶段及阈值来源，原result的fall/nonfoot不改。原因从train入口传到job和evaluate_job；命中则E30/E60记skipped_due_to_startup_regression，直接保存并推送全部紧凑证据，大文件本地保留，不自动延预算/重启/换候选。不读dense arrays、不新建监控、不要求Codex等首回合。

### 4.6 训练、评价与预算

- 单流PANN seed0、256条一次更新；更新不reset，普通物理/材料失败自然续采。启动后保持源码/配置固定。
- 交互≤4h；后台训练≤28h或新增2048/8达到；末端评价另≤4h；紧凑交付沿用另≤6h上限，push完成即退出，不等待大文件或消耗完整时限。旧r25 40h扩展不继承，不自动加时；速度/成功均待测。
- W37源码/配置已提交并由现有启动器启动`job-20261103T075623Z`；接续先读同job实际状态，不再次用ep144启动、不重置28h截止。确认真实采样和普通日志增长即可交接，不等256、不持续盯守、不kill后台。
- 正常结束且本phase有新完整更新时，固定**本phase同controller/reward最后实际更新**的checkpoint/归一化；nominal、seed0、20workers、确定性均值、无学习，独立E30，原same-case前级通过才独立E60。满目标用006545，部分用实际末次，不挑最好看的。
- W37已接入限定本phase与controller/reward身份的末次更新选择，不能退回目录最大checkpoint而误用旧r25。无新更新写new_update_unavailable，不把ep144冒充新训练成果，不补物理考核；已知退行停止直接跳评价。
- 原30→60→300→600阶梯不变；本段仅E30/eligible E60。按历史速度90sim-s约2.0–3.47h，E300约6.7–11.6h、E600约13.4–23.2h，只作预算估算。长级以后独立资格/预算，不拼接片段。
- **stage_status与legal_route分别报告。** 现E30直接要求合法去程，E60/300/600更依赖完成/裸圈数；独立E60不因旧E30合法就自动合法。原状态/阈值保留，实际合规往返看合法去程、返程和整圈，不能只看passed字面。

观察顺序是正向运动、真实材料加载、合规去程、返程/整圈、末端固定策略。loss/奖励/更新数、裸圈数和训练前缀不替代任务结果。

## 5. 紧凑证据先交付，大文件由用户备份和删除

W36已1060/1060项交付（978个Release文件＋82个Git记录）、27视频、pending/render_pending为空。记录清理1084文件/4,678,796,042字节，空闲804,752,240,640→809,433,489,408字节；这是上传记录，非本次实时磁盘实测。

**2026-11-03最新用户指令覆盖此前“全量上传后自动清理”。默认顺序：计算完成 → 普通Git推送紧凑证据 → 继续研究。** 大checkpoint、全长视频、原始数组留在服务器数据盘，不自动进入后台全量上传队列；用户定期导出到自己的存储盘，完成备份后由用户统一删除。历史已上传/已删除事实不回改，也不能被当成新授权。

| 默认立即推GitHub | 默认留服务器 | 研究需要时再选取 |
|---|---|---|
| 实际源码/commit、运行配置与resolved参数、job/phase/seed/环境身份 | 大checkpoint、resume完整权重、rollout、原始轨迹/物理/预测/材料数组、长日志 | 特定权重、故障附近的原采样率片段、少数相关K/C/D节点与短视频 |
| 全部逐回合JSON和attempt清单，包括成功、失败、零步初始化、中断/截断 | 全长视频；默认不要求每个回合先渲染 | 片段附episode、时间范围、原文件位置与选取理由 |
| collection/stopped/逐update学习记录、实际计数与旧尾排除、本phase有效resume位置 | 共享training目录、各episode和job目录内的原始证据 | 沿用既有选择性Release上传入口和已发布URL |
| 末端评价/skip/blocked原因、真实任务表现、结论和下一问题 | 当前有效resume和末次完成更新继续供接续使用 | 不为补视频重跑物理，不改判定阈值 |

1. **精简文件种类，不精简失败记录。** 保留完整逐回合结果与明确的缺失原因；不把仅有总成功率、最好回合或训练前缀当成全部证据。JSON/JSONL后缀不代表文件一定小，按已知结果类别选取；密集数组、连续大日志不进普通Git。现有`training_outcomes`可在本地读取rollout/tail后只提交数字，缺失统计如实标记，不伪造。
2. **紧凑发布独立完成。** `finish_s1_go2_r12_job.py`保存本job的小记录、摘要、index和6行交接后普通commit/push；默认不调用全量renderer或Release的`upload/finish`。未选上传的大文件记为本地保留，不列入必须清空的pending。Git失败保留记录并普通续推，不强推、不重训、不删除。
3. **大数据备份和删除由用户负责。** job摘要记录job目录、episode目录、共享`training/<group>-seed<seed>/`及有效resume位置，避免只导出job目录而漏掉权重/数组。Codex不另建归档、备份确认系统或清理服务，不自动删除研究输出。用户备份时覆盖所需完整目录；仍在写入或继续训练所需的文件须保留。GitHub紧凑发布成功不等于原始数据已备份。
4. **只改交付，不改已启动计算。** 已启动job的`execution-contract.json`继续作为计算/评价证据；新版finish单独读取当前主契约的交付策略，并在摘要记录覆盖来源，旧`cleanup=true`不能恢复自动删除。现有`run-job.sh`按路径启动finish：服务器更新到新版且finish尚未启动时可直接接续，无需重启训练。若旧后处理已运行，要检查其真实状态后处理交付阶段，不能通过重启/终止训练来换策略。
5. **部署边界如实交接。** 本次仓库更新没有登录D57或改变运行进程，不能宣称现场自动上传/清理已经关闭。下次Codex先同步交付相关改动并检查该job采用的finish，保留源码、输出、环境、运行资源及既有deadline。后处理6h仅保留为兼容上限，紧凑交付一完成即可研究；没有全量上传等待门槛。

## 6. 本次采用的论文和官方依据

一手来源与本项目证据共同支持下面的工程选择，不表示外部论文已验证r26。完整阅读范围及历史保留在[文献清单](../literature/terrain_memory_sources.json)。

| 来源 | 相关机制 | 本次用途与边界 |
|---|---|---|
| [TM20 MuJoCo全身MPC v3 §III、IV-A–D](https://arxiv.org/html/2503.04613v3)及[作者固定代码](https://github.com/johnzhang3/mujoco_mpc/tree/e00c47a5adb9856af2e0f24231bb3a60d5be23c4) | PD在预测中、前向差分，软接触/TV反馈有局限 | 保留有效骨架；impratio100早已采用，不当新修复。作者硬件不是本TM02/NX验证 |
| [MuJoCo3.2.3 Modeling](https://mujoco.readthedocs.io/en/3.2.3/modeling.html#solver-parameters) | solimp阻抗改变产生力的能力，d0=0使接触起始平滑 | MakeDifferentiable改变正向动力学，不仅改变导数；不能只关开关承诺成功 |
| [STANCE §IV-B、VI-C](https://iit-dlslab.github.io/papers/fahmi19tro.pdf) | 力、压陷、接触加速度联立，考虑加载/卸载和内环带宽 | 借鉴接触相容原则，不照搬HyQ增益、删除Go2 PD或更改材料 |
| [SBIP作者方法与论文](https://ddebenedittis.github.io/soft_bilinear_inverted_pendulum_astro/) | 规划和全身跟踪同时考虑软接触 | 借鉴两层一致；其接触短历史不是跨腿/返程世界记忆，当前main不等于论文固定快照 |
| [TM27结构化参考预印本](https://arxiv.org/abs/2610.24779)、[作者页](https://tarikbanus.com/) | 已能维持nominal trot的模型骨架参与完整训练 | 先恢复可执行闭环，再判断结构化权限；ICRA2027审稿中、平地仿真，RL＋WBIC不等于本iLQG，不照搬25维/100Hz/样本量 |
| [ICRA2025：平滑动力学上的线性反馈是否足以稳定接触计划](https://arxiv.org/html/2411.06542) | 局部线性反馈可能违反单边接触的实际可执行性 | 解释大Kδx为何不能靠输出裁剪保证稳定；研究是准动力学双臂操作，不直接移植四足参数或成功率 |

共同要点是使参考、载荷/压陷、候选模型和执行相容，再讨论学习贡献。r26只处理有证据的新增冲突、参考权限与风险导数成本；完整材料候选传播、真实接触一致与反馈有效域仍有边界。没有新增论文复现/阅读门禁。

## 7. 后续方向

若再次命中已知刚地退行，结束交付该版，依据首段保存证据继续解释模型/残差/反馈；不以无条件加训练或改物理参数替代。

若恢复向前与材料暴露，仍在压缩余量、触地冲击或返程承载失败，就沿完整链路继续建立允许载荷、压陷、接触加速度/时序的相容关系。只有cap权限确实不足时再选择一个有直接物理作用的有界动作通道；若改duty/cadence，同步分段时长和导数，不同时扩维/扫权重。

形成可用合规基线后，做同执行器/奖励/硬件/资源条件的fixed/RAW/PANN、多seed与记忆消融，区分在线估计、跨足共享、返程保持和普通历史；评测时屏蔽输入不替代重新训练对照。再推进候选材料传播和终端重访价值，使未来支撑真正改变当前动作。E300/E600、实测恢复样件、NX实时、Go2硬件独立按资格/预算推进，不提前宣称完成。

## 8. Codex启动词

```text
请接续 autodl-d57:/root/autodl-tmp/MCP-RL 的 W37_CONTACT_CONSISTENT_JOINT，按我最新的交付规则执行。先读GitHub最新main中的AGENTS.md、主计划顶部6行及第5节、当前R12契约，再查看既有tmux s1-joint和job-20261103T075623Z。仓库已记录r26实际构建/采样；4501/17只是启动快照。先核实已有作业真实状态，不重复迁移、构建、启动或重置deadline。

本次先落实交付改动：保存本地未提交工作，用正常Git同步最新finish/start和交付配置；只改变后处理授权，不改正在运行的控制、PPO、物理、奖励、训练/评价预算或冻结execution-contract。新版finish默认先推紧凑证据，不自动render、全量Release upload或cleanup；确认既有run-job.sh后续会使用新版路径。若旧finish已经运行，先检查它的真实阶段，仅处理交付进程，不为换策略中断/重启训练。不能把GitHub已推送当成服务器已部署。

默认顺序：服务器完成训练与既定评价 → 普通Git推送实际代码/config身份、全部逐回合JSON、零步/失败/中断结果、collection/stopped/学习小记录、独立评价和结论 → 继续研究。大checkpoint、全长视频、rollout和原始数组默认留数据盘，不自动全量渲染、上传、打包或列为阻塞研究的pending。需要解释某次失败时，再从既有原始数据选取相关时段、矩阵或短视频，注明来源和选择理由，不补跑物理。

我会定期把大文件导出到自己的存储盘，备份完成后由我统一删除。不要自动删除服务器研究输出，也不要替我在电脑上再做归档。记录job目录、episode目录、共享training目录和有效resume/末次update位置；紧凑Git交付完成不等于大数据已备份。旧冻结契约里的cleanup=true已被这条指令覆盖。Git失败保留记录、普通续推，不强推、不重训、不删除。

原W37单流20workers、Torch2/BLAS1，28h训练和4h条件末端评价上限不变；已知r25退行停phase，普通物理失败自然续采。固定本phase最后实际新更新做nominal E30，同case合格才E60；无新更新/已知退行如实skip，不自动E300/600。后处理6h仅为兼容上限，紧凑push完成即结束，不等待大文件。保持≤4h交互，不新增测试套件、恢复演习、参数网格、监控/备份平台或全量哈希审计。

收尾更新同job/index与主计划6行，说明真实训练/评价、紧凑发布、大数据本地保留、服务器是否实际采用新交付路径及具体未完项。依据完整逐回合证据分析向前、材料加载、合法去程/返程/整圈和失败机制，再确定下一研究动作；不以更新数或裸圈数冒充稳定能力。
```
