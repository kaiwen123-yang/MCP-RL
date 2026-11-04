# S1：r25停止后的诊断与完整联合推进计划

## 当前交接（执行后只更新这6行）

- 当前任务：W37_CONTACT_CONSISTENT_JOINT后台作业job-20261103T075623Z已结束；训练达标=True，评价=completed，发布=publication_pending，清理=pending_publication。
- 训练：本job新增2048条/8更新，累计6545/25，旧尾excluded=145；停止原因=target_reached；评价独立结果E30=physical_material_failure，E60=blocked_by_predecessor。
- 已完成：本job保留73个新attempt，视频可用73，资产已发布2267/2269；控制版S5-r26。
- 任务表现：最大向前位移0.9687671968487584m、材料加载50回合、合规去程0、合规返程0、合规整圈0、侧出1；末次原物理原因training_checkpoint_complete。
- 下一步：先查看本job真实路线/支撑结果；有发布待办则从现有文件续传，禁止为补视频重跑物理；再决定继续学习或正式阶梯。
- 续接位置：autodl-d57:/root/autodl-tmp/MCP-RL；摘要results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/jobs/job-20261103T075623Z.json；原Release及本job续传Release见摘要release_urls，用户电脑不另归档。

## 1. 原始目标与当前阶段

长期问题仍是：**同一个世界坐标地块被前足、后足及返程反复踩踏时，如何利用可恢复材料的持久状态，让未来支撑需求反过来约束当前加载、落点和重访时间？** 原始路线见[2026-10-21冻结计划](https://github.com/kaiwen123-yang/MCP-RL/blob/4d0af25c7bcea620c7e8c9a4f5b103c4956a3fda/docs/TERRAIN_MEMORY_PLAN.md)。先做受控黏弹材料与Go2完整闭环，再证明记忆收益、泛化并进入Go2＋NX；IROS是近期投稿目标，T-RO是更高目标。工程接线不等于已经完成论文贡献。

当前链路已经真实运行：冻结因果PANN观察器→标准PPO速度cap→全身MuJoCo iLQG→时变全状态反馈→PD→Go2/真实TM02→观测。PANN权重冻结，但世界地块状态演化；PPO训练Actor/Critic；MPC尚未在每个候选中传播完整PANN/TM02材料历史，也不是端到端学习。2Hz标量cap是联合基线，尚不是原计划的终端重访价值或跨步支撑协同算法。

| 层次 | 已达到 | 未达到 |
|---|---|---|
| 刚地基础 | R8项目S3九原任务9/9、独立连续300s任务3/3 | 不能替代材料任务或当前r25能力 |
| 材料任务 | r24出现合规去程及返程暴露 | 合规完整往返0，末端固定策略E30失败 |
| 联合学习 | PANN累计4497真实转移、17次PPO更新 | 无收敛、可靠成功率或PANN收益证据 |
| 当前版本 | r25源码、失败及同状态诊断已发布 | 刚地起步退行；r26只有未使用草稿 |
| 研究目标 | R1/R2/S0提供控制、材料辨识与决策基础 | 跨腿/返程记忆的因果收益、候选材料传播、终端重访价值、NX/实机仍待做 |

**可以推进实验，应推进修正后的完整联合版本。原样续跑r25缺乏价值，也不回到每修一个模块就另开资格测试。** 本次依据截至`718a9aff1a42eb491e57a5a237d8bf70ddfe314e`的内容；[该提交中的旧计划](https://github.com/kaiwen123-yang/MCP-RL/blob/718a9aff1a42eb491e57a5a237d8bf70ddfe314e/docs/TERRAIN_MEMORY_PLAN.md)保留历史，不回改旧结果。

## 2. 这一轮推进了多少

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

r24已有十子步A/B链式复合、节点起点C/D。r25在十子步都请求完整A/B/C/D再挑峰值风险行，产生重复sensor求导。尚无证据证明矩阵乘法顺序错。现有r26 path两个inc未构建应用，不算已经修好或提速。

### 3.5 r24的材料问题仍然存在

W35末端E30于7.516s进入材料，10.848s侧出，12.5732s材料域退出；RR压入1.032317mm、主应变.02064634，Newton残差约1e−8、无solver error。`material_domain_exit`是本构有效域，不是侧向出条带。

50mm厚、.02域限对应约1mm竖向压缩尺度。r24已有5mm摆足、慢下降、支撑高度、材料余量和7个躯干净空残差，真实落足仍出现约−1.41m/s，远高于参考−.05m/s；机身参考曾超前约1.116m。目标已写入不代表能执行，软代价不保证子步路径可行。恢复向前后，还要处理这一真实承载/冲击问题；不能调软材料、放宽域限或修改终止制造成功。

### 3.6 流程扩大了重复损失

现训练入口只在技术中断/预算结束停，对自然fall/nonfoot始终开下一回合，使同一刚地退行持续27回合。第4.5节只增加针对现成证据的小范围phase停止分支。

早期序列化/ABI、重复导数节点、步态覆盖、approach band整数截断、合法路线奖励等均有修复历史，不重新列为新任务。旧README/计划正文/契约里的ep117和r25待实现入口本次一起纠正，旧job和结果不回改。

## 4. 当前执行任务：W37_CONTACT_CONSISTENT_JOINT

### 4.1 实际实现一个协调的r26版本

先查D57同流tmux与最新结果，有活动作业就接续，不重复启动。复用`work/envs/d57`、MuJoCo3.2.3及实际ABI依赖、原资产、20 MPC workers/Torch2/BLAS1。只查一次实际空间。本次审查未SSH、未改服务器环境、未启动新物理。

1. **撤销新冲突。** 删除r25 MovingSupport、SupportPlacement及其support_share/requested_load分配；moving loaded-height/软落足恢复明确r24定义。保留r24已有全身动力学、moving姿态/高度/stance、输入参考、Stand平衡、PD/力矩界和平滑接触/前向差分。不混用native力常量/导数与平滑A/B，不任意清零或截断TV反馈。
2. **统一前进参考。** 按第4.2节把同一因果参考包用于runtime和候选未来节点；原路线、phase、终点、合法进度不重置，不复装r18/r19的六维速度残差。
3. **保留路径风险并去重复求导。** ImpactMargin从r12_stable_support打包条件拆出，保持其原公式、.8mm onset/.2mm scale及weight176；MaterialDomainMargin/TrunkClearance沿用。第4.3节统一数值定义、峰值选择和active-substep导数。它仍是软风险/代理模型，不包装成真实TM02硬约束。
4. **保留已实现的学习接口。** r25合法路线奖励、冻结因果PANN、658维观测和PPO数值不改；只加第4.4节明确迁移、第4.5/4.6节停止与评价的必要接线。

实际修改现有configuration/prepare/model/runtime、C++ task/trajectory/model_derivatives、policy/train/evaluate相关入口，按原方式构建S5-r26并记录配置、patch/source与运行资产。当前契约active r26明确是planned，不代表二进制或spec已经存在；实现后使实际build/spec/入口一致。不能只写一个未被读取的JSON，或给旧版改名。

必要构建和实际启动检查后直接完整联合采样，无pure MPC、单模块或E30训练准入；不重建环境、不新增测试套件/恢复演习/无关矩阵/参数网格。4h内实现不完就交具体未完代码，不启动旧版冒充r26，不自动串联多个新候选。

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

恢复Actor/Critic/optimizer/RNG/归一化/历史计数和episode编号。没有完整世界快照，另起物理会话；旧last_observation/pending不能接成新会话successor。PolicyStream当前遇不同controller非空tail会ValueError，需增加**一次明确版本迁移分支**：旧checkpoint与4353–4497这145条出处、版本、未使用状态保留，active buffer清空。不是删除默认保护，也不是每次resume静默清空；新r26同版resume保留自己的合法尾。

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

stopped/job保存触发attempt、原物理原因、三项x、加载字段、阶段及阈值来源，原result的fall/nonfoot不改。原因从train入口传到job和evaluate_job；命中则E30/E60记skipped_due_to_startup_regression，直接完整render/publish/cleanup，不自动延预算/重启/换候选。不读dense arrays、不新建监控、不要求Codex等首回合。

### 4.6 训练、评价与预算

- 单流PANN seed0、256条一次更新；更新不reset，普通物理/材料失败自然续采。启动后保持源码/配置固定。
- 交互≤4h；后台训练≤28h或新增2048/8达到；末端评价另≤4h；渲染/发布/清理另≤6h。旧r25 40h扩展不继承，不自动加时；速度/成功均待测。
- 源码/配置普通提交推送后，复用`start_s1_go2_r12_tmux.py --resume <实际ep144路径> --hours 28 --postprocess-hours 6 --network-mode <服务器独立可用模式>`。确认真实采样和普通日志增长即可交接，不等256、不持续盯守、不kill后台。
- 正常结束且本phase有新完整更新时，固定**本phase同controller/reward最后实际更新**的checkpoint/归一化；nominal、seed0、20workers、确定性均值、无学习，独立E30，原same-case前级通过才独立E60。满目标用006545，部分用实际末次，不挑最好看的。
- 当前评价用目录最大checkpoint，需改成限定本phase与controller/reward身份，避免残留旧r25。无新更新写new_update_unavailable，不把ep144冒充新训练成果，不补物理考核；已知退行停止直接跳评价。
- 原30→60→300→600阶梯不变；本段仅E30/eligible E60。按历史速度90sim-s约2.0–3.47h，E300约6.7–11.6h、E600约13.4–23.2h，只作预算估算。长级以后独立资格/预算，不拼接片段。
- **stage_status与legal_route分别报告。** 现E30直接要求合法去程，E60/300/600更依赖完成/裸圈数；独立E60不因旧E30合法就自动合法。原状态/阈值保留，实际合规往返看合法去程、返程和整圈，不能只看passed字面。

观察顺序是正向运动、真实材料加载、合规去程、返程/整圈、末端固定策略。loss/奖励/更新数、裸圈数和训练前缀不替代任务结果。

## 5. GitHub交付和服务器副本清理

W36已1060/1060项交付（978个Release文件＋82个Git记录）、27视频、pending/render_pending为空。记录清理1084文件/4,678,796,042字节，空闲804,752,240,640→809,433,489,408字节；这是上传记录，非本次实时磁盘实测。

**复用已经执行过的finish/publish/cleanup，不在用户电脑归档。** 清理不是新的待开发平台。

1. 所有实际源码、环境记录、正常/失败/截断结果、全长视频、原始轨迹、物理/预测/材料数组、transitions/rollout/checkpoint、隔离旧尾出处有Git/Release去向。复用partNN最多900资产、原URL和manifest；缺原轨迹不能渲染时记录真实原因，不补跑物理。
2. 全部应交付完成、pending/render_pending为空，现有一次名称/大小/上传状态核对，必要源码及紧凑结果普通push成功后，才清理。不增全量hash、回下载认证或PC复制。
3. 只删manifest覆盖、远端已确认且inactive的服务器研究副本，可含已结束旧job输出/checkpoint。源码/.git、紧凑记录、work/envs/d57、work/r11-frozen、work/assets、work/upstreams、work/terrain-memory、work/tools和未传/活跃文件保留，不递归清空项目/数据盘。
4. 文件数/字节、实际空间变化和保留原因写同job并普通push；小记录push失败保留续推。上传失败只续后处理，不重训，不删未交付文件。
5. 续训按需恢复最新checkpoint；不在用户电脑下载、复制、镜像、压缩或归档，不回下载全部历史。

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
请在 autodl-d57:/root/autodl-tmp/MCP-RL 执行 W37_CONTACT_CONSISTENT_JOINT。先同步GitHub最新main，读取AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md顶部6行与第4节、当前R12契约及直接相关源码；最多5行开场，随后实际实现，不重复全仓审查、新计划或论文复现。

先查同流tmux和最新实际结果，避免重复启动。r25已停止：累计4497条/17更新，最新resume-state-ep00144.pt含145旧尾，27回合全部未到材料，40h扩展取消。按计划Release链接只恢复需要的最新状态，不退ep117，不原样续跑r25；复用D57现有环境、ABI、资产和服务器独立网络。

实际实现并构建S5-r26：保留r24全身控制、PD/兼容输入参考和r25合法奖励；删除r25 MovingSupport/SupportPlacement及错误分配，不全局关闭MakeDifferentiable、不复装r18/r19速度残差、不任意截断反馈。按第4.2节将加速度有限、cap有效的同一局部前视packet贯穿runtime和MPC未来节点，处理启动/转向/越过端点、base-head .3m偏置和时钟；真实路线/材料状态不重置。按第4.3节统一子步风险值、峰值和active-substep导数，保留十子步A/B，避免每子步完整C/D；保留ImpactMargin原语义，已有inc仅草稿不能改名冒充实现。

必要构建后直接完整PANN+PPO+MPC联合训练，无单模块/E30训练准入、测试套件、恢复演习、参数网格或通用监控。原Go2、材料.02域、力矩/碰撞/跌倒/路线/评价定义不变，冻结因果PANN、原PPO/观测/2Hz单cap。显式迁移ep144：恢复Actor/Critic/optimizer/RNG/归一化/历史4497/17，145旧尾隔离留证，空新buffer与新物理会话开始；同版以后resume保留合法tail。新增2048/8观察点，累计6545/25，记录145 excluded，第一批4498–4753。

实现第4.5节已知r25起步退行的小停止分支：自然回合保存后用已有result判定，命中则保存stopped/job、跳E30/E60直接发布；原物理失败不改。普通材料失败自然续采，更新不reset；不设尝试次数上限、不持续盯守、不自动加时或串联新版本。

单流20 MPC workers、Torch2、BLAS1，只查一次实际空间。源码/配置先提交推送，再用现有tmux启动器；训练28h、末端评价4h、发布清理6h各自计时，4h只限交互。确认真实采样和日志正常即可交接。正常结束且本phase有新完整更新时，固定本phase同版本末次checkpoint/归一化做nominal E30，通过才E60；无新更新或已知退行停止则记原因，不误取旧权重。stage_status与legal_route分别报告，保留30/60/300/600阶梯，本段不自动跑长级/全矩阵。

复用已实现的finish/publish/cleanup：所有源码、环境、视频、原始轨迹、大数组、rollout/checkpoint及失败/部分结果从D57直传Git/Release。全部交付、普通push成功、待传/待渲染为空且文件inactive后，删除manifest覆盖的服务器研究副本；保留源码/.git、紧凑记录、环境、运行资产、未传/活跃文件。复用一次名称/大小/上传状态核对，不新增hash/回下载认证，不在我的电脑下载、复制、镜像、压缩或归档。清理数量/字节/空间变化写同job并推送；发布失败只续后处理。

收尾只更新正常结果及主计划6行，分开说明实现、训练、评价、发布、清理及未完成项。重点报告实际向前、材料加载、合规去程/返程/整圈和失败机制，不以样本数/loss/裸往返/训练前缀冒充任务能力。若4h内实现未完，留下具体代码差距，不启动旧版冒充r26。
```
