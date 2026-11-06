# S1：W39 原始数据二次复核与更正后的 W40 三任务计划

日期：2026-11-06（北京时间）。**本次在原始数据全部发布后复核并修订 W40；没有在 D57 实现、构建或启动 r29 / GPU。** 新证据基准为 [07b0ee3](https://github.com/kaiwen123-yang/MCP-RL/commit/07b0ee3e15c9f10e9b1a6a1de3997bec36f39556)，原始审查见 [W39-raw-review](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-raw-review/README.md)。上一轮 e51/7150 紧凑结果审查保留历史范围；**其 r29 Qux 中心 QP 的等价性承诺已撤回，以本文件4.1及新数学审查为准。** W38/W39 旧执行历史不改写。

## 当前交接（执行后只更新这6行）

- 当前任务：W39 / S5-r28 两流训练和末端评价已结束；下一任务 W40_STABILITY_WITH_PARALLEL_GPU_MPC 已获用户授权、尚未实现/启动，计划为两条 r29 联合主线加一条独立 GPU MPC 适配测试任务。
- w39-a：job-20261105T032354Z-w39-a 已结束；新增1024/4至8593/33，tail0、excluded145，最后ep284；自身update33 E30在10.476s body_fall，E60阻塞，紧凑Git已发布。
- w39-b：job-20261105T032354Z-w39-b 已结束；新增1024/4至8593/33，tail0、excluded145，最后ep280；自身update33 E30在17.592s body_fall，E60阻塞，紧凑Git已发布。
- 任务能力：W39训练62回合＝59次自然失败＋3次截断；58加载、30机身入条带、26进入且内部足加载；合法去程/返程/往返均0。原件显示58个loaded回合中57个先有大反馈/执行PD饱和；下一主修订为保留旧策略的非对角H投影，尚无稳定通过证据。
- 交付：W39紧凑Git与原始诊断均已完成；北京时间2026-11-06 17:25逐项核对GitHub Release，A492/492、B484/484，合计976项/7,680,574,905字节，missing/size/state差异均0；仅补传B一项未成功文件，未重传成功资产。远端W40计划已保留合并，原截止未重置，自动删除0、电脑归档0。
- 续接位置：autodl-d57:/root/autodl-tmp/MCP-RL；先读本计划4/5/6节、W39-raw-review/README.md与r29-math-audit.md及docs/S1_W40_START_PROMPT.md。主线各续自身W39完整resume，GPU独立目录/版本/时钟；先查现有会话，已启动的实际作业优先，不重复启动。

## 1. 目标、已经完成的工作与本次阅读范围

长期问题是：**同一世界坐标地块被前足、后足及返程反复加载时，怎样利用可恢复材料的持久状态，让未来支撑需求约束当前加载、落点和重访时间？** 最终需要完整 Go2/TM02 合法往返、记忆因果收益、泛化、NX 推理和实机；近期论文方向与更高层期刊目标不变。原始科学路线见[冻结计划](https://github.com/kaiwen123-yang/MCP-RL/blob/4d0af25c7bcea620c7e8c9a4f5b103c4956a3fda/docs/TERRAIN_MEMORY_PLAN.md)。

当前链路为冻结因果 PANN 观察器 → PPO 的2Hz标量速度cap → 全身MuJoCo iLQG → 时变局部反馈 → PD → Go2/真实TM02。冻结的是PANN参数，地块历史仍在更新；PPO训练Actor/Critic。MPC候选尚未传播完整PANN/TM02历史，cap也没有独立的落脚、支撑分配或重访时序权限。

| 层次 | 已有结果 | 当前边界 |
|---|---|---|
| R1 | 固定2.4Hz与adaptive控制对照、全部正负结果 | 自适应全面优势未成立 |
| R2 | 三臂60模型、600工况/700历史评价；B ideal供观察器 | noisy恢复/重访仍有负结果；辨识精度不等于控制收益 |
| S0/早期S1 | 世界地块持久历史、相容映射与局部决策 | 尚不能代替自由机器人通行 |
| R8刚地 | S3九任务9/9、独立连续300s三任务3/3 | 不转移为当前材料/r29资格 |
| 当前完整联合工程 | 独立谱系采样、PPO更新、冻结末次策略评价、紧凑自动交付均实际运转 | 稳定合法材料通行仍未建立 |
| 论文核心 | 对象与科学假设明确 | 跨腿/返程记忆改变动作并产生可重复收益尚待验证 |

上一轮逐文件读取e51快照的5303个tracked文件，共544,772,170 bytes；4969个可UTF-8解码，334个二进制/非UTF-8。**字节扫描不是声称语义理解了每个二进制字节。** 数值审计完整解析1000份W38/W39文件、66,008,495 bytes；覆盖两轮全部训练回合、四条E30、学习小表、配置及交付清单。人工/并行审查重点是实际r26/r27/r28补丁、直接执行/学习/交付链和项目历史。该次没有把当时尚在上传队列中的原NPZ当作已读。

上一轮历史证据：

- [统计、逐回合表及数值审计](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-review/results-audit.json)，[复算脚本](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-review/analyze_results.py)。
- [控制与首发时序审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-review/code-audit.md)。
- [文献方法/实验/局限记录](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-review/literature.json)，[逐文件阅读边界](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-review/read-coverage.json)。

本次补读全部请求的诊断种类：两 E30 与全部62训练回合的原始控制/计划/参考/执行轨迹、prediction、配置/事件，全部8个rollout及所选physical/material；所选914资产约4.998GB传输、6.072GB原源文件，另62个训练MJB只列清单不重复下载。逐文件实际完成状态见[新覆盖清单](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-raw-review/read-coverage.json)。两E30模型MJB逐字节读取一致；解析与数组遍历不等于逐数字因果解释。

新证据入口：[总审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-raw-review/README.md)、[控制链](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-raw-review/control-audit.md)、[材料/学习](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-raw-review/material-learning-review.md)、[r29数学纠偏](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-raw-review/r29-math-audit.md)、[GPU重放输入](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-raw-review/gpu-replay-audit.json)。旧完整accepted/K/H从未保存，A260没有展开dense数组；补传完成不能创造这些历史记录。

## 2. W39到底推进到了哪里

### 2.1 进入材料的机会增加，失败转移，通行仍为零

| 指标 | W38 / r27 | W39 / r28 |
|---|---:|---:|
| 两流独特新增transitions | 2048 | 2048 |
| 实际训练回合 | 70 | 62（A33/B29） |
| 自然物理/材料失败 | 68 | 59（A32/B27） |
| body_fall | 27 | 41（A21/B20） |
| 躯干geom13非法接触 | 40 | 4（A0/B4） |
| material_domain_exit | 1 | 14（A11/B3） |
| 预算达到截断 | 2 | 2 |
| 60s外部截断 | 0 | 1 |
| 实际材料加载回合 | 53 | 58 |
| 机身进入条带 | 15 | 30 |
| 进入且内部足加载 | 8 | 26 |
| 自然回合存活中位数 | 14.5119s | 15.0140s |
| 自然回合平均最远x | .736514m | .808886m |
| 最大x | .975877m | 1.135616m |
| 合法去程/返程/往返、到远端/开始返回 | 全0 | 全0 |

r28没有“完全没用”：部分可行性选择生效，躯干接触终止明显减少。但它没有使机器人在原任务上稳定下来。跨轮同时存在控制版本、策略继续训练和随机轨迹变化，以上差异不是隔离的因果消融。

W39有58个自然失败已加载材料；首次load到终止中位1.3808s，9次≤0.5s，A ep260约1.2ms即材料域退出。14次材料域退出的最大主应变约.020091–.024801，越过当前2%有效域；不能将其直接称为真实材料破裂或Newton数值发散。59次自然失败全部12轴都曾达到原扭矩上限，但峰值不提供持续时间和首因。

B ep272的60s `external_session_truncation` 是重要反例：maxx=.641864m、未加载材料、未入条带、未到远端；reward约−.909438。它说明存活但停滞可以占据大量样本，不能按“60秒没倒”宣布S1通过。A ep284和B ep280因1024样本目标达到分别在7.5/6s截断，也不算自然失败。

### 2.2 学习已经实际发生，但没有连续稳定改善证据

每支从7569/29新增1024/4到8593/33＝8448已训练＋145历史excluded，tail0。两条策略独立；实验树唯一采样为6545＋4×1024＝10641，其中唯一已训练10496，**不是一条10641样本或两支更新数相加的模型**。

剔除全部截断后，A前16→后16自然回合的存活中位15.036→14.752s、平均maxx .813946→.792920m；B前13→后13（略去中间一个自然回合）为16.7052→15.014s、.807577→.824474m。没有一致单向改善。

2048条真实transition逐行复算reward，最大误差1.776e−15；物理终止bootstrap全0，外部截断保留next_value；每支四个行为版本各256条。保存的gamma=.99、lambda=.95下，8次更新的GAE均值纯算术复算与float32日志最大差1.128e−6，行为log-prob自检最大误差9.537e−7。未发现当前证据内的奖励算术、跨回合bootstrap或行为版本混用错误；本次再用8个原NPZ核对全部2048×658 obs/next_obs：全部有限，1986条同回合相邻交接逐元素一致；action/logp/value/flags/PANN与紧凑记录一致，reward仅float32舍入差≤4.73e−7。没有重新运行网络。

总reward从−701.918变为−611.547，其中W39物理失败惩罚−590。固定样本预算下少9次失败、一个长停滞回合就会显著改变总和；不能把这个总reward改善解释为学会通行。本轮先保持W38奖励不变，避免同时改控制和学习目标。

当前动作使用的PANN非零行为A30/B56，共86/2048＝4.1992%（W38为2.7832%）；返程暴露仍0。58个加载回合中27个回合的全部2Hz当前PANN特征都为0。末次E30已有同状态清零记录：A21点动作差全0；B36点仅1点非零，最大cap差1.061335e−6m/s。这只说明当前评价中贡献极弱，不是完整重训因果对照，也不证明材料记忆没有科学价值。

原件将PANN零值进一步拆清：1820/2048（88.8672%）动作的四个未来查询足都在材料外；140条查询在材料内但未命中先前真实加载格；2条命中格的真实短负载位于2ms observer端点之间（A269 1.4ms、B258 .4ms）；另86条非零。查询mask与几何相符、逻辑仿真时间的计划age通常20ms/B最大40ms；不是墙钟规划延迟。隐藏真值只用于离线解释，不喂actor。本轮加暴露/短脉冲诊断，保持PANN/658维输入不变，不把低暴露当模型接线失效。

原actor没有RunningMeanStd/LayerNorm，接收固定尺度输入；GRF历史峰A86.525/B56.356值得跟踪，但无行为权重前向证据不能断言Tanh饱和。B272在10–20s接近跌倒后落入异常局部平衡：后40s持续2000次规划与约.2m/s命令，却只前进2.69mm，右后腿长期抬起、thigh/calf目标在上界。它不是actor选择零速，也不是planner停摆；少跌倒/少饱和不足以证明稳定通行。

### 2.3 末次策略资格与吞吐

| 本分支update33固定策略 | A | B |
|---|---:|---:|
| E30物理终止 | body_fall | body_fall |
| 仿真时间 | 10.476s | 17.592s |
| 最大x | .663673m | .755814m |
| 机身进入x≥.8 | 否 | 否 |
| E60 | blocked_by_predecessor | blocked_by_predecessor |

末次评价使用自身checkpoint008593、策略均值和冻结归一化，没有PPO更新。训练最好前缀、旧刚地结果、60s停滞均不替代此资格。原30→60→300→600阶梯保留，当前仍不安排300/600。

W39共享collection墙钟14.4560h，2048独特新样本＝141.671条/h；W38为14.0583h、145.679条/h，约下降2.75%。两轮都用原2×8；不是2×12/14的新实测。

W39全部训练回合wall之和103051.831s：普通planning占87.4371%、启动planning2.0593%，合计89.4964%；50453次普通规划平均1.785931s。physics占6.1988%，feedback .5298%，状态/RPC .9219%，其余2.8532%。material计时已经包含在physics中。这里是同步离线仿真，等待规划时物理不推进；慢会损害研究吞吐，不能直接当作这些跌倒的实时延迟根因，也没有实现墙钟50Hz MPC。

## 3. 为什么仍不稳定：本次能确认到哪一层

### 3.1 躯干几何可行只是整个任务的一部分

r28在有限候选中按原trunk违反量v、原代价J排序，保留无可行候选时的restoration。已上传E30窗口A51/B70个去重计划的accepted违反量均为0；A有两次nominal不可行→accepted可行，B窗口的十候选始终都trunk可行。这证明选择机制真实生效，也证明零trunk违反量并不足以排除后来的body_fall。

一个明确的**日志语义错误**是：r28只按accepted的v是否>0设置`acceptance_mode`，把nominal不可行→accepted可行也写成`feasible_cost`，实际正则化用的却是可行性分支、`cost_ratio_valid=false`。下一版记录前后状态并区分`restored_to_feasible`；历史原值保留。这不是已经证明的跌倒根因。

### 3.2 首发异常早于真实材料受载

| E30事件（Stand→outbound之后，原仿真时钟） | A | B |
|---|---:|---:|
| 首次raw-PD饱和 | 9.224s | 14.734s |
| 首次大raw Kδx（描述阈值>.5rad） | 9.224s，1.168303rad | 15.044s，1.295255rad |
| 首次目标角裁剪 | 9.244s，.444501rad | 15.044s，.422234rad |
| 首次真实材料load | 9.850s | 15.3816s |
| 首次非足端承载 | 10.0912s | 16.1296s |
| body_fall | 10.476s | 17.592s |

A首次大反馈时nominal十二轴不饱和，反馈新引入2轴饱和；首次目标裁剪时新引入4轴。B14.734s nominal自身已饱和，所以只限制K会漏问题；15.044s nominal不饱和而反馈新增6轴饱和。两次大反馈时base约.290m、姿态仍近水平；之后数毫秒roll角速度才大幅变化。

完整原件确认两E30全部14034个执行动作、nominal、切空间误差与反馈参考/裁剪身份一致，没有找到接错reference/K、漏clip或packet错位。accepted的+20ms与actual差仍存在，但这是**同策略下的闭环轨迹分歧**，不是固定同输入的开环模型误差；实际状态不同后反馈输入也不同。A7.72→7.74s、B13.64→13.66s更早已有差且仍能继续运行，不能把旧窗口首个差值等同最初根因。

新全量发现B在15.264s、首次load之前，rawKδx峰已48.914271rad，恰处旧窗口之间；全回合峰A5.851179@9.922、B86.680943@15.744。均为裁剪前请求，不是实际执行角度。全62训练回合中60个出现大rawK/执行PD饱和；58个loaded回合中57个两事件均早于load（A256是例外），说明不是只挑两E30窗口。全训练51693次planner调用、已记录51685个新计划，其中165次恢复至trunk可行、69次accepted仍不可行，后者分布18回合；不将两E30全部accepted可行推广全训练。

工作假设收紧为：**持续的接触/跟踪偏差与过强局部反馈、执行器活跃集变化相互作用，破坏支撑。** 高增益节点常见base-z列，单位rad/m，不能和速度列混为一类；爆发实际node的完整K未存，不能伪造分项归因。仍未唯一定位某个native接触参数或导数。两E30当前状态的target∩PD区间均非空；空交集是未来必须正确处理的实现边界，非此次已观察首因。

8次无新计划均在B276：13.48与13.52–13.64s，首个缺口的mu变化符合五次backward失败早退；余7次无返回状态，不能唯一归因。旧plan695从13.50复用到13.650s，年龄150ms，13.652s跌倒；首大反馈11.524s、首load12.9166s更早，因此这是晚期退化，不是上传漏文件或全部早期失败的解释。W40为每次planner调用记录return reason、是否安装新policy、复用plan及逻辑年龄。

材料原件也要求区分时钟与候选力：A260首载material-state事件14.3636s到physical终止14.3648s为1.2ms；压入.126→1.019mm对应boundary14.3634→14.3648s的同钟间隔1.4ms，2Hz actor没有新决策机会；A254/B256的183489条真实施力记录与材料输出一致，额外越域候选行未进入物理。当前.05m厚度×.02域约1mm，在高速触地时很快耗尽；不能因此放宽域，也不能把域退出统称Newton失败。

### 3.3 已经排查、不能再当作已证bug的解释

实际r26的`R26RiskJacobian`不是旧`.inc`草稿：风险当前直接u导数为0有几何依据，后续u影响通过10子步P/Q链传播。完整状态为19维qpos＋18维qvel，共37维；状态扰动为36维切空间；足序与时钟映射已有明确记录。未发现足以归因本轮的四元数、足序、风险直接D遗漏或plan/K配对错误。

plant与预测都含kp60/kd5和原力矩限幅，不能说“预测漏了限幅”。但是现有BoxQP只约束目标角ctrlrange，固定活跃集K与末端逐轴clip并不在实际状态下重新分配受限的十二轴修正。accepted与feedback reference允许不同；将reference强改为accepted而不变动作/K，反而改变刚刚评估的策略。r28已修候选0的这类重锚风险。

预测的`MakeDifferentiable`接触设置、warmstart规则与真实plant不同；材料候选使用四个冻结因果patch，并未传播完整恢复历史。这些是明确的模型近似。因首发发生在首次材料load前，本轮先处理执行反馈及native接触邻域一致性；不能把真实TM02或PANN统一判为首因。

## 4. 下一主任务 W40 / S5-r29：先修可执行的局部反馈

### 4.1 更正后的单一机制：保留旧策略的非对角 H 度量投影

**上一版以u_ff为中心、线性项s Q_ux δx的QP方案撤回。** 现有K含原目标角BoxQP活跃集合，一般不是全维−H⁻¹Q_ux；删去Q_u也丢失原活跃乘子。无新PD约束时旧公式也会改变既有策略。此次更正的是尚未实施的计划，不是倒叙解释W39失败。详见[数学审查与代数反例](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-raw-review/r29-math-audit.md)。

已有原件复核现已完成；启动时读新审查并核对实际源码/作业即可，不再固定花30分钟重复旧窗口提取。没有已证实更早接线bug；若实际源码与审查不同，先以具体差异纠偏，不叠加多机制。新主机制为：

```text
c(x,t;s) = 原target_clip(u_ff(t) + s*K_old(t)*delta_x)
l_i(x) = max(original_target_min_i, q_i + (tau_min_i + 5*qdot_i)/60)
h_i(x) = min(original_target_max_i, q_i + (tau_max_i + 5*qdot_i)/60)
u* = argmin_{l(x)<=u<=h(x)} .5*(u-c)'*H(t)*(u-c)
```

**先按旧逻辑clip出c，再做新投影。** c已在交集内时直接返回原值，保留旧K活跃行、alpha、插值及原target clip；nominal自身PD不可实现时，即使δx=0也投影。H的非对角耦合仅在新增PD界介入时重新分配十二轴。不能用逐轴clip冒充，也不把新线性项−Hc叫原Q_ux。此机制是策略投影；本轮不半改backward/value，不声称已解完整状态约束DDP或有稳定性证明。

H必须来自成功backward实际使用的完整Quu_reg、同一regularization_used/type；backward.Quu和boxqp.R都不能冒充它。完整H正定性在原backward正则化循环内处理，与k/K一起成功后才能安装，不在部署临时加任意对角。H进入Allocate/Reset/CopyFrom、候选、winner、previous_policy与端点处理；使用原时间网格上的线性正定凸插值，c的原state/四元数/K插值照旧，不另行由H/C重算K。

三个入口共用无副作用求值：FeedbackRollouts使用其搜索s；ActionRollouts显式s=1（不可读残留的feedback_scaling成员）；部署安装s=1。alpha已在u_ff里，不重复加Q_u。日志区分nominal_feedback_scale、action_step_alpha、installed_feedback_scaling。null-state nominal读数保持原名义目标语义，不假装有状态可求PD界；只读probe不改变QP warm-start、held target或下一动作。

第一次backward前seed没有H，显式legacy_seed_no_metric，保持原seed rollout/read-only探针；不默默用单位阵。正常既有startup次数不增加，首次真正持有动作推进物理前必须已有有效metric。若全部startup backward失败，记录初始化技术失败，不能用r28 seed继续而标作r29；不新增独立准入物理轮。

空交集按原target box内最小原始PD力矩违反恢复：若l_tau>b固定u=b，若h_tau<a固定u=a，其余轴正常投影。日志保留infeasible与违反Nm，不放宽原界。相等界/固定轴必须先消元，保留H_FA*u_A交叉项；当前mju_boxQP拒绝lower>=upper，全部固定时直接构造解。非负返回nfree不代表严格收敛，记录primal/KKT残差与显式solver状态，失败退化不能冒充正常投影。

选中候选被投影改变时，旧dV ratio不再自动适用：expected/surprise置null并说明原因；前后trunk可行且实际J下降则同时保持mu与regularization_rate，不把无ratio编码成surprise=0触发反复升mu；J不降沿用原增大规则。涉及trunk不可行时保留r28 violation分支，actuator-restoration与trunk-restoration分记。全程投影未改变且满足原可比条件才使用旧ratio。修acceptance_mode日志分类，历史原值保留。

分别记录RPC前double投影解u_star与实际float32往返后held target u_held、delta_u及同一采样状态下两者的PD/约束残差。QP状态按u_star的primal/KKT判定，执行统计仍用u_held；若u_held=double(float32(u_star))且delta_tau=60*delta_u能解释边界越出，单列serialization_boundary_crossing，不当求解失败，也不从原严格饱和计数中删除。此解释只限持有开始同状态，后续子步饱和另记；不改RPC协议/物理阈值。

投影有效Jacobian仅作解释：固定H和活跃集下，令G为旧clip后c的导数，则Keff_A=B_A，Keff_F=G_F−H_FF⁻¹H_FA(B_A−G_A)；原target界B=0，PD界B=e_q+(5/60)e_v。不要把它替换原backwardK或混进开环A/B。求导点u可由投影选出，但FD内仍对独立固定target的F(x,u)求导；不能对每个状态扰动重新运行policy/QP。投影落在PD折点时记录原one-sided FD分支，本轮不同时改差分/物理平滑。

原2ms目标保持、十个.2ms PD/力矩clip、r26风险P/Q链照常；不每子步重解QP。瞬时可实现不等于全hold、材料域或稳定。保留r28(v,J)、r27 C2、因果packet、冻结PANN、658观察、2Hz cap、W38奖励及原Go2/TM02 .02域/路线/终止/阶梯。不同时改奖励、步态、PANN训练、材料或整体预测物理，不恢复r25旧载荷分配。后续依据新完整窗口判断native接触/支撑一致性是否仍为瓶颈；不能把新投影的计划发布当稳定性结果。

### 4.2 两个主分支与计数

| 新stream | 唯一父stream | 父完整resume（相对large_root） | 新目录 |
|---|---|---|---|
| w40-a / seed1 | w39-a | training/w39-a/PANN_PPO_S5-seed1/resume-state-ep00284.pt | training/w40-a/PANN_PPO_S5-seed1 |
| w40-b / seed2 | w39-b | training/w39-b/PANN_PPO_S5-seed2/resume-state-ep00280.pt | training/w40-b/PANN_PPO_S5-seed2 |

恢复各自Actor/Critic、optimizer、RNG、既有固定归一化约定与状态、8593/33、tail0与excluded145；保留祖先fork_identity并新增phase_lineage。不得回退共同ep216、重seed、合并分支、覆盖W39目录或把checkpoint权重文件当完整resume。

每流新增1024/4，目标9617/37＝9472已训练＋145excluded，tail0；首新batch8594–8849形成update34。两流都达标时再新增2048，样本树唯一计数12689，仍不是单一模型。自然编号A从285、B从281继续，保留stream/job命名。

当前R12活动契约仍记录已执行W39；[W40计划JSON](../experiments/terrain_memory/s1-go2-w40-plan.json)是待实现计划，**不能仅改名字直接启动旧r28**。Codex需完成r29正常源码/构建，以及parent_map、training directory、r29日志开关、训练/评价/finisher消费者、W40 clock、实际身份和紧凑交付接线。只在实现后把主R12激活为W40；历史job冻结契约不改。

两主线默认各12 MPC workers；各14已授权。GPU第三任务并行期间优先2×12。现有workers就是C++求解器内部工作池；不只是外层Python进程。Torch1/BLAS1、原JAX共享与精度保持。25vCPU还承载控制/RPC/材料/I/O，不能把2×14再加CPU密集GPU参考任务当作无代价并行。

### 4.3 预算、评价和实际推进标准

交互≤4h；完成必要实现、正常构建、源码配置推送后，直接启动两条完整PANN→PPO→MPC→Go2/TM02联合续训，看到真实样本/日志推进即可交接，不等待整轮结束。无主线纯MPC、E30准入、smoke/mock、恢复演习、无关测试套件或参数网格。

两主线首次实际W40启动建立新共享28h绝对训练时钟；第二流和普通resume不重置。各流训练结束后自身≤4h，只对本phase本分支实际完成的新update（>33）固定策略E30，原同case合格才E60；零新update不拿父权重替代。原known_r25停止谓词保留，普通物理失败保存后自然续采，不新增失败次数上限。紧凑交付≤6h兼容上限，完成即结束；不自动加时或串接r30。

进展同时报告：原stage_pass与合法去程/返程/往返；材料加载后存活、进入且正足载、材料域退出；nominal自身饱和与反馈新增饱和的区别；accepted→actual闭环分歧（不叫固定同输入模型误差）；投影介入量/KKT/空交集/restoration；候选trunk恢复；触地vz/压入与实际施力；PANN查询暴露及短脉冲漏采；有命令却不进展的姿态退化。仅K变小、loss下降、更新更多、站住60s或GPU更快，都不等于稳定通行。

## 5. 第三任务：本轮实际做 GPU MPC 适配与测试

### 5.1 授权、隔离和资源

用户2026-11-06明确允许在原双并行之外**增加一个专门做GPU MPC测试的并行任务**。此授权覆盖旧文档的“两计算槽/不做GPU benchmark”限制；本轮允许必要数值一致性、性能和接入测试。第三任务不是第三条合并PPO训练分支，也不占用主线正式评价资格。

任务名`w40-gpu`，独立tmux、输出目录、版本、依赖记录与GPU计划身份。默认1–2个CPU host线程，必要编译最多2个job；CPU参考计算串行/有界，不再开一组12/14线程。主线2×12与第三任务允许适度超配；用实际训练吞吐和普通资源快照观察争用。显存以正常启动时实测空闲量为依据分配，保持主线PANN/PPO余量；不把“填满96GB”设为成功标准。

优先复用可兼容依赖；必要GPU包可放独立环境并锁版本，不升级或替换两主线已经使用的MuJoCo3.2.3 ABI/环境。第三任务与主线并行做代码、适配及GPU计算，Git/index短写串行。第三任务开始时设独立≤28h绝对上限并在测试完成时提前结束；普通resume不延长，不自动启动新训练campaign。

### 5.2 先搬真正耗时的批量部分

W39规划占89.4964%回合墙钟，支持现在投入GPU工程。仍需把已有内部六项计时导出，区分nominal、model derivatives、cost derivatives、backward、rollouts、policy update；报告单位、RPC/传输、JIT及整步时间。不能在没测内部构成时认定Riccati一定是首要瓶颈。

优先原12维PD目标问题的批量动力学/有限差分与候选rollout；参考MJX-JAX/MPX的JAX批处理与时间并行。现有.35s horizon、2ms节点含176状态节点/175转移；36维切空间＋12动作的前向差分每个转移最多49组含基准状态，按节点×扰动方向批量，比只有10条候选更有并行空间。保留r26十子步前缀传播与峰值时刻定义；terminal残差单独处理。buffers留GPU，整批进出，避免每个0.2ms子步CPU/GPU往返。

当前首选是保留FP64与既有差分语义的MJX-JAX原型；若已安装版本不兼容，写清接口并在隔离环境固定可用版本。MJWarp可用于独立rollout吞吐候选，但当前官方说明其主要为FP32、无自动微分、重吞吐而非单例延迟；不能把原1e−6 FD直接搬过去就称数值等价。不要为速度缩小接触/模型或降低精度后宣称是同一个后端。只有明确瓶颈需要且原型证据支持，才在后续独立身份采用MPX式multiple-shooting SQP/parallel scan；本轮不同时展开多个GPU求解器移植。

### 5.3 用真实已存事件完成有界测试

从实际W39保存状态固定选择：刚地正常、A首大反馈/裁剪、B nominal饱和与首大反馈、首次材料加载、首材料域退出。GPU端的预测只使用当时因果可得的patch/packet；真实TM02隐藏状态仅可用于离线plant真值对照，不可进入控制器。

加入旧窗口遗漏的B15.264s大反馈点。原件已具备q/v/time/target/mocap/96D userdata与applied_settings；见[重放输入审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-raw-review/gpu-replay-audit.json)。MJB不自动安装callback或预测端MakeDifferentiable/coldstart。成本/整planner测试还须显式重建同一task phase/transition历史与初始policy/warm start；旧完整accepted/K/H及私有task状态没保存，新的重放fixture不能冒充历史整次planner复现。缺失项以双方相同且声明的初始化生成，不等待不可恢复的旧字段。

CPU/GPU严格同版本：r28对r28，或本次更正H投影的r29对r29，不移植撤回的Qux公式。A/B是固定目标的开环动力学导数，FD状态扰动内不能再次调用投影混入B*Keff；同样匹配原单边FD、PD折点与风险最早峰值分支。

GPU必须逐项匹配：Go2模型与原碰撞几何、37存储/36切空间、四元数和足序、12个目标而非torque/GRF、原kp/kd/目标/力矩范围、2ms保持内10次PD更新、冻结四patch接触代理/禁用对应native foot contact、原C2与风险峰值/导数、因果参考packet、trunk(v,J)/restoration、r29约束策略（接r29时）。Python/C++自定义callback不会因device=cuda自动执行在GPU；必须显式实现并测对应语义。

第三任务按依赖推进：同状态一步及十子步预测 → A/B及风险C/D → 固定候选整段输出/排序 → 完整一次planner调用的端到端时间。小规模固定重复用于测真实耗时，包含编译/首次调用与稳态耗时的区别、同步、H2D/D2H、内存及p50/p95；不开展无关模型、worker或超参数大网格。输出原始差异，容差按变量单位、原FD步长/模型精度预先写明，不用放宽物理通过阈值掩盖不一致。

若首选后端及同问题求解器已经在这轮实现并通过上述针对性对齐，可在第三任务用**同一冻结W39策略、原CPU真实plant**做独立标记的短闭环集成测试，最长原E30时长；训练权重与主线资格不变。其目的是检查GPU求解器输出接入，不能自动升级E60/300/600，不在正在运行的A/B中热替换。若只完成动力学或导数batch，就准确交付该层的代码和结果，不冒称完整GPU MPC或生成伪造的E30成绩。

必要的受限测试是本轮授权工作，不再被旧“一律无benchmark”条款阻断；同时也不让A/B等待全部GPU移植完成。完成后给出实际结论：兼容且有益、兼容但开销大、具体模型/精度/接口不支持，或尚完成到哪个层次。新推送的GPU计划不等于GPU已部署。

## 6. 记录与交付：小结果先走，诊断后台，用户自己备份大文件

W39原诊断现已完整发布并核对，不重启旧上传队列、不重跑旧finisher、不重置时钟或旧物理。W40仍先推代码/配置/全部逐回合JSON/更新与末次评价/结论，再继续研究；已请求诊断沿既有prepare/run/queue后台上传。上传worker是网络并发，不是“两主线＋一GPU”的计算槽计数。复用既有发布路线与短Git/index写锁，完成状态以各项manifest为准。

每个回合完整原feedback-actions、planner-records（按实际记录范围）、reference-packets、trajectory及prediction、必要模型/配置/材料事件、该phase全部2Hz记录/新rollout原NPZ仍属于用户已请求证据。E30及每流各失败族首例的physical/material数组优先；额外关键case可显式追加。不因raw/NPZ/大于2MB自动排除，不用摘要替换已请求原件。

**补齐新版本真正需要的accepted/reference/K记录，同时控制体积。** 在policy buffer复用前复制该次完整accepted状态/动作/时间/代价与实际部署reference/nominal/K/scaling；r29另含实际Quu_reg度量H、基策略原Q_u/Q_xu及活跃集身份、投影边界/KKT/状态与原PD量。一个plan只存一次冻结H、原基策略诊断矩阵、参考K、regularization_used/type与插值定义，不在500Hz动作行重复完整矩阵。在线QP的有效Jacobian随实际状态/活跃集改变，不能把参考K当成全部实际反馈；每次500Hz实际动作保存δx、三种scale/alpha身份、query_kind/applied_action_id、metric来源、plan及插值节点/权重、旧clip后center、新投影delta、即时上下界、QP解/活跃集/KKT/状态、原始与最终动作、PD请求和边界导数所需身份，以便重建有效Jacobian。采用无损二进制/压缩；关键事件前.10s/后.20s、末端.10s和restoration窗口用内存环缓存保留完整horizon，重叠窗合并；普通时段保留全回合原标量/动作记录。manifest明确“哪些plan有完整horizon”，不能写成整回合全部plan都有完整K。此选择符合只上传有分析价值部分的要求，避免在2ms节点×50Hz重规划下把完整K重复写成几十GB JSON。事件触发至少包含nominal首次原PD饱和、反馈新增饱和、max|raw Kδx|>.5rad、首次目标裁剪、首次材料load、domain exit、非足承载、QP空交集/失败/restoration、trunk restoration与终止；只是记录触发，不改变物理终止，也不再受旧x区间日志门控限制。前窗最早动作所引用plan的origin即使早于窗口下界，也保留完整快照。开始/终止附近如实记录覆盖，终止时截断后窗，不为凑数据继续物理。不得先把既有原件删掉来腾空间。

旧r27/r28完整accepted/K从未记录，无法补传；prediction是反馈参考，不是accepted。尚未上传、文件缺失、未记录、合法空文件分别标记。2ms节点完整也不等于0.2ms接触子步全覆盖；关键窗的10子步PD/接触/材料记录保留其真实覆盖。

每次planner调用（包括所有提前return）独立记录call_id、逻辑origin、wall duration、return reason、backward/rollout失败状态、policy_updated、installed_plan_id与复用policy逻辑年龄；不只在成功安装新计划时记一行，不因新增记录改变既有失败续采规则。

补充小型因果诊断：2Hz query inside/历史命中分母，2ms observer漏过的短脉冲只作离线标签；负载boundary/材料state/物理terminal时钟，越域candidate是否实际进入积分；落足vz/压入与命令存在但合法进展为0的时段。它们不进入actor、不改变奖励/任务终止、不让隐藏真值进入预测器。

大checkpoint、完整resume和全长视频默认留D57，用户定期导出存储盘备份后自行删除。**Codex不在用户电脑另做归档；不自动删除服务器研究输出或已上传副本。** 紧凑push不等于大数据备份完成；上传进程exit0不等于manifest每项完成。Git普通重试/整合远端更新，不强推，不覆盖同时上传的状态。

## 7. 新文献如何帮助这次决策

本次核心8项：5篇新加入项目的控制论文、TM43补读、MPX明确复用、MPCGPU补充2026作者实现。不是把所有项目都称作2026新论文。具体方法、实验与局限见[完整阅读记录](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W39-review/literature.json)。

| 来源 | 本次有用的机制 | 迁移边界 |
|---|---|---|
| [Adaptive Non-linear Centroidal MPC with Stability Guarantees](https://iit-dlslab.github.io/papers/elobaid25ral.pdf)，RA-L2025 | 同时约束平移稳定、角动量内部动态、总前馈＋反馈力 | 质心/GRF控制；递归可行等假设，关节执行性仍是限制，不给我们全身PD/TM02直接稳定保证 |
| [Modular Residual Learning](https://arxiv.org/html/2507.18138v1)，RA-L2025 | 动力学估计与落脚/相位修正分别学习 | Go1实机；更多动作权限，不等同2Hz cap，也没有本项目恢复记忆 |
| [Dual-MPC Footstep Planning](https://arxiv.org/html/2511.07921v1)，2025预印本，实验室列2026期刊 | 落脚与GRF共同调节角动量 | 简化模型、近似量冻结；部分草地指标快基线更好，不支持盲加第二优化器 |
| [Contact-Implicit MPC / CI-FDDP](https://arxiv.org/html/2312.08961v2)，IJRR2025 | 硬接触rollout与松弛导数分开，多重射击减少warm-start接触连锁偏差 | HOUND实机，力矩规划；错误高度仍失败、动力学defect需报告，不是Go2材料即插即用 |
| [Feedback-MPPI](https://discovery.ucl.ac.uk/id/eprint/10218402/1/RA_L_FeedbackMPPI_Final.pdf)，RA-L2026 | 明确指出局部反馈不跟踪active-set跳变、常值约束指示不进入增益 | 四足为Aliengo仿真，实机是四旋翼；借约束/反馈一致性，不建议本轮换MPPI |
| [Adaptive Legged Locomotion via Online Learning for MPC](https://arxiv.org/html/2510.15626v1)，2025预印本，TM43补读 | 在线残差适应动力学偏差；其突发接触失败案例提示我们应单独诊断接触模式错误 | 仿真证据；突发落脚/意外接触仍失败，regret不等于不跌倒 |
| [MPX](https://arxiv.org/html/2506.07823v3)，RA-L2026，复核已有来源 | multiple-shooting SQP、parallel scan与JAX批处理 | 论文WB控制24维含力/力矩，单次迭代计时；短horizon可能慢于CPU |
| [MPCGPU](https://a2r-lab.org/MPCGPU/)，ICRA2024/作者2026-10实现 | block-tridiagonal PCG与并行线性求解 | 固定基座iiwa、更新实验存在deadline overrun/未开关节限制，内核微秒不是完整Go2 MPC速度 |

另筛选了2026 ABC-DDP、Meccanica可变形地形接触感知、CUDA MPC；仅取得摘要/公告的部分明确标注，不用其headline速度或理论替代全文证据。主线动作与模型要由我们真实事件决定，不能为了追新同时堆八种方法。

长期进入下一科学阶段的顺序仍是：可稳定合法通过材料 → 因果地块状态进入候选预测 → 赋予可解释的落脚/加载/重访时序权限 → 跨前后足与返程的记忆对照及长时终端价值 → 泛化/NX/实机。GPU工程从本轮并行推进，为这些实验提供吞吐，不替代其能力验证。

## 8. Codex启动入口

使用 [S1_W40_START_PROMPT.md](S1_W40_START_PROMPT.md)；机器可读计划为 [s1-go2-w40-plan.json](../experiments/terrain_memory/s1-go2-w40-plan.json)。先核对实际作业再按本轮授权推进：W39历史已完成，W40两主线与GPU第三任务均在实现/构建/真实启动后才更新为running。本次仅完成仓库审查和计划发布。
