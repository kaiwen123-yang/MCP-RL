# W39 / S5-r28 控制链、首发事件与下一轮稳定性修订审查

审查基准：仓库 `e51acd535445f617bd0486650e67df4bcc116481`。本报告只读取仓库代码、补丁、W39 紧凑结果及已经提交的 E30 数值窗口；没有启动物理实验，没有热改 r28，没有读取尚在后台传输的完整原始数组。两份 diagnostics `evidence.json` 在该提交均为 `publication_pending`。`missing=[]` 表示服务器侧选取清单未发现缺文件，不表示全部文件已上传，也不表示本次已阅读全部文件。

## 1. 结论

**r28 修复了候选选择与可行性记录的一部分，但还没有解决真实闭环的稳定性。** 它确保在已有候选中优先选择原 `TrunkClearance` 几何约束可行的轨迹；这不是足端接触、可执行扭矩、姿态稳定、材料有效域以及真实执行误差的联合可行性保证。

两个最终 E30 最有区分力的共同现象是：**机身尚接近水平、尚未第一次实际加载 TM02 时，关节速度的 accepted→actual 误差已经出现，随后局部状态反馈产生过大的目标修正与扭矩饱和。** 因此目前不能把首发异常全部归因于首次材料加载，也不能指望 2 Hz 的单一速度上限直接修复 500 Hz 的十二关节反馈问题。

下一版的主要修改宜集中为一个机制：**将原执行器目标范围及状态相关 PD 扭矩实现域纳入局部受约束反馈，并让候选 rollout 与部署执行使用同一条路径。** 保留 r28 的 trunk `(v,J)` 排序；不再原样延长 r28 训练，也不同时叠加新奖励、接触模型、步态、PANN 训练和 GPU 控制器替换。该建议是有事件证据支持的优先干预方向，并不是“已经证明能稳定”的结论。

## 2. 可复核证据与覆盖边界

主要源文件：

- `experiments/terrain_memory/s1-go2-r11-S5-r26.patch`：实际使用的 `R26RiskJacobian`、10 子步路径风险和导数传播；不是早期 `s1-go2-r26-path-derivatives.inc` 草稿。
- `experiments/terrain_memory/s1-go2-r11-S5-r27.patch`：动作/计划记录、accepted 与 feedback reference 的分离。
- `experiments/terrain_memory/s1-go2-r11-S5-r28.patch`：trunk 违反量、字典序选择、正则化分支与候选 0 重锚修复。
- `experiments/terrain_memory/s1-go2-r25-source/mjpc/planners/ilqg/{planner.cc,policy.cc,backward_pass.cc}`：仍沿用的候选 rollout、反馈、BoxQP 主体。
- `scripts/s1_go2_r11_model.py`、`scripts/s1_go2_r11_runtime.py`、`scripts/s1_go2_joint_material.py`：实际 PD、扭矩限幅、控制/物理时钟与 TM02 耦合。

E30 已提交的窗口根目录为：

`results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/jobs/job-20261105T032354Z-w39-a/numeric-evidence/event-windows/job-20261105T032354Z-w39-a-eval-PANN_PPO_S5-s1-E30/`

`results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/jobs/job-20261105T032354Z-w39-b/numeric-evidence/event-windows/job-20261105T032354Z-w39-b-eval-PANN_PPO_S5-s2-E30/`

以下事件时间取自两目录的 `event-summary.json`，逐步状态和动作来自 `window-*-control.json`，力来自 `window-*-physical.json`，计划来自 `window-*-plans.json`。复核把重复计划按 `plan_id` 去重，把同一原点的 node 10 与 20 ms 后实际 `qvel` 对齐；没有重新积分，也没有从状态合成接触力。RPC 动作时钟为 float32，真实 control 时钟为 float64，采用导出文件已完成的 executed-control 对齐。实际 `qpos/qvel` 为 500 Hz，力为 0.2 ms 采样。

计划的完整 horizon 标量 `accepted_trunk_violation_m` 已记录；完整 accepted 状态/动作/K horizon **没有记录**。可读的稠密节点只含 node 0 与 node 10（+20 ms）。`prediction-*.npz` 的角色是反馈参考，不能改名当作 accepted。故可以判断某计划记录的整段 trunk 违反量，却不能凭两个节点还原首个 +4 ms 的 accepted 接触模式或完整 K。

## 3. r28 做对了什么，又没有解决什么

`r28.patch:65–82` 从原平滑正部残差反解真实 `margin-gap`，在全部保存的路径风险节点与终端中取最大值；没有另加地面几何、margin 或物理容差。`r28.patch:210–230` 先比较违反量 `v`、再比较原代价 `J`，因此可行候选优先于不可行候选。

已上传 E30 窗口的去重统计：

| 项目 | A | B |
|---|---:|---:|
| 窗口内计划数 | 51 | 70 |
| accepted trunk 违反量 > 0 | 0 | 0 |
| 十个候选全 trunk 可行 | 49 | 70 |
| 仅四个候选 trunk 可行 | 2 | 0 |
| nominal 不可行、accepted 可行 | 2 | 0 |
| 使用正则化 1e-6 的计划数 | 43 | 69 |

这不是全训练统计，也不是完整 E30 每一计划的统计。A 的两个恢复事件分别发生在：

- 10.420 s，plan 542：nominal `v=0.002674630565 m`，nominal 可行候选 0，最终四个可行候选中选 winner 5，accepted `v=0`。原代价反而增加约 971.298，仍因几何可行性改善被接受。
- 10.460 s，plan 544：nominal `v=0.001889776410 m`，同样恢复至 `v=0`。

这证明排序实际起作用。它也直接说明“accepted trunk 可行”与“真实任务稳定”是两回事：A 仍于 10.476 s 触发原 body-fall 规则，B 于 17.592 s 触发。两个 E30 都没有完成合法 outbound，更没有 return。

### 一处确定的记录语义错误

`r28.patch:129,188–194` 用 `accepted_violation > 0` 定义 `restoration`，因此上面两次 nominal 不可行→accepted 可行也被命名为 `acceptance_mode="feasible_cost"`。但控制分支 `feasible_comparison` 要求前后均可行，此时实际使用的是可行性进展分支，且 `cost_ratio_valid=false`。所以不能按 `acceptance_mode` 字符串直接统计“是否发生过恢复”。

下一版可将日志明确分为 `feasible_to_feasible_cost`、`restored_to_feasible`、`infeasible_restoration`，同时保留前后违反量。**这是一处诊断语义错误，不是已证明的失败根因。** 历史数据应保留原样，分析时使用前后 `v` 分类。

## 4. 两个 E30 的首发次序

| 事件 | A E30 | B E30 |
|---|---:|---:|
| 首次原始 PD 请求达到饱和 | 9.224 s | 14.734 s |
| 首次大 raw Kδx | 9.224 s，1.168303 rad | 15.044 s，1.295255 rad |
| 首次目标角度裁剪 | 9.244 s，最大裁剪 0.444501 rad | 15.044 s，最大裁剪 0.422234 rad |
| 首次真实材料 load | 9.850 s | 15.3816 s |
| 首次已记录非足端承载 | 10.0912 s | 16.1296 s |
| 原 body-fall 终止 | 10.476 s | 17.592 s |

“大 raw Kδx”沿用 `scripts/export_s1_go2_event_windows.py:76–77` 的筛选阈值 `max(abs(raw_K_delta_x)) > 0.5 rad`；它是诊断事件阈值，不是新增的物理终止规则。事件扫描范围为 Stand→outbound 之后。

### A：反馈把原本未饱和的 nominal 变为饱和

9.224 s 的 nominal PD 请求十二轴均未饱和，反馈后 actuator 0/4 达到原扭矩限制。9.244 s 的 nominal 仍无饱和，反馈后 actuator 0/3/4/5 饱和；目标修正原始峰值为 1.570003 rad，其中 0.444501 rad 被目标范围裁掉。

9.224 s 时实际 base 为 `(0.547612,-0.000840,0.289587) m`，roll/pitch/yaw 约 `(0.000405,-0.001135,0.003866) rad`。这不是先整体倾倒后才产生大修正。9.222 s→9.224 s，参考误差最大项转为 FR thigh 速度约 +0.8924 rad/s，同时 RL thigh、FR calf、FL calf 的速度误差也出现。此时姿态角仍小；9.226 s 的实际 roll 角速度才达到约 −1.2287 rad/s。

首次材料加载比首次大反馈/扭矩饱和晚约 0.626 s。这里能排除“首次真实 TM02 加载导致最初反馈爆发”的解释；尚不能单凭事件先后把全部原因归结为某一个 native contact 参数。

### B：nominal 自身饱和与反馈放大并存

14.734 s 的 nominal 原始 FL hip PD 请求约 −63.8202 Nm，已经超过原 23.7 Nm 限制；当时 raw Kδx 最大仅 0.013055 rad，反馈没有新增饱和轴。因此**只给 Kδx 加上限会漏掉一类问题**。

15.044 s 的 nominal 十二轴均未饱和，反馈却新增 actuator 0/1/2/3/6/7 六轴饱和。实际 base 高度 0.289707 m，roll/pitch 约 −0.000908/−0.002011 rad。参考误差最大项包括 FL thigh +0.9054、FL calf −0.7386、RL calf −0.5366 rad/s。到 15.046 s，roll 角速度约 +0.9271 rad/s。

首次大反馈/目标裁剪比真实材料 load 早约 0.3376 s；首次 nominal 扭矩饱和更早约 0.6476 s。B 的 15.282–15.580 s 已存窗口中 raw Kδx 最大达 31.3037 rad，但这只是该窗口内最大原始修正，不能解释为最终实际关节角度，也不能冒充全回合最大值。

## 5. 不是仅仅把 feedback reference 与 accepted 混淆

下面直接比较所选 accepted 节点与 20 ms 后的真实速度；速度范数混合线速度和角速度单位，仅作同构向量误差诊断，不是物理能量度量。

| 原点→+20 ms | actual−accepted 速度向量范数 | 主要关节速度差 | base 高度误差 |
|---|---:|---|---:|
| A 9.16→9.18 s | 0.67475 | FR thigh −0.63363 rad/s | +0.213 mm |
| A 9.20→9.22 s | 0.39182 | FR thigh −0.27450、FR hip +0.19612 rad/s | +0.191 mm |
| A 9.22→9.24 s | 1.68542 | FR hip −1.05895、FR thigh −0.93274 rad/s | +0.367 mm |
| B 15.00→15.02 s | 0.60262 | FL thigh +0.49873、FL calf −0.26128 rad/s | +0.183 mm |
| B 15.02→15.04 s | 1.12297 | FL thigh −0.87977、RR thigh −0.40091 rad/s | +0.232 mm |
| B 15.04→15.06 s | 2.89875 | FL hip +2.32227、FL thigh −1.32775 rad/s | +0.112 mm |

A 在 FR 接触从有到无的邻域，B 在 FL/RR 离地、FR/RL 承载的邻域出现上述差异。该比较支持“接触邻域局部模型与真实关节速度演化不一致，之后反馈放大”的诊断方向。由于没有首 +2/+4 ms 的 accepted 接触力和完整 K，不能进一步断言某条腿在那一刻的预测接触状态一定与真实状态相反，更不能把全部速度误差解释为某一接触参数的单独因果效应。

## 6. 源码核对：确认的问题、没有证据的问题

### 6.1 风险导数没有已证实的直接控制遗漏

实际 r26 `R26RiskJacobian` 对几何风险的当前 q/v 求导。这些风险值在固定 pre-integration 状态下没有直接 u 项；控制通过子步状态转移影响后续风险。代码保存 `P_0=I,Q_0=0`，逐步传播 `P_{j+1}=A_jP_j`、`Q_{j+1}=A_jQ_j+B_j`，再对最大风险所选子步写入 `C_r=r_xP_j`、`D_r=r_xQ_j`。因此“风险函数当前直接 D=0”不等于“所有路径风险对 u 的导数被漏掉”。见 `r26.patch:77–102,263–338`。

姿态差使用 36 维切空间和 `mj_differentiatePos`，并非直接相减 37 维状态；风险足序 FL/RL/FR/RR 与 runtime FL/FR/RL/RR 已在窗口中显式映射。未发现可据此归因 W39 的顺序或四元数错误。

### 6.2 accepted 与 reference 不同本来可以合法

`ActionRollouts` 先给 nominal 动作加 line-search feedforward，再用原线性化参考计算状态反馈，滚出 accepted。部署复制生成该 accepted 的候选策略。因此 reference 不等于 accepted 本身不是 bug；若直接把参考换成 accepted 而保持原 feedforward 和 K，反而改变了刚刚评估过的策略。

r28 删除了候选 0 在复制前被 accepted 覆盖的语句，修复的正是这一风险。已记录动作满足 `action_before_clip=nominal_unclipped+feedback_scaling*raw_K_delta_x`，示例事件的等式最大误差为 0。没有找到足以宣称当前 r28 执行使用了错误 plan/reference/K 配对的证据。

### 6.3 确定的结构缺口：目标范围约束不等于可执行反馈

现有 `backward_pass.cc:158–196` 的 BoxQP 边界来自 `model->actuator_ctrlrange−nominal_action`，即十二个目标角度范围。`planner.cc:471–484` 传入的是这个固定目标范围。其固定活跃集反馈把活跃目标轴的反馈行置零；执行 `policy.cc:151–160` 再对 `u_nom+Kδx` 逐轴裁剪。

实际执行映射是 `tau=clip(60*(u−q)−5*qdot, 原 torque limits)`，见 `s1_go2_r11_model.py:91–105` 和 `s1_go2_r11_runtime.py:539,570–578`。预测 rollout 的真实执行映射中也有这个饱和，不能说预测完全没算扭矩限制；但“rollout 包含饱和”不等于“局部 QP/反馈对实际状态下的饱和活跃集重新优化”。B nominal 自身饱和和 A/B 的反馈新增饱和，恰好跨过这两个层次。

### 6.4 模型失配仍是真实候选原因，但未定位到单项

Agent 的梯度规划对预测模型调用 `MakeDifferentiable`，改变 native contact 的 `solimp[0]`，并按既有配置禁用预测 warmstart；真实 plant 保留原接触设置。材料预测使用四个冻结因果 patch，不在候选中推进真实 TM02/PANN 历史。参考：`agent.cc:299–328`、`utilities.cc` 的 `MakeDifferentiable`、`r10_contact.h`、`s1_go2_joint_material.py`。

这些是明确的模型差异，而不是新发现的运行代码错字。现有首发事件发生在首次实际材料 load 前，故 native 接触/局部导数/反馈有效域值得优先核查；但不能未经已存 raw 对齐就把 `MakeDifferentiable` 简单关闭并同时更换反馈，再把结果归因于任一个因素。

r28 的 cost surprise 仍是预测器内部的实际 rollout 代价改善与局部二次模型改善之比，不是“真实 plant 对预测状态的符合度”。首发附近 A plan 482/483 的 surprise 分别约 0.876938/0.971428，B plan 773 约 0.624434，三者使用的正则化均为 1e-6；其后真实闭环仍发生上述大修正与误差。因而“模型内 ratio 正常、正则化到下界”不能被当作反馈已处于真实系统有效域的证明。新受约束反馈也必须记录真实 accepted→actual 误差，不应只追求把同一 ratio 做得好看。

## 7. 下一版单一主要机制：状态相关的执行器约束反馈

目标是替换“固定活跃集 K 输出后再逐轴裁剪”的局部策略实现；原 plant、PD、扭矩上限、任务和终止均不变。

在实际状态 `(q,qdot)` 下，每个关节的瞬时可实现目标区间为：

```text
lower_i(x) = max(original_target_min_i,
                 q_i + (original_tau_min_i + 5*qdot_i)/60)
upper_i(x) = min(original_target_max_i,
                 q_i + (original_tau_max_i + 5*qdot_i)/60)
```

**不能仅把 u 逐轴投影到这个区间就宣称修复。** 在当前瞬间，这种投影与原扭矩硬裁剪可能给出完全相同的扭矩。真正需要保留局部二次模型的非对角耦合，并在活跃集变化时重分配十二轴修正。

一种边界清楚的实现是：以已经选中的 line-search feedforward `u_ff` 为中心，使用对应的正则化 `H=Q_uu`、交叉项 `C=Q_ux` 和该候选实际 `feedback_scaling=s` 求十二维 box-QP：

```text
eta* = argmin_eta  0.5 eta' H eta + (s C delta_x)' eta
subject to lower(x)-u_ff <= eta <= upper(x)-u_ff
u = u_ff + eta*
```

这样在 `delta_x=0` 且 feedforward 可实现时保持原已选动作；同一局部模型无活跃约束时恢复 `u_ff+sK delta_x`，其中 `K=−H^-1 C`。目前最终部署将 s 设为 1，但 nominal feedback rollouts 会搜索其他 s，因此新实现不能在所有调用点一律假设 s=1。若选用“以已有 `u_ff+sK delta_x` 为无约束目标、按完整 H 度量投影”的等价设计，应明确其线性项和原活跃集含义，不能混用一个 H/C 与不相容的 K。

若 nominal 本身违反原 PD 实现域，QP 也必须处理它，而非只截短反馈。H/C、参考状态、feedforward、s、regularization、plan_id、插值规则必须来自同一计划，不能用别的候选或上一计划矩阵拼接。

如果实现改用完整 `Q_u`，必须同时统一候选生成、line-search α、前馈定义与 predicted decrease；不能在执行时再次加一次完整 Newton 步，悄悄取消刚刚接受的 α。若投影/活跃集改变使旧 `−α(dV0+αdV1)` 不再描述实际动作变化，应重算局部模型预测变化或将 cost ratio 标为不可用，不能继续报可信 surprise。

### 活跃边界的导数不能仍全部置零

原固定目标界激活时，边界状态导数为 0；PD 扭矩边界激活时，`du_i/dx=e_qi+(5/60)e_vi`。若活跃集合 A 的边界导数记为 `B_A`，自由集合 F 的局部导数应包含：

```text
K_A = B_A
K_F = -H_FF^-1 (s C_F + H_FA B_A)
```

在同一局部约束定义下更新 backward/value 传播，不能把状态相关活跃轴仍按固定 ctrlrange 的旧规则置零。活跃集切换处允许分段导数，但必须记录当前分支；无需伪造一个处处光滑的全局导数。

### 必须保持同路径及原时钟

- 候选 rollout、nominal 计算、最终部署调用相同受约束策略；不可只在真实执行最后加安全滤波，而让优化器继续评价另一条策略。
- 继续每 2 ms 产生一次目标，并 hold 住该目标；每个 0.2 ms 原 PD 子步照常更新扭矩及其原 clip。不得把控制更新频率悄悄改成 5 kHz。
- 当前瞬间 QP 可实现不保证后续十个子步均可实现。十子步 rollout/FD 中保留原 clip，记录每个子步的实际饱和与总误差；需要整 hold 保证时另行明确预测约束，不能凭瞬时区间宣称已满足。
- 如目标范围与状态相关扭矩区间没有交集，必须显式标记 infeasible，并采用预先声明的最小违反 restoration/原物理饱和语义；不能静默冻结、换 stand、跳过更新或把它算成功。最小违反量须按原物理单位解释，不顺便扩大原扭矩或材料有效域。

这不是完整稳定性证明。它优先处理已被同时观察到的 nominal 不可实现与反馈放大，并使随后实验可以判断：执行活跃集失配解决后，native 接触/支撑预测失配还剩多少。如果同类误差与失败仍然出现，下一主要修订应转向接触/支撑模型一致性，而不是继续增加 episode 数或堆 trunk 权重。

## 8. 可执行顺序及需补齐的最少证据

下一版开始前先用不超过约 30 分钟读取 D57 **已经存在**的首发窗口原始文件：A 9.12–9.30 s、B 14.63–15.10 s，以及训练中首次 load 后立即 domain-exit 的回合。先确认执行 source fingerprint、plan_id、两个状态时钟、十子步 clip、实际 contact 与保存的 reference；不为了取得旧数据再跑物理。未保存的 accepted +2/+4 ms 不可重命名其他数组来补齐。

两条正式流可继续同一 r29 控制数学、独立 seed 与可追踪 checkpoint lineage。它们不是只因 seed 不同就构成 r28/r29 因果对照；和 W39 比较应注明训练状态、版本及暴露时长差异。第三 GPU 并行任务可验证同一模型的批量 rollout/导数或局部求解实现，但其结果单独记录，不能在未对齐原执行链前替换正式控制器。

下一次关键窗口应随紧凑结果上传：实际 q/v、原始反馈修正、最终动作、原 PD 请求/饱和、QP 活跃集/边界/状态导数、H/C（仅关键窗口）、accepted 状态/动作、feedback reference、原点与首个执行 20 ms 的逐节点数据、十个物理子步的接触与材料力。若数组是必要证据就上传对应小片段；未记录的完整 horizon 不能靠补传恢复。

判读应同时报告：原合法 outbound/return/roundtrip；首次材料 load 后存活及材料域退出；nominal 自身饱和与反馈新增饱和的分离；actual−accepted 关节速度误差；trunk `(v,J)` 恢复；执行器区间空和 QP restoration。减少大 K 数量但任务仍 0，不足以宣称稳定；反之不能只用速度 cap 或累计训练步数掩盖底层执行链仍不稳定。

本审查没有修改任何运行源码、配置、原始结果或物理规则。
