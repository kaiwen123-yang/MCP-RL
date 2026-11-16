# W53 控制链审查：从几何输入增量转向完整支撑与摆动控制

日期：2026-11-16。仓库 `/workspace/scratch/f55a41043871/MCP-RL`，只读核查 HEAD `5e036c6918ae72bb795c4b1b5be08c1e109ade32`。本报告是下一轮方案审查，不是新控制器已经实现或通过的结论。未改仓库、未编译、未运行 MuJoCo、native solver、有限差分、QP、训练或新物理 episode。唯一执行的计算是从已保存原件复原控制代数及比对数组。

## 1. 对下一步的直接结论

**不建议继续把下一轮主机制限制为 `c_B + q_IK − q_B` 的改良。** W52 的几何修复确实产生了合格的 5 mm 足端目标，D3 的双方向也确实独立执行；首个有效 S 的 native 轨迹仍四足承重，发生在执行器截断之前。现有证据已足够决定换一个能够同时表达支撑力、摆腿运动和身体动力学的控制内核，而不是继续把更多责任交给同一个旧支撑闭环上的输入增量。

建议并行保留两个清楚的工程对象：

1. **N：完整恢复 R8/S3 的作者运动内核，接当前原 TM02 plant 和当前冻结高层命令，作为新的材料未知基线。** 恢复的是整组运动 cost、gait、预测接触/积分、full-state policy 与动作执行语义，不能只把步高改回 3 cm。旧刚地成功不继承为 TM02 资格。
2. **F：独立的接触力 MPC + 全身控制执行内核。** 同一接触集合生成支撑力和摆腿 p/v/a，全身动力学生成完整关节命令，通过原 PD 60/5 与原物理限力执行。不要再把其结果仅作为旧搜索的 seed，也不要让旧 B 的 J 比较器决定它能否执行。

两条路线共享原任务、plant、执行器、起点、冻结高层输入和物理判据，区别明确落在控制器。它们不是一次单变量根因消融；收益是恢复正常运动基线，同时检验更直接的控制表示。不能预先保证 F 成功，也不能把 W52 的失败归结为一个已被单独证明的唯一根因。

## 2. W52 的代码接线与 B/S 独立性没有发现假执行

### 2.1 几何与 IK 实际做了什么

`experiments/terrain_memory/w52_seed.inc:7–32` 从实际 q/v 和 userdata 读取状态，以世界坐标足端位置、世界 Jacobian 乘实际 qvel 得到起点 p/v。`34–60` 沿冻结的原 gait 相位、当前锚点和 touchdown 构造分段；不是把时间重新归零后另造一条步态。`w52_geometry.h:127–162` 使用直接 quintic、固定 apex 与连续曲线极值/表面边界检查，原 W51 曲线相消问题已被处理。

`w52_seed.inc:67–72` 随后单独指定身体轨迹：xy 按命令速度匀速推进，z 平滑到 support height，姿态为 yaw 增量乘实际四元数；`73–100` 对每条腿做三关节位置 IK。这保证的是给定身体姿态下足端位置可达，**没有生成与接触力、关节速度、身体加速度一致的完整动力学轨迹**。曲线内部 C2 不等于完整闭环状态及力的 C2。

腿序与坐标没有查到错接：motion/gait 使用 native `FL, RL, FR, RR`，native foot rows 和公开轨迹使用 `FL, FR, RL, RR`，分析时显式 `[0,2,1,3]` 转换。IK 的 geom 祖先关节映射、世界足端误差和 `mj_jacGeom` 相互一致。不能因两种腿序并存而认定错腿。

### 2.2 D3 真的给 B 与 S 分配了独立方向

`experiments/terrain_memory/w52_search.inc:2–49` 保存同一入场 regularization，建立各自 planner workspace、derivative/cost/backward/QP/candidate 缓冲，复制同一实际起点及冻结任务，在不同 Q 上分别线性化、搜索。每支从相同入场 regularization/rate 开始，不是拿 B 的成功方向重复贴 S 标签。两支使用同一有界线程池顺序提交，避免把十线程配额倍增。

`71–87` 的合并次序为保留原 B/nominal、B positive、direct S、S positive；选中分支的有效 regularization 才提交，risk adjustment 只做一次。`scripts/w52_native_source.py:28–43` 为此分离候选收集和最终提交。全局 `r35_reference` 是冻结后只读对象（`mjpc/r35_input_reference.h:9–32`），分支并未漏拷贝一个可变私有参考。

原始 call 128 数组提供独立的行为核查：

- D2 的 direct S（candidate010）与 D3 direct S（candidate019）的 accepted states/actions、reference states/actions、K、H **逐值相同**。
- D2 的九个 S-positive 与 D3 的九个 S-positive 轨迹/动作/K/H **逐值相同**。
- D3 新增的九个 B-positive 与 S-positive 不是同一族，最大动作差 `0.2494819293 rad`。

因此“D3 根本没做另一方向”不成立。完整日志中只有 24 次有效 S，意味着双方向的作用范围确实有限；不能把这误报为分支未接通。S 失效原因和全程占比由原件审查报告给出，本报告不再重跑资格检查。

## 3. 首个有效 S 的精确控制代数：足端目标改变了，支撑内核没有改变

### 3.1 来源与可复算范围

原件目录：

`/workspace/scratch/36ce91dd3b5c/w53-originals/extracted/job-20261116T110800Z-w52-d2-eval-PANN_PPO_S5-s1-D8-attempt01/`

使用 `w51-seeds.jsonl` 的 call 128、`r33-candidates-call000128.npz` 及对应完整 B/S/IK/context 对象：

- B：`w50-objects/4cfa48a3866c7fed3082fe66eb3a3913b897e6cff9ecbc453c609f1dc096f7cf.npz`
- S：`w50-objects/71c88189a961a9d83c045552ff75bb833dd012287ae3d026b7c6665b0db1ae77.npz`
- IK：`w50-objects/e9d1d0250e0b84e5afd9f2363d2e545b6c2553d574bdeacf7d461aab4dcb271c.npz`
- context：`w50-objects/52e41a932b6360dce167813e772a8bbff21694a58334fceffcd00a092de21c5e.npz`

复算脚本和结果为同目录外的 `control_algebra.py` 与 `control-algebra.json`。脚本只用 NumPy/SciPy，对保存的 quaternion、state、K、action 做插值和代数，不调用任何 physics、FK、FD、native solver 或 QP。完整 B 对象确认原 delta 数为零，B/S feedback scale 都为 1。

### 3.2 实际政策恒等式

`w52_seed.inc:102–125` 明确执行：

\[
\Delta q(t)=q_{IK}(t)-q_{B,\mathrm{native}}(t),\qquad
c_S(x,t)=c_B(x,t)+\Delta q(t).
\]

这里的减数是 **B 重新 native rollout 的关节状态**，不是 B policy 原参考关节状态。seed 先完整复制 B，再加 delta；旧 reference state/action、K、H、feedback scale 继续存在。进入后向方向时才把实际 S rollout 吸收为候选参考并 ClearDelta。

对 direct S 的实际状态，可重写为：

\[
c_S(x_S,t)=q_{IK}(t)+\underbrace{[c_B(x_B,t)-q_B(t)]}_{\text{原闭环的支撑输入偏置}}
+\underbrace{[c_B(x_S,t)-c_B(x_B,t)]}_{\text{旧反馈对新轨迹的响应}}.
\]

在完整保存数组中：

- 保存 delta 与 `qIK−qB_native` 最大差 **0**。
- B/S reference states/actions/K/H 全部相同。
- 重构 raw B 与保存最终 B actions 最大差 `1.11e−16 rad`；raw S 与保存最终 S actions 也为 `1.11e−16 rad`。
- `uS−uB−delta−feedback_response` 的最大误差 `1.11e−16 rad`。

所以这段 S 中 actuator ctrlrange clamp 和 QP 没有改变重构的 raw 控制值。无需再把首个不卸载现象归因于后来的 clip。

### 3.3 第一个 apex 的量级

call 128 起点 `t=2.140 s`，最近 apex 节点 `t=2.188 s`。腿序如下表均为 `FL, FR, RL, RR`。

| 量 | FL | FR（预定摆动） | RL（预定摆动） | RR |
|---|---:|---:|---:|---:|
| 足底目标高度，mm | −0.003956 | 4.999946 | 4.999946 | −0.003892 |
| 实际 native 足底高度，mm | −0.004244 | −0.002760 | −0.003354 | −0.004449 |
| native 法向力，N | 40.60197 | 28.20487 | 33.74749 | 44.11551 |
| calf 几何 delta，rad | .001259 | −.028178 | −.029048 | .001585 |
| B 原支撑输入偏置，rad | .081816 | .084310 | .084419 | .084416 |
| 旧 K 对 S 状态的额外响应，rad | .003878 | .004277 | .005562 | .005793 |
| S 最终输入减 IK 角，rad | .085695 | .088587 | .089981 | .090209 |
| S requested calf PD 力矩，Nm | 5.47737 | 4.19825 | 4.22226 | 6.19153 |

两条预定摆动腿确实降低了承重，但仍有约 28/34 N 正法向力；整个 S horizon 四足承重。全 horizon requested PD 与原力矩上限的最小余量仍为 **19.89918 Nm**。第一 apex 的 K 响应约抵消 FR/RL calf 几何 delta 的 **15%/19%**，并非 K 把全部几何增量抵消。更大的保留量是原闭环约 `.084 rad` 的支撑输入偏置。

第二个 apex 最近节点 `2.438 s` 换成 FL/RR 摆动目标，native 法向力仍为 `[25.54996,46.09328,49.02535,27.36062] N`，继续呈现减载而未离地。IK 的最大位置误差约 `1e−7 m`，远小于 5 mm 的目标与实际足底差。

**这不是要求把支撑偏置直接删除。** 位置型 PD 要产生重力和动态支撑力矩，本来就必须有 `u−q` 偏置。`qIK` 不是可直接部署的完整动力学输入。证据支持的结论是：旧四足支撑闭环的偏置和反馈在新的摆动计划下保留，而当前表示没有显式安排该腿卸载及其他腿重新承重。下一内核应生成新的动态力矩和参考，不能把“去掉 .084 rad”当成修复。

此结果也不证明旧反馈是所有失败的唯一原因：尚未执行“完全相同条件下仅移除偏置/仅重算 K”的物理消融，本轮也不建议为此增设矩阵。它足以说明继续只做几何输入增量的信息收益已低。

## 4. 控制表示中的具体不一致与证据边界

### 4.1 身体运动先被假定，接触动力学没有共同求出

首 call 128 实际身体 vx 为约 `−0.001218 m/s`，传入 IK 的命令 vx 为 `+0.070527 m/s`。`w52_seed.inc:67` 在整个 horizon 立即按后者匀速安排身体 x；到最后控制节点约应前进 24.54 mm，native S 实际仅约 4.13 mm。5 mm 抬脚的几何有效性并没有消除这一身体与脚的动态不一致。这里的速度是当时任务斜坡值，不能误写为后来恒定的 `.22 m/s`。

Trot 2 Hz、duty .45 本身有约 25 ms 的短暂全摆动区间。若实际遵守无支撑阶段，浮基需要按动力学响应；把身体高度当几乎不变的独立几何轨迹并非自动可行。当前代码可以用实际接触“不卸载”来满足软代价，但这不代表实现了期望 gait。

### 4.2 现有 moving cost 不直接表达支撑分配

继承的 `quadruped.cc`：

- `156–197` 的 moving Gait 主要是足高度 residual，另启用 stance-height；没有 moving 足 xy 完整任务或 swing reaction force 零约束。
- `210–223` 的 moving Effort 是 actuator force，Posture 回 home；Stand 的 u_ref 与支撑足约束只在 Stand 打开。
- `377–386` 的 MaterialSupportHeight 由于 moving-body-support 标志，在刚地移动也锁定绝对 body-origin 高度。
- `401–403` 的 R35 输入参考围绕冻结 incumbent 的实际输入，对所有候选使用同一参考。它增加离开原输入的代价，但没有证据能单独证明其权重就是失败根因。

另有两个软高度目标不是同一个量：`108–132` 的 Height 是 torso inertial COM 相对平均足中心高度，moving 目标 `.27 m`；MaterialSupportHeight 是 body origin 的绝对高度，约 `.29020336 m`。本次 resolved 站立参考中前者约 `.26284194 m`，与 moving `.27` 已有约 7.16 mm 差；两足升高 5 mm 又使平均足高度增加约 2.5 mm。它们可以通过付出不同代价折中，**不是硬约束数学矛盾**，但不能把这组目标叫作已经共同动态可行。

由此恢复 R8 时需要恢复完整 cost/执行族；仅调高一个 amplitude 还保留当前其他项，无法清楚回答旧的正常运动是否仍可重现。

### 4.3 材料接触的潜在限制不能误当 W52 的实际原因

`w52_geometry.h:150–153` 对初始 clearance 小于 −10 μm 的段直接判失效。当前运行 `scripts/s1_go2_r11_runtime.py:517–525` 把材料 rest height 锚定为实际足底加 `Fz/k`；实际配置 `k=1,227,272.727 N/m`。在自身锚点下，若 Fz 大于约 **12.27 N**，材料的正常受载压入已超过 10 μm；静态单足约 37 N 对应约 30 μm，两足支撑约 74 N 对应约 61 μm。

因此未来进入材料时，不能把受载材料的自然压入当成与刚地穿透相同的几何失效。这个界限应在新 F 的接触/支撑模型中统一处理，不能强制所有 stance 法向速度和加速度均为零。**W52 尚未到材料，因此这一点不是 W52 当前倒地的原因，也不值得再单独开一轮小修实验。**

## 5. F 路线应替换执行内核，不应只替换 seed

公开 SRBD/centroidal MPC + WBIC/WBC 的价值，是让接触力、摆足任务和浮基动力学成为同一套控制决定。具体文献/官方代码依据见同次审查 `baseline-and-literature.md`，尤其 Kim 等 arXiv:1909.06586 的 §III、§IV 以及 MIT 官方 `LocomotionCtrl.cpp`、`WBIC.cpp`、`WBC_Ctrl.cpp`。本报告以下为项目适配建议，不把论文结果当作 Go2/TM02 验证。

### 5.1 一个可执行的完整范围

**计划层。** 同一实际状态和冻结高层任务输入，生成明确的 body reference 与 contact table。接触中的腿进入支撑力变量，摆动腿进入 p/v/a 任务且其规划反力为零；时间、腿序、frame 与实际反馈统一。摆动反力为零是计划约束，不能在 plant 中强行关闭真实碰撞或接触力。身体加速度须满足浮基动力学，不能继续独立假定命令速度瞬时达到、全摆动仍严格恒高。

F 建议一次冻结 `Trot 2 Hz / duty .60 / clearance .03 m`，明确作为新内核的参数选择，不与 R8 `.45` 软 gait 宣称同语义。`.60` 对每个对角对给出 .30 s 支撑、.20 s 摆动，每半周期有约 50 ms 四足支撑重叠，避免计划中无支撑的 flight；这符合当前先恢复稳定运动的优先级。它不保证实际稳定，也不把两足对角支撑变成静态稳定面。新的 swing p/v/a 必须随 .20 s 摆动周期共同生成，冻结一个 C2 曲线族与边界条件，**不能复用 W52 的 `apex=end−.20` 或旧 .075 s 上升 + .20 s 下降常数**。3 cm/.20 s 的真实跟踪和材料冲击仍由原物理任务判据判断。R 路线继续保留 R8 原 `.45` 软 gait；两条路线不做同参数因果比较。

**力规划。** 一个 SRBD/centroidal horizon 优化支撑力，满足原摩擦/单向接触和来自 Go2 的可执行力矩边界；使用原因果材料信息定义支撑高度/压入/相对速度，不能读未来真实材料状态。初始版本选择并冻结一套完整 gait、参考和控制权重来源，按 Go2 实际模型适配；不要为每次失败增加另一套 alpha、幅值或恢复动作网格。跨内核比较以物理任务结果为准，不比较不同 J 的数值。

**全身执行。** 使用同一 Go2 模型的质量矩阵、bias/passive、world contact Jacobian，求满足浮基方程的关节加速度、接触力和 actuated torque：

\[
S_f(M\ddot q+h-J_c^Tf)=0,\qquad
\tau_j=S_a(M\ddot q+h-J_c^Tf).
\]

`h` 的定义必须与该模型 bias、passive 和外力记账一致，避免把材料力既作为接触变量又作为额外外力加两遍。MuJoCo 的广义速度/旋转切空间与世界足 Jacobian 也必须由同一约定构造。全身 task 应产生 q_d/v_d/τ_ff 或等价的 total τ_d；不能只取 IK 角，再叠回旧 B 的 raw-center、K 或 InputReference。

**原 actuator 内表达完整命令。** 现有 gear=1、未截断 PD 为 `τ=60(u−q)−5v`。若 WBC 输出 total τ_d，可在每个实际 500 Hz 反馈点精确换算：

\[
u=q+\frac{\tau_d+5v}{60}.
\]

若设计的完整命令为 `τ_d=τ_ff+60(q_d−q)+5(v_d−v)`，则等价地：

\[
u=q_d+\frac{\tau_{ff}+5v_d}{60}.
\]

前者也覆盖采用其他 joint tracking gains 后得到的 total torque。两者都不要求改变真实 PD 参数或绕过 actuator 直接加 torque；但是 ctrlrange/力矩截断会破坏精确恒等式，必须让规划和 WBC 约束原执行范围，并记录实际执行偏差。旧 R30 的 H 是旧 iLQG policy 度量，不应无说明套在 F 上；F 应明确定义自己的关节/力矩可行约束，而不是偷偷借旧度量把结果投影回旧中心。

**控制器身份。** F 是独立闭环，不经原 R34 strict J comparator 获准后才部署，不从旧 B 的 K 获得 full-state feedback，也不在失效时悄悄回到另一个未记录内核。它每步自己的 body/swing feedback 和 WBC 构成反馈闭环。需要记录实际执行的控制器身份、source time、reference/force/contact 状态与命令。原物理停止条件保持，规划数值失败也须明确终止或按预先定义且明确记录的控制行为处理，不能临时再开一种恢复机制。

### 5.2 为什么不选“动力学一致 seed + 旧搜索”作为这轮主线

这当然比纯 IK 输入增量更完整，但仍可能被旧目标和 incumbent 比较器压回原支撑轨迹，造成“候选有正确力但没有部署”的新不确定性。它也需要同时构造新的 q/v/reference/K/H，工程量并不明显更小。用户当前需要恢复实际任务进展；与独立恢复 R8 并行时，直接 F 执行能更清楚区分“正常运动内核可用”和“接触一致动力学控制能否改善当前任务”。因此本轮不应再只交付另一个种子族。

## 6. 足够少而有信息量的实际推进

N 与 F 都完成必要的可运行适配、模型/命令代数检查及既有记录接线后，各进行**一次原 E30 任务**，从原始 t=0 起点到原观察终点或原失败条件。它们是明确的新机制范围，不是把 W52 D2/D3 改名重跑。无需先重跑历史 9+3，也无需新建 D8 admission matrix、多个短 horizon 或故障注入系列。某条路线完成后可独立开始，不必因另一条实现工程较多而串行阻塞。

这样一次任务本身已覆盖最有用的顺序：正常支撑交换与推进、接近材料、实际材料接触、路线完成及终段稳定。若在刚地再次失败，已有完整命令/接触/力证据可定位它属于哪一层；若进入材料，再沿原任务判据看结果。成功仅使该配置/case 进入原计划已规定的下一阶段，不自动获得其他 case、记忆收益或硬件资格。后续阶段由主计划与实际结果推进，不在此另造防御性矩阵。

已有记录应增加或复用的最小语义是：

| 控制意图 | 必须能与实际对齐的量 |
|---|---|
| body/reference | 原命令、p/v/a、orientation/angular rates、source time |
| contact/swing | leg identity、phase、计划接触、实际接触、足端 p/v/a 目标与实际 |
| force/WBC | MPC desired force、WBC corrected force、τ_ff/τ_total、动力学 residual/求解状态 |
| actuator | q_d/v_d、raw u、final u、requested/actual PD torque、ctrl/force clipping |
| material | 原因果估计/锚点、真实 strain/压入/力及原判据 |

这不是要求新增 offline replay 或 solver 重算，只是完整控制器本来拥有的变量在正式任务中落盘。记录器不能继续仅给 `action_after_clip` 一个名称就把 ctrlrange、QP 和物理 force clip 混在一起。

## 7. GPU 第三线的有限实际任务

W52 C 已完成 material-height 的真参数精度核查，不应再次包装一项小 residual reduction 作为新 GPU 进展。若并行保留 C，应服务 F 的真实数据接口：**在已有 W52 状态批次上，实现 Go2 FP64 批量 FK/Jacobian/动力学矩阵与 WBC/QP 矩阵装配路径，并与同一模型 CPU 基准逐字段比较，报告实际吞吐、数据搬运和显存。**

可先固定于 F 真实需要的一组输出，避免声称完整 GPU MuJoCo/contact/native MPC。已有 W52 状态主要覆盖刚地和失败前状态，材料接触覆盖不足应明确标注；不为 GPU benchmark 新跑物理 episode，也不补零冒充材料状态。对昂贵的 M/h 实现，若当前时间只能可靠完成 FK/J/装配子集，就明确交付子集，不把未完成模块写成支持。

批处理本身可以有实际工程价值；但一次控制的 12 维小 QP 未必适合 GPU，采集到的真实时延如果不占瓶颈，就应报告未有收益。96 GB 显存不是需要填满的指标。CPU 核配额也应按实际并行度使用；两个重任务各最多原 ten-worker 范围，其余留给独立实现/分析，不把跑满 25 核作为新的成功条件。

## 8. 覆盖边界与可直接引用的源码

已实读当前 W52 geometry/seed/search/source generator/recording 与相应 r40/r41 configs；已核对 r11 runtime 的 build gate、motion/dual env、actuator 和材料锚点；已读 inherited W50/W51 generator 及本地冻结 r37 policy/planner、r35 quadruped/input-reference。当前工作区没有完整展开的 r40/r41 source tree；其增量由仓库 generator/patch 和记录中的 binary identity 交叉核对。原件提供了 call 128 完整 policy 和 native arrays，与该代码语义吻合。

关键位置：

- `experiments/terrain_memory/w52_seed.inc:7–60,67–125`：实际状态、geometry、IK、输入增量、native rollout。
- `experiments/terrain_memory/w52_geometry.h:127–162`：直接曲线和连续极值/clearance 条件。
- `experiments/terrain_memory/w51_motion.h:24–44,46–74`：绝对 gait 相位、world touchdown、真实观察锚点。
- `experiments/terrain_memory/w52_search.inc:2–49,71–108`：分支隔离、确定合并、regularization 提交。
- `scripts/w50_native_source.py:28–41,124–168`：共享 exact raw center 及原政策导数语义。
- `scripts/w51_native_source.py:37–47,72–82`：Q 与 B 的职责、完整旧 bias 保留这一原设计。
- `/workspace/scratch/f55a41043871/w50-control/source-r37/mjpc/planners/ilqg/policy.cc:113–195`：插值、StateDiff、K、delta、clamp、QP 的实际顺序。
- `/workspace/scratch/f55a41043871/w47-source-r35/mjpc/tasks/quadruped/quadruped.cc:54–84,108–223,377–403`：moving/Stand 激活、不同高度量、moving foot cost 与 InputReference。
- `/workspace/scratch/f55a41043871/w47-source-r35/mjpc/r35_input_reference.h:9–32`：全局冻结参考。
- `scripts/s1_go2_r11_runtime.py:517–525`：材料 rest height 的因果锚定。

本报告未声称逐字节阅读全部历史轨迹，也未通过新运行证明任何替代控制器有效。全程实际结果、视频、资源占用与 R8 历史/文献核查分别在同轮另两份审查中负责；此处的强证据是具体源码语义、原始 call 128 完整对象和可复算的闭环控制代数。


当前实际实施范围以仓库 `docs/S1_W53_STABILITY_PLAN.md` 为准；本文保留独立审查的证据与建议形成过程。
