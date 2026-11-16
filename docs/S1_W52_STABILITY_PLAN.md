# W52：修复摆动参考，再用实际支撑交换决定是否进入原任务

**W52 执行更新（2026-11-16）：r40/r41 已实现并完成原 native/service 重编；D2/D3 各一次原 t=0→D8 已在 D57 tmux `s1-w52-pair` 运行，仍无原 E30 资格。状态与命令见 results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w52/phase.json。旧 301 原件纯数学核对完成；C 支撑高度重建误差来自实际 Agent XML 序列化精度，修后 8.88e-16，原比较器排序不变。下方审查时的 planned_not_implemented 属于历史状态。**
更新：2026-11-16（Asia/Shanghai）。依据结果提交 `a621d7d35e6016634b388a3aef0d39d8d008aab0`。**状态：planned_not_implemented；本次只完成审查与计划，不存在 W52 新实验结果。** 补充审查开始时远端 main 仍为 `28b8654`，没有新 W52 分支/Release 结果；不重开 W51，不跳号为 W53。

[W51 完整审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-review/README.md) · [控制原件审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-review/control/review.md) · [结果与资源复算](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-review/results/review.md) · [视频](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-review/video-review.md) · [启动提示词](S1_W52_START_PROMPT.md)

## 1. 决策与目的

新增依据见 [2026-11-16 补充复核](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-followup-20261116/README.md)、[命令/足力复算图](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-followup-20261116/command-support.png)与[六项补充原始文献](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-followup-20261116/literature.md)。D1 在 3/6/8 s 仍收到约 .22 m/s 前进命令，实际速度接近零；18 份材料 chunk 全为空。这进一步定位到命令到接触运动的实现，不把本轮静站归为零任务命令或材料记忆失败。

W51 已结束。D0 在 5.814 s 跌倒；修正命令后的 D1 完成 8 s、身体高度约 .289 m，但最终后退 12.449 mm，后 3.7 s 四足持续受载且没有支撑切换。两臂均未接触材料，原 E30 启动 0。**不提升 r38/r39 为正式已选版本，不重复运行已完成的 W51 D0/D1。**

W52 主问题是：修掉已证实的摆动曲线下穿后，几何意图能否经原 native 控制链转成实际卸载、落脚和向前推进。D2 是主实验，只有一个确定的几何改动。第二组 10 核可用时，D3 并行比较“相同几何修复 + 保留 B 与 S 两个独立优化方向”；它增加预算，属于次级实用机制对照，不是 D2 获准进入原任务之前必须通过的新资格。

目前没有证据证明 PD 支撑偏置是唯一根因、J 错拒了正常步态，或任意 K 限幅能够恢复稳定。W52 不顺带更改这些机制。若曲线修复后 S 仍四足承重，应如实交付该结果，并把受力/低层一致性列为具体剩余问题。

## 2. 已有证据，不再重开旧诊断

1. 有效 D1 的 301 份 IK 已逐份解析。650 个当前 swing 足×replan 快照中，360 个的已保存剩余目标节点全部不高于地面，96 个有节点低于 −1 mm，保存节点最深 −3.570516 mm。独立公式在保存节点的重建误差 2.36e−16 m；本轮未把这些节点统计冒充连续曲线极值。此处目标全在材料 x=.8 之前的平地，非材料高程误读。
2. 原式 `h_original(t)-H0*h_original(t0)-T*H1*hdot_original(t0)` 在实际足未按相位抬起时会抵消高度。当前节点位置/速度连续、IK 收敛均不保证剩余段仍有抬脚。
3. 301 次 S 全部数值有效并作 Q，299 次 J 高于 B；运动阶段只有 2 次正 homotopy 更新。直接 S 仅安装 1 次；最后 reference 为 `[2.14,2.49]`，8 s 时过期 5.51 s。
4. 已读候选中 S 自身没有正常卸载；存在前进但姿态/支撑不好的被拒候选，尚无被错误拒绝的合格步态。去掉 InputReference 只翻转所读 26 个 S 中 2 个排序，不能以此先降权。
5. D0 held 参考过期数为 0，但后续出现反馈尖峰、目标硬裁剪、QP 投影和明显执行 PD 饱和，最终跌倒。最大 Kδx=37.867 rad 出现在 4.05 s，晚于初始异常；不能把它冒称最初因果，也不通过强制刷新或简单截 K 来宣布修复。
6. W51 的首个 D1 因命令槽未填而技术无效；修正 `463e91b` 将原 mission 命令写入既有 userdata[78:84]。新实现继承此修正，不重新启用 twist cost，不把已停止 server 的 Connection refused 当作未解决随机网络故障。

这些均由已有原件得到。不得重跑 W49/W50 保存问题、错误 bridge、40 ms 准入、GPU F1/L/F2 或 W51 原对照来“再确认”。

## 3. 执行入口与冻结条件

先读取最新 main、根 AGENTS、本计划、W51 review/handoff；核对当前 D57 的相关 tmux、具体 job/PID、物理时钟、CPU 与显存占用。若已有同身份 W52 有效工作，续接它，不重复启动、不宽前缀 kill，不动其他项目。

默认服务器 CPU checkout `/root/autodl-tmp/MCP-RL-w42`，GPU checkout `/root/autodl-tmp/MCP-RL`，环境 `work/envs/d57` 与 `work/envs/w41-mjwarp`，以现场实际记录为准。构建和运行目录隔离，先完成真实依赖构建再启动；header/ABI 改动必须重编对应 service/server，不以修改版本标签代替实际接线。

D2、D3 均从原 evaluation t=0 完整初态开始，使用同 W44 A/seed1、checkpoint-014737、同归一化和确定性 actor，PolicyStream(training=False)。保留原 2 s Stand→outbound、因果 observer、原 Go2/TM02、PD60/5、原力矩、材料主应变 .02、路线/接触/跌倒规则、.35 s horizon、2 ms hold、.2 ms 物理。不是从历史不完整 plant 快照恢复。

W44 A/B 各 14737 transitions / update57、checkpoint-014737、resume ep00428/ep00417、全 actor/critic/optimizer/norm/RNG/祖先继续保留，tail0/excluded145。W52 新 PPO、donor 均为 0。训练 resume 不冒充 plant checkpoint。

### 3.1 新身份必须完整继承真实入口

现有代码不是按版本大小自动继承。实施 r40/r41 时同步接通 `prepare_s1_go2_r11.py` 的 build 分派与 `s1_go2_r11_runtime.py` 中显式截至 r39 的版本集合，包括 R33/R34 strict/R35、实际 native 身份、recorder、before_call、held 和 forbidden diagnostic environment 的原检查语义。`MJPC_W51_MOTION/MJPC_W51_RECORD` 当前仅由 S5-r39 开启，不能只给新 spec 改名。

**尤其保留原命令修复：** 当前 userdata[78:84] 写入和 W51 recorder 选择都受 `w51_fixed_policy` 合同字段影响。若创建 `w52_fixed_policy`，必须同步延续其语义，避免再次把非零 mission 命令变成零 seed 命令包；不因此启用关闭的 twist cost。native 可执行文件预期身份的合同读取也要支持新身份。用首次正常规划的同次 readback、实际进程与编译来源记录核验即可，不另加微物理准入。

## 4. D2 的唯一主变更：固定相位关键时刻的直接几何接续

建议控制器版本 r40（实现时记录真实名称，不占用已存在版本）。修改对象主要为 `w51_motion.h`、`w51_seed.inc` 及相应 patch/spec/prepare/recorder 链。可保留 W51 文件的历史含义并添加 W52 文件，但实际 compiled source 与运行版本必须一致。

### 4.1 保留的原量

- 原 2 Hz、duty .45、5 mm、足顺序/对角组、原绝对相位、世界 stance anchor 和已固定 touchdown。
- 原 swing 时长 .275 s；原 C2 上升时长 .075 s，apex 在真实名义 liftoff+.075 s，下降至原 touchdown。**关键时刻从原绝对相位和 segment 结束时刻推回，不能把首次 observed_start 当成真正 liftoff。**
- 不每 20 ms 重置 apex、锚点、落点、phase 或整步计时，不把未完成的抬脚一直推迟到下一个预测窗。
- 原 Gait 成本及其参考、原成本权重、风险比较和严格下降要求保持。几何种子修复不能偷偷改评价目标。

### 4.2 删除有问题的修正式，直接构造目标曲线

对当前 swing，从真实足位置与足速度生成分段五次 Hermite/Bézier；首节点用真实值，种子初始加速度取 0 并显式记录。**C2 是接缝连续，不代表已测量/匹配真实足加速度。**

- **apex 前：** 从当前 `(h,v,a_seed=0)` 接到固定原 apex 的 `(5 mm,0,0)`，再接原 touchdown 的 `(0,0,0)`。两段时刻保持原绝对相位。令当前段剩余时间为 `T`、末高度为 `h_end`，直接五次 Bézier 的六个控制点为 `P0=h, P1=h+T*v/5, P2=h+2*T*v/5, P3=P4=P5=h_end`；这同时明确了两端加速度为 0，不能继续复用旧的名义高度/速度相减式。
- **apex 后：** 已在空中并能按原剩余时间下降的足，可用同式 `h_end=0` 接原 touchdown；该段必须保持下降且不另造一个已错过原 apex 的新抬脚峰值。若真实足仍落地而已错过抬脚峰值，不能用一条水平拖脚、向地下插入的“剩余 swing”冒充摆动。该次整 S 明确不可用，转回正常 B 初始化与搜索；不停止 plant，不改变相位，后续原相位仍可再次提供候选。恰在 apex/段末时按原事件身份处理零时长边界，不除以 0，也不人为把事件顺延。
- **未来完整 swing：** 使用原绝对相位的完整 .075 s 上升与 .20 s 下降，原 5 mm 和固定落点。当前 stance 的世界锚点跨 replan 保持。
- x/y 从真实当前足位置/速度接原固定端点，仍沿原确定性几何规则；不增加步长、速度、步高、相位或 beta 网格。

高度基准必须明确：当前 segment 的 `z_touchdown` 沿用已冻结的世界落点高度，来自原 `s1_flat_ground_height + R10ContactRest(原冻结context, touchdown_xy) + 原足半径`；以它为固定标量基准令 `h=z_world-z_touchdown`，apex 世界高度为 `z_touchdown+.005`，`v` 为真实世界竖直足速。不得把每个未来时刻重查的地表高度叠加到 `h(t)` 后仍声称世界轨迹是同一 C2 五次曲线。未来完整段的 touchdown 仍由原 W51 几何规则确定。

不能仅因“五次”就假定曲线不下穿。对所生成多项式检查端点和区间内导数实根处的真实极值；凸包可作快速充分判定，但不能用固定时间采样漏掉极值。除相对固定 touchdown 基准的幅度/单调性外，足底间隙还要按当前 epoch 原生局部接触表面沿实际目标 xy 解释，不能把不同位置的地表高度混作相同零点。原 R10ContactRest 是冻结世界 patch 的分段函数；若轨迹跨其分段，极值处理必须包括原分段边界，不能只算世界 z 的驻点。没有足够上下文证明可行时明确 S 不可用；不更换地形模型，不把 W51 平地 z=0 的诊断假设延伸成整个材料区的新地形模型。

实施含义：原 `r10_contact.h` 的 patch 是双线性 tent 乘积，五次 x/y 下分段表面最高十次、间隙驻点最高九次；需包含 x/y 到 patch 中心/边缘的穿越时刻以及原 R10MaterialFoot 启用矩形边界，检查端点、单侧值和区间内真实根，或使用等价可靠的实根隔离。世界 z 的四次导数根只解决固定高度基准的极值。原 touchdown 冻结规则保持不变；W51 旧快照均在材料前平地，数学复核可按该事实简化。未来完整 swing 若 anchor 与 touchdown 不同高，起点相对高度应保留真实差值，不能默认为 0 造成接缝跳变。

初始真实接触可能有微小压入，真实速度也可能略向下。“精确保留负速度”与“绝不比初始更深”在数学上可能冲突。保留实际初值，在既有 IK 几何误差尺度解释微量不可避免部分，明确记录；不新增 `v>=0` 的浮点硬门，不能把毫厘级主动下穿合理化，也不能暗中归零速度、夹断高度后仍称原连续曲线。不可行时记录 S 不可用，不递归改 apex/加高/换端点追可行。

首个 Trot 切换 call121@2.002 的 FL/RR 原 apex=1.9385 已过；另一对下一 apex=2.1885，首次适合全 S 的旧轨窗口约 call128@2.14。旧 W51 301 次调用中只有 60 个“全部当前 swing 尚在 apex 前”的机会，但这不是修后轨迹的成功率预测。不能为避免最初不可用而重置 gait 或硬取消当前接触；也不能因此跳过整个早期起步窗口。

### 4.3 S 的控制表示和接受规则

仍从完整 B 构造 S，保留 raw center 的原支撑偏置、反馈、metric/scale/delta，通过 `delta_q=q_IK−q_B_joint` 增加几何意图。保留原 IK/限位求解与原 native rollout；不以裸 IK 或已经 projected 的动作替代 raw center，不强行写 contact/fz 或真实状态。

几何/IK 不可用时，不做 S native rollout，记录原因；正常 B 分支继续计算。几何有效时只增加 1 次 S native rollout；原失败/有限性检查不变。D2 按原 r39 规则：有效 S 可作 Q，即使 J(S)>J(B)，然后一个 first-valid backward 与原 9 个正 alpha；最终 S 和所有候选仍严格按原 risk/J/roundoff 与完整全局 B 比较，不能只比 S 好就安装。

当前尚未证明支撑偏置需要怎样重分配，故这轮先不改 Kp/Kd、执行 QP、PD bias、反馈裁剪或新加逆动力学/WBC。S 是否真正卸载由这一次真实 D8 内的候选和 plant 记录直接回答；不能把几何修复写成已经实现动力学可行。

### 4.4 必要核验直接服务于这处修复

复用已有 301 份 IK/真实状态，做一次无物理的修后曲线构造与端点/极值核对，记录可用/不可用原因。它是此确定错误的直接数学验证，不是新的物理实验矩阵。真实 D8 同时记录 q/v/命令实际传递、原库身份、候选来源及执行结果。只做本改动必要的编译/入口核验；不增加全仓 hash、mock、import/shape/smoke、无信息防御性测试或先跑一个微小物理任务再允许 D8。

## 5. D3：第二组资源上的次级搜索覆盖对照

建议版本 r41 = D2 的相同几何修复 + 下述唯一区别。D3 默认在两组独立 10 核余量充足时，与 D2 同时启动一次原 t=0→D8。资源不足优先 D2，D3 记未运行，不为等待它阻塞主实验。

每个 epoch 冻结同一个 B、S、当前真实状态、userdata、成本/风险上下文。S 有效时，B 和 S **各自**做一个 first-valid backward、各自原 9 个正 alpha；不共享会被后一个分支覆盖的轨迹、k/K/H、regularization 或 provenance。两族均使用原 r38 的当前时域/同 alpha 转移语义，alpha0 仍是完整 B 的缓存评分；S 直接候选与两族最终统一用原 comparator 竞争。

B 方向的计算不能依赖 S 方向先安装后的新状态，S 方向也不能以 B 方向的候选冒充原 B。以 epoch 入口的确定 regularization 状态分别开始，各自 first-valid；分支搜索期间不得写回真实部署 policy 或另一个分支的状态。全局选中正候选时，提交其来源分支的合法 regularization 状态；选中直接 S 时使用 S 分支的合法状态；保留 B/选中 nominal 时使用 B 分支的合法状态。对应分支没有有效状态则保留 epoch 入口状态，不借未选分支的状态代替。原风险引起的最终 regularization 调整在全局选定后只执行一次，不能两个分支各自先改全局。分支 μ/rate、first-valid attempt 和提交来源都记录，未选分支不得暗改下一 epoch。

这里的 `alpha0=B` 是全局缓存保留语义。S 族的正 alpha 趋于 0 时局部对应 S，不能声称它趋于 B；不新增零 alpha rollout，不为这族伪造相对 B 的 expected/ratio，继续为 null。所有正候选最终仍须胜过同一原 B。相同分数按预声明确定顺序保持 B/简单来源，不依赖线程完成顺序。S 不可用/无效时只走一个 B 方向，不能重复 B 两次。

每条 job 的总 native worker 仍为 10；两方向可在独立工作区顺序计算，方向内部的原 rollout/FD 可并行。不得为“并行两个方向”额外开两个不受配额控制的 10 线程 server。

现有 `W50Capture/W50Transfer` 会改 candidate 数组、archive、μ/rate 与 stopped 状态，不能原样执行两次来冒充独立分支。工作区还应包含各自完整 Q、native nominal trajectory、raw-center/Jacobian 缓冲和 first-valid 状态；最后才统一归并原 comparator 与提交。原每个新 epoch 的 μ/rate 重置保持，不把分支状态记录伪造成跨 epoch 累积。

D3 比 D2 多了计算，并非等预算因果对照。它回答“额外 B 搜索通道是否在真实支撑/推进上有用”，不能仅凭 J 降幅归因于双起点本身，也不能把 B 通道称为稳定保险。D0 已经说明该通道可能重新带入失稳。

## 6. 有限工作量与记录

| 工作 | 明确范围 |
|---|---|
| D2 | 1 次连续 D8，从原 t=0；原物理提前终止保留 |
| D3 | 可用第二组资源时 1 次连续 D8，同 seed/初态；不重复 D0/D1 |
| 典型 D2 epoch | 1 freeze + 10 nominal + 至多 1 S + 9 positive，通常≤21 full rollout |
| 典型 D3 epoch | S 原生有效时 1+10+1+18=30；几何/IK 不可用且未评分 S 时原单 B≤20；S 已消耗 1 次 rollout 后原生无效则单 B≤21 |
| 规模估计 | 若各 421 calls，上界量级约 21,471 full rollout；Stand/不可用/提前终止会减少 |
| FD/其他工作 | transition FD、risk FD、backward attempts、raw-center 纯代数、plant 子步分别计数，不折成 rollout |
| 新训练 | PPO samples/updates/donor=0；禁止用增加训练凑 GPU 工作 |

这不是硬墙钟预算，也不是要求做满次数。额外技术重启不得为了追成功；有真实实现/接线错误时先记录具体原因与已消耗工作，定向修复，只重启受影响身份；不得无限 retry。每个实际启动、技术无效、物理失败、诊断完成但目标未达成、blocked 与未运行分账。

记录原有三窗 0–2、2.002–4.3、4.3–8 s；提前结束只报告实际片段。覆盖实际 body x/z/vz/姿态、足/非足力、足间隙/速度、计划与实际支撑、raw center、反馈项、原目标 clip、QP 改写、物理 PD、policy generation 与 reference/K/H 来源。

对 S 另保存几何曲线有效性、原 apex/落点/相位、原初值与 a_seed、极值、不可用原因、IK 与 native rollout 的足轨迹/力/成本差；D3 记录两分支独立工作量和最终来源。沿既有内容寻址记录保留事件 B/S/Q/IK、完整候选轨迹和所选对象；特别避免 W51 call216 只有元数据而缺候选 rollout 的解释空洞。不是要求保存每个 FD 中间数组。

记录器必须在本次 r33/search ring 已读入后，保留实际选中的 direct S/正候选及关键分支来源变化的完整 candidate、accepted rollout 与完整 policy。原 W51 direct S 不是事件触发，仅在父类 plan 内过早触发可能仍漏掉同次 ring；不得为补记录再跑 rollout。几何首次可用/不可用、失稳窗沿原有界事件机制保留上下文。

保持 `previous_prediction` 的原语义：取上一个实际应用完整 policy 的 accepted trajectory 与下一实际状态的精确匹配节点；没有节点或超出 horizon 明确 unavailable，不拿旧 feedback reference 末端夹持冒充预测。动作分别记录 before_clip→ctrlrange 后 center_before_projection→QP 后 u_star→float32 u_held，物理 PD 请求与实际 actuator force 按各自原积分时钟判断。`trajectory.npz` 的逐足字段与 native `r33_foot_rows` 为 FL/FR/RL/RR；native gait 枚举及 IK/motion 为 FL/RL/FR/RR，两种逐足字段用 [0,2,1,3] 显式对应。

## 7. 如何推进到原任务，何时停止叠加方法

主要判断是实际持续支撑交换、卸载/落脚与向前推进，身体正常承重、未转为膝/躯干撑地；同时查看 native 预测是否与实际部署一致。诊断 1 N、曲线极值容差等只用于解释，不新增任务成功或物理停止阈值。原许可的短腾空保持。

- **D2 按原规则自然结束其一次 D8，并读取完整实际轨迹，满观察数据已支持可持续正常支撑推进：** 优先冻结简单 D2；原物理/技术提前终止仍保留原结果，不用早停片段代替满观察，不以尚在运行的前几秒提前完成本判断。不要求 D3 也通过。D3 若仍在运行可以自然完成，D2 可在已释放的 10 核上启动正式 A E30；待另一组资源释放再启动 B。任何时刻最多两个重 native job，冻结后不因后来的 D3 分数更低更换正式机制。
- **D2 没有正常推进，而 D3 有清楚、实际的支撑/推进改善：** 完整读完结果后冻结同一 D3 实现，再启动原 A/B E30。不能仅凭 D3 多活几秒或 J 更低选择它。
- **两者均无正常推进：** 交两臂真实结果，不直接开原阶梯、不换 seed 追成功、不临时加权/加步高/改 horizon/训 PPO。分清“曲线仍不可行”“曲线可行但 native 未卸载”“已有可执行候选但原评分拒绝”“候选预测—实际部署不符”中的实际剩余问题。
- **只有 D2 运行：** 按其真实结果作相同决定；D3 未运行本身不构成阻塞。

选定后使用原 A seed1 / B seed2 各一次独立 E30，从原完整 evaluation 初态开始；D8 不能拼接或计为 E30。逐 case 通过才 E60→E300→E600：E30 实际合法穿材料、到 3 m 远端原容差并完整观察；E60 合法完整往返、回家稳定 2 s；E300/E600 各在原 270/540 s 路线期限内完成 5/10 合法往返并完整观察。一个 case 失败只 blocked 本 case 后级，不借另一 case 的资格越级。

即使之后 E30/E60 通过，也不能自动声称材料记忆因果收益或泛化。稳定原任务后，才处理候选内材料状态传播、RAW/PANN/冻结或打乱历史等匹配对照，以及外层动作权限、NX 实时和实机；本轮不改研究目标。

## 8. 第三线 C：补一个明确的 GPU 成本缺口，不重复占卡

W51 C 已完成 114×176×113 原残差→22 项 J 的 FP64 归约；无需重复它、F1/L/F2 或全量吞吐轮。峰 tensor 43.45 MiB、归约 .104 s 不能说明整条 GPU MPC 或 96 GB 利用已解决。

W52 C 默认 **0 新 GPU 物理**，用已保存原件定位并修正 MaterialSupportHeight 的 state 重建误差（W51 最大 9.00991e−6）。对照实际 native 代码、地表/足几何、userdata/活动分支与坐标定义；有字段才重建，缺字段明确停止在缺口处，不填零、不额外启动原 episode 补档。只做此次改变必要的已有候选 FP64 计算和原 native 参考比较，说明最大误差和排序影响。

记录 22 项各自活动分支覆盖；Trot 下为零的六个 Stand 项不得写成已覆盖活动 Stand，另外缺失项仍保留缺失。不能把 saved residual→J 称为完整 state→residual→J，也不把代数顺序一致当成接触动力学正确。76 条保存问题与 38 条错误 bridge midpoint 分开，后者只作代数对照。C 不进入正式安装、不阻塞 CPU，完成后自然结束。

## 9. 并行、后台与交付

现场有 25 核时按 D2 10 + D3/正式第二臂10 + C2 + 系统/主控/其他任务约3 分配；数学线程库设1，构建最多-j6且先完成，不外挂第三重 native 求解。尊重其他项目已用配额；CPU 和 GPU checkout/进程树/可写 build 隔离。

把真实 job 的 user/system 核时、可得 CPU 占用、GPU 利用率/整卡显存及 C tensor 峰值随运行低频记录，区分采样峰值与绝对峰值，不另做 benchmark 或满载压力测试。没有数据就写未知，不以 affinity 代替利用率。额外显存不会自动加速当前 CPU native/FD，GPU 路线继续以覆盖真实缺口为目标。

**所有新后台 `deadline=null`、`wall_deadline_enabled=false`，省略 `deadline-unix`。** 不设固定墙钟完成时刻、不因对话结束杀后台、不做到点 kill。D8 与 E30/E60/E300/E600 是科学任务的物理定义；原物理失败、明确技术/数值/资源错误、用户停止和有限工作量保留。单次网络请求 timeout 仍可有界，不能偷换成整个实验的墙钟 deadline。用户允许 tmux 正常后结束对话，running 必须如实记录。

紧凑结果和代码先推 Git；必要原件及全时段视频从 D57 增量直传 Release；短写串行合并最新 main，不强推，回执丢失先核远端再重试。每次修复/运行/发布记录实际 commit 与身份。GitHub 远端 commit/URL/size/digest 核实后结束交付，**不在用户电脑额外下载、复制、打包、镜像或归档**。保留服务器仍被引用的 checkpoint/resume/科研原件/视频，不自动删除；只用已授权服务器网络，不自动借用用户电脑的 VPN/Clash/SSH 转发。

最终报告分别写实现状态、正常步态/推进证据、原任务资格、材料记忆收益与 GPU 覆盖；给实际工作量、失败分类、commit/资产、真实 running 或结束状态和下一续接点。计划发布不等于实验成功。
