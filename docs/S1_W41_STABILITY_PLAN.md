# W41：先稳定完整联合执行，再验证材料记忆收益

日期：2026-11-07，北京时间。依据是 W40 已完整交付的结果及 [最终审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/README.md)。当前审查未连接 D57、未在那里构建 qp2、未恢复 A，也未实现下述 r30；不能将计划写成实验成绩。

## 1. 目标与决策

长期目标保持不变：同一世界坐标可恢复材料被前足、后足及返程反复加载，利用其持久状态让未来支撑需求约束当前加载、落点和重访时间，最终在 Go2＋NX 上验证。S1 当前的首要缺口是原任务下稳定、合法的完整机器人通行。PANN→PPO→全身 MPC→PD→Go2/TM02 的联合采样、学习和评价已经实际运行；科学上的材料记忆优势尚未证明。

下一步执行三项有明确产出的工作：

1. **任务 A：qp2 正常构建后，续完 W40 A 原任务。** 从自身最新 ep314、9508/36、tail147 继续，原目标 9617/37 只差 109 条。不重置策略、优化器、RNG、祖先或 tail，不新建 phase。训练完成后评价自己的末次新 update37，E30 通过原判据才在同 case 继续 E60。
2. **任务 B：并行完成保持动力学的一致性定位，并据证据实现单一 S5-r30 控制变化。** 对已有真实输入核对饱和 actuator map、十子步动力学及其导数在候选/nominal/部署中的一致性；确定性接线或导数错误先修。若这些已一致，r30 应明确作为受约束控制方法实验，不能把全段不饱和投影预先认定为根因修复。正常构建后直接进入完整联合训练；不再安排独立刚地资格、smoke、恢复演习或大参数扫描。实现期间不改正在续跑 A 的 frozen artifact。
3. **任务 C：已授权的 GPU 扩展实验。** 与 CPU 工作并行，依次完成真实接触/导数定位、176 节点与候选的驻留批量计算、完整代价/排序/CPU iLQG 外层接入、独立闭环评价；若固定 MJX 路线缺乏收益，再选一个有依据的结构化 GPU 对照。详细清单见 [W41 GPU 计划](S1_W41_GPU_PLAN.md)，不再把工作限定为八个状态的诊断。保留 W40 负结果，新的实验有自己的版本和目录。

任务 B 构建好后，进入 W41 两条完整联合续训。默认等待 A 的 109 条收口到 tail0；这期间 B 实现与 GPU 诊断可以并行。若 A 再次发生新的技术中断，保存完整首次故障并修复；不要用丢弃 tail、回退父点、标成完成或无止境重复失败来绕过。已完成的 W40 B 不重跑，后续是明确的新 W41 phase。

## 2. 本轮证据与停止误判

W40 A 新增 915/3，B 新增 1024/4；共 64 个训练回合，60 次自然失败、3 次技术中断、1 次目标截断。自然失败为 38 跌倒、12 躯干接触、10 材料域退出。合法去程、返程和完整往返均为零；B 末次 E30 在 15.680 s 跌倒，A 末次评价没有真正运行。完整原件、计数去重和历轮对照见 [审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/README.md) 与 [历史评估](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/history-review.md)。

qp2 修复的是已复现的自由面未最小化就提前释放边界导致的零步长循环。12 个真实 QP 的 C++ 重放通过原证书；A314 从 qp1 的 608 次失败变为 3 步、scaled KKT 5.64466e-11、primal 0，9 个原生成功动作及 2 个旧 refinement 解逐 bit 不变。它没有改变 H、中心、约束、正则化或物理阈值。见 [技术报告](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/technical-error.md)。

这不等于控制已经稳定。r29 的瞬时 PD 投影确实工作：A314/B E30 首子步原始 PD 超限仅约 7e-6 Nm；同一动作保持期间的后九步最大原始请求超限却达到 18.788/28.492 Nm。实际 actuator clamp 仍守住原力矩上限，实际支撑响应取决于裁剪后的请求。54 个 loaded 训练回合中 53 个在首次加载前已经发生 nominal 或 feedback 的饱和请求。这定位了需要核对的控制分支，但没有证明预测器遗漏了裁剪，也没有证明饱和是跌倒的唯一原因；必要的恢复动作本身也可能饱和。

独立复核还修复了两个具体漏洞：同目录恢复时不再覆盖读取的原始 checkpoint，保存索引使用实际新路径；新 R12 E60/E300/E600 在原条件上增加已记录的合法去程、返程与完整往返要求。历史结果不改写。源码修复、12例实际重跑、真实 B E30 重评分和验证边界见 [独立技术复核](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/technical-independent-verification.md)。

当前同步仿真在等待 planner 时不推进物理。平均约 1.23 s 的墙钟规划会影响采样吞吐和未来实时部署，但不是本轮“物理多走了 1.23 s 没收到动作”的证据。逻辑 50 Hz 也不等于 NX 墙钟实时能力。

## 3. A 的现场续接：同一任务、原样本目标

执行端为 `autodl-d57:/root/autodl-tmp/MCP-RL`，环境 `work/envs/d57`，激活 `source scripts/autodl_d57_env.sh`，使用现有独立联网与持久 GitHub 登录。先核对精确 tmux 会话、PID、job、最新断点和工作树；已经在跑就不重复启动，不用前缀匹配误伤 GPU。

正常更新仓库时保留现场未提交研究修改和同时发生的上传提交，不强推、不覆盖 frozen 目录。若 qp2 还没正常构建：

```bash
python scripts/prepare_s1_go2_r12_support.py --build S5-r29-qp2
```

该产物基于独立 frozen `S5-r29-qp1`，逻辑契约仍为 `S5-r29`，保存为独立 `work/r11-frozen/S5-r29-qp2`。已有正确构建则复用，不能重建覆盖旧身份；本地记录输入回放已经完成，不在现场重复整个回放 campaign。

若现场仍以本次末次记录为准：

```bash
python scripts/start_s1_go2_r12_tmux.py \
  --job-dir outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261106T103028Z-w40-a \
  --resume outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w40-a/PANN_PPO_S5-seed1/resume-state-ep00314.pt \
  --controller-artifact S5-r29-qp2 \
  --no-wall-deadline \
  --session s1-w40-a \
  --network-mode academic
```

若存在更晚的同 job 有效断点，用实际最新值，不回退 ep314；若已正常完成则保留其完成状态。普通 resume 保留该 job 冻结的 workers，不另加 `--workers`。A 不能 `--new-phase`，B 已到 9617/37 不能被同 job 再启动。

同 episode 的原始恢复文件已存在时，新调用保存为 `resume-state-epNNNNN-restart-<time_ns>.pt`，后续读取 `progress.json` 指向的真实路径，不能假定最新文件总是无后缀名称。来源 checkpoint 保留；现场首次恢复核对样本9508、更新36、合法tail147及实际路径。本次只验证了文件保存逻辑，未在当前环境加载真实 A314 PyTorch 权重。

starter 会保留旧调用的 job/clock/执行契约/退出与未启动评价记录，并把当前有限 deadline 迁到历史、活动 deadline 设为 `null`。这不是删除原始证据。A 原先因 training_not_normally_completed 产生的行政评价记录不冒充一次物理 E30；完成训练后才真正评价。

## 4. r30：先对齐真实饱和保持动力学，再选择控制修复

首项工作是对 A314/B E30 可重建的最早事件固定同一 `x_0,u`，逐子步核对原 `tau=clip(60*(u-q)-5*v)`、接触分支、预测状态和实际施力状态，再对相同分支核对 composed A/B 与有限差分。接触或饱和活跃集变化要单独解释，不能将非光滑边界的任意两侧差分当作同一局部导数。实际闭环与采用另一条动作的 accepted 轨迹之间的差，不能直接称为模型误差。

据此只选一个可解释的控制改动：若发现映射或导数接线错误，r30 先修该错误；若预测/执行已一致而控制仍失稳，选择与真实饱和动力学一致的优化/反馈处理，或完整匹配 k/K/value 的受约束递推。**不预先要求所有保持子步的未裁剪 PD 都不饱和**：这是比原 actuator 物理界更强的限制，可能抑制恢复动作，必须作为有代价的控制设计来判断。

以下第4.1–4.3节给出“整段未裁剪请求可行投影”这个候选的正确数学边界，只有同输入证据与设计理由支持它时才采用；它不是必须先通过的新资格关卡。若采用，应记录新增查询成本与实际闭环效果。稳定性和实验吞吐一起判断，不因局部投影可行就声称完成完整 MPC+RL 目标。

### 4.1 可选保持段约束：避免把十个静态盒拼成假解

保持原 2 ms action hold、十个 0.2 ms 子步与十二维目标角。以当前真实可见状态和现有因果 patch 为起点，在预测里对同一目标 `u` 推进：

```text
x_0 = x
x_(j+1) = f_Δ(x_j, u; causal_context),  Δ = 0.2 ms, j = 0,...,9
c = original_target_clip(u_ff + s * K_old * δx)
min_u  0.5 * (u-c)^T H (u-c)
subject to original_target_min <= u <= original_target_max
           tau_min <= 60*(u-E_q*x_j) - 5*E_v*x_j <= tau_max, j=0,...,9
```

H 仍来自同次成功 backward 的 `Quu_reg`，不靠临时增大对角或放宽扭矩解决。十个约束对应每次积分前计算 PD 的状态。第十次积分得到 x10，但不额外把不存在的第十一个施力时刻加入执行统计。

q/v 随 u 变化，不能先用旧动作 rollout 出十组盒、取交集就称为解了这个问题；那样忽略了约束的输入依赖。优先做小规模序列线性化、活动集/近端 QP 与真实同输入非线性复算，继承 qp2 的“内部工作面条件与最终证书分开”原则。控制盒可以继续消元，**原生只解 box 的 QP 不能直接冒充一般不等式求解器**。可评估 ProxQP 的独立固定依赖，不能自动升级主环境。

### 4.2 用已有十子步导数，正确传播固定 u 的敏感度

对第 j 步 `A_j=∂f_j/∂x_j`、`B_j=∂f_j/∂u`，固定同一个 u：

```text
P_0 = I, R_0 = 0
P_(j+1) = A_j P_j
R_(j+1) = A_j R_j + B_j
T = -60 E_q - 5 E_v
∂tau_j/∂x_0 = T P_j
∂tau_j/∂u   = 60 I + T R_j
```

这样每次线性化只有 12 个决策变量，最多 240 条双侧力矩不等式与 24 条目标角边界。接触/限幅活跃集切换时导数可能不光滑，不能把这个局部模型当全局可行性证明。所有接受结果都按最终实际传输精度的目标，在同一预测器里保持十步重算，并保存最大残差及发生子步。

`HoldPredictContext` 必须包含现有预测需要的 time、q/v、act、mocap、四个因果 patch/userdata、task phase、接触与 coldstart 规则；每 worker 使用自己的 `mjData`，纯求值不能改全局材料历史、warm-start、下一 held target 或任务 phase。不得读真实 TM02 隐藏恢复状态给候选提供额外信息。

候选、nominal 搜索和部署调用同一个纯策略映射。保留现有 scale/alpha 语义：action candidate 显式 s=1，nominal 使用自己的 s，alpha 只进入前馈一次。原目标角 clip 仍在中心 c 中，不把 Qux 中心旧错误再引入。

### 4.3 不把局部投影冒充完整 constrained DDP

若选择上述投影路线，应准确称为“保持段约束一致的投影 iLQG”。原 K 只是产生 c 的提议增益；H 投影后旧 K 不再是实际策略 Jacobian。不能单独改 backward/value 后声称完整 constrained DDP。

若诊断需要冻结线性化 QP 的有效反馈，可对活跃约束 `A du + C dx=0` 解：

```text
[ H  A^T ] [ K_eff ] = [ H G ]
[ A    0 ] [ Lambda]   [ -C  ], G = dc/dx
```

这是固定 H、固定局部约束模型的导数；非线性约束的精确敏感度还需要相应 Lagrangian 二阶项。不要在局部诊断上暗示严格非线性敏感度或闭环稳定性保证。若决定采用完整约束 DDP，必须同时一致处理前馈、反馈和 value recursion，并作为另一个清晰版本，不能半改混合。

局部保持可行集可能为空。应显式记录最小违反 restoration，并保留真实 actuator clamp、目标角盒及原任务失败语义；不能靠改扭矩/材料阈值让它“可行”，也不能把每一次物理不可行都变成永久不启动实验的门槛。数值未收敛、非有限或证书失效属于技术错误，单独完整记录并拒绝该无效动作，不能静默 fallback 成功。

## 5. 实现与真实实验的边界

开发只需针对已有 A314/B E30 中最早可定位的 hold 饱和/支撑分歧做同输入核对，确认动作、接触规则、子步与约束求导一致。记录预测与实际同输入偏差，不能把实际闭环与 accepted 另一条动作轨迹之差叫开环模型误差。开发检查为实现服务，不另起一串资格关卡。

若发现更直接的确定性接线错误，先修具体错误并记录，不机械叠加机制。r17 已试过 native impedance 恢复并失败，r18 body-twist 路线已失败、r19 已回退；不把旧尝试换名称当新方向。r25 已撤回的 MovingSupport、SupportPlacement、support_share/requested_load、moving loaded-height/indentation、candidate-dependent lead clip 不复活。

保持原 Go2、TM02 .02 有效域、PD kp60/kd5、原扭矩、接触/跌倒/路线判据，保留 r28 trunk(v,J)、r27 C2、r26 因果 packet/十子步/TV、冻结 PANN 参数、658 维观察、2 Hz cap 与 W38 奖励。r30 此轮不同时改步态、奖励、动作维数、PANN 训练或物理。材料真正控制权限的扩展留在稳定基线之后，不污染当前合法 tail。

## 6. W41 两条完整联合续训

在 r30 已实现并正常构建、A 和 B 各自 W40 收口为 tail0 后，建立独立 `w41-a` 与 `w41-b`，分别从自己的最新完整 checkpoint 普通继承。A/B 的 actor、critic、optimizer、RNG、历史计数与 fork_identity 保持分开，不从同一个新父点再分叉、不重新 seed、不补归一化器。phase_lineage 明确记录 W40→W41 和 r29/qp2→r30 边界。

建议本次每流新增 **2048 条/8 次更新**，以减少过于频繁的人工收口。如果父点仍为 9617/37，则每流目标 11665/45、tail0；如现场更晚，目标由实际父点加 2048/8 生成，不能强制回退到文档数字。这是样本任务目标，不是墙钟完成时间。

现场实现需更新当前契约的 active_controller_build、parent_map、active_streams、训练目录、独立时钟路径和目标；不能直接对仍指向 W40 的契约执行 `--new-phase`。先将具体 r30 源码/冻结规格及 W41 父点映射推到 GitHub，再用 `start_s1_go2_r12_tmux.py --new-phase --no-wall-deadline` 启动相应声明流。看到真实 transition 增长后交接，不以“tmux 创建成功”冒充训练已经推进。

每支训练目标完成后只评价该支本 phase 最后新 update 的固定策略 E30；原同 case 合格才 E60。300/600 s 只在原前级真的通过后安排，不提前铺开 21-case 矩阵。新 R12 E60/E300/E600 使用 `bidirectional_material_roundtrips_v1`：在旧条件之外，合法去程、返程、完整往返均须达到1/5/10次；保留 `legacy_stage_criteria_met` 与 evaluator 源码哈希以区分旧评价。30/60/300/600 s 是物理任务长度，保留；后处理或评价不另设四小时墙钟完成上限。

成功优先级是：技术错误是否消失、合法通行是否出现并能复现、末次策略评价是否通过。同时报告首次 hold 饱和、持续比例、材料进入/加载、脚端冲击/压入、躯干接触、同输入残差和计算吞吐。loss 降低、reward 总和变化、站着存活、trunk 零违反或单次训练前缀更长均不能代替任务成功。W40/W41 仍属继续学习的工程对照，严格因果消融在稳定后用同任务/同信息/同预算设计。

## 7. 25 vCPU / 96 GB GPU 的并行方式

默认两条联合主线各 12 MPC workers，Torch/BLAS 各 1。25 vCPU 还要承担 controller/RPC、材料、I/O 与操作系统，worker 数不是固定等量 CPU 占用保证。普通 resume 不偷偷改冻结 worker；新 phase 的训练和评价 worker 必须一致记录。

第三任务可以使用 GPU，但 host 线程默认 1–2、编译最多 2 jobs，避免两个 MPC 主线同时遭遇大量编译或 JAX host 线程。记录同时运行时的真实样本/h、计划用时和资源快照；若竞争明显，优先暂停/降低第三任务 host 工作量。2×14 已有历史授权但不作为同时 GPU 工作时的默认，也不启动三个各 12 workers 的全身 MPC 流硬挤 25 核。

A 收口期间可由一组 CPU 跑 A，另一组处理 r30 构建/开发，GPU 同时推进扩展实验；W41 时恢复两条完整联合主线。GPU 使用实际空闲显存和可扩大 microbatch 的单独工作流，不另开第三条 CPU 密集 PPO 谱系；复用编译缓存。用户报告还有另一项 GPU 训练，必须保留它的资源和进程，不能把全卡空闲量都当本项目独占。

## 8. GPU 支线扩大到完整规划与闭环实验

用户本次要求扩大 GPU 工作。执行 [W41 GPU 扩展计划](S1_W41_GPU_PLAN.md) 的 G1–G5：真实状态接触/导数定位 → 驻留设备与节点/候选批量计算 → 完整 cost/phase/候选排序与 CPU iLQG 外层接入 → 独立完整联合栈 E30/E60 → 在已有证据指向时选一个结构化替代路线。每一步有明确产出，连续推进，不再只做一次小诊断后交接。

原 W40 8 fixtures、10份NPZ和 `aligned=false` 保持历史事实。新任务扩大为原八例加约24–48个有来源的真实状态；导数工作量从每节点49个输入扩展到实际176节点的分块计算，不能把8624个逻辑输入硬塞进显存。规划计时表明优先目标是动力学导数和forward轨迹，backward仅约占六项记录小计的1.98%。

等价迁移维持原FP64/接触/十子步/差分标准；确有结构性后端差异时允许以新预测器身份独立实验，并在原TM02 plant和原任务上评价。不以静默放宽对齐阈值通过，也不因不同算法不能逐bit相同而无限搁置有价值的支线。现有结果仍不支持宣布加速或主线热换。

新建 `w41-gpu` 入口与目录，不用含28h规则的历史W40入口。按阶段工作完成结束，后台无固定墙钟deadline。资源以当前共存任务和实际峰值选择microbatch；显存占用与利用率用于解释，正确完整规划/样本吞吐决定价值。

## 9. 当前统一结束规则与交付

用户 2026-11-07 指令覆盖旧交互 4h、训练 28h、评价 4h、GPU 28h 的预设完成时间。当前活动任务以样本/更新目标、声明的评价/GPU 阶段工作完成、明确技术失败或用户停止为结束条件。不得再添加后台 `timeout`、固定 deadline、到钟强停或 resume 时重置有限截止时间；保留历史记录，不改写当时实际运行条件。

网络请求超时、有限 QP/非线性 solver 迭代、原物理任务时长和 checkpoint 保存属于另一层机制，保留。无墙钟截止不等于无限错误重试或静默挂住：技术失败显式退出并保存首次原因；观察实际 PID、样本/更新和日志推进，遇到具体卡点再修，不能新加泛泛失败次数上限。

训练/恢复/评价/finisher/diagnostics/publisher 的新调用链已经支持 `None` deadline 和 `--no-wall-deadline`。已完成 B 的旧 deadline 仍属于 B 自己的历史，A 的共享 clock 迁移不能覆盖 B 的历史状态。旧 W40 GPU 入口按上一节退休。

交付先推紧凑结果与准确状态，再后台完成已请求原件；Git/index 的短写串行合并远端更新，禁止强推。完整 accepted/reference/K/H、Q_u/Q_xu、动作 c/投影 delta/活跃集合/KKT、十子步 hold 残差、每次 planner 返回原因与 policy age、学习 tail/版本/祖先、首次故障完整输入按已有事件窗保存，无损压缩。已请求大于 2 MB 的原始文件不能静默过滤。

只要 GitHub 已推送并核验，就不在用户电脑额外做归档。服务器大 checkpoint、完整 resume、视频及原件保留，用户备份后自行统一删除；不自动清理研究输出或上传副本。发布状态区分 queued/uploading/published，compact 成功不等于所有 raw 已交付。

## 10. 后续科研顺序与参考

稳定材料通行后，再依次验证：冻结 PANN 是否在实际重访中影响动作；候选是否传播可恢复材料状态；动作是否能够控制足时/落脚/载荷；无记忆、局部历史、完整地块记忆和 oracle 的同信息对照；跨恢复时间/噪声/路径泛化；NX 与实机。这些阶段不能用继续调同一个速度 cap 的 loss 代替。

新文献与版本状态更新见 [本次新增与深入核验](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/literature-update-20261107.md)，原 [12 篇文献记录](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-final-review/literature.json)：ProxQP 用于约束数值可靠性，IPDDP2/ProxDDP/2026 Control-Constrained DDP 用于完整约束轨迹优化，Chance-Constrained MPC 用于有测量依据的模型不确定性余量，DR-MPC/Parallel CLQR 用于结构化求解，Gait-Net 用于稳定后增加步时/落脚权限，MPC-guided RL 用于训练架构与 GPU 批量计算，差分接触仿真研究用于定位后端一致性。论文的速度、收敛和实机能力均不直接转移为本项目成绩。
