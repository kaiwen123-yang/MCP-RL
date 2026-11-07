# W40 独立技术复核与两项实际修复

复核日期：2026-11-07。真实结果基准为 `c438f5463659ab7c8d3a7f5eced251639e14b2ef`；复核起点是尚未部署到 D57 的 qp2/no-wall-deadline 候选版本。以下将实际数值回放、文件路径验证、原始轨迹重评分与尚未进行的物理验证分开记录。

## 1. qp2：重新编译并重放了全部 12 个真实输入

本次独立运行 `replay-qp2.py`，使用此前缓存的 MuJoCo 3.2.3 库，重新编译 native、qp1、qp2 数值函数，对 12 个已记录的真实 QP 输入逐项回放。不是复述已有 JSON，也不是随机生成矩阵测试。

| 验证项 | 本次结果 |
|---|---|
| 已记录的 native/qp1 输出 | 全部逐位复现 |
| qp2 原始 primal/KKT 证书 | 12/12 通过 |
| native 已通过解 | 9/9 不变 |
| qp1 已修复通过解 | 2/2 不变 |
| A314 新失败输入 | qp1 608 次释放/重新锁定失败；qp2 3 步通过 |
| A314 qp2 primal / scaled KKT | 0 / 5.6446565432203134e-11 |

阈值仍为 primal `1e-10`、scaled KKT `1e-7`；未改 H、c、约束盒或控制目标。此次结果与仓库 `qp2-replay.json` 一致，因此没有再加入一份大型重复 JSON。完整复现入口为同目录 `replay-qp2.py`，传入 `--mujoco-root <含 include 和 libmujoco.so.3.2.3 的目录> --output <临时结果路径>`。

A314 的矩阵是有限 SPD，条件数约 `2.964e8`，约束盒非空。qp1 将最终缩放 KKT 容差也用作当前自由面的驻点判断：自由轴梯度约 `0.006414` 小于约 `0.007201` 的阈值，跳过当前面求解；随后释放上界处梯度符号错误的轴，但耦合 Newton 方向朝盒外，步长为 0，重新锁定，重复到 608 次。qp2 要求先完成当前面的真实可行 Newton 步，再决定释放边界，解决了这个明确的算法分支问题。

**范围限制：这证明已知 12 个记录输入的数值修复，不证明完整控制器已在 D57 正常构建，也不证明闭环稳定。** A314 出错时机器人本身的姿态/进度已经很差；修好求解器不能据此推断它会恢复平衡。

## 2. 修复恢复训练会覆盖原始 checkpoint 的具体漏洞

实际调用链是 `train_s1_go2_r12.py` 在加载恢复点后立即 `checkpoint(resume_only=True)`；旧实现固定写 `resume-state-ep00314.pt`。因此在同目录从 A314 的 qp1 恢复为 qp2 时，新 episode 尚未启动就会覆盖原始 qp1 checkpoint，违背保留父断点的要求。

已修改：

- `scripts/s1_go2_r11_policy.py`：每次调用实例记录自己的恢复文件路径。若该 episode 的规范恢复文件已经存在，首次保存改写到带 `restart-<time_ns>` 后缀的新文件；同一次实例的后续保存仍原子替换自己的文件。`progress.json` 指向实际保存路径。`checkpoint()` 返回实际路径。
- `scripts/train_s1_go2_r12.py`：结果索引使用返回的真实路径，取消对规范文件名的假设。现有 finisher/handoff 从 progress 读取 checkpoint 路径，可接受后缀。

针对实际完整 `checkpoint` 方法做了文件系统回归：原始文件字节保持不变、同一次实例重复保存复用自身文件、下一次恢复使用另一个文件、新 episode 使用尚不存在的规范文件名、progress 指向对应文件。验证使用序列化替身以覆盖路径与原子发布逻辑，147 项 buffer 仍随 payload 保存。

**未进行真实 PyTorch checkpoint 恢复测试：当前审计环境没有 PyTorch，也没有 A314 的 `.pt` 实体。** 源码恢复链仍保留 actor/critic/optimizer/RNG/normalization/buffer；qp1→qp2 保持逻辑控制器与 reward 不变，147 条尾片段不经过跨逻辑版本清空分支。首次 D57 恢复应保留来源文件并核对147 条 tail、样本/更新计数与新 progress 路径，不能把本次替身测试写成真实训练续跑已完成。

## 3. 修复新 R12 长阶段漏检查合法返程

确认有缺口，而且不存在另一个已生效的上游判据：runtime 先 `joint.finish(result)` 写入 `legal_route`，再调用 `evaluate_stage`。旧 E60 只检查完整时长与 `first_completion_s`；E300/E600 检查 raw 圈数、时限、端点与 hold，未读合法材料往返计数。E30 先通过只约束首个去程，不能代替返程约束。

新 R12 的 E60/E300/E600 在原条件之外，要求现有 500Hz `LegalRoute` 记录的三个整数计数都至少达到该阶段目标：`completed_roundtrips`、`legal_outbound_traversals`、`legal_return_traversals`。缺失证据也不能通过。

新增输出：

- `qualification.route_criterion = bidirectional_material_roundtrips_v1`
- `qualification.legacy_stage_criteria_met`，保留原条件是否满足的独立观察。
- `stage_progress.legal_material_roundtrips`，含所需数、观察数、证据是否可用与是否完成。
- 原条件通过但合法计数不足时，reason 为 `legal_material_roundtrips_not_completed`；缺失记录时为 `legal_material_roundtrips_unavailable`。

旧保存结果没有被改写。训练、E30、历史 S1R11 评价保持原语义；新 evaluator 源码哈希可识别改版。物理失败、技术中断和预算中断的分类优先级未变。

实际验证：读取 W40-B E30 的原始 `trajectory.npz`（7841 个状态），重新评分后仍是 `body_fall / physical_material_failure`，首程材料穿越字典与原记录完全相同。再对 E60/E300/E600 各验证明确标记为人工构造的判据反例：raw 完成、去程合法而返程不合法会被拒绝；完整合法计数会通过；记录缺失会拒绝；历史 R11 判据不变。这是评价语义验证，不是新的机器人试验，也没有新 E60 成功。


### 固定基线及旧 reward 的计数支持

后续只读复核发现：`FIXED_S5` 使用 `JointMaterial`，旧 reward 的 `PolicyMaterial` 也可能没有 `legal_route`；若只在 evaluator 要求该字段，会把真实合法的固定基线错误判为证据缺失。已经补齐对应范围。

将原 `LegalRoute` 原样抽到仅依赖 NumPy 的 `scripts/s1_go2_legal_route.py`，学习分支继续使用同一个类。新 R12 评价若没有现成 reward 账本，则由 runtime 在每次真实 500Hz 控制状态、mission 更新之后，用实际 `mission.phase/laps`、足部 XY 与原生/材料合成的实际 GRF 更新独立计数器，结束时写入 `legal_route`。没有从 raw 总圈数猜造合法证据；不改变 reward、控制命令、物理积分或训练。该独立账本标记 `reward_version=null`、`account_role=independent_evaluation_only`。已有 W36/W38 账本继续原样使用。

`case_identity.execution_sources` 的现有源文件清单已增加公共模块，避免漏记实际计数实现。计数类的初始化、update、invalidate、terminal settlement 方法与移动前源码的 AST 一致。

另用真实 W40-B E30 轨迹和实际 mission transitions 重建公共计数器：使用7840个时间同步的500Hz状态及对应真实足载荷，明确排除1个沿用旧诊断字段的终止追加行，按实际失败时刻执行 terminal settlement。重建的 progress、合法往返数、两向合法穿越数、side exits、raw往返数及 passages 共7项，与现有 B 账本全部一致。这个检查验证共享计数逻辑和原记录的一致性；没有新机器人运行，也不声称已验证某条成功的固定基线长任务。

## 4. no-wall-deadline 调用链及历史嵌套字段

已检查 starter → train → runtime/PolicyMaterial → 独立评价 → finisher → GitHub/diagnostic publisher：新模式以 `deadline=None` 贯穿；新训练命令不带 `--deadline-unix`，评价等待为 `process.wait(timeout=None)`，到期判断 `expired(None)` 为 False。恢复时旧期限记入 `wall_limit_history`；重复迁移不会重复添加同一条历史。网络单次请求仍可有有限 timeout，不会变为整个实验的完成截止时间。

对整个 `scripts/` 搜索，以下字段均无读取点：`current_window_scope`、`window_budget_s`、`shared_background_training_budget_s`、`planned_implementation`、`actual_implementation`。因此保留这些历史嵌套文本不会重新启用 4h/28h 截止。当前运行开关来自 `execution.background_job.wall_deadline_enabled`；启动/恢复明确采用 goal policy。旧 W40 GPU 28h 入口属于历史工作流，不能用于新 W41 GPU 启动。

用当前 contract 验证了 goal policy、`None` 到期行为、迁移幂等性、保留历史嵌套字段和单次 I/O timeout。十个相关 Python 文件编译检查通过。本次没有启动后台训练，没有执行后台 finisher/publisher 的网络链路；不将静态链路检查声称为 D57 全流程已运行。

## 5. planner timer 与 r30 的证据边界

r29 patch 在每次 `OptimizePolicy` 开始重置 derivative/cost/backward/rollout/policy-update timer，并用当次起始时间输出 `wall_s`。正常 `NominalTrajectory` 每次覆盖自己的 elapsed；Iteration 正常和已记录的失败分支也用当次局部计时写回。因此 W40 正常候选数下，各 call 的 timer 可以跨 call 相加，不是累计值被重复计算。它们是被记录的 planner 分段墙钟时间，不等于 CPU core-seconds，也不能直接代表整机利用率或训练总耗时。候选数为零的异常配置不应套用这个结论。

r30 整个 10 个物理子步的 PD 约束可形成**局部线性化的一般不等式 QP**，不能继续当成简单 box-QP。其可行性依赖当前接触/饱和分支和局部导数；增加这层投影会增加每个候选上的求解负担。实际 actuator clamp 已保证施加扭矩不超界，进一步强制未裁剪的 PD 始终不饱和是更强要求，可能限制必要的饱和恢复动作。

因此，两条 episode 中饱和先于恶化的时间关联不足以证明它是唯一主因。优先对同一真实输入、同一接触/actuator 分支比对 composed A/B 与有限差分，并检查 nominal/candidate/deployed map 是否一致；发现接线或导数错误先修那个问题。只有证实 held-control 预测与约束失配主导后，才推进 r30。更直接的 MPC 方案是把真实饱和执行器映射及其一致的 held dynamics 纳入预测与导数/反向递推，或实现匹配 k/K/V 的完整受约束递推，而不是把额外防御投影本身当作长期 MPC+RL 研究目标。
