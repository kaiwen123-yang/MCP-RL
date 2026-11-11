# Codex：W44已按用户要求恢复，A评价/B训练/上传继续

**2026-11-11 11:47用户明确恢复：继续A的原E30评价、B的训练和既有上传，数据清理由用户同步进行。** A保留原评价进程组796687、原attempt及update57策略；B从自身ep00411（14551/56、tail70）普通续接，沿用原job/clock，目标仍14737/57，不新建phase。此前暂停已结束，时钟未重置；暂停及恢复记录见results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w44-user-pause.json。

2026-11-11现场更新：W44-B在301/512条、update56、tail45时因遗留learning.known_r25_startup_regression.enabled_for_planned_phase=true提前结束。W44声明早已规定自然物理失败继续采样，现已修正声明及B的执行开关，按原job/clock从ep00409普通续接剩余211条；保留原停止结果、45条tail和旧原件上传，不另开phase、不改控制器。A持续自己的任务，不热换活动进程。最新样本/回合数据见唯一索引w44_live_observation。

现有GPU G5已自然结束E30：满30s、约19.37h墙钟，material_traversal_not_completed，实际材料加载与合法往返均0；保留结果，不重启G5。

## W44 当前执行位置（2026-11-10）

S5-r33 已实施、正常构建并完成四问题有限比较，已冻结主机制 **N**：与 S 使用同一曲率度量 D，改用原生接触 FD 方向，完整重算约束 Riccati并由native评分决定安装。**W44双流已在原job/clock下普通续接运行，已观察真实transition持续增长。不要再次使用--new-phase。**
首次两个启动均在0物理步因artifact身份丢失17项父配置而中断，错误及gdb依据已保留。完整父配置继承现已恢复，同一二进制未改变；修正身份和失败原件见Release补充包。每流当前进展与磁盘续接点不同，实时计数见运行索引w44_live_observation；update57固定评价尚待训练目标完成。


两个4.00s问题中，N以更低原生成本给出+2.864/+2.620mm的0.35s预测前移；S仍后退。N晚段仍后退且不优于S，所有四个N计划均四足承载。这仅支持进入真实双流验证，不构成摆步、闭环起步、材料通行或记忆收益。

四例参数/权重/步态上下文一致，S/N的D逐字节一致，16安装策略完整身份与实际评分匹配。有效16调用有480次native rollout、113.092s；另16次重建selection索引错误造成的无效上下文调用（490次rollout、111.994s）完整保留并排除，旧物理回合重跑0。

两流分别继承自身W43 ep00410/ep00398，各14225/55、tail0、excluded145，目标14737/57（+512/+2），workers10+10，deadline=null。CPU使用/root/autodl-tmp/MCP-RL-w42私有可写source/build；现有GPU继续原作业，不重启、不热改。新W44网络声明direct，因为本次GitHub API的Clash/academic失败、direct成功；只改变新作业shell。

[四问题完整证据](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w44-mechanisms/summary.json) · [r33规格](experiments/terrain_memory/s1-go2-r11-S5-r33.json) · [运行索引](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/index.json)

> 以下W43审查和W44原始授权保留形成时语境；其中“尚未实施/尚未声明”不再代表最新状态。以本页上方和运行索引为准。

## 当前入口与事实

先读 [当前交接](docs/TERRAIN_MEMORY_PLAN.md)、[W43 最终原件审查](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W43-final-review/README.md)、[W44 实施计划](docs/S1_W44_CPU_PLAN.md)、[完整启动提示词](docs/S1_W44_START_PROMPT.md)。本文件及用户最新要求覆盖历史启动、墙钟截止、电脑归档和自动清理指令；历史结果与冻结契约保留。

- 审查基点 `6a94c94c8e21e3b50fcca40ab4363165dee2987e`，2026-11-10 北京时间 20:52:55。`1334f76` 的 update54 观察为中途历史，不是当前 CPU 状态。
- W43/S5-r32 两流已结束：A `job-20261110T035658Z-w43-a` / ep00410；B `job-20261110T035700Z-w43-b` / ep00398；各 +512/2 到 14225/55，tail0、excluded145。10 训练回合为 8 个 60 秒采集边界 + 2 个 16 秒目标截断，结果级技术/自然失败均 0。
- 各自末次 update55 E30 跑满 30 秒，均 `material_traversal_not_completed`，E60 blocked。出发段后退 13.2415/13.2178 mm，四足持续受力、无抬落足、材料加载与合法去返/往返全零。不能称稳定行走。
- r32 的问题 epoch 重置有效：12 份完整 planner calls 共 28,850 次，所有新 epoch 首次 λ/rate=1/1；同 epoch 停滞不重复领预算。最终 E30 accepted 后没有 λ=.5；实际 chosen-alpha native/proxy 比中位约 .236、均低于 .5。
- 已保存 34 个完整 accepted 计划自身全部预测后退；其中 32 个有后续实际 20 ms，base 预测误差最高约 2.30e-8 m。末次动作无 QP 失败/投影修正，500 Hz endpoint 没有力矩裁切迹象；此范围不能外推未发生的材料/摆足接触。
- 优先解释求解尺度与运动方向：λ=1 占已存 H 对角约 99.86%–99.93%；Stand/运动 Effort 直接曲率约差 1736 倍。末端一次成本下降约 97.6% 来自身高支撑项，非材料已加载证据。尚未证明减小 λ、增加迭代或改目标中的任何单项能令其起步。
- W43 Release 实际 391+388=779 项，共 5,357,128,793 字节。清单已与远端元数据核对，实际下载复算范围在审查文件另列；不冒称全量每字节语义重读，不重传已发布包。服务器检查点/原件保留。
- **S5-r33/W44 仍是待实施计划，根 r12 配置仍为已结束的 W43。** 不直接以旧配置 `--new-phase` 启动，也不重复 W43。GPU 用户确认仍进行；不因旧修复文档再执行 `--resume-from G5`。

长期目标仍是原 Go2/TM02 合法材料往返，以及同地块跨足/返程恢复历史的因果收益。冻结 PANN → PPO 2 Hz cap → MPC → TV → PD 链已经运行；当前起步、材料通行、候选内完整历史传播及记忆收益仍未建立。旧 r24 合法去程保留，不能代替当前资格。

## 现场与并行资源

默认 AutoDL 西北 B 区 D57，SSH `autodl-d57`。CPU 树 `/root/autodl-tmp/MCP-RL-w42`；GPU 树 `/root/autodl-tmp/MCP-RL`。主环境 `work/envs/d57`，GPU 环境 `work/envs/w41-mjwarp`。历史总容量 25 vCPU / 120 GB RAM / 96 GB GPU，实际空闲量必须现场读取。复用既有认证，凭据不进日志或 Git。

1. 开场最多五行后工作。精确核对 tmux/PID/job/clock、有效断点与未提交修改；已有有效 W44 或 GPU 任务就接续，不重复创建，不按宽前缀 kill，不覆盖并发上传和研究修改。
2. CPU 已结束且干净的树可以复用，保留 r32 产物/父输出；有活动任务或研究修改才隔离。新 CPU 可写 upstream/build 不与 GPU 共享；prepare 会改写该目录，不能软链。venv、静态资产、已完成父输出只读复用，不为改名归档整库。
3. 默认 CPU A/B 各 10 workers，Torch/BLAS 各 1，给 GPU host、发布、系统留余量；build `-j6` 在双流重负载前完成。现场充足可在新 phase 前明确 11/12，但两流合计与其他实际负载不得超过可用资源；普通 resume 保留冻结 workers。不另开第三条大 CPU 流，不热改 GPU/XLA/后端。以有效吞吐和运动信息判断收益，不以占满显存为目标。

## 本轮实施与实验

4. 以 [W44 计划](docs/S1_W44_CPU_PLAN.md) 的有限保存问题比较选择一个主机制：优先曲率尺度匹配的正则，native 局部方向与少量结构候选作原因对照。保持成本/物理/目标/初始上下文一致，缺失历史明确列出，不冒称精确 replay。不扩成参数网格或另一串准入实验。
5. 完整重算 r30 的约束 Riccati、k/K/V/dV/真实 H、native rollout 与候选身份。不能对旧 H 减 λ 后求逆冒充新求解；不能用代理下降替代 native 下降，不能人为修改 surprise 触发降阻尼。保持 r32 新问题数值重置与同问题有限重试。
6. 维护当前问题中已真实评分、可部署的最佳完整策略。nominal/incumbent 小缺口是契约修正，本轮最大遗漏改善不足 1e-9，不能当起步主因。均值/插值/精修结果须重新 native 评分；无有效候选明确失败，不吞错或安装未评分旧策略。
7. 比较后将一种获证机制写入 r33 spec/patch 和完整 build/prepare/runtime/recording/发布链，正常构建，声明 W44 真实父点/clock/目录，再启动双流。若有限候选仍无起步方向，先形成一个有证据的控制/目标修改，不能仅改版本名就追加原样 PPO。无新 smoke/mock、旧 QP 回放、纯 MPC/刚地/E30 准入链、全库 hash 或 worker 网格。
8. 保留原 Go2/TM02 .02 域、PD60/5、力矩/接触/跌倒/路线、r28 trunk、r27 C2、r26 packet/十子步/TV、冻结 PANN/658D/2Hz cap/W38 奖励。r25 撤回项不复活；不同时改变奖励、步态、时域、动作权限和物理。
9. W44 A/B 各自继承自身 W43 ep00410/00398 的完整 actor/critic/optimizer/normalization/RNG/fork_identity/phase_lineage。起点均 14225/55、tail0、excluded145；各 **+512/2 到 14737/57**。不混父点、重新 fork/reseed、丢 tail 或默认扩到 2048；两流同机制但共享祖先，不叫从零独立随机对照。
10. 新 phase 声明修正遗留 `phase_count_rule=+2048/+8` 描述，旧冻结 job 不改。每流自己的末次 update57 固定 E30，同 case 前级合格才 E60、300、600。合法去程/返程/完整往返三项在 E60/300/600 分别均须 ≥1/5/10，并满足所有原条件。60 秒训练采集不是 E60 成功。

## GPU、结束与交付

11. **保留现有 GPU 作业及冻结 W40-B/update37/r30 契约，不重启 G5、不热换当前 CPU 控制器、不重建其可写环境。** 历史 G1/G2/G4/G5 结果及错误分别保留。自然结束或明确技术失败后再按其原声明收集；不拿旧快照声称当前完成。
12. 所有新后台 `deadline=null`，不设固定墙钟完成时间，不用到点 kill。按声明样本/更新目标、评价完成、明确技术失败或用户停止结束。物理采集/评价时长、有限数值预算、单次网络 timeout 与实时性逻辑 deadline 保留；不无限重启。
13. 先推实现/spec/针对性证据与真实父点，再启动；观察两流真实 transition 增长后交接。记录真实成本分项、chosen-alpha 的 expected/actual、尺度来源、策略身份、预测与实际足部/运动、材料加载和合法路线；关键事件窗保存非赢家候选轨迹，不能只上传索引。
14. 结束先紧凑 Git 与逐条学习数值，所请求原件后台 Release。Git/index 短写串行合并 CPU/GPU 新提交，不强推、不覆盖并发结果；只发新增。发布元数据核验、实际下载与语义复算范围分开。
15. **GitHub 核验后不在用户电脑另做下载、打包、镜像或归档；不自动删除服务器 checkpoint/resume、原件、视频或上传副本。** 最终交接源码/产物、所选机制、job/PID/父点/样本/更新/tail、固定评价、GPU 当前状态、Git/Release 及准确续接点；计划、构建和存活不冒充稳定材料往返。
