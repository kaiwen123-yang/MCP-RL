# W49 Task C：一次同步初始化、固定 held 的首 2 ms 干预

本建议只基于已提交源码与元数据，不执行 GPU、不连接 D57、不下载原件、不修改仓库。审查基点 `61e16975fa6b38dc8e2d31779c2bd82f1aaeb8b6`。

## 结论

可以在现有 `work/envs/w41-mjwarp` 中安排一次有判别力的新实验。建议只取 **B158 nominal** 一个科学输入，使用 **2 个完全相同的 GPU world**，冻结同一个实际 FP32 held 命令，比较三个初始化/恢复条件，每个条件只执行首 **2 ms＝10 个 .2 ms 子步**。核心预算 **60 GPU world-step**，若现有版本必须执行一次双 world 编译预热，额外最多 **2 world-step**，合计上限 **62**。无新 optimizer solve、FD、PPO、原 plant 或 350 ms rollout。三个条件是重复测量与干预，不是三个候选参数。

这次回答的问题是：**在动作完全固定、所有可读取的持久状态已核对的条件下，旧式部分 reset 是否改变首 hold；完整恢复后是否可重复；相同初态的两个 world 首次在何处不同？** 它不承诺 GPU 与 native 等价，也不能仅凭一个二世界批次替历史 24 世界批次证明确定性。

## 一、为什么现在值得做，而不是继续只读旧数组

实际 `scripts/s1_go2_w46_gpu.py` 中：

- `make_data(m, nworld=24, nconmax=128, njmax=768)` 之后，先执行一次真实 `load(); step(); synchronize()` 预热。
- `load()` 只回写 `qpos/qvel/userdata/time/ctrl` 并清 `qacc_warmstart`；接触、约束、加速度、力、求解器工作区等没有通过该函数完整重新初始化。
- 之后捕获单步 CUDA graph，再次执行同一个部分 `load()` 后开始统计。旧数组保存的 `initial` 是 cast 前的 host `qpos/qvel`，没有证明真实 device Data 的完整初态相同。
- 控制器初始动作通过 GPU FP64 反馈公式加 native host H-QP 得到，写入设备 `ctrl` 时 cast 为 FP32。此后每个 2 ms 又闭环计算动作，因此旧 350 ms 差异混合了物理首次差异与反馈放大。

这构成一个可以直接干预的初始化问题，但**还不能说旧 reset 已造成错误**：大量派生数组可能会被每个 step 正确覆盖。新对照必须用结果检验，而非从“没清某数组”直接推断根因。

W48 已确认六组重复身份均在首 2 ms 后出现 GPU 差异；native 对应重复组全程完全一致。选 B158 nominal，旧索引 `[18,21]`，执行身份 `caae093842ea5017f5789daebc158b90f4093249381e202e2a938f4194f3fdb5`，`origin=2.739999999999731`。选择依据是输入完整、无需缺失扰动且旧重复组的全窗 held 差明显；**没有声称它的首 2 ms 误差最大**。

## 二、先冻结真实执行身份，随后才计 physics

新建独立 W49 入口及输出目录，不能调用硬编码 W46 路径的旧 `main()`。可复用其已核函数，但不得 import 后间接启动旧实验。

1. 用服务器已有 `B158/input-state.npz`、`input.json`、`original-baseline.bin`、`original-policy.bin`、W46 `paired-states-actions.npz` 和模型，核 SHA。核两个 policy 的执行数组 `states/actions/times/K/H/scale/representation` 相同，忽略不参与执行的 plan/header 差异。
2. 科学输入沿用该 W46 诊断的 cold-start 定义。构造一次 canonical CPU `MjData`：本模型的 `mj_resetData` 默认值加上明确输入的 qpos/qvel/time/userdata 及文件中实际提供的其他积分状态。使用本安装版本 `mjSTATE_INTEGRATION` 的尺寸/枚举，加字段清单核读。它不是历史原 plant 完整 mjData 的恢复；缺历史字段时使用声明的 reset 默认值，不能把它们假称为已保存值。
3. 完整列出非零维度的 act、mocap、eq_active、qfrc_applied、xfrc_applied、plugin/history 等该版本存在的积分输入；不存在或长度 0 要明确记录。CPU canonical 复制用本版本正式 `mj_copyData` 或同等受支持接口。GPU 导入/复制先检查本机 package 的实际接口与源码，不凭假设写一个不存在的 `put_data/reset_data` API。
4. GPU 必须由同一个 canonical 输入构造两个 world，控制 cast **只做一次再复制相同字节**。记录 host FP64 与 device 实际 qpos/qvel/act/time/mocap/userdata/ctrl 全部回读值，双 world 应在对应有效元素上逐 bit 一致。
5. 同时列出 MJWarp Data 的全部持久数组、标量和嵌套 contact/constraint 工作区，保存 field name/dtype/shape/bytes/hash、有效长度与初始化方式。对索引/哨兵使用构造器合法值；不能对整个对象盲目清零。packed contact/constraint 的 `worldid`、地址偏移不是物理差异，比较时保留原始排列并另按各 world 的局部编号归一。
6. 源码核哪些 solver scratch 是每次先写后读，哪些持久状态需恢复，哪些是 graph 持有的缓冲区。所有会在本步先读的持久输入都纳入完整 snapshot/restore。仅仅 `qpos/qvel` 相同不能通过初始化检查。无法触达的 driver/allocator 状态应如实列为范围限制，不能伪造 bitwise 完整性证明。
7. 模型/solver 原样冻结：同一个 compiled CPU MuJoCo4 模型经 `put_model` 生成 GPU 模型；dt=.0002、implicitfast、W46 原 contact/passive callbacks、原 geom/pair solimp、仅原 W46 的 `jnt_solimp[:,0]=0`、warmstart disabled、原 solver/integrator/cone/jacobian/iterations/ls_iterations/tolerance/disableflags/enableflags。记录 model SHA、各物理数组 hash、有效 host/device 选项值、容量 `nconmax=128/njmax=768` 及实际解释。`wm.opt.tolerance` 保持数组，用 `.fill_` 写原 tolerance 并回读，不能重犯 W41 的 float/array ABI 错误。
8. 固定环境 package/source SHA、MuJoCo/Warp/MJWarp/CUDA/GPU driver/device、callback SHA、CPU affinity 和 stream/graph 模式；不升级软件、改 solver、调 tolerance 或扩 contact 容量追求通过。先核原有容量够用，发现 overflow 就技术失败并保留。

完整 snapshot 的对象是**当前诊断所采用的合法 cold-start Data 与所有会先读的持久工作区**，不是承诺跨 native3.2.3、CPU4、GPU4 的内部 bytes 可以相同。后两者模型数据、精度和实现必须分别标身份。

## 三、同一个 held，禁止首 hold 内反馈

从原件取 `gpu_actions[0,18]` 和 `[0,21]`，先核两者数值及投影身份；生成唯一 `u_hold_fp32`，把同一份 bytes 写入两 world。记录原 FP64 held、FP32 写入值和设备回读值。

若旧动作两份不同，先保存差异并选择有完整 source/projection 身份的一份作为明确声明的**共同开环输入**；不得一 world 用一份动作。这个差异触发下述零积分动作批次，仍不妨碍共同输入的物理辨识。

整个 10 子步内 ctrl 不变，不再调用反馈、插值、H-QP、planner、gait/reference 更新，也不读新状态生成控制。userdata/material proxy 保持 W46 诊断原语义冻结，callbacks 继续真实执行。

## 四、60 world-step 的三个条件

始终同一模型、两个相同 world、同一 held、同一 kernel/stream/dispatch 模式、同一数据布局与容量。所有写入/恢复/读取边界显式 synchronize。正常子步本就按 .2 ms 串行推进，读回不会增加物理步。

| 条件 | 起点 | 新 GPU world-step | 作用 |
|---|---|---:|---|
| F1 完整起点 | 完整合法 canonical snapshot `S0` | 2×10=20 | 新鲜初始化下的双 world 首 hold |
| L 旧式部分恢复 | 先完整恢复 F1 **第 1 个子步后的** snapshot `S1`，然后只执行 W46 `load()` 的那 5 类回写及 qacc_warmstart 清零 | 2×10=20 | 模拟“已真实预热一步再部分 reset”，其余持久值来自同一个已保存 S1 |
| F2 完整恢复重复 | 完整恢复同一 S0，恢复所有需先读的持久/graph 工作区，核 hash | 2×10=20 | 检验完整恢复本身可重复，排除仅凭单次 fresh 成功的误判 |

F1 第一个子步的 `S1` 就是 L 的预热输入，不再额外执行它。这样 L 不会偷用“已经跑了 10 步”的残留冒充 W46 的一次预热。三个条件独立从相应恢复点开始，不把条件之间的输出串成 6 ms 轨迹。

优先保持 W46 的单步 graph 调度，三个条件必须一致。先检查 graph 持久缓冲区是否纳入恢复。如果当前接口无法确认这一点，**在任何实验 physics 前统一选直接同步 `mjw.step`，全部三个条件都用该模式**，将结论明确限定为非外层 capture 的同 backend 首步诊断。不得跑完一个条件后随意换 dispatch；不得把非 capture 下的成功冒充历史 24-world graph 已修复。

若编译/graph 准备确需真实 step，只准独立 disposable 双 world 一步，**最多 2 额外 world-step**；随后正式对象必须完整重新初始化。统计 compile/capture record、真实 warmup、测试 launch 和失败前已执行的步数；record graph 本身不自动算已执行 physics。预算最多 62，不能另开不计数 warmup。尝试期间允许技术错误自然结束，不设置墙钟 kill 或无限重试。

### 每 .2 ms 保存的最小读回

- before/after qpos、qvel、act、time、ctrl、qacc_warmstart，canonical 对照及所有残留字段差异；姿态误差用相同 quaternion 切空间，单位分开；额外提供 bitwise/ULP 和首 differing index。
- qacc、qfrc_actuator、actuator_force、qfrc_passive、qfrc_constraint 和实际施加力；保存存在的字段并列清 unavailable 项。
- 接触的 world、geom pair、distance、normal/frame、friction/solref/solimp、约束地址与 force；既保存原始排列，也按稳定接触键比较集合，不能把仅排列改变直接认作不同接触集合。
- solver 类型、每 world 活动约束数/迭代数/可读取 residual、efc state/force/地址、容量/overflow。若接口不暴露 KKT，写 unavailable；不能把 H-QP 的 KKT 当物理接触 solver KKT。
- callbacks 前后材料/接触过滤的可用读回和 held 原始字节。具体 package 没有某字段时，使用实际字段清单和源码定位，不能根据期望编造 API 或数值。

在第一处非零差异处确定“第 k 子步、哪个量、哪个 world/geom/joint、该步 before 相同到哪一层”。仅 after-step contact/force 不自动证明谁先因果发生；需要已保存的 stage/read-write 顺序，才能继续把首差异归于碰撞排序、力装配或求解。

## 五、判定表：所有结果都产生新信息

| 新结果 | 允许结论 | 不允许结论 |
|---|---|---|
| F1/F2 内双 world 相同且两次全迹相同，L 与 F 不同 | 旧部分 reset 的残留确实改变该单输入的首 hold；按字段差异缩小初始化残留集合 | 所有旧 GPU 重复误差唯一由 reset 造成；native/GPU 已等价 |
| F1/F2 相同，L 也相同 | 这个 cold-start 输入和双 world 布局下，所测 reset 残留不足以改变首 hold；此假设受反证 | 历史 24 世界批、其他初态、图调度已确定性成立 |
| F1 内双 world 已不同，F2 复现同一分叉 | 在已验证输入/工作区范围内，出现可复现的 world/layout/接触计算差异，闭环控制不是首因 | 已精确证明纯 FP32 或已排除所有未暴露内部状态 |
| F1 与 F2 同一 world 不能重复 | 优先检查完整恢复覆盖、图持久缓冲区与实际运行非确定性；结果直接给出首 differ 字段/步 | 任意挑一遍成功当已修复 |
| 子步 0 回读就不同 | 初始化或复制链本身有具体错误，终止积分，保留差异修复项 | 把错误算为物理不稳定或继续扩大场景 |
| 前置全部满足，但 contact/residual 某项不暴露 | 仍完成 bounded 首 hold 的可读状态/力辨识，结论限制到已记录层；明确下一项最小 instrumentation | 因非必要辅助字段缺失就继续只读旧资料，或宣称已定位具体接触根因 |

F1/F2/L 均完成后停止。哪一种结果都不授权第二场景、24 世界重跑或 350 ms 扩窗。若希望继续分辨原 24 世界布局、版本或纯精度，另据这次明确首分叉提出下一实验；不在本轮顺手加矩阵。

## 六、条件触发的零 physics 动作批次

核心固定 held 已把动作执行移出物理首因，动作批次不应自动再做旧的“CPU Python 公式≈GPU公式”一遍。只在存在明确缺项时做：旧两份初始 held 不相同；或第一分叉共享状态上的**真实 native Policy::Action**与 GPU kernel 的 raw/clip/held 链仍未核过。

最多 **8 个唯一 state/time/policy 身份**，按实际可得去重，不补扰动凑数：共同 S0；首分叉子步两 world 的状态；L 的首个有新信息状态（最多两份）；同一 native 插值且归一化后的 reference 自身及精确 antipode；若仍有名额，只取已有完整事件中一个实际 active H-QP 输入。缺输入直接减少条目。自差与 antipode 必须使用 native 真正 reference，不能拿未归一化线性插值替代。

逐条保存实际执行 native build 与 GPU kernel SHA，政策生成 build（这些旧 policy 来自 W45/r34）另列；保存 time 的 host/传输/device 值、native reference、tangent、Kδ、raw center、ctrl-clipped center、FP64 held、FP32 ctrl、active/KKT/converged。native 对照须调用真 Policy::Action；若只能用旧 Python raw_policy，就标这个限制，不写“完整 native 等价”。H-QP 若仍由 host native 完成，继续明确是 mixed path。

这个批次 **0 physics/0 solve/0 FD**。如果接口不能提供真实 native Action 或必要 reference，则保留具体接口差异并结束，不能为动作批次启动 planner 或长窗。

## 七、失败降级和交付

初始化字段导入、solver身份或设备同值回读的必需条件不成立：physics 停在实际已执行数，不扩大预算；交付缺字段/首不等值/异常栈及可执行的最小修复项，并在可用接口下完成上述零积分动作批次。非必需诊断字段缺失不自动挡住能够回答问题的首 hold。

用现有环境，不临时安装/升级 engine 或强改参数；新入口固定 CPU host 总额度≤2、线程库1，按实际显存余量准入，不动其他作业。它可以与两条 CPU 主线并行，不能成为主线等待依赖。后台 deadline=null/wall_deadline_enabled=false，完成有限工作自然退出。

交付紧凑 summary、模型/软件/输入/字段/solver身份、三条件初始化差异表、每子步 first-divergence 表、实际 world-step 账本及少量 NPZ 原件。原件经 D57 增量发布核 size/digest，不新增用户电脑归档。结论始终保留 `equivalent_native_ranking=false`、`full_native_cost_implemented_on_gpu=false`、`original_task_qualified=false`。

## 依据

- `scripts/s1_go2_w46_gpu.py`：实际 model/create/action/load/warmup/capture/loop 路径，尤其第 141—178 行。
- `scripts/s1_go2_w46_native_bridge.cc`：native create/reset/project/hold；旧 reset 也是声明 cold-start，并未导入完整历史 plant 状态。
- `scripts/s1_go2_w41_warp_backend.py`：实际 GPU contact/material callback 与 tolerance array 修复；不能调用其修改 geom/pair solimp 的旧 Predictor 初始化。
- `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w48/task-c/duplicates.json`：六组重复身份、首 2 ms 分叉、原件 SHA、策略版本及缺项。
- `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W46-final-review/gpu-report.md`：过去 350 ms scope、warmup/capture 计数与 native 对照限制。
