# W41 GPU 完整交付复核与技术修复

证据基线：[c4b3f9a](https://github.com/kaiwen123-yang/MCP-RL/commit/c4b3f9adc698b15de5c94ef952e2884f3abfec62)，审查日期2026-11-10。只分析已交付结果和实现；本次没有连接D57，没有运行新GPU或Go2物理实验。原始结果保持原样。

## 1. 已完成什么，什么没有发生

**GPU主进程已经结束，G2的176节点数据完整，但G4没有进入物理评价，G5没有完成第一个声明fixture。不能把顶层completed字符串当作GPU闭环完成或方法成功。**

| 阶段 | 实际证据 | 结论 |
|---|---|---|
| G1 | 32个新事件；原阈值下5/32对齐，27/32不对齐 | 完成了模型/局部导数比较；不是等价迁移 |
| G2 | 176/176状态探针，49输入/探针，10子步；176份数组均有限且无缺失 | 完成一条保存horizon的状态探针求导数据；没有完整planner速度证据 |
| G3 | 写了native外环接口说明，G4的MJX请求数为0 | 未观察到真实外环调用，原completed_actual_outer_loop标签不成立 |
| G4 native | 子进程exit2，0.0695s，日志只有argparse错误 | 尚未开始物理，不能称E30跌倒或任务失败 |
| G4 MJX | 子进程exit2，约1.002s，predictor requests=0 | 尚未开始物理，不能称GPU闭环完成 |
| G5 MJWarp | 第一fixture首个mjw.step在opt_tolerance类型检查报错，stages=[] | 无完成fixture、无闭环或吞吐结论 |

来源：[GPU结果](../../gpu/w41-gpu/result.json)、[MJWarp结果](../../gpu/w41-gpu/mjwarp-result.json)、[完整原件交付](../../gpu/w41-gpu/delivery.json)。本审查的逐事件/节点统计见[gpu-audit.json](gpu-audit.json)。

G1的最大记录位置误差4.4400e-7（仍低于5e-7位置阈值），速度误差最大0.0022200（阈值1e-5），局部AB误差最大2439.63（阈值5e-3）。相同CPU前缀的native A/B与GPU A/B重新相减，保存AB_error完全一致；非四元数状态差也与保存数组一致。四元数位置误差使用既有原MuJoCo差分记录，本次没有借新版本重跑旧模型。小位置差不等于导数或接触分支对齐。

初次G2曾有MJX scan contact索引int32/int64不一致，现行fix1保持进入scan的整数dtype，并续算完176节点。它是已经修复并有实际完成证据的历史故障，不需要再重复G1/G2。

## 2. 两个直接运行错误及一个状态误标

### 2.1 G4少了必需的评价起始时间

`run_s1_go2_r12.py`单次评价要求`--attempt`和`--evaluation-start-unix`。bridge同时漏传native/MJX的后一项，两份原日志都以同一argparse错误退出。这里的start是评价身份/起点，不是墙钟截止；补它不重新引入4h/28h期限。G5之后的评价命令也有同一遗漏，修复必须覆盖这条尚未到达的路径。

已新增共享[evaluation_command](../../../../../../scripts/s1_go2_gpu_evaluation.py)，三个评价入口均从它生成参数。每个backend的E30/条件E60使用同一真实起始时间，deadline维持None。没有通过删去原评价验证来绕过错误。

### 2.2 G5把设备数组覆盖成Python浮点数

`mjw.put_model(m)`产生的`wm.opt.tolerance`是`wp.array[wp.float32]`。旧代码随后执行`wm.opt.tolerance=float(m.opt.tolerance)`，破坏求解器kernel要求的参数类型；原始trace明确停在`opt_tolerance expects an array ... passed value has type float`。

修复为`wm.opt.tolerance.fill_(float(m.opt.tolerance))`，保留设备数组身份和声明容差。不能简单删掉旧赋值：MuJoCo Warp4.0.0的put_model会默认把低于1e-6的容差提升到1e-6，直接删除会悄悄改变原1e-8契约。新记录同时保存native目标值和实际FP32设备表示。

官方依据：[MuJoCo Warp4.0.0 API](https://mujoco.readthedocs.io/en/4.0.0/mjwarp/api.html)、[4.0.0 io源码](https://mujoco.readthedocs.io/en/4.0.0/_modules/mujoco_warp/_src/io.html)，后者先执行容差下限处理，再把Option的array字段搬到设备。本轮保留原实验目标值，不由类型修复顺带改变求解精度。

### 2.3 子任务失败被无条件completed掩盖

旧bridge在两次CLI失败后仍写G3=`completed_actual_outer_loop`；G5包装器收集exit1却不改变总阶段完成判断；主程序最后无条件写`completed_declared_gpu_stages`。所以顶层完成、内部技术失败和发布成功同时出现。

已按实际子进程exit、结果stage_status、物理simulation_s和MJX requests分别汇总：仍有技术失败时总状态为`completed_with_technical_failures`；当前选择阶段发生技术失败则返回非零。仅续G5时，旧G4失败保留在总状态，但用`active_stages/active_technical_failures/active_invocation_status`单独表达新G5结果，避免把旧缺参误报为新G5失败。G3仅在有实际MJX请求和物理推进时记`physical_execution_observed`。恢复载入后即纠正旧G3标签，再移除旧completion，确保后续running快照也不传播假闭环标签或已结束状态。科学对齐、实际运行、成功评价分别保留字段。历史JSON不改写；对旧真实记录的重新分类得到3个技术失败、0个开始的G4物理评价。

## 3. 为什么“96GB GPU跑了很久”还没有加速

G2记录的176次`resident.request`墙钟合计**152237.102s=42.288小时**，中位899.395s，p95=1182.541s；首节点637.621s，去掉首节点后的175节点平均866.283s。这里是176个状态探针，最后一个复用action174；原horizon只有175个控制节点，不能把两个工作量混同。计时范围包含request内部输入转换、同步计算；后续完整输出取回和压缩保存还有额外开销。它不是一个完整MPC调用的总时间，也不是严格CPU/GPU加速比。

源码中的实际组织是：host逐节点for循环，每节点只让49个FD输入进入设备端10子步scan，完成并同步后再取回/保存，然后进入下一个节点。**176节点没有组成驻留节点批次。** G4接口最多合并两个请求，但requests=0，尚无实际候选合批证据。

Resident的jit对象只初始化一次；G2输入shape固定。没有代码证据表明每节点必然重复编译，也没有XLA编译事件或设备profile可以量化编译比例。因此不能把42小时归结为首次JIT，不能假定再次运行就会自动很快。

### 可定位的结构性负担

每个保存子步包含1007个候选contact槽、3045个约束槽，实际负contact distance只有3–4项，非零约束力21–24项。固定MJX3.2.3的elliptic Newton `_update_gradient`显式构建约束维的稠密`cm=diag(...)`，逐接触写入摩擦锥block，再计算`J.T @ cm @ J`。49路3045×3045 FP64矩阵的概念存储约3.385GiB；报告JAX peak_bytes_in_use约3.468GiB。原保存XML的cone为elliptic。这个源码/形状对应关系给出可检验的内存与算子负担，尚不能替代profile将所有时间归给该矩阵。

固定源码：[MJX3.2.3 solver.py](https://github.com/google-deepmind/mujoco/blob/3.2.3/mjx/mujoco/mjx/_src/solver.py)。同一源码`solve`返回时没有把ctx.solver_niter写回d；本轮所有保存solver_niter为0，因此**不能用0推断求解器没迭代**。后续计数需读取真实求解上下文或明确标为不可用。

记录中的约3.47GiB是本JAX allocator峰值，不是整卡总占用或当前空闲显存。现有约16GiB预算也只是该任务声明，不能将其等同已经执行的显存硬上限。真正有用的效率指标是同工作量的稳态延迟、完整规划耗时及实际样本吞吐。

## 4. 当前修复如何续接，避免重算已完成工作

[start_s1_go2_w41_gpu.py](../../../../../../scripts/start_s1_go2_w41_gpu.py)和主脚本新增显式`--resume-from G4/G5`，保存旧result/job/clock及发布身份，复用G1/G2，仍无deadline。G4需要已完成176节点。G5是独立后端任务，不构建/重编译MJX的G1/G2执行图。

- G4新增可选`--g4-backends native`或`native mjx`，避免只能自动启动极慢的完整MJX评价。新attempt带invocation，旧物理或失败记录不覆盖。
- G4/G5恢复使用原保存的评价contract；G4产生具名副本，G5继续引用该副本或原保存native合同。固定policy仍为W40-B update37、r30/native对相应独立GPU artifact的比较；不把当前W42根配置或新的CPU版本悄悄混进来。
- G5新输出在`gpu/w41-gpu/continuations/<invocation>/`，保留旧错误与数组；第一fixture错误也保存完整输入。首次计时明确标出含编译；已完成的G5比较可被之后G4续接复用。
- 发布器读取新G5报告中的真实attempt清单，恢复时只纳入当前invocation的物理attempt目录，继续上传新增原件；既有G4/G5原件复用旧交付资产，不重新上传旧G1/G2包，不另归档到用户电脑，不自动删服务器原件。

接口修复并不使慢MJX突然适合闭环。**不建议未经计算组织改造，直接补跑原有MJX完整E30。** 下一GPU任务应复用全部G2数组，针对实际瓶颈选择可有效合批的候选前向或更适合的MJWarp工作量，并保留当前CPU主线版本的明确边界。G5修复后的真实CUDA/原Go2首输入与后续闭环仍需在D57执行；本次没有提前宣布通过。

## 5. 本次实际验证与未验证项

[verify-gpu-fixes.py](verify-gpu-fixes.py)直接执行现有R12 CLI parser及进入evaluate之前的真实参数验证AST，在物理调用前结束；旧缺参重现exit2，新命令通过，deadline=None。它没有mock机器人或伪造评价成功。

隔离安装与服务器声明一致的MuJoCo4.0.0、mujoco-warp4.0.0、Warp1.19.0，在**CPU设备**的真实put_model结果上执行生产代码的fill_行。设备数组身份不变；put_model默认约9.999999975e-7，修复后约9.999999939e-9，等于原1e-8的FP32表示。验证不改变用户base/服务器环境。[验证结果](gpu-fix-validation.json)

没有验证CUDA kernel、完整原Go2/MJWarp step、G4/G5物理评价、GPU吞吐或新版本稳定性。代码语法与差异空白检查通过；是否解决后续尚未到达的模型适配问题必须由下一次真实输入和执行记录说明。

## 6. 原件覆盖与复算

完整下载两份已发布ZIP，分别29,337,276与95,663,446字节，SHA256与交付digest一致。复算读取32份GPU G1、32份对应native FD、176份G2及实际日志；逐成员字节数/CRC与结果在[gpu-audit.json](gpu-audit.json)。不存在把目录清单等同完整语义阅读的问题；全部新GPU与物理运行数为0。

复算命令形态：`python audit-gpu-evidence.py --raw-root <两份已发布ZIP所在目录>`；类型/入口检查使用隔离的上述固定版本环境运行`verify-gpu-fixes.py --contract <保留的native-evaluation-contract.json>`。这些命令分析与验证已交付接口，不重跑G1/G2或训练。
