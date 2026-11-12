# W45 结果、GPU 有限研究与旧 G5 最终原件复核

审查基准：`kaiwen123-yang/MCP-RL` 的 `3e3f69c440d6a33c14131563e5f485d88da3a803`（2026-11-12）。本审查只读取 Git/Release 与已保存数组，没有运行新物理回合、训练、服务器任务或 GPU 求解，也没有改仓库。

## 1. 结论

W45 的执行与交付已完成，其核心 alpha0 接纳修复确实生效，但原任务稳定性没有成立。两条固定 update57 流分别在 13.072 s 和 5.582 s 跌倒；之后的 E60/E300/E600 按协议 blocked，并非调度器漏跑。A 在本次评价真正加载了材料，但没有完成贯穿，B 没有加载；合法去程、返程和往返全部为零。新增 PPO 样本与更新均为零。

GPU 第三支线也实际完成了所声明的有限研究：4 个保存问题 ×32 槽 ×1750 子步 =224,000 timed world-step，另有1280编译预热 world-step；127 次新的 native 完整策略评分，加一次重复身份缓存。结果证明当前近似筛选不可靠，尚不能用于在线替代 native。它没有忠实执行 K/H 与 H 度量投影，评分目标也换成了参考轨迹跟踪 proxy；这不是同一完整控制器仅仅换到 GPU 上执行。

**本次还找到并修正一处证据入口错误：旧 G5 最终原件早已在 GitHub Release 发布，审查基点`3e3f69c`的旧 result/delivery 与 AGENTS 仍指向较早的技术中断。** 已直接下载并验证该最终包，原件确认 G5 完整运行30 s、耗时19.366 h，但机器人基本驻留，材料任务失败。本次已追加最终 continuation 审查入口并更新当前说明，旧失败尝试保留；不需要再上服务器补跑或重新收集这份已发布数据。

## 2. W45 到底推进了什么

| 项目 | A | B |
|---|---:|---:|
| 固定父策略 | W44-A update57 | W44-B update57 |
| 样本/更新 |14737/57|14737/57|
| 新样本/更新 |0/0|0/0|
| E30实际物理时长 |13.072 s|5.582 s|
| 结束原因 |body_fall|body_fall|
| runner exit code |0|0|
| 实际材料加载 |有|无|
| 合法去程/返程/往返 |0/0/0|0/0/0|
| E60/E300/E600 |均由E30失败阻断|均由E30失败阻断|
| 全部planner calls（含startup20次） |674|300|
| 新alpha0策略安装 |0|0|
| 正式planning调用 |654|280|
| 正式planning合计 |1600.428 s|785.524 s|
| 评价总wall |2027.735 s|1020.166 s|
| 正式planning p50 / p95 |2.056 /4.811 s|2.333 /5.133 s|

这些结果区分了三个层面：后台作业正常自然结束；r34成功消除所定位的 alpha0 新反馈安装；原物理任务仍然失败。不能再把这次失败归因为作业固定截止时间，因为实际合同已关闭 wall deadline，命令也没有传 `--deadline-unix`。同时，alpha0问题消失不等于所有正alpha/nominal策略的闭环行为安全。

W45 A的材料带体投影 entry为12.872 s，末尾13.072 s跌倒，包含真实材料加载但不足一次完整贯穿。此时A实际路线进度字段约0.1325，不能把它转成“已完成13.25%的研究”或完整任务成功率。任务资格要求的合法跨带和完整往返仍未发生。

主 CPU 计算瓶颈也有实际数据：正式planning约占A总wall的78.93%、B的77.00%。A PANN observer12.973 s、recovery query3.330 s（27次），两者合计仅占0.804%，其中首次编译8.724 s；B observer0.04344 s、query0.001110 s（12次）、world_cells0，两者合计0.00437%。当前不能把 PANN 小查询的优化当作 S1 主提速方案。同步离线仿真中的wall耗时不是自动造成跌倒的“控制延迟”；硬件实时能力需要另行验证，当前几秒一次planner也不能称50 Hz实时。

来源：`w45/phase.json`、`w45/w45-a.json`、`w45/w45-b.json`、两份正式 E30 result、`scripts/run_s1_go2_w45.py`。控制原件首偏离与因果分析由本轮另一审查负责，此处不冒充已定位全部失稳原因。

## 3. W45 GPU 的原件覆盖与独立复算

Release：`tm-s1-go2-local-readiness-r12-w45-gpu-part01`。

| 资产 | 实际字节 | 下载后SHA256 |
|---|---:|---|
|w45-gpu-originals-0001.zip|243945433|a4b15bde257170e61b2ea7886638e88a639794a99465c852aede1214b438a5ec|
|w45-gpu-originals-0002.zip|178117603|e589bb21499a74b094071612507621310260bda12737414e84a99a92eafb0494|

两个资产的size和远端digest均匹配。按Git的`sources.json`逐项读取并匹配301个原始源文件的size/SHA256，合计464,415,127字节。包含128个完整policy文件、127份native rollout数组、GPU states/actions/risk数组、排名、逐次native元数据、脚本日志和结果。A158 slot1与slot0完整身份重复，故没有独立的slot01 native数组，而是明确使用同一身份缓存。

独立复算包括：128个policy文件哈希；每个GPU输入动作与policy reference_actions转FP32逐元素一致；每个native结果的reference states/actions/times/residual、K、H、Qu、Qxu、active与完整输入策略逐元素一致（最大差0）；从原始101维residual和21项成本定义重算所有native J（最大误差4.44e−16）；从GPU states与声明的尺度重算全部proxy（逐元素完全匹配）；重新排序并计算recall、regret、基线改善和缓存计数。没有发现本次native完整策略绑定或成本抄录造假的证据。

复算脚本与机器可读结果：`recompute_w45_gpu.py`、`gpu-recomputed.json`。

## 4. 排序质量：为什么它目前帮不了原任务

| 保存问题 |GPU top4|native top4|slot recall|选中slot / 最优slot|选中J / 最优J|regret J|
|---|---|---|---:|---|---|---:|
|A158（W44 2.74s）|0,1,14,24|0,1,15,27|0.5|0 /0|.150773603 /.150773603|0|
|A163（W44 2.84s）|1,3,16,14|27,9,29,21|0|3 /27|.133504799 /.116043648|.017461151|
|B153（W44 2.64s）|29,5,11,30|29,31,7,5|0.5|29 /29|.065050354 /.065050354|0|
|B173（W44 3.04s）|1,6,13,25|11,9,21,2|0|13 /11|.611886325 /.602510430|.009375895|

四问题平均slot recall为0.25，native最优只召回2/4。A158 top4把两个完全相同策略计为两槽；按完整身份去重后，其recall为0.25，四问题平均为0.1875。这是补充口径，不推翻原来明确声明的slot口径，也不能据此泛化成未知分布上的统计成功率。

A163当前选中相对最优基线只改善.00218950；同32槽的native最优可改善.01965066，GPU保住的改善仅11.14%。B173保住41.93%。因此“有3个问题更快”不等于“同质量提速”：两次较快确认牺牲了明显的候选质量。

**最直接的语义差异证据是两套在native里几乎同轨迹的完整策略，被GPU proxy排到两端：**

|问题|slot0/1 native J差|native全状态最大分量差|GPU slot0/1排名|GPU proxy0 /proxy1|
|---|---:|---:|---|---|
|A163|8.05e−15|4.15e−10|32 /1|1.883814 /.130060（约14.48倍）|
|B173|2.17e−12|4.44e−9|32 /1|2.285429 /.037714（约60.60倍）|

这里GPU实际做的是FP32 MuJoCo4中的参考动作推进、轴向限幅，并没有K/H闭环反馈与H度量QP；proxy又是归一化reference tracking，而非原native目标。不能把这个差异只归为精度误差或GPU模型“不够训练”。其输出与CPU完整策略的实际行为本来就不是同一个评估对象。

native成本分解进一步说明proxy丢失了什么：四个问题的`MaterialSupportHeight`占所选J的大部分，分别约.09453/.15077、.08063/.13350、.03410/.06505、.40718/.61189。A163漏掉最优的J差中，该项贡献.01105，Upright贡献.00583；B173该项贡献.00841，Upright贡献.00386，其他项有正负抵消。不能仅按“贴近一条旧reference”的程度替代原任务目标。这些是同一次原件成本分解，不是已证明MaterialSupportHeight定义本身有错误。

同动作的GPU world也有可测数值差异。A158 slot0/1完整policy哈希与GPU初态/动作均相同，第一2ms末状态只出现约1e−7级速度差，但后续接触时最大关节速度分量差可达.72447 rad/s，关节位置最大差.002413 rad；末端关节速度差已回落至.00546 rad/s。B153两个GPU动作相同的baseline也有类似现象。它提示接触切换可能放大FP32并行数值扰动；本审查没有在服务器重演，不能凭这一条就断言是某个GPU kernel bug。该差异远小于缺K/H和替换目标所造成的语义偏离，且不能成为把GPU当成native等价的依据。

## 5. 约束、计时及实现偏差

### 5.1 约束结论的真实边界

128槽的native `trunk_violation`全部0，failure全部false，metric_valid全部true，所以false-feasible=0、regret_v=0。这是没有正例的样本，不是风险检测灵敏度100%。这些早期地面保存问题不覆盖完整材料历史、非足碰撞后的失败状态或合法往返。

GPU原件的`hold_torque_excess`有A158 31槽、A163 31槽、B153 0槽、B173 32槽大于0，峰值约3.2992、2.6344、0、10.0025 Nm。但脚本在积分后用“仍保持的目标”和新状态计算raw PD request，且执行器有独立限幅；该字段不是实际施加力矩越界或原native完整hold约束违反。当前Git后加的torque语义说明是必要的，不能把这个数量混入false-feasible分母。

### 5.2 端到端时间按实际方案复核

|问题|GPU device推进s|GPU确认前全部s|native top4+必要baseline确认s（含native启动）|报告端到端s|native32全部s|
|---|---:|---:|---:|---:|---:|
|A158|2.2793|8.7838|4.0572|13.7673|10.3957|
|A163|1.7345|2.6003|3.7011|7.1643|9.7055|
|B153|2.4068|3.1514|4.9511|8.9205|11.7288|
|B173|1.5830|2.0940|3.5083|6.4291|9.7259|

表中端到端额外计入native子进程启动/收尾开销，不等于前三列简单相加。四批GPU device总8.0036s，确认前合计16.6295s；4次端到端确认合计36.2812s，native32合计41.5559s。包括全部离线native真值的研究wall为58.1873s，GPU job从开始到结束58.2306s。它与双CPU任务同时启动，但属于约一分钟就完成的有限工作，不应为了显卡占用而循环续跑。

实际native确认顺序确实是GPU top4→缺失的baseline→余下槽，未偷用以前已知的native真值。127次全新native评分/1次缓存与原件一致；GPU也只末尾取回完整状态，不再是旧G5每个节点往返host。

不过，`native32_total_s`是本实现中同样2CPU额度、顺序调用单个native agent并保存压缩NPZ的全部时间；没有证明它是优化后的最佳2CPU批量实现。报告的端到端确认时间用已记录的前缀时间加启动/收尾估计构成，外层进程实际仍等待全部32槽结束后才汇总。它支持该研究的有限耗时比较，不能作为线上服务p99延迟或实时资格。仅GPU推进已远高于20ms周期。

### 5.3 一个未在这4点触发的选择逻辑缺口

脚本确认了top4与必要baseline，但随后 `chosen=min(top4, key=native)`，没有把已确认baseline加入最终选择集合。当前四问题里最终选中都不差于baseline，因而没有污染上述regret结果；对于其他问题却可能在已知道baseline更优时仍选较差top4。下一实现应在确认集合中保留完整incumbent，并用原(v,J)/完整策略身份规则裁决。这是一个明确实现边界，不是这两次物理跌倒的已证原因。

## 6. 旧 G5 的最终原件已经在 GitHub

直接检查Release `tm-s1-go2-w41-gpu-evidence`发现：

- 资产：`w41-g5-20261110T033858Z-evidence-01.zip`。
- URL：https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-w41-gpu-evidence/w41-g5-20261110T033858Z-evidence-01.zip
- 实际size：354,912,832 bytes。
- SHA256：`0444b00fa33ec512da847252a3fea078898d5e47c650467368ad1f7fb730ea62`，匹配远端digest。
- Release资产时间：2026-11-10T23:06:03Z。
- ZIP112项：111完整源文件+1内置manifest。所有111项完整读取，ZIP CRC与manifest size核对通过；manifest没有逐文件SHA，因此对全部源独立计算并保存SHA，另外验证正式result所载trajectory/physical/prediction三份文件的size/SHA声明。

原件最终结果：`completed_independent_MJWarp_comparison`，`first_error=null`，评价exit0；独立原E30满30s，69717.867s/19.366h，`task_criteria_unmet`，`material_traversal_not_completed`。冻结的是W40-B checkpoint009617/update37、r30/新FP32 MuJoCo4-Warp predictor，实际CPU Go2/TM02 plant未替换，与W45 r34/update57不是同一次控制器比较。

从trajectory重算：15001个500Hz记录，最大body x仅.0000242083m，最终xyz约[-.0038003,.0002505,.2869639]m，最低body z=.2869517m；material world_cells=0、无实际材料加载、合法去/返/往返0。150000条0.2ms physical记录与1502个prediction origins齐备。这是真正的“完整时长但没有原任务进度”，不能称稳定通过E30。

planner原件共1521调用=20startup+1501正式规划，全部返回policy_installed；正式planning合计68577.060s，p50=48.645s、p95=81.444s，startup另988.537s。GPU记录请求5,589,675次，除以1521调用恰为3675请求/调用的总量比。旧桥接架构的极大量host交互、49个world与单节点推进已被W45 resident batch思路部分解决，但G5最终控制器仍没有运动进度。两代controller/精度/预算不同，不据此做公平速度提升倍数结论。

六个保存fixture各三FD尺度的18份AB原件也已复算：均有限，矩阵shape10×36×48，邻接尺度相对AB变化约0.1169–1.4600，和报告仅有FP64归约末位差。这是强FD尺度敏感性的探索性证据，不是native等价验证。

**本次修正了索引入口。** 审查基点`3e3f69c`的旧`gpu/w41-gpu/result.json`记录Python float替换device tolerance array的早期错误，旧delivery仅索引G2；该历史尝试继续保留。本次已追加`continuation-final-audit.json`指向已发布的G5最终bundle，AGENTS/当前计划已改为“最终原件已核验，G5任务负结果，禁止重跑”，消除后续代理因陈旧入口重复查找/收集/启动的机会。

本次机器可读证据：`g5-final-audit.json`、`g5-source-hashes.json`，复核脚本`audit_g5_final.py`。不存在“必须再上服务器才能补齐G5最终结果”的阻塞。

## 7. 下一轮并行建议

继续使用两个独立CPU额度推进真正的控制稳定性，固定各自update57，以便把下一项控制机制与学习收益分开。默认A10 workers、B10 workers、线程库各1；第三任务最多2个host CPU，约3CPU留主控/观察/发布。现场实际affinity、cgroup、其他进程、RAM、GPU占用仍须读取；历史25核/96GB不是当时可独占资源证明。两“CPU流”的Actor/PANN也可能在GPU，不能把整卡宣称为第三任务独占。

**第三任务不再重复本轮reference-proxy top4排序，不重启G5。** 当前最有价值的GPU方向是一个与W45首偏离直接相关、可以自然完成的完整策略执行一致性诊断：

1. 当前主审查已将四个W45来源具体化为A call148/plan136@2.54s、A call153/plan140@2.64s、B call155/plan141@2.68s、B call158/plan144@2.74s；GPU可用其中2–4个。它们与下一主机制的r34/恒定input-reference penalty/按接触承载调度penalty有限比较共用来源，但GPU自身不扩为这些变体的额外网格。W44 A158/A163本就晚于其约2.6646s首次全足离地，不能继续把旧4点当作当前首因果窗口。
2. 每问题仅取完整incumbent与原实际选中策略，绑定reference/K/H/feedback scaling/状态/context。同样的初态分别用名义、一个来自当前已保存实际误差的切空间偏差及其反向，共每问题6个scenario；2–4问题总12–24个scenario，不扩幅值、worker、horizon网格。这是离线诊断，不将未来actual状态用于线上决策。
3. 首先只在已经记录的相同states上比较完整action map：reference按同一时钟插值、MuJoCo切空间状态差、K乘积与实际scale、同H度量投影、bounds、QP活动面与KKT、最终序列化/held动作。该步不推进任何physics。随后才配对20ms推进：每2ms一次完整action query，各保持10个.2ms步；比较动作、H投影、保持期实际力矩、接触和状态。若GPU仍只实现open-loop动作/轴clip，直接报告语义缺口；不要把它重命名为“完整闭环”继续扩跑。
4. 只有完整执行语义已实现且差异能被明确解释，才做同一批12–24scenario的350ms闭环推进，并用原成本/风险定义复核；最多21,000–42,000 world-step。没有native完整成本实现就不能发布与native J等价的GPU排序结论。若跨引擎差异主导且当前可用接口无法局部解决，交付有限负结果并结束第三支线，不把本轮扩成新仿真引擎项目。
5. GPU测量单独列编译、初始化、拷贝、设备推进、CPU确认、共卡竞争与端到端；是否继续由与主失稳问题的关联、同策略差异和实际吞吐决定。它不阻塞双CPU原任务验证，也不以占满96GB为验收条件。

### 7.1 可实施边界和风险

同states上的action map可独立用明确FP64数组实现，从而先隔离线性代数/四元数/投影是否正确。H-QP必须实现当前native规则，包含空界/固定轴、恢复分支、正则、KKT与序列化，不能用一个通用axis clip或未验证的库求解器代替。不要只比较输出动作：病态H附近看起来很近的目标可能对应不同活动面和下一步反馈；同时保存绝对动作偏差、映射到PD的力矩偏差及native已有残差定义。切空间姿态用MuJoCo的wxyz约定与同样的差分/积分，不能直接加四元数或套Warp默认xyzw约定。

进入dynamics后必须把FP32边界写清楚。当前实际W45实现与官方MJWarp说明均是FP32，而native默认mjtNum是FP64；不能承诺一个dtype开关就得到完全等价的FP64 MJWarp。可以在GPU以FP64计算action map，然后明确记录写入FP32仿真数组的cast位置，但这仍是混合精度近似动力学。它必须与相同积分器、相同原接触参数、相同冷启动/容差/材料proxy身份的native predictor配对；plant Euler与planner implicitfast的差异另列，不能一边悄悄换integrator、一边把全部偏差称GPU误差。不得沿旧G5把geom/pair solimp改为0而继续声称同模型。

官方说明还明确GPU原子操作可能带来微小重复差异；本次A158重复world原件提供了实际例子。因此阶段1应先排除策略语义错误，阶段2报告首分歧时刻与分量，不要求跨不同引擎bitwise完全相同，也不能把native与GPU差异笼统塞进任意大容差。精度/活动面/接触分歧若足以改变原约束判断，第三任务到此交付证据，不进入350ms排名或在线替换。当前所选四个窗口都属于原任务的早期状态，结果不扩张为材料历史传播、完整路线或机器人硬件资格。

以上精度、四元数和非确定性边界来自[MuJoCo4.0 MJWarp官方说明](https://mujoco.readthedocs.io/en/4.0.0/mjwarp/index.html)及[当前FAQ](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html#frequently-asked-questions)。实际执行仍以服务器锁定版本及数组dtype读回为准，不在本轮升级依赖或更换物理引擎来追求parity。

这比继续扩大32槽reference-proxy集合更能回答当前问题，但它仍是建议的独立研究任务，不能预先声称一定能修复控制。若下一轮主机制不需要GPU证明，有限诊断完成后双任务运行就是合理资源配置。

所有后台无固定wall完成时间：有限工作自然完成、原任务判据结束、明确技术失败或用户停止即停。物理30/60/300/600s、有限求解器预算与网络单次超时仍保留。GitHub推送并核验后不在用户电脑做归档，服务器原件不自动删除。当前发现的G5入口问题通过追加审查索引解决，不能为了补文件重跑19小时实验。

## 8. 可复核来源入口

- [W45 phase](https://github.com/kaiwen123-yang/MCP-RL/blob/3e3f69c440d6a33c14131563e5f485d88da3a803/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w45/phase.json)
- [W45 GPU compact result](https://github.com/kaiwen123-yang/MCP-RL/blob/3e3f69c440d6a33c14131563e5f485d88da3a803/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w45/gpu.json)
- [GPU执行实现](https://github.com/kaiwen123-yang/MCP-RL/blob/3e3f69c440d6a33c14131563e5f485d88da3a803/scripts/s1_go2_w45_gpu.py)
- [native确认实现](https://github.com/kaiwen123-yang/MCP-RL/blob/3e3f69c440d6a33c14131563e5f485d88da3a803/scripts/s1_go2_w45_native_batch.py)
- [W45 GPU Release](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-local-readiness-r12-w45-gpu-part01)
- [G5最终包所在Release](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-w41-gpu-evidence)
- [W45原计划（含实施与最终状态）](https://github.com/kaiwen123-yang/MCP-RL/blob/3e3f69c440d6a33c14131563e5f485d88da3a803/docs/S1_W45_STABILITY_PLAN.md)
