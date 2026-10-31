# S1：支撑联调与tmux训练、自动渲染、自动发布

## 当前交接（按实际交接只更新这6行）

- 当前任务：W34后台作业job-20261030T161045Z已结束训练与自动后处理；训练达标=True，发布=publication_pending。
- 当前状态：本组实际累计1280条/5次更新，目标1280/5；训练记录不代表正式E30资格。
- 已完成：本job保留30个新attempt，视频可用30，资产已发布456/1010；控制版S5-r24。
- 当前问题：最后物理原因training_checkpoint_complete；待传554项，待处理视频0项，具体错误见本job摘要。
- 下一步：先查看本job真实路线/支撑结果；有发布待办则从现有文件续传，禁止为补视频重跑物理；再决定继续学习或正式阶梯。
- 续接位置：autodl-d57:/root/autodl-tmp/MCP-RL；摘要results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/jobs/job-20261030T161045Z.json；原Release，用户电脑不另归档。

## 1. 实际进展与结论

最新审查固定在[f8bfb462](https://github.com/kaiwen123-yang/MCP-RL/commit/f8bfb4629a1d342f49a63a446ebd2a19151075d8)，继续使用tm-s1-go2-local-readiness-r12，不因用户轮次名称或主机变化另建run清零。主证据为[PANN首次更新结果](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/pann-first-update.json)、原episode结果、transitions、实际rollout和[learning日志](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/records/training__PANN_PPO_S5-seed0__learning.jsonl)。

**PANN＋PPO＋MPC的首次真实学习工程目标已经完成，下一步应在完整闭环中联调支撑并连续学习。** 冻结PANN在线观察与恢复特征进入Actor，PPO实际更新Actor/Critic；不代表PANN/MPC全部端到端更新、材料任务成功或算法收益成立。

| 项目 | RAW历史 | PANN本轮 |
|---|---:|---:|
| 真实转移/有效平移 | 256/205 | 256/208 |
| PPO更新 | 1 | 1 |
| 实际物理会话 | 4 | 12 |
| 仿真秒 | 127.6484 | 125.0854 |
| 有效尾buffer | 0 | 0 |
| 原始loss | 缺失，保持原记录 | 已保存 |
| 首次更新后的物理行为 | 尚无 | 尚无 |
| 完整材料贯穿/往返/正式E30 | 0 | 0 |

共512条来自两个独立流，不是同一策略训练512条。PANN256条均dt>0、编号连续；11真实terminal和末条目标truncation正确，episode内next_obs与下一obs衔接，已检查rollout数值有限。48条站立/转向等其他真实样本保留，不删除重采。一次更新有16个optimizer steps，loss10.44815、value loss20.90845、approxKL.00100657、clip fraction.015625；发布摘要记录Actor/Critic参数改变。一次loss不能判收敛，未另做checkpoint认证或重跑更新。

77/256条输入有非零12维恢复通道；109条有软地查询，其中32条恢复值为0。最大恢复约75.8μm，归一化后.0758。零信号可能对应刚地/未加载/无恢复，不能要求人为制造非零。信号进入输入不等于预测准确、策略利用或任务收益；PANN仍未进入每个MPC候选的完整材料动力学。

## 2. 失败集中在哪里

| 会话 | 仿真秒 | 样本/有效 | 最大x/m | 结局 |
|---|---:|---:|---:|---|
| 01 | 14.8614 | 30/26 | 1.2071 | 机身非足端接触 |
| 02 | 8.9672 | 18/14 | 1.3766 | 机身非足端接触 |
| 03 | 8.4678 | 17/13 | 1.2195 | 机身非足端接触 |
| 04 | 7.6034 | 16/12 | 1.0005 | 机身非足端接触 |
| 05 | 12.0446 | 25/21 | 1.3815 | 材料域限，最大应变.0201540 |
| 06 | 19.2942 | 39/35 | 1.2615 | 材料域限，最大应变.0202249 |
| 07 | 9.0826 | 19/15 | 1.3522 | 机身非足端接触 |
| 08 | 12.9206 | 26/22 | 2.1661 | 机身非足端接触 |
| 09 | 10.0426 | 21/17 | 1.4413 | 机身非足端接触 |
| 10 | 6.5610 | 14/10 | .8207 | 机身非足端接触 |
| 11 | 11.7400 | 24/20 | 1.5115 | 机身跌倒 |
| 12 | 3.5000 | 7/3 | .1106 | 达256条后正常目标截断 |

这是11次自然物理失败＋1次目标截断；另有一次缺libEGL.so.1的零步启动失败已修复，渲染缺imageio也已修复，没有为渲染重跑物理。**12是达到256条后的实际会话数，不是12次实验上限。**

所有PANN会话的侧出标记均为空，所有自然失败都没有到3m远端/返程。当前主要问题应更新为去程材料进入与穿越时的支撑、冲击和姿态保持；旧RAW侧出与返程风险保留为历史，不能套作本轮原因。ep08距条带东界x2.2仅3.39cm，接近的是条带出口，不是完整往返。ep06运行最久却只到x1.2615，不能只看存活秒数。

[ep10结果](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/train-PANN_PPO_S5-s0-ep00010.json)及其已发布trajectory/XML可直接定位症状：geom0是floor，geom13是trunk前下部球形碰撞体。6.45s前右足采样法向力约423.4N；6.38→6.56s pitch从.00085升到.568rad、pitch角速度升到6.45rad/s，随后机身前下部触地。6.504s起到末尾500Hz样点四足接触均0，不能据此断言所有中间物理步连续失撑。末行qpos/qvel与诊断时间不同是已有记录规则，不是新日志错误。

ep04/05也有约−.96/−1.69m/s的落足速度峰值。r21已做因果卸载面、承载深度和向下参考修复，不能把它当作完全刚地或没有支撑参考。本次进一步读取既有预测数组后，已经定位到可优先纠正的目标错配；不据此同时扫描duty、提高力矩或放宽.02阈值。

### 2.1 新定位：预测允许躯干触地后恢复，执行却立即终止

使用同episode的[原预测数组](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12/train-PANN_PPO_S5-s0-ep00010__prediction-000.npz)、trajectory与XML：330条计划，每条0.35s/176节点；所检查计划首状态与对应真实qpos一致。躯干前下部球体局部位置[.293,0,−.06]m、半径.047m，接触margin为.001m。按保存的姿态旋转该局部位置并减去半径，得到相对地面的最低点。

| 计划起点/s | 所选预测最低球底/mm | 首次进入1mm接触margin/s |
|---|---:|---:|
| 6.20 | −7.476 | 6.514 |
| 6.30 | −8.620 | 6.522 |
| 6.54 | −7.465 | 6.562 |

6.20→6.514的0.314s处在原0.35s horizon内。实际6.561s终止时球底约+.363mm，已进入1mm margin；与geom0↔13接触一致。末段6.54→6.56s的预测pitch约.567rad，实际约.568rad。因此不是MPC完全没预见前俯；至少这些选中轨迹允许了真实任务禁止的接触，后续又预测恢复。

检查已发布cost与r21继承链，已有Upright/Height/Balance、MaterialSupportHeight、LandingVelocity/Indentation、MaterialDomainMargin等，未找到直接对应非足触地终止的躯干clearance/contact项。**下一窗口先核对服务器实际冻结源码；若缺失则加入相关可优化残差，若已有则修其激活/参考/尺度。** 使用原几何与接触margin，按现有cost分解定尺度，不预先拍权重，不改真实失败定义。这是当前明确的联调入口，不是“一项补丁保证解决所有失撑”的结论。

[r21配置](../experiments/terrain_memory/s1-go2-r11-S5-r21.json)与[补丁](../experiments/terrain_memory/s1-go2-r11-S5-r21.patch)已有因果卸载面、承载深度和向下参考修正；实际继承r20→r16，不包含另一分支r17/r18的全部修改。行走期support_active=False也不能解读为所有动态支撑残差关闭。

### 2.2 “完整算法联合”具体指什么

现有闭环已经包含：真实足端观测→因果PANN材料状态/恢复查询→12维恢复特征→PPO每0.5s输出cap→MPC关节目标→原Go2/材料执行→下一观测。RAW是基线，不是完整算法需要拼入的第四个模块。

当前继续冻结PANN权重、训练PPO Actor/Critic、保留MPC在线优化，是有意保持可解释的联合执行配置。PANN输出尚未进入每个MPC候选的完整材料状态及导数，不能称端到端联合优化。此后若需要扩大模型耦合或动作权限，应由同版多次更新的真实失效证据决定；现在不用重新接一次已接好的链路。

## 3. 计算与比较边界

| 环节 | 秒 | 占采集/更新8092.13s |
|---|---:|---:|
| 运行中MPC规划 | 5734.05 | 70.86% |
| 初始化规划 | 228.57 | 2.82% |
| 物理循环 | 1096.43 | 13.55% |
| PANN观察器 | 605.67 | 7.48% |
| PANN恢复查询 | 74.72 | .92% |
| 反馈与状态/参数RPC | 152.08 | 1.88% |

材料参考764.21s已含在物理循环中，不能再相加；首次编译8.97s是共享law重复记录的字段，不能乘12。PANN观察与查询合计8.41%，全部理想消除也只约1.09倍总加速，先改GPU scan不是当前优先事项。规划平均约.916s/次，相对20ms逻辑规划周期仍属离线慢仿真，未证明实时性。采集吞吐113.89条/h、有效92.53条/h；新256条参考成本约2.25h，后续轨迹变化会改变成本，不能承诺时长。

D57与旧RAW的r21已有二进制记录、MuJoCo3.2.3、主要数值/代价/物理设置一致；主机、worker5→20、软件环境、PANN输入与实际轨迹不同，现有数据不是匹配收益或加速对照。旧RAW与新PANN首会话前5个cap几乎一致，2.5–3s路线增量已经分离，而本轮首会话8.5s才有非零恢复输入。因此不能把更早失败归因于PANN特征或学习恶化；不为此增加确定性认证任务。D57可见CPU affinity范围不等于25vCPU配额，不按可见208个逻辑CPU增开worker。

## 4. 当前执行：短交互启动，tmux完成训练与自动发布

用户已明确授权在AutoDL用tmux运行十小时甚至更久，启动后确认真实采样正常即可结束对话，稍后查看日志。本节覆盖旧的“后台也必须4h内停止”和“下次对话才启动发布”规则。用户已进一步明确授权自动渲染和自动推送；支撑修正与连续多次更新的研究目标不变。

### 4.1 两个时间边界

| 对象 | 时间与停止方式 |
|---|---|
| Codex交互 | 从首次活动起最多4h；完成必要修改、启动并交接后可以提前结束 |
| tmux训练作业 | 使用独立记录的job start/deadline；默认允许16h，可用--hours明确设置；达到样本目标提前结束 |
| 普通物理失败 | 记录真实terminal/checkpoint，继续下一自然会话 |
| 技术中断 | 停止并保留实际日志，不加无限重启循环 |
| 自动后处理 | 从训练退出起另计默认4h，渲染保存轨迹、上传Release、推送结果；不使用已经耗尽的训练deadline |

当前1024新样本按已测吞吐参考约8.99h，16h是留有余量的作业预算，不是要求运行满16h或新的训练总额度。窗口结束不再重置环境、切断半批采样或要求重新计划。实例仍需保持运行；tmux支持客户端断开，不保留关机/重启后的进程。

### 4.2 本次交互直接做什么

先查看是否已有同一tmux作业。若正在运行，读取现有进度并交接，不重复启动、不覆盖其deadline。

若尚未启动，使用已有ep10与实际冻结r21源码处理已定位的非足接触目标错配；只做一项有依据的修正，保持原几何、margin、材料/力矩/失败定义。已完成的修正不重做，也不先跑重复基线。确认并编译后记录reuse.active_controller_build。

前台先提交实际源码/配置，准备服务器持久gh登录、Git作者身份和可独立运行的academic/clash/direct网络。启动器会读取目标Release并确认仓库写权限，使用gh凭据助手；不输出token、不写入job.json，不做凭据测试矩阵。仅有电脑端临时token或PC隧道时，先完成服务器实际发布准备再交接。

随后直接在完整PANN观察、PPO决策、MPC规划与Go2/材料反馈链路中训练。现有训练本身提供实验数据，不新增E30准入测试。

**交接只看实际启动现象：**物理仿真和样本计数继续增长、正常日志正在写入、没有即时技术错误。无需等到256条/一次更新，也不要求额外一轮“稳定性测试”。若观察中自然经历一次terminal并自动进入下一个episode，这是已有运行证据，不是新的必过门槛。

交互可以立即结束，留下session、job目录、目标、预计查看时间、实际deadline、当前计数和checkpoint位置。后台继续；后续不安排逐步盯守、截图或常驻轮询agent。

### 4.3 阶段目标与运行命令

| 项目 | 当前安排 |
|---|---|
| 起点 | PANN自身最新真实checkpoint；已知ep00012为256/1、空tail |
| 新阶段 | 同一记录控制版新增1024真实转移/4次PPO更新 |
| 当前累计目标 | 1280/5；旧r21数据保留，不能算成修正版本新增 |
| 采样/更新 | rollout256不变，普通更新后继续完整闭环 |
| 资源与算法 | 单流20MPC workers、Torch2、OMP/BLAS1；冻结因果PANN、原2Hz cap/奖励/归一化/物理设置 |
| 版本 | 换控制器时记录父checkpoint与空buffer边界；后台运行中保持配置和执行器不变 |
| 后续判断 | 看真实支撑/路线/失败变化，单seed小样本不称算法收益；正式评价仍逐级 |

启动入口如下，--resume换成实际最新本组checkpoint：

```bash
cd /root/autodl-tmp/MCP-RL
source scripts/autodl_d57_env.sh
python scripts/start_s1_go2_r12_tmux.py \
  --resume outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/PANN_PPO_S5-seed0/resume-state-ep00012.pt \
  --session s1-joint --hours 16 --postprocess-hours 4 --network-mode academic
```

启动器在原outputs根目录下生成带时间的job目录，保存job.json、run-job.sh、train.log，以及训练结束后的training-exit-code.txt/training-finished-at.txt；正常训练仍使用原stream输出位置。训练输出关闭后自动调用finish_s1_go2_r12_job.py，训练非零退出也会发布已有真实结果。它调用现有训练入口的--background-job和--job-start-unix，使同一个job deadline传给原runtime与PolicyMaterial。前台模式仍保留4h限制。

默认session名固定为s1-joint，整个训练和后处理期间都占用该session；已有同名session时tmux拒绝再次启动；先查看，不要换个名字绕开并发限制。不要额外套4h timeout，也不要在退出对话时kill这个用户已授权的作业。

### 4.4 日志已经足够，不新增监控框架

| 已有记录 | 何时更新 |
|---|---|
| transitions.jsonl、collection.json | 每条真实转移 |
| learning.jsonl、checkpoint-*.pt | 每次256条的PPO更新 |
| episode-*.json、resume-state-ep*.pt、progress.json | 每个自然物理会话结束/正常存盘 |
| stopped.json | 训练正常收尾，含实际计数、目标、尾buffer和最后stage/reason |
| 本次job的train.log | 标准输出与错误，使用现有无缓冲Python |

回来后看tmux是否仍在运行，再读collection/learning；已经退出则看stopped和最后episode，并确认stopped的invocation_started_unix_s属于本次job（不早于job.json的started_unix_s）；新作业运行时可能仍留有上一作业的stopped。exit0本身不证明目标完成，因为现有runtime可把技术中断写成结果后正常退出。原collection.wall_s可能含历史停机；本次吞吐仍以invocation新增计数/耗时为准。

### 4.5 自动收尾的实际步骤

1. 读取本job的新attempt列表，逐个用保存的trajectory/model/result调用现有渲染脚本。已有完整视频直接复用；零步或不完整轨迹说明原因，保留其余证据。渲染失败不重新运行物理，半截视频不作为正式视频上传。
2. 给本job变化的训练小记录保存固定版本，使用job前缀；新rollout/checkpoint仍用原服务器路径、按job命名。旧未变的大文件引用原发布，不重复归档。
3. 生成一份明确的本job资产清单，使用服务器gh认证调用现有publisher。源码及小结果进Git，全部实际新视频、原始轨迹、物理/预测/材料数组、rollout/checkpoint、job记录进原Release。
4. 只做一次名称/大小/上传状态核对。渲染和上传日志写紧凑记录，闭合后再进入Git；不上传持续变化的自身发布日志，不做全量hash/回下载复验。
5. 写jobs/<job名>.json、正常索引和顶部6行交接，记录训练实际计数/最后原因、视频链接、资产发布状态和pending。按本任务路径提交并普通push到main，不force、不顺手提交其他工作。
6. 渲染、上传或推送出现实际错误时，保留具体错误和pending，继续发布能完成的结果。修复后对同一个job运行finish入口即可续接，复用视频、快照和已上传资产，不重训。

结果入口为[现有R12结果目录](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/)下jobs/<job名>.json；实际作业结束并推送后可在GitHub直接查看摘要和原Release的视频/数据链接。训练达标与发布完成分别报告，失败实验也保留。Git推送失败时本地保留提交和pending，不能声称远端已更新。

例如只接续已有job的后处理（使用该job真实的训练退出码）：

```bash
source scripts/autodl_d57_env.sh
source scripts/autodl_network.sh academic
python scripts/finish_s1_go2_r12_job.py \
  --job-dir outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/<job名> \
  --training-exit-code <该job训练退出码> --hours 4
```

如果旧版tmux训练已经在运行，不重启它；先读取其job/退出记录，为该既有任务补接一次训练结束后的finish步骤。新启动的任务直接执行整条pipeline。后台运行中不修改该任务的控制器或同时启动第二条采样流。

### 4.6 结果到达后再做研究判断

用户回来直接查看已发布的job摘要、视频、路线与支撑记录。阶段终点的最后一次更新若没有后续物理执行，仍不能称已经评价该次更新的效果。依据实际多次更新趋势决定继续学习或同版E30→E60→E300→E600；存在发布pending时先续其后处理。

## 5. 完整发布与交接

本轮北京时间19:27:03开始、21:43:35训练停止、21:51:49发布清单确认，约2h24m46s完成，符合4h。Release现有563资产、3,742,547,022字节，其中本轮新增317资产/1,512,769,196字节与12段视频；全部窗口共17视频/364NPZ/17PT。现有602条清单包含563个Release条目及Git小记录，正常名称/大小/上传状态核对无差异，pending为空。

离开本次交互前推送已完成源码、配置和启动记录；训练退出后自动追加全部实际新增环境记录、日志、每个episode完整视频、原始轨迹、物理/预测/材料数组、rollout/checkpoint和有效尾buffer。视频从保存轨迹渲染，旧资产引用原链接。D57直接上传Git/原Release，正常看一次清单；不全量hash、不回下载复验，**不在用户电脑另做下载、镜像、复制、压缩或归档**。训练状态与发布状态分别报告。可变collection/progress/learning等记录按本次job发布新版本，不能覆盖旧Release同名证据。

开头最多5行后直接执行，退出交互时只更新顶部6行和正常结果，明确后台是否仍运行与发布待办；不要为交互收尾停止已授权训练。禁止额外单元/mock/冒烟/导入/形状测试套件、无关回归、独立benchmark、恢复演习、通用fallback、反复计划或审查报告。

## 6. 本次文献更新：以问题相关性决定用途

原清单已有27篇且包含2026条目；不足在上一版当前计划反复只用PPO/RLOC/STANCE概括，没有把近期软地接触和联合控制工作转成具体决策。本次在[原文献清单](../literature/terrain_memory_sources.json)新增TM28–TM31，并更新既有TM13/TM14的当前用途，记录实际阅读范围与证据级别，不声称已复现。

| 来源 | 可借鉴的具体内容 | 当前边界与用途 |
|---|---|---|
| [TM28 Soft Bilinear Inverted Pendulum](https://ddebenedittis.github.io/soft_bilinear_inverted_pendulum_astro/)，2024在线/2025卷 | MPC规划和全身跟踪都处理软接触；作者报告SOLO12实验 | 用于检查预测与执行的一致性；本项目非足接触缺口来自自身数据，不是论文替我们证明 |
| [TM29 Walking in Mud](https://portalcientifico.upm.es/en/ipublic/item/10381051)，2025期刊 | Go1实泥地；接触检测、质心位置、摆腿速度一起控制 | 说明支撑应在整机闭环处理；泥地屈服/吸附不同于本项目黏弹恢复，不能照搬慢步态 |
| [TM30 Terrain-adaptive MPC + impedance + SINDy](https://pure.qub.ac.uk/en/publications/terrain-adaptive-quadrupedal-locomotion-using-model-predictive-co/)，2026-08期刊在线 | 接触检测、阻抗/导纳、动力学修正一体化 | 已核对证据为PyBullet仿真；不用它当真机证明或新增SINDy任务 |
| [TM13 RK-MPC](https://arxiv.org/abs/2604.04221)，2026预印本 | 学习残差真正嵌入MPC预测，报告Go1仿真/真机 | 区分“PANN被调用”与“材料记忆进入MPC候选动力学”；后者是后续研究，不立即换Koopman |
| [TM31 Residual MPC](https://arxiv.org/abs/2510.12717)，2025投稿预印本 | GPU并行MPC在RL训练环中，扭矩层残差连续学习 | 支持完整闭环多轮采样/更新；MIT Humanoid、100Hz扭矩残差与本项目2Hz cap权限不同，不照抄训练量或重写GPU MPC |
| [TM14 LF2WB](https://arxiv.org/abs/2603.03138)，2026，作者注明ICRA接收 | 前进写入、倒退读取固定大小关联记忆 | 几何记忆与材料加载/恢复状态不同；借鉴记忆组织，不引入无关视觉模块 |

原[TM02黏弹PANN](https://arxiv.org/abs/2401.14270)仍是材料历史建模的重要直接来源，不能因2024年份就丢掉；PPO等基础方法也保留。近期论文没有给出“1024条/4更新就足够”的结论，该数值来自当前吞吐与工程观察需求。借鉴应落到控制/采样决策，不能变成大规模阅读、复现或认证前置阶段。

本次覆盖的旧第二更新计划见[baf4584b](https://github.com/kaiwen123-yang/MCP-RL/blob/baf4584b868379450202405c3e8d868027bf1ead/docs/TERRAIN_MEMORY_PLAN.md)；原RAW审查和AutoDL资源安排见[ac1753d](https://github.com/kaiwen123-yang/MCP-RL/blob/ac1753d69d2e17408a157d273fc08313de13628c/docs/TERRAIN_MEMORY_PLAN.md)，原PANN实际结果见[f8bfb462](https://github.com/kaiwen123-yang/MCP-RL/blob/f8bfb4629a1d342f49a63a446ebd2a19151075d8/results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/pann-first-update.json)。历史结果保持原样，旧当前任务不再驱动新窗口。
