# S1 当前推进：AutoDL 25vCPU计算安排与PANN首次学习

## 当前交接（每窗口结束只更新这6行）

- 当前任务：在用户现有AutoDL实例完成必要迁移，直接推进W3 PANN seed0首次累计256真实转移/一次PPO更新，发布后停止。
- 当前状态：仓库已确认RAW256条/1次更新/205条有效平移/尾buffer0；PANN0条/0次；原材料E30/E60和完整往返仍为0；D57依赖与资产已配置；20workers下真实采集/更新尚未执行。
- 已完成：W1及RAW全部实际产物已发布；R12入口已有真实采集/更新/恢复能力；已去掉旧主机路径及硬编码5workers，契约采用AutoDL单流20workers；D57项目环境、r21/Go2/MJPC接口/B ideal和独立Clash已配置。
- 当前问题：PANN在线观察/恢复查询待实际运行；共同执行器仍有侧出条带和返程支撑失败；RAW原loss缺失、更新后策略尚未评价；迁移只处理真实运行暴露的问题。
- 下一步：在D57项目目录执行source scripts/autodl_d57_env.sh后按新用户授权窗口启动；20MPC workers、Torch2、OMP/BLAS1，保留r21和原数值设置；不重跑RAW，不另做benchmark/恢复/哈希测试，单窗口全部工作≤4h。
- 续接位置：autodl-d57:/root/autodl-tmp/MCP-RL，环境work/envs/d57；实际配置见experiments/terrain_memory/compute-environment.json。沿用tm-s1-go2-local-readiness-r12及原Release；PANN独立目录/自身checkpoint；原RAW checkpoint与历史保留在5080和Release，未复制到D57。

## 1. 本轮实际结论

用户称为第13轮的这次推进，沿用`tm-s1-go2-local-readiness-r12`目录。无需改名、复制或重建run。上一窗口按用户要求在RAW后停止；本计划覆盖旧“RAW后自动接PANN”的文字，只安排一个独立PANN窗口。

**RAW首次真实学习工程目标完成，材料控制任务仍未通过。** [索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/index.json)和[全部256条转移](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/records/training__PPO_RAW_S5-seed0__transitions.jsonl)支持以下记录：

| 真实会话 | 转移/有效平移 | 仿真秒 | 结局与路线 |
|---|---:|---:|---|
| ep00002 | 75/54 | 37.4260 | 到远端并返程，10.304s曾侧出条带，最终非足端接触 |
| ep00004 | 44/40 | 21.8460 | 去程非足端接触，未完整穿过材料 |
| ep00005 | 77/57 | 38.3764 | 到远端并返程，16.174s曾侧出条带，最终非足端接触 |
| ep00006 | 60/54 | 30.0000 | 曾侧出条带；到第256条外部截断并更新，随后日志异常 |

ep00001/03是零步初始化失败；另有一次Tianshou缺失发生在物理会话前。四条物理任务共638,242步，最大应变均<.02、无material_domain_exit，原条带贯穿均未成立。编号不能当作六条物理失败。

全部256条dt>0；真实terminal在75/119/196，唯一truncated在256。RAW最后12个PANN特征通道为0；有效平移205，其余51是站立/转向等真实样本，不删除重采。实际cap约.189914～.239870m/s；旧结果顶层constant/.25是沿用元数据，以transitions/cap-decisions为准。

[更新证据](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/records/training__PPO_RAW_S5-seed0__learning.jsonl)显示一次更新、各参数16个Adam步、Actor/Critic发生变化、尾buffer0。旧`samples`键冲突发生在更新后，原loss标量未保存；保留缺项，不重新优化或虚构原始loss。全部四会话发生在首次更新之前，较长轨迹不能归因于学习；更新后的策略没有物理评价。

## 2. 已完成的修复与本次小修

R12入口、旧字段适配、六流等待关闭、CUDA设备、自然新会话创建Agent、Agent.model、截断顺序、零时长过滤及更新后计数均已实际接通。继续复用，不再列作待开发。

本次只做直接影响下一次启动/解释的小修：wrapper移除上窗口写死的start/deadline，统一接收既有Python参数；任务元数据读取当前契约；保存单流所需PPO和共同观测设置，去掉继承R11六流大目标和正式冻结门禁的无效说明；新策略结果标为`ppo_policy`，保留真实动作记录。未改变PPO数学、奖励、控制器或已有RAW结果。

## 3. 当前窗口：AutoDL迁移直接接W3

用户于2026-10-30指定新执行端，覆盖先在5080完成后再决定租卡的旧安排。使用已提供的AutoDL实例，不另购实例或扩大租赁。SSH别名autodl-d57与数据盘工作区已配置，具体版本见计算环境配置；本次设置未启动训练，不新建迁移报告或认证流程。

### 3.1 资源与计算分配

| 资源 | 截图配置 | 当前用途 |
|---|---|---|
| CPU | 25vCPU，Xeon Platinum8470Q | 单个真实环境，MPC20workers；其余约5vCPU调度余量给物理循环、JAX/Torch、RPC及I/O |
| GPU | RTX PRO6000，96GB×1 | PPO与PANN用CUDA；保留JAX按需分配和x64，不为占满显存扩大网络/batch |
| 内存 | 120GB | 正常物理状态、MPC线程状态、材料数组与记录；不以占满内存为目标 |
| 磁盘 | 系统盘30GB、数据盘50GB | 工作区与输出优先放数据盘；这不是120GB磁盘，实际新增产物逐窗口直传Git/Release |
| 镜像 | Ubuntu22.04，Python3.12，Torch2.8.0，CUDA12.8 | 按用户复用镜像要求使用独立Python3.12.3项目venv与Tianshou0.5.1，复用镜像Torch2.8/cu128；依赖已安装 |

**首窗具体值：单流、MPC20、Torch2、OMP/OpenBLAS/MKL各1。** 这是软计算预算，不代表系统所有运行时线程总数严格等于25，更不代表25个独占物理核。当前导数任务按预测时间节点并行，候选轨迹数和优化迭代数独立于worker数；因此先增加worker有源码依据，但不承诺线性加速或逐位相同。保持0.2ms物理步、2ms控制保持、20ms规划周期、0.35s预测，以及原节点/候选/迭代工作量。

用正常真实会话已有计时观察MPC、物理、PANN与总墙钟，再算全部/有效转移每小时。不额外做worker扫描。若实际任务发现20workers调度收益不足，依据该次记录在下一自然会话或下一窗口调整并记录；不丢弃真实样本，不反复启动同任务“证明配置”。旧RAW约102.28分钟/256条、规划占85.77%，只能说明旧链的主要瓶颈，不能直接当AutoDL PANN加速基线。

### 3.2 必要迁移与真实任务

| 项目 | 本窗执行 |
|---|---|
| 工作区 | 建议/root/autodl-tmp/MCP-RL；SSH记录已有正常工作区就沿用，不为符合建议路径重新复制 |
| 环境 | 激活项目环境，恢复需要的MuJoCo3.2.3、匹配的MPC Python客户端、PPO/PANN依赖；不搬整套旧IsaacLab工作站，不同时升级Tianshou接口 |
| 执行器/资产 | 沿用r21源码及已发布资产；只取本任务需要的包。二进制在真实启动暴露兼容问题才按原源码必要构建，不先复现全部旧轮次 |
| 权重 | 冻结B ideal，默认放work/assets/B_random1000--ideal-potentials.npz；实际已有路径则改契约resources.pann_weights_path |
| 唯一训练流 | PANN_PPO_S5 seed0；独立初始化或继续自己的实际checkpoint，绝不加载RAW权重/样本 |
| 采集与算法 | nominal、60s连续会话、课程关闭；累计256真实转移/首次更新；原PPO/奖励/归一化、2Hz cap[.18,.25]与因果信息边界 |
| 时间/交付 | 最多4h含全部准备/运行/渲染/上传，通常3.5h收尾；目标完成即完整发布并停止，未完保存真实进度 |
| 实际验证 | 本次真实采集、更新、普通loss/特征/轨迹与checkpoint就是证据；不增加测试套件、恢复演习或发布认证 |

已将wrapper改为从脚本位置找仓库、使用当前激活的Python；不再跳回/home/fanyuan或旧Conda。匹配的MPC客户端由当前项目环境提供。训练入口的元数据和Agent启动从同一个resources.mpc_workers读取20，冻结权重同样从契约读取，Torch设置不再被入口无声覆盖。SSH、所需依赖与复用资产已在D57配置完成；真实闭环适配仍由后续授权的实际训练确认。

入口仍为：

```bash
bash scripts/run_s1_go2_r12_training.sh --group PANN_PPO_S5 --seed 0 --window-start-unix <本窗口首次活动时间> --deadline-unix <预留收尾后的训练截止>
```

时间从首次阅读/连接活动计算，不到训练启动才重置。首次无PANN进度时不传--resume；实际已有PANN进度则只续自己的checkpoint和累计目标。如果最新交接已完成256/1，复用成果，不重复这个目标。真实physical terminal/外部截断和有效尾buffer沿用已有机制，普通更新不reset。

用正常episode/rollout记录观察器推进、查询次数、12个恢复特征和实际耗时；为零如实报告。当前PANN查询有多次主机/GPU往返的可能，只有实际运行确认耗时后，才在后续独立窗口处理相应实现；本窗不提前重写PANN积分或加入GPU压力测试。

### 3.3 后续并行方式与窗口拆分

当前PANN首次学习完成并审查后，下个窗口优先解决已有侧向/支撑问题或明确安排继续训练，不自动串起来。如果后续有两个明确独立训练seed/组或评价case，可先用**2流×8MPC workers，每流Torch2、OMP/BLAS1**，用真实任务总吞吐选择资源分配。

当前入口仍为单流。实际启动两流前，只需对具体共享资源做必要改动：现有training-deadline.json改为流内路径，各流独立策略/优化器/RNG/物理状态/输出/checkpoint；现有index已有flock、Agent会自动选端口，直接复用。多个环境给同一个PPO策略采样是另一项collector工作，当前不开发。旧六流等待和48h campaign继续关闭；并行子进程也共用同一窗口4h截止。

正式评价保持同控制器/worker/权重等配置的E30→E60→E300→E600。旧5080 RAW与新AutoDL PANN首批仅支持工程判断；后续匹配比较使用相同执行配置。256是首次更新目标，不是以后永远只准采256；后续训练在明确任务窗口内按实际吞吐持续接续，保留真实累计进度。

官方运行依据：[AutoDL数据盘](https://www.autodl.com/docs/instance_data/)、[PyTorch2.8/cu128的Python3.11发行包](https://download.pytorch.org/whl/cu128/torch/)、[Tianshou0.5.1](https://pypi.org/project/tianshou/0.5.1/)、[CPU线程与过度并行](https://docs.pytorch.org/docs/2.8/notes/cpu_threading_torchscript_inference.html)、[JAX显存](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)。Python3.11是减少迁移变量的选择，不意味着3.12必然不兼容；不得因此重建已经正常工作的环境。

## 4. 完整发布与轻量交接

[现有Release](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-local-readiness-r12)已有246资产、约2.23GB，包括5个完整视频、172个NPZ和4个checkpoint；环境/小记录另在Git。现有清单中应传Release的文件名/大小与资产列表一致，W1和RAW没有待传项。一个小episode JSON保存在Git，原loss缺失已明示。

下一窗口只追加本轮新增/变化源码、配置、环境记录、正常日志，以及全部实际PANN视频、原始轨迹、物理/预测/材料数组、rollout和checkpoint。继续每个实际episode完整记录，视频从保存轨迹渲染；不为补材料重跑物理。引用已发布且未变的r21/PANN/场景资产，避免重复上传。

AutoDL执行端直接采用实际可用的Git/Release传输；旧5080的curl代理是历史实现，不写死复制到新服务器，也不为迁移重建一套发布平台。现有索引记录大小/URL/状态，正常看一次清单即可；不全量hash、不回下载复验，不在用户电脑另做镜像/复制/压缩归档。训练状态与发布状态分开记录，未传完如实pending。

开头最多5行后直接工作；结束只更新顶部6行和正常结果。禁止重复全仓审查、单元/mock/冒烟/导入/形状测试、通用fallback、无关矩阵及新的日报/归档包。

## 5. 后续方向与论文依据

PANN首次链路完成后再审查，下一窗口不自动开始后续研究。共同控制器优先看侧向离开材料条带和返程承载交接。当前奖励含路线进度，却没有单独的原条带贯穿项；已经出现侧出后仍得进度分的样本。它们来自更新前策略，不能叫“学会绕路”，但在长时间扩训前应对齐任务目标、执行能力与学习信号；本次先保持两组相同条件，不同时重写奖励和控制器。

PPO目前仅调2Hz速度上限，不能承诺修复快速支撑失效。更换控制器后标明版本边界，不拿旧RAW和新PANN当收益对照。正式评价仍逐case同版E30→E60→E300→E600，训练前缀不替代资格。用户已选择AutoDL；按本计划的25vCPU资源配置继续解决在线链和控制问题，再用实际采样成本安排后续窗口，不再等待5080阶段作为云端前置门禁。

- [PPO](https://arxiv.org/abs/1707.06347)：交互采样与优化交替；256/四小时是项目工程安排，一次更新不代表收敛。
- [RLOC，§VI-H](https://arxiv.org/html/2012.03094v3)：优化控制在环时采样可能主导成本；本轮不据此换SAC/TD3或控制平台。
- [STANCE，§IV](https://arxiv.org/html/1904.12306v2)：软地支撑运动与接触动力学需要相容，指导后续具体支撑分析，不增加首次PANN训练前的复现门禁。
- [JAX GPU内存说明](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)：现有wrapper已关闭默认预分配以共用GPU；保持已有设置，在真实运行中处理具体问题。
