# 四足机器人地面记忆辨识与跨步协同控制

**默认计算位置（2026-10-30 起）：AutoDL 西北 B 区 D57，SSH 别名 `autodl-d57`。** 后续训练、仿真和项目计算统一使用该实例。RTX PRO 6000 96GB ×1、25 vCPU、120GB 内存；[实际环境与复用来源](experiments/terrain_memory/compute-environment.json)。

## 当前入口：AutoDL 25vCPU推进PANN，单窗口最多4小时

**第13轮RAW首次真实学习工程目标完成。** 现有R12 run中，RAW seed0已采256条真实转移（205条有效平移）、完成1次PPO更新、尾buffer0。四个真实会话共127.6484s，三次非足端接触、最后一次目标截断后发生日志异常。原loss标量缺失已保留，更新后checkpoint已保存。PANN尚未启动。[实际索引](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/index.json)

全部四条轨迹均来自首次更新前策略，完整往返和有效材料E30/E60仍为0；较长运行不能称训练后改善。两条到达远端并返程的轨迹都曾侧出材料条带。RAW实际cap约.189914～.239870m/s，旧顶层constant/.25字段不代表真实动作。

用户于2026-10-30指定现有AutoDL实例：**25vCPU Xeon8470Q、RTX PRO6000 96GB、120GB内存，系统盘30GB/数据盘50GB。下一窗口必要迁移后直接做PANN seed0累计256真实转移和一次更新，完整发布后停止。** 首窗单流20MPC workers、Torch2、OMP/BLAS1，复用r21与原物理/算法设置；这是未实测起点，按真实采样耗时调整。全部工作最多4h，不重复RAW、不补造loss、不自动串W4或长评价。

先读[AGENTS](AGENTS.md)和[主计划当前交接](docs/TERRAIN_MEMORY_PLAN.md)，按[当前契约](experiments/terrain_memory/s1-go2-local-readiness-r12.json)执行。wrapper现使用仓库相对目录与当前激活Python，workers和PANN权重路径读当前契约；显式传本窗口group/start/deadline。执行时读取用户正在补充的SSH/环境记录，不猜连接地址。D57依赖和资产已配置完成，真实PANN采集/更新仍未执行。

**W1和RAW完整科研产物已交付。** [Release](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-local-readiness-r12)有246资产、2,229,777,826字节，包括5个视频、172个NPZ、4个checkpoint；源码/环境/小记录在Git。下一窗口追加全部实际PANN产物，沿用现有上传方式，不回下载、不在用户电脑另做归档。

正式评价继续E30→E60→E300→E600。后续重点为横向路线保持、返程支撑以及任务目标与学习信号的一致性；首先完成现有PANN联合运行链。

## D57 环境与联网

在用户电脑执行 `ssh autodl-d57`。服务器项目目录为 `/root/autodl-tmp/MCP-RL`，进入项目环境：

```bash
cd /root/autodl-tmp/MCP-RL
source scripts/autodl_d57_env.sh
```

Python 3.12 项目环境直接复用镜像的 PyTorch 2.8.0+cu128；MuJoCo 3.2.3、JAX 0.6.2 CUDA、Optax 0.2.5、Tianshou 0.5.1 等依赖按 [requirements-autodl-d57.txt](requirements-autodl-d57.txt) 安装。r21 执行器及其运行资源、原 Go2 资产、MJPC Python 接口和冻结 B ideal 权重从原 5080 服务器复用。原 RAW checkpoint 与完整历史仍留在原执行端及 Release，PANN继续独立初始化。此次环境配置没有产生新训练、物理任务或性能结果。

默认联网顺序为官方学术加速优先；同一联网操作连续三次网络失败，再使用用户电脑 VPN 的 SSH 反向隧道。服务器端分别执行：

```bash
source scripts/autodl_network.sh academic
# 三次网络失败后，在电脑端先启动下面的隧道，再切换：
source scripts/autodl_network.sh pc
```

电脑端的备用隧道入口：

```powershell
& .\scripts\start_d57_vpn_tunnel.ps1 -LocalProxyPort <电脑当前Clash端口>
```

服务器也已安装 Mihomo（Clash 内核）1.19.33，并复用用户授权的两个节点。`source scripts/autodl_network.sh clash` 会启动或复用 `d57-clash` 后台会话并选择服务器本地代理；此方式不依赖电脑持续开机。该容器没有 systemd，实例重启后再次执行此命令即可启动。私有配置在 `/root/.config/mihomo/config.yaml`，只监听回环地址，未启用 TUN 或外部管理端口；配置及节点凭据不进 Git。`source scripts/autodl_network.sh direct` 清除当前 shell 的代理变量。

## 历史说明

以下保留R11及更早入口和结果原文。历史文字中的“当前”“本轮”、预算与门禁只适用于对应run；当前任务采用本页顶部的独立PANN窗口与最多4小时规则。[R11原计划](https://github.com/kaiwen123-yang/MCP-RL/blob/25947cfae9b2a48a872327bb63e9b36f76fe2dac/docs/TERRAIN_MEMORY_PLAN.md)和全部实际结果保留。

## 历史 R11：材料执行与阶梯评价

R11已按用户2026-10-29的要求暂停，发布现有成果供审查。累计23次真实物理启动（22次E30、1次E60），零步初始化失败0；20次物理/材料失败、1次技术中断、1次原判通过但资格无效、1次材料贯穿条件未完成。累计仿真271.553秒，任务执行耗时5.0646小时；有效E30/E60通过均为0，S5未冻结，PPO训练0，E300/E600、实机与NX测试均未执行。dev07的30秒记录及其后启动的dev08 E60因实际步态与声明不符而失去资格；dev18虽运行30秒并到达远端，但侧向绕出材料条带。最新dev23/S5-r21于14.358秒因非足端接触失败。全部原始记录、判定修正和连续轨迹保留。统一时钟仍从北京时间2026-10-28 23:36:01起算，预算不重置；暂停期间不继续实验或训练。

实际启动、失败、阶段资格和当前状态见[唯一R11索引](results/terrain_memory/simulation/tm-s1-go2-material-continuation-r11/index.json)。以下保留本轮阶梯协议；未执行阶段不作为成果。


用户最新要求评价按**30s→60s→300s→600s**逐case推进，覆盖此前60直接600和300仅观察点的安排：

| 级别 | 评价目标 |
|---|---|
| 30s | 原路线实际经过材料、到达3m远端并完整运行30秒；继续转向/返程、记录进度，不把短阶段称整程成功 |
| 60s | 原完整往返和连续2秒终点稳定，观察满60秒 |
| 300s | 独立连续5往返，270秒路线期限，保持并观察至300秒 |
| 600s | 独立连续10往返，540秒路线期限，保持并观察至600秒 |

同控制器/权重/归一化、seed、初态的case前一级通过才进后级；某case失败不阻塞其他case，不能借nominal或最佳seed越级。每级同一21-case清单，完整目标84条材料阶段记录＋1刚地回归；预声明资格可复用但不重复计数。长轨迹前缀和训练记录不冒充独立正式评价。

本轮从R10 GNC-r3/.215m/s继续，先解释返程快速落足并完成材料往返。候选先nominal30→60，60完整成功后冻结S5；固定组按300→600推进，同时六流真实联合训练从60秒采集开始，依据本流完整会话实际圈数逐步升300/600，不按8192/16384样本强制解锁。课程升级不替代正式资格，圈间和256更新不reset。

**R10真实结果：12启动＝3零步初始化＋9物理任务，完整任务0/9，PPO训练0。** dev11在19.582s开始远端转向、26.090s返程、32.5672s材料域退出，末态距起点1.98m，约完成2/3路线。已有观察器回放与PPO接口不算稳定联合闭环。R8刚地S3的9/9原任务＋3/3连续300s仍成立。
[R10索引](results/terrain_memory/simulation/tm-s1-go2-soft-landing-r10/index.json) · [逐任务数据](results/terrain_memory/simulation/tm-s1-go2-soft-landing-r10/episodes.csv) · [原始发布](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-soft-landing-r10)

取消12/61硬次数上限，保留24h开发与48h总campaign及证据约束。按实测吞吐预留完整名义阶梯依赖链，预算不足交真实最高通过级别和缺项，不能承诺48h跑满全矩阵。1800/3600只在四级核心之后按资格/预算条件考虑。

源码/紧凑结果推GitHub，大产物从执行端直传Release；核对commit/URL/size/digest后结束，**不在用户电脑再次下载、复制、镜像、压缩或归档**。

## 历史结果原文

以下保留旧结果与协议叙述；旧“当前”“本轮”、额度和阶段门槛仅属于对应历史run。当前执行以以上R11入口为准。R10原协议的完整版本见[02595603历史主计划](https://github.com/kaiwen123-yang/MCP-RL/blob/025956032e2a3aa1e0c9881f2eda307301c77522/docs/TERRAIN_MEMORY_PLAN.md)。


**R10 实际结果（2026-10-28）：开发额度12/12已结束，材料完整任务0/9，联合PPO未启动。** 12次启动包括3次零物理步初始化失败和9次真实Go2材料任务；累计107.9750秒、539875个0.2ms物理步。G、G＋N与G＋N＋C均保留完整记录，C的局部力参考和支撑高度修正逐版保存。最终0.215m/s版本通过去程和远端转向，在返程32.5672秒触发原材料域；仍未完成原任务。

冻结PANN的独立观察器完成既有R9记录的4169次状态推进与17次条件恢复查询，权重SHA256与原B ideal seed0一致；这不是闭环成功。标准PPO接口已准备但未执行，六流训练、PPO更新、新策略权重、21条60秒与21条600秒正式评价均为0。三模块稳定联合执行尚未成立，PANN改变动作及改善任务均未评价。真实材料、Go2物理、PD/力矩、任务阈值未放宽；未启动硬件，未测NX实时。

48小时墙钟上限未耗尽，开发启动额度已耗尽。剩余49个预定评价case保留清单与未执行原因，不自动刷新额度。当前已无R10仿真或训练作业；`tmux`保留执行日志入口。

[唯一运行索引](results/terrain_memory/simulation/tm-s1-go2-soft-landing-r10/index.json) · [全部启动与结果](results/terrain_memory/simulation/tm-s1-go2-soft-landing-r10/episodes.csv) · [轨迹与阈值图](results/terrain_memory/simulation/tm-s1-go2-soft-landing-r10/development-trajectories.png) · [未执行评价清单](results/terrain_memory/simulation/tm-s1-go2-soft-landing-r10/evaluation-manifest.csv) · [完整数据与负例视频](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-soft-landing-r10)

以下R9及更早内容保留其历史身份。

**R9实际状态（2026-10-28）：8次材料适配启动已用完，完整任务0/8，联合训练未启动。** 同一冻结R8-S3接原TM02中段条带，五种固定cap与三条预定2Hz时变序列均在6.5518–8.3448s于去程触发原主应变域上限.02。另有1次零物理步初始化失败。物理累计61.2944s/306472步，并发峰值3环境×2solver workers。PANN观察器与联合PPO环境未实现，训练/正式评价为0；材料、S3和阈值未放宽。48h预算未耗尽，8次适配额度已耗尽，不刷新额度。[唯一R9索引](results/terrain_memory/simulation/tm-s1-go2-joint-r9/index.json) · [八条真实轨迹](results/terrain_memory/simulation/tm-s1-go2-joint-r9/all-intents.png) · [完整负例数据与视频](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-joint-r9)

**最终研究问题（用户于 2026-10-21 确认）：机器狗在会被自己改变的软地面上，怎样决定当前加载、落点和重访时间，为后足及返程留下可用的支撑条件？**

同一空间地块的材料状态跨腿、跨步、跨返程持续演化；后续支撑需求反过来约束当前踩踏。第一版从具有可测恢复过程的黏弹性材料开始。

## R9冻结协议及本批执行范围

**第八轮已完成项目Go2同版9/9原任务＋3/3连续300秒，S3继续冻结。R9实际完成8次材料适配尝试，0/8整程成功；训练前置条件未通过且适配额度已用尽。** 以下保留原定联合协议，其中PANN观察器、PPO训练及正式21+3评价均未执行，不能作为已完成成果。实际数字见顶部与[主计划第5.10节](docs/TERRAIN_MEMORY_PLAN.md)。

近期目标是沿同一个S3执行器形成稳定的材料PANN＋标准PPO＋MPC联合版本，再研究方法创新、轻量化和泛化。Go2作者配置组与项目约束组均为四足Go2；实机目标仍为Go2＋NX。

### 从已经成功的执行器继续

R8实际19次物理启动，形成项目九原任务和三长任务完整覆盖，名义300秒在214.356秒完成五往返并保持至300秒；项目全部完成后保持无原判据超限。[原结果](results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/index.json)和四条开发负例冻结。S3的原模型、PD、力矩界、2ms目标保持、预测导数及驻足参考继续使用，不重开旧轮额度或把刚地修补继续当研究主体。

### 本轮材料、模型与策略各自做什么

真实仿真在去返程中段设置受控TM02材料条带：**x∈[0.8,2.2]m、y∈[−0.6,0.6]m**，起点、终点与转向区域保持刚地。材料本构、γ、厚度H及接触面积映射沿用原TM02定义；回合内材料状态跨足、跨步、跨返程持续演化，不因换腿或返程重置。

冻结R2已有 `B_random1000--ideal` PANN参数。PANN维护独立历史状态，根据允许观测产生材料预测特征供Actor使用；真实材料内部状态与未来真值不进入Actor。**MPC仍采用原生刚地预测近似，本轮不宣称PANN已经进入iLQG候选动力学，也不新增材料状态的候选分支求导。**

标准PPO从头训练，每0.5秒决策一次，即**2Hz**；第一版只调整去程/返程速度上限，范围 **[0.18,0.25]m/s**。Stand、转向、步态、关节目标及力矩不交给Actor直接修改。真实plant和材料按原0.2ms物理步推进，PANN按500Hz更新，Actor按2Hz决策；三种时钟分别记录。

### 用同一执行链完成三个对照

| 组别 | 本轮作用 |
|---|---|
| 固定意图＋S3 MPC | 同一材料任务上的非学习执行参照 |
| 无PANN预测特征的标准PPO＋S3 MPC | 判断普通策略学习本身的作用 |
| PANN预测特征＋标准PPO＋S3 MPC | 判断冻结材料模型是否实际改变动作和整程结果 |

两PPO组各用3个预定训练seed，保留相同动作权限、物理任务及信息边界。首批目标为每seed4096个高层转移；时间允许时两组共同扩到每seed8192，总硬上限 **49,152个高层转移**。整个R9实验campaign的墙钟预算为**48h，包含适配、训练与正式评价，至少预留6h正式评价**。4096是首批目标而非保证；实际共同样本量不足4096也如实记录，不让一组多训练后作不等量比较。

训练采用同一Actor→MPC→Go2真实仿真闭环。**训练episode首次满足完整原连续2s稳定判据后即可结束；正式评价仍按原60s或300s观察，不用成功后的被动Stand凑训练步数。** R8约12倍慢仿真须落实到本轮预算，利用完整训练episode估计实际吞吐量，不单独开benchmark轮，也不用另一快速控制器收集动作后冒充本执行器训练。

### 实验交付与推进边界

材料接入与适配最多8次完整任务，训练额度单列。正式评价为 **21条60秒任务**：固定意图3个初态，加两PPO组各3训练seed×3初态；另预定 **3条nominal300秒任务**，分别使用固定意图与两PPO组的seed0。对应名义原任务未成立时保留失败边界，不换最好seed制造长任务成果。

保留原任务阈值、完整执行时长和真实失败，长任务持续五往返且不在圈间重置物理或材料。比较实际任务成功、保持、材料加载/恢复、跨足或返程重访与动作差异。PANN被加载不等于有效集成，预测数值不同不等于改善决策；完整组可能不优于固定或无PANN组，结果照实保留。

本批次不重新训练PANN、不改PPO算法、不启用硬件，不把受控材料条带或慢仿真称未知地形泛化、材料记忆创新或NX实时。必要构建与动力学计算直接服务完整任务；禁止防御性编程、小型单元/mock/冒烟/导入/形状测试、无关矩阵和审查/状态报告。

复用现有SSH/tmux。交真实训练曲线、模型权重、完整任务视频、高率物理/材料/预测记录及同版对照。源码与紧凑结果推GitHub，大数组和视频由执行端直接发布Release；核对最终commit、资产URL/size/digest后结束，**不在用户电脑再次下载、复制、镜像、压缩或归档。**

以下R8及更早内容保留历史身份；旧“当前/本轮”、停止要求及预算只适用于对应已关闭run，当前执行以上述R9和主计划第5.10节为准。

## S1第八轮历史结果（已冻结）：项目Go2指定9＋3覆盖全部通过

**同一冻结S3控制器通过9/9条原任务和3/3条300秒连续任务。** `tm-s1-go2-wb-support-r8` 已完成指定实验范围，形成刚地项目物理约束下的MPC仿真执行基础。PANN/PPO/MPC联合闭环尚未接入；近期仍先形成联合基础，再研究创新。Go2作者配置组（G0）和Go2项目约束组（G1）始终都是四足Go2，实机目标仍为Go2＋NX。

| 实际任务 | 结果与观察范围 |
|---|---|
| 作者固定N三条300s | nominal、yaw±5°均通过；五往返在220.686–221.470s完成，均观察至300s |
| 项目开发C / H / S1 / S2 | 四条均观察到60s但未满足原连续2s稳定；最长稳定分别0.136 / 0.192 / 0.938 / 0.034s |
| 项目S3九条原任务 | nominal、yaw±5°、vx/vy±0.05m/s、wz±0.1rad/s全部通过；36.292–44.136s完成，均观察至60s，完成后零判据超限 |
| 项目S3三条300s | nominal、yaw+5°、yaw−5°分别214.356、216.564、218.454s完成五次连续往返；均观察至300s，完成后及末30s零判据超限 |

共**19次新物理启动＝作者3＋项目开发6＋正式新增10**；两条名义资格任务在同版正式覆盖中各复用一次，未重复启动，未动用正式修订。39次上限未用满。另保留7次进入物理积分前的初始化失败，physical_steps均为0。四条完整开发负结果、未收敛的初次参考求解和全部旧轮结果均保留。

作者三条长任务末30s仍分别存在12、32、5个角速度超限样本，最终连续稳定分别23.480、2.560、5.636s；“任务通过”不代表没有中断，作者模型也不能替代项目物理验证。九个初态是确定性覆盖，不是随机seed统计；一条长任务的十次到达不作为十个独立样本。

C在真实源码中实现2ms控制保持、同一输入下两次1ms积分及一致导数，0.350s为175转移/176状态；H加入可行绝对高度。两者不足后，S在同一MPC内使用固定世界足目标、零足速度和相容机身/关节/支撑输入参考，满足单位gear下 `u*=q*+tau*/60`。S1仍有接触短暂丢失；S2软竖直力目标使结果恶化；S3撤去该目标，按已观察到的反馈目标跳变将输入参考尺度从0.1rad改为0.02rad，保留其余S1设置。新增目标均为软代价，没有新增独立站立控制器或稳定性证明。

项目保留15.206408kg、0.0002s真实物理步、500Hz控制、20ms规划、原接触与23.7/45.43Nm力矩界；作者保留15.806408kg、0.002s物理步及原N二进制。每条任务2个求解worker、nice 10，正式最多4条并发；后处理及EGL/NVENC渲染以nice 15与部分物理任务重叠。全部是共享CPU上的同步慢仿真，**不代表NX实时性**。

[单一运行索引](results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/index.json)记录实际源码/配置/二进制、分片全局时钟、预算和最终发布核验。[全部任务](results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/episodes.csv)、[同版正式覆盖](results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/coverage.csv)、[完成后保持](results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/holding.csv)、[长任务末30秒](results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/last-30s-holding.csv)、[原生支撑参考](results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/support-references.csv)、[控制耗时](results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/timing.csv)和[渲染耗时](results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/rendering.csv)分别保留。

[名义原任务图](results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/figures/project-s3-single-05.png)与[名义300秒图](results/terrain_memory/simulation/tm-s1-go2-wb-support-r8/figures/project-s3-long-06.png)便于浏览；高率数组和全部成功/失败完整视频从执行端直接上传[本轮Release](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-wb-support-r8)。最终提交、正式URL、size和SHA256核验以索引publication为准；没有在用户电脑再次下载、复制、镜像、压缩或归档。

发布核验发现一个执行端模型副本与运行时记录相差2个字节；先前上传的原始模型及另一份冻结模型的SHA256仍与运行时一致。变化副本和字节差异记录保留，原始Release资产未替换；清单使用已有的相同原始模型作为核验来源，详见索引publication_integrity_events。

本轮PANN/PPO训练0、TM02运行0、硬件0，NX未测。后续联合顺序见[主计划第5.11节](docs/TERRAIN_MEMORY_PLAN.md)，当前新批次见第5.10节：冻结已有PANN参数、训练标准PPO并评测固定Actor；当前控制修正本身不是三项研究创新。第八轮指定实验已结束，不自动用满剩余额度。

下方第七轮及更早内容作为历史原文保留；其“本轮/当前”和旧停止规则仅指对应旧run。R8原授权契约与历史启动词见[冻结提交中的主计划第5.9及7节](https://github.com/kaiwen123-yang/MCP-RL/blob/e3abb8268353bb4900d8ec5b11cc63391cd8d897/docs/TERRAIN_MEMORY_PLAN.md)。

## S1第七轮历史结果（已冻结）：G0可完成，G1原任务仍未建立

**8/8次开发物理启动已执行：G0为1/3完整成功，G1为0/5。** 未形成可冻结的G1完整成功版本，9个正式物理初态和3条300s连续任务均未执行；未把正式额度转作额外开发。实验已停止，发布确认以运行索引为准。

| 实际任务 | 结果 | 物理记录 / s | 关键事实 |
|---|---|---:|---|
| G0 O | 失败 | 36.974 | 返程后跌倒 |
| G0 N | 原任务成功 | 60.000 | 37.556s首次完成；继续观察到60s |
| G0 ON | 失败 | 43.052 | 返程后非允许机体触地 |
| G1 N | 失败 | 16.522 | 原接触细步预测下非允许机体触地 |
| G1 N_S | 超时 | 60.000 | 已到远端；航向未进入原0.15rad转向门槛 |
| G1 N_SH | 超时 | 60.000 | 35.850s返到终点；连续稳定最长0.160s |
| G1 O_SH | 失败 | 50.524 | 加入按阈值代价定标的驻足目标后跌倒 |
| G1 O_SH_W | 失败 | 4.928 | 仅预测模型缩窄平滑宽度后跌倒 |

G0 N保留原目标，采用0.001s、implicitfast和原接触预测；实际返回350个节点、0.349s时域。它在37.556s达到原连续稳定2s条件，完成后11223个500Hz样本中仍有2个角速度超限样本，末段连续稳定11.698s。G0是作者15.806408kg及未显式施加同类力矩界的物理条件，不能替代项目G1。

G1保留15.206408kg、0.0002s真实物理步、原接触及23.7/45.43Nm逐关节力矩界。N_S启用作者MakeDifferentiable预测近似；它同时影响rollout与导数中的joint/geom/pair d0，真实plant未改。N_SH将Heading权重从0.03提高到1，恢复了原路线，但仍未停稳。O_SH只在Stand启用三维直立与归一化速度残差，速度权重按原容限的相等代价定为0.0224578441；O_SH_W另将预测floor/feet过渡宽度改为实测接触尺度60.743856μm，plant仍为原值。两项修订均未得到完整G1成功。

同输入推演与含未来反馈/重规划的轨迹偏差分别保存。预测均值误差降低没有转化为本轮闭环成功，部分反力峰值误差还增大。前五条任务未连续记录非足部地面反力，保持缺失；后三条补齐全部地面反力。未到终点的保持指标记为空值，不填零。全部原失败、R6与更早记录保留。

本轮训练0、硬件0、TM02运行0、NX实时性未测。联合PANN/PPO/MPC基础尚未建立。实际实现、控制配置、二进制及各次预算见[运行索引](results/terrain_memory/simulation/tm-s1-go2-wb-equilibrium-r7/index.json)、[完整结果](results/terrain_memory/simulation/tm-s1-go2-wb-equilibrium-r7/episodes.csv)、[保持过程](results/terrain_memory/simulation/tm-s1-go2-wb-equilibrium-r7/holding.csv)、[实际耗时](results/terrain_memory/simulation/tm-s1-go2-wb-equilibrium-r7/timing.csv)和[全程视频/大时程发布位置](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-wb-equilibrium-r7)。

## S1第六轮历史结果（已冻结）：完成路径往返，终点连续稳定未通过

以下保留第六轮当时的执行范围和停止决定；其中“本轮结束”等只适用于已关闭的旧run，当前执行以顶部第九轮及计划第5.10节为准。原第五轮及更早结果继续保留。

**第六轮已完成固定范围执行：1次G0完整任务启动，0次完整成功；G1开发与正式均未执行。** 本轮 `tm-s1-go2-wb-startup-r6` 使用实现 `ce3e03255e6e229fd61cac458b514fb6ea162314`，保留第五轮和所有历史结果。没有训练PANN/PPO、接入TM02或启用硬件。

三项启动修正已进入真实执行：作者Init/Reset恢复home后初始化策略；实际世界固定水平floor为−0.01m并写入 `s1_flat_ground_height`；同一任务t=0恰20次原规划，期间plant步数和reset次数均为0。优化前nominal与home ctrl最大差4.77×10⁻⁸rad，最后一次初始化反馈目标与首个实际目标一致。首次最大绝对力矩10.5164Nm，前2s最大机身线速度0.05862m/s，原affine PD最大核对差1.71×10⁻⁶Nm。

| 原任务阶段 | 实际记录 |
|---|---|
| 初始站立→去程 | 2.000s进入去程，初始站立阶段未触发跌倒 |
| 去程→转向 | 15.086s，机身x=2.9673m；全程最大x=3.00363m |
| 转向→返程 | 22.068s，实际yaw=2.99301rad |
| 返程→终点站立 | 35.112s，距原点约3.71cm |
| 最终结果 | 60.000s超时，距原点2.064cm、航向误差0.03737rad；未完成原2s连续稳定条件 |

**主导停止条件是终点角速度反复越过0.1rad/s。** 终点阶段全部稳定条件同时满足的采样比例为69.385%，最长连续区间仅0.250s，达不到2s。终点接触、位置、线速度、姿态分别有99.743%、100%、99.012%、100%的采样满足原条件；单个末状态看似静止不能替代连续稳定要求。没有确认新的实施错误，因此不重复同版运行；G0尚余1次上限未用，不转移到G1，也不放宽阈值或扫描参数。这是停止判据的定位，尚未隔离终点振荡的唯一动力学原因。

解析高度也没有消除几何重叠：保存的best预测中足球底部相对floor最低−30.543mm，实际轨迹最低−21.344mm；使用原0.022m足球半径作几何回放，未改真实接触和失败规则。全部best轨迹保留。0.02/0.10/0.20/0.35s时域按相同绝对时间对齐的平均机身位置误差为0.312/1.536/6.099/38.792mm；末端预测误差仍不能忽略。

物理60s耗时92.335s（另计视频回放/分析/上传）；20次初始化规划共0.4676s。后续3000次规划中1615次超过20ms，规划中位/P95为20.267/31.777ms；这是共享服务器同步仿真，不是NX实时验证。全程30001条500Hz控制状态、30000条真实物理反力记录、3001组best预测和20组启动预测均保留。视频为1801帧、30fps，覆盖实际0→60s失败全过程。

[运行索引](results/terrain_memory/simulation/tm-s1-go2-wb-startup-r6/index.json)、[任务结果](results/terrain_memory/simulation/tm-s1-go2-wb-startup-r6/episodes.csv)、[全程图](results/terrain_memory/simulation/tm-s1-go2-wb-startup-r6/figures/g0-01-mission.png)、[连续50Hz便览](results/terrain_memory/simulation/tm-s1-go2-wb-startup-r6/g0-01-continuous-50hz.csv)、[500Hz终点判据](results/terrain_memory/simulation/tm-s1-go2-wb-startup-r6/g0-01-terminal-criteria.csv)、[绝对时间预测误差](results/terrain_memory/simulation/tm-s1-go2-wb-startup-r6/g0-01-prediction-alignment.csv)与[全密度数组和完整视频](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-wb-startup-r6)。50Hz CSV仅作便览，完整500Hz时程在Release。构建和实验在服务器tmux执行，按用户最后授权覆盖此前前台限定；源码和紧凑结果进GitHub，大数组/视频由执行端直接发布并核对size/digest。

本轮证明启动修正生效并完成了路径往返，但**G0完整任务仍未通过，G1执行基础和PANN/PINN＋RL＋MPC联合基础版本仍未完成**。近期顺序继续是先建立稳定联合基础，再研究创新；标准PPO接入不以先证明创新为前提。本轮已结束，不自动开新轮。

### 第五轮实际做到哪里

审阅基点 [d9c04a9](https://github.com/kaiwen123-yang/MCP-RL/commit/d9c04a9850000916cc2d77ef741f3473c543d130)。**2次G0物理启动、0次完整任务，G1开发/正式均0；两次都未完成最初2s站立。** 另5次无plant推进的设置失败单列，不是5次行走实验。旧运行已经关闭，原配置/结果不改写。

| 启动 | 实际物理时长 | 能支持的结论 |
|---|---:|---|
| g0-01 | 0.178s | affine PD经XML序列化丢失，实际力矩成了 `60*q_target`，随后跌倒；不能据此否定预期作者控制器 |
| g0-02 | 0.040s | PD已正确生效，但首次目标与home关节角严重不匹配；第三次规划Ground射线fatal，真实执行提前中断，尚无稳定性结论 |

第二次初始thigh/calf为0.9/−1.8rad，首目标约0.0353/−0.83776rad，对应约−51.88/+57.73Nm；足端总竖直反力峰值910.25N，约为实际机器人重量155.06N的5.87倍。40ms最终状态向后约0.672m/s、向上约1.207m/s。**当时尚未跌倒，不能写成站稳，只是尚未观察到后续恢复或跌倒。** 数据见[连续状态](results/terrain_memory/simulation/tm-s1-go2-wb-baseline-20261026/continuous.csv)、[物理反力](results/terrain_memory/simulation/tm-s1-go2-wb-baseline-20261026/physical-forces.csv)和[g0-02结果](results/terrain_memory/simulation/tm-s1-go2-wb-baseline-20261026/g0-02.json)。

固定作者headless Init/Reset清零控制序列后才恢复home物理状态；[作者GUI](https://github.com/johnzhang3/mujoco_mpc/blob/e00c47a5adb9856af2e0f24231bb3a60d5be23c4/mjpc/app.cc)则使用home ctrl初始化策略。这是已确认的启动语义差异；第三次Ground错误的具体内部候选未记录，不能宣布唯一根因。

### 第六轮原计划与联合基础顺序

第六轮[实际配置](experiments/terrain_memory/s1-go2-wb-startup.json)、[作者补丁](experiments/terrain_memory/s1-go2-wb-startup-upstream.patch)和[计划第5.7/5.8节](docs/TERRAIN_MEMORY_PLAN.md)保留原边界与实际完成范围。第五轮[原索引](results/terrain_memory/simulation/tm-s1-go2-wb-baseline-20261026/index.json)和[Release](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-wb-baseline-evidence-20261026)保持冻结。第二轮B1刚地/KV8000各5/5仍是历史参照。

## S1第四轮已冻结：接口修复有效，动态执行仍未成立

审阅基点 [56fabe0](https://github.com/kaiwen123-yang/MCP-RL/commit/56fabe0024108fe933dd8207c3a97331b8fa6102)。**3次原完整任务启动，0完成，0丢记录，正式结果0；刚地正式、KV8000、TM02均未执行。** 实际实现均为 `96748c9d0d51041177892ef7081d9ac3e02a6e36`，执行指纹一致。

| seed | 结果 | 失败时刻/s | 最终机身x/m | 2s机身高度/m | 2s实际支撑足数 |
|---|---|---:|---:|---:|---:|
| 4 | 机身触地 | 2.7690 | 0.1398 | 0.2974 | 4 |
| 3 | 跌倒 | 2.1400 | −0.1632 | 0.2268 | 1 |
| 0 | 机身触地 | 5.7948 | 0.8946 | 0.3253 | 4 |

seed4越过旧1.344s空分支故障点，但seed3在2s起步时已经明显失稳，之后0.14s结束。进入outbound标签不是稳定站立完成证明，三次都未到3m转向。

原持续零请求力矩空分支、错误控制所有者、不可用足非零命令、命令法向/摩擦越界、非MPC周期修改保持力的问题，在已记录数据中均未再出现。力矩分项与请求相符。**这证明指定修复已进入实际路径；它没有证明该控制组合稳定。**

初始站立阶段seed4/3/0分别有29/65/27次支撑丢失事件。最大合力命令差224.600N、力矩命令差36.011Nm不是同一时刻的实测外力误差。seed4/0的初始实际反力采样均值接近体重，因此不能由命令不足直接断言全部失败只有一个原因；也不据此再开接触防抖/加载坡补丁。

254次同起点查询全部属于刚地，速度差为0。seed0的空间patch/事件记录不等于TM02加载或材料记忆实验。新增训练0、硬件0、NX未测。

[原索引](results/terrain_memory/simulation/tm-s1-contact-execution-20261026/index.json)、[全部启动结果](results/terrain_memory/simulation/tm-s1-contact-execution-20261026/episodes.csv)、[原执行量](results/terrain_memory/simulation/tm-s1-contact-execution-20261026/contact-execution.csv)、[本次阶段派生表](results/terrain_memory/simulation/tm-s1-contact-execution-20261026/execution-stage-summary.csv)、[派生脚本](scripts/analyze_s1_execution_stages.py)、[连续图](results/terrain_memory/simulation/tm-s1-contact-execution-20261026/figures/continuous-execution.png)及[原数组与三次完整失败视频](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-contact-execution-evidence-20261026)。

派生仅读取5个既有输入，无新增仿真/训练/测试，原数据和第四轮配置不改。5个Release资产size/digest与索引一致。新大产物后续从执行端直接发布，**GitHub/Release确认后结束，不在用户电脑再次下载、复制、镜像、压缩或归档。**

本次扩充[TM20源码与实施边界](literature/terrain_memory_sources.json)，新增TM22 Open-DiffLoco用于后续价值梯度/闭环决策的研究借鉴；不因此开启新训练或一轮全面文献复现。

## S1 第三轮已冻结：结果、具体缺口及科学边界

判断基点为 [f0a1632](https://github.com/kaiwen123-yang/MCP-RL/commit/f0a1632f445415be6c33a7eac31c3a2c974d1f4a)。**30条正式结果全部新增，8成功、22失败；实际新启动33次。** 6次开发中的前3次轨迹序列化失败、物理结局未知；后3次同版复用为正式结果。不能记为36次，缺失数据不臆造。

| 地面 | 第二轮冻结B1 | 第三轮共同修正B1 | T1 |
|---|---:|---:|---:|
| 刚地 | 5/5 | 3/5 | 0/5 |
| KV8000 | 5/5 | 5/5 | 0/5 |
| TM02 | 0/5 | 0/5 | 0/5 |

共同摆腿补项已实际实施，接触前查询也进入真实轨迹；但增加动力学项没有证明更好，旧B1不能被覆盖成“已修好基线”。T1没有任何双方成功配对，不能声称整程更快或节能，旧Phase B/T2完整任务未执行。

源码与原始NPZ共同确认：

- **初始失接触控制缺失。** T1 seed4在三种地面均于1.344s初始站立失败；FR/RL两足在0.160–1.344s连续1.184s的三关节命令为0。尚无材料加载或接近查询。未曾摆动的失接触腿没有进入运动分支，旧mode日志还错误保留support。
- **求解后分力改写姿态力矩。** seed4在0.112s原计划的力矩约[−0.041,−0.888,0.005]Nm，接口后变成[6.365,7.845,0.215]Nm，虽然合力未变。seed0还出现单足命令243.497N超过原149.175N上限，以及总法向命令161.587N被缩为59.418N，均发生在软区之前。这些是命令量，不能冒充真实测力。
- **局部接触较温和，任务仍失败。** T1仅3个TM02回合实际加载材料，已记录受载足竖直速度峰值比B1低；另外两个在材料前失败。所有TM02 B1/T1均无前足→后足同地块受载重访，不能称已解决整程超域或证明记忆价值。
- **查询差异需要看幅度。** 1036次查询只有KV出现16次数值动作差；最大速度差0.16558m/s，第二大约1e-6m/s，其余更小。不是16次有意义的独立长期决策。T1的KV跨腿重载前残余最大约1.02nm，目标TM02没有跨足证据。

[原运行索引](results/terrain_memory/simulation/tm-s1-contact-transition-20261025/index.json)、[逐回合结果](results/terrain_memory/simulation/tm-s1-contact-transition-20261025/episodes.csv)、[共同修正对照](results/terrain_memory/simulation/tm-s1-contact-transition-20261025/shared-correction-vs-frozen-B1.csv)、[本次派生表](results/terrain_memory/simulation/tm-s1-contact-transition-20261025/mechanism-summary.csv)、[派生脚本](scripts/analyze_s1_transition_mechanism.py)及[Release](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-contact-transition-evidence-20261025)。派生表仅读取47个既有文件，无新仿真/训练/测试，原结果不改。14个Release资产size/digest与索引一致。

新增TM19–21分别借鉴逆动力学一致性、实际执行输入纳入MPC预测、策略闭环响应；原文/PDF/作者代码与限制见[来源清单](literature/terrain_memory_sources.json)。这些已有机制用于避免执行错误，不包装为本项目创新，也不新增全文复现门槛。

## S1 第二轮历史结果：可用基线与尚未成立的科学证据

判断基点为[5af2c25](https://github.com/kaiwen123-yang/MCP-RL/commit/5af2c25c5223c7b74a473133fe97aed3d9962c34)。60条正式结果中20条旧B0复用、40条新增，36成功/24失败；实际新任务启动42次，含3次开发。不能把首轮14/85与本轮36/60直接当作同一实验的成功率提升。

| 地面 | B0 原C0 | B1 接触跟踪 | B2 0.24s | B3 2.5s |
|---|---:|---:|---:|---:|
| 刚地 | 5/5 | 5/5 | 4/5 | 5/5 |
| KV8000 | 4/5 | 5/5 | 3/5 | 5/5 |
| KV3500 | 0/5 | 0/5 | — | — |
| TM02 γ=0.025 | 0/5 | 0/5 | — | — |

B1把共同足底位移的面积积分逆本构接入实际WB力矩路径，并保留一次GRF前馈。它是已知材料状态下的有限腿部逆动力学基线，**不是完整STANCE c3WBC复现**。KV8000挽回一个终点站立超时；四个双方完成配对平均快1.239s，但滑移、姿态及速度误差均增加，不能说各指标全面改善。

B3相对B1在刚地/KV8000平均快0.2992/0.4272s，同样各5/5完成。B3也在刚地受益，不能把结果自动归为材料记忆。它克隆真实机器人/材料/控制状态，是昂贵已知模型参考；共同预测前缀接近机器精度证明执行对齐，不证明未知材料泛化。B3上层每次决策p50约20.7–22.6s，远超0.2s决策周期，NX整链路未测。

**新的任务适用性计算：**KV8000的零载恢复时间为400/8000=0.05s。B1/B2/B3共2390条跨腿材料格点事件中，最短卸载间隔0.7862s，实际重载前残余变形最大约1.25nm；这些格点行不是2390次独立任务。TM02 B1仅有两条FR→FL格点重访，前足→后足为0，尚未到转向。当前成功地面几乎已恢复，目标材料又未稳定通过，因此核心记忆假设仍未得到整机检验。

**第三轮当时启动的依据（历史记录）：**已核TM02 B1 seed0在5.540s名义落地时，FL足底仍离地15.449mm并向上运动；原生支撑力矩已接管，B1尚未激活。5.572s以约−1.3235m/s触地，5.573s超域，早于5.574s下一次低层更新。参考末端速度本来已接近零，需要检查实际摆腿动力学及接触切换，不能只改期望末速或扩训练。KV3500保留为非足部接触终止负例，不删除碰撞来提高完成率。

[运行索引](results/terrain_memory/simulation/tm-s1-control-repair-20261024/index.json)、[全部回合](results/terrain_memory/simulation/tm-s1-control-repair-20261024/episodes.csv)、[本次派生的任务支持表](results/terrain_memory/simulation/tm-s1-control-repair-20261024/task-support-summary.csv)、[派生脚本](scripts/analyze_s1_task_support.py)、[分项耗时](results/terrain_memory/simulation/tm-s1-control-repair-20261024/timing.csv)、[Release](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-control-repair-evidence-20261024)。派生表只计算既有数据，没有新增仿真；原配置和原始结果不改写。19个Release资产的size/digest已与索引一致。

新增论文只吸收TM17（2026-10-09修订）的动态触地/惯性归因和TM18（2025-09）的感知步态权衡；原文与PDF见[来源清单](literature/terrain_memory_sources.json)。不复制颗粒介质阈值到TM02，也不把慢落足称作新发明。

## 首轮 S1：85 回合固定预算完成，尚未建立整程控制收益

**85/85 回合已执行，14 个完成完整往返，71 个失败；新增神经网络/PPO 训练为 0，未启动硬件。** 任务为硬地站立后 x=0→3m、转向π、原路返回并稳定站立，全程保留世界地块状态。固定预算结束，R1、三轮 R2、S0 与本轮 S1 结果均保留；不为获得正结果继续扩量或改变材料。

| 地面 | C0 固定 2.4 Hz | C1 adaptive | C2 短时预测参考 | C3 较长时域预测参考 | C4 冻结 B ideal |
|---|---:|---:|---:|---:|---:|
| 原生刚地 | 5/5 | 2/5 | 0/5 | 0/5 | — |
| TM02 γ=0.025 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| KV k=3500 N/m | 0/5 | 0/5 | 0/5 | 0/5 | — |
| KV k=8000 N/m | 4/5 | 2/5 | 0/5 | 1/5 | — |

表中为完成数/5 个配对优化器种子，不是五种独立材料。C0 在刚地的成功整程平均为 **40.9864 s**，在 KV 8000 的四个成功回合平均为 **40.8985 s**。C3 在 KV 8000 的 seed 3 完成于 **39.424 s**，其余四个种子失败，不能由这个单例宣称方法优势。

71 个失败包括 **40 个无可行接触意图、14 个材料超域、16 个机体/非足部触地或跌倒、1 个任务超时**；材料 Newton 积分失败为 0。C2/C3 在刚地也不能稳定完成，因此本轮没有把整程瓶颈隔离为材料记忆问题。TM02 的实际加载回合最大分量应变率为 **18.746–99.558 s⁻¹**，高于冻结 R2 的 700 条参考历史中 **0.227082 s⁻¹** 的采样最大值。该总应变差分包含空间材料柱的接触激活/卸载跳变，不能直接当足端撞击速度或PANN必须扩训的证据；具体已核例见计划第5.2节。小应变超域记录不解释为真实土壤塌陷，KV 参数也不是 Go2 实测标定。

使用 Go2 实际质量 **15.206 kg**、原模型惯量/关节/初姿和力矩限制。名义前后足间距 **0.3868 m**，统一事件时域 **2.5 s**；物理步经实际错误修正后固定为 **0.2 ms**，低层控制 500 Hz、MPC 50 Hz、接触意图 5 Hz。软区实际压入与材料状态产生唯一环境反力，MPC GRF 经关节命令执行。全部控制参数在正式矩阵前固定；后续只补充冷启动计时字段，没有改变控制或材料参数。服务器上的实测计算量与超周期记录单列，**NX 实时性未测**。

可核验成果包括 [结果索引](results/terrain_memory/simulation/tm-s1-go2-contact-20261024/index.json)、[全部回合](results/terrain_memory/simulation/tm-s1-go2-contact-20261024/episodes.csv)、[五种子配对差值](results/terrain_memory/simulation/tm-s1-go2-contact-20261024/paired.csv)、[逐部分计时](results/terrain_memory/simulation/tm-s1-go2-contact-20261024/timing.csv)、[跨腿重访](results/terrain_memory/simulation/tm-s1-go2-contact-20261024/actual-cross-leg-revisits.csv)、[预测事件实际覆盖](results/terrain_memory/simulation/tm-s1-go2-contact-20261024/event-coverage.csv)和[全部大时程与连续视频](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-contact-evidence-20261024)。失败后的值不补零；较短的失败轨迹不作为更快或更省能的证据。接触功按 5000 Hz 物理记录积分，关节正机械功使用 500 Hz 记录，均不等于电池耗电。

首轮运行入口与原85组合属于冻结提交 [1bfcf24](https://github.com/kaiwen123-yang/MCP-RL/commit/1bfcf244bf5787cf0086f3f1846c19c42576f9d7)及[原配置](experiments/terrain_memory/s1-go2-contact.json)。历史默认参数不作为当前任务入口；下一轮只使用顶部指定的新配置，不以默认命令重跑旧矩阵。

代表视频保留 seed 0 的真实失败，同时给出注明编号的成功补充例。新数组和视频从运行端直接上传 Release，旧 B 权重引用原 R2 Release；不在用户电脑另做大产物下载或归档。当前证据不支持启动长期 RL 或实机 H。

## S0 已完成：相容接触与 oracle 决策价值

**2026-10-24 完成固定 S0：72 工况 × 12 当前动作 × 7 预测器，共 6,048 次完整双地块推演；新增训练、积分失败、物理约束失败均为零。** 其中参考任务 864 条、其他预测 5,184 条，合计 12,096 条地块历史。R1 与三轮 R2 保持冻结。

已实现 [run_terrain_memory_s0.py](scripts/run_terrain_memory_s0.py)，按[固定配置](experiments/terrain_memory/s0-contact-oracle.json)执行 `tm-s0-contact-oracle-20261023`。[结果索引](results/terrain_memory/simulation/tm-s0-contact-oracle-20261023/index.json)、[候选/Pareto CSV](results/terrain_memory/simulation/tm-s0-contact-oracle-20261023/candidates.csv)、[策略真实后果](results/terrain_memory/simulation/tm-s0-contact-oracle-20261023/policies.csv)、[跨历史动作移用](results/terrain_memory/simulation/tm-s0-contact-oracle-20261023/history-matched.csv)与 [Release 时程及动画](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s0-contact-oracle-evidence-20261023)保存全部成果。这是机架承载双足/双压头任务，尚不是自由四足步态。

面积 0.001 m²、厚度 0.05 m、50/100/150 N、γ=1/0.025 及全部运动/时间限制保持配置原值。**γ=0.025 是合成尺度**，以原模型等效应力 σ/γ 积分，旧权重及归一化未改。每块地面贯穿前史、跨足、返程与零载恢复；所有选中动作均在持续参考环境评价。全程最大参考压入为 **0.245067 mm**，最大足端水平/垂直速度为 **0.500000/0.093163 m/s**，最小飞行净空 **11 mm**。

以下为额外等待预算 **1.5 s** 下每尺度全部 36 工况的均值；其他预算、全部常数和逐工况失败字段均在 CSV 中。

| γ | 策略 | 剩余最大压入 / mm | 剩余完成时间 / s | 正向接触功 / J |
|---|---|---:|---:|---:|
| 1 | 完整 oracle | 0.00460684 | 5.40278 | 0.000764063 |
| 1 | 同真实状态 myopic | 0.00479447 | 5.25000 | 0.000759967 |
| 1 | 事后最佳单一常数 ID 6 | 0.00465025 | 5.25000 | 0.000766793 |
| 0.025 | 完整 oracle | 0.18903414 | 5.77778 | 0.031549552 |
| 0.025 | 同真实状态 myopic | 0.19660859 | 5.25000 | 0.031289343 |
| 0.025 | 事后最佳单一常数 ID 8 | 0.18975365 | 6.75000 | 0.031805065 |

**有有限的动作价值，作用量仍小。** 合成尺度完整 oracle 相比 myopic 平均/最大降低压入 **0.00757446/0.02824167 mm**，但平均多用 **0.52778 s**、增加 **0.000260209 J** 正向接触功。相比事后最佳常数，平均压入只改善 **0.00071951 mm**，同时节省 **0.97222 s** 和 **0.000255514 J**。事后常数每 γ/预算只选择一个跨 36 工况固定 ID，不是留出泛化。全部参考候选可行，因此没有完成率改善；原尺度相对强常数的平均收益仅 **0.00004341 mm**。

**响应误差不等于选择损失。** 合成尺度 B ideal 候选峰值预测 MAE 为 **0.00296086 mm**，三个 B noisy 为 **0.03606689–0.03620669 mm**；在 1.5 s 预算下，B ideal 与完整 oracle 只有一个近似平局动作不同，平均实际压入差 **0.000000888 mm**。三个 noisy 各有 9/36 个不同动作，平均/最大实际压入差均为 **0.00003884/0.00036763 mm**。这没有修复 noisy 材料辨识，也没有建立整机通行收益。S0 固定研究结束，不自动扩量、训练 RL 或启动硬件。

**补充分开了未来任务需求与踩踏历史的作用。** 对同一未来载荷/返程间隔，只在六种前史上选一个固定动作，得到不读取当前前史的任务条件对照。在γ=0.025、1.5s预算下，其平均峰值 **0.189048163mm**、时间 **6.5s**：比oracle仅多 **0.014027微米**压入，但多用 **0.722222s**。这比一个跨全部未来需求的常数更接近oracle，说明不能把所有动作变化都解释为历史价值；它仍是同网格事后描述，不是留出泛化。

新增[逐工况选择](results/terrain_memory/simulation/tm-s0-contact-oracle-20261023/task-conditioned-policy.csv)、[任务条件汇总](results/terrain_memory/simulation/tm-s0-contact-oracle-20261023/task-conditioned-summary.csv)和[分辨率敏感性](results/terrain_memory/simulation/tm-s0-contact-oracle-20261023/resolution-sensitivity.csv)由 [analyze_s0_task_value.py](scripts/analyze_s0_task_value.py)从已有完整候选直接计算，**新物理积分和训练均为0**，原协议与结果未改。把压入近似平局容差假设为10微米时，36/36工况都不再额外等待；1/10微米均不是已标定传感器或任务要求，只显示当前等待收益依赖目标分辨率。整机需要真实任务取舍。

完整时程与任务动画由运行端 `fy-5080` 直接上传 Release，旧权重引用原 Release 及摘要，不重复打包。动画显示实际 mm/N/s，并标明向下变形 **20 倍**、飞行高度 **1 倍**。运行端正常输出与接续记录保留；不在用户电脑另做大产物下载或归档。以下仅为S0历史独立复跑说明，不是下一轮命令；入口识别已保存候选，不重复积分：

```bash
work/envs/tm02/bin/python scripts/run_terrain_memory_s0.py --config experiments/terrain_memory/s0-contact-oracle.json --run tm-s0-contact-oracle-20261023 --source-output-root /path/to/existing/tm02-memory-study-20261023
```

## R2第三轮历史成果：三臂材料记忆适用性实验

**固定预算全部完成：50 个新模型、10 个复用模型、2,640 次已知历史双边界诊断，以及 600 个工况、700 条参考历史上的 8,400 次新预测；全部积分完成，失败为零。** 60 个模型均保留权重、优化器和 20,000 点损失；12 个主模型只按各单元 seed 0–4 的最终训练损失选择。R1 与 R2 前两轮保持冻结，本轮未启动 S、整机 RL 或实机。

[统一结果索引](results/terrain_memory/reproduction/material/tm02-memory-study-20261023/index.json)、[每家族每噪声完整表](results/terrain_memory/reproduction/material/tm02-memory-study-20261023/final/family-model-summary.csv)、[配对变化](results/terrain_memory/reproduction/material/tm02-memory-study-20261023/final/paired-arm-family-summary.csv)及[全部权重、数组和曲线归档](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm02-memory-study-evidence-20261023)提供实际命令、版本、seed、计算量、哈希和下载位置。图表包含 26 张汇总图、700 张新历史全曲线及 44 张已知诊断全曲线。

A 精确复用 100 条原随机历史；B 增加 900 条同生成器历史；C 使用 A 前 50 条与 50 条加载—自由恢复—探针历史。所有臂与噪声实现保留原条件的全部归一化。以下为 **ideal 全历史 RMSE 中位数**；三个噪声实现分别列在上述完整表中。

| 家族 | 单位 | A：随机 100 | B：随机 1,000 | C：混合 100 |
|---|---|---:|---:|---:|
| F1 原分布随机 | MPa | 0.04848 | 0.04081 | 0.06156 |
| F2 低幅变速 | MPa | 0.04236 | 0.03152 | 0.04693 |
| F3 应变松弛 | MPa | 0.02167 | 0.01833 | 0.02587 |
| F4 蠕变自由恢复 | strain | 6.849e-6 | 4.989e-6 | 8.282e-6 |
| F5 压剪循环 | strain | 9.698e-6 | 7.027e-6 | 1.201e-5 |
| F6 重访 | strain | 1.357e-5 | 9.804e-6 | 1.702e-5 |

**更多随机数据有效，但收益有边界。** B ideal 的 F4/F6 全部 100 个配对工况改善，F5 的五个前史簇均改善；F3 均值与最坏误差却略变差。B noisy 的 F4/F6 整程误差小幅下降，但 F2 的簇均值、F3 的均值与最坏误差恶化。不能用中位数改善概括全部工况。

**恢复激励没有修复含噪记忆辨识。** C noisy 在 F2/F3 有一致的平均 RMSE 收益，但 C ideal 六家族的平均 RMSE 均高于 A；C/noise2 的 F1/F4/F5/F6 退步。F6 恢复段 RMSE 中位数在 C 的三个噪声实现中均变差，恢复积分绝对误差中位数在 B/C 的三个噪声实现中均变差。全部 45 个 noisy 模型的零态体积恢复时间仍为 **0.602–0.639 s**，参考为 **1.200 s**；双边界重放已显示前向本构偏差，不能只归因于逆求解。

F6 参考有 **60 对 strict early、40 对近似平局、0 对 strict late**。全部主模型均为 60 early + 40 tie；always-early 的真实遗憾为零。预测平局按等概率选择计算，模型平均期望遗憾为 4.5071e-8 strain。这些排序不证明依历史改变最优动作的价值；含噪模型的动作响应幅值仍有显著误差。

材料候选限于已执行的固定 Table 1、小应变、已知 virgin 初态和持续状态历史：冻结 A ideal 与更大数据量的 B ideal 已供S0相容接触和oracle比较，冻结B ideal曾用于首轮S1有限替换；当前T1/T1R使用参考本构和真状态。未将离散网格推广为连续域保证；快速全幅随机、最大体积压缩保持与含噪恢复仍有明确误差。当前 noisy 模型保留为负面对照，接触几何、真实地面与机器人收益尚未验证。

用户要求提速后，B 的微批从 25 经 100 增至全部 1,000 条，实测每 250 次更新的耗时中位数为 **85.65 → 40.67 → 31.59 s**，约 **2.71 倍**提速；解算和曲线图使用 8 进程。检查点、Adam 状态和 20,000 次保留更新连续接续，切换的未落盘计算上界单独记录。B 仍使用更多训练计算，不能称等计算量比较。

已完成批次的复跑入口为 [run_tm02_memory_study.py](scripts/run_tm02_memory_study.py)，配置为 [r2-memory-study.json](experiments/tm02/r2-memory-study.json)，不用于重复启动。当前执行约定为 [AGENTS.md](AGENTS.md)，结论和后续任务见 [执行计划](docs/TERRAIN_MEMORY_PLAN.md)。停止 R2 自动扩训；S0也已完成，当前执行入口转入S1。

## R2第二轮历史成果与边界

第二轮实际完成 **50 个新模型、2,000 次新旧留出预测、50 次已知困难路径预测和 24 条新增重访**，积分失败为零。[材料结果索引](results/terrain_memory/reproduction/material/tm02-multihistory-20261023/index.json)、[重访结果索引](results/terrain_memory/transfer/material/tm02-multihistory-20261023/index.json)与[全部权重和数组归档](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm02-multihistory-evidence-20261023)提供命令、实际数组、源码版本、硬件、seed 和哈希。

100 条完整训练历史各 100 个增量；ideal/noisy 各 25 初始化，每模型 20,000 次完整数据集 Adam 更新。按最终训练损失选中 ideal seed0、noisy seed10。以下误差为真实 MPa；物理留出工况是 **20 条历史**，不是 2,000 个独立样本。

| 已执行比较 | ideal 旧 → 新 | noisy 旧 → 新 |
|---|---:|---:|
| 选中模型的 20 路径 RMSE 中位数 / MPa | 0.3785 → 0.0504 | 0.9132 → 0.5264 |
| RMSE 改善的留出路径 | 20/20 | 19/20 |
| 25 初始化的平均路径 RMSE 中位数 / MPa | 0.5373 → 0.0715 | 1.1064 → 0.5502 |
| 已知困难路径 RMSE / MPa | 2.5127 → 0.2127 | 3.5363 → 0.7465 |
| 12 条重访峰值压缩相对误差中位数 | 2.31% → 1.49% | 7.43% → **11.42%** |
| 12 条重访峰值压缩相对误差最大值 | 3.94% → 2.51% | 17.24% → **22.64%** |

新模型最差初始化的平均路径 RMSE 仍低于旧模型最好初始化，说明随机历史泛化改善不是只靠挑 seed。论文方法链条、原文效果边界与机器人收益分别判断：没有作者完整随机数组/原代码，不能称全文所有方法和数值全部精确复现。

**含噪重访退步必须保留。** 新 noisy 的峰值误差、完整应变 RMSE、重载轴向应变 ε11 MAE 在 12/12 条历史中均变差；恢复段轴向应变 ε11 MAE 为 10/12 变差。两组 5 s 恢复的预测候选差约为参考的 3.4%，旧模型约 12.1%。ideal 排序仍 6/6、noisy 仍 4/6 加两组平局；always-early 同样 6/6、零遗憾，尚不能证明依历史改变决策的价值。

### 第二轮提出的问题（第三轮已经检验）

对已保存势函数在零态的局部导数分析显示：第二轮 noisy 平衡体积模量约 501.28 MPa，已接近参考 500 MPa；但零外载小扰动的体积恢复时间约 **0.626 s**，参考 **1.200 s**。全部 25 noisy 初始化都在 0.604–0.635 s，偏量恢复时间同样偏短。这是局部动力学量，不是把非线性材料替换成常数时间模型；公式、来源和边界见计划 4.1。

第二轮随机历史没有专门的保持/自由恢复段；所有应力分量同时在 ±6 MPa 内的训练增量约占 5.37%。当时 25 初始化共用一套噪声、20 条留出历史，尚不足以覆盖长保持、低速、循环和尾部行为。这些问题已通过第三轮三套噪声与 600 工况的固定实验检验：更多数据及这次恢复激励都未修复 noisy 恢复，不能再次把这段历史分析当作待执行的扩量指令。

## R2第三轮固定协议（已执行）：三臂与六类完整工况

| 训练臂 | 数据 | 要回答的问题 |
|---|---|---|
| A | 精确复用第二轮 100 条随机历史 | 原数据量基线 |
| B | A 的 100 条 + 900 条同生成器随机历史 | 更多同类数据是否足够 |
| C | A 前 50 条 + 50 条加载—自由恢复—探针历史 | 同观测增量预算下，恢复激励是否更有价值 |

原材料、架构、损失、归一化与 20,000 次完整数据集更新固定。每臂五初始化，ideal 一组、noisy 三套独立噪声，共 **60 模型，复用 10，实际新训练 50**。B 每次更新使用 1,000 条历史，所以总训练数据/梯度工作量约为第二轮的 **4.6 倍**；不称等计算量比较，也不按此承诺墙钟时间。

每臂的 ideal 和三个噪声单元分别按最终训练 loss 选一个，共 **12 个主模型**进入下列新评价：

| 家族 | 物理工况 | 完整历史 |
|---|---:|---:|
| 原分布随机 | 100 | 100 |
| 低幅、变速 | 100 | 100 |
| 应变保持与松弛 | 100 | 100 |
| 应力蠕变与自由恢复 | 100 | 100 |
| 压剪循环 | 100 | 100 |
| 同前史重访候选 | 100 对 | 200 |
| 合计 | **600** | **700** |

主评价共 **8,400 条学习模型完整预测**，另有 700 条共享参考。训练前固定实际评价数组，训练和选模结束后再看新结果；旧 20 随机与 12 重访归为已知诊断，对全部 60 模型保留双边界重放与完整响应。600 固定工况不是 600 次 iid 抽样；模型次数、相邻时间点和两候选不算独立物理起点。

本轮已实现并执行下列入口；现有运行会接续真实产物，旧批次保持冻结：

```bash
work/envs/tm02/bin/python scripts/run_tm02_memory_study.py --config experiments/tm02/r2-memory-study.json --run tm02-memory-study-20261023
```

固定预算、每家族及每噪声实现的正负结果、全部曲线/权重/数组均已交付，停止自动扩量。S0及四轮S1固定预算均已结束，当前任务见顶部和计划第5.6节；不要求所有外推条件零误差，也不把材料预测叫作机器人收益。第三轮原计划和启动词保存在 Git 历史中，当前启动词见执行计划第 7 节。

## R2阶段的论文借鉴（历史）

- [TM03：STANCE](https://arxiv.org/html/1904.12306v2)：接触一致支撑控制；仅采用论文§VII-A仿真KV尺度k=3500/8000N/m、d=400Ns/m作为公开对照，不冒充Go2地面实测或长期记忆。
- [TM12：Load-bearing Assessment，2025](https://arxiv.org/html/2510.21369v1)：由实际机器人运动确定接触载荷需求，MPC/WBC落实；不另做塌陷地板/探测状态机复现。
- [TM13：RK-MPC，2026](https://arxiv.org/html/2604.04221v1)：保留名义刚体模型、有限增广与实际部署耗时；不另开Koopman研究或照搬实时频率。
- [TM14：LF2WB，2026](https://arxiv.org/html/2603.03138v1)：后续同信息的记忆RL对照和有界状态推理；几何记忆与材料状态不同，特权训练Critic也不能直接供实机MPC调用。

- [TM06：足端泥地反力，2026](https://arxiv.org/html/2604.12006v1)：借鉴面积—应力—反力与足端位移的相容关系，落实 S0 的 N/mm/J；不一并引入触变、吸力等机制。
- [TM08：可测反力/位移标定，2026](https://arxiv.org/html/2609.03645v1)：§4.2/附录 C 的隐式状态联解与伴随求导可用于后续部署一致训练；TM02 §4.1 已有积分训练，不能称新发明。当前先做S1整机任务，后续有真实控制误差瓶颈才开展这个单一对照，不建 FEM 系统。
- [TM09：预测误差与规划误差，2026](https://arxiv.org/html/2610.05550v1)：候选共同偏差、动作差及常数策略分别评价，不用单一 RMSE 代替决策证据。
- [TM10：历史本构与内变量可辨识性，2026](https://arxiv.org/html/2605.14179v1)：比较未来可观测响应，注意充分激励和理论假设，不追求隐状态坐标逐值一致。
- [TM11：软材料实验设计，2025](https://arxiv.org/html/2505.13283v1)：借鉴信息性加载设计，落实 A/B/C 比较；不引入 BOED/RBF 新支线。

来源、PDF 下载位置和有限阅读范围见 [资料清单](literature/terrain_memory_sources.json)。TM10/TM11 已服务第三轮，当前不追加复现。新核到 [foot-driven terrain manipulation，2026-07](https://arxiv.org/abs/2607.11855)这一近邻的原始摘要，提示宽泛“主动改变地面帮助行走”已有研究；未据全文排除重合。本项目仍需证明跨足/返程持续恢复状态如何改变当前决策，不能把三臂辨识实验写成机器人创新。

### 历史材料基线保留

第一轮 [方法结果](results/terrain_memory/reproduction/material/tm02-rnn-20261023/index.json)、[重访结果](results/terrain_memory/transfer/material/tm02-rnn-20261023/index.json)和[归档](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm02-r2-evidence-20261023)继续保存。第一轮 50 次自由预测只有一条独立路径；原结论不改写。前两轮原配置、权重、归一化、实际数组和结果目录不覆盖，旧复跑命令仅用于独立重现，不是当前执行任务。

## R1 冻结成果与边界

第三轮完成 **103 个新回合、零训练**；历史 205 + 101 回合保留原始来源。[第三轮结果索引](results/terrain_memory/reproduction/pympc/r1-closeout-20261022/index.json)提供四频率比较、机械正功、独占窗口计时、连续视频和原始数据；[总索引](results/terrain_memory/reproduction/pympc/index.json)保留全部方法与未复现部分。

60 秒完成数：固定 1.3 Hz **0/50**，2.0 Hz **8/50**，2.4 Hz **28/50**，adaptive **22/50**。固定 2.4 Hz 与 adaptive 均纳入后续强对照；未建立在线自适应更好的证据。

原图 4 / 表 I 的部分协议及比较尚未恢复，**不称完整论文复现成功**。已经获准推进独立 R2；不追凑 80%、不以让 acados 跌倒为目标，不重跑已完成批次。只有新的原始资料或证实的实现错误才定向重开。历史设置见[冻结计划](https://github.com/kaiwen123-yang/MCP-RL/blob/d7d18c00f84bca814356c269021a69878094e867/docs/TERRAIN_MEMORY_PLAN.md)和现有脚本。

## 三个待实验验证的贡献

| 贡献 | 实际要解决的问题 |
|---|---|
| 跨足材料记忆辨识 | 从加载、卸载、无载恢复和不同腿重访中，估计可预测后续支撑的材料状态 |
| 后续支撑约束当前决策 | MPC 将后足所需支撑转成当前加载、落点和时间的可行选择 |
| 远期重访价值与意图学习 | PPO Actor选择接触意图，Critic评价剩余任务/终端价值；与普通/循环PPO及充分事件规划比较 |

它们是待验证的机制假设，不能把 PINN、MPC、RL 三个名称本身当作三个创新点。R2学习固定参数、已知初态的合成材料；S0已有连续地块历史与相容力/位移映射，尚未解决自由机器人、真实未知状态或可部署辨识。S1第二轮的局部可用B1及昂贵B3参考保留，T1/T1R没有形成可用整程执行。当前先有界复用作者完整控制链，之后才决定材料状态接入；充分辨识的经典本构、同信息历史网络与充分规划仍是后续科学对照。Go2＋NX首版采用服务器训练、板端推理和状态估计，在线状态更新不等于边走边训练PPO。

## 成果与工作方式

实际源码、配置、紧凑数据和图表推送 GitHub；必要大数组和视频从运行端直接上传 Release，索引提供实际位置与摘要。S0 交完整接触任务数据和动画；S1/实机交连续机器人任务视频，不预建空结果。GitHub 发布后不在用户电脑另做归档，不全量回下载来重复校验，不重复打包旧权重；保留运行端正常输出和必要检查点。

禁止防御性编程、单元/mock/冒烟测试、无关回归、审查报告和状态流水。当前必须完成第5.6节的有限作者Go2执行器复用与完整刚地任务，新增训练为0；第四轮接触修补结束，旧矩阵不重启。实验结论由实际结果决定，主问题保持不变。



