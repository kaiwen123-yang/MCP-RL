# Codex：支撑联调＋tmux训练、自动渲染与发布

## 默认计算服务器（2026-10-30 起）

用户已指定后续训练、仿真、编译、渲染和其他项目计算统一使用 AutoDL 西北 B 区 D57，SSH 别名 `autodl-d57`。配置为 RTX PRO 6000 96GB ×1、25 vCPU、120GB 内存；项目目录 `/root/autodl-tmp/MCP-RL`，环境 `work/envs/d57`，进入命令 `source scripts/autodl_d57_env.sh`。具体版本与复用来源见 [计算环境配置](experiments/terrain_memory/compute-environment.json)。旧 5080 服务器只保留历史数据与复用来源，不再默认在其上启动新计算。

联网先 `source scripts/autodl_network.sh academic` 使用官方学术加速；同一联网操作连续三次网络失败后，直接在用户电脑启动 `scripts/start_d57_vpn_tunnel.ps1 -LocalProxyPort <电脑当前Clash端口>`，服务器切到 `source scripts/autodl_network.sh pc`。参数或代码错误先修正，不计作网络失败。用户已授权复用当前节点安装服务器独立 Clash；需脱离电脑联网时可显式选择 `source scripts/autodl_network.sh clash`。节点凭据保存在服务器私有目录，密码、密钥、订阅地址、外部代理端点不得提交 Git。D57已经完成PANN首次真实采集、更新和发布；当前联调范围以下述规则与主计划为准。

后台自动发布使用服务器独立的academic/clash/direct网络与持久gh登录；交接前完成实际访问准备，凭据不进入仓库、job记录或日志。电脑隧道不能作为无人值守发布的依赖。

当前任务以[主计划顶部6行与第4节](docs/TERRAIN_MEMORY_PLAN.md)为准：W34完整PANN→PPO→MPC支撑联调与连续学习。继续原R12 run/Release，保留所有实际进度。

1. 开始只读本文件、主计划当前交接/第4节及直接相关源码。最多5行说明后执行，不新增全仓审查或计划阶段。先查看已有tmux作业与日志；正在运行就读取进度，不重复启动同一训练流。
2. 若支撑修正尚未完成，按已有ep10证据核对“MPC选中预测允许躯干触地后恢复、真实首次非足接触即终止”的错配，做一项有依据的修正。修正后直接联合采样，不先加独立E30门槛；已完成的修正不重做。
3. **四小时限制只约束Codex交互工作。用户已明确授权tmux训练可持续十小时或更长，退出对话/SSH不终止已启动作业。** 使用scripts/start_s1_go2_r12_tmux.py、最新本组--resume、--hours、--postprocess-hours和--network-mode；默认16h训练预算，达到目标提前停止，再自动渲染/发布，后处理从训练退出起默认另计4h。后台deadline与本次交互deadline分开，不能再套timeout4h或在交接时kill。
4. 启动后确认进入真实物理采样、样本与仿真时间持续增长、普通日志写入且无即时技术错误即可交接。无需等满256条或一次更新，无需新增稳定性测试、逐步截图、常驻监控或轮询agent。离开时记录session、job目录、目标/截止、当前计数和checkpoint位置。
5. 阶段目标保持同一记录控制版新增1024条/4次PPO更新，当前256/1起点对应累计1280/5。rollout仍256，更新后继续；自然物理失败保留后进入下一会话，没有12次上限。真实技术中断停止并留下日志，不自动无限重启。
6. 恢复PANN自身actor/critic/optimizer/RNG/归一化，保留历史。修正后记录active_controller_build和空buffer版本边界，不混旧控制器未消费buffer。后台运行期间保持该配置/控制器/执行器不变，不并行修改同一训练流；下次实际停止后才继续改动。
7. 保持D57单流20MPC workers、Torch2、OMP/OpenBLAS/MKL各1，原PPO奖励/归一化、2Hz cap[.18,.25]、冻结因果PANN、材料/PD/力矩/失败条件。正式评价仍E30→E60→E300→E600，训练不冒充资格。
8. 复用transitions/collection/learning/episode/progress/stopped及正常checkpoint。用实际计数和属于本次job的stopped最后stage/reason判定，不能仅凭tmux消失或exit0说目标完成。原记录足够，不做新监控平台、无关测试、恢复演习、哈希认证、通用fallback或benchmark。
9. **训练结束自动渲染并推送，不等下一次对话启动发布。** 前台先提交真实源码/配置，后台finish_s1_go2_r12_job.py渲染本job所有可用轨迹、版本化可变记录、上传全部新增大产物到原Release，并提交/推送本job小结果、索引与6行交接。训练技术失败也收尾已有负例；零步/不完整轨迹写明不能渲染，不重跑物理。闭合日志后上传，不能一边写一边发布同名文件。
10. Git只提交本任务范围并普通push，不force、不覆盖旧Release证据。渲染/上传/推送失败记录具体错误与pending，尽量发布其余结果；修复后只续该job后处理，不重训或重复上传已完成资产。状态入口为原results根目录下jobs/<job名>.json，训练达标、视频、Release与Git交付分开。不在用户电脑另做下载、镜像、复制、压缩或归档。
11. 本次交互可以在启动正常后提前结束，不必耗满4h。结束更新顶部6行与正常记录；下次先查看后台实际状态，再继续分析；仅有pending时续后处理。服务器关机或重启不由tmux保持，按已有checkpoint恢复即可，不新增守护重启系统。
