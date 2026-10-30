# Codex：PANN同版续训256→512，单窗口最多4小时

## 默认计算服务器（2026-10-30 起）

用户已指定后续训练、仿真、编译、渲染和其他项目计算统一使用 AutoDL 西北 B 区 D57，SSH 别名 `autodl-d57`。配置为 RTX PRO 6000 96GB ×1、25 vCPU、120GB 内存；项目目录 `/root/autodl-tmp/MCP-RL`，环境 `work/envs/d57`，进入命令 `source scripts/autodl_d57_env.sh`。具体版本与复用来源见 [计算环境配置](experiments/terrain_memory/compute-environment.json)。旧 5080 服务器只保留历史数据与复用来源，不再默认在其上启动新计算。

联网先 `source scripts/autodl_network.sh academic` 使用官方学术加速；同一联网操作连续三次网络失败后，直接在用户电脑启动 `scripts/start_d57_vpn_tunnel.ps1 -LocalProxyPort <电脑当前Clash端口>`，服务器切到 `source scripts/autodl_network.sh pc`。参数或代码错误先修正，不计作网络失败。用户已授权复用当前节点安装服务器独立 Clash；需脱离电脑联网时可显式选择 `source scripts/autodl_network.sh clash`。节点凭据保存在服务器私有目录，密码、密钥、订阅地址、外部代理端点不得提交 Git。D57已经完成PANN首次真实采集、更新和发布；当前续训范围以下述规则与主计划为准。

当前任务以[主计划顶部6行](docs/TERRAIN_MEMORY_PLAN.md)为准。W1、RAW首次256/1、D57 PANN首次256/1均已完成并发布。本窗口只接续同版PANN；沿用R12 run和Release，不清零进度。

1. 开始只读本文件、主计划当前交接/本窗任务、直接相关入口与实际checkpoint位置；开头最多5行后直接执行。复用已有D57环境、SSH与联网流程，不重建迁移/接线方案。
2. 唯一目标：PANN_PPO_S5 seed0由累计256条/1次更新续到累计512条/2次更新，全部新增产物发布后停止。每个rollout仍为256，下一批是新采样；不重采首批，不重复原优化器更新，不启动RAW配对批次。
3. 显式--resume本组resume-state-ep00012.pt；若本任务已有后续部分进度，继续最新本组checkpoint与累计目标。恢复actor/critic/optimizer/RNG/归一化，禁止重新初始化或加载RAW样本/权重。无完整环境snapshot时开新的真实会话，正常记录边界；不做恢复演习。
4. 4h从首次活动起算，阅读、修改、运行、渲染、上传全部计入，通常3.5h收尾。传本窗口新的start/deadline；后台进程也受同一截止约束。未完保存真实进度/有效尾buffer/待传项，下个用户启动的窗口续接。
5. 保持D57/r21/20MPC workers、Torch2、OMP/OpenBLAS/MKL各1、原PPO/奖励/归一化/物理/材料/阈值、nominal60s会话及课程关闭。保持2Hz cap[.18,.25]、冻结B ideal、独立因果观察与原JAX x64/预分配设置。本窗不改controller/reward/worker，不提前实现并行或GPU查询重写。
6. 所有已完成首批会话都在更新前；本批直接采集第一次更新后的真实行为。正常物理失败作为真实terminal保留并开始下一自然会话；没有12次实验上限。真实terminal优先，外部截断在入buffer/GAE前处理，普通更新不reset，保留正常loss/checkpoint。
7. 本轮物理卡点是去程入/穿材料时的冲击、前俯与支撑，不是已证实的返程或侧出。用现有/新增正常轨迹观察同类失败、路线进度与PANN特征；尚无唯一具体代码错误，不凭极值拍脑袋改duty/权重或放宽.02材料域。
8. 用本次invocation新增样本/更新和invocation_wall_s报告吞吐，另报整窗时间。恢复后的旧collection.wall_s包含跨窗停机，不能拿累计512除它。普通训练记录即可，不新增诊断平台、独立benchmark或资产认证。
9. 不把E30/E60全通过设为继续本次训练的前置门槛。正式E30→E60→E300→E600仍按同版case资格执行，训练轨迹不冒充正式评价；本窗不自动追加评价或控制修复。
10. 禁止无关回归、单元/mock/冒烟/导入/形状测试套件、恢复演习、全量哈希认证、通用fallback、重复全仓审查与反复计划文档。只修本次实际启动/运行暴露的具体技术问题；已修libegl1/imageio不重装验证。
11. 新源码/配置/必要环境变化/小记录推Git；全部实际新视频、原始轨迹、物理/预测/材料数组、rollout与checkpoint从D57直传原Release。视频从保存轨迹渲染，未变资产引用旧链接；正常看一次清单，未完标pending。不在用户电脑另做下载、镜像、复制、压缩或归档。
12. 结束只更新主计划顶部6行及正常结果。分别报告本窗新增与累计训练进度、更新后真实表现、耗时和发布状态。两批单seed小样本变化不称算法收益；后续需要控制修复时另开一个明确窗口，不自动串成长campaign。
