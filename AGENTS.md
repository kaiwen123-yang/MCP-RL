# Codex：AutoDL实际推进，单窗口最多4小时

当前任务以[主计划顶部6行](docs/TERRAIN_MEMORY_PLAN.md)为准。用户已改用现有AutoDL实例：25vCPU、RTX PRO6000 96GB、120GB内存。此要求覆盖旧5080路径、固定5workers、暂不租卡及旧W123/48h多流campaign。沿用R12 run和Release，不清零进度。

1. 开始只读本文件、主计划当前交接、最新SSH/运行环境记录和直接相关入口；开头最多5行后立即做。SSH记录中的实际连接/目录优先，不猜地址、不改写其他Codex正在补的SSH内容。
2. 当前唯一目标是必要AutoDL迁移后直接完成PANN_PPO_S5 seed0累计256真实转移、首次PPO更新和完整发布，随后停止。W1/RAW256条/1次更新已完成，不重跑或补造原loss。已有实际PANN进度则只接自己的checkpoint，禁止用RAW权重/样本或重复已完成目标。
3. 4h从本窗口首次活动起算，包含阅读、依赖/构建、修改、训练、渲染和上传；通常3.5h收尾，上传大则更早。显式传本窗口start/deadline；任何后台任务也受同一截止约束。未完交真实进度，下个用户启动的窗口续接，不自动连跑。
4. 首窗单流20MPC workers、Torch2、OMP/OpenBLAS/MKL各1。它是25vCPU软预算和未实测起点，保留主循环/JAX/RPC/I/O余量；不得每个进程各开25线程。两处MPC worker统一读契约，不改预测节点、候选数、迭代数、物理时钟来换速度。
5. wrapper按脚本位置找仓库、使用当前已激活的项目Python；匹配的MPC Python客户端由项目环境提供。冻结PANN权重读resources.pann_weights_path。优先Python3.11/Tianshou0.5.1配Torch2.8/cu128；已真实跑通的现有环境直接复用，不重建环境矩阵或整套旧工作站。
6. 保持r21源码、nominal、60s连续会话、课程关闭、PPO/奖励/归一化及2Hz cap[.18,.25]；冻结B ideal、独立因果观察、不读真q/e。保留JAX预分配关闭与x64。复用现有终止/截断、零时长过滤、CUDA、自然会话新Agent与正常保存机制。
7. 用正式采样的正常episode/rollout/loss/计时判断资源收益与PANN特征。20workers未实测，不承诺加速倍数；零特征如实记录。GPU利用率/显存占用不是成果，不为占满资源扩大网络、修改batch或制造额外实验。
8. 后续确有两个独立研究任务时，可在新窗口用两流各8workers；各自策略/优化器/状态/checkpoint，并先把现有共享deadline改为流内路径。现有index锁和Agent自动端口复用，不提前写通用调度/vector collector/锁框架；本窗不启动该安排。
9. 只修实际启动/训练暴露的具体错误。禁止无关回归、单元/mock/冒烟/导入/形状测试、独立benchmark、恢复演习、全量哈希认证、通用fallback、反复计划/审查文档；已有自动记录可保留，不新增认证阶段。
10. 全部新增源码、环境、小记录推Git；全程视频、原始轨迹、实际数组、rollout/checkpoint从AutoDL直传原Release。视频用已有轨迹渲染；未变资产引用旧链接。正常看一次上传清单，未完标pending；不在用户电脑另做下载、镜像、复制或归档。
11. 结束只更新主计划顶部6行及正常结果，分别报告训练/发布状态。沿用E30→E60→E300→E600同版资格；跨主机/worker旧RAW与新PANN首次更新只作工程证据，不宣称匹配对照、收敛或算法收益。
