# Codex：S1同r24继续学习，末端评价与自动发布

## 默认执行位置

使用AutoDL西北B区D57，SSH别名autodl-d57；RTX PRO6000 96GB×1、25vCPU、120GB内存。项目/root/autodl-tmp/MCP-RL，环境work/envs/d57，激活source scripts/autodl_d57_env.sh。[实际环境](experiments/terrain_memory/compute-environment.json)已完成r24真实训练，旧5080仅是历史来源。

后台联网复用服务器独立clash（W34实际使用）、academic或direct与持久gh登录/Git身份；凭据不进Git、job或日志，不依赖用户电脑代理常开。不要重建已工作的环境或重新迁移。

当前执行以[主计划顶部6行和第4节](docs/TERRAIN_MEMORY_PLAN.md)及[契约](experiments/terrain_memory/s1-go2-local-readiness-r12.json)为准：W35_SAME_R24_CONTINUE_AND_EVAL。

1. 开始最多5行说明后直接执行；只读当前交接/第4节及直接相关源码，不再全仓审查、重复计划。先查实时tmux与最新真实checkpoint；同stream已在运行就读进度交接，不重复启动或改deadline。
2. W34已完成r24新增1024条/4更新，PANN累计1280/5、空tail；30真实会话有3次合规贯穿，0完整往返。r24支撑残差与MuJoCo构建修复已执行，直接复用，不再强制一个新补丁或E30准入。实际新错误按具体原因修。
3. 从本组最新checkpoint继续；已知resume-state-ep00044.pt。目标新2048条/8更新、累计3328/13；r24合计3072/12是观察点，不是12次物理启动限制或总训练上限。每256条更新后继续完整闭环。
4. 恢复Actor/Critic/optimizer/RNG/归一化/计数；不混RAW或清空历史。没有完整世界快照就从新物理会话续学并记录。保持r24、冻结因果PANN、原2Hz cap[.18,.25]、奖励/物理/材料/力矩/失败定义；后台不修改执行器或并行改同stream。
5. 使用单流20MPCworkers、Torch2、OMP/OpenBLAS/MKL各1。一次查看实际可用磁盘即可；本段预计新增约14GB级产物，不能静默删历史、买新盘或归档到用户电脑。不做worker/GPU benchmark或为了显存利用率扩大网络。
6. 启动前只做主计划4.3规定的必要评价接线：扩展现有R12入口到当前build/学习checkpoint/level及独立deadline，修前序resolved根目录使用当前run；复用PolicyStream(training=False)、原确定性均值动作/evaluator/qualification。当前代码不会仅因新增契约字段就自动评价，不能遗漏接线或宣称已实施。
7. 同一tmux pipeline按训练→末端固定策略E30→通过则E60→渲染/发布执行。评价固定最后实际完成更新的checkpoint与归一化、nominal/seed0/20workers，独立attempt/目录，不写训练状态或PPO更新。未达训练目标但有有效checkpoint可评价其实际末次更新；技术问题阻塞时保留原因并继续发布。E30失败只阻塞本case的E60，不重训补分，不拿训练前缀当正式资格。
8. **4h仅限制Codex交互。** 本阶段后台训练28h、随后评价4h、再渲染发布6h，各自独立计时；达到目标提前收尾。启动器使用--hours 28 --postprocess-hours 6和实际网络，评价预算按契约接入。确认真实样本/仿真时间增长和日志正常即可退出；不等一次更新，不逐步盯守，不退出时kill后台作业。
9. 普通物理失败保留后进入下一自然会话；实际技术中断保留日志，不加无限重启。复用普通collection/learning/transitions/episode/stopped/checkpoint；用本job计数与原因判断，不能凭exit0或旧stopped判达标。阶段结尾同时看合规贯穿、侧出、返程、往返、失败分布与固定策略评价，不能只看reward/loss。
10. 自动渲染全部训练/评价的保存轨迹，视频/原始轨迹/大数组/rollout/checkpoint/环境记录从D57直传。新job未分配非空资产自动进入最多900项的partNN Release；已有资产保留实际目的地并复用。评价attempt也必须加入同job manifest，不能只扫描train前缀。失败/部分结果同样发布，缺视频不重跑物理。
11. 源码/配置在启动前提交，后台提交本任务小结果、index及6行交接并普通push；不force、不覆盖旧Release。实际上传/推送失败留pending，之后仅续后处理。不全量hash、不回下载复验、不在用户电脑另做下载/镜像/复制/压缩/归档。
12. 不新增单元/mock/冒烟/导入/形状测试套件、恢复演习、通用fallback、无关回归、新监控平台或反复报告。结束仅更新当前6行与正常结果，注明训练/评价/发布分别是否完成。E30→E60→E300→E600规则保留；本阶段只做末端nominal E30/条件E60，后续按实际证据继续。
