# Codex：W44已完成；W45先验证完整策略接纳与原任务稳定性

## 当前事实与入口（2026-11-12）

先读 [当前主计划](docs/TERRAIN_MEMORY_PLAN.md)、[W44最终原件审查](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W44-final-review/README.md)、[W45稳定性计划](docs/S1_W45_STABILITY_PLAN.md)、[完整启动提示词](docs/S1_W45_START_PROMPT.md)。当前用户要求先解决稳定运行，尽可能双CPU/GPU并行、无后台固定墙钟完成时间、GitHub发布后不再电脑归档。本文件及用户最新要求覆盖历史启动/归档/自动清理规则。

**W44已完成，r34/W45仍是待实施计划。** 本次仅审查、复算和推送计划，没有登录D57、开始新仿真或学习。根r12运行配置与冻结r33保持已完成状态，不直接用旧W44配置开启下一轮。历史形成时的指令/快照保存在Git中；旧W43/W44启动文字不再描述当前状态。

- 结果基点 `69d5a6ff2aa9803fddc8ac3a78a906008ea526c6`。W44-A `job-20261110T143358Z-w44-a` / ep00428；B `job-20261110T143400Z-w44-b` / ep00417。各+512/2到14737/57、tail0、excluded145。
- 35个实际训练回合：32物理失败、3有效截断；另2次零物理步初始化失败。18条unavailable是两次attempt各9类未生成记录，不是18次实验失败。材料加载9/6回合，合法去返/往返0。
- 最终各自update57固定E30：A11.934s、B12.742s body_fall，均无实际材料加载，E60/E300/E600 blocked。前移已恢复，稳定任务未成立。
- 原件请求已交付13127项/167895058454字节，18条never-generated明确保留；服务器checkpoint/resume和未选视频保留。此次审查只定向取必要原件，不冒称全量156GiB重算。
- A call163/plan156(2.84s) alpha0仅约8.05e-15下降，nominal scale0→installed1；B call173/plan161(3.04s)仅约2.17e-12，scale.0133352→1，随后均有实际大反馈。候选到安装13类数组一致，不能说未评分K偷换。B在2.9282s已有hold内饱和，修接纳不保证唯一解决全部不稳。
- 当前Kδx重构、单侧native FD、原生successor、2ms hold/子步clip未显示上述简单实现错误；不重新添加已有PD/TV或无证据改真实物理。W43的驻留、lambda生命周期与W44的运动失稳要分开。
- G5按已有现场观察已自然完成30s/约19.37h、材料任务失败；冻结W40-B/update37/r30，与W44不同。最终完整原件尚未在当前Git取得，读取服务器已有continuation补齐，不重启G5。

长期主问题仍是同一世界地块跨足/返程恢复历史的因果收益。当前真实PANN→PPO cap→MPC→TV→PD链已运行；候选内完整材料历史、稳定合法往返、记忆因果收益及实机都未建立。W44恢复特征非零仅9/1024，最终评价全零；不能将刚地启动跌倒直接归因于PANN。

## W45控制与实验

1. 主机制只修完整策略接纳：当前epoch重评分完整基线；数值平手保留actions/times/reference/K/H/feedback scaling；alpha0不能仅凭名义浮点下降替换离轨反馈。comparison tolerance依据浮点/重算误差和成本尺度，原硬物理条件不放宽。
2. 同一规则覆盖nominal/BestRollout基线选择、r33 R33Consider/archive与继承r32 Iteration的descent、accepted_native_descent、停止/重试/正则状态。保留有真实可辨下降的正alpha反馈策略；不是全局关TV、K限幅或只修最终日志。
3. 四问题：A158/151@2.74、A163/156@2.84、B153/143@2.64、B173/161@3.04。保存的是输出plan；缺失私有历史时在同一重建上下文比较，不冒称精确历史replay。有限比较与最多24次同尺度偏差诊断只服务该修复，不扩参数网格或独立smoke/刚地准入链。
4. 实施r34 spec/patch/build/prepare/runtime/recording并正常构建，推代码/证据和真实新phase后启动。当前计划没有提前实现r34，不能改标签冒充已部署。
5. **A/B固定各自update57 Actor/归一化，W45新增PPO样本/更新为0。** 保持PANN因果观察与cap链。创建evaluation-only新任务，包装现有run_s1_go2_r12.py单次入口；其--job-dir分支依赖本phase新训练更新，不适用。合同wall_deadline_enabled=false且不传--deadline-unix，不能只在日志写null。各自原E30通过才同case E60/300/600，完整原任务/材料/稳定规则保留。失败按协议结束/blocked，不能换seed反复到成功。
6. 固定策略为各自 `outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w44-a/PANN_PPO_S5-seed1/checkpoint-014737.pt` 和w44-b/seed2同名文件。完整resume分别ep00428/ep00417，保留actor/critic/optimizer/norm/RNG/fork_identity/lineage。未来15249/59仅另立学习phase的参考，W45不自动扩样。
7. 保留原Go2/TM02 .02域、PD60/5、力矩、非足接触、跌倒、原路线与声明步态、r28 trunk/r27 C2/r26 packet十子步、冻结PANN/658D/2Hz cap/W38奖励。原truth不能为接近论文而放宽。E30不是完整往返；训练60秒采集也不是E60。

## 并行与现场

默认D57，SSH `autodl-d57`；CPU `/root/autodl-tmp/MCP-RL-w42`，GPU `/root/autodl-tmp/MCP-RL`；环境 `work/envs/d57` / `work/envs/w41-mjwarp`。25vCPU/120GB RAM/96GB GPU是历史总容量，现场先核当前cgroup/affinity、内存、GPU进程、磁盘、tmux/PID/job/clock及未提交修改。已有有效W45则普通接续，不重复新建、不宽前缀kill、不覆盖并发修改。

8. 默认CPU A/B各10workers，Torch/BLAS/OMP各1；GPU支线总host额度2CPU（含native确认），余约3核给主控/系统/发布。build-j6在双流重负载前完成。两条CPU流的FP64 JAX PANN与Actor也共卡，不把96GB当独享，不为占满显存扩大实验。
9. 第三任务是四个W44问题×32候选的完整350ms GPU近似proposal研究，CPU native完整策略裁决。device内1750子步、四问题224000 world-step；top4确认后计算余下native真值，计编译/上传/推进/回传/CPU确认与总耗时、排序/约束漏报。收益不足或明确技术失败就交付有限结果，不变成新G5或第三PPO流，不阻塞CPU原任务。
10. 可写upstream/build与GPU树隔离，prepare不走共享可写软链。结束且干净的CPU树可复用；环境、静态资产与完成父输出只读复用。保留其他有效任务。

## 后台与交付

11. **所有新后台deadline=null，无固定墙钟完成期限，无到点kill。** 按有限问题完成、原物理评价自然结束、明确技术失败或用户停止结束。物理30/60/300/600秒、有限求解预算、单次网络timeout保留，不无限重试。存活进程暂停恢复保留同job/clock；进程退出而无完整plant/材料/observer/MPC/任务状态时，仅Actor不能恢复物理中途，保留interrupted并新建独立attempt从原初态评价，不拼接或覆盖原段。本轮无新增训练tail。
12. 先推实现/spec/机制证据/真实phase，再启动并观察两条物理时间和记录实际增长。保留原高精度比较、alpha/各层scale、完整策略身份、raw与applied反馈、投影和5000Hz关键子步、首偏离/材料加载/合法路线。大矩阵按事件引用，历史原件不删。
13. 紧凑Git先发，请求原件增量Release后发。Git/index短写串行整合最新main，不强推、不覆盖并发数据。未生成文件标明never-generated，不补空壳。准确区别计划、构建、运行、任务资格和数据交付。
14. **GitHub核验后不在用户电脑额外下载、打包、镜像或归档；不自动删除服务器checkpoint/resume、原件、视频或上传副本。** 用户另行安排迁移/清理不属此任务。
15. 最新网络授权：D57上传仅服务器学术代理；用户禁止电脑VPN的SSH转发，旧失败后切电脑隧道规则废止，不自动切电脑/Clash/直连。复用认证，凭据不入日志或Git。

最终交接实际commit/build/spec、所选机制、job/PID/clock、固定父策略、每级评价、首异常是否消除及剩余问题、GPU价值与成本、Git/Release和有效续接点。原任务未通过时如实结束该有限轮次，不以构建或数值修复代替稳定材料往返。
