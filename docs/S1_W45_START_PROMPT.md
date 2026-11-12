# W45 完整 Codex 启动提示词

日期：2026-11-12。配套 [W45稳定性计划](S1_W45_STABILITY_PLAN.md) 与 [W44原件审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W44-final-review/README.md)。本启动提示词原文保留；2026-11-12执行端已完成r34实现/构建/有限证据和新phase声明。续接先读w45/phase.json及逐流状态，勿重复--declare或重启已运行进程。

以下全文可交给能够连接 D57 的 Codex，执行本轮控制修正、双 CPU 原任务评价和 GPU 独立支线。

---

请读取 `kaiwen123-yang/MCP-RL` 最新 main、`AGENTS.md`、`docs/TERRAIN_MEMORY_PLAN.md`、`docs/S1_W45_STABILITY_PLAN.md`，以及 `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W44-final-review/` 下的 README、cpu-report、solver-report、gpu-report、literature-mechanisms 和相应数值/来源文件。然后直接完成 W45 的实施、推送、真实运行和交付，不停在复述计划。开场简要说明后开始工作。

## 一、准确继承当前状态

W44/S5-r33 已完成计算与所请求已生成原件交付，基点 `69d5a6ff2aa9803fddc8ac3a78a906008ea526c6`。A/B 各 +512/2 到14737/update57，tail0、excluded145；35个实际训练回合中32个物理失败、3个有效截断。两次零物理步初始化失败另列；18条 unavailable 是两次attempt各9类未生成记录。最终固定E30分别11.934/12.742s跌倒，无实际材料加载、合法去返/往返全零，后续级别blocked。W43驻留、W44前移与任务成功是不同结论。

W44不能再用旧 `--new-phase` 重跑；G5已按现场记录自然完成30s、约19.37h、未通过材料任务，也不重启。G5完整最终原件在当前Git尚不齐，可读取服务器已有continuation结果补齐，缺失就如实记录，不为补证据重做旧实验。

W45先固定各自update57 Actor、归一化和原case，保持PANN观察与cap链正常，但**本轮PPO新增样本/更新目标为0**。这是控制稳定性隔离实验。完整继续学习状态保留，后续另立学习phase，当前不自动执行+512或2048扩样。

## 二、先修这一项具体机制

A call163/plan156（2.84s）alpha0只改善约8.05e−15，却从nominal反馈倍率0变为安装倍率1，2.852s出现大实际修正。B call173/plan161（3.04s）alpha0只改善约2.17e−12，却从.0133352变为1；3.044s已产生.499676rad反馈、.907618rad H投影修正，held目标偏accepted约.910716rad。两者名义轨迹几乎相同而K不同。

原件已确认候选与安装13类数组一致，Kδx可精确重构，native/FD successor相同，2ms hold和子步力矩裁切没有被证明漏做。**不要把它改写成未评分K偷换、坐标符号错误或简单漏裁切。** B在2.9282s已有hold内饱和，接纳修复不能预先保证全部稳定。

实施 `S5-r34` 的唯一主改变：

1. 当前epoch重新native评分完整incumbent/nominal基线；策略对象绑定actions/times/reference/K/H/feedback scaling及上下文。
2. 数值平手规则依据同输入重复评分、浮点累计量级和成本尺度在spec写清，不随意设绝对1e−6，不放宽原物理硬边界。
3. 平手保留完整基线；alpha0允许作为保留候选，不能仅凭名义浮点下降安装不同K/H或把反馈倍率重置为1。纯反馈更新需要独立离轨依据，本轮默认不启用该额外机制。正alpha有真实可分辨下降时照常允许完整反馈策略。
4. 规则同时覆盖nominal/BestRollout的基线选择、r33 `R33Consider`/archive和继承r32 `Iteration`中的descent、`accepted_native_descent`、停止/重试/正则状态；不要最终保留但内部仍虚报优化成功。no-step的surprise不靠极小分母包装成有效改善。
5. 保持native FD、原D、约束Riccati、原成本、r32 epoch重置及有限重试；所有新完整策略都用实际执行语义native评分。不要用全局关TV、K限幅、提高真实力矩或改材料域掩盖问题。

补全r34 spec/patch、build/prepare/runtime/recording/声明与发布接线，正常构建，不改写旧r33产物和冻结job。保留原Go2/TM02、.02应变域、PD60/5、原力矩/非足接触/跌倒/路线、声明步态、r28 trunk、r27 C2、r26 packet/十子步、658D观察、2Hz cap和W38奖励。不能同时更换horizon、动作空间、reward、接触模型和GPU后端。

## 三、四个现成问题作有限比较

用W44最终E30的 A call158/plan151（2.74s）、A call163/plan156（2.84s）、B call153/plan143（2.64s，scale0反例）、B call173/plan161（3.04s）。只取必要的已发布原件，避免再下载整个156GiB。

恢复可得state/time/userdata、task/gait、完整候选/reference/K/H和实际动作；这些是call输出，缺失私有历史就明确列出并建立同一重建快照。旧N/新规则各一次共同上下文比较，不把它叫历史精确replay，不重跑旧完整回合。沿现有有限求解预算，不扩参数网格。可按计划做最多24次偏差诊断：明确记录的实际−reference切空间误差向量及其相反向量，存源时刻、reference身份、缩放比例，正确处理四元数；它们检验离轨行为，不是新物理资格或未来信息注入。

保存原高精度成本、comparison tolerance、alpha、各级反馈倍率、完整策略身份及拒绝原因、实际施加修正、H投影和十子步力矩；记录真实新增rollout/FD/缓存/耗时。先推实现、spec、有限比较和真实新phase声明，再启动两条原任务评价。

## 四、两CPU直接验证原任务

固定策略路径：

```text
A: outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w44-a/PANN_PPO_S5-seed1/checkpoint-014737.pt
B: outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w44-b/PANN_PPO_S5-seed2/checkpoint-014737.pt
```

完整学习恢复点分别为同目录 `resume-state-ep00428.pt` 与 `resume-state-ep00417.pt`。保留actor/critic/optimizer/normalization/RNG/fork_identity/phase_lineage，不重新fork/reseed、不混两流权重。共享6545/update25祖先的两流不称从零独立随机对照。

创建evaluation-only调度/启动脚本与新W45 job/clock/目录。现有单次入口 `scripts/run_s1_go2_r12.py` 支持 `--attempt --evaluation-start-unix --checkpoint --level --seed --workers --contract`，按实际parser与完整合同调用；其 `--job-dir` 分支要求本phase训练结束和新PPO更新，不能直接套用。本轮合同设 `execution.background_job.wall_deadline_enabled=false` 并省略 `--deadline-unix`，不能只在日志写null。先验证checkpoint解析和固定评价身份，再同时启动A/B各10workers。各自固定Actor mean与原nominal case运行E30，同case前级通过才独立E60、E300、E600，按原完整判据。E30要满30s、穿过原带且无侧出（|body y|≤.6），穿越期间至少一个完整足迹在带内且该足Fz>0，并到达远端/进入转向；完整往返属于E60以及后续1/5/10圈资格。E300/E600保留原270/540s截止、目标圈数完成后的endpoint_position_violation_samples==0和final_stable_s≥2等条件，不把长测字段附加为E60新条件。

启动后确认两条物理时间与真实记录增长再交接。第一次不通过就保留对应阶段结果，按原协议阻断后级；不要换seed、偷偷改物理标准或重复到偶然成功。完整轮次结束时回答：已定位的接纳触发是否消除、是否获得合法去返/往返、剩余最早异常是什么。若仍失败，不自动叠第二套机制或继续大量PPO。

## 五、第三任务使用GPU做有限完整候选筛选

CPU主线独立运行，GPU支线不作为其启动条件。与CPU共用上述四个W44问题，每问题32候选槽，原有两基线身份+15组正负结构proposal，不加新的worker/幅值/horizon网格。

把32个独立候选在device内连续推进350ms：175段×10个.2ms子步，四问题共224000 world-step。预分配/一次上传/设备内推进/末次取回，消除旧G5逐子步host来回与49份重复名义world。若GPU未完整移植K/H/投影，明确近似proposal语义，CPU native完整策略仍是最终裁决。

先GPU排名，再CPU实际确认top4与必要基线并计时，随后完成其余候选建立32槽真值。报告top4召回、原约束漏报、lexicographic `(v,J)` regret、相对baseline改善与含CPU确认的端到端耗时；编译、真值成本和缓存另列。GPU失败/无收益就发布有限结果并自然结束，不续成新G5/E30或迁移整个MPC。

## 六、现场、资源和长期运行

现场默认 `ssh autodl-d57`，CPU `/root/autodl-tmp/MCP-RL-w42`、GPU `/root/autodl-tmp/MCP-RL`；环境复用 `work/envs/d57` 与 `work/envs/w41-mjwarp`。先精确读取tmux/PID/job/clock、未提交修改、父文件、CPU affinity/cgroup、GPU进程与实际内存/磁盘余量。有正在运行的有效W45则接续，不重复创建；不用宽前缀kill或覆盖研究修改。

默认10+10 CPU workers，GPU第三任务总host额度2CPU，约3CPU留给主控/系统/发布；Torch/BLAS/OMP各1。构建-j6先完成。PANN的FP64 JAX和Actor仍共用GPU，96GB总容量不是支线独享；保留数值和共享设置，必要时将四问题分片，不为占满显存扩大无意义负载。第三任务CPU真值计算也受2CPU额度约束。

可写upstream/build与GPU树隔离；prepare不能通过软链改其他任务。已结束且干净CPU树可复用，静态资产、环境、父输出只读复用。无需为改名复制或归档整个仓库。

**所有新后台deadline=null，没有固定墙钟完成时间、到点kill或周期性总超时。** 按有限问题完成、原物理评价自然结束、明确技术失败或用户停止结束。原物理30/60/300/600秒、求解器有限预算与单次网络timeout保留，不无限重启。原进程仍存活的暂停可对精确PID/session恢复，保持原job/clock。进程退出后仅Actor checkpoint不能恢复中途plant/材料/observer/MPC/任务私有状态；保留interrupted，若无完整运行态恢复能力则在同job下新独立attempt从原初态评价，不覆盖或拼接旧段，不称精确续接。本轮没有新增训练tail，历史训练恢复点仍保留。

## 七、交付

紧凑代码/spec/评价/逐项数值先Git，新增请求原件后台增量Release。Git/index写入短时串行，整合最新main，不强推，不覆盖其他流更新。保存完整轨迹、planner calls、反馈原始/施加值、策略身份、关键候选/参考/K/H、5000Hz事件窗、原材料/路线与最终评价；大矩阵按事件保存并引用，避免每动作行冗余复制。

GitHub核验后不在用户电脑再下载、打包、镜像或归档；服务器研究原件、checkpoint、resume、视频保留，不自动清理。D57上传仅用已授权学术代理，禁止用户电脑VPN的SSH转发，不自动切电脑、Clash或直连。认证复用，凭据不入日志/Git。

最终交付实际源码/产物身份、机制证据、job/PID/clock、固定策略与每级任务结果、GPU实际价值和成本、失败/未生成项、Git/Release及有效续接点。本轮学习计数仍14737/57；只在真实通过原任务后报告对应能力，不把构建、进程存活、数值修复或短窗改善称为完整任务成功。
