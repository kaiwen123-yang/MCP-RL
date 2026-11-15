# W52 Codex 启动提示词

下面正文可直接复制给具有 D57 访问权限的 Codex。W51 已结束，W52 计划发布时尚未实现/运行。

```text
继续 kaiwen123-yang/MCP-RL 的 W52。先读取最新 main 的 AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md、docs/S1_W52_STABILITY_PLAN.md，以及 results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-review/{README.md,handoff.json,control/review.md,results/review.md,video-review.md}。按 W52 计划实际实施、运行、分析并发布，不只给建议或再问启动权限。

已完成 W51 不要重开：D0/r38 5.814s跌倒；命令修正后 D1/r39 8s静站、最终x=-.012449m、后段四足100%受载；首个D1因命令槽未填技术无效。原E30启动0，没选出正常行走版本。301份IK已证实绝对C2减去当前高度/速度的接续公式会取消抬脚，650个当前swing足×replan快照中，360个的已保存剩余目标节点全部不高于地面，保存节点最深-3.5705mm。这是已保存节点统计，不是连续曲线极值，也不是独立失败次数。

先核现场具体job/tmux/PID/物理时钟和资源，已有同身份有效W52就续接，不重复启动、不宽前缀kill、不动其他项目。继承463e91b的原mission世界命令写入修复，不启用额外twist成本；真实spec/patch/prepare/runtime/native/service/server/recorder必须接通，先完成必要构建。

主实验D2仅修摆动几何（拟r40）：删除旧相位误差相减式，从真实足位置/速度直接接固定原apex和touchdown，按计划做分段五次曲线及解析极值核对。保留2Hz、duty.45、5mm、绝对相位、世界锚点、落点。高度以原冻结世界touchdown高度为固定基准，apex=z_touchdown+.005；世界足底间隙另按当前epoch原生局部表面核对，不把位置变化的地表叠加后冒称同一C2曲线，不把平地z=0推到整个材料区。apex不能每20ms推迟；apex后仅作可行下降，仍落地的late swing明确S不可用并回正常B搜索，不改真实速度/接触标记，不clip高度冒充C2。零时长事件按原身份处理，不添加浮点v>=0硬门。既有301份快照只作一次无物理的数学核验，不重开微物理准入。暂不改PD/支撑偏置/权重/K限幅/horizon/PPO。候选仍经原native rollout与原risk/J严格比较，不强制装差S。

现场有两组10核余量时并行次级D3（拟r41）：相同几何修复，S有效时B/S各一个独立first-valid backward与原9alpha，冻结同一epoch输入，分别保管k/K/H/regularization/provenance，再按原比较器统一选；S无效仅单B。按计划只提交所选来源的合法regularization，保留B/nominal时取B分支，无有效状态回入口值，最终风险调整仅一次；未选分支不写全局。全局alpha0为缓存完整B，但S族正alpha趋零对应S，expected/ratio保持null。D3每epoch通常最多30full rollout；几何不可用未rollout S时单B最多20，已rollout但S原生无效时最多21；D2最多21。分开计FD与预算，不能冒称等算力对照。D2是主方案，D3不是新的必须通过门槛，也不是稳定保险。资源不足先D2，不等D3空闲。

D2/D3各一次原evaluation t=0→D8，用同W44 A seed1 checkpoint-014737/归一化/完整初态，保留原2s Stand、正常plant/独立native/RPC/query/hold/因果observer、原物理提前终止。D8 qualification_usable=false。记录完整实际时段视频、足力/间隙/支撑交换/前进、身体和非足承重、clip/QP/真实PD、policy age与reference/K/H来源、S几何与native差异。只做本改动必要核验，不新增全仓hash/mock/shape/import/smoke或重复W49/W50/F1/L/F2。

D2按原规则自然结束其一次D8、读完整实际轨迹且满观察数据支持持续正常支撑推进后，才优先冻结简单D2；原物理/技术提前终止保留原结果，不用早停片段代替满观察。不要求D3也成功，也不凭尚在运行的前几秒提前选定。可在释放的10核先开原A E30，另一组释放后开B，任何时刻最多两个重native任务；冻结后不再换版本。若仅D3有清楚实际改善则读完结果后选同一D3。两者都不走就交真实负结果，不临时加权/抬高/换seed追成功。选定后原A seed1、B seed2各一次E30，逐case原E30→E60→E300→E600；D8不拼接、不算资格，失败仅blocked本case后级。W44全学习状态保留，新PPO/donor0。

第三线C不阻塞CPU、默认0新GPU物理：仅用已保存原件定位和修正W51 MaterialSupportHeight state重建的9.00991e-6误差，按真实native几何/context/活动分支核对；缺字段如实交缺口，不填零、不另开原episode。不要重复已完成114残差归约来占显存。3个活动纯项、6个Trot下为零的Stand项、12个未重建项分清，GPU不正式安装。

资源按现场25核余量10+10+2+3，线程库1、构建先-j6、checkout/build/进程树隔离。随真实job记录可得核时/CPU/GPU用量，不另开满载benchmark。所有新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix；无固定墙钟完成时间、无到点kill、对话结束不杀后台。物理任务定义、原失败条件、技术/资源错误和有限工作量保留，running如实记录。

代码与紧凑结果推GitHub，原件和全时段视频D57直传Release；合并最新main、短写串行、不强推，回执不明先核远端。GitHub核验后结束，不在我的电脑额外下载/复制/打包/镜像/归档；保留服务器仍被引用的checkpoint/resume/科研原件。不要自动借我的VPN/Clash/SSH转发。汇报真实实现、步态/任务资格、剩余机制、工作量和失败分类、commit/资产及续接点。不要把计划或D8静站写成完整任务成功。
```
