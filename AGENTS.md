# Codex 当前入口 · W55 原资格执行

2026-11-19，D57 从 `c8e0b38` 启动 W55。F/r53 已在原 W44 A/seed1 E30 进入实际物理。N 从 r48 实施支撑投影与整身候选，C 已从四个 W54 实载回合和一个独立 seed2 回合重建加载边界。阶段资格、终止原因、材料贯穿和 Release 回执以本轮实际记录为准。

续接先查 D57 各 lane 的具体 job、tmux、PID 和物理时钟。F、N 分别在 /root/autodl-tmp/MCP-RL-w55-f、MCP-RL-w55-n。材料与串行发布分别在 MCP-RL-w55-c、MCP-RL-w55-integration。原物理、判据、W44 checkpoint 和已引用原件保留。所有任务 deadline=null、wall_deadline_enabled=false。相同 case 与版本仅续接已有任务。

[W55 当前结果](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/index.json) · [任务、进程及阶段](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w55/phase.json) · [完整协议](docs/S1_W55_STABILITY_PLAN.md)

## 以下 W54 入口及执行约定属于历史

# Codex当前入口：W54已执行并交付，尚无原E30资格

更新：2026-11-17（Asia/Shanghai）。W54从df07c4b实际实施F/N/C，完成9次不同版本原E30、0次通过。全部物理任务按原条件自然结束，当前无活动物理任务；没有合格内核，seed2及E60/E300/E600未运行。新PPO samples/updates/donor=0。

先读轻量current.json及w54/index.json（均在results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/），再读w54/f-r52-actual-analysis.json、n-r48-analysis.json和task-c-r52-friction.json。续接工作区为/root/autodl-tmp/MCP-RL-w54；旧本地与服务器修改、W44完整checkpoint/resume和原件保留。

F/r52包含本轮成组修复及有证据的接触点、2ms held-input、载荷建立和真实摩擦可行集修订，但5.600s body_fall；4.920s已降高，先于5.325s材料加载。N/r48在8.3568s躯干触地；E/nominal接线已核对，候选仍可满足足域代理却预测低机身。记录尚未把剩余问题唯一归结为另一项明确接线错误，不能盲改权重、阈值或原样重跑负版本。12次跨腿地块重访不等于材料记忆收益评价。

C完成修后r52自然16个输入状态/224个力区间的完整GPU力序列优化和CPU独立复算；物理μ=.5，CPU/GPU共同内接轴斜率μ/√2，16/16通过。没有GPU WBC/材料/在线收益资格，历史外包方锥结果已明确纠正并保留。

共383713物理子步，9任务进程合计4263.539333 CPU-s（1.184316核时），有效双物理重叠41.1324s。29个Release资产含9全视频，size/digest已核实。全路径、命令、版本、失败与发布回执在独立w54索引；原大index只作历史。

后续仍先核具体job/tmux/PID和修改；如有已运行同身份则续接。所有后台deadline=null、wall_deadline_enabled=false，无到点kill或对话结束kill。后续有明确的模型/执行改动再建立新身份；不复用失败资格，不直接启动E60。GitHub交付后不在用户电脑下载、复制或归档科研文件，不删除被引用服务器原件。

## 以下为W54形成时的执行约定与历史

## 已确认事实与本轮工作

N/r42 恢复真实交替支撑与连续前进，最大 x=.723845m；6.0274s 首次材料受载，6.4656s 材料域退出，末 z=.290389m、无非足力。末次 FL 以-.9558m/s 落到新地块，1.6ms 内进入下一越域候选；最后实加1194.56N，1206.81N候选未施加。不是body_fall，也没有材料往返/跨足记忆收益资格。

F/r43 在材料前反复失去支撑与降高，3.120s控制器不可行：六行0>=1.70025764Nm。五项已证问题是当前静态力矩/映射界复制到未来、绝对时钟忽略迟落足、悬空stance anchor、删力改变身体加速度目标、未来落点回退旧锚；另有已知dry friction模型缺项。末速/末加速度本来为零，PD瞬时映射正确，首摆动无饱和，不能重复认领这些为新修。

W53 两有效物理任务重叠0s；零步初始化两次另算，E30通过0、后级未运行、PPO/donor=0。N导数热段约9.80核、整任务5.98核，F约1.02核。C仅局部GPU模型/矩阵，无完整求解/在线收益。

1. F一次完成上述接触事件、承载交接、实际触地锚、未来落点传播、独立身体目标与执行可行性修复。原WBC总力矩/mapped-u/关节硬界保留，SRB不冒称完整未来关节保证。摩擦处理进入模型和约束。名义2Hz允许被真实承载延后，实际相位如实记录。
2. N保留r42完整运动，以确定性触地前执行算子主动提前减速，统一真实500Hz/native forward/FD与原2ms位置hold。新增动态记忆必须进入状态/转移/导数，或真实/预测同样冻结。旧R10已试近地软惩罚/局部弹簧，不能原样重跑；不能挂预测外滤波器或只拒绝新候选后沿unsafe旧B。材料只用因果估计，不宣称硬安全。
3. C做有限保存状态上的完整GPU horizon力序列优化、首力/全序列和CPU独立复算；修后接触输入、力参数化与可行集一致，报告全路径时延/覆盖。无新GPU物理，不重复局部矩阵凑显存，不热替换正式控制器。
4. F/N就绪后各自同W44 A/seed1原E30，实际可并行就并行。首任务前已知问题成组修完，普通接线/构建错误直接修。失败后在同轮根据新增证据修本范围内明确问题再以新身份推进；不两次失败就结束，不重复运行同一版本、同一输入的已定负结果，不换seed追成功、不参数网格/第三物理族/D8/旧9+3。没有可定位新改动则报具体阻塞。
5. 第一套通过原E30且全记录支持正常运动的内核可冻结继续，复用同身份前序，另一槽释放后开W44 B/seed2。逐case原E30→E60→E300→E600，原路线/材料/3m远端/回家稳定2s/5与10往返条件不变，不跨版本/seed/初态/workers借资格。已运行另一任务自然结束。
6. 现场25核：初期N≤10workers、F≤4核、C/装配≤8核、系统/其他≥3核；两N case保持首次E30冻结workers，初选10时10+10+2+3。数学库线程1，构建合计先-j6且计入共享配额。记录真实有效区间重叠和热段核时，不以affinity冒充使用率。
7. 只做改动必要的保存数据/同输入核验、正常构建与原任务验证，不新增全仓hash/mock/shape/import/smoke或重复旧探针。spec/build/runtime/backend/命令/actor/hold/终止/记录/协议必须接通，不只改名。
8. 保留全程500Hz轨迹、5000Hz实际接触/足速/力矩、材料双时钟、全视频及已安装策略/关键原件；按计划增加接触事件、E前后动作、任务残差与摩擦项。新PPO样本/更新/donor=0，W44 A/B完整checkpoint/resume保留。
9. 所有新后台deadline=null、wall_deadline_enabled=false、省略deadline-unix；无固定墙钟截止、到点kill或对话结束kill。原物理长度/失败条件、具体数值资源错误和用户停止有效。running如实记录。
10. GitHub直交，紧凑代码/结果先推，必要原件与全视频D57增量Release直传；串行整合最新main、不强推，回执不明先核远端。核commit/URL/size/digest后结束，不在用户电脑额外下载/复制/镜像/打包/归档，不自动删服务器被引用原件，不借电脑VPN/Clash/SSH转发。

最终分别报告实现、真实正常行走、最高原任务资格、材料记忆评价、GPU覆盖/收益、失败与未运行、实际并行、GitHub链接和续接点。计划发布不等于实验成功。

## 以下 W53 入口为历史

### W53 已完成两条原 E30，均未取得后级资格

更新：2026-11-17（Asia/Shanghai）。W53 已在包含 `1832ba0` 的最新 main 上实现并运行两条原 E30。N 恢复 R8/S3 完整运动内核，在材料前形成交替支撑和连续前进，最大 x=0.723845 m；6.4656 s 触发原材料主应变域限制（候选材料时钟 6.4658 s，|strain|=0.0201782>0.02），末身体高度 0.290389 m。F 真实卸载并前进，但落足晚 74–114 ms、支撑缺口和机身下降；3.120 s 的力 MPC 静态力矩约束出现明确不可行，末身体高度 0.200225 m，材料加载0。

两条有效物理启动均未通过 E30。另有两次零物理初始化错误，已分别定向修正并完整保留：N 缺生成的任务注册资源，F 权重数组 dtype。无第三物理控制族、无新 D8、无 seed2、无 E60/E300/E600；新 PPO/donor=0。N 的进展限于本次材料前运动，材料未知 predictor 未完成材料记忆收益评价。F 的同次失败矩阵证实摆动零反力行要求 0≥1.70025764 Nm，不是随机求解器故障。

C 对既有 W52 6,152 状态完成一次 FP64 GPU模型/阶段矩阵比较。已覆盖量误差约1e-15；CPU参考2.224 s，含CPU h与传输的GPU路径1.908 s，含初始化2.303 s；未建立在线收益，M/h/Jdot/WBC与完整horizon求解未迁移。所有新后台均无墙钟截止；实际任务按原物理/数值条件结束。原件与全视频已从D57直传Release，后续入口为现有W53 phase/结果；不要把下方形成时的计划文字当新启动授权。

[W53 实际结果](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w53/paired-results.json) · [执行状态](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w53/phase.json) · [N 实际分析](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w53/n-actual-analysis.json) · [F 实际分析](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w53/f-actual-analysis.json) · [GPU结果](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w53/task-c.json)

## W53 计划形成时的执行约定（历史，保留）
更新：2026-11-16（Asia/Shanghai）。审查基点 `5e036c6918ae72bb795c4b1b5be08c1e109ade32`。**W52 已完成负结果和发布；W53 仅已规划，尚未实现或运行。** 本次提交没有新增物理、native rollout、FD 或 PPO；下一轮按启动提示词执行明确的 W53 范围。

先读 [W53 计划](docs/S1_W53_STABILITY_PLAN.md)、[启动提示词](docs/S1_W53_START_PROMPT.md)、[项目记忆](docs/TERRAIN_MEMORY_PLAN.md)、[W52 本次审查](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W52-review/README.md) 和 [机器交接](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W52-review/handoff.json)。本页下方 W51/W52 执行协议保留为历史，不能覆盖 W53 或重复启动旧任务。

## 当前必须继承的事实

W52 D2/r40 于 5.6716 s、D3/r41 于 6.6278 s 发生 floor–trunk 非法接触，材料加载均 0，原 E30 未启动。两次有效物理启动、零技术物理重启；后级未运行不是八次失败。真正参考尾部过期 0，命令读回正确，后台没有墙钟截止。首有效 S 约 5 mm 目标仍对应 native FR/RL 28.20/33.75 N：保存政策精确证实旧 B 的 reference/K/H 和支撑中心被保留，只加 qIK 差。不要把后期 K 尖峰、clip 或材料解释成已隔离的最初原因，也不要删偏置冒充已证修复。

C 的高度重建已经按实际 Agent XML 序列化精度修正至 8.88e−16，原排序不变；不是完整 GPU MPC，不再重做。W44 A/B checkpoint-014737 和完整学习 resume 保留，W45–W52 新 PPO 均为 0。

## W53 执行要点

1. 核最新 main、本地改动和具体 job/tmux/PID/物理时钟/配额，已有同身份有效工作则续接；隔离可写 build、输出和进程树，不重复启动、不宽前缀 kill、不动其他项目。
2. N 从冻结源码/pin/三补丁完整恢复 R8/S3 运动语义，接当前原 TM02 plant/任务和冻结 cap；原 2 Hz/.45/3 cm，支撑参考只在 Stand，保留明确的执行正确性适配。不能只改步高而保留整套当前 moving 控制包；历史刚地成功不继承新材料资格。
3. F 是独立接触力 MPC+WBC，统一身体、接触、足 p/v/a 和力矩；固定 2 Hz/.60/3 cm、.20 s swing，首任务前冻结起步和接触转换。原 PD60/5 接口必须表达完整命令，约束映射后输入及总力矩；不用旧 B 的 J/K/H 接管，不改原物理接触或材料判据。
4. N/F 同 W44 A seed1 各预声明一次原 E30，就绪者先跑，不新增 D8 或旧 9+3 矩阵。按计划选一个完整内核；所选同身份 seed1 E30 可直接作 E60 前序，另槽开所选 W44 B seed2 E30。各 case 原 E30→E60→E300→E600，不能借资格；两首轮都失败则交真实分阶段负结果，不临时加第三物理控制族。新 PPO/donor=0。
5. C 仅用已有 W52 状态一次有限遍历，做服务 F 的 FP64 Go2 模型/约束装配和同模型 CPU 对照，明确活动覆盖与含传输的收益；新 plant/native rollout/FD/PPO=0，不在线替换，不阻塞 N/F。
6. 25 核余量按 N≤10、F≤10、C/记录2、系统/其他3，最多两个重物理任务；数学线程库1，构建总上限先-j6。随正常 job 分段记 CPU/wall，不能以 affinity 冒充利用率，不另跑满载 benchmark。
7. 保留全程 500 Hz 轨迹、5000 Hz 实际接触/力矩/双时钟、材料和全视频，每次实际安装的完整 policy/参考/获选轨迹可追溯；按计划精简普通非赢家全 ring 重复写入，说明缺失边界，不补跑、不删历史原件。
8. 只做真实接线、构建及本改动必要验证，不新增全仓 hash/mock/shape/import/smoke 或重复旧保存问题。真实 spec/build/runtime/backend/命令/actor/hold/终止/记录/协议身份全部接通，不能只新增版本名。
9. 所有新后台 deadline=null、wall_deadline_enabled=false，省略 deadline-unix，无固定墙钟结束时间，无到点 kill，对话结束不杀后台。原物理长度、失败条件、明确技术/资源错误和用户停止仍有效；持续推进已授权阶段，running 如实记录。
10. GitHub 为交付处，紧凑代码/结论先推，必要原件和全视频 D57 直接增量上传 Release。短写串行整合最新 main、不强推，核 commit/URL/size/digest 后结束；不在用户电脑额外下载、复制、镜像、打包或归档。保留服务器被引用 checkpoint/resume/原件，不自动借电脑 VPN/Clash/SSH 转发。

## W51背景事实（历史，保留）

- 补充复算：修正D1在3/6/8s仍收到约.22m/s正向命令，实际速度近零；18份材料chunk及三份material-final全部空。W51当前障碍是起步/接触实现，不能归为任务命令为零或已加载材料失败。
- 新版本接线不能只改名字：runtime运动开关仅匹配r39，多处strict/记录/native身份/before_call/held使用显式旧版本集合；userdata[78:84]修复和W51 recorder依赖w51_fixed_policy合同字段。r40/r41与新合同需同步继承，避免重新丢掉原修复。D3隔离完整单分支工作区，统一提交；候选ring就绪后保存同次实际所选rollout/policy，不补跑记录。

- D0/r38 在 5.814s 原 body_fall，末 z=.118934m；它全部2907次held查询没有真正尾部过期，持续刷新仍会失稳。3.016s首目标硬clip，3.17s首次z<.28；明显PD超界始3.4614s；37.867rad最大Kδx是4.05s的后段极值，不是已证明的最初原因。
- 命令修正后的D1/r39完成8s，但最大x=.005452m、末x=-.012449m、末z=.288505m。4.3–8s四足承重100%、载荷切换0、策略更新0，是静站/微后退，不是正常步态。
- D1最后policy generation为4.2s，reference/K/H仍来自2.14s、参考截止2.49s，末龄5.51s；实际过期held为2755/4001（随后有积分口径2754/4000）。float32的covers=false不可直接算过期。
- 301份D1 IK已全部解析：650个“当前swing足×replan”快照中，360个的已保存剩余目标节点全部不高于平地，96个低于-1mm，已保存节点最深-3.570516mm；旧绝对C2减去当前名义高度/速度的接续式会取消抬脚。独立公式重建误差2.36e-16m，不能误读成650次物理失败。
- 301个S数值有效并作Q，299个J高于B；94次homotopy安装92次属于Stand，运动时仅2次。所读S native轨迹没有正常卸载；有向前但姿态/支撑较差的被拒候选，尚无J错拒正常步态的证据。PD支撑偏置确实保留，但去掉就能走尚未证实。
- 本轮3次D8启动=1个有效物理跌倒+1个诊断完成但不走+1个命令布线技术无效。首D1因userdata[78:84]未填而定向停止；继承463e91b修复，不把Connection refused归随机断网或deadline。两有效臂均未加载材料，原E30启动0，正式阶梯整链unrun，不套用W50八个blocked。
- 有效pair 14341 full rollout、2673288 transition FD、598194 risk FD、69070 plant子步；含技术尝试17647/3281904/726082/84670。新增PPO/donor/GPU物理0。
- C的114×176×113 FP64残差→22项J已完成，误差5.82077e-11、同问题比较全同；tensor峰43.45MiB、归约.104s，不是完整GPU MPC。MaterialSupportHeight state重建仍有9.00991e-6误差，活动分支覆盖有限。

## W52 冻结执行协议（已执行，勿重跑）

1. 核最新main、本地改动、D57具体job/tmux/PID/物理时钟与配额；已有同身份有效工作就续接，不重复启动、不宽前缀kill、不动其他项目。默认CPU树 `/root/autodl-tmp/MCP-RL-w42`、GPU树 `/root/autodl-tmp/MCP-RL`、环境 `work/envs/d57` / `work/envs/w41-mjwarp`，以现场为准。
2. 主D2（拟r40）仅修几何接续：从真实足位置/速度直接接固定原apex/touchdown，分段五次曲线，保留2Hz/.45/5mm/绝对相位、原锚点/落点。原swing=.275s、rise=.075s。apex从原phase求，不从observed_start重新计；不每20ms推迟apex，不clip高度伪造C2，不归零真实速度。
3. apex后仍落地的late swing、解析极值不可行等使S明确不可用，回正常B初始化/搜索，不停止plant、不重置步态。实际初始压入和微小负速度按既有IK尺度解释，不增加v>=0浮点硬门。复用301原快照做一次纯数学修后核对，不新增微物理准入。
4. S仍复制完整B、保留原raw支撑中心/feedback/metric/delta，追加qIK-qB_joint并走原native rollout；不顺带修改PD/支撑偏置/成本/步高/horizon/K裁剪/PPO。D2有效S作Q、一个first-valid backward与原9alpha，所有安装仍按原risk/J严格比较全局B。较差S可用于搜索，不能强制安装。
5. 第二组10核可用时并行D3（拟r41）：相同曲线，S有效时B/S各自一个独立first-valid方向与9alpha，同epoch冻结输入，各自k/K/H/regularization/provenance，统一原comparator选。S无效只单B。每job总worker10，不增加不受配额的server；D2通常≤21、D3≤30full rollout/epoch，FD另计，不能声称等计算预算。
6. D2/D3各一次原evaluation t=0→D8，同W44 A/seed1 checkpoint-014737、归一化/确定actor/完整初态。保留原2s Stand和物理提前终止，正常plant/独立native/RPC/query/hold/因果observer；D8 qualification_usable=false，不拼接E30。真实spec/patch/prepare/runtime/service/server/recorder全部接通，ABI变化正常重编。
7. 记录三窗实际卸载/落脚/支撑交换、身体稳定/前进/非足力、raw/clip/QP/物理PD、policy与reference/K/H两种年龄，S几何与native足轨迹差及分支来源。保存实际全时段视频和关键B/S/Q/IK/候选对象；不为满8s重跑。
8. D2按原规则自然结束D8、完整读取实际轨迹后确有持续正常支撑推进，才优先冻结简单D2，不要求D3也成功。D3尚运行可自然完成，D2可用释放10核先开原A E30，另组释放再B；任何时刻最多2个重native job，正式机制冻结后不再切换。只有D3有实质效果才读完后选D3。两者都差就交真实负结果，不临时叠加方法或换seed追成功。
9. 选定后原A seed1/B seed2各一次独立E30，逐case原E30→E60→E300→E600；原路线、材料加载、3m目标、完整观察、60s完整往返/回家稳定2s、300/600的270/540s路线期限内5/10往返不变。失败仅blocked本case后级，不借他例资格。W44全学习resume保留，新PPO/donor0。
10. 第三线C不阻塞CPU、0新GPU物理；仅用已有原件定位并修正MaterialSupportHeight重建误差及必要FP64比较，真实活动分支/缺字段如实列出；不填零、不新开原episode补档、不重跑已完成归约/F1/L/F2凑显存，GPU不正式安装。
11. 现场25核余量按10+10+2+3，数学线程库1，构建先-j6，隔离可写build和进程树，尊重其他任务。随真实job记录可得user/system核时、CPU/GPU占用和显存，不做额外benchmark，不用affinity冒充满载。
12. 只做本改动和真实入口必要核验；不新增全仓hash/mock/shape/import/smoke或无信息防御性轮次，不重跑旧保存问题、错误bridge/40ms/GPU探针。几何校验不可用不是一次新的实验失败。

## 后台与交付偏好持续有效

**所有新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix；无固定墙钟完成时刻、无到点kill、对话结束不杀后台。** 物理任务时长与原失败条件、明确技术/数值/资源错误、用户停止、单次网络timeout和有限科学工作量保持。running如实记录，不无限重试。

紧凑Git先推，必要原件和全时段视频D57直传增量Release；短写串行合并最新main、不强推，回执不明先核远端。commit/URL/size/digest核实后结束，**不在用户电脑额外下载、复制、打包、镜像或归档**。服务器仍被引用的科研原件/checkpoint/resume/视频保留，不自动删除；仅用已授权服务器网络，不自动借电脑VPN/Clash/SSH转发。

最终分开报告实现、正常步态/推进、原任务资格、材料记忆收益和GPU覆盖；给真实工作量、技术/物理/任务目标失败及blocked/未运行、GitHub commit/资产与续接点。计划发布不等于实验成功。
