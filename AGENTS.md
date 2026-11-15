# Codex 执行入口：W51 已审查，W52 修复摆动参考

更新：2026-11-16（Asia/Shanghai）。结果基点 `a621d7d35e6016634b388a3aef0d39d8d008aab0`。**W51 已结束；W52 planned_not_implemented。** 补充审查开始时实时 main 仍为 `28b8654`，未见 W52 新结果。本次审查没有启动新物理、native solve、FD、GPU 或 PPO。

先读 [项目记忆](docs/TERRAIN_MEMORY_PLAN.md)、[最新补充复核](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-followup-20261116/README.md)、[W51 完整审查](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-review/README.md)、[W52 详细计划](docs/S1_W52_STABILITY_PLAN.md)、[启动提示词](docs/S1_W52_START_PROMPT.md)、[最新机器交接](results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-followup-20261116/handoff.json)。实际实现与判定以 W52 计划为准；历史“running/当前/待实施”不覆盖新结果，不重开已完成 W51。

## 必须带入的新事实

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

## W52 执行要点

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
