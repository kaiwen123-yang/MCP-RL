# 地面材料记忆项目

截至 2026-11-28，W59 已按冻结方法自然结束，W61 已进入六线真实实验。长期主问题仍是同一世界地块在跨足和返程重访时的恢复与控制价值。材料、步态、MPC、WBC、学习以及联合适配的工作量和正负结论按运行身份追溯。[当前计划](S1_W61_RESEARCH_PLAN.md) · [W61 运行结果](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w61/index.json) · [每日记录](S1_DAILY_WORKLOG.md)

## 2026-11-26 状态

长期目标仍是可恢复世界地块材料状态的跨足、跨步、返程协同。W57/W58已结束；旧r55/r60及新W57 PR的A/B原E600成立。W57两头各8192/32updates，PR为冻结组合。W58最终G0/M1、联合学习0；56attempts中53自然终局、3技术中断。当前瓶颈为材料恢复链、动作条件执行预测、历史效应与同足印机制曝光。

后继以[W59完整计划](S1_W59_RESEARCH_PLAN.md)、[启动提示词](S1_W59_START_PROMPT.md)、[新审查](S1_W58_REVIEW_20261126.md)及[机器合同](../experiments/terrain_memory/s1-w59-research-contract.json)为准。只参考官方SuperMimic，不使用用户个人复现。模型M3/WM、规划P3、学习R3与联合算法分别对照，GPU批量训练与最多6物理槽相配；新计划尚未执行。无后台墙钟截止，无GitHub后用户电脑二次归档；唯一详细日记为[S1_DAILY_WORKLOG.md](S1_DAILY_WORKLOG.md)。

以下所有旧“当前/下一步”保留形成时事实，不覆盖本入口；旧资格、阴性结果和原合同均保留。

---

# 地面材料记忆项目 · W57实际进展与W58后继入口

2026-11-24 并发修订。W57 的 full/blind 均完成 8192 条及自然尾回合，用户随后要求在条件允许时将独立冻结评价扩到六个物理槽。当前 W57 评价按同一共享租约做 2→4→条件 6 的实测扩容，保留原 dispatcher 的资格判定及逐 case 身份。此前要求等待 W57 自然结束的条款继续适用于 W58 物理迁移。[评价调度合同](../experiments/terrain_memory/s1-w57-frozen-evaluation-slots.json) · [实际物理进度](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w57/dispatch/evaluation-parallelism.json)。

更新：2026-11-23，Asia/Shanghai。固定审查fd4ede8d、19:20:56 +08；真实现场继续核current/index/job。本次新增算法执行、机器人物理和PPO为0，完成的是审查与后继计划发布。

11 月 23 日 D57 已进入 W58 独立开发。v2 的 12 个前向真实同格事件和独立验证路线均已冻结。新资格、已完成回放、共同 G/M 选择及 W57 当前后继的实测状态见[W58 索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w58/index.json)与[每日实验记录](S1_DAILY_WORKLOG.md)。

## 长期目标与当前所在阶段

目标仍是可恢复世界地块材料状态的跨足/跨步协同：可观测载荷与运动→估计历史→动作条件下的未来接触预测→改变加载/交接/落点→任务、接触质量或效率收益。稳定名义行走、模块接通、材料历史因果收益、多seed/泛化、轻量化/Go2-NX实机是不同阶段。

- **稳定名义行走已成立。** r55/r60 A/B原E600各十往返，实际F为接触力QP MPC＋WBC，不是旧N全身iLQG。保留旧资格，不反复重跑证明能走。
- **W57模块与学习已实装。** M1短期整足力保留误差下降；P1已在线选择材料候选并安装WBC命令；83D独立timing head full8192/32updates完成，blind7424/29updates＋6partial继续。旧W44 actor仍冻结。新资格目前只有M1+B A/B E30、M1+P A E30，训练E60不继承为最终冻结资格。
- **科研收益仍待建立。** M1释放→重载误差变差；P1单对照滑移约−8.9%但CoT约+1.1%；333次非空private-history干预0赢家翻转；R回报趋势近乎平坦。PR当前是冻结组合，联合on-policy未完成。
- **步态首因更清楚。** 四E600限定1047事件为末摆动早触反弹，新P独立150000物理行也发现20例；不是这类事件中的已确认支撑失载。材料直行周期约.63s、刚地约.53s；下一G应对准实际事件接管与承载建立。
- **机制曝光需新设计。** 旧短任务身体12次穿越及回家完成，严格同格跨足只有6/12、7/12，反向足印几何不重叠。保留旧判，v2在比较算法前冻结真实可实现事件，不事后放宽或只延长时间。

## 连续推进的下一步

先让W57匹配blind、全部冻结队列、条件长资格及private-short-A自然完成，保持现两槽与不可变身份。同时独立三线做G接触质量、M2源释放至跨足重载、P2接触时长/历史作用分解。融合后冻结共同版本，16条开发机制对照；有真实历史动作/任务信号才启动新的联合runner、三学习seed full/private-blind、最多49152 distinct及独立验证。三seed机制保留最少32条，开发集不得再用作保留集。

W58统一池默认3物理槽，实测吞吐/其他项目余量允许扩4–6；双collector原子拿2槽，显式native材料worker纳入配额。当前W57全后继结束前不换池。后台无墙钟截止、不因对话结束kill；原模拟时长/硬界有效。GitHub直交并核验后无用户电脑二次归档。本计划不执行24路数据迁移或原件删除。

[本次证据审查](S1_W57_REVIEW_20261123.md) · [完整W58计划](S1_W58_RESEARCH_PLAN.md) · [启动提示词](S1_W58_START_PROMPT.md) · [八项文献](S1_W57_LITERATURE_UPDATE_20261123.md) · [每日详细记录](S1_DAILY_WORKLOG.md) · [最新执行索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w57/index.json)

以下保留历史形成时事实、协议和“当前/下一步”；其旧架构、未稳定、未学习、deadline和归档文字不覆盖此入口。W57已运行身份仍遵守原W57合同，W58仅是有明确条件的后继，不冒充已执行。

## 历史：W56（已完成）

长期目标是从可观测的载荷与运动估计同一世界地块的历史，并在完整往返中测量它对后续支撑的影响。F/r55 在两份冻结 nominal case 各自通过原 E600，A、B 的十圈与终点保持均已在执行端独立复算。r58 的两份原 E300 在计算停滞后技术中断；r60 用相同材料模型的 CPU FP64 在线实现，A、B 独立通过原 E600。两套终点保持分别持续 133.946 s 和 130.180 s，500 Hz 原轨迹复核均为零违规样本。W55 两参数材料候选在保留组的恢复误差增加，原任务继续使用 B ideal。

[W56 执行计划](S1_W56_STABILITY_PLAN.md)规定原资格链、显式恢复特征对照、观察窗口与法向参考、N 的全预测域修订及计算配额。实际执行与证据按身份更新到[统一索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w56/index.json)及[逐阶段结果](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w56/phase.json)。

# W54 实际结果（历史入口）

更新：2026-11-17（Asia/Shanghai）。长期主问题仍为同一世界地块的跨足、跨步、返程材料记忆辨识与协同控制，原Go2/TM02合法稳定往返尚未建立。

W54已实际实施F/N/C并完成9次不同版本原E30、0次通过；没有合格内核，seed2/E60/E300/E600未运行，新PPO samples/updates/donor=0。所有物理任务均按原条件自然结束。详细版本表、命令和原始结果在[独立索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w54/index.json)，当前机器入口为[current.json](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/current.json)。

最新F/r52修正真实接触与承载、球底任务、MPC/WBC运动对齐、2ms保持和内接真实摩擦圆锥，仍在5.600s body_fall；身体失稳早于材料加载。N/r48的E/nominal链已核对，但全部末次候选可满足足域代理而接受机身降低，最终8.3568s躯干触地。N出现12次跨腿地块重访事件，未完成材料贯穿/返程或记忆收益对照。现有记录不足以将剩余协同问题归结为另一项唯一接线错误，后续不以盲调权重或原样重跑替代明确模型变更。

C最新r52批覆盖16完整horizon/224个力区间，CPU独立复算16/16通过；物理μ=.5、共同内接轴斜率μ/√2。GPU包含传输.350782s、包括初始化编译2.171979s，CPU装配加QP .665930s；不同优化预算，未建立在线收益、GPU WBC或材料动力学覆盖。历史外包方锥结果保留并纠正。

9任务共383713物理子步、76.7426s仿真、4263.539333 CPU-s；有效双物理重叠41.1324s。29个Release资产含9全视频均已核size/digest。工作区/root/autodl-tmp/MCP-RL-w54，W44完整学习状态与服务器原件保留，用户电脑未额外下载或归档。

以下为历史形成时的计划与结果，其“当前/下一步”不覆盖本页顶部及current.json。

---

# 历史主计划：W53 及之前的事实与形成时协议

## 2026-11-17 实际执行结果

更新：2026-11-17（Asia/Shanghai）。W53 已在包含 `1832ba0` 的最新 main 上实现并运行两条原 E30。N 恢复 R8/S3 完整运动内核，在材料前形成交替支撑和连续前进，最大 x=0.723845 m；6.4656 s 触发原材料主应变域限制（候选材料时钟 6.4658 s，|strain|=0.0201782>0.02），末身体高度 0.290389 m。F 真实卸载并前进，但落足晚 74–114 ms、支撑缺口和机身下降；3.120 s 的力 MPC 静态力矩约束出现明确不可行，末身体高度 0.200225 m，材料加载0。

两条有效物理启动均未通过 E30。另有两次零物理初始化错误，已分别定向修正并完整保留：N 缺生成的任务注册资源，F 权重数组 dtype。无第三物理控制族、无新 D8、无 seed2、无 E60/E300/E600；新 PPO/donor=0。N 的进展限于本次材料前运动，材料未知 predictor 未完成材料记忆收益评价。F 的同次失败矩阵证实摆动零反力行要求 0≥1.70025764 Nm，不是随机求解器故障。

C 对既有 W52 6,152 状态完成一次 FP64 GPU模型/阶段矩阵比较。已覆盖量误差约1e-15；CPU参考2.224 s，含CPU h与传输的GPU路径1.908 s，含初始化2.303 s；未建立在线收益，M/h/Jdot/WBC与完整horizon求解未迁移。所有新后台均无墙钟截止；实际任务按原物理/数值条件结束。原件与全视频已从D57直传Release，后续入口为现有W53 phase/结果；不要把下方形成时的计划文字当新启动授权。

[W53 实际结果](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w53/paired-results.json) · [执行状态](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w53/phase.json) · [N 实际分析](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w53/n-actual-analysis.json) · [F 实际分析](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w53/f-actual-analysis.json) · [GPU结果](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w53/task-c.json)

## 以下保留 2026-11-16 计划与形成时证据

更新：2026-11-16（Asia/Shanghai）。本次审查基点 `5e036c6918ae72bb795c4b1b5be08c1e109ade32`。**W52 已结束且完成交付；W53 为已明确范围的下一实施计划，尚未实现或运行。** 当前原 Go2/TM02 稳定完整往返仍未建立。本次新增物理、native rollout、FD、PPO=0。

[W52 独立审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W52-review/README.md) · [W53 实施计划](S1_W53_STABILITY_PLAN.md) · [Codex 启动提示词](S1_W53_START_PROMPT.md) · [机器交接](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W52-review/handoff.json) · [W52 原始结果](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w52/paired-results.json)

## 最新结论与下一轮范围

D2/r40 在 5.6716 s、D3/r41 在 6.6278 s floor–trunk 非法接触，最大 x 仅 7.332/29.941 mm，末 x −78.349/−44.903 mm；两臂实际材料加载均 0。2.002–4.3 s 实际后退 15.757/39.063 mm，四足同时承重 81.897%/60.052%。两次有效物理启动、零技术物理重启，E30/E60/E300/E600 未运行。原视频全部 372 个发布帧与完整物理日志相符，没有正常持续推进。命令 readback 正确、真正 held 尾部过期为零、后台没有固定墙钟截止。

W52 几何修复生效，但最早有效 S 在原生预测中仍全程四足承重：目标 apex 约 5 mm，FR/RL 载荷 28.20/33.75 N。保存政策动作的精确重构误差 1.11e−16 rad，旧 B 的 reference/K/H 与支撑输入偏置保留，只追加 qIK 差；不能靠几何正确推出卸载，也不能直接删偏置或单独限制 K 当作已证修复。D3 真正增加独立 B 搜索而没有扭转运动。主问题已收敛到完整运动目标、支撑/摆动受力及执行一致性。

W53 改为两套完整控制器并行：N 从明确源码/pin/三补丁恢复 R8/S3 全身 iLQG 运动语义（2 Hz/.45/3 cm，Stand 支撑参考只在 Stand），接当前真实材料任务；F 独立接触力 MPC+WBC（2 Hz/.60/3 cm、.20 s 摆动），统一身体/足 p/v/a/支撑力，通过原 PD60/5 执行完整命令并约束总力矩。N 的旧刚地资格不继承；F 的显式接触表与 N 的软 gait 不构成单变量消融，材料支撑允许真实压入/回弹。

两线就绪后同 seed1 各一次原 E30，不加 D8、旧 9+3 或调参矩阵。依据原结果选定一个完整内核，复用其同身份 seed1 E30 资格，另开 seed2 E30，各 case 按原 E30→E60→E300→E600 推进。两首轮都失败交真实分阶段结果，不再临时生第三控制族。第三线 C 仅用已有 W52 状态做服务 F 的 FP64 模型/矩阵批量路径，不新增物理/rollout/FD/PPO，不在线更换后端。

本轮 W52 消耗 12,801 full rollout、2,472,984 transition FD、538,266 risk FD、61,497 物理子步、3.640 CPU 核时；新增 PPO/donor/GPU 物理 0。C 高度重建误差已修至 8.88e−16、排序不变。两 CPU job 全阶段平均 4.737/4.596 核，648 个候选 NPZ 共 20.918 GB，约 41–43% eval 墙钟尚未由已有命名计时覆盖。W53 在真实任务内分段计时、精简普通非赢家重复数组，保留全程实际物理、已安装完整政策/参考、材料与视频；不把未计时全归 IO，不把 10 核 affinity 冒充满载。

现场 25 核余量按 10+10+2+3、最多两个重物理任务。后台持续 deadline=null、wall_deadline_enabled=false，无固定墙钟结束时刻或到点 kill；原物理时长/失败条件保留。GitHub 直传并核验后结束，不在用户电脑额外下载、复制、镜像、打包或归档；被引用服务器原件/学习状态保留。

以下第 1 节保留长期目标与现有链路；第 2–8 节保留 W51 形成时分析与 W52 已执行方案。其“下一步”“当前”只属于历史，不覆盖本页顶部和 W53 计划。

## 1. 长期目标、真实控制链与当前阶段

主线是**面向重复踩踏的四足机器人地面记忆辨识与跨步协同控制**：同一世界地块被前足、后足和返程重复加载，利用材料持久状态，让未来支撑需求影响当前加载、落点和重访。首先建立原 Go2/TM02 合法稳定完整往返，再研究材料历史的因果决策收益、泛化、轻量化以及 Go2/NX 实机。

现有执行链：因果 PANN → 658D PPO 观察 → 2 Hz scalar cap（.18–.25 m/s）→ 逻辑50 Hz全身MPC → 500 Hz时变关节目标 → PD60/5 → 5000 Hz原物理。PPO确实曾学习，当前评价固定权重；observer和材料历史继续因果更新。逻辑频率不是墙钟实时。

候选仍未沿每条未来动作完整传播世界材料内态；scalar cap没有直接控制卸载、支撑力分配、落脚或快速关节反射的权限。因此稳定原控制、有效跨足/返程曝光、候选内材料历史、RAW/PANN/冻结或打乱历史对照、泛化和实时，是相关但不同的工作。现在尚未建立稳定材料往返，更没有可宣称的记忆因果收益。

| 历史阶段 | 保留的事实 | 不能据此声称 |
|---|---|---|
| 早期R1/R2/S0 | 材料辨识与学习的正负结果均留存 | 早期方向全部成功 |
| R8/S3刚地 | 原任务9/9，连续300s 3/3 | 当前材料/r40–r43资格继承 |
| R9/R10材料 | 完整任务0/8、0/9；R10曾到返程 | 已完成稳定材料往返 |
| R11 | 23次真实启动，有效E30/E60均0 | 恢复旧失效资格 |
| W34/W35 | 合法去程分别3、1，有返程进展 | 合法完整往返已成立 |
| W44学习 | A/B各14737/update57、完整resume | 同祖先数据可重复算独立样本 |
| W45/r34 | A13.072s、B5.582s跌倒 | strict接纳等于稳定 |
| W46/r35 | A26.432s触地、B11.638s跌倒 | 原E30合格 |
| W47/W48/r36 | self/antipode与alpha0修正；原轨迹同W46 | 数学修正已改善任务行为 |
| W49/r37 K | 4困难问题均有改进；原A/B各30s自然结束 | 低位/非足支撑算正常行走；E30实际0/2 |
| W50/r38 H | 4保存问题10/36 strict改进；原任务未跑 | bridge错误物理有效，或已恢复正常推进 |
| W51/r38-r39 | 两有效D8：D0跌倒，D1静站；定位参考下穿 | 8s观察完成等于正常步态或原资格 |
| W52/r40-r41 | 几何修复和独立B/S真实执行；两臂均在材料前触地 | 已恢复正常运动或获得E30资格 |

学习状态：A/B共享祖先6545/update25；PANN唯一转移22929，另RAW256，R12 distinct23185。W44 A/B checkpoint-014737、resume ep00428/ep00417、全actor/critic/optimizer/norm/RNG、tail0/excluded145保留。W45–W52新增PPO均0/0；15249/59不是自动续训预算。

## 2. 历史 W51 实际结果

原始 [phase](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w51/phase.json) 状态为 paired_D8_ended_pending_scientific_selection；这是形成时待审查记录。现在科学审查结论为**没有可提升的正常行走版本**，详见W51 review/handoff。旧“D1仍running、截至2.408s”已经失效。

| 指标 | D0：r38/H | D1：r39，命令修正后 |
|---|---:|---:|
| 结束 | 5.814s，body_fall | 8s，diagnostic_window_completed |
| 最大x | .095709m | .005452m |
| 末x / z | .095709 / .118934m | -.012449 / .288505m |
| 最低z | .118934m | .285769m |
| 2.002–4.3s Δx | -20.903mm | -6.091mm |
| 2.002–4.3s目标swing仍受载 | 71.18–80.19% | 94.89–100% |
| 后窗四足承重 | 4.3–5.814s仅2.90%，同时失稳 | 4.3–8s为100%、载荷切换0 |
| 材料实际加载/合法去程/往返 | 均0 | 均0 |
| 原E30启动 | 0 | 0 |

有效D0/D1的0–2s状态、力、动作、接触数组相同。首个技术D1另运行3.120s，原mission非零但userdata[78:84]未填，S错误使用零世界速度。执行端定向停其独立server，产生Connection refused；提交463e91b补原命令写入，保留D0不重跑，修正D1从t=0一次重启。该技术尝试不作第三个控制样本，也不是已证随机网络故障或deadline问题。

本轮3次启动分别为1个有效物理跌倒、1个诊断完成但不走、1个技术无效。原E30/E60/E300/E600整链unrun，没有W51原阶段blocked执行行，不把W50八行重复记为本轮失败。D8 qualification_usable=false，success=null，不冒充原资格。

[结果独立复算](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-review/results/review.md)保存全时段口径、工作量、时钟和资源缺口。

## 3. 视频判断与新机制证据

[三段视频审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-review/video-review.md)实际查看89个等间隔/末帧和12个放大关键姿态。D0先折腿降低、短暂顶起、后倒下；不是一直正常走到末尾才偶然跌倒。D1基本原地站立，身体高度保持却没有持续迈步。足端快速行为由500Hz轨迹和5000Hz物理记录核对。

有效D1的301份IK已全部解析，650个当前swing足×replan快照中，360个的已保存剩余目标节点全部不高于地面，362个比真实初始压入继续下移>1μm，96个低于-1mm，已保存节点最深-3.570516mm。所有相关目标位于材料x=.8之前，平地z=0、原足半径.022；不是材料形变高程误读。使用真实足位置/速度和原公式独立重建的最大误差2.36e-16m。

确定错误在旧接续式：原绝对C2高度减去H0乘当前名义高度、再减去T H1乘当前名义速度。实际足未按相位抬起时，这个连续性修正会抵消抬脚，甚至主动下穿。IK最大误差<1e-7、68个原件事件native命令逐值一致、同段锚点/落点漂移0，均不能纠正一个目标本身错误的曲线。650是重规划快照数，不是650次独立物理失败。

301次S均数值有效并作Q，其中299次J高于B。94次homotopy安装有92次属于Stand，运动阶段只有call121@2.002与call128@2.14；直接S只在call216@3.9安装一次。D1已读26call/520候选；早期S native仍四足承重，末call421 S后退29.476mm、足最大间隙22.46μm。也有向前但身体降低/足运动过大的被拒候选，不能说全部候选倒退，也未证明J错拒正常步态。

保留B作最终接受底线，不等于B附近仍有独立搜索。S即使较差也占据唯一线性化方向，会限制搜索覆盖；恢复B方向是可以比较的次级机制，但D0已证明它不是稳定保险。[控制审查与原件](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-review/control/review.md)还分解了raw动作：支撑偏置确实保留，若干S没有被QP修改；这不足以单独证明“去掉支撑偏置就会走”。

## 4. 参考刷新与D0失稳：不能继续沿用单一旧解释

D0共311calls、272次安装、256次正homotopy同问题J下降。2907个held查询真正尾部过期为0，最后reference=[5.8,6.15]，因此W51 D0不是旧Stand参考冻结。及时刷新并未保证正常支撑。

D1最后policy generation在4.2s，reference/K/H仍来自2.14s、截止2.49s；末policy age3.8s，reference expiry5.51s。2.492s首真正尾部查询，2755/4001；末查询未积分，实际积分口径2754/4000。float32的covers=false含起点舍入误差：D0有134条false而真实过期0，不能把标志直接当过期。

D0时序：2.976s首QP投影；3.016s首原目标硬裁剪、Kδx约1.968rad；3.17s首z<.28；3.4614s首执行PD明显越界>.01Nm；3.564/3.588s首z<.27/.25；4.05s最大Kδx=37.867rad；5.2304s首非足承重；5.814s跌倒。最初2.976s执行越界仅8.39e-7Nm，不能作明显饱和。raw目标硬clip648、QP改写627、包含QP的总改写708分开；原PD超界1557子步，其中>.01Nm为996、>1Nm为619。

这些证明时间先后与异常存在，未隔离D0最初失稳的唯一因果。不能把后段37.867rad反推为最初错误，也不因为D1反馈小、没倒就宣布稳定运动。用户感觉步态越来越不正常有现象依据，但不同版本/任务窗不构成统计意义的单调退化曲线。

## 5. 历史 W52 执行方案：已执行，不再重跑

[W52计划](S1_W52_STABILITY_PLAN.md)是实施依据。D2（拟r40）删除错误接续式，从真实足位置/速度直接接原固定apex/touchdown，分段五次曲线，保留2Hz/.45/5mm/绝对相位与锚点/落点。原swing=.275s、rise=.075s；关键时刻从原phase推，不从observed_start重计。不每20ms推迟apex、不clip高度冒充C2、不归零真实速度、不加v>=0浮点硬门；解析检查极值，真正不可行明确S不可用，回正常B搜索。

首次Trot的FL/RR已过原apex，不通过重置phase伪造起步；下一正常窗口仍保留。已有301份原件只做一次必要的纯数学修后验证，不新开微物理准入。暂不硬改PD支撑偏置/反馈/K限制/成本/步高/horizon/PPO，S仍走完整原native链和原risk/J严格比较。

两组资源可用时，D3（拟r41）在相同修复上为B/S各保留独立first-valid+9alpha，统一原comparator选择，分支状态和工作量独立。D2通常≤21、D3≤30full rollout/epoch，不冒充等预算比较。D2是主线，D3不是新资格门槛；资源不足优先D2，不等待D3。

D2/D3各一次同W44 A seed1/完整原初态t0→D8，正常物理提前终止，完整实际时段视频与足载/参考/动作记录。D2按原规则自然结束D8并完整读取实际轨迹后，若确有持续正常支撑推进，才优先冻结简单D2并进入原A/B E30，不要求D3也成功；最多两个重native job，资源逐个释放即可启动正式两臂。只有D3有实质效果才选择它。都差则交真实负结果，分清剩余问题，不临时叠加方法追成功。

原A seed1/B seed2各一次独立E30；逐case原E30→E60→E300→E600，原材料路线/远端3m/全观察/回家稳定2s/270与540s内5与10往返不变。D8不拼接资格，失败只blocked本case后级。固定新控制器后不随D3后来的分数更低更换版本。

## 6. 算力与GPU：能并行，但不能把容量当成工作成果

W51两个有效臂：732calls、14341native full rollout、2673288transition FD、598194risk FD、69070plant子步。含技术D1：909calls、17647rollout、3281904transition FD、726082risk FD、84670plant子步；只算已写完调用，中断在途未知。新PPO/donor/GPU物理0。

双CPU任务实际重叠合计869.19s（14.49min），其中D0与有效D1重叠332.68s。C与最初两臂只重叠.924s，没有与修正D1重叠；不能称持续三重满载。CPU核时/连续利用率/整卡GPU峰值本轮缺记录。D0/D1每1s仿真约需161.29/155.94s墙钟，未达实时。

C在RTX PRO 6000 Blackwell Server Edition上做114×176×113原残差、22项FP64归约，最大J误差5.82077e-11，六组各361内部比较一致；tensor峰45561344B≈43.45MiB，归约.104s、脚本.924s。state重建仅3个实际活动纯项；6个Stand项在Trot下为零不算活动覆盖，MaterialSupportHeight有9.00991e-6误差，12项未重建。不是完整native成本、接触求解、FD或GPU MPC迁移。

W52 C只用已有原件定位修正MaterialSupportHeight缺口，必要FP64比较、活动分支覆盖与缺字段明确记录；默认0新GPU物理，不重复归约/F1/L/F2、不填零、不新开episode补档、不正式安装。两重CPU各10核、C2核、其余约3核用于系统/主控/其他项目，线程库1，先-j6构建并隔离可写build。随真实job记录可得核时/CPU/GPU用量，不另做benchmark；完成的C自然结束，不为96GB空闲重复训练或批量。

## 7. 历史结果和排除项继续有效

W49仍是最新完整原任务：A/B各30s自然结束，E30 0/2，合法往返0；最大x .349667/1.020530m、末z .138718/.142818m。A未到材料已异常，B首材料足载12.1866s、机体west entry20.064s，异常早于材料；不能全部归材料/PANN。用户B视频4.4–5s下蹲/顶起，5.2–7s暂时抬高，7–8s后低伏挪动，不能写5s后一直趴地。

W50四保存问题76rollout、36正候选10strict，预测改善仍坍低/后退；加两midpoint共116rollout、114candidate、23112transition FD、4746risk FD。400bridge子步因same-SONAME错误库ABI失配，物理等价有效0；独立native保存问题评分仍有效，dlmopen修复无修后物理结果。原八阶段blocked/unrun不是八次失败。

W50另从231份旧policy与完整held确认W49 reference/K/H都来自initial Stand[0,.35]，每流14825/15001过期、末29.65s。raw-center/切空间/K/H混合/所选对象基本吻合，不能重复把极小FP32差当主要厘米级失效；公共runtime两处缩进当时已修。W51后来正常D8已经走过真实入口，不再停留在“仅AST通过”的旧状态。

历史GPU contact动力学差异尚未解决；canonical sort stub不当修复，不重复相同探针，不以bitexact作总门。没有权限自动借用户电脑网络、删除被引用科研原件或复活旧资格。

[W49审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W49-final-review/README.md) · [W50审查](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W50-final-review/README.md) · [W51原计划](S1_W51_STABILITY_PLAN.md)；更早W43–W48审查仍在同一reviews目录，按形成时点阅读。

## 8. 文献、阅读范围与交付

[W51文献对照](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-review/literature.md)核对ROM-guided CI-MPC、Perceptive NMPC、Reference-Free MPC、全局采样/局部接触优化、RL-augmented MPC五篇一手来源。共同启发是把几何意图落实为受力与执行，保留有用搜索覆盖；不直接移植人形/机械臂的假设，不把外文成功套作本项目资格。此前MuJoCo/HOUND/HiLQR/DIAL等结论仍保留。

本次相对354b14b完整读流52变化文件51178658B、全解析30JSON。实际下载必要原件15资产2532967906B，解包2653成员2769219396B，读取三臂全时序/301份IK/所需候选事件并查看三视频。未下载所有历史Release，不把逐字节读流等同所有数字语义理解。

W51 live GitHub两Release共38资产8470903356B，状态/size/digest与manifest匹配。[阅读清单](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-review/read-coverage.json) · [核验](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W51-review/remote-verification.json)。本次生成审查/图表/计划，没有新控制实验。

**新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix，无固定墙钟完成时刻、无到点kill、对话结束不杀后台。** 物理任务定义、原失败条件、明确技术/数值/资源错误、用户停止、单次网络timeout与有限工作量保持，running不伪写completed，无无限重试。

紧凑Git先推，必要原件/视频D57增量Release直传，短写串行合并最新main、不强推，回执不明先查远端。核commit/URL/size/digest后结束，**不在用户电脑额外下载、复制、打包、镜像或归档**。服务器仍被引用的科研输入/checkpoint/resume/视频保留，不自动删除；只用已授权服务器网络，不自动借电脑VPN/Clash/SSH转发。旧固定截止/归档/历史当前规则均不覆盖本页顶部和W53计划。
