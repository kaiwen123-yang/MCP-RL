# 地面材料记忆项目：W51 已审查，W52 修复摆动参考

更新：2026-11-16（Asia/Shanghai）。**W52 D2/D3已经结束并完成完整轨迹/视频判定，两臂均未达到持续正常支撑推进，原A/B E30不启动。** D2/r40在5.6716s、D3/r41在6.6278s发生floor–trunk非法接触；末x分别−78.349/−44.903mm，实际材料加载0。不是两次技术中断，D8不提供原任务资格。

[W52实际结论与冻结协议](S1_W52_STABILITY_PLAN.md) · [完整结果](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w52/paired-results.json) · [全时程图](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w52/full-trajectory.png) · [运行与发布状态](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w52/phase.json)

几何修复已生成原5mm目标，但首个有效S的native预测仍全程四足承重。D3确实实施24次独立B/S搜索、安装7次S方向候选，仍未形成持续推进。两臂held参考真正过期0；后续存在反馈尖峰、clip/QP与PD请求超界，最初下降早于这些明显执行改写，不能用晚段极值替代根因分离。剩余问题是几何、承载力与控制输出协调及接触切换闭环稳定性；本轮不继续叠加控制改动。

本轮两启动、零技术物理重启、12801 full rollout、2472984 transition FD、538266 risk FD、61497物理子步、3.640 CPU核时；新增PPO/donor/GPU物理0。C的MaterialSupportHeight误差已定位为Agent XML目标高度序列化精度，修后8.88e−16，原排序不变。原件与两个全时段视频已由D57直传并核验；见phase/delivery，不重复启动。下文W51及“W52下一步”均保留为形成时历史。

## 1. 长期目标、真实控制链与当前阶段

主线是**面向重复踩踏的四足机器人地面记忆辨识与跨步协同控制**：同一世界地块被前足、后足和返程重复加载，利用材料持久状态，让未来支撑需求影响当前加载、落点和重访。首先建立原 Go2/TM02 合法稳定完整往返，再研究材料历史的因果决策收益、泛化、轻量化以及 Go2/NX 实机。

现有执行链：因果 PANN → 658D PPO 观察 → 2 Hz scalar cap（.18–.25 m/s）→ 逻辑50 Hz全身MPC → 500 Hz时变关节目标 → PD60/5 → 5000 Hz原物理。PPO确实曾学习，当前评价固定权重；observer和材料历史继续因果更新。逻辑频率不是墙钟实时。

候选仍未沿每条未来动作完整传播世界材料内态；scalar cap没有直接控制卸载、支撑力分配、落脚或快速关节反射的权限。因此稳定原控制、有效跨足/返程曝光、候选内材料历史、RAW/PANN/冻结或打乱历史对照、泛化和实时，是相关但不同的工作。现在尚未建立稳定材料往返，更没有可宣称的记忆因果收益。

| 历史阶段 | 保留的事实 | 不能据此声称 |
|---|---|---|
| 早期R1/R2/S0 | 材料辨识与学习的正负结果均留存 | 早期方向全部成功 |
| R8/S3刚地 | 原任务9/9，连续300s 3/3 | 当前材料/r39资格继承 |
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

学习状态：A/B共享祖先6545/update25；PANN唯一转移22929，另RAW256，R12 distinct23185。W44 A/B checkpoint-014737、resume ep00428/ep00417、全actor/critic/optimizer/norm/RNG、tail0/excluded145保留。W45–W51新增PPO均0/0；15249/59不是自动续训预算。

## 2. W51 最新实际结果

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

## 5. W52 下一步：主几何修复，次级并行搜索对照

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

紧凑Git先推，必要原件/视频D57增量Release直传，短写串行合并最新main、不强推，回执不明先查远端。核commit/URL/size/digest后结束，**不在用户电脑额外下载、复制、打包、镜像或归档**。服务器仍被引用的科研输入/checkpoint/resume/视频保留，不自动删除；只用已授权服务器网络，不自动借电脑VPN/Clash/SSH转发。旧固定截止/归档/历史当前规则均不覆盖本页和W52计划。
