# Codex 启动提示词：W55 稳定行走与 PANN 实载检验

将以下完整提示词交给执行端 Codex。先拉取包含本文件的最新 main。本提交只完成审查和计划，不表示 W55 机器人控制已经实施或成功。

~~~text
继续 kaiwen123-yang/MCP-RL 的 S1，执行 W55。直接完成实现、必要验证、原任务、同轮有证据的修订和 GitHub 交付；不要停在重新写计划、只改一个小点或固定两次失败后结束。

先读根 AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md、docs/S1_W55_STABILITY_PLAN.md、results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/current.json，以及该结果目录 reviews/W54-review/ 下的 README.md、handoff.json、control.md、evidence.md、pann.md、pann-replay-summary.json、literature.md。实际 W54 结果基点为 1ae25272c86f551f8c6b3a65073bc05b24e3d8f6，以包含 W55 计划的最新 main 为实施起点。先核本地改动、服务器具体 job/tmux/PID/物理进度及可用资源；已有同身份有效任务就续接，不重复启动、不覆盖用户改动、不宽前缀 kill、不动其他项目。

W54 已完成9次不同版本原E30、0次通过：6次物理/材料失败、2次控制器中断、1次完整30s但停步未完成任务。F/r50不等于稳定走30s；F/r51和r52均在4.880s开始低于.28m，先于真实材料力约.44s。N/r48到x=1.312010m、最大应变.013724未越域，但8.3568s躯干触地；12次跨足单元事件首次7.555s，已经晚于失稳迹象。seed2/E60/E300/E600未运行，新PPO/donor=0。不要重复已确定的负版本或借旧资格。

并行完成三条线，最多两条重机器人物理任务；谁就绪谁先跑，不等总报告、上传或其他线全部完成：

F为首要稳定性修复。从r52出发，修MuJoCo3.2.3平移mj_jacDot的官方已知缺项。控制侧正确项为同一世界系omega_body×v_point：四足球心分别补，各body质心分别补后按质量平均得到COM drift，rolling继承修正后的球心项。angular Jdot不补，N的方向差分不补，防止双重修正。官方fix为de48f4178f971d945fc135afb42699b8ca6bec8e，3.3.1已修；本次保持3.2.3 plant，不整包升级。当前runtime已有mj_forward，不能把问题写成补一次forward。已保存数组中心差分证据r52为2.595→.009986m/s²、r51为2.566→.009796，只证明运动学缺项，不等于修后已经成功。

一次接全helper、runtime、backend、spec、记录和新身份。保留r52已修的实际接触/承载、actual touchdown anchor、未来落点、MPC优化身体运动、PD60/5、2ms held-input、dry friction、preload与真实内接摩擦方锥。只做本公式所需的同状态运动学核对和正常构建，直接同W44 A/seed1原E30，不另加D8、旧9+3或一长串smoke。F不等待N/PANN。修后仍失败就读新轨迹的最早偏差；只有新证据仍指向力/变形/身体运动不一致时，再在本轮补因果顺应接触层，不能在首个修复前同时调一堆gain或改软plant。

N从r48改成支撑条件下的完整E。现有Jz M^-1 S^T忽略其他脚反力，E会改动正在承载的腿。用当前实测/预测支撑下的浮基联合投影(qacc,tau,lambda_support)或等价消元，统一动力学、反力单边/真实摩擦、原PD输入/总力矩/hold可达界；硬地用实际球足接触，软材料法向用同一因果顺应状态，不硬钉未变形平面。制动目标只针对未承载下降足，优先维持支撑和身体可实现性，不可达要报告，不能牺牲支撑假装达标。

E须在真实500Hz、每个native rollout与FD中完全同源，保持原2ms hold与5kHz plant。当前支撑及因果材料状态若影响E输出，须作为显式输入或预测状态声明，不能仍宣称只依赖q/v/u_raw。增加的记忆必须进入可克隆状态/转移或一致冻结输入，不能读取隐藏真q。将单足-1mm代理扩展为统一全身可行性：原允许接触、姿态/高度与下降速度恢复余量、关节/执行界、因果材料域，同时生成完整同horizon重评的support-aware恢复候选或可行warm-start。所有候选和shifted nominal同状态、同E、同检查、同排序。只抬高body-height阈值会把10个低身姿候选全部拒绝，不是完整修复。无可行候选时如实记录controller_infeasible和首因，不能沿旧unsafe policy或放宽硬界；随后按新证据修生成器继续本轮。

C为PANN/材料部署范围与有效GPU工作。真实plant是解析ReferenceColumns，PANN只提供因果恢复特征，当前使用B_random1000--ideal，不是历史偏快一倍的noisy模型。权重冻结不等于世界地块e/q冻结。先复用已完成的F/r52三地块、N/r48六地块、两种时钟的独立自由积分及训练数据审计；当前有限路径不支持先大规模重训，按计划扩展独立地块/修后自然路径，分离本构误差、2ms观测采样误差、接触桥接和控制未使用状态。不得注入真q或把位移被指定的e11吻合当准确性证明。当前即时PANN置零cap差极小，不等于已完成闭环消融。

已测N48两个高速地块恢复峰尺度偏差约19–22%，C在新增独立历史上确认后，可直接定向GPU训练一套恢复更准的材料候选，不等F/N结束或下一轮许可；当前cap影响小与离线材料改进价值分开报告，不能宣称训练就解决摔倒；按完整episode、脚加载事件族及相似完整历史分训练与保留集，不能只按相邻world key拆分（相邻格可有相同应变历史）；九格只作已知诊断集，另设未见路径条件，使用部署一致的自由积分目标，保留能量/耗散结构、原域及旧F1–F6/恢复验证。合成历史明确标为合成，W54没有真实返程数据。若主要是采样漏事件先修时间聚合，不以重训吞掉时钟问题。新模型先离线保留集验证，正式任务内不热替换；确有动作影响证据才在释放的F/N槽以新身份原E30评估，不开第三物理族、不借资格。PPO样本/更新/donor保持0。可做所有查询地块e=q=0时的等价零恢复快捷路径，有历史释放地块不能误缓存。W54完整GPU force-horizon的16状态/224个力区间已完成，不重复凑显存，不将批次吞吐冒充在线时延。

F/N实现就绪后各用W44 A/seed1 checkpoint-014737、原normalizer/确定actor、完整t=0初态与virgin地面直接原E30。首套通过且全日志/视频支持正常行走的内核立即冻结，复用同身份seed1前序继续E60，槽位释放后开同内核W44 B/seed2 E30。各case独立原E30→E60→E300→E600，原路线/材料/3m远端/回家稳定/270及540s内5及10合法往返不变。代码、权重、seed、初态、workers与合同改变不借资格；blocked/unrun不算失败。已运行另一研发任务自然结束。

现场25核初期按N≤10workers、F≤4核、C/数据≤8核、系统与其他≥3核。数学线程库1，构建合计先-j6且计入共享配额；独立build/输出/进程树。两N正式case若首次E30冻结10workers，按10+10+2+3让C缩减CPU，不中途改身份workers。F的小QP不会因affinity变大自动满核。记录真正任务交集、CPU核时/热段/IO、GPU批量/编译/传输/峰值；不制造无意义重复任务占满96GB。

所有新后台deadline=null、wall_deadline_enabled=false，省略deadline-unix，无固定墙钟完成时刻、到点kill或对话结束kill。原物理任务长度、失败条件、明确数值/资源错误和用户停止有效。running如实记录。保留W44 A/B完整checkpoint/resume与服务器被引用原件。

保留全程500Hz轨迹、5kHz接触/足速/力矩、材料双时钟、已安装完整策略/关键候选和全视频，增加本轮明确需要的drift、支撑投影与恢复候选证据。只做改动必要验证，不新增全仓hash/mock/shape/import/smoke或重复旧探针；真实spec/build/runtime/backend/actor/hold/终止/记录/协议必须一起接通，不能只改名字。失败后同轮定位、改有证据的问题、新身份再推进，不原样重跑、不网格调参、不换seed追成功；没有可定位新改动则报告具体阻塞。

紧凑代码与结果及时串行整合最新main推GitHub，不强推。必要原件和全视频D57直接增量上传Release，回执不明先核远端再续传。核commit/URL/size/digest后交付，不在我的电脑额外下载、复制、镜像、打包或归档，不自动借电脑VPN/Clash/SSH转发。同步AGENTS、README、项目记忆、current.json与配置next_authorized_plan；实际结果建独立w55/index.json、phase和per-job合同，旧大index仅作历史。

最终分别报告：实现/正常行走/最高原资格/真实材料往返/记忆因果收益、PANN本构与采样误差、GPU实际收益、失败和未运行、真实并行、GitHub提交与续接点。公式正确、离线误差小、模型训练完成和机器人任务成功分别如实命名。
~~~
