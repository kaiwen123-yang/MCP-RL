# W46之后的文献依据与可执行迁移边界

审查日期2026-11-13，仓库结果基点15ee620b8a909246a12baf999c136c2be299655b。本文是文献与当前源码/证据的对应分析，不是新实验结果。执行以[W47计划](../../../../../../docs/S1_W47_STABILITY_PLAN.md)为准。

## 1. 三项新增一手依据

### Box-FDDP，2022

Mastalli等，**A Feasibility-Driven Approach to Control-Limited DDP**，Autonomous Robots 46，985–1005，DOI [10.1007/s10514-022-10061-w](https://doi.org/10.1007/s10514-022-10061-w)。[出版社全文](https://link.springer.com/article/10.1007/s10514-022-10061-w) · [作者PDF](https://cmastalli.github.io/publications/boxfddp22auro.pdf) · [Crocoddyl](https://github.com/loco-3d/crocoddyl)。重点§3、式14和17–23。

它通过动态缺口处理扩展可用初始化，缺口同时影响backward、forward和接纳，forward输入仍要投影。当前源码`fddp.hxx`/`box-fddp.hxx`也有对应实现。借鉴点是完整的可行性机制；适配边界是论文固定控制box与本项目状态相关PD目标角边界不同，接触和材料链仍须接入。仅保存旧状态数组没有实现这个算法。

### 统一MS-DDP，2023

Li、Yu、Zhang、Wensing，**A Unified Perspective on Multiple Shooting In Differential Dynamic Programming**，IROS2023，9978–9985，DOI [10.1109/IROS55552.2023.10342217](https://doi.org/10.1109/IROS55552.2023.10342217)。[原文](https://arxiv.org/pdf/2309.07872) · [作者代码](https://github.com/heli-sudoo/Multiple-Shooting-DDP)。重点§VI-B、§VII、Fig.4–9。

文中分离了状态保留、rollout与expected-change/接纳：一个数值例子仅修预期变化即解除停滞；四足跳跃中状态热启动帮助避免接触后的nominal发散，但普通trot/bound两者相当。硬件实验仍需应对模型失配。对本项目的推论是先查零端点和真实下降，再判断是否需要独立状态优化，不能承诺更换FDDP即可稳定。其四足模型与本项目MuJoCo/TM02并不相同。

### 结构化SQP，2025

Jordana、Kleff、Meduri、Carpentier、Mansard、Righetti，**Structure-Exploiting Sequential Quadratic Programming for Model-Predictive Control**，IEEE T-RO41，4960–4974，DOI [10.1109/TRO.2025.3595674](https://doi.org/10.1109/TRO.2025.3595674)。[作者出版目录](https://skleff1994.github.io/publications/) · [官方ICRA2026展示PDF](https://rasevents.org/uploads/documents/pdfviewer/35/b0/217364-5821.pdf) · [mim_solvers](https://github.com/machines-in-motion/mim_solvers)。展示年份不是论文年份；按2025 T-RO引用。重点§V-C、§VI-A、Table I。

作者分别比较默认FDDP、过滤接纳FDDP及线性rollout SQP，说明多重射击、接纳和rollout是不同影响因素。其可微性前提、解析导数和机械臂实验限制了直接迁移。对当前接触/投影问题，借鉴的是逐项分离与正确内部可行性评价，不是在W47同步替换动力学栈。

## 2. 与本仓库的对应推论

W46的540个正步都有有效native rollout，却没有任何真实J下降。constant B155/B158的alpha0也出现显著J差。标准iLQR若模型、完整初态、时标、实际输入与可行nominal相同，在alpha0、x=bar_x时应按节点复现轨迹，即便K更新。当前源码连续ActionImpl/节点ActionIndex、旧完整policy/新K-H中心都需精确对齐；“换K”不能单独充当原因解释。

有三类不同的问题：好移动候选根本没被提出；动作种子从当前x0重滚后失去有用接触/状态；候选本身有用却被当前J0或Jref排斥。不同结果应导向执行修复、初始化/状态保留、或目标修改。新文献没有取消这个辨别步骤。

未来若转向真正FDDP/MS/SQP，必须包括独立shooting states、流形动态defect、backward修正、forward更新、预期下降/merit或filter、原输入约束与完整部署评分。开放动态缺口时的内部接纳不能照搬当前可行rollout的J-only比较；最终原Go2/TM02任务判定仍保持。当前W47不默认实施整个架构。

## 3. R8种子有哪些、没有哪些

R8/S3成功记录提供500Hz实际qpos/qvel/q_target及接触、feet、goal/phase等；prediction分块另存plan_states/actions/times。依据是[原R8索引](../../../tm-s1-go2-wb-support-r8/index.json)与仓库`run_s1_go2_wb_baseline.py`记录schema。本轮未重读R8 payload，不把manifest当数组内容核验。

固定可用来源候选：`project-s3-formal-v1-single-yaw_plus5-trajectory.npz`，发布size **28,100,505B**，SHA256 **079f57e1ebad260c9cdac842dc089012e29cb46992ab64d2c31cd9d380b61625**。W47使用前在服务器现有原件上核size/hash/schema，缺失则明确停止该分支，不在电脑下载或换一个更好看的seed。

R8缺完整K/H和pre-call私有状态，不能按完整反馈策略部署。R8 predictor的2×1ms、3cm步态与当前10×.2ms native/coldstart、.005m C2步态不同。保留原任务配置，把旧已执行u作为初始化猜测，在当前模型重新评分；旧x仅用于相位/初差/defect诊断。

R8的phase数组是mission阶段枚举，原生gait phase需从已保存cadence设置和对应GetPhase重建；按同elapsed最近2ms节点、任务阶段和重建gait phase确定片段，不拉伸时间、不搜有利段；只对机器人参考做xy/yaw锚定，四元数及自由关节速度坐标必须正确处理。世界地块、材料历史、observer、goal/ground均使用当前context，不随donor搬动。

种子K可为0，H来自当前incumbent有效SPD metric并按当前时间网格对齐，仍调用原Clamp/H-QP，不以metric_valid=false绕过边界。随后当前native rollout从真实x0生成可执行states，正常backward得到新K/H。两个家庭共用从原incumbent冻结的ubar/F/R，保留current incumbent/current best/donor联合候选同J裁决；不能给donor自己的零代价参考。

donor动态缺口应逐节点从变换后的旧x计算当前f，再与下一旧x作流形差；不能把当前单射击自产x的近零缺口当作证据。详细计划为此单列最多700 held单步转移。部署规则固定为current N搜索停滞且源匹配时追加一个donor家庭，每epoch最多两solve；离线比较必须使用相同入口。

## 4. 当前可以得到的决策

| 有限结果 | 支持的下一方向 |
|---|---|
| 零端点首分叉源于时标/表示/执行不一致 | 修具体语义，保持r34真实下降及原物理 |
| 零端点一致，预测下降仍系统性失真 | 查接触/活动面下的有效导数与方向 |
| donor在当前模型可执行并被同J接纳、恢复和前移有用 | 支持候选/初始化不足，才进入原E30 |
| donor局部转移尚可，长链重滚破坏运动 | 单独论证真正状态保留/FDDP/MS/SQP |
| donor单步就严重不匹配 | 在当前模型生成可行参考，不能保留不可行x冒称成功 |
| J0或Jref排斥有益运动 | 分项证实目标冲突，另立单机制修改，当前不扫权重 |
| 两路无辨别性收益 | 如实交有限负结果，不重复同配置E30 |

## 5. 与此前文献的连续性

HiLQR、HOUND/接触隐式控制、IDTO、ROM引导、ProxDDP等已在历史审查讨论，不重新计为新增成果。ProxDDP本轮补查了[作者项目](https://simple-robotics.github.io/publications/proxddp-tro-2025/)及[官方Solo示例](https://github.com/Simple-Robotics/aligator/blob/main/examples/solo_jump.py)：其输入/接触安排与当前任务不同，不能直接替换执行器。更近条目有全文获取限制，未将未读正文的题名当作当前错误已获证明的依据。

本轮应先让可执行轨迹、优化方向和实际接纳一致，再继续原任务、有效材料历史曝光与因果对照。稳定性修复不替代最终的跨足/返程材料记忆研究。
