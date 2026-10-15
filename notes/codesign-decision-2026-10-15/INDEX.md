# 任务驱动闭环动力学共同设计：查新、可实现性与研究决策

日期：2026-10-15。科学问题已由用户确认：在输入受限、存在扰动和复杂执行动态的机器人上，共同设计对任务有用、实际能够实现、适合在线优化的闭环动力学与反馈策略，使规划与真实执行一起改善。结构固定为MPC规划、RL执行。

## 阅读与证据口径

本轮新增指定版本全文6篇，以专题补充编号CD01–CD06记录，不改主库100篇，不回收或分配MR编号。下载、全文阅读、证明核验和复现分开；6篇状态均为`fulltext_read_not_reproduced`。关键图式已按各篇笔记回到原图核对，不概括为全部页面均完成视觉核验。

MR-ARL读取24页arXiv v1，TAC正式16页版未取得；不能把预印本中的发现直接归于正式版。AAMAS正式10页全文与同一论文长版p10–29的补充阅读合并为一篇；长版19页附录不占第二个编号。全文范围、版本日期、作者、正式年份及源PDF哈希见[结构化补充元数据](../../literature/supplements/codesign-decision-2026-10-15.json)。原PDF只在忽略目录`work/codesign-decision-2026-10-15/`。

## 六篇新增全文

| 编号 | 论文与版本 | 直接作用 | 阅读笔记 |
|---|---|---|---|
| CD01 | Feedback Linearization for Uncertain Systems via Reinforcement Learning；首稿2019、ICRA2020，读v2共8页 | RL实现便于上层控制的参考动力学；坐标重新参数化风险 | [全文笔记](direct-feedback-linearization.md) |
| CD02 | MR-ARL；CDC2023前身、2024首稿、TAC2026正式记录，读v1共24页 | 可变任务最优参考、反馈适应和联合稳定已有先例 | [全文笔记](direct-mr-arl.md) |
| CD03 | Actuator Reality Shaping for Zero-Shot Sim-to-Real Robot Learning；2026预印本v2共15页 | 固定参考响应与经典反馈塑形的低复杂度替代 | [全文笔记](direct-actuator-shaping.md) |
| CD04 | Learning Tube-Certified Control using Robust Contraction Metrics；首稿2023，2025修订v3共15页 | 反馈、证书与状态/输入tube共同合成 | [全文笔记](abstraction-tube-certified.md) |
| CD05 | Transfer Learning for Control Systems via Neural Simulation Relations；2024 v1共14页 | 实现接口、近似模拟关系和网格证书代价 | [全文笔记](abstraction-neural-simulation.md) |
| CD06 | Composing Reinforcement Learning Policies, with Formal Guarantees；首稿2024、AAMAS2025正式10页＋长版补充 | 模型/策略共同学习、组合任务界、入口分布假设 | [全文与附录笔记](abstraction-composing-policies.md) |

## 综合判断、算法与理论

- [直接近邻查新与可实现性判断](direct-novelty.md)：哪些宽泛主张已有先例，真正值得检验的增量是什么。
- [跨领域近邻裁决](abstraction-verdict.md)：控制抽象、证书、局部到完整任务保证的条件与风险。
- [主baseline与冲刺期刊](baseline-and-venue.md)：SPlaTES官方源码、actor-only适配边界、ReCo/HIM角色，以及T-RO主目标。
- [联合梯度与优化结构核查](joint-gradient-audit.md)：联合更新的梯度、乘子与数据依赖；应以实际完成内容为准。
- [理论与MATLAB解析](theory-and-matlab.md)：数学问题、退化检查、解析/数值验证与尚未成立的主张。
- [完整研究计划与决策](../../docs/codesign-research-plan-2026-10-15.md)：综合上述证据后的主线、算法推进和阶段目标。

## 与既有阅读的连接

- [ReCo](../four-paper-audit-2026-10-14/ReCo.md)与[SPlaTES](../four-paper-audit-2026-10-14/SPlaTES.md)：本轮已读基础上的直接对照，不重复计入新增6篇。
- [此前直接近邻裁决](../../docs/direct-neighbor-verdict-2026-10-14.md)：保留双QP等旧支线的历史边界，不恢复为主线。
- [πMPC及MPC指导RL专题](../pimpc-2026-10-15/INDEX.md)：求解工具和训练接口参考，不代替共同设计机制。

本索引仅组织证据与研究产物；选择主baseline和目标期刊不表示已经复现成功、证明创新性或保证录用。
