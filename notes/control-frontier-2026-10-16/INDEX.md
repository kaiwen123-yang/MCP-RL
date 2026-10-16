# 2026控制与学习专题：全文阅读、补查和耦合机制

更新：2026-10-16（Asia/Shanghai）。本轮遵循既定科学问题，保持MPC规划、RL执行；不启动训练。结论入口：[控制如何帮助RL学出可规划能力](../../docs/control-informed-coupling-2026-10-16.md)。

## 本轮实际完成的阅读

| 论文 | 读取范围 | 正式版边界 | 原创笔记与证据 |
|---|---|---|---|
| Physics filtering favors the generalization of robot learning | 正式正文11页、科学补充27页全文；关键公式/全部编号图原页；官方代码关键路径 | 正式版全文及补充已读；未观看视频/运行策略，未独立复现 | [笔记](PhyFilter.md) · [证据JSON](PhyFilter-evidence.json) |
| Getting robots back on track by reconstituting control in unexpected situations with online learning | 正式正文14页、科学补充25页、公开审稿材料19页、视频说明1页全文；关键公式图表原页 | 正式版全文及补充已读；未看视频/重算数据，未独立复现 | [笔记](FLAIR.md) · [证据JSON](FLAIR-evidence.json) |
| Learning stable Koopman embeddings for identification and control | arXiv2401.08153v2的14页全文及全部内置证明；核心图式原页；前稿定位 | **2026正式版PDF未取得**，正式版仅核元数据及部分修订；不能称三篇正式版全部读完 | [笔记](Stable-Koopman.md) · [证据JSON](Stable-Koopman-evidence.json) |

全文读取不等于每条定理独立证明通过，也不等于复现。逐篇笔记把作者主张、作者未来工作、阅读者推断、版本差异和未核事项分开。全文/补充阅读状态已回写[原候选登记](../../literature/supplements/control-2026-screening-2026-10-16.json)，保留此前只筛选的历史状态。

## 补查与主线

- [2026 TAC/Automatica定向补查](TAC-Automatica-screening.md)：五项TAC、五项Automatica；含既有近邻复核，实际阅读范围分项列明，**不称又精读10篇**，不称穷尽2026全年。
- [结构化补充登记](../../literature/supplements/control-frontier-2026-10-16.json)：正式年份、首稿/扩展关系、DOI、实际读取范围和未核字段。
- [研究机制报告](../../docs/control-informed-coupling-2026-10-16.md)：对SPlaTES/ReCo的实质比较、学习保留的行为自由度、实际响应与目标响应的局部实现关系、输入资源缺口和强对照。

本轮最重要的区分是：更准确地认识/调用已有能力，与RL真的改变可用反馈能力。PhyFilter支持“控制改善学习过程”的机制；FLAIR提供“固定低层命令适配”的强替代解释；Stable Koopman支持“控制性质进入可学习结构”的方向。没有一篇直接替本项目完成任务响应与RL策略共同设计。

## 接续依据

已读取GitHub main并fetch核对，本轮起点本地/远端一致于`cde6dbc`。重读`docs/CURRENT_STATE.md`、已确认科学问题报告、SPlaTES/ReCo全文笔记以及近期负结果；另读项目两段前序对话的相关用户确认，核对近期文献优先、保持科学问题、MPC规划/RL执行、SPlaTES只是底座、及时同步和不扩训的要求。公开仓库只保存这些研究决策的归纳，不提交聊天原文、应用状态或本机身份信息。

全局记忆注册表未找到本项目专门条目，本次接续依据是项目内当前状态/研究文档和已核前序对话；没有写入全局记忆。用户没有要求更新全局记忆。

原始PDF、抽取全文、图片和外部代码只在忽略目录`.research-cache/`或`work/`，逐篇证据记录相对路径与hash供本机复查；这些文件不包含在GitHub公开提交中。主库v2及MR编号不变。
