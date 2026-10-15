# πMPC、πⁿMPC与当前研究主线

日期：2026-10-15。入口：[主线的清楚说明](../../docs/mainline-in-plain-language-2026-10-15.md)。不训练、不性能测试。

## 本轮读取

| 论文 | 版本与覆盖 | 记录 |
|---|---|---|
| πMPC: A Parallel-in-horizon and Construction-free NMPC Solver | arXiv:2601.14414v2，2026-05-20；8页完整文字、25参考；关键图式核查，独立审读看完8页原图 | [理论含义](theory-implications.md)、[身份与成本](source-and-cost-audit.md) |
| Accelerating and Scaling MPC-Guided Reinforcement Learning for Humanoid Locomotion and Manipulation | arXiv:2606.05687v1，2026-06-04；8页全文、34参考与附录TableIV，8页原图 | [MPC-RL全文](mpc-rl-followup.md) |
| CUDA MPC: A GPU-Native Solver for Model Predictive Control | arXiv:2608.03051；仅p1–2背景/方法局部核查，不计全文 | [成本审计中的范围](source-and-cost-audit.md) |

πMPC主论文首稿为2026-01-20，v2作者增为8人；正式出版身份未由本轮正式刊会记录确认。作者主页/实验室页的状态与第三方ACC单页条目分开保留，不将8页稿称为已核正式ACC论文。

## 与主线的关系

- πMPC提供时域并行的求解结构；它本身没有RL。
- πⁿMPC把批量求解用于训练指导，已有实际人形机器人结果；部署移除MPC，没有学习当前policy的响应模型。
- 本研究要解释规划与执行能力如何相互改善，不能只把这两篇与另一个网络拼接。
- 长时域能算、预测可信、执行能力提高是三个不同问题。论文有多时域与墙钟对照，不能把更大N当作自动收益。

原始文件／逐页文字／渲染／coverage在忽略目录 `work/pimpc-2026-10-15/`。主研究者也完整读取前两篇16页；首个原始提取输出截断后已重新分批读取。没有独立重建全部引用证明、看视频、运行代码或做新实验。

结构化来源见[专题元数据](../../literature/supplements/pimpc-2026-10-15.json)。不改变v2主库100篇与MR编号。
