# 七组方法的全文对照与本轮新增阅读

日期：2026-10-14。当前研究结论见[完整研究方案v3](../../docs/compact-response-codesign-v3-2026-10-14.md)。本目录保存原创审读；原始PDF、逐页文本、渲染与覆盖证据在忽略目录 `work/seven-paper-audit-2026-10-14/`。不修改主库100篇名额与稳定MR编号。

## 本轮实际读了什么

| 方法组 | 全文版本与范围 | 笔记 |
|---|---|---|
| Costate-focused RL | arXiv:1711.05817v5，7页全部文字/参考，7页原图；无独立附录 | [完整审读](Costate-focused-RL.md) |
| TaskMet | NeurIPS2023正式15页含附录、独立补充2页；另逐页读arXiv v2的15页核对版本；workshop仅核首页 | [完整审读](TaskMet.md) |
| GPS | ICML2014正文9页＋独立文字附录1页；MPC-GPS2016作者版8页；不代表整个GPS系列 | [两篇全文审读](GPS.md) |

新增主文与补充合计42页，另15页为TaskMet版本核对，不能当作新增论文。本轮顺序读取全部提取文字，并复核关键原图、公式、表和算法。PDF文字提取可能存在阅读顺序或数学符号问题，因此保留逐页和图式核查范围；不称“逐字符零遗漏认证”。没有独立重建所有证明、运行论文代码或做新实验。

前轮四篇全文共48页，沿用可追溯记录：[PIP-Loco](../four-paper-audit-2026-10-14/PIP-Loco.md)、[ReCo](../four-paper-audit-2026-10-14/ReCo.md)、[SPlaTES](../four-paper-audit-2026-10-14/SPlaTES.md)、[Dynamic Tube MPC](../four-paper-audit-2026-10-14/Dynamic-Tube-MPC.md)。因此“七组”包括GPS的两篇具体论文，不把方法组数与论文篇数混用。

## 由全文改变的判断

- 同一costate更新策略与模型：Costate-focused RL已明确实现。
- 下游任务学习预测误差度量：TaskMet已有更一般的全矩阵双/三层方法。
- 优化器与网络双向一致，昂贵教师和廉价部署分工：GPS已有直接先例。
- 塑造可预测响应、策略相关宏观模型、规划与误差管共同优化：分别有ReCo、SPlaTES和Dynamic Tube MPC的直接基础。

因而，v2中的共享敏感信号不是可独立主张的创新。当前研究转为：在明确的在线算力预算下，共同改善执行能力和紧凑模型的可用性，并控制策略更新带来的重新辨识与规划成本。该问题值得研究，不等于其候选解已被证明原创。

## 理论与可行性材料

- [严格理论审计](theory-audit.md)：量词、配对增量、误差传递、有限候选决策界及其边界。
- [定向近邻查新](targeted-novelty-check.md)：最容易否定新颖性的额外工作；相关章节核查与完整阅读分别标记。
- [硬件与预算](hardware-budget.md)：用户确认资源、官方规格/兼容性和待实测项目。

上述附加材料不自动计入全文清单。结构化记录见[本轮补充元数据](../../literature/supplements/costate-taskmet-gps-2026-10-14.json)。
