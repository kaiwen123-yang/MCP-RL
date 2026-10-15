# 直接近邻全文与研究判断修正

日期：2026-10-14。用户要求继续读透Gros/Zanon、COCP与分段扰动反馈，核查双QP是否又回到参数化MPC、为何需要RL，以及理论怎样走到机器人。本阶段明确不训练。

## 当前结论

[完整研究判断](../../docs/direct-neighbor-verdict-2026-10-14.md)：v4仍属结构化MPC控制器参数／表达学习；双QP保留为特定条件下的技术备选，尚不足以承担原主线或三个原创点。原“共同学习执行能力，规划理解并利用能力”目标保留。RL在已知可微模型中不是必需；未知复杂执行中的价值与机器人桥接条件需要独立建立。

## 全文记录

| 论文／版本组 | 完整范围 | 笔记 |
|---|---|---|
| Gros/Zanon两篇 | 17＋16物理页，含2机构封面；正文参考/作者简介完整 | [Gros-Zanon](Gros-Zanon.md) |
| COCP | 正式13页＋官网26页长版，全部参考与扩展；版本内容有大量重复 | [COCP](COCP.md) |
| Segregated DF | ACC2008 6页，参考与附录完整，6页全部原图 | [分段反馈](Piecewise-feedback.md) |
| Generalized Decision Rules | 47页作者稿，参考/附录A/B完整；另读出版者7页companion网页，与作者稿附录重叠 | [分段反馈](Piecewise-feedback.md) |
| Vector-space approach | arXiv v3 28页全文与附录；替代此前仅v2方法片段的覆盖 | [基础与机器人桥接](Vector-space-and-robot-bridge.md) |
| Robust nonlinear MPC | arXiv v1 9页全文与附录；补齐此前方法片段 | [基础与机器人桥接](Vector-space-and-robot-bridge.md) |

以上为7篇论文，COCP另核一个长版本。物理页总数包含版本重复与机构封面，不当作独立新增研究量。出版者companion直接下载未取得可用本地PDF，只记录已读网页PDF内容，不计本地附件覆盖。

关键图式原图核查范围见各笔记。没有宣称全部外引证明独立核验、论文代码复现或新控制结果。所有原始PDF/逐页文本/渲染/coverage保存在忽略目录；公开材料只保留来源与原创分析。

结构化记录：[本轮全文状态](../../literature/supplements/direct-neighbor-full-2026-10-14.json)。主库100篇与MR编号不变，专题经典不占用额外主库名额。
