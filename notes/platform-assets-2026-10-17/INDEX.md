# 真实平台资产核查索引

日期：2026-10-17。主结论见[整轮报告](../../docs/platform-assets-and-research-gap-2026-10-17.md)。本轮只做资产/源码读取和固定输入推理，不是机器人任务验证。

| 文件 | 作用 |
|---|---|
| [local-inventory.md](local-inventory.md) | 本机有界搜索、既有材料分类及未定位项 |
| [evaluation-readiness.json](evaluation-readiness.json) | 已允许有界冻结评价；当前所查运行环境/模型/任务缺件 |
| [go2-public-audit.md](go2-public-audit.md) | 官方与社区来源、两策略接口/许可/模型边界 |
| [go2-sources.json](go2-sources.json) | Go2固定commit、文件hash、源码定位、静态子审计成本 |
| [piper-public-audit.md](piper-public-audit.md) | H与普通型区别、控制模式、公开IL资产及数据语义 |
| [piper-sources.json](piper-sources.json) | Piper/HF固定版本、下载范围、静态文件hash |
| [strong-baseline-and-gap.md](strong-baseline-and-gap.md) | 完整强基线、单一科学问题/两个平台候选、最小记录与停止条件 |
| [offline-check-protocol.md](offline-check-protocol.md) | 推理前固定的两个权重、三输入及12次forward预算 |
| [inspect_public_go2_assets.py](../../scripts/inspect_public_go2_assets.py) | 原创CPU文件/固定输入核验，不导入外部部署代码 |
| [go2-fixed-input-results.json](go2-fixed-input-results.json) | 全部合成输入/输出、结构、前后hash、实际成本 |

两个子审计的0模型调用只描述各自静态工作。整轮另含root固定输入核验的12次Go2顶层调用；没有训练、环境或硬件。实际下载内容在`work/platform-assets-2026-10-17/`，由Git忽略，不随本文发布。
