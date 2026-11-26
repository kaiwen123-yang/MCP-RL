# W58 review / W59 handoff — 2026-11-26

固定来源：`ef500f31c7a96d94f83c9afb5c7a880864b447dc`。本次新增的是只读分析、原始日志复算、文献及后继合同，没有新的W59仿真/模型拟合/策略更新。

- [总审查](../../../../../../docs/S1_W58_REVIEW_20261126.md)
- [W59执行计划](../../../../../../docs/S1_W59_RESEARCH_PLAN.md)
- [Codex启动提示词](../../../../../../docs/S1_W59_START_PROMPT.md)
- [机器交接](handoff.json)
- [控制、材料、学习及资源审查笔记](control-and-learning.md)
- [日志、视频和机制分母详审](evidence-gait.md)
- [官方SuperMimic源码阅读记录](official-code-audit.md)
- [原件、视频与验证范围](evidence/source-manifest.json)

审查笔记保留形成时的分析建议，最终执行参数、预算、动作合成与同步语义以正式W59计划/机器合同为准；不能将笔记中的备选设计当成第二套调度合同。

## 核心实证

W57 full/blind各8192去重事件、32updates；PR A/B各自原E600完成十往返并停稳。W58 56attempts结束=53自然终局+3技术中断，选择G0/M1，joint训练0。P2八条full正式机制3413次非空private reset、0次winner change；32正式机制384目标中258实现、82无pair、44零足印交面积。

G1/A/E30独立5kHz复算证实早触冲击后仍有1.113mm/72ms回弹，虽然机身稳定。8段视频的关键取样与日志相符，视频30fps不用于辨认毫秒接触载荷。

## 可重算的小型证据

`evidence/audit_evidence.py`读取固定仓库中的最终结果，可选读取原始G1包。`audit_target_denominators.py`遍历全部384预声明目标，保留旧判据。`audit_missing_events.py`核查指定原始事件中的前驱、区域、脚序和释放链；`extract_remote_zip_members.py`仅取所需公开ZIP成员。

以下命令在仓库根目录运行，输出放任意独立临时目录；不会调用机器人仿真或训练。

```bash
python results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W58-review/evidence/audit_evidence.py --repo . --out /tmp/w58-review-audit.json
python results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W58-review/evidence/audit_target_denominators.py --repo . --out /tmp/w58-review-targets.json
```

完整G1复算还需要按`g1-raw-source-manifest.json`下载已发布的296,146,265-byte ZIP并验证SHA，解压后以`--raw-g1 <解压根目录>`传入。它需要其中`episode/physical-000.npz`、`episode/trajectory.npz`、model/context和对应control分块；不要把只有qpos/qvel的快照当完整输入。

代表v2原始事件可按已有`w58/delivery/<attempt>/manifest.json`或其等价发布manifest传给：

```bash
python results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W58-review/evidence/extract_remote_zip_members.py --manifest <该attempt发布manifest> --out <独立解压目录>
python results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W58-review/evidence/audit_missing_events.py --repo . --attempt <attempt-id> --events <独立解压目录>/episode/material-events.jsonl --out <独立输出.json>
```

`source-manifest.json`区分：8个完整视频和G1完整ZIP已重新SHA校验；两份v2事件仅验证所取成员CRC/SHA，没有重新校验完整ZIP。大ZIP/MP4不重复加入Git，不在用户电脑再次归档。

## 图与数据

- [PR A E600固定时点取样](evidence/pr-a-e600-overview.jpg)
- [G0/G1各自代表接触近邻序列](evidence/g0-g1-contact-sequences.jpg)（不是同初态反事实）
- [G1真实接触图](evidence/g1-FR-contact-trace.png)及[原始窗口CSV](evidence/g1-FR-contact-window.csv)
- [W57/W58数值重算](evidence/audit-summary.json)
- [384目标完整分母](evidence/w58-missing-targets.json)
- [short原事件复核](evidence/representative-missing-event-audit.json)与[recovered原事件复核](evidence/representative-recovered-event-audit.json)
- [视频时码](evidence/video-manifest.json)与[近邻帧时码](evidence/dense-frame-manifest.json)

阅读覆盖与路径库存见`evidence/repository-inventory.json`；10,904个tracked path不等于逐字阅读了全部历史原件。新计划是否执行，以未来独立W59运行证据为准。
