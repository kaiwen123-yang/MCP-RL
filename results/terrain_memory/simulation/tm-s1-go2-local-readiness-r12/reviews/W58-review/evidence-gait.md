# W57 / W58 日志与视频独立复核（2026-11-26，上海时间）

审阅固定到 MCP-RL `ef500f31c7a96d94f83c9afb5c7a880864b447dc`。本审阅只读取仓库及公开 Release 原始文件，没有运行新物理或训练，没有使用用户自己的 SuperMimic 复现记录。

## 1. 目前可以确认什么

W57 已自然结束，full / blind 分别完成 8192 个独立事件 transition、32 次更新，旧 W44 speed actor 更新数为 0。两头各有 A/B 4096，去重重复 ID 为 0。full 有 2247 次 liftoff 实际改变，blind 1966；请求改变不等于所有请求都产生控制变化。PR 是 P1 与已经在 P0 上训练的 timing head 冻结组合；不能称为 P1+R 新联合训练。

最重要的新事实是：**W57 的冻结 PR 已独立通过 A/B 原协议 E300→E600。现在不只有旧 r55/r60 的 600 秒基线。**

| PR 原协议 E600 | A | B |
|---|---:|---:|
| 物理观测 | 600 s | 600 s |
| 物理样本 | 3,000,000 | 3,000,000 |
| 首次到家稳定 | 466.128 s | 469.982 s |
| 随后保持到观测结束 | 133.872 s | 130.018 s |
| 全程最低 body height | 0.283007 m | 0.283007 m |
| 全程最大绝对 roll/pitch | 1.593882° | 1.658460° |
| 非足地面力峰值 | 0 | 0 |
| 运动段机械 CoT | 0.567804 | 0.568726 |
| 原定义 slip integral | 0.141453 s | 0.143071 s |

这支持“两个固定名义案例、原物理/材料/执行限制下长程不倒、能完成路线并停稳”的稳定基线。它不能单独证明多地形鲁棒性、最佳步态、材料历史机制有效或学习带来稳定性能提升。

### 原 E60 的冻结比较

所有 B/P/R/PR/blind 都通过各自 A/B 原 E30 与 E60。以下是相对同 case B 的百分比差，数值越负越好：

| 臂 | A slip integral | A moving CoT | B slip integral | B moving CoT |
|---|---:|---:|---:|---:|
| P | −3.153% | −0.129% | −2.574% | −0.422% |
| R | −5.284% | −0.473% | −2.640% | −0.630% |
| PR | +4.577% | −0.310% | −1.867% | −0.896% |
| blind | −2.576% | −0.354% | −0.896% | −0.458% |

单个模块有小幅描述性改善，组合未呈现一致叠加增益。各臂行走距离近似一致（差约0.25%以内），但只有两个固定名义 case 和一个 timing learning seed，不是统计显著性证明。

### W57 原 v1 机制

主矩阵 16 个机制任务以及 blind 的 4 个任务全部完成观测并停稳，但全部没有达到原定 12 个严格目标；不能把“试验完整结束”写成“机制验证成功”。PR 相对 B：A-short slip +0.630%、A-recovered +2.475%、B-short +5.661%、B-recovered +2.957%。四组 moving CoT 分别 +0.377%、−0.610%、−0.611%、−1.913%。R 相对 blind 也有正负，未得到稳定材料信息收益。

## 2. W58 的步态实验实际解决了什么

W58 `index.json` 已 finished、56/56 attempts completed、running=[]，选择 **G0 / M1**，`joint_eligible=false`、`joint_distinct={}`，无 W58 long attempt。故尚无 W58 新 joint PPO 训练结果。

G1 自己的 A/B E30、E60 全部通过；减少触地后承重前短暂离地，并改善周期均匀性，但滑动/单位距离全面恶化，因此未晋升。

| 配对 | 承重前 off time / swing | 原 slip / distance | moving CoT |
|---|---:|---:|---:|
| A E30 | −66.684% | +12.888% | +1.427% |
| A E60 | −63.736% | +18.942% | +0.324% |
| B E30 | −66.373% | +12.563% | +0.330% |
| B E60 | −65.895% | +9.928% | −0.261% |

这里 off time 是所有已提交 leg swing 的总离地时间均值，包含零离地的落脚，不是“每一次弹跳都缩短了三分之二”。A/E30 material straight：G0 的平均 off 为29.040 ms/swing，G1为11.812 ms/swing；刚性直线为6.926→0.575 ms。材料段 leg period 中位值从约0.628–0.642 s变为0.564–0.568 s、CV约4.94–5.88%变为4.34–4.60%，但前20 ms冲击峰值的中位数51.412→66.575 N。由于步态与落脚数同时变化，这些是段内分布比较，不是严格同事件反事实。

### 独立原始 5 kHz 复算：G1 减轻了弹跳，但没有消除

下载并完整 SHA256 校验：

- `w58-g1-m1-g1-a-original-e30-a01-originals-001.zip`
- 296,146,265 bytes
- SHA256 `bddaaae7ab4e7d232cc8dcdca6f43c7e7b410322d84ad470eff6a1c7cd816c89`

独立读取 `physical-000.npz` 的150,000×85行及 `trajectory.npz` 15,001行，复算确认：原5 kHz时钟连续；min body height0.2830073309 m；max roll/pitch1.442587°；非足接触样本0；实际actuator力矩最高为原限54.3032%。

代表 FR 事件：

1. 已提交 touchdown=10.9320 s。
2. 实际首次>1 N在10.9192 s，提前12.8 ms；进入时vz=−0.134676 m/s。
3. 前20 ms峰值183.967698 N，随后速度反向。
4. 10.9234–10.9954 s连续72.0 ms无>1 N载荷；相对初触最大回弹1.112970 mm。
5. 11.0006–11.0008 s还存在0.2 ms阈值往返，总off=72.2 ms。
6. WBC10.920进入search，11.000进入stance，11.006才满足bearing。

因此“提前冲击→小幅回弹→再次接触→承重”仍存在。创新方向若增加可微执行世界模型，必须学落脚速度、冲击/回弹、承重延迟及实际耗能，不能只学平滑机身下一状态或优化离地时间这一项。

## 3. W58 P2 和 v2 机制：新的主要证据

Development16全部完整、0 passed；validation16全部完整、4 passed，四个通过都来自 validation B-short 四臂。full/private 的8个 case×history×split 配对，以下字段完全相同：到家时间、运动段所有量、材料段所有量、原slip integral、requested wait、history_action_evidence；所有history_action_evidence=false。这不是“等待更多训练就必然有效”的证据。

### 全部 384 个预声明目标的漏项分类

对32正式episode×12目标逐一读取原 `targets` ledger；保留全部分母，不改变旧判据：

| 集合 | realized | 无匹配事件 pair | 有匹配事件但精确足印交面积全为0 | 总数 |
|---|---:|---:|---:|---:|
| development short | 78 | 14 | 4 | 96 |
| development recovered | 48 | 28 | 20 | 96 |
| validation short | 88 | 8 | 0 | 96 |
| validation recovered | 44 | 32 | 20 | 96 |
| 合计 | 258 | 82 | 44 | 384 |

所有目标的body traversal已完成。44项已有满足条件的source→release→target时间链，但实际两个足印与同world-cell没有正面积交集。其余82项没有通过完整前驱/区域/腿序/同traversal/正释放间隔过滤的事件链。**不能仅据compact ledger把82项全部判为某一种原因**。没有目标单独因已记录candidate间隔不满足short/recovered而失败；recovered连续clear时长均满足6秒阈值。

此表是完整描述性分母，不是384个独立随机样本；多个臂实际轨迹和事件高度相同。

### 代表原始事件链定位

为避免下载整套多GB压缩包，按 HTTP Range 读取公开Release ZIP中央目录和目标member。ZIP CRC成功校验，给每个解压member计算SHA256；**这不声称重新校验了整个ZIP SHA**。源asset的expected digest保留在manifest。

B2/A/development-short 中：

- F02 / traversal3（15.426–20.524 s）：预设ROI内FR source新载/重载0，RR33；ROI外仍有8条合法FR→释放→RR链。目标区域与当前步态落脚相位错开。
- F07 / traversal13（64.678–69.484 s）：ROI内FR24、RR9，但FR格集合x=67–69与RR的x=72完全分离（2 cm网格）；ROI外仍有11条合法链。

所以这两个“无pair”并非没有真实跨脚加载发生，而是没有在被预声明的狭窄区域中形成所要求的同格加载链；走完body路线无法保证每次同足印重访。

B2/A/development-recovered代表也已读取原始事件：F02/F03/F10均能形成事件pair但精确足印交面积0；F05/F08在ROI内均有FR与RR加载，但world-cell集合无交集。F05有27/6条FR/RR载荷事件，F08有12/24；ROI外分别还有13/20条合法链。连续6s清空得到确认，但不能保证停留前后的自由步态脚印对齐。

### 对下一实验协议的具体建议

1. 保留W58旧12目标判据和阴性结果；不要扩大区域再追认旧试验通过。
2. 主路线继续报告完整任务性能、全部预声明落脚机会和从未形成pair的原因。把“成功走过”与“产生可辨识材料事件”分开计数。
3. 单独建立固定目标足印的跨脚机制任务，让source负载、完整释放、short/recovered等待、target重载由同一预声明脚印时序约束。两种history条件匹配初始机身/目标载荷/实际脚印/动作权限，准备动作和时间另计。它是独立机制验证，不冒充原路线资格。
4. 为事件机会额外记录source候选数、target候选数、区域内同格数、最近前驱被哪脚覆盖、release存在性、positive-free gap、精确共同面积与载荷大小，构成逐层分母。不要只记录最后成功pair。
5. 准入重点从“可找到一条微小正交面积记录”推进到预声明、可观测、足够载荷覆盖的可辨识事件；最小有效载荷/共同面积/信噪比应在新任务开发时确定并冻结，不能用来删旧失败。
6. W58 validation已参与这次设计，应作为后续开发资料；新算法另设未使用的episode/状态/路线/负载holdout。

## 4. 实际视频检查范围与可见结论

完整下载并逐SHA256核对8个mp4；均1280×720、30 fps、物理记录状态回放。提取并实际查看8×9=72个overview面板，以及两个接触近邻序列18面板（其中有重复时点）；另外查看过G1/A的9.7s单帧。没有声称逐帧看完全部视频。

- W57 PR A/B E600：0、7.5、35、100、200、300、400、500、599.9 s。
- W57 G0 baseline A/E30 与 W58 G1 A/E30：2、6、8.4、10.9、12、16、20、25、29.9 s。
- W58 G1 B/E60：2、8、12、18、25、32、40、50、59.9 s。
- W58 P2-full A/development-short、P2-full和P2-private B/validation-short：0、10、20、40、60、80、100、140、167.9 s。
- 接触近邻序列：G0/A 8.3000–8.5667s；G1/A 10.8667–11.1333s，均按30fps提取并放大机器人区域。两个序列是各自代表落脚，不是同一物理初态的反事实对照。

可见：长程PR中取样到的前进、材料段、转身、回程和终态均保持机身稳定；G1 A/B也保持直立并完成相应路线；P2 full/private B验证视频在对应关键时点呈现相同姿态/位置，与指标相同一致。**这些镜头不足以从肉眼定量识别1 mm回弹、0.2–4 ms冲击、真实接触载荷、CoT或历史因果性**。millisecond接触结论来自5kHz日志，不来自视频猜测。

视频不能支持“已得到自然优美的最佳步态”的判断。现在更准确的表述是：稳定性基线已经可靠得多，微观接触质量仍有可测代价，三模块联动的材料历史收益尚未成立。

## 5. 文件与复核入口

所有独立输出在 `本审查目录下的 evidence/（原始大包及完整视频见 source-manifest）`：

- `audit_evidence.py` / `audit-summary.json`：W57 E60/E600、学习计数、W58 G1、P2配对、G1原始复算。
- `w58-missing-targets.json`：384目标完整分类和每个miss明细。
- `audit_missing_events.py` / `representative-missing-event-audit.json` / `representative-recovered-event-audit.json`：代表无pair来源核对。
- `extract_remote_zip_members.py`：HTTP Range抽取脚本（明确whole-asset hash验证范围）。
- `video-manifest.json` / `dense-frame-manifest.json`：真实mp4 URL、SHA256、尺寸、采样时码。
- `g1-FR-contact-trace.png` / `g1-FR-contact-window.csv`：独立5kHz事件证据。
- `g0-g1-contact-sequences.jpg`及8个`*-overview.jpg`：实际查看的帧面板。

仓库主证据路径前缀：`results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/`。

关键来源：`w57/index.json`、`w57/dispatch/frozen-evaluations.json`、`w57/r/pilot02/training-actuals.json`、`w57/paired-mechanisms.json`、`w57/mechanics/w57-m1-pr-{a,b}-original-e600-a01.json`；`w58/index.json`、`w58/outcome.json`、`w58/g/selection.json`、`w58/g/{g0,g1}/`、`w58/{development,validation}-matrix.json`、各`w58/mechanics/`与`w58/attempts/`；`scripts/s1_w58_revisit.py`、`scripts/s1_w57_mechanism.py`、`experiments/terrain_memory/s1-w58-revisit-v2.json`。原始多GB资产只引用现有Release，不重新入Git。
