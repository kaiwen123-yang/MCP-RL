# W57 视频与原始记录：最小复现包

本目录记录已完成的只读审查，不含原始大文件，不启动仿真、训练或远端作业。

## 内容

| 文件 | 内容 |
|---|---|
| `video-and-raw-review.md` | 完整审查结论、证据身份、所见范围与限制 |
| `evidence-derived.json` | 初始/最终快照计数、1047旧事件重聚合、新P独立复算、资源与原文件SHA |
| `p-a-e30-independent-metrics.json` | 新P/A/E30的150000条5kHz物理行与15001条500Hz轨迹独立复算，包括20次反弹和代表事件WBC时间线 |
| `media-sources-and-timecodes.json` | 7条视频、1个ZIP的源URL/大小/SHA；视频ffprobe、抽样时码、裁剪区域与观察范围 |
| `B-P-R-material-cycle-comparison.jpg` | B/P/R ep10材料区9.0–9.6s，21张同镜头裁剪帧 |
| `P-A-E30-9.4-9.9-contact-detail.jpg` | 新P 16张连续视频帧；30fps不能判定毫秒冲击 |
| `P-A-E30-FR-rebound-9.65s.png` | 对应原始5kHz力/位置/速度及500Hz参考曲线 |
| `fetch_review_assets.py` | 下载本次使用的固定SHA公共资产并核验 |
| `recompute_p_a.py` | 从已核验ZIP解出所需小文件并重新计算新P指标 |
| `make_sheets.py` | 重新生成7张跨全时长采样概览图 |
| `make_gait_detail.py` | 重新生成两张视频细节图 |
| `plot_rebound.py` | 重新生成代表反弹的PNG和PDF科学图 |

初始证据基点为 `e2d38923a32b0ea4d622001e963789203529de0f`；最终状态基点为 `fd4ede8d46572d638b736d4538f4d9946326265d`（2026-11-23 19:20:56 Asia/Shanghai）。后者补全发布回执、更新执行中head和资源计数；没有新增原始资格、没有改变已下载的这些源资产。最新full B ep10录像虽已发布，本审查未继续下载或观看，不混入原来7条的视觉范围。

## 仅重算新P原始数据

依赖 Python 3、NumPy。以下命令从本目录运行；把 `W57_MEDIA_DIR` 指向仓库之外的临时分析目录。下载量315,561,014 bytes，解出的数据仍保留在该外部目录。

```bash
W57_MEDIA_DIR=/path/outside/repo/w57-review-media
python fetch_review_assets.py --media-dir "$W57_MEDIA_DIR" --only-raw
python recompute_p_a.py --media-dir "$W57_MEDIA_DIR" > "$W57_MEDIA_DIR/recompute.log"
```

预期核心结果：

- ZIP SHA256：`74061506a1f6e42efc469a705a3a3976ba36a023f32cb90bc075dac1f86e9915`。
- controller：`S5-w57-p-56b53ddc5b73`，case A，original E30。
- `physical_rows=150000`、`trajectory_rows=15001`、`physical_finite=true`；没有150001行。
- 非足接触最大计数0、非足受力最大模长0；最小高度0.28300733088458757m，max|roll/pitch|1.3704596377291256°。
- 同一限定范围内20次小force-off：FL6、FR11、RR3，均符合早触地、原WBC仍swing、无已确认承载、向下到向上速度反转。
- 代表FR事件：9.6518s首次受力，9.6554–9.7578s离力102.4ms，峰167.489676N，足底反弹最高1.793667mm。

`recompute_p_a.py` 读取旧记录，不调用MuJoCo，也不改变实验物理、评分阈值或计划。所有大文件均来自固定Release URL，已存在但SHA不匹配的本地文件会报错，不会被静默覆盖。

## 重新生成图像与视频采样表

再需要 FFmpeg/ffprobe、Pillow、Matplotlib，以及Linux常见的DejaVu Sans字体。下载本次全部8个资产共435,915,926 bytes。该命令只下载本审查7条视频与一个ZIP，不下载整个约17.9GB Release。

```bash
python fetch_review_assets.py --media-dir "$W57_MEDIA_DIR"
python make_sheets.py --media-dir "$W57_MEDIA_DIR"
python make_gait_detail.py --media-dir "$W57_MEDIA_DIR"
python plot_rebound.py --media-dir "$W57_MEDIA_DIR"
```

画图前需要先运行原始复算。`make_gait_detail.py` 使用明确的视频frame index；`make_sheets.py` 使用12个ffmpeg seek位置，时码含义见manifest。视频帧是已记录状态的回放；“已下载完整视频”不等于“逐帧人工审查所有帧”。本次总共审看122张提取帧，含重复时刻，覆盖84张概览、21张B/P/R对照、16张连续细节和1张全幅。

## 旧E600与机制证据的边界

对旧r55/r60四条E600，repo中的W57分析原先覆盖12,000,000物理行；本次重读其1047条 `impact-sequence-*.jsonl.gz` 事件产物并重新聚合、检查分类算法，没有再下载四条E600全部原始物理数据。机制geometry审计来自已归档的精确重算结果，不声称本次从全部材料原始场重做。相对源目录与SHA均见完整报告和派生JSON。

资格与视觉仍有限：已有旧r60长时baseline；新P此处仅A/E30。最终full8192/32updates head、blind最终head的冻结资格与对应视频不能由训练ep10或旧r60代替。
