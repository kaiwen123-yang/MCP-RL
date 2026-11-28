# W59 视频证据范围、来源与时钟说明

固定审查快照：`c8378beae4be371eb4f104782b75e5d295b6f59a`。审查日期：2026-11-28。机器可读逐项记录见 `video-evidence.json`，科学指标与任务收据见 `evidence_gait.json`。

## 1. 谁看了什么，计数如何得到

根代理实际解码并观察了124个取样帧。步态子代理记录根代理报告的视觉范围，并独立读取三份原始phase manifest、ffprobe输出、下载资产校验和与E600任务原始收据。本说明不把子代理的元数据阅读算作额外视频观看。

|取样对象|取样位置|帧数|本地视觉证据|
|---|---|---:|---|
|5901B E600全片概览|源视频seek：0、2、5.5、9、16.5、20、26、30、40、60、100、130、150、300、450、599 s|16|`visual/e600_overview.png`|
|5901A E60六阶段|每个阶段4帧，全部6阶段|24|`visual/a_seed5901_phases_0.png`、`visual/a_seed5901_phases_1.png`|
|5902A E60六阶段|每个阶段4帧，全部6阶段|24|`visual/a_seed5902_phases_0.png`、`visual/a_seed5902_phases_1.png`|
|E600材料步态序列|8.4+0.08×i s，i=0…11，即8.40…9.28 s|12|`visual/material_stride_full.png`、`visual/material_stride_sequence.png`|
|E600反向行进序列|25.4+0.08×i s，i=0…11，即25.40…26.28 s|12|`visual/reverse_stride_full.png`、`visual/reverse_stride_sequence.png`|
|E600收尾序列|468.5+0.2×i s，i=0…11，即468.50…470.70 s|12|`visual/completion_full.png`、`visual/completion_sequence.png`|
|B3 A E60六阶段|拼接片seek=2×i+[0.2,0.7,1.2,1.7] s，i=0…5；不是按实际分段PTS起点取相对偏移|24|`visual/b3_a_phases.jpg`|
|合计||124||

同一序列的full图与sequence图不是两套不同帧，因此没有重复计数。124是视觉采样数量，不是独立实验、种子、步态周期或统计样本数。两份PPO A E60每阶段4帧的具体seek偏移若未另登记，不凭B3设置推断；目前保存的图和原manifest仍可追溯到对应阶段。

本次发布的视觉证据位于同目录 [visual/](visual/)；JSON保留审查临时路径、各文件字节数与SHA256，仓库中的同名内容可据此核对。

## 2. 原始资产与完整性

五个视频/视频包均已有本地副本，重新计算的SHA256全部与登记值一致；其中新补核的B3 A E60全片仅下载、核hash与ffprobe元数据，没有新增视觉观察，视觉帧数仍为124。没有重新下载全量175GB原始资产。另核的原始证据是E600最后一个原件ZIP，详见下文。

|资产|字节数|SHA256|
|---|---:|---|
|[w59-ppo-full-fallback-5901-a-original-e60-a01-gait-excerpts-originals-001.zip](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-w59/w59-ppo-full-fallback-5901-a-original-e60-a01-gait-excerpts-originals-001.zip)|1,383,622|`19b90748483915bc5b653f0d96c6f8586d95bba8a471b13c270fcb528d9fe745`|
|[w59-ppo-full-fallback-5902-a-original-e60-a01-gait-excerpts-originals-001.zip](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-w59/w59-ppo-full-fallback-5902-a-original-e60-a01-gait-excerpts-originals-001.zip)|1,389,944|`ff2f36a9d6bb1c7b0a8cbb81f8bd4c3f1010063546fb1a6f56c118b791d26078`|
|[w59-ppo-full-fallback-5901-b-original-e600-a01-full.mp4](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-w59/w59-ppo-full-fallback-5901-b-original-e600-a01-full.mp4)|181,841,351|`7dbc51765771ae87e821bdfc96f97cfd0dc8f1e61002a758af43da7ddf1e27af`|
|[w59-b3-a-original-e60-a01-gait-excerpts-originals-001.zip](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-w59/w59-b3-a-original-e60-a01-gait-excerpts-originals-001.zip)|1,385,417|`bc3a1015210ee87d92ff087b3e6687ac269af9cbf992b1d1f69774904ec7b1a7`|
|[w59-b3-a-original-e60-a01-full.mp4](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-w59/w59-b3-a-original-e60-a01-full.mp4)|14,482,890|`30f4ae38560c6f9fcf93a9226e87d76ca18ff59effc721d86242ea65f2bee133`|

[E600原件ZIP-021](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-w59/w59-ppo-full-fallback-5901-b-original-e600-a01-originals-021.zip)：19,168,406字节，完整SHA256 `ed33470316fbf4eefe013b39245a183cd6476dcc04417901379b65ab7f41c1d2`。该包的receipt、learner-result与actor-events用于对齐任务完成与保持观察，不能把它当作其余20个原件包也已重新校验的证据。

## 3. 三份A E60 phase manifest的精确映射

下表的每个单元为“日志事件时间 / 源视频帧区间”。帧区间按左闭右开 `[first_frame,end_frame_exclusive)`，脚本/manifest登记的源索引率为30fps。B3全片已独立实测30fps；两份PPO A E60全片本次尚未独立ffprobe，不把登记值写成全都实测。event_s是物理日志事件时间；所选片段一般从该事件前约0.5s开始，覆盖约2s。它不是成片中该阶段的播放秒数。

|事件|B3 A E60|5901 A E60|5902 A E60|
|---|---|---|---|
|start|2.002000 s / [46,106)|2.002000 s / [46,106)|2.002000 s / [46,106)|
|material_entry|5.395800 s / [147,207)|5.376000 s / [147,207)|5.413000 s / [148,208)|
|straight|9.208000 s / [262,322)|9.230000 s / [262,322)|9.360000 s / [266,326)|
|turn|16.194000 s / [471,531)|16.472000 s / [480,540)|16.530000 s / [481,541)|
|cross_foot_reload|29.553400 s / [872,932)|30.753600 s / [908,968)|26.434200 s / [779,839)|
|completed_home_hold|39.778000 s / [1179,1239)|40.110000 s / [1189,1249)|40.210000 s / [1192,1252)|

原manifest来自各自 `*-gait-excerpts-originals-001.zip` 中同名JSON。记录与剪辑脚本可交叉核对：[clip_s1_w59_episode.py](https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/scripts/clip_s1_w59_episode.py)。六个标签的定义：

- start：第一次phase1。
- material_entry：第一次非release材料加载事件且normal_N>1。
- straight：第一次phase1且base x≥1.5m。
- turn：第一次phase2。
- cross_foot_reload：优先选回程开始之后的首个positive_cross_foot_overlap；没有合格回程候选时取首个正交集事件。
- completed_home_hold：第一次phase5，即该E60任务完成。不能拿这个短任务时刻替代E600十往返的完成时刻。

这些阶段标签用于定位观察，并不说明三个控制器在完全相同材料cell、足对或动作历史上受试。尤其cross_foot_reload实际选择如下：

|episode|cell|足对|source s|release s|target s|gap s|exact overlap m²|
|---|---|---|---:|---:|---:|---:|---:|
|w59-b3-a-original-e60-a01|[88, 8]|FR→RR|28.344200|28.346400|29.553400|1.207000|6.335220261e-05|
|w59-ppo-full-fallback-5901-a-original-e60-a01|[79, 8]|FR→RR|29.542400|29.546600|30.753600|1.207000|1.271310168e-06|
|w59-ppo-full-fallback-5902-a-original-e60-a01|[106, -9]|RR→FL|12.922400|13.258200|26.434200|13.176000|8.483489188e-06|

5902示例是RR→FL，而B3/5901示例是FR→RR；它们的cell和gap也不同。因此这些clip可以佐证存在相应重访行为，不是严格的同事件步态优劣对照，更不证明W59固定cell、FR→RR、short/recovered的12目标机制任务全部通过。

## 4. 30fps源索引与25fps输出不要混为一谈

独立ffprobe确认：

|视频类型|实际输出fps|frame数|容器报告duration|分辨率|
|---|---:|---:|---:|---|
|三份A E60拼接片段，每份|25|306|12.240000 s|1280×720|
|5901B E600全片|30|18,001|600.033333 s|1280×720|
|B3 A E60全片，新补核|30|1,801|60.033333 s|1280×720|

phase manifest的fps=30是剪辑脚本用于原始全片的**源帧索引率**；本次只有B3 A E60与5901B E600全片分别经过独立ffprobe确认30fps，两份PPO A E60全片没有同样的实测核验。输出片段重新编码为25fps，且六段分别编码后拼接，因此成片12.24s与计划六段各2s不可粗暴视为物理计时失败。此差异需要明确记录，但现有证据不足以把它称为新的科学实验故障。

物理映射按脚本：先用trajectory首时刻t0和event_s确定start，再用ceil(30×(start−t0)−1e−7)得到first_frame；end同理。按该脚本约定，源视频相对时间为frame/30，若要还原物理时间需加回t0；对于尚未实测完整源片的两份PPO A E60，这仍是待源片率核验的约定。不能把“拼接片段第8秒”当作“机器人仿真第8秒”，也不能拿25fps输出第n帧直接除30来解释其时刻。

本说明为每阶段保留event_s与源帧区间；没有宣称重编码后每个取样帧和5kHz物理行逐一做了无误差配准。25/30fps帧间距约40/33ms，无法单独证实5.5ms、26.2ms的off统计或20ms峰值力；这些结论必须由物理日志与评分代码给出。


### B3取样采用的实际seek公式

根代理确认B3取样代码使用拼接视频时钟 `seek=2*i+u`，i=0…5、u为0.2/0.7/1.2/1.7s；它没有先读取每段实际PTS起点。进一步只读ffprobe得到B3成片关键帧PTS为0、2.04、4.08、6.12、8.16、10.20s，与六个分别编码的2.04s段吻合。因此第i段请求seek相对其对应关键帧PTS的偏移是 `u−0.04*i`，而不是固定u；到末段起点累积差0.20s。

|段索引i|请求的拼接片seek s|对应关键帧PTS s|请求相对该PTS偏移 s|
|---:|---|---:|---|
|0|0.20, 0.70, 1.20, 1.70|0.00|0.20, 0.70, 1.20, 1.70|
|1|2.20, 2.70, 3.20, 3.70|2.04|0.16, 0.66, 1.16, 1.66|
|2|4.20, 4.70, 5.20, 5.70|4.08|0.12, 0.62, 1.12, 1.62|
|3|6.20, 6.70, 7.20, 7.70|6.12|0.08, 0.58, 1.08, 1.58|
|4|8.20, 8.70, 9.20, 9.70|8.16|0.04, 0.54, 1.04, 1.54|
|5|10.20, 10.70, 11.20, 11.70|10.20|0.00, 0.50, 1.00, 1.50|

所有请求仍位于相应段内；此修正影响精确帧时间解释，不改变已观察124帧的计数。本表记录的是请求seek和已读PTS，不把解码/seek取整后的具体帧宣称为无误差逐帧对齐，也不据此静默移动日志事件时间。

## 5. E600的运动、收尾与任务吸收时刻

原件receipt直接给出：469.408 s进入最后一次home_hold；471.444 s进入complete、lap=10；之后128.556 s保持观察通过，直到约600s。actor-events第1534条在469.228s开始、471.444s结束并标success，之后不再添加W59事件动作。

需要避免一个容易发生的表述错误：名为completion的12帧序列实际覆盖468.50–470.70s，**结束于任务complete之前**。它展示回家收尾和足部调整，不能称为“已观察到471.444s吸收瞬间”。599s概览帧则位于完成后的保持段。完成时刻的精度来自原始receipt，而不是靠关键帧倒推。

E600在130s仍处于往返任务中，与471.444s才完成十次往返一致；不能因运动路径约62m便按命令上限速度估算130s已经完成。路线掉头/反向行进本身也不等于机身瞬间完成180°yaw；姿态角与相位要由日志或可辨认的朝向连续序列支持。

## 6. 视觉观察允许支持的结论

在已抽样的帧和短序列范围内，机器人机身体态总体平稳，能够看到去程、回程和收尾行为。B3与两份PPO E60样本间没有肉眼足以确证的巨大改善；不能由这些片段认定PPO已经得到明显优于B3的步态。

蓝色区域是material overlay显示。仅凭蓝色块、格子或颜色变化不能证明地形发生了真实几何变形，也不能量化沉陷、承载变化或材料参数辨识精度。实际材料响应应查仿真力、形变/内部状态和接触日志；视觉叠加层的用途是帮助定位材料区域。

材料段周期变长、off增加、冲击p95升高是日志结果，视频抽样可用于检查其体态背景，不能代替定量证据。滑移、CoT和模式转换是否改善，仍须用同case、同任务、明确时间mask的统计比较。

## 7. 明确的审查局限

- 没有连续逐帧审阅600s全片的18,001帧，更没有看完所有W59视频。
- 没有重新下载或逐文件核验整个175GB资产集合。
- 124帧不足以排除未采样时段的瞬时异常，不能用来估计稀有失稳概率。
- 这些短任务跨足片段不是固定cell机制的匹配因果实验。
- 姿态看似平稳不等于接触质量、能量效率、实机时延或分布外泛化已经合格。
- 本补充任务没有新增图像/视频、训练或修改仓库；只生成证据说明并修正论文版本与审查范围口径。
