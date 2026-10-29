# S1 第13轮复盘与下一窗口：独立完成PANN首次学习

## 当前交接（每窗口结束只更新这6行）

- 当前任务：RAW审查完成；下一窗口仅W3 PANN seed0首次256真实转移/一次PPO更新，发布后停止。
- 当前状态：RAW256条/1次更新/205条有效平移/尾buffer0；PANN0条/0次，尚未启动；原材料E30/E60和完整往返仍为0。
- 已完成：4个真实RAW会话127.6484s，3次非足端接触；第256条正确截断后更新，日志键冲突已修，更新后checkpoint保住；W1及RAW全部实际产物已发布。
- 当前问题：PANN在线观察/恢复查询尚未运行；共同执行器仍有侧出条带和返程支撑失败；RAW原loss缺失、更新后策略尚未评价。
- 下一步：复用现有R12单流入口/r21/5workers/Torch CUDA，独立初始化PANN；使用本窗口新绝对时间，正常记录实际loss/特征/耗时并完整上传，不重跑RAW或自动进入W4。
- 续接位置：`/home/fanyuan/projects/kaiwen-learning/terrain-memory-r12`；RAW保存在`outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/PPO_RAW_S5-seed0/resume-state-ep00006.pt`；审查基点`4f18fac`，现有Release的246资产/2,229,777,826字节已齐，PANN使用独立目录。

## 1. 本轮实际结论

用户称为第13轮的这次推进，沿用`tm-s1-go2-local-readiness-r12`目录。无需改名、复制或重建run。上一窗口按用户要求在RAW后停止；本计划覆盖旧“RAW后自动接PANN”的文字，只安排一个独立PANN窗口。

**RAW首次真实学习工程目标完成，材料控制任务仍未通过。** [索引](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/index.json)和[全部256条转移](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/records/training__PPO_RAW_S5-seed0__transitions.jsonl)支持以下记录：

| 真实会话 | 转移/有效平移 | 仿真秒 | 结局与路线 |
|---|---:|---:|---|
| ep00002 | 75/54 | 37.4260 | 到远端并返程，10.304s曾侧出条带，最终非足端接触 |
| ep00004 | 44/40 | 21.8460 | 去程非足端接触，未完整穿过材料 |
| ep00005 | 77/57 | 38.3764 | 到远端并返程，16.174s曾侧出条带，最终非足端接触 |
| ep00006 | 60/54 | 30.0000 | 曾侧出条带；到第256条外部截断并更新，随后日志异常 |

ep00001/03是零步初始化失败；另有一次Tianshou缺失发生在物理会话前。四条物理任务共638,242步，最大应变均<.02、无material_domain_exit，原条带贯穿均未成立。编号不能当作六条物理失败。

全部256条dt>0；真实terminal在75/119/196，唯一truncated在256。RAW最后12个PANN特征通道为0；有效平移205，其余51是站立/转向等真实样本，不删除重采。实际cap约.189914～.239870m/s；旧结果顶层constant/.25是沿用元数据，以transitions/cap-decisions为准。

[更新证据](../results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/records/training__PPO_RAW_S5-seed0__learning.jsonl)显示一次更新、各参数16个Adam步、Actor/Critic发生变化、尾buffer0。旧`samples`键冲突发生在更新后，原loss标量未保存；保留缺项，不重新优化或虚构原始loss。全部四会话发生在首次更新之前，较长轨迹不能归因于学习；更新后的策略没有物理评价。

## 2. 已完成的修复与本次小修

R12入口、旧字段适配、六流等待关闭、CUDA设备、自然新会话创建Agent、Agent.model、截断顺序、零时长过滤及更新后计数均已实际接通。继续复用，不再列作待开发。

本次只做直接影响下一次启动/解释的小修：wrapper移除上窗口写死的start/deadline，统一接收既有Python参数；任务元数据读取当前契约；保存单流所需PPO和共同观测设置，去掉继承R11六流大目标和正式冻结门禁的无效说明；新策略结果标为`ppo_policy`，保留真实动作记录。未改变PPO数学、奖励、控制器或已有RAW结果。

## 3. 下一窗口只做W3

| 项目 | 执行安排 |
|---|---|
| 唯一训练流 | `PANN_PPO_S5`，seed0，独立初始化；绝不从RAW checkpoint启动 |
| 现有执行器 | r21二进制/资产、5workers、5080/kaiwen-learn，保持原Go2/TM02/PD/力矩/时钟 |
| 采集 | nominal、60s连续采集会话、课程关闭、累计256真实转移；自然失败重开会话，普通更新不reset |
| 策略/材料 | 保留原PPO/奖励/归一化、2Hz cap[.18,.25]；冻结B ideal，因果观察，不读真q/e |
| GPU | 沿用Torch CUDA、现有`XLA_PYTHON_CLIENT_PREALLOCATE=false`和`JAX_ENABLE_X64=true`；PANN依赖/权重错误在实际启动中定点修 |
| 完成 | 一次实际PPO更新、正常loss/采样记录/checkpoint及全部本轮产物发布，然后停止 |
| 时间 | 最多4h含全部工作，通常3.5h收尾；未达256则保留真实样本和尾buffer，下窗口只接续PANN或发布 |

入口已经存在，启动参数形状为：

```bash
bash scripts/run_s1_go2_r12_training.sh --group PANN_PPO_S5 --seed 0 --window-start-unix <本窗口首次活动时间> --deadline-unix <预留收尾后的训练截止>
```

由执行者在窗口开始记录实际Unix时间，不复制上轮数字。Python将总上限限制到start+4h。首次不要传`--resume`；后续因窗口到期续接时只能传本组已有checkpoint。RAW目前累计目标已达256，不启动它来“试一下恢复”。

用正常episode/rollout记录看观察器推进、查询次数、12个恢复特征及额外耗时。特征为零或信息有限时如实报告，不人为灌入真状态、不要求制造非零特征，也不另开测试/benchmark/恢复演习。

RAW四次物理运行约102.28分钟，约150条转移/小时，运行规划占85.77%。PANN多出在线观察、查询和初次编译，不能承诺同样耗时；到点交真实进度，不后台越界。

## 4. 完整发布与轻量交接

[现有Release](https://github.com/kaiwen123-yang/MCP-RL/releases/tag/tm-s1-go2-local-readiness-r12)已有246资产、约2.23GB，包括5个完整视频、172个NPZ和4个checkpoint；环境/小记录另在Git。现有清单中应传Release的文件名/大小与资产列表一致，W1和RAW没有待传项。一个小episode JSON保存在Git，原loss缺失已明示。

下一窗口只追加本轮新增/变化源码、配置、环境记录、正常日志，以及全部实际PANN视频、原始轨迹、物理/预测/材料数组、rollout和checkpoint。继续每个实际episode完整记录，视频从保存轨迹渲染；不为补材料重跑物理。引用已发布且未变的r21/PANN/场景资产，避免重复上传。

已有执行端curl传输方式是实际解决gh408/TLS EOF后的可用路径，继续复用，不重新建设发布通道。现有索引记录大小/URL/状态，正常看一次清单即可；不全量hash、不回下载复验，不在用户电脑另做镜像/复制/压缩归档。训练状态与发布状态分开记录，未传完如实pending。

开头最多5行后直接工作；结束只更新顶部6行和正常结果。禁止重复全仓审查、单元/mock/冒烟/导入/形状测试、通用fallback、无关矩阵及新的日报/归档包。

## 5. 后续方向与论文依据

PANN首次链路完成后再审查，下一窗口不自动开始后续研究。共同控制器优先看侧向离开材料条带和返程承载交接。当前奖励含路线进度，却没有单独的原条带贯穿项；已经出现侧出后仍得进度分的样本。它们来自更新前策略，不能叫“学会绕路”，但在长时间扩训前应对齐任务目标、执行能力与学习信号；本次先保持两组相同条件，不同时重写奖励和控制器。

PPO目前仅调2Hz速度上限，不能承诺修复快速支撑失效。更换控制器后标明版本边界，不拿旧RAW和新PANN当收益对照。正式评价仍逐case同版E30→E60→E300→E600，训练前缀不替代资格。先在5080解决在线链和控制问题，再依据CPU/MPC与PANN真实成本决定云资源。

- [PPO](https://arxiv.org/abs/1707.06347)：交互采样与优化交替；256/四小时是项目工程安排，一次更新不代表收敛。
- [RLOC，§VI-H](https://arxiv.org/html/2012.03094v3)：优化控制在环时采样可能主导成本；本轮不据此换SAC/TD3或控制平台。
- [STANCE，§IV](https://arxiv.org/html/1904.12306v2)：软地支撑运动与接触动力学需要相容，指导后续具体支撑分析，不增加首次PANN训练前的复现门禁。
- [JAX GPU内存说明](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)：现有wrapper已关闭默认预分配以共用GPU；保持已有设置，在真实运行中处理具体问题。
