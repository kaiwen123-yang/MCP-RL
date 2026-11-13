# W47 Task A 独立控制审查与 W48 最小推进建议

执行规则以[最终W48计划](../../../../../../docs/S1_W48_STABILITY_PLAN.md)为准；本独立报告的建议经主审查整合，不另开并列实验。

审查基点：`6e9075f48718509d48f51d08d0a8f62a34edfb41`。本审查只读取仓库、既有源码重建和提交的数值证据；未启动服务器、物理仿真、native 优化或 PPO，未下载 Release 原件。完整原数组仍在 D57，不能把本次 JSON 算术核验称为独立重放。

## 1. 这轮真正新增的控制结论

W47 将上一轮“alpha0 为什么不再生 nominal”的疑问定位到一个可重复的浮点首扰动，并用最小源码修复验证了其零端点效果：**同一个四元数与自身作切空间差，MuJoCo 3.2.3 的原 primitive 路径出现非零舍入量，MJPC 反馈会将它当作真实姿态误差。**

现存 constant 分支的 20 个 alpha0（4 个 fixture × 5 次 backward）全部满足：反馈中心动作与实际 nominal 动作一致、时间网格一致；首个动作差异处 actual state 与 reference 逐元素相等；该点 H-QP 是内点恒等；记录的非零四元数自差与 K 乘积可解释首个输入差。其 self tangent 最大分量范围约 `1.69e-20`～`2.98e-19`，并非实际机器人发生了对应姿态扰动。

在第一 backward attempt 的配对中，最初极小动作差异经过 350 ms 后可产生显著命令差：

| fixture | 首动作分叉节点 | 初次命令差 rad | old 相对 nominal 的全窗最大命令差 rad | old ΔJ | corrected ΔJ | corrected 350 ms 前移 |
|---|---:|---:|---:|---:|---:|---:|
| A148 |0|8.67e-19|9.1686e-5|−9.6054e-10|0|−2.7418 mm|
| A153 |0|1.04e-17|8.1738e-5|−6.0345e-7|0|+1.4941 mm|
| B155 |0|1.43e-15|1.026335|+0.0234713|0|+1.0805 mm|
| B158 |2|2.17e-19|1.108094|+0.0181906|0|+0.3541 mm|

旧诊断二进制四处精确再现存档 alpha0；修复诊断二进制四处 states/actions/times/residual 均与 nominal 完全相同，J 差为 0。这是对**首扰动来源与零端点一致性**有因果意义的修复证据。它不等于移动、抗扰或 receding-horizon 稳定性已经得到改善；修正后的 nominal 本身仍主要驻留，A148 甚至后退。

证据文件：`w47/task-a/saved-array-summary.json`、四个 fixture JSON、`paired-evidence.json`。路径前缀均为 `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/`。

## 2. 修复的准确边界

`experiments/terrain_memory/w47-state-difference-identity.patch` 修改 MJPC `utilities.cc::StateDiff`：

- 在原 `mj_differentiatePos` 后，遍历 free/ball quaternion。
- 仅当四个分量逐元素完全相同，或逐元素精确反号时，把对应三维旋转切空间差置零。
- 没有邻域容差，没有将“足够接近”的不同旋转归零；位置、其它关节和后续速度/act 差不变。
- 精确反号是同一旋转的 quaternion antipode，不是姿态变化。
- 该补丁不改 MuJoCo 引擎的 `mjd_transitionFD`。现有 `model_derivatives.cc` 的 A/B 仍由引擎 FD 求得，不能称“动力学雅可比已经修好”。

`w47-fixed-candidate-diagnostic.patch` 则只为离线 `R34ScoreOnly` 增加 `MJPC_W47_INDEXED_SCORE` 分支，使保存候选用真实 `ActionIndex/RolloutDiscrete` 而不是连续 `ActionImpl`，并正确增加 rollout 计数。两个 old/corrected 诊断变体都有同一诊断补丁，只有 corrected 带 identity 修复，因此配对没有把执行路径差异混入修复效果。

本审查将两份 patch 离线应用到上一轮已核对的 r35 源码，结果 SHA 均与 W47 build 记录完全一致：

- corrected utilities：`b0187bf7c91e244495a595693d5493aeb915aaf06fec649dd32beedb97bcd19e`
- diagnostic planner：`e1fed3a78a645bb3c679adb19e93e08a214bd1c34d6eba831a9d2aa7e52de944`

W47 `deployment_qualified=false`；冻结 r35 二进制 SHA 未变。此次只是两个隔离诊断构建，没有部署正式 S5-r36。

### 仍需区别的执行路径

paired 只验证了第一 attempt、scale=1 的 alpha0 在 `ActionIndex/RolloutDiscrete` 路径。正式 `ActionImpl` 会做时间插值和 `mj_normalizeQuat`；相同旋转若经过归一化变成非逐元素相等表示，exact-identity 分支未必触发。这是尚未验证的路径边界，**不是已经确认的第二个错误**。

如做 source/deployment 比较，必须保留完整策略的真实 scale 与来源。`ActionIndex` 显式使用 scale=1，不能用它代替 scale 可能不同的 incumbent/nominal `ActionImpl`，否则会人为制造不一致。正步候选的 alpha 已经写进 feedforward，不得再次缩放。

## 3. 正方向失败尚未被这轮修复验证

独立重新聚合了四个 fixture × 三个分支的 540 个正候选。它们**全部预测下降、全部真实 J 增加**；记录的 dV 与 predicted-decrease 公式逐项一致。这个结论指旧 r35 保存问题，不是对修正后的新求解结果作判断。

| fixture | none 最小 ΔJ | constant 最小 ΔJ | load 最小 ΔJ |
|---|---:|---:|---:|
| A148 |+.0173864|+.0337678|+.0322384|
| A153 |+.0029141|+.0450258|+.0282996|
| B155 |+.0543733|+.0774750|+.0888540|
| B158 |+.0788072|+.1059959|+.0945374|

W47 **新优化求解 0、transition FD 0**，没有用 corrected StateDiff 重算正方向，也没有重跑原 540 候选。真实动作差不是仅靠正步参考代价解释：none 也有同样失败。新全程统计又显示大量真实在线调用可接受正步，故不应再把全系统描述成“完全不会下降”。有限保存问题集中在困难上下文，研究重点应转为修复后局部敏感性和闭环执行能力。

原大 K 说明反馈对状态误差可能敏感，但其数值依赖单位、参考与活动集；不能据 K 的绝对最大值给控制器任意剪裁或加入新成功阈值。exact identity 只消除了本不该存在的一种自差输入；真实微扰、状态传输、接触扰动和模型差仍会进入反馈。

## 4. 实际新增工作量及未发生的工作

根据 8 条逐运行记录独立相加：

- 8 条 350 ms fixed-policy native rollout；每条 175 held intervals、1750 个 .2 ms 子步。
- 总计 1400 held intervals、14000 子步；独立诊断时长相加 2.8 s，不能写成连续原任务跑了 2.8 s。
- 新原 plant episode、native optimizer solve、transition FD、donor 评估、PPO、GPU physics 全部 0。
- 隔离诊断构建墙钟 63.294 s，paired 总墙钟 35.256 s。
- 四个 old/corrected 问题的 input policy、context、reference、model、state 哈希逐对相同；实际 `/proc/PID/exe` 哈希与构建声明一致。

停止点是有限工作的结束与原先准入条件未满足，并非后台时限杀掉程序。原 E30/E60/... 资格没有新增通过或失败；最近的完整物理结果仍是 W46 A/B。

## 5. 上轮计划造成的推进障碍，应主动修正

W47 把原任务准入绑定到了唯一 R8 文件、最新 W46 A/B 完整策略与同 origin 预测误差；执行后确认这些数据不可用。这个门槛对“donor 是否有益”的研究有意义，但不应继续阻挡**已经定位并修复的数值一致性错误**的闭环因果评价。

如果继续要求“先在已不存在的最新上下文中证明移动收益”，下一轮仍只能报告缺件；如果只重复四个旧 alpha0 端点，则也无法回答修复是否改善真实行走。应将 correctness 修复独立于 donor 分支，并把一次原 A/B 运行同时用于效果评价和获得下一次真正需要的上下文。物理失败仍是有效结果，原成功判据不变。

R8 缺件也不能被解释为 donor 机制无效；但不应让它成为唯一可推进路径。

## 6. 建议 W48 的最小工作包

### A. 正式冻结最小 r36，与已有问题有限对照

仅把 exact quaternion identity 修复纳入正式 r36；记录改进不改变控制行为，诊断 score-only 开关不能带入原任务。保留 N、r34 strict、r35 constant、原动力学/步态/PD/判据和 A/B 冻结检查点。

先做一次不积分的原生 StateDiff 代数验证，覆盖已观测四元数的 self、精确 antipode 与真实微小正负旋转，确认不同旋转仍保留，位置/速度/act 没有改变。不要再重跑 W47 已完成的八条 alpha0 配对。

然后在四个**当前确实可用**的完整 fixture 上做 constant/N 的修前后求解：优先复用身份完全可核的旧结果；如为保证共同入口确需重算，每 fixture 最多一个 old、一个 corrected，总计最多 8 solve。两个变体冻结同一 P0/x0/context/ref，不能各自生成不同免费参考；记录正步预测与实际变化、候选完整身份、分项成本、实际动作/前移/恢复和全部 rollout/FD 工作。

若需要验证部署路径，可选四个有明确源身份的 scale=1 固定候选，各补一次 corrected ActionImpl 的 350 ms 诊断，与已有 indexed 结果比较，总计最多 4 条。它是路径诊断，不是新增求解；不能对 scale≠1 的 nominal 硬套 indexed 等价。

### B. 一次真实 A/B 闭环因果评价

正式构建、入口、约束、记录一致且没有新技术错误后，可以进行 A/B 各一次新的原 E30，评价修复本身的作用；**不要求先得到 donor，不要求先在旧困难 fixture 上证明毫米级前移变成稳定行走**。正步仍停滞应作为清楚记录的机制结果，原 E30 用来检验修复在真实调用分布中的影响。

以 W46 同 checkpoint/seed 的已完成结果作历史对照，不再增加 r35 原 E30 以凑配对。保持 A seed1/B seed2，各14737/update57，新 PPO0/0。新 E30 一次失败即结束该 case，并保留原失败条件；通过则按其自身原 E60/E300/E600 阶梯继续。不能保证修复会通过。

在两个真正独立 case 上运行，天然支持双任务并行，各 10 CPU；第三任务只做记录/分析等独立有限工作。无墙钟 deadline，原物理时长与有限工作量保留。

### C. 让新运行留下能回答问题的最小完整上下文

现有事件记录有 .12 s ring、事件前 .1 s/后 .2 s，且完整 plan 只是已列身份，当前 transport 文件不是历史。下一轮应让首次大执行目标变化/膝力矩反转、任务阶段切换和终止等固定事件能关联到完整已部署 policy、其生成/评分问题、pre-call incumbent、实际输入 x/time/mocap/userdata/private gait、共同 r35 reference。

另保存上一 accepted prediction 在下一实际 origin 上的对齐状态及流形误差。缺少时间匹配数据不能事后拿别的 W45 delta 代替。可用有限滚动缓存加事件落盘，不必为每一个搜索候选长期重复保存全部大数组；但关键对象缺失时必须让索引明确指出，不能以“已上传全部生成文件”冒充可重建。

这项记录应随真实 E30 同步完成，避免先跑一轮后发现下一轮所需数组又没有生成。

## 7. r36 必须连同接入检查，不能只换二进制名字

至少涉及：

- `experiments/terrain_memory/s1-go2-r11-S5-r36.json` / `.patch`：真实 based_on r35、完整配置、修复与源码SHA。
- `scripts/prepare_s1_go2_r12_support.py`：构建身份元数据白名单。
- `scripts/s1_go2_r11_runtime.py`：r27记录、r31/r32说明、N机制、r34 strict、尤其当前 `controller_build == 'S5-r35'` 的 r35 branch 条件，以及 r29 recorder。
- `scripts/s1_go2_r11_policy.py`：检查点/phase lineage 与 full-plan 信息路径。
- `scripts/s1_go2_r29_records.py`：candidate、r35分项、事件归档与 schema 白名单。
- 原 evaluation contract：active_controller_build、controller_artifact、selected_mechanism=N、input_reference_branch=constant、checkpoint/condition一致。

`run_s1_go2_r12.py --build` 并不会自动修复所有合同/运行时分支。忘记扩展白名单会把修复测试混成控制配置变化；应读回真实环境、task参数和二进制身份，而不是只看CLI或文件名。

## 8. 本审查的可复算证据

`audit_w47_control.py` 已实际执行，输出 `control-audit.json`：

- 540 个正候选成本/预测公式、20 个零步首分叉记录、8 条 paired 统计核验通过。
- 成本分项及 J0+Jref 算术误差最大 `4.44e-16`，ΔJ 与预测公式重算误差0。
- 195 个可解析内点节点，记录投影后的动作重构误差最大 `2.22e-16`；20 个零步首分叉节点误差均0。
- 2 份补丁 SHA 和重建源码 SHA 与诊断构建一致。

这些核验支持提交证据的内部一致性，不替代 D57 原数组复算，不扩大 W47 的实际实验范围。
