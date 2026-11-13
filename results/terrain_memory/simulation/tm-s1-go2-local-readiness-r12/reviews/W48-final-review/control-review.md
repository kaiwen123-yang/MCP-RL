# W48 控制实现、有限求解与正式部署审查

审查基点：`61e16975fa6b38dc8e2d31779c2bd82f1aaeb8b6`。本审查只读仓库、固定上游源码和既有r35源码重建；没有登录D57、下载Release原件、执行native求解或物理积分。完整轨迹逐元素相同的结论来自执行者已提交的原位比较；本次独立核对其身份、结构和内部一致性，不能称为重新读取全部原数组。

复算程序：[audit_w48_control.py](audit_w48_control.py)，结果：[control-audit.json](control-audit.json)。均只用Python标准库。

## 1. 最重要的结论：修复已部署，但没有改变实际赢家

W48正式r36二进制SHA256为`b05c8da8ca74551dc73ea0272329234217d7160731e5cea8e33101bba3241338`。四个corrected有限问题以及A/B原任务都记录了实际`/proc/PID/exe`及SHA；与冻结build一致，N/r34 strict/constant环境一致，原任务无离线import/score-only开关。

将正式r36 patch应用到已核r35源码，两个变更文件的SHA与build逐项一致：

- `mjpc/utilities.cc`：`b0187bf7c91e244495a595693d5493aeb915aaf06fec649dd32beedb97bcd19e`。
- `mjpc/planners/ilqg/planner.cc`：`13d22f571f0b2939530962d5873e50ad15b880a12cd50c1703b9b8e44d1bb693`。

因此没有证据支持“这一轮其实仍跑了旧r35”。无积分代数检查也显示self/antipode为0，±1e−8rad真实旋转保留，位置/速度/act差保留。

真正有解释力的是以下逐候选结果：

| 问题 | old/corrected赢家 | 最终scale | 修复后最小正步ΔJ | 正步J发生变化的配对数 | 最大正步J变化 |
|---|---|---:|---:|---:|---:|
| A148 |nominal_3 / nominal_3|.0133352143|+.0337677667|6/45|2.7767e−13|
| A153 |nominal_9 / nominal_9|0|+.0450258404|45/45|2.0902e−11|
| B155 |incumbent / incumbent|.001|+.0774749599|45/45|1.4613e−9|
| B158 |incumbent / incumbent|.0133352143|+.1059959183|2/45|4.6861e−8|

每侧180正候选全部预测下降、全部真实J增加、正步安装0；所有候选finite，rollout/QP技术失败0、trunk violation0。

四问题共240个old/corrected候选配对，保存的policy identity全部一致。该hash覆盖reference states/actions/times、K/H/Qu/Qxu/active、scale/representation，不包含实际评分轨迹，也不包含StateDiff可执行语义。不能将“策略数组相同”写成“两个build的映射函数完全相同”。

20个backward配对的dV、Qu最大值、投影梯度、k最大值、K最大值、lambda/rate逐项相同。四个最终selected-policy文件SHA也完全相同。98/180正候选的J受修复影响，但最大变化仅约4.69e−8，远不足以扭转上述实际正增量和赢家选择。

20个alpha0则全部恢复精确nominal成本。B155五次旧alpha0 ΔJ为约+.02185～+.05019，B158约+.01796～+.02970，修复后均0。**alpha0本来就被r34禁止作为新步安装；修好这些被排除的端点，并未产生新的可安装方向。**

这比“修复无效”更准确：数值恒等性修复有效，有限正方向/决策收益没有出现。

来源：`w48/task-a/{A148,A153,B155,B158}-{old,corrected}.json`、`positive-directions.csv`、`summary.json`。

## 2. 四条部署核对通过了什么，没有覆盖什么

四个最终策略均为nominal/incumbent，实际scale分别为.013335、0、.001、.013335。它们的原评分路径和新部署检查都为`ActionImpl/Rollout`，四组states/actions/times/residual最大差和ΔJ全部0。没有将scale≠1策略强用indexed scale1，也没有再次施加alpha。

这有效证明了这些固定完整策略按相同native连续执行路径可以精确再现。

但本轮四个问题没有正步赢家，所以这四条并未验证任何被接受正步的`ActionIndex/RolloutDiscrete → ActionImpl`跨路径。并且`scripts/w48_finite.py`在固定rollout生成之后才逐节点`set_state/get_action`查询，返回的RPC动作不参与该350ms积分；不能用native-to-native一致排除端到端RPC执行误差。

该边界不意味着还要重复四条旧诊断。新的W48事件已经提供实际部署策略，后续应围绕真正执行链和已有预测误差检查。

## 3. 原A/B与W46逐元素相同的含义

`w48/original-comparison.json`报告A13217行、B5820行的time、qpos、qvel、q_target、torque、actual_grf、phase、requested_speed_cap、contact全部逐元素相同；1342/602次调用的origin、candidate_source、chosen_alpha、stop_reason、installed_plan_id、policy_updated、rollout/FD数量也相同。

但`chosen_actual`在A123、B38次调用不同。这与“部分内部评分改变而决策不变”相容，不能把整个优化器说成没有发生任何数值变化。也没有证据显示这两条原失败是deadline/OOM/进程被杀造成。

有限实验已经说明一个具体机制：修复的主要可见效果集中在不安装的零端点；正方向和赢家未变。原任务本轮给出的因果结果是：**这一最小identity修复没有改变这两个固定case的实际行动与原资格。** 它不能证明其他所有case完全无效，也不足以支持再用同配置重跑相同E30。

## 4. 四次启动技术失败：配置继承错误，已修

初始`w48_build.py`用新spec的`configuration_overrides`直接替换继承r35的完整overrides，丢了17个既有配置键，包括`r12_trunk_clearance`，但`r12_substep_path`仍开启。

保留源码`Trajectory::UpdateTrunkViolation`：

1. 看到r12_substep_path便进入；
2. 获取TrunkClearance sensor；
3. 直接取`GetCustomNumericData(model,"r12_trunk_clearance")[0]`；
4. 初始正式生成模型缺该数据，实际栈在此SIGSEGV。

`startup-backtrace.txt`记录路径`UpdateTrunkViolation → NoisyRollout → Rollout → FeedbackRollouts`。此问题不是StateDiff代数修复引起，也不是MPC找不到下降方向。

当前代码已改成合并`dict(old['configuration_overrides'], **spec['configuration_overrides'])`；build记录恢复了17键，native binary不变。有限实验使用既有完整模型，因此不必为修复配置继承而重复8solve。

4个技术attempt保留，其中A3、B1。原plant子步为0，transition FD为0；但startup的native预测rollout在崩溃前可能已部分积分，精确量没有写出，不能报告“所有物理计算0”。提交记录给出40slot、70000预测子步上界。

下一版应从冻结r36完整配置继承，并在正常启动前对生成模型的原有必需字段/来源做一次静态身份核验。这直接针对已出现的故障，不需要增加另一组原plant“冒烟回合”。

## 5. 尚未闭合的执行边界：查询时刻精度（W39已识别）

W39已量化显式查询时间的float误差，旧首大反馈时量化参考速度差远小于实际参考速度误差；不能将它写成本轮首次发现。这里要量化的是新W48策略上的作用。

这不是把所有问题归给FP32。返回动作float32和`double(float32(u_star))`已在W40及r29–r34说明中讨论过，不能冒称本轮新发现或据此宣布主因。

本次核到的是两条不同链：

- `State.time/qpos/qvel`为double；
- `GetActionRequest.time`为optional float，`GetActionResponse.action`为repeated float；
- 正式runtime每2ms先`set_state(time=t,...)`，再显式`get_action(time=t,...)`；
- 因此内部真实state的时间与传入策略插值的查询时间可能不完全相同。

固定作者commit `e00c47a5adb9856af2e0f24231bb3a60d5be23c4` 的`grpc_agent_util.cc`加仓库`s1-go2-wb-startup-upstream.patch`中的Reset修正，得到SHA：

`94b7328afb705bb226ee4dd306defce1a3f09972c4432da46bef8ee4c033e5c1`。

它与r36实际build的源文件SHA精确相同。该源码`GetAction`明确执行：

`double time = request->has_time() ? request->time() : agent->state.time();`

然后使用current state与该time调用`ActionFromPolicy`，double临时动作再Assign到float response。`AgentService::GetAction`还将该返回动作写回server data.ctrl。

作者Python API的`get_action(time: Optional[float] = None)`支持不提供time。**省略显式time会使用刚刚SetState的native double时钟，无需修改protobuf或server。** 但是否改善本项目轨迹，需要先在现有同state/同policy/同context上比较真实查询路径，不能从接口类型直接推断跌倒原因。

若进一步改为double动作传输，必须同时处理client/server的协议生成和服务读回；仅把收到的float转成np.float64不会恢复丢失信息。当前r36 manifest没有单独列出proto/生成描述符SHA，下一轮在实际环境读回应一并记录；本次不能假装已经检查了D57安装包的反射描述符。

完整端到端核验应区分：查询time量化 → native内部raw/clip/H-QP动作 → RPC舍入 → plant held动作 → 下一真实状态。只检查native固定rollout无法替代这条链。

源：

- [固定上游grpc工具](https://github.com/johnzhang3/mujoco_mpc/blob/e00c47a5adb9856af2e0f24231bb3a60d5be23c4/mjpc/grpc/grpc_agent_util.cc)
- [固定上游Python接口](https://github.com/johnzhang3/mujoco_mpc/blob/e00c47a5adb9856af2e0f24231bb3a60d5be23c4/python/mujoco_mpc/agent.py)
- 仓库`experiments/terrain_memory/s1-go2-r25-source/mjpc/grpc/agent.proto`，SHA`e89a75bf8b80db46b0a5a2c8d4e2a80df12ca4fe1c6da9bfb6c47a06a9125016`；r35重建文件同SHA。

## 6. W49最小推进建议

1. 保留r36 identity修复，但结束重复验证其零端点和原固定case；这轮已给清楚负结果。
2. 优先利用新1904个实际预测配对与已保存事件，把首次预测—执行差分成控制命令差和同命令动力学差；以同state/policy/context下的比较为准。
3. 对查询时间边界，先用保存实际状态和完整policy做零积分原路径/implicit-time成对动作查询，分别记录native querytime与physical time。若有有意义差异，再以单一机制检验未来短窗/闭环；保持返回动作精度不变可单独识别时间效应。
4. 若仍有同命令动力学误差，再核预测器与plant的上下文/接触/材料/积分。不要先将偏差全归到时间或FP32。
5. 正方向仍失败与稳定性问题应使用新真正事件问题，不再把四个W45困难fixture作为所有后续机制的唯一入口。任何新原任务必须有实际控制差异或明确新可识别问题。
6. 不能将“先取得完整任务成功”再作为允许原任务的循环前提。具体有限预算和是否进入下一原A/B，由主计划根据上述证据冻结，不在本独立审查另起大网格。

## 7. 独立复算结果与范围

程序已完成：8solve对应480候选、400行方向CSV，其中每侧180正步；20个old/corrected零步配对；240个序列化policy配对；4个部署比较；两份正式源码patch重建；两个原任务实际exe/env读回。

实际有限工作为484 native rollout（480候选＋4部署）、30816 transition FD、5012 risk FD、16 derivative pass、40 backward、480候选记录；显式rollout子步847000，FD内部物理另计。J分项最大算术误差低于1e−12，CSV ΔJ/预测公式重算误差0。

没有新增MuJoCo原函数计算、优化、物理、原任务或GPU工作。无需为审查新增物理试验。
