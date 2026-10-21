# Go2 冻结策略完整任务：只读复核

日期：2026-10-19，Asia/Shanghai。Mode：PR Review。

范围：`src/go2_frozen/{simulator,task,planner}.py`、`scripts/run_go2_frozen.py`、六条正式运行的 summary、阶段切换、命令序列、公开 10 Hz 轨迹，以及本地保留的对应 50 Hz 原始轨迹。仅进行源码阅读、现有 CSV/JSON 重算和哈希核对；本次复核未加载或推理策略，未执行 MuJoCo 步进、未新增训练或机器人采样。

复核身份：本复核代理没有编写 task/planner/evaluator，但参与了 simulator 适配，因此对任务评价与规划流程属于交叉复核，对 simulator 部分属于源码自检与运行证据复核；不将整份工作包装为完全独立的第三方审查。

## 结论

**六条运行均满足预声明的完整任务成功条件；没有发现足以推翻这六条成功身份的代码或日志问题。**它们是真实 Go2 浮动基座、十二关节、PD、足地接触与冻结神经策略闭环执行，不是让平面模型代替机器人运动。普通 MPC 三条件均成功，几何 follower 三条件也均成功；这些是六条固定确定性评价，不是统计成功率估计，也不证明 MPC、RL 共同学习或 PINN 的独立收益。

需要保留两个报告口径边界：停止成功只是既定的**平面位置、平面速度和朝向误差连续保持**，不是全身静止；`planner_model_steps` 只统计 CEM 候选评分的模型步，不是求解函数内全部模型传播次数。下文给出实际量值，不改成功阈值，不重跑以改善结果。

## 已核事实

1. **真实执行与状态连续。**`run_go2_frozen.py` 每条运行只在任务开始前调用一次 `sim.reset()`，之后各阶段均调用 `sim.step(command)`。除 reset 外，没有把路径参考或 MPC 预测写入机器人 qpos/qvel 的操作。各条日志的 12 个关节均有实际运动；每个关节的全程位置跨度最小仍约 0.182 rad，最大约 1.078 rad；关节速度峰值为 16.74–22.02 rad/s。base z 在 0.290–0.346 m 之间变化，足接触状态持续切换。不能把这些动态量解释成运动质量收益，但它们与完整动力学执行一致。
2. **观测与动作契约。**当前 robot_lab 配置无 observation history；观测为 body angular velocity、投影重力、三维速度命令、相对关节位置、关节速度、前次原始动作，使用原配置缩放。policy 关节顺序 FR/FL/RR/RL；MuJoCo qpos/dof/actuator 通过 joint name 映射，没有把 MJCF 的 FL/FR/RL/RR qpos 顺序直接当成策略顺序。wxyz 四元数与逆旋转一致；warmup 后不清空前次动作。其他可选策略的 history 分支未在本次六条运行中验证，不能外推其部署正确性。
3. **时钟与冻结身份。**0.001 s 物理步、0.005 s PD、0.020 s policy、0.200 s 高层控制。全部运行满足 `physics_steps = 2000 + 20 * policy_steps`，推理次数等于 `50 + policy_steps`；原始日志时间间隔为 0.02 s。六条共 4,113 个任务 policy 步、4,413 次含 warmup 的推理、94,260 个含 warmup 的物理步。每条文件 SHA256 与 tensor SHA256 前后相同，`training=false`、`requires_grad=false`，policy/model/optimizer 更新全部为 0。所有 summary 的源码哈希与复核时源码一致。
4. **PD 与饱和。**策略输出关节位置目标，PD 产生驱动力矩；实际饱和来自 MJCF motor ctrlrange，未把 YAML 辅助 torque clip 冒充实际硬限。现有 50 Hz 记录中所有实际 torque 均在对应 ctrlrange 内。该核验不能替代未保存的每一物理步力矩轨迹；代码层的 MuJoCo actuator 限制仍在全程生效。
5. **碰撞与跌倒。**`simulator.py:_pd_step` 每个物理子步调用 `_update_contact_flags`，任何机器人 geom 与任务障碍 geom 接触都会设置历史失败；跌倒条件也是每个物理子步累计。warmup 后不清除这些标志。六条正式运行均无历史障碍接触和历史跌倒。足地接触与障碍接触分开，不会把合法落足当成撞墙，也不只检查 base 中心穿门。
6. **普通 MPC 确实优化命令序列。**每次对 3×256 条长度 20 的候选序列进行 CEM 评分，使用选中序列第一个命令；逐次核对所有 MPC 日志中的实际 command 都等于 command_sequence[0]。几何反馈序列作为初始化与一个候选，不被单独重命名为 MPC。follower 被明确另列，不能把其较快结果隐藏。
7. **阶段和成功判断。**六条均严格按 straight→turn→door→stop→complete 的顺序切换，时间递增，没有最近路径点投影直接宣布完成的路径。最终保持段全部满足预声明阈值：位置≤0.22 m、平面速度≤0.12 m/s、朝向误差≤0.20 rad，持续≥1 s（浮点显示误差约 1e-12 s）。门后最终状态远离门面板，不存在本次轨迹中心过门但身体留在门里的迹象。
8. **公开轨迹与原始记录。**三条 MPC 的公开 10 Hz CSV 与本地 50 Hz 原始记录每五行采样并保留末行的规则逐项完全相同。两种轨迹身份没有混用。资源指标使用 50 Hz 的力矩和速度采样积分，故属于所声明的机械代理指标，不能称电池能耗或高频精确能量。

## 末段停止的实际量值

以下均从对应 50 Hz CSV 的最后 1.00 s、51 个状态样本计算，未增加新的判定阈值。

| 运行 | 完成时间 s | 保持段最大位置误差 m | 最大平面速度 m/s | 最大朝向误差 rad | 最大绝对 yaw rate rad/s | 平均绝对 yaw rate rad/s | yaw 范围 rad |
|---|---:|---:|---:|---:|---:|---:|---:|
| mpc_nominal | 15.94 | 0.0830 | 0.0851 | 0.0330 | 0.1942 | 0.0564 | 0.0339 |
| mpc_narrow | 15.90 | 0.0786 | 0.1196 | 0.0668 | 0.2511 | 0.0726 | 0.0680 |
| mpc_wide | 16.48 | 0.0621 | 0.1150 | 0.0608 | 0.2351 | 0.0621 | 0.0628 |
| follower_nominal | 11.24 | 0.0643 | 0.1059 | 0.0814 | 0.2832 | 0.0710 | 0.0687 |
| follower_narrow | 11.20 | 0.0695 | 0.1134 | 0.0582 | 0.1406 | 0.0441 | 0.0428 |
| follower_wide | 11.50 | 0.0486 | 0.0809 | 0.0637 | 0.2188 | 0.0630 | 0.0636 |

**停止措辞边界（来源：task.py:191–200、PROTOCOL.md 的预声明）。**当前成功判断没有 yaw-rate、垂直速度、关节速度或足端静止阈值。末段数据没有显示持续大幅转圈，但仍有局部姿态和关节调整；例如 mpc_nominal 最后 1 s 的关节速度峰值约 5.11 rad/s、垂直速度峰值约 0.108 m/s。因此宜写“满足预声明平面停止条件”，不能写“所有自由度完全静止”或“已验证稳定静态站立”。如果后续任务需要精确放置、对接或静止传感，可在新的预声明协议中加入相应停止定义；不能据此回改本次成功统计。

## 计算统计的口径

`planner.py:185` 的 `model_steps = evaluations * horizon` 正确统计 CEM 候选评分，但 `feedback_sequence` 还传播 20 步，最后为输出预测轨迹的 score 又传播 20 步；因此每次 plan 另有 40 步未计入这个字段。三条 MPC 的计数关系为：

| 运行 | 日志中的候选评分模型步 | plan 内另外的初始化与结果轨迹步 | plan 内合计 |
|---|---:|---:|---:|
| mpc_nominal | 1,228,800 | 3,200 | 1,232,000 |
| mpc_narrow | 1,228,800 | 3,200 | 1,232,000 |
| mpc_wide | 1,274,880 | 3,320 | 1,278,200 |

额外部分约为候选评分计数的 0.26%，不改变任务成功身份或 wall latency。离线一步预测和开环重放另有独立传播，也不应混为在线求解候选数。报告应称该字段“候选评分模型步”；不建议为了改标签重新执行正式评价。

## 研究解释边界

当前地图与阶段顺序已知，参考路线及速度安排是工程预设，MPC 预测时域为 4 s，采用未辨识的一阶速度/偏航模型。成功证明这套公开策略和普通规划链可以连续完成本次多阶段任务；它不是学习到的跨阶段目标，也没有验证有限时域以外的价值学习。follower 同样成功且更快，当前证据不能证明必须引入复杂模型或共同学习。

名义预测与实际响应存在差异，是后续比较灰箱/辨识/PINN 的线索。仅凭误差不能断言缺少物理约束是根因；需区分策略闭环响应、接触、历史依赖、参数偏差与预测时域影响。本轮复核未提出新的训练或实机授权请求，也未启动这些工作。

```json
{
  "review_mode": "read_only_existing_artifacts",
  "runs_checked": 6,
  "declared_success_confirmed": 6,
  "ordered_stage_transitions_confirmed": 6,
  "source_hash_match": 6,
  "policy_file_and_tensor_hash_unchanged": 6,
  "clock_and_inference_counts_match": 6,
  "policy_updates": 0,
  "model_updates": 0,
  "review_policy_inferences": 0,
  "review_mujoco_steps": 0,
  "material_success_invalidating_issue_found": false,
  "reporting_boundaries": [
    "Planar stop criteria do not imply full body static equilibrium",
    "Logged planner_model_steps count candidate scoring, not every plan-internal propagation",
    "Deterministic cases are not independent statistical trials"
  ]
}
```
