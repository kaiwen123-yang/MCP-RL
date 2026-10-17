# 连续宏命令可行见证：冻结协议

日期：2026-10-17。只为区分旧有限候选池不足与固定反馈可能存在的限制，不训练、不改actor/design/checkpoint、不调整物理参数或约束阈值。该计算比旧34候选池预算更大，不宣称同预算性能优势。

## 1. 全覆盖集合

对象是`one-step-authority.json`的`all.unresolved_keys`全部42个查询，按该文件顺序处理。它们是140个旧空pool中，尚未被名义下一拍控制权界排除的部分；不按checkpoint或后续成功结果删选。原768查询与140失败分母继续保留。

变量仅为原8宏段×2个CEM命令坐标，16维，每维原bounds`[−.45,.45]`。第二坐标仍按原公式转为执行eta。初态x、持续z、design、真实训练后actor、goal preview均从相同原查询取得；80步原生完整执行预测，名义`w=0`而support W=.04不变，不给未来隐藏扰动phase。

## 2. 固定连续可行性问题

本阶段目标是**寻找满足既有gate的命令存在性见证**，不是重新定义RL奖励、训练策略或追求最优任务成本。

- SLSQP目标：原预测`tube_margin_max / (2e−4)`；分母仅为原tube_budget单位归一化。
- 约束保持原reference margin≤0、actual domain margin≤1e−7、input margin≤0。
- 成功还必须原tube margin≤1e−8；没有放宽tube门槛。
- 原actual task cost与reference cost原样报告，不用于本次可行性目标。
- 固定`maxiter=100, ftol=1e−10`；使用完整原生闭环对16维命令的autograd导数，包含penalty actor观测中的支持区间依赖。
- 不同分支的非光滑性保留；失败或未找到可行点标为UNKNOWN，不据solver返回代码宣称全局无解。

三种初值顺序固定：①旧pool中已通过reference/actual/input三门槛者的最小tube-margin候选；②旧reference-only gate的actual任务主赢家；③全零CEM坐标。即使前两者相同也按记录顺序保留，不按结果添加新初值。每次找到任何满足原四项gate的有限命令点，即做独立原生前向核验；核验通过后该query成功并停止剩余starts，无需继续成本最优。

## 3. 真实性核验

优化使用`closed_loop_prediction.py`的functional可微预测。成功候选另以历史`ResponseServo`的原始`observation/interval/response/f0/fallback`方法逐拍积分，不构造环境、不调用env.step或MPC；核对80拍状态、z、action、task与全部gate，cost/逐拍数值误差门槛`1e−10`。独立路径必须自己满足原gate，不能用比较容差放宽可行性。

失败时保留最佳gate违反情况、原task成本、solver状态、全部已尝试starts和资源消耗；没有可行命令则不返回伪可行fallback。若出现非有限梯度/数值异常，记录该start失败，不改solver或物理设置救结果。

## 4. 计算及输出

最多4个worker，每worker torch单线程。fun/jac对同一命令点共享缓存，避免重复模型展开。逐query记录forward调用/80步模型转移、命令Jacobian反向调用（4个输出：目标加3约束）、独立原生核验、elapsed wall和process CPU time；主进程记录总墙钟，不把并行缩短墙钟当减少总计算。

脚本` scripts/audit_continuous_command_feasibility.py`；输出本目录`continuous-results.json/md`，中间worker记录留`work/stage-feasibility-2026-10-17/continuous/`。协议内容与输入hash关联。新增模型计算与较旧pool预算分列；真实环境step、训练更新、actor参数更新、checkpoint写入均0。

本实验只处理旧名义tube合同，尚不是狗/臂完整阶段入口或长程受扰任务。即使找到连续可行命令，也只否定对应查询的“旧有限pool失败即固定反馈无能”；未找到则仍不能区分局部求解预算、可达性和反馈限制。
