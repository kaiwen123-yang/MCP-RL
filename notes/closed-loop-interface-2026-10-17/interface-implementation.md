# 原生实际执行闭环：接口实现说明

日期：2026-10-17。接续`e23a28a`与首篇双平台设计。本文件只说明三状态原生PPO接口；不把它接到六状态双执行器对象，不涉及机器人执行、训练或checkpoint修改。

## 1. 两层代码与依赖

- [`closed_loop_prediction.py`](../../src/response_ppo/closed_loop_prediction.py)：纯functional可微预测，不构造env、网络、RNG、planner或optimizer。
- [`execution_aware_env.py`](../../src/response_ppo/execution_aware_env.py)：独立`ExecutionAwareServo(ResponseServo)`，只替换规划预测接口。原`env.py/pilot.py`不修改，父类实际step、隐藏扰动和参考推进保留。

`NativeContext(cfg,P,B,E)`显式携带原参数与原DARE度量；若分析使用float64，将同一已保存P转换为float64，而不是悄悄重新求高精度P。native minimax fallback要求原生`B'PB>0`。

`actor_mean(obs)`回调返回未squash的Gaussian中心，形状为`(...,)`或`(...,1)`。可以直接传checkpoint网络的`actor.mean`，也可传以固定/功能化参数闭包实现的mean函数。接口不复制或改变参数，不停止actor/state梯度。

## 2. 原生微步语义

```python
native_step(actor_mean, x, z, command_exec, raw_design, goal,
            absolute_tick, context, support=bool, disturbance=None)
```

输入为物理x3、持续参考z3、执行单位命令`(position c,speed eta)`、四维raw_design及当前可用goal。完整17维observation与原源码一致；goal/未来扰动不会被添加到actor观测。

计算顺序为reference_step → state-dependent support → 17维obs → actor mean → tanh/support或fallback → 实际`xnext=f0(x)+Bu+Ew`。z只按自身reference dynamics推进，**从不重置为x**。

`disturbance=None`表示名义`w=0`，support仍用原`cfg.disturbance_bound=.04`。如显式传scenario扰动，时间序列入口要求提供scenario_label；历史隐藏phase只能由验证脚本重建用于identity诊断，不进入主规划。

返回核心字段：`x_next,z_next,obs,mean,action,task,reference_task_cost,total,reward,V,excess`。task使用**转移后实际x**的`(p-goal)^2+.005u²+.01v²`；reference_task_cost使用对应znext/uref同式，仅供原reference目标的同池对照。原total中的violation与empty惩罚独立保留，不并入主目标。

返回每拍约束原始量和原容差标志：actual/reference各坐标域margin、input margin、tube margin、实际一步excess、support扰动端点worst-case excess、support可行/空集、discriminant、活动分支间隙、fallback选项/间隙、reference clipping边界等。`actual/reference_domain_violation`保留原`bounds+1e-7`，tube保留原`budget+1e-8`；未修改物理门槛。

## 3. 时间线、候选池与状态输出

```python
predict_timeline(actor_mean, x0, z0, command_timeline, raw_design,
                 goal_preview, start_tick, context, support=bool,
                 disturbance_timeline=None, scenario_label=None)
```

command_timeline为`(...,L,2)`，goal_preview为`(...,L)`，其它初始量按相同leading axes广播。可用于段中时刻，只执行调用者明确给定的命令timeline；内部不规划或偷偷改命令。

输出`x,z`形状`(L+1,...,3)`；`steps`中的所有微步量为`(L,...)`，采用time-major。`task_cost_mean`为L步算术平均，`reference_task_cost_mean`提供相同长度reference目标；另有task sum及total mean diagnostic。没有新增terminal成本。

```python
predict_macro_commands(actor_mean, x0, z0, execution_commands,
                       raw_design, goal_preview, start_tick, context,
                       support=bool, ...)
```

这是MPC wrapper：x0/z0为`(B,3)`，design为`(B,4)`，候选为`(B,N,H,2)`，goal为`(B,H*command_steps)`。命令重复原10拍，原H8即80拍；输出x/z为`(81,B,N,3)`，cost为`(B,N)`。仅用于宏边界，非边界应显式给timeline。

原CEM第二坐标是`[-.45,.45]`，需先调用`planner_coordinates_to_execution`变为`eta=1+s*(.2/.45)`。接口名称特意分开两种单位。

`prefix_violation_summary(prediction)`逐类汇总最坏margin/空集/分支信息，不给混合feasible标签，不选择计划。

## 4. 可微分与不可微分边界

根号对非活跃`discriminant<=0`输入先mask为常数正数，再返回0，保留原forward值且避免inactive backward中的`0*inf`。正discriminant不加epsilon、不放宽区间。`discriminant==0`明确标为非regular，而不声称取得正确光滑导数。

在不可行支持分支，lo/hi元数据仍为原`+inf/-inf`；动作计算在midpoint/halfwidth运算之前将非活跃lo/hi置0，避免`inf-inf`污染未选中分支。真正动作继续取原minimax fallback。stable quadratic根、原`1e-20`交点阈值、原clamp和候选次序均保留。

min/max、reference clip、支持空集切换、fallback argmin本来分段光滑。结果给出branch诊断，不把数值有限梯度说成跨支光滑保证。`geometric_bound`仅重现原绝对tick诊断，不能在任意新初态上升级为新的误差定理。

## 5. 可运行实际规划adapter

```python
ExecutionAwareServo(config, raw_design, seed, steps, support, actor_mean)
```

actor_mean在父类构造前保存，确保初始化首次规划也走实际闭环。planner state从旧`[z,design]`扩为`[x,z,design]`，仍使用原CEM设置、seed和warm-start。预测成本变为实际闭环task mean，**primary可行门槛仍为原reference域margin**。actual/input/tube/support均为诊断，不偷偷加入新硬门槛。

每次目标调用记录已评估candidate及诊断，返回计划后从该池匹配所选sequence，避免为诊断额外推进chosen预测。`execution_prediction_low_steps`必须等于原bridge记录的`candidate_macro_steps*command_steps`。selected未能在已评估池找到时明确保留false标志，不伪造所选诊断。

adapter实际执行继承原step，包含原隐藏扰动；改的是相同冻结actor的预测/规划，不是学习新能力。新反事实仿真、额外actor调用和模型预测成本由验证报告独立记账。

## 6. 验证状态

按[冻结协议](protocol.md)完成首次验证：[结果](validation-results.md)及[逐项JSON](validation-results.json)。全部12组、每组256×16一步映射和最初80步历史scenario的所有核对量均精确一致，最大差0。历史scenario只供identity，不进入主规划。

随后完成全部12组原first-short条件的新actual planner反事实仿真，checkpoint和原生产源码未改。12组task cost均下降，简单平均下降约1.84390%；但4个penalty case的tube违反率上升。该结果只支持预测/规划接口有实际影响，不证明新反馈被学到或全面性能改善。共64512个identity模型微步、49152个新执行仿真转移和13578240个候选预测微步，全部独立记账。
