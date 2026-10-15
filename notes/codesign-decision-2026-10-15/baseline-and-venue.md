# 算法起点与目标期刊：任务驱动闭环动力学共同设计

核验日期：2026-10-15。科学问题和“MPC规划、RL执行”结构沿用用户已确认的决定。本轮完成官方源码只读检查、版本与许可证核验及期刊官方要求核验；没有安装依赖、运行外部算法、训练策略或测量推理时间。

## 1. 唯一主baseline建议

**选择 SPlaTES（Long-Horizon Planning with Predictable Skills，RLC/RLJ 2025）作为算法改造的主代码起点，并把其完整原版保留为强对照。** ReCo 是响应塑形问题上最近的科学对照，HIM 是以后连接真实四足执行的候选参考；两者都不替代这个主代码起点。

这个选择有明确理由：SPlaTES已经同时提供高层MPC、技能条件反馈控制、闭环技能预测模型、训练数据流与通用机器人环境。我们能够在同一实现中区分“学习可预测技能”与“共同设计有约束、可实现、对任务有用的响应”，避免先花大量时间从零拼出完整MPC+RL系统。它是强对手，并不是容易击败的弱基线。[正式论文](https://rlj.cs.umass.edu/2025/papers/Paper136.html)、[作者项目页](https://nicoguertler.github.io/splates-pages/)、[官方实验仓库](https://github.com/nicoguertler/SPlaTES)。

**关键结构差异必须保留：SPlaTES默认低层也是TD-MPC2在线规划，不等于用户已确定的神经策略执行结构。** 官方源码确有直接使用actor的模式，因此可以先在现有学习器上获得结构匹配的适配版；但切换后行为与闭环响应可能改变，不能把适配版称为原论文复现，也不能假定性能保持不变。

需要分别记录三项，虽然主要工程底座只有一个：

| 角色 | 具体选择 | 用途与边界 |
|---|---|---|
| 主代码起点、必须保留的强对照 | SPlaTES完整原版，高层iCEM＋低层TD-MPC2规划 | 证明我们没有依靠降低对手能力获得优势 |
| 结构匹配的改造起点 | SPlaTES官方actor-only执行路径＋原高层规划 | 匹配MPC规划、RL神经执行；重新评估/学习对应响应，明确标注适配版 |
| 科学最近对照 | ReCo式固定响应塑形＋策略冻结后辨识＋MPC | 测试“任务共同决定响应”比固定响应塑形增加了什么；当前未取得官方实现，不声称可直接fork |

理论阶段另采用固定响应模型＋受限反馈实现＋同一任务MPC作为可解析对照。它是定义明确的数学控制组，不冒充任何论文的完整复现。特别要允许固定模型在相同任务、相同参数族和相同预算下充分整定；不能故意用一个差的固定响应衬托共同设计。

## 2. 代码来源、版本和许可

已使用认证`gh`读取GitHub；API取得的当前默认分支SHA与本地已有SPlaTES/LayeredRL副本一致。所有外部代码留在忽略目录`work/`。未运行其安装脚本或训练代码。

| 仓库 | 2026-10-15核验SHA | 许可与取得状态 |
|---|---|---|
| `nicoguertler/SPlaTES` | `90b3749abd318ff3817f14159e207d6aa6f3dc1d` | 根LICENSE为MIT；实验脚本/配置已在本地 |
| `nicoguertler/layeredrl` | `37111d3538e18e4c6bcf1d98eef07407624c2e77` | 根LICENSE为MIT；内含TD-MPC2目录也有许可文本；实际算法所在 |
| `nicoguertler/long-horizon-gym` | `f6ba5d9870d7aa7e2e3b59e55d7fe15cceceab34` | 根LICENSE为MIT；本轮只读clone取得环境源码 |
| `InternRobotics/HIMLoco` | `ef289acaa62795009363b7b819c9186690630441` | 根LICENSE为CC BY-NC-SA 4.0；本轮只读clone取得，README中部署指导仍未勾选发布 |

许可原始证据：[SPlaTES LICENSE](https://github.com/nicoguertler/SPlaTES/blob/90b3749abd318ff3817f14159e207d6aa6f3dc1d/LICENSE)、[LayeredRL LICENSE](https://github.com/nicoguertler/layeredrl/blob/37111d3538e18e4c6bcf1d98eef07407624c2e77/LICENSE)、[Long-Horizon Gym LICENSE](https://github.com/nicoguertler/long-horizon-gym/blob/f6ba5d9870d7aa7e2e3b59e55d7fe15cceceab34/LICENSE)、[HIMLoco LICENSE](https://github.com/InternRobotics/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/LICENSE)。以上只记录仓库文本，不把根许可视为对所有第三方依赖/资产的完整法律审查。

本地副本：

- `work/four-paper-audit-2026-10-14/splates-code/`
- `work/four-paper-audit-2026-10-14/layeredrl-code/`
- `work/codesign-decision-2026-10-15/long-horizon-gym/`
- `work/codesign-decision-2026-10-15/HIMLoco/`

SPlaTES的`pyproject.toml`要求Python≥3.10、PyTorch≥2.0、Gymnasium≥1.1和gymnasium-robotics；两个Git依赖没有固定SHA。LayeredRL还依赖`tianshou==0.5.1`、torchrl、tensordict、scipy等。因此“有代码”不等于“当前依赖组合已复现成功”。正式复现时需要固定上述提交和依赖版本；本轮没有完成依赖兼容或Blackwell GPU验证，不给训练小时承诺。[依赖声明](https://github.com/nicoguertler/SPlaTES/blob/90b3749abd318ff3817f14159e207d6aa6f3dc1d/pyproject.toml)。

## 3. 现成的算法接口与具体改造位置

### 3.1 已有完整接口

1. `SPlaTES/get_hierarchy.py`的`get_planner_level()`组装编码器、动力学ensemble、reward/value predictor和CEMPlanner；`get_splates_hierarchy()`组装`[PlannerLevel, SPlaTESLevel]`。这已经提供上下层数据流。[源码](https://github.com/nicoguertler/SPlaTES/blob/90b3749abd318ff3817f14159e207d6aa6f3dc1d/get_hierarchy.py)。
2. `layeredrl/levels/planner_level.py`的`get_action()`进行高层滚动规划，`process_transition()`与`learn()`记录技能层转移并训练预测器。这里是加入明确的可实现响应域和轨迹约束的接入点；原有iCEM采样范围限制不是整个机器人系统的硬约束证明。[源码](https://github.com/nicoguertler/layeredrl/blob/37111d3538e18e4c6bcf1d98eef07407624c2e77/layeredrl/levels/planner_level.py)。
3. `layeredrl/levels/splates_level.py`已有技能执行计时、技能起始状态、技能条件动作、奖励重算及学习入口。`get_reward()`约第597行在`torch.no_grad()`下用高层预测器构造技能奖励；因此原有模型—策略耦合不能被说成高层任务梯度直接贯穿整个系统。[源码](https://github.com/nicoguertler/layeredrl/blob/37111d3538e18e4c6bcf1d98eef07407624c2e77/layeredrl/levels/splates_level.py)。
4. `layeredrl/predictors/reward_predictor.py`分开处理动力学一致性、奖励和值损失；约287行模型训练状态detach，355行value输入detach。这允许定位哪些变量由任务影响，但不能简单删掉detach就声称得到正确的共同优化。[源码](https://github.com/nicoguertler/layeredrl/blob/37111d3538e18e4c6bcf1d98eef07407624c2e77/layeredrl/predictors/reward_predictor.py)。

### 3.2 官方actor-only路径：真实存在，但需要公平处理

- `SPlaTESLevel.__init__`约173行参数`use_policy=False`，动作函数约701–708行传给TD-MPC2。
- `layeredrl/tdmpc2/splates_tdmpc2.py`约74–77行：若`self.cfg.mpc and not use_policy`，调用`plan()`；否则直接调用`self.model.pi(...)`生成动作。[永久链接](https://github.com/nicoguertler/layeredrl/blob/37111d3538e18e4c6bcf1d98eef07407624c2e77/layeredrl/tdmpc2/splates_tdmpc2.py#L74)。
- `SPlaTES/test.py`公开`--use-policy`选项，约77–78行设置技能层的`use_policy`。[永久链接](https://github.com/nicoguertler/SPlaTES/blob/90b3749abd318ff3817f14159e207d6aa6f3dc1d/test.py#L77)。

这条路径仍是学习的反馈策略：它基于当前观测的编码、技能、技能开始状态和执行阶段输出动作。不是把一串固定动作播放到底。与此同时，训练仍可能使用TD-MPC2的世界模型和critic，不能把“部署不调用低层MPC”写成“训练没有模型学习开销”。

推荐处理：先保持原学习器，采用官方actor-only路径研究结构匹配版本；不在第一轮就改成PPO。LayeredRL虽然有`TianshouLevel`通用RL接口，`SPlaTESLevel`却继承TD-MPC2并依赖其专用buffer、技能条件变量和跨技能bootstrap。换PPO/SAC不是替换一行配置，需要重做学习接口，当前不应把这项额外工程包装成创新。

### 3.3 真正需要改的科学部分

| 现有机制 | 我们需要增加或改变的对象 | 不能冒充的结论 |
|---|---|---|
| 通过技能可预测性/多样性目标学习技能 | 响应参数和反馈策略受同一实际可实现关系约束，并由完整任务选择响应 | 改奖励权重本身不等于共同设计 |
| 预测技能结束后的抽象状态 | 若理论/任务需要，给出执行过程的误差与输入占用描述，处理阶段之间的接续 | 终点准确不自动保证中途约束 |
| 采样式高层规划 | 规划限制在已实现或有证据支持的响应范围，明确优化误差 | 一个置信分数不等于鲁棒可行域 |
| 固定actor-only切换 | 在实际执行机制下重估响应，并比较完整原版与结构匹配适配版 | actor-only降性能后赢适配版不等于赢SPlaTES |

因此主底座可以复用数据流、学习器、通用环境和可重复的基准；新算法必须独立定义共同优化变量与实现条件，不能由几处源码改动来决定科学问题。

## 4. 为什么没有选ReCo或HIM作为主代码底座

**ReCo：科学距离最近，当前代码可用性不足。** 本轮复核作者arXiv HTML方法、参考文献和链接，只有OCS2与对比方法qm_control的GitHub入口，没有ReCo官方代码入口；认证GitHub仓库搜索`ReCo locomotion`未返回匹配。应写“本轮未定位官方实现”，不能写“作者一定没有公开代码”。原文结构是先响应塑形、再冻结策略辨识，再用MPC协调5维机体命令和6维机械臂速度。完整硬件复现需要腿臂系统及相关接口；用户分别持有Go2、Piper H、M1，不代表已经有论文同构的腿臂整机。不能把安装机械臂作为开展本研究的前提。[原文方法与实验](https://arxiv.org/html/2610.01612v1)。已有全文阅读见[ReCo笔记](../four-paper-audit-2026-10-14/ReCo.md)。

**HIM：执行器参考很好，但缺少本研究的规划和响应共同设计部分。** 源码有`him_ppo.py`、`him_actor_critic.py`、`him_estimator.py`，当前环境目录是A1、Aliengo、Go1，没有现成Go2配置。README报告的验证栈是Ubuntu20.04、Python3.7、torch1.10/cu113、Isaac Gym Preview4，并非当前Blackwell训练环境已验证方案。部署指导仍标待发布。把HIM接上现有MPC，仍要自行实现命令条件响应模型、上层任务数据流与联合优化；不能称为已具备完整基线。[官方README](https://github.com/InternRobotics/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/README.md)。

这两个判断不等于否定HIM能力或ReCo科学价值。选择主底座的目的是尽早判断新机制有没有增量，避免先承担平台移植和完整系统重建成本。

## 5. 唯一冲刺期刊：IEEE Transactions on Robotics（T-RO）Regular Paper

**按当前成果形态，主目标选T-RO，不同时准备四种投稿故事。** 预期成果是机器人执行能力的学习—规划共同设计方法、清楚的条件性理论、通用仿真中的可辨别增益，以及实机的连续任务与约束证据。机器人中的真实执行问题是贡献主体，理论服务于解释能力与规划之间的关系，这与T-RO涵盖计算机、控制、数学和机械等学科的机器人基础研究范围相符。[官方作者说明](https://www.ieee-ras.org/publications/t-ro/t-ro-information-for-authors/)。

下面“适配性判断”是基于本项目预期成果的研究判断，不是编辑承诺或录用概率。

| 刊物 | 官方范围/本轮可核要求 | 对本方向的判断 |
|---|---|---|
| **T-RO，主目标** | 机器人各方面的基础性研究；Regular首次提交≤18页，最终≤20页，最终超过12页有超页费；当前作者页面采用匿名投稿要求 | 最适合方法＋条件性理论＋真实机器人能力结果的完整叙事 |
| IEEE TAC | 控制理论、设计和应用；Full Paper通常12页、最多16页（含规定的作者材料）；新颖性和贡献重要性进入预筛 | 只有当核心结果成为较一般的受限系统共同设计定理/算法，机器人仅作例子时，才比T-RO更自然 |
| Automatica | IFAC官方介绍强调系统与控制中理论和应用的长期价值，含regular/brief/survey等类型 | 若主要贡献转为普适控制理论则匹配；本轮Elsevier作者细则访问失败，未核当前页数，不能借第三方数字补齐 |
| IEEE/ASME T-Mech | 官方强调机电系统理论与实践、智能控制的协同集成；作者页写初稿最多15 Transactions页，Regular最终超过8页有超页费 | 若主要增量最终落在具体系统实现与集成，更匹配；当前已确认的问题更希望形成可推广的机器人控制方法，因此不选作主冲刺目标 |

其他官方来源：[TAC作者说明](https://www.ieeecss.org/publication/transactions-automatic-control/author-info)、[IFAC期刊介绍](https://wwwold.ifac-control.org/publications/journals)、[T-Mech范围](https://www.ieee-asme-mechatronics.info/)、[T-Mech作者说明](https://www.ieee-asme-mechatronics.info/information-for-authors/)。Automatica当前[官方作者指南入口](https://www.sciencedirect.com/journal/automatica/publish/guide-for-authors)和[范围入口](https://www.sciencedirect.com/journal/automatica/about/aims-and-scope)本轮未取得正文；不使用名称相似的`sciencedirectelsevier.com`作为Elsevier官方证据。所有页数与规则在正式投稿前再核；未查询/使用影响因子、中科院分区或录用周期。

### 要达到T-RO目标，需要什么结果

1. 共同设计让机器人获得原有固定响应或独立训练难以取得的任务能力，且收益不能由更多训练、更强模型或不公平的在线计算解释。
2. 可实现条件在输入受限、有扰动的系统中有清楚的意义；局部或规定范围内的结论必须与实机适用范围一致，不泛化成全局接触安全。
3. 原版SPlaTES、结构匹配适配版、充分整定的固定响应对照各有作用；不能只保留最容易击败的一项。
4. 实机证据回答预测模型怎样改变了规划、执行策略怎样实现了要求、哪些条件下失败；并报告整个控制链的延迟和控制输入，不能只给网络推理耗时。

这些是本研究为支撑目标期刊应准备的证据，并不是声称T-RO所有稿件都强制实机或规定固定数量的机器人。

## 6. 当前结论与未核边界

- 已确认：主代码起点SPlaTES具备真实MPC/RL/闭环响应接口；官方actor-only通路存在；源码和许可可取得；通用环境不要求额外购置硬件。
- 已确认：ReCo是直接科学近邻，但本轮没有定位官方实现；HIM缺少上层MPC与本研究的共同设计算法。
- 尚未确认：依赖是否在用户训练设备上可直接运行，SPlaTES原版性能能否复现，actor-only性能和真实响应变化，修改后训练/推理成本，新机制的实际增益。
- 不作承诺：存在代码不等于实现成功，当前问题有意义不等于已完成创新判定，选T-RO不等于保证录用。

下一项工程动作应由理论与联合更新定义驱动：固定以上代码提交，建立原版/actor-only的明确配置差异，然后把共同设计的响应变量、可实现条件和数据记录接入既有接口。当前没有启动这项训练或改动外部源码。
