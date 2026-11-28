# W59 学习分支审查（2026-11-28）

固定仓库快照：`c8378beae4be371eb4f104782b75e5d295b6f59a`。本文件为只读审查；没有训练、改动控制代码或改变W59结果。正文事实来自固定提交的JSON、56个实际update文件、相关源码；PDF缓存只作本轮阅读，不需推送。

## 1. 最终能力

W59最终科学报告于2026-11-28 07:25:58（上海）记录`earned_chain_terminal`，学习范围`partial`，不是matched四臂完成。旧dispatch仍是11月26的dispatching记录，不能盖过最终结果。

| seed | 实际用于更新的distinct事件 | optimizer batches | 实际接受actor步 | real critic步 | 冻结评价节点 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 5901 | 5888 | 23 | 272 | 368 | 4096 | 向8192推进时WBC数值中断，使用最后完整登记节点 |
| 5902 | 8192 | 32 | 366 | 512 | 8192 | 完成最高登记节点 |
| 5903 | 256 | 1 | 0 | 16 | 无 | 首次真实参数更新方向按原规则拒收并回滚 |
| 合计 | 14336 | 56 | 638 | 896 | 两个冻结head | imagined actor/critic/transition均0 |

三seed观测到闭合事件14634、请求14637、尾样本排除274，不能将observed或优化epochs重复计入distinct。5903在KL层曾接受9个proposal step，但实际首更新回滚后actor贡献是0。不能把`actor_steps_accepted_by_KL`直接作为实际总接受步数。

5901冻结策略SHA为`a870c1550f3dae64016f2a9f1014951b638793978f6b2f4260c9dfb5a16a90e2`，对应policy-016/4096；5902为`0073814da158722078464fdd728bb0e2585dc2d8cc472042cc521169b7665e9c`，policy-032/8192。两者的WM列表均空。不能把attempt中旧W44底层actor的checkpoint SHA当这两个新三维actor的SHA。

两个冻结head各自A/B E30/E60全部通过。预定展示seed5901的B E300/E600通过；A在原E60上CoT和滑移两项同时恶化，所以依原规则不延长。5902未预定长测，不能计作长测失败。原合同和实际结果都不支持将三seed称作全部成功。

## 2. 步态/机制收益

8个learned heldout均走完对应168s或288s窗口、23次body traversal并home_stable=true；严格机制8条都未通过：

| head | A short | A recovered | B short | B recovered |
|---|---:|---:|---:|---:|
| 5901@4096 | 2/12 | 4/12 | 4/12 | 3/12 |
| 5902@8192 | 3/12 | 3/12 | 3/12 | 0/12 |

合计22/96是固定目标暴露/匹配数，不是22次完整任务成功。只看`success=false`会把机制未达标误说成摔倒。与同case/history的B3相比，8组learned heldout接触点滑移变化范围为−1.741%到+0.901%，正机械CoT为−0.490%到+0.216%，方向混合且每工况只有一个确定回合，尚不支持明显、普适效率收益。

E600的控制系统确实完成10次材料往返且持续观察合格。源码在mission第一次complete时调用LiveEventPolicy.complete，随后absorbed=true，新的request直接回base_request。因此不能把600s全说成持续PPO运动。index的full_stable_s来自mission.final_stable_s，ContinuousMission将它定义为末段连续稳定时长，不是完成时间戳；已由Release原件originals-021.zip再次核验：任务在471.444s完成，冻结PPO实际评价请求1534个，之后128.556s为完成后的资格观察；末段稳定130.556s包含完成前约2s，不是600s持续行走。

## 3. 必须分开的三个问题

### 3.1 5901中断是WBC数值收敛问题

attempt `w59-r3-5901-ppo-full-fallback-b-e60-0023` 在36.774s：
`ControllerNumericalFailure: wbc: maximum iterations reached`。
发生在源码runtime `4ad3a5f28b99ad9ccc84455915fdffa9dfcb57af` 的WBC QP求解链。分类technical_interruption/execution_error，不是已观测跌倒，也不是ENOSPC。当前审查未看到该精确状态的反事实solver重放证据，不能断言只是增加迭代就能修。下一阶段针对这个已知状态比较缩放、热启动、原容差/迭代与有限预算求解，而不重做整个W59。

5902另有ENOSPC恢复：保存policy019、优化器、118个已闭合事件；新回合补138条成256，边界GAE截断且bootstrap；真实首恢复update020接受8步，imagined0。恢复不等于重复使用118条数据。

### 3.2 首更新小负均值不等于整个seed不能学

5903首更新8状态平均−0.0000402145，paired95%区间[−0.000236372,+0.000184993]，前后hard failure均0。当前代码实际上按mean_delta>replay_floor（0）接受，并不按置信区间或实际意义阈值。5903因此按旧规则停止，不能改旧结论；但这不是统计上证明更新有害，更不是证明该seed无法训练。

下一轮将这种8状态检查作为诊断并保留全部seed，按照事前固定相同预算训练；只有已定义物理/数值终止或更新自身约束触发时处理。不按首步极小收益正负筛seed。这样并非放松W59旧标准，而是新研究采用更符合比较目的的协议。

### 3.3 学习任务和机制目标并不完全相同

当前reward是原W38的路线进展/往返、耗时、压陷、滑移、旧cap变化和物理失败；并没有直接优化12个指定跨足重载目标。动作仅due offset/ramp/load preference，无XY落足权限。因而“路线成功、机制暴露不足”不能简单归因为PPO年代老或训练量不足。

critic输入主要是elapsed/progress/mission phase/Fz/hold，未包含完整材料历史与执行器/控制记忆；是否导致价值混叠与高方差是待测假设，不把它直接当已证明根因。下一轮可以在同输入权限的共同critic上比较更新算法，并单独消融更完整因果状态/历史critic。

## 4. 新发现：探索seed的高32位别名风险

当前源码使用：
`seed = learning_seed * 2**32 + 2*serial + case_index`，
再给CPU的`torch.Generator().manual_seed(seed)`采样。

核对官方PyTorch源码：
- CPUGeneratorImpl::set_current_seed直接设置`engine_=mt19937(seed)`；
- MT19937RNGEngine构造调用`init_with_uint32`；
- 初始化`state_[0] = seed & 0xffffffff`。

在这套实现中learning_seed位于高32位，其信息不进入初始状态；相同case/serial跨学习seed的噪声流相同。三个seed首batch return_mean精确同为8.6787033081、ramp饱和均17/256，与此一致。不同网络初始化和后续参数更新仍使训练不同，不能说三条训练完全重复或全部无效。

当前审查环境没有torch，尚未在AutoDL精确版本或原raw动作上做现场验证。下一轮用能把seed混入有效位的组合hash/SeedSequence或独立持久Generator，保存随机流指纹；相同scenario跨算法若故意使用common random numbers则明确记录，独立seed之间使用独立流。只修新研究身份，不覆盖W59历史。

官方来源：
https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/MT19937RNGEngine.h
https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/CPUGeneratorImpl.cpp

## 5. 对下一轮学习线的可执行建议

1. 保留原真实PPO作为基线；新线先在固定执行链上实现独立工作的score/自然梯度版本，不等待WM路径梯度。
2. 首批比较同权限PPO、SPO和所提更新算法，统一critic/奖励/动作权限、固定相同样本节点和至少3个seed；新策略采自己的on-policy数据，不复用W59轨迹冒充on-policy。
3. 先在现有连续事件接口上验证更新核；新增接触模式/轨道参数后，全部对照拥有相同混合动作接口。不能把多了动作权限算作新优化器收益。
4. 主学习问题应包含任务进展、能量/滑移或明确尾风险目标，而不仅是“模型梯度不报错”。评价既看固定数据预算也看端到端耗时。
5. 模型线与学习线并行。世界模型可服务值估计、规划teacher、score imagination和局部路径梯度等不同角色；不因为路径梯度未准入就封死全部模型研究，但每种角色仍需检验它影响的预测/决策偏差。
6. 联合时使用相同模块分别训练vs共同训练、score-only vs加模型导数、原信任几何vs执行响应几何，保留真正能归因的比较。

56个update记录中的optimization_wall_s合计约6.114s，只包括更新内部计时、不是全部训练成本。这表明给当前小MLP继续堆显存未必提高整体吞吐；应把GPU用于材料族/世界模型批训练、并行规划和未来大规模采集。真实物理采集、日志/存储和求解开销由各自profile决定，不由显存利用率推断。

## 6. 四篇直接相关论文的方法复核

以下是本次重新读取的核心方法与实验，不冒充全部新增文献。PDF已实际取得并提取正文。

### DreamerV3 — Nature 2025
来源：https://www.nature.com/articles/s41586-025-08744-2
阅读：主文世界模型/actor/critic、消融、Methods中网络/优化器/replay。
RSSM用循环状态和离散随机潜变量，训练重建、奖励、continuation及双向stop-gradient KL/free bits。正式版连续和离散actor均用REINFORCE，模型用于想象回报而非强制对动力学做路径求导；分布critic、symlog/two-hot和分位数回报缩放稳定多尺度训练。
借鉴：世界模型和策略导数路线可解耦；任务相关与任务无关状态监督共同保留。限制：score imagination仍有模型偏差，不能绕过W59前向失真；大量通用仿真/游戏结果不是Go2材料接触的实机证明。

### Simple Policy Optimization — ICML 2025
来源：https://proceedings.mlr.press/v267/xie25m.html
阅读：§4–6、Algorithm1、主要对照与深网络实验。
将PPO平顶clip替换为带恢复方向的比例二次目标：
`rho*A - |A|*(rho-1)^2/(2*epsilon)`。
实现变化可很小，适合保留采样/critic/动作的一致性做强对照。其标量概率比目标性质不能自动升格为任意神经网络、多epoch、接触机器人全局单调保证。借鉴重点是更新方向而非更大模型。直接使用SPO只算基线，不算本项目创新。

### HPO / Mixed-Gradient Policy Optimization — 2026
来源：https://arxiv.org/abs/2605.14297 ，官方NeurIPS材料使用MGPO名；本文技术阅读以arXiv正文为准。
阅读：§2–5、估计器伪代码、Appendix C.2实现、数值问题。
随机离散模式+每模式连续输出；连续梯度不仅有pathwise cost项，还包含连续控制改变未来状态、进而改变离散策略概率的cross term。实现连续分支每batch一次全批更新，离散/critic追加PPO epochs；其无偏定理依赖真实平滑转移与正则性，近似WM不能直接继承。
借鉴：适合将来接触模式+轨道参数+持续时间联合信用；当前仅3连续动作不是论文高维优势的主要适用区。库存与切换LQR结果不是足式硬件结果。

### Dream-MPC — ICML 2026
来源：https://proceedings.mlr.press/v306/spieler26a.html
阅读：§4–5、Algorithm1、消融/计时、Appendix A局限；正式PMLR PDF。
策略产生少量候选，latent梯度优化，重用上时刻动作并惩罚Q ensemble分歧；政策先验强弱直接影响效率。正式文中Acrobot/RTX4090计时Dream-MPC(BMPC)18.15ms、BMPC18.77ms，模型调用数减少不能等同同倍墙钟加速。借鉴：冻结W59 PPO可作为候选提议来源；局部梯度只在相应模型可信时用，另保留无梯度比较。要求强policy prior，不能靠多步梯度补救漂移模型。

## 7. 可直接引用的仓库入口

共同前缀：
https://github.com/kaiwen123-yang/MCP-RL/blob/c8378beae4be371eb4f104782b75e5d295b6f59a/

- results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w59/outcome.json
- .../w59/learning/learning-series.json
- .../w59/learning/pre-heldout-frozen.json
- .../w59/learning/seed-5903/PPO-full-fallback/first-update-real.json
- .../w59/attempts/w59-r3-5901-ppo-full-fallback-b-e60-0023.json
- .../w59/learning/recovery-20261127.json
- scripts/s1_w59_learning.py
- scripts/s1_w59_learner_update.py
- scripts/s1_w59_learner_series.py

## 8. 建议E线首批固定预算矩阵（供整合者与其他线共同排程）

主比较三个臂：同接口PPO、SPO、所提事件策略/自然梯度更新。每臂3个独立学习seed，每head固定8192条实际用于更新的事件，共9 heads、73728 distinct used事件；2048/4096是公开学习曲线节点，不按微小正收益决定是否让某个seed继续。物理/数值故障按既定规则记录，未完成head保留分母，不暗补成满预算。若新混合动作接口尚未准备好，先在固定连续事件接口完成更新内核比较，再在统一新接口上重新比较所有臂，不能把前者称完整混合创新。

三个臂的奖励、critic表示、观测权限、动作权限、初始行为、采样噪声和每head样本预算一致；资源比较同时报告总physical time、wall、GPU/CPU。选择3个新seed并明确探索流：同seed跨算法可用预注册common random numbers，跨seed独立；所有实际采集分别计费。最终评价使用固定相同工况和预算，W59已读保留集只作为开发资料。下一层联合模型导数另作score-only/添加模型导数的匹配消融，学习主线不等待这层。

这9个head是工作矩阵，不是同时9个重进程。沿用共享租约，允许2–3个双collector campaign并行，材料和模型GPU工作按现场余量同驻留。开始前只对已发现问题做直接验证：seed映射、已知WBC failure状态、批数据身份和自然恢复边界；不增加无依据的全仓测试或任意后台结束时刻。
