# 全候选独立原生射击法复算：补充协议

日期：2026-10-17（Asia/Shanghai）。在执行本补充核算前冻结。本项只补强同目标计算一致性，不改变原协议、候选、锚点、梯度点、任务或结论标准。此前同池`same_actual_objective_single_shooting_winner_index`只是对同数组argmin；不得把此次追加核算倒写成此前已完成。

## 固定输入与规模

- 使用[已保存同池结果](pool-and-gradient-results.json)中的全部12个检查点、4个tick（0/60/120/180）、16个环境、每查询34条原候选；不得重新建池、换seed或挑case。
- 每候选原H=8×10=80拍，dt=.02；全部26,112条完整候选，预计2,088,960次离线实际状态模型转移。
- 检查点、原重放x/z/design/P/B/E、C池均先核对已存hash；权重冻结并转float64，CPU单线程。原任务目标与已存预览完全相同。
- 名义转移w=0，support仍用原W=.04；不读取或生成未来扰动phase。

## 独立实现边界

通过`object.__new__(ResponseServo)`建立仅有原方法所需字段的holder，直接调用原`f0/response/interval/observation/fallback`，手动推进`xnext=f0(x)+B*u`及`znext`。**不得调用`ResponseServo.__init__/step`、planner、训练器或新增`closed_loop_prediction`的步进/预测函数。**原`step`会更新命令及引入隐藏扰动，不适用于本核算。

可复用已核的纯actor均值展开与原任务目标生成规则，但原动力学、support与观察计算来自原环境方法。每一步对B×N展开批一次actor均值；support和penalty执行各自原映射，fallback保留。x与z持续80拍，不作边界重置。

独立累计转移后actual任务成本、reference任务成本及每前缀actual/reference域、输入、tube、一步excess、support最坏端点excess。共同原reference门槛保持；次级actual/tube/input筛选仅用于复核已报告的共同诊断，不改主选择。

## 冻结验收规则

1. 全部actual/reference成本及六项数值margin与已存数组的最大绝对差≤`1e−10`；检查所有候选，而非只检查赢家。
2. 原reference可行掩码及次级共同筛选掩码一致；空support/fallback标志一致。
3. 全768个query的独立actual成本argmin与已存选择相比，允许完全相同命令的重复索引，不能出现不同命令赢家。reference选择同样复核。
4. 不删除失败case。若有非有限量、hash/结构错误、数值或选支差异，原样登记，继续列出全部输入范围，不放宽阈值或选更有利数据。
5. checkpoint与原生产代码前后hash不变；训练更新、optimizer.step、权重写入、env.step、MPC调用均为0。

## 预算与解释

除实际状态模型转移和actor观察数外，分别报告original observation、显式interval、内部interval、reference response和fallback调用。本实现的observation内部先算interval，随后手动执行再次算interval，含重复几何计算；不得以同候选数或同模型步数称其与新接口等墙钟成本。报告实测墙钟，但不作独占实时率benchmark。

通过意味着**保存的全部固定候选成本/约束/选支可由原生方法独立积分复现**，不是另一个连续TO优化器已运行，也不是学习收益、性能优势或最优性新证明。新的离线计算单列入总账，不算新增真实交互；所有报告均保留原候选与该补充协议的hash。
