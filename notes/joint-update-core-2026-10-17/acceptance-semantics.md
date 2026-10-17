# 联合更新核心的接受语义：确定性任务、完整分母与有限作用域

日期：2026-10-17（Asia/Shanghai）。已读取本轮独立复审及 `independent-review-checks.json`。本稿修正接受/拒绝的数学含义，不运行训练、环境、actor更新或模型拟合，不修改CURRENT_STATE或Git。用户后续获准训练时优先使用服务器的地点偏好保留；它不改变本轮训练为0的范围。

## 1. 唯一的部署改进对象

研究/部署能力对象仍为确定性潜均值经实际执行映射后的控制器：

\[
\bar\pi_\theta(s,c)=\Psi(s,c,m_\theta(o)).
\]

接受依据是**同一物理任务、相同信息及权限下的确定性部署表现**。随机Gaussian PPO提供有真实概率语义的训练提案；其原batch surrogate、探索方差及旧状态上的KL不是另一个必须同时逐步改进的部署目标。

因此从mean局部子问题与有限接受中删除 \(g_S^Td\ge0\) 及“原surrogate绝不下降”的质量门槛。原surrogate、KL及覆盖诊断仍完整报告；预声明的KL/有效域限制可以用于判断更新是否超出当前信息支持，而不能靠事后放宽epsilon让一个反例通过。若更新因信息支持不足被拒绝，应记为 `UNSUPPORTED_UPDATE`，不写成物理任务变差。

独立复审的四样本Gaussian构造已说明：一个满足原KL限额、改进声明确定性目标的非零方向，可能使旧surrogate严格下降。新规则应允许这种**目标冲突被如实记录**，而不是通过改变reward、advantage或数值容差隐藏冲突。它不证明实际随机训练质量不会受损；随机训练指标必须继续单列。

## 2. 两个比较基准不得混淆

一轮有至少三个策略身份：旧已部署配对 \((\theta_B,G_B)\)、随机PPO提案 \(\theta_P\)、mean模型步骤后的候选 \(\theta_C\)。

- **局部mean步骤是否有用：**在固定支持域和相同有限任务上比较 \(\theta_C\) 与 \(\theta_P\)。这判断额外任务步骤是否改善提案。
- **是否替换部署配对：**比较 \((\theta_C,G_C)\) 与旧部署 \((\theta_B,G_B)\)。不能因为候选优于一个较差的PPO提案，就忽略它仍差于旧部署。

模型内比较、解析核算与真实执行比较分别标记 `MODEL`、`ANALYTIC`、`EXECUTION`。模型预测的完成/失败不冒充机器人真实执行；不同证据级别不能直接拼成同一性能表。允许使用旧baseline缓存的前提是任务、案例、扰动、执行映射、策略/模型版本和原始记录都可核对一致。

## 3. 每个入口/案例必须保留的结果类型

对预声明的完整集合 \(\mathcal I=\{1,\ldots,N\}\)，每个方法、每个案例都返回一条记录，不删除未解决者。

| 状态 | 意义 | 完整任务成本 |
|---|---|---|
| `COMPLETED` | 按原物理事件完成，持续条件和所检查合同满足 | 必须给真实定义的有限成本及单位 |
| `OBSERVED_TASK_FAIL` | 已观察到真正的任务失败事件；包括明确不可逆失败或真正任务deadline被违反 | 不伪造“完成成本”；保存失败时刻、实际prefix成本与原因 |
| `HARD_CONSTRAINT_FAIL` | 被任务明确标为硬条件的几何/硬件/其他合同违反 | 独立硬失败标志，不靠平均性能抵消 |
| `EVALUATION_TRUNCATED` | 仅因计算/评价预算耗尽而未看到完成，且未观察到真正任务失败 | 未知；prefix成本可作记录，不作完整成本 |
| `MODEL_UNSUPPORTED` | 无有效模型、缺信息/导数、入口在支持域之外或误差条件不满足 | 未知 |
| `SEARCH_UNRESOLVED` | 有限候选/求解预算内未找到完成方案，但未证明不能完成 | 未知 |

实际实现可把硬失败作为状态或独立布尔字段，但必须能与其他失败清楚区分。仅有仿真/模型轨迹时，应在 `evidence_class` 中明确，不将 `OBSERVED_TASK_FAIL` 误读为已发生的硬件故障。

真正任务deadline与评价截断不是同一件事。任务并未要求400步内完成时，400步评价结束属于 `EVALUATION_TRUNCATED`，不能为方便比较改成失败；有事前物理deadline时，超过它才按该任务定义判失败。

除 `COMPLETED` 外，完整成本字段为 `None` 并附原因。不得填 \(0\)、\(+\infty\)、任意大惩罚数，也不得以prefix成本替代。优化器内部临时过滤值与对外任务结果分开，过滤值不能进入接受平均数。

## 4. 完整分母的有限比较记录

案例及权重 \(w_i\ge0,\sum_iw_i=1\) 在比较前固定。每个case的旧/新结果一一对应。定义

\[
\mathcal C_B=\{i:B_i=\mathrm{COMPLETED}\},\quad
\mathcal C_C=\{i:C_i=\mathrm{COMPLETED}\},
\]

\[
\mathcal U=\{i:\text{旧或新至少一项是TRUNCATED、UNSUPPORTED或UNRESOLVED}\}.
\]

始终报告基于原完整分母的状态权重：

\[
p_B^{\rm known}=\sum_{i\in\mathcal C_B}w_i,
\quad p_C^{\rm known}=\sum_{i\in\mathcal C_C}w_i,
\quad u_B,u_C,
\]

以及旧/新硬失败、实际任务失败、各未知原因、救回案例 \(\mathcal C_C\setminus\mathcal C_B\)、失去完成案例 \(\mathcal C_B\setminus\mathcal C_C\) 的ID和权重。未知不能混进实际失败计数。

有未知时，有限集合上的完成比例只能给出已知区间

\[
p^{\rm known}\le p^{\rm true\ on\ cases}\le p^{\rm known}+u.
\]

这只是未解决案例的逻辑区间，不是总体成功概率或统计置信区间。

双方均完成的交集 \(\mathcal B=\mathcal C_B\cap\mathcal C_C\) 可以报告配对成本差，但必须同时显示 \(|\mathcal B|/N\) 及其权重。该交集平均不是全任务平均，不能据此掩盖其余案例失败或未知。只有整个声明的比较作用域中双方完整成本均已定义，才计算该作用域的完整加权平均任务成本。

## 5. 两条有效研究路线的明确接受规则

下面给一个可执行的默认比较策略。若实际任务有不同的成功率—时间—资源优先级，应在运行前替换该策略，而不是看完结果调整。

### 5.1 共同前提

1. case IDs、任务合同、观测/执行权限、扰动口径、评价预算与证据级别一致。
2. 候选不得违反任务中声明的硬条件。硬条件不能通过平均时间收益抵消；任务若允许风险，必须事先另定该风险对象，不能事后把硬失败降级成软失败。
3. 任何UNKNOWN阻止对整个预声明集合给出 `FULL_CASESET_COMPARISON_PASSED` 或全作用域发布。可以继续报告已知部分与可行见证，但不填补未知比较值。
4. 模型/数值误差检查只使用事前容差，物理合同门槛不因数值结果改变。

### 5.2 路线A：有限集合上的完成能力改善

在没有未知且硬条件满足时，若任务事先采用“完成率优先”，\(p_C^{\rm known}>p_B^{\rm known}\) 可判为 `COMPLETION_PRIORITY_IMPROVEMENT`。原失败必须是真正的已观察失败，不是有限搜索未知。

救回、丢失及仍失败的每个case均按原完整分母列出；不能只报道救回者。完成交集上的时间/资源取舍以及新完成案例的成本另外报告，不给原失败案例虚构一个完成时间。若任务还有绝对性能/风险预算，候选同时满足它们才可进入相应发布状态。

例如13/16到15/16可以是有限集合的完成改善，不要求先把所有16例都做成成功。它不是总体成功概率证明；若实际生产要求全部任务实例确定完成，这一有限成功率改善仍不足以获得该生产保证。

若任务未事先规定完成率与其他性能的优先级，且存在明显取舍，只返回 `TRADEOFF_REQUIRES_OBJECTIVE`，不由算法临时发明权重。

### 5.3 路线B：都能完成时的平均性能改善

当旧/新的完成集合相同且非空、没有未知、没有新增硬失败时，可以对该明确相同的完成作用域比较加权平均物理成本；尤其全部案例都完成时，这就是原完整集合平均。

\[
\bar J_C=\sum_iw_iJ_{C,i},\qquad
\bar J_B=\sum_iw_iJ_{B,i},
\]

这里的求和权重与作用域必须匹配；若仅是固定的已完成子域，应另归一该子域权重并报告其占原集合的比例，不能称为全域平均。

只要满足预声明平均改进条件，就允许个别案例稍慢。例如两例都安全完成，旧成本 \((10,10)\)、新成本 \((8,11)\)，等权平均由10降至9.5，不能因第二例变慢而自动否决。如果任务确有逐案例上限，它作为真正合同另行检查，不等同于要求每例都比baseline更好。

若完成权重相同但成功案例换位，单看交集成本可能产生选择偏差。本默认规则返回 `TRADEOFF_UNRESOLVED`，除非原任务已经给出适用全部终局状态的统一物理效用/比较规则。不能把缺失完成成本补成惩罚数来制造一个平均值。

没有严格改进、只有容差内相同的候选记录为 `NO_CLEAR_TASK_GAIN`；不能把“没变坏”自动写成反馈能力增长。

## 6. 部分接受不等于共享策略全域替换

若完整集合中存在UNKNOWN，可以在**候选评估前固定**的已完成且受模型支持子域 \(\mathcal D_0\) 上计算定义完整的比较。不能看到候选失败后才删掉这些案例来形成一个更有利子域。原完整分母、所有外部UNKNOWN和未通过案例继续保存。

通过时标签为 `PARTIAL_CASESET_PROPOSAL`，附：明确的case/可观测上下文、覆盖比例、比较基准、合同、actor/model版本和域外状态。它不意味着一个共享新 \(\theta_C\) 可以在全部任务上替换旧 \(\theta_B\)。

允许的实现只有两类：

1. 保留旧部署pair，只保存受限候选及其证据，不切换生产策略。
2. 存在真正可执行、使用合法信息的固定scope gate，能在允许的新任务入口选择候选、域外选择旧pair；同时把gate、切换、模型版本和整段闭环纳入验证。若域会在执行中被离开，须验证切换与后续能力，不能默认瞬间回退安全。

只知道16个case ID，不自动得到对新任务可执行的域分类器；只把“新权重适用这些case”写进文档也不能限制共享参数的物理影响。未实现并验证scope gate时，只能保留proposal。

“全作用域”在本稿中最多指预声明有限案例集合或另有有效依据的可观测域，不是全真实世界的概率/鲁棒证书。超出有限集合的能力声明仍需要任务、抽样与误差证据。

## 7. 相同梯度的并列最大值可合并，不同梯度必须保留

设合同聚合为 \(G(\theta)=\max_{b\in\mathcal A}g_b(\theta)\)，当前活跃集为 \(\mathcal A_0\)。在各分支可微时，

\[
DG(\theta)[d]=\max_{b\in\mathcal A_0}\nabla g_b(\theta)^Td.
\]

若所有活跃梯度相同，最大值在当前点可微，重复行可以合并而不丢方向信息。对称正负冲击的同一净空margin就是这种情况，不能因分支数量大于1就拒绝。

若活跃梯度不同，局部约束 \(G+DG[d]\le0\) 等价于对**全部活跃分支**施加对应线性行；不能任取一条，也不能用solver最后给出的某个 \(\lambda\) 来先删掉难满足的入口。活跃行数与数值秩决定实际子问题维数，不继续固定承诺18维。

数值上“很接近”不等于数学相等。若只有数值证据，保留全部无法可靠区分的活跃行是安全的默认做法；若以误差界合并，应把梯度误差乘信赖步长的最坏影响计入约束。不能仅凭一个方便的阈值删去不同方向。

当两个活跃梯度分别为 \(+1,-1\) 时，\(G\) 的方向导数是 \(|d|\)；两行共同保留可能使可行方向只剩零。这是正确的合同结果，不应通过调整乘子或删场景制造可更新方向。

## 8. 全量pullback先保留共享后段直接项

对同一共享actor，单入口完整任务梯度必须为

\[
g_i=\nabla_\theta J_{{\rm pre},i}
+(D_\theta Y_i)^T\lambda_i
+(\partial_\theta V_{2,i})_{Y_i}.
\]

入口空间的 \(\lambda_i\) 相同不代表参数空间的 \(g_i\) 相同。独立复审中 \(\lambda_1=\lambda_2=1\)、\(J_1=1,J_2=-1\) 的抵消例必须保留：全量等权梯度为零，只留一个packet则为 \(\pm1\)。

本轮核心应先实现全量完整pullback及全部活跃约束，不按 \(\lambda\) 相似性预先压缩。以后若研究省算力的选择规则，必须把需求可实现方向、抵消及共享后段直接项纳入误差分析，并与不重复预处理的直接同信息方法比较；不能把全量梯度成本先免费隐藏。

## 9. 纯功能比较器的输出与验收例

比较函数应无训练副作用，仅输入固定case记录、版本和预声明比较规则，返回状态、全部逐case判定及理由：

```text
compare_task_cases(old, candidate, fixed_case_weights, policy):
    validate identity / task / evidence / full case denominator
    retain every status and distinguish failures from unknowns
    reject candidate hard-contract violations
    if any UNKNOWN: return unresolved-full + any valid predeclared partial proposal
    if completion-priority and completed_weight increases:
        return finite completion improvement, with all rescues/losses/failures
    if completed masks match and comparable costs improve in fixed aggregate:
        return finite average-performance improvement
    if objective tradeoff is undefined: return TRADEOFF_UNRESOLVED
    otherwise return reject/no-clear-task-gain
```

必须覆盖的数学验收例包括：旧surrogate下降而共同确定性任务改善；相同梯度并列约束；不同梯度并列约束；参数pullback抵消；混合COMPLETE与UNKNOWN；模型不支持与真正失败的区别；平均改善但单例变慢；旧失败变完成但旧完成成本不存在；部分scope没有gate时不得全域发布。

这些是纯功能接受语义的核验，不是新机器人数据、真实PPO batch或训练结果。规则通过说明代码能够接受应接受的提案、拒绝无依据的声明；不能由此宣布学得反馈、模型泛化或独立算法优势。

## 10. 当前状态

接受目标、UNKNOWN分母、部分作用域、并列约束和完整pullback现已给出明确规则。核心实现及其数学反例应逐项对应这些规则。本轮训练、环境执行、模型拟合和权重发布均为0；后续服务器训练地点偏好不构成本轮执行训练的授权。
