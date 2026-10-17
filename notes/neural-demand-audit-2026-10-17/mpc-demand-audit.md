# 现有MPC任务是否提出非零反馈微分需求：只读实现审计

日期：2026-10-17（Asia/Shanghai）。本轮读取时HEAD为`6de8bc6a5398ee8f0ab316fd049b9c8887d78d11`。范围为`src/response_ppo/`、其实际SPlaTES规划桥、`matlab/capability/`及现存配置/checkpoint/trace；未调用环境构造器、物理step、MPC、actor前向/反向或优化器，未改生产代码，未新增文献检索。

## 1. 直接结论

**不能从原现有planner及其已保存结果宣称已取得F5的非零反馈需求。**两条实际规划路径都未接入候选扰动响应 \(S=(S_x,S_u)\)、其方向性约束收紧或显式增益目标。它们使用有限CEM，没有产生KKT乘子。代码有非零任务成本和输入/状态限制，不等于已存在

\[
\Lambda_l=\lambda_l r\,t_l/\|t_l\|_2
\]

所需的响应变量、鲁棒约束、扰动集合语义和实际乘子。[候选F4/F5及零需求条件](../../docs/feedback-sensitivity-mechanism-2026-10-17.md#4-mpc怎样提出具体需求)

原神经原型确实有固定扰动界和一步鲁棒support，不能说它“完全没有鲁棒控制”；但该结构不使用当前神经反馈Jacobian更新MPC裕量。六状态MATLAB原型确实在预测中展开经典反馈，不能说其任务与反馈完全无关；但这也不等于F5所定义的扰动响应需求。

在**保持当前planner目标与约束不变**并把S视为独立新增变量的语义下，目标对S没有显式依赖。正确状态是“原实现未定义/未产生F5通道”，不是“已计算出非零λ”；也不能只因缺字段而推断一个假想鲁棒MPC的所有约束都松弛。

## 2. 原神经策略的native planner实际做什么

### 2.1 输入、预测对象与目标

`ResponseServo`为三状态 \(x=(p,v,a)\) 的明确Euler离散对象，不是六状态双执行器对象。actor观察17维，物理动作1维；规划从持续名义状态z及四个解析响应设计参数出发，而不从实际x重置。证据：[`env.py:66`](../../src/response_ppo/env.py#L66)、[`env.py:95`](../../src/response_ppo/env.py#L95)、[`env.py:131`](../../src/response_ppo/env.py#L131)、[`env.py:188`](../../src/response_ppo/env.py#L188)。

规划调用链为：

`ResponseServo._prepare_command → SPlaTESPlannerBridge.plan → 原CEMPlanner/BlackboxPlanner → ResponseServo.sequence_objective`。

每段命令为 \((c,\eta)\)，候选原始两分量均在±0.45，映射后 \(\eta\in[0.8,1.2]\)。预测时域8宏段，每段10步、dt=.02，即1.6 s。预测模型是`response()`中的解析响应律：使用四参数设计产生jerk和参考输入，参考输入裁剪到±1，再由解析离散方程推进。**候选评估不调用actor或其权重/Jacobian。**[`env.py:88`](../../src/response_ppo/env.py#L88)、[`env.py:100`](../../src/response_ppo/env.py#L100)、[`env.py:119`](../../src/response_ppo/env.py#L119)

实际候选任务成本为

\[
J_{\rm native}(C)=\frac1{HK}\sum_{l=0}^{HK-1}
\left[(z_{p,l+1}-g_l)^2+0.005u_{{\rm ref},l}^2+0.01z_{v,l+1}^2\right].
\tag{M1}
\]

代码先推进参考，再计该拍成本。唯一回传给规划桥的约束标量为所有预测前缀、所有三个参考状态的最大超界：

\[
m(C)=\max_{l,d}\{|z_{d,l+1}|-b_d\},
\quad b=(0.6311721,0.9401247,1.2).
\tag{M2}
\]

见[`env.py:112–129`](../../src/response_ppo/env.py#L112)。这些是固定参考域限制；并没有 \(r\|t_l(S)\|\) 或 \(\Gamma(S)\) 计算。

### 2.2 CEM选择与真实返回信息

桥对候选评分使用 \(J+10^4\max(m,0)\)，同时单独保存已检查可行候选中的最低任务成本者。没有可行候选时返回有限搜索结果并明确标记失败。对应源码：[`splates_bridge.py:319–333`](../../src/response_codesign/splates_bridge.py#L319)、[`splates_bridge.py:373–390`](../../src/response_codesign/splates_bridge.py#L373)。

终端value与termination预测均关闭；桥将完整候选成本放在第一项reward，未使用完整SPlaTES的学习奖励/MI算法。配置为原CEM，非iCEM：[`splates_bridge.py:335–370`](../../src/response_codesign/splates_bridge.py#L335)。

`PlanResult`当时在内存中可含首动作、整段命令序列、成本、最大margin、feasible标志和优化器info；实际upstream CEM的info只有最终候选分布`mean,std`，不是Hessian、约束梯度、活跃集或dual。桥额外记录可行候选数量和调用预算。[`splates_bridge.py:77–84`](../../src/response_codesign/splates_bridge.py#L77)、[`splates_bridge.py:393–408`](../../src/response_codesign/splates_bridge.py#L393)；只读核对固定上游`optimizers/cem.py:106–169`，其`info`构造位于165行。

有限罚系数 \(10^4\)不能标为真实约束乘子。即使某个候选超界罚项有方向导数，也只是既定罚函数在该候选上的敏感度；它不自动成为F5的最优值需求，更没有引入缺失的S变量。

### 2.3 已有一步鲁棒support究竟在哪里

环境配置固定 \(W=.04\)、\(\rho=.98\)、\(\delta=4\times10^{-6}\)、\(U=1.2\)。P由固定名义DARE得到，未随神经权重学习。`interval()`使用实际x、名义z及当拍参考，计算对 \(w\in[-W,W]\) 的精确一步二次support，然后将actor的tanh动作映入区间；空区间用明确的minimax失败回退。相关代码：

- [`env.py:20–36`](../../src/response_ppo/env.py#L20)：固定配置；[`env.py:47–63`](../../src/response_ppo/env.py#L47)：固定P/名义K。
- [`env.py:142–157`](../../src/response_ppo/env.py#L142)：support及17维观察；[`env.py:181–197`](../../src/response_ppo/env.py#L181)：实际执行和训练成本。
- [`realization.py:26–85`](../../src/response_ppo/realization.py#L26)：\(V(e^+)\le\rho V(e)+\delta\)的端点约束；[`realization.py:88–97`](../../src/response_ppo/realization.py#L88)：动作映射。

这个W是已有的逐拍标量扰动幅值假设，**不是本次新机制尚未配置的有限时域归一化L2半径r**。可以另行推导两种集合之间的关系，但将它加入F4并求解，是新的规划计算，不能称为原planner已执行。support在固定 \((x,z,c,design)\) 下不含神经θ；其存在不证明高层已经要求θ改变 \(D_x\mu_\theta\)。

PPO训练的实际成本额外有一步违反项和空区间罚项，见`env.py:194–197`。外层四参数响应设计的更新是episode成本score估计，见[`pilot.py:114–123`](../../src/response_ppo/pilot.py#L114)；不是MPC对神经反馈响应S的梯度。

## 3. 六状态MATLAB经典MPC实际做什么

### 3.1 对象与目标

状态为 \((p,q,v,r,a_1,a_2)\)，两个实际动作经过±1.8限幅；每宏命令为一个速度类标量c，默认每.1 s重规划，预测12宏段×5步×.02 s=1.2 s。[`capabilityConfig.m:6–12`](../../matlab/capability/capabilityConfig.m#L6)

低层是已选经典函数

\[
f=d_vv+k_v(c-v),\qquad
g=-\theta_1\sin q-
\left(\theta_2+\theta_3\frac{q^2}{q_0^2+q^2}\right)r,
\quad u^{raw}=(f+g,f-g).
\tag{M3}
\]

见[`capabilityControlLaw.m:1–6`](../../matlab/capability/capabilityControlLaw.m#L1)。theta是经典参数，未加载PPO checkpoint。

预测从每次实际六状态x出发；每拍重新计算该经典反馈，限幅后以 **w=0** 的RK4推进。候选任务成本为

\[
\sum_l\Delta\,[0.3(p_l-p_l^{ref})^2+18q_l^2+
5(v_l-v_l^{ref})^2+0.6r_l^2+0.025\|u_l\|_2^2]
+0.2(18q_N^2+0.6r_N^2).
\tag{M4}
\]

见[`capabilityStageCost.m:1–6`](../../matlab/capability/capabilityStageCost.m#L1)、[`capabilityMPC.m:41–63`](../../matlab/capability/capabilityMPC.m#L41)。参考速度序列为[.65,.3,.8,.2]，切换时刻[0,1.8,3,4.6]，来自[`capabilityReference.m:3–14`](../../matlab/capability/capabilityReference.m#L3)。

### 3.2 约束与实际评分

命令裁剪到[-.2,1.1]；候选可行判据是所有名义拍的raw输入不超过U，且 \(|q|\le.35\)。评分为

\[
J_{\rm cap}+100\,v_{\rm violation}+10^4\mathbf1_{\neg feasible},
\]

其中违反量是raw输入超限平方及角度超限平方的时间积分。64候选、3次CEM迭代、8精英；没有求连续NLP/QP的KKT系统。[`capabilityMPC.m:13–35`](../../matlab/capability/capabilityMPC.m#L13)、[`capabilityMPC.m:50–59`](../../matlab/capability/capabilityMPC.m#L50)

预测中的同一经典反馈会影响名义任务成本与raw输入限制，因此“改变theta可能改变原MPC”是成立的。但原优化变量只有命令序列，theta固定输入；不存在神经策略权重、扰动响应S或可训练反馈Jacobian。改变theta后的名义成本差不是F5对S的实际需求。

实际仿真施加预先列出的扰动脉冲，case3包括−1.5的扰动段；它们由[`capabilityDisturbance.m:3–13`](../../matlab/capability/capabilityDisturbance.m#L3)生成，而预测端显式传0。不能把实际扰动幅值事后直接标成原MPC的鲁棒设计半径。

## 4. 两套接口不能混写

| 项目 | 原训练PPO native系统 | 六状态经典capability系统 |
|---|---|---|
| 物理状态 | 三状态p/v/a，显式Euler | 六状态p/q/v/r/a1/a2，RK4 |
| actor/低层 | 17维观察→1维Gaussian潜动作；support/tanh执行 | 经典 \((\theta_1,\theta_2,\theta_3)\) 函数→2维动作 |
| 高层预测起点 | 持续名义z＋四个解析响应设计参数 | 当前实际六状态x |
| 规划模型 | 解析参考响应，不运行actor | 同一经典反馈与已知物理方程 |
| 高层命令 | c与eta两分量 | 一个速度类c |
| 执行器/上限 | τ=.12，U=1.2 | 默认τ=.1/.1，U=1.8 |
| 规划扰动 | 参考预测无S传播；另有固定一步support | 预测w=0；实际轨迹有扰动 |
| 新候选S/F5 | 未接入 | 未接入 |

因此，不能将六状态旧片段上的5步Jacobian或半径读出称作原17维PPO网络的native MPC需求。若之后要接入，必须定义新的状态/动作接口、动力学与任务对应，不能静默移植checkpoint。源码对比不表示哪一系统更好，仅确认它们不是同一对象。

## 5. 配置、checkpoint与trace能恢复哪些信息

本次直接只读检查了正式PPO目录的24份trace和12份checkpoint。NPZ使用`allow_pickle=False`，checkpoint使用`weights_only=True`；没有实例化PPOBackend或ResponseServo。还检查了MATLAB `study.mat`的字段。

| 保存内容 | 本次核实字段 | 可恢复与不可恢复 |
|---|---|---|
| 24份PPO评估trace | 全部同一15字段schema：task,total,V,excess,empty,action,reference_input,support_width,position,reference_position,actual_domain_violation,reference_domain_violation,tube_violation,target,geometric_bound；短形状(256,16)、长(1024,16)。 | 能审查已执行评估成本、部分状态和support统计；未直接保存完整17维obs、完整x/z、逐步高层命令、所有候选预测/排序、原训练潜动作/logprob/advantage。完整状态若另作确定性重建，必须独立标注并核验；本次没有重建。 |
| 12份最终PPO checkpoint | top keys全为actor、critic、design_mean；actor为17→64→64→1的Tanh网络与一个log_std。 | 能在另行给定合法输入时分析最终网络；不包含MPC的S、r、dual、候选池、旧训练buffer或逐更新策略版本。不能只凭checkpoint恢复原PPO批次的经验秩。 |
| `study.mat`评价3×4 runs | 每run只有theta,cfg,t,x,raw,u,command,disturbance,stageCost,metrics；x=(301,6)，raw/u=(300,2)。 | 有完整实际经典轨迹；未保存12步整条计划、CEM候选/精英、预测全轨迹、每拍最大margin与活跃约束索引。 |
| MATLAB metrics | cost、RMS/peak、饱和率、mpcCalls、noFeasibleSample、时间与转移数等汇总。 | `noFeasibleSample`是有限候选失败次数，不是活跃集或λ；实际raw接近U也不能据此反推名义预测中的KKT乘子。 |

保存逻辑证据：[`pilot.py:33–49`](../../src/response_ppo/pilot.py#L33)只在内存构造训练字段；[`pilot.py:130–153`](../../src/response_ppo/pilot.py#L130)保存评估metrics trace及最终权重；[`capabilitySimulate.m:15–50`](../../matlab/capability/capabilitySimulate.m#L15)只将当前计划留在运行内存，最终run字段没有计划；[`runCapabilityStudy.m:69–70`](../../matlab/capability/runCapabilityStudy.m#L69)保存这些runs。

本次另核六个Python生产源文件的实际SHA-256，全部与[原冻结config.json](../ppo-mechanism-2026-10-16/results/config.json)中记录一致。因此这里读到的native planner不是后来偷偷改过的F5实现。

## 6. 活跃信息、秩与乘子的准确状态

- **当前任务定义与常数：可读。**目标权重、输入/状态界、真实扰动函数、CEM预算与神经模型结构均可从源码/配置确定。
- **已执行实际轨迹的局部是否接近限制：部分可查。**MATLAB完整实际x/raw/u允许只对记录计算实际余量；PPO记录只含部分状态/超限统计。实际余量不等于未来名义候选的活跃约束。
- **原CEM候选的排序与逐行活跃约束：未保存。**本只读子审计没有运行规划。整个研究阶段另按[冻结重放协议](replay-protocol.md)重新执行了原评估并核对记录，其新增规划计算单独计账；即使重现也不会产生原算法没有的KKT乘子。
- **F5的λ和约束资格/rank：原算法不产生，不能由上述汇总恢复。**需要明确的鲁棒约束优化问题、实际分支、完整名义解及相应导数。不能把CEM罚系数、精英排名或`dual_clip=None`当作λ。
- **J0的经验秩：本只读子审计未计算。**本阶段另有[真实网络结果](neural-results.md)，使用重建的确定性评估条件，不能称为原训练batch的秩。
- **新S及其可实现性：不在原planner/checkpoint中。**即使可从已知动力学及某个反馈另算局部S，它也只是新增分析量，尚不是原MPC实际使用的能力模型。

## 7. 对本轮研究决定的含义

此次只读检查不能完成“原MPC提出F5非零需求”的正证据，且给出了明确原因：现有实现没有该需求定义，历史结果没有相应优化证据。应将需求状态报告为**未接入、未验证非零**，而不借普通任务成本或之前非零Γ梯度代替。

### C2：原神经MPC对actor参数的直接需求严格为零

固定原规划输入 \(z\)、四参数响应设计 \(\phi\)、任务预览及同一候选命令 \(C\)，源码定义的是

\[
J_{\rm native}(z,\phi,C),\qquad m_{\rm native}(z,\phi,C),
\]

其计算图完全不读神经actor的 \(\theta\)。因此

\[
\partial_\theta J_{\rm native}=0,
\qquad\partial_\theta m_{\rm native}=0.
\tag{C2}
\]

这不依赖具体checkpoint的数值，也不是自动微分被`no_grad`截断造成的假零：即使撤掉`no_grad`、换符号微分或有限差分，只要保持同一数学回调和输入，仍没有θ依赖。`no_grad`仅是实现层的另外事实。

**同信息、同问题的比较。**普通梯度、鲁棒policy optimization或TO若严格沿用这个原样规划目标/约束、固定φ和同C，不额外引入S、扰动性能项或actor闭环预测，它们对θ看到的同样是零导数。不能把这个结果解释为某个优化器较弱，或声称候选非零Γ梯度因此优于同问题的强求解器。

这里比较的是**相同规划预测问题**。真实执行成本 \(J_{\rm actual}(\theta)\) 仍可能随actor变化；一个直接展开真实actor/动力学的控制方法可以对该实际任务取得非零梯度，但它使用了原native planner没有的闭环预测依赖。不能由(C2)推广出“原任务不需要反馈”或“所有同信息真实任务policy gradient都为零”。

若另外定义Γ、H∞目标或F4约束并计算标准鲁棒梯度，就引入了尚待明确的候选鲁棒任务/预测问题。必须说明该要求的原始任务依据、扰动集合与成本，给同信息强基线相同定义；不能加一项人工权重后把产生的非零数值称为原MPC早已提出的需求。本轮的**完整非零任务需求验证、同目标鲁棒PO/TO比较及闭环性能比较均为NOT_RUN**。

若后续仍在不训练条件下检查真实任务是否需要该能力，首先必须明确哪些原有任务/扰动条件能合法构成F4，以及使用哪套真实策略—状态接口；再有一个明确标记的新离线鲁棒规划/候选评价，才能讨论活跃需求。不能看结果挑扰动半径、追加目标权重或将六状态经典反馈需求移植成原PPO需求。当前审计没有执行这一步，也未授权或启动训练。

### 继续/停止裁决

- **停止当前主张：**不再说“现有native MPC已经要求actor学出反馈微分新能力”，不据Γ局部非零宣布现有闭环更优。
- **保留可研究对象：**实际神经网络若具有值保持且能改变有用反馈导数的局部方向，它仍可用于后续接口设计；本审计没有计算或否定这种方向。
- **下一门槛：**先使实际任务、实际神经策略和规划预测对象在同一接口上形成可核的需求，再比较同信息方法。未过这一门槛，不以额外训练或人工鲁棒权重替代。

**本只读子审计预算：**训练更新0、参数写入0、物理转移0、环境构造0、MPC调用0、策略前向/导数0。整个研究阶段的原评估重放与神经导数核算另见[总报告](../../docs/neural-demand-verification-2026-10-17.md)，不能将本段的0调用用作整个阶段预算。
