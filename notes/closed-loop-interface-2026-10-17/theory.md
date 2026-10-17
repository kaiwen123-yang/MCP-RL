# 原生神经执行闭环进入规划：实际任务导数与同信息核验

日期：2026-10-17（Asia/Shanghai）。接续 `e23a28a` 的接口缺口。本稿只推导并规定核验对象，不训练、不运行环境、不修改或发布策略权重。对象是现有PPO checkpoint对应的原生三状态 `ResponseServo`，其参考状态与物理状态合计构成六维增广预测状态；它不是另一个双执行器六状态探针。

## 1. 要修复的是规划对象，而非另加一个鲁棒奖励

原planner只预测reference响应 \(z\)，没有使用神经actor所执行的物理动作，因此原目标对actor直接需求为零。修复后的候选预测必须沿同一命令序列同时推进实际执行状态 \(x\) 与reference状态 \(z\)：

\[
y_j=(x_j,z_j),\qquad
u_j=a_\theta(y_j,c_j;\phi),
\]

\[
x_{j+1}=f_0(x_j)+Bu_j+Ew_j,
\qquad z_{j+1}=R(z_j,c_j;\phi). \tag{C1}
\]

\(\phi\) 是本次固定的native design；\(c_j\) 是按宏周期保持的实际二分量命令。初始 \(x_0,z_0\)、目标序列、design和声明的扰动序列均固定。跨宏段不把 \(z\) 重置为 \(x\)，也不清零执行器状态。

在源码命令坐标下，若宏候选为 \(C_k=(C_{k,0},C_{k,1})\)，实际命令为

\[
c_j=\chi(C_k)=\left(C_{k,0},\ 1+\frac{0.2}{0.45}C_{k,1}\right),
\qquad j\in\text{宏段 }k.
\]

必须记录是否使用该原始坐标或已经转换后的 \(c\)，不能重复缩放第二个命令。

实际任务成本按原 `step` 的时间顺序定义：

\[
\ell_j=(x_{j+1,p}-r_j)^2+0.005u_j^2+0.01x_{j+1,v}^2,
\qquad J(\theta,C;w)=\sum_{j=0}^{N-1}\ell_j. \tag{C2}
\]

若实现采用 \(1/N\) 平均，全部梯度统一乘 \(1/N\)。这里不擅自增加 \(\Delta t\) 系数或终端项；原native目标就是动作后状态的上述任务成本。源码 PPO reward 还包含一步条件超限惩罚和empty惩罚，故 (C2) 的task-only导数不能直接称为已保存PPO的完整reward梯度。若需要比较完整reward，必须把这些项按其实际分支另行纳入同一目标。

## 2. 完整物理动作与原生条件链

原观测包括 \(x,z,x-z\)、两命令、四design变量以及support区间端点，共17维。support端点由 \(x,z,c,\phi\) 决定，且计算过程中使用当前reference后继 \(R(z,c;\phi)\)。因此

\[
o=o(x,z,c;\phi)
\]

不是可在全部预测步固定的一组旧特征。

可行support分支的中心动作是

\[
a_\theta=a_m(y,c)+a_h(y,c)\tanh m_\theta(o(y,c)),
\]

其中 \(a_m=(u_{\rm hi}+u_{\rm lo})/2\)、\(a_h=(u_{\rm hi}-u_{\rm lo})/2\)。design固定、support不直接依赖actor参数时，

\[
a_\theta^{[\theta]}
:=D_\theta a_\theta\big|_{y,c}
=a_h\operatorname{sech}^2m_\theta\,D_\theta m_\theta, \tag{C3}
\]

\[
D_ya_\theta
=D_ya_m+(D_ya_h)\tanh m_\theta
+a_h\operatorname{sech}^2m_\theta\,(D_om_\theta)D_yo. \tag{C4}
\]

命令导数 \(D_ca\) 形式相同，并包括命令经reference后继影响support及观测的路径。penalty组的动作是 \(U\tanh m\)，但其观测仍含support端点；不能删除该观测的状态或命令依赖。

support不可行时，原fallback是 \((x,z,c)\) 的分段函数、不直接读取actor。该分支内 \(a_\theta^{[\theta]}=0\)，但 \(D_xa,D_za,D_ca\) 一般不为零。**不能因为当前fallback与actor直接无关，就切断状态costate或整个未来的参数信用。**以前的actor动作仍能通过当前状态影响fallback和后续任务。

本稿分析确定性中心执行。Gaussian+tanh的真实随机动作期望一般不等于该中心动作；若要分析随机策略期望，需要对声明的噪声/轨迹分布积分，不能继承确定性单条预测的结论。

## 3. 增广状态Jacobian与实际时间顺序

令

\[
A_{0,j}=D_xf_0(x_j),\qquad
a_{x,j}=D_xa_\theta,\quad a_{z,j}=D_za_\theta,\quad
R_{z,j}=D_zR.
\]

固定 \(\phi\) 下reference没有直接actor参数依赖。增广转移 \(F_j\) 的Jacobian为

\[
D_yF_j=
\begin{bmatrix}
A_{0,j}+Ba_{x,j}&Ba_{z,j}\\
0&R_{z,j}
\end{bmatrix},\qquad
D_\theta F_j=
\begin{bmatrix}B a_\theta^{[\theta]}\\0\end{bmatrix}. \tag{C5}
\]

对原Euler物理对象，\(x=(p,v,a)\)，有

\[
A_0=
\begin{bmatrix}
1&\Delta&0\\
-\Delta\cos p&1-\Delta(0.2+0.2|v|)&\Delta\\
0&0&1-\Delta/\tau
\end{bmatrix},\qquad
B=(0,0,\Delta/\tau)^T. \tag{C6}
\]

因此当前动作首先改变执行器状态，不能在同一拍直接改变位置/速度跟踪误差。短时域的task梯度可能主要来自输入成本；实际时序必须由 (C1) 传播，而非通过一个无滞后动作—位置近似制造信用。

令

\[
q_j=\nabla_{x_{j+1}}\ell_j
=\begin{bmatrix}2(x_{j+1,p}-r_j)\\0.02x_{j+1,v}\\0\end{bmatrix},
\qquad b_j=\left.\frac{\partial\ell_j}{\partial u_j}\right|_{x_{j+1}}
=0.01u_j. \tag{C7}
\]

这里区分了动作对下一状态的间接代价与显式输入代价，避免重复计算或漏算。

## 4. 命题1：固定宏命令的反向costate与actor需求

假设所用物理和reference函数、support/fallback及网络分支在所分析轨迹附近可微，初态/命令/外扰/目标/design固定。取无额外终端成本的边界

\[
\lambda_N^x=0,\qquad\lambda_N^z=0.
\]

倒序计算

\[
v_j=q_j+\lambda_{j+1}^x,
\qquad \xi_j=b_j+B^Tv_j, \tag{C8}
\]

\[
\boxed{
\lambda_j^x=A_{0,j}^Tv_j+a_{x,j}^T\xi_j,
\qquad
\lambda_j^z=R_{z,j}^T\lambda_{j+1}^z+a_{z,j}^T\xi_j.
} \tag{C9}
\]

则实际任务的完整actor参数梯度为

\[
\boxed{
\nabla_\theta J
=\sum_{j=0}^{N-1}
\left(a_{\theta,j}^{[\theta]}\right)^T\xi_j.
} \tag{C10}
\]

**证明。** 对 \(\ell_j=g(x_{j+1},u_j,r_j)\) 作全微分：

\[
d\ell_j=q_j^Tdx_{j+1}+b_jdu_j.
\]

用 \(dx_{j+1}=A_{0,j}dx_j+Bdu_j\)、\(dz_{j+1}=R_{z,j}dz_j\) 以及

\[
du_j=a_{x,j}dx_j+a_{z,j}dz_j+a_{\theta,j}^{[\theta]}d\theta
\]

代入 \(d\ell_j+\lambda_{j+1}^{xT}dx_{j+1}+\lambda_{j+1}^{zT}dz_{j+1}\)。整理 \(dx_j,dz_j\) 的系数得到 (C8)–(C9)，剩余参数项为 \(\xi_j a_{\theta,j}^{[\theta]}d\theta\)。跨时刻求和时状态costate项相消；初态固定、终端costate为零，故得到 (C10)。证毕。

若初始 \(x_0,z_0\) 也由 \(\theta\) 决定，需增加 \(\lambda_0^{xT}D_\theta x_0+\lambda_0^{zT}D_\theta z_0\)。本轮固定锚点不含这项。若添加真实已有终端成本，则用其状态梯度作为末端costate，不得仍取零。

### 可选命令导数

在目标 \(r_j\) 不直接依赖候选命令的情况下，固定一步的命令梯度为

\[
\nabla_{c_j}J=a_{c,j}^T\xi_j+R_{c,j}^T\lambda_{j+1}^z. \tag{C11}
\]

宏命令导数为对应低层拍求和后乘 \(D\chi(C_k)^T\)。其中 \(a_{c,j}\) 已包括命令通过当前support的作用，\(R_{c,j}\) 项描述其对未来reference状态的作用，二者不可相互替代。

由于固定命令下 \(z\) 不直接依赖actor，单独检验 \(\nabla_\theta J\) 可能发现不了错误的 \(\lambda^z\)。若要核验完整增广实现，应另核初始 \(z_0\) 或命令的方向导数；这不要求更换任务或优化命令。

## 5. 命题2：真实名义任务梯度与值保持方向的必然关系

把当前完整预测轨迹上固定状态条件的动作参数Jacobian堆叠为

\[
J_{\rm value,current}
=\operatorname{stack}_{j=0}^{N-1}a_{\theta,j}^{[\theta]}.
\]

由 (C10)，

\[
\nabla_\theta J
=J_{\rm value,current}^T\operatorname{col}(\xi_j),
\]

所以

\[
\boxed{
P_{\ker J_{\rm value,current}}\nabla_\theta J=0.
} \tag{C12}
\]

这是对**同一个固定命令、同一条完整确定性名义预测轨迹**的精确结论。若参数方向保持该轨迹上全部实际动作值的一阶不变，因果递推使全部状态及原task成本也一阶不变。只修复实际执行预测接口，不会让该同一名义任务突然给纯值保持斜率方向产生非零一阶信用。

这不否认接口修复的必要性：普通actor动作方向的 \(\nabla_\theta J\) 可以变为非零，planner也可以据实际执行质量改变计划。但它必须准确叫作实际任务的模型化策略梯度，不能直接升级为上一轮关于名义数据盲方向的鲁棒反馈信用。

若投影使用旧评价记录的128个锚点而非当前预测轨迹，投影后梯度可以非零。这通常意味着方向在新规划访问条件下改变了名义动作值；应同时报告这些新条件上的动作变化。它不自动等同于“当前计划值保持但反馈导数改善”。同样，旧选定方向不保持留出轨迹的事实仍然有效。

若显式增加actor参数正则、扰动增益 \(\Gamma\) 或其他不只经名义动作值进入的目标，(C12)可能不再适用；那是目标改变，不能在本轮无新增 \(\Gamma\) 的声明下隐式引入。

## 6. 固定候选池的需求与包络边界

令 \(\mathcal P_k\) 为本轮冻结的宏命令池，定义

\[
V_k(\theta)=\min_{C\in\mathcal P_k}J(\theta,C;w).
\]

若当前赢家唯一、各候选分支可微，并且候选可行性在邻域内保持，则

\[
\nabla_\theta V_k=\nabla_\theta J(\theta,C_k^\star;w). \tag{C13}
\]

**证明。** 有限池唯一赢家具有正成本间隔；连续性与可行集合局部不变保证足够小邻域内赢家保持，从而局部价值就是该分支。证毕。

这只是冻结有限池的局部包络，不是对CEM内部采样、精英选择、均值/方差更新的KKT导数，也不是连续命令域全局最优值。若池由当前策略下的CEM生成，之后冻结池是一个明确的局部外层模型；不称整个自适应CEM算法的精确总导数。

平局、活跃可行边界、候选进出可行域或全池不可行时，需按实际选支/最小违约规则另行分析；不能只报告成本赢家而把不可行候选称为有效MPC解。若未来采用连续命令优化并讨论乘子，需要另给原问题、约束资格、驻点和残差，不能从有限池结果推造乘子。

## 7. 名义w=0、固定扰动和鲁棒W是不同对象

1. \(w_j=0\)：(C10)是确定性名义实际执行预测的梯度。
2. 已知且固定的同一 \(w_{0:N-1}\)：公式不变，但这是该指定扰动下的任务梯度；不能向候选提供基线无法知道的真实未来扰动。
3. 对声明分布取期望：需对相同扰动及策略噪声分布积分，并检查交换微分与期望的条件。
4. \(J_{\rm rob}=\sup_{w\in\mathcal W}J(\theta,C;w)\)：这是不同风险目标。仅在最坏扰动唯一及相应正则条件下，可对当前最坏分支使用包络导数；非唯一时需多活跃方向/次梯度。

support中使用原生 \(W\) 计算一步区间，不意味着以 \(w=0\) 预测的 (C2) 已经是全时域最坏情况任务成本。原一步条件也不自动推出递归可行或全过程安全。本轮不把名义规划修复包装成robust优化已经实现。

## 8. 分支与状态依赖必须进入证据

所用网络为Tanh，但完整执行图不一定全光滑。需要记录并核查support判别式、上下界的active root／硬限幅、feasible／fallback选择、fallback有限候选赢家，以及reference输入clamp的active分支。若分支唯一且有正间隔，可在对应局部区域使用普通导数；若处于切换边界，只能报告实际方向导数或分段结果。

\(v|v|\) 在 \(v=0\) 处有连续一阶导数，(C6)仍成立；其二阶光滑性与一次costate所需条件不同，不能因为存在abs就笼统否定所有一阶导数。相反，平方根判别式为零、min/max并列等真正边界不能忽略。

对于infeasible区间，不能让未选中分支的NaN/Inf通过数值计算污染梯度后再补零。需按与原执行相同的分支构造有效可微局部映射，并把适用条件留在记录中。

## 9. 同信息标准方法应得到同一导数

给标准 model-based policy optimization 相同的checkpoint、完整动作映射、增广状态、命令池、目标、扰动假设、物理模型及精度，对同一固定 \(C\) 展开计算图，反向模式自动微分必须与 (C10) 一致。相等是正确性证据，不是候选比标准法更强的结果。

同信息TO若仍要求 \(u_j=a_\theta(x_j,z_j,c_j)\)，则消去状态/动作约束得到的reduced gradient同样是 (C10)。若TO自由优化每拍 \(u_j\) 或任意反馈 \(K_j\)，它拥有更大的可行函数类，只能另列为理想控制基准/教师，不能将两者性能差归因于导数算法。

这个实现修复可以让实际执行成为规划对象，也可以揭示原reference-only计划的代价误判；但新的最优域、独有反馈信用、学习效率或策略收益都还需要额外证据。完整相同问题经标准工具得到同解应如实报告。

## 10. 无训练独立核验方案

在同一冻结checkpoint、设计、初态、目标和命令序列上，不更新权重：

1. **逐拍对象一致。**完整动作、reference后继、物理后继与(C2)阶段成本对照原生函数；必须核动作后跟踪时刻，不能用reference成本代替。
2. **两个梯度实现。**比较完整展开图反向与(C8)–(C10)的独立局部costate。物理 \(A_0\) 可用(C6)独立写出；动作的local VJP必须通过完整观测/support/fallback，不能沿用缓存旧观测。
3. **增广路径。**若声称实现了命令梯度或完整增广costate，再核(C11)或固定 \(z_0\) 方向，避免只检验actor梯度时漏掉reference通路错误。
4. **固定方向差分。**在内存参数副本上做预声明尺度的固定命令中心差分，核costate方向导数；同步检查active分支。它是导数核验，不是训练或步长搜索，不发布新策略。
5. **候选选择。**实际重评冻结池，报告旧reference赢家、新actual-task赢家、全部成本/可行性、winner gap及是否改变。没有改变就报告未变，不调整候选或代价来制造翻转。
6. **需求与方向分别判断。**报告完整actual-task梯度是否非零，以及选定策略方向的需求是否非零；若用旧锚点投影，明确它不等于(C12)中的当前完整预测轨迹投影。

模型预测计算及导数调用必须计账，不能称零计算；本稿本身只推导，不执行上述预测或环境步骤。通过这些核验最多说明actor—实际执行—规划任务接口确实接通、梯度正确。未训练或发布新策略时，不宣称RL已经学到新能力。
