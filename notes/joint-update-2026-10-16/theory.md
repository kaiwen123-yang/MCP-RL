# 联合更新的有限步实现关系、误差条件与估计边界

日期：2026-10-16（Asia/Shanghai）。本文服务于本轮具体联合更新算法，不运行训练、不生成新物理轨迹。所用物理对象沿用已有双执行器探针。下面的执行器可达性、换测度与三角不等式是经典工具；本轮工作是把它们放到同一个“策略改变—输入占用—执行记忆—模型发布”对象中，并明确哪些条件实际改变能力需求。它们不构成已证明的全机器人新定理或学习收益。

## 1. 对象、量词和固定条件

低层周期为 \(\Delta\)，一个命令段有 \(K\) 拍。两执行器具有相同时间常数 \(\tau>0\)，实际输入经过固定执行映射，满足 \(|u_{i,j}|\le U\)。定义实际执行量

\[
f_j=(u_{1,j}+u_{2,j})/2,\quad
g_j=(u_{1,j}-u_{2,j})/2,\quad
a_{-,j}=(a_{1,j}-a_{2,j})/2.
\]

这里的 \(f_j,g_j\) 是限幅后的量，不是尚未实现的动作需求。恒有

\[
|f_j|+|g_j|=\max(|u_{1,j}|,|u_{2,j}|)\le U. \tag{1}
\]

若输入在每拍保持，连续一阶执行器的精确离散模型为

\[
a_{-,j+1}=\alpha a_{-,j}+(1-\alpha)g_j,
\qquad \alpha=e^{-\Delta/\tau}\in(0,1). \tag{2}
\]

已有 MATLAB 轨迹使用 RK4，而非精确 ZOH 离散或前向 Euler。对子步 \(h\)、每拍 \(n=\Delta/h\) 个子步的该线性执行器，数值实现精确对应

\[
\alpha_{\rm RK}=
\left(1-\frac h\tau+\frac12\frac{h^2}{\tau^2}
-\frac16\frac{h^3}{\tau^3}+\frac1{24}\frac{h^4}{\tau^4}\right)^n. \tag{3}
\]

因此对既有记录核验数值恒等式时使用 \(\alpha_{\rm RK}\)；对物理模型陈述用 \(\alpha\)。两者不混用。对于未建模执行器差异、输入保持误差、测量误差等，使用第 4 节的残差模型，不直接继承精确等式。

以 \(\mathcal T\) 表示完整轨迹。本轮条件信息 \(\xi\) 包含段初可用状态/历史与固定宏命令。若条件不足以确定初始执行器状态，以下 \(a_{-,0}\) 的确定值须改为其条件分布及相应估计误差。未知未来扰动不作为部署策略输入。所有期望均相对于声明的同一条件分布；不能把条件均值、路径约束和最坏情况保证互换。

## 2. 命题 1：有限策略变化下不变的执行记忆与截止时刻容量

令

\[
\beta_j=(1-\alpha)\alpha^{K-1-j},\qquad
\sum_{j=0}^{K-1}\beta_j=1-\alpha^K.
\]

**命题。** 对任意满足 (1)–(2) 的实际输入序列，无论它由哪个反馈策略生成、是否经过饱和、策略参数变化是否足够小，都有

\[
d(\mathcal T):=a_{-,K}-\alpha^Ka_{-,0}
=\sum_{j=0}^{K-1}\beta_jg_j, \tag{4}
\]

\[
|d(\mathcal T)|\le B(\mathcal T):=
\sum_{j=0}^{K-1}\beta_j\bigl(U-|f_j|\bigr). \tag{5}
\]

固定整个实际共模序列 \(f_0,\ldots,f_{K-1}\) 且 \(|f_j|\le U\) 后，允许自由选择满足 (1) 的实际输入序列时，可达的 \(d\) 恰为区间 \([-B,B]\)。这里的充分性是**存在一条输入序列**；不是当前神经反馈、当前策略参数邻域、当前采样数据或同一任务约束下的可实现性结论。

**证明。** 递推展开 (2) 得

\[
a_{-,K}=\alpha^Ka_{-,0}
+(1-\alpha)\sum_{j=0}^{K-1}\alpha^{K-1-j}g_j,
\]

即 (4)。因为每个 \(\beta_j>0\)，由 (1) 有

\[
\left|\sum_j\beta_jg_j\right|
\le\sum_j\beta_j|g_j|
\le\sum_j\beta_j(U-|f_j|)=B.
\]

反过来，若 \(B>0\)，对任意 \(d_\star\in[-B,B]\) 取

\[
g_j=\frac{d_\star}{B}(U-|f_j|).
\]

则 \(|f_j|+|g_j|\le U\)，且 \(\sum_j\beta_jg_j=d_\star\)。由 \(u_{1,j}=f_j+g_j\)、\(u_{2,j}=f_j-g_j\) 得合法实际输入。若 \(B=0\)，每个非负加权余量为零，只能取 \(g_j=0\)，可达集合为 \(\{0\}\)。证毕。

**机制含义。** 截止时刻越靠近某拍，该拍输入对末态执行记忆的权重越大：\(\beta_{j+1}=\beta_j/\alpha\)。所以同样总量的前进输入占用，发生在段末可能比发生在段初更严重地挤占接续恢复能力。不能只用不带时间权的平均输入或段末姿态来替代 (4)–(5)。MPC 若要求当前段末留下特定的 \(a_-\)，就必须同时提出共模占用的时序安排，或接受减弱/推迟该要求。

**随机反馈的合法推论。** 在固定 \(\xi\) 下定义

\[
\mu_{g,j}^{\theta}=\mathbb E_\theta[g_j\mid\xi],\quad
m_j^\theta=\mathbb E_\theta[|f_j|\mid\xi],\quad
d_\theta=\mathbb E_\theta[a_{-,K}-\alpha^Ka_{-,0}\mid\xi].
\]

则

\[
d_\theta=\sum_j\beta_j\mu_{g,j}^\theta,\qquad
\mathbb E_\theta[|d(\mathcal T)|\mid\xi]
\le\sum_j\beta_j(U-m_j^\theta), \tag{6}
\]

从而 \(|d_\theta|\le\sum_j\beta_j(U-m_j^\theta)\)。这些式子对有限策略变化仍成立，不是 Taylor 近似。\(m_j\) 必须估计 \(\mathbb E|f_j|\)，不能换成 \(|\mathbb E f_j|\)：例如 \(U=1\)、\(f_j=\pm1\) 等概率时，共模均值为零，但实际已占满输入，不能因此声称还有单位差模余量。

对于任意两个策略，若段初分布、执行器系数和命令条件相同，还得到有限变化恒等式

\[
d_{\theta'}-d_\theta
=\sum_j\beta_j
\left(\mu_{g,j}^{\theta'}-\mu_{g,j}^{\theta}\right). \tag{7}
\]

模型宣称的能力变化须由同一组实际输入矩变化解释。仅改变模型的 \(d\) 头、而没有对应的真实输入矩变化，不满足 (7)。然而 (7) 不说明一个给定的输入矩变化一定能被选定反馈策略类实现。

## 3. 恢复工作量是整条轨迹的联合特征

对已有机械对象

\[
\dot q=r,\qquad
\dot r=a_- -k\sin q-d_r r+w,
\]

取物理能量 \(E(q,r)=\tfrac12r^2+k(1-\cos q)\)，而不是把某个已知控制律的增益预先写入储能函数。沿真实连续轨迹有

\[
\dot E=r a_--d_r r^2+rw. \tag{8}
\]

在有限段 \([0,T]\) 上令

\[
L_r=\int_0^T e^{-t/\tau}r(t)\,dt,\qquad
K_r(s)=\frac1\tau\int_s^T e^{-(t-s)/\tau}r(t)\,dt.
\]

将执行器卷积代入 (8)，并在可积条件下交换积分，得到

\[
E(0)-E(T)-d_r\int_0^T r^2dt+\int_0^T rw\,dt
=-a_-(0)L_r-\int_0^T g(s)K_r(s)\,ds. \tag{9}
\]

由 (1) 进一步有

\[
E(0)-E(T)-d_r\int_0^T r^2dt+\int_0^T rw\,dt
+a_-(0)L_r
\le\int_0^T |K_r(s)|(U-|f(s)|)\,ds. \tag{10}
\]

式 (9) 的推导没有忽略执行器滞后，也没有把期望动作当作真实动作。\(K_r\) 依赖同一条未来轨迹，不能提前当成已知外部信号；\(\mathbb E[gK_r]\) 一般不等于 \(\mathbb E[g]\mathbb E[K_r]\)。因此恢复 work 应作为完整轨迹的联合统计或联合分布输出，而非几个独立均值头的乘积。未知 \(w\) 的积分需观测、已知界或明确估计误差；不能靠训练时特权扰动值给部署 MPC 提供未来真值。

这里 (8)–(10) 是连续物理关系。既有稀疏记录采用 RK4 时，要核对积分/插值残差；本轮不以离散近似精确等于连续工作量来声称能量证书。(10) 是沿实际轨迹成立的必要关系，既不保证任意目标轨迹可实现，也不证明全局姿态恢复。该周期势能不是对所有实数 \(q\) 径向无界的函数。

## 4. 命题 2：有限更新后的末态预测误差与可判别容量

将实际对象放宽为

\[
a_{-,j+1}=\alpha a_{-,j}+(1-\alpha)g_j+e_j,
\qquad |\mathbb E[e_j\mid\xi]|\le\varepsilon_{e,j}. \tag{11}
\]

候选策略 \(\theta'\) 可以是有限更新后的任意策略。设候选模型用

\[
\widehat a_K=\widehat\alpha^K\widehat\mu_0
+\sum_j\widehat\beta_j\widehat\mu_{g,j},\qquad
\widehat\beta_j=(1-\widehat\alpha)\widehat\alpha^{K-1-j},
\quad 0<\widehat\alpha<1, \tag{12}
\]

并有与该候选策略及同一 \(\xi\) 对应的估计误差条件

\[
|\widehat\mu_0-\mathbb E_{\theta'}a_{-,0}|\le\varepsilon_0,
\quad
|\widehat\mu_{g,j}-\mathbb E_{\theta'}g_j|\le\varepsilon_{g,j}.
\]

另外 \(|\mathbb E a_{-,0}|\le A_0\)，由实际限幅有 \(|\mathbb E g_j|\le U\)。则

\[
\boxed{
|\widehat a_K-\mathbb E_{\theta'}a_{-,K}|
\le
\widehat\alpha^K\varepsilon_0
+\sum_j\widehat\beta_j\varepsilon_{g,j}
+|\widehat\alpha^K-\alpha^K|A_0
+U\sum_j|\widehat\beta_j-\beta_j|
+\sum_j\alpha^{K-1-j}\varepsilon_{e,j}.
} \tag{13}
\]

**证明。** 递推展开 (11)，取条件期望，得到

\[
\mathbb E a_{-,K}=\alpha^K\mathbb E a_{-,0}
+\sum_j\beta_j\mathbb E g_j
+\sum_j\alpha^{K-1-j}\mathbb E e_j.
\]

从 (12) 减去该式，在初始项与每个输入项中分别加减 \(\widehat\alpha^K\mathbb E a_{-,0}\)、\(\widehat\beta_j\mathbb E g_j\)，再用三角不等式及所列误差条件，即得 (13)。证毕。

该界无需策略参数步长小，也不对未知真实对象求策略导数；它的前提是候选策略输入矩的误差条件实际成立。若 \(\alpha\) 精确已知、初态精确、无动力学残差，则只剩 \(\sum_j\beta_j\varepsilon_{g,j}\)。统一 \(\varepsilon_g\) 时等于 \((1-\alpha^K)\varepsilon_g\)，而不是随段长无条件线性膨胀。段末输入矩的误差得到更大权重，直接决定应在哪里改善数据或收紧模型发布余量。

若候选模型允许额外终端头偏离 (12) 不超过 \(\varepsilon_{\rm head}\)，右端必须再加该项。若报告“新旧能力增量”的误差，要为新、旧两组相应误差共同计账，不能只报候选组拟合误差。

**需求筛选条件。** 对精确 \(\beta\)，若 \(|\widehat m_j-m_j|\le\varepsilon_{m,j}\)，定义

\[
\widehat B=\sum_j\beta_j(U-\widehat m_j),\quad
\varepsilon_B=\sum_j\beta_j\varepsilon_{m,j}.
\]

则 \(B\in[\widehat B-\varepsilon_B,\widehat B+\varepsilon_B]\)。因此

- 若 \(|d_\star|>\widehat B+\varepsilon_B\)，该共模占用条件下的需求违反物理容量必要条件，必须改变需求、共模分配或截止时刻。
- 若 \(|d_\star|\le\widehat B-\varepsilon_B\)，需求处在有余量的物理外界内。只有**固定确定性共模序列、同一精确执行器模型、自由输入选择**时，命题 1 的构造才给输入序列存在性；随机均值和神经反馈情况下仍须策略实现证据。
- 其余为误差范围内未判定，不通过增加任意乐观预测将其写为可实现。

若 \(\beta\) 也不确定，估计误差采用 \(\sum_j\widehat\beta_j\varepsilon_{m,j}\)，另加系数误差 \(U\sum_j|\widehat\beta_j-\beta_j|\)。从终端状态换算 \(d_\star\) 时，还需计入一致的初始执行器记忆误差。式 (13) 与上述筛选都只是声明误差条件下的结论；同批经验误差和 ESS 不能自动充当这些误差上界。

## 5. 一批 PPO 数据怎样合法估计候选能力

### 5.1 完整轨迹换测度

策略采样潜动作 \(z_j\sim p_\theta(\cdot\mid h_j,c_j)\)，实际动作由固定、与 \(\theta\) 无直接依赖的执行映射 \(u_j=\psi(z_j)\) 得到。可有固定 tanh 或限幅；密度在潜动作上计算，不能把限幅后的执行动作硬套原高斯密度。历史 \(h_j\) 是部署时实际可用历史；若网络有参数相关内部记忆，评估候选概率时必须按候选参数在整段历史上重算，或使用等价的正确总导数，不能把旧网络隐藏状态当作固定的候选状态。

固定段初分布、环境核、命令条件/生成规则。假设候选与旧策略有共同支持，环境和执行映射没有未计入的直接参数依赖，并满足所需可积条件。旧策略为 \(\theta_k\)，轨迹比率

\[
R_\theta(\mathcal T)=
\prod_{j=0}^{K-1}
\frac{p_\theta(z_j\mid h_j,c_j)}
{p_{\theta_k}(z_j\mid h_j,c_j)}.
\]

以 \(\mathcal T\) 表示轨迹，避免与执行器时间常数 \(\tau\) 混淆。对不显式依赖 \(\theta\) 的完整轨迹特征 \(b(\mathcal T)\)，有总体恒等式

\[
\mathbb E_\theta b
=\mathbb E_{\theta_k}[R_\theta b],\qquad
\nabla_\theta\mathbb E_\theta b
=\mathbb E_{\theta_k}[R_\theta bS_\theta^\top],\quad
S_\theta=\sum_j\nabla_\theta\log p_\theta(z_j\mid h_j,c_j). \tag{14}
\]

证明是轨迹密度中相同初态/环境/执行核相消，以及在可微可积条件下交换积分与导数。若 \(b\) 或执行映射还有显式参数依赖，需加入相应直接导数或重新定义完整概率模型；不能只保留 score 项。

未归一化经验 IS 对**固定候选参数、合适独立样本条件下**的总体量无偏，不表示优化同一批数据后选出的候选无选择偏差。长段 ratio 可有极高方差。PPO 的单步 ratio clipping 只用于任务 surrogate；对 (14) 的完整段 ratio 另作 clipping 将改变估计目标，需要单独处理其偏差。

### 5.2 SNIS 的精确样本导数

对保存潜动作与旧 log probability 的固定批次 \(\{\mathcal T_i\}_{i=1}^N\)，定义

\[
w_i(\theta)=\frac{R_i(\theta)}{\sum_\ell R_\ell(\theta)},\qquad
\widehat\mu_\theta=\sum_iw_i(\theta)b_i.
\]

若分母非零，且 \(b_i\) 在该更新中停止梯度，则对这个**有限样本目标**有精确导数

\[
\boxed{
\nabla_\theta\widehat\mu_\theta
=\sum_iw_i(b_i-\widehat\mu_\theta)S_i^\top.
} \tag{15}
\]

因为 \(\nabla R_i=R_iS_i\)，商法则给 \(\nabla w_i=w_i(S_i-\sum_\ell w_\ell S_\ell)\)，代入即可。SNIS 总体估计通常存在有限样本偏差；式 (15) 精确不等于其是总体梯度的无偏估计。权重集中、ESS 下降或新策略与旧策略支持不一致都不能靠这条代数公式消除。

**不同段初条件下的配对。** 主算法采用固定非负核 \(\kappa_{\ell i}\)，故每组权重为 \(w_{\ell i}=\kappa_{\ell i}R_i/\sum_s\kappa_{\ell s}R_s\)。固定核时商法则仍然成立。因为各 \((\xi_i,c_i)\) 不同，应对各样本的模型残差加权：

\[
e_i=q_\phi(\xi_i,c_i)-b_i,\qquad
C_\ell=\sum_iw_{\ell i}e_i,\qquad
\nabla_\theta C_\ell
=\sum_iw_{\ell i}(e_i-C_\ell)S_i^T. \tag{15b}
\]

固定 \(\phi\) 时 \(e_i\) 对 \(\theta\) 无直接依赖，但它仍通过权重参与导数；不能把 \(\sum_iw_{\ell i}q_\phi(\xi_i,c_i)\) 的权重导数删去。只有模型查询确实相同且固定时，\(C=q_\phi-\widehat\mu_\theta\) 才可约成 \(\nabla C=-\nabla\widehat\mu\)。局部核平均也不自动等于锚点逐点条件期望；用于命题 2 时需要覆盖/平滑性误差支持。

令容差变量 \(s_\ell\) 在 actor 步固定，\(D_\ell=C_\ell-s_\ell\)。一致性增广拉格朗日项的 actor 梯度为

\[
\sum_\ell(\nabla_\theta C_\ell)^T
\operatorname{stopgrad}(\lambda_\ell+\rho D_\ell). \tag{16}
\]

若预测损失为 \(L_{\rm pred}=\sum_{\ell,i}w_{\ell i}\ell_i\)，其中 \(\ell_i=\|q_\phi(\xi_i,c_i)-b_i\|_W^2\)，则同一联合目标还需要保留

\[
\nabla_\theta L_{\rm pred}
=\sum_{\ell,i}w_{\ell i}
\left(\ell_i-\sum_s w_{\ell s}\ell_s\right)S_i. \tag{16b}
\]

每次 actor 反向前，按当前 \(\theta,\phi\) 重算权重、\(C_\ell,D_\ell\)，随后仅在该次微分中停止 \(\lambda_\ell+\rho D_\ell\) 的梯度；不把旧残差系数固定多个 epoch 后仍称原增广目标的精确梯度。模型、当前乘子及 \(s\) 固定；保留 \(R_i\) 和候选策略完整历史概率的梯度。更新模型时反过来固定 actor 权重。任务 PPO ratio、clip、advantage/critic 仍按其明确任务目标定义，不把 (15) 伪称为 PPO 原算法已有的可行保证。

若容差集合为 \(\|s_\ell\|_W\le\varepsilon_\ell\)、增广项使用欧氏二次范数，则正确的 \(s\) 步为在欧氏距离下把 \(C_\ell+\lambda_\ell/\rho\) 投影到该容差椭球；一般 \(W\) 下不能未经推导改成径向缩放。随后 \(\lambda_\ell\leftarrow\lambda_\ell+\rho(C_\ell-s_\ell)\)。

**时间信用不能误写。** 对终端能力 \(d=\sum_j\beta_jg_j\)，\(\beta_j\) 确实给输入特征提供截止时刻权重；但第 \(t\) 拍策略影响后续状态及所有后续输入，不只影响 \(g_t\)。总体的因果 score 简化形式为

\[
\nabla_\theta\mathbb E_\theta d
=\mathbb E_\theta\left[
\sum_{t=0}^{K-1}\nabla_\theta\log p_\theta(z_t\mid h_t,c_t)
\sum_{j=t}^{K-1}\beta_jg_j\right], \tag{17}
\]

在相应因果、可积和无直接参数依赖条件下成立。不能把它缩成只含 \(\beta_tg_t\) 的即时信用。SNIS 的有限样本精确实现应使用 (15)；未经另行推导，不能宣称 reward-to-go 删除项在自归一化有限样本下仍给完全相同的导数。

## 6. 哪些约束会冗余，哪里仍可能改变决策

若所有 \(b_i\) 都是同一物理对象生成的真实轨迹，而且同一个非负权重 \(w_i\) 同时用于输入矩、终端执行器与资源统计，则 (4) 的线性关系在样本重加权后自动成立，(5) 的凸必要界也自动成立。此时再施加相同关系不会凭空产生新的 actor 可行性约束。

因此不能把“加入物理等式”直接称为新的 RL 反馈学习机制。其本轮可检查用途是：

1. MPC 提出的能力需求必须同时描述响应、共模占用时序和末端执行记忆，不能独立要求多个相互挤占的理想指标。
2. 模型的候选能力更新须被同一个候选策略的整段统计支持，防止优化器只把模型写得乐观；有限数据下还有第 4 节误差条件。
3. 规划需求经 (5) 与误差余量筛选或重新分配后，通过 (15)–(16) 改变 actor 面对的真实反馈目标。必须在见证中具体展示这个需求变化，不能只展示一个模型等式残差下降。

如果普通共同训练已经使用完全相同的轨迹表示、物理关系、统计估计和完整目标，采用通用约束优化求同一问题可以得到相同解。这不构成反例，也不能靠改求解器名称制造优势。候选增量在具体表示及可估关系如何组织有限数据学习；它相对已有方法是否具备研究创新及效率收益仍需后续核验。

模型固定旧策略时的校准与候选新策略能力拟合也必须分开。若一个头继续用未加权旧策略目标拟合，同时又被要求代表新策略统计，两种训练信号可能矛盾。可采用明确的旧策略校准阶段与候选策略加权阶段，或分别命名两类变量；不能把辨识误差修正强迫解释成实际能力变化。

## 7. 发布与完整滚动闭环的边界

以下边界是本推导本身要求的适用范围，不是额外停止用户工作：

- **当前历史数据语义。** 若已有轨迹没有随机潜动作、旧策略密度和必要历史，就不能据此计算合法的 PPO/IS 更新见证。本轮可用它验证 (4)–(5) 及规划需求差异，不伪造 score、采样噪声或新的策略性能。
- **同批拟合不等于候选验证。** 自适应选择 \(\theta,\phi\) 后同批一致性残差很小，不能自行证明 (13) 所需误差条件。误差需由独立校准、明确模型假设或其他有效估计支持；ESS 是诊断而非置信保证。
- **随机与确定部署。** (6)、(13) 的期望与 (14) 的换测度针对声明的随机执行策略。把高斯策略换成均值动作会改变轨迹分布，确定策略通常也不与连续随机策略绝对连续。除非另有耦合/误差论证或对应数据，不把随机策略统计发布为确定性部署保证。
- **新 MPC 命令分布。** 本轮换测度固定命令条件或相同命令生成规则。若模型更新后 MPC 改了命令，命令分布及访问状态也随之改变，旧动作 ratio 不能直接估计新 MPC 加新策略的完整任务回报。冻结候选池得到的规划敏感信号可以作为本轮需求，但不是已证明的完整闭环性能提升。
- **有限候选池。** 若规划价值是固定候选池上的最小值，唯一赢家且各候选分支可微时，可对赢家分支求局部导数；平局需明确选支/广义导数。候选池的离散最小值不是 CEM 的 KKT 最优性条件，也不是连续命令域全局最优。
- **范围之外。** 均值误差界不推出路径安全、递归可行、任意切换稳定或 PPO 收敛。不同执行器时间常数、接触、观测不足和真实机器人部署需要重新定义关系与误差。

本轮已完成的是明确对象下的有限策略变化恒等式、截止时刻容量的必要/有限意义充分性、带时序权的候选模型误差界，以及可逐行实现的 SNIS 导数。它们足以审查一次联合更新是否自洽；尚不足以宣布学习能力、完整任务收益或论文创新已经成立。
