# W59→W61 论文来源与方法阅读台账（2026-11-28）

## 口径

本台账合并本轮材料/控制、步态、学习三条审查与根审查的 APT-RL 记录，核心表共 **16 篇去重论文：15 篇取得正文并阅读方法及相关实验/假设，1 篇仅官方摘要**。E线实施设计另补3篇基础方法的指定正文范围和1篇仅引言/动机，见末尾附表，不计为近期新增论文。HPO/MGPO 的名称变化只计一篇；Residual MPC 的预印本与正式出版只计一篇。这里包含本轮复核过的旧文献，**不称为“新增16篇”**，也不把 2026-11-26 研究地图既有的89条重新计作新阅读。

“正文方法阅读”表示下文列明的范围，不表示每篇及全部补充材料逐字读完。阅读由对应团队审查者完成，汇编者综合三份 evidence，并复读 APT-RL 本地 Methods。未重新运行任何论文代码、训练或实机实验；静态源码阅读另列。下文“本项目差异”是候选研究空间与待验证设计，不是已完成的新颖性证明。

排序先列 Nature/Science 系和已核期刊，再列会议及预印本，便于按证据强弱阅读。**本轮未取得并核对2025中科院分区表，不虚构一区/二区/Top标签。** HANO 按已读 arXiv 版本登记；HPO/MGPO 未核正式出版状态时仅写预印本。每篇方法与局限只作紧凑记录，详细审查在三份 review 中。

## 正文方法阅读：15 篇

### 01. Mastering diverse control tasks through world models

**身份与来源：** Nature，2025；本轮依据正式版方法阅读。[主来源](https://www.nature.com/articles/s41586-025-08744-2)。

**方法：** RSSM 的循环状态与离散随机潜变量学习观测、奖励和 continuation；配合双向 stop-gradient KL、free bits、symlog/two-hot、分布 critic 与回报尺度处理。正式版连续/离散 actor 均使用 REINFORCE。

**假设与限制：** 想象回报仍受世界模型误差影响；采用 score gradient 不会消除预测偏差。广泛任务结果不能替代本项目材料接触与实机证据。

**本项目应吸收/区分：** 将模型学习与策略求导路线分开：保留真实策略梯度对照，对材料记忆/接触事件的预测单独检验，再研究何时启用局部物理梯度。

**实际阅读范围：** 主文 world model/actor/critic、消融，以及 Methods 的分布表示、优化器、replay。PDF 已取得。（审查来源：learning）

**实现核查：** 本轮核方法与实验；没有重新运行官方代码。

### 02. Agile perceptive multiskill locomotion for quadrupedal robots in the wild

**身份与来源：** Science Robotics，2026；DOI 10.1126/scirobotics.adz7397。[主来源](https://skillquadsr.github.io/)；[作者论文](https://arxiv.org/abs/2607.13579)。

**方法：** 2D SRB Bézier/动量周期轨道生成约18万条轨迹；TVAE 学共享16维 latent，trot/bound 两个独立 torque decoder。PPO 输出16 latent+12 auxiliary+1 gait logit；2Hz 选 decoder、100Hz 动作；前馈 torque+PD，depth/LiDAR teacher-student 蒸馏。

**假设与限制：** 离线2D平地两种步态先验；后续3D泛化靠学习与感知。本文只提供两种 decoder，不直接证明软材料参数适应、材料历史利用或我们约束接口下的稳定性。

**本项目应吸收/区分：** 借鉴快速轨道数据、共享表示/分模式输出和前馈控制；保持材料状态、接触拓扑、受约束 MPC/WBC 各自研究问题，不把全部六线收敛成复现该网络。

**实际阅读范围：** 已下载 apt_rl.pdf/txt；本地文本约 1280–1545 行，Materials and Methods 的轨迹优化、TVAE、RL 动作、感知蒸馏及 Discussion 限制。并非声称逐页精读全部补充材料。（审查来源：root；汇编者复读本地 Methods）

**实现核查：** 本轮 Methods 阅读；未声称复现或核过完整官方训练代码。

### 03. Fully data-driven inverse characterization of heterogeneous materials with hyper-network neural ODEs

**身份与来源：** npj Computational Materials 12,165，2026；DOI 10.1038/s41524-026-02027-8。[主来源](https://www.nature.com/articles/s41524-026-02027-8)。

**方法：** 材料坐标经 hypernetwork 生成局部 NODE 本构权重，用结构保证能量相关条件；通过全场位移和边界牵引反演连续异质超弹性响应，结合场插值、平衡和牵引残差。

**假设与限制：** 材料为超弹性场；需要远多于本项目整足合力的全场观测。NODE 积分深度不是实际黏弹松弛时间。

**本项目应吸收/区分：** 条件化本构族可借鉴；需新解决稀疏接触观测下材料身份 c 与历史状态 z 的分离、可辨识参数范围和高频状态/低频参数适应。

**实际阅读范围：** 官方全文 PDF；Methods 方程9–19、模型/场插值/初始化与损失、合成和打印试件实验、Discussion；静态核官方 node_inv 部分核心。（审查来源：model_control）

**实现核查：** https://github.com/tajtac/node_inv/blob/85d551a4dd8dc2777fdb18fa1aa1a8886a6ee132/utils_hyperelasticity.py；读 common/sample 权重、NODE/RK、应力计算，未复现。

### 04. Robot Tracking Control With Natural Task-Space Decoupling

**身份与来源：** IEEE Transactions on Robotics 42:734–749，2026；DOI 10.1109/TRO.2025.3647777。[主来源](https://elib.dlr.de/221903/1/main_final_with_CCBY.pdf)。

**方法：** 保留自然任务惯量的对角块并处理跨任务耦合；匹配惯量变化与 Coriolis 关系，在相应条件下得到局部跟踪与被动性性质。

**假设与限制：** 全驱动、局部可逆任务映射等条件；上身力矩控制硬件不同于欠驱动浮动基和软接触切换。力观测误差验证也有边界。

**本项目应吸收/区分：** 现有 WBC 已含 M/h/J，研究对象应是可实现子空间内的任务惯量、软法向耗散及模式内混合内模；需要重新推导，不能直接继承论文定理。

**实际阅读范围：** DLR 官方全文；II–IV、VI–VIII，任务惯量/Coriolis 一致性、稳定/被动性假设、仿真及 Rollin Justin 实验。（审查来源：model_control）

**实现核查：** 本轮正文核方法，未执行论文实现。

### 05. Residual MPC: Blending Reinforcement Learning With GPU-Parallelized Model Predictive Control

**身份与来源：** IEEE Transactions on Robotics 42:3469–3488，2026；DOI 10.1109/TRO.2026.3721669。方法阅读为作者2025v1预印本。[主来源](https://arxiv.org/html/2510.12717v1)；[正式元数据](https://ieeexplore.ieee.org/abstract/document/11644391/media)。

**方法：** 12节点浮动基 kinodynamic MPC；一次SQP、25次ADMM，KKT分解复用回代；CusADi+cuDSS 批处理；RNEA前馈+PD，关节空间 residual 在力矩层混合。

**假设与限制：** 省略部分关节动力学/力矩及自碰撞约束。千环境100Hz是仿真控制频率，实际单步约0.5s；硬件100Hz与训练吞吐要区分。

**本项目应吸收/区分：** GPU稀疏求解和前馈是强近邻；研究材料/执行器/接触约束下可行残差、尾延迟和可信模型导数，不能仅把MPC加PPO称独有创新。

**实际阅读范围：** 作者 PDF/HTML III–V、算法1–2、表I–II、硬件与限制；正式2026出版信息核实，但不声称读完整正式20页版。（审查来源：model_control）

**实现核查：** 静态读 se-hwan/cusadi commit bffea25849a901277a8f8dc4db16748e629f9053 的 README、src/CusadiFunction.py、src/generateCUDACode.py；未见PyTorch backward；不是已核完整Residual MPC开源训练栈。

### 06. Exciting families of passive gaits in an elastic quadruped via natural motion manifold control

**身份与来源：** IJRR，online 2025；45(2):233–258，2026；DOI 10.1177/02783649251347305。[主来源](https://journals.sagepub.com/doi/10.1177/02783649251347305)；[全文](https://pure.tudelft.nl/ws/portalfiles/portal/249500525/calzolari-et-al-2025-exciting-families-of-passive-gaits-in-an-elastic-quadruped-via-natural-motion-manifold-control.pdf)。

**方法：** Poincaré 截面上的固定点曲线以能量参数化；横向反馈稳定与零空间中的PI能量注入分离，构造轨道族而非单一周期。

**假设与限制：** 保守轨道族、输入秩/可控性、局部Schur条件及小损耗近似；3kg BERT SEA 的矢状面跑步机 bounding 实验不同于Go2三维耗散软地。

**本项目应吸收/区分：** 构造材料条件的受控周期轨道族，验证局部返回映射和扰动衰减；不能把塑性接触任务直接叫已有证明的被动流形。

**实际阅读范围：** 官方 TU Delft PDF；§3保守轨道族、§4横向坐标/线性化/输入零空间/稳定/能量调节、§6硬件及限制。（审查来源：gait）

**实现核查：** 本轮方法/硬件阅读，未复现。

### 07. History-Aware Neural Operator: Robust Data-Driven Constitutive Modeling of Path-Dependent Materials

**身份与来源：** 按已读 arXiv:2506.10352v2（2025）登记；不未经核实附加期刊信息。[主来源](https://arxiv.org/html/2506.10352v2)。

**方法：** 以可观测应变/应力窗口和应变增量学习路径依赖响应；结合历史神经算子、递减teacher forcing和递增输入噪声训练自由滚动，RNN对照匹配输入/预算。

**假设与限制：** 所需历史长度取决于材料、载荷与时间跨度；没有可直接继承的普适热力学保证，也不能从单实验取一个通用历史步数。

**本项目应吸收/区分：** 可借鉴可辨识历史条件和滚动训练；保留PANN能量/耗散结构，并显式区分材料参数、接触状态与记忆。

**实际阅读范围：** 全文 PDF；§3结构/训练、§4比较、§5.4历史长度、§6限制。（审查来源：model_control）

**实现核查：** 本轮读取正文方法，未运行官方实现。

### 08. Variance-Reduced Model Predictive Path Integral via Quadratic Model Approximation

**身份与来源：** RSS，2026；DOI 10.15607/RSS.2026.XXII.108。[主来源](https://roboticsproceedings.org/rss22/p108.html)；[全文](https://arxiv.org/pdf/2602.03639)。

**方法：** 代价分为二次模型和残差；二次部分塑造Gaussian均值/协方差，用残差做重要性权重。比较精确/Gauss–Newton/BFGS/随机平滑曲率。

**假设与限制：** 局部曲率可偏，单Gaussian可能漏掉模式；接触示例曲率约1412ms、采样约706ms，不是四足实时证据。辅助样本和曲率代价必须计入。

**本项目应吸收/区分：** 保留接触拓扑分支，在分支内用便宜结构曲率提高少样本效率；以同墙钟和同候选预算分别对照MPPI/CEM。

**实际阅读范围：** 全文 PDF II–IV、算法1、二次提议/重要性修正、Hessian近似、静态/摆杆/接触实验与表II。（审查来源：model_control）

**实现核查：** 正文与算法细节阅读；未复现其控制器。

### 09. Simple Policy Optimization

**身份与来源：** ICML，2025；正式PMLR。[主来源](https://proceedings.mlr.press/v267/xie25m.html)。

**方法：** 将平顶clip换成有恢复方向的概率比二次目标：rho*A−|A|(rho−1)^2/(2*epsilon)。可以在不改采样、critic、动作的条件下改变更新内核。

**假设与限制：** 标量概率比目标性质不自动推广为任意网络、多epoch或接触机器人全局单调保证。

**本项目应吸收/区分：** 适合作为同权限强更新对照；真正新贡献须在事件信用、混合动作或可信模型梯度融合，并用匹配预算消融证明。直接使用SPO不算本项目创新。

**实际阅读范围：** §4–6、算法1、主要对照与深网络实验；正式PDF。（审查来源：learning）

**实现核查：** 本轮方法/实验阅读，未重新训练或运行官方实现。

### 10. Dream-MPC: Gradient-Based Model Predictive Control with Latent Imagination

**身份与来源：** ICML，2026；正式PMLR。[主来源](https://proceedings.mlr.press/v306/spieler26a.html)。

**方法：** 策略产生少量候选，在latent imagination中做梯度改进，重用上一步解，以Q ensemble分歧约束过度乐观。

**假设与限制：** 依赖较强policy prior与可用的预测梯度；模型调用下降不等比墙钟加速。正文Acrobot/4090约18.15ms对18.77ms，非普遍数量级提速。

**本项目应吸收/区分：** 冻结策略可作候选提议；把梯度优化限制在通过局部响应检查的模式/时域，保留无梯度同预算对照，不用多步优化掩盖模型漂移。

**实际阅读范围：** §4–5、算法1、消融与计时、Appendix A限制；正式PMLR PDF。（审查来源：learning）

**实现核查：** 本轮正文方法/计时核查，未复现。

### 11. Sampling-based Model Predictive Control Using Trust Regions

**身份与来源：** arXiv:2605.07801v2；当前arXiv元数据标注 Accepted CDC 2026。[主来源](https://arxiv.org/abs/2605.07801)；[全文](https://arxiv.org/pdf/2605.07801v2)。

**方法：** 用KL信赖域和熵下界优化候选分布，经双变量求解、密度校正与矩匹配更新；预计算采样点、随机旋转与缓存减少采样方差。

**假设与限制：** 有限样本和Gaussian投影为近似；时延和光滑度收益依任务。论文摆杆/卡车时延不是千赫控制证据。

**本项目应吸收/区分：** 是CEM/MPPI外的强分布更新对照；需要相同拓扑库、材料模型、全成本计时与本任务约束，不能只把精英更新重命名。

**实际阅读范围：** 全文 PDF II–VI；双变量eta/alpha、重要性权重、Gaussian投影、确定性Gaussian采样、buffer/warm-start、实验时延。（审查来源：model_control）

**实现核查：** 本轮正文算法阅读；未运行控制实验。

### 12. Policy Optimization in Hybrid Discrete-Continuous Action Spaces via Mixed Gradients

**身份与来源：** 按 arXiv:2605.14297（2026）预印本登记；HPO/MGPO名称合并一条，正式会议信息不作已核事实。[主来源](https://arxiv.org/abs/2605.14297)。

**方法：** 混合离散模式和每模式连续控制；梯度含pathwise项、离散score项及连续控制经未来状态改变离散策略概率的cross term。连续分支与离散/critic更新节奏不同。

**假设与限制：** 无偏结论依赖真实平滑转移和正则性；近似世界模型不能直接继承。库存/切换LQR并非足式实机验证。

**本项目应吸收/区分：** 适合接触模式、轨道参数和持续时间的联合信用；当前三个连续宏动作不体现主要优势。需要模式内梯度与跨模式score的可检验融合。

**实际阅读范围：** §2–5、梯度估计器伪代码、Appendix C.2实现及数值问题。（审查来源：learning）

**实现核查：** 正文实现节奏核查；未声称完整官方代码复现。

### 13. Learning optimal gaits and styles for terrain-adaptive quadruped locomotion

**身份与来源：** arXiv:2611.25958v1，2026-11-19；未核期刊/会议接收。[主来源](https://arxiv.org/abs/2611.25958)；[项目页](https://guillaume10.github.io/terrain-gaits)。

**方法：** 11个预设CPG耦合矩阵；33Hz的11维身体样式/振幅/频率动作，1kHz CPG+IK+PD；低层训练后冻结，再学步态选择；heightmap感知经DAgger。

**假设与限制：** 耦合矩阵是预设，命令相位不强制真实接触；几何地形为主。非成功率指标按成功样本计算，需注意失败代价口径。

**本项目应吸收/区分：** 实际接触与材料历史驱动的受控轨道/三维落足/过渡，比单纯多CPG选择更有研究内容；先有强同接口多步态对照。

**实际阅读范围：** 官方全文 PDF；rhythm/pattern formation、两阶段训练、heightmap、训练评估、限制和补充参数。（审查来源：gait）

**实现核查：** 本轮未核读官方仓库；不得称已经复现。

### 14. LoComposition: Terrain-Adaptive Energy-Efficient Quadruped Locomotion without Gait Priors

**身份与来源：** arXiv:2606.15896v2，修订2026-10-12；接收信息未核。[主来源](https://arxiv.org/abs/2606.15896)；[项目页](https://www.loukaskordos.de/publications/locomposition/)。

**方法：** 速度任务、CaT操作约束、逐步增加的机械绝对功惩罚、13×11高度图；不规定air-time/接触数/净空先验，50Hz关节位置偏移。

**假设与限制：** 测试速度中主要策略仍为trot；CaT不是运行时正式安全证书；跨机体有重新训练，硬件依赖映射/估计栈。

**本项目应吸收/区分：** 作为能耗/约束产生步态的整栈强对照；保留同速度任务、失败和实际能耗口径。不同底层整栈收益不能归因于单个新模块。

**实际阅读范围：** 官方全文 PDF；§3方法、§4匹配消融/结果、§5限制、硬件与感知范围。（审查来源：gait）

**实现核查：** 正文与消融阅读；未复现。

### 15. PROMO: Preference-conditioned Multi-Objective Reinforcement Learning for Quadrupedal Robots

**身份与来源：** arXiv:2610.01260v1，2026；TRO投稿不等于接收。[主来源](https://arxiv.org/abs/2610.01260)；[项目页](https://amrmousa.com/promo/)。

**方法：** 跟踪/稳定/效率三个critic head，固定步态prior与偏好分开；history encoder对齐privileged表示，actor不反传到辅助latent/速度估计，scalarized GAE/PPO及小KL多样性。

**假设与限制：** 保留步态/接触先验；报告评估偏好每条rollout固定，torque-effort proxy不等于电池能耗；没有动力学模型路径梯度。

**本项目应吸收/区分：** 材料风险与支撑质量可作可解释折中条件；需检验动态切换、任务速度和真实接触风险，防止将慢走换稳定称全面提升。

**实际阅读范围：** III方法的objective/prior分解、encoder stop-gradient、critic/GAE/PPO/diversity，IV实验比较；非声称所有附录逐字读完。（审查来源：gait）

**实现核查：** 本轮正文方法核查，未训练。


## 仅摘要核实：1 篇

### 16. Terramechanics-Aware Adaptive Quadrupedal Locomotion via Foot-Terrain Interaction Model

**身份与来源：** IEEE RA-L11(11):12552–12559，2026；DOI10.1109/LRA.2026.3732903；仅摘要层级。[主来源](https://ieeexplore.ieee.org/abstract/document/11685272/)。

**方法：** 摘要报告塑性屈服足地模型，涉及非线性沉陷、剪切饱和、承载损失；用接触历史估计摩擦/可变形性，并有IsaacSim、MPM Newton与Go2沙地验证。

**假设与限制：** 具体方程、估计器、网络/动作和世界记忆区分未核；不能依据摘要排除其已包含某功能。

**本项目应吸收/区分：** 已足以说明“材料感知+自适应步态”有强近邻；完整方法差异和新颖性权利要求须等可合法取得全文再核。

**实际阅读范围：** 仅官方IEEE元数据和摘要；未取得全文/官方代码，不能写已读方程、实验设计或实现细节。（审查来源：gait）

**实现核查：** 全文/代码未取得。

## 与 W61 六线的对应

| 工作线 | 直接方法参照 | 必须有的差异与强对照 |
|---|---|---|
| A：条件 PANN 与执行世界模型 | HyperNODE、HANO、DreamerV3 | 材料身份与历史分离；稀疏可观测辨识；结构化离散与自由滚动；同观测参数模型/黑箱/PANN |
| B：受控步态/轨道族 | terrain-gaits、NMM、APT-RL、LoComposition、PROMO | 从命令相位走向真实接触与材料历史；固定/切换轨道库、CPG和能耗整栈强对照 |
| C：拓扑与 GPU 稀疏 MPC | Residual MPC、VR-MPPI、TRMPC、Dream-MPC | 相同权限分离拓扑、模型、求解贡献；计全部墙钟/尾延迟；不以更多模型调用掩盖误差 |
| D：非线性受约束 WBC | Natural Task-Space Decoupling、Residual MPC、NMM | 现有M/h/J不是新增；模式内可实现惯量、软法向耗散与有界内模；重新推导适用条件 |
| E：事件/混合策略更新 | SPO、HPO/MGPO、DreamerV3、PROMO | 同接口同critic强基线；跨离散/连续信用；模型梯度偏差校准；不能把纯换SPO当创新 |
| F：联合算法与实机路线 | 上述跨模块方法 | 先单模块，再A+C、B+D，最后整体；共享状态与权限，分别评估有效样本效率和单机器人实时性 |

Terramechanics-Aware 作为必须继续取得全文的近邻记录，当前只能限制过宽的新颖性表述，不能据摘要作精确算法排除或优势结论。

## 来源与计数核验

合并来源为 evidence_model_control.json（6篇正文）、evidence_gait.json（4篇正文+1篇摘要）、evidence_learning.json（4篇正文），另加根审查 APT-RL（1篇正文）。总计15篇正文+1篇摘要。原始 evidence 中的标题/URL/阅读范围保存在同名 JSON 便于后续追踪。后续新增全文阅读应更新具体范围和日期，不把已记录论文重复计数。


## E线实施设计补充来源（与上述核心表分开计数）

为核实新更新器的已有理论来源，学习设计另定向阅读下列工作。核心16篇加此4条共20条记录：18篇取得正文并读指定方法范围、1篇仅官方摘要、1篇仅引言/动机；不是20篇新增或逐字全文阅读，也不与旧89条简单相加。

| 来源 | 实际范围 | 用途与边界 |
|---|---|---|
| [TRPO，ICML 2015](https://proceedings.mlr.press/v37/schulman15.pdf) | §2–4 surrogate、KL近似及实际信任域 | 自然梯度/信任域本身是既有理论 |
| [Q-Prop，ICLR 2017](https://arxiv.org/pdf/1611.02247) | §2–3、Eq.9–12、状态相关权重 | 控制变量抵消需正确补偿，不保证必然降方差 |
| [LAX/RELAX，ICLR 2018](https://arxiv.org/pdf/1711.00123) | §2、§3.1、§3.4，Eq.5/10 | 参数直接依赖、score/path补偿与辅助函数条件 |
| [CAPO，2026预印本](https://arxiv.org/html/2603.12596v1) | 仅引言和方法动机 | 标注已有优化方向研究；不作全文比较或理论依据 |

详细推导、假设和实施口径见[w61_learning_design.md](w61_learning_design.md)，没有运行这些论文代码。
