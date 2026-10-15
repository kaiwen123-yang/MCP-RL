# MR135 · Model-Based Diffusion Sampling for Predictive Control in Offline Decision Making

覆盖：ICML2026/PMLR306标识附件23/23页全部文字通读，含参考文献和附录A—L；p.7、9、22、23原页核对负例、理论边界与实机流程。未独立复现/全符号核验。

机制：分别训练state-action扩散planner与给定action的state扩散dynamics，采样时交替去噪，最后用reward/cost ranker择优。相比只在最后重评分，dynamics会在生成过程改变下一轮planner输入。不是显式硬投影；全部模块都是学得近似，预算过滤无可行候选时回退最小预测成本（p.3—5）。

关键作者负例：planner用medium-expert、dynamics用更广但缺高速专家的medium-replay时，Hopper得分降至70.3，低于两模块都用replay的98.2；作者解释是dynamics把planner过度修正回自身数据分布（p.9 Table10）。这是可继续研究的数据兼容性限制，不能仅因模块化模型已提出而排除。新动力学下ranker会放大高回报偏好导致的分布偏移，100条play-data适应后恢复部分性能（p.6—7）。

理论：附录K通过分布KL约束得到product-of-experts形式，再以score相加和算子分裂解释交替采样；p.23明确**不提供正式理论保证**。不能把学得dynamics一致性改述成已保证所有生成轨迹真实可行或满足CMDP约束。

比较与成本：作者补了同去噪工作量对照，planner200步约.575s、MPDiffuser100交替步约.583s，而后者平均回报更好，故增益不能只归于算力加倍；但HalfCheetah不一定更好（p.8—9）。D4RL ranker64候选、DSRL16候选，和实机单候选不是同一计算模式。Go2机载采用单候选、4动作chunk、异步、热启动7步；速度来自关节状态训练的MLP估计，不是独立外测速度（p.8、23）。

作者明确展望：online探索与适应、复杂视觉latent、多环境及跨域数据（p.9），物理结构与组合扩散结合（p.15）。

可承接推进：使模块在异质数据/动力学变化后仍相容，而不仅把数据量增大；建立何时允许dynamics纠正planner、何时需要新数据或保留行为模式的可检验机制。先固定ranker和采样预算，分离错模型、支持不重叠与目标条件化偏差。对照须含同数据训练、混合数据直接重训、仅ranker适应和同墙钟planner，证明减少真实失败/提升收益而非只减预测误差。
