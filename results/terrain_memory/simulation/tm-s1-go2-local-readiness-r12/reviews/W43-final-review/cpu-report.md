# W43 CPU 最终结果审查：运行完成，起步与材料通行未成立

审查基点：`6a94c94c8e21e3b50fcca40ab4363165dee2987e`（2026-11-10 北京时间 20:52:55）。本报告仅分析既有文件，没有启动物理任务、访问服务器进程或修改控制器。GPU 保持原作业，不能把此前 GPU 的中途快照当本轮最终状态。

机器可读复算：[cpu-summary.json](cpu-summary.json)。发布核验及本次实际下载清单：[publication-audit.json](publication-audit.json)。

离线复算脚本：[analyze_cpu_raw.py](analyze_cpu_raw.py)。它接收仓库、原件目录和[公开 Release 元数据快照](release-assets-snapshot.json)，不联网、不启动物理、不读取 checkpoint 权重。原件目录沿脚本头部声明的短名保存；publication-audit 给出了每项对应的原始 source_path、Release 文件名、URL 和 SHA256。以下示例从本审查目录执行。

```bash
python analyze_cpu_raw.py \
  --repo /path/to/MCP-RL \
  --raw /path/to/w43-raw \
  --release-metadata ./release-assets-snapshot.json \
  --output /path/to/recomputed-review
```

`release-assets-snapshot.json` 是本次两份公开 GitHub Release API 响应的归一化快照，属于发布元数据，不是新增的物理原件。顶层以 job 名为键，保留 `id/url/assets`；每个 asset 保留 `name/size/state/digest/url`。这与 Git 仓库内的发布方声明分开，避免拿同一份声明自证远端上传。元数据来源为两个 `tm-s1-go2-local-readiness-r12-job-20261110T035658Z-w43-a-diagnostics-part01`、`tm-s1-go2-local-readiness-r12-job-20261110T035700Z-w43-b-diagnostics-part01` Release 的 GitHub API 响应，快照不代表未来资产状态不会变化。脚本会重新校验所读 52 项本地原件的 SHA256；参数化脚本已用本次同一原件离线复算，CPU JSON 对象完全一致，发布 JSON 除并行下载造成的清单顺序外完全一致。

## 1. 本次证据与上一次中途审查的区别

`1334f76` 只记录了 update54 运行中状态，公开四个完整回合均在第一次新更新前结束。本次两 CPU 已各完成 `+512 / +2`，并各自使用末次 update55 完成固定 E30。此前尚缺的训练 trajectory、planner calls、rollout 等现在已发布，不能沿用“这些原件尚未上传”的旧结论。

本子审查实际读取：

- 10 份训练 compact 和对应 resolved；2 份最终 E30 compact 和对应 resolved；两份 job、evaluation、collection、progress/stopped、learning、任务身份及诊断清单。
- 两流共 1024 条 Git transition 数值导出、1024 条 PANN/value 导出，按 `(stream_id, transition_serial)` 审核，未把共同祖先样本相加。
- 12 个完整 `trajectory.npz` 的全部 500 Hz 时序相关列、12 份完整 `planner-calls.jsonl`（共 28,850 次调用）、24 个 material events/failures 日志及 4 个完整 rollout NPZ。
- 上述 52 个下载资产共 313,922,243 字节，逐个重算 SHA256 并与 GitHub asset digest 相符。完整资产 URL、原 source_path、大小与哈希见 publication-audit。

本报告没有将未读取的全量 5 kHz physical、全部 prediction/feedback/accepted 矩阵或 checkpoint 权重宣称为已复算。技术审查中的选定矩阵另行列出范围。500 Hz endpoint 统计也不等于十个 0.2 ms 物理子步的全量证明。

## 2. 两条任务已按原样本目标结束

| 字段 | A | B |
|---|---:|---:|
| job | job-20261110T035658Z-w43-a | job-20261110T035700Z-w43-b |
| 父点 samples / updates | 13713 / 53 | 13713 / 53 |
| 本轮新增 samples / updates | 512 / 2 | 512 / 2 |
| 最终 samples / updates | 14225 / 55 | 14225 / 55 |
| 最终 tail / excluded | 0 / 145 | 0 / 145 |
| 最后训练 episode | 410 | 398 |
| `training_exit_code` | 0 | 0 |
| `training_stop_reason` | target_reached | target_reached |
| 训练 invocation 墙钟秒 | 21501.3512 | 21748.6405 |
| 最终固定评价 | 自身 update55 E30 | 自身 update55 E30 |
| E30 结果 | material_traversal_not_completed | material_traversal_not_completed |
| E60 | blocked_by_predecessor | blocked_by_predecessor |

来源：[A job](../../jobs/job-20261110T035658Z-w43-a/job.json)、[B job](../../jobs/job-20261110T035700Z-w43-b/job.json)，各自 `training/collection.json` 的 `collected_transitions`、`updates`、`unupdated_fragment`、`phase_new_transitions`、`phase_new_updates`；各自 `evaluation.json` 的 `checkpoint_updates=55`、`deterministic_policy_mean=true`、`frozen_normalization=true`、`ppo_updates=false`。

两者 `deadline_unix_s=null`、`wall_deadline_enabled=false`。10 个训练回合由 **8 个 60 秒 acquisition boundary + 2 个约 16 秒 training_checkpoint_complete** 组成。最后两个是达到本轮样本目标，不是 4 秒短截断，也不是后台墙钟强杀。10 个训练与 2 个评价 compact 的 `error` 均为 null；没有结果级技术中断、跌倒或材料域退出终止。E30 的任务不通过是有效负结果，不是进程异常。

所有训练数据的行为版本是每流 `update53:256`、`update54:256`；update55 的表现由最后固定 E30 提供。不能将最早四回合称为 update54 后测；本次也不再只有中途策略证据。

## 3. 最终 E30 仍未形成步态

下表直接来自两份真实 trajectory，出发段从约 2.002 秒开始，包含 14,000 个 500 Hz 样本。

| 复算量 | A update55 E30 | B update55 E30 |
|---|---:|---:|
| 物理时长 | 30.000 s | 30.000 s |
| 出发段净前向位移 | **−13.241529 mm** | **−13.217802 mm** |
| 出发段机身高度变化 | −9.330448 mm | −9.311880 mm |
| 出发段平均命令 vx | +0.221778523 m/s | +0.206270548 m/s |
| 出发段平均实际 vx | −0.000472923 m/s | −0.000472075 m/s |
| requested cap 平均值 | 0.223549261 m/s | 0.207837814 m/s |
| 全回合最大 base x | +0.0245458 mm | +0.0245458 mm |
| 最大足心 x | 0.192156780 m | 0.192156780 m |
| 四足接触同时为真比例 | 1.0 | 1.0 |
| 离地 / 落地次数 | 0 / 0 | 0 / 0 |
| 相对出发首帧目标角最大变化 | 0.056808949 rad | 0.056701183 rad |

来源：[A E30 结果及其 trajectory 指纹](../../job-20261110T035658Z-w43-a-eval-PANN_PPO_S5-s1-E30.json)、[B E30](../../job-20261110T035700Z-w43-b-eval-PANN_PPO_S5-s2-E30.json)，实际 Release 名称见 publication-audit。`time`、`qpos`、`qvel`、`command_velocity`、`requested_speed_cap`、`contact`、`q_target`、`feet` 均来自原始 NPZ；终点、最大 base x 和时长与 compact 一致。

四足不是接近零力却被布尔标记误判为接触。A 各足出发段最小法向力分别约 `33.977 / 34.837 / 36.882 / 36.875 N`，B 类似；足心 z 始终约 `0.021996 m`。因此当前是持续四足承载、缓慢下沉与后移，没有实际换支撑。

两评价出发段 `support_active` 样本数均为 0，表明执行记录中的 Stand 支撑开关未持续打开。原始 500 Hz `unclipped_pd_request` 与 `torque` 最大差约 `1.42e-14 Nm`，请求绝对值峰值 A `6.744541 Nm`、B `6.742868 Nm`。**这个记录频率上没有发生请求被力矩限幅的迹象；不能把它扩大为所有十个 0.2 ms 子步都绝不饱和。** 当前证据不支持继续把这两个 E30 的驻留直接归咎于扭矩饱和。

## 4. 训练回合也没有获得前进

| 回合 | 物理时长 | 出发段 Δx，mm | native 接受 / incumbent 复用 |
|---|---:|---:|---:|
| A406 | 60 s | −27.983358 | 2886 / 135 |
| A407 | 60 s | −27.966655 | 2884 / 137 |
| A408 | 60 s | −27.978053 | 2885 / 136 |
| A409 | 60 s | −28.000516 | 2888 / 133 |
| A410 | 16 s | −6.217506 | 685 / 135 |
| B394 | 60 s | −27.945102 | 2881 / 140 |
| B395 | 60 s | −27.978665 | 2885 / 136 |
| B396 | 60 s | −27.953771 | 2884 / 137 |
| B397 | 60 s | −27.965399 | 2884 / 137 |
| B398 | 16 s | −6.199253 | 683 / 137 |

这张表的位移、接触与规划调用来自下载原件。10 个训练回合出发段均四足接触比例 1.0、离地/落地 0/0；最大 base x 均只有 0.0245458 mm。两个短回合在 16 秒样本目标边界终止，其 planner 末调用时刻及数量与长回合不同，不人为补入未执行的终端调用。

因此，新增一次 PPO 更新后的采集仍停驻，最终第二次更新后的固定 E30 也停驻。问题已超出上次“尚未测到新策略”的证据限制。

## 5. λ 生命周期有效；局部成本下降没有兑现为任务进展

12 份原始 planner calls 共 **28,850 次**：训练 25,808，评价 3,042。其中：

- `deployment_status=accepted_native_descent`：27,213 次。
- `deployment_status=reused_current_native_scored_incumbent`：1,637 次。
- 对所有真正新 epoch，首个实际 attempt 都是 `lambda_used=1, rate_used=1`；发现的错误 reset 为 0。
- startup 第 12 call 用完五次尝试后停止，13–20 call 为零次新尝试，不继续累积阻尼。
- 本次 calls 的 return reason 只有 `policy_installed` 与 `native_search_stagnation`，不是数千次物理或技术失败。

最终 E30：A 共 1521 次 call，1385 accepted /136 reused；B 同为 1521，1383/138。出发段 A 1374 accepted /27 reused，B 1372/29。

出发段接受尝试的 native 成本下降中位数 A `6.2159e-5`、B `6.1604e-5`，代理预期下降中位数 A `2.6359e-4`、B `2.6137e-4`。这些是**同一当前问题内相对名义轨迹的改善**。A 的 `nominal_current_cost` 在不同物理时刻从约 0.00402 增至 0.04345，不能写成“整个闭环总代价持续下降”或据局部接受判任务稳定。

本轮事实是：旧高阻尼污染被修正、求解能做出非零动作并接受局部下降，但仍没有起步。确切的成本分项、接触候选与部署机制由技术报告的事件矩阵分析补充，不能凭本节直接认定唯一根因。

## 6. 学习数据确实增长，但材料记忆没有获得有效任务数据

每条流 transition ID 恰为 `13714..14225`，512 行且无重复/缺号；PANN/value 行的 ID 与其完全一致。所有 4 个 rollout 包均为 256 样本、658 维 obs/next_obs，全部数组有限。

独立对照发现：

- `raw_action`、`log_prob`、terminated/truncated、value/next_value 与 Git 行导出完全相符。
- reward 最大差仅 `1.6203e-9`，符合 rollout 的 float32 存储。
- obs/next_obs 最后 12 维恢复特征与导出的 current/next PANN 恢复值一致，全部为零。
- 1024 条 `failure=false`；这不是合法材料成功，而是没有真实失败终止。

| PPO 指标 | A update54 | A update55 | B update54 | B update55 |
|---|---:|---:|---:|---:|
| `loss` | 0.00657055 | 0.00570674 | 0.00744800 | 0.00169331 |
| `approx_kl` | 0.00908764 | 0.00103841 | 0.00270479 | 0.00660566 |
| `clip_fraction` | 0.15625 | 0.00390625 | 0.03125 | 0.08984375 |
| `behavior_logprob_max_error` | 9.54e-7 | 1.91e-6 | 4.77e-7 | 4.77e-7 |

来源：[A learning](../../jobs/job-20261110T035658Z-w43-a/training/learning.json)、[B learning](../../jobs/job-20261110T035700Z-w43-b/training/learning.json)。这些优化汇总是已发布 training log 选择记录，未重新运行 PPO 或读取权重以验证梯度。

采样 cap 均值：A 行为 update53/54 为 `0.222694/0.222838 m/s`，B 为 `0.211127/0.210400 m/s`。最终确定性 E30 请求 cap A/B 相差约 0.0157 m/s，运动结果却都后退约 13.2 mm。这表明当前 cap 权限下的学习变化没有改善起步。不能进一步推出 PPO 算法本身失效或奖励应立即重写。

## 7. 材料加载与任务资格均为零

12 个结果的 `material.world_cells/loaded_ever_cells/events` 均为 0，最大应变、最大应力为 0；24 个下载 material events/failures 日志解压后均为 0 字节。最大足心 x 只有 0.1922 m，远未进入原材料带 x `[0.8, 2.2]`。

因此材料组件与观察器已启用，不等于发生材料承载。`query_calls`、空材料记录文件和记录耗时不是加载证据。所有合法去程、返程和完整往返均为零；E30 不通过后 E60 按原协议阻断，未取得更长任务资格。

`material.integral_slip_sq_s` 非零也不能改变这个结论：当前记录代码的 slip mask 为受力足，未限定软材料，所以刚地足滑移可入账。PANN 当前/下一恢复特征全部零与没有受载世界网格相符；不能据此评判恢复记忆的因果收益。

## 8. 发布已经补齐指定诊断范围，历史上传错误应保留但不混算训练失败

| 范围 | A | B |
|---|---:|---:|
| diagnostics manifest 项数 | 391 | 388 |
| 实际 Release 资产数 | 391 | 388 |
| 实际 Release 字节数 | 2,686,344,007 | 2,670,784,786 |
| 清单名称/状态/大小不匹配 | 0 | 0 |
| `evidence.status` | published | published |
| `missing` | [] | [] |

两 Release 合计 779 项、5,357,128,793 字节。全部清单成员已做远端元数据核对；本报告独立下载复算其中 52 项，不声称把约 5.36 GB 全部读完。完整 Release 链接和核验对象见 publication-audit。

A 的 evidence 仍保留两次历史上传/finish 的 `curl (28) SSL connection timeout`。后续已完成缺失资产恢复与 remote verification；当前该 manifest 完整，不能把已恢复网络上传错误算成物理求解失败。B 同样按其最终清单核验。

未包含 checkpoint 权重和全视频符合现行 `delivery-policy.configured.release_parts.scope`，它们仍属未请求 bulk，不能把没有权重资产称为“请求原件未完成”。本次也没有从 checkpoint tensor 独立核验模型内容。此前 `prediction-model.xml` 的 optional_absent 对应实际模型资产命名差异，实际清单提供 `model.mjb` 与 `G1-project-go2.xml`，不应凭一个旧文件名推断模型没有保存。

## 9. 下一步判断

当前应该保留 r32 的数值修正与本轮完整负结果，推进能产生有效步态的控制机制。继续无变化地增加 cap-only PPO 数据，不能替代对 native 候选是否抬脚、是否前进及成本分项的检查。

本轮已保存的首次 `command_nonprogress_2s` 窗口可以串联 native 候选 → 实际获选策略 → feedback/projection → PD/物理执行。事件窗已存在，分析不需要额外物理回合。[求解器报告第 7 节](solver-report.md)和 [solver-summary.json](solver-summary.json) 对最终 E30 出发段全部 incumbent 复用调用进行了更完整的比较：在 A 27 次、B 29 次复用中，将 `incumbent_current_cost` 与所有 `search_attempts[].native_candidates` 内同等 trunk violation 的有限非失败候选比较，分别有 **3 次、4 次**存在更低的已计算成本，最大差为 **3.7890566726e-10、9.5388197861e-10**。差异微小但不严格为零，旧稿的 `positive_reuse_gaps=0` 断言已撤回。

这里的遗漏改善与本报告第 5 节的 accepted `native_decrease` 不是同一口径：后者比较获选候选与同一次尝试的 nominal，而遗漏改善比较复用 incumbent 与全部已评分候选。重新读取首 attempt 的 `nominal_current_cost` 也能发现上述 3/4 次微小正差，不能用 nominal 口径把它们解释成严格零。当前证据支持修正候选保留的完整性，不支持将这个极小缺口视为全部驻留的主因。实际执行对齐证据也需按覆盖解释：每流 17 个保存计划中只有 16 个有完整后续 20 ms（合计 32 段）；各自 30 秒终端计划的未来轨迹没有被继续执行。

后续两 CPU 可以并行运行明确不同的研究流或两条既定父点的工程复核，GPU 现有任务继续。继续采用样本/更新及物理评价目标、`deadline=null`、原件增量发布和用户自行备份清理；不追加电脑归档，不把等待 GPU 结束设为 CPU 推进的条件。
