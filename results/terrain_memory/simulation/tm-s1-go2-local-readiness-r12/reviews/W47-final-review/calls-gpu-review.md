# W47 全程调用、GPU、输入可用性与记录链独立审查

执行规则以[最终W48计划](../../../../../../docs/S1_W48_STABILITY_PLAN.md)为准；本独立报告的建议经主审查整合，不另开并列实验。

审查基点：`6e9075f48718509d48f51d08d0a8f62a34edfb41`。本子审查未运行服务器、plant、MPC solve 或 GPU，也未下载 Release payload。已独立解析仓库的全部 1,944 行 CSV，核对全部相应分层统计；GPU 部分核对现有 JSON/源码/执行身份，不能冒称重新读取 paired 原数组。可复算脚本为 [audit_w47_calls_gpu.py](audit_w47_calls_gpu.py)，派生结果为 [calls-gpu-audit.json](calls-gpu-audit.json)。R8 远端元数据核验单列 `r8-release-metadata.json`。

## 1. 最重要的结论修正

W47 Task C 证明，W46 的全程搜索并非一直没有新下降方向。四个旧 fixture 的 540 个正步全拒结论仍然有效，但不是全程代表。1,944 次调用包含 40 次启动调用：

| 范围 | 调用 | 正步安装 | 正步 J 下降 | 正步风险优先 J 上升 | 停滞 | nominal 安装 | incumbent 保留 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全部 A/B |1944|1601|1598|3|343|137|206|
| A moving/r35 |1222|1066|1063|3|156|73|83|
| B moving/r35 |482|317|317|0|165|58|107|
| 两流 moving |1704|1383|1380|3|321|131|190|
| 两流 Stand |240|218|218|0|22|6|16|
| 去掉启动的正式调用 |1904|1561|1558|3|343|137|206|

全程正步安装占 82.36%；moving 占 81.16%，A moving 87.23%，B moving 65.77%。`native_search_stagnation` 与顶层 `policy_installed` 不能直接互换：343 次停滞中 137 次仍安装某个 nominal，206 次保留 incumbent。`alpha_0_0` 等名字的末尾是槽位，不能把它直接解释为 alpha=0；本次按明确 `chosen_alpha>0` 计算。

仅 J 下降也不等于原任务稳定：这些值是同一调用内部、同一目标的候选比较。跨 epoch 的参考和状态改变，不能把不同调用的 J 或下降量直接求和当路线收益。3 次 J 上升都是 A 晚期风险优先恢复，分别在 25.44、26.36、26.38 s，降低 trunk violation。第一次由约 0.007718 m 降至 0.001421 m，而 J 从约 242.89 增至 15,115,504.21；因此均值会被极端事件主导，不能用全程平均 chosen_actual 评价稳定性。

### 失稳附近仍在下降的具体证据

独立按 CSV 原时标截取事件前 1 秒的 moving 调用：

- B 第一次 z<0.2 m 在 8.464 s；前 1 秒 50 次调用全部正步安装、全部 J 下降，停滞 0。最后一条 call444@8.46 s，alpha≈0.00237137，chosen_actual≈0.00751684。
- B 第一次 |roll|>0.5 rad 在 8.572 s；前 1 秒同样 50/50 正步下降，停滞 0。call449@8.56 s 的 chosen_actual≈0.00128148。
- B 首次完整膝限幅反转 6.15 s 前一秒 48/50 正步安装；事件对应 call328@6.14 s、plan267、alpha≈0.177828，chosen_actual≈0.00952207。
- A 首次低高度前一秒 43/50 正步安装，首次明显横滚前一秒 49/50；末次调用也有正步下降。

这些是描述性窗口，不是独立实验或因果证明。但它们足以否定“全程搜索没有下降，所以必然跌倒”这一过强解释。下一步应将精确状态差修复的实际闭环作用，与目标/模型/反馈恢复问题分开检查。

局部停滞仍存在：A 最长连续 moving 停滞 36 次，3.12–3.82 s；B 40 次，3.08–3.86 s。A >1rad 首跳发生于 3.846 s，其前一秒仅 9/50 正步、41 次停滞；B >1rad 首跳 3.106 s 发生在上述停滞段开头。既要保留这个早期证据，也不能以它覆盖后续持续下降时的失稳。

全程原 summary 统计 32,364 个正步候选、8,922 个严格成本下降候选，正步 rollout/QP failure 为 0；1,176 个正步有正 trunk violation，495 个 actuator restoration 节点。有限旧 fixture 的“全无风险/恢复”不能外推全程。真实工作量汇总为 57,104 次 native rollout、55,400 candidate records、3,888 derivative passes、7,488,288 transition FD、1,442,978 risk FD、3,596 backward attempts。它们是历史 W46 工作，并非 W47 新算。

## 2. W47 没有新增原任务失败

W47 的实际新工作为 8 条固定策略 350 ms 诊断，共 14,000 个 0.2 ms 子步、1,400 held 区间，独立诊断时长合计 2.8 s。新 optimizer solve、原 plant episode、PPO、GPU physics 均为 0。不能将这 2.8 s 拼成完整原任务进展；也不能将原任务 blocked 当成新跌倒。

phase 的 2026-11-13 北京时间约 12:29:41–12:40:06 包含准备和发布。paired 计算约 35.26 s，构建约 63.29 s，Task C 43 个源文件、122,624,900 B 的原位复算约 2.49 s。B 完成的是可用性负结果，未启动 donor solve；C 没有 GPU 分配。预设 A/B/C affinity 并不等于三项重负载实际同时运行。

12:22:15 的现场快照显示 cgroup 25 CPU、120 GiB RAM、GPU 97,887 MiB，其中一个无关 SuperMimic O4 low_lr 进程占 34,318 MiB。它是历史快照，不能当当前可用量，也不能当 96 GB 全空闲。本轮应优先让最小修复进入有信息增量的 A/B 实验；C 用现有数组形成结论，结束即释放。

server CSV planning 时间与客户端 RPC 时间不是同一口径：去启动后 CSV A 3214.876145 s、B 1873.160146 s；上一审查轨迹 planner_s 约 3216.135/1873.711 s，包括调用周边开销。保留两种来源，不能当成数值矛盾或把它们混加。

## 3. R8 并未被证明丢失：只是 D57 本地没有

W47 `task-b-source-manifest.json` 使用 `find /root/autodl-tmp` 查固定名称，并检查旧 `/home/fanyuan/...` 路径，均无匹配。`downloaded_bytes=0`，未读源 payload、未验证相位映射。

但同一 manifest 已保存历史 Release URL、size、digest。本审查再次读取 GitHub Release 元数据，确认资产仍存在：

- tag：`tm-s1-go2-wb-support-r8`
- 名称：`project-s3-formal-v1-single-yaw_plus5-trajectory.npz`
- asset id：681937433
- state：uploaded
- size：28,100,505 B
- SHA256：`079f57e1ebad260c9cdac842dc089012e29cb46992ab64d2c31cd9d380b61625`
- URL：<https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-wb-support-r8/project-s3-formal-v1-single-yaw_plus5-trajectory.npz>

仅元数据核验，不代表本次完成 payload 重算。W47 的 `donor_unavailable` 应读作“执行服务器本地不可用”，不是“全球已丢失”，更不是“donor 思路无效”。下一计划可以允许 D57 通过既有授权服务器路径，只恢复这一有明确哈希的科学输入，然后原位使用；这与 GitHub 后向用户电脑额外归档是不同的具体操作。无需重跑 R8 来造替代数据，也不需要下载整批历史 Release。

旧计划把本地缺源、最新两个事件快照不足，与最小正确性修复进入原 E30 的条件耦合过紧。现在有已证 identity 缺陷时，应该允许按最小修复冻结并做一次 A/B 原 E30，补齐新记录；donor 可独立准备，成功与否不应成为该正确性验证的硬门。

## 4. W46 关键上下文为何没保存

这不是上传漏文件。`task-b-fixture-manifest.json` 已检查 A 的 50 个交付 ZIP、B 的 34 个交付 ZIP 文件名，所需成员均不存在。

源头是 `scripts/s1_go2_r29_records.py` 的事件记录设计：

- `prune()` 仅保留约 0.12 s 环。
- `event()` 保存前 0.1 s、后 0.2 s。
- 大部分事件按类型 `first=True` 只触发首次。
- 触发是 nominal 饱和、raw Kδ 大、目标 ctrl clip、恢复、材料/非足加载、terminal 等；没有实际 held 目标跳变或实际膝 ±力矩限制反转的触发。

A 6.86 s 与 B 6.15 s 的首膝反转均不在任何现有保存窗里。A 所需 plan305/call364 缺，B plan266/267/call328 缺；同 origin 的前一条 accepted prediction 也没有。A/B 实际只保留了 155/37 个 full plans；不能根据标量日志伪造完整 K/H 或前次预测。

已有另两项直接相关原件可用于补充：

- A 首次 >1rad 目标跳变 3.846 s 对应 call213@3.84、plan178，full NPZ 4,650,170 B，SHA `05cbfeefdfad94cf8aef2df930e3bc5b4e73e72c4d971fb59a4c9ab3ac79b394`。
- B 首跳 3.106 s 对应 call176 保留 call175@3.08 的 plan161，full NPZ 4,683,880 B，SHA `51e82f54de3d39b4f2fc053774b47af89a2602735e2d988e3cff2350fb4cc91a`。

Task C 已在服务器核了它们的哈希，但仅这些 policy 仍不等于完整 pre-call context，不能冒称 exact replay。

### 最小记录接入建议

1. 每个 planner 边界，在调用 `agent.planner_step()` 之前保存当前真实 x0/time/act/mocap/userdata、private gait 和当前 incumbent P0 身份。从**实际部署的上一策略已接受 rollout**取同 origin 的预测状态，记录 source plan/call/node/time/hash，并计算 MuJoCo 位置切空间差与速度差。
2. `accepted_states` 与 `reference_states` 严格分开。前者是被评分的实际 rollout，后者定义反馈中心；不能为了再现而把两者互换，也不能用本次新求解轨迹冒充上次预测。无匹配节点/已超 horizon 就标 missing，不外推成零误差。
3. 所有调用保留上述小型 prediction/observation/身份记录；完整 policy/context 使用约 0.12 s 内存环。`r35 ubar/F/R` 同 epoch 只存一次，以 hash 引用。
4. 在 runtime 的实际 `mj_forward` 后、进入物理 substeps 前，已有 held u 与实际力矩可读。添加首 >1rad 相邻 held 跳变、首膝 ±45.43Nm 反转事件，加现有 terminal，保存前0.1/后0.2 s。阈值只用于选择诊断窗口，不改变原任务终止或物理限幅。
5. 重叠窗口合并；完整策略按不可变 plan/hash 去重。每个 packet 保留 pre-solve context、P0、实际安装 winner、上一策略 accepted prediction、同 epoch reference；不复制每次所有几十个候选或全套导数大数组。若需 exact pre-call optimizer replay，必须额外保存相关 numerical lifecycle；否则明确 common reconstruction。
6. 在 r36 的 recorder/controller 白名单中显式接入，防止新 build 不在当前 `S5-r33/r34/r35` 元组里，导致新诊断字段静默不记录。run/prepare/contract/recording 身份应同版。

大小估算（FP64、T176/ns37/nu12/nx36，仅作设计预算，非实测压缩比）：一份最小执行策略 reference x/u/time/K/H/header 约 881,440 B；另加 accepted states 约52 KB及必要 context。每个 50 Hz 调用保存 actual+previous predicted 两个37维状态只需592 B，30 s约0.888 MB、600 s约17.76 MB，另加元数据。单个前0.1/后0.2 s窗口约17个新policy，上界约15 MB基础数组，再加P0/accepted/context；按hash去重可降低。无需为了新记录再产生几十GB全候选副本，旧原件保留不删。

## 5. GPU 实际进展与残留问题

W47 在服务器读取了此前未独立复算的 paired/action 原数组，追加 corrected JSON；本子审查只复核该 JSON 和代码，没有重读 raw payload。24 个名义场景的执行哈希再次得到 6 个不同完整策略、18 种场景。源仍是 W45 baseline/selected，不是 r35 新候选，更不是 identity 修复后的新策略。

初始同 state 的 held 差约 2.4893e-16 rad；24 个初始中心均在 PD box 内，最小裕量0.323743 rad。第一2 ms后的关节位置差约1.4625e-5 rad、关节速度差0.02664 rad/s；+10 ms决定的下一hold命令差可到1.50136 rad。20 ms端点关节角差0.0235766 rad。完整350 ms中，最大held命令差1.75833 rad、关节位置差0.130953 rad，后者在252 ms出现。

离线恢复 raw 公式得最大 raw 差784.29 rad，必须注明这是各自已分叉状态带入旧策略的重算值，不是设备原raw输出，更不是实际关节转动或施加命令。所有24场景的2 ms采样 body_fall和trunk终值仍为0，不足以证明逐0.2 ms硬条件等价。

已修 raw 别名、决策/hold时标、扩窗解释和 capture计数。可选三引擎首hold未运行，原因是既有接口不支持预算内所有必要读数。W47 GPU新分配/物理步均0，不能宣称本轮加速验证通过。

### 新的同批重复性线索

六组“完整策略×初态×origin×model哈希”相同的重复场景，其 native min-z/max-roll/max-pitch 统计逐项相同；GPU同类统计并不相同。最大同身份组的 min-z统计差约0.00018048m、max-roll统计差0.000823864rad。

这是根据已提交分场景极值计算，尚未逐时读取paired数组，也没有证明隐藏初始化/接触排序/solver workspace完全一致。它说明GPU同批重复性应加入解释，不能将全部差异直接归为纯FP32或CPU/GPU引擎版本。若第三线有余量，服务器现有paired数组可零新physics计算这些重复组的逐时差和首次分叉，形成有用报告；无需再开一轮24world。

### 历史 driver 的输出仍硬编码 W46

`scripts/s1_go2_w46_gpu.py` 虽修了测量代码，但 `OUT` 仍固定 `gpu/w46`、`SMALL` 固定 `w46`，直接运行会覆盖旧结果/原数组。W47没有运行它，因此未发生覆盖。下一轮若需要GPU执行，必须新输出根和新身份，不能以“修正版本”直接重跑这个历史入口。修改该脚本本身也不证明新代码已在GPU执行。

## 6. 对 W48 的具体建议

优先将已证 state identity 最小修复从诊断构建变成真实 r36，完整核接入和已知四fixture的必要有限验证；其准入与 donor 数据可用性解耦。随后一次固定 A/B 原 E30，以新记录判断修复对早期跳变、下沉/翻转、真实材料路线有无作用；各自通过后才按原阶梯继续。保持新 PPO0/0、原物理/原路线/原r34。

R8输入可独立恢复并核schema/time/phase，供后续研究使用；最终W48计划本轮不做donor求解，以免混入identity-only闭环评价，也不阻塞主线。第三线只完成现有GPU重复性、计时/源码对照及新记录覆盖检查，没有明确新问题就结束。不要把有限/诊断限制变成必须把所有独立科学假设一起验证后才允许最小修复上原任务的循环。

后台继续无固定墙钟截止；保留物理阶段和有限数值预算。GitHub发布后不在用户电脑额外归档；服务器原位读取与仅恢复一个明确科学输入应允许，既有服务器原件不自动删除。
