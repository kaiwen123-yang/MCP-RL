# W48 记录、原任务、GPU与交付独立审查

结果基点：`61e16975fa6b38dc8e2d31779c2bd82f1aaeb8b6`。本审查只读仓库代码、提交JSON/CSV与源清单，不登录D57、不下载Release payload、不执行新native优化/plant/GPU物理。`audit_w48_records.py`已运行，派生证据见`records-audit.json`。

## 1. 本轮实际完成和失败分类

W48已经完成正式r36、8次有限solve、4次350ms实际scale的ActionImpl核对、两条原E30。原任务A26.432s躯干触地、B11.638s body_fall，E30 0/2，后六级blocked，新PPO0/0、donor0、GPU新physics0。

两条原任务的500Hz time/qpos/qvel/target/torque/GRF/contact/cap/phase与W46逐元素相同；全部选择来源、alpha、安装plan相同。此全轨迹结论来自D57执行者的`w48/original-comparison.json`，本审查没有重新读原NPZ。独立核对的277个事件call，其选择/alpha/plan/rollout和FD数量全部与W47提交的W46全程CSV一致；其中15个事件call的chosen_actual不同，说明成本计算受修复影响，但未改变这些选择。

四个此前零plant技术attempt（A三个，B一个）是配置继承导致的启动技术错误，和两次物理失败分开。它们完成的native call记录为0、plant物理步为0；启动过程中已做但未完成记录的native计算工作量未知，不能写0。这四次不是四次新增跌倒。

| 工作 | 实际数值 |
|---|---:|
| 原任务completed native calls |1944|
| 原任务native rollout |57104|
| 原任务candidate records |55400|
| 原任务derivative passes |3888|
| 原任务transition FD |7488288|
| 原任务risk geometry FD |1442978|
| 原任务backward attempts |3596|
| 原plant物理子步 |190350（38.07s）|
| 有限工作native rollout |484，含4部署rollout|
| 有限工作transition FD |30816|
| 有限工作risk FD |5012|
| 有限工作显式rollout物理步 |847000，不含FD内部积分|

本轮比上一轮进了一步：真正测试了修复版原闭环，取得“这两条轨迹不变”的有限负结果。不能再重复相同r36配置指望资格变化。

## 2. 1904条上一accepted预测配对的含义

全程汇总为A1322/B582可用配对，A/B各20个startup调用尚未有实际部署策略，标unavailable合法。所有原任务call合计1944，因此1904配对加40startup一致。

源码的生命周期是合理的：

1. runtime在`agent.planner_step()`之前调用`before_call`，此时`self.deployed`仍指上次实际applied_hold的策略。
2. 从该策略的`accepted_states/accepted_times`选同时间节点，不用反馈reference或本次新解；仅允许时间差≤1e−9，不插值。
3. 新求解后从native当前transport读取完整policy与实际scored accepted rollout，作为`self.current`。
4. 实际applied_hold之后才令`self.deployed=self.current`；retained incumbent也使用本次从真实起点重新评分的accepted rollout。
5. pre-call native state与Python actual state逐元素断言一致。P0从native在修改epoch/策略之前保存，normal运行没有offline import。

因此这些记录不是“本次新解预测本次初态”的伪零误差，也没有把反馈reference当accepted rollout。

### 本审查独立覆盖的事件子集

巨大resolved JSON已经内嵌完整事件manifest，可不下载原件读到：

| 指标 | A | B |
|---|---:|---:|
| 已提交event call |131|146|
| 其中available预测 |111|126|
| 原始startup unavailable |20|20|
| 独立完整对象 |484|511|
| 对象总字节 |178497406|152006857|

277个event call和995个对象全部与交付sources清单的size/SHA对应。237个available记录全部满足：

- `source_call == current_call-1`；
- P0 policy hash与上一deployed policy hash完全相同；
- P0 plan与上一source plan相同；
- `node_time_s == origin_s`精确成立；
- translation/hinge位置、所有速度的actual−predicted算术误差0；
- 向量与分单位字段切片一致。

其中233条对应node10（通常20ms），2条node1、2条node9。后两类来自每流2.002s阶段额外规划：2.000→2.002为2ms，2.002→2.020为18ms，不能机械把所有配对都叫20ms。

223条上一source call也在已提交事件集合中，其installed-policy/accepted-rollout hash交叉核验全通过；14条处于事件窗口起点，上一call在保存窗之外，本审查未读完整上一轨迹，但当前边界仍保留预测状态和P0对象。这不是1904条原JSONL都已独立复算；全JSONL仍需D57原位分析。

### 首次误差与显著失稳不是同一时刻

A/B首次非零都出现在0.02s的call21，source call20/node10，仍处initial_stand。两流actual/predicted状态一致。其最大平移差约5.50e−7m、旋转切空间差8.25e−6rad、关节差3.19e−5rad、关节速度差.00365755rad/s。它提供了早期执行一致性的检查点，不能直接认定为数秒后跌倒的唯一原因。

完整汇总报告的预测误差极值如下（本审查只核提交极值，未重读全1904原件）：

| 单位分组 | A最大绝对分量 | B最大绝对分量 |
|---|---:|---:|
| base位置/m |.000556610|.000209935|
| base姿态/rad |.0144450|.00653313|
| 关节角/rad |.0492763|.0164553|
| base线速度/m/s |.131490|.0333780|
| base角速度/rad/s |2.27472|1.14723|
| 关节速度/rad/s |7.15560|4.30498|

不能把这些不同单位的量合成一个无意义“最大state error”。小的base位置误差也不能证明支撑恢复良好，因为速度和关节误差可以明显不同。

## 3. 新事件已覆盖上一轮缺失的关键位置

| 流/事件 | 事件时间 | 前一规划边界 | 策略选择 | 该边界最大关节角误差 | 最大关节速度误差 |
|---|---:|---|---|---:|---:|
| A首held>1rad |3.846s|call213@3.84，plan178|alpha=1，J下降.03243049|8.53e−6rad|.0010774rad/s|
| B首held>1rad |3.106s|call176@3.10，plan161|incumbent、停滞|.000150774rad|.0283081rad/s|
| A首膝反转 |6.860s|call364@6.86，plan305|incumbent、停滞|.00136426rad|.514934rad/s|
| B首膝反转 |6.150s|call328@6.14，plan267|alpha=.17782794，J下降.00952207|.000671347rad|.0727536rad/s|

这些是规划边界的误差；前三个相邻关系中的held事件发生在边界之后，A/B首大跳均晚6ms，B膝反转晚10ms。下一步必须结合逐2ms实际action/native feedback logs，不把边界误差移到事件时刻。

四个事件均有policy、accepted、P0、native/Python context、同epoch参考五类完整对象，且size/hash已进入原件清单。因此不必再为这些事件重新跑W46/W48来采集快照。

事件cost分项也已可用：A首大跳前call213的J≈2.18749，其中MaterialSupportHeight≈1.86129；B首大跳前call176的J≈.136767，其中该项≈.078966。分项大不等于因果责任，但它允许后续把“哪个目标鼓励/排斥某个候选”定量分析，不能只看J总数。

### 一个范围限制

`W48Records.plan`覆盖父类plan而没有调用父类解析`planner-records.jsonl`的trunk-restoration事件逻辑；本次事件清单也没有`trunk_restoration`。新held/膝反转、阶段、terminal触发都有效，但不宜声称旧recorder全部事件与大候选/导数保存已完整继承。当前call_record保留风险/接纳信息，完整候选池和FD数组不逐call保留是明确的体积控制。后续若需要特定事件的候选/导数，按当前完整context有限重算，不因此重开整轮。

## 4. 正式调用与native固定部署检查尚有接口边界

runtime每2ms在SetState(double)后显式调用`get_action(time=t)`。仓库继承的`agent.proto`定义GetActionRequest.time为float，GetActionResponse.action为repeated float；State的time/qpos/qvel与BestTrajectory是double。动作FP32早在W40计划和r29/r34说明中已知，不是此次新发现的唯一根因，也不能解释此前native pre-RPC QP问题。

W39已识别的路径在新W48中仍有未闭合的检查点：实际requested action time经float传输，而native求解/固定部署用double时间；实际held还经float返回。W48的四条ActionImpl固定核对是native-to-native，没有覆盖完整RPC传输。应使用现有`feedback-actions.jsonl`、`action-execution.jsonl`、`w48-held-clock.jsonl`对齐native查询时间/状态、native命令和实际held。这里只指出未覆盖的真实路径，不预先断定它对任务的因果影响。

如root控制源码复核确认省略time能使用服务端已设置的double state.time，可以作为最小且可解释的接口试验；是否进入原任务应由精确同状态/同policy比较决定，避免再次跑一条实际上没有行为变化的控制器。

## 5. GPU重复身份现在有了逐时结论

W48 C原位读取并核对既有四个源文件，6组重复场景，每组两条：

- 保存的初态在两条重复world之间逐元素相同。
- native全350ms状态和held动作逐元素相同。
- GPU六组均在首个2ms状态节点首次不同，随后在2ms决策的held动作首次不同。
- 六组最大base位置差.000513100m，姿态差.00604694rad，关节差.00608480rad，关节速度差1.69877rad/s，held差.0746461rad。

这是“同一GPU批次内重复身份”的差，不是此前CPU/GPU全窗1.758rad held差，不能混用。来源仍W45策略、历史W46 GPU公式＋host H-QP＋FP32 engine，与r36新策略无关。

源数组没有保存各world初始qacc_warmstart、逐substep接触/法向力/KKT/solver痕迹、完整allocator/初始化状态，无法唯一拆分隐藏状态、world索引和浮点运算顺序。W48 C没有新增GPU分配或physics，wall约1.935s；可选动作批次没有运行。

不应再重复24world长窗。若GPU需要独立推进，应限定一次很小的同状态/同held/同初始化首hold差分，补这些具体缺失字段；否则现有有限负结果已足以阻止把GPU评分器当正式native替代。

## 6. 并行确实发生，显存并未因此全被利用

A/B正式native进程affinity读回分别0–9、10–19。两原evaluation子进程生命周期真实重叠2030.4135s（约33.84min）。这是实际双任务同时运行，不只是设置了两个目录或tmux。

A报告wall3265.949s（54.43min），B2017.731s（33.63min）；与W46墙钟变化不能单独归因记录优化或并行效率，因为同机其它负载/实际利用率未形成受控对照。完整控制轨迹相同，至少说明本轮额外记录没有改变这两条物理轨迹。

第三任务C仅1.935s既有数组工作，没有GPU动态计算，因此没有持续三重GPU负载。下一计划可以继续10+10+2+3额度、线程库1、构建先完成；不要拿目录affinity或显存容量冒充已得到MPC加速。

## 7. 交付成功与残留尾项

当前科学产物finite8个资产、原A3、原B2，共13资产、1,775,245,315B，提交的remote size/digest记录一致。A原件曾TLS失败，续传既有manifest完成，最终科学结果不再待传。两段视频在原件源清单中。

仍有两份旧技术job manifest：

- `w48/delivery/w48-a`：publication_pending，upload_exit_code1，1ZIP/20个源文件。
- `w48/delivery/w48-b`：同上。

后续finite/A/B源清单没有覆盖这些旧原路径；按字节SHA共享已有覆盖A14/B13项，**坏模型MJB已经有相同字节进入后续科学包**，不能说它全部丢失。仍有A6/B7个非空文件的字节没有被后续清单覆盖，主要是旧日志、contract/job/runtime；B另外缺1,672,688B pre-call原件。两个旧ZIP本身仍有必要按原manifest完成交付调和。

root实时GitHub核验另确认旧技术tag不存在，故不是简单回执未写。但这只是发布尾项，不阻塞控制分析/新实验。应仅在D57续传已生成原manifest，既不重建实验，也不重新打包覆盖旧ZIP；GitHub核验后不在用户电脑归档，服务器旧原件不删。

R8本轮下载明确失败于`unexpected EOF`，downloaded_bytes0。它仍是网络可用性负结果，不是donor机制失败；本轮donor0、R8 schema/time/phase没有完成。主线无需继续绑定它。

## 8. 给W49的最小建议

1. 原位使用已有W48全1904边界、逐2ms动作日志和四个关键事件完整对象，先查实际API time/action transport与native实际求值的第一分叉；不重跑原任务来找已存在的数据。
2. 严格区分nominal accepted rollout、反馈reference、native输出、RPC held、真实state。动作FP32历史已知，timestamp的float路径需要定量核而不是只凭源码命名断言。
3. 有明确行为干预才进入下一次原A/B；未改变选择/动作的重复r36无新科学信息。若最小接口干预成立，可保持原N/r34/constant/约束和checkpoint作一次明确对照；不同时扫步态、成本、PPO。
4. 同步把有限实验中source/deployment一致性的范围扩到真实接口，避免再用native-to-native对照回答包含RPC的实际执行问题。
5. 第三线只做有缺失字段的有限GPU问题或旧交付续传，R8保持独立。整个工作继续无固定墙钟截止、无电脑额外归档。

这些建议由root整合为唯一W49计划；不另行启动或增加本审查工作量。
