# Piper H 公开资产核查：接口和真实参数文件存在，任务闭环仍缺证据

日期：2026-10-17。接续项目`24b392c`及[纯功能核心复审](../../docs/joint-update-core-independent-review-2026-10-17.md)，对应[臂侧完整任务合同](../physical-task-instance-2026-10-17/robot-task-contracts.md#3-臂候选-a预夹持长形刚体经通道后转姿态放到承托区)。本次只做官方及作者一手来源的静态读取、元数据下载和文本/URDF解析，没有执行任何外部仓库Python、训练、推理、环境rollout、CAN/ROS命令或服务器连接。

**当前有可取得的Piper H官方SDK/URDF，以及作者明确标注Piper-H的视觉模仿策略参数文件。它们足以推进接口审查；尚不足以提供本项目已校准的低层反馈、命令到持续状态的响应模型，或受扰通道—转姿态—承托释放的闭环证据。**用户的Piper H未到货、未调试状态不变。没有据此选定主平台。

可机器复查的提交、文件大小、来源声明、关键文件hash和未执行项见[piper-sources.json](piper-sources.json)。clone和外部原文只放忽略目录`work/platform-assets-2026-10-17/piper/`；公开提交只有本原创审计及来源元数据。搜索范围是下列具体官方仓库和三个具体作者模型，不作“全网不存在”结论。

## 1. 已核资产分层

| 资产 | 固定版本与许可证 | 本次实际取得/核实 | 能支撑什么，不能支撑什么 |
|---|---|---|---|
| AgileX `pyAgxArm` | `d90e234ba8e408cfa964b38aa9120b3131fc9f85`；LGPL-3.0；本次API未列release/tag | 官方文档、机型常量、H驱动与示例源码静态读取 | 有H路由及模式/固件约定；不是实机固件验收或已辨识执行模型 |
| AgileX `agx_arm_urdf` | `680fa925cd00db7e2628563d2edc5540963045de`；MIT | `piper_h`与普通`piper`独立URDF文本、mesh路径和Xacro清单 | 有规划/可视化几何起点；惯量/effort字段不是实机动态校准或安全上限 |
| AgileX `piper_sdk` | `b847e5393a7ce6ff8508d713eca7bdd3a42421b4`；MIT；最新release `0.7.0a1`，2026-10-14 | README、release说明、仓库版本 | 旧接口及V3迁移依据；不能用未区分机型/固件的脚本控制H |
| AgileX `lerobot-agilex` | `8469f3ab67e27987d217e8e1d144b8710a4170a5`；Apache-2.0 | README、ROS2推理脚本、静态文件树 | ACT/Diffusion框架/ROS桥接代码；README指定本地训练输出路径，不是已提供的Piper H任务checkpoint |
| Kavin初版 Piper-H plate-pick | HF `5156880ee9d4cbd09ad86099b6759b890ae88fed`；作者卡Apache-2.0 | 模型API树、Orbax tensor metadata、归一化、配置及数据转换源码；**未下载参数分片** | 确有参数文件清单；作者明确没有physical rollout评价，验证loss不能当成功率 |
| Kavin `ft-v2` | HF `d2ba481695bf29097f559018f2bf7935305480ea`；作者卡Apache-2.0 | 指定revision的API树、卡片、serve/converter/config/policy、归一化和参数metadata；**未下载参数分片** | 更具体的H右臂抓板候选；私有训练源、固定home/camera且无本项目持续响应校准 |
| angkul `pi05-piper1h-ea`候选 | HF访问未成功；GitHub `openpi`分支`piper1h-pi05`：`a079613d5d36072f7b4ac7c2dfba67c244f54951`，Apache-2.0 | GitHub配置和policy transform可读；HF页面/API/README resolve本次均HTTP 401 | 代码recipe已核；不能据搜索模型卡称权重已取得、可加载或可运行。401不等于资产不存在 |

官方来源：[pyAgxArm](https://github.com/agilexrobotics/pyAgxArm/tree/d90e234ba8e408cfa964b38aa9120b3131fc9f85)、[URDF](https://github.com/agilexrobotics/agx_arm_urdf/tree/680fa925cd00db7e2628563d2edc5540963045de)、[Piper SDK release](https://github.com/agilexrobotics/piper_sdk/releases/tag/0.7.0a1)、[LeRobot fork](https://github.com/agilexrobotics/lerobot-agilex/tree/8469f3ab67e27987d217e8e1d144b8710a4170a5)。这里记录许可证标识与文件，不提供法律结论；第三方数据/基础模型的各自条件不能由代码许可证代替。

## 2. Piper普通型与H有具体差异

[官方常量](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/pyAgxArm/api/constants.py)与[H URDF](https://github.com/agilexrobotics/agx_arm_urdf/blob/680fa925cd00db7e2628563d2edc5540963045de/piper_h/urdf/piper_h_description.urdf)明确区分：

| 字段 | 普通Piper | Piper H | 对本任务的影响 |
|---|---:|---:|---|
| J4预设范围 | ±100° | ±135° | 转姿态路径和可用关节空间不同 |
| J5预设范围 | ±70° | ±89.5° | 载物姿态、腕部余量不同 |
| MDH第4项d | .25075 m | .25575 m | 不能原样复用FK/扫掠几何 |
| MDH第6项d | .091 m | .0963 m | 法兰几何不同 |
| 电流→力矩系数 | 普通型k/b/c | H J2、J4–6若干系数不同 | 原始电流/编码值不能不经H配置当作相同物理力矩 |

H的k为`(1.18125,1.65375,1.18125,.9576,.9576,.9576)`，b为`(4,2.5,4,1.7,1.7,1)`，c为`(1,1,1,.75692,.75692,1.28676)`；这是所核SDK换算配置，**不是本机实测最大连续/峰值力矩**。URDF各关节写`effort=100, velocity=5`，也不能把它们直接当本任务已验收硬限额。

[固件表](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/docs/piper/firmware_reference.md)中H使用Piper固件路由的薄子类，但不同机型常量仍生效。固件≤S-V1.8-2、S-V1.8-3至7、S-V1.8-8、≥S-V1.8-9分别选DEFAULT/V183/V188/V189。MIT从8-bit＋CRC切换到12-bit无CRC，扭矩缩放路径和模式码也变化；代码兼容并不意味着任意固件均可用同一wire command。用户实机版本尚未知，本次没有读硬件固件。

[API](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/docs/piper/piper_api.md)给出三类不同对象：`move_j`是6关节目标角位置/速度模式；`move_js`是关节跟随透传、无轨迹规划/平滑；`move_mit`逐关节给`p_des,v_des,kp,kd,t_ff`，概念力矩为PD项加前馈。状态可读关节角及电机速度/电流/估计力矩，消息带接收统计。**get_fps的接收Hz、1 Mbps CAN、示例sleep或编码可表达范围都不等于本机端到端反馈带宽、可持续输入和测量精度。**官方力矩来自电流系数换算，不能作独立末端六维力测量。

## 3. 两个Kavin H策略是不同版本，不能混为同一个已验证策略

### 初版：公开参数和转换链，未报告实机闭环

[初版固定模型树](https://huggingface.co/Kavin60606/pi0-fast-piper-h-plate-pick/tree/5156880ee9d4cbd09ad86099b6759b890ae88fed)有六个Orbax checkpoint。`main/4000/params`包含21个文件，总计**5,577,621,598 bytes**；包括train_state的整个`main/4000`为**8,585,065,819 bytes**。API提供实际LFS分片尺寸及SHA-256；小型`_METADATA`中存在PaliGemma参数及LoRA张量形状。这超过“只提供训练代码”的证据，但本次没有下载、校验权重内容或反序列化，所以不标为已load/已推理。

已核[policy transform](https://huggingface.co/Kavin60606/pi0-fast-piper-h-plate-pick/blob/5156880ee9d4cbd09ad86099b6759b890ae88fed/scripts/piper_policy.py)和[config](https://huggingface.co/Kavin60606/pi0-fast-piper-h-plate-pick/blob/5156880ee9d4cbd09ad86099b6759b890ae88fed/scripts/config.py)：当前7维state是右臂6关节＋夹爪；输入front及右腕图像、固定抓板语言；右腕填入名为`left_wrist_0_rgb`的模型slot，不能因slot名字改成左腕图像。输出10×7动作chunk，记录数据20 fps。关节训练目标转为相对当前state的delta，夹爪保持绝对量；输出变换恢复关节绝对目标。无显式速度、力/载荷、观测历史栈、MPC局部命令/保持时钟或本任务阶段输入。

初版将源14维向量取`[0,1,2,3,4,5,12]`，左臂由外部控制器park。卡片将机型称H，源数据`robot_type`只写`agilex_piper_bimanual`；尚无硬件序号/固件与H几何核验。模型卡明确无physical-rollout评价，只报告FAST token验证loss。其训练混入人类视频重定向片段，不能称全部数据均为真实H执行。

### ft-v2：有更具体执行约定，但不能把重采样数据当实际闭环响应

[ft-v2固定revision](https://huggingface.co/Kavin60606/pi0-fast-piper-h-plate-pick-ft-v2/tree/d2ba481695bf29097f559018f2bf7935305480ea)总文件树245项；指定`checkpoints/ppv4ppv5_run1/2000/params`21项为**5,577,614,656 bytes**，包含train_state的整个2000目录为**8,512,426,442 bytes**。卡片把params约写8.2 GB，文件树表明应区分参数与整个checkpoint，不能把两数互换。此版本从初版main/4000继续微调，不是另外一个从零训练的独立证据。

[模型卡](https://huggingface.co/Kavin60606/pi0-fast-piper-h-plate-pick-ft-v2/blob/d2ba481695bf29097f559018f2bf7935305480ea/README.md)将源数据标为私有`PranayTest/piper-ppv4-ppv5`，121个抓板episode、48,043重采样帧；只报告固定划分的验证loss，并称step2000已部署。未提供可本次复核的闭环成功分母、失败/扰动轨迹、执行时序或本项目任务数据，故“作者称部署”不等于本次验证其闭环能力。卡片还记录前版在相机安装和home变化后部署失败，这直接说明场景、相机和初态不能自由迁移。

已核输入输出：7维右臂状态（关节rad、夹爪0–1）、front和右腕RGB、抓板prompt→10×7绝对关节/夹爪目标；serve脚本将图像resize到640×480。作者要求固定home约`[1.252,2.198,-1.865,1.192,1.146,-1.113,.985]`，训练home分布很窄；本项目不能直接采用该姿态当实机可行初态。

两版实际下载的`norm_stats.json`均有7维state/actions统计，但数值不同。例如J4/J5/J6的state均值，初版约`[-1.189,-.954,1.176]`，ft-v2约`[1.052,.993,-1.168]`。这是对应数据分布的统计，不是本机零位/符号校准；不能跨版本复用旧归一化或只凭这组符号变化推断驱动有错误。公开serve只返回动作数组，未落实本机如何选择`move_j`/`move_js`/MIT、限幅和保持规则，故实际控制模式仍未核验。

**最影响响应辨识的证据在公开[converter](https://huggingface.co/Kavin60606/pi0-fast-piper-h-plate-pick-ft-v2/blob/d2ba481695bf29097f559018f2bf7935305480ea/scripts/ppv4ppv5_to_lerobot.py#L234)：同一个插值后的`actions[i]`同时写入`observation.state`与`action`。**原字段是right `joint_positions`和gripper open amount。没有独立实际发送指令与关节测量的因果配对，这份格式不能直接供`F(s,u)→s_next`辨识，不能当原PPO采样动作/old log-prob；“动作列存在”并不足够。

原telemetry约17.83 Hz、视频30 fps，由转换器插值/取帧到20 Hz。因此20 Hz是数据/目标序列约定，**不是已测低层反馈周期**。模型卡称推理约0.8 s，而一个chunk覆盖0.5 s；未提供本机端到端请求年龄、chunk消费、伺服响应和后续状态对齐记录。只知道10步预测跨度，不知道真实有效反馈延迟`Δ`和扰动后重观测生效时间。本次没有运行serve脚本或访问任何部署服务。

该资产没有本任务MPC命令输入、当前宏命令保持时钟/阶段、载荷与滑移状态；将它接在MPC后必须先定义合法命令条件化方式及共享执行映射。直接让MPC覆盖其动作或只下发任务语言，均不是已经验证的“可调低层技能＋配对响应模型”。

## 4. 真数据、训练代码与未取得checkpoint分别记录

已实际读取两个初版上游数据集API和`meta/info.json`：

| 数据 | revision | 原始metadata | 已核字段与边界 |
|---|---|---|---|
| [single plate](https://huggingface.co/datasets/PranayTest/02062026_single_plate_pick/tree/40774fcc921e9ee4e0a535c081f1a57f9ba510cc) | `40774fcc921e9ee4e0a535c081f1a57f9ba510cc` | 113 episodes、73,232 frames、20 fps；Apache-2.0 | 14维state/action、front/right/left三路视频、camera timestamps、两臂joint_angles_deg telemetry字段 |
| [multi plate](https://huggingface.co/datasets/PranayTest/02062026_multi_plate_pick/tree/244d31f58c5c8a83fc8b505d2e0ee50ca097e801) | `244d31f58c5c8a83fc8b505d2e0ee50ca097e801` | 198 episodes、110,137 frames、20 fps；Apache-2.0 | 14维state/action、三路视频及camera timestamps；此meta未列joint_angles_deg telemetry字段 |

原metadata帧数与初版卡片40,633/90,827的转换后帧数不同；作者脚本在夹爪闭合后保留60帧并截断，因此不能混报原始/转换覆盖。只取得metadata与文件清单，未读parquet/视频样本、未验证时间戳同步或动作语义。公开遥操轨迹是行为数据，不能称本项目当前policy的闭环响应标签。ft-v2私有源不能由这两套初版公开数据替代。

[angkul公开代码](https://github.com/angkul07/openpi/tree/a079613d5d36072f7b4ac7c2dfba67c244f54951)有`pi05_piper1h_ea`/`pi05_piper1h_teleop`配置：14维双臂state/action、左臂基本静止、两路实际图像、内部action_dim32、horizon30、20 Hz；delta mask为每臂6关节加1绝对夹爪。它与Kavin的7维顺序不同，不能共用归一化/列索引。代码将部分数据注明private；本次HF模型HTTP 401，因此不确认所传模型卡的599 checkpoint内容。源码还以J5±1.22 rad讨论重定向限位，与官方H的±1.56207不同；需追溯其具体采集/重定向几何，不能只凭“H”名称消除这个差异。[相关config](https://github.com/angkul07/openpi/blob/a079613d5d36072f7b4ac7c2dfba67c244f54951/src/openpi/training/config.py#L605)

官方LeRobot fork的README给出ACT训练和本地`checkpoints/100000/pretrained_model`推理例；其ROS2脚本默认ctrl_freq25、obs_history_num1。**这是桥接程序默认值，不是上述Kavin模型或Piper H已测控制频率。**静态树中若干safetensors属于ALOHA/PushT/XArm测试fixture，不能把任何同后缀文件算作Piper任务权重。[官方桥接源码](https://github.com/agilexrobotics/lerobot-agilex/blob/8469f3ab67e27987d217e8e1d144b8710a4170a5/scripts/lerobot_inference-ros2.py)

## 5. 对臂侧各阶段的覆盖，以及真正还缺什么

| 本项目阶段 | 现有资产可贡献 | 仍缺的直接证据 |
|---|---|---|
| A0统一预夹持/启动 | H SDK/夹爪接口、几何；抓板IL可供研究动作/图像接口 | 当前载物夹持变换、滑移保持、实机home和已夹持启动状态。抓板策略不等于统一预夹持后的搬运策略 |
| A1受扰、载物绕障/过通道 | 独立H URDF可构建机器人几何；公开q/图像示范提供格式参考 | 物体attached geometry、外力波形、q/dq/伺服/延迟及合法MPC命令到响应，受扰后的全连杆＋载物连续无碰撞证据 |
| A2转姿态、接近与承托 | H较大腕部范围和真实模式区分，支持后续建模 | 已测载荷/重心、接近速度/姿态、接触与支承事件；抓板loss不能替代这些能力 |
| A3释放、退出、物体放稳 | 夹爪接口可作为待验收释放模块 | 放置后外部物体稳定测量、真实释放时序、退出无碰撞、完整成功/失败分母 |

这些公开模型都是具体视觉模仿策略候选，**不是当前已训练的、可由MPC局部命令调节且经过响应辨识的低层RL策略**。当前最强可立即做的工作是静态接口/几何一致性审查，并为厂家伺服＋合法轨迹跟踪＋完整载物几何的强基线准备真实输入输出合同；不能拿受限示范policy故意削弱经典规划基线。

如果以后要用这些权重作为初始化，需先补齐实际joint/state单位与列顺序、相机位置/标定、完整动作映射/模式/固件、chunk消费及传感执行时间戳、可用命令和真实闭环片段，再判断它是否适合本任务。实际策略版本的响应必须用其执行数据或经独立校准的组合模型获得；URDF、遥操数据和训练loss各自不能提供该标签。

本次结论只关于所查版本，不排除其它公开或供应商资产。没有请求访问私有数据，没有连接设备/服务器，没有启动训练或新仿真；后续真正需要训练的位置仍是用户服务器，必要性须由实际任务缺口决定。
