# Go2公开资产静态审计：两份可冻结权重，实机和任务能力仍待验证

日期：2026-10-17（Asia/Shanghai）。已读当前接续入口、[联合核心独立复审](../../docs/joint-update-core-independent-review-2026-10-17.md)及[狗侧物理合同](../physical-task-instance-2026-10-17/robot-task-contracts.md)。该解析例已经收尾；本页只核真实公开资产，不再增加解析反馈例子。

**现有可交给后续独立检查的资产是两份真实、较小的Go2导出策略及其部署配置；尚不是本机可部署策略，也不是狗侧“受扰—转弯/绕行—过门—停靠”任务的已验证控制器。**两个保留候选都来自 `fan-ziqi/rl_sar`，不是Unitree官方发布的Go2权重，也不是HIM原论文作者发布的Go2 checkpoint。用户报告Go2接口已调通，不能代替本机控制模式、PD、频率、延迟和限幅的测量。

本页的**静态子审计**使用已认证 `gh` 查询和HTTPS clone；初次默认SSH clone因本机公钥认证失败，改用显式HTTPS成功，未修改全局Git/认证配置。所有外部文件仅在 `work/platform-assets-2026-10-17/go2/`。该子审计只读文本、Git树和ZIP内容/文件hash，**没有运行任何外部仓库脚本、导入模型、模型推理、安装依赖、环境rollout、训练或硬件操作**。这不是整轮预算：根代理随后已经完成两份权重的固定合成输入核验，共12次CPU策略前向，见[独立记录](go2-fixed-input-results.json)及本页第6节；不能再称整轮“0推理”。

结构化版本、文件hash与证据位置见 [go2-sources.json](go2-sources.json)。

## 1. 优先来源实际具备什么

| 已核来源及固定commit | 当前可见资产 | 本轮判断 |
|---|---|---|
| [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym/tree/276801e46c5d433564f24658bac64f254b7d2d4b) | Go2训练配置、URDF；树内预训练仅G1/H1/H1_2；release API为空 | 有Go2训练代码，不能称已有Go2权重 |
| [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3) | Go2训练/部署代码；Go2部署指向logs；树内ONNX为G1，release为空 | 训练与部署接口资源；未取得对应Go2导出策略 |
| [Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab/tree/1425b15f73bd4095f0df53709d7c389c3eb9e790) | Go2模型、速度任务、deploy.yaml；树内Go2目录无policy.onnx/pt，release为空 | 有名义模型与配置，未取得对应Go2权重 |
| [HIM作者原仓库](https://github.com/InternRobotics/HIMLoco/tree/ef289acaa62795009363b7b819c9186690630441) | A1/Go1/Aliengo配置、HIM训练/导出源码；当前树无Go2配置和pt/onnx，release为空 | 方法与源码来源，不能把其他机型或论文视频说成Go2 checkpoint |
| [Project-Robo/HIMLOCO_GO2](https://github.com/Project-Robo/HIMLOCO_GO2/tree/2e70bcf66d2a479add5a688ed4d78d863d166c11) | Go2适配代码及约0.99MB的真实pt；README说明阶段性快照，无历史；未发现顶层LICENSE | 筛查后不保留为第三候选；不是已核HIM论文作者发布，训练/许可来源不足 |
| [rl_sar](https://github.com/fan-ziqi/rl_sar/tree/376d42c9b128f963ab08579762d5a216a976ce39) | 两份Go2权重、各自配置、C++观测/历史/动作与硬件接口 | 本轮只保留以下两个资产候选 |

这里的“无权重”严格限于已核commit的完整树、公开release和已读README，不声称网上不存在其他文件。未把非官方fork的“官方checkpoint”自称或搜索摘要作为官方身份依据。HIM身份由[作者项目页](https://junfenglong.com/HIMLoco/)指向原仓库核对；作者项目页的抗扰展示不是本候选的性能证据。

官方配置也不能混用：RL Gym Go2的继承配置为48维观测、PD20/0.5、动作scale .25、5ms仿真×4 decimation；RL Lab的Go2 actor配置为45维、gyro scale .2、无启用history；RL Mjlab的Go2 deploy配置另有2维gait phase，共47维及不同PD。这些不是下面两份导出策略的可替换配置。[Gym配置](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/legged_gym/envs/go2/go2_config.py#L23-L31)、[Lab观测](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py#L219-L237)、[Mjlab部署配置](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml)

## 2. 两个保留候选及权重真实性

共同仓库固定为 `fan-ziqi/rl_sar@376d42c9b128f963ab08579762d5a216a976ce39`。每份均实际取得二进制、核SHA256并只读ZIP目录和内嵌TorchScript源码；均不是Git LFS文本指针。**静态子审计**没有调用 `torch.load`、`torch.jit.load` 或 `.forward()`；根代理另行完成的加载/固定输入前向见第6节。

| 项目 | robot_lab导出策略 | HIM适配导出策略 |
|---|---|---|
| 仓库路径 | `policy/go2/robot_lab/policy.pt` | `policy/go2/himloco/himloco.pt` |
| 字节数 | 769,376 | 991,615 |
| SHA256 | `9f14cb95e74ac9e5e30954da0fc0c33eaa41e337852ebed19fcee869279ade0b` | `d9945f85a39944aaf241a67f7d2b25a15326a0222d0e503aeb05a7f188430509` |
| ZIP成员数 | 29 | 41 |
| 静态结构 | exporter→Identity normalizer→45–512–256–128–12 MLP，隐藏ELU | 270–128–64–19 estimator；分3维velocity与16维归一化latent；拼当前45后进入64–512–256–128–12 MLP，隐藏ELU |
| 发布者标注来源 | robot_lab / IsaacSim | himloco / IsaacGym |
| 已核任务域 | Go2速度命令运动部署资产 | Go2速度命令运动部署资产 |
| 尚缺 | 原训练run、seed、原始checkpoint、精确环境版本/地形/随机化与评估记录的文件级对应 | 同左；另有原HIM到Go2适配及该权重训练来源链缺口 |

权重稳定链接：[robot_lab policy](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/policy/go2/robot_lab/policy.pt)、[HIM policy](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/policy/go2/himloco/himloco.pt)。本地前缀为 `work/platform-assets-2026-10-17/go2/rl_sar/`。它们是导出推理资产，不能当成已核可续训PPO checkpoint；目前没有配套优化器、critic、训练数据和版本证据。

静态结构来源是ZIP内 `policy/code/__torch__/.../exporter.py` 或 `legged_gym/utils/helpers.py`，及各 `torch/nn/modules/linear*.py` 的 `in_features/out_features` 常量。HIM导出forward确实调用estimator、拆分3+16并归一化latent，然后调用actor；仅加载actor子模块不能复现该冻结反馈。两个候选都没有把未来门/停车目标或外部力真值作为已声明输入。

## 3. 必须冻结的实际观测—命令—动作链

以下全部是**该commit的部署代码配置**，不是本机实测性能或厂家认证限额。静态来源：[robot_lab config 4–55](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/policy/go2/robot_lab/config.yaml#L4-L55)、[HIM config 4–55](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/policy/go2/himloco/config.yaml#L4-L55)。

| 字段 | robot_lab | HIM适配 |
|---|---|---|
| 单帧顺序及维数 | gyro3, gravity3, commands3, q-relative12, dq12, previous-action12 =45 | commands3, gyro3, gravity3, q-relative12, dq12, previous-action12 =45 |
| 历史 | 无显式多帧history；previous action仍是执行记忆 | `[0,1,2,3,4,5]`，当前到5拍前，time-major，共270 |
| gyro/q/dq比例 | .25 / 1 / .05 | .25 / 1 / .05 |
| 命令比例 | `[1,1,1]` | `[2,2,.25]` |
| 观测/输出裁剪 | 观测每分量±100；输出每分量±100 | 同左 |
| 关节输出 | 12个位置增量；hip .125，其余 .25，叠加默认角 | 同左 |
| RL PD | kp20，kd.5，每关节相同 | kp40，kd1，每关节相同 |
| 默认角，按策略顺序 | 每腿 `[0,.8,−1.5]` | FL `[.1,.8,−1.5]`，FR `[−.1,.8,−1.5]`，RL `[.1,1,−1.5]`，RR `[−.1,1,−1.5]` |
| 策略关节顺序 | FR, FL, RR, RL，每腿hip/thigh/calf | FL, FR, RL, RR，每腿hip/thigh/calf |
| 策略索引→SDK索引 | `[0,1,2,3,4,5,6,7,8,9,10,11]` | `[3,4,5,0,1,2,9,10,11,6,7,8]` |

### 3.1 观测和历史具有可定位实现

- [ComputeObservation 64–112](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/library/core/rl_sdk/rl_sdk.cpp#L64-L112)按配置顺序拼接，重力由四元数逆旋转，q减默认角，dq缩放；[180–197](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/library/core/rl_sdk/rl_sdk.cpp#L180-L197)包含观测clip与初值。
- [真实状态读取144–161](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/src/rl_real_go2.cpp#L144-L161)使用机身IMU gyro、wxyz四元数和映射后的关节q/dq。`tau_est`虽读取，两个候选的actor观测列表都未包含它；`lin_vel_scale`虽在yaml存在，也不代表robot_lab输入实测线速度。
- [history构造239–244](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/library/core/rl_sdk/rl_sdk.cpp#L239-L244)、[buffer 43–51及79–98](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/library/core/observation_buffer/observation_buffer.cpp#L43-L98)、[134–148](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/library/core/observation_buffer/observation_buffer.cpp#L134-L148)分别说明零初始化、历史移位和当前到过去的拼接。HIM按20ms目标间隔对应当前及约100ms前的六帧，不等于传感器实际延迟已知。

冻结反馈必须包含这些预处理、previous action、历史初始化/更新、HIM estimator+actor、后处理、PD、关节映射与版本。只更新HIM估计器也会改变闭环，不能仍称原冻结控制器；若改变估计器，应与“策略反馈学习”分别归因。

### 3.2 命令权限足以给强MPC真实选择，但范围待核

代码输入是三维速度命令，不是这里人为固定的一个分配参数。[RunModel 212–235](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/src/rl_real_go2.cpp#L212-L235)在普通模式使用手柄量，在启用ROS且navigation模式时读取 `/cmd_vel` 的 `linear.x/linear.y/angular.z`。这提供未来MPC接入的具体位置；当前没有给这两个资产接入本项目MPC，也没有实测其命令响应。

同一命令还要经过各策略不同的 `commands_scale`。该段代码没有把 `/cmd_vel` 物理范围自动限制成已验证训练域；不能把yaml缩放数字说成所有速度都能可靠跟踪。需要核实实际训练/工作范围、速度变化率、逆向/侧向/转向和零命令停稳响应。基线必须允许原接口支持的减速、等待、转向、侧移与绕行；若这些已解决通道接续，不再主张缺新反馈能力。

### 3.3 频率和执行器限制不能只读一行yaml

[base.yaml 5–6](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/policy/go2/base.yaml#L5-L6)为 `dt=.005,decimation=4`，[真实循环76–81](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/src/rl_real_go2.cpp#L76-L81)分别将控制发布和策略推理设为5ms/20ms目标。它们是200Hz/50Hz调度目标，非本机实测稳定频率、延迟或deadline保证。推理锁忙时[Forward 262–268](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/src/rl_real_go2.cpp#L262-L268)返回上一动作；队列、丢帧/超期和传感时序均需进执行模型。

动作先经输出clip，再经scale+默认角形成目标，[ComputeOutput 256–270](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/library/core/rl_sdk/rl_sdk.cpp#L256-L270)还计算并夹紧一份PD估计力矩。**但真实RLControl只消费位置/速度队列，并发kp/kd、feedforward tau=0**：[589–607](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/library/core/rl_sdk/rl_sdk.cpp#L589-L607)，再由[SetCommand 183–194](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/src/rl_real_go2.cpp#L183-L194)发送。

所以配置中的robot_lab `23.5`、HIM `33.5` **不是本审计已经证实的实机硬扭矩限制**。`TorqueProtect`/`AttitudeProtect`在当前[RunModel 250–251](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/src/rl_real_go2.cpp#L250-L251)还是注释调用。不能在后续理论中以这些数字认证输入球/制动距离。需要本机电机模式、固件饱和与日志，而不是再选一个方便的力限值。

此外，[Go2 FSM 159–169](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/fsm_robot/fsm_go2.hpp#L159-L169)默认硬编码 `himloco`；robot_lab文件存在不等于这个入口已经选择它。真实程序构造器还会建立DDS并尝试释放厂商运动服务；本轮未调用该入口，后续静态/固定输入检查也应与硬件入口分开。

## 4. 许可、依赖和模型来源能确认到哪一步

`rl_sar`顶层[LICENSE](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/LICENSE)和两份配置标Apache-2.0。HIM作者原仓库[LICENSE](https://github.com/InternRobotics/HIMLoco/blob/ef289acaa62795009363b7b819c9186690630441/LICENSE)标CC BY-NC-SA 4.0。当前未找到两份二进制各自的model card、训练归属声明及完整上游许可链；不能仅凭外层仓库许可推断所有继承权利已逐项核清。本轮公开提交只保存来源/hash/原创审计，不提交权重。

部署仓库声明Ubuntu20.04/22.04、C++、可选ROS1/ROS2及LibTorch/ONNX后端；本次pt需要TorchScript加载路径。下载脚本的默认LibTorch为2.3.0、ONNX Runtime为1.22.0，SDK2 gitlink固定到 `65e19c625b76b697f94e0b953e296ecc0ce7b4a1`；未初始化/编译该子模块。源定位为[README依赖段](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/README.md#L69-L91)、[运行库下载脚本47–51](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/scripts/download_inference_runtime.sh#L47-L51)、[CMake SDK2路径](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/src/rl_sar/CMakeLists.txt#L289-L292)。读取这些信息不代表本机已经满足/安装依赖。

`rl_sar`没有把Go2完整几何与策略训练环境一起锁在该树内；其[模型下载脚本31–32](https://github.com/fan-ziqi/rl_sar/blob/376d42c9b128f963ab08579762d5a216a976ce39/scripts/download_robot_descriptions.sh#L31-L32)指向 `fan-ziqi/rl_sar_zoo` 的可变 `main`。本次没有运行该脚本，也未声称这个可变模型与当前权重配套。

可用的官方名义几何包括Gym的 [go2.urdf](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/resources/robots/go2/urdf/go2.urdf) 和Mjlab的 [go2.xml](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/assets/robots/unitree_go2/xmls/go2.xml)。它们能作为核对关节/名义几何的来源，**不能直接称为这两份策略及用户机器的准确执行模型**。还缺实物版本、碰撞/足端包络、载荷、摩擦、伺服/通讯时延与约束标定，以及配套轨迹数据。

## 5. 给冻结策略准确模型和充分MPC后，究竟还不知道什么

当前只知道有速度跟踪反馈和合法命令入口，**不知道它们经过正确建模与规划后在本任务仍有什么差距**。应把下列问题作为待判别事实，而非写成基线已经失败：

1. 侧向脉冲后，通过改变三速度命令、提前减速、宽路/等待，冻结整条反馈能否达到门入口和停车条件？输入、姿态、接触及历史的响应要一起评价；不能沿用解析横向优先的固定前段。
2. 步态/接触、previous action与HIM历史相同还是不同，是否改变转向、制动和足端扫掠？只用当前位置/速度拟合一个无记忆模型可能有混淆，但本轮未采数据证明误差大小，更未证明必须学习新策略。
3. 同一完整任务下，准确但昂贵的直接闭环模型与较小的响应模型，预测可靠性—计算量如何取舍？若强MPC用准确模型已达到所需取舍，就停止该实例的反馈增量主张；若差距只是模型失配或选择器错误，先归因于模型/规划。
4. 若确有任务驱动反馈更新，HIM估计器、actor、PD或信息入口改变各占什么作用？同信息直接鲁棒PO/策略受限TO也取得模型、导数和任务资料后，是否仍有有限数据/计算差距？当前没有证据回答肯定。

在实物几何、观测时序、可用输入和当前冻结策略下，先定义强基线可用的命令域、等待和路线；再由数据/可核模型判断“已有能力调用”“模型改进”“反馈改变”分别有何贡献。无需证明所有固定策略全局无解，也不能因尚无本机数据就先训练寻找优势。

本页的最终状态为：**2个真实公开推理资产已静态冻结，另通过固定合成输入的加载与输出检查；本机部署未核、完整多阶段任务性能未测、配套准确响应模型未建立、强基线剩余差距未知。**静态子审计的外部源码执行/模型调用/环境步/训练/权重更新/硬件操作均0；整轮模型前向另计12次，不能混成0。

## 6. 根代理固定输入记录的只读复核

已只读核对[本地检查脚本](../../scripts/inspect_public_go2_assets.py)与[保存结果](go2-fixed-input-results.json)，没有重新加载模型或重复前向。保存脚本hash与当前文件一致。记录为两个模型，每个使用零命令、前向命令 `.2`、偏航命令 `.2` 三种合成输入，各重复一次：**12次完整policy前向，内部含12次actor和6次HIM estimator前向**。后两项是内部组件计数，不是在12次以外又运行18次完整策略。

拼接与缩放在这组输入上和前述C++定义一致：robot_lab命令落在单帧索引6–8，前向/偏航分量分别为 `.2/.2`；HIM命令在索引0–2，经比例得到 `.4/.05`，随后按time-major重复六帧。gravity为 `[0,0,−1]`，q-relative、gyro、dq和previous action为0。q-relative为0意味着假设每个策略自己的默认关节角，不是实际关节角全零。

输出后处理也一致：先按各配置裁剪动作，再做 `q_default+action_scale*action`；`sdk_target[joint_mapping[i]]=target[i]` 与真实SetCommand的策略索引到SDK索引方向相同。对保存的全部6条输入/输出数组，用NumPy float32只做观测和后处理复算，观测、policy-order目标角及SDK-order目标角最大差均为0。这些是保存数组算术，新增模型加载、前向、环境步和硬件指令均0。

结果记录中的六条输出均为12维有限数，重复输入的两次前向精确相同，权重前后hash不变。这里的重复一致性来自已保存执行记录及其脚本检查，本只读复核没有再做第二次独立模型运行。

适用范围保持很窄：六帧相同并不检验真实history移位、启动零填充或非平稳时序，且不保证是可达机器人历史；零gyro/q-relative/dq也没有数值激励它们的非零缩放；本组输出没有触及±100动作clip。C++公式的静态一致性与这些未覆盖行为分开。该结果只说明指定文件在记录的CPU/PyTorch环境能接受这些合成输入并按配置解释输出，不证明真实反馈、力矩限制、行走稳定、完成任务或训练价值。无需为本次资产审计再扩输入扫查。
