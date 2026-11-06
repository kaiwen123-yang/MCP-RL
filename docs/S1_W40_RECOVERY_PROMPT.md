# W40 两条主线 QP 技术退出后的续接提示词

这是已有 W40 job 的故障续接入口。原始 [W40 完整启动提示词](S1_W40_START_PROMPT.md) 保留；不要再次执行其中的初次 `--new-phase`、W39 父断点或新建时钟步骤。

## 当前已确认事实

- A ep286 / B ep282 在 `applied_hold` 前因 `qp_kkt_failure` 技术停止；普通物理失败后的续采正常。两处失败已精确重放并在原容差下修复。两支各 12 workers，GPU 独立任务保持不动。
- A 当前 8643/33、tail50，恢复 `training/w40-a/PANN_PPO_S5-seed1/resume-state-ep00286.pt`。
- B 当前 8644/33、tail51，恢复 `training/w40-b/PANN_PPO_S5-seed2/resume-state-ep00282.pt`。
- 两支 phase/job 不变：`job-20261106T103028Z-w40-a`、`job-20261106T103028Z-w40-b`；excluded145 保持一次。
- 主线共同截止为北京时间 **2026-11-07 22:30:28.226467**；GPU 原截止为 **2026-11-07 22:31:28.128057**。任何 resume 不重置它们。
- 仓库修复产物是 `S5-r29-qp1`，保留原数学控制契约 `S5-r29`、独立冻结新二进制，绝不覆盖原 r29。当前审查没有连接 D57，不能把仓库修复当成服务器已经构建或恢复。
- 审查期间新提交 `03ea48d` 发布 GPU `completed_batch_layer / aligned=false`，交付记录为 `publication_interrupted`。这不等于完整 GPU MPC 已完成或可接入主线；真实进程状态仍须在 D57 只读核验。保留已完成计算，如原 finisher 已结束，允许仅补传缺失的既有数组，不重新启动计算或重置时钟。

## 复制以下提示词执行

```text
继续 MPC+RL 项目的 W40 故障修复与原 job 续采。用户要求两条主线恢复工作，独立 GPU 测试继续。先同步 GitHub main 并保留服务器未提交修改，读 AGENTS.md、docs/TERRAIN_MEMORY_PLAN.md 顶部当前交接、本文件及 results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/reviews/W40-stop-review/README.md、qp-replay.json。直接处理这次已定位的故障，不重新做泛泛规划、论文轮或物理准入实验。

在 autodl-d57:/root/autodl-tmp/MCP-RL，激活 source scripts/autodl_d57_env.sh。先只读检查精确 tmux 会话、对应 Python/agent PID、job/clock/checkpoint 和未提交修改。s1-w40-gpu 在用户提出本次要求时仍在进行，随后自行发布了batch层完成且aligned=false的结果：按实际进程续接原任务，保留会话/PID、work/gpu-w40-python 依赖、原版本、CPU206/207绑定、输出和 deadline1794061888.128057。不要对 s1-w40 做前缀匹配后 kill，不执行 kill-server/pkill，不因修复主线重启、升级或热换 GPU 后端。若GPU计算确已自然结束、原finisher也已结束，publication_interrupted只补传实际缺失的已生成数组，可复用现有finisher的跳过已存在文件逻辑，不重新跑计算或JIT；不能把部分批量层结果说成完整GPU MPC通过。GPU到原期限仍按原脚本正常结束。

两条主线原 job 都因真实 QP KKT 非收敛技术退出，详见原始四回合和精确失败输入重放。不要把检查改成只看 nfree>=0，不放宽 KKT/力矩/材料/跌倒阈值，不吞掉异常，不回退 r28。采用新 S5-r29-qp1 数值修复：原 native 解通过原证书时保持原输出；未通过时在同一个 H/c/边界上做可行活跃集 Newton，并用原容差再次核验。保持原 Go2/TM02、12维PD、2ms hold/十子步、r28候选排序、r27/r26、冻结PANN、658维观察、2Hzcap和W38奖励。

保留 work/r11-frozen/S5-r29 原样。若 S5-r29-qp1 尚未构建，按 scripts/prepare_s1_go2_r12_support.py --build S5-r29-qp1 做正常 agent_server 构建；若该冻结目录已存在，读取它的真实 identity/完成状态，不覆盖或重复创建。需要 build=S5-r29-qp1、based_on=S5-r29、controller_contract_build=S5-r29 的实际 identity；保存正常构建结果和版本。已提交的精确失败 QP 重放是这次修复证据，不另开纯 MPC/E30 smoke、参数网格或新物理资格门槛。

如两主线仍未运行，分别从自己的 W40 最新完整 resume 恢复：A为training/w40-a/PANN_PPO_S5-seed1/resume-state-ep00286.pt，8643/33、buffer50；B为training/w40-b/PANN_PPO_S5-seed2/resume-state-ep00282.pt，8644/33、buffer51。若服务器已有更新的同job断点或本次修复已经启动，以真实最新同job状态续接，不重复启动。使用 --job-dir、--resume、--controller-artifact S5-r29-qp1，分别命名 s1-w40-a 和 s1-w40-b，不用 --new-phase，不从 W39 父点重来，不清空tail，不重新seed，不合并optimizer/RNG。保留 phase、祖先fork、excluded145和累计目标9617/37。既有job保持12 MPC workers、Torch1/BLAS1。

主线两支继续共用原 deadline1794061828.2264671；若已经到期，禁止自行延长。恢复代码已处理旧的“无新update且无实际attempt”评价记录、旧完成标记、陈旧collection/snapshot和按前缀误匹配GPU会话的问题。只允许将从未启动评价的管理记录保留到原服务器job的小型resume-history后再等待真正训练完成；任何实际启动过的评价保留原时钟和结果。不得从父checkpoint伪造本轮新评价。

正常开始采样后确认A累计超过8643、B超过8644，记录各自真实新episode、artifact、tail和PID；若有更早技术故障，保留新的紧凑QP失败包并直接处理本次具体原因。普通body_fall、材料域退出按原规则继续采样，保留known_r25原停止谓词。各支首个新update34仍在累计8849，之后正常继续到9617/37或原时限；最终仅使用自身本phase最后真实新update做原E30，同case合格才E60，每支原评价预算≤4h。没有新update就明确未评价。

先推代码/配置/逐回合JSON、实际恢复状态、学习与结论，再让已请求的反馈/选中与参考轨迹/必要数组后台上传。保留当前诊断队列及其来源，Git短写使用既有锁并整合并发远端提交，不能强推或把队列启动当上传完成。原诊断期限不偷偷重置；新增诊断若需后续队列，准确标记未发布范围。大checkpoint/完整resume/全视频留D57由用户备份后手动删除，不自动删除，不在用户电脑归档。交接必须区分代码修复、数值验证、服务器构建、真实恢复采样、正式能力和独立GPU工程进度。
```

## 正常构建和两个原 job 的恢复命令

下列命令仅在核对对应主线尚未运行、原期限尚未到达且检查点仍是上述最新状态后执行。不要再次启动 GPU。

```bash
python scripts/prepare_s1_go2_r12_support.py --build S5-r29-qp1
```

```bash
python scripts/start_s1_go2_r12_tmux.py \
  --job-dir outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261106T103028Z-w40-a \
  --resume outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w40-a/PANN_PPO_S5-seed1/resume-state-ep00286.pt \
  --controller-artifact S5-r29-qp1 --session s1-w40-a
```

```bash
python scripts/start_s1_go2_r12_tmux.py \
  --job-dir outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/job-20261106T103028Z-w40-b \
  --resume outputs/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/training/w40-b/PANN_PPO_S5-seed2/resume-state-ep00282.pt \
  --controller-artifact S5-r29-qp1 --session s1-w40-b
```
