# 纯功能联合更新核心：源码与完整任务证据

日期：2026-10-17，承接e4e1b68/c56bd4f。不训练，不写神经权重；后续真正训练使用用户服务器，本轮不启动。

- [整轮研究报告](../../docs/joint-update-core-2026-10-17.md)
- [冻结协议](protocol.md)／[接受语义](acceptance-semantics.md)／[信用与策略可改变性](learning-signal-and-scope.md)
- [完整核心](../../src/joint_update_core.py)／[本轮运行入口](../../scripts/run_joint_update_core.py)／[全部结果](round-results.json)
- [独立直接对照源码](../../scripts/compare_joint_core_direct.py)／[协议与报告](direct-comparison.md)
- [真实数值TO记录](direct-results.json)／[独立有限步骤核验](direct-finite-step-results.json)／[最终结果只读重对](final-independent-reconciliation.json)
- [包含首版开销的累计账本](execution-ledger.json)

主轮r=0→.1接受，完整时间5→4.8817418254；硬净空、模型冒配和三种UNKNOWN不被错误发布。同信息直接有限步骤相同；充分预算同族TO给4.1501904。没有神经训练、机器人结果或独立算法优势。

后端是已经声明的归一化解析任务，不是新机器人模型；PPO提案版本只是输入接口，本轮没有真实PPO提案生成、F/G拟合或checkpoint。全部候选为不可变函数参数值，输出证据不等于写入模型权重。
