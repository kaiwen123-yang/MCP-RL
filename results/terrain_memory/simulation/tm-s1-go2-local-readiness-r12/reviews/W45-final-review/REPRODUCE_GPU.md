# GPU 原件复算 CLI

需要 Python 3.11+、NumPy。两个脚本都只读取已发布原件，不联网、不运行仿真、不训练、不调用 GPU。

## G5 最终包

在任意输入目录放置以下两项：

- `w41-g5-20261110T033858Z-evidence-01.zip`：已有 [Release 资产](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-w41-gpu-evidence/w41-g5-20261110T033858Z-evidence-01.zip)。
- 本审查交付的 `g5-final-audit.json`：提供冻结的资产 size/digest、源成员路径及历史索引背景。

运行：

```bash
python audit_g5_final.py --audit-dir /path/to/g5-inputs
```

默认 `--audit-dir` 为脚本所在目录。可用 `--metadata /path/to/g5-final-audit.json` 单独指定元数据；没有显式参数时先在输入目录寻找，再在脚本目录寻找。脚本直接读取 ZIP，无需事先解包、完整 Release API 清单、旧 Git result/delivery 副本或仓库 checkout。

输出为输入目录的 `g5-final-recomputed.json` 和 `g5-source-hashes-recomputed.json`，保留原 `g5-final-audit.json` 不变。可通过 `--output` 改变主输出路径。Release 信息与旧 Git 入口差异来自冻结审查元数据；本次重新验证 ZIP/hash/manifest、原始数组、调用统计和结论，不冒充实时查询了远端状态。

## W45 四问题 GPU 支线

输入目录只需放置已发布的两份资产：

- [w45-gpu-originals-0001.zip](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-w45-gpu-part01/w45-gpu-originals-0001.zip)
- [w45-gpu-originals-0002.zip](https://github.com/kaiwen123-yang/MCP-RL/releases/download/tm-s1-go2-local-readiness-r12-w45-gpu-part01/w45-gpu-originals-0002.zip)

另外指定 MCP-RL 的本地 checkout。脚本读取该仓库 `results/terrain_memory/simulation/tm-s1-go2-local-readiness-r12/w45/` 内的 `phase.json`、`gpu.json`、`mechanism-evidence.json` 和 `delivery/gpu/sources.json`；这些审查输入对应提交 `3e3f69c440d6a33c14131563e5f485d88da3a803` 的 W45 数据。

```bash
python recompute_w45_gpu.py --repo /path/to/MCP-RL --audit /path/to/w45-inputs
```

`--repo` 默认当前工作目录；`--audit`（别名 `--audit-dir`）默认脚本所在目录。ZIP整体 size/SHA256 及每个源文件 size/SHA256 验证通过后，脚本自动在输入目录的 `originals/` 恢复缺失文件。已有文件若不匹配会失败，不覆盖。输出为 `recomputed.json`，可用 `--output` 改变路径。

脚本不需要 MuJoCo、Warp、Torch 或服务器凭据。不要用 `python -O` 运行：验证使用 assertions，脚本会拒绝关闭 assertions 的执行方式。
