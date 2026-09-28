# research/scripts 脚本说明（v0.2）

## `check_environment.py`
检查 Python / Torch / CUDA / GPU / 数据集 hash。只读，不训练。

## `check_scaffold.py`
检查实验框架目录是否完整。只读。

## `check_v02.py`
检查 v0.2 新增文件是否完整。只读。

## `task00_source_audit.py`
直接审计服务器当前 MiniMind 源码，输出 TASK-00 Markdown/JSON 报告。
**这是 v0.2 首先要跑的脚本。**

## `check_native_batch_stream.py`
验证 MiniMind 单 GPU非 DDP 的 `setup_seed -> randperm -> SkipBatchSampler`
是否可以重复构造同一 batch index 序列，并与 FR 显式 manifest 一致。

## `e0_preflight.py`
检查 TASK-00 是否已允许进入 E0 smoke。不运行训练。

## `e0_minimind_smoke.py`
真正执行一个很小的 MiniMind E0 重放 smoke：
- 使用真实 MiniMind 代码；
- 使用真实数据格式的小子集；
- warmup AdamW；
- 保存无损 full-state；
- 从同一点运行 A/B 两条无干预轨迹；
- 比较 loss/model/optimizer/batch hash。

这是 v0.2 最重要的运行脚本。

## `e0_replay_skeleton.py`
v0.1 留下的旧骨架，仅作历史参考。v0.2 运行时以 `e0_minimind_smoke.py` 为准。
