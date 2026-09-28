# `e0_pilot.json` 字段中文说明

JSON 本身不支持注释，所以配置 key 保留英文；本文件负责中文解释。

- `protocol_version`：实验协议版本。
- `stage`：当前实验阶段，E0 表示精确重放。
- `scientific_intervention_enabled`：是否允许科学干预；E0 必须为 false。
- `project_root`：项目根目录。
- `minimind_root`：MiniMind 基线源码目录。
- `checkpoint_path`：E0 起始 checkpoint；当前 TASK-00 前故意为 TODO。
- `device`：程序内部看到的 CUDA 设备。
- `physical_gpu_note`：当前机器如何通过 `CUDA_VISIBLE_DEVICES` 映射物理 GPU。
- `replay_steps`：E0 从同一状态向前重放多少 optimizer/micro steps，最终定义需由正式实现明确。
- `stream_id`：随机流编号；同一个 paired branch 必须相同。
- `require_same_batch_stream`：是否要求 batch 流一致，E0 必须 true。
- `require_same_rng`：是否要求 RNG 一致，E0 必须 true。
- `compare`：E0 需要比较的输出，包括 loss trace、模型 hash、优化器 hash 等。
- `pass_policy.prefer_bitwise_equal`：优先要求 bitwise 一致。
- `allow_tolerance_only_after_documented_nondeterminism_audit`：只有明确证明存在不可消除的非确定性后，才允许改用数值容差。
