# CHANGE_LOG — FR_G1C_EXTERNAL_1B_STRESS_TEST_V01

## §1 预注册与核验（2026-09-16，任何科学结果前）

- 建立版本化目录 `outputs/generalization_g1c_external_1b_v01/`；写入
  G1C_PROTOCOL.md（§1-§17 协议冻结）、experiment_config.json（判定标签
  与全部规则冻结）、EXTERNAL_MODEL_PROVENANCE.md、
  RESOURCE_FEASIBILITY.md、external_checkpoint_manifest.json、
  future_stream_manifest.json（NOT_CONSTRUCTED_BLOCKED）、smoke_report.json
  （NOT_RUN_BLOCKED）、results/ 五个文件（全部标注阻塞、零数据行）。
- B2 核验执行：HF API 逐 revision 文件列表检查 allenai/OLMo-1B
  （main + 9 抽样 revision / 336 branch）、OLMo-1B-hf、OLMo-1B-0724-hf、
  OLMo-1B-0724-954000steps-unsharded（main）、OLMo-2-0425-1B（main +
  3 抽样 branch / 268）、OLMoE-1B-7B-0924（MoE，架构排除）、OLMo-7B
  （1 抽样 / 559）；probe olmo-checkpoints.org = HTTP 404。
- 结果：**0 个 state 文件**（optim*/train*/scheduler*/rng*）；结论
  EXTERNAL_CHECKPOINT_UNAVAILABLE；在任何科学响应结果查看之前得出。
- 小证据文件 4 个下载并 SHA256 记录（screening_evidence/）；
  大文件**零下载**；GPU **零运行**。

## §2 异常 / 事实修正

- 任务书假设硬件"单卡 A100-40GB"，实测 2× RTX 5090 32GB
  （GPU0 共享余量 ~15.8GB）。登记于 RESOURCE_FEASIBILITY.md §1 与
  ENVIRONMENT.json；对本轮无影响（阻塞在 GPU 之前）。

## §3 冻结声明

未修改任何 A0A–A0G / A0H / 5M anchor / G1a / G1b / G2 / G3 冻结制品；
未重训练/微调任何 1B 模型；未启动第二个外部模型 / 7B / D2 /
A0G-lite / 新 optimizer / 算法实验。所有产物在本版本化目录内。
