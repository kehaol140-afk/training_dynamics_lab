# E0 MiniMind Smoke 报告

- 总判定：**PASS**
- device：`cuda:0`
- dtype：`bfloat16`
- batch manifest：`66f4348d8565078cc6c935fe5b2c7efdaa50a84d3c6d213525fc719788e6ec14`
- 人工扰动：**关闭**

## 配对检查

- metadata_equal: **True**
- batch_hashes_equal: **True**
- loss_trace_equal: **True**
- model_hash_equal: **True**
- optimizer_hash_equal: **True**

## 边界

本结果只说明当前机器上 FR 完整状态 + 锁定 batch stream 的 smoke 能否重放。
它不是 B→χ 验证，也不是涨落—响应关系成立的证据。
