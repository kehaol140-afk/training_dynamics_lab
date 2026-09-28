# E0 Smoke 首次通过记录（2026-09-02）

服务器首次真实 GPU smoke 输出：

```text
Model Params: 0.53M
Trainable Params: 0.533M
E0_MINIMIND_SMOKE_PASS
checks= {
  'metadata_equal': True,
  'batch_hashes_equal': True,
  'loss_trace_equal': True,
  'model_hash_equal': True,
  'optimizer_hash_equal': True
}
```

结论边界：这证明 v0.2 的 full-state + batch-lock 基础设施在当前 RTX 5090 环境下能够精确重放；不代表涨落—响应关系成立。
