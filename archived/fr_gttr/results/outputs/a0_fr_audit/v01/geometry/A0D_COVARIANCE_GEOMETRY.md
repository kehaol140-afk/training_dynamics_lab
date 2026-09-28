# A0d Covariance Geometry Sufficiency Audit

A0D_DATA_INTEGRITY = PASS

E1_STREAM_KEYS = `['batch_manifest_hash', 'batch_manifest_path', 'batch_manifest_sha256', 'batch_seed', 'first_preclip_grad_norm', 's_r', 'stream_id', 'stream_seed']`
A0D_EXISTING_DATA_SUFFICIENT = NO

## Minimal replay if insufficient

- 复用原 v0.7 E1 16 streams/state 的 state、manifest、stream_seed；baseline-only exact replay。
- 在 first optimizer boundary、unscale 后 clip 前，仅 observation-only 获取每条 centered gradient 的低存储量统计。
- 为了精确 Sigma q：在线累计 vector sum_g 与 scalar s=<q,g> 加权 vector sum_sg；最终 Sigma q = E[(g-gbar)(s-sbar)]。可只保存最终 Sigma q vector/state，不保存每条 full gradient。
- 为了 PR(Sigma)：在线/分块计算 16x16 centered-gradient Gram matrix G_ab=<delta g_a,delta g_b>；只保存 Gram，不保存 full gradient tensors。
- 保存每条 ||g||^2、s_r、first clipping、manifest/hash；最终验证 replay model/optimizer hash 与原 v0.7 E1 baseline 对应（若原 E1 未保存 final hash，则需使用 deterministic branch audit/重新定义只读 replay integrity）。
- 由 Sigma q 与 q 计算 cos(Sigma q,q)、r_perp；由 centered Gram 估 tr Sigma 与 tr Sigma^2 / PR，并 bootstrap 16 streams uncertainty。

## Boundary

- 本轮不运行 GPU。
- 不能用仅有的 <q,g> 与 ||g|| 恢复 q^perp 中的 Sigma q。
- 不构造 5M×5M covariance；目标是在线累计 Sigma q 与 16×16 Gram。
