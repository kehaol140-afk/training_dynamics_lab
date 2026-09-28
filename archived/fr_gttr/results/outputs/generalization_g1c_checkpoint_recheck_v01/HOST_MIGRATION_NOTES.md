# HOST_MIGRATION_NOTES — 官方 host 与目录-vs-文件语义（2026-09-16）

## 1. 本轮实测的 host 事实（curl，HEAD + range GET）

| host | 用途 | 实测 |
|---|---|---|
| `olmo-checkpoints.org` | checkpoint 文件服务（官方 CSV 所列 URL 的 host） | **存活，直接服务文件，HTTP 200，redirect_count=0**（无 storage.googleapis.com 跳转） |
| `olmo-checkpoints.org/`（根） | — | HTTP 404（Cloudflare），**仅为对照记录，不作结论** |
| `…/stepXXXX-unsharded/`（目录 URL） | — | HTTP 404（Cloudflare），**仅为对照记录，不作结论**；与官方 docs "directory URLs cannot currently be directly accessed, but files within the directory are publicly accessible" **逐字一致** |
| `olmo-data.org` | Dolma 预处理语料与 eval 数据（stage1 YAML `paths:` 所列 .npy） | 本轮未探测（不在范围内）；登记为未来 B5 future-stream 的官方来源 |

## 2. 目录 404 vs 具体文件 200 —— 关键区分（本轮核验逻辑修正的核心）

- 同一秒级时间窗内：`https://olmo-checkpoints.org/ai2-llm/peteish1/step950000-unsharded/`
  → HTTP 404；而 `…/step950000-unsharded/config.yaml` → **HTTP 200**。
- 结论：**目录 URL 的 404 是对象存储网关的预期行为**（无前缀列表权限），
  与目录内对象的存在性无关。上一轮以 `olmo-checkpoints.org` 根 404
  作为负面证据登记，本轮明确修正该解读。

## 3. 替代 host 检查

- 全部 24 个 200 响应的 redirect chain 均为 0——**checkpoint 文件由
  olmo-checkpoints.org 直接服务，未发现迁移到其它 host**。
- 官方 docs（Checkpoints.md）、官方 CSV、stage1 YAML 均未声明 checkpoint
  文件迁移；历史 README/issue 提到的 `olmo-data.org` 仅承载数据而非
  checkpoint。**无官方替代 host 需要做对应文件级核验。**

## 4. 历史一致性说明（storage availability ≠ "训练时未保存 optimizer"）

- 本轮 4 个冻结 checkpoint 均为 **model + optimizer + trainer + config
  四件齐全**，不触发 partial-availability 分支。
- 但仍明确登记该区别（§7 要求）：即便在某些 step 出现
  "config/train.pt 可下载而 model/optim 不可下载" 的情形，也只说明
  **storage availability 问题**（对象存储逐对象生命周期），而**不是**
  "训练时没有保存 optimizer state"。OLMo standard checkpoint 的保存
  语义由 trainer 写盘逻辑决定（save_interval=1000，见 stage1 YAML 与
  config.yaml 快照），四件同目录原子发布。
- 上一轮 HF screening 的解读同步修正：HF model repos（OLMo-1B 等）
  是 **Transformers/HF OLMo 转换产物**的 publishing path，按
  Checkpoints.md 定义本就不含 training state；这**从不**等价于
  OLMo standard checkpoint 无 training state。
