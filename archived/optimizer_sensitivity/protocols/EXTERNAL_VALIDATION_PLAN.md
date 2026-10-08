# External Validation Plan（设计稿，不运行）

目标：检验 F4A/A.1 phase 定律与 EARLY λ_SO 单调性是否超越当前 recipe/corpus。两候选：

## 方案 1：300M lightweight phase replication（~6×100M 算力/状态）
- 设计：1 个新架构档（hidden~1024 层 ~17 的合法 MiniMind 配置），2 seeds，steps {26,64,102}，同 D0 corpus 同协议训练；每 state 跑 F4-A 三臂（B/C/SO）+ cheap indicators。预计 ~6 state × (训练 ~1h + 度量 ~30min)。
- 回答：λ_SO EARLY 单调第 4 点；R_curv 非单调是否在更大 N 消失。风险：仍是同一 corpus/recipe——只验 scale 侧。

## 方案 2（推荐）：second recipe / second dataset replication（50M–150M，2 seeds）
- 设计：同一 MiniMind 架构族选 ~26M 与 ~100M 各 1 档，2 seeds；替换 (a) corpus（第二 4096 行子集或 tokenizer 兼容的第二个语料）与 (b) 至少一项训练超参（如 wd 或 warmup 变体，保持 AdamW+cosine+clip 语义）；steps {26,64,102}；同 F4-A 冻结度量管线 + F4B 风格 canary 分布等价 gate。
- 回答：phase 自愈定律与 seed 异质是否 recipe-invariant（机制论文 A 的外部效度核心）；顺带产出第二 recipe 下的 observability 复核（B 线种子数据）。
- 预算：4 states × 2 架构 × 训练 + 度量 ≈ 8 × (~20min 训练 + ~20min 度量) ≈ 半天 GPU。

## 优先级理由
机制链的 scale 侧已有 5M/26M/100M 三点；单一 recipe/corpus 是当前最大外部效度缺口 → 方案 2 优先；两者非互斥，人工可并行批准。
