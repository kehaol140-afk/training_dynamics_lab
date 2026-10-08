# MUON_PROTOCOL_v1.2 — M0.5 freeze patch

Date frozen: 2026-09-26. Nature: **amendment patch over `MUON_PROTOCOL_v1.1.md`** — resolves the
v1.1 §8 unresolved-fields list using M0 evidence (`outputs/m0/M0_REPORT.md`). v1.1 text remains
frozen and in force except where superseded below. Any later change requires a dated v1.3 section.

**Status: FROZEN. No M1/M2 training starts before the pre-launch checklist
(`docs/M1_PRELAUNCH_CHECKLIST.md`) is signed off.**

---

## 1. Model freeze (supersedes v1.1 §1 "pending review" rows)

MiniMind 30M dense decoder-only Transformer. **Model changes are prohibited for the entire
experiment (M1/M2/M3).**

| Field | Frozen value |
|---|---|
| family / file | MiniMind, `minimind_base/model/model_minimind.py` (SHA256 `47f124f5…`, read-only) |
| **total parameters** | **30,025,216** (verified, `outputs/m0/routing_report.json`) |
| hidden_size | 512 |
| num_hidden_layers | 8 |
| attention | 8 heads, GQA with 4 KV heads, head_dim 64, RoPE (theta 1e6), QK-RMSNorm |
| FFN | SwiGLU, intermediate_size 1664 (= `ceil(512π/64)·64`) |
| vocab / tokenizer | 6400, MiniMind BPE (`tokenizer.json`, SHA256 `71f32c68…`) |
| **tied embedding/lm_head** | **True** — one tensor, registered once in the AdamW route |
| sequence length | 256 (all training and eval windows) |
| dtype | fp32 master weights; bf16 autocast compute; **no GradScaler** (both arms, identical loop) |
| initialization | MiniMind default `post_init()`, per-run seed-derived |
| MoE / dropout | disabled / 0.0 |

## 2. Muon configuration freeze (supersedes v1.1 §3.2 pending items)

| Field | Frozen value |
|---|---|
| implementation | **`torch.optim._muon.Muon`** (imported as `torch.optim.Muon`), torch==2.11.0 wheel, file SHA256 `38e95918424bd26604f786195a4aa72afb60674b4ac9ccbde777974013cb1f52`; upstream permalink KellerJordan/Muon@`f90a42b` (in-file). Full audit: `docs/MUON_IMPLEMENTATION_AUDIT.md` |
| provenance rule | torch upgrade/patch **invalidates this freeze** → re-audit required |
| momentum | 0.95 |
| Nesterov | True (`u = (1−μ)g + μB`, then orthogonalize) |
| Newton-Schulz | 5 steps, quintic coefficients (3.4445, −4.7750, 2.0315), eps 1e-7, bf16 internal |
| scaling rule | `adjust_lr_fn=None` → "original": `lr_adj = lr·sqrt(max(1, A/B))` |
| **reference LR** | **0.005** (set from M0 stability screen: 0.02 unstable, 0.001–0.005 stable at micro-scale; F3) |
| LR search | multiplier {0.5, 1, 2} → **{0.0025, 0.005, 0.01}** |
| weight decay | from shared grid (§3), decoupled, applied to all Muon-route params (`θ ← θ(1−lr·λ)`, unadjusted lr) |
| routing | hidden 2D linear matrices only: 56 tensors, 26,738,688 params; aux route (AdamW): embedding(tied)+norms, 34 tensors, 3,286,528 params; overlap 0 / missing 0 (Gate 2) |
| aux LR rule | aux AdamW uses the same reference LR, same multiplier, same schedule as the Muon route |

## 3. AdamW configuration freeze

| Field | Frozen value |
|---|---|
| implementation | `torch.optim.AdamW`, single group (all params), torch==2.11.0 |
| **reference LR** | **3e-4** |
| LR search | multiplier {0.5, 1, 2} → **{1.5e-4, 3e-4, 6e-4}** |
| betas / eps | (0.9, 0.95) / 1e-8 |
| **WD grid (both arms)** | **{0.01, 0.02}**, decoupled, **applied uniformly to every trainable parameter — embedding and norms included, no exemptions** (resolves v1.1 §3.1 open item) |

Per-optimizer reference LRs are frozen and **asymmetric by design**; each optimizer searches its
own 3×2 grid of equal size (6 cells per optimizer per mixture). **Modifying any LR/WD value or
grid point after observing M1/M2 results is prohibited** (red line; v1.1 §5 single
boundary-extension contingency remains the only exception, logged as amendment).

## 4. Budget freeze

| Field | Frozen value |
|---|---|
| **B0** | **67,108,864 tokens = 2^26** per run, identical for every run of M1/M2/M3 |
| optimizer steps | 2,048 |
| tokens per optimizer step | 32,768 |
| micro-batch | 32 sequences × 256 tokens = 8,192 tokens |
| gradient accumulation | 4 micro-steps per optimizer step |
| effective loss tokens | per window: 255 (BOS + 255 stream tokens; targets = window shifted by one) |
| epoch policy | single-epoch at every p ∈ {0.25, 0.5, 0.75} (A train 98.19M, B train 75.44M ≥ 0.75·B0 = 50.33M ✓) |
| schedule | linear warmup 32 steps (`s=t/32`), then cosine to 10% of peak (`s = 0.1 + 0.45(1+cos(π(t−32)/2016))`), identical factor for both arms and both routes |
| grad clip | global norm 1.0 over all trainable params, both arms, before optimizer step |

All values live in `src/muon_transition/config.py` and are emitted verbatim into
`configs/m1_baseline_prereg.json` (generator `scripts/gen_m1_config.py`; config.py SHA256 embedded).

## 5. Dataset provenance — see `docs/DATASET_CARD.md`

Frozen: Domain A = EN `common_en_70k.jsonl` (post-dedup 60,861 docs / 104,303,425 tokens), Domain B
= ZH `common_zh_70k.jsonl` (74,591 docs / 80,091,588 tokens), both SHA256-registered; exact-dedup
keep-first; thread-aligned 94/3/3 train/dev/confirm split (seed 777, manifest
`outputs/m0/data_audit/split_manifest.json`, integrity verified: no overlap, full coverage, no
thread/pair straddling — **no leakage by construction and by check**); eval pools = first 512
windows per domain per dev/confirm split (130,560 tokens each, equal across domains).

## 6. Observation hooks freeze (diagnostic only)

Implemented in `src/muon_transition/observation_hooks.py`; attached to the training loop for
M1/M2; **never a primary or selection metric** (v1.1 §7 stands).

- Cadence: every **100 optimizer steps** (and at the final step 2048); JSONL to
  `outputs/observations/<run_id>.jsonl`.
- Required fields per record: `domain_gradient_cosine`, `domain_conflict_ratio`,
  `grad_global_norm`, `update_norm`, `update_work_proxy = −gᵀΔθ`.
- Probe definition (frozen): first 4 batches × 8 windows per domain from the frozen dev pools;
  cosine = mean over 16 pairwise values; conflict ratio = fraction of the 16 pairwise dots < 0.
- Optional additional fields already implemented: weight norm, update/weight ratio, sampled
  gradient singular values.
- **Naming rule: the quantity −gᵀΔθ is named `update_work_proxy`; the word "energy" is banned
  in all project outputs.**

## 7. Resolved / carried-forward status

| v1.1 §8 item | Status under v1.2 |
|---|---|
| 1 model config | **frozen** (§1) |
| 2 datasets + provenance | **frozen** (§5, DATASET_CARD.md; upstream second-hand origin recorded honestly) |
| 3 tokenizer stats | **frozen** (coverage A 78.3% / B 98.1%) |
| 4 token budget | **frozen** (§4) |
| 5 Muon version | **frozen** (§2; resolved at M0) |
| 6 reference LRs / WD | **frozen** (§2–3) |
| 7 split proportions / pool sizes | **frozen** (94/3/3; 512 windows) |

M1 remains as v1.1 §6: p=0.5, 12 runs, seed 2030, gate = both optimizers train normally;
selection on dev V_Q only. M2/M3 unchanged.
