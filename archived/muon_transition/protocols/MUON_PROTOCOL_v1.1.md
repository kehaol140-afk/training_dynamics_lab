# MUON_PROTOCOL_v1.1 — Optimizer-Dependent Data Mixture Interaction (Screening)

Status: **FROZEN CANDIDATE — awaiting protocol review. No GPU training before review sign-off.**
Date frozen: 2026-09-26
Project: `muon_transition_experiment/` (independent of `legacy_fluctuation_response/`, `optimizer_sensitivity/`, `continuation_adaptation/`; those directories are untouched and read-only for this project).

Version: v1.1. Changes after freeze require a dated v1.2 amendment section; the frozen text is never edited in place.

---

## 0. Scope and Non-Goals (frozen)

**Research question:**

> Under identical model, token budget, data domains, and a fair (equal-size) hyperparameter search:
> Does changing the optimizer from AdamW to Muon change the optimal training data mixture?
> Formally: test whether `p*_AdamW != p*_Muon`, where `p` is the training token mixture ratio
> between two domains (fraction of training tokens drawn from Domain A).

This is a **screening experiment**.

Non-goals (frozen, binding):

- NOT a new optimizer project. No optimizer design, no modification of Muon or AdamW update rules.
- NOT a new training theory project. No mechanism claims beyond what the mixture comparison supports.
- NOT an adaptive controller. No data selection/mixture adaptation during training. Mixture `p` is fixed per run.
- We do NOT assume Muon is better. **Negative results (no interaction, `p*` unchanged) are valid outcomes.**

---

## 1. Model (fields frozen at M0; proposal below)

One dense decoder-only Transformer for the entire experiment. Selected family: **MiniMind**
(`minimind_base/model/model_minimind.py`, SHA256 in PROVENANCE.md) — dense decoder-only, RMSNorm,
RoPE, QK-Norm, GQA, SwiGLU, **tied embedding/lm_head**.

Proposed configuration (target 20–40M parameters; exact count verified at M0):

| Field | Proposed value | Status |
|---|---|---|
| hidden_size | 512 | pending review |
| num_hidden_layers | 8 | pending review |
| num_attention_heads | 8 | pending review |
| num_key_value_heads | 4 (GQA) | pending review |
| head_dim | 64 (= 512/8) | pending review |
| intermediate_size | 1664 (MiniMind default formula `ceil(h·π/64)·64`) | pending review |
| vocab_size | 6400 (MiniMind BPE tokenizer) | pending review |
| use_moe | False | frozen |
| tie_word_embeddings | True (must be handled explicitly, see §4) | frozen |
| Measured parameter count | **30,025,216 ≈ 30.03M** (unique tensors; tied embed/head counted once — verified: `embed_tokens.weight is lm_head.weight == True`; embed = 3,276,800) | verify at M0 |
| Sequence length | 256 tokens (fixed for all runs) | pending review |
| dtype — parameters | fp32 master weights | frozen proposal |
| dtype — compute | bf16 autocast (both optimizers, identical loop; no GradScaler) | pending review |
| Initialization | MiniMind default `post_init()` (PyTorch defaults, no custom init); per-run seed-derived | frozen proposal |

Record at M0 into `outputs/m0/model_card.json`: full config dict, exact parameter count
(per-tensor, Muon-routed vs AdamW-routed), init RNG seed and init checksum.

**Do not run large experiments before M0 passes.**

---

## 2. Data (fields frozen at M0; candidates registered in PROVENANCE.md)

### 2.1 Two domains

| Field | Primary candidate | Status |
|---|---|---|
| Domain axis | **Language**: English vs Chinese, content-matched (translation-paired corpora) | pending review |
| Domain A (EN) | `sharegpt_datasets/common_en_70k.jsonl` (79,107 conversations) | pending token stats + dedup |
| Domain B (ZH) | `sharegpt_datasets/common_zh_70k.jsonl` (76,399 conversations) | pending token stats + dedup |
| Backup axis | Topic: `computer_*` vs `common_*` (same sharegpt family) | registered only |
| Preprocessing | Flatten `conversation[]` turns to plain text (`H: ...\nA: ...\n` template, exact template frozen at M0), tokenize with MiniMind BPE, concat-to-sequence-length packing, no cross-domain mixing within a packed sequence | pending review |

Rationale: translation-paired EN/ZH maximizes content match between domains, so mixture preference
reflects language-specific optimization, not topic confounds. **Unresolved**: original upstream
provenance of these files (inherited from another local project, no network access to re-download;
see PROVENANCE.md §4) — confirm or replace at M0.

### 2.2 Required records (before any training)

- Source path + SHA256 per file (registered in PROVENANCE.md; re-verified at M0).
- Preprocessing script + its config, hashed.
- Tokenizer statistics per domain: token count, unique tokens, fertility (chars/token),
  OOV/unk rate, sequence-length histogram after packing.
- **Split discipline (per domain, disjoint)**: `train / dev / confirm` — partition by *conversation*
  (translation pairs split by identical index so a content pair never straddles splits), using the
  legacy `pilot_partition.py` permutation+SHA pattern; proportions frozen at M0 (proposal: 94% / 3% / 3%).
- **Duplicate checking**: exact-duplicate and near-duplicate check within each domain (normalized-text
  hash); cross-domain overlap is *by design* (translation pairs) and is reported, not removed;
  dev/confirm must be duplicate-checked against train within-domain.

### 2.3 Training mixture (frozen)

- `p ∈ {0.25, 0.50, 0.75}` = fraction of **training tokens** from Domain A (nominal).
- Mixing granularity: token-level accounting. Per run, the data plan is materialized in advance as an
  explicit batch manifest (legacy `batch_manifest.py` pattern): every batch draws from domains so that
  cumulative realized `p_hat` matches nominal `p` within ±0.01 at every decile of the run; realized
  `p_hat` is logged and reported. No within-sequence cross-domain packing.
- Identical total token budget for every run (budget value itself: **unresolved**, see §8).

### 2.4 Evaluation mixture (frozen)

Fixed `q = 0.5` for all evaluation and all selection:

```
V_Q = 0.5 · V_A + 0.5 · V_B
```

where `V_A`, `V_B` are per-domain dev (M1/M2) or confirm (M3) token NLLs computed on fixed,
seed-frozen eval pools. `q` never changes across arms, stages, or optimizers.

---

## 3. Optimizers (frozen)

### 3.1 AdamW arm

| Field | Value | Status |
|---|---|---|
| Implementation | `torch.optim.AdamW` (torch 2.11.0, env `fr_minimind`), single param group covering all params | frozen |
| beta1 | 0.9 | frozen proposal |
| beta2 | 0.95 | frozen proposal (deviates from minimind native 0.999; modern LM practice — confirm at review) |
| eps | 1e-8 | frozen |
| Weight decay | decoupled, from the 2-value grid (§5); applied to all matrices AND norm weights, NOT embedding (standard minimind-style exemption optional — **decide at review**, single choice for whole experiment) | pending review |
| LR | base × multiplier grid (§5) | frozen mechanism |

### 3.2 Muon arm

| Field | Value | Status |
|---|---|---|
| Implementation source | **Canonical** Keller Jordan `muon.py` (github.com/KellerJordan/Muon), single file, vendored with commit hash + SHA256. **UNRESOLVED: no network access on this server (2026-09-26); acquisition pending** — see PROVENANCE.md §3 | **unresolved (blocking M0)** |
| Local candidate found | `skewadam/train.py` Muon variant (commit `50f3da6`) — **REJECTED as primary**: no Nesterov, no update-scaling rule, shape-heuristic AdamW fallback with dims hardcoded for another model, bf16 stochastic-rounding hacks. Usable only as a secondary cross-check after review | frozen (rejection) |
| momentum | 0.95 | frozen proposal |
| Nesterov | True | frozen proposal |
| Newton-Schulz | 5 steps, quintic coefficients (3.4445, −4.7750, 2.0315), bf16 internal compute | frozen proposal (must equal vendored file) |
| Update scaling rule | canonical rule of the vendored version (e.g. `0.2·sqrt(max(A,B))` family); recorded verbatim from the frozen file at vendor time | frozen at vendor time |
| Weight decay | same decoupled WD grid as AdamW arm, applied to all params of both routes | frozen proposal |

Any deviation discovered between the vendored file and this table **blocks M0** until re-registered.

### 3.3 Parameter routing (frozen)

**Muon arm** (two routes, single optimizer wrapper with two param groups):

- **Muon route — hidden linear matrices ONLY** (2D weights of): `self_attn.{q,k,v,o}_proj`,
  `mlp.{gate,up,down}_proj` of every layer. For the proposed config: 56 tensors, 26.74M params.
- **AdamW route — everything else**: `model.embed_tokens.weight` (= tied `lm_head.weight`),
  all RMSNorm weights (`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, final
  `norm`), any bias or scalar/vector parameters (none in MiniMind; assert empty). For the proposed
  config: 34 tensors, 3.29M params.

**Tied embedding/head (explicit rule, frozen):** `embed_tokens.weight` and `lm_head.weight` are the
same tensor (`model_minimind.py:236,242`). It is registered in the AdamW route **exactly once**
(by `id()`); the routing audit must show `n_unique_routed == n_unique_params` and an empty
intersection between routes.

**AdamW arm:** all parameters, single group (embedding/norm treatment identical to Muon arm's AdamW
route, including the tie handling — one entry).

Aux LR of the Muon arm's AdamW route: tied to the same LR schedule and the same LR multiplier as the
Muon route (single LR per run; no separate aux-LR search — frozen, prevents search-size asymmetry).

---

## 4. Other Training Mechanics (frozen proposal)

| Field | Value |
|---|---|
| Schedule | cosine with fixed short warmup, identical formula for both arms; recompute per run from budget; schedule multiplier applied to both routes |
| Grad clip | global norm 1.0 across ALL params (both routes) before optimizer step, both arms |
| Batch | identical microbatch/accumulation for all runs (exact values with budget, §8) |
| Seeds | M1/M2 single seed (e.g. 2030); M3 seeds {2040, 2041, 2042} (disjoint from M1/M2) |
| Determinism | four-way RNG capture/restore; per-run stream seeds via legacy `rng.py`/`stream_rng.py` pattern; cudnn deterministic |
| Eval pools | fixed, frozen in config with SHA; identical across all runs |
| Fairness | both arms share the identical training loop, identical data manifests per (seed, p), identical eval; the ONLY differences are optimizer + its LR/WD grid values |

---

## 5. Hyperparameter Search Space (frozen)

Per optimizer × mixture `p`:

- LR multiplier: `{0.5, 1, 2}` relative to that optimizer's own reference base LR
  (reference bases: AdamW `3e-4`, Muon `0.02` — **pending review**, then frozen;
  multipliers are identical across optimizers).
- WD: two predefined values `{W1, W2}` — **values pending review** (proposal: `{0.0, 0.1}`;
  identical set of *meanings* per arm: Muon arm applies WD to all routed params).
- Grid = 3 LR × 2 WD = **6 cells** per (optimizer, p). Equal search size per optimizer by construction.

**No additional search after observing results.** No cell may be re-run with new values; no
boundary extension except the single pre-registered contingency: if the best cell for either
optimizer sits at an extreme multiplier (0.5 or 2), one halving/doubling extension is allowed ONCE,
logged as an amendment, before M1 gate decisions — never after M2 begins.

---

## 6. Experimental Stages (frozen)

### M0 — Implementation validation (no scientific conclusion)

Gate checklist (all must pass; see M0 plan in PROJECT_CONTEXT.md §6):

1. Optimizer parameter grouping: routing table matches §3.3 exactly (per-tensor audit artifact).
2. No parameter overlap: Muon route ∩ AdamW route = ∅; union = all unique params.
3. Tied weights handled once (id-based registry; save/load roundtrip preserves tie).
4. Muon update correctness: one-step update on synthetic tensors matches an independent
   NumPy/torch reference implementation of the vendored algorithm to numerical tolerance.
5. AdamW auxiliary groups: Muon-arm AdamW route produces bit-identical (or tol-verified) updates to
   `torch.optim.AdamW` on the same tensors.
6. Checkpoint save/load: full-state roundtrip (model + optimizer state incl. Muon momentum buffers
   + RNG + step counter) with post-roundtrip single-step bit-replay (legacy `full_state.py` pattern,
   extended to the two-group Muon optimizer; format version bumped and registered).
7. Deterministic seed: two same-config runs → identical loss curves for N steps (bit-level CRN replay).
8. Token counting: realized `p_hat` per run matches nominal `p` within tolerance; token accounting
   is by sum (not mean of means).
9. Evaluation sum/count: `NLL_A`, `NLL_B` computed as total-NLL/total-tokens over fixed pools;
   `V_Q = 0.5·NLL_A + 0.5·NLL_B`; a dedicated test proves sum/count ≠ mean-of-batch-means.

M0 consumes GPU only for smoke-scale validation (minutes). **No scientific reading of M0 numbers.**

### M1 — p = 0.5 baseline (12 runs)

- AdamW: 3 LR × 2 WD at p=0.5 → 6 runs.
- Muon: 3 LR × 2 WD at p=0.5 → 6 runs.
- Goal: confirm both optimizers train normally (loss decreases; at least one non-diverged cell per
  optimizer; realized p_hat within tolerance).
- Gate to M2: both optimizers have ≥1 stable cell; infra artifacts complete. Selection maps
  (`best cell per optimizer`) recorded from dev `V_Q` only.

### M2 — add p = 0.25, 0.75 (24 new runs; 36 cumulative)

- Same 6-cell search at p ∈ {0.25, 0.75} for both optimizers.
- Goal: detect possible optimizer-dependent mixture preference.
- Per optimizer: `p*_opt = argmin_p [ best-cell dev V_Q at p ]`.
- **Signal criterion (pre-registered)**: `p*_AdamW != p*_Muon` AND for each optimizer the V_Q
  advantage of its own `p*` over the other's `p*` exceeds the M3-seed noise scale estimated from
  pilot replays (formal threshold frozen in the M3 config BEFORE any M3 run).

### M3 — confirmation (only if M2 shows signal)

- Freeze `p*_AdamW`, `p*_Muon` exactly as selected in M2. **No retuning, no grid changes.**
- Runs: {p*_AdamW, p*_Muon} × {AdamW, Muon} × 3 new seeds = 12 runs, at the M1/M2-selected
  hyperparameters per (optimizer, p).
- Verdict metric on the **confirm** split (read once, sealed-eval discipline):
  - Primary: `p*_AdamW != p*_Muon` reproduces across seeds (sign consistency 3/3 per optimizer).
  - Migration metric: `R_s = NLL_Muon(p*_AdamW) − NLL_Muon(p*_Muon)` (reported with the symmetric
    AdamW version `R_s^AdamW`). `R_s < 0` would mean Muon trained at AdamW's mixture is better —
    i.e., no real optimizer-dependent preference; `R_s > 0` beyond seed noise supports the interaction.
- If M2 shows no signal → M3 does not run; the screening verdict is "no evidence of optimizer-dependent
  mixture preference at this scale" — a valid, publishable-in-lab negative result.

---

## 7. Metrics (frozen)

**Primary: final token NLL** at end of training (last step, fixed budget exhausted).

Report per run:

- Domain A NLL (`NLL_A`), Domain B NLL (`NLL_B`), weighted `V_Q = 0.5·NLL_A + 0.5·NLL_B`.

Explicitly banned as decision metrics: best intermediate checkpoint, training loss, early-curve
comparisons, recovery speed. (Curves may be *logged* as diagnostics, never *selected on* —
V_monitor/V_final separation per lab discipline.)

Migration metric: `R_s` as defined in §6/M3.

---

## 8. Unresolved Fields (blocking items — the ONLY things that may be filled after this freeze)

| # | Field | Blocking stage | Resolution procedure |
|---|---|---|---|
| 1 | Model config final values (§1 "pending review" rows) | pre-M0 | protocol review sign-off → M0 param audit |
| 2 | Datasets final choice + upstream provenance confirmation | pre-M0 | review + dedup/token-stat audit; replace candidate only via v1.2 amendment |
| 3 | Tokenizer statistics + per-domain coverage report | M0 | M0 data audit artifact |
| 4 | Token budget per run; batch size; total compute estimate | pre-M1 | review; then frozen in `configs/m1_baseline.json` |
| 5 | Muon implementation commit/hash + verbatim constants (§3.2) | **pre-M0 (hard blocker)** | acquire canonical `muon.py` (network restore, offline copy, or user-provided file) → vendor + SHA256 + reference-implementation test |
| 6 | Reference base LRs and WD grid values (§5) | pre-M1 | review; frozen in M1 config |
| 7 | Split proportions + eval pool sizes | pre-M1 | M0 partition audit |

---

## 9. Red Lines (frozen, inherited from lab discipline)

1. Legacy and sibling projects are read-only; vendor-with-SHA256 only.
2. No code without provenance. The Muon source hash is frozen BEFORE its first use.
3. No threshold or metric changes after observing results; amendments only as dated v1.x sections.
4. Confirm split is read exactly once (M3 verdict); dev split is the only selection surface in M1/M2.
5. No adaptive mixture, no new algorithm, no optimizer modification.
6. Negative results terminate the line without "rescue" attempts.
7. Every run's full config + data manifest + code hash is archived before launch (preregistration).
