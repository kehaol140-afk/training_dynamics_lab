# PROJECT_CONTEXT — Muon Transition Experiment

Title: **Muon Transition Experiment: Optimizer-Dependent Data Mixture Interaction**

Created: 2026-09-26. Nature of this document: read-only audit + experiment framing.
This document accompanies **zero training runs, zero GPU launches, zero algorithm changes**.
The governing frozen protocol is `docs/MUON_PROTOCOL_v1.1.md`.

---

## 1. Position within the lab

| Directory | Nature | Relation to this project |
|---|---|---|
| `legacy_fluctuation_response/` | frozen GTTR/FR line (git `6ac4942`, read-only) | source of vendored infrastructure (with SHA256 registration); no other use |
| `minimind_base/` | minimind native framework copy (read-only baseline) | model + tokenizer + native training-loop reference |
| `optimizer_sensitivity/` | active project (F0 done) | vendor/provenance pattern example; optional secondary statistics code |
| `continuation_adaptation/` | active project (P4 frozen, not launched) | most recent new-project template (structure, README, provenance discipline) |
| `muon_transition_experiment/` | **this project** | independent; siblings untouched |

Environment (verified 2026-09-26): 2 × RTX 5090 (32G), conda env `fr_minimind`
(torch 2.11.0 + CUDA, transformers 4.57.6, datasets 5.0.1), ~246G free disk.
**No outbound network** (curl to GitHub → HTTP 000; pip → SSL EOF via gateway). This blocks the
canonical Muon acquisition (protocol §8 item 5) until network restore, an offline copy, or a
user-provided file.

## 2. Why this question, and why now

The lab's prior lines established (root `PROJECT_CONTEXT.md` §3.3): short-horizon gains decouple
from final NLL; strong tuned baselines are mandatory; derivative-based methods failed at real step
sizes. A *screening* question about optimizer–data interaction is deliberately the opposite shape:
one factor (optimizer) × one factor (mixture p), plain grid, final-NLL verdict, negative-result-friendly.
No known lab asset answers it; no lab asset is consumed destructively by it.

## 3. Reusable-asset audit (implementation-first mandate; details + SHA256 in PROVENANCE.md)

| Capability | Asset | Status |
|---|---|---|
| Model | `minimind_base/model/model_minimind.py` — `MiniMindForCausalLM`, dense decoder-only, tied embedding (`model_minimind.py:236,242`) | reusable read-only; 30.02M-param config measured |
| Tokenizer | `minimind_base/model/tokenizer.json` (BPE, vocab 6400) | reusable; per-domain stats pending (M0) |
| Training loop | `minimind_base/trainer/train_pretrain.py` (`train_epoch`, AMP, grad-accum, clip-1.0 order) as *reference*; deterministic replication pattern from legacy `e3c_prepare_state_bank_v07.py:170-191` | adapt to single generic loop shared by both optimizer arms |
| Checkpoint | legacy `full_state.py` (fr_state v0.2: model+opt+scaler+RNG+sha, atomic, roundtrip-tested) | extend to two-group Muon optimizer state; bump+register format version at M0 |
| RNG | `rng.py`, `stream_rng.py` (four-way RNG, bit-level replay) | vendor pattern |
| Data plans | `batch_manifest.py`, `batch_stream.py`, `pilot_partition.py` (disjoint splits + sha) | extend: per-domain partition + token-level mixture manifests |
| Evaluation | `eval_observables.py::evaluate_token_losses` (fixed pool, per-token CE, sum/count, RNG-protected) | extend: per-domain pools `V_A`/`V_B` + fixed `V_Q` |
| Mixture precedent | `a0l_mixture_runner_v01.py` (data-composition-only treatment arms) | reuse *fixed*-mixture arm pattern ONLY; its adaptive controller is out of scope by protocol |
| Statistics/sealing | `confirmatory_stats.py`, `hashing.py` (`stable_hash`), write_seal/verify_seal discipline | vendor pattern |
| Secondary diagnostics | `optimizer_sensitivity` `f0_bank_audit_v01.py` (effective-LR / moment stats) | optional, diagnostic only |

Not found on this server (searched `~/`, all conda envs): any canonical Muon implementation
(no `muon` module, no `pytorch-optimizer`; the only local Muon is the rejected skewadam variant —
PROVENANCE.md §3).

## 4. Design consequences carried into the protocol

1. Single training loop, optimizer-agnostic; the optimizer object is the only arm-dependent module
   (fairness by construction, not by tuning).
2. Mixture `p` materialized as explicit pre-generated manifests → CRN pairing across optimizers at
   same (seed, p); realized `p_hat` audited.
3. dev (selection in M1/M2) vs confirm (one-shot M3 verdict) separation, sealing discipline.
4. Muon hard blocker (no network) resolved before ANY M0 GPU work on the Muon arm.

## 5. Cost framing (exact budget = unresolved field, protocol §8 item 4)

36 runs (M1+M2) + 12 runs (M3) ≈ 48 runs of a 30M model. With a per-run budget in the
10⁸–10⁹ token range, each run is tens of minutes to a few hours on one 5090 → full matrix is
bounded by a few GPU-days across both GPUs. M0 is minutes. No large model, no 1B assets needed.

## 6. M0 implementation plan (deliverable; executes only after protocol review)

Order of work, each step gated by the named test (all under `tests/`, artifacts under `outputs/m0/`):

1. **Vendor + freeze** (no GPU): copy `full_state.py`, `rng.py`, `stream_rng.py`,
   `batch_manifest.py`, `batch_stream.py`, `pilot_partition.py`, `eval_observables.py`,
   `hashing.py` verbatim into `src/vendored_fr/`; record SHA256 both sides (PROVENANCE.md pattern).
   Acquire + vendor canonical `muon.py` with commit hash (BLOCKING).
2. **Model card**: instantiate proposed config → `model_card.json` (param counts per route,
   init checksum, tie verification by `id()`).
3. **Routing audit** (`tests/test_routing.py`): build Muon-arm param groups; assert §3.3 mapping,
   empty intersection, complete union, tie registered once. Mirrors M0 checklist items 1–3.
4. **Optimizer correctness** (`tests/test_muon_reference.py`, `tests/test_adamw_aux.py`):
   one-step synthetic-tensor comparison of vendored Muon vs an independent in-test reference
   (NS5 iteration written from the frozen constants); AdamW-route equality vs `torch.optim.AdamW`.
   Checklist items 4–5.
5. **Checkpoint roundtrip** (`tests/test_full_state_muon.py`): extend `full_state` for two-group
   optimizer; save→load→single-step bit-replay; tie preserved after roundtrip. Item 6.
6. **Determinism** (`tests/test_crn_replay.py`): two fresh runs, same (seed, p) manifest → identical
   loss sequences for N=20 steps, both arms. Item 7.
7. **Data audit** (`scripts/m0_data_audit.py`): SHA re-verify, preprocessing freeze, per-domain
   tokenizer stats, dedup report, partition plan (train/dev/confirm per domain, pair-aligned),
   eval-pool manifests. Items 8 prerequisites + protocol §2.2 records.
8. **Token accounting + eval** (`tests/test_token_accounting.py`, `tests/test_eval_sum_count.py`):
   realized `p_hat` tolerance test; `V_Q` sum/count semantics test (≠ mean-of-means). Items 8–9.
9. **Smoke run** (minutes, 1 cell × 100 steps × both arms): logging/checkpoint/manifest plumbing
   verified end-to-end. **No scientific reading.**
10. M0 report → `outputs/m0/M0_REPORT.md`; only after all 9 checklist items pass does M1 config
    (`configs/m1_baseline.json`) get frozen.

## 7. Explicitly out of scope

New optimizers; optimizer modifications; adaptive data selection; theory naming; scale-up promises
beyond the 20–40M screening tier; any use of GTTR/FR line artifacts beyond registered vendoring.
