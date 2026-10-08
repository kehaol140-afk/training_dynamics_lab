# HANDOFF_AUDIT_20260926 — M2 launch state

Date: 2026-09-26 (handoff audit, pre-M2-launch). Auditor: taking-over agent.
No training was run during this audit. No protocol/config/data/model/budget item was modified.

---

## 1. Repository / integrity

- `muon_transition_experiment/` is **not a git repo** (nor is its parent). Integrity verified by
  SHA256 + frozen-file timestamps instead.
- `docs/MUON_PROTOCOL_v1.2.md` mtime 2026-09-26 18:47 — predates all M1 runs (19:00+) and all M2
  artifacts; no modification after freeze. v1.1 remains untouched (17:18).
- Muon freeze intact: `torch/optim/_muon.py` SHA256
  `38e95918424bd26604f786195a4aa72afb60674b4ac9ccbde777974013cb1f52` = v1.2 §2 recorded value;
  torch==2.11.0 (env `fr_minimind`).
- Data SHAs match DATASET_CARD §1 / prereg: A `7a61fda2…` ✓, B `a78dfae7…` ✓;
  tokenizer `71f32c68…` ✓. split_manifest + eval_pools SHAs recorded in m1 prereg.

## 2. M0 / M0.5 status: PASS

- `outputs/m0/M0_REPORT.md`: 11/11 suite checks PASS (6 test modules + 5 gate scripts).
- Gate 1 Muon freeze/audit PASS; Gate 2 routing PASS (30,025,216 total; Muon route 26,738,688 /
  56 tensors; aux 3,286,528 / 34 tensors; overlap 0, missing 0, tie once); Gate 3 data audit PASS
  (dedup A 23.06% / B 2.37%; thread split 94/3/3 seed 777; no-leakage verified).

## 3. M1 status: PASS (12/12)

- All 12 cells under `outputs/m1/<run_id>/` have result.json + final_state.pt + metrics.jsonl +
  config.json. `M1_SUMMARY.md`: stable cells 6/6 per arm, gate PASS.
- Best cell both arms at LR multiplier 2.0 (AdamW V_Q 2.6412 @ lr 6e-4; Muon V_Q 2.3512 @ lr 0.01)
  → boundary condition; addendum §1 extension justified and preregistered (4 cells).

## 4. M2 pre-launch state

- `docs/M2_PROTOCOL_ADDENDUM.md`: **exists**, frozen 2026-09-26 21:36. LR boundary rule + M2 spec +
  stop rule as required. No creation needed.
- `configs/m2_prereg.json`: 28 local cells = 24 new base (p=0.25/0.75) + 4 justified extensions
  (p=0.5, mult 4.0). M1 p=0.5 cells reused (12).
- Progress at takeover: **3/28 complete**
  (m2_adamw_lr0.5_wd0.01_p0.25, m2_adamw_lr0.5_wd0.02_p0.25, m2_muon_lr1.0_wd0.01_p0.25).
- **2/28 partial-interrupted** (no result.json; metrics only): adamw_lr1.0_wd0.01_p0.25 (step 100),
  muon_lr1.0_wd0.02_p0.25 (step 1700). `pending()` in scripts/m2_run.py treats them as pending →
  clean re-run from step 0 with frozen seed 2030 (CRN deterministic per M0 test 6).
- GPUs idle at takeover (0 compute processes, no lock files).
- Observation hooks verified live in completed M2 runs: `obs.jsonl` 21 records, required fields
  present (domain_gradient_cosine, domain_conflict_ratio, grad_global_norm, update_norm,
  update_work_proxy). Diagnostic only; not read by selection path.
- **Cosmetic note (no action)**: runner hardcodes `"stage": "M1"` inside result.json/config.json
  metadata for all runs incl. M2. run_id (`m2_*`) + prereg registry are authoritative; no code
  change made per freeze.

## 5. Launch decision

Proceed per frozen protocol: `python scripts/m2_run.py` runs the 25 pending cells, 2-GPU shard.
No retuning, no new cells, no new optimizers, no new mixtures.
