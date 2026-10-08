# M1_PRELAUNCH_CHECKLIST — sign-off required before any GPU training

Project: muon_transition_experiment. Protocol in force: v1.1 + v1.2 patch. M1 = p=0.5,
2 arms × 3 LR × 2 WD = **12 runs**, seed 2030, B0 = 2^26 tokens = 2048 steps each.
**No M1 run may start until every box below is checked and this file is signed.**

---

## A. Protocol & config (paper side)

- [ ] `docs/MUON_PROTOCOL_v1.2.md` reviewed and accepted (model / Muon / AdamW / budget / WD freezes)
- [ ] `docs/DATASET_CARD.md` reviewed and accepted (incl. honest upstream-provenance limitation §1)
- [ ] `configs/m1_baseline_prereg.json` frozen: 12 run IDs, LRs {1.5e-4, 3e-4, 6e-4} /
      {0.0025, 0.005, 0.01}, WD {0.01, 0.02}, schedule warmup 32 → cosine to 10%
- [ ] No-retuning clause acknowledged: no grid/LR/WD change after observing results
      (single v1.1 §5 boundary-extension contingency excepted, must be logged as amendment)
- [ ] M1 gate criteria acknowledged: both arms train normally; ≥1 non-diverged cell per arm;
      selection reads dev V_Q only; confirm split untouched

## B. Code & determinism

- [ ] M0 suite green on current tree: `python scripts/run_m0_suite.py` (11/11)
- [ ] `src/muon_transition/config.py` values identical to `configs/m1_baseline_prereg.json`
      (re-run `python scripts/gen_m1_config.py` and diff; config.py SHA embedded in the JSON)
- [ ] Muon freeze intact: torch==2.11.0, `_muon.py` SHA256 `38e95918…` unchanged
      (`sha256sum …/torch/optim/_muon.py`)
- [ ] Routing audit re-run clean (overlap 0 / missing 0 / tie once)
- [ ] Mixture manifest generator for p=0.5 exists and realized `p_hat` ∈ 0.5 ± 0.01 at every
      decile (token-level accounting, pre-generated, SHA-logged per run)

## C. Data & evaluation

- [ ] Domain A/B file SHA256 match `DATASET_CARD.md` §1
- [ ] `split_manifest.json` SHA matches the one recorded in `configs/m1_baseline_prereg.json`
- [ ] `eval_pools.pt` SHA matches; dev pools 2 × 130,560 tokens; confirm pools sealed (no read)
- [ ] Split integrity check passes (0 problems)

## D. Compute & runtime

- [ ] GPU probed free (`nvidia-smi`); `CUDA_VISIBLE_DEVICES` set explicitly; both runs of any
      same-cell debugging on the same GPU model
- [ ] bf16 autocast smoke (100 steps, 1 arm × 1 cell, GPU, minutes) completes; checkpoints and
      observation hooks write correctly under `outputs/observations/`
- [ ] Disk space for 12 × (final checkpoint ~360 MB + logs) ≈ 5 GB confirmed
- [ ] Run IDs and artifact layout: `outputs/m1/<run_id>/` with config.json + manifest +
      metrics.jsonl + final_state.pt + obs.jsonl

## E. Observation hooks (diagnostic only)

- [ ] Hook cadence 100 steps + final step; required fields present
      (domain_gradient_cosine, domain_conflict_ratio, grad_global_norm, update_norm,
      update_work_proxy)
- [ ] Verified in hooks dry-run that records land in `outputs/observations/` and are NOT read by
      any selection/evaluation code path
- [ ] Naming rule enforced: no occurrence of the word "energy" in hook outputs

## F. Sign-off

| Role | Name / evidence | Date |
|---|---|---|
| Protocol owner | ______ | ____ |
| Implementation owner | ______ | ____ |

After sign-off: launch M1 per `configs/m1_baseline_prereg.json`. M2 (p=0.25/0.75) still requires
the M1 gate decision — it does not start automatically.
