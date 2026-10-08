# muon_transition_experiment

**Muon Transition Experiment: Optimizer-Dependent Data Mixture Interaction**

Screening experiment (frozen question): under identical model, token budget, data domains, and a
fair equal-size hyperparameter search, does changing the optimizer from AdamW to Muon change the
optimal training token mixture `p` between two domains — i.e., is `p*_AdamW != p*_Muon`?

**Not** a new optimizer project. No new algorithms. No adaptive controllers. Muon is not assumed
better. Negative results are valid and terminal.

## Status (2026-09-26, post-M1)

**M0 / M0.5 / M1-0 smoke / M1 all complete.** M1: 12/12 runs OK (p=0.5, seed 2030, B0=2^26,
~10–13 min/run on 2×5090), gate **PASS** (6/6 stable cells per arm). Summary:
`outputs/m1/M1_SUMMARY.md`. Best cells (dev V_Q): AdamW 2.6412 (lr 6e-4/wd0.01), Muon 2.3512
(lr 0.01/wd0.02) — both at the lr_mult=2.0 grid boundary; the frozen v1.1 §5 one-time boundary
extension (mult=4.0 × 2 WD × 2 arms = 4 cells) is available and **pending user decision** before
M2 selection maps freeze. Observation hooks logged 21 records/run (diagnostic only).
**M2 not started.**

## Layout

```
docs/MUON_PROTOCOL_v1.1.md       frozen protocol (research question, model, data, optimizers,
                                  search space, stages M0–M3, metrics, red lines)
docs/MUON_IMPLEMENTATION_AUDIT.md M0 Gate 1: frozen Muon source + full algorithm audit
docs/PROJECT_CONTEXT.md          lab context, audit summary, M0 implementation plan
docs/PROVENANCE.md               asset registry: SHA256 of sources, Muon freeze, dataset audit,
                                  append-only vendoring log
configs/                         (empty — m1_baseline.json lands here after review)
src/muon_transition/             config/model/routing/optimizers/data/partitions/eval_nll/
                                  checkpoint/train_loop/observation_hooks/reference_muon
src/vendored_fr/                 legacy infrastructure verbatim (7 modules, SHA-verified)
scripts/                          m0_gate1..4 + run_m0_suite.py
tests/                            6 test modules (routing, muon reference, adamw aux,
                                  eval sum/count, repro+checkpoint, observation hooks)
outputs/m0/                       routing_report.json, data_audit/, pipeline_audit.json,
                                  muon_freeze.json, M0_REPORT.md
outputs/observations/            (future: optional hook logs during M1/M2)
```

## Stage gates

M0 implementation validation (no scientific reading) → M1 p=0.5, 12 runs → M2 p∈{0.25,0.75},
36 cumulative → M3 (only if M2 signal) 12 confirmation runs, no retuning. Primary metric: final
token NLL on fixed eval; `V_Q = 0.5·NLL_A + 0.5·NLL_B`; migration metric `R_s`.

## Environment

conda env `fr_minimind`; run from project root with `PYTHONPATH=src`; explicit
`CUDA_VISIBLE_DEVICES`; legacy and sibling projects strictly read-only.
