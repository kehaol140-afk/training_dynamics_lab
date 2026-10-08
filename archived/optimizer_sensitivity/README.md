# optimizer_sensitivity — Stable Training, Unstable Sensitivities (archived)

**Status: COMPLETED 2026-10-08 (research program closed at F4C + decision memo; external validation plan written but NOT executed).**

## What this was

A falsification-driven research line asking why AdamW-trained Transformers show *stable
primal training* but *unstable sensitivities* (tangent / hypergradient / curvature
dynamics). Chain: F0 (bank audit) → F1 (mechanism closure) → F2/F2.5 (utility boundaries)
→ F3/F3.5 (curvature feedback mechanism) → F4A/F4A.1/F4A.5 (scale×phase×seed map and
formal modeling) → F4B/F4B-postmortem (method attempt and its honest negative) → F4C
(intervention-scale mining) → consistency audit → decision memo.

**Final scientific one-liner (frozen synthesis):** ill-conditioning infrastructure
(v-floor) is always present; curvature feedback is excited direction-selectively in the
early window; phase-dominated self-healing follows; macro predictable / micro
unpredictable; and the only methodized attempt (SAU) produced **NO_METHOD_GAIN** — this
archive must not be cited as a successful optimization method.

## Layout

```
archived/optimizer_sensitivity/
├── README.md               this file
├── RESULTS.md              12-section evidence-linked synthesis
├── EXPERIMENT_INDEX.md     all units F0→F4C with verdicts
├── ERRATA_20261008.md      cross-cutting numeric corrections (canonical values)
├── code/                   src/opsens + scripts + tests (no checkpoints, no nested .git)
├── configs/                frozen config JSONs
├── protocols/              all 45 frozen prereg/report/erratum/plan docs
├── reviews/2026-10-08/     independent re-audit bundle (report, UPDATE_DIFF, 5 CSVs,
│                           SELF_CHECK, recompute notes + verify_all.py, evidence/, zip)
└── results/
    ├── outputs/            591 small evidence files (summaries, verdict CSVs, figures)
    └── ARTIFACT_MANIFEST.md  excluded large artifacts (73.1 GB, incl. F3 state sidecars
                              and the F4A.1 5M matched-coverage bank)
```

## Source

`/home/b205/projects/training_dynamics_lab/optimizer_sensitivity/` (independent git
repo; HEAD `98ed984` at archive time). Read-only copy; nothing modified, nothing re-run.

## Not executed (do not cite as done)

- EXTERNAL_VALIDATION_PLAN (second-recipe / second-dataset 50M–150M) — **written, not run**.
- F4B SAU budget-grid relaxation (B_λ ≥ 0.5) — would require a new preregistration.
- F4C STAGE D confirmatory — DISABLED_BY_PREREG.
