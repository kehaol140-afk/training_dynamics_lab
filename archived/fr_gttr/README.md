# FR / GTTR — Fluctuation-Response & Geometry-To-Training-Rule (archived)

**Status: COMPLETED & FROZEN (2026-09-26). Read-only archive.**

## What this was

A research line on AdamW-trained MiniMind Transformers studying:

1. **FR (Fluctuation-Response)**: whether perturbation-response statistics of training
   (pulse a direction q at the pre-step gradient, measure the response) can be predicted
   from smooth tangent models, generalized across scale/data, and used for cheap
   action scoring / LR control.
2. **GTTR (Geometry-To-Training-Rule)**: whether locally-identified first/second-order
   geometry (gradient + Hessian-vector products, terminal hypergradients) could be turned
   into a training rule that beats tuned baselines.

The line ran 2026-09-01 → 2026-09-26 and was **terminated by its own preregistered stop
rules** (GTTR final negative: derivatives could be computed correctly but did not transfer
into usable training-rule gains). 91 experiment/version entries; every verdict frozen.

## Directory layout

```
archived/fr_gttr/
├── README.md                 this file
├── RESULTS.md                structured scientific summary (evidence-linked)
├── EXPERIMENT_INDEX.md       all 91 entries, verdicts transcribed from frozen index
├── code/
│   ├── research_src/         research/src (fr framework v0.6.0-cn + all experiment modules)
│   ├── research_scripts/     171 runner scripts (run/analyze drivers)
│   ├── research_tests/       unit tests incl. bit-level roundtrip tests
│   ├── research_configs/     frozen config JSONs
│   ├── a0h15/ adapters/ analysis/ reports/
│   ├── tools/gttr_f0_ic_v1/  GTTR-F0-IC independent verification package
│   └── FR_A0C_AdamW_m_v_Theory_v06_gradient.py
├── protocols/
│   ├── root_docs/            18 root FR_*.md protocol/report docs + preregistration
│   │                         (CONFIRMATORY_PREREGISTRATION_V07.md, AGENTS.md, ...)
│   └── docs/                 docs/ selection: FR_PROJECT_STATUS.md (49a sections),
│                             FR_EXPERIMENT_INDEX.md, research manual/plan (v2),
│                             handoffs, retrospectives, specs/, project_inventory/,
│                             report_audit/ (incl. A0G audit docx)
└── results/
    ├── outputs/              small evidence files (<5 MB each) mirroring outputs/ layout:
    │                         summary.json / REPORT.md / verdict CSVs / figures / seals
    │                         (2913 files, ~48 MB). NO checkpoints, NO datasets, NO raw states.
    ├── server_audit_20260926/ full 12-doc server audit bundle (incl. the 39-column
    │                         01_COMPLETE_EXPERIMENT_INDEX.csv — source of truth)
    └── ARTIFACT_MANIFEST.md  everything NOT uploaded: size, reason, delete-safety
```

## Known source-code loss (important)

- `code/research_src/src/fr/theta_protocol.pyc` exists **only as compiled bytecode**.
  The `.py` source was lost before archiving (recovered-as-pyc forensic event, experiment
  index #85). The interface can be reconstructed from `research_tests/test_theta_protocol.py`,
  but **no source exists and none is pretended to exist**.
- Everything else under `code/` is original `.py` source, copied verbatim read-only.

## Model / environment (shared by almost all experiments)

MiniMind dense Transformer, 4,983,168 unique params (hidden 256 × 4 layers, GQA 8/4,
vocab 6400, tied embeddings); AdamW β=(0.9, 0.999) eps 1e-8 wd 0.01; global grad-clip 1.0;
bf16 autocast; batch 4 × accum 2; seq 128; corpus D0 = 4096-row subset
(`outputs/e3_confirmatory/v07/e3c_subset.jsonl`, SHA256 in `DATASET_SHA256.txt`).
Frozen-state banks: v07 (56 states = 8 seeds × 7 optimizer steps 46–82, `full_state` v0.2),
G1a 26.68M × 9 states, G1b 100.78M × 9 states — **banks are NOT in this repo** (too large);
see `results/ARTIFACT_MANIFEST.md`.

## Where to start reading

1. `RESULTS.md` — the one-file scientific summary with evidence links.
2. `EXPERIMENT_INDEX.md` — all 91 entries.
3. `protocols/docs/FR_PROJECT_STATUS.md` — the authoritative long-form status log.
4. `results/server_audit_20260926/00_CHATGPT_HANDOFF.md` — narrative handoff.
