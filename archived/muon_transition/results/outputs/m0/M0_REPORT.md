# M0_REPORT — Muon Transition Experiment validation gate

Date: 2026-09-26. Suite: `scripts/run_m0_suite.py` — **11/11 checks PASSED** (6 test modules,
5 gate scripts). CPU only; no full training; no scientific reading of any number below.

---

## Gate 1 — Muon implementation freeze and audit: **PASS**

- Primary implementation: **`torch.optim.Muon`** (official PyTorch prototype, torch==2.11.0 wheel,
  env `fr_minimind`; file SHA256 `38e95918…`; upstream permalink KellerJordan/Muon@`f90a42b`).
- Full audit: `docs/MUON_IMPLEMENTATION_AUDIT.md` (update rule, momentum, Nesterov, NS5 quintic,
  scaling rule, dtype, state format). Frozen constants in `config.py::MUON_FREEZE`.
- Verified against an independent fp64 reference (cosine ≥ 0.9999, rel err ≤ 1.2% attributable to
  bf16 NS arithmetic); NS scale-invariance cosine = 1.0000; non-2D params structurally rejected.
- The previously-rejected skewadam variant and the LlamaFactory Moonlight variant remain registered
  alternatives (PROVENANCE.md §3); neither is used.
- Artifact: `outputs/m0/muon_freeze.json`.

## Gate 2 — Parameter routing audit: **PASS**

Artifact `outputs/m0/routing_report.json` (required fields):

| Field | Value |
|---|---|
| total parameters | **30,025,216** |
| muon parameters | **26,738,688** (56 tensors: q/k/v/o_proj + gate/up/down_proj × 8 layers) |
| adamw auxiliary parameters | **3,286,528** (34 tensors: embedding 3,276,800 [tied with lm_head, registered once] + 33 norm weights 9,728) |
| overlap count | **0** |
| missing count | **0** |

QKV routing explicit: `self_attn.{q,k,v,o}_proj.weight` → Muon (24 tensors; k/v are GQA-shaped
[256,512], still 2D matrices — included by the frozen rule "hidden linear matrices"). Extra counts
add up exactly; no bias/scalar in Muon route; tie handled once by `id()`.

## Gate 3 — Dataset candidate audit: **PASS (with required actions taken: dedup + thread-level split)**

Artifacts: `outputs/m0/data_audit/{data_report.json, split_manifest.json, eval_pools.pt,
gradient_geometry.csv, gradient_geometry_summary.json}`.

### A. Token statistics (after exact-dedup, keep-first; see finding F1)

| Statistic | Domain A (EN) | Domain B (ZH) |
|---|---|---|
| documents (raw → dedup) | 79,107 → 60,861 | 76,399 → 74,591 |
| exact duplicate rate | **23.1%** | 2.4% |
| tokens (dedup) | 104.3M | 80.1M |
| avg tokens/doc | 1,714 | 1,074 |
| vocab coverage (unique/6400) | 5,015 (78.4%) | 6,280 (98.1%) |
| train effective loss tokens | 98.19M | 75.44M |

### B. Split integrity — **verified** (`mte_split_v2_thread`)

- `conversation_id` is a **thread id** (up to 116 lines/thread); each line = one exchange. All
  18,486 shared threads are translation-paired across A/B (16,476 with equal line counts).
- Split unit = whole thread → a translation pair can never straddle splits; all lines covered
  exactly once in both domains; verified programmatically (0 problems).
- Fractions 94/3/3, seed 777; manifest SHA-frozen (`e62864ad…`); eval pools = first 512 windows
  per domain per split → **130,560 tokens per domain per pool, exactly equal**.

### C. Gradient geometry probe (diagnostic ONLY — no mechanism claims)

Untrained 30M model, fp32, 8 batches × 8×256 tokens per domain, 64 cross-domain pairs:

| Metric | Value |
|---|---|
| cos(g_A, g_B) mean [min, max] | 0.103 [−0.044, 0.376] |
| conflict ratio P(g_A·g_B < 0) | 0.1875 |
| ‖g_A‖/‖g_B‖ mean [min, max] | 1.039 [0.78, 1.32] |
| within-domain mean cosine (A / B) | 0.294 / 0.174 |
| update-direction cosine, first batch (AdamW / Muon) | 0.042 / 0.399 |

Domains are neither aligned nor maximally conflicting; distinct-enough geometry to run the mixture
question. **No interpretation beyond "audit produced numbers".**

## Gate 4 — Training/evaluation pipeline audit: **PASS**

Artifact `outputs/m0/pipeline_audit.json`.

1. **Token accounting frozen**: tokenizer SHA `71f32c68…`; seq_len 256; packing = per-doc
   `enc(doc)+DOC_SEP(0)` concatenated, windows of `[BOS(1) + 255 stream tokens]`; reserved ids
   {0,1,2} filtered from encodings; effective loss tokens = n_windows × 255.
2. **Budget B0 frozen (proposal)**: **B0 = 67,108,864 tokens = 2^26 = 2048 optimizer steps ×
   32,768 tokens/step** (micro-batch 32×256, accumulation 4). Chosen so the full p-grid is
   **single-epoch feasible** in both domains (p=0.75 needs 50.3M A-tokens ≤ 98.2M available ✓).
   Magnitude subject to review; definition frozen.
3. **Evaluation**: Q_dev / Q_confirm pools frozen (equal token counts per domain); token NLL =
   sum/count per domain; `V_Q = 0.5·NLL_A + 0.5·NLL_B`; sum/count ≠ mean-of-batch-means proven
   (2.940 vs 3.152 on the constructed case).
4. **Reproducibility**: same seed + same stream → identical losses AND identical parameter
   fingerprints for 3 steps, **both arms**.
5. **Checkpoint**: `mte_state_v1` saves model + optimizer states + RNG + data cursor +
   payload SHA; save@step2 → load → step3 bit-matches uninterrupted run; RNG draw sequence
   restored (torch.equal); works for the two-optimizer Muon arm.
6. **Real-data 15-step smoke (plumbing only)**: AdamW lr 3e-4: 8.85 → 7.37; Muon lr 0.005:
   8.85 → 7.49 (min 6.92). Both arms train on real data.

## Additional observation hooks — added, not used

`src/muon_transition/observation_hooks.py`: JSONL logger (default every 100 steps) for grad norm,
sampled gradient singular values, domain cosine/conflict, update/weight norms + ratio, and
`update_work_proxy = −gᵀΔθ` (name per mandate; not "energy"). Unit-tested; **no primary-evidence
status**; outputs would land in `outputs/observations/` during M1/M2 only.

---

## M0 findings that require review decisions (not implementation defects)

- **F1 — Domain A duplicate rate 23.1%**: exact-duplicate removal (keep-first, within domain,
  pre-split) is now IN the audited pipeline. Asymmetric raw duplication would have silently biased
  mixtures; dedup is mandatory and applied.
- **F2 — Thread structure**: `conversation_id` ≠ document. Split unit corrected to thread level
  (line-level pairing was initially wrong; caught by this gate's integrity check and fixed).
- **F3 — Muon base LR anchor**: protocol placeholder 0.02 (Keller-speedrun scale) is **unstable**
  at our micro-scale smoke (2048 tokens/step, no warmup): stable and competitive at 0.001–0.005.
  The M1 reference base LR for Muon must be reset at review (e.g. 0.005 with grid {0.0025, 0.005,
  0.01}); multiplier grid {0.5,1,2} unchanged. Note smoke conditions (tiny batch, no warmup, no
  accumulation) are harsher than M1 — review should anchor accordingly.
- **F4 — B0 magnitude**: 2^26 tokens proposed for single-epoch feasibility at all p; review may
  trade off against repetition policy.
- **F5 — WD semantics note**: Muon's decoupled decay uses unadjusted lr; both arms share the λ
  grid semantics (θ ← θ(1−lr·λ)). Recorded; no action needed unless review objects.

## Remaining unresolved issues (updated protocol §8 list)

1. ~~Muon implementation source~~ **RESOLVED** (torch.optim.Muon, Gate 1).
2. ~~Parameter routing~~ **RESOLVED** (Gate 2).
3. Model config final values — **proposed & validated** (30,025,216 params); needs review sign-off.
4. Datasets — candidate **audited & integrity-passed** (dedup + thread split + pools); upstream
   provenance of the sharegpt pair still second-hand (no network) → confirm or replace at review.
5. Tokenizer stats — **done** (coverage 78%/98%); frozen.
6. Budget B0 — **definition frozen, magnitude proposed** (2^26); review sign-off.
7. Reference base LRs / WD values — **requires review** (F3 finding central here).
8. M1 config file (`configs/m1_baseline.json`) — to be written and frozen after review.

## M0 success criteria check

| Criterion | Status |
|---|---|
| 1. Muon implementation provenance fixed | **PASS** |
| 2. Parameter routing complete | **PASS** |
| 3. Dataset candidate passes integrity checks | **PASS** (post dedup + thread-split fix) |
| 4. Training/evaluation accounting frozen | **PASS** (B0 magnitude + LR anchors pended to review) |
| 5. First smoke update reproducible | **PASS** (both arms, bit-level) |

## Verdict

**M0: PASS.** All five success criteria met; suite reproducible via `scripts/run_m0_suite.py`.

**Is the M1 baseline allowed?** Conditionally yes — the *implementation* is fully validated and
M1-ready. The gate that remains is protocol review, not code: sign-off on F3 (Muon LR anchor),
F4 (B0), dataset provenance confirmation (item 4), and the frozen `configs/m1_baseline.json`.
No M1 training has been started.
