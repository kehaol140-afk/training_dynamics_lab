# DATASET_CARD — Muon Transition Experiment (frozen 2026-09-26, protocol v1.2 §5)

Two-domain corpus for the mixture experiment. Domain axis = **language** (English vs Chinese) on
content-matched (translation-paired) conversational data. All artifacts under
`outputs/m0/data_audit/`; this card records provenance, processing and the no-leakage case.

---

## 1. Sources

| | Domain A (EN) | Domain B (ZH) |
|---|---|---|
| role | training/eval domain A | training/eval domain B |
| local path | `/home/b205/FDA_INIT/pretrained_models/sharegpt_datasets/common_en_70k.jsonl` | `…/common_zh_70k.jsonl` |
| lines (raw) | 79,107 | 76,399 |
| **SHA256** | `7a61fda234ff634d35c5bb42b8ff761f0c1b54a895fec02ef411a9a212f31797` | `a78dfae704e5ca29f18b881b0d298f4bf401093bf33d4ea993dd5728e591c565` |
| format | JSONL, sharegpt-style: `conversation_id` (thread), `category`, `conversation[{human, assistant}]` | same |

**Upstream provenance (honest record):** inherited from another local project (`FDA_INIT`);
original public source/dataset card could not be verified — the server has no outbound network
(2026-09-26). The two files are internally translation-consistent (see §2), consistent with a
publicly released EN/ZH paired sharegpt corpus, but the upstream release itself is unverified.
Decision (v1.2): proceed with this pair; replacement would have required a protocol amendment
before M1 and was ruled out by review.

## 2. Pairing structure (M0 finding F2)

`conversation_id` is a **thread** id; each line is one exchange (1–116 lines per thread).
- Threads: 18,636 total; 17,279 shared between A and B; 201 only in A; 1,156 only in B.
- Shared threads are line-by-line translation pairs (spot-verified content match);
  12,465 shared threads have equal line counts in both files (unequal tails are unpaired lines —
  they stay in their own domain, never dropped).
- Consequence: the split unit is the **thread** (below), so a translation pair can never end up
  in different splits.

## 3. Preprocessing (frozen)

1. Flatten each line to text: turns joined as `"H: {human}\nA: {assistant}"` with newlines
   between turns (exact template in `src/muon_transition/data.py::conversation_to_text`).
2. Tokenize with MiniMind BPE (`tokenizer.json`, SHA256 `71f32c68…`, vocab 6400),
   `add_special_tokens=False`; **reserved ids {0,1,2} filtered out** of encodings
   (2 EN docs contained a literal special token).
3. Packing: per-domain stream = concat over docs of `enc(doc) + DOC_SEP(0)`; windows =
   consecutive slices of 255 stream tokens each prefixed by `BOS(1)` → window shape (256,);
   effective loss tokens per window = 255 (next-token targets, window-internal shift).
4. No cross-domain packing within a window.

## 4. Deduplication (frozen; M0 finding F1)

- Method: exact match on whitespace-normalized text, within domain, keep first occurrence,
  applied **before** splitting.
- Domain A: 79,107 → **60,861 docs** (23.06% duplicates removed — mandatory; raw corpus would
  have biased mixtures toward A-internal repetition).
- Domain B: 76,399 → **74,591 docs** (2.37% removed).
- Limitation (recorded): near-duplicate detection (e.g., MinHash) was NOT performed; only exact
  duplicates. Cross-domain content overlap is by design (translation pairs) and is not removed.

## 5. Split (frozen: `mte_split_v2_thread`)

- Unit: whole thread; permutation seed 777; fractions 94% / 3% / 3% (train / dev / confirm).
- Manifest: `outputs/m0/data_audit/split_manifest.json`
  (file SHA256 recorded in `configs/m1_baseline_prereg.json`).

| | A docs | A windows | A eff. tokens | B docs | B windows | B eff. tokens |
|---|---|---|---|---|---|---|
| train | 57,252 | 385,071 | 98,193,105 | 70,209 | 295,825 | 75,435,375 |
| dev | 1,813 | 11,969 | 3,052,095 | 2,240 | 9,262 | 2,361,810 |
| confirm | 1,796 | 12,230 | 3,118,650 | 2,142 | 9,288 | 2,368,440 |

Eval pools (frozen, `outputs/m0/data_audit/eval_pools.pt`): first **512 windows** per domain per
dev/confirm split → **130,560 loss tokens per domain per pool, exactly equal across domains**;
`V_Q = 0.5·NLL_A + 0.5·NLL_B` on equal evidence.

## 6. Token statistics (deduped)

| | Domain A (EN) | Domain B (ZH) |
|---|---|---|
| tokens | 104,303,425 | 80,091,588 |
| avg tokens/doc | 1,713.8 | 1,073.7 |
| length p50 / p95 / max | 1,533 / 2,934 / 8,506 | 979 / 2,181 / 18,858 |
| vocab coverage (unique/6400) | 5,014 (78.3%) | 6,280 (98.1%) |
| empty docs | 0 | 0 |

## 7. No-leakage verification (M0, re-runnable)

1. **Split disjointness & coverage**: train/dev/confirm index sets are pairwise disjoint and
   cover every deduped line exactly once, in both domains — verified programmatically
   (`partitions.py::verify_split_integrity`, 0 problems).
2. **No thread straddles splits** within a domain; **no shared thread is in different splits
   across domains** (17,279 shared threads checked).
3. **No exact-duplicate straddling**: dedup precedes splitting, so an exact duplicate of any
   train document cannot appear in dev/confirm within its domain.
4. **Eval pools are drawn from dev/confirm only** and are frozen with SHA; confirm is read once
   at M3 verdict (sealed-eval discipline, v1.1 §9).
5. Residual risk (recorded, accepted): near-duplicates across splits are not checked (exact-dup
   only); translation overlap across domains is intentional (it is the experimental axis).

## 8. Diagnostic snapshot (M0 Gate 3C; no mechanism claims)

Untrained 30M model, fp32, 8 batches × 8×256 per domain: cross-domain gradient cosine mean 0.103
[−0.044, 0.376]; conflict ratio P(g_A·g_B<0) = 0.1875; ‖g_A‖/‖g_B‖ ≈ 1.04. Domains are neither
aligned nor maximally conflicting.
