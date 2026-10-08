# PROVENANCE — muon_transition_experiment asset registry

Date: 2026-09-26 (updated same day after M0 execution). Rule (inherited from
`optimizer_sensitivity/docs/LEGACY_PROVENANCE.md`): **no code or data enters this project without
a source path + SHA256 entry here.** Vendoring is verbatim copy + dual-side hash check; originals
are never modified.

---

## 1. Planned vendored code (legacy read-only sources, to be copied at M0 step 1)

| File (future `src/vendored_fr/`) | Source (read-only) | SHA256 (source, verified 2026-09-26) | Role |
|---|---|---|---|
| full_state.py | `legacy_fluctuation_response/research/src/fr/full_state.py` | `709334f8fb229fc38869787dc22f2f7a57a3d687276b6f2392564a90f0e5853f` | full training-state checkpoint (fr_state v0.2); to be extended for two-group Muon state with version bump |
| rng.py | `…/fr/rng.py` | `0203e0ef0e81e71b4c58ae2ec502579307655a85b3f9eab4566c3c1c77aad40d` | four-way RNG capture/restore |
| stream_rng.py, batch_manifest.py, batch_stream.py | `…/fr/` | to hash at vendor time | deterministic data streams / manifests |
| pilot_partition.py | `…/fr/pilot_partition.py` | `9fece66acd9a81e523597e24d7a08e8bd0c0754d4174eaaaeaae905892a72c9f` | disjoint split discipline (extend to per-domain train/dev/confirm) |
| eval_observables.py | `…/fr/eval_observables.py` | `f131c7f0f8a965f0c22af2939201c9520ffc45907498e6798bae1c208a386319` | fixed-pool per-token NLL (sum/count); extend to V_A/V_B/V_Q |
| hashing.py | `…/fr/hashing.py` | to hash at vendor time | stable_hash |

Legacy repo provenance: git HEAD `6ac4942eeedaf3ed0e675a7f67c0e5f99a5a64e8`, status clean,
post-migration SHA checks in `PATH_MIGRATION_20260926.md`.

## 2. Reference (read-only, not copied) assets

| Asset | Path | SHA256 (verified 2026-09-26) | Role |
|---|---|---|---|
| MiniMind model | `minimind_base/model/model_minimind.py` | `47f124f5f386a8cd76b488b36ecc3e00fc43e3894cdb3ba602420188d9ba8860` | model family; imported in place (read-only) or vendored at M0 — decide at review |
| MiniMind tokenizer | `minimind_base/model/tokenizer.json` | `71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668` | BPE vocab 6400, both domains |
| Native training loop (reference only) | `minimind_base/trainer/train_pretrain.py` | `0c630aa6b5dce11a35f9176ff8337930a72a79bffee0a9282a4378b8005955f8` | loop-ordering reference (AMP, accum, clip-1.0); experiment loop is new, shared by both arms |

## 3. Muon implementation — M0 resolution (frozen 2026-09-26)

**PRIMARY (frozen): `torch.optim._muon.Muon`** — official PyTorch optimizer in the experiment's
own env: torch==2.11.0 wheel, file
`/home/b205/miniforge3/envs/fr_minimind/lib/python3.11/site-packages/torch/optim/_muon.py`,
SHA256 `38e95918424bd26604f786195a4aa72afb60674b4ac9ccbde777974013cb1f52`, in-file upstream
permalink KellerJordan/Muon@`f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd`. Imported as
`from torch.optim import Muon`; not copied, not patched. Full audit:
`docs/MUON_IMPLEMENTATION_AUDIT.md`. **Torch upgrade invalidates this freeze and requires re-audit.**

Server-wide search 2026-09-26 (`find`/`grep` over `/home/b205`, all three conda envs):

1. Canonical standalone repo unreachable (GitHub/pip blocked by gateway) — **moot**: torch ships
   the optimizer; no network dependency remains.
2. **Registered backup:** LlamaFactory vendored Muon (`framework-audit/llamafactory/src/llamafactory/third_party/muon/muon.py`,
   Moonshot-Moonlight + KellerJordan hybrid), repo commit `7af909522a951e3ad9f022ea6f88b6755257eaa5`.
   Not used.
3. **Rejected:** skewadam `train.py` variant (`/home/b205/baselines/skewadam/`, commit
   `50f3da6bdb0d527c640ee60ec22302af5d4148eb`, file SHA256
   `88f277f6434f0beba984d2f9bd7d1e5622f3929429568611ca2da481a3edd543`): no Nesterov, no update
   scaling rule, shape-heuristic AdamW fallback hardcoded for another model, bf16 stochastic
   rounding. Never used.

## 4. Data candidates (M0 audit done 2026-09-26; final selection pending review)

| Candidate | Path | Lines | SHA256 (verified 2026-09-26) | M0 notes |
|---|---|---|---|---|
| **Domain A (EN), primary** | `/home/b205/FDA_INIT/pretrained_models/sharegpt_datasets/common_en_70k.jsonl` | 79,107 | `7a61fda234ff634d35c5bb42b8ff761f0c1b54a895fec02ef411a9a212f31797` | **23.1% exact dups → deduped to 60,861 docs / 104.3M tokens**; 18,700 threads, translation-paired with B |
| **Domain B (ZH), primary** | `…/common_zh_70k.jsonl` | 76,399 | `a78dfae704e5ca29f18b881b0d298f4bf401093bf33d4ea993dd5728e591c565` | 2.4% dups → 74,591 docs / 80.1M tokens; 18,486 threads, all shared with A |
| Backup: topic axis EN | `…/computer_en_26k.jsonl` | 20,692 | `f387fe1dd3fe4bbac1be473f8d0ed64916642ee85c5d584e76bc33853096949f` | unaudited |
| Backup: topic axis ZH | `…/computer_zh_26k.jsonl` | 20,692 | `978dd4e762e769c577fb7dd88aea38e42b3dc3511936373af03e926c93b0a7ff` | unaudited |
| Rejected backup | `minimind_base/dataset/pretrain_t2t_mini.jsonl` | 1,270,238 | (1.2G; hash at need) | single-domain ~90% ZH, no domain metadata |

M0 audit results (full detail `outputs/m0/data_audit/data_report.json`):
- Thread structure discovered: `conversation_id` is a thread (session) id with 1–116 line-exchanges
  each; EN/ZH threads are line-by-line translation pairs. Split unit = thread (`mte_split_v2_thread`,
  seed 777, 94/3/3, integrity verified 0 problems, manifest sha `e62864ad…`).
- Dedup policy frozen: exact normalized-text, keep-first, within domain, pre-split (mandatory —
  see M0_REPORT F1).
- Eval pools frozen: first 512 windows per domain per dev/confirm split (130,560 tokens each,
  equal across domains); `outputs/m0/data_audit/eval_pools.pt`.
- Upstream provenance of the pair remains second-hand (inherited from `FDA_INIT` local project; no
  network to re-download or verify upstream). Open review item — replacement, if required, happens
  via protocol amendment BEFORE M1.
- MiniMind tokenizer (6400 BPE): Domain A coverage 78.4%, Domain B 98.1%; reserved ids {0,1,2}
  filtered from encodings; packing rule frozen (BOS(1)-prefixed windows, DOC_SEP(0) between docs).

## 5. Environment provenance

- conda env `fr_minimind` (`/home/b205/miniforge3/envs/fr_minimind/bin/python`): torch 2.11.0+cu,
  transformers 4.57.6, datasets 5.0.1, tokenizers 0.22.2 (2026-09-26).
- GPUs: 2 × RTX 5090 32G (one partially occupied at audit time — schedule with explicit
  `CUDA_VISIBLE_DEVICES` and probe first).
- Audit trail for this document: root `PROJECT_CONTEXT.md`, `PATH_MIGRATION_20260926.md`,
  `audits/FULL_SERVER_AUDIT_20260926/`, `optimizer_sensitivity/docs/LEGACY_PROVENANCE.md`.

## 6. Vendoring log (append-only; executed 2026-09-26, M0 step 1)

Legacy source repo: `legacy_fluctuation_response/` (git HEAD `6ac4942`, read-only, status clean
before and after). Copies under `src/vendored_fr/`; all hashes below are of BOTH source and copy
(verified identical, dual-side).

| date | file | sha256 (source = vendored) | match | role in this project |
|---|---|---|---|---|
| 2026-09-26 | vendored_fr/full_state.py | `709334f8fb229fc38869787dc22f2f7a57a3d687276b6f2392564a90f0e5853f` | ✅ | reference pattern for `mte_state_v1` checkpoint (RNG-last restore, atomic write) |
| 2026-09-26 | vendored_fr/rng.py | `0203e0ef0e81e71b4c58ae2ec50e579307655a85b3f9eab4566c3c1c779ad40d` | ✅ | four-way RNG capture/restore (used by checkpoint save/load) |
| 2026-09-26 | vendored_fr/stream_rng.py | `242956de271a8b839ddb3b39cc7e63d48b63ed253841255580d34e39ecd867a4` | ✅ | stream-seed derivation (M1 data stream planning) |
| 2026-09-26 | vendored_fr/batch_manifest.py | `6bf1703f0e09f4b713113bfa05ad3eccb83119c8ad0838d4a1df5688ac932ffb` | ✅ | batch-manifest discipline (M1 mixture manifests) |
| 2026-09-26 | vendored_fr/batch_stream.py | `2614f7bd1ad95cfb3d3cf10b37da8c8c7c683eba2f45131d946ce80430c6d9bb` | ✅ | deterministic batch streams (M1) |
| 2026-09-26 | vendored_fr/pilot_partition.py | `9fece66acd9a81e523597e24d7a08e8bd0c0754d4174eaaaeaae905892a72c9f` | ✅ | reference for split discipline (thread-split implemented fresh in `partitions.py`) |
| 2026-09-26 | vendored_fr/hashing.py | `ccc94714b4c2c00c5d5c3bd05fffad3b2c4da811ca8ff0e802651b0daddf7536` | ✅ | stable_hash for split manifests |

Deviation from protocol v1.1 M0 step 1 (logged): `eval_observables.py` was **not** vendored — its
import chain drags `minimind_v03_runtime` (legacy model runtime). Sum/count per-domain evaluation
was implemented fresh in `src/muon_transition/eval_nll.py` following the same semantics
(fixed pool, per-token CE, RNG-protected); the legacy module remains registered read-only above.
New checkpoint format `mte_state_v1` (`src/muon_transition/checkpoint.py`) follows full_state's
pattern with a registered version bump rather than extending the vendored AdamW-specific saver.
