# DATASET_PROVENANCE (FR_G2_DATA_DISTRIBUTION_GENERALIZATION_V01)

Created: 2026-09-16T14:33:31Z (frozen BEFORE any G2 response result).

## D0 (frozen G1a condition, read-only, never rerun)

- Path: `outputs/e3_confirmatory/v07/e3c_subset.jsonl`
- SHA256: `1e657af9baae0784e582d82ef936160473aa069878e422ef48683a718056b0f4`
- Records: 4096; bytes: 3093545
- Schema keys: ['text']
- Note: D0 is the 4096-record G1a pool (e3c_subset.jsonl), NOT the full
  pretrain_t2t_mini.jsonl; the frozen G1a config/manifest governs (B1).

## D1 (pretrain_hq.jsonl, MiniMind official asset)

- Source: https://huggingface.co/datasets/jingyaogong/minimind_dataset
- Revision: `3a9bbe611244811e45b96661a84047f112c8ed01` (pre-deletion official revision, uploaded 2026-01-17; file deleted from
HEAD 2026-03-26 commit 3d8cf7da91c2 — recorded for transparency)
- LFS SHA256 (HF): `82b19375460d44feb15937ea3712121b5292271cf18eb87407b6d3e27427c18b`
- Local path: `outputs/generalization_g2_data_distribution_v01/data/pretrain_hq.jsonl`
- Local SHA256: `82b19375460d44feb15937ea3712121b5292271cf18eb87407b6d3e27427c18b` (verified equal to LFS oid)
- Records: 1,413,103; bytes: 1,627,188,411
- Schema keys: ['text'] (identical to D0 -> no adapter; B4)

## B2 selection audit

- Priority rule: (1) MiniMind pretrain_hq.jsonl if compatible+sufficient+
  different corpus; (2) other local MiniMind official assets; (3) download
  to versioned dir with provenance; (4) else DATASET_UNAVAILABLE.
- Executed: rule (3). The file was absent locally and absent from current
  HEAD of the official HF repo, but present at the pre-deletion official
  revision, which was downloaded and hash-verified. No alternative corpus
  was ever compared on results; D1 was locked before training began.
- No random substitutes, no FineWeb/English, no 'scientific' dataset.

## Pool / partition freeze

- Pool: first 4096 indices of make_permutation(1,413,103, seed=60202601), sorted
  -> `outputs/generalization_g2_data_distribution_v01/data/d1_pool.jsonl` sha256 `5d4c5e6b181e7791f12ceff2635011f17ba6e81d208942f22e5b1a24ad85a882`
  No content filtering, no balancing, no D0 mixing, no content picking.
- Partition: build_partition(4096, seed=60202603, probe=32, eval=128)
  -> `outputs/generalization_g2_data_distribution_v01/partition_g2_d1.json` sha256 `63fed3888a2b91296c8471147f54f1f9ffa74fd2e6134e82d94aa9e39384f050`

## Duplicate / overlap diagnostic (B3)

- D0 exact duplicate lines: 0 / 4096 (0.000000)
- D1 exact duplicate lines: 820 / 1,413,103 (0.000580)
- 13-gram shingle Jaccard (sampled 150x150 pairs): mean 0.000000, max 0.003546
- Interpretation: these are descriptive properties of the corpora; nonzero
  overlap does NOT trigger a dataset switch.

## Token statistics sample (frozen tokenizer, both corpora)

- D0: {"n": 150, "token_len_mean": 114.93333333333334, "token_len_std": 22.459409073446512, "token_len_min": 30, "token_len_max": 126, "token_len_truncated_at_126_fraction": 0.7333333333333333, "char_len_mean": 265.6666666666667}
- D1: {"n": 150, "token_len_mean": 126.0, "token_len_std": 0.0, "token_len_min": 126, "token_len_max": 126, "token_len_truncated_at_126_fraction": 1.0, "char_len_mean": 464.32666666666665}
- literal '<|im_end|>' fraction: D0 0.0000, D1 1.0000 (same tokenizer path; no adapter)
