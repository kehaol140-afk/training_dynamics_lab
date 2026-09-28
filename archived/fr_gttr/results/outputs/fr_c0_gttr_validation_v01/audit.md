# C0-GTTR audit.md — interface & semantics record (frozen before GPU)

Experiment: FR_C0_GTTR_VALIDATION_V01 · 2026-09-21
Companion: `config.json` (sha256 e00ce243cd05527b38c63e1a7c81beead1ee89396b3f025399ffbb41892bd1a5), `live_manifest.json`.
Everything below was read/verified on the live code in this repo on 2026-09-21. Imports only; no redefinitions of frozen primitives (a0c `clip_jvp`/`adamw_tangent_step` are import-only by mandate).

## 1. Model source and build

- Runtime builder: `fr.a0k_closed_loop_runner_v01.build_live_runtime(seed, device)` — research/src/fr/a0k_closed_loop_runner_v01.py
  - Config: `research/configs/e3c_confirmatory_v07.json` (dtype=bfloat16, batch_size=4, accumulation_steps=2, max_seq_len=128, grad_clip=1.0, learning_rate=5e-4, planned_total_optimizer_steps=128).
  - Asserts bf16 / accum=2 / clip=1.0 (config-drift guard).
  - Model: `minimind.model.model_minimind.MiniMindConfig` (hidden_size/num_hidden_layers from cfg, use_moe=False) via `trainer.trainer_utils.init_model(..., from_weight="none")`; `setup_seed(seed)` BEFORE init.
  - Dataset: `PretrainDataset(outputs/e3_confirmatory/v07/e3c_subset.jsonl, tokenizer, max_length=128)`; train pool = `partition_v07.json train_indices` (3,936 samples).
- Bank checkpoints for both cells: `outputs/e3_confirmatory/v07/seed_2030/states/state_step_046.pt` (step 46); loaded via `fr.full_state.load_full_training_state` (model+optimizer+scaler+RNG) — the same frozen loader B0/B0.5/A0T used.

## 2. Parameter groups and decay mask

- `torch.optim.AdamW(model.parameters(), lr=5e-4)` — torch defaults, enforced by `fr.native_adamw_guard.assert_native_adamw` (must be torch.optim.AdamW).
- Single param group; NO decay mask: `weight_decay=0.01` applies to ALL parameters (cfg has no weight_decay key; PyTorch default). betas=(0.9,0.999), eps=1e-8, amsgrad=False, maximize=False, capturable=False, differentiable=False, fused=False.

## 3. Precision and attention backend

- Model weights bf16; training forward under the autocast context from `minimind_v03_runtime.make_amp` (bf16, GradScaler disabled — asserted).
- Native training attention: model's own `flash` path = `F.scaled_dot_product_attention` when available (`config.flash_attn`), causal, dropout 0 in eval / training branch as configured.
- HVP (frozen primitive `a0g_decision_utility_v01._hvp`): forces `SDPBackend.MATH` via `_math_sdpa_context` for the double backward; `hvp_fp32=True` gate asserted (A0O convention). Returns `(hvp_by_name, gradient_by_name)`; the paired first gradient is recorded per HVP call as the math-SDPA proxy-vs-native gap check.
- Inner products for probes/readout are accumulated in fp64 on CPU/GPU; loss values recorded in fp32 (autocast CE mean).

## 4. Gradient clipping (exact semantics)

- Training: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` — PyTorch implementation: `coef = max_norm / (total_norm + 1e-6)`, `clip_coef_clamped = min(coef, 1)`, all p.grad multiplied by it. Recorded here as `g * min(tau/(||g||+1e-6), 1)` with `CLIP_DENOMINATOR_EPS=1e-6`.
- Infinitesimal JVP of the same map: `fr.a0c_adamw_response_propagation_v01.clip_jvp` (exact derivative, returns clip_active/kink_margin).
- Probe gradients: algebraic re-clip with the identical formula on detached probe grads (no optimizer involvement).

## 5. beta / eps / step semantics (frozen AdamW rewrite)

- Real optimizer step counting: `optimizer.state[p]['step']` = cumulative real optimizer steps since bank load (step 46 bank → first C0 step is real step 47).
- Bias correction: `1 - beta**t` with the REAL cumulative t (not a local counter).
- Rewrite: `bar_m_t = a_t * bar_m_{t-1} + (1 - a_t) * c_t`, `a_t = beta1*(1-beta1**t)/(1-beta1**(t+1))` (equivalent to native EMA at real t); `bar_v_t` same with beta2 on c_t^2. Adaptive: `m_hat/sqrt(v_hat)+eps`, decoupled decay `theta *= (1 - lr*wd)`.
- LR input channel: current log-LR input is `b_eta = [s_t, 0, 0]` — the realized displacement s_t (which already contains -eta); do NOT multiply by eta again anywhere in the Geometry chain.

## 6. Gradient accumulation and microbatch composition

- accumulation_steps=2; each optimizer step consumes exactly 2 microbatches.
- Microbatch = batch_size 4 samples × max_seq_len 128 tokens = 512 tokens; per optimizer step 1,024 tokens.
- Stream: `b0_run_v01.make_stream(seed,404)` → `derive_stream_seed(seed,404)` (e.g. 2030→7110, 2030's second stream 7111) then `batches_from_train_pool(train_indices, 4, stream_seed, 128)` — permutation without replacement.
- LR timing: `fr.native_schedule.set_native_micro_lr(optimizer, get_lr, global_micro_step, total_micro_steps=256, base_lr)` before EVERY micro forward. `get_lr(s,T,lr)=lr*(0.1+0.45*(1+cos(pi*s/T)))` — cosine 1.0×→0.1×. AdamW consumes the lr present at the LAST micro of the accumulation boundary.
- Arm semantics (A0T contract): per step `g['lr'] = snap['lr'] * exp(u)`, u∈{0,+0.05,−0.05}, persistent every step (same as `b0_run_v01.run_arm`).

## 7. Validation sets and random sources

- Split: `outputs/fr_decision_utility_v01/validation_split.json`; tensors built in `build_live_runtime` via `a0g._stack_all` (EVAL_BATCH_SIZE=8).
- `V_diag` = **v_score** fixed subset: FIRST 8 sample IDs of `validation_split['v_score']` (1 chunk, 1,024 tokens) — prefix-ID rule frozen in config, result-independent. NEVER mixed with v_eval.
- `V_eval` = frozen 64-sample v_eval (8 chunks), read-only at h=4/16/64 for historical cross-check vs A0T losses.
- Probe banks: train-distribution microbatches from `batches_from_train_pool` with dedicated seeds DEV=2026092101, AUDIT=2026092102 (8 probes × 2 micros each, no replacement). Verified CPU-side before GPU: 16 unique micros per bank, dev∩audit=0, zero exact-microbatch collision with streams 7110/7111; sample-level overlap 15/64 & 14/64 is expected same-pool sharing (the actual future batches are NOT revealed).
- RNG discipline: every diagnostic probe/HVP/eval wrapped in `fr.rng.capture_rng_state`/`restore_rng_state`; model mode (train/eval) restored after each eval-mode pass; p.grad untouched by `a0g._grad_on_batches` (autograd.grad); `fr.stream_rng.seed_future_stream` after restores when forward movement continues.

## 8. Frozen primitives imported (path → symbols used)

- research/scripts/a0g_decision_utility_v01.py → `_grad_on_batches`, `_val_loss_only`, `_hvp`, `_stack_all`
- research/scripts/a0h15_production_entry_v01.py → `_baseline_no_step`, `_clip_only`, `_step_once`, `_clip_and_step`, `_restore_and_seed`, `_endpoint_grad`
- research/src/fr/a0c_adamw_response_propagation_v01.py → `clip_jvp`, `adamw_tangent_step` (import-only)
- research/src/fr/a0k_closed_loop_runner_v01.py → `build_live_runtime`, `make_event_ctx`, `eval_both`
- research/scripts/b0_run_v01.py → `make_stream`, `cell_snapshot`, `run_arm`
- research/src/fr/stage_c_runtime.py → `batches_from_train_pool`
- research/src/fr/stream_rng.py → `derive_stream_seed`, `seed_future_stream`
- research/src/fr/rng.py → `capture_rng_state`, `restore_rng_state`
- research/src/fr/full_state.py → `load_full_training_state`, `save_full_training_state`

## 9. Budget logging points

- LAUNCH_ID-pattern budget log (as B0.5/B0.6): every optimizer step appends to `budget_log.jsonl`; counters {grad_evals, hvp_calls, forward_calls, eval_mode_forwards, shadow_steps} hard-capped (1,400 / 256 / 2,400 combined / — / separate) and checked BEFORE each diagnostic; wall-clock cap 3,600 s GPU-active per run phase; smoke=24 updates, main=384 updates, cumulative project 17,577→17,985 (cap 18,000; headroom 15).
- Shadow updates: pure algebraic U-map on clones; recorded separately; never call optimizer.step().
