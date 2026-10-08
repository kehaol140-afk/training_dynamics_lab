# MUON_IMPLEMENTATION_AUDIT — M0 Gate 1

Date: 2026-09-26. Status: **APPROVED as primary implementation** (torch-first-party provenance;
verified against an independent reference). Artifact: `outputs/m0/muon_freeze.json`.

---

## 1. Source and provenance (frozen)

| Field | Value |
|---|---|
| Primary source | `torch.optim._muon.Muon`, exposed as `torch.optim.Muon` |
| Repository | PyTorch official wheel, **torch==2.11.0** (CUDA 13.0), conda env `fr_minimind` |
| Commit hash | n/a (binary wheel). In-file provenance permalink to upstream: `github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L16` |
| File imported | `/home/b205/miniforge3/envs/fr_minimind/lib/python3.11/site-packages/torch/optim/_muon.py` |
| File SHA256 | `38e95918424bd26604f786195a4aa72afb60674b4ac9ccbde777974013cb1f52` |
| Import path used by this project | `from torch.optim import Muon` (no local copy, no patching, no wrapping of the update rule) |
| Registration | `docs/PROVENANCE.md` §3; constants echoed in `src/muon_transition/config.py::MUON_FREEZE` |

Registered alternatives (NOT used):

- LlamaFactory `third_party/muon/muon.py` (Moonlight/KellerJordan hybrid), repo commit
  `7af909522a951e3ad9f022ea6f88b6755257eaa5` — registered backup.
- skewadam `train.py` variant, commit `50f3da6…` — **rejected** (no Nesterov, no scaling rule,
  shape-heuristic AdamW fallback hardcoded for another model, bf16 stochastic rounding).

Provenance quality: first-party PyTorch prototype optimizer inside the exact torch version already
frozen for the experiment environment. No network needed; wheel pin == provenance pin.
Upgrade/patch of torch **invalidates this freeze** and requires re-audit.

## 2. Muon update rule (as implemented, `_muon.py:304-318`)

For each 2D parameter `θ` with gradient `g_t` and momentum buffer `B_t`:

```
B_t     = μ · B_{t-1} + (1 - μ) · g_t          # buf.lerp_(grad, 1 - momentum)
u_t     = (1 - μ) · g_t + μ · B_t              # nesterov=True: grad.lerp(buf, momentum)
                                                   nesterov=False: u_t = B_t
O_t     = NewtonSchulz5(u_t)                   # orthogonalization, bf16 internal
lr_adj  = lr · sqrt(max(1, A/B))               # adjust_lr_fn=None -> "original" rule
θ_t     = θ_{t-1} · (1 - lr · λ)               # decoupled weight decay, UNadjusted lr
θ_t    -= lr_adj · O_t
```

Note the buffer uses `(1-μ)`-weighted gradient accumulation (`lerp`), while the class docstring
writes `B_t ← μB + g`. Because Newton-Schulz is scale-invariant (verified:
`test_newton_schulz_scale_invariance`, cosine 1.0000), the two bookkeepings produce the same
update direction — verified numerically in `tests/test_muon_reference.py`
(documented-variant vs implementation: cosine ≥ 0.9999).

## 3. Momentum handling

- Single momentum buffer per 2D param, created lazily as `zeros_like(p.grad)`
  (`_init_group`, `_muon.py:156-159`) → fp32 when grads are fp32.
- momentum default 0.95; **frozen at 0.95**.
- No dampening parameter; complex/sparse params rejected.

## 4. Nesterov behavior

`nesterov=True` (frozen): update uses `u = (1-μ)·g + μ·B_t` (extrapolated gradient+buffer),
then orthogonalizes `u`. `nesterov=False` would use the buffer directly. Only the orthogonalized
combination enters the step; there is no second momentum pass after the update.

## 5. Newton-Schulz iteration (`_muon.py:31-70`)

- Quintic polynomial, coefficients `(a,b,c) = (3.4445, -4.7750, 2.0315)` (Keller Jordan constants,
  cited in-file with permalink) — **frozen**.
- **5 steps** (`ns_steps=5` default) — **frozen**. Hard cap <100 steps.
- 2D inputs only (constructor raises `ValueError` otherwise — structurally enforces our routing
  discipline).
- Transpose so rows ≤ cols; spectral normalization `X /= max(||X||_F, eps)` with `eps=1e-7`
  — **frozen**; 5 iterations of `X ← aX + (b(A) + c(A²))X` with `A = XXᵀ`.
- Produces `≈ U S' Vᵀ` with `S'ᵢᵢ ~ U(0.5, 1.5)` (in-file docstring): not exact SVD — by design.

## 6. Scaling rule (`_muon.py:73-84`)

Two selectable rules; we freeze the constructor default:

- `adjust_lr_fn=None` → **"original"**: `lr_adj = lr · sqrt(max(1, A/B))` (Keller Jordan).
  **FROZEN CHOICE.**
- `"match_rms_adamw"`: `lr_adj = 0.2 · sqrt(max(A,B)) · lr` (Moonlight). Not used; registered.

Important detail: the decoupled weight-decay factor uses the **unadjusted** `lr`, while the update
uses `lr_adj`. Both arms of our experiment implement decay as `θ ← θ(1 − lr·λ)` per step, so the WD
*semantic* (decoupled, lr-coupled λ) is identical across optimizers; magnitudes differ with each
arm's lr — same as any two optimizers sharing a λ grid. Recorded for the review.

## 7. dtype behavior

- Parameters: fp32 master weights (our loop). Muon accepts any float dtype.
- NS iteration runs internally in **bfloat16** (`grad.bfloat16()`), regardless of param dtype;
  the orthogonalized update is returned in bf16 and added to the fp32 param via
  `param.add_(update, alpha=-lr_adj)` (upcast at the in-place add).
- Momentum buffer in grad dtype (fp32 here). No GradScaler interaction (bf16 autocast, no scaler,
  per protocol §1) — Muon does not use `found_inf`/scale semantics at all.

## 8. Optimizer state format

- Standard `torch.optim.Optimizer` state container: `state[p] = {"momentum_buffer": Tensor}`;
  `param_groups` carry `lr, weight_decay, momentum, nesterov, ns_coefficients, eps, ns_steps,
  adjust_lr_fn`.
- `state_dict()/load_state_dict()` standard → saved/restored by our checkpoint format
  `mte_state_v1` (`src/muon_transition/checkpoint.py`) inside `optimizer_states`; roundtrip +
  single-step bit-replay verified for both arms (Gate 4).
- No step counter in state (bias correction not applicable — Muon has none).

## 9. Verification evidence

| Check | Result |
|---|---|
| One-step vs independent fp64 reference (lerp-coded variant), 5 shapes | cosine ≥ 0.99994; max rel-Frobenius error 1.19% (bf16 NS arithmetic) |
| One-step vs documented math (`B ← μB + g`) | cosine ≥ 0.99994 (scale-invariance of NS) |
| NS scale invariance | cosine = 1.0000 |
| Non-2D rejection | `ValueError` raised |
| State format | exactly `{"momentum_buffer"}`; buffer = `(1-μ)·g` on first step |
| Group defaults vs frozen constants | momentum 0.95, nesterov True, ns_steps 5, coeff (3.4445, −4.7750, 2.0315), eps 1e-7, adjust_lr_fn None, wd 0.1 |
| Determinism in training loop | identical losses + parameter fingerprints across replays (Gate 4) |
| Stability on real data (15-step smoke) | trains at lr ∈ {0.001, 0.002, 0.005}; **unstable at lr 0.02** at micro-scale (see M0_REPORT §finding F3) |

## 10. Verdict

Gate 1 **PASS**: provenance fixed (torch 2.11.0 wheel + SHA256 + upstream permalink), semantics
fully audited, reference-verified, state format checkpoint-compatible. The lr-0.02 instability is a
hyperparameter-anchor finding for review (F3), not an implementation defect.
