#!/usr/bin/env python3
"""FR-C1 GTTR probe-validation runner (protocol FR_C0X_C1 §5-§9).

Fixed design: C0 cells R/C (same checkpoint step 46, streams 7110/7111 from C0
config), anchors per cell: A/t0 (h=1), Bp/t16 (h=17), A/t48 (h=49).
A-path replays 49 real updates, Bp-path 17 -> 132 main updates + 8 smoke.

New probe banks: 32 dev + 32 audit (seeds fixed below), disjoint from C0 banks
and from rollout batches by sample-ID check (state_provenance).

Budget (ROUND-TOTAL, no per-process reset; failures/rechecks count):
  optimizer_updates <= 192 | grad_evals <= 1400 | hvp_calls <= 416
  micro_forward_calls <= 3200 | gpu wall <= 5400 s
Two counter sets kept simultaneously (effective-batch evals and micro-forward
calls). Old 18,000 ledger untouched; C1 billed separately.

Leakage order per anchor (enforced by a global order_index; the analyzer
verifies predictions.order_index < every audit-touching record):
  1. dev+audit g0 at theta_t; V_diag(theta_t), q_pre
  2. real clip+step -> s_t; post moments S+; V_diag(theta+s), q+ (fixed main
     readout direction, determined BEFORE any audit endpoint)
  3. dev HVPs at theta_t tangent s_t -> P_D, Q_D (+stop-v); predictions at
     K=8/16/32 SAVED to disk (sha256 over group sums) BEFORE any audit read
  4. CONDITIONED (post-save) diagnostics: audit HVPs; audit g1; dev g1; P_A,
     Y_A, Q_A, Z_A, Y_D, Z_D, stop-v realized pairs, same-probe residuals
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

import torch  # noqa: E402

import fr.c0_gttr_validation_v01 as c0  # noqa: E402
import fr.c1_gttr_probe_validation_v01 as c1  # noqa: E402
from fr.rng import capture_rng_state, restore_rng_state  # noqa: E402
from fr.minimind_v03_runtime import raw_model  # noqa: E402

C0_OUT = PROJECT_ROOT / "outputs" / "fr_c0_gttr_validation_v01"
OUT = PROJECT_ROOT / "outputs" / "fr_c1_gttr_probe_validation_v01"
RUNS = OUT / "runs"
BUDGET_LOG = OUT / "budget_log.jsonl"

CFG0 = json.loads((C0_OUT / "config.json").read_text())
CELLS = {"R": CFG0["cells"]["R"], "C": CFG0["cells"]["C"]}
BANK_CKPT = PROJECT_ROOT / CELLS["R"]["checkpoint"]

C1CFG = {
    "experiment_id": "FR_C1_GTTR_PROBE_VALIDATION_V01",
    "protocol": "FR_C0X_C1_GTTR_Offline_and_Probe_Validation_Claude_Code.md",
    "anchors": {"R": [("A", 0), ("Bp", 16), ("A", 48)],
                "C": [("A", 0), ("Bp", 16), ("A", 48)]},
    "executed_anchors": {"R": [("A", 0), ("Bp", 16)],
                         "C": [("A", 0), ("Bp", 16)]},
    "coverage_note": ("协议固定 6 anchor；本轮因计入失败的首跑（32 HVP）与 smoke "
                      "(16 HVP) 后全量 6 anchor 需 432>416 HVP，超出 round cap。"
                      "对称减覆盖：每 cell 测 A/t0 + Bp/t16（t∈{0,16}），"
                      "A/t48 两 anchor 本轮不测；round HVP 预算 304/416。"),
    "path_lengths": {"A": 49, "Bp": 17},
    "probe_banks": {"n_dev": 32, "n_audit": 32, "micros_per_probe": 2,
                    "dev_seed": 2026092103, "audit_seed": 2026092104},
    "k_prefixes": [8, 16, 32],
    "half_split": 16,
    "budget": {
        "optimizer_updates_max": 192, "grad_evals_max": 1400,
        "hvp_calls_max": 416, "micro_forward_calls_max": 3200,
        "gpu_wall_s_max": 5400,
        "planned": {"smoke_updates": 8, "main_updates": 132},
        "accounting": ("round-total; failures/rechecks/salvage count; "
                       "effective-batch evals and micro-forward calls kept as "
                       "two simultaneous sets; old 18,000 ledger untouched, "
                       "C1 billed separately; never divide C1 totals by 18000"),
    },
    "precision": {"model_dtype": "bfloat16", "inner_products": "fp64",
                  "hvp_fp32": True},
    "clip": {"tau_source": "rt cfg grad_clip (same as C0)",
             "CLIP_DENOMINATOR_EPS": c0.CLIP_DENOMINATOR_EPS},
    "optim": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "wd": 0.01},
    "stop_v": ("P frozen from the AUDIT bank's mean clipped g0 (C0 pair-baseline "
               "convention, a2v/b2v at t_after); applied to both banks for "
               "within-anchor comparability; ablation only, not an optimizer."),
    "d_c_to_u": ("local c->U derivative, dev bank, post-update shadow state; "
                 "clip-Jacobian factor not included; diagnostic only"),
    "memory": ("per-probe vectors freed after accumulation; only fp64 group "
               "sums and scalar projection lists persist within an anchor"),
}

CAPS = C1CFG["budget"]
B1C, B2C, EPS_ADAM, WD = 0.9, 0.999, 1e-8, 0.01
N_PROBES = 32
# groups carrying K-prefix snapshots and fixed-ID half-banks
MAIN_GROUPS = ["P_D", "Y_A", "P_A", "Q_D", "Z_A", "Q_A"]
# accumulated at K=32 only; stop-v lives in the SHADOW channel (it changes U,
# not the gradient), so stop-v groups are Qstop (prediction pair) and
# Zstop (realized pair); the field channel is stop-v-invariant by construction
EXTRA_GROUPS = ["Y_D", "Z_D", "Qstop_D", "Qstop_A", "Zstop_D", "Zstop_A"]
ALL_GROUPS = MAIN_GROUPS + EXTRA_GROUPS

_counters = None
_t_start = None
_t_shadow = {"s": 0.0}
_order = None
_vdiag_tensors = None


def _new_counters():
    return {"optimizer_updates": 0, "grad_evals": 0, "hvp_calls": 0,
            "micro_forward_calls": 0, "shadow_steps": 0, "shadow_seconds": 0.0}


def _log(phase, cell, arm, h, note=""):
    rec = {"ts": time.time(), "phase": phase, "cell": cell, "arm": arm, "h": h,
           "wall_run_s": round(time.perf_counter() - _t_start, 3),
           **_counters, "note": note}
    with BUDGET_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _check_budget(phase):
    c = _counters
    if c["optimizer_updates"] > CAPS["optimizer_updates_max"]:
        raise SystemExit("BUDGET_BLOCKED: optimizer_updates")
    if c["grad_evals"] > CAPS["grad_evals_max"]:
        raise SystemExit("BUDGET_BLOCKED: grad_evals")
    if c["hvp_calls"] > CAPS["hvp_calls_max"]:
        raise SystemExit("BUDGET_BLOCKED: hvp_calls")
    if c["micro_forward_calls"] > CAPS["micro_forward_calls_max"]:
        raise SystemExit("BUDGET_BLOCKED: micro_forward_calls")
    if time.perf_counter() - _t_start > CAPS["gpu_wall_s_max"]:
        raise SystemExit("BUDGET_BLOCKED: gpu wall")


# ---------------------------------------------------------------------------
# counted diagnostic wrappers (two counter sets; RNG-wrapped; mode restored)
# ---------------------------------------------------------------------------

def probe_grad(rt, tensors):
    import a0g_decision_utility_v01 as a0g
    _counters["grad_evals"] += 1
    _counters["micro_forward_calls"] += len(tensors)
    rng = capture_rng_state()
    try:
        return a0g._grad_on_batches(rt["torch"], rt["model"], rt["named"], tensors,
                                    rt["autocast_ctx"],
                                    int(rt["cfg"]["accumulation_steps"]))
    finally:
        restore_rng_state(rng)


def probe_hvp(rt, tensors, tangent):
    import a0g_decision_utility_v01 as a0g
    _counters["hvp_calls"] += 1
    _counters["micro_forward_calls"] += len(tensors)
    hvp_ctx = nullcontext() if C1CFG["precision"]["hvp_fp32"] else rt["autocast_ctx"]
    rng = capture_rng_state()
    try:
        return a0g._hvp(rt["torch"], rt["model"], rt["named"], tensors,
                        tangent, hvp_ctx, int(rt["cfg"]["accumulation_steps"]))
    finally:
        restore_rng_state(rng)


def v_diag_q(rt):
    """V_diag value + gradient in ONE eval-mode forward+backward (1 chunk)."""
    _counters["grad_evals"] += 1
    _counters["micro_forward_calls"] += 1
    torch = rt["torch"]
    rng = capture_rng_state()
    model = rt["model"]
    was_training = model.training
    params = [p for _, p in rt["named"]]
    try:
        model.eval()
        for x, y in _vdiag_tensors:
            with rt["autocast_ctx"]:
                result = model(x, labels=y)
            loss = (result.loss + result.aux_loss).float()
            value = float(loss.detach().cpu())
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            q = {}
            for (name, p), g in zip(rt["named"], grads):
                q[name] = (p.detach().new_zeros(p.shape) if g is None
                           else g.detach().clone())
            break
    finally:
        if was_training:
            model.train()
        restore_rng_state(rng)
    return value, q


def _shadow(fn, *a, **k):
    t0 = time.perf_counter()
    try:
        _counters["shadow_steps"] += 1
        return fn(*a, **k)
    finally:
        dt = time.perf_counter() - t0
        _counters["shadow_seconds"] += dt
        _t_shadow["s"] += dt


# ---------------------------------------------------------------------------
# shared helpers (semantics borrowed from c0_run_v01; C0 files only read)
# ---------------------------------------------------------------------------

def _stack(rt, micros):
    import a0g_decision_utility_v01 as a0g
    return a0g._stack_all(rt["dataset"], micros, rt["device"])


def materialize_bank(rt, seed):
    from fr.stage_c_runtime import batches_from_train_pool
    cfg = rt["cfg"]
    micros = batches_from_train_pool(rt["train_indices"], int(cfg["batch_size"]),
                                     int(seed), 2 * N_PROBES)
    return [[list(map(int, m)) for m in micros[2 * j: 2 * j + 2]]
            for j in range(N_PROBES)]


def make_ctx():
    return SimpleNamespace(
        checkpoint_path=BANK_CKPT,
        bank_meta={"global_micro_step": 46 * 2, "total_micro_steps": 256,
                   "global_optimizer_step": 46,
                   "planned_total_optimizer_steps": 128},
        candidate_batches=lambda stream_seed: {})


def make_stream(rt, stream_seed):
    import b0_run_v01 as b0
    return b0.make_stream(rt, int(stream_seed), 128)


def v_diag_tensors_from_split(rt):
    import a0g_decision_utility_v01 as a0g
    split = json.loads((PROJECT_ROOT / "outputs" / "fr_decision_utility_v01" /
                        "validation_split.json").read_text())
    ids = sorted(map(int, split["v_score"]))[:8]  # frozen prefix-ID rule (as C0)
    return a0g._stack_all(rt["dataset"], [ids], rt["device"])


def snap(rt, name):
    from fr.full_state import save_full_training_state
    p = RUNS / name
    save_full_training_state(p, rt["model"], rt["optimizer"], rt["scaler"],
                             metadata={"protocol": "FR_C1_ANCHOR_SNAPSHOT"})
    return p


def load(rt, path):
    from fr.full_state import load_full_training_state
    load_full_training_state(path, rt["model"], rt["optimizer"], rt["scaler"],
                             restore_rng=False)
    rt["model"].train()


def post_moments(rt, t_after):
    opt_state = rt["optimizer"].state
    m_hat, v_hat = {}, {}
    for n, p in rt["named"]:
        st = opt_state[p]
        m_hat[n] = (st["exp_avg"] / (1 - B1C ** t_after)).clone()
        v_hat[n] = (st["exp_avg_sq"] / (1 - B2C ** t_after)).clone()
    return m_hat, v_hat


def _rel(a, b):
    nb = c0.norm_td(b)
    return c0.norm_td(c0.sub_td(a, b)) / nb if nb > 0 else float("nan")


def to_host(td):
    """Offload a tensor-dict to host RAM (fp32 clones). GPU-memory discipline:
    the mandated GPU is shared, so long-lived vectors never live on device."""
    return {n: t.detach().to("cpu", torch.float32).clone() for n, t in td.items()}


def to_gpu(rt, td):
    return {n: t.to(rt["device"]) for n, t in td.items()}


def _mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    se = (math.sqrt(sum((x - m) ** 2 for x in xs) / (n * (n - 1)))
          if n > 1 else float("nan"))
    return {"mean": m, "se": se, "n": n}


# ---------------------------------------------------------------------------
# bank disjointness (CPU index math; verified, not assumed)
# ---------------------------------------------------------------------------

def check_banks(rt, banks, stream_r, stream_c):
    dev, audit = banks
    dev_flat = {tuple(m) for probe in dev for m in probe}
    audit_flat = {tuple(m) for probe in audit for m in probe}
    report = {"dev_audit_overlap": sorted(dev_flat & audit_flat)}
    c0_banks = json.loads((C0_OUT / "runs" / "probe_banks.json").read_text())
    c0_flat = set()
    for label in ("dev", "audit"):
        for probe in c0_banks[label]["micros"]:
            for m in probe:
                c0_flat.add(tuple(m))
    report["c0_overlap"] = sorted((dev_flat | audit_flat) & c0_flat)
    used = set()
    for st, n_steps in ((stream_r, 49), (stream_c, 49)):
        for b in st["batches"][: 2 * n_steps]:
            used.add(tuple(int(x) for x in b))
    report["rollout_overlap"] = sorted((dev_flat | audit_flat) & used)
    report["dev_probes"] = len(dev)
    report["audit_probes"] = len(audit)
    report["status"] = ("PASS" if not (report["dev_audit_overlap"]
                                       or report["c0_overlap"]
                                       or report["rollout_overlap"])
                        else "FAIL")
    return report


# ---------------------------------------------------------------------------
# anchor machinery
# ---------------------------------------------------------------------------

def _sha_of_means(prefix_snaps_by_group):
    """SHA256 over canonical fp64 bytes of the K=8/16/32 group SUM vectors.
    Predictions are fixed by this hash; post-save backfill would change it.
    Frozen prediction groups per protocol §7.1: mean(P_D), mean(Q_D)."""
    h = hashlib.sha256()
    for g in sorted(prefix_snaps_by_group):
        snaps = prefix_snaps_by_group[g]
        for k in C1CFG["k_prefixes"]:
            snap = snaps[k]
            for n in sorted(snap["sum"]):
                h.update(snap["sum"][n].detach().cpu().numpy().tobytes())
            h.update(str(snap["n"]).encode())
    return h.hexdigest()


def run_anchor(rt, snap_pre, cell, arm, h, t_pre, native_grad, snap_step,
               banks, predictions_path, records_path):
    """C1 anchor measurement, leak-ordered. Memory discipline: every
    long-lived vector lives on HOST RAM (to_host); the GPU holds only the
    model, optimizer, q+/s_hat and one probe's transient tensors."""
    import a0h15_production_entry_v01 as pe
    dev_micros, audit_micros = banks
    torch = rt["torch"]
    tau = float(rt["cfg"]["grad_clip"])
    eta_t = float(rt["optimizer"].param_groups[0]["lr"])
    mult = eta_t / float(snap_step["lr"])

    # --- stage 1: starting points at theta_t (pre-update) --------------------
    dev_g0 = [to_host(probe_grad(rt, _stack(rt, dev_micros[j])))
              for j in range(N_PROBES)]
    audit_g0 = [to_host(probe_grad(rt, _stack(rt, audit_micros[j])))
                for j in range(N_PROBES)]
    v_pre, q_pre = v_diag_q(rt)

    pe._clip_only(rt, native_grad, snap_step)
    pe._step_once(rt)
    _counters["optimizer_updates"] += 1
    theta_post = {n: p.detach().float() for n, p in rt["named"]}
    theta_pre = {n: snap_step["theta_start"][n].float() for n, _ in rt["named"]}
    s_t = {n: theta_post[n] - theta_pre[n] for n in theta_post}
    s_norm = c0.norm_td(s_t)
    s_hat = (c0.scale_td(s_t, 1.0 / s_norm) if s_norm > 0 else None)
    t_after = int(snap_step["step_after"])
    m_hat, v_hat = post_moments(rt, t_after)
    lr_next = float(rt["get_lr"]((46 + h) * 2, 256,
                                 float(rt["cfg"]["learning_rate"]))) * mult
    v_post, q_plus = v_diag_q(rt)          # fixed main direction, pre-audit

    # shadow baselines (full-v and stop-v), computed on GPU then offloaded
    u_state = dict(m_hat=m_hat, v_hat=v_hat, beta1=B1C, beta2=B2C,
                   t_prev=t_after, eps=EPS_ADAM, lr=lr_next, wd=WD,
                   theta=theta_post)
    dev_c0 = [to_host(c0.apply_clip(to_gpu(rt, g), tau)) for g in dev_g0]
    audit_c0 = [to_host(c0.apply_clip(to_gpu(rt, g), tau)) for g in audit_g0]
    dev_U0 = [to_host(_shadow(c0.u_map, to_gpu(rt, g), **u_state, tau=tau))
              for g in dev_g0]
    audit_U0 = [to_host(_shadow(c0.u_map, to_gpu(rt, g), **u_state, tau=tau))
                for g in audit_g0]

    # stop-v pair-baseline (C0 convention): P frozen from the AUDIT bank's
    # mean clipped g0 (a2v/b2v at t_after); ablation only, never an optimizer
    c_bar = c0.ybar([to_gpu(rt, g) for g in audit_c0])
    a2v, b2v = c0.ema_coeffs(B2C, t_after)
    P_fixed = {n: 1.0 / (torch.sqrt(a2v * v_hat[n] + b2v * c_bar[n] * c_bar[n])
                         + EPS_ADAM) for n in c_bar}
    sv = dict(m_hat=m_hat, v_hat=v_hat, beta1=B1C, beta2=B2C, t_prev=t_after,
              eps=EPS_ADAM, lr=lr_next, wd=WD, theta=theta_post, tau=tau,
              P_fixed=P_fixed)
    sv0_d = [to_host(_shadow(c0.u_map_stop_v, to_gpu(rt, g), **sv))
             for g in dev_g0]
    sv0_a = [to_host(_shadow(c0.u_map_stop_v, to_gpu(rt, g), **sv))
             for g in audit_g0]

    snap_post = snap(rt, f"_c1_{cell}_{arm}_t{t_pre}_post.pt")
    load(rt, snap_pre)

    accs = {g: c1.GroupAcc(k_prefixes=C1CFG["k_prefixes"] if g in MAIN_GROUPS else (),
                           half_split=C1CFG["half_split"]) for g in ALL_GROUPS}
    proj = {g: {"q+": [], "s_hat": []} for g in
            ALL_GROUPS + ["resid_field_dev", "resid_field_audit"]}
    resid_norm = {"dev": [], "audit": []}
    sv_inner = {g: {"dev": [], "audit": []} for g in ("Q", "Z")}
    sv_diff = {g: {"dev": [], "audit": []} for g in ("Q", "Z")}

    def acc_and_project(gname, vec, j):
        accs[gname].add(vec, j)
        proj[gname]["q+"].append(float(c0.dot_td(vec, q_plus)))
        if s_hat is not None:
            proj[gname]["s_hat"].append(float(c0.dot_td(vec, s_hat)))

    # --- stage 2: dev predictions; SAVED BEFORE ANY AUDIT READ ---------------
    dev_P = [None] * N_PROBES
    for j in range(N_PROBES):
        g0 = to_gpu(rt, dev_g0[j])
        hv, _ = probe_hvp(rt, _stack(rt, dev_micros[j]), s_t)
        g1p = c0.predict_geom(g0, hv, s_t)
        P = c0.sub_td(c0.apply_clip(g1p, tau), to_gpu(rt, dev_c0[j]))
        Q = c0.sub_td(_shadow(c0.u_map, g1p, **u_state, tau=tau),
                      to_gpu(rt, dev_U0[j]))
        Qs = c0.sub_td(_shadow(c0.u_map_stop_v, g1p, **sv),
                       to_gpu(rt, sv0_d[j]))
        acc_and_project("P_D", P, j)
        acc_and_project("Q_D", Q, j)
        acc_and_project("Qstop_D", Qs, j)
        sv_inner["Q"]["dev"].append(float(c0.dot_td(Q, Qs)))
        sv_diff["Q"]["dev"].append(c0.norm_td(c0.sub_td(Q, Qs)))
        dev_P[j] = to_host(P)
        del g0, hv, g1p, P, Q, Qs
    order_pred = next(_order)
    pred_groups = ["P_D", "Q_D"]   # frozen §7.1: predictions are mean(P_D), mean(Q_D)
    pred_sha = _sha_of_means({g: accs[g].prefixes() for g in pred_groups})
    pred_rec = {"cell": cell, "arm": arm, "anchor": t_pre, "kind": "predictions",
                "sha256_fixed_predictions": pred_sha,
                "k_prefixes": C1CFG["k_prefixes"],
                "per_K": {g: {str(k): {"mean_norm": c1.mean_norm(sn),
                                       "se": c1.se_vec_of(sn)}
                              for k, sn in accs[g].prefixes().items()}
                          for g in pred_groups},
                "order_index": order_pred}
    with predictions_path.open("a") as f:
        f.write(json.dumps(pred_rec) + "\n")

    # --- stage 3: CONDITIONED diagnostics (post-save only) -------------------
    audit_P = [None] * N_PROBES
    for j in range(N_PROBES):
        g0 = to_gpu(rt, audit_g0[j])
        hv, _ = probe_hvp(rt, _stack(rt, audit_micros[j]), s_t)
        g1p = c0.predict_geom(g0, hv, s_t)
        P = c0.sub_td(c0.apply_clip(g1p, tau), to_gpu(rt, audit_c0[j]))
        Q = c0.sub_td(_shadow(c0.u_map, g1p, **u_state, tau=tau),
                      to_gpu(rt, audit_U0[j]))
        Qs = c0.sub_td(_shadow(c0.u_map_stop_v, g1p, **sv),
                       to_gpu(rt, sv0_a[j]))
        acc_and_project("P_A", P, j)
        acc_and_project("Q_A", Q, j)
        acc_and_project("Qstop_A", Qs, j)
        sv_inner["Q"]["audit"].append(float(c0.dot_td(Q, Qs)))
        sv_diff["Q"]["audit"].append(c0.norm_td(c0.sub_td(Q, Qs)))
        audit_P[j] = to_host(P)
        del g0, hv, g1p, P, Q, Qs

    load(rt, snap_post)
    for bank, P_list, key in (("dev", dev_P, "dev"), ("audit", audit_P, "audit")):
        micros = dev_micros if bank == "dev" else audit_micros
        c0_list = dev_c0 if bank == "dev" else audit_c0
        U0_list = dev_U0 if bank == "dev" else audit_U0
        sv0_list = sv0_d if bank == "dev" else sv0_a
        gname = (lambda base: f"{base}_{'D' if bank == 'dev' else 'A'}")
        for j in range(N_PROBES):
            g1 = probe_grad(rt, _stack(rt, micros[j]))
            Y = c0.sub_td(c0.apply_clip(g1, tau), to_gpu(rt, c0_list[j]))
            Z = c0.sub_td(_shadow(c0.u_map, g1, **u_state, tau=tau),
                          to_gpu(rt, U0_list[j]))
            Zs = c0.sub_td(_shadow(c0.u_map_stop_v, g1, **sv),
                           to_gpu(rt, sv0_list[j]))
            acc_and_project(gname("Y"), Y, j)
            acc_and_project(gname("Z"), Z, j)
            acc_and_project(gname("Zstop"), Zs, j)
            sv_inner["Z"][key].append(float(c0.dot_td(Z, Zs)))
            sv_diff["Z"][key].append(c0.norm_td(c0.sub_td(Z, Zs)))
            rd = c0.sub_td(to_gpu(rt, P_list[j]), Y)
            resid_norm[key].append(c0.norm_td(rd))
            proj[f"resid_field_{key}"]["q+"].append(float(c0.dot_td(rd, q_plus)))
            if s_hat is not None:
                proj[f"resid_field_{key}"]["s_hat"].append(
                    float(c0.dot_td(rd, s_hat)))
            del g1, Y, Z, Zs, rd

    d_proj = {"q+": [], "s_hat": []}
    for j in range(N_PROBES):
        D = c1.d_c_to_u(c=to_gpu(rt, dev_c0[j]), m_hat=m_hat, v_hat=v_hat,
                        beta1=B1C, beta2=B2C, t_prev=t_after, eps=EPS_ADAM,
                        lr=lr_next)
        d_proj["q+"].append(float(c0.dot_td(D, q_plus)))
        if s_hat is not None:
            d_proj["s_hat"].append(float(c0.dot_td(D, s_hat)))
    proj["d_c_to_u"] = d_proj

    # ---- assemble scalar record ----
    def gstats(gname_):
        snap_full = accs[gname_].snapshot()
        out = {"n": snap_full["n"], "mean_norm": c1.mean_norm(snap_full),
               "mean_sq_norm": c1.mean_sq_norm(snap_full),
               "se": c1.se_vec_of(snap_full)}
        for k, sn in accs[gname_].prefixes().items():
            out[f"K{k}_mean_norm"] = c1.mean_norm(sn)
            out[f"K{k}_se"] = c1.se_vec_of(sn)
            out[f"K{k}_mean_sq_norm"] = c1.mean_sq_norm(sn)
        for hn, hs in accs[gname_].halves().items():
            if hs["sum"] is not None:
                out[f"{hn}_mean_sq_norm"] = c1.mean_sq_norm(hs)
                out[f"{hn}_n"] = hs["n"]
        return out

    def pair_stats(acc_a, acc_b):
        """SSE/TSS of ||Abar-Bbar||^2 at each K prefix and each fixed half
        (scalars only — recoverable from group sums, no stored vectors)."""
        pa, pb = acc_a.prefixes(), acc_b.prefixes()
        ha, hb = acc_a.halves(), acc_b.halves()
        out = {}
        for k in acc_a.k_prefixes:
            if k not in pa or k not in pb:
                continue
            sa, sb = pa[k], pb[k]
            inner = c1.inner_means(sa, sb)
            out[f"K{k}"] = {"sse": c1.mean_sq_norm(sa) + c1.mean_sq_norm(sb)
                            - 2.0 * inner,
                            "tss": c1.mean_sq_norm(sb), "inner": inner}
        for hn in c1.HALF_IDS:
            if ha[hn]["sum"] is None:
                continue
            inner = c1.inner_means(ha[hn], hb[hn])
            out[hn] = {"sse": c1.mean_sq_norm(ha[hn]) + c1.mean_sq_norm(hb[hn])
                       - 2.0 * inner,
                       "tss": c1.mean_sq_norm(hb[hn]), "n": ha[hn]["n"]}
        return out

    record = {"cell": cell, "arm": arm, "anchor": t_pre, "h": h,
              "t_after": t_after, "eta_t": eta_t, "lr_next": lr_next,
              "tau": tau, "s_norm": s_norm,
              "v_pre": v_pre, "v_post": v_post,
              "q_pre_norm": c0.norm_td(q_pre), "q_plus_norm": c0.norm_td(q_plus),
              "groups": {g: gstats(g) for g in ALL_GROUPS},
              "pair_stats": {"field": pair_stats(accs["P_D"], accs["Y_A"]),
                             "shadow": pair_stats(accs["Q_D"], accs["Z_A"])},
              "same_probe_resid_norm_mean_se": {k: _mean_se(v)
                                                for k, v in resid_norm.items()},
              "stopv_pairs": {g: {b: {"inner_mean_se": _mean_se(sv_inner[g][b]),
                                      "diff_norm_mean_se": _mean_se(sv_diff[g][b])}
                                  for b in ("dev", "audit")}
                              for g in ("Q", "Z")},
              "fixed_dir_proj": {g: {d: c1.fixed_direction_stats(v)
                                     for d, v in proj[g].items()}
                                 for g in proj},
              "order_index_pred": order_pred,
              "order_index_audit": next(_order)}

    means = {g: c1.mean_of(accs[g].snapshot()) for g in MAIN_GROUPS}
    dec_f = c1.decompose(means["P_D"], means["Y_A"], means["P_A"])
    dec_s = c1.decompose(means["Q_D"], means["Z_A"], means["Q_A"])
    record["decomposition"] = {
        "field": {k: dec_f[k] for k in ("e_norm", "e_bank_norm", "e_local_norm",
                                        "cross", "add_identity_resid",
                                        "sq_identity_resid")},
        "shadow": {k: dec_s[k] for k in ("e_norm", "e_bank_norm", "e_local_norm",
                                         "cross", "add_identity_resid",
                                         "sq_identity_resid")}}
    with records_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    del accs, dev_P, audit_P, means, dev_g0, audit_g0
    del dev_c0, audit_c0, dev_U0, audit_U0, sv0_d, sv0_a
    return record


def rollout(rt, ctx, stream, cell, arm, u, T, anchor_ts, banks, phase,
            predictions_path, records_path, steps_path, diag_steps):
    import a0h15_production_entry_v01 as pe
    mult = math.exp(u)
    pe._restore_and_seed(rt, ctx, stream)
    for h in range(1, T + 1):
        _check_budget(phase)
        t0 = time.perf_counter()
        native_grad, _, snap_step = pe._baseline_no_step(
            rt, ctx, stream["batches"], h, want_snapshot=True)
        if mult != 1.0:
            for g in rt["optimizer"].param_groups:
                g["lr"] = float(snap_step["lr"]) * mult
        t_pre = h - 1
        row = {"cell": cell, "arm": arm, "h": h, "u": u,
               "lr_native": float(snap_step["lr"]),
               "lr_applied": float(rt["optimizer"].param_groups[0]["lr"])}
        if t_pre in anchor_ts:
            snap_pre = snap(rt, f"_c1_{cell}_{arm}_t{t_pre}_pre.pt")
            rec = run_anchor(rt, snap_pre, cell, arm, h, t_pre,
                             native_grad, snap_step, banks,
                             predictions_path, records_path)
            row.update({"anchor": t_pre, "s_norm": rec["s_norm"],
                        "v_pre": rec["v_pre"], "v_post": rec["v_post"]})
        else:
            pe._clip_only(rt, native_grad, snap_step)
            pe._step_once(rt)
            _counters["optimizer_updates"] += 1
            if diag_steps:
                v, q = v_diag_q(rt)
                row["v_diag"] = v
                row["q_norm"] = c0.norm_td(q)
        row["wall_s"] = time.perf_counter() - t0
        with steps_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        _log(phase, cell, arm, h)
    return


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def run_smoke(gpu):
    global _counters, _t_start, _order
    _counters = _new_counters()
    _t_start = time.perf_counter()
    _order = iter(range(10 ** 9))
    from fr import a0k_closed_loop_runner_v01 as a0k
    import a0h15_production_entry_v01 as pe

    rt = a0k.build_live_runtime(2030, f"cuda:{gpu}")
    ctx = make_ctx()
    RUNS.mkdir(parents=True, exist_ok=True)
    streams = {c: make_stream(rt, CELLS[c]["stream_seed"]) for c in ("R", "C")}
    banks = (materialize_bank(rt, C1CFG["probe_banks"]["dev_seed"]),
             materialize_bank(rt, C1CFG["probe_banks"]["audit_seed"]))
    global _vdiag_tensors
    _vdiag_tensors = v_diag_tensors_from_split(rt)
    steps_path = RUNS / "smoke_steps.jsonl"
    if steps_path.exists():
        steps_path.unlink()
    dummy_pred, dummy_rec = RUNS / "smoke_predictions.jsonl", RUNS / "smoke_anchor_records.jsonl"

    bank_report = check_banks(rt, banks, streams["R"], streams["C"])
    (RUNS / "probe_banks.json").write_text(json.dumps({
        "dev_seed": C1CFG["probe_banks"]["dev_seed"],
        "audit_seed": C1CFG["probe_banks"]["audit_seed"],
        "disjointness": bank_report,
        "micros": {"dev": banks[0], "audit": banks[1]}}, indent=1) + "\n")
    if bank_report["status"] != "PASS":
        raise SystemExit("BANK_DISJOINTNESS_FAIL")

    def state_sig():
        sd = raw_model(rt["model"]).state_dict()
        ost = rt["optimizer"].state_dict()["state"]
        parts = [v.detach().cpu() for v in sd.values()]
        for k in sorted(ost, key=str):
            for f2 in sorted(ost[k], key=str):
                t2 = ost[k][f2]
                if torch.is_tensor(t2):
                    parts.append(t2.detach().cpu())
        return hashlib.sha256(
            torch.cat([p.reshape(-1).float() for p in parts]).numpy().tobytes()
        ).hexdigest()

    rollout(rt, ctx, streams["R"], "R", "A", 0.0, 4, set(), banks, "smoke",
            dummy_pred, dummy_rec, steps_path, diag_steps=False)
    sig_off = state_sig()
    rollout(rt, ctx, streams["R"], "R", "A", 0.0, 4, set(), banks, "smoke",
            dummy_pred, dummy_rec, steps_path, diag_steps=True)
    sig_on = state_sig()

    rows = [json.loads(l) for l in open(steps_path)]
    rows_off, rows_on = rows[:4], rows[4:]
    shared = ["lr_native", "lr_applied"]
    field_match = all(a[k] == b[k] for k in shared for a, b in zip(rows_off, rows_on))
    identity = (sig_off == sig_on) and field_match

    # --- same-probe floor + HVP proxy gate at bank state ----------------------
    pe._restore_and_seed(rt, ctx, streams["R"])
    g_reps = {j: [probe_grad(rt, _stack(rt, banks[0][j])) for _ in range(3)]
              for j in range(4)}
    tangent = c0.clone_td(g_reps[0][0])
    h_reps = {j: [probe_hvp(rt, _stack(rt, banks[0][j]), tangent)[0]
                  for _ in range(3)] for j in range(4)}
    v_reps, q_reps, gaps = [], [], []
    for _ in range(3):
        v_r, q_r = v_diag_q(rt)
        v_reps.append(v_r)
        q_reps.append(q_r)
    for j in range(4):
        _, first = probe_hvp(rt, _stack(rt, banks[0][j]), tangent)
        gaps.append(_rel(first, g_reps[j][0]))

    def same_probe_max(reps_by_probe):
        worst = 0.0
        for _, reps in reps_by_probe.items():
            for i in range(len(reps)):
                for j2 in range(i + 1, len(reps)):
                    worst = max(worst, float(_rel(reps[i], reps[j2])))
        return worst

    floor = {"grad": same_probe_max(g_reps), "hvp": same_probe_max(h_reps),
             "v": max(abs(v_reps[i] - v_reps[j]) for i in range(3)
                      for j in range(i + 1, 3)),
             "q": max(_rel(q_reps[i], q_reps[j]) for i in range(3)
                      for j in range(i + 1, 3))}
    hvp_gap = max(gaps)
    checks = {
        "definition": ("same-probe pairing only (identical evaluations); "
                       "cross-probe diversity is NOT a floor"),
        "diag_state_identity": bool(identity),
        "diag_state_identity_detail": {
            "sig_off": sig_off, "sig_on": sig_on,
            "shared_fields_checked": shared + ["full_state_sha256"],
            "note": ("no external C0/B0-style full-state table exists for C1; "
                     "only these common fields + full-state hash checked; not "
                     "extrapolated beyond"),
            "updates": 8},
        "numerical_floor_same_probe": floor,
        "hvp_proxy_native_rel_gap_max": hvp_gap,
        "hvp_precision_gate": "PASS" if hvp_gap < 1e-2 else "NUMERICAL_UNRESOLVED",
        "bank_disjointness": bank_report,
        "per_step_wall_s_smoke_on": [r["wall_s"] for r in rows_on],
        "counters": dict(_counters),
        "shadow_seconds": _counters["shadow_seconds"],
    }
    (OUT / "smoke_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    _log("smoke", "R", "A", 8, note="smoke_complete")
    print("SMOKE_COMPLETE", json.dumps({k: checks[k] for k in
                                        ("diag_state_identity",
                                         "numerical_floor_same_probe",
                                         "hvp_precision_gate",
                                         "bank_disjointness")}))
    return 0 if (identity and hvp_gap < 1e-2) else 2


def run_main(gpu):
    global _counters, _t_start, _order
    _counters = _new_counters()
    _t_start = time.perf_counter()
    _order = iter(range(10 ** 9))
    if not (OUT / "smoke_checks.json").exists():
        raise SystemExit("SMOKE_CHECKS_MISSING")
    floor = json.loads((OUT / "smoke_checks.json").read_text())["numerical_floor_same_probe"]
    from fr import a0k_closed_loop_runner_v01 as a0k
    import csv

    # Round-total accounting across processes. The first main attempt crashed
    # at the R/A/t0 anchor (KeyError in the prediction-sha step, fixed since);
    # per protocol, failures count. Its measured cost is reconciled here.
    failed_path = RUNS / "failed_attempts.json"
    failed = json.loads(failed_path.read_text()) if failed_path.exists() else []
    failed_tot = {k: sum(f["counters"].get(k, 0) for f in failed) for k in
                  ("optimizer_updates", "grad_evals", "hvp_calls",
                   "micro_forward_calls")}

    rt = a0k.build_live_runtime(2030, f"cuda:{gpu}")
    ctx = make_ctx()
    banks_d = json.loads((RUNS / "probe_banks.json").read_text())
    banks = (banks_d["micros"]["dev"], banks_d["micros"]["audit"])
    global _vdiag_tensors
    _vdiag_tensors = v_diag_tensors_from_split(rt)
    predictions_path = RUNS / "anchor_predictions.jsonl"
    records_path = RUNS / "anchor_records.jsonl"
    steps_path = RUNS / "step_records.jsonl"
    for f in (predictions_path, records_path, steps_path):
        if f.exists():
            f.unlink()

    # Reduced symmetric coverage (disclosed): the protocol's six anchors need
    # 384 HVPs; with the counted failure (32) and smoke (16) the round would
    # need 432 > 416. In-cap symmetric plan: A/t0 + Bp/t16 per cell (256 HVPs,
    # t in {0,16} for both cells). A/t48 anchors NOT measured this round.
    anchors = C1CFG["executed_anchors"]
    path_lengths = {"A": 1, "Bp": 17}
    try:
        provenance = []
        for cell_name, cell in CELLS.items():
            stream = make_stream(rt, cell["stream_seed"])
            provenance.append({"cell": cell_name, "checkpoint": str(BANK_CKPT),
                               "stream_seed": cell["stream_seed"],
                               "bank_global_step": 46,
                               "A_path_updates": path_lengths["A"],
                               "Bp_path_updates": path_lengths["Bp"],
                               "anchors": ";".join(f"{a}@t{t}" for a, t in
                                                   anchors[cell_name])})
            for arm_name, T in path_lengths.items():
                u = 0.05 if arm_name == "Bp" else 0.0
                anchor_ts = {t for a, t in anchors[cell_name] if a == arm_name}
                rollout(rt, ctx, stream, cell_name, arm_name, u, T, anchor_ts,
                        banks, "main", predictions_path, records_path,
                        steps_path, diag_steps=False)
        with open(OUT / "state_provenance.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(provenance[0].keys()))
            w.writeheader()
            w.writerows(provenance)
    finally:
        _log("main", "ALL", "ALL", _counters["optimizer_updates"],
             note="main_complete" if _counters["optimizer_updates"] > 0 else
                  "main_aborted")
        wall = time.perf_counter() - _t_start
        smoke_c = json.loads((OUT / "smoke_checks.json").read_text())["counters"]
        round_tot = {k: _counters[k] + failed_tot[k] + smoke_c.get(k, 0)
                     for k in failed_tot}
        summary = {"counters_this_process": dict(_counters),
                   "counters_smoke_process": smoke_c,
                   "counters_failed_attempts": failed_tot,
                   "counters_round_total": round_tot,
                   "caps": {k: CAPS[k] for k in
                            ("optimizer_updates_max", "grad_evals_max",
                             "hvp_calls_max", "micro_forward_calls_max",
                             "gpu_wall_s_max")},
                   "within_caps": all(
                       round_tot[k] <= CAPS[k + "_max"] for k in round_tot)
                   and wall <= CAPS["gpu_wall_s_max"],
                   "coverage": "4/6 anchors (A/t0, Bp/t16 per cell); A/t48 "
                               "not measured: full 6-anchor plan would need "
                               "432 HVPs round-total (>416) after counting "
                               "the failed first attempt (32) and smoke (16)",
                   "floor_used_from_smoke": floor,
                   "shadow_seconds": _counters["shadow_seconds"],
                   "wall_run_s": wall}
        (OUT / "budget_log.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("MAIN_COMPLETE", json.dumps(summary))
    return 0 if summary["within_caps"] else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["smoke", "main"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(
        json.dumps(C1CFG, indent=1, ensure_ascii=False) + "\n")
    raise SystemExit({"smoke": run_smoke, "main": run_main}[args.phase](args.gpu))
