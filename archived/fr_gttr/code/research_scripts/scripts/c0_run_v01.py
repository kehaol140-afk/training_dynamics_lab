#!/usr/bin/env python3
"""C0-GTTR validation runner: 24-update smoke + 384-update main pilot.

Frozen semantics per outputs/fr_c0_gttr_validation_v01/{config.json,audit.md}.
Uses ONLY frozen primitives (a0g/a0h15/a0k/b0/full_state) plus the pure c0
algebra module. Hard budget caps are checked BEFORE each diagnostic; every
real optimizer step appends to budget_log.jsonl; failures count.

Anchor protocol order (per anchor, pre-update index t, h = t+1):
  1. pre-update dev/audit probe gradients at theta_t (RNG-wrapped)
  2. V_diag value + q in ONE eval-mode forward+backward
  3. real clip+step -> s_t (fp32 param diff), post moments
  4. snapshot post state; rewind to pre; dev HVPs (tangent s_t) at theta_t
  5. predictions M0/Mgeom/MnoJ + U-map written to disk BEFORE audit reads
  6. restore post; audit endpoint gradients at theta_{t+1}; y statistics
K_V for the previous step's displacement closes one step later (lagged row).
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
from fr.rng import capture_rng_state, restore_rng_state  # noqa: E402
from fr.minimind_v03_runtime import raw_model  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "fr_c0_gttr_validation_v01"
RUNS = OUT / "runs"
BUDGET_LOG = OUT / "budget_log.jsonl"

CFG = json.loads((OUT / "config.json").read_text())
BUDGET = CFG["budget"]
CAPS = BUDGET["diagnostic_caps"]
CELLS = {"R": CFG["cells"]["R"], "C": CFG["cells"]["C"]}
ARMS = {"A": 0.0, "Bp": 0.05, "Bm": -0.05}
ANCHORS = set(int(x) for x in CFG["anchors"]["t_pre_update"])
READOUT_H = [4, 16, 64]
BANK_CKPT = PROJECT_ROOT / CELLS["R"]["checkpoint"]
DEV_SEED = int(CFG["probe_banks"]["dev_seed"])
AUDIT_SEED = int(CFG["probe_banks"]["audit_seed"])
N_PROBES = int(CFG["probe_banks"]["n_dev"])
HVP_FP32 = True
B1C, B2C, EPS_ADAM, WD = 0.9, 0.999, 1e-8, 0.01

_counters = None
_t_start = None


def _new_counters():
    return {"optimizer_updates": 0, "grad_evals": 0, "hvp_calls": 0,
            "forward_calls": 0, "shadow_steps": 0}


def _log_step(phase, cell, arm, h, note=""):
    rec = {"ts": time.time(), "phase": phase, "cell": cell, "arm": arm, "h": h,
           "project_cumulative": BUDGET["consumed_before_c0_verified_from_logs"]["sum"]
           + _counters["optimizer_updates"],
           **_counters, "note": note}
    with BUDGET_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _check_budget(phase):
    c = _counters
    if c["optimizer_updates"] + BUDGET["consumed_before_c0_verified_from_logs"]["sum"] \
            > BUDGET["project_cap_optimizer_updates"]:
        raise SystemExit("BUDGET_BLOCKED: project 18000 cap")
    if phase == "smoke" and c["optimizer_updates"] > BUDGET["smoke_optimizer_updates"]:
        raise SystemExit("BUDGET_BLOCKED: smoke 24 cap")
    if phase == "main" and c["optimizer_updates"] > BUDGET["main_optimizer_updates"]:
        raise SystemExit("BUDGET_BLOCKED: main 384 cap")
    if c["grad_evals"] > CAPS["nongradient_evals_max"]:
        raise SystemExit("BUDGET_BLOCKED: grad-evals cap")
    if c["hvp_calls"] > CAPS["hvp_calls_max"]:
        raise SystemExit("BUDGET_BLOCKED: hvp cap")
    if c["forward_calls"] > CAPS["forward_calls_max"]:
        raise SystemExit("BUDGET_BLOCKED: forward cap")
    if time.perf_counter() - _t_start > CAPS["gpu_active_wall_s"]:
        raise SystemExit("BUDGET_BLOCKED: wall-clock cap")


# ---------------------------------------------------------------------------
# diagnostics (counters first, RNG restored, train mode restored)
# ---------------------------------------------------------------------------

def probe_grad(rt, tensors):
    import a0g_decision_utility_v01 as a0g
    _counters["grad_evals"] += 1
    _counters["forward_calls"] += len(tensors)
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
    _counters["forward_calls"] += len(tensors)
    hvp_ctx = nullcontext() if HVP_FP32 else rt["autocast_ctx"]
    rng = capture_rng_state()
    try:
        return a0g._hvp(rt["torch"], rt["model"], rt["named"], tensors,
                        tangent, hvp_ctx, int(rt["cfg"]["accumulation_steps"]))
    finally:
        restore_rng_state(rng)


def v_diag_and_q(rt, tensors):
    """V_diag value + validation gradient q in ONE eval-mode forward+backward."""
    _counters["grad_evals"] += 1
    _counters["forward_calls"] += len(tensors)
    torch = rt["torch"]
    rng = capture_rng_state()
    model = rt["model"]
    was_training = model.training
    params = [p for _, p in rt["named"]]
    try:
        model.eval()
        value = None
        q = None
        for x, y in tensors:  # exactly one chunk
            with rt["autocast_ctx"]:
                result = model(x, labels=y)
            loss = (result.loss + result.aux_loss).float()  # fp32 loss value
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


def v_eval_read(rt):
    import a0g_decision_utility_v01 as a0g
    _counters["forward_calls"] += len(rt["val_eval_tensors"])
    rng = capture_rng_state()
    try:
        return float(a0g._val_loss_only(rt["torch"], rt["model"],
                                        rt["val_eval_tensors"],
                                        rt["autocast_ctx"]))
    finally:
        restore_rng_state(rng)


def shadow_u(rt, g_raw, state, tau):
    _counters["shadow_steps"] += 1
    return c0.u_map(g_raw, **state, tau=tau)


# ---------------------------------------------------------------------------
# banks / streams / ctx
# ---------------------------------------------------------------------------

def materialize_bank(rt, seed):
    from fr.stage_c_runtime import batches_from_train_pool
    cfg = rt["cfg"]
    micros = batches_from_train_pool(rt["train_indices"], int(cfg["batch_size"]),
                                     int(seed), 2 * N_PROBES)
    return [micros[2 * j: 2 * j + 2] for j in range(N_PROBES)]


def write_bank_manifest(rt, banks):
    doc = {}
    for label, bank in (("dev", banks[0]), ("audit", banks[1])):
        payload = json.dumps(bank, sort_keys=True).encode()
        doc[label] = {"sha256": hashlib.sha256(payload).hexdigest(),
                      "micros": [[list(map(int, m)) for m in probe] for probe in bank]}
    (RUNS / "probe_banks.json").write_text(json.dumps(doc, indent=1) + "\n")


def stack_probe(rt, micros):
    import a0g_decision_utility_v01 as a0g
    return a0g._stack_all(rt["dataset"], micros, rt["device"])


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
    ids = sorted(map(int, split["v_score"]))[:8]  # frozen prefix-ID rule
    return a0g._stack_all(rt["dataset"], [ids], rt["device"])


def snap_full(rt, name):
    from fr.full_state import save_full_training_state
    p = RUNS / name
    save_full_training_state(p, rt["model"], rt["optimizer"], rt["scaler"],
                             metadata={"protocol": "FR_C0_ANCHOR_SNAPSHOT"})
    return p


def load_full(rt, path):
    from fr.full_state import load_full_training_state
    load_full_training_state(path, rt["model"], rt["optimizer"], rt["scaler"],
                             restore_rng=False)
    rt["model"].train()


def post_moments(rt, t_after):
    opt_state = rt["optimizer"].state  # keyed by param object
    m_hat, v_hat = {}, {}
    for n, p in rt["named"]:
        st = opt_state[p]
        m_hat[n] = (st["exp_avg"] / (1 - B1C ** t_after)).clone()
        v_hat[n] = (st["exp_avg_sq"] / (1 - B2C ** t_after)).clone()
    return m_hat, v_hat


def _rel(a, b):
    nb = c0.norm_td(b)
    return c0.norm_td(c0.sub_td(a, b)) / nb if nb > 0 else float("nan")


def _e_stats(pred_list, y_list, idx=None):
    p = pred_list if idx is None else [pred_list[i] for i in idx]
    y = y_list if idx is None else [y_list[i] for i in idx]
    return {"E": c0.E_model(c0.ybar(p), c0.ybar(y)),
            "y_norm": c0.norm_td(c0.ybar(y)), "se": c0.se_vec(y)}


# ---------------------------------------------------------------------------
# anchor protocol
# ---------------------------------------------------------------------------

def anchor_pre(rt, banks, tau, order):
    """Steps 1-2 at theta_t. Returns dict of live tensors + first record."""
    dev_micros, audit_micros = banks
    dev_g, dev_c = [], []
    for j in range(N_PROBES):
        g = probe_grad(rt, stack_probe(rt, dev_micros[j]))
        dev_g.append(g)
        dev_c.append(c0.apply_clip(g, tau))
    audit_g0, audit_c0 = [], []
    for j in range(N_PROBES):
        g = probe_grad(rt, stack_probe(rt, audit_micros[j]))
        audit_g0.append(g)
        audit_c0.append(c0.apply_clip(g, tau))
    v_val, q = v_diag_and_q(rt, _CURRENT["v_diag_tensors"])
    return {"dev_g": dev_g, "dev_c": dev_c, "audit_g0": audit_g0,
            "audit_c0": audit_c0, "v": v_val, "q": q,
            "order_pre": next(order)}


def anchor_post(rt, pre, banks, s_t, theta_post, m_hat_post, v_hat_post,
                t_after, eta_t, lr_next, tau, floor, order, predictions_path,
                cell, arm, t_pre):
    """Steps 4-6. Predictions are written to predictions_path BEFORE audit
    endpoint gradients are read; order_index logged in both records."""
    dev_micros, audit_micros = banks
    dev_g, dev_c = pre["dev_g"], pre["dev_c"]
    audit_g0, audit_c0 = pre["audit_g0"], pre["audit_c0"]
    q = pre["q"]
    snap_post = snap_full(rt, "_c0_anchor_post.pt")
    load_full(rt, _CURRENT["snap_pre"])

    # --- 4. dev HVPs at theta_t with tangent s_t ---------------------------
    hvps, gaps = [], []
    for j in range(N_PROBES):
        hvp, first = probe_hvp(rt, stack_probe(rt, dev_micros[j]), s_t)
        hvps.append(hvp)
        gaps.append(_rel(first, dev_g[j]))

    # --- 5. predictions; WRITE BEFORE AUDIT --------------------------------
    u_state = dict(m_hat=m_hat_post, v_hat=v_hat_post, beta1=B1C, beta2=B2C,
                   t_prev=t_after, eps=EPS_ADAM, lr=lr_next, wd=WD,
                   theta=theta_post)
    p_mu, p_noJ, p_U = [], [], []
    clip_flags, kink_margins = [], []
    for j in range(N_PROBES):
        pred_raw = c0.predict_geom(dev_g[j], hvps[j], s_t)
        npr = c0.global_norm(pred_raw)
        clip_flags.append(c0.clip_active(npr, tau))
        kink_margins.append(npr + c0.CLIP_DENOMINATOR_EPS - tau)
        p_mu.append(c0.sub_td(c0.apply_clip(pred_raw, tau), dev_c[j]))
        p_noJ.append(hvps[j])
        p_U.append(c0.sub_td(shadow_u(rt, pred_raw, u_state, tau),
                             shadow_u(rt, dev_g[j], u_state, tau)))
    # stop-v ablation (zero gradient cost): P frozen from pair-baseline
    c_bar = c0.ybar(audit_c0)
    a2v, b2v = c0.ema_coeffs(B2C, t_after)
    P_fixed = {n: 1.0 / (torch.sqrt(a2v * v_hat_post[n] + b2v * c_bar[n] * c_bar[n])
                         + EPS_ADAM) for n in c_bar}
    sv = dict(m_hat=m_hat_post, v_hat=v_hat_post, beta1=B1C, beta2=B2C,
              t_prev=t_after, eps=EPS_ADAM, lr=lr_next, wd=WD,
              theta=theta_post, tau=tau, P_fixed=P_fixed)
    p_stop = [c0.sub_td(c0.u_map_stop_v(c0.predict_geom(dev_g[j], hvps[j], s_t), **sv),
                        c0.u_map_stop_v(dev_g[j], **sv)) for j in range(N_PROBES)]
    pred_rec = {
        "cell": cell, "arm": arm, "anchor": t_pre, "kind": "predictions",
        "p_mu_norm": c0.norm_td(c0.ybar(p_mu)),
        "p_noJ_norm": c0.norm_td(c0.ybar(p_noJ)),
        "p_U_norm": c0.norm_td(c0.ybar(p_U)),
        "p_stopv_norm": c0.norm_td(c0.ybar(p_stop)),
        "mgeom_clip_active_frac": sum(clip_flags) / len(clip_flags),
        "mgeom_kink_margin_min": min(kink_margins),
        "hvp_first_grad_rel_gap_max": max(gaps),
        "order_index": next(order)}
    with predictions_path.open("a") as f:
        f.write(json.dumps(pred_rec) + "\n")

    # --- 6. restore post; audit endpoint gradients at theta_{t+1} ----------
    load_full(rt, snap_post)
    y_mu, y_U, y_stop = [], [], []
    for j in range(N_PROBES):
        g_end = probe_grad(rt, stack_probe(rt, audit_micros[j]))
        y_mu.append(c0.sub_td(c0.apply_clip(g_end, tau), audit_c0[j]))
        y_U.append(c0.sub_td(shadow_u(rt, g_end, u_state, tau),
                             shadow_u(rt, audit_g0[j], u_state, tau)))
        y_stop.append(c0.sub_td(c0.u_map_stop_v(g_end, **sv),
                                c0.u_map_stop_v(audit_g0[j], **sv)))
    halves = {"first4": [0, 1, 2, 3], "last4": [4, 5, 6, 7]}
    stats_mu = _e_stats(p_mu, y_mu)
    rec = {"cell": cell, "arm": arm, "anchor": t_pre,
           "y_mu": stats_mu, "E_M0_mu": 1.0,
           "E_Mgeom_mu": stats_mu["E"],
           "E_MnoJ_mu": _e_stats(p_noJ, y_mu)["E"],
           "y_U": _e_stats(p_U, y_U),
           "E_M0_U": 1.0,
           "E_Mgeom_U": _e_stats(p_U, y_U)["E"],
           "y_stopv": _e_stats(p_stop, y_stop),
           "halfbank": {k: {"E_Mgeom_mu": _e_stats(p_mu, y_mu, idx)["E"],
                            "E_Mgeom_U": _e_stats(p_U, y_U, idx)["E"]}
                        for k, idx in halves.items()},
           "resolved_mu": c0.resolved(stats_mu["y_norm"], stats_mu["se"], floor["grad"]),
           "resolved_U": c0.resolved(_e_stats(p_U, y_U)["y_norm"],
                                     _e_stats(p_U, y_U)["se"], floor["grad"]),
           "order_index_audit": next(order)}
    # tracking diagnostics (transport T_t decomposition)
    e_m = {n: m_hat_post[n] - c0.ybar(audit_c0)[n] for n in m_hat_post}
    hat_nu = c0.ybar([{n: v * v for n, v in c.items()} for c in audit_c0])
    e_v = {n: v_hat_post[n] - hat_nu[n] for n in v_hat_post}
    P = c0.p_diag(v_hat_post, EPS_ADAM)
    terms = c0.readout_terms(q=q, s=s_t, P=P, mu=c0.ybar(audit_c0), e_m=e_m,
                             wd=WD, theta=_CURRENT["theta_pre"], eta=eta_t)
    rec["tracking"] = {"e_m_norm": c0.norm_td(e_m), "e_v_norm": c0.norm_td(e_v),
                       "W_field": terms["W_field"], "W_track": terms["W_track"],
                       "W_wd": terms["W_wd"], "W_total_qs": terms["W_total"]}
    # transport T_t: analytic displacement from consumed moments vs realized s_t
    r_cons = {n: m_hat_post[n] / (torch.sqrt(v_hat_post[n]) + EPS_ADAM)
              for n in m_hat_post}
    s_analytic = {n: -eta_t * r_cons[n] - eta_t * WD * _CURRENT["theta_pre"][n]
                  for n in r_cons}
    rec["tracking"]["s_analytic_norm"] = c0.norm_td(s_analytic)
    rec["tracking"]["transport_err_rel"] = _rel(s_analytic, s_t)
    return rec


_CURRENT = {}


# ---------------------------------------------------------------------------
# shared rollout machinery
# ---------------------------------------------------------------------------

def _arm_rollout(rt, ctx, stream, cell_name, arm_name, u, T, tau, v_diag_tensors,
                 banks, floor, order, predictions_path, anchors_path, steps_path,
                 phase, diag=True, anchor_at=None):
    """T real steps for one (cell, arm). Anchor machinery at pre-update indices
    in anchor_at (smoke) or ANCHORS (main). Returns list of step rows."""
    import a0h15_production_entry_v01 as pe
    torch = rt["torch"]
    mult = math.exp(u)
    pe._restore_and_seed(rt, ctx, stream)
    rows = []
    prev = {"v": None, "q": None, "s": None, "W_total": None}
    for h in range(1, T + 1):
        _check_budget(phase)
        t_h0 = time.perf_counter()
        native_gradient, _, snap = pe._baseline_no_step(
            rt, ctx, stream["batches"], h, want_snapshot=True)
        _counters["optimizer_updates"] += 1
        if mult != 1.0:
            for g in rt["optimizer"].param_groups:
                g["lr"] = float(snap["lr"]) * mult
        eta_t = float(rt["optimizer"].param_groups[0]["lr"])
        t_pre = h - 1
        is_anchor = diag and ((anchor_at is None and t_pre in ANCHORS)
                              or (anchor_at is not None and t_pre in anchor_at))
        if is_anchor:
            _CURRENT["snap_pre"] = snap_full(rt, "_c0_anchor_pre.pt")
            _CURRENT["theta_pre"] = {n: p.detach().float() for n, p in rt["named"]}
            pre = anchor_pre(rt, banks, tau, order)
            v_val, q = pre["v"], pre["q"]
        elif diag:
            v_val, q = v_diag_and_q(rt, v_diag_tensors)
        if not diag:
            v_val, q = None, None
        preclip, clipped = pe._clip_only(rt, native_gradient, snap)
        pe._step_once(rt)
        theta_post = {n: p.detach().float() for n, p in rt["named"]}
        theta_start = {n: snap["theta_start"][n].float() for n, _ in rt["named"]}
        s_t = {n: theta_post[n] - theta_start[n] for n in theta_post}
        t_after = int(snap["step_after"])
        m_hat_post, v_hat_post = post_moments(rt, t_after)
        lr_next = float(rt["get_lr"]((46 + h) * 2, 256,
                                     float(rt["cfg"]["learning_rate"]))) * mult
        row = {"cell": cell_name, "arm": arm_name, "h": h, "u": u,
               "lr_native": float(snap["lr"]), "lr_applied": eta_t,
               "preclip_norm": float(preclip), "clip_active": bool(preclip > tau),
               "s_norm": c0.norm_td(s_t), "v_diag": v_val,
               "q_norm": c0.norm_td(q) if q is not None else None}
        if prev["v"] is not None:
            W_total = -c0.dot_td(prev["q"], prev["s"])
            row["W_total_prev"] = W_total
            row["K_V"] = c0.k_v(v_val, prev["v"], W_total)
            row["K_V_note"] = "V_h - V_{h-1} + W_total(s_{h-1}); lags one step"
        if is_anchor:
            _CURRENT["v_diag_tensors"] = v_diag_tensors
            rec = anchor_post(rt, pre, banks, s_t, theta_post, m_hat_post,
                              v_hat_post, t_after, eta_t, lr_next, tau, floor,
                              order, predictions_path, cell_name, arm_name, t_pre)
            with anchors_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        if h in READOUT_H and diag:
            row["v_eval"] = v_eval_read(rt)
        row["wall_s"] = time.perf_counter() - t_h0
        rows.append(row)
        with steps_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        prev = {"v": v_val, "q": q, "s": s_t}
        _log_step(phase, cell_name, arm_name, h)
    return rows


# ---------------------------------------------------------------------------
# phase: smoke
# ---------------------------------------------------------------------------

def run_smoke(gpu):
    global _counters, _t_start
    _counters = _new_counters()
    _t_start = time.perf_counter()
    from fr import a0k_closed_loop_runner_v01 as a0k
    import a0h15_production_entry_v01 as pe

    rt = a0k.build_live_runtime(2030, f"cuda:{gpu}")
    tau = float(rt["cfg"]["grad_clip"])
    ctx = make_ctx()
    RUNS.mkdir(parents=True, exist_ok=True)
    stream = make_stream(rt, CELLS["R"]["stream_seed"])
    banks = (materialize_bank(rt, DEV_SEED), materialize_bank(rt, AUDIT_SEED))
    write_bank_manifest(rt, banks)
    v_diag_tensors = v_diag_tensors_from_split(rt)
    _CURRENT["v_diag_tensors"] = v_diag_tensors
    order = iter(range(10 ** 9))
    predictions_path = RUNS / "smoke_predictions.jsonl"
    anchors_path = RUNS / "smoke_anchors.jsonl"
    steps_path = RUNS / "smoke_steps.jsonl"
    for f in (predictions_path, anchors_path, steps_path):
        if f.exists():
            f.unlink()

    floor = {"grad": 0.0, "hvp": 0.0, "v": 0.0, "q": 0.0}

    def state_sig():
        sd = raw_model(rt["model"]).state_dict()
        ost = rt["optimizer"].state_dict()["state"]
        parts = []
        for v in sd.values():
            parts.append(v.detach().cpu())
        for k in sorted(ost, key=str):
            for f2 in sorted(ost[k], key=str):
                t2 = ost[k][f2]
                if torch.is_tensor(t2):
                    parts.append(t2.detach().cpu())
        return hashlib.sha256(torch.cat([p.reshape(-1).float() for p in parts]).numpy().tobytes()).hexdigest()

    # --- arm A diag-off: pure 8 steps (replica baseline) -------------------
    rows_off = _arm_rollout(rt, ctx, stream, "R", "A", 0.0, 8, tau,
                            v_diag_tensors, banks, floor, order,
                            predictions_path, anchors_path, steps_path,
                            "smoke", diag=False)
    sig_off = state_sig()
    # --- arm A diag-on: 8 steps with full anchor machinery at t=0 ----------
    rows_on = _arm_rollout(rt, ctx, stream, "R", "A", 0.0, 8, tau,
                           v_diag_tensors, banks, floor, order,
                           predictions_path, anchors_path, steps_path,
                           "smoke", diag=True, anchor_at={0})
    sig_on = state_sig()
    identity = (sig_off == sig_on)
    # --- B+/B-: 4 steps each, diag on --------------------------------------
    _arm_rollout(rt, ctx, stream, "R", "Bp", 0.05, 4, tau, v_diag_tensors,
                 banks, floor, order, predictions_path, anchors_path,
                 steps_path, "smoke", diag=True, anchor_at={0})
    _arm_rollout(rt, ctx, stream, "R", "Bm", -0.05, 4, tau, v_diag_tensors,
                 banks, floor, order, predictions_path, anchors_path,
                 steps_path, "smoke", diag=True, anchor_at={0})

    # --- numerical floor: 3x repeats of dev-bank grads/HVPs/V at bank state
    pe._restore_and_seed(rt, ctx, stream)
    g_reps, h_reps, v_reps, q_reps = [], [], [], []
    tangent = None
    for r in range(3):
        for j in range(N_PROBES):
            g_reps.append(probe_grad(rt, stack_probe(rt, banks[0][j])))
        if tangent is None:
            tangent = c0.clone_td(g_reps[0])
        for j in range(N_PROBES):
            hvp, _ = probe_hvp(rt, stack_probe(rt, banks[0][j]), tangent)
            h_reps.append(hvp)
        v_r, q_r = v_diag_and_q(rt, v_diag_tensors)
        v_reps.append(v_r)
        q_reps.append(q_r)

    def max_pairwise_diff(reps):
        worst = 0.0
        for i in range(len(reps)):
            for j2 in range(i + 1, len(reps)):
                if torch.is_tensor(reps[i]):
                    d = _rel(reps[i], reps[j2]) * c0.norm_td(reps[j2])
                elif isinstance(reps[i], dict):
                    d = _rel(reps[i], reps[j2])
                else:
                    d = abs(reps[i] - reps[j2])
                worst = max(worst, float(d))
        return worst

    floor = {"grad": max_pairwise_diff(g_reps),
             "hvp": max_pairwise_diff(h_reps),
             "v": max_pairwise_diff(v_reps),
             "q": max_pairwise_diff(q_reps)}

    # --- HVP proxy-vs-native gap gate at bank state ------------------------
    gaps = []
    for j in range(N_PROBES):
        hvp, first = probe_hvp(rt, stack_probe(rt, banks[0][j]), tangent)
        g_native = probe_grad(rt, stack_probe(rt, banks[0][j]))
        gaps.append(_rel(first, g_native))
    hvp_gap = max(gaps)
    hvp_gate = bool(hvp_gap < 1e-2)  # A0C-recorded math-SDPA proxy magnitude

    checks = {
        "smoke_diag_identity": bool(identity),
        "smoke_diag_identity_detail": {"sig_off": sig_off, "sig_on": sig_on,
                                       "updates": 24},
        "numerical_floor": floor,
        "hvp_proxy_native_rel_gap_max": hvp_gap,
        "hvp_precision_gate": "PASS" if hvp_gate else "NUMERICAL_UNRESOLVED",
        "counters": dict(_counters),
        "budget_project_cumulative": BUDGET["consumed_before_c0_verified_from_logs"]["sum"]
        + _counters["optimizer_updates"],
    }
    (OUT / "smoke_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    _log_step("smoke", "R", "ALL", 24, note="smoke_complete")
    print("SMOKE_COMPLETE", json.dumps(checks))
    return 0 if (identity and hvp_gate) else 2


# ---------------------------------------------------------------------------
# phase: main
# ---------------------------------------------------------------------------

def run_main(gpu):
    global _counters, _t_start
    _counters = _new_counters()
    _t_start = time.perf_counter()
    from fr import a0k_closed_loop_runner_v01 as a0k

    if not (OUT / "smoke_checks.json").exists():
        raise SystemExit("SMOKE_CHECKS_MISSING: run --phase smoke first")
    floor = json.loads((OUT / "smoke_checks.json").read_text())["numerical_floor"]

    rt = a0k.build_live_runtime(2030, f"cuda:{gpu}")
    tau = float(rt["cfg"]["grad_clip"])
    ctx = make_ctx()
    RUNS.mkdir(parents=True, exist_ok=True)
    banks = (materialize_bank(rt, DEV_SEED), materialize_bank(rt, AUDIT_SEED))
    write_bank_manifest(rt, banks)
    v_diag_tensors = v_diag_tensors_from_split(rt)
    _CURRENT["v_diag_tensors"] = v_diag_tensors
    order = iter(range(10 ** 9))
    predictions_path = RUNS / "anchor_predictions.jsonl"
    anchors_path = RUNS / "anchor_records.jsonl"
    steps_path = RUNS / "step_records.jsonl"
    for f in (predictions_path, anchors_path, steps_path):
        if f.exists():
            f.unlink()

    for cell_name, cell in CELLS.items():
        stream = make_stream(rt, cell["stream_seed"])
        for arm_name, u in ARMS.items():
            _arm_rollout(rt, ctx, stream, cell_name, arm_name, u, 64, tau,
                         v_diag_tensors, banks, floor, order,
                         predictions_path, anchors_path, steps_path, "main")
    _log_step("main", "ALL", "ALL", 64, note="main_complete")
    print("MAIN_COMPLETE", json.dumps(_counters))
    return 0


def run_floor(gpu):
    """Zero-update floor/gate measurement at the Cell R bank state.

    After two failed smoke attempts (33 updates, disclosed), the smoke gates
    are completed without new optimizer updates: numerical floors from
    repeated dev-bank diagnostics + HVP proxy-vs-native gap. Diag-identity is
    established by the analyzer from attempt-2 step rows (diag off vs on) and
    the external B0 stress_diag bitwise gate on main Cell R arm A.
    """
    global _counters, _t_start
    _counters = _new_counters()
    _t_start = time.perf_counter()
    from fr import a0k_closed_loop_runner_v01 as a0k
    import a0h15_production_entry_v01 as pe

    rt = a0k.build_live_runtime(2030, f"cuda:{gpu}")
    ctx = make_ctx()
    RUNS.mkdir(parents=True, exist_ok=True)
    stream = make_stream(rt, CELLS["R"]["stream_seed"])
    banks = (materialize_bank(rt, DEV_SEED), materialize_bank(rt, AUDIT_SEED))
    write_bank_manifest(rt, banks)
    v_diag_tensors = v_diag_tensors_from_split(rt)
    _CURRENT["v_diag_tensors"] = v_diag_tensors
    pe._restore_and_seed(rt, ctx, stream)

    g_reps, h_reps, v_reps, q_reps = [], [], [], []
    tangent = None
    for r in range(3):
        for j in range(N_PROBES):
            g_reps.append(probe_grad(rt, stack_probe(rt, banks[0][j])))
        if tangent is None:
            tangent = c0.clone_td(g_reps[0])
        for j in range(N_PROBES):
            hvp, _ = probe_hvp(rt, stack_probe(rt, banks[0][j]), tangent)
            h_reps.append(hvp)
        v_r, q_r = v_diag_and_q(rt, v_diag_tensors)
        v_reps.append(v_r)
        q_reps.append(q_r)

    def _diff(a, b):
        if isinstance(a, dict):
            return _rel(a, b)
        return abs(a - b)

    def max_pairwise_diff(reps):
        return max(float(_diff(reps[i], reps[j2]))
                   for i in range(len(reps)) for j2 in range(i + 1, len(reps)))

    floor = {"grad": max_pairwise_diff(g_reps), "hvp": max_pairwise_diff(h_reps),
             "v": max_pairwise_diff(v_reps), "q": max_pairwise_diff(q_reps)}
    gaps = []
    for j in range(N_PROBES):
        hvp, first = probe_hvp(rt, stack_probe(rt, banks[0][j]), tangent)
        g_native = probe_grad(rt, stack_probe(rt, banks[0][j]))
        gaps.append(_rel(first, g_native))
    hvp_gap = max(gaps)
    checks = {
        "smoke_diag_identity": "DEFERRED_TO_ANALYZER",
        "smoke_attempts_disclosed": {"attempt1_updates": 9, "attempt2_updates": 24,
                                     "total_wasted": 33},
        "numerical_floor": floor,
        "hvp_proxy_native_rel_gap_max": hvp_gap,
        "hvp_precision_gate": "PASS" if hvp_gap < 1e-2 else "NUMERICAL_UNRESOLVED",
        "counters": dict(_counters),
        "budget_project_cumulative": BUDGET["consumed_before_c0_verified_from_logs"]["sum"] + 33,
    }
    (OUT / "smoke_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    _log_step("floor", "R", "ALL", 0, note="floor_only_zero_updates")
    print("FLOOR_COMPLETE", json.dumps(checks))
    return 0


def run_floor2(gpu):
    """Zero-update CORRECTED numerical floor per the frozen config definition:
    'max ||diff|| of repeated IDENTICAL evaluations'.  The floor/smoke phases
    pooled pairwise diffs across DIFFERENT probes into the floor (an
    implementation deviation that inflated the floor to cross-probe gradient
    diversity).  Here repeats of the SAME probe are paired only.  0 updates.
    """
    global _counters, _t_start
    _counters = _new_counters()
    _t_start = time.perf_counter()
    from fr import a0k_closed_loop_runner_v01 as a0k
    import a0h15_production_entry_v01 as pe

    rt = a0k.build_live_runtime(2030, f"cuda:{gpu}")
    ctx = make_ctx()
    RUNS.mkdir(parents=True, exist_ok=True)
    stream = make_stream(rt, CELLS["R"]["stream_seed"])
    banks = (materialize_bank(rt, DEV_SEED), materialize_bank(rt, AUDIT_SEED))
    v_diag_tensors = v_diag_tensors_from_split(rt)
    _CURRENT["v_diag_tensors"] = v_diag_tensors
    pe._restore_and_seed(rt, ctx, stream)

    R = 4  # repeats per identical evaluation
    g_reps = {j: [] for j in range(N_PROBES)}
    h_reps = {j: [] for j in range(N_PROBES)}
    v_reps, q_reps = [], []
    tangent = None
    for r in range(R):
        for j in range(N_PROBES):
            g = probe_grad(rt, stack_probe(rt, banks[0][j]))
            g_reps[j].append(g)
            if tangent is None:
                tangent = c0.clone_td(g)
        for j in range(N_PROBES):
            hvp, _ = probe_hvp(rt, stack_probe(rt, banks[0][j]), tangent)
            h_reps[j].append(hvp)
        v_r, q_r = v_diag_and_q(rt, v_diag_tensors)
        v_reps.append(v_r)
        q_reps.append(q_r)

    def _max_same_probe_diff(reps_by_probe):
        worst = 0.0
        for j, reps in reps_by_probe.items():
            for i in range(len(reps)):
                for j2 in range(i + 1, len(reps)):
                    worst = max(worst, float(_rel(reps[i], reps[j2])))
        return worst

    def _diff(a, b):
        return _rel(a, b) if isinstance(a, dict) else abs(a - b)

    def _max_pairwise(reps):
        return max(float(_diff(reps[i], reps[j2]))
                   for i in range(len(reps)) for j2 in range(i + 1, len(reps)))

    floor2 = {"grad": _max_same_probe_diff(g_reps),
              "hvp": _max_same_probe_diff(h_reps),
              "v": _max_pairwise(v_reps), "q": _max_pairwise(q_reps)}
    checks = {
        "definition": ("max ||diff|| of repeated IDENTICAL evaluations "
                       "(same probe paired with its own repeats only), fp64"),
        "repeats_per_probe": R,
        "numerical_floor_corrected": floor2,
        "counters": dict(_counters),
        "note": ("supersedes smoke_checks.json numerical_floor, which pooled "
                 "cross-probe pairs (different evaluations) contrary to the "
                 "frozen definition; supersession disclosed in REPORT.md"),
    }
    (OUT / "smoke_checks_floor2.json").write_text(json.dumps(checks, indent=1) + "\n")
    _log_step("floor2", "R", "ALL", 0, note="floor2_only_zero_updates")
    print("FLOOR2_COMPLETE", json.dumps(checks))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["smoke", "main", "floor", "floor2"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    raise SystemExit({"smoke": run_smoke, "main": run_main, "floor": run_floor,
                      "floor2": run_floor2}[args.phase](args.gpu))
