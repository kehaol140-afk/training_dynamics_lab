#!/usr/bin/env python3
"""FR-C1-Late: complete the two missing A/t48 anchors (R/C, native arm A,
pre-update local t=48) with extended sufficient statistics (protocol
FR_C1L_GTTR_LateAnchor_DataFirst_Claude_Code.md).

Reuses the frozen C1 runner machinery by importing c1_run_v01 and patching
only output paths / budget caps / config. The anchor function is replaced by
an extended version that additionally saves, per protocol §7.1/§6.1/§6.2:
  - per-group n, sum_j||x_j||^2, ||mean||^2, centered_SS (full/K prefixes/halves)
  - mean(p_D)/mean(p_A)/mean(y_A) norm^2 + pairwise inner products per K/half
  - shadow same-probe paired residual (||Q_j - Z_j||) mean + SE, both banks
  - clip-active counts at g0 / g_pred / g1 + active-set crossing counts
  - stop-v per-probe full/stop norms, norm ratio, cosine (mean +- SE)
  - paired full-minus-stop projections on the fixed q+ / s_hat directions

Leakage order is unchanged and stricter-than-required: dev predictions are
sha-saved BEFORE any audit endpoint is read (order_index logged).

Round budget (independent ledger; old 18,000 and old C1 ledgers untouched):
  optimizer updates <= 128 | effective-batch grads <= 600 | HVPs <= 192
  micro-forward calls <= 1800 | GPU wall <= 2700 s
State recovery: full native replay from the frozen step-46 checkpoint
(49 real updates per cell; no usable on-disk snapshot at A/t48 exists —
C1 snapshots cover t0/t16 only; disclosed in provenance).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

import torch  # noqa: E402

import c1_run_v01 as R  # noqa: E402  (frozen C1 runner; paths/cfg patched below)
import fr.c0_gttr_validation_v01 as c0  # noqa: E402
import fr.c1_gttr_probe_validation_v01 as c1  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "fr_c1_late_completion_v01"
RUNS = OUT / "runs"
BUDGET_LOG = OUT / "resource_log.jsonl"

CLATE = {
    "experiment_id": "FR_C1_LATE_COMPLETION_V01",
    "protocol": "FR_C1L_GTTR_LateAnchor_DataFirst_Claude_Code.md",
    "date": "2026-09-21",
    "scope": ("Complete the two A/t48 anchors missing from C1 (R and C cells, "
              "native arm A, pre-update local t=48) + offline recheck of C0X/C1 "
              "key data. Coverage completion of the pre-registered 6 anchors; "
              "NOT an independent replication and NOT a new population sample."),
    "anchors_planned_original_C1": {"R": [["A", 0], ["Bp", 16], ["A", 48]],
                                    "C": [["A", 0], ["Bp", 16], ["A", 48]]},
    "anchors_executed_this_round": {"R": [["A", 48]], "C": [["A", 48]]},
    "state_recovery": ("full native replay from frozen checkpoint "
                       "outputs/e3_confirmatory/v07/seed_2030/states/state_step_046.pt; "
                       "49 real updates per cell (h=1..49; anchor at t_pre=48). "
                       "No on-disk snapshot at A/t48 exists (C1 snapshots: t0/t16 "
                       "only); snapshot-continuation shortcuts NOT used."),
    "probe_banks": {"source": "identical C1 banks (runs/probe_banks.json copied "
                              "and re-materialized from the same frozen seeds; "
                              "ID order and batch recipe unchanged)",
                    "n_dev": 32, "n_audit": 32, "micros_per_probe": 2,
                    "dev_seed": 2026092103, "audit_seed": 2026092104},
    "k_prefixes": [8, 16, 32],
    "half_split": 16,
    "half_note": ("first16 == K16 prefix (same fixed probe IDs, alias); the "
                  "independent half is last16"),
    "budget": {
        "optimizer_updates_max": 128, "grad_evals_max": 600, "hvp_calls_max": 192,
        "micro_forward_calls_max": 1800, "gpu_wall_s_max": 2700,
        "accounting": ("round-total across processes; failures/rechecks/salvage "
                       "count; two simultaneous counter sets (effective-batch "
                       "evals and micro-forward calls); shadow arithmetic counted "
                       "separately (steps+seconds); old 17,994/18,000 ledger and "
                       "old C1 round ledger untouched; never divide by 18,000"),
    },
    "precision": {"model_dtype": "bfloat16", "inner_products": "fp64",
                  "hvp_fp32": True,
                  "note": "FP64 used only for stable scalar accumulation of "
                          "already-produced outputs (GroupAcc host fp64)"},
    "clip": {"tau_source": "rt cfg grad_clip (same as C0/C1)",
             "CLIP_DENOMINATOR_EPS": c0.CLIP_DENOMINATOR_EPS},
    "optim": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "wd": 0.01},
    "stop_v": ("P frozen from the AUDIT bank's mean clipped g0 (C0/C1 pair-"
               "baseline convention, a2v/b2v at t_after); ablation only, never "
               "an optimizer; shadow step never written back"),
    "leakage_order": ("per anchor: dev+audit g0 -> real clip+step -> q+ fixed "
                      "-> dev HVP predictions sha-saved (order_index) -> audit "
                      "HVPs / g1 (CONDITIONED, no backfill)"),
    "terminology_bans": ["temperature", "free energy", "entropy production",
                         "equilibrium", "detailed balance", "field theory",
                         "phase transition"],
}

# patch the frozen runner's module-level configuration
R.OUT = OUT
R.RUNS = RUNS
R.BUDGET_LOG = BUDGET_LOG
R.C1CFG = CLATE
R.CAPS = CLATE["budget"]

B1C, B2C, EPS_ADAM, WD = 0.9, 0.999, 1e-8, 0.01
N_PROBES = 32
MAIN_GROUPS = R.MAIN_GROUPS
EXTRA_GROUPS = R.EXTRA_GROUPS
ALL_GROUPS = R.ALL_GROUPS


# ---------------------------------------------------------------------------
# extended anchor (C1 run_anchor + protocol §7.1/§6.1/§6.2 statistics)
# ---------------------------------------------------------------------------

def _centered_ss(snap):
    """sum_j||x_j - xbar||^2 = sumsq - ||sum||^2/n  (host fp64)."""
    n = snap["n"]
    return snap["sumsq"] - c0.dot_td(snap["sum"], snap["sum"]) / n


def _group_stats_all(acc):
    """n / sum_j||x||^2 / ||mean||^2 / centered_SS for full + K prefixes + halves."""
    def block(snap, n):
        return {"n": n, "sum_sq_norm": snap["sumsq"],
                "mean_sq_norm": c1.mean_sq_norm(snap),
                "centered_SS": _centered_ss(snap)}
    out = {"full32": block(acc.snapshot(), acc.snapshot()["n"])}
    for k, sn in acc.prefixes().items():
        out[f"K{k}"] = block(sn, sn["n"])
    for hn, hs in acc.halves().items():
        if hs["sum"] is not None:
            out[hn] = block(hs, hs["n"])
    return out


def _triple_stats(acc_p, acc_a, acc_y):
    """Per K prefix and fixed halves: ||p_D||^2, ||p_A||^2, ||y_A||^2, the three
    pairwise inner products of means, SSE, TSS (field: P; shadow: Q)."""
    out = {}
    src = {}
    for k, snp in acc_p.prefixes().items():
        src[f"K{k}"] = (snp, acc_a.prefixes()[k], acc_y.prefixes()[k])
    hp, ha, hy = acc_p.halves(), acc_a.halves(), acc_y.halves()
    for hn in c1.HALF_IDS:
        if hp[hn]["sum"] is not None:
            src[hn] = (hp[hn], ha[hn], hy[hn])
    for key, (sp, sa, sy) in src.items():
        np_, na, ny = sp["n"], sa["n"], sy["n"]
        pd = c1.mean_sq_norm(sp)
        pa = c1.mean_sq_norm(sa)
        yy = c1.mean_sq_norm(sy)
        i_pa = c1.inner_means(sp, sa)
        i_py = c1.inner_means(sp, sy)
        i_ay = c1.inner_means(sa, sy)
        out[key] = {"n": np_, "pD_sq": pd, "pA_sq": pa, "yA_sq": yy,
                    "inner_pD_pA": i_pa, "inner_pD_yA": i_py, "inner_pA_yA": i_ay,
                    "sse": pd + yy - 2.0 * i_py, "tss": yy}
    return out


def late_run_anchor(rt, snap_pre, cell, arm, h, t_pre, native_grad, snap_step,
                    banks, predictions_path, records_path):
    """C1-Late anchor measurement. Identical leak order and core math to
    c1_run_v01.run_anchor; adds the §7.1 sufficient statistics listed above."""
    import a0h15_production_entry_v01 as pe
    dev_micros, audit_micros = banks
    torch = rt["torch"]
    tau = float(rt["cfg"]["grad_clip"])
    eta_t = float(rt["optimizer"].param_groups[0]["lr"])
    mult = eta_t / float(snap_step["lr"])

    # --- stage 1: starting points at theta_t (pre-update) --------------------
    dev_g0 = [R.to_host(R.probe_grad(rt, R._stack(rt, dev_micros[j])))
              for j in range(N_PROBES)]
    audit_g0 = [R.to_host(R.probe_grad(rt, R._stack(rt, audit_micros[j])))
                for j in range(N_PROBES)]
    v_pre, q_pre = R.v_diag_q(rt)

    clip_act = {"dev": {"g0": [], "gpred": [], "g1": []},
                "audit": {"g0": [], "gpred": [], "g1": []}}
    for key, glist in (("dev", dev_g0), ("audit", audit_g0)):
        clip_act[key]["g0"] = [c0.clip_active(c0.global_norm(g), tau)
                               for g in glist]

    pe._clip_only(rt, native_grad, snap_step)
    pe._step_once(rt)
    R._counters["optimizer_updates"] += 1
    theta_post = {n: p.detach().float() for n, p in rt["named"]}
    theta_pre = {n: snap_step["theta_start"][n].float() for n, _ in rt["named"]}
    s_t = {n: theta_post[n] - theta_pre[n] for n in theta_post}
    s_norm = c0.norm_td(s_t)
    s_hat = (c0.scale_td(s_t, 1.0 / s_norm) if s_norm > 0 else None)
    t_after = int(snap_step["step_after"])
    m_hat, v_hat = R.post_moments(rt, t_after)
    lr_next = float(rt["get_lr"]((46 + h) * 2, 256,
                                 float(rt["cfg"]["learning_rate"]))) * mult
    v_post, q_plus = R.v_diag_q(rt)          # fixed main direction, pre-audit

    u_state = dict(m_hat=m_hat, v_hat=v_hat, beta1=B1C, beta2=B2C,
                   t_prev=t_after, eps=EPS_ADAM, lr=lr_next, wd=WD,
                   theta=theta_post)
    dev_c0 = [R.to_host(c0.apply_clip(R.to_gpu(rt, g), tau)) for g in dev_g0]
    audit_c0 = [R.to_host(c0.apply_clip(R.to_gpu(rt, g), tau)) for g in audit_g0]
    dev_U0 = [R.to_host(R._shadow(c0.u_map, R.to_gpu(rt, g), **u_state, tau=tau))
              for g in dev_g0]
    audit_U0 = [R.to_host(R._shadow(c0.u_map, R.to_gpu(rt, g), **u_state, tau=tau))
                for g in audit_g0]

    c_bar = c0.ybar([R.to_gpu(rt, g) for g in audit_c0])
    a2v, b2v = c0.ema_coeffs(B2C, t_after)
    P_fixed = {n: 1.0 / (torch.sqrt(a2v * v_hat[n] + b2v * c_bar[n] * c_bar[n])
                         + EPS_ADAM) for n in c_bar}
    sv = dict(m_hat=m_hat, v_hat=v_hat, beta1=B1C, beta2=B2C, t_prev=t_after,
              eps=EPS_ADAM, lr=lr_next, wd=WD, theta=theta_post, tau=tau,
              P_fixed=P_fixed)
    sv0_d = [R.to_host(R._shadow(c0.u_map_stop_v, R.to_gpu(rt, g), **sv))
             for g in dev_g0]
    sv0_a = [R.to_host(R._shadow(c0.u_map_stop_v, R.to_gpu(rt, g), **sv))
             for g in audit_g0]

    snap_post = R.snap(rt, f"_c1late_{cell}_{arm}_t{t_pre}_post.pt")
    R.load(rt, snap_pre)

    accs = {g: c1.GroupAcc(k_prefixes=CLATE["k_prefixes"] if g in MAIN_GROUPS else (),
                           half_split=CLATE["half_split"]) for g in ALL_GROUPS}
    proj = {g: {"q+": [], "s_hat": []} for g in
            ALL_GROUPS + ["resid_field_dev", "resid_field_audit",
                          "resid_shadow_dev", "resid_shadow_audit",
                          "Zdiff_audit"]}
    resid_norm = {"dev": [], "audit": []}
    resid_shadow_norm = {"dev": [], "audit": []}
    sv_inner = {g: {"dev": [], "audit": []} for g in ("Q", "Z")}
    sv_diff = {g: {"dev": [], "audit": []} for g in ("Q", "Z")}
    sv_norms = {g: {"dev": {"full": [], "stop": [], "ratio": [], "cos": []},
                    "audit": {"full": [], "stop": [], "ratio": [], "cos": []}}
                for g in ("Q", "Z")}

    def _sv_norms(g, vec, vecs, key):
        zn = c0.norm_td(vec)
        sn_ = c0.norm_td(vecs)
        sv_norms[g][key]["full"].append(zn)
        sv_norms[g][key]["stop"].append(sn_)
        sv_norms[g][key]["ratio"].append(sn_ / zn if zn > 0 else float("nan"))
        sv_norms[g][key]["cos"].append(
            sv_inner[g][key][-1] / (zn * sn_) if zn > 0 and sn_ > 0
            else float("nan"))

    def acc_and_project(gname, vec, j):
        accs[gname].add(vec, j)
        proj[gname]["q+"].append(float(c0.dot_td(vec, q_plus)))
        if s_hat is not None:
            proj[gname]["s_hat"].append(float(c0.dot_td(vec, s_hat)))

    def _dot_or_none(d, vec):
        return float(c0.dot_td(vec, d)) if d is not None else None

    # --- stage 2: dev predictions; SAVED BEFORE ANY AUDIT READ ---------------
    dev_P = [None] * N_PROBES
    dev_Q = [None] * N_PROBES
    for j in range(N_PROBES):
        g0 = R.to_gpu(rt, dev_g0[j])
        hv, _ = R.probe_hvp(rt, R._stack(rt, dev_micros[j]), s_t)
        g1p = c0.predict_geom(g0, hv, s_t)
        clip_act["dev"]["gpred"].append(
            c0.clip_active(c0.global_norm(g1p), tau))
        P = c0.sub_td(c0.apply_clip(g1p, tau), R.to_gpu(rt, dev_c0[j]))
        Q = c0.sub_td(R._shadow(c0.u_map, g1p, **u_state, tau=tau),
                      R.to_gpu(rt, dev_U0[j]))
        Qs = c0.sub_td(R._shadow(c0.u_map_stop_v, g1p, **sv),
                       R.to_gpu(rt, sv0_d[j]))
        acc_and_project("P_D", P, j)
        acc_and_project("Q_D", Q, j)
        acc_and_project("Qstop_D", Qs, j)
        sv_inner["Q"]["dev"].append(float(c0.dot_td(Q, Qs)))
        sv_diff["Q"]["dev"].append(c0.norm_td(c0.sub_td(Q, Qs)))
        _sv_norms("Q", Q, Qs, "dev")
        dev_P[j] = R.to_host(P)
        dev_Q[j] = R.to_host(Q)
        del g0, hv, g1p, P, Q, Qs
    order_pred = next(R._order)
    pred_groups = ["P_D", "Q_D"]   # frozen §7.1: predictions are mean(P_D), mean(Q_D)
    pred_sha = R._sha_of_means({g: accs[g].prefixes() for g in pred_groups})
    pred_rec = {"cell": cell, "arm": arm, "anchor": t_pre, "kind": "predictions",
                "sha256_fixed_predictions": pred_sha,
                "k_prefixes": CLATE["k_prefixes"],
                "per_K": {g: {str(k): {"mean_norm": c1.mean_norm(sn),
                                       "se": c1.se_vec_of(sn)}
                              for k, sn in accs[g].prefixes().items()}
                          for g in pred_groups},
                "order_index": order_pred}
    with predictions_path.open("a") as f:
        f.write(json.dumps(pred_rec) + "\n")

    # --- stage 3: CONDITIONED diagnostics (post-save only) -------------------
    audit_P = [None] * N_PROBES
    audit_Q = [None] * N_PROBES
    for j in range(N_PROBES):
        g0 = R.to_gpu(rt, audit_g0[j])
        hv, _ = R.probe_hvp(rt, R._stack(rt, audit_micros[j]), s_t)
        g1p = c0.predict_geom(g0, hv, s_t)
        clip_act["audit"]["gpred"].append(
            c0.clip_active(c0.global_norm(g1p), tau))
        P = c0.sub_td(c0.apply_clip(g1p, tau), R.to_gpu(rt, audit_c0[j]))
        Q = c0.sub_td(R._shadow(c0.u_map, g1p, **u_state, tau=tau),
                      R.to_gpu(rt, audit_U0[j]))
        Qs = c0.sub_td(R._shadow(c0.u_map_stop_v, g1p, **sv),
                       R.to_gpu(rt, sv0_a[j]))
        acc_and_project("P_A", P, j)
        acc_and_project("Q_A", Q, j)
        acc_and_project("Qstop_A", Qs, j)
        sv_inner["Q"]["audit"].append(float(c0.dot_td(Q, Qs)))
        sv_diff["Q"]["audit"].append(c0.norm_td(c0.sub_td(Q, Qs)))
        _sv_norms("Q", Q, Qs, "audit")
        audit_P[j] = R.to_host(P)
        audit_Q[j] = R.to_host(Q)
        del g0, hv, g1p, P, Q, Qs

    R.load(rt, snap_post)
    for bank, P_list, Q_list, key in (
            ("dev", dev_P, dev_Q, "dev"), ("audit", audit_P, audit_Q, "audit")):
        micros = dev_micros if bank == "dev" else audit_micros
        c0_list = dev_c0 if bank == "dev" else audit_c0
        U0_list = dev_U0 if bank == "dev" else audit_U0
        sv0_list = sv0_d if bank == "dev" else sv0_a
        gname = (lambda base: f"{base}_{'D' if bank == 'dev' else 'A'}")
        for j in range(N_PROBES):
            g1 = R.probe_grad(rt, R._stack(rt, micros[j]))
            clip_act[key]["g1"].append(c0.clip_active(c0.global_norm(g1), tau))
            Y = c0.sub_td(c0.apply_clip(g1, tau), R.to_gpu(rt, c0_list[j]))
            Z = c0.sub_td(R._shadow(c0.u_map, g1, **u_state, tau=tau),
                          R.to_gpu(rt, U0_list[j]))
            Zs = c0.sub_td(R._shadow(c0.u_map_stop_v, g1, **sv),
                           R.to_gpu(rt, sv0_list[j]))
            acc_and_project(gname("Y"), Y, j)
            acc_and_project(gname("Z"), Z, j)
            acc_and_project(gname("Zstop"), Zs, j)
            sv_inner["Z"][key].append(float(c0.dot_td(Z, Zs)))
            sv_diff["Z"][key].append(c0.norm_td(c0.sub_td(Z, Zs)))
            _sv_norms("Z", Z, Zs, key)
            proj["Zdiff_audit"]["q+"].append(_dot_or_none(q_plus, c0.sub_td(Z, Zs))
                                             if key == "audit" else None)
            if key == "audit":
                if s_hat is not None:
                    proj["Zdiff_audit"]["s_hat"].append(
                        float(c0.dot_td(c0.sub_td(Z, Zs), s_hat)))
                else:
                    proj["Zdiff_audit"]["s_hat"].append(None)
            rd = c0.sub_td(R.to_gpu(rt, P_list[j]), Y)
            resid_norm[key].append(c0.norm_td(rd))
            proj[f"resid_field_{key}"]["q+"].append(float(c0.dot_td(rd, q_plus)))
            if s_hat is not None:
                proj[f"resid_field_{key}"]["s_hat"].append(
                    float(c0.dot_td(rd, s_hat)))
            rd_s = c0.sub_td(R.to_gpu(rt, Q_list[j]), Z)
            resid_shadow_norm[key].append(c0.norm_td(rd_s))
            proj[f"resid_shadow_{key}"]["q+"].append(
                float(c0.dot_td(rd_s, q_plus)))
            if s_hat is not None:
                proj[f"resid_shadow_{key}"]["s_hat"].append(
                    float(c0.dot_td(rd_s, s_hat)))
            del g1, Y, Z, Zs, rd, rd_s

    d_proj = {"q+": [], "s_hat": []}
    for j in range(N_PROBES):
        D = c1.d_c_to_u(c=R.to_gpu(rt, dev_c0[j]), m_hat=m_hat, v_hat=v_hat,
                        beta1=B1C, beta2=B2C, t_prev=t_after, eps=EPS_ADAM,
                        lr=lr_next)
        d_proj["q+"].append(float(c0.dot_td(D, q_plus)))
        if s_hat is not None:
            d_proj["s_hat"].append(float(c0.dot_td(D, s_hat)))
    proj["d_c_to_u"] = d_proj

    def crossings(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    record = {"cell": cell, "arm": arm, "anchor": t_pre, "h": h,
              "t_after": t_after, "eta_t": eta_t, "lr_next": lr_next,
              "tau": tau, "s_norm": s_norm,
              "v_pre": v_pre, "v_post": v_post,
              "q_pre_norm": c0.norm_td(q_pre), "q_plus_norm": c0.norm_td(q_plus),
              "clip_activity": {
                  key: {"active_g0": sum(clip_act[key]["g0"]),
                        "active_gpred": sum(clip_act[key]["gpred"]),
                        "active_g1": sum(clip_act[key]["g1"]),
                        "cross_g0_gpred": crossings(clip_act[key]["g0"],
                                                    clip_act[key]["gpred"]),
                        "cross_g0_g1": crossings(clip_act[key]["g0"],
                                                 clip_act[key]["g1"]),
                        "cross_gpred_g1": crossings(clip_act[key]["gpred"],
                                                    clip_act[key]["g1"]),
                        "n": N_PROBES}
                  for key in ("dev", "audit")},
              "group_stats": {g: _group_stats_all(accs[g]) for g in ALL_GROUPS},
              "triple_stats": {
                  "field": _triple_stats(accs["P_D"], accs["P_A"], accs["Y_A"]),
                  "shadow": _triple_stats(accs["Q_D"], accs["Q_A"], accs["Z_A"])},
              "same_probe_resid_norm_mean_se": {k: R._mean_se(v)
                                                for k, v in resid_norm.items()},
              "same_probe_resid_shadow_norm_mean_se": {
                  k: R._mean_se(v) for k, v in resid_shadow_norm.items()},
              "stopv_pairs": {g: {b: {"inner_mean_se": R._mean_se(sv_inner[g][b]),
                                      "diff_norm_mean_se": R._mean_se(sv_diff[g][b])}
                                  for b in ("dev", "audit")}
                              for g in ("Q", "Z")},
              "stopv_norms": {g: {b: {m: R._mean_se(sv_norms[g][b][m])
                                      for m in ("full", "stop", "ratio", "cos")}
                                  for b in ("dev", "audit")}
                              for g in ("Q", "Z")},
              "fixed_dir_proj": {g: {d: c1.fixed_direction_stats(
                  [x for x in v if x is not None])
                                     for d, v in proj[g].items()}
                                 for g in proj},
              "order_index_pred": order_pred,
              "order_index_audit": next(R._order)}

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
    del accs, dev_P, audit_P, dev_Q, audit_Q, means
    del dev_g0, audit_g0, dev_c0, audit_c0, dev_U0, audit_U0, sv0_d, sv0_a
    return record


R.run_anchor = late_run_anchor


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def run_smoke(gpu):
    return R.run_smoke(gpu)


def run_main(gpu):
    R._counters = R._new_counters()
    R._t_start = time.perf_counter()
    R._order = iter(range(10 ** 9))
    if not (OUT / "smoke_checks.json").exists():
        raise SystemExit("SMOKE_CHECKS_MISSING")
    floor = json.loads((OUT / "smoke_checks.json").read_text())["numerical_floor_same_probe"]
    from fr import a0k_closed_loop_runner_v01 as a0k

    rt = a0k.build_live_runtime(2030, f"cuda:{gpu}")
    ctx = R.make_ctx()
    banks_d = json.loads((RUNS / "probe_banks.json").read_text())
    banks = (banks_d["micros"]["dev"], banks_d["micros"]["audit"])
    R._vdiag_tensors = R.v_diag_tensors_from_split(rt)
    predictions_path = RUNS / "late_predictions.jsonl"
    records_path = RUNS / "late_records.jsonl"
    steps_path = RUNS / "late_steps.jsonl"
    for f in (predictions_path, records_path, steps_path):
        if f.exists():
            f.unlink()

    anchors = CLATE["anchors_executed_this_round"]
    path_lengths = {"A": 49}
    try:
        provenance = []
        for cell_name, cell in R.CELLS.items():
            stream = R.make_stream(rt, cell["stream_seed"])
            provenance.append({
                "cell": cell_name,
                "checkpoint": str(R.BANK_CKPT),
                "stream_seed": cell["stream_seed"],
                "bank_global_step": 46,
                "replay_updates_to_reach_t48": 48,
                "anchor_update_h": 49,
                "cumulative_step_at_anchor": 46 + 48,
                "t_after_post_update": 95,
                "arm": "A", "u": 0.0,
                "anchors": "A@t48",
                "state_recovery": CLATE["state_recovery"],
                "probe_bank_source": "C1 runs/probe_banks.json (same seeds/order)",
            })
            for arm_name, T in path_lengths.items():
                u = 0.0
                anchor_ts = {t for a, t in anchors[cell_name] if a == arm_name}
                R.rollout(rt, ctx, stream, cell_name, arm_name, u, T, anchor_ts,
                          banks, "late_main", predictions_path, records_path,
                          steps_path, diag_steps=False)
        with open(OUT / "state_and_probe_provenance.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(provenance[0].keys()))
            w.writeheader()
            w.writerows(provenance)
    finally:
        R._log("late_main", "ALL", "ALL", R._counters["optimizer_updates"],
               note="late_main_complete" if R._counters["optimizer_updates"] > 0
               else "late_main_aborted")
        wall = time.perf_counter() - R._t_start
        smoke_c = json.loads((OUT / "smoke_checks.json").read_text())["counters"]
        this = {k: R._counters[k] for k in
                ("optimizer_updates", "grad_evals", "hvp_calls",
                 "micro_forward_calls", "shadow_steps", "shadow_seconds")}
        round_tot = {k: this[k] + smoke_c.get(k, 0) for k in this}
        summary = {"counters_this_process": this,
                   "counters_smoke_process": smoke_c,
                   "counters_round_total": round_tot,
                   "caps": {k: CLATE["budget"][k] for k in
                            ("optimizer_updates_max", "grad_evals_max",
                             "hvp_calls_max", "micro_forward_calls_max",
                             "gpu_wall_s_max")},
                   "within_caps": all(
                       round_tot[k] <= CLATE["budget"][k + "_max"]
                       for k in ("optimizer_updates", "grad_evals", "hvp_calls",
                                 "micro_forward_calls"))
                   and wall <= CLATE["budget"]["gpu_wall_s_max"],
                   "coverage": "2/2 planned late anchors (R/A/t48, C/A/t48); "
                               "original C1 6-anchor coverage completed to 6/6 "
                               "in aggregate reporting (OLD4 + LATE2)",
                   "floor_used_from_smoke": floor,
                   "shadow_seconds": R._counters["shadow_seconds"],
                   "wall_run_s": wall}
        (OUT / "resource_log.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("LATE_MAIN_COMPLETE", json.dumps(summary))
    return 0 if summary["within_caps"] else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["smoke", "main", "reference_tests"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(
        json.dumps(CLATE, indent=1, ensure_ascii=False) + "\n")
    if args.phase == "reference_tests":
        import subprocess
        raise SystemExit(subprocess.call(
            [sys.executable, "-m", "unittest",
             "research.tests.test_c1_late_reference_v01", "-v"],
            cwd=str(PROJECT_ROOT),
            env={"PATH": "/usr/bin:/bin",
                 "CUDA_VISIBLE_DEVICES": "",
                 "PYTHONPATH": "research/src"}))
    raise SystemExit({"smoke": run_smoke, "main": run_main}[args.phase](args.gpu))
