#!/usr/bin/env python3
"""FR B0.5 mechanism identification runner (v01).

Frozen config: outputs/fr_b05_mechanism_identification_v01/config.json
(read it before editing anything here).

Structure per cell: arm A (native) runs first and stores per-step theta/m/v
references in CPU memory (<=3.5GB, freed between cells); arms B (persistent),
C (pulse4_release), D (pulse4 + m/v transplant intervention) then run with
per-step distances against the stored A references. All arms share the same
checkpoint, CRN stream, and frozen step primitives (pe._baseline_no_step +
pe._clip_and_step), identical to B0/a0o/a0q/a0u.

Arm D is an INTERVENTION arm (m/v transplant at h=4), not a training method.

Budget: project cap 18000 optimizer updates total (B0 consumed 14208);
this runner hard-stops if the cap would be exceeded.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/fr_b05_mechanism_identification_v01"
RUNS = OUT / "runs"
BUDGET_LOG = OUT / "gpu_budget_log.json"
V07 = PROJECT_ROOT / "outputs/e3_confirmatory/v07"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "minimind"))

import fr.a0k_closed_loop_runner_v01 as a0k  # noqa: E402
import fr.b05_mechanism_v01 as b5  # noqa: E402

T_STEPS = 64
DELTA_U = 0.05
VAL_EVERY = 4
H2_MAX_LAG = 16
H2_W = 9
STREAM = 7500
MAIN_CELLS = [(2030, 46), (2030, 82), (2031, 46), (2031, 82), (2032, 46), (2032, 82)]
UPDATE_CAP_TOTAL = 18000
PRIOR_CONSUMED = 14208  # B0 totals, frozen

BUDGET = {"optimizer_updates": 0, "eval_calls": 0, "theta_captures": 0,
          "opt_captures": 0}

STEP_FIELDS = ["tag", "seed", "step", "h", "arm", "u", "lr_native", "lr_applied",
               "multiplier", "preclip_norm", "clip_active", "grad_norm",
               "update_norm", "update_cos_prev", "train_loss", "tokens",
               "d_theta", "d_m", "d_v", "cos_sep", "v_norm", "s_norm", "r_norm",
               "wall_s"]
VAL_FIELDS = ["tag", "seed", "step", "arm", "u", "h", "v_eval", "v_score"]
H2_FIELDS = ["tag", "seed", "step", "lag", "c_v_raw", "c_v_resid", "dir_persist"]


def pick_gpu() -> str:
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    best, best_free = None, -1
    for line in out.stdout.strip().splitlines():
        idx, used, total = [x.strip() for x in line.split(",")]
        free = float(total[:-3]) - float(used[:-3])
        if free > best_free:
            best, best_free = int(idx), free
    dev = f"cuda:{best}"
    print(f"[gpu] selected {dev} (free {best_free:.1f} MiB)", flush=True)
    return dev


def log_phase(name, t0, t1, detail, gpu):
    entry = {"phase": name, "t_start": t0, "t_end": t1,
             "wall_s": round(t1 - t0, 3), "gpu": gpu, "detail": detail,
             **{k: v for k, v in BUDGET.items()}}
    log = json.loads(BUDGET_LOG.read_text()) if BUDGET_LOG.exists() else {"phases": []}
    log["phases"].append(entry)
    log["TOTALS"] = dict(BUDGET)
    log["UPDATE_CAP_TOTAL"] = UPDATE_CAP_TOTAL
    log["PRIOR_CONSUMED"] = PRIOR_CONSUMED
    BUDGET_LOG.write_text(json.dumps(log, indent=1))


def budget_remaining() -> int:
    return UPDATE_CAP_TOTAL - PRIOR_CONSUMED - BUDGET["optimizer_updates"]


def guard_budget(T: int) -> bool:
    if BUDGET["optimizer_updates"] + T > UPDATE_CAP_TOTAL - PRIOR_CONSUMED:
        print(f"[budget] HARD STOP: need {T}, remaining {budget_remaining()}",
              flush=True)
        return False
    return True


class _TrainLossProbe:
    """Per-forward training loss (eval-mode forwards filtered out)."""

    def __init__(self, model):
        self.values = []
        self._handle = model.register_forward_hook(self._hook)

    def _hook(self, module, _inp, out):
        if module.training:
            self.values.append(float((out.loss + out.aux_loss).detach().float().cpu()))

    def step_loss(self, n_micro):
        vals = self.values[-n_micro:]
        return float(sum(vals) / max(len(vals), 1))

    def close(self):
        self._handle.remove()


def make_stream(rt, stream_seed, n_micro):
    from fr.stage_c_runtime import batches_from_train_pool
    from fr.stream_rng import derive_stream_seed
    cfg = rt["cfg"]
    batch_seed = derive_stream_seed(int(stream_seed), 404)  # frozen A0K recipe
    batches = batches_from_train_pool(rt["train_indices"], int(cfg["batch_size"]),
                                      batch_seed, n_micro)
    return {"stream_id": 0, "stream_seed": int(stream_seed), "batches": batches}


def cell_snapshot(rt, seed, step):
    from fr.full_state import save_full_training_state
    snap = RUNS / f"_cell_snap_s{seed}_t{step}.pt"
    save_full_training_state(snap, rt["model"], rt["optimizer"], rt["scaler"],
                             metadata={"protocol": "FR_B0_5_MECHANISM_ID",
                                       "train_seed": int(seed),
                                       "event_step": int(step)})
    return snap


def opt_state_capture(rt):
    torch = rt["torch"]
    st = rt["optimizer"].state_dict()["state"]
    return {k: {f: (t.detach().to("cpu").clone() if torch.is_tensor(t) else t)
                for f, t in v.items()}
            for k, v in st.items()}


# ---------------------------------------------------------------------------
# core per-arm rollout
# ---------------------------------------------------------------------------

def run_arm_full(rt, ctx, stream, T, arm_name, u, refs, val_writer, tag,
                 seed, step, step_writer, h2_writer, do_h2=False,
                 transplant_at4=None, opt_capture_H=()):
    """One T-step rollout with per-step recordings.

    refs: from a previous native pass — theta/m/v per-step CPU captures,
    per-step native update vectors (GPU), lr_native and s_A lists. Pass None
    for the native arm itself; its return value then provides the references.

    transplant_at4 (arm D only): {"m": {k: t}, "v": {k: t}} with arm A's
    post-step-4 exp_avg / exp_avg_sq captures; applied IN PLACE right after
    the h=4 optimizer step. Theta and the optimizer step counter are untouched.
    """
    import a0g_decision_utility_v01 as a0g
    import a0h15_production_entry_v01 as pe
    from fr.rng import capture_rng_state, restore_rng_state
    from fr.minimind_v03_runtime import raw_model

    torch = rt["torch"]
    cfg = rt["cfg"]
    named = rt["named"]
    accum = int(cfg["accumulation_steps"])
    tau0 = float(cfg["grad_clip"])
    dev = next(rt["model"].parameters()).device
    is_native = refs is None

    def mult(h):
        if arm_name == "native":
            return 1.0
        if arm_name == "persistent":
            return math.exp(u)
        # pulse4_release / D_transplant
        return math.exp(u) if h <= 4 else 1.0

    probe = _TrainLossProbe(rt["model"])
    pe._restore_and_seed(rt, ctx, stream)
    readouts, opt_caps = {}, {}
    rows = []
    prev_update = None
    vel_seq = []
    theta_refs, m_refs, v_refs, update_refs = [], [], [], []

    for h in range(1, int(T) + 1):
        th0 = time.perf_counter()
        native_gradient, _bt, snap = pe._baseline_no_step(
            rt, ctx, stream["batches"], h, want_snapshot=True)
        BUDGET["optimizer_updates"] += 1
        m_mult = float(mult(h))
        if m_mult != 1.0:
            for g in rt["optimizer"].param_groups:
                g["lr"] = float(snap["lr"]) * m_mult
        preclip, _clipped = pe._clip_and_step(rt, native_gradient, snap)

        if transplant_at4 is not None and h == 4:
            st = rt["optimizer"].state_dict()["state"]
            for k, sub in st.items():
                sub["exp_avg"].copy_(transplant_at4["m"][k].to(dev))
                sub["exp_avg_sq"].copy_(transplant_at4["v"][k].to(dev))

        theta_now = {k: v.detach().to("cpu").clone()
                     for k, v in raw_model(rt["model"]).state_dict().items()}
        st_now = rt["optimizer"].state_dict()["state"]
        m_now = {k: sub["exp_avg"].detach().to("cpu").clone() for k, sub in st_now.items()}
        v_now = {k: sub["exp_avg_sq"].detach().to("cpu").clone() for k, sub in st_now.items()}
        BUDGET["theta_captures"] += 1
        BUDGET["opt_captures"] += 1
        # distance/cosine bookkeeping is over the trainable parameter set
        # (state_dict additionally contains frozen buffers such as rotary
        # tables that the optimizer never touches)
        named_names = [name for name, _p in named]
        theta_named = {k: theta_now[k] for k in named_names}

        upd = torch.cat([(p.detach() - snap["theta_start"][name]).reshape(-1).float()
                         for name, p in named])
        upd_norm = float(torch.linalg.vector_norm(upd))
        cos_prev = (float(torch.dot(prev_update, upd)
                          / (torch.linalg.vector_norm(prev_update) * upd_norm))
                    if prev_update is not None else None)
        prev_update = upd
        grad_norm = math.sqrt(sum(
            float((g.float() ** 2).sum()) for g in native_gradient.values()))
        tokens = sum(int(x.numel()) for x, _y in snap["batch_tensors"])

        d_theta = d_m = d_v = cos_sep = 0.0
        v_norm = s_norm = r_norm = 0.0
        if is_native:
            update_refs.append(upd.detach().clone())
            vel_seq.append(upd)
        else:
            d_theta = flat_sq_dist(torch, dev, theta_named, refs["theta"][h - 1])
            d_m = flat_sq_dist(torch, dev, m_now, refs["m"][h - 1])
            d_v = flat_sq_dist(torch, dev, v_now, refs["v"][h - 1])
            delta = flat_sub(torch, dev, theta_named, refs["theta"][h - 1])
            ref_u = refs["updates"][h - 1]
            n_d = float(torch.linalg.vector_norm(delta))
            n_r = float(torch.linalg.vector_norm(ref_u))
            cos_sep = (float(torch.dot(delta, ref_u) / (n_d * n_r))
                       if n_d > 0.0 and n_r > 0.0 else None)

        if is_native:
            theta_refs.append(theta_named)
            m_refs.append(m_now)
            v_refs.append(v_now)

        rows.append({"tag": tag, "seed": seed, "step": step, "h": h, "arm": arm_name,
                     "u": u, "lr_native": float(snap["lr"]),
                     "lr_applied": float(rt["optimizer"].param_groups[0]["lr"]),
                     "multiplier": float(rt["optimizer"].param_groups[0]["lr"]) / float(snap["lr"]),
                     "preclip_norm": float(preclip), "clip_active": bool(preclip > tau0),
                     "grad_norm": grad_norm, "update_norm": upd_norm,
                     "update_cos_prev": cos_prev, "train_loss": probe.step_loss(accum),
                     "tokens": tokens, "d_theta": d_theta, "d_m": d_m, "d_v": d_v,
                     "cos_sep": cos_sep, "v_norm": v_norm, "s_norm": s_norm,
                     "r_norm": r_norm, "wall_s": time.perf_counter() - th0})

        if h % VAL_EVERY == 0 or h == T:
            rng_now = capture_rng_state()
            v_eval = float(a0g._val_loss_only(torch, rt["model"],
                                              rt["val_eval_tensors"], rt["autocast_ctx"]))
            v_score = float(a0g._val_loss_only(torch, rt["model"],
                                               rt["val_score_tensors"], rt["autocast_ctx"]))
            restore_rng_state(rng_now)
            BUDGET["eval_calls"] += 2
            readouts[str(h)] = {"v_eval": v_eval, "v_score": v_score}
            val_writer.writerow({"tag": tag, "seed": seed, "step": step,
                                 "arm": arm_name, "u": u, "h": h,
                                 "v_eval": v_eval, "v_score": v_score})
        if h in opt_capture_H:
            opt_caps[h] = opt_state_capture(rt)

    probe.close()

    if do_h2 and len(vel_seq) >= H2_MAX_LAG + 2:
        V = torch.stack([x.float() for x in vel_seq])            # [T, D] GPU
        sm = moving_avg_rows(V, H2_W)
        S = sm[1:] - sm[:-1]                                     # smooth velocity
        R = V[1:] - S                                            # residual velocity
        # torch-native equivalents of b5.normalized_autocorr / direction_persistence
        # (b5 python versions are the CPU-tested reference; these avoid 5M-wide
        # python lists)
        def t_autocorr(M):
            den = float((M * M).sum())
            out = []
            for d in range(1, H2_MAX_LAG + 1):
                if M.shape[0] - d < 2 or den == 0.0:
                    out.append(float("nan"))
                else:
                    out.append(float((M[:-d] * M[d:]).sum()) / den)
            return out

        def t_persist(M):
            out = []
            for d in range(1, H2_MAX_LAG + 1):
                if M.shape[0] - d < 1:
                    out.append(float("nan"))
                    continue
                a, b = M[:-d], M[d:]
                na = a.norm(dim=1)
                nb = b.norm(dim=1)
                ok = (na > 0) & (nb > 0)
                if not bool(ok.any()):
                    out.append(float("nan"))
                    continue
                c = (a[ok] * b[ok]).sum(dim=1) / (na[ok] * nb[ok])
                out.append(float(c.mean()))
            return out

        c_raw = t_autocorr(V)
        c_res = t_autocorr(R)
        pers = t_persist(S)
        for lag in range(1, H2_MAX_LAG + 1):
            h2_writer.writerow({"tag": tag, "seed": seed, "step": step,
                                "lag": lag, "c_v_raw": c_raw[lag - 1],
                                "c_v_resid": c_res[lag - 1],
                                "dir_persist": pers[lag - 1]})
        v_n = V.norm(dim=1).tolist()
        s_n = S.norm(dim=1).tolist()
        r_n = R.norm(dim=1).tolist()
        for i, row in enumerate(rows):
            if i == 0:
                row["v_norm"], row["s_norm"], row["r_norm"] = 0.0, 0.0, 0.0
            else:
                row["v_norm"], row["s_norm"], row["r_norm"] = v_n[i], s_n[i - 1], r_n[i - 1]

    for row in rows:
        step_writer.writerow(row)

    out_refs = {"theta": theta_refs, "m": m_refs, "v": v_refs,
                "updates": update_refs,
                "lr_native": [r["lr_native"] for r in rows],
                "s_A": b5.cumulative_lr_time([r["lr_native"] for r in rows],
                                             [r["multiplier"] for r in rows])}
    return readouts, out_refs, opt_caps


def moving_avg_rows(V, w):
    """Centered moving average along dim 0 with edge-shrinking window. V: [T, D]."""
    torch = __import__("torch")
    T = V.shape[0]
    half = w // 2
    csum = torch.cumsum(V, dim=0)
    out = torch.empty_like(V)
    for i in range(T):
        lo = max(0, i - half)
        hi = min(T, i + half + 1)
        out[i] = (csum[hi - 1] - (csum[lo - 1] if lo > 0 else 0.0)) / float(hi - lo)
    return out


def flat_sq_dist(torch, dev, now_cpu, ref_cpu):
    """||now - ref||_F over all entries of two same-structure state dicts."""
    acc = 0.0
    for k in now_cpu:
        d = now_cpu[k].to(dev).float() - ref_cpu[k].to(dev).float()
        acc += float(torch.dot(d.reshape(-1), d.reshape(-1)))
    return math.sqrt(acc)


def flat_sub(torch, dev, now_cpu, ref_cpu):
    """flattened (now - ref) as a single fp32 GPU vector."""
    parts = [(now_cpu[k].to(dev).float() - ref_cpu[k].to(dev).float()).reshape(-1)
             for k in now_cpu]
    return torch.cat(parts)


# ---------------------------------------------------------------------------
# cell block
# ---------------------------------------------------------------------------

def run_cell(rt, ctx, snap, seed, step, stream, T, tag, step_writer, val_writer,
             h2_writer, cells_f,
             arms=("native", "persistent", "pulse4_release", "D_transplant")):
    if not guard_budget(T * len(arms)):
        return False
    t0 = time.perf_counter()
    read_A, refs, opt_A = run_arm_full(
        rt, ctx, stream, T, "native", 0.0, None, val_writer, tag, seed, step,
        step_writer, h2_writer, do_h2=(T == T_STEPS),
        opt_capture_H=(4,) if T <= 8 else ())
    transplant_src = {"m": refs["m"][3], "v": refs["v"][3]} if T >= 4 else None
    results = {"A": read_A, "A_opt4": opt_A}
    for arm in arms:
        if arm == "native":
            continue
        opt_cap = (4,) if T <= 8 else ()
        if arm == "persistent":
            r, _r, _o = run_arm_full(rt, ctx, stream, T, "persistent", DELTA_U,
                                     refs, val_writer, tag, seed, step,
                                     step_writer, h2_writer, opt_capture_H=opt_cap)
            results["B"] = r
        elif arm == "pulse4_release":
            r, _r, o = run_arm_full(rt, ctx, stream, T, "pulse4_release", DELTA_U,
                                    refs, val_writer, tag, seed, step,
                                    step_writer, h2_writer, opt_capture_H=opt_cap)
            results["C"] = r
            results["C_opt4"] = o
        elif arm == "D_transplant":
            r, _r, o = run_arm_full(rt, ctx, stream, T, "D_transplant", DELTA_U,
                                    refs, val_writer, tag, seed, step,
                                    step_writer, h2_writer,
                                    transplant_at4=transplant_src,
                                    opt_capture_H=opt_cap)
            results["D"] = r
            results["D_opt4"] = o
    rec = {"tag": tag, "seed": seed, "step": step,
           "stream_seed": stream["stream_seed"], "T": T, "arms": list(arms),
           "readouts": {k: results[k] for k in ("A", "B", "C", "D") if k in results},
           "wall_s": round(time.perf_counter() - t0, 3)}
    json_rec = dict(rec)
    json_rec["opt4_captured"] = {k: sorted(results[k].keys()) for k in
                                 ("A_opt4", "C_opt4", "D_opt4") if k in results}
    cells_f.write(json.dumps(json_rec) + "\n")
    cells_f.flush()
    rec["opt4"] = {k: results[k] for k in ("A_opt4", "C_opt4", "D_opt4") if k in results}
    del refs, transplant_src
    return True, rec


def setup_cell(gpu, seed, step):
    from fr.full_state import load_full_training_state
    rt = a0k.build_live_runtime(seed, gpu)
    bank_ckpt = V07 / f"seed_{seed}/states/state_step_{step:03d}.pt"
    load_full_training_state(bank_ckpt, rt["model"], rt["optimizer"], rt["scaler"],
                             restore_rng=False)
    rt["model"].train()
    rt["optimizer"].zero_grad(set_to_none=True)
    snap = cell_snapshot(rt, seed, step)
    ctx = a0k.make_event_ctx(rt, {"stream_seed": STREAM, "lookahead_batches": [],
                                  "lookahead_sha256": None}, seed, step, snap)
    return rt, ctx, snap


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def smoke(gpu):
    from fr.full_state import load_full_training_state
    from fr import a0q_response_control_map_v01 as a0q
    import a0h15_production_entry_v01 as pe
    import torch as _torch

    t_smoke0 = time.perf_counter()
    checks = {}
    T = 8
    seed, step = MAIN_CELLS[0]
    rt, ctx, snap = setup_cell(gpu, seed, step)

    h_model_before = a0k.model_hash(rt)
    h_opt_before = a0k.opt_hash(rt)
    load_full_training_state(snap, rt["model"], rt["optimizer"], rt["scaler"],
                             restore_rng=False)
    checks["snapshot_identity"] = bool(a0k.model_hash(rt) == h_model_before
                                       and a0k.opt_hash(rt) == h_opt_before)

    stream = make_stream(rt, STREAM, 2 * T_STEPS)
    fstep = open(RUNS / "smoke_step_records.csv", "w", newline="")
    fval = open(RUNS / "smoke_val_curves.csv", "w", newline="")
    fh2 = open(RUNS / "smoke_h2.csv", "w", newline="")
    sw = csv.DictWriter(fstep, STEP_FIELDS)
    vw = csv.DictWriter(fval, VAL_FIELDS)
    hw = csv.DictWriter(fh2, H2_FIELDS)
    sw.writeheader(); vw.writeheader(); hw.writeheader()

    with open(RUNS / "smoke_cells.jsonl", "w") as fc:
        ok, rec = run_cell(rt, ctx, snap, seed, step, stream, T, "smoke", sw, vw,
                           hw, fc)
    rd = rec["readouts"]

    if ok and guard_budget(T):
        rA2, refs2, _o2 = run_arm_full(rt, ctx, stream, T, "native", 0.0, None,
                                       vw, "smoke_crn", seed, step, sw, hw)
        checks["crn_native_repeat"] = bool(rd["A"] == rA2)
    if ok and guard_budget(T):
        run_arm_full(rt, ctx, stream, T, "native", 0.0, refs2, vw,
                     "smoke_selfdist", seed, step, sw, hw)
        fstep.flush()
        rows_sd = [r for r in csv.DictReader(open(RUNS / "smoke_step_records.csv"))
                   if r["tag"] == "smoke_selfdist"]
        checks["distances_native_zero"] = bool(
            rows_sd and all(float(r["d_theta"]) == 0.0 and float(r["d_m"]) == 0.0
                            and float(r["d_v"]) == 0.0 for r in rows_sd))
    if ok:
        c_frozen = pe._new_counters()
        v_frozen = a0q.run_channel_rollout(rt, ctx, stream, T, "lr", 0.0, c_frozen)
        BUDGET["optimizer_updates"] += T
        checks["frozen_runner_agreement_endpoint"] = bool(
            rd["A"][str(T)]["v_score"] == v_frozen)
    if ok and guard_budget(T):
        global VAL_EVERY
        saved = VAL_EVERY
        VAL_EVERY = 10 ** 6  # disable mid-run evals
        rA4, refs4, _o4 = run_arm_full(rt, ctx, stream, T, "native", 0.0, None,
                                       vw, "smoke_noeval", seed, step, sw, hw)
        VAL_EVERY = saved
        checks["val_eval_rng_isolation"] = bool(
            all(_torch.equal(refs4["theta"][T - 1][k], refs2["theta"][T - 1][k])
                for k in refs2["theta"][T - 1])
            and rd["A"][str(T)]["v_score"] == rA4[str(T)]["v_score"])
    fstep.close(); fval.close(); fh2.close()

    # state identities at h=4
    checks["pulse4_eq_persistent_H4_losses"] = bool(rd["B"]["4"] == rd["C"]["4"])
    checks["D_val_eq_C_at_h4"] = bool(rd["D"]["4"] == rd["C"]["4"])
    opt_a4, opt_c4, opt_d4 = (rec["opt4"]["A_opt4"][4], rec["opt4"]["C_opt4"][4],
                              rec["opt4"]["D_opt4"][4])
    checks["D_mv_eq_A_mv_bitwise"] = bool(all(
        _torch.equal(opt_d4[k]["exp_avg"], opt_a4[k]["exp_avg"])
        and _torch.equal(opt_d4[k]["exp_avg_sq"], opt_a4[k]["exp_avg_sq"])
        for k in opt_a4))
    checks["D_step_counter_eq_A"] = bool(
        all(opt_d4[k]["step"] == opt_a4[k]["step"] for k in opt_a4))
    checks["C_differs_from_persistent_post_release_m_v"] = bool(any(
        not _torch.equal(opt_c4[k][f], opt_a4[k][f])
        for k in opt_a4 for f in ("exp_avg", "exp_avg_sq")))

    rows = [r for r in csv.DictReader(open(RUNS / "smoke_step_records.csv"))
            if r["tag"] == "smoke"]
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    lrA = [float(r["lr_native"]) for r in by_arm["native"]]
    sA = b5.cumulative_lr_time(lrA, [1.0] * len(lrA))
    checks["s_monotonic"] = bool(all(x > 0 for x in lrA)
                                 and all(sA[i + 1] > sA[i] for i in range(len(sA) - 1)))
    multC = [float(r["multiplier"]) for r in by_arm["pulse4_release"]]
    sC = b5.cumulative_lr_time(lrA, multC)
    excess = [c - a for c, a in zip(sC, sA)]
    checks["pulse_excess_frozen_post_release"] = bool(
        all(abs(excess[i] - excess[4]) < 1e-15 for i in range(4, len(excess))))
    checks["tokens_per_step_1024"] = bool(all(int(r["tokens"]) == 1024 for r in rows))
    checks["budget_within_cap"] = bool(BUDGET["optimizer_updates"]
                                       <= UPDATE_CAP_TOTAL - PRIOR_CONSUMED)

    snap.unlink()
    del rt
    (RUNS / "smoke_checks.json").write_text(json.dumps(
        {"checks": checks, "budget": dict(BUDGET)}, indent=1))
    n_pass = sum(1 for v in checks.values() if v)
    print(f"[smoke] {n_pass}/{len(checks)} checks PASS", flush=True)
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
    log_phase("smoke", t_smoke0, time.perf_counter(), {"checks": checks}, gpu)
    return all(checks.values())


# ---------------------------------------------------------------------------
# main cells
# ---------------------------------------------------------------------------

def main_cells(gpu):
    with open(RUNS / "step_records.csv", "w", newline="") as fstep, \
         open(RUNS / "val_curves.csv", "w", newline="") as fval, \
         open(RUNS / "h2_curves.csv", "w", newline="") as fh2, \
         open(RUNS / "cells.jsonl", "w") as fc:
        sw = csv.DictWriter(fstep, STEP_FIELDS)
        vw = csv.DictWriter(fval, VAL_FIELDS)
        hw = csv.DictWriter(fh2, H2_FIELDS)
        sw.writeheader(); vw.writeheader(); hw.writeheader()
        ok = True
        for (seed, step) in MAIN_CELLS:
            t0 = time.perf_counter()
            rt, ctx, snap = setup_cell(gpu, seed, step)
            stream = make_stream(rt, STREAM, 2 * T_STEPS)
            ok, _rec = run_cell(rt, ctx, snap, seed, step, stream, T_STEPS, "main",
                                sw, vw, hw, fc)
            snap.unlink()
            log_phase(f"main_s{seed}_t{step}", t0, time.perf_counter(),
                      {"updates": BUDGET["optimizer_updates"]}, gpu)
            del rt
            if not ok:
                break
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "main"], required=True)
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    gpu = pick_gpu()
    if args.mode == "smoke":
        ok = smoke(gpu)
    else:
        ok = main_cells(gpu)
    print(f"[done] mode={args.mode} ok={ok} budget={dict(BUDGET)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
