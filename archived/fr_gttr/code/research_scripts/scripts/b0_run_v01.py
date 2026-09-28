#!/usr/bin/env python3
"""FR B0 driver: 2-cell GPU smoke -> (numerical gates) -> main 24-cell
diagnosis -> 2 stress-case re-checks.

Every optimizer step is counted against the frozen 18,000-update cap
(smoke 448 + main 13,824 + stress 384 planned = 14,656). Evals and theta
captures are counted separately in gpu_budget_log.json as overhead.

Semantics (frozen in config.json):
  u = log(eta/eta_native); multiplier = exp(u)
  persistent:      mult = exp(u) at every optimizer step 1..64
  pulse4_release:  mult = exp(u) at steps 1..4; native afterwards (m/v/theta
                   formed under intervention retained; m1/v1 are
                   lr-independent so post-release m/v match the persistent
                   arm bitwise given identical gradients)
  readouts H in {4,16,64} from ONE 64-step nested trajectory per arm
  all arms of a cell share one full snapshot + one future stream (CRN)
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "outputs/fr_b0_response_mechanisms_v01"
RUNS = OUT / "runs"
BUDGET_LOG = OUT / "gpu_budget_log.json"
V07 = PROJECT_ROOT / "outputs/e3_confirmatory/v07"

sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "minimind"))

import fr.a0k_closed_loop_runner_v01 as a0k  # noqa: E402

T_STEPS = 64
READOUT_H = (4, 16, 64)
U_GRID = (-0.05, -0.025, 0.0, 0.025, 0.05)
NONZERO_U = (-0.05, -0.025, 0.025, 0.05)
DEV_STREAMS = (7500, 7501)
AUDIT_STREAMS = (7502, 7503)
MAIN_SEEDS = (2030, 2031, 2032)
MAIN_STEPS = (46, 82)
STRESS_CELLS = ((2030, 46, 7110), (2030, 64, 7110))
UPDATE_CAP = 18000

BUDGET = {"optimizer_updates": 0, "eval_calls": 0, "theta_bar_evals": 0,
          "theta_captures": 0}


def pick_gpu() -> str:
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
    log["UPDATE_CAP"] = UPDATE_CAP
    BUDGET_LOG.write_text(json.dumps(log, indent=1))


class _TrainLossProbe:
    """Records per-forward training loss (eval-mode forwards filtered out)."""

    def __init__(self, model):
        self.values = []
        self._model = model
        self._handle = model.register_forward_hook(self._hook)

    def _hook(self, module, _inp, out):
        if module.training:
            self.values.append(float((out.loss + out.aux_loss).detach().float().cpu()))

    def step_loss(self, n_micro):
        vals = self.values[-n_micro:]
        return float(sum(vals) / max(len(vals), 1))

    def close(self):
        self._handle.remove()


def run_arm(rt, ctx, stream, T, mult_fn, theta_capture=True, opt_capture_H=()):
    """One T-step nested rollout. Returns (readouts, theta_caps, diag).

    Calls only frozen primitives: pe._baseline_no_step + pe._clip_and_step
    (identical step semantics to a0o/a0q/a0u rollouts).
    """
    import a0g_decision_utility_v01 as a0g
    import a0h15_production_entry_v01 as pe
    from fr.rng import capture_rng_state, restore_rng_state
    import math

    torch = rt["torch"]
    cfg = rt["cfg"]
    named = rt["named"]
    accum = int(cfg["accumulation_steps"])
    tau0 = float(cfg["grad_clip"])

    probe = _TrainLossProbe(rt["model"])
    pe._restore_and_seed(rt, ctx, stream)
    readouts, theta_caps, diag, opt_caps = {}, {}, [], {}
    prev_update = None
    for h in range(1, int(T) + 1):
        t_h0 = time.perf_counter()
        native_gradient, _, snap = pe._baseline_no_step(
            rt, ctx, stream["batches"], h, want_snapshot=True)
        BUDGET["optimizer_updates"] += 1
        m = float(mult_fn(h))
        if m != 1.0:
            for g in rt["optimizer"].param_groups:
                g["lr"] = float(snap["lr"]) * m
        preclip, _clipped = pe._clip_and_step(rt, native_gradient, snap)

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
        row = {"h": h,
               "lr_native": float(snap["lr"]),
               "lr_applied": float(rt["optimizer"].param_groups[0]["lr"]),
               "multiplier": float(rt["optimizer"].param_groups[0]["lr"]) / float(snap["lr"]),
               "preclip_norm": float(preclip),
               "clip_active": bool(preclip > tau0),
               "grad_norm": grad_norm,
               "update_norm": upd_norm,
               "update_cos_prev": cos_prev,
               "train_loss": probe.step_loss(accum),
               "tokens": tokens,
               "wall_s": time.perf_counter() - t_h0}
        diag.append(row)

        if h in READOUT_H:
            rng_now = capture_rng_state()
            v_eval = float(a0g._val_loss_only(torch, rt["model"],
                                              rt["val_eval_tensors"], rt["autocast_ctx"]))
            v_score = float(a0g._val_loss_only(torch, rt["model"],
                                               rt["val_score_tensors"], rt["autocast_ctx"]))
            restore_rng_state(rng_now)
            BUDGET["eval_calls"] += 2
            readouts[str(h)] = {"v_eval": v_eval, "v_score": v_score}
            if theta_capture:
                from fr.minimind_v03_runtime import raw_model
                theta_caps[str(h)] = {k: v.detach().to("cpu").clone()
                                      for k, v in raw_model(rt["model"]).state_dict().items()}
                BUDGET["theta_captures"] += 1
        if h in opt_capture_H:
            st = rt["optimizer"].state_dict()["state"]
            opt_caps[h] = {k: {f: (t.detach().to("cpu").clone() if torch.is_tensor(t) else t)
                               for f, t in v.items()}
                           for k, v in st.items()}
    probe.close()
    return readouts, theta_caps, diag, opt_caps


def state_dict_bitwise_equal(a, b):
    import torch
    if set(a.keys()) != set(b.keys()):
        return False
    return all(bool(torch.equal(a[k], b[k])) for k in a)


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
                             metadata={"protocol": "FR_B0_MECHANISM_VALIDATION",
                                       "train_seed": int(seed),
                                       "event_step": int(step)})
    return snap


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def smoke(gpu):
    import a0g_decision_utility_v01 as a0g
    from fr import a0q_response_control_map_v01 as a0q
    import a0h15_production_entry_v01 as pe
    from fr.full_state import load_full_training_state
    from fr.minimind_v03_runtime import raw_model
    from fr.rng import capture_rng_state, restore_rng_state

    checks = {}
    seed, step = MAIN_SEEDS[0], MAIN_STEPS[0]
    rt = a0k.build_live_runtime(seed, gpu)
    bank_ckpt = V07 / f"seed_{seed}/states/state_step_{step:03d}.pt"
    load_full_training_state(bank_ckpt, rt["model"], rt["optimizer"], rt["scaler"],
                             restore_rng=False)
    rt["model"].train()
    rt["optimizer"].zero_grad(set_to_none=True)
    snap = cell_snapshot(rt, seed, step)
    ctx = a0k.make_event_ctx(rt, {"stream_seed": DEV_STREAMS[0],
                                  "lookahead_batches": [], "lookahead_sha256": None},
                             seed, step, snap)

    # check 0: snapshot save/load identity (model + optimizer m/v/step)
    h_model_before = a0k.model_hash(rt)
    h_opt_before = a0k.opt_hash(rt)
    load_full_training_state(snap, rt["model"], rt["optimizer"], rt["scaler"],
                             restore_rng=False)
    checks["snapshot_identity"] = bool(a0k.model_hash(rt) == h_model_before
                                       and a0k.opt_hash(rt) == h_opt_before)

    streams = {x: make_stream(rt, x, 2 * T_STEPS) for x in DEV_STREAMS}

    # check 2+3: new-runner native vs frozen runner, CRN repeat
    r1, caps1, d1, _ = run_arm(rt, ctx, streams[7500], T_STEPS, lambda h: 1.0)
    r2, caps2, d2, _ = run_arm(rt, ctx, streams[7500], T_STEPS, lambda h: 1.0)
    checks["crn_native_repeat"] = bool(r1 == r2 and state_dict_bitwise_equal(caps1["64"], caps2["64"]))
    c_frozen = pe._new_counters()
    v_frozen = a0q.run_channel_rollout(rt, ctx, streams[7500], T_STEPS, "lr", 0.0, c_frozen)
    BUDGET["optimizer_updates"] += T_STEPS
    checks["frozen_runner_agreement_endpoint"] = bool(r1["64"]["v_score"] == v_frozen)
    # prefix consistency of nested readouts: fresh 4-step run == readout@4
    r4, _c4, _d4, _ = run_arm(rt, ctx, streams[7500], 4, lambda h: 1.0)
    checks["nested_readout_prefix_H4"] = bool(r4["4"] == r1["4"])

    # check 4: pulse4 == persistent at H=4 (bitwise losses + theta + m/v)
    u = 0.05
    rp, capsp, dp, optp = run_arm(rt, ctx, streams[7500], T_STEPS,
                                  lambda h: __import__("math").exp(u) if h <= 4 else 1.0,
                                  opt_capture_H=(4, 16))
    rs, capss, ds, opts = run_arm(rt, ctx, streams[7500], T_STEPS,
                                  lambda h: __import__("math").exp(u),
                                  opt_capture_H=(4, 16))
    checks["pulse4_eq_persistent_H4_losses"] = bool(rp["4"] == rs["4"])
    checks["pulse4_eq_persistent_H4_theta"] = state_dict_bitwise_equal(capsp["4"], capss["4"])
    import torch
    checks["pulse4_eq_persistent_H4_optstate"] = bool(
        optp[4].keys() == opts[4].keys() and all(
            bool(torch.equal(optp[4][k][f], opts[4][k][f]))
            for k in optp[4] for f in ("exp_avg", "exp_avg_sq", "step")))
    # check 5: release semantics at H=16 —
    #  (a) pulse m/v DIVERGED from persistent m/v (theta paths differ -> grads
    #      differ -> m/v differ): release != persistent continuation;
    #  (b) pulse m/v CONTINUED from its own H=4 state (retention-not-reset;
    #      step counter advanced by 12, exp_avg moved).
    mv_diverged = any(not bool(torch.equal(optp[16][k][f], opts[16][k][f]))
                      for k in optp[16] for f in ("exp_avg", "exp_avg_sq"))
    step_adv = all(int(optp[16][k]["step"]) == int(optp[4][k]["step"]) + 12
                   for k in optp[16])
    mv_moved = any(not bool(torch.equal(optp[16][k]["exp_avg"],
                                        optp[4][k]["exp_avg"]))
                   for k in optp[16])
    no_none = all(optp[16][k]["exp_avg"] is not None and
                  optp[16][k]["exp_avg_sq"] is not None for k in optp[16])
    theta_diff = any(not bool(torch.equal(capsp["16"][k], capss["16"][k]))
                     for k in capsp["16"] if "weight" in k or "bias" in k)
    checks["release_m_v_diverged_from_persistent_H16"] = bool(mv_diverged)
    checks["release_m_v_retained_H16"] = bool(step_adv and mv_moved and no_none)
    checks["release_theta_diverges_H16"] = bool(theta_diff)
    lr_ok = all(abs(row["multiplier"] - (round(__import__("math").exp(u), 12) if row["h"] <= 4 else 1.0)) < 1e-9
                for row in dp)
    checks["release_native_lr_restored"] = bool(lr_ok)

    # negative-u arm on second stream
    um = -0.05
    rpm, caps_pm, dpm, _ = run_arm(rt, ctx, streams[7501], 16,
                                   lambda h: __import__("math").exp(um) if h <= 4 else 1.0)
    rsm, caps_sm, dsm, _ = run_arm(rt, ctx, streams[7501], 16,
                                   lambda h: __import__("math").exp(um))
    checks["pulse4_eq_persistent_H4_losses_negu"] = bool(rpm["4"] == rsm["4"])

    # check 6: microbatch / token accounting
    checks["tokens_per_step_1024"] = bool(all(row["tokens"] == 1024 for row in d1))
    checks["n_micro_assert"] = bool(len(streams[7500]["batches"]) == 2 * T_STEPS)

    # check 7: eval isolation (RNG unchanged by readout evals)
    rng_before = capture_rng_state()
    _ = a0g._val_loss_only(rt["torch"], rt["model"], rt["val_eval_tensors"], rt["autocast_ctx"])
    rng_after = capture_rng_state()
    restore_rng_state(rng_before)
    same = all(bool(torch.equal(a, b)) for a, b in
               zip(rng_before["torch_cuda_all"], rng_after["torch_cuda_all"]))
    checks["eval_rng_isolation"] = bool(same and rt["model"].training)

    n_steps = {"smoke": BUDGET["optimizer_updates"]}
    checks["budget_after_smoke"] = n_steps
    checks["ALL_PASS"] = bool(all(v for k, v in checks.items()
                                  if isinstance(v, bool)))
    (RUNS / "smoke_checks.json").write_text(json.dumps(checks, indent=1))
    log_phase("smoke", time.perf_counter(), time.perf_counter(), checks, gpu)
    print(json.dumps(checks, indent=1), flush=True)
    del rt
    return checks["ALL_PASS"]


# ---------------------------------------------------------------------------
# main + stress
# ---------------------------------------------------------------------------

def write_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def eval_theta_bar(rt, theta_sum, n_streams):
    """theta_bar = theta_sum/n; evaluate V_eval/V_score; leave model in
    theta_bar state (caller's next run_arm re-restores from snapshot)."""
    import a0g_decision_utility_v01 as a0g
    from fr.minimind_v03_runtime import raw_model
    from fr.rng import capture_rng_state, restore_rng_state
    theta_bar = {k: (v / n_streams).to(dtype=__import__("torch").float32).to(__import__("torch").bfloat16)
                 for k, v in theta_sum.items()}
    raw_model(rt["model"]).load_state_dict(theta_bar, strict=True)
    rng_now = capture_rng_state()
    v_eval = float(a0g._val_loss_only(rt["torch"], rt["model"],
                                      rt["val_eval_tensors"], rt["autocast_ctx"]))
    v_score = float(a0g._val_loss_only(rt["torch"], rt["model"],
                                       rt["val_score_tensors"], rt["autocast_ctx"]))
    restore_rng_state(rng_now)
    BUDGET["eval_calls"] += 2
    BUDGET["theta_bar_evals"] += 1
    return {"v_eval": v_eval, "v_score": v_score}


def run_cell_block(rt, ctx, snap, seed, step, stream_seeds, arms, tag,
                   diag_writer, decomp_writer):
    """Run all arms for (seed,step) x stream_seeds; per-arm readout JSONL +
    diag CSV; theta_bar decomposition rows for multi-stream groups.

    Decomposition identity per (arm,u,H): L_ens = mean over streams of
    V(theta_x(H)); L_center = V(theta_bar), theta_bar = mean_x theta_x(H)
    (fp32 sum -> fp32 mean -> bf16 cast -> load_state_dict strict);
    J_disp = L_ens - L_center >= 0 for convex V. No cross-seed averaging:
    theta sums never leave this (seed,step) block.
    """
    import math
    import torch as _torch
    streams = {x: make_stream(rt, x, 2 * T_STEPS) for x in stream_seeds}
    n_streams = len(stream_seeds)
    # theta accumulation buffers keyed by (arm,u) -> {H: fp32 summap}
    theta_sum = {}
    # readout cache keyed by (stream, arm, u) -> readouts (avoids JSONL re-read)
    readout_cache = {}
    for x in stream_seeds:
        stream = streams[x]
        sclass = ("dev" if x in DEV_STREAMS else
                  "audit" if x in AUDIT_STREAMS else "stress")
        for (arm, u) in arms:
            if arm == "native":
                mult = lambda h: 1.0
            elif arm == "persistent":
                mult = lambda h, uu=u: math.exp(uu)
            else:  # pulse4_release
                mult = lambda h, uu=u: math.exp(uu) if h <= 4 else 1.0
            t0 = time.perf_counter()
            readouts, caps, diag, _ = run_arm(rt, ctx, stream, T_STEPS, mult)
            readout_cache[(x, arm, u)] = readouts
            rec = {"tag": tag, "seed": seed, "step": step, "stream_seed": x,
                   "stream_class": sclass, "arm": arm, "u": u,
                   "readouts": readouts}
            write_jsonl(RUNS / f"{tag}_cells.jsonl", rec)
            for row in diag:
                row.update({"tag": tag, "seed": seed, "step": step,
                            "stream_seed": x, "arm": arm, "u": u})
                diag_writer.writerow(row)
            if n_streams > 1:
                acc = theta_sum.setdefault((arm, u), {})
                for H in READOUT_H:
                    if H not in acc:
                        acc[H] = {kk: vv.to(_torch.float32)
                                  for kk, vv in caps[str(H)].items()}
                    else:
                        for kk, vv in caps[str(H)].items():
                            acc[H][kk] += vv.to(_torch.float32)
            print(f"  [{tag}] s{seed} t{step} x{x} {arm}({u:+.3f}): "
                  f"ve4={readouts['4']['v_eval']:.6f} "
                  f"ve64={readouts['64']['v_eval']:.6f} "
                  f"({time.perf_counter()-t0:.1f}s) "
                  f"budget={BUDGET['optimizer_updates']}", flush=True)
            if BUDGET["optimizer_updates"] + T_STEPS > UPDATE_CAP:
                print("BUDGET GUARD: approaching 18,000-update cap; stopping.",
                      flush=True)
                return False
    # theta_bar decomposition for complete stream groups
    for (arm, u), per_h in theta_sum.items():
        for H in READOUT_H:
            ens = {"v_eval": 0.0, "v_score": 0.0}
            for x in stream_seeds:
                ro = readout_cache[(x, arm, u)]
                for k in ens:
                    ens[k] += ro[str(H)][k] / n_streams
            center = eval_theta_bar(rt, per_h[H], n_streams)
            decomp_writer.writerow({
                "tag": tag, "seed": seed, "step": step, "arm": arm,
                "u": u, "H": H, "n_streams": n_streams,
                "L_ens_v_eval": ens["v_eval"],
                "L_center_v_eval": center["v_eval"],
                "J_disp_v_eval": ens["v_eval"] - center["v_eval"],
                "L_ens_v_score": ens["v_score"],
                "L_center_v_score": center["v_score"],
                "J_disp_v_score": ens["v_score"] - center["v_score"]})
    return True


DECOMP_FIELDS = ["tag", "seed", "step", "arm", "u", "H", "n_streams",
                 "L_ens_v_eval", "L_center_v_eval", "J_disp_v_eval",
                 "L_ens_v_score", "L_center_v_score", "J_disp_v_score"]


def main_cells(gpu):
    from fr.full_state import load_full_training_state
    arms = [("native", 0.0)] + [("persistent", u) for u in NONZERO_U] \
        + [("pulse4_release", u) for u in NONZERO_U]
    ok = True
    with open(RUNS / "main_diag.csv", "w", newline="") as fdiag, \
         open(RUNS / "main_decomposition.csv", "w", newline="") as fdec:
        diag_writer = csv.DictWriter(fdiag, fieldnames=[
            "tag", "seed", "step", "stream_seed", "arm", "u", "h",
            "lr_native", "lr_applied", "multiplier", "preclip_norm",
            "clip_active", "grad_norm", "update_norm", "update_cos_prev",
            "train_loss", "tokens", "wall_s"])
        diag_writer.writeheader()
        decomp_writer = csv.DictWriter(fdec, fieldnames=DECOMP_FIELDS)
        decomp_writer.writeheader()
        for (seed, step) in [(s, t) for s in MAIN_SEEDS for t in MAIN_STEPS]:
            t0 = time.perf_counter()
            rt = a0k.build_live_runtime(seed, gpu)
            bank_ckpt = V07 / f"seed_{seed}/states/state_step_{step:03d}.pt"
            load_full_training_state(bank_ckpt, rt["model"], rt["optimizer"],
                                     rt["scaler"], restore_rng=False)
            rt["model"].train()
            rt["optimizer"].zero_grad(set_to_none=True)
            snap = cell_snapshot(rt, seed, step)
            ctx = a0k.make_event_ctx(rt, {"stream_seed": DEV_STREAMS[0],
                                          "lookahead_batches": [],
                                          "lookahead_sha256": None},
                                     seed, step, snap)
            ok = run_cell_block(rt, ctx, snap, seed, step,
                                DEV_STREAMS + AUDIT_STREAMS, arms, "main",
                                diag_writer, decomp_writer)
            snap.unlink()
            log_phase(f"main_s{seed}_t{step}", t0, time.perf_counter(),
                      {"optimizer_updates": BUDGET["optimizer_updates"]}, gpu)
            del rt
            if not ok:
                break
    return ok


def stress_cells(gpu):
    from fr.full_state import load_full_training_state
    arms = [("native", 0.0), ("persistent", -0.05), ("persistent", 0.05)]
    with open(RUNS / "stress_diag.csv", "w", newline="") as fdiag, \
         open(RUNS / "stress_decomposition.csv", "w", newline="") as fdec:
        diag_writer = csv.DictWriter(fdiag, fieldnames=[
            "tag", "seed", "step", "stream_seed", "arm", "u", "h",
            "lr_native", "lr_applied", "multiplier", "preclip_norm",
            "clip_active", "grad_norm", "update_norm", "update_cos_prev",
            "train_loss", "tokens", "wall_s"])
        diag_writer.writeheader()
        decomp_writer = csv.DictWriter(fdec, fieldnames=DECOMP_FIELDS)
        decomp_writer.writeheader()
        for (seed, step, x) in STRESS_CELLS:
            rt = a0k.build_live_runtime(seed, gpu)
            bank_ckpt = V07 / f"seed_{seed}/states/state_step_{step:03d}.pt"
            load_full_training_state(bank_ckpt, rt["model"], rt["optimizer"],
                                     rt["scaler"], restore_rng=False)
            rt["model"].train()
            rt["optimizer"].zero_grad(set_to_none=True)
            snap = cell_snapshot(rt, seed, step)
            ctx = a0k.make_event_ctx(rt, {"stream_seed": x,
                                          "lookahead_batches": [],
                                          "lookahead_sha256": None},
                                     seed, step, snap)
            run_cell_block(rt, ctx, snap, seed, step, (x,), arms, "stress",
                           diag_writer, decomp_writer)
            snap.unlink()
            del rt
    log_phase("stress", time.perf_counter(), time.perf_counter(),
              {"optimizer_updates": BUDGET["optimizer_updates"]}, gpu)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "main", "stress", "all"],
                    required=True)
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    gpu = pick_gpu()

    if args.mode in ("smoke", "all"):
        ok = smoke(gpu)
        if not ok:
            print("SMOKE FAILED — numerical gates not met; stop per protocol.",
                  flush=True)
            return 2
        if args.mode == "smoke":
            return 0
    if args.mode in ("main", "all"):
        ok = main_cells(gpu)
        if not ok:
            print("MAIN incomplete: budget guard triggered.", flush=True)
            return 3
    if args.mode in ("stress", "all"):
        stress_cells(gpu)
    summary = {"done": True, "budget": dict(BUDGET),
               "budget_frac_of_cap": BUDGET["optimizer_updates"] / UPDATE_CAP}
    (RUNS / "driver_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
