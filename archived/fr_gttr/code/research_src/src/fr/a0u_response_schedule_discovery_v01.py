"""A0U response-guided schedule discovery pilot (core module).

Question (mandate): can LR response dynamics reveal a better TRAINING
SCHEDULE than fixed hand-designed LR rules? Response is used as an OFFLINE
schedule-discovery signal: the schedule is generated BEFORE training from
frozen response information (chi trajectories), then rolled out open-loop.
This is NOT another online step-level controller (A0T-extra lesson).

Stage 1 (offline, pure): response trajectory / phase analysis over the
frozen A0O stage1 chi bank: 12 trajectories = 3 seeds {2030,2031,2037}
x 4 streams {7110..7113}, knots at steps (46, 52, 64, 82). Frozen phases:
early={46}, middle={52,64}, late={82}.

Stage 2 (offline, pure): frozen schedule rule. Per trajectory, per knot:
R_avg = mean of chi over the previous knots (up to K=4; partial windows
allowed; empty -> keep native). R_avg < 0 -> u=+alpha; R_avg > 0 -> u=-alpha;
multiplier m = exp(u). Knots (46,52,64,82) define segment multipliers for
optimizer-step offsets 1..6 (native), 7..18 (m52), 19..36 (m64), 37..64
(m82) in a W=64 rollout from bank state step 46 (window steps 47..110).

Stage 3 (GPU): run_schedule_rollout reuses the frozen primitives
(pe._restore_and_seed / _baseline_no_step / _clip_and_step,
a0g._val_loss_only with RNG wrap at checkpoints), mirroring the frozen
a0q.run_channel_rollout semantics with a per-segment LR multiplier and
loss checkpoints at bank steps (58, 70, 82, 94, 110).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats as _stats

KNOTS = (46, 52, 64, 82)             # frozen chi-trajectory knots
K = 4                                # frozen averaging window
ALPHA = 0.05                         # frozen schedule step (log-space)
W = 64                               # frozen rollout length (optimizer steps)
START_STEP = 46                      # rollout starts from this bank state
CHECKPOINTS = (58, 70, 82, 94, 110)  # frozen eval bank-steps (offsets 12,24,36,48,64)
SEEDS = (2030, 2031, 2037)
STREAMS = (7110, 7111, 7112, 7113)
PHASES = {"early": (46,), "middle": (52, 64), "late": (82,)}
B_MULT = 1.05                        # frozen arm B constant multiplier


# ---------------------------------------------------------------------------
# Stage 1: load frozen trajectories + phase analysis (pure, offline)
# ---------------------------------------------------------------------------

def load_trajectories(cells_path):
    """Frozen A0O stage1 H=4 h_eta -> {(seed, stream): {step: chi}}."""
    trajs = {}
    for line in Path(cells_path).read_text().splitlines():
        r = json.loads(line)
        if int(r["H"]) != 4:
            continue
        trajs.setdefault((int(r["seed"]), int(r["stream_seed"])), {})[
            int(r["step"])] = float(r["h_eta"])
    for key, d in trajs.items():
        if sorted(d) != list(KNOTS):
            raise ValueError(f"trajectory {key} has knots {sorted(d)}, expected {KNOTS}")
    return trajs


def trajectory_stats(chi_by_step):
    """Frozen Stage-1 metrics for one trajectory."""
    xs = np.array([chi_by_step[s] for s in KNOTS], dtype=np.float64)
    phases = {}
    for name, steps in PHASES.items():
        p = np.array([chi_by_step[s] for s in steps], dtype=np.float64)
        phases[name] = {"mean_abs": float(np.mean(np.abs(p))),
                        "mean_sign": float(np.sign(np.mean(p))),
                        "all_negative": bool(np.all(p < 0))}
    return {"mean": float(np.mean(xs)), "var": float(np.var(xs)),
            "sign_consistency": float(np.mean(xs < 0.0)),
            "all_negative": bool(np.all(xs < 0.0)),
            "phases": phases}


def cross_seed_similarity(trajs):
    """Per stream: Pearson of the 4-knot chi vectors across seed pairs +
    per-knot sign agreement across seeds."""
    out = {}
    for stream in STREAMS:
        vecs = {s: np.array([trajs[(s, stream)][k] for k in KNOTS],
                            dtype=np.float64) for s in SEEDS}
        pairs = {}
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                a, b = vecs[SEEDS[i]], vecs[SEEDS[j]]
                pearson = (float(np.corrcoef(a, b)[0, 1])
                           if a.std() > 0 and b.std() > 0 else None)
                pairs[f"{SEEDS[i]}v{SEEDS[j]}"] = {
                    "pearson": pearson,
                    "sign_agreement": float(np.mean(np.sign(a) == np.sign(b)))}
        mat = np.array([vecs[s] for s in SEEDS])
        out[str(stream)] = {"pairs": pairs,
                            "per_knot_seed_sign_agreement":
                            [float(np.mean(np.sign(mat[:, k]) ==
                                           np.sign(np.median(mat[:, k]))))
                             for k in range(len(KNOTS))]}
    return out


# ---------------------------------------------------------------------------
# Stage 2: frozen schedule rule (pure, offline)
# ---------------------------------------------------------------------------

def build_schedule(chi_by_step):
    """Per-knot decisions and segment multipliers (frozen partial-window rule).

    Returns one entry per knot: knot, previous knots used (<=K), R_avg
    (None for the empty window at knot 46), multiplier. Multiplier applies
    to optimizer-step offsets after the knot: 46->1..6, 52->7..18,
    64->19..36, 82->37..64.
    """
    segs = []
    for i, knot in enumerate(KNOTS):
        prev_steps = [s for s in KNOTS[:i]][-K:]
        prev = [chi_by_step[s] for s in prev_steps]
        if not prev:
            segs.append({"knot": knot, "prev_steps": [], "R_avg": None,
                         "multiplier": 1.0})
            continue
        r_avg = float(np.mean(prev))
        if r_avg < 0.0:
            u = ALPHA
        elif r_avg > 0.0:
            u = -ALPHA
        else:
            u = 0.0
        segs.append({"knot": knot, "prev_steps": prev_steps, "R_avg": r_avg,
                     "multiplier": float(np.exp(u))})
    return segs


def multiplier_at_offset(h, segments):
    """Frozen segment map: offset h in 1..W -> multiplier."""
    h = int(h)
    if h <= 6:
        return 1.0                     # knot-46 segment: keep native
    if h <= 18:
        return float(segments[1]["multiplier"])   # m(52)
    if h <= 36:
        return float(segments[2]["multiplier"])   # m(64)
    return float(segments[3]["multiplier"])       # m(82)


def schedule_properties(segments):
    """Frozen schedule descriptors: n LR changes across the 4 applied
    segments, and the multiplier vector."""
    ms = [float(s["multiplier"]) for s in segments]
    n_changes = sum(1 for i in range(1, len(ms)) if ms[i] != ms[i - 1])
    up = float(np.exp(ALPHA))
    return {"multipliers": ms, "n_lr_changes": int(n_changes),
            "n_up_segments": int(sum(1 for m in ms if abs(m - up) < 1e-12)),
            "n_down_segments": int(sum(1 for m in ms if abs(m - 1.0 / up) < 1e-12)),
            "n_native_segments": int(sum(1 for m in ms if m == 1.0))}


# ---------------------------------------------------------------------------
# Stage 3: schedule rollout (GPU; frozen-primitive reuse)
# ---------------------------------------------------------------------------

def run_schedule_rollout(rt, ctx, stream, W_: int, checkpoints, mult_fn,
                         counters):
    """W native optimizer steps with per-step LR multiplier mult_fn(h) and
    validation-loss checkpoints at offsets in `checkpoints`.

    Mirrors frozen a0q.run_channel_rollout: native lr set per micro-step by
    _baseline_no_step; when m != 1 the group lr is overridden to
    snap['lr']*m BEFORE _clip_and_step (identical override point/semantics).
    m == 1 never touches anything -> bitwise native (CRN convention).
    """
    import a0g_decision_utility_v01 as a0g
    import a0h15_production_entry_v01 as pe
    from fr.rng import capture_rng_state, restore_rng_state

    pe._restore_and_seed(rt, ctx, stream)
    losses = {}
    for h in range(1, int(W_) + 1):
        native_gradient, _, snap = pe._baseline_no_step(
            rt, ctx, stream["batches"], h, want_snapshot=True)
        counters["baseline_steps"] += 1
        m = float(mult_fn(h))
        if m != 1.0:
            for g in rt["optimizer"].param_groups:
                g["lr"] = float(snap["lr"]) * m
        pe._clip_and_step(rt, native_gradient, snap)
        if h in checkpoints:
            rng_now = capture_rng_state()
            losses[str(h)] = float(a0g._val_loss_only(
                rt["torch"], rt["model"], rt["val_score_tensors"],
                rt["autocast_ctx"]))
            restore_rng_state(rng_now)
    return losses


# ---------------------------------------------------------------------------
# Metrics + statistics (pure)
# ---------------------------------------------------------------------------

def curve_metrics(checkpoint_losses):
    """Frozen optimization metrics from a {offset: loss} dict."""
    offs = sorted(int(k) for k in checkpoint_losses)
    ls = np.array([float(checkpoint_losses[str(o)]) for o in offs],
                  dtype=np.float64)
    last3 = ls[-3:]
    return {"checkpoints": {str(o): float(checkpoint_losses[str(o)])
                            for o in offs},
            "final_loss": float(ls[-1]),
            "auc": float(np.trapezoid(ls, offs)),
            "late_std": float(np.std(last3)),
            "late_slope": float(np.polyfit(offs[-3:], last3, 1)[0])}


def early_improvement(losses_a, losses_x):
    """Frozen: L_A(offset 12, bank step 58) - L_X(same)."""
    return float(losses_a["12"] - losses_x["12"])


def wilcoxon_safe(diff):
    d = np.asarray(diff, dtype=np.float64)
    nz = d[d != 0.0]
    if len(nz) < 5:
        return {"n_nonzero": int(len(nz)), "statistic": None, "p": None,
                "note": "fewer than 5 non-zero differences"}
    res = _stats.wilcoxon(d)
    return {"n_nonzero": int(len(nz)), "statistic": float(res.statistic),
            "p": float(res.pvalue)}


def paired_block(records, key_a, key_x):
    """Paired per-cell statistics of metric(key_a) - metric(key_x)."""
    d = np.array([key_a(r) - key_x(r) for r in records], dtype=np.float64)
    nz = d[d != 0.0]
    bt = _stats.binomtest(int(np.sum(d > 0)), len(d), 0.5) if len(d) else None
    return {"n": len(d), "median": float(np.median(d)) if len(d) else None,
            "mean": float(np.mean(d)) if len(d) else None,
            "frac_positive": float(np.mean(d > 0)) if len(d) else None,
            "wilcoxon_vs0": wilcoxon_safe(d),
            "binom_p_positive": float(bt.pvalue) if bt else None,
            "n_nonzero": int(len(nz))}


def verdict_case(block_ca_final, block_cb_final, block_ba_final):
    """Frozen case rule from contract (final-loss basis; AUC reported)."""
    sig = lambda b: (b["wilcoxon_vs0"]["p"] is not None
                     and b["wilcoxon_vs0"]["p"] < 0.05)
    case_a = (block_ca_final["median"] is not None
              and block_ca_final["median"] > 0 and sig(block_ca_final)
              and block_cb_final["median"] is not None
              and block_cb_final["median"] > 0 and sig(block_cb_final))
    case_b = (not sig(block_cb_final) and sig(block_ca_final)
              and block_ba_final["median"] is not None
              and block_ba_final["median"] > 0)
    case_c = (block_ca_final["median"] is not None
              and block_ca_final["median"] < 0)
    label = ("A_response_beats_fixed_and_native" if case_a
             else "B_response_like_fixed_scaling" if case_b
             else "C_response_worse" if case_c
             else "inconclusive")
    return {"case_A": bool(case_a), "case_B": bool(case_b),
            "case_C": bool(case_c), "label": label}
