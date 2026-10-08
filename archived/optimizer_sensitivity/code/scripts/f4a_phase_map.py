#!/usr/bin/env python3
"""F4-A scale-phase-seed stability map runner (frozen:
docs/F4A_SCALE_PHASE_SEED_MAP_PREREG.md).

unit  --scale v07_5m|g1a_26m|g1b_100m --seed S --step T
    One bank state: 32-step replay propagating THREE arms with F2/F3 frozen
    semantics -- B (alpha=0), C (alpha=1), SO (q0 impulse, source-off) --
    plus cheap local indicators. Atomic unit dir with DONE marker; resume =
    skip completed units. The three F3-reference g1b cells additionally
    enforce bitwise gates vs committed F3 raws + frozen-reference losses.

postprocess
    raw/ -> phase_map_long.csv, phase_map_summary.csv, healing_intervals.csv,
    indicator_correlations.csv, exploratory_models.csv, figures/, summary.json

Run with CUDA_VISIBLE_DEVICES set. Long maps belong to the tmux/setsid driver.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
LAB = ROOT.parent
MM = LAB / "minimind_base"

import torch  # noqa: E402

import f3_curvature_mechanism as f3m  # noqa: E402 (proven pipeline)
from f2_hypergradient import Rig  # noqa: E402
from opsens import f0_bank_audit_v01 as f0  # noqa: E402
from opsens import f1_mechanism_v01 as fm  # noqa: E402
from opsens import f3_curvature_mechanism_v01 as f3  # noqa: E402
from opsens.batch_manifest import manifest_sha256  # noqa: E402

OUT = ROOT / "outputs" / "f4a_phase_map"
RAW = Path(os.environ.get("F4A_RAW_ROOT") or (OUT / "raw"))
H_READOUTS = (1, 2, 4, 8, 16, 32)
EPS_METRIC = 1e-30
H_MAX = 32
ARCH = {
    "v07_5m": dict(hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
                   num_key_value_heads=4),
    "g1a_26m": dict(hidden_size=512, num_hidden_layers=7, num_attention_heads=8,
                    num_key_value_heads=4),
    "g1b_100m": dict(hidden_size=768, num_hidden_layers=13,
                     num_attention_heads=12, num_key_value_heads=6),
}
PARAMS = {"v07_5m": 4983296, "g1a_26m": 26681728, "g1b_100m": 100784000}
F3_STREAMS = {(2042, 26): 1, (2042, 64): 2, (2043, 26): 3}  # F3 CELL_STREAM (verified)
# frozen new-stream rule (prereg §3): id = 4000 + bank_index (inventory order)
BANK_ORDER = ([( "v07_5m", s, st) for s in range(2030, 2038)
               for st in (46, 52, 58, 64, 70, 76, 82)] +
              [("g1a_26m", s, st) for s in (2042, 2043, 2044)
               for st in (26, 64, 102)] +
              [("g1b_100m", s, st) for s in (2042, 2043, 2044)
               for st in (26, 64, 102)])
BANK_INDEX = {k: i for i, k in enumerate(BANK_ORDER)}
# F4A.1 prereg stream ids: 4074+i over (seed,step) pairs 2030:26..2037:102;
# canary control 2030:64 -> 4090
F4A1_STREAMS = {}
_i = 0
for _s in range(2030, 2038):
    for _t in (26, 102):
        F4A1_STREAMS[(_s, _t)] = 4074 + _i
        _i += 1
F4A1_STREAMS[(2030, 64)] = 4090
FROZEN = ROOT / "outputs" / "f3_compatibility" / "frozen_reference" / "baseline_summary.json"
F3RAW = ROOT / "outputs" / "f3_curvature_mechanism" / "raw"
F2OUT = ROOT / "outputs" / "f2_hypergradient"


class F4ARig(Rig):
    """Rig with a per-scale MiniMind architecture (everything else identical
    to the frozen f2_hypergradient.Rig)."""

    def __init__(self, device: str, scale: str):
        from model.model_minimind import MiniMindConfig
        from trainer.trainer_utils import get_lr, init_model
        from dataset.lm_dataset import PretrainDataset
        import f2_hypergradient as f2m

        mmcfg = MiniMindConfig(max_seq_len=f2m.MAX_SEQ, use_moe=False,
                               **ARCH[scale])
        model, tokenizer = init_model(mmcfg, from_weight="none",
                                      tokenizer_path=str(MM / "model"),
                                      device="cpu")
        self.template = model
        self.tokenizer = tokenizer
        self.get_lr = get_lr
        self.ds = PretrainDataset(str(f2m.D0), tokenizer, max_length=f2m.MAX_SEQ)
        self.device = device
        self.model_gpu = copy.deepcopy(model).to(device)
        part = json.loads(f2m.PARTITION.read_text())
        self.train_idx = part["train_indices"]
        self.eval_idx = part["eval_indices"]
        self.eval_batches = [self.eval_idx[i:i + 16]
                             for i in range(0, len(self.eval_idx), 16)]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def state_norms(dz):
    n_th = math.sqrt(sum(float(t.double().square().sum()) for t in dz["th"].values()))
    n_m = math.sqrt(sum(float(t.double().square().sum()) for t in dz["m"].values()))
    n_v = math.sqrt(sum(float(t.double().square().sum()) for t in dz["v"].values()))
    return n_th, n_m, n_v, math.sqrt(n_th ** 2 + n_m ** 2 + n_v ** 2)


def unit_dir(ch: str):
    return {ch: {n: torch.zeros_like(p, dtype=torch.float64) for n, p in f3m.named}}


def cheap_indicators():
    """Frozen cheap indicators from the restored state (prereg §5)."""
    named, opt = f3m.named, f3m.opt
    ind = {}
    # v-floor fractions (sqrt of bias-corrected v-hat the NEXT step sees)
    fr = {"eps": 0.0, "10eps": 0.0, "100eps": 0.0}
    total = 0
    beta2 = 0.999
    for _, p in named:
        st = opt.state[p]
        step_t = float(st["step"])
        vhat = st["exp_avg_sq"].double() / (1.0 - beta2 ** (step_t + 1.0))
        sv = torch.sqrt(vhat)
        n = sv.numel()
        total += n
        fr["eps"] += float((sv < 1e-8).sum())
        fr["10eps"] += float((sv < 1e-7).sum())
        fr["100eps"] += float((sv < 1e-6).sum())
    for k in fr:
        ind[f"fraction_sqrtv_lt_{k}"] = fr[k] / max(total, 1)
    # state-channel amplification via the F1 frozen estimator (c=0 anchor,
    # unit random direction per channel, full-state output norm; seed frozen)
    dirs = fm.channel_directions(named, 91011)
    for ch in ("theta", "m", "v"):
        out = fm.run_tangent_state_channels(
            named, opt, **{"q_" + ch: dirs[ch]})
        n = math.sqrt(sum(float(t.double().square().sum())
                          for c in ("tangent_theta", "tangent_m", "tangent_v")
                          for t in out[c].values()))
        ind[f"A_{ch}"] = n
    return ind


def cmd_unit(scale: str, seed: int, step: int, bank_dir=None) -> int:
    ud = RAW / scale / f"seed_{seed}" / f"step_{step}"
    if (ud / "DONE").exists():
        log(f"unit {scale} {seed}:{step}: DONE exists, skip")
        return 0
    ud.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = "cuda:0"
    f3m.rig = F4ARig(device, scale)
    if bank_dir:
        bp = Path(bank_dir) / f"seed_{seed}" / f"state_step_{step:03d}.pt"
    else:
        bp = fm.bank_state_path(scale, seed, step)
    f3m.state_g = f0.load_bank_state(bp)
    micro0 = int(f3m.state_g["metadata"]["global_micro_step"])
    f3m.micro0 = micro0
    if scale == "g1b_100m" and (seed, step) in F3_STREAMS:
        stream_id = F3_STREAMS[(seed, step)]
    elif scale == "v07_5m" and (seed, step) in F4A1_STREAMS and bank_dir:
        stream_id = F4A1_STREAMS[(seed, step)]
    else:
        stream_id = 4000 + BANK_INDEX[(scale, seed, step)]
    f3m.batches_g, stream_seed = f3m.rig.batches_for(stream_id)
    msha = manifest_sha256(f3m.batches_g)
    f3m.named, f3m.opt, f3m.model_cpu = f3m.rig.restore(f3m.state_g)
    cfg = {"scale": scale, "seed": seed, "step": step,
           "bank_path": str(bp),
           "params": PARAMS[scale], "arch": ARCH[scale], "micro0": micro0,
           "stream_id": stream_id, "stream_seed": stream_seed,
           "stream_manifest_sha256": msha,
           "protocol": "F4A prereg 3ac9762; F2/F3 frozen arm semantics",
           "planned_total_steps": 128, "progress": step / 128}
    (ud / "config.json").write_text(json.dumps(cfg, indent=1))
    log(f"unit {scale} {seed}:{step} stream_id={stream_id} "
        f"manifest={msha[:12]} micro0={micro0}")

    arms = {
        "B": {"th": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in f3m.named},
              "m": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in f3m.named},
              "v": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in f3m.named},
              "alpha": 0.0},
        "C": {"th": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in f3m.named},
              "m": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in f3m.named},
              "v": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in f3m.named},
              "alpha": 1.0},
        "SO": None,  # created at h==1 from q0
    }
    ind = cheap_indicators()
    traj_rows = []
    vg_cache = {}
    g1_norms = {"B": {}, "C": {}, "SO": {}}
    K_lr = None
    for h in range(1, H_MAX + 1):
        t0 = time.time()
        theta_prev, lr, grads, clipped, loss, info = f3m.primal_grads(h - 1, micro0)
        pend = {}
        for key, a in arms.items():
            if a is None:
                continue
            dc, _ = f3m.dc_for(a["th"], grads, a["alpha"], h)
            pend[key] = f3.hom_step(f3m.named, f3m.opt, clipped, dc, a)
        theta_next = f3m.primal_advance()
        if h == 1:
            q0 = f3.lr_source_q0(f3m.named, theta_prev, theta_next)
            zero_dc = {n: torch.zeros_like(p, dtype=torch.float64)
                       for n, p in f3m.named}
            arms["SO"] = {ch: {n: t.clone() for n, t in q0[ch].items()}
                          for ch in ("th", "m", "v")}
            arms["SO"]["alpha"] = 1.0
            pend["SO"] = f3.hom_step(f3m.named, f3m.opt, clipped, zero_dc,
                                     arms["SO"])
            # K_lr: HVP along the LR-source direction at the first step
            src = f3m.rig.hvp(f3m.model_cpu, f3m.batches_g, 0,
                              {n: t.to(torch.float32) for n, t in q0["th"].items()},
                              fp32=True)
            K_lr = math.sqrt(sum(float(t.double().square().sum())
                                 for t in src.values()))
        inh = {n: (theta_next[n].double() - theta_prev[n].double())
               for n, _ in f3m.named}
        row = {"h": h, "loss": loss, "clip_coef": info["coef"], "lr": lr}
        for key, a in arms.items():
            if a is None:
                continue
            out = pend[key]
            if key == "SO":
                a["th"], a["m"], a["v"] = out["th"], out["m"], out["v"]
            else:
                a["th"] = {n: out["th"][n] + inh[n] for n, _ in f3m.named}
                a["m"], a["v"] = out["m"], out["v"]
            n_th, n_m, n_v, n_z = state_norms(a)
            row[f"{key}_dtheta"] = n_th
            row[f"{key}_dz"] = n_z
            if h in H_READOUTS:
                g1_norms[key][h] = n_th if key != "SO" else n_z
            if key == "C" and h in H_READOUTS:
                if h not in vg_cache:
                    vg_cache[h] = f3m.rig.val_grad(f3m.model_cpu)
                vg = vg_cache[h]
                row["h_eta"] = sum(float((vg[n].double() * a["th"][n].double()).sum())
                                   for n, _ in f3m.named)
        traj_rows.append(row)
        log(f"unit {scale} {seed}:{step} h={h} loss={loss:.5f} "
            f"|dth|B={row['B_dtheta']:.4g} |dth|C={row['C_dtheta']:.4g} "
            f"|dz|SO={row['SO_dz']:.4g} ({time.time() - t0:.0f}s)")

    # ---- metrics ----
    def G(key, H):
        base = g1_norms[key].get(1)
        if base is None or base <= 0:
            return None
        return g1_norms[key].get(H, 0.0) / max(base, EPS_METRIC)

    metrics = {"scale": scale, "seed": seed, "step": step,
               "params": PARAMS[scale], "progress": step / 128,
               "K_lr": K_lr, "indicators": ind, "G": {}, "labels": {}}
    for H in H_READOUTS:
        gb, gc, gso = G("B", H), G("C", H), G("SO", H)
        metrics["G"][str(H)] = {
            "G_B_theta": gb, "G_C_theta": gc,
            "R_curv": (gc / (gb + EPS_METRIC)) if gb is not None and gc is not None else None,
            "G_SO_theta": gso,
            "lambda_SO": (math.log(max(gso, EPS_METRIC)) / H) if gso else None,
            "S_eta": traj_rows[H - 1]["C_dtheta"] if H <= len(traj_rows) else None,
            "h_eta": traj_rows[H - 1].get("h_eta") if H <= len(traj_rows) else None,
        }
    r32 = metrics["G"]["32"]["R_curv"]
    l32 = metrics["G"]["32"]["lambda_SO"]
    metrics["labels"]["curvature"] = (
        "CURVATURE_DOMINANT" if r32 is not None and r32 >= 10 else
        "CURVATURE_MODERATE" if r32 is not None and r32 >= 3 else
        "CURVATURE_WEAK" if r32 is not None else "NOT_ESTIMATED")
    metrics["labels"]["source_off"] = (
        "SOURCE_OFF_GROWING" if l32 is not None and l32 > 0 else
        "SOURCE_OFF_DECAYING" if l32 is not None else "NOT_ESTIMATED")
    # clip stats
    metrics["clip"] = {"clip_active_1": traj_rows[0]["clip_coef"] < 1.0,
                       "clip_coefficient_1": traj_rows[0]["clip_coef"],
                       "clip_coefficient_med": sorted(
                           r["clip_coef"] for r in traj_rows)[H_MAX // 2],
                       "lr_1": traj_rows[0]["lr"],
                       "normalized_lr_1": traj_rows[0]["lr"] / 5e-4}
    # module shares of the C-arm theta at H=32
    shares = fm.modulewise_shares(f3m.named, arms["C"]["th"])
    metrics["module_shares_C32"] = shares
    # ---- gates for the three F3-reference cells ----
    gates = None
    if scale == "g1b_100m" and (seed, step) in F3_STREAMS:
        cell = f"{seed}:{step}"
        ref_losses = json.load(FROZEN.open())["losses"][cell]
        losses = [r["loss"] for r in traj_rows]
        assert losses == list(ref_losses), f"loss gate FAIL {cell}"
        rq1 = json.load((F3RAW / f"rq1_{seed}_{step}.json").open())
        rq2 = json.load((F3RAW / f"rq2_{seed}_{step}.json").open())
        ref_c = rq1["steps"][-1]["arms"]["1.0"]["dtheta_norm"]
        ref_b = rq1["steps"][-1]["arms"]["0.0"]["dtheta_norm"]
        ref_so = rq2["steps"][-1]["arms"]["a1.0|q0_lrsource"]["dz_norm"]
        got_c, _, _, got_cz = state_norms(arms["C"])
        got_b, _, _, got_bz = state_norms(arms["B"])
        _, _, _, got_so = state_norms(arms["SO"])
        # rq1 dtheta uses its own aggregation == our n_th; SO uses dz (th/m/v)
        assert abs(got_c - ref_c) <= 0.0, f"C gate FAIL {got_c!r} vs {ref_c!r}"
        assert abs(got_b - ref_b) <= 0.0, f"B gate FAIL {got_b!r} vs {ref_b!r}"
        assert abs(got_so - ref_so) <= 0.0, f"SO gate FAIL {got_so!r} vs {ref_so!r}"
        gates = {"loss_bitwise": True, "C_dtheta_bitwise": True,
                 "B_dtheta_bitwise": True, "SO_dz_bitwise": True}
        log(f"unit {scale} {seed}:{step}: F3 GATES PASS (bitwise)")
    metrics["gates"] = gates
    tmp = ud / "metrics.json.tmp"
    tmp.write_text(json.dumps(metrics, indent=1))
    os.replace(tmp, ud / "metrics.json")
    with (ud / "trajectory.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(traj_rows[0]))
        w.writeheader()
        w.writerows(traj_rows)
    (ud / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    log(f"unit {scale} {seed}:{step}: COMPLETE ({time.time() - t_start:.0f}s)")
    return 0


# ------------------------------------------------------------ postprocess ----
def cmd_postprocess() -> int:
    import numpy as np
    units = []
    for p in sorted(RAW.glob("*/*/step_*/metrics.json")):
        units.append(json.loads(p.read_text()))
    log(f"postprocess: {len(units)} units")

    long_rows = []
    for m in units:
        for H in H_READOUTS:
            g = m["G"][str(H)]
            long_rows.append({
                "scale": m["scale"], "params": m["params"], "seed": m["seed"],
                "step": m["step"], "progress": m["progress"], "H": H,
                "G_B": g["G_B_theta"], "G_C": g["G_C_theta"],
                "R_curv": g["R_curv"], "G_SO": g["G_SO_theta"],
                "lambda_SO": g["lambda_SO"], "S_eta": g["S_eta"],
                "h_eta": g["h_eta"], "K_lr": m["K_lr"],
                "A_theta": m["indicators"]["A_theta"],
                "A_m": m["indicators"]["A_m"], "A_v": m["indicators"]["A_v"],
                "v_floor_fraction": m["indicators"]["fraction_sqrtv_lt_eps"],
                "clip_ratio": m["clip"]["clip_coefficient_1"],
                "curvature_label_32": (m["labels"]["curvature"] if H == 32 else ""),
            })
    with (OUT / "phase_map_long.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(long_rows[0]))
        w.writeheader()
        w.writerows(long_rows)

    def phase_of(p):
        return "EARLY" if p < 0.34 else ("MID" if p < 0.67 else "LATE")

    sum_rows = []
    for scale in ("v07_5m", "g1a_26m", "g1b_100m"):
        ph_bins = {}
        for m in units:
            if m["scale"] != scale:
                continue
            ph_bins.setdefault(phase_of(m["progress"]), []).append(m)
        for ph, ms in sorted(ph_bins.items()):
            for key, getter in (
                    ("log_R_curv_32", lambda m: math.log10(max(m["G"]["32"]["R_curv"], 1e-30))),
                    ("lambda_SO_32", lambda m: m["G"]["32"]["lambda_SO"]),
                    ("log_S_eta_32", lambda m: math.log10(max(m["G"]["32"]["S_eta"], 1e-300)))):
                vs = np.array([getter(m) for m in ms])
                sum_rows.append({"scale": scale, "phase": ph, "n_seed": len(ms),
                                 "quantity": key,
                                 "median": float(np.median(vs)),
                                 "mean": float(vs.mean()),
                                 "std": float(vs.std(ddof=1)) if len(vs) > 1 else 0.0,
                                 "iqr": float(np.percentile(vs, 75) - np.percentile(vs, 25)),
                                 "min": float(vs.min()), "max": float(vs.max()),
                                 "seeds": " ".join(str(m["seed"]) for m in ms)})
    with (OUT / "phase_map_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sum_rows[0]))
        w.writeheader()
        w.writerows(sum_rows)

    # healing intervals (prereg §11)
    heal_rows = []
    for scale in ("v07_5m", "g1a_26m", "g1b_100m"):
        for seed in sorted({m["seed"] for m in units if m["scale"] == scale}):
            ms = sorted([m for m in units if m["scale"] == scale
                         and m["seed"] == seed], key=lambda m: m["progress"])
            r32 = [m["G"]["32"]["R_curv"] for m in ms]
            ps = [m["progress"] for m in ms]
            status, lo, hi = "INSUFFICIENT_POINTS", None, None
            if len(ms) >= 3:
                healed_idx = None
                for i in range(len(ms)):
                    if all(v < 3 for v in r32[i:]):
                        healed_idx = i
                        break
                if healed_idx is None:
                    status = "NOT_HEALED_IN_COVERAGE"
                elif healed_idx == 0:
                    status, lo, hi = "OBSERVED", ps[0], ps[0]
                else:
                    status = "INTERVAL_CENSORED"
                    lo, hi = ps[healed_idx - 1], ps[healed_idx]
                    if r32[healed_idx] < 3 and healed_idx > 0 and \
                            all(v >= 3 for v in r32[:healed_idx]):
                        status = "INTERVAL_CENSORED"
            heal_rows.append({"scale": scale, "seed": seed, "status": status,
                              "p_heal_lower": lo, "p_heal_upper": hi,
                              "r32_seq": " ".join(f"{v:.2f}" for v in r32),
                              "p_seq": " ".join(f"{p:.3f}" for p in ps)})
    with (OUT / "healing_intervals.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(heal_rows[0]))
        w.writeheader()
        w.writerows(heal_rows)

    # indicator correlations vs log R_curv_32
    ind_rows = []
    def get_ind(m, name):
        if name == "clip_ratio":
            return m["clip"]["clip_coefficient_1"]
        if name == "K_lr":
            return m["K_lr"]
        if name == "v_floor_fraction":
            return m["indicators"]["fraction_sqrtv_lt_eps"]
        return m["indicators"][name]
    inds = ["A_v", "A_m", "A_theta", "K_lr", "v_floor_fraction", "clip_ratio"]
    y = np.array([math.log10(max(m["G"]["32"]["R_curv"], 1e-30)) for m in units])
    from scipy.stats import spearmanr
    for name in inds:
        raw = np.array([get_ind(m, name) for m in units], dtype=float)
        x = np.log10(np.maximum(raw, 1e-300))
        r = float(np.corrcoef(x, y)[0, 1])
        rs = float(spearmanr(raw, y).statistic)  # spearman on RAW indicator
        ind_rows.append({"indicator": name, "pearson_loglog": r,
                         "spearman_raw": rs, "n": len(units)})
    with (OUT / "indicator_correlations.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ind_rows[0]))
        w.writeheader()
        w.writerows(ind_rows)

    # exploratory LOSO models (prereg §12)
    mod_rows = []
    data = [(math.log10(m["params"]), m["progress"], math.log(m["step"] + 1),
             m["seed"], math.log10(max(m["G"]["32"]["R_curv"], 1e-30)))
            for m in units]
    XN = np.array([d[0] for d in data]); P = np.array([d[1] for d in data])
    LS = np.array([d[2] for d in data]); SEED = np.array([d[3] for d in data])
    Y = np.array([d[4] for d in data])
    models = {
        "A_logN_p": np.column_stack([np.ones_like(XN), XN, P]),
        "B_logN_logstep": np.column_stack([np.ones_like(XN), XN, LS]),
        "C_scale_fixed_p": None,  # built per-fit
        "D_interaction": np.column_stack([np.ones_like(XN), XN, P, XN * P]),
    }
    scales = np.array([m["scale"] for m in units])
    def fit_predict(X, y, seeds_out):
        tr = ~np.isin(SEED, seeds_out)
        te = np.isin(SEED, seeds_out)
        if X is None:
            cols = [np.ones_like(P[tr])]
            for sc in ("v07_5m", "g1a_26m", "g1b_100m"):
                cols.append((scales[tr] == sc).astype(float))
            cols.append(P[tr])
            Xtr = np.column_stack(cols)
            cols_te = [np.ones_like(P[te])]
            for sc in ("v07_5m", "g1a_26m", "g1b_100m"):
                cols_te.append((scales[te] == sc).astype(float))
            cols_te.append(P[te])
            Xte = np.column_stack(cols_te)
        else:
            Xtr, Xte = X[tr], X[te]
        beta, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        return Xte @ beta, te
    for name, X in models.items():
        preds = np.zeros_like(Y)
        mask = np.zeros_like(Y, dtype=bool)
        for seed in sorted(set(SEED.tolist())):
            pred, te = fit_predict(X, Y, [seed])
            preds[te] = pred
            mask[te] = True
        resid = Y[mask] - preds[mask]
        ss = 1 - float((resid ** 2).sum()) / float(
            ((Y[mask] - Y[mask].mean()) ** 2).sum())
        try:
            from scipy.stats import spearmanr
            rc = float(spearmanr(preds[mask], Y[mask]).statistic)
        except Exception:
            rc = float("nan")
        mod_rows.append({"model": name, "r2_loso": ss,
                         "mae_loso": float(np.abs(resid).mean()),
                         "spearman_loso": rc, "n": int(mask.sum())})
    with (OUT / "exploratory_models.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mod_rows[0]))
        w.writeheader()
        w.writerows(mod_rows)

    # figures
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"v07_5m": "tab:blue", "g1a_26m": "tab:orange",
              "g1b_100m": "tab:red"}
    for qty, fname in (("R_curv", "fig1_Rcurv32_vs_progress.png"),
                       ("lambda_SO", "fig2_lambdaSO32_vs_progress.png")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for scale in ("v07_5m", "g1a_26m", "g1b_100m"):
            ms = [m for m in units if m["scale"] == scale]
            seeds = sorted({m["seed"] for m in ms})
            allp, allv = [], []
            for s in seeds:
                pts = sorted([(m["progress"],
                               (m["G"]["32"]["R_curv"] if qty == "R_curv"
                                else m["G"]["32"]["lambda_SO"])) for m in ms
                              if m["seed"] == s])
                ax.plot([p for p, _ in pts], [v for _, v in pts],
                        color=colors[scale], alpha=0.25, lw=1)
                allp.extend(p for p, _ in pts)
                allv.extend(v for _, v in pts)
            # median line over shared progress grid
            grid = sorted(set(round(p, 3) for p in allp))
            med = []
            for g in grid:
                vs = [v for p, v in zip(allp, allv) if abs(p - g) < 1e-6]
                med.append(float(np.median(vs)) if vs else np.nan)
            ax.plot(grid, med, color=colors[scale], lw=2, label=scale)
        ax.set_xlabel("training progress p = step/128")
        ax.set_ylabel("R_curv_32" if qty == "R_curv" else "lambda_SO_32")
        if qty == "R_curv":
            ax.axhline(10, ls="--", c="gray", lw=0.8)
            ax.axhline(3, ls=":", c="gray", lw=0.8)
        ax.axhline(0, ls="-", c="k", lw=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figdir / fname, dpi=130)
        plt.close(fig)
    # heatmap tiles: scale x progress grid, log10 R_curv_32
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    grid = sorted({round(m["progress"], 3) for m in units})
    for i, scale in enumerate(("v07_5m", "g1a_26m", "g1b_100m")):
        for m in units:
            if m["scale"] != scale:
                continue
            v = math.log10(max(m["G"]["32"]["R_curv"], 1e-30))
            ax.scatter(m["progress"], i, c=v, cmap="viridis", vmin=-1, vmax=2,
                       s=120, marker="s", edgecolors="k", linewidths=0.4)
    ax.set_yticks([0, 1, 2], ["5M", "26M", "100M"])
    ax.set_xlabel("progress p")
    ax.set_title("log10 R_curv_32 (tile per state)")
    fig.tight_layout()
    fig.savefig(figdir / "fig3_scale_progress_heatmap.png", dpi=130)
    plt.close(fig)

    # summary + verdict
    summary = {"study": "F4A_SCALE_PHASE_SEED_MAP",
               "n_units": len(units),
               "exploratory_model_only": True}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"units": len(units),
                      "models": {r["model"]: round(r["r2_loso"], 3)
                                 for r in mod_rows}}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("unit", "postprocess"))
    ap.add_argument("--scale", default=None,
                    choices=("v07_5m", "g1a_26m", "g1b_100m"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--step", type=int)
    ap.add_argument("--bank-dir", default=None,
                    help="override bank root for this scale (F4A.1 new states)")
    args = ap.parse_args()
    if args.cmd == "postprocess":
        return cmd_postprocess()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") is not None, \
        "set CUDA_VISIBLE_DEVICES"
    torch.set_num_threads(min(32, torch.get_num_threads()))
    torch.manual_seed(0)
    assert args.scale and args.seed is not None and args.step is not None
    return cmd_unit(args.scale, args.seed, args.step, args.bank_dir)


if __name__ == "__main__":
    sys.exit(main())
