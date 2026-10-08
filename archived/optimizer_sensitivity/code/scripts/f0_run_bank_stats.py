#!/usr/bin/env python3
"""F0 Phase 3: optimizer second-moment / first-moment floor statistics over the
frozen v07 (4.98M), G1a (26.68M), G1b (100.78M) banks — READ-ONLY.

Outputs (under outputs/f0_existing_bank_audit/):
  per_parameter_group.csv     — per (scale, seed, step, group) v/m floor stats
  embedding_frequency_bins.csv— per (scale, seed, step, freq-bin) v/denominator/gain stats
  token_frequency_table.csv   — per token id: count + bin
  summary.json                — aggregates + preclip context + provenance
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

import opsens.f0_bank_audit_v01 as f0

CONFIG = json.loads((ROOT / "configs/f0_bank_audit_v01.json").read_text())
OUT = ROOT / "outputs/f0_existing_bank_audit"
OUT.mkdir(parents=True, exist_ok=True)


def bank_states(scale: str):
    spec = CONFIG["banks"][scale]
    for seed in spec["seeds"]:
        for step in spec["steps"]:
            p = Path(spec["root"]) / spec["state_glob"].format(seed=seed, step=step)
            if not p.exists():
                raise FileNotFoundError(p)
            yield seed, step, p


def dynamics_preclip(scale: str, seed: int) -> dict:
    spec = CONFIG["banks"][scale]
    if "dynamics" not in spec:
        return {}
    p = Path(spec["root"]) / spec["dynamics"].format(seed=seed)
    d = json.loads(p.read_text())["dynamics"]
    return {int(r["optimizer_step"]): r for r in d}


def main() -> None:
    t0 = time.time()
    freq = f0.token_frequency_bins()
    rows_tok = freq["counts"]
    print(f"[freq] vocab={freq['vocab']} rows={freq['n_rows']} observed={freq['observed']} "
          f"head/mid/tail/never={len(freq['head'])}/{len(freq['mid'])}/{len(freq['tail'])}/{len(freq['never'])} "
          f"bos/eos/pad={freq['bos_id']}/{freq['eos_id']}/{freq['pad_id']}", flush=True)
    with open(OUT / "token_frequency_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["token_id", "count", "bin"])
        bins = {}
        for b in ("head", "mid", "tail", "never"):
            for i in freq[b]:
                bins[i] = b
        for i in range(freq["vocab"]):
            w.writerow([i, int(rows_tok[i]), bins.get(i, "never")])

    embed_name = "model.embed_tokens.weight"
    group_rows = []
    emb_rows = []
    summary_states = []

    for scale in ("v07_5m", "g1a_26m", "g1b_100m"):
        dyn = {s: dynamics_preclip(scale, s) for s in CONFIG["banks"][scale]["seeds"]}
        n_states = 0
        for seed, step, path in bank_states(scale):
            state = f0.load_bank_state(path)
            hp = f0.group_hyperparams(state)
            eps = hp["eps"]
            per_group_v = {}
            per_group_m = {}
            embed_slot = None
            step_t = None
            for name, idx, slot in f0.named_optimizer_entries(state):
                g = f0.classify_group(name)
                per_group_v.setdefault(g, []).append(slot["exp_avg_sq"])
                per_group_m.setdefault(g, []).append(slot["exp_avg"])
                if step_t is None:
                    step_t = float(slot["step"])
                if name == embed_name:
                    embed_slot = slot
            for g in sorted(per_group_v):
                v = torch.cat(per_group_v[g]).flatten()
                m = torch.cat(per_group_m[g]).flatten()
                vs = f0.tensor_floor_stats(v, eps)
                ms = f0.signed_stats(m, eps)
                gain = f0.diagonal_gain(v, step_t, hp["lr"], hp["beta1"], hp["beta2"], eps)
                gs = f0.signed_stats(gain, eps)
                row = {
                    "scale": scale, "seed": seed, "step": step, "group": g,
                    "lr": hp["lr"], "eps": eps,
                    "diag_gain_max": gs["abs_max"], "diag_gain_median": gs["abs_median"],
                    "diag_gain_p99_9": gs["abs_p99_9"], "diag_gain_p0_1": gs["abs_p0_1"],
                    **{f"v_{k}": v_ for k, v_ in vs.items()},
                    **{f"m_{k}": v_ for k, v_ in ms.items()},
                }
                group_rows.append(row)
                del v, m, gain

            v = embed_slot["exp_avg_sq"].flatten()
            denom = f0.denominator_anchor(embed_slot["exp_avg_sq"], step_t, hp["beta2"], eps)
            gain = f0.diagonal_gain(embed_slot["exp_avg_sq"], step_t, hp["lr"], hp["beta1"], hp["beta2"], eps)
            counts = rows_tok
            for b in ("head", "mid", "tail", "never"):
                ids = torch.tensor(freq[b], dtype=torch.long)
                vb = v[ids]
                db = denom.flatten()[ids]
                gb = gain.flatten()[ids]
                cb = counts[ids].to(torch.float64)
                vs = f0.tensor_floor_stats(vb, eps)
                ds = f0.tensor_floor_stats(db, eps)
                gs = f0.signed_stats(gb, eps)
                emb_rows.append({
                    "scale": scale, "seed": seed, "step": step, "bin": b,
                    "n_tokens": int(ids.numel()),
                    "freq_mean": float(cb.mean()) if ids.numel() else float("nan"),
                    "freq_median": float(cb.median()) if ids.numel() else float("nan"),
                    "freq_min": float(cb.min()) if ids.numel() else float("nan"),
                    "denom_median": ds["median"], "denom_min": ds["min"], "denom_p0_1": ds["p0_1"],
                    "adam_scale_median": gs["abs_median"], "adam_scale_max": gs["abs_max"],
                    "adam_scale_p99_9": gs["abs_p99_9"],
                    **{f"v_{k}": v_ for k, v_ in vs.items()},
                })
            preclip = dyn.get(seed, {}).get(step, {}).get("preclip_grad_norm")
            summary_states.append({
                "scale": scale, "seed": seed, "step": step, "lr": hp["lr"], "eps": eps,
                "beta1": hp["beta1"], "beta2": hp["beta2"], "weight_decay": hp["weight_decay"],
                "preclip_grad_norm": preclip,
                "optimizer_step_t": step_t,
            })
            n_states += 1
            print(f"[{scale}] seed={seed} step={step} done ({time.time()-t0:.0f}s)", flush=True)
            del state

    with open(OUT / "per_parameter_group.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(group_rows[0].keys()))
        w.writeheader()
        w.writerows(group_rows)
    with open(OUT / "embedding_frequency_bins.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(emb_rows[0].keys()))
        w.writeheader()
        w.writerows(emb_rows)

    def worst(rows, key, ident=("scale", "seed", "step", "group")):
        cand = [r for r in rows if r.get(key) is not None]
        if not cand:
            return None
        r = max(cand, key=lambda r: r[key])
        return {k: r.get(k) for k in ident + (key,)}

    summary = {
        "audit_id": CONFIG["audit_id"],
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frequency_bins": {
            "vocab": freq["vocab"], "n_rows": freq["n_rows"], "observed": freq["observed"],
            "head": len(freq["head"]), "mid": len(freq["mid"]), "tail": len(freq["tail"]),
            "never": len(freq["never"]),
            "bos_id": freq["bos_id"], "eos_id": freq["eos_id"], "pad_id": freq["pad_id"],
            "never_ids_preview": freq["never"][:20],
            "tail_min_count": int(rows_tok[min(freq["tail"])].item()) if freq["tail"] else None,
            "head_max_count": int(rows_tok[max(freq["head"])].item()),
        },
        "states": summary_states,
        "worst_v_exact_zero_fraction": worst(group_rows, "v_exact_zero_fraction"),
        "worst_diag_gain_max": worst(group_rows, "diag_gain_max"),
        "worst_sqrt_v_lt_eps_fraction": worst(group_rows, "v_sqrt_v_lt_eps_fraction"),
        "worst_embed_bin_adam_scale_max": worst(emb_rows, "adam_scale_max", ("scale", "seed", "step", "bin")),
        "n_group_rows": len(group_rows), "n_embedding_rows": len(emb_rows),
        "provenance": "legacy banks READ-ONLY; vendored frozen tangent; rules frozen in configs/f0_bank_audit_v01.json",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] rows: groups={len(group_rows)} embed={len(emb_rows)} wall={time.time()-t0:.0f}s", flush=True)




if __name__ == "__main__":
    torch.manual_seed(0)
    main()
