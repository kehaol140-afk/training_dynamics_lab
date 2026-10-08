#!/usr/bin/env python3
"""Rebuild summary.json from the already-written Phase-3 CSVs (no tensor reload)."""
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs/f0_existing_bank_audit"
CONFIG = json.loads((ROOT / "configs/f0_bank_audit_v01.json").read_text())


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def to_f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    groups = read(OUT / "per_parameter_group.csv")
    embs = read(OUT / "embedding_frequency_bins.csv")
    toks = read(OUT / "token_frequency_table.csv")
    never = [int(t["token_id"]) for t in toks if t["bin"] == "never"]
    tail = [int(t["token_id"]) for t in toks if t["bin"] == "tail"]
    head = [int(t["token_id"]) for t in toks if t["bin"] == "head"]
    counts = {int(t["token_id"]): int(t["count"]) for t in toks}

    def worst(rows, key, ident):
        cand = [(to_f(r[key]), r) for r in rows if to_f(r[key]) is not None]
        if not cand:
            return None
        v, r = max(cand, key=lambda x: x[0])
        return {k: r.get(k) for k in ident + (key,)} | {key: v}

    states = []
    seen = set()
    for r in groups:
        k = (r["scale"], int(r["seed"]), int(r["step"]))
        if k not in seen:
            seen.add(k)
            states.append({"scale": r["scale"], "seed": int(r["seed"]), "step": int(r["step"]),
                           "lr": to_f(r["lr"]), "eps": to_f(r["eps"]), "optimizer_step_t": int(r["step"])})
    for scale in ("g1a_26m", "g1b_100m"):
        for st in states:
            if st["scale"] != scale:
                continue
            p = Path(CONFIG["banks"][scale]["root"]) / CONFIG["banks"][scale]["dynamics"].format(seed=st["seed"])
            d = json.loads(p.read_text())["dynamics"]
            st["preclip_grad_norm"] = next((r["preclip_grad_norm"] for r in d if int(r["optimizer_step"]) == st["step"]), None)

    summary = {
        "audit_id": CONFIG["audit_id"],
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frequency_bins": {
            "vocab": len(toks), "n_rows": None, "observed": len([t for t in toks if int(t["count"]) > 0]),
            "head": len(head), "mid": len(toks) - len(head) - len(tail) - len(never),
            "tail": len(tail), "never": len(never),
            "bos_id": 1, "eos_id": 2, "pad_id": 0,
            "never_ids_preview": never[:20],
            "tail_min_count": min((counts[i] for i in tail), default=None),
            "head_max_count": max((counts[i] for i in head), default=None),
        },
        "states": states,
        "worst_v_exact_zero_fraction": worst(groups, "v_exact_zero_fraction", ("scale", "seed", "step", "group")),
        "worst_diag_gain_max": worst(groups, "diag_gain_max", ("scale", "seed", "step", "group")),
        "worst_sqrt_v_lt_eps_fraction": worst(groups, "v_sqrt_v_lt_eps_fraction", ("scale", "seed", "step", "group")),
        "worst_embed_bin_adam_scale_max": worst(embs, "adam_scale_max", ("scale", "seed", "step", "bin")),
        "n_group_rows": len(groups), "n_embedding_rows": len(embs),
        "provenance": "legacy banks READ-ONLY; vendored frozen tangent; rules frozen in configs/f0_bank_audit_v01.json; summary rebuilt from CSVs by f0_rebuild_summary.py",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("summary.json rebuilt:", len(states), "states,", len(groups), "group rows,", len(embs), "embedding rows")


if __name__ == "__main__":
    main()
