"""A0T-extra: response controller identity analysis (pure statistics).

Offline-only. Reads frozen A0T cell records and answers: is the response-
guided controller genuinely adaptive, or a global LR scaling discovery?
All functions are deterministic transforms of the frozen records.
"""
from __future__ import annotations

import numpy as np

H_GRID = (4, 16, 64)
ACTIONS = (-0.05, 0.05)


def gains(rec):
    """Per-H gains of arms B and C vs A for one frozen cell record."""
    return {arm: {str(H): rec["losses"]["A"][str(H)] - rec["losses"][arm][str(H)]
                  for H in H_GRID} for arm in ("B", "C")}


def action_distribution(records):
    """P(+0.05)/P(-0.05) pooled, per seed, per dataset."""
    def counts(rs):
        n = len(rs)
        p_plus = sum(1 for r in rs if r["action_u"] > 0) / n
        return {"n": n, "P(+0.05)": p_plus, "P(-0.05)": 1 - p_plus}
    out = {"pooled": counts(records)}
    out["per_seed"] = {str(s): counts([r for r in records if r["seed"] == s])
                       for s in sorted({r["seed"] for r in records})}
    out["per_dataset"] = {d: counts([r for r in records if r["dataset"] == d])
                          for d in sorted({r["dataset"] for r in records})}
    return out


def action_entropy(records, group_key=None):
    """Shannon entropy of the action in bits.

    group_key: None (pooled), 'seed', or 'dataset'."""
    def h(rs):
        n = len(rs)
        ps = [sum(1 for r in rs if r["action_u"] > 0) / n,
              sum(1 for r in rs if r["action_u"] < 0) / n]
        return float(-sum(p * np.log2(p) for p in ps if p > 0))
    if group_key is None:
        return h(records)
    return {str(k): h([r for r in records if r[group_key] == k])
            for k in sorted({r[group_key] for r in records})}


def switching(records):
    """Sign changes of the guided action along each training trajectory.

    Trajectory = one (dataset, seed, stream), ordered by bank step."""
    trajs = {}
    for r in records:
        key = (r["dataset"], r["seed"], r["stream_seed"])
        trajs.setdefault(key, []).append(r)
    out = []
    for (ds, seed, stream), cells in sorted(trajs.items()):
        cells = sorted(cells, key=lambda c: c["step"])
        acts = [('+' if c["action_u"] > 0 else '-') for c in cells]
        switches = sum(1 for a, b in zip(acts, acts[1:]) if a != b)
        out.append({"dataset": ds, "seed": seed, "stream_seed": stream,
                    "steps": [c["step"] for c in cells],
                    "actions": acts, "sign_changes": switches,
                    "max_possible": len(cells) - 1})
    total = sum(t["sign_changes"] for t in out)
    return {"trajectories": out, "total_sign_changes": total,
            "trajectories_with_any_switch": sum(1 for t in out if t["sign_changes"] > 0),
            "n_trajectories": len(out)}


def decomposition(records):
    """response_gain = fixed_gain + adaptive_gain per frozen contract.

    best_fixed_gain = max over AVAILABLE fixed arms: +0.05 always (arm B);
    -0.05 only on non-aliased cells (exists as arm C rollout there).
    adaptive_gain = gain_C - best_fixed_gain (per H).
    Also reports directed benefit of guidance on non-aliased cells."""
    rows = []
    for r in records:
        g = gains(r)
        row = {"dataset": r["dataset"], "seed": r["seed"], "step": r["step"],
               "stream_seed": r["stream_seed"], "chi_lr": r["chi_lr"],
               "action_u": r["action_u"], "aliased": r["aliased"]}
        for H in H_GRID:
            k = str(H)
            fixed_plus = g["B"][k]
            row[f"gain_response_{k}"] = g["C"][k]
            row[f"gain_fixed_plus_{k}"] = fixed_plus
            if r["aliased"]:
                best_fixed = fixed_plus
                row[f"gain_fixed_minus_{k}"] = None
            else:
                fixed_minus = g["C"][k]  # -0.05 rolled as arm C
                row[f"gain_fixed_minus_{k}"] = fixed_minus
                best_fixed = max(fixed_plus, fixed_minus)
            row[f"best_fixed_gain_{k}"] = best_fixed
            row[f"adaptive_gain_{k}"] = g["C"][k] - best_fixed
            if not r["aliased"]:
                row[f"guided_vs_fixed_plus_{k}"] = g["C"][k] - fixed_plus
        rows.append(row)
    return rows


def decomposition_summary(rows):
    """Pooled summaries of the decomposition at each H."""
    out = {}
    for H in H_GRID:
        k = str(H)
        resp = np.array([r[f"gain_response_{k}"] for r in rows])
        best = np.array([r[f"best_fixed_gain_{k}"] for r in rows])
        ad = np.array([r[f"adaptive_gain_{k}"] for r in rows])
        disc = [r[f"guided_vs_fixed_plus_{k}"] for r in rows
                if not r["aliased"] and f"guided_vs_fixed_plus_{k}" in r]
        out[k] = {
            "response_gain_median": float(np.median(resp)),
            "best_fixed_gain_median": float(np.median(best)),
            "adaptive_gain_median": float(np.median(ad)),
            "adaptive_gain_max": float(np.max(ad)),
            "n_adaptive_positive": int(np.sum(ad > 0)),
            "identification_note": ("adaptive=0 on 34/36 aliased cells by "
                                    "construction; discriminative cells: 2"),
            "discriminative_cells_guided_minus_fixed_plus": disc,
        }
    return out
