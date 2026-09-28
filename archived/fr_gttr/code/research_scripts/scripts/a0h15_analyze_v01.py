"""A0H-1.5 diagnostics analysis: aggregate diag_*.json into branch-decision inputs.

Frozen contract: docs/FR_A0H15_PROTOCOL.md §6. Read-only wrt frozen artifacts.
Aggregates per dev unit:
  CHK-TELE closure  (max |term| per horizon/candidate)
  CHK-LIN         (repeatability spread, linearity residuals, K(0) norm)
  CHK-SA vs CHK-IVJP paired (self-adjoint defect vs independent-VJP defect;
      if SA ~ IVJP >> solver noise -> bf16-proxy asymmetry is the root cause;
      if SA >> IVJP -> true non-self-adjointness of the frozen HVP map)
  wrapper_vs_frozen_gap (wrapper consistency)
Emits branch inputs: per §6 A requires tele+lin+shared gates; B requires
fp32-hvp different-config evaluation. Never re-applies the original elementwise
gate as sole standard (correction §3).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIAG_DIR = ROOT / "outputs" / "a0h15_vjp_shared_v01" / "diagnostics"


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def summarize_unit(doc: dict) -> dict:
    out = {"state": doc.get("state"), "stream": doc.get("stream")}
    tele = doc.get("tele", {})
    tele_summary = {}
    for h, per_j in tele.items():
        worst = None
        for j, rec in per_j.items():
            if worst is None or rec["max_abs_term"] > worst["max_abs_term"]:
                worst = {"j": j, **rec}
        tele_summary[h] = worst
    out["tele_worst_per_h"] = tele_summary

    diag = doc.get("diag", {})
    lin = [r for r in diag.get("lin", []) if "rep_mean_abs_gap" in r]
    lin_summary = {}
    for r in lin:
        lin_summary[str(r["horizon"])] = {
            "rep_mean_abs_gap": r["rep_mean_abs_gap"],
            "rep_abs_gaps": r["rep_abs_gaps"],
            "k0_rel_to_kx": r["k0_rel_to_kx"],
            "lin_neg_rel": r["lin_neg_rel"],
            "lin_alpha_rel": r["lin_alpha_rel"],
            "lin_add_rel": r["lin_add_rel"],
        }
    out["lin"] = lin_summary

    sa = diag.get("sa", [])
    ivjp = diag.get("ivjp", [])
    paired = []
    sa_by_key = {(r["horizon"], r["anchor_j"]): r for r in sa}
    for r in ivjp:
        key = (r["horizon"], r["anchor_j"])
        s = sa_by_key.get(key)
        paired.append({
            "h": r["horizon"], "j": r["anchor_j"],
            "sa_rel": s["rel"] if s else None,
            "ivjp_rel": r.get("rel"),
            "ivjp_status": r.get("status"),
            "lhs": r.get("lhs"), "rhs": r.get("rhs"),
            "ivjp_map_norm": r.get("ivjp_map_norm"),
        })
    out["sa_ivjp_paired"] = paired
    out["ivjp_precheck"] = diag.get("ivjp_precheck")
    out["wrapper_gaps"] = [r.get("wrapper_vs_frozen_gap")
                           for r in diag.get("lin", [])
                           if "wrapper_vs_frozen_gap" in r]
    return out


def main(paths=None) -> int:
    diag_files = paths or sorted(DIAG_DIR.glob("diag_*_s0.json"))
    docs = [_load(p) for p in diag_files]
    summary = [summarize_unit(d) for d in docs]
    out_path = DIAG_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_path}")
    for s in summary:
        print(f"== {s['state']} ==")
        print("  tele_worst:", {h: (w['max_abs_term'], w['pass_tele']) if w else None
                                for h, w in s['tele_worst_per_h'].items()})
        print("  lin:", {h: {"rep": v["rep_mean_abs_gap"],
                            "k0_rel": v["k0_rel_to_kx"],
                            "add_rel": v["lin_add_rel"]}
                         for h, v in s["lin"].items()})
        print("  wrapper_gaps:", s["wrapper_gaps"])
        print("  ivjp precheck:", s["ivjp_precheck"])
        for p in s["sa_ivjp_paired"][:8]:
            print("   ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
