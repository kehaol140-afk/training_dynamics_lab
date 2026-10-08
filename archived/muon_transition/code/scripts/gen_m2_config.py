"""Emit the frozen M2 preregistration: 24 new base cells (p=0.25/0.75) + 4 justified p=0.5
extensions per M2_PROTOCOL_ADDENDUM §1.5. M1 p=0.5 cells are referenced, not re-run."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from muon_transition import config

LR_MULT_EXT = 4.0


def cells(arm, p, mults):
    ref = config.ADAMW_REF_LR if arm == "adamw" else config.MUON_REF_LR
    out = []
    for mult in mults:
        for wd in config.WD_GRID:
            out.append({
                "run_id": f"m2_{arm}_lr{mult}_wd{wd}_p{p}_s{config.BASE_SEED_M1_M2}",
                "stage": "M2", "arm": arm, "p": p, "seed": config.BASE_SEED_M1_M2,
                "lr_multiplier": mult, "lr": ref * mult, "reference_lr": ref,
                "weight_decay": wd,
                "extension": mult == LR_MULT_EXT,
                "justification": (
                    "M2 addendum §1.5: M1 best at mult=2.0 for this arm at p=0.5"
                    if mult == LR_MULT_EXT else None
                ),
            })
    return out


def main():
    new_cells = []
    for arm in ("adamw", "muon"):
        for p in (0.25, 0.75):
            new_cells.extend(cells(arm, p, config.LR_MULTIPLIERS))
    ext_p05 = cells("adamw", 0.5, [LR_MULT_EXT]) + cells("muon", 0.5, [LR_MULT_EXT])

    m1_runs = [c for c in json.loads((ROOT / "configs" / "m1_baseline_prereg.json").read_text())["m1_runs"]]

    prereg = {
        "format": "mte_m2_prereg_v1",
        "date_frozen": "2026-09-26",
        "protocol": "docs/MUON_PROTOCOL_v1.1.md + v1.2 + docs/M2_PROTOCOL_ADDENDUM.md",
        "inherits": "model/tokenizer/budget/eval/hooks from configs/m1_baseline_prereg.json",
        "m1_cells_p05": [{"run_id": c["run_id"], "result": f"outputs/m1/{c['run_id']}/result.json"} for c in m1_runs],
        "boundary_rule": {
            "base_mult": list(config.LR_MULTIPLIERS), "extension_mult": LR_MULT_EXT,
            "rule": "one extension to 4 iff base-grid best (dev V_Q, tie-break lower lr then lower wd) sits at mult=2; if best remains at 4 -> LR_RANGE_UNRESOLVED; no further extension ever",
        },
        "selection_rule": "per (arm,p): min dev V_Q, tie-break lower lr then lower wd; p*_arm = argmin_p best V_Q, tie-break p=0.5 then lower p",
        "new_runs": new_cells,
        "justified_extension_runs": ext_p05,
        "counts": {"new_base": len(new_cells), "extensions": len(ext_p05),
                   "m1_reused": len(m1_runs), "m2_total_grid": len(new_cells) + len(ext_p05) + len(m1_runs)},
    }
    out = ROOT / "configs" / "m2_prereg.json"
    out.write_text(json.dumps(prereg, indent=2))
    print(f"wrote {out}")
    print(json.dumps(prereg["counts"], indent=1))


if __name__ == "__main__":
    main()
