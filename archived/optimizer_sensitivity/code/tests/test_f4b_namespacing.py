"""Anti-overwrite tests (F4 consistency A4): secondary runs never touch
primary namespaces; delta/dir consistency; headline rebuildability."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "f4b_sau"

def test_delta_namespaces_exist_and_match():
    for d, delta in (("delta_0p05", 0.05), ("delta_0p02", 0.02)):
        s = json.load((OUT / d / "summary.json").open())
        assert abs(s["delta"] - delta) < 1e-12, (d, s["delta"])

def test_namespaced_summaries_identical_content():
    # primary namespace must equal the historical primary artifact numbers
    old = json.load((OUT / "summary.json").open())          # historical d=0.05
    new = json.load((OUT / "delta_0p05" / "summary.json").open())
    assert old["verdict"] == new["verdict"] == "NO_METHOD_GAIN"
    assert abs(old["per_method"]["SAU"]["mean"] - new["per_method"]["SAU"]["mean"]) < 1e-9
    old2 = json.load((OUT / "summary_delta0.02.json").open())
    new2 = json.load((OUT / "delta_0p02" / "summary.json").open())
    assert abs(old2["per_method"]["SAU"]["mean"] - new2["per_method"]["SAU"]["mean"]) < 1e-9

def test_secondary_did_not_overwrite_primary_tables():
    a05 = list(csv.DictReader((OUT / "delta_0p05" / "action_outcomes.csv").open()))
    a02 = list(csv.DictReader((OUT / "delta_0p02" / "action_outcomes.csv").open()))
    assert len(a05) == len(a02) == 42
    r = [x for x in a05 if x["scale"] == "g1b_100m" and x["seed"] == "2042" and x["step"] == "26"][0]
    assert abs(float(r["L_minus"]) - 4.755785) < 1e-4  # canary primary value

def test_headline_rebuildable_from_canonical():
    mc = {x["model"]: x for x in csv.DictReader((ROOT / "outputs/f4a5_modeling/model_comparison.csv").open())}
    v = float(mc["M0"]["mae"])
    assert abs(v - 0.4249) < 5e-4
    # report MAE claim reconstructs; RMSE/Spearman/R2 do NOT (stale) -- locked
    assert float(mc["M0"]["rmse"]) > 0.75 and float(mc["M0"]["spearman"]) > 0.81

def test_metric_names_not_mixed():
    rc = json.load((ROOT / "outputs/f4_consistency/f4a5_metric_recompute.json").open())
    assert set(rc["M0"]) >= {"POOLED_OOF", "MACRO_FOLD"}
    assert abs(rc["M0"]["POOLED_OOF"]["spearman"] - 0.8195) < 5e-4
    assert abs(rc["M0"]["MACRO_FOLD"]["spearman"] - 0.8987) < 5e-4

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"{len(fns)}/{len(fns)} PASS")
