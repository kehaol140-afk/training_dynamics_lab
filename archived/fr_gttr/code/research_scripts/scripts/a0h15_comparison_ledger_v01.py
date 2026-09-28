"""A0H-1.5 comparison ledger generator (round 2).

Builds comparison_ledger.csv/json from EXISTING artifacts only — every numeric
comparison pair gets an explicit row: methods on both ends (with precision/backend),
unit, H, action, score kind, values, abs diff, pre-registered tolerance, per-item
verdict. No new GPU work is performed here.

Sources:
  S1 frozen dev_check units      -> frozen_BF16_forward (hhat_fwd), old analytic adjoint (hhat_adj)
  S2 diag_2031_52_s0.json (BF16) -> rerun_BF16_forward (hhat_fwd_C), BF16_IVJP_shared (hhat_adj)
  S3 diag_*_fp32hvp_s0.json      -> FP32_HVP_forward, FP32_HVP_adjoint (shared)
  S4 accept/accept_*.json (dev-validation on demoted batch) -> rows (fwd vs shared adj)
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/a0h15_vjp_shared_v01/ledger"
OUT.mkdir(parents=True, exist_ok=True)

ATOL, RTOL = 1e-6, 1e-4          # frozen original gate
ATOL32, RTOL32 = 1e-5, 1e-4      # pre-registered B1 tolerance

rows = []

def add(*, source_file, run_id, state, stream, H, action_id,
        reference_method, candidate_method, ref_prec, cand_prec,
        score_kind, reference_value, candidate_value, atol, rtol, note=""):
    thr = atol + rtol * abs(reference_value)
    diff = abs(candidate_value - reference_value)
    rows.append({
        "source_file": source_file, "run_id": run_id, "unit": state, "stream": stream,
        "H": int(H), "action_id": action_id,
        "reference_method": reference_method, "candidate_method": candidate_method,
        "reference_precision_backend": ref_prec, "candidate_precision_backend": cand_prec,
        "score_kind": score_kind,
        "reference_value": reference_value, "candidate_value": candidate_value,
        "abs_diff": diff, "atol": atol, "rtol": rtol, "threshold": thr,
        "verdict": "PASS" if diff <= thr else "FAIL", "note": note,
    })

BF16PB = "bf16 autocast / math-SDPA HVP proxy"
FP32PB = "fp32 HVP (nullcontext) / bf16 fwd-bwd otherwise"

# ---- S1: frozen unit JSONs (frozen_BF16_forward vs old analytic adjoint; frozen gate) ----
for unit, label in (("2031_52", "dev:2031/52"), ("2034_64", "dev:2034/64"), ("2036_76", "dev:2036/76")):
    p = ROOT / f"outputs/a0h_adjoint_cost_v01/dev_check/unit_{unit}_s0.json"
    d = json.loads(p.read_text())
    for hk in ("1", "4", "8"):
        m = d["methods"][f"C|{hk}"]
        for r in m["per_candidate"]:
            j = r["j"] if "j" in r else r.get("candidate")
            add(source_file=str(p.relative_to(ROOT)), run_id="a0h1_frozen_dev_check",
                state=label, stream=0, H=hk, action_id=int(j),
                reference_method="frozen_BF16_forward", candidate_method="old_analytic_adjoint_BF16",
                ref_prec=BF16PB, cand_prec=BF16PB, score_kind="hhat",
                reference_value=float(r["hhat_fwd"]), candidate_value=float(r["hhat_adj"]),
                atol=ATOL, rtol=RTOL, note="original frozen gate, unchanged")

# ---- S2: a0h15 BF16 pass on D1 (rerun fwd vs shared adjoint; also vs frozen fwd) ----
p = ROOT / "outputs/a0h15_vjp_shared_v01/diagnostics/diag_2031_52_s0.json"
d = json.loads(p.read_text())
fz = json.loads((ROOT / "outputs/a0h_adjoint_cost_v01/dev_check/unit_2031_52_s0.json").read_text())
for hk in ("1", "4", "8"):
    fzrows = {int(r["j"] if "j" in r else r.get("candidate")): r for r in fz["methods"][f"C|{hk}"]["per_candidate"]}
    for js in sorted(d["hhat_fwd_C"][hk], key=int):
        j = int(js)
        # rerun forward reproduction vs frozen forward
        add(source_file=str(p.relative_to(ROOT)), run_id="a0h15_rerun_bf16_d1",
            state="dev:2031/52", stream=0, H=hk, action_id=j,
            reference_method="frozen_BF16_forward", candidate_method="rerun_BF16_forward",
            ref_prec=BF16PB, cand_prec=BF16PB, score_kind="hhat",
            reference_value=float(fzrows[j]["hhat_fwd"]), candidate_value=float(d["hhat_fwd_C"][hk][js]),
            atol=ATOL, rtol=RTOL, note="forward reproduction across separate runs")
        # rerun forward vs shared adjoint (same pass, original gate)
        add(source_file=str(p.relative_to(ROOT)), run_id="a0h15_shared_bf16_d1",
            state="dev:2031/52", stream=0, H=hk, action_id=j,
            reference_method="rerun_BF16_forward", candidate_method="BF16_IVJP_shared",
            ref_prec=BF16PB, cand_prec=BF16PB, score_kind="hhat",
            reference_value=float(d["hhat_fwd_C"][hk][js]),
            candidate_value=float(d["vjp_shared"][hk]["hhat_adj"][js]),
            atol=ATOL, rtol=RTOL, note="shared adjoint vs same-pass forward, original gate")
        # shared adjoint vs frozen forward (the pair the crashed probe claimed bitwise)
        add(source_file=str(p.relative_to(ROOT)), run_id="a0h15_shared_vs_frozenfwd_d1",
            state="dev:2031/52", stream=0, H=hk, action_id=j,
            reference_method="frozen_BF16_forward", candidate_method="BF16_IVJP_shared",
            ref_prec=BF16PB, cand_prec=BF16PB, score_kind="hhat",
            reference_value=float(fzrows[j]["hhat_fwd"]),
            candidate_value=float(d["vjp_shared"][hk]["hhat_adj"][js]),
            atol=ATOL, rtol=RTOL, note="cross-run pair; round-1 'bitwise' claim corrected (see ledger README)")

# ---- S3: FP32-HVP passes (B1 same-config + B2 vs frozen) ----
for unit, label in (("2031_52", "dev:2031/52"), ("2034_64", "dev:2034/64"), ("2036_76", "dev:2036/76")):
    p = ROOT / f"outputs/a0h15_vjp_shared_v01/diagnostics/diag_{unit}_fp32hvp_s0.json"
    d = json.loads(p.read_text())
    fz = json.loads((ROOT / f"outputs/a0h_adjoint_cost_v01/dev_check/unit_{unit}_s0.json").read_text())
    for hk in ("1", "4", "8"):
        fzrows = {int(r["j"] if "j" in r else r.get("candidate")): r for r in fz["methods"][f"C|{hk}"]["per_candidate"]}
        for js in sorted(d["hhat_fwd_C"][hk], key=int):
            j = int(js)
            add(source_file=str(p.relative_to(ROOT)), run_id="a0h15_fp32hvp_b1",
                state=label, stream=0, H=hk, action_id=j,
                reference_method="FP32_HVP_forward", candidate_method="FP32_HVP_adjoint_shared",
                ref_prec=FP32PB, cand_prec=FP32PB, score_kind="hhat",
                reference_value=float(d["hhat_fwd_C"][hk][js]),
                candidate_value=float(d["vjp_shared"][hk]["hhat_adj"][js]),
                atol=ATOL32, rtol=RTOL32, note="Branch B1 same-config gate")
            add(source_file=str(p.relative_to(ROOT)), run_id="a0h15_fp32hvp_b2_elementwise",
                state=label, stream=0, H=hk, action_id=j,
                reference_method="frozen_BF16_forward", candidate_method="FP32_HVP_adjoint_shared",
                ref_prec=BF16PB, cand_prec=FP32PB, score_kind="hhat",
                reference_value=float(fzrows[j]["hhat_fwd"]),
                candidate_value=float(d["vjp_shared"][hk]["hhat_adj"][js]),
                atol=ATOL, rtol=RTOL, note="B2 elementwise reporting (different config; NOT the B2 decision criterion)")

# ---- S4: demoted-batch dev-validation rows ----
for p in sorted((ROOT / "outputs/a0h15_vjp_shared_v01/accept").glob("accept_*.json")):
    d = json.loads(p.read_text())
    if d.get("disposition") != "DEV_VALIDATION_ONLY":
        continue
    for mk in ("C|1", "C|4", "C|8"):
        m = d["methods"][mk]
        for r in m["rows"]:
            add(source_file=str(p.relative_to(ROOT)), run_id="a0h15_devval_demotedA",
                state=d["state"], stream=0, H=int(mk.split("|")[1]), action_id=int(r["j"]),
                reference_method="rerun_BF16_forward", candidate_method="BF16_IVJP_shared",
                ref_prec=BF16PB, cand_prec=BF16PB, score_kind="hhat",
                reference_value=float(r["hhat_fwd"]), candidate_value=float(r["hhat_adj"]),
                atol=ATOL, rtol=RTOL, note="dev-validation on demoted batch; no acceptance status")

with open(OUT / "comparison_ledger.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
(OUT / "comparison_ledger.json").write_text(json.dumps(rows, indent=1))

# summary
from collections import Counter, defaultdict
c = Counter((r["run_id"], r["verdict"]) for r in rows)
print(f"{len(rows)} ledger rows -> {OUT}")
for (rid, v), n in sorted(c.items()):
    print(f"  {rid}: {v} {n}")
