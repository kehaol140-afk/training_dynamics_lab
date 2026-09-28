"""FR_A0H-0 离线审计库的单元测试（纯 CPU，不碰 GPU，不改冻结制品）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/src"))
sys.path.insert(0, str(ROOT / "research/scripts"))

from fr.a0h_response_cost_audit_v01 import (  # noqa: E402
    baseline_never,
    apply_dev_fixed,
    check_gain_regret_identity,
    descriptive_rates,
    margins,
    oracle_capture,
    paired_delta_regret_check,
    recompute_level3,
    select_dev_fixed_group,
    replay_score_availability,
)


def _synthetic_level2(split="formal", n_units=6):
    """构造确定性的合成 level2 行：j=0..4，actual 有固定模式。"""
    rows = []
    for i in range(n_units):
        seed, step, stream = 2038 + i % 2, 46 + (i % 3) * 18, i % 4
        for h in (1, 4, 8):
            for j in range(5):
                actual = {
                    0: 0.0,
                    1: -3e-5 - i * 1e-6,
                    2: 1e-5,
                    3: -2e-5,
                    4: 5e-5,
                }[j]
                rows.append({
                    "split": split, "seed": str(seed), "step": str(step),
                    "stream_id": str(stream), "j": str(j), "H": str(h),
                    "lambda": "0.05",
                    "actual_delta_eval": str(actual),
                    "pred_delta_alignment": str(actual * 0.5),
                    "pred_delta_B": str(actual if j else 0.0),
                    "pred_delta_C": str(actual * 1.1 if j else 0.0),
                })
    return rows


class TestRecomputeAndIdentity(unittest.TestCase):
    def test_recompute_level3_runs_and_identity(self):
        l2 = _synthetic_level2()
        l3 = recompute_level3(l2)
        n_units = 6
        self.assertEqual(len(l3), n_units * 3 * 4)
        idc = check_gain_regret_identity(l3)
        self.assertEqual(idc["status"], "PASS")
        d = paired_delta_regret_check(l3, "C", "B")
        self.assertEqual(d["status"], "PASS")

    def test_never_and_dev_fixed(self):
        l2 = _synthetic_level2()
        never = baseline_never(l2, "formal", 4)
        for r in never:
            self.assertEqual(r["j_hat"], 0)
            self.assertEqual(r["gain"], 0.0)
            self.assertFalse(r["harmful"])
            self.assertGreaterEqual(r["regret"], 0.0)
        sel = select_dev_fixed_group(_synthetic_level2(split="dev"), split="dev")
        self.assertIn(sel["j_star"], (1, 2, 3, 4))
        fixed = apply_dev_fixed(l2, "formal", 4, sel["j_star"])
        self.assertEqual(len(fixed), 6)
        for r in fixed:
            self.assertEqual(r["j_hat"], sel["j_star"])

    def test_oracle_capture(self):
        l2 = _synthetic_level2()
        l3 = recompute_level3(l2)
        l3h4 = [r for r in l3 if r["split"] == "formal" and r["H"] == 4]
        l3h4 += baseline_never(l2, "formal", 4)
        oc = oracle_capture(l3h4, "formal", 4, methods=("C", "never"))
        self.assertEqual(oc["C"]["n_units"], 6)
        self.assertGreater(oc["C"]["oracle_capture"], oc["never"]["oracle_capture"])

    def test_descriptive_and_margins(self):
        l2 = _synthetic_level2()
        l3 = recompute_level3(l2)
        desc = descriptive_rates(l3, l2, "formal", 4, ("C",))
        self.assertEqual(desc[0]["method"], "C")
        self.assertEqual(desc[0]["n_units"], 6)
        m = margins(l2, "formal", 4)
        self.assertIn("true_margin", m)

    def test_replay_score_availability_missing(self):
        avail = replay_score_availability(_synthetic_level2(), [])
        self.assertEqual(avail["status"], "MISSING_REPLAY_SCORE")


class TestFrozenCsvInvariants(unittest.TestCase):
    """只读冻结 CSV 的不变式（轻量集成测试）。"""

    def test_frozen_level2_invariants(self):
        import csv
        path = ROOT / "outputs/fr_decision_utility_v01/level2_action_predictions.csv"
        if not path.exists():
            self.skipTest("frozen CSV not present")
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        keys = set()
        for r in rows:
            key = (r["split"], r["seed"], r["step"], r["stream_id"], r["j"], r["H"], r["lambda"])
            self.assertNotIn(key, keys)
            keys.add(key)
        self.assertEqual(len(rows), 1800)
        # no-op 全 0
        for r in rows:
            if r["j"] == "0":
                self.assertEqual(float(r["actual_delta_eval"]), 0.0)
                self.assertEqual(float(r["pred_delta_C"]), 0.0)


if __name__ == "__main__":
    unittest.main()
