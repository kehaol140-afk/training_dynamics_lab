#!/usr/bin/env python3
"""Unit tests for a0u pure functions (offline, no GPU)."""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

import fr.a0u_response_schedule_discovery_v01 as a0u


class TestScheduleRule(unittest.TestCase):
    def test_partial_windows(self):
        chi = {46: -1.0, 52: -1.0, 64: -1.0, 82: -1.0}
        segs = a0u.build_schedule(chi)
        self.assertEqual([s["knot"] for s in segs], [46, 52, 64, 82])
        self.assertIsNone(segs[0]["R_avg"])
        self.assertEqual(segs[0]["multiplier"], 1.0)
        self.assertEqual(segs[1]["prev_steps"], [46])
        self.assertEqual(segs[2]["prev_steps"], [46, 52])
        self.assertEqual(segs[3]["prev_steps"], [46, 52, 64])
        up = float(__import__("numpy").exp(a0u.ALPHA))
        self.assertTrue(all(abs(s["multiplier"] - up) < 1e-12 for s in segs[1:]))

    def test_sign_flip(self):
        chi = {46: -1.0, 52: -1.0, 64: 2.0, 82: 2.0}
        segs = a0u.build_schedule(chi)
        up = float(__import__("numpy").exp(a0u.ALPHA))
        # mean(chi[46], chi[52]) = -1.0 -> u=+alpha -> multiplier up
        self.assertAlmostEqual(segs[2]["multiplier"], up)

    def test_segment_map(self):
        chi = {46: -1.0, 52: -1.0, 64: -1.0, 82: -1.0}
        segs = a0u.build_schedule(chi)
        up = float(__import__("numpy").exp(a0u.ALPHA))
        self.assertEqual(a0u.multiplier_at_offset(1, segs), 1.0)
        self.assertEqual(a0u.multiplier_at_offset(6, segs), 1.0)
        self.assertEqual(a0u.multiplier_at_offset(7, segs), up)
        self.assertEqual(a0u.multiplier_at_offset(18, segs), up)
        self.assertEqual(a0u.multiplier_at_offset(19, segs), up)
        self.assertEqual(a0u.multiplier_at_offset(36, segs), up)
        self.assertEqual(a0u.multiplier_at_offset(64, segs), up)

    def test_properties(self):
        chi = {46: -1.0, 52: -1.0, 64: 1.0, 82: -1.0}
        props = a0u.schedule_properties(a0u.build_schedule(chi))
        # multipliers = [1, up, up, up]: knot 64 averages knots {46,52}
        self.assertEqual(props["n_lr_changes"], 1)
        self.assertEqual(props["n_native_segments"], 1)
        self.assertEqual(props["n_up_segments"], 3)


class TestStats(unittest.TestCase):
    def test_curve_metrics(self):
        losses = {"12": 3.0, "24": 2.0, "36": 1.5, "48": 1.2, "64": 1.0}
        m = a0u.curve_metrics(losses)
        self.assertAlmostEqual(m["final_loss"], 1.0)
        self.assertEqual(len(m["checkpoints"]), 5)
        self.assertAlmostEqual(m["late_std"],
                               __import__("numpy").std([1.5, 1.2, 1.0]))

    def test_wilcoxon_degenerate(self):
        r = a0u.wilcoxon_safe([0.0, 0.0, 0.0])
        self.assertIsNone(r["p"])

    def test_verdict_cases(self):
        def block(median, p):
            return {"median": median,
                    "wilcoxon_vs0": {"p": p, "statistic": 1.0}}
        v = a0u.verdict_case(block(1e-4, 1e-3), block(1e-4, 1e-3), block(1e-4, 1e-3))
        self.assertTrue(v["case_A"])
        v = a0u.verdict_case(block(1e-4, 1e-3), block(0.0, 0.5), block(1e-4, 1e-3))
        self.assertTrue(v["case_B"])
        v = a0u.verdict_case(block(-1e-4, 1e-3), block(0.0, 0.5), block(1e-4, 1e-3))
        self.assertTrue(v["case_C"])


if __name__ == "__main__":
    unittest.main()
