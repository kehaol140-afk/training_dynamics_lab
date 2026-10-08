"""T2b round-3 regression tests (from the frozen outputs)."""
from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "t2b_target_screen_v1"


class TestT2bScreen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mani = json.loads((OUT / "data_and_perms_manifest.json").read_text())
        cls.lock = json.loads((OUT / "selection_lock.json").read_text())
        cls.shadow = list(csv.DictReader(open(OUT / "shadow_scores.csv")))
        cls.actual = list(csv.DictReader(open(OUT / "actual_objectives.csv")))
        cls.cfg = json.loads((ROOT / "configs" / "t2b_target_screen_v1.json").read_text())
        cls.meta = json.loads((OUT / "replay_stage_meta.json").read_text())

    def test_w0_matches_t2a_and_blocks_disjoint(self):
        t2a = json.loads((ROOT / "outputs" / "t2a_scale_v1" / "batch_manifest.json").read_text())["steps"]
        m = self.mani["data_blocks"]
        self.assertEqual(m["W0"], t2a)
        ids = {b: {(e["domain"], e["index"]) for st in m[b] for e in st} for b in ("W0", "W1", "C")}
        self.assertTrue(ids["W0"].isdisjoint(ids["W1"]))
        self.assertTrue(ids["W0"].isdisjoint(ids["C"]))
        self.assertTrue(ids["W1"].isdisjoint(ids["C"]))
        self.assertEqual(len(m["C"]), 128)

    def test_permutation_formula_and_uniqueness(self):
        for b in (0, 1):
            ps = [tuple(p["perm"]) for p in self.mani["permutations"][f"W{b}"]]
            self.assertEqual(len(set(ps)), 8)
            for k in range(8):
                ref = tuple(int(x) for x in np.random.Generator(
                    np.random.PCG64(883000 + 100 * b + k)).permutation(10))
                self.assertEqual(ps[k], ref)  # collision rule not triggered

    def test_one_step_virtual_equals_real(self):
        self.assertTrue(json.loads((OUT / "one_step_consistency.json").read_text())["one_step_bitwise_ok"])
        self.assertTrue(self.meta["aliasing_probe_ok"])

    def test_w0_t2a_crosscheck_zero(self):
        cc = json.loads((OUT / "w0_t2a_crosscheck.json").read_text())["w0_vs_t2a_max_abs_diff"]
        for v in cc.values():
            self.assertEqual(v, 0.0)

    def test_plateau_lr_from_checkpoint(self):
        payload = torch.load(ROOT.parent / "muon_transition_experiment/outputs/m1/m1_muon_lr2.0_wd0.02_p0.5_s2030/final_state.pt",
                             map_location="cpu", weights_only=False)
        self.assertEqual(payload["optimizer_states"][0]["param_groups"][0]["lr"], 0.001)
        self.assertEqual(payload["optimizer_states"][1]["param_groups"][0]["lr"], 0.001)
        self.assertEqual(payload["optimizer_states"][0]["param_groups"][0]["momentum"], 0.95)

    def test_selection_lock_matches_argmin(self):
        for b in (0, 1):
            sub = [r for r in self.shadow if r["window"] == str(b)]
            best = min(float(r["S_fin"]) for r in sub)
            cands = [int(r["k"]) for r in sub if float(r["S_fin"]) <= best + 1e-6]
            self.assertEqual(self.lock[f"W{b}"]["kF"], min(cands))

    def test_q_group_isolation_and_counts(self):
        q = torch.load(Path("/tmp/opencode/t2b_q_targets.pt"), weights_only=True)
        for dom in ("A", "B"):
            gs = set(q["sides"]["select"][dom]["group_ids"])
            gc = set(q["sides"]["check"][dom]["group_ids"])
            self.assertTrue(gs.isdisjoint(gc))
        for r in self.actual:
            self.assertEqual(int(r["step10_count_A"]), 512 * 255)
            self.assertEqual(int(r["step138_count_B"]), 512 * 255)

    def test_all_finite_objectives(self):
        for r in self.actual:
            for h in ("step10", "step138"):
                self.assertTrue(np.isfinite(float(r[f"{h}_V"])), r)

    def test_repeat_run_identical(self):
        rep = [r for r in self.actual if r["repeat"] == "True"][0]
        orig = [r for r in self.actual if r["repeat"] == "False"
                and r["window"] == rep["window"] and r["k"] == rep["k"]][0]
        self.assertEqual(rep["step138_V"], orig["step138_V"])
        self.assertEqual(rep["step10_V"], orig["step10_V"])

    def test_summary_verdict_consistency(self):
        s = json.loads((OUT / "target_gain_summary.json").read_text())
        self.assertEqual(s["thresholds"]["e_repeat"], 0.0)
        self.assertIn(s["verdict"], ("TARGET_SELECTION_SIGNAL",
                                     "EFFECT_WITHOUT_RELIABLE_SELECTION",
                                     "NO_USEFUL_SIGNAL_THIS_SETTING"))

    def test_budget(self):
        import os
        size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
        self.assertLess(size, 8 * 1024 ** 3)
        gpu = self.meta["gpu_seconds_replay_stage"] + \
            json.loads((OUT / "real_stage_meta.json").read_text())["gpu_seconds_real_stage"]
        self.assertLess(gpu, 3 * 3600)


if __name__ == "__main__":
    unittest.main(verbosity=2)
