"""T2a regression tests: verify invariants of the executed calibration from its outputs."""
from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "t2a_scale_v1"


class TestT2aInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((OUT / "summary.json").read_text())
        cls.lock = json.loads((OUT / "calibration_lock.json").read_text())
        cls.resp = list(csv.DictReader(open(OUT / "response_errors.csv")))
        cls.mapr = list(csv.DictReader(open(OUT / "map_errors.csv")))
        cls.raw = torch.load(OUT / "raw_natural.pt", map_location="cpu", weights_only=False)

    def test_matrix_elements(self):
        self.assertEqual(sum(self.summary["matrix_elements_actual"].values()), 2490368)

    def test_targets_and_batches(self):
        self.assertEqual(len(self.summary["batch_info"]), 10)
        for b in self.summary["batch_info"]:
            self.assertEqual(b["valid_targets"], 32640)
        self.assertEqual(sum(b["valid_targets"] for b in self.summary["batch_info"]), 326400)

    def test_no_model_updates(self):
        self.assertEqual(self.summary["model_updates"], 0)
        self.assertEqual(self.summary["eval_access"], 0)

    def test_manifest_sha_in_lock_context(self):
        blob = json.loads((OUT / "batch_manifest.json").read_text())
        import hashlib
        sha = hashlib.sha256(json.dumps(blob["steps"], sort_keys=True).encode()).hexdigest()
        self.assertEqual(sha, "1e8b29c7a1bbe492c7203c0b53bc1de8f7fa3f2433e39ab2db9e883c1ccf292e")
        # first5/last5 window disjointness
        s = blob["steps"]
        f5 = {(e["domain"], e["index"]) for st in s[:5] for e in st}
        l5 = {(e["domain"], e["index"]) for st in s[5:] for e in st}
        self.assertTrue(f5.isdisjoint(l5))

    def test_h_construction_math(self):
        beta = 0.95
        for n, H in self.raw["H_i"].items():
            M = self.raw["M_star"][n]
            for bi in range(10):
                G = self.raw["G_post_clip"][n][bi]
                expect = beta ** 2 * M.double() + (1 - beta ** 2) * G.double()
                self.assertLess(float((H[bi].double() - expect).abs().max()), 1e-6)

    def test_h_ref_is_first5_fp64_mean_cast_once(self):
        for n, H in self.raw["H_i"].items():
            expect = torch.stack([H[i] for i in range(5)]).double().mean(0).float()
            self.assertLess(float((self.raw["H_ref"][n] - expect).abs().max()), 1e-7)

    def test_lock_hashes_match_saved(self):
        for n, sha in self.lock["H_ref_sha256"].items():
            h = hashlib.sha256(self.raw["H_ref"][n].numpy().tobytes()).hexdigest() if False else None
        # torch.save may reorder; verify via recompute on the saved tensor bytes
        import hashlib
        for n, sha in self.lock["H_ref_sha256"].items():
            t = self.raw["H_ref"][n].detach().cpu().contiguous()
            self.assertEqual(hashlib.sha256(t.numpy().tobytes()).hexdigest(), sha)

    def test_response_identity_all_points(self):
        # delta_native - L == r_real + r_numeric, recomputed residual error recorded per point
        self.assertLess(self.summary["identity_max_abs_err_over_all_points"], 1e-7)
        self.assertEqual(self.summary["n_response_points"], 660)

    def test_alpha_zero_all_zero_response(self):
        z = [r for r in self.resp if r["alpha"] == "0.0"]
        self.assertEqual(len(z), 60)
        for r in z:
            self.assertEqual(r["native_zero_response"], "True")
            self.assertEqual(float(r["norm_delta_native"]), 0.0)

    def test_no_zero_response_at_nonzero_alpha(self):
        nz = [r for r in self.resp if r["alpha"] != "0.0"]
        self.assertEqual(len(nz), 600)
        self.assertEqual(sum(1 for r in nz if r["native_zero_response"] == "True"), 0)

    def test_alpha_plus_one_uses_saved_H(self):
        for r in self.resp:
            if r["alpha"] == "1.0" and r["sign"] == "+":
                # natural point: actual perturbation equals the natural offset of that batch
                self.assertLess(abs(float(r["actual_rel_perturb"]) -
                                    float(r["natural_rel_offset"])), 1e-9)

    def test_same_H_decomposition_identities(self):
        for r in self.mapr:
            if r["comparison"] in ("fp32_vs_fp64", "native_vs_fp64"):
                self.assertEqual(r["identity_ok"], "True")

    def test_cal_input_mean_drift_is_zero(self):
        # exactly zero up to the single fp64-mean->fp32 cast rounding (~5e-8 relative);
        # mean(fp32 H_i) != fp32(mean fp64 of H_i) by construction of the H_ref rule
        for r in self.mapr:
            if r["half"] == "cal":
                self.assertLess(float(r["input_mean_drift_rel"]), 1e-6)

    def test_repetition_bitwise(self):
        self.assertTrue(all(self.summary["repetition_all_bitwise"].values()))

    def test_natural_offsets_scale(self):
        # documented finding: real-scale natural H offsets are O(0.5-0.73), far above toy deltas
        allv = [v for vals in self.summary["natural_rel_offsets"].values() for v in vals]
        self.assertGreater(min(allv), 0.4)
        self.assertLess(max(allv), 0.8)

    def test_budget(self):
        c = json.loads((OUT / "cost_ledger.json").read_text())
        self.assertLess(c["gpu_wall_seconds_including_failed_attempt"], 1800)
        self.assertLess(c["output_bytes_t2a"], 1073741824)


if __name__ == "__main__":
    unittest.main(verbosity=2)
