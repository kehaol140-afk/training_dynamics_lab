"""Unit tests for G3 clipping-intervention module and runner (pure, no GPU).

Run from repo root:
  PYTHONPATH=research/src /home/b205/miniforge3/envs/fr_minimind/bin/python \
      -m unittest research.tests.test_g3_clipping_intervention_v01
"""
from __future__ import annotations

import inspect
import json
import math
import unittest
from pathlib import Path

import torch

from fr import g3_clipping_intervention_v01 as g3
from fr import g1a_26m_scale_generalization_v01 as g1a
from fr.a0c_adamw_response_propagation_v01 import (
    _metric as frozen_metric,
    adamw_tangent_step,
    clip_jvp,
    relative_error as frozen_rel_err,
)
from fr.batch_manifest import manifest_sha256, save_manifest

ROOT = Path(__file__).resolve().parents[2]
G1A_THETA = ROOT / "outputs/generalization_g1a_26m_v01/theta/seed_2042/theta_mapping_raw.json"
G3_RUNNER = ROOT / "research/scripts/g3_clipping_intervention_v01.py"


def _toy_session(seed: int = 0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    x = torch.randn(2, 4)
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()  # populate step/exp_avg/exp_avg_sq
    optimizer.zero_grad(set_to_none=True)
    return torch, model, optimizer, named


def _random_tangent(named, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    return {
        n: torch.randn(p.shape, generator=gen, dtype=p.dtype)
        for n, p in named
    }


class TestNoClipJVPIsIdentity(unittest.TestCase):
    def test_clip_jvp_identity_under_disabled_max_norm(self):
        torch, _model, _opt, named = _toy_session()
        g = _random_tangent(named, seed=7)
        t = _random_tangent(named, seed=8)
        out = clip_jvp(
            gradient_by_name=g,
            tangent_gradient_by_name=t,
            max_norm=g3.CLIP_DISABLED_MAX_NORM,
        )
        self.assertFalse(bool(out["clip_active"]))
        self.assertEqual(float(out["clip_coefficient"]), 1.0)
        for n, _p in named:
            self.assertTrue(torch.equal(out["clip_jvp"][n], t[n]))
            self.assertTrue(torch.equal(out["clipped_gradient"][n], g[n]))

    def test_disabled_max_norm_matches_arithmetics(self):
        torch, _model, _opt, named = _toy_session()
        g = _random_tangent(named, seed=9)
        t = _random_tangent(named, seed=10)
        norm = math.sqrt(sum(float(v.pow(2).sum()) for v in g.values()))
        self.assertLess(norm, g3.CLIP_DISABLED_MAX_NORM)
        out = clip_jvp(
            gradient_by_name=g, tangent_gradient_by_name=t,
            max_norm=g3.CLIP_DISABLED_MAX_NORM,
        )
        self.assertFalse(bool(out["clip_active"]))


class TestNoClipBHFixedEqualsCSynthetic(unittest.TestCase):
    def _step_pair(self, grad_clip: float):
        torch, model, optimizer, named = _toy_session()
        g = _random_tangent(named, seed=11)
        q = _random_tangent(named, seed=12)
        t = _random_tangent(named, seed=13)
        zero = {n: torch.zeros_like(p) for n, p in named}
        # B pulse identical for both models at h=1 semantics is irrelevant
        # here: we directly exercise the h>1 construction.
        b_clip = clip_jvp(
            gradient_by_name=g, tangent_gradient_by_name=q, max_norm=grad_clip,
        )
        c_real = clip_jvp(
            gradient_by_name=g, tangent_gradient_by_name=t, max_norm=grad_clip,
        )["clip_jvp"]
        bh_fixed_in = t                       # raw HVP tangent
        c_identity_in = t                     # [FR-G3A] future clip-JVP -> identity
        common = dict(
            named_parameters=named, optimizer=optimizer,
            clipped_gradient_by_name=b_clip["clipped_gradient"],
            tangent_theta_by_name=zero, tangent_m_by_name=zero, tangent_v_by_name=zero,
        )
        bh = adamw_tangent_step(
            tangent_clipped_gradient_by_name=bh_fixed_in, **common,
        )
        c_ident = adamw_tangent_step(
            tangent_clipped_gradient_by_name=c_identity_in, **common,
        )
        c_real_out = adamw_tangent_step(
            tangent_clipped_gradient_by_name=c_real, **common,
        )
        return named, bh, c_ident, c_real_out

    def test_bh_fixed_equals_c_identity_when_clip_active(self):
        named, bh, c_ident, c_real = self._step_pair(grad_clip=0.5)
        for n, _p in named:
            self.assertTrue(torch.equal(bh["tangent_theta"][n], c_ident["tangent_theta"][n]))
            self.assertTrue(torch.equal(bh["tangent_m"][n], c_ident["tangent_m"][n]))
            self.assertTrue(torch.equal(bh["tangent_v"][n], c_ident["tangent_v"][n]))
        differs = any(
            not torch.equal(bh["tangent_theta"][n], c_real["tangent_theta"][n])
            for n, _p in named
        )
        self.assertTrue(differs)  # sanity: the test can tell the legs apart

    def test_bh_fixed_equals_c_real_when_clip_inactive(self):
        named, bh, c_ident, c_real = self._step_pair(grad_clip=g3.CLIP_DISABLED_MAX_NORM)
        for n, _p in named:
            self.assertTrue(torch.equal(bh["tangent_theta"][n], c_ident["tangent_theta"][n]))
            self.assertTrue(torch.equal(bh["tangent_theta"][n], c_real["tangent_theta"][n]))


class TestH1TripleIdentity(unittest.TestCase):
    def test_h1_b_bh_c_same_input(self):
        torch, _model, _opt, named = _toy_session()
        g = _random_tangent(named, seed=21)
        q = _random_tangent(named, seed=22)
        b_in = q  # optimizer-only tangent at h=1
        b_clip = clip_jvp(
            gradient_by_name=g, tangent_gradient_by_name=b_in, max_norm=0.5,
        )["clip_jvp"]
        bh_in = b_clip          # initial pulse through J_clip(g_t)
        c_in = clip_jvp(
            gradient_by_name=g, tangent_gradient_by_name=q, max_norm=0.5,
        )["clip_jvp"]
        for n, _p in named:
            self.assertTrue(torch.equal(b_clip[n], bh_in[n]))
            self.assertTrue(torch.equal(bh_in[n], c_in[n]))


class TestInteractionMetric(unittest.TestCase):
    def _cells(self, delta: float, seeds=(2042, 2043, 2044)):
        cells = []
        for seed in seeds:
            for stage in ("EARLY", "MIDDLE", "LATE"):
                for stream_id in range(4):
                    for h in (4, 8):
                        cells.append({
                            "seed": seed, "stage": stage, "stream_id": stream_id, "H": h,
                            "gain_clip_clip": 0.10,
                            "gain_clip_noclip": 0.10 - delta,
                        })
        return cells

    def test_bootstrap_positive_interaction(self):
        cells = self._cells(delta=0.08)
        out = g3.cluster_bootstrap_interaction(cells, n_resamples=2000, seed=7)
        self.assertEqual(out["n_cells"], len(cells))
        self.assertAlmostEqual(out["observed_mean_delta_clip"], 0.08, places=12)
        self.assertGreater(out["bootstrap"]["pi95_low"], 0.0)
        self.assertEqual(out["bootstrap"]["p_positive"], 1.0)

    def test_bootstrap_zero_interaction(self):
        cells = self._cells(delta=0.0)
        out = g3.cluster_bootstrap_interaction(cells, n_resamples=2000, seed=7)
        self.assertAlmostEqual(out["observed_mean_delta_clip"], 0.0, places=12)
        self.assertLess(out["bootstrap"]["p_positive"], 1.0)


class TestManifestFreezeValidation(unittest.TestCase):
    def test_roundtrip_and_tamper(self):
        import tempfile

        batches = [[0, 1], [2, 3], [4, 5], [6, 7]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            sha = save_manifest(
                path, batches,
                metadata={
                    "protocol": g3.E2_MANIFEST_PROTOCOL,
                    "train_seed": 2042, "state_step": 26, "stream_id": 0,
                    "stream_seed": 123, "batch_seed": 456,
                    "epsilon": 0.1, "rho": g3.RHO,
                },
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            out = g3.validate_g3_e2_manifest(
                payload, expected_seed=2042, expected_step=26,
                expected_stream=0, expected_content_sha=sha, accumulation_steps=2,
            )
            self.assertEqual(out["batches"], batches)
            payload["batches"][0][0] = 999
            with self.assertRaises(ValueError):
                g3.validate_g3_e2_manifest(
                    payload, expected_seed=2042, expected_step=26,
                    expected_stream=0, expected_content_sha=sha, accumulation_steps=2,
                )

    def test_wrong_protocol_rejected(self):
        payload = {
            "manifest_version": "0.2",
            "metadata": {
                "protocol": "WRONG", "train_seed": 2042, "state_step": 26,
                "stream_id": 0, "stream_seed": 1, "batch_seed": 2,
                "epsilon": 0.1, "rho": g3.RHO,
            },
            "batches": [[0, 1]],
            "sha256": manifest_sha256([[0, 1]]),
        }
        with self.assertRaises(ValueError):
            g3.validate_g3_e2_manifest(
                payload, expected_seed=2042, expected_step=26,
                expected_stream=0, expected_content_sha=payload["sha256"],
                accumulation_steps=2,
            )


class TestPairedStreamIdentity(unittest.TestCase):
    @unittest.skipUnless(G1A_THETA.is_file(), "frozen G1a theta record required")
    def test_g3_e2_seeds_match_frozen_g1a_streams(self):
        theta = json.loads(G1A_THETA.read_text(encoding="utf-8"))
        self.assertTrue(theta.get("technical_pass"))
        for state in theta["states"]:
            step = int(state["step"])
            for rec in state["e2_streams"]:
                sid = int(rec["stream_id"])
                expected = g1a.two_level_seed(g1a.E2_BASE_SEED, 2042, step, sid)
                self.assertEqual(int(rec["stream_seed"]), expected)
                self.assertEqual(
                    int(rec["batch_seed"]),
                    g1a.batch_seed_from_stream_seed(expected),
                )

    def test_e1_derivation_matches_g1a_module(self):
        self.assertEqual(g3.E1_BASE_SEED, g1a.E1_BASE_SEED)
        self.assertEqual(g3.E2_BASE_SEED, g1a.E2_BASE_SEED)
        self.assertEqual(g3.E2_STREAMS, g1a.E2_STREAMS)


class TestNoClipTrainingFlagSemantics(unittest.TestCase):
    def test_train_loop_never_calls_clip_grad_norm(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("g3_runner_under_test", G3_RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = inspect.getsource(module.cmd_train)
        self.assertNotIn("clip_grad_norm_(", source)
        self.assertIn('clip_grad_norm_called": False', source.replace("'", '"'))
        self.assertIn("counterfactual_would_clip_threshold_1", source)
        self.assertIn("grad_clip_enabled", source)

    def test_train_config_declares_clip_disabled(self):
        cfg = json.loads(
            (ROOT / "research/configs/g3_clipping_intervention_v01.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(cfg["training"]["grad_clip_enabled"])
        self.assertIsNone(cfg["training"]["grad_clip"])


class TestFrozenThresholdVerdict(unittest.TestCase):
    def _noclip_stats(self, gain_clip: float, r2: float = 0.99, mre: float = 0.01,
                      bh_c_rel: float = 0.0):
        rows = []
        for seed in g3.TRAIN_SEEDS:
            for stage, step in g1a.STAGE_STEPS.items():
                for h in g3.HORIZONS_OFFICIAL:
                    # fd needs genuine cross-cell variance or the pooled R²
                    # polyfit is degenerate (RankWarning) -- mimic real data.
                    fd = 0.1 + 0.01 * (seed - 2042) + 0.0004 * step + 0.02 * h
                    fd *= -1.0 if (seed + step + h) % 2 else 1.0
                    err_b = mre + 0.05
                    rows.append({
                        "seed": seed, "stage": stage, "step": step, "H": h,
                        "chi_fd": fd,
                        "chi_B": fd * (1 + err_b),
                        "chi_BH_fixed": fd * (1 + mre),
                        "chi_C": fd * (1 + mre - gain_clip),
                    })
        stats = g3.aggregate_noclip_metrics(rows)
        return stats

    def test_confirmed_when_all_conditions_hold(self):
        stats = self._noclip_stats(gain_clip=1e-9, mre=0.05)
        verdict = g3.decide_mechanism_verdict(
            g3a_pass=True, g3a_max_abs_gap=0.0,
            clip_gains={"gain_curvature_mean": 0.05, "gain_clip_mean": 0.02},
            noclip_gains=stats["pooled_gains_h_gt_1"],
            noclip_stats=stats, delta_clip=0.019,
        )
        self.assertEqual(verdict["verdict"], "MECHANISM_CONFIRMED")
        self.assertTrue(verdict["conditions"]["cond2_gain_clip_collapse_le_20pct"]["pass"])

    def test_not_confirmed_when_gap_persists(self):
        # mre must exceed the injected gain so the signed linearization
        # rel_err_BH - rel_err_C reproduces the requested gain exactly.
        stats = self._noclip_stats(gain_clip=0.02, mre=0.05)
        verdict = g3.decide_mechanism_verdict(
            g3a_pass=True, g3a_max_abs_gap=0.0,
            clip_gains={"gain_curvature_mean": 0.05, "gain_clip_mean": 0.02},
            noclip_gains=stats["pooled_gains_h_gt_1"],
            noclip_stats=stats, delta_clip=0.0,
        )
        self.assertEqual(verdict["verdict"], "MECHANISM_NOT_CONFIRMED")

    def test_partial_when_shrink_between_20_and_50_pct(self):
        stats = self._noclip_stats(gain_clip=0.008, mre=0.05)
        verdict = g3.decide_mechanism_verdict(
            g3a_pass=True, g3a_max_abs_gap=0.0,
            clip_gains={"gain_curvature_mean": 0.05, "gain_clip_mean": 0.02},
            noclip_gains=stats["pooled_gains_h_gt_1"],
            noclip_stats=stats, delta_clip=0.012,
        )
        self.assertEqual(verdict["verdict"], "MECHANISM_PARTIAL")

    def test_contract_mismatch_short_circuits(self):
        verdict = g3.decide_mechanism_verdict(
            g3a_pass=True, g3a_max_abs_gap=0.0,
            clip_gains={"gain_curvature_mean": 0.05, "gain_clip_mean": 0.02},
            noclip_gains={"gain_curvature_mean": 0.0, "gain_clip_mean": 0.0},
            noclip_stats=self._noclip_stats(gain_clip=0.0),
            delta_clip=0.02, contract_mismatch=True,
        )
        self.assertEqual(verdict["verdict"], "CONTRACT_MISMATCH")


class TestMetricReuseFrozen(unittest.TestCase):
    def test_metrics_are_the_frozen_a0c_ones(self):
        self.assertIs(g3._metric, frozen_metric)
        self.assertIs(g3.relative_error, frozen_rel_err)
        self.assertIs(g3.CLIP_DISABLED_MAX_NORM, g3.CLIP_DISABLED_MAX_NORM)
        self.assertEqual(g3.CLIP_DISABLED_MAX_NORM, 1e30)


if __name__ == "__main__":
    unittest.main()
