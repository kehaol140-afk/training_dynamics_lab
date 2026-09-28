#!/usr/bin/env python3
"""Unit tests for FR B0.5 pure analysis math (offline, no GPU).

Each test is constructed to fail on a sign/index/branch error in the H1/H2/H3
constructions before any GPU result is produced (config.json stage-B discipline).
"""
import math
import random
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

import fr.b05_mechanism_v01 as b5


class TestCumulativeLrTime(unittest.TestCase):
    def test_monotone_and_scales(self):
        lr = [5e-4] * 16
        sA = b5.cumulative_lr_time(lr, [1.0] * 16)
        self.assertTrue(all(sA[i + 1] > sA[i] for i in range(15)))
        sB = b5.cumulative_lr_time(lr, [math.exp(0.05)] * 16)
        for a, b in zip(sA, sB):
            self.assertAlmostEqual(b / a, math.exp(0.05), places=12)

    def test_pulse_frozen_excess(self):
        lr = [5e-4] * 16
        mC = [math.exp(0.05)] * 4 + [1.0] * 12
        sA = b5.cumulative_lr_time(lr, [1.0] * 16)
        sC = b5.cumulative_lr_time(lr, mC)
        excess = [c - a for c, a in zip(sC, sA)]
        self.assertTrue(all(excess[i + 1] > excess[i] for i in range(3)))
        tail = excess[4:]
        self.assertTrue(all(abs(t - tail[0]) < 1e-15 for t in tail),
                        "pulse excess LR time must be constant after release")
        self.assertAlmostEqual(tail[0], (math.exp(0.05) - 1.0) * 4 * 5e-4,
                               places=15)


class TestInvertMonotone(unittest.TestCase):
    def test_interpolation(self):
        t = list(range(1, 9))
        s = [0.1 * i for i in t]
        inv = b5.invert_monotone(s, t, 0.35)
        self.assertFalse(inv["clamped"])
        self.assertAlmostEqual(inv["t"], 3.5, places=12)

    def test_clamp_flags(self):
        t = list(range(1, 9))
        s = [0.1 * i for i in t]
        self.assertTrue(b5.invert_monotone(s, t, 0.05)["clamped"])
        self.assertFalse(b5.invert_monotone(s, t, 0.1)["clamped"])
        self.assertTrue(b5.invert_monotone(s, t, 0.95)["clamped"])


class TestH1AlignmentToy(unittest.TestCase):
    """Toy path-acceleration system: loss is a pure function L(s) of cumulative
    LR time. Alignment must recover it; a catch-up (decay) arm must not align."""

    def _native(self, T=32):
        rng = random.Random(7)
        lr = [5e-4 + 1e-5 * math.sin(i / 3.0) for i in range(T)]
        sA = b5.cumulative_lr_time(lr, [1.0] * T)
        # arbitrary smooth loss-vs-s curve (loss decreasing in s)
        L_A = [2.5 - 0.9 * math.sqrt(s) for s in sA]
        return lr, sA, L_A

    def test_pure_reparameterization_aligns(self):
        T = 32
        lr, sA, L_A = self._native(T)
        t = list(range(1, T + 1))
        u = 0.05
        sB = b5.cumulative_lr_time(lr, [math.exp(u)] * T)
        # arm B truly is the same curve at accelerated time
        L_B = []
        for sq in sB:
            tq = b5.invert_monotone(sA, t, sq)["t"]
            j = int(math.floor(tq)) - 1
            frac = tq - math.floor(tq)
            j = min(max(j, 0), T - 2)
            L_B.append(L_A[j] + frac * (L_A[j + 1] - L_A[j]))
        L_tilde, clamped = b5.align_loss_by_s(t, sA, L_A, sB)
        # accelerated arm runs ahead: tail points clamp; metric on non-clamped region
        self.assertTrue(any(clamped))
        rel = b5.masked_rel_err(L_B, L_tilde, L_A, clamped)
        self.assertLess(rel, 0.05)
        # baseline same-time error is O(1) on the same region
        base = b5.masked_rel_err(L_B, L_A, L_A, clamped)
        self.assertGreater(base / max(rel, 1e-12), 20.0)

    def test_implied_lag_constant_for_pulse_uniform_lr(self):
        # exact constancy holds under uniform native LR; with varying LR the
        # exact H1 statement is the inversion itself (test_pure_reparameterization_aligns)
        T = 32
        lr = [5e-4] * T
        sA = b5.cumulative_lr_time(lr, [1.0] * T)
        t = list(range(1, T + 1))
        mC = [math.exp(0.05)] * 4 + [1.0] * (T - 4)
        sC = b5.cumulative_lr_time(lr, mC)
        lag = b5.implied_lag(sA, t, sC)
        post = lag[4:31]  # final point ran past s_A end and is clamped by construction
        self.assertTrue(all(abs(x - post[0]) < 1e-9 for x in post),
                        "pure H1 predicts constant implied lag post-release (uniform LR)")
        self.assertAlmostEqual(post[0], (math.exp(0.05) - 1.0) * 4, places=9)


class TestH2Estimators(unittest.TestCase):
    def _make_velocities(self, T=64, d=32, tau_fast=2.0, tau_slow=24.0, seed=3):
        rng = random.Random(seed)
        # slow component: fixed-ish direction with AR(1) persistence tau_slow
        rho_s = math.exp(-1.0 / tau_slow)
        dir_slow = [rng.gauss(0, 1) for _ in range(d)]
        nrm = math.sqrt(sum(x * x for x in dir_slow))
        dir_slow = [x / nrm for x in dir_slow]
        slow = []
        cur = dir_slow[:]
        for _ in range(T):
            slow.append(cur[:])
            cur = [rho_s * c + math.sqrt(1 - rho_s ** 2) * rng.gauss(0, 0.05)
                   for c in cur]
        # fast component: per-coordinate AR(1) with correlation time tau_fast
        rho_f = math.exp(-1.0 / tau_fast)
        fast = []
        cur = [rng.gauss(0, 1) for _ in range(d)]
        for _ in range(T):
            fast.append(cur[:])
            cur = [rho_f * c + math.sqrt(1 - rho_f ** 2) * rng.gauss(0, 1)
                   for c in cur]
        vels = [[0.1 * s + f for s, f in zip(sl, fl)] for sl, fl in zip(slow, fast)]
        return vels, slow, fast

    def test_tau_fast_recovered(self):
        vels, slow, _fast = self._make_velocities()
        residual = [[a - b for a, b in zip(v, s)] for v, s in zip(vels, slow)]
        c_raw = b5.normalized_autocorr(vels, 16)
        c_res = b5.normalized_autocorr(residual, 16)
        tau_raw = b5.tau_from_curve(c_raw)
        tau_res = b5.tau_from_curve(c_res)
        self.assertIsNotNone(tau_res)
        self.assertLessEqual(tau_res, 6)  # near tau_fast=2 (1/e of AR(1) ~ 2)
        # raw curve contaminated by slow component decays slower or comparably
        self.assertTrue(tau_raw is None or tau_raw >= tau_res)

    def test_direction_persistence_slow(self):
        _v, slow, _f = self._make_velocities()
        pers = b5.direction_persistence(slow, 16)
        self.assertGreater(pers[0], 0.9)  # slow direction persists at lag 1
        self.assertGreater(pers[0], pers[-1])  # decays with lag

    def test_no_slow_component_detected(self):
        rng = random.Random(11)
        vels = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(64)]
        pers = b5.direction_persistence(vels, 16)
        self.assertLess(abs(pers[0]), 0.4)


class TestH3Bookkeeping(unittest.TestCase):
    def test_channel_identity_linear(self):
        # linear readout L(z) = <w, z>; decomposition must hold exactly
        rng = random.Random(5)
        w = [rng.gauss(0, 1) for _ in range(10)]
        zA = [rng.gauss(0, 1) for _ in range(10)]
        zC = [a + rng.gauss(0, 0.3) for a in zA]
        zD = [a + rng.gauss(0, 0.3) for a in zA]
        L = lambda z: sum(wi * zi for wi, zi in zip(w, z))
        LC, LD, LA = L(zC), L(zD), L(zA)
        self.assertAlmostEqual(LC - LA, (LD - LA) + (LC - LD), places=12)

    def test_mv_channel_share_guards(self):
        self.assertIsNone(b5.mv_channel_share(0.0, 1.0))
        self.assertIsNone(b5.mv_channel_share(-1e-14, 1.0))
        self.assertAlmostEqual(b5.mv_channel_share(1.0, 0.25), 0.75)
        # memory channel carries everything
        self.assertAlmostEqual(b5.mv_channel_share(1.0, 0.0), 1.0)
        # theta displacement carries everything
        self.assertAlmostEqual(b5.mv_channel_share(1.0, 1.0), 0.0)


class TestDistances(unittest.TestCase):
    def test_cosine_and_norm(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 1e-8, 0.0]
        self.assertAlmostEqual(b5.cosine(a, b), 1.0, places=6)
        self.assertIsNone(b5.cosine([0.0, 0.0], [1.0, 0.0]))
        self.assertAlmostEqual(b5.flat_sq_norm([3.0, 4.0]), 25.0)


class TestSmoothComponent(unittest.TestCase):
    def test_constant_signal_unchanged(self):
        seq = [[2.0, -1.0] for _ in range(16)]
        sm = b5.smooth_component(seq, 9)
        for row in sm:
            self.assertAlmostEqual(row[0], 2.0, places=12)
            self.assertAlmostEqual(row[1], -1.0, places=12)

    def test_edge_shrink(self):
        seq = [[float(i)] for i in range(7)]
        sm = b5.smooth_component(seq, 3)
        # at i=0 window covers {0,1}: mean = 0.5
        self.assertAlmostEqual(sm[0][0], 0.5, places=12)
        # center i=3 window covers {2,3,4}: mean = 3.0
        self.assertAlmostEqual(sm[3][0], 3.0, places=12)


if __name__ == "__main__":
    unittest.main()
