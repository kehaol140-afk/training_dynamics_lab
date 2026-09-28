#!/usr/bin/env python
"""CPU theory tests for FR-C1 (T1-T9). No GPU, no model.

T1 aggregate-vs-single-point E are different statistics (frozen gate is aggregate)
T2 e = e_bank + e_local and squared-norm identity WITH cross term
T3 same-probe repeatability floor vs cross-probe diversity are different quantities
T4 mean(U(g)) != U(mean(g)) for the real nonlinear shadow update
T5 W/K cumulative gain sign convention
T6 GroupAcc prefix/halves/SE equivalence to direct computation
T7 d_c_to_u matches finite differences of the shadow U wrt c
T8 inner_means via group sums equals direct mean inner product
T9 end-to-end mini-anchor pipeline equivalence to brute force
"""
import math
import unittest

import torch

import fr.c0_gttr_validation_v01 as c0
import fr.c1_gttr_probe_validation_v01 as c1


def rand_td(names, shape=(7, 3), seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return {n: torch.randn(*shape, generator=g, dtype=torch.float64) * scale
            for n in names}


class TestC1Theory(unittest.TestCase):
    def test_T1_aggregate_vs_single_point(self):
        # two anchors: each single-point E > 1, but aggregate E < 1
        # (large target norm at the accurate anchor dominates the ratio)
        f = torch.float64
        y1 = torch.tensor([1.0, 0.0], dtype=f); p1 = torch.tensor([2.2, 0.0], dtype=f)   # E1 = 1.2 > 1
        y2 = torch.tensor([10.0, 0.0], dtype=f); p2 = torch.tensor([10.1, 0.0], dtype=f)  # E2 ~ 1.01 > 1
        rows = [{"sse": float(((p1 - y1) ** 2).sum()), "tss": float((y1 ** 2).sum())},
                {"sse": float(((p2 - y2) ** 2).sum()), "tss": float((y2 ** 2).sum())}]
        agg = c1.E_over_anchors(rows)
        self.assertGreater(agg, 0.0)
        self.assertLess(agg, 1.0)                       # aggregate passes
        self.assertGreater(1.2, 1.0)                    # single points fail
        # converse: one single point < 1 but aggregate > 1 (large error at the
        # small-target anchor dominates the ratio)
        y3 = torch.tensor([1.0, 0.0], dtype=f); p3 = torch.tensor([0.4, 0.0], dtype=f)    # E3 = 0.6 < 1
        y4 = torch.tensor([0.1, 0.0], dtype=f); p4 = torch.tensor([1.0, 0.0], dtype=f)    # E4 = 9.0
        rows2 = [{"sse": float(((p3 - y3) ** 2).sum()), "tss": float((y3 ** 2).sum())},
                 {"sse": float(((p4 - y4) ** 2).sum()), "tss": float((y4 ** 2).sum())}]
        self.assertGreater(c1.E_over_anchors(rows2), 1.0)
        # hand-check the frozen formula: SSE1 = 1.2^2, SSE2 = 0.1^2
        want = math.sqrt((1.44 + 0.01) / (1.0 + 100.0))
        self.assertAlmostEqual(agg, want, places=12)

    def test_T2_decomposition_identity(self):
        names = ["a", "b"]
        p_d = {n: torch.randn(5, 4, dtype=torch.float64) for n in names}
        y_a = {n: torch.randn(5, 4, dtype=torch.float64) for n in names}
        p_a = {n: torch.randn(5, 4, dtype=torch.float64) for n in names}
        d = c1.decompose(p_d, y_a, p_a)
        self.assertLess(d["add_identity_resid"], 1e-12)
        self.assertAlmostEqual(d["sq_identity_resid"], 0.0, delta=1e-9)
        # cross term can be large and negative: force the case
        p_d = {"a": torch.tensor([1.0, 0.0], dtype=torch.float64)}
        p_a = {"a": torch.tensor([2.0, 0.0], dtype=torch.float64)}
        y_a = {"a": torch.tensor([3.0, 0.0], dtype=torch.float64)}
        d = c1.decompose(p_d, y_a, p_a)
        self.assertAlmostEqual(d["cross"], 1.0, places=12)   # <e_bank,e_local> = <[-1],[ -1]> = +1
        self.assertAlmostEqual(d["e_norm"] ** 2,
                               d["e_bank_norm"] ** 2 + d["e_local_norm"] ** 2
                               + 2 * d["cross"], places=12)

    def test_T3_same_probe_vs_cross_probe_floor(self):
        names = ["a"]
        base = rand_td(names, seed=1)
        repeats = [c0.clone_td(base) for _ in range(4)]      # identical evaluation
        same_worst = max(c0.norm_td(c0.sub_td(repeats[i], repeats[j]))
                         for i in range(4) for j in range(i + 1, 4))
        others = [rand_td(names, seed=2 + k) for k in range(4)]
        cross_worst = max(c0.norm_td(c0.sub_td(others[i], others[j]))
                          for i in range(4) for j in range(i + 1, 4))
        self.assertEqual(same_worst, 0.0)                    # bitwise repeatability
        self.assertGreater(cross_worst, 1.0)                 # probe diversity, NOT a floor
        self.assertNotAlmostEqual(same_worst, cross_worst)

    def test_T4_mean_U_vs_U_of_mean(self):
        names = ["a", "b"]
        state = {"m_hat": {n: torch.zeros(6, dtype=torch.float64) for n in names},
                 "v_hat": {n: torch.full((6,), 0.5, dtype=torch.float64) for n in names},
                 "beta1": 0.9, "beta2": 0.999, "t_prev": 46,
                 "eps": 1e-8, "lr": 3e-4, "wd": 0.01,
                 "theta": {n: torch.randn(6, dtype=torch.float64) for n in names},
                 "tau": 1.0}
        grads = [rand_td(names, shape=(6,), seed=k, scale=0.7) for k in range(8)]
        u_of_mean = c0.u_map(c0.ybar(grads), **state)
        mean_of_u = c0.ybar([c0.u_map(g, **state) for g in grads])
        rel = c0.norm_td(c0.sub_td(mean_of_u, u_of_mean)) / max(c0.norm_td(mean_of_u), 1e-12)
        self.assertGreater(rel, 1e-3)   # clip + sqrt(v) nonlinearity: no commutation

    def test_T5_w_k_gain_sign(self):
        # consistent bookkeeping: gain = sum(dW - dK) == V_native - V_arm change
        dW, dK = [0.02, 0.03], [0.01, 0.02]     # sums: dW=0.05, dK=0.03
        # arm's V drops 0.02 more than native's -> arm better by 0.02
        gain, direct, resid = c1.w_k_gain_identity(dW, dK, -0.10, -0.12)
        self.assertAlmostEqual(gain, 0.02, places=12)
        self.assertAlmostEqual(gain, direct, places=12)
        self.assertAlmostEqual(resid, 0.0, places=12)
        self.assertGreater(direct, 0.0)   # positive gain = arm beat native
        # k_v: W>0, dV<0 -> K_V = dV + W can be negative; not a curvature penalty
        self.assertAlmostEqual(c0.k_v(0.8, 1.0, 0.1), -0.1, places=12)

    def test_T6_groupacc_prefix_halves_se(self):
        names = ["a", "b"]
        vecs = [rand_td(names, seed=100 + k) for k in range(32)]
        acc = c1.GroupAcc()
        for j, v in enumerate(vecs):
            acc.add(v, j)
        # prefix means match direct
        for k in (8, 16, 32):
            snap = acc.prefixes()[k]
            direct = c0.ybar(vecs[:k])
            self.assertLess(c0.norm_td(c0.sub_td(c1.mean_of(snap), direct)), 1e-10)
            self.assertAlmostEqual(c1.se_vec_of(snap), c0.se_vec(vecs[:k]), delta=1e-9)
            self.assertAlmostEqual(c1.mean_norm(snap), c0.norm_td(direct), delta=1e-9)
        # halves are the fixed-ID split
        h = acc.halves()
        d_first = c0.ybar(vecs[:16])
        self.assertLess(c0.norm_td(c0.sub_td(c1.mean_of(h["first16"]), d_first)), 1e-10)
        self.assertEqual(h["last16"]["n"], 16)

    def test_T7_d_c_to_u_finite_diff(self):
        torch.manual_seed(7)
        names = ["a"]
        shape = (11,)
        state = {"m_hat": {n: torch.rand(shape, dtype=torch.float64) + 0.1 for n in names},
                 "v_hat": {n: torch.rand(shape, dtype=torch.float64) + 0.2 for n in names},
                 "beta1": 0.9, "beta2": 0.999, "t_prev": 63,
                 "eps": 1e-8, "lr": 3e-4, "wd": 0.0,
                 "theta": {n: torch.zeros(shape, dtype=torch.float64) for n in names},
                 "tau": 10.0}  # clip inactive: d c = d g
        c0v = {n: torch.randn(shape, dtype=torch.float64) * 0.3 for n in names}
        D = c1.d_c_to_u(c=c0v, m_hat=state["m_hat"], v_hat=state["v_hat"],
                        beta1=0.9, beta2=0.999, t_prev=63, eps=1e-8, lr=3e-4)
        eps_fd = 1e-6
        for idx in (0, 5, 10):
            cp = c0.clone_td(c0v); cp["a"][idx] += eps_fd
            cm = c0.clone_td(c0v); cm["a"][idx] -= eps_fd
            up = c0.u_map(cp, **state)["a"][idx]
            um = c0.u_map(cm, **state)["a"][idx]
            fd = float((up - um) / (2 * eps_fd))
            self.assertAlmostEqual(float(D["a"][idx]), fd, delta=1e-7)

    def test_T8_inner_means_via_sums(self):
        names = ["a"]
        xs = [rand_td(names, seed=20 + k) for k in range(8)]
        ys = [rand_td(names, seed=40 + k) for k in range(8)]
        ax, ay = c1.GroupAcc(), c1.GroupAcc()
        for k, (x, y) in enumerate(zip(xs, ys)):
            ax.add(x, k); ay.add(y, k)
        via_sums = c1.inner_means(ax.snapshot(), ay.snapshot())
        direct = c0.dot_td(c0.ybar(xs), c0.ybar(ys))
        self.assertAlmostEqual(via_sums, direct, delta=1e-9)

    def test_T9_end_to_end_mini_anchor(self):
        names = ["w1", "w2"]
        torch.manual_seed(99)
        # three fake anchors, each with 32-probe D/A banks; brute-force check
        # of aggregate E, decomposition and half-bank aggregation.
        per_anchor = []
        for a in range(3):
            p_d = [rand_td(names, seed=a * 100 + k) for k in range(32)]
            y_a = [rand_td(names, seed=a * 100 + 50 + k) for k in range(32)]
            p_a = [rand_td(names, seed=a * 200 + k) for k in range(32)]
            acc_pd, acc_ya, acc_pa = c1.GroupAcc(), c1.GroupAcc(), c1.GroupAcc()
            for k in range(32):
                acc_pd.add(p_d[k], k); acc_ya.add(y_a[k], k); acc_pa.add(p_a[k], k)
            sp, sy, sa = acc_pd.snapshot(), acc_ya.snapshot(), acc_pa.snapshot()
            sse = c1.sse_error(sp, sy)
            tss = c1.mean_sq_norm(sy)
            # brute force
            e = c0.sub_td(c0.ybar(p_d), c0.ybar(y_a))
            self.assertAlmostEqual(c0.norm_td(e) ** 2, sse, delta=1e-6)
            d = c1.decompose(c0.ybar(p_d), c0.ybar(y_a), c0.ybar(p_a))
            self.assertAlmostEqual(d["sq_identity_resid"], 0.0, delta=1e-6)
            per_anchor.append({"sse": sse, "tss": tss})
        self.assertAlmostEqual(
            c1.E_over_anchors(per_anchor),
            math.sqrt(sum(x["sse"] for x in per_anchor)
                      / sum(x["tss"] for x in per_anchor)), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
