"""A0J unit tests (CPU only): cached-p + fresh-G primitives and pure math.

test_apply_Gt_T_matches_FD is the load-bearing test: it verifies the extracted
G_t^T (one optimizer-local analytic adjoint step + clip transpose) against a
central finite-difference of a REAL torch AdamW + clip_grad_norm_ step map in
float64, for both clip-active and clip-inactive regimes.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

import torch  # noqa: E402

from fr.a0j_future_sensitivity_cache_v01 import (  # noqa: E402
    LAMBDA, apply_Gt_T, choose, clone_p, cosine_maps, cosine_p, gap_table,
    kr_ks_projection, map_bytes, max_abs_score_diff, norm_map, p_bytes,
    predicted, preservation, ranking, score_actions, top_margin,
)
from fr.a0g_decision_utility_v01 import select_action as frozen_select_action  # noqa: E402


def _make_snapshot(torch, *, clip_active: bool, seed: int = 0):
    """One real AdamW+clip step on a tiny float64 problem; returns the A0J
    snapshot dict plus the step function g -> z'(g)."""
    g = torch.Generator().manual_seed(seed)
    p1 = torch.nn.Parameter(torch.randn(7, generator=g, dtype=torch.float64))
    p2 = torch.nn.Parameter(torch.randn(3, 4, generator=g, dtype=torch.float64))
    named = [("a", p1), ("b", p2)]
    opt = torch.optim.AdamW([p1, p2], lr=3e-3, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.01)
    m0 = {n: torch.randn_like(p) * 0.1 for n, p in named}
    v0 = {n: torch.rand_like(p) * 0.05 + 0.01 for n, p in named}
    for n, p in named:
        st = opt.state[p]
        st["step"] = torch.tensor(41.0, dtype=torch.float64)
        st["exp_avg"] = m0[n].clone()
        st["exp_avg_sq"] = v0[n].clone()

    scale = 2.5 if clip_active else 0.1
    raw = {n: torch.randn_like(p) * scale for n, p in named}
    theta0 = {n: p.detach().clone() for n, p in named}

    def step_map(gmap):
        # every FD branch starts from the SAME pre-step state (theta0, m0, v0, step 41)
        with torch.no_grad():
            for n, p in named:
                p.copy_(theta0[n])
            st_step = 41.0
            for n, p in named:
                st = opt.state[p]
                st["step"] = torch.tensor(st_step, dtype=torch.float64)
                st["exp_avg"].copy_(m0[n])
                st["exp_avg_sq"].copy_(v0[n])
        for n, p in named:
            p.grad = gmap[n].clone()
        preclip = float(torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0))
        clipped = {n: p.grad.detach().clone() for n, p in named}
        opt.step()
        z = {n: (p.detach().clone(), opt.state[p]["exp_avg"].clone(),
                 opt.state[p]["exp_avg_sq"].clone()) for n, p in named}
        return z, preclip, clipped

    z0, preclip, clipped = step_map({n: t.clone() for n, t in raw.items()})
    snap = {"exp_avg": {n: m0[n].clone() for n, _ in named},
            "exp_avg_sq": {n: v0[n].clone() for n, _ in named},
            "clipped_gradient": clipped,
            "step_after": 42.0, "lr": 3e-3, "weight_decay": 0.01,
            "beta1": 0.9, "beta2": 0.999, "adam_eps": 1e-8,
            "native_gradient": {n: t.clone() for n, t in raw.items()},
            "preclip_norm": preclip}
    return named, opt, snap, step_map


class TestApplyGtT(unittest.TestCase):
    def _fd_check(self, clip_active: bool):
        torch.manual_seed(7)
        named, opt, snap, step_map = _make_snapshot(torch, clip_active=clip_active)
        gen = torch.Generator().manual_seed(123 if clip_active else 456)
        p = {ch: {n: torch.randn_like(pr, generator=gen) * 0.7 for n, pr in named}
             for ch in ("theta", "m", "v")}
        out = apply_Gt_T(named=named, optimizer=opt, snapshot=snap, p=p, grad_clip=1.0)
        w = out["w"]
        self.assertTrue(bool(snap["preclip_norm"] > 1.0) == clip_active)
        eps = 1e-6
        worst = 0.0
        for trial in range(6):
            delta = {n: torch.randn_like(pr, generator=gen) for n, pr in named}
            zp, _, _ = step_map({n: snap["native_gradient"][n] + eps * delta[n] for n, _ in named})
            zm, _, _ = step_map({n: snap["native_gradient"][n] - eps * delta[n] for n, _ in named})
            lhs = 0.0
            for n, _ in named:
                for ch, idx in (("theta", 0), ("m", 1), ("v", 2)):
                    lhs += float(torch.sum(p[ch][n] * (zp[n][idx] - zm[n][idx]))) / (2 * eps)
            rhs = float(sum(torch.sum(w[n].to(torch.float64) * delta[n]) for n, _ in named))
            worst = max(worst, abs(lhs - rhs))
        self.assertLess(worst, 1e-6,
                        f"FD mismatch (clip_active={clip_active}): {worst:.3e}")

    def test_FD_clip_active(self):
        self._fd_check(clip_active=True)

    def test_FD_clip_inactive(self):
        self._fd_check(clip_active=False)

    def test_determinism_same_inputs_same_outputs(self):
        named, opt, snap, _ = _make_snapshot(torch, clip_active=True, seed=3)
        gen = torch.Generator().manual_seed(9)
        p = {ch: {n: torch.randn_like(pr, generator=gen) for n, pr in named}
             for ch in ("theta", "m", "v")}
        w1 = apply_Gt_T(named=named, optimizer=opt, snapshot=snap, p=p, grad_clip=1.0)["w"]
        w2 = apply_Gt_T(named=named, optimizer=opt, snapshot=snap, p=p, grad_clip=1.0)["w"]
        for n, _ in named:
            self.assertTrue(torch.equal(w1[n], w2[n]))

    def test_clone_and_bytes(self):
        named, opt, snap, _ = _make_snapshot(torch, clip_active=True, seed=5)
        gen = torch.Generator().manual_seed(11)
        p = {ch: {n: torch.randn_like(pr, generator=gen) for n, pr in named}
             for ch in ("theta", "m", "v")}
        pc = clone_p(p)
        self.assertEqual(p_bytes(p), p_bytes(pc))
        self.assertEqual(p_bytes(p), 3 * (7 + 12) * 8)
        w = apply_Gt_T(named=named, optimizer=opt, snapshot=snap, p=p, grad_clip=1.0)["w"]
        self.assertEqual(map_bytes(w), (7 + 12) * 8)


class TestSelectorAndScoring(unittest.TestCase):
    def test_argmin_with_noop_and_tie(self):
        h = {0: 0.0, 1: -2.0, 2: 1.0, 3: -2.0, 4: 0.5}
        self.assertEqual(choose(h), 1)            # tie j=1/j=3 -> lowest id
        self.assertEqual(ranking(h), [1, 3, 0, 4, 2])
        # frozen select_action agreement incl. lambda path
        self.assertEqual(choose(h), int(frozen_select_action(predicted(h))))
        self.assertEqual(LAMBDA, 0.05)

    def test_noop_wins_when_all_positive(self):
        h = {0: 0.0, 1: 2.0, 2: 1.0, 3: 3.0, 4: 0.5}
        self.assertEqual(choose(h), 0)

    def test_score_actions_zero_noop(self):
        a = {1: {"x": torch.ones(3, dtype=torch.float64)},
             2: {"x": -torch.ones(3, dtype=torch.float64)}}
        w = {"x": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)}
        h = score_actions(w, a)
        self.assertEqual(h[0], 0.0)
        self.assertAlmostEqual(h[1], 3.0 ** 0.5 * 0 + 1.0)
        self.assertAlmostEqual(h[2], -1.0)
        self.assertIsNone(score_actions(None, a))

    def test_max_abs_score_diff(self):
        self.assertAlmostEqual(max_abs_score_diff({0: 0.0, 1: 1.0}, {0: 0.0, 1: 1.5}), 0.5)


class TestGapDiagnostics(unittest.TestCase):
    def test_pair_count_and_flags(self):
        h_ex = {0: 0.0, 1: -1.0, 2: -3.0, 3: 2.0, 4: -2.5}
        h_ca = {0: 0.0, 1: -1.1, 2: -2.9, 3: 2.2, 4: -2.5}
        rows = gap_table(h_ex, h_ca, None)
        self.assertEqual(len(rows), 10)  # C(5,2)
        flags_top = [r for r in rows if r["is_exact_top1top2_pair"]]
        self.assertEqual(len(flags_top), 1)
        self.assertEqual(sorted((flags_top[0]["j"], flags_top[0]["l"])), [2, 4])
        flags_bn = [r for r in rows if r["is_bestnonzero_noop_pair"]]
        self.assertEqual(len(flags_bn), 1)
        self.assertEqual(sorted((flags_bn[0]["j"], flags_bn[0]["l"])), [0, 2])
        for r in rows:
            self.assertIsNone(r["gap_stale"])
            self.assertAlmostEqual(r["gap_err_cache"],
                                   r["gap_cache"] - r["gap_exact"], places=12)
        tm = top_margin(h_ex)
        self.assertEqual((tm["top1"], tm["top2"]), (2, 4))
        self.assertGreater(tm["margin_pred"], 0.0)


class TestPreservationAndProjection(unittest.TestCase):
    def _rows(self):
        rows = []
        for age, (ce, cc, cs) in {1: (2, 2, 1), 2: (1, 3, 1), 4: (4, 4, 0)}.items():
            for s in range(2):
                rows.append({"age": age, "chosen_exact": ce, "chosen_cache": cc,
                             "chosen_stale": cs, "rank_agreement_cache": cc == ce,
                             "rank_agreement_stale": cs == ce})
        return rows

    def test_preservation(self):
        rows = self._rows()
        pc = preservation(rows, method="cache")
        ps = preservation(rows, method="stale")
        self.assertEqual(pc[1]["chosen_id_preservation"], "2/2")
        self.assertEqual(pc[2]["chosen_id_preservation"], "0/2")
        self.assertEqual(pc[4]["chosen_id_preservation"], "2/2")
        self.assertEqual(ps[1]["chosen_id_preservation"], "0/2")
        self.assertEqual(ps[4]["chosen_id_preservation"], "0/2")
        self.assertAlmostEqual(pc[1]["chosen_id_fraction"], 1.0)

    def test_kr_ks_projection(self):
        rows = kr_ks_projection(T_refresh=1.0, T_score=0.4, T_execute=0.0,
                                T_GTp=0.01, T_native=0.05)
        by = {(r["K_r"], r["K_s"]): r for r in rows}
        r11 = by[(1, 1)]
        self.assertAlmostEqual(r11["extra_native_step_equiv"], (1.0 + 0.41) / 0.05)
        r84 = by[(8, 4)]
        self.assertAlmostEqual(r84["refresh_share_equiv"], 1.0 / 8 / 0.05)
        self.assertAlmostEqual(r84["score_share_equiv"], 0.41 / 4 / 0.05)
        self.assertTrue(by[(8, 4)]["beyond_measured_max_age"])   # 8 > max age 4
        self.assertFalse(by[(4, 4)]["beyond_measured_max_age"])
        self.assertTrue(all(r["claim"] == "COST_ONLY / UTILITY_NOT_VALIDATED" for r in rows))


class TestVectorDiagnostics(unittest.TestCase):
    def test_cosines(self):
        a = {"x": torch.tensor([1.0, 0.0]), "y": torch.tensor([0.0, 1.0])}
        b = {"x": torch.tensor([1.0, 0.0]), "y": torch.tensor([0.0, -1.0])}
        self.assertAlmostEqual(cosine_maps(a, a), 1.0)
        self.assertAlmostEqual(cosine_maps(a, b), 0.0, places=12)
        pa = {ch: dict(a) for ch in ("theta", "m", "v")}
        pb = {ch: dict(b) for ch in ("theta", "m", "v")}
        self.assertAlmostEqual(cosine_p(pa, pa), 1.0)
        self.assertIsNone(cosine_maps(None, a))
        self.assertAlmostEqual(norm_map(a) ** 2, 2.0)


if __name__ == "__main__":
    unittest.main()
