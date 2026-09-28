#!/usr/bin/env python3
"""F0 有限源码核验 —— clip JVP 全局点积与完整一步状态 JVP 审计（CPU）。

协议：用户核验指令（本轮无 GPU、不重跑、不读 confirm）。

审计点：
  A. 双参数块反例：g=(3,4), Hs=(4,-3)（两块各一标量），tau=1, eps=1e-6。
     正确 dc（全局 dot=0）=(0.8,-0.6)；逐块局部 dot 的错误结果≈(0.512,-0.216)。
     三路独立核对：手算公式 / autograd functional.jvp / 中心差分 FD；
     再抽取生产 f0_adamw_sens_step 的实际 dc（S_m_old=0 -> S_m_new=(1-b1)*dc）。
  B. 完整一步（两参数块、active/inactive clip、非零旧 m/v、两独立方向）：
     生产 SENS 与独立可微参考对照 theta/m/v 输出导数；参考为中心差分 JVP
     （对 S_theta/S_m/S_v 完整切向），方向 detach；直接项 -d_eta*r64 单列。
  C. sv=0 分支：dvbar=0 -> 结构性零导数合法；dvbar!=0 -> 必须
     ZeroDivisionError；不加 eps/v_floor 掩盖。
  D. value/sensitivity 的 clip active 判定一致性。
  E. init_sensitivity 三组 S 不得共享内层 dict（注入隔离反例）。
"""
import math
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

from fr import f0_terminal_rule_v01 as f0  # noqa: E402
from fr.a0c_adamw_response_propagation_v01 import clip_jvp  # noqa: E402

TAU, EPS_CLIP = 1.0, 1e-6
B1, B2, EPS, WD = 0.9, 0.999, 1e-8, 0.01


def _correct_dc(g_list, hs_list, tau=TAU, eps=EPS_CLIP):
    """独立手算：全局范数 + 全局点积的正确 clip JVP。"""
    r = math.sqrt(sum(float((g ** 2).sum()) for g in g_list))
    coef = min(tau / (r + eps), 1.0)
    dot_global = sum(float((g * h).sum()) for g, h in zip(g_list, hs_list))
    radial = tau * dot_global / (r * (r + eps) ** 2)
    if coef < 1.0:
        return [coef * h - radial * g for g, h in zip(g_list, hs_list)]
    return [h.clone() for h in hs_list]


def _buggy_dc(g_list, hs_list, tau=TAU, eps=EPS_CLIP):
    """逐块局部 dot 的错误版本（缺陷语义复现，供判别）。"""
    r = math.sqrt(sum(float((g ** 2).sum()) for g in g_list))
    coef = min(tau / (r + eps), 1.0)
    out = []
    for g, h in zip(g_list, hs_list):
        dot_local = float((g * h).sum())
        radial = tau * dot_local / (r * (r + eps) ** 2)
        out.append(coef * h - radial * g if coef < 1.0 else h.clone())
    return out


class A_TwoBlockCounterexample(unittest.TestCase):
    def setUp(self):
        self.g = [torch.tensor([3.0], dtype=torch.float64),
                  torch.tensor([4.0], dtype=torch.float64)]
        self.hs = [torch.tensor([4.0], dtype=torch.float64),
                   torch.tensor([-3.0], dtype=torch.float64)]

    def test_independent_formula(self):
        dc = _correct_dc(self.g, self.hs)
        self.assertAlmostEqual(float(dc[0][0]), 0.8, places=6)
        self.assertAlmostEqual(float(dc[1][0]), -0.6, places=6)

    def test_autograd_jvp(self):
        def clip_fn(*gs):
            nrm = torch.sqrt(sum((g * g).sum() for g in gs))
            coef = torch.clamp(TAU / (nrm + EPS_CLIP), max=1.0)
            return tuple(coef * g for g in gs)

        _, jvp = torch.autograd.functional.jvp(
            clip_fn, tuple(self.g), tuple(self.hs))
        for a, b in zip(jvp, _correct_dc(self.g, self.hs)):
            self.assertTrue(torch.allclose(a, b, rtol=1e-10, atol=1e-12))

    def test_fd(self):
        delta = 1e-6

        def clip_val(off):
            gs = [g + off * h for g, h in zip(self.g, self.hs)]
            nrm = torch.sqrt(sum((g * g).sum() for g in gs))
            coef = min(TAU / (float(nrm) + EPS_CLIP), 1.0)
            return [coef * g for g in gs]

        fd = [(a - b) / (2 * delta)
              for a, b in zip(clip_val(delta), clip_val(-delta))]
        for a, b in zip(fd, _correct_dc(self.g, self.hs)):
            self.assertTrue(torch.allclose(a, b, rtol=1e-5, atol=1e-9))

    def test_buggy_version_differs(self):
        bug = _buggy_dc(self.g, self.hs)
        self.assertAlmostEqual(float(bug[0][0]), 0.512, places=6)
        self.assertAlmostEqual(float(bug[1][0]), -0.216, places=6)
        self.assertFalse(all(torch.allclose(a, b, atol=1e-9) for a, b in
                             zip(bug, _correct_dc(self.g, self.hs))))

    def test_production_matches_correct(self):
        theta = {"p0": torch.randn(2, dtype=torch.float64),
                 "p1": torch.randn(3, dtype=torch.float64)}
        g = {"p0": self.g[0].clone(), "p1": self.g[1].clone()}
        m = {k: torch.zeros_like(v) for k, v in theta.items()}
        v = {k: torch.ones_like(v) * 1e-4 for k, v in theta.items()}
        sens = f0.init_sensitivity(theta, [("a", 0.1)])
        sens["Hg"] = {"a": {"p0": self.hs[0].clone(),
                            "p1": self.hs[1].clone()}}
        _, _, _, out, info = f0.f0_adamw_sens_step(
            theta, m, v, g, B1, B2, EPS, WD, 1e-3, 47, tau=TAU, sens=sens)
        dc0 = out["S_m"]["p0"]["a"] / (1 - B1)
        dc1 = out["S_m"]["p1"]["a"] / (1 - B1)
        self.assertTrue(info["clip_active"])
        self.assertAlmostEqual(float(dc0[0]), 0.8, places=6,
                               msg="生产 dc 与全局 dot 公式不一致 -> 缺陷")
        self.assertAlmostEqual(float(dc1[0]), -0.6, places=6)


class B_FullStepStateJVP(unittest.TestCase):
    """两参数块、非零旧 m/v、两独立方向；active 与 inactive 各测一次。

    参考：对 S_theta/S_m/S_v 完整切向的中心差分 JVP（独立可微 step 链），
    生产输出 = 参考链 − d_eta*r64（直接项）。
    """

    def _run_case(self, active, seed=0):
        gt = torch.Generator().manual_seed(seed)
        d0, d1 = 3, 2
        A0 = torch.randn(d0, d0, generator=gt, dtype=torch.float64)
        b0 = torch.randn(d0, generator=gt, dtype=torch.float64)
        A1 = torch.randn(d1, d1, generator=gt, dtype=torch.float64)
        b1 = torch.randn(d1, generator=gt, dtype=torch.float64)
        scale = 6.0 if active else 0.02
        loss_fn = lambda a, b: ((A0 @ a - b0) ** 2).mean() * scale \
            + ((A1 @ b - b1) ** 2).mean() * scale
        th0 = torch.randn(d0, generator=gt, dtype=torch.float64) * 0.3
        th1 = torch.randn(d1, generator=gt, dtype=torch.float64) * 0.3
        m0 = torch.randn(d0, generator=gt, dtype=torch.float64) * 0.05
        m1 = torch.randn(d1, generator=gt, dtype=torch.float64) * 0.05
        v0 = torch.rand(d0, generator=gt, dtype=torch.float64) * 1e-4 + 1e-5
        v1 = torch.rand(d1, generator=gt, dtype=torch.float64) * 1e-4 + 1e-5
        t_step, eta = 47, 3.0e-4
        bc1, bc2 = 1 - B1 ** t_step, 1 - B2 ** t_step

        def step(th0_, th1_, m0_, m1_, v0_, v1_):
            a = th0_.clone().requires_grad_(True)
            b = th1_.clone().requires_grad_(True)
            loss = loss_fn(a, b)
            g0, g1 = torch.autograd.grad(loss, [a, b], create_graph=True)
            nrm = torch.sqrt((g0 * g0).sum() + (g1 * g1).sum())
            coef = torch.clamp(TAU / (nrm + EPS_CLIP), max=1.0)
            c0, c1 = coef * g0, coef * g1
            mp0 = B1 * m0_ + (1 - B1) * c0
            mp1 = B1 * m1_ + (1 - B1) * c1
            vp0 = B2 * v0_ + (1 - B2) * c0 * c0
            vp1 = B2 * v1_ + (1 - B2) * c1 * c1
            r0 = (mp0 / bc1) / ((vp0 / bc2).sqrt() + EPS) + WD * th0_
            r1 = (mp1 / bc1) / ((vp1 / bc2).sqrt() + EPS) + WD * th1_
            return th0_ - eta * r0, th1_ - eta * r1, mp0, mp1, vp0, vp1

        # active 判定（数值）
        th0d_ = th0.clone().requires_grad_(True)
        th1d_ = th1.clone().requires_grad_(True)
        gg = torch.autograd.grad(loss_fn(th0d_, th1d_), [th0d_, th1d_])
        nrm_ = float(torch.sqrt((gg[0] ** 2).sum() + (gg[1] ** 2).sum()))
        self.assertEqual(nrm_ > TAU, active)

        # 生产 SENS
        th0d = th0.clone().requires_grad_(True)
        th1d = th1.clone().requires_grad_(True)
        gs = torch.autograd.grad(loss_fn(th0d, th1d), [th0d, th1d],
                                 create_graph=True)
        theta = {"p0": th0.clone(), "p1": th1.clone()}
        m = {"p0": m0.clone(), "p1": m1.clone()}
        v = {"p0": v0.clone(), "p1": v1.clone()}
        g = {"p0": gs[0].detach(), "p1": gs[1].detach()}
        dirs = [("a", 0.37), ("b", -0.21)]
        sens = f0.init_sensitivity(theta, dirs)
        D0 = torch.randn(d0, generator=gt, dtype=torch.float64) * 0.1
        D1 = torch.randn(d1, generator=gt, dtype=torch.float64) * 0.1
        M0 = torch.randn(d0, generator=gt, dtype=torch.float64) * 0.05
        M1 = torch.randn(d1, generator=gt, dtype=torch.float64) * 0.05
        V0 = torch.randn(d0, generator=gt, dtype=torch.float64) * 1e-6
        V1 = torch.randn(d1, generator=gt, dtype=torch.float64) * 1e-6
        sens["S_theta"]["p0"]["a"] = D0.clone()
        sens["S_theta"]["p1"]["b"] = D1.clone()
        sens["S_m"]["p0"]["b"] = M0.clone()
        sens["S_m"]["p1"]["a"] = M1.clone()
        sens["S_v"]["p0"]["a"] = V0.clone()
        sens["S_v"]["p1"]["b"] = V1.clone()
        hg = {}
        for d, _ in dirs:
            dot = None
            for name, gr in (("p0", gs[0]), ("p1", gs[1])):
                t_ = torch.sum(gr * sens["S_theta"][name][d].detach(),
                               dtype=torch.float64)
                dot = t_ if dot is None else dot + t_
            vj = torch.autograd.grad(dot, [th0d, th1d], retain_graph=True)
            hg[d] = {"p0": vj[0].detach(), "p1": vj[1].detach()}
        sens["Hg"] = hg
        _, _, _, out, info = f0.f0_adamw_sens_step(
            theta, m, v, g, B1, B2, EPS, WD, eta, t_step, tau=TAU, sens=sens)
        self.assertEqual(info["clip_active"], active)

        # 参考：逐方向完整切向中心差分
        delta = 1e-6

        def ref_dir(d):
            def step_d(off):
                args = [th0 + off * sens["S_theta"]["p0"][d],
                        th1 + off * sens["S_theta"]["p1"][d],
                        m0 + off * sens["S_m"]["p0"][d],
                        m1 + off * sens["S_m"]["p1"][d],
                        v0 + off * sens["S_v"]["p0"][d],
                        v1 + off * sens["S_v"]["p1"][d]]
                return step(*args)
            pp = step_d(delta)
            mm = step_d(-delta)
            return [(a - b) / (2 * delta) for a, b in zip(pp, mm)]

        refs = {d: ref_dir(d) for d, _ in dirs}
        # r64（直接项补偿）
        coef2 = min(TAU / (nrm_ + EPS_CLIP), 1.0)
        r64 = {}
        for blk, gg_, th, mm_, vv in (("p0", gg[0].detach(), th0, m0, v0),
                                      ("p1", gg[1].detach(), th1, m1, v1)):
            cc = coef2 * gg_
            mp = B1 * mm_ + (1 - B1) * cc
            vp = B2 * vv + (1 - B2) * cc * cc
            r64[blk] = ((mp / bc1) / ((vp / bc2).sqrt() + EPS)
                        + WD * th).to(torch.float64)
        out_idx = {"theta": (0, 1), "m": (2, 3), "v": (4, 5)}
        for d, d_eta in dirs:
            for name in ("p0", "p1"):
                for which in ("theta", "m", "v"):
                    prod = out[f"S_{which}"][name][d]
                    sel = out_idx[which][0 if name == "p0" else 1]
                    ref = refs[d][sel].to(torch.float64)
                    if which == "theta":
                        ref = ref - d_eta * r64[name]
                    self.assertTrue(torch.allclose(
                        prod, ref, rtol=1e-6, atol=1e-9),
                        f"active={active} d={d} {name} S_{which}: "
                        f"max|diff|={float((prod - ref).abs().max()):.3e}")

    def test_step_jvp_active(self):
        self._run_case(active=True, seed=1)

    def test_step_jvp_inactive(self):
        self._run_case(active=False, seed=2)


class C_ZeroBranches(unittest.TestCase):
    def _mk(self, c0_value):
        theta = {"p": torch.tensor([1.0, 2.0], dtype=torch.float64)}
        g = {"p": torch.tensor([c0_value, 0.5], dtype=torch.float64)}
        m = {"p": torch.tensor([0.0, 0.2], dtype=torch.float64)}
        v = {"p": torch.tensor([0.0, 1e-4], dtype=torch.float64)}
        return theta, g, m, v

    def test_sv0_dvbar0_zero_derivative_ok(self):
        theta, g, m, v = self._mk(0.0)
        sens = f0.init_sensitivity(theta, [("a", 0.1)])
        sens["Hg"] = {"a": {"p": torch.zeros_like(theta["p"])}}
        theta_n, m_n, v_n, out, info = f0.f0_adamw_sens_step(
            theta, m, v, g, B1, B2, EPS, 0.0, 1e-3, 47, tau=1e9, sens=sens)
        self.assertTrue(torch.isfinite(out["S_theta"]["p"]["a"]).all())
        self.assertEqual(float(out["S_theta"]["p"]["a"][0]), 0.0)
        self.assertTrue(torch.isfinite(theta_n["p"]).all())

    def test_sv0_dvbar_nonzero_raises(self):
        theta, g, m, v = self._mk(0.0)
        sens = f0.init_sensitivity(theta, [("a", 0.1)])
        sens["Hg"] = {"a": {"p": torch.zeros_like(theta["p"])}}
        sens["S_v"]["p"]["a"] = torch.tensor([5e-9, 0.0],
                                             dtype=torch.float64)
        with self.assertRaises(ZeroDivisionError):
            f0.f0_adamw_sens_step(theta, m, v, g, B1, B2, EPS, 0.0, 1e-3,
                                  47, tau=1e9, sens=sens)


class D_ActiveMaskConsistency(unittest.TestCase):
    def test_value_and_sens_masks_agree(self):
        for scale in (6.0, 0.02, 0.35):
            g0 = torch.full((4,), scale, dtype=torch.float32)
            g = {"p": g0}
            zero = {"p": torch.zeros_like(g0)}
            res = clip_jvp(gradient_by_name=g,
                           tangent_gradient_by_name=zero, max_norm=TAU)
            theta = {"p": torch.randn(4, dtype=torch.float64)}
            m = {k: torch.zeros_like(vv) for k, vv in theta.items()}
            v = {k: torch.ones_like(vv) * 1e-4 for k, vv in theta.items()}
            sens = f0.init_sensitivity(theta, [("a", 0.1)])
            sens["Hg"] = {"a": {"p": torch.zeros_like(
                theta["p"], dtype=torch.float64)}}
            _, _, _, _, info = f0.f0_adamw_sens_step(
                theta, m, v, g, B1, B2, EPS, WD, 1e-3, 47, tau=TAU,
                sens=sens)
            self.assertEqual(res["clip_active"], info["clip_active"],
                             f"scale={scale}")


class E_InitNoAliasing(unittest.TestCase):
    def test_s_dicts_independent(self):
        theta = {"p0": torch.randn(2, dtype=torch.float64),
                 "p1": torch.randn(3, dtype=torch.float64)}
        dirs = [("a", 0.1), ("b", 0.2)]
        sens = f0.init_sensitivity(theta, dirs)
        sens["S_theta"]["p0"]["a"] = torch.ones(2, dtype=torch.float64)
        self.assertEqual(float(sens["S_m"]["p0"]["a"].abs().sum()), 0.0)
        self.assertEqual(float(sens["S_v"]["p0"]["a"].abs().sum()), 0.0)
        sens["S_m"]["p1"]["b"] = torch.full((3,), 2.0, dtype=torch.float64)
        self.assertEqual(float(sens["S_theta"]["p1"]["b"].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
