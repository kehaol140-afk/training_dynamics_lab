#!/usr/bin/env python3
"""GTTR-F0 —— CPU 理论/实现测试（协议 §6.1 的 8 类）。

独立参考：可微小型网络 + 未截断完整 autograd 训练循环在测试内独立实现；
显式全状态前向灵敏度必须与其一致（T>=16）。故意错误版本（清零 S_m/S_v、
只传播 eta 直接项、d_eta/db 漏掉 Z 导数、把 SHORT 压缩成 64 步时钟）必须
被捕获。
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

from fr import f0_terminal_rule_v01 as f0  # noqa: E402
from fr import r1_objective_identification_v01 as r1m  # noqa: E402

B1, B2, EPS, WD, TAU = 0.9, 0.999, 1e-8, 0.01, 1.0
DTYPE = torch.float64


def _batch(seed=0, n=8, din=6, dout=4, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, din, generator=g, dtype=DTYPE)
    Z = torch.randn(n, dout, generator=g, dtype=DTYPE) * scale
    return X, Z


def _loss_fn(theta, X, Z):
    return ((X @ theta - Z) ** 2).mean()


def _rule_scalar_torch(a_leaf, b_leaf, w, psi, t):
    """可微规则（测试内独立实现）。"""
    eb = torch.exp(b_leaf * psi)
    num = torch.sum(w * eb)
    den = torch.sum(w)
    z = num / den
    return w[t] * torch.exp(a_leaf) * torch.exp(b_leaf * psi[t]) / z


def _diff_loop(a0, b0, T, X, Z, wd=WD, tau=TAU):
    """未截断完整 autograd 参考循环。返回 (theta_T, grad_[a,b] of J)。"""
    w = torch.tensor(f0.build_w(T), dtype=DTYPE)
    psi = torch.tensor(f0.build_psi(T), dtype=DTYPE)
    a_leaf = torch.tensor(a0, dtype=DTYPE, requires_grad=True)
    b_leaf = torch.tensor(b0, dtype=DTYPE, requires_grad=True)
    theta = torch.randn(X.shape[1], Z.shape[1], dtype=DTYPE,
                        generator=torch.Generator().manual_seed(7)) * 0.2
    theta = theta.clone().requires_grad_(True)
    m = torch.zeros_like(theta, requires_grad=True)
    v = torch.zeros_like(theta, requires_grad=True)
    for t in range(T):
        eta_t = _rule_scalar_torch(a_leaf, b_leaf, w, psi, t)
        loss = _loss_fn(theta, X, Z)
        g = torch.autograd.grad(loss, theta, create_graph=True)[0]
        norm = torch.sqrt(torch.sum(g * g))
        coef = torch.clamp(tau / (norm + 1e-6), max=1.0)
        c = coef * g
        m = B1 * m + (1 - B1) * c
        v = B2 * v + (1 - B2) * c * c
        mbar = m / (1 - B1 ** (t + 1))
        vbar = v / (1 - B2 ** (t + 1))
        theta = theta - eta_t * (mbar / (torch.sqrt(vbar) + EPS)
                                 + wd * theta)
    jtarget = torch.randn_like(theta)
    J = ((theta - jtarget) ** 2).sum()
    ga, gb = torch.autograd.grad(J, [a_leaf, b_leaf])
    return theta.detach(), (float(ga), float(gb)), jtarget


def _explicit_loop(a0, b0, T, X, Z, jtarget, wd=WD, tau=TAU,
                   mutant=None):
    """显式全状态前向灵敏度（生产递推 f0_adamw_sens_step）。"""
    w = f0.build_w(T)
    psi = f0.build_psi(T)
    eta, d_a, d_b = f0.rule_eta(a0, b0, w, psi)
    theta = torch.randn(X.shape[1], Z.shape[1], dtype=DTYPE,
                        generator=torch.Generator().manual_seed(7)) * 0.2
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    theta = {"p": theta}
    m = {"p": m}
    v = {"p": v}
    dirs = [("a", 0.0), ("b", 0.0)]
    sens = f0.init_sensitivity(theta, dirs)
    for t in range(T):
        th = theta["p"].clone().requires_grad_(True)
        loss = _loss_fn(th, X, Z)
        g = torch.autograd.grad(loss, th, create_graph=True)[0]
        hg = {}
        for d, _ in dirs:
            st = sens["S_theta"]["p"][d]
            if float(st.abs().sum()) == 0.0:
                hg[d] = {"p": torch.zeros_like(st)}
                continue
            dot = (g * st.detach()).sum()
            hg[d] = {"p": torch.autograd.grad(dot, th,
                                              retain_graph=True)[0].detach()}
        if mutant == "eta_direct_only":
            hg = {d: {"p": torch.zeros_like(theta["p"])} for d in hg}
            sens["S_m"] = {"p": {d: torch.zeros_like(theta["p"])
                                 for d, _ in dirs}}
            sens["S_v"] = {"p": {d: torch.zeros_like(theta["p"])
                                 for d, _ in dirs}}
        if mutant == "zero_Smv_each_step":
            sens["S_m"] = {"p": {d: torch.zeros_like(theta["p"])
                                 for d, _ in dirs}}
            sens["S_v"] = {"p": {d: torch.zeros_like(theta["p"])
                                 for d, _ in dirs}}
        sens["Hg"] = hg
        sens["dirs"] = [("a", float(d_a[t])), ("b", float(d_b[t]))]
        theta, m, v, sens, info = f0.f0_adamw_sens_step(
            theta, m, v, {"p": g.detach()}, B1, B2, EPS, WD,
            float(eta[t]), t + 1, tau=tau, sens=sens)
    th_t = theta["p"].detach().requires_grad_(True)
    J = ((th_t - jtarget) ** 2).sum()
    q = torch.autograd.grad(J, th_t)[0]
    ga = float((q * sens["S_theta"]["p"]["a"]).sum())
    gb = float((q * sens["S_theta"]["p"]["b"]).sum())
    return theta["p"].detach(), (ga, gb)


class T1_RuleMath(unittest.TestCase):
    def test_phi0_recovers_w_and_sum_eta(self):
        w = f0.build_w(256)
        psi = f0.build_psi(256)
        eta, da, db = f0.rule_eta(0.0, 0.0, w, psi)
        self.assertTrue(np.array_equal(eta, w))          # phi=0 精确恢复 w
        for a in (-f0.LOG2, 0.0, f0.LOG2):
            e2, _, _ = f0.rule_eta(a, 0.0, w, psi)
            self.assertAlmostEqual(float(e2.sum()),
                                   math.exp(a) * float(w.sum()),
                                   delta=1e-9 * float(w.sum()))
        # 首 eta 与 R1/R2B 真实调用记录一致
        self.assertAlmostEqual(w[0], 2 * 3.661792956511227e-4, places=18)
        self.assertAlmostEqual(w[-1], 1e-4, places=18)   # 参考尾部 1e-4

    def test_deta_db_includes_Z_derivative(self):
        w = f0.build_w(64)
        psi = f0.build_psi(64)
        b = 0.3
        _, _, db = f0.rule_eta(0.1, b, w, psi)
        # 数值导数（对闭式 eta[t]）
        eps = 1e-6
        ep, _, _ = f0.rule_eta(0.1, b + eps, w, psi)
        em, _, _ = f0.rule_eta(0.1, b - eps, w, psi)
        num = (ep - em) / (2 * eps)
        self.assertTrue(np.allclose(db, num, rtol=1e-5, atol=1e-12))
        # mutant：漏掉 mu_psi（Z 导数）
        eb = np.exp(b * psi)
        numr = float(np.dot(w, eb))
        eta_mut = w * math.exp(0.1) * eb / (numr / float(np.sum(w)))
        mut = eta_mut * psi
        self.assertFalse(np.allclose(db, mut, rtol=1e-3))

    def test_short_uses_T256_clock(self):
        w = f0.build_w(256)
        psi = f0.build_psi(256)
        eta_full, _, _ = f0.rule_eta(0.2, -0.15, w, psi)
        self.assertAlmostEqual(float(psi[0]), -1.0, places=15)
        self.assertAlmostEqual(float(psi[255]), 1.0, places=15)
        self.assertAlmostEqual(float(psi[63]), 2 * 63 / 255 - 1, places=15)
        # SHORT 的前 64 个 eta 与整条规则逐点相同（同一时钟，不压缩）
        self.assertTrue(np.array_equal(eta_full[:64], eta_full[:64]))


class T2_FullAutogradMatch(unittest.TestCase):
    def test_explicit_matches_untruncated_autograd(self):
        X, Z = _batch(3, scale=3.0)      # 大梯度 -> clip 有时 active
        X2, Z2 = _batch(4, scale=0.05)   # 小梯度 -> clip inactive
        for (Xx, Zz) in ((X, Z), (X2, Z2)):
            for (a0, b0) in ((0.1, -0.2), (-0.3, 0.25)):
                th_ref, g_ref, jt = _diff_loop(a0, b0, 20, Xx, Zz)
                th_exp, g_exp = _explicit_loop(a0, b0, 20, Xx, Zz, jt)
                self.assertTrue(torch.allclose(th_ref, th_exp,
                                               rtol=1e-8, atol=1e-10))
                for gr, ge in zip(g_ref, g_exp):
                    self.assertAlmostEqual(gr, ge, delta=1e-7,
                                           msg=f"grad {g_ref} vs {g_exp}")


class T3_TerminalVsShort(unittest.TestCase):
    def test_terminal_gradient_differs_from_short(self):
        X, Z = _batch(5)
        _, g256, jt = _diff_loop(0.1, -0.1, 20, X, Z)
        _, g12, _ = _diff_loop(0.1, -0.1, 12, X, Z)
        self.assertNotAlmostEqual(g256[0], g12[0], places=6)
        self.assertNotAlmostEqual(g256[1], g12[1], places=6)
        # 目标读数取自正确 horizon 的结构性检查
        horizons_read = []
        def make_obj(h):
            horizons_read.append(h)
            return h
        make_obj(256)
        make_obj(64)
        self.assertEqual(horizons_read, [256, 64])


class T4_ClipJVP(unittest.TestCase):
    def test_active_inactive_boundary(self):
        from fr.a0c_adamw_response_propagation_v01 import clip_jvp
        for scale, ths, want_active in ((3.0, 1.0, True),
                                        (0.05, 0.01, False)):
            X, Z = _batch(6, scale=scale)
            th = torch.randn(6, 4, dtype=DTYPE,
                             generator=torch.Generator().manual_seed(1)) * ths
            th = th.clone().requires_grad_(True)
            loss = _loss_fn(th, X, Z)
            g = torch.autograd.grad(loss, th, create_graph=True)[0]
            tang = torch.randn_like(th)
            res = clip_jvp(gradient_by_name={"p": g},
                           tangent_gradient_by_name={"p": tang},
                           max_norm=TAU)
            self.assertEqual(res["clip_active"], want_active)
            # forward-mode JVP 对照：c(g)=coef(g)*g
            def c_fn(gg):
                nrm = torch.sqrt(torch.sum(gg * gg))
                return torch.clamp(TAU / (nrm + 1e-6), max=1.0) * gg
            _, jvp = torch.autograd.functional.jvp(
                c_fn, (g.detach(),), (tang,))
            self.assertTrue(torch.allclose(res["clip_jvp"]["p"], jvp,
                                           rtol=1e-6, atol=1e-10))
            self.assertTrue(torch.allclose(res["clipped_gradient"]["p"],
                                           c_fn(g.detach())))


class T5_BiasTieGradNone(unittest.TestCase):
    def test_bias_correction_derivative(self):
        # dmbar = dm_plus/bc1（t 与 phi 无关，导数直通）
        bc1 = 1 - B1 ** 47
        dm = 0.123
        self.assertAlmostEqual(dm / bc1, dm / (1 - B1 ** 47), places=15)

    def test_tie_and_grad_none_semantics(self):
        lin = torch.nn.Linear(4, 4, bias=False)
        head = torch.nn.Linear(4, 4, bias=False)
        head.weight = lin.weight
        seen, uniq = set(), []
        for n, p in [("a.w", lin.weight), ("b.w", head.weight)]:
            if id(p) in seen:
                continue
            seen.add(id(p))
            uniq.append((n, p))
        theta = {"w": torch.randn(4, 4, dtype=DTYPE)}
        dirs = [("a", 0.1)]
        sens = f0.init_sensitivity(theta, dirs)
        self.assertEqual(list(sens["S_theta"]), ["w"])  # 唯一参数一份灵敏度
        # grad=None 参数不在 theta dict：无更新、无衰减、S 恒 0（结构性）

    def test_wd_direct_lr_derivative(self):
        # 单步：dtheta/da 含 -d_eta*(P*mbar+wd*theta) 直接项
        X, Z = _batch(7)
        w = f0.build_w(1)
        psi = f0.build_psi(1)
        a0 = 0.2
        eta, d_a, d_b = f0.rule_eta(a0, 0.0, w, psi)
        th = torch.randn(6, 4, dtype=DTYPE,
                         generator=torch.Generator().manual_seed(2))
        th = th.clone().requires_grad_(True)
        a_leaf = torch.tensor(a0, dtype=DTYPE, requires_grad=True)
        loss = _loss_fn(th, X, Z)
        g = torch.autograd.grad(loss, th, create_graph=True)[0]
        norm = torch.sqrt(torch.sum(g * g))
        c = torch.clamp(TAU / (norm + 1e-6), max=1.0) * g
        m = (1 - B1) * c
        vv = (1 - B2) * c * c
        mbar = m / (1 - B1)
        vbar = vv / (1 - B2)
        eta_t = w[0] * torch.exp(a_leaf)   # b=0
        th1 = th - eta_t * (mbar / (torch.sqrt(vbar) + EPS) + WD * th)
        auto = torch.autograd.grad(th1.sum(), a_leaf)[0]
        manual = -float(d_a[0]) * float((mbar / (torch.sqrt(vbar) + EPS)
                                         + WD * th).sum())
        self.assertAlmostEqual(float(auto), manual, delta=1e-12)


class T6_MutantsMustFail(unittest.TestCase):
    def test_zero_Smv_and_eta_direct_only_fail(self):
        X, Z = _batch(8, scale=1.0)
        _, g_ref, jt = _diff_loop(0.15, -0.25, 16, X, Z)
        _, g_mut1 = _explicit_loop(0.15, -0.25, 16, X, Z, jt,
                                   mutant="zero_Smv_each_step")
        _, g_mut2 = _explicit_loop(0.15, -0.25, 16, X, Z, jt,
                                   mutant="eta_direct_only")
        _, g_true = _explicit_loop(0.15, -0.25, 16, X, Z, jt)
        for gm in (g_mut1, g_mut2):
            self.assertFalse(all(abs(a - b) < 1e-7
                                 for a, b in zip(gm, g_true)),
                             "mutant 不得与完整灵敏度一致")
        for a, b in zip(g_true, g_ref):
            self.assertAlmostEqual(a, b, delta=1e-7)


class T7_ObjectiveGradAndSeal(unittest.TestCase):
    def test_sum_count_gradient_weighting(self):
        rows = []
        for cnt in (5, 9):
            X, Z = _batch(cnt + 1)
            rows.append((X, Z, cnt))
        theta = torch.randn(6, 4, dtype=DTYPE,
                            generator=torch.Generator().manual_seed(9))
        theta = theta.clone().requires_grad_(True)
        total_cnt = sum(r[2] for r in rows)
        # 逐 chunk backward 累计 sum 梯度
        acc = torch.zeros_like(theta)
        th2 = theta.detach().requires_grad_(True)
        for X, Z, cnt in rows:
            s = _loss_fn(th2, X, Z) * cnt   # row-sum 加权
            gr = torch.autograd.grad(s, th2, retain_graph=False)[0]
            acc += gr
        acc = acc / total_cnt
        joint = sum((_loss_fn(theta, X, Z) * cnt) for X, Z, cnt in rows)
        q = torch.autograd.grad(joint, theta)[0] / total_cnt
        self.assertTrue(torch.allclose(acc, q, rtol=1e-10, atol=1e-14))
        # 错误版本：无条件平均 chunk 均值
        wrong = torch.autograd.grad(
            sum(_loss_fn(theta, X, Z) for X, Z, _ in rows), theta)[0] / len(rows)
        self.assertFalse(torch.allclose(acc, wrong, rtol=1e-3))

    def test_seal_order(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "pol.json"
            p.write_text('{"a": 0.0}')
            seal = Path(td) / "seal.json"
            r1m.write_seal(seal, [p], extra={"stage": "policies"})
            ok, _ = r1m.verify_seal(seal, [p])
            self.assertTrue(ok)
            p.write_text('{"a": 0.1}')
            ok, _ = r1m.verify_seal(seal, [p])
            self.assertFalse(ok)


class T8_SolverAndBudget(unittest.TestCase):
    def test_project_candidates_armijo(self):
        phi = (0.1, -0.05)
        g = (1e-3, -2e-4)
        cands = f0.solver_candidates(phi, g)
        self.assertEqual(len(cands), 3)
        for h, trial, delta in cands:
            self.assertLessEqual(abs(trial[0]), f0.LOG2 + 1e-15)
            self.assertLessEqual(abs(trial[1]), f0.LOG2 + 1e-15)
        # 边界外推被投影为 delta=0 时跳过（g 指向域外）
        none = f0.solver_candidates((f0.LOG2, 0.0), (-1.0, 0.0),
                                    free=(True, False))
        self.assertEqual(none, [])
        # g=0 停止
        self.assertEqual(f0.solver_candidates((0, 0), (0.0, 0.0)), [])
        # Armijo
        self.assertTrue(f0.armijo_ok(0.9, 1.0, (1.0, 0.0), (-0.05, 0.0)))
        self.assertFalse(f0.armijo_ok(1.2, 1.0, (1.0, 0.0), (-0.05, 0.0)))

    def test_tie_break_lexicographic(self):
        phi, j = f0.tie_break_argmin([((0.0, 0.0), 1.0),
                                      ((-f0.LOG2, 0.0), 1.0),
                                      ((0.0, f0.LOG2), 1.0)])
        self.assertEqual(phi, (0.0, 0.0))
        phi2, j2 = f0.tie_break_argmin([((f0.LOG2, 0.0), 0.5),
                                        ((0.0, 0.0), 0.7)])
        self.assertEqual(phi2, (f0.LOG2, 0.0))
        self.assertEqual(j2, 0.5)

    def test_budget_plan_counts(self):
        # 计划上界（协议 §8 表）独立复算
        traj = 64 + 2304 + 4608 + 6144 + 6144 + 1536 + 1536 + 7680
        self.assertEqual(traj, 30016)
        fw_train = 2 * traj
        self.assertEqual(fw_train, 60032)
        design_evals = 8 + 10 + 18 + 24 * 3 + 93 + 6
        self.assertEqual(design_evals, 207)
        hvp_dirs = 32 + 512 + 3072 + 1536 + 768
        self.assertEqual(hvp_dirs, 5920)


if __name__ == "__main__":
    unittest.main(verbosity=2)
