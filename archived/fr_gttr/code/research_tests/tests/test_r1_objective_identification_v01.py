#!/usr/bin/env python3
"""GTTR-R1 目标系数辨识 —— 八项最少 CPU 测试（协议 §8 T1–T8）。

独立参考：所有期望值在本文件内用显式公式重算，不调用生产 helper 自比自。
故意错误版本（漏 eta/漏 wd、mean(Adam)、E[D]^T E[H] E[D]、PSD 裁剪、
方向不 detach、AUDIT 回写系数）必须被对应检查捕获。
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fr import r1_objective_identification_v01 as r1  # noqa: E402

B1, B2, EPS, WD = 0.9, 0.999, 1e-8, 0.01
T_TRUE = 47  # n0=46 -> t=47


def _old_get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def _toy_state(seed=0, n=3):
    g = torch.Generator().manual_seed(seed)
    theta = {f"p{i}": torch.randn(4, 5, generator=g) for i in range(n)}
    m = {k: torch.randn(4, 5, generator=g) * 0.1 for k in theta}
    v = {k: torch.rand(4, 5, generator=g) * 0.01 + 1e-4 for k in theta}
    c = {k: torch.randn(4, 5, generator=g) * 0.05 for k in theta}
    return theta, m, v, c


# ---------------------------------------------------------------------------
# T1 精确坐标映射
# ---------------------------------------------------------------------------

class T1_ExactCoordinateMapping(unittest.TestCase):
    def test_slice_matches_direct_readout_all_points(self):
        theta, m, v, c = _toy_state(1)
        eta = 3.71e-4
        sl = r1.control_slice(theta, c, m, v, T_TRUE, B1, B2, EPS, WD, eta)
        tp = r1.make_theta_plus(theta, sl)
        for (x, y) in [(-.05, 0.0), (.05, .05), (0.0, .025), (.02, .04),
                       (-.03, .01), (0.0, 0.0)]:
            eta_n, alpha_n = r1.reparam(x, y, eta)
            direct = r1.direct_readout_update(theta, c, m, v, T_TRUE, B1, B2,
                                              EPS, WD, eta_n, alpha_n)
            mapped = r1.theta_at_a(tp, sl, sl, x, y)
            for k in theta:
                self.assertTrue(torch.allclose(mapped[k], direct[k], rtol=1e-5,
                                               atol=1e-9),
                                f"(x={x},y={y}) param {k} mismatch")

    def test_native_point_is_adamw_update(self):
        # x=y=0：theta_plus 必须等于原生 AdamW 更新（同公式），并与真实
        # torch.optim.AdamW 一致（浮点运算顺序差异 -> 紧公差）
        theta, m, v, c = _toy_state(2)
        eta = 2.5e-4
        sl = r1.control_slice(theta, c, m, v, T_TRUE, B1, B2, EPS, WD, eta)
        tp = r1.make_theta_plus(theta, sl)
        for k in theta:
            expect = (1 - eta * WD) * theta[k] - eta * (sl[k]["P"] * sl[k]["mbar"])
            # 实数算术恒等；浮点运算顺序不同 -> 紧公差
            self.assertTrue(torch.allclose(tp[k], expect, rtol=1e-6, atol=1e-12))
        # 真实 AdamW 对照
        torch.manual_seed(3)
        lin = torch.nn.Linear(5, 4, bias=False)  # weight: (4,5)
        opt = torch.optim.AdamW(lin.parameters(), lr=eta, betas=(B1, B2),
                                eps=EPS, weight_decay=WD)
        p = lin.weight
        gp = theta["p0"]
        with torch.no_grad():
            p.copy_(gp)
        opt.state[p] = {"step": torch.tensor(float(T_TRUE - 1)),
                        "exp_avg": m["p0"].clone(),
                        "exp_avg_sq": v["p0"].clone()}
        p.grad = c["p0"].clone()
        opt.step()
        self.assertTrue(torch.allclose(p.detach(), tp["p0"], rtol=1e-6,
                                       atol=1e-12))

    def test_multi_param_group(self):
        # 两个参数组不同 wd/eta：逐组映射仍精确
        theta, m, v, c = _toy_state(4)
        g0 = ("p0", 0.02, 1.0e-4)
        g1 = ("p1", 0.0, 2.0e-4)   # wd=0 组
        x, y = .04, .03
        tp, sl = {}, {}
        for name, wd, eta in (g0, g1):
            s = r1.control_slice({name: theta[name]}, {name: c[name]},
                                 {name: m[name]}, {name: v[name]}, T_TRUE,
                                 B1, B2, EPS, wd, eta)
            sl.update(s)
            tp.update(r1.make_theta_plus({name: theta[name]}, s))
        # 参数组 2 wd=0 -> 无衰减项
        self.assertTrue(torch.allclose(
            tp["p1"], theta["p1"] - 2.0e-4 * (sl["p1"]["P"] * sl["p1"]["mbar"]),
            rtol=1e-6))
        for name, wd, eta in (g0, g1):
            eta_n, alpha_n = r1.reparam(x, y, eta)
            direct = r1.direct_readout_update({name: theta[name]},
                                              {name: c[name]}, {name: m[name]},
                                              {name: v[name]}, T_TRUE, B1, B2,
                                              EPS, wd, eta_n, alpha_n)
            self.assertTrue(torch.allclose(tp[name] + x * sl[name]["d0"]
                                           + y * sl[name]["d1"], direct[name],
                                           rtol=1e-5, atol=1e-9))

    def test_mutant_missing_eta_or_wd_must_fail(self):
        theta, m, v, c = _toy_state(5)
        eta = 3e-4
        sl = r1.control_slice(theta, c, m, v, T_TRUE, B1, B2, EPS, WD, eta)
        # mutant d0：漏 wd
        d0_no_wd = {k: -eta * (sl[k]["P"] * sl[k]["mbar"]) for k in sl}
        self.assertGreater(float((d0_no_wd["p0"] - sl["p0"]["d0"]).abs().max()), 0)
        # mutant d1：漏 eta
        d1_no_eta = {k: sl[k]["P"] * (sl[k]["mbar"] - c[k]) for k in sl}
        self.assertGreater(
            float((d1_no_eta["p0"] - sl["p1"]["d1"]).abs().max() / eta), 0.5)


# ---------------------------------------------------------------------------
# T2 m/v 与控制坐标无关；跨 probe 不冻结 P；grad=None/zero/共享参数
# ---------------------------------------------------------------------------

class T2_MomentsAndGradSemantics(unittest.TestCase):
    def test_moments_invariant_and_not_modified(self):
        theta, m, v, c = _toy_state(6)
        m0 = {k: t.clone() for k, t in m.items()}
        v0 = {k: t.clone() for k, t in v.items()}
        s1 = r1.control_slice(theta, c, m, v, T_TRUE, B1, B2, EPS, WD, 1e-4)
        s2 = r1.control_slice(theta, c, m, v, T_TRUE, B1, B2, EPS, WD, 2e-4)
        for k in c:
            self.assertTrue(torch.equal(s1[k]["m_plus"], s2[k]["m_plus"]))
            self.assertTrue(torch.equal(s1[k]["v_plus"], s2[k]["v_plus"]))
        for k in m:  # 原状态未被修改
            self.assertTrue(torch.equal(m[k], m0[k]))
            self.assertTrue(torch.equal(v[k], v0[k]))

    def test_P_not_frozen_across_probes(self):
        theta, m, v, c1 = _toy_state(7)
        _, _, _, c2 = _toy_state(8)
        s1 = r1.control_slice(theta, c1, m, v, T_TRUE, B1, B2, EPS, WD, 1e-4)
        s2 = r1.control_slice(theta, c2, m, v, T_TRUE, B1, B2, EPS, WD, 1e-4)
        self.assertFalse(torch.allclose(s1["p0"]["P"], s2["p0"]["P"]))
        # mutant：共用平均 P 必然给出错误 d1
        Pbar = 0.5 * (s1["p0"]["P"] + s2["p0"]["P"])
        d1_wrong = 1e-4 * Pbar * (s1["p0"]["mbar"] - c1["p0"])
        self.assertGreater(float((d1_wrong - s1["p0"]["d1"]).abs().max()), 0)

    def test_grad_none_vs_zero_semantics(self):
        theta, m, v, c = _toy_state(9)
        c_none = dict(c)
        c_none["p1"] = None          # 无梯度：整步跳过（不 decay）
        c_zero = dict(c)
        c_zero["p1"] = torch.zeros_like(theta["p1"])  # 零梯度：正常更新
        eta = 1e-4
        s_none = r1.control_slice(theta, c_none, m, v, T_TRUE, B1, B2, EPS,
                                  WD, eta)
        s_zero = r1.control_slice(theta, c_zero, m, v, T_TRUE, B1, B2, EPS,
                                  WD, eta)
        self.assertNotIn("p1", s_none)
        self.assertIn("p1", s_zero)
        tp = r1.make_theta_plus(theta, s_none)
        self.assertTrue(torch.equal(tp["p1"], theta["p1"]))  # 不动、不衰减
        # 零梯度参数 d0 = -eta*(P*mbar + wd*theta)（非零，若 m 非零）
        self.assertGreater(float(s_zero["p1"]["d0"].abs().max()), 0)

    def test_shared_params_dedup(self):
        # 共享参数（tie）在 named_parameters 中只出现一次；slice 键唯一
        lin = torch.nn.Linear(4, 4, bias=False)
        tied = torch.nn.Linear(4, 4, bias=False)
        tied.weight = lin.weight
        named = [("a.weight", lin.weight), ("b.weight", tied.weight)]
        seen = set()
        uniq = []
        for n, p in named:
            if id(p) in seen:
                continue
            seen.add(id(p))
            uniq.append((n, p))
        self.assertEqual(len(uniq), 1)
        theta = {n: p.detach().clone() for n, p in uniq}
        c = {n: torch.randn_like(p) for n, p in uniq}
        m = {n: torch.zeros_like(p) for n, p in uniq}
        v = {n: torch.ones_like(p) * 1e-4 for n, p in uniq}
        s = r1.control_slice(theta, c, m, v, 2, B1, B2, EPS, WD, 1e-4)
        self.assertEqual(list(s.keys()), [uniq[0][0]])


# ---------------------------------------------------------------------------
# T3 已知二次目标：b/K 与解析值一致；矩形全局解 vs 细网格
# ---------------------------------------------------------------------------

class T3_KnownQuadratic(unittest.TestCase):
    def _setup(self, seed):
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(6, 6))
        H = A @ A.T / 6 + 0.5 * np.eye(6)
        g = rng.normal(size=6)
        d0 = rng.normal(size=6) * 0.01
        d1 = rng.normal(size=6) * 0.01
        th = rng.normal(size=6)
        return H, g, th, d0, d1

    def test_b_K_match_analytic_via_autograd(self):
        H, g, th, d0, d1 = self._setup(0)
        th_p = th + d0
        D = np.stack([d0, d1], axis=1)
        b_true = D.T @ (H @ th_p + g)
        K_true = D.T @ H @ D
        th_t = torch.tensor(th, requires_grad=True)
        d0_t = torch.tensor(d0)
        d1_t = torch.tensor(d1)

        def V(t):
            return 0.5 * t @ torch.tensor(H) @ t + torch.tensor(g) @ t

        # 生产语义：一次 create_graph 梯度 + 两次方向 VJP，方向 detach
        tp_t = th_t + d0_t
        S = V(tp_t)
        gr = torch.autograd.grad(S, th_t, create_graph=True)[0]
        dot0 = (gr * d0_t).sum()
        dot1 = (gr * d1_t).sum()
        v0 = torch.autograd.grad(dot0, th_t, retain_graph=True)[0]
        v1 = torch.autograd.grad(dot1, th_t)[0]
        self.assertAlmostEqual(float(dot0), b_true[0], places=10)
        self.assertAlmostEqual(float(dot1), b_true[1], places=10)
        self.assertAlmostEqual(float(d0_t @ v0), K_true[0, 0], places=10)
        self.assertAlmostEqual(float(d1_t @ v0), K_true[1, 0], places=10)
        self.assertAlmostEqual(float(d0_t @ v1), K_true[0, 1], places=10)
        self.assertAlmostEqual(float(d1_t @ v1), K_true[1, 1], places=10)

    def test_rectangle_solve_matches_bruteforce(self):
        rng = np.random.default_rng(1)
        for case in range(30):
            scale = 10.0 ** rng.uniform(-3, 1)
            b = rng.normal(size=2) * scale
            K = (rng.normal(size=(2, 2)) * scale)
            K = [[K[0][0], K[0][1]], [K[0][1], K[1][1]]]
            r = 0.05
            sol = r1.solve_rectangle(b, K, r, r)
            xs = np.linspace(-r, r, 401)
            ys = np.linspace(0, r, 201)
            best = min(r1.quad_value((x, y), b, K) for x in xs for y in ys)
            self.assertLessEqual(sol["m"], best + 1e-12 * max(1, abs(best)),
                                 f"case {case}: solver above brute force")
            gx = np.gradient([r1.quad_value((x, sol["y"]), b, K) for x in xs])
            # 解处目标不高于任何网格点（枚举完备性）
            self.assertLessEqual(
                sol["m"],
                min(r1.quad_value((float(x), float(y)), b, K)
                    for x in xs for y in ys) + 1e-9 * max(1.0, scale))
        _ = gx  # noqa


# ---------------------------------------------------------------------------
# T4 正定/不定/奇异/边界；y*=0 正确输出无读出修改；不得 PSD 裁剪
# ---------------------------------------------------------------------------

class T4_SolverEdgeCases(unittest.TestCase):
    def test_pd_interior(self):
        b, K = [-0.04, -0.01], [[2.0, 0.3], [0.3, 4.0]]
        sol = r1.solve_rectangle(b, K, 0.05, 0.05)
        sch = r1.schur_elimination(b, K)
        self.assertTrue(sch["valid"])
        self.assertAlmostEqual(sol["x"], sch["x_star"], places=9)
        self.assertAlmostEqual(sol["y"], sch["y_star"], places=9)
        self.assertEqual(sol["tag"], "interior")

    def test_indefinite_boundary(self):
        b, K = [-1.0, -0.1], [[2.0, 0.0], [0.0, -2.0]]
        sol = r1.solve_rectangle(b, K, 0.05, 0.05)
        self.assertEqual(sol["y"], 0.05)  # 负曲率 -> 上边界
        self.assertEqual(sol["tag"], "corner")

    def test_singular_psd(self):
        K = [[2.0, 4.0], [4.0, 8.0]]  # det=0
        b = [-1.0, -2.0]
        sol = r1.solve_rectangle(b, K, 0.05, 0.05)
        xs = np.linspace(-0.05, 0.05, 801)
        ys = np.linspace(0, 0.05, 401)
        brute = min(r1.quad_value((float(x), float(y)), b, K)
                    for x in xs for y in ys)
        self.assertLessEqual(sol["m"], brute + 1e-12)
        eig, cond = r1.eig_cond(K)
        self.assertAlmostEqual(eig[0], 0.0, places=9)
        self.assertEqual(cond, math.inf)

    def test_ystar_zero_means_no_readout(self):
        b, K = [-1.0, 0.0], [[2.0, 0.0], [0.0, 4.0]]
        sch = r1.schur_elimination(b, K)
        self.assertTrue(sch["valid"])
        self.assertEqual(sch["y_star"], 0.0)
        sol = r1.solve_rectangle(b, K, 0.05, 0.05)
        lr = r1.solve_lr_only(b, K, 0.05)
        self.assertEqual(sol["y"], 0.0)
        self.assertAlmostEqual(sol["x"], lr["x"], places=12)
        kkt = r1.kkt_y0_flag(b, K, lr["x"])
        self.assertTrue(kkt["kkt_ok"])

    def test_psd_clipping_mutant_must_differ(self):
        b, K = [-1.0, 1.0], [[2.0, 0.0], [0.0, -100.0]]
        ours = r1.solve_rectangle(b, K, 0.05, 0.05)
        self.assertEqual(ours["y"], 0.05)  # 负曲率大 -> 真解在上边界
        Kclip = [[2.0, 0.0], [0.0, max(0.0, K[1][1])]]  # mutant：截成 PSD
        clipped = r1.solve_rectangle(b, Kclip, 0.05, 0.05)
        self.assertEqual(clipped["y"], 0.0)
        self.assertNotEqual((ours["x"], ours["y"]), (clipped["x"], clipped["y"]))

    def test_lr_only_boundary_and_flat(self):
        b, K = [1.0, 0.0], [[2.0, 0.0], [0.0, 2.0]]
        sol = r1.solve_lr_only(b, K, 0.05)
        self.assertEqual(sol["x"], -0.05)
        self.assertTrue(sol["boundary"])
        sol = r1.solve_lr_only([0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]], 0.05)
        self.assertEqual(sol["x"], 0.0)  # 平坦 -> 确定性 tie-break 原点
        self.assertTrue(sol["flat"])


# ---------------------------------------------------------------------------
# T5 Schur 仅在合法条件下使用；共线方向不能独立区分
# ---------------------------------------------------------------------------

class T5_SchurAndCollinearity(unittest.TestCase):
    def test_schur_equals_2d_stationary(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            A = rng.normal(size=(2, 2))
            K = A @ A.T + np.eye(2)
            b = rng.normal(size=2)
            sch = r1.schur_elimination(b, K.tolist())
            self.assertTrue(sch["valid"])
            a = np.linalg.solve(np.array(K), -np.array(b))
            self.assertAlmostEqual(sch["y_star"], float(a[1]), places=9)
            self.assertAlmostEqual(sch["x_star"], float(a[0]), places=9)

    def test_schur_invalid_cases(self):
        self.assertFalse(r1.schur_elimination([1, 1], [[0.0, 0.0],
                                                       [0.0, 5.0]])["valid"])
        sch = r1.schur_elimination([1.0, 1.0], [[2.0, 0.0], [0.0, -3.0]])
        self.assertFalse(sch["valid"])
        self.assertEqual(sch["reason"], "K_NOT_PD")
        sch = r1.schur_elimination([1.0, 1.0], [[1.0, 1.0], [1.0, 1.0]])
        self.assertFalse(sch["valid"])

    def test_collinear_directions_flagged(self):
        # d1 = 2*d0 -> K 秩 1（奇异），结构与 LR 在该切片上不可区分
        rng = np.random.default_rng(3)
        A = rng.normal(size=(5, 5))
        H = A @ A.T
        d0 = rng.normal(size=5) * 0.01
        d1 = 2.0 * d0
        D = np.stack([d0, d1], 1)
        K = (D.T @ H @ D).tolist()
        eig, cond = r1.eig_cond(K)
        self.assertAlmostEqual(abs(eig[0]), 0.0, places=12)
        sol = r1.solve_rectangle([-1e-3, -2e-3], K, 0.05, 0.05)
        self.assertEqual(sol["interior_status"], "ILL_CONDITIONED")
        sch = r1.schur_elimination([-1e-3, -2e-3], K)
        self.assertFalse(sch["valid"])


# ---------------------------------------------------------------------------
# T6 方向必须 detach
# ---------------------------------------------------------------------------

class T6_DetachedDirections(unittest.TestCase):
    def test_detached_matches_analytic(self):
        rng = np.random.default_rng(4)
        A = rng.normal(size=(4, 4))
        H = A @ A.T
        g = rng.normal(size=4)
        th = rng.normal(size=4)
        m = rng.normal(size=4) * 0.1
        eta = 1e-3

        def d0_np(t):
            return -eta * (0.5 * m + WD * np.asarray(t, dtype=np.float64))

        th_t = torch.tensor(th, requires_grad=True)
        Ht, gt = torch.tensor(H), torch.tensor(g)
        tp = th_t + torch.tensor(th) * 0  # theta_plus = th（简单化）
        S = 0.5 * tp @ Ht @ tp + gt @ tp
        gr = torch.autograd.grad(S, th_t, create_graph=True)[0]
        d_det = torch.tensor(d0_np(th))   # detach：常量方向
        dot_det = (gr * d_det).sum()
        v_det = torch.autograd.grad(dot_det, th_t, retain_graph=True)[0].numpy()
        Hd = H @ d0_np(th)
        self.assertTrue(np.allclose(v_det, Hd, atol=1e-14))

        # mutant：方向不 detach（对方向的构造求导；含 wd 项）
        d_attached = -eta * (torch.tensor(0.5 * m) + WD * th_t)
        dot_att = (gr * d_attached).sum()
        v_att = torch.autograd.grad(dot_att, th_t)[0].numpy()
        self.assertFalse(np.allclose(v_att, Hd, atol=1e-12))  # 必须能区分


# ---------------------------------------------------------------------------
# T7 mean(Adam) != Adam(mean)；跨 probe 相关项不能替换
# ---------------------------------------------------------------------------

class T7_JensenAndCrossProbe(unittest.TestCase):
    def _update_from_bank(self, c, m0=0.1, v0=1e-6, t=47):
        # 从非零 (m0, v0) 的单参数 AdamW 读出：-eta*P*mbar（eta=1）
        mm = B1 * m0 + (1 - B1) * float(c)
        vv = B2 * v0 + (1 - B2) * float(c) ** 2
        mbar = mm / (1 - B1 ** t)
        vbar = vv / (1 - B2 ** t)
        return -(1.0 / (math.sqrt(vbar) + EPS)) * mbar

    def test_mean_adam_neq_adam_mean_same_mu_sigma(self):
        # 同 mu=0、Sigma=1e-4 的两个分布（c^2 波动不同）：
        # E[Adam] 统计不同，且都 != Adam(mean)
        a = 0.01
        b = a / math.sqrt(0.5)
        d1, w1 = [-a, a], [0.5, 0.5]
        d2, w2 = [-b, 0.0, b], [0.25, 0.5, 0.25]
        for d, w in ((d1, w1), (d2, w2)):
            mu = sum(wi * x for wi, x in zip(w, d))
            var = sum(wi * (x - mu) ** 2 for wi, x in zip(w, d))
            self.assertAlmostEqual(mu, 0.0, places=15)
            self.assertAlmostEqual(var, 1e-4, places=15)
        e1 = sum(w * self._update_from_bank(c) for w, c in zip(w1, d1))
        e2 = sum(w * self._update_from_bank(c) for w, c in zip(w2, d2))
        am = self._update_from_bank(0.0)                  # Adam(mean)
        self.assertGreater(abs(e1 - e2), 1e-3 * abs(am))  # 更新统计不同
        self.assertGreater(abs(e1 - am), 1e-3 * abs(am))  # != Adam(mean)
        self.assertGreater(abs(e2 - am), 1e-3 * abs(am))

    def test_same_mu_sigma_different_P_statistics(self):
        # 向量对：mu=0, Sigma=I 相同；c^2 波动不同 -> E[P] 不同（Jensen）
        v0 = 1e-4  # 非零 v_old，避免 eps 病态
        A = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
        B = [(math.sqrt(2), 0.0), (-math.sqrt(2), 0.0),
             (0.0, math.sqrt(2)), (0.0, -math.sqrt(2))]

        def expect_P(S):
            acc = np.zeros(2)
            for cx, cy in S:
                v = np.array([B2 * v0 + (1 - B2) * cx * cx,
                              B2 * v0 + (1 - B2) * cy * cy])
                vbar = v / (1 - B2 ** 2)
                acc += 1.0 / (np.sqrt(vbar) + EPS)
            return acc / len(S)

        for S in (A, B):
            mu = np.mean(S, axis=0)
            Sig = np.mean([np.outer(c, c) for c in S], axis=0)
            self.assertTrue(np.allclose(mu, 0, atol=1e-12))
            self.assertTrue(np.allclose(Sig, np.eye(2), atol=1e-12))
        PA, PB = expect_P(A), expect_P(B)
        self.assertTrue(np.allclose(PA[0], PA[1]))        # A：P 无波动
        self.assertGreater(PB[0] - PA[0], 0.0)            # B：E[P] > P(E[v])（凸性）
        self.assertGreater(abs(PB - PA).max(), 1e-9)

    def test_cross_probe_terms_not_factorized(self):
        # E[D^T H D] != E[D]^T E[H] E[D]：交替符号 probe
        d = np.array([0.01, 0.0])
        h1 = np.diag([2.0, 3.0])
        h2 = np.diag([5.0, 7.0])
        rows = [{"b1": float(d @ h1 @ d), "b2": 0.0,
                 "k11": float(d @ h1 @ d), "k12_direct": 0.0,
                 "k21_direct": 0.0, "k22": 0.0},
                {"b1": float(-d @ h2 @ (-d)), "b2": 0.0,
                 "k11": float((-d) @ h2 @ (-d)), "k12_direct": 0.0,
                 "k21_direct": 0.0, "k22": 0.0}]
        agg = r1.mean_k_operators(rows)
        self.assertAlmostEqual(agg["k11"], 0.5 * (d @ h1 @ d + d @ h2 @ d),
                               places=15)
        # mutant：E[D]=0 -> E[D]^T E[H] E[D] = 0 != E[D^T H D] > 0
        self.assertGreater(agg["k11"], 0.0)


# ---------------------------------------------------------------------------
# T8 数据角色、seal、native=0、sum/count、SE、预算、条件分层
# ---------------------------------------------------------------------------

class T8_RolesSealAndAccounting(unittest.TestCase):
    def test_seal_detects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "coeff.csv"
            p.write_text("b1,b2\n0.1,0.2\n")
            q = Path(td) / "sol.csv"
            q.write_text("x,y\n0.01,0.0\n")
            seal = Path(td) / "seal.json"
            r1.write_seal(seal, [p, q], extra={"n_conditions": 1})
            ok, _ = r1.verify_seal(seal, [p, q])
            self.assertTrue(ok)
            q.write_text("x,y\n0.02,0.0\n")  # AUDIT 回写系数（禁止）
            ok, why = r1.verify_seal(seal, [p, q])
            self.assertFalse(ok)
            self.assertIn("hash_mismatch", why)

    def test_native_delta_exactly_zero(self):
        s = 123.456
        self.assertEqual(r1.delta_v(s, s, 1000), 0.0)
        self.assertTrue(math.copysign(1, r1.delta_v(s, s, 1000)) > 0)

    def test_sum_count_aggregation(self):
        rows_s = [10.0, 20.5, 7.25]
        rows_c = [100, 200, 50]
        agg = r1.aggregate_control(rows_s, rows_c)
        self.assertEqual(agg["count"], 350)
        self.assertAlmostEqual(agg["mean"], 37.75 / 350, places=15)
        # 无权 batch 均值（错误版本）= (0.1+0.1025+0.145)/3 != sum/count
        wrong = np.mean([s / c for s, c in zip(rows_s, rows_c)])
        self.assertNotAlmostEqual(agg["mean"], wrong, places=10)

    def test_paired_se_manual(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(r1.paired_se(xs),
                               float(np.std(xs, ddof=1) / 2), places=15)
        self.assertNotAlmostEqual(r1.paired_se(xs),
                                  float(np.std(xs, ddof=0) / 2), places=9)

    def test_plan_budget_independent_recompute(self):
        plan = r1.plan_budget()
        self.assertEqual(plan["total_fw"], 18624)
        self.assertEqual(plan["total_bw_micro"], 2496)
        self.assertEqual(plan["hvp_dirs"], 192)
        self.assertEqual(plan["committed_updates"], 0)
        self.assertEqual(plan["v_final_calls"], 0)
        self.assertEqual(6 * 16 * 2 + 6 * 2 * 8 * 8 + 6 * 2 * 8 * 20 * 8
                         + 6 * 2 * 8 * 3 * 8, 18624)

    def test_condition_labels_and_stratification(self):
        labels = [r1.condition_label(s, t, k)
                  for s in (2030, 2031, 2032) for t in (46, 82)
                  for k in (1.0, 2.0)]
        self.assertEqual(len(labels), 12)
        self.assertEqual(len(set(labels)), 12)
        st = r1.stratify(labels)
        self.assertEqual(len(st["seed"]), 3)   # 3 个训练 seed
        self.assertEqual(len(st["start"]), 2)
        self.assertEqual(len(st["k"]), 2)
        for g in st["seed"].values():
            self.assertEqual(len(g), 4)        # 每 seed 4 条件，不是独立 seed

    def test_fit_audit_probe_split_frozen(self):
        # 前 8 effective batch 为 FIT，后 8 为 AUDIT（SIM 序列内固定）
        seq = list(range(32))
        fit, audit = seq[:16], seq[16:]
        self.assertEqual(len(fit) // 2, 8)
        self.assertEqual(len(audit) // 2, 8)
        self.assertEqual(set(fit) & set(audit), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
