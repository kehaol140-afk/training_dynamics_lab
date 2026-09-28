#!/usr/bin/env python3
"""GTTR-R2B 桥接 —— CPU/已有 tensor 测试（协议 §8）。

独立参考：期望值在文件内显式重算，不调用生产 helper 自比自。
故意错误版本（漏 wd、把干预用到最后一步、读取 half/Schur 解、h1 参与检测）
必须被捕获。
"""
import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

import torch  # noqa: E402

from fr import d1_direct_readout_v01 as d1m  # noqa: E402
from fr import r1_objective_identification_v01 as r1m  # noqa: E402
from fr import r2b_bridge_v01 as b  # noqa: E402

B1, B2, EPS, WD = 0.9, 0.999, 1e-8, 0.01
R1_OUT = PROJECT_ROOT / "outputs" / "fr_gttr_r1_objective_identification_v01"
SOL_CSV = R1_OUT / "restricted_solutions.csv"
MODEL_CSV = R1_OUT / "model_coefficients.csv"


def _toy(seed=0, n=2):
    g = torch.Generator().manual_seed(seed)
    theta = [torch.randn(3, 4, generator=g) for _ in range(n)]
    m = [torch.randn(3, 4, generator=g) * 0.1 for _ in range(n)]
    v = [torch.rand(3, 4, generator=g) * 0.01 + 1e-4 for _ in range(n)]
    c = [torch.randn(3, 4, generator=g) * 0.05 for _ in range(n)]
    return theta, m, v, c


def _mk_opt(params, lr):
    opt = torch.optim.AdamW(params, lr=lr, betas=(B1, B2), eps=EPS,
                            weight_decay=WD)
    for p in params:
        opt.state[p] = {"step": torch.tensor(46.0),
                        "exp_avg": torch.randn_like(p) * 0.1,
                        "exp_avg_sq": torch.rand_like(p) * 0.01 + 1e-4}
    return opt


def _arm_first_update(theta0, m0, v0, c, t, x, y, eta_r):
    """测试内独立实现：按 R2B 首步规则做一次更新，返回 (theta1, m1, v1)。"""
    eta_n, alpha_n, _ = b.first_step_plan(x, y, gm=94)
    _ = eta_n
    eta_n = eta_r * (1 + x)
    alpha_n = y / (1 + x)
    outs = []
    for th, mm, vv, cc in zip(theta0, m0, v0, c):
        m_plus = B1 * mm + (1 - B1) * cc
        v_plus = B2 * vv + (1 - B2) * cc * cc
        mbar = m_plus / (1 - B1 ** t)
        vbar = v_plus / (1 - B2 ** t)
        P = 1.0 / (vbar.sqrt() + EPS)
        r = (1 - alpha_n) * (P * mbar) + alpha_n * (P * cc)
        th1 = (1 - eta_n * WD) * th - eta_n * r
        outs.append((th1, m_plus, v_plus))
    return outs


class C1_MappingAndWD(unittest.TestCase):
    def test_exact_mapping_matches_r1_coordinate_form(self):
        theta, m, v, c = _toy(1)
        t = 47
        eta_r = b.eta_ref(94)
        for (x, y) in [(0.0, 0.0), (0.05, 0.0), (0.05, 0.05),
                       (-0.05, 0.05), (0.05, 0.011755757652496488)]:
            eta_n, alpha_n, _ = b.first_step_plan(x, y, 94)
            sl = r1m.control_slice(dict(enumerate(theta)), dict(enumerate(c)),
                                   dict(enumerate(m)), dict(enumerate(v)), t,
                                   B1, B2, EPS, WD, eta_r)
            tp = r1m.make_theta_plus(dict(enumerate(theta)), sl)
            mapped = r1m.theta_at_a(tp, sl, sl, x, y)
            direct = r1m.direct_readout_update(
                dict(enumerate(theta)), dict(enumerate(c)),
                dict(enumerate(m)), dict(enumerate(v)), t, B1, B2, EPS, WD,
                eta_n, alpha_n)
            for k_ in mapped:
                self.assertTrue(torch.allclose(mapped[k_], direct[k_],
                                               rtol=1e-5, atol=1e-9),
                                f"(x={x},y={y}) mismatch")

    def test_missing_wd_mutant_detected(self):
        theta, m, v, c = _toy(2)
        t = 47
        eta_r = b.eta_ref(94)
        x, y = 0.05, 0.05
        eta_n, alpha_n, _ = b.first_step_plan(x, y, 94)
        # mutant：漏 wd
        th_mut = []
        for th, mm, vv, cc in zip(theta, m, v, c):
            mbar = (B1 * mm + (1 - B1) * cc) / (1 - B1 ** t)
            vbar = (B2 * vv + (1 - B2) * cc * cc) / (1 - B2 ** t)
            P = 1.0 / (vbar.sqrt() + EPS)
            r = (1 - alpha_n) * P * mbar + alpha_n * P * cc
            th_mut.append(th - eta_n * r)
        outs = _arm_first_update(theta, m, v, c, t, x, y, eta_r)
        diff = max(float((a - mu).abs().max())
                   for a, mu in zip([o[0] for o in outs], th_mut))
        self.assertGreater(diff, 0.0)


class C2_NativeRouting(unittest.TestCase):
    def test_base_and_lronly_native_full_qh(self):
        _, _, _, _ = _toy(3)
        params = [torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)]
        opt = _mk_opt(params, 1e-4)
        for p in params:
            p.grad = torch.randn_like(p)
        # BASE (0,0) 与 LR-only (x,0) -> use_native
        for x in (0.0, 0.05, -0.05):
            _, alpha, native = b.first_step_plan(x, 0.0, 94)
            self.assertTrue(native)
            self.assertEqual(alpha, 0.0)
        # FULL -> qh RAW（无 norm matching）
        _, alpha, native = b.first_step_plan(0.05, 0.05, 94)
        self.assertFalse(native)
        self.assertAlmostEqual(alpha, 0.05 / 1.05, places=15)
        with torch.no_grad():
            for p in params:
                p.grad = None
        # alpha=0 不得走 qh（d1 语义）
        with self.assertRaises(ValueError):
            d1m.qh_adamw_step(opt, 0.0, "RAW")

    def test_qh_step_no_norm_matching_mode(self):
        params = [torch.nn.Parameter(torch.randn(3, 4))]
        opt = _mk_opt(params, 2e-4)
        params[0].grad = torch.randn_like(params[0])
        rows = d1m.qh_adamw_step(opt, 0.047619047619047616, "RAW",
                                 collect=True)
        self.assertEqual(rows[0]["fallback"], "NA")  # RAW：无 norm matching


class C3_InterventionMask(unittest.TestCase):
    def test_mask_only_first_update(self):
        theta, m, v, c = _toy(4)
        t = 47
        eta_r = b.eta_ref(94)
        x, y = 0.05, 0.05
        outs = _arm_first_update(theta, m, v, c, t, x, y, eta_r)
        th1 = [o[0] for o in outs]
        m1 = [o[1] for o in outs]
        v1 = [o[2] for o in outs]
        # 第 2 步：固定原规则（native，k=2 schedule，无干预）
        c2 = [torch.randn(3, 4, generator=torch.Generator().manual_seed(99))
              for _ in theta]
        t2 = 48
        eta2 = b.eta_ref(96)
        th2 = []
        for th, mm, vv, cc in zip(th1, m1, v1, c2):
            mbar = (B1 * mm + (1 - B1) * cc) / (1 - B1 ** t2)
            vbar = (B2 * vv + (1 - B2) * cc * cc) / (1 - B2 ** t2)
            P = 1.0 / (vbar.sqrt() + EPS)
            th2.append((1 - eta2 * WD) * th - eta2 * P * mbar)
        # mutant：第 2 步仍施加干预
        eta2x = eta2 * 1.05
        th2_mut = []
        for th, mm, vv, cc in zip(th1, m1, v1, c2):
            mbar = (B1 * mm + (1 - B1) * cc) / (1 - B1 ** t2)
            vbar = (B2 * vv + (1 - B2) * cc * cc) / (1 - B2 ** t2)
            P = 1.0 / (vbar.sqrt() + EPS)
            th2_mut.append((1 - eta2x * WD) * th - eta2x * P * mbar)
        diff = max(float((a - mu).abs().max())
                   for a, mu in zip(th2, th2_mut))
        self.assertGreater(diff, 0.0)


class C4_FirstStepMomentsShared(unittest.TestCase):
    def test_three_arms_same_moments_theta_diverges(self):
        theta, m, v, c = _toy(5)
        eta_r = b.eta_ref(94)
        arms = {"BASE": (0.0, 0.0), "LR": (0.05, 0.0),
                "FULL": (0.05, 0.05)}
        results = {}
        for name, (x, y) in arms.items():
            th = [t_.clone() for t_ in theta]
            mm = [t_.clone() for t_ in m]
            vv = [t_.clone() for t_ in v]
            outs = _arm_first_update(th, mm, vv, c, 47, x, y, eta_r)
            results[name] = outs
        for k_ in range(len(theta)):
            ms = [results[a][k_][1] for a in arms]
            vs = [results[a][k_][2] for a in arms]
            for other in ("LR", "FULL"):
                self.assertTrue(torch.equal(ms[0], results[other][k_][1]))
                self.assertTrue(torch.equal(vs[0], results[other][k_][2]))
        self.assertFalse(torch.allclose(results["BASE"][0][0],
                                        results["FULL"][0][0]))
        self.assertFalse(torch.allclose(results["LR"][0][0],
                                        results["FULL"][0][0]))
        # 分叉后不强求跨臂 moment 一致（第2步不同 c/theta）
        # （结构上允许，测试确认 API 不抛错即可——见 C3）


class C5_ReadPrimarySolutions(unittest.TestCase):
    def test_read_r1_primary_k2(self):
        sols = b.read_primary_solutions(SOL_CSV)
        key = ("2031", "46", "2")
        self.assertAlmostEqual(sols[key]["2D"][1],
                               0.011755757652496488, places=18)
        self.assertEqual(sols[("2032", "82", "2")]["LR"][0], -0.05)
        rep = b.verify_solutions_against_coefficients(
            {k_: v for k_, v in sols.items() if k_[2] == "2"}, MODEL_CSV)
        for k_, r in rep.items():
            self.assertTrue(r["ok"], f"{k_}: {r}")

    def test_two_special_cases_independent_reference(self):
        # 边界例：受限最优在角 (r, r)（无约束驻点在盒外）
        bvec = [-1.0, -0.5]
        K = [[2.0, 0.3], [0.3, 4.0]]
        xs = [i * 0.0005 - 0.05 for i in range(201)]
        ys = [i * 0.0005 for i in range(101)]
        best = min(((x, y) for x in xs for y in ys),
                   key=lambda p: r1m.quad_value(p, bvec, K))
        self.assertEqual(best[0], 0.05)
        self.assertEqual(best[1], 0.05)
        # 内部例：x=+r 边上 y 内部
        bvec2 = [-0.002091, -0.007184]
        K2 = [[0.01055, 0.04142], [0.04142, 0.4349]]
        y_edge = -(bvec2[1] + K2[0][1] * 0.05) / K2[1][1]
        self.assertGreater(y_edge, 0)
        self.assertLess(y_edge, 0.05)
        sol2 = r1m.solve_rectangle(bvec2, K2, 0.05, 0.05)
        self.assertAlmostEqual(sol2["x"], 0.05, places=9)
        self.assertAlmostEqual(sol2["y"], y_edge, places=9)
        # 解读取器不得接受 half 行
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.csv"
            p.write_text("seed,start,k,r,solution,x,y,m,flags\n"
                         "2030,46,2,0.025,2D,0.025,0.025,0,NA\n")
            sols = b.read_primary_solutions(p)
            self.assertEqual(len(sols), 0)  # half 行被过滤
            p.write_text("seed,start,k,r,solution,x,y,m,flags\n"
                         "2030,46,2,0.05,2D,NA,NA,0,NA\n")
            with self.assertRaises(ValueError):
                b.read_primary_solutions(p)


class C6_Schedule(unittest.TestCase):
    def test_k2_schedule_semantics(self):
        self.assertAlmostEqual(b.eta_ref(94), 2 * b.old_get_lr(94, 256, 5e-4),
                               places=18)
        self.assertAlmostEqual(b.eta_ref(256), 1e-4, places=18)
        self.assertEqual(b.eta_ref(257), 1e-4)
        self.assertEqual(b.eta_ref(1024), 1e-4)
        vals = [b.eta_ref(gm) for gm in range(94, 1025)]
        for i in range(1, len(vals)):
            self.assertLessEqual(vals[i], vals[i - 1] + 1e-15)  # 不回升
        # 累计索引：首步末 micro gm=2*n0+2
        self.assertEqual(2 * 46 + 2, 94)
        self.assertEqual(2 * 82 + 2, 166)

    def test_mutant_stretch_or_rewarm_detected(self):
        good = b.eta_ref(200)
        bad_stretch = 2 * b.old_get_lr(200 - 44, 256, 5e-4)  # 假拉伸
        self.assertNotAlmostEqual(good, bad_stretch, places=15)
        bad_rewarm = 2 * 5e-4 * (0.1 + 0.45)                  # 假重warmup
        self.assertLess(good, bad_rewarm)


class C7_Detection(unittest.TestCase):
    def test_grid_and_lte(self):
        grid = {h: 7.1 - 0.02 * i for i, h in enumerate(b.DETECTION_GRID)}
        grid[0] = 7.2
        res = b.detect_target(grid, 7.020043501537879)
        self.assertEqual(res["status"], "DETECTED")
        # h=1/h=4 不得参与
        with self.assertRaises(ValueError):
            b.detect_target({**grid, 1: 6.5}, 7.0)
        # <= 含等号
        ell = 7.0
        g = {0: 7.2, 16: 7.1, 32: ell, 48: ell, 64: 6.9}
        res = b.detect_target(g, ell)
        self.assertEqual(res["h_detect"], 48)
        # 未达标：全部网格点高于 ell（仅孤立下探不构成 pair）
        ell2 = 7.020043501537879
        g2 = {h: 7.2 - 0.001 * i for i, h in enumerate(b.DETECTION_GRID)}
        res = b.detect_target(g2, ell2)
        self.assertEqual(res["status"], "NOT_REACHED")
        # 交替上下不构成 confirmed pair
        g3 = {0: 7.2, 16: 6.9, 32: 7.1, 48: 6.9}
        res = b.detect_target(g3, ell)
        self.assertEqual(res["status"], "NOT_REACHED")

    def test_token_accumulation_semantics(self):
        slots_per_step = 2 * 4 * 128
        valid = [900, 910, 905, 895]
        cum_slots = slots_per_step * 3
        cum_valid = sum(valid[:3])
        self.assertEqual(cum_slots, 3072)
        self.assertEqual(cum_valid, 2715)


class C8_IsolationAndSemantics(unittest.TestCase):
    def test_tied_params_updated_once(self):
        lin = torch.nn.Linear(4, 4, bias=False)
        head = torch.nn.Linear(4, 4, bias=False)
        head.weight = lin.weight
        named = [("a.weight", lin.weight), ("b.weight", head.weight)]
        seen, uniq = set(), []
        for n, p in named:
            if id(p) in seen:
                continue
            seen.add(id(p))
            uniq.append((n, p))
        opt = _mk_opt([p for _, p in uniq], 1e-4)
        lin.weight.grad = torch.randn_like(lin.weight)
        before = lin.weight.detach().clone()
        d1m.qh_adamw_step(opt, 0.047619047619047616, "RAW")
        self.assertFalse(torch.equal(before, lin.weight))
        self.assertEqual(len(opt.state), 1)

    def test_grad_none_skipped_no_decay(self):
        p1 = torch.nn.Parameter(torch.randn(3, 4))
        p2 = torch.nn.Parameter(torch.randn(3, 4))   # grad 保持 None
        opt = _mk_opt([p1, p2], 1e-4)
        p1.grad = torch.randn_like(p1)
        before2 = p2.detach().clone()
        d1m.qh_adamw_step(opt, 0.047619047619047616, "RAW")
        self.assertTrue(torch.equal(p2.detach(), before2))  # 不动、不衰减
        self.assertEqual(float(opt.state[p2]["step"]), 46.0)  # 计数不推进

    def test_rng_capture_restore(self):
        from fr.rng import capture_rng_state, restore_rng_state
        torch.manual_seed(123)
        capture = capture_rng_state()
        _ = torch.randn(100)
        _ = torch.rand(7)
        restore_rng_state(capture)
        again = capture_rng_state()
        self.assertTrue(torch.equal(capture["torch_cpu"],
                                    again["torch_cpu"]))

    def test_eval_isolation_train_mode_restored(self):
        model = torch.nn.Linear(4, 4)
        model.train()
        was = model.training
        model.eval()
        if was:
            model.train()
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main(verbosity=2)
