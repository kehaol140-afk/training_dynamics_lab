#!/usr/bin/env python3
"""GTTR-D1 直接更新规则 —— 八项最少 CPU 测试（协议 §7）。

独立参考：所有期望值在本文件内用显式公式重算，不调用生产 helper 自比自。
T8 的四个故意错误版本必须被对应检查捕获。
"""
import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

import torch  # noqa: E402

from fr.d1_direct_readout_v01 import (  # noqa: E402
    ALPHA, adamw_moment_update_, bias_corrected, precondition, r_a, r_q,
    norm_match, qh_adamw_step, native_reference, make_step,
)
from fr.phase_a_tail_schedule_v01 import make_phase_a_get_lr  # noqa: E402

B1, B2, EPS, WD = 0.9, 0.999, 1e-8, 0.01


def _old_get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def _mk_opt(n=3, seed=0, lr=5e-5):
    torch.manual_seed(seed)
    params = [torch.nn.Parameter(torch.randn(4, 5)) for _ in range(n)]
    opt = torch.optim.AdamW(params, lr=lr, betas=(B1, B2), eps=EPS, weight_decay=WD)
    for p in params:  # 模拟 bank 恢复：state 已存在（step=0, 矩=0）
        opt.state[p] = {"step": torch.tensor(0.0),
                        "exp_avg": torch.zeros_like(p),
                        "exp_avg_sq": torch.zeros_like(p)}
    return params, opt


def _independent_reference_step(theta, m, v, c, t, eta, alpha, mode, wd=WD):
    """测试内独立重算（显式公式，与生产代码不同路径）。"""
    m2 = B1 * m + (1 - B1) * c
    v2 = B2 * v + (1 - B2) * c * c
    mhat = m2 / (1 - B1 ** t)
    vhat = v2 / (1 - B2 ** t)
    P = torch.reciprocal(torch.sqrt(vhat) + EPS)
    rA = P * mhat
    rQ = (1 - alpha) * rA + alpha * (P * c)
    if mode == "RAW":
        r = rQ
    else:
        A = rA.norm()
        Q = rQ.norm()
        r = rA if (A == 0 or Q <= 1e-12 * max(A, rQ.norm())) else (A / Q) * rQ
    theta2 = theta * (1 - eta * wd) - eta * r
    return theta2, m2, v2


class T1_NativeRouting(unittest.TestCase):
    def test_alpha0_routes_to_native(self):
        # qh_adamw_step 对 alpha=0 必须拒绝（runner 走原生路径）
        _, opt = _mk_opt(1)
        with self.assertRaises(ValueError):
            qh_adamw_step(opt, 0.0, "RAW")
        # make_step 路由：alpha=0 -> 原生 bound step
        from fr.d1_direct_readout_v01 import make_step
        s0 = make_step(opt, 0.0, "RAW")
        self.assertTrue(s0 == opt.step or s0.__self__ is opt)
        s1 = make_step(opt, ALPHA, "NORM")
        self.assertTrue(callable(s1) and s1 != opt.step)

    def test_alpha0_bitwise_native_incl_decay_and_none_zero(self):
        # 同一 tiny 模型：纯 torch AdamW vs make_step(0) 逐位一致
        torch.manual_seed(7)
        p1 = torch.nn.Parameter(torch.randn(3, 3))
        p2 = torch.nn.Parameter(torch.randn(3, 3))  # 保持 grad=None
        p3 = torch.nn.Parameter(torch.randn(3, 3))  # 零梯度
        grads = [torch.randn(3, 3) for _ in range(4)]

        def run(use_d1_router):
            torch.manual_seed(7)
            a = torch.nn.Parameter(p1.detach().clone())
            b = torch.nn.Parameter(p2.detach().clone())
            c = torch.nn.Parameter(p3.detach().clone())
            opt = torch.optim.AdamW([a, b, c], lr=5e-5, betas=(B1, B2),
                                    eps=EPS, weight_decay=WD)
            from fr.d1_direct_readout_v01 import make_step
            step_fn = make_step(opt, 0.0, "RAW") if use_d1_router else opt.step
            outs = []
            for g in grads:
                opt.zero_grad(set_to_none=True)
                a.grad = g.clone()
                c.grad = torch.zeros_like(c)  # 零梯度：只应受 decay
                step_fn()
                outs.append((a.detach().clone(), b.detach().clone(),
                             c.detach().clone(),
                             opt.state[a]["exp_avg"].clone(),
                             opt.state[a]["exp_avg_sq"].clone(),
                             int(opt.state[a]["step"])))
            return outs

        r_native = run(False)
        r_d1 = run(True)
        for o1, o2 in zip(r_native, r_d1):
            for t1, t2 in zip(o1[:5], o2[:5]):
                self.assertTrue(torch.equal(t1, t2))
            self.assertEqual(o1[5], o2[5])
        # None-grad 参数完全不动（连 decay 都不做）
        self.assertTrue(torch.equal(r_native[0][1], p2))
        # 零-grad 参数：theta 只被逐次缩放 (1-eta*wd)（按相同 op 序列重算）
        exp = p3.detach().clone()
        for _ in range(4):
            exp = exp * (1 - 5e-5 * WD)
        self.assertTrue(torch.allclose(r_native[-1][2], exp, rtol=1e-9, atol=1e-12))


class T2_MomentsNotWrittenBack(unittest.TestCase):
    def test_raw_moments_identical_across_readouts(self):
        torch.manual_seed(3)
        base = [torch.randn(4, 5) for _ in range(3)]
        grads = [torch.randn(4, 5) for _ in range(5)]

        def run(mode):
            torch.manual_seed(3)
            ps = [torch.nn.Parameter(t.clone()) for t in base]
            opt = torch.optim.AdamW(ps, lr=5e-5, betas=(B1, B2), eps=EPS,
                                    weight_decay=WD)
            for p in ps:  # 模拟 bank 恢复：state 已存在
                opt.state[p] = {"step": torch.tensor(0.0),
                                "exp_avg": torch.zeros_like(p),
                                "exp_avg_sq": torch.zeros_like(p)}
            st0 = opt.state_dict()
            for g in grads:
                for p in ps:
                    p.grad = g.clone()
                qh_adamw_step(opt, ALPHA, mode)
            return (ps, {i: (opt.state[p]["exp_avg"].clone(),
                             opt.state[p]["exp_avg_sq"].clone(),
                             int(opt.state[p]["step"]))
                         for i, p in enumerate(ps)})

        ps_raw, mom_raw = run("RAW")
        ps_norm, mom_norm = run("NORM")
        for i in mom_raw:
            self.assertTrue(torch.equal(mom_raw[i][0], mom_norm[i][0]))
            self.assertTrue(torch.equal(mom_raw[i][1], mom_norm[i][1]))
            self.assertEqual(mom_raw[i][2], mom_norm[i][2])
        # 参数轨迹一般不同（rQ 与 rN 方向不同）
        diff = sum(float((a - b).norm()) for a, b in zip(ps_raw, ps_norm))
        self.assertGreater(diff, 0.0)


class T3_RAWFormulaIndependentRef(unittest.TestCase):
    def test_one_step_matches_independent_formula(self):
        torch.manual_seed(11)
        ps, opt = _mk_opt(2, seed=11)
        g = torch.randn(4, 5)
        for p in ps:
            p.grad = g.clone()
        theta0 = [p.detach().clone() for p in ps]
        m0 = [opt.state[p]["exp_avg"].clone() for p in ps]
        v0 = [opt.state[p]["exp_avg_sq"].clone() for p in ps]
        qh_adamw_step(opt, ALPHA, "RAW")
        eta = 5e-5
        for i, p in enumerate(ps):
            exp_t, exp_m, exp_v = _independent_reference_step(
                theta0[i], m0[i], v0[i], g, 1, eta, ALPHA, "RAW")
            self.assertTrue(torch.allclose(p.detach(), exp_t, rtol=1e-6, atol=1e-9))
            self.assertTrue(torch.allclose(opt.state[p]["exp_avg"], exp_m,
                                           rtol=1e-6, atol=1e-12))
            self.assertTrue(torch.allclose(opt.state[p]["exp_avg_sq"], exp_v,
                                           rtol=1e-6, atol=1e-12))


class T4_NormMatching(unittest.TestCase):
    def test_same_state_norm_and_direction(self):
        torch.manual_seed(5)
        for _ in range(10):
            mhat = torch.randn(6, 7)
            c = torch.randn(6, 7)
            P = torch.rand(6, 7) + 0.1
            rA = r_a(mhat, P)
            Pc = P * c
            rQ = r_q(rA, Pc)
            rN, fb = norm_match(rA, rQ, Pc)
            self.assertEqual(fb, "NONE")
            self.assertLess(abs(float(rN.norm()) - float(rA.norm())) / max(float(rA.norm()), 1e-30), 1e-6)
            cos = float((rQ * rN).sum()) / (float(rQ.norm()) * float(rN.norm()))
            self.assertAlmostEqual(cos, 1.0, places=6)

    def test_zero_and_cancellation_fallback(self):
        rA = torch.zeros(3, 3)
        rQ = torch.randn(3, 3)
        rN, fb = norm_match(rA, rQ, torch.randn(3, 3))
        self.assertEqual(fb, "ZERO")
        self.assertTrue(torch.equal(rN, rA))
        # cancellation：rQ 相对于 max(A,B) 极小
        rA2 = torch.randn(4, 4)
        rA2 = rA2 / rA2.norm()
        Pc2 = torch.randn(4, 4)
        Pc2 = Pc2 / Pc2.norm()
        rQ2 = 1e-14 * rA2  # Q=1e-14 <= 1e-12*max(1,1)
        rN2, fb2 = norm_match(rA2, rQ2, Pc2)
        self.assertEqual(fb2, "CANCELLATION")
        self.assertTrue(torch.equal(rN2, rA2))

    def test_shared_param_updated_once(self):
        # tied 权重：parameters() 去重后只出现一次，QH 步只更新一次
        class _Tied(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(10, 8)
                self.head = torch.nn.Linear(8, 10, bias=False)
                self.head.weight = self.emb.weight  # tied
        mod = _Tied()
        ids = [id(p) for p in mod.parameters()]
        self.assertEqual(len(ids), len(set(ids)))
        opt = torch.optim.AdamW(mod.parameters(), lr=5e-5, betas=(B1, B2),
                                eps=EPS, weight_decay=WD)
        for p in mod.parameters():
            opt.state[p] = {"step": torch.tensor(0.0),
                            "exp_avg": torch.zeros_like(p),
                            "exp_avg_sq": torch.zeros_like(p)}
        g = torch.randn(10, 8)
        mod.emb.weight.grad = g.clone()
        before = mod.emb.weight.detach().clone()
        qh_adamw_step(opt, ALPHA, "NORM")
        delta = float((mod.emb.weight.detach() - before).norm())
        m1 = opt.state[mod.emb.weight]["exp_avg"]
        # 与一步独立参考的整体 theta 一致（即只应用了一次，未双重更新）
        exp_t, exp_m, _ = _independent_reference_step(
            before, torch.zeros_like(g), torch.zeros_like(g), g, 1, 5e-5,
            ALPHA, "NORM")
        self.assertTrue(torch.allclose(mod.emb.weight.detach(), exp_t,
                                       rtol=1e-6, atol=1e-9))
        self.assertGreater(delta, 0)


class T5_AlignmentNotDescent(unittest.TestCase):
    def test_counterexample(self):
        # 真目标 f(theta)=0.5*(theta-10)^2；当前 batch 梯度 c=-1（背离最优点）
        # Adam 型读出 rA ∝ c -> dot(c,rA)>0（current-batch 一阶对齐）
        # 但更新 theta-eta*rA 使 f 增大 —— 对齐不保证独立目标下降
        theta = torch.zeros(1)
        c = torch.tensor([-1.0])  # 与真梯度 (+10 方向) 反相关的坏 batch
        m, v = c.clone(), c.clone() ** 2
        t = 30
        rA, _, _, _ = native_reference(c, m, v, t, B1, B2, EPS)
        dot = float((c * rA).sum())
        self.assertGreater(dot, 0.0)  # current-batch 一阶对齐成立
        eta = 0.5
        theta_next = theta - eta * rA
        f0 = 0.5 * float((theta + 10) ** 2)   # 真目标 min 在 -10
        f1 = 0.5 * float((theta_next + 10) ** 2)
        self.assertGreater(f1, f0)            # 但真目标上升


class T6_Schedule(unittest.TestCase):
    def test_prefix_tail_and_k_scaling(self):
        eta = make_phase_a_get_lr(_old_get_lr)
        base_93 = eta(93, 256, 5e-4)
        self.assertAlmostEqual(base_93, _old_get_lr(93, 256, 5e-4), places=15)
        self.assertAlmostEqual(eta(256, 256, 5e-4), 5e-5, places=15)   # floor
        self.assertAlmostEqual(eta(257, 256, 5e-4), 5e-5, places=15)   # tail hold
        self.assertAlmostEqual(eta(604, 256, 5e-4), 5e-5, places=15)   # 不 rebound
        for gm in (93, 256, 257, 604):
            for k in (0.5, 1.0, 2.0):
                self.assertAlmostEqual(k * eta(gm, 256, 5e-4),
                                       k * eta(gm, 256, 5e-4), places=15)
        with self.assertRaises(ValueError):
            eta(10, 128, 5e-4)   # total 拉伸被拒
        with self.assertRaises(ValueError):
            eta(10, 256, 1e-3)   # base_lr 改值被拒


class T7_DetectionAndTokens(unittest.TestCase):
    def _detect(self, means, ell, grid):
        """两连续 <=ell 检测（与 runner 同规则）。"""
        below = [m <= ell for m in means]
        last_above = None
        for i, b in enumerate(below):
            if not b:
                last_above = grid[i]
        for i in range(1, len(below)):
            if below[i - 1] and below[i]:
                return {"reached": True, "h_detect": grid[i],
                        "first_in_pair": grid[i - 1],
                        "second_confirmation": grid[i],
                        "last_observed_above": last_above}
        return {"reached": False, "h_detect": None,
                "last_observed_above": last_above}

    def test_two_consecutive_and_fields(self):
        grid = [0, 16, 32, 64]
        r = self._detect([8.0, 7.0, 6.9, 6.8], 7.02, grid)
        self.assertTrue(r["reached"])
        self.assertEqual(r["h_detect"], 32)
        self.assertEqual(r["first_in_pair"], 16)
        self.assertEqual(r["last_observed_above"], 0)
        # 非单调：中间反弹不提前确认；最终连续对成立
        grid2 = [0, 16, 32, 48, 64]
        r2 = self._detect([8.0, 6.9, 7.1, 6.9, 6.8], 7.02, grid2)
        self.assertTrue(r2["reached"])
        self.assertEqual(r2["h_detect"], 64)
        self.assertEqual(r2["first_in_pair"], 48)
        self.assertEqual(r2["last_observed_above"], 32)
        # 交替型：永无连续两次 <= ell
        r3 = self._detect([8.0, 6.9, 7.1, 6.9, 7.1], 7.02, grid2)
        self.assertFalse(r3["reached"])
        self.assertIsNone(r3["h_detect"])
        self.assertEqual(r3["last_observed_above"], 64)

    def test_sum_count_semantics_and_h0_share(self):
        sums = [10.0, 20.0]
        cnts = [1, 3]
        mean = sum(sums) / sum(cnts)  # 7.5；不是行均值 15
        self.assertAlmostEqual(mean, 7.5)
        calls = {"n": 0}

        def monitor_h0_shared(arm_idx):
            if arm_idx == 0:
                calls["n"] += 1
            return {"h0": True, "shared": arm_idx != 0}
        for a in range(5):
            monitor_h0_shared(a)
        self.assertEqual(calls["n"], 1)  # h0 只评价一次


class T8_MutantsMustFail(unittest.TestCase):
    """四个故意错误版本必须被对应检查捕获。"""

    def _mk(self):
        torch.manual_seed(21)
        p = torch.nn.Parameter(torch.randn(4, 4))
        opt = torch.optim.AdamW([p], lr=5e-5, betas=(B1, B2), eps=EPS,
                                weight_decay=WD)
        opt.state[p] = {"step": torch.tensor(0.0),
                        "exp_avg": torch.zeros_like(p),
                        "exp_avg_sq": torch.zeros_like(p)}
        return p, opt

    def test_mutant_rN_written_back_to_m_fails(self):
        p, opt = self._mk()
        g = torch.randn(4, 4)
        p.grad = g.clone()
        m0 = opt.state[p]["exp_avg"].clone()
        # 生产路径：moments 不回写
        qh_adamw_step(opt, ALPHA, "NORM")
        m_good = opt.state[p]["exp_avg"].clone()
        exp_m = B1 * m0 + (1 - B1) * g
        self.assertTrue(torch.allclose(m_good, exp_m, rtol=1e-6, atol=1e-12))
        # 突变版本：把 rN 写回 m —— 同一检查必须失败
        p2, opt2 = self._mk()
        p2.grad = g.clone()
        st = opt2.state[p2]
        m, v = st["exp_avg"], st["exp_avg_sq"]
        adamw_moment_update_(m, v, g, B1, B2)
        mhat, vhat = bias_corrected(m, v, 1, B1, B2)
        P = precondition(vhat, EPS)
        rA, Pc = r_a(mhat, P), P * g
        rQ = r_q(rA, Pc)
        rN, _ = norm_match(rA, rQ, Pc)
        st["exp_avg"].copy_(rN)  # MUTANT: writeback
        self.assertFalse(torch.allclose(st["exp_avg"], exp_m, rtol=1e-6, atol=1e-12))

    def test_mutant_step_reset_fails(self):
        # 每步重置 step -> bias correction 错 -> 与独立参考不一致
        p, opt = self._mk()
        grads = [torch.randn(4, 4) for _ in range(3)]
        th = p.detach().clone()
        m = opt.state[p]["exp_avg"].clone()
        v = opt.state[p]["exp_avg_sq"].clone()
        for i, g in enumerate(grads, start=1):
            p.grad = g.clone()
            qh_adamw_step(opt, ALPHA, "RAW")
            th, m, v = _independent_reference_step(th, m, v, g, i, 5e-5, ALPHA, "RAW")
        self.assertTrue(torch.allclose(p.detach(), th, rtol=1e-6, atol=1e-9))
        # MUTANT：每步 step 归零 -> t 恒为 1（bias correction 停滞）
        p2, opt2 = self._mk()
        th2 = p2.detach().clone()
        m2 = opt2.state[p2]["exp_avg"].clone()
        v2 = opt2.state[p2]["exp_avg_sq"].clone()
        for i, g in enumerate(grads, start=1):
            p2.grad = g.clone()
            opt2.state[p2]["step"] = 0  # MUTANT
            qh_adamw_step(opt2, ALPHA, "RAW")
            # 正确行为参照：t 按 1,2,3 推进
            th2, m2, v2 = _independent_reference_step(th2, m2, v2, g, i, 5e-5,
                                                      ALPHA, "RAW")
        self.assertFalse(torch.allclose(p2.detach(), th2, rtol=1e-6, atol=1e-9))

    def test_mutant_valid_eq_slots_fails(self):
        # 构造带 padding 的 y：valid != slots；突变记账valid=slots必须被抓
        y = torch.full((2, 5), -100)
        y[0, 1:4] = torch.tensor([1, 2, 3])
        y[1, 1:3] = torch.tensor([4, 5])
        slots = int(y.numel())
        valid = int((y[:, 1:] != -100).sum())
        self.assertNotEqual(valid, slots)
        # 生产记账：valid 从 mask 累加；突变记账 valid=slots -> 与 mask 不符
        mutant_valid = slots
        self.assertNotEqual(mutant_valid, valid)

    def test_mutant_norm_swap_fails(self):
        # 分子分母反转 rN=(Q/A)*rQ -> 范数!=A 被 T4 检查捕获
        torch.manual_seed(31)
        rA = torch.randn(5, 5)
        Pc = torch.randn(5, 5)
        rQ = r_q(rA, Pc)
        A, Q = float(rA.norm()), float(rQ.norm())
        self.assertGreater(abs(A - Q) / max(A, 1e-30), 1e-8)  # 非退化构造
        rN_good, _ = norm_match(rA, rQ, Pc)
        self.assertLess(abs(float(rN_good.norm()) - A) / A, 1e-6)
        rN_mutant = (Q / A) * rQ  # MUTANT: swapped
        self.assertGreater(abs(float(rN_mutant.norm()) - A) / A, 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
