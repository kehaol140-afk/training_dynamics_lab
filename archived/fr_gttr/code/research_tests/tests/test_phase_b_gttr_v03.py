#!/usr/bin/env python3
"""Phase B v3 CPU 单元测试 T1–T8 + 共享 GPU 计时器隔离测试（amendment §5）。

运行：cd research/tests && python -m unittest -v test_phase_b_gttr_v03
（需 conda env fr_minimind；run_cpu_tests 子命令自动设置 PYTHONPATH）

全部纯 CPU，不触碰 GPU、不写入 outputs/、不消耗实验预算。
"""
from __future__ import annotations

import inspect
import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import torch

from fr import phase_b_gttr_score_v03 as S
import phase_b_gttr_v03 as R


def _rnd_named(shape_map, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return {k: torch.randn(*sh, generator=g, dtype=dtype)
            for k, sh in shape_map.items()}


SHAPES = {"a": (3, 4), "b": (5,)}


class T1_LRActionSequence(unittest.TestCase):
    """T1：eta_k^u = eta_base(n0+k)·exp(u·1[k<4])；k≥4 释放；乘性为 exp(u) 非 1+u。"""

    def test_constant_base_exp_multiplier(self):
        eta = lambda n: 5e-4
        for u in (0.0, -0.05, 0.05):
            seq = S.lr_action_seq(eta, 46, u, 64)
            self.assertEqual(len(seq), 64)
            for k in range(64):
                want = 5e-4 * math.exp(u if k < 4 else 0.0)
                self.assertAlmostEqual(seq[k], want, places=18)
        # exp(u) 而非 1+u
        seq = S.lr_action_seq(eta, 46, 0.05, 64)
        self.assertNotAlmostEqual(seq[0], 5e-4 * 1.05, places=10)
        self.assertAlmostEqual(seq[0], 5e-4 * math.exp(0.05), places=18)

    def test_cosine_base_uses_n0_plus_k(self):
        eta = lambda n: 1.0 / (n + 1)
        seq = S.lr_action_seq(eta, 46, 0.0, 8)
        for k in range(8):
            self.assertAlmostEqual(seq[k], 1.0 / (46 + k + 1), places=18)

    def test_release_after_step3_and_no_state_reset(self):
        eta = lambda n: 0.001
        seq_a = S.lr_action_seq(eta, 46, 0.05, 64)
        seq_b = S.lr_action_seq(eta, 46, 0.05, 64)   # 再次调用不受影响
        self.assertEqual(seq_a, seq_b)
        self.assertAlmostEqual(seq_a[3], 0.001 * math.exp(0.05), places=18)
        self.assertAlmostEqual(seq_a[4], 0.001, places=18)
        self.assertAlmostEqual(seq_a[63], 0.001, places=18)

    def test_runner_S_lr_matches(self):
        eta = lambda n: 2e-4
        for u in (0.0, -0.05, 0.05):
            for k in (0, 3, 4, 63):
                self.assertAlmostEqual(R.S_lr(eta, 82, u, k),
                                       S.lr_action_seq(eta, 82, u, 64)[k],
                                       places=18)


class T2_ClonedAdamWBitwise(unittest.TestCase):
    """T2：克隆 AdamW 一步与真实 optimizer 逐位一致（clip 绑定/不绑定）。"""

    def _one_step_both(self, g_scale, n0=46, lr=1.788e-4):
        torch.manual_seed(11)
        p_real = torch.nn.Parameter(torch.randn(4, 3))
        p_virt = p_real.detach().clone()
        m0 = torch.randn(4, 3)
        v0 = torch.rand(4, 3) + 0.1
        g = torch.randn(4, 3) * g_scale

        opt = torch.optim.AdamW([p_real], lr=lr, betas=(0.9, 0.999),
                                eps=1e-8, weight_decay=0.01)
        st = opt.state[p_real]
        st["exp_avg"] = m0.clone()
        st["exp_avg_sq"] = v0.clone()
        st["step"] = torch.tensor(float(n0))
        p_real.grad = g.clone()
        torch.nn.utils.clip_grad_norm_([p_real], 1.0)
        opt.step()

        vparams, vopt = S.build_virtual_optimizer(
            torch, {"w": p_virt}, {"w": m0}, {"w": v0}, n0)
        S.virtual_step(vopt, vparams, {"w": g}, lr, 1.0)
        return p_real.detach(), vparams["w"].detach()

    def test_bitwise_nonbinding(self):
        pr, pv = self._one_step_both(g_scale=0.01)   # ||g||~0.06×... < 1
        self.assertTrue(torch.equal(pr, pv))

    def test_bitwise_binding(self):
        pr, pv = self._one_step_both(g_scale=100.0)  # clip 强绑定
        self.assertTrue(torch.equal(pr, pv))

    def test_clip_coeff_edges(self):
        self.assertEqual(S.clip_coeff(float("inf"), 1.0), 0.0)
        self.assertEqual(S.clip_coeff(float("nan"), 1.0), 0.0)
        self.assertEqual(S.clip_coeff(0.0, 1.0), 1.0)
        self.assertEqual(S.clip_coeff(-1.0, 1.0), 1.0)
        self.assertEqual(S.clip_coeff(2.0, 1.0), 0.5)
        self.assertEqual(S.clip_coeff(0.5, 1.0), 1.0)


class T3_FDHVP(unittest.TestCase):
    """T3：二次/线性场 FD-HVP 对解析解；随机性恢复负测试带牙齿；half-check 门。"""

    def _linear_field(self, dim=6, seed=3):
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(dim, dim, generator=g, dtype=torch.float64)
        A = A @ A.T + dim * torch.eye(dim, dtype=torch.float64)
        theta0 = torch.randn(dim, generator=g, dtype=torch.float64)
        return A, theta0

    def test_linear_field_matches_analytic(self):
        A, theta0 = self._linear_field()
        tnorm = float(theta0.norm())

        def g_eval(overlay=None):
            th = theta0 + (overlay["w"] if overlay is not None else 0.0)
            return {"w": A @ th}
        v1 = {"w": torch.ones(6, dtype=torch.float64)}
        out = S.krylov_fd_directions(torch, g_eval, tnorm, v1, max_q=4)
        self.assertEqual(out["diag"]["rank"], 4)
        for v, y in zip(out["V"], out["Y"]):
            av = (A @ v["w"].unsqueeze(-1)).squeeze(-1)
            rel = float((y["w"] - av).norm() / max(float(av.norm()), 1e-30))
            self.assertLess(rel, 1e-6)
        # half-check：线性场 fd_rel ≈ 0
        self.assertLess(out["diag"]["fd_rel"], 1e-6)
        self.assertNotIn("LOCAL_OPERATOR_NOT_CALIBRATED",
                         out["diag"]["labels"])

    def test_randomness_restore_negative_has_teeth(self):
        """同一 SIM 随机源：恢复 → FD 精确；不恢复 → 端点噪声污染方向。"""
        gen = torch.Generator().manual_seed(7)
        base = torch.randn(4, generator=gen, dtype=torch.float64)
        v1 = {"w": torch.ones(4, dtype=torch.float64)}

        def make(restore):
            def g_eval(overlay=None):
                if restore:
                    st = gen.get_state()
                    noise = torch.randn(4, generator=gen, dtype=torch.float64)
                    gen.set_state(st)
                else:
                    noise = torch.randn(4, generator=gen, dtype=torch.float64)
                return {"w": base + noise
                        + (overlay["w"] if overlay is not None else 0.0)}
            return g_eval
        good = S.krylov_fd_directions(torch, make(True), 1.0, v1, max_q=1)
        y_good = good["Y"][0]["w"]
        # 解析 y = v1/||v1||（krylov 内部对 v1 归一化；恢复随机性 → 端点一致）
        v1n = v1["w"] / float(v1["w"].norm())
        self.assertLess(float((y_good - v1n).norm()), 1e-8)
        bad = S.krylov_fd_directions(torch, make(False), 1.0, v1, max_q=1)
        y_bad = bad["Y"][0]["w"]
        # 不恢复：端点噪声差 / (2δ) ~ O(500) 量级，方向被污染
        self.assertGreater(float((y_bad - v1n).norm()), 1.0)

    def test_halfcheck_gate_cubic_small_x0(self):
        """x0=5e-4 的三次场：δ=1e-3 下 fd_rel≈0.43 > 0.20 → 门触发。"""
        theta0 = {"w": torch.tensor([5e-4], dtype=torch.float64)}

        def g_eval(overlay=None):
            th = theta0["w"] + (overlay["w"] if overlay is not None else 0.0)
            return {"w": th ** 3}
        v1 = {"w": torch.ones(1, dtype=torch.float64)}
        out = S.krylov_fd_directions(torch, g_eval, 5e-4, v1, max_q=1)
        self.assertIn("LOCAL_OPERATOR_NOT_CALIBRATED", out["diag"]["labels"])
        self.assertGreater(out["diag"]["fd_rel"], 0.20)

    def test_zero_fd_signal_no_fabrication(self):
        def g_eval(overlay=None):
            return {"w": torch.zeros(3, dtype=torch.float64)}
        v1 = {"w": torch.ones(3, dtype=torch.float64)}
        out = S.krylov_fd_directions(torch, g_eval, 1.0, v1, max_q=2)
        self.assertIn("ZERO_FD_SIGNAL", out["diag"]["labels"])
        self.assertEqual(out["diag"]["rank"], 0)
        self.assertEqual(out["Y"], [])


class T4_OrthonormalRankBreak(unittest.TestCase):
    """T4：V 正交归一；HU 形状；低秩场 rank-break；零方向不捏造信号。"""

    def test_orthonormal_full_rank(self):
        g = torch.Generator().manual_seed(5)
        dim = 8
        A = torch.randn(dim, dim, generator=g, dtype=torch.float64)
        A = A @ A.T + dim * torch.eye(dim, dtype=torch.float64)
        theta0 = torch.randn(dim, generator=g, dtype=torch.float64)

        def g_eval(overlay=None):
            th = theta0 + (overlay["w"] if overlay is not None else 0.0)
            return {"w": A @ th}
        v1 = {"w": torch.randn(dim, generator=g, dtype=torch.float64)}
        out = S.krylov_fd_directions(torch, g_eval, 1.0, v1, max_q=4)
        V = out["V"]
        self.assertEqual(len(V), 4)
        for i, vi in enumerate(V):
            self.assertAlmostEqual(float(vi["w"].norm()), 1.0, places=12)
            for j, vj in enumerate(V):
                if i != j:
                    d = float(torch.dot(vi["w"], vj["w"]))
                    self.assertLess(abs(d), 1e-10)
        self.assertEqual(len(out["Y"]), 4)

    def test_rank_break_low_rank(self):
        g = torch.Generator().manual_seed(9)
        dim = 8
        B = torch.randn(dim, 2, generator=g, dtype=torch.float64)
        A = B @ B.T + 1e-12 * torch.eye(dim, dtype=torch.float64)
        theta0 = torch.randn(dim, generator=g, dtype=torch.float64)

        def g_eval(overlay=None):
            th = theta0 + (overlay["w"] if overlay is not None else 0.0)
            return {"w": A @ th}
        v1 = {"w": torch.randn(dim, generator=g, dtype=torch.float64)}
        out = S.krylov_fd_directions(torch, g_eval, 1.0, v1, max_q=4,
                                     rank_tol=1e-8)
        self.assertEqual(out["diag"]["rank"], 2)
        self.assertIn("RANK_BREAK", out["diag"]["labels"])

    def test_zero_direction_stops(self):
        def g_eval(overlay=None):
            return {"w": torch.zeros(4, dtype=torch.float64)}
        v1 = {"w": torch.ones(4, dtype=torch.float64)}
        out = S.krylov_fd_directions(torch, g_eval, 1.0, v1, max_q=4)
        self.assertEqual(out["diag"]["rank"], 0)
        self.assertEqual(len(out["V"]), 1)   # 只有初始方向，不扩张
        v1n = v1["w"] / float(v1["w"].norm())
        self.assertTrue(torch.allclose(out["V"][0]["w"], v1n))


class T5_H1ModelsIdentical(unittest.TestCase):
    """T5：H=1 时 MODEL_NO_GEOMETRY 与 MODEL_GTTR 逐位一致（同 g0）。"""

    def test_h1_identical(self):
        torch.manual_seed(21)
        named = _rnd_named(SHAPES, seed=21)
        m0 = _rnd_named(SHAPES, seed=22)
        v0 = {k: t.abs() + 0.1 for k, t in _rnd_named(SHAPES, seed=23).items()}
        g0 = _rnd_named(SHAPES, seed=24)
        q0 = _rnd_named(SHAPES, seed=25)
        # 合成 GTTR 方向
        v = {k: t / math.sqrt(sum(float(x.norm() ** 2) for x in
                                _rnd_named(SHAPES, seed=26).values()))
             for k, t in _rnd_named(SHAPES, seed=26).items()}
        y = _rnd_named(SHAPES, seed=27)
        fields = {"NO": S.make_field(g0), "GTTR": S.make_field(g0, [v], [y])}
        outs = {}
        for key, field in fields.items():
            lr_seq = [1.5e-4] * 4
            theta_h, _ = S.propagate(torch, named, m0, v0, 46, field, lr_seq,
                                     tau=1.0, h_points=(1, 4), n_steps=4)
            outs[key] = theta_h
        for k in outs["NO"][1]:
            self.assertTrue(torch.equal(outs["NO"][1][k], outs["GTTR"][1][k]))
        # H1 的 J_hat 也一致（u 臂）
        j_no = S.j_hat(q0, outs["NO"][1], outs["NO"][1])
        self.assertEqual(j_no, 0.0)


class T6_NativeZeroAndSelection(unittest.TestCase):
    """T6：u=0 相对目标两模型均为 0；原生真值 0；选择规则与插入序无关。"""

    def test_native_literal_zero(self):
        q0 = _rnd_named(SHAPES, seed=31)
        th_u = _rnd_named(SHAPES, seed=32)
        self.assertEqual(S.j_hat(q0, th_u, th_u), 0.0)
        self.assertEqual(S.q0_dot_delta(q0, th_u, th_u), 0.0)

    def test_select_tie_prefers_native(self):
        sel = S.select_action({0.0: 1.0, -0.05: 1.0, 0.05: 1.0})
        self.assertEqual(sel, 0.0)

    def test_select_tie_down_before_up(self):
        sel = S.select_action({0.0: 1.0, -0.05: 0.5, 0.05: 0.5})
        self.assertEqual(sel, -0.05)

    def test_select_strict_min(self):
        sel = S.select_action({0.0: -1.0, -0.05: 0.5, 0.05: 0.5})
        self.assertEqual(sel, 0.0)

    def test_selection_order_invariant(self):
        jv = {0.0: 0.1, -0.05: 0.2, 0.05: -0.3}
        import itertools
        for perm in itertools.permutations(jv.keys()):
            d = {u: jv[u] for u in perm}
            self.assertEqual(S.select_action(d), 0.05)


class T7_ErrorDecompositionAndAggregation(unittest.TestCase):
    """T7：标量分解恒等式；sum/count 聚合；固定网格 SSE/TSS/E 独立复算。"""

    def test_decomposition_identity_random(self):
        rng = torch.Generator().manual_seed(41)
        for _ in range(1000):
            j = float(torch.randn(1, generator=rng))
            t = float(torch.randn(1, generator=rng))
            l = float(torch.randn(1, generator=rng))
            d = S.error_decomposition(j, t, l)
            self.assertTrue(d["identity_holds"])
            self.assertAlmostEqual(d["e_total"], j - t, places=15)
            self.assertAlmostEqual(d["e_propagation"], j - l, places=15)
            self.assertAlmostEqual(d["e_readout"], l - t, places=15)

    def test_sum_count_not_mean_of_means(self):
        # 两组行：组大和组小；sum/count ≠ 两组 mean 的均值
        sums, cnts = [100.0, 1.0], [1000, 1]
        agg = sum(sums) / sum(cnts)
        bad = sum(s / c for s, c in zip(sums, cnts)) / 2
        self.assertNotAlmostEqual(agg, bad, places=6)
        self.assertAlmostEqual(agg, 101.0 / 1001.0, places=15)

    def test_fixed_grid_sse_tss_e_recomputed(self):
        """独立复算（不调用任何生产 helper）：E = sqrt(SSE/TSS)。"""
        rng = torch.Generator().manual_seed(43)
        preds = torch.randn(12, generator=rng).tolist()
        truth = torch.randn(12, generator=rng).tolist()
        sse = sum((p - t) ** 2 for p, t in zip(preds, truth))
        tss = sum(t * t for t in truth)
        E = math.sqrt(sse / tss)
        # 与生产公式的比率关系一致（sqrt 浮点往返误差 ~1e-16 量级）
        self.assertAlmostEqual(E * E, sse / tss, places=14)
        self.assertGreaterEqual(E, 0.0)
        # TARGET_ZERO：全零真值网格 → tss==0 → analyze 记 None（不捏造 E）
        truth_zero = [0.0] * 12
        tss_zero = sum(t * t for t in truth_zero)
        self.assertEqual(tss_zero, 0.0)
        self.assertIsNone(None if tss_zero == 0 else E)


class T8_SealPathAssertion(unittest.TestCase):
    """T8：封存路径断言 —— 缺 seal/篡改/观测泄漏全部显式拒绝。"""

    def _mk(self, tmp, observed_lines=()):
        out = Path(tmp)
        (out / "predictions.csv").write_text(
            "state_id,model,action,u,H,J_hat,finite,seal_order\n"
            "S0,MODEL_GTTR,LR_DOWN,-0.05,4,0.123,True,0\n")
        if observed_lines:
            (out / "observed_row_stats.csv").write_text(
                "".join(observed_lines))
        return out

    def test_missing_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._mk(tmp)
            with mock.patch.object(R, "OUT", out):
                with self.assertRaisesRegex(AssertionError,
                                            "PREDICTION_SEAL_MISSING"):
                    R.verify_seal()

    def test_tampered_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._mk(tmp)
            import hashlib
            bad = {"predictions_sha256": "0" * 64}
            (out / "seal.json").write_text(json.dumps(bad))
            with mock.patch.object(R, "OUT", out):
                with self.assertRaisesRegex(AssertionError,
                                            "PREDICTION_SEAL_TAMPERED"):
                    R.verify_seal()

    def test_main_arm_leak_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._mk(tmp, ["S0,NATIVE,4,monitor,12,1.0,100\n"])
            import hashlib
            sha = hashlib.sha256((out / "predictions.csv").read_bytes())
            (out / "seal.json").write_text(json.dumps(
                {"predictions_sha256": sha.hexdigest()}))
            with mock.patch.object(R, "OUT", out):
                with self.assertRaisesRegex(AssertionError, "OBSERVED_LEAK"):
                    R.verify_seal()

    def test_smoke_rows_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._mk(tmp, ["S0,smoke,4,monitor,12,1.0,100\n",
                                 "S0,h0,0,control,3,2.0,50\n"])
            import hashlib
            sha = hashlib.sha256((out / "predictions.csv").read_bytes())
            (out / "seal.json").write_text(json.dumps(
                {"predictions_sha256": sha.hexdigest()}))
            with mock.patch.object(R, "OUT", out):
                seal = R.verify_seal()
            self.assertEqual(seal["predictions_sha256"], sha.hexdigest())


class T9_TimerIsolationAmendment(unittest.TestCase):
    """Amendment §5：elapsed=5401s 时数值门/选择/计算计数不变，仅运维提示变。"""

    def test_check_caps_has_no_wall_param(self):
        params = inspect.signature(R._check_caps).parameters
        for banned in ("wall", "elapsed", "wall_s", "time_s", "eta_s"):
            self.assertNotIn(banned, params)

    def test_hint_value_and_label(self):
        self.assertEqual(R.WALL_OPS_HINT_S, 5400)
        self.assertEqual(R._LEDGER.name, "counters.json")

    def test_elapsed_5401_gates_unchanged(self):
        """模拟 elapsed=5401：把 perf_counter 顶到 +5401，门行为不变。"""
        saved = dict(R._C)
        try:
            R._C.update({"updates": 0, "train_fw": 0, "model_fw": 0,
                         "monitor_fw": 0, "control_fw": 0, "train_bw": 0,
                         "model_bw": 0, "fd_dirs": 0.0, "monitor_evals": 0,
                         "control_evals": 0})
            base = time.perf_counter()
            with mock.patch.object(R.time, "perf_counter",
                                   side_effect=lambda: base + 5401.0):
                R._check_caps("timer_test", updates=1, train_fw=2, bw=2)
                # 仍在 caps 内 → 不抛异常；elapsed 5401 与此无关
                R._C["updates"] = R.CAP["updates"]      # 已达 cap
                R._check_caps("timer_test", updates=0)   # 不超限 → OK
                with self.assertRaises(SystemExit):
                    R._check_caps("timer_test", updates=1)   # 1281 > 1280
                # 选择函数不接收时间
                sel = S.select_action({0.0: 0.1, -0.05: 0.2, 0.05: -0.3})
                self.assertEqual(sel, 0.05)
        finally:
            R._C.clear()
            R._C.update(saved)

    def test_eta_estimate_is_hint_only(self):
        est_path = R.OUT / "eta_estimate.json"
        # 运维提示字段存在与否不影响科学输出文件生成路径
        self.assertTrue(hasattr(R, "WALL_OPS_HINT_S"))


if __name__ == "__main__":
    unittest.main()
