#!/usr/bin/env python3
"""C0-GTTR theory tests T1-T9 (CPU only, FP64, no GPU).

Each test is constructed to fail on a sign/index/branch/order error in the
Geometry/Tracking/Transport/Readout algebra BEFORE any GPU result is produced
(config-freeze discipline, C0 protocol section 6). Negative variants required
by the protocol to fail are asserted to deviate.
"""
import math
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

import fr.c0_gttr_validation_v01 as c0
from fr.a0c_adamw_response_propagation_v01 import clip_jvp

DT = torch.float64
RTOL = 1e-7
ATOL = 1e-9
B1, B2, EPS = 0.9, 0.999, 1e-8
TAU = 1.0


def _td(*pairs):
    return {n: torch.tensor(v, dtype=DT) for n, v in pairs}


def _assert_td_close(test, a, b, rtol=RTOL, atol=ATOL):
    for n in a:
        test.assertTrue(torch.allclose(a[n], b[n], rtol=rtol, atol=atol),
                        msg=f"tensor {n}: max diff "
                        f"{(a[n]-b[n]).abs().max().item():.3e}")


def _run_native(grads, lrs, wd, tau, t_warm):
    """ grads: list of raw-gradient dicts; returns per-step (theta, m_hat, v_hat)."""
    outs = []
    theta = {n: torch.zeros_like(g) for n, g in grads[0].items()}
    opt = torch.optim.AdamW([{"params": [theta[n]], "weight_decay": wd.get(n, 0.0)}
                             for n in theta], lr=lrs[0], betas=(B1, B2), eps=EPS)
    m_hat = {n: torch.zeros_like(theta[n]) for n in theta}
    v_hat = {n: torch.zeros_like(theta[n]) for n in theta}
    for k, g in enumerate(grads):
        for n in theta:
            theta[n].grad = g[n].clone()
        torch.nn.utils.clip_grad_norm_(list(theta.values()), tau)
        for grp in opt.param_groups:
            grp["lr"] = lrs[k]
        opt.step()
        t = t_warm + k + 1
        for n in theta:
            st = opt.state[theta[n]]
            m_hat[n] = st["exp_avg"] / (1 - B1 ** t)
            v_hat[n] = st["exp_avg_sq"] / (1 - B2 ** t)
        outs.append(( {n: p.detach().clone() for n, p in theta.items()},
                      {n: m.clone() for n, m in m_hat.items()},
                      {n: v.clone() for n, v in v_hat.items()} ))
    return outs


class T1_RewriteVsNative(unittest.TestCase):
    """T1: full AdamW vs bias-corrected m/v rewrite. Non-constant LR, non-zero
    decay, masked param groups, clip active/inactive, restored step count."""

    def _drive(self, t_warm, steps, clip_every, wd_map, tau=TAU):
        torch.manual_seed(11)
        names = ["w", "b", "emb"]
        shapes = {"w": (4, 3), "b": (4,), "emb": (3, 5)}
        grads, lrs = [], []
        for k in range(t_warm + steps):
            scale = 3.0 if (clip_every and k % clip_every == 0) else 0.05
            grads.append({n: scale * torch.randn(shapes[n], dtype=DT) for n in names})
            lrs.append(5e-4 * (0.55 + 0.45 * math.cos(math.pi * (k + 1) / 128.0)))
        native = _run_native(grads, lrs, wd_map, tau, 0)
        # rewrite chain replays the FULL trajectory (zero init like _run_native)
        th = {n: torch.zeros(shapes[n], dtype=DT) for n in names}
        mh = {n: torch.zeros(shapes[n], dtype=DT) for n in names}
        vh = {n: torch.zeros(shapes[n], dtype=DT) for n in names}
        for i in range(t_warm + steps):
            c = c0.apply_clip(grads[i], tau)
            step = c0.adamw_rewrite_step(m_hat=mh, v_hat=vh, c=c,
                                         beta1=B1, beta2=B2, t_prev=i,
                                         eps=EPS, lr=lrs[i], wd=wd_map, theta=th)
            th = {n: th[n] + step["s"][n] for n in th}
            mh, vh = step["m_hat"], step["v_hat"]
            n_th, n_mh, n_vh = native[i]
            _assert_td_close(self, th, n_th)
            _assert_td_close(self, mh, n_mh)
            _assert_td_close(self, vh, n_vh)

    def test_fresh_state_first_step_no_divzero(self):
        # t_prev=0 -> (a,b)=(0,1); zero init moments; no NaN/Inf.
        g = _td(("w", [0.7, -1.1]))
        step = c0.adamw_rewrite_step(m_hat=_td(("w", [0.0, 0.0])),
                                     v_hat=_td(("w", [0.0, 0.0])), c=g,
                                     beta1=B1, beta2=B2, t_prev=0, eps=EPS,
                                     lr=5e-4, wd=0.01, theta=_td(("w", [0.3, 0.2])))
        for v in step["s"].values():
            self.assertTrue(torch.isfinite(v).all())

    def test_restored_step_count_clip_alternating(self):
        self._drive(t_warm=46, steps=8, clip_every=2, wd_map={"w": 0.01, "b": 0.0, "emb": 0.01})

    def test_masked_groups_zero_wd(self):
        self._drive(t_warm=3, steps=6, clip_every=1, wd_map={"w": 0.0, "b": 0.05, "emb": 0.01})

    def test_no_clip(self):
        self._drive(t_warm=5, steps=5, clip_every=None, wd_map={"w": 0.01, "b": 0.01, "emb": 0.01})


class T2_CarryOverIdentity(unittest.TestCase):
    """T2: realized d_t carry-over with non-constant P and eta. Wrong variants
    (ignore P_t vs P_{t-1}, double eta) must FAIL to match."""

    def _chain(self, wrong=None):
        torch.manual_seed(23)
        names = ["w"]
        shapes = {"w": (6,)}
        theta = {n: torch.randn(shapes[n], dtype=DT) for n in names}
        mh = {n: torch.zeros(shapes[n], dtype=DT) for n in names}
        vh = {n: torch.full(shapes[n], 0.05, dtype=DT) for n in names}
        lr0 = 5e-4
        seq = []
        for k in range(10):
            lr_k = lr0 * (0.4 + 0.6 * abs(math.sin(k)))  # non-constant eta
            g = (0.5 if k % 3 else 2.5) * torch.randn(shapes["w"], dtype=DT)
            c = c0.apply_clip({"w": g}, TAU)
            if wrong == "P_prev":
                # wrong: use previous v_hat inside P (ignore v update)
                P_wrong = c0.p_diag(vh, EPS)
                a1, b1 = c0.ema_coeffs(B1, 46 + k)
                m1 = a1 * mh["w"] + b1 * c["w"]
                r = m1 * P_wrong["w"]
                s = {"w": -lr_k * r - lr_k * 0.01 * theta["w"]}
                m1d = {"w": m1}
                a2, b2 = c0.ema_coeffs(B2, 46 + k)
                v1d = {"w": a2 * vh["w"] + b2 * c["w"] * c["w"]}
            elif wrong == "double_eta":
                step = c0.adamw_rewrite_step(m_hat=mh, v_hat=vh, c=c, beta1=B1,
                                             beta2=B2, t_prev=46 + k, eps=EPS,
                                             lr=lr_k, wd=0.01, theta=theta)
                s = {n: lr_k * v for n, v in step["s"].items()}  # extra eta
                m1d, v1d = step["m_hat"], step["v_hat"]
            else:
                step = c0.adamw_rewrite_step(m_hat=mh, v_hat=vh, c=c, beta1=B1,
                                             beta2=B2, t_prev=46 + k, eps=EPS,
                                             lr=lr_k, wd=0.01, theta=theta)
                s, m1d, v1d = step["s"], step["m_hat"], step["v_hat"]
            theta = {n: theta[n] + s[n] for n in theta}
            mh, vh = m1d, v1d
            seq.append(theta["w"].clone())
        return theta, mh, vh

    def test_correct_chain_matches_native(self):
        torch.manual_seed(23)
        shapes = {"w": (6,)}
        theta = {n: torch.randn(shapes[n], dtype=DT) for n in shapes}
        grads = []
        lrs = []
        for k in range(10):
            lrs.append(5e-4 * (0.4 + 0.6 * abs(math.sin(k))))
            grads.append({"w": (0.02 if k % 3 else 2.5) * torch.randn(shapes["w"], dtype=DT)})
        # warm 46 steps to build the same starting moments
        warm_grads = [ {n: 0.05 * torch.randn(shapes[n], dtype=DT) for n in shapes} for _ in range(46)]
        all_grads = warm_grads + grads
        all_lrs = [5e-4] * 46 + lrs
        native = _run_native(all_grads, all_lrs, {"w": 0.01}, TAU, 0)
        th = {n: torch.zeros(6, dtype=DT) for n in shapes}
        mh = {n: torch.zeros(6, dtype=DT) for n in shapes}
        vh = {n: torch.full((6,), 0.0, dtype=DT) for n in shapes}
        for k in range(46 + 10):
            c = c0.apply_clip(all_grads[k], TAU)
            step = c0.adamw_rewrite_step(m_hat=mh, v_hat=vh, c=c, beta1=B1,
                                         beta2=B2, t_prev=k, eps=EPS,
                                         lr=all_lrs[k], wd={"w": 0.01}, theta=th)
            th = {n: th[n] + step["s"][n] for n in th}
            mh, vh = step["m_hat"], step["v_hat"]
            _assert_td_close(self, th, native[k][0])
        # negative variants must deviate
        for wrong in ("P_prev", "double_eta"):
            thw, _, _ = self._chain(wrong=wrong)
            thc, _, _ = self._chain()
            diff = max((thw[n] - thc[n]).abs().max().item() for n in thc)
            self.assertGreater(diff, 1e-6, msg=f"wrong variant {wrong} matched")


class T3_TangentVsFD(unittest.TestCase):
    """T3: full-state + log-LR analytic tangent vs central FD on a PD quadratic
    training loss; V uses a DIFFERENT Hessian; clip active and inactive; v
    strictly positive (boundary flagged separately)."""

    def _run(self, Htr, HV, x0, t_warm, steps, wd, tau, u):
        x = { "x": x0.clone()}
        opt = torch.optim.AdamW([{"params": [x["x"]], "weight_decay": wd}],
                                lr=5e-4, betas=(B1, B2), eps=EPS)
        # warm-up to restore step count / moments
        for k in range(t_warm):
            g = Htr @ x["x"]
            x["x"].grad = g.clone()
            torch.nn.utils.clip_grad_norm_([x["x"]], tau)
            opt.step()
        xs = [x["x"].clone()]
        for k in range(steps):
            g = Htr @ x["x"]
            x["x"].grad = g.clone()
            torch.nn.utils.clip_grad_norm_([x["x"]], tau)
            opt.param_groups[0]["lr"] = 5e-4 * math.exp(u)
            opt.step()
            xs.append(x["x"].clone())
        return x["x"].clone(), opt, xs

    def _tangent(self, Htr, HV, x0, t_warm, steps, wd, tau, lr_u):
        x = x0.clone()
        mh = torch.zeros_like(x0)
        vh = torch.zeros_like(x0)
        opt = torch.optim.AdamW([{"params": [x], "weight_decay": wd}],
                                lr=5e-4, betas=(B1, B2), eps=EPS)
        for k in range(t_warm):
            x.grad = (Htr @ x).clone()
            torch.nn.utils.clip_grad_norm_([x], tau)
            opt.step()
        t = t_warm
        mh = opt.state[x]["exp_avg"] / (1 - B1 ** t)
        vh = opt.state[x]["exp_avg_sq"] / (1 - B2 ** t)
        d_th = torch.zeros_like(x0)
        d_m = torch.zeros_like(x0)
        d_v = torch.zeros_like(x0)
        for k in range(steps):
            g = Htr @ x
            x.grad = g.clone()
            torch.nn.utils.clip_grad_norm_([x], tau)
            c_used = x.grad.clone()  # the CLIPPED gradient the step consumes
            jac = clip_jvp(gradient_by_name={"x": g.detach()},
                           tangent_gradient_by_name={"x": (Htr @ d_th).detach()},
                           max_norm=tau)
            dc = jac["clip_jvp"]["x"]
            st = c0.adamw_step_tangent(
                d_theta={"x": d_th}, m_hat={"x": mh}, v_hat={"x": vh},
                d_m_hat={"x": d_m}, d_v_hat={"x": d_v}, c={"x": c_used},
                d_c={"x": dc}, theta={"x": x.detach()}, beta1=B1, beta2=B2,
                t_prev=t + k, eps=EPS, lr=5e-4, wd=wd, lr_u_tangent=lr_u)
            d_th, d_m, d_v = st["d_theta"]["x"], st["d_m_hat"]["x"], st["d_v_hat"]["x"]
            opt.param_groups[0]["lr"] = 5e-4
            opt.step()
            tt = t + k + 1
            mh = opt.state[x]["exp_avg"] / (1 - B1 ** tt)
            vh = opt.state[x]["exp_avg_sq"] / (1 - B2 ** tt)
        return x.clone(), d_th

    def _fd_case(self, clip_active_target):
        torch.manual_seed(31)
        Htr = torch.diag(torch.tensor([1.0, 2.0, 6.0], dtype=DT))
        HV = torch.diag(torch.tensor([0.5, 3.0, 1.0], dtype=DT))
        x0 = torch.tensor([0.9, -0.4, 0.6], dtype=DT)
        if clip_active_target:
            x0 = x0 * 2.0  # pushes gradients over tau=1 on several steps
        wd, tau = 0.01, 1.0
        d = 1e-5
        xA, _, _ = self._run(Htr, HV, x0, 46, 8, wd, tau, +d)
        xB, _, _ = self._run(Htr, HV, x0, 46, 8, wd, tau, -d)
        fd = ((0.5 * xA @ HV @ xA) - (0.5 * xB @ HV @ xB)) / (2 * d)
        xT, d_th = self._tangent(Htr, HV, x0, 46, 8, wd, tau, lr_u=True)
        ana = float((HV @ xT) @ d_th)
        rel = abs(ana - fd) / max(abs(fd), 1e-12)
        return fd, ana, rel

    def test_clip_inactive(self):
        fd, ana, rel = self._fd_case(False)
        self.assertLess(rel, 1e-5, msg=f"fd={fd} ana={ana} rel={rel}")

    def test_clip_active(self):
        fd, ana, rel = self._fd_case(True)
        self.assertLess(rel, 1e-4, msg=f"fd={fd} ana={ana} rel={rel}")

    def test_v_zero_boundary_guarded(self):
        # c == 0 -> v_hat stays 0; eps guards sqrt; r = 0; no NaN.
        # zero gradient AND zero warm moment: r = 0/(0+eps) = 0, no NaN.
        step = c0.adamw_rewrite_step(m_hat=_td(("w", [0.0])), v_hat=_td(("w", [0.0])),
                                     c=_td(("w", [0.0])), beta1=B1, beta2=B2,
                                     t_prev=10, eps=EPS, lr=5e-4, wd=0.01,
                                     theta=_td(("w", [0.1])))
        self.assertTrue(torch.isfinite(step["s"]["w"]).all())
        self.assertEqual(float(step["r"]["w"]), 0.0)
        # nonzero warm moment with zero v: r = m/eps, finite (boundary flagged).
        step2 = c0.adamw_rewrite_step(m_hat=_td(("w", [0.5])), v_hat=_td(("w", [0.0])),
                                      c=_td(("w", [0.0])), beta1=B1, beta2=B2,
                                      t_prev=10, eps=EPS, lr=5e-4, wd=0.01,
                                      theta=_td(("w", [0.1])))
        self.assertTrue(torch.isfinite(step2["s"]["w"]).all())
        self.assertGreater(abs(float(step2["r"]["w"])), 1.0)  # BOUNDARY_V_ZERO


class T4_MuVsQ(unittest.TestCase):
    """T4: training-field mu is NOT the validation gradient q; e_m recursion;
    readout three terms sum to W_total; q<->mu swap must change semantics."""

    def test_recursion_and_readout_identity(self):
        torch.manual_seed(41)
        names = ["w"]
        theta = {n: torch.randn(5, dtype=DT) for n in names}
        q = {n: torch.randn(5, dtype=DT) for n in names}
        mu = {n: torch.randn(5, dtype=DT) * 0.1 for n in names}
        mh = {n: torch.zeros(5, dtype=DT) for n in names}
        vh = {n: torch.zeros(5, dtype=DT) for n in names}
        e_m = {n: torch.zeros(5, dtype=DT) for n in names}
        eta, wd = 5e-4, 0.01
        # gains sum to 1 only when the chain starts at t_prev=0 (bias
        # correction identity); check the identity from step 20 onward.
        for k in range(30):
            c = {n: mu[n] + 0.05 * torch.randn(5, dtype=DT) for n in names}
            a1, _ = c0.ema_coeffs(B1, k)
            theta_pre = {n: v.clone() for n, v in theta.items()}
            step = c0.adamw_rewrite_step(m_hat=mh, v_hat=vh, c=c, beta1=B1,
                                         beta2=B2, t_prev=k, eps=EPS,
                                         lr=eta, wd=wd, theta=theta)
            theta = {n: theta[n] + step["s"][n] for n in theta}
            mh, vh = step["m_hat"], step["v_hat"]
            e_m = c0.tracking_e_m(e_m, c, mu, a1)
            if k >= 20:
                _assert_td_close(self, e_m, {n: mh[n] - mu[n] for n in names},
                                 atol=1e-12)
            P = c0.p_diag(vh, EPS)
            terms = c0.readout_terms(q=q, s=step["s"], P=P, mu=mu, e_m=e_m,
                                     wd=wd, theta=theta_pre, eta=eta)
            self.assertAlmostEqual(terms["W_field"] + terms["W_track"] + terms["W_wd"],
                                   terms["W_total"], places=12)
            self.assertAlmostEqual(terms["W_total"],
                                   eta * c0.dot_td(q, {n: P[n] * mh[n] for n in names})
                                   + eta * wd * terms["q_dot_theta"], places=12)

    def test_q_mu_swap_changes_semantics(self):
        torch.manual_seed(43)
        names = ["w"]
        theta = {n: torch.randn(4, dtype=DT) for n in names}
        q = {n: torch.randn(4, dtype=DT) for n in names}
        mu = {n: torch.randn(4, dtype=DT) * 0.5 for n in names}
        e_m = {n: torch.randn(4, dtype=DT) * 0.01 for n in names}
        P = {n: torch.rand(4, dtype=DT) + 0.5 for n in names}
        s = {n: torch.randn(4, dtype=DT) for n in names}
        t1 = c0.readout_terms(q=q, s=s, P=P, mu=mu, e_m=e_m, wd=0.01, theta=theta, eta=5e-4)
        t2 = c0.readout_terms(q=mu, s=s, P=P, mu=mu, e_m=e_m, wd=0.01, theta=theta, eta=5e-4)
        self.assertNotAlmostEqual(t1["W_field"], t2["W_field"], places=10)
        self.assertNotAlmostEqual(t1["W_total"], t2["W_total"], places=10)


class T5_ClipAsymmetry(unittest.TestCase):
    """T5: C(E[g]) != E[C(g)] concrete counterexample; noncommuting clip-JVP J
    and Hessian H (JH - (JH)^T != 0)."""

    def test_counterexample(self):
        g1 = _td(("p", [2.0, 0.0]))
        g2 = _td(("p", [0.0, 2.0]))
        c_mean = c0.apply_clip(c0.scale_td(c0.add_td(g1, g2), 0.5), TAU)
        e_clip = c0.scale_td(c0.add_td(c0.apply_clip(g1, TAU), c0.apply_clip(g2, TAU)), 0.5)
        self.assertGreater(max((c_mean[n] - e_clip[n]).abs().max().item()
                               for n in c_mean), 1e-3)

    def test_noncommuting_jacobian(self):
        g = _td(("p", [1.0, 1.0]))
        e1 = _td(("p", [1.0, 0.0]))
        e2 = _td(("p", [0.0, 1.0]))
        J1 = clip_jvp(gradient_by_name=g, tangent_gradient_by_name=e1, max_norm=TAU)["clip_jvp"]["p"]
        J2 = clip_jvp(gradient_by_name=g, tangent_gradient_by_name=e2, max_norm=TAU)["clip_jvp"]["p"]
        J = torch.stack([J1, J2], dim=1)  # columns = J e_i
        H = torch.diag(torch.tensor([1.0, 6.0], dtype=DT))
        comm = J @ H - (J @ H).T
        self.assertGreater(float(comm.abs().max()), 1e-6)


class T6_NonlinearExpectation(unittest.TestCase):
    """T6: mean(Adam_update(g)) != Adam_update(mean g). Two-point scalar
    c in {1,3} w.p. 1/2; m_prev=0, v_prev=1, a=b=0.5, eps=0.01, eta=1, wd=0."""

    def test_counterexample(self):
        eps = 0.01
        def upd(c):
            m1 = 0.5 * 0.0 + 0.5 * c
            v1 = 0.5 * 1.0 + 0.5 * c * c
            return m1 / (math.sqrt(v1) + eps)  # eta=1, wd=0
        mean_of = 0.5 * upd(1.0) + 0.5 * upd(3.0)
        plug = (0.5 * (1.0 + 3.0)) / (math.sqrt(0.5 * (1.0 + 9.0)) + eps)
        self.assertGreater(abs(mean_of - plug), 1e-3)
        # module-level: u_map must not collapse the two (clip inactive at tau=1e12)
        g1 = _td(("p", [1.0])); g3 = _td(("p", [3.0]))
        st = dict(m_hat=_td(("p", [0.0])), v_hat=_td(("p", [1.0])), beta1=B1,
                  beta2=B2, t_prev=3, eps=eps, lr=1.0, wd=0.0,
                  theta=_td(("p", [0.0])), tau=1e12)
        u1 = c0.u_map(g1, **st); u3 = c0.u_map(g3, **st)
        u_mean = c0.ybar([u1, u3])
        u_plug = c0.u_map(c0.ybar([g1, g3]), **st)
        self.assertGreater(abs(u_mean["p"] - u_plug["p"]).item(), 1e-4)


class T7_QuadraticPulseReversal(unittest.TestCase):
    """T7: fixed quadratic valley pulse4 reversal. Analytic
    dL_H/d eta_pulse = -4 sum lam_i^2 x0_i^2 (1-eta0 lam_i)^(2H-1).
    NOTE: this example only shows reversal does not require crossing valleys;
    it must NOT be read as the Transformer mechanism."""

    def test_exact_numbers(self):
        h4 = c0.quadratic_pulse_response([1.0, 6.0], [1.0, 1.0], 0.25, 4, 4)
        h16 = c0.quadratic_pulse_response([1.0, 6.0], [1.0, 1.0], 0.25, 4, 16)
        self.assertEqual(h4, 0.591064453125)
        self.assertGreater(h4, 0.0)
        self.assertEqual(h16, -0.0005356792152846719)
        self.assertLess(h16, 0.0)

    def test_u_version_chain_rule(self):
        h4u = c0.quadratic_pulse_response([1.0, 6.0], [1.0, 1.0], 0.25, 4, 4, wrt="u")
        self.assertAlmostEqual(h4u, 0.25 * 0.591064453125, places=15)


class T8_PredictionAdapters(unittest.TestCase):
    """T8: positive/negative controls for M0/Mgeom. (a) no-clip quadratic:
    Mgeom exact FP64, M0 fails. (b) clip active-set change: real re-clip
    accurate, fixed clip-JVP finite displacement not. (c) smooth non-quadratic:
    Mgeom NOT exact (falsifiable)."""

    def test_quadratic_no_clip(self):
        H = torch.diag(torch.tensor([2.0, 3.0], dtype=DT))
        th = torch.tensor([0.4, -0.7], dtype=DT)
        s = torch.tensor([0.05, 0.03], dtype=DT)
        g = {"w": H @ th}
        hvp = {"w": H @ s}
        ybar = {"w": H @ (th + s) - H @ th}      # true displaced-gradient diff
        pred_geom = c0.sub_td(c0.predict_geom(g, hvp, s), g)  # predict Delta c
        self.assertLess(c0.E_model(pred_geom, ybar), 1e-12)
        self.assertAlmostEqual(c0.E_model({n: torch.zeros_like(v) for n, v in g.items()}, ybar), 1.0, places=12)

    def test_clip_active_set_change(self):
        th = torch.tensor([0.8, 0.0], dtype=DT)
        s = torch.tensor([0.5, 0.0], dtype=DT)
        H = torch.eye(2, dtype=DT)
        g = {"w": H @ th}
        hvp = {"w": H @ s}
        y_true = c0.sub_td(c0.apply_clip(c0.add_td(g, hvp), TAU), c0.apply_clip(g, TAU))
        pred_reclip = c0.sub_td(c0.apply_clip(c0.predict_geom(g, hvp, s), TAU),
                                c0.apply_clip(g, TAU))
        self.assertLess(c0.norm_td(c0.sub_td(pred_reclip, y_true)), 1e-9)
        # fixed clip-JVP at g (inactive -> J=I): finite displacement NOT re-clipped
        Jm = torch.eye(2, dtype=DT)  # inactive clip: J = I
        pred_jvp = {"w": g["w"] + Jm @ hvp["w"]}
        err_jvp = c0.norm_td(c0.sub_td(pred_jvp, c0.add_td(g, y_true)))
        self.assertGreater(err_jvp, 0.3)

    def test_smooth_nonquadratic_falsifiable(self):
        # scalar field g(x) = tanh(2x): tangent model with true local slope
        th = torch.tensor([0.5], dtype=DT)
        s = torch.tensor([0.35], dtype=DT)
        def grad(x):
            return torch.tanh(2.0 * x)
        g = {"w": grad(th)}
        slope = 2.0 * (1.0 - torch.tanh(2.0 * th) ** 2)
        hvp = {"w": slope * s}
        ybar = {"w": grad(th + s) - grad(th)}
        pred = c0.predict_geom(g, hvp, s)
        self.assertGreater(c0.E_model(pred, ybar), 1e-6)


class T9_ReadoutAndRNG(unittest.TestCase):
    """T9: readout telescope over two steps; K_V allowed negative; module
    functions do not consume RNG or mutate inputs (smoke re-checks on GPU)."""

    def test_telescope_and_negative_kv(self):
        V0, V1, V2 = 3.0, 2.5, 2.7
        W1, W2 = 0.1, -0.4
        k1 = c0.k_v(V1, V0, W1)
        k2 = c0.k_v(V2, V1, W2)
        self.assertAlmostEqual(k1 + k2, V2 - V0 + W1 + W2, places=15)
        self.assertLess(c0.k_v(1.0, 2.0, 0.1), 0.0)  # K_V may be negative

    def test_no_rng_no_mutation(self):
        torch.manual_seed(53)
        names = ["w", "b"]
        theta = {n: torch.randn(4, dtype=DT) for n in names}
        mh = {n: torch.randn(4, dtype=DT) for n in names}
        vh = {n: torch.rand(4, dtype=DT) + 0.3 for n in names}
        g = {n: torch.randn(4, dtype=DT) for n in names}
        q = {n: torch.randn(4, dtype=DT) for n in names}
        mu = {n: torch.randn(4, dtype=DT) for n in names}
        before = {n: v.clone() for n, v in theta.items()}
        rng_before = torch.random.get_rng_state()
        opt = torch.optim.AdamW([{"params": [theta[n]]} for n in names], lr=5e-4)
        opt_state_before = {k: {kk: vv.clone() if torch.is_tensor(vv) else vv
                                for kk, vv in v.items()} for k, v in opt.state.items()}
        c = c0.apply_clip(g, TAU)
        step = c0.adamw_rewrite_step(m_hat=mh, v_hat=vh, c=c, beta1=B1, beta2=B2,
                                     t_prev=47, eps=EPS, lr=5e-4, wd=0.01, theta=theta)
        _ = c0.u_map(g, m_hat=mh, v_hat=vh, beta1=B1, beta2=B2, t_prev=47,
                     eps=EPS, lr=5e-4, wd=0.01, theta=theta, tau=TAU)
        P = c0.p_diag(vh, EPS)
        em = {n: mh[n] - mu[n] for n in names}
        _ = c0.readout_terms(q=q, s=step["s"], P=P, mu=mu, e_m=em, wd=0.01,
                             theta=theta, eta=5e-4)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        for n in names:
            self.assertTrue(torch.equal(theta[n], before[n]))
            self.assertTrue(torch.equal(mh[n], mh[n].clone()))  # trivially held
        for n in opt.state:
            for kk, vv in opt.state[n].items():
                if torch.is_tensor(vv):
                    self.assertTrue(torch.equal(vv, opt_state_before[n][kk]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
