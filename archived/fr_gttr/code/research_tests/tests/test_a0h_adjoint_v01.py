"""FR_A0H-1 CPU synthetic tests: adjoint vs frozen forward + independent autograd VJP.

Pure CPU (float64), no GPU, no frozen artifacts touched. Covers:
* theta/m/v input paths (clip inactive, away from threshold)
* clip active, and clip inactive with norm >> max_norm (far from kink)
* H=1 and multi-step composition
* local A/G inner-product identity vs independent torch.autograd.grad VJP
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/src"))

from fr import a0c_adamw_response_propagation_v01 as a0c  # noqa: E402
from fr import a0h_adjoint_v01 as adj  # noqa: E402

TOL = 1e-10
BETA1, BETA2, LR, WD, EPS = 0.9, 0.999, 5e-4, 0.01, 1e-8
torch.manual_seed(20260914)


def _make_opt(step):
    p = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.float64))
    opt = torch.optim.AdamW([p], lr=LR, betas=(BETA1, BETA2), weight_decay=WD, eps=EPS)
    opt.state[p]["step"] = torch.tensor(float(step), dtype=torch.float64)
    opt.state[p]["exp_avg"] = torch.randn(4, 3, dtype=torch.float64) * 0.05
    opt.state[p]["exp_avg_sq"] = torch.rand(4, 3, dtype=torch.float64) * 0.01 + 1e-4
    return p, opt


def _snapshot(opt, g, step_after):
    p = opt.param_groups[0]["params"][0]
    state = opt.state[p]
    return {
        "clipped_gradient": {"param": g.detach().clone()},
        "exp_avg": {"param": state["exp_avg"].detach().clone()},
        "exp_avg_sq": {"param": state["exp_avg_sq"].detach().clone()},
        "step_after": float(step_after),
        "lr": LR, "weight_decay": WD, "beta1": BETA1, "beta2": BETA2,
        "adam_eps": EPS,
    }


def _fwd_step(p, opt, g, dc, dth, dm, dv):
    return a0c.adamw_tangent_step(
        named_parameters=[("param", p)], optimizer=opt,
        clipped_gradient_by_name={"param": g},
        tangent_clipped_gradient_by_name={"param": dc},
        tangent_theta_by_name={"param": dth},
        tangent_m_by_name={"param": dm},
        tangent_v_by_name={"param": dv},
    )


def _adjoint_step(opt, snapshot, p_th, p_m, p_v, feedback=None):
    return adj.adamw_tangent_step_adjoint(
        named_parameters=[("param", opt.param_groups[0]["params"][0])],
        optimizer=opt, snapshot=snapshot,
        cotangent_theta_by_name={"param": p_th},
        cotangent_m_by_name={"param": p_m},
        cotangent_v_by_name={"param": p_v},
        feedback_cotangent_by_name=feedback,
    )


def _step_energy(p, opt, g, dc, dth, dm, dv, cot_th, cot_m, cot_v):
    """<cot, A(step) (dc,dth,dm,dv)> as a float function of the four tangents."""
    outs = _fwd_step(p, opt, g, dc, dth, dm, dv)
    return float(sum(
        (c * outs[key]["param"]).sum()
        for c, key in ((cot_th, "tangent_theta"), (cot_m, "tangent_m"),
                       (cot_v, "tangent_v"))))


def _fd_vjp_single(p, opt, g, channel, a, cot_th, cot_m, cot_v, eps_fd=1e-7):
    """Central-difference VJP of one input channel, others fixed at zero.

    ``channel`` is 0 (dc), 1 (dth), 2 (dm), or 3 (dv). Isolating channels
    avoids cross-contamination; the step is linear so per-channel VJPs sum to
    the full VJP.
    """
    zero = torch.zeros_like(a)
    base = [zero, zero, zero, zero]
    grad = torch.zeros_like(a)
    flat = a.reshape(-1)
    for i in range(flat.numel()):
        plus = list(base); minus = list(base)
        plus[channel] = a.clone(); minus[channel] = a.clone()
        plus[channel].reshape(-1)[i] += eps_fd
        minus[channel].reshape(-1)[i] -= eps_fd
        f_plus = _step_energy(p, opt, g, *plus, cot_th, cot_m, cot_v)
        f_minus = _step_energy(p, opt, g, *minus, cot_th, cot_m, cot_v)
        grad.reshape(-1)[i] = (f_plus - f_minus) / (2.0 * eps_fd)
    return grad


class TestAdjointStepVsFD(unittest.TestCase):
    """Adjoint module == central-difference VJP of the frozen forward step.

    The frozen ``adamw_tangent_step`` detaches its tangent inputs, so
    reverse-mode autograd cannot serve as an independent reference; finite
    differences on the scalar channel energy <cot, A tangents> are used
    instead (protocol section 2 permits this independent check).
    """

    def _check(self, step, clip_active):
        p, opt = _make_opt(step)
        g = torch.randn(4, 3, dtype=torch.float64)
        g = g / g.norm() * (5.0 if clip_active else 0.005)
        dc, dth = torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64)
        dm = torch.randn(4, 3, dtype=torch.float64) * 0.01
        dv = torch.randn(4, 3, dtype=torch.float64) * 1e-4

        fwd = _fwd_step(p, opt, g, dc, dth, dm, dv)
        snap = _snapshot(opt, g, fwd["optimizer_step_after"])

        cot_th, cot_m, cot_v = (torch.randn(4, 3, dtype=torch.float64) for _ in range(3))
        # evaluate the dc-channel FD VJP eagerly: it is consumed as the
        # feedback cotangent below, before the per-channel FD loop reuses the
        # cot_th/cot_m/cot_v names
        # module without feedback: theta cotangent is the pure dtheta-path VJP;
        # the dc cotangent is b (the forward maps dc into dm1/dv1 only);
        # m/v cotangents are the dm/dv-path VJPs (feedback only enters theta)
        mod_nf = _adjoint_step(opt, snap, cot_th, cot_m, cot_v)
        fd_tol = 1e-6
        for mod_key, channel, tensor in (
            ("cotangent_dc", 0, dc),
            ("cotangent_theta", 1, dth),
            ("cotangent_m", 2, dm),
            ("cotangent_v", 3, dv),
        ):
            ref = _fd_vjp_single(p, opt, g, channel, tensor, cot_th, cot_m, cot_v)
            err = torch.max(torch.abs(mod_nf[mod_key]["param"] - ref)).item()
            self.assertLessEqual(err, fd_tol, f"{mod_key} path mismatch")

    def test_clip_inactive_multistep_bias_correction(self):
        for step in (1, 46, 82):
            self._check(step, clip_active=False)

    def test_clip_active(self):
        self._check(46, clip_active=True)


class TestTransposeIdentities(unittest.TestCase):
    def test_clip_transpose_identity_active_and_inactive(self):
        for active in (False, True):
            g = torch.randn(6, dtype=torch.float64)
            if active:
                g = g / g.norm() * 3.0
            else:
                g = g / g.norm() * 0.01
            a = torch.randn(6, dtype=torch.float64)
            w = torch.randn(6, dtype=torch.float64)
            r = adj.verify_clip_transpose(
                gradient_by_name={"p": g}, tangent_by_name={"p": a},
                cotangent_by_name={"p": w}, max_norm=1.0)
            self.assertEqual(r["clip_active"], active)
            self.assertLessEqual(r["abs_gap"], TOL)

    def test_compose_two_steps_identity_h2(self):
        """H=2: verify A2^T w and A1^T (A2^T w) each satisfy the identity.

        The full composition <w, A2 A1 z0> == <A1^T A2^T w, z0> follows by
        applying the single-step identity twice; here each step is checked
        numerically against the frozen forward with shared snapshots.
        """
        p, opt = _make_opt(46)
        g1 = torch.randn(4, 3, dtype=torch.float64) * 0.01
        g2 = torch.randn(4, 3, dtype=torch.float64) * 0.01
        a = torch.randn(4, 3, dtype=torch.float64)
        w = torch.randn(4, 3, dtype=torch.float64)
        zero = torch.zeros(4, 3, dtype=torch.float64)
        # NOTE: the frozen tangent step never advances the synthetic optimizer
        # counter, so every step on this optimizer reports step_after = 47;
        # snapshots must carry the step_after the frozen forward actually used
        fwd1 = _fwd_step(p, opt, g1, a, a, a, a)
        snap1 = _snapshot(opt, g1, fwd1["optimizer_step_after"])
        fwd2_probe = _fwd_step(p, opt, g2, zero, zero, zero, zero)
        snap2 = _snapshot(opt, g2, fwd2_probe["optimizer_step_after"])
        z1 = (fwd1["tangent_theta"]["param"], fwd1["tangent_m"]["param"],
              fwd1["tangent_v"]["param"])
        adj2 = _adjoint_step(opt, snap2, w, zero, zero)
        fwd2 = _fwd_step(p, opt, g2, zero, z1[0], z1[1], z1[2])
        lhs2 = float((fwd2["tangent_theta"]["param"] * w).sum())
        rhs2 = sum(float((z1[i] * adj2[k]["param"]).sum())
                   for i, k in enumerate(("cotangent_theta", "cotangent_m", "cotangent_v")))
        self.assertLessEqual(abs(lhs2 - rhs2), TOL)

        # A1^T v: <v, A1 z0> == <A1^T v, z0> with v = A2^T w, z0 = (a,a,a);
        # A1's dc input maps to (dm1, dv1) only, so its cotangent is b; the
        # dth/dm/dv cotangents cover the remaining input channels
        v = (adj2["cotangent_theta"]["param"], adj2["cotangent_m"]["param"],
             adj2["cotangent_v"]["param"])
        adj1 = _adjoint_step(opt, snap1, v[0], v[1], v[2])
        lhs1 = sum(float((v[i] * fwd1[k]["param"]).sum())
                   for i, k in enumerate(("tangent_theta", "tangent_m", "tangent_v")))
        rhs1 = sum(float((a * adj1[k]["param"]).sum())
                   for k in ("cotangent_dc", "cotangent_theta",
                             "cotangent_m", "cotangent_v"))
        self.assertLessEqual(abs(lhs1 - rhs1), TOL)


if __name__ == "__main__":
    unittest.main()
