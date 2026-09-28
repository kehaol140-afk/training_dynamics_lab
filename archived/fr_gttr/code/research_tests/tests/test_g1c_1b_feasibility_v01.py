"""Unit tests for FR_G1C_1B_FEASIBILITY_GATE_V01 adapter code (pure, no GPU).

Run from repo root:
  PYTHONPATH=research/src /home/b205/miniforge3/envs/fr_minimind/bin/python -m \
      unittest research.tests.test_g1c_1b_feasibility_v01 -v
"""
from __future__ import annotations

import unittest

import torch

from fr.a0c_adamw_response_propagation_v01 import (
    adamw_tangent_step, clip_jvp, project_tangent,
)
from fr.g1c_olmo2_architecture import G1COLMo2, chunked_mean_ce
from fr.g1c_olmo2_tangent import g1c_adamw_tangent_step

torch.manual_seed(0)


def _toy_params():
    return [torch.nn.Parameter(torch.randn(5, 4)),
            torch.nn.Parameter(torch.randn(7)),
            torch.nn.Parameter(torch.randn(3, 3))]


def _named(params, names=("a", "b", "c")):
    return list(zip(names, params))


def _fill_state(optimizer, params, step=3.0):
    for p in params:
        st = optimizer.state[p]
        st["step"] = torch.tensor(step)
        st["exp_avg"] = torch.randn_like(p) * 0.1
        st["exp_avg_sq"] = torch.rand_like(p) * 0.01 + 0.001


def _tangents(names, params):
    return (
        {n: torch.randn_like(p) for n, p in zip(names, params)},
        {n: torch.randn_like(p) * 0.1 for n, p in zip(names, params)},
        {n: torch.randn_like(p) * 0.01 for n, p in zip(names, params)},
    )


class TestAdapterMatchesFrozen(unittest.TestCase):
    def test_adapter_equals_frozen_on_homogeneous_group(self):
        params = _toy_params()
        opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                                weight_decay=0.1)
        _fill_state(opt, params)
        names = [n for n, _ in _named(params)]
        g = {n: torch.randn_like(p) for n, p in _named(params)}
        dg = {n: torch.randn_like(p) for n, p in _named(params)}
        clip = clip_jvp(gradient_by_name=g, tangent_gradient_by_name=dg, max_norm=1.0)
        dtheta, dm, dv = _tangents(names, params)
        kwargs = dict(
            named_parameters=_named(params), optimizer=opt,
            clipped_gradient_by_name=clip["clipped_gradient"],
            tangent_clipped_gradient_by_name=clip["clip_jvp"],
            tangent_theta_by_name=dtheta, tangent_m_by_name=dm, tangent_v_by_name=dv,
        )
        frozen = adamw_tangent_step(**kwargs)
        adapted = g1c_adamw_tangent_step(**kwargs)
        for key in ("tangent_theta", "tangent_m", "tangent_v"):
            for name in frozen[key]:
                self.assertTrue(torch.equal(frozen[key][name], adapted[key][name]), key)
        for key in ("tangent_theta_norm", "tangent_m_norm", "tangent_v_norm"):
            self.assertEqual(float(frozen[key]), float(adapted[key]), key)
        self.assertEqual(len(adapted["per_group"]), 1)
        g0 = adapted["per_group"][0]
        self.assertEqual(g0["weight_decay_values"], [0.1])
        self.assertEqual(g0["step_after_values"], [float(frozen["optimizer_step_after"])])
        self.assertEqual(g0["lr_values"], [float(frozen["lr"])])
        self.assertEqual(g0["beta1_values"], [float(frozen["beta1"])])
        self.assertEqual(g0["beta2_values"], [float(frozen["beta2"])])
        self.assertEqual(g0["adam_eps_values"], [float(frozen["adam_eps"])])
        self.assertEqual(g0["weight_decay_values"], [float(frozen["weight_decay"])])


def _naive_reference(*, params, names, optimizer, clipped, dclipped, dtheta, dm, dv):
    out_theta, out_m, out_v = {}, {}, {}
    group_of = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            group_of[p] = group
    for name, p in zip(names, params):
        group = group_of[p]
        beta1, beta2 = group["betas"]
        lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
        st = optimizer.state[p]
        step = float(st["step"]) + 1.0
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        m, v = st["exp_avg"], st["exp_avg_sq"]
        c, dc = clipped[name], dclipped[name]
        m1 = beta1 * m + (1.0 - beta1) * c
        v1 = beta2 * v + (1.0 - beta2) * c * c
        dm1 = beta1 * dm[name] + (1.0 - beta1) * dc
        dv1 = beta2 * dv[name] + 2.0 * (1.0 - beta2) * c * dc
        mhat, vhat = m1 / bc1, v1 / bc2
        dmhat, dvhat = dm1 / bc1, dv1 / bc2
        sq = torch.sqrt(vhat)
        den = sq + eps
        dsq = torch.where(sq > 0, dvhat / (2.0 * sq), torch.zeros_like(dvhat))
        jvp = dmhat / den - mhat * dsq / (den * den)
        out_theta[name] = (1.0 - lr * wd) * dtheta[name] - lr * jvp
        out_m[name] = dm1
        out_v[name] = dv1
    return out_theta, out_m, out_v


class TestAdapterHeterogeneous(unittest.TestCase):
    def test_adapter_equals_naive_on_heterogeneous_groups(self):
        params = _toy_params()
        opt = torch.optim.AdamW(
            [{"params": [params[0]], "weight_decay": 0.1},
             {"params": [params[1], params[2]], "weight_decay": 0.0}],
            lr=2e-3, betas=(0.85, 0.95), eps=1e-7)
        _fill_state(opt, params)
        names = [n for n, _ in _named(params)]
        g = {n: torch.randn_like(p) for n, p in _named(params)}
        dg = {n: torch.randn_like(p) for n, p in _named(params)}
        clip = clip_jvp(gradient_by_name=g, tangent_gradient_by_name=dg, max_norm=0.5)
        dtheta, dm, dv = _tangents(names, params)
        adapted = g1c_adamw_tangent_step(
            named_parameters=_named(params), optimizer=opt,
            clipped_gradient_by_name=clip["clipped_gradient"],
            tangent_clipped_gradient_by_name=clip["clip_jvp"],
            tangent_theta_by_name=dtheta, tangent_m_by_name=dm, tangent_v_by_name=dv,
        )
        ref_t, ref_m, ref_v = _naive_reference(
            params=params, names=names, optimizer=opt,
            clipped=clip["clipped_gradient"], dclipped=clip["clip_jvp"],
            dtheta=dtheta, dm=dm, dv=dv)
        for n in names:
            self.assertTrue(torch.equal(adapted["tangent_theta"][n], ref_t[n]), n)
            self.assertTrue(torch.equal(adapted["tangent_m"][n], ref_m[n]), n)
            self.assertTrue(torch.equal(adapted["tangent_v"][n], ref_v[n]), n)
        self.assertEqual(adapted["per_group"][0]["weight_decay_values"], [0.1])
        self.assertEqual(adapted["per_group"][1]["weight_decay_values"], [0.0])


def _tiny_cfg():
    return dict(d_model=64, n_heads=4, n_layers=2, mlp_hidden=128,
                vocab_size=97, embedding_size=100, eps=1e-6,
                rope_theta=500000.0, max_seq=64, qk_norm_dim=16)


class TestOptimizerRebuildStepable(unittest.TestCase):
    def test_build_optimizer_rebuilt_groups_are_stepable(self):
        import tempfile
        from fr.g1c_olmo2_checkpoint import build_optimizer

        class _ModB(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(4))

        def _tiny_model():
            m = torch.nn.Module()
            m.a = torch.nn.Linear(3, 3, bias=False)
            m.b = _ModB()
            return m

        torch.manual_seed(7)
        p = [torch.nn.Parameter(torch.randn(3, 3)), torch.nn.Parameter(torch.randn(4))]
        opt = torch.optim.AdamW([{"params": [p[0]], "weight_decay": 0.1},
                                 {"params": [p[1]], "weight_decay": 0.0}],
                                lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
        for pi in p:
            pi.grad = torch.randn_like(pi)
        opt.step()  # materialize state
        sd = opt.state_dict()
        names = ["a.weight", "b.weight"]
        for group, name in zip(sd["param_groups"], names):
            group["param_names"] = [name]
        sd["state"] = {i: sd["state"][i] for i in (0, 1)}
        with tempfile.NamedTemporaryFile(suffix=".pt") as fh:
            torch.save(sd, fh.name)
            opt2, report = build_optimizer(fh.name, _tiny_model())
        for pi in opt2.param_groups[0]["params"] + opt2.param_groups[1]["params"]:
            pi.grad = torch.randn_like(pi)
        opt2.step()  # must not raise KeyError on group keys
        self.assertEqual(report["n_state_entries"], 2)


class TestArchitecture(unittest.TestCase):
    def test_keys_match_official_layout(self):
        cfg = _tiny_cfg()
        model = G1COLMo2(d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                         n_layers=cfg["n_layers"], mlp_hidden=cfg["mlp_hidden"],
                         vocab_size=cfg["vocab_size"], embedding_size=cfg["embedding_size"],
                         eps=cfg["eps"], rope_theta=cfg["rope_theta"],
                         max_seq=cfg["max_seq"], qk_norm_dim=cfg["qk_norm_dim"])
        keys = set(model.state_dict().keys())
        for i in range(cfg["n_layers"]):
            for sub in ("att_proj.weight", "attn_out.weight", "q_norm.weight",
                        "k_norm.weight", "attn_norm.weight", "ff_norm.weight",
                        "ff_proj.weight", "ff_out.weight"):
                self.assertIn(f"transformer.blocks.{i}.{sub}", keys)
        self.assertIn("transformer.wte.weight", keys)
        self.assertIn("transformer.ln_f.weight", keys)
        self.assertIn("transformer.ff_out.weight", keys)

    def test_forward_backward_double_backward(self):
        torch.manual_seed(1)
        cfg = _tiny_cfg()
        model = G1COLMo2(d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                         n_layers=cfg["n_layers"], mlp_hidden=cfg["mlp_hidden"],
                         vocab_size=cfg["vocab_size"], embedding_size=cfg["embedding_size"],
                         eps=cfg["eps"], rope_theta=cfg["rope_theta"],
                         max_seq=cfg["max_seq"], qk_norm_dim=cfg["qk_norm_dim"])
        tokens = torch.randint(0, cfg["vocab_size"], (2, 33))
        ids = tokens[:, :-1]
        targets = tokens[:, 1:].contiguous()
        hidden = model.forward_hidden(ids, checkpoint_blocks=True)
        loss = chunked_mean_ce(hidden, model.transformer.ff_out.weight, targets, chunk=16)
        self.assertTrue(torch.isfinite(loss))
        params = list(model.parameters())
        grads = torch.autograd.grad(loss, params, create_graph=True)
        for g in grads:
            self.assertTrue(torch.isfinite(g).all())
        qvec = [torch.randn_like(p) for p in params]
        hvp = torch.autograd.grad(sum((g * q).sum() for g, q in zip(grads, qvec)),
                                  params, retain_graph=False)
        for h in hvp:
            self.assertTrue(torch.isfinite(h).all())
        plain = torch.nn.functional.cross_entropy(
            model(ids).float().reshape(-1, cfg["embedding_size"]), targets.reshape(-1))
        self.assertLess(abs(float(loss) - float(plain)), 1e-5)

    def test_project_tangent_imported_frozen(self):
        q = {"a": torch.randn(4, 4)}
        t = {"a": torch.randn(4, 4)}
        val = project_tangent(q, t)
        ref = float((q["a"].double() * t["a"].double()).sum())
        self.assertLess(abs(val - ref), 1e-12)


class TestHvpChunkingMath(unittest.TestCase):
    """Amendment A5: batch-dim chunked double backward must equal the direct
    full-batch HVP of the mean-CE objective (equal-weight linear split)."""

    def _loss(self, model, ids, targets):
        return chunked_mean_ce(model.forward_hidden(ids, checkpoint_blocks=False),
                               model.transformer.ff_out.weight, targets, chunk=16,
                               recompute_logits=False)

    def _hvp_direct(self, model, ids, targets, dtheta):
        params = list(model.parameters())
        loss = self._loss(model, ids, targets)
        g = torch.autograd.grad(loss, params, create_graph=True)
        dot = sum((gi * dti).sum() for gi, dti in zip(g, dtheta))
        return torch.autograd.grad(dot, params, retain_graph=False)

    def _hvp_chunked(self, model, ids, targets, dtheta, chunk_seqs):
        params = list(model.parameters())
        acc = None
        n = 0
        for start in range(0, ids.shape[0], chunk_seqs):
            ids_c = ids[start:start + chunk_seqs].contiguous()
            tg_c = targets[start:start + chunk_seqs].contiguous()
            loss = self._loss(model, ids_c, tg_c)
            g = torch.autograd.grad(loss, params, create_graph=True)
            dot = sum((gi * dti).sum() for gi, dti in zip(g, dtheta))
            hvp = torch.autograd.grad(dot, params, retain_graph=False)
            acc = hvp if acc is None else [a + h for a, h in zip(acc, hvp)]
            n += 1
        return [a / float(n) for a in acc]

    def test_chunked_equals_direct(self):
        torch.manual_seed(11)
        cfg = _tiny_cfg()
        model = G1COLMo2(d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                         n_layers=cfg["n_layers"], mlp_hidden=cfg["mlp_hidden"],
                         vocab_size=cfg["vocab_size"], embedding_size=cfg["embedding_size"],
                         eps=cfg["eps"], rope_theta=cfg["rope_theta"],
                         max_seq=cfg["max_seq"], qk_norm_dim=cfg["qk_norm_dim"])
        tokens = torch.randint(0, cfg["vocab_size"], (4, 33))
        ids = tokens[:, :-1]
        targets = tokens[:, 1:].contiguous()
        dtheta = [torch.randn_like(p) for p in model.parameters()]
        ref = self._hvp_direct(model, ids, targets, dtheta)
        for chunk_seqs in (1, 2):
            got = self._hvp_chunked(model, ids, targets, dtheta, chunk_seqs)
            for gr, gg in zip(ref, got):
                self.assertLess(float((gr - gg).abs().max()), 1e-5,
                                f"chunk_seqs={chunk_seqs}")


if __name__ == "__main__":
    unittest.main()
