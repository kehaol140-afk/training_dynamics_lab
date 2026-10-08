"""F3 rq3 chunked-runner regression tests (execution-layer rewrite, 2026-09-29).

Validates on a synthetic deterministic fake environment that the chunked /
resumable / priority-ordered execution in f3_curvature_mechanism.run_rq3 is
BIT-IDENTICAL to an independent single-pass reference implementing the frozen
prereg §5 semantics (forced arms: masked dc at alpha=1 + LR source; SO| twins:
SAME masked dc at every h>=2, q0 init at h=1, s_h == 0). Also pins:
- subset (--arms) + resume-later == one-shot full run;
- interrupted-chunk sidecar resume == uninterrupted run;
- SO twins actually propagate beyond h=1 (the 2026-09-28 bug);
- inline _rq3_decomposition == f3.curvature_decomposition.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import f3_curvature_mechanism as M  # noqa: E402
from opsens.a0c_adamw_response_propagation_v01 import clip_jvp  # noqa: E402

# names covering attn_q / attn_k / mlp_gate / embed_tied / rmsnorm groups
NAMES_SHAPES = [
    ("model.layers.0.self_attn.q_proj.weight", (6, 5)),
    ("model.layers.0.self_attn.k_proj.weight", (4, 5)),
    ("model.layers.0.mlp.gate_proj.weight", (7, 6)),
    ("model.embed_tokens.weight", (9, 5)),
    ("model.norm.weight", (6,)),
]


class FakeRig:
    """Deterministic stand-in for Rig: restore() replays identical theta
    trajectories; hvp/val_grad are fixed linear maps (time-varying in h0)."""

    def __init__(self):
        self.k = 0
        gen = torch.Generator().manual_seed(3)
        self.template = [(n, torch.randn(sh, generator=gen, dtype=torch.float64) * 0.1)
                         for n, sh in NAMES_SHAPES]

    def restore(self, state_g):
        self.k = 0
        named = [(n, torch.nn.Parameter(t.clone())) for n, t in self.template]
        g = torch.Generator().manual_seed(42)
        state = {}
        for _, p in named:
            state[p] = {"step": torch.tensor(26.0),
                        "exp_avg": torch.zeros_like(p),
                        "exp_avg_sq": torch.rand(p.shape, generator=g,
                                                 dtype=torch.float64) * 1e-6 + 1e-12}

        class _O:
            pass

        opt = _O()
        opt.param_groups = [{"params": [p for _, p in named], "lr": 3e-4,
                             "betas": (0.9, 0.999), "eps": 1e-8,
                             "weight_decay": 0.01, "amsgrad": False,
                             "maximize": False, "differentiable": False}]
        opt.state = state
        return named, opt, object()

    def hvp(self, model, batches, h0, dtheta, fp32=True):
        scale = 2.0 + 0.01 * h0
        return {n: (scale * t).to(torch.float32) for n, t in dtheta.items()}

    def val_grad(self, model):
        g = torch.Generator().manual_seed(123)
        return {n: torch.randn(p.shape, generator=g, dtype=torch.float64) * 0.01
                for n, p in M.named}


class FakeEnv:
    """Patches the module-level globals run_rq3 depends on."""

    def __init__(self, tmp: Path):
        self.rig = FakeRig()
        self._saved = {k: getattr(M, k) for k in
                       ("rig", "named", "opt", "model_cpu", "batches_g",
                        "state_g", "RAW", "F3LOG", "primal_grads",
                        "primal_advance")}
        M.rig = self.rig
        M.batches_g = object()
        M.state_g = object()
        M.RAW = tmp
        M.RAW.mkdir(parents=True, exist_ok=True)
        M.F3LOG = tmp / "f3_run.log"
        named, opt, model = self.rig.restore(None)
        M.named, M.opt, M.model_cpu = named, opt, model
        M.primal_grads = self._primal_grads
        M.primal_advance = self._primal_advance

    def close(self):
        for k, v in self._saved.items():
            setattr(M, k, v)

    def _primal_grads(self, h0, micro0):
        g = torch.Generator().manual_seed(1000 + h0)
        grads = {n: torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.01
                 for n, p in M.named}
        theta_prev = {n: p.detach().clone() for n, p in M.named}
        clipped = {n: t * 0.9 for n, t in grads.items()}
        loss = 1.0 + 0.01 * h0
        return theta_prev, 3e-4, grads, clipped, loss, {"coef": 0.9, "lr": 3e-4}

    def _primal_advance(self):
        self.rig.k += 1
        for _, p in M.named:
            p.data.add_(0.001 * self.rig.k)
        return {n: p.detach().clone() for n, p in M.named}


def _arm_dc(env, key, dz_th, grads, h):
    """Reference masked dc at alpha=1 (independent of M.dc_for)."""
    from opsens.f3_curvature_mechanism_v01 import apply_mask, module_mask
    src = env.rig.hvp(M.model_cpu, M.batches_g, h - 1,
                      {n: t.to(torch.float32) for n, t in dz_th.items()})
    cj = clip_jvp(gradient_by_name=grads, tangent_gradient_by_name=src,
                  max_norm=M.MAX_NORM)
    dc = {n: t.double() for n, t in cj["clip_jvp"].items()}
    arm = key[3:] if key.startswith("SO|") else key
    return apply_mask(dc, module_mask(M.named, arm))


def _reference_rq3(env):
    """Independent single-pass, all-28-arms implementation of prereg §5."""
    from opsens import f3_curvature_mechanism_v01 as f3
    named, opt, model = env.rig.restore(None)
    M.named, M.opt, M.model_cpu = named, opt, model
    arms = {}
    for arm in M.RQ3_ARMS:
        arms[arm] = {"th": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
                     "m": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
                     "v": {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named},
                     "key": arm}
        arms[f"SO|{arm}"] = {"dz": None, "key": f"SO|{arm}"}
    rows = []
    for h in range(1, M.H_MAX + 1):
        theta_prev, lr, grads, clipped, loss, info = env._primal_grads(h - 1, 0)
        pending = {}
        for key, a in arms.items():
            if key.startswith("SO|"):
                if h == 1:
                    continue
                dc = _arm_dc(env, key, a["dz"]["th"], grads, h)
                pending[key] = f3.hom_step(named, opt, clipped, dc, a["dz"])
            else:
                dc = _arm_dc(env, key, a["th"], grads, h)
                pending[key] = f3.hom_step(named, opt, clipped, dc,
                                           {"th": a["th"], "m": a["m"], "v": a["v"]})
        theta_next = env._primal_advance()
        if h == 1:
            q0 = f3.lr_source_q0(named, theta_prev, theta_next)
            zero = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named}
            for arm in M.RQ3_ARMS:
                key = f"SO|{arm}"
                if arms[key]["dz"] is None:
                    arms[key]["dz"] = {ch: {n: t.clone() for n, t in q0[ch].items()}
                                       for ch in ("th", "m", "v")}
                    pending[key] = f3.hom_step(named, opt, clipped, zero,
                                               arms[key]["dz"])
        inh = {n: (theta_next[n].double() - theta_prev[n].double()) for n, _ in named}
        row = {"h": h, "loss": loss, "clip_coef": info["coef"], "arms": {}}
        for key, a in arms.items():
            if key.startswith("SO|"):
                a["dz"] = pending[key]
                n_th, n_m, n_v, n_z = M.state_norms(a["dz"])
                row["arms"][key] = {"dz_norm": n_z, "dtheta_norm": n_th,
                                    "dm_norm": n_m, "dv_norm": n_v,
                                    "stopped": False}
            else:
                out = pending[key]
                a["th"] = {n: out["th"][n] + inh[n] for n, _ in named}
                a["m"] = out["m"]
                a["v"] = out["v"]
                n_th, n_m, n_v, n_z = M.state_norms(a)
                row["arms"][key] = {"dtheta_norm": n_th, "dm_norm": n_m,
                                    "dv_norm": n_v, "dz_norm": n_z,
                                    "stopped": False}
        rows.append(row)
    return rows


def _compare(raw_rows, ref_rows, tag):
    assert len(raw_rows) == len(ref_rows) == M.H_MAX, tag
    for r, ref in zip(raw_rows, ref_rows):
        assert r["h"] == ref["h"] and r["loss"] == ref["loss"], tag
        for key, ref_rec in ref["arms"].items():
            got = r["arms"][key]
            for f in ref_rec:
                assert got[f] == ref_rec[f], (tag, r["h"], key, f,
                                              got[f], ref_rec[f])


def test_rq3_chunked_bitwise_vs_single_pass_reference(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        env = FakeEnv(Path(td))
        try:
            ref = _reference_rq3(env)
            rc = M.run_rq3(2042, 26, 52)
            assert rc == 0
            raw = json.load((M.RAW / "rq3_2042_26.json").open())
            _compare(raw["steps"], ref, "chunked-vs-reference")
            assert raw["probes"]["q0_lrsource"]
        finally:
            env.close()


def test_rq3_subset_then_full_equals_one_shot(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        env = FakeEnv(Path(td1))
        try:
            M.run_rq3(2042, 26, 52, only_arms=["FULL", "NO_ATTN_Q"])
            partial = json.load((M.RAW / "rq3_2042_26.json").open())
            assert set(partial["meta"]["completed_arms"]) == {
                "FULL", "NO_ATTN_Q", "SO|FULL", "SO|NO_ATTN_Q"}
            # subset run must not fabricate other arms at h=32
            assert "NO_MLP" not in partial["steps"][-1]["arms"]
            M.run_rq3(2042, 26, 52)  # completes the rest
            full = json.load((M.RAW / "rq3_2042_26.json").open())
        finally:
            env.close()
        env2 = FakeEnv(Path(td2))
        try:
            M.run_rq3(2042, 26, 52)
            oneshot = json.load((M.RAW / "rq3_2042_26.json").open())
        finally:
            env2.close()
        _compare(full["steps"], oneshot["steps"], "subset+full-vs-oneshot")
        assert sorted(full["meta"]["completed_arms"]) == \
            sorted(oneshot["meta"]["completed_arms"])


def test_rq3_interrupted_chunk_sidecar_resume(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        # one-shot reference run
        env = FakeEnv(Path(td2))
        try:
            M.run_rq3(2043, 26, 52)
            oneshot = json.load((M.RAW / "rq3_2043_26.json").open())
        finally:
            env.close()
        # interrupted run: save_state blows up at done==12 (mid first chunk)
        env = FakeEnv(Path(td1))
        orig_save_state = M.save_state
        try:
            def bomb(name, arms_state, done):
                if done == 12 and not getattr(bomb, "armed", False):
                    bomb.armed = True
                    # sidecar for h=12 must exist for a realistic resume:
                    # emulate "crashed right after checkpointing"
                    orig_save_state(name, arms_state, done)
                    raise RuntimeError("simulated interruption")
                return orig_save_state(name, arms_state, done)
            M.save_state = bomb
            try:
                M.run_rq3(2043, 26, 52)
                raise AssertionError("expected simulated interruption")
            except RuntimeError as e:
                assert "simulated" in str(e)
            M.save_state = orig_save_state
            M.run_rq3(2043, 26, 52)  # resume
            resumed = json.load((M.RAW / "rq3_2043_26.json").open())
        finally:
            M.save_state = orig_save_state
            env.close()
        _compare(resumed["steps"], oneshot["steps"], "sidecar-resume-vs-oneshot")


def test_rq3_so_twins_propagate_beyond_h1():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        env = FakeEnv(Path(td))
        try:
            M.run_rq3(2042, 26, 52, only_arms=["FULL"])
            raw = json.load((M.RAW / "rq3_2042_26.json").open())
            n1 = raw["steps"][0]["arms"]["SO|FULL"]["dz_norm"]
            n2 = raw["steps"][1]["arms"]["SO|FULL"]["dz_norm"]
            n32 = raw["steps"][-1]["arms"]["SO|FULL"]["dz_norm"]
            assert n2 != n1, "SO twin must take a masked-dc step at h=2"
            assert n32 != n1
            # forced twin present at every h; sidecar released on full-arm
            # completion only (here only FULL subset ran -> sidecar kept)
            assert "FULL" in raw["steps"][-1]["arms"]
        finally:
            env.close()


def test_rq3_decomposition_inline_matches_library():
    import tempfile
    from opsens import f3_curvature_mechanism_v01 as f3
    with tempfile.TemporaryDirectory() as td:
        env = FakeEnv(Path(td))
        try:
            g = torch.Generator().manual_seed(7)
            dc = {n: torch.randn(p.shape, generator=g, dtype=torch.float64) * 1e-3
                  for n, p in M.named}
            clipped = {n: t.float() for n, t in dc.items()}
            dz = {"th": {n: torch.randn(p.shape, generator=g, dtype=torch.float64) * 1e-3
                         for n, p in M.named} for _ in (0,)}
            dz["m"] = {n: torch.randn(p.shape, generator=g, dtype=torch.float64) * 1e-3
                       for n, p in M.named}
            dz["v"] = {n: torch.randn(p.shape, generator=g, dtype=torch.float64) * 1e-3
                       for n, p in M.named}
            ref = f3.curvature_decomposition(M.named, M.opt, clipped, dc, dz)
            out_ad = f3.hom_step(M.named, M.opt, clipped, dc, dz)
            out_a = f3.hom_step(M.named, M.opt,
                                clipped,
                                {n: torch.zeros_like(t) for n, t in dc.items()},
                                dz)
            got = M._rq3_decomposition(out_a, out_ad)
            for k in ref:
                assert got[k] == ref[k], (k, got[k], ref[k])
        finally:
            env.close()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} PASS")
