"""Dry-run exercise of F1 stage 3/4/5 scripts on a synthetic tiny OLMo-shaped
checkpoint (payload-format faithful: param_groups with FSDP-prefixed
param_names, state-by-id, fp32 m/v/step). Catches script-level runtime bugs
(field mismatches, key errors, dtype issues) BEFORE the real 17.8G download.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import opsens.f1_1b_replication_v01 as f1
from opsens.g1c_olmo2_architecture import build_from_config

TINY_CFG = {"d_model": 64, "n_heads": 4, "n_layers": 2, "mlp_hidden_size": 32,
            "vocab_size": 128, "embedding_size": 128, "layer_norm_eps": 1e-5,
            "rope_theta": 10000.0, "max_sequence_length": 64, "qk_norm_dim": 16}


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_synthetic_payload(model, with_zero_v=True):
    torch.manual_seed(123)
    named = list(model.named_parameters())
    pid = 0
    group0_names, group1_names, state = [], [], {}
    for name, p in named:
        entry = {
            "step": torch.tensor(380001.0),
            "exp_avg": (torch.randn(p.shape) * 1e-3).to(torch.float32),
            "exp_avg_sq": (torch.rand(p.shape) * 1e-6 + 1e-12).to(torch.float32),
        }
        if with_zero_v and name == "transformer.wte.weight":
            entry["exp_avg_sq"][5:9] = 0.0
            entry["exp_avg"][5:9] = 0.0
        state[pid] = entry
        (group1_names if name == "transformer.wte.weight" else group0_names).append(
            (pid, f"_fsdp_wrapped_module.{name}"))
        pid += 1
    payload = {
        "param_groups": [
            {"params": [p for p, _ in group0_names],
             "param_names": [n for _, n in group0_names],
             "lr": 3.78285e-4, "betas": [0.9, 0.95], "eps": 1e-8,
             "weight_decay": 0.1, "initial_lr": 4e-4, "max_grad_norm": 1.0,
             "amsgrad": False},
            {"params": [p for p, _ in group1_names],
             "param_names": [n for _, n in group1_names],
             "lr": 3.78285e-4, "betas": [0.9, 0.95], "eps": 1e-8,
             "weight_decay": 0.0, "initial_lr": 4e-4, "max_grad_norm": 1.0,
             "amsgrad": False},
        ],
        "state": state,
    }
    return payload


def main():
    tmp = Path(tempfile.mkdtemp(prefix="f1_synth_"))
    ext = tmp / "peteish1"
    ext.mkdir()
    out_dir = tmp / "out"
    out_dir.mkdir()

    model = build_from_config(TINY_CFG)
    payload = build_synthetic_payload(model)
    torch.save(payload, ext / "step1_optim.pt")
    torch.save({k: v for k, v in model.state_dict().items()}, ext / "step1_model.pt")
    (ext / "step1_config.yaml").write_text("model: {d_model: 64}\n")

    orig = {
        "STEPS": f1.STEPS, "EXT": f1.EXT, "OUT": f1.OUT,
        "CHECKPOINT_SHA256": dict(f1.CHECKPOINT_SHA256),
        "CHECKPOINT_SIZES": dict(f1.CHECKPOINT_SIZES),
        "load_model_cfg": f1.load_model_cfg,
    }
    f1.STEPS = (1,)
    f1.EXT = ext
    f1.OUT = out_dir
    f1.CHECKPOINT_SHA256 = {(1, "optim.pt"): "x", (1, "model.pt"): "x"}
    f1.CHECKPOINT_SIZES = {(1, "optim.pt"): (ext / "step1_optim.pt").stat().st_size,
                           (1, "model.pt"): (ext / "step1_model.pt").stat().st_size}
    f1.load_model_cfg = lambda step: TINY_CFG

    def patched_verify(step, name, skip_sha=False):
        p = ext / f"step{step}_{name}"
        return {"path": str(p), "exists": p.exists(),
                "size": p.stat().st_size if p.exists() else -1,
                "size_ok": True, "sha256": "x", "sha_ok": p.exists()}

    try:
        # stage 3
        f1.verify_checkpoint = patched_verify
        s3 = load_script("f1_state_audit")
        s3.OUT = out_dir
        s3.LOG = out_dir / "s3.log"
        rc = s3.main()
        assert rc == 0, "stage3 rc"
        rows = (out_dir / "state_floor_by_module.csv").read_text().strip().splitlines()
        assert len(rows) > 5, len(rows)
        assert "v_exact_zero_fraction" in rows[0]
        print(f"PASS stage3 synthetic ({len(rows)-1} rows)")

        # stage 4
        s4 = load_script("f1_local_conditioning")
        s4.OUT = out_dir
        s4.LOG = out_dir / "s4.log"
        f1.verify_checkpoint = patched_verify
        rc = s4.main()
        assert rc == 0, "stage4 rc"
        lc = (out_dir / "local_conditioning.csv").read_text().strip().splitlines()
        assert "A_total" in lc[0] and len(lc) == 3 * 8 + 1, len(lc)
        pi = (out_dir / "power_iteration_1b.csv").read_text().strip().splitlines()
        assert "sigma_max" in pi[0] and len(pi) == 2
        print(f"PASS stage4 synthetic ({len(lc)-1} rows, sigma={float(pi[1].split(',')[1]):.4g})")

        # stage 5
        s5 = load_script("f1_fd_validation")
        s5.OUT = out_dir
        s5.LOG = out_dir / "s5.log"
        f1.verify_checkpoint = patched_verify
        rc = s5.main()
        assert rc == 0, "stage5 rc"
        fdv = (out_dir / "fd_validation.csv").read_text().strip().splitlines()
        gate = json.loads((out_dir / "fd_gate.json").read_text())
        print(f"PASS stage5 synthetic ({len(fdv)-1} rows)")
        for k, v in gate["per_slice"].items():
            print(f"  {k}: gate64={v['gate64_pass']} best_rel64={v['best_rel64']:.3g}")
        # synthetic-state expectations: normal slices resolve to ~1e-9; the
        # synthetic v-floor (1e-12, ~uniform) puts a truncation floor on the
        # full-model aggregate (~1e-5) that the REAL checkpoint bulk will not
        # have; zero-v slice passes by kink-convergence prefix logic.
        assert gate["per_slice"]["1/attn_q_low_denom"]["gate64_pass"]
        assert gate["per_slice"]["1/mlp_gate_sample"]["gate64_pass"]
        assert gate["per_slice"]["1/wte_low_v_nonzero"]["gate64_pass"]
        assert gate["per_slice"]["1/wte_zero_v_rows"]["gate64_pass"]
        assert gate["per_slice"]["1/full_model_random"]["best_rel64"] < 1e-3
    finally:
        f1.STEPS = orig["STEPS"]
        f1.EXT = orig["EXT"]
        f1.OUT = orig["OUT"]
        f1.CHECKPOINT_SHA256 = orig["CHECKPOINT_SHA256"]
        f1.CHECKPOINT_SIZES = orig["CHECKPOINT_SIZES"]
        f1.load_model_cfg = orig["load_model_cfg"]
    print("ALL SYNTHETIC SCRIPT TESTS PASS")


if __name__ == "__main__":
    main()
