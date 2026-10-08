"""M1-0 smoke: both arms, lr_mult=1.0, WD=0.01, p=0.5, 100 steps on GPU. No science."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from muon_transition import config
from muon_transition.checkpoint import load_training_state, parameter_fingerprint, save_training_state
from muon_transition.m1_runner import prepare_data, run_cell

N_STEPS = 100
OUT = config.OUTPUTS / "m1" / "smoke_m1_0"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    data = prepare_data()
    report = {"n_steps": N_STEPS, "device": device, "arms": {}}

    for arm in ("adamw", "muon"):
        run_id = f"smoke_{arm}_lr1.0_wd0.01_p0.5_s2030"
        res = run_cell(
            arm=arm, lr_multiplier=1.0, weight_decay=0.01, p=0.5, seed=2030,
            run_id=run_id, out_dir=OUT / arm, n_steps=N_STEPS, device=device,
            data=data, save_state=True,
        )
        checks = {
            "loss_decreased": res["loss_decreased"],
            "no_nan_inf_loss": not res["any_nan_loss"],
            "no_nan_inf_gradnorm": not res["any_nonfinite_gradnorm"],
            "optimizer_state_ok": res["optimizer_state_ok"],
            "muon_momentum_buffers": res["muon_momentum_buffers"],
            "first20_last20": [res["train_loss_mean_first20"], res["train_loss_mean_last20"]],
        }

        ckpt = OUT / arm / "final_state.pt"
        from muon_transition.model import build_model
        from muon_transition.optimizers import build_adamw_arm, build_muon_arm
        m = build_model(seed=999).to(device)
        if arm == "adamw":
            opts = [build_adamw_arm(m, lr=1e-3, weight_decay=0.01)]
        else:
            mu, aux = build_muon_arm(m, lr=1e-3, weight_decay=0.01)
            opts = [mu, aux]
        cursor = load_training_state(ckpt, m, opts)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        fp_loaded = parameter_fingerprint_from_blob({"model_state": {k: v.cpu() for k, v in m.state_dict().items()}})
        fp_saved = parameter_fingerprint_from_blob(blob)
        checks["checkpoint_roundtrip_model_match"] = fp_loaded == fp_saved
        checks["checkpoint_cursor"] = cursor

        obs_path = OUT / arm / "obs.jsonl"
        lines = [json.loads(l) for l in open(obs_path)] if obs_path.exists() else []
        required = {"domain_gradient_cosine", "domain_conflict_ratio", "grad_global_norm",
                    "update_norm", "update_work_proxy"}
        checks["obs_records"] = len(lines)
        checks["obs_has_all_required_fields"] = all(required <= set(r) for r in lines)
        checks["obs_every_100"] = [r["step"] for r in lines]

        report["arms"][arm] = checks
        print(arm, json.dumps(checks, indent=2), flush=True)

    report["smoke_pass"] = all(
        v["loss_decreased"] and v["no_nan_inf_loss"] and v["no_nan_inf_gradnorm"]
        and v["optimizer_state_ok"] and v["checkpoint_roundtrip_model_match"]
        and v["obs_records"] >= 1 and v["obs_has_all_required_fields"]
        for v in report["arms"].values()
    )
    (OUT / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print("SMOKE_PASS:", report["smoke_pass"], flush=True)


def parameter_fingerprint_from_blob(blob):
    import hashlib

    h = hashlib.sha256()
    for name, t in blob["model_state"].items():
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    main()
