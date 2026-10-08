"""Emit the frozen M1 preregistration config from config.py (single source of truth)."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from muon_transition import config


def schedule_factor(t: int) -> float:
    if t <= config.WARMUP_STEPS:
        return t / config.WARMUP_STEPS
    span = config.B0_OPTIMIZER_STEPS - config.WARMUP_STEPS
    w = (t - config.WARMUP_STEPS) / span
    return config.COSINE_MIN_FRAC + 0.45 * (1.0 + __import__("math").cos(__import__("math").pi * w))


def main():
    cfg_sha = hashlib.sha256((ROOT / "src" / "muon_transition" / "config.py").read_bytes()).hexdigest()
    data_dir = ROOT / "outputs" / "m0" / "data_audit"
    split_sha = hashlib.sha256((data_dir / "split_manifest.json").read_bytes()).hexdigest()
    pools_sha = hashlib.sha256((data_dir / "eval_pools.pt").read_bytes()).hexdigest()

    runs = []
    for arm in ("adamw", "muon"):
        ref = config.ADAMW_REF_LR if arm == "adamw" else config.MUON_REF_LR
        for mult in config.LR_MULTIPLIERS:
            for wd in config.WD_GRID:
                runs.append(
                    {
                        "run_id": f"m1_{arm}_lr{mult}_wd{wd}_p0.5_s{config.BASE_SEED_M1_M2}",
                        "stage": "M1",
                        "arm": arm,
                        "p": 0.5,
                        "seed": config.BASE_SEED_M1_M2,
                        "lr_multiplier": mult,
                        "lr": ref * mult,
                        "reference_lr": ref,
                        "weight_decay": wd,
                    }
                )

    prereg = {
        "format": "mte_m1_prereg_v1",
        "date_frozen": "2026-09-26",
        "protocol": "docs/MUON_PROTOCOL_v1.2.md (patch over docs/MUON_PROTOCOL_v1.1.md)",
        "config_py_sha256": cfg_sha,
        "model": {
            **config.MODEL_CONFIG,
            "total_parameters": 30_025_216,
            "tie_word_embeddings": True,
            "seq_len": config.SEQ_LEN,
            "param_dtype": "float32 master weights",
            "compute_dtype": "bfloat16 autocast, no GradScaler",
        },
        "tokenizer": {
            "path": str(config.TOKENIZER_PATH),
            "sha256": config.TOKENIZER_SHA256,
            "vocab_size": config.VOCAB_SIZE,
            "bos_id": config.BOS_ID,
            "doc_sep_id": config.DOC_SEP_ID,
        },
        "budget": {
            "B0_tokens": config.B0_TOKENS,
            "B0_tokens_expr": "2**26",
            "optimizer_steps": config.B0_OPTIMIZER_STEPS,
            "tokens_per_optimizer_step": config.TOKENS_PER_OPTIMIZER_STEP,
            "micro_batch_sequences": config.MICRO_BATCH_SEQUENCES,
            "grad_accum_steps": config.GRAD_ACCUM_STEPS,
            "seq_len": config.SEQ_LEN,
            "effective_loss_tokens_per_window": config.SEQ_LEN - 1,
            "single_epoch_at_all_p": True,
        },
        "schedule": {
            "form": "linear warmup then cosine to 10% of peak",
            "warmup_steps": config.WARMUP_STEPS,
            "warmup_rule": "s(t) = t/32 for t <= 32 (1-based step)",
            "cosine_rule": "s(t) = 0.1 + 0.45*(1+cos(pi*(t-32)/(2048-32))) for t > 32",
            "applies_to": "both arms, both routes (Muon and its AdamW-aux share the factor)",
            "checkpoints": [schedule_factor(t) for t in (1, 32, 1024, 2048)],
        },
        "optimizers": {
            "adamw": {
                "implementation": "torch.optim.AdamW",
                "reference_lr": config.ADAMW_REF_LR,
                "lr_grid": [config.ADAMW_REF_LR * m for m in config.LR_MULTIPLIERS],
                "betas": list(config.ADAMW_BETAS),
                "eps": config.ADAMW_EPS,
                "wd_grid": list(config.WD_GRID),
                "wd_applies_to": "all trainable params, no exemptions",
            },
            "muon": {
                **config.MUON_FREEZE,
                "reference_lr": config.MUON_REF_LR,
                "lr_grid": [config.MUON_REF_LR * m for m in config.LR_MULTIPLIERS],
                "wd_grid": list(config.WD_GRID),
                "wd_applies_to": "all trainable params, no exemptions",
                "route": "hidden 2D linear matrices only (56 tensors, 26,738,688 params)",
                "aux_adamw_route": "embedding (tied head) + norms + any bias/scalar (34 tensors, 3,286,528 params)",
                "aux_lr_rule": "same reference LR, same multiplier, same schedule",
            },
        },
        "grad_clip": {"global_norm": config.GRAD_CLIP_NORM, "scope": "all trainable params, both arms"},
        "data": {
            "domain_A": {"path": str(config.DOMAIN_A_PATH), "sha256": config.DOMAIN_A_SHA256},
            "domain_B": {"path": str(config.DOMAIN_B_PATH), "sha256": config.DOMAIN_B_SHA256},
            "split_manifest": "outputs/m0/data_audit/split_manifest.json",
            "split_manifest_sha256": split_sha,
            "eval_pools": "outputs/m0/data_audit/eval_pools.pt",
            "eval_pools_sha256": pools_sha,
            "dedup": "exact normalized-text, keep-first, within domain, pre-split",
            "mixture": "p = 0.5 (M1); token-level accounting via pre-generated batch manifests; realized p_hat logged",
        },
        "evaluation": {
            "primary": "final token NLL at step 2048 (dev pools for M1/M2 selection)",
            "nll_semantics": "sum/count per domain; V_Q = 0.5*NLL_A + 0.5*NLL_B; q frozen at 0.5",
            "dev_pool_tokens_per_domain": 130_560,
            "banned": ["best intermediate checkpoint", "training loss", "early curve"],
        },
        "observation_hooks": {
            "status": "diagnostic only, never primary evidence",
            "every_n_steps": config.HOOK_EVERY_N_STEPS,
            "required_fields": [
                "domain_gradient_cosine",
                "domain_conflict_ratio",
                "grad_global_norm",
                "update_norm",
                "update_work_proxy (-g^T delta_theta)",
            ],
            "probe": "first 4 batches x 8 windows per domain from frozen dev pools; 16 pairwise dots define conflict ratio",
            "naming_rule": "the quantity -g^T delta_theta is named update_work_proxy; no other name is permitted for it in any config or output",
        },
        "m1_runs": runs,
        "m1_total_runs": len(runs),
        "no_retuning_clause": "no LR/WD/cell may be added, removed or re-run after observing M1/M2 results (v1.1 single boundary-extension contingency excepted)",
    }

    out = ROOT / "configs" / "m1_baseline_prereg.json"
    out.write_text(json.dumps(prereg, indent=2))
    print(f"wrote {out} ({len(runs)} runs); config.py sha {cfg_sha[:16]}…")
    print(json.dumps(prereg["schedule"]["checkpoints"], indent=1))


if __name__ == "__main__":
    main()
