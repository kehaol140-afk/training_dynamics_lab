"""M0 Gate 4: token accounting, budget B0, eval pools, reproducibility, checkpoint.
Writes outputs/m0/pipeline_audit.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from muon_transition import config
from muon_transition.checkpoint import load_training_state, parameter_fingerprint, save_training_state
from muon_transition.eval_nll import evaluate_q
from muon_transition.model import build_model
from muon_transition.optimizers import build_adamw_arm, build_muon_arm
from muon_transition.train_loop import train_step


def make_batches(seed, n, bs, seq):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(3, config.VOCAB_SIZE, (bs, seq), generator=g) for _ in range(n)]


def run_arm(arm, batches, steps, seed=2030):
    model = build_model(seed=seed)
    if arm == "adamw":
        opts = [build_adamw_arm(model, lr=3e-4, weight_decay=0.1)]
    else:
        mu, aux = build_muon_arm(model, lr=0.02, weight_decay=0.1)
        opts = [mu, aux]
    losses = []
    for i in range(steps):
        losses.append(train_step(model, opts, batches[i]))
    return model, opts, losses


def real_data_smoke():
    """15-step plumbing smoke on real Domain A windows (not scientific evidence).

    Muon smoke LR = 0.005 chosen from the M0 stability screen (lr 0.02 unstable,
    0.001-0.005 stable at this micro-scale); the M1 reference base LR is a
    protocol-review decision, not set here.
    """
    from muon_transition.data import TokenizerHandle, conversation_to_text, load_domain, pack_domain_stream
    from muon_transition.partitions import build_pair_aligned_split

    records = load_domain(config.DOMAIN_A_PATH)
    manifest = build_pair_aligned_split(records, load_domain(config.DOMAIN_B_PATH))
    tok = TokenizerHandle()
    idx = sorted(manifest["A"]["train"])[:64]
    tl = []
    for i in idx:
        tl.extend(tok.encode_texts([conversation_to_text(records[i])])[0])
        tl.append(config.DOC_SEP_ID)
    windows = pack_domain_stream([tl], config.SEQ_LEN, config.BOS_ID, config.DOC_SEP_ID)[: 15 * 8]
    batches = [windows[i * 8 : (i + 1) * 8] for i in range(15)]

    out = {}
    for arm, lr in (("adamw", 3e-4), ("muon", 0.005)):
        model = build_model(seed=2030)
        if arm == "adamw":
            opts = [build_adamw_arm(model, lr=lr, weight_decay=0.1)]
        else:
            mu, aux = build_muon_arm(model, lr=lr, weight_decay=0.1)
            opts = [mu, aux]
        losses = [train_step(model, opts, b)["loss"] for b in batches]
        out[arm] = {
            "smoke_lr": lr,
            "losses": losses,
            "first": losses[0],
            "last": losses[-1],
            "min": min(losses),
            "decreased": losses[-1] < losses[0],
        }
    return out


def main():
    report = {"gate": 4, "checks": {}}

    report["checks"]["token_accounting"] = {
        "tokenizer_sha256_frozen": config.TOKENIZER_SHA256,
        "seq_len": config.SEQ_LEN,
        "packing": "per-doc enc(doc)+DOC_SEP(0); windows = consecutive (seq_len-1) slices each prefixed by BOS(1)",
        "effective_loss_tokens_per_window": config.SEQ_LEN - 1,
        "effective_tokens_formula": "n_windows * (seq_len - 1)",
        "token_counting_rule": "sum of per-token CE over valid targets only; never mean-of-batch-means",
    }

    assert config.B0_TOKENS % config.TOKENS_PER_OPTIMIZER_STEP == 0
    report["checks"]["budget_B0"] = {
        "definition": "B0 = total training tokens per run, identical for every run of M1/M2/M3",
        "tokens": config.B0_TOKENS,
        "tokens_per_optimizer_step": config.TOKENS_PER_OPTIMIZER_STEP,
        "micro_batch_sequences": config.MICRO_BATCH_SEQUENCES,
        "grad_accum_steps": config.GRAD_ACCUM_STEPS,
        "optimizer_steps": config.B0_OPTIMIZER_STEPS,
        "status": "frozen definition; B0 magnitude pending protocol review before M1",
    }

    pools_path = config.M0_OUTPUTS / "data_audit" / "eval_pools.pt"
    pools_pt = torch.load(pools_path, weights_only=True)
    pools = {"A": pools_pt["A_dev"], "B": pools_pt["B_dev"]}
    confirm = {"A": pools_pt["A_confirm"], "B": pools_pt["B_confirm"]}
    model = build_model(seed=2030)
    q_dev = evaluate_q(model, pools)
    q_confirm = evaluate_q(model, confirm)
    n_tokens = {k: v for k, v in q_dev.items() if k.startswith("tokens_")}
    report["checks"]["evaluation"] = {
        "q": 0.5,
        "pools": "outputs/m0/data_audit/eval_pools.pt (frozen at Gate 3)",
        "Q_dev_token_counts": n_tokens,
        "Q_dev_at_init": q_dev,
        "Q_confirm_at_init": q_confirm,
        "equal_token_counts_across_domains": n_tokens["tokens_A"] == n_tokens["tokens_B"],
        "metric": "token NLL = sum/count per domain; V_Q = 0.5*NLL_A + 0.5*NLL_B",
    }

    repro = {}
    ckpt = {}
    for arm in ("adamw", "muon"):
        batches = make_batches(101, 4, 4, 64)
        m1, o1, l1 = run_arm(arm, batches, 3)
        m2, o2, l2 = run_arm(arm, batches, 3)
        repro[arm] = {
            "identical_losses": l1 == l2,
            "identical_param_fingerprint": parameter_fingerprint(m1) == parameter_fingerprint(m2),
            "first_losses": [x["loss"] for x in l1],
        }

        m3, o3, _ = run_arm(arm, batches, 2)
        path = config.M0_OUTPUTS / f"ckpt_roundtrip_{arm}.pt"
        save_training_state(path, m3, o3, cursor={"batch_index": 2, "arm": arm})
        m4, o4, _ = run_arm(arm, batches, 0)
        cursor = load_training_state(path, m4, o4)
        l4 = train_step(m4, o4, batches[2])
        ckpt[arm] = {
            "cursor_restored": cursor["batch_index"] == 2,
            "resume_step3_loss_matches": abs(l4["loss"] - l1[2]["loss"]) < 1e-12,
            "param_fingerprint_matches_uninterrupted": parameter_fingerprint(m4) == parameter_fingerprint(m1),
            "saved_fields": ["model", "optimizers", "rng", "cursor", "payload_sha256"],
        }
        path.unlink()

    report["checks"]["reproducibility_first_updates"] = repro
    report["checks"]["checkpoint_save_load"] = ckpt

    report["checks"]["real_data_smoke_15_steps"] = real_data_smoke()

    smoke_ok = all(
        v["decreased"] and v["last"] < 8.9 for v in report["checks"]["real_data_smoke_15_steps"].values()
    )
    report["gate4_pass"] = bool(
        repro["adamw"]["identical_losses"]
        and repro["adamw"]["identical_param_fingerprint"]
        and repro["muon"]["identical_losses"]
        and repro["muon"]["identical_param_fingerprint"]
        and all(v["resume_step3_loss_matches"] and v["param_fingerprint_matches_uninterrupted"] for v in ckpt.values())
        and report["checks"]["evaluation"]["equal_token_counts_across_domains"]
        and smoke_ok
    )

    config.M0_OUTPUTS.mkdir(parents=True, exist_ok=True)
    (config.M0_OUTPUTS / "pipeline_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
