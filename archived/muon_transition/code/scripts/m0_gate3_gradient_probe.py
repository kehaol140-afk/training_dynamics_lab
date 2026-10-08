"""M0 Gate 3C: gradient geometry probe (diagnostic only, no mechanism claims).
Writes outputs/m0/data_audit/{gradient_geometry.csv, gradient_geometry_summary.json}."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from muon_transition import config
from muon_transition.data import TokenizerHandle, conversation_to_text, load_domain, pack_domain_stream
from muon_transition.model import build_model, unique_named_parameters
from muon_transition.optimizers import build_adamw_arm, build_muon_arm
from muon_transition.partitions import build_pair_aligned_split
from muon_transition.train_loop import batch_token_loss

N_BATCHES = 8
BATCH_SEQUENCES = 8


def get_grad_vector(model):
    return torch.cat([p.grad.detach().reshape(-1).double() for _, p in unique_named_parameters(model)])


def clear_grads(model):
    for _, p in unique_named_parameters(model):
        p.grad = None


def snapshot_params(model):
    return [p.detach().clone() for _, p in unique_named_parameters(model)]


def restore_params(model, snap):
    for (name, p), s in zip(unique_named_parameters(model), snap):
        p.data.copy_(s)


def flat_delta(model, snap):
    return torch.cat([(p.detach() - s).reshape(-1).double() for (_, p), s in zip(unique_named_parameters(model), snap)])


def main():
    torch.manual_seed(0)
    model = build_model(seed=config.BASE_SEED_M1_M2)
    model.train()

    tok = TokenizerHandle()
    records = {d: load_domain(p) for d, p in (("A", config.DOMAIN_A_PATH), ("B", config.DOMAIN_B_PATH))}
    manifest = build_pair_aligned_split(records["A"], records["B"])

    streams = {}
    for dom in ("A", "B"):
        train_idx = sorted(manifest[dom]["train"])[: N_BATCHES * BATCH_SEQUENCES * 4]
        tl = []
        for i in train_idx:
            tl.extend(tok.encode_texts([conversation_to_text(records[dom][i])])[0])
            tl.append(config.DOC_SEP_ID)
        windows = pack_domain_stream([tl], config.SEQ_LEN, config.BOS_ID, config.DOC_SEP_ID)
        streams[dom] = windows[: N_BATCHES * BATCH_SEQUENCES].reshape(N_BATCHES, BATCH_SEQUENCES, config.SEQ_LEN)

    grads = {"A": [], "B": []}
    for dom in ("A", "B"):
        for b in range(N_BATCHES):
            clear_grads(model)
            loss = batch_token_loss(model, streams[dom][b])
            loss.backward()
            grads[dom].append(get_grad_vector(model))

    rows = []
    cosines, dots, ratios = [], [], []
    for ia, ga in enumerate(grads["A"]):
        for ib, gb in enumerate(grads["B"]):
            cos = float(torch.nn.functional.cosine_similarity(ga, gb, dim=0))
            dot = float(ga @ gb)
            ratio = float(ga.norm() / gb.norm())
            cosines.append(cos)
            dots.append(dot)
            ratios.append(ratio)
            rows.append(
                {
                    "metric": "grad_cosine",
                    "batch_A": ia,
                    "batch_B": ib,
                    "value": cos,
                    "detail": f"||gA||={float(ga.norm()):.4f} ||gB||={float(gb.norm()):.4f}",
                }
            )
            rows.append({"metric": "grad_dot_sign", "batch_A": ia, "batch_B": ib,
                         "value": 1.0 if dot < 0 else 0.0, "detail": f"dot={dot:.4e}"})
            rows.append({"metric": "grad_norm_ratio", "batch_A": ia, "batch_B": ib,
                         "value": ratio, "detail": "||gA||/||gB||"})

    within = {}
    for dom in ("A", "B"):
        cs = []
        for i in range(len(grads[dom])):
            for j in range(i + 1, len(grads[dom])):
                cs.append(float(torch.nn.functional.cosine_similarity(grads[dom][i], grads[dom][j], dim=0)))
        within[dom] = sum(cs) / len(cs)

    update_cosines = {}
    snap0 = snapshot_params(model)
    for arm in ("adamw", "muon"):
        for dom in ("A", "B"):
            restore_params(model, snap0)
            if arm == "adamw":
                opts = [build_adamw_arm(model, lr=3e-4, weight_decay=0.1)]
            else:
                mu, aux = build_muon_arm(model, lr=0.02, weight_decay=0.1)
                opts = [mu, aux]
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            loss = batch_token_loss(model, streams[dom][0])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
            for opt in opts:
                opt.step()
            update_cosines[f"{arm}_{dom}_delta"] = flat_delta(model, snap0)
        a = update_cosines.pop(f"{arm}_A_delta")
        b = update_cosines.pop(f"{arm}_B_delta")
        update_cosines[arm] = float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    summary = {
        "gate": "3C",
        "diagnostic_only": True,
        "model_state": "untrained fixed-seed init, fp32, CPU",
        "n_batches_per_domain": N_BATCHES,
        "batch_sequences": BATCH_SEQUENCES,
        "seq_len": config.SEQ_LEN,
        "cross_domain_grad_cosine": {
            "mean": sum(cosines) / len(cosines),
            "min": min(cosines),
            "max": max(cosines),
            "median": sorted(cosines)[len(cosines) // 2],
        },
        "conflict_ratio_P_dot_lt_0": sum(1 for d in dots if d < 0) / len(dots),
        "grad_norm_ratio": {
            "mean": sum(ratios) / len(ratios),
            "min": min(ratios),
            "max": max(ratios),
        },
        "within_domain_mean_cosine": within,
        "update_direction_cosine_first_batch": update_cosines,
    }

    out = config.M0_OUTPUTS / "data_audit"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "gradient_geometry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "batch_A", "batch_B", "value", "detail"])
        w.writeheader()
        w.writerows(rows)
    (out / "gradient_geometry_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
