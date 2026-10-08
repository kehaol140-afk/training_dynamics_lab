"""T2b stage 2: 16 real branches + one repeat (W0/pi0), evaluated on Q_check.

Requires selection_lock.json to exist (written by stage 1). Each branch: restore full
anchor (model + both optimizer states + RNG) -> 10 real permuted batches (fresh grads
each step, global clip, Muon+AdamW step) -> Q_check NLL at step10 -> 128 common-tail
real batches -> Q_check NLL at step138. Constant plateau LR from checkpoint.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "muon_transition_experiment"))
sys.path.insert(0, str(ROOT.parent / "muon_transition_experiment" / "src"))
sys.path.insert(0, str(ROOT.parent / "minimind_base"))

from muon_transition.model import build_model
from muon_transition.optimizers import build_muon_arm
from muon_transition.mixture import MixtureStream
from vendored_fr.rng import restore_rng_state
from src.t2b_common import (ANCHOR_PATH, ANCHOR_SHA, build_round3_manifests, build_q_targets,
                            eval_q_checkpoints, sha256_file, flat_state_hash)

OUT = ROOT / "outputs" / "t2b_target_screen_v1"
DEV = "cuda:0"


def main():
    t0 = time.time()
    gpu_t0 = time.time()
    logf = open(OUT / "real_log.txt", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    assert (OUT / "selection_lock.json").exists(), "selection lock must precede real branches"
    lock = json.loads((OUT / "selection_lock.json").read_text())
    perms_blob = json.loads((OUT / "data_and_perms_manifest.json").read_text())
    perms = perms_blob["permutations"]
    blocks = build_round3_manifests()
    q = build_q_targets(verbose=False)
    assert sha256_file(ANCHOR_PATH) == ANCHOR_SHA

    from muon_transition.m1_data_cache import get_domain_windows
    a_train, b_train = get_domain_windows()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    payload = torch.load(ANCHOR_PATH, map_location="cpu", weights_only=False)
    muon_end_lr = payload["optimizer_states"][0]["param_groups"][0]["lr"]
    aux_end_lr = payload["optimizer_states"][1]["param_groups"][0]["lr"]
    log(f"plateau LRs from checkpoint: muon={muon_end_lr} aux={aux_end_lr}")

    model = build_model().to(DEV)
    anchor_model_state = payload["model_state"]
    anchor_opt_states = payload["optimizer_states"]
    anchor_rng = payload["rng"]
    anchor_params_cpu = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

    def fresh_restore():
        model.load_state_dict(anchor_model_state)
        model.train()
        muon_opt, aux_opt = build_muon_arm(model, lr=1e-9, weight_decay=0.0)
        muon_opt.load_state_dict(anchor_opt_states[0])
        aux_opt.load_state_dict(anchor_opt_states[1])
        restore_rng_state(anchor_rng)
        return [muon_opt, aux_opt]

    rows = []
    diag_rows = []
    plan = ([(0, k, False) for k in range(8)] + [(1, k, False) for k in range(8)]
            + [(0, 0, True)])
    for b, k, is_repeat in plan:
        perm = perms[f"W{b}"][k]["perm"]
        w_steps = [blocks[f"W{b}"][perm[j]] for j in range(10)]
        stream_w = MixtureStream({"steps": w_steps}, a_train, b_train)
        opts = fresh_restore()
        for i in range(10):
            for p in model.parameters():
                p.grad = None
            for mb in stream_w.micro_batches(i):
                batch = mb.to(DEV)
                with autocast:
                    logits = model(batch).logits[..., :-1, :].contiguous()
                    targets = batch[..., 1:].contiguous()
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                           targets.reshape(-1), reduction="mean")
                (loss / 4).backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
            for opt in opts:
                opt.step()
        ev10 = eval_q_checkpoints(model, q, "check", DEV)
        # 128 common-tail steps (production order, same for every branch)
        stream_c = MixtureStream({"steps": blocks["C"]}, a_train, b_train)
        for i in range(128):
            for p in model.parameters():
                p.grad = None
            for mb in stream_c.micro_batches(i):
                batch = mb.to(DEV)
                with autocast:
                    logits = model(batch).logits[..., :-1, :].contiguous()
                    targets = batch[..., 1:].contiguous()
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                           targets.reshape(-1), reduction="mean")
                (loss / 4).backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
            for opt in opts:
                opt.step()
        ev138 = eval_q_checkpoints(model, q, "check", DEV)
        # diagnostics: param shift + Muon momentum tail vs anchor
        mu_sq = aux_sq = 0.0
        m_tail_sq = 0.0
        for n, p in model.named_parameters():
            d = float((p.detach().cpu() - anchor_params_cpu[n]).norm() ** 2)
            if p.ndim == 2 and any(pat in n for pat in ("q_proj", "k_proj", "v_proj", "o_proj",
                                                        "gate_proj", "up_proj", "down_proj")):
                mu_sq += d
            else:
                aux_sq += d
        for st in opts[0].state.values():
            m_tail_sq += float(st["momentum_buffer"].norm() ** 2)
        rows.append({"window": b, "k": k, "repeat": is_repeat,
                     "step10_V": ev10["V"], "step10_nll_A": ev10["nll_A"], "step10_nll_B": ev10["nll_B"],
                     "step10_sum_A": ev10["nll_sum_A"], "step10_count_A": ev10["count_A"],
                     "step10_sum_B": ev10["nll_sum_B"], "step10_count_B": ev10["count_B"],
                     "step138_V": ev138["V"], "step138_nll_A": ev138["nll_A"], "step138_nll_B": ev138["nll_B"],
                     "step138_sum_A": ev138["nll_sum_A"], "step138_count_A": ev138["count_A"],
                     "step138_sum_B": ev138["nll_sum_B"], "step138_count_B": ev138["count_B"]})
        diag_rows.append({"window": b, "k": k, "repeat": is_repeat,
                          "param_shift_muon_sq": mu_sq, "param_shift_aux_sq": aux_sq,
                          "M_tail_norm_sq": m_tail_sq})
        log(f"[{time.time()-t0:.0f}s] W{b} k{k}{' REP' if is_repeat else ''}: "
            f"V10={ev10['V']:.6f} V138={ev138['V']:.6f}")
        finite = all(np.isfinite([ev10["V"], ev138["V"]]))
        if not finite:
            log(f"  !! non-finite objective for W{b} k{k} — recipe divergence")

    import csv
    with open(OUT / "actual_objectives.csv", "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "real_diag.csv", "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(diag_rows[0].keys()))
        w.writeheader()
        w.writerows(diag_rows)
    json.dump({"gpu_seconds_real_stage": time.time() - gpu_t0},
              open(OUT / "real_stage_meta.json", "w"), indent=1)
    log(f"real stage done in {time.time()-gpu_t0:.0f}s GPU")


if __name__ == "__main__":
    main()
