"""A0H-1.5 bookkeeping closure collector (round 2 amendment §3).

Runs ONE a0h15 diagnostics pass per unit with `collect` filled (raw per-step
vectors, no computational change), then computes the FULL step-by-step closure
in FP64 entirely outside the pass:

    R0  = p1^T x1 - w0^T a
    Rk  = p(k+1)^T x(k+1) - pk^T xk          (k = 1..H-1)
    EF  = SF - pH^T xH        (endpoint gap, production forward vs stored pairing)
    EA  = SA - w0^T a         (adjoint scoring-consistency term)
    identity: SF - SA = EF - EA + R0 + sum(Rk)

SF = production forward score <v_H, r_H_C>; SA = production adjoint score
<w_0, a_j>. All pairings in float64. The closure is a BOOKKEEPING identity only
(storage aliasing, indexing, endpoint gradient, lambda/loss normalization); it
cannot prove adjoint correctness (amendment §3).

Usage:
  PYTHONPATH=research/src CUDA_VISIBLE_DEVICES=1 python research/scripts/a0h15_bookkeeping_v01.py \
      --device cuda:0 --states 2031/52 [--hvp-fp32] [--budget-seconds 300]
Output: outputs/a0h15_vjp_shared_v01/bookkeeping/closure_<unit>[_fp32hvp]_s0.json
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "research" / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUTPUT_DIR = "outputs/a0h15_vjp_shared_v01"

_spec = importlib.util.spec_from_file_location(
    "a0h_adjoint_dev_check_v01",
    PROJECT_ROOT / "research" / "scripts" / "a0h_adjoint_dev_check_v01.py")
_a0h1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_a0h1)


def _pair64(torch, a, b) -> float:
    """FP64 pairing of two name->tensor maps."""
    total = None
    for name in a:
        x = a[name].detach().to(torch.float64).reshape(-1)
        y = b[name].detach().to(torch.float64).reshape(-1)
        s = (x * y).sum()
        total = s if total is None else total + s
    return float(total)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--states", default="2031/52")
    ap.add_argument("--hvp-fp32", action="store_true")
    ap.add_argument("--budget-seconds", type=float, default=600.0)
    args = ap.parse_args()

    import a0g_decision_utility_v01 as a0g
    import a0h15_vjp_diagnostics_v01 as diag_mod
    from fr.minimind_v03_runtime import make_amp, raw_model
    from fr.native_adamw_guard import assert_native_adamw
    from fr.pilot_partition import load_partition

    root = PROJECT_ROOT
    out_dir = root / OUTPUT_DIR / "bookkeeping"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((root / "research/configs/e3c_confirmatory_v07.json").read_text())
    v07_out = root / cfg["output_dir"]
    partition = load_partition(v07_out / "partition_v07.json")
    data_path = v07_out / "e3c_subset.jsonl"
    a0g_out = root / a0g.OUTPUT_DIR
    data_group_definition = json.loads((a0g_out / "data_group_definition.json").read_text())
    validation_split = json.loads((a0g_out / "validation_split.json").read_text())

    minimind = root / "minimind"
    if str(minimind) not in sys.path:
        sys.path.insert(0, str(minimind))

    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model

    mmcfg = MiniMindConfig(
        hidden_size=int(cfg["hidden_size"]), num_hidden_layers=int(cfg["num_hidden_layers"]),
        max_seq_len=int(cfg["max_seq_len"]), use_moe=False)
    model, tokenizer = init_model(
        mmcfg, from_weight="none", tokenizer_path=str(minimind / "model"), device=args.device)
    dataset = PretrainDataset(str(data_path), tokenizer, max_length=int(cfg["max_seq_len"]))
    autocast_ctx, scaler, _ = make_amp(torch, args.device, cfg["dtype"])
    if scaler.is_enabled():
        raise RuntimeError("bf16 replay unexpectedly enabled GradScaler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    assert_native_adamw(optimizer)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]

    def _chunk(indices):
        idx = sorted(map(int, indices))
        return [idx[i:i + a0g.EVAL_BATCH_SIZE] for i in range(0, len(idx), a0g.EVAL_BATCH_SIZE)]

    val_score_tensors = a0g._stack_all(dataset, _chunk(validation_split["v_score"]), args.device)
    timers = a0g._Timers(torch)
    dev_ctxs = a0g._load_v07_state_contexts(root, cfg, partition, data_group_definition)
    if args.states:
        wanted = set(tuple(s.split("/")) for s in args.states.split(","))
        dev_ctxs = [c for c in dev_ctxs if (str(c.seed), str(c.step)) in wanted]

    wall_start = time.time()
    for ctx in dev_ctxs:
        if time.time() - wall_start > args.budget_seconds:
            print("BUDGET_EXHAUSTED", flush=True)
            break
        stream = next(s for s in ctx.streams if int(s["stream_id"]) == 0)
        label = f"dev:{ctx.seed}/{ctx.step}"
        collect: dict = {}
        t0 = time.perf_counter()
        res = diag_mod._run_diagnostics_pass(
            torch=torch, model=model, optimizer=optimizer, scaler=scaler,
            autocast_ctx=autocast_ctx, dataset=dataset, named=named,
            checkpoint_path=ctx.checkpoint_path, batches=stream["batches"],
            stream_seed=stream["stream_seed"],
            cand_batches_by_j=ctx.candidate_batches(stream["stream_seed"]),
            val_score_tensors=val_score_tensors,
            accumulation_steps=int(cfg["accumulation_steps"]), device=str(model.device),
            grad_clip=float(cfg["grad_clip"]), base_lr=float(cfg["learning_rate"]),
            global_micro_step=int(ctx.bank_meta["global_micro_step"]),
            total_micro_steps=int(ctx.bank_meta["total_micro_steps"]),
            get_lr=get_lr, timers=timers, hvp_fp32=bool(args.hvp_fp32),
            full_diagnostics=False, collect=collect)
        wall = time.perf_counter() - t0

        x_all = collect["x_all"]            # (h, j) -> (th, m, v)
        p_store = collect["p_store"]        # H -> {k: (th, m, v)}
        w0_by_H = collect["w_0"]            # H -> map
        a_by_Hj = collect["a_by_j"]         # H -> {j: map}
        v_by_H = collect["v_score_grad"]    # h -> map
        r_by_H = collect["r_H_C"]           # h -> {j: map}

        report = {"state": label, "stream_id": 0, "wall_seconds": wall,
                  "hvp_fp32": bool(args.hvp_fp32),
                  "identity": "SF - SA = EF - EA + R0 + sum(Rk)",
                  "denominator_rule": "denom = |SF| + |SA| (pre-defined, same for all terms)",
                  "horizons": {}}
        for horizon in a0g.HORIZONS:
            hk = str(horizon)
            per_j = {}
            for j in sorted({jj for (hh, jj) in x_all if hh == horizon}):
                js = str(j)
                v_H = v_by_H[hk]
                r_Hj = r_by_H[hk][js]
                w_0 = w0_by_H[hk]
                a_j = a_by_Hj[hk][js]
                pS = p_store[hk]            # {k: triple}
                # production scores recomputed independently from dumped vectors
                SF = _pair64(torch, v_H, r_Hj)
                SA = _pair64(torch, w_0, a_j)
                # cross-check against pass-produced scores
                SF_pass = float(res["hhat_fwd_C"][hk][j])
                SA_pass = float(res["vjp_shared"][hk]["hhat_adj"][js])
                # endpoint / scoring-consistency terms
                xH = x_all[(horizon, j)]
                pH_xH = (_pair64(torch, pS[horizon][0], xH[0])
                         + _pair64(torch, pS[horizon][1], xH[1])
                         + _pair64(torch, pS[horizon][2], xH[2]))
                w0_a = _pair64(torch, w_0, a_j)
                EF = SF - pH_xH
                EA = SA - w0_a
                # step terms
                x1 = x_all[(1, j)]
                R0 = (_pair64(torch, pS[1][0], x1[0]) + _pair64(torch, pS[1][1], x1[1])
                      + _pair64(torch, pS[1][2], x1[2]) - w0_a)
                Rk = []
                for k in range(1, horizon):
                    xk1 = x_all[(k + 1, j)]
                    xk = x_all[(k, j)]
                    Rk.append(
                        _pair64(torch, pS[k + 1][0], xk1[0])
                        + _pair64(torch, pS[k + 1][1], xk1[1])
                        + _pair64(torch, pS[k + 1][2], xk1[2])
                        - _pair64(torch, pS[k][0], xk[0])
                        - _pair64(torch, pS[k][1], xk[1])
                        - _pair64(torch, pS[k][2], xk[2]))
                residual = (SF - SA) - (EF - EA + R0 + sum(Rk))
                denom = abs(SF) + abs(SA)
                per_j[js] = {
                    "SF": SF, "SA": SA,
                    "SF_pass_produced": SF_pass, "SA_pass_produced": SA_pass,
                    "SF_recompute_gap": abs(SF - SF_pass), "SA_recompute_gap": abs(SA - SA_pass),
                    "pH_xH": pH_xH, "w0_a": w0_a,
                    "EF": EF, "EA": EA, "R0": R0, "Rk": Rk,
                    "abs_closure_residual": abs(residual),
                    "rel_closure_residual": abs(residual) / max(denom, 1e-30),
                    "denominator": denom,
                    "EF_over_denom": EF / max(denom, 1e-30),
                    "EA_over_denom": EA / max(denom, 1e-30),
                    "Rk_abs_max": max((abs(r) for r in Rk), default=0.0),
                    "Rk_abs_argmax": (max(range(len(Rk)), key=lambda i: abs(Rk[i])) + 1) if Rk else None,
                }
            report["horizons"][hk] = per_j

        suffix = "_fp32hvp" if args.hvp_fp32 else ""
        out_path = out_dir / f"closure_{ctx.seed}_{ctx.step}{suffix}_s0.json"
        out_path.write_text(json.dumps(report, indent=1, default=float))
        print(f"[{label}] wrote {out_path.name} ({wall:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
