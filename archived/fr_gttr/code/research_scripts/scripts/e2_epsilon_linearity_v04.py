#!/usr/bin/env python3
"""E2：±epsilon 单步 impulse 的局部线性扫描。

LEGACY_PILOT_DO_NOT_USE_FOR_FORMAL_RESULT

设计
----
- 从同一个 MID Z_t 出发；
- 每个 stream 的 baseline / plus / minus 共用 future RNG + batch manifest；
- epsilon = rho * sigma_q；
- rho = [0.10, 0.25, 0.50, 1.00]；
- 只在第一个 optimizer step 注入一次；
- H=[1,8,32]；
- 输出 X_theta / Y_mean / Y_tail 的 chi 与 odd_error；
- 输出 perturbation 是否改变 clipping。

本脚本只给技术结果和诊断，不自动宣布“线性区成立”。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

LEGACY_PILOT_STATUS = "LEGACY_PILOT_DO_NOT_USE_FOR_FORMAL_RESULT"

from fr.batch_manifest import save_manifest
from fr.direction import load_direction, parameter_displacement_projection
from fr.eval_observables import evaluate_token_losses
from fr.full_state import load_full_training_state
from fr.hashing import stable_hash
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch
from fr.native_adamw_guard import assert_native_adamw
from fr.perturbation import apply_signed_gradient_impulse_
from fr.pilot_partition import load_partition
from fr.response_stats import paired_linearity_summary
from fr.stage_c_runtime import batches_from_train_pool
from fr.stream_rng import derive_stream_seed, seed_future_stream


def run_branch(
    *,
    torch, model, optimizer, scaler, autocast_ctx, ds,
    state_path, reference_theta, q, named, batches,
    stream_seed, sign, epsilon, cfg, meta, ref_eval, partition,
):
    """运行一个 paired branch。

    sign:
      0  baseline
      +1 plus
      -1 minus
    """
    load_full_training_state(state_path, model, optimizer, scaler)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # ============================================================
    # [FR-RNG-CRITICAL]
    # 同一 stream 的所有 branch 必须使用完全相同 future seed。
    # ============================================================
    seed_future_stream(stream_seed)

    horizons = sorted(map(int, cfg["horizons"]))
    max_h = max(horizons)
    accum = int(cfg["accumulation_steps"])

    branch_batch_hashes = []
    observations = {}
    first_preclip_norm = None
    first_clipped = None

    # trainer_utils.get_lr 运行时再由脚本注入，避免模块依赖 minimind。
    from trainer.trainer_utils import get_lr

    for h in range(1, max_h + 1):
        for a in range(accum):
            batch_idx = batches[(h - 1) * accum + a]
            x, y = stack_batch(ds, batch_idx, cfg["_device"])
            branch_batch_hashes.append(stable_hash({"x": x, "y": y}))
            with autocast_ctx:
                res = model(x, labels=y)
                loss = (res.loss + res.aux_loss) / accum
            scaler.scale(loss).backward()

        # native MiniMind 顺序：backward -> unscale -> clip -> step
        scaler.unscale_(optimizer)

        if h == 1:
            # ====================================================
            # [FR-PERTURBATION-CRITICAL]
            #
            # E2 唯一允许的因果差异。
            # 必须处于：
            #   accumulation 完成
            #   -> AMP unscale 完成
            #   -> [这里 ±epsilon*q]
            #   -> gradient clip
            #   -> native AdamW step
            #
            # h>1 时绝对禁止再次施加 perturbation。
            # ====================================================
            apply_signed_gradient_impulse_(
                named, q,
                epsilon=float(epsilon),
                sign=int(sign),
            )

        preclip = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg["grad_clip"])
        ).detach().cpu())
        if h == 1:
            first_preclip_norm = preclip
            first_clipped = bool(preclip > float(cfg["grad_clip"]))

        # ========================================================
        # [FR-NATIVE-ADAMW]
        # perturbation 之后仍由原生 AdamW 消费 gradient。
        # ========================================================
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        next_global_micro = int(meta["global_micro_step"]) + h * accum
        if h < max_h:
            next_lr = float(get_lr(
                next_global_micro + 1,
                int(meta["total_micro_steps"]),
                float(cfg["learning_rate"]),
            ))
            for g in optimizer.param_groups:
                g["lr"] = next_lr

        if h in horizons:
            x_theta = parameter_displacement_projection(model, reference_theta, q)
            ev = evaluate_token_losses(
                model, ds, partition["eval_indices"],
                int(cfg["eval_batch_size"]), cfg["_device"], autocast_ctx,
            )
            observations[str(h)] = {
                "X_theta": x_theta,
                "eval": ev,
                "Y_mean": float(ref_eval["mean"] - ev["mean"]),
                "Y_tail": float(ref_eval["cvar99"] - ev["cvar99"]),
            }

    return {
        "sign": int(sign),
        "epsilon": float(epsilon),
        "first_preclip_grad_norm": first_preclip_norm,
        "first_clipped": first_clipped,
        "batch_hashes": branch_batch_hashes,
        "observations": observations,
        "model_hash_final": stable_hash(raw_model(model).state_dict()),
        "optimizer_hash_final": stable_hash(optimizer.state_dict()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config", default="research/configs/stage_c_v04.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    cfg["_device"] = args.device
    minimind = root / "minimind"
    sys.path.insert(0, str(minimind))

    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import init_model

    out = root / "outputs/stage_c_v04"
    e1 = json.loads((out / "e1_formal_v04_report.json").read_text(encoding="utf-8"))
    if not e1.get("technical_pass"):
        raise SystemExit("E1 v0.4 尚未 PASS，禁止 E2")

    sigma_q = float(e1["sigma_q"])
    partition = load_partition(out / "partition_v04.json")
    q, q_meta, q_sha = load_direction(out / "full_model_q_v04.pt")

    mmcfg = MiniMindConfig(
        hidden_size=int(cfg["hidden_size"]),
        num_hidden_layers=int(cfg["num_hidden_layers"]),
        max_seq_len=int(cfg["max_seq_len"]),
        use_moe=False,
    )
    model, tokenizer = init_model(
        mmcfg, from_weight="none",
        tokenizer_path=str(minimind / "model"),
        device=args.device,
    )
    ds = PretrainDataset(str(out / "stage_c_subset.jsonl"), tokenizer, max_length=int(cfg["max_seq_len"]))
    autocast_ctx, scaler, _ = make_amp(torch, args.device, cfg["dtype"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    assert_native_adamw(optimizer)

    state_path = out / "mid_state_v04.pt"
    meta = load_full_training_state(state_path, model, optimizer, scaler)
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    reference_theta = payload["model"]

    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]
    ref_eval = evaluate_token_losses(
        model, ds, partition["eval_indices"],
        int(cfg["eval_batch_size"]), args.device, autocast_ctx,
    )

    R = int(cfg["e2_streams"])
    horizons = sorted(map(int, cfg["horizons"]))
    max_h = max(horizons)
    accum = int(cfg["accumulation_steps"])
    rhos = list(map(float, cfg["rho"]))

    all_streams = []
    baseline_by_stream = {}

    # ------------------------------------------------------------
    # 1. 每个 stream 的 batch manifest 固定一次。
    # ------------------------------------------------------------
    stream_specs = []
    for r in range(R):
        stream_seed = derive_stream_seed(int(cfg["stream_base_seed_e2"]), r)
        batch_seed = derive_stream_seed(stream_seed, 404)
        batches = batches_from_train_pool(
            partition["train_indices"],
            int(cfg["batch_size"]),
            batch_seed,
            max_h * accum,
        )
        manifest_sha = save_manifest(
            out / f"e2_stream_{r:03d}_batches.json",
            batches,
            metadata={"stream_id": r, "stream_seed": stream_seed, "batch_seed": batch_seed},
        )
        stream_specs.append((r, stream_seed, batch_seed, batches, manifest_sha))

    # ------------------------------------------------------------
    # 2. rho=0 no-op 路径自检。
    #    baseline(sign=0) 与 plus(sign=+1, epsilon=0) 必须完全一致。
    # ------------------------------------------------------------
    r0, seed0, _, batches0, _ = stream_specs[0]
    noop_a = run_branch(
        torch=torch, model=model, optimizer=optimizer, scaler=scaler,
        autocast_ctx=autocast_ctx, ds=ds, state_path=state_path,
        reference_theta=reference_theta, q=q, named=named, batches=batches0,
        stream_seed=seed0, sign=0, epsilon=0.0, cfg=cfg, meta=meta,
        ref_eval=ref_eval, partition=partition,
    )
    noop_b = run_branch(
        torch=torch, model=model, optimizer=optimizer, scaler=scaler,
        autocast_ctx=autocast_ctx, ds=ds, state_path=state_path,
        reference_theta=reference_theta, q=q, named=named, batches=batches0,
        stream_seed=seed0, sign=+1, epsilon=0.0, cfg=cfg, meta=meta,
        ref_eval=ref_eval, partition=partition,
    )
    noop_pass = bool(
        noop_a["batch_hashes"] == noop_b["batch_hashes"]
        and noop_a["model_hash_final"] == noop_b["model_hash_final"]
        and noop_a["optimizer_hash_final"] == noop_b["optimizer_hash_final"]
        and noop_a["observations"] == noop_b["observations"]
    )
    if not noop_pass:
        raise RuntimeError("rho=0 perturbation no-op 自检失败，禁止继续 E2")

    # ------------------------------------------------------------
    # 3. baseline 每个 stream 只运行一次。
    # ------------------------------------------------------------
    for r, stream_seed, batch_seed, batches, manifest_sha in stream_specs:
        b0 = run_branch(
            torch=torch, model=model, optimizer=optimizer, scaler=scaler,
            autocast_ctx=autocast_ctx, ds=ds, state_path=state_path,
            reference_theta=reference_theta, q=q, named=named, batches=batches,
            stream_seed=stream_seed, sign=0, epsilon=0.0, cfg=cfg, meta=meta,
            ref_eval=ref_eval, partition=partition,
        )
        baseline_by_stream[r] = b0

    # ------------------------------------------------------------
    # 4. 每个 rho 跑 plus/minus；同 stream 完全复用 seed+manifest。
    # ------------------------------------------------------------
    rho_records = {}
    for rho in rhos:
        epsilon = float(rho * sigma_q)
        rows = []
        for r, stream_seed, batch_seed, batches, manifest_sha in stream_specs:
            bp = run_branch(
                torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, ds=ds, state_path=state_path,
                reference_theta=reference_theta, q=q, named=named, batches=batches,
                stream_seed=stream_seed, sign=+1, epsilon=epsilon, cfg=cfg, meta=meta,
                ref_eval=ref_eval, partition=partition,
            )
            bm = run_branch(
                torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, ds=ds, state_path=state_path,
                reference_theta=reference_theta, q=q, named=named, batches=batches,
                stream_seed=stream_seed, sign=-1, epsilon=epsilon, cfg=cfg, meta=meta,
                ref_eval=ref_eval, partition=partition,
            )
            b0 = baseline_by_stream[r]

            # [FR-BATCH-STREAM] 三分支 batch bytes 必须一致。
            if not (b0["batch_hashes"] == bp["batch_hashes"] == bm["batch_hashes"]):
                raise RuntimeError(f"stream {r} paired branch batch hash 不一致")

            rows.append({
                "stream_id": r,
                "stream_seed": stream_seed,
                "batch_seed": batch_seed,
                "manifest_sha256": manifest_sha,
                "baseline": b0,
                "plus": bp,
                "minus": bm,
            })

        summaries = {}
        for h in horizons:
            hk = str(h)
            summaries[hk] = {}
            for obs_name in ["X_theta", "Y_mean", "Y_tail"]:
                y0 = [x["baseline"]["observations"][hk][obs_name] for x in rows]
                yp = [x["plus"]["observations"][hk][obs_name] for x in rows]
                ym = [x["minus"]["observations"][hk][obs_name] for x in rows]
                summaries[hk][obs_name] = paired_linearity_summary(y0, yp, ym, epsilon)

        clip = {
            "baseline_rate": sum(x["baseline"]["first_clipped"] for x in rows) / R,
            "plus_rate": sum(x["plus"]["first_clipped"] for x in rows) / R,
            "minus_rate": sum(x["minus"]["first_clipped"] for x in rows) / R,
        }
        rho_records[f"{rho:g}"] = {
            "rho": rho,
            "epsilon": epsilon,
            "clipping": clip,
            "summaries": summaries,
            "streams": rows,
        }

    report = {
        "protocol": "E2_V04_EPSILON_LINEARITY",
        "technical_pass": True,
        "scientific_linearity_pass": None,
        "rho0_noop_pass": noop_pass,
        "sigma_q_source": sigma_q,
        "q_sha256": q_sha,
        "R": R,
        "horizons": horizons,
        "rho_records": rho_records,
        "boundaries": [
            "technical_pass 只表示 paired branching / no-op / 数值流程成功。",
            "scientific_linearity_pass 故意保持 None；必须人工看 odd_error、chi(rho)、clipping 后决定。",
            "如果全部 rho 非线性，按研究计划再缩小到 0.025/0.05/0.10。",
        ],
    }
    (out / "e2_linearity_v04_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# E2 v0.4 epsilon 局部线性扫描", "",
        "- 技术执行：**PASS**",
        f"- rho=0 no-op：**{'PASS' if noop_pass else 'FAIL'}**",
        f"- sigma_q：`{sigma_q:.12g}`",
        f"- R：`{R}`",
        "- 科学线性判定：**未自动判定，需人工审核**",
        "",
    ]
    for rho in rhos:
        rr = rho_records[f"{rho:g}"]
        lines += [
            f"## rho={rho:g}, epsilon={rr['epsilon']:.12g}",
            f"- clipping baseline/plus/minus: "
            f"`{rr['clipping']['baseline_rate']:.4f}` / "
            f"`{rr['clipping']['plus_rate']:.4f}` / "
            f"`{rr['clipping']['minus_rate']:.4f}`",
            ""
        ]
        for h in horizons:
            lines.append(f"### H={h}")
            for obs in ["X_theta", "Y_mean", "Y_tail"]:
                s = rr["summaries"][str(h)][obs]
                lines.append(
                    f"- {obs}: chi=`{s['chi']:.12g}`, "
                    f"chi_std=`{s['chi_std']:.12g}`, "
                    f"odd_error=`{s['odd_error']:.12g}`"
                )
            lines.append("")
    (out / "e2_linearity_v04_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("E2_V04_TECHNICAL_PASS")
    print("rho0_noop_pass=", noop_pass)
    for rho in rhos:
        rr = rho_records[f"{rho:g}"]
        print(f"rho={rho:g} epsilon={rr['epsilon']:.12g} clipping={rr['clipping']}")
        for h in horizons:
            sm = rr["summaries"][str(h)]
            print(
                f"  H={h} "
                f"theta_odd={sm['X_theta']['odd_error']:.6g} "
                f"mean_odd={sm['Y_mean']['odd_error']:.6g} "
                f"tail_odd={sm['Y_tail']['odd_error']:.6g}"
            )
    print("report=", out / "e2_linearity_v04_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
