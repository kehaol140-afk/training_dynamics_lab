#!/usr/bin/env python3
"""E4B-1：baseline-only exact replay 测量 q-direction rotation。

科学边界：
- exploratory mechanism study；不改写 v0.7 H8_CONFIRMATION=FAIL；
- 只重放 v0.7 E2 baseline stream，不运行 +/- perturbation；
- 每个 stream 一次 32-step rollout，在固定 horizons 上 observation-only probe；
- 每条 replay 最终 model/optimizer hash 必须与 v0.7 baseline 原记录一致。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fr.e4b_q_rotation import (
    direction_cosine,
    load_and_verify_manifest,
    probe_direction_observation,
    resolve_recorded_path,
)
from fr.direction import load_direction
from fr.full_state import load_full_training_state
from fr.hashing import stable_hash
from fr.native_schedule import set_native_micro_lr
from fr.pilot_partition import load_partition
from fr.stream_rng import seed_future_stream


def parse_csv_ints(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def find_state_record(rows, step):
    for r in rows:
        if int(r["step"]) == int(step):
            return r
    raise KeyError(f"state step {step} 不存在")


def run_one_stream(*, torch, model, optimizer, scaler, autocast_ctx, ds, named,
                   state_path, meta, q_t, q_t_sha, batches, stream_rec,
                   horizons, partition, cfg_src, device, get_lr, stack_batch):
    # [FR-REPLAY-CRITICAL] 每个 stream 从完全相同 Z_t 恢复。
    load_full_training_state(state_path, model, optimizer, scaler)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # [FR-RNG-CRITICAL] 使用 v0.7 formal raw 中保存的 stream_seed。
    seed_future_stream(int(stream_rec["stream_seed"]))

    accum = int(cfg_src["accumulation_steps"])
    max_h = max(horizons)
    if len(batches) < max_h * accum:
        raise RuntimeError("manifest micro-batches 数量不足 max(H)*accum")

    obs = []
    first_preclip = None
    first_clipped = None
    last_lr = None
    for h in range(1, max_h + 1):
        for a in range(accum):
            future_offset = (h - 1) * accum + a + 1
            gm = int(meta["global_micro_step"]) + future_offset
            last_lr = set_native_micro_lr(
                optimizer, get_lr, gm, int(meta["total_micro_steps"]), float(cfg_src["learning_rate"])
            )
            batch_idx = batches[(h - 1) * accum + a]
            x, y = stack_batch(ds, batch_idx, device)
            with autocast_ctx:
                res = model(x, labels=y)
                loss = (res.loss + res.aux_loss) / accum
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        # [FR-PERTURBATION-CRITICAL] E4B-1 永远不注入人工梯度。
        preclip = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg_src["grad_clip"])
        ).detach().cpu())
        if h == 1:
            first_preclip = preclip
            first_clipped = bool(preclip > float(cfg_src["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if h in horizons:
            # [FR-OBSERVATION-ONLY] probe 会自行恢复 RNG，不进入 optimizer。
            q_future, qmeta = probe_direction_observation(
                torch=torch, model=model, optimizer=optimizer, autocast_ctx=autocast_ctx,
                ds=ds, probe_indices=partition["probe_indices"], device=device,
                named=named, stack_batch=stack_batch,
            )
            rq = direction_cosine(q_t, q_future)
            obs.append({
                "horizon": int(h),
                "R_q": float(rq),
                "q_t_sha256": q_t_sha,
                "q_future_sha256": qmeta["q_sha256"],
                "probe_loss_future": qmeta["probe_loss"],
                "probe_grad_norm_future": qmeta["probe_grad_norm_before_normalize"],
            })
            del q_future

    final_model_hash = stable_hash(getattr(model, "module", model).state_dict())
    final_optimizer_hash = stable_hash(optimizer.state_dict())
    expected = stream_rec["baseline"]
    exact = {
        "model_hash_equal": final_model_hash == expected["model_hash_final"],
        "optimizer_hash_equal": final_optimizer_hash == expected["optimizer_hash_final"],
        "first_clipped_equal": bool(first_clipped) == bool(expected["first_clipped"]),
        "last_lr_equal": float(last_lr) == float(expected["last_lr"]),
    }
    exact["pass"] = all(exact.values())
    return {
        "stream_id": int(stream_rec["stream_id"]),
        "stream_seed": int(stream_rec["stream_seed"]),
        "batch_seed": int(stream_rec["batch_seed"]),
        "first_preclip_grad_norm": float(first_preclip),
        "first_clipped": bool(first_clipped),
        "last_lr": float(last_lr),
        "replay_exact": exact,
        "observations": obs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config", default="research/configs/e4b_q_rotation_v08.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--states", default="")
    ap.add_argument("--streams", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="仅首个 seed/state/stream，但仍跑满32步并做最终hash验证")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    cfg_src = json.loads((root / cfg["source_config"]).read_text(encoding="utf-8"))
    src_out = root / cfg_src["output_dir"]
    out = root / (cfg["smoke_output_dir"] if args.smoke else cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    all_seeds = list(map(int, cfg["confirmation_train_seeds"]))
    all_states = list(map(int, cfg["state_steps"]))
    all_streams = list(range(int(cfg["source_streams"])))
    horizons = sorted(map(int, cfg["horizons"]))
    seeds = parse_csv_ints(args.seeds) if args.seeds else all_seeds
    states = parse_csv_ints(args.states) if args.states else all_states
    streams = parse_csv_ints(args.streams) if args.streams else all_streams
    if args.smoke:
        seeds, states, streams = [all_seeds[0]], [all_states[0]], [all_streams[0]]
    if not set(seeds) <= set(all_seeds) or not set(states) <= set(all_states) or not set(streams) <= set(all_streams):
        raise SystemExit("seeds/states/streams 超出 E4B-1 冻结集合")

    minimind = root / "minimind"
    sys.path.insert(0, str(minimind))
    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model
    from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch
    from fr.native_adamw_guard import assert_native_adamw

    partition = load_partition(src_out / "partition_v07.json")
    data_path = src_out / "e3c_subset.jsonl"
    if not data_path.exists():
        raise SystemExit("缺少 v0.7 e3c_subset.jsonl")

    run_summary = []
    all_exact = True
    for seed in seeds:
        seed_src = src_out / f"seed_{seed}"
        raw = json.loads((seed_src / "theta_mapping_raw.json").read_text(encoding="utf-8"))
        bank = json.loads((seed_src / "state_bank_report.json").read_text(encoding="utf-8"))

        mmcfg = MiniMindConfig(
            hidden_size=int(cfg_src["hidden_size"]),
            num_hidden_layers=int(cfg_src["num_hidden_layers"]),
            max_seq_len=int(cfg_src["max_seq_len"]),
            use_moe=False,
        )
        model, tokenizer = init_model(
            mmcfg, from_weight="none", tokenizer_path=str(minimind / "model"), device=args.device
        )
        ds = PretrainDataset(str(data_path), tokenizer, max_length=int(cfg_src["max_seq_len"]))
        autocast_ctx, scaler, _ = make_amp(torch, args.device, cfg_src["dtype"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg_src["learning_rate"]))
        assert_native_adamw(optimizer)
        named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]

        for step in states:
            state_out = out / f"seed_{seed}" / f"state_step_{step:03d}.json"
            if state_out.exists() and not args.force:
                old = json.loads(state_out.read_text(encoding="utf-8"))
                done_streams = {int(x["stream_id"]) for x in old.get("streams", []) if x.get("replay_exact", {}).get("pass")}
            else:
                old = None
                done_streams = set()

            raw_state = find_state_record(raw["states"], step)
            bank_state = find_state_record(bank["states"], step)
            state_path = resolve_recorded_path(root, bank_state["path"])
            meta = load_full_training_state(state_path, model, optimizer, scaler)
            q_path = resolve_recorded_path(root, raw_state["q_path"])
            q_t, q_meta, q_sha = load_direction(q_path)
            if q_sha != raw_state["q_sha256"]:
                raise RuntimeError(f"seed={seed},state={step} q SHA 与 formal raw 不一致")
            if list(map(int, q_meta.get("probe_indices", []))) != list(map(int, partition["probe_indices"])):
                raise RuntimeError(f"seed={seed},state={step} q probe indices 与 partition 不一致")

            existing = {int(x["stream_id"]): x for x in (old or {}).get("streams", [])}
            e2_by_stream = {int(x["stream_id"]): x for x in raw_state["e2_streams"]}
            for r in streams:
                if r in done_streams and not args.force:
                    print(f"SKIP PASS seed={seed} state={step} stream={r}")
                    continue
                rec = e2_by_stream[r]
                manifest_path = resolve_recorded_path(root, rec["batch_manifest_path"])
                batches, manifest_obj, manifest_sha = load_and_verify_manifest(
                    manifest_path, rec["batch_manifest_sha256"]
                )
                if stable_hash(batches) != rec["batch_manifest_hash"]:
                    raise RuntimeError(f"seed={seed},state={step},stream={r} manifest stable_hash 不一致")
                row = run_one_stream(
                    torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                    autocast_ctx=autocast_ctx, ds=ds, named=named,
                    state_path=state_path, meta=meta, q_t=q_t, q_t_sha=q_sha,
                    batches=batches, stream_rec=rec, horizons=horizons,
                    partition=partition, cfg_src=cfg_src, device=args.device,
                    get_lr=get_lr, stack_batch=stack_batch,
                )
                row["batch_manifest_path"] = str(manifest_path)
                row["batch_manifest_sha256"] = manifest_sha
                existing[r] = row
                if not row["replay_exact"]["pass"]:
                    all_exact = False
                    atomic_json(state_out, {
                        "protocol": cfg["protocol_version"], "study_role": cfg["study_role"],
                        "seed": seed, "state_step": step, "state_fraction": float(raw_state["state_fraction"]),
                        "q_t_sha256": q_sha, "streams": [existing[k] for k in sorted(existing)],
                        "technical_pass": False,
                    })
                    raise RuntimeError(f"EXACT REPLAY FAIL seed={seed} state={step} stream={r}: {row['replay_exact']}")
                atomic_json(state_out, {
                    "protocol": cfg["protocol_version"], "study_role": cfg["study_role"],
                    "seed": seed, "state_step": step, "state_fraction": float(raw_state["state_fraction"]),
                    "q_t_sha256": q_sha,
                    "B_theta": raw_state["B_theta"], "chi_theta": raw_state["chi_theta"],
                    "streams": [existing[k] for k in sorted(existing)],
                    "technical_pass": all(x["replay_exact"]["pass"] for x in existing.values()),
                })
                print(f"PASS seed={seed} state={step} stream={r} Rq8=" +
                      f"{next(x['R_q'] for x in row['observations'] if int(x['horizon']) == 8):.6f}")

            # state summary 只有冻结的全部 streams 齐全时才算 complete。
            final_obj = json.loads(state_out.read_text(encoding="utf-8"))
            final_obj["complete"] = len(final_obj.get("streams", [])) == int(cfg["source_streams"])
            atomic_json(state_out, final_obj)
            run_summary.append({"seed": seed, "state_step": step, "complete": final_obj["complete"]})

        del model, optimizer, scaler, ds
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    summary = {
        "protocol": cfg["protocol_version"],
        "study_role": cfg["study_role"],
        "smoke": bool(args.smoke),
        "requested_seeds": seeds, "requested_states": states, "requested_streams": streams,
        "horizons": horizons,
        "all_exact_replay_pass_this_run": bool(all_exact),
        "states_touched": run_summary,
    }
    atomic_json(out / "run_summary.json", summary)
    print("E4B1_REPLAY_INTEGRITY =", "PASS" if all_exact else "FAIL")
    print("E4B1_OUTPUT =", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
