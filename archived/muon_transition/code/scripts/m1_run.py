"""Launch the frozen M1 matrix (configs/m1_baseline_prereg.json): 12 runs, 2 GPUs.

Usage: python scripts/m1_run.py [--smoke-after] ; writes outputs/m1/<run_id>/ per cell and
outputs/m1/m1_summary.json + M1_SUMMARY.md at the end.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PY = sys.executable


def cell_worker(cells_json: str, gpu: str):
    from muon_transition import config
    from muon_transition.m1_runner import prepare_data, run_cell
    import torch

    torch.cuda.set_device(0)
    data = prepare_data()
    cells = json.loads(cells_json)
    for cell in cells:
        run_cell(
            arm=cell["arm"], lr_multiplier=cell["lr_multiplier"], weight_decay=cell["weight_decay"],
            p=cell["p"], seed=cell["seed"], run_id=cell["run_id"],
            out_dir=config.OUTPUTS / "m1" / cell["run_id"],
            device="cuda:0", data=data,
        )


def main():
    from muon_transition import config

    prereg = json.loads((ROOT / "configs" / "m1_baseline_prereg.json").read_text())
    cells = prereg["m1_runs"]
    done = [c for c in cells if (config.OUTPUTS / "m1" / c["run_id"] / "result.json").exists()]
    todo = [c for c in cells if c not in done]
    print(f"M1: {len(cells)} cells, {len(done)} already done, {len(todo)} to run", flush=True)
    if not todo:
        summarize(cells)
        return

    half = (len(todo) + 1) // 2
    shards = [todo[:half], todo[half:]]
    procs = []
    for gpu, shard in zip(("0", "1"), shards):
        if not shard:
            continue
        script = f'''
import sys, json
from pathlib import Path
sys.path.insert(0, "{ROOT}")
sys.path.insert(0, "{ROOT / 'src'}")
from scripts.m1_run import cell_worker
cell_worker(json.loads({json.dumps(json.dumps(shard))!r}), "{gpu}")
'''
        (ROOT / "outputs" / "m1").mkdir(parents=True, exist_ok=True)
        sf = ROOT / "outputs" / "m1" / f"worker_gpu{gpu}.py"
        sf.write_text(script)
        env = {"CUDA_VISIBLE_DEVICES": gpu, "PATH": "/usr/bin:/bin"}
        import os

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        procs.append(subprocess.Popen([PY, str(sf)], cwd=str(ROOT), env=env))
        print(f"launched GPU {gpu}: {len(shard)} cells: {[c['run_id'] for c in shard]}", flush=True)

    fail = False
    for p in procs:
        rc = p.wait()
        if rc != 0:
            fail = True
    if fail:
        raise SystemExit("a worker failed")
    summarize(cells)


def summarize(cells):
    from muon_transition import config

    rows = []
    for c in cells:
        rj = config.OUTPUTS / "m1" / c["run_id"] / "result.json"
        if not rj.exists():
            rows.append({**c, "status": "MISSING"})
            continue
        r = json.loads(rj.read_text())
        rows.append(
            {
                **{k: c[k] for k in ("run_id", "arm", "lr", "weight_decay", "lr_multiplier")},
                "status": "done",
                "final_V_Q": r["final_dev"]["v_q"],
                "NLL_A": r["final_dev"]["nll_A"],
                "NLL_B": r["final_dev"]["nll_B"],
                "final_train_loss": r["final_train_loss"],
                "loss_decreased": r["loss_decreased"],
                "no_nan_inf": (not r["any_nan_loss"]) and (not r["any_nonfinite_gradnorm"]),
                "wall_s": round(r["wall_seconds"], 1),
            }
        )

    complete = [r for r in rows if r.get("status") == "done"]
    for arm in ("adamw", "muon"):
        arm_rows = [r for r in complete if r["arm"] == arm]
        if arm_rows:
            best = min(arm_rows, key=lambda r: r["final_V_Q"])
            rows.append({**{k: best[k] for k in ("final_V_Q", "NLL_A", "NLL_B", "run_id")},
                         "run_id": f"BEST_{arm}", "best_cell": best["run_id"]})

    out = config.OUTPUTS / "m1"
    (out / "m1_summary.json").write_text(json.dumps(rows, indent=2))

    lines = ["# M1 summary (p=0.5, 12 runs, seed 2030, B0=2^26)", "",
             "| run | arm | lr | wd | V_Q | NLL_A | NLL_B | train_loss(末) | ok | wall_s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("status") == "MISSING":
            lines.append(f"| {r['run_id']} | {r['arm']} | {r['lr']} | {r['weight_decay']} | MISSING | | | | | |")
            continue
        if r["run_id"].startswith("BEST_"):
            lines.append(f"| **{r['run_id']}** | | | | **{r['final_V_Q']:.4f}** | {r['NLL_A']:.4f} | {r['NLL_B']:.4f} | | | |")
            continue
        if r["run_id"].startswith("BEST_") or r.get("status") == "MISSING":
            continue
        ok = r["loss_decreased"] and r["no_nan_inf"]
        lines.append(f"| {r['run_id']} | {r['arm']} | {r['lr']:g} | {r['weight_decay']:g} | "
                     f"{r['final_V_Q']:.4f} | {r['NLL_A']:.4f} | {r['NLL_B']:.4f} | "
                     f"{r['final_train_loss']:.4f} | {'OK' if ok else 'CHECK'} | {r['wall_s']} |")
    gate = all(r.get("status") != "MISSING" for r in rows) and all(
        r.get("loss_decreased") and r.get("no_nan_inf") for r in complete
    )
    n_stable = {arm: sum(1 for r in complete if r["arm"] == arm and r["loss_decreased"] and r["no_nan_inf"])
                for arm in ("adamw", "muon")}
    lines += ["", f"stable cells per arm: {n_stable}",
              f"M1 gate (both arms train normally): {'PASS' if gate and all(v >= 1 for v in n_stable.values()) else 'PENDING/FAIL'}"]
    (out / "M1_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
