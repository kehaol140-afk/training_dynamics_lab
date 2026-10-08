"""Launch/resume M2 cells from configs/m2_prereg.json across 2 GPUs."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PY = sys.executable


def cell_worker(cells_json: str):
    from muon_transition import config
    from muon_transition.m1_runner import prepare_data, run_cell
    import torch

    torch.cuda.set_device(0)
    data = prepare_data()
    for cell in json.loads(cells_json):
        out_dir = config.OUTPUTS / "m2" / cell["run_id"]
        if (out_dir / "result.json").exists():
            continue
        run_cell(
            arm=cell["arm"], lr_multiplier=cell["lr_multiplier"], weight_decay=cell["weight_decay"],
            p=cell["p"], seed=cell["seed"], run_id=cell["run_id"],
            out_dir=out_dir, device="cuda:0", data=data,
        )


def launch(all_cells, tag):
    todo = all_cells
    if not todo:
        return []
    half = (len(todo) + 1) // 2
    shards = [todo[:half], todo[half:]]
    procs = []
    for gpu, shard in zip(("0", "1"), shards):
        if not shard:
            continue
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
            "from scripts.m2_run import cell_worker\n"
            f"cell_worker(json.loads({json.dumps(json.dumps(shard))!r}))\n"
        )
        (ROOT / "outputs" / "m2").mkdir(parents=True, exist_ok=True)
        sf = ROOT / "outputs" / "m2" / f"worker_{tag}_gpu{gpu}.py"
        sf.write_text(script)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        procs.append(subprocess.Popen([PY, str(sf)], cwd=str(ROOT), env=env))
        print(f"[{tag}] GPU {gpu}: {len(shard)} cells: {[c['run_id'] for c in shard]}", flush=True)
    for p in procs:
        rc = p.wait()
        if rc != 0:
            raise SystemExit(f"worker failed rc={rc}")


def pending(cells):
    from muon_transition import config

    return [c for c in cells if not (config.OUTPUTS / "m2" / c["run_id"] / "result.json").exists()]


def main():
    prereg = json.loads((ROOT / "configs" / "m2_prereg.json").read_text())
    cells = prereg["new_runs"] + prereg["justified_extension_runs"]
    todo = pending(cells)
    print(f"M2: {len(cells)} local cells, {len(cells) - len(todo)} done, {len(todo)} to run", flush=True)
    if todo:
        launch(todo, "base")
    print("M2 local cells complete", flush=True)


if __name__ == "__main__":
    main()
