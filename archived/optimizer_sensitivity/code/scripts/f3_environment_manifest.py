#!/usr/bin/env python3
"""Capture the frozen environment manifest for the F3 current-environment
compatibility study (amendment docs/F3_REPRODUCTION_GATE_AMENDMENT.md §3).
Run in the SAME shell environment as the compat/R0/R1/F3 runs, immediately
before the first compat computation. Read-only: changes nothing."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LAB = ROOT.parent

OUT = ROOT / "outputs" / "f3_compatibility"
OUT.mkdir(parents=True, exist_ok=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True).stdout.strip()


import torch

manifest = {
    "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "git": {
        "HEAD": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": git("status", "--porcelain"),
    },
    "python": {
        "version": platform.python_version(),
        "executable": sys.executable,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    },
    "torch": {
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "num_threads": torch.get_num_threads(),
    },
    "gpu": {},
    "env_vars": {
        k: os.environ.get(k)
        for k in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG",
                  "CUBLASLT_WORKSPACE_SIZE", "CUDA_LAUNCH_BLOCKING",
                  "CUDA_MODULE_LOADING", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
    },
    "sdpa": {},
    "primal_precision": {
        "forward": "bf16 autocast (GPU), math-SDPA only inside HVP (fp32)",
        "tangent": "fp64 CPU",
        "grads_to_cpu": "float32",
    },
    "cell_inputs": {},
    "code_files": {},
}

try:
    import numpy
    manifest["numpy"] = numpy.__version__
except Exception:
    manifest["numpy"] = None
try:
    import transformers
    manifest["transformers"] = transformers.__version__
except Exception:
    manifest["transformers"] = None

try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    manifest["gpu"] = {"nvidia_smi_rows": out.strip().splitlines()}
except Exception as exc:
    manifest["gpu"] = {"error": str(exc)}

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    manifest["sdpa"] = {
        "default_backward_order_note": "eager SDPA (flash/cudnn/mem_efficient/math) "
                                       "for primal; MATH forced inside rig.hvp",
        "math_for_hvp": True,
    }
except Exception as exc:
    manifest["sdpa"] = {"error": str(exc)}

# ---- cell inputs: bank checkpoint hashes + future-stream manifests --------
from opsens import f0_bank_audit_v01 as f0
from opsens import f1_mechanism_v01 as fm
from opsens.batch_manifest import make_permutation, manifest_sha256
from opsens.stream_rng import derive_stream_seed

CELL_STREAM = {(2042, 26): 1, (2043, 26): 3, (2042, 64): 2}
BATCH_SIZE = 4

from f2_hypergradient import Rig  # noqa: E402  (sys.path has scripts/)

rig = Rig("cuda:0")
for (seed, step), sid in sorted(CELL_STREAM.items()):
    bank = Path(fm.bank_state_path("g1b_100m", seed, step))
    batches, stream_seed = rig.batches_for(sid)
    msha = manifest_sha256(batches)
    manifest["cell_inputs"][f"{seed}:{step}"] = {
        "bank_state_path": str(bank),
        "bank_state_sha256": sha256_file(bank),
        "stream_seed": stream_seed,
        "future_stream_manifest_sha256": msha,
        "n_batches": len(batches),
    }

# ---- code file hashes ------------------------------------------------------
for rel in ("scripts/f2_hypergradient.py", "scripts/f3_curvature_mechanism.py",
            "src/opsens/f3_curvature_mechanism_v01.py",
            "src/opsens/a0c_adamw_response_propagation_v01.py",
            "src/opsens/f0_bank_audit_v01.py", "src/opsens/f1_mechanism_v01.py",
            "src/opsens/batch_manifest.py", "src/opsens/stream_rng.py",
            "../minimind_base/model/model_minimind.py",
            "../minimind_base/dataset/lm_dataset.py",
            "docs/F3_CURVATURE_MECHANISM_PREREG.md",
            "docs/F3_REPRODUCTION_GATE_AMENDMENT.md"):
    p = (ROOT / rel).resolve()
    if p.exists():
        manifest["code_files"][rel] = sha256_file(p)

dst = OUT / "environment_manifest.json"
dst.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print("wrote", dst)
print(json.dumps({k: manifest[k] for k in ("git", "torch", "gpu")}, indent=1)[:800])
