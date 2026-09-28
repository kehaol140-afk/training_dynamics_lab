#!/usr/bin/env python3
"""v0.7 confirmatory FR 的 GPU 前科学门禁与 prereg lock 工具。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


LOCKED_RELATIVE_PATHS = [
    "research/configs/e3c_confirmatory_v07.json",
    "research/src/fr/confirmatory_stats.py",
    "research/scripts/e3c_analyze_confirmatory_v07.py",
    "CONFIRMATORY_PREREGISTRATION_V07.md",
    "research/scripts/e3c_prepare_state_bank_v07.py",
    "research/scripts/e3c_multiseed_theta_v07.py",
    "research/src/fr/theta_protocol.py",
    "research/src/fr/full_state.py",
    "research/src/fr/rng.py",
    "research/src/fr/direction.py",
    "research/src/fr/perturbation.py",
    "research/src/fr/native_schedule.py",
    "research/src/fr/stream_rng.py",
    "research/src/fr/batch_manifest.py",
    "research/src/fr/pilot_partition.py",
    "research/src/fr/stage_c_runtime.py",
    "research/src/fr/response_stats.py",
]

DEPENDENCY_RELATIVE_PATHS = [
    "minimind/trainer/train_pretrain.py",
    "minimind/trainer/trainer_utils.py",
    "minimind/dataset/lm_dataset.py",
    "minimind/model/model_minimind.py",
]

CURRENT_LOCK_REVISION = 2
PREVIOUS_LOCK_SHA256 = "4053024c178fb061e7c8a3b4a9b6c89908a5034c07c390b75796b5e09410a81b"
LOCK_REVISION_REASON = (
    "Expand prereg lock to all confirmatory data-generation and scientific-critical "
    "dependencies before any confirmatory data were generated."
)
PREFLIGHT_ARTIFACT_NAMES = {
    "ENVIRONMENT_V07.txt",
    "scientific_preflight_v07.json",
    "PREREG_LOCK.json",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def locked_file_hashes(root: Path, paths=None):
    out = []
    for rel in (paths or LOCKED_RELATIVE_PATHS):
        p = root / rel
        if not p.exists():
            raise FileNotFoundError(f"lock file missing: {rel}")
        out.append({"path": rel, "sha256": file_sha256(p)})
    return out


def formal_confirmatory_scientific_data_files(out: Path):
    """列出正式 confirmatory 科学数据文件，排除 preflight/lock/env 产物。"""
    out = Path(out)
    if not out.exists():
        return []
    files = []
    for p in out.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(out)
        if len(rel.parts) == 1 and rel.name in PREFLIGHT_ARTIFACT_NAMES:
            continue
        files.append(str(p))
    return sorted(files)


def _hashes_by_path(entries):
    return {x["path"]: x["sha256"] for x in entries}


def compare_lock_to_current(root: Path, lock_path: Path):
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    current_locked = _hashes_by_path(locked_file_hashes(root, LOCKED_RELATIVE_PATHS))
    locked = _hashes_by_path(lock.get("locked_files", []))
    current_dependencies = _hashes_by_path(locked_file_hashes(root, DEPENDENCY_RELATIVE_PATHS))
    dependencies = _hashes_by_path(lock.get("dependency_files", []))
    mismatches = []
    for rel, sha in current_locked.items():
        if locked.get(rel) != sha:
            mismatches.append({
                "path": rel,
                "role": "locked_file",
                "locked_sha256": locked.get(rel),
                "current_sha256": sha,
            })
    for rel, sha in current_dependencies.items():
        if dependencies.get(rel) != sha:
            mismatches.append({
                "path": rel,
                "role": "dependency_file",
                "locked_sha256": dependencies.get(rel),
                "current_sha256": sha,
            })
    unexpected_locked = sorted(set(locked) - set(current_locked))
    unexpected_dependencies = sorted(set(dependencies) - set(current_dependencies))
    for rel in unexpected_locked:
        mismatches.append({
            "path": rel,
            "role": "unexpected_locked_file",
            "locked_sha256": locked.get(rel),
            "current_sha256": None,
        })
    for rel in unexpected_dependencies:
        mismatches.append({
            "path": rel,
            "role": "unexpected_dependency_file",
            "locked_sha256": dependencies.get(rel),
            "current_sha256": None,
        })
    revision_matches = int(lock.get("lock_revision", 1)) == CURRENT_LOCK_REVISION
    if not revision_matches:
        mismatches.append({
            "path": "PREREG_LOCK.json",
            "role": "lock_revision",
            "locked_sha256": str(lock.get("lock_revision")),
            "current_sha256": str(CURRENT_LOCK_REVISION),
        })
    return {
        "matches": not mismatches,
        "mismatches": mismatches,
        "lock_revision_matches": revision_matches,
        "lock": lock,
    }


def assert_prereg_lock_valid(root: Path, out: Path | None = None):
    """正式 v0.7 数据产生前后都必须通过的 lock 校验。"""
    out = out or (root / "outputs/e3_confirmatory/v07")
    lock_path = out / "PREREG_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError("正式 v0.7 confirmatory run 前必须先生成 PREREG_LOCK.json")
    status = compare_lock_to_current(root, lock_path)
    if not status["matches"]:
        raise RuntimeError(
            "PREREG_LOCK 与当前 locked files 不一致；必须升级 protocol version，"
            "并明确旧 confirmatory run 作废。"
        )
    return status["lock"]


def build_prereg_lock_payload(
    root: Path,
    out: Path,
    cfg: dict,
    *,
    previous_lock_sha256: str = PREVIOUS_LOCK_SHA256,
):
    formal_data = formal_confirmatory_scientific_data_files(out)
    formal_data_existed = bool(formal_data)
    lock_path = out / "PREREG_LOCK.json"
    return {
        "protocol": "E3C_CONFIRMATORY_V07_PREREG_LOCK",
        "lock_revision": CURRENT_LOCK_REVISION,
        "previous_lock_sha256": previous_lock_sha256,
        "revision_reason": LOCK_REVISION_REASON,
        "formal_confirmatory_data_existed_before_revision": formal_data_existed,
        "formal_confirmatory_scientific_data_file_count_before_revision": len(formal_data),
        "formal_confirmatory_scientific_data_files_before_revision": formal_data,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "locked_files": locked_file_hashes(root, LOCKED_RELATIVE_PATHS),
        "dependency_files": locked_file_hashes(root, DEPENDENCY_RELATIVE_PATHS),
        "locked_file_count": len(LOCKED_RELATIVE_PATHS),
        "dependency_file_count": len(DEPENDENCY_RELATIVE_PATHS),
        "confirmation_train_seeds": list(map(int, cfg["confirmation_train_seeds"])),
        "discovery_train_seeds": list(map(int, cfg["discovery_train_seeds"])),
        "discovery_role": cfg["discovery_role"],
        "primary_endpoint": {
            "horizon": int(cfg["primary_horizon"]),
            "conditions": [
                "A: K_8 > 0 and training-seed cluster bootstrap 95% CI lower > 0",
                "B: within-seed detrended circular-shift one-sided p < 0.05, positive direction",
                "C: 8-fold LOSO mean relative MAE improvement > 0 and at least 6/8 folds improved",
            ],
        },
        "bootstrap_reps": int(cfg["confirmatory_bootstrap_reps"]),
        "permutation_reps": int(cfg["confirmatory_permutation_reps"]),
        "random_seed": int(cfg["confirmatory_random_seed"]),
        "lock_rule": (
            "正式 confirmatory 数据开始产生后，locked_files/dependency_files 禁止修改；"
            "如必须修改，必须生成新的 protocol version 并废弃旧 run。"
        ),
    }


def write_prereg_lock(root: Path, out: Path, cfg: dict):
    lock_path = out / "PREREG_LOCK.json"
    if lock_path.exists():
        lock_sha = file_sha256(lock_path)
        status = compare_lock_to_current(root, lock_path)
        if status["matches"]:
            return status["lock"]
        formal_data = formal_confirmatory_scientific_data_files(out)
        old_revision = int(status["lock"].get("lock_revision", 1))
        allowed_revision = (
            old_revision < CURRENT_LOCK_REVISION
            and not formal_data
            and lock_sha == PREVIOUS_LOCK_SHA256
        )
        if not allowed_revision:
            raise RuntimeError("已有 PREREG_LOCK 与当前文件不一致，禁止覆盖。")

    lock = build_prereg_lock_payload(root, out, cfg)
    if lock["formal_confirmatory_data_existed_before_revision"]:
        raise RuntimeError("已有正式 confirmatory 科学数据，禁止修改 prereg lock。")
    out.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    return lock


def ordered_positions(text: str, patterns):
    positions = []
    start = 0
    for pattern in patterns:
        idx = text.find(pattern, start)
        if idx < 0:
            return None
        positions.append(idx)
        start = idx + 1
    return positions


def collect_environment(root: Path, out: Path) -> dict:
    env_name = os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
    info = {
        "python_version": sys.version.replace("\n", " "),
        "sys_prefix": sys.prefix,
        "conda_env": env_name,
        "torch_version": None,
        "torch_cuda": None,
        "cuda_available": None,
        "gpus": [],
        "conda_list_relevant": [],
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                info["gpus"].append({
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory": int(props.total_memory),
                    "capability": list(torch.cuda.get_device_capability(idx)),
                })
    except Exception as exc:
        info["torch_error"] = repr(exc)

    try:
        proc = subprocess.run(
            ["conda", "list"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        relevant = ("python", "numpy", "torch", "pytorch", "cuda", "cudnn", "triton", "transformers", "tokenizers")
        for line in proc.stdout.splitlines():
            low = line.lower()
            if any(name in low.split()[:1] or name in low for name in relevant):
                info["conda_list_relevant"].append(line)
        if proc.returncode != 0:
            info["conda_list_error"] = proc.stderr.strip()
    except Exception as exc:
        info["conda_list_error"] = repr(exc)

    lines = [
        "# ENVIRONMENT V07",
        "",
        f"python version: {info['python_version']}",
        f"sys.prefix: {info['sys_prefix']}",
        f"conda env: {info['conda_env']}",
        f"torch version: {info['torch_version']}",
        f"torch CUDA: {info['torch_cuda']}",
        f"cuda available: {info['cuda_available']}",
        "",
        "## GPU",
    ]
    if info["gpus"]:
        for gpu in info["gpus"]:
            lines.append(
                f"- cuda:{gpu['index']} {gpu['name']} memory={gpu['total_memory']} capability={gpu['capability']}"
            )
    else:
        lines.append("- NONE")
    lines += ["", "## conda list relevant packages"]
    lines.extend(info["conda_list_relevant"] or ["- NONE"])

    env_path = root / "ENVIRONMENT_V07.txt"
    out.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "ENVIRONMENT_V07.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    info["environment_path"] = str(env_path)
    info["environment_archive_path"] = str(out / "ENVIRONMENT_V07.txt")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config", default="research/configs/e3c_confirmatory_v07.json")
    ap.add_argument("--write-prereg-lock", action="store_true")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "research/src"))
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    out = root / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    checks = {}
    expected_confirmation = list(range(2030, 2038))
    expected_discovery = [2026, 2027, 2028, 2029]
    expected_states = [46, 52, 58, 64, 70, 76, 82]
    expected_horizons = [1, 2, 4, 8, 12, 16, 24, 32]

    checks["protocol_version"] = cfg.get("protocol_version") == "E3C_CONFIRMATORY_V07"
    checks["discovery_marked_only"] = cfg.get("discovery_role") == "DISCOVERY_ONLY_DO_NOT_USE_FOR_PRIMARY_CONFIRMATORY"
    checks["discovery_seeds_exact"] = list(map(int, cfg["discovery_train_seeds"])) == expected_discovery
    checks["confirmation_seeds_exact"] = list(map(int, cfg["confirmation_train_seeds"])) == expected_confirmation
    checks["discovery_confirmation_disjoint"] = not (set(cfg["discovery_train_seeds"]) & set(cfg["confirmation_train_seeds"]))
    checks["horizons_exact"] = list(map(int, cfg["horizons"])) == expected_horizons
    checks["primary_horizon_H8"] = int(cfg["primary_horizon"]) == 8
    checks["state_grid_exact"] = list(map(int, cfg["e3_state_steps"])) == expected_states
    checks["rho_fixed_0_25"] = float(cfg["e3_primary_rho"]) == 0.25
    checks["E1_R_16"] = int(cfg["e3_e1_streams"]) == 16
    checks["E2_R_8"] = int(cfg["e3_e2_streams"]) == 8
    checks["bootstrap_reps_10000"] = int(cfg["confirmatory_bootstrap_reps"]) == 10000
    checks["permutation_reps_20000"] = int(cfg["confirmatory_permutation_reps"]) == 20000
    checks["conda_env_fr_minimind_config"] = cfg.get("environment_conda_env") == "fr_minimind"
    checks["state_plus_maxH_within_budget"] = max(expected_states) + max(expected_horizons) <= int(cfg["planned_total_optimizer_steps"])
    checks["exclude_probe_eval_from_trajectory_step0"] = cfg.get("exclude_probe_eval_from_trajectory_step0") is True

    train = (root / "minimind/trainer/train_pretrain.py").read_text(encoding="utf-8")
    checks["native_train_order"] = ordered_positions(train, [
        "lr = get_lr(",
        "with autocast_ctx:",
        "scaler.scale(loss).backward()",
        "scaler.unscale_(optimizer)",
        "clip_grad_norm_",
        "scaler.step(optimizer)",
        "optimizer.zero_grad(set_to_none=True)",
    ]) is not None

    theta = (root / "research/src/fr/theta_protocol.py").read_text(encoding="utf-8")
    checks["fr_branch_perturbation_order"] = ordered_positions(theta, [
        "set_native_micro_lr(",
        "with autocast_ctx:",
        "scaler.scale(loss).backward()",
        "scaler.unscale_(optimizer)",
        "apply_signed_gradient_impulse_(",
        "clip_grad_norm_",
        "scaler.step(optimizer)",
        "optimizer.zero_grad(set_to_none=True)",
    ]) is not None

    full = (root / "research/src/fr/full_state.py").read_text(encoding="utf-8")
    checks["full_state_model"] = '"model"' in full
    checks["full_state_optimizer"] = '"optimizer"' in full
    checks["full_state_capture_rng"] = "capture_rng_state()" in full
    checks["full_state_restore_rng"] = 'restore_rng_state(state["rng"])' in full

    from fr import confirmatory_stats

    checks["confirmatory_stats_bootstrap_default"] = confirmatory_stats.DEFAULT_BOOTSTRAP_REPS == 10000
    checks["confirmatory_stats_permutation_default"] = confirmatory_stats.DEFAULT_PERMUTATION_REPS == 20000
    checks["locked_files_exist"] = all((root / rel).exists() for rel in LOCKED_RELATIVE_PATHS)
    checks["dependency_files_exist"] = all((root / rel).exists() for rel in DEPENDENCY_RELATIVE_PATHS)
    checks["locked_file_count_17"] = len(LOCKED_RELATIVE_PATHS) == 17
    checks["dependency_file_count_4"] = len(DEPENDENCY_RELATIVE_PATHS) == 4

    env = collect_environment(root, out)
    current_env = env["conda_env"]
    checks["running_in_fr_minimind"] = current_env == "fr_minimind" or Path(sys.prefix).name == "fr_minimind"
    checks["torch_cuda_available"] = env.get("cuda_available") is True

    lock_path = out / "PREREG_LOCK.json"
    formal_data = formal_confirmatory_scientific_data_files(out)
    checks["formal_confirmatory_scientific_data_absent"] = len(formal_data) == 0
    if lock_path.exists():
        current_lock_sha = file_sha256(lock_path)
        lock_status = compare_lock_to_current(root, lock_path)
        old_revision = int(lock_status["lock"].get("lock_revision", 1))
        revision_update_allowed = (
            args.write_prereg_lock
            and old_revision < CURRENT_LOCK_REVISION
            and len(formal_data) == 0
            and current_lock_sha == PREVIOUS_LOCK_SHA256
        )
        checks["existing_prereg_lock_matches_current_files"] = (
            lock_status["matches"] or revision_update_allowed
        )
        checks["old_lock_revision_update_allowed"] = (
            True if lock_status["matches"] else revision_update_allowed
        )
    else:
        lock_status = None
        current_lock_sha = None
        checks["existing_prereg_lock_matches_current_files"] = True
        checks["old_lock_revision_update_allowed"] = True

    passed = all(checks.values())
    lock = None
    if args.write_prereg_lock:
        if not passed:
            raise SystemExit("preflight 未通过，禁止写入 PREREG_LOCK.json")
        lock = write_prereg_lock(root, out, cfg)

    report = {
        "protocol": "E3C_CONFIRMATORY_V07_SCIENTIFIC_PREFLIGHT",
        "pass": passed,
        "checks": checks,
        "environment": env,
        "formal_confirmatory_scientific_data_file_count": len(formal_data),
        "formal_confirmatory_scientific_data_files": formal_data,
        "lock_written_or_validated": lock is not None,
        "lock_path": str(lock_path),
        "lock_sha256_before_optional_write": current_lock_sha,
        "lock": lock,
        "existing_lock_status": lock_status,
    }
    (out / "scientific_preflight_v07.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("V07_SCIENTIFIC_PREFLIGHT_PASS" if passed else "V07_SCIENTIFIC_PREFLIGHT_FAIL")
    for key, value in checks.items():
        print(f"{key}={value}")
    print("environment=", env["environment_path"])
    if lock is not None:
        print("prereg_lock=", lock_path)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
