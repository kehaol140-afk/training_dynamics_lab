"""E4B-0 动态状态数据可用性审计。

只读扫描 v0.7 confirmation 产物，判断 q-direction rotation 是否可直接从既有数据计算，
或是否需要 baseline-only replay。该模块不修改 v0.7 数据，不执行 GPU 训练。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


# [FR-HASH]
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_json(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """把嵌套 JSON 展开为 (路径, 值)，便于兼容不同历史版本字段名。"""
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.extend(flatten_json(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            out.extend(flatten_json(v, p))
    else:
        out.append((prefix, obj))
    return out


def match_paths(flat: Iterable[tuple[str, Any]], patterns: Iterable[str]) -> list[tuple[str, Any]]:
    pats = [re.compile(p, re.I) for p in patterns]
    hits: list[tuple[str, Any]] = []
    for key, value in flat:
        if any(p.search(key) for p in pats):
            hits.append((key, value))
    return hits


def existing_file_refs(root: Path, flat: Iterable[tuple[str, Any]]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for key, value in flat:
        if not isinstance(value, str):
            continue
        s = value.strip()
        if not s or len(s) > 4096:
            continue
        # [FR-IO] 只尝试看起来像文件路径的字符串。
        if not any(tok in s.lower() for tok in (".json", ".jsonl", ".pt", ".pth", ".bin", ".txt", ".csv")):
            continue
        p = Path(s)
        if not p.is_absolute():
            p = root / p
        refs.append({"json_path": key, "raw": s, "resolved": str(p), "exists": str(p.exists())})
    return refs


@dataclass
class SeedAudit:
    seed: int
    theta_raw: str | None
    theta_raw_exists: bool
    state_report: str | None
    state_report_exists: bool
    state_files_found: int
    current_q_tensor_hits: int
    current_q_hash_hits: int
    future_q_tensor_hits: int
    future_q_hash_hits: int
    future_probe_gradient_hits: int
    baseline_future_state_hits: int
    optimizer_future_state_hits: int
    manifest_hits: int
    batch_hash_hits: int
    rng_hits: int
    paired_branch_hits: int
    x_theta_hits: int
    referenced_existing_files: int


CURRENT_Q_PATTERNS = [
    r"(^|\.)q(_tensor|_path|_file)?$",
    r"current.*q",
    r"probe.*direction",
]
CURRENT_Q_HASH_PATTERNS = [r"q.*sha", r"q.*hash", r"direction.*sha", r"direction.*hash"]
FUTURE_Q_PATTERNS = [r"future.*q", r"q.*future", r"q_t\+", r"q_future"]
FUTURE_Q_HASH_PATTERNS = [r"future.*q.*(sha|hash)", r"q.*future.*(sha|hash)"]
FUTURE_PROBE_PATTERNS = [r"future.*probe.*grad", r"probe.*grad.*future", r"future_gradient"]
BASELINE_STATE_PATTERNS = [r"baseline.*(model|state|checkpoint)", r"future.*model.*state"]
OPT_STATE_PATTERNS = [r"baseline.*optimizer", r"future.*optimizer.*state"]
MANIFEST_PATTERNS = [r"manifest", r"batch.*path", r"future.*batch"]
BATCH_HASH_PATTERNS = [r"batch.*hash", r"batch.*sha"]
RNG_PATTERNS = [r"rng", r"random.*state", r"stream.*seed", r"batch.*seed"]
PAIRED_PATTERNS = [r"branch", r"plus", r"minus", r"paired", r"rho0", r"epsilon"]
XTHETA_PATTERNS = [r"x_theta", r"theta.*displacement", r"parameter.*displacement"]


def _count(flat: list[tuple[str, Any]], patterns: list[str]) -> int:
    return len(match_paths(flat, patterns))


def audit_seed(project_root: Path, seed: int, expected_states: list[int]) -> tuple[SeedAudit, dict[str, Any]]:
    base = project_root / "outputs" / "e3_confirmatory" / "v07" / f"seed_{seed}"
    theta = base / "theta_mapping_raw.json"
    state_report = base / "state_bank_report.json"

    combined: dict[str, Any] = {}
    for label, path in (("theta", theta), ("state_report", state_report)):
        if path.exists():
            try:
                combined[label] = load_json(path)
            except Exception as exc:  # [FR-IO]
                combined[label] = {"_load_error": repr(exc)}

    flat = flatten_json(combined)
    refs = existing_file_refs(project_root, flat)

    state_files = []
    for s in expected_states:
        p = base / "states" / f"state_step_{s:03d}.pt"
        if p.exists():
            state_files.append(p)
        else:
            # 兼容历史目录里 state 文件直接位于 seed 目录。
            alt = base / f"state_step_{s:03d}.pt"
            if alt.exists():
                state_files.append(alt)

    # 目录级兜底扫描，仅匹配明确 future/q/state 文件名，不读取大型 tensor。
    file_names = [p.name.lower() for p in base.rglob("*") if p.is_file()]
    future_q_file_count = sum(
        1 for n in file_names if ("future" in n and ("q_" in n or n.startswith("q") or "direction" in n))
    )
    future_state_file_count = sum(1 for n in file_names if "baseline" in n and ("state" in n or "checkpoint" in n))

    audit = SeedAudit(
        seed=seed,
        theta_raw=str(theta) if theta.exists() else None,
        theta_raw_exists=theta.exists(),
        state_report=str(state_report) if state_report.exists() else None,
        state_report_exists=state_report.exists(),
        state_files_found=len(state_files),
        current_q_tensor_hits=_count(flat, CURRENT_Q_PATTERNS),
        current_q_hash_hits=_count(flat, CURRENT_Q_HASH_PATTERNS),
        future_q_tensor_hits=_count(flat, FUTURE_Q_PATTERNS) + future_q_file_count,
        future_q_hash_hits=_count(flat, FUTURE_Q_HASH_PATTERNS),
        future_probe_gradient_hits=_count(flat, FUTURE_PROBE_PATTERNS),
        baseline_future_state_hits=_count(flat, BASELINE_STATE_PATTERNS) + future_state_file_count,
        optimizer_future_state_hits=_count(flat, OPT_STATE_PATTERNS),
        manifest_hits=_count(flat, MANIFEST_PATTERNS),
        batch_hash_hits=_count(flat, BATCH_HASH_PATTERNS),
        rng_hits=_count(flat, RNG_PATTERNS),
        paired_branch_hits=_count(flat, PAIRED_PATTERNS),
        x_theta_hits=_count(flat, XTHETA_PATTERNS),
        referenced_existing_files=sum(1 for r in refs if r["exists"] == "True"),
    )

    details = {
        "base_dir": str(base),
        "state_files": [str(p) for p in state_files],
        "json_file_refs": refs[:200],
        "keyword_hits": {
            "current_q": match_paths(flat, CURRENT_Q_PATTERNS)[:50],
            "current_q_hash": match_paths(flat, CURRENT_Q_HASH_PATTERNS)[:50],
            "future_q": match_paths(flat, FUTURE_Q_PATTERNS)[:50],
            "future_q_hash": match_paths(flat, FUTURE_Q_HASH_PATTERNS)[:50],
            "future_probe_gradient": match_paths(flat, FUTURE_PROBE_PATTERNS)[:50],
            "baseline_future_state": match_paths(flat, BASELINE_STATE_PATTERNS)[:50],
            "optimizer_future_state": match_paths(flat, OPT_STATE_PATTERNS)[:50],
            "manifest": match_paths(flat, MANIFEST_PATTERNS)[:50],
            "batch_hash": match_paths(flat, BATCH_HASH_PATTERNS)[:50],
            "rng": match_paths(flat, RNG_PATTERNS)[:50],
            "paired": match_paths(flat, PAIRED_PATTERNS)[:50],
            "x_theta": match_paths(flat, XTHETA_PATTERNS)[:50],
        },
    }
    return audit, details


def audit_code_capabilities(project_root: Path) -> dict[str, Any]:
    expected = [
        "research/src/fr/full_state.py",
        "research/src/fr/rng.py",
        "research/src/fr/theta_protocol.py",
        "research/src/fr/native_schedule.py",
        "research/src/fr/stream_rng.py",
        "research/src/fr/batch_manifest.py",
        "research/src/fr/pilot_partition.py",
        "research/src/fr/stage_c_runtime.py",
        "research/src/fr/direction.py",
        "research/scripts/e3c_multiseed_theta_v07.py",
        "research/scripts/e3c_prepare_state_bank_v07.py",
        "research/configs/e3c_confirmatory_v07.json",
    ]
    files = {rel: (project_root / rel).exists() for rel in expected}
    core_replay = all(
        files[k]
        for k in [
            "research/src/fr/full_state.py",
            "research/src/fr/theta_protocol.py",
            "research/src/fr/native_schedule.py",
            "research/src/fr/batch_manifest.py",
            "research/src/fr/stage_c_runtime.py",
        ]
    )
    return {"files": files, "core_replay_modules_present": core_replay}


def build_audit(project_root: Path, seeds: list[int], states: list[int], horizons: list[int]) -> dict[str, Any]:
    seed_rows: list[SeedAudit] = []
    details: dict[str, Any] = {}
    for seed in seeds:
        row, det = audit_seed(project_root, seed, states)
        seed_rows.append(row)
        details[str(seed)] = det

    code = audit_code_capabilities(project_root)

    all_theta = all(r.theta_raw_exists for r in seed_rows)
    all_state_report = all(r.state_report_exists for r in seed_rows)
    all_states = all(r.state_files_found == len(states) for r in seed_rows)

    # [FR-OBSERVATION-ONLY] 只有所有 seed 都已保存 future q/gradient，才判定无需 replay。
    future_q_available_all = all(
        (r.future_q_tensor_hits > 0 or r.future_probe_gradient_hits > 0) for r in seed_rows
    )

    current_q_evidence_all = all((r.current_q_tensor_hits > 0 or r.current_q_hash_hits > 0) for r in seed_rows)
    manifest_evidence_all = all(r.manifest_hits > 0 for r in seed_rows)
    rng_evidence_all = all(r.rng_hits > 0 for r in seed_rows)

    existing_data_sufficient = bool(all_theta and future_q_available_all)
    baseline_replay_required = not existing_data_sufficient
    exact_replay_possible = bool(
        all_theta
        and all_state_report
        and all_states
        and code["core_replay_modules_present"]
        and manifest_evidence_all
        and rng_evidence_all
    )

    if existing_data_sufficient:
        gpu_work = "无需 GPU replay；可直接由既有 future q/probe-gradient 计算 R_q。"
    elif exact_replay_possible:
        gpu_work = (
            f"需要 baseline-only replay：{len(seeds)} seeds × {len(states)} states × 8 streams；"
            f"每个 stream 仅做一次 max(H)={max(horizons)} step rollout，并在 {horizons} 中途 observation-only probe。"
        )
    else:
        gpu_work = "需要 baseline-only replay，但当前审计未能证明 exact replay 所需 state/manifest/RNG/代码证据全部齐全；先补人工检查。"

    return {
        "schema": "E4B0_DYNAMIC_STATE_DATA_AUDIT_V08",
        "project_root": str(project_root),
        "seeds": seeds,
        "states": states,
        "horizons": horizons,
        "seed_audits": [asdict(x) for x in seed_rows],
        "code_capabilities": code,
        "aggregate": {
            "all_theta_mapping_raw_present": all_theta,
            "all_state_bank_reports_present": all_state_report,
            "all_expected_state_pt_present": all_states,
            "current_q_evidence_all_seeds": current_q_evidence_all,
            "future_q_or_probe_gradient_all_seeds": future_q_available_all,
            "manifest_evidence_all_seeds": manifest_evidence_all,
            "rng_evidence_all_seeds": rng_evidence_all,
        },
        "decision": {
            "E4B_EXISTING_DATA_SUFFICIENT_FOR_Q_ROTATION": "YES" if existing_data_sufficient else "NO",
            "E4B_BASELINE_REPLAY_REQUIRED": "YES" if baseline_replay_required else "NO",
            "E4B_EXACT_REPLAY_POSSIBLE": "YES" if exact_replay_possible else "NO",
            "E4B_EXPECTED_GPU_WORK": gpu_work,
            "E4B0_READY": "YES" if (existing_data_sufficient or exact_replay_possible) else "NO",
        },
        "details": details,
        "boundaries": [
            "[FR-OBSERVATION-ONLY] 本审计不运行模型、不修改 optimizer、不生成新的 B/chi。",
            "[FR-REPLAY-CRITICAL] exact replay 判定只基于现有文件/字段证据；若为 NO，不应盲目开始 GPU replay。",
            "[FR-RNG-CRITICAL] E4B-1 必须复用原 future stream 的 batch order 与 RNG 语义。",
            "[FR-HASH] E4B-1 应记录 q_t/q_future hash，但 summary 不保存高维 tensor。",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    d = report["decision"]
    a = report["aggregate"]
    lines = [
        "# E4B-0 v0.8 Dynamic State Data Audit",
        "",
        "本报告为只读数据可用性审计，不改变 v0.7 / E4A 科学结论。",
        "",
        "## Decision",
        "",
        f"- E4B_EXISTING_DATA_SUFFICIENT_FOR_Q_ROTATION: `{d['E4B_EXISTING_DATA_SUFFICIENT_FOR_Q_ROTATION']}`",
        f"- E4B_BASELINE_REPLAY_REQUIRED: `{d['E4B_BASELINE_REPLAY_REQUIRED']}`",
        f"- E4B_EXACT_REPLAY_POSSIBLE: `{d['E4B_EXACT_REPLAY_POSSIBLE']}`",
        f"- E4B_EXPECTED_GPU_WORK: {d['E4B_EXPECTED_GPU_WORK']}",
        f"- E4B0_READY: `{d['E4B0_READY']}`",
        "",
        "## Aggregate Evidence",
        "",
    ]
    for k, v in a.items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Per-seed Summary", "", "| seed | states | current q | future q/probe | manifest | RNG | paired |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["seed_audits"]:
        lines.append(
            f"| {r['seed']} | {r['state_files_found']} | {r['current_q_tensor_hits'] + r['current_q_hash_hits']} | "
            f"{r['future_q_tensor_hits'] + r['future_probe_gradient_hits']} | {r['manifest_hits']} | {r['rng_hits']} | {r['paired_branch_hits']} |"
        )
    lines += ["", "## Boundaries", ""]
    for x in report["boundaries"]:
        lines.append(f"- {x}")
    lines.append("")
    return "\n".join(lines)
