"""FR A0C Momentum Theory v02 read-only analysis.

[FR-OBSERVATION-ONLY] This module only reads existing A0C, E1 and E3C
artifacts. It never trains, perturbs, resumes, or rewrites experiment state.

[FR-NATIVE-ADAMW] ``exp_avg_sq``, ``lr``, ``betas`` and ``eps`` are read
directly from the saved native AdamW state. The requested v02 equation uses
raw ``exp_avg_sq`` without adding an unrequested bias correction or fit scale.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


CSV_RELATIVE_PATH = Path(
    "outputs/a0_fr_audit/v01/direct_path/a0c_state_horizon.csv"
)
STATE_ROOT_RELATIVE_PATH = Path("outputs/e3_confirmatory/v07")
Q_RELATIVE_PATH = Path("outputs/e1_fluctuation/v03_pilot/probe_direction_q.pt")
E1_REPORT_RELATIVE_PATH = Path(
    "outputs/e1_fluctuation/v03_pilot/e1_fluctuation_report.json"
)
A0_AUDIT_RELATIVE_PATH = Path("outputs/a0_fr_audit/v01/A0_DATA_AUDIT.json")

REQUIRED_COLUMNS = {"seed", "state_step", "horizon", "chi"}
_SEED_DIR_RE = re.compile(r"^seed_(-?\d+)$")


def _torch():
    import torch

    return torch


def _load_torch(path: Path):
    # [FR-OBSERVATION-ONLY] CPU load prevents any optimizer/device mutation.
    return _torch().load(path, map_location="cpu", weights_only=False)


def _finite_float(value, *, field: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return out


def _optimizer_param_order(optimizer_state: Mapping) -> List[object]:
    order: List[object] = []
    for group in optimizer_state.get("param_groups", []):
        order.extend(list(group.get("params", [])))
    return order


def _optimizer_groups_by_param(optimizer_state: Mapping) -> Dict[object, Mapping]:
    result: Dict[object, Mapping] = {}
    for group in optimizer_state.get("param_groups", []):
        for param_id in group.get("params", []):
            if param_id in result:
                raise ValueError(f"optimizer param id appears in multiple groups: {param_id!r}")
            result[param_id] = group
    return result


def _lookup_state_slot(state: Mapping, param_id: object):
    if param_id in state:
        return state[param_id]
    text_id = str(param_id)
    if text_id in state:
        return state[text_id]
    raise KeyError(f"optimizer state missing param id {param_id!r}")


def compute_mu_q_raw_v(q_by_name: Mapping[str, object], checkpoint: Mapping) -> Dict:
    """Compute ``sum(q^2 / (sqrt(exp_avg_sq) + eps))`` by parameter name.

    Saved optimizer state IDs do not contain names. The project creates q from
    ``named_parameters()`` and AdamW from the same trainable-parameter order.
    We therefore recover that order from checkpoint model names, but require
    exact q-name coverage and a per-name tensor-shape match before using it.
    """
    torch = _torch()
    if not isinstance(q_by_name, Mapping) or not q_by_name:
        raise ValueError("q payload must contain a non-empty name-to-tensor mapping")
    model_state = checkpoint.get("model")
    optimizer = checkpoint.get("optimizer")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint missing model state mapping")
    if not isinstance(optimizer, Mapping):
        raise ValueError("checkpoint missing optimizer state mapping")

    q_names = list(q_by_name.keys())
    q_name_set = set(q_names)
    parameter_names = [name for name in model_state.keys() if name in q_name_set]
    missing_model_names = sorted(q_name_set.difference(parameter_names))
    if missing_model_names:
        raise ValueError(f"q parameter names absent from checkpoint model: {missing_model_names}")

    param_order = _optimizer_param_order(optimizer)
    if len(parameter_names) != len(q_names):
        raise ValueError(
            f"checkpoint matched {len(parameter_names)} q names but q has {len(q_names)} tensors"
        )
    if len(param_order) != len(parameter_names):
        raise ValueError(
            f"optimizer parameter count {len(param_order)} != named q count {len(parameter_names)}"
        )

    state = optimizer.get("state", {})
    group_by_param = _optimizer_groups_by_param(optimizer)
    mu_q = torch.zeros((), dtype=torch.float64)
    eta_values: List[float] = []
    beta1_values: List[float] = []
    eps_values: List[float] = []

    for name, param_id in zip(parameter_names, param_order):
        if param_id not in group_by_param:
            raise KeyError(f"optimizer param group missing param id {param_id!r}")
        group = group_by_param[param_id]
        slot = _lookup_state_slot(state, param_id)
        if "exp_avg_sq" not in slot:
            raise KeyError(f"optimizer state[{param_id!r}] missing exp_avg_sq for {name}")

        q = q_by_name[name]
        v = slot["exp_avg_sq"]
        if not torch.is_tensor(q) or not torch.is_tensor(v):
            raise TypeError(f"q and exp_avg_sq must be tensors for {name}")
        q_cpu = q.detach().to(dtype=torch.float64, device="cpu")
        v_cpu = v.detach().to(dtype=torch.float64, device="cpu")
        if tuple(q_cpu.shape) != tuple(v_cpu.shape):
            raise ValueError(
                f"q/exp_avg_sq shape mismatch at {name}: "
                f"{tuple(q_cpu.shape)} vs {tuple(v_cpu.shape)}"
            )
        if bool(torch.any(v_cpu < 0.0)):
            raise ValueError(f"exp_avg_sq contains negative values at {name}")

        eta = _finite_float(group.get("lr"), field=f"lr[{name}]")
        betas = group.get("betas")
        if not isinstance(betas, (tuple, list)) or len(betas) < 1:
            raise ValueError(f"optimizer group missing betas for {name}")
        beta1 = _finite_float(betas[0], field=f"beta1[{name}]")
        eps = _finite_float(group.get("eps"), field=f"eps[{name}]")
        if eta < 0.0 or not 0.0 <= beta1 < 1.0 or eps < 0.0:
            raise ValueError(
                f"invalid AdamW hyperparameters at {name}: lr={eta}, beta1={beta1}, eps={eps}"
            )

        mu_q += torch.sum((q_cpu * q_cpu) / (torch.sqrt(v_cpu) + eps))
        eta_values.append(eta)
        beta1_values.append(beta1)
        eps_values.append(eps)

    def require_single(values: Sequence[float], label: str) -> float:
        unique = sorted(set(values))
        if len(unique) != 1:
            raise ValueError(
                f"requested scalar formula does not define heterogeneous {label}: {unique}"
            )
        return float(unique[0])

    result = {
        "mu_q": float(mu_q.item()),
        "eta": require_single(eta_values, "lr"),
        "beta1": require_single(beta1_values, "beta1"),
        "eps": require_single(eps_values, "eps"),
        "n_parameter_tensors": len(parameter_names),
        "parameter_names": parameter_names,
        "alignment": "MODEL_STATE_PARAMETER_NAMES_TO_SAVED_OPTIMIZER_PARAM_ORDER_WITH_SHAPE_CHECK",
        "v_semantics": "RAW_EXP_AVG_SQ_NO_BIAS_CORRECTION",
    }
    if not math.isfinite(result["mu_q"]):
        raise ValueError("mu_q is not finite")
    return result


def safe_r2(observed: Sequence[float], predicted: Sequence[float]):
    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or y.size < 2:
        return None
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        return None
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0.0:
        return None
    return float(1.0 - np.sum((y - p) ** 2) / sst)


def safe_corr(observed: Sequence[float], predicted: Sequence[float]):
    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or y.size < 2:
        return None
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        return None
    if float(np.sum((y - y.mean()) ** 2)) <= 0.0:
        return None
    if float(np.sum((p - p.mean()) ** 2)) <= 0.0:
        return None
    return float(np.corrcoef(y, p)[0, 1])


def _read_csv_rows(path: Path) -> Tuple[List[Dict], List[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS.difference(fieldnames))
        if missing:
            raise ValueError(f"A0C CSV missing required columns: {missing}")
        parsed: List[Dict] = []
        seen = set()
        for line_number, raw in enumerate(reader, start=2):
            seed = int(raw["seed"])
            state_step = int(raw["state_step"])
            horizon = int(raw["horizon"])
            if horizon <= 0:
                raise ValueError(f"horizon must be positive at CSV line {line_number}")
            key = (seed, state_step, horizon)
            if key in seen:
                raise ValueError(f"duplicate seed/state/horizon at CSV line {line_number}: {key}")
            seen.add(key)
            row = dict(raw)
            row.update(
                seed=seed,
                state_step=state_step,
                horizon=horizon,
                chi=_finite_float(raw["chi"], field=f"chi line {line_number}"),
            )
            if raw.get("state_fraction", "") != "":
                row["state_fraction"] = _finite_float(
                    raw["state_fraction"], field=f"state_fraction line {line_number}"
                )
            parsed.append(row)
    if not parsed:
        raise ValueError("A0C CSV contains no data rows")
    return parsed, fieldnames


def _read_json(path: Path) -> Mapping:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return obj


def _resolve_sigma_q(q_payload: Mapping, e1_report_path: Path) -> Tuple[float | None, Dict]:
    q_metadata = q_payload.get("metadata", {})
    if isinstance(q_metadata, Mapping) and q_metadata.get("sigma_q") is not None:
        value = _finite_float(q_metadata["sigma_q"], field="q.metadata.sigma_q")
        if value < 0.0:
            raise ValueError("sigma_q must be nonnegative")
        return value, {"kind": "q_payload_metadata", "path": None, "field": "metadata.sigma_q"}
    if e1_report_path.exists():
        report = _read_json(e1_report_path)
        stats = report.get("stats", {})
        if isinstance(stats, Mapping) and stats.get("sigma_q") is not None:
            value = _finite_float(stats["sigma_q"], field="e1_report.stats.sigma_q")
            if value < 0.0:
                raise ValueError("sigma_q must be nonnegative")
            return value, {
                "kind": "e1_report_stats",
                "path": str(e1_report_path.resolve()),
                "field": "stats.sigma_q",
            }
    return None, {
        "kind": "missing",
        "path": str(e1_report_path.resolve()),
        "searched_fields": ["q_payload.metadata.sigma_q", "e1_report.stats.sigma_q"],
    }


def _resolve_rho(csv_rows: Sequence[Mapping], audit_path: Path) -> Tuple[float | None, Dict]:
    csv_values = {
        _finite_float(row["rho"], field="CSV rho")
        for row in csv_rows
        if row.get("rho", "") != ""
    }
    if csv_values:
        if len(csv_values) != 1:
            raise ValueError(f"CSV contains multiple rho values: {sorted(csv_values)}")
        value = next(iter(csv_values))
        return value, {"kind": "a0c_csv", "path": None, "field": "rho"}
    if audit_path.exists():
        audit = _read_json(audit_path)
        values = audit.get("rho_values")
        if isinstance(values, list) and values:
            unique = sorted({_finite_float(x, field="A0 rho_values") for x in values})
            if len(unique) != 1:
                raise ValueError(f"A0 metadata contains multiple rho values: {unique}")
            return unique[0], {
                "kind": "a0_data_audit",
                "path": str(audit_path.resolve()),
                "field": "rho_values[0]",
            }
    return None, {
        "kind": "missing",
        "path": str(audit_path.resolve()),
        "searched_fields": ["CSV.rho", "A0_DATA_AUDIT.rho_values"],
    }


def _available_checkpoint_seeds(state_root: Path) -> List[int]:
    seeds = []
    if not state_root.exists():
        return seeds
    for path in state_root.iterdir():
        if not path.is_dir():
            continue
        match = _SEED_DIR_RE.fullmatch(path.name)
        if match:
            seeds.append(int(match.group(1)))
    return sorted(seeds)


def _metric_by(rows: Sequence[Mapping], key: str, values: Iterable[int]) -> Dict[str, float | None]:
    result: Dict[str, float | None] = {}
    for value in sorted(set(int(x) for x in values)):
        selected = [row for row in rows if int(row[key]) == value]
        result[str(value)] = safe_r2(
            [row["chi"] for row in selected],
            [row["chi_hat"] for row in selected],
        )
    return result


def _corr_by(rows: Sequence[Mapping], key: str, values: Iterable[int]) -> Dict[str, float | None]:
    result: Dict[str, float | None] = {}
    for value in sorted(set(int(x) for x in values)):
        selected = [row for row in rows if int(row[key]) == value]
        result[str(value)] = safe_corr(
            [row["chi"] for row in selected],
            [row["chi_hat"] for row in selected],
        )
    return result


def build_summary(project_root: str | Path) -> Dict:
    root = Path(project_root).resolve()
    csv_path = root / CSV_RELATIVE_PATH
    state_root = root / STATE_ROOT_RELATIVE_PATH
    q_path = root / Q_RELATIVE_PATH
    e1_report_path = root / E1_REPORT_RELATIVE_PATH
    a0_audit_path = root / A0_AUDIT_RELATIVE_PATH

    csv_rows, csv_columns = _read_csv_rows(csv_path)
    q_payload = _load_torch(q_path)
    if not isinstance(q_payload, Mapping) or not isinstance(q_payload.get("q"), Mapping):
        raise ValueError("probe_direction_q.pt missing q name-to-tensor mapping")
    q_by_name = q_payload["q"]
    sigma_q, sigma_source = _resolve_sigma_q(q_payload, e1_report_path)
    rho, rho_source = _resolve_rho(csv_rows, a0_audit_path)

    requested_states = sorted({(row["seed"], row["state_step"]) for row in csv_rows})
    requested_seeds = sorted({seed for seed, _ in requested_states})
    requested_horizons = sorted({row["horizon"] for row in csv_rows})
    available_seeds = _available_checkpoint_seeds(state_root)
    rows_by_state: Dict[Tuple[int, int], List[Mapping]] = defaultdict(list)
    for row in csv_rows:
        rows_by_state[(row["seed"], row["state_step"])].append(row)

    missing_entries = []
    state_results = []
    prediction_rows = []
    matched_state_count = 0
    for seed, state_step in requested_states:
        expected_path = state_root / f"seed_{seed}" / "states" / f"state_step_{state_step:03d}.pt"
        if not expected_path.is_file():
            missing_entries.append(
                {
                    "seed": seed,
                    "state_step": state_step,
                    "expected_path": str(expected_path),
                    "reason": "EXACT_SEED_STATE_CHECKPOINT_NOT_FOUND",
                }
            )
            continue

        try:
            checkpoint = _load_torch(expected_path)
            metadata = checkpoint.get("metadata", {})
            metadata_seed = metadata.get("train_seed") if isinstance(metadata, Mapping) else None
            metadata_step = metadata.get("state_optimizer_step") if isinstance(metadata, Mapping) else None
            if metadata_seed is not None and int(metadata_seed) != seed:
                raise ValueError(f"checkpoint metadata train_seed={metadata_seed}, expected {seed}")
            if metadata_step is not None and int(metadata_step) != state_step:
                raise ValueError(
                    f"checkpoint metadata state_optimizer_step={metadata_step}, expected {state_step}"
                )
            mu = compute_mu_q_raw_v(q_by_name, checkpoint)
        except Exception as exc:
            missing_entries.append(
                {
                    "seed": seed,
                    "state_step": state_step,
                    "expected_path": str(expected_path),
                    "reason": "CHECKPOINT_OR_ALIGNMENT_INVALID",
                    "detail": str(exc),
                }
            )
            continue

        matched_state_count += 1
        state_result = {
            "seed": seed,
            "state_step": state_step,
            "checkpoint": str(expected_path),
            "mu_q": mu["mu_q"],
            "eta": mu["eta"],
            "beta1": mu["beta1"],
            "eps": mu["eps"],
            "n_parameter_tensors": mu["n_parameter_tensors"],
            "alignment": mu["alignment"],
            "v_semantics": mu["v_semantics"],
        }
        state_results.append(state_result)
        if sigma_q is None or rho is None:
            continue
        for row in sorted(rows_by_state[(seed, state_step)], key=lambda x: x["horizon"]):
            horizon = int(row["horizon"])
            chi_hat = (
                -mu["eta"]
                * rho
                * sigma_q
                * mu["mu_q"]
                * (1.0 - mu["beta1"] ** horizon)
            )
            prediction_rows.append(
                {
                    "seed": seed,
                    "state_step": state_step,
                    "horizon": horizon,
                    "chi": float(row["chi"]),
                    "chi_hat": float(chi_hat),
                }
            )

    if matched_state_count == 0:
        status = "NO_MATCHED_STATES"
    elif sigma_q is None:
        status = "MISSING_SIGMA_Q"
    elif rho is None:
        status = "MISSING_RHO"
    elif missing_entries:
        status = "PARTIAL_MISSING_STATES"
    else:
        status = "COMPLETE"

    observed = [row["chi"] for row in prediction_rows]
    predicted = [row["chi_hat"] for row in prediction_rows]
    summary = {
        "protocol": "FR_A0C_MOMENTUM_THEORY_V02",
        "status": status,
        "formula": "chi_hat=-eta*rho*sigma_q*mu_q*(1-beta1^H)",
        "mu_q_formula": "sum_i(q_i^2/(sqrt(exp_avg_sq_i)+eps))",
        "target_column": "chi",
        "fit_parameters": [],
        "source_paths": {
            "a0c_state_horizon_csv": str(csv_path),
            "checkpoint_root": str(state_root),
            "probe_direction_q": str(q_path),
        },
        "checkpoint_source": str(state_root),
        "input_row_count": len(csv_rows),
        "input_columns": csv_columns,
        "requested_state_count": len(requested_states),
        "matched_state_count": matched_state_count,
        "prediction_row_count": len(prediction_rows),
        "rho": rho,
        "rho_source": rho_source,
        "sigma_q": sigma_q,
        "sigma_q_source": sigma_source,
        "overall_R2": safe_r2(observed, predicted),
        "overall_corr": safe_corr(observed, predicted),
        "seed_R2": _metric_by(prediction_rows, "seed", requested_seeds),
        "horizon_R2": _metric_by(prediction_rows, "horizon", requested_horizons),
        "seed_corr": _corr_by(prediction_rows, "seed", requested_seeds),
        "horizon_corr": _corr_by(prediction_rows, "horizon", requested_horizons),
        "missing_state_report": {
            "requested_seeds": requested_seeds,
            "available_seeds": available_seeds,
            "requested_state_count": len(requested_states),
            "matched_state_count": matched_state_count,
            "missing_state_count": len(missing_entries),
            "entries": missing_entries,
        },
        "matched_states": [
            {
                "seed": row["seed"],
                "state_step": row["state_step"],
                "checkpoint": row["checkpoint"],
            }
            for row in state_results
        ],
        "missing_states": missing_entries,
        "state_results": state_results,
        "prediction_rows": prediction_rows,
        "constraints": [
            "EXACT_SEED_STATE_MATCH_ONLY",
            "NO_CROSS_SEED_SUBSTITUTION",
            "RAW_EXP_AVG_SQ_NO_BIAS_CORRECTION",
            "NO_EMPIRICAL_SCALE_PARAMETER",
            "FR_OBSERVATION_ONLY",
        ],
    }
    return summary
