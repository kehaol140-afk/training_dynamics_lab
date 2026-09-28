"""Read-only AdamW first- and second-moment response theory helpers.

[FR-OBSERVATION-ONLY] The module consumes saved tensors and replayed
gradients. It does not train, step an optimizer, perturb a model, or rewrite an
experiment artifact.

[FR-NATIVE-ADAMW] ``m`` and ``v`` mean the raw native AdamW ``exp_avg`` and
``exp_avg_sq`` slots. ``adam_eps`` is kept distinct from the finite
perturbation scale ``rho * sigma_q``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .batch_manifest import manifest_sha256


EXPECTED_SEEDS = tuple(range(2030, 2038))
EXPECTED_HORIZONS = (1, 2, 4, 8, 12, 16, 24, 32)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch():
    import torch

    return torch


def _finite(value, label: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return out


def _optimizer_param_order(optimizer_state: Mapping) -> list:
    result = []
    for group in optimizer_state.get("param_groups", []):
        result.extend(group.get("params", []))
    return result


def _groups_by_param(optimizer_state: Mapping) -> Dict[object, Mapping]:
    result = {}
    for group in optimizer_state.get("param_groups", []):
        for param_id in group.get("params", []):
            if param_id in result:
                raise ValueError(f"optimizer parameter appears in multiple groups: {param_id!r}")
            result[param_id] = group
    return result


def _slot(state: Mapping, param_id: object) -> Mapping:
    if param_id in state:
        return state[param_id]
    if str(param_id) in state:
        return state[str(param_id)]
    raise KeyError(f"optimizer state missing parameter id {param_id!r}")


def _single(values: Sequence[float], label: str) -> float:
    unique = sorted(set(values))
    if len(unique) != 1:
        raise ValueError(f"scalar theory does not define heterogeneous {label}: {unique}")
    return float(unique[0])


def compute_mu_nu(
    q_by_name: Mapping[str, object],
    gradient_by_name: Mapping[str, object],
    checkpoint: Mapping,
) -> Dict:
    """Compute the requested raw-state ``mu_q`` and ``nu_q`` in float64."""
    torch = _torch()
    if not isinstance(q_by_name, Mapping) or not q_by_name:
        raise ValueError("q must be a non-empty parameter-name mapping")
    if not isinstance(gradient_by_name, Mapping):
        raise ValueError("gradient must be a parameter-name mapping")
    if set(q_by_name) != set(gradient_by_name):
        missing = sorted(set(q_by_name).difference(gradient_by_name))
        extra = sorted(set(gradient_by_name).difference(q_by_name))
        raise ValueError(f"q/gradient name mismatch: missing={missing}, extra={extra}")

    model_state = checkpoint.get("model")
    optimizer = checkpoint.get("optimizer")
    if not isinstance(model_state, Mapping) or not isinstance(optimizer, Mapping):
        raise ValueError("checkpoint must contain model and optimizer mappings")

    q_names = set(q_by_name)
    parameter_names = [name for name in model_state if name in q_names]
    absent = sorted(q_names.difference(parameter_names))
    if absent:
        raise ValueError(f"q names absent from checkpoint model: {absent}")

    param_order = _optimizer_param_order(optimizer)
    if len(parameter_names) != len(q_names) or len(param_order) != len(parameter_names):
        raise ValueError(
            "optimizer/q/model parameter counts disagree: "
            f"optimizer={len(param_order)}, q={len(q_names)}, model={len(parameter_names)}"
        )

    groups = _groups_by_param(optimizer)
    state = optimizer.get("state", {})
    mu = torch.zeros((), dtype=torch.float64)
    nu = torch.zeros((), dtype=torch.float64)
    eta_values = []
    beta1_values = []
    beta2_values = []
    adam_eps_values = []
    zero_q_count = 0

    for name, param_id in zip(parameter_names, param_order):
        group = groups[param_id]
        slot = _slot(state, param_id)
        if "exp_avg" not in slot or "exp_avg_sq" not in slot:
            raise KeyError(f"optimizer state missing exp_avg/exp_avg_sq for {name}")

        tensors = [q_by_name[name], gradient_by_name[name], slot["exp_avg"], slot["exp_avg_sq"]]
        if not all(torch.is_tensor(tensor) for tensor in tensors):
            raise TypeError(f"q/g/m/v must all be tensors for {name}")
        q, gradient, m, v = [
            tensor.detach().to(device="cpu", dtype=torch.float64) for tensor in tensors
        ]
        shapes = {tuple(tensor.shape) for tensor in (q, gradient, m, v)}
        if len(shapes) != 1:
            raise ValueError(f"q/g/m/v shape mismatch for {name}: {sorted(shapes)}")
        if bool(torch.any(v < 0.0)):
            raise ValueError(f"exp_avg_sq contains negative values for {name}")

        eta = _finite(group.get("lr"), f"lr[{name}]")
        betas = group.get("betas")
        if not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise ValueError(f"optimizer betas missing for {name}")
        beta1 = _finite(betas[0], f"beta1[{name}]")
        beta2 = _finite(betas[1], f"beta2[{name}]")
        adam_eps = _finite(group.get("eps"), f"adam_eps[{name}]")
        if eta < 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or adam_eps < 0:
            raise ValueError(
                f"invalid AdamW hyperparameters for {name}: "
                f"lr={eta}, betas=({beta1}, {beta2}), eps={adam_eps}"
            )

        active = q != 0.0
        zero_q_count += int((~active).sum().item())
        if bool(active.any()):
            qa = q[active]
            ga = gradient[active]
            ma = m[active]
            va = v[active]
            sqrt_v = torch.sqrt(va)
            if bool(torch.any(sqrt_v <= 0.0)):
                raise ValueError(f"nu_q denominator is zero where q is nonzero for {name}")
            denom = sqrt_v + adam_eps
            mu += torch.sum((qa * qa) / denom)
            nu += torch.sum((qa * qa * ma * ga) / (sqrt_v * denom * denom))

        eta_values.append(eta)
        beta1_values.append(beta1)
        beta2_values.append(beta2)
        adam_eps_values.append(adam_eps)

    result = {
        "mu_q": float(mu.item()),
        "nu_q": float(nu.item()),
        "eta_checkpoint": _single(eta_values, "lr"),
        "beta1": _single(beta1_values, "beta1"),
        "beta2": _single(beta2_values, "beta2"),
        "adam_eps": _single(adam_eps_values, "adam_eps"),
        "parameter_names": parameter_names,
        "n_parameter_tensors": len(parameter_names),
        "zero_q_coordinate_count": zero_q_count,
        "alignment": "MODEL_PARAMETER_NAMES_TO_OPTIMIZER_ORDER_WITH_EXACT_SHAPE_CHECK",
        "moment_semantics": "RAW_EXP_AVG_AND_RAW_EXP_AVG_SQ_NO_BIAS_CORRECTION",
    }
    if not math.isfinite(result["mu_q"]) or not math.isfinite(result["nu_q"]):
        raise ValueError("mu_q or nu_q is not finite")
    return result


def predict_chi(
    *, eta: float, beta1: float, beta2: float, horizon: int,
    mu_q: float, nu_q: float, perturbation_epsilon: float,
) -> Dict[str, float]:
    """Return the user's explicitly epsilon-scaled m and m+v predictions."""
    eta = _finite(eta, "eta")
    beta1 = _finite(beta1, "beta1")
    beta2 = _finite(beta2, "beta2")
    mu_q = _finite(mu_q, "mu_q")
    nu_q = _finite(nu_q, "nu_q")
    perturbation_epsilon = _finite(perturbation_epsilon, "perturbation_epsilon")
    horizon = int(horizon)
    if eta < 0 or perturbation_epsilon < 0 or horizon <= 0:
        raise ValueError("eta/epsilon must be nonnegative and horizon must be positive")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("beta1 and beta2 must lie in [0, 1)")
    chi_m = -eta * perturbation_epsilon * mu_q * (1.0 - beta1**horizon)
    chi_mv = chi_m + eta * perturbation_epsilon * nu_q * (1.0 - beta2**horizon)
    return {"chi_m": float(chi_m), "chi_mv": float(chi_mv)}


def regression_metrics(observed: Sequence[float], predicted: Sequence[float]) -> Dict:
    """Direct prediction metrics plus descriptive OLS observed~predicted."""
    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or y.size < 2:
        raise ValueError("observed/predicted must be equal-length vectors with at least two rows")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("observed/predicted contain NaN or Inf")

    residual = y - p
    sst = float(np.sum((y - y.mean()) ** 2))
    pred_ss = float(np.sum((p - p.mean()) ** 2))
    r2 = None if sst <= 0 else float(1.0 - np.sum(residual * residual) / sst)
    corr = None
    slope = None
    intercept = None
    if sst > 0 and pred_ss > 0:
        corr = float(np.corrcoef(y, p)[0, 1])
        slope = float(np.sum((p - p.mean()) * (y - y.mean())) / pred_ss)
        intercept = float(y.mean() - slope * p.mean())
    return {
        "n": int(y.size),
        "corr": corr,
        "R2": r2,
        "MAE": float(np.mean(np.abs(residual))),
        "slope": slope,
        "intercept": intercept,
        "regression_direction": "observed=intercept+slope*predicted",
    }


def validate_e2_manifest(
    payload: Mapping,
    *, expected_seed: int, expected_state_step: int, expected_stream_id: int,
) -> Dict:
    """Validate the identity and content digest of one registered E2 stream."""
    if not isinstance(payload, Mapping):
        raise ValueError("manifest root must be a mapping")
    if payload.get("manifest_version") != "0.2":
        raise ValueError(f"unexpected manifest_version: {payload.get('manifest_version')!r}")
    metadata = payload.get("metadata")
    batches = payload.get("batches")
    if not isinstance(metadata, Mapping) or not isinstance(batches, list):
        raise ValueError("manifest must contain metadata mapping and batches list")
    expected_metadata = {
        "protocol": "E3C_CONFIRMATORY_V07_E2_PAIRED_STREAM",
        "train_seed": int(expected_seed),
        "state_step": int(expected_state_step),
        "stream_id": int(expected_stream_id),
    }
    for field, expected in expected_metadata.items():
        actual = metadata.get(field)
        if actual != expected:
            raise ValueError(f"manifest {field}={actual!r}, expected {expected!r}")
    actual_sha = manifest_sha256(batches)
    if payload.get("sha256") != actual_sha:
        raise ValueError(
            f"manifest sha256={payload.get('sha256')!r}, computed {actual_sha!r}"
        )
    if len(batches) < 2 or any(not isinstance(batch, list) or not batch for batch in batches):
        raise ValueError("manifest batches are missing or malformed")
    stream_seed = int(metadata.get("stream_seed"))
    rho = _finite(metadata.get("rho"), "manifest rho")
    epsilon = _finite(metadata.get("epsilon"), "manifest epsilon")
    if rho <= 0 or epsilon <= 0:
        raise ValueError("manifest rho and epsilon must be positive")
    return {
        "batches": batches,
        "sha256": actual_sha,
        "stream_seed": stream_seed,
        "rho": rho,
        "epsilon": epsilon,
    }


def aggregate_stream_results(streams: Sequence[Mapping]) -> Dict:
    """Average stream-linear theory features and full pre-clip gradients."""
    torch = _torch()
    if not streams:
        raise ValueError("at least one stream result is required")
    first = streams[0]
    names = list(first["gradient"])
    if not names:
        raise ValueError("stream gradient mapping must be non-empty")
    mu0 = _finite(first["mu_q"], "mu_q")
    eta0 = _finite(first["eta"], "eta")
    gradient_sum = {
        name: first["gradient"][name].detach().to(device="cpu", dtype=torch.float32).clone()
        for name in names
    }
    nu_values = [_finite(first["nu_q"], "nu_q")]
    for stream in streams[1:]:
        if _finite(stream["mu_q"], "mu_q") != mu0:
            raise ValueError("mu_q differs across streams of one state")
        if _finite(stream["eta"], "eta") != eta0:
            raise ValueError("eta differs across streams of one state")
        gradient = stream["gradient"]
        if list(gradient) != names:
            raise ValueError("gradient parameter names/order differ across streams")
        for name in names:
            tensor = gradient[name].detach().to(device="cpu", dtype=torch.float32)
            if tensor.shape != gradient_sum[name].shape:
                raise ValueError(f"gradient shape differs across streams for {name}")
            gradient_sum[name].add_(tensor)
        nu_values.append(_finite(stream["nu_q"], "nu_q"))
    count = len(streams)
    mean_gradient = {name: tensor.div(float(count)) for name, tensor in gradient_sum.items()}
    return {
        "stream_count": count,
        "mu_q": mu0,
        "nu_q": float(np.mean(np.asarray(nu_values, dtype=np.float64))),
        "eta": eta0,
        "mean_gradient": mean_gradient,
    }


def build_metric_breakdowns(rows: Sequence[Mapping]) -> Dict:
    """Build direct prediction metrics overall and by seed/horizon."""
    if not rows:
        raise ValueError("prediction rows are empty")

    def metrics_for(items, prediction_key):
        return regression_metrics(
            [float(row["chi"]) for row in items],
            [float(row[prediction_key]) for row in items],
        )

    output = {}
    for prediction_key in ("chi_m", "chi_mv"):
        by_seed = defaultdict(list)
        by_horizon = defaultdict(list)
        for row in rows:
            by_seed[int(row["seed"])].append(row)
            by_horizon[int(row["horizon"])].append(row)
        output[prediction_key] = {
            "overall": metrics_for(rows, prediction_key),
            "per_seed": {
                str(key): metrics_for(by_seed[key], prediction_key) for key in sorted(by_seed)
            },
            "per_horizon": {
                str(key): metrics_for(by_horizon[key], prediction_key)
                for key in sorted(by_horizon)
            },
        }
    return output


def assemble_summary(
    prediction_rows: Sequence[Mapping],
    *, state_results: Sequence[Mapping], matched_stream_count: int,
    source_paths: Mapping, replay_metadata: Mapping,
) -> Dict:
    """Assemble the complete JSON-safe experiment result."""
    metrics = build_metric_breakdowns(prediction_rows)
    r2_m = metrics["chi_m"]["overall"]["R2"]
    r2_mv = metrics["chi_mv"]["overall"]["R2"]
    r2_improvement = None if r2_m is None or r2_mv is None else float(r2_mv - r2_m)
    remaining = None if r2_mv is None else float(1.0 - r2_mv)
    residuals = np.asarray(
        [float(row["chi"]) - float(row["chi_mv"]) for row in prediction_rows],
        dtype=np.float64,
    )
    sensitivity = None
    if state_results and all(
        {"seed", "state_step", "nu_q"}.issubset(state) for state in state_results
    ):
        max_state = max(state_results, key=lambda state: abs(float(state["nu_q"])))
        reduced_rows = [
            row
            for row in prediction_rows
            if not (
                int(row["seed"]) == int(max_state["seed"])
                and int(row["state_step"]) == int(max_state["state_step"])
            )
        ]
        if len(reduced_rows) >= 2:
            reduced_m = regression_metrics(
                [float(row["chi"]) for row in reduced_rows],
                [float(row["chi_m"]) for row in reduced_rows],
            )
            reduced_mv = regression_metrics(
                [float(row["chi"]) for row in reduced_rows],
                [float(row["chi_mv"]) for row in reduced_rows],
            )
            reduced_gain = (
                None
                if reduced_m["R2"] is None or reduced_mv["R2"] is None
                else float(reduced_mv["R2"] - reduced_m["R2"])
            )
            sensitivity = {
                "max_abs_nu_state": {
                    "seed": int(max_state["seed"]),
                    "state_step": int(max_state["state_step"]),
                    "nu_q": float(max_state["nu_q"]),
                },
                "excluding_max_abs_nu_state": {
                    "n": len(reduced_rows),
                    "chi_m": reduced_m,
                    "chi_mv": reduced_mv,
                    "R2_improvement": reduced_gain,
                },
                "role": "POSTHOC_ROBUSTNESS_DIAGNOSTIC_NOT_USED_TO_SELECT_PRIMARY_RESULT",
            }
    seed_gains = {
        seed: float(metrics["chi_mv"]["per_seed"][seed]["R2"] - value["R2"])
        for seed, value in metrics["chi_m"]["per_seed"].items()
        if value["R2"] is not None and metrics["chi_mv"]["per_seed"][seed]["R2"] is not None
    }
    horizon_gains = {
        horizon: float(metrics["chi_mv"]["per_horizon"][horizon]["R2"] - value["R2"])
        for horizon, value in metrics["chi_m"]["per_horizon"].items()
        if value["R2"] is not None
        and metrics["chi_mv"]["per_horizon"][horizon]["R2"] is not None
    }
    return {
        "protocol": "FR_A0C_ADAMW_MV_THEORY_FINAL_V07",
        "status": "COMPLETE",
        "formula": {
            "mu_q": "sum_i(q_i^2/(sqrt(exp_avg_sq_i)+adam_eps))",
            "nu_q": (
                "sum_i(q_i^2*exp_avg_i*g_t_i/"
                "(sqrt(exp_avg_sq_i)*(sqrt(exp_avg_sq_i)+adam_eps)^2))"
            ),
            "chi_m": "-eta*(rho*sigma_q)*mu_q*(1-beta1^H)",
            "chi_mv": (
                "-eta*(rho*sigma_q)*mu_q*(1-beta1^H)"
                "+eta*(rho*sigma_q)*nu_q*(1-beta2^H)"
            ),
        },
        "target_definition": "chi=mean_stream[(X_plus-X_minus)/(2*epsilon)]",
        "target_is_unit_perturbation_response": True,
        "requested_prediction_includes_perturbation_epsilon": True,
        "dimensional_caveat": (
            "A0C chi is divided by perturbation epsilon, while the requested prediction "
            "multiplies by rho*sigma_q; prediction and target therefore use different "
            "response-scale conventions."
        ),
        "fit_parameters": [],
        "metrics": metrics,
        "comparison": {
            "R2_improvement": r2_improvement,
            "v_path_improves_R2": None if r2_improvement is None else bool(r2_improvement > 0),
            "remaining_unexplained_fraction": remaining,
            "remaining_residual_MAE": metrics["chi_mv"]["overall"]["MAE"],
            "remaining_residual_SSE": float(np.sum(residuals * residuals)),
            "seed_R2_improvement": seed_gains,
            "horizon_R2_improvement": horizon_gains,
            "seeds_improved": int(sum(value > 0 for value in seed_gains.values())),
            "horizons_improved": int(sum(value > 0 for value in horizon_gains.values())),
        },
        "sensitivity": sensitivity,
        "matched_state_count": len(state_results),
        "matched_stream_count": int(matched_stream_count),
        "prediction_row_count": len(prediction_rows),
        "source_paths": dict(source_paths),
        "replay": dict(replay_metadata),
        "state_results": list(state_results),
        "prediction_rows": list(prediction_rows),
        "constraints": [
            "FR_OBSERVATION_ONLY",
            "NO_OPTIMIZER_STEP",
            "PRECLIP_UNSCALED_GRADIENT",
            "EXACT_V07_SEED_STATE_E2_MANIFEST_MATCH",
            "RAW_EXP_AVG_AND_EXP_AVG_SQ",
            "NO_BIAS_CORRECTION",
            "NO_EMPIRICAL_OR_FITTED_SCALE",
        ],
    }


def _load_json(path: Path) -> Mapping:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_a0c(path: Path) -> list[Dict]:
    required = {"seed", "state_step", "horizon", "chi"}
    rows = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"A0C CSV missing columns: {missing}")
        for line_number, raw in enumerate(reader, start=2):
            row = dict(raw)
            row.update(
                seed=int(raw["seed"]),
                state_step=int(raw["state_step"]),
                horizon=int(raw["horizon"]),
                chi=_finite(raw["chi"], f"chi line {line_number}"),
            )
            key = (row["seed"], row["state_step"], row["horizon"])
            if key in seen:
                raise ValueError(f"duplicate A0C seed/state/horizon: {key}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("A0C CSV is empty")
    seeds = tuple(sorted({row["seed"] for row in rows}))
    horizons = tuple(sorted({row["horizon"] for row in rows}))
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"A0C seeds={seeds}, expected {EXPECTED_SEEDS}")
    if horizons != EXPECTED_HORIZONS:
        raise ValueError(f"A0C horizons={horizons}, expected {EXPECTED_HORIZONS}")
    return rows


def _checkpoint_metadata(checkpoint: Mapping, seed: int, state_step: int) -> Mapping:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata missing")
    if int(metadata.get("train_seed", -1)) != int(seed):
        raise ValueError(
            f"checkpoint train_seed={metadata.get('train_seed')}, expected {seed}"
        )
    if int(metadata.get("state_optimizer_step", -1)) != int(state_step):
        raise ValueError(
            "checkpoint state_optimizer_step="
            f"{metadata.get('state_optimizer_step')}, expected {state_step}"
        )
    return metadata


def run_experiment(project_root: str | Path, *, device: str = "cpu") -> tuple[Dict, Dict]:
    """Execute the full read-only v07 replay and return summary/gradient payload."""
    root = Path(project_root).resolve()
    if str(device).startswith("cuda"):
        raise ValueError("this observation audit is restricted to CPU; no GPU training")
    torch = _torch()

    config_path = root / "research/configs/e3c_confirmatory_v07.json"
    a0c_path = root / "outputs/a0_fr_audit/v01/direct_path/a0c_state_horizon.csv"
    state_root = root / "outputs/e3_confirmatory/v07"
    q_path = root / "outputs/e1_fluctuation/v03_pilot/probe_direction_q.pt"
    data_path = state_root / "e3c_subset.jsonl"
    partition_path = state_root / "partition_v07.json"
    for path in (config_path, a0c_path, q_path, data_path, partition_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = _load_json(config_path)
    if tuple(map(int, config["confirmation_train_seeds"])) != EXPECTED_SEEDS:
        raise ValueError("v07 configuration confirmation seeds differ from expected seeds")
    if tuple(map(int, config["horizons"])) != EXPECTED_HORIZONS:
        raise ValueError("v07 configuration horizons differ from expected horizons")
    if int(config["accumulation_steps"]) != 2:
        raise ValueError("expected v07 accumulation_steps=2")
    if int(config["e3_e2_streams"]) != 8:
        raise ValueError("expected eight v07 E2 streams")

    a0c_rows = _read_a0c(a0c_path)
    rows_by_state = defaultdict(list)
    for row in a0c_rows:
        rows_by_state[(row["seed"], row["state_step"])].append(row)
    requested_states = sorted(rows_by_state)
    if len(requested_states) != 56 or len(a0c_rows) != 448:
        raise ValueError(
            f"expected 56 states/448 A0C rows, got {len(requested_states)}/{len(a0c_rows)}"
        )

    q_payload = torch.load(q_path, map_location="cpu", weights_only=False)
    if not isinstance(q_payload, Mapping) or not isinstance(q_payload.get("q"), Mapping):
        raise ValueError("probe_direction_q.pt missing q mapping")
    q_by_name = q_payload["q"]

    minimind_root = root / "minimind"
    if str(minimind_root) not in sys.path:
        sys.path.insert(0, str(minimind_root))
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model

    from .hashing import stable_hash
    from .minimind_v03_runtime import make_amp, raw_model, stack_batch
    from .native_adamw_guard import assert_native_adamw
    from .theta_protocol import run_theta_branch

    model_config = MiniMindConfig(
        hidden_size=int(config["hidden_size"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        max_seq_len=int(config["max_seq_len"]),
        use_moe=False,
    )
    model, tokenizer = init_model(
        model_config,
        from_weight="none",
        tokenizer_path=str(minimind_root / "model"),
        device=str(device),
    )
    dataset = PretrainDataset(
        str(data_path), tokenizer, max_length=int(config["max_seq_len"])
    )
    autocast_ctx, scaler, _ = make_amp(torch, str(device), str(config["dtype"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    assert_native_adamw(optimizer)
    named = [(name, param) for name, param in raw_model(model).named_parameters() if param.requires_grad]
    named_names = [name for name, _ in named]
    if set(named_names) != set(q_by_name) or len(named_names) != len(q_by_name):
        raise ValueError(
            f"live model/q parameter names differ: model={len(named_names)}, q={len(q_by_name)}"
        )

    state_metadata_by_key = {}
    theta_sources = {}
    for seed in EXPECTED_SEEDS:
        theta_path = state_root / f"seed_{seed}/theta_mapping_raw.json"
        theta = _load_json(theta_path)
        if int(theta.get("train_seed", -1)) != seed or theta.get("technical_pass") is not True:
            raise ValueError(f"theta mapping invalid for seed {seed}")
        theta_sources[str(seed)] = str(theta_path)
        for record in theta.get("states", []):
            key = (seed, int(record["step"]))
            if key in state_metadata_by_key:
                raise ValueError(f"duplicate theta state record: {key}")
            state_metadata_by_key[key] = record
    if set(state_metadata_by_key) != set(requested_states):
        missing = sorted(set(requested_states).difference(state_metadata_by_key))
        extra = sorted(set(state_metadata_by_key).difference(requested_states))
        raise ValueError(f"theta/A0C state mismatch: missing={missing}, extra={extra}")

    state_results = []
    prediction_rows = []
    gradient_states = {}
    matched_stream_count = 0
    for state_index, (seed, state_step) in enumerate(requested_states, start=1):
        checkpoint_path = (
            state_root / f"seed_{seed}/states/state_step_{state_step:03d}.pt"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_meta = _checkpoint_metadata(checkpoint, seed, state_step)
        theta_state = state_metadata_by_key[(seed, state_step)]
        rho = _finite(theta_state.get("rho"), f"rho seed={seed} state={state_step}")
        sigma_q = _finite(
            theta_state.get("sigma_q"), f"sigma_q seed={seed} state={state_step}"
        )
        perturbation_epsilon = _finite(
            theta_state.get("epsilon"), f"epsilon seed={seed} state={state_step}"
        )
        if not math.isclose(perturbation_epsilon, rho * sigma_q, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"epsilon != rho*sigma_q for seed={seed}, state={state_step}")

        stream_records = sorted(theta_state.get("e2_streams", []), key=lambda x: int(x["stream_id"]))
        if [int(record["stream_id"]) for record in stream_records] != list(range(8)):
            raise ValueError(f"E2 stream IDs incomplete for seed={seed}, state={state_step}")
        replayed_streams = []
        stream_audit = []
        for stream_record in stream_records:
            stream_id = int(stream_record["stream_id"])
            expected_manifest_path = (
                state_root
                / f"seed_{seed}/e2_stream_manifests/state_step_{state_step:03d}"
                / f"e2_stream_{stream_id:03d}_batches.json"
            )
            recorded_path = Path(stream_record["batch_manifest_path"]).resolve()
            if recorded_path != expected_manifest_path.resolve():
                raise ValueError(
                    f"E2 manifest path mismatch for seed={seed}, state={state_step}, stream={stream_id}"
                )
            manifest_payload = _load_json(expected_manifest_path)
            manifest = validate_e2_manifest(
                manifest_payload,
                expected_seed=seed,
                expected_state_step=state_step,
                expected_stream_id=stream_id,
            )
            if manifest["sha256"] != stream_record["batch_manifest_sha256"]:
                raise ValueError(f"theta record manifest sha mismatch: {expected_manifest_path}")
            if manifest["stream_seed"] != int(stream_record["stream_seed"]):
                raise ValueError(f"theta record stream seed mismatch: {expected_manifest_path}")
            if not math.isclose(manifest["epsilon"], perturbation_epsilon, rel_tol=1e-12):
                raise ValueError(f"manifest epsilon mismatch: {expected_manifest_path}")

            replay = run_theta_branch(
                torch=torch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                autocast_ctx=autocast_ctx,
                ds=dataset,
                state_path=checkpoint_path,
                reference_theta=checkpoint["model"],
                q=q_by_name,
                named=named,
                batches=manifest["batches"],
                stream_seed=manifest["stream_seed"],
                sign=0,
                epsilon=0.0,
                horizons=[1],
                accumulation_steps=int(config["accumulation_steps"]),
                batch_size=int(config["batch_size"]),
                device=str(device),
                grad_clip=float(config["grad_clip"]),
                base_lr=float(config["learning_rate"]),
                global_micro_step=int(checkpoint_meta["global_micro_step"]),
                total_micro_steps=int(checkpoint_meta["total_micro_steps"]),
                get_lr=get_lr,
                stack_batch=stack_batch,
                measure_s_first=False,
                restore_checkpoint_rng=False,
                capture_first_unscaled_gradient=True,
                stop_after_first_unscaled_gradient=True,
            )
            if replay["batch_manifest_hash"] != stream_record["batch_manifest_hash"]:
                raise ValueError(f"replayed manifest hash mismatch: {expected_manifest_path}")
            if not replay.get("stopped_before_clip_and_step"):
                raise RuntimeError("gradient replay did not stop before clip/step")
            features = compute_mu_nu(q_by_name, replay["first_unscaled_gradient"], checkpoint)
            if not math.isclose(features["beta1"], 0.9, abs_tol=0.0):
                raise ValueError(f"beta1 != 0.9 in {checkpoint_path}")
            if not math.isclose(features["beta2"], 0.999, abs_tol=0.0):
                raise ValueError(f"beta2 != 0.999 in {checkpoint_path}")
            replayed_streams.append(
                {
                    "mu_q": features["mu_q"],
                    "nu_q": features["nu_q"],
                    "eta": float(replay["first_step_lr"]),
                    "gradient": replay["first_unscaled_gradient"],
                }
            )
            original_preclip = float(stream_record["baseline"]["first_preclip_grad_norm"])
            stream_audit.append(
                {
                    "stream_id": stream_id,
                    "stream_seed": manifest["stream_seed"],
                    "manifest_path": str(expected_manifest_path),
                    "manifest_sha256": manifest["sha256"],
                    "manifest_hash": replay["batch_manifest_hash"],
                    "first_step_lr": float(replay["first_step_lr"]),
                    "preclip_unscaled_grad_norm_cpu": float(replay["first_unscaled_grad_norm"]),
                    "recorded_preclip_grad_norm_cuda_bfloat16": original_preclip,
                    "recorded_would_clip_at_norm_1": bool(
                        stream_record["baseline"]["first_clipped"]
                    ),
                    "preclip_grad_norm_difference": float(
                        replay["first_unscaled_grad_norm"] - original_preclip
                    ),
                    "nu_q": features["nu_q"],
                }
            )
            matched_stream_count += 1

        aggregate = aggregate_stream_results(replayed_streams)
        beta1 = features["beta1"]
        beta2 = features["beta2"]
        adam_eps = features["adam_eps"]
        state_key = f"seed_{seed}/state_step_{state_step:03d}"
        gradient_states[state_key] = {
            "seed": seed,
            "state_step": state_step,
            "aggregation": "ARITHMETIC_MEAN_OF_8_REAL_E2_PRECLIP_UNSCALED_GRADIENTS",
            "gradient": aggregate["mean_gradient"],
            "stream_audit": stream_audit,
        }
        state_result = {
            "seed": seed,
            "state_step": state_step,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "q_path": str(q_path),
            "target_state_specific_q_path": str(theta_state.get("q_path")),
            "rho": rho,
            "sigma_q": sigma_q,
            "perturbation_epsilon": perturbation_epsilon,
            "mu_q": aggregate["mu_q"],
            "nu_q": aggregate["nu_q"],
            "eta_first_step": aggregate["eta"],
            "eta_checkpoint": features["eta_checkpoint"],
            "beta1": beta1,
            "beta2": beta2,
            "adam_eps": adam_eps,
            "n_parameter_tensors": features["n_parameter_tensors"],
            "stream_count": aggregate["stream_count"],
            "stream_audit": stream_audit,
        }
        state_results.append(state_result)
        for row in sorted(rows_by_state[(seed, state_step)], key=lambda item: item["horizon"]):
            prediction = predict_chi(
                eta=aggregate["eta"],
                beta1=beta1,
                beta2=beta2,
                horizon=int(row["horizon"]),
                mu_q=aggregate["mu_q"],
                nu_q=aggregate["nu_q"],
                perturbation_epsilon=perturbation_epsilon,
            )
            prediction_rows.append(
                {
                    "seed": seed,
                    "state_step": state_step,
                    "horizon": int(row["horizon"]),
                    "chi": float(row["chi"]),
                    "chi_m": prediction["chi_m"],
                    "chi_mv": prediction["chi_mv"],
                    "mu_q": aggregate["mu_q"],
                    "nu_q": aggregate["nu_q"],
                    "eta": aggregate["eta"],
                    "perturbation_epsilon": perturbation_epsilon,
                }
            )
        print(
            f"MV_REPLAY {state_index}/{len(requested_states)} seed={seed} "
            f"state={state_step} mu_q={aggregate['mu_q']:.8g} "
            f"nu_q={aggregate['nu_q']:.8g}",
            flush=True,
        )
        del replayed_streams, aggregate, checkpoint

    if matched_stream_count != 448 or len(prediction_rows) != 448:
        raise RuntimeError(
            f"incomplete replay: streams={matched_stream_count}, predictions={len(prediction_rows)}"
        )
    source_paths = {
        "project_root": str(root),
        "a0c_state_horizon_csv": str(a0c_path),
        "checkpoint_root": str(state_root),
        "probe_direction_q": str(q_path),
        "dataset": str(data_path),
        "partition": str(partition_path),
        "config": str(config_path),
        "theta_mapping_by_seed": theta_sources,
    }
    replay_metadata = {
        "device": str(device),
        "configured_dtype": str(config["dtype"]),
        "effective_cpu_autocast": "DISABLED_BY_EXISTING_make_amp_CPU_PATH",
        "gradient_capture": "AFTER_SCALER_UNSCALE_BEFORE_CLIP_BEFORE_OPTIMIZER_STEP",
        "accumulation_steps": int(config["accumulation_steps"]),
        "micro_batches_replayed_per_stream": int(config["accumulation_steps"]),
        "streams_per_state": 8,
        "optimizer_steps_called": 0,
        "gradient_aggregation": "ARITHMETIC_MEAN_ACROSS_MATCHED_E2_STREAMS",
        "eta_semantics": "NATIVE_LR_AT_SECOND_MICROBATCH_OF_FIRST_REPLAYED_STEP",
    }
    summary = assemble_summary(
        prediction_rows,
        state_results=state_results,
        matched_stream_count=matched_stream_count,
        source_paths=source_paths,
        replay_metadata=replay_metadata,
    )
    summary["input_hashes"] = {
        "a0c_state_horizon_csv": file_sha256(a0c_path),
        "probe_direction_q": file_sha256(q_path),
        "config": file_sha256(config_path),
        "partition": file_sha256(partition_path),
    }
    summary["direction_caveat"] = (
        "The requested global v03_pilot probe_direction_q.pt differs from the "
        "state-specific q paths used to measure v07 A0C chi."
    )
    gradient_payload = {
        "protocol": "FR_A0C_ADAMW_MV_THEORY_FINAL_V07_GRADIENT",
        "definition": "AFTER_SCALER_UNSCALE_BEFORE_CLIP_BEFORE_OPTIMIZER_STEP",
        "device": str(device),
        "dtype": "float32",
        "q_path": str(q_path),
        "state_count": len(gradient_states),
        "stream_count": matched_stream_count,
        "states": gradient_states,
    }
    return summary, gradient_payload


def _fmt(value) -> str:
    return "undefined" if value is None else f"{float(value):.12g}"


def render_report(summary: Mapping) -> str:
    """Render the required answer-first experiment report."""
    metrics = summary["metrics"]
    momentum = metrics["chi_m"]["overall"]
    mv = metrics["chi_mv"]["overall"]
    comparison = summary["comparison"]
    improved = float(comparison["R2_improvement"]) > 0
    sufficient = mv["R2"] is not None and float(mv["R2"]) >= 1.0
    sensitivity = summary.get("sensitivity")
    sensitivity_lines = []
    if isinstance(sensitivity, Mapping):
        extreme = sensitivity["max_abs_nu_state"]
        reduced = sensitivity["excluding_max_abs_nu_state"]
        sensitivity_lines = [
            "",
            "## Pre-clip sensitivity audit",
            "",
            f"最大 `|nu_q|` 位于 seed `{extreme['seed']}`、state `{extreme['state_step']}`，"
            f"`nu_q={_fmt(extreme['nu_q'])}`。排除此单一 state 仅作为事后稳健性检查时，"
            f"m+v 相对 momentum-only 的 R2 变化仍为 `{_fmt(reduced['R2_improvement'])}`；"
            "该排除不改变主结果，也未用于选择或校准模型。",
        ]
    lines = [
        "# A0C AdamW m+v Theory Response",
        "",
        "## 结论",
        "",
        f"- Does adding v-path improve? **{'Yes' if improved else 'No'}**. "
        f"R2 变化为 `{_fmt(comparison['R2_improvement'])}`。",
        f"- AdamW adaptive preconditioning sufficient? **{'Yes' if sufficient else 'No'}**. "
        f"m+v 的直接预测 R2 为 `{_fmt(mv['R2'])}`。",
        f"- 剩余未解释比例（`1-R2`）为 "
        f"`{_fmt(comparison['remaining_unexplained_fraction'])}`，残差 MAE 为 "
        f"`{_fmt(comparison['remaining_residual_MAE'])}`。",
        f"- 分层一致性：逐 seed 有 `{comparison.get('seeds_improved', 'n/a')}` 个 R2 改善；"
        f"逐 horizon 有 `{comparison.get('horizons_improved', 'n/a')}` 个 R2 改善。",
        "",
        "## Overall metrics",
        "",
        "| prediction | n | corr | R2 | MAE | slope | intercept |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| chi_m | {momentum['n']} | {_fmt(momentum['corr'])} | {_fmt(momentum['R2'])} | "
        f"{_fmt(momentum['MAE'])} | {_fmt(momentum['slope'])} | {_fmt(momentum['intercept'])} |",
        f"| chi_mv | {mv['n']} | {_fmt(mv['corr'])} | {_fmt(mv['R2'])} | "
        f"{_fmt(mv['MAE'])} | {_fmt(mv['slope'])} | {_fmt(mv['intercept'])} |",
        "",
        "## 量纲与 epsilon 约定",
        "",
        "A0C 的观测 `chi=mean[(X_plus-X_minus)/(2*epsilon)]` 已经除以 "
        "`epsilon=rho*sigma_q`，因而是单位扰动响应。按本次明确指令，主预测仍乘一次 "
        "`rho*sigma_q`；这使预测具有原始位移尺度，而目标具有单位响应尺度。该约定不是 "
        "拟合参数，但构成必须保留的量纲不一致，R2/MAE 的幅值解释须带此 caveat。",
        "",
        "另一个预先指定的不一致是方向：本审计按要求使用全局 "
        "`v03_pilot/probe_direction_q.pt`，而 v07 A0C `chi` 是沿各 seed/state 自身的 "
        "state-specific q 测得。因此结果是所指定全局 q 理论量与 state-specific q 响应的比较。",
        "",
        "## Replay audit",
        "",
        f"- Matched states: `{summary.get('matched_state_count')}`",
        f"- Matched E2 streams: `{summary.get('matched_stream_count')}`",
        "- Gradient capture: after `scaler.unscale_(optimizer)`, before clipping and before "
        "`optimizer.step()`.",
        "- No optimizer step, training, checkpoint write, empirical scale, intercept correction, "
        "or calibration was used to form predictions.",
    ] + sensitivity_lines + [
        "",
        "## Numerical replay caveat",
        "",
        "原 v07 分支在 CUDA bfloat16 下运行；本次因约束与运行环境使用 CPU，沿既有 "
        "`make_amp` CPU 路径以 FP32 forward/backward。保存的原始 pre-clip 范数与 CPU "
        "重放范数可在 `state_results[*].stream_audit` 中逐流核对。",
        "",
        "## Per-seed metrics",
        "",
        "```json",
        __import__("json").dumps(
            {key: value["per_seed"] for key, value in metrics.items()},
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Per-horizon metrics",
        "",
        "```json",
        __import__("json").dumps(
            {key: value["per_horizon"] for key, value in metrics.items()},
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
    ]
    return "\n".join(lines)
