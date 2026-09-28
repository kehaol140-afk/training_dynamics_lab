"""Read-only AdamW m+v response theory for the v07 A0C audit.

[FR-OBSERVATION-ONLY] This module never advances optimizer or scaler state.
[FR-NATIVE-ADAMW] Moment tensors and hyperparameters come from the saved
native AdamW checkpoint without fitted scales or bias-correction additions.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence


def _torch():
    import torch

    return torch


def _finite_float(value, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


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
                raise ValueError(
                    f"optimizer param id appears in multiple groups: {param_id!r}"
                )
            result[param_id] = group
    return result


def _lookup_state_slot(state: Mapping, param_id: object):
    if param_id in state:
        return state[param_id]
    text_id = str(param_id)
    if text_id in state:
        return state[text_id]
    raise KeyError(f"optimizer state missing param id {param_id!r}")


def _require_single(values: Sequence[float], label: str) -> float:
    unique = sorted(set(values))
    if len(unique) != 1:
        raise ValueError(
            f"scalar response formula does not define heterogeneous {label}: {unique}"
        )
    return float(unique[0])


def compute_mv_features(
    q_by_name: Mapping[str, object],
    gradient_by_name: Mapping[str, object],
    checkpoint: Mapping,
) -> Dict:
    """Compute the requested raw-m/raw-v ``mu_q`` and ``nu_q``.

    Saved AdamW parameter IDs are positional. The position is recovered from
    checkpoint model order, while q and replayed gradients are required to
    cover exactly the trainable names represented by the optimizer.
    """
    torch = _torch()
    if not isinstance(q_by_name, Mapping) or not q_by_name:
        raise ValueError("q payload must contain a non-empty name-to-tensor mapping")
    if not isinstance(gradient_by_name, Mapping) or not gradient_by_name:
        raise ValueError("gradient payload must contain a non-empty name-to-tensor mapping")

    model_state = checkpoint.get("model")
    optimizer_state = checkpoint.get("optimizer")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint missing model state mapping")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("checkpoint missing optimizer state mapping")

    q_names = set(q_by_name)
    gradient_names = set(gradient_by_name)
    parameter_names = [name for name in model_state if name in q_names]
    if set(parameter_names) != q_names:
        raise ValueError(
            "q parameter names absent from checkpoint model: "
            f"{sorted(q_names.difference(parameter_names))}"
        )
    if gradient_names != q_names:
        raise ValueError(
            "gradient/q parameter-name mismatch: "
            f"missing={sorted(q_names - gradient_names)}, "
            f"extra={sorted(gradient_names - q_names)}"
        )

    param_order = _optimizer_param_order(optimizer_state)
    if len(param_order) != len(parameter_names):
        raise ValueError(
            f"optimizer parameter count {len(param_order)} != q count {len(parameter_names)}"
        )
    group_by_param = _optimizer_groups_by_param(optimizer_state)
    state = optimizer_state.get("state", {})

    mu_q = torch.zeros((), dtype=torch.float64)
    nu_q = torch.zeros((), dtype=torch.float64)
    eta_values: List[float] = []
    beta1_values: List[float] = []
    beta2_values: List[float] = []
    eps_values: List[float] = []

    for name, param_id in zip(parameter_names, param_order):
        group = group_by_param.get(param_id)
        if group is None:
            raise KeyError(f"optimizer param group missing param id {param_id!r}")
        slot = _lookup_state_slot(state, param_id)
        if "exp_avg" not in slot or "exp_avg_sq" not in slot:
            raise KeyError(f"optimizer state missing exp_avg/exp_avg_sq for {name}")

        q = q_by_name[name]
        g = gradient_by_name[name]
        m = slot["exp_avg"]
        v = slot["exp_avg_sq"]
        if not all(torch.is_tensor(x) for x in (q, g, m, v)):
            raise TypeError(f"q, gradient, exp_avg and exp_avg_sq must be tensors for {name}")
        q64 = q.detach().to(dtype=torch.float64, device="cpu")
        g64 = g.detach().to(dtype=torch.float64, device="cpu")
        m64 = m.detach().to(dtype=torch.float64, device="cpu")
        v64 = v.detach().to(dtype=torch.float64, device="cpu")
        shapes = {tuple(x.shape) for x in (q64, g64, m64, v64)}
        if len(shapes) != 1:
            raise ValueError(f"q/g/m/v shape mismatch at {name}: {sorted(shapes)}")
        if not all(bool(torch.isfinite(x).all()) for x in (q64, g64, m64, v64)):
            raise ValueError(f"q/g/m/v contains NaN or Inf at {name}")
        if bool(torch.any(v64 < 0.0)):
            raise ValueError(f"exp_avg_sq contains negative values at {name}")

        eta = _finite_float(group.get("lr"), field=f"lr[{name}]")
        betas = group.get("betas")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError(f"optimizer group missing beta1/beta2 for {name}")
        beta1 = _finite_float(betas[0], field=f"beta1[{name}]")
        beta2 = _finite_float(betas[1], field=f"beta2[{name}]")
        adam_eps = _finite_float(group.get("eps"), field=f"eps[{name}]")
        if eta < 0.0 or not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(
                f"invalid AdamW lr/betas at {name}: {eta}, {beta1}, {beta2}"
            )
        if adam_eps < 0.0:
            raise ValueError(f"invalid AdamW eps at {name}: {adam_eps}")

        sqrt_v = torch.sqrt(v64)
        mu_q += torch.sum((q64 * q64) / (sqrt_v + adam_eps))

        numerator = q64 * q64 * m64 * g64
        denominator = sqrt_v * (sqrt_v + adam_eps) ** 2
        singular = denominator == 0.0
        if bool(torch.any(singular & (numerator != 0.0))):
            raise ValueError(f"nu_q is singular at zero exp_avg_sq for {name}")
        term = torch.zeros_like(numerator)
        nonsingular = ~singular
        term[nonsingular] = numerator[nonsingular] / denominator[nonsingular]
        nu_q += torch.sum(term)

        eta_values.append(eta)
        beta1_values.append(beta1)
        beta2_values.append(beta2)
        eps_values.append(adam_eps)

    result = {
        "mu_q": float(mu_q.item()),
        "nu_q": float(nu_q.item()),
        "eta": _require_single(eta_values, "lr"),
        "beta1": _require_single(beta1_values, "beta1"),
        "beta2": _require_single(beta2_values, "beta2"),
        "eps": _require_single(eps_values, "eps"),
        "n_parameter_tensors": len(parameter_names),
        "parameter_names": parameter_names,
        "alignment": "MODEL_STATE_NAMES_TO_OPTIMIZER_ORDER_WITH_EXACT_Q_G_SHAPE_CHECK",
        "m_semantics": "RAW_EXP_AVG_NO_BIAS_CORRECTION",
        "v_semantics": "RAW_EXP_AVG_SQ_NO_BIAS_CORRECTION",
        "g_semantics": "REPLAYED_UNSCALED_GLOBAL_NORM_CLIPPED_FIRST_STEP_GRADIENT",
    }
    if not math.isfinite(result["mu_q"]) or not math.isfinite(result["nu_q"]):
        raise ValueError("mu_q or nu_q is not finite")
    return result


def predict_unit_response(
    *,
    mu_q: float,
    nu_q: float,
    eta: float,
    beta1: float,
    beta2: float,
    horizon: int,
) -> Dict[str, float]:
    """Predict A0C's epsilon-normalized response without another epsilon."""
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be positive")
    values = {
        "mu_q": _finite_float(mu_q, field="mu_q"),
        "nu_q": _finite_float(nu_q, field="nu_q"),
        "eta": _finite_float(eta, field="eta"),
        "beta1": _finite_float(beta1, field="beta1"),
        "beta2": _finite_float(beta2, field="beta2"),
    }
    if values["eta"] < 0.0:
        raise ValueError("eta must be nonnegative")
    if not 0.0 <= values["beta1"] < 1.0 or not 0.0 <= values["beta2"] < 1.0:
        raise ValueError("beta1 and beta2 must lie in [0, 1)")
    chi_m = -values["eta"] * values["mu_q"] * (1.0 - values["beta1"] ** h)
    chi_v = values["eta"] * values["nu_q"] * (1.0 - values["beta2"] ** h)
    return {
        "chi_m": float(chi_m),
        "chi_v": float(chi_v),
        "chi_mv": float(chi_m + chi_v),
    }
