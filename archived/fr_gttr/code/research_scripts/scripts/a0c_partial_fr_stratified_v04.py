#!/usr/bin/env python3
"""Seed- and horizon-stratified robustness checks for A0C partial FR.

This script is analysis-only. It reads the frozen v01 A0C state/horizon export,
fits the two pre-specified OLS models with intercepts, and writes new v03
artifacts. It does not modify the v01 input or any experiment data.

[FR-RNG-CRITICAL] The pooled row bootstrap uses a fixed RNG seed.
[FR-OBSERVATION-ONLY] Results are diagnostic associations, not causal effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


INPUT_RELATIVE_PATH = Path(
    "outputs/a0_fr_audit/v01/direct_path/a0c_state_horizon.csv"
)
OUTPUT_RELATIVE_DIR = Path("outputs/a0_fr_audit/v03/partial_fr_stratified")
REQUIRED_COLUMNS = ("seed", "state_step", "horizon", "B", "chi", "alpha_direct")
TARGET_HORIZONS = (1, 2, 4, 8)
BOOTSTRAP_REPS = 1_000
BOOTSTRAP_SEED = 20260909
RESULT_COLUMNS = (
    "n",
    "R2_base",
    "R2_control",
    "delta_R2",
    "B_coef",
    "B_pvalue",
    "alpha_coef",
    "alpha_pvalue",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_input(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    data = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if data.isna().any().any():
        null_counts = data.isna().sum()
        bad = {column: int(count) for column, count in null_counts.items() if count}
        raise ValueError(f"Required columns contain missing values: {bad}")

    numeric = data.loc[:, ("state_step", "B", "chi", "alpha_direct")].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric).all():
        raise ValueError("Required numeric columns contain non-finite values")

    duplicate_count = int(
        data.duplicated(subset=["seed", "state_step", "horizon"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            "Input contains duplicate (seed, state_step, horizon) rows: "
            f"{duplicate_count}"
        )

    present_horizons = set(int(value) for value in data["horizon"].unique())
    missing_horizons = sorted(set(TARGET_HORIZONS) - present_horizons)
    if missing_horizons:
        raise ValueError(f"Input is missing requested horizons: {missing_horizons}")

    return data


def fit_ols(frame: pd.DataFrame, predictors: Iterable[str]) -> dict[str, Any]:
    predictor_names = tuple(predictors)
    y = frame["chi"].to_numpy(dtype=float)
    values = frame.loc[:, predictor_names].to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(frame), dtype=float), values))

    n, parameter_count = design.shape
    residual_df = n - parameter_count
    if residual_df <= 0:
        raise ValueError(
            f"OLS needs positive residual degrees of freedom; n={n}, "
            f"parameters={parameter_count}"
        )

    rank = int(np.linalg.matrix_rank(design))
    if rank != parameter_count:
        raise ValueError(
            f"OLS design is rank deficient; rank={rank}, parameters={parameter_count}"
        )

    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    sse = float(residual @ residual)
    centered = y - float(y.mean())
    sst = float(centered @ centered)
    if not sst > 0.0:
        raise ValueError("OLS response has zero total sum of squares")

    r2 = 1.0 - sse / sst
    mse = sse / residual_df
    covariance = np.linalg.inv(design.T @ design) * mse
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistics = beta / standard_errors
    pvalues = 2.0 * student_t.sf(np.abs(t_statistics), residual_df)

    return {
        "n": int(n),
        "R2": float(r2),
        "residual_df": int(residual_df),
        "coefficients": {
            name: float(value)
            for name, value in zip(("intercept", *predictor_names), beta, strict=True)
        },
        "pvalues": {
            name: float(value)
            for name, value in zip(
                ("intercept", *predictor_names), pvalues, strict=True
            )
        },
    }


def partial_fr_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    base = fit_ols(frame, ("B",))
    control = fit_ols(frame, ("B", "alpha_direct"))
    return {
        "n": int(len(frame)),
        "R2_base": float(base["R2"]),
        "R2_control": float(control["R2"]),
        "delta_R2": float(control["R2"] - base["R2"]),
        "B_coef": float(control["coefficients"]["B"]),
        "B_pvalue": float(control["pvalues"]["B"]),
        "alpha_coef": float(control["coefficients"]["alpha_direct"]),
        "alpha_pvalue": float(control["pvalues"]["alpha_direct"]),
    }


def stratified_results(
    data: pd.DataFrame, group_column: str, values: Iterable[int]
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for value in values:
        group = data.loc[data[group_column] == value]
        if group.empty:
            raise ValueError(f"No rows found for {group_column}={value}")
        rows.append({group_column: int(value), **partial_fr_metrics(group)})
    return pd.DataFrame(rows, columns=(group_column, *RESULT_COLUMNS))


def bootstrap_delta_r2(data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(data)
    deltas = np.empty(BOOTSTRAP_REPS, dtype=float)

    for replicate in range(BOOTSTRAP_REPS):
        row_indices = rng.integers(0, n, size=n)
        sample = data.iloc[row_indices]
        base = fit_ols(sample, ("B",))
        control = fit_ols(sample, ("B", "alpha_direct"))
        deltas[replicate] = float(control["R2"] - base["R2"])

    if not np.isfinite(deltas).all():
        raise RuntimeError("Bootstrap produced non-finite delta_R2 values")

    point = partial_fr_metrics(data)
    ci_lower, ci_upper = np.quantile(deltas, (0.025, 0.975), method="linear")
    summary = {
        "analysis": "overall_delta_R2_bootstrap",
        "models": {
            "base": "chi ~ 1 + B",
            "control": "chi ~ 1 + B + alpha_direct",
        },
        "n": int(n),
        "R2_base": float(point["R2_base"]),
        "R2_control": float(point["R2_control"]),
        "point_estimate_delta_R2": float(point["delta_R2"]),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "valid_reps": int(len(deltas)),
        "random_seed": BOOTSTRAP_SEED,
        "sampling_unit": "row",
        "method": "nonparametric_percentile",
        "confidence_level": 0.95,
        "ci95": [float(ci_lower), float(ci_upper)],
        "bootstrap_mean_delta_R2": float(deltas.mean()),
        "bootstrap_std_delta_R2": float(deltas.std(ddof=1)),
    }
    return deltas, summary


def markdown_table(frame: pd.DataFrame, key_column: str) -> str:
    columns = [key_column, *RESULT_COLUMNS]
    labels = {
        key_column: key_column,
        "n": "n",
        "R2_base": "R2_base",
        "R2_control": "R2_control",
        "delta_R2": "delta_R2",
        "B_coef": "B_coef",
        "B_pvalue": "B_pvalue",
        "alpha_coef": "alpha_coef",
        "alpha_pvalue": "alpha_pvalue",
    }
    lines = [
        "| " + " | ".join(labels[column] for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in frame.to_dict(orient="records"):
        cells: list[str] = []
        for column in columns:
            value = record[column]
            if column in (key_column, "n"):
                cells.append(str(int(value)))
            elif column.endswith("pvalue"):
                cells.append(f"{float(value):.6g}")
            else:
                cells.append(f"{float(value):.6f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(
    data: pd.DataFrame,
    seed_results: pd.DataFrame,
    horizon_results: pd.DataFrame,
    bootstrap: dict[str, Any],
    input_path: Path,
    input_sha256: str,
) -> str:
    positive_seed_count = int((seed_results["delta_R2"] > 0.0).sum())
    positive_horizon_count = int((horizon_results["delta_R2"] > 0.0).sum())
    seed_min = float(seed_results["delta_R2"].min())
    seed_max = float(seed_results["delta_R2"].max())
    horizon_min = float(horizon_results["delta_R2"].min())
    horizon_max = float(horizon_results["delta_R2"].max())
    ci_low, ci_high = bootstrap["ci95"]

    return "\n".join(
        [
            "# A0C Partial FR：Seed 与 Horizon 分层稳健性验证",
            "",
            "## 技术摘要",
            "",
            f"- 整体 pooled 结果复现：`R2_base={bootstrap['R2_base']:.8f}`，"
            f"`R2_control={bootstrap['R2_control']:.8f}`，"
            f"`delta_R2={bootstrap['point_estimate_delta_R2']:.8f}`。",
            f"- `alpha_direct` 加入后，`delta_R2` 在 {positive_seed_count}/{len(seed_results)} 个 seed "
            f"和 {positive_horizon_count}/{len(horizon_results)} 个指定 horizon 上为正。",
            f"- 1000 次行级非参数 bootstrap 的 95% percentile CI 为 "
            f"`[{ci_low:.8f}, {ci_high:.8f}]`。",
            "- 本结果是分层关联稳健性检查，不改变实验定义、数据或阈值，也不构成因果识别。",
            "",
            "## Seed 分层：每个 seed 的解释率均因控制项而上升",
            "",
            f"8 个 seed 的 `delta_R2` 范围为 `[{seed_min:.6f}, {seed_max:.6f}]`。"
            "下表的系数与 p 值来自控制模型。",
            "",
            markdown_table(seed_results, "seed"),
            "",
            "## Horizon 分层：H=1/2/4/8 均保持正增量",
            "",
            f"4 个指定 horizon 的 `delta_R2` 范围为 "
            f"`[{horizon_min:.6f}, {horizon_max:.6f}]`。"
            "采用精确结果表而非图形，以保留四个 horizon 的全部审计数值。",
            "",
            markdown_table(horizon_results, "horizon"),
            "",
            "## 数据范围与指标定义",
            "",
            f"- 输入：`{input_path}`",
            f"- 输入 SHA256：`{input_sha256}`",
            f"- 样本：{len(data)} 行；{data['seed'].nunique()} 个 seed；输入包含 "
            f"{data['horizon'].nunique()} 个 horizon。",
            "- `R2_base`：含截距 OLS `chi ~ B` 的样本内 R²。",
            "- `R2_control`：含截距 OLS `chi ~ B + alpha_direct` 的样本内 R²。",
            "- `delta_R2 = R2_control - R2_base`。",
            "- `B_coef/B_pvalue` 与 `alpha_coef/alpha_pvalue` 均来自控制模型。",
            "",
            "## 方法与可复现设置",
            "",
            "- Seed level：每个 seed 独立拟合两个预设模型。",
            "- Horizon level：仅对 H=1、2、4、8 分别拟合两个预设模型。",
            f"- Bootstrap：对整体 {len(data)} 行 pooled 数据有放回抽取同样行数，"
            f"固定 RNG seed={BOOTSTRAP_SEED}，共 {BOOTSTRAP_REPS} 次；"
            "区间取 `delta_R2` 的 2.5% 与 97.5% 分位数。",
            "- p 值使用常规 OLS 同方差标准误与双侧 Student-t 检验。",
            "- 计算仅使用 CPU；未导入或调用 GPU 框架。",
            "",
            "## 不确定性、稳健性边界与限制",
            "",
            "- 行级 bootstrap 与常规 OLS p 值沿用 pooled 行观测口径；它们不校正同一 seed、"
            "state 或跨 horizon 的相关性。",
            "- 分层结果检验的是符号与量级是否跨 seed/horizon 保持，不是新的确认性阈值。",
            "- `alpha_direct` 是诊断性直接通道；这些回归不把相关性解释为精确 AdamW 因果响应。",
            "- [FR-RNG-CRITICAL] bootstrap RNG 固定，结果可复现。",
            "- [FR-OBSERVATION-ONLY] 本报告只做观察性稳健性验证。",
            "",
            "## 建议的后续解读",
            "",
            "- 将 seed 与 horizon 表作为整体 `delta_R2` 的异质性审计，不据此改写既有实验结论或阈值。",
            "- 若未来需要依赖结构稳健的区间，应预先另行定义 seed/state cluster bootstrap 或稳健标准误；"
            "本脚本不引入该新定义。",
            "",
            "## 待回答问题",
            "",
            "- 在预先规定相关结构与重采样单位后，依赖感知区间是否仍与当前行级 percentile CI 一致？",
            "",
        ]
    )


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    inferred_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run A0C seed/horizon partial-FR robustness checks."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=inferred_root,
        help=f"Project root (default: {inferred_root})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    input_path = root / INPUT_RELATIVE_PATH
    output_dir = root / OUTPUT_RELATIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    input_sha256 = sha256_file(input_path)
    data = read_input(input_path)

    seeds = sorted(int(value) for value in data["seed"].unique())
    seed_results = stratified_results(data, "seed", seeds)
    horizon_results = stratified_results(data, "horizon", TARGET_HORIZONS)
    _, bootstrap = bootstrap_delta_r2(data)
    bootstrap["input"] = str(INPUT_RELATIVE_PATH)
    bootstrap["input_sha256"] = input_sha256

    seed_path = output_dir / "seed_partial_fr.csv"
    horizon_path = output_dir / "horizon_partial_fr.csv"
    bootstrap_path = output_dir / "bootstrap_delta_r2.json"
    report_path = output_dir / "A0C_PARTIAL_FR_STRATIFIED.md"

    atomic_write_csv(seed_path, seed_results)
    atomic_write_csv(horizon_path, horizon_results)
    atomic_write_text(
        bootstrap_path,
        json.dumps(bootstrap, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write_text(
        report_path,
        build_report(
            data,
            seed_results,
            horizon_results,
            bootstrap,
            INPUT_RELATIVE_PATH,
            input_sha256,
        ),
    )

    print(f"INPUT_SHA256 = {input_sha256}")
    print(f"ROWS = {len(data)}")
    print(f"SEEDS = {len(seed_results)}")
    print(f"HORIZONS = {','.join(str(value) for value in TARGET_HORIZONS)}")
    print(f"R2_BASE = {bootstrap['R2_base']:.12g}")
    print(f"R2_CONTROL = {bootstrap['R2_control']:.12g}")
    print(f"DELTA_R2 = {bootstrap['point_estimate_delta_R2']:.12g}")
    print(f"BOOTSTRAP_CI95 = [{bootstrap['ci95'][0]:.12g}, {bootstrap['ci95'][1]:.12g}]")
    print(f"OUTPUT_DIR = {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
