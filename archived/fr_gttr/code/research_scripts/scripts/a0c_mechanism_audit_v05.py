#!/usr/bin/env python3
"""A0C v05 mechanism audit using the frozen v01 state/horizon export.

The audit adds seed-cluster bootstrap uncertainty and Pearson correlation
checks while retaining the two A0C partial-FR OLS specifications from v04.
It is CPU-only and does not enter covariance geometry.

[FR-RNG-CRITICAL] Seed-cluster bootstrap uses a fixed RNG seed.
[FR-OBSERVATION-ONLY] Results are diagnostic associations, not causal effects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from a0c_partial_fr_stratified_v04 import (
    INPUT_RELATIVE_PATH,
    TARGET_HORIZONS,
    atomic_write_csv,
    atomic_write_text,
    fit_ols,
    partial_fr_metrics,
    read_input,
    sha256_file,
)


OUTPUT_RELATIVE_DIR = Path("outputs/a0_fr_audit/v05/mechanism")
BOOTSTRAP_REPS = 1_000
BOOTSTRAP_SEED = 20260909
CORRELATION_COLUMNS = (
    "level",
    "group",
    "n",
    "corr_alpha_chi",
    "corr_alpha_B",
)


def resample_seed_clusters(
    data: pd.DataFrame, drawn_seeds: Sequence[int]
) -> pd.DataFrame:
    """Concatenate complete seed blocks in the supplied bootstrap draw order."""
    if not drawn_seeds:
        raise ValueError("At least one seed must be drawn")

    blocks: list[pd.DataFrame] = []
    for seed in drawn_seeds:
        block = data.loc[data["seed"] == seed]
        if block.empty:
            raise ValueError(f"Bootstrap draw references unknown seed: {seed}")
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def cluster_bootstrap_delta_r2(
    data: pd.DataFrame,
    *,
    reps: int = BOOTSTRAP_REPS,
    random_seed: int = BOOTSTRAP_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bootstrap pooled delta_R2 by sampling complete seed clusters."""
    if reps <= 0:
        raise ValueError("Bootstrap reps must be positive")

    seeds = np.asarray(sorted(int(value) for value in data["seed"].unique()))
    if len(seeds) < 2:
        raise ValueError("Seed-cluster bootstrap requires at least two seeds")

    rng = np.random.default_rng(random_seed)
    deltas = np.empty(reps, dtype=float)
    for replicate in range(reps):
        drawn_seeds = rng.choice(seeds, size=len(seeds), replace=True).tolist()
        sample = resample_seed_clusters(data, drawn_seeds)
        metrics = partial_fr_metrics(sample)
        deltas[replicate] = float(metrics["delta_R2"])

    if not np.isfinite(deltas).all():
        raise RuntimeError("Seed-cluster bootstrap produced non-finite delta_R2 values")

    point = partial_fr_metrics(data)
    ci_lower, ci_upper = np.quantile(deltas, (0.025, 0.975), method="linear")
    cluster_sizes = {
        str(int(seed)): int((data["seed"] == seed).sum()) for seed in seeds
    }
    summary = {
        "analysis": "overall_delta_R2_seed_cluster_bootstrap",
        "models": {
            "base": "chi ~ 1 + B",
            "control": "chi ~ 1 + B + alpha_direct",
        },
        "n": int(len(data)),
        "cluster_count": int(len(seeds)),
        "cluster_ids": [int(seed) for seed in seeds],
        "cluster_sizes": cluster_sizes,
        "clusters_per_replicate": int(len(seeds)),
        "bootstrap_reps": int(reps),
        "valid_reps": int(len(deltas)),
        "random_seed": int(random_seed),
        "sampling_unit": "seed",
        "method": "nonparametric_cluster_percentile",
        "confidence_level": 0.95,
        "R2_base_point_estimate": float(point["R2_base"]),
        "R2_control_point_estimate": float(point["R2_control"]),
        "point_estimate_delta_R2": float(point["delta_R2"]),
        "bootstrap_mean_delta_R2": float(deltas.mean()),
        "bootstrap_std_delta_R2": float(deltas.std(ddof=1)),
        "ci95": [float(ci_lower), float(ci_upper)],
    }
    return deltas, summary


def correlation_audit(
    data: pd.DataFrame, *, horizons: Iterable[int] = TARGET_HORIZONS
) -> pd.DataFrame:
    """Return Pearson correlations overall, by seed, and by selected horizon."""
    rows: list[dict[str, str | int | float]] = []

    def append_row(level: str, group_name: str, group: pd.DataFrame) -> None:
        if len(group) < 2:
            raise ValueError(
                f"Correlation requires at least two rows: {level}={group_name}"
            )
        corr_alpha_chi = float(group["alpha_direct"].corr(group["chi"]))
        corr_alpha_B = float(group["alpha_direct"].corr(group["B"]))
        if not np.isfinite((corr_alpha_chi, corr_alpha_B)).all():
            raise ValueError(
                f"Correlation is undefined for constant/non-finite data: "
                f"{level}={group_name}"
            )
        rows.append(
            {
                "level": level,
                "group": group_name,
                "n": int(len(group)),
                "corr_alpha_chi": corr_alpha_chi,
                "corr_alpha_B": corr_alpha_B,
            }
        )

    append_row("overall", "all", data)
    for seed in sorted(int(value) for value in data["seed"].unique()):
        append_row("seed", str(seed), data.loc[data["seed"] == seed])
    for horizon in horizons:
        group = data.loc[data["horizon"] == horizon]
        if group.empty:
            raise ValueError(f"No rows found for horizon={horizon}")
        append_row("horizon", str(int(horizon)), group)

    return pd.DataFrame(rows, columns=CORRELATION_COLUMNS)


def model_summary(data: pd.DataFrame) -> dict[str, Any]:
    base = fit_ols(data, ("B",))
    control = fit_ols(data, ("B", "alpha_direct"))
    return {
        "analysis": "A0C_partial_FR_models_retained",
        "n": int(len(data)),
        "models": {
            "base": {"formula": "chi ~ 1 + B", **base},
            "control": {
                "formula": "chi ~ 1 + B + alpha_direct",
                **control,
            },
        },
        "delta_R2": float(control["R2"] - base["R2"]),
    }


def correlation_markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| level | group | n | corr(alpha_direct, chi) | corr(alpha_direct, B) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            f"| {row['level']} | {row['group']} | {int(row['n'])} | "
            f"{float(row['corr_alpha_chi']):.8f} | "
            f"{float(row['corr_alpha_B']):.8f} |"
        )
    return "\n".join(lines)


def build_report(
    data: pd.DataFrame,
    models: dict[str, Any],
    bootstrap: dict[str, Any],
    correlations: pd.DataFrame,
    input_sha256: str,
) -> str:
    base = models["models"]["base"]
    control = models["models"]["control"]
    overall_corr = correlations.loc[correlations["level"] == "overall"].iloc[0]
    seed_corr = correlations.loc[correlations["level"] == "seed"]
    horizon_corr = correlations.loc[correlations["level"] == "horizon"]
    ci_low, ci_high = bootstrap["ci95"]

    return "\n".join(
        [
            "# A0C Mechanism Report",
            "",
            "## 技术摘要",
            "",
            f"- 两个既定模型保持不变：`{base['formula']}` 与 "
            f"`{control['formula']}`。整体 `delta_R2={models['delta_R2']:.8f}`。",
            f"- {bootstrap['bootstrap_reps']} 次 seed-cluster bootstrap 的 `delta_R2` mean 为 "
            f"`{bootstrap['bootstrap_mean_delta_R2']:.8f}`，95% percentile CI 为 "
            f"`[{ci_low:.8f}, {ci_high:.8f}]`。",
            f"- Overall correlation：`corr(alpha_direct, chi)="
            f"{float(overall_corr['corr_alpha_chi']):.8f}`，"
            f"`corr(alpha_direct, B)={float(overall_corr['corr_alpha_B']):.8f}`。",
            "- 本审计保持观察性边界，不修改实验定义或阈值，不进入 covariance geometry。",
            "",
            "## 两个预设模型仍给出相同的整体解释率增量",
            "",
            "| model | formula | n | R2 |",
            "| --- | --- | --- | --- |",
            f"| base | `{base['formula']}` | {base['n']} | {base['R2']:.8f} |",
            f"| control | `{control['formula']}` | {control['n']} | "
            f"{control['R2']:.8f} |",
            "",
            f"控制模型相对基础模型的样本内解释率增量为 "
            f"`delta_R2={models['delta_R2']:.8f}`。模型含截距，定义与 v04 一致。",
            "",
            "## Seed-cluster bootstrap 在完整 seed block 上重采样",
            "",
            f"每个 replicate 从 {bootstrap['cluster_count']} 个 seed 中有放回抽取 "
            f"{bootstrap['clusters_per_replicate']} 个 seed，并拼接每个被抽 seed 的全部行；"
            "未在 cluster 内抽 row。",
            "",
            "| quantity | value |",
            "| --- | --- |",
            f"| point estimate delta_R2 | {bootstrap['point_estimate_delta_R2']:.8f} |",
            f"| bootstrap mean delta_R2 | {bootstrap['bootstrap_mean_delta_R2']:.8f} |",
            f"| bootstrap standard deviation | {bootstrap['bootstrap_std_delta_R2']:.8f} |",
            f"| percentile 95% CI | [{ci_low:.8f}, {ci_high:.8f}] |",
            f"| valid replicates | {bootstrap['valid_reps']}/{bootstrap['bootstrap_reps']} |",
            "",
            "该区间把 seed 作为重采样单位，因此保留一个 seed 内 state 与 horizon 行的共同变动；"
            f"只有 {bootstrap['cluster_count']} 个独立 cluster，区间精度仍受 seed 数量限制。",
            "",
            "## Correlation audit",
            "",
            "Pearson 相关系数按 overall、逐 seed 和 v04 指定的 H=1/2/4/8 输出。"
            "表格保留全部精确分层值，因此不另加图形。",
            "",
            correlation_markdown_table(correlations),
            "",
            f"逐 seed 的 `corr(alpha_direct, chi)` 范围为 "
            f"`[{seed_corr['corr_alpha_chi'].min():.8f}, "
            f"{seed_corr['corr_alpha_chi'].max():.8f}]`；逐 horizon 范围为 "
            f"`[{horizon_corr['corr_alpha_chi'].min():.8f}, "
            f"{horizon_corr['corr_alpha_chi'].max():.8f}]`。"
            "相关结果用于检查 direct channel 与 chi/B 的共同变化，不提供因果分解。",
            "",
            "## 数据、定义与复现设置",
            "",
            f"- 输入：`{INPUT_RELATIVE_PATH}`",
            f"- 输入 SHA256：`{input_sha256}`",
            f"- 样本：{len(data)} 行，{data['seed'].nunique()} 个 seed；"
            f"correlation horizon 范围固定为 {list(TARGET_HORIZONS)}。",
            "- 相关系数：未残差化的 Pearson sample correlation。",
            f"- Bootstrap：seed 为 cluster，{bootstrap['bootstrap_reps']} 次，固定 RNG seed="
            f"{bootstrap['random_seed']}，percentile 95% CI。",
            "- 计算仅使用 CPU。",
            "",
            "## 限制与审计边界",
            "",
            f"- Cluster bootstrap 对 {bootstrap['cluster_count']} 个 seed 重采样，不把 "
            f"{bootstrap['n']} 行当作独立 cluster；"
            "但 seed 数较少，percentile CI 可能较离散。",
            "- Pearson correlation 未控制 B、state_step 或其他路径，只描述成对线性共变。",
            "- `alpha_direct` 仍是诊断性直接通道；模型和相关均不识别精确 AdamW 因果机制。",
            "- Covariance geometry 明确排除在 v05 范围之外，未读取或计算其产物。",
            "- [FR-RNG-CRITICAL] bootstrap 随机种子固定。",
            "- [FR-OBSERVATION-ONLY] 本报告仅作观察性机制审计。",
            "",
            "## 建议的后续解读",
            "",
            "- 联合阅读 cluster CI、逐 seed correlation 与逐 horizon correlation，"
            "判断 direct channel 关联是否由单一 seed 或单一短 horizon 主导。",
            "- 不依据本审计调整既有阈值，也不将其用于进入 covariance geometry。",
            "",
            "## 待回答问题",
            "",
            "- 在更多预先指定的独立 seed 上，cluster bootstrap 区间宽度是否稳定？",
            "",
        ]
    )


def run_audit(
    root: Path,
    *,
    bootstrap_reps: int = BOOTSTRAP_REPS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    input_path = root / INPUT_RELATIVE_PATH
    output_dir = root / OUTPUT_RELATIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    input_sha256 = sha256_file(input_path)
    data = read_input(input_path)
    models = model_summary(data)
    _, bootstrap = cluster_bootstrap_delta_r2(
        data, reps=bootstrap_reps, random_seed=bootstrap_seed
    )
    correlations = correlation_audit(data)

    source_metadata = {
        "input": str(INPUT_RELATIVE_PATH),
        "input_sha256": input_sha256,
    }
    models.update(source_metadata)
    bootstrap.update(source_metadata)

    atomic_write_text(
        output_dir / "model_summary.json",
        json.dumps(models, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write_text(
        output_dir / "cluster_bootstrap_delta_r2.json",
        json.dumps(bootstrap, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write_csv(output_dir / "correlation_audit.csv", correlations)
    atomic_write_text(
        output_dir / "A0C_MECHANISM_REPORT.md",
        build_report(data, models, bootstrap, correlations, input_sha256),
    )

    return {
        "output_dir": output_dir,
        "models": models,
        "bootstrap": bootstrap,
        "correlations": correlations,
    }


def parse_args() -> argparse.Namespace:
    inferred_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the A0C v05 mechanism audit.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=inferred_root,
        help=f"Project root (default: {inferred_root})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_audit(args.project_root)
    bootstrap = outputs["bootstrap"]
    models = outputs["models"]
    overall = outputs["correlations"].loc[
        outputs["correlations"]["level"] == "overall"
    ].iloc[0]

    print(f"INPUT_SHA256 = {bootstrap['input_sha256']}")
    print(f"R2_BASE = {models['models']['base']['R2']:.12g}")
    print(f"R2_CONTROL = {models['models']['control']['R2']:.12g}")
    print(f"DELTA_R2 = {models['delta_R2']:.12g}")
    print(f"CLUSTER_BOOTSTRAP_MEAN = {bootstrap['bootstrap_mean_delta_R2']:.12g}")
    print(
        f"CLUSTER_BOOTSTRAP_CI95 = "
        f"[{bootstrap['ci95'][0]:.12g}, {bootstrap['ci95'][1]:.12g}]"
    )
    print(f"CORR_ALPHA_CHI_OVERALL = {overall['corr_alpha_chi']:.12g}")
    print(f"CORR_ALPHA_B_OVERALL = {overall['corr_alpha_B']:.12g}")
    print(f"OUTPUT_DIR = {outputs['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
