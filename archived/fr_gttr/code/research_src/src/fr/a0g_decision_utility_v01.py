"""FR_A0G decision-utility v01: analysis module.

把完整切线响应用作训练动作（当前单步数据改权）的决策评分，回答：

    完整切线能否帮助选择一个真正降低未来验证损失的训练动作，
    并且该收益是否值得额外计算成本？

[FR-OBSERVATION-ONLY] 本模块只做分析/记录，不改任何冻结制品。
输入只取自 v07 bank / 冻结 summary / 本实验新建 confirmatory bank。

方法阶梯（正文中文名，代码键）：
  random     随机选择基线
  alignment  当前梯度对齐基线   hhat = v_t^T r_1^(j)（只用当前态验证梯度，无传播）
  B          仅优化器切线预测   hhat = v_H^T r_H^(j)（未来步 delta_g=0）
  C          完整切线预测       hhat = v_H^T r_H^(j)（未来步 clip_jvp(HVP)）

约定：tangent r_H 始终定义为对 lambda 的导数（每单位 lambda），
有限动作预测 = lambda * hhat。禁止重复乘 lambda。

定位声明：本实验是 trajectory-conditioned local counterfactual scoring
（给定 baseline future trajectory 的条件决策），不是 current-state-only forecast。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .a0c_adamw_response_propagation_v01 import _metric, relative_error


METHOD_KEYS = ("random", "alignment", "B", "C")
METHOD_NAMES_ZH = {
    "random": "随机选择基线",
    "alignment": "当前梯度对齐基线",
    "B": "仅优化器切线预测",
    "C": "完整切线预测",
}

DATA_GROUP_VERSION = "FR_DECISION_UTILITY_V01_DATA_GROUP_V01"
VALIDATION_SPLIT_VERSION = "FR_DECISION_UTILITY_V01_VAL_SPLIT_V01"
N_DATA_GROUPS = 4
LAMBDA_CANDIDATES = (0.01, 0.025, 0.05, 0.10)
PRIMARY_HORIZON = 4
HORIZONS = (1, 4, 8)
NEAR_TIE_FRACTION = 0.05


# ---------------------------------------------------------------------------
# 数据组定义（确定性、互斥、可复现）
# ---------------------------------------------------------------------------

def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sample_text_sha256(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


def assign_data_group(sample_index: int, text_sha256: str, n_groups: int = N_DATA_GROUPS) -> int:
    """把训练池样本确定性分配到 n_groups 个互斥桶之一。

    输入只有稳定的 sample id（e3c_subset.jsonl 行号）与内容哈希；
    不使用任何 validation/probe 信息，不使用随机数。
    该分桶仅用于决策管线验证，不代表语义数据域。
    """
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    key = f"{DATA_GROUP_VERSION}|id={int(sample_index)}|sha={text_sha256}".encode("utf-8")
    return int(sha256_hex(key)[:12], 16) % int(n_groups)


def build_data_group_definition(train_indices: Sequence[int], texts: Sequence[str]) -> Dict:
    """返回四组确定性数据组定义（成员按 id 升序，含大小与完整性校验）。"""
    train_set = set(map(int, train_indices))
    if len(train_set) != len(train_indices):
        raise ValueError("train_indices contain duplicates")
    if len(texts) <= max(train_set):
        raise ValueError("texts do not cover all train indices")
    groups: Dict[int, list[int]] = {g: [] for g in range(N_DATA_GROUPS)}
    for idx in sorted(train_set):
        bucket = assign_data_group(idx, sample_text_sha256(str(texts[int(idx)])))
        groups[bucket].append(int(idx))
    memberships = [set(v) for v in groups.values()]
    for a in range(N_DATA_GROUPS):
        for b in range(a + 1, N_DATA_GROUPS):
            if memberships[a] & memberships[b]:
                raise RuntimeError(f"data groups {a}/{b} overlap")
    if set().union(*memberships) != train_set:
        raise RuntimeError("data groups do not cover the train pool")
    sizes = {str(g): len(v) for g, v in groups.items()}
    if min(sizes.values()) < 100 and len(train_set) >= 4 * 100:
        raise RuntimeError(f"data group too small: {sizes}")
    rule_payload = {
        "group_version": DATA_GROUP_VERSION,
        "n_groups": N_DATA_GROUPS,
        "hash_input": "sha256(f'{DATA_GROUP_VERSION}|id=<line_index>|sha=<sha256_utf8_text>') first 12 hex chars mod n_groups",
        "disclaimer": "仅用于决策管线验证，不代表语义数据域 (hash-bucketed by stable sample id + content hash; no source/category/domain metadata exists in e3c_subset.jsonl)",
        "members": {str(g): v for g, v in groups.items()},
        "sizes": sizes,
    }
    rule_payload["definition_sha256"] = sha256_hex(
        json.dumps(
            {k: rule_payload[k] for k in ("group_version", "n_groups", "members", "sizes")},
            sort_keys=True,
        ).encode("utf-8")
    )
    return rule_payload


# ---------------------------------------------------------------------------
# validation 拆分（V_score / V_eval，不重叠）
# ---------------------------------------------------------------------------

def split_validation_pool(eval_indices: Sequence[int]) -> Dict:
    """把冻结 eval pool 确定性拆成 V_score / V_eval（排序后奇偶位）。

    V_score: 计算 validation readout gradient、动作打分、动作选择、开发 lambda。
    V_eval : 最终评价所选动作真实收益。
    两块不重叠，且与 probe/train 天然不相交（eval pool 自轨迹第 0 步永久排除）。
    """
    ordered = sorted(set(map(int, eval_indices)))
    if len(set(eval_indices)) != len(eval_indices):
        raise ValueError("eval_indices contain duplicates")
    v_score = ordered[0::2]
    v_eval = ordered[1::2]
    if set(v_score) & set(v_eval):
        raise RuntimeError("V_score/V_eval overlap")
    payload = {
        "split_version": VALIDATION_SPLIT_VERSION,
        "rule": "sorted(eval_indices): even positions -> V_score, odd positions -> V_eval",
        "v_score": v_score,
        "v_eval": v_eval,
        "n_v_score": len(v_score),
        "n_v_eval": len(v_eval),
        "disjoint": True,
        "source": "outputs/e3_confirmatory/v07/partition_v07.json eval_indices (held out from every trajectory since step 0)",
    }
    payload["split_sha256"] = sha256_hex(
        json.dumps({"v_score": v_score, "v_eval": v_eval}, sort_keys=True).encode("utf-8")
    )
    return payload


# ---------------------------------------------------------------------------
# 指标（Level 1 / Level 3）
# ---------------------------------------------------------------------------

def _rankdata(values: Sequence[float]) -> np.ndarray:
    """平均名次（1-based），并列取平均。"""
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=np.float64)
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _safe_spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    ra, rb = _rankdata(a), _rankdata(b)
    if float(np.std(ra)) == 0.0 or float(np.std(rb)) == 0.0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def sign_of(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def level1_row_metrics(
    *, hhat_alignment: float, hhat_B: float, hhat_C: float, h_fd: float,
) -> Dict[str, Dict]:
    """对单点 (state,stream,j,H) 给三种方法的误差字典。"""
    out = {}
    for key, pred in (("alignment", hhat_alignment), ("B", hhat_B), ("C", hhat_C)):
        out[key] = {
            "predicted": float(pred),
            "fd": float(h_fd),
            "abs_error": abs(float(pred) - float(h_fd)),
            "relative_error": relative_error(float(h_fd), float(pred)),
            "sign_match": sign_of(float(pred)) == sign_of(float(h_fd)),
        }
    return out


def per_state_spearman(pairs: Sequence[Mapping]) -> Dict[str, float | None]:
    """pairs: [{'hhat':..., 'h_fd':..., 'method':...}] -> 每方法 Spearman（同一 state 内）。"""
    out = {}
    for m in ("alignment", "B", "C"):
        vals = [p for p in pairs if p["method"] == m]
        if len(vals) < 2:
            out[m] = None
            continue
        out[m] = _safe_spearman([p["hhat"] for p in vals], [p["h_fd"] for p in vals])
    return out


def pooled_metric(rows: Sequence[Mapping], pred_key: str, actual_key: str) -> Dict:
    """复用冻结 a0c 指标定义（禁止重定义）。"""
    return _metric(
        [float(r[actual_key]) for r in rows],
        [float(r[pred_key]) for r in rows],
    )


# ---------------------------------------------------------------------------
# Level 3：决策效用
# ---------------------------------------------------------------------------

def select_action(predicted: Mapping[int, float]) -> int:
    """j_hat = argmin_j predicted Delta L（j=0 为不干预，保留在候选集中）。"""
    if not predicted:
        raise ValueError("empty prediction set")
    return int(min(predicted, key=lambda j: (float(predicted[j]), j)))


def level3_decision(
    *, predicted_by_j: Mapping[int, float], actual_by_j: Mapping[int, float],
    near_tie_fraction: float = NEAR_TIE_FRACTION,
) -> Dict:
    """给定一个 (state,stream,H,method) 的预测/实际 Delta L，返回决策指标。

    Top-1 带 near-tie tolerance：实际最优与次优差 < tol 时记为等价最优。
    tol = near_tie_fraction * (max_actual - min_actual)，floor 1e-12。
    """
    if set(predicted_by_j) != set(actual_by_j) or not predicted_by_j:
        raise ValueError("prediction/actual candidate sets differ")
    j_hat = select_action(predicted_by_j)
    actual = {int(j): float(v) for j, v in actual_by_j.items()}
    best = min(actual.values())
    worst = max(actual.values())
    tol = max(1e-12, float(near_tie_fraction) * (worst - best))
    tied_optimal = sorted(j for j, v in actual.items() if v - best <= tol)
    return {
        "j_hat": j_hat,
        "predicted_delta_selected": float(predicted_by_j[j_hat]),
        "actual_delta_selected": actual[j_hat],
        "best_possible_delta": best,
        "gain": -actual[j_hat],
        "regret": actual[j_hat] - best,
        "harmful": bool(actual[j_hat] > 0.0),
        "top1_correct": bool(j_hat in tied_optimal),
        "near_tie_tolerance": tol,
        "tied_optimal": tied_optimal,
        "direction_correct": sign_of(float(predicted_by_j[j_hat])) == sign_of(actual[j_hat]),
    }


def random_choice(rng: np.random.Generator, n_candidates: int) -> int:
    return int(rng.integers(0, int(n_candidates)))


# ---------------------------------------------------------------------------
# 成本核算（§12）
# ---------------------------------------------------------------------------

def cost_model(measured: Mapping[str, float], horizons=HORIZONS, m_grid=(1, 2, 4)) -> list[Dict]:
    """direct replay vs shared-baseline tangent 成本模型。

    measured 需要（每个 state-stream 平均的 GPU 秒）：
      t_native_step        单步 native forward/backward(+clip/step) 平均
      t_val_loss_one       一次 endpoint validation loss 测量（V 集 forward）
      t_val_grad_one       一次 endpoint validation gradient（V_score backward）
      t_cand_grad_one      一个候选数据组梯度 g_j（2 microbatch fwd+bwd）
      t_hvp_one            单方向单步 HVP
      t_tangent_step_one   单步全部 tangent 传播（clip_jvp+adamw_tangent_step）
    显式不含 GPU 空闲/加载 checkpoint 时间（两者同等摊销）。
    """
    t_native = float(measured["t_native_step"])
    t_vloss = float(measured["t_val_loss_one"])
    t_vgrad = float(measured["t_val_grad_one"])
    t_cand = float(measured["t_cand_grad_one"])
    t_hvp = float(measured["t_hvp_one"])
    t_tstep = float(measured["t_tangent_step_one"])
    rows = []
    for h in map(int, horizons):
        for m in map(int, m_grid):
            c_base = h * t_native
            direct = c_base + t_vloss + m * (c_base + t_vloss)
            tangent = c_base + t_vloss + t_vgrad + m * (
                t_cand + max(h - 1, 0) * t_hvp + h * t_tstep
            )
            rows.append({
                "H": h,
                "M_candidates": m,
                "direct_replay_total_s": direct,
                "shared_tangent_total_s": tangent,
                "direct_per_candidate_s": direct / m,
                "shared_tangent_per_candidate_s": tangent / m,
                "cheaper": "shared_tangent" if tangent < direct else ("direct_replay" if direct < tangent else "tie"),
                "tangent_speedup_vs_direct": direct / tangent if tangent > 0 else None,
            })
    return rows


def breakeven_m(measured: Mapping[str, float], h: int, m_max: int = 64) -> int | None:
    """返回使 shared-tangent 比 direct replay 便宜的最小候选数 M（无则 None）。"""
    rows = cost_model(measured, horizons=(int(h),), m_grid=tuple(range(1, m_max + 1)))
    for row in rows:
        if row["shared_tangent_total_s"] < row["direct_replay_total_s"]:
            return int(row["M_candidates"])
    return None


# ---------------------------------------------------------------------------
# 汇总与报告
# ---------------------------------------------------------------------------

def _fmt(value) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.12g}"


def assemble_summary(
    *, config: Mapping, data_group_definition: Mapping, validation_split: Mapping,
    lambda_selection: Mapping, level1_rows: Sequence[Mapping],
    level1b_rows: Sequence[Mapping], level2_rows: Sequence[Mapping],
    level3_rows: Sequence[Mapping], level3_aggregate: Mapping,
    cost_rows: Sequence[Mapping], runtime: Mapping,
    conditional_lookahead_limitation: str,
) -> Dict:
    return {
        "protocol": "FR_DECISION_UTILITY_V01",
        "status": "COMPLETE",
        "config": dict(config),
        "data_group_definition_sha256": data_group_definition["definition_sha256"],
        "validation_split_sha256": validation_split["split_sha256"],
        "lambda_selection": dict(lambda_selection),
        "primary_horizon": PRIMARY_HORIZON,
        "level1_local_derivative_rows": len(level1_rows),
        "level1_finite_action_rows": len(level1b_rows),
        "level2_action_prediction_rows": len(level2_rows),
        "level3_decision_rows": len(level3_rows),
        "level1_rows": list(level1_rows),
        "level1b_rows": list(level1b_rows),
        "level2_rows": list(level2_rows),
        "level3_rows": list(level3_rows),
        "level3_aggregate": dict(level3_aggregate),
        "cost_rows": list(cost_rows),
        "runtime": dict(runtime),
        "conditional_lookahead_limitation": conditional_lookahead_limitation,
        "math_sdpa_hvp_caveat": (
            "C/完整切线的未来步 HVP 使用 math-SDPA 代理（native flash-attention 无 double "
            "backward），是混合后端解析代理而非精确 flash 核 tangent；与 A0C/A0E2 相同。"
        ),
        "constraints": [
            "EXACT_V07_STATE_Q_CHECKPOINT_MANIFEST_STREAM_FOR_DEV_STATES",
            "NEW_SEED_CONFIRMATORY_BANK_2038_2041_FOR_FORMAL_STATES",
            "V_SCORE_USED_ONLY_FOR_READOUT_SCORING_SELECTION_LAMBDA_DEV",
            "V_EVAL_USED_ONLY_FOR_FINAL_ACTION_EVALUATION",
            "PULSE_POSITION_IDENTICAL_TO_CAUSAL_PULSE_POST_UNSCALE_PRE_CLIP",
            "TANGENT_R_IS_DERIVATIVE_PER_UNIT_LAMBDA_FINITE_PREDICTION_IS_LAMBDA_TIMES_HHAT",
            "NO_LAMBDA_RETUNING_ON_FORMAL_STATES",
            "METRICS_REUSED_FROM_FROZEN_A0C_MODULE",
            "CONDITIONAL_LOOKAHEAD_BASELINE_FUTURE_TRAJECTORY_GIVEN",
        ],
    }


def render_lambda_selection_report(summary_lambda: Mapping) -> str:
    lines = [
        "# LAMBDA SELECTION REPORT — FR_DECISION_UTILITY_V01",
        "",
        "lambda 只在开发状态（seeds 2030–2037 中未参与 A0C–A0F2 锁定的少量状态）上扫描确定；",
        "正式评价状态（新 seeds 2038–2041）不参与 lambda 选择。",
        "",
        "## 开发扫描（候选 lambda，+lambda 分支，V_score 读数）",
        "",
        "准则 A（严格局部线性）：mean slope-dev vs λ0 ≤ 0.25 且 clip 穿越 ≤ 0.10。",
        "准则 B（决策信号，回退）：点态线性不成立时，在满足 clip 穿越 ≤ 0.10 的候选中取",
        "最大 lambda（最大化 SNR/符号一致率）；排序信号（组内 Spearman）不依赖点态线性。",
        "",
        "| lambda | mean |hhat_C| | mean slope-dev vs λ0 | p90 slope-dev | clip crossing | sign agree | mean signal | mean dev | SNR | Spearman |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_lambda["scan_rows"]:
        lines.append(
            f"| {_fmt(row['lambda_value'])} | {_fmt(row['mean_abs_hhat_C'])} | "
            f"{_fmt(row.get('mean_slope_dev_vs_lambda0'))} | "
            f"{_fmt(row.get('p90_slope_dev_vs_lambda0'))} | "
            f"{_fmt(row['clip_crossing_rate'])} | "
            f"{_fmt(row.get('sign_agreement'))} | "
            f"{_fmt(row.get('mean_abs_signal'))} | "
            f"{_fmt(row.get('mean_abs_dev'))} | "
            f"{_fmt(row.get('snr'))} | "
            f"{_fmt(row.get('mean_within_state_spearman'))} |"
        )
    lines.extend([
        "",
        "## 冻结决定",
        "",
        f"- 主 lambda（primary）: **{_fmt(summary_lambda['primary_lambda'])}**",
        f"- 保守 lambda（conservative）: **{_fmt(summary_lambda['conservative_lambda'])}**"
        if summary_lambda.get("conservative_lambda") is not None
        else "- 保守 lambda: 未启用",
        f"- 冻结规则: {summary_lambda.get('freeze_rule', 'n/a')}",
        f"- 开发组内 Spearman（hhat_C vs 实际 ΔL，跨动作）: "
        f"{_fmt(summary_lambda.get('mean_within_state_spearman'))}",
        "",
        f"理由: {summary_lambda['rationale']}",
        "",
        "## 通过准则核对",
        "",
    ])
    for check, ok in summary_lambda["criteria"].items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} — {check}")
    lines.extend([
        "",
        "说明: action gradient 为 g_action = g + lambda*(g_j - g)，是与基准 batch 梯度同归一化",
        "（8 样本均值）的凸混合；lambda 冻结后对正式状态只读使用。",
        "",
    ])
    return "\n".join(lines)


def render_report(summary: Mapping) -> str:
    agg = summary["level3_aggregate"]
    verdict = summary.get("final_verdict")
    lines = [
        "# FR DECISION UTILITY REPORT v01",
        "",
        "问题: 完整切线响应能否帮助选择一个真正降低未来验证损失的训练动作，",
        "且该收益是否值得额外计算成本？",
        "",
        "## 结论（按 §14 成功标准）",
        "",
        f"- 条件定位: {summary['conditional_lookahead_limitation']}",
        f"- math-SDPA HVP caveat: {summary['math_sdpa_hvp_caveat']}",
    ]
    if verdict:
        lines.extend([
            f"- **{verdict['headline']}**",
            f"- Level 1 局部导数: {verdict['level1']}",
            f"- Level 3 决策效用（formal H=4）: {verdict['level3_h4']}",
            f"- 决策效用的稳健性（分层审计）: {verdict['audit']}",
            f"- 计算效率（§12 成本补测）: {verdict['cost']}",
            f"- §14 判定: {verdict['criteria']}",
            "",
            "注: 决策效用（A）与计算效率价值（B）分别判定——A 已由独立 V_eval 评价支持；"
            "B 由独占 GPU 计时补测单独核验。",
            "",
        ])
    else:
        lines.append("")
    lines.extend([
        "## Level 3 决策效用汇总（正式评价状态，V_eval 实际收益）",
        "",
        "| split | H | 方法 | mean Gain | mean Regret | harmful rate | top-1 accuracy | mean Spearman(within state) | direction accuracy | MAE(pred vs actual) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, block in sorted(agg.items()):
        for m in ("random", "alignment", "B", "C"):
            if m not in block:
                continue
            row = block[m]
            lines.append(
                f"| {block['_split']} | {block['H']} | {METHOD_NAMES_ZH[m]} | "
                f"{_fmt(row['mean_gain'])} | "
                f"{_fmt(row['mean_regret'])} | {_fmt(row['harmful_rate'])} | "
                f"{_fmt(row['top1_accuracy'])} | {_fmt(row['mean_spearman'])} | "
                f"{_fmt(row['direction_accuracy'])} | {_fmt(row['MAE'])} |"
            )
    lines.extend([
        "",
        "## Level 1 局部导数（中心有限差分验证, V_score）",
        "",
        "见 level1_local_derivative.csv；hhat vs h_FD 的 Pearson/MAE/RMSE/relative error/sign accuracy",
        "按 split×H 池化（level1_aggregate，指标复用冻结 a0c 定义）；Spearman 只在单个",
        "state 内计算（level1_per_state_spearman.csv），未跨 state 池化。",
        "",
    ])
    for key, block in sorted(summary.get("level1_aggregate", {}).items()):
        lines.append(f"### Level 1 池化 [{key}]")
        lines.append("")
        lines.append("| 方法 | n | Pearson | MAE | RMSE | mean rel err | sign acc |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for m in ("alignment", "B", "C"):
            row = block.get(m) or {}
            if row.get("n", 0) == 0:
                continue
            lines.append(
                f"| {METHOD_NAMES_ZH[m]} | {row['n']} | {_fmt(row.get('pearson'))} | "
                f"{_fmt(row.get('mae'))} | {_fmt(row.get('rmse'))} | "
                f"{_fmt(row.get('mean_relative_error'))} | {_fmt(row.get('sign_accuracy'))} |"
            )
        lines.append("")
    lines.extend([
        "## Level 1b 有限正向动作（V_eval）",
        "",
        "见 level1_finite_action.csv；actual ΔL vs predicted ΔL = lambda·hhat（r_H 为导数约定，已乘 lambda）。",
        "level1b_aggregate 按 split×H 汇总各方法的 MAE/符号一致率/非线性残差。",
        "",
    ])
    if summary.get("cost_rows"):
        lines.extend([
            "## 成本核算",
            "",
            "见 cost_breakdown.csv；含 direct replay 与 shared-baseline tangent 在 "
            "M∈{1,2,4} 的总成本/摊销成本与 break-even M。",
            "",
        ])
    elif summary.get("cost_audit"):
        ca = summary["cost_audit"]
        lines.extend([
            "## 成本核算（§12 cost-only benchmark）",
            "",
            f"环境: {ca['env']}；独占性: {ca['exclusivity']}；"
            f"目标状态 {ca['targets']} × 8 streams（n={ca['n_samples']} stream 样本）。",
            "",
            "计时方法: 每组件独立计时键 + 前后 cuda.synchronize + perf_counter；"
            "分母一律实际 tick 数；分支引擎与切线 pass 分离 timers；"
            "主口径 median[IQR]（明细 cost_timing_raw.csv / cost_summary.json）。",
            "",
            "### 实测组件（median s）",
            "",
            "| 组件 | median | IQR | n |",
            "|---|---:|---:|---:|",
        ])
        for row in ca["components"]:
            lines.append(
                f"| {row['name']} | {_fmt(row['median'])} | "
                f"{_fmt(row.get('iqr')) if row.get('iqr') is not None else '—'} | {row['n']} |"
            )
        lines.extend([
            "",
            "### 成本表（M∈{1,2,4}；外推行已标注）",
            "",
            "| H | M | 方法 | total s | s/候选 | vs direct speedup |",
            "|---:|---:|---|---:|---:|---:|",
        ])
        for row in ca["cost_table"]:
            lines.append(
                f"| {row['H']} | {row['M']} | {row['method']} | {_fmt(row['total_s'])} | "
                f"{_fmt(row['per_candidate_s'])} | {_fmt(row.get('speedup'))} |"
            )
        lines.extend(["", f"break-even M*: {ca['breakeven']}", ""])
    else:
        lines.extend([
            "## 成本核算",
            "",
            "**待补测**：主进程在写完原始行 CSV 后、聚合阶段崩溃（计时器累计随进程丢失），"
            "成本核算需一次确定性的计时补测（只重测各操作 GPU 时间，不触碰已写盘的"
            "决策结果数据；补测完成前本章不给出结论，§14 准则 5 暂不判定）。",
            "",
        ])
    if summary.get("statistical_audit"):
        sa = summary["statistical_audit"]
        lines.extend([
            "## 统计审计（分层）",
            "",
            f"hierarchical bootstrap {sa['n_boot']} 次（seed→state→stream 有放回），"
            "效应 delta = C − baseline；详见 statistical_audit/STATISTICAL_AUDIT.md。",
            "",
            "| H | 对比 | 指标 | boot meanΔ | 95%CI | P(有利方向) | 结论 |",
            "|---:|---|---|---:|---|---:|---|",
        ])
        for row in sa["rows"]:
            lines.append(
                f"| {row['H']} | {row['comparison']} | {row['metric']} | "
                f"{_fmt(row['mean'])} | [{_fmt(row['ci_lo'])}, {_fmt(row['ci_hi'])}] | "
                f"{_fmt(row['p_beneficial'])} | {row['verdict']} |"
            )
        lines.append("")
    lines.extend([
        "## 数据组",
        "",
        f"定义哈希: `{summary['data_group_definition_sha256']}`（4 组，确定性 id+内容哈希分桶，",
        "仅用于决策管线验证，不代表语义数据域；详见 data_group_definition.json）。",
        f"validation split 哈希: `{summary['validation_split_sha256']}`（V_score/V_eval 不重叠）。",
        "",
        "## 限制",
        "",
        "- CONDITIONAL_LOOKAHEAD_LIMITATION: " + summary["conditional_lookahead_limitation"],
        "- " + summary["math_sdpa_hvp_caveat"],
        "- tangent 是零幅光滑对象；有限 lambda 动作可能跨 clip/activation 边界（Level 1b 已测非线性残差）。",
        "- j=0（不干预）始终保留在候选集中；任何动作必须真正优于不干预才值得选。",
        "",
    ])
    return "\n".join(lines)


def json_dumps(payload: Mapping) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
