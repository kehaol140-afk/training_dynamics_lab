#!/usr/bin/env python3
"""FR_A0H-1 GPU dev equivalence check: reverse adjoint vs frozen forward tangent.

协议：docs/FR_A0H_PROTOCOL.md（§2 容差与 §3 清单在写本脚本前已冻结）。

每个 dev 状态 × stream 0：
  1. forward-C：冻结 ``a0g._run_tangent_pass``（hvp_direction_mode="pre"）→ hhat_fwd。
  2. joint native pass：同一 RNG/快照纪律复演 baseline rollout，逐步快照 native 量，
     同 runner 内用冻结积木（clip_jvp / adamw_tangent_step / _hvp）重算前向切线
     （hhat_fwd_joint，供确定性自校验），再对每个 H∈{1,4,8} 做反向伴随行走
     （H−1 次快照重算 HVP，与候选数 M 无关），得 hhat_adj（C 反馈 + B 无反馈）。
  3. 比较：逐候选容差 atol=1e-6 / rtol=1e-4（冻结值）；排序、no-op 差（恒 0）、
     选择（argmin，tie 取小 j）必须一致；快照重算自校验与 HVP 代理对称性残差
     作为诊断报告（不事后放宽容差）。

只读输入：A0G 冻结制品 / v07 bank / dev manifests；输出只写 outputs/a0h_adjoint_cost_v01/。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RESEARCH_ROOT.parents[0]
SRC_ROOT = RESEARCH_ROOT / "src"
for p in (str(SRC_ROOT), str(RESEARCH_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from fr.a0c_adamw_response_propagation_v01 import (  # noqa: E402
    adamw_tangent_step,
    clip_jvp,
    project_tangent,
)
from fr.a0h_adjoint_v01 import (  # noqa: E402
    adamw_tangent_step_adjoint,
    clip_jvp_transpose,
)
from fr.full_state import load_full_training_state  # noqa: E402
from fr.hashing import stable_hash  # noqa: E402
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch  # noqa: E402
from fr.native_adamw_guard import assert_native_adamw  # noqa: E402
from fr.native_schedule import set_native_micro_lr  # noqa: E402
from fr.pilot_partition import load_partition  # noqa: E402
from fr.rng import capture_rng_state, restore_rng_state  # noqa: E402
from fr.stream_rng import seed_future_stream  # noqa: E402

import a0g_decision_utility_v01 as a0g  # noqa: E402

# 冻结容差（docs/FR_A0H_PROTOCOL.md §2，先于任何 GPU 观测固定）
ATOL = 1e-6
RTOL = 1e-4
# 冻结 Rule B 全局 lambda（offline_audit 确认 0.05；排序不受 lambda 影响）
FROZEN_LAMBDA = 0.05
# 快照重算自校验的诊断告警阈值（非等价容差；仅用于定位后端/快照问题）
SNAPSHOT_RECHECK_WARN = 1e-3
SELF_CHECK_SEED = 77070001
OUTPUT_DIR = "outputs/a0h_adjoint_cost_v01"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _zeros_like_named(named):
    return {name: p.detach().new_zeros(p.shape) for name, p in named}


def _snapshot_grads(named):
    return {
        name: (p.detach().new_zeros(p.shape) if p.grad is None else p.grad.detach().clone())
        for name, p in named
    }


def _clone_mapping(mapping):
    return {k: v.detach().clone() for k, v in mapping.items()}


def _mapping_norm(mapping) -> float:
    import torch

    total = torch.zeros((), dtype=torch.float64)
    for value in mapping.values():
        total += torch.sum(value.detach().to(torch.float64).cpu().square())
    return float(torch.sqrt(total))


def _difference_norm(left, right) -> float:
    import torch

    _require(set(left) == set(right), "tensor mapping names differ")
    total = torch.zeros((), dtype=torch.float64)
    for name in left:
        a = left[name].detach()
        b = right[name].detach().to(device=a.device, dtype=a.dtype)
        total += torch.sum((a - b).to(torch.float64).cpu().square())
    return float(torch.sqrt(total))


def _optimizer_moments(named, optimizer):
    exp_avg = {}
    exp_avg_sq = {}
    for name, p in named:
        st = optimizer.state[p]
        exp_avg[name] = st["exp_avg"].detach().clone()
        exp_avg_sq[name] = st["exp_avg_sq"].detach().clone()
    return exp_avg, exp_avg_sq


def _clip_transpose(*, native_gradient, b, grad_clip, preclip_norm):
    return clip_jvp_transpose(
        gradient_by_name=native_gradient,
        cotangent_by_name=b,
        max_norm=float(grad_clip),
        active=bool(preclip_norm > float(grad_clip)),
        clip_before_norm=float(preclip_norm),
    )["clip_jvp_transpose"]


def _run_joint_pass(*, torch, model, optimizer, scaler, autocast_ctx, dataset, named,
                    checkpoint_path, batches, stream_seed, cand_batches_by_j,
                    val_score_tensors, accumulation_steps, device, grad_clip,
                    base_lr, global_micro_step, total_micro_steps, get_lr, timers,
                    hvp_fp32=False):
    """联合 native pass：冻结前向切线 + 逐步快照 + 反向伴随（C 与 B）。

    前向段逐行镜像冻结 ``a0g._run_tangent_pass``（hvp_direction_mode="pre"），
    native 轨迹与冻结 runner 的位级一致性由 frozen-vs-joint 确定性比较验证。
    ``hvp_fp32=True`` 仅为诊断探针：HVP 重算不使用 bf16 autocast（其余与
    冻结管线完全一致），用于度量冻结 bf16 HVP 代理自身的噪声地板。
    """
    hvp_ctx = nullcontext() if hvp_fp32 else autocast_ctx
    load_full_training_state(checkpoint_path, model, optimizer, scaler, restore_rng=False)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seed_future_stream(int(stream_seed))

    zero = _zeros_like_named(named)
    b_theta = {j: _clone_mapping(zero) for j in cand_batches_by_j}
    b_m = {j: _clone_mapping(zero) for j in cand_batches_by_j}
    b_v = {j: _clone_mapping(zero) for j in cand_batches_by_j}
    c_theta = {j: _clone_mapping(zero) for j in cand_batches_by_j}
    c_m = {j: _clone_mapping(zero) for j in cand_batches_by_j}
    c_v = {j: _clone_mapping(zero) for j in cand_batches_by_j}

    max_h = max(a0g.HORIZONS)
    g_by_j = {}
    a_by_j = {}
    hhat_fwd = {m: {j: {} for j in cand_batches_by_j} for m in ("B", "C")}
    r_H_B = {j: {} for j in cand_batches_by_j}
    r_H_C = {j: {} for j in cand_batches_by_j}
    v_score_grad = {}
    snapshots = {}          # horizon -> native snapshot for the adjoint walk
    hvp_by_j = {}           # horizon -> j -> HVP mapping (forward, for recheck)
    hvp_dir_by_j = {}       # horizon -> j -> direction copy used for the HVP
    clip_trace = []
    hvp_relative_gaps = []
    pg0 = optimizer.param_groups[0]
    for pg in optimizer.param_groups:
        _require(
            (pg["betas"], pg["eps"], pg["weight_decay"])
            == (pg0["betas"], pg0["eps"], pg0["weight_decay"]),
            "heterogeneous optimizer hyperparameters",
        )
    hyper = {
        "weight_decay": float(pg0["weight_decay"]),
        "beta1": float(pg0["betas"][0]),
        "beta2": float(pg0["betas"][1]),
        "adam_eps": float(pg0["eps"]),
    }

    for horizon in range(1, max_h + 1):
        theta_start = _clone_mapping(raw_model(model).state_dict())
        batch_tensors = []
        current_lr = None
        rng_before_native_forward = None
        for accumulation_index in range(accumulation_steps):
            future_offset = (horizon - 1) * accumulation_steps + accumulation_index + 1
            current_lr = set_native_micro_lr(
                optimizer, get_lr,
                int(global_micro_step) + future_offset,
                int(total_micro_steps), float(base_lr),
            )
            batch_index = batches[(horizon - 1) * accumulation_steps + accumulation_index]
            x, y = stack_batch(dataset, batch_index, device)
            batch_tensors.append((x, y))
            if horizon > 1 and accumulation_index == 0:
                rng_before_native_forward = capture_rng_state()
            with autocast_ctx:
                result = model(x, labels=y)
                loss = (result.loss + result.aux_loss) / accumulation_steps
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        native_gradient = _snapshot_grads(named)

        if horizon == 1:
            for j, cand_batches in cand_batches_by_j.items():
                cand_tensors = a0g._stack_all(dataset, cand_batches, device)
                rng_snap = capture_rng_state()
                g_by_j[j] = a0g._grad_on_batches(
                    torch, model, named, cand_tensors, autocast_ctx, accumulation_steps,
                )
                restore_rng_state(rng_snap)
                del cand_tensors
            for j in cand_batches_by_j:
                a_by_j[j] = {
                    name: g_by_j[j][name] - native_gradient[name] for name, _ in named
                }

        native_preclip = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            .detach().cpu()
        )
        native_clipped_gradient = _snapshot_grads(named)
        clip_probe = clip_jvp(
            gradient_by_name=native_gradient,
            tangent_gradient_by_name=zero,
            max_norm=float(grad_clip),
        )
        if abs(native_preclip - float(clip_probe["clip_before_norm"])) > 1e-5:
            raise RuntimeError(
                "analytic/native preclip norm mismatch: "
                f"native={native_preclip!r} analytic={clip_probe['clip_before_norm']!r} "
                f"horizon={horizon}"
            )

        hvp_by_j[horizon] = {}
        hvp_dir_by_j[horizon] = {}
        if horizon > 1:
            for j in cand_batches_by_j:
                hvp_dir_by_j[horizon][j] = _clone_mapping(c_theta[j])
                rng_after_native_gradient = capture_rng_state()
                restore_rng_state(rng_before_native_forward)
                hvp_j, hvp_first_grad = a0g._hvp(
                    torch, model, named, batch_tensors, c_theta[j], hvp_ctx,
                    accumulation_steps,
                )
                restore_rng_state(rng_after_native_gradient)
                hvp_by_j[horizon][j] = hvp_j
                gap = _difference_norm(hvp_first_grad, native_gradient)
                hvp_relative_gaps.append(
                    float(gap / max(_mapping_norm(native_gradient), 1e-30))
                )

        c_step_after = None
        for j in cand_batches_by_j:
            tB = a_by_j[j] if horizon == 1 else zero
            tC = a_by_j[j] if horizon == 1 else hvp_by_j[horizon][j]
            b_clip = clip_jvp(
                gradient_by_name=native_gradient, tangent_gradient_by_name=tB,
                max_norm=float(grad_clip),
            )
            c_clip = clip_jvp(
                gradient_by_name=native_gradient, tangent_gradient_by_name=tC,
                max_norm=float(grad_clip),
            )
            b_next = adamw_tangent_step(
                named_parameters=named, optimizer=optimizer,
                clipped_gradient_by_name=native_clipped_gradient,
                tangent_clipped_gradient_by_name=b_clip["clip_jvp"],
                tangent_theta_by_name=b_theta[j],
                tangent_m_by_name=b_m[j],
                tangent_v_by_name=b_v[j],
            )
            c_next = adamw_tangent_step(
                named_parameters=named, optimizer=optimizer,
                clipped_gradient_by_name=native_clipped_gradient,
                tangent_clipped_gradient_by_name=c_clip["clip_jvp"],
                tangent_theta_by_name=c_theta[j],
                tangent_m_by_name=c_m[j],
                tangent_v_by_name=c_v[j],
            )
            b_theta[j], b_m[j], b_v[j] = (
                b_next["tangent_theta"], b_next["tangent_m"], b_next["tangent_v"],
            )
            c_theta[j], c_m[j], c_v[j] = (
                c_next["tangent_theta"], c_next["tangent_m"], c_next["tangent_v"],
            )
            if c_step_after is None:
                c_step_after = float(c_next["optimizer_step_after"])
                _require(
                    abs(float(b_next["optimizer_step_after"]) - c_step_after) < 1e-9,
                    "B/C step_after mismatch",
                )
            if horizon in a0g.HORIZONS:
                r_H_B[j][str(horizon)] = _clone_mapping(b_theta[j])
                r_H_C[j][str(horizon)] = _clone_mapping(c_theta[j])

        # ---- 步前快照（scaler.step 之前；伴随侧从此处取 native 量） ----
        exp_avg, exp_avg_sq = _optimizer_moments(named, optimizer)
        snapshots[horizon] = {
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
            "clipped_gradient": _clone_mapping(native_clipped_gradient),
            "native_gradient": _clone_mapping(native_gradient),
            "preclip_norm": float(native_preclip),
            "lr": float(current_lr),
            "step_after": float(c_step_after),
            "batch_tensors": batch_tensors,
            "rng_before_native_forward": rng_before_native_forward,
            "theta_start": theta_start,
            **hyper,
        }
        clip_trace.append({
            "horizon": horizon,
            "clip_active": bool(native_preclip > float(grad_clip)),
            "preclip_norm": float(native_preclip),
            "lr": float(current_lr),
            "step_after": float(c_step_after),
        })

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if horizon in a0g.HORIZONS:
            rng_snap = capture_rng_state()
            v_score_grad[str(horizon)] = a0g._grad_on_batches(
                torch, model, named, val_score_tensors, autocast_ctx,
                len(val_score_tensors),
            )
            restore_rng_state(rng_snap)

    timers.snapshot_peak("peak_joint_pass")
    for horizon in a0g.HORIZONS:
        hk = str(horizon)
        v_h = v_score_grad[hk]
        for j in cand_batches_by_j:
            hhat_fwd["B"][j][hk] = project_tangent(v_h, r_H_B[j][hk])
            hhat_fwd["C"][j][hk] = project_tangent(v_h, r_H_C[j][hk])

    # ---- 反向伴随行走（对每个 H；C 含 K^T J^T 反馈，B 无反馈） ----
    def _walk(horizon, feedback, do_recheck, store_w=False):
        p_th = _clone_mapping(v_score_grad[str(horizon)])
        p_m = _clone_mapping(zero)
        p_v = _clone_mapping(zero)
        recheck = None
        w_by_k = {}
        for k in range(horizon, 0, -1):
            snap = snapshots[k]
            adj = adamw_tangent_step_adjoint(
                named_parameters=named, optimizer=optimizer,
                snapshot=snap,
                cotangent_theta_by_name=p_th,
                cotangent_m_by_name=p_m,
                cotangent_v_by_name=p_v,
                feedback_cotangent_by_name=None,
            )
            b_k = adj["cotangent_dc"]
            w_k = _clip_transpose(
                native_gradient=snap["native_gradient"], b=b_k,
                grad_clip=grad_clip, preclip_norm=snap["preclip_norm"],
            )
            if store_w and k >= 2:
                w_by_k[k] = _clone_mapping(w_k)
            p_th = adj["cotangent_theta"]
            p_m = adj["cotangent_m"]
            p_v = adj["cotangent_v"]
            if k >= 2 and feedback:
                rng_now = capture_rng_state()
                restore_rng_state(snap["rng_before_native_forward"])
                raw_model(model).load_state_dict(snap["theta_start"])
                if do_recheck and recheck is None:
                    hvp_check, _ = a0g._hvp(
                        torch, model, named, snap["batch_tensors"],
                        hvp_dir_by_j[k][min(hvp_dir_by_j[k])], hvp_ctx,
                        accumulation_steps,
                    )
                    ref = hvp_by_j[k][min(hvp_by_j[k])]
                    recheck = {
                        "horizon": k,
                        "relative_gap": float(
                            _difference_norm(hvp_check, ref)
                            / max(_mapping_norm(ref), 1e-30)
                        ),
                        "warn_threshold": SNAPSHOT_RECHECK_WARN,
                    }
                    # 自校验消耗了 RNG；恢复 rng_before 再算本步伴随 HVP，
                    # 与冻结前向纪律（HVP forward 看到同一 dropout mask）一致
                    restore_rng_state(snap["rng_before_native_forward"])
                hvp_u, _ = a0g._hvp(
                    torch, model, named, snap["batch_tensors"], w_k, hvp_ctx,
                    accumulation_steps,
                )
                restore_rng_state(rng_now)
                p_th = {name: p_th[name] + hvp_u[name] for name, _ in named}
        snap1 = snapshots[1]
        w_0 = _clip_transpose(
            native_gradient=snap1["native_gradient"], b=b_k,
            grad_clip=grad_clip, preclip_norm=snap1["preclip_norm"],
        )
        return {
            "w_0": w_0,
            "recheck": recheck,
            "w_by_k": w_by_k,
            "hhat": {j: project_tangent(w_0, a_by_j[j]) for j in cand_batches_by_j},
        }

    adjoint = {}
    # 自校验放在第一个含未来步 HVP 的行走（H=4）里执行（H=1 无 HVP 可校验）
    first_feedback_h = sorted(h for h in a0g.HORIZONS if h >= 2)[0]
    max_h_walk = max(a0g.HORIZONS)
    for horizon in a0g.HORIZONS:
        adjoint[("C", horizon)] = _walk(
            horizon, feedback=True,
            do_recheck=(horizon == first_feedback_h),
            store_w=(horizon == max_h_walk),
        )
        adjoint[("B", horizon)] = _walk(horizon, feedback=False, do_recheck=False)

    # 前向段终点哈希（与冻结 pass 的 final_model_hash 同口径：前向循环结束、
    # 伴随行走/诊断尚未改动模型状态之前）
    final_model_hash = stable_hash(raw_model(model).state_dict())

    # ---- 转置一致性探针（诊断，非门控）：真实尺度方向上测 HVP 代理的
    # <x,Hy> vs <Hx,y> 不对称性，定位 H=8 残差的来源 ----
    def _symmetry_probe(k):
        x = hvp_dir_by_j[k][min(hvp_dir_by_j[k])]       # 前向 HVP 方向（真实切线尺度）
        y = adjoint[("C", max_h_walk)]["w_by_k"][k]     # 伴随行走方向 J_k^T b_k
        snap_k = snapshots[k]
        rng_now = capture_rng_state()
        restore_rng_state(snap_k["rng_before_native_forward"])
        raw_model(model).load_state_dict(snap_k["theta_start"])
        hx = a0g._hvp(torch, model, named, snap_k["batch_tensors"], x, hvp_ctx,
                      accumulation_steps)[0]
        restore_rng_state(snap_k["rng_before_native_forward"])
        hy = a0g._hvp(torch, model, named, snap_k["batch_tensors"], y, hvp_ctx,
                      accumulation_steps)[0]
        restore_rng_state(rng_now)
        lhs = project_tangent(x, hy)
        rhs = project_tangent(hx, y)
        return {"horizon": k, "lhs": lhs, "rhs": rhs,
                "abs_gap": abs(lhs - rhs),
                "relative": abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)}

    transpose_probes = [_symmetry_probe(k) for k in (2, max_h_walk)]

    # ---- HVP 代理对称性残差（诊断；k=2 步起点，固定种子方向） ----
    gen = torch.Generator().manual_seed(SELF_CHECK_SEED)
    x_dir = {name: torch.randn(p.shape, generator=gen, dtype=torch.float32, device="cpu").to(device)
             for name, p in named}
    y_dir = {name: torch.randn(p.shape, generator=gen, dtype=torch.float32, device="cpu").to(device)
             for name, p in named}
    snap2 = snapshots[2]
    rng_now = capture_rng_state()
    restore_rng_state(snap2["rng_before_native_forward"])
    raw_model(model).load_state_dict(snap2["theta_start"])
    hx = a0g._hvp(torch, model, named, snap2["batch_tensors"], x_dir, hvp_ctx,
                  accumulation_steps)[0]
    restore_rng_state(snap2["rng_before_native_forward"])
    hy = a0g._hvp(torch, model, named, snap2["batch_tensors"], y_dir, hvp_ctx,
                  accumulation_steps)[0]
    restore_rng_state(rng_now)
    lhs = project_tangent(x_dir, hy)
    rhs = project_tangent(hx, y_dir)

    return {
        "hhat_fwd": hhat_fwd,
        "adjoint": {f"{m}|{h}": v for (m, h), v in adjoint.items()},
        "clip_trace": clip_trace,
        "max_hvp_gradient_relative_gap": max(hvp_relative_gaps, default=0.0),
        "hvp_proxy_symmetry": {
            "lhs": lhs, "rhs": rhs,
            "abs_gap": abs(lhs - rhs),
            "relative": abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30),
        },
        "transpose_probes": transpose_probes,
        "final_model_hash": final_model_hash,
    }


def _compare_hhat(*, fwd_by_j, adj_by_j):
    rows = []
    ok = True
    for j in sorted(fwd_by_j):
        fwd = float(fwd_by_j[j])
        adj = float(adj_by_j[j])
        diff = abs(adj - fwd)
        tol = ATOL + RTOL * abs(fwd)
        rows.append({
            "j": int(j), "hhat_fwd": fwd, "hhat_adj": adj,
            "abs_diff": diff, "tol": tol, "pass": bool(diff <= tol),
        })
        ok = ok and diff <= tol
    return rows, ok


def _ranking_and_selection(rows_by_j):
    """rows_by_j: {j: score}；no-op j=0 分数恒 0；排序升序，tie 取小 j。"""
    items = [(0, 0.0)] + [(int(j), float(v)) for j, v in rows_by_j.items()]
    order = sorted(items, key=lambda kv: (kv[1], kv[0]))
    ranking = [j for j, _ in order]
    return ranking, ranking[0]


def compare_unit(*, state_label, stream_id, frozen, joint):
    report = {
        "state": state_label,
        "stream_id": int(stream_id),
        "lambda": FROZEN_LAMBDA,
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "forward_determinism": {},
        "methods": {},
        "clip_trace": joint["clip_trace"],
        "max_hvp_gradient_relative_gap": joint["max_hvp_gradient_relative_gap"],
        "hvp_proxy_symmetry_residual": joint["hvp_proxy_symmetry"],
        "transpose_probes": joint["transpose_probes"],
    }
    det_ok = True
    det_max = {"B": 0.0, "C": 0.0}
    for m in ("B", "C"):
        for j in frozen["hhat"][m]:
            for hk in frozen["hhat"][m][j]:
                gap = abs(
                    float(frozen["hhat"][m][j][hk])
                    - float(joint["hhat_fwd"][m][j][hk])
                )
                det_max[m] = max(det_max[m], gap)
                det_ok = det_ok and gap == 0.0
    report["forward_determinism"] = {
        "max_abs_gap_B": det_max["B"],
        "max_abs_gap_C": det_max["C"],
        "bit_identical": bool(det_max["B"] == 0.0 and det_max["C"] == 0.0),
        "final_model_hash_match": bool(
            frozen["final_model_hash"] == joint["final_model_hash"]
        ),
    }
    overall = det_ok and report["forward_determinism"]["final_model_hash_match"]

    for m in ("C", "B"):
        for hk in sorted({hk for j in frozen["hhat"][m] for hk in frozen["hhat"][m][j]},
                         key=int):
            fwd_by_j = {j: frozen["hhat"][m][j][hk] for j in frozen["hhat"][m]}
            adj_by_j = {
                j: joint["adjoint"][f"{m}|{hk}"]["hhat"][j] for j in frozen["hhat"][m]
            }
            rows, ok = _compare_hhat(fwd_by_j=fwd_by_j, adj_by_j=adj_by_j)
            max_abs_hhat = max(abs(float(v)) for v in fwd_by_j.values())
            near_tie_threshold = 2.0 * (ATOL + RTOL * max_abs_hhat)
            scores_fwd = {j: FROZEN_LAMBDA * float(v) for j, v in fwd_by_j.items()}
            scores_adj = {j: FROZEN_LAMBDA * float(v) for j, v in adj_by_j.items()}
            ranking_fwd, selection_fwd = _ranking_and_selection(scores_fwd)
            ranking_adj, selection_adj = _ranking_and_selection(scores_adj)
            ranking_match = ranking_fwd == ranking_adj
            # 近并列：所有排序反转的候选对，其冻结前向分数差都在近并列阈值内
            # （阈值先于 GPU 观测固定，见协议 §2）
            near_tie = False
            inversion_within_tie = True
            for j1 in scores_fwd:
                for j2 in scores_fwd:
                    if j1 >= j2:
                        continue
                    d_fwd = scores_fwd[j1] - scores_fwd[j2]
                    d_adj = scores_adj[j1] - scores_adj[j2]
                    inverted = (d_fwd == 0.0) != (d_adj == 0.0) or (
                        d_fwd != 0.0 and d_adj != 0.0 and (d_fwd < 0) != (d_adj < 0)
                    )
                    if inverted:
                        near_tie = True
                        if abs(d_fwd) > near_tie_threshold:
                            inversion_within_tie = False
            consistent = ranking_match or (near_tie and inversion_within_tie)
            selection_score_close = abs(
                scores_fwd[selection_fwd] - scores_fwd[selection_adj]
            ) <= near_tie_threshold
            selection_ok = (selection_fwd == selection_adj) or selection_score_close
            entry = {
                "method": m, "H": int(hk),
                "per_candidate": rows,
                "all_candidates_pass": ok,
                "scores_fwd": scores_fwd,
                "scores_adj": scores_adj,
                "ranking_fwd": ranking_fwd,
                "ranking_adj": ranking_adj,
                "ranking_match": bool(ranking_match),
                "ranking_consistent_incl_near_tie": bool(consistent),
                "near_tie_within_threshold": bool(near_tie and inversion_within_tie),
                "selection_fwd": int(selection_fwd),
                "selection_adj": int(selection_adj),
                "selection_match": bool(selection_fwd == selection_adj),
                "selection_ok_incl_near_tie": bool(selection_ok),
                "noop_score_fwd": 0.0,
                "noop_score_adj": 0.0,
                "noop_diff_exact_zero": True,
                "near_tie_threshold": near_tie_threshold,
                "pass": bool(ok and consistent and selection_ok),
            }
            if m == "C" and int(hk) == sorted(h for h in a0g.HORIZONS if h >= 2)[0]:
                entry["snapshot_recheck"] = joint["adjoint"][f"C|{hk}"]["recheck"]
            report["methods"][f"{m}|{hk}"] = entry
            overall = overall and entry["pass"]
    report["unit_pass"] = bool(overall)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-states", type=int, default=0,
                    help="debug: only first N dev states")
    ap.add_argument("--budget-seconds", type=float, default=1500.0,
                    help="GPU wall-clock budget for this script (30-min round cap share)")
    ap.add_argument("--probe-fp32-hvp", action="store_true",
                    help="diagnostic: rerun joint pass with fp32 HVP recompute to measure "
                         "the frozen bf16 HVP proxy noise floor")
    args = ap.parse_args()

    wall_start = time.time()
    root = Path(args.project_root).resolve()
    out_dir = root / OUTPUT_DIR / "dev_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((root / "research/configs/e3c_confirmatory_v07.json").read_text())
    _require(cfg.get("dtype") == "bfloat16" and int(cfg.get("accumulation_steps")) == 2
             and float(cfg.get("grad_clip")) == 1.0, "config drift")
    v07_out = root / cfg["output_dir"]
    partition = load_partition(v07_out / "partition_v07.json")
    data_path = v07_out / "e3c_subset.jsonl"
    a0g_out = root / a0g.OUTPUT_DIR
    data_group_definition = json.loads(
        (a0g_out / "data_group_definition.json").read_text(encoding="utf-8"))
    validation_split = json.loads(
        (a0g_out / "validation_split.json").read_text(encoding="utf-8"))

    minimind = root / "minimind"
    if str(minimind) not in sys.path:
        sys.path.insert(0, str(minimind))

    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model

    _require(str(args.device).startswith("cuda") and torch.cuda.is_available(),
             "strict BF16 replay requires CUDA")
    mmcfg = MiniMindConfig(
        hidden_size=int(cfg["hidden_size"]), num_hidden_layers=int(cfg["num_hidden_layers"]),
        max_seq_len=int(cfg["max_seq_len"]), use_moe=False,
    )
    model, tokenizer = init_model(
        mmcfg, from_weight="none", tokenizer_path=str(minimind / "model"), device=args.device,
    )
    dataset = PretrainDataset(str(data_path), tokenizer, max_length=int(cfg["max_seq_len"]))
    autocast_ctx, scaler, _ = make_amp(torch, args.device, cfg["dtype"])
    if scaler.is_enabled():
        raise RuntimeError("bf16 replay unexpectedly enabled GradScaler")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    assert_native_adamw(optimizer)
    named = [(n, p) for n, p in raw_model(model).named_parameters() if p.requires_grad]

    def _chunk(indices):
        idx = sorted(map(int, indices))
        return [idx[i:i + a0g.EVAL_BATCH_SIZE] for i in range(0, len(idx), a0g.EVAL_BATCH_SIZE)]

    val_score_tensors = a0g._stack_all(dataset, _chunk(validation_split["v_score"]), args.device)
    val_eval_tensors = a0g._stack_all(dataset, _chunk(validation_split["v_eval"]), args.device)

    timers = a0g._Timers(torch)
    dev_ctxs = a0g._load_v07_state_contexts(root, cfg, partition, data_group_definition)
    if args.limit_states:
        dev_ctxs = dev_ctxs[:args.limit_states]

    env_info = {
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(args.device),
        "budget_seconds": args.budget_seconds,
    }

    units = []
    probes = []
    status = "PASS"
    for unit_index, ctx in enumerate(dev_ctxs):
        elapsed = time.time() - wall_start
        if elapsed > args.budget_seconds:
            status = "PARTIAL_BUDGET_EXHAUSTED"
            break
        stream = next(s for s in ctx.streams if int(s["stream_id"]) == 0)
        state_label = f"dev:{ctx.seed}/{ctx.step}"
        print(f"[{state_label}] stream={stream['stream_id']} forward-C (frozen) ...", flush=True)
        t0 = time.perf_counter()
        v_t = a0g._compute_v_t(torch, model, optimizer, scaler, ctx, val_score_tensors,
                               autocast_ctx, named)
        cand = ctx.candidate_batches(stream["stream_seed"])
        frozen = a0g._run_tangent_pass(
            torch=torch, model=model, optimizer=optimizer, scaler=scaler,
            autocast_ctx=autocast_ctx, dataset=dataset, named=named,
            checkpoint_path=ctx.checkpoint_path,
            batches=stream["batches"], stream_seed=stream["stream_seed"],
            cand_batches_by_j=cand, val_score_tensors=val_score_tensors,
            val_eval_tensors=val_eval_tensors, v_t=v_t,
            accumulation_steps=int(cfg["accumulation_steps"]), device=str(model.device),
            grad_clip=float(cfg["grad_clip"]), base_lr=float(cfg["learning_rate"]),
            global_micro_step=int(ctx.bank_meta["global_micro_step"]),
            total_micro_steps=int(ctx.bank_meta["total_micro_steps"]),
            get_lr=get_lr, timers=timers, hvp_direction_mode="pre",
        )
        t_frozen = time.perf_counter() - t0
        print(f"[{state_label}] frozen pass done in {t_frozen:.1f}s; joint pass ...", flush=True)

        t0 = time.perf_counter()
        joint = _run_joint_pass(
            torch=torch, model=model, optimizer=optimizer, scaler=scaler,
            autocast_ctx=autocast_ctx, dataset=dataset, named=named,
            checkpoint_path=ctx.checkpoint_path,
            batches=stream["batches"], stream_seed=stream["stream_seed"],
            cand_batches_by_j=cand, val_score_tensors=val_score_tensors,
            accumulation_steps=int(cfg["accumulation_steps"]), device=str(model.device),
            grad_clip=float(cfg["grad_clip"]), base_lr=float(cfg["learning_rate"]),
            global_micro_step=int(ctx.bank_meta["global_micro_step"]),
            total_micro_steps=int(ctx.bank_meta["total_micro_steps"]),
            get_lr=get_lr, timers=timers,
        )
        t_joint = time.perf_counter() - t0
        print(f"[{state_label}] joint pass done in {t_joint:.1f}s; comparing ...", flush=True)

        if args.probe_fp32_hvp:
            t0 = time.perf_counter()
            joint_fp32 = _run_joint_pass(
                torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, dataset=dataset, named=named,
                checkpoint_path=ctx.checkpoint_path,
                batches=stream["batches"], stream_seed=stream["stream_seed"],
                cand_batches_by_j=cand, val_score_tensors=val_score_tensors,
                accumulation_steps=int(cfg["accumulation_steps"]), device=str(model.device),
                grad_clip=float(cfg["grad_clip"]), base_lr=float(cfg["learning_rate"]),
                global_micro_step=int(ctx.bank_meta["global_micro_step"]),
                total_micro_steps=int(ctx.bank_meta["total_micro_steps"]),
                get_lr=get_lr, timers=timers, hvp_fp32=True,
            )
            t_probe = time.perf_counter() - t0
            det_max = 0.0
            for m in ("B", "C"):
                for j in joint["hhat_fwd"][m]:
                    for hk in joint["hhat_fwd"][m][j]:
                        det_max = max(det_max, abs(
                            float(joint["hhat_fwd"][m][j][hk])
                            - float(joint_fp32["hhat_fwd"][m][j][hk])
                        ))
            probe = {
                "state": state_label,
                "stream_id": int(stream["stream_id"]),
                "hhat_fwd_bf16": joint["hhat_fwd"],
                "hhat_fwd_fp32hvp": joint_fp32["hhat_fwd"],
                "hhat_adj_fp32": {
                    key: {"hhat": val["hhat"]} for key, val in joint_fp32["adjoint"].items()
                },
                "transpose_probes_bf16": joint["transpose_probes"],
                "transpose_probes_fp32": joint_fp32["transpose_probes"],
                "hvp_proxy_symmetry_residual_fp32": joint_fp32["hvp_proxy_symmetry"],
                # bf16 前向在 fp32-HVP 参照下的自身偏差（冻结管线噪声地板）
                "fwd_bf16_vs_fp32hvp_max_abs_gap": det_max,
                "wall_seconds": {"joint_pass_fp32": t_probe},
            }
            probe_path = out_dir / f"fp32_hvp_probe_{ctx.seed}_{ctx.step}_s{stream['stream_id']}.json"
            probe_path.write_text(json.dumps(probe, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
            probes.append(probe)
            print(f"[{state_label}] fp32-HVP probe done in {t_probe:.1f}s -> {probe_path}",
                  flush=True)

        unit = compare_unit(
            state_label=state_label, stream_id=stream["stream_id"],
            frozen={"hhat": frozen["hhat"], "final_model_hash": frozen["final_model_hash"]},
            joint=joint,
        )
        unit["wall_seconds"] = {"frozen_pass": t_frozen, "joint_pass": t_joint}
        unit["unit_index"] = unit_index
        units.append(unit)
        unit_path = out_dir / f"unit_{ctx.seed}_{ctx.step}_s{stream['stream_id']}.json"
        unit_path.write_text(json.dumps(unit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{state_label}] unit_pass={unit['unit_pass']} -> {unit_path}", flush=True)
        if not unit["unit_pass"]:
            status = "FAIL"

    summary = {
        "protocol": "FR_A0H_PROTOCOL.md §2/§3 (frozen before GPU runs)",
        "status": status if units else "MISSING_DATA",
        "units": [
            {
                "state": u["state"],
                "stream_id": u["stream_id"],
                "unit_pass": u["unit_pass"],
                "bit_identical_forward": u["forward_determinism"]["bit_identical"],
                "methods": {
                    key: {
                        "pass": e["pass"],
                        "ranking_match": e["ranking_match"],
                        "selection_match": e["selection_match"],
                        "max_abs_diff": max(r["abs_diff"] for r in e["per_candidate"]),
                    }
                    for key, e in u["methods"].items()
                },
                "snapshot_recheck": next(
                    (e.get("snapshot_recheck") for e in u["methods"].values()
                     if e.get("snapshot_recheck")), None),
                "hvp_proxy_symmetry_residual": u["hvp_proxy_symmetry_residual"],
                "wall_seconds": u["wall_seconds"],
            }
            for u in units
        ],
        "env": env_info,
        "total_wall_seconds": time.time() - wall_start,
    }
    summary_path = out_dir / "dev_check_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"STATUS = {summary['status']}")
    print(f"SUMMARY -> {summary_path}")
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
