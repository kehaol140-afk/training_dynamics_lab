"""GTTR-R2B 桥接 v1 —— R1 封存解读取、k2 参考调度、检测与首步动作（纯函数）。

协议：FR_GTTR_R2B_Solved_Update_To_Training_Bridge_v1.md §2–§5、§8。

- 只读 R1 restricted_solutions.csv 的 primary r=0.05 行（全精度，不抄截图）；
  不取 half / 无约束 Schur / 一阶伴随解；NA 解报错。
- eta_ref(gm) = 2*old_get_lr(gm,256,5e-4)，gm>256 时恒 1e-4（已声明 tail）。
- 首步动作 (x,y)：eta_new=eta_ref*(1+x)，alpha_new=y/(1+x)；
  y==0 -> 原生 AdamW 路径（含 BASE x=0）；y>0 -> qh RAW 读出（无 norm
  matching，不回写 moments）。
- 正式检测网格 h=0,16,...,256；连续两次 <=ell 的第二时点为 h_detect；
  h1/h4 不参与检测。<= 为带等号比较。

本模块不拟合、不重求解：verify_solutions_against_coefficients 仅用 r1m
的受限求解器对封存 b/K 做一致性复算（同一数学，非新拟合）。
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

R_PRIMARY = 0.05
BASE_LR = 5e-4
TOTAL_MICRO = 256
TAIL_REF = 1e-4           # 2 * 5e-5
K_REF = 2.0
DETECTION_GRID = tuple(range(0, 257, 16))


def old_get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def eta_base(gm: int) -> float:
    if gm <= TOTAL_MICRO:
        return old_get_lr(int(gm), TOTAL_MICRO, BASE_LR)
    return BASE_LR * 0.1  # 5e-5


def eta_ref(gm: int) -> float:
    """参考倍率 k=2 的端到端学习率（含 tail）。"""
    return K_REF * eta_base(gm)


def read_primary_solutions(sol_csv):
    """读 R1 restricted_solutions.csv 的 primary r=0.05 行。

    返回 {(seed,start,k_str): {"LR": (x, 0.0), "2D": (x2, y2)}}，全精度。
    NA/缺失 -> ValueError（不静默替换）。
    """
    out = {}
    with Path(sol_csv).open() as f:
        for r in csv.DictReader(f):
            if float(r["r"]) != R_PRIMARY:
                continue
            key = (r["seed"], r["start"], r["k"])
            if r["x"] in ("NA", "") or r["y"] in ("NA", ""):
                raise ValueError(f"NA solution row: {r}")
            out.setdefault(key, {})[r["solution"]] = (float(r["x"]),
                                                      float(r["y"]))
    for key, sols in out.items():
        for name in ("LR", "2D"):
            if name not in sols:
                raise ValueError(f"missing {name} solution for {key}")
        if sols["LR"][1] != 0.0:
            raise ValueError(f"LR solution y!=0 for {key}")
    return out


def verify_solutions_against_coefficients(sols, model_csv, tol=1e-9):
    """用封存 b/K（symmetrized）经 r1m 受限求解器复算并与 CSV 解比对。

    一致性检查，不构成新拟合。返回 {key: {"ok": bool, "recomputed": ...}}。
    """
    from fr import r1_objective_identification_v01 as r1m
    rows = {}
    with Path(model_csv).open() as f:
        for r in csv.DictReader(f):
            rows[(r["seed"], r["start"], r["k"])] = r
    report = {}
    for key, sols in sols.items():
        m = rows[key]
        b = [float(m["b1_mean"]), float(m["b2_mean"])]
        sym = r1m.symmetrized_K(float(m["k12_direct_mean"]),
                                float(m["k21_direct_mean"]),
                                float(m["k11_mean"]), float(m["k22_mean"]))
        if not sym["ok"]:
            report[key] = {"ok": False, "reason": "asym"}
            continue
        lr = r1m.solve_lr_only(b, sym["K"], R_PRIMARY)
        s2 = r1m.solve_rectangle(b, sym["K"], R_PRIMARY, R_PRIMARY)
        ok = (abs(lr["x"] - sols["LR"][0]) <= tol and
              abs(s2["x"] - sols["2D"][0]) <= tol and
              abs(s2["y"] - sols["2D"][1]) <= tol)
        report[key] = {"ok": ok, "x_LR": lr["x"], "x_2D": s2["x"],
                       "y_2D": s2["y"]}
    return report


def first_step_plan(x, y, gm):
    """首步动作 -> (eta_new, alpha_new, use_native)。

    use_native=True 走原生 AdamW（BASE x=0 与 LR-only y=0）；
    use_native=False 走 qh RAW 读出（y>0）。
    """
    eta = eta_ref(gm)
    if not (x > -1.0):
        raise ValueError(f"x must be > -1, got {x}")
    eta_new = eta * (1.0 + float(x))
    alpha_new = float(y) / (1.0 + float(x))
    use_native = (float(y) == 0.0)
    if use_native:
        alpha_new = 0.0
    return eta_new, alpha_new, use_native


def detect_target(grid_means, ell):
    """正式检测：grid_means 为 {h: mean}（须只含 h=0,16,...,256）。

    返回 dict(h_detect 或 None, status, pair)。连续两次 <=ell 的第二时点。
    """
    hs = sorted(grid_means)
    for h in hs:
        if h not in DETECTION_GRID:
            raise ValueError(f"h={h} 不在正式检测网格")
    prev = None
    for h in hs:
        if grid_means[h] <= ell:
            if prev is not None and prev <= ell:
                return {"h_detect": h, "status": "DETECTED",
                        "pair": (hs[hs.index(h) - 1], h)}
            prev = grid_means[h]
            prev_h = h
        else:
            prev = None
    return {"h_detect": None, "status": "NOT_REACHED", "pair": None}


def block_proxy_paired_bootstrap(row_sums_a, row_counts, row_sums_b,
                                 n_draws=1000, block=64, seed=2026092402):
    """冻结 64 连续行 BLOCK_PROXY paired bootstrap（monitor 数据级不确定性）。

    同一组 block 索引配对用于两臂；返回 (mean_diff, ci_lo, ci_hi)。
    mean_diff = (Σsum_a-Σsum_b)/Σcount 按重抽 block 重建。
    """
    import numpy as np
    sa = np.asarray(row_sums_a, dtype=np.float64)
    sb = np.asarray(row_sums_b, dtype=np.float64)
    cnt = np.asarray(row_counts, dtype=np.int64)
    n = len(sa)
    if n % block != 0:
        raise ValueError("rows not divisible by block")
    nb = n // block
    sa_b = sa.reshape(nb, block).sum(axis=1)
    sb_b = sb.reshape(nb, block).sum(axis=1)
    cnt_b = cnt.reshape(nb, block).sum(axis=1)
    mean_diff = float((sa.sum() - sb.sum()) / cnt.sum())
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(int(n_draws)):
        idx = rng.integers(0, nb, nb)
        diffs.append((sa_b[idx].sum() - sb_b[idx].sum()) / cnt_b[idx].sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return mean_diff, float(lo), float(hi)


def condition_stream_label(seed, start, stream_idx):
    return f"s{seed}_t{start}_st{stream_idx}"
