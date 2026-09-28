"""GTTR-R1 目标系数辨识 v1 —— 控制切片坐标、目标系数与受限二次求解（纯数学模块）。

协议：FR_GTTR_R1_Model_To_Rule_Identification_v1.md §1–§2、§8(T1–T8)。

固定公式（p 为唯一参数张量，c 为 clip 后累计梯度，t 为 true step=n0+1，
eta 为含参考倍率 k 的本步学习率，wd 为 decoupled weight decay）：

    m_plus = beta1*m + (1-beta1)*c ; v_plus = beta2*v + (1-beta2)*c^2
    mbar = m_plus/(1-beta1**t)     ; vbar = v_plus/(1-beta2**t)
    P = 1/(sqrt(vbar)+eps)                        # [FR-NATIVE-ADAMW] 原生算术
    d0 = -eta*(P*mbar + wd*theta)                 # native 完整位移（含衰减）
    d1 =  eta*P*(mbar - c)                        # 当前梯度读出方向
    theta_plus = theta + d0 ; D = [d0, d1]
    theta(a) = theta_plus + x*d0 + y*d1           # [E3] theta mapping
    eta_new = eta*(1+x) ; alpha_new = y/(1+x),  x > -1

这是下式的一次更新精确重参数化（实数算术；T1 以直接读出式对照）：

    theta_new = (1-eta_new*wd)*theta
                - eta_new*P*((1-alpha_new)*mbar + alpha_new*c)

GTTR 目标系数（theta(a) 对 a 仿射，控制坐标 Hessian 即 K）：
    b = E[D^T grad V_ctrl(theta_plus)]            # 2 维
    K = E[D^T Hessian_V_ctrl(theta_plus) D]       # 2x2 对称（报告原始双向）

模型：m(a) = b^T a + 0.5 a^T K a；可行域主域 x∈[-r,r], y∈[0,r]。
y=0 是“只改 LR”。求解用矩形域二次函数完整候选枚举（角点、四边内部
极小点、可行内部驻点、原点），保留负特征值，不做 PSD 裁剪，不加 jitter。

数值纪律：REL_TOL 仅用于线性代数的数值处理（奇异/病态判定）；
原特征值/条件数/非对称残差一律照报。无法数值求解则 status=UNRESOLVED。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

REL_TOL = 1e-12          # 线性代数数值处理阈值（声明值；不影响报告原值）
K_ASYM_REL_TOL = 1e-4    # K12_direct/K21_direct 相对非对称超过此值 -> 不求解(报错信号)
COND_SKIP_INTERIOR = 1e14  # cond(K) 超过此值跳过内部驻点候选（仍报原值）


# ---------------------------------------------------------------------------
# 控制切片坐标（T1/T2 被测对象）
# ---------------------------------------------------------------------------

def control_slice(theta, c, m, v, t, beta1, beta2, eps, wd, eta):
    """从 (theta, c, m_old, v_old, t) 计算 {name: {d0,d1,P,mbar,vbar,m_plus,v_plus}}。

    [FR-NATIVE-ADAMW] 与 torch.optim.AdamW 相同的矩更新/bias 校正/P/decay
    算术；c 为 None 的参数整体跳过（原生语义：无更新、无 decay）。
    不修改任何输入张量（矩更新在新张量上进行）。
    """
    bc1 = 1.0 - float(beta1) ** int(t)
    bc2 = 1.0 - float(beta2) ** int(t)
    out = {}
    for n, th in theta.items():
        cc = c.get(n)
        if cc is None:
            continue
        mm = float(beta1) * m[n] + (1.0 - float(beta1)) * cc
        vv = float(beta2) * v[n] + (1.0 - float(beta2)) * (cc * cc)
        mbar = mm / bc1
        vbar = vv / bc2
        P = 1.0 / (vbar.sqrt() + float(eps))
        d0 = -float(eta) * (P * mbar + float(wd) * th)
        d1 = float(eta) * P * (mbar - cc)
        out[n] = {"d0": d0, "d1": d1, "P": P, "mbar": mbar, "vbar": vbar,
                  "m_plus": mm, "v_plus": vv}
    return out


def direct_readout_update(theta, c, m, v, t, beta1, beta2, eps, wd,
                          eta_new, alpha_new):
    """直接读出式参考实现：theta_new=(1-eta_n*wd)theta - eta_n*P((1-a)mbar+a*c)。

    用于 T1 与控制切片坐标的一致性对照；同样跳过 c=None 参数。
    """
    bc1 = 1.0 - float(beta1) ** int(t)
    bc2 = 1.0 - float(beta2) ** int(t)
    out = {}
    for n, th in theta.items():
        cc = c.get(n)
        if cc is None:
            out[n] = th.clone()
            continue
        mm = float(beta1) * m[n] + (1.0 - float(beta1)) * cc
        vv = float(beta2) * v[n] + (1.0 - float(beta2)) * (cc * cc)
        mbar = mm / bc1
        vbar = vv / bc2
        P = 1.0 / (vbar.sqrt() + float(eps))
        r = (1.0 - float(alpha_new)) * (P * mbar) + float(alpha_new) * (P * cc)
        out[n] = (1.0 - float(eta_new) * float(wd)) * th - float(eta_new) * r
    return out


def reparam(x, y, eta):
    """(x,y,eta) -> (eta_new, alpha_new)；x>-1。"""
    if not (x > -1.0):
        raise ValueError(f"x must be > -1, got {x}")
    return float(eta) * (1.0 + float(x)), float(y) / (1.0 + float(x))


def make_theta_plus(theta, slice_):
    """theta_plus = theta + d0（无梯度参数保持 theta）。新张量。"""
    out = {n: th.clone() for n, th in theta.items()}
    for n, s in slice_.items():
        out[n] = theta[n] + s["d0"]
    return out


def theta_at_a(theta_plus, d0, d1, x, y):
    """theta(a) = theta_plus + x*d0 + y*d1（无方向参数不动）。新张量。"""
    out = {}
    for n, tp in theta_plus.items():
        shift = tp.clone()
        if n in d0:
            shift = shift + float(x) * d0[n]["d0"] + float(y) * d0[n]["d1"]
        out[n] = shift
    return out


# ---------------------------------------------------------------------------
# 受限二次求解（T3/T4/T5 被测对象）
# ---------------------------------------------------------------------------

def quad_value(a, b, K):
    """m(a) = b^T a + 0.5 a^T K a（a 为长度 2 序列）。"""
    x, y = float(a[0]), float(a[1])
    b1, b2 = float(b[0]), float(b[1])
    k11, k12, k21, k22 = float(K[0][0]), float(K[0][1]), float(K[1][0]), float(K[1][1])
    return b1 * x + b2 * y + 0.5 * (k11 * x * x + (k12 + k21) * x * y + k22 * y * y)


def _sym_ok(K):
    """K12/K21 相对非对称检查（报告原值；超阈则不建议求解）。"""
    k12, k21 = float(K[0][1]), float(K[1][0])
    denom = max(1.0, abs(k12), abs(k21))
    return abs(k12 - k21) / denom <= K_ASYM_REL_TOL, abs(k12 - k21) / denom


def eig_cond(K):
    """2x2 对称矩阵特征值（升序）与条件数 |lmax|/|lmin|（lmin=0 -> inf）。"""
    k11, k12, k22 = float(K[0][0]), 0.5 * (float(K[0][1]) + float(K[1][0])), float(K[1][1])
    tr = k11 + k22
    det = k11 * k22 - k12 * k12
    disc = max(0.0, 0.25 * tr * tr - det)
    root = math.sqrt(disc)
    l1 = 0.5 * tr - root
    l2 = 0.5 * tr + root
    lo, hi = min(l1, l2), max(l1, l2)
    if abs(lo) <= 0.0:
        cond = math.inf
    else:
        cond = abs(hi) / abs(lo)
    return (lo, hi), cond


def _clip(v, lo, hi):
    return min(max(v, lo), hi)


def _best_candidate(cands, b, K):
    """按 (m, y, |x|, x) 确定性选优；并列时偏好 y=0、小 |x|。"""
    best = None
    for (x, y, tag) in cands:
        key = (quad_value((x, y), b, K), y, abs(x), x)
        if best is None or key < best[0]:
            best = (key, (x, y, tag))
    return best[1]


def solve_lr_only(b, K, r):
    """a_LR = argmin_{x∈[-r,r]} m(x,0)。

    返回 dict(x, m, boundary, flat)。K11<=0 时极小在端点（角点语义）。
    """
    b1, k11 = float(b[0]), float(K[0][0])
    cands = [(-r, 0.0, "corner"), (0.0, 0.0, "origin"), (r, 0.0, "corner")]
    if k11 > 0.0:
        x0 = -b1 / k11
        if -r < x0 < r:
            cands.append((_clip(x0, -r, r), 0.0, "edge_interior"))
    x, y, tag = _best_candidate(cands, b, K)
    mvals = [quad_value((cx, cy), b, K) for cx, cy, _ in cands]
    flat = (max(mvals) - min(mvals)) <= 0.0
    return {"x": x, "m": quad_value((x, y), b, K),
            "boundary": tag.startswith("corner"),
            "flat": bool(flat)}


def solve_rectangle(b, K, rx, ry):
    """a_2D = argmin_{x∈[-r,r], y∈[0,r]} m(x,y)（完整候选枚举，无 PSD 裁剪）。

    返回 dict(x, y, m, tag, candidates, eig, cond, interior_stationary,
    interior_status)。候选：4 角点、原点、四边内部极小点、可行内部驻点。
    """
    import numpy as np
    b1, b2 = float(b[0]), float(b[1])
    k11 = 0.5 * (float(K[0][0]) + float(K[0][0]))
    k12 = 0.5 * (float(K[0][1]) + float(K[1][0]))
    k22 = 0.5 * (float(K[1][1]) + float(K[1][1]))
    Ksym = [[k11, k12], [k12, k22]]
    eig, cond = eig_cond(Ksym)

    cands = [(-rx, 0.0, "corner"), (0.0, 0.0, "origin"), (rx, 0.0, "corner"),
             (-rx, ry, "corner"), (rx, ry, "corner")]
    if k11 > 0.0:  # y=0 边
        x0 = _clip(-b1 / k11, -rx, rx)
        cands.append((x0, 0.0, "edge_y0"))
    if k11 > 0.0:  # y=ry 边
        x0 = _clip(-(b1 + k12 * ry) / k11, -rx, rx)
        cands.append((x0, ry, "edge_yr"))
    if k22 > 0.0:  # x=-rx 边
        y0 = _clip(-(b2 - k12 * rx) / k22, 0.0, ry)
        cands.append((-rx, y0, "edge_xm"))
    if k22 > 0.0:  # x=+rx 边
        y0 = _clip(-(b2 + k12 * rx) / k22, 0.0, ry)
        cands.append((rx, y0, "edge_xp"))

    interior_status = "SKIPPED_NO_ATTEMPT"
    xs = ys = float("nan")
    if all(map(lambda v: math.isfinite(v), (k11, k12, k22, b1, b2))):
        Km = np.array(Ksym, dtype=np.float64)
        if cond < COND_SKIP_INTERIOR:
            try:
                sol = np.linalg.solve(Km, -np.array([b1, b2], dtype=np.float64))
                xs, ys = float(sol[0]), float(sol[1])
                if -rx < xs < rx and 0.0 < ys < ry:
                    cands.append((xs, ys, "interior"))
                    interior_status = "INSIDE"
                else:
                    interior_status = "OUTSIDE"
            except np.linalg.LinAlgError:
                interior_status = "SINGULAR"
        else:
            interior_status = "ILL_CONDITIONED"

    x, y, tag = _best_candidate(cands, b, Ksym)
    return {"x": x, "y": y, "m": quad_value((x, y), b, Ksym), "tag": tag,
            "n_candidates": len(cands), "eig": eig, "cond": cond,
            "x_interior": xs, "y_interior": ys,
            "interior_status": interior_status,
            "boundary": tag.startswith("corner"),
            "k_asym_rel": _sym_ok(K)[1]}


def schur_elimination(b, K):
    """Schur 消元：仅在 K 正定（特征值均 > tol）且 K11、K_eff 非退化时有效。

    b_eff/K_eff 是该任务二次模型在允许 LR 调整后，对读出改动的剩余一阶
    驱动力与曲率；不是新 proxy。
    """
    (lo, hi), cond = eig_cond(K)
    tol = REL_TOL * max(1.0, abs(hi))
    if not (lo > tol and hi > tol):
        return {"valid": False, "reason": "K_NOT_PD",
                "eig": (lo, hi), "cond": cond}
    k11 = float(K[0][0])
    if abs(k11) <= tol:
        return {"valid": False, "reason": "K11_DEGENERATE",
                "eig": (lo, hi), "cond": cond}
    k12 = 0.5 * (float(K[0][1]) + float(K[1][0]))
    k21 = k12
    k22 = float(K[1][1])
    b1, b2 = float(b[0]), float(b[1])
    k_eff = k22 - k21 * k12 / k11
    if abs(k_eff) <= tol:
        return {"valid": False, "reason": "KEFF_ZERO", "eig": (lo, hi),
                "cond": cond, "b_eff": b2 - k21 * b1 / k11, "K_eff": 0.0}
    b_eff = b2 - k21 * b1 / k11
    y_star = -b_eff / k_eff
    x_star = -(b1 + k12 * y_star) / k11
    return {"valid": True, "reason": "OK", "eig": (lo, hi), "cond": cond,
            "b_eff": b_eff, "K_eff": k_eff, "y_star": y_star,
            "x_star": x_star}


def kkt_y0_flag(b, K, x_LR):
    """y=0 的 KKT 依据：K 半正定（容差内）且 LR-only 解处 b2+K21*x_LR>=0。"""
    (lo, hi), _ = eig_cond(K)
    tol = REL_TOL * max(1.0, abs(hi))
    k21 = 0.5 * (float(K[0][1]) + float(K[1][0]))
    grad_y = float(b[1]) + k21 * float(x_LR)
    return {"psd_within_tol": bool(lo >= -tol), "grad_y_at_xLR": grad_y,
            "kkt_ok": bool(lo >= -tol and grad_y >= -tol)}


# ---------------------------------------------------------------------------
# 聚合/统计/seal（T8 被测对象）
# ---------------------------------------------------------------------------

def aggregate_control(row_sums, row_counts):
    """V_ctrl = sum(row_sums)/sum(row_counts)；全库有效分母，非 batch 均值。"""
    total = float(sum(float(s) for s in row_sums))
    cnt = int(sum(int(c) for c in row_counts))
    return {"sum": total, "count": cnt, "mean": total / max(cnt, 1)}


def delta_v(sum_a, sum_base, count):
    """DeltaV = (S_a - S_base)/N（fp64 标量算术；native 点严格 0）。"""
    return (float(sum_a) - float(sum_base)) / float(count)


def paired_se(values):
    """SE = std(ddof=1)/sqrt(n)。"""
    xs = [float(v) for v in values]
    n = len(xs)
    if n < 2:
        return float("nan")
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var / n)


def mean_k_operators(per_probe_coeffs):
    """按 probe 先乘后平均：b_hat/K_hat = E[per-probe b/K]。

    [FR-NATIVE-ADAMW] 不做 E[D]^T E[H] E[D] 替换（T7 mutant 必须失败）。
    输入为 [{b1,b2,k11,k12_direct,k21_direct,k22}, ...]。
    """
    n = len(per_probe_coeffs)
    if n == 0:
        raise ValueError("empty probe list")
    keys = ("b1", "b2", "k11", "k12_direct", "k21_direct", "k22")
    out = {k: sum(float(r[k]) for r in per_probe_coeffs) / n for k in keys}
    out["n_probes"] = n
    return out


def symmetrized_K(k12_direct, k21_direct, k11, k22):
    """数值合理的小非对称 -> (K+K^T)/2 求解用；原值仍照报。"""
    ok, rel = _sym_ok([[0.0, float(k12_direct)], [float(k21_direct), 0.0]])
    k12s = 0.5 * (float(k12_direct) + float(k21_direct))
    return {"ok": ok, "rel_asym": rel, "K": [[float(k11), k12s],
                                             [k12s, float(k22)]]}


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_seal(seal_path, payload_paths, extra=None):
    """封存 FIT 系数与解：记录文件 sha256。AUDIT 前必须通过 verify。"""
    rec = {"files": {str(Path(p).name): _sha256_file(p) for p in payload_paths},
           "extra": extra or {}, "sealed": True}
    Path(seal_path).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def verify_seal(seal_path, payload_paths):
    """任何文件被改动（包括 AUDIT 回写系数）-> False。"""
    rec = json.loads(Path(seal_path).read_text())
    for p in payload_paths:
        name = str(Path(p).name)
        if name not in rec["files"]:
            return False, f"missing:{name}"
        if _sha256_file(p) != rec["files"][name]:
            return False, f"hash_mismatch:{name}"
    return True, ""


def condition_label(seed, start, k):
    """12 个条件的标识（6 state × 2 k）；不把条件当独立 seed。"""
    return f"s{int(seed)}_t{int(start)}_k{float(k):g}"


def stratify(labels):
    """按 seed/start/k 分层汇总键（T8：6x2 不是 12 个独立 seeds）。"""
    groups = {"seed": {}, "start": {}, "k": {}}
    for lab in labels:
        seed, start, k = lab.split("_")
        groups["seed"].setdefault(seed[1:], []).append(lab)
        groups["start"].setdefault(start[1:], []).append(lab)
        groups["k"].setdefault(k[1:], []).append(lab)
    return groups


def plan_budget():
    """计划用量的独立重算（协议 §7 表；与 runner 常量互为参照）。"""
    train_fw = 6 * 16 * 2
    fit_fw = 6 * 2 * 8 * 8
    hvp_dirs = 6 * 2 * 8 * 2
    ctrl_fw = 6 * 2 * 8 * 20 * 8
    check_fw = 6 * 2 * 8 * 3 * 8
    total_fw = train_fw + fit_fw + ctrl_fw + check_fw
    train_bw = 6 * 16 * 2
    fit_bw = fit_fw  # 每目标 chunk 一次一阶 backward
    vjp_micro = hvp_dirs * 8  # 每个 VJP 走 8-chunk 图
    total_bw = train_bw + fit_bw + vjp_micro
    return {"train_fw": train_fw, "fit_fw": fit_fw, "hvp_dirs": hvp_dirs,
            "ctrl_fw": ctrl_fw, "check_fw": check_fw, "total_fw": total_fw,
            "train_bw": train_bw, "fit_bw": fit_bw, "vjp_micro": vjp_micro,
            "total_bw_micro": total_bw,
            "committed_updates": 0, "v_final_calls": 0}
