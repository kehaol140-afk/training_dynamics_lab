"""Phase B v3 —— GTTR 动作前评分模型的可执行构件。

FR_PhaseB_GTTR_Action_Scoring_Execution_v3.md §3/§4 的实现：
- lr_action_seq：eta_k^u = eta_base(n0+k) * exp(u * 1[k<4])（k=0..63）
- 参考梯度场 G_SIM（P=2 SIM microbatch、FP32、train-mode 固定随机掩码、无 step）
- s_ref0：g0 经真实 clip + 一次 AdamW 参考算术的 native 位移（虚拟，不计真实 update）
- Krylov/Arnoldi FD-HVP 方向（delta_fd = 1e-3*max(||theta0||,1)，MGS，rank<=4，
  第一方向 half-check：fd_rel/fd_cos，LOCAL_OPERATOR_NOT_CALIBRATED 门）
- MODEL_NO_GEOMETRY: g_hat = g0；MODEL_GTTR: g_hat = g0 + Y V^T (theta - theta0)
- 虚拟传播：每步对 g_hat 重新执行真实全局 clip，然后调用真实 torch.optim.AdamW
  算术（bias correction、解耦 wd、完整坐标），u=0 时与真实一步逐位一致（T2）
- J_hat_M(H,u) = q0 . (theta_hat[H;u] - theta_hat[H;0])；native 项恒 0
- 选择：argmin_u，平局 |u| 小优先，再按 {0,-.05,+.05}，浮点容差 1e-12

只依赖 torch/numpy，不在本模块内做任何 I/O 或 GPU 全局状态假设；
所有张量操作以 unique 参数集合为准（tied 权重去重，与 optimizer 一致）。
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ACTIONS = (0.0, -0.05, +0.05)          # 固定动作顺序（平局用）
H_POINTS = (1, 4, 16, 64)
N_VIRTUAL = 64
FD_DELTA_SCALE = 1e-3
FD_RANK_TOL = 1e-6
FD_REL_GATE = 0.20
HALF_CHECK_TOL = 1e-12                 # 浮点平局容差（nats）
INTERVENTION_STEPS = 4


# ---------------------------------------------------------------------------
# 基础：dict-of-tensor 代数（key 为参数名；调用方保证已按 id 去重）
# ---------------------------------------------------------------------------

def flat_dot(a: Dict[str, object], b: Dict[str, object]) -> float:
    """a·b（FP64 语义由 torch 归约保证；返回 python float）。"""
    import torch
    tot = 0.0
    for k in a:
        tot += float(torch.dot(a[k].reshape(-1).double(), b[k].reshape(-1).double()))
    return tot


def flat_norm(a: Dict[str, object]) -> float:
    return math.sqrt(max(flat_dot(a, a), 0.0))


def flat_add_(a: Dict[str, object], b: Dict[str, object], scale: float = 1.0) -> None:
    for k in a:
        a[k].add_(b[k], alpha=scale)


def flat_scale_(a: Dict[str, object], c: float) -> None:
    for k in a:
        a[k].mul_(c)


def flat_clone(a: Dict[str, object]) -> Dict[str, object]:
    return {k: v.detach().clone() for k, v in a.items()}


def clip_coeff(global_norm: float, tau: float) -> float:
    """全局 L2 clip 系数；norm<=0 时系数 1（梯度保持为零）。"""
    if not math.isfinite(global_norm):
        return 0.0
    if global_norm <= 0.0:
        return 1.0
    return min(1.0, tau / global_norm)


# ---------------------------------------------------------------------------
# 动作 LR 序列
# ---------------------------------------------------------------------------

def lr_action_seq(eta_base: Callable[[int], float], n0: int,
                  u: float, n_steps: int = N_VIRTUAL) -> List[float]:
    """eta_k^u = eta_base(n0+k) * exp(u * 1[k<4])，k=0..n_steps-1。

    eta_base(n) 是 optimizer step 索引 n 消费的 base LR（last micro 写入值）。
    """
    return [float(eta_base(n0 + k)) * math.exp(u if k < INTERVENTION_STEPS else 0.0)
            for k in range(n_steps)]


# ---------------------------------------------------------------------------
# 虚拟 AdamW 传播（真实 torch.optim.AdamW 算术；clip 由调用方先作用于 g）
# ---------------------------------------------------------------------------

def build_virtual_optimizer(torch, named_params: Dict[str, object],
                            m0: Dict[str, object], v0: Dict[str, object],
                            n0: int, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
    """克隆参数 + 预载 moments/step 的 AdamW；step() 与真实 optimizer 逐位一致。"""
    vparams = {k: p.detach().clone().requires_grad_(True)
               for k, p in named_params.items()}
    opt = torch.optim.AdamW(list(vparams.values()), lr=1.0,
                            betas=betas, eps=eps, weight_decay=wd)
    for k, p in vparams.items():
        st = opt.state[p]
        st["exp_avg"] = m0[k].detach().clone()
        st["exp_avg_sq"] = v0[k].detach().clone()
        st["step"] = torch.tensor(float(n0), device=p.device)
    return vparams, opt


def virtual_step(opt, vparams: Dict[str, object],
                 g: Dict[str, object], lr: float, tau: float) -> Dict[str, float]:
    """对 g 重新执行真实全局 clip，赋为 .grad，执行一步 AdamW。返回指标。"""
    import torch
    theta_before = {k: p.detach().clone() for k, p in vparams.items()}
    gn = flat_norm(g)
    c = clip_coeff(gn, tau)
    g_clip = {k: g[k] * c for k in g}
    for k, p in vparams.items():
        p.grad = g_clip[k].detach().clone()
    for group in opt.param_groups:
        group["lr"] = float(lr)
    opt.step()
    upd = math.sqrt(sum(
        float(torch.dot(
            (vparams[k].detach() - theta_before[k]).reshape(-1).double(),
            (vparams[k].detach() - theta_before[k]).reshape(-1).double()))
        for k in vparams))
    finite = bool(math.isfinite(upd) and math.isfinite(gn))
    return {"preclip_norm": gn, "clip_coeff": c, "clip_active": bool(gn > tau),
            "update_norm": upd, "finite": finite}


def reference_displacement(torch, named_params, m0, v0, n0, g0, lr0,
                           betas=(0.9, 0.999), eps=1e-8, wd=0.01, tau=1.0):
    """s_ref0：g0 经 clip + 一次 AdamW 参考算术的 native 位移（独立克隆，零副作用）。"""
    theta0 = flat_clone(named_params)
    vparams, opt = build_virtual_optimizer(torch, named_params, m0, v0, n0,
                                           betas, eps, wd)
    metrics = virtual_step(opt, vparams, g0, lr0, tau)
    s_ref0 = {k: vparams[k].detach() - theta0[k] for k in theta0}
    return s_ref0, metrics


# ---------------------------------------------------------------------------
# Krylov/Arnoldi FD-HVP 方向
# ---------------------------------------------------------------------------

def _mgs_orthogonalize(torch, y, V: List[Dict[str, object]]):
    """两次 modified Gram-Schmidt；返回 (y_perp, 系数列表)。"""
    h = []
    y_cur = flat_clone(y)
    for _ in range(2):
        h.append([flat_dot(y_cur, v) for v in V])
        for v, c in zip(V, h[-1]):
            flat_add_(y_cur, v, scale=-c)
    return y_cur, h


def krylov_fd_directions(torch, g_eval: Callable[[Optional[Dict[str, object]]], Dict[str, object]],
                         theta0_norm: float, v1: Dict[str, object],
                         max_q: int = 4, delta: Optional[float] = None,
                         rank_tol: float = FD_RANK_TOL):
    """Arnoldi 式 FD-HVP 方向构建。

    g_eval(overlay) 必须在内部固定 SIM 随机性（同一批、同一掩码）。
    返回 dict：V, Y, rank, delta, fd_rel/fd_cos（第一方向 half-check）、
    per-direction ||y_j||、expanded 标志、诊断标签。
    """
    if delta is None:
        delta = FD_DELTA_SCALE * max(float(theta0_norm), 1.0)
    V: List[Dict[str, object]] = [flat_clone(v1)]
    flat_scale_(V[0], 1.0 / max(flat_norm(V[0]), 1e-30))
    Y: List[Dict[str, object]] = []
    diag = {"delta": delta, "y_norms": [], "expanded": [], "labels": []}
    fd_rel = fd_cos = None
    for j in range(max_q):
        vj = V[j]
        overlay_p = {k: vj[k] * (+delta) for k in vj}
        overlay_m = {k: vj[k] * (-delta) for k in vj}
        yj = {k: (gp - gm) / (2.0 * delta)
              for (k, gp), gm in zip(g_eval(overlay_p).items(),
                                     g_eval(overlay_m).values())}
        y_norm = flat_norm(yj)
        diag["y_norms"].append(y_norm)
        if not math.isfinite(y_norm):
            diag["labels"].append("NONFINITE_FD")
            break
        if y_norm <= 0.0:
            diag["labels"].append("ZERO_FD_SIGNAL")
            break
        if j == 0:
            half_delta = delta / 2.0
            ohp = {k: vj[k] * (+half_delta) for k in vj}
            ohm = {k: vj[k] * (-half_delta) for k in vj}
            y_half = {k: (gp - gm) / (2.0 * half_delta)
                      for (k, gp), gm in zip(g_eval(ohp).items(),
                                             g_eval(ohm).values())}
            hn = flat_norm(y_half)
            fd_rel = (flat_norm({k: yj[k] - y_half[k] for k in yj})
                      / max(y_norm, hn, 1e-30))
            fd_cos = (flat_dot(yj, y_half) / max(y_norm * hn, 1e-30)) if hn > 0 else None
            diag["fd_rel"] = fd_rel
            diag["fd_cos"] = fd_cos
            if hn <= 0.0 or not math.isfinite(hn):
                diag["labels"].append("HALF_CHECK_NONFINITE_OR_ZERO")
                Y.append(yj)
                diag["expanded"].append(False)
                break
            if fd_rel > FD_REL_GATE:
                diag["labels"].append("LOCAL_OPERATOR_NOT_CALIBRATED")
                Y.append(yj)
                diag["expanded"].append(False)
                break
        if j == max_q - 1:
            Y.append(yj)
            diag["expanded"].append(False)
            break
        y_perp, _ = _mgs_orthogonalize(torch, yj, V)
        r_norm = flat_norm(y_perp)
        if r_norm <= rank_tol * y_norm:
            # yj 不提供新子空间（H v_j 可由既有 (v_i,y_i) 表示），不入列
            diag["labels"].append("RANK_BREAK")
            break
        Y.append(yj)
        diag["expanded"].append(True)
        V.append(y_perp)
        flat_scale_(V[-1], 1.0 / r_norm)
    diag["rank"] = len(Y)
    diag["max_rank"] = max_q
    return {"V": V, "Y": Y, "diag": diag, "delta": delta}


# ---------------------------------------------------------------------------
# 场模型与虚拟传播
# ---------------------------------------------------------------------------

def make_field(g0: Dict[str, object], V: Optional[List[Dict[str, object]]] = None,
               Y: Optional[List[Dict[str, object]]] = None):
    """返回 g_hat(Δtheta) dict。NO_GEOMETRY: V=None/Y=None → g_hat=g0。"""
    def field(delta_theta: Dict[str, object]) -> Dict[str, object]:
        if not V:
            return {k: g.clone() for k, g in g0.items()}
        coeffs = [flat_dot(delta_theta, v) for v in V]
        out = {k: g.clone() for k, g in g0.items()}
        for y, c in zip(Y, coeffs):
            flat_add_(out, y, scale=c)
        return out
    return field


def propagate(torch, named_params, m0, v0, n0, field, lr_seq,
              tau: float = 1.0, h_points: Sequence[int] = H_POINTS,
              n_steps: int = N_VIRTUAL, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
    """虚拟 AdamW 传播；返回 {h: theta_hat dict} 与逐步指标。

    非有限时停止：已有点保留，后续 h 标 INVALID（调用方处理）。
    """
    vparams, opt = build_virtual_optimizer(torch, named_params, m0, v0, n0,
                                           betas, eps, wd)
    theta0 = flat_clone(named_params)
    theta_h = {}
    metrics = []
    invalid_from = None
    for k in range(n_steps):
        delta = {kk: vparams[kk].detach() - theta0[kk] for kk in theta0}
        g_hat = field(delta)
        m = virtual_step(opt, vparams, g_hat, lr_seq[k], tau)
        m["k"] = k
        metrics.append(m)
        if not m["finite"]:
            invalid_from = k + 1
            break
        if (k + 1) in h_points:
            theta_h[k + 1] = flat_clone(vparams)
    if invalid_from is not None:
        for h in h_points:
            if h >= invalid_from and h not in theta_h:
                theta_h[h] = None
    return theta_h, metrics


# ---------------------------------------------------------------------------
# 评分与选择
# ---------------------------------------------------------------------------

def j_hat(q0, theta_u, theta_0arm) -> float:
    """J_hat = q0 · (theta_hat[H;u] - theta_hat[H;0])；u=0 时恒 0。"""
    return flat_dot(q0, {k: theta_u[k] - theta_0arm[k] for k in theta_u})


def select_action(j_values: Dict[float, float]) -> float:
    """argmin_u J；平局 |u| 小优先，再按 ACTIONS 固定顺序；1e-12 容差仅数值平局。"""
    best = None
    for u in ACTIONS:
        j = j_values[u]
        if best is None:
            best = (u, j)
            continue
        bu, bj = best
        if j < bj - HALF_CHECK_TOL:
            best = (u, j)
        elif abs(j - bj) <= HALF_CHECK_TOL:
            if abs(u) < abs(bu) or (abs(u) == abs(bu) and
                                    ACTIONS.index(u) < ACTIONS.index(bu)):
                best = (u, j)
    return best[0]


def q0_dot_delta(q0, theta_a, theta_b) -> float:
    """标量 q0·(theta_a - theta_b)（linear_obs 用）。"""
    return flat_dot(q0, {k: theta_a[k] - theta_b[k] for k in theta_a})


# ---------------------------------------------------------------------------
# 误差分解（纯函数；analyze 独立复算用）
# ---------------------------------------------------------------------------

def error_decomposition(j_hat_val: float, j_obs_ctrl: float, linear_obs: float):
    """e_total = e_propagation + e_readout（恒等式，T7 独立复算）。"""
    e_total = j_hat_val - j_obs_ctrl
    e_prop = j_hat_val - linear_obs
    e_read = linear_obs - j_obs_ctrl
    return {"e_total": e_total, "e_propagation": e_prop,
            "e_readout": e_read, "identity_holds": abs(e_total - (e_prop + e_read)) <= 1e-12}
