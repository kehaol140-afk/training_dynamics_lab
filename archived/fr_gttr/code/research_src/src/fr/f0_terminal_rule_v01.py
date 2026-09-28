"""GTTR-F0 全时域终点规则求解 v1 —— 规则数学、全状态灵敏度与求解器纯函数。

协议：FR_GTTR_F0_Full_Horizon_Terminal_Rule_Solver_v1.md §2、§4–§5、§6.1。

规则（T=256，w[t]=2*eta_base[gm(t)]，start46 首 gm=94）：
    psi[t] = 2t/(T-1)-1
    Z(b) = sum(w*exp(b*psi))/sum(w)
    eta[t;phi] = w[t]*exp(a+b*psi[t])/Z(b)
    sum_t eta[t] = exp(a)*sum_t w[t]
    d_eta/da = eta ; d_eta/db = eta*(psi[t]-mu_psi(b))
    mu_psi(b) = sum(w*exp(b*psi)*psi)/sum(w*exp(b*psi))
a,b ∈ [-log2,+log2]^2。Z/psi/eta 全部 fp64。

全状态前向灵敏度（§4.2，torch，逐有效 batch）：正常 AdamW 更新 +
S_theta/S_m/S_v 递推；clip 用冻结 a0c.clip_jvp（真实 clip_grad_norm_
语义含 1e-6 分母）；每步 HVP=H_batch@S_theta[j]，方向 detach；不冻结 H、
不清零历史灵敏度、不截断时域；vbar=0 不加 v_floor：结构性恒零分支才允许
零导数，否则报错中止。d_eta/db 包含 Z 对 b 的导数（mu_psi 项）。
"""
from __future__ import annotations

import math

import numpy as np

LOG2 = math.log(2.0)
T_MAIN = 256
N0 = 46


def eta_base_at(gm: int) -> float:
    from fr.r2b_bridge_v01 import eta_base as _eb
    return float(_eb(int(gm)))


def build_w(t_horizon: int = T_MAIN, n0: int = N0) -> np.ndarray:
    """w[t] = 2*eta_base(gm(t))，gm(t)=2*n0+2*(t+1)（真实调用点）。fp64。"""
    g = [2.0 * eta_base_at(2 * n0 + 2 * (t + 1)) for t in range(t_horizon)]
    return np.asarray(g, dtype=np.float64)


def build_psi(t_horizon: int = T_MAIN) -> np.ndarray:
    """psi 按 T=256 时钟定义（SHORT64 也用同一时钟，不压缩）。"""
    return 2.0 * np.arange(t_horizon, dtype=np.float64) / (T_MAIN - 1) - 1.0


def rule_eta(a: float, b: float, w: np.ndarray, psi: np.ndarray):
    """返回 (eta, d_eta_a, d_eta_b)，fp64。d_eta/db 含 Z 的导数（mu_psi）。"""
    eb = np.exp(float(b) * psi)
    num = float(np.dot(w, eb))
    den = float(np.sum(w))
    z = num / den
    eta = w * math.exp(float(a)) * eb / z
    mu = float(np.dot(w, eb * psi)) / num
    d_a = eta.copy()
    d_b = eta * (psi - mu)
    return eta, d_a, d_b


def project_box(phi, free=(True, True)):
    lo, hi = -LOG2, LOG2
    return (min(max(phi[0], lo), hi) if free[0] else phi[0],
            min(max(phi[1], lo), hi) if free[1] else phi[1])


def solver_candidates(phi, g, free=(True, True)):
    """h=(log2)/2*{1,1/2,1/4}；phi_trial=project(phi-h*g/||g||inf)；delta=0 跳过。"""
    gvec = np.asarray([g[0] if free[0] else 0.0, g[1] if free[1] else 0.0],
                      dtype=np.float64)
    ginf = float(np.max(np.abs(gvec)))
    if ginf == 0.0:
        return []
    step = gvec / ginf
    out = []
    for frac in (1.0, 0.5, 0.25):
        h = (LOG2 / 2.0) * frac
        trial = project_box((phi[0] - h * step[0], phi[1] - h * step[1]),
                            free)
        delta = (trial[0] - phi[0], trial[1] - phi[1])
        if delta[0] == 0.0 and delta[1] == 0.0:
            continue
        out.append((h, trial, delta))
    return out


def armijo_ok(j_trial: float, j_cur: float, g, delta, c: float = 1e-4):
    return j_trial <= j_cur + c * (g[0] * delta[0] + g[1] * delta[1])


def schedule_summary(eta: np.ndarray):
    return {"first": float(eta[0]), "last": float(eta[-1]),
            "min": float(eta.min()), "max": float(eta.max()),
            "sum": float(eta.sum()),
            "sum_eta_eq": math.isclose(float(eta.sum()), float(eta.sum()),
                                       rel_tol=1e-12)}


def tie_break_argmin(cands):
    """精确相等按 (J, |a|+|b|, a, b) 字典序。cands: [(phi, J)]。"""
    best = None
    for phi, j in cands:
        key = (j, abs(phi[0]) + abs(phi[1]), phi[0], phi[1])
        if best is None or key < best[0]:
            best = (key, phi, j)
    return best[1], best[2]


# ---------------------------------------------------------------------------
# torch：AdamW 更新 + 全状态灵敏度（value/sens 共用同一算术路径）
# ---------------------------------------------------------------------------

def f0_adamw_sens_step(theta, m, v, g, beta1, beta2, eps, wd, eta,
                       t_step, tau=1.0, sens=None):
    """一次 AdamW 更新（真实 clip 语义）；可选两方向灵敏度递推。

    theta/m/v/g: {name: tensor}（唯一参数；grad=None 参数不在 dict 中）。
    值轨迹按输入 dtype（fp32）位级不变；**灵敏度递推整体在 float64**
    （真实网络上 1/sqrt(v)^3 量级的条件数会使 fp32 灵敏度发生灾难性
    消零；fp64 是同一数学的精度升级，不改变值路径）。
    sens: {"dirs": [("a", d_eta_a), ("b", d_eta_b)],
           "S_theta"/"S_m"/"S_v": {name: {dir: tensor(float64)}},
           "Hg": {dir: {name: tensor}}}（Hg 由调用方 VJP 计算，方向 detach）。
    dc 按冻结 a0c.clip_jvp 公式（含 1e-6 分母）在 fp64 展开。
    返回 (theta_new, m_new, v_new, sens_new, info)。
    [FR-NATIVE-ADAMW] 与原生 AdamW 相同的矩更新/bias 校正/P/decay 顺序。
    """
    import torch
    from fr.a0c_adamw_response_propagation_v01 import clip_jvp

    names = list(theta)
    zero_t = {n: torch.zeros_like(g[n]) for n in names}
    clip_first = clip_jvp(gradient_by_name=g,
                          tangent_gradient_by_name=zero_t, max_norm=tau)
    c = clip_first["clipped_gradient"]

    bc1 = 1.0 - beta1 ** int(t_step)
    bc2 = 1.0 - beta2 ** int(t_step)
    theta_new, m_new, v_new = {}, {}, {}
    mbar32, sv32, pden32 = {}, {}, {}
    for n in names:
        m_plus = beta1 * m[n] + (1.0 - beta1) * c[n]
        v_plus = beta2 * v[n] + (1.0 - beta2) * c[n] * c[n]
        mbar = m_plus / bc1
        vbar = v_plus / bc2
        sv = vbar.sqrt()
        p_den = sv + eps
        theta_new[n] = theta[n] - eta * (mbar / p_den + wd * theta[n])
        m_new[n] = m_plus
        v_new[n] = v_plus
        mbar32[n], sv32[n], pden32[n] = mbar, sv, p_den

    sens_new = None
    if sens is not None:
        dirs = list(sens["dirs"])
        # clip JVP（冻结公式，fp64）：coef*tang - tau*(g.tang)/(n(n+eps)^2)*g
        clip_eps = 1e-6
        norm64 = 0.0
        for n in names:
            norm64 += float(torch.sum(
                g[n].detach().to(torch.float64).square()).item())
        norm64 = math.sqrt(norm64)
        coef = min(tau / (norm64 + clip_eps), 1.0)
        active = coef < 1.0
        s_th, s_m, s_v = {}, {}, {}
        for d, d_eta in dirs:
            # 修订（核验轮发现）：radial 必须用全局点积
            # dot_global = sum_p <g_p, Hs_p>（同一标量作用于所有参数块），
            # 不得逐块使用局部 <g_p, Hs_p>。
            if active:
                dot_global = 0.0
                for n in names:
                    dot_global += float(torch.sum(
                        g[n].detach().to(torch.float64)
                        * sens["Hg"][d][n].to(torch.float64)).item())
            st, sm, svs = {}, {}, {}
            for n in names:
                c64 = c[n].detach().to(torch.float64)
                hg64 = sens["Hg"][d][n].to(torch.float64)
                if active:
                    g64 = g[n].detach().to(torch.float64)
                    radial = tau * dot_global / (norm64 * (norm64 + clip_eps)
                                                 ** 2)
                    dc64 = coef * hg64 - radial * g64
                else:
                    dc64 = hg64
                sm64 = beta1 * sens["S_m"][n][d] + (1.0 - beta1) * dc64
                sv64 = beta2 * sens["S_v"][n][d] \
                    + (1.0 - beta2) * 2.0 * c64 * dc64
                dmbar = sm64 / bc1
                dvbar = sv64 / bc2
                sv = sv32[n].to(torch.float64)
                p_den = pden32[n].to(torch.float64)
                mbar = mbar32[n].to(torch.float64)
                zero_mask = (sv == 0)
                if bool((zero_mask & (dvbar != 0)).any()):
                    raise ZeroDivisionError(
                        f"vbar==0 with nonzero dvbar at {n}/{d}: 非光滑分支，"
                        "不加 v_floor，中止导数实验")
                dP = torch.where(zero_mask, torch.zeros_like(dvbar),
                                 -dvbar / (2.0 * sv * p_den * p_den))
                r64 = mbar / p_den + wd * theta[n].detach().to(torch.float64)
                st[n] = sens["S_theta"][n][d] - d_eta * r64 \
                    - eta * (dmbar / p_den + dP * mbar
                             + wd * sens["S_theta"][n][d])
                sm[n] = sm64
                svs[n] = sv64
            s_th[d], s_m[d], s_v[d] = st, sm, svs
        sens_new = {
            "dirs": dirs,
            "S_theta": {n: {d: s_th[d][n] for d, _ in dirs} for n in names},
            "S_m": {n: {d: s_m[d][n] for d, _ in dirs} for n in names},
            "S_v": {n: {d: s_v[d][n] for d, _ in dirs} for n in names}}
    info = {"clip_active": clip_first["clip_active"],
            "clip_coefficient": clip_first["clip_coefficient"],
            "preclip_norm": clip_first["clip_before_norm"]}
    return theta_new, m_new, v_new, sens_new, info


def init_sensitivity(theta, dirs):
    """S 初值全 0（起点不依赖 phi；灵敏度恒 float64）。Hg 每步填入。

    修订（核验轮发现）：三组 S 必须各自独立内层 dict——浅拷贝会使
    S_theta/S_m/S_v 共享同一 {dir: tensor}，任何注入赋值泄漏到其余两组。
    """
    import torch
    def fresh():
        return {n: {d: torch.zeros(t.shape, dtype=torch.float64,
                                    device=t.device) for d, _ in dirs}
                for n, t in theta.items()}
    return {"dirs": dirs, "S_theta": fresh(), "S_m": fresh(),
            "S_v": fresh(), "Hg": None}


def terminal_rule_gradient(q, sens, names):
    """grad_phi_j = dot(q, S_theta_H[j])（fp64 累加）。"""
    total = {d: 0.0 for d, _ in sens["dirs"]}
    for n in names:
        for d, _ in sens["dirs"]:
            total[d] += float(torch_sum64(q[n] * sens["S_theta"][n][d]))
    return (total.get("a", 0.0), total.get("b", 0.0))


def torch_sum64(t):
    import torch
    return torch.sum(t.detach().to(torch.float64)).item()
