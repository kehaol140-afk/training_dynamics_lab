"""GTTR-D1 直接更新规则 pilot v1 —— 纯 tensor 读出与 QHAdamW 步进。

协议：FR_GTTR_D1_Direct_Update_Pilot_Claude_Code_v1.md §1.3。
唯一公式（p 为唯一参数张量，c 为 clip 后的真实梯度，t 为 true_parameter_step）：

    m_plus = beta1*m + (1-beta1)*c ; v_plus = beta2*v + (1-beta2)*c^2
    mhat = m_plus/(1-beta1**t)     ; vhat = v_plus/(1-beta2**t)
    P = 1/(sqrt(vhat)+eps)
    rA = P*mhat
    rQ = (1-alpha)*rA + alpha*(P*c)              # alpha=0.10 冻结
    rN: A=||rA||_F, Q=||rQ||_F, B=||P*c||_F
        A==0                 -> rN = rA   (fallback ZERO)
        Q <= 1e-12*max(A,B)  -> rN = rA   (fallback CANCELLATION)
        else                 -> rN = (A/Q)*rQ
    theta = (1-eta*wd)*theta - eta*r_method

性质声明（不声称创新）：
- rQ 读出与 QHAdamW(nu1=.9, nu2=1) 的读出结构同族（Ma & Yarats, ICLR 2019;
  qhoptim/pyt/qhadam.py）；范数匹配属于 grafting 家族（Agarwal et al. 2020）。
- 不新增全维持久状态；不回写 rQ/rN 到 moments。
- alpha==0 不走本模块步进 —— runner 必须调用原生 torch.optim.AdamW 路径。

数值纪律：任何输入/中间量非有限 -> NumericalFailure（不 fallback 吞掉）。
"""
from __future__ import annotations

import math

ALPHA = 0.10            # 冻结
CANCEL_EPS = 1e-12      # 冻结 cancellation 回退阈值系数


class NumericalFailure(RuntimeError):
    """输入/中间/输出出现 NaN/Inf。携带证据。"""

    def __init__(self, where: str, name: str = ""):
        self.where = where
        self.name = name
        super().__init__(f"NUMERICAL_FAILURE at {where} param={name}")


def _finite(t) -> bool:
    import torch
    if torch.is_tensor(t):
        return bool(torch.isfinite(t.detach()).all().item())
    return bool(t == t and t not in (float("inf"), float("-inf")))


def adamw_moment_update_(m, v, c, beta1: float, beta2: float):
    """与 torch.optim.AdamW 相同的矩更新（in-place）。

    m <- beta1*m + (1-beta1)*c ; v <- beta2*v + (1-beta2)*c^2
    """
    m.mul_(beta1).add_(c, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(c, c, value=1.0 - beta2)


def bias_corrected(m, v, t: int, beta1: float, beta2: float):
    """mhat, vhat（新张量）。t 为 true_parameter_step（>=1）。"""
    bc1 = 1.0 - beta1 ** int(t)
    bc2 = 1.0 - beta2 ** int(t)
    return m / bc1, v / bc2


def precondition(vhat, eps: float):
    """P = 1/(sqrt(vhat)+eps)（新张量）。"""
    import torch
    return 1.0 / (torch.sqrt(vhat) + float(eps))


def r_a(mhat, P):
    """rA = P*mhat。"""
    return P * mhat


def r_q(rA, Pc, alpha: float = ALPHA):
    """rQ = (1-alpha)*rA + alpha*(P*c)。"""
    return (1.0 - alpha) * rA + alpha * Pc


def norm_match(rA, rQ, Pc, alpha: float = ALPHA, cancel_eps: float = CANCEL_EPS):
    """逐唯一张量范数匹配。返回 (rN, fallback)。

    fallback ∈ {"NONE","ZERO","CANCELLATION"}。alpha==0 调用方不应进入本函数
    （runner 走原生路径）；此处保留分支供测试与防御。
    """
    if float(alpha) == 0.0:
        return rA, "NONE"
    A = float(rA.norm())
    Q = float(rQ.norm())
    B = float(Pc.norm())
    if not (math.isfinite(A) and math.isfinite(Q) and math.isfinite(B)):
        raise NumericalFailure("norm_match_input", "")
    if A == 0.0:
        return rA, "ZERO"
    if Q <= float(cancel_eps) * max(A, B):
        return rA, "CANCELLATION"
    return (A / Q) * rQ, "NONE"


def qh_adamw_step(optimizer, alpha: float, mode: str, collect: bool = False):
    """对 optimizer 执行一步 QH 读出 AdamW（grad 已 unscale+clip）。

    mode ∈ {"RAW","NORM"}。与原生 AdamW 相同的矩更新/bias 校正/P/decay 顺序；
    仅数据更新读出换为 rQ 或 rN。grad=None 的参数整体跳过（原生语义，不做 decay）；
    新增零持久状态；返回 per-param 诊断 dict 列表（collect=True 时）。

    非有限输入/中间量 -> NumericalFailure（调用方终止该轨迹并保存证据）。
    """
    import torch
    if float(alpha) == 0.0:
        raise ValueError("alpha==0 必须走原生 AdamW 路径，不得调用本函数")
    if mode not in ("RAW", "NORM"):
        raise ValueError(mode)
    group = optimizer.param_groups[0]
    beta1, beta2 = float(group["betas"][0]), float(group["betas"][1])
    eps = float(group["eps"])
    wd = float(group["weight_decay"])
    eta = float(group["lr"])
    rows = []
    for p in group["params"]:
        with torch.no_grad():  # 与原生 optimizer.step() 一致：参数原地更新不回传
          c = p.grad
          c = p.grad
          if c is None:
              continue  # 原生语义：无梯度参数整体跳过
          st = optimizer.state[p]
          step_t = st["step"]
          if torch.is_tensor(step_t):
              step_t.add_(1)
              t = int(step_t.item())
          else:
              st["step"] = step_t + 1
              t = int(st["step"])
          m, v = st["exp_avg"], st["exp_avg_sq"]
          if not (_finite(c) and _finite(m) and _finite(v)):
              raise NumericalFailure("pre_moments", getattr(p, "_d1_name", ""))
          adamw_moment_update_(m, v, c, beta1, beta2)
          mhat, vhat = bias_corrected(m, v, t, beta1, beta2)
          P = precondition(vhat, eps)
          rA = r_a(mhat, P)
          Pc = P * c
          rQ = r_q(rA, Pc, alpha)
          if not (_finite(mhat) and _finite(vhat) and _finite(rA) and _finite(Pc)
                  and _finite(rQ)):
              raise NumericalFailure("readout", getattr(p, "_d1_name", ""))
          fallback = "NA"
          if mode == "RAW":
              r = rQ
          else:
              r, fallback = norm_match(rA, rQ, Pc, alpha)
          if not _finite(r):
              raise NumericalFailure("r_method", getattr(p, "_d1_name", ""))
          p.mul_(1.0 - eta * wd)
          p.add_(r, alpha=-eta)
          if not _finite(p):
              raise NumericalFailure("theta_next", getattr(p, "_d1_name", ""))
          if collect:
              A = float(rA.norm())
              Q = float(rQ.norm())
              N = float(r.norm())
              dot_ar = float((rA * r).sum())
              cos_ar = dot_ar / (A * N) if A > 0 and N > 0 else float("nan")
              crA = float((c * rA).sum())
              crN = float((c * r).sum())
              rows.append({
                  "A_norm": A, "Q_norm": Q, "rN_norm": N,
                  "norm_resid": abs(N - A),
                  "dot_rA_r": dot_ar, "cos_rA_r": cos_ar,
                  "c_dot_rA": crA, "c_dot_rN": crN,
                  "c_dot_diff": crN - crA,
                  "c_norm": float(c.norm()), "Pc_norm": float(Pc.norm()),
                  "fallback": fallback,
                  "_tmp": {"P": P, "rA": rA, "Pc": Pc, "rQ": rQ, "r": r},
              })
    return rows


def make_step(optimizer, alpha: float, mode: str):
    """更新路由：alpha==0 -> 原生 optimizer.step（精确原生语义，含 decay 与
    grad=None/zero 处理）；alpha>0 -> QH 步进闭包。runner 一律经此入口。
    """
    if float(alpha) == 0.0:
        return optimizer.step
    def _step():
        return qh_adamw_step(optimizer, alpha, mode)
    return _step


def native_reference(c, m_pre, v_pre, t: int, beta1: float, beta2: float,
                     eps: float):
    """只读参考：从 (c, m_pre, v_pre) 计算本步 rA（供 native 臂诊断）。

    不修改任何状态；用于 ADAMW 臂在诊断点的同状态 rA 参照（它是该臂本状态的
    native-formula reference，不是另一条真实 native 轨迹的量）。
    """
    import torch
    m = m_pre.clone()
    v = v_pre.clone()
    adamw_moment_update_(m, v, c, beta1, beta2)
    mhat, vhat = bias_corrected(m, v, t, beta1, beta2)
    P = precondition(vhat, eps)
    return r_a(mhat, P), mhat, vhat, P


def temp_bytes_estimate(named_shapes, dtype_bytes: int = 4) -> int:
    """每步临时张量字节估计：P, mhat, vhat, rA, Pc, rQ, rN 各一份。"""
    n = sum(int(s) for s in named_shapes)
    return 7 * n * int(dtype_bytes)
