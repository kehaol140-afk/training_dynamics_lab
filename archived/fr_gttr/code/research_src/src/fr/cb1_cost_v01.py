"""GTTR-CB1 成本读出与进展联系 v1 —— 纯函数（CPU）。

协议：FR_GTTR_CB1_Cost_Measurement_And_Progress_Link_v1.md §1–§2、§6、§9。

- 达标成本 = 存活指示量的 token 加权和：C_capped = Σ (N_j-N_{j-1})·A_{j-1}；
  逐路径应与 direct capped hit 相等（已确认 -> N(T)；未确认 -> 预算上限，
  censor=True）。
- OLD_GRID16_LAG16：T16 = min{t∈{16,32,...}: V(t)<=ell ∧ V(t-16)<=ell}。
- REFINED_GRID4_LAG16：T4lag16 = min{t∈{16,20,24,...}: V(t)<=ell ∧
  V(t-16)<=ell}。确认间隔恒为 16 个 optimizer updates，不是连续两次 4 步。
- gap/rho 为 past-only 最低阶无拟合诊断；rho<=0 -> NO_POSITIVE_PAST_PROGRESS，
  不加 epsilon；保留未截断预测，超 256 标 FORECAST_BEYOND_BUDGET。

token 主口径 = train input slots（每 optimizer update 1024）；
valid loss tokens 单列。重放计算是研究成本，不进训练 token 轴。
"""
from __future__ import annotations

import math

SLOTS_PER_STEP = 1024
BUDGET_H = 256


def alive_capped_cost(means, ell, grid, slots_per_step=SLOTS_PER_STEP):
    """独立计算 alive 加权和成本与 direct 成本；返回残差。

    means: {h: V(h)}（全精度）；grid: 升序检测网格（含 0 起点则 alive 计数
    从 0 开始；本实现要求 grid 覆盖检测点）。
    A_{j-1}=1 表示到 t_{j-1} 仍未确认（确认发生在某网格点 t 上时，该点
    自身 A=0，其前一区间仍计费）。
    """
    hs = sorted(grid)
    h_det = detect_lagN(means, ell, grid, lag=16)["h_detect"]
    cost_alive = 0.0
    for j in range(1, len(hs)):
        t_prev, t_cur = hs[j - 1], hs[j]
        confirmed_before = (h_det is not None and t_prev >= h_det)
        if not confirmed_before:
            cost_alive += (t_cur - t_prev) * slots_per_step
    if h_det is not None:
        cost_direct = float(h_det * slots_per_step)
        censor = False
    else:
        cost_direct = float(hs[-1] * slots_per_step)
        censor = True
    return {"cost_alive": cost_alive, "cost_direct": cost_direct,
            "residual": cost_alive - cost_direct, "h_detect": h_det,
            "censor": censor}


def detect_lagN(means, ell, grid, lag=16):
    """min{t∈grid, t>=lag: V(t)<=ell ∧ V(t-lag)<=ell}（<= 含等号）。

    t-lag 不在 means 中 -> KeyError（不允许插值/跳过；非单调路径必须全网格
    扫描找最早合法 pair）。
    """
    best = None
    pair = None
    for t in sorted(grid):
        if t < lag:
            continue
        if t not in means:
            continue
        if t - lag not in means:
            raise KeyError(f"missing V({t - lag}) for lag pair at t={t}")
        if means[t] <= ell and means[t - lag] <= ell:
            if best is None or t < best:
                best = t
                pair = (t - lag, t)
    return {"h_detect": best, "pair": pair,
            "status": "DETECTED" if best is not None else "NOT_REACHED"}


def old_grid16(means, ell):
    """OLD_GRID16_LAG16（与 R2B detect_target 同规则；交叉核对用）。"""
    grid = list(range(0, BUDGET_H + 1, 16))
    return detect_lagN(means, ell, grid, lag=16)


def refined_grid4_points(t_max):
    """4 步网格 t∈{0,4,...,t_max}（用于 T4lag16 求值与补点计划）。"""
    return list(range(0, int(t_max) + 1, 4))


def new_points_to_measure(t16):
    """需新测的 4 步点：[8, T16] 内非 16 倍数（0/4/16 倍数用旧同轴值）。"""
    return [t for t in range(8, int(t16) + 1, 4) if t % 16 != 0]


def gap_rho(v_l_minus, v_l, ell, n_l, n_l_minus):
    """past-only gap/rho（只用 landmark 及之前的目标值）。

    rho = (V(l-16)-V(l))/(N(l)-N(l-16))；n 差必须 >0。
    """
    dn = float(n_l) - float(n_l_minus)
    if dn <= 0:
        raise ValueError("non-positive slot denominator")
    gap = float(v_l) - float(ell)
    rho = (float(v_l_minus) - float(v_l)) / dn
    return {"gap": gap, "rho": rho}


def predict_times(l, gap, rho, grid_step, budget=BUDGET_H):
    """完全预定义的检测预测（无拟合）。

    t_pred_first = grid_step*ceil((l+gap/(1024*rho))/grid_step)
    t_pred_confirm = t_pred_first + 16
    rho<=0 -> abstain。保留 raw 未截断值；超预算标 FORECAST_BEYOND_BUDGET。
    """
    if rho <= 0:
        return {"abstain": "NO_POSITIVE_PAST_PROGRESS"}
    raw_cross_after_l = gap / (SLOTS_PER_STEP * rho)
    t_first = grid_step * int(math.ceil((l + raw_cross_after_l) / grid_step))
    t_confirm = t_first + 16
    return {"abstain": None, "raw_cross_steps_after_l": raw_cross_after_l,
            "raw_cross_tokens_after_l": raw_cross_after_l * SLOTS_PER_STEP,
            "t_pred_first": t_first, "t_pred_confirm": t_confirm,
            "predicted_remaining_tokens": (t_confirm - l) * SLOTS_PER_STEP,
            "forecast_beyond_budget": t_confirm > budget}


def first_stream_of(streams):
    """固定规则：取数值 ID 较小的一条 stream（预算范围，非择优）。"""
    return min(int(s) for s in streams)


# ---------------------------------------------------------------------------
# 记账分离（§9-7：训练 token 轴 vs 重放研究成本）
# ---------------------------------------------------------------------------

def trajectory_slots(h):
    """轨迹 elapsed-training-token 轴：N(h) = h*1024 input slots。"""
    return int(h) * SLOTS_PER_STEP


def replay_research_cost(replay_updates, replay_monitor_evals,
                         micro_per_update=2, chunks_per_eval=128):
    """本轮重放的研究计算成本（不进训练 token 轴）。"""
    return {"replay_updates": int(replay_updates),
            "replay_train_fw": int(replay_updates) * micro_per_update,
            "replay_monitor_fw": int(replay_monitor_evals) * chunks_per_eval,
            "replay_monitor_evals": int(replay_monitor_evals)}
