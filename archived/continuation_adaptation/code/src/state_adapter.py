"""state_adapter — 实验臂的唯一定义与施加逻辑。

臂表（与 configs/smoke_v01.json 的 arms 表双向一致性断言）：

  A  : betas (0.9, 0.999) 原值不动，state 完全继承（强 baseline）
  B  : betas (0.9, 0.99)，state 完全继承（HP 迁移）
  C1 : betas (0.9, 0.99)，v 清零（m/step 继承）
  C2 : betas (0.9, 0.99)，m 清零（v/step 继承）
  D  : betas (0.9, 0.99)，m/v/step 全部归零（reset baseline）

规则（docs/EXPERIMENT_PROTOCOL.md §3 冻结）：
- 必须在 optimizer.load_state_dict（即 load_full_training_state）之后调用——
  load 会用 checkpoint 的 param_groups 覆盖 betas；
- 任务定义未提及的通道一律不动（C1/C2 的 step 保持 t0，其 bias-correction
  后果是定义的一部分，不属于实现问题）；
- 施加后立即回读验证（betas 生效、清零通道精确为 0、保留通道逐位不变）。
"""
from __future__ import annotations

from typing import Dict

from vendored_fr.hashing import stable_hash
from vendored_fr.native_adamw_guard import optimizer_signature

from optimizer_utils import reset_optimizer_moments, set_betas

ARM_SPECS: Dict[str, Dict] = {
    "A":  {"betas": (0.9, 0.999), "reset_m": False, "reset_v": False, "reset_step": False},
    "B":  {"betas": (0.9, 0.99),  "reset_m": False, "reset_v": False, "reset_step": False},
    "C1": {"betas": (0.9, 0.99),  "reset_m": False, "reset_v": True,  "reset_step": False},
    "C2": {"betas": (0.9, 0.99),  "reset_m": True,  "reset_v": False, "reset_step": False},
    "D":  {"betas": (0.9, 0.99),  "reset_m": True,  "reset_v": True,  "reset_step": True},
}


def _state_channel_hashes(optimizer) -> Dict[str, str]:
    """对 m / v / step 三个通道分别做稳定哈希（逐位变化的检测器）。"""
    import torch

    m_obj, v_obj, s_obj = {}, {}, {}
    for i, entry in enumerate(sorted(optimizer.state.values(), key=lambda e: id(e))):
        if not entry:
            continue
        m_obj[i] = entry["exp_avg"].detach().clone()
        v_obj[i] = entry["exp_avg_sq"].detach().clone()
        s_obj[i] = entry["step"].detach().clone() if torch.is_tensor(entry["step"]) else entry["step"]
    return {"m": stable_hash(m_obj), "v": stable_hash(v_obj), "step": stable_hash(s_obj)}


def check_arms_table_consistency(cfg_arms: Dict) -> None:
    """config 臂表与代码臂表（唯一定义）必须完全一致，否则中止。"""
    if set(cfg_arms) != set(ARM_SPECS):
        raise RuntimeError(f"臂表不一致: config={sorted(cfg_arms)} code={sorted(ARM_SPECS)}")
    for arm, spec in ARM_SPECS.items():
        c = cfg_arms[arm]
        for key in ("reset_m", "reset_v", "reset_step"):
            if bool(c[key]) != spec[key]:
                raise RuntimeError(f"臂 {arm} 字段 {key} 不一致: config={c[key]} code={spec[key]}")
        if [float(x) for x in c["betas"]] != [float(x) for x in spec["betas"]]:
            raise RuntimeError(f"臂 {arm} betas 不一致: config={c['betas']} code={spec['betas']}")


def apply_arm(optimizer, arm_name: str) -> Dict:
    """施加实验臂并返回验证记录。必须在 load_full_training_state 之后调用。"""
    import torch

    if arm_name not in ARM_SPECS:
        raise ValueError(f"未知实验臂: {arm_name}（合法: {sorted(ARM_SPECS)}）")

    spec = ARM_SPECS[arm_name]
    sig_pre = optimizer_signature(optimizer)
    hashes_pre = _state_channel_hashes(optimizer)

    # 1) betas（load 之后显式写回）
    set_betas(optimizer, spec["betas"][0], spec["betas"][1])

    # 2) state 通道清零
    reset_info = reset_optimizer_moments(
        optimizer,
        reset_m=spec["reset_m"],
        reset_v=spec["reset_v"],
        reset_step=spec["reset_step"],
    )

    # 3) 回读验证
    sig_post = optimizer_signature(optimizer)
    hashes_post = _state_channel_hashes(optimizer)

    checks = {"betas_applied": True, "reset_counts": reset_info}

    # 保留通道必须逐位不变
    for channel, reset_flag in (("m", spec["reset_m"]), ("v", spec["reset_v"]), ("step", spec["reset_step"])):
        if not reset_flag and hashes_pre[channel] != hashes_post[channel]:
            raise RuntimeError(f"臂 {arm_name} 不允许改动通道 {channel}，但哈希发生变化")

    # 清零通道必须精确为 0 / step 精确为 0
    n_zero_m = n_zero_v = n_zero_step = 0
    for entry in optimizer.state.values():
        if not entry:
            continue
        if spec["reset_m"]:
            if not bool(entry["exp_avg"].eq(0).all()):
                raise RuntimeError(f"臂 {arm_name}: exp_avg 清零失败")
            n_zero_m += 1
        if spec["reset_v"]:
            if not bool(entry["exp_avg_sq"].eq(0).all()):
                raise RuntimeError(f"臂 {arm_name}: exp_avg_sq 清零失败")
            n_zero_v += 1
        if spec["reset_step"]:
            t = float(entry["step"])
            if t != 0.0:
                raise RuntimeError(f"臂 {arm_name}: step 归零失败（={t}）")
            n_zero_step += 1

    got_betas = sig_post["param_groups"][0]["betas"]
    if tuple(float(x) for x in got_betas) != tuple(float(x) for x in spec["betas"]):
        raise RuntimeError(f"臂 {arm_name}: betas 回读不一致 {got_betas}")

    return {
        "arm": arm_name,
        "spec": {**spec, "betas": list(spec["betas"])},
        "signature_pre": sig_pre,
        "signature_post": sig_post,
        "checks": checks,
        "channel_hashes_pre": hashes_pre,
        "channel_hashes_post": hashes_post,
        "zeroed": {"m": n_zero_m, "v": n_zero_v, "step": n_zero_step},
    }
