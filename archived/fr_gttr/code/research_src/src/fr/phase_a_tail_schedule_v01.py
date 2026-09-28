"""Phase A native calibration 的声明式 schedule 尾部（新入口包装）。

schedule_id : finite_cosine_then_hold_5e-5_v01
label       : NATIVE_WITH_DECLARED_TAIL

定义（用户明文选择，2026-09-22 冻结）：
    eta(gm) = old_get_lr(gm, 256, 5e-4),  gm <= 256
    eta(gm) = 5e-5,                        gm >  256

- gm 与旧 runner 真实传入索引一致（micro-step；accum=2 时 AdamW 在每个
  micro-batch 前被 set_native_micro_lr 写入，真正消费 pair 最后一个值）。
- 不重 warmup、不重置 moments / 累计 step / RNG；不把 total_micro 改为 512；
  不新增 cooldown；不使用 max(floor, old_get_lr)（无法阻止越界回弹）。
- 单参数组现场已核实（bank optimizer_signature 与 assert_native_adamw），
  尾部值直接写入该组。
- 本模块只在新 PhaseA 入口被调用；minimind/trainer/trainer_utils.py 的
  get_lr 原函数与全部历史输出保持不变。
"""
from __future__ import annotations

SCHEDULE_ID = "finite_cosine_then_hold_5e-5_v01"
SCHEDULE_LABEL = "NATIVE_WITH_DECLARED_TAIL"
TAIL_LR = 5e-5
TOTAL_MICRO_STEPS = 256
BASE_LR = 5e-4


def make_phase_a_get_lr(old_get_lr):
    """构造本轮 PhaseA 入口使用的 learning-rate 函数。

    返回 callable(current_step, total_steps, lr) -> float，签名与旧 get_lr
    兼容，可直接传给 fr.native_schedule.set_native_micro_lr。
    total_steps / lr 仅接受冻结值（256 / 5e-4）；传其他值会报错而不是
    静默拉伸 schedule，避免把 horizon 输入用于重新缩放。
    """

    def eta(current_step, total_steps, lr):
        if int(total_steps) != TOTAL_MICRO_STEPS:
            raise ValueError(
                f"PhaseA tail schedule 冻结 total_micro={TOTAL_MICRO_STEPS}，"
                f"收到 {total_steps}")
        if float(lr) != BASE_LR:
            raise ValueError(
                f"PhaseA tail schedule 冻结 base_lr={BASE_LR}，收到 {lr}")
        gm = int(current_step)
        if gm <= 0:
            raise ValueError("global_micro_step 必须从 1 开始")
        if gm <= TOTAL_MICRO_STEPS:
            return float(old_get_lr(gm, TOTAL_MICRO_STEPS, BASE_LR))
        return float(TAIL_LR)

    eta.schedule_id = SCHEDULE_ID
    eta.schedule_label = SCHEDULE_LABEL
    return eta
