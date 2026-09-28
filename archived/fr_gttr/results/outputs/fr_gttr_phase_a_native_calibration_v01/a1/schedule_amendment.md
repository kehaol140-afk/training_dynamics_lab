# Schedule Amendment — NATIVE_WITH_DECLARED_TAIL（本轮明文选择）

日期：2026-09-22（a1 执行轮）。依据：用户启动指令明文 + Roadmap §3.2。

## 冻结声明

- `schedule_id = finite_cosine_then_hold_5e-5_v01`
- `label = NATIVE_WITH_DECLARED_TAIL`

```
eta(gm) = old_get_lr(gm, 256, 5e-4),  gm <= 256
eta(gm) = 5e-5,                        gm >  256
```

- `gm` 与旧 runner 真实传入索引一致：micro-step 索引；accum=2，AdamW 在
  每个 micro-batch 前由 `set_native_micro_lr` 写入，真正消费 pair 最后一值。
  现场核实：入口 runner `research/src/fr/a0k_closed_loop_runner_v01.py`，
  scheduler `minimind/trainer/trainer_utils.py:40-41`。
- 单参数组（bank `optimizer_signature` 与 `assert_native_adamw` 双核实），
  wd=0.01 应用于全组；无 decay/no-decay 组比例问题，尾部值 5e-5 直写该组。
- 不重 warmup、不重置 moments/bias correction/累计 step/RNG；不把
  total_micro 从 256 改 512；不新增 cooldown/峰值保持。
- 不使用 `max(floor, old_get_lr)`（不能阻止越界回弹）；本尾部的语义是
  gm>256 后直接返回常数，旧公式不再被调用。
- 只通过新 PhaseA 入口 `fr.phase_a_tail_schedule_v01.make_phase_a_get_lr`
  包装；旧 `get_lr` 函数与全部历史输出保持不变。
- `0.1*base_lr = 5e-5` 是有限 cosine 的最小值；本尾部令其在 gm>256 后
  保持，边界处连续（gm=256 处旧公式值=5e-5，见 schedule.csv）。

## 与 BLOCKED 前态的关系

2026-09-22 早前的 PhaseA 预检（../preflight.md）按协议 §4 case C 判定
SCHEDULE_BLOCKED，并列出了 4 个候选问题。本轮用户以明文选择了其中的
选项 2（声明 tail = 冻结 LR 于 5e-5），阻塞解除。原 BLOCKED 记录保留未动。

## 各 cell 消费 LR（节选，全表见 schedule.csv）

| start | h=0 (gm=2·start) | 到达 5e-5 的 h | h=256 |
|---|---|---|---|
| 46 | 3.712e-4 | h=82（gm=256 恰为 cosine 终点） | 5e-5 |
| 82 | 1.788e-4 | h=46 | 5e-5 |

start=46 的轨迹沿 cosine 下降至 floor 后保持；start=82 在 floor 之后起步。
两条起点曲线不是同一 schedule 段，token/LR 历史不可混算。
