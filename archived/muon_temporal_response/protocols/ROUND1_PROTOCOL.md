# ROUND1_PROTOCOL — 第一轮（T1+T0）口径备忘

日期：2026-09-28。本文件是本轮实际执行口径的简报；权威配置 = `configs/t0_round1.json`
（运行前写盘并含全部源文件 SHA），权威范围 = `references/` 三份源文件（冲突时以执行计划为准，
本轮未发现需要改写源文件的分叉，发现的对照差异登记于
`docs/IMPLEMENTATION_ALIGNMENT.md` §2.1）。

## 范围

- 只读平台核对（T1）+ CPU/单 GPU 小矩阵受控实验（T0）。
- 新增 Transformer 训练 / 真实 batch 梯度 / 父轨迹 / 确认集 / T2–T4 运行：全部 0。
- 旧 M2 `NO_SIGNAL_STOP_LINE` 原样保留；不重开配比搜索。

## 预算执行（实测 vs 上限）

CPU ≪ 30 min（合计 < 1 min 数值）；GPU 110.7 s / 15 min（单卡 cuda:0，空闲探测后使用）；
输出 2.2 MB / 1 GB；合成递推单独计数（FP64 67,072 + 原生 67,072 步）。

## 统计口径

- Dbar_t/A_j/ΔA/ratio/B_j(t) 定义见 config `statistics` 节；先跨相位中心化、再时间平均。
- 64 相位 = 有限枚举总体，不作独立种子检验。
- `delta_relative`（相对量）/ `epsilon_perturb`（主矩阵扰动绝对量 = delta·‖S‖F/√2）/
  `eps_norm`（安装 NS clamp eps=1e-7）三字段分离。
- v2 的 epsilon_perturb=0.001 绝对值旧例不进主矩阵（legacy_supplement 未重跑：原配置
  eps_norm 等不可完整复原，标记不可完整复算）。

## 产物

`docs/` 四份报告 + 本备忘；`outputs/t0_round1/` 全部数据/图/台账；源码 `src/`、
`scripts/`、`tests/`；复跑命令见 `docs/T0_REPORT.md` §6。
