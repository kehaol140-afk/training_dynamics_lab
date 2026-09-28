# GTTR_EVIDENCE_UPDATE（C0X + C1 之后）

日期：2026-09-21。范围：C0X 离线规则复算 + C1 probe 验证。不进入 C2。

## 数学恒等式（全部逐 anchor/逐区间验证）

- e = e_bank + e_local 及其平方范数形式（cross 项保留）：恒等残差 ≤8.7e-19（fp64 舍入），8 anchor-channel 全部成立。
- K_V = ΔV + W_prev 行级记账、区间 gain = ΔW−ΔK 恒等：C0X 已验（≤1.4e-17）；h=65 行不存在，末区间 [16,63]。
- mean(U(g)) ≠ U(mean(g))：CPU 测试 T4（clip+√v 非线性，rel>1e-3）——逐 probe 非线性更新不可换序，C1 全程逐 probe 计算。
- first16 半库 ≡ K=16 前缀（同一固定 ID 集合），非独立重复。
- 局部 c→U 导数 D_i 与 shadow U 的有限差分一致（T7，tol 1e-7）。

## 已测经验支持

- **field 通道一阶几何运输（g+Hs 再 clip）跨库预测**：在已测 4-anchor 集（A/t0、Bp/t16 × R/C）上通过全部冻结门——K32 聚合 E：R 0.505、C 0.398；半库 E≤0.700；每 cell 2 个不同 t 可分辨 → LOCAL_SUPPORT_ON_SELECTED_ANCHORS（GTTR Geometry/Transport 段的最强正向证据）。
- **同 probe 近似**：8/8 anchor-channel 优于 M0；t0 处 field 误差仅目标的 6.6–10.3%，跨库差是同 probe 差的 2–20 倍——独立库均值预测的误差主体是 probe 间波动。
- **v 通道承载实信号**（stop-v ablation）：冻结 P 使 shadow 增量范数增大 11–33%，‖Z_full−Z_stop‖ 达 full 范数 0.5–1.0 倍，方向结构保持。
- **AdamW 耦合方向依赖**：D_i 在主读出方向投影 C/A/t0 +0.158、R/A/t0 +0.153、C/Bp/t16 +0.050、R/Bp/t16 −0.064（反号）。
- **K 增大不单调改善**（R field 0.68→0.70→0.50），符合事先声明；K=32 相对 K=8 的改善稳定。
- 数值可重复性：same-probe floor 四量全 0.0；HVP 代理截断 ~4.5e-3（<1e-2 门）。

## 尚不分辨

- **shadow 通道跨库预测**：E 门值全达标（K32：R 0.667、C 0.585；半库 ≤0.98），但 t0 处目标在所有 K≤32 低于 3SE 下限（比值 ≈2.8–2.9），"≥2 distinct t 可分辨"前置不成立 → UNRESOLVED（当前测量未解决，非理论无效）。
- e_local 在干预臂（Bp/t16）增大到目标的 25–52%：有限位移截断 vs HVP 代理截断不可分离。
- D_i 投影反号的机制；stop-v 的逐 probe 方向余弦（只有均值级统计）。
- A/t48（late-time）局部预测是否仍支持；与 A0U 延迟收益区（t≥48 gain 才兑现）直接相关。

## 被特定实验削弱的假设

- "C0 field 局部几何在试点中不支持"——C0X 证明该结论的依据是规则误用（逐点替聚合）；C1 在独立新库上给出 field 支持的正面证据。原 C0 verdict 保留作历史记录，但"不支持"作为科学结论已被削弱。
- "半库失败由单点 1.166 代表"——被 C0X 证伪（1.166 是单点比值；聚合门当时不可求值）。
- "冻结 v 无害/无碍"——stop-v 显示幅度系统性改变 11–33%（方向保持）；不支持把局部增量当作与预条件无关的量。
- 隐含于早期读法的"误差主要来局部模型缺陷"——被分解实验削弱：跨库波动（e_bank）在 t0 处是局部缺陷（e_local）的 2–20 倍。

## 未检验

- future_H16_H64_prediction / adaptive_training_value / energy_benefit：NOT_TESTED（协议固定）。
- tracking 的条件化（mu/nu）本身；clip-Jacobian 因子进入 D_i；任何 controller/optimizer 候选。
- A/t48 两 anchor（本轮预算算术排除，见 REPORT §6）。

## 下一步唯一最有价值的问题

**局部几何支持是否在 gain 兑现的后期区间仍然成立？** 即在 A/t48（A0U 延迟收益区，t≥48）重复同一冻结测量：若 field E 仍 ≤0.8 且 shadow 分辨改善，GTTR 的 Geometry/Transport 段从"早期局部支持"升级为"跨训练阶段支持"；若 e_local 随干预增益继续增长，则一阶运输的适用边界被划在干预臂上——两者都是对现有支持的决定性检验。成本已知：2 anchor × 64 HVP + 37 步真实更新，可在新的一轮预算内完成（本轮余量 112 HVP 不足以在 cap 内并入，需独立成轮）。
