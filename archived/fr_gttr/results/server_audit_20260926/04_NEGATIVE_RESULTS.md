# 04 — 负结果全集（含作废/推翻结论）与每条排除的方向

> 本项目负结果密度极高且被制度性保留。每条注明：verdict、证据、**它排除了什么未来方向**。

## N1. 原始涨落—响应比例关系失败（E3 系）

- E3 theta mapping v05：H=8 Pearson −0.083。E3-R v06：LOSO mean −0.0175。
- **E3C confirmatory v07：H8_CONFIRMATION = FAIL**（LOSO 5/8 < 预注册 6/8）。RAW_VERIFIED
- 排除方向：**直接类比物理 FR 定律的"普适比例常数"路线**。项目因此转向机制分解（A0C），这是全项目最重要的一次转向。
- 残留事实（不同口径并存，非矛盾）：within-seed 去趋势 H=8 r=0.5406（circular-shift p=1.5e-4）；pooled 448 行 R²=0.0008。

## N2. 旧 A0E "raw curvature 有害" —— 已作废（BH 初始化 bug）

- 旧 A0E gain_H=−0.1357 的根源：BH 初始脉冲用裸 q 跳过 J_clip(g_t)（H=1 即与 B/C 不一致）。
- A0E2 修复后 raw curvature 为**正贡献 +0.0686**；旧值 −0.204352 全来自初始化污染。RAW_VERIFIED
- 排除方向：**"Hessian 反馈有害"的错误结论**（正确表述：裸 HVP 不经 clip-JVP 有害；经精确 clip Jacobian 后有益）。
- 列入 `docs/FR_PROJECT_STATUS.md` §14 禁引清单第 1–3 条（含 9 行/12 行口径混淆）。

## N3. mv_theory v 路径 —— 作废

- mv_theory_final 声称的 v 路径改善被 mv_theory_consistent（严格 q/梯度语义）否定：0/8 seed、0/8 horizon 改善；
  chi_mv R²=−6.33；"AdamW m+v sufficient? No"（R²=−409.4）；ν 反常（−1.3e6 量级）来自冻结 raw-v 近似假象。RAW_VERIFIED
- 排除方向：**闭式 m/v 双指数近似理论**（作为机制解释）；保留其作为零级对照的地位。

## N4. 响应计算无墙钟成本优势（A0G/A0I/A0K）

- A0G：H=4 M=4 shared-tangent-C 6.77s vs direct replay 2.65s（0.39×）；H{1,4,8}×M{1,2,4} speedup 全 <1；break-even 扫描 M≤64 全 None。RAW_VERIFIED
- A0I-B：候选评分子阶段 −25~−30%、显存 −17%，但总决策成本 SIMILAR（shared prefix ~1s 主导）。RAW_VERIFIED
- A0K：ROI median ~5e-5 nats/s。RAW_VERIFIED
- 排除方向：**"response 比回放便宜"的工程路线（5M 规模）**。研究价值只能来自回放不可行的场景（大模型/大候选池）。

## N5. 选择器不优于固定动作（四重独立收敛：A0K M5 / A0I-A / A0L / A0N）

- A0K M5：adaptive vs fixed-j1 median **−1.29e-4**（C≤B 6/9 对）。A0I-A：ΔGain +5.76e-5 但 seed 高度异质（−1.8e-4..+3.2e-4）。
- A0L：B−C median −7.46e-3（固定混合 ≥ response controller，2/3 seed）。A0N：B−C median −6.78e-4。全部 RAW_VERIFIED
- 排除方向：**"response 智能选择器/自适应控制器"（pilot 规模 + 当前动作集）**。任何未来控制器必须先打赢自己的 fixed-action 消融 + equal-wall-clock。

## N6. LR 脉冲核无稳定时间尺度（A0P —— 项目最清晰护栏）

- Stage-1 gate FAIL：pooled Pearson +0.7314 < 0.8（sign 0.9405 通过）；Q1 NO（best R² 0.351、CV(τ_c)=3.32；
  exp R² 0.071 / power 0.306 均不可用）；核形=快初始衰减（半衰期 4–8 步）+长记忆噪声尾（|R|~2e-3 不收敛）。RAW_VERIFIED
- 排除方向：**短期响应→长期训练收益的一阶外推**；任何 kernel/记忆核拟合（A0C 也明令禁止用有限 ε 残差拟合记忆项）。

## N7. 可利用信息 ≈ 1 bit；控制器身份=全局缩放（A0T-extra / A0U）

- A0T-extra：P(+0.05)=0.944；熵 0.310 bit；2 个可判别 cell 的状态依赖反转在 H=16/64 反转为有害。RAW_VERIFIED
- A0U：response schedule ≈ 延迟 6 步的固定 ×1.05；AUC 上 C 显著更差（p=1.2e-2）；冻结规则 K=4 窗从未兑现。RAW_VERIFIED
- 排除方向：**单符号 bank 上的一切"自适应"叙事**。未来自适应方法必须先建立多符号、horizon-persistent 的 χ bank。

## N8. 缓存 future sensitivity 无效（A0J）

- chosen-id 保持 5/6→4/6→1/6（age 1/2/4）；dGain 全负；根因：exact margin median 3.8e-5 vs age1 误差 ~2e-3（差 2–3 个量级）；
  cosine 0.955 与 argmin 翻转共存（cosine 不得作 gate）。RAW_VERIFIED
- 排除方向：**按固定 cadence 摊销伴随/sensitivity 的路线**。若重访只能做 drift 触发 refresh 或状态表示压缩（需新预注册）。

## N9. G1c 1B：C 分支失败（scale-dependent limitation）

- Phase2 首个 cell：B 切线稳定（χ −1.85→−10.7），C 分支 h=7 NaN（wte 冻结行 guard）。四分类裁决：主因 C（scale-dependent），
  近因 B（fp16 溢出）；A（实现 bug）/D（optimizer 异常）排除。机制：1B 罕见 token 的精确零 exp_avg_sq 坐标 → AdamW 分母落 eps 地板（1e-4）→
  逐坐标增益 ~1e4 → 深剪裁 regime（0.003–0.012）下 C 递推超线性放大。26M/100M 同协议 O(1)。RAW_VERIFIED
- 排除方向：**"完整切线机制自动外推到 1B"**；外部 ~1B 轴保持未覆盖（须先修 eps 地板/罕见 token 处理再试）。

## N10. 无亚稳态 regime（B0.6）；记忆核 12 步内衰减（B0.5）

- B0.6：k=2 HMM LL 不击败 phase-null；persistence 方向与亚稳态相反；local vs global 一步映射无差（0.2961 vs 0.2956）；
  across-transition 测试步 n=0/360；谱半径 0.996–1.013。headline NOT_TESTED（Q1 NOT_SUPPORTED）。RAW_VERIFIED
- B0.5：H1 时间重参数化 0/6；tau_fast<<tau_slow 6/6 反向；m/v 记忆携带 share<0.2（5/6）。RAW_VERIFIED
- 排除方向：**离散 regime 切换修正、Markov 记忆核、时间重参数化**（64 步/4.98M/9 维 ψ_t 投影范围内）。长期失效的解释余量收窄到：
  窗外连续非平稳 / 一步线性未捕获的非线性 / ψ 投影外变量。

## N11. C0 局部几何 pilot 不支持（后被 C0X 勘误削弱 + C1 翻转为正面）

- C0：field 通道 half-bank 规则失败（E=1.166>1）→ LOCAL_GEOMETRY_NOT_SUPPORTED_IN_PILOT；shadow UNRESOLVED。RAW_VERIFIED
- C0X 复算：**判定器把聚合半库门实现成逐点门（ERRATUM_REQUIRED）**；按原规则复算聚合 E R=0.529/C=0.468 通过。RAW_VERIFIED
- 净结论：C0 的"不支持"被削弱；C1 在独立库给出 field 正面支持（见 P10）。原始 verdict 保留不覆盖。
- 排除/修正：**分析器实现纪律**——冻结规则必须按原文复算（这也是 C0X 的教训本体）。

## N12. GTTR 线的当前失败链（PhaseB → D1 → R2B → CB1 → F0 → F2）

- PhaseB v3：S3 局部算子 fd_rel=0.706 > 0.20 → **LOCAL_OPERATOR_NOT_CALIBRATED**（S0–S2 通过；FD diagnostic 证明非 bug）。RAW_VERIFIED
- D1：**NO_OBSERVED_GAIN**（预算终点 NLL 与目标 token 效应均无增益）；NORM 匹配劣于 RAW。RAW_VERIFIED
- R2B：一步效应 −2.88e-4 有利（9/12）但 **终态 NLL 未获益（8/12 为正=更差）**；反转 3/12。RAW_VERIFIED
- CB1：rho 预测剩余成本中位相对误差 0.20 → **不作可靠预测子**。RAW_VERIFIED
- F0：terminal 导数 SENS=−5060 vs FD≈−0.005（6 量级）→ **BLOCKED_AT_CALIBRATION**；
  CALFIX 修复 D3/D4 后 h256 仍未校准；F1 闭合后 b 坐标一致（−0.311 vs −0.345）但 **F2a 方向不确认**（sign 3/4，最优在 b=+0.02）、
  **F2b 二阶预测失败**（曲率全尺度负，J 最优在 +0.02 而非预测的 −0.0146）。RAW_VERIFIED
- 排除方向（累计）：**"局部几何场 → 未执行动作判别 → 终局规则求解"这条当前路径**在现有校准下不通；
  GTTR 的可用正证据收缩回 C1/C1L 的 field 一步输运层。

## N13. 数值精度类负结果

- bf16 math-SDPA HVP 代理转置不对称：A0H-1 C|H=8 逐元 gate 不可满足（噪声地板 5.78e-5 > 冻结容差）；
  A0H-1.5 H=8 同签名 FAIL。fp32-HVP 分支（B1）4.1e-10 通过 → 结论：地板属冻结管线而非伴随实现。RAW_VERIFIED
- ambient flash SDPA 无 forward-mode AD：A0I-B probe NotImplementedError → JVP 只能在 forced math backend。RAW_VERIFIED
- F1 Phase2 state sensitivity RELATIVE_DIVERGENCE：FP32 路径 FD 分辨限制（非实现缺陷）。RAW_VERIFIED

## N14. 工程/复现类事件（如实登记，不删除）

- theta_protocol.py 源码 09-12 被删（原因未查明），48 副本无匹配；以 pyc 恢复，行为位级等价。RAW_VERIFIED
- A0J implementation gate 首次 FAIL（模块 `_clip` 漏调 `pe._step_once`）→ 探针链定位修复后 30/30（教训：自检一致不够，生产复现行必要）。
- A0K B_wc 臂事件排除缺陷、A0L arm B off-by-one、A0O 校准符号取反（离线修正 +0.942）、A0U launch-1 设备序——全部"结果前修复+留痕"。
- A0H15 A′ 驱动 selector 方向 bug（argmax vs argmin）→ 3/3 FAIL 记录保留 + 决策语义审计 Branch A。
- 排除方向：**"只靠单元测试/自检 gate 就信任实现"** 的工程文化；本项目以生产复现行 + 位级 cross-check 补足。

## 负结果计数

- 科学负/护栏结果：N1–N13 共 13 类（对应 20+ 个实验条目）；
- 作废/推翻结论：2 项（旧 A0E、mv_theory_final）+ 2 项过强表述撤回（A0H15"位级一致"、A0H-1"H≥4 对任何伴随不可满足"）+ 1 项 C0 判定器勘误；
- 工程 bug 类（结果前修复留痕）：~10 起，全部有 hash/manifest 记录。
