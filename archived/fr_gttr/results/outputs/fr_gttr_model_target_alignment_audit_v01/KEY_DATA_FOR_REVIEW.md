# KEY DATA FOR REVIEW — 审计关键数据速查

（供人工复核；全部数字可溯源至列出的输出文件）

## 1. 覆盖判定分布（25 claims）

| 判定 | 数 |
|---|---|
| EXACT_FOR_IMPLEMENTATION | 2 |
| EMPIRICALLY_COVERED | 16 |
| NOT_SUPPORTED | 4 |
| NOT_TESTED | 4 |
| PROTOCOL_MISMATCH（已勘误） | 1 |

## 2. 精确恒等式残差

| 恒等式 | 残差 |
|---|---|
| 记账 fp64 | ≤2.8e-20 |
| 聚合门独立复算 | 68/68 行吻合 |
| 统计充分统计 | 48/48 行 diff=0.0 |
| C0 恒等残差（readout） | 2e-7 |

## 3. 核心经验数字

| 实验 | 关键数字 | 文件 |
|---|---|---|
| A0C/A0F2 C模型 rel err | 0.0045-0.0219 (H≤8) | a0_fr_audit |
| C beat B | 168/224 | a0f2 REPORT |
| A0O LR响应 | Pearson 0.9552 / sign 0.9688 / median rel err 0.225 | a0o |
| A0P 脉冲核 gate | FAIL Pearson 0.7314 | a0p |
| A0Q 控制图 | LR PASS 0.842; clip 7/12死; wd 2e-6 | a0q |
| A0S 可控性 | Spearman 0.762/0.818; direction 46/48+12/12 | a0s |
| A0T | H4 median +4.41e-4 sign 35/36; C==B 34/36 | a0t |
| A0U | AUC B-C p=1.2e-2 (C更差); ~1 bit | a0u |
| B0 | 14,208 updates（申报）; 假设一/二 SUPPORTED; 假设三 UNRESOLVED | b0 summary.json |
| B0.5 | 三假设全 NOT_SUPPORTED; m kernel ~12步（描述性） | b05 |
| B0.6 | k=2 LL -2261.0 vs phase-null p95 -2114.6; rho=0.996 | b06 |
| C0 | field NOT_SUPPORTED_IN_PILOT; C/Bp/t16=1.166 逐点 | c0 |
| C1 | K32 E R0.505/C0.398; 4/6 anchors | c1 |
| C1L | COMBINED6 field R0.4914/C0.3618 PASS 3t | c1_late |
| G3 | gain_clip +0.094 MECHANISM_CONFIRMED | generalization_g3 |

## 4. 预算台账（审计后）

| 项 | 申报 | 审计 |
|---|---|---|
| B0 TOTALS | 14,208 | 15,097（含 smoke 712）|
| B0 累计（含 B0.5+B0.6） | 17,577 | **18,289（超 18,000 cap by 289）** |
| 全部累计（至 C1L/G系列） | 约19,000 | 超 cap 更多 |

## 5. 必记边界（发布前逐条核对）

1. H≤8 预测（C模型） / H=4 控制（A0Q/A0S/A0T）——不得外推。
2. T1/T2/T3 反例成立——短期 gain ≠ 长期收益。
3. 全部读出为 LM NLL——无下游能力声明。
4. A0T/A0U 为 in-sample 同流——无 out-of-sample。
5. C1L t48 为重建态（step94/128）——bank 无该真实状态。
6. 无独立 final 测试集——v_eval@128 是最接近者但已参与选择。
7. clip 因果（G3）为单规模单配置——观察非定律。

## 6. Errata 清单

1. ERR-B0-LEDGER（本审计新发, HIGH）
2. C0X 聚合门勘误（已发, 维持）
3. C1L t48 重建态披露（已发, 维持）
