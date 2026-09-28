# SOURCE_FIX_AUDIT — FR_A0E2 BH 对照修正源码审计

生成时间：2026-09-12 · 协议：`FR_A0E2_CURVATURE_DECOMPOSITION_V02_BH_FIXED`
输出目录：`outputs/a0_fr_audit/curvature_decomposition_v02_bh_fixed/`

本文件回答任务第二节（源码审计）与第十节（A0F/A0F2 非干扰），并记录本次审计中发现的
框架级文件缺失事件及处置。

---

## 1. 旧 A0E / A0C 中 B / BH / C 的 tangent 初始化路径

### 1.1 涉及文件与函数

| 对象 | 位置 |
|---|---|
| Model B / C 传播（冻结） | `research/src/fr/a0c_adamw_response_propagation_v01.py::adamw_tangent_step`（L160-276）、`clip_jvp`（L50-136） |
| B / BH / C runner 逻辑（冻结） | `research/scripts/a0e_curvature_decomposition_v01.py::_run_baseline_tangents`（L207-449） |
| BH 修正版 runner（本次新增） | `research/scripts/a0e2_curvature_decomposition_v02_bh_fixed.py::_run_baseline_tangents` |
| 修正版分析原语（本次新增） | `research/src/fr/a0e2_curvature_decomposition_v02_bh_fixed.py` |

### 1.2 初始 pulse q 在三个旧模型中的进入方式（冻结代码逐行核对）

初始步（`horizon == 1`）在旧 A0E runner 中的代码路径：

```
L285-286  if horizon == 1:
              full_gradient_tangent = q_device          # C 的 clip-JVP 输入；也被 BH 直接使用
L309      optimizer_gradient_tangent = q_device if horizon == 1 else zero   # B 的输入
L311-315  b_clip = clip_jvp(native_gradient, optimizer_gradient_tangent, ...)  # B: J_clip(g) q
L316-320  c_clip = clip_jvp(native_gradient, full_gradient_tangent, ...)       # C: J_clip(g) q
L324      bh_gradient_tangent = full_gradient_tangent   # BH: 裸 q，未经 clip-JVP
L348-356  bh_next = adamw_tangent_step(..., tangent_clipped_gradient_by_name=bh_gradient_tangent, ...)
```

逐模型回答任务问题 1–3：

1. **初始 pulse q 如何进入 delta_m / delta_v / delta_theta**：
   - 三个模型都通过 `adamw_tangent_step` 的 `tangent_clipped_gradient_by_name`（记 dc）进入
     tangent 更新：`dm1 = β1·dm + (1−β1)·dc`，`dv1 = β2·dv + 2(1−β2)·c·dc`，
     `dθ ← (1−lr·wd)·dθ − lr·[dm̂/√v̂ − m̂·d√v̂/v̂]`（完整 m/v/bias-correction/eps/wd）。
   - **B**：dc = `b_clip['clip_jvp']` = J_clip(g₁)·q（经全局裁剪 Jacobian）。
   - **C**：dc = `c_clip['clip_jvp']` = J_clip(g₁)·q（与 B 在 H=1 输入完全相同）。
   - **旧 BH**：dc = `full_gradient_tangent` = **裸 q**。
2. **clipping 激活时是否分别使用 J_clip(g_t)q**：B 和 C 是；**旧 BH 否**——这正是缺陷。
3. **旧 BH 在 H=1 已与 B/C 不一致的原因**：H=1 时尚无 future HVP，
   `BH ≡ B + future raw HVP` 退化为 `BH ≡ B`，理论要求三者恒等。但旧实现中
   `bh_gradient_tangent = full_gradient_tangent` 在 horizon==1 取值为裸 q，
   而 B/C 的 pulse 经过 `clip_jvp`。当初始步裁剪激活（`||g₁|| > 1`）时，
   `J_clip(g₁)q = c·q − radial·g₁`（c = 1/(||g₁||+1e-6)，radial 为正标量），
   与裸 q 的差为 radial·g₁ 项——不可忽略。
   数学对象层面：B/C 的初始 tangent 位于 clip 映射的 JVP 像空间（径向修正后），
   旧 BH 的初始 tangent 是裸脉冲方向；两者之差恰好是 clipping 的径向修正项。

### 1.3 冻结制品中的实证

- `curvature_decomposition_v01/summary.json`（冻结）H=1 行：
  seed_2037/step_046：`B = C = -0.05611986226622187`，`BH = -0.08155611474847858`；
  seed_2037/step_082：`B = C = -0.0280386989096241`，`BH = -0.02928530512198406`；
  seed_2030/step_046：`B = C = BH = -0.0655277072243726`（该状态初始步 clip 未激活）。
- stream 级（冻结）：seed_2037/step_046 stream 2 初始 preclip norm = 18.2689，clip 系数 0.0547，
  `B1 = C1 = -0.003690` vs `BH1 = -0.071910`；streams 0/1（clip 未激活）三者逐位一致。

## 2. BH_fixed 修正

`research/scripts/a0e2_curvature_decomposition_v02_bh_fixed.py`（新文件，L340-349 附近）：

```python
if horizon == 1:
    bh_fixed_gradient_tangent = b_clip["clip_jvp"]   # 与 B/C 完全相同的 J_clip(g_t) q
else:
    bh_fixed_gradient_tangent = full_gradient_tangent  # future raw HVP，不做 clip-JVP
```

- 初始步：BH_fixed 与 B、C 使用**同一份** `b_clip['clip_jvp']` 张量（b_clip 与 c_clip 在
  horizon==1 的输入逐位相同，输出逐位相同）。
- 未来步：与旧 BH 相同——裸 HVP 直接进入 gradient-tangent 通道，不做径向修正。
- 因此 BH_fixed 与 C 的唯一差异严格是未来步的 clip-JVP；与 B 的差异是 future raw curvature。
- B / C / BH_old 代码块保持冻结原样；BH_old 按原冻结路径保留在本 runner 中，
  用于同 run 内逐位复现旧制品（见 summary.json 的 `a0e_bh_old_reproduction`，max gap = 0.0）。

## 3. theta_protocol.py 缺失事件（框架级，只读审计发现）

- 2026-09-12 审计时发现 `research/src/fr/theta_protocol.py` 缺失。证据：
  - `research/src/fr/__pycache__/theta_protocol.cpython-311.pyc` 仍在，头部记录源文件
    mtime 2026-09-10 18:12:09、size **6341 bytes**；
  - 全部冻结制品（A0C/A0E/A0F2 等）的 `input_hashes['replay_primitives']` 记录该源文件
    SHA256 = `4a127209954b80864561314880482ecb7aaccc60970d21f4ff6cc4afe2a3422a`；
  - A0F2 于 2026-09-12 00:16 完成时该文件仍存在，故删除发生在 2026-09-12 00:16–19:00 之间；
  - 全文件系统搜索 48 个 `theta_protocol.py` 副本（含 v05/v06 快照、备份、VSCode 历史），
    无一个匹配冻结哈希；现存最相近者为 4359 字节旧版（2026-09-02）与 4887 字节
    VSCode 历史版（2026-09-10 14:06），均非 18:12 的 6341 字节冻结版。
- 处置（最小干预、不改变任何现有文件）：将该 pyc 复制为
  `research/src/fr/theta_protocol.pyc`（SHA256 `61e1bae38a2b20d0407e134f215094060624178da215d2232c94739d1d7751e7`），
  由 Python 标准 sourceless loader 解析。该字节码正是冻结哈希源文件在 2026-09-10 18:12:24
  编译的产物，A0E（09-11 21:33）与 A0F2（09-12 00:16）均以其运行。
- 行为等价性端到端验证：本 run 中 `assess_baseline_replay` 要求 H=8 shadow 分支
  （经 `run_theta_branch`）与主 replay 的 model/optimizer state_dict 哈希逐位一致——
  `baseline_replay = PASS_EXACT_H8_SHADOW_HASHES_AND_V07_PROJECTIONS`；且 A0C 复现
  max gap = 0.0。任何行为漂移都会使这些检查失败。

## 4. A0F / A0F2 非干扰审计（第十节）

1. **代码依赖**：`grep -rniE '\bbh\b|bh_|_bh'` 于
   `research/src/fr/a0f_cross_state_generalization_v01.py`、
   `research/src/fr/a0f2_cross_partition_extension_v01.py` 及对应 runner：
   **0 处匹配**。A0F/A0F2 的 B/C 传播取自
   `research.scripts.a0c_adamw_response_propagation_v01._run_baseline_tangents`
   （A0F L50-51 import；A0F2 经 A0F 继承），该函数只含 B/C。
2. **本次新增文件**：`research/src/fr/a0e2_*.py`（新）、
   `research/scripts/a0e2_*.py`（新）、`research/tests/test_a0e2_*.py`（新）、
   `research/src/fr/theta_protocol.pyc`（恢复性新增）、
   `outputs/a0_fr_audit/curvature_decomposition_v02_bh_fixed/`（新输出目录）。
   未编辑/覆盖任何冻结文件。
3. **冻结哈希运行前后比对**（本 run 前 2026-09-12 ~19:07 记录，run 完成后复验）：

   | 文件 | 结果 |
   |---|---|
   | `outputs/a0_fr_audit/response_propagation_v01/summary.json` | 一致 |
   | `outputs/a0_fr_audit/curvature_decomposition_v01/summary.json` | 一致 |
   | `outputs/a0_fr_audit/cross_state_generalization_v01/summary.json` | 一致 |
   | `outputs/a0_fr_audit/cross_partition_extension_v01/summary.json` | 一致 |
   | `research/src/fr/a0e_curvature_decomposition_v01.py` | 一致 |
   | `research/scripts/a0e_curvature_decomposition_v01.py` | 一致 |
   | `research/src/fr/a0f{,2}_cross_*_v01.py`、`research/scripts/a0f{,2}_*.py` | 一致 |

4. **结论**：A0F（96 wins/32 H1 ties/0 losses，n=128，mean +0.124154）与
   A0F2（168 wins/56 H1 ties/0 losses，n=224，mean +0.121789）的 B/C 预测、
   wins/ties/losses、R²、mean rel err 均不依赖 BH，本次修正对其**零影响**。
