# FR A0E2 — BH 对照修正与曲率 / clipping 机制分解 v02

## Verdict 1 — 旧 BH 的 H=1 不一致根因

根因：旧 BH 在初始 pulse 步把**原始 pulse q 直接送入 AdamW tangent**（research/scripts/a0e_curvature_decomposition_v01.py:324 `bh_gradient_tangent = full_gradient_tangent`，而 horizon==1 时 `full_gradient_tangent = q_device`，第 285-286 行），跳过了 B/C 在初始步对 pulse 施加的 global-clip Jacobian `J_clip(g_t) q = c·q − [c·<g,q>/(||g||·(||g||+ε))]·g`（B: 311-315 行经 `b_clip['clip_jvp']`；C: 316-320 行经 `c_clip['clip_jvp']`）。H=1 时不存在 future HVP，理论上必须 B=BH=C；旧实现中只要初始步 clip 激活（如 seed_2037/step_046 stream 2：preclip norm 18.27，clip 系数 0.0547），BH 就把未径向修正的 q 写进 tangent，导致 BH(H1)≠B(H1)=C(H1)。数学对象层面：B/C 的 pulse tangent 属于 clip 映射的光滑 JVP 像空间，旧 BH 的 pulse tangent 是裸 q——两者之差恰好是 radial 项 `radial·g`，在 clip 激活时不可忽略。

## Verdict 2 — 修正后 H=1 恒等性

满足。修复后 BH_fixed 在初始步使用与 B/C 完全相同的 `b_clip['clip_jvp'] = J_clip(g_t) q`（b_clip 与 c_clip 在 horizon==1 输入相同，输出逐位相同），24 条 stream 上 max|B(H1)−BH_fixed(H1)| = 0，max|B(H1)−C(H1)| = 0，max|BH_fixed(H1)−C(H1)| = 0（机器精度，≤1e-10 要求达成，实际为 0）。

- max |B(H1) - BH_fixed(H1)| over 24 streams: `0`
- max |B(H1) - C(H1)| over 24 streams: `0`
- max |BH_fixed(H1) - C(H1)| over 24 streams: `0`
- streams where BH_fixed and C tangents stay bit-identical through all 8 steps: 3/24 (first divergence step reported per stream in per_stream_diagnostics.csv).

## Verdict 3 — 严格隔离 future raw curvature 后的 B → BH_fixed 方向

严格隔离 future raw curvature 后（H>1，9 行口径）：B mean rel err = 0.205255，BH_fixed = 0.136628，即 raw curvature 本身使误差 改善 （gain_raw_curvature_fixed = +0.068627）。对比旧 BH_old = 0.340979（旧 gain_H = -0.135725），旧表中 gain_H<0 的 -0.204352 完全来自初始 pulse 初始化错误（初始 clip 激活的 stream 被污染）。逐 horizon mean rel err H=2: B 0.090254 → BH_fixed 0.103579；H=4: B 0.182465 → BH_fixed 0.159327；H=8: B 0.343044 → BH_fixed 0.146977。clip 未激活的未来步上 BH_fixed 与 C 逐位一致（3/24 条 stream 全程 8 步逐位相同）；clip 激活的未来步上 BH_fixed−C gap 与 clip 激活一致出现（future clip-active steps 48/168，max |BH_fixed−C| = 0.609221）。C 的主结论不变：gain_clip_JVP = +0.114762，C mean rel err = 0.021866。

## Verdict 4 — A0F/A0F2 主结论不受影响

完全不受影响。A0F/A0F2 的 B 与 C 传播路径不依赖 BH（其 src 与 runner 模块对 BH/bh_ 的引用为零，仅 A0F runner 报告里有一句指向 A0E 的注释字符串）；本次修复新增独立模块/脚本/输出目录，未改动任何冻结文件；A0F 冻结结论 96 wins/32 H1 ties/0 losses（n=128，mean +0.124154）、A0F2 冻结结论 168 wins/56 H1 ties/0 losses（n=224，mean +0.121789）经 SHA256 运行前后比对确认无任何变化。「full tangent C 跨 56 states 稳定优于 optimizer-only B」保持成立。

## Model ladder

- **A_momentum** — historical kernel `-eta*mu_raw*(1-beta1**H)` (FR_A0C, unchanged).
- **B_optimizer_only** — FR_A0C complete stepwise AdamW tangent; pulse `delta_g=q` through `J_clip(g_t)`, future `delta_g=0`.
- **BH_old** — frozen FR_A0E model, reproduced bit-exactly here; its *initial pulse bypasses* `J_clip(g_t)` (raw `q` enters the AdamW tangent).
- **BH_fixed** — initial pulse identical to B/C (`J_clip(g_t) q`); future steps raw HVP `H(loss) delta_theta` *without* the clip-JVP radial correction. Differs from C only by the future clip-JVP.
- **C_full_tangent** — FR_A0C full tangent, unchanged.

Wherever clipping is inactive at every future step, BH_fixed and C coincide bit-exactly by construction.

## Per state-horizon comparison (12 rows)

`rel X` = |prediction - chi_fd| / |chi_fd|. Each scope is labeled by its actual row count.

| state | H | chi_fd | A | B | BH_old | BH_fixed | C | rel A | rel B | rel BH_old | rel BH_fixed | rel C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed_2037/state_step_046 | 1 | -0.0498724233704 | -0.482909712342 | -0.0561198622662 | -0.0815561147485 | -0.0561198622662 | -0.0561198622662 | 8.68290048301 | 0.125268404332 | 0.635294802956 | 0.125268404332 | 0.125268404332 |
| seed_2037/state_step_046 | 2 | -0.0831821646531 | -0.91752845345 | -0.101397219059 | -0.154627767686 | -0.106755273442 | -0.090524686705 | 10.0303507642 | 0.218977884042 | 0.858905311387 | 0.283391384284 | 0.0882703892419 |
| seed_2037/state_step_046 | 4 | -0.131721904397 | -1.66072650074 | -0.176069341039 | -0.269871397029 | -0.184582390931 | -0.139873131962 | 11.6078233408 | 0.336674730338 | 1.04879665433 | 0.401303691871 | 0.0618820962465 |
| seed_2037/state_step_046 | 8 | -0.187717752006 | -2.75032915788 | -0.275889740975 | -0.379236380705 | -0.241639658662 | -0.192524920027 | 13.6514068515 | 0.469705118598 | 1.02024782767 | 0.287249906203 | 0.0256084891824 |
| seed_2030/state_step_046 | 1 | -0.0651866565067 | -0.324094286889 | -0.0655277072244 | -0.0655277072244 | -0.0655277072244 | -0.0655277072244 | 3.97178877176 | 0.00523190996332 | 0.00523190996332 | 0.00523190996332 | 0.00523190996332 |
| seed_2030/state_step_046 | 2 | -0.121867887571 | -0.615779145088 | -0.122930828219 | -0.122505760615 | -0.122505760615 | -0.122505760615 | 4.05284170722 | 0.0087220733022 | 0.00523413555977 | 0.00523413555977 | 0.00523413555977 |
| seed_2030/state_step_046 | 4 | -0.213234390119 | -1.11456025261 | -0.217869509132 | -0.214351066191 | -0.214351066191 | -0.21434959922 | 4.22692541286 | 0.0217372020075 | 0.00523684791878 | 0.00523684791878 | 0.00522996829738 |
| seed_2030/state_step_046 | 8 | -0.325153506104 | -1.84582323435 | -0.34743554015 | -0.325353203041 | -0.325353203041 | -0.326869252535 | 4.67677481466 | 0.0685277372943 | 0.000614162030515 | 0.000614162030515 | 0.005276727451 |
| seed_2037/state_step_082 | 1 | -0.0279827403554 | -0.108684469327 | -0.0280386989096 | -0.029285305122 | -0.0280386989096 | -0.0280386989096 | 2.88398233864 | 0.00199975247338 | 0.0465488636944 | 0.00199975247338 | 0.00199975247338 |
| seed_2037/state_step_082 | 2 | -0.049851961905 | -0.206500491722 | -0.0519987564118 | -0.0510615349797 | -0.0487496701261 | -0.0499075718341 | 3.14227412183 | 0.0430633905826 | 0.0242632993467 | 0.0221113018784 | 0.00111550131525 |
| seed_2037/state_step_082 | 4 | -0.0761822229814 | -0.373765890016 | -0.0905794462491 | -0.074757730486 | -0.0707396527191 | -0.0763000911819 | 3.90620876352 | 0.188984026775 | 0.018698489485 | 0.0714414734739 | 0.00154718772889 |
| seed_2037/state_step_082 | 8 | -0.094801363302 | -0.618993690456 | -0.141339406829 | -0.0865708332323 | -0.0802903258037 | -0.0950502782739 | 5.52937541081 | 0.49090057259 | 0.0868186889203 | 0.153067814564 | 0.00262564759796 |

## Aggregate metrics — ALL 12 ROWS (3 states x H in {1,2,4,8})

Metrics are computed by the frozen FR_A0C code (`_metric` imported verbatim).

| model | overall R2 | corr | MAE | RMSE | mean relative error |
|---|---:|---:|---:|---:|---:|
| A_momentum | -167.728835442 | 0.745277565937 | 0.799411692634 | 1.06531210944 | 6.3635543984 |
| B_optimizer_only | 0.838828326821 | 0.966736929314 | 0.0207034235996 | 0.0329250593449 | 0.164982733525 |
| BH_old_curvature_feedback | 0.232497511557 | 0.833083498466 | 0.0372716560767 | 0.0718492278627 | 0.312990916105 |
| BH_curvature_fixed | 0.918975725597 | 0.971952561379 | 0.013334174645 | 0.0233447799405 | 0.113512565379 |
| C_full_tangent | 0.99767960167 | 0.999336137038 | 0.00256971562366 | 0.00395060383803 | 0.0274408507825 |

### Contribution decomposition — ALL 12 ROWS (3 states x H in {1,2,4,8})

| quantity | mean | H=1 | H=2 | H=4 | H=8 |
|---|---:|---:|---:|---:|---:|
| gain_optimizer = relA - relB | 6.19857166488 | 5.13539050888 | 5.65156774845 | 6.39785385268 | 7.60947454949 |
| gain_raw_curvature_fixed = relB - relBH_fixed | 0.0514701681456 | 0 | -0.0133244912652 | 0.0231379819522 | 0.196067181895 |
| gain_clip_JVP = relBH_fixed - relC | 0.0860717145968 | 0 | 0.0720389318686 | 0.13644092033 | 0.135807006188 |
| total = relA - relC | 6.33611354762 | 5.13539050888 | 5.71028218905 | 6.55743275497 | 7.94134873757 |
| identity check: gain sum == total | `True` (max deviation 8.881784197e-16) | | | |

## Aggregate metrics — H>1 ONLY, 9 ROWS (3 states x H in {2,4,8})

Metrics are computed by the frozen FR_A0C code (`_metric` imported verbatim).

| model | overall R2 | corr | MAE | RMSE | mean relative error |
|---|---:|---:|---:|---:|---:|
| A_momentum | -222.642709017 | 0.669101330561 | 0.980032629253 | 1.21826887484 | 6.75822013193 |
| B_optimizer_only | 0.782854252234 | 0.956017173538 | 0.0268662927808 | 0.0379613318679 | 0.205254748392 |
| BH_old_curvature_feedback | -0.020335821038 | 0.778001711185 | 0.0459925073398 | 0.0822881584863 | 0.340979490738 |
| BH_curvature_fixed | 0.891162376234 | 0.962109131236 | 0.0170406275081 | 0.0268754226136 | 0.136627857531 |
| C_full_tangent | 0.997519778961 | 0.99930632154 | 0.00268801547958 | 0.00405705569912 | 0.0218655714023 |

### Contribution decomposition — H>1 ONLY, 9 ROWS (3 states x H in {2,4,8})

| quantity | mean | H=2 | H=4 | H=8 |
|---|---:|---:|---:|---:|
| gain_optimizer = relA - relB | 6.55296538354 | 5.65156774845 | 6.39785385268 | 7.60947454949 |
| gain_raw_curvature_fixed = relB - relBH_fixed | 0.0686268908608 | -0.0133244912652 | 0.0231379819522 | 0.196067181895 |
| gain_clip_JVP = relBH_fixed - relC | 0.114762286129 | 0.0720389318686 | 0.13644092033 | 0.135807006188 |
| total = relA - relC | 6.73635456053 | 5.71028218905 | 6.55743275497 | 7.94134873757 |
| identity check: gain sum == total | `True` (max deviation 8.881784197e-16) | | | |

## Diagnostics (24 streams)

- Streams with initial clip active: 6/24.
- Future clip-active steps (H>1): 48/168.
- BH_old(H1) - B(H1) nonzero streams: 6/24 (max |gap| 0.0682194993123).
- BH_fixed(H1) - B(H1) nonzero streams: 0/24 (max |gap| 0).
- max |BH_fixed - C| response gap over streams x H: 0.609220728188.
- seed_2037/step_046 explanation: 4/8 streams have the initial clip active (initial preclip norms 0.9876, 0.9685, 18.2689, 0.9156, 1.0788, 7.3252, 0.9598, 6.8603). On those streams BH_old fed the raw pulse q into the AdamW tangent instead of J_clip(g) q (clip coefficient 0.054738, 0.926993, 0.136515, 0.145767), producing BH_old(H1)-B(H1) gaps of -0.068219, -0.008939, -0.063576, -0.062755. BH_fixed removes the gap entirely (max |BH_fixed-B| at H1 = 0).

## Consistency with frozen artifacts

- FR_A0C reproduction (A/B/C): `True`; max abs gap by field: chi_A_momentum=0, chi_full_tangent=0, chi_optimizer_only=0, stream:chi_full_tangent=0, stream:chi_optimizer_only=0.
- FR_A0E BH_old reproduction: `True`; max abs gap: chi_BH_curvature=0, stream:chi_BH_curvature=0.
- H1 propagation vs frozen H1-Jacobian artifact: max abs gap `2.32111924126e-09`.

## A0F/A0F2 non-interference (read-only audit)

- Code dependency audit: A0F/A0F2 src+runner modules contain zero references to BH (grep evidence recorded in SOURCE_FIX_AUDIT.md).
- Frozen artifact SHA256 recorded at A0E2 runtime: a0c_summary=9a507eabc3343197…, a0e_summary=6e0a3bb2bab08cd4…, a0f2_summary=d5840dedc3e70755…, a0f_summary=fbb5eebad454a851…; before/after comparison against pre-run hashes: `PASS_PRE_POST_SHA256_IDENTICAL` (see SOURCE_FIX_AUDIT.md).
- A0F frozen verdict: 96 wins / 32 H1 ties / 0 losses over 128 rows, mean improvement +0.124154490635.
- A0F2 frozen verdict: 168 wins / 56 H1 ties / 0 losses over 224 rows, mean improvement +0.121789019603.

## Scope and method

- Same three locked v07 states, eight paired E2 streams per state, and propagation horizons H=1,2,4,8 as FR_A0C/FR_A0E; the test scopes are all-12-rows and the locked H>1 nine rows, each explicitly labeled.
- BH_fixed shares the complete AdamW m/v/bias-correction/weight-decay tangent with B and C; its only difference from C is the absence of the clip-JVP radial correction on the future HVP feedback.
- BH_old is carried with the exact frozen FR_A0E code path so the old artifact is reproduced bit-exactly inside the same run.

## Robustness, replay integrity, and limitations

- Baseline replay: `PASS_EXACT_H8_SHADOW_HASHES_AND_V07_PROJECTIONS`; max saved-chi aggregation gap: `0`.
- Input integrity: `PASS_SHA256_AND_INTERNAL_Q_MANIFEST_VALIDATION`; HVP backend: `PYTORCH_AUTOGRAD_REVERSE_OVER_REVERSE_WITH_MATH_SDPA`.
- Max math-SDPA HVP-forward gradient relative gap vs native flash: `0.00505791896242`; as in FR_A0C this does not bound the HVP error and qualifies the smallest residuals.
- Clipping was active on 48/168 future steps (H>1).
- Runtime: `cuda:0`, AMP `bfloat16`, PyTorch `2.11.0`.
- As in FR_A0C/A0E, all tangents are zero-pulse smooth tangents while `chi_fd` uses the registered finite epsilon; remaining residuals may include finite-epsilon clip/activation boundary crossings.