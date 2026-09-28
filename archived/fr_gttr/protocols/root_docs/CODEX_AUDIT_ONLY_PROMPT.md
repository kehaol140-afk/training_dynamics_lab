请只审计这个 E4B Dynamic Low-dimensional v0.8 代码包，不要重新设计实验，不要自动修改代码，不要运行 GPU。

项目根目录：
`/home/b205/projects/fluctuation_response`

请重点核对：

1. 新文件是否只写：
   - `outputs/e4b_dynamic_state/v08/branching/`
   - `outputs/e4b_dynamic_state/v08/tau_q/`
2. 是否完全只读：
   - `outputs/e4b_dynamic_state/v08/q_rotation/`
   - v0.7 raw/locked files
3. `B/chi` 是否直接来自 E4B-1 state JSON 中的 `B_theta/chi_theta`，没有重算。
4. E4B1b 是否固定 `D_q = sample std_r(R_q), ddof=1`。
5. LOSO 是否严格 leave-one-training-seed-out，candidate 标准化是否 training-fold-only。
6. baseline 是否保持 `chi = seed_FE + state_fraction + B`。
7. candidate 是否只加 `z + B*z`。
8. `tau_q` 是否只拟合：
   `Rbar_q(H)=c+(1-c)exp(-H/tau_q)`。
9. tau 拟合是否 deterministic、bounded、无 scipy、无随机初始化、无双指数/扩模。
10. H8 的 `tau_q` 与 `c` 是否分开 screening，没有组合。
11. curve-collapse 是否不用 `chi/B` 作为主要统计，固定 bins，且只作 descriptive exploratory evidence。
12. 是否没有 controller、没有 GPU、没有新 training、没有修改 v0.7 结论。
13. 运行：py_compile + 新 unit tests；如项目已有完整 q_rotation raw，可运行两个 CPU analysis 脚本。

最终只输出：

- `AUDIT_IMPLEMENTATION = PASS/FAIL`
- `AUDIT_SCIENTIFIC_SEMANTICS = PASS/FAIL`
- `AUDIT_LOSO_LEAKAGE = PASS/FAIL`
- `AUDIT_WRITE_SCOPE = PASS/FAIL`
- `AUDIT_TAU_FIT = PASS/FAIL`
- `AUDIT_CURVE_COLLAPSE = PASS/FAIL`
- 若 FAIL，列出具体文件、行号和最小修复建议。

不要自动进入 E4C/E4B2B。
