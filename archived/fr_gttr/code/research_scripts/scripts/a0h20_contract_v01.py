#!/usr/bin/env python3
"""A0H-2.0 corrected-acceptance pilot: contract generator (single source of truth).

Emits the machine-readable contract JSON and the human-readable protocol
amendment from ONE Python dict, and records cross-checks (the MD embeds the
JSON's SHA256; the JSON embeds the MD's SHA256 computed over a template token
replaced after first write — done as two-pass with a fixed marker).

Semantics frozen here BEFORE any A'' exposure (mandate §二). This amendment
does NOT re-run the old L2 unchanged; it replaces the selector clause.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/a0h_corrected_accept_v01" / "contract"
OUT.mkdir(parents=True, exist_ok=True)

CONTRACT = {
    "contract_id": "A0H20_CORRECTED_ACCEPTANCE_CONTRACT_V01",
    "date": "2026-09-16",
    "status": "FROZEN_BEFORE_A_DOUBLE_PRIME_EXPOSURE",
    "supersedes_clauses": {
        "old_clause_source": "outputs/a0h15_vjp_shared_v01/accept_aprime/PREREGISTRATION.md (L1b/L2 'argmax-ĥ' selector clause)",
        "revision_reason": ("Round-3 decision-semantics audit (Branch A) proved the argmax-ĥ clause "
                            "violated frozen A0G selector semantics (argmin predicted Delta L over "
                            "j=0..4 with no-op). The clause and the implementation had drifted "
                            "together. Original text preserved in that file and in L2_CONTRACT_AUDIT.md; "
                            "this contract replaces it for the new batch only."),
        "old_records_preserved": True,
    },
    "semantics": {
        "action_ids": [0, 1, 2, 3, 4],
        "action_0": ("no-op: NO additional intervention; continue the original training update "
                     "(NOT skipping the optimizer step)"),
        "action_j_positive": ("at post-unscale/pre-clip, exactly ONE optimizer step at horizon h=1: "
                              "g' = g + lambda*(g_j - g); afterwards resume the perturbed trajectory normally"),
        "lambda": 0.05,
        "lambda_rule": "one common positive lambda for all non-zero actions, fixed",
        "H": 4,
        "readout": "theta_{t+H} post-step",
        "hhat_semantics": "dL_score/dlambda (Case 1; no sign transform, no utility transform)",
        "predicted_delta_loss_j": "lambda*hhat_j",
        "predicted_gain_j": "-lambda*hhat_j",
        "hhat_0": 0.0,
        "predicted_delta_loss_0": 0.0,
        "selected_id": "min(action_ids, key=(predicted_delta_loss_j, action_id))",
        "tie_break": ("exact A0G comparison, lowest action_id; NO floating-point rounding, "
                      "no tolerance-tie, no gain threshold introduced"),
        "lambda_note": ("the common positive lambda does not change ranking; the old program's "
                        "error was argmax plus dropping no-op, NOT omitting lambda in ranking"),
        "score_outputs": ("persist hhat, lambda*hhat and -lambda*hhat per action with explicit units "
                          "(loss per unit lambda; loss; loss)"),
    },
    "revoked_clauses": {
        "clause": "must choose_noop=False AND every unit Gain>0",
        "status": "REVOKED_FOR_THIS_BATCH",
        "original_text_preserved_in": "accept_aprime/PREREGISTRATION.md + decision_semantics_audit/L2_CONTRACT_AUDIT.md",
        "reason": ("no-op is a legal output with Gain=0; per-unit mandatory positive gain is not "
                   "required by A0G frozen semantics (A0G itself chose no-op 6/360 at method C). "
                   "All-no-op is NOT a positive result; harmful actions are counted honestly."),
    },
    "separate_status_fields": ["COMPLETION_STATUS", "PROTOCOL_VALID", "NUMERICAL_GATE",
                               "DECISION_PRESERVATION", "UTILITY_PILOT_RESULT"],
    "no_composite_pass": True,
    "early_stop_meaning": ("stopping per pre-registered conditions is not a protocol violation and "
                           "does not imply all stages completed"),
    "numerical_gate": {
        "formula": "abs(hhat_fwd_C - hhat_adj) <= atol32 + rtol32*abs(hhat_fwd_C)",
        "atol32": 1e-5, "rtol32": 1e-4,
        "source": ("frozen A0H-1.5 round-2 B1 gate as actually applied to the FP32-HVP candidate "
                   "(a0h15_accept_aprime_v01.py L35; PREREGISTRATION.md 'L1 numerics'); thresholds "
                   "parsed from artifacts, not fitted"),
        "scope": "all 4 active actions at H=4 (decision horizon); H=1/H=8 diffs reported as diagnostic",
        "finite_check": "all scores/losses/diffs/thresholds must pass isfinite; threshold non-negative; NaN/Inf -> NON_FINITE, blocked from gate/aggregation/argmin",
        "claim_limit": "pass means same-config implementation agreement only; NOT native-derivative absolute correctness",
    },
    "methods": {
        "scorers": {
            "bf16_forward_C": {"hvp_precision": "bf16", "role": "frozen historical config comparator"},
            "fp32hvp_forward": {"hvp_precision": "fp32", "role": "same-config FP32 numerical reference"},
            "fp32hvp_adjoint": {"hvp_precision": "fp32", "role": "THIS ROUND'S CANDIDATE (shared VJP adjoint)"},
        },
        "selection_strategies": ["noop", "bf16C_argmin", "fp32fwd_argmin", "fp32adj_argmin",
                                 "replay_score_oracle", "uniform_random_expectation", "dev_fixed"],
        "replay": "native +lambda real replay, 5 actions per unit, single pass computing V_score and V_eval separately per branch",
        "adjoint_constraint": ("adjoint_only must NOT depend on candidate future tangents to build w0 "
                               "and must not hide a joint forward; forward reference is a separate "
                               "audit-cost run; mode fingerprints + call counters persisted"),
        "eval_isolation": ("V_eval per-candidate labels computed in the same pass but written to a "
                           "separate module/file; selectors/thresholds/sample selection never read "
                           "eval; analysis reads eval only after ALL choices are locked"),
    },
    "batch_plan": {
        "target_states": 6, "streams_per_state": 2, "target_units": 12,
        "min_seeds_target": 3,
        "reduced_batch_rule": ("if only 3-5 eligible states, register the full eligible set x2 streams "
                               "BEFORE opening any result; if <3 states or any state lacks 2 legal "
                               "streams -> NO_ELIGIBLE_BATCH (deliver ledger+manifest only)"),
        "selection_rule": ("sort eligible states by (train_seed, true optimizer step, checkpoint ID); "
                           "round-robin across seeds; per state take the first 2 streams in stable "
                           "registered stream-ID order; selection by identity/availability/exposure "
                           "ONLY - never by hhat/gain/loss/history"),
        "seed_short_rule": "<3 seeds does not auto-block but must be registered with tightened interpretation",
    },
    "analysis_plan": {
        "main_endpoint": ("FP32 adjoint chosen action's V_eval Gain: per-state mean over its 2 streams, "
                          "then equal weight over states; ALL units incl. no-op and harmful"),
        "paired_baselines": ["noop", "bf16C", "fp32fwd", "replay_score", "random_expectation", "dev_fixed"],
        "secondary": ["harmful_rate=mean(Gain_eval<0)", "noop_fraction",
                      "Regret_eval=delta_L_eval_selected-min_j(delta_L_eval_j)",
                      "within_state_ranking", "oracle_hit_rate", "finite_action_prediction_error"],
        "bootstrap": {"design": "seed->state->stream hierarchical paired bootstrap", "n": 10000,
                      "rng_seed": 260916, "interval": "95% percentile", "pairing": "same action labels and methods paired",
                      "aggregation": "same equal-weight state aggregation", "single_seed_rule": "no seed-level interval if only 1 seed",
                      "interpretation": "exploratory uncertainty only; stream count != independent seed count; favorable proportion is NOT a frequentist p-value"},
        "utility_pilot_labels": {"NOT_COVERED": "utility stage not entered per pre-registered branch",
                                 "INCOMPLETE": "entered but data incomplete",
                                 "POSITIVE/ZERO/NEGATIVE_POINT_ESTIMATE": "complete; by main endpoint mean sign"},
        "no_translation": "the label never means 'proven significantly better than all methods'",
        "failure_attribution_limit": ("observational localization only: lambda*hhat vs delta_L_score then "
                                      "delta_L_score vs delta_L_eval; NO lambda/nonlinearity/proxy-failure "
                                      "claims without magnitude or native-derivative probe evidence; "
                                      "no immediate retuning/re-candidating/A-triple-prime"),
    },
    "budget": {"gpu_wall_seconds_cap": 1800, "scope": "dev rehearsal + all acceptance GPU work, summed",
               "waiting_total_max_minutes": 10},
    "forbidden_this_round": ["new decision rules", "hybrid/replay-gain fitting", "lambda changes",
                             "AdamW/lr/beta/eps/clip changes", "new training seeds/checkpoints/long training",
                             "v06", "low-rank", "sparse-HVP", "H=8 utility", "full timing matrix redo"],
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    json_path = OUT / "contract.json"
    md_path = OUT / "PROTOCOL_AMENDMENT_A0H20.md"
    json_path.write_text(json.dumps(CONTRACT, indent=1, ensure_ascii=False), encoding="utf-8")
    h_json = sha256_file(json_path)

    md = f"""# A0H-2.0 修正验收 pilot 协议修订（人可读版；与 contract.json 同源互校验）

> 生成于 2026-09-16，**在任何 A″ 评分/损失/标签暴露之前冻结**。
> 机器可读版：`contract.json`（SHA256 `{h_json}`）。本文件与 contract.json
> 由同一配置生成（`research/scripts/a0h20_contract_v01.py`），互为校验。

## 1. 语义冻结（mandate §二）

- action_ids = [0,1,2,3,4]；**j=0 = no-op**：不施加额外干预、继续原始训练更新
  （**不是跳过 optimizer step**）；j>0：post-unscale/pre-clip 仅 h=1 一步施加
  `g' = g + λ(g_j − g)`，λ=0.05 对所有非零动作同一正值；
- H=4，读出 θ_{{t+H}} post-step；
- **hhat = dL_score/dλ**（Case 1，无符号变换、无 utility 变换）；
- predicted_ΔL_j = λ·ĥ_j；predicted_gain_j = −λ·ĥ_j；ĥ_0=0、predicted_ΔL_0=0；
- **selected_id = min(action_ids, key=(predicted_ΔL_j, action_id))**，精确比较、
  tie 取最小 ID；**不新增**浮点舍入、容差并列或获益阈值；
- 公共正 λ 不改变排序——旧程序的错误是 **argmax + 遗漏 no-op**，不是排序时没乘 λ；
- 分数输出同时保存 ĥ、λ·ĥ、−λ·ĥ 并标明单位。

## 2. 撤销条款（旧原文保留）

撤销「必须 choose_noop=False 且每个单元 Gain>0」。原因：选择 no-op 是合法输出
（Gain=0），A0G 冻结语义本身在 formal 中 6/360 合法选 no-op；全部 no-op 不算
观察到正收益；有害动作如实计入。旧条款原文保留于
`accept_aprime/PREREGISTRATION.md` 与 `decision_semantics_audit/L2_CONTRACT_AUDIT.md`。

## 3. 状态字段分离输出

**COMPLETION_STATUS / PROTOCOL_VALID / NUMERICAL_GATE / DECISION_PRESERVATION /
UTILITY_PILOT_RESULT 分开输出，绝不合成单一 PASS。** 按预注册条件合理停止
不等于协议违规，也不等于所有阶段完成。

## 4. 数值 gate（沿用冻结阈值，来源可解析）

`|hhat_fwd_C − hhat_adj| ≤ atol32 + rtol32·|hhat_fwd_C|`，atol32=1e-5、rtol32=1e-4
（来源：a0h15_accept_aprime_v01.py L35 与 round-2 预注册文本实际应用于 FP32-HVP
候选的公式）。适用范围：H=4 全部 4 个非零动作；H=1/H=8 差值仅作诊断。
全部数值须 isfinite；NaN/Inf → NON_FINITE 并阻止 gate/聚合/argmin。
通过只称「同配置实现一致」，不称 native 导数绝对正确。

## 5. 方法与信息隔离

scorers：BF16 前向 C（历史配置比较）、FP32-HVP 前向（同配置参照）、
FP32-HVP 共享伴随（本轮候选）。选择策略：no-op、BF16C argmin、FP32fwd argmin、
FP32adj argmin、replay-score oracle（V_score 最小终点损失，tie 最小 ID）、
均匀随机期望（由五动作标签直接求均值）、dev-fixed（A″ 结果打开前由旧 A′
三状态五动作 V_score 标签冻结的固定动作）。
adjoint_only 不得依赖候选未来 tangent 构造 w0、不得隐藏 joint forward；
forward reference 单独运行（审计开销），保存运行模式指纹与调用计数。
**V_eval 五动作标签同次 replay 分开落盘；任何被测选择器不得读 eval；
全部分析预定选择锁定后才读 eval**（测试验证此隔离；不能保证则降级）。

## 6. 批次计划

6 状态 × 每态 2 条预登记未来流 = 12 单元；≥3 训练 seed 目标；只用已有
checkpoint。状态按 (train_seed, 真实 optimizer step, checkpoint ID) 排序后
跨 seed 轮转选取；每态按已登记 stream ID 稳定顺序取两条合法流。仅按
身份/可用性/暴露记录选取，**禁止按 hhat/gain/损失/历史效果挑选**。
3–5 态 → 开任何结果前登记缩减批次；<3 态或无 2 条合法流 → NO_ELIGIBLE_BATCH。

## 7. 分析计划

主终点：FP32 伴随所选动作 V_eval Gain（态内两流平均 → 态等权）。
配对基线：no-op、BF16 C、FP32 前向、replay-score、随机期望、dev-fixed。
次指标：harmful_rate、no-op 比例、Regret_eval、组内排名、oracle 命中率、
有限动作预测误差。bootstrap：seed→state→stream 分层配对 10000 次、
rng 260916、95% 百分位；仅 1 seed 时不报 seed 层区间；区间仅为探索性
不确定性，有利比例不是频率学 p 值。UTILITY_PILOT_RESULT 取
NOT_COVERED / INCOMPLETE / POSITIVE|ZERO|NEGATIVE_POINT_ESTIMATE。

## 8. 预算与禁止项

GPU 累计墙钟 ≤1800 s（含 dev 演练），短时等待合计 ≤10 分钟。
禁止：新决策规则、hybrid/replay-gain、改 λ、改 AdamW/lr/β/ε/clip、
新训练 seed/checkpoint/长训练、v06、low-rank、sparse-HVP、H=8 效用、
全矩阵计时重做。

## 9. 互校验

- contract.json SHA256：`{h_json}`
- 本 MD 文件 SHA256（生成后由脚本回写 contract.sha256 记录）：见
  `contract.sha256` 与 `contract_crosscheck.json`。
"""
    md_path.write_text(md, encoding="utf-8")
    h_md = sha256_file(md_path)
    (OUT / "contract.sha256").write_text(f"{h_json}  contract.json\n{h_md}  PROTOCOL_AMENDMENT_A0H20.md\n")
    (OUT / "contract_crosscheck.json").write_text(json.dumps(
        {"contract_json_sha256": h_json, "protocol_md_sha256": h_md,
         "same_source_generator": "research/scripts/a0h20_contract_v01.py",
         "frozen_before_A_double_prime_exposure": True}, indent=1))
    print("contract.json      ", h_json)
    print("PROTOCOL_AMENDMENT ", h_md)


if __name__ == "__main__":
    main()
