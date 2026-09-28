#!/usr/bin/env python3
"""Phase B v3 —— GTTR 动作前评分 + 真实分支验证 runner。

执行 FR_PhaseB_GTTR_Action_Scoring_Execution_v3.md + SharedGPU_TokenPrimary
Amendment v1。共享 GPU：无墙钟上限（OBSERVATIONAL_SHARED_GPU，5400s 仅一次性
运维提示）；计算量 caps 硬约束；token 分账；预测先封存后分支。

子命令：
    preflight    CPU 现场核查 + manifest + config（无 GPU）
    cpu_tests    CPU 单元测试 T1–T8 + 计时器隔离测试（无 GPU）
    model_smoke  S0 完整 P2/Q4/half-check/q0 数值 smoke（GPU，30fw+30bwd）
    native_smoke S0 native 8 步 off/on + 双目标 h0/4/8 评价（GPU，16 updates）
    build        6 状态建模 + 144 预测行 + 48 选择行 + 封存 sha256（GPU）
    main         6 状态 × 3 臂 × 64 步真实分支 + 78×2 整库评价（GPU）
    analyze      纯 CPU：bootstrap/误差分解/decision pairs/regret/报告
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs" / "fr_phase_b_gttr_action_scoring_v03"
RUNS = OUT / "runs"
DATA = OUT / "data"
MODELS = OUT / "model_bundles"
V07 = PROJECT_ROOT / "outputs" / "e3_confirmatory" / "v07"
D0 = PROJECT_ROOT / "minimind" / "dataset" / "pretrain_t2t_mini.jsonl"
A1_OUT = PROJECT_ROOT / "outputs" / "fr_gttr_phase_a_native_calibration_v01" / "a1"
A1_MONITOR_JSONL = A1_OUT / "data" / "v_monitor_rows4096_8192.jsonl"
MONITOR_JSONL = DATA / "v_monitor_rows4096_8192.jsonl"
VAL_SPLIT_JSON = PROJECT_ROOT / "outputs" / "fr_decision_utility_v01" / "validation_split.json"

SCHEDULE_ID = "finite_cosine_then_hold_5e-5_v01"
SCHEDULE_LABEL = "NATIVE_WITH_DECLARED_TAIL"

# 状态 manifest（v3 §1.1 冻结）：state_id, seed, n0, real_stream, sim_stream
STATES = [
    ("S0", 2030, 46, 97006, 197006),
    ("S1", 2030, 82, 97007, 197007),
    ("S2", 2031, 46, 97008, 197008),
    ("S3", 2031, 82, 97009, 197009),
    ("S4", 2032, 46, 97010, 197010),
    ("S5", 2032, 82, 97011, 197011),
]
SMOKE_STATE = STATES[0]
SMOKE_STREAM = 97020            # 独立固定 smoke 流（预核查无注册）
ARMS = (("NATIVE", 0.0), ("LR_DOWN", -0.05), ("LR_UP", +0.05))
ARM_BY_U = {"0.0": "NATIVE", "-0.05": "LR_DOWN", "0.05": "LR_UP"}
H_POINTS = (1, 4, 16, 64)
H_MAIN = 64
EVAL_BATCH_ROWS = 32            # monitor 纯评价 batch（4096/32=128 fw）
CONTROL_BATCH_ROWS = 8          # control 纯评价 batch（64/8=8 fw）
MONITOR_N_ROWS = 4096

MODELS_KEYS = ("MODEL_NO_GEOMETRY", "MODEL_GTTR")
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 77071001       # BLOCK_PROXY（与 A1 的 77070001 区分）

# ---- 计算量预算（v3 §6 + amendment §3；全部含 smoke/失败/恢复）-------------
PLAN = {"updates": 1168, "train_fw": 2336, "train_bw": 2336,
        "model_fw": 210, "model_bw": 210, "eval_fw": 11016,
        "total_fw": 13562, "total_bw": 2546, "fd_dirs": 35,
        "monitor_evals": 81, "control_evals": 81, "virtual_steps": 2688}
CAP = {"updates": 1280, "total_fw": 14500, "total_bw": 2920,
       "model_fw": 360, "model_bw": 360, "fd_dirs": 50,
       "monitor_evals": 84, "control_evals": 90}
WALL_OPS_HINT_S = 5400          # 一次性运维提示（amendment §2：非硬上限）

_B1, _B2, _EPS, _WD, _TAU = 0.9, 0.999, 1e-8, 0.01, 1.0

# 计数账本（跨 phase 持久化；不重置）
_C = {"updates": 0, "train_fw": 0, "train_bw": 0, "model_fw": 0, "model_bw": 0,
      "monitor_fw": 0, "control_fw": 0, "monitor_evals": 0, "control_evals": 0,
      "fd_dirs": 0.0, "virtual_steps": 0, "failures": 0,
      "sim_input_slots": 0, "sim_nonpad_tokens": 0, "sim_valid_tokens": 0,
      "peak_mem_gb": 0.0}
_LEDGER = OUT / "counters.json"
_T0 = None


def _load_ledger():
    global _C
    if _LEDGER.exists():
        disk = json.loads(_LEDGER.read_text())
        for k in _C:
            if k in disk:
                _C[k] = disk[k]


def _save_ledger():
    tmp = _LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(_C, indent=1))
    tmp.replace(_LEDGER)


def _log(phase, state_id, note="", **kw):
    rec = {"ts": time.time(), "phase": phase, "state": state_id,
           "wall_s": round(time.perf_counter() - _T0, 3) if _T0 else 0.0,
           "wall_label": "OBSERVATIONAL_SHARED_GPU",
           "note": note, **_C, **kw}
    RUNS.mkdir(parents=True, exist_ok=True)
    with (OUT / "resource_log.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    _save_ledger()


def _check_caps(phase, updates=0, train_fw=0, model_fw=0, monitor_fw=0,
                control_fw=0, bw=0, model_bw=0, fd_dirs=0,
                monitor_evals=0, control_evals=0):
    """计算量硬上限（amendment：墙钟不作为门）。计时器不参与任何判定。"""
    total_fw = (_C["train_fw"] + _C["model_fw"] + _C["monitor_fw"]
                + _C["control_fw"] + train_fw + model_fw + monitor_fw + control_fw)
    total_bw = (_C["train_bw"] + _C["model_bw"] + bw + model_bw)
    if _C["updates"] + updates > CAP["updates"]:
        raise SystemExit("BUDGET_BLOCKED: updates cap 1280")
    if total_fw > CAP["total_fw"]:
        raise SystemExit("BUDGET_BLOCKED: total forward cap 14500")
    if total_bw > CAP["total_bw"]:
        raise SystemExit("BUDGET_BLOCKED: total backward cap 2920")
    if _C["model_fw"] + model_fw > CAP["model_fw"]:
        raise SystemExit("BUDGET_BLOCKED: model-build forward cap 360")
    if _C["model_bw"] + model_bw > CAP["model_bw"]:
        raise SystemExit("BUDGET_BLOCKED: model-build backward cap 360")
    if _C["fd_dirs"] + fd_dirs > CAP["fd_dirs"]:
        raise SystemExit("BUDGET_BLOCKED: FD directions cap 50")
    if _C["monitor_evals"] + monitor_evals > CAP["monitor_evals"]:
        raise SystemExit("BUDGET_BLOCKED: monitor evals cap 84")
    if _C["control_evals"] + control_evals > CAP["control_evals"]:
        raise SystemExit("BUDGET_BLOCKED: control evals cap 90")


def _peak_mem(torch):
    if torch.cuda.is_available():
        gb = torch.cuda.max_memory_allocated() / 2**30
        if gb > _C["peak_mem_gb"]:
            _C["peak_mem_gb"] = round(gb, 4)


def _old_get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 运行时构造
# ---------------------------------------------------------------------------

def build_rt(seed, device):
    from fr import a0k_closed_loop_runner_v01 as a0k
    rt = a0k.build_live_runtime(seed, device)
    assert len(rt["optimizer"].param_groups) == 1, "param group drift"
    from fr.phase_a_tail_schedule_v01 import make_phase_a_get_lr
    rt["eta"] = make_phase_a_get_lr(rt["get_lr"])
    return rt


def eta_base_factory(rt):
    """optimizer step 索引 n 消费的 base LR = eta(2*(n+1))（A1 已核验语义）。"""
    eta = rt["eta"]
    return lambda n: float(eta(2 * (int(n) + 1), 256, 5e-4))


def load_bank(rt, seed, start):
    from fr.full_state import load_full_training_state
    p = V07 / f"seed_{seed}/states/state_step_{start:03d}.pt"
    load_full_training_state(p, rt["model"], rt["optimizer"], rt["scaler"],
                             restore_rng=False)
    rt["model"].train()
    rt["optimizer"].zero_grad(set_to_none=True)
    return p


def named_unique(rt):
    seen, out = set(), {}
    for n, p in rt["named"]:
        if id(p) in seen:
            continue
        seen.add(id(p))
        out[n] = p
    return out


def make_train_stream(rt, stream_id, n_micro):
    import b0_run_v01 as b0
    return {"stream_id": int(stream_id), **b0.make_stream(rt, int(stream_id), n_micro)}


def make_monitor_chunks(rt):
    import a0g_decision_utility_v01 as a0g
    from transformers import AutoTokenizer
    from dataset.lm_dataset import PretrainDataset
    ds = PretrainDataset(str(MONITOR_JSONL),
                         AutoTokenizer.from_pretrained(
                             str(PROJECT_ROOT / "minimind" / "model")),
                         max_length=int(rt["cfg"]["max_seq_len"]))
    chunks = [list(range(i, i + EVAL_BATCH_ROWS))
              for i in range(0, MONITOR_N_ROWS, EVAL_BATCH_ROWS)]
    assert len(chunks) == 128
    return ds, chunks, a0g._stack_all(ds, chunks, rt["device"])


def stack_batch_rt(rt, batch_indices):
    from fr.minimind_v03_runtime import stack_batch
    return stack_batch(rt["dataset"], batch_indices, rt["device"])


# ---------------------------------------------------------------------------
# 评价（monitor = A1 原 dtype/模式；control = FP32 no-autocast 参考）
# ---------------------------------------------------------------------------

def eval_monitor(rt, stacked, state_id, arm, h):
    """V_monitor 4096 行 sum/count（A1 语义：autocast + float() logits）。"""
    torch = rt["torch"]
    _check_caps("eval_monitor", monitor_fw=len(stacked), monitor_evals=1)
    from fr.rng import capture_rng_state, restore_rng_state
    rng_snap = capture_rng_state()
    model = rt["model"]
    was_training = model.training
    total_sum, total_cnt, slots, nonpad = 0.0, 0, 0, 0
    row_recs = []
    t0 = time.perf_counter()
    model.eval()
    try:
        with torch.no_grad():
            for ci, (x, y) in enumerate(stacked):
                with rt["autocast_ctx"]:
                    res = model(x)
                logits = res.logits.float()
                tgt = y[:, 1:]
                ce = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tgt.reshape(-1), ignore_index=-100, reduction="none"
                ).view(tgt.shape[0], tgt.shape[1])
                per_sum = ce.sum(dim=1)
                per_cnt = (tgt != -100).sum(dim=1)
                per_nonpad = (y != -100).sum(dim=1)
                for r in range(tgt.shape[0]):
                    row_recs.append((4096 + ci * EVAL_BATCH_ROWS + r,
                                     float(per_sum[r]), int(per_cnt[r])))
                total_sum += float(per_sum.sum())
                total_cnt += int(per_cnt.sum())
                slots += int(x.numel())
                nonpad += int(per_nonpad.sum())
    finally:
        if was_training:
            model.train()
        restore_rng_state(rng_snap)
    _C["monitor_fw"] += len(stacked)
    _C["monitor_evals"] += 1
    with (OUT / "observed_row_stats.csv").open("a") as f:
        for rid, s, c in row_recs:
            f.write(f"{state_id},{arm},{h},monitor,{rid},{s:.6f},{c}\n")
    _log("eval_monitor", state_id, f"arm={arm} h={h}", monitor_fw=len(stacked))
    return {"H": h, "sum": total_sum, "count": total_cnt,
            "mean": total_sum / max(total_cnt, 1),
            "input_slots": slots, "nonpad_tokens": nonpad,
            "wall_s": time.perf_counter() - t0}


def eval_control(rt, state_id, arm, h, want_grad=False):
    """V_control_ref64：FP32、no-autocast、eval batch 8、sum/count 聚合。

    want_grad=True 时返回 q0（全库 token 分母 sum-reduction 梯度，8 chunks
    逐 chunk backward 累积；不无权平均 microbatch mean）。
    """
    torch = rt["torch"]
    stacked = rt["val_score_tensors"]
    # want_grad（q0 构建）的 8 fw 计入 model 预算而非 control 纯评价预算
    _check_caps("eval_control", control_fw=0 if want_grad else len(stacked),
                control_evals=0 if want_grad else 1,
                model_fw=len(stacked) if want_grad else 0,
                model_bw=len(stacked) if want_grad else 0)
    from fr.rng import capture_rng_state, restore_rng_state
    rng_snap = capture_rng_state()
    model = rt["model"]
    was_training = model.training
    total_sum, total_cnt = 0.0, 0
    row_recs = []
    q0 = None
    t0 = time.perf_counter()
    model.eval()
    try:
        if want_grad:
            model.zero_grad(set_to_none=True)
        for ci, (x, y) in enumerate(stacked):
            res = model(x)                     # no autocast：FP32 reference
            logits = res.logits
            tgt = y[:, 1:]
            ce = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                tgt.reshape(-1), ignore_index=-100, reduction="none"
            ).view(tgt.shape[0], tgt.shape[1])
            per_sum = ce.sum(dim=1)
            per_cnt = (tgt != -100).sum(dim=1)
            for r in range(tgt.shape[0]):
                row_recs.append((ci * CONTROL_BATCH_ROWS + r,
                                 float(per_sum[r].detach()),
                                 int(per_cnt[r].detach())))
            total_sum += float(per_sum.sum().detach())
            total_cnt += int(per_cnt.sum().detach())
            if want_grad:
                per_sum.sum().backward()       # 全库 sum 梯度累积
        if want_grad:
            q0 = {n: p.grad.detach().clone() for n, p in named_unique(rt).items()}
            model.zero_grad(set_to_none=True)
    finally:
        if was_training:
            model.train()
        restore_rng_state(rng_snap)
    _C["control_fw"] += 0 if want_grad else len(stacked)
    if want_grad:
        _C["model_fw"] += len(stacked)
        _C["model_bw"] += len(stacked)
    else:
        _C["control_evals"] += 1
    if not want_grad:   # q0 构建行不进观测文件（避免与观测轴混同）
        with (OUT / "observed_row_stats.csv").open("a") as f:
            for rid, s, c in row_recs:
                f.write(f"{state_id},{arm},{h},control,{rid},{s:.6f},{c}\n")
    _log("eval_control", state_id, f"arm={arm} h={h} grad={want_grad}",
         control_fw=0 if want_grad else len(stacked))
    out = {"H": h, "sum": total_sum, "count": total_cnt,
           "mean": total_sum / max(total_cnt, 1),
           "wall_s": time.perf_counter() - t0}
    if want_grad:
        out["q0"] = q0
    return out


# ---------------------------------------------------------------------------
# SIM 参考场 G_SIM（P=2 microbatch、FP32 no-autocast、train-mode 固定掩码）
# ---------------------------------------------------------------------------

class SimField:
    """G_SIM(θ) = 2 个 SIM microbatch 按真实 (loss+aux)/accum 配方的梯度。

    每次求值前恢复 rng0（同一 SIM 随机性贯穿全部 FD 端点/方向）；
    求值后恢复参数与现场 RNG。零副作用。
    """

    def __init__(self, rt, sim_tensors):
        self.rt = rt
        self.torch = rt["torch"]
        self.sim_tensors = sim_tensors
        self.named = named_unique(rt)
        self.theta0 = {n: p.detach().clone() for n, p in self.named.items()}
        from fr.rng import capture_rng_state
        self.rng_live = capture_rng_state()
        self.rng0 = capture_rng_state()
        self.sim_tokens = {"input_slots": 0, "nonpad_tokens": 0, "valid_tokens": 0}
        for x, y in sim_tensors:
            self.sim_tokens["input_slots"] += int(x.numel())
            self.sim_tokens["nonpad_tokens"] += int((y != -100).sum())
            self.sim_tokens["valid_tokens"] += int((y[:, 1:] != -100).sum())

    def __call__(self, overlay=None):
        from fr.rng import restore_rng_state
        torch = self.torch
        rt = self.rt
        model = rt["model"]
        accum = int(rt["cfg"]["accumulation_steps"])
        if overlay is not None:
            for n, p in self.named.items():
                p.detach().add_(overlay[n])
        try:
            restore_rng_state(self.rng0)      # 固定 SIM 随机性
            was_training = model.training
            model.train()
            try:
                model.zero_grad(set_to_none=True)
                total = None
                for x, y in self.sim_tensors:
                    restore_rng_state(self.rng0)   # 每个 micro 同一掩码
                    res = model(x, labels=y)       # no autocast：FP32
                    loss = (res.loss + res.aux_loss) / accum
                    total = loss if total is None else total + loss
                total.backward()
                g = {n: p.grad.detach().clone() for n, p in self.named.items()}
            finally:
                if not was_training:
                    model.eval()
                model.zero_grad(set_to_none=True)
        finally:
            for n, p in self.named.items():
                p.detach().copy_(self.theta0[n])
            restore_rng_state(self.rng_live)
        return g

    def close(self):
        from fr.rng import restore_rng_state
        for n, p in self.named.items():
            p.detach().copy_(self.theta0[n])
        self.rt["model"].zero_grad(set_to_none=True)
        restore_rng_state(self.rng_live)


# ---------------------------------------------------------------------------
# 模型构建（v3 §3：P2 + s_ref0 + Krylov FD-HVP + q0 → 虚拟传播 → 预测）
# ---------------------------------------------------------------------------

def construct_state_model(rt, state_id, sim_stream_id, eta_base):
    """单状态完整建模。返回 dict（predictions/construction/bundle 路径）。"""
    import torch
    from fr import phase_b_gttr_score_v03 as S
    from fr.stream_rng import seed_future_stream

    n0 = int(rt["_n0"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    seed_future_stream(sim_stream_id)
    sim_stream = make_train_stream(rt, sim_stream_id, 2)
    sim_tensors = [stack_batch_rt(rt, b) for b in sim_stream["batches"]]

    field_probe = SimField(rt, sim_tensors)
    _C["sim_input_slots"] += field_probe.sim_tokens["input_slots"]
    _C["sim_nonpad_tokens"] += field_probe.sim_tokens["nonpad_tokens"]
    _C["sim_valid_tokens"] += field_probe.sim_tokens["valid_tokens"]

    named = named_unique(rt)
    theta0 = {n: p.detach().clone() for n, p in named.items()}
    theta0_norm = S.flat_norm(theta0)
    m0, v0 = {}, {}
    for n, p in named.items():
        st = rt["optimizer"].state[p]
        m0[n] = st["exp_avg"].detach().clone()
        v0[n] = st["exp_avg_sq"].detach().clone()
        assert abs(float(st["step"]) - float(n0)) < 1e-9, "n0 mismatch"
    n_micro = len(sim_tensors)

    # 1) g0（P=2 SIM probe；2 fw + 2 bwd）
    _check_caps("model", model_fw=n_micro, model_bw=n_micro)
    g0 = field_probe(None)
    _C["model_fw"] += n_micro
    _C["model_bw"] += n_micro
    g0_norm = S.flat_norm(g0)

    # 2) s_ref0（参考位移；虚拟算术，无网络成本）
    s_ref0, ref_metrics = S.reference_displacement(
        torch, named, m0, v0, n0, g0, eta_base(n0), betas=(_B1, _B2),
        eps=_EPS, wd=_WD, tau=_TAU)
    s_ref0_norm = S.flat_norm(s_ref0)

    # 3) Krylov FD-HVP 方向（含第一方向 half-check）
    if s_ref0_norm > 0.0:
        v1 = {k: s_ref0[k].clone() for k in s_ref0}
    elif g0_norm > 0.0:
        v1 = {k: g0[k].clone() for k in g0}
    else:
        v1 = None
    if v1 is None:
        kry = {"V": [], "Y": [],
               "diag": {"rank": 0, "max_rank": 4, "delta": None,
                        "y_norms": [], "expanded": [],
                        "labels": ["NO_REFERENCE_DIRECTION"],
                        "fd_rel": None, "fd_cos": None},
               "delta": None}
    else:
        def g_eval(overlay):
            # 每次调用 = 一个端点的一次 field_probe = n_micro fw + n_micro bwd；
            # fd_dirs 按方向口径计（两个端点合计 1 个方向）
            _check_caps("model_fd", model_fw=n_micro, model_bw=n_micro,
                        fd_dirs=0.5)
            g = field_probe(overlay)
            _C["model_fw"] += n_micro
            _C["model_bw"] += n_micro
            _C["fd_dirs"] += 0.5
            return g
        kry = S.krylov_fd_directions(torch, g_eval, theta0_norm, v1, max_q=4)

    # half-check 门（v3 §3.2）：LOCAL_OPERATOR_NOT_CALIBRATED → 停止主流程
    labels = list(kry["diag"]["labels"])
    if "LOCAL_OPERATOR_NOT_CALIBRATED" in labels:
        field_probe.close()
        _C["failures"] += 1
        _log("model_build_stopped", state_id,
             f"fd_rel={kry['diag'].get('fd_rel')}", failures=1)
        raise SystemExit(
            "LOCAL_OPERATOR_NOT_CALIBRATED: state=%s fd_rel=%s fd_cos=%s"
            " —— 停止 GPU 主流程，账本与已有数据已保存"
            % (state_id, kry["diag"].get("fd_rel"), kry["diag"].get("fd_cos")))

    # 4) q0（V_control_ref64 梯度；8 fw + 8 bwd）
    q0_rec = eval_control(rt, state_id, "BUILD", 0, want_grad=True)
    q0 = q0_rec["q0"]
    q0_norm = S.flat_norm(q0)

    field_probe.close()

    # 5) 两个评分模型 → 3 臂虚拟传播 → 4 H 评分
    V, Y = kry["V"], kry["Y"]
    rank = kry["diag"]["rank"]
    orth_max = 0.0
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            orth_max = max(orth_max, abs(S.flat_dot(V[i], V[j])))
    fields = {"MODEL_NO_GEOMETRY": S.make_field(g0),
              "MODEL_GTTR": S.make_field(g0, V, Y)}
    pred_rows, sel_rows, virt_rows = [], [], []
    order = 0
    for model_key in MODELS_KEYS:
        theta_ref = {}
        for arm_name, u in ARMS:
            lr_seq = S.lr_action_seq(eta_base, n0, u, H_MAIN)
            theta_h, metrics = S.propagate(
                torch, named, m0, v0, n0, fields[model_key], lr_seq,
                tau=_TAU, h_points=H_POINTS, n_steps=H_MAIN,
                betas=(_B1, _B2), eps=_EPS, wd=_WD)
            _C["virtual_steps"] += len(metrics)
            if u == 0.0:
                theta_ref = theta_h
            for H in H_POINTS:
                finite = theta_h[H] is not None
                if u == 0.0:
                    j_val = 0.0                    # 原生项严格为 0（v3 §2.3）
                elif finite:
                    j_val = S.j_hat(q0, theta_h[H], theta_ref[H])
                else:
                    j_val = None
                if finite:
                    dth = {k: theta_h[H][k] - theta0[k] for k in theta0}
                    theta_dist = S.flat_norm(dth)
                    if V:
                        coeffs = [S.flat_dot(dth, v) for v in V]
                        recon = {k: sum(c * v[k] for c, v in zip(coeffs, V))
                                 for k in dth}
                        proj_resid = (S.flat_norm({k: dth[k] - recon[k]
                                                   for k in dth})
                                      / max(theta_dist, 1e-30))
                    else:
                        proj_resid = None
                    upd_norm = metrics[H - 1]["update_norm"]
                    clip_frac = sum(1 for m in metrics[:H] if m["clip_active"]) / H
                    m0_norm = S.flat_norm(m0)
                else:
                    theta_dist = proj_resid = upd_norm = clip_frac = m0_norm = None
                pred_rows.append({
                    "state_id": state_id, "model": model_key, "action": arm_name,
                    "u": u, "H": H, "J_hat": j_val,
                    "finite": bool(finite), "seal_order": order})
                virt_rows.append({
                    "state_id": state_id, "model": model_key, "action": arm_name,
                    "H": H, "finite": bool(finite), "theta_dist": theta_dist,
                    "proj_resid_ratio": proj_resid, "clip_frac": clip_frac,
                    "update_norm": upd_norm, "m0_norm": m0_norm,
                    "virtual_steps": len(metrics)})
                order += 1
        # 选择行（每 state/model/H；虚拟分支非有限 → FALLBACK NATIVE）
        for H in H_POINTS:
            jmap = {float(r["u"]): (r["J_hat"] if r["J_hat"] is not None
                                    else math.inf)
                    for r in pred_rows
                    if r["model"] == model_key and r["H"] == H}
            all_finite = all(r["finite"] for r in pred_rows
                             if r["model"] == model_key and r["H"] == H)
            if all_finite:
                sel = S.select_action(jmap)
                fallback = False
            else:
                sel = 0.0
                fallback = True
            sel_rows.append({"state_id": state_id, "model": model_key, "H": H,
                             "selected_u": sel, "fallback_native": fallback,
                             "J_native": jmap.get(0.0), "J_down": jmap.get(-0.05),
                             "J_up": jmap.get(0.05)})

    # 6) construction 记录 + bundle 落盘（q0/g0/V/Y 供 analyze 独立复算）
    MODELS.mkdir(parents=True, exist_ok=True)
    bundle_path = MODELS / f"{state_id}_bundle.pt"
    torch.save({"state_id": state_id, "q0": q0, "g0": g0, "V": V, "Y": Y,
                "theta0_norm": theta0_norm, "s_ref0_norm": s_ref0_norm,
                "g0_norm": g0_norm, "q0_norm": q0_norm, "n0": n0},
               bundle_path)
    constr = {"state_id": state_id, "P_microbatches": n_micro,
              "Q_requested": 4, "rank": rank,
              "delta_fd": kry["diag"]["delta"],
              "theta0_norm": theta0_norm, "g0_norm": g0_norm,
              "s_ref0_norm": s_ref0_norm, "q0_norm": q0_norm,
              "v1_norm": (1.0 if v1 is not None else 0.0),
              "y_norms": "|".join(f"{x:.6e}" for x in kry["diag"]["y_norms"]),
              "fd_rel": kry["diag"].get("fd_rel"),
              "fd_cos": kry["diag"].get("fd_cos"),
              "orth_max_offdiag": orth_max,
              "krylov_labels": "|".join(labels) if labels else "FULL_Q4",
              "ref_clip_active": ref_metrics["clip_active"],
              "ref_preclip_norm": ref_metrics["preclip_norm"],
              "sim_stream_id": sim_stream_id,
              "sim_input_slots": field_probe.sim_tokens["input_slots"],
              "sim_nonpad_tokens": field_probe.sim_tokens["nonpad_tokens"],
              "sim_valid_tokens": field_probe.sim_tokens["valid_tokens"],
              "grad_source": "G_SIM:P2_FP32_trainmode_fixedmask|"
                             "q0:V_control_ref64_sumgrad",
              "bundle_path": str(bundle_path)}
    _peak_mem(torch)
    _log("model_built", state_id,
         f"rank={rank} fd_rel={kry['diag'].get('fd_rel')}")
    return {"predictions": pred_rows, "selections": sel_rows,
            "virtual": virt_rows, "construction": constr}


# ---------------------------------------------------------------------------
# 真实分支（3 臂共用一流；动作 LR 序列）
# ---------------------------------------------------------------------------

def S_lr(eta_base, n0, u, k):
    return float(eta_base(n0 + k)) * math.exp(u if k < 4 else 0.0)


def _flat_dot(a, b):
    import torch
    tot = 0.0
    for k in a:
        tot += float(torch.dot(a[k].reshape(-1).double(), b[k].reshape(-1).double()))
    return tot


def run_arm(rt, state_id, arm_name, u, stream, eta_base, monitor_stacked,
            q0, theta0):
    """单臂 64 步真实训练 + H={1,4,16,64} 双目标评价 + q0dot 标量。"""
    torch = rt["torch"]
    cfg = rt["cfg"]
    accum = int(cfg["accumulation_steps"])
    n0 = int(rt["_n0"])
    model = rt["model"]
    optimizer = rt["optimizer"]
    scaler = rt["scaler"]
    named = named_unique(rt)
    step_rows = []
    eval_records = []
    cum = {"input_slots": 0, "nonpad_tokens": 0, "valid_tokens": 0}
    theta_prev = theta0
    for k in range(H_MAIN):
        _check_caps("main", updates=1, train_fw=accum, bw=accum)
        t_k0 = time.perf_counter()
        row = {"k": k, "lr": None, "preclip_norm": None, "clip_coeff": None,
               "clip_active": None, "train_nll_sum": 0.0, "train_valid_cnt": 0,
               "input_slots": 0, "nonpad_tokens": 0, "update_norm": None,
               "finite": True, "true_step": n0 + k + 1, "wall_s": None}
        lr_k = S_lr(eta_base, n0, u, k)
        for a in range(accum):
            micro_idx = 2 * k + a + 1
            for group in optimizer.param_groups:
                group["lr"] = float(lr_k)
            if a == accum - 1:
                row["lr"] = float(lr_k)
            x, y = stack_batch_rt(rt, stream["batches"][micro_idx - 1])
            with rt["autocast_ctx"]:
                res = model(x, labels=y)
                loss = (res.loss + res.aux_loss) / accum
            scaler.scale(loss).backward()
            _C["train_fw"] += 1
            _C["train_bw"] += 1
            row["train_nll_sum"] += float(res.loss) * int((y[:, 1:] != -100).sum())
            row["train_valid_cnt"] += int((y[:, 1:] != -100).sum())
            row["input_slots"] += int(x.numel())
            row["nonpad_tokens"] += int((y != -100).sum())
            if not math.isfinite(float(res.loss)):
                row["finite"] = False
        cum["input_slots"] += row["input_slots"]
        cum["nonpad_tokens"] += row["nonpad_tokens"]
        cum["valid_tokens"] += row["train_valid_cnt"]
        scaler.unscale_(optimizer)
        preclip = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg["grad_clip"])).detach().cpu())
        row["preclip_norm"] = preclip
        row["clip_coeff"] = (_TAU / preclip) if preclip > 0 else None
        row["clip_active"] = bool(preclip > _TAU)
        if not math.isfinite(preclip):
            row["finite"] = False
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        _C["updates"] += 1
        theta_now = {n: p.detach().clone() for n, p in named.items()}
        upd = float(torch.cat([(theta_now[n] - theta_prev[n]).reshape(-1)
                               for n in named]).norm())
        theta_prev = theta_now
        row["update_norm"] = upd
        if not math.isfinite(upd):
            row["finite"] = False
        row["wall_s"] = time.perf_counter() - t_k0
        step_rows.append(row)
        with (OUT / "step_metrics.csv").open("a") as f:
            f.write(f"{state_id},{u}," + ",".join(str(row[key]) for key in
                    ("k", "lr", "preclip_norm", "clip_coeff", "clip_active",
                     "train_nll_sum", "train_valid_cnt", "input_slots",
                     "nonpad_tokens", "update_norm", "finite", "true_step",
                     "wall_s")) + "\n")
        if not row["finite"]:
            _C["failures"] += 1
            _log("main", state_id, f"NON-FINITE u={u} k={k}", failures=1)
            break
        if (k + 1) in H_POINTS:
            rec = {"H": k + 1}
            rec["monitor"] = eval_monitor(rt, monitor_stacked, state_id,
                                          arm_name, k + 1)
            rec["control"] = eval_control(rt, state_id, arm_name, k + 1)
            rec["q0dot"] = _flat_dot(q0, {n: p.detach() - theta0[n]
                                          for n, p in named.items()})
            rec["cum"] = dict(cum)
            rec["updates_successful"] = len(step_rows)
            eval_records.append(rec)
            _log("main_eval", state_id, f"u={u:+} H={k+1}")
    return {"step_rows": step_rows, "evals": eval_records, "cum": cum}


# 允许出现在封存前观测文件中的臂标签（smoke/h0 为封存前合法探针行）；
# 三大真实臂的任何行在封存前出现即为泄漏。
_PRESEAL_OK_ARMS = {"smoke", "h0"}


def _main_arm_leak(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open() as f:
        for line in f:
            arm = line.split(",")[1].strip()
            if arm not in _PRESEAL_OK_ARMS:
                return arm
    return None


def verify_seal():
    """MAIN 启动前断言：封存完好且无非白名单观测行（T8 路径断言）。"""
    seal_path = OUT / "seal.json"
    assert seal_path.exists(), "PREDICTION_SEAL_MISSING: 拒绝启动真实分支"
    seal = json.loads(seal_path.read_text())
    assert _sha(OUT / "predictions.csv") == seal["predictions_sha256"], \
        "PREDICTION_SEAL_TAMPERED"
    leak = _main_arm_leak(OUT / "observed_row_stats.csv")
    assert leak is None, f"OBSERVED_LEAK: 封存前存在真实臂行 arm={leak}，拒绝启动"
    return seal


# ---------------------------------------------------------------------------
# preflight（CPU）
# ---------------------------------------------------------------------------

def run_preflight():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rep = {}

    # 1) D0 身份 + V_monitor 复制（内容必须与 A1 逐字节一致）
    n_lines, h = 0, hashlib.sha256()
    with D0.open("rb") as f:
        for line in f:
            n_lines += 1
            h.update(line)
    assert n_lines == 1270238 and h.hexdigest().startswith("6dd6716c"), "D0 drift"
    with D0.open("r", encoding="utf-8") as fi, MONITOR_JSONL.open("w", encoding="utf-8") as fo:
        for i, line in enumerate(fi):
            if 4096 <= i < 8192:
                fo.write(line)
    mon_sha = _sha(MONITOR_JSONL)
    a1_sha = _sha(A1_MONITOR_JSONL)
    assert mon_sha == a1_sha, "V_monitor copy differs from A1"
    rep["d0"] = {"lines": n_lines, "sha256": h.hexdigest()}
    rep["v_monitor"] = {"rows": "[4096:8192)", "n": 4096, "sha256": mon_sha,
                        "identical_to_a1": True}

    # 2) V_control_ref64 ID（冻结 split 复算 sha + 角色声明）
    split = json.loads(VAL_SPLIT_JSON.read_text())
    payload = {"v_score": split["v_score"], "v_eval": split["v_eval"]}
    split_sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert split_sha == split["split_sha256"], "validation split sha mismatch"
    part = json.loads((V07 / "partition_v07.json").read_text())
    tr, ev = set(part["train_indices"]), set(part["eval_indices"])
    v_score = list(map(int, split["v_score"]))
    assert len(v_score) == 64 and set(v_score) <= ev and not (set(v_score) & tr)
    assert max(v_score) < 4096 and min(v_score) >= 0
    v_score_set = set(v_score)
    vm_texts, vs_texts = [], {}
    with D0.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if 4096 <= i < 8192:
                vm_texts.append(json.loads(line)["text"])
            elif i in v_score_set:
                vs_texts[i] = json.loads(line)["text"]
    assert len(vm_texts) == 4096 and len(vs_texts) == 64
    assert len(set(vs_texts.values())) == 64, "V_score exact-text duplicate"
    assert not (set(vs_texts.values()) & set(vm_texts)), "V_score ∩ V_monitor"
    rep["v_control_ref64"] = {
        "target_id": "V_control_ref64_v1",
        "n_rows": 64, "row_ids_sorted_first8": sorted(v_score)[:8],
        "split_sha256": split_sha,
        "semantics": "sum(row_nll_sum)/sum(row_valid_target_count); FP32; "
                     "no-autocast; eval batch 8; eval mode",
        "disjoint_train": True, "disjoint_v_monitor": True,
        "not_independent_test_set": "复用开发资料，只作本轮开发评分轴",
        "no_concat_with_a0t_absolute": True}

    # 3) bank 6 状态元数据
    bank = {}
    for sid, seed, start, _, _ in STATES:
        p = V07 / f"seed_{seed}/states/state_step_{start:03d}.pt"
        assert p.exists(), f"missing {p}"
        st = torch.load(p, map_location="cpu", weights_only=False)
        md = st["metadata"]
        assert md["global_optimizer_step"] == start
        assert md["global_micro_step"] == 2 * start
        groups = md["optimizer_signature"]["param_groups"]
        assert len(groups) == 1 and list(groups[0]["betas"]) == [0.9, 0.999] \
            and groups[0]["weight_decay"] == 0.01 and groups[0]["eps"] == 1e-8
        k0 = next(iter(st["optimizer"]["state"]))
        assert float(st["optimizer"]["state"][k0]["step"]) == float(start)
        bank[sid] = {"seed": seed, "n0": start, "gm0": 2 * start, "path": str(p)}
    rep["bank"] = bank

    # 4) stream ID 冲突核查（真实/SIM/smoke；纯元数据，结果前冻结）
    used = set()
    for p in PROJECT_ROOT.glob("outputs/**/*.json"):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        for tok in txt.split('"stream_seed":')[1:]:
            digits = tok.strip()[:20].split(",")[0].strip()
            if digits.isdigit():
                used.add(int(digits))
    real_ids = [s[3] for s in STATES]
    sim_ids = [s[4] for s in STATES]
    shift = 0
    while any(i in used for i in sim_ids):
        shift += 1000
        sim_ids = [i + 1000 for i in sim_ids]
    if shift:
        for i, s in enumerate(STATES):
            STATES[i] = (s[0], s[1], s[2], s[3], sim_ids[i])
    conflicts = [i for i in real_ids + [SMOKE_STREAM] if i in used]
    assert not conflicts, f"stream id conflict {conflicts}; 停止，不覆盖旧结果"
    rep["streams"] = {
        "real": {s[0]: s[3] for s in STATES},
        "sim": {s[0]: s[4] for s in STATES},
        "sim_shift_applied": shift,
        "smoke": SMOKE_STREAM,
        "conflict_check": "no registered stream_seed at real/sim/smoke ids"}

    # 5) schedule 冻结
    from fr.phase_a_tail_schedule_v01 import make_phase_a_get_lr
    eta_check = make_phase_a_get_lr(_old_get_lr)
    assert abs(eta_check(2 * 128, 256, 5e-4) - 5e-5) < 1e-12
    assert eta_check(2 * 129, 256, 5e-4) == 5e-5
    rep["schedule"] = {"id": SCHEDULE_ID, "label": SCHEDULE_LABEL,
                       "formula": "old_get_lr(gm,256,5e-4) if gm<=256 else 5e-5",
                       "action": "LR'=eta_base(n0+k)*exp(u*1[k<4]), u in {0,-0.05,+0.05}"}

    # 6) manifest.csv
    with (OUT / "manifest.csv").open("w") as f:
        f.write("state_id,seed,n0,gm0,bank_path,real_stream,sim_stream,"
                "target_control,target_monitor,schedule_id,tokens_axis\n")
        for sid, seed, start, rs, ss in STATES:
            f.write(f"{sid},{seed},{start},{2*start},{bank[sid]['path']},"
                    f"{rs},{ss},V_control_ref64_v1,V_monitor_D0_4096_8192,"
                    f"{SCHEDULE_ID},input_slots|nonpad|valid_loss_tokens\n")

    rep["budget"] = {"plan": PLAN, "cap": CAP,
                     "wall": "OBSERVATIONAL_SHARED_GPU no cap; 5400s ops hint only",
                     "v_final_eval_calls": 0}
    rep["hardware_mode"] = "SHARED_GPU"
    (OUT / "config.json").write_text(json.dumps({
        "round": "fr_phase_b_gttr_action_scoring_v03",
        "protocol": ["FR_PhaseB_GTTR_Action_Scoring_Execution_v3.md",
                     "FR_PhaseB_SharedGPU_TokenPrimary_Amendment_v1.md"],
        "states": [{"state_id": s, "seed": sd, "n0": n0, "real_stream": r,
                    "sim_stream": sm} for s, sd, n0, r, sm in STATES],
        "arms": [{"name": n, "u": u} for n, u in ARMS],
        "h_points": list(H_POINTS), "intervention_steps": 4,
        "models": list(MODELS_KEYS),
        "p_microbatches": 2, "q_dirs_max": 4,
        "fd_delta_scale": 1e-3, "fd_rank_tol": 1e-6, "fd_rel_gate": 0.20,
        "tie_tol_nats": 1e-12,
        "optimizer": {"name": "torch.optim.AdamW", "betas": [_B1, _B2],
                      "eps": _EPS, "weight_decay": _WD, "clip_tau": _TAU},
        "targets": {"control": "V_control_ref64_v1 (FP32 no-autocast batch8)",
                    "monitor": "V_monitor D0[4096:8192) (A1 dtype/mode batch32)",
                    "v_final": "V_final_eval_calls=0"},
        "statuses_expected": {
            "TOKENS_TO_TARGET_IMPROVEMENT": "NOT_TESTED_PHASE_B",
            "TIME_TO_TARGET": "NOT_EVALUATED_SHARED_GPU",
            "WALL_CLOCK_SPEEDUP": "NOT_EVALUATED_SHARED_GPU",
            "ENERGY_BENEFIT": "NOT_EVALUATED_SHARED_GPU"},
        "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
                      "label": "BLOCK_PROXY"},
        "budget": rep["budget"],
        "preflight": rep}, indent=1) + "\n")
    print(json.dumps(rep["streams"], indent=1))
    print("PREFLIGHT_OK", OUT)
    return rep


# ---------------------------------------------------------------------------
# smoke：模型数值 smoke + native off/on
# ---------------------------------------------------------------------------

def run_model_smoke(gpu):
    global _T0
    _T0 = time.perf_counter()
    import torch
    sid, seed, start, _, sim_id = SMOKE_STATE
    rt = build_rt(seed, f"cuda:{gpu}")
    rt["_n0"] = start
    load_bank(rt, seed, start)
    eta_base = eta_base_factory(rt)
    _check_caps("model_smoke")
    t0 = time.perf_counter()
    res = construct_state_model(rt, sid, sim_id, eta_base)
    wall = time.perf_counter() - t0
    c = res["construction"]
    pred = res["predictions"]
    h1 = {(r["model"], r["action"]): r["J_hat"] for r in pred if r["H"] == 1}
    checks = {
        "state": sid, "P": c["P_microbatches"], "rank": c["rank"],
        "g0_norm": c["g0_norm"], "q0_norm": c["q0_norm"],
        "s_ref0_norm": c["s_ref0_norm"],
        "fd_rel": c["fd_rel"], "fd_cos": c["fd_cos"],
        "orth_max_offdiag": c["orth_max_offdiag"],
        "labels": c["krylov_labels"],
        "wall_s": wall,
        "numeric_pass": bool(
            math.isfinite(c["g0_norm"]) and c["g0_norm"] > 0
            and math.isfinite(c["q0_norm"]) and c["q0_norm"] > 0
            and (c["fd_rel"] is None or c["fd_rel"] <= 0.20)
            and c["orth_max_offdiag"] < 1e-6),
        "T5_H1_models_equal": bool(
            h1[("MODEL_NO_GEOMETRY", "NATIVE")] == 0.0 and
            all(h1[("MODEL_NO_GEOMETRY", a)] == h1[("MODEL_GTTR", a)]
                for a in ("NATIVE", "LR_DOWN", "LR_UP"))),
    }
    (OUT / "model_smoke_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    _log("model_smoke_done", sid, str(checks["numeric_pass"]))
    _peak_mem(torch)
    del rt
    torch.cuda.empty_cache()
    print(json.dumps(checks, indent=1))
    return checks["numeric_pass"] and checks["T5_H1_models_equal"]


def _state_sig(rt):
    torch = rt["torch"]
    sd = {n: p.detach().cpu().clone() for n, p in named_unique(rt).items()}
    ost = rt["optimizer"].state_dict()
    return {"theta": sd, "opt_state": {
        str(k): {kk: (vv.detach().cpu().clone() if torch.is_tensor(vv) else vv)
                 for kk, vv in v.items()}
        for k, v in ost["state"].items()},
        "opt_step": {str(k): int(v["step"]) for k, v in ost["state"].items()},
        "param_groups": [{"lr": g["lr"], "betas": g["betas"],
                          "weight_decay": g["weight_decay"]}
                         for g in ost["param_groups"]]}


def _sig_equal(a, b):
    import torch
    for n in a["theta"]:
        if not torch.equal(a["theta"][n], b["theta"][n]):
            return False, f"theta:{n}"
    for k in a["opt_state"]:
        for kk in a["opt_state"][k]:
            va, vb = a["opt_state"][k][kk], b["opt_state"][k][kk]
            if torch.is_tensor(va):
                if not torch.equal(va, vb):
                    return False, f"opt:{k}.{kk}"
            elif va != vb:
                return False, f"opt:{k}.{kk}"
    if a["opt_step"] != b["opt_step"]:
        return False, "opt_step"
    if a["param_groups"] != b["param_groups"]:
        return False, "param_groups"
    return True, ""


def _rng_equal(a, b):
    import torch
    if a["python_random"] != b["python_random"]:
        return False, "python_random"
    na, nb = a.get("numpy_random"), b.get("numpy_random")
    if (na is None) != (nb is None):
        return False, "numpy_rng_presence"
    if na is not None and not (na[0] == nb[0] and (na[1] == nb[1]).all()):
        return False, "numpy_random"
    if not torch.equal(a["torch_cpu"], b["torch_cpu"]):
        return False, "torch_cpu"
    ca, cb = a.get("torch_cuda_all"), b.get("torch_cuda_all")
    if (ca is None) != (cb is None):
        return False, "cuda_rng_presence"
    if ca is not None:
        for x, y in zip(ca, cb):
            if not torch.equal(x, y):
                return False, "torch_cuda"
    return True, ""


def run_native_smoke(gpu):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    import torch
    from fr.rng import capture_rng_state, restore_rng_state
    from fr.stream_rng import seed_future_stream
    sid, seed, start, _, _ = SMOKE_STATE
    rt = build_rt(seed, f"cuda:{gpu}")
    rt["_n0"] = start
    load_bank(rt, seed, start)
    mon_ds, mon_chunks, mon_stacked = make_monitor_chunks(rt)
    eta_base = eta_base_factory(rt)
    bundle = torch.load(MODELS / f"{sid}_bundle.pt", map_location=rt["device"],
                        weights_only=False)
    q0 = bundle["q0"]
    theta0 = {n: p.detach().clone() for n, p in named_unique(rt).items()}

    # u=0 动作 LR 与冻结 native schedule 的一致性（在训练前核验，写 lr 无害）
    from fr.native_schedule import set_native_micro_lr
    lr_seq_ok = all(
        abs(S_lr(eta_base, start, 0.0, k)
            - float(set_native_micro_lr(rt["optimizer"], rt["eta"],
                                        2 * start + 2 * (k + 1), 256, 5e-4))) < 1e-18
        for k in range(8))
    rt["optimizer"].zero_grad(set_to_none=True)

    checks = {"state": sid, "smoke_stream": SMOKE_STREAM,
              "updates_planned": 16, "u": 0.0,
              "lr_u0_equals_native_schedule": bool(lr_seq_ok)}

    def _run(n_steps, evals):
        stream = make_train_stream(rt, SMOKE_STREAM, 2 * n_steps)
        rows, recs = [], []
        cum = {"input_slots": 0, "nonpad_tokens": 0, "valid_tokens": 0}
        for k in range(n_steps):
            _check_caps("native_smoke", updates=1, train_fw=2, bw=2)
            for group in rt["optimizer"].param_groups:
                group["lr"] = float(eta_base(start + k))
            for a in range(2):
                micro_idx = 2 * k + a + 1
                x, y = stack_batch_rt(rt, stream["batches"][micro_idx - 1])
                with rt["autocast_ctx"]:
                    res = rt["model"](x, labels=y)
                    loss = (res.loss + res.aux_loss) / 2
                rt["scaler"].scale(loss).backward()
                _C["train_fw"] += 1
                _C["train_bw"] += 1
                cum["input_slots"] += int(x.numel())
                cum["nonpad_tokens"] += int((y != -100).sum())
                cum["valid_tokens"] += int((y[:, 1:] != -100).sum())
            rt["scaler"].unscale_(rt["optimizer"])
            preclip = float(torch.nn.utils.clip_grad_norm_(
                rt["model"].parameters(), 1.0).detach().cpu())
            rt["scaler"].step(rt["optimizer"])
            rt["scaler"].update()
            rt["optimizer"].zero_grad(set_to_none=True)
            _C["updates"] += 1
            rows.append({"k": k, "preclip_norm": preclip,
                         "finite": math.isfinite(preclip)})
            if evals and (k + 1) in evals:
                recs.append({
                    "h": k + 1,
                    "monitor": eval_monitor(rt, mon_stacked, sid, "smoke", k + 1),
                    "control": eval_control(rt, sid, "smoke", k + 1),
                    "q0dot": _flat_dot(q0, {n: p.detach() - theta0[n]
                                           for n, p in named_unique(rt).items()})})
        return rows, recs, cum

    # off：8 步纯训练
    seed_future_stream(SMOKE_STREAM)
    rng_mark = capture_rng_state()
    off_rows, _, off_cum = _run(8, ())
    sig_off = _state_sig(rt)
    rng_off = capture_rng_state()
    cursor_off = 16

    # on：恢复初态 + 同流，8 步 + h1/4/8 双目标评价
    load_bank(rt, seed, start)
    restore_rng_state(rng_mark)
    on_rows, on_recs, on_cum = _run(8, (1, 4, 8))
    sig_on = _state_sig(rt)
    rng_on = capture_rng_state()
    cursor_on = 16

    same, why = _sig_equal(sig_off, sig_on)
    rsame, rwhy = _rng_equal(rng_off, rng_on)
    data_same = (off_cum == on_cum)
    checks.update({
        "theta_m_v_step_lr_bitwise_equal": bool(same), "mismatch": why,
        "rng_state_equal": bool(rsame), "rng_mismatch": rwhy,
        "data_cursor_equal": (cursor_off == cursor_on == 16),
        "off_on_data_tokens_equal": bool(data_same),
        "off_updates": len(off_rows), "on_updates": len(on_rows),
        "on_monitor_evals": [dict((k, m["monitor"][k]) for k in
                                  ("H", "sum", "count", "mean")) for m in on_recs],
        "on_control_evals": [dict((k, m["control"][k]) for k in
                                  ("H", "sum", "count", "mean")) for m in on_recs],
        "q0dot_rows": [dict(h=r["h"], q0dot=r["q0dot"]) for r in on_recs],
        "verdict": "PASS" if (same and rsame and cursor_off == cursor_on and data_same
                              and lr_seq_ok and len(off_rows) == 8
                              and len(on_rows) == 8) else "FAIL"})
    (OUT / "smoke_checks.json").write_text(json.dumps(checks, indent=1,
                                                      default=str) + "\n")
    wall = time.perf_counter() - _T0
    mon_per_eval = (sum(r["monitor"]["wall_s"] for r in on_recs)
                    / max(len(on_recs), 1))
    est = {"smoke_wall_s": wall,
           "per_step_s": wall / 16.0,
           "monitor_per_eval_s": mon_per_eval,
           "ops_hint_s": WALL_OPS_HINT_S,
           "note": "OBSERVATIONAL_SHARED_GPU: 非硬上限，超时不触发任何科学/预算门"}
    est["eta_main_train_s"] = 1152 * est["per_step_s"]
    est["eta_main_eval_s"] = 156 * mon_per_eval * (1 + 1 / 16.0)
    est["eta_total_s"] = est["eta_main_train_s"] + est["eta_main_eval_s"]
    est["exceeds_ops_hint"] = bool(est["eta_total_s"] > WALL_OPS_HINT_S)
    (OUT / "eta_estimate.json").write_text(json.dumps(est, indent=1) + "\n")
    _log("native_smoke_done", sid, checks["verdict"])
    _peak_mem(torch)
    del rt, mon_stacked
    torch.cuda.empty_cache()
    print(json.dumps({k: v for k, v in checks.items()
                      if k not in ("on_monitor_evals", "on_control_evals")},
                     indent=1, default=str))
    print("ETA", json.dumps(est, indent=1))
    return checks["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# build：6 状态建模 + 预测 + 封存
# ---------------------------------------------------------------------------

def run_build(gpu):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    import torch
    leak = _main_arm_leak(OUT / "observed_row_stats.csv")
    assert leak is None, f"OBSERVED_LEAK: build 前存在真实臂行 arm={leak}（T8）"
    assert not (OUT / "progress.jsonl").exists(), \
        "BUILD_AFTER_MAIN: main 已启动，拒绝重建/覆盖预测"
    # 全新构建起点：清空 probe 行与逐 step 记录（保留 counters 账本不重置）
    for name in ("observed_row_stats.csv", "observed_actions.csv",
                 "step_metrics.csv", "token_identity.jsonl"):
        p = OUT / name
        if p.exists():
            p.unlink()
    _check_caps("build")
    all_pred, all_sel, all_virt, all_constr = [], [], [], []
    for sid, seed, start, _, sim_id in STATES:
        rt = build_rt(seed, f"cuda:{gpu}")
        rt["_n0"] = start
        load_bank(rt, seed, start)
        eta_base = eta_base_factory(rt)
        res = construct_state_model(rt, sid, sim_id, eta_base)
        all_pred += res["predictions"]
        all_sel += res["selections"]
        all_virt += res["virtual"]
        all_constr.append(res["construction"])
        del rt
        torch.cuda.empty_cache()
    # predictions.csv（144 行，seal_order 全局连续）
    for i, r in enumerate(all_pred):
        r["seal_order"] = i
    with (OUT / "predictions.csv").open("w") as f:
        f.write("state_id,model,action,u,H,J_hat,finite,seal_order\n")
        for r in all_pred:
            f.write(f"{r['state_id']},{r['model']},{r['action']},{r['u']},{r['H']},"
                    f"{r['J_hat']!r},{r['finite']},{r['seal_order']}\n")
    with (OUT / "selections.csv").open("w") as f:
        f.write("state_id,model,H,selected_u,fallback_native,J_native,J_down,J_up\n")
        for r in all_sel:
            f.write(f"{r['state_id']},{r['model']},{r['H']},{r['selected_u']},"
                    f"{r['fallback_native']},{r['J_native']!r},{r['J_down']!r},"
                    f"{r['J_up']!r}\n")
    with (OUT / "virtual_h_metrics.csv").open("w") as f:
        f.write("state_id,model,action,H,finite,theta_dist,proj_resid_ratio,"
                "clip_frac,update_norm,m0_norm,virtual_steps\n")
        for r in all_virt:
            f.write(f"{r['state_id']},{r['model']},{r['action']},{r['H']},"
                    f"{r['finite']},{r['theta_dist']},{r['proj_resid_ratio']},"
                    f"{r['clip_frac']},{r['update_norm']},{r['m0_norm']},"
                    f"{r['virtual_steps']}\n")
    with (OUT / "model_construction.csv").open("w") as f:
        keys = list(all_constr[0].keys())
        f.write(",".join(keys) + "\n")
        for r in all_constr:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    seal = {"predictions_sha256": _sha(OUT / "predictions.csv"),
            "selections_sha256": _sha(OUT / "selections.csv"),
            "model_construction_sha256": _sha(OUT / "model_construction.csv"),
            "config_sha256": _sha(OUT / "config.json"),
            "n_prediction_rows": len(all_pred),
            "n_selection_rows": len(all_sel),
            "sealed_wall_s": (time.perf_counter() - _T0) if _T0 else None,
            "timestamp": time.time(),
            "label": "PREDICTION_SEAL"}
    (OUT / "seal.json").write_text(json.dumps(seal, indent=1) + "\n")
    _log("build_sealed", None, json.dumps({"rows": len(all_pred),
                                           "sha": seal["predictions_sha256"][:16]}))
    print("BUILD_SEALED", seal["predictions_sha256"][:16],
          "rows", len(all_pred), "sels", len(all_sel))
    return seal


# ---------------------------------------------------------------------------
# main：6 状态 × 3 臂真实分支
# ---------------------------------------------------------------------------

def _write_observed_actions(sid, arm_name, u, rec, upd_ok, upd_att):
    mon, ctrl = rec["monitor"], rec["control"]
    with (OUT / "observed_actions.csv").open("a") as f:
        f.write(f"{sid},{arm_name},{u},{rec['H']},"
                f"{mon['sum']!r},{mon['count']},{mon['mean']!r},"
                f"{ctrl['sum']!r},{ctrl['count']},{ctrl['mean']!r},"
                f"{rec['q0dot']!r},{upd_ok},{upd_att},"
                f"{rec['cum']['input_slots']},{rec['cum']['nonpad_tokens']},"
                f"{rec['cum']['valid_tokens']}\n")


def run_main(gpu):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    import torch
    from fr.rng import capture_rng_state, restore_rng_state
    from fr.stream_rng import seed_future_stream
    seal = verify_seal()
    _check_caps("main_start")
    if not (OUT / "observed_actions.csv").exists():
        with (OUT / "observed_actions.csv").open("w") as f:
            f.write("state_id,arm,u,H,monitor_sum,monitor_count,monitor_mean,"
                    "control_sum,control_count,control_mean,q0dot,"
                    "updates_successful,updates_attempts,"
                    "train_input_token_slots,train_nonpad_tokens,"
                    "train_valid_loss_tokens\n")

    progress_path = OUT / "progress.jsonl"
    done = set()
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["state"], rec["arm"]))

    status_path = OUT / "main_status.json"
    try:
        for sid, seed, start, real_id, _ in STATES:
            rt = build_rt(seed, f"cuda:{gpu}")
            rt["_n0"] = start
            load_bank(rt, seed, start)
            mon_ds, mon_chunks, mon_stacked = make_monitor_chunks(rt)
            bundle = torch.load(MODELS / f"{sid}_bundle.pt",
                                map_location=rt["device"], weights_only=False)
            q0 = bundle["q0"]
            theta0 = {n: p.detach().clone() for n, p in named_unique(rt).items()}
            eta_base = eta_base_factory(rt)

            # h0 共享评价（每状态 1 次 × 2 目标；3 臂共用）
            if (sid, "h0") not in done:
                h0_rec = {"H": 0,
                          "monitor": eval_monitor(rt, mon_stacked, sid, "h0", 0),
                          "control": eval_control(rt, sid, "h0", 0),
                          "q0dot": 0.0,
                          "cum": {"input_slots": 0, "nonpad_tokens": 0,
                                  "valid_tokens": 0}}
                _write_observed_actions(sid, "h0", 0.0, h0_rec, 0, 0)
                with progress_path.open("a") as f:
                    f.write(json.dumps({"state": sid, "arm": "h0"}) + "\n")
                done.add((sid, "h0"))
            seed_future_stream(real_id)
            rng_stream = capture_rng_state()
            stream = make_train_stream(rt, real_id, 2 * H_MAIN)

            arm_data_tok = {}
            for arm_name, u in ARMS:
                if (sid, arm_name) in done:
                    continue
                load_bank(rt, seed, start)
                restore_rng_state(rng_stream)
                _check_caps("main", updates=H_MAIN, train_fw=2 * H_MAIN,
                            bw=2 * H_MAIN, monitor_fw=4 * 128, control_fw=4 * 8,
                            monitor_evals=4, control_evals=4)
                out = run_arm(rt, sid, arm_name, u, stream, eta_base,
                              mon_stacked, q0, theta0)
                arm_data_tok[arm_name] = [
                    (r["input_slots"], r["nonpad_tokens"], r["train_valid_cnt"])
                    for r in out["step_rows"]]
                upd_ok = len(out["step_rows"])
                for rec in out["evals"]:
                    _write_observed_actions(sid, arm_name, u, rec, upd_ok, H_MAIN)
                with progress_path.open("a") as f:
                    f.write(json.dumps({"state": sid, "arm": arm_name,
                                        "steps": len(out["step_rows"])}) + "\n")
                done.add((sid, arm_name))
                _log("arm_done", sid, f"{arm_name} steps={len(out['step_rows'])}")
            # 3 臂数据 token 同一性核查（同流严格配对）
            if len(arm_data_tok) == 3:
                seqs = [arm_data_tok[n] for n, _ in ARMS]
                identical = (seqs[0] == seqs[1] == seqs[2])
                with (OUT / "token_identity.jsonl").open("a") as f:
                    f.write(json.dumps({"state": sid, "identical": identical,
                                        "per_step_triples": len(seqs[0])}) + "\n")
                assert identical, f"3-arm token identity violated at {sid}"
            del rt, mon_stacked
            torch.cuda.empty_cache()
        (OUT / "main_summary.json").write_text(json.dumps({
            "status": "MAIN_DONE", "seal": seal["predictions_sha256"],
            "counters": dict(_C)}, indent=1) + "\n")
        _log("main_done", None, "MAIN_DONE")
        print("MAIN_DONE", json.dumps(_C))
        return True
    except (KeyboardInterrupt, RuntimeError) as exc:
        _C["failures"] += 1
        _save_ledger()
        status_path.write_text(json.dumps({
            "status": "RESOURCE_INTERRUPTED", "error": repr(exc),
            "counters": dict(_C),
            "note": "执行覆盖状态，非科学失败；可从 progress.jsonl 安全边界恢复"},
            indent=1) + "\n")
        _log("main_interrupted", None, repr(exc), failures=1)
        print("RESOURCE_INTERRUPTED", repr(exc))
        return False


# ---------------------------------------------------------------------------
# analyze（纯 CPU；bootstrap 用预计算 block-sum 矩阵向量化实现）
# ---------------------------------------------------------------------------

def run_analyze():
    import numpy as np
    import csv as _csv
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()

    states = [s[0] for s in STATES]

    # ---- 读取逐行观测 --------------------------------------------------------
    obs_rows = {}
    with (OUT / "observed_row_stats.csv").open() as f:
        rdr = _csv.reader(f)
        for row in rdr:
            sid, arm, h, target, rid, s, c = row
            obs_rows.setdefault((sid, arm, int(h), target), []).append(
                (int(rid), float(s), int(c)))
    obs_actions = []
    with (OUT / "observed_actions.csv").open() as f:
        rdr = _csv.reader(f)
        header = next(rdr)
        for row in rdr:
            rec = dict(zip(header, row))
            obs_actions.append(rec)

    preds = []
    with (OUT / "predictions.csv").open() as f:
        rdr = _csv.DictReader(f)
        for r in rdr:
            preds.append({"state_id": r["state_id"], "model": r["model"],
                          "action": r["action"], "u": float(r["u"]),
                          "H": int(r["H"]),
                          "J_hat": (None if r["J_hat"] == "None"
                                    else float(r["J_hat"])),
                          "finite": r["finite"] == "True",
                          "seal_order": int(r["seal_order"])})

    # ---- block-sum 预计算 + paired bootstrap（BLOCK_PROXY）------------------
    BLOCK = {"control": 8, "monitor": 64}
    block_sums = {}     # (sid,h,target) -> dict arm -> (n_blocks,2) array
    boot_idx = {}       # (sid,h,target) -> (reps,n_blocks) int array
    rng_master = np.random.default_rng(BOOTSTRAP_SEED)
    for sid in states:
        for h in H_POINTS:
            for target in ("control", "monitor"):
                key = (sid, h, target)
                per_arm = {}
                n_blocks = None
                for arm, _u in ARMS:
                    rows = sorted(obs_rows.get((sid, arm, h, target), []))
                    assert rows, f"missing rows {key} {arm}"
                    blk = BLOCK[target]
                    assert len(rows) % blk == 0
                    nb = len(rows) // blk
                    arr = np.array([[sum(r[1] for r in rows[i * blk:(i + 1) * blk]),
                                     sum(r[2] for r in rows[i * blk:(i + 1) * blk])]
                                    for i in range(nb)], dtype=np.float64)
                    per_arm[arm] = arr
                    n_blocks = nb
                block_sums[key] = per_arm
                boot_idx[key] = rng_master.integers(0, n_blocks,
                                                    (BOOTSTRAP_REPS, n_blocks))

    def boot_means(sid, h, target):
        """返回 {arm: (mean, se_of_mean_over_blocks)} —— SE 仅参考。"""
        key = (sid, h, target)
        idx = boot_idx[key]
        out = {}
        for arm, arr in block_sums[key].items():
            s = arr[idx, 0].sum(axis=1)
            c = arr[idx, 1].sum(axis=1)
            means = s / np.maximum(c, 1)
            out[arm] = (float(np.mean(means)), float(np.std(means, ddof=1)))
        return out

    def boot_pair_se(sid, h, target, a1, a2):
        key = (sid, h, target)
        idx = boot_idx[key]
        def _m(arm):
            arr = block_sums[key][arm]
            return (arr[idx, 0].sum(axis=1)
                    / np.maximum(arr[idx, 1].sum(axis=1), 1))
        return float(np.std(_m(a1) - _m(a2), ddof=1))

    evals = {}
    for sid in states:
        for h in H_POINTS:
            for target in ("control", "monitor"):
                evals[(sid, h, target)] = {a: boot_means(sid, h, target)[a][0]
                                           for a, _ in ARMS}

    # ---- observed_actions.csv 增补 J_obs/gain/paired SE ----------------------
    oa_keys = header + ["J_obs_ctrl", "gain_monitor", "paired_se_vs_native",
                        "monitor_mean_se"]
    oa_out = []
    for r in obs_actions:
        sid, h, arm = r["state_id"], int(r["H"]), r["arm"]
        if arm == "h0":
            r.update({"J_obs_ctrl": "None", "gain_monitor": "None",
                      "paired_se_vs_native": "None",
                      "monitor_mean_se": str(boot_means(sid, h, "monitor")[
                          "NATIVE"][1] if h in H_POINTS else "None")})
        else:
            j_obs = (evals[(sid, h, "control")][arm]
                     - evals[(sid, h, "control")]["NATIVE"])
            gain = (evals[(sid, h, "monitor")]["NATIVE"]
                    - evals[(sid, h, "monitor")][arm])
            se = boot_pair_se(sid, h, "control", arm, "NATIVE")
            r.update({"J_obs_ctrl": repr(j_obs), "gain_monitor": repr(gain),
                      "paired_se_vs_native": repr(se),
                      "monitor_mean_se": repr(boot_means(sid, h, "monitor")[arm][1])})
        oa_out.append(r)
    with (OUT / "observed_actions.csv").open("w") as f:
        f.write(",".join(oa_keys) + "\n")
        for r in oa_out:
            f.write(",".join(str(r.get(k)) for k in oa_keys) + "\n")

    # ---- prediction_error.csv（含 linear_obs 与误差分解恒等式）--------------
    q0dot = {(r["state_id"], r["arm"], int(r["H"])): float(r["q0dot"])
             for r in obs_actions}
    with (OUT / "prediction_error.csv").open("w") as f:
        f.write("state_id,model,action,u,H,pred_J_hat,truth_J_obs_ctrl,error,"
                "error_sq,truth_sq,linear_obs,e_propagation,e_readout,"
                "e_identity_holds\n")
        for p in preds:
            sid, h, arm = p["state_id"], p["H"], p["action"]
            truth = (0.0 if arm == "NATIVE"
                     else evals[(sid, h, "control")][arm]
                     - evals[(sid, h, "control")]["NATIVE"])
            if p["J_hat"] is None:
                f.write(f"{sid},{p['model']},{arm},{p['u']},{h},None,"
                        f"{truth!r},None,None,{truth * truth!r},None,None,"
                        f"None,False\n")
                continue
            err = p["J_hat"] - truth
            lin = (0.0 if arm == "NATIVE"
                   else q0dot[(sid, arm, h)] - q0dot[(sid, "NATIVE", h)])
            e_prop = p["J_hat"] - lin
            e_read = lin - truth
            ident = abs(err - (e_prop + e_read)) <= 1e-12
            f.write(f"{sid},{p['model']},{arm},{p['u']},{h},{p['J_hat']!r},"
                    f"{truth!r},{err!r},{err * err!r},{truth * truth!r},"
                    f"{lin!r},{e_prop!r},{e_read!r},{ident}\n")

    # ---- aggregate_inputs.csv + 模型误差 E（固定集合）------------------------
    def _truth(sid, h, arm):
        return (evals[(sid, h, "control")][arm]
                - evals[(sid, h, "control")]["NATIVE"])

    model_E, model_SSE, model_TSS, model_cov = {}, {}, {}, {}
    with (OUT / "aggregate_inputs.csv").open("w") as f:
        f.write("scope,state_id,model,H,n_points,fixed_set,SSE,TSS,E,coverage\n")
        for model in MODELS_KEYS:
            for h in H_POINTS:
                sse = tss = 0.0
                n = 0
                for sid in states:
                    for arm in ("LR_DOWN", "LR_UP"):
                        p = next(r for r in preds
                                 if r["state_id"] == sid and r["model"] == model
                                 and r["H"] == h and r["action"] == arm)
                        if p["J_hat"] is None:
                            continue
                        t = _truth(sid, h, arm)
                        sse += (p["J_hat"] - t) ** 2
                        tss += t * t
                        n += 1
                E = (None if tss == 0 else math.sqrt(sse / tss))
                model_E[(model, h)] = E
                model_SSE[(model, h)], model_TSS[(model, h)] = sse, tss
                model_cov[(model, h)] = f"{n}/12"
                f.write(f"ALL,ALL,{model},{h},{n},6states x +/-0.05,{sse!r},"
                        f"{tss!r},{E},{n}/12\n")
                for sid in states:
                    sse_s = tss_s = 0.0
                    ns = 0
                    for arm in ("LR_DOWN", "LR_UP"):
                        p = next(r for r in preds
                                 if r["state_id"] == sid and r["model"] == model
                                 and r["H"] == h and r["action"] == arm)
                        if p["J_hat"] is None:
                            continue
                        t = _truth(sid, h, arm)
                        sse_s += (p["J_hat"] - t) ** 2
                        tss_s += t * t
                        ns += 1
                    E_s = (None if tss_s == 0 else math.sqrt(sse_s / tss_s))
                    f.write(f"state,{sid},{model},{h},{ns},state x +/-0.05,"
                            f"{sse_s!r},{tss_s!r},{E_s},{ns}/2\n")

    # ---- decision_pairs.csv -------------------------------------------------
    with (OUT / "decision_pairs.csv").open("w") as f:
        f.write("state_id,H,pair,model,pred_D,truth_D,paired_se,resolved\n")
        for sid in states:
            for h in H_POINTS:
                for a1, a2 in (("NATIVE", "LR_DOWN"), ("NATIVE", "LR_UP"),
                               ("LR_DOWN", "LR_UP")):
                    truth_d = (evals[(sid, h, "control")][a1]
                               - evals[(sid, h, "control")][a2])
                    se = boot_pair_se(sid, h, "control", a1, a2)
                    resolved = abs(truth_d) > 3 * se
                    for model in MODELS_KEYS:
                        pj = {}
                        for arm, _u in ARMS:
                            r = next(x for x in preds
                                     if x["state_id"] == sid and x["model"] == model
                                     and x["H"] == h and x["action"] == arm)
                            pj[arm] = (r["J_hat"] if r["J_hat"] is not None
                                       else math.inf)
                        pred_d = pj[a1] - pj[a2]
                        f.write(f"{sid},{h},{a1}-{a2},{model},{pred_d!r},"
                                f"{truth_d!r},{se!r},{resolved}\n")

    # ---- choices_and_regret.csv ---------------------------------------------
    sels = []
    with (OUT / "selections.csv").open() as f:
        rdr = _csv.DictReader(f)
        for r in rdr:
            sels.append(r)
    with (OUT / "choices_and_regret.csv").open("w") as f:
        f.write("state_id,model,H_sel,selected_arm,ctrl_regret_vs_grid_best,"
                "monitor_gain_vs_native,monitor_gain_vs_fixed_up,"
                "monitor_gain_vs_fixed_down,best_arm_HINDSIGHT,"
                "second_best,gap_best_second,se_gap,note\n")
        for sid in states:
            for hsel in (4, 64):
                ranked = sorted(evals[(sid, hsel, "control")],
                                key=lambda a: evals[(sid, hsel, "control")][a])
                best_arm = ranked[0]
                gap = (evals[(sid, hsel, "control")][ranked[1]]
                       - evals[(sid, hsel, "control")][ranked[0]])
                se_gap = boot_pair_se(sid, hsel, "control", ranked[0], ranked[1])
                for model in MODELS_KEYS:
                    sel = next(r for r in sels if r["state_id"] == sid
                               and r["model"] == model and int(r["H"]) == hsel)
                    sel_arm = ARM_BY_U[str(float(sel["selected_u"]))]
                    fallback = sel["fallback_native"] == "True"
                    ctrl_sel = evals[(sid, hsel, "control")][sel_arm]
                    regret = ctrl_sel - evals[(sid, hsel, "control")][best_arm]
                    g_native = (evals[(sid, hsel, "monitor")]["NATIVE"]
                                - evals[(sid, hsel, "monitor")][sel_arm])
                    g_up = (evals[(sid, hsel, "monitor")]["LR_UP"]
                            - evals[(sid, hsel, "monitor")][sel_arm])
                    g_down = (evals[(sid, hsel, "monitor")]["LR_DOWN"]
                              - evals[(sid, hsel, "monitor")][sel_arm])
                    f.write(f"{sid},{model},H{hsel},{sel_arm},{regret!r},"
                            f"{g_native!r},{g_up!r},{g_down!r},"
                            f"{best_arm},{ranked[1]},{gap!r},{se_gap!r},"
                            f"{'FALLBACK_NATIVE' if fallback else 'OK'};"
                            f"3seed_pilot;BLOCK_PROXY\n")

    # ---- resource summary ---------------------------------------------------
    total_fw = (_C["train_fw"] + _C["model_fw"] + _C["monitor_fw"]
                + _C["control_fw"])
    total_bw = _C["train_bw"] + _C["model_bw"]
    summary = {
        "planned": PLAN,
        "actual": dict(_C),
        "caps": CAP,
        "within_caps": bool(
            _C["updates"] <= CAP["updates"] and total_fw <= CAP["total_fw"]
            and total_bw <= CAP["total_bw"] and _C["model_fw"] <= CAP["model_fw"]
            and _C["model_bw"] <= CAP["model_bw"]
            and _C["fd_dirs"] <= CAP["fd_dirs"]
            and _C["monitor_evals"] <= CAP["monitor_evals"]
            and _C["control_evals"] <= CAP["control_evals"]),
        "total_fw_actual": total_fw, "total_bw_actual": total_bw,
        "wall_label": "OBSERVATIONAL_SHARED_GPU",
        "time_to_target": "NOT_EVALUATED_SHARED_GPU",
        "wall_clock_speedup": "NOT_EVALUATED_SHARED_GPU",
        "energy_benefit": "NOT_EVALUATED_SHARED_GPU",
        "tokens_to_target_improvement": "NOT_TESTED_PHASE_B",
        "model_cost_field": "MODEL_COMPUTE_FOOTPRINT",
        "model_compute": {"model_fw": _C["model_fw"], "model_bw": _C["model_bw"],
                          "fd_dirs": _C["fd_dirs"],
                          "virtual_steps": _C["virtual_steps"],
                          "sim_input_slots": _C["sim_input_slots"],
                          "sim_nonpad_tokens": _C["sim_nonpad_tokens"],
                          "sim_valid_tokens": _C["sim_valid_tokens"],
                          "peak_mem_gb": _C["peak_mem_gb"]},
        "v_final_eval_calls": 0,
        "model_E": {f"{m}_H{h}": model_E[(m, h)] for m in MODELS_KEYS
                    for h in H_POINTS},
        "model_SSE": {f"{m}_H{h}": model_SSE[(m, h)] for m in MODELS_KEYS
                      for h in H_POINTS},
        "model_coverage": {f"{m}_H{h}": model_cov[(m, h)]
                           for m in MODELS_KEYS for h in H_POINTS},
    }
    (OUT / "resource_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    _log("analyze_done", None, json.dumps({"E_H4": {
        m: model_E[(m, 4)] for m in MODELS_KEYS}}))
    print("ANALYZE_DONE", json.dumps(summary["model_E"], indent=1))
    return summary


# ---------------------------------------------------------------------------
# main CLI
# ---------------------------------------------------------------------------

def run_cpu_tests():
    """T1–T8 + 计时器隔离（amendment §5）。委托 unittest 模块。"""
    import subprocess
    test_dir = PROJECT_ROOT / "research" / "tests"
    env = dict(os.environ)
    env["PYTHONPATH"] = (str(PROJECT_ROOT / "research" / "src") + os.pathsep
                         + str(PROJECT_ROOT / "research" / "scripts"))
    r = subprocess.run([sys.executable, "-m", "unittest", "-v",
                        "test_phase_b_gttr_v03"],
                       cwd=str(test_dir), env=env)
    if r.returncode != 0:
        raise SystemExit("CPU_TESTS_FAILED")
    print("CPU_TESTS_PASS")


def main():
    global _T0
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["preflight", "cpu_tests", "model_smoke",
                                      "native_smoke", "build", "main",
                                      "analyze", "all"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    _load_ledger()
    _T0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "preflight":
        run_preflight()
    elif args.phase == "cpu_tests":
        run_cpu_tests()
    elif args.phase == "model_smoke":
        run_model_smoke(args.gpu)
    elif args.phase == "native_smoke":
        run_native_smoke(args.gpu)
    elif args.phase == "build":
        run_build(args.gpu)
    elif args.phase == "main":
        run_main(args.gpu)
    elif args.phase == "analyze":
        run_analyze()
    else:
        run_preflight()
        run_cpu_tests()
        if run_model_smoke(args.gpu) and run_native_smoke(args.gpu):
            run_build(args.gpu)
            if run_main(args.gpu):
                run_analyze()


if __name__ == "__main__":
    main()
