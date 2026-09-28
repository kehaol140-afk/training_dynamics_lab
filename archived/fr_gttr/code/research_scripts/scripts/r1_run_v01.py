#!/usr/bin/env python3
"""GTTR-R1 目标系数辨识 runner —— preflight / fit / audit。

协议：FR_GTTR_R1_Model_To_Rule_Identification_v1.md（R1 v1，2026-09-23）。
本轮只执行 R0（准备/CPU）与 R1（FIT 辨识 + AUDIT 验证）；不启动 R2/R3。

  R0 preflight : 只读核查 + 冻结 config/manifest + CPU 测试 T1–T8
  R1a fit      : 6 state × k∈{1,2} × FIT8 -> b/K 系数 + 受限解，全部封存
  R1b audit    : 独立 AUDIT8 真实有限偏移（固定网格+两解）+ V_check256

固定公式（eta 含参考倍率 k）：
  d0 = -eta*(P*mbar + wd*theta) ; d1 = eta*P*(mbar-c)
  theta_plus = theta + d0 ; theta(a) = theta_plus + x*d0 + y*d1
  eta_new = eta*(1+x) ; alpha_new = y/(1+x)
  b = E[D^T grad V_ctrl(theta_plus)] ; K = E[D^T H V_ctrl(theta_plus) D]
  （仅两次目标方向 HVP；方向 detach；P 保留对当前样本依赖）

预算硬上限：forward<=22000；backward/VJP microcall<=3000；目标 HVP 方向<=224；
committed optimizer updates=0；V_final calls=0。失败/重试/smoke 全部计入账本。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs" / "fr_gttr_r1_objective_identification_v01"
RUNS = OUT / "runs"
A1 = PROJECT_ROOT / "outputs" / "fr_gttr_phase_a_native_calibration_v01" / "a1"
V07 = PROJECT_ROOT / "outputs" / "e3_confirmatory" / "v07"
D0 = PROJECT_ROOT / "minimind" / "dataset" / "pretrain_t2t_mini.jsonl"
MONITOR_JSONL = A1 / "data" / "v_monitor_rows4096_8192.jsonl"
VALSPLIT = PROJECT_ROOT / "outputs" / "fr_decision_utility_v01" / "validation_split.json"
PROTOCOL = PROJECT_ROOT / "FR_GTTR_R1_Model_To_Rule_Identification_v1.md"

SCHEDULE_ID = "finite_cosine_then_hold_5e-5_v01"
SEEDS = (2030, 2031, 2032)
STARTS = (46, 82)
CELLS = [(s, t) for s in SEEDS for t in STARTS]        # 按 (seed,start) 排序
KS = (1.0, 2.0)
SIM_SEEDS_BASE = 202609230100                          # 每 state 一个序列
N_FIT, N_AUDIT = 8, 8
N_MICROS = 2 * (N_FIT + N_AUDIT)                       # 32

R_PRIMARY, R_HALF = 0.05, 0.025
LOW_SIGNAL_ABS = 1e-9          # 分析用诊断阈值（非过关门）
BOOT_N, BOOT_SEED = 1000, 2026092301

CAP_TOTAL_FW = 22000
CAP_TOTAL_BW = 3000
CAP_HVP_DIRS = 224

_B1, _B2, _EPS, _WD, _TAU = 0.9, 0.999, 1e-8, 0.01, 1.0
_VSPLIT_SHA_FROZEN = "2dfdba24188aad8e4fb2fad0c28f15e281d5ec5d3a81f3f05e7ad3c17bb73259"

_C = {"train_fw": 0, "train_bw": 0, "fit_fw": 0, "fit_bw": 0, "vjp_micro": 0,
      "hvp_dirs": 0, "ctrl_fw": 0, "check_fw": 0, "ctrl_evals": 0,
      "check_evals": 0, "failures": 0, "retries": 0}
_T0 = None

FIT_CSV = OUT / "per_probe_coefficients.csv"
MODEL_CSV = OUT / "model_coefficients.csv"
SOL_CSV = OUT / "restricted_solutions.csv"
SEAL_JSON = OUT / "prediction_seal.json"
AUDIT_CSV = OUT / "audit_target_values.csv"
CHECK_CSV = OUT / "check_transfer.csv"


# ---------------------------------------------------------------------------
# 账本与硬上限
# ---------------------------------------------------------------------------

def _total_fw():
    return _C["train_fw"] + _C["fit_fw"] + _C["ctrl_fw"] + _C["check_fw"]


def _total_bw():
    return _C["train_bw"] + _C["fit_bw"] + _C["vjp_micro"]


def _replay_ledger():
    p = OUT / "resource_log.jsonl"
    if not p.exists():
        return
    last = None
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                last = json.loads(line)
            except Exception:
                continue
    if last:
        for k in _C:
            if k in last:
                _C[k] = int(last[k])


def _log(phase, cell, note="", **kw):
    rec = {"ts": time.time(), "phase": phase,
           "cell": f"{cell[0]}/{cell[1]}" if cell else None,
           "wall_s": round(time.perf_counter() - _T0, 3) if _T0 else 0.0,
           "note": note, "total_fw": _total_fw(), "total_bw": _total_bw(),
           **_C, **kw}
    RUNS.mkdir(parents=True, exist_ok=True)
    with (OUT / "resource_log.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _check_caps(up_fw=0, up_bw=0, up_dirs=0):
    if _total_fw() + up_fw > CAP_TOTAL_FW:
        raise SystemExit(f"BUDGET_BLOCKED: total forward cap {CAP_TOTAL_FW}")
    if _total_bw() + up_bw > CAP_TOTAL_BW:
        raise SystemExit(f"BUDGET_BLOCKED: backward/VJP cap {CAP_TOTAL_BW}")
    if _C["hvp_dirs"] + up_dirs > CAP_HVP_DIRS:
        raise SystemExit(f"BUDGET_BLOCKED: HVP direction cap {CAP_HVP_DIRS}")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _old_get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


# ---------------------------------------------------------------------------
# 运行时（复用冻结路径，与 Phase A / D1 相同）
# ---------------------------------------------------------------------------

def build_rt(seed, device):
    from fr import a0k_closed_loop_runner_v01 as a0k
    rt = a0k.build_live_runtime(seed, device)
    assert len(rt["optimizer"].param_groups) == 1, "param group drift"
    from fr.phase_a_tail_schedule_v01 import make_phase_a_get_lr
    rt["eta"] = make_phase_a_get_lr(rt["get_lr"])
    return rt


def load_bank(rt, seed, start):
    from fr.full_state import load_full_training_state
    p = V07 / f"seed_{seed}/states/state_step_{start:03d}.pt"
    load_full_training_state(p, rt["model"], rt["optimizer"], rt["scaler"],
                             restore_rng=False)
    rt["model"].train()
    rt["optimizer"].zero_grad(set_to_none=True)
    return p


def make_sim_stream(rt, sim_seed):
    import b0_run_v01 as b0
    return {"stream_seed": int(sim_seed),
            **b0.make_stream(rt, int(sim_seed), N_MICROS)}


def stack_batch_rt(rt, batch_indices):
    from fr.minimind_v03_runtime import stack_batch
    return stack_batch(rt["dataset"], batch_indices, rt["device"])


def _named_unique(rt):
    seen, out = set(), []
    for n, p in rt["named"]:
        if id(p) in seen:
            continue
        seen.add(id(p))
        out.append((n, p))
    return out


def _apply_theta(rt, theta_map):
    torch = rt["torch"]
    with torch.no_grad():
        for n, p in rt["named"]:
            p.copy_(theta_map[n])


def _theta_hash(rt):
    from fr.hashing import stable_hash
    return stable_hash({n: p.detach() for n, p in rt["named"]})


# ---------------------------------------------------------------------------
# 训练 probe 梯度（真实 BF16 路径 + 全局 clip）
# ---------------------------------------------------------------------------

def compute_clipped_grad(rt, micro_pair):
    """一个 effective batch（2 micro）累计梯度 + 全局 clip -> c。真实路径。"""
    torch = rt["torch"]
    model = rt["model"]
    optimizer = rt["optimizer"]
    cfg = rt["cfg"]
    accum = int(cfg["accumulation_steps"])
    _check_caps(up_fw=accum, up_bw=accum)
    optimizer.zero_grad(set_to_none=True)
    for (x, y) in micro_pair:
        with rt["autocast_ctx"]:
            res = model(x, labels=y)
            loss = (res.loss + res.aux_loss) / accum
        loss.backward()
        _C["train_fw"] += 1
        _C["train_bw"] += 1
    scaler = rt["scaler"]
    scaler.unscale_(optimizer)          # disabled scaler: no-op
    preclip = float(torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(cfg["grad_clip"])).detach().cpu())
    c = {n: p.grad.detach().clone() for n, p in rt["named"]}
    optimizer.zero_grad(set_to_none=True)
    return c, preclip, bool(preclip > _TAU)


# ---------------------------------------------------------------------------
# V_ctrl（FP32 / no-autocast / eval batch8 / sum/count / MATH 后端）
# ---------------------------------------------------------------------------

def ctrl_valid_count(rt):
    if "_r1_ctrl_count" in rt:
        return rt["_r1_ctrl_count"]
    n = 0
    for _, y in rt["val_score_tensors"]:
        n += int((y[:, 1:] != -100).sum())
    rt["_r1_ctrl_count"] = n
    return n


def eval_ctrl_plain(rt, theta_map):
    """V_ctrl 端点求值：返回全库 NLL sum（float）。确定性（MATH 后端）。"""
    import torch.nn.functional as F
    from fr.rng import capture_rng_state, restore_rng_state
    import a0g_decision_utility_v01 as a0g
    torch = rt["torch"]
    model = rt["model"]
    _check_caps(up_fw=len(rt["val_score_tensors"]))
    rng = capture_rng_state()
    was = model.training
    _apply_theta(rt, theta_map)
    model.eval()
    total = 0.0
    try:
        with torch.no_grad(), a0g._math_sdpa_context(torch):
            for x, y in rt["val_score_tensors"]:
                res = model(x)                     # FP32 no-autocast
                logits = res.logits
                tgt = y[:, 1:]
                ce = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tgt.reshape(-1), ignore_index=-100, reduction="none"
                ).view(tgt.shape[0], tgt.shape[1])
                total += float(ce.sum())
    finally:
        if was:
            model.train()
        restore_rng_state(rng)
    _C["ctrl_fw"] += len(rt["val_score_tensors"])
    _C["ctrl_evals"] += 1
    return total


def eval_ctrl_qh(rt, theta_plus, sl, N):
    """q_plus 与 H_plus 作用于 d0,d1（一次 create_graph + 两次方向 VJP）。

    方向 detach；同一 theta_plus、同一 scalar 目标（全库 sum）、同一 MATH
    后端与 FP32 精度。返回系数（已除以全库有效分母 N）与 base sum。
    """
    import torch.nn.functional as F
    from fr.rng import capture_rng_state, restore_rng_state
    import a0g_decision_utility_v01 as a0g
    torch = rt["torch"]
    model = rt["model"]
    named = _named_unique(rt)
    params = [p for _, p in named]
    nchunk = len(rt["val_score_tensors"])
    _check_caps(up_fw=nchunk, up_bw=nchunk + 2 * nchunk, up_dirs=2)
    rng = capture_rng_state()
    was = model.training
    _apply_theta(rt, theta_plus)
    model.eval()
    d0l = [sl[n]["d0"].detach() if n in sl else torch.zeros_like(p)
           for n, p in named]
    d1l = [sl[n]["d1"].detach() if n in sl else torch.zeros_like(p)
           for n, p in named]
    try:
        with a0g._math_sdpa_context(torch):
            S = None
            for x, y in rt["val_score_tensors"]:
                res = model(x)
                logits = res.logits
                tgt = y[:, 1:]
                ce = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tgt.reshape(-1), ignore_index=-100, reduction="none"
                ).view(tgt.shape[0], tgt.shape[1])
                s = ce.sum()
                S = s if S is None else S + s
            grads = torch.autograd.grad(S, params, create_graph=True,
                                        allow_unused=True)
            dot0 = None
            dot1 = None
            qsq = None
            for g, d0, d1 in zip(grads, d0l, d1l):
                if g is None:
                    continue
                t0 = torch.sum(g * d0, dtype=torch.float64)
                t1 = torch.sum(g * d1, dtype=torch.float64)
                tq = torch.sum(g * g, dtype=torch.float64)
                dot0 = t0 if dot0 is None else dot0 + t0
                dot1 = t1 if dot1 is None else dot1 + t1
                qsq = tq if qsq is None else qsq + tq
            v0 = torch.autograd.grad(dot0, params, retain_graph=True,
                                     allow_unused=True)
            v1 = torch.autograd.grad(dot1, params, allow_unused=True)
            k11 = k12d = k21d = k22 = None
            nd0 = nd1 = dd = None
            for g0, g1, d0, d1 in zip(v0, v1, d0l, d1l):
                a_ = torch.sum(d0 * d0, dtype=torch.float64)
                b_ = torch.sum(d1 * d1, dtype=torch.float64)
                c_ = torch.sum(d0 * d1, dtype=torch.float64)
                nd0 = a_ if nd0 is None else nd0 + a_
                nd1 = b_ if nd1 is None else nd1 + b_
                dd = c_ if dd is None else dd + c_
                if g0 is not None:
                    t = torch.sum(d0 * g0, dtype=torch.float64)
                    k11 = t if k11 is None else k11 + t
                    t = torch.sum(d1 * g0, dtype=torch.float64)
                    k21d = t if k21d is None else k21d + t
                if g1 is not None:
                    t = torch.sum(d0 * g1, dtype=torch.float64)
                    k12d = t if k12d is None else k12d + t
                    t = torch.sum(d1 * g1, dtype=torch.float64)
                    k22 = t if k22 is None else k22 + t
            base_sum = float(S.detach())
    finally:
        if was:
            model.train()
        restore_rng_state(rng)
    _C["fit_fw"] += nchunk
    _C["fit_bw"] += nchunk
    _C["hvp_dirs"] += 2
    _C["vjp_micro"] += 2 * nchunk
    return {"b1": float(dot0) / N, "b2": float(dot1) / N,
            "k11": float(k11) / N, "k12_direct": float(k12d) / N,
            "k21_direct": float(k21d) / N, "k22": float(k22) / N,
            "q_norm": math.sqrt(float(qsq)) / N,
            "d0_norm": math.sqrt(float(nd0)), "d1_norm": math.sqrt(float(nd1)),
            "d0_d1_dot": float(dd), "base_sum": base_sum, "base_count": N}


# ---------------------------------------------------------------------------
# V_check256（PhaseA monitor 语义：bf16 autocast、batch32、默认后端）
# ---------------------------------------------------------------------------

def make_check256(rt):
    import a0g_decision_utility_v01 as a0g
    from transformers import AutoTokenizer
    from dataset.lm_dataset import PretrainDataset
    ds = PretrainDataset(str(MONITOR_JSONL),
                         AutoTokenizer.from_pretrained(
                             str(PROJECT_ROOT / "minimind" / "model")),
                         max_length=int(rt["cfg"]["max_seq_len"]))
    n_rows = 256
    chunks = [list(range(i, i + 32)) for i in range(0, n_rows, 32)]
    assert len(chunks) == 8
    return a0g._stack_all(ds, chunks, rt["device"])


def eval_check_plain(rt, stacked, theta_map):
    """V_check256 sum/count（仅 primary 的 native/a_LR/a_2D 端点）。"""
    import torch.nn.functional as F
    from fr.rng import capture_rng_state, restore_rng_state
    torch = rt["torch"]
    model = rt["model"]
    _check_caps(up_fw=len(stacked))
    rng = capture_rng_state()
    was = model.training
    _apply_theta(rt, theta_map)
    model.eval()
    total = 0.0
    cnt = 0
    try:
        with torch.no_grad():
            for x, y in stacked:
                with rt["autocast_ctx"]:
                    res = model(x)
                logits = res.logits.float()
                tgt = y[:, 1:]
                ce = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tgt.reshape(-1), ignore_index=-100, reduction="none"
                ).view(tgt.shape[0], tgt.shape[1])
                total += float(ce.sum())
                cnt += int((tgt != -100).sum())
    finally:
        if was:
            model.train()
        restore_rng_state(rng)
    _C["check_fw"] += len(stacked)
    _C["check_evals"] += 1
    return total, cnt


# ---------------------------------------------------------------------------
# 网格与坐标
# ---------------------------------------------------------------------------

def fixed_grid_coords():
    """两半径网格合并（16 唯一坐标，含 native）。"""
    pts = {}
    for r in (R_PRIMARY, R_HALF):
        for x in (-r, 0.0, r):
            for y in (0.0, r / 2, r):
                key = (round(x, 12), round(y, 12))
                lab = "native" if key == (0.0, 0.0) else "grid"
                pts.setdefault(key, set()).add(f"r{r:g}")
    assert len(pts) == 16, f"merged grid must be 16 unique, got {len(pts)}"
    return {k: "native" if k == (0.0, 0.0) else "grid" for k in pts}


def audit_coord_list(solutions):
    """16 固定网格 + 两半径各自的 a_LR/a_2D（按坐标去重，仅复用不删除）。"""
    coords = dict(fixed_grid_coords())
    for (seed, start, k, r, sol, x, y) in solutions:
        key = (round(float(x), 12), round(float(y), 12))
        lab = f"a{sol}_r{float(r):g}"
        if key not in coords:
            coords[key] = lab
    return [(k_[0], k_[1], v) for k_, v in sorted(coords.items())]


# ---------------------------------------------------------------------------
# preflight（R0：CPU 只读 + 冻结）
# ---------------------------------------------------------------------------

DESIGN_MD = """# GTTR-R1 设计与推导记录（DESIGN_AND_DERIVATION）

协议：FR_GTTR_R1_Model_To_Rule_Identification_v1.md（2026-09-23）。本文件只
记录公式与本地接口映射，不扩展宏观理论。

## 1. 控制切片坐标（协议 §1）

在同一恢复状态 z=(theta, m_old, v_old, n0) 与当前训练样本 omega 下，按冻结
配方计算累计梯度、全局 clip、raw moments、bias correction（[FR-NATIVE-ADAMW]）：

    c = clip(g(omega))                  # 2 microbatch 累计，tau=1.0
    m_plus = beta1*m_old + (1-beta1)*c
    v_plus = beta2*v_old + (1-beta2)*c^2
    mbar = m_plus/(1-beta1**t),  t = n0+1（true step）
    vbar = v_plus/(1-beta2**t)
    P = 1/(sqrt(vbar)+eps)              # 逐 probe 计算，不冻结随机 P

参考学习率 eta = k * schedule_eta(gm)，gm = 2*n0+2（本步最后一个 micro 的
gm，与 Phase A/D1 的 per-micro 写入语义一致）；k∈{1,2}。相同样本的 k1/k2
共用相同 m_plus/v_plus/P（本配方中它们不依赖本次 LR）。

    d0 = -eta*(P*mbar + wd*theta)       # native 完整位移（含衰减）
    d1 =  eta*P*(mbar - c)
    theta_plus = theta + d0
    theta(a) = theta_plus + x*d0 + y*d1
    eta_new = eta*(1+x), alpha_new = y/(1+x),  x > -1

精确重参数化（T1 CPU 测试对照直接读出式）：
    theta_new = (1-eta_new*wd)*theta
                - eta_new*P*((1-alpha_new)*mbar + alpha_new*c)

## 2. 目标系数（协议 §2）

V_ctrl = V_control_ref64：outputs/fr_decision_utility_v01/validation_split.json
的 v_score 64 行；FP32、autocast off、eval mode、batch 8、全库 sum/count。
q/H 与全部端点求值统一使用 MATH SDPA 后端（二阶可微；协议允许的已知数学
后端例外），记录 base loss/q/dtype/backend。

    b_i = D_i^T grad V_ctrl(theta_plus)      （全库 sum 梯度 / N_total）
    K_ij = D_i^T Hessian V_ctrl(theta_plus) D_j

实现：8 chunk 前向累计 S=sum(ce)；一次 create_graph 一阶梯度；dot0/dot1
（方向 detach，fp64 dot）；两次二阶 VJP 得 H d0、H d1；所有系数除以全库
有效分母 N_total（不能无权平均 batch means）。K12_direct/K21_direct 均保存，
相对非对称 <=1e-4 才用 (K+K^T)/2 求解；原值照报。

## 3. 受限求解（协议 §2.1）

m(a)=b^T a+0.5 a^T K a；主可行域 x∈[-r,r], y∈[0,r]，r_primary=.05（主），
r_half=.025（固定尺度诊断）。矩形域二次函数完整候选枚举：4 角点、原点、
四边内部极小点、可行内部驻点（cond<1e14 才尝试）；按 (m, y, |x|, x) 确定性
选优；保留负特征值，不 PSD 裁剪，不加 jitter；无法数值求解 -> UNRESOLVED。
Schur 消元仅在 K 正定且 K11、K_eff 非退化时输出 b_eff/K_eff/y_star/x_star。
y=0 的 KKT 依据：K 半正定（容差 1e-12 相对）且 b2+K21*x_LR>=0。

## 4. 状态/probe/预算映射（协议 §3、§7）

- 6 state：seeds 2030-2032 × start 46/82（v07 bank）；真实 n0 从 checkpoint 读。
- 每 state 16 个 effective-batch probes（FIT 8 / AUDIT 8，各 2 micro）；
  SIM seeds 202609230100..105（按 (seed,start) 排序；冲突整块 +1000，结果前冻结）。
  采样：derive_stream_seed(sim_seed, 404) -> make_permutation(3936) -> batch 4
  （[FR-BATCH-STREAM] 冻结 train pool 内，不碰 probe/eval）。
- k1/k2 共享 16 probes；FIT 全部封存（prediction_seal.json 覆盖
  per_probe_coefficients.csv / model_coefficients.csv / restricted_solutions.csv）
  后才读 AUDIT；AUDIT 不回写系数（T8）。
- AUDIT 坐标：两半径网格 x∈{-r,0,r}×y∈{0,r/2,r} 合并 16 唯一坐标 + 两半径
  各自 a_LR/a_2D（坐标去重仅复用）；native 差严格 0（同参数确定性求值）。
- V_check256：monitor 前 256 行（ID 4096..4351），batch 32，PhaseA 语义
  （bf16 autocast + logits.float()），仅 primary 的 native/a_LR/a_2D。
- 预算（协议 §7）：train fw/bw 192；FIT q/H fw 768、一阶 bw 768、VJP 方向
  192（micro 口径 1536）；AUDIT ctrl fw<=15360+base 1536；check fw 2304；
  全部计入 resource_log.jsonl 硬上限。

## 5. 分析口径（协议 §5）

- bootstrap：effective-batch probe 为单位，1000 次，PCG64 seed 2026092301；
  FIT 与 AUDIT 独立重抽；k1/k2 与动作保持配对（同 state 共用索引抽）。
- 模型误差：固定网格（16 坐标，不含必为 0 的 native）SSE/TSS；
  max|J_audit|<1e-9 -> LOW_SIGNAL（诊断阈值，非过关门）。
- 判定标签：MODEL_RECOMMENDS_NO_READOUT_CHANGE / PREDICTED_BENEFIT_NOT_
  CONFIRMED / TARGET_TRANSFER_NOT_ESTABLISHED / UNRESOLVED_*；
  汇总按 seed/start/k 分层（6x2 条件不是 12 个独立 seeds）。
"""


def run_preflight():
    import torch
    from fr.phase_a_tail_schedule_v01 import make_phase_a_get_lr

    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    rep = {}

    # 1) D0 / v_monitor 身份
    n_lines = 0
    h = hashlib.sha256()
    with D0.open("rb") as f:
        for line in f:
            n_lines += 1
            h.update(line)
    d0_sha = h.hexdigest()
    assert n_lines == 1270238 and d0_sha.startswith("6dd6716c"), \
        f"D0 identity conflict: n={n_lines} sha={d0_sha[:16]}"
    assert MONITOR_JSONL.exists()
    mon_lines = sum(1 for _ in MONITOR_JSONL.open("rb"))
    assert mon_lines == 4096
    mon_sha = _sha256(MONITOR_JSONL)
    assert mon_sha.startswith("607ba1cd"), f"v_monitor sha {mon_sha[:16]}"
    rep["d0"] = {"path": str(D0), "lines": n_lines, "sha256": d0_sha}
    rep["v_monitor"] = {"path": str(MONITOR_JSONL), "rows": "[4096:8192)",
                        "sha256": mon_sha}

    # 2) V_control_ref64 split 冻结
    vs = json.loads(VALSPLIT.read_text())
    assert vs["split_sha256"] == _VSPLIT_SHA_FROZEN, "validation split drift"
    assert len(vs["v_score"]) == 64
    rep["v_control_ref64"] = {
        "n_rows": 64, "rule": vs["rule"],
        "first8_sorted": sorted(vs["v_score"])[:8],
        "split_sha256": vs["split_sha256"]}

    # 3) bank 6 checkpoint
    bank = {}
    for seed, start in CELLS:
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
        bank[f"{seed}/{start}"] = {
            "path": str(p), "n0": start, "global_micro": 2 * start,
            "model_tensors": len(st["model"]),
            "optimizer_state_entries": len(st["optimizer"]["state"]),
            "sha256": _sha256(p)}
    rep["bank"] = bank

    # 4) 配方核对（Phase A 最近纠正版）
    a1cfg = json.loads((A1 / "config.json").read_text())
    a1train = a1cfg["train"]
    recipe_expect = {"optimizer": "torch.optim.AdamW", "betas": [0.9, 0.999],
                     "eps": 1e-8, "weight_decay": 0.01, "clip_tau": 1.0,
                     "batch_size": 4, "accumulation_steps": 2}
    conflicts = {k: {"a1": a1train.get(k), "expected": v}
                 for k, v in recipe_expect.items() if a1train.get(k) != v}
    assert not conflicts, f"recipe conflict vs Phase A: {conflicts}"
    assert a1cfg["schedule_id"] == SCHEDULE_ID
    rep["recipe"] = {"from": str(A1 / "config.json"), "conflicts": conflicts}

    # 5) schedule 边界 + 各 state 的 applied eta
    eta = make_phase_a_get_lr(_old_get_lr)
    assert abs(eta(256, 256, 5e-4) - 5e-5) < 1e-12
    assert eta(257, 256, 5e-4) == 5e-5
    etas = {}
    for seed, start in CELLS:
        gm = 2 * start + 2
        e1 = float(eta(gm, 256, 5e-4))
        etas[f"{seed}/{start}"] = {"gm": gm, "eta_native": e1,
                                   "eta_k1": e1, "eta_k2": 2 * e1}
    rep["etas"] = etas
    rep["schedule"] = {"id": SCHEDULE_ID, "configured_peak": 5e-4,
                       "note": "记录 configured_peak 与 applied_eta；"
                               "start46 的 peak 不是当前 LR"}

    # 6) SIM seed 冲突核查（纯元数据；整块 +1000）
    used = set()
    pat = re.compile(r"(?<!\d)(\d{12,13})(?!\d)")
    for p in PROJECT_ROOT.glob("outputs/**/*.json"):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        for mch in pat.finditer(txt):
            v = mch.group(1)
            if len(v) >= 12:
                used.add(int(v))
    base = SIM_SEEDS_BASE
    proposed = {cell: base + i for i, cell in enumerate(CELLS)}
    shifted = False
    while any(s in used for s in proposed.values()):
        base += 1000
        proposed = {cell: base + i for i, cell in enumerate(CELLS)}
        shifted = True
    rep["sim_seeds"] = {"proposed_base": SIM_SEEDS_BASE, "final_base": base,
                        "shifted": shifted,
                        "by_cell": {f"{c[0]}/{c[1]}": proposed[c] for c in CELLS},
                        "metadata_only_decision": True}
    globals()["SIM_FINAL"] = proposed

    # 7) CPU 测试 T1-T8
    from fr import r1_objective_identification_v01 as r1m
    test_mod = "tests.test_r1_objective_identification_v01"
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "unittest", test_mod, "-v"],
                       cwd=PROJECT_ROOT / "research", capture_output=True,
                       text=True,
                       env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src",
                            "HOME": "/home/b205"})
    out_txt = r.stdout + r.stderr
    passed = r.returncode == 0 and out_txt.strip().splitlines()[-1].strip() == "OK"
    impl = {"unittest_module": test_mod, "returncode": r.returncode,
            "tail_line": out_txt.strip().splitlines()[-1].strip()
            if out_txt.strip() else "",
            "all_pass": bool(passed), "wall_s": round(time.time() - t0, 3),
            "t1_to_t8": {
                "T1_exact_coordinate_mapping": "covered_by_suite",
                "T2_moments_P_grad_semantics": "covered_by_suite",
                "T3_known_quadratic_b_K_and_grid": "covered_by_suite",
                "T4_pd_indef_singular_boundary_noclip": "covered_by_suite",
                "T5_schur_validity_and_collinearity": "covered_by_suite",
                "T6_directions_detached": "covered_by_suite",
                "T7_mean_adam_and_cross_probe": "covered_by_suite",
                "T8_seal_native0_sumcount_se_budget_labels": "covered_by_suite"},
            "module": {"path": str(Path(r1m.__file__).resolve()),
                       "sha256": _sha256(Path(r1m.__file__).resolve())}}
    (OUT / "implementation_checks.json").write_text(json.dumps(impl, indent=1))
    rep["implementation"] = {"all_pass": bool(passed)}
    assert passed, f"CPU unittest not clean:\n{out_txt[-2000:]}"

    # 8) 冻结 config / manifest / DESIGN
    plan = r1m.plan_budget()
    config = {
        "round": "fr_gttr_r1_objective_identification_v01",
        "protocol": "FR_GTTR_R1_Model_To_Rule_Identification_v1.md",
        "protocol_sha256": _sha256(PROTOCOL),
        "cells": [{"seed": s, "start": t, "sim_seed": proposed[(s, t)]}
                  for (s, t) in CELLS],
        "k_values": list(KS),
        "schedule": {"id": SCHEDULE_ID, "configured_peak": 5e-4,
                     "eta_by_cell": {f"{s}/{t}": etas[f"{s}/{t}"]
                                     for (s, t) in CELLS}},
        "probes": {"fit": N_FIT, "audit": N_AUDIT,
                   "micro_per_probe": 2, "micros_per_state": N_MICROS,
                   "sim_seed_rule": "202609230100+idx by (seed,start) sorted; "
                                    "conflict block-shift +1000 (metadata only)",
                   "fit_micros": [0, 15], "audit_micros": [16, 31],
                   "sampler": "derive_stream_seed(sim,404)->permutation(3936)"
                              "->batch4; frozen train pool only"},
        "radii": {"primary": R_PRIMARY, "half": R_HALF,
                  "role": "固定尺度诊断；primary 报告为主；不得看 AUDIT 后选主半径"},
        "audit_grid": {"per_radius": "x in {-r,0,r} x y in {0,r/2,r}",
                       "merged_unique": 16, "plus": "a_LR,a_2D per radius"},
        "objectives": {
            "v_ctrl": {"rows": 64, "dtype": "FP32", "autocast": "off",
                       "eval_batch": 8, "agg": "sum/count full library",
                       "backend": "MATH SDPA（q/H 与端点统一；记录 base loss）",
                       "role": "建模目标，不是 final"},
            "v_check256": {"rows": "[4096,4351) by ID", "eval_batch": 32,
                           "dtype": "bf16 autocast + logits.float()",
                           "role": "primary 半径 native/a_LR/a_2D 目标迁移检查；"
                                   "不参与求导和求解"},
            "v_final": {"calls": 0}},
        "numerics": {"rel_tol": r1m.REL_TOL,
                     "k_asym_rel_tol": r1m.K_ASYM_REL_TOL,
                     "cond_skip_interior": r1m.COND_SKIP_INTERIOR,
                     "low_signal_abs": LOW_SIGNAL_ABS,
                     "no_psd_clip": True, "no_jitter": True,
                     "tie_break": "(m, y, |x|, x)"},
        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "unit": "effective-batch",
                      "fit_audit_independent": True,
                      "paired_k_and_actions": True},
        "budget": {"caps": {"total_fw": CAP_TOTAL_FW,
                            "total_bw_vjp_micro": CAP_TOTAL_BW,
                            "hvp_dirs": CAP_HVP_DIRS,
                            "committed_updates": 0, "v_final_calls": 0},
                   "plan": plan,
                   "extra_base_eval_note": "AUDIT base(theta_plus) 独立计费以"
                                           "验证 native 差严格 0；计划总额含余量"},
        "code": {"git_head": _git_head(),
                 "runner": str(Path(__file__).resolve()),
                 "runner_sha256": _sha256(Path(__file__).resolve())},
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=1) + "\n")

    # state_probe_manifest（实际 batch 索引；CPU 确定性生成）
    from fr.stage_c_runtime import batches_from_train_pool
    from fr.stream_rng import derive_stream_seed
    partition = json.loads((V07 / "partition_v07.json").read_text())
    train_idx = list(map(int, partition["train_indices"]))
    manifest = {}
    for (seed, start) in CELLS:
        sim = proposed[(seed, start)]
        bseed = derive_stream_seed(sim, 404)
        batches = batches_from_train_pool(train_idx, 4, bseed, N_MICROS)
        manifest[f"{seed}/{start}"] = {
            "bank": bank[f"{seed}/{start}"], "sim_seed": sim,
            "batch_seed": bseed,
            "effective_batches": [[int(i) for i in b] for b in
                                  (batches[2 * j] for j in range(16))],
            "micro_batches": [[int(i) for i in b] for b in batches],
            "fit_effective": list(range(8)), "audit_effective": list(range(8, 16)),
            "k_shares_probes": True}
    (OUT / "state_probe_manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n")
    (OUT / "DESIGN_AND_DERIVATION.md").write_text(DESIGN_MD)

    print(json.dumps({"bank_ok": len(bank) == 6, "etas": etas,
                      "sim_seeds": {f"{c[0]}/{c[1]}": proposed[c]
                                    for c in CELLS},
                      "unittest_pass": passed, "out": str(OUT)}, indent=1))
    print("PREFLIGHT_OK")


# ---------------------------------------------------------------------------
# 通用：读取已完成行（resume）
# ---------------------------------------------------------------------------

def _csv_keys(path, keycols):
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines()[1:]:
            if not line.strip():
                continue
            row = line.split(",")
            keys.add(tuple(row[:len(keycols)]))
    return keys


def _append_csv(path, header, rows):
    if not path.exists():
        path.write_text(header)
    with path.open("a") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def _fmt(v):
    return repr(float(v))


FIT_HEADER = ("seed,start,k,probe,eta,t,gm,preclip_norm,clip_active,b1,b2,"
              "k11,k12_direct,k21_direct,k22,asym_rel,q_norm,d0_norm,d1_norm,"
              "d0_d1_dot,base_sum,base_count,finite\n")


# ---------------------------------------------------------------------------
# R1a：FIT 系数辨识
# ---------------------------------------------------------------------------

def run_fit(gpu):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    _replay_ledger()
    from fr import r1_objective_identification_v01 as r1m
    from fr.stream_rng import seed_future_stream
    cfg_all = json.loads((OUT / "config.json").read_text())
    manifest = json.loads((OUT / "state_probe_manifest.json").read_text())
    device = f"cuda:{gpu}"

    done = _csv_keys(FIT_CSV, ("seed", "start", "k", "probe"))
    smoke_written = (RUNS / "smoke.json").exists()

    for seed in SEEDS:
        rt = build_rt(seed, device)
        N = ctrl_valid_count(rt)
        for start in STARTS:
            cell = (seed, start)
            key = f"{seed}/{start}"
            sim = manifest[key]["sim_seed"]
            load_bank(rt, seed, start)
            theta0 = {n: p.detach().clone() for n, p in rt["named"]}
            h0 = _theta_hash(rt)
            m_old = {}
            v_old = {}
            for n, p in rt["named"]:
                st = rt["optimizer"].state[p]
                m_old[n] = st["exp_avg"].detach().clone()
                v_old[n] = st["exp_avg_sq"].detach().clone()
            t = start + 1
            gm = 2 * start + 2
            eta_native = float(rt["eta"](gm, 256, 5e-4))
            seed_future_stream(sim)
            stream = make_sim_stream(rt, sim)
            batches = stream["batches"]
            assert len(batches) == N_MICROS
            for probe in range(N_FIT + N_AUDIT):
                role = "FIT" if probe < N_FIT else "AUDIT"
                if role != "FIT":
                    continue
                micro_pair = [stack_batch_rt(rt, batches[2 * probe]),
                              stack_batch_rt(rt, batches[2 * probe + 1])]
                todo = [k for k in KS
                        if (str(seed), str(start), _kstr(k), str(probe))
                        not in done]
                if not todo:
                    continue
                c, preclip, clip_act = compute_clipped_grad(rt, micro_pair)
                if not math.isfinite(preclip):
                    _C["failures"] += 1
                    _log("fit", cell, f"probe{probe} nonfinite preclip",
                         failures=1)
                    raise SystemExit(f"NONFINITE preclip {key} probe {probe}")
                rows = []
                for k in todo:
                    eta = k * eta_native
                    sl = r1m.control_slice(theta0, c, m_old, v_old, t,
                                           _B1, _B2, _EPS, _WD, eta)
                    slim = {n: {"d0": s["d0"], "d1": s["d1"]}
                            for n, s in sl.items()}
                    tp = r1m.make_theta_plus(theta0, sl)
                    co = eval_ctrl_qh(rt, tp, slim, N)
                    finite = all(math.isfinite(co[q]) for q in
                                 ("b1", "b2", "k11", "k12_direct",
                                  "k21_direct", "k22"))
                    if not finite:
                        _C["failures"] += 1
                        _log("fit", cell, f"probe{probe} k{k} nonfinite coeff",
                             failures=1)
                        raise SystemExit(
                            f"NONFINITE coeff {key} probe {probe} k {k}")
                    asym = abs(co["k12_direct"] - co["k21_direct"]) / \
                        max(1.0, abs(co["k12_direct"]), abs(co["k21_direct"]))
                    rows.append((seed, start, _kstr(k), probe, _fmt(eta), t,
                                 gm, _fmt(preclip), clip_act,
                                 _fmt(co["b1"]), _fmt(co["b2"]),
                                 _fmt(co["k11"]), _fmt(co["k12_direct"]),
                                 _fmt(co["k21_direct"]), _fmt(co["k22"]),
                                 _fmt(asym), _fmt(co["q_norm"]),
                                 _fmt(co["d0_norm"]), _fmt(co["d1_norm"]),
                                 _fmt(co["d0_d1_dot"]), _fmt(co["base_sum"]),
                                 co["base_count"], finite))
                    _apply_theta(rt, theta0)
                    if _theta_hash(rt) != h0:
                        _C["failures"] += 1
                        _log("fit", cell, "theta restore mismatch", failures=1)
                        raise SystemExit(f"THETA_RESTORE_MISMATCH {key}")
                    if not smoke_written:
                        (RUNS / "smoke.json").write_text(json.dumps({
                            "cell": key, "probe": probe, "k": k,
                            "note": "第一固定 FIT 样本数值 smoke；不计调参；"
                                    "版本/语义一致时结果保留",
                            "coeffs": {q: co[q] for q in
                                       ("b1", "b2", "k11", "k12_direct",
                                        "k21_direct", "k22", "base_sum")},
                            "finite": True}, indent=1) + "\n")
                        smoke_written = True
                        _log("fit_smoke", cell, "first FIT probe finite ok")
                _append_csv(FIT_CSV, FIT_HEADER, rows)
                _log("fit", cell, f"probe{probe} k={todo} done")
            # optimizer step 未动（committed updates = 0）
            for n, p in rt["named"]:
                st = rt["optimizer"].state[p]
                assert float(st["step"]) == float(start), \
                    "optimizer step drifted (committed_updates must stay 0)"
        del rt
        import torch
        torch.cuda.empty_cache()
    finalize_fit()
    print("FIT_DONE")


def _kstr(k):
    return f"{float(k):g}"


def finalize_fit():
    """聚合 FIT 系数 -> model_coefficients.csv + restricted_solutions.csv + seal。"""
    from fr import r1_objective_identification_v01 as r1m
    rows = []
    for line in FIT_CSV.read_text().splitlines()[1:]:
        if line.strip():
            rows.append(line.split(","))
    conds = {}
    for r in rows:
        key = (r[0], r[1], r[2])
        conds.setdefault(key, []).append(
            {"b1": float(r[9]), "b2": float(r[10]), "k11": float(r[11]),
             "k12_direct": float(r[12]), "k21_direct": float(r[13]),
             "k22": float(r[14]), "asym_rel": float(r[15]),
             "d0_norm": float(r[17]), "d1_norm": float(r[18]),
             "d0_d1_dot": float(r[19])})
    mrows, srows = [], []
    for key in sorted(conds, key=lambda kk: (int(kk[0]), int(kk[1]),
                                             float(kk[2]))):
        probes = conds[key]
        agg = r1m.mean_k_operators(probes)
        sym = r1m.symmetrized_K(agg["k12_direct"], agg["k21_direct"],
                                agg["k11"], agg["k22"])
        b = [agg["b1"], agg["b2"]]
        sch = r1m.schur_elimination(b, sym["K"])
        eig, cond = r1m.eig_cond(sym["K"])
        d0n = [p_["d0_norm"] for p_ in probes]
        d1n = [p_["d1_norm"] for p_ in probes]
        dd = [p_["d0_d1_dot"] for p_ in probes]
        cos_list = [x / max(a * b_, 1e-300)
                    for x, a, b_ in zip(dd, d0n, d1n)]
        cos = sum(cos_list) / len(cos_list)
        import statistics as stats
        mrows.append((key[0], key[1], key[2], agg["n_probes"],
                      _fmt(agg["b1"]), _fmt(r1m.paired_se(
                          [p_["b1"] for p_ in probes])),
                      _fmt(agg["b2"]), _fmt(r1m.paired_se(
                          [p_["b2"] for p_ in probes])),
                      _fmt(agg["k11"]), _fmt(agg["k12_direct"]),
                      _fmt(agg["k21_direct"]), _fmt(agg["k22"]),
                      _fmt(eig[0]), _fmt(eig[1]), _fmt(cond),
                      _fmt(stats.mean(d0n)), _fmt(stats.mean(d1n)),
                      _fmt(stats.mean(dd)), _fmt(cos),
                      sch["valid"], sch.get("reason", ""),
                      _fmt(sch["b_eff"]) if sch.get("b_eff") is not None else "NA",
                      _fmt(sch["K_eff"]) if sch.get("K_eff") is not None else "NA",
                      _fmt(sch["y_star"]) if sch.get("y_star") is not None else "NA",
                      sym["ok"], _fmt(sym["rel_asym"])))
        if not sym["ok"]:
            for r_ in (R_PRIMARY, R_HALF):
                srows.append((key[0], key[1], key[2], _fmt(r_), "LR", "NA",
                              "NA", "NA", "K_ASYMMETRY_EXCEEDED"))
                srows.append((key[0], key[1], key[2], _fmt(r_), "2D", "NA",
                              "NA", "NA", "K_ASYMMETRY_EXCEEDED"))
            continue
        for r_ in (R_PRIMARY, R_HALF):
            lr = r1m.solve_lr_only(b, sym["K"], r_)
            sol = r1m.solve_rectangle(b, sym["K"], r_, r_)
            kkt = r1m.kkt_y0_flag(b, sym["K"], lr["x"])
            eta_mult = 1.0 + lr["x"]
            srows.append((key[0], key[1], key[2], _fmt(r_), "LR",
                          _fmt(lr["x"]), "0", _fmt(lr["m"]),
                          f"boundary={lr['boundary']};flat={lr['flat']};"
                          f"kkt_y0={kkt['kkt_ok']};eta_mult={eta_mult:.6f}"))
            eta_mult2 = 1.0 + sol["x"]
            alpha2 = sol["y"] / eta_mult2 if eta_mult2 > 0 else float("nan")
            srows.append((key[0], key[1], key[2], _fmt(r_), "2D",
                          _fmt(sol["x"]), _fmt(sol["y"]), _fmt(sol["m"]),
                          f"tag={sol['tag']};interior={sol['interior_status']};"
                          f"y_star={sch['y_star'] if sch['valid'] else 'NA'};"
                          f"eta_mult={eta_mult2:.6f};alpha={alpha2:.6f}"))
    _append_csv(MODEL_CSV,
                "seed,start,k,n_probes,b1_mean,b1_se,b2_mean,b2_se,k11_mean,"
                "k12_direct_mean,k21_direct_mean,k22_mean,eig_min,eig_max,"
                "cond_K,d0_norm_mean,d1_norm_mean,dot_mean,cos_mean,"
                "schur_valid,schur_reason,b_eff,K_eff,y_star,asym_ok,asym_rel\n",
                mrows)
    _append_csv(SOL_CSV,
                "seed,start,k,r,solution,x,y,m,flags\n", srows)
    r1m.write_seal(SEAL_JSON, [FIT_CSV, MODEL_CSV, SOL_CSV],
                   extra={"n_conditions": len(conds), "phase": "R1a"})
    _log("fit_sealed", None, f"conditions={len(conds)}")


# ---------------------------------------------------------------------------
# R1b：AUDIT 真实有限偏移
# ---------------------------------------------------------------------------

def _load_solutions():
    sols = []
    for line in SOL_CSV.read_text().splitlines()[1:]:
        if line.strip():
            r = line.split(",")
            sols.append((int(r[0]), int(r[1]), float(r[2]), float(r[3]),
                         r[4], float(r[5]), float(r[6])))
    return sols


def run_audit(gpu):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    _replay_ledger()
    from fr import r1_objective_identification_v01 as r1m
    from fr.stream_rng import seed_future_stream
    ok, why = r1m.verify_seal(SEAL_JSON, [FIT_CSV, MODEL_CSV, SOL_CSV])
    assert ok, f"seal verification failed ({why}); FIT 必须先封存且不可回写"
    manifest = json.loads((OUT / "state_probe_manifest.json").read_text())
    device = f"cuda:{gpu}"
    sols = _load_solutions()

    # resume：已完成 (cond, probe) 的坐标集合 / check 标签集合
    done_coord, done_checklab = set(), set()
    if AUDIT_CSV.exists():
        for line in AUDIT_CSV.read_text().splitlines()[1:]:
            if line.strip():
                r = line.split(",")
                done_coord.add((r[0], r[1], r[2], r[3],
                                round(float(r[5]), 12), round(float(r[6]), 12)))
    if CHECK_CSV.exists():
        for line in CHECK_CSV.read_text().splitlines()[1:]:
            if line.strip():
                r = line.split(",")
                done_checklab.add((r[0], r[1], r[2], r[3], r[4]))

    import torch
    for seed in SEEDS:
        rt = build_rt(seed, device)
        N = ctrl_valid_count(rt)
        stacked_check = make_check256(rt)
        for start in STARTS:
            cell = (seed, start)
            key = f"{seed}/{start}"
            sim = manifest[key]["sim_seed"]
            load_bank(rt, seed, start)
            theta0 = {n: p.detach().clone() for n, p in rt["named"]}
            h0 = _theta_hash(rt)
            m_old, v_old = {}, {}
            for n, p in rt["named"]:
                st = rt["optimizer"].state[p]
                m_old[n] = st["exp_avg"].detach().clone()
                v_old[n] = st["exp_avg_sq"].detach().clone()
            t = start + 1
            gm = 2 * start + 2
            eta_native = float(rt["eta"](gm, 256, 5e-4))
            seed_future_stream(sim)
            stream = make_sim_stream(rt, sim)
            batches = stream["batches"]
            cond_sols = [s for s in sols if s[0] == seed and s[1] == start]
            coords = audit_coord_list(cond_sols)
            for probe in range(N_FIT, N_FIT + N_AUDIT):
                micro_pair = [stack_batch_rt(rt, batches[2 * probe]),
                              stack_batch_rt(rt, batches[2 * probe + 1])]
                c = None
                for k in KS:
                    pkey = (str(seed), str(start), _kstr(k), str(probe))
                    pend_coords = [(x, y, lab) for (x, y, lab) in coords
                                   if (pkey[0], pkey[1], pkey[2], pkey[3],
                                       round(x, 12), round(y, 12))
                                   not in done_coord]
                    prim = [s for s in cond_sols if s[2] == k
                            and abs(s[3] - R_PRIMARY) < 1e-15]
                    pend_check = []
                    for s in prim:
                        lab = f"a{s[4]}"
                        if (pkey[0], pkey[1], pkey[2], pkey[3], lab) \
                                not in done_checklab:
                            pend_check.append((lab, s[5], s[6]))
                    if not pend_coords and not pend_check:
                        continue
                    if c is None:
                        c, preclip, clip_act = compute_clipped_grad(
                            rt, micro_pair)
                        if not math.isfinite(preclip):
                            _C["failures"] += 1
                            _log("audit", cell, f"probe{probe} nonfinite",
                                 failures=1)
                            raise SystemExit("NONFINITE in audit")
                    eta = k * eta_native
                    sl = r1m.control_slice(theta0, c, m_old, v_old, t,
                                           _B1, _B2, _EPS, _WD, eta)
                    tp = r1m.make_theta_plus(theta0, sl)
                    if pend_coords:
                        base_sum = eval_ctrl_plain(rt, tp)
                        rows = []
                        for (x, y, lab) in pend_coords:
                            th_a = tp if (x == 0.0 and y == 0.0) else \
                                r1m.theta_at_a(tp, sl, sl, x, y)
                            S = eval_ctrl_plain(rt, th_a)
                            if lab == "native":
                                assert S == base_sum, \
                                    f"native delta != 0: {S} vs {base_sum}"
                            rows.append((seed, start, _kstr(k), probe, lab,
                                         _fmt(x), _fmt(y),
                                         _fmt((S - base_sum) / N),
                                         _fmt(S), N))
                        _append_csv(AUDIT_CSV,
                                    "seed,start,k,probe,coord_label,x,y,"
                                    "delta_v,sum,count\n", rows)
                    crows = []
                    if pend_check:
                        for lab, xs, ys in pend_check:
                            assert abs(xs) <= R_PRIMARY + 1e-12 and \
                                0.0 <= ys <= R_PRIMARY + 1e-12, \
                                "solution out of box"
                            th = tp if (xs == 0.0 and ys == 0.0) else \
                                r1m.theta_at_a(tp, sl, sl, xs, ys)
                            S_c, cnt = eval_check_plain(rt, stacked_check, th)
                            crows.append((seed, start, _kstr(k), probe, lab,
                                          _fmt(xs), _fmt(ys), _fmt(S_c), cnt,
                                          _fmt(S_c / max(cnt, 1))))
                        if (pkey[0], pkey[1], pkey[2], pkey[3], "native") \
                                not in done_checklab:
                            th = tp
                            S_c, cnt = eval_check_plain(rt, stacked_check, th)
                            crows.insert(0, (seed, start, _kstr(k), probe,
                                             "native", "0", "0", _fmt(S_c),
                                             cnt, _fmt(S_c / max(cnt, 1))))
                        _append_csv(CHECK_CSV,
                                    "seed,start,k,probe,label,x,y,sum,count,"
                                    "mean\n", crows)
                    if pend_coords or pend_check:
                        _apply_theta(rt, theta0)
                        if _theta_hash(rt) != h0:
                            _C["failures"] += 1
                            _log("audit", cell, "theta restore mismatch",
                                 failures=1)
                            raise SystemExit(f"THETA_RESTORE_MISMATCH {key}")
                        _log("audit", cell,
                             f"probe{probe} k={k} coords={len(pend_coords)} "
                             f"check={len(crows)}")
            for n, p in rt["named"]:
                st = rt["optimizer"].state[p]
                assert float(st["step"]) == float(start), \
                    "optimizer step drifted (committed_updates must stay 0)"
        del rt, stacked_check
        torch.cuda.empty_cache()
    print("AUDIT_DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["preflight", "fit", "audit", "all"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    if args.phase == "preflight":
        run_preflight()
    elif args.phase == "fit":
        run_fit(args.gpu)
    elif args.phase == "audit":
        run_audit(args.gpu)
    else:
        run_preflight()
        run_fit(args.gpu)
        run_audit(args.gpu)


if __name__ == "__main__":
    main()
