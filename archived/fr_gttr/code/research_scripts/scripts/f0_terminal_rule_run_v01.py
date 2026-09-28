#!/usr/bin/env python3
"""GTTR-F0 全时域终点规则求解 runner。

阶段：preflight → smoke → calibrate → grid → solve → audit → confirm。
协议：FR_GTTR_F0_Full_Horizon_Terminal_Rule_Solver_v1.md。

- 规则：eta[t]=w[t]*exp(a+b*psi[t])/Z(b)，w=2*eta_base（start46 首 gm=94，
  尾部 1e-4），a,b∈[-log2,log2]；REF32 搜索（FP32/autocast off/MATH/TF32
  highest），NATIVE 确认（BF16）。
- 全状态前向灵敏度：S_theta/S_m/S_v 传到终点（FULL 256 步；SHORT 在 64
  读目标）；每步真实 HVP；clip 用冻结 a0c.clip_jvp；d_eta/db 含 Z 导数。
- 预算硬上限：traj_updates<=34000；network fw<=110000；bw/VJP<=90000；
  HVP dirs<=7000；design evals<=250；confirm evals<=40；target grad<=24；
  V_final=0。失败/试步/重放全部计费；wall 仅 OBSERVATIONAL_SHARED_GPU。
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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs" / "fr_gttr_f0_terminal_rule_solver_v01"
RUNS = OUT / "runs"
CONF_DATA = OUT / "data"
A1 = PROJECT_ROOT / "outputs" / "fr_gttr_phase_a_native_calibration_v01" / "a1"
V07 = PROJECT_ROOT / "outputs" / "e3_confirmatory" / "v07"
D0 = PROJECT_ROOT / "minimind" / "dataset" / "pretrain_t2t_mini.jsonl"
R1_OUT = PROJECT_ROOT / "outputs" / "fr_gttr_r1_objective_identification_v01"
PROTOCOL = PROJECT_ROOT / "FR_GTTR_F0_Full_Horizon_Terminal_Rule_Solver_v1.md"

SEEDS = (2030, 2031, 2032)
START = 46
FIT = {2030: None, 2031: None}          # stream ids（preflight 填）
AUDIT = {2030: None, 2031: None, 2032: None}
STREAM_BASE = 202609250400

T_MAIN = 256
H_SHORT = 64
LOG2 = math.log(2.0)

CAP_TRAJ = 34000
CAP_FW = 110000
CAP_BW = 90000
CAP_HVP_DIRS = 7000
CAP_DESIGN_EVALS = 250
CAP_CONFIRM_EVALS = 40
CAP_TARGET_GRAD = 24

B1_, B2_, EPS_, WD_, TAU_ = 0.9, 0.999, 1e-8, 0.01, 1.0

_C = {"traj_updates": 0, "train_fw": 0, "train_bw": 0, "hvp_dirs": 0,
      "hvp_micro": 0, "design_fw": 0, "design_evals": 0,
      "design_grad_evals": 0, "confirm_fw": 0, "confirm_evals": 0,
      "failures": 0, "v_final_calls": 0}
_T0 = None
_cache = {}   # (kind, a, b, horizon, seed, stream) -> value


def _total_fw():
    return _C["train_fw"] + _C["design_fw"] + _C["confirm_fw"]


def _total_bw():
    return _C["train_bw"] + _C["hvp_micro"] + _C["design_grad_evals"] * 128


def _load_stream_config():
    """跨阶段恢复流配置（preflight 写入 config.json）。"""
    cfgp = OUT / "config.json"
    if not cfgp.exists():
        return
    cfg = json.loads(cfgp.read_text())
    st = cfg.get("streams")
    if not st:
        return
    for k, v in st["fit"].items():
        FIT[int(k)] = int(v)
    for k, v in st["audit"].items():
        AUDIT[int(k)] = [int(x) for x in v]


def _replay_ledger():
    _load_stream_config()
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


def _log(stage, note="", **kw):
    rec = {"ts": time.time(), "stage": stage, "note": note,
           "wall_s": round(time.perf_counter() - _T0, 3) if _T0 else 0.0,
           "total_fw": _total_fw(), "total_bw": _total_bw(), **_C, **kw}
    RUNS.mkdir(parents=True, exist_ok=True)
    with (OUT / "resource_log.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _check_caps(up_traj=0, up_fw=0, up_bw=0, up_dirs=0, up_design=0,
                up_grad=0, up_confirm=0):
    if _C["traj_updates"] + up_traj > CAP_TRAJ:
        raise SystemExit(f"BUDGET_BLOCKED traj {CAP_TRAJ}")
    if _total_fw() + up_fw > CAP_FW:
        raise SystemExit(f"BUDGET_BLOCKED fw {CAP_FW}")
    if _total_bw() + up_bw > CAP_BW:
        raise SystemExit(f"BUDGET_BLOCKED bw {CAP_BW}")
    if _C["hvp_dirs"] + up_dirs > CAP_HVP_DIRS:
        raise SystemExit(f"BUDGET_BLOCKED hvp {CAP_HVP_DIRS}")
    if _C["design_evals"] + up_design > CAP_DESIGN_EVALS:
        raise SystemExit(f"BUDGET_BLOCKED design evals {CAP_DESIGN_EVALS}")
    if _C["design_grad_evals"] + up_grad > CAP_TARGET_GRAD:
        raise SystemExit(f"BUDGET_BLOCKED target grad {CAP_TARGET_GRAD}")
    if _C["confirm_evals"] + up_confirm > CAP_CONFIRM_EVALS:
        raise SystemExit(f"BUDGET_BLOCKED confirm {CAP_CONFIRM_EVALS}")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 运行时
# ---------------------------------------------------------------------------

def build_rt(seed, device):
    from fr import a0k_closed_loop_runner_v01 as a0k
    rt = a0k.build_live_runtime(seed, device)
    assert len(rt["optimizer"].param_groups) == 1
    from fr.phase_a_tail_schedule_v01 import make_phase_a_get_lr
    rt["eta"] = make_phase_a_get_lr(rt["get_lr"])
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rt["_f0_design"] = None
    rt["_f0_confirm"] = None
    return rt


def load_bank(rt, seed, start=START):
    from fr.full_state import load_full_training_state
    p = V07 / f"seed_{seed}/states/state_step_{start:03d}.pt"
    load_full_training_state(p, rt["model"], rt["optimizer"], rt["scaler"],
                             restore_rng=False)
    rt["model"].train()
    rt["optimizer"].zero_grad(set_to_none=True)
    return p


def make_stream(rt, stream_seed, n_micro):
    import b0_run_v01 as b0
    return b0.make_stream(rt, int(stream_seed), n_micro)


def stack_batch_rt(rt, idx):
    from fr.minimind_v03_runtime import stack_batch
    return stack_batch(rt["dataset"], idx, rt["device"])


def design_tensors(rt):
    if rt["_f0_design"] is None:
        import a0g_decision_utility_v01 as a0g
        from transformers import AutoTokenizer
        from dataset.lm_dataset import PretrainDataset
        mon = A1 / "data" / "v_monitor_rows4096_8192.jsonl"
        ds = PretrainDataset(str(mon),
                             AutoTokenizer.from_pretrained(
                                 str(PROJECT_ROOT / "minimind" / "model")),
                             max_length=int(rt["cfg"]["max_seq_len"]))
        chunks = [list(range(i, i + 32)) for i in range(0, 4096, 32)]
        rt["_f0_design"] = a0g._stack_all(ds, chunks, rt["device"])
    return rt["_f0_design"]


def confirm_tensors(rt):
    if rt["_f0_confirm"] is None:
        import a0g_decision_utility_v01 as a0g
        from transformers import AutoTokenizer
        from dataset.lm_dataset import PretrainDataset
        f = CONF_DATA / "v_confirm_rows12288_16384.jsonl"
        ds = PretrainDataset(str(f),
                             AutoTokenizer.from_pretrained(
                                 str(PROJECT_ROOT / "minimind" / "model")),
                             max_length=int(rt["cfg"]["max_seq_len"]))
        chunks = [list(range(i, i + 32)) for i in range(0, 4096, 32)]
        rt["_f0_confirm"] = a0g._stack_all(ds, chunks, rt["device"])
    return rt["_f0_confirm"]


def eval_full_library(rt, stacked, want_grad=False):
    """FP32/no-autocast/MATH sum/count 全库评价；可选全库 sum 梯度。

    返回 (mean, count, q|None)。梯度按 chunk sum 累积 backward（非均值平均）。
    """
    import torch
    import torch.nn.functional as F
    import a0g_decision_utility_v01 as a0g
    from fr.rng import capture_rng_state, restore_rng_state
    torch_ = rt["torch"]
    model = rt["model"]
    rng = capture_rng_state()
    was = model.training
    total = 0.0
    cnt = 0
    q = None
    model.eval()
    try:
        ctx_on = want_grad
        if want_grad:
            model.zero_grad(set_to_none=True)
        with a0g._math_sdpa_context(torch_):
            for x, y in stacked:
                if ctx_on:
                    res = model(x)
                else:
                    with torch_.no_grad():
                        res = model(x)
                logits = res.logits
                tgt = y[:, 1:]
                ce = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tgt.reshape(-1), ignore_index=-100, reduction="none"
                ).view(tgt.shape[0], tgt.shape[1])
                s = ce.sum()
                total += float(s.detach())
                cnt += int((tgt != -100).sum())
                if want_grad:
                    s.backward()
        if want_grad:
            # [IC-D3 修复] V = sum/count -> q = grad(total)/count。
            # 逐 chunk sum backward 后统一除以全库有效 token 数。
            q = {n: p.grad.detach().clone() / max(cnt, 1)
                 for n, p in rt["named"]}
            model.zero_grad(set_to_none=True)
    finally:
        if was:
            model.train()
        restore_rng_state(rng)
    return total / max(cnt, 1), cnt, q


def _apply_theta(rt, theta):
    torch = rt["torch"]
    with torch.no_grad():
        for n, p in rt["named"]:
            p.copy_(theta[n])


def _theta_state(rt):
    return {n: p.detach().clone() for n, p in rt["named"]}


# ---------------------------------------------------------------------------
# REF32 rollout（FP32/MATH/TF32-highest；可选全状态灵敏度）
# ---------------------------------------------------------------------------

def ref32_rollout(rt, seed, stream_seed, phi, T, sens=False,
                  want_grad=False, hvp_skips_zero=True, tag=""):
    """从 bank start46 起跑 T 步 REF32；返回终点状态/目标值/规则梯度。

    CRN：同 stream 固定 batch 序列；FP32 前向不消耗 RNG；评价 RNG 包裹。
    """
    import torch
    import a0g_decision_utility_v01 as a0g
    from fr import f0_terminal_rule_v01 as f0
    from fr.stream_rng import seed_future_stream

    # [IC-D4 修复] 规则恒按 T_MAIN=256 时钟构建（Z 对 256 项归一），
    # 再截取前 T 步；不再用 build_w(T) 改变 Z 的归一范围。
    w = f0.build_w(T_MAIN)
    psi = f0.build_psi(T_MAIN)
    eta, d_a, d_b = f0.rule_eta(phi[0], phi[1], w, psi)
    eta, d_a, d_b = eta[:T], d_a[:T], d_b[:T]
    load_bank(rt, seed)
    theta = _theta_state(rt)
    m = {}
    v = {}
    for n, p in rt["named"]:
        st = rt["optimizer"].state[p]
        m[n] = st["exp_avg"].detach().clone()
        v[n] = st["exp_avg_sq"].detach().clone()
    seed_future_stream(int(stream_seed))
    stream = make_stream(rt, stream_seed, 2 * T)
    dirs = [("a", 0.0), ("b", 0.0)]
    sens_state = f0.init_sensitivity(theta, dirs) if sens else None
    clip_active = 0
    params = [p for _, p in rt["named"]]
    names = [n for n, _ in rt["named"]]
    _check_caps(up_traj=T, up_fw=2 * T, up_bw=2 * T,
                up_dirs=(2 * T if sens else 0),
                up_design=1, up_grad=(1 if want_grad else 0))
    for t in range(T):
        losses = []
        with a0g._math_sdpa_context(torch):
            for a_ in range(2):
                x, y = stack_batch_rt(rt, stream["batches"][2 * t + a_])
                res = rt["model"](x, labels=y)
                losses.append((res.loss + res.aux_loss) / 2)
            loss_tot = losses[0] + losses[1]
            # 恒用 create_graph=True：SENS 与 value 路径同一算术（位级一致；
            # create_graph 会改变最后一位舍入，不允许两种梯度实现混用）
            grads = torch.autograd.grad(loss_tot, params,
                                        create_graph=True,
                                        allow_unused=True)
            g = {n: (gr.detach() if gr is not None
                     else torch.zeros_like(theta[n]))
                 for n, gr in zip(names, grads)}
            if sens:
                hg = {}
                for d, _ in dirs:
                    st_ = sens_state["S_theta"]
                    zero = all(float(st_[n][d].abs().sum()) == 0.0
                               for n in names)
                    if zero:
                        hg[d] = {n: torch.zeros_like(theta[n]) for n in names}
                        continue  # 解析跳过：S_theta 精确为 0
                    dot = None
                    for n, gr in zip(names, grads):
                        if gr is None:
                            continue
                        term = torch.sum(gr * st_[n][d].detach(),
                                         dtype=torch.float64)
                        dot = term if dot is None else dot + term
                    vj = torch.autograd.grad(dot, params,
                                             retain_graph=True,
                                             allow_unused=True)
                    hg[d] = {n: (z.detach().to(torch.float64)
                                 if z is not None else
                                 torch.zeros(theta[n].shape,
                                             dtype=torch.float64,
                                             device=theta[n].device))
                             for n, z in zip(names, vj)}
                sens_state["Hg"] = hg
                _C["hvp_dirs"] += sum(
                    1 for d, _ in dirs
                    if not all(float(sens_state["S_theta"][n][d].abs().sum())
                               == 0.0 for n in names))
                _C["hvp_micro"] += 2 * sum(
                    1 for d, _ in dirs
                    if not all(float(sens_state["S_theta"][n][d].abs().sum())
                               == 0.0 for n in names))
        _C["train_fw"] += 2
        _C["train_bw"] += 2
        sens_in = None
        if sens:
            sens_in = {"dirs": [("a", float(d_a[t])), ("b", float(d_b[t]))],
                       "S_theta": sens_state["S_theta"],
                       "S_m": sens_state["S_m"], "S_v": sens_state["S_v"],
                       "Hg": sens_state["Hg"]}
        theta, m, v, sens_out, info = f0.f0_adamw_sens_step(
            theta, m, v, g, B1_, B2_, EPS_, WD_, float(eta[t]), START + t + 1,
            tau=TAU_, sens=sens_in)
        if sens:
            sens_state = sens_out
        if info["clip_active"]:
            clip_active += 1
        _apply_theta(rt, theta)
        _C["traj_updates"] += 1
        if not all(math.isfinite(float(p.norm())) for _, p in rt["named"][:3]):
            _C["failures"] += 1
            raise RuntimeError(f"non-finite params at t={t} tag={tag}")
    mean_v, cnt, q = eval_design(rt, want_grad=want_grad)
    _C["design_fw"] += 128
    _C["design_evals"] += 1
    if want_grad:
        _C["design_grad_evals"] += 1
    out = {"V": mean_v, "count": cnt, "theta": theta, "m": m, "v": v,
           "clip_active_steps": clip_active,
           "grad": None, "q": q}
    if sens:
        out["sens"] = sens_state
    if sens and want_grad:
        ga = sum(float(torch.sum(q[n].to(torch.float64)
                                 * sens_state["S_theta"][n]["a"]).item())
                 for n in names)
        gb = sum(float(torch.sum(q[n].to(torch.float64)
                                 * sens_state["S_theta"][n]["b"]).item())
                 for n in names)
        out["grad"] = (ga, gb)
    return out


def eval_design(rt, want_grad=False):
    return eval_full_library(rt, design_tensors(rt), want_grad=want_grad)


def J_of(phi, T, seeds, sens=False, want_grad=False):
    """两条 FIT 流平均目标；缓存键 (a,b,T,seed,stream)，值/梯度复用。"""
    vals = []
    grads = []
    for seed in seeds:
        key = (round(phi[0], 12), round(phi[1], 12), T, seed, FIT[seed])
        rec = _cache.get(key)
        need_sens = sens or want_grad
        if rec is None or (need_sens and rec.get("grad") is None):
            rt = _rts[seed]
            out = ref32_rollout(rt, seed, FIT[seed], phi, T,
                                sens=need_sens, want_grad=want_grad,
                                tag=f"J{T}")
            rec = {"V": out["V"], "grad": out["grad"],
                   "clip": out["clip_active_steps"]}
            _cache[key] = rec
        vals.append(rec["V"])
        if want_grad:
            grads.append(rec["grad"])
    j = float(np.mean(vals))
    g = None
    if want_grad and grads and all(x is not None for x in grads):
        g = (float(np.mean([x[0] for x in grads])),
             float(np.mean([x[1] for x in grads])))
    return j, g


_rts = {}


# ---------------------------------------------------------------------------
# NATIVE rollout（BF16 真实执行；lr 查表自规则或原 k2）
# ---------------------------------------------------------------------------

def native_rollout(rt, seed, stream_seed, phi, T, mode="rule",
                   read_at=(64, 128, 256), ckpt=None, tag=""):
    """BF16 原生训练：mode='rule' 用 eta[t]；mode='k2' 用原 2*eta_base(gm)。

    返回 {"V_by_h": {h: mean}, "ckpt": path}。评价 RNG 包裹。
    """
    import torch
    from fr import f0_terminal_rule_v01 as f0
    from fr.native_schedule import set_native_micro_lr
    from fr.stream_rng import seed_future_stream
    from fr.full_state import save_full_training_state

    if mode == "rule":
        # [IC-D4 修复] 同 ref32：恒按 T256 时钟构建再截取。
        w = f0.build_w(T_MAIN)
        psi = f0.build_psi(T_MAIN)
        eta, _, _ = f0.rule_eta(phi[0], phi[1], w, psi)
        eta = eta[:T]
    load_bank(rt, seed)
    seed_future_stream(int(stream_seed))
    stream = make_stream(rt, stream_seed, 2 * T)
    model = rt["model"]
    optimizer = rt["optimizer"]
    scaler = rt["scaler"]
    cfg = rt["cfg"]
    accum = int(cfg["accumulation_steps"])
    gm0 = 2 * START
    reads = {}
    _check_caps(up_traj=T, up_fw=2 * T + 128 * len(
        [h for h in read_at if h <= T]), up_bw=2 * T,
        up_design=len([h for h in read_at if h <= T]))
    for t in range(T):
        for a_ in range(2):
            micro_idx = 2 * t + a_ + 1
            if mode == "rule":
                optimizer.param_groups[0]["lr"] = float(eta[t])
            else:
                lr_base = set_native_micro_lr(optimizer, rt["eta"],
                                              gm0 + micro_idx, 256,
                                              float(cfg["learning_rate"]))
                optimizer.param_groups[0]["lr"] = 2.0 * lr_base
            x, y = stack_batch_rt(rt, stream["batches"][micro_idx - 1])
            with rt["autocast_ctx"]:
                res = model(x, labels=y)
                loss = (res.loss + res.aux_loss) / accum
            scaler.scale(loss).backward()
            _C["train_fw"] += 1
            _C["train_bw"] += 1
            if not math.isfinite(float(res.loss)):
                _C["failures"] += 1
                raise RuntimeError(f"non-finite loss t={t} {tag}")
        scaler.unscale_(optimizer)
        preclip = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg["grad_clip"])).detach().cpu())
        if not math.isfinite(preclip):
            _C["failures"] += 1
            raise RuntimeError(f"non-finite preclip t={t} {tag}")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        _C["traj_updates"] += 1
        h = t + 1
        if h in read_at:
            mean_v, cnt, _ = eval_design(rt)
            _C["design_fw"] += 128
            _C["design_evals"] += 1
            reads[h] = mean_v
    if ckpt is not None:
        save_full_training_state(
            ckpt, model, optimizer, scaler,
            metadata={"protocol": "FR_GTTR_F0_AUDIT_CKPT", "seed": seed,
                      "stream": int(stream_seed), "mode": mode,
                      "phi": list(phi) if mode == "rule" else None,
                      "T": T, "h": T})
    return {"V_by_h": reads, "ckpt": str(ckpt) if ckpt else None}


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def run_preflight():
    global _T0
    _T0 = time.perf_counter()
    from fr import f0_terminal_rule_v01 as f0
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    CONF_DATA.mkdir(parents=True, exist_ok=True)
    rep = {}

    # CPU 测试
    r = subprocess.run([sys.executable, "-m", "unittest",
                        "tests.test_f0_terminal_rule_v01"],
                       cwd=PROJECT_ROOT / "research", capture_output=True,
                       text=True,
                       env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src",
                            "HOME": "/home/b205"})
    out_txt = r.stdout + r.stderr
    passed = r.returncode == 0 and out_txt.strip().splitlines()[-1] == "OK"
    (OUT / "cpu_tests.json").write_text(json.dumps(
        {"all_pass": bool(passed),
         "tail": out_txt.strip().splitlines()[-1]}, indent=1))
    assert passed, f"CPU tests failed:\n{out_txt[-1500:]}"

    # bank
    import torch
    for seed in SEEDS:
        p = V07 / f"seed_{seed}/states/state_step_{START:03d}.pt"
        st = torch.load(p, map_location="cpu", weights_only=False)
        md = st["metadata"]
        assert md["global_optimizer_step"] == START
        assert md["global_micro_step"] == 2 * START
    rep["bank_ok"] = True

    # 配方摘要核对（冲突即报）
    a1cfg = json.loads((A1 / "config.json").read_text())
    tr = a1cfg["train"]
    expect = {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01,
              "clip_tau": 1.0, "batch_size": 4, "accumulation_steps": 2}
    conflicts = {k: {"a1": tr.get(k), "expect": v} for k, v in
                 expect.items() if tr.get(k) != v}
    assert not conflicts, f"recipe conflict: {conflicts}"
    rep["recipe_ok"] = True

    # schedule 数组 + 首/尾 eta
    w = f0.build_w(T_MAIN)
    psi = f0.build_psi(T_MAIN)
    eta00, _, _ = f0.rule_eta(0.0, 0.0, w, psi)
    assert eta00[0] == w[0] and abs(w[0] - 7.323585913022454e-4) < 1e-16
    assert abs(w[-1] - 1e-4) < 1e-15
    np.savez(OUT / "schedule_arrays.npz", w=w, psi=psi)
    rep["schedule"] = {"w_first": float(w[0]), "w_last": float(w[-1]),
                       "w_sum": float(w.sum()),
                       "sha256": _sha256(OUT / "schedule_arrays.npz"),
                       "schedule_id": "gttr_terminal_normalized_tilt_T256_v1"}

    # 流冲突（整块 +100 顺延）
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
    base = STREAM_BASE
    while any(base + i in used for i in range(8)):
        base += 100
    FIT[2030] = base + 0
    FIT[2031] = base + 1
    AUDIT[2030] = [base + 2, base + 3]
    AUDIT[2031] = [base + 4, base + 5]
    AUDIT[2032] = [base + 6, base + 7]
    rep["streams"] = {"base": base, "shifted": base != STREAM_BASE,
                      "fit": {str(k): v for k, v in FIT.items()},
                      "audit": {str(k): v for k, v in AUDIT.items()}}

    # V_confirm_F0：D0[12288:16384) 提取 + 身份核查
    cfile = CONF_DATA / "v_confirm_rows12288_16384.jsonl"
    if not cfile.exists():
        with D0.open("rb") as fin, cfile.open("wb") as fout:
            for i, line in enumerate(fin):
                if 12288 <= i < 16384:
                    fout.write(line)
                if i >= 16384:
                    break
    n_lines = sum(1 for _ in cfile.open("rb"))
    assert n_lines == 4096
    hits = []
    block_pat = re.compile(
        r"(12288\s*[,:\-]\s*16384|\[12288[,\]]|rows?[_ ]*(id)?[_ ]*12288|"
        r"\"start[_ ]row\":\s*12288)", re.IGNORECASE)
    for p in PROJECT_ROOT.glob("outputs/**/*.json"):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        for mch in block_pat.finditer(txt):
            hits.append(f"{p.name}:{txt[max(0, mch.start()-30):mch.end()+30]}")
    rep["v_confirm"] = {
        "rows": "[12288,16384)", "n": n_lines, "sha256": _sha256(cfile),
        "usage_scan_hits": hits[:10],
        "identity": "首个 V_final 之后未使用 4096 行块；无使用记录命中"
        if not hits else "CONFIRM_SET_IDENTITY_UNRESOLVED",
        "doc_independence": "DOC_INDEPENDENCE_UNVERIFIED（无 doc_id）",
        "sealed_before_use": True}
    assert not hits, f"confirm block usage hits: {hits[:5]}"

    config = {
        "round": "fr_gttr_f0_terminal_rule_solver_v01",
        "protocol": PROTOCOL.name,
        "protocol_sha256": _sha256(PROTOCOL),
        "T": T_MAIN, "n0": START,
        "rule": {"psi": "2t/255-1", "Z": "sum(w e^{b psi})/sum(w)",
                 "eta": "w exp(a+b psi)/Z", "box": "[-log2,log2]^2",
                 "sum_eta": "exp(a)*sum(w)", "tail_w": 1e-4,
                 "schedule_id": "gttr_terminal_normalized_tilt_T256_v1"},
        "paths": {"REF32": "FP32/autocast off/MATH/TF32 highest",
                  "NATIVE": "BF16 autocast 原生"},
        "streams": rep["streams"],
        "design_target": {"target_id": "V_design_monitor_FP32_MATH_sumcount_v1",
                          "rows": "[4096,8192)", "sha256": _sha256(
                              A1 / "data/v_monitor_rows4096_8192.jsonl")},
        "confirm_target": rep["v_confirm"],
        "solvers": {"FULL256": "a,b 终点256", "SCALE256": "a 终点256",
                    "SHORT64": "a,b 目标64（T256 时钟）",
                    "iters": 3, "h_frac": [1.0, 0.5, 0.25],
                    "armijo_c": 1e-4,
                    "common_start": "b=0 三尺度点中 FIT 最优"},
        "budget": {"caps": {"traj": CAP_TRAJ, "fw": CAP_FW, "bw": CAP_BW,
                            "hvp_dirs": CAP_HVP_DIRS,
                            "design_evals": CAP_DESIGN_EVALS,
                            "confirm_evals": CAP_CONFIRM_EVALS,
                            "target_grad": CAP_TARGET_GRAD, "v_final": 0}},
        "code": {"runner_sha256": _sha256(Path(__file__).resolve())}}
    (OUT / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    _log("preflight", "ok")
    print(json.dumps(rep, indent=1))
    print("PREFLIGHT_OK")


# ---------------------------------------------------------------------------
# smoke（64 updates；4×16；V_design <=8，grad<=1）
# ---------------------------------------------------------------------------

def run_smoke(gpu):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    _replay_ledger()
    from fr import f0_terminal_rule_v01 as f0
    from fr.rng import capture_rng_state
    seed = 2030
    device = f"cuda:{gpu}"
    rt = build_rt(seed, device)
    _rts[seed] = rt
    checks = {}
    t0 = time.perf_counter()
    traj0 = _C["traj_updates"]
    evals0 = _C["design_evals"]
    grads0 = _C["design_grad_evals"]

    r_plain = ref32_rollout(rt, seed, FIT[seed], (0.0, 0.0), 16,
                            sens=False, tag="smoke_ref_plain")
    theta_plain = r_plain["theta"]
    m_plain = r_plain["m"]
    rng_plain = capture_rng_state()
    r_sens = ref32_rollout(rt, seed, FIT[seed], (0.0, 0.0), 16, sens=True,
                           want_grad=True, tag="smoke_ref_sens")
    import torch
    same_theta = all(torch.equal(theta_plain[n], r_sens["theta"][n])
                     for n in theta_plain)
    same_mv = all(torch.equal(m_plain[n], r_sens["m"][n]) for n in m_plain)
    rng_sens = capture_rng_state()
    checks["ref_sens_bitwise_theta"] = bool(same_theta)
    checks["ref_sens_bitwise_mv"] = bool(same_mv)
    import torch as _torch
    checks["ref_rng_equal"] = bool(
        _torch.equal(rng_plain["torch_cpu"], rng_sens["torch_cpu"]))
    checks["smoke_grad"] = r_sens["grad"]

    n1 = native_rollout(rt, seed, FIT[seed], None, 16, mode="k2",
                        read_at=(), tag="smoke_nat_k2")
    sig1 = _native_sig(rt)
    rng1 = capture_rng_state()
    n2 = native_rollout(rt, seed, FIT[seed], (0.0, 0.0), 16, mode="rule",
                        read_at=(16,), tag="smoke_nat_rule")
    sig2 = _native_sig(rt)
    rng2 = capture_rng_state()
    checks["native_wrapper_bitwise"] = all(
        torch.equal(sig1["theta"][n], sig2["theta"][n]) for n in sig1["theta"]) \
        and sig1["steps"] == sig2["steps"]
    checks["native_rng_equal"] = bool(
        torch.equal(rng1["torch_cpu"], rng2["torch_cpu"]))
    checks["native_read_h16"] = n2["V_by_h"].get(16)
    checks["traj_updates"] = _C["traj_updates"] - traj0
    checks["design_evals"] = _C["design_evals"] - evals0
    checks["grad_evals"] = _C["design_grad_evals"] - grads0
    checks["verdict"] = "PASS" if (
        same_theta and same_mv and checks["ref_rng_equal"]
        and checks["native_wrapper_bitwise"]
        and checks["native_rng_equal"]
        and checks["traj_updates"] <= 64
        and checks["design_evals"] <= 8 and checks["grad_evals"] <= 1
        and n2["V_by_h"].get(16) is not None) else "FAIL"
    (OUT / "smoke_checks.json").write_text(json.dumps(checks, indent=1))
    _log("smoke_done", checks["verdict"])
    print(json.dumps(checks, indent=1))
    return checks["verdict"] == "PASS"


def _native_sig(rt):
    sd = {n: p.detach().cpu().clone() for n, p in rt["named"]}
    ost = rt["optimizer"].state_dict()
    steps = sorted(int(v["step"]) for v in ost["state"].values())
    return {"theta": sd, "steps": steps}


# ---------------------------------------------------------------------------
# calibrate（SENS + 8 FD；terminal_gradient_check.csv）
# ---------------------------------------------------------------------------

def run_calibrate(gpu):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    _replay_ledger()
    from fr import f0_terminal_rule_v01 as f0
    seed = 2030
    if seed not in _rts:
        _rts[seed] = build_rt(seed, f"cuda:{gpu}")
    rt = _rts[seed]
    base = ref32_rollout(rt, seed, FIT[seed], (0.0, 0.0), T_MAIN, sens=True,
                         want_grad=True, tag="calib_sens")
    g_sens = base["grad"]
    # 重复评价同终点一次（eps_V）
    _apply_theta(rt, base["theta"])
    v_repeat, _, _ = eval_design(rt)
    _C["design_fw"] += 128
    _C["design_evals"] += 1
    eps_v = max(abs(v_repeat - base["V"]), 1e-7)
    rows = []
    fd = {}
    for j, coord in enumerate(("a", "b")):
        for eps in (0.01, 0.005):
            v_p = ref32_rollout(rt, seed, FIT[seed],
                                (eps if coord == "a" else 0.0,
                                 eps if coord == "b" else 0.0), T_MAIN,
                                tag=f"calib_fd_p_{coord}_{eps}")["V"]
            v_m = ref32_rollout(rt, seed, FIT[seed],
                                (-eps if coord == "a" else 0.0,
                                 -eps if coord == "b" else 0.0), T_MAIN,
                                tag=f"calib_fd_m_{coord}_{eps}")["V"]
            fd[(coord, eps)] = (v_p - v_m) / (2 * eps)
    abs_floor = max(5e-4, 4 * eps_v / 0.005)
    verdict = {}
    for coord in ("a", "b"):
        g_j = g_sens[0] if coord == "a" else g_sens[1]
        fd01 = fd[(coord, 0.01)]
        fd005 = fd[(coord, 0.005)]
        tol = abs_floor + 0.20 * max(abs(g_j), abs(fd01), abs(fd005))
        consistent = (abs(fd01 - fd005) <= tol
                      and abs(g_j - fd005) <= tol)
        all_small = (abs(g_j) < abs_floor and abs(fd01) < abs_floor
                     and abs(fd005) < abs_floor)
        if all_small:
            v_ = "LOW_SIGNAL"
        elif consistent:
            v_ = "CALIBRATED"
        else:
            v_ = "FULL_HORIZON_GRADIENT_UNRESOLVED"
        verdict[coord] = v_
        rows.append((coord, g_j, fd01, fd005, abs(fd01 - fd005),
                     abs(g_j - fd005), tol, abs_floor, eps_v, v_))
    with (OUT / "terminal_gradient_check.csv").open("w") as f:
        f.write("coord,g_sens,FD_eps0.01,FD_eps0.005,|FD01-FD005|,"
                "|SENS-FD005|,tol,abs_floor,eps_V,verdict\n")
        for r_ in rows:
            f.write(",".join(repr(x) for x in r_) + "\n")
    ok = all(v_ in ("CALIBRATED", "LOW_SIGNAL") for v_ in verdict.values())
    _log("calibrate_done", json.dumps(verdict),
         clip_active=base["clip_active_steps"])
    print(json.dumps({"verdict": verdict, "rows": rows,
                      "clip_active_steps": base["clip_active_steps"],
                      "eps_V": eps_v}, indent=1))
    if not ok:
        raise SystemExit("FULL_HORIZON_GRADIENT_UNRESOLVED：停止导数求解阶段"
                         "（证据已存 terminal_gradient_check.csv）")
    print("CALIBRATE_OK")


# ---------------------------------------------------------------------------
# grid + solve
# ---------------------------------------------------------------------------

def run_grid(gpu):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    _replay_ledger()
    from fr import f0_terminal_rule_v01 as f0
    for seed in FIT:
        if seed not in _rts:
            _rts[seed] = build_rt(seed, f"cuda:{gpu}")
    grid = [(-LOG2, -LOG2), (-LOG2, 0.0), (-LOG2, LOG2), (0.0, -LOG2),
            (0.0, 0.0), (0.0, LOG2), (LOG2, -LOG2), (LOG2, 0.0),
            (LOG2, LOG2)]
    rows = []
    for phi in grid:
        j, _ = J_of(phi, T_MAIN, list(FIT))
        rows.append((phi[0], phi[1], j))
        _log("grid", f"phi=({phi[0]:.4f},{phi[1]:.4f}) J={j:.6f}")
    best_phi, best_j = f0.tie_break_argmin([((r_[0], r_[1]), r_[2])
                                            for r_ in rows])
    scale_pts = [((r_[0], r_[1]), r_[2]) for r_ in rows if r_[1] == 0.0]
    init_phi, init_j = f0.tie_break_argmin(scale_pts)
    with (OUT / "grid_terminal.csv").open("w") as f:
        f.write("a,b,J256_mean_FIT\n")
        for r_ in rows:
            f.write(",".join(repr(x) for x in r_) + "\n")
    json_ = {"grid_best": list(best_phi), "grid_best_J": best_j,
             "phi_init": list(init_phi), "phi_init_J": init_j}
    (RUNS / "grid_summary.json").write_text(json.dumps(json_, indent=1))
    _log("grid_done", json.dumps(json_))
    print(json.dumps(json_, indent=1))
    print("GRID_OK")


def run_solve(gpu, name="FULL256"):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    _replay_ledger()
    from fr import f0_terminal_rule_v01 as f0
    for seed in FIT:
        if seed not in _rts:
            _rts[seed] = build_rt(seed, f"cuda:{gpu}")
    gs = json.loads((RUNS / "grid_summary.json").read_text())
    free = (True, True) if name in ("FULL256", "SHORT64") else (True, False)
    horizon = H_SHORT if name == "SHORT64" else T_MAIN
    phi = tuple(gs["phi_init"])
    j_cur, _ = J_of(phi, horizon, list(FIT))
    hist = []
    status = "ITER_LIMIT"
    low_signal = False
    abs_floor = 5e-4
    p168 = (OUT / "terminal_gradient_check.csv")
    if p168.exists():
        lines = p168.read_text().splitlines()[1:]
        abs_floor = max(abs_floor, max(float(l.split(",")[7]) for l in lines
                                       if l.strip()))
    for it in range(1, 4):
        j_g, g = J_of(phi, horizon, list(FIT), sens=True, want_grad=True)
        g_free = (g[0] if free[0] else 0.0, g[1] if free[1] else 0.0)
        if abs(g_free[0]) == 0.0 and abs(g_free[1]) == 0.0:
            status = "ZERO_GRADIENT"
            hist.append((it, phi, j_cur, g_free, None, None, None,
                         "g=0 stop"))
            break
        if max(abs(g_free[0]), abs(g_free[1])) < abs_floor:
            status = "LOW_SIGNAL"
            low_signal = True
            hist.append((it, phi, j_cur, g_free, None, None, None,
                         "below calibration floor"))
            break
        accepted = None
        for (h, trial, delta) in f0.solver_candidates(phi, g_free, free):
            j_t, _ = J_of(trial, horizon, list(FIT))
            hist.append((it, phi, j_cur, g_free, trial, j_t, delta,
                         f"trial h={h}"))
            if f0.armijo_ok(j_t, j_cur, g_free, delta):
                accepted = (trial, j_t)
                break
        if accepted is None:
            status = "NO_ARMIJO_ACCEPT"
            break
        phi, j_cur = accepted[0], accepted[1]
        hist.append((it, phi, j_cur, g_free, accepted[0], accepted[1],
                     None, "accepted"))
    with (OUT / "solver_history.csv").open("a") as f:
        if f.tell() == 0:
            f.write("solver,iter,phi,phi_trial,g_a,g_b,J,J_trial,note\n")
        for i_, (it, phi_, j_, g_, trial, jt, delta, note) in enumerate(hist):
            f.write(",".join(str(x) for x in (
                name, it, f"({phi_[0]:.6f},{phi_[1]:.6f})",
                f"({trial[0]:.6f},{trial[1]:.6f})" if trial else "",
                g_[0], g_[1], j_, jt if jt is not None else "", note)) + "\n")
    sol = {"solver": name, "phi": list(phi), "J": j_cur, "status": status,
           "low_signal": low_signal, "free": list(free), "horizon": horizon}
    (RUNS / f"solution_{name}.json").write_text(json.dumps(sol, indent=1))
    _log("solve_done", json.dumps(sol))
    print(json.dumps(sol, indent=1))
    print("SOLVE_OK")


# ---------------------------------------------------------------------------
# audit（封存 policy → NATIVE 30 + REF32 6）
# ---------------------------------------------------------------------------

def run_audit(gpu):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    _replay_ledger()
    import torch
    from fr import f0_terminal_rule_v01 as f0
    from fr import r1_objective_identification_v01 as r1m
    sol = {}
    for name in ("FULL256", "SCALE256", "SHORT64"):
        sol[name] = json.loads((RUNS / f"solution_{name}.json").read_text())
    gs = json.loads((RUNS / "grid_summary.json").read_text())
    policies = {"BASE_K2": (0.0, 0.0),
                "SCALE256": tuple(sol["SCALE256"]["phi"]),
                "FULL256": tuple(sol["FULL256"]["phi"]),
                "SHORT64": tuple(sol["SHORT64"]["phi"]),
                "GRID2D256": tuple(gs["grid_best"])}
    # eta 数组与别名
    w = f0.build_w(T_MAIN)
    psi = f0.build_psi(T_MAIN)
    arrs = {}
    for k_, phi in policies.items():
        e, _, _ = f0.rule_eta(phi[0], phi[1], w, psi)
        arrs[k_] = e
    canon = {}
    alias = {}
    for k_ in policies:
        h_ = hashlib.sha256(
            np.asarray(arrs[k_]).tobytes()).hexdigest()
        if h_ in canon:
            alias[k_] = canon[h_]
        else:
            canon[h_] = k_
            alias[k_] = k_
    frozen = {"policies": {k_: list(v_) for k_, v_ in policies.items()},
              "alias": alias,
              "eta_summary": {k_: f0.schedule_summary(arrs[k_])
                              for k_ in policies},
              "solutions": sol, "grid": gs}
    (OUT / "frozen_policies.json").write_text(
        json.dumps(frozen, indent=1) + "\n")
    r1m.write_seal(OUT / "policy_seal.json",
                   [OUT / "frozen_policies.json",
                    OUT / "grid_terminal.csv",
                    OUT / "solver_history.csv",
                    OUT / "terminal_gradient_check.csv"],
                   extra={"stage": "policies_frozen"})
    _log("policies_frozen", json.dumps(alias))

    ckpt_dir = OUT / "audit_ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    man = []
    for seed in SEEDS:
        rt = _rts.get(seed) or build_rt(seed, f"cuda:{gpu}")
        _rts[seed] = rt
        load_bank(rt, seed)
        h0v, _, _ = eval_design(rt)
        _C["design_fw"] += 128
        _C["design_evals"] += 1
        rows.append((seed, "shared", "shared", "h0", h0v))
        for s_i, stream in enumerate(AUDIT[seed]):
            for arm in ("BASE_K2", "SCALE256", "FULL256", "SHORT64",
                        "GRID2D256"):
                if alias[arm] != arm:
                    rows.append((seed, stream, arm, "ALIAS",
                                 f"IDENTICAL_POLICY->{alias[arm]}"))
                    continue
                phi = policies[arm]
                ck = ckpt_dir / f"s{seed}_st{stream}_{arm}_h256.pt"
                rec = native_rollout(rt, seed, stream, phi, T_MAIN,
                                     mode="rule", read_at=(64, 128, 256),
                                     ckpt=ck, tag=arm)
                for h, v_ in sorted(rec["V_by_h"].items()):
                    rows.append((seed, stream, arm, f"h{h}", v_))
                man.append({"seed": seed, "stream": stream, "arm": arm,
                            "path": str(ck), "sha256": _sha256(Path(ck)),
                            "backend": "NATIVE"})
                _log("audit_native_done", f"{seed}/{stream}/{arm}")
        # REF32 对照（每 seed 第一条 audit 流：BASE_K2 与 FULL256）
        for arm in ("BASE_K2", "FULL256"):
            if alias[arm] != arm:
                continue
            stream = AUDIT[seed][0]
            phi = policies[arm]
            out = ref32_rollout(rt, seed, stream, phi, T_MAIN, sens=False,
                                want_grad=False, tag=f"audit_ref_{arm}")
            ck = ckpt_dir / f"s{seed}_st{stream}_{arm}_REF32_h256.pt"
            _apply_theta(rt, out["theta"])
            # REF32 的 m/v 不在 optimizer state（解析更新）-> 快照保存数值
            torch.save({"theta": out["theta"], "m": out["m"], "v": out["v"],
                        "V_design_h256": out["V"], "seed": seed,
                        "stream": stream, "arm": arm, "backend": "REF32"},
                       ck)
            rows.append((seed, stream, arm + "_REF32", "h256", out["V"]))
            man.append({"seed": seed, "stream": stream, "arm": arm,
                        "path": str(ck), "sha256": _sha256(Path(ck)),
                        "backend": "REF32"})
            _log("audit_ref_done", f"{seed}/{stream}/{arm}")
    with (OUT / "audit_terminal.csv").open("w") as f:
        f.write("seed,stream,arm,reading,value\n")
        for r_ in rows:
            f.write(",".join(str(x) for x in r_) + "\n")
    (OUT / "audit_ckpt_manifest.json").write_text(json.dumps(man, indent=1))
    r1m.write_seal(OUT / "audit_seal.json",
                   [OUT / "audit_terminal.csv",
                    OUT / "audit_ckpt_manifest.json"],
                   extra={"stage": "audit_sealed"})
    _log("audit_done", f"rows={len(rows)} ckpts={len(man)}")
    print(f"AUDIT_DONE rows={len(rows)} ckpts={len(man)}")


# ---------------------------------------------------------------------------
# confirm（全部 checkpoint 封存后一次性读 V_confirm_F0）
# ---------------------------------------------------------------------------

def run_confirm(gpu):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    _replay_ledger()
    import torch
    from fr import r1_objective_identification_v01 as r1m
    ok, why = r1m.verify_seal(OUT / "audit_seal.json",
                              [OUT / "audit_terminal.csv",
                               OUT / "audit_ckpt_manifest.json"])
    assert ok, f"audit seal broken: {why}"
    man = json.loads((OUT / "audit_ckpt_manifest.json").read_text())
    rows = []
    for i, m in enumerate(man):
        seed = m["seed"]
        if seed not in _rts:
            _rts[seed] = build_rt(seed, f"cuda:{gpu}")
        rt = _rts[seed]
        _check_caps(up_confirm=1, up_fw=128)
        if m["backend"] == "NATIVE":
            from fr.full_state import load_full_training_state
            load_full_training_state(m["path"], rt["model"],
                                     rt["optimizer"], rt["scaler"],
                                     restore_rng=False)
            rt["model"].eval()
        else:
            blob = torch.load(m["path"], map_location=rt["device"],
                              weights_only=False)
            _apply_theta(rt, blob["theta"])
        mean_v, cnt, _ = eval_full_library(rt, confirm_tensors(rt))
        _C["confirm_fw"] += 128
        _C["confirm_evals"] += 1
        rows.append((m["seed"], m["stream"], m["arm"], m["backend"],
                     mean_v, cnt))
        print(f"CONFIRM {i+1}/{len(man)} {m['seed']}/{m['stream']}/"
              f"{m['arm']}/{m['backend']} V={mean_v:.6f}", flush=True)
    with (OUT / "confirm_terminal.csv").open("w") as f:
        f.write("seed,stream,arm,backend,confirm_mean,confirm_count\n")
        for r_ in rows:
            f.write(",".join(str(x) for x in r_) + "\n")
    _log("confirm_done", f"n={len(rows)}")
    print(f"CONFIRM_DONE n={len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["preflight", "smoke", "calibrate",
                                      "grid", "solve", "audit", "confirm",
                                      "all"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--solver", default="FULL256")
    args = ap.parse_args()
    if args.phase == "preflight":
        run_preflight()
    elif args.phase == "smoke":
        assert run_smoke(args.gpu)
    elif args.phase == "calibrate":
        run_calibrate(args.gpu)
    elif args.phase == "grid":
        run_grid(args.gpu)
    elif args.phase == "solve":
        run_solve(args.gpu, args.solver)
    elif args.phase == "audit":
        run_audit(args.gpu)
    elif args.phase == "confirm":
        run_confirm(args.gpu)
    else:
        run_preflight()
        assert run_smoke(args.gpu)
        run_calibrate(args.gpu)
        run_grid(args.gpu)
        for s in ("FULL256", "SCALE256", "SHORT64"):
            run_solve(args.gpu, s)
        run_audit(args.gpu)
        run_confirm(args.gpu)


if __name__ == "__main__":
    main()
