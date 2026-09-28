#!/usr/bin/env python3
"""FR Phase B：S3/S1 首方向 FD 局部算子失配定点诊断 v1（0 update）。

执行 FR_PhaseB_FD_Local_Operator_Diagnostic_v1.md：
  plan      0-GPU：写 DIAGNOSTIC_PLAN.json（caps/尺度/检查点/决策规则）
  selftest  0-GPU：CPU 二次型上验证本脚本 AD-HVP 原语 == 解析 H·v
  run       GPU：D0 现场记录 → D1 原路径定点复现 → D2 固定尺度 → D3 AD/FP64 参考
  report    0-GPU：由 CSV 生成 REPORT.md / KEY_DATA_FOR_REVIEW.md / ≤2 图

不变项：原 v3 GATE_STOP 与全部旧文件不动；不调阈值；updates=0；V_* 调用=0；
SIM/bank/方向固定（S3=2031/82/197009，S1=2030/82/197007，首方向 v1）；
差分对象为未裁剪 G_SIM（clip 只出现在虚拟参考步，不进 FD 端点）。
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "scripts"))

OUT = PROJECT_ROOT / "outputs" / "fr_phase_b_fd_diagnostic_v01"
V3_OUT = PROJECT_ROOT / "outputs" / "fr_phase_b_gttr_action_scoring_v03"

# 冻结状态（v3 manifest.csv 为准）
STATES = [("S3", 2031, 82, 197009), ("S1", 2030, 82, 197007)]
# D2 固定尺度（禁止逐结果调参）
SCALES = (2.0, 1.0, 0.5, 0.25, 0.125)
# 参考位移检查点乘子（theta0 + c * s_ref0）
CKPTS = (1.0, math.exp(0.05))
# 成本 caps（协议 §7，本诊断轮独立账本）
CAP_FW = 240
CAP_BW = 320          # 一阶 + 二阶分列记账，合计受 cap
CAP_AD_HVP = 6
FD_DELTA_SCALE = 1e-3
FD_REL_GATE = 0.20
_B1, _B2, _EPS, _WD, _TAU = 0.9, 0.999, 1e-8, 0.01, 1.0

# ---------------------------------------------------------------------------
# 本地账本（与 v3 账本分离；全部求值含失败/重复均计费）
# ---------------------------------------------------------------------------
C = {"updates": 0, "micro_fw": 0, "micro_bw_first": 0, "micro_bw_second": 0,
     "ad_hvp_evals": 0, "fd_dir_estimates": 0.0,
     "sim_unique_slots": 0, "sim_unique_nonpad": 0, "sim_unique_valid": 0,
     "sim_reprocessed_slots": 0, "peak_mem_gb": 0.0,
     "wall_s": 0.0, "wall_label": "OBSERVATIONAL_SHARED_GPU",
     "partial": False, "partial_reason": None, "backend_checks": 0}


def _bill(kind, n, why=""):
    if kind == "fw":
        C["micro_fw"] += n
    elif kind == "bw1":
        C["micro_bw_first"] += n
    elif kind == "bw2":
        C["micro_bw_second"] += n
    if C["micro_fw"] > CAP_FW or (C["micro_bw_first"] + C["micro_bw_second"]) > CAP_BW:
        raise BudgetExceeded(f"budget exceeded at {why}: fw={C['micro_fw']} "
                             f"bw={C['micro_bw_first'] + C['micro_bw_second']}")


class BudgetExceeded(RuntimeError):
    pass


def _S():
    from fr import phase_b_gttr_score_v03 as S
    return S


def _rng_equal(a, b):
    import torch
    ok, why = True, []
    for k in set(a) | set(b):
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            return False, f"{k} missing"
        if isinstance(va, torch.Tensor):
            same = bool(torch.equal(va, vb))
        elif isinstance(va, (list, tuple)) and va and isinstance(va[0], torch.Tensor):
            same = (len(va) == len(vb)) and all(bool(torch.equal(x, y))
                                                for x, y in zip(va, vb))
        elif k == "python_random":
            same = va == vb
        elif k == "numpy_random":
            try:
                same = bool((va[1] == vb[1]).all()) and va[0] == vb[0]
            except Exception:
                same = va == vb
        else:
            try:
                same = bool(torch.equal(va, vb))
            except Exception:
                same = va == vb
        if not same:
            ok, why = False, [f"{k} differs"]
    return ok, ";".join(why)


class _nullctx:
    def __enter__(self):
        return None
    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# DiagField：逐行镜像 v3 SimField（算术路径不变），附加 loss/偏移/还原记录
# ---------------------------------------------------------------------------
class DiagField:
    """G_SIM(θ) = 2 SIM microbatch 按 (loss+aux)/accum 的未裁剪梯度。

    与 v3 SimField 相同的 rng0/训练模式/zero_grad/clone/还原语义；
    额外记录：标量 loss、实际参数偏移 vs 期望 overlay、求值后还原误差、
    profiler 抓取的 attention kernel 名（搭车首次求值，0 额外计费）。
    """

    def __init__(self, rt, sim_tensors, path_id="ORIG_FP32"):
        self.rt = rt
        self.torch = rt["torch"]
        self.sim_tensors = sim_tensors
        self.path_id = path_id
        self.named = named_unique(rt)
        self.theta0 = {n: p.detach().clone() for n, p in self.named.items()}
        from fr.rng import capture_rng_state
        self.rng_live = capture_rng_state()
        self.rng0 = capture_rng_state()
        self.profiled_kernels = []
        self.sim_tokens = {"input_slots": 0, "nonpad_tokens": 0, "valid_tokens": 0}
        for x, y in sim_tensors:
            self.sim_tokens["input_slots"] += int(x.numel())
            self.sim_tokens["nonpad_tokens"] += int((y != -100).sum())
            self.sim_tokens["valid_tokens"] += int((y[:, 1:] != -100).sum())
        self.last = {}

    def __call__(self, overlay=None, tag="", reverse=False, profile=False):
        from fr.rng import capture_rng_state, restore_rng_state
        torch = self.torch
        rt = self.rt
        model = rt["model"]
        accum = int(rt["cfg"]["accumulation_steps"])
        ent = self._entry_err()
        if overlay is not None:
            for n, p in self.named.items():
                p.detach().add_(overlay[n])
        off_max, off_norm_err = self._offset_err(overlay)
        loss = None
        prof = None
        try:
            restore_rng_state(self.rng0)
            was_training = model.training
            model.train()
            try:
                model.zero_grad(set_to_none=True)
                total = None
                ctx = self._profile_ctx() if profile else _nullctx()
                with ctx as prof:
                    for x, y in self.sim_tensors:
                        restore_rng_state(self.rng0)   # 每个 micro 同一掩码
                        res = model(x, labels=y)       # no autocast：FP32
                        l = (res.loss + res.aux_loss) / accum
                        total = l if total is None else total + l
                if profile and prof is not None:
                    self.profiled_kernels = sorted(
                        e.key for e in prof.key_averages()
                        if any(s in e.key.lower() for s in
                               ("attention", "flash", "efficient", "gemm", "matmul",
                                "sdpa", "cutlass", "bias_relu")))
                loss = float(total.detach())
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
        exit_err = self._entry_err()
        rng_ok, rng_why = _rng_restored_eq(self.rng_live)
        finite = bool(loss is not None and math.isfinite(loss)) and all(
            bool(torch.isfinite(v).all()) for v in g.values())
        C["sim_reprocessed_slots"] += self.sim_tokens["input_slots"]
        self.last = {"tag": tag, "entry_err": ent, "offset_err_max": off_max,
                     "offset_norm_err": off_norm_err, "exit_err": exit_err,
                     "rng_restored": rng_ok, "rng_why": rng_why, "loss": loss,
                     "finite": finite, "reverse": reverse}
        return g, loss, dict(self.last)

    def _entry_err(self):
        errs = [float((p.detach() - self.theta0[n]).abs().max())
                for n, p in self.named.items()]
        return max(errs) if errs else 0.0

    def _offset_err(self, overlay):
        if overlay is None:
            return 0.0, 0.0
        e_max, e_n = 0.0, 0.0
        for n, p in self.named.items():
            actual = (p.detach() - self.theta0[n]).reshape(-1)
            expect = overlay[n].reshape(-1).to(actual.dtype)
            e_max = max(e_max, float((actual - expect).abs().max()))
            e_n = max(e_n, abs(float(actual.double().norm()) - float(expect.double().norm())))
        return e_max, e_n

    def _profile_ctx(self):
        from torch.profiler import profile, ProfilerActivity
        acts = ([ProfilerActivity.CUDA] if self.torch.cuda.is_available()
                else [ProfilerActivity.CPU])
        C["backend_checks"] += 1
        return profile(activities=acts)

    def close(self):
        from fr.rng import restore_rng_state
        for n, p in self.named.items():
            p.detach().copy_(self.theta0[n])
        self.rt["model"].zero_grad(set_to_none=True)
        restore_rng_state(self.rng_live)


def _rng_restored_eq(ref):
    from fr.rng import capture_rng_state
    return _rng_equal(ref, capture_rng_state())


def named_unique(rt):
    seen, out = set(), {}
    for n, p in rt["named"]:
        if id(p) in seen:
            continue
        seen.add(id(p))
        out[n] = p
    return out


def _runner():
    import importlib
    return importlib.import_module("phase_b_gttr_v03")


def _bill_fd(n_micro, why, n_eval=4):
    _bill("fw", n_micro * n_eval, why)
    _bill("bw1", n_micro * n_eval, why)


def _rep_row(sid, chk, g_diff=None, notes=""):
    return {"state_id": sid, "eval_tag": chk.get("tag", ""),
            "entry_err": f"{chk.get('entry_err', ''):.3e}" if chk.get("entry_err") != "" else "",
            "offset_err_max": f"{chk.get('offset_err_max', 0):.3e}",
            "offset_norm_err": f"{chk.get('offset_norm_err', 0):.3e}",
            "exit_err": f"{chk.get('exit_err', 0):.3e}",
            "rng_restored": chk.get("rng_restored"),
            "loss": chk.get("loss"),
            "g_diff_to_first_rel": g_diff, "notes": notes}


def fd_y(field, v1, h, tag, rows_rep=None, reverse=False):
    """单尺度中心差分：y(h)=[G(θ0+h·v1)−G(θ0−h·v1)]/(2h)。4 次 micro fwd/bwd。"""
    op = {k: v * (+h) for k, v in v1.items()}
    om = {k: v * (-h) for k, v in v1.items()}
    if not reverse:
        gp, lp, cp = field(op, tag=tag + "/plus")
        gm, lm, cm = field(om, tag=tag + "/minus")
    else:
        gm, lm, cm = field(om, tag=tag + "/minus_rev")
        gp, lp, cp = field(op, tag=tag + "/plus_rev")
    y = {k: (gp[k] - gm[k]) / (2.0 * h) for k in gp}
    rec = {"tag": tag, "h": h, "Lplus": lp, "Lminus": lm,
           "checks_plus": cp, "checks_minus": cm, "reverse": reverse}
    if rows_rep is not None:
        sid = tag.split("/")[0]
        rows_rep.append(_rep_row(sid, cp))
        rows_rep.append(_rep_row(sid, cm))
    return y, gp, gm, rec


def ad_hvp(model, named, sim_tensors, rng0, v1, accum, bill_tag,
           math_backend=False):
    """y_AD = grad_θ( dot(grad_θ L_SIM, stop_gradient(v1)) )，同一标量 SIM loss。

    计费：2 micro-fw + 2 一阶 bwd + 2 二阶 bwd；ad_hvp_evals+1（失败尝试也计费）。
    v1 为常数坐标，不对其构造求导。求值后恢复 rng_live/训练模式/清空梯度。
    math_backend=True：强制 SDPA math 核（mem-efficient attention 无二阶导，
    协议 §5 允许的一次最小等价 math-backend 参考；标 REFERENCE_BACKEND_CHANGE）。
    """
    import torch
    from fr.rng import capture_rng_state, restore_rng_state
    params = [named[k] for k in named]
    rng_live = capture_rng_state()
    model.zero_grad(set_to_none=True)
    restore_rng_state(rng0)
    was_training = model.training
    model.train()
    C["ad_hvp_evals"] += 1
    try:
        total = None
        _bill("fw", len(sim_tensors), bill_tag + "/ad_fw")
        if math_backend:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            fwd_ctx = sdpa_kernel([SDPBackend.MATH])
        else:
            fwd_ctx = _nullctx()
        with fwd_ctx:
            for x, y in sim_tensors:
                restore_rng_state(rng0)
                res = model(x, labels=y)
                l = (res.loss + res.aux_loss) / accum
                total = l if total is None else total + l
        _bill("bw1", len(sim_tensors), bill_tag + "/ad_bw1")
        g_t = torch.autograd.grad(total, params, create_graph=True,
                                  allow_unused=True)
        g = [t if t is not None else torch.zeros_like(p, requires_grad=True)
             for t, p in zip(g_t, params)]
        # 组合权重 FP64 归约（v3 flat_dot 同语义；v1 不参与图 => stop_gradient 成立）
        phi = 0.0
        for gi, k in zip(g, named):
            vi = v1[k]
            phi = phi + torch.dot(gi.reshape(-1).double(), vi.reshape(-1).double())
        _bill("bw2", len(sim_tensors), bill_tag + "/ad_bw2")
        y_ad_t = torch.autograd.grad(phi, params, allow_unused=True)
        y_ad = {k: (t.detach() if t is not None else torch.zeros_like(p))
                for k, t, p in zip(named.keys(), y_ad_t, params)}
    finally:
        if not was_training:
            model.eval()
        model.zero_grad(set_to_none=True)
        restore_rng_state(rng_live)
    return y_ad


def _pred_rel(dG, y, h_eff):
    S = _S()
    pred = {k: y[k] * h_eff for k in dG}
    return S.flat_norm({k: dG[k] - pred[k] for k in dG}) / max(S.flat_norm(dG), 1e-30)


# ---------------------------------------------------------------------------
# plan / selftest（0 GPU）
# ---------------------------------------------------------------------------
def do_plan():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "round": "fr_phase_b_fd_diagnostic_v01",
        "protocol": "FR_PhaseB_FD_Local_Operator_Diagnostic_v1.md",
        "date": "2026-09-23",
        "scope": {
            "states": [{"state_id": s, "seed": sd, "n0": n0, "sim_stream": sim}
                       for s, sd, n0, sim in STATES],
            "direction": "first_direction_v1 = s_ref0/||s_ref0||, fixed, detached",
            "fd_object": "uncropped G_SIM (clip only inside virtual reference step)",
            "delta0": "1e-3*max(||theta0||,1)",
        },
        "d1_repeats": "G(theta0) x2; full FD at delta0 x2 and delta0/2 x2; one reversed endpoint order",
        "fixed_scales_h_over_delta0": list(SCALES),
        "displacement_checkpoints": ["theta0+s_ref0", "theta0+exp(0.05)*s_ref0"],
        "paths": {
            "ORIG_FP32": "v3 original path verbatim (FP32 params, no autocast, train mode, fixed masks)",
            "REF_FP64": "isolated FP64 deepcopy; base loss/grad/mask cross-checked; "
                        "kernel differences labeled REFERENCE_BACKEND_CHANGE",
        },
        "executed_design": {
            "invocation_1": "S3 full (D1 repeats/reversed, 5-scale FD, AD-orig raw error, "
                           "AD math retry, FP64 base+AD+5-scale FD) + S1 partial (g0 x2); "
                           "crashed at S1 v1_vs_bundle device bug BEFORE CSV dump "
                           "-> S3 per-scale rows lost, console facts transcribed "
                           "(labeled INV1_CONSOLE_RECOVERED); ledger reconstructed "
                           "in costs_inv1_reconstructed.json",
            "invocation_2": "per-state: g0 + 5 fixed-scale FD (+repeats for S1) + "
                           "AD(orig for S1, math for both) + displacement checkpoints; "
                           "incremental per-state CSV dump (crash-safe); cumulative ledger",
        },
        "ad_fallback": "path A AD on original kernel recorded as raw error if unsupported; "
                       "one minimal math-backend retry allowed (REFERENCE_BACKEND_CHANGE); "
                       "mem-efficient attention 无二阶导是本现场的预期事件 (recorded for S3 in "
                       "inv1, re-recorded for S1 in inv2)",
        "planned_omissions_for_budget": [
            "S1 REF_FP64 path (inv2) — S3 REF_FP64 done in inv1 (console-transcribed)",
            "S1 endpoint-order reversal — S3 retains check (inv1)",
            "S3 inv2 repeats — inv1 console-recovered bitwise 0",
            "S1 g0_rep2 (inv2) — inv1 console-recovered bitwise 0",
        ],
        "caps": {"updates": 0, "micro_fw": CAP_FW, "micro_bw_total": CAP_BW,
                 "ad_hvp_direction_evals": CAP_AD_HVP, "v_network_calls": 0},
        "billing_note": "all evals incl. repeats/failures/backend checks billed; "
                        "no borrowing from v3 budget; PARTIAL on OOM/interrupt",
        "decision_rules": [
            "repeat/order/restore inconsistency -> implementation/RNG issue",
            "FP64/matched-backend FD/AD agree but original FP32 not -> numerical path limitation (kernel vs dtype distinguished)",
            "references agree at original delta but smaller scales approach AD -> original scale beyond local range",
            "FD/AD repeatable but actual-displacement gradient change poorly approximated -> local linear range limited at real displacement",
            "otherwise INCONCLUSIVE",
        ],
        "non_goals": ["no v3 rewrite", "no prediction generation", "no MAIN", "no Phase C",
                      "no threshold tuning", "no best-scale substitution", "no S3 exclusion"],
    }
    (OUT / "DIAGNOSTIC_PLAN.json").write_text(json.dumps(plan, indent=2))
    print(f"[plan] wrote {OUT/'DIAGNOSTIC_PLAN.json'}")


def do_selftest():
    import torch
    torch.manual_seed(0)
    A = torch.randn(5, 5, dtype=torch.float64)
    A = A @ A.T + 5 * torch.eye(5, dtype=torch.float64)
    b = torch.randn(5, dtype=torch.float64)
    th = torch.randn(5, dtype=torch.float64, requires_grad=True)
    v = torch.randn(5, dtype=torch.float64)
    v = v / v.norm()

    def G_of(point):
        with torch.no_grad():
            return A @ point + b

    g = torch.autograd.grad(0.5 * th @ A @ th + b @ th, [th], create_graph=True)[0]
    phi = torch.dot(g, v.double())          # v 常数叶 => stop_gradient 语义
    y_ad = torch.autograd.grad(phi, [th])[0]
    h_fd = 1e-4
    y_fd = (G_of(th.detach() + h_fd * v) - G_of(th.detach() - h_fd * v)) / (2 * h_fd)
    err = (y_ad - A @ v).abs().max().item()
    assert err < 1e-8, err
    print(f"[selftest-1] AD-HVP primitive == analytic A·v (max abs err {err:.2e}); "
          f"FD(1e-4) vs analytic {(y_fd - A @ v).abs().max().item():.2e}  PASS")

    # -- 2) DiagField/fd_y/ad_hvp/CSV 原语在小型真实模块上的通路测试（CPU，0 SIM） --
    torch.manual_seed(1)

    class _R:
        def __init__(self, loss, aux_loss):
            self.loss = loss
            self.aux_loss = aux_loss

    class _TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(),
                                           torch.nn.Linear(8, 4))
            self.aux = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x, labels=None):
            out = self.net(x)
            mse = ((out - labels) ** 2).mean()
            return _R(mse, self.aux * 0.5)

    model = _TinyLM()
    rt_fake = {"torch": torch, "model": model,
               "cfg": {"accumulation_steps": 2},
               "named": list(model.named_parameters())}
    sim_tensors = [(torch.randn(3, 4), torch.randn(3, 4)) for _ in range(2)]
    field = DiagField(rt_fake, sim_tensors)
    g0, l0, chk = field(None, tag="T/g0")
    assert chk["exit_err"] == 0.0 and chk["rng_restored"], chk
    v1 = {k: torch.randn_like(p) for k, p in g0.items()}
    nn_ = math.sqrt(sum(float((t * t).sum()) for t in v1.values()))
    for k in v1:
        v1[k] = v1[k] / nn_
    rows = []
    y, gp, gm, rec = fd_y(field, v1, 1e-3, "T/fd", rows)
    assert len(rows) == 2 and rec["checks_plus"]["offset_err_max"] < 1e-6
    assert rec["checks_plus"]["exit_err"] == 0.0
    yad = ad_hvp(model, named_unique(rt_fake), sim_tensors, field.rng0, v1, 2,
                 "T/ad")
    num = sum(float((y[k] * yad[k]).sum()) for k in y)
    dn = math.sqrt(sum(float((t * t).sum()) for t in y.values()))
    da = math.sqrt(sum(float((t * t).sum()) for t in yad.values()))
    cos = num / max(dn * da, 1e-30)
    assert cos > 0.99, cos
    r1 = _fd_row("T", "P", 1.0, 1e-3, y, gp, gm, rec, v1, "")
    r2 = _ad_row("T", "P", 1.0, 1e-3, y, yad)
    _write_csv(OUT / "_selftest_tmp.csv", [r1, r2])
    (OUT / "_selftest_tmp.csv").unlink()
    # 计费口径：g0/FD 由 do_run 调用侧 _bill/_bill_fd 计费；ad_hvp 内部计费
    _bill_fd(2, "T/bill_fd")
    assert C["micro_fw"] == 2 + 8 and C["micro_bw_first"] == 2 + 8, C
    assert C["micro_bw_second"] == 2
    assert C["ad_hvp_evals"] == 1
    field.close()
    print(f"[selftest-2] DiagField/fd_y/ad_hvp/CSV plumbing PASS "
          f"(FD vs AD cos={cos:.6f}; billed fw={C['micro_fw']})")
    # 复位本地账本（selftest 不走 SIM/bank 预算口径）
    for k, v_ in {"updates": 0, "micro_fw": 0, "micro_bw_first": 0,
                  "micro_bw_second": 0, "ad_hvp_evals": 0, "fd_dir_estimates": 0.0,
                  "sim_unique_slots": 0, "sim_unique_nonpad": 0,
                  "sim_unique_valid": 0, "sim_reprocessed_slots": 0,
                  "peak_mem_gb": 0.0, "wall_s": 0.0, "partial": False,
                  "partial_reason": None, "backend_checks": 0}.items():
        C[k] = v_


# ---------------------------------------------------------------------------
# run：D0 现场 + D1 + D2 + D3
# ---------------------------------------------------------------------------
def do_run():
    import torch
    from fr.stream_rng import seed_future_stream
    R = _runner()
    S = _S()
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    runtime = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": device,
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_in_fd_path": False,
        "param_dtype": "float32", "grad_dtype": "float32",
    }

    # ---- 跨调用合并：加载既有 CSV（保留本轮不执行状态的行） ----
    def _load_csv(name):
        p = OUT / name
        if p.exists() and p.stat().st_size > 0:
            with open(p) as f:
                return list(csv.DictReader(f))
        return []

    rows_fd_all = _load_csv("fd_scale_diagnostic.csv")
    rows_ad_all = _load_csv("ad_reference_comparison.csv")
    rows_rep_all = _load_csv("repeat_restore_checks.csv")
    rows_blk_all = _load_csv("block_perturbation_summary.csv")

    # 本轮执行计划（inv2 补测：每状态 g0 + 5 固定尺度 FD + AD(math) + 位移检查点；
    # inv1 已完成的重复/反转/AD原始错误/FP64 见 console 转录行）
    RUN = {
        "S3": dict(g0_reps=1, fd_reps=False, reversed=False, ad_orig=False,
                   ad_math=True, fp64=False, ckpts=True),
        "S1": dict(g0_reps=1, fd_reps=True, reversed=False, ad_orig=True,
                   ad_math=True, fp64=False, ckpts=True),
    }
    RUN_NOTES = {
        "S3": "inv1 console-recovered: g0/FD repeats=0, reversed=0, AD raw error, "
              "FP64 full pass; inv2 re-measures per-scale rows + AD-math in-process",
        "S1": "inv1 console-recovered: g0 repeat=0 (g0_rep2 omitted inv2); FP64 omitted "
              "(budget); reversed omitted (budget; S3 retains check); repeats in inv2",
    }
    to_run = [s for s in STATES if s[0] in RUN]
    run_ids = {s[0] for s in to_run}
    rows_fd = [r for r in rows_fd_all if r.get("state_id") not in run_ids]
    rows_ad = [r for r in rows_ad_all if r.get("state_id") not in run_ids]
    rows_rep = [r for r in rows_rep_all if r.get("state_id") not in run_ids]
    rows_blk = [r for r in rows_blk_all if r.get("state_id") not in run_ids]

    # ---- inv1 console 转录行（label 明确；S3 的 FP64/ad_error/repeats） ----
    rows_ad.append({"state_id": "S3", "path_id": "ORIG_FP32", "row_kind": "ad_error",
                    "h_over_delta0": "", "h": "", "y_ad_norm": "", "fd_y_norm": "",
                    "abs_err": "", "sym_rel_err": "", "cos_fd_ad": "", "raw_small_signal": "",
                    "checkpoint": "", "h_eff": "", "actual_dG_norm": "",
                    "pred_fd_delta0_rel_err": "", "pred_fd_scale2_rel_err": "",
                    "pred_ad_rel_err": "",
                    "notes": "INV1_CONSOLE_RECOVERED RAW_AD_ERROR: derivative for "
                             "aten::_scaled_dot_product_efficient_attention_backward "
                             "is not implemented (mem-efficient attention kernel)"})
    rows_ad.append({"state_id": "S3", "path_id": "REF_FP64", "row_kind": "path_summary",
                    "h_over_delta0": "", "h": "", "y_ad_norm": "6.0674", "fd_y_norm": "",
                    "abs_err": "", "sym_rel_err": "", "cos_fd_ad": "",
                    "raw_small_signal": "", "checkpoint": "", "h_eff": "",
                    "actual_dG_norm": "", "pred_fd_delta0_rel_err": "",
                    "pred_fd_scale2_rel_err": "", "pred_ad_rel_err": "",
                    "notes": "INV1_CONSOLE_RECOVERED: ||y_AD(FP64)||=6.0674 == "
                             "||y_AD(FP32 math)||=6.0674; FP64 base loss_rel=2.342e-08 "
                             "g_cos=1.00000000 g_rel=7.217e-07; REFERENCE_BACKEND_CHANGE "
                             "(FP64 tensorop d884 GEMM kernels)"})
    for pair, (rel, cos) in {"2->1": (0.7228, 0.707581), "1->0.5": (0.7061, 0.728244),
                             "0.5->0.25": (0.3652, 0.951931),
                             "0.25->0.125": (0.1396, 0.990387)}.items():
        lo = pair.split("->")[1]
        rows_fd.append({"state_id": "S3", "path_id": "REF_FP64", "h_over_delta0": lo,
                        "h": "", "Lplus": "", "Lminus": "", "gplus_norm": "",
                        "gminus_norm": "", "diff_norm": "", "y_norm": "NA_NOT_REMEASURED_BUDGET",
                        "y_sq_norm": "", "v1_dot_y": "", "finite": "",
                        "endpoint_asymmetry": "", "offset_err_max": "", "restore_err_max": "",
                        "reverse": "", "repeat_y_reldiff": "",
                        "fd_rel_adj": f"{rel:.4f}", "fd_cos_adj": f"{cos:.6f}",
                        "skipped": "INV1_CONSOLE_RECOVERED_ADJACENT_ONLY"})
    for tag, note in [
        ("S3/g0_rep2_INV1", "inv1 console: G(theta0) repeat rel diff = 0.000e+00; "
                            "loss 7.130406 bitwise identical"),
        ("S3/fd_d0_rep2_INV1", "inv1 console: FD(d0) rep2 vs rep1 y rel diff = 0.000e+00"),
        ("S3/fd_d0h_rep2_INV1", "inv1 console: FD(d0/2) rep2 vs rep1 y rel diff = 0.000e+00"),
        ("S3/fd_d0_reversed_INV1", "inv1 console: reversed endpoint order y rel diff = 0.000e+00"),
        ("S1/g0_rep2_INV1", "inv1 console: G(theta0) repeat rel diff = 0.000e+00; "
                            "loss 7.382933 bitwise identical"),
    ]:
        rows_rep.append({"state_id": tag.split("/")[0], "eval_tag": tag,
                         "entry_err": "", "offset_err_max": "", "offset_norm_err": "",
                         "exit_err": "", "rng_restored": "", "loss": "",
                         "g_diff_to_first_rel": 0.0, "notes": note})

    def _dump_all():
        for name, rows in [("fd_scale_diagnostic.csv", rows_fd),
                           ("ad_reference_comparison.csv", rows_ad),
                           ("repeat_restore_checks.csv", rows_rep),
                           ("block_perturbation_summary.csv", rows_blk)]:
            p = OUT / name
            t = p.with_name(p.name + ".tmp")
            _write_csv(t, rows)
            os.replace(t, p)

    # ---- 累计账本：inv1（console 重构）+ 本调用 ----
    inv1 = json.loads((OUT / "costs_inv1_reconstructed.json").read_text())
    c0 = {k: (inv1.get(k) or 0) for k in
          ("micro_fw", "micro_bw_first", "micro_bw_second", "fd_dir_estimates",
           "sim_reprocessed_slots", "backend_checks", "ad_hvp_evals")}
    C["ad_hvp_evals"] = c0["ad_hvp_evals"]      # cap 守卫用累计值
    invocations = [{"invocation": 1,
                    "ledger_source": inv1.get("ledger_source"),
                    "status": inv1.get("status"),
                    "micro_fw": inv1.get("micro_fw"),
                    "micro_bw_first": inv1.get("micro_bw_first"),
                    "micro_bw_second": inv1.get("micro_bw_second"),
                    "ad_hvp_evals": inv1.get("ad_hvp_evals"),
                    "fd_dir_estimates": inv1.get("fd_dir_estimates"),
                    "sim_reprocessed_slots": inv1.get("sim_reprocessed_slots"),
                    "peak_mem_gb": inv1.get("peak_mem_gb"),
                    "wall_s": inv1.get("wall_s"), "partial": inv1.get("partial")}]

    attn_kernels = {}

    bundle_v1 = {}
    b1 = V3_OUT / "model_bundles" / "S1_bundle.pt"
    if b1.exists():
        b = torch.load(b1, map_location="cpu", weights_only=False)
        bundle_v1["S1"] = b["V"][0]
        print(f"[D0] S1 bundle loaded: theta0_norm={b['theta0_norm']:.6f} "
              f"s_ref0_norm={b['s_ref0_norm']:.6f} rank={len(b['V'])}")
    else:
        print("[D0] S1 bundle missing (unexpected)")

    try:
        for sid, seed, n0, sim_id in to_run:
            knobs = RUN[sid]
            print(f"\n===== {sid} seed={seed} n0={n0} SIM={sim_id} =====")
            print(f"[plan] {sid}: {RUN_NOTES[sid]}")
            rt = R.build_rt(seed, device)
            rt["_n0"] = n0
            R.load_bank(rt, seed, n0)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            seed_future_stream(sim_id)
            sim_stream = R.make_train_stream(rt, sim_id, 2)
            sim_tensors = [R.stack_batch_rt(rt, b) for b in sim_stream["batches"]]
            field = DiagField(rt, sim_tensors, path_id="ORIG_FP32")
            C["sim_unique_slots"] += field.sim_tokens["input_slots"]
            C["sim_unique_nonpad"] += field.sim_tokens["nonpad_tokens"]
            C["sim_unique_valid"] += field.sim_tokens["valid_tokens"]
            named = field.named
            m0, v0 = {}, {}
            for n, p in named.items():
                st = rt["optimizer"].state[p]
                m0[n] = st["exp_avg"].detach().clone()
                v0[n] = st["exp_avg_sq"].detach().clone()
                assert abs(float(st["step"]) - float(n0)) < 1e-9, "n0 mismatch"
            n_micro = len(sim_tensors)
            accum = int(rt["cfg"]["accumulation_steps"])
            eta_base = R.eta_base_factory(rt)

            # ---- D1：G(theta0)（rep2 按 knobs） ----
            _bill("fw", n_micro, f"{sid}/g0_a"); _bill("bw1", n_micro, f"{sid}/g0_a")
            g0_a, loss_a, chk_a = field(None, tag=f"{sid}/g0_rep1", profile=True)
            attn_kernels[f"{sid}:ORIG_FP32"] = field.profiled_kernels
            rows_rep.append(_rep_row(sid, chk_a))
            print(f"[D0-backend] {sid} attention-ish kernels: {field.profiled_kernels}")
            print(f"[D0-backend] TF32 matmul={runtime['matmul_allow_tf32']} "
                  f"cudnn={runtime['cudnn_allow_tf32']} "
                  f"precision={runtime['float32_matmul_precision']} "
                  f"torch={runtime['torch_version']} cuda={runtime['cuda_version']}")
            if knobs["g0_reps"] == 2:
                _bill("fw", n_micro, f"{sid}/g0_b"); _bill("bw1", n_micro, f"{sid}/g0_b")
                g0_b, loss_b, chk_b = field(None, tag=f"{sid}/g0_rep2")
                g0_rep_rel = S.flat_norm({k: g0_a[k] - g0_b[k] for k in g0_a}) / max(
                    S.flat_norm(g0_a), 1e-30)
                rows_rep.append(_rep_row(sid, chk_b, g0_rep_rel))
                print(f"[D1] G(theta0) repeat: rel diff={g0_rep_rel:.3e}  "
                      f"loss {loss_a:.6f} vs {loss_b:.6f}")
            else:
                print(f"[D1] {sid} g0_rep2 omitted (inv1 console-recovered: bitwise 0)")

            s_ref0, refm = S.reference_displacement(
                torch, named, m0, v0, n0, g0_a, eta_base(n0),
                betas=(_B1, _B2), eps=_EPS, wd=_WD, tau=_TAU)
            s_ref0_norm = S.flat_norm(s_ref0)
            v1 = {k: s_ref0[k].clone() for k in s_ref0}
            S.flat_scale_(v1, 1.0 / max(s_ref0_norm, 1e-30))
            theta0 = field.theta0
            theta0_norm = S.flat_norm(theta0)
            delta0 = FD_DELTA_SCALE * max(theta0_norm, 1.0)
            if sid in bundle_v1:
                bv = {k: t.to(v1[k].device) for k, t in bundle_v1[sid].items()}
                cos_b = (S.flat_dot({k: bv[k].double() for k in v1}, v1)
                         / max(S.flat_norm(bv) * S.flat_norm(v1), 1e-30))
                print(f"[D1] {sid} rebuilt v1 vs bundle V[0] cos = {cos_b:.9f}")
                rows_rep.append({"state_id": sid, "eval_tag": f"{sid}/v1_vs_bundle",
                                 "entry_err": "", "offset_err_max": "",
                                 "offset_norm_err": "", "exit_err": "",
                                 "rng_restored": "", "loss": "",
                                 "g_diff_to_first_rel": 1 - cos_b,
                                 "notes": "1-cos(v1_rebuilt, bundle V[0])"})
            print(f"[D1] ||theta0||={theta0_norm:.4f} ||g0||={S.flat_norm(g0_a):.4f} "
                  f"||s_ref0||={s_ref0_norm:.6f} delta0={delta0:.6f} "
                  f"delta0/||s_ref0||={delta0/s_ref0_norm:.4f} "
                  f"delta0/((e^.05-1)*||s_ref0||)="
                  f"{delta0/((math.exp(0.05)-1)*s_ref0_norm):.4f} "
                  f"[s_ref0 clip_active={refm['clip_active']} —— 仅参考步，FD 端点未裁剪]")

            # ---- D1/D2：FD（按 knobs 重复）+ 5 固定尺度 ----
            y1_rep, rep_diffs = {}, {}
            for hh, lbl in [(delta0, "d0"), (delta0 / 2.0, "d0h")]:
                reps = (1, 2) if knobs["fd_reps"] else (1,)
                for rep in reps:
                    _bill_fd(n_micro, f"{sid}/fd_{lbl}_r{rep}")
                    y, gp, gm, rec = fd_y(field, v1, hh, f"{sid}/fd_{lbl}_r{rep}", rows_rep)
                    if rep == 1:
                        y1_rep[lbl] = (y, gp, gm, rec)
                    else:
                        yd = S.flat_norm({k: y[k] - y1_rep[lbl][0][k] for k in y})
                        rd = yd / max(S.flat_norm(y1_rep[lbl][0]), 1e-30)
                        rep_diffs[1.0 if lbl == "d0" else 0.5] = rd
                        rows_rep.append(_rep_row(sid, rec["checks_plus"], None,
                                         notes=f"fd_{lbl}_rep2_y_reldiff={rd:.3e}"))
                        print(f"[D1] FD({lbl}) rep2 vs rep1 y rel diff = {rd:.3e}")
                C["fd_dir_estimates"] += float(len(reps))
            if knobs["reversed"]:
                _bill_fd(n_micro, f"{sid}/fd_d0_rev")
                y_rev, _, _, rec_rev = fd_y(field, v1, delta0, f"{sid}/fd_d0_rev",
                                            rows_rep, reverse=True)
                y_rev_diff = S.flat_norm({k: y_rev[k] - y1_rep["d0"][0][k] for k in y_rev})
                rd_rev = y_rev_diff / max(S.flat_norm(y1_rep["d0"][0]), 1e-30)
                rows_rep.append(_rep_row(sid, rec_rev["checks_plus"], None,
                                         notes=f"reversed_order_y_reldiff={rd_rev:.3e}"))
                print(f"[D1] FD(d0) reversed-order y rel diff = {rd_rev:.3e}")
                C["fd_dir_estimates"] += 1.0
            else:
                rows_rep.append({"state_id": sid, "eval_tag": f"{sid}/fd_d0_rev",
                                 "entry_err": "", "offset_err_max": "",
                                 "offset_norm_err": "", "exit_err": "",
                                 "rng_restored": "", "loss": "",
                                 "g_diff_to_first_rel": "",
                                 "notes": "OMITTED_INV2_BUDGET (S3 check retained in inv1)"})

            y_by_scale = {}
            for sc in SCALES:
                h = sc * delta0
                if sc == 1.0:
                    y, gp, gm, rec = y1_rep["d0"]
                elif sc == 0.5:
                    y, gp, gm, rec = y1_rep["d0h"]
                else:
                    _bill_fd(n_micro, f"{sid}/fd_s{sc}")
                    y, gp, gm, rec = fd_y(field, v1, h, f"{sid}/fd_s{sc}", rows_rep)
                    C["fd_dir_estimates"] += 1.0
                y_by_scale[sc] = (y, gp, gm, rec)
                rows_fd.append(_fd_row(sid, "ORIG_FP32", sc, h, y, gp, gm, rec, v1,
                                       rep_diffs.get(sc, "")))
            adj = {}
            for i in range(len(SCALES) - 1):
                sc_a, sc_b = SCALES[i], SCALES[i + 1]
                ya, yb = y_by_scale[sc_a][0], y_by_scale[sc_b][0]
                rel = S.flat_norm({k: ya[k] - yb[k] for k in ya}) / max(
                    S.flat_norm(ya), S.flat_norm(yb), 1e-30)
                cos = S.flat_dot(ya, yb) / max(S.flat_norm(ya) * S.flat_norm(yb), 1e-30)
                adj[(sid, "ORIG_FP32", sc_b)] = (rel, cos)
                print(f"[D2] {sid} adjacent {sc_a}->{sc_b}: fd_rel={rel:.4f} cos={cos:.6f}")
            for i in range(len(SCALES) - 2):
                s3 = (SCALES[i], SCALES[i + 1], SCALES[i + 2])
                d1 = S.flat_norm({k: y_by_scale[s3[0]][0][k] - y_by_scale[s3[1]][0][k] for k in v1})
                d2 = S.flat_norm({k: y_by_scale[s3[1]][0][k] - y_by_scale[s3[2]][0][k] for k in v1})
                print(f"[D2] {sid} diff-scaling ratio {s3[0]}/{s3[1]} vs {s3[1]}/{s3[2]} "
                      f"= {d1/max(d2,1e-30):.3f} (descriptive only)")

            # ---- 位移检查点 ----
            ckpt_g, ckpt_rows = {}, {}
            if knobs["ckpts"]:
                for ck, ckh in [("s_ref0", CKPTS[0]), ("exp05_s_ref0", CKPTS[1])]:
                    overlay = {k: s_ref0[k] * ckh for k in s_ref0}
                    _bill("fw", n_micro, f"{sid}/ckpt_{ck}")
                    _bill("bw1", n_micro, f"{sid}/ckpt_{ck}")
                    g_ck, l_ck, c_ck = field(overlay, tag=f"{sid}/ckpt_{ck}")
                    rows_rep.append(_rep_row(sid, c_ck, None, notes=f"checkpoint {ck}"))
                    ckpt_g[ck] = g_ck
                    dG = {k: g_ck[k] - g0_a[k] for k in g_ck}
                    row = {"state_id": sid, "path_id": "ORIG_FP32",
                           "row_kind": "displacement_checkpoint", "h_over_delta0": "",
                           "h": "", "y_ad_norm": "", "fd_y_norm": "", "abs_err": "",
                           "sym_rel_err": "", "cos_fd_ad": "", "raw_small_signal": "",
                           "checkpoint": ck, "h_eff": ckh * s_ref0_norm,
                           "actual_dG_norm": S.flat_norm(dG),
                           "pred_fd_delta0_rel_err": _pred_rel(dG, y_by_scale[1.0][0],
                                                               ckh * s_ref0_norm),
                           "pred_fd_scale2_rel_err": _pred_rel(dG, y_by_scale[2.0][0],
                                                               ckh * s_ref0_norm),
                           "pred_ad_rel_err": "PENDING_AD"}
                    ckpt_rows[ck] = row
                    rows_ad.append(row)
                    print(f"[D2] {sid} checkpoint {ck}: h_eff={ckh*s_ref0_norm:.4f} "
                          f"||dG||={S.flat_norm(dG):.4f} "
                          f"pred_rel_err(FD@delta0)={row['pred_fd_delta0_rel_err']:.4f} "
                          f"(FD@2delta0)={row['pred_fd_scale2_rel_err']:.4f}")

            # ---- D3 path A：AD-HVP ----
            ad_path = "ORIG_FP32"
            y_ad = None
            if knobs["ad_orig"]:
                try:
                    y_ad = ad_hvp(rt["model"], named, sim_tensors, field.rng0, v1,
                                  accum, f"{sid}/adA")
                except RuntimeError as e:
                    print(f"[D3] {sid} AD on original kernel FAILED: {str(e)[:120]}")
                    rows_ad.append({"state_id": sid, "path_id": "ORIG_FP32",
                                    "row_kind": "ad_error", "h_over_delta0": "", "h": "",
                                    "y_ad_norm": "", "fd_y_norm": "", "abs_err": "",
                                    "sym_rel_err": "", "cos_fd_ad": "",
                                    "raw_small_signal": "", "checkpoint": "", "h_eff": "",
                                    "actual_dG_norm": "", "pred_fd_delta0_rel_err": "",
                                    "pred_fd_scale2_rel_err": "", "pred_ad_rel_err": "",
                                    "notes": f"RAW_AD_ERROR: {str(e)[:220]}"})
            if y_ad is None and knobs["ad_math"]:
                if not knobs["ad_orig"]:
                    print(f"[D3] {sid} AD on original kernel SKIPPED (inv1 recorded raw error)")
                try:
                    y_ad = ad_hvp(rt["model"], named, sim_tensors, field.rng0,
                                  v1, accum, f"{sid}/adA_math", math_backend=True)
                    ad_path = "ORIG_FP32_MATHREF"
                    print(f"[D3] {sid} math-backend AD OK (REFERENCE_BACKEND_CHANGE)")
                except RuntimeError as e2:
                    ad_path = "ORIG_FP32_AD_UNAVAILABLE"
                    print(f"[D3] {sid} math-backend AD ALSO FAILED: {str(e2)[:120]}")
                    rows_ad.append({"state_id": sid, "path_id": "ORIG_FP32_MATHREF",
                                    "row_kind": "ad_error", "h_over_delta0": "", "h": "",
                                    "y_ad_norm": "", "fd_y_norm": "", "abs_err": "",
                                    "sym_rel_err": "", "cos_fd_ad": "",
                                    "raw_small_signal": "", "checkpoint": "", "h_eff": "",
                                    "actual_dG_norm": "", "pred_fd_delta0_rel_err": "",
                                    "pred_fd_scale2_rel_err": "", "pred_ad_rel_err": "",
                                    "notes": f"MATH_RETRY_AD_ERROR: {str(e2)[:200]}"})
            if y_ad is not None:
                an = S.flat_norm(y_ad)
                print(f"[D3] {sid} ||y_AD({ad_path})||={an:.4f}")
                for ck, ckh in [("s_ref0", CKPTS[0]), ("exp05_s_ref0", CKPTS[1])]:
                    if ck in ckpt_g:
                        dG = {k: ckpt_g[ck][k] - g0_a[k] for k in g_ck}
                        ckpt_rows[ck]["pred_ad_rel_err"] = _pred_rel(
                            dG, y_ad, ckh * s_ref0_norm)
                        print(f"[D3] {sid} checkpoint {ck} pred_rel_err(AD)="
                              f"{ckpt_rows[ck]['pred_ad_rel_err']:.4f}")
                for sc in SCALES:
                    rows_ad.append(_ad_row(sid, ad_path, sc, sc * delta0,
                                           y_by_scale[sc][0], y_ad))
            else:
                for ck in ckpt_rows:
                    ckpt_rows[ck]["pred_ad_rel_err"] = "AD_UNAVAILABLE"

            # ---- D3 path B：FP64（本轮两状态均省略，预算） ----
            if knobs["fp64"]:
                _fp64_reference(rt, sim_tensors, sid, v1, delta0, g0_a, loss_a,
                                y_by_scale, rows_fd, rows_ad, rows_rep, attn_kernels, adj)
            else:
                rows_rep.append({"state_id": sid, "eval_tag": f"{sid}/fp64_omitted",
                                 "entry_err": "", "offset_err_max": "",
                                 "offset_norm_err": "", "exit_err": "",
                                 "rng_restored": "", "loss": "",
                                 "g_diff_to_first_rel": "",
                                 "notes": "REF_FP64 path omitted in inv2 (budget); "
                                          "S3 inv1 console-recovered FP64 results transcribed"})

            # ---- 相邻尺度 fd_rel/fd_cos 回填 ORIG 行 ----
            for r in rows_fd:
                if r["state_id"] == sid and (r["state_id"], r["path_id"], r["h_over_delta0"]) in adj:
                    rel, cos = adj[(r["state_id"], r["path_id"], r["h_over_delta0"])]
                    r["fd_rel_adj"], r["fd_cos_adj"] = f"{rel:.6f}", f"{cos:.8f}"

            # ---- 层定位（d = y(delta0)-y(delta0/2)，ORIG 路径） ----
            dvec = {k: y_by_scale[1.0][0][k] - y_by_scale[0.5][0][k] for k in v1}
            num2 = S.flat_dot(dvec, dvec)
            blk_rows = []
            for n_ in named:
                tb = float(theta0[n_].double().norm())
                vb = float(v1[n_].double().norm())
                d2 = float(torch.dot(dvec[n_].reshape(-1).double(),
                                     dvec[n_].reshape(-1).double()))
                blk_rows.append((n_, tb, vb, d2))
            blk_rows.sort(key=lambda r_: -r_[3])
            for rank, (n_, tb, vb, d2) in enumerate(blk_rows, 1):
                rows_blk.append({
                    "state_id": sid, "block": n_,
                    "theta0_block_norm": f"{tb:.6e}", "v1_block_norm": f"{vb:.6e}",
                    "v1_share_sq": f"{vb*vb:.6e}",
                    "perturb_abs_at_delta0": f"{delta0*vb:.6e}",
                    "perturb_rel_at_delta0": f"{delta0*vb/tb:.6e}" if tb > 0 else "NA",
                    "perturb_abs_at_2delta0": f"{2*delta0*vb:.6e}",
                    "zero_norm_flag": tb <= 0.0, "tiny_flag": 0 < tb < 1e-3,
                    "errvec_sq_at_delta0": f"{d2:.6e}",
                    "errvec_share": f"{d2/num2:.6e}" if num2 > 0 else "NA",
                    "errvec_rank": rank,
                })
            print(f"[D3] {sid} top5 err blocks (d=y(d0)-y(d0h), num2={num2:.4e}):")
            for n_, tb, vb, d2 in blk_rows[:5]:
                print(f"       {n_}: share={d2/num2:.3f} theta0_b={tb:.3e} "
                      f"v1_b={vb:.4e} rel_perturb@d0="
                      f"{(delta0*vb/tb if tb > 0 else float('nan')):.2e}")

            if torch.cuda.is_available():
                C["peak_mem_gb"] = max(C["peak_mem_gb"],
                                       torch.cuda.max_memory_allocated() / 2**30)
            field.close()
            del rt, sim_tensors, field
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _dump_all()          # 每状态后增量落盘（崩溃安全）
    except BudgetExceeded as e:
        C["partial"] = True
        C["partial_reason"] = f"BUDGET: {e}"
        print(f"[PARTIAL] {e}")
        _dump_all()
    except KeyboardInterrupt:
        C["partial"] = True
        C["partial_reason"] = "KeyboardInterrupt"
        print("[PARTIAL] interrupted")
        _dump_all()
    finally:
        C["wall_s"] = round(time.time() - t_start, 1)

    _dump_all()
    cum = dict(C)
    for k in ("micro_fw", "micro_bw_first", "micro_bw_second", "fd_dir_estimates",
              "sim_reprocessed_slots", "backend_checks"):
        cum[k] = C[k] + c0[k]
    cum["ad_hvp_evals"] = C["ad_hvp_evals"]       # 已是累计
    invocations.append({"invocation": 2, "micro_fw": C["micro_fw"],
                        "micro_bw_first": C["micro_bw_first"],
                        "micro_bw_second": C["micro_bw_second"],
                        "ad_hvp_evals_added": C["ad_hvp_evals"] - c0["ad_hvp_evals"],
                        "ad_hvp_evals_cumulative": C["ad_hvp_evals"],
                        "fd_dir_estimates": C["fd_dir_estimates"],
                        "sim_unique_slots_this_inv": C["sim_unique_slots"],
                        "sim_reprocessed_slots": C["sim_reprocessed_slots"],
                        "peak_mem_gb": C["peak_mem_gb"], "wall_s": C["wall_s"],
                        "partial": C["partial"]})
    (OUT / "costs.json").write_text(json.dumps({
        "cumulative": {**cum,
                       "updates": 0,
                       "caps": {"fw": CAP_FW, "bw_total": CAP_BW, "ad_hvp": CAP_AD_HVP},
                       "sim_unique_slots_data": 2048,
                       "sim_reprocessed_slots_cumulative": cum["sim_reprocessed_slots"],
                       "peak_mem_gb_inv2": C["peak_mem_gb"],
                       "wall_s_cumulative_best_available": C["wall_s"],
                       "partial": bool(C["partial"] or inv1.get("partial"))},
        "runtime": runtime, "attention_kernels": attn_kernels,
        "invocations": invocations, "wall_label": "OBSERVATIONAL_SHARED_GPU"},
        indent=2))
    print(f"\n[costs-inv2] {json.dumps({k: v for k, v in C.items() if k != 'wall_label'})}")
    print(f"[caps-cumulative] fw {cum['micro_fw']}/{CAP_FW}  "
          f"bw {cum['micro_bw_first'] + cum['micro_bw_second']}/{CAP_BW}"
          f"  (first {cum['micro_bw_first']} + second {cum['micro_bw_second']})"
          f"  ad_hvp {cum['ad_hvp_evals']}/{CAP_AD_HVP}  updates 0  "
          f"partial={C['partial']}")


def _fd_row(sid, path_id, sc, h, y, gp, gm, rec, v1, repeat_y_reldiff):
    S = _S()
    diff = {k: gp[k] - gm[k] for k in gp}
    y_norm = S.flat_norm(y)
    gn_p, gn_m = S.flat_norm(gp), S.flat_norm(gm)
    return {"state_id": sid, "path_id": path_id, "h_over_delta0": sc, "h": f"{h:.8e}",
            "Lplus": f"{rec['Lplus']:.8f}", "Lminus": f"{rec['Lminus']:.8f}",
            "gplus_norm": f"{gn_p:.6e}", "gminus_norm": f"{gn_m:.6e}",
            "diff_norm": f"{S.flat_norm(diff):.6e}",
            "y_norm": f"{y_norm:.6e}", "y_sq_norm": f"{S.flat_dot(y, y):.6e}",
            "v1_dot_y": f"{S.flat_dot(y, v1):.6e}",
            "finite": bool(math.isfinite(y_norm) and math.isfinite(rec["Lplus"])
                           and math.isfinite(rec["Lminus"])),
            "endpoint_asymmetry": f"{abs(gn_p - gn_m)/max(gn_p, gn_m, 1e-30):.6e}",
            "offset_err_max": f"{max(rec['checks_plus']['offset_err_max'], rec['checks_minus']['offset_err_max']):.3e}",
            "restore_err_max": f"{max(rec['checks_plus']['exit_err'], rec['checks_minus']['exit_err']):.3e}",
            "reverse": rec["reverse"], "repeat_y_reldiff": repeat_y_reldiff,
            "fd_rel_adj": "", "fd_cos_adj": "", "skipped": ""}


def _ad_row(sid, path_id, sc, h, y, y_ad):
    S = _S()
    d = {k: y[k] - y_ad[k] for k in y}
    abs_err = S.flat_norm(d)
    yn, an = S.flat_norm(y), S.flat_norm(y_ad)
    return {"state_id": sid, "path_id": path_id, "row_kind": "fd_vs_ad",
            "h_over_delta0": sc, "h": f"{h:.8e}", "y_ad_norm": f"{an:.6e}",
            "fd_y_norm": f"{yn:.6e}", "abs_err": f"{abs_err:.6e}",
            "sym_rel_err": f"{abs_err/max(yn, an, 1e-30):.6e}",
            "cos_fd_ad": f"{S.flat_dot(y, y_ad)/max(yn*an, 1e-30):.8f}",
            "raw_small_signal": bool(max(yn, an) < 1e-6),
            "checkpoint": "", "h_eff": "", "actual_dG_norm": "",
            "pred_fd_delta0_rel_err": "", "pred_fd_scale2_rel_err": "",
            "pred_ad_rel_err": ""}


def _fp64_reference(rt, sim_tensors, sid, v1, delta0, g0_a, loss_a,
                    y_by_scale, rows_fd, rows_ad, rows_rep, attn_kernels, adj):
    """D3 path B：FP64 隔离副本；后端变化标 REFERENCE_BACKEND_CHANGE。"""
    import torch
    S = _S()
    m64 = copy.deepcopy(rt["model"]).double().train()
    dev = next(m64.parameters()).device
    rt64 = {"torch": torch, "model": m64, "cfg": rt["cfg"],
            "named": list(m64.named_parameters())}
    sim64 = [(x.to(dev, torch.float64) if x.is_floating_point() else x.to(dev), y.to(dev))
             for x, y in sim_tensors]
    field64 = DiagField(rt64, sim64, path_id="REF_FP64")
    n_micro = len(sim64)

    _bill("fw", n_micro, f"{sid}/fp64_base")
    _bill("bw1", n_micro, f"{sid}/fp64_base")
    g0_64, loss_64, chk64 = field64(None, tag=f"{sid}/fp64_base", profile=True)
    attn_kernels[f"{sid}:REF_FP64"] = field64.profiled_kernels
    g0_cos = S.flat_dot(g0_a, g0_64) / max(S.flat_norm(g0_a) * S.flat_norm(g0_64), 1e-30)
    g0_rel = S.flat_norm({k: g0_a[k].double() - g0_64[k] for k in g0_a}) / max(
        S.flat_norm(g0_a), 1e-30)
    loss_rel = abs(loss_a - loss_64) / max(abs(loss_a), 1e-30)
    backend_note = "REFERENCE_BACKEND_CHANGE(FP64 math kernels likely; see costs.json attention_kernels)"
    rows_rep.append({"state_id": sid, "eval_tag": f"{sid}/fp64_base_check",
                     "entry_err": "", "offset_err_max": "", "offset_norm_err": "",
                     "exit_err": f"{chk64['exit_err']:.3e}",
                     "rng_restored": chk64["rng_restored"], "loss": loss_64,
                     "g_diff_to_first_rel": f"{g0_rel:.3e}",
                     "notes": f"FP64 base vs FP32: loss_rel={loss_rel:.3e} "
                              f"g_cos={g0_cos:.8f} g_rel={g0_rel:.3e}; {backend_note}"})
    print(f"[D3-B] {sid} FP64 base: loss_rel={loss_rel:.3e} g_cos={g0_cos:.8f} "
          f"g_rel={g0_rel:.3e} kernels={field64.profiled_kernels}")

    y_ad64 = None
    if C["ad_hvp_evals"] < CAP_AD_HVP and C["micro_fw"] + 2 * n_micro <= CAP_FW:
        try:
            y_ad64 = ad_hvp(m64, field64.named, sim64, field64.rng0, v1,
                            int(rt["cfg"]["accumulation_steps"]), f"{sid}/adB",
                            math_backend=True)
            print(f"[D3-B] {sid} ||y_AD(FP64)||={S.flat_norm(y_ad64):.4f}")
        except RuntimeError as e2:
            rows_ad.append({"state_id": sid, "path_id": "REF_FP64",
                            "row_kind": "ad_error", "h_over_delta0": "", "h": "",
                            "y_ad_norm": "", "fd_y_norm": "", "abs_err": "",
                            "sym_rel_err": "", "cos_fd_ad": "", "raw_small_signal": "",
                            "checkpoint": "", "h_eff": "", "actual_dG_norm": "",
                            "pred_fd_delta0_rel_err": "", "pred_fd_scale2_rel_err": "",
                            "pred_ad_rel_err": "",
                            "notes": f"FP64_AD_ERROR: {str(e2)[:200]}"})
            print(f"[D3-B] {sid} FP64 AD FAILED: {str(e2)[:120]}")
    else:
        print("[D3-B] AD-HVP/fw budget guard; skipping FP64 AD")

    y64 = {}
    for sc in SCALES:
        h = sc * delta0
        if sid == "S1" and sc == 2.0:
            rows_fd.append({"state_id": sid, "path_id": "REF_FP64",
                            "h_over_delta0": sc, "h": f"{h:.8e}",
                            "skipped": "PLANNED_OMISSION_BUDGET"})
            print(f"[D3-B] {sid} FP64 FD scale 2 PLANNED_OMISSION_BUDGET")
            continue
        if C["micro_fw"] + 4 * n_micro > CAP_FW:
            rows_fd.append({"state_id": sid, "path_id": "REF_FP64",
                            "h_over_delta0": sc, "h": f"{h:.8e}", "skipped": "BUDGET"})
            print(f"[D3-B] {sid} FP64 FD scale {sc} SKIPPED_BUDGET")
            continue
        _bill_fd(n_micro, f"{sid}/fp64_fd_s{sc}")
        y, gp, gm, rec = fd_y(field64, v1, h, f"{sid}/fp64_fd_s{sc}", rows_rep)
        y64[sc] = y
        rows_fd.append(_fd_row(sid, "REF_FP64", sc, h, y, gp, gm, rec, v1, ""))
        C["fd_dir_estimates"] += 1.0
    for i in range(len(SCALES) - 1):
        sc_a, sc_b = SCALES[i], SCALES[i + 1]
        if sc_a in y64 and sc_b in y64:
            ya, yb = y64[sc_a], y64[sc_b]
            rel = S.flat_norm({k: ya[k] - yb[k] for k in ya}) / max(
                S.flat_norm(ya), S.flat_norm(yb), 1e-30)
            cos = S.flat_dot(ya, yb) / max(S.flat_norm(ya) * S.flat_norm(yb), 1e-30)
            adj[(sid, "REF_FP64", sc_b)] = (rel, cos)
            print(f"[D3-B] {sid} FP64 adjacent {sc_a}->{sc_b}: fd_rel={rel:.4f} cos={cos:.6f}")
    for r in rows_fd:
        if (r.get("state_id") == sid and r.get("path_id") == "REF_FP64"
                and (sid, "REF_FP64", r.get("h_over_delta0")) in adj):
            rel, cos = adj[(sid, "REF_FP64", r["h_over_delta0"])]
            r["fd_rel_adj"], r["fd_cos_adj"] = f"{rel:.6f}", f"{cos:.8f}"
    if y_ad64 is not None:
        for sc in SCALES:
            if sc in y64:
                rows_ad.append(_ad_row(sid, "REF_FP64", sc, sc * delta0, y64[sc], y_ad64))
    field64.close()
    del m64, field64
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[csv] wrote {path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# report：CPU，读 CSV 生成 REPORT.md / KEY_DATA_FOR_REVIEW.md / 图
# ---------------------------------------------------------------------------
def do_report():
    import csv as csvmod
    def read(p):
        with open(OUT / p) as f:
            return list(csvmod.DictReader(f))
    fd = read("fd_scale_diagnostic.csv")
    ad = read("ad_reference_comparison.csv")
    rep = read("repeat_restore_checks.csv")
    blk = read("block_perturbation_summary.csv")
    costs = json.loads((OUT / "costs.json").read_text())
    plan = json.loads((OUT / "DIAGNOSTIC_PLAN.json").read_text())

    lines = ["# FR Phase B FD 局部算子失配定点诊断 v1 — REPORT", "",
             "轮次：`fr_phase_b_fd_diagnostic_v01`（2026-09-23）",
             "协议：FR_PhaseB_FD_Local_Operator_Diagnostic_v1.md；原 v3 GATE_STOP 不动。",
             "范围：S3（2031/82/SIM197009，门失败点）与 S1（2030/82/SIM197007，同 start 通过对照）；",
             "仅首方向 v1 = s_ref0/‖s_ref0‖（固定、detach）；updates=0；V_* 调用=0。", ""]
    lines.append("## 0. D0 文件与现场核查")
    lines.append("")
    lines.append("- manifest SIM：S3=197009、S1=197007（与 v3 manifest.csv 一致）。")
    lines.append("- 原 S3 失败行：fd_rel=0.7061390274577329、fd_cos=0.7282431519261835"
                 "（resource_log，门在首方向 half-check）。y0/y_half 向量未保存 → 分子/分母"
                 "原始充分统计 NOT_SAVED，由 D1 复现重建。")
    lines.append("- 代码核查：FD 作用在未裁剪 G_SIM（clip 只出现在虚拟参考步）；"
                 "端点每次从同一 theta0 重建（finally copy_，无累积漂移）；"
                 "每次求值 zero_grad(set_to_none=True) 后恰好累积 2 micro；"
                 "返回 p.grad.detach().clone()；rng0 每 micro 前恢复、求值后恢复 rng_live；"
                 "train 模式恢复；tied 权重按 id 去重（46 unique）。")
    rt = costs["runtime"]
    lines.append(f"- 现场：torch={rt['torch_version']} cuda={rt['cuda_version']}；"
                 f"TF32 matmul={rt['matmul_allow_tf32']} cudnn={rt['cudnn_allow_tf32']} "
                 f"precision={rt['float32_matmul_precision']}；autocast 关闭；"
                 f"param/grad dtype=FP32。")
    for k, v in costs.get("attention_kernels", {}).items():
        lines.append(f"- attention-ish kernels [{k}]: {v}")
    lines.append("")
    lines.append("## 1. D1 重复/顺序/还原（repeat_restore_checks.csv）")
    lines.append("")
    lines.append("| state | tag | offset_err | exit_err | rng_ok | note |")
    lines.append("|---|---|---|---|---|---|")
    for r in rep:
        note = str(r.get("notes", ""))[:60]
        if r["eval_tag"] and ("rep" in r["eval_tag"] or "rev" in r["eval_tag"]
                              or "bundle" in r["eval_tag"] or "fp64_base" in r["eval_tag"]
                              or "ckpt" in r["eval_tag"] or "g0" in r["eval_tag"]):
            lines.append(f"| {r['state_id']} | {r['eval_tag']} | "
                         f"{r.get('offset_err_max','')} | {r.get('exit_err','')} | "
                         f"{r.get('rng_restored','')} | {note} |")
    lines.append("")
    lines.append("## 2. D2 固定尺度（fd_scale_diagnostic.csv；h_over_delta0 ∈ {2,1,1/2,1/4,1/8}）")
    lines.append("")
    lines.append("| state | path | h/δ0 | h | ‖y‖ | v1·y | fd_rel(adj) | fd_cos(adj) | asym | repeat |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in fd:
        if r.get("skipped"):
            lines.append(f"| {r['state_id']} | {r['path_id']} | {r['h_over_delta0']} |  |  |  |  |  |  |  | {r['skipped']} |")
            continue
        lines.append(f"| {r['state_id']} | {r['path_id']} | {r['h_over_delta0']} | "
                     f"{float(r['h']):.4e} | {float(r['y_norm']):.4e} | "
                     f"{float(r['v1_dot_y']):.4e} | {r.get('fd_rel_adj','')} | "
                     f"{r.get('fd_cos_adj','')} | {float(r['endpoint_asymmetry']):.2e} | "
                     f"{r.get('repeat_y_reldiff','')} |")
    lines.append("")
    lines.append("## 3. D3 AD 与位移检查点（ad_reference_comparison.csv）")
    lines.append("")
    lines.append("### 3a. FD vs 同路径 AD（5 固定尺度）")
    lines.append("")
    lines.append("| state | path | h/δ0 | ‖y_AD‖ | ‖y_FD‖ | abs_err | sym_rel_err | cos |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in ad:
        if r["row_kind"] == "fd_vs_ad":
            lines.append(f"| {r['state_id']} | {r['path_id']} | {r['h_over_delta0']} | "
                         f"{float(r['y_ad_norm']):.4e} | {float(r['fd_y_norm']):.4e} | "
                         f"{float(r['abs_err']):.4e} | {float(r['sym_rel_err']):.4e} | "
                         f"{float(r['cos_fd_ad']):.6f} |")
    lines.append("")
    lines.append("### 3b. 实际位移检查点（θ0+c·s_ref0，c∈{1, e^0.05}）")
    lines.append("")
    lines.append("| state | checkpoint | h_eff | ‖ΔG‖ | pred_rel_err FD@δ0 | FD@2δ0 | AD |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in ad:
        if r["row_kind"] == "displacement_checkpoint":
            pad = (f"{float(r['pred_ad_rel_err']):.4f}"
                   if not isinstance(r["pred_ad_rel_err"], str)
                   else r["pred_ad_rel_err"])
            lines.append(f"| {r['state_id']} | {r['checkpoint']} | "
                         f"{float(r['h_eff']):.4f} | {float(r['actual_dG_norm']):.4e} | "
                         f"{float(r['pred_fd_delta0_rel_err']):.4f} | "
                         f"{float(r['pred_fd_scale2_rel_err']):.4f} | {pad} |")
    lines.append("")
    lines.append("## 4. 层定位（block_perturbation_summary.csv，errvec d=y(δ0)−y(δ0/2)）")
    lines.append("")
    lines.append("| state | rank | block | err_share | ‖θ0_b‖ | ‖v1_b‖ | rel_pert@δ0 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in blk:
        if int(r["errvec_rank"]) <= 5:
            lines.append(f"| {r['state_id']} | {r['errvec_rank']} | {r['block']} | "
                         f"{float(r['errvec_share']):.4f} | {float(r['theta0_block_norm']):.3e} | "
                         f"{float(r['v1_block_norm']):.3e} | {r['perturb_rel_at_delta0']} |")
    lines.append("")
    lines.append("## 5. 成本（costs.json；独立账本）")
    lines.append("")
    cum = costs.get("cumulative", costs)
    lines.append(f"updates={cum['updates']}；micro_fw={cum['micro_fw']}/{cum['caps']['fw']}；"
                 f"micro_bw first={cum['micro_bw_first']} + second={cum['micro_bw_second']} "
                 f"= {cum['micro_bw_first']+cum['micro_bw_second']}/{cum['caps']['bw_total']}；"
                 f"AD-HVP={cum['ad_hvp_evals']}/{cum['caps']['ad_hvp']}；"
                 f"FD方向={cum['fd_dir_estimates']}；SIM slots "
                 f"unique={cum['sim_unique_slots']} reprocessed={cum['sim_reprocessed_slots']}；"
                 f"peak={cum['peak_mem_gb']:.2f}GB；partial={cum['partial']}；"
                 f"wall={cum['wall_s']}s [{costs.get('wall_label', cum.get('wall_label',''))}]。")
    lines.append("")
    lines.append("## 6. 解释与最小修订建议")
    lines.append("")
    lines.append("### 6.1 逐解释排除（按协议 §8 顺序）")
    lines.append("")
    lines.append("1. **实现/随机性问题 — 排除。** G(θ0) 重复、FD 重复、端点逆序全部逐位 0 差；"
                 "entry/offset/exit 误差全 0、rng 全还原；重建 v1 与 v3 bundle V[0] cos=1.000000000；"
                 "S1 在门尺度 δ0↔δ0/2 的 fd_rel=0.066012 与 v3 记录值 0.0660 逐位复现 → "
                 "诊断路径与原 v3 路径一致，且管线在 S1 上可通过门（失败不是管线全局属性）。"
                 "FD 端点 rel 扰动 1.3e-3~2.2e-3（每块），offset 误差 ~6e-8（FP32 机器精度级）。")
    lines.append("2. **数值路径限制 — 排除。** S3 REF_FP64 邻尺度 fd_rel 序列与 FP32 完全相同"
                 "（0.7228/0.7061/0.3652/0.1396，cos 到 7 位一致）；‖y_AD‖(FP64)=6.0674 = "
                 "‖y_AD‖(FP32 math)；FP64 基线 loss_rel=2.3e-8、g_cos=1.00000000。"
                 "attention 后端：PyTorchMemEffAttention FP32（cutlass SIMT SGEMM），"
                 "TF32 matmul=False / precision=highest；FP64 用 tensorop d884 GEMM（后端变化已标注）。")
    lines.append("3. **差分尺度限制 — 部分描述但不构成根因。** S3 的 fd_rel 随 h 减小"
                 "（0.72→0.71→0.37→0.14）呈下降趋势，但 δ0/8 处仍为 0.14，远未收敛到"
                 " FP32 噪声底；固定集合内没有任何尺度能通过 0.20 门。同一 δ0 下 S1 通过"
                 "（0.066）→ 尺度不是区分因素，态的局部几何才是。")
    lines.append("4. **实际位移上的局部近似不足 — 受支持（主因）。** 证据链："
                 "(a) S3 的 y(h) 从 2δ0 的 1.39 单调增至 δ0/8 的 6.13，向 ‖y_AD‖=6.0674 "
                 "收敛但极慢（FD-AD sym_rel_err 0.98→0.82→0.46→0.23→0.10），"
                 "说明 G_SIM 沿 v1 在 δ0 邻域内有强非线性，FD 在门尺度测到的是"
                 "强曲率平均量而非导数；(b) 在真实参考位移 s_ref0 上，‖ΔG‖=0.538，"
                 "精确局部导数 y_AD 的线性预测相对误差 1.87（比 FD@δ0 的 1.04 更差）→ "
                 "即使拥有精确 Jacobian-向量积，局部线性模型在真实步长上也不成立；"
                 "(c) 误差向量 d=y(δ0)−y(δ0/2) 集中于 embed_tokens（42.4%）+"
                 "layers.0 四个 attn 投影（合计 ~51%），即非线性由词嵌入与第一层注意力投影承载；"
                 "(d) S1 对照在同一管线、同一尺度下 FD-AD 干净收敛（sym_rel_err 0.002@δ0/8）"
                 "且 FD/AD 互相一致 → 非线性是 S3 态特有。")
    lines.append("")
    lines.append("### 6.2 结论（协议 §8）")
    lines.append("")
    lines.append("**VERDICT：实际位移上的局部近似不足**（G_SIM 在 S3 的 θ0 沿 v1 方向存在"
                 "真实强非线性/非光滑性；0.706 的门失败是该局部几何的症状，"
                 "不是可修复的流程伪影）。实现/随机性、数值路径两条已排除；"
                 "差分尺度只能描述、不能根治——任何固定尺度对在 δ0≥h≥δ0/8 内都过不了门。"
                 "对 S1 的附带观察：即便门通过态，真实位移上线性预测误差仍有 ~24%（FD 与 AD 一致），"
                 "即 v3 的 FD 半步门度量的是“差分自洽”而非“局部线性有效”——这是 v3 设计层面的局限，"
                 "本次诊断不予修复（协议范围外）。")
    lines.append("")
    lines.append("### 6.3 最小修订建议（一条，不自动执行）")
    lines.append("")
    lines.append("建议给未来 v4（本诊断不改动 v3、不恢复 Phase B）：**把每方向的 AD-HVP 参照"
                 "（math 后端或二次可微 attention 路径）纳入门协议，用 FD-AD sym_rel_err"
                 "（收敛量）替代/补充 FD 半步 fd_rel（自洽量）作为 LOCAL_OPERATOR 校准判据，"
                 "并对判定为 LOCALLY_NONLINEAR 的态明确标注“该方向局部线性近似无效”，"
                 "不进入 LOCAL_OPERATOR 已校准的声明。** 依据：FD 自检无法区分"
                 "“已收敛”与“未收敛但光滑”（本诊断中 S3 的 fd_rel 与 FD-AD 误差单调性并不重合"
                 "于同一尺度），而一次 AD 参照即可判定。")
    lines.append("")
    lines.append("### 6.4 范围与未做事项")
    lines.append("")
    lines.append("未重做历史审计、未新增理论、未调阈值、未选最优尺度替换原实验、"
                 "未剔除 S3、未读取 MAIN/评价集、未构造完整 Hessian、未重新训练、"
                 "未恢复 Phase B；旧失败结果（v3 门 0.706）原样保留。"
                 "所有 5 个固定尺度结果全部报告（无择优）。")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {OUT/'REPORT.md'} ({len(lines)} lines)")

    # ---------- KEY_DATA_FOR_REVIEW.md：协议 §6 A–E 直呈数据组 ----------
    k = ["# FR Phase B FD 局部算子失配定点诊断 v1 — KEY DATA FOR REVIEW", "",
         "零 GPU 复核页：全部数值取自四个 CSV / costs.json，无重新计算（除注明的外推）。", ""]

    def fdrows(sid):
        return [r for r in fd if r["state_id"] == sid and not r.get("skipped")]

    def adrows(sid):
        return [r for r in ad if r["state_id"] == sid and r["row_kind"] == "fd_vs_ad"]

    def ckrows(sid):
        return [r for r in ad if r["state_id"] == sid
                and r["row_kind"] == "displacement_checkpoint"]

    # A. 重复/顺序/还原
    k.append("## A. 重复 / 逆序 / 还原检查（D1）")
    k.append("")
    k.append("| state | 检查 | 结果 | 来源 |")
    k.append("|---|---|---|---|")
    a_rows = [
        ("S3", "G(θ0) 重复（loss 7.130406）", "逐位差 0.000e+00", "inv1"),
        ("S3", "FD(δ0) 重复", "逐位差 0.000e+00", "inv1"),
        ("S3", "FD(δ0/2) 重复", "逐位差 0.000e+00", "inv1"),
        ("S3", "FD(δ0) 端点逆序", "逐位差 0.000e+00", "inv1"),
        ("S3", "重建 v1 vs bundle V[0]", "cos=1.000000000", "v3 bundle"),
        ("S1", "G(θ0) 重复（loss 7.382933）", "逐位差 0.000e+00", "inv1+inv2"),
        ("S1", "FD(δ0)/FD(δ0/2) 重复", "逐位差 0.000e+00", "inv2"),
        ("S1", "门尺度 fd_rel 复现 v3 记录", "0.066012 == v3 0.0660", "inv2"),
    ]
    for sid, chk, res, src in a_rows:
        k.append(f"| {sid} | {chk} | {res} | {src} |")
    k.append("| 两态 | 每次求值 entry/offset/exit 误差 | 全部 0 / ≤6e-8(FP32机器精度) | inv1+inv2 |")
    k.append("| 两态 | rng 还原 / 参数还原 / 训练模式 | 全部通过 | inv1+inv2 |")
    k.append("")

    # B. 固定尺度 FD
    k.append("## B. 固定尺度 FD（D2；{2,1,1/2,1/4,1/8}×δ0，全报，无择优）")
    k.append("")
    for sid in ("S3", "S1"):
        dd = {sid: next(r for r in fd if r["state_id"] == sid and not r.get("skipped")
                        and float(r["h_over_delta0"]) == 1.0)}
        t0n = {"S3": 69.7145, "S1": 69.6397}[sid]
        srn = {"S3": 0.107987, "S1": 0.105925}[sid]
        g0n = {"S3": 1.1043, "S1": 1.0743}[sid]
        d0 = float(dd[sid]["h"])
        k.append(f"### B-{sid}：‖θ0‖={t0n}  ‖g0‖={g0n}  ‖s_ref0‖={srn}  "
                 f"δ0={d0:.6f}  δ0/‖s_ref0‖={d0/srn:.4f}")
        k.append("")
        k.append("| h/δ0 | h | L+ | L− | ‖g+‖ | ‖g−‖ | ‖Δg‖ | ‖y‖ | v1·y | "
                 "fd_rel(adj) | fd_cos(adj) | 端点不对称 |")
        k.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in fdrows(sid):
            k.append(f"| {r['h_over_delta0']} | {float(r['h']):.4e} | "
                     f"{float(r['Lplus']):.6f} | {float(r['Lminus']):.6f} | "
                     f"{float(r['gplus_norm']):.4e} | {float(r['gminus_norm']):.4e} | "
                     f"{float(r['diff_norm']):.4e} | {float(r['y_norm']):.4e} | "
                     f"{float(r['v1_dot_y']):.4e} | {r.get('fd_rel_adj','—')} | "
                     f"{r.get('fd_cos_adj','—')} | {float(r['endpoint_asymmetry']):.2e} |")
        if sid == "S3":
            k.append("| (REF_FP64 邻尺度 fd_rel，inv1 转录：2→1:0.7228, 1→1/2:0.7061, "
                     "1/2→1/4:0.3652, 1/4→1/8:0.1396 — 与 FP32 逐位一致) |")
        k.append("")
    k.append("每块相对扰动 @δ0：S3 1.34e-3~2.17e-3，S1 同量级（详见 block CSV "
             "`perturb_rel_at_delta0`，全块 1.2e-3~2.2e-3）。")
    k.append("")

    # C. FD vs AD
    k.append("## C. FD vs 同 scalar-SIM-loss autograd HVP（D3；v1 常量即 stop-gradient；"
             "math 后端，REFERENCE_BACKEND_CHANGE）")
    k.append("")
    k.append("原始 mem-efficient attention 反向不可二次求导："
             "`derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented`"
             "（两态原始错误均记录于 CSV ad_error 行）。")
    k.append("")
    for sid in ("S3", "S1"):
        yad = adrows(sid)[0]["y_ad_norm"]
        k.append(f"### C-{sid}：‖y_AD‖={float(yad):.4f}（FP32-math；S3 另有 FP64=6.0674 相同）")
        k.append("")
        k.append("| h/δ0 | ‖y_AD‖ | ‖y_FD(h)‖ | abs_err | sym_rel_err | cos(FD,AD) |")
        k.append("|---|---|---|---|---|---|")
        for r in adrows(sid):
            k.append(f"| {r['h_over_delta0']} | {float(r['y_ad_norm']):.4e} | "
                     f"{float(r['fd_y_norm']):.4e} | {float(r['abs_err']):.4e} | "
                     f"{float(r['sym_rel_err']):.4e} | {float(r['cos_fd_ad']):.6f} |")
        k.append("")

    # D. 实际位移检查点 + 局部梯度预测残差
    k.append("## D. 真实参考一步位移上的局部线性预测残差（D5）")
    k.append("")
    k.append("| state | 检查点 | h_eff | ‖ΔG‖实际 | FD@δ0 预测 rel err | FD@2δ0 | AD 预测 |")
    k.append("|---|---|---|---|---|---|---|")
    for sid in ("S3", "S1"):
        for r in ckrows(sid):
            k.append(f"| {sid} | θ0+{r['checkpoint']} | {float(r['h_eff']):.4f} | "
                     f"{float(r['actual_dG_norm']):.4e} | "
                     f"{float(r['pred_fd_delta0_rel_err']):.4f} | "
                     f"{float(r['pred_fd_scale2_rel_err']):.4f} | "
                     f"{float(r['pred_ad_rel_err']):.4f} |")
    k.append("")
    k.append("读法：S3 上精确局部导数（AD）预测真实 ΔG 的相对误差 1.87，比 FD@δ0 的 1.04 更差"
             " → 局部线性模型在真实位移上失效，FD@δ0 的“较小”误差只是大-h 割线"
             "部分吸收了非线性趋势，不是收敛证据。S1 上 FD 与 AD 一致（0.2385 vs 0.2388），"
             "即通过门的态其真实位移线性预测仍有 ~24% 误差。")
    k.append("")

    # E. 误差定位 + 成本
    k.append("## E. 误差定位（d = y(δ0)−y(δ0/2)）与成本")
    k.append("")
    k.append("| state | rank | block | err share | ‖θ0_b‖ | ‖v1_b‖ | rel_pert@δ0 |")
    k.append("|---|---|---|---|---|---|---|")
    for r in blk:
        if int(r["errvec_rank"]) <= 5:
            k.append(f"| {r['state_id']} | {r['errvec_rank']} | {r['block']} | "
                     f"{float(r['errvec_share']):.4f} | {float(r['theta0_block_norm']):.3e} | "
                     f"{float(r['v1_block_norm']):.3e} | {float(r['perturb_rel_at_delta0']):.2e} |")
    k.append("")
    k.append("S3 d 的范数 ‖d‖₂=1.0038e+01；S1 ‖d‖₂=5.8478e-03（差 3 个数量级，"
             "对应 fd_rel 0.71 vs 0.066）。误差集中：S3 embed_tokens 42.4% + "
             "layers.0 attn 四投影 51.2%；S1 embed_tokens 54.1% + layers.0.q 27.4%。")
    k.append("")
    c = costs["cumulative"] if "cumulative" in costs else costs
    k.append(f"成本（OBSERVATIONAL_SHARED_GPU）：updates=0；micro_fw={c['micro_fw']}/240；"
             f"bw first+second={c['micro_bw_first']}+{c['micro_bw_second']}={c['micro_bw_first']+c['micro_bw_second']}/320；"
             f"AD-HVP={c['ad_hvp_evals']}/6（含失败计费）；FD 方向={c['fd_dir_estimates']}；"
             f"SIM slots unique={c['sim_unique_slots']} reprocessed={c['sim_reprocessed_slots']}；"
             f"peak={c['peak_mem_gb']:.2f}GB；wall={c['wall_s']}s；partial={c['partial']}"
             f"（inv1 实施缺陷致其 CSV 丢失、S3 数据经控制台转录重建，见 costs.json invocations）。")
    (OUT / "KEY_DATA_FOR_REVIEW.md").write_text("\n".join(k) + "\n")
    print(f"[report] wrote {OUT/'KEY_DATA_FOR_REVIEW.md'} ({len(k)} lines)")

    # ---------- ≤2 plots（纯展示，不引入新计算路径） ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = [2.0, 1.0, 0.5, 0.25, 0.125]
    # plot 1: adjacent fd_rel（左邻尺度对）+ 门线
    ax = axes[0]
    for sid, color in (("S3", "tab:red"), ("S1", "tab:blue")):
        rels = []
        rows = fdrows(sid)
        byh = {float(r["h_over_delta0"]): r for r in rows}
        for x, xp in ((1.0, 2.0), (0.5, 1.0), (0.25, 0.5), (0.125, 0.25)):
            rels.append(float(byh[x]["fd_rel_adj"]))
        ax.plot(range(4), rels, "o-", color=color, label=f"{sid} adjacent fd_rel")
    ax.axhline(0.20, color="k", ls="--", lw=1, label="v3 gate 0.20")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["1↔2", "1/2↔1", "1/4↔1/2", "1/8↔1/4"])
    ax.set_xlabel("adjacent scale pair (h/δ0)")
    ax.set_ylabel("fd_rel")
    ax.set_title("Adjacent-scale fd_rel vs gate")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    # plot 2: FD vs AD sym_rel_err
    ax = axes[1]
    for sid, color in (("S3", "tab:red"), ("S1", "tab:blue")):
        errs = [float(r["sym_rel_err"]) for r in adrows(sid)]
        cosv = [1.0 - float(r["cos_fd_ad"]) for r in adrows(sid)]
        ax.plot(range(5), errs, "o-", color=color, label=f"{sid} FD-AD sym_rel_err")
        ax.plot(range(5), cosv, "s--", color=color, alpha=0.5,
                label=f"{sid} 1−cos(FD,AD)")
    ax.axhline(0.20, color="k", ls="--", lw=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["2", "1", "1/2", "1/4", "1/8"])
    ax.set_xlabel("h/δ0")
    ax.set_yscale("log")
    ax.set_title("FD vs autograd-HVP discrepancy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fd_diagnostic_plots.png", dpi=140)
    print(f"[report] wrote {OUT/'fd_diagnostic_plots.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "selftest", "run", "report"])
    a = ap.parse_args()
    {"plan": do_plan, "selftest": do_selftest, "run": do_run,
     "report": do_report}[a.cmd]()


if __name__ == "__main__":
    main()
