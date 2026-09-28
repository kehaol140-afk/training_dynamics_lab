"""A0：FR 正结果可靠度、混杂与 AdamW direct-path 诊断工具。

科学边界
--------
- [FR-OBSERVATION-ONLY] 只读取既有 v0.7 raw/state/q；不运行训练、不改状态。
- [FR-METRIC] v0.7 H8_CONFIRMATION=FAIL 永不改写。
- 2030..2037 在 v0.7 后已成为 mechanism-development set；A0 只作 post-confirmatory diagnostic。
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def atomic_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)


def atomic_json(path: str | Path, obj) -> None:
    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_recorded_path(project_root: str | Path, recorded: str | Path) -> Path:
    root = Path(project_root).resolve()
    p = Path(recorded)
    if p.exists():
        return p.resolve()
    if not p.is_absolute():
        q = (root / p).resolve()
        if q.exists():
            return q
    parts = list(p.parts)
    for anchor in ("outputs", "research", "minimind"):
        if anchor in parts:
            q = root.joinpath(*parts[parts.index(anchor):])
            if q.exists():
                return q.resolve()
    raise FileNotFoundError(f"无法解析记录路径: {recorded}")


def load_formal_states(project_root: str | Path, cfg: Mapping) -> Tuple[List[Dict], List[Path]]:
    """读取 v0.7 theta_mapping_raw.json 的 56 个 state record。"""
    root = Path(project_root).resolve()
    src = root / str(cfg["source_output_dir"])
    seeds = list(map(int, cfg["confirmation_train_seeds"]))
    steps = list(map(int, cfg["state_steps"]))
    rows: List[Dict] = []
    files: List[Path] = []
    for seed in seeds:
        p = src / f"seed_{seed}" / "theta_mapping_raw.json"
        if not p.exists():
            raise FileNotFoundError(p)
        obj = read_json(p)
        if obj.get("technical_pass") is not True:
            raise RuntimeError(f"formal raw technical_pass != True: {p}")
        by_step = {int(s["step"]): s for s in obj.get("states", [])}
        if set(by_step) != set(steps):
            raise RuntimeError(f"seed={seed} state set 不匹配: {sorted(by_step)}")
        for step in steps:
            r = dict(by_step[step])
            r["train_seed"] = seed
            rows.append(r)
        files.append(p)
    if len(rows) != len(seeds) * len(steps):
        raise RuntimeError("formal state count mismatch")
    return rows, files


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size != b.size or a.size < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    if float(a.std()) <= 1e-15 or float(b.std()) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata_average(x: Sequence[float]) -> np.ndarray:
    """无 scipy 的 average-tie ranks。"""
    a = np.asarray(x, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and a[order[j]] == a[order[i]]:
            j += 1
        rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(rankdata_average(x), rankdata_average(y))


def observational_slope(s_values: Sequence[float], y_values: Sequence[float], eps: float = 1e-30) -> float:
    s = np.asarray(s_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if s.shape != y.shape or s.size < 2:
        raise ValueError("s/y 尺寸不一致或样本不足")
    ds = s - s.mean()
    denom = float(np.mean(ds * ds))
    cov = float(np.mean(ds * (y - y.mean())))
    return float(cov / (denom + eps))


def paired_chi_per_stream(state: Mapping, horizon: int) -> List[float]:
    eps = float(state["epsilon"])
    out = []
    hk = str(int(horizon))
    for rec in state.get("e2_streams", []):
        yp = float(rec["plus"]["X_theta"][hk])
        ym = float(rec["minus"]["X_theta"][hk])
        out.append((yp - ym) / (2.0 * eps))
    return out


def e1_stream_xy_if_available(state: Mapping, horizon: int) -> Tuple[List[float], List[float]] | None:
    """仅当 raw 真正保存了每条 E1 stream 的未来 X_theta 时返回。

    v0.7 原脚本默认 e1_streams 只保存 s_r 与 first_preclip_grad_norm，
    因而通常会返回 None。绝不从聚合 B 反推/伪造 y。
    """
    hk = str(int(horizon))
    ss, yy = [], []
    streams = list(state.get("e1_streams", []))
    if not streams:
        return None
    for rec in streams:
        if "s_r" not in rec:
            return None
        y = None
        if isinstance(rec.get("X_theta"), Mapping) and hk in rec["X_theta"]:
            y = rec["X_theta"][hk]
        elif isinstance(rec.get("observations"), Mapping):
            ob = rec["observations"].get(hk)
            if isinstance(ob, Mapping) and "X_theta" in ob:
                y = ob["X_theta"]
        elif isinstance(rec.get("observations"), list):
            for ob in rec["observations"]:
                if int(ob.get("horizon", -1)) == int(horizon) and "X_theta" in ob:
                    y = ob["X_theta"]
                    break
        if y is None:
            return None
        ss.append(float(rec["s_r"]))
        yy.append(float(y))
    return ss, yy


def balanced_splits_unique(n: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """枚举互补等价去重的 balanced split；固定 index 0 在 A 半边。"""
    if n % 2 != 0 or n < 4:
        raise ValueError("balanced split 需要偶数 n>=4")
    k = n // 2
    out = []
    rest = list(range(1, n))
    for tail in itertools.combinations(rest, k - 1):
        A = (0,) + tuple(tail)
        aset = set(A)
        B = tuple(i for i in range(n) if i not in aset)
        out.append((A, B))
    return out


def spearman_brown(r_half: float) -> float:
    if not math.isfinite(r_half):
        return float("nan")
    den = 1.0 + float(r_half)
    if abs(den) < 1e-12:
        return float("nan")
    return float(2.0 * r_half / den)


def quantiles(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if x.size == 0:
        return {k: float("nan") for k in ("median", "p05", "p25", "p75", "p95", "mean")}
    return {
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
        "mean": float(x.mean()),
    }


def split_half_chi_distribution(states: Sequence[Mapping], horizon: int) -> List[Dict]:
    per_state = [paired_chi_per_stream(s, horizon) for s in states]
    n = len(per_state[0])
    if any(len(x) != n for x in per_state):
        raise RuntimeError("chi stream count 不一致")
    splits = balanced_splits_unique(n)
    rows = []
    for split_id, (A, B) in enumerate(splits):
        aa = [float(np.mean([x[i] for i in A])) for x in per_state]
        bb = [float(np.mean([x[i] for i in B])) for x in per_state]
        rp = pearson(aa, bb)
        rs = spearman(aa, bb)
        rows.append({
            "split_id": split_id,
            "A": list(A), "B": list(B),
            "pearson_half": rp,
            "spearman_half": rs,
            "rho_full_spearman_brown": spearman_brown(rp),
        })
    return rows


def split_half_B_distribution(states: Sequence[Mapping], horizon: int) -> List[Dict] | None:
    pairs = [e1_stream_xy_if_available(s, horizon) for s in states]
    if any(p is None for p in pairs):
        return None
    assert all(p is not None for p in pairs)
    n = len(pairs[0][0])
    if any(len(p[0]) != n or len(p[1]) != n for p in pairs if p is not None):
        raise RuntimeError("E1 stream count 不一致")
    splits = balanced_splits_unique(n)
    rows = []
    for split_id, (A, Bidx) in enumerate(splits):
        bA, bB = [], []
        for ss, yy in pairs:  # type: ignore[misc]
            bA.append(observational_slope([ss[i] for i in A], [yy[i] for i in A]))
            bB.append(observational_slope([ss[i] for i in Bidx], [yy[i] for i in Bidx]))
        rp = pearson(bA, bB)
        rs = spearman(bA, bB)
        rows.append({
            "split_id": split_id,
            "A": list(A), "B": list(Bidx),
            "pearson_half": rp,
            "spearman_half": rs,
            "rho_full_spearman_brown": spearman_brown(rp),
        })
    return rows


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    c, *_ = np.linalg.lstsq(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), rcond=None)
    return c


def seed_fe_slope(rows: Sequence[Mapping], b_key: str = "B", y_key: str = "chi") -> Dict:
    seeds = sorted({int(r["seed"]) for r in rows})
    X, y = [], []
    for r in rows:
        row = [1.0]
        for s in seeds[1:]:
            row.append(1.0 if int(r["seed"]) == s else 0.0)
        row.extend([float(r["state_fraction"]), float(r[b_key])])
        X.append(row); y.append(float(r[y_key]))
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    c = _ols(X, y)
    pred = X @ c
    ssr = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"K": float(c[-1]), "r2": float(1.0 - ssr / sst) if sst > 0 else float("nan")}


def within_seed_detrended_corr(rows: Sequence[Mapping], b_key: str = "B", y_key: str = "chi") -> float:
    rb, ry = [], []
    for seed in sorted({int(r["seed"]) for r in rows}):
        rr = sorted([r for r in rows if int(r["seed"]) == seed], key=lambda x: int(x["state_step"]))
        sf = np.asarray([float(r["state_fraction"]) for r in rr], dtype=np.float64)
        X = np.column_stack([np.ones(sf.size), sf])
        b = np.asarray([float(r[b_key]) for r in rr], dtype=np.float64)
        y = np.asarray([float(r[y_key]) for r in rr], dtype=np.float64)
        rb.extend((b - X @ _ols(X, b)).tolist())
        ry.extend((y - X @ _ols(X, y)).tolist())
    return pearson(rb, ry)


def first_difference_corr(rows: Sequence[Mapping], b_key: str = "B", y_key: str = "chi") -> float:
    db, dy = [], []
    for seed in sorted({int(r["seed"]) for r in rows}):
        rr = sorted([r for r in rows if int(r["seed"]) == seed], key=lambda x: int(x["state_step"]))
        b = np.asarray([float(r[b_key]) for r in rr], dtype=np.float64)
        y = np.asarray([float(r[y_key]) for r in rr], dtype=np.float64)
        db.extend(np.diff(b).tolist()); dy.extend(np.diff(y).tolist())
    return pearson(db, dy)


def pooled_corr(rows: Sequence[Mapping], b_key: str = "B", y_key: str = "chi") -> float:
    return pearson([float(r[b_key]) for r in rows], [float(r[y_key]) for r in rows])


def loso_step_vs_B(rows: Sequence[Mapping], b_key: str = "B", y_key: str = "chi") -> Dict:
    """复刻 v0.7 思路：held seed 下比较 step-only vs step+B。训练 fold 标准化。"""
    seeds = sorted({int(r["seed"]) for r in rows})
    folds = []
    for held in seeds:
        tr = [r for r in rows if int(r["seed"]) != held]
        te = [r for r in rows if int(r["seed"]) == held]
        levels = sorted({int(r["seed"]) for r in tr})
        sfv = np.asarray([float(r["state_fraction"]) for r in tr], dtype=np.float64)
        bv = np.asarray([float(r[b_key]) for r in tr], dtype=np.float64)
        sf_mu, sf_sd = float(sfv.mean()), float(sfv.std(ddof=0)) or 1.0
        b_mu, b_sd = float(bv.mean()), float(bv.std(ddof=0)) or 1.0
        def design(rr, include_b):
            out=[]
            for r in rr:
                row=[1.0]
                for s in levels[1:]: row.append(1.0 if int(r["seed"]) == s else 0.0)
                row.append((float(r["state_fraction"])-sf_mu)/sf_sd)
                if include_b: row.append((float(r[b_key])-b_mu)/b_sd)
                out.append(row)
            return np.asarray(out,dtype=np.float64)
        ytr=np.asarray([float(r[y_key]) for r in tr],dtype=np.float64)
        yte=np.asarray([float(r[y_key]) for r in te],dtype=np.float64)
        X0tr,X0te=design(tr,False),design(te,False)
        X1tr,X1te=design(tr,True),design(te,True)
        c0,c1=_ols(X0tr,ytr),_ols(X1tr,ytr)
        m0=float(np.mean(np.abs(yte-X0te@c0))); m1=float(np.mean(np.abs(yte-X1te@c1)))
        gain=float((m0-m1)/m0) if m0>0 else 0.0
        folds.append({"held_out_seed":held,"step_only_mae":m0,"step_plus_B_mae":m1,"relative_gain":gain})
    gains=np.asarray([f["relative_gain"] for f in folds],dtype=np.float64)
    return {"folds":folds,"mean_gain":float(gains.mean()),"median_gain":float(np.median(gains)),"folds_improved":int((gains>0).sum()),"n_folds":len(folds)}


def build_state_h_rows(states: Sequence[Mapping], horizons: Sequence[int]) -> List[Dict]:
    rows=[]
    for s in states:
        for h in horizons:
            rows.append({
                "seed":int(s["train_seed"]),"state_step":int(s["step"]),"state_fraction":float(s["state_fraction"]),
                "horizon":int(h),"B":float(s["B_theta"][str(h)]),"chi":float(s["chi_theta"][str(h)]),
            })
    return rows


def analysis_summary(rows: Sequence[Mapping], b_key: str = "B", y_key: str = "chi") -> Dict:
    return {
        "pooled_pearson": pooled_corr(rows,b_key,y_key),
        "seed_fe": seed_fe_slope(rows,b_key,y_key),
        "detrended_pearson": within_seed_detrended_corr(rows,b_key,y_key),
        "first_difference_pearson": first_difference_corr(rows,b_key,y_key),
        "loso_step_vs_B": loso_step_vs_B(rows,b_key,y_key),
    }


def inspect_make_amp_source(project_root: str | Path) -> Dict:
    p = Path(project_root).resolve()/"research/src/fr/minimind_v03_runtime.py"
    if not p.exists():
        return {"source_path":str(p),"exists":False,"scaler_enabled_inference":"UNKNOWN"}
    txt=p.read_text(encoding="utf-8",errors="replace")
    i=txt.find("def make_amp")
    snippet=txt[i:i+2200] if i>=0 else ""
    low=snippet.lower()
    inference="UNKNOWN"
    # 保守识别常见写法；无法明确则 UNKNOWN。
    if "gradscaler" in low:
        if ("enabled=(dtype" in low and "float16" in low) or ("enabled = (dtype" in low and "float16" in low):
            inference="BF16_DISABLED_BY_SOURCE"
        elif "enabled=false" in low.replace(" ",""):
            inference="DISABLED_BY_SOURCE"
    return {"source_path":str(p),"exists":True,"scaler_enabled_inference":inference,"make_amp_snippet":snippet[:1800]}


def sampler_source_audit(project_root: str | Path) -> Dict:
    p=Path(project_root).resolve()/"research/src/fr/stage_c_runtime.py"
    if not p.exists(): return {"source_path":str(p),"exists":False,"semantics":"UNKNOWN"}
    txt=p.read_text(encoding="utf-8",errors="replace")
    semantics="UNKNOWN"
    if "make_permutation" in txt and "batchify_indices" in txt:
        semantics="WITHOUT_REPLACEMENT_SINGLE_PERMUTATION_OF_TRAIN_POOL"
    return {"source_path":str(p),"exists":True,"semantics":semantics}


def future_stream_semantics_source(project_root: str | Path) -> Dict:
    p=Path(project_root).resolve()/"research/scripts/e3c_multiseed_theta_v07.py"
    if not p.exists(): return {"source_path":str(p),"exists":False,"natural":"UNKNOWN","causal":"UNKNOWN"}
    txt=p.read_text(encoding="utf-8",errors="replace")
    natural="UNKNOWN"; causal="UNKNOWN"
    if "v07_e1_stream_base_seed" in txt and "batches_from_train_pool" in txt:
        natural="E1_EACH_STREAM_HAS_OWN_STREAM_SEED_AND_BATCH_MANIFEST_THROUGH_HORIZON"
    if "common = dict(" in txt and "b0 = run_theta_branch" in txt and "bp = run_theta_branch" in txt and "bm = run_theta_branch" in txt:
        causal="E2_BASELINE_PLUS_MINUS_SHARE_STREAM_SEED_BATCHES_AND_LR_PATH_WITHIN_STREAM"
    return {"source_path":str(p),"exists":True,"natural":natural,"causal":causal}


def load_state_payload(path: str | Path):
    import torch
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def load_q_payload(path: str | Path):
    import torch
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def optimizer_hparams_from_payload(payload: Mapping) -> Dict:
    opt=payload["optimizer"]
    groups=opt.get("param_groups",[])
    if len(groups)!=1:
        raise RuntimeError(f"A0 direct baseline 暂只支持 1 param_group，实际 {len(groups)}")
    g=groups[0]
    return {
        "betas":tuple(float(x) for x in g.get("betas",(float('nan'),float('nan')))),
        "eps":float(g.get("eps",float('nan'))),
        "weight_decay":float(g.get("weight_decay",0.0)),
        "amsgrad":bool(g.get("amsgrad",False)),
        "maximize":bool(g.get("maximize",False)),
        "param_ids":[int(x) for x in g.get("params",[])],
    }


def _optimizer_step_value(x) -> float:
    try:
        import torch
        if torch.is_tensor(x): return float(x.detach().cpu())
    except Exception: pass
    return float(x)


def compute_mu_q_from_payload(state_payload: Mapping, q_payload: Mapping) -> Dict:
    """按保存顺序将 q 与 AdamW exp_avg_sq 对齐，计算 state-specific mu_q。

    [FR-METRIC] 若 tensor 数量/shape 不能逐项一致，立即 FAIL；绝不猜映射。
    """
    import torch
    qmap=q_payload["q"]
    qitems=list(qmap.items())
    hp=optimizer_hparams_from_payload(state_payload)
    if hp["amsgrad"]:
        raise RuntimeError("A0 frozen baseline 尚未定义 amsgrad 语义")
    ids=hp["param_ids"]
    ost=state_payload["optimizer"].get("state",{})
    if len(ids)!=len(qitems):
        raise RuntimeError(f"q tensor count {len(qitems)} != optimizer param count {len(ids)}")
    steps=[]
    mu=torch.zeros((),dtype=torch.float64)
    qnorm=torch.zeros((),dtype=torch.float64)
    beta2=hp["betas"][1]
    for (name,q),pid in zip(qitems,ids):
        st=ost.get(pid,ost.get(str(pid)))
        if st is None or "exp_avg_sq" not in st or "step" not in st:
            raise RuntimeError(f"optimizer state 缺失: param_id={pid}, q={name}")
        v=st["exp_avg_sq"].detach().double().cpu()
        qq=q.detach().double().cpu()
        if tuple(v.shape)!=tuple(qq.shape):
            raise RuntimeError(f"q/exp_avg_sq shape mismatch at {name}: {tuple(qq.shape)} vs {tuple(v.shape)}")
        step=_optimizer_step_value(st["step"]); steps.append(step)
        bc2=1.0-beta2**step
        if bc2<=0: raise RuntimeError("invalid beta2 bias correction")
        vhat=v/bc2
        D=1.0/(torch.sqrt(vhat)+hp["eps"])
        mu += torch.sum(qq*qq*D)
        qnorm += torch.sum(qq*qq)
    if max(steps)-min(steps)>1e-6:
        raise RuntimeError(f"optimizer per-param step 不一致: min={min(steps)}, max={max(steps)}")
    return {"mu_q":float(mu),"q_norm_sq":float(qnorm),"optimizer_step":float(np.median(steps)),**hp}


def direct_alpha_frozen_preconditioner(*, mu_q: float, beta1: float, optimizer_step: int,
                                       lrs: Sequence[float], weight_decay: float=0.0) -> List[float]:
    """unclipped + frozen-vhat + first-moment direct channel 的 dX_theta/d epsilon。

    recurrence:
      delta m_h = (1-beta1) beta1^(h-1) epsilon q
      delta theta_h = (1-lr_h*wd) delta theta_(h-1)
                      - lr_h * delta m_h/(1-beta1^(t+h)) * D_state

    返回每个 future optimizer step 结束时累计 alpha。
    changing-v、Hessian feedback、activation switching、clip Jacobian 均不在此 baseline。
    """
    x=0.0; out=[]
    for h,lr in enumerate(map(float,lrs),start=1):
        decay=1.0-lr*float(weight_decay)
        coeff=(1.0-beta1)*(beta1**(h-1))/(1.0-beta1**(int(optimizer_step)+h))
        x=decay*x - lr*coeff*float(mu_q)
        out.append(float(x))
    return out


def native_future_lrs(project_root: str | Path, metadata: Mapping, base_lr: float, accum: int, max_h: int) -> List[float]:
    """按 MiniMind 原生 micro-LR 语义取每个 future optimizer step 实际消费的最后 micro LR。"""
    import sys
    root=Path(project_root).resolve()
    minimind=root/"minimind"
    if str(minimind) not in sys.path: sys.path.insert(0,str(minimind))
    from trainer.trainer_utils import get_lr
    gm0=int(metadata["global_micro_step"]); total=int(metadata["total_micro_steps"])
    out=[]
    for h in range(1,int(max_h)+1):
        gm=gm0+h*int(accum)
        out.append(float(get_lr(gm,total,float(base_lr))))
    return out


def state_direct_alpha(project_root: str | Path, formal_state: Mapping, source_cfg: Mapping, horizons: Sequence[int]) -> Dict:
    root=Path(project_root).resolve()
    seed=int(formal_state["train_seed"]); step=int(formal_state["step"])
    report=read_json(root/str(source_cfg["output_dir"])/f"seed_{seed}"/"state_bank_report.json")
    br={int(x["step"]):x for x in report["states"]}[step]
    state_path=resolve_recorded_path(root,br["path"])
    q_path=resolve_recorded_path(root,formal_state["q_path"])
    st=load_state_payload(state_path); qp=load_q_payload(q_path)
    m=compute_mu_q_from_payload(st,qp)
    meta=st.get("metadata",{})
    max_h=max(map(int,horizons)); accum=int(source_cfg["accumulation_steps"])
    lrs=native_future_lrs(root,meta,float(source_cfg["learning_rate"]),accum,max_h)
    alpha=direct_alpha_frozen_preconditioner(mu_q=m["mu_q"],beta1=m["betas"][0],optimizer_step=int(round(m["optimizer_step"])),lrs=lrs,weight_decay=m["weight_decay"])
    return {
        "seed":seed,"state_step":step,"state_fraction":float(formal_state["state_fraction"]),
        "mu_q":m["mu_q"],"q_norm_sq":m["q_norm_sq"],"optimizer_step":m["optimizer_step"],
        "beta1":m["betas"][0],"beta2":m["betas"][1],"eps":m["eps"],"weight_decay":m["weight_decay"],
        "state_path":str(state_path),"q_path":str(q_path),"lrs":lrs,
        "alpha_by_h":{str(h):float(alpha[int(h)-1]) for h in horizons},
        "assumptions":["UNCLIPPED_LOCAL_DIRECT_PATH","FROZEN_VHAT_PRECONDITIONER","NO_FUTURE_GRADIENT_JACOBIAN","NO_HESSIAN_FEEDBACK","NO_ACTIVATION_SWITCHING","INCLUDES_DECOUPLED_WEIGHT_DECAY_PROPAGATION"]
    }


def shape_corr(a: Sequence[float], b: Sequence[float]) -> float:
    aa=np.asarray(a,dtype=np.float64); bb=np.asarray(b,dtype=np.float64)
    # 去幅值：中心化后 correlation；若近常量则 nan。
    return pearson(aa,bb)


def classify_direct_path(original: Mapping, residual: Mapping, median_shape_corr: float, cfg: Mapping) -> Dict:
    ko=abs(float(original["seed_fe"]["K"])); kr=abs(float(residual["seed_fe"]["K"]))
    ro=abs(float(original["detrended_pearson"])); rr=abs(float(residual["detrended_pearson"]))
    kred=1.0-kr/ko if ko>1e-12 else float("nan")
    rred=1.0-rr/ro if ro>1e-12 else float("nan")
    yes=float(cfg.get("yes_fraction_reduction",0.5)); partial=float(cfg.get("partial_fraction_reduction",0.25)); scgood=float(cfg.get("shape_corr_good",0.8))
    if math.isfinite(kred) and math.isfinite(rred) and kred>=yes and rred>=yes and math.isfinite(median_shape_corr) and median_shape_corr>=scgood:
        status="YES"
    elif (math.isfinite(kred) and kred>=partial) or (math.isfinite(rred) and rred>=partial) or (math.isfinite(median_shape_corr) and median_shape_corr>=0.7):
        status="PARTIAL"
    else:
        status="NO"
    return {"status":status,"K_abs_fraction_reduction":kred,"detrended_r_abs_fraction_reduction":rred,"median_shape_corr":median_shape_corr}


def write_csv(path: str | Path, rows: Sequence[Mapping]) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        p.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r.keys():
            if k not in fields: fields.append(k)
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
