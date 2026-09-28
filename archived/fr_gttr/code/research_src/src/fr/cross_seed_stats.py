"""跨 seed 的 FR 映射认证统计。

本模块明确区分：
- training seed：独立科学重复单位；
- state：同一 seed 内的相关纵向观测；
- random stream：某一个 state 内用于估计 B/chi 的局部重复。
"""
from __future__ import annotations
import math
from typing import List, Dict, Sequence

from .mapping_stats import mapping_summary


def _np():
    import numpy as np
    return np


def _pearson(x, y):
    np = _np()
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    xc=x-x.mean(); yc=y-y.mean()
    d=float(np.sqrt(np.sum(xc*xc)*np.sum(yc*yc)))
    return float(np.sum(xc*yc)/d) if d>0 else math.nan


def _rankdata(x):
    np = _np()
    x=np.asarray(x,dtype=np.float64)
    order=np.argsort(x,kind='mergesort')
    ranks=np.empty(len(x),dtype=np.float64)
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:
            j+=1
        rank=0.5*((i+1)+j)
        ranks[order[i:j]]=rank
        i=j
    return ranks


def _spearman(x,y):
    return _pearson(_rankdata(x),_rankdata(y))


def _ols_predict(train_X, train_y, test_X):
    np = _np()
    X=np.asarray(train_X,dtype=np.float64)
    y=np.asarray(train_y,dtype=np.float64)
    Xt=np.asarray(test_X,dtype=np.float64)
    beta,*_=np.linalg.lstsq(X,y,rcond=None)
    return Xt@beta, beta


def residualize_within_seed(rows: List[Dict], value_key: str) -> List[float]:
    """每个 seed 内线性去除 state_fraction 趋势。"""
    np = _np()
    residuals=[None]*len(rows)
    by_seed={}
    for i,r in enumerate(rows):
        by_seed.setdefault(int(r['train_seed']),[]).append((i,r))
    for seed, items in by_seed.items():
        x=np.asarray([r['state_fraction'] for _,r in items],dtype=np.float64)
        y=np.asarray([r[value_key] for _,r in items],dtype=np.float64)
        X=np.column_stack([np.ones_like(x),x])
        beta,*_=np.linalg.lstsq(X,y,rcond=None)
        res=y-X@beta
        for (idx,_),v in zip(items,res):
            residuals[idx]=float(v)
    return residuals


def first_differences(rows: List[Dict], value_key: str) -> List[float]:
    """同 seed 相邻 state 的一阶差分。"""
    out=[]
    by_seed={}
    for r in rows:
        by_seed.setdefault(int(r['train_seed']),[]).append(r)
    for seed, items in by_seed.items():
        items=sorted(items,key=lambda z:int(z['step']))
        for a,b in zip(items[:-1],items[1:]):
            out.append(float(b[value_key]-a[value_key]))
    return out


def pooled_detrended_summary(rows: List[Dict]) -> Dict:
    rb=residualize_within_seed(rows,'B')
    rc=residualize_within_seed(rows,'chi')
    db=first_differences(rows,'B')
    dc=first_differences(rows,'chi')
    return {
        'residual_pearson': _pearson(rb,rc),
        'residual_spearman': _spearman(rb,rc),
        'first_diff_pearson': _pearson(db,dc),
        'first_diff_spearman': _spearman(db,dc),
        'n_residual_points': len(rb),
        'n_first_differences': len(db),
    }


def step_adjusted_pooled(rows: List[Dict]) -> Dict:
    """比较 step-only 与 step+B 的拟合解释力。"""
    np=_np()
    y=np.asarray([r['chi'] for r in rows],dtype=np.float64)
    s=np.asarray([r['state_fraction'] for r in rows],dtype=np.float64)
    b=np.asarray([r['B'] for r in rows],dtype=np.float64)

    X0=np.column_stack([np.ones_like(s),s])
    X1=np.column_stack([np.ones_like(s),s,b])
    p0,b0=_ols_predict(X0,y,X0)
    p1,b1=_ols_predict(X1,y,X1)
    sse0=float(np.sum((y-p0)**2))
    sse1=float(np.sum((y-p1)**2))
    sst=float(np.sum((y-y.mean())**2))
    return {
        'step_only_r2': float(1-sse0/sst) if sst>0 else math.nan,
        'step_plus_B_r2': float(1-sse1/sst) if sst>0 else math.nan,
        'incremental_partial_r2_B': float((sse0-sse1)/sse0) if sse0>0 else math.nan,
        'coef_intercept': float(b1[0]),
        'coef_step': float(b1[1]),
        'coef_B_K': float(b1[2]),
    }


def leave_one_seed_out(rows: List[Dict]) -> Dict:
    """LOSO：比较未见 seed 上 step-only 和 step+B。"""
    np=_np()
    seeds=sorted({int(r['train_seed']) for r in rows})
    folds=[]
    for held in seeds:
        tr=[r for r in rows if int(r['train_seed'])!=held]
        te=[r for r in rows if int(r['train_seed'])==held]

        ytr=np.asarray([r['chi'] for r in tr],dtype=np.float64)
        yte=np.asarray([r['chi'] for r in te],dtype=np.float64)
        str_=np.asarray([r['state_fraction'] for r in tr],dtype=np.float64)
        ste=np.asarray([r['state_fraction'] for r in te],dtype=np.float64)
        btr=np.asarray([r['B'] for r in tr],dtype=np.float64)
        bte=np.asarray([r['B'] for r in te],dtype=np.float64)

        X0tr=np.column_stack([np.ones_like(str_),str_])
        X0te=np.column_stack([np.ones_like(ste),ste])
        X1tr=np.column_stack([np.ones_like(str_),str_,btr])
        X1te=np.column_stack([np.ones_like(ste),ste,bte])

        p0,_=_ols_predict(X0tr,ytr,X0te)
        p1,beta1=_ols_predict(X1tr,ytr,X1te)
        mae0=float(np.mean(np.abs(p0-yte)))
        mae1=float(np.mean(np.abs(p1-yte)))
        improvement=float((mae0-mae1)/mae0) if mae0>0 else math.nan

        folds.append({
            'held_out_seed':held,
            'mae_step_only':mae0,
            'mae_step_plus_B':mae1,
            'relative_mae_improvement':improvement,
            'trained_K_B':float(beta1[2]),
        })

    improvements=np.asarray([f['relative_mae_improvement'] for f in folds],dtype=np.float64)
    return {
        'folds':folds,
        'mean_relative_mae_improvement':float(improvements.mean()),
        'median_relative_mae_improvement':float(np.median(improvements)),
        'folds_improved':int(np.sum(improvements>0)),
        'total_folds':len(folds),
    }


def analyze_horizon(rows: List[Dict]) -> Dict:
    """一个固定 H 的完整跨 seed 分析。"""
    seeds=sorted({int(r['train_seed']) for r in rows})
    per_seed={}
    for seed in seeds:
        rr=[r for r in rows if int(r['train_seed'])==seed]
        per_seed[str(seed)]=mapping_summary(
            [r['B'] for r in rr],
            [r['chi'] for r in rr],
        )
    pooled=mapping_summary([r['B'] for r in rows],[r['chi'] for r in rows])
    return {
        'per_seed':per_seed,
        'pooled_raw':pooled,
        'detrended':pooled_detrended_summary(rows),
        'step_adjusted':step_adjusted_pooled(rows),
        'loso':leave_one_seed_out(rows),
    }
