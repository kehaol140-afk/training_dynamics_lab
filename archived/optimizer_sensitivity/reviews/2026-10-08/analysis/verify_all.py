#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ADAMW audit 20261008 — light CPU recompute of F4-chain + P4 key numbers.
Read-only: only reads CSV/JSON from the two project dirs. Outputs a check report.
"""
import json, math, sys
import numpy as np
import pandas as pd

OPS = "/home/b205/projects/training_dynamics_lab/optimizer_sensitivity"
CON = "/home/b205/projects/training_dynamics_lab/continuation_adaptation"
OUT = []
def log(s): OUT.append(s); print(s, flush=True)
def approx(a, b, tol=1e-3): return abs(a-b) <= tol*max(1.0, abs(b))
def check(name, cond, detail=""):
    log(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

# ============ F4A ============
log("== F4A phase_map ==")
pm = pd.read_csv(f"{OPS}/outputs/f4a_phase_map/phase_map_long.csv")
log(f"phase_map_long rows={len(pm)} states={pm.groupby(['scale','seed','step']).ngroups} H values={sorted(pm.H.unique())}")
pm32 = pm[pm.H==32].copy()
check("F4A n states at H=32 == 74", len(pm32)==74, f"n={len(pm32)}")

# median R_curv_32 per scale x step
med = pm32.groupby(['scale','step']).R_curv.median().unstack(0)
log("median R_curv_32 by scale x step:\n"+med.to_string())
expect = {('v07_5m',46):25.99,('v07_5m',52):16.77,('v07_5m',58):4.49,('v07_5m',64):2.61,
          ('v07_5m',70):2.52,('v07_5m',76):2.04,('v07_5m',82):1.47,
          ('g1a_26m',26):9.13,('g1a_26m',64):1.85,('g1a_26m',102):0.94,
          ('g1b_100m',26):743.95,('g1b_100m',64):6.31,('g1b_100m',102):0.92}
bad = {k:(med.loc[k[1],k[0]],v) for k,v in expect.items() if not approx(med.loc[k[1],k[0]],v,0.02)}
check("F4A per-step STANDARD median R_curv_32 matches canonical long CSV", not bad, f"mismatch={bad}")
log("NOTE report table values (28.6/17.8/4.65/2.68/2.64/2.16/1.63 for 5M) correspond to median_high convention; see check below")

# lambda_SO > 0 all 74
lam = pm32.set_index(['scale','seed','step']).lambda_SO
check("F4A lambda_SO_32 > 0 at all 74 states", (lam>0).all(), f"min={lam.min():.5f} max={lam.max():.4f}")
l100 = pm32[pm32.scale=='g1b_100m'].groupby('step').lambda_SO.median()
check("F4A 100M lambda medians 0.335/0.107/0.014", approx(l100.loc[26],0.335,0.02) and approx(l100.loc[64],0.107,0.02) and approx(l100.loc[102],0.014,0.05), str(dict(l100.round(4))))
l5 = pm32[pm32.scale=='v07_5m'].groupby('step').lambda_SO.median()
check("F4A 5M lambda 0.143->0.038", approx(l5.loc[46],0.143,0.02) and approx(l5.loc[82],0.038,0.05), str(dict(l5.round(4))))

# seed spread 100M step26
s = pm32[(pm32.scale=='g1b_100m')&(pm32.step==26)].set_index('seed').R_curv.sort_values()
log(f"100M step26 R_curv by seed: {dict(s.round(2))}")
check("F4A 100M step26 spread 86/744/1.6e7", approx(s.loc[2042],86.3,0.02) and approx(s.loc[2043],744,0.02) and approx(s.loc[2044],1.58e7,0.02))

# indicator Spearman vs log10 R_curv_32
from scipy.stats import spearmanr
lr = np.log10(pm32.R_curv)
for col,exp in [('A_theta',-0.813),('clip_ratio',0.692),('A_m',0.513),('A_v',0.482),('v_floor_fraction',0.195),('K_lr',-0.129)]:
    r,_ = spearmanr(pm32[col], lr)
    check(f"F4A Spearman {col} ~ {exp}", approx(r,exp,0.01), f"got {r:.4f}")

# exploratory models
em = pd.read_csv(f"{OPS}/outputs/f4a_phase_map/exploratory_models.csv")
log("exploratory_models.csv:\n"+em.to_string())
best = em.sort_values('r2_loso',ascending=False).iloc[0]
check("F4A exploratory best is C with R2~0.306", best['model']=='C_scale_fixed_p' and approx(best['r2_loso'],0.306,0.01), f"{best['model']} {best['r2_loso']:.4f}")
# report table used median_high convention (upper middle for even n) for per-step 5M medians
import statistics
s5m = pm32[pm32.scale=='v07_5m']
medhigh = s5m.groupby('step').R_curv.apply(statistics.median_high)
log(f"5M per-step median_high: {dict(medhigh.round(2))}  (report: 28.6/17.8/4.65/2.68/2.64/2.16/1.63)")
check("F4A report 5M table == median_high convention", all(approx(medhigh.loc[st],v,0.01) for st,v in [(46,28.6),(52,17.8),(58,4.65),(64,2.68),(70,2.64),(76,2.16),(82,1.63)]))

# healing
hl = pd.read_csv(f"{OPS}/outputs/f4a_phase_map/healing_intervals.csv")
log(f"healing_intervals rows={len(hl)} cols={list(hl.columns)}")
log(hl.to_string())
stat = hl.status.value_counts().to_dict() if 'status' in hl else {}
check("F4A healing 13 INTERVAL_CENSORED + 1 NOT_HEALED", stat.get('INTERVAL_CENSORED',0)==13 and stat.get('NOT_HEALED_IN_COVERAGE',0)==1, str(stat))
row2032 = hl[hl.seed==2032] if 'seed' in hl else hl[hl.astype(str).apply(lambda r: r.str.contains('2032').any(),axis=1)]
log(f"seed 2032 row:\n{row2032.to_string()}")
seq2032 = pm32[(pm32.scale=='v07_5m')&(pm32.seed==2032)].sort_values('step').R_curv.values
log(f"5M seed2032 R_curv_32 seq: {np.round(seq2032,2)}")
check("F4A 2032 relapse present (31.6 at p~.55, last 2.9/6.3)", approx(seq2032[4],31.6,0.01) and approx(seq2032[5],2.9,0.05) and approx(seq2032[6],6.3,0.05))

# summary.json
fsum = json.load(open(f"{OPS}/outputs/f4a_phase_map/summary.json"))
log("f4a summary.json: "+json.dumps(fsum)[:600])

# ============ F4A.1 ============
log("\n== F4A.1 matched coverage ==")
mm = pd.read_csv(f"{OPS}/outputs/f4a1_matched_coverage/matched_phase_map.csv")
log(f"matched_phase_map rows={len(mm)} by scale: {mm.scale.value_counts().to_dict()}")
check("F4A.1 matched_phase_map rows == 42 (matched cells; report claims 90 rows — doc-level discrepancy)", len(mm)==42, f"{len(mm)}")
msum = pd.read_csv(f"{OPS}/outputs/f4a1_matched_coverage/matched_summary.csv")
log("matched_summary.csv:\n"+msum.to_string())
piv = mm[mm.step.isin([26,64,102])].groupby(['scale','step']).agg(
    R_med=('R_curv_32','median'), R_min=('R_curv_32','min'), R_max=('R_curv_32','max'),
    lam_med=('lambda_SO_32','median')).reset_index()
log(piv.to_string())
exp_matched = {('v07_5m',26):(11.9,0.084),('g1a_26m',26):(9.1,0.134),('g1b_100m',26):(744,0.335),
               ('v07_5m',64):(2.61,0.064),('g1a_26m',64):(1.85,0.055),('g1b_100m',64):(6.31,0.107),
               ('v07_5m',102):(1.07,0.016),('g1a_26m',102):(0.94,0.0087),('g1b_100m',102):(0.92,0.0139)}
badm = []
for k,(er,el) in exp_matched.items():
    row = piv[(piv.scale==k[0])&(piv.step==k[1])]
    if len(row)!=1 or not approx(row.R_med.iloc[0],er,0.02) or not approx(row.lam_med.iloc[0],el,0.03):
        badm.append((k, None if len(row)!=1 else (row.R_med.iloc[0],row.lam_med.iloc[0]), (er,el)))
check("F4A.1 matched medians table", not badm, f"mismatch={badm}")
# Spearman on EARLY medians
e = piv[piv.step==26].set_index('scale').loc[['v07_5m','g1a_26m','g1b_100m']]
r_lam,_ = spearmanr([6.6975,7.4262,7.6189], e.lam_med)  # logN approx from dataset
check("F4A.1 EARLY lambda_SO Spearman(logN)=+1", r_lam==1.0, f"{r_lam}")
r_R,_ = spearmanr([6.6975,7.4262,7.6189], e.R_med)
check("F4A.1 EARLY R_curv Spearman nonmonotone (0.5)", r_R==0.5, f"{r_R}")
late = piv[piv.step==102]
check("F4A.1 LATE all R_curv_32<3 (24 obs)", (mm[mm.step==102].R_curv_32<3).all(), f"max={mm[mm.step==102].R_curv_32.max():.3f}")
# canary
ce = json.load(open(f"{OPS}/outputs/f4a1_matched_coverage/canary_envelope.json"))
cr = json.load(open(f"{OPS}/outputs/f4a_phase_map/raw/v07_5m/seed_2030/step_64_canary/metrics.json")) if False else None
import glob
cand = glob.glob(f"{OPS}/outputs/f4a_phase_map/raw/v07_5m/seed_2030/*64*")
log(f"canary raw candidates: {cand}")
can = None
for c in cand:
    mf = f"{c}/metrics.json"
    if __import__('os').path.exists(mf):
        can = json.load(open(mf)); log(f"canary metrics keys sample: {list(can.keys())[:20]}")
        # find R_curv_32 / lambda_SO_32 / loss
        def find(o, keys, p=''):
            res={}
            if isinstance(o,dict):
                for k,v in o.items():
                    if any(t in k.lower() for t in keys) and isinstance(v,(int,float)): res[p+k]=v
                    res.update(find(v,keys,p+k+'.'))
            return res
        flat = find(can, ['r_curv_32','lambda_so_32','loss'])
        log(f"canary flat: {flat}")
check("F4A.1 canary envelope R range [1.873,9.272]", approx(ce['R_curv_32']['min'],1.8732,1e-3) and approx(ce['R_curv_32']['max'],9.2725,1e-3))
# canary metrics direct from raw
cm = json.load(open(f"{OPS}/outputs/f4a1_matched_coverage/canary_raw/v07_5m/seed_2030/step_64/metrics.json"))
cR = cm['G']['32']['R_curv']; clam = cm['G']['32']['lambda_SO']; cc2 = json.load(open(f"{OPS}/outputs/f4a1_matched_coverage/canary_raw/v07_5m/seed_2030/step_64/config.json"))
import csv
with open(f"{OPS}/outputs/f4a1_matched_coverage/canary_raw/v07_5m/seed_2030/step_64/trajectory.csv") as f:
    r1 = next(csv.DictReader(f))
closs = float(r1['loss'])
check("F4A.1 canary R_curv 6.03 in envelope", approx(cR,6.0305,1e-3) and ce['R_curv_32']['min']<=cR<=ce['R_curv_32']['max'], f"{cR:.4f}")
check("F4A.1 canary lambda 0.0887 in envelope", approx(clam,0.08872,1e-3) and ce['lambda_SO_32']['min']<=clam<=ce['lambda_SO_32']['max'], f"{clam:.5f}")
check("F4A.1 canary stream_id 4090 (=4074+16)", cc2.get('stream_id')==4090, str(cc2.get('stream_id')))
vio = closs - ce['loss_h1']['max']
check("F4A.1 canary loss 7.4437, violation 0.080 < IQR 0.111 (prereg allowed)", approx(closs,7.4437,1e-3) and vio < 7.363274574279785-7.066202163696289, f"loss {closs:.4f} vio {vio:.4f}")
log(f"canary_envelope: {json.dumps(ce)[:400]}")

# ============ F4A.5 ============
log("\n== F4A.5 modeling ==")
ds = pd.read_csv(f"{OPS}/outputs/f4a5_modeling/dataset.csv")
lp = pd.read_csv(f"{OPS}/outputs/f4a5_modeling/loso_predictions.csv")
mc = pd.read_csv(f"{OPS}/outputs/f4a5_modeling/model_comparison.csv")
check("F4A.5 dataset 90 rows", len(ds)==90, f"{len(ds)}")
check("F4A.5 loso_predictions 90 rows", len(lp)==90, f"{len(lp)}")
check("F4A.5 dataset/loso keys align", (ds[['scale','seed','step']].values==lp[['scale','seed','step']].values).all() or (set(map(tuple,ds[['scale','seed','step']].values))==set(map(tuple,lp[['scale','seed','step']].values))))
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
recomp = {}
for m in ['M0','M1','M2','M3','M4','M5']:
    pred = lp[f'pred_{m}']
    mae = mean_absolute_error(lp.y_R, pred)
    rmse = math.sqrt(mean_squared_error(lp.y_R, pred))
    sp,_ = spearmanr(lp.y_R, pred)
    r2 = r2_score(lp.y_R, pred)
    recomp[m] = dict(MAE=mae,RMSE=rmse,Spearman=sp,R2=r2)
    row = mc[mc.model==m].iloc[0]
    ok = approx(mae,row.mae,5e-4) and approx(rmse,row.rmse,5e-4) and approx(sp,row.spearman,5e-4) and approx(r2,row.r2,5e-4)
    check(f"F4A.5 {m} pooled-OOF recompute == model_comparison", ok, f"recomp {mae:.4f}/{rmse:.4f}/{sp:.4f}/{r2:.4f} vs csv {row.mae}/{row.rmse}/{row.spearman}/{row.r2}")
# macro fold
seeds = sorted(lp.seed.unique())
macro = {}
for m in ['M0','M1','M2','M3','M4','M5']:
    f_mae,f_rmse,f_sp = [],[],[]
    for sd in seeds:
        sub = lp[lp.seed==sd]
        f_mae.append(mean_absolute_error(sub.y_R, sub[f'pred_{m}']))
        f_rmse.append(math.sqrt(mean_squared_error(sub.y_R, sub[f'pred_{m}'])))
        if len(sub)>2:
            sp,_ = spearmanr(sub.y_R, sub[f'pred_{m}']); f_sp.append(sp)
    macro[m] = dict(MAE=np.mean(f_mae), RMSE=np.mean(f_rmse), Spearman=np.mean(f_sp), MEDIAN_FOLD_MAE=float(np.median(f_mae)))
rec = json.load(open(f"{OPS}/outputs/f4_consistency/f4a5_metric_recompute.json"))
m0 = macro['M0']
check("F4A.5 M0 MACRO MAE 0.4509", approx(m0['MAE'], rec['M0']['MACRO_FOLD']['mae'], 5e-4), f"{m0['MAE']:.4f} vs {rec['M0']['MACRO_FOLD']['mae']}")
check("F4A.5 M0 MACRO Spearman 0.8987", approx(m0['Spearman'], rec['M0']['MACRO_FOLD']['spearman'], 5e-4), f"{m0['Spearman']:.4f}")
check("F4A.5 M0 MEDIAN_FOLD_MAE vs recompute.json", approx(m0['MEDIAN_FOLD_MAE'], rec['M0']['MEDIAN_FOLD_MAE'], 5e-4), f"{m0['MEDIAN_FOLD_MAE']:.4f} vs json {rec['M0']['MEDIAN_FOLD_MAE']}")
log(f"NOTE erratum doc states MEDIAN_FOLD_MAE 0.4219 but recompute.json says {rec['M0']['MEDIAN_FOLD_MAE']} — checking against my value {m0['MEDIAN_FOLD_MAE']:.4f}")
check("F4A.5 M4==M5 predictions identical", (lp.pred_M4==lp.pred_M5).all())
# variance decomposition
vd = pd.read_csv(f"{OPS}/outputs/f4a5_modeling/variance_decomposition.csv")
log("variance_decomposition:\n"+vd.to_string())
check("F4A.5 seed var frac 0.418", any(approx(v,0.418,0.001) for v in vd.select_dtypes('number').values.flatten()), vd.to_string().replace('\n',' | '))
# gate summary
s55 = json.load(open(f"{OPS}/outputs/f4a5_modeling/summary.json"))
log("f4a5 summary.json: "+json.dumps(s55)[:700])
# ablation
ab = pd.read_csv(f"{OPS}/outputs/f4a5_modeling/predictor_ablation.csv")
log("predictor_ablation:\n"+ab.to_string())
mx = ab.delta.abs().max()
check("F4A.5 ablation deltas small vs M3-M2 gap (0.0546); report claims all |dMAE|<=0.01 — CSV shows otherwise", mx<0.055, f"max|delta|={mx:.4f} (report claim of <=0.01 NOT matched by artifact — registered)")

# ============ F4B ============
log("\n== F4B SAU ==")
s5 = json.load(open(f"{OPS}/outputs/f4b_sau/summary.json"))
s5d = json.load(open(f"{OPS}/outputs/f4b_sau/delta_0p05/summary.json"))
check("F4B top summary.json == delta_0p05/summary.json", s5==s5d)
pmr = pd.read_csv(f"{OPS}/outputs/f4b_sau/delta_0p05/per_method_regret.csv")
log("delta_0p05 per_method_regret:\n"+pmr.to_string())
for _,r in pmr.iterrows():
    ref = s5['per_method'][r['method']]
    ok = approx(r['mean'],ref['mean'],5e-3) and approx(r['median_norm_regret'],ref['median'],5e-4) and approx(r['harmful_rate'],ref['harmful_rate'],5e-3) and approx(r['oracle_acc'],ref['oracle_acc'],5e-3)
    check(f"F4B per_method_regret {r['method']} == summary", ok, f"csv {r['mean']:.4f}/{r['harmful_rate']:.4f} vs json {ref['mean']:.4f}/{ref['harmful_rate']:.4f}")
check("F4B SAU == FIXED1 (mean/median/harmful/oracle/H)", s5['per_method']['SAU']==s5['per_method']['FIXED1'] if s5['per_method']['SAU']==s5['per_method']['FIXED1'] else approx(s5['per_method']['SAU']['mean'],s5['per_method']['FIXED1']['mean'],1e-12))
check("F4B gate A false (median tie), B-E true", s5['primary_gate']=={'A_regret_median':False,'B_harmful':True,'C_dominant':True,'D_neutral':True,'E_compute':True} and s5['primary_pass']==False and s5['verdict']=='NO_METHOD_GAIN')
# per-cell SAU recompute from selected_horizons + action_outcomes
sh = pd.read_csv(f"{OPS}/outputs/f4b_sau/delta_0p05/selected_horizons.csv")
ao = pd.read_csv(f"{OPS}/outputs/f4b_sau/delta_0p05/action_outcomes.csv")
mg = sh.merge(ao, on=['scale','seed','step'], suffixes=('','_ao'))
check("F4B all SAU H_safe==1", (mg.H_safe==1).all(), f"H set {sorted(mg.H_safe.unique())}")
Lmap = {'-0.05':'L_minus','0':'L_zero','0.05':'L_plus'}
mg['L_chosen'] = [r[Lmap[str(r.action)]] for _,r in mg.iterrows()]
mg['regret_rc'] = mg.L_chosen - mg[['L_minus','L_zero','L_plus']].min(axis=1)
mg['regret_norm_rc'] = mg.regret_rc/(mg['range']+1e-30)
mg['harmful_rc'] = mg.L_chosen > mg.L_zero
check("F4B SAU per-cell recompute: mean norm regret 0.1035", approx(mg.regret_norm_rc.mean(), s5['per_method']['SAU']['mean'], 5e-3), f"{mg.regret_norm_rc.mean():.4f}")
check("F4B SAU harmful rate 0.119", approx(mg.harmful_rc.mean(), s5['per_method']['SAU']['harmful_rate'], 5e-3), f"{mg.harmful_rc.mean():.4f}")
check("F4B SAU median norm regret 0", mg.regret_norm_rc.median()==0)
# delta 0.02
s2 = json.load(open(f"{OPS}/outputs/f4b_sau/summary_delta0.02.json"))
check("F4B d0.02 NO_METHOD_GAIN; FIXED32 mean 0.132 best among fixed", s2['verdict']=='NO_METHOD_GAIN' and approx(s2['per_method']['FIXED32']['mean'],0.1317,0.01) and approx(s2['per_method']['SAU']['mean'],0.181,0.01), f"SAU {s2['per_method']['SAU']['mean']:.4f} FIXED32 {s2['per_method']['FIXED32']['mean']:.4f}")
# fold budgets all (8,4,0.1)
check("F4B all 11 folds selected budget (8,4,0.1)", all(v['budget']==[8,4,0.1] for v in s5['fold_hp'].values()))

# ============ F4B postmortem ============
log("\n== F4B postmortem ==")
ps = json.load(open(f"{OPS}/outputs/f4b_postmortem/summary.json"))
uh = pd.read_csv(f"{OPS}/outputs/f4b_postmortem/utility_by_horizon.csv")
hs = pd.read_csv(f"{OPS}/outputs/f4b_postmortem/horizon_sufficiency.csv")
oo = pd.read_csv(f"{OPS}/outputs/f4b_postmortem/online_observability.csv")
log(f"utility_by_horizon rows={len(uh)} cols={list(uh.columns)}")
check("F4B-postmortem utility_by_horizon 504 rows (2x42x6H)", len(uh)==504)
f5 = ps['frontier'][0]; f2 = ps['frontier'][1]
check("F4B-post d.05 median H_suff=1 frac<=2 0.9286", f5['median_H_suff_eps05']==1.0 and approx(f5['frac_H_suff_le_2'],0.928571,5e-4))
check("F4B-post d.02 median H_suff=1 frac<=2 0.9048, H_sep=1", f2['median_H_suff_eps05']==1.0 and approx(f2['frac_H_suff_le_2'],0.904762,5e-4) and f2['H_sep']==1)
check("F4B-post d.05 H_sep NOT_OBSERVED, offline AUROC 0.8367", f5['H_sep']=='NOT_OBSERVED' and approx(f5['offline_harmful32']['auroc'],0.836734,5e-4))
check("F4B-post d.05 online h2 AUROC 0.685", approx(f5['online_by_h']['2']['auroc'],0.685185,5e-4))
check("F4B-post d.02 online h1 AUROC 0.790 balacc 0.673", approx(f2['online_by_h']['1']['auroc'],0.790441,5e-4) and approx(f2['online_by_h']['1']['balacc'],0.672794,5e-4))
check("F4B-post verdict", ps['verdict']=='OBSERVABILITY_GAP_SUPPORTED')

# ============ F4C ============
log("\n== F4C mining ==")
ms = json.load(open(f"{OPS}/outputs/f4c_mining/mining_summary.json"))
d5b = {int(k):v for k,v in ms['C2'][0]['mean_nregret_by_H'].items()}
d2b = {int(k):v for k,v in ms['C2'][1]['mean_nregret_by_H'].items()}
pmr5 = pmr.set_index('method')
diff5 = pmr5.loc['FIXED2','mean']-pmr5.loc['FIXED1','mean']
pmr2 = pd.read_csv(f"{OPS}/outputs/f4b_sau/delta_0p02/per_method_regret.csv").set_index('method')
diff2 = pmr2.loc['FIXED2','mean']-pmr2.loc['FIXED1','mean']
check("F4C H2-H1 d.05 = +0.0512", approx(diff5,0.0512,0.005), f"{diff5:.4f}")
check("F4C H2-H1 d.02 = -0.0746", approx(diff2,-0.0746,0.005), f"{diff2:.4f}")
check("F4C H_best<=2 frac .05 93% / .02 88%", approx(ms['C2'][0]['frac_Hbest_le2'],0.928571,5e-4) and approx(ms['C2'][1]['frac_Hbest_le2'],0.880952,5e-4))
check("F4C F2.5 H_best==1 all 8 deltas x 3 cells", all(v==1 for c in ms['C3_Hbest'].values() for v in c.values()))
# bootstrap CI from utility_by_horizon (h_min=0 semantics — approximation note)
u5 = uh[(uh.delta==0.05) if 'delta' in uh.columns else uh.iloc[:,0]==0.05]
log(f"utility_by_horizon columns check: {list(uh.columns)}; delta values {uh['delta'].unique() if 'delta' in uh else None}")
if 'delta' in uh.columns:
    for dl in [0.05,0.02]:
        sub = uh[uh.delta==dl]
        p = sub.pivot_table(index=['scale','seed','step'], columns='H', values='nregret')
        d2 = (p[2]-p[1]).dropna().rename('diff').reset_index()
        rng = np.random.default_rng(42)
        seeds = sorted(d2.seed.unique())
        dgs = {s: grp['diff'].values for s,grp in d2.groupby('seed')}
        boots=[]
        for _ in range(10000):
            pick = rng.choice(seeds, size=len(seeds), replace=True)
            boots.append(np.concatenate([dgs[s] for s in pick]).mean())
        lo,hi = np.percentile(boots,[2.5,97.5])
        log(f"F4C bootstrap (h_min=0 utility variant) delta={dl}: H2-H1 mean {d2['diff'].mean():+.4f} CI95 [{lo:+.4f},{hi:+.4f}] (report: .05 +0.0512 [+0.000,+0.113]; .02 -0.0746 [-0.158,+0.000])")

# ============ P4 continuation ============
log("\n== P4 sealed table recompute ==")
tb = pd.read_csv(f"{CON}/outputs/p4_full/stage2/sealed_evaluation_table.csv")
check("P4 sealed table 280 rows", len(tb)==280, f"{len(tb)}")
check("P4 arms per cell = 5", tb.groupby('cell_id').arm.nunique().eq(5).all())
check("P4 56 cells x 5 arms", tb.cell_id.nunique()==56 and set(tb.arm.unique())>={'A_best','B','C1','C2','D'})
pv = tb.pivot_table(index='cell_id', columns='arm', values='nll_mean')
check("P4 all 280 nll present", pv.notna().all().all())
for arm in ['C1','C2','B','D']:
    d = (pv[arm]-pv['A_best'])
    log(f"P4 {arm}-A_best: mean {d.mean():+.6f} median {d.median():+.6f} seeds_improved {(d<0).groupby([c.split('_')[0] for c in pv.index]).any().sum() if False else ''}")
    per_seed = d.groupby(pv.index.map(lambda c: c.split('_')[0])).mean()
    log(f"  per-seed mean: {dict(per_seed.round(4))}")
    rng = np.random.default_rng(77080001)
    seeds = sorted(per_seed.index)
    boots=[]
    for _ in range(10000):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        boots.append(np.concatenate([d[pv.index.map(lambda c: c.split('_')[0])==s].values for s in pick]).mean())
    lo,hi = np.percentile(boots,[2.5,97.5])
    log(f"  seed-cluster bootstrap CI95 [{lo:.4f},{hi:.4f}]  (stage3 report to compare)")
ar = json.load(open(f"{CON}/outputs/p4_full/stage3/analysis_report.json"))
for arm in ['C1','C2']:
    p = ar['primary'][arm]['paired_diff']
    log(f"stage3 {arm}: mean {p['aggregate_mean_diff']:.6f} CI [{p['cluster_bootstrap_ci']['ci_low']:.4f},{p['cluster_bootstrap_ci']['ci_high']:.4f}] p={p['permutation']['p_value_one_sided']} seeds_improved={p['seeds_improved']}/{p['n_seeds']}")
sec = ar.get('secondary',{})
log("secondary keys: "+json.dumps(sec)[:800])

with open("/home/b205/projects/training_dynamics_lab/audits/ADAMW_RESEARCH_REVIEW_UPDATED_20261008_051201UTC/analysis/verify_out.txt","w") as f:
    f.write("\n".join(OUT))
print("WROTE verify_out.txt", flush=True)
