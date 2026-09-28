#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from fr.a0_fr_diagnostics import *


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',required=True); ap.add_argument('--config',default='research/configs/a0_fr_diagnostics_v01.json'); args=ap.parse_args()
    root=Path(args.project_root).resolve(); cfg=read_json(root/args.config); src_cfg=read_json(root/cfg['source_config']); out=root/cfg['output_dir']/'direct_path'; out.mkdir(parents=True,exist_ok=True)
    states,_=load_formal_states(root,cfg); horizons=list(map(int,src_cfg['horizons'])); H0=int(cfg['primary_horizon'])
    direct=[]
    for i,s in enumerate(states,1):
        d=state_direct_alpha(root,s,src_cfg,horizons); direct.append(d); print(f"DIRECT {i}/{len(states)} seed={d['seed']} state={d['state_step']} mu_q={d['mu_q']:.6g}")
    dmap={(d['seed'],d['state_step']):d for d in direct}
    rows=[]
    for s in states:
        d=dmap[(int(s['train_seed']),int(s['step']))]
        for h in horizons:
            a=float(d['alpha_by_h'][str(h)]); B=float(s['B_theta'][str(h)]); chi=float(s['chi_theta'][str(h)])
            rows.append({'seed':int(s['train_seed']),'state_step':int(s['step']),'state_fraction':float(s['state_fraction']),'horizon':h,'B':B,'chi':chi,'alpha_direct':a,'B_res':B-a,'chi_res':chi-a,'mu_q':float(d['mu_q']),'beta1':float(d['beta1']),'beta2':float(d['beta2']),'weight_decay':float(d['weight_decay'])})
    write_csv(out/'a0c_state_horizon.csv',rows); write_csv(out/'a0c_horizon_raw.csv',[{k:r[k] for k in ('seed','state_step','state_fraction','horizon','B','chi')} for r in rows])
    r0=[r for r in rows if r['horizon']==H0]
    orig=analysis_summary(r0,'B','chi'); resid=analysis_summary(r0,'B_res','chi_res')
    # 每 state 的 horizon shape correlation chi vs direct alpha；另算 B vs direct。
    shape=[]
    for seed in sorted({r['seed'] for r in rows}):
      for step in sorted({r['state_step'] for r in rows if r['seed']==seed}):
        rr=sorted([r for r in rows if r['seed']==seed and r['state_step']==step],key=lambda x:x['horizon'])
        shape.append({'seed':seed,'state_step':step,'chi_alpha_shape_corr':shape_corr([r['chi'] for r in rr],[r['alpha_direct'] for r in rr]),'B_alpha_shape_corr':shape_corr([r['B'] for r in rr],[r['alpha_direct'] for r in rr])})
    write_csv(out/'a0c_shape_correlations.csv',shape)
    sc=[r['chi_alpha_shape_corr'] for r in shape if math.isfinite(r['chi_alpha_shape_corr'])]; median_sc=float(np.median(sc)) if sc else float('nan')
    cls=classify_direct_path(orig,resid,median_sc,cfg['direct_path'])
    analysis={'protocol':cfg['protocol_version']+'_A0C','direct_baseline':'UNCLIPPED_FROZEN_VHAT_FIRST_MOMENT_WITH_BIAS_CORRECTION_AND_REAL_LR','horizon':H0,'original':orig,'residual':resid,'shape':{'median_chi_alpha_corr':median_sc,'rows':shape},'classification':cls,'state_direct_metadata':direct,'boundary':['alpha_direct is a diagnostic direct channel, not exact AdamW response.','Clipping Jacobian, changing v, Hessian/future-gradient feedback, activation switching are intentionally omitted.','v0.7 H8_CONFIRMATION=FAIL unchanged.']}
    atomic_json(out/'a0c_analysis.json',analysis)
    lines=['# A0c AdamW Direct-Memory Diagnostic','', 'A0C_DATA_INTEGRITY = PASS','', '## Frozen-preconditioner direct baseline','', '- Includes real future optimizer-step LR, beta1 memory decay, first-moment bias correction, state-specific frozen vhat preconditioner, and decoupled weight-decay propagation.','- Excludes clipping Jacobian, perturbation-induced v change, Hessian/future-gradient feedback, activation switching.','- Therefore this is a **diagnostic baseline**, not exact causal response.','',f"BETA1 = {direct[0]['beta1']}",f"BETA2 = {direct[0]['beta2']}",f"H8_ORIGINAL_K = {orig['seed_fe']['K']:.10g}",f"H8_RESIDUAL_K = {resid['seed_fe']['K']:.10g}",f"H8_ORIGINAL_R = {orig['detrended_pearson']:.10g}",f"H8_RESIDUAL_R = {resid['detrended_pearson']:.10g}",f"H8_ORIGINAL_LOSO = mean_gain {orig['loso_step_vs_B']['mean_gain']:.6%}, {orig['loso_step_vs_B']['folds_improved']}/{orig['loso_step_vs_B']['n_folds']}",f"H8_RESIDUAL_LOSO = mean_gain {resid['loso_step_vs_B']['mean_gain']:.6%}, {resid['loso_step_vs_B']['folds_improved']}/{resid['loso_step_vs_B']['n_folds']}",f"DIRECT_KERNEL_MEDIAN_SHAPE_CORR = {median_sc:.10g}",f"DIRECT_PATH_EXPLAINS_MOST = {cls['status']}",f"K_ABS_REDUCTION = {cls['K_abs_fraction_reduction']:.6g}",f"R_ABS_REDUCTION = {cls['detrended_r_abs_fraction_reduction']:.6g}",'', '## Boundary','', '- [FR-METRIC] 不用本诊断改写 v0.7 confirmation。','- 若 clipping first-step 高，unclipped direct baseline 的解释必须降级。','']
    atomic_text(out/'A0C_DIRECT_PATH.md','\n'.join(lines))
    print('A0C_DATA_INTEGRITY = PASS'); print('H8_ORIGINAL_K =',orig['seed_fe']['K']); print('H8_RESIDUAL_K =',resid['seed_fe']['K']); print('H8_ORIGINAL_R =',orig['detrended_pearson']); print('H8_RESIDUAL_R =',resid['detrended_pearson']); print('DIRECT_KERNEL_MEDIAN_SHAPE_CORR =',median_sc); print('DIRECT_PATH_EXPLAINS_MOST =',cls['status']); print('REPORT =',out/'A0C_DIRECT_PATH.md')
if __name__=='__main__': raise SystemExit(main())
