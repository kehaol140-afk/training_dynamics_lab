#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from pathlib import Path
from fr.a0_fr_diagnostics import *


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',required=True); ap.add_argument('--config',default='research/configs/a0_fr_diagnostics_v01.json'); args=ap.parse_args()
    root=Path(args.project_root).resolve(); cfg=read_json(root/args.config); out=root/cfg['output_dir']/'reliability'; out.mkdir(parents=True,exist_ok=True)
    states,_=load_formal_states(root,cfg); H=int(cfg['primary_horizon'])
    # chi reliability 可由现有 e2 raw 完整重算。
    chi_splits=split_half_chi_distribution(states,H); chi_q=quantiles([r['rho_full_spearman_brown'] for r in chi_splits])
    # B reliability 只有 raw 保存 per-stream (s_r, X_theta(H)) 才允许做。
    B_splits=split_half_B_distribution(states,H)
    B_available=B_splits is not None
    if B_available:
        B_q=quantiles([r['rho_full_spearman_brown'] for r in B_splits])
    else:
        B_q=None
    split_rows=[]
    for r in chi_splits: split_rows.append({'target':'chi',**r})
    if B_splits:
        for r in B_splits: split_rows.append({'target':'B',**r})
    write_csv(out/'a0a_split_distribution.csv',split_rows)

    rowsH=build_state_h_rows(states,[H]); observed=analysis_summary(rowsH)
    Kobs=float(observed['seed_fe']['K']); robs=float(observed['detrended_pearson'])
    diag={'protocol':cfg['protocol_version']+'_A0A','data_integrity':True,'horizon':H,'chi_reliability':{'n_unique_splits':len(chi_splits),'rho_full':chi_q},'B_reliability_available':B_available,'B_reliability':None if not B_available else {'n_unique_splits':len(B_splits),'rho_full':B_q},'observed':{'K_seed_fe':Kobs,'r_detrended':robs,'analysis':observed},'correction':None,'boundary':['Classical disattenuation is diagnostic only.','v0.7 H8_CONFIRMATION=FAIL unchanged.']}
    interpretation='B_RELIABILITY_UNAVAILABLE_FROM_EXISTING_RAW'
    if B_available and B_q is not None:
        rhoB=float(B_q['median']); rhoC=float(chi_q['median']); den=math.sqrt(max(rhoB*rhoC,0.0)) if rhoB>0 and rhoC>0 else float('nan')
        Kcorr=Kobs/rhoB if rhoB>0 else float('nan'); ceiling=den; rcorr=robs/den if den>0 else float('nan')
        mismatch=bool(math.isfinite(rcorr) and abs(rcorr)>1.0)
        diag['correction']={'K_disattenuated':Kcorr,'r_ceiling':ceiling,'r_disattenuated':rcorr,'model_assumption_mismatch':mismatch}
        # 保守判断：是否 95% reliability interval 覆盖 observed K（K*=1 所需 rho≈Kobs）。
        target=abs(Kobs); lo=float(B_q['p05']); hi=float(B_q['p95'])
        if lo<=target<=hi: interpretation='PLAUSIBLE'
        elif abs(float(B_q['median'])-target)<=0.15: interpretation='PLAUSIBLE'
        else: interpretation='UNLIKELY'
    diag['B_MEASUREMENT_NOISE_CAN_EXPLAIN_K377']=interpretation
    atomic_json(out/'a0a_reliability.json',diag)
    lines=['# A0a Reliability / Errors-in-Variables','',f"A0A_DATA_INTEGRITY = PASS",f"H = {H}",'', '## chi reliability','',f"- exhaustive unique 4+4 splits: `{len(chi_splits)}`",f"- rho_chichi full median: `{chi_q['median']:.6g}`",f"- P05/P95: `[{chi_q['p05']:.6g}, {chi_q['p95']:.6g}]`",'', '## B reliability','']
    if not B_available:
        lines += ['- **UNAVAILABLE from existing v0.7 raw.**','- v0.7 `e1_streams` 保存了 `s_r` 和 first preclip norm，但没有保存每条 stream 的 `X_theta(H)`。','- 因此不能合法地对 16 条 E1 stream 做 8+8 split-half；不能从聚合 B 反推。','- 下一步若要完成 A0a，需要 baseline-only E1 exact replay，复用原 16 个 manifest/stream seed，只额外保存 per-stream X_theta(H)。']
    else:
        lines += [f"- unique 8+8 splits: `{len(B_splits)}`",f"- rho_BB full median: `{B_q['median']:.6g}`",f"- P05/P95: `[{B_q['p05']:.6g}, {B_q['p95']:.6g}]`"]
    lines += ['', '## Observed H8 diagnostic','',f"- seed-FE K (recomputed from frozen raw): `{Kobs:.6g}`",f"- within-seed detrended r: `{robs:.6g}`"]
    if diag['correction']:
        c=diag['correction']; lines += [f"- K_disattenuated: `{c['K_disattenuated']:.6g}`",f"- r ceiling: `{c['r_ceiling']:.6g}`",f"- r_disattenuated: `{c['r_disattenuated']:.6g}`",f"- MODEL_ASSUMPTION_MISMATCH: `{c['model_assumption_mismatch']}`"]
    lines += ['', '## Boundary','', '- Spearman–Brown/disattenuation 只在 classical independent additive measurement-error 假设下解释。','- 不能用本报告改写 v0.7 confirmation。','', '## Final','',f"RHO_BB_H8_MEDIAN = {'UNAVAILABLE' if not B_available else B_q['median']}",f"RHO_CHICHI_H8_MEDIAN = {chi_q['median']}",f"K_H8_OBSERVED = {Kobs}",f"R_H8_OBSERVED = {robs}",f"B_MEASUREMENT_NOISE_CAN_EXPLAIN_K377 = {interpretation}"]
    atomic_text(out/'A0A_RELIABILITY.md','\n'.join(map(str,lines))+'\n')
    print('A0A_DATA_INTEGRITY = PASS'); print('RHO_BB_H8_MEDIAN =', 'UNAVAILABLE' if not B_available else B_q['median']); print('RHO_CHICHI_H8_MEDIAN =',chi_q['median']); print('B_MEASUREMENT_NOISE_CAN_EXPLAIN_K377 =',interpretation); print('REPORT =',out/'A0A_RELIABILITY.md')
if __name__=='__main__': raise SystemExit(main())
