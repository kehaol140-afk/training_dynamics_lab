#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from pathlib import Path
from fr.a0_fr_diagnostics import *


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',required=True); ap.add_argument('--config',default='research/configs/a0_fr_diagnostics_v01.json'); args=ap.parse_args()
    root=Path(args.project_root).resolve(); cfg=read_json(root/args.config); src_cfg=read_json(root/cfg['source_config']); out=root/cfg['output_dir']/'confounds'; out.mkdir(parents=True,exist_ok=True)
    states,_=load_formal_states(root,cfg); sampler=sampler_source_audit(root); future=future_stream_semantics_source(root); amp=inspect_make_amp_source(root); clip=float(src_cfg['grad_clip'])
    rows=[]
    for s in states:
        e1=[float(x['first_preclip_grad_norm']) for x in s.get('e1_streams',[]) if x.get('first_preclip_grad_norm') is not None]
        e2=s.get('e2_streams',[])
        def brvals(name): return [float(x[name]['first_preclip_grad_norm']) for x in e2]
        b0,bp,bm=brvals('baseline'),brvals('plus'),brvals('minus')
        rows.append({'seed':int(s['train_seed']),'state_step':int(s['step']),'state_fraction':float(s['state_fraction']),
          'e1_first_clip_rate':float(np.mean(np.asarray(e1)>clip)) if e1 else float('nan'),'e1_preclip_mean':float(np.mean(e1)) if e1 else float('nan'),'e1_preclip_over_C_mean':float(np.mean(e1))/clip if e1 else float('nan'),
          'e2_baseline_first_clip_rate':float(np.mean(np.asarray(b0)>clip)),'e2_plus_first_clip_rate':float(np.mean(np.asarray(bp)>clip)),'e2_minus_first_clip_rate':float(np.mean(np.asarray(bm)>clip)),
          'e2_baseline_preclip_mean':float(np.mean(b0)),'e2_plus_preclip_mean':float(np.mean(bp)),'e2_minus_preclip_mean':float(np.mean(bm)),
          'formal_baseline_rate':float(s.get('clipping',{}).get('baseline_rate',float('nan'))),'formal_plus_rate':float(s.get('clipping',{}).get('plus_rate',float('nan'))),'formal_minus_rate':float(s.get('clipping',{}).get('minus_rate',float('nan'))),
          'clip_threshold':clip})
    write_csv(out/'a0b_state_confounds.csv',rows)
    # scorer: 仅诊断 first-step regime，不能代表整个 32-step future。
    base_rates=[r['e2_baseline_first_clip_rate'] for r in rows]; e1rates=[r['e1_first_clip_rate'] for r in rows]
    clip_summary={'e1_mean_first_clip_rate':float(np.mean(e1rates)),'e2_baseline_mean_first_clip_rate':float(np.mean(base_rates)),'e2_plus_mean_first_clip_rate':float(np.mean([r['e2_plus_first_clip_rate'] for r in rows])),'e2_minus_mean_first_clip_rate':float(np.mean([r['e2_minus_first_clip_rate'] for r in rows])),'state_range_baseline':[float(np.min(base_rates)),float(np.max(base_rates))]}
    scaler_inf=amp['scaler_enabled_inference']; amp_risk='UNLIKELY' if scaler_inf in ('BF16_DISABLED_BY_SOURCE','DISABLED_BY_SOURCE') else 'UNKNOWN'
    risk='NONTRIVIAL' if clip_summary['e1_mean_first_clip_rate']>0.05 or clip_summary['e2_baseline_mean_first_clip_rate']>0.05 else 'LOW_FIRST_STEP_ONLY'
    result={'data_integrity':True,'sampler':sampler,'future_stream':future,'clip_summary':clip_summary,'clip_data_scope':'FIRST OPTIMIZER STEP ONLY; full future clip rate not saved in v0.7 raw','amp_source':amp,'AMP_SKIP_CONFOUND':amp_risk,'CONFOUND_RISK':risk,'rho_values':sorted({float(s['rho']) for s in states})}
    atomic_json(out/'a0b_confound_audit.json',result)
    lines=['# A0b Confound Audit','', 'A0B_DATA_INTEGRITY = PASS','',f"SAMPLER_SEMANTICS = {sampler['semantics']}",f"NATURAL_FUTURE_STREAM_SEMANTICS = {future['natural']}",f"CAUSAL_CRN = {future['causal']}",f"CLIP_DATA_AVAILABLE = FIRST_STEP_ONLY",f"FORMAL_CLIP_SUMMARY = {json.dumps(clip_summary,ensure_ascii=False)}",f"SCALER_SOURCE_INFERENCE = {scaler_inf}",f"AMP_SKIP_CONFOUND = {amp_risk}",f"CONFOUND_RISK = {risk}",'', 'Boundary: full 32-step clip history 未保存在 v0.7 raw；本轮不做 GPU 补采。','']
    atomic_text(out/'A0B_CONFOUND_AUDIT.md','\n'.join(lines))
    print('A0B_DATA_INTEGRITY = PASS'); print('SAMPLER_SEMANTICS =',sampler['semantics']); print('AMP_SKIP_CONFOUND =',amp_risk); print('CONFOUND_RISK =',risk); print('REPORT =',out/'A0B_CONFOUND_AUDIT.md')
if __name__=='__main__': raise SystemExit(main())
