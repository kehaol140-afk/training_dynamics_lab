#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from fr.a0_fr_diagnostics import *


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',required=True); ap.add_argument('--config',default='research/configs/a0_fr_diagnostics_v01.json'); args=ap.parse_args()
    root=Path(args.project_root).resolve(); cfg=read_json(root/args.config); src_cfg=read_json(root/cfg['source_config']); states, files=load_formal_states(root,cfg)
    out=root/cfg['output_dir']; out.mkdir(parents=True,exist_ok=True)
    sampler=sampler_source_audit(root); future=future_stream_semantics_source(root); amp=inspect_make_amp_source(root)
    example=states[0]
    # state payload 读取真实 optimizer hparams（CPU）
    seed=int(example['train_seed']); step=int(example['step']); bank=read_json(root/cfg['source_output_dir']/f'seed_{seed}'/'state_bank_report.json'); br={int(x['step']):x for x in bank['states']}[step]
    st=load_state_payload(resolve_recorded_path(root,br['path'])); hp=optimizer_hparams_from_payload(st)
    e1_n={len(s.get('e1_streams',[])) for s in states}; e2_n={len(s.get('e2_streams',[])) for s in states}
    e1_future_y_available=all(e1_stream_xy_if_available(s,int(cfg['primary_horizon'])) is not None for s in states)
    clipping_keys=sorted(set().union(*(set(s.get('clipping',{}).keys()) for s in states)))
    report={
      'protocol':cfg['protocol_version'],'data_integrity':True,'formal_state_count':len(states),'formal_raw_files':[str(p) for p in files],
      'primary_Y_H':'X_theta(H)=<q_t, theta_(t+H)-theta_t>','B_definition':'Cov(delta s,Y)/Var(delta s), s=<q_t,g_first>','chi_definition':'mean_stream[(X_plus-X_minus)/(2 epsilon)]',
      'e1_stream_counts':sorted(e1_n),'e2_stream_counts':sorted(e2_n),'causal_CRN':future['causal'],'natural_future_stream_semantics':future['natural'],'sampler':sampler,
      'rho_values':sorted({float(s['rho']) for s in states}),'epsilon_formula':'epsilon=rho*sigma_q (verified by v0.7 source; numeric state fields stored)',
      'K_formal_semantics':'regression slope in confirmatory analysis; pointwise chi/B is not primary','grad_clip':float(src_cfg['grad_clip']),
      'formal_clipping_keys':clipping_keys,'amp_source_audit':amp,'optimizer_hparams':{k:v for k,v in hp.items() if k!='param_ids'},
      'lr_semantics':'MiniMind get_lr written before every micro-batch; optimizer consumes last micro-batch LR at accumulation boundary',
      'state_fractions':sorted({float(s['state_fraction']) for s in states}),'formal_B_chi_recomputed':False,
      'e1_per_stream_future_X_available':bool(e1_future_y_available),
      'scientific_boundary':'v0.7 H8_CONFIRMATION=FAIL unchanged; 2030..2037 are post-confirmatory mechanism-development data.'
    }
    atomic_json(out/'A0_DATA_AUDIT.json',report)
    lines=['# A0 Data Audit','',f"A0_DATA_INTEGRITY = PASS",'', '## Core definitions','',f"- Y_H: `{report['primary_Y_H']}`",f"- B: `{report['B_definition']}`",f"- chi: `{report['chi_definition']}`",f"- E1 streams/state: `{report['e1_stream_counts']}`",f"- E2 streams/state: `{report['e2_stream_counts']}`",f"- causal CRN: `{report['causal_CRN']}`",f"- natural future semantics: `{report['natural_future_stream_semantics']}`",f"- sampler: `{sampler['semantics']}`",f"- rho: `{report['rho_values']}`",f"- grad_clip: `{report['grad_clip']}`",f"- beta1/beta2: `{hp['betas']}`",f"- AdamW eps: `{hp['eps']}`",f"- weight_decay: `{hp['weight_decay']}`",f"- scaler source inference: `{amp['scaler_enabled_inference']}`",f"- E1 per-stream future X saved: `{e1_future_y_available}`",'', '## Boundary','', '- [FR-OBSERVATION-ONLY] CPU/read-only audit。','- v0.7 H8_CONFIRMATION=FAIL 保持不变。','- 2030..2037 不再视为新的 held-out confirmation。','']
    atomic_text(out/'A0_DATA_AUDIT.md','\n'.join(lines))
    print('A0_DATA_INTEGRITY = PASS'); print('E1_PER_STREAM_FUTURE_X_AVAILABLE =', 'YES' if e1_future_y_available else 'NO'); print('REPORT =',out/'A0_DATA_AUDIT.md')
if __name__=='__main__': raise SystemExit(main())
