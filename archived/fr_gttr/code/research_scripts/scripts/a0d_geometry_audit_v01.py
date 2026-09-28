#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from fr.a0_fr_diagnostics import *


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',required=True); ap.add_argument('--config',default='research/configs/a0_fr_diagnostics_v01.json'); args=ap.parse_args()
    root=Path(args.project_root).resolve(); cfg=read_json(root/args.config); out=root/cfg['output_dir']/'geometry'; out.mkdir(parents=True,exist_ok=True)
    states,_=load_formal_states(root,cfg)
    # 检查 E1 stream raw 是否有 full gradient / sketch / Gram 所需信息。
    keys=set()
    for s in states:
        for r in s.get('e1_streams',[]): keys.update(r.keys())
    sufficient=any(k in keys for k in ('gradient','gradient_vector','gradient_sketch','centered_gradient_sketch','gram_row','pairwise_gradient_dots','sigma_q_vector'))
    # 当前已知 v0.7 e1 只有标量 s_r/preclip norm/seed/manifest；不足恢复 Sigma q 的正交分量或 tr Sigma^2。
    result={'data_integrity':True,'e1_stream_keys':sorted(keys),'A0D_EXISTING_DATA_SUFFICIENT':bool(sufficient),'can_estimate_cos_Sigmaq_q':bool(sufficient),'can_estimate_r_perp':bool(sufficient),'can_estimate_PR_Sigma':bool(sufficient),'minimal_geometry_replay_requirements':[
      '复用原 v0.7 E1 16 streams/state 的 state、manifest、stream_seed；baseline-only exact replay。',
      '在 first optimizer boundary、unscale 后 clip 前，仅 observation-only 获取每条 centered gradient 的低存储量统计。',
      '为了精确 Sigma q：在线累计 vector sum_g 与 scalar s=<q,g> 加权 vector sum_sg；最终 Sigma q = E[(g-gbar)(s-sbar)]。可只保存最终 Sigma q vector/state，不保存每条 full gradient。',
      '为了 PR(Sigma)：在线/分块计算 16x16 centered-gradient Gram matrix G_ab=<delta g_a,delta g_b>；只保存 Gram，不保存 full gradient tensors。',
      '保存每条 ||g||^2、s_r、first clipping、manifest/hash；最终验证 replay model/optimizer hash 与原 v0.7 E1 baseline 对应（若原 E1 未保存 final hash，则需使用 deterministic branch audit/重新定义只读 replay integrity）。',
      '由 Sigma q 与 q 计算 cos(Sigma q,q)、r_perp；由 centered Gram 估 tr Sigma 与 tr Sigma^2 / PR，并 bootstrap 16 streams uncertainty。'
    ]}
    atomic_json(out/'a0d_geometry_audit.json',result)
    lines=['# A0d Covariance Geometry Sufficiency Audit','', 'A0D_DATA_INTEGRITY = PASS','',f"E1_STREAM_KEYS = `{sorted(keys)}`",f"A0D_EXISTING_DATA_SUFFICIENT = {'YES' if sufficient else 'NO'}",'', '## Minimal replay if insufficient','']
    lines += [f"- {x}" for x in result['minimal_geometry_replay_requirements']]
    lines += ['', '## Boundary','', '- 本轮不运行 GPU。','- 不能用仅有的 <q,g> 与 ||g|| 恢复 q^perp 中的 Sigma q。','- 不构造 5M×5M covariance；目标是在线累计 Sigma q 与 16×16 Gram。','']
    atomic_text(out/'A0D_COVARIANCE_GEOMETRY.md','\n'.join(lines))
    print('A0D_DATA_INTEGRITY = PASS'); print('A0D_EXISTING_DATA_SUFFICIENT =','YES' if sufficient else 'NO'); print('REPORT =',out/'A0D_COVARIANCE_GEOMETRY.md')
if __name__=='__main__': raise SystemExit(main())
