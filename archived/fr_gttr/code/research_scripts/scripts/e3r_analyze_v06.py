#!/usr/bin/env python3
"""CPU-only 跨 seed 认证分析。"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from fr.cross_seed_stats import analyze_horizon


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root',required=True)
    ap.add_argument('--config',default='research/configs/e3r_cross_seed_v06.json')
    args=ap.parse_args()

    root=Path(args.project_root).resolve()
    cfg=json.loads((root/args.config).read_text(encoding='utf-8'))
    seeds=list(map(int,cfg['replication_train_seeds']))
    horizons=sorted(map(int,cfg['horizons']))
    out=root/'outputs/e3_replication/v06'

    raw=[]
    for seed in seeds:
        p=out/f'seed_{seed}/theta_mapping_raw.json'
        if not p.exists():
            raise SystemExit(f'缺少 seed {seed} mapping result')
        x=json.loads(p.read_text(encoding='utf-8'))
        if not x.get('technical_pass') or not x.get('rho0_noop_pass'):
            raise SystemExit(f'seed {seed} technical/no-op 未 PASS')
        raw.append(x)

    analysis={}
    flat_by_h={str(h):[] for h in horizons}
    for x in raw:
        seed=int(x['train_seed'])
        for st in x['states']:
            for h in horizons:
                hk=str(h)
                flat_by_h[hk].append({
                    'train_seed':seed,
                    'step':int(st['step']),
                    'state_fraction':float(st['state_fraction']),
                    'B':float(st['B_theta'][hk]),
                    'chi':float(st['chi_theta'][hk]),
                    'odd_error':float(st['odd_error_theta'][hk]),
                })

    for h in horizons:
        analysis[str(h)]=analyze_horizon(flat_by_h[str(h)])

    report={
        'protocol':'E3R_V06_CROSS_SEED_ANALYSIS',
        'technical_pass':True,
        'scientific_replication_pass':None,
        'seeds':seeds,
        'n_training_seeds':len(seeds),
        'states_per_seed':len(cfg['e3_state_steps']),
        'analysis':analysis,
        'boundaries':[
            '4 个 training seed 才是独立科学重复单位。',
            'pooled 28 state 仅作描述，不能替代 seed-level 复现。',
            'scientific_replication_pass 故意不自动判定。',
        ],
    }
    (out/'e3r_cross_seed_analysis.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'
    )

    lines=[
        '# v0.6 跨 seed FR 认证', '',
        f'- training seeds: `{seeds}`',
        f'- states/seed: `{len(cfg["e3_state_steps"])}`',
        '- 科学结论：**未自动判定**', '',
    ]

    for h in horizons:
        a=analysis[str(h)]
        lines += [f'## H={h}', '', '### 每 seed 独立映射']
        for seed in seeds:
            m=a['per_seed'][str(seed)]
            lines.append(
                f"- seed {seed}: Pearson=`{m['pearson_r']:.6g}`, "
                f"Spearman=`{m['spearman_rho']:.6g}`, "
                f"R2=`{m['r2']:.6g}`, K=`{m['slope_K']:.6g}`, "
                f"sign=`{m['sign_agreement']:.3f}`"
            )

        p=a['pooled_raw']
        d=a['detrended']
        s=a['step_adjusted']
        l=a['loso']
        lines += [
            '',
            '### Pooled raw（仅描述）',
            f"- Pearson: `{p['pearson_r']:.6g}`",
            f"- Spearman: `{p['spearman_rho']:.6g}`",
            f"- R2: `{p['r2']:.6g}`",
            f"- K: `{p['slope_K']:.6g}`",
            '',
            '### Within-seed 去训练趋势',
            f"- residual Pearson: `{d['residual_pearson']:.6g}`",
            f"- residual Spearman: `{d['residual_spearman']:.6g}`",
            f"- first-difference Pearson: `{d['first_diff_pearson']:.6g}`",
            f"- first-difference Spearman: `{d['first_diff_spearman']:.6g}`",
            '',
            '### step-adjusted pooled',
            f"- step-only R2: `{s['step_only_r2']:.6g}`",
            f"- step+B R2: `{s['step_plus_B_r2']:.6g}`",
            f"- incremental partial R2(B): `{s['incremental_partial_r2_B']:.6g}`",
            f"- adjusted K_B: `{s['coef_B_K']:.6g}`",
            '',
            '### Leave-One-Seed-Out',
            f"- folds improved: `{l['folds_improved']}/{l['total_folds']}`",
            f"- mean relative MAE improvement: `{l['mean_relative_mae_improvement']:.6g}`",
            f"- median relative MAE improvement: `{l['median_relative_mae_improvement']:.6g}`",
        ]
        for fold in l['folds']:
            lines.append(
                f"- hold seed {fold['held_out_seed']}: "
                f"step-only MAE=`{fold['mae_step_only']:.6g}`, "
                f"step+B MAE=`{fold['mae_step_plus_B']:.6g}`, "
                f"improvement=`{fold['relative_mae_improvement']:.6g}`, "
                f"train K_B=`{fold['trained_K_B']:.6g}`"
            )
        lines.append('')

    md=out/'e3r_cross_seed_analysis.md'
    md.write_text('\n'.join(lines),encoding='utf-8')

    print('E3R_V06_CROSS_SEED_ANALYSIS_DONE')
    for h in horizons:
        a=analysis[str(h)]
        d=a['detrended']; l=a['loso']; s=a['step_adjusted']
        print(
            f"H={h} residual_r={d['residual_pearson']:.6g} "
            f"diff_r={d['first_diff_pearson']:.6g} "
            f"partialR2={s['incremental_partial_r2_B']:.6g} "
            f"LOSO_improved={l['folds_improved']}/{l['total_folds']} "
            f"LOSO_mean_impr={l['mean_relative_mae_improvement']:.6g}"
        )
    print('report=',md)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
