#!/usr/bin/env python3
"""E3 v0.5：多个 state 上的 B_theta -> chi_theta 映射。

主检验：FULL_MODEL_Q, rho=0.25, H=1/8/32。
每个 state 是一个映射点；stream 仅用于 state 内估计 B 和 chi。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from fr.direction import normalize_direction, direction_norm
from fr.full_state import load_full_training_state
from fr.mapping_stats import mapping_summary
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch
from fr.native_adamw_guard import assert_native_adamw
from fr.pilot_partition import load_partition
from fr.response_stats import observational_slope, paired_linearity_summary
from fr.stage_c_runtime import batches_from_train_pool
from fr.stream_rng import derive_stream_seed
from fr.theta_protocol import run_theta_branch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', required=True)
    ap.add_argument('--config', default='research/configs/e3_theta_v05.json')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    cfg = json.loads((root/args.config).read_text(encoding='utf-8'))
    minimind = root/'minimind'; sys.path.insert(0, str(minimind))

    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model

    out = root/'outputs/e3_mapping/v05'
    bank = json.loads((out/'state_bank_report.json').read_text(encoding='utf-8'))
    if not bank.get('technical_pass'):
        raise SystemExit('state bank 未 PASS')
    partition = load_partition(out/'partition_v05.json')

    mmcfg = MiniMindConfig(
        hidden_size=int(cfg['hidden_size']), num_hidden_layers=int(cfg['num_hidden_layers']),
        max_seq_len=int(cfg['max_seq_len']), use_moe=False,
    )
    model, tokenizer = init_model(mmcfg, from_weight='none', tokenizer_path=str(minimind/'model'), device=args.device)
    ds = PretrainDataset(str(out/'e3_subset.jsonl'), tokenizer, max_length=int(cfg['max_seq_len']))
    autocast_ctx, scaler, _ = make_amp(torch, args.device, cfg['dtype'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']))
    assert_native_adamw(optimizer)
    named = [(n,p) for n,p in raw_model(model).named_parameters() if p.requires_grad]

    horizons = sorted(map(int, cfg['horizons']))
    max_h = max(horizons)
    accum = int(cfg['accumulation_steps'])
    rho = float(cfg['e3_primary_rho'])
    e1_R = int(cfg['e3_e1_streams'])
    e2_R = int(cfg['e3_e2_streams'])

    state_results = []
    rho0_noop_checked = False

    for state_rec in bank['states']:
        step = int(state_rec['step'])
        state_path = Path(state_rec['path'])
        meta = load_full_training_state(state_path, model, optimizer, scaler)
        payload = torch.load(state_path, map_location='cpu', weights_only=False)
        reference_theta = payload['model']

        # --------------------------------------------------------
        # q_t：每个 state 用同一 train-probe 数据重新构造。
        # --------------------------------------------------------
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        x_probe, y_probe = stack_batch(ds, partition['probe_indices'], args.device)
        with autocast_ctx:
            res = model(x_probe, labels=y_probe)
            probe_loss = res.loss + res.aux_loss
        params = [p for _,p in named]
        grads = torch.autograd.grad(probe_loss, params, allow_unused=True)
        q_raw = {
            name: (torch.zeros_like(p.detach()) if g is None else g.detach().clone())
            for (name,p),g in zip(named,grads)
        }
        q, probe_grad_norm = normalize_direction(q_raw)
        q_norm = direction_norm(q)
        optimizer.zero_grad(set_to_none=True)
        model.train()

        # --------------------------------------------------------
        # E1：自然 stream -> B_theta
        # --------------------------------------------------------
        s_values = []
        e1_rows = []
        state_e1_base = derive_stream_seed(int(cfg['e3_e1_stream_base_seed']), step)
        for r in range(e1_R):
            stream_seed = derive_stream_seed(state_e1_base, r)
            batch_seed = derive_stream_seed(stream_seed, 404)
            batches = batches_from_train_pool(
                partition['train_indices'], int(cfg['batch_size']), batch_seed, max_h*accum
            )
            br = run_theta_branch(
                torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, ds=ds, state_path=state_path,
                reference_theta=reference_theta, q=q, named=named, batches=batches,
                stream_seed=stream_seed, sign=0, epsilon=0.0,
                horizons=horizons, accumulation_steps=accum, batch_size=int(cfg['batch_size']),
                device=args.device, grad_clip=float(cfg['grad_clip']), base_lr=float(cfg['learning_rate']),
                global_micro_step=int(meta['global_micro_step']), total_micro_steps=int(meta['total_micro_steps']),
                get_lr=get_lr, stack_batch=stack_batch, measure_s_first=True,
            )
            s_values.append(float(br['s']))
            e1_rows.append(br)

        B_by_h = {}
        for h in horizons:
            y = [x['observations'][str(h)]['X_theta'] for x in e1_rows]
            B_by_h[str(h)] = observational_slope(s_values, y)['B']

        # sigma_q 是本 state 的自然尺度，epsilon 也 state-specific。
        import numpy as np
        sigma_q = float(np.std(np.asarray(s_values, dtype=np.float64), ddof=1))
        if sigma_q <= 0:
            raise RuntimeError(f'step={step} sigma_q 无效')
        epsilon = rho * sigma_q

        # --------------------------------------------------------
        # E2：paired ±epsilon -> chi_theta
        # --------------------------------------------------------
        e2_rows = []
        state_e2_base = derive_stream_seed(int(cfg['e3_e2_stream_base_seed']), step)
        for r in range(e2_R):
            stream_seed = derive_stream_seed(state_e2_base, r)
            batch_seed = derive_stream_seed(stream_seed, 404)
            batches = batches_from_train_pool(
                partition['train_indices'], int(cfg['batch_size']), batch_seed, max_h*accum
            )
            common = dict(
                torch=torch, model=model, optimizer=optimizer, scaler=scaler,
                autocast_ctx=autocast_ctx, ds=ds, state_path=state_path,
                reference_theta=reference_theta, q=q, named=named, batches=batches,
                stream_seed=stream_seed, horizons=horizons, accumulation_steps=accum,
                batch_size=int(cfg['batch_size']), device=args.device,
                grad_clip=float(cfg['grad_clip']), base_lr=float(cfg['learning_rate']),
                global_micro_step=int(meta['global_micro_step']), total_micro_steps=int(meta['total_micro_steps']),
                get_lr=get_lr, stack_batch=stack_batch, measure_s_first=False,
            )
            b0 = run_theta_branch(sign=0, epsilon=0.0, **common)
            bp = run_theta_branch(sign=+1, epsilon=epsilon, **common)
            bm = run_theta_branch(sign=-1, epsilon=epsilon, **common)
            if not (b0['batch_manifest_hash'] == bp['batch_manifest_hash'] == bm['batch_manifest_hash']):
                raise RuntimeError(f'step={step}, stream={r} paired batch manifest 不一致')
            e2_rows.append({'baseline': b0, 'plus': bp, 'minus': bm})

            # 只在全实验第一条 stream 做 rho=0 no-op，避免重复浪费算力。
            if not rho0_noop_checked:
                bz = run_theta_branch(sign=+1, epsilon=0.0, **common)
                if not (
                    b0['model_hash_final'] == bz['model_hash_final']
                    and b0['optimizer_hash_final'] == bz['optimizer_hash_final']
                    and b0['observations'] == bz['observations']
                ):
                    raise RuntimeError('E3 rho=0 no-op 自检失败')
                rho0_noop_checked = True

        chi_by_h = {}; odd_by_h = {}
        for h in horizons:
            hk = str(h)
            y0 = [x['baseline']['observations'][hk]['X_theta'] for x in e2_rows]
            yp = [x['plus']['observations'][hk]['X_theta'] for x in e2_rows]
            ym = [x['minus']['observations'][hk]['X_theta'] for x in e2_rows]
            sm = paired_linearity_summary(y0, yp, ym, epsilon)
            chi_by_h[hk] = sm['chi']
            odd_by_h[hk] = sm['odd_error']

        clip = {
            'baseline_rate': sum(x['baseline']['first_clipped'] for x in e2_rows)/e2_R,
            'plus_rate': sum(x['plus']['first_clipped'] for x in e2_rows)/e2_R,
            'minus_rate': sum(x['minus']['first_clipped'] for x in e2_rows)/e2_R,
        }

        row = {
            'step': step,
            'state_fraction': meta['state_fraction'],
            'q_norm': q_norm,
            'probe_grad_norm': probe_grad_norm,
            'sigma_q': sigma_q,
            'epsilon': epsilon,
            'B_theta': B_by_h,
            'chi_theta': chi_by_h,
            'odd_error_theta': odd_by_h,
            'clipping': clip,
        }
        state_results.append(row)
        print(f"state={step} sigma_q={sigma_q:.6g} epsilon={epsilon:.6g} B={B_by_h} chi={chi_by_h}")

    # ------------------------------------------------------------
    # state-level mapping：每个 H 独立统计，绝不把 stream 当成独立 state。
    # ------------------------------------------------------------
    mappings = {}
    for h in horizons:
        hk = str(h)
        B = [x['B_theta'][hk] for x in state_results]
        chi = [x['chi_theta'][hk] for x in state_results]
        mappings[hk] = mapping_summary(B, chi)

    report = {
        'protocol': 'E3_V05_THETA_STATE_MAPPING',
        'technical_pass': True,
        'scientific_mapping_pass': None,
        'rho0_noop_pass': rho0_noop_checked,
        'primary_rho': rho,
        'horizons': horizons,
        'n_states': len(state_results),
        'e1_streams_per_state': e1_R,
        'e2_streams_per_state': e2_R,
        'states': state_results,
        'mappings': mappings,
        'boundaries': [
            '7 个 state 来自同一 seed/同一轨迹，不能当作 7 个独立训练 seed。',
            'E3 主检验仅 theta；mean/tail 未在本版扩展。',
            'scientific_mapping_pass 故意不自动判定，需人工审核。',
        ],
    }
    (out/'e3_theta_mapping_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# E3 v0.5：B_theta -> chi_theta 状态级映射', '',
        '- 技术执行：**PASS**',
        f'- rho=0 no-op：**{rho0_noop_checked}**',
        f'- states：`{len(state_results)}`',
        f'- primary rho：`{rho}`',
        f'- E1 streams/state：`{e1_R}`',
        f'- E2 streams/state：`{e2_R}`',
        '- 科学映射判定：**未自动判定**', '',
        '## 各 state', ''
    ]
    for x in state_results:
        lines += [
            f"### step={x['step']} ({x['state_fraction']:.3f} of pilot)",
            f"- sigma_q: `{x['sigma_q']:.12g}`",
            f"- epsilon: `{x['epsilon']:.12g}`",
            f"- clipping b/p/m: `{x['clipping']['baseline_rate']:.3f}` / `{x['clipping']['plus_rate']:.3f}` / `{x['clipping']['minus_rate']:.3f}`",
        ]
        for h in horizons:
            hk=str(h)
            lines.append(
                f"- H={h}: B_theta=`{x['B_theta'][hk]:.12g}`, chi_theta=`{x['chi_theta'][hk]:.12g}`, odd=`{x['odd_error_theta'][hk]:.6g}`"
            )
        lines.append('')

    lines += ['## 映射统计', '']
    for h in horizons:
        m = mappings[str(h)]
        lines += [
            f'### H={h}',
            f"- Pearson r: `{m['pearson_r']:.12g}`",
            f"- Spearman rho: `{m['spearman_rho']:.12g}`",
            f"- OLS: chi = `{m['intercept_a']:.12g}` + `{m['slope_K']:.12g}` * B",
            f"- R2: `{m['r2']:.12g}`",
            f"- sign agreement: `{m['sign_agreement']:.6f}`",
            f"- median(chi/B): `{m['ratio_chi_over_B_median']:.12g}`",
            f"- MAD(chi/B): `{m['ratio_chi_over_B_mad']:.12g}`",
            f"- LOO MAE: `{m['loo_mae']:.12g}`",
            ''
        ]
    (out/'e3_theta_mapping_report.md').write_text('\n'.join(lines), encoding='utf-8')

    print('E3_V05_THETA_MAPPING_TECHNICAL_PASS')
    print('rho0_noop_pass=', rho0_noop_checked)
    for h in horizons:
        m=mappings[str(h)]
        print(f"H={h} pearson={m['pearson_r']:.6g} spearman={m['spearman_rho']:.6g} R2={m['r2']:.6g} sign={m['sign_agreement']:.3f} K={m['slope_K']:.6g}")
    print('report=', out/'e3_theta_mapping_report.md')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
