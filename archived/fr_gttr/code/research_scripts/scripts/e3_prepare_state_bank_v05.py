#!/usr/bin/env python3
"""建立 v0.5 E3 的 7-state bank。

脚本作用：
- 只训练一条 seed=2026 的原生 AdamW trajectory；
- 保存 step=[46,52,58,64,70,76,82] 的完整 Z_t；
- probe/eval 永远不进入训练 pool；
- step64 若存在 v0.4 MID state，会做 model/optimizer hash 桥接检查。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from fr.full_state import save_full_training_state
from fr.hashing import stable_hash
from fr.minimind_v03_runtime import copy_jsonl_prefix, make_amp, stack_batch
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature
from fr.native_schedule import set_native_micro_lr
from fr.pilot_partition import build_partition, save_partition
from fr.stage_c_runtime import batches_from_train_pool


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
    from trainer.trainer_utils import get_lr, init_model, setup_seed

    out = root/'outputs/e3_mapping/v05'
    states_dir = out/'states'; states_dir.mkdir(parents=True, exist_ok=True)
    data_path = out/'e3_subset.jsonl'
    copy_jsonl_prefix(minimind/'dataset/pretrain_t2t_mini.jsonl', data_path, int(cfg['subset_lines']))

    setup_seed(int(cfg['state_seed']))
    mmcfg = MiniMindConfig(
        hidden_size=int(cfg['hidden_size']),
        num_hidden_layers=int(cfg['num_hidden_layers']),
        max_seq_len=int(cfg['max_seq_len']), use_moe=False,
    )
    model, tokenizer = init_model(mmcfg, from_weight='none', tokenizer_path=str(minimind/'model'), device=args.device)
    model.train()
    ds = PretrainDataset(str(data_path), tokenizer, max_length=int(cfg['max_seq_len']))

    partition = build_partition(len(ds), int(cfg['partition_seed']), int(cfg['probe_pool_size']), int(cfg['eval_pool_size']))
    save_partition(out/'partition_v05.json', partition)

    autocast_ctx, scaler, _ = make_amp(torch, args.device, cfg['dtype'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']))
    assert_native_adamw(optimizer)
    optimizer.zero_grad(set_to_none=True)

    accum = int(cfg['accumulation_steps'])
    total_opt = int(cfg['planned_total_optimizer_steps'])
    total_micro = total_opt * accum
    state_steps = sorted(map(int, cfg['e3_state_steps']))
    if max(state_steps) + max(map(int, cfg['horizons'])) > total_opt:
        raise RuntimeError('state_steps + max(H) 超出预注册总预算')

    max_micro = max(state_steps) * accum
    batches = batches_from_train_pool(
        partition['train_indices'], int(cfg['batch_size']), int(cfg['state_seed']) + 1704, max_micro
    )

    state_records = []
    used = set()
    for micro_idx, batch_idx in enumerate(batches, start=1):
        used.update(batch_idx)
        set_native_micro_lr(optimizer, get_lr, micro_idx, total_micro, float(cfg['learning_rate']))
        x, y = stack_batch(ds, batch_idx, args.device)
        with autocast_ctx:
            res = model(x, labels=y)
            loss = (res.loss + res.aux_loss) / accum
        scaler.scale(loss).backward()

        if micro_idx % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['grad_clip']))
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            opt_step = micro_idx // accum

            if opt_step in state_steps:
                path = states_dir/f'state_step_{opt_step:03d}.pt'
                state = save_full_training_state(
                    path, model, optimizer, scaler,
                    metadata={
                        'protocol': 'E3_V05_STATE_BANK',
                        'state_optimizer_step': opt_step,
                        'state_fraction': opt_step/total_opt,
                        'global_optimizer_step': opt_step,
                        'global_micro_step': micro_idx,
                        'planned_total_optimizer_steps': total_opt,
                        'total_micro_steps': total_micro,
                        'partition_sha256': partition['sha256'],
                        'optimizer_signature': optimizer_signature(optimizer),
                        'config': cfg,
                    },
                )
                state_records.append({
                    'step': opt_step,
                    'path': str(path),
                    'model_hash': stable_hash(state['model']),
                    'optimizer_hash': stable_hash(state['optimizer']),
                })

    if used & set(partition['probe_indices']) or used & set(partition['eval_indices']):
        raise RuntimeError('state bank 训练误用了 probe/eval')

    # v0.4 step64 桥接：只作为 reproducibility check。
    bridge = {'v04_state_exists': False}
    old = root/'outputs/stage_c_v04/mid_state_v04.pt'
    if old.exists():
        bridge['v04_state_exists'] = True
        old_payload = torch.load(old, map_location='cpu', weights_only=False)
        new64 = next(x for x in state_records if x['step'] == 64)
        bridge.update({
            'model_hash_equal': stable_hash(old_payload['model']) == new64['model_hash'],
            'optimizer_hash_equal': stable_hash(old_payload['optimizer']) == new64['optimizer_hash'],
        })
        if not bridge['model_hash_equal'] or not bridge['optimizer_hash_equal']:
            raise RuntimeError('v0.5 state64 与 v0.4 MID state 不一致，停止 E3')

    report = {
        'technical_pass': len(state_records) == len(state_steps),
        'state_steps': state_steps,
        'partition_sha256': partition['sha256'],
        'probe_overlap': len(used & set(partition['probe_indices'])),
        'eval_overlap': len(used & set(partition['eval_indices'])),
        'bridge_v04': bridge,
        'states': state_records,
    }
    (out/'state_bank_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('E3_V05_STATE_BANK_PASS')
    print('state_steps=', state_steps)
    print('probe_overlap=', report['probe_overlap'])
    print('eval_overlap=', report['eval_overlap'])
    print('bridge_v04=', bridge)
    print('report=', out/'state_bank_report.json')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
