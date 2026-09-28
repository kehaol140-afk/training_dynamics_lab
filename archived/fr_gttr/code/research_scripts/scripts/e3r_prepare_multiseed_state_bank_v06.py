#!/usr/bin/env python3
"""为 4 个独立训练 seed 建立 7-state bank。

每个 seed：
- 独立模型初始化；
- 独立训练 batch trajectory；
- 同一固定 probe/eval partition；
- 保存 step 46/52/58/64/70/76/82 的完整 Z_t。

支持重复运行：已完整 PASS 的 seed 默认跳过。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from fr.full_state import save_full_training_state
from fr.hashing import stable_hash
from fr.minimind_v03_runtime import copy_jsonl_prefix, make_amp, stack_batch
from fr.native_adamw_guard import assert_native_adamw, optimizer_signature
from fr.native_schedule import set_native_micro_lr
from fr.pilot_partition import build_partition, save_partition, load_partition
from fr.stage_c_runtime import batches_from_train_pool


def parse_seeds(text, defaults):
    if not text:
        return list(map(int,defaults))
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root',required=True)
    ap.add_argument('--config',default='research/configs/e3r_cross_seed_v06.json')
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--seeds',default='')
    ap.add_argument('--force',action='store_true')
    args=ap.parse_args()

    root=Path(args.project_root).resolve()
    cfg=json.loads((root/args.config).read_text(encoding='utf-8'))
    seeds=parse_seeds(args.seeds,cfg['replication_train_seeds'])
    minimind=root/'minimind'; sys.path.insert(0,str(minimind))

    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model, setup_seed

    out=root/'outputs/e3_replication/v06'
    out.mkdir(parents=True,exist_ok=True)
    data_path=out/'e3r_subset.jsonl'
    copy_jsonl_prefix(minimind/'dataset/pretrain_t2t_mini.jsonl',data_path,int(cfg['subset_lines']))

    # partition 与 training seed 无关，四个 seed 共用完全相同的 probe/eval 定义。
    part_path=out/'partition_v06.json'
    if part_path.exists():
        partition=load_partition(part_path)
    else:
        # 只需加载一次 dataset 来得到长度。
        setup_seed(int(seeds[0]))
        mmcfg0=MiniMindConfig(
            hidden_size=int(cfg['hidden_size']),
            num_hidden_layers=int(cfg['num_hidden_layers']),
            max_seq_len=int(cfg['max_seq_len']),use_moe=False,
        )
        m0,tok0=init_model(mmcfg0,from_weight='none',tokenizer_path=str(minimind/'model'),device=args.device)
        ds0=PretrainDataset(str(data_path),tok0,max_length=int(cfg['max_seq_len']))
        partition=build_partition(
            len(ds0),int(cfg['partition_seed']),
            int(cfg['probe_pool_size']),int(cfg['eval_pool_size'])
        )
        save_partition(part_path,partition)
        del m0, ds0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_reports=[]

    for train_seed in seeds:
        seed_dir=out/f'seed_{train_seed}'
        states_dir=seed_dir/'states'
        report_path=seed_dir/'state_bank_report.json'

        if report_path.exists() and not args.force:
            old=json.loads(report_path.read_text(encoding='utf-8'))
            if old.get('technical_pass') and int(old.get('train_seed',-1))==train_seed:
                print(f'SEED_{train_seed}_STATE_BANK_SKIP_EXISTING_PASS')
                all_reports.append(old)
                continue

        states_dir.mkdir(parents=True,exist_ok=True)
        setup_seed(train_seed)

        mmcfg=MiniMindConfig(
            hidden_size=int(cfg['hidden_size']),
            num_hidden_layers=int(cfg['num_hidden_layers']),
            max_seq_len=int(cfg['max_seq_len']),use_moe=False,
        )
        model,tokenizer=init_model(
            mmcfg,from_weight='none',
            tokenizer_path=str(minimind/'model'),device=args.device
        )
        model.train()
        ds=PretrainDataset(str(data_path),tokenizer,max_length=int(cfg['max_seq_len']))
        autocast_ctx,scaler,_=make_amp(torch,args.device,cfg['dtype'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg['learning_rate']))
        assert_native_adamw(optimizer)
        optimizer.zero_grad(set_to_none=True)

        accum=int(cfg['accumulation_steps'])
        total_opt=int(cfg['planned_total_optimizer_steps'])
        total_micro=total_opt*accum
        state_steps=sorted(map(int,cfg['e3_state_steps']))
        max_micro=max(state_steps)*accum

        # [FR-BATCH-STREAM] 每个 training seed 使用不同的训练 batch trajectory。
        train_batch_seed=train_seed+1704
        batches=batches_from_train_pool(
            partition['train_indices'],int(cfg['batch_size']),
            train_batch_seed,max_micro
        )
        used=set()
        records=[]

        for micro_idx,batch_idx in enumerate(batches,start=1):
            used.update(batch_idx)
            set_native_micro_lr(
                optimizer,get_lr,micro_idx,total_micro,float(cfg['learning_rate'])
            )
            x,y=stack_batch(ds,batch_idx,args.device)
            with autocast_ctx:
                res=model(x,labels=y)
                loss=(res.loss+res.aux_loss)/accum
            scaler.scale(loss).backward()

            if micro_idx%accum==0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['grad_clip']))
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

                opt_step=micro_idx//accum
                if opt_step in state_steps:
                    path=states_dir/f'state_step_{opt_step:03d}.pt'
                    st=save_full_training_state(
                        path,model,optimizer,scaler,
                        metadata={
                            'protocol':'E3R_V06_STATE_BANK',
                            'train_seed':train_seed,
                            'state_optimizer_step':opt_step,
                            'state_fraction':opt_step/total_opt,
                            'global_optimizer_step':opt_step,
                            'global_micro_step':micro_idx,
                            'planned_total_optimizer_steps':total_opt,
                            'total_micro_steps':total_micro,
                            'partition_sha256':partition['sha256'],
                            'optimizer_signature':optimizer_signature(optimizer),
                            'config':cfg,
                        }
                    )
                    records.append({
                        'step':opt_step,
                        'path':str(path),
                        'model_hash':stable_hash(st['model']),
                        'optimizer_hash':stable_hash(st['optimizer']),
                    })

        probe_overlap=len(used & set(partition['probe_indices']))
        eval_overlap=len(used & set(partition['eval_indices']))
        if probe_overlap or eval_overlap:
            raise RuntimeError(f'seed={train_seed} state trajectory 使用了 probe/eval')

        report={
            'technical_pass':len(records)==len(state_steps),
            'train_seed':train_seed,
            'train_batch_seed':train_batch_seed,
            'state_steps':state_steps,
            'partition_sha256':partition['sha256'],
            'probe_overlap':probe_overlap,
            'eval_overlap':eval_overlap,
            'states':records,
        }

        # seed2026 state64 与 v0.5 做额外锚点检查。
        if train_seed==2026:
            old=root/'outputs/e3_mapping/v05/states/state_step_064.pt'
            if old.exists():
                old_payload=torch.load(old,map_location='cpu',weights_only=False)
                new64=next(x for x in records if x['step']==64)
                report['v05_anchor_step64']={
                    'exists':True,
                    'model_hash_equal':stable_hash(old_payload['model'])==new64['model_hash'],
                    'optimizer_hash_equal':stable_hash(old_payload['optimizer'])==new64['optimizer_hash'],
                }
                if not all([
                    report['v05_anchor_step64']['model_hash_equal'],
                    report['v05_anchor_step64']['optimizer_hash_equal']
                ]):
                    raise RuntimeError('seed2026 step64 无法复现 v0.5 state')
            else:
                report['v05_anchor_step64']={'exists':False}

        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        all_reports.append(report)
        print(f'SEED_{train_seed}_STATE_BANK_PASS')
        print(' probe_overlap=',probe_overlap,'eval_overlap=',eval_overlap)
        if train_seed==2026:
            print(' v05_anchor_step64=',report.get('v05_anchor_step64'))

        del model, optimizer, scaler, ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary={
        'technical_pass':all(x.get('technical_pass') for x in all_reports),
        'seeds':seeds,
        'partition_sha256':partition['sha256'],
        'seed_reports':[str(out/f'seed_{s}/state_bank_report.json') for s in seeds],
    }
    (out/'multiseed_state_bank_summary.json').write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'
    )
    print('E3R_V06_MULTISEED_STATE_BANK_PASS')
    print('seeds=',seeds)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
