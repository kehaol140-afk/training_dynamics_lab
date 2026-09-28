#!/usr/bin/env python3
"""v0.6 跨 seed 的 theta FR 原始测量。

每个 seed 单独保存结果，支持断点式重跑。
本脚本只生成 state-level B_theta / chi_theta，不做最终跨 seed 解释。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from fr.batch_manifest import save_manifest
from fr.direction import normalize_direction, direction_norm, save_direction
from fr.full_state import load_full_training_state
from fr.minimind_v03_runtime import make_amp, raw_model, stack_batch
from fr.native_adamw_guard import assert_native_adamw
from fr.pilot_partition import load_partition
from fr.response_stats import observational_slope, paired_linearity_summary
from fr.stage_c_runtime import batches_from_train_pool
from fr.stream_rng import derive_stream_seed
from fr.theta_protocol import run_theta_branch


def parse_seeds(text, defaults):
    if not text:
        return list(map(int,defaults))
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def two_level_seed(base, train_seed, state_step, stream_id):
    """training seed 也进入 future seed 派生，避免跨 seed 复用局部随机流。"""
    x=derive_stream_seed(int(base),int(train_seed))
    x=derive_stream_seed(x,int(state_step))
    return derive_stream_seed(x,int(stream_id))


def collect_passed_seed_results(out, all_seeds):
    """扫描现有 seed raw result，partial rerun 不覆盖其它已完成 seed。"""
    out=Path(out)
    passed=[]
    for seed in map(int,all_seeds):
        p=out/f'seed_{seed}/theta_mapping_raw.json'
        if not p.exists():
            continue
        try:
            x=json.loads(p.read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError):
            continue
        if (
            x.get('technical_pass') is True
            and x.get('rho0_noop_pass') is True
            and int(x.get('train_seed',-1))==seed
        ):
            passed.append(seed)
    return passed


def write_multiseed_theta_summary(out, *, all_seeds, requested_seeds, completed_this_run):
    """写入汇总；汇总来源为磁盘上所有已 PASS seed，而不是本次请求子集。"""
    out=Path(out)
    seeds=list(map(int,all_seeds))
    completed=collect_passed_seed_results(out,seeds)
    summary={
        'technical_pass':len(completed)==len(seeds),
        'seeds':seeds,
        'requested_seeds':list(map(int,requested_seeds)),
        'completed_this_run':list(map(int,completed_this_run)),
        'completed':completed,
        'missing_or_not_passed':[s for s in seeds if s not in set(completed)],
    }
    (out/'multiseed_theta_summary.json').write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'
    )
    return summary


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
    all_seeds=list(map(int,cfg['replication_train_seeds']))
    seeds=parse_seeds(args.seeds,all_seeds)
    minimind=root/'minimind'; sys.path.insert(0,str(minimind))

    import numpy as np
    import torch
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig
    from trainer.trainer_utils import get_lr, init_model

    out=root/'outputs/e3_replication/v06'
    partition=load_partition(out/'partition_v06.json')
    data_path=out/'e3r_subset.jsonl'

    horizons=sorted(map(int,cfg['horizons']))
    max_h=max(horizons)
    accum=int(cfg['accumulation_steps'])
    rho=float(cfg['e3_primary_rho'])
    e1_R=int(cfg['e3_e1_streams'])
    e2_R=int(cfg['e3_e2_streams'])

    completed=[]

    for train_seed in seeds:
        seed_dir=out/f'seed_{train_seed}'
        bank_path=seed_dir/'state_bank_report.json'
        result_path=seed_dir/'theta_mapping_raw.json'

        if result_path.exists() and not args.force:
            old=json.loads(result_path.read_text(encoding='utf-8'))
            if old.get('technical_pass') and int(old.get('train_seed',-1))==train_seed:
                print(f'SEED_{train_seed}_MAPPING_SKIP_EXISTING_PASS')
                completed.append(train_seed)
                continue

        if not bank_path.exists():
            raise SystemExit(f'缺少 seed {train_seed} state bank')
        bank=json.loads(bank_path.read_text(encoding='utf-8'))
        if not bank.get('technical_pass'):
            raise SystemExit(f'seed {train_seed} state bank 未 PASS')

        mmcfg=MiniMindConfig(
            hidden_size=int(cfg['hidden_size']),
            num_hidden_layers=int(cfg['num_hidden_layers']),
            max_seq_len=int(cfg['max_seq_len']),use_moe=False,
        )
        model,tokenizer=init_model(
            mmcfg,from_weight='none',
            tokenizer_path=str(minimind/'model'),device=args.device
        )
        ds=PretrainDataset(str(data_path),tokenizer,max_length=int(cfg['max_seq_len']))
        autocast_ctx,scaler,_=make_amp(torch,args.device,cfg['dtype'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg['learning_rate']))
        assert_native_adamw(optimizer)
        named=[(n,p) for n,p in raw_model(model).named_parameters() if p.requires_grad]

        state_results=[]
        rho0_noop=False

        for state_rec in bank['states']:
            step=int(state_rec['step'])
            state_path=Path(state_rec['path'])
            meta=load_full_training_state(state_path,model,optimizer,scaler)
            payload=torch.load(state_path,map_location='cpu',weights_only=False)
            reference_theta=payload['model']

            # q_t：每个 state 从同一 held-out probe 重新构造。
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            xp,yp=stack_batch(ds,partition['probe_indices'],args.device)
            with autocast_ctx:
                rr=model(xp,labels=yp)
                probe_loss=rr.loss+rr.aux_loss
            params=[p for _,p in named]
            grads=torch.autograd.grad(probe_loss,params,allow_unused=True)
            q_raw={
                n:(torch.zeros_like(p.detach()) if g is None else g.detach().clone())
                for (n,p),g in zip(named,grads)
            }
            q,probe_grad_norm=normalize_direction(q_raw)
            q_norm=direction_norm(q)
            q_path=seed_dir/'q'/f'q_state_step_{step:03d}.pt'
            q_sha=save_direction(
                q_path,q,
                metadata={
                    'protocol':'E3R_V06_MULTISEED_THETA',
                    'train_seed':train_seed,
                    'state_step':step,
                    'state_path':str(state_path),
                    'partition_sha256':partition['sha256'],
                    'probe_indices':partition['probe_indices'],
                    'probe_loss':float(probe_loss.detach().float().cpu()),
                    'probe_grad_norm_before_normalize':probe_grad_norm,
                    'q_norm':q_norm,
                },
            )
            optimizer.zero_grad(set_to_none=True)
            model.train()

            # E1
            s_values=[]
            e1_rows=[]
            e1_stream_records=[]
            for r in range(e1_R):
                stream_seed=two_level_seed(
                    cfg['v06_e1_stream_base_seed'],train_seed,step,r
                )
                batch_seed=derive_stream_seed(stream_seed,404)
                batches=batches_from_train_pool(
                    partition['train_indices'],int(cfg['batch_size']),
                    batch_seed,max_h*accum
                )
                manifest_path=(
                    seed_dir/'e1_stream_manifests'
                    /f'state_step_{step:03d}'
                    /f'e1_stream_{r:03d}_batches.json'
                )
                manifest_sha=save_manifest(
                    manifest_path,batches,
                    metadata={
                        'protocol':'E3R_V06_E1_STREAM',
                        'train_seed':train_seed,
                        'state_step':step,
                        'stream_id':r,
                        'stream_seed':stream_seed,
                        'batch_seed':batch_seed,
                    },
                )
                br=run_theta_branch(
                    torch=torch,model=model,optimizer=optimizer,scaler=scaler,
                    autocast_ctx=autocast_ctx,ds=ds,state_path=state_path,
                    reference_theta=reference_theta,q=q,named=named,batches=batches,
                    stream_seed=stream_seed,sign=0,epsilon=0.0,
                    horizons=horizons,accumulation_steps=accum,
                    batch_size=int(cfg['batch_size']),device=args.device,
                    grad_clip=float(cfg['grad_clip']),
                    base_lr=float(cfg['learning_rate']),
                    global_micro_step=int(meta['global_micro_step']),
                    total_micro_steps=int(meta['total_micro_steps']),
                    get_lr=get_lr,stack_batch=stack_batch,measure_s_first=True
                )
                s_values.append(float(br['s']))
                e1_rows.append(br)
                e1_stream_records.append({
                    'stream_id':r,
                    's_r':float(br['s']),
                    'stream_seed':stream_seed,
                    'batch_seed':batch_seed,
                    'batch_manifest_path':str(manifest_path),
                    'batch_manifest_sha256':manifest_sha,
                    'batch_manifest_hash':br['batch_manifest_hash'],
                    'first_preclip_grad_norm':br['first_preclip_grad_norm'],
                })

            sigma_q=float(np.std(np.asarray(s_values,dtype=np.float64),ddof=1))
            if not np.isfinite(sigma_q) or sigma_q<=0:
                raise RuntimeError(f'seed={train_seed}, step={step} sigma_q 无效')
            epsilon=rho*sigma_q
            B={}
            for h in horizons:
                B[str(h)]=observational_slope(
                    s_values,
                    [x['observations'][str(h)]['X_theta'] for x in e1_rows]
                )['B']

            # E2
            e2_rows=[]
            for r in range(e2_R):
                stream_seed=two_level_seed(
                    cfg['v06_e2_stream_base_seed'],train_seed,step,r
                )
                batch_seed=derive_stream_seed(stream_seed,404)
                batches=batches_from_train_pool(
                    partition['train_indices'],int(cfg['batch_size']),
                    batch_seed,max_h*accum
                )
                common=dict(
                    torch=torch,model=model,optimizer=optimizer,scaler=scaler,
                    autocast_ctx=autocast_ctx,ds=ds,state_path=state_path,
                    reference_theta=reference_theta,q=q,named=named,batches=batches,
                    stream_seed=stream_seed,horizons=horizons,
                    accumulation_steps=accum,batch_size=int(cfg['batch_size']),
                    device=args.device,grad_clip=float(cfg['grad_clip']),
                    base_lr=float(cfg['learning_rate']),
                    global_micro_step=int(meta['global_micro_step']),
                    total_micro_steps=int(meta['total_micro_steps']),
                    get_lr=get_lr,stack_batch=stack_batch,measure_s_first=False,
                )
                b0=run_theta_branch(sign=0,epsilon=0.0,**common)
                bp=run_theta_branch(sign=+1,epsilon=epsilon,**common)
                bm=run_theta_branch(sign=-1,epsilon=epsilon,**common)
                if not (
                    b0['batch_manifest_hash']==bp['batch_manifest_hash']
                    ==bm['batch_manifest_hash']
                ):
                    raise RuntimeError(
                        f'seed={train_seed},step={step},stream={r} paired batch 不一致'
                    )
                e2_rows.append({'baseline':b0,'plus':bp,'minus':bm})

                # 每个 training seed 至少一次 rho=0 no-op。
                if not rho0_noop:
                    bz=run_theta_branch(sign=+1,epsilon=0.0,**common)
                    if not (
                        b0['model_hash_final']==bz['model_hash_final']
                        and b0['optimizer_hash_final']==bz['optimizer_hash_final']
                        and b0['observations']==bz['observations']
                    ):
                        raise RuntimeError(f'seed={train_seed} rho=0 no-op 失败')
                    rho0_noop=True

            chi={}; odd={}
            for h in horizons:
                hk=str(h)
                sm=paired_linearity_summary(
                    [x['baseline']['observations'][hk]['X_theta'] for x in e2_rows],
                    [x['plus']['observations'][hk]['X_theta'] for x in e2_rows],
                    [x['minus']['observations'][hk]['X_theta'] for x in e2_rows],
                    epsilon
                )
                chi[hk]=sm['chi']
                odd[hk]=sm['odd_error']

            clip={
                'baseline_rate':sum(x['baseline']['first_clipped'] for x in e2_rows)/e2_R,
                'plus_rate':sum(x['plus']['first_clipped'] for x in e2_rows)/e2_R,
                'minus_rate':sum(x['minus']['first_clipped'] for x in e2_rows)/e2_R,
            }
            state_results.append({
                'train_seed':train_seed,
                'step':step,
                'state_fraction':float(meta['state_fraction']),
                'q_norm':q_norm,
                'q_path':str(q_path),
                'q_sha256':q_sha,
                'probe_grad_norm':probe_grad_norm,
                'sigma_q':sigma_q,
                'epsilon':epsilon,
                'B_theta':B,
                'chi_theta':chi,
                'odd_error_theta':odd,
                'clipping':clip,
                'e1_streams':e1_stream_records,
            })
            print(
                f'seed={train_seed} state={step} sigma_q={sigma_q:.6g} '
                f'B={B} chi={chi}'
            )

        result={
            'protocol':'E3R_V06_MULTISEED_THETA',
            'technical_pass':True,
            'train_seed':train_seed,
            'rho0_noop_pass':rho0_noop,
            'primary_rho':rho,
            'horizons':horizons,
            'states':state_results,
        }
        result_path.write_text(
            json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'
        )
        completed.append(train_seed)
        print(f'SEED_{train_seed}_THETA_MAPPING_PASS')
        del model,optimizer,scaler,ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary=write_multiseed_theta_summary(
        out,
        all_seeds=all_seeds,
        requested_seeds=seeds,
        completed_this_run=completed,
    )
    print('E3R_V06_MULTISEED_THETA_TECHNICAL_PASS')
    print('completed=',summary['completed'])
    return 0

if __name__=='__main__':
    raise SystemExit(main())
