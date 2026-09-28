#!/usr/bin/env python3
"""GPU 前的轻量科学代码门禁。

它不能取代人工/Codex 审计，但用于阻止最常见的结构性错误进入跨 seed 大实验。
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def ordered_positions(text, patterns):
    pos=[]
    start=0
    for p in patterns:
        i=text.find(p,start)
        if i<0:
            return None
        pos.append(i)
        start=i+1
    return pos


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root',required=True)
    args=ap.parse_args()
    root=Path(args.project_root).resolve()

    checks={}

    # A. MiniMind 原生训练顺序
    train=(root/'minimind/trainer/train_pretrain.py').read_text(encoding='utf-8')
    native_patterns=[
        "lr = get_lr(",
        "with autocast_ctx:",
        "scaler.scale(loss).backward()",
        "scaler.unscale_(optimizer)",
        "clip_grad_norm_",
        "scaler.step(optimizer)",
        "optimizer.zero_grad(set_to_none=True)",
    ]
    checks['native_train_order']=ordered_positions(train,native_patterns) is not None

    # B. v0.5/v0.6 future branch 顺序
    theta=(root/'research/src/fr/theta_protocol.py').read_text(encoding='utf-8')
    branch_patterns=[
        "set_native_micro_lr(",
        "with autocast_ctx:",
        "scaler.scale(loss).backward()",
        "scaler.unscale_(optimizer)",
        "apply_signed_gradient_impulse_(",
        "clip_grad_norm_",
        "scaler.step(optimizer)",
        "optimizer.zero_grad(set_to_none=True)",
    ]
    checks['fr_branch_order']=ordered_positions(theta,branch_patterns) is not None

    # C. full state 必须明确包含 optimizer/RNG
    full=(root/'research/src/fr/full_state.py').read_text(encoding='utf-8')
    for token,name in [
        ('"model"', 'full_state_model'),
        ('"optimizer"', 'full_state_optimizer'),
        ('capture_rng_state()', 'full_state_capture_rng'),
        ('restore_rng_state(state["rng"])', 'full_state_restore_rng'),
    ]:
        checks[name]=token in full

    # D. 历史 E3 证据（如果存在）
    old_bank=root/'outputs/e3_mapping/v05/state_bank_report.json'
    if old_bank.exists():
        x=json.loads(old_bank.read_text(encoding='utf-8'))
        checks['v05_probe_overlap_zero']=int(x.get('probe_overlap',-1))==0
        checks['v05_eval_overlap_zero']=int(x.get('eval_overlap',-1))==0
        bridge=x.get('bridge_v04',{})
        checks['v05_v04_model_bridge']=bridge.get('model_hash_equal') is True
        checks['v05_v04_optimizer_bridge']=bridge.get('optimizer_hash_equal') is True
    else:
        checks['v05_probe_overlap_zero']=False
        checks['v05_eval_overlap_zero']=False
        checks['v05_v04_model_bridge']=False
        checks['v05_v04_optimizer_bridge']=False

    old_map=root/'outputs/e3_mapping/v05/e3_theta_mapping_report.json'
    if old_map.exists():
        x=json.loads(old_map.read_text(encoding='utf-8'))
        checks['v05_rho0_noop']=x.get('rho0_noop_pass') is True
        checks['v05_technical_pass']=x.get('technical_pass') is True
    else:
        checks['v05_rho0_noop']=False
        checks['v05_technical_pass']=False

    passed=all(checks.values())
    report={'pass':passed,'checks':checks}
    out=root/'outputs/e3_replication/v06'
    out.mkdir(parents=True,exist_ok=True)
    (out/'scientific_code_preflight_v06.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'
    )

    print('V06_SCIENTIFIC_CODE_PREFLIGHT_PASS' if passed else 'V06_SCIENTIFIC_CODE_PREFLIGHT_FAIL')
    for k,v in checks.items():
        print(f'{k}={v}')
    return 0 if passed else 2

if __name__=='__main__':
    raise SystemExit(main())
