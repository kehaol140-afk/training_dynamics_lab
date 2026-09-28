#!/usr/bin/env python3
"""TASK-00 服务器真实源码自动审计脚本。

作用
----
不修改任何训练代码，只读取当前服务器 `minimind/` 的真实文件，检查 E0 所依赖的
训练顺序、checkpoint、RNG、数据顺序与 AdamW 语义，并输出 Markdown/JSON 报告。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_def(path: Path, name: str) -> Tuple[str, int, int]:
    txt = path.read_text(encoding="utf-8")
    lines = txt.splitlines()
    tree = ast.parse(txt)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            end = getattr(node, "end_lineno", node.lineno)
            return "\n".join(lines[node.lineno - 1:end]), node.lineno, end
    raise KeyError(f"在 {path} 中找不到 {name}")


def pos_order(txt: str, tokens):
    positions = [txt.find(t) for t in tokens]
    return all(p >= 0 for p in positions) and positions == sorted(positions), positions


def main() -> int:
    ap = argparse.ArgumentParser(description="TASK-00 MiniMind 实现审计")
    ap.add_argument("--project-root", required=True)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    train_p = root / "minimind/trainer/train_pretrain.py"
    utils_p = root / "minimind/trainer/trainer_utils.py"
    data_p = root / "minimind/dataset/lm_dataset.py"
    for p in (train_p, utils_p, data_p):
        if not p.exists():
            raise SystemExit(f"缺少源码文件：{p}")

    train = train_p.read_text(encoding="utf-8")
    lm_ckpt, ck_l1, ck_l2 = extract_def(utils_p, "lm_checkpoint")
    setup_seed, seed_l1, seed_l2 = extract_def(utils_p, "setup_seed")
    skip_sampler, skip_l1, skip_l2 = extract_def(utils_p, "SkipBatchSampler")
    pretrain_ds, ds_l1, ds_l2 = extract_def(data_p, "PretrainDataset")

    order_tokens = [
        "scaler.scale(loss).backward()",
        "scaler.unscale_(optimizer)",
        "clip_grad_norm_",
        "scaler.step(optimizer)",
        "scaler.update()",
        "optimizer.zero_grad(set_to_none=True)",
    ]
    order_ok, positions = pos_order(train, order_tokens)

    findings: Dict[str, object] = {
        "source_sha256": {
            "train_pretrain.py": sha256_file(train_p),
            "trainer_utils.py": sha256_file(utils_p),
            "lm_dataset.py": sha256_file(data_p),
        },
        "native_adamw": "optim.AdamW(model.parameters(), lr=args.learning_rate)" in train,
        "gradient_accumulation": "step % args.accumulation_steps == 0" in train,
        "training_order_ok": order_ok,
        "training_order_positions": positions,
        "native_checkpoint_model_half": ".half().cpu()" in lm_ckpt,
        "native_checkpoint_optimizer": "'optimizer'" in lm_ckpt and "optimizer.state_dict()" in lm_ckpt,
        "native_checkpoint_epoch": "'epoch'" in lm_ckpt,
        "native_checkpoint_step": "'step'" in lm_ckpt,
        "native_checkpoint_rng": any(x in lm_ckpt for x in ["get_rng_state", "random.getstate", "np.random.get_state"]),
        "scaler_passed_to_checkpoint": "scaler=scaler" in train,
        "seed_controls_python": "random.seed" in setup_seed,
        "seed_controls_numpy": "np.random.seed" in setup_seed,
        "seed_controls_torch": "torch.manual_seed" in setup_seed,
        "seed_controls_cuda": "torch.cuda.manual_seed_all" in setup_seed,
        "single_gpu_randperm": "torch.randperm(len(train_ds)).tolist()" in train,
        "skip_batch_sampler": "SkipBatchSampler" in train,
        "pretrain_dataset_random_in_getitem": "random." in pretrain_ds,
        "line_ranges": {
            "lm_checkpoint": [ck_l1, ck_l2],
            "setup_seed": [seed_l1, seed_l2],
            "SkipBatchSampler": [skip_l1, skip_l2],
            "PretrainDataset": [ds_l1, ds_l2],
        },
    }

    # E0 判断：原生 checkpoint 不够，但 FR v0.2 提供独立 full-state 后可进入 smoke。
    critical_source_ok = bool(findings["native_adamw"] and findings["gradient_accumulation"] and findings["training_order_ok"])
    findings["native_resume_exact_replay_ready"] = False
    findings["fr_v02_e0_smoke_ready"] = critical_source_ok

    out_dir = root / "research/reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "TASK00_SOURCE_AUDIT.json").write_text(
        json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = [
        ("原生 AdamW", "PASS" if findings["native_adamw"] else "FAIL", "train_pretrain.py"),
        ("梯度累积", "PASS" if findings["gradient_accumulation"] else "FAIL", "按 micro-step 判断 accumulation 边界"),
        ("backward→unscale→clip→step→zero_grad 顺序", "PASS" if findings["training_order_ok"] else "FAIL", "未来扰动首选 unscale 后、clip 前"),
        ("原生 checkpoint 保存 optimizer", "PASS" if findings["native_checkpoint_optimizer"] else "FAIL", "包含 AdamW state"),
        ("原生 checkpoint 保存 epoch/step", "PASS" if findings["native_checkpoint_epoch"] and findings["native_checkpoint_step"] else "FAIL", "训练位置"),
        ("原生 checkpoint 保持模型原始 dtype", "FAIL" if findings["native_checkpoint_model_half"] else "PASS", "检测到 half().cpu()" if findings["native_checkpoint_model_half"] else "未检测到强制 half"),
        ("原生 checkpoint 保存 RNG", "PASS" if findings["native_checkpoint_rng"] else "MISSING", "E0 由 FR full_state 补齐"),
        ("单 GPU randperm 数据顺序", "PASS" if findings["single_gpu_randperm"] else "FAIL", "每 epoch 可重建，但正式 paired branch 仍保存 manifest"),
        ("PretrainDataset __getitem__ 随机性", "PARTIAL" if findings["pretrain_dataset_random_in_getitem"] else "PASS", "未检测到 random.*" if not findings["pretrain_dataset_random_in_getitem"] else "检测到随机调用，需人工看"),
    ]

    md = [
        "# TASK-00 服务器真实源码审计报告",
        "",
        "> `[FR-OBSERVATION-ONLY]`：本报告只读源码，没有修改 `minimind/`。",
        "",
        "## 核心判定",
        "",
        f"- 原生 MiniMind resume 是否满足 E0 精确重放：**NO**",
        f"- 使用 FR v0.2 full-state 后是否允许进入 E0 smoke：**{'YES' if findings['fr_v02_e0_smoke_ready'] else 'NO'}**",
        "",
        "## 状态表",
        "",
        "| 项目 | 状态 | 说明 |",
        "|---|---|---|",
    ]
    md += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    md += [
        "",
        "## 关键结论",
        "",
        "1. 当前训练主路径必须保持 `backward → unscale → clip → scaler.step(AdamW) → scaler.update → zero_grad`。",
        "2. 未来 `±epsilon*q` 的首选候选插入位置是 `unscale` 后、`clip` 前；E0 当前禁止任何扰动。",
        "3. MiniMind 原生 resume 面向常规续训，不是 E0 无损 checkpoint；FR v0.2 单独保存原始 dtype + optimizer + scaler + RNG。",
        "4. 单 GPU非 DDP 的 epoch 顺序可由 `42+epoch` 重建，但 paired branch 使用显式 batch manifest 更安全。",
        "5. FR checkpoint 必须放在 optimizer step + zero_grad 之后的干净边界。",
        "",
        "## 源码 SHA256",
        "",
    ]
    md += [f"- `{k}`: `{v}`" for k, v in findings["source_sha256"].items()]
    (out_dir / "TASK00_SOURCE_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("[FR-OBSERVATION-ONLY] TASK00_SOURCE_AUDIT_DONE")
    print(f"native_resume_exact_replay_ready={findings['native_resume_exact_replay_ready']}")
    print(f"fr_v02_e0_smoke_ready={findings['fr_v02_e0_smoke_ready']}")
    print(f"report={out_dir / 'TASK00_SOURCE_AUDIT.md'}")
    return 0 if findings["fr_v02_e0_smoke_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
