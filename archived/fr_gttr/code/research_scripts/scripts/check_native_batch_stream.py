#!/usr/bin/env python3
"""验证 MiniMind 单 GPU非 DDP 的原生 batch 顺序是否能够确定性重建。

这里只比较 index 顺序，不运行模型，不修改训练状态。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fr.batch_manifest import build_native_single_gpu_batches, manifest_sha256


def main() -> int:
    ap = argparse.ArgumentParser(description="MiniMind 原生 batch stream 重放检查")
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--dataset-len", type=int, default=1000, help="只需一个正整数即可检查 sampler 语义")
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip", type=int, default=3)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    minimind = root / "minimind"
    sys.path.insert(0, str(minimind))

    import torch
    from trainer.trainer_utils import SkipBatchSampler, setup_seed

    def native_once():
        # ============================================================
        # [FR-BATCH-STREAM] 逐字复现当前 MiniMind 单 GPU非 DDP 构造顺序的逻辑。
        # ============================================================
        setup_seed(42 + args.epoch)
        indices = torch.randperm(args.dataset_len).tolist()
        return [list(x) for x in SkipBatchSampler(indices, args.batch_size, args.skip)]

    a = native_once()
    b = native_once()
    explicit = build_native_single_gpu_batches(
        args.dataset_len, args.epoch, args.batch_size, skip_batches=args.skip
    )

    ok_repeat = a == b
    ok_explicit = a == explicit
    print("[FR-BATCH-STREAM] 原生重复构造一致:", ok_repeat)
    print("[FR-BATCH-STREAM] FR显式manifest与原生一致:", ok_explicit)
    print("manifest_sha256:", manifest_sha256(a))
    if ok_repeat and ok_explicit:
        print("NATIVE_BATCH_STREAM_PASS")
        return 0
    print("NATIVE_BATCH_STREAM_FAIL")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
