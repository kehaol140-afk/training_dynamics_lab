#!/usr/bin/env python3
"""只读环境与数据基线检查脚本。

用途
----
1. 检查当前 Python 版本；
2. 检查固定预训练数据是否存在、大小和 SHA256；
3. 检查 PyTorch / CUDA / 可见 GPU；
4. 输出 GPU 名称与计算能力。

科学安全性
--------
这是 `[FR-OBSERVATION-ONLY]` 脚本：只读取状态，不修改模型、优化器、RNG、
batch 顺序或数据集内容。

什么时候运行
----------
每次换机器、换环境、正式实验前都可以运行一次。
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import platform


def sha256_file(path: Path, chunk=8 * 1024 * 1024):
    """分块计算大文件 SHA256，避免一次把 1.2GB 数据读入内存。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="检查涨落—响应项目环境与数据基线")
    ap.add_argument("--project-root", required=True, help="项目根目录")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    data = root / "minimind/dataset/pretrain_t2t_mini.jsonl"

    # [FR-OBSERVATION-ONLY] 以下只读，不改变训练状态。
    print("[FR-OBSERVATION-ONLY] 环境检查")
    print("python:", platform.python_version())
    print("project_root:", root)
    print("dataset_exists:", data.exists())
    if data.exists():
        print("dataset_bytes:", data.stat().st_size)
        print("dataset_sha256:", sha256_file(data))

    try:
        import torch
        print("torch:", torch.__version__)
        print("torch_cuda:", torch.version.cuda)
        print("cuda_available:", torch.cuda.is_available())
        print("visible_gpu_count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"gpu[{i}]:", torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
    except Exception as e:
        print("torch_check_error:", repr(e))
        raise


if __name__ == "__main__":
    main()
