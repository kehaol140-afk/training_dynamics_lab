from __future__ import annotations

import sys
from collections import defaultdict

from . import config


def _stable_hash(obj) -> str:
    sys.path.insert(0, str(config.PROJECT_ROOT / "src"))
    from vendored_fr.hashing import stable_hash

    return stable_hash(obj)


def make_permutation(n: int, seed: int):
    import random

    rng = random.Random(int(seed))
    perm = list(range(n))
    rng.shuffle(perm)
    return perm


def build_thread_index(records: list[dict]) -> dict:
    threads = defaultdict(list)
    for i, r in enumerate(records):
        threads[r.get("conversation_id")].append(i)
    return dict(threads)


def build_pair_aligned_split(records_a: list[dict], records_b: list[dict]) -> dict:
    """Thread-aligned train/dev/confirm split.

    conversation_id is a THREAD id: each line is one exchange inside the thread.
    A thread's lines (in both domains) are assigned to exactly one split, so
    translation pairs (cid, k) can never straddle splits. Threads are permuted
    with the frozen seed and cut into three parts by thread count.
    """
    threads_a = build_thread_index(records_a)
    threads_b = build_thread_index(records_b)
    all_cids = sorted(set(threads_a) | set(threads_b))
    perm = [all_cids[i] for i in make_permutation(len(all_cids), config.SPLIT_SEED)]

    fr = config.SPLIT_FRACTIONS
    n = len(perm)
    n_train = int(round(fr["train"] * n))
    n_dev = int(round(fr["dev"] * n))
    thread_splits = {
        "train": perm[:n_train],
        "dev": perm[n_train : n_train + n_dev],
        "confirm": perm[n_train + n_dev :],
    }

    idx = {"A": {"train": [], "dev": [], "confirm": []},
           "B": {"train": [], "dev": [], "confirm": []}}
    for split, cids in thread_splits.items():
        for cid in cids:
            if cid in threads_a:
                idx["A"][split].extend(threads_a[cid])
            if cid in threads_b:
                idx["B"][split].extend(threads_b[cid])
    for dom in idx:
        for split in idx[dom]:
            idx[dom][split] = sorted(idx[dom][split])

    pairable = sum(1 for cid in set(threads_a) & set(threads_b)
                   if len(threads_a[cid]) == len(threads_b[cid]))
    shared = set(threads_a) & set(threads_b)
    manifest = {
        "split_version": "mte_split_v2_thread",
        "split_seed": config.SPLIT_SEED,
        "fractions": fr,
        "unit": "thread (conversation_id); every line of a thread goes to one split in both domains",
        "counts": {
            "A": {s: len(v) for s, v in idx["A"].items()},
            "B": {s: len(v) for s, v in idx["B"].items()},
        },
        "thread_counts": {s: len(v) for s, v in thread_splits.items()},
        "threads_total": len(all_cids),
        "threads_shared": len(shared),
        "threads_only_A": len(set(threads_a) - set(threads_b)),
        "threads_only_B": len(set(threads_b) - set(threads_a)),
        "shared_threads_with_equal_line_counts": pairable,
        "A": idx["A"],
        "B": idx["B"],
    }
    return manifest


def verify_split_integrity(manifest: dict, records_a: list[dict], records_b: list[dict]) -> dict:
    problems = []
    for dom, records in (("A", records_a), ("B", records_b)):
        idx = manifest[dom]
        tr, dv, cf = set(idx["train"]), set(idx["dev"]), set(idx["confirm"])
        if tr & dv or tr & cf or dv & cf:
            problems.append(f"{dom}: split overlap")
        if (tr | dv | cf) != set(range(len(records))):
            problems.append(f"{dom}: split does not cover all lines exactly once")

    threads_a = build_thread_index(records_a)
    threads_b = build_thread_index(records_b)
    split_of = {}
    for dom, threads in (("A", threads_a), ("B", threads_b)):
        for split in ("train", "dev", "confirm"):
            for i in manifest[dom][split]:
                cid = (records_a if dom == "A" else records_b)[i].get("conversation_id")
                key = (dom, cid)
                if key in split_of and split_of[key] != split:
                    problems.append(f"thread {cid} straddles splits in {dom}")
                split_of[key] = split
    for cid in set(threads_a) & set(threads_b):
        sa = split_of.get(("A", cid))
        sb = split_of.get(("B", cid))
        if sa != sb:
            problems.append(f"shared thread {cid} in different splits: A:{sa} vs B:{sb}")

    return {
        "ok": not problems,
        "problems": problems[:20],
        "shared_threads_checked": len(set(threads_a) & set(threads_b)),
        "split_of_manifest_sha256": _stable_hash({d: manifest[d] for d in ("A", "B")}),
    }
