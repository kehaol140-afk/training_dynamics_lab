"""M0 Gate 3 (A+B): dataset token statistics, duplicates, pair-aligned splits, packing.
Writes outputs/m0/data_audit/{data_report.json, split_manifest.json}."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from muon_transition import config
from muon_transition.data import (
    TokenizerHandle,
    conversation_to_text,
    duplicate_stats,
    load_domain,
    pack_domain_stream,
    sha256_file,
)
from muon_transition.partitions import build_pair_aligned_split, verify_split_integrity


def domain_stats(name, path, expected_sha, records, texts, token_lists, tok):
    total_tokens = sum(len(t) for t in token_lists)
    used = set()
    for t in token_lists:
        used.update(t)
    lengths = [len(t) for t in token_lists]
    dup = duplicate_stats(texts)
    empty = sum(1 for t in texts if not t.strip())
    stats = {
        "source": str(path),
        "sha256": sha256_file(path),
        "sha256_matches_registration": sha256_file(path) == expected_sha,
        "documents": len(records),
        "empty_documents": empty,
        "tokens": total_tokens,
        "average_tokens_per_document": total_tokens / max(len(records), 1),
        "vocab_coverage": {
            "unique_tokens_used": len(used),
            "vocab_size": tok.vocab_size,
            "coverage_fraction": len(used) / tok.vocab_size,
            "special_token_documents": {
                str(sid): sum(1 for t in token_lists if sid in t)
                for sid in (0, 1, 2)
            },
        },
        "token_length_percentiles": {
            "p50": sorted(lengths)[len(lengths) // 2] if lengths else 0,
            "p95": sorted(lengths)[int(len(lengths) * 0.95)] if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "duplicates": dup,
    }
    return stats


def dedup_keep_first(texts: list[str]) -> list[int]:
    """Indices of documents surviving exact-normalized-text dedup (first occurrence kept)."""
    seen = set()
    keep = []
    for i, t in enumerate(texts):
        key = " ".join(t.split())
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return keep


def main():
    tok = TokenizerHandle()
    assert tok.vocab_size == config.VOCAB_SIZE, tok.vocab_size
    assert config.BOS_ID == 1 and config.DOC_SEP_ID == 0

    domains = {}
    for name, path, sha in (
        ("A", config.DOMAIN_A_PATH, config.DOMAIN_A_SHA256),
        ("B", config.DOMAIN_B_PATH, config.DOMAIN_B_SHA256),
    ):
        records = load_domain(path)
        texts = [conversation_to_text(r) for r in records]
        token_lists = tok.encode_texts(texts)
        stats = domain_stats(name, path, sha, records, texts, token_lists, tok)
        keep = dedup_keep_first(texts)
        stats["dedup_keep_indices"] = len(keep)
        stats["documents_after_dedup"] = len(keep)
        stats["dedup_rate"] = 1.0 - len(keep) / max(len(records), 1)
        domains[name] = {
            "records": [records[i] for i in keep],
            "texts": [texts[i] for i in keep],
            "token_lists": [token_lists[i] for i in keep],
            "stats": stats,
        }
        tok_total = sum(len(t) for t in domains[name]["token_lists"])
        stats["tokens_after_dedup"] = tok_total
        stats["average_tokens_per_document_after_dedup"] = tok_total / max(len(keep), 1)

    ids_a = [r.get("conversation_id") for r in domains["A"]["records"]]
    ids_b = [r.get("conversation_id") for r in domains["B"]["records"]]
    shared = set(ids_a) & set(ids_b)
    cross = {
        "conversation_ids_A": len(set(ids_a)),
        "conversation_ids_B": len(set(ids_b)),
        "shared_ids": len(shared),
        "duplicate_ids_within_A": len(ids_a) - len(set(ids_a)),
        "duplicate_ids_within_B": len(ids_b) - len(set(ids_b)),
        "note": "shared ids are the translation pairs; content overlap across domains is by design",
    }

    manifest = build_pair_aligned_split(domains["A"]["records"], domains["B"]["records"])
    integrity = verify_split_integrity(manifest, domains["A"]["records"], domains["B"]["records"])

    packing = {}
    for name in ("A", "B"):
        packing[name] = {}
        for split in ("train", "dev", "confirm"):
            idx = manifest[name][split]
            docs = sorted(idx)
            tl = [domains[name]["token_lists"][i] for i in docs]
            windows = pack_domain_stream(tl, config.SEQ_LEN, config.BOS_ID, config.DOC_SEP_ID)
            packing[name][split] = {
                "documents": len(docs),
                "windows": int(windows.shape[0]),
                "effective_loss_tokens": int(windows.shape[0] * (config.SEQ_LEN - 1)),
                "first_window_sha": None,
            }
            if split in ("dev", "confirm"):
                if windows.shape[0] < config.EVAL_POOL_WINDOWS:
                    raise RuntimeError(
                        f"{name}/{split} has {windows.shape[0]} windows < required {config.EVAL_POOL_WINDOWS}"
                    )
                domains[name][f"{split}_windows"] = windows[: config.EVAL_POOL_WINDOWS]
                domains[name][f"{split}_windows_available"] = int(windows.shape[0])

    pools = {}
    for split in ("dev", "confirm"):
        pools[split] = {
            dom: {
                "pool_windows": int(domains[dom][f"{split}_windows"].shape[0]),
                "available_windows": domains[dom][f"{split}_windows_available"],
                "pool_tokens": int(domains[dom][f"{split}_windows"].shape[0]) * (config.SEQ_LEN - 1),
            }
            for dom in ("A", "B")
        }

    report = {
        "gate": "3A+3B",
        "tokenizer": {
            "path": str(config.TOKENIZER_PATH),
            "sha256": sha256_file(config.TOKENIZER_PATH),
            "sha256_matches_registration": sha256_file(config.TOKENIZER_PATH) == config.TOKENIZER_SHA256,
            "vocab_size": tok.vocab_size,
            "packing_rule": {
                "window_prefix_bos_id": config.BOS_ID,
                "document_separator_id": config.DOC_SEP_ID,
                "seq_len": config.SEQ_LEN,
                "effective_loss_tokens_per_window": config.SEQ_LEN - 1,
            },
        },
        "domain_A": domains["A"]["stats"],
        "domain_B": domains["B"]["stats"],
        "cross_domain": cross,
        "split": {
            "version": manifest["split_version"],
            "unit": manifest["unit"],
            "seed": manifest["split_seed"],
            "fractions": manifest["fractions"],
            "line_counts": manifest["counts"],
            "thread_counts": manifest["thread_counts"],
            "threads_total": manifest["threads_total"],
            "threads_shared": manifest["threads_shared"],
            "threads_only_A": manifest["threads_only_A"],
            "threads_only_B": manifest["threads_only_B"],
            "shared_threads_with_equal_line_counts": manifest["shared_threads_with_equal_line_counts"],
            "integrity": integrity,
        },
        "packing": packing,
        "eval_pool_definition": "first EVAL_POOL_WINDOWS windows of each dev/confirm stream, deterministic order",
        "eval_pool_windows_per_domain_per_split": config.EVAL_POOL_WINDOWS,
        "eval_pools": pools,
    }
    report["gate3ab_pass"] = bool(
        domains["A"]["stats"]["sha256_matches_registration"]
        and domains["B"]["stats"]["sha256_matches_registration"]
        and integrity["ok"]
        and report["tokenizer"]["sha256_matches_registration"]
        and all(
            packing[d][s]["effective_loss_tokens"] > 0 for d in ("A", "B") for s in ("train", "dev", "confirm")
        )
        and all(
            packing[d]["train"]["effective_loss_tokens"] >= config.B0_TOKENS * (1.0 - config.SPLIT_FRACTIONS["train"])
            for d in ("A", "B")
        )
    )
    report["single_epoch_feasibility"] = {
        "b0_tokens": config.B0_TOKENS,
        "max_p_single_epoch_A": packing["A"]["train"]["effective_loss_tokens"] / config.B0_TOKENS,
        "max_p_single_epoch_B": packing["B"]["train"]["effective_loss_tokens"] / config.B0_TOKENS,
        "note": "p grid {0.25,0.5,0.75} is feasible single-epoch iff max_p >= 0.75 for both domains",
        "p_grid_feasible_single_epoch": bool(
            packing["A"]["train"]["effective_loss_tokens"] / config.B0_TOKENS >= 0.75
            and packing["B"]["train"]["effective_loss_tokens"] / config.B0_TOKENS >= 0.75
        ),
    }

    out = config.M0_OUTPUTS / "data_audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_report.json").write_text(json.dumps(report, indent=2))
    slim_manifest = {k: v for k, v in manifest.items() if k not in ("A", "B")}
    slim_manifest["full_manifest"] = "split_manifest.json"
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    torch.save(
        {
            "A_dev": domains["A"]["dev_windows"],
            "B_dev": domains["B"]["dev_windows"],
            "A_confirm": domains["A"]["confirm_windows"],
            "B_confirm": domains["B"]["confirm_windows"],
        },
        out / "eval_pools.pt",
    )
    print(json.dumps({k: report[k] for k in ("cross_domain", "split", "packing", "gate3ab_pass")}, indent=2))


if __name__ == "__main__":
    main()
