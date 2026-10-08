"""T2b/T3-mini shared components: Q-target split, manifests, permutations, evaluation.

Round-3 (ROUND3_TARGET_SCREEN_OPENCODE_CN.md). All conventions follow the task spec.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
sys.path.insert(0, str(LAB / "muon_transition_experiment"))
sys.path.insert(0, str(LAB / "muon_transition_experiment" / "src"))
sys.path.insert(0, str(LAB / "minimind_base"))

from muon_transition import config as mte_config
from muon_transition.data import TokenizerHandle, conversation_to_text, load_domain, pack_domain_stream
from muon_transition.partitions import build_pair_aligned_split

Q_CACHE = Path("/tmp/opencode/t2b_q_targets.pt")
Q_VERSION = "T2b_Q_v1"
WINDOWS_PER_TARGET = 512
ANCHOR_RUN = "m1_muon_lr2.0_wd0.02_p0.5_s2030"
ANCHOR_PATH = (LAB / "muon_transition_experiment" / "outputs" / "m1" / ANCHOR_RUN / "final_state.pt")
ANCHOR_SHA = "4f68ce2423d663f87fbb44f051abe92d6ccd09118ca42a80261497ee13f095e4"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def q_side_of_group(group_id: str) -> int:
    return hashlib.sha256((Q_VERSION + "|" + str(group_id)).encode()).digest()[0] % 2


def build_q_targets(verbose: bool = True):
    """Split the OLD dev pool by translation-connected thread group into Q_select/Q_check.

    Returns dict: side -> domain -> {"windows": LongTensor (n,256), "doc_indices", "group_ids",
    "n_docs", "n_windows", "token_count"} plus split metadata. Cached at Q_CACHE.
    """
    if Q_CACHE.exists():
        blob = torch.load(Q_CACHE, weights_only=True)
        return blob
    sys.path.insert(0, str(LAB / "muon_transition_experiment"))
    from scripts.m0_gate3_data_audit import dedup_keep_first
    tok = TokenizerHandle()
    out = {"version": Q_VERSION, "sides": {}}
    deduped = {}
    for name, path in (("A", mte_config.DOMAIN_A_PATH), ("B", mte_config.DOMAIN_B_PATH)):
        records = load_domain(path)
        texts = [conversation_to_text(r) for r in records]
        keep = dedup_keep_first(texts)
        deduped[name] = ([records[i] for i in keep], [texts[i] for i in keep])
    manifest = build_pair_aligned_split(deduped["A"][0], deduped["B"][0])
    side_meta = {"select": {"A": [], "B": []}, "check": {"A": [], "B": []}}
    for dom in ("A", "B"):
        records, texts = deduped[dom]
        dev_sorted = manifest[dom]["dev"]
        for i in dev_sorted:
            gid = records[i].get("conversation_id")
            s = q_side_of_group(gid)
            side_meta["select" if s == 0 else "check"][dom].append(i)
    for side_name in ("select", "check"):
        dom_out = {}
        for dom in ("A", "B"):
            idx = sorted(side_meta[side_name][dom])
            recs, texts = deduped[dom]
            sel_texts = [texts[i] for i in idx]
            encs = tok.encode_texts(sel_texts)
            windows = pack_domain_stream(encs, mte_config.SEQ_LEN, mte_config.BOS_ID, mte_config.DOC_SEP_ID)
            n_win = min(WINDOWS_PER_TARGET, windows.shape[0])
            windows = windows[:n_win]
            gids = [recs[i].get("conversation_id") for i in idx]
            dom_out[dom] = {"windows": windows, "doc_indices": idx, "group_ids": gids,
                            "n_docs": len(idx), "n_windows": int(n_win),
                            "token_count": int(n_win * (mte_config.SEQ_LEN - 1))}
            if verbose:
                print(f"Q_{side_name}[{dom}]: docs={len(idx)} windows_used={n_win} "
                      f"tokens={dom_out[dom]['token_count']}")
            if n_win == 0:
                raise RuntimeError(f"empty target domain {side_name}/{dom} — BLOCKED per protocol")
        out["sides"][side_name] = dom_out
    # group isolation check: no group appears in both sides
    for dom in ("A", "B"):
        gs = set(out["sides"]["select"][dom]["group_ids"])
        gc = set(out["sides"]["check"][dom]["group_ids"])
        assert gs.isdisjoint(gc), f"group leaked across Q sides in {dom}"
    out["group_isolation_ok"] = True
    torch.save(out, Q_CACHE)
    return out


def eval_q_checkpoints(model, q_blob, side: str, device: str, batch_size: int = 32):
    """V = 0.5*NLL_A + 0.5*NLL_B with per-domain sum/count. Returns dict."""
    was_training = model.training
    model.eval()
    res = {}
    with torch.no_grad():
        for dom in ("A", "B"):
            wins = q_blob["sides"][side][dom]["windows"]
            nll_sum = torch.zeros((), dtype=torch.float64)
            count = 0
            for i in range(0, wins.shape[0], batch_size):
                batch = wins[i:i + batch_size].to(device)
                logits = model(batch).logits[..., :-1, :].contiguous()
                targets = batch[..., 1:].contiguous()
                ls = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                     targets.reshape(-1), reduction="sum")
                nll_sum += ls.double().cpu()
                count += int(targets.numel())
            res[f"nll_sum_{dom}"] = float(nll_sum)
            res[f"count_{dom}"] = count
            res[f"nll_{dom}"] = float(nll_sum) / count
    res["V"] = 0.5 * res["nll_A"] + 0.5 * res["nll_B"]
    if was_training:
        model.train()
    return res


def q_select_gradient(model, q_blob, device: str, batch_size: int = 32):
    """Gradient of V_Qselect at current theta* (no clip, no write-back). Returns
    {name: grad fp32 cpu tensor} keyed by unique named parameters, plus V value."""
    side = "select"
    model.eval()
    grads = None
    V_parts = {"A": (0.0, 0), "B": (0.0, 0)}
    # first pass: counts (targets per domain)
    counts = {}
    for dom in ("A", "B"):
        counts[dom] = q_blob["sides"][side][dom]["n_windows"] * (mte_config.SEQ_LEN - 1)
    for p in model.parameters():
        p.grad = None
    for dom in ("A", "B"):
        wins = q_blob["sides"][side][dom]["windows"]
        w = 0.5 / counts[dom]
        for i in range(0, wins.shape[0], batch_size):
            batch = wins[i:i + batch_size].to(device)
            logits = model(batch).logits[..., :-1, :].contiguous()
            targets = batch[..., 1:].contiguous()
            ls = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                 targets.reshape(-1), reduction="sum")
            (w * ls).backward()
            V_parts[dom] = (V_parts[dom][0] + float(ls.detach()), V_parts[dom][1] + int(targets.numel()))
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().cpu().clone()
        else:
            grads[name] = torch.zeros_like(p).cpu()
    for p in model.parameters():
        p.grad = None
    model.train()
    V = 0.5 * (V_parts["A"][0] / V_parts["A"][1]) + 0.5 * (V_parts["B"][0] / V_parts["B"][1])
    return grads, V


def build_round3_manifests():
    """W0=[2048:2058), W1=[2058:2068), C=[2068:2196) from the production seed-2030 stream."""
    from muon_transition.m1_data_cache import get_domain_windows
    from muon_transition.mixture import build_step_manifest
    a_train, b_train = get_domain_windows()
    m = build_step_manifest(a_train, b_train, 0.5, 2196, seed=2030)
    return {"W0": m["steps"][2048:2058], "W1": m["steps"][2058:2068],
            "C": m["steps"][2068:2196]}


def generate_permutations():
    """Eight per-window 10-element uniform random permutations per the task formula.

    PCG64(883000 + 100*b + k).permutation(10), b in {0,1}, k in 0..7. Collision rule:
    seed += 10000 and regenerate until unique within the window, decided before any run.
    """
    perms = {}
    for b in (0, 1):
        seen = set()
        lst = []
        for k in range(8):
            seed = 883000 + 100 * b + k
            while True:
                p = tuple(int(x) for x in np.random.Generator(np.random.PCG64(seed)).permutation(10))
                if p not in seen:
                    seen.add(p)
                    lst.append({"k": k, "seed": seed, "perm": list(p)})
                    break
                seed += 10000  # pre-declared collision rule
        perms[f"W{b}"] = lst
    return perms


def flat_state_hash(model, opts) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    for opt in opts:
        for k in sorted(opt.state.keys(), key=str):
            st = opt.state[k]
            for kk in sorted(st.keys()):
                v = st[kk]
                if torch.is_tensor(v):
                    h.update(v.detach().cpu().contiguous().numpy().tobytes())
                else:
                    h.update(str(v).encode())
    return h.hexdigest()
