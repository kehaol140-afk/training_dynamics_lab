#!/usr/bin/env python3
"""Determinism probe: same cell, same batch, compute h=1 grads+loss in one
process, print full precision. Run in two processes and diff."""
import copy, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
LAB = ROOT.parent
MM = LAB / "minimind_base"
sys.path.insert(0, str(MM))
import torch
import opsens.f0_bank_audit_v01 as f0
import opsens.f1_mechanism_v01 as fm
from f2_hypergradient import Rig, BATCH_SIZE, ACCUM, MAX_NORM, CELL_STREAM
from opsens.batch_manifest import manifest_sha256

torch.set_num_threads(min(32, torch.get_num_threads()))
torch.manual_seed(0)
rig = Rig("cuda:0")
batches, stream_seed = rig.batches_for(CELL_STREAM[(2042, 26)])
state = f0.load_bank_state(fm.bank_state_path("g1b_100m", 2042, 26))
named, opt, model_cpu = rig.restore(state)
grads, loss = rig.compute_grads(model_cpu, batches, 0)
print("manifest", manifest_sha256(batches)[:16])
print("loss", repr(loss))
for n in list(grads)[:3]:
    g = grads[n]
    print(n, float(g.sum()), float(g.square().sum()))
