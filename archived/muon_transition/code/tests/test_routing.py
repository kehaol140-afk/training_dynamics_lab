import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from muon_transition import config
from muon_transition.model import build_model
from muon_transition.routing import audit_routing, route_parameters


def test_routing():
    model = build_model(seed=2030)
    report = audit_routing(model)

    assert report["total_parameters"] == 30_025_216, report["total_parameters"]
    assert report["overlap_count"] == 0
    assert report["missing_count"] == 0
    assert report["extra_count"] == 0
    assert report["counts_add_up"]
    assert report["muon_parameters"] + report["adamw_aux_parameters"] == report["total_parameters"]
    assert report["tied_embedding_is_lm_head"]
    assert report["tie_registered_once"]

    muon_params, aux_params, table = route_parameters(model)
    assert len(muon_params) == 56, len(muon_params)

    qkv = [t for t in table if t["name"].endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight"))]
    assert len(qkv) == 24 and all(t["route"] == "muon" for t in qkv)

    norms = [t for t in table if "norm" in t["name"] and t["route"] != "muon"]
    assert len(norms) == 34 - 1
    assert all(t["numel"] < 1024 for t in norms)

    aux_names = [t["name"] for t in table if t["route"] == "adamw_aux"]
    assert "model.embed_tokens.weight" in aux_names
    assert not any(n == "lm_head.weight" for n in aux_names)

    embed = [t for t in table if t["name"] == "model.embed_tokens.weight"][0]
    assert embed["numel"] == 512 * 6400

    print("test_routing OK:", {k: v for k, v in report.items() if k != "routing_table"})


if __name__ == "__main__":
    test_routing()
