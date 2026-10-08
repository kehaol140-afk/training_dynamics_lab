"""M0 Gate 2: parameter routing audit. Writes outputs/m0/routing_report.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from muon_transition import config
from muon_transition.model import build_model
from muon_transition.routing import audit_routing


def main():
    model = build_model(seed=config.BASE_SEED_M1_M2)
    report = audit_routing(model)

    qkv_table = [
        t for t in report["routing_table"]
        if t["name"].endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"))
    ]
    report["qkv_routing"] = {
        "rule": "self_attn.{q,k,v,o}_proj.weight -> Muon route (2D hidden linear matrices)",
        "count": len(qkv_table),
        "all_routed_muon": all(t["route"] == "muon" for t in qkv_table),
        "per_layer_shapes": {
            t["name"].split(".")[-3] + "." + t["name"].split(".")[-2]: t["shape"]
            for t in qkv_table if t["name"].startswith("model.layers.0.")
        },
    }
    report["embedding_routing"] = {
        "rule": "model.embed_tokens.weight -> AdamW aux route; lm_head.weight is the same tensor "
                "(tied) and is NOT registered separately",
        "embedding_shape": [6400, 512],
        "tied": report["tied_embedding_is_lm_head"],
        "registered_once": report["tie_registered_once"],
    }
    report["norm_bias_scalar_routing"] = {
        "rule": "all RMSNorm weights, any bias/scalar/vector params -> AdamW aux route",
        "norm_tensor_count": sum(
            1 for t in report["routing_table"] if "norm" in t["name"] and t["route"] == "adamw_aux"
        ),
        "bias_or_scalar_in_muon_route": [
            t["name"] for t in report["routing_table"]
            if t["route"] == "muon" and t["shape"][0] < 8
        ],
    }
    report["model_config"] = config.MODEL_CONFIG
    report["required_fields"] = {
        "total_parameters": report["total_parameters"],
        "muon_parameters": report["muon_parameters"],
        "adamw_aux_parameters": report["adamw_aux_parameters"],
        "overlap_count": report["overlap_count"],
        "missing_count": report["missing_count"],
    }

    ok = (
        report["overlap_count"] == 0
        and report["missing_count"] == 0
        and report["extra_count"] == 0
        and report["counts_add_up"]
        and report["tie_registered_once"]
    )
    report["gate2_pass"] = ok

    out = config.M0_OUTPUTS
    out.mkdir(parents=True, exist_ok=True)
    (out / "routing_report.json").write_text(json.dumps(report, indent=2))
    compact = {k: v for k, v in report.items() if k not in ("routing_table", "model_config")}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
