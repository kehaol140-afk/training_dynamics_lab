"""M0 Gate 1: freeze torch.optim.Muon provenance + self-check. Writes outputs/m0/muon_freeze.json."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from muon_transition import config


def main():
    import torch.optim as O

    spec = importlib.util.find_spec("torch.optim._muon")
    file_path = Path(spec.origin)
    h = hashlib.sha256(file_path.read_bytes()).hexdigest()

    from torch.optim import Muon

    record = {
        "gate": 1,
        "primary_source": "torch.optim._muon.Muon (exposed as torch.optim.Muon)",
        "repository": f"PyTorch installed wheel, torch=={torch.__version__} (cuda {torch.version.cuda})",
        "commit_hash": "n/a (binary wheel; provenance chain via in-file permalink)",
        "upstream_permalink_in_file": "https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L16",
        "files_imported": [str(file_path)],
        "sha256": h,
        "frozen_constants": config.MUON_FREEZE,
        "secondary_local_sources": [
            {
                "source": "/home/b205/framework-audit/llamafactory third_party muon (Moonlight/KellerJordan hybrid)",
                "repo_commit": "7af909522a951e3ad9f022ea6f88b6755257eaa5",
                "status": "registered alternative, not used",
            },
            {
                "source": "/home/b205/baselines/skewadam train.py Muon variant",
                "repo_commit": "50f3da6bdb0d527c640ee60ec22302af5d4148eb",
                "status": "rejected (no nesterov, no scale rule, hardcoded shape heuristics)",
            },
        ],
        "checks": {},
    }

    sig = str(Muon.__init__.__doc__ or "")[:0]
    p = torch.randn(64, 32)
    opt = Muon([p], lr=0.02, momentum=0.95, nesterov=True, ns_steps=5)
    p.grad = torch.randn(64, 32)
    opt.step()
    sd = opt.state_dict()
    state_key = next(iter(sd["state"]))
    record["checks"]["steps_on_random_tensor"] = True
    record["checks"]["state_keys"] = sorted(sd["state"][state_key].keys())
    record["checks"]["group_defaults"] = {
        k: sd["param_groups"][0][k]
        for k in ("lr", "momentum", "nesterov", "ns_steps", "ns_coefficients", "eps", "weight_decay", "adjust_lr_fn")
    }

    out = config.M0_OUTPUTS
    out.mkdir(parents=True, exist_ok=True)
    (out / "muon_freeze.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({k: v for k, v in record.items() if k != "secondary_local_sources"}, indent=2, default=str))


if __name__ == "__main__":
    main()
