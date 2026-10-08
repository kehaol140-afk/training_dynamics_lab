import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from muon_transition.checkpoint import load_training_state, parameter_fingerprint, save_training_state
from muon_transition.model import build_model
from muon_transition.optimizers import build_adamw_arm, build_muon_arm
from muon_transition.train_loop import train_step


def _batches(seed, n, bs, seq, vocab):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(3, vocab, (bs, seq), generator=g) for _ in range(n)]


def _run(arm, steps, batches, seed):
    model = build_model(seed=2030)
    if arm == "adamw":
        opts = [build_adamw_arm(model, lr=3e-4, weight_decay=0.1)]
    else:
        muon_opt, aux_opt = build_muon_arm(model, lr=0.02, weight_decay=0.1)
        opts = [muon_opt, aux_opt]
    losses = []
    for i in range(steps):
        losses.append(train_step(model, opts, batches[i]))
    return model, opts, losses


def test_reproducibility_and_checkpoint():
    for arm in ("adamw", "muon"):
        batches = _batches(99, 4, 4, 64, 6400)

        m1, o1, l1 = _run(arm, 3, batches, 2030)
        m2, o2, l2 = _run(arm, 3, batches, 2030)
        assert l1 == l2, (arm, l1, l2)
        assert parameter_fingerprint(m1) == parameter_fingerprint(m2)

        m3, o3, _ = _run(arm, 2, batches, 2030)
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "state.pt"
            save_training_state(ckpt, m3, o3, cursor={"batch_index": 2, "stream": "test"})
            m4 = build_model(seed=555)
            if arm == "adamw":
                o4 = [build_adamw_arm(m4, lr=3e-4, weight_decay=0.1)]
            else:
                mu, aux = build_muon_arm(m4, lr=0.02, weight_decay=0.1)
                o4 = [mu, aux]
            cursor = load_training_state(ckpt, m4, o4)
            assert cursor["batch_index"] == 2
            l4 = train_step(m4, o4, batches[2])
            assert abs(l4["loss"] - l1[2]["loss"]) < 1e-12, (arm, l4, l1[2])
            assert parameter_fingerprint(m4) == parameter_fingerprint(m1)

            m5, o5, _ = _run(arm, 3, batches, 2030)
            save_training_state(Path(td) / "s2.pt", m5, o5, cursor={"i": 3})
            rand_ref = torch.rand(4)
            m6 = build_model(seed=777)
            if arm == "adamw":
                o6 = [build_adamw_arm(m6, lr=3e-4, weight_decay=0.1)]
            else:
                mu6, aux6 = build_muon_arm(m6, lr=0.02, weight_decay=0.1)
                o6 = [mu6, aux6]
            load_training_state(Path(td) / "s2.pt", m6, o6)
            rand_after = torch.rand(4)
            assert torch.equal(rand_ref, rand_after), "RNG state not restored"
        print(f"test_repro_and_checkpoint OK [{arm}] losses={[x['loss'] for x in l1]}")


if __name__ == "__main__":
    test_reproducibility_and_checkpoint()
