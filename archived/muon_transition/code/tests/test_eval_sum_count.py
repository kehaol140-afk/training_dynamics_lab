import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn

from muon_transition.eval_nll import domain_nll_sum_count, evaluate_q


class FakeLM(nn.Module):
    """Returns logits independent of input so per-token CE depends only on the target."""

    def __init__(self, vocab=5, seq=4):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(1, 1, vocab))

    def forward(self, ids, **kw):
        b, s = ids.shape
        out = self.logits.expand(b, s, -1)
        from types import SimpleNamespace
        return SimpleNamespace(logits=out)


def _manual_nll(windows, vocab=5):
    torch.manual_seed(0)
    logits = torch.randn(vocab).double()
    logp = torch.log_softmax(logits, dim=0)
    total, count = 0.0, 0
    for w in windows.tolist():
        for t in w[1:]:
            total += -float(logp[t])
            count += 1
    return total, count


def test_sum_count_semantics():
    model = FakeLM()
    torch.manual_seed(0)
    model.logits.data = torch.randn(1, 1, 5).double().float()
    logits = model.logits.detach().double().reshape(-1)
    logp = torch.log_softmax(logits, dim=0)

    windows = torch.tensor(
        [[0, 1, 2, 3], [0, 4, 4, 4], [0, 1, 1, 2], [0, 3, 3, 1], [0, 2, 2, 2]]
    )
    res = domain_nll_sum_count(model, windows, batch_size=2)
    total = 0.0
    count = 0
    for w in windows.tolist():
        for t in w[1:]:
            total += -float(logp[t])
            count += 1
    assert res["token_count"] == count == 5 * 3
    assert abs(res["nll_sum"] - total) < 1e-3

    per_batch_means = []
    for i in range(0, 5, 2):
        chunk = windows[i : i + 2]
        sc = domain_nll_sum_count(model, chunk, batch_size=2)
        per_batch_means.append(sc["nll_sum"] / sc["token_count"])
    mean_of_means = sum(per_batch_means) / len(per_batch_means)
    sum_over_count = res["nll_sum"] / res["token_count"]
    assert abs(mean_of_means - sum_over_count) > 1e-6, (
        "unequal-content batches should make mean-of-means differ from sum/count"
    )

    pools = {"A": windows, "B": windows}
    q = evaluate_q(model, pools)
    assert q["q"] == 0.5
    assert abs(q["v_q"] - 0.5 * q["nll_A"] - 0.5 * q["nll_B"]) < 1e-12
    assert abs(q["nll_A"] - sum_over_count) < 1e-6
    assert abs(q["nll_B"] - sum_over_count) < 1e-6
    print("test_sum_count_semantics OK: sum/count =", sum_over_count,
          "mean-of-means =", mean_of_means)


if __name__ == "__main__":
    test_sum_count_semantics()
