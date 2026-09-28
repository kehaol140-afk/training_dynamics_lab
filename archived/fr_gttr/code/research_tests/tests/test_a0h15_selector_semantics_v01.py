"""A0H-1.5 round-3 decision-semantics audit: frozen A0G selector contract tests.

Read-only semantics tests (CPU, no GPU). Guards the restored frozen selector in
research/scripts/a0h15_accept_aprime_v01.frozen_decision against the round-2
regression (raw-hhat argmax instead of argmin predicted Delta L).

Tests required by the round-3 mandate (Branch A, item 4):
  1. all four active hhat > 0  -> frozen rule selects no-op (j=0)
  2. exists hhat < 0           -> selects most negative hhat
  3. argmin(predicted_delta_loss) == argmax(predicted_gain)
  4. shuffled CSV/dict row order -> action ID unchanged
  5. tie -> frozen tie-break (lowest j)
Plus offline re-scoring of the saved A' labels (no GPU):
  6. frozen rule on saved A' hhat == replay gain-argmax oracle, 3/3 units
  7. the round-2 raw-argmax program choice != oracle on the same labels
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from fr.a0g_decision_utility_v01 import select_action as frozen_select_action

LAM = 0.05  # FROZEN_LAMBDA (A0G Rule B, global)
APRIME_DIR = Path("outputs/a0h15_vjp_shared_v01/accept_aprime")


def frozen_decision(hhat_by_j, lam=LAM):
    """Same construction as the fixed A' driver (kept inline to avoid importing
    the GPU driver in a unit test)."""
    predicted = {0: 0.0}
    for j, v in hhat_by_j.items():
        predicted[int(j)] = float(lam) * float(v)
    return frozen_select_action(predicted)


class FrozenSelectorContract(unittest.TestCase):
    def test1_all_positive_selects_noop(self):
        hhat = {1: 0.01, 2: 0.02, 3: 0.005, 4: 0.03}
        self.assertEqual(frozen_decision(hhat), 0)

    def test2_negative_present_selects_most_negative(self):
        hhat = {1: -0.0157, 2: 0.0064, 3: -0.0018, 4: 0.0024}
        self.assertEqual(frozen_decision(hhat), 1)

    def test3_argmin_delta_equals_argmax_gain(self):
        import random
        rng = random.Random(0)
        for _ in range(200):
            hhat = {j: rng.uniform(-1, 1) for j in (1, 2, 3, 4)}
            delta = {0: 0.0, **{j: LAM * v for j, v in hhat.items()}}
            gain = {j: -d for j, d in delta.items()}
            j_min = min(delta, key=lambda j: (delta[j], j))
            j_max = max(gain, key=lambda j: (gain[j], -j))
            self.assertEqual(j_min, j_max)
            self.assertEqual(frozen_decision(hhat), j_min)

    def test4_order_invariance(self):
        hhat = {1: -0.0157, 2: 0.0064, 3: -0.0018, 4: 0.0024}
        import itertools
        for perm in itertools.permutations(hhat.items()):
            self.assertEqual(frozen_decision(dict(perm)), 1)

    def test5_tie_breaks_to_lowest_j(self):
        # active minimum tied between j=1/j=2 -> lowest j wins
        hhat = {1: -0.01, 2: -0.01, 3: 0.001, 4: 0.002}
        self.assertEqual(frozen_decision(hhat), 1)
        # no-op tied with an active zero-response candidate -> lowest j (no-op=0)
        hhat2 = {1: 0.0, 2: 0.01, 3: 0.02, 4: 0.03}
        self.assertEqual(frozen_decision(hhat2), 0)


class AprimeOfflineRescoring(unittest.TestCase):
    """Re-score the SAVED A' labels (read-only, no GPU)."""

    @classmethod
    def setUpClass(cls):
        cls.units = {}
        for label in ("2030_052", "2033_064", "2037_082"):
            diag = json.loads((APRIME_DIR / f"diag_fp32hvp_{label}_s0.json").read_text())
            replay = json.loads((APRIME_DIR / f"replay_{label}_s0.json").read_text())
            verdict = json.loads((APRIME_DIR / f"verdict_{label}_s0.json").read_text())
            hhat4 = {int(j): v for j, v in diag["vjp_shared_hhat_adj"]["4"].items()}
            cls.units[label] = {
                "hhat4": hhat4,
                "gain": {int(j): v for j, v in replay["gain_vs_noop"].items()},
                "oracle": int(replay["chosen_id"]),
                "old_choice": int(verdict["chosen_action_fp32_adj"]),
                "old_gain": float(verdict["L2_replay"]["v_score_gain"]),
            }

    def test6_frozen_rule_matches_replay_oracle(self):
        for label, u in self.units.items():
            chosen = frozen_decision(u["hhat4"])
            self.assertEqual(chosen, u["oracle"],
                             f"{label}: frozen {chosen} != oracle {u['oracle']}")
            self.assertGreater(u["gain"][chosen], 0.0,
                               f"{label}: frozen-rule gain must be positive")

    def test7_round2_raw_argmax_was_anti_selector(self):
        for label, u in self.units.items():
            raw_argmax = max(u["hhat4"], key=lambda j: (u["hhat4"][j], -j))
            self.assertEqual(raw_argmax, u["old_choice"])
            self.assertNotEqual(u["old_choice"], u["oracle"])
            self.assertLess(u["old_gain"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
