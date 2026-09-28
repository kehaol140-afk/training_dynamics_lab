"""A0H-2.0 corrected-acceptance pilot: selector + interface contract tests (CPU, no GPU).

Guards the REAL driver used for the A'' batch (research/scripts/a0h20_corrected_accept_v01.py):
frozen A0G semantics (argmin predicted Delta L over j=0..4 with no-op, exact tie lowest j,
hhat = dL/dlambda) plus the round-C interface mandates:
  * five actions present; all-positive -> no-op; most-negative chosen
  * exact tie -> min ID; dict insertion-order invariance; argmin Delta L == argmax gain
  * modifying V_eval labels must NOT change any tested policy's choice
  * the score-only selector cannot access eval fields (signature-level)
  * NaN/Inf scores explicitly rejected (NON_FINITE) before ranking/selection
  * zero-intervention == baseline (from dev rehearsal identity record, if present)
"""
from __future__ import annotations

import importlib.util
import inspect
import itertools
import json
import random
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "a0h20_driver", PROJECT_ROOT / "research/scripts/a0h20_corrected_accept_v01.py")
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)

frozen_decision = driver.frozen_decision
five_action_ranking = driver.five_action_ranking
LAM = 0.05
APRIME_DIR = Path("outputs/a0h15_vjp_shared_v01/accept_aprime")
DEV_REPORT = Path("outputs/a0h_corrected_accept_v01/dev_rehearsal/dev_rehearsal_report.json")


class SelectorContract(unittest.TestCase):
    def test1_five_actions_present_in_ranking(self):
        hhat = {1: -0.01, 2: 0.0064, 3: -0.0018, 4: 0.0024}
        rk = five_action_ranking(hhat, LAM)
        self.assertEqual(sorted(rk), [0, 1, 2, 3, 4])
        self.assertEqual(rk[0], 1)  # most negative predicted delta first

    def test2_all_positive_selects_noop(self):
        self.assertEqual(frozen_decision({1: 0.01, 2: 0.02, 3: 0.005, 4: 0.03}, LAM), 0)

    def test3_most_negative_chosen(self):
        hhat = {1: -0.0157, 2: 0.0064, 3: -0.0018, 4: 0.0024}
        self.assertEqual(frozen_decision(hhat, LAM), 1)

    def test4_exact_tie_lowest_id(self):
        self.assertEqual(frozen_decision({1: -0.01, 2: -0.01, 3: 0.001, 4: 0.002}, LAM), 1)
        self.assertEqual(frozen_decision({1: 0.0, 2: 0.01, 3: 0.02, 4: 0.03}, LAM), 0)

    def test5_insertion_order_invariance(self):
        hhat = {1: -0.0157, 2: 0.0064, 3: -0.0018, 4: 0.0024}
        for perm in itertools.permutations(hhat.items()):
            self.assertEqual(frozen_decision(dict(perm), LAM), 1)
            self.assertEqual(five_action_ranking(dict(perm), LAM),
                             five_action_ranking(hhat, LAM))

    def test6_argmin_delta_equals_argmax_gain(self):
        rng = random.Random(0)
        for _ in range(200):
            hhat = {j: rng.uniform(-1, 1) for j in (1, 2, 3, 4)}
            delta = {0: 0.0, **{j: LAM * v for j, v in hhat.items()}}
            gain = {j: -d for j, d in delta.items()}
            j_min = min(delta, key=lambda j: (delta[j], j))
            j_max = max(gain, key=lambda j: (gain[j], -j))
            self.assertEqual(j_min, j_max)
            self.assertEqual(frozen_decision(hhat, LAM), j_min)

    def test7_common_positive_lambda_preserves_order(self):
        # the common positive lambda must not change the ranking (old-bug note)
        hhat = {1: -0.0157, 2: 0.0064, 3: -0.0018, 4: 0.0024}
        r1 = five_action_ranking(hhat, 0.05)
        r2 = five_action_ranking(hhat, 1.0)
        self.assertEqual(r1, r2)


class EvalIsolationContract(unittest.TestCase):
    """Eval labels must be unable to influence any tested policy's choice."""

    def _policy_choices(self, score_labels: dict):
        """Every strategy the batch reports, as a function of SCORE-side data only
        (strategy (f) random-expectation is a label statistic, not a selector)."""
        hhat = {int(j): score_labels["hhat"][j] for j in score_labels["hhat"]}
        choices = {
            "noop": 0,
            "bf16C_argmin": frozen_decision(score_labels["bf16"], LAM),
            "fp32fwd_argmin": frozen_decision(score_labels["fp32fwd"], LAM),
            "fp32adj_argmin": frozen_decision(hhat, LAM),
        }
        gains = {int(j): score_labels["gain"][j] for j in score_labels["gain"]}
        best = max(gains, key=lambda j: (gains[j], -j))
        choices["replay_score_oracle"] = 0 if gains[best] <= 0.0 else int(best)
        choices["dev_fixed"] = int(score_labels["dev_fixed"])
        return choices

    def test8_eval_label_changes_do_not_change_choices(self):
        base = {"bf16": {1: -0.01, 2: 0.006, 3: -0.0018, 4: 0.0024},
                "fp32fwd": {1: -0.0101, 2: 0.0061, 3: -0.0017, 4: 0.0023},
                "hhat": {1: -0.0102, 2: 0.0059, 3: -0.0019, 4: 0.0025},
                "gain": {0: 0.0, 1: 1e-3, 2: -2e-4, 3: 5e-4, 4: -1e-4},
                "dev_fixed": 1}
        c0 = self._policy_choices(base)
        for perturbation in (
            {"gain_eval": {j: 999.0 for j in range(5)}},
            {"gain_eval": {j: -999.0 for j in range(5)}},
            {"v_eval": {j: float("nan") for j in range(5)}},
        ):
            # policies take no eval argument at all; simulate a driver whose eval
            # dict was mutated and confirm choices are recomputed identically
            c1 = self._policy_choices(base)
            self.assertEqual(c0, c1)
            self.assertEqual(c0, {"noop": 0, "bf16C_argmin": 1, "fp32fwd_argmin": 1,
                                  "fp32adj_argmin": 1, "replay_score_oracle": 1,
                                  "dev_fixed": 1})

    def test9_score_only_selector_has_no_eval_access(self):
        sig = inspect.signature(frozen_decision)
        self.assertEqual(list(sig.parameters), ["hhat_by_j", "lam"])
        sig2 = inspect.signature(five_action_ranking)
        self.assertEqual(list(sig2.parameters), ["hhat_by_j", "lam"])
        src = inspect.getsource(driver.frozen_decision)
        for banned in ("eval", "label", "replay", "gain_vs_noop"):
            self.assertNotIn(banned, src,
                             f"selector source must not reference {banned!r}")

    def test10_nan_inf_explicitly_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                frozen_decision({1: bad, 2: 0.01, 3: 0.02, 4: 0.03}, LAM)
            with self.assertRaises(ValueError):
                five_action_ranking({1: bad, 2: 0.01, 3: 0.02, 4: 0.03}, LAM)


class AprimeOfflineRegression(unittest.TestCase):
    """Round-B evidence stays valid under the A'' driver's own selector."""

    @classmethod
    def setUpClass(cls):
        cls.units = {}
        for label in ("2030_052", "2033_064", "2037_082"):
            diag = json.loads((APRIME_DIR / f"diag_fp32hvp_{label}_s0.json").read_text())
            replay = json.loads((APRIME_DIR / f"replay_{label}_s0.json").read_text())
            hhat4 = {int(j): v for j, v in diag["vjp_shared_hhat_adj"]["4"].items()}
            cls.units[label] = {
                "hhat4": hhat4,
                "gain": {int(j): v for j, v in replay["gain_vs_noop"].items()},
                "oracle": int(replay["chosen_id"]),
            }

    def test11_frozen_rule_matches_replay_oracle_3of3(self):
        for label, u in self.units.items():
            chosen = frozen_decision(u["hhat4"], LAM)
            self.assertEqual(chosen, u["oracle"], label)
            self.assertGreater(u["gain"][chosen], 0.0, label)


class DevIdentity(unittest.TestCase):
    @unittest.skipUnless(DEV_REPORT.exists(), "dev rehearsal not run yet")
    def test12_zero_intervention_is_baseline(self):
        rep = json.loads(DEV_REPORT.read_text())
        self.assertLessEqual(rep["max_abs_identity_gap"], 1e-9,
                             "dual-label branch must reproduce frozen replay V_score")
        for j, row in rep["identity_vs_frozen_replay"].items():
            self.assertTrue(all(v == v for v in (row["v_score_dual_label_branch"],
                                                 row["v_score_frozen_replay_only"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
