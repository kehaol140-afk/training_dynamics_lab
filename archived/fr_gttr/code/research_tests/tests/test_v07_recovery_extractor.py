import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

try:
    from e3c_extract_locked_confirmation_v07 import (
        flatten_confirmation_by_horizon,
        load_confirmation_seed_results,
    )
except ImportError:
    flatten_confirmation_by_horizon = None
    load_confirmation_seed_results = None


def write_seed_result(out, seed):
    p = out / f"seed_{seed}/theta_mapping_raw.json"
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps({
            "protocol": "E3C_CONFIRMATORY_V07_MULTISEED_THETA",
            "technical_pass": True,
            "rho0_noop_pass": True,
            "train_seed": seed,
            "states": [
                {
                    "step": 46,
                    "state_fraction": 46 / 128,
                    "B_theta": {"1": 1.0 + seed, "8": 2.0 + seed},
                    "chi_theta": {"1": 3.0 + seed, "8": 4.0 + seed},
                    "odd_error_theta": {"1": 0.1, "8": 0.2},
                    "clipping": {
                        "baseline_rate": 0.0,
                        "plus_rate": 0.0,
                        "minus_rate": 0.0,
                    },
                }
            ],
        }),
        encoding="utf-8",
    )


class TestV07RecoveryExtractor(unittest.TestCase):
    def test_loads_only_confirmation_seed_results_and_flattens_horizons(self):
        # 会被这个测试抓住的 bug：recovery extractor 读取 discovery seeds/path，
        # 或把非 confirmation seed 混入 confirmation rows。
        self.assertIsNotNone(load_confirmation_seed_results, "missing recovery extractor")
        self.assertIsNotNone(flatten_confirmation_by_horizon, "missing recovery extractor")

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "outputs/e3_confirmatory/v07"
            write_seed_result(out, 2030)
            write_seed_result(out, 2031)

            raw = load_confirmation_seed_results(out, [2030, 2031])
            rows = flatten_confirmation_by_horizon(raw, [1, 8], expected_states=[46])

        self.assertEqual([x["train_seed"] for x in raw], [2030, 2031])
        self.assertEqual(sorted(rows), ["1", "8"])
        self.assertEqual(len(rows["8"]), 2)
        self.assertEqual({r["source_role"] for r in rows["8"]}, {"CONFIRMATION_PRIMARY_ELIGIBLE"})
        self.assertEqual({r["train_seed"] for r in rows["8"]}, {2030, 2031})


if __name__ == "__main__":
    unittest.main()
