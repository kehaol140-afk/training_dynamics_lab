import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from e3r_multiseed_theta_v06 import write_multiseed_theta_summary


class TestE3RMultiseedThetaSummary(unittest.TestCase):
    def test_partial_rerun_summary_preserves_existing_passed_seeds(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            for seed in (2026, 2027):
                seed_dir = out / f"seed_{seed}"
                seed_dir.mkdir(parents=True)
                (seed_dir / "theta_mapping_raw.json").write_text(
                    json.dumps({
                        "technical_pass": True,
                        "rho0_noop_pass": True,
                        "train_seed": seed,
                    }),
                    encoding="utf-8",
                )

            summary = write_multiseed_theta_summary(
                out,
                all_seeds=[2026, 2027, 2028],
                requested_seeds=[2027],
                completed_this_run=[2027],
            )

            self.assertEqual(summary["seeds"], [2026, 2027, 2028])
            self.assertEqual(summary["requested_seeds"], [2027])
            self.assertEqual(summary["completed"], [2026, 2027])
            self.assertEqual(summary["completed_this_run"], [2027])
            self.assertFalse(summary["technical_pass"])
            saved = json.loads((out / "multiseed_theta_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, summary)


if __name__ == "__main__":
    unittest.main()
