from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fr.e4b_data_audit import build_audit


class E4BDataAuditTest(unittest.TestCase):
    def _make_project(self, future_q: bool) -> Path:
        td = Path(tempfile.mkdtemp())
        for rel in [
            "research/src/fr/full_state.py",
            "research/src/fr/theta_protocol.py",
            "research/src/fr/native_schedule.py",
            "research/src/fr/batch_manifest.py",
            "research/src/fr/stage_c_runtime.py",
        ]:
            p = td / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# synthetic\n", encoding="utf-8")
        seed_dir = td / "outputs/e3_confirmatory/v07/seed_2030"
        (seed_dir / "states").mkdir(parents=True, exist_ok=True)
        for s in [46, 52]:
            (seed_dir / "states" / f"state_step_{s:03d}.pt").write_bytes(b"state")
        payload = {
            "q_sha256": "abc",
            "future_stream": {"manifest_path": "m.json", "rng_seed": 123},
            "paired": {"plus": True, "minus": True},
        }
        if future_q:
            payload["future_q_sha256"] = "def"
        (seed_dir / "theta_mapping_raw.json").write_text(json.dumps(payload), encoding="utf-8")
        (seed_dir / "state_bank_report.json").write_text(json.dumps({"technical_pass": True}), encoding="utf-8")
        return td

    def test_existing_future_q_needs_no_replay(self):
        root = self._make_project(True)
        r = build_audit(root, [2030], [46, 52], [1, 2])
        self.assertEqual(r["decision"]["E4B_EXISTING_DATA_SUFFICIENT_FOR_Q_ROTATION"], "YES")
        self.assertEqual(r["decision"]["E4B_BASELINE_REPLAY_REQUIRED"], "NO")

    def test_missing_future_q_requests_replay(self):
        root = self._make_project(False)
        r = build_audit(root, [2030], [46, 52], [1, 2])
        self.assertEqual(r["decision"]["E4B_EXISTING_DATA_SUFFICIENT_FOR_Q_ROTATION"], "NO")
        self.assertEqual(r["decision"]["E4B_BASELINE_REPLAY_REQUIRED"], "YES")


if __name__ == "__main__":
    unittest.main()
