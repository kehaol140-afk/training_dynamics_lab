import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from scientific_preflight_v07 import (
    DEPENDENCY_RELATIVE_PATHS,
    LOCKED_RELATIVE_PATHS,
    build_prereg_lock_payload,
    compare_lock_to_current,
    formal_confirmatory_scientific_data_files,
    locked_file_hashes,
)


ROOT = Path(__file__).resolve().parents[2]
OLD_LOCK_SHA = "4053024c178fb061e7c8a3b4a9b6c89908a5034c07c390b75796b5e09410a81b"


EXPECTED_LOCKED = {
    "research/configs/e3c_confirmatory_v07.json",
    "research/src/fr/confirmatory_stats.py",
    "research/scripts/e3c_analyze_confirmatory_v07.py",
    "CONFIRMATORY_PREREGISTRATION_V07.md",
    "research/scripts/e3c_prepare_state_bank_v07.py",
    "research/scripts/e3c_multiseed_theta_v07.py",
    "research/src/fr/theta_protocol.py",
    "research/src/fr/full_state.py",
    "research/src/fr/rng.py",
    "research/src/fr/direction.py",
    "research/src/fr/perturbation.py",
    "research/src/fr/native_schedule.py",
    "research/src/fr/stream_rng.py",
    "research/src/fr/batch_manifest.py",
    "research/src/fr/pilot_partition.py",
    "research/src/fr/stage_c_runtime.py",
    "research/src/fr/response_stats.py",
}

EXPECTED_DEPENDENCIES = {
    "minimind/trainer/train_pretrain.py",
    "minimind/trainer/trainer_utils.py",
    "minimind/dataset/lm_dataset.py",
    "minimind/model/model_minimind.py",
}


class TestV07PreregLock(unittest.TestCase):
    def test_revision2_lock_payload_covers_formal_code_and_readonly_dependencies(self):
        # 会被这个测试抓住的 bug：只锁原来的 4 个文件，漏锁数据生成脚本、
        # FR 主路径依赖或 MiniMind baseline dependency。
        cfg = json.loads((ROOT / "research/configs/e3c_confirmatory_v07.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            payload = build_prereg_lock_payload(ROOT, out, cfg, previous_lock_sha256=OLD_LOCK_SHA)

        self.assertEqual(payload["lock_revision"], 2)
        self.assertEqual(payload["previous_lock_sha256"], OLD_LOCK_SHA)
        self.assertFalse(payload["formal_confirmatory_data_existed_before_revision"])
        self.assertEqual({x["path"] for x in payload["locked_files"]}, EXPECTED_LOCKED)
        self.assertEqual({x["path"] for x in payload["dependency_files"]}, EXPECTED_DEPENDENCIES)
        self.assertEqual(set(LOCKED_RELATIVE_PATHS), EXPECTED_LOCKED)
        self.assertEqual(set(DEPENDENCY_RELATIVE_PATHS), EXPECTED_DEPENDENCIES)

    def test_compare_lock_fails_when_dependency_sha_changes(self):
        # 会被这个测试抓住的 bug：preflight 只校验 locked_files，忽略 dependency_files。
        with tempfile.TemporaryDirectory() as td:
            lock_path = Path(td) / "PREREG_LOCK.json"
            payload = {
                "protocol": "E3C_CONFIRMATORY_V07_PREREG_LOCK",
                "lock_revision": 2,
                "locked_files": locked_file_hashes(ROOT, LOCKED_RELATIVE_PATHS),
                "dependency_files": locked_file_hashes(ROOT, DEPENDENCY_RELATIVE_PATHS),
            }
            payload["dependency_files"][0]["sha256"] = "0" * 64
            lock_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            status = compare_lock_to_current(ROOT, lock_path)

        self.assertFalse(status["matches"])
        self.assertEqual(status["mismatches"][0]["path"], "minimind/trainer/train_pretrain.py")

    def test_formal_data_counter_excludes_preflight_artifacts_and_counts_seed_outputs(self):
        # 会被这个测试抓住的 bug：把 ENVIRONMENT/preflight/lock 误判为正式科学数据，
        # 或者漏掉 seed/state/theta/analysis 文件。
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "ENVIRONMENT_V07.txt").write_text("env\n", encoding="utf-8")
            (out / "scientific_preflight_v07.json").write_text("{}\n", encoding="utf-8")
            (out / "PREREG_LOCK.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(formal_confirmatory_scientific_data_files(out), [])

            seed_file = out / "seed_2030/theta_mapping_raw.json"
            seed_file.parent.mkdir(parents=True)
            seed_file.write_text("{}\n", encoding="utf-8")

            counted = formal_confirmatory_scientific_data_files(out)
            self.assertEqual(counted, [str(seed_file)])


if __name__ == "__main__":
    unittest.main()
