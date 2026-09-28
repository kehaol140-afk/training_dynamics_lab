"""Test C + Test D — P4 runner 集成测试（真实 cell s2030_t064，arm A ×1.0，B=32）。

Test C（Stage 1 不得访问 V_final）：
- 运行期插桩 run_eval_nll：所有调用池大小必须 == probe pool（32），
  调用次数 == 监控节奏点数（B=32 → {8,16,24,32} → 4 次）；
- metrics.json 不含任何 V_final 语义字段；输出文件集合恰为协议规定的 4 个；
- 静态扫描 p4_runner.py：eval_indices 仅出现在 CRN 重叠断言中。

Test D（确定性重跑）：CLI 子进程两次完整 run → loss_curve.csv 逐字节一致、
model_hash_final 一致、final_state.pt 的 model 字段哈希一致。

运行要求：CUDA_VISIBLE_DEVICES 指向单卡（C0 CUDA RNG 按 1 设备保存）。
设备：FR_TEST_DEVICE（默认 cuda:0，不可用时回退 cpu）。输出写入
outputs/test_p4_infra/（与正式 P4 的 outputs/p4_full/ 严格分离）。
"""
import _bootstrap  # noqa: F401
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = Path(__file__).resolve().parents[2]
for p in (str(SRC), str(LAB_ROOT / "minimind_base")):
    if p not in sys.path:
        sys.path.insert(0, p)

CELL = "s2030_t064"          # t0=64 → B=32，监控节奏 {8,16,24,32}
PROBE_POOL_SIZE = 32
EXPECTED_CADENCE_COUNT = 4
OUT_ROOT = ROOT / "outputs" / "test_p4_infra"


def _device():
    import torch
    env = os.environ.get("FR_TEST_DEVICE")
    if env:
        return env
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class TestCNoVFinalAccess(unittest.TestCase):
    def test_stage1_run_never_touches_vfinal(self):
        import torch
        import p4_runner
        from p4_runner import load_p4_config, run_one_cell_arm

        cfg = load_p4_config(ROOT / "configs" / "p4_full.json")
        out_dir = OUT_ROOT / "c_check"
        calls = []
        original = p4_runner.run_eval_nll

        def spy(model, ds, indices, batch_size, device, autocast_ctx):
            calls.append(len(indices))
            return original(model, ds, indices, batch_size, device, autocast_ctx)

        p4_runner.run_eval_nll = spy
        try:
            metrics = run_one_cell_arm(cfg, LAB_ROOT, CELL, "A", 1.0, _device(), out_dir)
        finally:
            p4_runner.run_eval_nll = original

        # 所有评测调用都落在 V_monitor（probe pool，32 样本），次数 = 节奏点数
        self.assertEqual(calls, [PROBE_POOL_SIZE] * EXPECTED_CADENCE_COUNT,
                         f"run 期间只允许 V_monitor 评测: {calls}")
        self.assertEqual(len(metrics["monitor_points"]), EXPECTED_CADENCE_COUNT)
        self.assertEqual(metrics["n_optimizer_steps"], 32)
        self.assertAlmostEqual(metrics["schedule"]["lr_last"], 1.0 * 5e-5, places=12)

        # metrics.json 无 V_final 语义字段
        blob = (out_dir / "metrics.json").read_text(encoding="utf-8")
        for forbidden in ("v_final", "V_final", "eval_final", "final_eval", "eval_indices"):
            self.assertNotIn(forbidden, blob)
        for required in ("final_monitor_nll", "best_monitor_nll", "seed",
                         "checkpoint_step", "lr_multiplier",
                         "budget_optimizer_steps",
                         "batch_manifest_sha256", "model_hash_final"):
            self.assertIn(required, blob)

        # 输出文件集合恰为协议规定的 4 个
        self.assertEqual(sorted(q.name for q in out_dir.iterdir()),
                         ["final_state.pt", "loss_curve.csv", "metrics.json",
                          "optimizer_state_statistics.csv"])

        # final_state.pt metadata 自描述
        st = torch.load(out_dir / "final_state.pt", map_location="cpu", weights_only=False)
        md = st["metadata"]
        self.assertEqual((md["cell_id"], md["arm"], md["run_id"]), (CELL, "A", "A_lr1"))
        self.assertEqual(md["budget_optimizer_steps"], 32)

    def test_static_source_scan(self):
        src = (ROOT / "src" / "p4_runner.py").read_text(encoding="utf-8")
        # eval_indices 仅允许出现在 CRN 重叠断言（与 probe 一起的 forbidden_pool）
        occurrences = [i for i in range(len(src)) if src.startswith("eval_indices", i)]
        self.assertEqual(len(occurrences), 1, "eval_indices 在 runner 中只应出现一次（重叠断言）")
        ctx = src[occurrences[0] - 120:occurrences[0] + 40]
        self.assertIn("forbidden_pool", ctx)
        self.assertIn("probe_indices", ctx)


class TestDDeterministicRerun(unittest.TestCase):
    def test_same_cell_same_arm_bitwise_identical(self):
        from vendored_fr.hashing import stable_hash
        import torch

        env = os.environ.copy()
        results = []
        for i in (1, 2):
            out_dir = OUT_ROOT / f"rerun{i}"
            cmd = [sys.executable, str(ROOT / "src" / "p4_runner.py"),
                   "--config", str(ROOT / "configs" / "p4_full.json"),
                   "--lab-root", str(LAB_ROOT), "--cell", CELL, "--arm", "A",
                   "--lr-multiplier", "1.0", "--device", _device(),
                   "--out-dir", str(out_dir)]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            results.append(out_dir)

        r1, r2 = results
        self.assertEqual((r1 / "loss_curve.csv").read_bytes(),
                         (r2 / "loss_curve.csv").read_bytes(),
                         "两次 run 的 loss 曲线必须逐字节一致")
        m1 = json.loads((r1 / "metrics.json").read_text(encoding="utf-8"))
        m2 = json.loads((r2 / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(m1["model_hash_final"], m2["model_hash_final"])
        self.assertEqual(m1["optimizer_hash_final"], m2["optimizer_hash_final"])
        self.assertEqual(m1["final_monitor_nll"], m2["final_monitor_nll"])
        self.assertEqual(m1["batch_manifest_sha256"], m2["batch_manifest_sha256"])

        s1 = torch.load(r1 / "final_state.pt", map_location="cpu", weights_only=False)
        s2 = torch.load(r2 / "final_state.pt", map_location="cpu", weights_only=False)
        self.assertEqual(stable_hash(s1["model"]), stable_hash(s2["model"]))
        self.assertEqual(stable_hash(s1["optimizer"]), stable_hash(s2["optimizer"]))


if __name__ == "__main__":
    unittest.main()
