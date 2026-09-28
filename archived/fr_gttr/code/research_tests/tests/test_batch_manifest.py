"""显式 batch manifest 的确定性单元测试。"""
import unittest

from fr.batch_manifest import build_native_single_gpu_batches, make_permutation


class TestBatchManifest(unittest.TestCase):
    def test_same_seed_same_permutation(self):
        self.assertEqual(make_permutation(101, 42), make_permutation(101, 42))
        self.assertNotEqual(make_permutation(101, 42), make_permutation(101, 43))

    def test_skip_is_batch_based(self):
        all_batches = build_native_single_gpu_batches(20, 0, 4, skip_batches=0)
        skipped = build_native_single_gpu_batches(20, 0, 4, skip_batches=2)
        self.assertEqual(skipped, all_batches[2:])


if __name__ == "__main__":
    unittest.main()
