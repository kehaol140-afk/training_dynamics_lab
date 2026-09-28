"""stream seed 派生逻辑测试。"""
import unittest
from fr.stream_rng import derive_stream_seed


class TestStreamRNG(unittest.TestCase):
    def test_deterministic_and_distinct(self):
        a = [derive_stream_seed(2026, i) for i in range(8)]
        b = [derive_stream_seed(2026, i) for i in range(8)]
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 8)


if __name__ == "__main__":
    unittest.main()
