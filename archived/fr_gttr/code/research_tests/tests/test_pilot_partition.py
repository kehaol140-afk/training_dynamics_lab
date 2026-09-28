import unittest
from fr.pilot_partition import build_partition

class TestPilotPartition(unittest.TestCase):
    def test_disjoint_and_complete(self):
        x = build_partition(100, 123, 10, 20)
        a=set(x["train_indices"]); b=set(x["probe_indices"]); c=set(x["eval_indices"])
        self.assertFalse(a & b)
        self.assertFalse(a & c)
        self.assertFalse(b & c)
        self.assertEqual(len(a|b|c), 100)

    def test_deterministic(self):
        a=build_partition(100, 123, 10, 20)
        b=build_partition(100, 123, 10, 20)
        self.assertEqual(a["sha256"], b["sha256"])
