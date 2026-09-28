"""rng.py 的随机状态往返恢复测试。

测试逻辑：
1. capture 当前 RNG；
2. 生成一次 Python / NumPy / Torch 随机数；
3. restore 到捕获状态；
4. 再生成一次；
5. 两次结果必须完全一致。

当前测试只验证 Python、NumPy、Torch CPU；CUDA RNG 会在正式服务器接线阶段进一步验证。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fr.rng import capture_rng_state, restore_rng_state


class TestRNGRoundtrip(unittest.TestCase):
    def test_python_numpy_torch_cpu_roundtrip(self):
        """Python / NumPy / Torch CPU RNG 恢复后应产生相同随机结果。"""
        import random
        import numpy as np
        import torch

        state = capture_rng_state()
        a = (random.random(), float(np.random.rand()), torch.rand(4))
        restore_rng_state(state)
        b = (random.random(), float(np.random.rand()), torch.rand(4))

        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])
        self.assertTrue(torch.equal(a[2], b[2]))


if __name__ == "__main__":
    unittest.main()
