"""perturbation.py 的基础单元测试。

这里只验证玩具梯度上的两个数学性质：
1. sign=0 时必须严格 no-op；
2. +epsilon 和 -epsilon 相对于 baseline 的改变应近似反对称。

这不是正式 FR 科学实验。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fr.perturbation import apply_signed_gradient_impulse_, normalize_direction


class TestPerturbation(unittest.TestCase):
    def test_noop_and_symmetry(self):
        """检查 no-op 与正负中心对称性。"""
        import torch

        model = torch.nn.Linear(3, 2, bias=False)
        x = torch.randn(4, 3)
        model(x).sum().backward()
        base = model.weight.grad.detach().clone()

        q, _ = normalize_direction({"weight": torch.ones_like(base)})

        # baseline：sign=0，不允许改变梯度。
        model.weight.grad.copy_(base)
        apply_signed_gradient_impulse_(model.named_parameters(), q, epsilon=0.1, sign=0)
        self.assertTrue(torch.equal(model.weight.grad, base))

        # plus 分支。
        model.weight.grad.copy_(base)
        apply_signed_gradient_impulse_(model.named_parameters(), q, epsilon=0.1, sign=1)
        plus = model.weight.grad.detach().clone()

        # minus 分支。
        model.weight.grad.copy_(base)
        apply_signed_gradient_impulse_(model.named_parameters(), q, epsilon=0.1, sign=-1)
        minus = model.weight.grad.detach().clone()

        # plus-base 与 -(minus-base) 应近似相等。
        self.assertTrue(torch.allclose(plus - base, -(minus - base), atol=1e-7, rtol=0))


if __name__ == "__main__":
    unittest.main()
