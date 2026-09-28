"""方向归一化和梯度投影测试。"""
import unittest
import torch
from fr.direction import normalize_direction, direction_norm, project_current_gradients


class TestDirection(unittest.TestCase):
    def test_normalize_and_projection(self):
        p = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        q, n = normalize_direction({"p": torch.tensor([3.0, 4.0])})
        self.assertAlmostEqual(n, 5.0, places=10)
        self.assertAlmostEqual(direction_norm(q), 1.0, places=6)
        p.grad = torch.tensor([3.0, 4.0])
        obs = project_current_gradients([("p", p)], q)
        self.assertAlmostEqual(obs["projection"], 5.0, places=10)
        self.assertAlmostEqual(obs["grad_norm"], 5.0, places=10)


if __name__ == "__main__":
    unittest.main()
