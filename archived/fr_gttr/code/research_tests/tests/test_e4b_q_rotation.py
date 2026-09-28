import unittest


class TestE4BQRotation(unittest.TestCase):
    def test_direction_cosine(self):
        import torch
        from fr.e4b_q_rotation import direction_cosine
        a = {"x": torch.tensor([1.0, 0.0]), "y": torch.tensor([0.0, 1.0])}
        b = {"x": torch.tensor([1.0, 0.0]), "y": torch.tensor([0.0, 1.0])}
        c = {"x": torch.tensor([-1.0, 0.0]), "y": torch.tensor([0.0, -1.0])}
        self.assertAlmostEqual(direction_cosine(a, b), 1.0, places=12)
        self.assertAlmostEqual(direction_cosine(a, c), -1.0, places=12)

    def test_loso_screen_shapes(self):
        from fr.e4b_q_rotation import loso_conditional_screen
        rows = []
        for seed in [1, 2, 3, 4]:
            for j in range(5):
                b = -0.2 + 0.1*j + 0.01*seed
                rq = 0.9 - 0.05*j + 0.01*seed
                chi = 0.03*seed + 0.2*j + 0.5*b + 0.4*b*rq
                rows.append({"seed": seed, "state_fraction": j/4, "B": b, "mean_Rq": rq, "chi": chi})
        out = loso_conditional_screen(rows)
        self.assertEqual(out["n_folds"], 4)
        self.assertEqual(len(out["folds"]), 4)
        self.assertTrue(all("delta_interaction" in f for f in out["folds"]))


if __name__ == "__main__":
    unittest.main()
