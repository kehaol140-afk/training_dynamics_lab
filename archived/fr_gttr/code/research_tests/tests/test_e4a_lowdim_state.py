import math
import unittest


try:
    import torch

    from fr.e4a_lowdim_state import (
        adamw_lowdim_from_optimizer_state,
        classify_candidate,
        single_factor_loso,
    )
except ImportError:
    torch = None
    adamw_lowdim_from_optimizer_state = None
    classify_candidate = None
    single_factor_loso = None


class TestE4ALowdimState(unittest.TestCase):
    def test_adamw_lowdim_features_apply_bias_correction(self):
        # [FR-OBSERVATION-ONLY] 这个极小例子只验证读取 AdamW state 的标量公式。
        self.assertIsNotNone(adamw_lowdim_from_optimizer_state, "missing fr.e4a_lowdim_state")

        beta1 = 0.9
        beta2 = 0.999
        step = 2.0
        one_minus_b1 = 1.0 - beta1**step
        one_minus_b2 = 1.0 - beta2**step
        q = torch.tensor([0.6, 0.8], dtype=torch.float32)
        m_hat = torch.tensor([1.0, 2.0], dtype=torch.float64)
        v_hat = torch.tensor([4.0, 9.0], dtype=torch.float64)
        opt = {
            "param_groups": [
                {"params": [0], "betas": (beta1, beta2), "eps": 1e-8, "lr": 0.01}
            ],
            "state": {
                0: {
                    "exp_avg": (m_hat * one_minus_b1).to(torch.float32),
                    "exp_avg_sq": (v_hat * one_minus_b2).to(torch.float32),
                    "step": torch.tensor(step, dtype=torch.float32),
                }
            },
        }

        out = adamw_lowdim_from_optimizer_state({"w": q}, opt)

        denom = torch.sqrt(v_hat) + 1e-8
        expected_mu = float(((q.double() * q.double()) / denom).sum())
        expected_m_cos = float((q.double() * m_hat).sum() / torch.linalg.vector_norm(m_hat))
        update = m_hat / denom
        expected_u_cos = float((q.double() * update).sum() / torch.linalg.vector_norm(update))

        self.assertAlmostEqual(out["mu_q"], expected_mu, places=6)
        self.assertAlmostEqual(out["momentum_alignment"], expected_m_cos, places=6)
        self.assertAlmostEqual(out["update_alignment"], expected_u_cos, places=6)
        self.assertAlmostEqual(out["lr"], 0.01)
        self.assertEqual(out["adamw_step_min"], 2.0)
        self.assertEqual(out["adamw_step_max"], 2.0)
        self.assertTrue(out["finite"])

    def test_loso_standardizes_candidate_with_training_fold_only(self):
        self.assertIsNotNone(single_factor_loso, "missing fr.e4a_lowdim_state")

        rows = []
        for seed_pos, seed in enumerate(range(2030, 2038)):
            seed_effect = 0.05 * seed_pos
            for state_pos, step in enumerate([46, 52, 58, 64, 70, 76, 82]):
                sf = step / 128.0
                b = -0.4 + 0.13 * state_pos + 0.02 * seed_pos
                z = 1000.0 + state_pos if seed == 2030 else 0.2 * seed_pos + state_pos
                rows.append({
                    "train_seed": seed,
                    "step": step,
                    "state_fraction": sf,
                    "mu_q": z,
                    "B_H8": b,
                    "chi_H8": 1.0 + seed_effect + 0.3 * sf + 0.7 * b + 0.01 * z + 0.02 * b * z,
                })

        out = single_factor_loso(rows, candidate="mu_q", horizon=8)
        held_2030 = next(f for f in out["folds"] if f["held_out_seed"] == 2030)

        self.assertEqual(out["total_folds"], 8)
        self.assertEqual(out["feature_standardization_scope"], "LOSO_TRAINING_FOLD_ONLY")
        self.assertLess(held_2030["train_z_mean"], 10.0)
        self.assertTrue(math.isfinite(held_2030["relative_mae_improvement"]))

    def test_candidate_classification_is_exploratory_only(self):
        self.assertIsNotNone(classify_candidate, "missing fr.e4a_lowdim_state")

        promising = {
            "mean_relative_mae_improvement": 0.01,
            "folds_improved": 6,
            "delta_sign_stability_count": 6,
        }
        weak = {
            "mean_relative_mae_improvement": 0.01,
            "folds_improved": 5,
            "delta_sign_stability_count": 8,
        }
        unsupported = {
            "mean_relative_mae_improvement": -0.001,
            "folds_improved": 8,
            "delta_sign_stability_count": 8,
        }

        self.assertEqual(classify_candidate(promising), "PROMISING")
        self.assertEqual(classify_candidate(weak), "WEAK")
        self.assertEqual(classify_candidate(unsupported), "UNSUPPORTED")


if __name__ == "__main__":
    unittest.main()
