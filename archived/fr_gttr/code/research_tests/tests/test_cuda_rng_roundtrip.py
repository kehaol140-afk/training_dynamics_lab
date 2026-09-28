"""CUDA RNG roundtrip check for FR full-state auditability.

Default unittest discovery skips this file unless FR_RUN_CUDA_TESTS=1 is set.
Run explicitly for GPU validation:

    FR_RUN_CUDA_TESTS=1 PYTHONPATH=research/src python -m unittest \
        discover -s research/tests -p 'test_cuda_rng_roundtrip.py' -v
"""
import os
import unittest

from fr.rng import capture_rng_state, restore_rng_state


class TestCUDARNGRoundtrip(unittest.TestCase):
    def test_cuda_rng_roundtrip_exactly_replays_random_tensors(self):
        if os.environ.get("FR_RUN_CUDA_TESTS") != "1":
            self.skipTest("set FR_RUN_CUDA_TESTS=1 to run the CUDA RNG roundtrip check")

        import torch

        if not torch.cuda.is_available():
            self.fail("CUDA RNG roundtrip requested, but torch.cuda.is_available() is False")

        state = capture_rng_state()
        first = [
            torch.rand(64, device=f"cuda:{idx}")
            for idx in range(torch.cuda.device_count())
        ]
        for idx in range(torch.cuda.device_count()):
            _ = torch.rand(17, device=f"cuda:{idx}")

        restore_rng_state(state)
        second = [
            torch.rand(64, device=f"cuda:{idx}")
            for idx in range(torch.cuda.device_count())
        ]

        for idx, (a, b) in enumerate(zip(first, second)):
            self.assertTrue(
                torch.equal(a, b),
                f"CUDA RNG mismatch after restore on device cuda:{idx}",
            )


if __name__ == "__main__":
    unittest.main()
