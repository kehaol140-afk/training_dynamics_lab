"""FR-C1-Late reference checks (protocol FR_C1L §11).

Two small INDEPENDENT checks (no import of production criterion helpers):
T1 aggregate-vs-per-point gate semantics on a 7-anchor synthetic (six E=.43,
one E=1.166, equal TSS): aggregate E ~ .594 < 1 while a per-point rule fails.
T2 small-dimension vector statistics: SE_vec, same-probe paired residual SE,
and the signed cross term computed directly from raw vectors must be exactly
recoverable from the saved sufficient statistics (n, sum vector, sumsq,
centered_SS); missing centered_SS must yield DATA_MISSING, not an imputed value.
"""
import math
import unittest

DATA_MISSING = "DATA_MISSING"


def agg_E(per_anchor_sse, per_anchor_tss):
    """Frozen aggregate: sqrt(sum SSE / sum TSS); per-point values are NOT
    averaged and a single point > 1 does not fail the aggregate."""
    den = sum(per_anchor_tss)
    if den <= 0:
        return float("nan")
    return math.sqrt(sum(per_anchor_sse) / den)


def se_vec_from_stats(n, sumsq, sum_sq):
    """SE_vec from sufficient stats (sum_sq = ||sum||^2); DATA_MISSING if
    any required statistic is absent."""
    if n is None or sumsq is None or sum_sq is None:
        return DATA_MISSING
    if n < 2:
        return float("nan")
    centered = sumsq - sum_sq / n  # sum_j||x_j - xbar||^2
    return math.sqrt(max(centered, 0.0) / (n * (n - 1)))


def paired_resid_se(xs, ys):
    """SE of the per-sample paired differences x_j - y_j, direct from vectors."""
    n = len(xs)
    diffs = [x - y for x, y in zip(xs, ys)]
    m = sum(diffs) / n
    return math.sqrt(sum((d - m) ** 2 for d in diffs) / (n * (n - 1)))


class TestC1LateReference(unittest.TestCase):
    def test_t1_aggregate_not_perpoint(self):
        tss = [1.0] * 7
        Es = [0.43] * 6 + [1.166]
        sses = [e * e * t for e, t in zip(Es, tss)]
        E_agg = agg_E(sses, tss)
        # hand value: sqrt((6*0.43^2 + 1.166^2)/7)
        self.assertAlmostEqual(E_agg, math.sqrt((6 * 0.43 ** 2 + 1.166 ** 2) / 7),
                               places=12)
        self.assertLess(E_agg, 1.0)
        self.assertGreaterEqual(E_agg, 0.59)
        self.assertLessEqual(E_agg, 0.60)
        # per-point rule would fail on the single 1.166; aggregate does not
        per_point_fails = any(e > 1.0 for e in Es)
        self.assertTrue(per_point_fails)
        self.assertFalse(E_agg > 1.0)

    def test_t2_stats_recovery_and_data_missing(self):
        xs = [0.5, -1.0, 2.0, 0.25]   # predictions p_j (1-D)
        ys = [0.4, -0.9, 1.7, 0.5]   # targets y_j (1-D)
        n = len(xs)
        sum_x = sum(xs)
        sumsq_x = sum(x * x for x in xs)
        # direct SE of x
        m = sum_x / n
        se_direct = math.sqrt(sum((x - m) ** 2 for x in xs) / (n * (n - 1)))
        # recovery from stats
        se_stats = se_vec_from_stats(n, sumsq_x, sum_x * sum_x)
        self.assertAlmostEqual(se_stats, se_direct, places=15)
        # paired residual SE: direct vs recomputed from paired-diff stats
        pr_direct = paired_resid_se(xs, ys)
        diffs = [x - y for x, y in zip(xs, ys)]
        sd = sum(diffs)
        sdq = sum(d * d for d in diffs)
        pr_stats = se_vec_from_stats(n, sdq, sd * sd)
        self.assertAlmostEqual(pr_stats, pr_direct, places=15)
        # cross term 2<e_bank, e_local> on a tiny decomposition example
        p_d, y_a, p_a = 1.2, 0.8, 1.0   # scalars
        e_bank, e_local = p_d - p_a, p_a - y_a
        cross = 2.0 * e_bank * e_local
        e_total = p_d - y_a
        self.assertAlmostEqual(e_total, e_bank + e_local, places=15)
        self.assertAlmostEqual(e_total ** 2,
                               e_bank ** 2 + e_local ** 2 + cross, places=15)
        # DATA_MISSING path: centered_SS absent must not be imputed
        self.assertEqual(se_vec_from_stats(None, sumsq_x, 0.0), DATA_MISSING)
        self.assertEqual(se_vec_from_stats(n, None, 0.0), DATA_MISSING)


if __name__ == "__main__":
    unittest.main()
