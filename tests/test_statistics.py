import unittest

from countdown_grpo.statistics import paired_outcomes, wilson_interval


def row(target, correct):
    return {'nums': [1, 2, 3], 'target': target, 'correct': correct}


class StatisticsTests(unittest.TestCase):
    def test_wilson_known_reference_and_boundaries(self):
        # Standard Wilson example: 5 successes in 10 trials, 95% interval.
        low, high = wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.236593, places=6)
        self.assertAlmostEqual(high, 0.763407, places=6)
        self.assertAlmostEqual(wilson_interval(0, 10)[0], 0)
        self.assertAlmostEqual(wilson_interval(10, 10)[1], 1)
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)

    def test_pairing_is_order_independent(self):
        before = [row(i, False) for i in range(1, 7)]
        after = [row(i, True) for i in range(6, 0, -1)]
        result = paired_outcomes(before, after)
        self.assertEqual(result['gained'], 6)
        self.assertEqual(result['lost'], 0)
        self.assertEqual(result['exact_mcnemar_p_unadjusted'], 0.03125)
        self.assertEqual(paired_outcomes(after, after)['exact_mcnemar_p_unadjusted'], 1)

    def test_bad_pairings_rejected(self):
        for before, after in [([], []), ([row(1, False)] * 2, [row(1, True)] * 2),
                              ([row(1, False)], [row(2, True)])]:
            with self.assertRaises(ValueError):
                paired_outcomes(before, after)
