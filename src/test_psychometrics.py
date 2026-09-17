import math
import unittest

import numpy as np

from psychometrics import (
    direct_c80_probability,
    fit_b192,
    fit_b384,
    fit_c192,
    fit_d384,
    parametric_bootstrap,
)


def deterministic_trials(c80_at_width, *, w, lapse=0.02, repeats=50):
    """확률에 가까운 성공 비율을 고정해 난수 없는 회복 자료를 만든다."""
    x_rows = []
    answer_rows = []
    width_rows = []
    for width in (2.0, 8.0, 16.0):
        threshold = c80_at_width(width)
        for offset in np.linspace(-1.4, 1.4, 11):
            x = threshold * math.exp(float(offset))
            probability = direct_c80_probability(x, threshold, w, lapse)
            successes = int(round(probability * repeats))
            x_rows.extend([x] * repeats)
            answer_rows.extend([1] * successes + [0] * (repeats - successes))
            width_rows.extend([width] * repeats)
    return np.asarray(x_rows), np.asarray(answer_rows), np.asarray(width_rows)


class DirectC80ProbabilityTests(unittest.TestCase):
    def test_probability_is_exactly_point_eight_at_c80(self) -> None:
        for lapse in (0.0, 0.02, 0.10, 0.199):
            with self.subTest(lapse=lapse):
                self.assertAlmostEqual(
                    direct_c80_probability(2.7, 2.7, 0.31, lapse), 0.80, places=14
                )

    def test_array_probability_is_monotone_and_broadcasts(self) -> None:
        probability = direct_c80_probability(
            np.asarray([0.4, 2.0, 8.0]), np.asarray([2.0]), 0.4, 0.02
        )
        self.assertEqual(probability.shape, (3,))
        self.assertTrue(np.all(np.diff(probability) > 0.0))
        self.assertAlmostEqual(probability[1], 0.80)

    def test_parameter_boundaries_are_rejected(self) -> None:
        invalid_calls = (
            lambda: direct_c80_probability(0.0, 2.0, 0.3),
            lambda: direct_c80_probability(1.0, 0.0, 0.3),
            lambda: direct_c80_probability(1.0, 2.0, 0.0),
            lambda: direct_c80_probability(1.0, 2.0, 0.3, -0.01),
            lambda: direct_c80_probability(1.0, 2.0, 0.3, 0.20),
            lambda: direct_c80_probability(float("nan"), 2.0, 0.3),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises((TypeError, ValueError)):
                call()


class MaximumLikelihoodModelTests(unittest.TestCase):
    def test_b192_and_b384_recover_a_constant_c80(self) -> None:
        x, answers, widths = deterministic_trials(
            lambda _width: 2.4, w=0.32, repeats=50
        )
        b192 = fit_b192(x, answers, widths, reference_width=8.0, lapse=0.02)
        b384 = fit_b384(x, answers, widths, lapse=0.02)
        self.assertTrue(b192.converged)
        self.assertTrue(b384.converged)
        self.assertEqual(b192.n_trials, 550)
        self.assertEqual(b384.n_trials, 1650)
        self.assertAlmostEqual(b192.parameters["c80"], 2.4, delta=0.08)
        self.assertAlmostEqual(b384.parameters["c80"], 2.4, delta=0.08)
        self.assertAlmostEqual(b384.parameters["w"], 0.32, delta=0.05)

    def test_c192_uses_reference_data_and_relative_scale(self) -> None:
        reference = 8.0
        scale = lambda width: (width / reference) ** -0.40
        x, answers, widths = deterministic_trials(
            lambda width: 2.2 * scale(width), w=0.30, repeats=50
        )
        fit = fit_c192(
            x,
            answers,
            widths,
            reference_width=reference,
            relative_scale=scale,
            lapse=0.02,
        )
        self.assertTrue(fit.converged)
        self.assertEqual(fit.n_trials, 550)
        self.assertAlmostEqual(fit.parameters["c80_ref"], 2.2, delta=0.08)
        self.assertAlmostEqual(fit.c80(2.0) / fit.c80(8.0), scale(2.0), places=12)

    def test_d384_recovers_negative_beta_without_forcing_it(self) -> None:
        reference = 8.0
        x, answers, widths = deterministic_trials(
            lambda width: 2.6 * (width / reference) ** -0.55,
            w=0.34,
            repeats=50,
        )
        fit = fit_d384(
            x, answers, widths, reference_width=reference, lapse=0.02
        )
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.parameters["c80_ref"], 2.6, delta=0.10)
        self.assertAlmostEqual(fit.parameters["beta"], -0.55, delta=0.08)

    def test_d384_can_return_positive_beta(self) -> None:
        reference = 8.0
        x, answers, widths = deterministic_trials(
            lambda width: 2.1 * (width / reference) ** 0.48,
            w=0.33,
            repeats=50,
        )
        fit = fit_d384(
            x, answers, widths, reference_width=reference, lapse=0.02
        )
        self.assertTrue(fit.converged)
        self.assertGreater(fit.parameters["beta"], 0.35)
        self.assertAlmostEqual(fit.parameters["beta"], 0.48, delta=0.08)

    def test_trial_shape_and_reference_validation(self) -> None:
        with self.assertRaises(ValueError):
            fit_b384([1.0, 2.0], [1], [8.0, 8.0])
        with self.assertRaises(ValueError):
            fit_b192([1.0], [1], [8.0], reference_width=4.0)
        with self.assertRaises(ValueError):
            fit_c192(
                [1.0],
                [1],
                [8.0],
                reference_width=8.0,
                relative_scale=lambda _width: 2.0,
            )


class BootstrapTests(unittest.TestCase):
    def test_parametric_bootstrap_is_reproducible_and_has_ci(self) -> None:
        x, answers, widths = deterministic_trials(
            lambda _width: 2.3, w=0.31, repeats=15
        )
        fit = fit_b384(x, answers, widths, lapse=0.02)
        first = parametric_bootstrap(
            fit, x, answers, widths, replicates=8, seed=12345
        )
        second = parametric_bootstrap(
            fit, x, answers, widths, replicates=8, seed=12345
        )
        self.assertEqual(first.estimates, second.estimates)
        self.assertEqual(first.intervals, second.intervals)
        self.assertTrue(first.converged)
        self.assertEqual(first.successful_replicates, 8)
        low, high = first.c80_interval(8.0)
        self.assertLess(low, high)


if __name__ == "__main__":
    unittest.main()
