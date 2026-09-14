from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from psychometrics import FitResult
from validation_psychometrics import SharedWFit, compare_shared_vs_free_w_aic


def _shared_fit(*, converged: bool, nll: float = 10.0) -> SharedWFit:
    return SharedWFit(
        c80_by_block=(1.0, 1.1, 1.2),
        w=0.3,
        lapse=0.02,
        negative_log_likelihood=nll,
        converged=converged,
        iterations=5,
        message="fixture",
    )


def _free_fit(*, converged: bool, nll: float = 1.0) -> FitResult:
    return FitResult(
        model="B288",
        parameters={"c80": 1.0, "w": 0.3},
        lapse=0.02,
        n_trials=2,
        negative_log_likelihood=nll,
        converged=converged,
        iterations=5,
        message="fixture",
    )


class SharedFreeWSensitivityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.asarray([0.5, 1.0, 0.5, 1.0, 0.5, 1.0])
        self.y = np.asarray([0, 1, 0, 1, 0, 1])
        self.blocks = np.asarray([0, 0, 1, 1, 2, 2])

    def test_shared_fit_failure_is_unresolved_even_when_free_aic_is_lower(self) -> None:
        with (
            patch(
                "validation_psychometrics.fit_shared_w",
                return_value=_shared_fit(converged=False),
            ),
            patch(
                "validation_psychometrics.fit_b288",
                return_value=_free_fit(converged=True),
            ),
        ):
            result = compare_shared_vs_free_w_aic(
                self.x, self.y, self.blocks, lapse=0.02
            )
        self.assertFalse(result["comparison_resolved"])
        self.assertFalse(result["free_w_preferred"])
        self.assertEqual(result["interpretation"], "unresolved_numerical_convergence")

    def test_any_free_fit_failure_is_unresolved(self) -> None:
        free_fits = [
            _free_fit(converged=True),
            _free_fit(converged=False),
            _free_fit(converged=True),
        ]
        with (
            patch(
                "validation_psychometrics.fit_shared_w",
                return_value=_shared_fit(converged=True),
            ),
            patch("validation_psychometrics.fit_b288", side_effect=free_fits),
        ):
            result = compare_shared_vs_free_w_aic(
                self.x, self.y, self.blocks, lapse=0.02
            )
        self.assertFalse(result["comparison_resolved"])
        self.assertFalse(result["free_w_preferred"])
        self.assertEqual(result["interpretation"], "unresolved_numerical_convergence")

    def test_lower_free_aic_is_preferred_only_after_both_models_converge(self) -> None:
        with (
            patch(
                "validation_psychometrics.fit_shared_w",
                return_value=_shared_fit(converged=True),
            ),
            patch(
                "validation_psychometrics.fit_b288",
                return_value=_free_fit(converged=True),
            ),
        ):
            result = compare_shared_vs_free_w_aic(
                self.x, self.y, self.blocks, lapse=0.02
            )
        self.assertTrue(result["comparison_resolved"])
        self.assertTrue(result["free_w_preferred"])
        self.assertEqual(
            result["interpretation"], "free_w_lower_aic_warning_not_selected"
        )


if __name__ == "__main__":
    unittest.main()
