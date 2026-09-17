"""독립 검증 세 블록의 C80와 공통 기울기 w를 함께 적합한다."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from psychometrics import direct_c80_probability, fit_b384


@dataclass(frozen=True)
class SharedWFit:
    c80_by_block: tuple[float, float, float]
    w: float
    lapse: float
    negative_log_likelihood: float
    converged: bool
    iterations: int
    message: str

    def probability(self, delta_e00: np.ndarray, block_index: np.ndarray) -> np.ndarray:
        thresholds = np.asarray(self.c80_by_block, dtype=float)[block_index]
        return np.asarray(
            direct_c80_probability(delta_e00, thresholds, self.w, self.lapse),
            dtype=float,
        )


@dataclass(frozen=True)
class SharedWBootstrap:
    requested_replicates: int
    successful_replicates: int
    convergence_rate: float
    converged: bool
    confidence: float
    seed: int
    c80_samples: np.ndarray
    w_samples: np.ndarray
    c80_intervals: tuple[tuple[float, float], ...]
    w_interval: tuple[float, float] | None


def _arrays(delta_e00, correct, block_index):
    x = np.asarray(delta_e00, dtype=float)
    y = np.asarray(correct, dtype=float)
    blocks = np.asarray(block_index, dtype=int)
    if x.ndim != 1 or y.ndim != 1 or blocks.ndim != 1 or not (len(x) == len(y) == len(blocks)):
        raise ValueError("delta_e00/correct/block_index는 같은 길이의 1차원이어야 합니다.")
    if len(x) == 0 or np.any(~np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("delta_e00은 비어 있지 않은 양의 유한값이어야 합니다.")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("correct는 0/1이어야 합니다.")
    if set(blocks.tolist()) != {0, 1, 2}:
        raise ValueError("block_index에는 0, 1, 2가 모두 필요합니다.")
    return x, y, blocks


def fit_shared_w(delta_e00, correct, block_index, *, lapse: float) -> SharedWFit:
    """세 block C80 3개와 공통 w 하나를 joint MLE로 적합한다."""
    x, y, blocks = _arrays(delta_e00, correct, block_index)
    if not 0 <= float(lapse) < 0.20:
        raise ValueError("lapse는 0 이상 0.20 미만이어야 합니다.")
    log_x = np.log(x)
    bounds = [(float(log_x.min() - 4), float(log_x.max() + 4))] * 3 + [
        (math.log(0.02), math.log(10.0))
    ]

    def objective(parameters: np.ndarray) -> float:
        c80 = np.exp(parameters[:3])[blocks]
        w = math.exp(float(parameters[3]))
        p = np.asarray(direct_c80_probability(x, c80, w, lapse), dtype=float)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))

    block_medians = [float(np.median(log_x[blocks == block])) for block in range(3)]
    results = []
    for initial_w in (0.20, 0.35, 0.60):
        initial = np.asarray(block_medians + [math.log(initial_w)], dtype=float)
        results.append(minimize(objective, initial, method="L-BFGS-B", bounds=bounds))
    result = min(results, key=lambda row: float(row.fun))
    return SharedWFit(
        c80_by_block=tuple(float(math.exp(value)) for value in result.x[:3]),
        w=float(math.exp(result.x[3])),
        lapse=float(lapse),
        negative_log_likelihood=float(result.fun),
        converged=bool(result.success and math.isfinite(result.fun)),
        iterations=int(result.nit),
        message=str(result.message),
    )


def bootstrap_shared_w(
    fit: SharedWFit,
    delta_e00,
    correct,
    block_index,
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
    minimum_convergence_rate: float = 0.90,
) -> SharedWBootstrap:
    """각 replicate에서 C80 3개+w를 모두 joint 재적합한다."""
    x, _, blocks = _arrays(delta_e00, correct, block_index)
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise ValueError("replicates는 2 이상의 정수여야 합니다.")
    generator = np.random.default_rng(seed)
    probability = fit.probability(x, blocks)
    c80_rows: list[tuple[float, float, float]] = []
    w_rows: list[float] = []
    for _ in range(replicates):
        simulated = (generator.random(len(x)) < probability).astype(int)
        refit = fit_shared_w(x, simulated, blocks, lapse=fit.lapse)
        if refit.converged:
            c80_rows.append(refit.c80_by_block)
            w_rows.append(refit.w)
    successful = len(c80_rows)
    rate = successful / replicates
    if successful:
        samples = np.asarray(c80_rows, dtype=float)
        w_samples = np.asarray(w_rows, dtype=float)
        alpha = (1 - confidence) / 2
        intervals = tuple(
            tuple(float(value) for value in np.quantile(samples[:, block], (alpha, 1 - alpha)))
            for block in range(3)
        )
        w_interval = tuple(float(value) for value in np.quantile(w_samples, (alpha, 1 - alpha)))
    else:
        samples = np.empty((0, 3), dtype=float)
        w_samples = np.empty((0,), dtype=float)
        intervals = ()
        w_interval = None
    return SharedWBootstrap(
        requested_replicates=replicates,
        successful_replicates=successful,
        convergence_rate=rate,
        converged=successful >= 2 and rate >= minimum_convergence_rate,
        confidence=confidence,
        seed=seed,
        c80_samples=samples,
        w_samples=w_samples,
        c80_intervals=intervals,
        w_interval=w_interval,
    )


def compare_shared_vs_free_w_aic(delta_e00, correct, block_index, *, lapse: float) -> dict:
    """공통 w(k=4)와 block별 자유 w(k=6)의 AIC 민감도 분석."""
    x, y, blocks = _arrays(delta_e00, correct, block_index)
    shared = fit_shared_w(x, y, blocks, lapse=lapse)
    free_fits = [
        fit_b384(x[blocks == block], y[blocks == block], np.ones(np.sum(blocks == block)), lapse=lapse)
        for block in range(3)
    ]
    free_converged = all(row.converged for row in free_fits)
    shared_aic = 2 * 4 + 2 * shared.negative_log_likelihood
    free_nll = sum(row.negative_log_likelihood for row in free_fits)
    free_aic = 2 * 6 + 2 * free_nll
    comparison_resolved = bool(shared.converged and free_converged)
    free_preferred = bool(comparison_resolved and free_aic < shared_aic)
    if not comparison_resolved:
        interpretation = "unresolved_numerical_convergence"
    elif free_preferred:
        interpretation = "free_w_lower_aic_warning_not_selected"
    else:
        interpretation = "shared_w_retained"
    return {
        "shared_w_aic": float(shared_aic),
        "free_w_aic": float(free_aic),
        "delta_aic_free_minus_shared": float(free_aic - shared_aic),
        "shared_w_converged": shared.converged,
        "free_w_converged": free_converged,
        "comparison_resolved": comparison_resolved,
        "free_w_preferred": free_preferred,
        "interpretation": interpretation,
    }


__all__ = [
    "SharedWBootstrap",
    "SharedWFit",
    "bootstrap_shared_w",
    "compare_shared_vs_free_w_aic",
    "fit_shared_w",
]
