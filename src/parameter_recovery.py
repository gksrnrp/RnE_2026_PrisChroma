"""direct-C80 분석 코드의 결정론적 합성 parameter-recovery 실행기.

이 파일이 만드는 값은 모두 ``synthetic``이다. 실제 참가자 결론이나 실제
역치 파일을 만들지 않으며, 본 시행 전 수식·적합·부트스트랩이 알려진 모수를
되찾는지만 검사한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from experiment_design import szafir_relative_scale
from implementation_provenance import implementation_hashes
from psychometrics import (
    BootstrapResult,
    FitResult,
    direct_c80_probability,
    fit_b96,
    fit_b288,
    fit_c96,
    fit_d288,
    parametric_bootstrap,
)


DEFAULT_WIDTHS_DEG = (0.06, 0.18, 0.30)
DEFAULT_REFERENCE_WIDTH_DEG = 0.18
DEFAULT_DIRECTION_LAB = (0.0, 1.0, 0.0)
DEFAULT_SZAFIR_PROBABILITY = 0.5

# 사람 자료를 보기 전에 분석기 동작만 점검하기 위한 절대 색차 후보다. 실제
# 참가자 자극으로 자동 승인되는 값이 아니며, 장치·색역·개인 profile을 동결한
# 뒤 그 config의 12개 실제 수준으로 recovery를 다시 실행해야 한다.
DEFAULT_FIXED_DELTA_E00_LEVELS = (
    0.35,
    0.4826,
    0.6654,
    0.9175,
    1.2651,
    1.7443,
    2.4050,
    3.3160,
    4.5720,
    6.3038,
    8.6915,
    12.0,
)

# 한 가지 '잘 맞는' 가상 참가자만 검사하지 않는다. 이름과 로그 범위상의
# 위치만 사전에 고정하고, 실제 C80과 w는 매번 동결된 12수준 범위에서 유도한다.
# 따라서 후보 수준을 바꿔도 예전의 편리한 진실값을 그대로 재사용하지 않는다.
RECOVERY_PROFILE_SPECIFICATIONS = (
    {"name": "low-steep", "log_position": 0.10, "w_step_scale": 0.70},
    {"name": "central", "log_position": 0.50, "w_step_scale": 1.00},
    {"name": "high-shallow", "log_position": 0.90, "w_step_scale": 1.50},
)

# 실제 참가자 데이터를 보기 전에 고정하는 수치 회복 판정 기준이다.
RECOVERY_CRITERIA = {
    "minimum_simulation_replicates_for_pass": 200,
    "minimum_bootstrap_replicates_for_pass": 200,
    "minimum_convergence_rate": 0.95,
    "maximum_c80_median_absolute_relative_error": 0.20,
    "maximum_c80_90th_percentile_absolute_relative_error": 0.50,
    "minimum_d_beta_sign_recovery_rate": 0.80,
    "maximum_d_beta_median_absolute_error": 0.25,
    "maximum_d_beta_90th_percentile_absolute_error": 0.65,
    "maximum_held_out_c80_median_absolute_relative_error": 0.20,
    "maximum_held_out_c80_90th_percentile_absolute_relative_error": 0.50,
    "minimum_bootstrap_convergence_rate": 0.80,
}


def _positive_integer(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name}은 {minimum} 이상의 정수여야 합니다.")
    if value < minimum:
        raise ValueError(f"{name}은 {minimum} 이상의 정수여야 합니다.")
    return value


def _szafir_scale(
    *,
    reference_width: float = DEFAULT_REFERENCE_WIDTH_DEG,
    direction_lab: Sequence[float] = DEFAULT_DIRECTION_LAB,
    probability: float = DEFAULT_SZAFIR_PROBABILITY,
) -> Callable[[float], float]:
    """기준 굵기에서 1이 되는 Szafir 상대 배율 함수를 만든다."""

    def scale(width: float) -> float:
        return szafir_relative_scale(
            direction_lab,
            probability,
            width,
            reference_width,
        )

    return scale


def _derive_recovery_profiles(
    *,
    delta_e00_levels: Sequence[float],
    widths: Sequence[float],
    reference_width: float,
    direction_lab: Sequence[float],
    probability: float,
) -> tuple[dict[str, float | str], ...]:
    """동결 수준 안에 모든 생성 C80이 놓이는 세 stress profile을 만든다.

    low/high를 단순 상수로 두면 사람이 실제로 고른 자극 범위와 무관한 쉬운
    recovery가 될 수 있다. 여기서는 D288과 C96의 굵기 배율까지 고려해 모든
    진짜 C80이 표시 범위 안에 남는 기준-C80 구간을 먼저 계산한다.
    """
    levels = np.asarray(delta_e00_levels, dtype=float)
    log_levels = np.log(levels)
    typical_log_step = float(np.median(np.diff(log_levels)))
    scale = _szafir_scale(
        reference_width=reference_width,
        direction_lab=direction_lab,
        probability=probability,
    )
    beta = -0.48
    factors = [1.0]
    factors.extend((width / reference_width) ** beta for width in widths)
    factors.extend(scale(width) for width in widths)
    minimum_factor = min(factors)
    maximum_factor = max(factors)
    feasible_low = float(levels[0]) / minimum_factor
    feasible_high = float(levels[-1]) / maximum_factor
    if not (
        math.isfinite(feasible_low)
        and math.isfinite(feasible_high)
        and 0.0 < feasible_low < feasible_high
    ):
        raise ValueError(
            "동결 delta_e00_levels가 D288/C96 stress profile의 C80 범위를 감싸지 못합니다."
        )

    low_log = math.log(feasible_low)
    high_log = math.log(feasible_high)
    profiles: list[dict[str, float | str]] = []
    for specification in RECOVERY_PROFILE_SPECIFICATIONS:
        position = float(specification["log_position"])
        c80_ref = math.exp(low_log + position * (high_log - low_log))
        w = typical_log_step * float(specification["w_step_scale"])
        profiles.append(
            {
                "name": str(specification["name"]),
                "c80_ref": c80_ref,
                "w": w,
                "log_position": position,
                "w_step_scale": float(specification["w_step_scale"]),
                "feasible_reference_c80_low": feasible_low,
                "feasible_reference_c80_high": feasible_high,
                "typical_log_level_step": typical_log_step,
            }
        )
    return tuple(profiles)


def simulate_trials(
    *,
    widths: Sequence[float],
    trials_per_width: int,
    c80_at_width: Callable[[float], float],
    w: float,
    lapse: float,
    seed: int,
    levels: int = 12,
    delta_e00_levels: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """각 굵기에서 지정 색차를 제시한 합성 0/1 시행을 생성한다.

    ``delta_e00_levels``를 주면 모든 굵기에 똑같은 절대 수준을 사용한다.
    생략한 기존 경로는 수치 단위시험용으로만 진짜 C80 주변 수준을 만들며,
    사람 실험 sampling plan의 recovery로 해석하면 안 된다.
    """
    trials = _positive_integer(trials_per_width, "trials_per_width", minimum=12)
    if delta_e00_levels is None:
        level_count = _positive_integer(levels, "levels", minimum=3)
        fixed_levels = None
    else:
        fixed_levels = np.asarray(delta_e00_levels, dtype=float)
        if (
            fixed_levels.ndim != 1
            or fixed_levels.size < 3
            or np.any(~np.isfinite(fixed_levels))
            or np.any(fixed_levels <= 0.0)
            or np.any(np.diff(fixed_levels) <= 0.0)
        ):
            raise ValueError(
                "delta_e00_levels는 엄격히 증가하는 양의 유한값 세 개 이상이어야 합니다."
            )
        level_count = int(fixed_levels.size)
        if trials < level_count:
            raise ValueError("trials_per_width는 delta_e00_levels 수 이상이어야 합니다.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed는 정수여야 합니다.")
    width_values = np.asarray(widths, dtype=float)
    if (
        width_values.ndim != 1
        or width_values.size < 2
        or np.any(~np.isfinite(width_values))
        or np.any(width_values <= 0.0)
    ):
        raise ValueError("widths에는 0보다 큰 유한한 굵기가 둘 이상 필요합니다.")

    generator = np.random.default_rng(seed)
    all_x: list[np.ndarray] = []
    all_correct: list[np.ndarray] = []
    all_widths: list[np.ndarray] = []
    base_repeats, remainder = divmod(trials, level_count)
    counts = np.full(level_count, base_repeats, dtype=int)
    counts[:remainder] += 1
    log_offsets = np.linspace(-1.45, 1.45, level_count)

    for width in width_values:
        threshold = float(c80_at_width(float(width)))
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("c80_at_width는 0보다 큰 유한한 C80을 반환해야 합니다.")
        level_x = (
            threshold * np.exp(log_offsets)
            if fixed_levels is None
            else fixed_levels
        )
        x = np.repeat(level_x, counts)
        # 동일 색차가 한 덩어리로 나오지 않도록 순서를 섞되 seed로 고정한다.
        x = x[generator.permutation(x.size)]
        probability = np.asarray(direct_c80_probability(x, threshold, w, lapse))
        answers = (generator.random(x.size) < probability).astype(int)
        all_x.append(x)
        all_correct.append(answers)
        all_widths.append(np.full(x.size, width, dtype=float))

    x = np.concatenate(all_x)
    correct = np.concatenate(all_correct)
    trial_widths = np.concatenate(all_widths)
    order = generator.permutation(x.size)
    return x[order], correct[order], trial_widths[order]


def _fit_summary(fit: FitResult, evaluation_widths: Sequence[float]) -> dict:
    return {
        "model": fit.model,
        "converged": fit.converged,
        "message": fit.message,
        "iterations": fit.iterations,
        "n_trials": fit.n_trials,
        "negative_log_likelihood": fit.negative_log_likelihood,
        "lapse_fixed": fit.lapse,
        "reference_width": fit.reference_width,
        "parameters": dict(fit.parameters),
        "predicted_c80": {
            f"{width:.8g}": float(fit.c80(width)) for width in evaluation_widths
        },
    }


def _bootstrap_summary(
    result: BootstrapResult, evaluation_widths: Sequence[float]
) -> dict:
    return {
        "seed": result.seed,
        "confidence": result.confidence,
        "requested_replicates": result.requested_replicates,
        "successful_replicates": result.successful_replicates,
        "convergence_rate": result.convergence_rate,
        "converged": result.converged,
        "parameter_intervals": {
            key: list(interval) for key, interval in result.intervals.items()
        },
        "c80_intervals": {
            f"{width:.8g}": list(result.c80_interval(width))
            for width in evaluation_widths
        }
        if result.successful_replicates
        else {},
    }


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _quantile(values: list[float], probability: float) -> float | None:
    return float(np.quantile(values, probability)) if values else None


def _relative_error(estimate: float, truth: float) -> float:
    return abs(estimate - truth) / truth


def _empty_accumulator() -> dict[str, int | list[float]]:
    return {
        "converged": 0,
        "c80_relative_errors": [],
        "held_out_relative_errors": [],
        "beta_absolute_errors": [],
        "beta_sign_recovered": 0,
    }


def _record_fit(
    accumulator: dict[str, int | list[float]],
    *,
    model: str,
    fit: FitResult,
    truth: dict,
    held_out_width: float,
) -> None:
    if not fit.converged or not math.isfinite(fit.negative_log_likelihood):
        return
    accumulator["converged"] = int(accumulator["converged"]) + 1
    parameter_name = "c80" if model in {"B96", "B288"} else "c80_ref"
    reference_errors = accumulator["c80_relative_errors"]
    held_out_errors = accumulator["held_out_relative_errors"]
    assert isinstance(reference_errors, list) and isinstance(held_out_errors, list)
    reference_errors.append(
        _relative_error(float(fit.parameters[parameter_name]), truth["c80_ref"])
    )
    true_held_out = float(truth["c80_at_width"](held_out_width))
    held_out_errors.append(
        _relative_error(float(fit.c80(held_out_width)), true_held_out)
    )
    if model == "D288":
        beta_errors = accumulator["beta_absolute_errors"]
        assert isinstance(beta_errors, list)
        estimated_beta = float(fit.parameters["beta"])
        true_beta = float(truth["beta"])
        beta_errors.append(abs(estimated_beta - true_beta))
        if estimated_beta * true_beta > 0.0:
            accumulator["beta_sign_recovered"] = (
                int(accumulator["beta_sign_recovered"]) + 1
            )


def _accumulator_summary(
    accumulator: dict[str, int | list[float]],
    *,
    attempts: int,
    model: str,
) -> dict:
    reference_errors = accumulator["c80_relative_errors"]
    held_out_errors = accumulator["held_out_relative_errors"]
    beta_errors = accumulator["beta_absolute_errors"]
    assert (
        isinstance(reference_errors, list)
        and isinstance(held_out_errors, list)
        and isinstance(beta_errors, list)
    )
    successful = int(accumulator["converged"])
    summary = {
        "scenario": (
            "constant"
            if model in {"B96", "B288"}
            else "log_width"
            if model == "D288"
            else "szafir_relative"
        ),
        "attempted_fits": attempts,
        "successful_fits": successful,
        "convergence_rate": successful / attempts if attempts else 0.0,
        "c80_median_absolute_relative_error": _median(reference_errors),
        "c80_90th_percentile_absolute_relative_error": _quantile(
            reference_errors, 0.90
        ),
        "held_out_c80_median_absolute_relative_error": _median(held_out_errors),
        "held_out_c80_90th_percentile_absolute_relative_error": _quantile(
            held_out_errors, 0.90
        ),
    }
    if model == "D288":
        summary.update(
            {
                # 수렴 실패도 부호 회복 실패로 세어 낙관적 비율을 막는다.
                "beta_sign_recovery_rate": (
                    int(accumulator["beta_sign_recovered"]) / attempts
                    if attempts
                    else 0.0
                ),
                "beta_median_absolute_error": _median(beta_errors),
                "beta_90th_percentile_absolute_error": _quantile(
                    beta_errors, 0.90
                ),
            }
        )
    return summary


def _metric_failure(
    *, model: str, scope: str, metrics: dict
) -> list[str]:
    """한 model/profile의 사전 고정 회복 기준 위반을 설명한다."""
    reasons: list[str] = []
    comparisons = (
        (
            "convergence_rate",
            "minimum_convergence_rate",
            "min",
            "수렴률",
        ),
        (
            "c80_median_absolute_relative_error",
            "maximum_c80_median_absolute_relative_error",
            "max",
            "기준 굵기 C80 중앙 절대상대오차",
        ),
        (
            "c80_90th_percentile_absolute_relative_error",
            "maximum_c80_90th_percentile_absolute_relative_error",
            "max",
            "기준 굵기 C80 90백분위 절대상대오차",
        ),
        (
            "held_out_c80_median_absolute_relative_error",
            "maximum_held_out_c80_median_absolute_relative_error",
            "max",
            "미측정 굵기 C80 중앙 절대상대오차",
        ),
        (
            "held_out_c80_90th_percentile_absolute_relative_error",
            "maximum_held_out_c80_90th_percentile_absolute_relative_error",
            "max",
            "미측정 굵기 C80 90백분위 절대상대오차",
        ),
    )
    for metric_name, criterion_name, direction, label in comparisons:
        value = metrics.get(metric_name)
        threshold = RECOVERY_CRITERIA[criterion_name]
        failed = (
            value is None
            or (direction == "min" and value < threshold)
            or (direction == "max" and value > threshold)
        )
        if failed:
            rendered = "없음" if value is None else f"{value:.3f}"
            operator = "<" if direction == "min" else ">"
            reasons.append(
                f"{scope}/{model}: {label} {rendered} {operator} {threshold:.3f}"
            )
    if model == "D288":
        beta_comparisons = (
            (
                "beta_sign_recovery_rate",
                "minimum_d_beta_sign_recovery_rate",
                "min",
                "beta 부호 회복률",
            ),
            (
                "beta_median_absolute_error",
                "maximum_d_beta_median_absolute_error",
                "max",
                "beta 중앙 절대오차",
            ),
            (
                "beta_90th_percentile_absolute_error",
                "maximum_d_beta_90th_percentile_absolute_error",
                "max",
                "beta 90백분위 절대오차",
            ),
        )
        for metric_name, criterion_name, direction, label in beta_comparisons:
            value = metrics.get(metric_name)
            threshold = RECOVERY_CRITERIA[criterion_name]
            failed = (
                value is None
                or (direction == "min" and value < threshold)
                or (direction == "max" and value > threshold)
            )
            if failed:
                rendered = "없음" if value is None else f"{value:.3f}"
                operator = "<" if direction == "min" else ">"
                reasons.append(
                    f"{scope}/{model}: {label} {rendered} {operator} {threshold:.3f}"
                )
    return reasons


def _monte_carlo_recovery(
    *,
    seed: int,
    simulation_replicates: int,
    trials_per_width: int,
    widths: Sequence[float],
    reference_width: float,
    lapse: float,
    delta_e00_levels: Sequence[float],
    direction_lab: Sequence[float],
    probability: float,
) -> dict:
    """동결 수준에서 유도한 low/central/high profile별 회복을 요약한다."""
    held_out_width = math.sqrt(widths[0] * widths[1])
    scale = _szafir_scale(
        reference_width=reference_width,
        direction_lab=direction_lab,
        probability=probability,
    )
    profiles = _derive_recovery_profiles(
        delta_e00_levels=delta_e00_levels,
        widths=widths,
        reference_width=reference_width,
        direction_lab=direction_lab,
        probability=probability,
    )
    model_names = ("B96", "B288", "C96", "D288")
    accumulators = {
        model: _empty_accumulator() for model in model_names
    }
    profile_accumulators = {
        str(profile["name"]): {
            model: _empty_accumulator() for model in model_names
        }
        for profile in profiles
    }
    profile_counts = {str(profile["name"]): 0 for profile in profiles}

    for replicate in range(simulation_replicates):
        profile = profiles[replicate % len(profiles)]
        profile_name = str(profile["name"])
        profile_counts[profile_name] += 1
        c80_ref = float(profile["c80_ref"])
        w = float(profile["w"])
        truth = {
            "B96": {
                "c80_ref": c80_ref,
                "w": w,
                "c80_at_width": lambda _width, value=c80_ref: value,
            },
            "B288": {
                "c80_ref": c80_ref,
                "w": w,
                "c80_at_width": lambda _width, value=c80_ref: value,
            },
            "D288": {
                "c80_ref": c80_ref,
                "beta": -0.48,
                "w": w,
                "c80_at_width": lambda width, value=c80_ref: value
                * (width / reference_width) ** -0.48,
            },
            "C96": {
                "c80_ref": c80_ref,
                "w": w,
                "c80_at_width": lambda width, value=c80_ref: value
                * scale(width),
            },
        }
        # B96와 B288은 같은 상수 생성 자료를 공유해야 자료량만 다른 비교가 된다.
        constant_x, constant_y, constant_s = simulate_trials(
            widths=widths,
            trials_per_width=trials_per_width,
            c80_at_width=truth["B288"]["c80_at_width"],
            w=truth["B288"]["w"],
            lapse=lapse,
            seed=seed + 10_000 + replicate * 3,
            delta_e00_levels=delta_e00_levels,
        )
        log_x, log_y, log_s = simulate_trials(
            widths=widths,
            trials_per_width=trials_per_width,
            c80_at_width=truth["D288"]["c80_at_width"],
            w=truth["D288"]["w"],
            lapse=lapse,
            seed=seed + 10_001 + replicate * 3,
            delta_e00_levels=delta_e00_levels,
        )
        szafir_x, szafir_y, szafir_s = simulate_trials(
            widths=widths,
            trials_per_width=trials_per_width,
            c80_at_width=truth["C96"]["c80_at_width"],
            w=truth["C96"]["w"],
            lapse=lapse,
            seed=seed + 10_002 + replicate * 3,
            delta_e00_levels=delta_e00_levels,
        )
        fits = {
            "B96": fit_b96(
                constant_x,
                constant_y,
                constant_s,
                reference_width=reference_width,
                lapse=lapse,
            ),
            "B288": fit_b288(
                constant_x, constant_y, constant_s, lapse=lapse
            ),
            "D288": fit_d288(
                log_x,
                log_y,
                log_s,
                reference_width=reference_width,
                lapse=lapse,
            ),
            "C96": fit_c96(
                szafir_x,
                szafir_y,
                szafir_s,
                reference_width=reference_width,
                relative_scale=scale,
                lapse=lapse,
            ),
        }

        for model, fit in fits.items():
            if not fit.converged or not math.isfinite(fit.negative_log_likelihood):
                continue
            _record_fit(
                accumulators[model],
                model=model,
                fit=fit,
                truth=truth[model],
                held_out_width=held_out_width,
            )
            _record_fit(
                profile_accumulators[profile_name][model],
                model=model,
                fit=fit,
                truth=truth[model],
                held_out_width=held_out_width,
            )

    models = {
        model: _accumulator_summary(
            accumulator,
            attempts=simulation_replicates,
            model=model,
        )
        for model, accumulator in accumulators.items()
    }
    profile_results = {}
    for profile in profiles:
        profile_name = str(profile["name"])
        count = profile_counts[profile_name]
        model_results = {
            model: _accumulator_summary(
                profile_accumulators[profile_name][model],
                attempts=count,
                model=model,
            )
            for model in model_names
        }
        profile_reasons = [
            reason
            for model, metrics in model_results.items()
            for reason in _metric_failure(
                model=model, scope=profile_name, metrics=metrics
            )
        ]
        profile_results[profile_name] = {
            "truth": dict(profile),
            "simulation_replicates": count,
            "models": model_results,
            "passed": not profile_reasons,
            "reasons": profile_reasons,
        }

    aggregate_reasons = [
        reason
        for model, metrics in models.items()
        for reason in _metric_failure(model=model, scope="overall", metrics=metrics)
    ]
    profile_reasons = [
        reason
        for result in profile_results.values()
        for reason in result["reasons"]
    ]
    return {
        "simulation_replicates": simulation_replicates,
        "held_out_width_deg": held_out_width,
        "sampling_design": "fixed_absolute_delta_e00_levels",
        "delta_e00_levels": [float(value) for value in delta_e00_levels],
        "profile_counts": profile_counts,
        "profiles": [dict(profile) for profile in profiles],
        "models": models,
        "profile_results": profile_results,
        "worst_case_and_upper_quantile_gate": {
            "passed": not aggregate_reasons and not profile_reasons,
            "reasons": aggregate_reasons + profile_reasons,
            "rule": (
                "all aggregate and every low/central/high profile must pass "
                "convergence, median-error, and 90th-percentile-error criteria"
            ),
        },
    }


def _bootstrap_diagnostic(
    result: BootstrapResult,
    *,
    evaluation_truth: dict[float, float],
    required_parameter_truth: dict[str, float],
) -> dict:
    """대표 bootstrap의 수렴과 알려진 진실값 CI 포함 여부를 기록한다."""
    summary = _bootstrap_summary(result, tuple(evaluation_truth))
    c80_coverage: dict[str, bool] = {}
    if result.successful_replicates:
        for width, truth in evaluation_truth.items():
            low, high = result.c80_interval(width)
            c80_coverage[f"{width:.8g}"] = bool(low <= truth <= high)
    parameter_coverage = {
        name: bool(
            name in result.intervals
            and result.intervals[name][0] <= truth <= result.intervals[name][1]
        )
        for name, truth in required_parameter_truth.items()
    }
    coverage_passed = bool(c80_coverage) and all(c80_coverage.values()) and all(
        parameter_coverage.values()
    )
    convergence_passed = (
        result.converged
        and result.convergence_rate
        >= RECOVERY_CRITERIA["minimum_bootstrap_convergence_rate"]
    )
    return {
        **summary,
        "truth_coverage": {
            "c80_by_width": c80_coverage,
            "required_parameters": parameter_coverage,
            "all_required_truth_covered": coverage_passed,
        },
        "passed": convergence_passed and coverage_passed,
    }


def _paired_reference_integrity(
    b96_fit: FitResult,
    c96_fit: FitResult,
    b96_bootstrap: BootstrapResult,
    c96_bootstrap: BootstrapResult,
    *,
    reference_width: float,
) -> dict:
    """B96/C96가 같은 기준굵기 자료·난수열을 썼음을 수치적으로 증명한다."""
    b_c80 = tuple(b96_bootstrap.estimates.get("c80", ()))
    c_c80 = tuple(c96_bootstrap.estimates.get("c80_ref", ()))
    b_w = tuple(b96_bootstrap.estimates.get("w", ()))
    c_w = tuple(c96_bootstrap.estimates.get("w", ()))
    checks = {
        "same_reference_width": bool(
            b96_fit.reference_width == c96_fit.reference_width == reference_width
        ),
        "same_input_trial_count": b96_fit.n_trials == c96_fit.n_trials,
        "same_point_c80": math.isclose(
            b96_fit.parameters["c80"], c96_fit.parameters["c80_ref"],
            rel_tol=0.0, abs_tol=1e-12,
        ),
        "same_point_w": math.isclose(
            b96_fit.parameters["w"], c96_fit.parameters["w"],
            rel_tol=0.0, abs_tol=1e-12,
        ),
        "same_bootstrap_seed": b96_bootstrap.seed == c96_bootstrap.seed,
        "same_successful_replicates": (
            b96_bootstrap.successful_replicates
            == c96_bootstrap.successful_replicates
        ),
        "same_bootstrap_c80_draws": b_c80 == c_c80 and bool(b_c80),
        "same_bootstrap_w_draws": b_w == c_w and bool(b_w),
    }
    return {
        "rule": (
            "B96 and C96 use the identical reference-width trials and bootstrap "
            "uniform stream; their reference fits and draws must match exactly"
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _assess_recovery(
    monte_carlo: dict,
    representative_bootstrap: dict,
    *,
    simulation_replicates: int,
    bootstrap_replicates: int,
) -> tuple[bool, list[str], dict]:
    """횟수·stress profile·대표 CI를 모두 AND로 묶어 fail-closed 판정한다."""
    reasons: list[str] = []
    enough_simulations = (
        simulation_replicates
        >= RECOVERY_CRITERIA["minimum_simulation_replicates_for_pass"]
    )
    enough_bootstraps = (
        bootstrap_replicates
        >= RECOVERY_CRITERIA["minimum_bootstrap_replicates_for_pass"]
    )
    if not enough_simulations:
        reasons.append(
            f"simulation_replicates {simulation_replicates} < "
            f"{RECOVERY_CRITERIA['minimum_simulation_replicates_for_pass']}"
        )
    if not enough_bootstraps:
        reasons.append(
            f"bootstrap_replicates {bootstrap_replicates} < "
            f"{RECOVERY_CRITERIA['minimum_bootstrap_replicates_for_pass']}"
        )
    profile_gate = monte_carlo.get("worst_case_and_upper_quantile_gate", {})
    if profile_gate.get("passed") is not True:
        reasons.extend(profile_gate.get("reasons", ["stress profile gate 실패"]))
    if representative_bootstrap.get("passed") is not True:
        reasons.extend(representative_bootstrap.get("reasons", ["대표 bootstrap gate 실패"]))
    integrity = {
        "minimum_simulation_replicates_met": enough_simulations,
        "minimum_bootstrap_replicates_met": enough_bootstraps,
        "all_profile_gates_passed": profile_gate.get("passed") is True,
        "representative_bootstrap_passed": representative_bootstrap.get("passed") is True,
    }
    integrity["passed"] = all(integrity.values())
    return not reasons and integrity["passed"], reasons, integrity


def run_parameter_recovery(
    *,
    seed: int = 20260914,
    trials_per_width: int = 96,
    bootstrap_replicates: int = 40,
    simulation_replicates: int = 100,
    delta_e00_levels: Sequence[float] = DEFAULT_FIXED_DELTA_E00_LEVELS,
    widths_deg: Sequence[float] = DEFAULT_WIDTHS_DEG,
    reference_width_deg: float | None = None,
    direction_lab: Sequence[float] = DEFAULT_DIRECTION_LAB,
    source_probability_coordinate: float = DEFAULT_SZAFIR_PROBABILITY,
    experiment_config_sha256: str | None = None,
    fixed_lapse: float = 0.02,
) -> dict:
    """상수·log-width·Szafir 생성 시나리오에서 알려진 모수를 회복한다.

    결과는 연구 결론이 아니라 분석 코드 점검 자료다. B96과 C96은 기준 굵기
    시행만, B288과 D288은 세 굵기 전체 시행을 사용한다.
    """
    trials = _positive_integer(trials_per_width, "trials_per_width", minimum=12)
    bootstrap_count = _positive_integer(
        bootstrap_replicates, "bootstrap_replicates"
    )
    simulation_count = _positive_integer(
        simulation_replicates, "simulation_replicates"
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed는 정수여야 합니다.")
    fixed_levels = tuple(float(value) for value in delta_e00_levels)
    # simulate_trials의 단일 검증 경로를 이용해 수준 자체도 여기서 검증한다.
    if len(fixed_levels) != 12:
        raise ValueError("동결 설계 후보에는 정확히 12개 delta_e00_levels가 필요합니다.")

    widths = tuple(float(value) for value in widths_deg)
    if (
        len(widths) != 3
        or any(not math.isfinite(value) or value <= 0.0 for value in widths)
        or not widths[0] < widths[1] < widths[2]
    ):
        raise ValueError("widths_deg에는 엄격히 증가하는 양의 굵기 세 개가 필요합니다.")
    reference = (
        widths[1]
        if reference_width_deg is None
        else float(reference_width_deg)
    )
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference_width_deg는 양의 유한값이어야 합니다.")
    if not any(math.isclose(reference, width, rel_tol=1e-10, abs_tol=1e-12) for width in widths):
        raise ValueError("reference_width_deg는 세 교정 굵기 중 하나여야 합니다.")
    direction = tuple(float(value) for value in direction_lab)
    if len(direction) != 3 or any(not math.isfinite(value) for value in direction):
        raise ValueError("direction_lab에는 유한한 값 세 개가 필요합니다.")
    probability = float(source_probability_coordinate)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("source_probability_coordinate는 0과 1 사이여야 합니다.")
    if experiment_config_sha256 is not None and not (
        isinstance(experiment_config_sha256, str)
        and len(experiment_config_sha256) == 64
        and all(character in "0123456789abcdef" for character in experiment_config_sha256.lower())
    ):
        raise ValueError("experiment_config_sha256은 64자리 16진수여야 합니다.")
    lapse = float(fixed_lapse)
    if not math.isfinite(lapse) or not 0.0 <= lapse < 0.20:
        raise ValueError("fixed_lapse는 0 이상 0.20 미만이어야 합니다.")

    scale = _szafir_scale(
        reference_width=reference,
        direction_lab=direction,
        probability=probability,
    )
    derived_profiles = _derive_recovery_profiles(
        delta_e00_levels=fixed_levels,
        widths=widths,
        reference_width=reference,
        direction_lab=direction,
        probability=probability,
    )
    central_profile = next(
        profile for profile in derived_profiles if profile["name"] == "central"
    )
    central_c80 = float(central_profile["c80_ref"])
    central_w = float(central_profile["w"])

    # 1) 폭과 무관한 상수 C80: B96/B288의 대표 회복을 확인한다.
    constant_truth = {"c80": central_c80, "w": central_w, "lapse": lapse}
    x_constant, y_constant, s_constant = simulate_trials(
        widths=widths,
        trials_per_width=trials,
        c80_at_width=lambda _width: constant_truth["c80"],
        w=constant_truth["w"],
        lapse=lapse,
        seed=seed,
        delta_e00_levels=fixed_levels,
    )
    b96 = fit_b96(
        x_constant,
        y_constant,
        s_constant,
        reference_width=reference,
        lapse=lapse,
    )
    b288 = fit_b288(x_constant, y_constant, s_constant, lapse=lapse)
    b288_bootstrap = parametric_bootstrap(
        b288,
        x_constant,
        y_constant,
        s_constant,
        replicates=bootstrap_count,
        seed=seed + 1000,
    )

    # 2) 굵기 의존 C80: beta 부호를 강제하지 않은 D288 회복을 확인한다.
    log_truth = {
        "c80_ref": central_c80,
        "beta": -0.48,
        "w": central_w,
        "lapse": lapse,
    }

    def log_c80(width: float) -> float:
        return log_truth["c80_ref"] * (width / reference) ** log_truth["beta"]

    x_log, y_log, s_log = simulate_trials(
        widths=widths,
        trials_per_width=trials,
        c80_at_width=log_c80,
        w=log_truth["w"],
        lapse=lapse,
        seed=seed + 1,
        delta_e00_levels=fixed_levels,
    )
    d288 = fit_d288(
        x_log, y_log, s_log, reference_width=reference, lapse=lapse
    )
    d_bootstrap = parametric_bootstrap(
        d288,
        x_log,
        y_log,
        s_log,
        replicates=bootstrap_count,
        seed=seed + 1001,
    )

    # 3) 정상색각 선 모델에서는 절대값이 아니라 상대 형태만 빌려온다.
    szafir_truth = {"c80_ref": central_c80, "w": central_w, "lapse": lapse}
    x_szafir, y_szafir, s_szafir = simulate_trials(
        widths=widths,
        trials_per_width=trials,
        c80_at_width=lambda width: szafir_truth["c80_ref"] * scale(width),
        w=szafir_truth["w"],
        lapse=lapse,
        seed=seed + 2,
        delta_e00_levels=fixed_levels,
    )
    c96 = fit_c96(
        x_szafir,
        y_szafir,
        s_szafir,
        reference_width=reference,
        relative_scale=scale,
        lapse=lapse,
    )
    # 주 비교 B96/C96는 반드시 동일한 기준굵기 96회에서 출발해야 한다.
    paired_b96 = fit_b96(
        x_szafir,
        y_szafir,
        s_szafir,
        reference_width=reference,
        lapse=lapse,
    )
    c_bootstrap = parametric_bootstrap(
        c96,
        x_szafir,
        y_szafir,
        s_szafir,
        replicates=bootstrap_count,
        seed=seed + 1002,
    )
    paired_b96_bootstrap = parametric_bootstrap(
        paired_b96,
        x_szafir,
        y_szafir,
        s_szafir,
        replicates=bootstrap_count,
        seed=seed + 1002,
    )

    monte_carlo = _monte_carlo_recovery(
        seed=seed,
        simulation_replicates=simulation_count,
        trials_per_width=trials,
        widths=widths,
        reference_width=reference,
        lapse=lapse,
        delta_e00_levels=fixed_levels,
        direction_lab=direction,
        probability=probability,
    )
    # 대표 bootstrap의 coverage gate는 직접 적합되는 기준 C80(그리고 D의
    # beta)에 적용한다. 다른 굵기의 C80은 이 모수들의 결정론적 변환이므로
    # 개별 백분위 구간까지 동시에 진실값을 감싸라고 요구하지 않는다.
    reference_truth = {reference: central_c80}
    representative_models = {
        "B96": _bootstrap_diagnostic(
            paired_b96_bootstrap,
            evaluation_truth=reference_truth,
            required_parameter_truth={"c80": central_c80},
        ),
        "B288": _bootstrap_diagnostic(
            b288_bootstrap,
            evaluation_truth=reference_truth,
            required_parameter_truth={"c80": central_c80},
        ),
        "C96": _bootstrap_diagnostic(
            c_bootstrap,
            evaluation_truth=reference_truth,
            required_parameter_truth={"c80_ref": central_c80},
        ),
        "D288": _bootstrap_diagnostic(
            d_bootstrap,
            evaluation_truth=reference_truth,
            required_parameter_truth={"c80_ref": central_c80, "beta": -0.48},
        ),
    }
    paired_integrity = _paired_reference_integrity(
        paired_b96,
        c96,
        paired_b96_bootstrap,
        c_bootstrap,
        reference_width=reference,
    )
    bootstrap_reasons = [
        f"{model}: 대표 bootstrap 수렴 또는 진실값 CI 포함 실패"
        for model, diagnostic in representative_models.items()
        if diagnostic["passed"] is not True
    ]
    if paired_integrity["passed"] is not True:
        bootstrap_reasons.append("B96/C96 paired-reference bootstrap 무결성 실패")
    representative_bootstrap = {
        "models": representative_models,
        "paired_reference_integrity": paired_integrity,
        "passed": not bootstrap_reasons,
        "reasons": bootstrap_reasons,
    }
    passed, reasons, recovery_integrity = _assess_recovery(
        monte_carlo,
        representative_bootstrap,
        simulation_replicates=simulation_count,
        bootstrap_replicates=bootstrap_count,
    )

    all_fits = (b96, b288, d288, c96, paired_b96)
    provenance = implementation_hashes("parameter_recovery")
    return {
        "artifact_type": "synthetic_parameter_recovery",
        "data_origin": "synthetic",
        "research_conclusion_allowed": False,
        "implementation_hashes": provenance,
        # Deprecated compatibility alias. It must remain exactly equal to the
        # complete dependency map; it is no longer a wrapper-only shortcut.
        "implementation_sha256": dict(provenance),
        "formula": (
            "p=0.25+(0.75-lapse)*sigmoid(q80+(ln(x)-ln(C80(s)))/w)"
        ),
        "seed": seed,
        "widths_deg": list(widths),
        "reference_width_deg": reference,
        "config_sha256": (
            experiment_config_sha256.lower()
            if experiment_config_sha256 is not None
            else None
        ),
        # Deprecated compatibility alias. New consumers use config_sha256.
        "experiment_config_sha256": (
            experiment_config_sha256.lower()
            if experiment_config_sha256 is not None
            else None
        ),
        "trials_per_width": trials,
        "bootstrap_replicates": bootstrap_count,
        "simulation_replicates": simulation_count,
        "fixed_lapse": lapse,
        "sampling_plan": {
            "status": (
                "frozen_human_config_bound"
                if experiment_config_sha256 is not None
                else "synthetic_candidate_not_human_frozen"
            ),
            "design": "same_fixed_absolute_levels_at_every_width",
            "delta_e00_levels": list(fixed_levels),
            "must_rerun_with_frozen_human_config": experiment_config_sha256 is None,
        },
        "all_fits_converged": all(fit.converged for fit in all_fits),
        "recovery_criteria": dict(RECOVERY_CRITERIA),
        "passed": passed,
        "reasons": reasons,
        "recovery_integrity": recovery_integrity,
        "monte_carlo": monte_carlo,
        "representative_bootstrap": representative_bootstrap,
        "constant_scenario": {
            "truth": constant_truth,
            "fits": {
                "B96": _fit_summary(b96, widths),
                "B288": _fit_summary(b288, widths),
            },
            "bootstrap": {
                "B288": representative_models["B288"],
            },
        },
        "log_width_scenario": {
            "truth": log_truth,
            "fits": {"D288": _fit_summary(d288, widths)},
            "bootstrap": representative_models["D288"],
        },
        "szafir_relative_scenario": {
            "truth": szafir_truth,
            "direction_lab": list(direction),
            "source_probability_coordinate": probability,
            "relative_scale": {f"{width:.8g}": scale(width) for width in widths},
            "fits": {
                "B96_same_reference_trials": _fit_summary(paired_b96, widths),
                "C96": _fit_summary(c96, widths),
            },
            "bootstrap": {
                "B96": representative_models["B96"],
                "C96": representative_models["C96"],
                "paired_reference_integrity": paired_integrity,
            },
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="합성 direct-C80 자료로 결정론적 parameter recovery를 실행합니다."
    )
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--trials-per-width", type=int, default=96)
    parser.add_argument("--bootstrap-replicates", type=int, default=40)
    parser.add_argument("--simulation-replicates", type=int, default=100)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        help=(
            "사람 실험용 동결 1.0 config. 지정하면 그 파일의 실제 굵기, "
            "12개 색차, Lab 방향과 고정 lapse로 recovery하고 SHA256을 묶습니다."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="생략하면 stdout, 지정하면 같은 JSON을 UTF-8 파일로 저장합니다.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    recovery_arguments = {
        "seed": args.seed,
        "trials_per_width": args.trials_per_width,
        "bootstrap_replicates": args.bootstrap_replicates,
        "simulation_replicates": args.simulation_replicates,
    }
    try:
        if args.experiment_config is not None:
            from experiment_runner import assert_schedule_config

            raw = args.experiment_config.read_bytes()
            config = json.loads(raw.decode("utf-8"))
            if not isinstance(config, dict):
                raise ValueError("experiment config 최상위는 객체여야 합니다.")
            assert_schedule_config(config)
            recovery_arguments.update(
                {
                    "trials_per_width": int(config["calibration"]["trials_per_width"]),
                    "delta_e00_levels": tuple(config["sampling"]["delta_e00_levels"]),
                    "widths_deg": tuple(
                        row["width_deg"] for row in config["calibration"]["widths"]
                    ),
                    "reference_width_deg": config["calibration"]["widths"][1]["width_deg"],
                    "direction_lab": tuple(config["color_design"]["direction_lab"]),
                    "source_probability_coordinate": config["model_plan"][
                        "szafir_probability_coordinate"
                    ],
                    "fixed_lapse": config["observer_profile"]["fixed_lapse"],
                    "experiment_config_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        payload = run_parameter_recovery(**recovery_arguments)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        print(f"모수 회복 차단됨: {error}")
        return 2
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["all_fits_converged"] and payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RECOVERY_CRITERIA", "run_parameter_recovery", "simulate_trials"]
