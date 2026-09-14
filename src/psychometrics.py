"""4AFC direct-C80 심리측정 함수와 사전등록 모델의 최대우도 적합.

이 모듈은 실제 참가자 결과를 만들지 않는다. 양의 색차 ``x``에서의 관측
정답(0/1)을 입력받아 B96, B288, C96, D288을 적합하고, 모수적
부트스트랩으로 수치적 불확실성을 계산하는 분석 기반만 제공한다.

SciPy가 없는 실행 환경에서도 같은 계산을 재현할 수 있도록 작은 차원의
결정론적 Nelder--Mead 구현을 사용한다.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from numbers import Real

import numpy as np


GUESS_RATE = 0.25
TARGET_ACCURACY = 0.80
MAX_LAPSE = 0.20

RelativeScale = Callable[[float], float]


def _finite_scalar(value: Real, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        qualifier = "0보다 큰 " if positive else ""
        raise TypeError(f"{name}은 {qualifier}유한한 숫자여야 합니다.")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "0보다 큰 " if positive else ""
        raise ValueError(f"{name}은 {qualifier}유한한 숫자여야 합니다.")
    return number


def _validate_lapse(lapse: Real) -> float:
    value = _finite_scalar(lapse, "lapse")
    if not 0.0 <= value < MAX_LAPSE:
        raise ValueError("lapse는 0 이상 0.20 미만이어야 합니다.")
    return value


def _positive_array(values: Sequence[Real] | np.ndarray | Real, name: str) -> np.ndarray:
    if isinstance(values, (bool, np.bool_)):
        raise TypeError(f"{name}은 0보다 큰 유한한 숫자여야 합니다.")
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name}은 0보다 큰 유한한 숫자여야 합니다.") from error
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name}의 모든 값은 0보다 큰 유한한 숫자여야 합니다.")
    return result


def _sigmoid(value: np.ndarray) -> np.ndarray:
    """overflow 없이 logistic sigmoid를 계산한다."""
    result = np.empty_like(value, dtype=float)
    nonnegative = value >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
    exp_value = np.exp(value[~nonnegative])
    result[~nonnegative] = exp_value / (1.0 + exp_value)
    return result


def direct_c80_probability(
    delta_e00: Sequence[Real] | np.ndarray | Real,
    c80: Sequence[Real] | np.ndarray | Real,
    w: Real,
    lapse: Real = 0.02,
) -> np.ndarray | float:
    """4AFC direct-C80 logistic의 실제 정답률을 반환한다.

    식은 다음과 같다.

    ``p=.25+(.75-lapse)*sigmoid(q80+(ln(x)-ln(C80))/w)``

    여기서 ``q80=logit((.80-.25)/(.75-lapse))``다. 따라서 허용된 모든
    ``lapse``와 ``w``에서 ``x == C80``이면 부동소수점 오차 안에서
    정확히 0.80이 된다.
    """
    x = _positive_array(delta_e00, "delta_e00")
    threshold = _positive_array(c80, "c80")
    width = _finite_scalar(w, "w", positive=True)
    lapse_value = _validate_lapse(lapse)
    try:
        x, threshold = np.broadcast_arrays(x, threshold)
    except ValueError as error:
        raise ValueError("delta_e00와 c80은 서로 broadcast 가능한 모양이어야 합니다.") from error

    normalized_target = (TARGET_ACCURACY - GUESS_RATE) / (
        (1.0 - GUESS_RATE) - lapse_value
    )
    q80 = math.log(normalized_target / (1.0 - normalized_target))
    linear = q80 + (np.log(x) - np.log(threshold)) / width
    probability = GUESS_RATE + ((1.0 - GUESS_RATE) - lapse_value) * _sigmoid(linear)
    if probability.ndim == 0:
        return float(probability)
    return probability


def _trial_arrays(
    delta_e00: Sequence[Real] | np.ndarray,
    correct: Sequence[int | bool] | np.ndarray,
    stroke_width: Sequence[Real] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _positive_array(delta_e00, "delta_e00")
    widths = _positive_array(stroke_width, "stroke_width")
    answers = np.asarray(correct)
    if x.ndim != 1 or widths.ndim != 1 or answers.ndim != 1:
        raise ValueError("delta_e00, correct, stroke_width는 모두 1차원이어야 합니다.")
    if not (x.size == widths.size == answers.size):
        raise ValueError("delta_e00, correct, stroke_width의 길이가 같아야 합니다.")
    if x.size == 0:
        raise ValueError("적합에는 시행이 하나 이상 필요합니다.")
    if answers.dtype.kind not in "biuf" or np.any(~np.isfinite(answers.astype(float))):
        raise TypeError("correct는 0 또는 1이어야 합니다.")
    numeric_answers = answers.astype(float)
    if np.any((numeric_answers != 0.0) & (numeric_answers != 1.0)):
        raise ValueError("correct는 0 또는 1이어야 합니다.")
    return x.astype(float), numeric_answers, widths.astype(float)


def _evaluate_relative_scale(
    relative_scale: RelativeScale, widths: np.ndarray, name: str = "relative_scale"
) -> np.ndarray:
    if not callable(relative_scale):
        raise TypeError(f"{name}은 굵기를 받아 양의 배율을 반환하는 함수여야 합니다.")
    values = np.empty(widths.shape, dtype=float)
    for index, width in np.ndenumerate(widths):
        try:
            raw = relative_scale(float(width))
        except Exception as error:  # 호출자 함수의 문맥을 보존해 명확한 오류로 바꾼다.
            raise ValueError(f"{name}({float(width):g}) 계산에 실패했습니다.") from error
        values[index] = _finite_scalar(raw, f"{name}({float(width):g})", positive=True)
    return values


@dataclass(frozen=True)
class FitResult:
    """한 심리측정 모델의 MLE와 수렴 정보를 보관한다."""

    model: str
    parameters: dict[str, float]
    lapse: float
    n_trials: int
    negative_log_likelihood: float
    converged: bool
    iterations: int
    message: str
    reference_width: float | None = None
    relative_scale: RelativeScale | None = field(default=None, repr=False, compare=False)

    def c80(self, stroke_width: Sequence[Real] | np.ndarray | Real) -> np.ndarray | float:
        """지정한 굵기에서 모델이 예측한 C80을 반환한다."""
        widths = _positive_array(stroke_width, "stroke_width")
        if self.model in {"B96", "B288"}:
            result = np.full(widths.shape, self.parameters["c80"], dtype=float)
        elif self.model == "D288":
            if self.reference_width is None:
                raise RuntimeError("D288 적합에 reference_width가 없습니다.")
            result = self.parameters["c80_ref"] * (
                widths / self.reference_width
            ) ** self.parameters["beta"]
        elif self.model == "C96":
            if self.relative_scale is None:
                raise RuntimeError("C96 적합에 relative_scale 함수가 없습니다.")
            result = self.parameters["c80_ref"] * _evaluate_relative_scale(
                self.relative_scale, widths
            )
        else:
            raise RuntimeError(f"알 수 없는 모델입니다: {self.model}")
        if result.ndim == 0:
            return float(result)
        return result

    def probability(
        self,
        delta_e00: Sequence[Real] | np.ndarray | Real,
        stroke_width: Sequence[Real] | np.ndarray | Real,
    ) -> np.ndarray | float:
        """지정한 자극에서 적합 모델의 정답 확률을 반환한다."""
        return direct_c80_probability(
            delta_e00,
            self.c80(stroke_width),
            self.parameters["w"],
            self.lapse,
        )


@dataclass(frozen=True)
class BootstrapResult:
    """모수적 부트스트랩 추정치, 백분위 CI, 수렴률."""

    model: str
    seed: int
    confidence: float
    requested_replicates: int
    successful_replicates: int
    convergence_rate: float
    converged: bool
    intervals: dict[str, tuple[float, float]]
    estimates: dict[str, tuple[float, ...]]
    reference_width: float | None = None
    relative_scale: RelativeScale | None = field(default=None, repr=False, compare=False)

    def c80_interval(self, stroke_width: Real) -> tuple[float, float]:
        """저장된 bootstrap 복제에서 특정 굵기의 C80 백분위 CI를 구한다."""
        width = _finite_scalar(stroke_width, "stroke_width", positive=True)
        if not self.successful_replicates:
            raise ValueError("성공한 bootstrap 복제가 없어 C80 CI를 계산할 수 없습니다.")
        if self.model in {"B96", "B288"}:
            values = np.asarray(self.estimates["c80"], dtype=float)
        elif self.model == "D288":
            if self.reference_width is None:
                raise RuntimeError("D288 bootstrap에 reference_width가 없습니다.")
            values = np.asarray(self.estimates["c80_ref"], dtype=float) * (
                width / self.reference_width
            ) ** np.asarray(self.estimates["beta"], dtype=float)
        elif self.model == "C96":
            if self.relative_scale is None:
                raise RuntimeError("C96 bootstrap에 relative_scale 함수가 없습니다.")
            scale = _evaluate_relative_scale(
                self.relative_scale, np.asarray(width, dtype=float)
            )
            values = np.asarray(self.estimates["c80_ref"], dtype=float) * float(scale)
        else:
            raise RuntimeError(f"알 수 없는 모델입니다: {self.model}")
        alpha = (1.0 - self.confidence) / 2.0
        low, high = np.quantile(values, (alpha, 1.0 - alpha))
        return float(low), float(high)


def _negative_log_likelihood(probability: np.ndarray, correct: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(-np.sum(correct * np.log(clipped) + (1.0 - correct) * np.log1p(-clipped)))


@dataclass(frozen=True)
class _OptimizationResult:
    x: np.ndarray
    fun: float
    converged: bool
    iterations: int
    message: str


def _nelder_mead(
    objective: Callable[[np.ndarray], float],
    initial: np.ndarray,
    *,
    step: np.ndarray,
    max_iterations: int = 3000,
    x_tolerance: float = 1e-8,
    f_tolerance: float = 1e-8,
) -> _OptimizationResult:
    """작은 차원의 결정론적 Nelder--Mead 최소화."""
    dimension = initial.size
    simplex = np.tile(initial, (dimension + 1, 1))
    for index in range(dimension):
        simplex[index + 1, index] += step[index]
    values = np.asarray([objective(point) for point in simplex], dtype=float)

    for iteration in range(1, max_iterations + 1):
        order = np.argsort(values, kind="stable")
        simplex = simplex[order]
        values = values[order]
        coordinate_spread = float(np.max(np.abs(simplex[1:] - simplex[0])))
        value_spread = float(np.max(np.abs(values[1:] - values[0])))
        if coordinate_spread <= x_tolerance and value_spread <= f_tolerance:
            return _OptimizationResult(
                simplex[0].copy(), float(values[0]), True, iteration, "수렴"
            )

        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + (centroid - simplex[-1])
        reflected_value = objective(reflected)
        if reflected_value < values[0]:
            expanded = centroid + 2.0 * (reflected - centroid)
            expanded_value = objective(expanded)
            if expanded_value < reflected_value:
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
            continue
        if reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
            continue

        if reflected_value < values[-1]:
            contracted = centroid + 0.5 * (reflected - centroid)
            comparison = reflected_value
        else:
            contracted = centroid + 0.5 * (simplex[-1] - centroid)
            comparison = values[-1]
        contracted_value = objective(contracted)
        if contracted_value < comparison:
            simplex[-1], values[-1] = contracted, contracted_value
            continue

        best = simplex[0].copy()
        simplex[1:] = best + 0.5 * (simplex[1:] - best)
        values[1:] = np.asarray([objective(point) for point in simplex[1:]])

    order = np.argsort(values, kind="stable")
    best_index = int(order[0])
    return _OptimizationResult(
        simplex[best_index].copy(),
        float(values[best_index]),
        False,
        max_iterations,
        "최대 반복 횟수에 도달",
    )


def _bounded_objective(
    objective: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]]
) -> Callable[[np.ndarray], float]:
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)

    def wrapped(parameters: np.ndarray) -> float:
        below = np.maximum(lower - parameters, 0.0)
        above = np.maximum(parameters - upper, 0.0)
        if np.any(below) or np.any(above):
            return 1e12 + 1e8 * float(np.sum(below * below + above * above))
        value = objective(parameters)
        return value if math.isfinite(value) else 1e12

    return wrapped


def _constant_fit(
    model: str,
    x: np.ndarray,
    answers: np.ndarray,
    widths: np.ndarray,
    *,
    lapse: float,
    reference_width: float | None,
    relative_scale: RelativeScale | None = None,
) -> FitResult:
    log_x = np.log(x)
    initial = np.asarray([float(np.median(log_x)), math.log(0.35)])
    bounds = [
        (float(np.min(log_x) - 4.0), float(np.max(log_x) + 4.0)),
        (math.log(0.02), math.log(10.0)),
    ]

    if model == "C96":
        assert relative_scale is not None
        scale = _evaluate_relative_scale(relative_scale, widths)
    else:
        scale = np.ones_like(x)

    def nll(parameters: np.ndarray) -> float:
        c80 = math.exp(float(parameters[0])) * scale
        w = math.exp(float(parameters[1]))
        probability = np.asarray(direct_c80_probability(x, c80, w, lapse))
        return _negative_log_likelihood(probability, answers)

    optimized = _nelder_mead(
        _bounded_objective(nll, bounds), initial, step=np.asarray([0.25, 0.20])
    )
    c80 = math.exp(float(optimized.x[0]))
    parameters = {
        ("c80_ref" if model == "C96" else "c80"): c80,
        "w": math.exp(float(optimized.x[1])),
    }
    return FitResult(
        model=model,
        parameters=parameters,
        lapse=lapse,
        n_trials=int(x.size),
        negative_log_likelihood=optimized.fun,
        converged=optimized.converged,
        iterations=optimized.iterations,
        message=optimized.message,
        reference_width=reference_width,
        relative_scale=relative_scale,
    )


def fit_b96(
    delta_e00: Sequence[Real] | np.ndarray,
    correct: Sequence[int | bool] | np.ndarray,
    stroke_width: Sequence[Real] | np.ndarray,
    *,
    reference_width: Real,
    lapse: Real = 0.02,
) -> FitResult:
    """기준 굵기 자료만 사용해 상수 C80 모델(B96)을 적합한다."""
    x, answers, widths = _trial_arrays(delta_e00, correct, stroke_width)
    reference = _finite_scalar(reference_width, "reference_width", positive=True)
    lapse_value = _validate_lapse(lapse)
    selected = np.isclose(widths, reference, rtol=1e-10, atol=1e-12)
    if not np.any(selected):
        raise ValueError("reference_width와 일치하는 시행이 없습니다.")
    return _constant_fit(
        "B96",
        x[selected],
        answers[selected],
        widths[selected],
        lapse=lapse_value,
        reference_width=reference,
    )


def fit_b288(
    delta_e00: Sequence[Real] | np.ndarray,
    correct: Sequence[int | bool] | np.ndarray,
    stroke_width: Sequence[Real] | np.ndarray,
    *,
    lapse: Real = 0.02,
) -> FitResult:
    """모든 굵기 자료를 하나의 상수 C80로 적합한다(B288)."""
    x, answers, widths = _trial_arrays(delta_e00, correct, stroke_width)
    lapse_value = _validate_lapse(lapse)
    return _constant_fit(
        "B288", x, answers, widths, lapse=lapse_value, reference_width=None
    )


def fit_c96(
    delta_e00: Sequence[Real] | np.ndarray,
    correct: Sequence[int | bool] | np.ndarray,
    stroke_width: Sequence[Real] | np.ndarray,
    *,
    reference_width: Real,
    relative_scale: RelativeScale,
    lapse: Real = 0.02,
) -> FitResult:
    """기준 굵기의 B96과 Szafir 상대 배율을 결합해 C96을 적합한다."""
    x, answers, widths = _trial_arrays(delta_e00, correct, stroke_width)
    reference = _finite_scalar(reference_width, "reference_width", positive=True)
    lapse_value = _validate_lapse(lapse)
    reference_scale = float(
        _evaluate_relative_scale(relative_scale, np.asarray(reference, dtype=float))
    )
    if not math.isclose(reference_scale, 1.0, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("relative_scale(reference_width)는 1이어야 합니다.")
    selected = np.isclose(widths, reference, rtol=1e-10, atol=1e-12)
    if not np.any(selected):
        raise ValueError("reference_width와 일치하는 시행이 없습니다.")
    return _constant_fit(
        "C96",
        x[selected],
        answers[selected],
        widths[selected],
        lapse=lapse_value,
        reference_width=reference,
        relative_scale=relative_scale,
    )


def fit_d288(
    delta_e00: Sequence[Real] | np.ndarray,
    correct: Sequence[int | bool] | np.ndarray,
    stroke_width: Sequence[Real] | np.ndarray,
    *,
    reference_width: Real | None = None,
    lapse: Real = 0.02,
) -> FitResult:
    """``log(C80)=log(C80_ref)+beta*log(s/s_ref)``를 MLE로 적합한다.

    ``beta``에는 부호 제약을 걸지 않는다. 따라서 자료가 반대 방향을 지지하면
    양수 추정치가 그대로 반환되어 사전 방향 가정을 점검할 수 있다.
    """
    x, answers, widths = _trial_arrays(delta_e00, correct, stroke_width)
    lapse_value = _validate_lapse(lapse)
    reference = (
        float(np.exp(np.mean(np.log(np.unique(widths)))))
        if reference_width is None
        else _finite_scalar(reference_width, "reference_width", positive=True)
    )
    log_x = np.log(x)
    log_ratio = np.log(widths / reference)
    initial = np.asarray([float(np.median(log_x)), 0.0, math.log(0.35)])
    bounds = [
        (float(np.min(log_x) - 4.0), float(np.max(log_x) + 4.0)),
        (-8.0, 8.0),
        (math.log(0.02), math.log(10.0)),
    ]

    def nll(parameters: np.ndarray) -> float:
        c80 = np.exp(float(parameters[0]) + float(parameters[1]) * log_ratio)
        probability = np.asarray(
            direct_c80_probability(x, c80, math.exp(float(parameters[2])), lapse_value)
        )
        return _negative_log_likelihood(probability, answers)

    optimized = _nelder_mead(
        _bounded_objective(nll, bounds),
        initial,
        step=np.asarray([0.25, 0.20, 0.20]),
    )
    return FitResult(
        model="D288",
        parameters={
            "c80_ref": math.exp(float(optimized.x[0])),
            "beta": float(optimized.x[1]),
            "w": math.exp(float(optimized.x[2])),
        },
        lapse=lapse_value,
        n_trials=int(x.size),
        negative_log_likelihood=optimized.fun,
        converged=optimized.converged,
        iterations=optimized.iterations,
        message=optimized.message,
        reference_width=reference,
    )


def _refit_like(
    fit: FitResult,
    x: np.ndarray,
    answers: np.ndarray,
    widths: np.ndarray,
) -> FitResult:
    if fit.model == "B96":
        assert fit.reference_width is not None
        return fit_b96(
            x, answers, widths, reference_width=fit.reference_width, lapse=fit.lapse
        )
    if fit.model == "B288":
        return fit_b288(x, answers, widths, lapse=fit.lapse)
    if fit.model == "C96":
        assert fit.reference_width is not None and fit.relative_scale is not None
        return fit_c96(
            x,
            answers,
            widths,
            reference_width=fit.reference_width,
            relative_scale=fit.relative_scale,
            lapse=fit.lapse,
        )
    if fit.model == "D288":
        assert fit.reference_width is not None
        return fit_d288(
            x, answers, widths, reference_width=fit.reference_width, lapse=fit.lapse
        )
    raise ValueError(f"지원하지 않는 모델입니다: {fit.model}")


def parametric_bootstrap(
    fit: FitResult,
    delta_e00: Sequence[Real] | np.ndarray,
    correct: Sequence[int | bool] | np.ndarray,
    stroke_width: Sequence[Real] | np.ndarray,
    *,
    replicates: int = 200,
    confidence: Real = 0.95,
    seed: int = 0,
    minimum_convergence_rate: Real = 0.80,
) -> BootstrapResult:
    """적합 모델에서 반응을 재생성해 MLE의 백분위 CI를 구한다."""
    x, _, widths = _trial_arrays(delta_e00, correct, stroke_width)
    if isinstance(replicates, bool) or not isinstance(replicates, int):
        raise TypeError("replicates는 1 이상의 정수여야 합니다.")
    if replicates < 1:
        raise ValueError("replicates는 1 이상의 정수여야 합니다.")
    confidence_value = _finite_scalar(confidence, "confidence")
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence는 0보다 크고 1보다 작아야 합니다.")
    convergence_threshold = _finite_scalar(
        minimum_convergence_rate, "minimum_convergence_rate"
    )
    if not 0.0 <= convergence_threshold <= 1.0:
        raise ValueError("minimum_convergence_rate는 0 이상 1 이하여야 합니다.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed는 정수여야 합니다.")

    probability = np.asarray(fit.probability(x, widths), dtype=float)
    generator = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {
        key: [] for key in fit.parameters
    }
    for _ in range(replicates):
        simulated = (generator.random(x.size) < probability).astype(float)
        refit = _refit_like(fit, x, simulated, widths)
        if not refit.converged or not math.isfinite(refit.negative_log_likelihood):
            continue
        for key, value in refit.parameters.items():
            collected[key].append(float(value))

    successful = len(next(iter(collected.values()))) if collected else 0
    intervals: dict[str, tuple[float, float]] = {}
    if successful:
        alpha = (1.0 - confidence_value) / 2.0
        for key, values in collected.items():
            low, high = np.quantile(values, (alpha, 1.0 - alpha))
            intervals[key] = (float(low), float(high))
    rate = successful / replicates
    return BootstrapResult(
        model=fit.model,
        seed=seed,
        confidence=confidence_value,
        requested_replicates=replicates,
        successful_replicates=successful,
        convergence_rate=rate,
        converged=successful >= 2 and rate >= convergence_threshold,
        intervals=intervals,
        estimates={key: tuple(values) for key, values in collected.items()},
        reference_width=fit.reference_width,
        relative_scale=fit.relative_scale,
    )


__all__ = [
    "BootstrapResult",
    "FitResult",
    "GUESS_RATE",
    "MAX_LAPSE",
    "TARGET_ACCURACY",
    "direct_c80_probability",
    "fit_b96",
    "fit_b288",
    "fit_c96",
    "fit_d288",
    "parametric_bootstrap",
]
