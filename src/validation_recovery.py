"""고정 8수준×3블록 검증 설계의 합성 parameter recovery.

사람 반응을 만들거나 대체하지 않는다. 동결 validation plan의 실제 양자화
ΔE00 수준에서 가상 4AFC 반응을 생성해 block 적합, 안정성 게이트, pooled C80
오차와 95% CI coverage를 사전에 점검한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from implementation_provenance import implementation_hashes
from psychometrics import direct_c80_probability
from validation_design import BLOCK_COUNT, ValidationDesignError, validate_plan
from validation_psychometrics import bootstrap_shared_w, fit_shared_w


BLOCK_FIT_CONVERGENCE_GATE = 0.95
BOOTSTRAP_CONVERGENCE_GATE = 0.90
STABLE_RATE_GATE = 0.70
MEDIAN_RELATIVE_ERROR_GATE = 0.20
COVERAGE_GATE = 0.80
CV_LIMIT = 0.20


class ValidationRecoveryError(ValueError):
    """동결 계획 불일치 또는 유효하지 않은 recovery 설정."""


def _read_json(path: str | Path) -> tuple[dict[str, Any], str]:
    file_path = Path(path)
    try:
        raw = file_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except FileNotFoundError as error:
        raise ValidationRecoveryError(f"파일이 없습니다: {file_path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationRecoveryError(f"UTF-8 JSON을 읽지 못했습니다: {file_path}") from error
    if not isinstance(payload, dict):
        raise ValidationRecoveryError("JSON 최상위는 객체여야 합니다.")
    return payload, hashlib.sha256(raw).hexdigest()


def _pool(
    estimates: Sequence[float],
    intervals: Sequence[tuple[float, float]],
    log_covariance: np.ndarray,
) -> tuple[bool, bool, bool, float | None, float | None, float | None]:
    values = np.asarray(estimates, dtype=float)
    cv = float(np.std(values, ddof=1) / np.mean(values))
    compatible = max(low for low, _ in intervals) <= min(high for _, high in intervals)
    if cv > CV_LIMIT or not compatible:
        return False, cv <= CV_LIMIT, compatible, None, None, None
    inverse = np.linalg.pinv(np.asarray(log_covariance, dtype=float), rcond=1e-12)
    ones = np.ones(3, dtype=float)
    denominator = float(ones @ inverse @ ones)
    if not math.isfinite(denominator) or denominator <= 0:
        return False, cv <= CV_LIMIT, compatible, None, None, None
    weights = (inverse @ ones) / denominator
    pooled_log = float(weights @ np.log(values))
    pooled_se = math.sqrt(1.0 / denominator)
    return (
        True,
        True,
        True,
        math.exp(pooled_log),
        math.exp(pooled_log - 1.959963984540054 * pooled_se),
        math.exp(pooled_log + 1.959963984540054 * pooled_se),
    )


def run_validation_recovery(
    actual_delta_e00_levels: Sequence[float],
    *,
    fixed_lapse: float,
    repetitions: int = 200,
    bootstrap_replicates: int = 50,
    seed: int = 20260916,
    plan_sha256: str | None = None,
    config_sha256: str | None = None,
    true_c80_bounds: tuple[float, float] | None = None,
    true_w_bounds: tuple[float, float] | None = None,
    repetitions_per_level_per_block: int = 8,
) -> dict[str, Any]:
    """주어진 실제 8수준 설계로 가상 참가자 recovery를 수행한다."""
    levels = np.asarray(actual_delta_e00_levels, dtype=float)
    if levels.shape != (8,) or np.any(~np.isfinite(levels)) or np.any(levels <= 0):
        raise ValidationRecoveryError("actual_delta_e00_levels에는 양의 유한값 8개가 필요합니다.")
    if np.any(np.diff(levels) <= 0):
        raise ValidationRecoveryError("실제 ΔE00 8수준은 엄격히 증가해야 합니다.")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValidationRecoveryError("repetitions는 1 이상의 정수여야 합니다.")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 2
    ):
        raise ValidationRecoveryError("bootstrap_replicates는 2 이상의 정수여야 합니다.")
    if not 0.0 <= float(fixed_lapse) < 0.20:
        raise ValidationRecoveryError("fixed_lapse는 0 이상 0.20 미만이어야 합니다.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValidationRecoveryError("seed는 0 이상의 정수여야 합니다.")
    if (
        isinstance(repetitions_per_level_per_block, bool)
        or not isinstance(repetitions_per_level_per_block, int)
        or repetitions_per_level_per_block < 1
    ):
        raise ValidationRecoveryError("repetitions_per_level_per_block은 1 이상의 정수여야 합니다.")

    generator = np.random.default_rng(seed)
    # 설계 양 끝의 과도한 경계 효과가 아니라, 8수준 내부에서 C80을 회수할 수
    # 있는지를 검사한다. true C80은 두 번째~일곱 번째 수준 사이 log-uniform,
    # w는 사전 고정 범위 0.20~0.60 log-uniform이다.
    c80_bounds = (
        (float(levels[1]), float(levels[-2]))
        if true_c80_bounds is None
        else tuple(float(value) for value in true_c80_bounds)
    )
    w_bounds = (0.20, 0.60) if true_w_bounds is None else tuple(float(value) for value in true_w_bounds)
    if len(c80_bounds) != 2 or not 0.0 < c80_bounds[0] < c80_bounds[1]:
        raise ValidationRecoveryError("true_c80_bounds는 증가하는 양의 두 값이어야 합니다.")
    if len(w_bounds) != 2 or not 0.0 < w_bounds[0] < w_bounds[1]:
        raise ValidationRecoveryError("true_w_bounds는 증가하는 양의 두 값이어야 합니다.")
    if c80_bounds[0] < levels[0] or c80_bounds[1] > levels[-1]:
        raise ValidationRecoveryError("true C80 범위는 실제 8수준 표시 범위 안이어야 합니다.")
    log_c80_bounds = (math.log(c80_bounds[0]), math.log(c80_bounds[1]))
    x_block = np.repeat(levels, repetitions_per_level_per_block)
    x = np.tile(x_block, BLOCK_COUNT)
    block_indices = np.repeat(np.arange(BLOCK_COUNT), len(x_block))
    block_fit_success = 0
    bootstrap_success = 0
    stable_count = 0
    cv_pass_count = 0
    interval_compatible_count = 0
    relative_errors: list[float] = []
    coverage_count = 0
    numerical_incomplete = 0
    truth_rows: list[dict[str, Any]] = []

    for repetition in range(repetitions):
        true_c80 = math.exp(generator.uniform(*log_c80_bounds))
        true_w = math.exp(generator.uniform(math.log(w_bounds[0]), math.log(w_bounds[1])))
        probability = np.asarray(
            direct_c80_probability(x, true_c80, true_w, fixed_lapse), dtype=float
        )
        responses = (generator.random(x.size) < probability).astype(int)
        fit = fit_shared_w(x, responses, block_indices, lapse=fixed_lapse)
        if fit.converged:
            block_fit_success += BLOCK_COUNT
        else:
            numerical_incomplete += 1
            truth_rows.append(
                {"repetition": repetition + 1, "true_c80": true_c80, "true_w": true_w, "status": "numerical_incomplete"}
            )
            continue
        bootstrap = bootstrap_shared_w(
            fit,
            x,
            responses,
            block_indices,
            replicates=bootstrap_replicates,
            seed=seed + repetition + 1,
            minimum_convergence_rate=BOOTSTRAP_CONVERGENCE_GATE,
        )
        if not bootstrap.converged:
            numerical_incomplete += 1
            truth_rows.append(
                {"repetition": repetition + 1, "true_c80": true_c80, "true_w": true_w, "status": "numerical_incomplete"}
            )
            continue
        bootstrap_success += BLOCK_COUNT
        log_covariance = np.cov(np.log(bootstrap.c80_samples), rowvar=False, ddof=1)
        stable, cv_pass, interval_compatible, pooled, low, high = _pool(
            fit.c80_by_block, bootstrap.c80_intervals, log_covariance
        )
        cv_pass_count += int(cv_pass)
        interval_compatible_count += int(interval_compatible)
        if not stable:
            truth_rows.append(
                {"repetition": repetition + 1, "true_c80": true_c80, "true_w": true_w, "status": "validation_unstable"}
            )
            continue
        stable_count += 1
        assert pooled is not None and low is not None and high is not None
        relative_error = abs(pooled - true_c80) / true_c80
        covered = low <= true_c80 <= high
        relative_errors.append(relative_error)
        coverage_count += int(covered)
        truth_rows.append(
            {
                "repetition": repetition + 1,
                "true_c80": true_c80,
                "true_w": true_w,
                "status": "stable_pooled",
                "pooled_c80": pooled,
                "pooled_ci_low": low,
                "pooled_ci_high": high,
                "relative_error": relative_error,
                "covered": covered,
            }
        )

    block_total = repetitions * BLOCK_COUNT
    block_fit_rate = block_fit_success / block_total
    bootstrap_rate = bootstrap_success / block_total
    stable_rate = stable_count / repetitions
    median_error = float(np.median(relative_errors)) if relative_errors else None
    coverage = coverage_count / stable_count if stable_count else None
    gates = {
        "block_fit_convergence_rate": block_fit_rate >= BLOCK_FIT_CONVERGENCE_GATE,
        "bootstrap_convergence_rate": bootstrap_rate >= BOOTSTRAP_CONVERGENCE_GATE,
        "stable_participant_rate": stable_rate >= STABLE_RATE_GATE,
        "median_pooled_c80_relative_error": (
            median_error is not None and median_error <= MEDIAN_RELATIVE_ERROR_GATE
        ),
        "pooled_95ci_coverage": coverage is not None and coverage >= COVERAGE_GATE,
    }
    recovery_implementation_hashes = implementation_hashes("validation_recovery")
    return {
        "schema_version": "1.0",
        "artifact_type": "validation_parameter_recovery",
        "data_origin": "synthetic",
        "research_conclusion_allowed": False,
        "status": "passed" if all(gates.values()) else "failed",
        "passed": all(gates.values()),
        "must_rerun_with_frozen_validation_plan": plan_sha256 is None,
        "plan_sha256": plan_sha256,
        "config_sha256": config_sha256,
        "implementation_sha256": recovery_implementation_hashes[
            "src/validation_recovery.py"
        ],
        "implementation_hashes": recovery_implementation_hashes,
        "simulation": {
            "repetitions": repetitions,
            "blocks_per_participant": BLOCK_COUNT,
            "trials_per_block": int(x_block.size),
            "actual_delta_e00_levels": [float(value) for value in levels],
            "repetitions_per_level_per_block": repetitions_per_level_per_block,
            "fixed_lapse": float(fixed_lapse),
            "true_c80_sampling": (
                "log-uniform-between-second-and-seventh-actual-level"
                if true_c80_bounds is None
                else "log-uniform-over-calibration-b96-c96-95ci-envelope"
            ),
            "true_c80_bounds": list(c80_bounds),
            "true_w_sampling": "log-uniform-over-recorded-bounds",
            "true_w_bounds": list(w_bounds),
            "bootstrap_replicates_per_block": bootstrap_replicates,
            "seed": seed,
        },
        "metrics": {
            "block_fit_convergence_rate": block_fit_rate,
            "bootstrap_convergence_rate": bootstrap_rate,
            "stable_participant_rate": stable_rate,
            "stable_participant_count": stable_count,
            "cv_pass_rate": cv_pass_count / repetitions,
            "interval_compatibility_rate": interval_compatible_count / repetitions,
            "numerical_incomplete_count": numerical_incomplete,
            "median_pooled_c80_relative_error": median_error,
            "pooled_95ci_coverage": coverage,
            "coverage_denominator": stable_count,
        },
        "gate_thresholds": {
            "minimum_block_fit_convergence_rate": BLOCK_FIT_CONVERGENCE_GATE,
            "minimum_bootstrap_convergence_rate": BOOTSTRAP_CONVERGENCE_GATE,
            "minimum_stable_participant_rate": STABLE_RATE_GATE,
            "maximum_median_pooled_c80_relative_error": MEDIAN_RELATIVE_ERROR_GATE,
            "minimum_pooled_95ci_coverage": COVERAGE_GATE,
        },
        "gates": gates,
        "replicates": truth_rows,
    }


def run_plan_bound_recovery(
    config_path: str | Path,
    plan_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    config, config_sha = _read_json(config_path)
    plan, _ = _read_json(plan_path)
    try:
        validate_plan(plan)
    except ValidationDesignError as error:
        raise ValidationRecoveryError(str(error)) from error
    if plan.get("config_sha256") != config_sha:
        raise ValidationRecoveryError("validation plan과 config raw SHA-256이 다릅니다.")
    levels = [row["actual_delta_e00_simulated"] for row in plan["levels"]]
    lapse = config.get("observer_profile", {}).get("fixed_lapse")
    selected_width = int(plan["selected_width"]["stroke_width_px"])
    selected = [
        row for row in plan["candidate_predictions"]
        if int(row["stroke_width_px"]) == selected_width
    ]
    if len(selected) != 1:
        raise ValidationRecoveryError("plan의 선택 굵기 후보 예측이 정확히 하나가 아닙니다.")
    candidate = selected[0]
    true_c80_bounds = (
        min(float(candidate["b96"]["ci_low"]), float(candidate["c96"]["ci_low"])),
        max(float(candidate["b96"]["ci_high"]), float(candidate["c96"]["ci_high"])),
    )
    true_w_bounds = (
        0.8 * min(float(candidate["b96"]["w"]), float(candidate["c96"]["w"])),
        1.2 * max(float(candidate["b96"]["w"]), float(candidate["c96"]["w"])),
    )
    return run_validation_recovery(
        levels,
        fixed_lapse=float(lapse),
        plan_sha256=plan["plan_sha256"],
        config_sha256=config_sha,
        true_c80_bounds=true_c80_bounds,
        true_w_bounds=true_w_bounds,
        **kwargs,
    )


def _write_json_atomic(payload: Mapping[str, Any], output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise ValidationRecoveryError(f"출력 파일이 이미 있습니다: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="동결 validation plan의 200회 합성 recovery를 수행합니다.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--bootstrap-replicates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = run_plan_bound_recovery(
            args.config,
            args.validation_plan,
            repetitions=args.repetitions,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        _write_json_atomic(payload, args.output, overwrite=args.overwrite)
    except (OSError, TypeError, ValueError) as error:
        print(f"validation recovery 차단: {error}", file=sys.stderr)
        return 2
    return 0 if payload["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ValidationRecoveryError",
    "run_plan_bound_recovery",
    "run_validation_recovery",
]
