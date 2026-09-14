"""검증된 C80 산출물을 동일 자극 경로의 최소 색이동 보정에 연결한다.

사람 자료 모드는 동결 설정, held-out 검증 계획, 최종 참가자 C80 산출물의
식별자와 SHA-256을 서로 대조한 뒤에만 실행한다. 이 계산으로 보장되는 것은
명시한 유한 8-bit 동일경로 후보 집합에서 C80 제약을 만족하는 최소 OKLab
이동이다. 보정색을 사람이 더 잘 구별한다는 행동 결론은 별도 실험 없이는
허용하지 않는다.

config의 목표 허용오차는 동결 자극 생성에만 사용한다. 보정에서는 표시 RGB의
Machado 후 ΔE00가 C80 이상인 후보 전부를 비교하며 목표 초과량 상한을 두지 않는다.

사람 자료에서는 validation plan에 동결된 8개 색차 수준을 모두 같은 규칙으로
분석한다. 연구자가 결과를 본 뒤 한 수준만 고르는 선택은 허용하지 않는다.
합성 연결 시험은 ``development_mode=True``/``--development-mode``로 완전히
분리하며, 사람 산출물을 이 우회 경로로 읽는 것은 거부한다. 합성 모드에서만
기존 단일 수준 선택을 개발 편의를 위해 허용한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

from experiment_runner import CollectionBlockedError, assert_schedule_config
from implementation_provenance import (
    declared_implementation_files,
    implementation_hashes,
)
from path_correction import find_path_constrained_correction
from threshold_adapter import (
    ThresholdAdapterError,
    ThresholdEstimate,
    load_threshold_payload,
)
from validation_design import ValidationDesignError, validate_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUMAN_CONFIG_SCHEMA = (
    PROJECT_ROOT / "schemas" / "human" / "experiment_config.schema.json"
)
ANALYZE_VALIDATION_SCRIPT = PROJECT_ROOT / "src" / "analyze_validation.py"
MODELS = ("B96", "B288", "C96", "D288")


class PathCorrectionAnalysisError(ValueError):
    """산출물 결합 또는 보정 전제 위반으로 분석을 실행할 수 없을 때 사용한다."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PathCorrectionAnalysisError(f"JSON 객체에 중복 키가 있습니다: {key}")
        result[key] = value
    return result


def _read_json(path: str | Path, *, label: str) -> tuple[dict[str, Any], str]:
    file_path = Path(path)
    try:
        raw = file_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except FileNotFoundError as error:
        raise PathCorrectionAnalysisError(f"{label} 파일이 없습니다: {file_path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PathCorrectionAnalysisError(
            f"{label}을 UTF-8 JSON으로 읽을 수 없습니다: {file_path}"
        ) from error
    if not isinstance(payload, dict):
        raise PathCorrectionAnalysisError(f"{label} 최상위는 JSON 객체여야 합니다.")
    return payload, hashlib.sha256(raw).hexdigest()


def _validate_human_config(config: Mapping[str, Any]) -> None:
    try:
        schema = json.loads(HUMAN_CONFIG_SCHEMA.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(
                schema, format_checker=FormatChecker()
            ).iter_errors(config),
            key=lambda item: list(item.path),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SchemaError) as error:
        raise PathCorrectionAnalysisError(
            f"human experiment config 스키마를 검사할 수 없습니다: {error}"
        ) from error
    if errors:
        details = "; ".join(
            f"{'.'.join(map(str, error.path)) or '$'}: {error.message}"
            for error in errors[:8]
        )
        raise PathCorrectionAnalysisError(f"human experiment config 1.0 오류: {details}")
    try:
        assert_schedule_config(config)
    except CollectionBlockedError as error:
        raise PathCorrectionAnalysisError(f"동결 human config 검사 실패: {error}") from error


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PathCorrectionAnalysisError(f"{label}은 0보다 큰 유한한 숫자여야 합니다.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise PathCorrectionAnalysisError(f"{label}은 0보다 큰 유한한 숫자여야 합니다.")
    return number


def _same_number(first: Any, second: Any) -> bool:
    try:
        return math.isclose(
            float(first), float(second), rel_tol=1e-10, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        return False


def _participant(payload: Mapping[str, Any], participant_id: str) -> Mapping[str, Any]:
    rows = [
        row
        for row in payload.get("participants", ())
        if isinstance(row, Mapping) and row.get("participant_id") == participant_id
    ]
    if len(rows) != 1:
        raise PathCorrectionAnalysisError(
            f"participant_id={participant_id!r}인 결과를 정확히 하나 찾지 못했습니다."
        )
    return rows[0]


def _validate_development_thresholds(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != "0.1-development"
        or payload.get("artifact_type") != "synthetic_validation_analysis"
        or payload.get("data_origin") != "synthetic"
        or payload.get("research_conclusion_allowed") is not False
    ):
        raise PathCorrectionAnalysisError(
            "개발 모드는 research_conclusion_allowed=false인 "
            "synthetic_validation_analysis만 허용합니다."
        )
    if payload.get("coordinate") != "post_machado_delta_e00":
        raise PathCorrectionAnalysisError("합성 C80 좌표가 post_machado_delta_e00이 아닙니다.")
    if payload.get("formula_version") != "direct-c80-4afc-v1":
        raise PathCorrectionAnalysisError("합성 C80 formula_version이 다릅니다.")


def _estimate_from_participant(
    participant: Mapping[str, Any],
    *,
    model: str,
    width_px: int,
    data_origin: str,
    schema_version: str,
    confirmed: bool,
) -> ThresholdEstimate:
    predictions = [
        row
        for row in participant.get("predictions", ())
        if isinstance(row, Mapping)
        and row.get("model") == model
        and row.get("stroke_width_px") == width_px
    ]
    if len(predictions) != 1:
        raise PathCorrectionAnalysisError(
            f"산출물에 model={model}, width={width_px}px 예측이 정확히 하나 필요합니다."
        )
    row = predictions[0]
    c80 = _finite_positive(row.get("c80_sim"), f"{model} C80")
    low = _finite_positive(row.get("ci_low"), f"{model} CI 하한")
    high = _finite_positive(row.get("ci_high"), f"{model} CI 상한")
    if not low <= c80 <= high:
        raise PathCorrectionAnalysisError(f"{model} C80이 신뢰구간 밖입니다.")
    if row.get("fit_converged") is not True:
        raise PathCorrectionAnalysisError(f"{model} 합성 적합이 수렴하지 않았습니다.")
    return ThresholdEstimate(
        participant_id=str(participant["participant_id"]),
        cvd_type=str(participant["cvd_type"]),
        severity=float(participant["severity"]),
        severity_source=str(participant["severity_source"]),
        model=model,
        stroke_width_px=width_px,
        stroke_width_deg=_finite_positive(row.get("stroke_width_deg"), f"{model} width_deg"),
        c80_sim=c80,
        ci_low=low,
        ci_high=high,
        value_type=str(row.get("value_type", "predicted")),
        confirmed=confirmed,
        data_origin=data_origin,
        schema_version=schema_version,
    )


def _validate_observer_binding(
    config: Mapping[str, Any], participant: Mapping[str, Any]
) -> None:
    observer = config.get("observer_profile")
    if not isinstance(observer, Mapping):
        raise PathCorrectionAnalysisError("config.observer_profile이 없습니다.")
    exact_fields = (
        ("participant_id", "participant_id"),
        ("cvd_type", "cvd_type"),
        ("severity_source", "severity_source"),
        ("severity_session_id", "severity_session_id"),
        ("lapse_source", "lapse_source"),
        ("lapse_session_id", "lapse_session_id"),
    )
    for config_key, participant_key in exact_fields:
        if observer.get(config_key) != participant.get(participant_key):
            raise PathCorrectionAnalysisError(
                f"config와 C80의 observer {config_key}가 다릅니다."
            )
    for config_key, participant_key in (
        ("severity", "severity"),
        ("fixed_lapse", "fixed_lapse"),
    ):
        if not _same_number(observer.get(config_key), participant.get(participant_key)):
            raise PathCorrectionAnalysisError(
                f"config와 C80의 observer {config_key}가 다릅니다."
            )


def _validate_width_and_observation(
    participant: Mapping[str, Any], selected_width: Mapping[str, Any]
) -> tuple[float, float, float]:
    validation = participant.get("validation")
    if not isinstance(validation, Mapping):
        raise PathCorrectionAnalysisError("참가자 held-out validation 결과가 없습니다.")
    if validation.get("stable") is not True:
        raise PathCorrectionAnalysisError("불안정한 held-out validation으로 보정을 실행할 수 없습니다.")
    if validation.get("intervals_compatible") is not True:
        raise PathCorrectionAnalysisError("held-out validation block 신뢰구간이 호환되지 않습니다.")
    if validation.get("stroke_width_px") != selected_width.get("stroke_width_px"):
        raise PathCorrectionAnalysisError("C80 결과와 plan의 선택 굵기(px)가 다릅니다.")
    if not _same_number(
        validation.get("stroke_width_deg"), selected_width.get("stroke_width_deg")
    ):
        raise PathCorrectionAnalysisError("C80 결과와 plan의 선택 굵기(deg)가 다릅니다.")
    pooled = _finite_positive(validation.get("pooled_c80"), "관측 pooled C80")
    low = _finite_positive(validation.get("pooled_ci_low"), "관측 pooled C80 CI 하한")
    high = _finite_positive(validation.get("pooled_ci_high"), "관측 pooled C80 CI 상한")
    if not low <= pooled <= high:
        raise PathCorrectionAnalysisError("관측 pooled C80이 신뢰구간 밖입니다.")
    return pooled, low, high


def _classification(value: float, low: float, high: float) -> str:
    if value < low:
        return "below_observed_pooled_c80_95ci"
    if value > high:
        return "above_observed_pooled_c80_95ci"
    return "within_observed_pooled_c80_95ci"


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip().lower()
    if len(value) == 40 and all(channel in "0123456789abcdef" for channel in value):
        return value
    return None


def _require_clean_tracked_implementation(scope: str) -> str:
    """Return HEAD only when every numerical dependency is tracked and unmodified."""
    paths = list(declared_implementation_files(scope))
    commit = _git_commit()
    if commit is None:
        raise PathCorrectionAnalysisError(
            "보정 구현을 재현 가능한 Git commit에 결박할 수 없습니다."
        )
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", *paths],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", *paths],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise PathCorrectionAnalysisError(
            f"보정 구현의 Git 상태를 검사할 수 없습니다: {error}"
        ) from error
    if tracked.returncode != 0:
        raise PathCorrectionAnalysisError(
            "보정 구현 파일 중 Git에 추적되지 않은 파일이 있습니다. 먼저 commit해야 합니다."
        )
    if dirty.returncode != 0:
        raise PathCorrectionAnalysisError(
            "보정 구현 파일에 commit되지 않은 변경이 있습니다. 먼저 검토·commit해야 합니다."
        )
    return commit


def run_path_correction_analysis(
    config_path: str | Path,
    validation_plan_path: str | Path,
    participant_thresholds_path: str | Path,
    *,
    participant_id: str,
    source_level_index: int | None = None,
    models: Sequence[str] = MODELS,
    target_policy: str = "point",
    development_mode: bool = False,
) -> dict[str, Any]:
    """결합 검사를 통과한 C80별 동일경로 최소 보정을 계산한다."""
    if not isinstance(development_mode, bool):
        raise PathCorrectionAnalysisError("development_mode는 bool이어야 합니다.")
    if not isinstance(participant_id, str) or not participant_id.strip():
        raise PathCorrectionAnalysisError("participant_id가 필요합니다.")
    if source_level_index is not None and (
        isinstance(source_level_index, bool)
        or not isinstance(source_level_index, int)
        or not 1 <= source_level_index <= 8
    ):
        raise PathCorrectionAnalysisError("source_level_index는 1~8 정수여야 합니다.")
    if development_mode and source_level_index is None:
        raise PathCorrectionAnalysisError(
            "합성 개발 모드에는 source_level_index 1~8이 필요합니다."
        )
    if not development_mode and source_level_index is not None:
        raise PathCorrectionAnalysisError(
            "사람 모드에서는 source_level_index를 선택할 수 없습니다. "
            "동결된 8개 validation 수준을 모두 분석해야 합니다."
        )
    if target_policy not in {"point", "ci_high"}:
        raise PathCorrectionAnalysisError("target_policy는 point 또는 ci_high여야 합니다.")
    selected_models = tuple(models)
    if (
        not selected_models
        or len(set(selected_models)) != len(selected_models)
        or any(model not in MODELS for model in selected_models)
    ):
        raise PathCorrectionAnalysisError(
            "models는 B96/B288/C96/D288 중 중복 없는 한 개 이상이어야 합니다."
        )
    if not development_mode and selected_models != MODELS:
        raise PathCorrectionAnalysisError(
            "사람 분석은 선택 보고를 막기 위해 models를 B96/B288/C96/D288 "
            "전체 고정 순서로 사용해야 합니다."
        )

    config, config_raw_sha = _read_json(config_path, label="experiment config")
    plan, plan_raw_sha = _read_json(validation_plan_path, label="validation plan")
    thresholds, thresholds_raw_sha = _read_json(
        participant_thresholds_path, label="participant thresholds"
    )
    try:
        validate_plan(plan)
    except ValidationDesignError as error:
        raise PathCorrectionAnalysisError(f"validation plan 검사 실패: {error}") from error

    expected_origin = "synthetic" if development_mode else "human"
    if plan.get("data_origin") != expected_origin:
        raise PathCorrectionAnalysisError(
            f"이 실행 모드에는 data_origin={expected_origin} validation plan이 필요합니다."
        )
    if development_mode:
        _validate_development_thresholds(thresholds)
    else:
        _validate_human_config(config)
        try:
            thresholds = load_threshold_payload(participant_thresholds_path)
        except ThresholdAdapterError as error:
            raise PathCorrectionAnalysisError(str(error)) from error
        _, verified_thresholds_raw_sha = _read_json(
            participant_thresholds_path, label="participant thresholds"
        )
        if verified_thresholds_raw_sha != thresholds_raw_sha:
            raise PathCorrectionAnalysisError(
                "검증 중 participant thresholds 파일 원문이 변경됐습니다."
            )
        if thresholds.get("data_origin") != "human":
            raise PathCorrectionAnalysisError("사람 모드에는 confirmed human C80이 필요합니다.")

    if plan.get("config_sha256") != config_raw_sha:
        raise PathCorrectionAnalysisError("validation plan과 config 원문 SHA-256이 다릅니다.")
    if thresholds.get("config_sha256") != config_raw_sha:
        raise PathCorrectionAnalysisError("C80 산출물과 config 원문 SHA-256이 다릅니다.")
    if thresholds.get("config_version") != config.get("config_version"):
        raise PathCorrectionAnalysisError("C80 산출물과 config_version이 다릅니다.")
    if thresholds.get("source_validation_plan_sha256") != plan.get("plan_sha256"):
        raise PathCorrectionAnalysisError("C80 산출물과 validation plan canonical SHA-256이 다릅니다.")
    if plan.get("participant_id") != participant_id:
        raise PathCorrectionAnalysisError("validation plan의 participant_id가 다릅니다.")

    participant = _participant(thresholds, participant_id)
    if not development_mode and participant.get("confirmed") is not True:
        raise PathCorrectionAnalysisError("confirmed=false인 human C80은 보정에 쓸 수 없습니다.")
    current_commit = _git_commit()
    if not development_mode:
        try:
            current_analysis_sha = hashlib.sha256(
                ANALYZE_VALIDATION_SCRIPT.read_bytes()
            ).hexdigest()
        except OSError as error:
            raise PathCorrectionAnalysisError(
                f"현재 validation 분석 코드를 읽을 수 없습니다: {error}"
            ) from error
        if thresholds.get("analysis_script_sha256") != current_analysis_sha:
            raise PathCorrectionAnalysisError(
                "C80 산출물의 analysis_script_sha256이 현재 analyze_validation.py와 다릅니다."
            )
        if thresholds.get("git_commit") != plan.get("source_git_commit"):
            raise PathCorrectionAnalysisError(
                "C80 산출물의 자료 수집 git_commit이 validation plan과 다릅니다."
            )
    _validate_observer_binding(config, participant)
    selected_width = plan.get("selected_width")
    if not isinstance(selected_width, Mapping):
        raise PathCorrectionAnalysisError("validation plan에 selected_width가 없습니다.")
    observed, observed_low, observed_high = _validate_width_and_observation(
        participant, selected_width
    )

    levels = plan.get("levels")
    if not isinstance(levels, list) or len(levels) != 8:
        raise PathCorrectionAnalysisError("validation plan에 8개 수준이 필요합니다.")
    selected_source_levels = (
        [levels[source_level_index - 1]] if development_mode else list(levels)
    )

    color_design = config.get("color_design")
    observer = config.get("observer_profile")
    if not isinstance(color_design, Mapping) or not isinstance(observer, Mapping):
        raise PathCorrectionAnalysisError("config의 color_design/observer_profile이 없습니다.")
    grid_steps = color_design.get("search_grid_steps")
    model_rules: list[tuple[str, ThresholdEstimate, float]] = []
    for model in selected_models:
        estimate = _estimate_from_participant(
            participant,
            model=model,
            width_px=int(selected_width["stroke_width_px"]),
            data_origin=expected_origin,
            schema_version=str(thresholds["schema_version"]),
            confirmed=bool(participant.get("confirmed", False)),
        )
        if not _same_number(estimate.stroke_width_deg, selected_width["stroke_width_deg"]):
            raise PathCorrectionAnalysisError(f"{model} 예측과 plan의 선택 굵기(deg)가 다릅니다.")
        target = estimate.target(target_policy)
        model_rules.append((model, estimate, target))

    calibration_sensitivity = participant.get(
        "calibration_width_slope_sensitivity"
    )
    if isinstance(calibration_sensitivity, Mapping):
        d288_beta_claim_allowed = bool(
            not development_mode
            and calibration_sensitivity.get("d288_beta_interpretation_eligible")
            is True
        )
        d288_beta_claim_reason = calibration_sensitivity.get(
            "d288_beta_interpretation_reason"
        )
    else:
        d288_beta_claim_allowed = False
        d288_beta_claim_reason = "calibration_width_slope_sensitivity_unavailable"

    level_results: list[dict[str, Any]] = []
    for expected_index, source_level in enumerate(selected_source_levels, start=1):
        level_index = source_level.get("validation_level_index")
        if development_mode:
            expected_level_index = source_level_index
        else:
            expected_level_index = expected_index
        if level_index != expected_level_index:
            raise PathCorrectionAnalysisError("source level index와 plan 수준이 다릅니다.")
        try:
            source_a = tuple(source_level["rgb_a"])
            source_b = tuple(source_level["rgb_b"])
        except (KeyError, TypeError) as error:
            raise PathCorrectionAnalysisError(
                f"validation level {level_index} source RGB가 없습니다."
            ) from error
        source_reported = _finite_positive(
            source_level.get("actual_delta_e00_simulated"), "source level 실제 ΔE00"
        )
        corrections: list[dict[str, Any]] = []
        for model, estimate, target in model_rules:
            try:
                correction = find_path_constrained_correction(
                    color_design["midpoint_lab"],
                    color_design["direction_lab"],
                    source_a,
                    source_b,
                    target,
                    str(observer["cvd_type"]),
                    float(observer["severity"]),
                    grid_steps=grid_steps,
                )
            except (TypeError, ValueError) as error:
                raise PathCorrectionAnalysisError(
                    f"validation level {level_index}, {model} 동일경로 보정 실패: {error}"
                ) from error
            if not _same_number(
                correction["source_simulated_delta_e00"], source_reported
            ):
                raise PathCorrectionAnalysisError(
                    f"validation level {level_index} source RGB의 재계산 ΔE00가 "
                    "validation plan과 다릅니다."
                )
            achieved = float(correction["achieved_simulated_delta_e00"])
            corrections.append(
                {
                    "model": model,
                    "target_policy": target_policy,
                    "predicted_c80_sim": estimate.c80_sim,
                    "predicted_c80_95ci": {
                        "ci_low": estimate.ci_low,
                        "ci_high": estimate.ci_high,
                    },
                    "target_simulated_delta_e00": target,
                    "achieved_vs_observed_pooled_c80_95ci": _classification(
                        achieved, observed_low, observed_high
                    ),
                    "correction": correction,
                }
            )
        level_results.append(
            {
                "source_level": {
                    "validation_level_index": int(level_index),
                    "rgb_a": list(source_a),
                    "rgb_b": list(source_b),
                    "actual_delta_e00_simulated": float(source_reported),
                },
                "corrections": corrections,
            }
        )

    correction_implementation_hashes = implementation_hashes(
        "path_correction_analysis"
    )
    correction_git_commit = (
        None
        if development_mode
        else _require_clean_tracked_implementation("path_correction_analysis")
    )

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "path_constrained_correction_analysis",
        "data_origin": expected_origin,
        "development_mode": development_mode,
        "research_conclusion_allowed": False,
        "coordinate": "post_machado_delta_e00",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "participant_id": participant_id,
        "observer": {
            "cvd_type": participant["cvd_type"],
            "severity": float(participant["severity"]),
            "severity_source": participant["severity_source"],
        },
        "selected_width": {
            "stroke_width_px": int(selected_width["stroke_width_px"]),
            "stroke_width_deg": float(selected_width["stroke_width_deg"]),
        },
        "analysis_scope": {
            "source_levels": (
                "single_explicit_level_development_only"
                if development_mode
                else "all_eight_frozen_validation_levels"
            ),
            "level_count": len(level_results),
            "selection_after_validation_allowed": False,
            "partial_result_allowed": False,
        },
        "level_results": level_results,
        "observed_validation": {
            "stable": True,
            "pooled_c80": observed,
            "pooled_c80_95ci": {"ci_low": observed_low, "ci_high": observed_high},
        },
        "models": list(selected_models),
        "target_policy": target_policy,
        "provenance": {
            "config_raw_sha256": config_raw_sha,
            "validation_plan_raw_sha256": plan_raw_sha,
            "validation_plan_canonical_sha256": plan["plan_sha256"],
            "participant_thresholds_raw_sha256": thresholds_raw_sha,
            "participant_thresholds_source_trials_sha256": thresholds.get(
                "source_trials_sha256"
            ),
            "participant_thresholds_source_stimulus_manifest_sha256": thresholds.get(
                "source_stimulus_manifest_sha256"
            ),
            "participant_thresholds_analysis_script_sha256": thresholds.get(
                "analysis_script_sha256"
            ),
            "source_git_commit": thresholds.get("git_commit"),
            "current_git_commit": current_commit,
            "correction_analysis_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "correction_implementation_git_commit": correction_git_commit,
            "correction_implementation_hashes": correction_implementation_hashes,
        },
        "claim_flags": {
            "computational_constraint_claim_allowed": not development_mode,
            "human_readability_improvement_claim_allowed": False,
            "all_frozen_validation_levels_reported": not development_mode,
            "d288_correction_computation_allowed": True,
            "d288_beta_or_width_effect_claim_allowed": d288_beta_claim_allowed,
        },
        "interpretation": {
            "minimum_scope": "exact_over_finite_symmetric_path_grid_not_continuous_global",
            "behavioral_scope": "same_lab_midpoint_direction_as_c80_measurement",
            "observed_ci_classification_is_not_a_new_behavioral_test": True,
            "calibration_width_slope_sensitivity": (
                dict(calibration_sensitivity)
                if isinstance(calibration_sensitivity, Mapping)
                else None
            ),
            "d288_beta_or_width_effect_claim_reason": d288_beta_claim_reason,
            "level_scope": (
                "all_eight_frozen_validation_levels"
                if not development_mode
                else "single_level_for_software_development_only"
            ),
            "required_for_readability_claim": (
                "independent human comparison of corrected and uncorrected displayed pairs"
            ),
        },
    }
    # 합성 software 연결 시험의 기존 단일 수준 필드는 하위 호환을 위해서만
    # 유지한다. 사람 자료에서는 이 필드를 만들지 않아 단일 수준 선택을 막는다.
    if development_mode:
        result["source_level"] = level_results[0]["source_level"]
        result["corrections"] = level_results[0]["corrections"]
    return result


def _write_json_atomic(payload: Mapping[str, Any], output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise PathCorrectionAnalysisError(f"출력 파일이 이미 있습니다: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
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
    parser = argparse.ArgumentParser(
        description="검증된 C80으로 동일 Lab 경로의 유한 후보 최소 색이동을 계산합니다."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-plan", type=Path, required=True)
    parser.add_argument("--participant-thresholds", type=Path, required=True)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument(
        "--source-level-index",
        type=int,
        choices=range(1, 9),
        help="합성 --development-mode 단일 수준 시험에서만 사용(사람 모드에서는 금지)",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--target-policy", choices=("point", "ci_high"), default="point")
    parser.add_argument("--development-mode", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = run_path_correction_analysis(
            args.config,
            args.validation_plan,
            args.participant_thresholds,
            participant_id=args.participant_id,
            source_level_index=args.source_level_index,
            models=args.models,
            target_policy=args.target_policy,
            development_mode=args.development_mode,
        )
        _write_json_atomic(payload, args.output, overwrite=args.overwrite)
    except (OSError, TypeError, ValueError) as error:
        print(f"동일경로 보정 분석 차단: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MODELS",
    "PathCorrectionAnalysisError",
    "run_path_correction_analysis",
]
