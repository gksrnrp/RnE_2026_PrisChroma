"""교정 자료만으로 held-out 4AFC 검증 계획을 동결한다.

이 모듈은 검증 반응을 읽지 않는다. 동결된 실험 설정과
``calibration_model_fits``의 B192/C192 예측만 사용해 검증 굵기 하나, 로그 간격
8수준, 완전 균형 64시행 블록 세 개를 결정한다. 출력 ``plan_sha256``은 이후
검증 시행이 정확히 이 계획에 속하는지 묶는 불변 식별자다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from analyze_calibration import validate_calibration_artifact
from implementation_provenance import (
    ImplementationProvenanceError,
    implementation_hashes,
    verify_implementation_hashes,
)
from psychometrics import direct_c80_probability
from stimulus_design import generate_symmetric_pair


PLAN_VERSION = "validation-plan-v1"
CANDIDATE_POLICY = (
    "maximum-conservative-95ci-log-gap-then-absolute-log-c80-ratio-"
    "then-lower-width-v1"
)
LEVEL_POLICY = "b192-c192-ci-envelope-log-x-plusminus-log9w-v1"
LEVEL_COUNT = 8
BLOCK_COUNT = 3
TRIALS_PER_BLOCK = 64
VALIDATION_FIRST_TRIAL_ID = 385
VALIDATION_LAST_TRIAL_ID = VALIDATION_FIRST_TRIAL_ID + BLOCK_COUNT * TRIALS_PER_BLOCK - 1
LOGIT_SPAN = math.log(0.9 / 0.1)


class ValidationDesignError(ValueError):
    """입력·양자화·동결 규칙 위반으로 검증 계획을 만들 수 없을 때 사용한다."""


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationDesignError(f"JSON 객체에 중복 key가 있습니다: {key}")
        result[key] = value
    return result


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _raw_sha256(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except FileNotFoundError as error:
        raise ValidationDesignError(f"파일이 없습니다: {path}") from error


def _read_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    try:
        payload = json.loads(
            file_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except FileNotFoundError as error:
        raise ValidationDesignError(f"파일이 없습니다: {file_path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationDesignError(f"UTF-8 JSON을 읽을 수 없습니다: {file_path}") from error
    if not isinstance(payload, dict):
        raise ValidationDesignError(f"JSON 최상위는 객체여야 합니다: {file_path}")
    return payload


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValidationDesignError(f"{label}은 0보다 큰 유한한 숫자여야 합니다.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationDesignError(f"{label}은 0보다 큰 유한한 숫자여야 합니다.") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValidationDesignError(f"{label}은 0보다 큰 유한한 숫자여야 합니다.")
    return number


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValidationDesignError(f"{label}은 0 이상의 유한한 숫자여야 합니다.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationDesignError(f"{label}은 0 이상의 유한한 숫자여야 합니다.") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValidationDesignError(f"{label}은 0 이상의 유한한 숫자여야 합니다.")
    return number


def _conservative_95ci_log_gap(
    b_low: float, b_high: float, c_low: float, c_high: float
) -> float:
    return max(
        0.0,
        max(math.log(b_low), math.log(c_low))
        - min(math.log(b_high), math.log(c_high)),
    )


def _model_row(participant: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    matches = [row for row in participant.get("models", ()) if row.get("model") == name]
    if len(matches) != 1:
        raise ValidationDesignError(f"calibration artifact에 {name} 모델이 정확히 하나 필요합니다.")
    row = matches[0]
    if row.get("fit", {}).get("converged") is not True:
        raise ValidationDesignError(f"{name} 적합이 수렴하지 않았습니다.")
    if row.get("bootstrap", {}).get("converged") is not True:
        raise ValidationDesignError(f"{name} bootstrap이 수렴하지 않았습니다.")
    return row


def _prediction(row: Mapping[str, Any], width_px: int) -> Mapping[str, Any]:
    matches = [
        item
        for item in row.get("held_out_candidate_predictions", ())
        if item.get("stroke_width_px") == width_px
    ]
    if len(matches) != 1:
        raise ValidationDesignError(
            f"{row.get('model')}의 {width_px}px calibration-only 예측이 정확히 하나 필요합니다."
        )
    result = matches[0]
    if result.get("observed_held_out_used") is not False:
        raise ValidationDesignError("검증 계획 선택에 held-out 관측값이 들어갔습니다.")
    for field in ("c80_sim", "ci_low", "ci_high"):
        _finite_positive(result.get(field), f"{row.get('model')} {width_px}px {field}")
    return result


def _candidate_rows(
    config: Mapping[str, Any], participant: Mapping[str, Any]
) -> list[dict[str, Any]]:
    candidates = config.get("validation", {}).get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValidationDesignError("config.validation.candidates에는 두 후보가 필요합니다.")
    b192 = _model_row(participant, "B192")
    c192 = _model_row(participant, "C192")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        width_px = int(candidate["rendered_width_px"])
        b_prediction = _prediction(b192, width_px)
        c_prediction = _prediction(c192, width_px)
        b_value = float(b_prediction["c80_sim"])
        c_value = float(c_prediction["c80_sim"])
        # 두 95% CI 사이의 보수적 로그 간격. 겹치면 0이다.
        b_low, b_high = float(b_prediction["ci_low"]), float(b_prediction["ci_high"])
        c_low, c_high = float(c_prediction["ci_low"]), float(c_prediction["ci_high"])
        conservative_gap = _conservative_95ci_log_gap(
            b_low, b_high, c_low, c_high
        )
        rows.append(
            {
                "stroke_width_px": width_px,
                "stroke_width_deg": float(candidate["width_deg"]),
                "b192": {
                    "c80_sim": b_value,
                    "ci_low": b_low,
                    "ci_high": b_high,
                    "w": _finite_positive(b192["fit"]["parameters"].get("w"), "B192 w"),
                },
                "c192": {
                    "c80_sim": c_value,
                    "ci_low": c_low,
                    "ci_high": c_high,
                    "w": _finite_positive(c192["fit"]["parameters"].get("w"), "C192 w"),
                },
                "absolute_log_c80_ratio": abs(math.log(b_value / c_value)),
                "conservative_95ci_log_gap": conservative_gap,
                "prediction_95ci_nonoverlap": conservative_gap > 0.0,
                "predicted_separability": conservative_gap > 0.0,
            }
        )
        for model_name in ("b192", "c192"):
            model = rows[-1][model_name]
            c80 = float(model["c80_sim"])
            w = float(model["w"])
            lapse = float((b192 if model_name == "b192" else c192)["fit"]["fixed_lapse"])
            lower = c80 * math.exp(-LOGIT_SPAN * w)
            upper = c80 * math.exp(LOGIT_SPAN * w)
            model["point_centered_log_x_envelope"] = {
                "lower_delta_e00": lower,
                "upper_delta_e00": upper,
                "lower_expected_actual_accuracy": float(
                    direct_c80_probability(lower, c80, w, lapse)
                ),
                "upper_expected_actual_accuracy": float(
                    direct_c80_probability(upper, c80, w, lapse)
                ),
            }
    return rows


def _select_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    # calibration-only 95% CI가 보수적으로 가장 멀리 떨어진 후보를 먼저 택한다.
    # 그 간격이 같으면 점예측 로그비, 그래도 같으면 낮은 굵기로 고정한다.
    # 두 후보의 CI 간격이 모두 0이어도 점예측으로 하나를 고르되, plan에는
    # predicted_separability=false를 남겨 사전 구분 가능성이 미확인임을 보존한다.
    return min(
        rows,
        key=lambda row: (
            -float(row["conservative_95ci_log_gap"]),
            -float(row["absolute_log_c80_ratio"]),
            int(row["stroke_width_px"]),
        ),
    )


def _target_levels(selected: Mapping[str, Any]) -> list[float]:
    endpoints: list[float] = []
    for model in ("b192", "c192"):
        low_c80 = float(selected[model]["ci_low"])
        high_c80 = float(selected[model]["ci_high"])
        w = float(selected[model]["w"])
        endpoints.extend(
            (
                low_c80 * math.exp(-LOGIT_SPAN * w),
                high_c80 * math.exp(LOGIT_SPAN * w),
            )
        )
    low, high = min(endpoints), max(endpoints)
    if not 0.0 < low < high:
        raise ValidationDesignError("B192/C192 예측으로 유효한 검증 수준 범위를 만들지 못했습니다.")
    levels = np.geomspace(low, high, LEVEL_COUNT)
    return [float(value) for value in levels]


def _quantize_levels(
    config: Mapping[str, Any],
    targets: Sequence[float],
    *,
    pair_generator: Callable[..., Mapping[str, Any]],
) -> list[dict[str, Any]]:
    design = config["color_design"]
    profile = config["observer_profile"]
    absolute_tolerance = _finite_positive(
        design["target_absolute_tolerance_delta_e00"], "absolute target tolerance"
    )
    relative_tolerance = float(design["target_relative_tolerance"])
    grid_steps = int(design["search_grid_steps"])
    quantized: list[dict[str, Any]] = []
    seen_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    previous_actual = -math.inf
    for index, target in enumerate(targets, start=1):
        allowed = max(absolute_tolerance, relative_tolerance * target)
        try:
            pair = pair_generator(
                design["midpoint_lab"],
                design["direction_lab"],
                target,
                profile["cvd_type"],
                float(profile["severity"]),
                max_target_error=allowed,
                grid_steps=grid_steps,
            )
        except (TypeError, ValueError) as error:
            raise ValidationDesignError(
                f"검증 수준 {index}({target:.6g})을 8-bit sRGB로 만들 수 없습니다: {error}"
            ) from error
        rgb_a = tuple(int(value) for value in pair["rgb_a"])
        rgb_b = tuple(int(value) for value in pair["rgb_b"])
        # odd identity를 뒤집었을 때 A/B 순서만 바뀐 같은 색쌍도 중복이다.
        key = tuple(sorted((rgb_a, rgb_b)))
        actual = _finite_positive(pair["simulated_delta_e00"], f"수준 {index} 실제 ΔE00")
        error = abs(actual - target)
        if error > allowed + 1e-10:
            raise ValidationDesignError(f"검증 수준 {index}의 실제 양자화 오차가 허용값을 넘습니다.")
        if key in seen_pairs:
            raise ValidationDesignError(f"검증 수준 {index}가 앞 수준과 같은 8-bit RGB 쌍입니다.")
        if actual <= previous_actual:
            raise ValidationDesignError("실제 양자화된 검증 ΔE00가 엄격히 증가하지 않습니다.")
        seen_pairs.add(key)
        previous_actual = actual
        quantized.append(
            {
                "validation_level_index": index,
                "target_delta_e00_simulated": target,
                "actual_delta_e00_simulated": actual,
                "target_error_delta_e00": error,
                "allowed_target_error_delta_e00": allowed,
                "rgb_a": list(rgb_a),
                "rgb_b": list(rgb_b),
                "displayed_lab_a": [float(value) for value in pair["displayed_lab_a"]],
                "displayed_lab_b": [float(value) for value in pair["displayed_lab_b"]],
                "simulated_rgb_a": [float(value) for value in pair["simulated_rgb_a"]],
                "simulated_rgb_b": [float(value) for value in pair["simulated_rgb_b"]],
                "simulated_lab_a": [float(value) for value in pair["simulated_lab_a"]],
                "simulated_lab_b": [float(value) for value in pair["simulated_lab_b"]],
                "original_delta_e00": float(pair["original_delta_e00"]),
            }
        )
    return quantized


def _schedule(
    *,
    participant_id: str,
    config_sha256: str,
    width_px: int,
    width_deg: float,
    levels: Sequence[Mapping[str, Any]],
    seed: int,
    first_trial_id: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trial_id = first_trial_id
    for block in range(1, BLOCK_COUNT + 1):
        block_seed = seed + block - 1
        combinations = [
            (int(level["validation_level_index"]), position, odd)
            for level in levels
            for position in range(1, 5)
            for odd in ("a", "b")
        ]
        generator = np.random.default_rng(block_seed)
        generator.shuffle(combinations)
        for within_block_index, (level_index, position, odd) in enumerate(
            combinations, start=1
        ):
            level = levels[level_index - 1]
            base = "b" if odd == "a" else "a"
            position_colors = [base] * 4
            position_colors[position - 1] = odd
            rows.append(
                {
                    "participant_id": participant_id,
                    "config_sha256": config_sha256,
                    "validation_plan_sha256": None,
                    "trial_id": trial_id,
                    "global_trial_key": f"{participant_id}:{config_sha256}:{trial_id}",
                    "phase": "validation",
                    "validation_block_id": f"V{block:02d}",
                    "within_block_index": within_block_index,
                    "validation_level_index": level_index,
                    "stroke_width_px": width_px,
                    "stroke_width_deg": width_deg,
                    "target_delta_e00_simulated": level["target_delta_e00_simulated"],
                    "actual_delta_e00_simulated": level["actual_delta_e00_simulated"],
                    "correct_position": position,
                    "odd_color": odd,
                    "position_colors": position_colors,
                    "random_seed": block_seed,
                }
            )
            trial_id += 1
    return rows


def _validated_candidate_predictions(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = plan.get("candidate_predictions")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValidationDesignError(
            "candidate_predictions에는 사전 동결된 두 후보가 정확히 필요합니다."
        )
    validated: list[Mapping[str, Any]] = []
    widths: set[int] = set()
    for index, row in enumerate(rows, start=1):
        label = f"candidate_predictions[{index}]"
        if not isinstance(row, Mapping):
            raise ValidationDesignError(f"{label}은 객체여야 합니다.")
        width_px = row.get("stroke_width_px")
        if isinstance(width_px, bool) or not isinstance(width_px, int) or width_px <= 0:
            raise ValidationDesignError(f"{label}.stroke_width_px는 양의 정수여야 합니다.")
        if width_px in widths:
            raise ValidationDesignError("candidate_predictions의 두 px 굵기는 달라야 합니다.")
        widths.add(width_px)
        width_deg = row.get("stroke_width_deg")
        if isinstance(width_deg, bool) or not isinstance(width_deg, (int, float)):
            raise ValidationDesignError(f"{label}.stroke_width_deg는 양의 유한 수여야 합니다.")
        _finite_positive(width_deg, f"{label}.stroke_width_deg")

        model_values: dict[str, tuple[float, float, float, float]] = {}
        for model_name in ("b192", "c192"):
            model = row.get(model_name)
            if not isinstance(model, Mapping):
                raise ValidationDesignError(f"{label}.{model_name}은 객체여야 합니다.")
            values: list[float] = []
            for field in ("c80_sim", "ci_low", "ci_high", "w"):
                value = model.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValidationDesignError(
                        f"{label}.{model_name}.{field}는 양의 유한 수여야 합니다."
                    )
                values.append(
                    _finite_positive(value, f"{label}.{model_name}.{field}")
                )
            c80, ci_low, ci_high, w = values
            if ci_low > ci_high:
                raise ValidationDesignError(
                    f"{label}.{model_name}의 95% CI 순서가 뒤집혔습니다."
                )
            model_values[model_name] = (c80, ci_low, ci_high, w)

        b_c80, b_low, b_high, _ = model_values["b192"]
        c_c80, c_low, c_high, _ = model_values["c192"]
        expected_ratio = abs(math.log(b_c80 / c_c80))
        expected_gap = _conservative_95ci_log_gap(b_low, b_high, c_low, c_high)
        reported_ratio_value = row.get("absolute_log_c80_ratio")
        reported_gap_value = row.get("conservative_95ci_log_gap")
        if (
            isinstance(reported_ratio_value, bool)
            or not isinstance(reported_ratio_value, (int, float))
            or isinstance(reported_gap_value, bool)
            or not isinstance(reported_gap_value, (int, float))
        ):
            raise ValidationDesignError(f"{label}의 선택 점수는 유한 수여야 합니다.")
        reported_ratio = _finite_nonnegative(
            reported_ratio_value, f"{label}.absolute_log_c80_ratio"
        )
        reported_gap = _finite_nonnegative(
            reported_gap_value, f"{label}.conservative_95ci_log_gap"
        )
        if reported_ratio != expected_ratio or reported_gap != expected_gap:
            raise ValidationDesignError(
                f"{label}의 선택 점수가 B192/C192 예측에서 재계산한 값과 다릅니다."
            )
        expected_separability = expected_gap > 0.0
        if (
            row.get("prediction_95ci_nonoverlap") is not expected_separability
            or row.get("predicted_separability") is not expected_separability
        ):
            raise ValidationDesignError(
                f"{label}의 구분 가능성 표지가 재계산한 95% CI 간격과 다릅니다."
            )
        validated.append(row)
    return validated


def _hash_preimage(plan: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(plan, ensure_ascii=False, allow_nan=False))
    result.pop("plan_sha256", None)
    for row in result.get("schedule", ()):
        row.pop("validation_plan_sha256", None)
    return result


def compute_plan_sha256(plan: Mapping[str, Any]) -> str:
    """명시된 canonical preimage 규칙으로 plan ID를 다시 계산한다."""
    return hashlib.sha256(_canonical_bytes(_hash_preimage(plan))).hexdigest()


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("artifact_type") != "validation_plan" or plan.get("schema_version") != "1.0":
        raise ValidationDesignError("validation plan 종류 또는 버전이 잘못됐습니다.")
    if plan.get("status") != "frozen_before_held_out" or plan.get("observed_held_out_used") is not False:
        raise ValidationDesignError("검증 계획은 held-out 반응 전에 동결되어야 합니다.")
    source_git_commit = plan.get("source_git_commit")
    if (
        not isinstance(source_git_commit, str)
        or len(source_git_commit) != 40
        or any(channel not in "0123456789abcdef" for channel in source_git_commit)
    ):
        raise ValidationDesignError("validation plan에 40자리 source_git_commit이 필요합니다.")
    try:
        verify_implementation_hashes(
            plan.get("implementation_hashes"),
            "validation_plan",
            label="validation plan",
        )
    except ImplementationProvenanceError as error:
        raise ValidationDesignError(str(error)) from error
    expected = compute_plan_sha256(plan)
    if plan.get("plan_sha256") != expected:
        raise ValidationDesignError("plan_sha256이 canonical preimage와 다릅니다.")

    selection = plan.get("selection_specification")
    if not isinstance(selection, Mapping):
        raise ValidationDesignError("selection_specification은 객체여야 합니다.")
    exact_selection_fields = {
        "plan_version": PLAN_VERSION,
        "candidate_policy": CANDIDATE_POLICY,
        "level_policy": LEVEL_POLICY,
        "candidate_count": 2,
        "level_count": LEVEL_COUNT,
        "independent_blocks": BLOCK_COUNT,
        "trials_per_block": TRIALS_PER_BLOCK,
        "balance": "8-levels-x-4-positions-x-2-odd-identities",
        "first_trial_id": VALIDATION_FIRST_TRIAL_ID,
    }
    for field, expected_value in exact_selection_fields.items():
        if selection.get(field) != expected_value:
            raise ValidationDesignError(
                f"selection_specification.{field}가 동결 정책과 다릅니다."
            )
    span = selection.get("level_log_x_span_w_multiplier")
    if isinstance(span, bool) or not isinstance(span, (int, float)) or float(span) != LOGIT_SPAN:
        raise ValidationDesignError(
            "selection_specification.level_log_x_span_w_multiplier가 동결 정책과 다릅니다."
        )
    base_seed = selection.get("base_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValidationDesignError("selection_specification.base_seed가 잘못됐습니다.")

    candidates = _validated_candidate_predictions(plan)
    expected_selected = _select_candidate(candidates)
    width = plan.get("selected_width")
    if not isinstance(width, Mapping):
        raise ValidationDesignError("selected_width는 객체여야 합니다.")
    expected_gap = float(expected_selected["conservative_95ci_log_gap"])
    expected_ratio = float(expected_selected["absolute_log_c80_ratio"])
    expected_separability = expected_gap > 0.0
    expected_discriminability = (
        "supported-by-nonoverlapping-calibration-95ci"
        if expected_separability
        else "not-established-before-held-out-collection"
    )
    selected_numeric_fields = {
        "stroke_width_px": int(expected_selected["stroke_width_px"]),
        "stroke_width_deg": float(expected_selected["stroke_width_deg"]),
        "selection_score_conservative_95ci_log_gap": expected_gap,
        "selection_score_absolute_log_c80_ratio": expected_ratio,
        "conservative_95ci_log_gap": expected_gap,
    }
    for field, expected_value in selected_numeric_fields.items():
        value = width.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (field == "stroke_width_px" and not isinstance(value, int))
        ):
            raise ValidationDesignError(f"selected_width.{field}가 유효한 수가 아닙니다.")
        if value != expected_value:
            raise ValidationDesignError(
                f"selected_width.{field}가 사전등록 후보 선택 재계산값과 다릅니다."
            )
    if (
        width.get("prediction_95ci_nonoverlap") is not expected_separability
        or width.get("predicted_separability") is not expected_separability
        or width.get("pre_validation_discriminability") != expected_discriminability
    ):
        raise ValidationDesignError(
            "selected_width의 구분 가능성 표지가 후보 95% CI 재계산값과 다릅니다."
        )
    if width.get("model_superiority_claim_allowed_before_held_out") is not False:
        raise ValidationDesignError("held-out 전 모형 우월성 주장은 허용되지 않습니다.")
    if width.get("interpretation_scope") != "selected-most-discriminating-candidate-only":
        raise ValidationDesignError("selected_width.interpretation_scope가 동결 정책과 다릅니다.")
    expected_targets = _target_levels(expected_selected)

    levels = plan.get("levels", ())
    schedule = plan.get("schedule", ())
    if len(levels) != LEVEL_COUNT or len(schedule) != BLOCK_COUNT * TRIALS_PER_BLOCK:
        raise ValidationDesignError("검증 계획은 8수준과 3×64 시행이어야 합니다.")
    actual_values: list[float] = []
    rgb_pairs: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for index, level in enumerate(levels, start=1):
        if level.get("validation_level_index") != index:
            raise ValidationDesignError("검증 수준 index는 1부터 8까지 순서대로여야 합니다.")
        target = _finite_positive(
            level.get("target_delta_e00_simulated"), f"검증 수준 {index} target"
        )
        if target != expected_targets[index - 1]:
            raise ValidationDesignError(
                f"검증 수준 {index} target이 동결 level policy 재계산값과 다릅니다."
            )
        actual = _finite_positive(
            level.get("actual_delta_e00_simulated"), f"검증 수준 {index} actual"
        )
        reported_error = float(level.get("target_error_delta_e00", math.nan))
        allowed_error = float(level.get("allowed_target_error_delta_e00", math.nan))
        if not math.isclose(reported_error, abs(actual - target), rel_tol=1e-8, abs_tol=1e-10):
            raise ValidationDesignError(f"검증 수준 {index}의 target_error가 재계산값과 다릅니다.")
        if not math.isfinite(allowed_error) or allowed_error < 0.0 or reported_error > allowed_error + 1e-10:
            raise ValidationDesignError(f"검증 수준 {index}이 허용 양자화 오차를 넘습니다.")
        try:
            rgb_a = tuple(int(value) for value in level["rgb_a"])
            rgb_b = tuple(int(value) for value in level["rgb_b"])
            if len(rgb_a) != 3 or len(rgb_b) != 3 or any(
                channel < 0 or channel > 255 for channel in rgb_a + rgb_b
            ):
                raise ValueError
            pair = tuple(sorted((rgb_a, rgb_b)))
        except (KeyError, TypeError, ValueError) as error:
            raise ValidationDesignError(f"검증 수준 {index} RGB 쌍이 잘못됐습니다.") from error
        if pair in rgb_pairs:
            raise ValidationDesignError("검증 수준끼리 같은 8-bit RGB 쌍을 사용합니다.")
        rgb_pairs.add(pair)
        actual_values.append(actual)
    if any(first >= second for first, second in zip(actual_values, actual_values[1:])):
        raise ValidationDesignError("실제 양자화된 검증 ΔE00가 엄격히 증가하지 않습니다.")
    if any(row.get("validation_plan_sha256") != expected for row in schedule):
        raise ValidationDesignError("schedule의 validation_plan_sha256이 plan과 다릅니다.")
    keys = [row.get("global_trial_key") for row in schedule]
    trial_ids = [row.get("trial_id") for row in schedule]
    if len(set(keys)) != len(keys) or len(set(trial_ids)) != len(trial_ids):
        raise ValidationDesignError("schedule의 참가자/config/trial 전역 ID가 중복됩니다.")
    if sorted(trial_ids) != list(
        range(VALIDATION_FIRST_TRIAL_ID, VALIDATION_LAST_TRIAL_ID + 1)
    ):
        raise ValidationDesignError(
            "validation trial_id는 예약 범위 385~576을 정확히 한 번씩 사용해야 합니다."
        )
    participant_id = plan.get("participant_id")
    config_sha = plan.get("config_sha256")
    for row in schedule:
        level_index = int(row["validation_level_index"])
        if not 1 <= level_index <= LEVEL_COUNT:
            raise ValidationDesignError("schedule validation_level_index 범위가 잘못됐습니다.")
        level = levels[level_index - 1]
        trial_id = int(row["trial_id"])
        if (
            row.get("participant_id") != participant_id
            or row.get("config_sha256") != config_sha
            or row.get("global_trial_key") != f"{participant_id}:{config_sha}:{trial_id}"
            or row.get("stroke_width_px") != width.get("stroke_width_px")
            or not math.isclose(
                float(row.get("stroke_width_deg")),
                float(width.get("stroke_width_deg")),
                rel_tol=1e-8,
                abs_tol=1e-10,
            )
            or not math.isclose(
                float(row.get("target_delta_e00_simulated")),
                float(level["target_delta_e00_simulated"]),
                rel_tol=1e-8,
                abs_tol=1e-10,
            )
            or not math.isclose(
                float(row.get("actual_delta_e00_simulated")),
                float(level["actual_delta_e00_simulated"]),
                rel_tol=1e-8,
                abs_tol=1e-10,
            )
        ):
            raise ValidationDesignError(f"schedule trial {trial_id}가 plan source/수준과 다릅니다.")
    for block in range(1, BLOCK_COUNT + 1):
        block_rows = [row for row in schedule if row.get("validation_block_id") == f"V{block:02d}"]
        observed = {
            (row["validation_level_index"], row["correct_position"], row["odd_color"])
            for row in block_rows
        }
        expected_combinations = {
            (level, position, odd)
            for level in range(1, LEVEL_COUNT + 1)
            for position in range(1, 5)
            for odd in ("a", "b")
        }
        if len(block_rows) != TRIALS_PER_BLOCK or observed != expected_combinations:
            raise ValidationDesignError(f"V{block:02d}가 8수준×4위치×2 odd로 균형화되지 않았습니다.")


def build_validation_plan(
    config_path: str | Path,
    calibration_fits_path: str | Path,
    *,
    seed: int = 20260914,
    first_trial_id: int = VALIDATION_FIRST_TRIAL_ID,
    allow_synthetic: bool = False,
    pair_generator: Callable[..., Mapping[str, Any]] = generate_symmetric_pair,
) -> dict[str, Any]:
    """교정 전용 산출물로부터 held-out 반응을 쓰지 않고 계획을 만든다."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValidationDesignError("seed는 0 이상의 정수여야 합니다.")
    if first_trial_id != VALIDATION_FIRST_TRIAL_ID:
        raise ValidationDesignError(
            "validation 전역 trial_id는 예약 범위 385~576을 사용하므로 "
            "first_trial_id는 385여야 합니다."
        )
    config = _read_json(config_path)
    calibration = _read_json(calibration_fits_path)
    try:
        validate_calibration_artifact(calibration)
    except Exception as error:
        raise ValidationDesignError(f"calibration artifact 검증 실패: {error}") from error
    origin = calibration.get("data_origin")
    if origin == "synthetic" and not allow_synthetic:
        raise ValidationDesignError("합성 교정 산출물은 allow_synthetic=True가 필요합니다.")
    if origin not in {"human", "synthetic"}:
        raise ValidationDesignError("calibration data_origin은 human 또는 synthetic이어야 합니다.")
    if calibration.get("analysis_scope") != "calibration_only_intermediate":
        raise ValidationDesignError("calibration-only 중간 산출물만 사용할 수 있습니다.")
    if calibration.get("contains_observed_held_out") is not False:
        raise ValidationDesignError("held-out 관측값이 포함된 산출물로 계획을 고를 수 없습니다.")
    current_calibration_analysis_sha256 = hashlib.sha256(
        Path(__file__).with_name("analyze_calibration.py").read_bytes()
    ).hexdigest()
    if calibration.get("analysis_script_sha256") != current_calibration_analysis_sha256:
        raise ValidationDesignError(
            "calibration artifact가 현재 analyze_calibration.py와 결합되지 않았습니다. "
            "현재 코드로 교정 분석을 다시 실행하세요."
        )

    config_sha = _raw_sha256(config_path)
    calibration_sha = _raw_sha256(calibration_fits_path)
    if calibration.get("source", {}).get("config_sha256") != config_sha:
        raise ValidationDesignError("calibration artifact와 원 experiment config SHA-256이 다릅니다.")
    participants = calibration.get("participants", ())
    if len(participants) != 1:
        raise ValidationDesignError("검증 계획 하나에는 참가자 한 명의 교정 적합만 필요합니다.")
    participant = participants[0]
    profile = config.get("observer_profile", {})
    if participant.get("participant_id") != profile.get("participant_id"):
        raise ValidationDesignError("config와 calibration artifact의 participant_id가 다릅니다.")
    if participant.get("analysis_status") != "complete":
        raise ValidationDesignError("교정 모델 전체가 수렴하기 전에는 검증 계획을 동결할 수 없습니다.")
    validation_config = config.get("validation", {})
    if validation_config.get("independent_blocks") != BLOCK_COUNT:
        raise ValidationDesignError("validation.independent_blocks는 3이어야 합니다.")
    if validation_config.get("delta_e00_level_count") != LEVEL_COUNT:
        raise ValidationDesignError("validation.delta_e00_level_count는 8이어야 합니다.")
    if validation_config.get("trials_per_block") != TRIALS_PER_BLOCK:
        raise ValidationDesignError("validation.trials_per_block는 64여야 합니다.")

    candidates = _candidate_rows(config, participant)
    selected = dict(_select_candidate(candidates))
    targets = _target_levels(selected)
    levels = _quantize_levels(config, targets, pair_generator=pair_generator)
    schedule = _schedule(
        participant_id=participant["participant_id"],
        config_sha256=config_sha,
        width_px=int(selected["stroke_width_px"]),
        width_deg=float(selected["stroke_width_deg"]),
        levels=levels,
        seed=seed,
        first_trial_id=first_trial_id,
    )
    plan: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "validation_plan",
        "status": "frozen_before_held_out",
        "data_origin": origin,
        "observed_held_out_used": False,
        "research_conclusion_allowed": False,
        "implementation_hashes": implementation_hashes("validation_plan"),
        "config_sha256": config_sha,
        "calibration_model_fits_sha256": calibration_sha,
        "source_calibration_trials_sha256": calibration["source"]["trial_file_sha256"],
        "source_git_commit": calibration["source"]["git_commit"],
        "participant_id": participant["participant_id"],
        "selection_specification": {
            "plan_version": PLAN_VERSION,
            "candidate_policy": CANDIDATE_POLICY,
            "level_policy": LEVEL_POLICY,
            "level_log_x_span_w_multiplier": LOGIT_SPAN,
            "candidate_count": 2,
            "level_count": LEVEL_COUNT,
            "independent_blocks": BLOCK_COUNT,
            "trials_per_block": TRIALS_PER_BLOCK,
            "balance": "8-levels-x-4-positions-x-2-odd-identities",
            "base_seed": seed,
            "first_trial_id": first_trial_id,
            "plan_hash_rule": (
                "sha256(canonical-compact-sorted-json-without-top-plan_sha256-"
                "and-without-schedule-validation_plan_sha256)"
            ),
        },
        "candidate_predictions": candidates,
        "selected_width": {
            "stroke_width_px": selected["stroke_width_px"],
            "stroke_width_deg": selected["stroke_width_deg"],
            "selection_score_conservative_95ci_log_gap": selected[
                "conservative_95ci_log_gap"
            ],
            "selection_score_absolute_log_c80_ratio": selected["absolute_log_c80_ratio"],
            "conservative_95ci_log_gap": selected["conservative_95ci_log_gap"],
            "prediction_95ci_nonoverlap": selected["prediction_95ci_nonoverlap"],
            "predicted_separability": selected["predicted_separability"],
            "pre_validation_discriminability": (
                "supported-by-nonoverlapping-calibration-95ci"
                if selected["predicted_separability"]
                else "not-established-before-held-out-collection"
            ),
            "model_superiority_claim_allowed_before_held_out": False,
            "interpretation_scope": "selected-most-discriminating-candidate-only",
        },
        "levels": levels,
        "schedule": schedule,
    }
    plan_sha = compute_plan_sha256(plan)
    plan["plan_sha256"] = plan_sha
    for row in plan["schedule"]:
        row["validation_plan_sha256"] = plan_sha
    validate_plan(plan)
    return plan


def _write_json_atomic(payload: Mapping[str, Any], path: str | Path, *, overwrite: bool) -> None:
    output = Path(path)
    if output.exists() and not overwrite:
        raise ValidationDesignError(f"출력 파일이 이미 있습니다: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="교정 전용 적합으로 검증 계획 3×64회를 동결합니다.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-fits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--first-trial-id", type=int, default=VALIDATION_FIRST_TRIAL_ID)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_validation_plan(
            args.config, args.calibration_fits, seed=args.seed,
            first_trial_id=args.first_trial_id, allow_synthetic=args.allow_synthetic,
        )
        _write_json_atomic(plan, args.output, overwrite=args.overwrite)
    except (OSError, TypeError, ValueError) as error:
        print(f"검증 계획 동결 차단: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BLOCK_COUNT",
    "CANDIDATE_POLICY",
    "LEVEL_COUNT",
    "LEVEL_POLICY",
    "TRIALS_PER_BLOCK",
    "VALIDATION_FIRST_TRIAL_ID",
    "VALIDATION_LAST_TRIAL_ID",
    "ValidationDesignError",
    "build_validation_plan",
    "compute_plan_sha256",
    "validate_plan",
]
