"""세 팀원이 교환하는 연구 데이터의 형식과 의미를 검증한다.

현재 스키마는 ``0.1-draft``이며 가짜 데이터만 허용한다. 팀 합의와
parameter recovery 전에 실제 참가자 자료를 받지 못하게 의도적으로 잠갔다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from experiment_design import (
    SZAFIR_LINE_MAX_VISUAL_ANGLE_DEG,
    SZAFIR_LINE_MIN_VISUAL_ANGLE_DEG,
    pixels_to_visual_angle_deg,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = PROJECT_ROOT / "schemas"
EXAMPLE_DIR = PROJECT_ROOT / "data" / "contracts"

SCHEMA_PATHS = {
    "experiment_config": SCHEMA_DIR / "experiment_config.schema.json",
    "trial": SCHEMA_DIR / "trial.schema.json",
    "application_conditions": SCHEMA_DIR / "application_conditions.schema.json",
    "validation_pairs": SCHEMA_DIR / "validation_pairs.schema.json",
    "participant_thresholds": SCHEMA_DIR / "participant_thresholds.schema.json",
    "correction_results": SCHEMA_DIR / "correction_results.schema.json",
}

EXAMPLE_PATHS = {
    "experiment_config": EXAMPLE_DIR / "experiment_config.example.json",
    "trial": EXAMPLE_DIR / "trial.example.jsonl",
    "application_conditions": EXAMPLE_DIR / "application_conditions.example.json",
    "validation_pairs": EXAMPLE_DIR / "validation_pairs.example.json",
    "participant_thresholds": EXAMPLE_DIR / "participant_thresholds.example.json",
    "correction_results": EXAMPLE_DIR / "correction_results.example.json",
}


class ContractError(ValueError):
    """스키마 또는 연구 의미 규칙을 어긴 데이터에 사용한다."""


def _read_json(path: str | Path) -> Any:
    file_path = Path(path)
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ContractError(f"파일이 없습니다: {file_path}") from error
    except json.JSONDecodeError as error:
        raise ContractError(f"JSON 형식이 깨졌습니다: {file_path}: {error}") from error


def _schema(contract: str) -> dict:
    try:
        path = SCHEMA_PATHS[contract]
    except KeyError as error:
        valid = ", ".join(sorted(SCHEMA_PATHS))
        raise ContractError(f"알 수 없는 계약 {contract!r}. 가능한 값: {valid}") from error
    schema = _read_json(path)
    Draft202012Validator.check_schema(schema)
    return schema


def _json_path(parts: Any) -> str:
    result = "$"
    for part in parts:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def _validate_schema(contract: str, payload: Any) -> None:
    validator = Draft202012Validator(
        _schema(contract), format_checker=FormatChecker()
    )
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    if errors:
        details = "\n".join(
            f"- {_json_path(error.path)}: {error.message}" for error in errors
        )
        raise ContractError(f"{contract} 스키마 오류:\n{details}")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)


def _validate_experiment_config(payload: dict) -> None:
    direction = payload["color_design"]["direction_lab"]
    _assert(
        math.sqrt(sum(float(value) ** 2 for value in direction)) > 0.0,
        "direction_lab은 영벡터일 수 없습니다.",
    )

    display = payload["display"]
    conversion = {
        "display_width_mm": display["width_mm"],
        "horizontal_resolution_px": display["resolution_width_px"],
        "viewing_distance_mm": display["viewing_distance_mm"],
    }
    calibration = payload["calibration"]["widths"]
    calibration_px = [row["width_px"] for row in calibration]
    _assert(
        calibration_px[0] < calibration_px[1] < calibration_px[2],
        "교정 굵기 세 개는 작은 값부터 엄격히 증가해야 합니다.",
    )

    all_width_rows = [(row["width_px"], row["width_deg"]) for row in calibration]
    validation = payload["validation"]["candidates"]
    rendered_px = [row["rendered_width_px"] for row in validation]
    _assert(
        len(set(rendered_px)) == 2 and not set(rendered_px) & set(calibration_px),
        "검증 굵기는 서로 달라야 하고 교정 굵기와 겹치면 안 됩니다.",
    )
    for index, row in enumerate(validation):
        expected_target = math.sqrt(
            calibration_px[index] * calibration_px[index + 1]
        )
        _assert(
            _close(row["geometric_target_px"], expected_target),
            f"검증 후보 {index + 1}의 geometric_target_px가 기하평균과 다릅니다.",
        )
        _assert(
            calibration_px[index]
            < row["rendered_width_px"]
            < calibration_px[index + 1],
            f"검증 굵기 {index + 1}가 대응하는 교정 구간 안에 없습니다.",
        )
        all_width_rows.append((row["rendered_width_px"], row["width_deg"]))

    for width_px, stored_deg in all_width_rows:
        calculated = pixels_to_visual_angle_deg(width_px, **conversion)
        _assert(
            _close(stored_deg, calculated),
            f"{width_px}px의 width_deg가 화면 조건으로 재계산한 값과 다릅니다. "
            f"저장={stored_deg}, 계산={calculated}",
        )

    if payload["model_plan"]["include_szafir_transfer"]:
        for width_px, width_deg in all_width_rows:
            _assert(
                SZAFIR_LINE_MIN_VISUAL_ANGLE_DEG
                <= width_deg
                <= SZAFIR_LINE_MAX_VISUAL_ANGLE_DEG,
                f"{width_px}px({width_deg}도)는 Szafir 측정 범위 밖입니다.",
            )

    staircase = payload["staircase"]
    _assert(
        staircase["minimum_delta_e00"]
        < staircase["starting_delta_e00"]
        < staircase["maximum_delta_e00"],
        "계단 시작 색차는 최소값과 최대값 사이여야 합니다.",
    )
    if staircase["lapse_policy"] == "fixed":
        _assert(
            staircase.get("fixed_lapse") is not None,
            "lapse_policy가 fixed면 fixed_lapse가 필요합니다.",
        )
    else:
        _assert(
            staircase.get("fixed_lapse") is None,
            "lapse_policy가 fixed가 아니면 fixed_lapse는 null이어야 합니다.",
        )


def _validate_trial(payload: dict) -> None:
    if payload["phase"] == "validation":
        _assert(
            payload["staircase_id"] is not None,
            "검증 시행에는 staircase_id가 필요합니다.",
        )
    if payload["trial_status"] == "completed":
        _assert(
            payload["response_position"] is not None
            and payload["is_correct"] is not None,
            "완료 시행에는 response_position과 is_correct가 필요합니다.",
        )
        _assert(
            payload["is_correct"]
            == (payload["response_position"] == payload["correct_position"]),
            "is_correct가 정답 위치와 응답 위치에 맞지 않습니다.",
        )
    else:
        _assert(
            payload["response_position"] is None and payload["is_correct"] is None,
            "중단 시행의 응답과 정오 값은 null이어야 합니다.",
        )


def _validate_application_conditions(payload: dict) -> None:
    seen = set()
    for row in payload["cases"]:
        _assert(row["case_id"] not in seen, f"case_id가 중복됩니다: {row['case_id']}")
        seen.add(row["case_id"])
        if row["measurement_method"] == "detection_failed":
            _assert(
                row["quality_flag"] == "excluded"
                and row["stroke_width_px"] is None
                and row["stroke_width_ratio"] is None,
                f"{row['case_id']}: 검출 실패 자료는 굵기·비율이 null이고 "
                "quality_flag가 excluded여야 합니다.",
            )
        else:
            _assert(
                row["stroke_width_px"] is not None
                and row["stroke_width_ratio"] is not None,
                f"{row['case_id']}: 성공한 측정에는 굵기와 비율이 필요합니다.",
            )
            expected_ratio = row["stroke_width_px"] / row["viewport_width_px"]
            _assert(
                _close(row["stroke_width_ratio"], expected_ratio),
                f"{row['case_id']}: stroke_width_ratio가 굵기/화면 폭과 다릅니다.",
            )


def _validate_validation_pairs(payload: dict) -> None:
    seen = set()
    for row in payload["pairs"]:
        _assert(row["pair_id"] not in seen, f"pair_id가 중복됩니다: {row['pair_id']}")
        seen.add(row["pair_id"])
        _assert(row["rgb_a"] != row["rgb_b"], f"{row['pair_id']}: 두 색이 같습니다.")
        _assert(
            row["rgb_a"] not in row["palette"] and row["rgb_b"] not in row["palette"],
            f"{row['pair_id']}: palette에는 A와 B 자신을 넣지 않습니다.",
        )


def _validate_participant_thresholds(payload: dict) -> None:
    participant_ids = set()
    for participant in payload["participants"]:
        participant_id = participant["participant_id"]
        _assert(
            participant_id not in participant_ids,
            f"participant_id가 중복됩니다: {participant_id}",
        )
        participant_ids.add(participant_id)
        prediction_keys = set()
        for prediction in participant["predictions"]:
            key = (prediction["model"], prediction["stroke_width_px"])
            _assert(key not in prediction_keys, f"{participant_id}: 예측이 중복됩니다: {key}")
            prediction_keys.add(key)
            _assert(
                prediction["ci_low"]
                <= prediction["c80_sim"]
                <= prediction["ci_high"],
                f"{participant_id} {key}: C80이 신뢰구간 안에 없습니다.",
            )
        validation = participant["validation"]
        if validation["stable"]:
            _assert(
                validation["coefficient_of_variation"] <= 0.20,
                f"{participant_id}: stable=true인데 CV가 0.20을 넘습니다.",
            )
            _assert(
                validation["pooled_c80"] is not None,
                f"{participant_id}: 안정적 검증에는 pooled_c80이 필요합니다.",
            )
        else:
            _assert(
                validation["pooled_c80"] is None,
                f"{participant_id}: 불안정 검증은 pooled_c80을 만들면 안 됩니다.",
            )


def _validate_correction_results(payload: dict) -> None:
    threshold = payload["threshold"]
    _assert(
        threshold["ci_low"] <= threshold["c80_sim"] <= threshold["ci_high"],
        "correction_results: C80이 신뢰구간 안에 없습니다.",
    )
    expected_target = (
        threshold["c80_sim"]
        if threshold["target_policy"] == "point"
        else threshold["ci_high"]
    )
    _assert(
        _close(threshold["selected_target"], expected_target),
        "correction_results: target_policy와 selected_target이 일치하지 않습니다.",
    )
    _assert(
        threshold["selected_target"] <= threshold["original_min"],
        "correction_results: selected_target은 original_min보다 클 수 없습니다.",
    )
    _assert(
        payload["search"]["max_shift"] >= payload["search"]["step"],
        "correction_results: max_shift는 step 이상이어야 합니다.",
    )

    seen = set()
    counts = {"unchanged": 0, "corrected": 0, "search_failed": 0}
    target_met_count = 0
    for row in payload["results"]:
        pair_id = row["pair_id"]
        _assert(pair_id not in seen, f"correction_results: pair_id 중복: {pair_id}")
        seen.add(pair_id)
        counts[row["status"]] += 1
        target_met_count += int(row["target_met"])
        _assert(
            row["target_met"]
            == (row["simulated_delta_e00_after"] + 1e-9 >= expected_target),
            f"{pair_id}: target_met이 보정 후 색차와 맞지 않습니다.",
        )
        _assert(
            _close(
                row["total_color_change"],
                row["color_change_a"] + row["color_change_b"],
            ),
            f"{pair_id}: total_color_change가 A+B와 다릅니다.",
        )
        if row["status"] == "corrected":
            _assert(row["strategy"] is not None, f"{pair_id}: 보정 전략이 없습니다.")
            _assert(row["target_met"], f"{pair_id}: corrected인데 목표를 못 맞췄습니다.")
            _assert(row["in_srgb_gamut"], f"{pair_id}: corrected 색이 sRGB gamut 밖입니다.")
            _assert(
                row["total_color_change"] > 0,
                f"{pair_id}: corrected인데 색 이동량이 0입니다.",
            )
            _assert(
                row["corrected_rgb_a"] != row["input_rgb_a"]
                or row["corrected_rgb_b"] != row["input_rgb_b"],
                f"{pair_id}: corrected인데 RGB가 바뀌지 않았습니다.",
            )
        elif row["status"] == "unchanged":
            _assert(row["strategy"] is None, f"{pair_id}: unchanged에 전략이 있습니다.")
            _assert(
                _close(row["total_color_change"], 0.0),
                f"{pair_id}: unchanged인데 색 이동이 있습니다.",
            )
            _assert(
                row["corrected_rgb_a"] == row["input_rgb_a"]
                and row["corrected_rgb_b"] == row["input_rgb_b"],
                f"{pair_id}: unchanged인데 RGB가 바뀌었습니다.",
            )
            _assert(
                not row["palette_constraint_used"],
                f"{pair_id}: 탐색을 생략했는데 팔레트 제약 사용으로 기록됐습니다.",
            )
        else:
            _assert(row["strategy"] is None, f"{pair_id}: 실패 결과에 전략이 있습니다.")
            _assert(not row["target_met"], f"{pair_id}: 실패 결과가 목표를 만족합니다.")
        _assert(
            not row["palette_constraint_used"] or bool(row["palette"]),
            f"{pair_id}: 빈 palette에 제약 사용으로 기록됐습니다.",
        )

    summary = payload["summary"]
    _assert(summary["pair_count"] == len(payload["results"]), "요약 pair_count 오류")
    for status, count in counts.items():
        _assert(
            summary[f"{status}_count"] == count,
            f"요약 {status}_count 오류",
        )
    _assert(summary["target_met_count"] == target_met_count, "요약 target_met_count 오류")


_SEMANTIC_VALIDATORS = {
    "experiment_config": _validate_experiment_config,
    "trial": _validate_trial,
    "application_conditions": _validate_application_conditions,
    "validation_pairs": _validate_validation_pairs,
    "participant_thresholds": _validate_participant_thresholds,
    "correction_results": _validate_correction_results,
}


def validate_payload(contract: str, payload: Any) -> Any:
    """메모리의 payload를 스키마와 연구 의미 규칙으로 검증한다."""
    _validate_schema(contract, payload)
    _SEMANTIC_VALIDATORS[contract](payload)
    return payload


def validate_file(contract: str, path: str | Path) -> Any:
    """일반 JSON 파일 하나를 검증한다."""
    return validate_payload(contract, _read_json(path))


def validate_jsonl(path: str | Path) -> list[dict]:
    """시행 JSONL을 한 줄씩 검증하고 레코드 목록을 반환한다."""
    file_path = Path(path)
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ContractError(f"파일이 없습니다: {file_path}") from error
    _assert(bool(lines), f"JSONL 파일이 비었습니다: {file_path}")

    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ContractError(f"{file_path}:{line_number}: 빈 줄은 허용하지 않습니다.")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(
                f"{file_path}:{line_number}: JSON 형식이 깨졌습니다: {error}"
            ) from error
        try:
            validate_payload("trial", record)
        except ContractError as error:
            raise ContractError(f"{file_path}:{line_number}: {error}") from error
        records.append(record)
    return records


def validate_all_examples() -> dict[str, int]:
    """저장소의 계약 예제를 모두 검증한다."""
    counts = {}
    for contract, path in EXAMPLE_PATHS.items():
        if contract == "trial":
            counts[contract] = len(validate_jsonl(path))
        else:
            validate_file(contract, path)
            counts[contract] = 1
    return counts


__all__ = [
    "ContractError",
    "EXAMPLE_PATHS",
    "SCHEMA_PATHS",
    "validate_all_examples",
    "validate_file",
    "validate_jsonl",
    "validate_payload",
]
