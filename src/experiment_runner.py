"""사람 실험을 시작하기 전의 안전 점검과 시행표 생성을 담당한다.

이 모듈은 동결 설정 검사, 균형 시행표 생성, 픽셀 렌더러 게이트 연결과
내구성 있는 JSONL 저널을 제공한다. 자극 생성은 ``stimulus_renderer``, 실제
전체화면 제시·응답 수집은 ``response_gui``가 담당한다. 설정만 검사한 결과는
manifest·화면·Git·세션 동의가 없으므로 실제 수집 허가가 아니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiment_design import (
    SZAFIR_LINE_MAX_VISUAL_ANGLE_DEG,
    SZAFIR_LINE_MIN_VISUAL_ANGLE_DEG,
    geometric_validation_widths,
    pixels_to_visual_angle_deg,
)
from stimulus_renderer import RendererValidationError, verify_renderer_environment


class CollectionBlockedError(ValueError):
    """설정이 실제 사람 자료 수집 요건을 충족하지 않을 때 발생한다."""


class JournalError(ValueError):
    """시행 저널이 손상됐거나 중복 레코드를 받았을 때 발생한다."""


_PLACEHOLDER_RE = re.compile(
    r"(?:example|placeholder|not[_ -]?frozen|tbd|todo|unknown|미정|예시)",
    re.IGNORECASE,
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_JOURNAL_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.jsonl$")
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_COLLECTION_PHASES = ("practice", "calibration", "validation")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 객체에 중복 key가 있습니다: {key}")
        result[key] = value
    return result


def _value_at(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise CollectionBlockedError(f"필수 설정값이 없습니다: {path}")
        value = value[part]
    return value


def _require_concrete(payload: Mapping[str, Any], path: str) -> Any:
    value = _value_at(payload, path)
    if value is None or isinstance(value, str) and (
        not value.strip() or _PLACEHOLDER_RE.search(value)
    ):
        raise CollectionBlockedError(f"필수 설정값이 확정되지 않았습니다: {path}")
    return value


def _positive_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CollectionBlockedError(f"{path}는 양수여야 합니다.")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise CollectionBlockedError(f"{path}는 양수여야 합니다.")
    return number


def _rgb(value: Any, path: str) -> None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
        or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255 for v in value)
    ):
        raise CollectionBlockedError(f"{path}는 0~255 정수 RGB 세 값이어야 합니다.")


def _validated_journal_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the frozen journal contract after strict path/session checks.

    Human journals intentionally support only repository-relative storage. A
    phase therefore has exactly one config-bound destination and replacing the
    CLI path cannot create a second history that hides an abort or elapsed time.
    """

    journal = _value_at(config, "journal")
    if not isinstance(journal, Mapping):
        raise CollectionBlockedError("journal은 객체여야 합니다.")
    if journal.get("storage_policy") != "repository-relative-gitignored":
        raise CollectionBlockedError(
            "journal.storage_policy는 repository-relative-gitignored여야 합니다."
        )
    root = journal.get("root_repository_relative")
    if root != "data/experiment/raw":
        raise CollectionBlockedError(
            "journal.root_repository_relative는 data/experiment/raw로 고정해야 합니다."
        )

    phase_files = journal.get("phase_files")
    if not isinstance(phase_files, Mapping) or set(phase_files) != set(_COLLECTION_PHASES):
        raise CollectionBlockedError(
            "journal.phase_files에는 practice/calibration/validation 파일이 정확히 필요합니다."
        )
    filenames: list[str] = []
    participant_id = _value_at(config, "observer_profile.participant_id")
    for phase in _COLLECTION_PHASES:
        filename = phase_files.get(phase)
        if not isinstance(filename, str) or not _JOURNAL_FILENAME_RE.fullmatch(filename):
            raise CollectionBlockedError(
                f"journal.phase_files.{phase}는 경로 구분자 없는 안전한 .jsonl 파일명이어야 합니다."
            )
        expected_filename = f"{participant_id}-{phase}.jsonl"
        if filename != expected_filename:
            raise CollectionBlockedError(
                f"journal.phase_files.{phase}는 참가자와 결박된 {expected_filename}이어야 합니다."
            )
        filenames.append(filename)
    if len(set(filenames)) != len(filenames):
        raise CollectionBlockedError("phase별 journal 파일명은 서로 달라야 합니다.")

    session_ids = journal.get("session_ids")
    if not isinstance(session_ids, Mapping) or set(session_ids) != set(_COLLECTION_PHASES):
        raise CollectionBlockedError(
            "journal.session_ids에는 practice/calibration/validation이 정확히 필요합니다."
        )
    practice_id = session_ids.get("practice")
    calibration_ids = session_ids.get("calibration")
    validation_ids = session_ids.get("validation")
    if not isinstance(practice_id, str):
        raise CollectionBlockedError("journal.session_ids.practice가 필요합니다.")
    if not isinstance(calibration_ids, Mapping) or set(calibration_ids) != {
        "1", "2", "3", "4"
    }:
        raise CollectionBlockedError(
            "journal.session_ids.calibration에는 block 1~4가 정확히 필요합니다."
        )
    if not isinstance(validation_ids, Mapping) or set(validation_ids) != {
        "V01", "V02", "V03"
    }:
        raise CollectionBlockedError(
            "journal.session_ids.validation에는 block V01~V03가 정확히 필요합니다."
        )
    frozen_ids = [
        practice_id,
        *(calibration_ids[key] for key in ("1", "2", "3", "4")),
        *(validation_ids[key] for key in ("V01", "V02", "V03")),
    ]
    if any(
        not isinstance(value, str) or not _SESSION_ID_RE.fullmatch(value)
        for value in frozen_ids
    ):
        raise CollectionBlockedError(
            "동결 session_id는 영문·숫자·밑줄·하이픈만 포함해야 합니다."
        )
    if len(set(frozen_ids)) != len(frozen_ids):
        raise CollectionBlockedError("모든 collection block의 session_id는 서로 달라야 합니다.")
    observer_session_ids = {
        _value_at(config, "observer_profile.severity_session_id"),
        _value_at(config, "observer_profile.lapse_session_id"),
    }
    if observer_session_ids & set(frozen_ids):
        raise CollectionBlockedError(
            "collection session_id는 독립 severity/lapse session_id와 달라야 합니다."
        )
    return journal


def canonical_journal_path(
    config: Mapping[str, Any], phase: str, repo_root: str | Path
) -> Path:
    """Resolve the only allowed journal path for one frozen config phase."""

    if phase not in _COLLECTION_PHASES:
        raise CollectionBlockedError(f"지원하지 않는 journal phase입니다: {phase}")
    journal = _validated_journal_config(config)
    root = Path(repo_root).resolve()
    target = (
        root
        / str(journal["root_repository_relative"])
        / str(journal["phase_files"][phase])
    ).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise CollectionBlockedError("canonical journal이 저장소 밖을 가리킵니다.") from error
    return target


def expected_collection_session_id(
    config: Mapping[str, Any],
    phase: str,
    *,
    session_block: int | None = None,
    validation_block_id: str | None = None,
) -> str:
    """Return the preregistered session identifier for a selected block."""

    journal = _validated_journal_config(config)
    session_ids = journal["session_ids"]
    if phase == "practice":
        if session_block is not None or validation_block_id is not None:
            raise CollectionBlockedError("practice에는 block 식별자를 지정하지 않습니다.")
        return str(session_ids["practice"])
    if phase == "calibration":
        if session_block not in {1, 2, 3, 4} or validation_block_id is not None:
            raise CollectionBlockedError("calibration session block은 1~4여야 합니다.")
        return str(session_ids["calibration"][str(session_block)])
    if phase == "validation":
        if session_block is not None or validation_block_id not in {"V01", "V02", "V03"}:
            raise CollectionBlockedError("validation block은 V01~V03여야 합니다.")
        return str(session_ids["validation"][validation_block_id])
    raise CollectionBlockedError(f"지원하지 않는 collection phase입니다: {phase}")


def assert_schedule_config(config: Mapping[str, Any]) -> None:
    """사전 동결·승인·시행표 생성값을 모두 갖춘 설정만 통과시킨다.

    기존 ``0.1-draft`` 계약은 합성 자료 전용이므로 의도적으로 거부한다.
    이 검사는 스키마 검증의 대체물이 아니며 렌더러·응답 GUI 검증과도
    별개다. 통과는 시행표를 만들 수 있다는 뜻이지 사람 자료 수집 허가가 아니다.
    """

    if not isinstance(config, Mapping):
        raise CollectionBlockedError("설정은 JSON 객체여야 합니다.")
    if config.get("schema_version") != "1.0":
        raise CollectionBlockedError("사람 자료 수집에는 schema_version 1.0이 필요합니다.")
    if config.get("status") not in {"frozen", "preregistered"}:
        raise CollectionBlockedError("status는 frozen 또는 preregistered여야 합니다.")
    if config.get("human_data_collection_allowed") is not True:
        raise CollectionBlockedError("human_data_collection_allowed=true 승인이 필요합니다.")

    _validated_journal_config(config)

    if _value_at(config, "task.type") != "4afc_odd_one_out":
        raise CollectionBlockedError("task.type은 4afc_odd_one_out이어야 합니다.")
    if _value_at(config, "task.layout") != "2x2":
        raise CollectionBlockedError("task.layout은 2x2여야 합니다.")
    if _value_at(config, "task.choice_count") != 4:
        raise CollectionBlockedError("task.choice_count는 4여야 합니다.")
    for path in ("task.practice_feedback", "task.main_feedback"):
        if not isinstance(_value_at(config, path), bool):
            raise CollectionBlockedError(f"{path}는 명시적인 true/false여야 합니다.")
    if _value_at(config, "task.main_feedback") is not False:
        raise CollectionBlockedError(
            "본 시행의 반복 색쌍 학습과 phase 차이를 막기 위해 task.main_feedback=false여야 합니다."
        )

    for path in (
        "config_version",
        "display.model",
        "display.color_profile",
        "display.colorimetry_policy",
        "display.calibration_id",
        "display.luminance_measurement_method",
        "display.width_measurement_method",
        "display.viewing_distance_control",
        "display.operating_system",
        "display.native_gui_runtime",
        "display.pixel_mapping_verification",
        "display.ambient_light_condition",
        "display.ambient_light_measurement_method",
        "line.cap",
        "line.antialiasing",
        "line.renderer",
        "line.pixel_snapping",
        "color_design.stimulus_family_id",
        "color_design.gamut_policy",
        "color_design.rgb_quantization",
        "observer_profile.participant_id",
        "observer_profile.cvd_type",
        "observer_profile.severity_source",
        "observer_profile.severity_session_id",
        "observer_profile.lapse_source",
        "observer_profile.lapse_session_id",
        "sampling.randomization",
        "privacy.approval_id",
        "privacy.approved_at_utc",
        "privacy.consent_version",
        "privacy.pseudonymization",
        "privacy.storage_policy",
        "timing.stop_rules",
        "timing.resume_policy",
        "timing.active_trial_abort_policy",
        "timing.incomplete_block_time_limit_policy",
    ):
        _require_concrete(config, path)

    for path in (
        "display.width_mm",
        "display.resolution_width_px",
        "display.resolution_height_px",
        "display.os_scale_percent",
        "display.app_scale_percent",
        "display.viewing_distance_mm",
        "display.brightness_percent",
        "line.length_px",
        "coordinates.center_spacing_px",
        "timing.fixation_ms",
        "timing.inter_trial_ms",
        "timing.max_session_duration_min",
        "timing.planned_break_interval_trials",
        "timing.planned_break_duration_min",
        "sampling.practice_trials",
    ):
        # practice_trials may be zero, unlike the remaining physical/timing values.
        value = _require_concrete(config, path)
        if path == "sampling.practice_trials":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CollectionBlockedError(f"{path}는 0 이상의 정수여야 합니다.")
        else:
            _positive_number(value, path)

    brightness_percent = float(_value_at(config, "display.brightness_percent"))
    if brightness_percent > 100.0:
        raise CollectionBlockedError("display.brightness_percent는 100 이하여야 합니다.")

    colorimetry_policy = _value_at(config, "display.colorimetry_policy")
    if colorimetry_policy not in {
        "measured-display-colorimetry",
        "nominal-srgb-device-specific",
    }:
        raise CollectionBlockedError(
            "display.colorimetry_policy는 measured-display-colorimetry 또는 "
            "nominal-srgb-device-specific이어야 합니다."
        )
    luminance = _value_at(config, "display.luminance_cd_m2")
    luminance_method = _value_at(config, "display.luminance_measurement_method")
    if luminance is None:
        if colorimetry_policy != "nominal-srgb-device-specific":
            raise CollectionBlockedError(
                "휘도 미측정은 nominal-srgb-device-specific 정책에서만 허용합니다."
            )
        if luminance_method != "not_measured_no_luminance_meter":
            raise CollectionBlockedError(
                "휘도 미측정이면 display.luminance_measurement_method를 "
                "not_measured_no_luminance_meter로 기록해야 합니다."
            )
    else:
        _positive_number(luminance, "display.luminance_cd_m2")

    _rgb(_require_concrete(config, "display.background_rgb"), "display.background_rgb")
    if _value_at(config, "display.night_mode_disabled") is not True:
        raise CollectionBlockedError("display.night_mode_disabled=true 확인이 필요합니다.")
    if _value_at(config, "timing.resume_policy") != "append-only-retry-incomplete":
        raise CollectionBlockedError(
            "timing.resume_policy는 append-only-retry-incomplete여야 합니다."
        )
    if _value_at(config, "timing.active_trial_abort_policy") != "invalidate-analysis-no-retry":
        raise CollectionBlockedError(
            "timing.active_trial_abort_policy는 invalidate-analysis-no-retry여야 합니다."
        )
    if (
        _value_at(config, "timing.incomplete_block_time_limit_policy")
        != "invalidate-participant-dataset-no-retry"
    ):
        raise CollectionBlockedError(
            "timing.incomplete_block_time_limit_policy는 "
            "invalidate-participant-dataset-no-retry여야 합니다."
        )

    orientation = _require_concrete(config, "line.orientation_deg")
    if (
        isinstance(orientation, bool)
        or not isinstance(orientation, (int, float))
        or not math.isfinite(float(orientation))
    ):
        raise CollectionBlockedError("line.orientation_deg는 유한한 수여야 합니다.")

    color_design = _value_at(config, "color_design")
    if not isinstance(color_design, Mapping):
        raise CollectionBlockedError("color_design은 객체여야 합니다.")
    for field in ("midpoint_lab", "direction_lab"):
        color = color_design.get(field)
        if (
            not isinstance(color, Sequence)
            or isinstance(color, (str, bytes))
            or len(color) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in color
            )
        ):
            raise CollectionBlockedError(
                f"color_design.{field}는 유한한 수 세 값이어야 합니다."
            )
    if math.sqrt(sum(float(value) ** 2 for value in color_design["direction_lab"])) <= 0:
        raise CollectionBlockedError("color_design.direction_lab은 영벡터일 수 없습니다.")
    if color_design.get("symmetric_about_midpoint") is not True:
        raise CollectionBlockedError(
            "color_design.symmetric_about_midpoint=true가 필요합니다."
        )
    if color_design.get("gamut_policy") != "reject":
        raise CollectionBlockedError("color_design.gamut_policy는 reject여야 합니다.")
    if color_design.get("rgb_quantization") != "nearest_8bit":
        raise CollectionBlockedError(
            "color_design.rgb_quantization은 nearest_8bit여야 합니다."
        )
    _positive_number(
        color_design.get("target_absolute_tolerance_delta_e00"),
        "color_design.target_absolute_tolerance_delta_e00",
    )
    relative_tolerance = color_design.get("target_relative_tolerance")
    if (
        isinstance(relative_tolerance, bool)
        or not isinstance(relative_tolerance, (int, float))
        or not math.isfinite(float(relative_tolerance))
        or not 0 <= float(relative_tolerance) < 1
    ):
        raise CollectionBlockedError(
            "color_design.target_relative_tolerance는 0 이상 1 미만이어야 합니다."
        )
    grid_steps = color_design.get("search_grid_steps")
    if isinstance(grid_steps, bool) or not isinstance(grid_steps, int) or grid_steps < 32:
        raise CollectionBlockedError(
            "color_design.search_grid_steps는 32 이상의 정수여야 합니다."
        )

    observer = _value_at(config, "observer_profile")
    if not isinstance(observer, Mapping):
        raise CollectionBlockedError("observer_profile은 객체여야 합니다.")
    if observer.get("cvd_type") not in {"protan", "deutan"}:
        raise CollectionBlockedError(
            "observer_profile.cvd_type은 protan 또는 deutan이어야 합니다."
        )
    severity = observer.get("severity")
    if (
        isinstance(severity, bool)
        or not isinstance(severity, (int, float))
        or not math.isfinite(float(severity))
        or not 0 <= float(severity) <= 1
    ):
        raise CollectionBlockedError("observer_profile.severity는 0~1이어야 합니다.")
    fixed_lapse = observer.get("fixed_lapse")
    if (
        isinstance(fixed_lapse, bool)
        or not isinstance(fixed_lapse, (int, float))
        or not math.isfinite(float(fixed_lapse))
        or not 0 <= float(fixed_lapse) < 0.20
    ):
        raise CollectionBlockedError(
            "observer_profile.fixed_lapse는 0 이상 0.20 미만이어야 합니다."
        )
    positions = _require_concrete(config, "coordinates.positions_px")
    if (
        not isinstance(positions, Sequence)
        or isinstance(positions, (str, bytes))
        or len(positions) != 4
        or any(
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes))
            or len(point) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in point)
            for point in positions
        )
    ):
        raise CollectionBlockedError("coordinates.positions_px에는 유한한 [x,y] 좌표 네 개가 필요합니다.")
    if len({(float(point[0]), float(point[1])) for point in positions}) != 4:
        raise CollectionBlockedError("coordinates.positions_px의 네 좌표는 서로 달라야 합니다.")

    deadline_policy = _require_concrete(config, "timing.response_deadline_policy")
    deadline = _value_at(config, "timing.response_deadline_ms")
    if deadline_policy != "unlimited" or deadline is not None:
        raise CollectionBlockedError(
            "본 연구는 선택적 timeout 재시도로 C80이 낙관 편향되지 않도록 "
            "timing.response_deadline_policy=unlimited와 response_deadline_ms=null만 허용합니다."
        )

    widths = _require_concrete(config, "calibration.widths")
    if not isinstance(widths, Sequence) or isinstance(widths, (str, bytes)) or len(widths) != 3:
        raise CollectionBlockedError("calibration.widths에는 굵기 세 개가 필요합니다.")
    width_values = []
    display_conversion = {
        "display_width_mm": float(_value_at(config, "display.width_mm")),
        "horizontal_resolution_px": float(
            _value_at(config, "display.resolution_width_px")
        ),
        "viewing_distance_mm": float(
            _value_at(config, "display.viewing_distance_mm")
        ),
    }
    for index, row in enumerate(widths):
        if not isinstance(row, Mapping):
            raise CollectionBlockedError(f"calibration.widths[{index}]가 객체가 아닙니다.")
        width_px = row.get("width_px")
        if isinstance(width_px, bool) or not isinstance(width_px, int) or width_px < 1:
            raise CollectionBlockedError(
                f"calibration.widths[{index}].width_px는 1 이상의 정수여야 합니다."
            )
        width_values.append(width_px)
        width_deg = _positive_number(
            row.get("width_deg"), f"calibration.widths[{index}].width_deg"
        )
        calculated = pixels_to_visual_angle_deg(width_px, **display_conversion)
        if not math.isclose(width_deg, calculated, rel_tol=1e-6, abs_tol=1e-9):
            raise CollectionBlockedError(
                f"calibration.widths[{index}].width_deg가 실측 화면·거리의 px 환산값과 다릅니다."
            )
        if not SZAFIR_LINE_MIN_VISUAL_ANGLE_DEG <= width_deg <= SZAFIR_LINE_MAX_VISUAL_ANGLE_DEG:
            raise CollectionBlockedError(
                f"calibration.widths[{index}]가 Szafir 선 측정범위 "
                f"{SZAFIR_LINE_MIN_VISUAL_ANGLE_DEG:g}~"
                f"{SZAFIR_LINE_MAX_VISUAL_ANGLE_DEG:g}도 밖입니다."
            )
    if len(set(width_values)) != 3:
        raise CollectionBlockedError("세 교정 굵기는 서로 달라야 합니다.")
    if width_values != sorted(width_values):
        raise CollectionBlockedError("세 교정 굵기는 작은 값부터 오름차순이어야 합니다.")

    validation = _value_at(config, "validation")
    if not isinstance(validation, Mapping):
        raise CollectionBlockedError("validation은 객체여야 합니다.")
    candidates = validation.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or len(candidates) != 2:
        raise CollectionBlockedError("validation.candidates에는 미사용 굵기 두 개가 필요합니다.")
    geometric_targets = geometric_validation_widths(width_values)
    candidate_values: list[int] = []
    for index, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            raise CollectionBlockedError(f"validation.candidates[{index}]가 객체가 아닙니다.")
        rendered = row.get("rendered_width_px")
        if isinstance(rendered, bool) or not isinstance(rendered, int) or rendered < 1:
            raise CollectionBlockedError(
                f"validation.candidates[{index}].rendered_width_px는 1 이상의 정수여야 합니다."
            )
        if rendered in width_values or not width_values[index] < rendered < width_values[index + 1]:
            raise CollectionBlockedError(
                f"validation.candidates[{index}]는 대응 교정 구간 내부의 미사용 정수 px여야 합니다."
            )
        target = _positive_number(
            row.get("geometric_target_px"),
            f"validation.candidates[{index}].geometric_target_px",
        )
        if not math.isclose(target, geometric_targets[index], rel_tol=1e-6, abs_tol=1e-9):
            raise CollectionBlockedError(
                f"validation.candidates[{index}].geometric_target_px가 교정 굵기 기하평균과 다릅니다."
            )
        width_deg = _positive_number(
            row.get("width_deg"), f"validation.candidates[{index}].width_deg"
        )
        calculated = pixels_to_visual_angle_deg(rendered, **display_conversion)
        if not math.isclose(width_deg, calculated, rel_tol=1e-6, abs_tol=1e-9):
            raise CollectionBlockedError(
                f"validation.candidates[{index}].width_deg가 실측 화면·거리의 px 환산값과 다릅니다."
            )
        if not SZAFIR_LINE_MIN_VISUAL_ANGLE_DEG <= width_deg <= SZAFIR_LINE_MAX_VISUAL_ANGLE_DEG:
            raise CollectionBlockedError(
                f"validation.candidates[{index}]가 Szafir 선 측정범위 밖입니다."
            )
        candidate_values.append(rendered)
    if len(set(candidate_values)) != 2:
        raise CollectionBlockedError("validation 후보 두 개는 서로 달라야 합니다.")
    if validation.get("independent_blocks") != 3 or validation.get("trials_per_block") != 64:
        raise CollectionBlockedError("validation은 독립 3 block×64시행이어야 합니다.")
    if validation.get("delta_e00_level_count") != 8:
        raise CollectionBlockedError("validation은 정확히 8개 색차 수준을 사용해야 합니다.")
    if validation.get("randomization") != "balanced-position-and-odd-color-within-block":
        raise CollectionBlockedError("validation.randomization 정책이 동결 설계와 다릅니다.")

    model_plan = _value_at(config, "model_plan")
    if not isinstance(model_plan, Mapping):
        raise CollectionBlockedError("model_plan은 객체여야 합니다.")
    if model_plan.get("models") != ["B96", "B288", "C96", "D288"]:
        raise CollectionBlockedError("model_plan.models는 B96/B288/C96/D288 순서여야 합니다.")
    if model_plan.get("include_szafir_transfer") is not True:
        raise CollectionBlockedError("C96 검증에는 include_szafir_transfer=true가 필요합니다.")
    probability = model_plan.get("szafir_probability_coordinate")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0 < float(probability) < 1
    ):
        raise CollectionBlockedError(
            "model_plan.szafir_probability_coordinate는 0과 1 사이여야 합니다."
        )

    trials_per_width = _require_concrete(config, "calibration.trials_per_width")
    if trials_per_width != 96:
        raise CollectionBlockedError("동결 설계는 굵기당 정확히 96시행이어야 합니다.")

    levels = _require_concrete(config, "sampling.delta_e00_levels")
    if (
        not isinstance(levels, Sequence)
        or isinstance(levels, (str, bytes))
        or len(levels) != 12
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) <= 0 for v in levels)
        or len({float(v) for v in levels}) != 12
    ):
        raise CollectionBlockedError(
            "sampling.delta_e00_levels에는 서로 다른 양의 유한값 12개가 필요합니다."
        )
    if list(map(float, levels)) != sorted(map(float, levels)):
        raise CollectionBlockedError("sampling.delta_e00_levels는 오름차순이어야 합니다.")
    if (
        _value_at(config, "sampling.randomization")
        != "balanced-4-session-blocks-within-block-shuffle"
    ):
        raise CollectionBlockedError(
            "sampling.randomization은 "
            "balanced-4-session-blocks-within-block-shuffle여야 합니다."
        )

    if _value_at(config, "privacy.approved") is not True:
        raise CollectionBlockedError("privacy.approved=true인 승인 기록이 필요합니다.")


def assess_experiment_gates(config: Mapping[str, Any]) -> dict[str, Any]:
    """설정·픽셀 렌더러·응답 GUI 게이트를 서로 섞지 않고 보고한다."""

    blockers: list[dict[str, str]] = []
    config_gate_passed = False
    renderer_verified = False
    renderer_details: dict[str, Any] | None = None
    try:
        assert_schedule_config(config)
        config_gate_passed = True
    except CollectionBlockedError as error:
        blockers.append({"gate": "config", "message": str(error)})

    if config_gate_passed:
        try:
            renderer_details = verify_renderer_environment(config)
            renderer_verified = True
        except RendererValidationError as error:
            blockers.append({"gate": "renderer", "message": str(error)})
    else:
        blockers.append(
            {
                "gate": "renderer",
                "message": "설정 게이트가 차단되어 렌더러 검증을 실행하지 않았습니다.",
            }
        )

    # GUI 구현의 존재와 특정 세션의 런타임 허가는 다른 상태다. 실제 허가는
    # response_gui.prepare_collection이 manifest·화면·Git을 검사하고 CLI가
    # 세션 동의를 명시한 뒤에만 결정한다.
    human_response_collection_implemented = (
        Path(__file__).with_name("response_gui.py").is_file()
    )
    blockers.append(
        {
            "gate": "session_runtime",
            "message": (
                "config-only 검사는 실제 수집을 허가하지 않습니다. "
                "response_gui에서 phase별 manifest·화면·Git·세션 동의를 검증하세요."
            ),
        }
    )
    allowed = False
    return {
        "config_gate_passed": config_gate_passed,
        # 이전 키는 보고서 소비자의 전환을 위해 뜻을 좁혀 함께 둔다.
        "allowed_by_config_gate": config_gate_passed,
        "renderer_implemented": True,
        "renderer_verified": renderer_verified,
        "renderer_details": renderer_details,
        "human_response_collection_implemented": human_response_collection_implemented,
        "human_data_collection_allowed": allowed,
        "blockers": blockers,
    }


def assert_human_collection_allowed(config: Mapping[str, Any]) -> None:
    """모든 실행 게이트가 통과하지 않으면 사람 자료 수집을 차단한다."""

    report = assess_experiment_gates(config)
    if not report["human_data_collection_allowed"]:
        reasons = "; ".join(
            f"{item['gate']}: {item['message']}" for item in report["blockers"]
        )
        raise CollectionBlockedError(f"사람 자료 수집 차단: {reasons}")


def _canonical_config_sha256(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _detect_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip().lower()
    return commit if _COMMIT_RE.fullmatch(commit) else None


def build_balanced_schedule(
    config: Mapping[str, Any],
    seed: int,
    *,
    config_sha256: str | None = None,
    git_commit: str | None = None,
) -> list[dict[str, Any]]:
    """3굵기×12수준×4위치×odd색을 4개 세션 블록에 균형화한다.

    각 ``굵기×ΔE00`` 셀에는 odd 위치 1~4와 odd 색 a/b의 조합이 한 번씩
    들어가므로 셀당 8회, 굵기당 96회가 된다. 각 블록에는 셀마다 두 조합,
    굵기마다 24회, 총 72회가 들어가며 블록 내부 순서만 ``seed``로
    결정론적으로 무작위화한다. 실제 session_id는 실행 시 블록에 매핑한다.
    """

    assert_schedule_config(config)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed는 0 이상의 정수여야 합니다.")

    digest = (config_sha256 or _canonical_config_sha256(config)).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("config_sha256은 64자리 소문자 16진수여야 합니다.")
    commit = (git_commit or _detect_git_commit())
    if commit is not None:
        commit = commit.lower()
        if not _COMMIT_RE.fullmatch(commit):
            raise ValueError("git_commit은 40자리 소문자 16진수여야 합니다.")

    widths = config["calibration"]["widths"]
    levels = [float(value) for value in config["sampling"]["delta_e00_levels"]]
    observer = config["observer_profile"]
    stimulus_family_id = config["color_design"]["stimulus_family_id"]
    blocks: dict[int, list[dict[str, Any]]] = {index: [] for index in range(1, 5)}
    for width_index, width in enumerate(widths, start=1):
        for level_index, level in enumerate(levels, start=1):
            cell_shift = ((width_index - 1) * len(levels) + level_index - 1) % 4
            for session_block in range(1, 5):
                combinations = (
                    ((session_block - 1 + cell_shift) % 4 + 1, "a"),
                    ((session_block + cell_shift) % 4 + 1, "b"),
                )
                for correct_position, odd_color in combinations:
                    blocks[session_block].append(
                        {
                            "session_block": session_block,
                            "width_index": width_index,
                            "stroke_width_px": int(width["width_px"]),
                            "stroke_width_deg": float(width["width_deg"]),
                            "delta_e00_level_index": level_index,
                            "target_delta_e00_simulated": level,
                            "correct_position": correct_position,
                            "odd_color": odd_color,
                            "random_seed": seed,
                            "config_version": config["config_version"],
                            "config_sha256": digest,
                            "git_commit": commit,
                            "stimulus_family_id": stimulus_family_id,
                            "participant_id": observer["participant_id"],
                            "cvd_type": observer["cvd_type"],
                            "machado_severity": float(observer["severity"]),
                            "severity_source": observer["severity_source"],
                            "severity_session_id": observer["severity_session_id"],
                            "fixed_lapse": float(observer["fixed_lapse"]),
                            "lapse_source": observer["lapse_source"],
                            "lapse_session_id": observer["lapse_session_id"],
                        }
                    )

    rows: list[dict[str, Any]] = []
    for session_block in range(1, 5):
        block = blocks[session_block]
        random.Random(seed + session_block * 1_000_003).shuffle(block)
        rows.extend(block)
    for trial_id, row in enumerate(rows, start=1):
        row["trial_id"] = trial_id
    return rows


class TrialJournal:
    """한 시행씩 디스크에 동기화하는 재개 가능한 append-only JSONL 저널."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        # Completed trials and aborted main trials are terminal. Only an
        # aborted practice trial is eligible for another attempt.
        self._completed_keys: set[tuple[str, str, int]] = set()
        self._terminal_keys: set[tuple[str, str, int]] = set()
        self._attempted_keys: set[tuple[str, str, int]] = set()
        self._records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._load_existing()

    @staticmethod
    def _is_completed(record: Mapping[str, Any]) -> bool:
        status = record.get("trial_status")
        if status is None:
            # TrialJournal의 기존 범용 사용처/옛 회귀 fixture는 상태 필드가
            # 없으며, 이 경우에는 정상 완료 레코드로 취급한다.
            return True
        if status not in {"completed", "aborted"}:
            raise JournalError("trial_status는 completed 또는 aborted여야 합니다.")
        return status == "completed"

    @classmethod
    def _is_terminal(cls, record: Mapping[str, Any]) -> bool:
        if cls._is_completed(record):
            return True
        phase = record.get("phase")
        if phase == "practice":
            return False
        if phase in {"calibration", "validation"}:
            return True
        raise JournalError(
            "aborted 레코드는 practice/calibration/validation phase가 필요합니다."
        )

    @staticmethod
    def _key(record: Mapping[str, Any]) -> tuple[str, str, int]:
        try:
            participant = record["participant_id"]
            session = record["session_id"]
            config_sha256 = record["config_sha256"]
            trial = record["trial_id"]
        except KeyError as error:
            raise JournalError(f"저널 키 필드가 없습니다: {error.args[0]}") from error
        if not isinstance(participant, str) or not participant:
            raise JournalError("participant_id는 빈 문자열이 아니어야 합니다.")
        if not isinstance(session, str) or not session:
            raise JournalError("session_id는 빈 문자열이 아니어야 합니다.")
        if not isinstance(config_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", config_sha256
        ):
            raise JournalError("config_sha256은 64자리 소문자 16진수여야 합니다.")
        if isinstance(trial, bool) or not isinstance(trial, int) or trial < 1:
            raise JournalError("trial_id는 1 이상의 정수여야 합니다.")
        # session_id를 바꿔 예정된 같은 시행을 다시 실행하지 못하게 한다.
        # session_id는 계약 검증에서 한 세션의 설정 일관성을 묶는 용도로 남는다.
        return participant, config_sha256, trial

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.endswith("\n"):
                        raise JournalError(
                            f"{self.path}:{line_number}: 끝나지 않은 레코드가 있습니다."
                        )
                    if not line.strip():
                        raise JournalError(f"{self.path}:{line_number}: 빈 줄입니다.")
                    try:
                        record = json.loads(
                            line, object_pairs_hook=_reject_duplicate_json_keys
                        )
                    except (json.JSONDecodeError, ValueError) as error:
                        raise JournalError(
                            f"{self.path}:{line_number}: JSON이 손상됐습니다: {error}"
                        ) from error
                    if not isinstance(record, Mapping):
                        raise JournalError(f"{self.path}:{line_number}: JSON 객체가 아닙니다.")
                    key = self._key(record)
                    completed = self._is_completed(record)
                    terminal = self._is_terminal(record)
                    if key in self._terminal_keys:
                        raise JournalError(
                            f"기존 저널에 terminal 시행 뒤 중복 기록이 있습니다: {key}"
                        )
                    if completed:
                        self._completed_keys.add(key)
                    if terminal:
                        self._terminal_keys.add(key)
                    self._attempted_keys.add(key)
                    self._records.append(dict(record))
        except UnicodeDecodeError as error:
            raise JournalError(f"{self.path}: UTF-8 파일이 아닙니다.") from error

    @property
    def completed_keys(self) -> frozenset[tuple[str, str, int]]:
        return frozenset(self._completed_keys)

    @property
    def attempted_keys(self) -> frozenset[tuple[str, str, int]]:
        """중단을 포함해 저널에 한 번이라도 등장한 계획 시행 식별자."""

        return frozenset(self._attempted_keys)

    @property
    def terminal_keys(self) -> frozenset[tuple[str, str, int]]:
        """완료 또는 aborted main으로 다시 제시할 수 없는 시행 식별자."""

        return frozenset(self._terminal_keys)

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """Return defensive copies of the append-only records seen by this journal."""

        with self._lock:
            return tuple(dict(record) for record in self._records)

    def append(self, record: Mapping[str, Any]) -> None:
        if not isinstance(record, Mapping):
            raise JournalError("레코드는 JSON 객체여야 합니다.")
        key = self._key(record)
        completed = self._is_completed(record)
        terminal = self._is_terminal(record)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            if key in self._terminal_keys:
                raise JournalError(
                    f"terminal 시행 중복 기록을 추가할 수 없습니다: {key}"
                )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            if completed:
                self._completed_keys.add(key)
            if terminal:
                self._terminal_keys.add(key)
            self._attempted_keys.add(key)
            self._records.append(dict(record))


def _read_config(path: str | Path) -> tuple[dict[str, Any], str]:
    file_path = Path(path)
    raw = file_path.read_bytes()
    try:
        config = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CollectionBlockedError(f"설정 JSON을 읽을 수 없습니다: {file_path}: {error}") from error
    if not isinstance(config, dict):
        raise CollectionBlockedError("설정 최상위는 JSON 객체여야 합니다.")
    return config, hashlib.sha256(raw).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="동결 설정·픽셀 렌더러 게이트 검사 및 균형 시행표 생성(응답 GUI 미포함)"
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check-config", metavar="CONFIG_JSON")
    actions.add_argument("--make-schedule", metavar="CONFIG_JSON")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--git-commit")
    parser.add_argument("--output", help="시행표 JSON 출력 경로; 생략하면 표준 출력")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path = args.check_config or args.make_schedule
    try:
        config, digest = _read_config(path)
        if args.check_config:
            report = assess_experiment_gates(config)
            report["config_sha256"] = digest
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["config_gate_passed"] else 2

        assert_schedule_config(config)
        gates = assess_experiment_gates(config)
        rows = build_balanced_schedule(
            config,
            args.seed,
            config_sha256=digest,
            git_commit=args.git_commit,
        )
        payload = {
            "artifact_type": "balanced_schedule_only",
            "renderer_implemented": gates["renderer_implemented"],
            "renderer_verified": gates["renderer_verified"],
            "human_response_collection_implemented": gates[
                "human_response_collection_implemented"
            ],
            "human_data_collection_allowed": False,
            "blockers": gates["blockers"],
            "trial_count": len(rows),
            "config_sha256": digest,
            "git_commit": rows[0]["git_commit"] if rows else None,
            "random_seed": args.seed,
            "trials": rows,
        }
        rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 0
    except (CollectionBlockedError, OSError, ValueError) as error:
        print(f"차단됨: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CollectionBlockedError",
    "JournalError",
    "TrialJournal",
    "assess_experiment_gates",
    "assert_human_collection_allowed",
    "assert_schedule_config",
    "build_balanced_schedule",
    "canonical_journal_path",
    "expected_collection_session_id",
    "main",
]
