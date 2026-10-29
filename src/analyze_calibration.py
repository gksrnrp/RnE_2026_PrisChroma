"""동결된 3굵기 교정 JSONL에서 참가자별 심리측정 모형을 적합한다.

출력은 독립 held-out 계단을 전혀 사용하지 않은 ``calibration-only`` 중간
산출물이다. 따라서 이 모듈은 ``participant_thresholds`` 최종 계약을 만들지
않으며, 실제 또는 합성 입력이 없을 때 수치를 생성하는 경로도 제공하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

from calibration_design import (
    CALIBRATION_MODELS,
    CALIBRATION_TOTAL_TRIALS,
    CALIBRATION_WIDTH_TRIAL_COUNTS,
    REFERENCE_WIDTH_TRIALS,
    block_count_from_config,
    block_trial_count,
    trial_counts_from_config,
    width_block_trial_counts,
)
from color_spaces import srgb_to_lab
from data_contracts import ContractError, validate_payload
from distance import delta_e_2000
from experiment_design import (
    SZAFIR_LINE_MAX_VISUAL_ANGLE_DEG,
    SZAFIR_LINE_MIN_VISUAL_ANGLE_DEG,
    geometric_validation_widths,
    pixels_to_visual_angle_deg,
    szafir_relative_scale,
)
from experiment_runner import (
    CollectionBlockedError,
    assert_schedule_config,
    canonical_journal_path,
)
from implementation_provenance import (
    ImplementationProvenanceError,
    implementation_hashes,
    verify_implementation_hashes,
)
from machado import simulate_color_float
from psychometrics import (
    BootstrapResult,
    FitResult,
    _bootstrap_interval,
    fit_b192,
    fit_b384,
    fit_c192,
    fit_d384,
    parametric_bootstrap,
)
from stimulus_renderer import (
    RendererValidationError,
    verify_renderer_environment,
    verify_stimulus_artifact,
)
from validation_psychometrics import compare_shared_vs_free_w_aic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUMAN_CONFIG_SCHEMA = PROJECT_ROOT / "schemas" / "human" / "experiment_config.schema.json"
HUMAN_TRIAL_SCHEMA = PROJECT_ROOT / "schemas" / "human" / "trial.schema.json"
OUTPUT_SCHEMA = PROJECT_ROOT / "schemas" / "calibration_model_fits.schema.json"

MODELS = CALIBRATION_MODELS
PRIMARY_BOOTSTRAP_PAIRING_ID = "B192-C192-reference-width-parametric-v1"
CONFIDENCE = 0.95
MINIMUM_BOOTSTRAP_CONVERGENCE_RATE = 0.90
MINIMUM_HUMAN_BOOTSTRAP_REPLICATES = 1000
_PLACEHOLDER_RE = re.compile(
    r"(?:example|placeholder|not[_ -]?frozen|tbd|todo|unknown|미정|예시)",
    re.IGNORECASE,
)


class AnalysisBlockedError(ValueError):
    """입력 계약 또는 사전 고정 조건 때문에 분석을 시작할 수 없을 때 사용한다."""


def _json_path(parts: Sequence[object]) -> str:
    result = "$"
    for part in parts:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisBlockedError(f"JSON 객체에 중복 키가 있습니다: {key}")
        result[key] = value
    return result


def _decode_json(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AnalysisBlockedError(f"{label}: UTF-8 파일이 아닙니다.") from error
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise AnalysisBlockedError(f"{label}: JSON 형식이 깨졌습니다: {error}") from error


def _read_json_object(path: str | Path) -> tuple[dict[str, Any], str]:
    file_path = Path(path)
    try:
        raw = file_path.read_bytes()
    except FileNotFoundError as error:
        raise AnalysisBlockedError(f"설정 파일이 없습니다: {file_path}") from error
    payload = _decode_json(raw, label=str(file_path))
    if not isinstance(payload, dict):
        raise AnalysisBlockedError("설정 JSON 최상위는 객체여야 합니다.")
    return payload, hashlib.sha256(raw).hexdigest()


def _read_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    file_path = Path(path)
    try:
        raw = file_path.read_bytes()
    except FileNotFoundError as error:
        raise AnalysisBlockedError(f"시행 파일이 없습니다: {file_path}") from error
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AnalysisBlockedError(f"{file_path}: UTF-8 파일이 아닙니다.") from error
    if not text:
        raise AnalysisBlockedError(f"JSONL 파일이 비었습니다: {file_path}")
    if not text.endswith("\n"):
        raise AnalysisBlockedError(f"{file_path}: 마지막 JSONL 레코드가 개행으로 끝나지 않습니다.")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise AnalysisBlockedError(f"{file_path}:{line_number}: 빈 줄은 허용하지 않습니다.")
        try:
            record = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as error:
            raise AnalysisBlockedError(
                f"{file_path}:{line_number}: JSON 형식이 깨졌습니다: {error}"
            ) from error
        if not isinstance(record, dict):
            raise AnalysisBlockedError(f"{file_path}:{line_number}: JSON 객체가 아닙니다.")
        records.append(record)
    return records, hashlib.sha256(raw).hexdigest()


def _read_schema(path: Path) -> dict[str, Any]:
    try:
        schema = _decode_json(path.read_bytes(), label=str(path))
    except FileNotFoundError as error:
        raise AnalysisBlockedError(f"필요한 스키마 파일이 없습니다: {path}") from error
    if not isinstance(schema, dict):
        raise AnalysisBlockedError(f"스키마 최상위가 객체가 아닙니다: {path}")
    Draft202012Validator.check_schema(schema)
    return schema


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _safe_manifest_artifact(base: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise AnalysisBlockedError(f"{label}는 manifest 기준 상대 경로여야 합니다.")
    target = (base / relative).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError as error:
        raise AnalysisBlockedError(f"{label}가 manifest 폴더 밖을 가리킵니다.") from error
    return target


def _validate_schema(payload: Any, path: Path, *, label: str) -> None:
    validator = Draft202012Validator(
        _read_schema(path), format_checker=FormatChecker()
    )
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    if errors:
        details = "\n".join(
            f"- {_json_path(list(error.path))}: {error.message}" for error in errors
        )
        raise AnalysisBlockedError(f"{label} 스키마 오류:\n{details}")


def _close(actual: float, expected: float) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1e-5, abs_tol=1e-6)


def _require_close(actual: float, expected: float, message: str) -> None:
    if not _close(actual, expected):
        raise AnalysisBlockedError(f"{message}: 저장={actual}, 계산={expected}")


def _validate_trial_semantics(record: Mapping[str, Any], *, label: str) -> None:
    """human 1.0 시행에도 draft와 같은 수치 재계산 검사를 적용한다."""
    displayed_lab_a = srgb_to_lab(record["displayed_rgb_a"])
    displayed_lab_b = srgb_to_lab(record["displayed_rgb_b"])
    simulated_rgb_a = simulate_color_float(
        record["displayed_rgb_a"], record["cvd_type"], record["machado_severity"]
    )
    simulated_rgb_b = simulate_color_float(
        record["displayed_rgb_b"], record["cvd_type"], record["machado_severity"]
    )
    simulated_lab_a = srgb_to_lab(simulated_rgb_a)
    simulated_lab_b = srgb_to_lab(simulated_rgb_b)

    for field, stored, calculated in (
        ("displayed_lab_a", record["displayed_lab_a"], displayed_lab_a),
        ("displayed_lab_b", record["displayed_lab_b"], displayed_lab_b),
        ("simulated_rgb_a", record["simulated_rgb_a"], simulated_rgb_a),
        ("simulated_rgb_b", record["simulated_rgb_b"], simulated_rgb_b),
        ("simulated_lab_a", record["simulated_lab_a"], simulated_lab_a),
        ("simulated_lab_b", record["simulated_lab_b"], simulated_lab_b),
    ):
        if not all(_close(float(a), float(b)) for a, b in zip(stored, calculated)):
            raise AnalysisBlockedError(f"{label}: {field}가 표시 RGB에서 재계산한 값과 다릅니다.")

    original = float(delta_e_2000(displayed_lab_a, displayed_lab_b))
    simulated = float(delta_e_2000(simulated_lab_a, simulated_lab_b))
    _require_close(
        float(record["delta_e00_original"]),
        original,
        f"{label}: delta_e00_original 불일치",
    )
    _require_close(
        float(record["delta_e00_simulated"]),
        simulated,
        f"{label}: delta_e00_simulated 불일치",
    )

    positions = record["position_colors"]
    odd = record["odd_color"]
    other = "b" if odd == "a" else "a"
    if positions.count(odd) != 1 or positions.count(other) != 3:
        raise AnalysisBlockedError(f"{label}: position_colors의 odd/base 개수가 잘못됐습니다.")
    if positions[int(record["correct_position"]) - 1] != odd:
        raise AnalysisBlockedError(f"{label}: correct_position과 odd_color가 다릅니다.")
    expected_rgbs = [
        record["displayed_rgb_a"] if color == "a" else record["displayed_rgb_b"]
        for color in positions
    ]
    if record["position_rgbs"] != expected_rgbs:
        raise AnalysisBlockedError(f"{label}: position_rgbs가 position_colors와 다릅니다.")
    if record["phase"] == "validation":
        validation_id = record.get("validation_block_id", record.get("staircase_id"))
        if validation_id is None:
            raise AnalysisBlockedError(f"{label}: validation 시행에는 block ID가 필요합니다.")
    if record["trial_status"] == "completed":
        if any(record[field] is None for field in ("response_position", "is_correct", "response_time_ms")):
            raise AnalysisBlockedError(f"{label}: 완료 시행의 응답 필드가 비었습니다.")
        expected_correct = record["response_position"] == record["correct_position"]
        if record["is_correct"] != expected_correct:
            raise AnalysisBlockedError(f"{label}: is_correct가 실제 응답 위치와 다릅니다.")
    elif any(record[field] is not None for field in ("response_position", "is_correct", "response_time_ms")):
        raise AnalysisBlockedError(f"{label}: 중단 시행의 응답 필드는 모두 null이어야 합니다.")


def _validate_config(config: dict[str, Any], *, data_origin: str) -> None:
    if data_origin == "synthetic":
        try:
            validate_payload("experiment_config", config)
        except ContractError as error:
            raise AnalysisBlockedError(str(error)) from error
    elif data_origin == "human":
        _validate_schema(config, HUMAN_CONFIG_SCHEMA, label="human experiment_config")
        try:
            assert_schedule_config(config)
        except CollectionBlockedError as error:
            raise AnalysisBlockedError(f"사람 자료 동결 설정 검사 실패: {error}") from error
    else:
        raise AnalysisBlockedError(f"지원하지 않는 data_origin입니다: {data_origin!r}")

    model_plan = config.get("model_plan")
    if not isinstance(model_plan, Mapping):
        raise AnalysisBlockedError("config.model_plan이 없습니다.")
    if set(model_plan.get("models", ())) != set(MODELS) or len(model_plan.get("models", ())) != 4:
        raise AnalysisBlockedError("config.model_plan.models에는 B192/B384/C192/D384이 각각 한 번 필요합니다.")
    if model_plan.get("include_szafir_transfer") is not True:
        raise AnalysisBlockedError("C192 분석에는 include_szafir_transfer=true가 필요합니다.")

    staircase = config.get("staircase")
    profile = config.get("observer_profile")
    if not isinstance(profile, Mapping):
        raise AnalysisBlockedError("config.observer_profile이 필요합니다.")
    # 이전 synthetic draft에는 staircase에도 lapse가 중복 저장된다. 있으면 두
    # 위치의 고정값을 반드시 대조하되, fixed-level human 1.0에는 이 절이 없다.
    if staircase is not None:
        if not isinstance(staircase, Mapping) or staircase.get("lapse_policy") != "fixed":
            raise AnalysisBlockedError("분석 전에 lapse_policy를 fixed로 동결해야 합니다.")
        if staircase.get("fixed_lapse") is None:
            raise AnalysisBlockedError("staircase.fixed_lapse가 없습니다.")
        _require_close(
            float(staircase["fixed_lapse"]),
            float(profile["fixed_lapse"]),
            "staircase와 observer_profile의 fixed_lapse 불일치",
        )
    lapse = float(profile["fixed_lapse"])
    if not math.isfinite(lapse) or not 0.0 <= lapse < 0.20:
        raise AnalysisBlockedError("observer_profile.fixed_lapse는 0 이상 0.20 미만이어야 합니다.")
    for field in ("lapse_source", "lapse_session_id"):
        value = profile.get(field)
        if not isinstance(value, str) or not value.strip():
            raise AnalysisBlockedError(f"observer_profile.{field}가 비었습니다.")
        if data_origin == "human" and _PLACEHOLDER_RE.search(value):
            raise AnalysisBlockedError(f"사람 자료의 observer_profile.{field}가 미확정 값입니다.")

    calibration = config.get("calibration")
    if not isinstance(calibration, Mapping):
        raise AnalysisBlockedError("config.calibration이 필요합니다.")
    try:
        trial_counts_from_config(config)
        block_count_from_config(config)
    except ValueError as error:
        raise AnalysisBlockedError(str(error)) from error
    widths = calibration.get("widths")
    if not isinstance(widths, list) or len(widths) != 3:
        raise AnalysisBlockedError("config.calibration.widths에는 굵기 세 개가 필요합니다.")

    display = config["display"]
    conversion = {
        "display_width_mm": display["width_mm"],
        "horizontal_resolution_px": display["resolution_width_px"],
        "viewing_distance_mm": display["viewing_distance_mm"],
    }
    calibration_px = [int(row["width_px"]) for row in widths]
    if calibration_px != sorted(calibration_px) or len(set(calibration_px)) != 3:
        raise AnalysisBlockedError("교정 굵기 세 개는 중복 없이 오름차순이어야 합니다.")
    for row in widths:
        calculated = pixels_to_visual_angle_deg(row["width_px"], **conversion)
        _require_close(row["width_deg"], calculated, f"{row['width_px']}px 시각도 불일치")

    candidates = config.get("validation", {}).get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise AnalysisBlockedError("held-out 예측에는 사전 동결된 검증 후보 두 개가 필요합니다.")
    geometric_targets = geometric_validation_widths(calibration_px)
    candidate_px: list[int] = []
    for index, row in enumerate(candidates):
        rendered = int(row["rendered_width_px"])
        candidate_px.append(rendered)
        if rendered in calibration_px or not calibration_px[index] < rendered < calibration_px[index + 1]:
            raise AnalysisBlockedError(f"검증 후보 {index + 1}가 대응 교정 구간 내부의 미사용 px가 아닙니다.")
        _require_close(
            row["geometric_target_px"],
            geometric_targets[index],
            f"검증 후보 {index + 1} 기하평균 불일치",
        )
        calculated = pixels_to_visual_angle_deg(rendered, **conversion)
        _require_close(row["width_deg"], calculated, f"검증 후보 {rendered}px 시각도 불일치")
    if len(set(candidate_px)) != 2:
        raise AnalysisBlockedError("검증 후보 두 개는 서로 다른 정수 px여야 합니다.")

    all_angles = [float(row["width_deg"]) for row in widths + candidates]
    for angle in all_angles:
        if not SZAFIR_LINE_MIN_VISUAL_ANGLE_DEG <= angle <= SZAFIR_LINE_MAX_VISUAL_ANGLE_DEG:
            raise AnalysisBlockedError(
                f"C192 굵기 {angle:g}도는 Szafir 측정 범위 "
                f"{SZAFIR_LINE_MIN_VISUAL_ANGLE_DEG:g}~"
                f"{SZAFIR_LINE_MAX_VISUAL_ANGLE_DEG:g}도 밖입니다. 외삽하지 않습니다."
            )


def _validate_trials(
    records: list[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    data_origin: str,
) -> None:
    completed_keys: set[tuple[str, str, int]] = set()
    session_settings: dict[tuple[str, str], tuple[Any, ...]] = {}
    profile = config["observer_profile"]
    expected_fields = {
        "participant_id": profile["participant_id"],
        "cvd_type": profile["cvd_type"],
        "machado_severity": profile["severity"],
        "severity_source": profile["severity_source"],
        "severity_session_id": profile["severity_session_id"],
        "fixed_lapse": profile["fixed_lapse"],
        "lapse_source": profile["lapse_source"],
        "lapse_session_id": profile["lapse_session_id"],
    }

    for index, record in enumerate(records, start=1):
        label = f"trial JSONL {index}행"
        if record.get("data_origin") != data_origin:
            raise AnalysisBlockedError(f"{label}: data_origin이 파일 안에서 바뀌었습니다.")
        if data_origin == "synthetic":
            try:
                validate_payload("trial", record)
            except ContractError as error:
                raise AnalysisBlockedError(f"{label}: {error}") from error
        else:
            _validate_schema(record, HUMAN_TRIAL_SCHEMA, label=label)
            _validate_trial_semantics(record, label=label)

        if record["config_sha256"] != config_sha256:
            raise AnalysisBlockedError(
                f"{label}: config_sha256이 설정 파일 원문 SHA-256과 다릅니다."
            )
        if record["config_version"] != config["config_version"]:
            raise AnalysisBlockedError(f"{label}: config_version이 설정과 다릅니다.")
        if record["stimulus_family_id"] != config["color_design"]["stimulus_family_id"]:
            raise AnalysisBlockedError(f"{label}: stimulus_family_id가 동결 설정과 다릅니다.")
        if record["renderer_id"] != config["line"]["renderer"]:
            raise AnalysisBlockedError(f"{label}: renderer_id가 동결 설정과 다릅니다.")
        for field, expected in expected_fields.items():
            actual = record[field]
            equal = _close(actual, expected) if field in {"machado_severity", "fixed_lapse"} else actual == expected
            if not equal:
                raise AnalysisBlockedError(f"{label}: {field}가 동결 observer_profile과 다릅니다.")

        key = (
            record["participant_id"],
            record["config_sha256"],
            record["trial_id"],
        )
        # append-only 원자료에는 같은 계획 시행의 중단 시도 뒤 완료 재시도가
        # 함께 남을 수 있다. 완료 전 중단은 감사 기록으로 허용하지만, 완료한
        # 뒤의 추가 기록이나 두 번째 완료는 허용하지 않는다.
        if key in completed_keys:
            raise AnalysisBlockedError(
                "JSONL에 같은 participant/config/trial_id의 완료 시행 뒤 중복 기록이 있습니다: "
                f"{key}"
            )
        if record["trial_status"] == "completed":
            completed_keys.add(key)
        session_key = (record["participant_id"], record["session_id"])
        settings = (
            record["config_version"],
            record["config_sha256"],
            record["git_commit"],
            record["stimulus_family_id"],
            record["renderer_id"],
            *(record[field] for field in expected_fields if field != "participant_id"),
        )
        previous = session_settings.setdefault(session_key, settings)
        if previous != settings:
            raise AnalysisBlockedError(f"{session_key}: 한 세션 안에서 설정 또는 observer provenance가 바뀌었습니다.")


def _validate_calibration_manifest(
    manifest_path: str | Path,
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    records: Sequence[Mapping[str, Any]],
    require_record_manifest_hash: bool,
) -> str:
    """교정 응답 전체를 실제 compiled PNG manifest와 다시 결박한다.

    PNG는 원문 SHA와 해독 RGB SHA를 모두 대조한다. sidecar는 현재 렌더러로
    다시 읽어 config와 해당 시행에서 완전히 재구성한 픽셀 명세와 대조하며,
    manifest의 ``pixel_metadata_sha256``로 sidecar 원문 SHA까지 대조한다. 중단 후
    재시도한 중복 trial_id도 보존·검증하며, 완료 시행 수는 적합 단계에서 검사한다.
    """

    manifest_file = Path(manifest_path).resolve()
    manifest, manifest_sha256 = _read_json_object(manifest_file)
    git_commits = {row.get("git_commit") for row in records}
    if len(git_commits) != 1:
        raise AnalysisBlockedError("교정 응답의 git_commit이 하나로 고정되지 않았습니다.")
    expected_headers = {
        "schema_version": "1.0",
        "artifact_type": "compiled_pixel_stimulus_manifest",
        "config_version": config["config_version"],
        "config_canonical_sha256": _canonical_sha256(config),
        "schedule_config_sha256": config_sha256,
        "git_commit": next(iter(git_commits)),
        "stimulus_family_id": config["color_design"]["stimulus_family_id"],
        "participant_id": config["observer_profile"]["participant_id"],
        "cvd_type": config["observer_profile"]["cvd_type"],
        "machado_severity": config["observer_profile"]["severity"],
        "trial_count": CALIBRATION_TOTAL_TRIALS,
        "png_files_written": True,
        "human_response_collection_implemented": True,
    }
    for field, expected in expected_headers.items():
        if manifest.get(field) != expected:
            raise AnalysisBlockedError(
                f"calibration manifest.{field}가 동결 설정/응답과 다릅니다."
            )
    generation = manifest.get("color_generation")
    if not isinstance(generation, Mapping):
        raise AnalysisBlockedError("calibration manifest.color_generation이 없습니다.")
    color_design = config["color_design"]
    expected_generation = {
        "gamut_policy": color_design["gamut_policy"],
        "rgb_quantization": color_design["rgb_quantization"],
        "target_absolute_tolerance_delta_e00": color_design[
            "target_absolute_tolerance_delta_e00"
        ],
        "target_relative_tolerance": color_design["target_relative_tolerance"],
        "allowed_error_rule": "max(absolute_tolerance, relative_tolerance * target)",
        "search_grid_steps": color_design["search_grid_steps"],
        "unique_target_count": 12,
        "configured_target_count": 12,
        "development_partial_schedule": False,
    }
    for field, expected in expected_generation.items():
        if generation.get(field) != expected:
            raise AnalysisBlockedError(
                f"calibration manifest.color_generation.{field}가 config와 다릅니다."
            )

    try:
        current_renderer = verify_renderer_environment(config)
    except (RendererValidationError, OSError, TypeError, ValueError) as error:
        raise AnalysisBlockedError(f"현재 픽셀 렌더러 환경 검증 실패: {error}") from error
    manifest_renderer = manifest.get("renderer")
    if not isinstance(manifest_renderer, Mapping):
        raise AnalysisBlockedError("calibration manifest.renderer가 없습니다.")
    for field in (
        "renderer_verified",
        "renderer_id",
        "implementation_version",
        "implementation_sha256",
        "pillow_version",
    ):
        if manifest_renderer.get(field) != current_renderer.get(field):
            raise AnalysisBlockedError(
                f"calibration manifest.renderer.{field}가 현재 렌더러와 다릅니다."
            )

    manifest_rows = manifest.get("trials")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != CALIBRATION_TOTAL_TRIALS:
        raise AnalysisBlockedError("calibration manifest는 정확히 384개 시행이어야 합니다.")
    by_id: dict[int, Mapping[str, Any]] = {}
    for index, row in enumerate(manifest_rows):
        if (
            not isinstance(row, Mapping)
            or isinstance(row.get("trial_id"), bool)
            or not isinstance(row.get("trial_id"), int)
        ):
            raise AnalysisBlockedError(
                f"calibration manifest.trials[{index}] 형식이 잘못됐습니다."
            )
        trial_id = int(row["trial_id"])
        if trial_id in by_id:
            raise AnalysisBlockedError(
                f"calibration manifest trial_id가 중복됩니다: {trial_id}"
            )
        by_id[trial_id] = row
    if set(by_id) != set(range(1, CALIBRATION_TOTAL_TRIALS + 1)):
        raise AnalysisBlockedError("calibration manifest trial_id는 정확히 1~384여야 합니다.")

    color_pairs = manifest.get("color_pairs")
    levels = list(config["sampling"]["delta_e00_levels"])
    if not isinstance(color_pairs, list) or len(color_pairs) != 12 or len(levels) != 12:
        raise AnalysisBlockedError(
            "calibration manifest에는 config 순서의 color pair 12개가 필요합니다."
        )
    pair_fields = (
        "displayed_rgb_a",
        "displayed_rgb_b",
        "displayed_lab_a",
        "displayed_lab_b",
        "simulated_rgb_a",
        "simulated_rgb_b",
        "simulated_lab_a",
        "simulated_lab_b",
        "delta_e00_original",
        "delta_e00_simulated",
    )
    previous_actual = -math.inf
    for level_index, (target, pair) in enumerate(zip(levels, color_pairs), start=1):
        if (
            not isinstance(pair, Mapping)
            or pair.get("target_delta_e00_simulated") != target
        ):
            raise AnalysisBlockedError(
                "calibration manifest color pair 순서/target가 config와 다릅니다."
            )
        actual = pair.get("delta_e00_simulated")
        if (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not math.isfinite(float(actual))
        ):
            raise AnalysisBlockedError(
                "calibration manifest color pair 실제 ΔE00가 유한한 수가 아닙니다."
            )
        actual = float(actual)
        tolerance = max(
            float(color_design["target_absolute_tolerance_delta_e00"]),
            float(color_design["target_relative_tolerance"]) * float(target),
        )
        if abs(actual - float(target)) > tolerance + 1e-12:
            raise AnalysisBlockedError(
                "calibration manifest color pair가 config target 허용오차를 넘습니다."
            )
        if actual <= previous_actual:
            raise AnalysisBlockedError(
                "calibration manifest color pair 실제 ΔE00가 엄격히 증가하지 않습니다."
            )
        previous_actual = actual
        matching_rows = [
            row
            for row in manifest_rows
            if row.get("delta_e00_level_index") == level_index
        ]
        if len(matching_rows) != 32:
            raise AnalysisBlockedError(
                f"calibration manifest level {level_index} 시행은 정확히 32개여야 합니다."
            )
        for row in matching_rows:
            for field in pair_fields:
                if row.get(field) != pair.get(field):
                    raise AnalysisBlockedError(
                        f"calibration manifest level {level_index}의 {field}가 "
                        "color pair와 다릅니다."
                    )

    immutable_fields = (
        "config_version",
        "config_sha256",
        "git_commit",
        "stimulus_family_id",
        "participant_id",
        "session_block",
        "trial_id",
        "width_index",
        "stroke_width_px",
        "stroke_width_deg",
        "delta_e00_level_index",
        "target_delta_e00_simulated",
        "displayed_rgb_a",
        "displayed_rgb_b",
        "displayed_lab_a",
        "displayed_lab_b",
        "simulated_rgb_a",
        "simulated_rgb_b",
        "simulated_lab_a",
        "simulated_lab_b",
        "delta_e00_original",
        "delta_e00_simulated",
        "correct_position",
        "position_colors",
        "odd_color",
        "cvd_type",
        "machado_severity",
        "severity_source",
        "severity_session_id",
        "fixed_lapse",
        "lapse_source",
        "lapse_session_id",
        "random_seed",
    )
    base = manifest_file.parent
    response_ids: set[int] = set()
    for record in records:
        trial_id = int(record["trial_id"])
        expected = by_id.get(trial_id)
        if expected is None:
            raise AnalysisBlockedError(
                f"trial {trial_id}: calibration manifest에 없는 시행입니다."
            )
        response_ids.add(trial_id)
        stored_manifest_sha256 = record.get("stimulus_manifest_sha256")
        if require_record_manifest_hash and stored_manifest_sha256 != manifest_sha256:
            raise AnalysisBlockedError(
                f"trial {trial_id}: stimulus_manifest_sha256이 실제 manifest 원문과 다릅니다."
            )
        if stored_manifest_sha256 is not None and stored_manifest_sha256 != manifest_sha256:
            raise AnalysisBlockedError(
                f"trial {trial_id}: stimulus_manifest_sha256이 실제 manifest 원문과 다릅니다."
            )
        if record.get("phase") != "calibration":
            raise AnalysisBlockedError(f"trial {trial_id}: calibration 응답이 아닙니다.")
        for field in immutable_fields:
            if record.get(field) != expected.get(field):
                raise AnalysisBlockedError(
                    f"trial {trial_id}: {field}가 calibration manifest와 다릅니다."
                )

        artifact = expected.get("raster_artifact")
        if not isinstance(artifact, Mapping):
            raise AnalysisBlockedError(f"trial {trial_id}: raster_artifact가 없습니다.")
        png = _safe_manifest_artifact(base, artifact.get("png"), f"trial {trial_id} PNG")
        sidecar = _safe_manifest_artifact(
            base, artifact.get("pixel_metadata"), f"trial {trial_id} sidecar"
        )
        try:
            verification = verify_stimulus_artifact(
                png, sidecar, config=config, trial=expected
            )
        except (RendererValidationError, OSError, TypeError, ValueError) as error:
            raise AnalysisBlockedError(
                f"trial {trial_id}: PNG/sidecar 재검증 실패: {error}"
            ) from error
        if (
            verification["png_sha256"] != artifact.get("png_sha256")
            or verification["decoded_rgb_sha256"]
            != artifact.get("decoded_rgb_sha256")
            or record.get("stimulus_png_sha256") != verification["png_sha256"]
            or record.get("stimulus_decoded_rgb_sha256")
            != verification["decoded_rgb_sha256"]
        ):
            raise AnalysisBlockedError(
                f"trial {trial_id}: 응답·manifest·PNG 픽셀 hash가 서로 다릅니다."
            )
        try:
            sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        except OSError as error:
            raise AnalysisBlockedError(
                f"trial {trial_id}: sidecar 원문 hash를 계산할 수 없습니다: {error}"
            ) from error
        expected_sidecar_sha256 = artifact.get("pixel_metadata_sha256")
        if (
            not isinstance(expected_sidecar_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sidecar_sha256) is None
        ):
            raise AnalysisBlockedError(
                f"trial {trial_id}: calibration manifest에 sidecar raw hash가 없습니다."
            )
        if sidecar_sha256 != expected_sidecar_sha256:
            raise AnalysisBlockedError(
                f"trial {trial_id}: sidecar raw hash가 calibration manifest와 다릅니다."
            )
        try:
            repository_path = sidecar.resolve().relative_to(
                PROJECT_ROOT.resolve()
            ).as_posix()
        except ValueError as error:
            raise AnalysisBlockedError(
                f"trial {trial_id}: 사람 자극 sidecar가 저장소 밖에 있습니다."
            ) from error
        if record.get("stimulus_pixel_metadata_path") != repository_path:
            raise AnalysisBlockedError(
                f"trial {trial_id}: sidecar 경로가 실제 artifact와 다릅니다."
            )
    if set(by_id) != response_ids:
        raise AnalysisBlockedError("응답과 calibration manifest의 시행 집합이 다릅니다.")
    return manifest_sha256


def _participant_calibration(
    records: list[dict[str, Any]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    participant_id = config["observer_profile"]["participant_id"]
    calibration_records = [row for row in records if row["phase"] == "calibration"]
    completed = [row for row in calibration_records if row["trial_status"] == "completed"]
    aborted = len(calibration_records) - len(completed)
    ignored = len(records) - len(calibration_records)
    widths = config["calibration"]["widths"]
    expected = {int(row["width_px"]): float(row["width_deg"]) for row in widths}
    expected_width_counts = {
        int(row["width_px"]): count
        for row, count in zip(widths, CALIBRATION_WIDTH_TRIAL_COUNTS)
    }
    counts = Counter(int(row["stroke_width_px"]) for row in completed)
    if counts != Counter(expected_width_counts):
        raise AnalysisBlockedError(
            f"{participant_id}: 완료 교정 시행은 굵기 순서대로 96/192/96회여야 합니다. "
            f"관측={dict(sorted(counts.items()))}"
        )
    if {int(row["trial_id"]) for row in completed} != set(
        range(1, CALIBRATION_TOTAL_TRIALS + 1)
    ):
        raise AnalysisBlockedError(
            f"{participant_id}: 완료 calibration trial_id는 정확히 1~384여야 합니다."
        )
    block_count = block_count_from_config(config)
    trials_per_block = block_trial_count(config)
    block_counts = Counter(int(row["session_block"]) for row in completed)
    if block_counts != Counter(
        {block: trials_per_block for block in range(1, block_count + 1)}
    ):
        raise AnalysisBlockedError(
            f"{participant_id}: 완료 교정 시행의 block 균형이 잘못됐습니다. "
            f"관측={dict(sorted(block_counts.items()))}"
        )
    block_sessions = {
        block: {row["session_id"] for row in completed if int(row["session_block"]) == block}
        for block in range(1, block_count + 1)
    }
    if any(len(sessions) != 1 for sessions in block_sessions.values()):
        raise AnalysisBlockedError(
            f"{participant_id}: 각 calibration block의 완료 시행은 정확히 한 session_id에 속해야 합니다."
        )
    completed_session_ids = {next(iter(sessions)) for sessions in block_sessions.values()}
    if len(completed_session_ids) != block_count:
        raise AnalysisBlockedError(
            f"{participant_id}: calibration block은 서로 다른 session_id여야 합니다."
        )
    width_block_counts = Counter(
        (int(row["stroke_width_px"]), int(row["session_block"]))
        for row in completed
    )
    expected_per_width_block = width_block_trial_counts(config)
    expected_width_block_counts = Counter(
        {
            (int(widths[index]["width_px"]), block): expected_per_width_block[index]
            for index in range(3)
            for block in range(1, block_count + 1)
        }
    )
    if width_block_counts != expected_width_block_counts:
        raise AnalysisBlockedError(
            f"{participant_id}: 각 굵기×session_block 시행 수가 동결 설계와 다릅니다."
        )
    level_counts = Counter(
        (int(row["stroke_width_px"]), int(row["delta_e00_level_index"]))
        for row in completed
    )
    if set(level_counts) != {
        (width_px, level) for width_px in expected for level in range(1, 13)
    } or {
        width_px: {count for (cell_width, _), count in level_counts.items() if cell_width == width_px}
        for width_px in expected
    } != {
        int(widths[index]["width_px"]): {CALIBRATION_WIDTH_TRIAL_COUNTS[index] // 12}
        for index in range(3)
    }:
        raise AnalysisBlockedError(
            f"{participant_id}: 각 굵기×12 ΔE 수준 시행 수가 8/16/8이 아닙니다."
        )
    balanced_cells = Counter(
        (
            int(row["stroke_width_px"]),
            int(row["delta_e00_level_index"]),
            int(row["correct_position"]),
            str(row["odd_color"]),
        )
        for row in completed
    )
    expected_cells = {
        (width_px, level, position, odd)
        for width_px in expected
        for level in range(1, 13)
        for position in range(1, 5)
        for odd in ("a", "b")
    }
    expected_cell_counts = {
        int(widths[index]["width_px"]): CALIBRATION_WIDTH_TRIAL_COUNTS[index] // 96
        for index in range(3)
    }
    if set(balanced_cells) != expected_cells or any(
        count != expected_cell_counts[cell[0]]
        for cell, count in balanced_cells.items()
    ):
        raise AnalysisBlockedError(
            f"{participant_id}: 굵기×수준×정답위치×odd A/B 완료 셀이 완전 균형이 아닙니다."
        )
    for row in calibration_records:
        width_px = int(row["stroke_width_px"])
        if width_px not in expected:
            raise AnalysisBlockedError(f"{participant_id}: config에 없는 교정 굵기 {width_px}px가 있습니다.")
        _require_close(
            row["stroke_width_deg"],
            expected[width_px],
            f"{participant_id}: {width_px}px 시행 시각도 불일치",
        )
        width_index = int(row["width_index"])
        expected_row = widths[width_index - 1]
        if int(expected_row["width_px"]) != width_px:
            raise AnalysisBlockedError(
                f"{participant_id}: width_index와 stroke_width_px가 동결 설정과 다릅니다."
            )
        level_index = int(row["delta_e00_level_index"])
        expected_target = float(config["sampling"]["delta_e00_levels"][level_index - 1])
        _require_close(
            row["target_delta_e00_simulated"],
            expected_target,
            f"{participant_id}: delta_e00_level_index와 target 값 불일치",
        )
        absolute_tolerance = float(
            config["color_design"]["target_absolute_tolerance_delta_e00"]
        )
        relative_tolerance = float(config["color_design"]["target_relative_tolerance"])
        tolerance = max(absolute_tolerance, relative_tolerance * expected_target)
        if abs(float(row["delta_e00_simulated"]) - expected_target) > tolerance + 1e-12:
            raise AnalysisBlockedError(
                f"{participant_id}: 실제 표시 RGB의 Machado 후 ΔE00가 동결 target 허용오차를 넘습니다."
            )
    for row in completed:
        value = float(row["delta_e00_simulated"])
        if not math.isfinite(value) or value <= 0:
            raise AnalysisBlockedError(f"{participant_id}: 적합에 쓸 Machado 후 ΔE00은 0보다 커야 합니다.")

    calibration_sessions = {row["session_id"] for row in calibration_records}
    lapse_session = config["observer_profile"]["lapse_session_id"]
    if lapse_session in calibration_sessions:
        raise AnalysisBlockedError(
            f"{participant_id}: lapse_session_id는 교정 session_id와 분리돼야 합니다."
        )
    git_commits = {row["git_commit"] for row in records}
    stimulus_families = {row["stimulus_family_id"] for row in records}
    renderer_ids = {row["renderer_id"] for row in records}
    if len(git_commits) != 1:
        raise AnalysisBlockedError(f"{participant_id}: 입력 파일에 둘 이상의 git_commit이 있습니다.")
    if len(stimulus_families) != 1:
        raise AnalysisBlockedError(f"{participant_id}: 입력 파일에 둘 이상의 stimulus_family_id가 있습니다.")
    if len(renderer_ids) != 1:
        raise AnalysisBlockedError(f"{participant_id}: 입력 파일에 둘 이상의 renderer_id가 있습니다.")

    width_summary = [
        {
            "stroke_width_px": int(row["width_px"]),
            "stroke_width_deg": float(row["width_deg"]),
            "completed_trials": counts[int(row["width_px"])],
        }
        for row in widths
    ]
    session_block_quality = []
    for block in range(1, block_count + 1):
        block_rows = [
            row for row in completed if int(row["session_block"]) == block
        ]
        correct_trials = sum(bool(row["is_correct"]) for row in block_rows)
        session_block_quality.append(
            {
                "session_block": block,
                "session_id": next(iter(block_sessions[block])),
                "completed_trials": len(block_rows),
                "correct_trials": correct_trials,
                "accuracy": correct_trials / len(block_rows),
            }
        )
    reference = widths[1]
    summary = {
        "completed_trial_count": len(completed),
        "aborted_trial_count": aborted,
        "ignored_noncalibration_record_count": ignored,
        "widths": width_summary,
        "session_block_quality": session_block_quality,
        "session_block_accuracy_use": (
            "descriptive_quality_only_not_exclusion_or_inference"
        ),
        "reference_width_px": int(reference["width_px"]),
        "reference_width_deg": float(reference["width_deg"]),
        "git_commit": next(iter(git_commits)),
        "stimulus_family_id": next(iter(stimulus_families)),
        "renderer_id": next(iter(renderer_ids)),
    }
    return completed, summary


def _model_seed(base_seed: int, participant_id: str, model: str) -> int:
    # B192과 C192은 같은 기준 굵기 192시행을 사용하고 그 굵기에서는 같은
    # 심리측정 모형이다. 두 모델에 같은 난수열을 써야 같은 인덱스의
    # bootstrap 복제가 같은 모의 반응 자료에 대응하고, held-out 예측의
    # 손실 차이를 paired bootstrap으로 계산할 수 있다.
    seed_key = PRIMARY_BOOTSTRAP_PAIRING_ID if model in {"B192", "C192"} else model
    encoded = f"{base_seed}\0{participant_id}\0{seed_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def _bootstrap_payload(
    result: BootstrapResult, *, pairing_id: str | None
) -> dict[str, Any]:
    return {
        "seed": result.seed,
        "confidence": result.confidence,
        "interval_method": result.interval_method,
        "requested_replicates": result.requested_replicates,
        "successful_replicates": result.successful_replicates,
        "convergence_rate": result.convergence_rate,
        "converged": result.converged,
        "replicate_pairing_id": pairing_id,
        "parameter_intervals": [
            {
                "parameter": parameter,
                "ci_low": interval[0],
                "ci_high": interval[1],
            }
            for parameter, interval in sorted(result.intervals.items())
        ],
    }


def _bootstrap_c80_samples(
    result: BootstrapResult, stroke_width_deg: float
) -> np.ndarray:
    """특정 굵기의 C80 bootstrap 복제값을 원래 순서대로 변환한다."""
    width = float(stroke_width_deg)
    if result.model in {"B192", "B384"}:
        values = np.asarray(result.estimates.get("c80", ()), dtype=float)
    elif result.model == "C192":
        if result.relative_scale is None:
            raise AnalysisBlockedError("C192 bootstrap relative scale이 없습니다.")
        values = np.asarray(result.estimates.get("c80_ref", ()), dtype=float) * float(
            result.relative_scale(width)
        )
    elif result.model == "D384":
        if result.reference_width is None:
            raise AnalysisBlockedError("D384 bootstrap reference width가 없습니다.")
        values = np.asarray(result.estimates.get("c80_ref", ()), dtype=float) * (
            width / result.reference_width
        ) ** np.asarray(result.estimates.get("beta", ()), dtype=float)
    else:
        raise AnalysisBlockedError(f"지원하지 않는 bootstrap 모델입니다: {result.model}")
    if (
        values.ndim != 1
        or values.size != result.successful_replicates
        or np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise AnalysisBlockedError(f"{result.model}: C80 bootstrap 표본이 잘못됐습니다.")
    return values


def _prediction_payload(
    fit: FitResult,
    bootstrap: BootstrapResult,
    candidates: Sequence[Mapping[str, Any]],
    *,
    pairing_id: str | None,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for candidate in candidates:
        width = float(candidate["width_deg"])
        ci_low: float | None = None
        ci_high: float | None = None
        bootstrap_samples: list[float] = []
        jackknife_samples: list[float] = []
        if bootstrap.successful_replicates:
            ci_low, ci_high = bootstrap.c80_interval(width)
            bootstrap_samples = [
                float(value) for value in _bootstrap_c80_samples(bootstrap, width)
            ]
            if bootstrap.interval_method == "bca-log":
                jackknife_samples = [float(value) for value in (
                    np.asarray(bootstrap.jackknife_estimates["c80_ref"]) *
                    (width/bootstrap.reference_width)**np.asarray(bootstrap.jackknife_estimates["beta"])
                )]
        predictions.append(
            {
                "stroke_width_px": int(candidate["rendered_width_px"]),
                "stroke_width_deg": width,
                "c80_sim": float(fit.c80(width)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bootstrap_c80_samples": bootstrap_samples,
                "jackknife_c80_samples": jackknife_samples,
                "bootstrap_pairing_id": pairing_id,
                "value_type": "calibration_prediction",
                "observed_held_out_used": False,
            }
        )
    return predictions


def _training_range_checks(
    fit: FitResult,
    bootstrap: BootstrapResult,
    x: np.ndarray,
    widths: np.ndarray,
    checked_widths: Sequence[float],
) -> list[dict[str, Any]]:
    """Check that C80 and its 95% CI are identified inside displayed levels."""

    checks: list[dict[str, Any]] = []
    for width in checked_widths:
        mask = np.isclose(widths, float(width), rtol=1e-10, atol=1e-12)
        presented = x[mask]
        if presented.size == 0:
            raise AnalysisBlockedError(
                f"{fit.model}: {float(width):.12g}도에서 제시된 ΔE00가 없습니다."
            )
        low = float(np.min(presented))
        high = float(np.max(presented))
        point = float(fit.c80(float(width)))
        ci_low: float | None = None
        ci_high: float | None = None
        if bootstrap.successful_replicates >= 2:
            interval = bootstrap.c80_interval(float(width))
            ci_low, ci_high = float(interval[0]), float(interval[1])
        point_bracketed = low <= point <= high
        ci_bracketed = (
            ci_low is not None
            and ci_high is not None
            and low <= ci_low <= ci_high <= high
        )
        checks.append(
            {
                "stroke_width_deg": float(width),
                "presented_delta_e00_min": low,
                "presented_delta_e00_max": high,
                "c80_sim": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "point_bracketed": point_bracketed,
                "ci_bracketed": ci_bracketed,
                "range_bracketed": point_bracketed and ci_bracketed,
            }
        )
    return checks


def _fit_all_models(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    require_c80_bracketing: bool = False,
) -> list[dict[str, Any]]:
    x = np.asarray([float(row["delta_e00_simulated"]) for row in records])
    correct = np.asarray([bool(row["is_correct"]) for row in records])
    widths = np.asarray([float(row["stroke_width_deg"]) for row in records])
    reference = float(config["calibration"]["widths"][1]["width_deg"])
    lapse = float(config["observer_profile"]["fixed_lapse"])
    direction = tuple(float(value) for value in config["color_design"]["direction_lab"])
    probability = float(config["model_plan"]["szafir_probability_coordinate"])

    def scale(width: float) -> float:
        return szafir_relative_scale(
            direction,
            probability,
            width,
            reference,
            allow_extrapolation=False,
        )

    # 모든 교정·held-out 후보에서 미리 호출해 C192 외삽이나 식 오류를 적합 전에 막는다.
    prediction_widths = [
        float(row["width_deg"]) for row in config["validation"]["candidates"]
    ]
    for width in [*np.unique(widths).tolist(), *prediction_widths]:
        scale(float(width))

    fits = {
        "B192": fit_b192(x, correct, widths, reference_width=reference, lapse=lapse),
        "B384": fit_b384(x, correct, widths, lapse=lapse),
        "C192": fit_c192(
            x,
            correct,
            widths,
            reference_width=reference,
            relative_scale=scale,
            lapse=lapse,
        ),
        "D384": fit_d384(x, correct, widths, reference_width=reference, lapse=lapse),
    }
    reference_mask = np.isclose(widths, reference, rtol=1e-10, atol=1e-12)
    results: list[dict[str, Any]] = []
    participant_id = str(config["observer_profile"]["participant_id"])
    for model in MODELS:
        fit = fits[model]
        if model in {"B192", "C192"}:
            bootstrap_x = x[reference_mask]
            bootstrap_correct = correct[reference_mask]
            bootstrap_widths = widths[reference_mask]
        else:
            bootstrap_x, bootstrap_correct, bootstrap_widths = x, correct, widths
        bootstrap = parametric_bootstrap(
            fit,
            bootstrap_x,
            bootstrap_correct,
            bootstrap_widths,
            replicates=bootstrap_replicates,
            confidence=CONFIDENCE,
            seed=_model_seed(bootstrap_seed, participant_id, model),
            minimum_convergence_rate=MINIMUM_BOOTSTRAP_CONVERGENCE_RATE,
        )
        pairing_id = (
            PRIMARY_BOOTSTRAP_PAIRING_ID if model in {"B192", "C192"} else None
        )
        checked_widths = (
            [reference]
            if model in {"B192", "C192"}
            else [float(value) for value in np.unique(widths)]
        )
        range_checks = _training_range_checks(
            fit, bootstrap, x, widths, checked_widths
        )
        if require_c80_bracketing and any(
            not check["range_bracketed"] for check in range_checks
        ):
            failed = [
                f"{check['stroke_width_deg']:.6g}도 "
                f"(제시 {check['presented_delta_e00_min']:.6g}~"
                f"{check['presented_delta_e00_max']:.6g}, "
                f"C80 {check['c80_sim']:.6g}, "
                f"CI {check['ci_low']}~{check['ci_high']})"
                for check in range_checks
                if not check["range_bracketed"]
            ]
            raise AnalysisBlockedError(
                f"{model}: C80 점추정과 95% bootstrap CI가 실제 제시 ΔE00 "
                f"범위 안에서 식별되지 않았습니다: {'; '.join(failed)}"
            )
        results.append(
            {
                "model": model,
                "training_trial_count": fit.n_trials,
                "fit": {
                    "parameters": {key: float(value) for key, value in fit.parameters.items()},
                    "fixed_lapse": fit.lapse,
                    "negative_log_likelihood": fit.negative_log_likelihood,
                    "converged": fit.converged,
                    "iterations": fit.iterations,
                    "message": fit.message,
                },
                "bootstrap": _bootstrap_payload(bootstrap, pairing_id=pairing_id),
                "training_range_checks": range_checks,
                "held_out_candidate_predictions": _prediction_payload(
                    fit,
                    bootstrap,
                    config["validation"]["candidates"],
                    pairing_id=pairing_id,
                ),
            }
        )
    return results


def _calibration_width_slope_sensitivity(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit the shared-w assumption across the three calibration widths.

    The shared model has three width-specific C80 values and one common ``w``
    (k=4).  The sensitivity model has a separate C80 and ``w`` at each width
    (k=6).  This audit only gates interpretation of D384's beta; the same-data
    C192-vs-B192 primary comparison uses the reference-width trials alone.
    """

    x = np.asarray([float(row["delta_e00_simulated"]) for row in records])
    correct = np.asarray([bool(row["is_correct"]) for row in records])
    widths = np.asarray([float(row["stroke_width_deg"]) for row in records])
    unique_widths = np.asarray(sorted(set(widths.tolist())), dtype=float)
    if unique_widths.size != 3:
        raise AnalysisBlockedError(
            "교정 공통-w 민감도 분석에는 정확히 세 굵기가 필요합니다."
        )
    width_index = np.full(widths.shape, -1, dtype=int)
    for index, width in enumerate(unique_widths):
        width_index[np.isclose(widths, width, rtol=1e-10, atol=1e-12)] = index
    if np.any(width_index < 0):
        raise AnalysisBlockedError("교정 굵기를 공통-w 민감도 분석 block에 매핑하지 못했습니다.")

    comparison = compare_shared_vs_free_w_aic(
        x,
        correct,
        width_index,
        lapse=float(config["observer_profile"]["fixed_lapse"]),
    )
    d384_rows = [row for row in model_results if row.get("model") == "D384"]
    if len(d384_rows) != 1:
        raise AnalysisBlockedError("D384 결과가 정확히 하나여야 합니다.")
    d384_ready = bool(
        d384_rows[0]["fit"]["converged"]
        and d384_rows[0]["bootstrap"]["converged"]
    )
    if not comparison["comparison_resolved"]:
        eligible = False
        reason = "shared_free_w_comparison_unresolved"
    elif comparison["free_w_preferred"]:
        eligible = False
        reason = "free_w_lower_aic"
    elif not d384_ready:
        eligible = False
        reason = "d384_numerically_incomplete"
    else:
        eligible = True
        reason = "shared_w_retained_and_d384_numerically_complete"

    return {
        "criterion": "AIC",
        "shared_w_parameter_count": 4,
        "free_w_parameter_count": 6,
        **comparison,
        "d384_beta_interpretation_eligible": eligible,
        "d384_beta_interpretation_reason": reason,
        "primary_c192_vs_b192_affected": False,
    }


def validate_calibration_artifact(payload: Mapping[str, Any]) -> None:
    """새 중간 산출물 스키마와 모델별 불변조건을 함께 검사한다."""
    _validate_schema(payload, OUTPUT_SCHEMA, label="calibration_model_fits")
    try:
        verify_implementation_hashes(
            payload.get("implementation_hashes"),
            "calibration_analysis",
            label="calibration_model_fits",
        )
    except ImplementationProvenanceError as error:
        raise AnalysisBlockedError(str(error)) from error
    if (
        payload.get("analysis_script_sha256")
        != payload["implementation_hashes"].get("src/analyze_calibration.py")
    ):
        raise AnalysisBlockedError(
            "calibration_model_fits: analysis_script_sha256가 현재 "
            "analyze_calibration.py의 implementation_hashes 값과 다릅니다."
        )
    participant = payload["participants"][0]
    quality_rows = participant["calibration"]["session_block_quality"]
    block_count = int(payload["analysis_specification"]["session_block_count"])
    if [row["session_block"] for row in quality_rows] != list(
        range(1, block_count + 1)
    ):
        raise AnalysisBlockedError(
            "calibration session block 품질표가 동결 block 순서와 다릅니다."
        )
    if len({row["session_id"] for row in quality_rows}) != block_count:
        raise AnalysisBlockedError(
            "calibration session block 품질표의 session_id는 네 개 모두 달라야 합니다."
        )
    for quality in quality_rows:
        completed_trials = int(quality["completed_trials"])
        correct_trials = int(quality["correct_trials"])
        expected_accuracy = correct_trials / completed_trials
        _require_close(
            quality["accuracy"],
            expected_accuracy,
            f"calibration block {quality['session_block']}: accuracy 불일치",
        )
    models = participant["models"]
    if [row["model"] for row in models] != list(MODELS):
        raise AnalysisBlockedError("출력 모델은 B192/B384/C192/D384 순서로 각각 한 번이어야 합니다.")
    expected_parameter_keys = {
        "B192": {"c80", "w"},
        "B384": {"c80", "w"},
        "C192": {"c80_ref", "w"},
        "D384": {"c80_ref", "beta", "w"},
    }
    expected_training = {
        "B192": REFERENCE_WIDTH_TRIALS,
        "B384": CALIBRATION_TOTAL_TRIALS,
        "C192": REFERENCE_WIDTH_TRIALS,
        "D384": CALIBRATION_TOTAL_TRIALS,
    }
    requested_bootstrap = int(payload["analysis_specification"]["bootstrap_replicates"])
    if (
        payload.get("data_origin") == "human"
        and requested_bootstrap < MINIMUM_HUMAN_BOOTSTRAP_REPLICATES
    ):
        raise AnalysisBlockedError(
            "사람 calibration 산출물은 bootstrap 1000회 이상이어야 합니다."
        )
    all_converged = True
    for row in models:
        model = row["model"]
        if set(row["fit"]["parameters"]) != expected_parameter_keys[model]:
            raise AnalysisBlockedError(f"{model}: 출력 파라미터 구성이 잘못됐습니다.")
        if row["training_trial_count"] != expected_training[model]:
            raise AnalysisBlockedError(f"{model}: 훈련 시행 수가 사전 정의와 다릅니다.")
        range_checks = row.get("training_range_checks")
        if payload.get("data_origin") == "human" and not isinstance(
            range_checks, list
        ):
            raise AnalysisBlockedError(
                f"{model}: 사람 교정 산출물에는 C80 제시범위 검사가 필요합니다."
            )
        if isinstance(range_checks, list):
            expected_check_count = 1 if model in {"B192", "C192"} else 3
            if len(range_checks) != expected_check_count:
                raise AnalysisBlockedError(
                    f"{model}: C80 제시범위 검사 굵기 수가 잘못됐습니다."
                )
            for check in range_checks:
                low = float(check["presented_delta_e00_min"])
                high = float(check["presented_delta_e00_max"])
                point = float(check["c80_sim"])
                ci_low = check["ci_low"]
                ci_high = check["ci_high"]
                point_bracketed = low <= point <= high
                ci_bracketed = (
                    ci_low is not None
                    and ci_high is not None
                    and low <= float(ci_low) <= float(ci_high) <= high
                )
                if (
                    check["point_bracketed"] != point_bracketed
                    or check["ci_bracketed"] != ci_bracketed
                    or check["range_bracketed"]
                    != (point_bracketed and ci_bracketed)
                ):
                    raise AnalysisBlockedError(
                        f"{model}: C80 제시범위 검사 플래그가 수치와 다릅니다."
                    )
                if payload.get("data_origin") == "human" and not check[
                    "range_bracketed"
                ]:
                    raise AnalysisBlockedError(
                        f"{model}: 사람 C80 점추정/95% CI가 제시범위 밖입니다."
                    )
        bootstrap = row["bootstrap"]
        if bootstrap["requested_replicates"] != requested_bootstrap:
            raise AnalysisBlockedError(
                f"{model}: bootstrap 요청 수가 analysis specification과 다릅니다."
            )
        expected_pairing_id = (
            PRIMARY_BOOTSTRAP_PAIRING_ID if model in {"B192", "C192"} else None
        )
        if (
            "replicate_pairing_id" in bootstrap
            and bootstrap["replicate_pairing_id"] != expected_pairing_id
        ):
            raise AnalysisBlockedError(f"{model}: bootstrap pairing ID가 잘못됐습니다.")
        if bootstrap["successful_replicates"] > bootstrap["requested_replicates"]:
            raise AnalysisBlockedError(f"{model}: 성공 bootstrap 수가 요청 수보다 큽니다.")
        expected_rate = bootstrap["successful_replicates"] / bootstrap["requested_replicates"]
        _require_close(bootstrap["convergence_rate"], expected_rate, f"{model}: bootstrap 수렴률 불일치")
        expected_converged = (
            bootstrap["successful_replicates"] >= 2
            and expected_rate >= payload["analysis_specification"]["minimum_bootstrap_convergence_rate"]
        )
        if bootstrap["converged"] != expected_converged:
            raise AnalysisBlockedError(f"{model}: bootstrap converged 플래그가 수렴률과 다릅니다.")
        for interval in bootstrap["parameter_intervals"]:
            if interval["ci_low"] > interval["ci_high"]:
                raise AnalysisBlockedError(f"{model}: 파라미터 CI 순서가 뒤집혔습니다.")
        for prediction in row["held_out_candidate_predictions"]:
            low, high = prediction["ci_low"], prediction["ci_high"]
            if (low is None) != (high is None):
                raise AnalysisBlockedError(f"{model}: 예측 CI의 두 경계는 함께 null이어야 합니다.")
            if low is not None and low > high:
                raise AnalysisBlockedError(f"{model}: 예측 C80 CI 순서가 뒤집혔습니다.")
            if (
                "bootstrap_pairing_id" in prediction
                and prediction["bootstrap_pairing_id"] != expected_pairing_id
            ):
                raise AnalysisBlockedError(f"{model}: 예측 bootstrap pairing ID가 잘못됐습니다.")
            samples = prediction.get("bootstrap_c80_samples")
            if samples is None:
                # 구형 산출물은 백분위 CI만 보존했다. 스키마상 읽기는 허용하되
                # analyze_validation이 명시적으로 unresolvable 판정을 내린다.
                continue
            if not isinstance(samples, list) or len(samples) != bootstrap["successful_replicates"]:
                raise AnalysisBlockedError(
                    f"{model}: 예측 bootstrap 표본 수가 성공 복제 수와 다릅니다."
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in samples
            ):
                raise AnalysisBlockedError(f"{model}: 예측 bootstrap 표본이 잘못됐습니다.")
            if samples and low is not None:
                method=bootstrap.get("interval_method","percentile")
                jackknife=prediction.get("jackknife_c80_samples")
                if method=="bca-log" and (
                    not isinstance(jackknife,list) or len(jackknife)!=row["training_trial_count"]
                    or any(isinstance(v,bool) or not isinstance(v,(int,float))
                           or not math.isfinite(v) or v<=0 for v in jackknife)
                ):
                    raise AnalysisBlockedError(f"{model}: BCa jackknife 표본이 없거나 잘못됐습니다.")
                try:
                    sample_low,sample_high=_bootstrap_interval(
                        samples,float(prediction["c80_sim"]),float(bootstrap["confidence"]),
                        positive=True,method=method,jackknife=jackknife)
                except ValueError as error:
                    raise AnalysisBlockedError(f"{model}: CI 재계산 실패: {error}") from error
                _require_close(
                    low, sample_low, f"{model}: 예측 bootstrap CI 하한 불일치"
                )
                _require_close(
                    high, sample_high, f"{model}: 예측 bootstrap CI 상한 불일치"
                )
        all_converged &= bool(row["fit"]["converged"] and bootstrap["converged"])

    by_model = {row["model"]: row for row in models}
    b192_predictions = {
        row["stroke_width_px"]: row
        for row in by_model["B192"]["held_out_candidate_predictions"]
    }
    c192_predictions = {
        row["stroke_width_px"]: row
        for row in by_model["C192"]["held_out_candidate_predictions"]
    }
    if set(b192_predictions) != set(c192_predictions):
        raise AnalysisBlockedError("B192/C192 held-out 후보 굵기가 다릅니다.")
    for width_px, b_prediction in b192_predictions.items():
        c_prediction = c192_predictions[width_px]
        if (
            b_prediction.get("bootstrap_pairing_id") != PRIMARY_BOOTSTRAP_PAIRING_ID
            or c_prediction.get("bootstrap_pairing_id") != PRIMARY_BOOTSTRAP_PAIRING_ID
            or "bootstrap_c80_samples" not in b_prediction
            or "bootstrap_c80_samples" not in c_prediction
        ):
            continue
        b_samples = np.asarray(b_prediction["bootstrap_c80_samples"], dtype=float)
        c_samples = np.asarray(c_prediction["bootstrap_c80_samples"], dtype=float)
        if b_samples.shape != c_samples.shape:
            raise AnalysisBlockedError("B192/C192 paired bootstrap 표본 수가 다릅니다.")
        if b_samples.size:
            point_ratio = float(c_prediction["c80_sim"]) / float(
                b_prediction["c80_sim"]
            )
            if not np.allclose(
                c_samples, b_samples * point_ratio, rtol=1e-8, atol=1e-10
            ):
                raise AnalysisBlockedError(
                    "B192/C192 bootstrap 표본이 같은 모의 반응 복제로 짝지어지지 않았습니다."
                )

    sensitivity = participant["width_slope_sensitivity"]
    numeric_fields = (
        "shared_w_aic",
        "free_w_aic",
        "delta_aic_free_minus_shared",
    )
    if any(not math.isfinite(float(sensitivity[field])) for field in numeric_fields):
        raise AnalysisBlockedError("교정 공통-w 민감도 AIC 값은 유한수여야 합니다.")
    _require_close(
        sensitivity["delta_aic_free_minus_shared"],
        float(sensitivity["free_w_aic"]) - float(sensitivity["shared_w_aic"]),
        "교정 공통-w 민감도 delta AIC 불일치",
    )
    comparison_resolved = bool(
        sensitivity["shared_w_converged"] and sensitivity["free_w_converged"]
    )
    if sensitivity["comparison_resolved"] != comparison_resolved:
        raise AnalysisBlockedError(
            "교정 공통-w 민감도 comparison_resolved가 수렴 상태와 다릅니다."
        )
    free_w_preferred = bool(
        comparison_resolved
        and float(sensitivity["free_w_aic"]) < float(sensitivity["shared_w_aic"])
    )
    if sensitivity["free_w_preferred"] != free_w_preferred:
        raise AnalysisBlockedError(
            "교정 공통-w 민감도 free_w_preferred가 AIC와 다릅니다."
        )
    expected_interpretation = (
        "unresolved_numerical_convergence"
        if not comparison_resolved
        else (
            "free_w_lower_aic_warning_not_selected"
            if free_w_preferred
            else "shared_w_retained"
        )
    )
    if sensitivity["interpretation"] != expected_interpretation:
        raise AnalysisBlockedError("교정 공통-w 민감도 interpretation이 수치와 다릅니다.")
    d384_ready = bool(
        by_model["D384"]["fit"]["converged"]
        and by_model["D384"]["bootstrap"]["converged"]
    )
    if not comparison_resolved:
        expected_eligible = False
        expected_reason = "shared_free_w_comparison_unresolved"
    elif free_w_preferred:
        expected_eligible = False
        expected_reason = "free_w_lower_aic"
    elif not d384_ready:
        expected_eligible = False
        expected_reason = "d384_numerically_incomplete"
    else:
        expected_eligible = True
        expected_reason = "shared_w_retained_and_d384_numerically_complete"
    if (
        sensitivity["d384_beta_interpretation_eligible"] != expected_eligible
        or sensitivity["d384_beta_interpretation_reason"] != expected_reason
    ):
        raise AnalysisBlockedError(
            "D384 beta 해석 가능 플래그가 공통-w 민감도/수렴 상태와 다릅니다."
        )
    expected_status = "complete" if all_converged else "incomplete_numerical_convergence"
    if participant["analysis_status"] != expected_status:
        raise AnalysisBlockedError("analysis_status가 적합/bootstrap 수렴 상태와 다릅니다.")


def analyze_calibration(
    config_path: str | Path,
    trials_path: str | Path,
    *,
    calibration_manifest_path: str | Path | None = None,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260914,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """검증된 입력만 사용해 한 참가자의 네 교정 모형과 후보 예측을 만든다."""
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 2:
        raise AnalysisBlockedError("bootstrap_replicates는 2 이상의 정수여야 합니다.")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise AnalysisBlockedError("bootstrap_seed는 0 이상의 정수여야 합니다.")
    if not isinstance(allow_synthetic, bool):
        raise AnalysisBlockedError("allow_synthetic은 bool이어야 합니다.")

    config, config_sha256 = _read_json_object(config_path)
    records, trial_file_sha256 = _read_jsonl(trials_path)
    origins = {record.get("data_origin") for record in records}
    if len(origins) != 1:
        raise AnalysisBlockedError("한 분석 입력에 서로 다른 data_origin을 섞을 수 없습니다.")
    data_origin = next(iter(origins))
    if data_origin == "synthetic" and not allow_synthetic:
        raise AnalysisBlockedError(
            "합성 자료는 연구 결과로 오인하지 않도록 --allow-synthetic을 명시해야 합니다."
        )
    if data_origin not in {"human", "synthetic"}:
        raise AnalysisBlockedError(f"지원하지 않는 data_origin입니다: {data_origin!r}")

    held_out_rows = [
        index
        for index, record in enumerate(records, start=1)
        if record.get("phase") == "validation"
    ]
    if held_out_rows:
        preview = ", ".join(str(index) for index in held_out_rows[:5])
        raise AnalysisBlockedError(
            "교정 분석 입력에 validation/held-out 시행이 섞여 있습니다. "
            "독립 검증을 관측하지 않은 calibration-only JSONL만 사용해야 합니다"
            f" (행: {preview})."
        )
    if data_origin == "human" and calibration_manifest_path is None:
        raise AnalysisBlockedError(
            "사람 calibration 분석에는 --calibration-manifest가 필요합니다."
        )

    _validate_config(config, data_origin=data_origin)
    _validate_trials(
        records,
        config=config,
        config_sha256=config_sha256,
        data_origin=data_origin,
    )
    stimulus_manifest_sha256: str | None = None
    if calibration_manifest_path is not None:
        stimulus_manifest_sha256 = _validate_calibration_manifest(
            calibration_manifest_path,
            config=config,
            config_sha256=config_sha256,
            records=records,
            require_record_manifest_hash=(data_origin == "human"),
        )
    if data_origin == "human" and any(
        record.get("phase") == "calibration"
        and record.get("trial_status") == "aborted"
        for record in records
    ):
        raise AnalysisBlockedError(
            "사람 calibration 원자료에 활성 자극 중 중단 시행이 있습니다. "
            "manifest 감사는 완료했지만 사전등록 정책상 재시도 자료를 포함한 "
            "확증 분석은 허용하지 않습니다."
        )
    if (
        data_origin == "human"
        and bootstrap_replicates < MINIMUM_HUMAN_BOOTSTRAP_REPLICATES
    ):
        raise AnalysisBlockedError(
            "사람 calibration 분석은 bootstrap_replicates를 1000 이상으로 "
            "고정해야 합니다."
        )
    if data_origin == "human":
        try:
            expected_trials_path = canonical_journal_path(
                config, "calibration", PROJECT_ROOT
            )
        except CollectionBlockedError as error:
            raise AnalysisBlockedError(str(error)) from error
        if os.path.normcase(str(Path(trials_path).resolve())) != os.path.normcase(
            str(expected_trials_path)
        ):
            raise AnalysisBlockedError(
                "사람 calibration 분석의 --trials는 config에 동결된 canonical "
                f"journal이어야 합니다: {expected_trials_path}"
            )
    completed, calibration_summary = _participant_calibration(records, config)
    model_results = _fit_all_models(
        completed,
        config,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        require_c80_bracketing=(data_origin == "human"),
    )
    width_slope_sensitivity = _calibration_width_slope_sensitivity(
        completed,
        config,
        model_results,
    )
    all_converged = all(
        row["fit"]["converged"] and row["bootstrap"]["converged"]
        for row in model_results
    )
    profile = config["observer_profile"]
    analysis_implementation_hashes = implementation_hashes("calibration_analysis")
    payload = {
        "schema_version": "1.0",
        "artifact_type": "calibration_model_fits",
        "analysis_scope": "calibration_only_intermediate",
        "data_origin": data_origin,
        "research_conclusion_allowed": False,
        "final_thresholds_ready": False,
        "contains_observed_held_out": False,
        "coordinate": "post_machado_delta_e00",
        "formula_version": "direct-c80-4afc-v1",
        "analysis_script_sha256": analysis_implementation_hashes[
            "src/analyze_calibration.py"
        ],
        "implementation_hashes": analysis_implementation_hashes,
        "source": {
            "config_version": config["config_version"],
            "config_sha256": config_sha256,
            "trial_file_sha256": trial_file_sha256,
            "stimulus_manifest_sha256": stimulus_manifest_sha256,
            "git_commit": calibration_summary.pop("git_commit"),
            "stimulus_family_id": calibration_summary.pop("stimulus_family_id"),
            "renderer_id": calibration_summary.pop("renderer_id"),
            "total_record_count": len(records),
        },
        "analysis_specification": {
            "models": list(MODELS),
            "calibration_trials_per_width": list(CALIBRATION_WIDTH_TRIAL_COUNTS),
            "session_block_count": block_count_from_config(config),
            "reference_width_policy": "middle_of_three_sorted_calibration_widths",
            "confidence": CONFIDENCE,
            "bootstrap_replicates": bootstrap_replicates,
            "minimum_bootstrap_convergence_rate": MINIMUM_BOOTSTRAP_CONVERGENCE_RATE,
            "bootstrap_seed": bootstrap_seed,
            "szafir_direction_lab": [float(value) for value in config["color_design"]["direction_lab"]],
            "szafir_probability_coordinate": float(config["model_plan"]["szafir_probability_coordinate"]),
            "szafir_extrapolation_allowed": False,
        },
        "participants": [
            {
                "participant_id": profile["participant_id"],
                "cvd_type": profile["cvd_type"],
                "machado_severity": float(profile["severity"]),
                "severity_source": profile["severity_source"],
                "severity_session_id": profile["severity_session_id"],
                "fixed_lapse": float(profile["fixed_lapse"]),
                "lapse_source": profile["lapse_source"],
                "lapse_session_id": profile["lapse_session_id"],
                "calibration": calibration_summary,
                "width_slope_sensitivity": width_slope_sensitivity,
                "analysis_status": "complete" if all_converged else "incomplete_numerical_convergence",
                "models": model_results,
            }
        ],
    }
    validate_calibration_artifact(payload)
    return payload


def _write_json_atomic(payload: Mapping[str, Any], output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise AnalysisBlockedError(
            f"출력 파일이 이미 있습니다: {output}. 덮어쓰려면 --overwrite를 명시하세요."
        )
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
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
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "동결 설정과 96/192/96회 완료 trial JSONL을 검증한 뒤 "
            "B192/B384/C192/D384 calibration-only 중간 산출물을 만듭니다."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="동결 experiment_config JSON")
    parser.add_argument("--trials", type=Path, required=True, help="실제 또는 합성 trial JSONL")
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        help="사람 자료에 필수인 전체 384시행 compiled calibration pixel manifest",
    )
    parser.add_argument("--output", type=Path, required=True, help="새 calibration_model_fits JSON")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="합성 fixture 분석임을 명시적으로 승인합니다. 사람 결과로 승격되지는 않습니다.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = analyze_calibration(
            args.config,
            args.trials,
            calibration_manifest_path=args.calibration_manifest,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            allow_synthetic=args.allow_synthetic,
        )
        _write_json_atomic(payload, args.output, overwrite=args.overwrite)
    except (AnalysisBlockedError, ContractError, OSError, TypeError, ValueError) as error:
        print(f"분석 차단: {error}", file=sys.stderr)
        return 2
    participant = payload["participants"][0]
    return 0 if participant["analysis_status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AnalysisBlockedError",
    "analyze_calibration",
    "main",
    "validate_calibration_artifact",
]
