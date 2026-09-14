"""동결된 독립 검증 블록 세 개에서 direct C80을 추정한다.

세 64시행 블록의 C80 세 개와 공통 w를 joint MLE로 적합하고, joint 모수적
bootstrap으로 구간·공분산을 구한다. 세 C80의 CV가 0.20 이하이고 세 95%
구간에 공통 교집합이 있을 때만 GLS로 log(C80) 통합값을 만든다. 불안정하면
통합값은 ``null``이다. B96/C96의 우열은 교정 단계의 paired 예측 bootstrap과
검증 단계의 pooled bootstrap을 함께 전파한 Δloss 95% 구간으로만 판정한다.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

from analyze_calibration import (
    AnalysisBlockedError,
    _validate_trial_semantics,
    validate_calibration_artifact,
)
from implementation_provenance import implementation_hashes
from experiment_runner import CollectionBlockedError, canonical_journal_path
from stimulus_renderer import RendererValidationError, verify_stimulus_artifact
from validation_design import (
    BLOCK_COUNT,
    TRIALS_PER_BLOCK,
    ValidationDesignError,
    validate_plan,
)
from validation_psychometrics import (
    bootstrap_shared_w,
    compare_shared_vs_free_w_aic,
    fit_shared_w,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUMAN_TRIAL_SCHEMA = PROJECT_ROOT / "schemas" / "human" / "trial.schema.json"
HUMAN_THRESHOLDS_SCHEMA = (
    PROJECT_ROOT / "schemas" / "human" / "participant_thresholds.schema.json"
)
CV_LIMIT = 0.20
MINIMUM_BOOTSTRAP_CONVERGENCE_RATE = 0.90
MINIMUM_HUMAN_BOOTSTRAP_REPLICATES = 1000
ERROR_TIE_TOLERANCE_DELTA_E00 = 1e-9
_REVIEWER_ID_RE = re.compile(r"^R[0-9A-Za-z_-]+$")


class ValidationAnalysisError(ValueError):
    """검증 자료가 동결 계획과 맞지 않거나 분석이 수치적으로 불가능할 때."""


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationAnalysisError(f"JSON 객체에 중복 key가 있습니다: {key}")
        result[key] = value
    return result


def _read_bytes(path: str | Path) -> bytes:
    try:
        return Path(path).read_bytes()
    except FileNotFoundError as error:
        raise ValidationAnalysisError(f"파일이 없습니다: {path}") from error


def _read_json(path: str | Path) -> tuple[dict[str, Any], str]:
    raw = _read_bytes(path)
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_object_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationAnalysisError(f"UTF-8 JSON을 읽을 수 없습니다: {path}") from error
    if not isinstance(payload, dict):
        raise ValidationAnalysisError(f"JSON 최상위는 객체여야 합니다: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _validate_confirmation_review(
    review_record_path: str | Path,
    *,
    participant_id: str,
    config_sha256: str,
    source_trials_sha256: str,
    validation_plan_sha256: str,
    validation_manifest_sha256: str,
    latest_completed_timestamp_utc: datetime,
) -> dict[str, Any]:
    """Validate and bind an independent human review record to exact inputs."""

    review, review_sha256 = _read_json(review_record_path)
    required = {
        "schema_version",
        "review_type",
        "reviewer_id",
        "reviewed_at_utc",
        "review_note",
        "approved_for_confirmation",
        "participant_id",
        "config_sha256",
        "source_trials_sha256",
        "validation_plan_sha256",
        "validation_manifest_sha256",
    }
    if set(review) != required:
        missing = sorted(required - set(review))
        extra = sorted(set(review) - required)
        raise ValidationAnalysisError(
            f"validation review record 필드가 정확하지 않습니다: missing={missing}, extra={extra}"
        )
    if review.get("schema_version") != "1.0" or review.get("review_type") != (
        "human_validation_confirmation"
    ):
        raise ValidationAnalysisError("validation review record 종류/버전이 잘못됐습니다.")
    reviewer_id = review.get("reviewer_id")
    if not isinstance(reviewer_id, str) or not _REVIEWER_ID_RE.fullmatch(reviewer_id):
        raise ValidationAnalysisError(
            "reviewer_id는 직접 식별자가 아닌 R 접두 영문·숫자·밑줄·하이픈 코드여야 합니다."
        )
    reviewed_at = review.get("reviewed_at_utc")
    if not isinstance(reviewed_at, str) or not reviewed_at.endswith("Z"):
        raise ValidationAnalysisError("reviewed_at_utc는 Z로 끝나는 UTC 시각이어야 합니다.")
    try:
        parsed_reviewed_at = datetime.fromisoformat(reviewed_at[:-1] + "+00:00")
    except ValueError as error:
        raise ValidationAnalysisError("reviewed_at_utc 형식이 잘못됐습니다.") from error
    if parsed_reviewed_at.tzinfo is None:
        raise ValidationAnalysisError("reviewed_at_utc에는 시간대가 필요합니다.")
    parsed_reviewed_at = parsed_reviewed_at.astimezone(timezone.utc)
    if parsed_reviewed_at < latest_completed_timestamp_utc.astimezone(timezone.utc):
        raise ValidationAnalysisError(
            "reviewed_at_utc는 마지막 완료 validation 시행 뒤여야 합니다."
        )
    if parsed_reviewed_at > datetime.now(timezone.utc):
        raise ValidationAnalysisError("reviewed_at_utc는 현재 UTC보다 미래일 수 없습니다.")
    note = review.get("review_note")
    if not isinstance(note, str) or not note.strip():
        raise ValidationAnalysisError("review_note에는 실제 검토 내용을 기록해야 합니다.")
    if review.get("approved_for_confirmation") is not True:
        raise ValidationAnalysisError("review record가 confirmed 승인을 명시하지 않았습니다.")
    expected = {
        "participant_id": participant_id,
        "config_sha256": config_sha256,
        "source_trials_sha256": source_trials_sha256,
        "validation_plan_sha256": validation_plan_sha256,
        "validation_manifest_sha256": validation_manifest_sha256,
    }
    for field, value in expected.items():
        if review.get(field) != value:
            raise ValidationAnalysisError(
                f"validation review record.{field}가 실제 분석 입력과 다릅니다."
            )
    return {
        "review_record_sha256": review_sha256,
        "review_type": review["review_type"],
        "reviewer_id": reviewer_id,
        "reviewed_at_utc": reviewed_at,
        "review_note": note,
        "approved_for_confirmation": True,
        **expected,
    }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    raw = _read_bytes(path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValidationAnalysisError(f"UTF-8 JSONL이 아닙니다: {path}") from error
    if not text or not text.endswith("\n"):
        raise ValidationAnalysisError("검증 JSONL은 비어 있지 않고 마지막 레코드가 개행으로 끝나야 합니다.")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValidationAnalysisError(f"{path}:{line_number}: 빈 줄은 허용하지 않습니다.")
        try:
            row = json.loads(
                line, object_pairs_hook=_reject_duplicate_object_keys
            )
        except json.JSONDecodeError as error:
            raise ValidationAnalysisError(f"{path}:{line_number}: JSON 오류: {error}") from error
        if not isinstance(row, dict):
            raise ValidationAnalysisError(f"{path}:{line_number}: 객체가 아닙니다.")
        records.append(row)
    return records, hashlib.sha256(raw).hexdigest()


def _schema_validator(path: Path) -> Draft202012Validator:
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _validate_schema(payload: Any, schema_path: Path, label: str) -> None:
    errors = sorted(_schema_validator(schema_path).iter_errors(payload), key=lambda item: list(item.path))
    if errors:
        details = "; ".join(f"{list(error.path)}: {error.message}" for error in errors[:8])
        raise ValidationAnalysisError(f"{label} 스키마 오류: {details}")


def _close(first: Any, second: Any) -> bool:
    return math.isclose(float(first), float(second), rel_tol=1e-6, abs_tol=1e-8)


def _safe_manifest_artifact(base: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValidationAnalysisError(f"{label}는 manifest 기준 상대 경로여야 합니다.")
    target = (base / relative).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError as error:
        raise ValidationAnalysisError(f"{label}가 manifest 폴더 밖을 가리킵니다.") from error
    return target


def _validate_validation_manifest(
    manifest_path: str | Path,
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    calibration: Mapping[str, Any],
    calibration_sha256: str,
    plan: Mapping[str, Any],
    plan_raw_sha256: str,
    records: Sequence[Mapping[str, Any]],
) -> str:
    """사람 응답을 정확한 validation PNG manifest와 다시 결박한다."""

    manifest_file = Path(manifest_path).resolve()
    manifest, manifest_sha256 = _read_json(manifest_file)
    expected_headers = {
        "schema_version": "1.0",
        "artifact_type": "compiled_validation_pixel_stimulus_manifest",
        "phase": "validation",
        "data_origin": "human",
        "config_version": config["config_version"],
        "config_canonical_sha256": _canonical_sha256(config),
        "schedule_config_sha256": config_sha256,
        "source_config_raw_sha256": config_sha256,
        "validation_plan_sha256": plan["plan_sha256"],
        "validation_plan_raw_file_sha256": plan_raw_sha256,
        "calibration_model_fits_sha256": calibration_sha256,
        "source_calibration_trials_sha256": calibration["source"]["trial_file_sha256"],
        "git_commit": calibration["source"]["git_commit"],
        "stimulus_family_id": config["color_design"]["stimulus_family_id"],
        "participant_id": config["observer_profile"]["participant_id"],
        "cvd_type": config["observer_profile"]["cvd_type"],
        "machado_severity": config["observer_profile"]["severity"],
        "trial_count": BLOCK_COUNT * TRIALS_PER_BLOCK,
        "png_files_written": True,
    }
    for field, expected in expected_headers.items():
        if manifest.get(field) != expected:
            raise ValidationAnalysisError(
                f"validation manifest.{field}가 동결 입력과 다릅니다."
            )
    manifest_rows = manifest.get("trials")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != BLOCK_COUNT * TRIALS_PER_BLOCK:
        raise ValidationAnalysisError("validation manifest는 정확히 192개 시행을 가져야 합니다.")
    by_id: dict[int, Mapping[str, Any]] = {}
    for row in manifest_rows:
        if not isinstance(row, Mapping) or isinstance(row.get("trial_id"), bool) or not isinstance(
            row.get("trial_id"), int
        ):
            raise ValidationAnalysisError("validation manifest trial 형식이 잘못됐습니다.")
        trial_id = int(row["trial_id"])
        if trial_id in by_id:
            raise ValidationAnalysisError(f"validation manifest trial_id가 중복됩니다: {trial_id}")
        by_id[trial_id] = row
    if set(by_id) != set(range(289, 481)):
        raise ValidationAnalysisError(
            "validation manifest trial_id는 예약 범위 289~480이어야 합니다."
        )

    immutable_fields = (
        "config_version",
        "config_sha256",
        "git_commit",
        "stimulus_family_id",
        "participant_id",
        "trial_id",
        "phase",
        "validation_plan_sha256",
        "validation_block_id",
        "validation_level_index",
        "stroke_width_px",
        "stroke_width_deg",
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
    for record in records:
        trial_id = int(record["trial_id"])
        expected = by_id.get(trial_id)
        if expected is None:
            raise ValidationAnalysisError(
                f"trial {trial_id}: stimulus manifest에 없는 시행입니다."
            )
        if record.get("stimulus_manifest_sha256") != manifest_sha256:
            raise ValidationAnalysisError(
                f"trial {trial_id}: stimulus_manifest_sha256이 실제 manifest와 다릅니다."
            )
        for field in immutable_fields:
            if record.get(field) != expected.get(field):
                raise ValidationAnalysisError(
                    f"trial {trial_id}: {field}가 stimulus manifest와 다릅니다."
                )
        artifact = expected.get("raster_artifact")
        if not isinstance(artifact, Mapping):
            raise ValidationAnalysisError(f"trial {trial_id}: raster_artifact가 없습니다.")
        png = _safe_manifest_artifact(base, artifact.get("png"), f"trial {trial_id} PNG")
        sidecar = _safe_manifest_artifact(
            base, artifact.get("pixel_metadata"), f"trial {trial_id} sidecar"
        )
        try:
            verification = verify_stimulus_artifact(
                png, sidecar, config=config, trial=expected
            )
        except (OSError, RendererValidationError) as error:
            raise ValidationAnalysisError(
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
            raise ValidationAnalysisError(
                f"trial {trial_id}: 응답·manifest·PNG 픽셀 hash가 서로 다릅니다."
            )
        sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        if sidecar_sha256 != artifact.get("pixel_metadata_sha256"):
            raise ValidationAnalysisError(
                f"trial {trial_id}: sidecar raw hash가 manifest와 다릅니다."
            )
        try:
            repository_path = sidecar.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError as error:
            raise ValidationAnalysisError(
                f"trial {trial_id}: 사람 자극 sidecar가 저장소 밖에 있습니다."
            ) from error
        if record.get("stimulus_pixel_metadata_path") != repository_path:
            raise ValidationAnalysisError(
                f"trial {trial_id}: sidecar 경로가 실제 artifact와 다릅니다."
            )
    if set(by_id) != {int(record["trial_id"]) for record in records}:
        raise ValidationAnalysisError("응답과 validation manifest의 시행 집합이 다릅니다.")
    return manifest_sha256


def _validate_and_group_trials(
    records: Sequence[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    human: bool,
) -> list[list[dict[str, Any]]]:
    if len(records) < BLOCK_COUNT * TRIALS_PER_BLOCK:
        raise ValidationAnalysisError(
            "검증에는 중단 기록과 별개로 정확히 3×64=192개 완료 시행이 필요합니다."
        )
    schedule = {int(row["trial_id"]): row for row in plan["schedule"]}
    if len(schedule) != len(plan["schedule"]):
        raise ValidationAnalysisError("동결 계획에 중복 trial_id가 있습니다.")
    completed_keys: set[tuple[str, str, int]] = set()
    completed_trial_ids: set[int] = set()
    grouped: list[list[dict[str, Any]]] = [[] for _ in range(BLOCK_COUNT)]
    profile = config["observer_profile"]
    for index, record in enumerate(records, start=1):
        if human:
            _validate_schema(record, HUMAN_TRIAL_SCHEMA, f"trial {index}")
            try:
                _validate_trial_semantics(record, label=f"trial {index}")
            except AnalysisBlockedError as error:
                raise ValidationAnalysisError(str(error)) from error
        trial_id = int(record.get("trial_id", -1))
        expected = schedule.get(trial_id)
        if expected is None:
            raise ValidationAnalysisError(f"동결 계획에 없는 trial_id입니다: {trial_id}")
        key = (record.get("participant_id"), record.get("config_sha256"), trial_id)
        if key in completed_keys:
            raise ValidationAnalysisError(
                f"완료한 참가자/config/trial 뒤 중복 기록이 있습니다: {key}"
            )
        for field in (
            "participant_id",
            "config_sha256",
            "validation_plan_sha256",
            "validation_block_id",
            "validation_level_index",
            "stroke_width_px",
            "correct_position",
            "odd_color",
            "position_colors",
        ):
            if record.get(field) != expected.get(field):
                raise ValidationAnalysisError(f"trial {trial_id}: {field}가 동결 계획과 다릅니다.")
        for field in ("stroke_width_deg", "target_delta_e00_simulated"):
            if not _close(record.get(field), expected.get(field)):
                raise ValidationAnalysisError(f"trial {trial_id}: {field}가 동결 계획과 다릅니다.")
        level = plan["levels"][int(record["validation_level_index"]) - 1]
        if not _close(record.get("delta_e00_simulated"), level["actual_delta_e00_simulated"]):
            raise ValidationAnalysisError(f"trial {trial_id}: 실제 표시 ΔE00가 plan 양자화값과 다릅니다.")
        if record.get("phase") != "validation":
            raise ValidationAnalysisError(f"trial {trial_id}: validation 시행만 분석합니다.")
        status = record.get("trial_status")
        if status not in {"completed", "aborted"}:
            raise ValidationAnalysisError(
                f"trial {trial_id}: trial_status가 completed/aborted가 아닙니다."
            )
        if status == "aborted":
            if any(
                record.get(field) is not None
                for field in ("response_position", "is_correct", "response_time_ms")
            ):
                raise ValidationAnalysisError(
                    f"trial {trial_id}: 중단 시행 응답 필드는 null이어야 합니다."
                )
            if human:
                raise ValidationAnalysisError(
                    "사람 validation 원자료에 활성 자극 중 중단 시행이 있습니다. "
                    "manifest 감사는 완료했지만 사전등록 정책상 재시도 자료를 "
                    "포함한 확증 분석은 허용하지 않습니다."
                )
            continue
        completed_keys.add(key)
        completed_trial_ids.add(trial_id)
        if not isinstance(record.get("is_correct"), bool):
            raise ValidationAnalysisError(f"trial {trial_id}: is_correct는 boolean이어야 합니다.")
        response = record.get("response_position")
        if not isinstance(response, int) or isinstance(response, bool) or not 1 <= response <= 4:
            raise ValidationAnalysisError(f"trial {trial_id}: response_position이 잘못됐습니다.")
        if record["is_correct"] != (response == record["correct_position"]):
            raise ValidationAnalysisError(f"trial {trial_id}: is_correct가 실제 응답과 다릅니다.")
        if not _close(record.get("fixed_lapse"), profile["fixed_lapse"]):
            raise ValidationAnalysisError(f"trial {trial_id}: fixed_lapse가 config와 다릅니다.")
        block_number = int(str(record["validation_block_id"])[1:])
        if not 1 <= block_number <= BLOCK_COUNT:
            raise ValidationAnalysisError(f"trial {trial_id}: validation_block_id가 잘못됐습니다.")
        grouped[block_number - 1].append(record)

    if set(schedule) != completed_trial_ids:
        raise ValidationAnalysisError(
            "동결 계획의 완료 시행이 누락됐거나 계획 밖 완료가 있습니다."
        )
    for block_number, block in enumerate(grouped, start=1):
        if len(block) != TRIALS_PER_BLOCK:
            raise ValidationAnalysisError(f"V{block_number:02d}는 정확히 64시행이어야 합니다.")
        sessions = {row.get("session_id") for row in block}
        if len(sessions) != 1 or None in sessions:
            raise ValidationAnalysisError(f"V{block_number:02d}는 독립된 단일 session_id여야 합니다.")
    completed_session_ids = {
        next(iter({row["session_id"] for row in block})) for block in grouped
    }
    if len(completed_session_ids) != BLOCK_COUNT:
        raise ValidationAnalysisError(
            "V01/V02/V03 완료 시행은 서로 다른 session_id 세 개에서 수집해야 합니다."
        )
    return grouped


def _canonical_block_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(records, key=lambda row: int(row["trial_id"]))
    raw = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in ordered
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _prediction_rows(calibration_participant: Mapping[str, Any], width_px: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in calibration_participant["models"]:
        matches = [
            item for item in model["held_out_candidate_predictions"]
            if item["stroke_width_px"] == width_px
        ]
        if len(matches) != 1:
            raise ValidationAnalysisError(f"{model['model']}의 선택 굵기 예측이 없습니다.")
        prediction = matches[0]
        if prediction["ci_low"] is None or prediction["ci_high"] is None:
            raise ValidationAnalysisError(f"{model['model']}의 선택 굵기 CI가 없습니다.")
        if not float(prediction["ci_low"]) <= float(prediction["c80_sim"]) <= float(
            prediction["ci_high"]
        ):
            raise ValidationAnalysisError(f"{model['model']}의 C80 점예측이 CI 밖입니다.")
        rows.append(
            {
                "model": model["model"],
                "stroke_width_px": width_px,
                "stroke_width_deg": float(prediction["stroke_width_deg"]),
                "c80_sim": float(prediction["c80_sim"]),
                "ci_low": float(prediction["ci_low"]),
                "ci_high": float(prediction["ci_high"]),
                "value_type": "predicted",
                "w": float(model["fit"]["parameters"]["w"]),
                "lapse": float(model["fit"]["fixed_lapse"]),
                "fit_trial_count": int(model["training_trial_count"]),
                "fit_converged": bool(model["fit"]["converged"]),
                "bootstrap_replicates": int(model["bootstrap"]["requested_replicates"]),
                "bootstrap_successful_replicates": int(
                    model["bootstrap"]["successful_replicates"]
                ),
                "bootstrap_convergence_rate": float(model["bootstrap"]["convergence_rate"]),
                "bootstrap_c80_samples": [
                    float(value)
                    for value in prediction.get("bootstrap_c80_samples", ())
                ],
                "bootstrap_pairing_id": prediction.get("bootstrap_pairing_id"),
            }
        )
    return rows


def _pool_blocks(
    blocks: Sequence[Mapping[str, Any]],
    log_covariance: np.ndarray,
    bootstrap_c80_samples: np.ndarray,
) -> tuple[
    float,
    bool,
    bool,
    float | None,
    float | None,
    float | None,
    np.ndarray,
    bool,
]:
    values = np.asarray([row["c80_sim"] for row in blocks], dtype=float)
    cv = float(np.std(values, ddof=1) / np.mean(values))
    intervals_compatible = max(row["c80_ci"]["ci_low"] for row in blocks) <= min(
        row["c80_ci"]["ci_high"] for row in blocks
    )
    all_block_ranges_bracketed = all(
        bool(row.get("range_bracketed")) for row in blocks
    )
    stable = cv <= CV_LIMIT and intervals_compatible and all_block_ranges_bracketed
    if not stable:
        return (
            cv,
            intervals_compatible,
            False,
            None,
            None,
            None,
            np.empty(0),
            all_block_ranges_bracketed,
        )
    covariance = np.asarray(log_covariance, dtype=float)
    if covariance.shape != (3, 3) or np.any(~np.isfinite(covariance)):
        raise ValidationAnalysisError("joint bootstrap log(C80) 공분산을 계산할 수 없습니다.")
    inverse = np.linalg.pinv(covariance, rcond=1e-12)
    ones = np.ones(3, dtype=float)
    denominator = float(ones @ inverse @ ones)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValidationAnalysisError("joint bootstrap GLS pooled 분산을 계산할 수 없습니다.")
    weights = (inverse @ ones) / denominator
    pooled_log = float(weights @ np.log(values))
    pooled_se = math.sqrt(1.0 / denominator)
    samples = np.asarray(bootstrap_c80_samples, dtype=float)
    if (
        samples.ndim != 2
        or samples.shape[1] != 3
        or samples.shape[0] < 2
        or np.any(~np.isfinite(samples))
        or np.any(samples <= 0.0)
    ):
        raise ValidationAnalysisError("joint bootstrap pooled C80 표본을 계산할 수 없습니다.")
    pooled_samples = np.exp(np.log(samples) @ weights)
    pooled = math.exp(pooled_log)
    pooled_low = math.exp(pooled_log - 1.959963984540054 * pooled_se)
    pooled_high = math.exp(pooled_log + 1.959963984540054 * pooled_se)
    presented_low = max(float(row["presented_delta_e00_min"]) for row in blocks)
    presented_high = min(float(row["presented_delta_e00_max"]) for row in blocks)
    pooled_range_bracketed = (
        presented_low <= pooled_low <= pooled <= pooled_high <= presented_high
    )
    if not pooled_range_bracketed:
        return (
            cv,
            intervals_compatible,
            False,
            None,
            None,
            None,
            np.empty(0),
            all_block_ranges_bracketed,
        )
    return (
        cv,
        intervals_compatible,
        True,
        pooled,
        pooled_low,
        pooled_high,
        pooled_samples,
        True,
    )


def _delta_loss_bootstrap(
    by_model: Mapping[str, Mapping[str, Any]],
    observed: float | None,
    observed_bootstrap_samples: np.ndarray,
    shared_w_sensitivity: Mapping[str, Any],
) -> dict[str, Any]:
    """B96/C96 예측과 held-out 관측의 불확실성을 함께 전파한다.

    B96/C96은 교정 단계에서 같은 모의 반응 복제로 짝지어 적합한다. 검증
    bootstrap은 교정 자료와 독립이므로 동일 인덱스로 결합해
    ``|C96-O|-|B96-O|``의 empirical percentile 구간을 만든다. 필요한 원시
    복제값이나 pairing 표지가 없는 구형 산출물은 점추정 승패로 대체하지
    않고 명시적으로 ``unresolvable``을 반환한다.
    """

    definition = "abs(C96_prediction-observed)-abs(B96_prediction-observed)"
    result: dict[str, Any] = {
        "definition": definition,
        "point_estimate": None,
        "ci_low": None,
        "ci_high": None,
        "confidence": 0.95,
        "successful_combined_replicates": 0,
        "calibration_pairing_id": None,
        "calibration_prediction_uncertainty_included": False,
        "validation_observation_uncertainty_included": False,
        "status": "unresolvable",
        "reason": "validation_unstable",
    }
    if observed is None:
        return result

    b96 = by_model["B96"]
    c96 = by_model["C96"]
    point = abs(float(c96["c80_sim"]) - observed) - abs(
        float(b96["c80_sim"]) - observed
    )
    result["point_estimate"] = point
    pairing_id = b96.get("bootstrap_pairing_id")
    if (
        not isinstance(pairing_id, str)
        or not pairing_id
        or c96.get("bootstrap_pairing_id") != pairing_id
    ):
        result["reason"] = "paired_calibration_bootstrap_unavailable"
        return result
    result["calibration_pairing_id"] = pairing_id

    try:
        b_samples = np.asarray(b96.get("bootstrap_c80_samples", ()), dtype=float)
        c_samples = np.asarray(c96.get("bootstrap_c80_samples", ()), dtype=float)
        observed_samples = np.asarray(observed_bootstrap_samples, dtype=float)
    except (TypeError, ValueError):
        result["reason"] = "invalid_bootstrap_samples"
        return result
    if (
        b_samples.ndim != 1
        or c_samples.ndim != 1
        or observed_samples.ndim != 1
        or b_samples.size != c_samples.size
    ):
        result["reason"] = "invalid_bootstrap_samples"
        return result
    if min(b_samples.size, observed_samples.size) < 2:
        result["reason"] = "insufficient_bootstrap_replicates"
        return result
    if (
        np.any(~np.isfinite(b_samples))
        or np.any(~np.isfinite(c_samples))
        or np.any(~np.isfinite(observed_samples))
        or np.any(b_samples <= 0.0)
        or np.any(c_samples <= 0.0)
        or np.any(observed_samples <= 0.0)
    ):
        result["reason"] = "invalid_bootstrap_samples"
        return result

    combined_count = min(b_samples.size, observed_samples.size)
    delta_loss = (
        np.abs(c_samples[:combined_count] - observed_samples[:combined_count])
        - np.abs(b_samples[:combined_count] - observed_samples[:combined_count])
    )
    low, high = np.quantile(delta_loss, (0.025, 0.975))
    result.update(
        ci_low=float(low),
        ci_high=float(high),
        successful_combined_replicates=int(combined_count),
        calibration_prediction_uncertainty_included=True,
        validation_observation_uncertainty_included=True,
    )
    tolerance = ERROR_TIE_TOLERANCE_DELTA_E00
    if shared_w_sensitivity.get("comparison_resolved") is not True:
        result.update(status="indeterminate", reason="shared_w_sensitivity_unresolved")
    elif shared_w_sensitivity["free_w_preferred"]:
        result.update(status="indeterminate", reason="shared_w_sensitivity_warning")
    elif high < -tolerance:
        result.update(status="resolved_C96_lower_error", reason="bootstrap_ci_below_negative_tolerance")
    elif low > tolerance:
        result.update(status="resolved_B96_lower_error", reason="bootstrap_ci_above_positive_tolerance")
    elif low >= -tolerance and high <= tolerance:
        result.update(status="equivalent_within_tolerance", reason="bootstrap_ci_within_equivalence_band")
    else:
        result.update(status="indeterminate", reason="bootstrap_ci_overlaps_decision_boundary")
    return result


def _model_comparisons(
    predictions: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    shared_w_sensitivity: Mapping[str, Any],
    observed_bootstrap_samples: np.ndarray,
) -> dict[str, Any]:
    by_model = {row["model"]: row for row in predictions}
    required = {"B96", "B288", "C96", "D288"}
    if set(by_model) != required:
        raise ValidationAnalysisError("모델 비교에는 B96/B288/C96/D288 예측이 모두 필요합니다.")
    b96 = float(by_model["B96"]["c80_sim"])
    c96 = float(by_model["C96"]["c80_sim"])
    d288 = float(by_model["D288"]["c80_sim"])
    b288 = float(by_model["B288"]["c80_sim"])
    observed = validation["pooled_c80"]
    low = validation["pooled_ci_low"]
    high = validation["pooled_ci_high"]
    separation = abs(b96 - c96)
    delta_loss = _delta_loss_bootstrap(
        by_model, observed, observed_bootstrap_samples, shared_w_sensitivity
    )
    if observed is None:
        half_width = None
        resolution = False
        primary_outcome = "validation_unstable"
        errors: dict[str, float | None] = {model: None for model in required}
    else:
        observed = float(observed)
        low = float(low)
        high = float(high)
        half_width = (high - low) / 2.0
        resolution = False
        errors = {
            "B96": abs(b96 - observed),
            "C96": abs(c96 - observed),
            "D288": abs(d288 - observed),
            "B288": abs(b288 - observed),
        }
        outcome_by_status = {
            "resolved_C96_lower_error": "C96_lower_error",
            "resolved_B96_lower_error": "B96_lower_error",
            "equivalent_within_tolerance": "equal_within_tolerance",
            "indeterminate": "model_indistinguishable",
            "unresolvable": "comparison_unresolvable",
        }
        primary_outcome = outcome_by_status[delta_loss["status"]]
        resolution = delta_loss["status"] in {
            "resolved_C96_lower_error",
            "resolved_B96_lower_error",
            "equivalent_within_tolerance",
        }
    return {
        "stroke_width_px": int(validation["stroke_width_px"]),
        "stroke_width_deg": float(validation["stroke_width_deg"]),
        "observed_pooled_c80": observed,
        "observed_ci_low": low,
        "observed_ci_high": high,
        "primary": {
            "first_model": "B96",
            "first_prediction": b96,
            "first_abs_error": errors["B96"],
            "second_model": "C96",
            "second_prediction": c96,
            "second_abs_error": errors["C96"],
        },
        "secondary": {
            "first_model": "D288",
            "first_prediction": d288,
            "first_abs_error": errors["D288"],
            "second_model": "B288",
            "second_prediction": b288,
            "second_abs_error": errors["B288"],
        },
        "prediction_separation": separation,
        "observed_ci_half_width": half_width,
        "resolution_check": resolution,
        "primary_outcome": primary_outcome,
        "primary_delta_loss": delta_loss,
        "shared_w_sensitivity": dict(shared_w_sensitivity),
        "metadata": {
            "comparison_unit": "absolute_delta_e00_error",
            "resolution_rule": (
                "paired_calibration_and_validation_bootstrap_delta_loss_95ci"
            ),
            "error_tie_tolerance_delta_e00": ERROR_TIE_TOLERANCE_DELTA_E00,
            "inference_scope": (
                "paired-bootstrap-uncertainty-propagation-not-population-inference"
            ),
            "model_or_width_reselected_after_validation": False,
        },
    }
def analyze_validation(
    config_path: str | Path,
    calibration_fits_path: str | Path,
    validation_plan_path: str | Path,
    trials_path: str | Path,
    *,
    validation_manifest_path: str | Path | None = None,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260915,
    allow_synthetic: bool = False,
    confirm_human_after_review: bool = False,
    review_record_path: str | Path | None = None,
) -> dict[str, Any]:
    if isinstance(bootstrap_replicates, bool) or bootstrap_replicates < 2:
        raise ValidationAnalysisError("bootstrap_replicates는 2 이상이어야 합니다.")
    config, config_sha = _read_json(config_path)
    calibration, calibration_sha = _read_json(calibration_fits_path)
    plan, plan_raw_sha = _read_json(validation_plan_path)
    try:
        validate_plan(plan)
    except ValidationDesignError as error:
        raise ValidationAnalysisError(str(error)) from error
    try:
        validate_calibration_artifact(calibration)
    except Exception as error:
        raise ValidationAnalysisError(f"calibration artifact 검증 실패: {error}") from error
    current_calibration_analysis_sha256 = hashlib.sha256(
        Path(__file__).with_name("analyze_calibration.py").read_bytes()
    ).hexdigest()
    if calibration.get("analysis_script_sha256") != current_calibration_analysis_sha256:
        raise ValidationAnalysisError(
            "calibration artifact가 현재 analyze_calibration.py와 결합되지 않았습니다."
        )
    origin = plan.get("data_origin")
    if origin == "synthetic" and not allow_synthetic:
        raise ValidationAnalysisError("합성 검증 자료는 allow_synthetic=True가 필요합니다.")
    if origin not in {"human", "synthetic"}:
        raise ValidationAnalysisError("validation plan data_origin이 잘못됐습니다.")
    if origin != calibration.get("data_origin"):
        raise ValidationAnalysisError("plan과 calibration의 data_origin이 다릅니다.")
    if plan.get("config_sha256") != config_sha:
        raise ValidationAnalysisError("plan과 experiment config raw SHA-256이 다릅니다.")
    if plan.get("calibration_model_fits_sha256") != calibration_sha:
        raise ValidationAnalysisError("plan과 calibration artifact raw SHA-256이 다릅니다.")
    if plan.get("source_git_commit") != calibration.get("source", {}).get("git_commit"):
        raise ValidationAnalysisError("plan과 calibration의 자료 수집 Git commit이 다릅니다.")
    participant = calibration["participants"][0]
    if participant["participant_id"] != plan["participant_id"]:
        raise ValidationAnalysisError("plan과 calibration의 participant_id가 다릅니다.")
    records, trials_sha = _read_jsonl(trials_path)
    manifest_sha256: str | None = None
    if origin == "human":
        if validation_manifest_path is None:
            raise ValidationAnalysisError(
                "사람 validation 분석에는 --validation-manifest가 필요합니다."
            )
        manifest_sha256 = _validate_validation_manifest(
            validation_manifest_path,
            config=config,
            config_sha256=config_sha,
            calibration=calibration,
            calibration_sha256=calibration_sha,
            plan=plan,
            plan_raw_sha256=plan_raw_sha,
            records=records,
        )
    confirmation_requested = confirm_human_after_review or review_record_path is not None
    if confirmation_requested and origin != "human":
        raise ValidationAnalysisError("합성 결과는 confirmed=true로 만들 수 없습니다.")
    if origin == "human" and bootstrap_replicates < MINIMUM_HUMAN_BOOTSTRAP_REPLICATES:
        raise ValidationAnalysisError(
            "사람 validation 분석은 bootstrap_replicates를 1000 이상으로 "
            "고정해야 합니다."
        )
    if origin == "human":
        try:
            expected_trials_path = canonical_journal_path(
                config, "validation", PROJECT_ROOT
            )
        except CollectionBlockedError as error:
            raise ValidationAnalysisError(str(error)) from error
        if os.path.normcase(str(Path(trials_path).resolve())) != os.path.normcase(
            str(expected_trials_path)
        ):
            raise ValidationAnalysisError(
                "사람 validation 분석의 --trials는 config에 동결된 canonical "
                f"journal이어야 합니다: {expected_trials_path}"
            )
    if confirm_human_after_review and review_record_path is None:
        raise ValidationAnalysisError(
            "--confirm-human-after-review 플래그만으로는 확정할 수 없습니다. "
            "정확한 입력 hash를 담은 --review-record가 필요합니다."
        )
    confirmation_review: dict[str, Any] | None = None
    grouped = _validate_and_group_trials(
        records, config=config, plan=plan, human=(origin == "human")
    )
    flattened = [row for block in grouped for row in block]
    if review_record_path is not None:
        assert manifest_sha256 is not None  # origin=human was established above
        completed_timestamps: list[datetime] = []
        for row in flattened:
            raw_timestamp = row.get("timestamp_utc")
            if not isinstance(raw_timestamp, str) or not raw_timestamp.endswith("Z"):
                raise ValidationAnalysisError(
                    "완료 validation 시행의 timestamp_utc가 잘못됐습니다."
                )
            try:
                completed_timestamps.append(
                    datetime.fromisoformat(raw_timestamp[:-1] + "+00:00")
                )
            except ValueError as error:
                raise ValidationAnalysisError(
                    "완료 validation 시행의 timestamp_utc 형식이 잘못됐습니다."
                ) from error
        confirmation_review = _validate_confirmation_review(
            review_record_path,
            participant_id=participant["participant_id"],
            config_sha256=config_sha,
            source_trials_sha256=trials_sha,
            validation_plan_sha256=plan["plan_sha256"],
            validation_manifest_sha256=manifest_sha256,
            latest_completed_timestamp_utc=max(completed_timestamps),
        )
    x = np.asarray([row["delta_e00_simulated"] for row in flattened], dtype=float)
    y = np.asarray([row["is_correct"] for row in flattened], dtype=int)
    block_index = np.concatenate(
        [np.full(len(block), index, dtype=int) for index, block in enumerate(grouped)]
    )
    lapse = float(config["observer_profile"]["fixed_lapse"])
    joint_fit = fit_shared_w(x, y, block_index, lapse=lapse)
    if not joint_fit.converged:
        raise ValidationAnalysisError("세 block C80+공통 w joint MLE가 수렴하지 않았습니다.")
    joint_bootstrap = bootstrap_shared_w(
        joint_fit,
        x,
        y,
        block_index,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        minimum_convergence_rate=MINIMUM_BOOTSTRAP_CONVERGENCE_RATE,
    )
    if not joint_bootstrap.converged or joint_bootstrap.w_interval is None:
        raise ValidationAnalysisError(
            "세 block C80+공통 w joint bootstrap 수렴률이 기준에 못 미칩니다."
        )
    fitted: list[dict[str, Any]] = []
    for index, block in enumerate(grouped):
        c80 = joint_fit.c80_by_block[index]
        c80_ci = joint_bootstrap.c80_intervals[index]
        w_ci = joint_bootstrap.w_interval
        if not c80_ci[0] <= c80 <= c80_ci[1] or not w_ci[0] <= joint_fit.w <= w_ci[1]:
            raise ValidationAnalysisError("joint MLE 점추정치가 bootstrap 95% 구간 밖입니다.")
        presented_low = float(min(row["delta_e00_simulated"] for row in block))
        presented_high = float(max(row["delta_e00_simulated"] for row in block))
        range_bracketed = (
            presented_low
            <= float(c80_ci[0])
            <= float(c80)
            <= float(c80_ci[1])
            <= presented_high
        )
        fitted.append(
            {
                "block_id": block[0]["validation_block_id"],
                "session_id": block[0]["session_id"],
                "completed_trial_count": TRIALS_PER_BLOCK,
                "source_trials_sha256": _canonical_block_sha256(block),
                "c80_sim": float(c80),
                "c80_ci": {"ci_low": float(c80_ci[0]), "ci_high": float(c80_ci[1])},
                "presented_delta_e00_min": presented_low,
                "presented_delta_e00_max": presented_high,
                "range_bracketed": range_bracketed,
                "w": float(joint_fit.w),
                "w_ci": {"ci_low": float(w_ci[0]), "ci_high": float(w_ci[1])},
                "fixed_lapse": lapse,
                "fit_converged": True,
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_successful_replicates": int(
                    joint_bootstrap.successful_replicates
                ),
                "bootstrap_convergence_rate": float(joint_bootstrap.convergence_rate),
            }
        )
    log_covariance = np.cov(np.log(joint_bootstrap.c80_samples), rowvar=False, ddof=1)
    (
        cv,
        compatible,
        stable,
        pooled,
        pooled_low,
        pooled_high,
        pooled_samples,
        range_bracketed,
    ) = _pool_blocks(fitted, log_covariance, joint_bootstrap.c80_samples)
    shared_w_sensitivity = compare_shared_vs_free_w_aic(
        x, y, block_index, lapse=lapse
    )
    selected_width = plan["selected_width"]
    predictions = _prediction_rows(participant, int(selected_width["stroke_width_px"]))
    validation_result = {
        "stroke_width_px": int(selected_width["stroke_width_px"]),
        "stroke_width_deg": float(selected_width["stroke_width_deg"]),
        "blocks": fitted,
        "coefficient_of_variation": cv,
        "intervals_compatible": compatible,
        "range_bracketed": range_bracketed,
        "stable": stable,
        "pooled_c80": pooled,
        "pooled_ci_low": pooled_low,
        "pooled_ci_high": pooled_high,
    }
    model_comparisons = _model_comparisons(
        predictions,
        validation_result,
        shared_w_sensitivity,
        pooled_samples,
    )
    confirmed = confirmation_review is not None
    if confirmed and not stable:
        raise ValidationAnalysisError("불안정한 세 블록은 confirmed=true로 만들 수 없습니다.")
    if (
        confirmed
        and model_comparisons["primary_delta_loss"]["status"] == "unresolvable"
    ):
        raise ValidationAnalysisError(
            "교정·검증 bootstrap 불확실성을 결합할 수 없어 confirmed=true로 만들 수 없습니다."
        )
    output_participant = {
        "participant_id": participant["participant_id"],
        "cvd_type": participant["cvd_type"],
        "severity": float(participant["machado_severity"]),
        "severity_source": participant["severity_source"],
        "severity_session_id": participant["severity_session_id"],
        "fixed_lapse": float(participant["fixed_lapse"]),
        "lapse_source": participant["lapse_source"],
        "lapse_session_id": participant["lapse_session_id"],
        "confirmed": confirmed,
        "confirmation_review": confirmation_review,
        "calibration_width_slope_sensitivity": dict(
            participant["width_slope_sensitivity"]
        ),
        "predictions": predictions,
        "validation": validation_result,
        "model_comparisons": model_comparisons,
    }
    analysis_implementation_hashes = implementation_hashes("validation_analysis")
    common = {
        "coordinate": "post_machado_delta_e00",
        "formula_version": "direct-c80-4afc-v1",
        "config_version": config["config_version"],
        "config_sha256": config_sha,
        "git_commit": calibration["source"]["git_commit"],
        "source_trials_sha256": trials_sha,
        "source_validation_plan_sha256": plan["plan_sha256"],
        **(
            {"source_stimulus_manifest_sha256": manifest_sha256}
            if manifest_sha256 is not None
            else {}
        ),
        "analysis_script_sha256": analysis_implementation_hashes[
            "src/analyze_validation.py"
        ],
        "implementation_hashes": analysis_implementation_hashes,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "bootstrap_seed": bootstrap_seed,
        "participants": [output_participant],
    }
    if origin == "human":
        payload = {"schema_version": "1.0", "data_origin": "human", **common}
        _validate_schema(payload, HUMAN_THRESHOLDS_SCHEMA, "participant_thresholds")
        return payload
    return {
        "schema_version": "0.1-development",
        "artifact_type": "synthetic_validation_analysis",
        "data_origin": "synthetic",
        "research_conclusion_allowed": False,
        "validation_plan_raw_file_sha256": plan_raw_sha,
        **common,
    }


def _write_json_atomic(payload: Mapping[str, Any], output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise ValidationAnalysisError(f"출력 파일이 이미 있습니다: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
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
    parser = argparse.ArgumentParser(description="독립 검증 3×64회에서 C80 재현성과 통합값을 분석합니다.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-fits", type=Path, required=True)
    parser.add_argument("--validation-plan", type=Path, required=True)
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        help="사람 validation 응답을 생성한 compiled PNG manifest (human에는 필수)",
    )
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument(
        "--review-record",
        type=Path,
        help="입력 hash와 검토자 가명 ID·시각·메모를 담은 validation 검토 JSON",
    )
    parser.add_argument(
        "--confirm-human-after-review",
        action="store_true",
        help="호환용 표지이며 단독 사용은 차단됩니다. --review-record가 반드시 필요합니다.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = analyze_validation(
            args.config, args.calibration_fits, args.validation_plan, args.trials,
            validation_manifest_path=args.validation_manifest,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            allow_synthetic=args.allow_synthetic,
            confirm_human_after_review=args.confirm_human_after_review,
            review_record_path=args.review_record,
        )
        _write_json_atomic(payload, args.output, overwrite=args.overwrite)
    except (OSError, TypeError, ValueError) as error:
        print(f"검증 분석 차단: {error}", file=sys.stderr)
        return 2
    stable = payload["participants"][0]["validation"]["stable"]
    return 0 if stable else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ValidationAnalysisError", "analyze_validation", "main"]
