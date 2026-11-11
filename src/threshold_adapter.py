"""한정훈의 C80 결과를 이창민의 보정 임계값으로 안전하게 연결한다."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

from config import ConfusionThresholds
from data_contracts import validate_file
from implementation_provenance import (
    ImplementationProvenanceError,
    verify_implementation_hashes,
)


class ThresholdAdapterError(ValueError):
    """C80 결과가 없거나 연구용으로 확정되지 않았을 때 사용한다."""


MINIMUM_HUMAN_BOOTSTRAP_REPLICATES = 1000
MINIMUM_BOOTSTRAP_CONVERGENCE_RATE = 0.90


@dataclass(frozen=True)
class ThresholdEstimate:
    """특정 참가자·모델·굵기의 C80 추정치와 출처."""

    participant_id: str
    cvd_type: str
    severity: float
    severity_source: str
    model: str
    stroke_width_px: int
    stroke_width_deg: float
    c80_sim: float
    ci_low: float
    ci_high: float
    value_type: str
    confirmed: bool
    data_origin: str
    schema_version: str

    def target(self, policy: str = "point") -> float:
        """주 분석 점추정값 또는 민감도 분석 CI 상한을 반환한다."""
        if policy == "point":
            return self.c80_sim
        if policy == "ci_high":
            return self.ci_high
        raise ThresholdAdapterError("target policy는 'point' 또는 'ci_high'여야 합니다.")


def _positive_number(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ThresholdAdapterError(f"{name}은 0보다 큰 유한한 숫자여야 합니다.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ThresholdAdapterError(f"{name}은 0보다 큰 유한한 숫자여야 합니다.")
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ThresholdAdapterError(f"JSON 객체에 중복 키가 있습니다: {key}")
        result[key] = value
    return result


def _load_human_thresholds(path: Path) -> dict[str, Any]:
    """엄격한 human 1.0 스키마와 핵심 의미 규칙을 함께 검사한다."""
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "human"
        / "participant_thresholds.schema.json"
    )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
        schema = json.loads(
            schema_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ThresholdAdapterError(f"human C80 JSON을 읽을 수 없습니다: {error}") from error
    if not isinstance(payload, dict) or not isinstance(schema, dict):
        raise ThresholdAdapterError("human C80과 스키마의 최상위는 JSON 객체여야 합니다.")
    try:
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(
                schema, format_checker=FormatChecker()
            ).iter_errors(payload),
            key=lambda item: list(item.path),
        )
    except SchemaError as error:
        raise ThresholdAdapterError(f"human C80 스키마를 검사할 수 없습니다: {error}") from error
    if errors:
        details = "; ".join(
            f"{'.'.join(map(str, error.path)) or '$'}: {error.message}"
            for error in errors[:8]
        )
        raise ThresholdAdapterError(f"human C80 1.0 스키마 오류: {details}")

    participant_ids: set[str] = set()
    for participant in payload["participants"]:
        participant_id = participant["participant_id"]
        if participant_id in participant_ids:
            raise ThresholdAdapterError(
                f"human C80에 participant_id가 중복됩니다: {participant_id}"
            )
        participant_ids.add(participant_id)
        prediction_keys: set[tuple[str, int]] = set()
        for prediction in participant["predictions"]:
            key = (prediction["model"], int(prediction["stroke_width_px"]))
            if key in prediction_keys:
                raise ThresholdAdapterError(
                    f"{participant_id}: model/굵기 C80이 중복됩니다: {key}"
                )
            prediction_keys.add(key)
            if not (
                float(prediction["ci_low"])
                <= float(prediction["c80_sim"])
                <= float(prediction["ci_high"])
            ):
                raise ThresholdAdapterError(
                    f"{participant_id} {key}: C80이 신뢰구간 밖입니다."
                )
            requested = int(prediction["bootstrap_replicates"])
            successful = int(prediction["bootstrap_successful_replicates"])
            samples = prediction["bootstrap_c80_samples"]
            if successful > requested or len(samples) != successful:
                raise ThresholdAdapterError(
                    f"{participant_id} {key}: bootstrap 복제 수와 예측 표본 수가 다릅니다."
                )
            if participant["confirmed"] and (
                requested < MINIMUM_HUMAN_BOOTSTRAP_REPLICATES
                or successful
                < math.ceil(MINIMUM_BOOTSTRAP_CONVERGENCE_RATE * requested)
            ):
                raise ThresholdAdapterError(
                    f"{participant_id} {key}: confirmed human 예측은 bootstrap "
                    "1000회 이상과 90% 이상 성공 복제가 필요합니다."
                )
            expected_rate = successful / requested
            if not math.isclose(
                float(prediction["bootstrap_convergence_rate"]),
                expected_rate,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ThresholdAdapterError(
                    f"{participant_id} {key}: bootstrap 수렴률이 복제 수와 다릅니다."
                )
        sensitivity = participant["calibration_width_slope_sensitivity"]
        for field in (
            "shared_w_aic",
            "free_w_aic",
            "delta_aic_free_minus_shared",
        ):
            if not math.isfinite(float(sensitivity[field])):
                raise ThresholdAdapterError(
                    f"{participant_id}: calibration 공통-w 민감도 AIC는 유한수여야 합니다."
                )
        expected_delta = float(sensitivity["free_w_aic"]) - float(
            sensitivity["shared_w_aic"]
        )
        if not math.isclose(
            float(sensitivity["delta_aic_free_minus_shared"]),
            expected_delta,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ThresholdAdapterError(
                f"{participant_id}: calibration 공통-w delta AIC가 두 AIC와 다릅니다."
            )
        comparison_resolved = bool(
            sensitivity["shared_w_converged"] and sensitivity["free_w_converged"]
        )
        if sensitivity["comparison_resolved"] != comparison_resolved:
            raise ThresholdAdapterError(
                f"{participant_id}: calibration 공통-w 비교 해결 플래그가 수렴 상태와 다릅니다."
            )
        free_w_preferred = bool(
            comparison_resolved
            and float(sensitivity["free_w_aic"]) < float(sensitivity["shared_w_aic"])
        )
        if sensitivity["free_w_preferred"] != free_w_preferred:
            raise ThresholdAdapterError(
                f"{participant_id}: calibration free-w 선호 플래그가 AIC와 다릅니다."
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
            raise ThresholdAdapterError(
                f"{participant_id}: calibration 공통-w 해석이 수치와 다릅니다."
            )
        d384_rows = [
            row for row in participant["predictions"] if row["model"] == "D384"
        ]
        d384_ready = bool(
            len(d384_rows) == 1
            and d384_rows[0]["fit_converged"]
            and float(d384_rows[0]["bootstrap_convergence_rate"])
            >= MINIMUM_BOOTSTRAP_CONVERGENCE_RATE
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
            raise ThresholdAdapterError(
                f"{participant_id}: D384 beta 해석 가능 플래그가 공통-w 감사와 다릅니다."
            )
        block_ids: set[str] = set()
        session_ids: set[str] = set()
        validation = participant["validation"]
        validation_blocks = validation["blocks"]
        for block in validation_blocks:
            if block["block_id"] in block_ids or block["session_id"] in session_ids:
                raise ThresholdAdapterError(
                    f"{participant_id}: validation block/session ID가 중복됩니다."
                )
            block_ids.add(block["block_id"])
            session_ids.add(block["session_id"])
            if not (
                float(block["c80_ci"]["ci_low"])
                <= float(block["c80_sim"])
                <= float(block["c80_ci"]["ci_high"])
            ):
                raise ThresholdAdapterError(
                    f"{participant_id} {block['block_id']}: C80이 신뢰구간 밖입니다."
                )
            numerical_range_bracketed = (
                float(block["presented_delta_e00_min"])
                <= float(block["c80_ci"]["ci_low"])
                <= float(block["c80_sim"])
                <= float(block["c80_ci"]["ci_high"])
                <= float(block["presented_delta_e00_max"])
            )
            if bool(block["range_bracketed"]) != numerical_range_bracketed:
                raise ThresholdAdapterError(
                    f"{participant_id} {block['block_id']}: range_bracketed가 "
                    "제시 ΔE00 범위와 C80/CI 수치에 맞지 않습니다."
                )
            if not (
                float(block["w_ci"]["ci_low"])
                <= float(block["w"])
                <= float(block["w_ci"]["ci_high"])
            ):
                raise ThresholdAdapterError(
                    f"{participant_id} {block['block_id']}: w가 신뢰구간 밖입니다."
                )
            requested = int(block["bootstrap_replicates"])
            successful = int(block["bootstrap_successful_replicates"])
            if successful > requested:
                raise ThresholdAdapterError(
                    f"{participant_id} {block['block_id']}: 성공 bootstrap 수가 요청 수보다 큽니다."
                )
            expected_rate = successful / requested
            if not math.isclose(
                float(block["bootstrap_convergence_rate"]),
                expected_rate,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ThresholdAdapterError(
                    f"{participant_id} {block['block_id']}: bootstrap 성공 수와 수렴률이 다릅니다."
                )
            if participant["confirmed"] and (
                requested < MINIMUM_HUMAN_BOOTSTRAP_REPLICATES
                or successful
                < math.ceil(MINIMUM_BOOTSTRAP_CONVERGENCE_RATE * requested)
            ):
                raise ThresholdAdapterError(
                    f"{participant_id} {block['block_id']}: confirmed human 검증은 "
                    "bootstrap 1000회 이상과 90% 이상 성공 복제가 필요합니다."
                )
        all_block_ranges_bracketed = all(
            bool(block["range_bracketed"]) for block in validation_blocks
        )
        if bool(validation["range_bracketed"]) != all_block_ranges_bracketed:
            raise ThresholdAdapterError(
                f"{participant_id}: validation.range_bracketed는 세 block의 "
                "range_bracketed 논리곱과 같아야 합니다."
            )
        block_c80 = [float(block["c80_sim"]) for block in validation_blocks]
        block_mean = sum(block_c80) / len(block_c80)
        reported_cv = float(validation["coefficient_of_variation"])
        calculated_cv = math.sqrt(
            sum((value - block_mean) ** 2 for value in block_c80)
            / (len(block_c80) - 1)
        ) / block_mean
        if not math.isclose(
            reported_cv, calculated_cv, rel_tol=1e-8, abs_tol=1e-10
        ):
            raise ThresholdAdapterError(
                f"{participant_id}: validation CV가 block C80에서 재계산한 값과 다릅니다."
            )
        calculated_compatible = max(
            float(block["c80_ci"]["ci_low"]) for block in validation_blocks
        ) <= min(
            float(block["c80_ci"]["ci_high"]) for block in validation_blocks
        )
        if bool(validation["intervals_compatible"]) != calculated_compatible:
            raise ThresholdAdapterError(
                f"{participant_id}: intervals_compatible가 block CI와 다릅니다."
            )
        if participant["confirmed"]:
            review = participant.get("confirmation_review")
            if not isinstance(review, Mapping):
                raise ThresholdAdapterError(
                    f"{participant_id}: confirmed human 결과에 결박된 review record가 없습니다."
                )
            review_expected = {
                "participant_id": participant_id,
                "config_sha256": payload.get("config_sha256"),
                "source_trials_sha256": payload.get("source_trials_sha256"),
                "validation_plan_sha256": payload.get(
                    "source_validation_plan_sha256"
                ),
                "validation_manifest_sha256": payload.get(
                    "source_stimulus_manifest_sha256"
                ),
            }
            for field, expected in review_expected.items():
                if review.get(field) != expected:
                    raise ThresholdAdapterError(
                        f"{participant_id}: confirmation_review.{field}가 "
                        "participant_thresholds provenance와 다릅니다."
                    )
            if not all_block_ranges_bracketed:
                raise ThresholdAdapterError(
                    f"{participant_id}: confirmed human 검증의 모든 block C80/CI는 "
                    "실제 제시 ΔE00 범위 안에 있어야 합니다."
                )
            if not participant["validation"]["stable"]:
                raise ThresholdAdapterError(
                    f"{participant_id}: validation이 불안정한데 confirmed=true입니다."
                )
            delta_loss = participant["model_comparisons"].get("primary_delta_loss", {})
            if delta_loss.get("status") == "unresolvable":
                raise ThresholdAdapterError(
                    f"{participant_id}: 교정·검증 bootstrap 불확실성을 결합하지 "
                    "못한 결과는 confirmed=true일 수 없습니다."
                )
            if any(
                not row["fit_converged"]
                or float(row["bootstrap_convergence_rate"]) < 0.90
                for row in participant["predictions"]
            ):
                raise ThresholdAdapterError(
                    f"{participant_id}: 수렴하지 않은 예측을 confirmed=true로 표시했습니다."
                )
    try:
        verify_implementation_hashes(
            payload.get("implementation_hashes"),
            "validation_analysis",
            label="participant_thresholds",
        )
    except ImplementationProvenanceError as error:
        raise ThresholdAdapterError(str(error)) from error
    if (
        payload.get("analysis_script_sha256")
        != payload["implementation_hashes"].get("src/analyze_validation.py")
    ):
        raise ThresholdAdapterError(
            "participant_thresholds: analysis_script_sha256와 implementation_hashes의 "
            "analyze_validation.py 값이 다릅니다."
        )
    return payload


def load_threshold_payload(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    try:
        header = json.loads(
            file_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ThresholdAdapterError(f"C80 JSON을 읽을 수 없습니다: {error}") from error
    if not isinstance(header, dict):
        raise ThresholdAdapterError("C80 JSON 최상위는 객체여야 합니다.")
    if header.get("schema_version") == "1.0" or header.get("data_origin") == "human":
        if header.get("schema_version") != "1.0" or header.get("data_origin") != "human":
            raise ThresholdAdapterError(
                "human C80은 schema_version=1.0과 data_origin=human을 함께 써야 합니다."
            )
        return _load_human_thresholds(file_path)
    return validate_file("participant_thresholds", file_path)


def load_threshold_estimate(
    path: str | Path,
    *,
    participant_id: str,
    model: str,
    stroke_width_px: int,
    development_mode: bool = False,
) -> ThresholdEstimate:
    """JSON에서 정확히 일치하는 C80 하나를 찾는다.

    기본 모드에서는 미확정 자료와 synthetic 자료를 모두 거부한다. 현재
    ``0.1-draft`` 예제를 연결 시험에 쓰려면 ``development_mode=True``를
    명시해야 한다. 해당 굵기가 없을 때 보간하거나 임시값으로 대체하지 않는다.
    """
    if not isinstance(development_mode, bool):
        raise TypeError("development_mode는 bool이어야 합니다.")
    if isinstance(stroke_width_px, bool) or not isinstance(stroke_width_px, int):
        raise TypeError("stroke_width_px는 1 이상의 정수여야 합니다.")
    if stroke_width_px < 1:
        raise ValueError("stroke_width_px는 1 이상의 정수여야 합니다.")

    payload = load_threshold_payload(path)
    participants = [
        row for row in payload["participants"] if row["participant_id"] == participant_id
    ]
    if len(participants) != 1:
        raise ThresholdAdapterError(
            f"participant_id={participant_id!r}인 참가자를 정확히 하나 찾지 못했습니다."
        )
    participant = participants[0]
    if not development_mode:
        if payload["data_origin"] != "human":
            raise ThresholdAdapterError(
                "synthetic C80은 연구 보정에 사용할 수 없습니다. "
                "가짜 연결 시험에서만 development_mode=True를 쓰세요."
            )
        if not participant["confirmed"]:
            raise ThresholdAdapterError("confirmed=false인 C80은 연구 보정에 쓸 수 없습니다.")

    matches = [
        row
        for row in participant["predictions"]
        if row["model"] == model and row["stroke_width_px"] == stroke_width_px
    ]
    if len(matches) != 1:
        raise ThresholdAdapterError(
            f"model={model!r}, stroke_width_px={stroke_width_px}인 C80을 "
            "정확히 하나 찾지 못했습니다. 보간이나 임시값 대체는 하지 않습니다."
        )
    prediction = matches[0]
    return ThresholdEstimate(
        participant_id=participant["participant_id"],
        cvd_type=participant["cvd_type"],
        severity=float(participant["severity"]),
        severity_source=participant["severity_source"],
        model=prediction["model"],
        stroke_width_px=int(prediction["stroke_width_px"]),
        stroke_width_deg=float(prediction["stroke_width_deg"]),
        c80_sim=float(prediction["c80_sim"]),
        ci_low=float(prediction["ci_low"]),
        ci_high=float(prediction["ci_high"]),
        value_type=prediction["value_type"],
        confirmed=bool(participant["confirmed"]),
        data_origin=payload["data_origin"],
        schema_version=payload["schema_version"],
    )


def correction_thresholds(
    estimate: ThresholdEstimate,
    *,
    original_min: Real,
    target_policy: str = "point",
    development_mode: bool = False,
) -> ConfusionThresholds:
    """C80 하나를 기존 baseline의 임계값 묶음으로 변환한다.

    ``simulated_max``와 ``target``에 같은 C80 정책값을 사용한다.
    ``original_min``은 의미가 다른 별도 실험 변수이므로 호출자가 반드시 준다.
    """
    if not isinstance(development_mode, bool):
        raise TypeError("development_mode는 bool이어야 합니다.")
    if not development_mode and (not estimate.confirmed or estimate.data_origin != "human"):
        raise ThresholdAdapterError("미확정 또는 synthetic C80은 연구 임계값으로 바꿀 수 없습니다.")
    original = _positive_number(original_min, "original_min")
    target = estimate.target(target_policy)
    try:
        return ConfusionThresholds(
            original_min=original,
            simulated_max=target,
            target=target,
        )
    except (TypeError, ValueError) as error:
        raise ThresholdAdapterError(
            "C80과 original_min의 관계가 baseline 제약을 만족하지 않습니다: "
            f"{error}"
        ) from error


__all__ = [
    "ThresholdAdapterError",
    "ThresholdEstimate",
    "correction_thresholds",
    "load_threshold_estimate",
    "load_threshold_payload",
]
