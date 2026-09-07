"""한정훈의 C80 결과를 이창민의 보정 임계값으로 안전하게 연결한다."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

from config import ConfusionThresholds
from data_contracts import validate_file


class ThresholdAdapterError(ValueError):
    """C80 결과가 없거나 연구용으로 확정되지 않았을 때 사용한다."""


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

    payload = validate_file("participant_thresholds", path)
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
]
