"""팀원이 채워 넣는 입력 파일을 읽는다.

②단계는 수치만 만들고 임계값을 정하지 않는다. 그 임계값은 두 곳에서
들어온다.

* 한정훈 → `data/inputs/personal_params.json`
  피험자별 `type`, `severity`, 조건별로 잰 역치.
* 이창민 → `data/inputs/thresholds.json`
  보정 판단에 쓸 ΔE00 임계값 묶음.

이 모듈은 그 파일을 읽고 형식을 검사하기만 한다. `measure_pairs.py`는
이 모듈을 부르지 않는다. 측정 결과에 판정을 섞지 않기 위해서다.
"""

from __future__ import annotations

import json
from pathlib import Path

from config import ConfusionThresholds


VALID_CVD_TYPES = ("protan", "deutan")

class InputError(ValueError):
    """입력 파일 형식이 맞지 않을 때 낸다. 메시지에 고칠 곳을 적는다."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InputError(message)


def _load_json(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        raise InputError(
            f"입력 파일이 없습니다: {file_path}\n"
            f"같은 폴더의 .example.json을 복사해 채우세요."
        )
    with open(file_path, encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as error:
            raise InputError(
                f"{file_path}의 JSON 형식이 깨졌습니다: {error}"
            ) from error


def _check_number(value, name: str, low: float, high: float) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name}은 숫자여야 합니다. 받은 값: {value!r}",
    )
    number = float(value)
    _require(
        low <= number <= high,
        f"{name}은 {low}~{high} 범위여야 합니다. 받은 값: {number}",
    )
    return number


def load_personal_params(
    path: str | Path = "../data/inputs/personal_params.json",
) -> list[dict]:
    """한정훈이 채운 피험자별 개인 파라미터를 읽는다.

    Returns:
        피험자 기록 목록. 각 기록은 ``participant_id``, ``cvd_type``,
        ``severity``, ``threshold_conditions``를 가진다.
    """
    payload = _load_json(path)
    participants = payload.get("participants")
    _require(
        isinstance(participants, list) and participants,
        "participants가 비어 있습니다. 피험자를 최소 한 명 넣으세요.",
    )

    records = []
    for index, entry in enumerate(participants):
        label = entry.get("participant_id", f"{index}번째 피험자")
        _require(
            entry.get("cvd_type") in VALID_CVD_TYPES,
            f"{label}: cvd_type은 {VALID_CVD_TYPES} 중 하나여야 합니다. "
            f"받은 값: {entry.get('cvd_type')!r}",
        )
        severity = _check_number(
            entry.get("severity"), f"{label}: severity", 0.0, 1.0
        )

        conditions = entry.get("threshold_conditions", [])
        _require(
            isinstance(conditions, list),
            f"{label}: threshold_conditions는 목록이어야 합니다.",
        )
        checked = []
        for condition in conditions:
            threshold = condition.get("threshold")
            if threshold is None:
                # 아직 안 잰 조건은 그대로 둔다. 실험 전에는 비어 있다.
                checked.append({**condition, "threshold": None})
                continue
            checked.append(
                {
                    "stroke_width_px": _check_number(
                        condition.get("stroke_width_px"),
                        f"{label}: stroke_width_px",
                        0.0,
                        10_000.0,
                    ),
                    "distance_px": _check_number(
                        condition.get("distance_px"),
                        f"{label}: distance_px",
                        0.0,
                        100_000.0,
                    ),
                    "threshold": _check_number(
                        threshold, f"{label}: threshold", 0.0, 200.0
                    ),
                }
            )

        reference_raw = entry.get("reference_threshold")
        reference_threshold = (
            None
            if reference_raw is None
            else _check_number(
                reference_raw,
                f"{label}: reference_threshold",
                1e-12,
                200.0,
            )
        )
        records.append(
            {
                "participant_id": str(label),
                "cvd_type": entry["cvd_type"],
                "severity": severity,
                "reference_threshold": reference_threshold,
                "threshold_conditions": checked,
                "notes": entry.get("notes", ""),
            }
        )
    return records


def to_cvd_conditions(
    participants: list[dict],
) -> tuple[tuple[str, float], ...]:
    """피험자 목록을 ``measure_image(cvd_conditions=...)`` 형태로 바꾼다.

    같은 (type, severity) 조합은 한 번만 넣는다. 열이 중복될 이유가 없다.
    """
    seen = []
    for participant in participants:
        pair = (participant["cvd_type"], round(participant["severity"], 2))
        if pair not in seen:
            seen.append(pair)
    return tuple(seen)


def measured_conditions(participant: dict) -> list[dict]:
    """역치를 실제로 잰 조건만 골라 환산 비율을 붙인다.

    환산 비율 = (그 조건에서 잰 역치) / (맞붙여 잰 기준 역치).
    1보다 크면 그 조건에서는 색차가 더 벌어져야 구별된다는 뜻이다.
    """
    completed = [
        condition
        for condition in participant["threshold_conditions"]
        if condition.get("threshold") is not None
    ]
    if not completed:
        return []
    reference = participant.get("reference_threshold")
    _require(
        reference is not None and reference > 0.0,
        "측정된 조건의 환산비를 계산하려면 해당 참가자·세션에서 잰 "
        "reference_threshold를 명시해야 합니다. 0.62를 기본값으로 "
        "대입하지 않습니다.",
    )
    rows = []
    for condition in completed:
        rows.append(
            {
                **condition,
                "scaling_ratio": condition["threshold"] / reference,
            }
        )
    return rows


def load_thresholds(
    path: str | Path = "../data/inputs/thresholds.json",
) -> tuple[ConfusionThresholds, dict]:
    """이창민 쪽에서 쓸 ΔE00 임계값 묶음을 읽는다.

    Returns:
        ``(ConfusionThresholds, 메타정보)``. 메타정보의 ``confirmed``가
        False면 아직 실험으로 확정되지 않은 개발용 값이라는 뜻이다.
    """
    payload = _load_json(path)
    thresholds = ConfusionThresholds(
        original_min=_check_number(
            payload.get("original_min"), "original_min", 0.0, 200.0
        ),
        simulated_max=_check_number(
            payload.get("simulated_max"), "simulated_max", 0.0, 200.0
        ),
        target=_check_number(payload.get("target"), "target", 0.0, 200.0),
    )
    meta = {
        "confirmed": bool(payload.get("confirmed", False)),
        "source": payload.get("source", ""),
        "updated_on": payload.get("updated_on", ""),
    }
    return thresholds, meta


def describe_inputs(
    personal_path: str | Path = "../data/inputs/personal_params.json",
    thresholds_path: str | Path = "../data/inputs/thresholds.json",
) -> str:
    """지금 어떤 입력이 들어와 있고 무엇이 비었는지 한눈에 보여준다."""
    lines = ["팀 입력 상태", ""]

    try:
        participants = load_personal_params(personal_path)
    except InputError as error:
        lines.append(f"[한정훈] 읽지 못함 — {error}")
    else:
        lines.append(f"[한정훈] 피험자 {len(participants)}명")
        for participant in participants:
            measured = measured_conditions(participant)
            total = len(participant["threshold_conditions"])
            lines.append(
                f"  - {participant['participant_id']}: "
                f"{participant['cvd_type']} "
                f"severity={participant['severity']:.2f}, "
                f"역치 {len(measured)}/{total}개 측정됨"
            )
            for row in measured:
                lines.append(
                    f"      굵기 {row['stroke_width_px']:g}px "
                    f"거리 {row['distance_px']:g}px → "
                    f"역치 {row['threshold']:.2f} "
                    f"(환산 비율 {row['scaling_ratio']:.2f}배)"
                )

    lines.append("")
    try:
        thresholds, meta = load_thresholds(thresholds_path)
    except InputError as error:
        lines.append(f"[이창민] 읽지 못함 — {error}")
    else:
        state = "실험으로 확정됨" if meta["confirmed"] else "개발용 임시값"
        lines.append(
            f"[이창민] original_min={thresholds.original_min:g} "
            f"simulated_max={thresholds.simulated_max:g} "
            f"target={thresholds.target:g} — {state}"
        )
        if not meta["confirmed"]:
            lines.append(
                "  주의: confirmed=false다. 이 값으로 낸 결과를 연구 "
                "결과로 쓰면 안 된다."
            )
    return "\n".join(lines)


__all__ = [
    "InputError",
    "VALID_CVD_TYPES",
    "describe_inputs",
    "load_personal_params",
    "load_thresholds",
    "measured_conditions",
    "to_cvd_conditions",
]
