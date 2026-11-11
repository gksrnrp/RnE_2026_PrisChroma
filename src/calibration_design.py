"""Frozen mixed-allocation calibration design constants and validators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


CALIBRATION_WIDTH_TRIAL_COUNTS = (96, 192, 96)
CALIBRATION_TOTAL_TRIALS = sum(CALIBRATION_WIDTH_TRIAL_COUNTS)
REFERENCE_WIDTH_INDEX = 1
REFERENCE_WIDTH_TRIALS = CALIBRATION_WIDTH_TRIAL_COUNTS[REFERENCE_WIDTH_INDEX]
ALLOWED_CALIBRATION_BLOCK_COUNTS = (4, 6)
CALIBRATION_MODELS = ("B192", "B384", "C192", "D384")
CALIBRATION_RANDOMIZATION = (
    "balanced-mixed-width-session-blocks-within-block-shuffle"
)


def trial_counts_from_config(config: Mapping[str, Any]) -> tuple[int, int, int]:
    calibration = config.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("calibration은 객체여야 합니다.")
    raw = calibration.get("trials_per_width")
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, (str, bytes))
        or tuple(raw) != CALIBRATION_WIDTH_TRIAL_COUNTS
    ):
        raise ValueError(
            "calibration.trials_per_width는 굵기 오름차순으로 [96, 192, 96]이어야 합니다."
        )
    return CALIBRATION_WIDTH_TRIAL_COUNTS


def block_count_from_config(config: Mapping[str, Any]) -> int:
    calibration = config.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("calibration은 객체여야 합니다.")
    value = calibration.get("session_block_count")
    if isinstance(value, bool) or value not in ALLOWED_CALIBRATION_BLOCK_COUNTS:
        raise ValueError("calibration.session_block_count는 4 또는 6이어야 합니다.")
    return int(value)


def block_trial_count(config: Mapping[str, Any]) -> int:
    count = block_count_from_config(config)
    if CALIBRATION_TOTAL_TRIALS % count:
        raise ValueError("전체 교정 시행 수가 session block 수로 나누어지지 않습니다.")
    return CALIBRATION_TOTAL_TRIALS // count


def width_block_trial_counts(config: Mapping[str, Any]) -> tuple[int, int, int]:
    count = block_count_from_config(config)
    totals = trial_counts_from_config(config)
    if any(total % count for total in totals):
        raise ValueError("굵기별 시행 수가 session block 수로 나누어지지 않습니다.")
    return tuple(total // count for total in totals)


__all__ = [
    "ALLOWED_CALIBRATION_BLOCK_COUNTS",
    "CALIBRATION_MODELS",
    "CALIBRATION_RANDOMIZATION",
    "CALIBRATION_TOTAL_TRIALS",
    "CALIBRATION_WIDTH_TRIAL_COUNTS",
    "REFERENCE_WIDTH_INDEX",
    "REFERENCE_WIDTH_TRIALS",
    "block_count_from_config",
    "block_trial_count",
    "trial_counts_from_config",
    "width_block_trial_counts",
]
