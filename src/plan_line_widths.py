"""실험 화면에서 선 굵기(px)를 시각도로 바꿔 점검하는 명령줄 도구."""

from __future__ import annotations

import argparse
import json
import sys

from experiment_design import (
    SZAFIR_LINE_MAX_VISUAL_ANGLE_DEG,
    SZAFIR_LINE_MIN_VISUAL_ANGLE_DEG,
    geometric_validation_widths,
    pixel_pitch_mm,
    pixels_to_visual_angle_deg,
)


def build_width_report(
    widths_px: list[float],
    *,
    display_width_mm: float,
    horizontal_resolution_px: float,
    viewing_distance_mm: float,
) -> dict:
    """입력한 세 굵기의 물리 조건과 검증 후보를 JSON용 dict로 만든다."""
    if len(widths_px) != 3:
        raise ValueError("교정 굵기는 정확히 세 개를 입력해야 합니다.")
    candidates_px = geometric_validation_widths(widths_px)

    def describe(width_px: float) -> dict:
        angle = pixels_to_visual_angle_deg(
            width_px,
            display_width_mm=display_width_mm,
            horizontal_resolution_px=horizontal_resolution_px,
            viewing_distance_mm=viewing_distance_mm,
        )
        return {
            "width_px": float(width_px),
            "visual_angle_deg": angle,
            "inside_szafir_measured_range": (
                SZAFIR_LINE_MIN_VISUAL_ANGLE_DEG
                <= angle
                <= SZAFIR_LINE_MAX_VISUAL_ANGLE_DEG
            ),
        }

    calibration = [describe(value) for value in widths_px]
    validation = [describe(value) for value in candidates_px]
    warnings = []
    if not all(row["inside_szafir_measured_range"] for row in calibration):
        warnings.append(
            "교정 굵기 중 Szafir 원 논문 측정 범위(0.05~0.35도) 밖의 값이 "
            "있습니다. 전이 모델 C의 주 분석에 그대로 사용하지 마세요."
        )
    if any(not row["width_px"].is_integer() for row in validation):
        warnings.append(
            "기하평균 검증 후보가 소수 픽셀입니다. 실제 래스터 자극에서는 "
            "가까운 서로 다른 정수 픽셀을 사전에 선택하고, 실제 렌더링 굵기의 "
            "시각도를 다시 계산해 기록하세요."
        )
    rounded_unique = {round(row["width_px"]) for row in validation}
    if len(rounded_unique) < len(validation) or any(
        round(row["width_px"]) in {round(value) for value in widths_px}
        for row in validation
    ):
        warnings.append(
            "검증 후보를 정수 픽셀로 반올림하면 서로 겹치거나 교정 굵기와 "
            "겹칩니다. 화면 해상도 또는 굵기 조합을 다시 정하세요."
        )

    return {
        "display": {
            "display_width_mm": float(display_width_mm),
            "horizontal_resolution_px": float(horizontal_resolution_px),
            "viewing_distance_mm": float(viewing_distance_mm),
            "pixel_pitch_mm": pixel_pitch_mm(
                display_width_mm, horizontal_resolution_px
            ),
        },
        "calibration_widths": calibration,
        "validation_candidates": validation,
        "warnings": warnings,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="실험 선 굵기를 px에서 시각도로 변환해 범위를 점검합니다."
    )
    parser.add_argument("--display-width-mm", type=float, required=True)
    parser.add_argument("--resolution-px", type=float, required=True)
    parser.add_argument("--viewing-distance-mm", type=float, required=True)
    parser.add_argument(
        "--widths-px",
        type=float,
        nargs=3,
        metavar=("THIN", "MIDDLE", "THICK"),
        required=True,
    )
    return parser


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = _parser().parse_args()
    report = build_width_report(
        args.widths_px,
        display_width_mm=args.display_width_mm,
        horizontal_resolution_px=args.resolution_px,
        viewing_distance_mm=args.viewing_distance_mm,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
