"""두 색 baseline을 터미널에서 실행하는 간단한 프로그램."""

from __future__ import annotations

import argparse
import json
import sys

from baseline import run_baseline
from config import ConfusionThresholds, SearchConfig


def _ask_rgb(label: str) -> tuple[int, int, int]:
    while True:
        try:
            values = tuple(int(value) for value in input(f"{label} RGB를 공백으로 입력하세요 (예: 220 50 50): ").split())
            if len(values) != 3 or not all(0 <= value <= 255 for value in values):
                raise ValueError
            return values  # type: ignore[return-value]
        except ValueError:
            print("0~255 정수 세 개를 입력해야 합니다.")


def _ask_float(label: str, minimum: float, maximum: float | None = None) -> float:
    while True:
        try:
            value = float(input(label))
            if value < minimum or (maximum is not None and value > maximum):
                raise ValueError
            return value
        except ValueError:
            range_text = f"{minimum} 이상" if maximum is None else f"{minimum}~{maximum}"
            print(f"{range_text}의 숫자를 입력해야 합니다.")


def main() -> None:
    # Windows 콘솔의 기본 인코딩(cp949)으로 출력하면 결과를 파일로 저장했을 때
    # UTF-8을 기대하는 다른 도구와 팀원 환경에서 깨진다. 결과 JSON은 항상
    # UTF-8로 쓴다.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # 재설정할 수 없는 스트림은 그대로 둔다.
        pass

    parser = argparse.ArgumentParser(description="개인화 두 색 최소 보정 baseline")
    parser.add_argument("--color-a", nargs=3, type=int)
    parser.add_argument("--color-b", nargs=3, type=int)
    parser.add_argument("--type", choices=("protan", "deutan"))
    parser.add_argument("--severity", type=float)
    parser.add_argument("--original-threshold", type=float)
    parser.add_argument("--confusion-threshold", type=float)
    parser.add_argument("--target-threshold", type=float)
    parser.add_argument("--search-step", type=float, default=SearchConfig.step)
    parser.add_argument("--max-shift", type=float, default=SearchConfig.max_shift)
    parser.add_argument(
        "--palette",
        nargs="+",
        type=int,
        metavar="V",
        help="재혼동 검사에 쓸 주변 색들. R G B를 3개씩 이어서 입력한다.",
    )
    args = parser.parse_args()

    palette = None
    if args.palette:
        if len(args.palette) % 3 != 0:
            parser.error("--palette 값의 개수는 3의 배수여야 합니다.")
        palette = [
            tuple(args.palette[index : index + 3])
            for index in range(0, len(args.palette), 3)
        ]

    color_a = tuple(args.color_a) if args.color_a else _ask_rgb("색 A")
    color_b = tuple(args.color_b) if args.color_b else _ask_rgb("색 B")
    cvd_type = args.type or input("유형을 입력하세요 (protan/deutan): ").strip().lower()
    severity = args.severity if args.severity is not None else _ask_float("simulation severity (0~1): ", 0.0, 1.0)

    # 안내문은 대화형으로 물어볼 때만 출력한다. 항상 출력하면 stdout이
    # 순수 JSON이 아니게 되어 결과를 파일로 파이프할 수 없다.
    needs_prompt = any(
        value is None
        for value in (
            args.original_threshold,
            args.confusion_threshold,
            args.target_threshold,
        )
    )
    if needs_prompt:
        print("다음 ΔE00 임계값은 아직 실험 변수이며 의학적 정답값이 아닙니다.")
    original_threshold = (
        args.original_threshold
        if args.original_threshold is not None
        else _ask_float("원본 최소 색차 tau_high: ", 0.0)
    )
    confusion_threshold = (
        args.confusion_threshold
        if args.confusion_threshold is not None
        else _ask_float("혼동 최대 색차 tau_low: ", 0.0)
    )
    target_threshold = (
        args.target_threshold
        if args.target_threshold is not None
        else _ask_float("보정 목표 색차 tau_target: ", 0.0)
    )

    result = run_baseline(
        color_a,
        color_b,
        cvd_type,
        severity,
        ConfusionThresholds(original_threshold, confusion_threshold, target_threshold),
        search_config=SearchConfig(step=args.search_step, max_shift=args.max_shift),
        palette=palette,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
