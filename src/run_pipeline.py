"""②단계 전체를 한 번에 돌린다.

팀원 입력과 실제 이미지가 들어온 뒤 이 파일 하나만 실행하면 측정부터
전달 문서까지 다시 만들어진다.

    cd src
    python run_pipeline.py                    # 실제 이미지로
    python run_pipeline.py --synthetic        # 합성 이미지로 (연결 확인용)
    python run_pipeline.py --check            # 입력 상태만 보기

개인 파라미터가 있으면 그 피험자의 시뮬레이션 색차 열이 함께 만들어진다.
없으면 protan/deutan severity 1.0 두 조건으로만 잰다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from export_tables import export_all
from measure_pairs import DEFAULT_CVD_CONDITIONS, measure_directory
from team_inputs import (
    InputError,
    describe_inputs,
    load_personal_params,
    to_cvd_conditions,
)


DEFAULT_INPUTS = Path("../data/inputs")
REAL_IMAGES = Path("../data/real/images")
SYNTHETIC_IMAGES = Path("../data/synthetic/images")
RESULTS = Path("../results")

# 민감도 분석에서 고른 값. 근거는 results/sweep_summary.md에 있다.
PIPELINE_PARAMS = {
    "n_segments": 800,
    "compactness": 3.0,
    "merge_thresh": 1.0,
    "d_max": 200.0,
}


def _resolve_cvd_conditions(inputs_dir: Path, verbose: bool):
    """개인 파라미터가 있으면 그 조건을, 없으면 기본 조건을 쓴다."""
    path = inputs_dir / "personal_params.json"
    try:
        participants = load_personal_params(path)
    except InputError as error:
        if verbose:
            print(f"개인 파라미터를 쓰지 않는다: {error}")
            print(
                "  기본 조건(protan 1.0, deutan 1.0)으로 측정한다.\n"
            )
        return DEFAULT_CVD_CONDITIONS, []

    conditions = to_cvd_conditions(participants)
    if verbose:
        names = ", ".join(
            f"{participant['participant_id']}({participant['cvd_type']} "
            f"{participant['severity']:.2f})"
            for participant in participants
        )
        print(f"개인 파라미터 {len(participants)}명 적용: {names}\n")
    return conditions, participants


def _count_images(image_dir: Path) -> int:
    from slic_extract import IMAGE_SUFFIXES

    if not image_dir.is_dir():
        return 0
    return sum(
        1
        for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )


def run(
    *,
    synthetic: bool = False,
    inputs_dir: Path = DEFAULT_INPUTS,
    results_dir: Path = RESULTS,
    verbose: bool = True,
) -> dict[str, str]:
    """측정 → CSV → conditions.md 순으로 실행한다."""
    image_dir = SYNTHETIC_IMAGES if synthetic else REAL_IMAGES
    label = "합성" if synthetic else "실제"
    count = _count_images(image_dir)
    if count == 0:
        raise SystemExit(
            f"{image_dir}에 이미지가 없다.\n"
            "실제 이미지를 모으는 규칙은 data/real/README.md에 있다.\n"
            "파이프라인만 확인하려면 --synthetic으로 돌린다."
        )
    if verbose:
        print(f"{label} 이미지 {count}장으로 측정한다.")
        print(f"설정: {PIPELINE_PARAMS}\n")

    conditions, _ = _resolve_cvd_conditions(inputs_dir, verbose)
    out_json = results_dir / (
        "measurements_synthetic.json" if synthetic else "measurements_real.json"
    )

    measure_directory(
        image_dir,
        out_json,
        sources_csv=(
            None if synthetic else Path("../data/real/sources.csv")
        ),
        cvd_conditions=conditions,
        verbose=verbose,
        **PIPELINE_PARAMS,
    )
    paths = export_all(out_json, results_dir, verbose=verbose)

    if verbose:
        print()
        print("전달할 것:")
        print(f"  한정훈  {paths['conditions']}")
        print(f"  이창민  {paths['pairs']}, {paths['regions']}")
        if synthetic:
            print()
            print(
                "합성 이미지로 돌린 결과다. 실험 조건으로 쓰면 안 된다."
            )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="②단계 탐지·측정 파이프라인을 실행한다."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="실제 이미지 대신 합성 이미지로 돌린다 (연결 확인용)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="측정하지 않고 팀 입력 상태만 보여준다",
    )
    parser.add_argument(
        "--inputs", default=str(DEFAULT_INPUTS), help="입력 폴더 경로"
    )
    parser.add_argument(
        "--results", default=str(RESULTS), help="산출물 폴더 경로"
    )
    arguments = parser.parse_args(argv)

    inputs_dir = Path(arguments.inputs)
    if arguments.check:
        print(
            describe_inputs(
                inputs_dir / "personal_params.json",
                inputs_dir / "thresholds.json",
            )
        )
        return 0

    run(
        synthetic=arguments.synthetic,
        inputs_dir=inputs_dir,
        results_dir=Path(arguments.results),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
