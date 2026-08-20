"""분할 실패 유형 분류.

합성 이미지에는 요소별 정답 픽셀 지도가 있으므로, 각 요소가 분할 결과에
어떻게 나타났는지 셀 수 있다. 판정 기준은 작업 계획의 네 유형을 그대로
쓴다.

* ``merged_with_background`` 대상을 덮는 영역이 대상 면적의 4배 초과
* ``fragmented``             대상 위치에 겹치는 영역이 8개 초과
* ``color_shifted``          영역은 있으나 대표색이 대상과 크게 다름
* ``not_covered``            대상 위치를 덮는 영역이 없음

네 유형 중 어디에도 걸리지 않으면 ``detected``다. 판정 순서가 결과를
바꾸므로 위 순서를 고정한다. 예를 들어 배경에 흡수된 요소는 대표색도
어긋나 있지만 ``merged_with_background``로만 센다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from distance import color_distance
from slic_extract import extract_regions, load_image


# 저장소 규칙이 LF다. 윈도우에서 돌려도 CRLF가 섞이지 않게 명시한다.
LF = "\n"

SWALLOW_FACTOR = 4.0
FRAGMENT_LIMIT = 8
COLOR_TOLERANCE = 10.0

# 조각 수를 셀 때 어디까지 세는가. 겹치는 영역을 큰 것부터 세어 요소
# 면적의 이만큼을 덮는 데 필요한 개수를 조각 수로 본다.
#
# 작업 계획의 기준은 "겹치는 영역이 8개 초과"인데, 스친 영역까지 모두
# 세면 길게 뻗은 선은 어떤 설정에서도 8개를 넘는다. 400 px짜리 선이
# superpixel 하나에 통째로 들어가는 일은 없기 때문이다. 그래서 개수를
# 세는 규칙만 "요소의 90%를 덮는 데 필요한 개수"로 바꾸고, 8개라는
# 경계값은 그대로 둔다.
FRAGMENT_COVERAGE = 0.90

# 색 어긋남을 잴 때 무시할 만큼 작은 조각의 기준.
COLOR_MIN_SHARE = 0.01

# 굵기별 검출률 표를 묶을 구간. 얇은 쪽은 1 px 단위로 잘게 나눈다.
# 검출 여부가 갈리는 곳이 4~8 px 사이라서다.
WIDTH_BINS = (
    (0.0, 2.0, "1~2 미만"),
    (2.0, 4.0, "2~4 미만"),
    (4.0, 6.0, "4~6 미만"),
    (6.0, 8.0, "6~8 미만"),
    (8.0, 12.0, "8~12 미만"),
    (12.0, 16.0, "12~16 미만"),
    (16.0, float("inf"), "16 이상"),
)

VERDICTS = (
    "detected",
    "merged_with_background",
    "fragmented",
    "color_shifted",
    "not_covered",
)


def load_ground_truth(root: str | Path) -> dict:
    """generate_synthetic이 만든 정답 파일을 읽는다."""
    path = Path(root) / "ground_truth.json"
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def classify_elements(
    labels: np.ndarray,
    regions: list,
    elements: list[dict],
    id_map: np.ndarray,
    *,
    swallow_factor: float = SWALLOW_FACTOR,
    fragment_limit: int = FRAGMENT_LIMIT,
    color_tolerance: float = COLOR_TOLERANCE,
) -> list[dict]:
    """요소 하나하나가 분할 결과에서 어떻게 되었는지 판정한다."""
    area_of = {region.region_id: region.area_px for region in regions}
    color_of = {region.region_id: region.rgb for region in regions}
    verdicts = []

    for element in elements:
        element_id = element["element_id"]
        mask = id_map == element_id
        truth_area = int(mask.sum())
        record = {
            "element_id": element_id,
            "kind": element["kind"],
            "nominal_stroke_width_px": element["nominal_stroke_width_px"],
            "truth_area_px": truth_area,
            "truth_r": element["rgb"][0],
            "truth_g": element["rgb"][1],
            "truth_b": element["rgb"][2],
        }
        if truth_area == 0:
            record.update(
                verdict="not_covered",
                overlap_region_count=0,
                overlap_region_count_any=0,
                dominant_region_id=-1,
                dominant_share=0.0,
                dominant_area_px=0,
                delta_e_to_truth=float("nan"),
            )
            verdicts.append(record)
            continue

        covering, counts = np.unique(labels[mask], return_counts=True)
        known = np.asarray(
            [label in area_of for label in covering], dtype=bool
        )
        covering, counts = covering[known], counts[known]
        if covering.size == 0:
            record.update(
                verdict="not_covered",
                overlap_region_count=0,
                overlap_region_count_any=0,
                dominant_region_id=-1,
                dominant_share=0.0,
                dominant_area_px=0,
                delta_e_to_truth=float("nan"),
            )
            verdicts.append(record)
            continue

        order = np.argsort(counts)[::-1]
        covering, counts = covering[order], counts[order]
        shares = counts / truth_area

        # 큰 조각부터 더해 요소의 90%를 덮는 데 몇 개가 필요한지 센다.
        cumulative = np.cumsum(shares)
        fragment_count = int(
            np.searchsorted(cumulative, FRAGMENT_COVERAGE) + 1
        )
        fragment_count = min(fragment_count, covering.size)

        dominant = int(covering[0])
        dominant_share = float(shares[0])
        dominant_area = int(area_of[dominant])

        # 색 어긋남은 이 요소를 덮은 영역들의 겹침 비율로 가중해 잰다.
        # 요소가 여러 조각으로 나뉘면 그중 하나만 보는 것은 뜻이 없다.
        significant = shares >= COLOR_MIN_SHARE
        weights = shares[significant]
        differences = np.asarray(
            [
                color_distance(color_of[int(label)], tuple(element["rgb"]))
                for label in covering[significant]
            ],
            dtype=float,
        )
        difference = (
            float((differences * weights).sum() / weights.sum())
            if weights.size
            else float(
                color_distance(color_of[dominant], tuple(element["rgb"]))
            )
        )

        if dominant_area > swallow_factor * truth_area:
            verdict = "merged_with_background"
        elif fragment_count > fragment_limit:
            verdict = "fragmented"
        elif difference > color_tolerance:
            verdict = "color_shifted"
        else:
            verdict = "detected"

        record.update(
            verdict=verdict,
            overlap_region_count=fragment_count,
            overlap_region_count_any=int(covering.size),
            dominant_region_id=dominant,
            dominant_share=round(dominant_share, 4),
            dominant_area_px=dominant_area,
            delta_e_to_truth=round(difference, 3),
        )
        verdicts.append(record)
    return verdicts


def evaluate_image(
    root: str | Path,
    name: str,
    manifest: dict,
    *,
    n_segments: int = 800,
    compactness: float = 3.0,
    sigma: float = 1.0,
    merge_thresh: float = 0.0,
    min_area_px: int = 4,
) -> tuple[np.ndarray, list, list[dict]]:
    """합성 이미지 한 장을 분할하고 요소별 판정을 낸다."""
    entry = manifest["images"][name]
    image = load_image(Path(root) / entry["image"])
    id_map = np.load(Path(root) / entry["truth"])
    labels, regions, _ = extract_regions(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        merge_thresh=merge_thresh,
        min_area_px=min_area_px,
    )
    verdicts = classify_elements(labels, regions, entry["elements"], id_map)
    for record in verdicts:
        record["image"] = name
    return labels, regions, verdicts


def region_color_map(
    image: np.ndarray,
    labels: np.ndarray,
    regions,
) -> np.ndarray:
    """영역을 대표색으로 칠한 그림. 파이프라인이 무엇을 봤는지 보여준다."""
    painted = np.zeros_like(image)
    for region in regions:
        painted[labels == region.region_id] = np.asarray(
            region.rgb, dtype=np.uint8
        )
    return painted


def save_comparison(
    image: np.ndarray,
    labels: np.ndarray,
    regions,
    out_path: str | Path,
) -> None:
    """원본과 영역 대표색 그림을 나란히 저장한다."""
    from PIL import Image
    from skimage.segmentation import mark_boundaries

    painted = region_color_map(image, labels, regions)
    outlined = (
        mark_boundaries(image, labels, color=(0.0, 0.0, 0.0), mode="thick")
        * 255.0
    ).astype(np.uint8)
    gap = np.full((image.shape[0], 8, 3), 235, dtype=np.uint8)
    canvas = np.concatenate(
        (image, gap, painted, gap, outlined), axis=1
    )
    Image.fromarray(canvas).save(out_path)


def analyze_dataset(
    root: str | Path = "../data/synthetic",
    out_dir: str | Path = "../results",
    *,
    n_segments: int = 800,
    compactness: float = 3.0,
    merge_thresh: float = 0.0,
    save_figures: bool = True,
    verbose: bool = True,
) -> dict[str, object]:
    """합성 데이터셋 전체의 실패 유형을 세고 대표 사례 그림을 남긴다."""
    manifest = load_ground_truth(root)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_dir = output / "failure_cases"
    if save_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for name in manifest["images"]:
        labels, regions, verdicts = evaluate_image(
            root,
            name,
            manifest,
            n_segments=n_segments,
            compactness=compactness,
            merge_thresh=merge_thresh,
        )
        all_records.extend(verdicts)
        if save_figures:
            image = load_image(Path(root) / manifest["images"][name]["image"])
            save_comparison(
                image, labels, regions, figure_dir / f"{name}.png"
            )
        if verbose:
            counts = _count_verdicts(verdicts)
            summary = " ".join(
                f"{key}={counts[key]}" for key in VERDICTS if counts[key]
            )
            print(f"{name}: {summary}")

    fieldnames = [
        "image",
        "element_id",
        "kind",
        "nominal_stroke_width_px",
        "truth_area_px",
        "verdict",
        "overlap_region_count",
        "overlap_region_count_any",
        "dominant_region_id",
        "dominant_share",
        "dominant_area_px",
        "delta_e_to_truth",
        "truth_r",
        "truth_g",
        "truth_b",
    ]
    csv_path = output / "failure_cases.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator=LF
        )
        writer.writeheader()
        for record in all_records:
            writer.writerow({key: record[key] for key in fieldnames})

    summary = _summary_markdown(
        all_records, n_segments, compactness, merge_thresh
    )
    (output / "failure_summary.md").write_text(
        summary, encoding="utf-8", newline=LF
    )
    if verbose:
        print(f"저장: {csv_path}")
        print(f"저장: {output / 'failure_summary.md'}")
    return {"records": all_records, "csv": str(csv_path)}


def _count_verdicts(records: list[dict]) -> dict[str, int]:
    counts = {verdict: 0 for verdict in VERDICTS}
    for record in records:
        counts[record["verdict"]] += 1
    return counts


def _summary_markdown(
    records: list[dict],
    n_segments: int,
    compactness: float,
    merge_thresh: float,
) -> str:
    counts = _count_verdicts(records)
    total = len(records)
    lines = ["# 분할 실패 유형별 건수", ""]
    lines.append(
        f"설정: n_segments={n_segments}, compactness={compactness}, "
        f"merge_thresh={merge_thresh}"
    )
    lines.append("")
    lines.append("| 유형 | 건수 | 비율 |")
    lines.append("|---|---|---|")
    for verdict in VERDICTS:
        ratio = counts[verdict] / total if total else 0.0
        lines.append(f"| `{verdict}` | {counts[verdict]} | {ratio:.1%} |")
    lines.append(f"| 합계 | {total} | |")
    lines.append("")
    lines.append(
        "요소 수를 그대로 합친 값이라 점이 320개인 `dot_plate` 한 장이 "
        "전체 비율을 좌우한다. 장면별 표를 함께 본다."
    )
    lines.append("")

    lines.append("## 요소 굵기별 검출률")
    lines.append("")
    lines.append(
        "굵기를 구간으로 묶었다. `dot_plate`의 점은 지름이 무작위라 "
        "값 하나하나로 세면 표가 요소 수만큼 길어진다."
    )
    lines.append("")
    lines.append("| 굵기 구간 (px) | 요소 수 | detected | 검출률 |")
    lines.append("|---|---|---|---|")
    for low, high, label in WIDTH_BINS:
        subset = [
            record
            for record in records
            if low <= record["nominal_stroke_width_px"] < high
        ]
        if not subset:
            continue
        detected = sum(
            1 for record in subset if record["verdict"] == "detected"
        )
        lines.append(
            f"| {label} | {len(subset)} | {detected} | "
            f"{detected / len(subset):.1%} |"
        )
    lines.append("")

    lines.append("## 장면별")
    lines.append("")
    lines.append("| 장면 | " + " | ".join(VERDICTS) + " |")
    lines.append("|---|" + "---|" * len(VERDICTS))
    for image in sorted({record["image"] for record in records}):
        subset = [record for record in records if record["image"] == image]
        counts_image = _count_verdicts(subset)
        cells = " | ".join(str(counts_image[verdict]) for verdict in VERDICTS)
        lines.append(f"| {image} | {cells} |")
    lines.append("")
    lines.append(
        "그림은 `results/failure_cases/`에 장면별로 저장했다. "
        "왼쪽부터 원본, 영역 대표색, 경계선이다."
    )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "COLOR_TOLERANCE",
    "FRAGMENT_LIMIT",
    "SWALLOW_FACTOR",
    "VERDICTS",
    "analyze_dataset",
    "classify_elements",
    "evaluate_image",
    "load_ground_truth",
    "region_color_map",
    "save_comparison",
]
