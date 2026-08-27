"""파라미터 민감도 분석.

``n_segments``, ``merge_thresh``, ``d_max``를 왜 그 값으로 골랐는지
수치로 남긴다. 예비 측정에서 compactness가 얇은 선의 생사를 가른다는
것이 드러나 네 번째 축으로 함께 넣었다.

정답을 아는 합성 이미지에서만 검출률을 셀 수 있으므로, 실제 이미지에는
쌍 수와 영역 수 같은 관측값만 적용한다.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from failure_analysis import (
    VERDICTS,
    classify_elements,
    load_ground_truth,
)
from measure_pairs import measure_image
from slic_extract import extract_regions, load_image


N_SEGMENTS_GRID = (100, 200, 300, 500, 800, 1200, 2000, 3000)
COMPACTNESS_GRID = (1.0, 3.0, 5.0, 10.0, 20.0)
MERGE_THRESH_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
D_MAX_GRID: tuple[float | None, ...] = (50.0, 100.0, 200.0, 400.0, None)

# 굵기별 검출률을 보려면 굵기만 다른 장면이 필요하다.
LINE_SCENES = tuple(
    f"lines_w{width:02d}" for width in (1, 2, 3, 4, 6, 8, 12, 16)
)


# 저장소 규칙이 LF다. csv 모듈의 기본 줄바꿈은 CRLF라 명시해야 한다.
LF = "\n"


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8", newline=LF)
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator=LF
        )
        writer.writeheader()
        writer.writerows(rows)


def _verdict_counts(records: list[dict]) -> dict[str, int]:
    counts = {verdict: 0 for verdict in VERDICTS}
    for record in records:
        counts[record["verdict"]] += 1
    return counts


def _run_case(
    root: Path,
    manifest: dict,
    name: str,
    *,
    n_segments: int,
    compactness: float,
    merge_thresh: float,
) -> dict[str, object]:
    """한 장면을 한 설정으로 분할하고 판정 개수를 센다."""
    entry = manifest["images"][name]
    image = load_image(root / entry["image"])
    id_map = np.load(root / entry["truth"])
    started = time.perf_counter()
    labels, regions, adjacency = extract_regions(
        image,
        n_segments=n_segments,
        compactness=compactness,
        merge_thresh=merge_thresh,
    )
    elapsed = time.perf_counter() - started
    verdicts = classify_elements(labels, regions, entry["elements"], id_map)
    counts = _verdict_counts(verdicts)
    total = len(verdicts)
    fragments = [record["overlap_region_count"] for record in verdicts]
    return {
        "image": name,
        "n_segments": n_segments,
        "compactness": compactness,
        "merge_thresh": merge_thresh,
        "label_count": int(labels.max()) + 1,
        "region_count": len(regions),
        "adjacent_edge_count": len(adjacency),
        "element_count": total,
        **counts,
        "detection_rate": (
            round(counts["detected"] / total, 4) if total else 0.0
        ),
        "mean_regions_per_element": round(float(np.mean(fragments)), 3),
        "seconds": round(elapsed, 3),
    }


def sweep_n_segments(
    root: str | Path = "../data/synthetic",
    out_dir: str | Path = "../results",
    *,
    grid: tuple[int, ...] = N_SEGMENTS_GRID,
    compactness: float = 3.0,
    merge_thresh: float = 1.0,
    scenes: tuple[str, ...] = LINE_SCENES,
    verbose: bool = True,
) -> list[dict]:
    """선 두께별로 n_segments가 검출률에 미치는 영향을 잰다."""
    root = Path(root)
    manifest = load_ground_truth(root)
    rows = []
    for n_segments in grid:
        for name in scenes:
            row = _run_case(
                root,
                manifest,
                name,
                n_segments=n_segments,
                compactness=compactness,
                merge_thresh=merge_thresh,
            )
            row["nominal_stroke_width_px"] = manifest["images"][name][
                "elements"
            ][0]["nominal_stroke_width_px"]
            rows.append(row)
        if verbose:
            print(f"n_segments={n_segments} 완료")
    _write_csv(rows, Path(out_dir) / "n_segments_sweep.csv")
    return rows


def sweep_compactness(
    root: str | Path = "../data/synthetic",
    out_dir: str | Path = "../results",
    *,
    grid: tuple[float, ...] = COMPACTNESS_GRID,
    n_segments: int = 800,
    merge_thresh: float = 1.0,
    scenes: tuple[str, ...] = LINE_SCENES,
    verbose: bool = True,
) -> list[dict]:
    """compactness가 얇은 선 검출에 미치는 영향을 잰다."""
    root = Path(root)
    manifest = load_ground_truth(root)
    rows = []
    for compactness in grid:
        for name in scenes:
            row = _run_case(
                root,
                manifest,
                name,
                n_segments=n_segments,
                compactness=compactness,
                merge_thresh=merge_thresh,
            )
            row["nominal_stroke_width_px"] = manifest["images"][name][
                "elements"
            ][0]["nominal_stroke_width_px"]
            rows.append(row)
        if verbose:
            print(f"compactness={compactness} 완료")
    _write_csv(rows, Path(out_dir) / "compactness_sweep.csv")
    return rows


def sweep_merge_thresh(
    root: str | Path = "../data/synthetic",
    out_dir: str | Path = "../results",
    *,
    grid: tuple[float, ...] = MERGE_THRESH_GRID,
    n_segments: int = 800,
    compactness: float = 3.0,
    verbose: bool = True,
) -> list[dict]:
    """merge_thresh에 따른 영역 수와 중복 탐지 변화를 잰다."""
    root = Path(root)
    manifest = load_ground_truth(root)
    rows = []
    for merge_thresh in grid:
        for name in manifest["images"]:
            rows.append(
                _run_case(
                    root,
                    manifest,
                    name,
                    n_segments=n_segments,
                    compactness=compactness,
                    merge_thresh=merge_thresh,
                )
            )
        if verbose:
            print(f"merge_thresh={merge_thresh} 완료")
    _write_csv(rows, Path(out_dir) / "merge_thresh_sweep.csv")
    return rows


def _region_elements(
    labels: np.ndarray,
    regions: list,
    id_map: np.ndarray,
) -> dict[int, int]:
    """영역마다 가장 많이 겹치는 정답 요소 번호를 붙인다."""
    mapping = {}
    for region in regions:
        mask = labels == region.region_id
        values, counts = np.unique(id_map[mask], return_counts=True)
        mapping[region.region_id] = int(values[np.argmax(counts)])
    return mapping


def sweep_d_max(
    root: str | Path = "../data/synthetic",
    out_dir: str | Path = "../results",
    *,
    grid: tuple[float | None, ...] = D_MAX_GRID,
    n_segments: int = 800,
    compactness: float = 3.0,
    merge_thresh: float = 1.0,
    scenes: tuple[str, ...] = (
        "legend_lines_w08",
        "legend_lines",
        "route_map",
        "bar_chart",
    ),
    verbose: bool = True,
) -> list[dict]:
    """d_max에 따른 쌍 수와 공간 관계 분포를 잰다.

    ``legend_lines`` 장면에서는 범례와 같은 색 선의 쌍을 몇 개나 잡는지도
    센다. 의미로는 짝이지만 멀리 떨어져 있어, 거리로만 쌍을 만드는 방식의
    한계가 그대로 수치로 나온다.
    """
    root = Path(root)
    manifest = load_ground_truth(root)
    rows = []
    for d_max in grid:
        for name in scenes:
            entry = manifest["images"][name]
            image_path = root / entry["image"]
            id_map = np.load(root / entry["truth"])
            started = time.perf_counter()
            measurement = measure_image(
                image_path,
                n_segments=n_segments,
                compactness=compactness,
                merge_thresh=merge_thresh,
                d_max=d_max,
                max_pairs=500_000,
            )
            elapsed = time.perf_counter() - started

            labels, regions, _ = extract_regions(
                load_image(image_path),
                n_segments=n_segments,
                compactness=compactness,
                merge_thresh=merge_thresh,
            )
            element_of = _region_elements(labels, regions, id_map)
            kind_of = {
                element["element_id"]: element["kind"]
                for element in entry["elements"]
            }
            color_of = {
                element["element_id"]: tuple(element["rgb"])
                for element in entry["elements"]
            }

            cross_reference = 0
            for pair in measurement["pairs"]:
                left = element_of.get(pair["region_a"], 0)
                right = element_of.get(pair["region_b"], 0)
                if not left or not right:
                    continue
                kinds = {kind_of.get(left), kind_of.get(right)}
                if kinds == {"legend_swatch", "line"} and color_of.get(
                    left
                ) == color_of.get(right):
                    cross_reference += 1

            pairs = measurement["pairs"]
            distances = np.asarray(
                [pair["centroid_distance_px"] for pair in pairs], dtype=float
            )
            rows.append(
                {
                    "image": name,
                    "d_max": "inf" if d_max is None else d_max,
                    "n_segments": n_segments,
                    "merge_thresh": merge_thresh,
                    "region_count": measurement["region_count"],
                    "pair_count": len(pairs),
                    "pairs_dropped": measurement["pairs_dropped"],
                    "adjacent_ratio": round(
                        float(
                            np.mean([pair["adjacent"] for pair in pairs])
                        ),
                        4,
                    )
                    if pairs
                    else 0.0,
                    "background_pair_ratio": round(
                        float(
                            np.mean(
                                [pair["involves_background"] for pair in pairs]
                            )
                        ),
                        4,
                    )
                    if pairs
                    else 0.0,
                    "median_distance_px": round(
                        float(np.median(distances)), 2
                    )
                    if pairs
                    else 0.0,
                    "max_distance_px": round(float(distances.max()), 2)
                    if pairs
                    else 0.0,
                    "legend_line_pairs": cross_reference,
                    "seconds": round(elapsed, 3),
                }
            )
        if verbose:
            print(f"d_max={d_max} 완료")
    _write_csv(rows, Path(out_dir) / "d_max_sweep.csv")
    return rows


def _detection_table(rows: list[dict], axis: str) -> list[str]:
    """굵기 × 파라미터 검출률 표를 만든다."""
    widths = sorted({row["nominal_stroke_width_px"] for row in rows})
    values = sorted({row[axis] for row in rows})
    lines = ["| 굵기 (px) | " + " | ".join(f"{v:g}" for v in values) + " |"]
    lines.append("|---|" + "---|" * len(values))
    for width in widths:
        cells = []
        for value in values:
            subset = [
                row
                for row in rows
                if row["nominal_stroke_width_px"] == width
                and row[axis] == value
            ]
            if not subset:
                cells.append("-")
                continue
            detected = sum(row["detected"] for row in subset)
            total = sum(row["element_count"] for row in subset)
            cells.append(f"{detected / total:.0%}" if total else "-")
        lines.append(f"| {width:g} | " + " | ".join(cells) + " |")
    return lines


def write_summary(
    n_segments_rows: list[dict],
    compactness_rows: list[dict],
    merge_rows: list[dict],
    d_max_rows: list[dict],
    out_dir: str | Path = "../results",
) -> str:
    """네 표를 한 문서로 묶는다."""
    lines = ["# 파라미터 민감도 요약", ""]
    lines.append(
        "합성 이미지에서 잰 값이다. 검출률은 정답 요소 중 `detected`로 "
        "판정된 비율이고, 판정 기준은 `failure_analysis.py`에 있다."
    )
    lines.append("")

    lines.append("## n_segments")
    lines.append("")
    lines.append(
        f"compactness={n_segments_rows[0]['compactness']:g}, "
        f"merge_thresh={n_segments_rows[0]['merge_thresh']:g} 고정."
    )
    lines.append("")
    lines.extend(_detection_table(n_segments_rows, "n_segments"))
    lines.append("")

    lines.append("## compactness")
    lines.append("")
    lines.append(
        f"n_segments={compactness_rows[0]['n_segments']}, "
        f"merge_thresh={compactness_rows[0]['merge_thresh']:g} 고정."
    )
    lines.append("")
    lines.extend(_detection_table(compactness_rows, "compactness"))
    lines.append("")

    lines.append("## merge_thresh")
    lines.append("")
    lines.append(
        "검출률은 장면별 검출률의 평균이다. 요소 수를 그대로 합치면 점이 "
        "320개인 `dot_plate` 한 장이 전체를 결정해 버린다."
    )
    lines.append("")
    line_scenes = {
        row["image"] for row in merge_rows if "lines_w" in row["image"]
    }
    lines.append(
        "| merge_thresh | 평균 영역 수 | 요소당 평균 영역 수 | "
        "장면 평균 검출률 | 선 장면 검출률 |"
    )
    lines.append("|---|---|---|---|---|")
    for value in sorted({row["merge_thresh"] for row in merge_rows}):
        subset = [row for row in merge_rows if row["merge_thresh"] == value]
        line_subset = [
            row for row in subset if row["image"] in line_scenes
        ]
        line_detected = sum(row["detected"] for row in line_subset)
        line_total = sum(row["element_count"] for row in line_subset)
        mean_regions = np.mean([row["region_count"] for row in subset])
        mean_fragments = np.mean(
            [row["mean_regions_per_element"] for row in subset]
        )
        mean_detection = np.mean([row["detection_rate"] for row in subset])
        lines.append(
            f"| {value:g} | {mean_regions:.1f} | {mean_fragments:.2f} | "
            f"{mean_detection:.1%} | {line_detected / line_total:.1%} |"
        )
    lines.append("")
    lines.append("장면별 검출률:")
    lines.append("")
    scene_names = sorted({row["image"] for row in merge_rows})
    grid_values = sorted({row["merge_thresh"] for row in merge_rows})
    lines.append(
        "| 장면 | " + " | ".join(f"{v:g}" for v in grid_values) + " |"
    )
    lines.append("|---|" + "---|" * len(grid_values))
    for name in scene_names:
        cells = []
        for value in grid_values:
            match = next(
                (
                    row
                    for row in merge_rows
                    if row["image"] == name and row["merge_thresh"] == value
                ),
                None,
            )
            cells.append(f"{match['detection_rate']:.0%}" if match else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## d_max")
    lines.append("")
    lines.append(
        "| d_max | 장면 | 쌍 수 | 맞닿은 쌍 비율 | 범례-선 쌍 | 초 |"
    )
    lines.append("|---|---|---|---|---|---|")
    for row in d_max_rows:
        lines.append(
            f"| {row['d_max']} | {row['image']} | {row['pair_count']} | "
            f"{row['adjacent_ratio']:.2f} | {row['legend_line_pairs']} | "
            f"{row['seconds']:.1f} |"
        )
    lines.append("")
    lines.append(
        "`범례-선 쌍`은 범례 조각과 같은 색 선이 한 쌍으로 잡힌 횟수다. "
        "정답은 장면마다 5쌍이다. d_max를 늘려야 잡히지만, 늘리면 관계없는 "
        "쌍도 함께 늘어난다. `legend_lines`는 선이 3 px이라 분할 단계에서 "
        "이미 사라지므로 어떤 d_max에서도 0이 나온다. 거리 때문에 못 잡는 "
        "것과 선이 얇아 못 잡는 것을 가르려면 `legend_lines_w08`을 본다."
    )
    lines.append("")

    path = Path(out_dir) / "sweep_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline=LF)
    return str(path)


def run_all(
    root: str | Path = "../data/synthetic",
    out_dir: str | Path = "../results",
    *,
    verbose: bool = True,
) -> str:
    """네 가지 민감도 분석을 순서대로 돌리고 요약을 쓴다."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n_segments_rows = sweep_n_segments(root, out_dir, verbose=verbose)
    compactness_rows = sweep_compactness(root, out_dir, verbose=verbose)
    merge_rows = sweep_merge_thresh(root, out_dir, verbose=verbose)
    d_max_rows = sweep_d_max(root, out_dir, verbose=verbose)
    path = write_summary(
        n_segments_rows, compactness_rows, merge_rows, d_max_rows, out_dir
    )
    if verbose:
        print(f"저장: {path}")
    return path


__all__ = [
    "COMPACTNESS_GRID",
    "D_MAX_GRID",
    "MERGE_THRESH_GRID",
    "N_SEGMENTS_GRID",
    "run_all",
    "sweep_compactness",
    "sweep_d_max",
    "sweep_merge_thresh",
    "sweep_n_segments",
    "write_summary",
]
