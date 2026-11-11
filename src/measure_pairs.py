"""영역 쌍의 원본 수치 측정. 혼동 여부는 판정하지 않는다.

②단계의 출력은 "이 쌍이 혼동된다"가 아니라 "이 쌍은 원본 색차 얼마,
시뮬레이션 색차 얼마, 굵기 얼마, 거리 얼마"라는 수치다. 임계값을 정하는
일은 ①단계 사용자 실험의 몫이므로 여기에는 is_confused 같은 열을 만들지
않는다.

측정된 개인 역치 0.62는 큰 색면 두 개를 맞붙여 본 최적 조건의 값이라
얇은 선이나 떨어진 영역에 그대로 쓸 수 없다. 그래서 이 모듈은 굵기와
거리를 함께 기록해, 2차 실험이 어떤 조건에서 역치를 다시 재야 하는지
정할 근거를 남긴다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from color_spaces import srgb_to_lab
from distance import delta_e_2000
from machado import simulate_color_float
from slic_extract import (
    IMAGE_SUFFIXES,
    estimate_background,
    extract_regions,
    load_image,
    px_to_arcmin,
)


# 기본 시뮬레이션 조건. ①단계에서 개인 파라미터가 오면 그 값으로 바꾼다.
DEFAULT_CVD_CONDITIONS: tuple[tuple[str, float], ...] = (
    ("protan", 1.0),
    ("deutan", 1.0),
)

CAPTURE_META_KEYS = (
    "screen_diagonal_inch",
    "screen_resolution",
    "viewing_distance_cm",
)


def _condition_key(cvd_type: str, severity: float) -> str:
    return f"{cvd_type}_{int(round(float(severity) * 100)):03d}"


def _simulated_lab(
    regions,
    cvd_type: str,
    severity: float,
) -> np.ndarray:
    """영역 대표색을 시뮬레이션한 뒤 CIELAB으로 바꾼다.

    8-bit 반올림 오차를 피하려고 simulate_color_float를 쓴다.
    """
    return np.asarray(
        [
            srgb_to_lab(simulate_color_float(region.rgb, cvd_type, severity))
            for region in regions
        ],
        dtype=float,
    )


def _bbox_gap(boxes: np.ndarray, index_a: np.ndarray, index_b: np.ndarray):
    """두 영역 경계상자 사이 빈 픽셀 수의 최단거리를 계산한다.

    bbox의 최댓값은 포함되는 픽셀 좌표다. 따라서 한 축에서 ``A.max=10``,
    ``B.min=11``이면 두 상자는 맞닿아 있고 그 사이 빈 픽셀은 0개다.
    실제 영역 경계가 아니라 경계상자를 쓰므로 대각선처럼 상자가 헐거운
    모양에서는 여전히 실제 간격보다 작게 나올 수 있다.
    """
    a_min_row, a_min_col, a_max_row, a_max_col = boxes[index_a].T
    b_min_row, b_min_col, b_max_row, b_max_col = boxes[index_b].T
    row_gap = np.maximum(
        np.maximum(b_min_row - a_max_row - 1.0, a_min_row - b_max_row - 1.0),
        0.0,
    )
    column_gap = np.maximum(
        np.maximum(b_min_col - a_max_col - 1.0, a_min_col - b_max_col - 1.0),
        0.0,
    )
    return np.hypot(row_gap, column_gap)


def _validate_capture_meta(capture_meta: dict | None) -> dict | None:
    if capture_meta is None:
        return None
    missing = [key for key in CAPTURE_META_KEYS if key not in capture_meta]
    if missing:
        raise ValueError(
            "capture_meta에 다음 항목이 없습니다: "
            + ", ".join(missing)
            + ". 비워두면 시각도 환산을 할 수 없습니다."
        )
    return dict(capture_meta)


def measure_image(
    path: str | Path,
    *,
    n_segments: int = 800,
    compactness: float = 3.0,
    sigma: float = 1.0,
    merge_thresh: float = 0.0,
    min_area_px: int = 4,
    d_max: float | None = 200.0,
    cvd_conditions: tuple[tuple[str, float], ...] = DEFAULT_CVD_CONDITIONS,
    capture_meta: dict | None = None,
    max_pairs: int = 50_000,
) -> dict[str, object]:
    """이미지 한 장을 영역 목록과 쌍별 원본 수치로 바꾼다.

    Args:
        d_max: 중심 사이 거리가 이 값 이하인 쌍만 기록한다. ``None``이면
            거리 제한 없이 모든 쌍을 본다.
        max_pairs: 한 이미지에서 기록할 쌍의 상한. 넘치면 가까운 쌍부터
            남기고, 잘라낸 개수와 실제 적용된 거리 상한을 결과에 적는다.
    """
    meta = _validate_capture_meta(capture_meta)
    image = load_image(path)
    height, width = image.shape[:2]
    labels, regions, adjacency = extract_regions(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        merge_thresh=merge_thresh,
        min_area_px=min_area_px,
    )

    result: dict[str, object] = {
        "image": Path(path).name,
        "width": width,
        "height": height,
        "params": {
            "n_segments": int(n_segments),
            "compactness": float(compactness),
            "sigma": float(sigma),
            "merge_thresh": float(merge_thresh),
            "min_area_px": int(min_area_px),
            "d_max": None if d_max is None else float(d_max),
            "max_pairs": int(max_pairs),
        },
        "capture_meta": meta,
        "cvd_conditions": [
            {"cvd_type": kind, "severity": float(value)}
            for kind, value in cvd_conditions
        ],
        "label_count": int(labels.max()) + 1,
        "region_count": len(regions),
        "background_region_id": estimate_background(regions),
        "regions": [region.as_dict() for region in regions],
        "pairs": [],
        "pairs_dropped": 0,
        "applied_distance_limit_px": (
            None if d_max is None else float(d_max)
        ),
    }
    if len(regions) < 2:
        return result

    index_of = {
        region.region_id: index
        for index, region in enumerate(regions)
    }
    centroids = np.asarray([region.centroid_xy for region in regions])
    boxes = np.asarray([region.bbox for region in regions], dtype=float)
    areas = np.asarray([region.area_px for region in regions], dtype=float)
    strokes = np.asarray(
        [region.stroke_width_px for region in regions], dtype=float
    )
    spreads = np.asarray(
        [region.color_spread for region in regions], dtype=float
    )
    original_lab = np.asarray([region.lab for region in regions], dtype=float)

    count = len(regions)
    index_a, index_b = np.triu_indices(count, k=1)
    distance = np.linalg.norm(
        centroids[index_a] - centroids[index_b], axis=-1
    )

    # 맞닿은 쌍은 거리 제한과 무관하게 남긴다. 인접한데도 멀다고 빠지는
    # 일(길게 뻗은 영역)을 막기 위해서다.
    #
    # 쌍 위치는 triu 순서에서 계산으로 구한다. 모든 쌍을 사전에 넣으면
    # 영역이 수천 개일 때 항목이 수백만 개가 되어 메모리를 크게 쓴다.
    shared_border = np.zeros(index_a.shape, dtype=float)
    edges = [
        (index_of[left], index_of[right], border)
        for (left, right), border in adjacency.items()
        if left in index_of and right in index_of
    ]
    if edges:
        edge_array = np.asarray(edges, dtype=float)
        low = np.minimum(edge_array[:, 0], edge_array[:, 1]).astype(np.int64)
        high = np.maximum(edge_array[:, 0], edge_array[:, 1]).astype(np.int64)
        positions = (
            low * (2 * count - low - 1) // 2 + (high - low - 1)
        )
        shared_border[positions] = edge_array[:, 2]
    adjacent = shared_border > 0.0

    if d_max is not None:
        keep = (distance <= float(d_max)) | adjacent
        index_a, index_b = index_a[keep], index_b[keep]
        distance, shared_border = distance[keep], shared_border[keep]
        adjacent = adjacent[keep]

    if index_a.size > max_pairs:
        # 무작위로 버리지 않고 가까운 쌍부터 남긴다. 잘라낸 사실은
        # 결과에 남겨 "전부 봤다"로 읽히지 않게 한다.
        order = np.argsort(distance, kind="stable")[:max_pairs]
        order.sort()
        result["pairs_dropped"] = int(index_a.size - max_pairs)
        result["applied_distance_limit_px"] = float(distance[order].max())
        index_a, index_b = index_a[order], index_b[order]
        distance, shared_border = distance[order], shared_border[order]
        adjacent = adjacent[order]

    gap = _bbox_gap(boxes, index_a, index_b)
    delta_original = np.asarray(
        delta_e_2000(original_lab[index_a], original_lab[index_b]), dtype=float
    )
    simulated_delta = {}
    for cvd_type, severity in cvd_conditions:
        simulated_lab = _simulated_lab(regions, cvd_type, severity)
        simulated_delta[_condition_key(cvd_type, severity)] = np.asarray(
            delta_e_2000(simulated_lab[index_a], simulated_lab[index_b]),
            dtype=float,
        )

    background = result["background_region_id"]
    region_ids = [region.region_id for region in regions]

    arcmin_scale = None
    if meta is not None:
        arcmin_scale = {
            "screen_diagonal_inch": float(meta["screen_diagonal_inch"]),
            "screen_resolution": tuple(meta["screen_resolution"]),
            "viewing_distance_cm": float(meta["viewing_distance_cm"]),
        }

    pairs = []
    for position in range(index_a.size):
        left = int(index_a[position])
        right = int(index_b[position])
        record = {
            "region_a": region_ids[left],
            "region_b": region_ids[right],
            "adjacent": int(bool(adjacent[position])),
            "shared_border_px": float(shared_border[position]),
            "centroid_distance_px": round(float(distance[position]), 3),
            "bbox_gap_px": round(float(gap[position]), 3),
            "delta_e_original": round(float(delta_original[position]), 4),
            "area_min_px": int(min(areas[left], areas[right])),
            "area_max_px": int(max(areas[left], areas[right])),
            "stroke_min_px": round(
                float(min(strokes[left], strokes[right])), 3
            ),
            "stroke_max_px": round(
                float(max(strokes[left], strokes[right])), 3
            ),
            "color_spread_max": round(
                float(max(spreads[left], spreads[right])), 3
            ),
            "involves_background": int(
                background is not None
                and background in (region_ids[left], region_ids[right])
            ),
        }
        for key, values in simulated_delta.items():
            record[f"delta_e_sim_{key}"] = round(float(values[position]), 4)
        if arcmin_scale is not None:
            record["centroid_distance_arcmin"] = round(
                px_to_arcmin(distance[position], **arcmin_scale), 3
            )
            record["bbox_gap_arcmin"] = round(
                px_to_arcmin(gap[position], **arcmin_scale), 3
            )
            record["stroke_min_arcmin"] = round(
                px_to_arcmin(record["stroke_min_px"], **arcmin_scale), 3
            )
        pairs.append(record)

    result["pairs"] = pairs
    result["pair_count"] = len(pairs)

    if arcmin_scale is not None:
        for region_record in result["regions"]:
            region_record["stroke_width_arcmin"] = round(
                px_to_arcmin(
                    region_record["stroke_width_px"], **arcmin_scale
                ),
                3,
            )
    return result


def load_capture_meta_csv(path: str | Path) -> dict[str, dict]:
    """sources.csv에서 이미지별 화면 조건을 읽는다.

    실제 이미지는 서로 다른 화면에서 모으게 되므로 한 벌의 값을 전체에
    적용할 수 없다. 파일 이름으로 찾아 이미지마다 다른 조건을 쓴다.
    """
    import csv

    table: dict[str, dict] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            try:
                meta = {
                    "screen_diagonal_inch": float(
                        row["screen_diagonal_inch"]
                    ),
                    "screen_resolution": [
                        int(row["screen_width_px"]),
                        int(row["screen_height_px"]),
                    ],
                    "viewing_distance_cm": float(row["viewing_distance_cm"]),
                    "source": row.get("source_url", ""),
                    "license": row.get("license", ""),
                    "category": row.get("category", ""),
                }
            except (KeyError, TypeError, ValueError):
                # 화면 조건이 비었거나 숫자가 아니면 그 줄은 건너뛰고,
                # 그 이미지는 capture_meta 인자의 값을 쓰게 둔다.
                continue
            table[filename] = meta
    return table


def measure_directory(
    image_dir: str | Path,
    out_json: str | Path,
    *,
    sources_csv: str | Path | None = None,
    verbose: bool = True,
    **kwargs,
) -> dict[str, object]:
    """디렉터리 안의 모든 이미지를 측정해 JSON 하나로 저장한다.

    ``sources_csv``를 주면 이미지마다 그 파일의 화면 조건을 쓴다. 표에
    없는 이미지는 ``capture_meta`` 인자의 값으로 넘어간다.
    """
    directory = Path(image_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"이미지 디렉터리가 없습니다: {directory}")

    paths = sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(
            f"{directory}에 측정할 이미지가 없습니다. "
            f"지원 확장자: {', '.join(IMAGE_SUFFIXES)}"
        )

    per_image_meta = (
        load_capture_meta_csv(sources_csv) if sources_csv else {}
    )
    fallback_meta = kwargs.pop("capture_meta", None)
    missing_meta = []

    measurements = []
    for path in paths:
        meta = per_image_meta.get(path.name, fallback_meta)
        if meta is None:
            missing_meta.append(path.name)
        measurement = measure_image(path, capture_meta=meta, **kwargs)
        measurements.append(measurement)
        if verbose:
            print(
                f"{path.name}: 영역 {measurement['region_count']}개, "
                f"쌍 {measurement.get('pair_count', 0)}개"
                + (
                    f" (거리 상한으로 {measurement['pairs_dropped']}쌍 제외)"
                    if measurement["pairs_dropped"]
                    else ""
                )
            )

    if missing_meta and verbose:
        print(
            f"주의: 화면 조건이 없는 이미지 {len(missing_meta)}장은 "
            "시각도 환산을 하지 못했다 — "
            + ", ".join(missing_meta[:5])
            + (" 외" if len(missing_meta) > 5 else "")
        )

    payload = {
        "image_dir": str(directory),
        "image_count": len(measurements),
        "measurements": measurements,
    }
    output = Path(out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    if verbose:
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"저장: {output} ({size_mb:.1f} MB)")
    return payload


__all__ = [
    "DEFAULT_CVD_CONDITIONS",
    "measure_directory",
    "measure_image",
]
