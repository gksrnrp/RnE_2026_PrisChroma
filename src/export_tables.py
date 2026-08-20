"""측정 JSON을 CSV와 조건 통계로 바꾼다.

산출물은 세 가지다.

* ``regions.csv``    영역별 색·기하 속성 (이창민)
* ``pairs.csv``      영역 쌍 원본 수치 (이창민, 한정훈)
* ``conditions.md``  2차 실험 자극 조건 (한정훈)

``conditions.md``만 저장소에 커밋한다. 나머지 둘은 크기 때문에 제외한다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PERCENTILES = (5, 25, 50, 75, 95)

# 저장소 규칙이 LF다. 윈도우에서 돌려도 CRLF가 섞이지 않게 명시한다.
LF = "\n"

# 1차 실험에서 측정된 개인 역치. 큰 색면을 맞붙여 잰 최적 조건 값이므로
# 실제 이미지에 그대로 쓰지 않는다. 여기서는 상대 비교의 기준으로만 쓴다.
MEASURED_OPTIMAL_THRESHOLD = 0.62


def _load(json_path: str | Path) -> dict:
    with open(json_path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(rows: list[dict], out_path: Path) -> int:
    # 줄바꿈은 저장소 규칙에 맞춰 LF로 쓴다. csv 모듈의 기본값은 CRLF다.
    if not rows:
        out_path.write_text("", encoding="utf-8", newline=LF)
        return 0
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator=LF
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _percentile_row(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {f"p{p}": float("nan") for p in PERCENTILES}
    return {
        f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES
    }


def _format_percentiles(
    label: str,
    values: np.ndarray,
    unit: str,
    decimals: int = 1,
) -> str:
    row = _percentile_row(values)
    cells = " | ".join(
        f"{row[f'p{p}']:.{decimals}f}" for p in PERCENTILES
    )
    return f"| {label} ({unit}) | {values.size} | {cells} |"


def collect_rows(payload: dict) -> tuple[list[dict], list[dict]]:
    """측정 JSON을 영역 행과 쌍 행으로 펼친다."""
    region_rows: list[dict] = []
    pair_rows: list[dict] = []
    for measurement in payload["measurements"]:
        image = measurement["image"]
        background = measurement.get("background_region_id")
        for record in measurement["regions"]:
            row = {"image": image}
            row.update(record)
            row["is_background"] = int(record["region_id"] == background)
            region_rows.append(row)
        for record in measurement["pairs"]:
            row = {"image": image}
            row.update(record)
            pair_rows.append(row)
    return region_rows, pair_rows


def _element_regions(region_rows: list[dict]) -> np.ndarray:
    """배경을 뺀 영역의 두께 배열. 자극 굵기 조건의 근거가 된다."""
    return np.asarray(
        [
            row["stroke_width_px"]
            for row in region_rows
            if not row.get("is_background")
        ],
        dtype=float,
    )


def build_conditions(
    payload: dict,
    region_rows: list[dict],
    pair_rows: list[dict],
) -> str:
    """한정훈에게 넘길 2차 실험 조건 문서를 만든다."""
    images = payload["measurements"]
    has_arcmin = any(
        "stroke_min_arcmin" in row for row in pair_rows
    )
    capture_meta = next(
        (
            measurement["capture_meta"]
            for measurement in images
            if measurement.get("capture_meta")
        ),
        None,
    )

    strokes_px = _element_regions(region_rows)
    non_background = [
        row for row in pair_rows if not row.get("involves_background")
    ]
    stroke_pair_px = np.asarray(
        [row["stroke_min_px"] for row in non_background], dtype=float
    )
    centroid_px = np.asarray(
        [row["centroid_distance_px"] for row in non_background], dtype=float
    )
    gap_px = np.asarray(
        [row["bbox_gap_px"] for row in non_background], dtype=float
    )
    adjacent_ratio = (
        float(np.mean([row["adjacent"] for row in non_background]))
        if non_background
        else float("nan")
    )

    lines: list[str] = []
    lines.append("# 2차 실험 자극 조건 (측정 기반)")
    lines.append("")
    lines.append("받는 사람: 한정훈 · 만든 사람: 정인교 (②단계 탐지·측정)")
    lines.append("")
    source_dir = str(payload.get("image_dir", ""))
    lines.append(
        f"측정 대상: `{source_dir}` · 이 문서는 굵기와 거리의 분포이고 "
        "혼동 여부 판정은 들어 있지 않다."
    )
    lines.append("")
    if "synthetic" in source_dir.replace("\\", "/").lower():
        # 합성 이미지로 돌린 결과를 실제 측정값으로 오해하면 실험 조건이
        # 통째로 틀어진다. 문서 첫머리에서 못 박아 둔다.
        lines.append(
            "> **이 수치는 합성 이미지에서 나온 값이다.** 실험 조건을 이대로 "
            "쓰면 안 된다. `data/real/images`를 채우고 다시 돌린 뒤의 값을 "
            "써야 한다."
        )
        lines.append("")

    lines.append("## 데이터셋")
    lines.append("")
    lines.append("| 항목 | 값 |")
    lines.append("|---|---|")
    lines.append(f"| 이미지 수 | {len(images)} |")
    lines.append(f"| 영역 수 | {len(region_rows)} |")
    lines.append(f"| 영역 쌍 수 | {len(pair_rows)} |")
    lines.append(f"| 배경이 끼지 않은 쌍 | {len(non_background)} |")
    lines.append(f"| 맞닿은 쌍 비율 | {adjacent_ratio:.3f} |")
    dropped = sum(
        measurement.get("pairs_dropped", 0) for measurement in images
    )
    if dropped:
        lines.append(
            f"| 거리 상한으로 제외된 쌍 | {dropped} |"
        )
    lines.append("")

    if capture_meta is None:
        lines.append(
            "> **주의** capture_meta가 비어 있어 시각도(arcmin) 환산을 "
            "하지 못했다. 화면 대각 인치·해상도·관찰 거리를 넣고 다시 "
            "돌려야 실험 자극 크기를 정할 수 있다."
        )
    else:
        lines.append("촬영 조건:")
        lines.append("")
        lines.append(
            f"- 화면 대각 {capture_meta['screen_diagonal_inch']}인치, "
            f"해상도 {capture_meta['screen_resolution']}, "
            f"관찰 거리 {capture_meta['viewing_distance_cm']} cm"
        )
        if capture_meta.get("source"):
            lines.append(f"- 출처 메모: {capture_meta['source']}")
    lines.append("")

    lines.append("## 분포 (백분위)")
    lines.append("")
    header = " | ".join(f"{p}%" for p in PERCENTILES)
    lines.append(f"| 항목 | n | {header} |")
    lines.append("|---|---|" + "---|" * len(PERCENTILES))
    lines.append(_format_percentiles("영역 두께", strokes_px, "px", 2))
    lines.append(
        _format_percentiles("쌍의 얇은 쪽 두께", stroke_pair_px, "px", 2)
    )
    lines.append(_format_percentiles("중심 사이 거리", centroid_px, "px", 1))
    lines.append(_format_percentiles("경계상자 간격", gap_px, "px", 1))

    if has_arcmin:
        stroke_arcmin = np.asarray(
            [
                row["stroke_min_arcmin"]
                for row in non_background
                if "stroke_min_arcmin" in row
            ],
            dtype=float,
        )
        centroid_arcmin = np.asarray(
            [
                row["centroid_distance_arcmin"]
                for row in non_background
                if "centroid_distance_arcmin" in row
            ],
            dtype=float,
        )
        lines.append(
            _format_percentiles("쌍의 얇은 쪽 두께", stroke_arcmin, "arcmin", 2)
        )
        lines.append(
            _format_percentiles(
                "중심 사이 거리", centroid_arcmin, "arcmin", 1
            )
        )
    lines.append("")

    lines.append("## 제안하는 자극 조건")
    lines.append("")
    lines.append(
        "굵기는 5%와 50%, 거리는 25%와 75% 두 수준씩 잡으면 2×2 네 조건이 "
        "된다. 얇고 먼 조건과 두껍고 가까운 조건 사이에서 역치가 얼마나 "
        f"올라가는지를 재면, 1차 실험의 최적 조건 값 "
        f"{MEASURED_OPTIMAL_THRESHOLD}에 곱할 환산 비율을 구할 수 있다."
    )
    lines.append("")
    lines.append("| 조건 | 굵기 | 중심 거리 |")
    lines.append("|---|---|---|")
    stroke_levels = _percentile_row(stroke_pair_px)
    distance_levels = _percentile_row(centroid_px)
    for stroke_name, stroke_key in (("얇음(5%)", "p5"), ("보통(50%)", "p50")):
        for distance_name, distance_key in (
            ("가까움(25%)", "p25"),
            ("멂(75%)", "p75"),
        ):
            lines.append(
                f"| {stroke_name} × {distance_name} | "
                f"{stroke_levels[stroke_key]:.2f} px | "
                f"{distance_levels[distance_key]:.1f} px |"
            )
    lines.append("")
    lines.append(
        f"환산 비율 = (그 조건에서 잰 역치) / {MEASURED_OPTIMAL_THRESHOLD}. "
        "이 비율이 나오면 ②단계 측정값에 곱해 쓸 기준을 만들 수 있다."
    )
    lines.append("")

    lines.append("## 읽을 때 주의할 점")
    lines.append("")
    lines.append(
        f"- {MEASURED_OPTIMAL_THRESHOLD}는 큰 색면 두 개를 맞붙여 본 최적 "
        "조건의 값이다. 위 표의 얇은 선에는 그대로 쓸 수 없다."
    )
    lines.append(
        "- 두께는 거리변환으로 추정한 값이다. 덩어리 모양 영역에서는 "
        "가장 좁은 폭이 아니라 내접원 지름을 준다."
    )
    lines.append(
        "- 경계상자 간격은 실제 픽셀 사이 거리보다 작게 나온다. "
        "대각선으로 뻗은 선에서 특히 그렇다."
    )
    lines.append(
        "- 안티앨리어싱 때문에 얇은 영역의 대표색에는 배경색이 섞인다. "
        "``color_spread`` 열이 큰 영역은 색 신뢰도가 낮다."
    )
    lines.append("")
    return "\n".join(lines)


def export_all(
    json_path: str | Path,
    out_dir: str | Path,
    *,
    verbose: bool = True,
) -> dict[str, str]:
    """측정 JSON에서 CSV 두 개와 conditions.md를 만든다."""
    payload = _load(json_path)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    region_rows, pair_rows = collect_rows(payload)
    region_count = _write_csv(region_rows, output / "regions.csv")
    pair_count = _write_csv(pair_rows, output / "pairs.csv")

    conditions_path = output / "conditions.md"
    conditions_path.write_text(
        build_conditions(payload, region_rows, pair_rows),
        encoding="utf-8",
        newline=LF,
    )

    if verbose:
        print(f"regions.csv    {region_count}행")
        print(f"pairs.csv      {pair_count}행")
        print(f"conditions.md  {conditions_path.stat().st_size}바이트")
    return {
        "regions": str(output / "regions.csv"),
        "pairs": str(output / "pairs.csv"),
        "conditions": str(conditions_path),
    }


__all__ = [
    "MEASURED_OPTIMAL_THRESHOLD",
    "PERCENTILES",
    "build_conditions",
    "collect_rows",
    "export_all",
]
