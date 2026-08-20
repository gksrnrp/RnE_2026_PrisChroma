"""보고서에 넣을 그림과 표를 만든다.

측정과 분석이 끝난 뒤 실행한다. 앞 단계가 남긴 CSV를 읽어 그리므로
이 파일은 새 수치를 만들지 않는다.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


# 한글이 깨지지 않게 윈도우 기본 한글 글꼴을 먼저 찾는다.
KOREAN_FONTS = ("Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim")


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in available:
            pyplot.rcParams["font.family"] = name
            break
    pyplot.rcParams["axes.unicode_minus"] = False
    return pyplot


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def figure_pipeline(out_path: Path) -> None:
    """②단계 파이프라인 구성도."""
    pyplot = _setup_matplotlib()
    figure, axes = pyplot.subplots(figsize=(11, 2.6))
    axes.set_xlim(0, 11)
    axes.set_ylim(0, 2.6)
    axes.axis("off")

    steps = (
        ("입력 이미지", "실제 화면 캡처\n합성 이미지"),
        ("SLIC 분할", "superpixel\n+ 유사색 병합"),
        ("영역 속성", "대표색(중앙값)\n두께·면적·위치"),
        ("쌍 측정", "ΔE00 원본\nΔE00 시뮬레이션\n거리·인접"),
        ("출력", "pairs.csv\nregions.csv\nconditions.md"),
    )
    width, height = 1.85, 1.25
    for index, (title, body) in enumerate(steps):
        left = 0.25 + index * 2.15
        axes.add_patch(
            pyplot.Rectangle(
                (left, 0.75),
                width,
                height,
                facecolor="#eef2f7" if index < 4 else "#e6f0e6",
                edgecolor="#33475b",
                linewidth=1.2,
            )
        )
        axes.text(
            left + width / 2, 1.72, title,
            ha="center", va="center", fontsize=11, fontweight="bold",
        )
        axes.text(
            left + width / 2, 1.22, body,
            ha="center", va="center", fontsize=8.5, color="#33475b",
        )
        if index < len(steps) - 1:
            axes.annotate(
                "",
                xy=(left + width + 0.28, 1.37),
                xytext=(left + width + 0.02, 1.37),
                arrowprops={"arrowstyle": "-|>", "color": "#33475b"},
            )
    axes.text(
        5.5, 0.35,
        "혼동 여부는 판정하지 않는다. 임계값은 ①단계 사용자 실험이 정한다.",
        ha="center", va="center", fontsize=9.5, color="#8a3b3b",
    )
    figure.tight_layout()
    figure.savefig(out_path, dpi=200)
    pyplot.close(figure)


def figure_synthetic_examples(
    synthetic_root: Path, out_path: Path
) -> None:
    """합성 이미지 예시 모음."""
    pyplot = _setup_matplotlib()
    from slic_extract import load_image

    names = (
        "lines_w02",
        "lines_w08",
        "route_map",
        "bar_chart",
        "legend_lines_w08",
        "dot_plate",
    )
    figure, axes = pyplot.subplots(2, 3, figsize=(10.5, 6))
    for axis, name in zip(axes.ravel(), names):
        axis.imshow(load_image(synthetic_root / "images" / f"{name}.png"))
        axis.set_title(name, fontsize=10)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(out_path, dpi=170)
    pyplot.close(figure)


def figure_segmentation_contrast(
    synthetic_root: Path, out_path: Path
) -> None:
    """분할이 되는 경우와 안 되는 경우를 나란히 보여준다."""
    pyplot = _setup_matplotlib()
    from failure_analysis import region_color_map
    from slic_extract import extract_regions, load_image

    cases = (
        ("lines_w08", "굵기 8 px — 선이 영역으로 잡힌다"),
        ("lines_w02", "굵기 2 px — 배경에 흡수된다"),
    )
    figure, axes = pyplot.subplots(2, 2, figsize=(10, 7.4))
    for row, (name, caption) in enumerate(cases):
        image = load_image(synthetic_root / "images" / f"{name}.png")
        labels, regions, _ = extract_regions(
            image, n_segments=800, compactness=3.0, merge_thresh=1.0
        )
        axes[row][0].imshow(image)
        axes[row][0].set_title(
            f"{name} 원본\n{caption}", fontsize=10.5, color="#33475b"
        )
        axes[row][1].imshow(region_color_map(image, labels, regions))
        axes[row][1].set_title(
            f"영역 대표색\n{len(regions)}개 영역", fontsize=10.5
        )
        for axis in axes[row]:
            axis.axis("off")
    figure.tight_layout(h_pad=2.0)
    figure.savefig(out_path, dpi=170)
    pyplot.close(figure)


def figure_detection_rate(results_dir: Path, out_path: Path) -> None:
    """n_segments에 따른 굵기별 검출률 그래프."""
    pyplot = _setup_matplotlib()
    rows = _read_csv(results_dir / "n_segments_sweep.csv")
    widths = sorted({float(row["nominal_stroke_width_px"]) for row in rows})
    grid = sorted({int(row["n_segments"]) for row in rows})

    figure, axes = pyplot.subplots(figsize=(8, 5))
    colors = pyplot.cm.viridis(np.linspace(0, 0.9, len(widths)))
    for color, width in zip(colors, widths):
        rates = []
        for value in grid:
            subset = [
                row
                for row in rows
                if float(row["nominal_stroke_width_px"]) == width
                and int(row["n_segments"]) == value
            ]
            detected = sum(int(row["detected"]) for row in subset)
            total = sum(int(row["element_count"]) for row in subset)
            rates.append(100.0 * detected / total if total else 0.0)
        axes.plot(
            grid, rates, marker="o", color=color, label=f"{width:g} px"
        )
    axes.set_xscale("log")
    axes.set_xticks(grid)
    axes.set_xticklabels([str(value) for value in grid])
    axes.set_xlabel("n_segments (로그 눈금)")
    axes.set_ylabel("요소 검출률 (%)")
    axes.set_ylim(-4, 104)
    axes.set_title("선 두께별 검출률 (compactness=3, merge_thresh=1)")
    axes.grid(alpha=0.3)
    axes.legend(title="선 두께", fontsize=9)
    figure.tight_layout()
    figure.savefig(out_path, dpi=180)
    pyplot.close(figure)


def figure_merge_thresh(results_dir: Path, out_path: Path) -> None:
    """merge_thresh에 따른 영역 수와 검출률."""
    pyplot = _setup_matplotlib()
    rows = _read_csv(results_dir / "merge_thresh_sweep.csv")
    grid = sorted({float(row["merge_thresh"]) for row in rows})

    region_counts, detection = [], []
    for value in grid:
        subset = [
            row for row in rows if float(row["merge_thresh"]) == value
        ]
        region_counts.append(
            float(np.mean([int(row["region_count"]) for row in subset]))
        )
        detection.append(
            100.0
            * float(np.mean([float(row["detection_rate"]) for row in subset]))
        )

    figure, axes = pyplot.subplots(figsize=(8, 4.6))
    axes.plot(grid, region_counts, marker="o", color="#33475b")
    axes.set_yscale("log")
    axes.set_xlabel("merge_thresh (ΔE00)")
    axes.set_ylabel("장면당 평균 영역 수 (로그 눈금)", color="#33475b")
    axes.grid(alpha=0.3)

    twin = axes.twinx()
    twin.plot(grid, detection, marker="s", color="#a8442a")
    twin.set_ylabel("장면 평균 검출률 (%)", color="#a8442a")
    twin.set_ylim(0, 100)

    axes.set_title("merge_thresh: 영역 수는 급히 줄고 검출률은 평평해진다")
    figure.tight_layout()
    figure.savefig(out_path, dpi=180)
    pyplot.close(figure)


def dataset_table(
    synthetic_root: Path,
    measurements_json: Path | None,
) -> list[str]:
    """데이터셋 구성 표를 만든다."""
    import json

    lines = ["## 데이터셋 구성", "", "### 합성 이미지", ""]
    manifest = json.loads(
        (synthetic_root / "ground_truth.json").read_text(encoding="utf-8")
    )
    lines.append("| 장면 | 요소 수 | 공칭 굵기 (px) | 용도 |")
    lines.append("|---|---|---|---|")
    purpose = {
        "lines": "굵기별 검출률",
        "route_map": "노선도 모사",
        "bar_chart": "큰 색면",
        "legend_lines": "멀리 떨어진 의미 쌍",
        "legend_lines_w08": "멀리 떨어진 의미 쌍 (굵은 선)",
        "dot_plate": "색각 검사표 모사",
    }
    for name, entry in manifest["images"].items():
        elements = entry["elements"]
        widths = sorted({e["nominal_stroke_width_px"] for e in elements})
        key = "lines" if name.startswith("lines_w") else name
        lines.append(
            f"| {name} | {len(elements)} | "
            f"{', '.join(f'{w:g}' for w in widths[:3])} | "
            f"{purpose.get(key, '')} |"
        )
    lines.append("")

    if measurements_json and measurements_json.exists():
        payload = json.loads(measurements_json.read_text(encoding="utf-8"))
        lines.append("### 실제 이미지")
        lines.append("")
        lines.append("| 이미지 | 영역 수 | 쌍 수 |")
        lines.append("|---|---|---|")
        for measurement in payload["measurements"]:
            lines.append(
                f"| {measurement['image']} | {measurement['region_count']} | "
                f"{measurement.get('pair_count', 0)} |"
            )
        lines.append("")
    else:
        lines.append(
            "실제 이미지 측정 결과가 아직 없다. `data/real/images`를 채우고 "
            "`measure_directory`를 돌린 뒤 이 표를 다시 만든다."
        )
        lines.append("")
    return lines


def timing_table(results_dir: Path) -> list[str]:
    """처리 시간 표. 보고서 본문의 '처리 시간' 항목에 쓴다."""
    rows = _read_csv(results_dir / "n_segments_sweep.csv")
    grid = sorted({int(row["n_segments"]) for row in rows})
    lines = ["## 처리 시간", "", "| n_segments | 이미지 한 장당 초 |", "|---|---|"]
    for value in grid:
        seconds = [
            float(row["seconds"])
            for row in rows
            if int(row["n_segments"]) == value
        ]
        lines.append(f"| {value} | {np.mean(seconds):.2f} |")
    lines.append("")
    lines.append(
        "520×420 이미지 기준이고 분할과 영역 속성 계산까지의 시간이다. "
        "쌍 측정 시간은 영역 수의 제곱에 비례하므로 따로 본다."
    )
    lines.append("")
    return lines


def build_all(
    synthetic_root: str | Path = "../data/synthetic",
    results_dir: str | Path = "../results",
    measurements_json: str | Path | None = None,
    *,
    verbose: bool = True,
) -> str:
    """그림 다섯 장과 표 문서를 한 번에 만든다."""
    synthetic_root = Path(synthetic_root)
    results_dir = Path(results_dir)
    figure_dir = results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "fig_pipeline.png": lambda path: figure_pipeline(path),
        "fig_synthetic_examples.png": lambda path: figure_synthetic_examples(
            synthetic_root, path
        ),
        "fig_segmentation_contrast.png": (
            lambda path: figure_segmentation_contrast(synthetic_root, path)
        ),
        "fig_detection_rate.png": lambda path: figure_detection_rate(
            results_dir, path
        ),
        "fig_merge_thresh.png": lambda path: figure_merge_thresh(
            results_dir, path
        ),
    }
    for name, draw in figures.items():
        draw(figure_dir / name)
        if verbose:
            print(f"그림 저장: {figure_dir / name}")

    lines = ["# 보고서용 표 모음", ""]
    lines.extend(
        dataset_table(
            synthetic_root,
            Path(measurements_json) if measurements_json else None,
        )
    )
    lines.extend(timing_table(results_dir))
    lines.append("## 민감도와 실패 유형")
    lines.append("")
    lines.append(
        "`results/sweep_summary.md`와 `results/failure_summary.md`의 표를 "
        "그대로 쓴다. 여기서 다시 만들지 않는다."
    )
    lines.append("")

    path = results_dir / "report_tables.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    if verbose:
        print(f"표 저장: {path}")
    return str(path)


__all__ = [
    "build_all",
    "dataset_table",
    "figure_detection_rate",
    "figure_merge_thresh",
    "figure_pipeline",
    "figure_segmentation_contrast",
    "figure_synthetic_examples",
    "timing_table",
]
