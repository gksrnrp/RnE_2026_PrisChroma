"""SLIC 분할과 영역 속성 추출.

②단계(탐지·측정)의 입력을 만든다. 이미지를 superpixel로 나누고 각
영역의 대표색·기하 속성·인접 관계를 계산한다. 혼동 여부는 여기서
판정하지 않는다. 판정은 ①단계 실험으로 임계값이 정해진 뒤의 일이다.

대표색으로 평균 대신 중앙값을 쓰는 이유: 얇은 선은 안티앨리어싱으로
가장자리에 배경색이 섞이는데, 평균은 그 섞인 값에 끌려간다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from color_spaces import srgb_to_lab
from distance import delta_e_2000


# 지원하는 입력 확장자. 나머지는 measure_directory에서 건너뛴다.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass
class Region:
    """superpixel 하나의 측정값 묶음."""

    region_id: int
    rgb: tuple[int, int, int]
    area_px: int
    centroid_xy: tuple[float, float]
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    stroke_width_px: float
    color_spread: float
    touches_border: bool
    lab: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))

    def as_dict(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "r": self.rgb[0],
            "g": self.rgb[1],
            "b": self.rgb[2],
            "area_px": self.area_px,
            "centroid_x": round(self.centroid_xy[0], 2),
            "centroid_y": round(self.centroid_xy[1], 2),
            "bbox_min_row": self.bbox[0],
            "bbox_min_col": self.bbox[1],
            "bbox_max_row": self.bbox[2],
            "bbox_max_col": self.bbox[3],
            "stroke_width_px": round(self.stroke_width_px, 3),
            "color_spread": round(self.color_spread, 3),
            "touches_border": int(self.touches_border),
            "lab_L": round(self.lab[0], 3),
            "lab_a": round(self.lab[1], 3),
            "lab_b": round(self.lab[2], 3),
        }


def load_image(path: str | Path) -> np.ndarray:
    """이미지를 (H, W, 3) uint8 sRGB 배열로 읽는다."""
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


def _slic_labels(
    image: np.ndarray,
    n_segments: int,
    compactness: float,
    sigma: float,
) -> np.ndarray:
    """SLIC 라벨 맵을 0부터 시작하는 정수 배열로 반환한다."""
    from skimage.segmentation import slic

    labels = slic(
        image,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=0,
        channel_axis=-1,
        enforce_connectivity=True,
    )
    return np.ascontiguousarray(labels.astype(np.int64))


def build_adjacency(labels: np.ndarray) -> dict[tuple[int, int], int]:
    """맞닿은 라벨 쌍과 공유 경계 길이(픽셀)를 반환한다.

    키는 항상 작은 라벨이 앞에 오도록 정규화한다.
    """
    horizontal = np.stack(
        (labels[:, :-1].ravel(), labels[:, 1:].ravel()), axis=-1
    )
    vertical = np.stack(
        (labels[:-1, :].ravel(), labels[1:, :].ravel()), axis=-1
    )
    pairs = np.concatenate((horizontal, vertical), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.size == 0:
        return {}
    pairs = np.sort(pairs, axis=1)
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    return {
        (int(left), int(right)): int(count)
        for (left, right), count in zip(unique, counts)
    }


def _stroke_width(mask: np.ndarray) -> float:
    """영역의 대표 두께를 거리변환으로 추정한다.

    선 모양이면 거리변환 최댓값이 선의 반폭이므로 2배가 선 두께가 된다.
    덩어리 모양이면 내접원 지름이 되어 가장 좁은 곳이 아니라 가장 두꺼운
    곳의 폭을 준다는 점에 주의한다.
    """
    from scipy.ndimage import distance_transform_edt

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    distance = distance_transform_edt(padded)
    return float(2.0 * distance.max())


def _median_color(pixels: np.ndarray) -> tuple[int, int, int]:
    median = np.median(pixels, axis=0)
    return tuple(int(min(255, max(0, round(value)))) for value in median)


def _color_spread(pixels: np.ndarray, median: tuple[int, int, int]) -> float:
    """대표색 신뢰도 지표. 채널별 중앙값 절대편차의 평균이다.

    값이 크면 한 영역 안에 서로 다른 색이 섞여 있다는 뜻이므로 그 영역의
    대표색은 실제 요소의 색과 다를 수 있다.
    """
    deviation = np.abs(pixels.astype(np.int32) - np.asarray(median))
    return float(np.median(deviation, axis=0).mean())


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return False
        self._parent[root_right] = root_left
        return True


def _merge_labels(
    labels: np.ndarray,
    adjacency: dict[tuple[int, int], int],
    colors: np.ndarray,
    areas: np.ndarray,
    merge_thresh: float,
) -> np.ndarray:
    """대표색이 비슷한 인접 영역을 합친 새 라벨 맵을 만든다.

    색차가 작은 경계부터 합치되, 매번 합쳐진 덩어리의 넓이 가중 평균색을
    다시 계산해 그 색으로 다음 판단을 한다. 원래 대표색만 보고 합치면
    안티앨리어싱으로 생긴 중간색 superpixel을 징검다리 삼아 선이 배경까지
    이어져 통째로 사라진다.

    우선순위 큐에 넣은 색차는 합병이 일어나면 옛값이 되므로, 꺼낸 뒤
    현재 덩어리 색으로 다시 재어 값이 커졌으면 다시 넣는다.
    """
    if merge_thresh <= 0.0 or not adjacency:
        return labels

    import heapq

    cluster_lab = np.asarray(
        [srgb_to_lab(color) for color in colors], dtype=float
    )
    cluster_area = np.asarray(areas, dtype=float).copy()
    union_find = _UnionFind(len(colors))

    heap = []
    for left, right in adjacency:
        difference = float(delta_e_2000(cluster_lab[left], cluster_lab[right]))
        if difference < merge_thresh:
            heapq.heappush(heap, (difference, left, right))

    while heap:
        stale_difference, left, right = heapq.heappop(heap)
        root_left, root_right = union_find.find(left), union_find.find(right)
        if root_left == root_right:
            continue
        difference = float(
            delta_e_2000(cluster_lab[root_left], cluster_lab[root_right])
        )
        if difference >= merge_thresh:
            continue
        if difference > stale_difference + 1e-9:
            heapq.heappush(heap, (difference, root_left, root_right))
            continue

        area_left = max(cluster_area[root_left], 1.0)
        area_right = max(cluster_area[root_right], 1.0)
        merged_lab = (
            cluster_lab[root_left] * area_left
            + cluster_lab[root_right] * area_right
        ) / (area_left + area_right)
        union_find.union(root_left, root_right)
        root = union_find.find(root_left)
        cluster_lab[root] = merged_lab
        cluster_area[root] = area_left + area_right

    roots = np.asarray(
        [union_find.find(index) for index in range(len(colors))]
    )
    _, remapped = np.unique(roots, return_inverse=True)
    return remapped[labels]


def describe_regions(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    min_area_px: int = 4,
) -> list[Region]:
    """라벨 맵에서 영역별 대표색과 기하 속성을 계산한다."""
    height, width = labels.shape
    flat_labels = labels.ravel()
    flat_pixels = image.reshape(-1, 3)
    order = np.argsort(flat_labels, kind="stable")
    sorted_labels = flat_labels[order]
    label_count = int(sorted_labels[-1]) + 1
    boundaries = np.searchsorted(sorted_labels, np.arange(label_count + 1))

    regions: list[Region] = []
    for label in range(label_count):
        indices = order[boundaries[label]:boundaries[label + 1]]
        if indices.size < min_area_px:
            continue
        rows, columns = np.divmod(indices, width)
        pixels = flat_pixels[indices]
        rgb = _median_color(pixels)
        min_row, max_row = int(rows.min()), int(rows.max())
        min_col, max_col = int(columns.min()), int(columns.max())
        mask = np.zeros(
            (max_row - min_row + 1, max_col - min_col + 1), dtype=bool
        )
        mask[rows - min_row, columns - min_col] = True
        regions.append(
            Region(
                region_id=int(label),
                rgb=rgb,
                area_px=int(indices.size),
                centroid_xy=(float(columns.mean()), float(rows.mean())),
                bbox=(min_row, min_col, max_row, max_col),
                stroke_width_px=_stroke_width(mask),
                color_spread=_color_spread(pixels, rgb),
                touches_border=(
                    min_row == 0
                    or min_col == 0
                    or max_row == height - 1
                    or max_col == width - 1
                ),
                lab=tuple(srgb_to_lab(rgb)),
            )
        )
    return regions


def estimate_background(regions: list[Region]) -> int | None:
    """테두리에 닿은 영역 중 가장 넓은 것을 배경으로 본다.

    한계: 화면 가장자리를 큰 색면이 차지하는 그림에서는 배경이 아닌
    영역을 배경으로 잘못 고를 수 있다.
    """
    candidates = [region for region in regions if region.touches_border]
    if not candidates:
        return None
    return max(candidates, key=lambda region: region.area_px).region_id


def extract_regions(
    image: np.ndarray,
    *,
    n_segments: int = 800,
    compactness: float = 3.0,
    sigma: float = 1.0,
    merge_thresh: float = 0.0,
    min_area_px: int = 4,
) -> tuple[np.ndarray, list[Region], dict[tuple[int, int], int]]:
    """이미지 하나를 영역 목록과 인접 그래프로 바꾼다.

    Returns:
        ``(labels, regions, adjacency)``. adjacency의 값은 두 영역이
        공유하는 경계 길이(픽셀)다.
    """
    labels = _slic_labels(image, n_segments, compactness, sigma)
    adjacency = build_adjacency(labels)

    if merge_thresh > 0.0:
        provisional = describe_regions(image, labels, min_area_px=1)
        colors = np.zeros((int(labels.max()) + 1, 3), dtype=int)
        areas = np.ones(int(labels.max()) + 1, dtype=float)
        for region in provisional:
            colors[region.region_id] = region.rgb
            areas[region.region_id] = float(region.area_px)
        labels = _merge_labels(
            labels, adjacency, colors, areas, merge_thresh
        )
        adjacency = build_adjacency(labels)

    regions = describe_regions(image, labels, min_area_px=min_area_px)
    return labels, regions, adjacency


def px_to_arcmin(
    size_px: float,
    *,
    screen_diagonal_inch: float,
    screen_resolution: tuple[int, int],
    viewing_distance_cm: float,
) -> float:
    """픽셀 크기를 시각도(arcmin)로 환산한다.

    화면 대각 길이와 해상도로 1 cm당 픽셀 수를 구한 뒤 관찰 거리에서의
    사이각을 계산한다. 정사각 픽셀을 가정한다.
    """
    width, height = screen_resolution
    diagonal_px = math.hypot(float(width), float(height))
    diagonal_cm = float(screen_diagonal_inch) * 2.54
    if diagonal_cm <= 0.0 or float(viewing_distance_cm) <= 0.0:
        raise ValueError("화면 크기와 관찰 거리는 0보다 커야 합니다.")
    px_per_cm = diagonal_px / diagonal_cm
    size_cm = float(size_px) / px_per_cm
    angle_rad = 2.0 * math.atan(size_cm / (2.0 * float(viewing_distance_cm)))
    return math.degrees(angle_rad) * 60.0


__all__ = [
    "IMAGE_SUFFIXES",
    "Region",
    "build_adjacency",
    "describe_regions",
    "estimate_background",
    "extract_regions",
    "load_image",
    "px_to_arcmin",
]
