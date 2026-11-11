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


def _distance_transform_1d(values: np.ndarray) -> np.ndarray:
    """Felzenszwalb--Huttenlocher의 1차원 squared EDT를 계산한다."""
    length = values.size
    locations = np.empty(length, dtype=np.int64)
    boundaries = np.empty(length + 1, dtype=float)
    result = np.empty(length, dtype=float)
    last = 0
    locations[0] = 0
    boundaries[0] = -np.inf
    boundaries[1] = np.inf

    for position in range(1, length):
        previous = locations[last]
        crossing = (
            (values[position] + position * position)
            - (values[previous] + previous * previous)
        ) / (2.0 * (position - previous))
        while crossing <= boundaries[last]:
            last -= 1
            previous = locations[last]
            crossing = (
                (values[position] + position * position)
                - (values[previous] + previous * previous)
            ) / (2.0 * (position - previous))
        last += 1
        locations[last] = position
        boundaries[last] = crossing
        boundaries[last + 1] = np.inf

    last = 0
    for position in range(length):
        while boundaries[last + 1] < position:
            last += 1
        nearest = locations[last]
        result[position] = (
            (position - nearest) * (position - nearest) + values[nearest]
        )
    return result


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    """SciPy가 없을 때도 같은 픽셀중심 Euclidean EDT를 반환한다."""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        # 연구 실행 환경에는 SciPy를 고정 설치하지만, 핵심 정의와 단위
        # 테스트는 NumPy만 있는 최소 환경에서도 검증할 수 있게 한다.
        height, width = mask.shape
        unreachable = float((height + width + 1) ** 2)
        squared = np.where(mask, unreachable, 0.0)
        horizontal = np.empty_like(squared)
        for row in range(height):
            horizontal[row] = _distance_transform_1d(squared[row])
        transformed = np.empty_like(horizontal)
        for column in range(width):
            transformed[:, column] = _distance_transform_1d(horizontal[:, column])
        return np.sqrt(transformed)
    return np.asarray(distance_transform_edt(mask), dtype=float)


def _stroke_width(mask: np.ndarray) -> float:
    """영역의 최대 국소 두께를 이산 픽셀 폭으로 추정한다.

    픽셀 중심 EDT의 최댓값을 무조건 두 배로 하면 1픽셀 선을 2픽셀로,
    3픽셀 선을 4픽셀로 세는 반 픽셀 경계 오류가 난다. 배경 픽셀 중심까지의
    거리에서 양쪽 반 픽셀을 빼 ``2*r-1``로 바꾸고, 짝수 폭에서만 생기는
    2x2 최대거리 plateau가 있으면 한 픽셀을 더한다. 따라서 충분히 긴
    수평·수직 직선은 홀수와 짝수 폭 모두 정확히 센다.

    굽거나 갈라진 선, 대각선, 덩어리형 영역에서는 이 값이 가장 큰 내접
    국소 지름에 가까운 *조작적 추정치*다. 전체 영역의 평균 두께나 가장
    좁은 통로 폭을 뜻하지 않는다.
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("stroke-width mask는 2차원이어야 합니다.")
    foreground = array.astype(bool, copy=False)
    if not np.any(foreground):
        return 0.0

    padded = np.pad(foreground, 1, mode="constant", constant_values=False)
    distance = _distance_transform(padded)
    radius = float(distance.max())
    maximum_layer = padded & np.isclose(distance, radius, rtol=0.0, atol=1e-12)
    has_even_plateau = bool(
        maximum_layer.shape[0] > 1
        and maximum_layer.shape[1] > 1
        and np.any(
            maximum_layer[:-1, :-1]
            & maximum_layer[1:, :-1]
            & maximum_layer[:-1, 1:]
            & maximum_layer[1:, 1:]
        )
    )
    return float(max(1.0, 2.0 * radius - 1.0) + int(has_even_plateau))


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
    현재 덩어리 색으로 다시 잰다. 합병 직후에는 현재 이웃도 모두 다시
    계산하여 처음에는 임계값 밖이던 경계가 새로 안으로 들어오는 경우를
    놓치지 않는다.
    """
    if merge_thresh <= 0.0 or not adjacency:
        return labels

    import heapq

    cluster_lab = np.asarray(
        [srgb_to_lab(color) for color in colors], dtype=float
    )
    cluster_area = np.asarray(areas, dtype=float).copy()
    union_find = _UnionFind(len(colors))

    neighbors = [set() for _ in range(len(colors))]
    heap = []
    for left, right in adjacency:
        neighbors[left].add(right)
        neighbors[right].add(left)
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
        if abs(difference - stale_difference) > 1e-9:
            heapq.heappush(heap, (difference, root_left, root_right))
            continue

        merged_neighbors = {
            union_find.find(neighbor)
            for neighbor in neighbors[root_left] | neighbors[root_right]
        }
        merged_neighbors.discard(root_left)
        merged_neighbors.discard(root_right)
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
        neighbors[root] = merged_neighbors
        retired_root = root_right if root == root_left else root_left
        neighbors[retired_root] = set()

        # 처음에는 threshold 밖이던 B-C 경계도 A-B 합병으로 대표색이
        # 바뀌면 새로 threshold 안에 들어올 수 있다. 합병된 덩어리의 모든
        # 현재 이웃을 다시 계산해 그런 경계를 빠뜨리지 않는다.
        for neighbor in sorted(merged_neighbors):
            neighbor_root = union_find.find(neighbor)
            if neighbor_root == root:
                continue
            neighbors[neighbor_root].discard(root_left)
            neighbors[neighbor_root].discard(root_right)
            neighbors[neighbor_root].add(root)
            new_difference = float(
                delta_e_2000(cluster_lab[root], cluster_lab[neighbor_root])
            )
            if new_difference < merge_thresh:
                left_key, right_key = sorted((root, neighbor_root))
                heapq.heappush(
                    heap, (new_difference, left_key, right_key)
                )

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
