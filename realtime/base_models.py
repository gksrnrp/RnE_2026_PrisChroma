"""기준(base) 모델: 화면 전체를 바꾸는 기존 실시간 색보정.

1) classic_daltonization_delta: 고전적 달토니제이션(오차를 다른 채널로 옮김).
   시뮬레이터는 연구와 같은 Machado(유형·severity)를 쓴다. 외부 파일 없이 바로 동작.
2) load_ebelin_lut: Ebelin 외(Eurographics 2023, NVIDIA)의 공개 '참조 변환' 이미지로
   만든 LUT. 연구 페이지에서 zip을 직접 내려받아 convert 후 쓴다(라이선스가 명시되지
   않았으므로 비교 실험에만 쓰고 저장소에 올리지 않는다).
   https://research.nvidia.com/labs/rtr/publication/ebelin2023luminance

모든 LUT는 64³ 격자(채널값 >> 2)의 sRGB 변화량(int16, H×W 적용 시 frame + delta)으로 통일한다.
우리 모델은 이 기준 모델을 '개인·굵기별 C80으로 혼동되는 색 근처에서만' 쓰도록 바꾼다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import common as c

GRID = 64
STEP = 256 // GRID


def grid_rgb() -> np.ndarray:
    """64³ 격자 셀의 대표색 (262144, 3)."""
    v = np.arange(GRID) * STEP + (STEP - 1) / 2.0
    r, g, b = np.meshgrid(v, v, v, indexing="ij")
    return np.stack([r, g, b], -1).reshape(-1, 3)


def classic_daltonization_delta(cvd_type: str, severity: float, strength: float = 1.0) -> np.ndarray:
    """(64,64,64,3) int16 변화량. 선형 RGB에서 err = 원본 − 시뮬, 보정 = M·err."""
    rgb = grid_rgb()
    lin = c._srgb_array_to_linear(rgb)
    mat = np.asarray(c.get_machado_matrix(cvd_type, severity))
    sim = np.clip(lin @ mat.T, 0, 1)
    err = lin - sim
    shift = np.array([[0.0, 0.0, 0.0], [0.7, 1.0, 0.0], [0.7, 0.0, 1.0]])
    out = c._linear_array_to_srgb(np.clip(lin + strength * err @ shift.T, 0, 1))
    return np.rint(out - rgb).astype(np.int16).reshape(GRID, GRID, GRID, 3)


def lut_from_image_pair(original_png: str | Path, transformed_png: str | Path) -> np.ndarray:
    """'원본 색 이미지'와 '변환된 이미지'의 같은 위치 픽셀 대응으로 64³ 변화량 LUT를 만든다."""
    from PIL import Image

    src = np.asarray(Image.open(original_png).convert("RGB"), dtype=np.int64).reshape(-1, 3)
    dst = np.asarray(Image.open(transformed_png).convert("RGB"), dtype=np.int64).reshape(-1, 3)
    if src.shape != dst.shape:
        raise ValueError("두 이미지의 크기가 같아야 합니다.")
    cell = (src[:, 0] // STEP) * GRID * GRID + (src[:, 1] // STEP) * GRID + src[:, 2] // STEP
    n = GRID**3
    cnt = np.bincount(cell, minlength=n)
    delta = np.stack([np.bincount(cell, weights=(dst - src)[:, k], minlength=n) for k in range(3)], -1)
    delta = delta / np.maximum(cnt, 1)[:, None]
    if (cnt == 0).any():
        raise ValueError(f"색이 없는 격자 칸이 {int((cnt == 0).sum())}개입니다. 전체 24-bit 색을 담은 이미지인지 확인하세요.")
    return np.rint(delta).astype(np.int16).reshape(GRID, GRID, GRID, 3)


def save_lut(delta: np.ndarray, path: str | Path) -> None:
    np.save(path, delta)


def load_lut(path: str | Path) -> np.ndarray:
    return np.load(path).astype(np.int16)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Ebelin 참조 변환 이미지 → 64³ LUT(.npy)")
    p.add_argument("--original", required=True)
    p.add_argument("--transformed", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    save_lut(lut_from_image_pair(a.original, a.transformed), a.out)
    print("saved", a.out)
