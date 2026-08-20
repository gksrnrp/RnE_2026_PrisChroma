"""색공간 변환과 색차 지표.

모든 함수는 마지막 축이 3인 numpy 배열을 받아 같은 모양으로 돌려준다.
스칼라 색 하나(shape (3,))도 그대로 쓸 수 있다.

- sRGB      : [0,1] 감마 인코딩된 표시용 값
- linear RGB: [0,1] 선형 광량. Machado 행렬은 반드시 이 공간에서 곱한다.
- CIE Lab   : D65 기준. 평가 지표(ΔE00)용.
- OKLab     : 학습 손실용 (실행계획 v4 §6-1). 여기서는 참고용으로만 제공.
"""

import numpy as np

# sRGB(D65) -> XYZ, IEC 61966-2-1
_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)

# D65 백색점
_WHITE = _RGB2XYZ @ np.ones(3)


def _as3(x):
    a = np.asarray(x, dtype=np.float64)
    if a.shape[-1] != 3:
        raise ValueError(f"마지막 축이 3이어야 한다: {a.shape}")
    return a


# ---------------------------------------------------------------- sRGB <-> linear

def srgb_to_linear(rgb):
    c = _as3(rgb)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin):
    c = _as3(lin)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)


def to_bytes(rgb):
    """sRGB [0,1] -> uint8 (pygame에 넘길 값)."""
    return np.clip(np.round(_as3(rgb) * 255), 0, 255).astype(np.uint8)


def hex_of(rgb):
    """sRGB [0,1] 색 하나 -> '#rrggbb'."""
    r, g, b = to_bytes(rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def in_gamut(rgb, eps=1e-9):
    """sRGB 표현 가능 범위 안인가."""
    c = _as3(rgb)
    return np.all((c >= -eps) & (c <= 1 + eps), axis=-1)


# ---------------------------------------------------------------- Lab

def linear_to_xyz(lin):
    return _as3(lin) @ _RGB2XYZ.T


def xyz_to_lab(xyz):
    t = _as3(xyz) / _WHITE
    d = 6 / 29
    f = np.where(t > d ** 3, np.cbrt(np.clip(t, 1e-12, None)), t / (3 * d ** 2) + 4 / 29)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def srgb_to_lab(rgb):
    return xyz_to_lab(linear_to_xyz(srgb_to_linear(rgb)))


def linear_to_lab(lin):
    return xyz_to_lab(linear_to_xyz(lin))


# ---------------------------------------------------------------- OKLab

_LMS_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_LMS_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])


def linear_to_oklab(lin):
    lms = _as3(lin) @ _LMS_M1.T
    return np.cbrt(lms) @ _LMS_M2.T


def srgb_to_oklab(rgb):
    return linear_to_oklab(srgb_to_linear(rgb))


def oklab_distance(a, b):
    """OKLab 유클리드 거리 (실행계획 §6-1의 학습용 지표)."""
    d = _as3(a) - _as3(b)
    return np.sqrt(np.sum(d * d, axis=-1) + 1e-12)


# ---------------------------------------------------------------- ΔE00

def delta_e00(lab1, lab2, kL=1.0, kC=1.0, kH=1.0):
    """CIEDE2000 색차. 입력은 Lab, 출력은 스칼라 배열.

    Sharma, Wu, Dalal (2005)의 정식화를 따른다.
    """
    lab1, lab2 = _as3(lab1), _as3(lab2)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7 + 1e-30)))

    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    # C'가 0이면 색상각은 정의되지 않는다 -> 0으로 두고 아래 항에서 무력화된다
    h1p = np.where(C1p < 1e-12, 0.0, h1p)
    h2p = np.where(C2p < 1e-12, 0.0, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where(C1p * C2p < 1e-12, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2

    hsum = h1p + h2p
    hdiff = np.abs(h1p - h2p)
    hbp = np.where(hdiff <= 180, hsum / 2,
                   np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2))
    hbp = np.where(C1p * C2p < 1e-12, hsum, hbp)

    T = (1
         - 0.17 * np.cos(np.radians(hbp - 30))
         + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6))
         - 0.20 * np.cos(np.radians(4 * hbp - 63)))

    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7 + 1e-30))
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc

    return np.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )


def delta_e00_srgb(rgb1, rgb2):
    return delta_e00(srgb_to_lab(rgb1), srgb_to_lab(rgb2))


def delta_e00_linear(lin1, lin2):
    return delta_e00(linear_to_lab(lin1), linear_to_lab(lin2))
