"""색보정 baseline에서 사용하는 색공간 변환 함수.

공개 함수의 RGB 입력과 출력은 모두 0~255 범위의 sRGB 값이다.
CIELAB은 D65 백색점을 사용하고, OKLab은 0~1 부근의 좌표계를 사용한다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


RGB = tuple[float, float, float]
Color3 = tuple[float, float, float]


def validate_rgb(rgb: Sequence[int | float]) -> RGB:
    """RGB가 0~255 범위의 숫자 세 개인지 검사한다."""
    if isinstance(rgb, (str, bytes)):
        raise TypeError("rgb는 R, G, B 숫자 3개로 구성된 시퀀스여야 합니다.")
    try:
        channels = list(rgb)
    except TypeError as exc:
        raise TypeError(
            "rgb는 R, G, B 숫자 3개로 구성된 시퀀스여야 합니다."
        ) from exc
    if len(channels) != 3:
        raise ValueError("rgb에는 R, G, B 값이 정확히 3개 있어야 합니다.")

    result: list[float] = []
    for channel in channels:
        if isinstance(channel, bool):
            raise TypeError("RGB 채널은 0~255 사이의 숫자여야 합니다.")
        try:
            value = float(channel)
        except (TypeError, ValueError) as exc:
            raise TypeError("RGB 채널은 0~255 사이의 숫자여야 합니다.") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 255.0:
            raise ValueError("RGB 채널은 0~255 범위여야 합니다.")
        result.append(value)
    return result[0], result[1], result[2]


def srgb_channel_to_linear(channel_255: float) -> float:
    """0~255 sRGB 채널을 0~1 선형 RGB 채널로 변환한다."""
    channel = channel_255 / 255.0
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def linear_channel_to_srgb(channel: float, *, clip: bool = True) -> float:
    """선형 RGB 채널을 0~255 sRGB 채널로 변환한다."""
    if clip:
        channel = min(1.0, max(0.0, channel))
    if channel <= 0.0031308:
        encoded = 12.92 * channel
    else:
        encoded = 1.055 * (channel ** (1.0 / 2.4)) - 0.055
    return encoded * 255.0


def srgb_to_linear_rgb(rgb: Sequence[int | float]) -> Color3:
    red, green, blue = validate_rgb(rgb)
    return tuple(srgb_channel_to_linear(c) for c in (red, green, blue))  # type: ignore[return-value]


def linear_rgb_to_srgb(
    linear_rgb: Sequence[int | float], *, clip: bool = True
) -> RGB:
    if len(linear_rgb) != 3:
        raise ValueError("linear_rgb에는 값이 정확히 3개 있어야 합니다.")
    values = tuple(float(c) for c in linear_rgb)
    if not all(math.isfinite(c) for c in values):
        raise ValueError("linear_rgb 값은 유한한 숫자여야 합니다.")
    return tuple(linear_channel_to_srgb(c, clip=clip) for c in values)  # type: ignore[return-value]


def srgb_to_xyz(rgb: Sequence[int | float]) -> Color3:
    """sRGB를 D65 CIE XYZ(0~1 스케일)로 변환한다."""
    red, green, blue = srgb_to_linear_rgb(rgb)
    return (
        0.4124564 * red + 0.3575761 * green + 0.1804375 * blue,
        0.2126729 * red + 0.7151522 * green + 0.0721750 * blue,
        0.0193339 * red + 0.1191920 * green + 0.9503041 * blue,
    )


def srgb_to_lab(rgb: Sequence[int | float]) -> Color3:
    """sRGB를 D65 CIELAB으로 변환한다."""
    x, y, z = srgb_to_xyz(rgb)
    x /= 0.95047
    z /= 1.08883

    delta = 6.0 / 29.0

    def f(value: float) -> float:
        if value > delta**3:
            return value ** (1.0 / 3.0)
        return value / (3.0 * delta**2) + 4.0 / 29.0

    fx, fy, fz = f(x), f(y), f(z)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def lab_to_xyz(lab: Sequence[int | float]) -> Color3:
    """D65 CIELAB을 CIE XYZ(0~1 스케일)로 역변환한다."""
    if isinstance(lab, (str, bytes)) or len(lab) != 3:
        raise ValueError("lab에는 L*, a*, b* 값이 정확히 3개 있어야 합니다.")
    lightness, a, b = (float(value) for value in lab)
    if not all(math.isfinite(value) for value in (lightness, a, b)):
        raise ValueError("CIELAB 값은 유한한 숫자여야 합니다.")

    fy = (lightness + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    delta = 6.0 / 29.0

    def inverse_f(value: float) -> float:
        if value > delta:
            return value**3
        return 3.0 * delta**2 * (value - 4.0 / 29.0)

    return (
        0.95047 * inverse_f(fx),
        inverse_f(fy),
        1.08883 * inverse_f(fz),
    )


def xyz_to_linear_rgb(xyz: Sequence[int | float]) -> Color3:
    """D65 CIE XYZ(0~1 스케일)를 선형 sRGB로 변환한다.

    이 함수는 색역 밖 값을 자르지 않는다. 따라서 반환 채널이 0~1 밖이면
    해당 XYZ 좌표는 sRGB 색역 밖에 있는 것이다.
    """
    if isinstance(xyz, (str, bytes)) or len(xyz) != 3:
        raise ValueError("xyz에는 X, Y, Z 값이 정확히 3개 있어야 합니다.")
    x, y, z = (float(value) for value in xyz)
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError("XYZ 값은 유한한 숫자여야 합니다.")
    return (
        3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
        -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
        0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
    )


def lab_to_linear_rgb(lab: Sequence[int | float]) -> Color3:
    """D65 CIELAB을 clipping하지 않은 선형 sRGB로 변환한다."""
    return xyz_to_linear_rgb(lab_to_xyz(lab))


def is_lab_in_srgb_gamut(
    lab: Sequence[int | float], *, tolerance: float = 1e-6
) -> bool:
    """D65 CIELAB 좌표를 clipping 없이 sRGB로 표시할 수 있는지 확인한다."""
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance는 0 이상의 유한한 숫자여야 합니다.")
    return all(
        -tolerance <= channel <= 1.0 + tolerance
        for channel in lab_to_linear_rgb(lab)
    )


def lab_to_srgb(lab: Sequence[int | float]) -> RGB:
    """D65 CIELAB을 0~255 sRGB로 역변환한다.

    색역 밖 좌표는 clipping하지 않고 ``ValueError``로 알린다. 경계에서
    생기는 1e-7 수준의 행렬 왕복 오차만 허용한다.
    """
    linear = lab_to_linear_rgb(lab)
    if not all(-1e-6 <= channel <= 1.0 + 1e-6 for channel in linear):
        raise ValueError("해당 CIELAB 색은 sRGB gamut 밖에 있습니다.")
    return linear_rgb_to_srgb(linear, clip=True)


def srgb_to_oklab(rgb: Sequence[int | float]) -> Color3:
    """sRGB를 OKLab으로 변환한다."""
    red, green, blue = srgb_to_linear_rgb(rgb)
    l = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    m = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    s = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue

    l_root = math.copysign(abs(l) ** (1.0 / 3.0), l)
    m_root = math.copysign(abs(m) ** (1.0 / 3.0), m)
    s_root = math.copysign(abs(s) ** (1.0 / 3.0), s)

    return (
        0.2104542553 * l_root + 0.7936177850 * m_root - 0.0040720468 * s_root,
        1.9779984951 * l_root - 2.4285922050 * m_root + 0.4505937099 * s_root,
        0.0259040371 * l_root + 0.7827717662 * m_root - 0.8086757660 * s_root,
    )


def oklab_to_linear_rgb(oklab: Sequence[int | float]) -> Color3:
    """OKLab을 선형 sRGB로 변환한다. 이 단계에서는 clipping하지 않는다."""
    if len(oklab) != 3:
        raise ValueError("oklab에는 L, a, b 값이 정확히 3개 있어야 합니다.")
    lightness, a, b = (float(value) for value in oklab)
    if not all(math.isfinite(value) for value in (lightness, a, b)):
        raise ValueError("OKLab 값은 유한한 숫자여야 합니다.")

    l_root = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_root = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_root = lightness - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_root**3, m_root**3, s_root**3

    return (
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )


def is_oklab_in_srgb_gamut(
    oklab: Sequence[int | float], *, tolerance: float = 1e-9
) -> bool:
    """OKLab 색이 clipping 없이 sRGB에서 표현 가능한지 확인한다."""
    return all(
        -tolerance <= channel <= 1.0 + tolerance
        for channel in oklab_to_linear_rgb(oklab)
    )


def oklab_to_srgb(
    oklab: Sequence[int | float], *, clip: bool = False
) -> RGB:
    """OKLab을 0~255 sRGB로 변환한다.

    기본값에서는 색역 밖 좌표를 조용히 clipping하지 않고 ``ValueError``로
    알려준다. 표시용 clipping이 꼭 필요할 때만 ``clip=True``를 사용한다.
    """
    linear = oklab_to_linear_rgb(oklab)
    if not clip:
        # 행렬 왕복에서 생기는 1e-7 수준의 부동소수점 오차만 허용한다.
        if not all(-1e-7 <= channel <= 1.0 + 1e-7 for channel in linear):
            raise ValueError("해당 OKLab 색은 sRGB gamut 밖에 있습니다.")
        return linear_rgb_to_srgb(linear, clip=True)
    return linear_rgb_to_srgb(linear, clip=True)


__all__ = [
    "is_lab_in_srgb_gamut",
    "is_oklab_in_srgb_gamut",
    "lab_to_linear_rgb",
    "lab_to_srgb",
    "lab_to_xyz",
    "linear_channel_to_srgb",
    "linear_rgb_to_srgb",
    "oklab_to_linear_rgb",
    "oklab_to_srgb",
    "srgb_channel_to_linear",
    "srgb_to_lab",
    "srgb_to_linear_rgb",
    "srgb_to_oklab",
    "srgb_to_xyz",
    "validate_rgb",
    "xyz_to_linear_rgb",
]
