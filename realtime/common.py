"""실시간 보정 시스템 공용 모듈.

색 변환·Machado 시뮬레이션·CIEDE2000은 연구 저장소 ``src/``의 구현을 그대로
불러 쓴다(값이 연구 분석과 똑같아야 하기 때문). 이 파일은 그 함수들을
배열 단위로 묶고, 사용자 프로파일(C80(s))과 화면 기하(px → 시각도)를 다룬다.

주의: 이 폴더의 결과는 "소프트웨어 시연·계산상 결과"다. 사람이 보정색을
더 잘 구별한다는 근거가 아니다.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from correction import (  # noqa: E402  (연구 코드 재사용)
    _linear_array_to_srgb,
    _oklab_array_to_linear,
    _srgb_array_to_lab,
    _srgb_array_to_linear,
    _srgb_array_to_oklab,
    find_minimum_correction,
)
from distance import delta_e_2000  # noqa: E402
from machado import get_machado_matrix  # noqa: E402,F401

__all__ = [
    "DisplayGeometry",
    "UserProfile",
    "delta_e_2000",
    "find_minimum_correction",
    "git_commit",
    "oklab_to_srgb",
    "simulate_lab",
    "srgb_to_lab",
    "srgb_to_oklab",
]


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return _srgb_array_to_lab(np.asarray(rgb, dtype=float))


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    return _srgb_array_to_oklab(np.asarray(rgb, dtype=float))


def oklab_to_srgb(oklab: np.ndarray) -> np.ndarray:
    """OKLab → 0~255 실수 sRGB. 색역 밖은 잘린다."""
    return _linear_array_to_srgb(np.clip(_oklab_array_to_linear(oklab), 0.0, 1.0))


def in_gamut_oklab(oklab: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    linear = _oklab_array_to_linear(oklab)
    return np.all((linear >= -tol) & (linear <= 1.0 + tol), axis=-1)


def simulate_lab(rgb: np.ndarray, cvd_type: str, severity: float) -> np.ndarray:
    """sRGB 배열을 Machado로 시뮬레이션한 뒤 CIELAB으로 (연구 코드와 같은 순서)."""
    matrix = np.asarray(get_machado_matrix(cvd_type, severity), dtype=float)
    linear = _srgb_array_to_linear(np.asarray(rgb, dtype=float))
    simulated = np.clip(linear @ matrix.T, 0.0, 1.0)
    return _srgb_array_to_lab(_linear_array_to_srgb(simulated))


def simulate_lab_batch(rgb: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    """행마다 다른 Machado 행렬(N×3×3)을 적용한다(사용자별 severity가 다를 때)."""
    linear = _srgb_array_to_linear(np.asarray(rgb, dtype=float))
    simulated = np.clip(np.einsum("nij,nj->ni", matrices, linear), 0.0, 1.0)
    return _srgb_array_to_lab(_linear_array_to_srgb(simulated))


def simulated_delta_e(a: np.ndarray, b: np.ndarray, cvd_type: str, severity: float) -> np.ndarray:
    return delta_e_2000(simulate_lab(a, cvd_type, severity), simulate_lab(b, cvd_type, severity))


def original_delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return delta_e_2000(srgb_to_lab(a), srgb_to_lab(b))


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')


def _positive_number(value, name):
    _finite_number(value, name)
    if value <= 0:
        raise ValueError(f'{name} must be positive')


@dataclass(frozen=True)
class DisplayGeometry:
    """px를 시각도로 바꾸기 위한 화면 조건. 값은 실제 장치에서 직접 잰다."""

    width_mm: float
    resolution_px: int
    viewing_distance_mm: float

    def __post_init__(self):
        for name in ('width_mm', 'resolution_px', 'viewing_distance_mm'):
            _positive_number(getattr(self, name), name)
        if int(self.resolution_px) != self.resolution_px:
            raise ValueError('resolution_px must be an integer')

    def px_to_deg(self, px: np.ndarray | float) -> np.ndarray | float:
        pitch = self.width_mm / self.resolution_px
        return np.degrees(2.0 * np.arctan(np.asarray(px) * pitch / (2.0 * self.viewing_distance_mm)))


@dataclass(frozen=True)
class UserProfile:
    """사용자 색각 프로파일.

    C80(s) = c80_ref * (s / s_ref_deg) ** beta  (D384 형식, s는 시각도).
    측정 범위 [s_min_deg, s_max_deg] 밖은 외삽하지 않고 끝값으로 고정한다.
    ``source``에는 측정 또는 합성 설정의 출처를 반드시 적는다.
    ``original_min``: 원래부터 비슷한 색을 색각이상 혼동으로 세지 않기 위한
    원본 ΔE00 하한. 연구 임계값이 아니라 시스템 설정값이며 출처와 함께 둔다.
    """

    participant_id: str
    cvd_type: str
    severity: float
    c80_ref: float
    s_ref_deg: float
    beta: float
    s_min_deg: float
    s_max_deg: float
    original_min: float
    source: str

    def __post_init__(self):
        if self.cvd_type not in ('protan', 'deutan'):
            raise ValueError('cvd_type must be protan or deutan')
        if not isinstance(self.participant_id, str) or not self.participant_id.strip():
            raise ValueError('participant_id is required')
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError('profile source is required')
        for name in ('severity', 'beta', 'original_min'):
            _finite_number(getattr(self, name), name)
        if not 0 <= self.severity <= 1 or self.original_min < 0:
            raise ValueError('severity must be in [0,1]; original_min must be nonnegative')
        for name in ('c80_ref', 's_ref_deg', 's_min_deg', 's_max_deg'):
            _positive_number(getattr(self, name), name)
        if not self.s_min_deg <= self.s_ref_deg <= self.s_max_deg:
            raise ValueError('reference width must be within the measured range')

    def c80(self, s_deg: np.ndarray | float) -> np.ndarray:
        s = np.clip(np.asarray(s_deg, dtype=float), self.s_min_deg, self.s_max_deg)
        return self.c80_ref * (s / self.s_ref_deg) ** self.beta

    def out_of_range(self, s_deg: np.ndarray | float) -> np.ndarray:
        s = np.asarray(s_deg, dtype=float)
        return (s < self.s_min_deg) | (s > self.s_max_deg)

    def with_fixed_threshold(self) -> "UserProfile":
        """굵기 효과를 끈 비교용 프로파일(ablation A3): 모든 굵기에 C80_ref."""
        data = asdict(self)
        data.update(beta=0.0, participant_id=self.participant_id + "-fixed", source=self.source + " / beta=0")
        return UserProfile(**data)

    @staticmethod
    def load(path: str | Path) -> "UserProfile":
        return UserProfile(**json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - Colab 등 git 정보가 없는 환경
        return "unknown"
