"""증류된 학생 모델: (색 A, 색 B, 사용자, 목표 C80) → 보정색. 실행 시 numpy만 쓴다.

학습은 ``train_student.py``(PyTorch)가 하고, 가중치를 ``student.npz``로 내보낸다.
실시간 경로에서는 이 파일의 numpy 순전파를 쓴다.
학생의 답은 그대로 믿지 않는다: 8-bit로 반올림한 뒤 Machado 후 ΔE00 ≥ 목표를
다시 계산하고, 모자라면 이동량을 키워 다시 확인한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import common as c

FEATURE_NAMES = (
    "okA_L", "okA_a", "okA_b", "okB_L", "okB_a", "okB_b",
    "log_target", "severity", "is_protan", "is_deutan", "log_sim_over_target", "log_orig",
)
OUTPUT_SCALE = 10.0  # ΔOKLab이 작아서 학습할 때 10배로 키운다


def make_features(a_rgb, b_rgb, cvd_type, severity, target) -> np.ndarray:
    """a_rgb, b_rgb: (N,3) 0~255. cvd_type/severity/target: 스칼라 또는 (N,)."""
    a_rgb = np.atleast_2d(np.asarray(a_rgb, float))
    b_rgb = np.atleast_2d(np.asarray(b_rgb, float))
    n = len(a_rgb)
    types = np.broadcast_to(np.asarray(cvd_type), (n,))
    sev = np.broadcast_to(np.asarray(severity, float), (n,))
    tgt = np.broadcast_to(np.asarray(target, float), (n,))
    if a_rgb.shape != (n, 3) or b_rgb.shape != (n, 3) or n == 0:
        raise ValueError('colors must be nonempty matching (N,3) arrays')
    if not np.all(np.isfinite(a_rgb)) or not np.all(np.isfinite(b_rgb)) or np.any(a_rgb < 0) or np.any(a_rgb > 255) or np.any(b_rgb < 0) or np.any(b_rgb > 255):
        raise ValueError('RGB values must be finite and in [0,255]')
    if not np.all(np.isin(types, ('protan', 'deutan'))):
        raise ValueError('cvd_type must be protan or deutan')
    if not np.all(np.isfinite(sev)) or np.any((sev < 0) | (sev > 1)):
        raise ValueError('severity must be finite and in [0,1]')
    if not np.all(np.isfinite(tgt)) or np.any(tgt <= 0):
        raise ValueError('target must be finite and positive')
    sim = np.empty(n)
    for t in ("protan", "deutan"):
        rows = np.flatnonzero(types == t)
        if len(rows):
            mats = np.stack([np.asarray(c.get_machado_matrix(t, float(s))) for s in sev[rows]])
            sim[rows] = c.delta_e_2000(c.simulate_lab_batch(a_rgb[rows], mats), c.simulate_lab_batch(b_rgb[rows], mats))
    orig = c.original_delta_e(a_rgb, b_rgb)
    return np.column_stack(
        [
            c.srgb_to_oklab(a_rgb),
            c.srgb_to_oklab(b_rgb),
            np.log(tgt),
            sev,
            (types == "protan").astype(float),
            (types == "deutan").astype(float),
            np.log((sim + 1e-3) / tgt),
            np.log(orig + 1e-3),
        ]
    )


class Student:
    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as z:
            self.mean, self.std = z['mean'], z['std']
            n_layers = int(z['n_layers'])
            if not 1 <= n_layers <= 16:
                raise ValueError('invalid layer count')
            self.layers = [(z[f'W{i}'], z[f'b{i}']) for i in range(n_layers)]
        if self.mean.shape != (12,) or self.std.shape != (12,) or not np.all(self.std > 0):
            raise ValueError('invalid feature normalization')
        width = 12
        for w, b in self.layers:
            if w.ndim != 2 or w.shape[1] != width or b.shape != (w.shape[0],):
                raise ValueError('incompatible model shapes')
            width = w.shape[0]
        if width != 6 or not all(np.isfinite(v).all() for v in (self.mean, self.std, *(v for layer in self.layers for v in layer))):
            raise ValueError('invalid model output or nonfinite weights')

    def predict_delta(self, feats: np.ndarray) -> np.ndarray:
        """(N,12) → (N,6) ΔOKLab [A(3), B(3)]."""
        h = (feats - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers) - 1:
                h = h * (h > 0) + 0.01 * h * (h <= 0)  # LeakyReLU(0.01) — train_student.py와 같아야 함
        return h / OUTPUT_SCALE

    def correct(self, a_rgb, b_rgb, cvd_type, severity, target, scales=None):
        """보정색과 통과 여부. 학생이 낸 '방향'은 유지하고 이동 크기만 여러 배율로 동시에
        시험해 8-bit 재검사(Machado 후 ΔE00 ≥ 목표)를 통과하는 가장 작은 배율을 고른다.
        반환: a_new(N,3), b_new(N,3), ok(N,), scale_used(N,; 실패는 nan)"""
        scales = np.geomspace(0.5, 6.0, 24) if scales is None else np.asarray(scales, float)
        if scales.ndim != 1 or len(scales) == 0 or not np.all(np.isfinite(scales)) or np.any(scales <= 0) or np.any(np.diff(scales) <= 0):
            raise ValueError('scales must be a nonempty increasing positive finite grid')
        a_rgb = np.atleast_2d(np.asarray(a_rgb, float))
        b_rgb = np.atleast_2d(np.asarray(b_rgb, float))
        n, m = len(a_rgb), len(scales)
        feats = make_features(a_rgb, b_rgb, cvd_type, severity, target)
        delta = self.predict_delta(feats)
        tgt = np.broadcast_to(np.asarray(target, float), (n,))
        types = np.broadcast_to(np.asarray(cvd_type), (n,))
        sev = np.broadcast_to(np.asarray(severity, float), (n,))
        ok_a = feats[:, None, 0:3] + scales[None, :, None] * delta[:, None, :3]
        ok_b = feats[:, None, 3:6] + scales[None, :, None] * delta[:, None, 3:]
        ca = np.rint(c.oklab_to_srgb(ok_a)).clip(0, 255).reshape(-1, 3)
        cb = np.rint(c.oklab_to_srgb(ok_b)).clip(0, 255).reshape(-1, 3)
        sim = np.empty(n * m)
        rep_types, rep_sev = np.repeat(types, m), np.repeat(sev, m)
        for t in ("protan", "deutan"):
            r = np.flatnonzero(rep_types == t)
            if len(r):
                mats = np.stack([np.asarray(c.get_machado_matrix(t, float(s))) for s in rep_sev[r]])
                sim[r] = c.delta_e_2000(c.simulate_lab_batch(ca[r], mats), c.simulate_lab_batch(cb[r], mats))
        passed = (sim.reshape(n, m) >= tgt[:, None])
        ok = passed.any(axis=1)
        first = np.where(ok, passed.argmax(axis=1), 0)
        pick = np.arange(n) * m + first
        a_new = np.where(ok[:, None], ca[pick], a_rgb)
        b_new = np.where(ok[:, None], cb[pick], b_rgb)
        return a_new, b_new, ok, np.where(ok, scales[first], np.nan)
