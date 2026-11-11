"""실시간 개인화 선택 보정 파이프라인.

매 프레임: 변화 감지 → (바뀌었을 때만) 분석·보정·LUT 생성 → LUT 적용.
분석: 색 팔레트(5-bit 히스토그램 + OKLab 병합) → 색별 선 굵기(2·면적/둘레) →
      C80(s) 혼동 판단 → 보정(학생 MLP / 교사 / 기준 모델) → 64³ 변화량 LUT.

mode
  none            보정 안 함
  base_full       기준 모델(고전 달토니제이션 또는 Ebelin LUT)을 화면 전체에 적용
  base_gated      같은 기준 모델을 '혼동 판단된 색 근처에만' 적용 (기존 모델 변형 1)
  student         혼동 쌍에 증류 MLP와 유한 배율 재검사 적용
  teacher         혼동 쌍에 쌍 단위 근사 탐색기 적용 (비교용)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import common as c
from base_models import GRID, STEP, classic_daltonization_delta, grid_rgb

_GRID_OKLAB = None


def grid_oklab() -> np.ndarray:
    global _GRID_OKLAB
    if _GRID_OKLAB is None:
        _GRID_OKLAB = c.srgb_to_oklab(grid_rgb())
    return _GRID_OKLAB


@dataclass
class Settings:
    min_cluster_px: int = 150      # 이보다 적은 픽셀 색은 무시(안티앨리어싱 경계색 제거)
    merge_tol: float = 0.015       # OKLab 거리: 이보다 가까운 색은 한 색으로 병합
    assign_tol: float = 0.05       # 픽셀(빈)을 팔레트 색에 배정하는 최대 거리
    max_colors: int = 10
    min_component_px: int = 150     # 짧은 안티앨리어싱 조각을 얇은 선으로 오인하지 않도록
    width_bias_px: float = 0.5     # 안티앨리어싱 경계가 빠져 생기는 과소추정 보정(평가에서 재보정)
    lut_sigma: float = 0.02        # LUT 이동이 퍼지는 OKLab 반경(가장 가까운 팔레트 색 기준)
    ema: float = 0.5               # 새 LUT 반영 비율(깜빡임 방지)
    target_margin: float = 1.05    # 굵기 추정 오차 대비: 보정 목표 = C80 × 이 값
    change_thresh: float = 1.5     # 썸네일 평균 차이 > 이 값이면 재분석
    aa_unmix: bool = False         # 안티앨리어싱 혼합색 제거 + 1px 선 원래 색 복원
    blend_tol: float = 6.0         # 혼합색 판정: 배경→선 색 직선에서 벗어난 거리(sRGB)
    adjacency_min: float = 0.3     # 혼합색과 원래 색 픽셀이 맞닿은 비율 하한


@dataclass
class Analysis:
    colors: np.ndarray              # (K,3) 팔레트 색 sRGB, 0번이 배경
    counts: np.ndarray
    widths_px: np.ndarray           # (K,) 가장 얇은 요소의 굵기, 배경은 inf
    pairs: list = field(default_factory=list)
    moved: dict = field(default_factory=dict)  # 팔레트 idx → 새 sRGB
    timings: dict = field(default_factory=dict)


class RealtimeCorrector:
    def __init__(self, profile: c.UserProfile, display: c.DisplayGeometry, mode: str = "student",
                 student=None, base_delta: np.ndarray | None = None, settings: Settings | None = None):
        self.p, self.d, self.mode, self.student = profile, display, mode, student
        self.s = settings or Settings()
        self.base_delta = base_delta if base_delta is not None else classic_daltonization_delta(profile.cvd_type, profile.severity)
        self.delta = np.zeros((GRID, GRID, GRID, 3), np.float32)
        self.target_delta = self.delta.copy()
        self.last_thumb = None
        self._version = 0
        self.analysis: Analysis | None = None
        if mode == "base_full":
            self.delta = self.base_delta.astype(np.float32)
            self.target_delta = self.delta.copy()
        if mode == "student" and student is None:
            raise ValueError("student 모드에는 학습된 Student가 필요합니다.")

    # ---------- 분석 ----------
    def palette(self, frame: np.ndarray):
        q = (frame >> 3).astype(np.int32)
        idx = (q[..., 0] << 10 | q[..., 1] << 5 | q[..., 2]).ravel()
        cnt = np.bincount(idx, minlength=32768)
        flat = frame.reshape(-1, 3).astype(np.float64)
        sums = np.stack([np.bincount(idx, weights=flat[:, k], minlength=32768) for k in range(3)], -1)
        bins = np.flatnonzero(cnt >= self.s.min_cluster_px)
        if len(bins) == 0:
            bins = np.array([int(cnt.argmax())])
        bins = bins[np.argsort(-cnt[bins])]
        mean = sums[bins] / cnt[bins, None]
        pair_bins = {}
        self._self_alpha = []
        if self.s.aa_unmix and len(bins) > 2:
            bins, mean, bcnt, pair_bins, explained = self._unmix(idx, cnt, bins, mean, frame.shape[:2])
        else:
            bcnt = cnt[bins]
            explained = []
        ok = c.srgb_to_oklab(mean)
        centers, weights = [], []
        order = np.argsort(-bcnt, kind="stable")
        for k in order:
            if centers:
                dist = np.linalg.norm(np.array(centers) - ok[k], axis=1)
                j = int(np.argmin(dist))
                if dist[j] < self.s.merge_tol:
                    w = weights[j] + bcnt[k]
                    centers[j] = (centers[j] * weights[j] + ok[k] * bcnt[k]) / w
                    weights[j] = w
                    continue
            if len(centers) < self.s.max_colors:
                centers.append(ok[k].copy())
                weights.append(bcnt[k])
        centers = np.array(centers)
        colors = np.rint(c.oklab_to_srgb(centers)).clip(0, 255)
        # 모든 빈 → 가장 가까운 팔레트 색 (멀면 -1)
        nz = np.flatnonzero(cnt)
        ok_nz = c.srgb_to_oklab(sums[nz] / cnt[nz, None])
        dist = np.linalg.norm(ok_nz[:, None, :] - centers[None], axis=2)
        lab = np.full(32768, -1, np.int16)
        near = dist.argmin(1)
        lab[nz] = np.where(dist[np.arange(len(nz)), near] < self.s.assign_tol, near, -1)
        # 혼합색 빈은 원래 색에 배정하고 피복률(alpha)을 기록 → 굵기 = 피복률 가중 면적 기반
        self._alpha = None
        if self.s.aa_unmix:
            alpha = np.ones(32768, np.float32)
            for b, mj, a in explained:
                ci = int(np.argmin(np.linalg.norm(centers - c.srgb_to_oklab(mj[None])[0], axis=1)))
                if ci != 0:
                    lab[b] = ci
                    alpha[b] = a
            for b, a, L in getattr(self, "_self_alpha", []):
                ci = int(np.argmin(np.linalg.norm(centers - c.srgb_to_oklab(L[None])[0], axis=1)))
                if ci != 0:
                    lab[b] = ci   # 복원된 1px 선의 진한 줄도 복원색에 배정
                    alpha[b] = a
            for k, members in pair_bins.items():
                ci = int(np.argmin(np.linalg.norm(centers - ok[k], axis=1)))
                if ci == 0:
                    continue
                for b, a in members:
                    lab[b] = ci
                    alpha[b] = a
            self._alpha = alpha[idx].reshape(frame.shape[:2])
        label = lab[idx].reshape(frame.shape[:2])
        return colors, np.array(weights), label

    def _unmix(self, idx, cnt, bins, mean, shape):
        """안티앨리어싱 혼합색 처리.

        선 경계 픽셀의 색은 배경 B와 선 색 L의 혼합 c = B + a(L - B) (sRGB)이다.
        (1) 더 진한 후보 L과 같은 방향에 있고 픽셀이 맞닿은 후보는 혼합색으로 보고 뺀다.
        (2) 설명되지 않는 두 혼합색이 같은 방향이고 맞닿아 있으면 1px 선이 두 줄에 나뉜 것으로 보고
            L = B + (c1 - B) + (c2 - B)로 원래 색을 복원한다(두 줄의 피복률 합 = 1).
        """
        k_n = len(bins)
        pos = np.full(32768, -1, np.int32)
        pos[bins] = np.arange(k_n)
        lab = pos[idx].reshape(shape)
        co = np.zeros((k_n, k_n))
        for a, b in ((lab[1:, :], lab[:-1, :]), (lab[:, 1:], lab[:, :-1])):
            m = (a >= 0) & (b >= 0) & (a != b)
            np.add.at(co, (a[m], b[m]), 1)
        co = co + co.T
        bc = cnt[bins].astype(float)
        adj = co / np.maximum(bc[:, None], 1)          # adj[k, j]: k 픽셀이 j와 맞닿은 비율
        bg = mean[0]
        dvec = mean - bg
        norm = np.linalg.norm(dvec, axis=1)
        blend = np.zeros(k_n, bool)
        explained, explained_self, expl_by = [], [], {}
        for k in range(1, k_n):
            if norm[k] < 3:
                blend[k] = True                          # 배경과 거의 같은 경계색
                continue
            for j in range(1, k_n):
                if j == k or norm[j] < norm[k] * 1.1 or adj[k, j] < self.s.adjacency_min:
                    continue
                a = float(dvec[k] @ dvec[j]) / float(norm[j] ** 2)
                if 0.03 < a < 0.97 and np.linalg.norm(dvec[k] - a * dvec[j]) < self.s.blend_tol:
                    blend[k] = True
                    expl_by[k] = (j, a)
                    break
        # (1b) 1px 선이 피복률이 다른 두 줄(예: 0.75/0.25)로 나뉜 경우: 진한 줄 j도 혼합색이다.
        #      j가 한쪽은 배경, 다른 쪽은 혼합색 k와 맞닿아 있으면 L = B + d_j + d_k (피복률 합 = 1)
        recovered = {}
        for k, (j, a) in list(expl_by.items()):
            if blend[j] or j in recovered:
                continue
            if adj[j, 0] >= self.s.adjacency_min and adj[k, 0] >= self.s.adjacency_min and adj[j, k] >= self.s.adjacency_min:
                L = bg + dvec[j] + dvec[k]
                if np.all(L > -8) and np.all(L < 263):
                    recovered[j] = (k, np.clip(L, 0, 255))
        for j, (k, L) in recovered.items():
            nL = max(np.linalg.norm(L - bg), 1e-9)
            mean[j] = L
            expl_by[k] = (j, norm[k] / nL)
            self_alpha_j = norm[j] / nL
            explained_self.append((bins[j], self_alpha_j, L.copy()))
        for k, (j, a) in expl_by.items():
            explained.append((bins[k], mean[j], a))
        keep = [k for k in range(k_n) if not blend[k]]
        # (2) 1px 선 복원
        used, pairs = set(), []
        cand = [k for k in keep if k != 0]
        scores = []
        for x in range(len(cand)):
            for y in range(x + 1, len(cand)):
                k, m = cand[x], cand[y]
                if min(adj[k, m], adj[m, k]) < self.s.adjacency_min:
                    continue
                cos = float(dvec[k] @ dvec[m]) / float(norm[k] * norm[m] + 1e-9)
                if cos < 0.98:
                    continue
                L = bg + dvec[k] + dvec[m]
                if np.any(L < -8) or np.any(L > 263):
                    continue
                scores.append((-(adj[k, m] + adj[m, k]), k, m, np.clip(L, 0, 255)))
        for _, k, m, L in sorted(scores, key=lambda t: t[0]):
            if k in used or m in used:
                continue
            used |= {k, m}
            pairs.append((k, m, L))
        new_mean, new_cnt, new_bins = [], [], []
        pair_bins, cov = {}, {}
        for k in keep:
            if k in used:
                continue
            new_bins.append(bins[k]); new_mean.append(mean[k]); new_cnt.append(cnt[bins[k]])
        for k, m, L in pairs:
            ak = norm[k] / max(np.linalg.norm(L - bg), 1e-9)
            am = norm[m] / max(np.linalg.norm(L - bg), 1e-9)
            pair_bins[len(new_mean)] = ((bins[k], ak), (bins[m], am))
            new_bins.append(-1); new_mean.append(L); new_cnt.append(bc[k] + bc[m])
        self._self_alpha = explained_self
        return (np.array(new_bins), np.array(new_mean, float), np.array(new_cnt, float), pair_bins, explained)

    def widths(self, label: np.ndarray, k: int) -> np.ndarray:
        out = np.full(k, np.inf)
        for ci in range(1, k):  # 0 = 배경(가장 넓은 색)
            mask = (label == ci).astype(np.uint8)
            n, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            best = np.inf
            alpha = getattr(self, "_alpha", None)
            if alpha is not None:
                eff = np.bincount(comp.ravel(), weights=(alpha * mask).ravel(), minlength=n)
            for cnt_ in contours:
                y, x = cnt_[0, 0, 1], cnt_[0, 0, 0]
                area = stats[comp[y, x], cv2.CC_STAT_AREA]
                if area < self.s.min_component_px:
                    continue
                perim = max(cv2.arcLength(cnt_, True), 1.0)
                if alpha is not None:   # 피복률 가중 면적: 경계 혼합 픽셀을 부분 픽셀로 셈 → 보정항 불필요
                    best = min(best, max(0.5, 2.0 * eff[comp[y, x]] / perim))
                else:
                    best = min(best, max(1.0, 2.0 * area / perim + self.s.width_bias_px))
            out[ci] = best
        return out

    def judge(self, colors: np.ndarray, widths_px: np.ndarray) -> list[dict]:
        pairs = []
        k = len(colors)
        for i in range(1, k):
            for j in range(i + 1, k):
                if not (np.isfinite(widths_px[i]) and np.isfinite(widths_px[j])):
                    continue
                s_deg = float(self.d.px_to_deg(min(widths_px[i], widths_px[j])))
                need = float(self.p.c80(s_deg))
                sim = float(c.simulated_delta_e(colors[i], colors[j], self.p.cvd_type, self.p.severity))
                orig = float(c.original_delta_e(colors[i], colors[j]))
                pairs.append({"i": i, "j": j, "sim": sim, "orig": orig, "need": need, "s_deg": s_deg,
                              "out_of_range": bool(self.p.out_of_range(s_deg)),
                              "confused": sim < need and orig >= self.p.original_min})
        return pairs

    def correct(self, colors: np.ndarray, pairs: list[dict]) -> dict:
        cur = colors.astype(float).copy()
        moved: dict[int, np.ndarray] = {}
        for pr in sorted((q for q in pairs if q["confused"]), key=lambda q: q["sim"] / q["need"]):
            i, j = pr["i"], pr["j"]
            now = float(c.simulated_delta_e(cur[i], cur[j], self.p.cvd_type, self.p.severity))
            if now >= pr["need"]:
                pr["status"] = "resolved_by_earlier"
                continue
            if self.mode == "student":
                na, nb, ok, _ = self.student.correct(cur[i], cur[j], self.p.cvd_type, self.p.severity, pr["need"] * self.s.target_margin)
                na, nb, ok = na[0], nb[0], bool(ok[0])
            elif self.mode == "teacher":
                others = [cur[k] for k in range(1, len(cur)) if k not in (i, j)]
                r = c.find_minimum_correction(cur[i], cur[j], self.p.cvd_type, self.p.severity, pr["need"] * self.s.target_margin, palette=others or None)
                na, nb, ok = np.array(r["corrected_A"], float), np.array(r["corrected_B"], float), bool(r["success"])
            else:
                continue
            pr["status"] = "corrected" if ok else "failed_kept_original"
            if ok:
                cur[i], cur[j] = na, nb
                if np.any(na != colors[i]):
                    moved[i] = na
                if np.any(nb != colors[j]):
                    moved[j] = nb
        return {i: cur[i].copy() for i in range(len(cur)) if np.any(cur[i] != colors[i])}

    def build_delta(self, colors: np.ndarray, moved: dict, confused_idx: set[int]) -> np.ndarray:
        """64³ 변화량 LUT. 각 격자 칸은 OKLab에서 '가장 가까운 팔레트 색'의 이동만 따른다.
        움직이지 않는 팔레트 색(배경·짝 색·무관한 선)은 이동 0인 닻(anchor)이 되어
        근처 칸이 끌려가지 않게 막는다. 이동한 색이 들어 있는 칸은 정확히 그 이동량을 갖는다."""
        g = grid_oklab()
        shifts = np.zeros((len(colors), 3))
        if self.mode in ("student", "teacher"):
            for i, new in moved.items():
                shifts[i] = new - colors[i]
        elif self.mode == "base_gated":
            flat = self.base_delta.reshape(-1, 3)
            for i in confused_idx:
                shifts[i] = flat[self._cell(colors[i])]
        if not np.any(shifts):
            return np.zeros((GRID, GRID, GRID, 3), np.float32)
        src = c.srgb_to_oklab(colors)
        dist = np.sqrt(((g[:, None, :] - src[None]) ** 2).sum(-1))
        near = dist.argmin(1)
        dn = dist[np.arange(len(g)), near]
        w = np.exp(-(dn**2) / (2 * self.s.lut_sigma**2))
        w[dn > 3 * self.s.lut_sigma] = 0.0
        delta = w[:, None] * shifts[near]
        for i in np.flatnonzero(np.any(shifts != 0, axis=1)):
            delta[self._cell(colors[i])] = shifts[i]
        return delta.reshape(GRID, GRID, GRID, 3).astype(np.float32)

    @staticmethod
    def _cell(col) -> int:
        col = np.asarray(col, int)
        return int(col[0] // STEP) * GRID * GRID + int(col[1] // STEP) * GRID + int(col[2] // STEP)

    def analyze(self, frame: np.ndarray) -> Analysis:
        t = {}
        t0 = time.perf_counter()
        colors, counts, label = self.palette(frame)
        t["palette_ms"] = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        widths = self.widths(label, len(colors))
        t["width_ms"] = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        pairs = self.judge(colors, widths)
        t["judge_ms"] = (time.perf_counter() - t2) * 1000
        t3 = time.perf_counter()
        moved = self.correct(colors, pairs) if self.mode in ("student", "teacher") else {}
        t["correct_ms"] = (time.perf_counter() - t3) * 1000
        t4 = time.perf_counter()
        confused_idx = {q["i"] for q in pairs if q["confused"]} | {q["j"] for q in pairs if q["confused"]}
        if self.mode != "base_full" and self.mode != "none":
            self.target_delta = self.build_delta(colors, moved, confused_idx)
        t["lut_ms"] = (time.perf_counter() - t4) * 1000
        return Analysis(colors, counts, widths, pairs, moved, t)

    # ---------- 매 프레임 ----------
    def changed(self, frame: np.ndarray) -> bool:
        thumb = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        ch = self.last_thumb is None or float(np.abs(thumb - self.last_thumb).mean()) > self.s.change_thresh
        if ch:
            self.last_thumb = thumb
        return ch

    def _refresh_tables(self) -> None:
        flat = np.rint(self.delta).astype(np.int16).reshape(-1, 3)
        self._flat_delta = flat
        self._moving_cell = np.any(flat != 0, axis=1)
        self._delta_version = self._version

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """바뀌는 격자 칸에 속한 픽셀만 고친다. 나머지 픽셀은 원본 그대로(선택적 보정)."""
        if getattr(self, "_delta_version", None) != self._version:
            self._refresh_tables()
        q = (frame >> 2).astype(np.int32)
        cell = (q[..., 0] << 12) | (q[..., 1] << 6) | q[..., 2]
        hit = self._moving_cell[cell]
        if not hit.any():
            return frame
        out = frame.copy()
        px = out[hit].astype(np.int16) + self._flat_delta[cell[hit]]
        out[hit] = np.clip(px, 0, 255).astype(np.uint8)
        return out

    def process(self, frame: np.ndarray, instant: bool = False) -> tuple[np.ndarray, dict]:
        """frame: (H,W,3) uint8 RGB. instant=True면 EMA 없이 바로 새 LUT(평가용)."""
        timings = {}
        t0 = time.perf_counter()
        if self.mode != "none" and self.changed(frame):
            self.analysis = self.analyze(frame)
            timings.update(self.analysis.timings)
            timings["analyzed"] = 1
        if self.mode not in ("none", "base_full"):
            if not np.array_equal(self.delta, self.target_delta):
                a = 1.0 if instant else self.s.ema
                new = (1 - a) * self.delta + a * self.target_delta
                if np.abs(new - self.target_delta).max() < 0.5:
                    new = self.target_delta.copy()
                self.delta = new
                self._version += 1
        t1 = time.perf_counter()
        out = frame if self.mode == "none" else self.apply(frame)
        timings["apply_ms"] = (time.perf_counter() - t1) * 1000
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return out, timings
