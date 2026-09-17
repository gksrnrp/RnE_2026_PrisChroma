"""검증된 선 자극을 1:1 픽셀로 제시하고 4AFC 응답을 내구성 있게 기록한다.

기본 실행은 검증만 하는 dry-run이다. 실제 전체화면 자료 수집은 명시적인
``--run-gui --confirm-consent`` 조합에서만 허용한다. calibration, 별도 practice
manifest, held-out 응답 전에 동결한 validation plan과 validation manifest를
서로 다른 계약으로 검증한다.

``render_time_ms``와 onset 시각은 Tk가 화면 갱신 요청을 처리한 직후의
소프트웨어 시각이다. 실제 LCD 광자 onset을 뜻하지 않으며, 그 정밀도가 필요한
연구에서는 광센서 같은 별도 장비가 필요하다.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker

from calibration_design import (
    CALIBRATION_TOTAL_TRIALS,
    block_count_from_config,
    block_trial_count,
)
from experiment_runner import (
    CollectionBlockedError,
    JournalError,
    TrialJournal,
    assert_schedule_config,
    canonical_journal_path,
    expected_collection_session_id,
)
from stimulus_renderer import (
    RENDERER_ID,
    RendererValidationError,
    StimulusCompilationError,
    renderer_implementation_sha256,
    validate_calibration_manifest_structure,
    validate_practice_manifest_structure,
    verify_renderer_environment,
    verify_stimulus_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRIAL_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "human" / "trial.schema.json"
CONFIG_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "human" / "experiment_config.schema.json"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PSEUDONYM_RE = re.compile(r"^P[0-9A-Za-z_-]+$")
_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResponseGUIBlockedError(f"JSON 객체에 중복 key가 있습니다: {key}")
        result[key] = value
    return result


class ResponseGUIBlockedError(ValueError):
    """자료 수집의 어느 안전 게이트든 통과하지 못했을 때 발생한다."""


@dataclass(frozen=True)
class ScreenMetrics:
    width_px: int
    height_px: int
    os_scale_percent: float | None
    dpi_awareness_enabled: bool
    source: str
    operating_system: str
    native_gui_runtime: str


@dataclass(frozen=True)
class GitState:
    commit: str
    tracked_worktree_clean: bool
    required_runtime_files_tracked: bool


@dataclass(frozen=True)
class VerifiedTrial:
    row: dict[str, Any]
    png_path: Path
    metadata_path: Path
    metadata_repository_path: str
    png_sha256: str
    decoded_rgb_sha256: str


@dataclass(frozen=True)
class CollectionPlan:
    config: dict[str, Any]
    config_sha256: str
    manifest_sha256: str
    manifest_path: Path
    repo_root: Path
    participant_id: str
    phase: str
    session_block: int | None
    validation_block_id: str | None
    validation_plan_sha256: str | None
    trials: tuple[VerifiedTrial, ...]
    screen_metrics: ScreenMetrics
    git_state: GitState


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(raw)


def _read_json_object(path: str | Path, label: str) -> tuple[dict[str, Any], str]:
    file_path = Path(path)
    try:
        raw = file_path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_object_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResponseGUIBlockedError(
            f"{label} JSON을 읽을 수 없습니다: {file_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ResponseGUIBlockedError(f"{label} 최상위는 JSON 객체여야 합니다.")
    return payload, _sha256_bytes(raw)


def _validate_schema(payload: Mapping[str, Any], schema_path: Path, label: str) -> None:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResponseGUIBlockedError(f"{label} 스키마를 읽지 못했습니다: {error}") from error
    if errors:
        first = errors[0]
        location = ".".join(map(str, first.absolute_path)) or "$"
        raise ResponseGUIBlockedError(
            f"{label} 스키마 위반({location}): {first.message}"
        )


def _set_windows_dpi_awareness() -> None:
    """Tk 생성 전에 Windows Per-Monitor V2 DPI awareness를 확증한다."""

    if os.name != "nt":
        return
    user32 = ctypes.windll.user32
    try:
        setter = user32.SetProcessDpiAwarenessContext
        setter.argtypes = [ctypes.c_void_p]
        setter.restype = ctypes.c_bool
        # 이미 다른 코드가 awareness를 고정한 프로세스에서는 FALSE가 반환될 수
        # 있으므로 반환값만 믿지 않고 현재 thread context를 아래에서 확인한다.
        setter(_PER_MONITOR_AWARE_V2)
        get_context = user32.GetThreadDpiAwarenessContext
        get_context.restype = ctypes.c_void_p
        contexts_equal = user32.AreDpiAwarenessContextsEqual
        contexts_equal.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        contexts_equal.restype = ctypes.c_bool
        if not bool(contexts_equal(get_context(), _PER_MONITOR_AWARE_V2)):
            raise ResponseGUIBlockedError(
                "Windows Per-Monitor V2 DPI awareness를 확증하지 못했습니다."
            )
    except ResponseGUIBlockedError:
        raise
    except (AttributeError, OSError, ValueError) as error:
        raise ResponseGUIBlockedError(
            f"Windows Per-Monitor V2 DPI awareness를 설정하지 못했습니다: {error}"
        ) from error


def inspect_screen_metrics() -> ScreenMetrics:
    """기본 모니터의 물리 픽셀·DPI 배율과 프로세스 DPI 상태를 읽는다."""

    if os.name != "nt":
        raise ResponseGUIBlockedError(
            "이 구현은 Windows에서만 물리 픽셀/DPI 배율을 안전하게 검증합니다."
        )
    _set_windows_dpi_awareness()
    user32 = ctypes.windll.user32
    try:
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        dpi = float(user32.GetDpiForSystem())
    except (AttributeError, OSError, ValueError) as error:
        raise ResponseGUIBlockedError(
            f"Windows 화면 물리 픽셀/DPI 상태를 읽지 못했습니다: {error}"
        ) from error
    if width <= 0 or height <= 0 or dpi <= 0:
        raise ResponseGUIBlockedError("Windows가 유효한 화면 크기/DPI를 반환하지 않았습니다.")
    try:
        import tkinter as tk
    except ImportError as error:
        raise ResponseGUIBlockedError(f"Tk runtime을 불러오지 못했습니다: {error}") from error
    try:
        tk_patchlevel = str(tk.Tcl().call("info", "patchlevel"))
    except (RuntimeError, tk.TclError) as error:
        raise ResponseGUIBlockedError(f"Tk runtime 버전을 확인하지 못했습니다: {error}") from error
    return ScreenMetrics(
        width_px=width,
        height_px=height,
        os_scale_percent=dpi / 96.0 * 100.0,
        dpi_awareness_enabled=True,
        source="Windows Per-Monitor-V2/GetSystemMetrics/GetDpiForSystem",
        operating_system=platform.platform(),
        native_gui_runtime=f"CPython {platform.python_version()}; Tk {tk_patchlevel}",
    )


def validate_screen_metrics(
    config: Mapping[str, Any], metrics: ScreenMetrics
) -> None:
    display = config.get("display")
    if not isinstance(display, Mapping):
        raise ResponseGUIBlockedError("config.display가 없습니다.")
    if metrics.operating_system != display.get("operating_system"):
        raise ResponseGUIBlockedError(
            "운영체제 식별자가 동결 config와 다릅니다: "
            f"관측={metrics.operating_system!r}, 동결={display.get('operating_system')!r}"
        )
    if metrics.native_gui_runtime != display.get("native_gui_runtime"):
        raise ResponseGUIBlockedError(
            "Python/Tk runtime이 동결 config와 다릅니다: "
            f"관측={metrics.native_gui_runtime!r}, 동결={display.get('native_gui_runtime')!r}"
        )
    expected_width = display.get("resolution_width_px")
    expected_height = display.get("resolution_height_px")
    if (metrics.width_px, metrics.height_px) != (expected_width, expected_height):
        raise ResponseGUIBlockedError(
            "화면 해상도가 동결 config와 다릅니다: "
            f"관측={metrics.width_px}x{metrics.height_px}, "
            f"동결={expected_width}x{expected_height}"
        )
    app_scale = display.get("app_scale_percent")
    if not isinstance(app_scale, (int, float)) or isinstance(app_scale, bool):
        raise ResponseGUIBlockedError("display.app_scale_percent가 수치가 아닙니다.")
    if not math.isclose(float(app_scale), 100.0, abs_tol=1e-9):
        raise ResponseGUIBlockedError("1:1 픽셀 표시는 app_scale_percent=100만 허용합니다.")
    if not metrics.dpi_awareness_enabled:
        raise ResponseGUIBlockedError("프로세스 DPI awareness를 확인하지 못했습니다.")
    expected_scale = display.get("os_scale_percent")
    if metrics.os_scale_percent is None:
        raise ResponseGUIBlockedError("운영체제 화면 배율을 확인하지 못했습니다.")
    if not isinstance(expected_scale, (int, float)) or isinstance(expected_scale, bool):
        raise ResponseGUIBlockedError("display.os_scale_percent가 수치가 아닙니다.")
    tolerance = max(0.75, abs(float(expected_scale)) * 0.005)
    if not math.isclose(
        metrics.os_scale_percent, float(expected_scale), abs_tol=tolerance
    ):
        raise ResponseGUIBlockedError(
            "운영체제 화면 배율이 동결 config와 다릅니다: "
            f"관측={metrics.os_scale_percent:.2f}%, 동결={float(expected_scale):.2f}%"
        )


def _validate_tk_window_dpi(root: Any, expected_scale_percent: float) -> None:
    """실제 전체화면 Tk HWND가 동결 배율의 물리 DPI를 쓰는지 확인한다."""

    if os.name != "nt":
        raise ResponseGUIBlockedError("Tk 창 DPI 검증은 Windows에서만 지원합니다.")
    try:
        root.update_idletasks()
        hwnd = int(root.winfo_id())
        get_dpi = ctypes.windll.user32.GetDpiForWindow
        get_dpi.argtypes = [ctypes.c_void_p]
        get_dpi.restype = ctypes.c_uint
        observed_dpi = int(get_dpi(ctypes.c_void_p(hwnd)))
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise ResponseGUIBlockedError(
            f"Tk 창의 물리 DPI를 읽지 못했습니다: {error}"
        ) from error
    expected_dpi = int(round(96.0 * float(expected_scale_percent) / 100.0))
    if observed_dpi <= 0 or observed_dpi != expected_dpi:
        raise ResponseGUIBlockedError(
            "Tk 창 DPI가 동결 Windows 배율과 다릅니다: "
            f"관측={observed_dpi}, 동결={expected_dpi}"
        )


def inspect_git_state(repo_root: str | Path) -> GitState:
    root = Path(repo_root)

    def run(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(args), cwd=root, capture_output=True, text=True, timeout=8
        )

    try:
        revision = run(["git", "rev-parse", "HEAD"])
        diff = run(["git", "diff", "--quiet", "HEAD", "--"])
        tracked = run(
            [
                "git",
                "ls-files",
                "--error-unmatch",
                "src/response_gui.py",
                "src/stimulus_renderer.py",
                "src/experiment_runner.py",
                "schemas/human/trial.schema.json",
                "schemas/human/experiment_config.schema.json",
            ]
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ResponseGUIBlockedError(f"Git 실행 상태를 확인하지 못했습니다: {error}") from error
    commit = revision.stdout.strip().lower()
    if revision.returncode != 0 or not _COMMIT_RE.fullmatch(commit):
        raise ResponseGUIBlockedError("현재 Git commit을 확정하지 못했습니다.")
    return GitState(
        commit=commit,
        tracked_worktree_clean=diff.returncode == 0,
        required_runtime_files_tracked=tracked.returncode == 0,
    )


def _safe_artifact_path(base: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ResponseGUIBlockedError(f"{label}는 manifest 기준 상대 경로여야 합니다.")
    target = (base / relative).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError as error:
        raise ResponseGUIBlockedError(f"{label}가 manifest 폴더 밖을 가리킵니다.") from error
    return target


def _repository_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise ResponseGUIBlockedError(
            "PNG sidecar는 재현 가능한 provenance를 위해 저장소 안에 있어야 합니다."
        ) from error


def _validate_manifest_header(
    manifest: Mapping[str, Any],
    *,
    phase: str,
    config: Mapping[str, Any],
    config_sha256: str,
    git_state: GitState,
) -> None:
    if manifest.get("schema_version") != "1.0":
        raise ResponseGUIBlockedError("compiled manifest schema_version은 1.0이어야 합니다.")
    expected_artifact = {
        "calibration": "compiled_pixel_stimulus_manifest",
        "practice": "compiled_practice_pixel_stimulus_manifest",
        "validation": "compiled_validation_pixel_stimulus_manifest",
    }.get(phase)
    if expected_artifact is None or manifest.get("artifact_type") != expected_artifact:
        raise ResponseGUIBlockedError(
            f"{phase} 전용 compiled pixel stimulus manifest가 아닙니다."
        )
    if phase in {"practice", "validation"} and manifest.get("phase") != phase:
        raise ResponseGUIBlockedError(f"manifest.phase가 {phase}가 아닙니다.")
    if manifest.get("png_files_written") is not True:
        raise ResponseGUIBlockedError("manifest에 실제 PNG가 컴파일되지 않았습니다.")
    if manifest.get("config_canonical_sha256") != _canonical_sha256(config):
        raise ResponseGUIBlockedError("manifest의 canonical config hash가 현재 config와 다릅니다.")
    if manifest.get("schedule_config_sha256") != config_sha256:
        raise ResponseGUIBlockedError("manifest의 raw config hash가 현재 config 파일과 다릅니다.")
    if phase in {"practice", "validation"} and manifest.get(
        "source_config_raw_sha256"
    ) != config_sha256:
        raise ResponseGUIBlockedError("manifest의 source config raw hash가 현재 config와 다릅니다.")
    if manifest.get("git_commit") != git_state.commit:
        raise ResponseGUIBlockedError("manifest Git commit과 현재 실행 commit이 다릅니다.")
    if not git_state.tracked_worktree_clean:
        raise ResponseGUIBlockedError("추적 중인 작업 파일이 수정되어 Git commit과 일치하지 않습니다.")
    if not git_state.required_runtime_files_tracked:
        raise ResponseGUIBlockedError("GUI·렌더러·스키마가 현재 Git commit에 포함되지 않았습니다.")

    observer = config["observer_profile"]
    if manifest.get("participant_id") != observer["participant_id"]:
        raise ResponseGUIBlockedError("manifest 참가자 코드가 config와 다릅니다.")
    if manifest.get("stimulus_family_id") != config["color_design"]["stimulus_family_id"]:
        raise ResponseGUIBlockedError("manifest 자극군 ID가 config와 다릅니다.")
    if manifest.get("cvd_type") != observer["cvd_type"]:
        raise ResponseGUIBlockedError("manifest CVD 유형이 config와 다릅니다.")
    if float(manifest.get("machado_severity", -1)) != float(observer["severity"]):
        raise ResponseGUIBlockedError("manifest Machado severity가 config와 다릅니다.")

    current = verify_renderer_environment(config)
    renderer = manifest.get("renderer")
    if not isinstance(renderer, Mapping):
        raise ResponseGUIBlockedError("manifest.renderer가 없습니다.")
    for field in (
        "renderer_id",
        "implementation_version",
        "implementation_sha256",
        "pillow_version",
    ):
        if renderer.get(field) != current.get(field):
            raise ResponseGUIBlockedError(
                f"manifest.renderer.{field}가 현재 검증된 렌더러와 다릅니다."
            )
    if renderer.get("implementation_sha256") != renderer_implementation_sha256():
        raise ResponseGUIBlockedError("manifest 렌더러 구현 해시가 현재 파일과 다릅니다.")


def _manifest_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("trials")
    if not isinstance(rows, list) or not rows:
        raise ResponseGUIBlockedError("manifest.trials가 비었습니다.")
    if manifest.get("trial_count") != len(rows):
        raise ResponseGUIBlockedError("manifest.trial_count가 trials 길이와 다릅니다.")
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ResponseGUIBlockedError(f"manifest.trials[{index}]가 객체가 아닙니다.")
        trial_id = row.get("trial_id")
        if isinstance(trial_id, bool) or not isinstance(trial_id, int) or trial_id < 1:
            raise ResponseGUIBlockedError(f"manifest.trials[{index}].trial_id가 잘못됐습니다.")
        if trial_id in seen:
            raise ResponseGUIBlockedError(f"manifest에서 trial_id {trial_id}가 중복됩니다.")
        seen.add(trial_id)
        result.append(dict(row))
    return result


def _select_rows(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    phase: str,
    session_block: int | None,
    validation_block_id: str | None,
    configured_practice_trials: int,
) -> list[dict[str, Any]]:
    rows = _manifest_rows(manifest)
    if phase == "calibration":
        if validation_block_id is not None:
            raise ResponseGUIBlockedError("calibration에는 validation block을 지정하지 않습니다.")
        block_count = block_count_from_config(config)
        if session_block not in set(range(1, block_count + 1)):
            raise ResponseGUIBlockedError(
                f"calibration은 --session-block 1~{block_count}가 필요합니다."
            )
        if len(rows) != CALIBRATION_TOTAL_TRIALS:
            raise ResponseGUIBlockedError("calibration manifest는 전체 384시행이어야 합니다.")
        generation = manifest.get("color_generation")
        if isinstance(generation, Mapping) and generation.get("development_partial_schedule") is True:
            raise ResponseGUIBlockedError("부분 개발 manifest로 사람 calibration을 실행할 수 없습니다.")
        if any(row.get("phase", "calibration") != "calibration" for row in rows):
            raise ResponseGUIBlockedError("calibration manifest에 다른 phase 시행이 섞였습니다.")
        selected = [row for row in rows if row.get("session_block") == session_block]
        expected = block_trial_count(config)
        if len(selected) != expected:
            raise ResponseGUIBlockedError(
                f"선택한 calibration session block은 정확히 {expected}시행이어야 합니다."
            )
        return selected
    if phase == "practice":
        if session_block is not None or validation_block_id is not None:
            raise ResponseGUIBlockedError("practice에는 session/validation block을 지정하지 않습니다.")
        if not 6 <= configured_practice_trials <= 10:
            raise ResponseGUIBlockedError("동결 practice_trials는 6~10 범위여야 합니다.")
        if len(rows) != configured_practice_trials:
            raise ResponseGUIBlockedError(
                "practice manifest 시행 수가 config.sampling.practice_trials와 다릅니다."
            )
        if any(row.get("phase") != "practice" for row in rows):
            raise ResponseGUIBlockedError("practice manifest의 모든 시행에 phase=practice가 필요합니다.")
        return rows
    if phase == "validation":
        if session_block is not None:
            raise ResponseGUIBlockedError("validation에는 calibration session block을 지정하지 않습니다.")
        if validation_block_id not in {"V01", "V02", "V03"}:
            raise ResponseGUIBlockedError("validation은 --validation-block V01~V03가 필요합니다.")
        if len(rows) != 192 or any(row.get("phase") != "validation" for row in rows):
            raise ResponseGUIBlockedError("validation manifest는 validation 192시행이어야 합니다.")
        if {row.get("trial_id") for row in rows} != set(range(385, 577)):
            raise ResponseGUIBlockedError(
                "validation manifest는 예약 trial_id 385~576을 정확히 사용해야 합니다."
            )
        selected = [
            row for row in rows if row.get("validation_block_id") == validation_block_id
        ]
        if len(selected) != 64:
            raise ResponseGUIBlockedError("선택한 validation block은 정확히 64시행이어야 합니다.")
        combinations = {
            (
                row.get("validation_level_index"),
                row.get("correct_position"),
                row.get("odd_color"),
            )
            for row in selected
        }
        expected = {
            (level, position, odd)
            for level in range(1, 9)
            for position in range(1, 5)
            for odd in ("a", "b")
        }
        if combinations != expected:
            raise ResponseGUIBlockedError(
                "validation block이 8수준×4위치×2 odd 완전 균형이 아닙니다."
            )
        within = {row.get("within_block_index") for row in selected}
        if within != set(range(1, 65)):
            raise ResponseGUIBlockedError("validation within_block_index가 1~64 순열이 아닙니다.")
        return selected
    raise ResponseGUIBlockedError(f"지원하지 않는 phase입니다: {phase}")


def _verify_selected_artifacts(
    rows: Sequence[dict[str, Any]],
    *,
    manifest_path: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    git_state: GitState,
    repo_root: Path,
    phase: str,
) -> tuple[VerifiedTrial, ...]:
    verified: list[VerifiedTrial] = []
    base = manifest_path.resolve().parent
    for row in rows:
        trial_id = row["trial_id"]
        for field, expected in (
            ("config_sha256", config_sha256),
            ("git_commit", git_state.commit),
            ("participant_id", config["observer_profile"]["participant_id"]),
            ("stimulus_family_id", config["color_design"]["stimulus_family_id"]),
        ):
            if row.get(field) != expected:
                raise ResponseGUIBlockedError(
                    f"trial {trial_id}의 {field}가 동결 입력과 다릅니다."
                )
        artifact = row.get("raster_artifact")
        if not isinstance(artifact, Mapping):
            raise ResponseGUIBlockedError(f"trial {trial_id}의 raster artifact가 없습니다.")
        png = _safe_artifact_path(base, artifact.get("png"), f"trial {trial_id} PNG")
        metadata = _safe_artifact_path(
            base, artifact.get("pixel_metadata"), f"trial {trial_id} sidecar"
        )
        try:
            check = verify_stimulus_artifact(
                png, metadata, config=config, trial=row
            )
        except (RendererValidationError, OSError) as error:
            raise ResponseGUIBlockedError(
                f"trial {trial_id} 픽셀 재검증 실패: {error}"
            ) from error
        for artifact_field, result_field in (
            ("png_sha256", "png_sha256"),
            ("decoded_rgb_sha256", "decoded_rgb_sha256"),
        ):
            expected = artifact.get(artifact_field)
            if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
                raise ResponseGUIBlockedError(
                    f"trial {trial_id} raster_artifact.{artifact_field} 형식이 잘못됐습니다."
                )
            if check[result_field] != expected:
                raise ResponseGUIBlockedError(
                    f"trial {trial_id} raster_artifact.{artifact_field}가 sidecar/PNG와 다릅니다."
                )
        metadata_sha256 = artifact.get("pixel_metadata_sha256")
        if not isinstance(metadata_sha256, str) or not _SHA256_RE.fullmatch(
            metadata_sha256
        ):
            raise ResponseGUIBlockedError(
                f"trial {trial_id}의 pixel_metadata_sha256이 없습니다."
            )
        if _sha256_bytes(metadata.read_bytes()) != metadata_sha256:
            raise ResponseGUIBlockedError(
                f"trial {trial_id}의 sidecar raw SHA-256이 manifest와 다릅니다."
            )
        verified.append(
            VerifiedTrial(
                row=dict(row),
                png_path=png,
                metadata_path=metadata,
                metadata_repository_path=_repository_relative(metadata, repo_root),
                png_sha256=check["png_sha256"],
                decoded_rgb_sha256=check["decoded_rgb_sha256"],
            )
        )
    return tuple(verified)


def _validate_validation_sources(
    manifest: Mapping[str, Any],
    *,
    validation_plan_path: str | Path | None,
    calibration_model_fits_path: str | Path | None,
    config_sha256: str,
) -> tuple[dict[str, Any], str]:
    """validation manifest를 held-out 이전 동결 plan과 calibration fit에 결박한다."""

    if validation_plan_path is None or calibration_model_fits_path is None:
        raise ResponseGUIBlockedError(
            "validation에는 --validation-plan과 --calibration-model-fits가 모두 필요합니다."
        )
    plan, plan_raw_sha256 = _read_json_object(validation_plan_path, "validation plan")
    calibration, calibration_raw_sha256 = _read_json_object(
        calibration_model_fits_path, "calibration model fits"
    )
    try:
        from analyze_calibration import validate_calibration_artifact
        from validation_design import compute_plan_sha256, validate_plan

        validate_calibration_artifact(calibration)
        validate_plan(plan)
        canonical_plan_sha256 = compute_plan_sha256(plan)
    except (TypeError, ValueError) as error:
        raise ResponseGUIBlockedError(
            f"validation plan/calibration fit 검증 실패: {error}"
        ) from error
    plan_sha256 = plan.get("plan_sha256")
    if not isinstance(plan_sha256, str) or not _SHA256_RE.fullmatch(plan_sha256):
        raise ResponseGUIBlockedError("validation plan.plan_sha256 형식이 잘못됐습니다.")
    for actual, expected, label in (
        (canonical_plan_sha256, plan_sha256, "validation plan canonical hash"),
        (manifest.get("validation_plan_sha256"), plan_sha256, "manifest canonical plan hash"),
        (
            manifest.get("validation_plan_raw_file_sha256"),
            plan_raw_sha256,
            "manifest raw plan file hash",
        ),
        (
            manifest.get("calibration_model_fits_sha256"),
            calibration_raw_sha256,
            "manifest calibration fit raw hash",
        ),
        (
            plan.get("calibration_model_fits_sha256"),
            calibration_raw_sha256,
            "plan calibration fit raw hash",
        ),
        (plan.get("config_sha256"), config_sha256, "plan config raw hash"),
    ):
        if actual != expected:
            raise ResponseGUIBlockedError(f"{label}가 외부 동결 파일과 다릅니다.")
    selected_manifest = manifest.get("selected_width")
    selected_plan = plan.get("selected_width")
    if not isinstance(selected_manifest, Mapping) or not isinstance(selected_plan, Mapping):
        raise ResponseGUIBlockedError("validation selected_width provenance가 없습니다.")
    for field in ("stroke_width_px", "stroke_width_deg"):
        if selected_manifest.get(field) != selected_plan.get(field):
            raise ResponseGUIBlockedError(
                f"manifest selected_width.{field}가 validation plan과 다릅니다."
            )
    return plan, plan_sha256


def prepare_collection(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    phase: str,
    session_block: int | None,
    validation_block_id: str | None = None,
    screen_metrics: ScreenMetrics | None = None,
    git_state: GitState | None = None,
    repo_root: str | Path = PROJECT_ROOT,
    calibration_manifest_path: str | Path | None = None,
    validation_plan_path: str | Path | None = None,
    calibration_model_fits_path: str | Path | None = None,
) -> CollectionPlan:
    """모든 정적·런타임 게이트와 선택 시행의 픽셀을 검증한다."""

    root = Path(repo_root).resolve()
    config, config_sha256 = _read_json_object(config_path, "experiment config")
    _validate_schema(config, CONFIG_SCHEMA_PATH, "experiment config")
    try:
        assert_schedule_config(config)
    except CollectionBlockedError as error:
        raise ResponseGUIBlockedError(str(error)) from error
    participant = config["observer_profile"]["participant_id"]
    if not isinstance(participant, str) or not _PSEUDONYM_RE.fullmatch(participant):
        raise ResponseGUIBlockedError("participant_id는 직접 식별자가 아닌 P 접두 가명 코드여야 합니다.")
    if config["privacy"].get("approved") is not True:
        raise ResponseGUIBlockedError("연구/개인정보 승인이 확인되지 않았습니다.")

    observed = screen_metrics or inspect_screen_metrics()
    validate_screen_metrics(config, observed)
    current_git = git_state or inspect_git_state(root)

    manifest_file = Path(manifest_path).resolve()
    manifest, manifest_sha256 = _read_json_object(manifest_file, "compiled manifest")
    _validate_manifest_header(
        manifest,
        phase=phase,
        config=config,
        config_sha256=config_sha256,
        git_state=current_git,
    )
    if phase == "calibration":
        try:
            validate_calibration_manifest_structure(
                manifest, config=config, config_sha256=config_sha256
            )
        except (StimulusCompilationError, RendererValidationError) as error:
            raise ResponseGUIBlockedError(
                f"calibration manifest 전체 구조 검증 실패: {error}"
            ) from error
    rows = _select_rows(
        manifest,
        config=config,
        phase=phase,
        session_block=session_block,
        validation_block_id=validation_block_id,
        configured_practice_trials=int(config["sampling"]["practice_trials"]),
    )

    validation_plan: dict[str, Any] | None = None
    validation_plan_sha256: str | None = None
    if phase == "validation":
        validation_plan, validation_plan_sha256 = _validate_validation_sources(
            manifest,
            validation_plan_path=validation_plan_path,
            calibration_model_fits_path=calibration_model_fits_path,
            config_sha256=config_sha256,
        )
        plan_schedule = {
            row["trial_id"]: row for row in validation_plan.get("schedule", [])
        }
        compared_fields = (
            "validation_plan_sha256",
            "validation_block_id",
            "validation_level_index",
            "within_block_index",
            "global_trial_key",
            "stroke_width_px",
            "stroke_width_deg",
            "target_delta_e00_simulated",
            "actual_delta_e00_simulated",
            "correct_position",
            "odd_color",
            "position_colors",
            "random_seed",
            "participant_id",
            "config_sha256",
        )
        for row in rows:
            source = plan_schedule.get(row["trial_id"])
            if not isinstance(source, Mapping):
                raise ResponseGUIBlockedError(
                    f"validation trial {row['trial_id']}가 동결 plan schedule에 없습니다."
                )
            if any(row.get(field) != source.get(field) for field in compared_fields):
                raise ResponseGUIBlockedError(
                    f"validation trial {row['trial_id']}가 동결 plan schedule과 다릅니다."
                )

    if phase == "practice":
        if calibration_manifest_path is None:
            raise ResponseGUIBlockedError(
                "practice 실행에는 전역 trial_id 비중복 확인용 calibration manifest가 필요합니다."
            )
        calibration, calibration_manifest_sha256 = _read_json_object(
            calibration_manifest_path, "calibration manifest"
        )
        _validate_manifest_header(
            calibration,
            phase="calibration",
            config=config,
            config_sha256=config_sha256,
            git_state=current_git,
        )
        calibration_rows = _select_rows(
            calibration,
            config=config,
            phase="calibration",
            session_block=1,
            validation_block_id=None,
            configured_practice_trials=int(config["sampling"]["practice_trials"]),
        )
        # session 1만 선택했지만 전체 manifest의 ID 집합으로 교차 충돌을 본다.
        del calibration_rows
        calibration_ids = {row["trial_id"] for row in _manifest_rows(calibration)}
        practice_ids = {row["trial_id"] for row in rows}
        overlap = sorted(calibration_ids & practice_ids)
        if overlap:
            raise ResponseGUIBlockedError(
                f"practice와 calibration의 전역 trial_id가 겹칩니다: {overlap[:5]}"
            )
        try:
            validate_practice_manifest_structure(
                manifest,
                calibration_manifest=calibration,
                calibration_manifest_sha256=calibration_manifest_sha256,
                config=config,
                config_sha256=config_sha256,
            )
        except (StimulusCompilationError, RendererValidationError) as error:
            raise ResponseGUIBlockedError(
                f"practice/calibration manifest 전체 구조 검증 실패: {error}"
            ) from error

    trials = _verify_selected_artifacts(
        rows,
        manifest_path=manifest_file,
        config=config,
        config_sha256=config_sha256,
        git_state=current_git,
        repo_root=root,
        phase=phase,
    )
    return CollectionPlan(
        config=config,
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha256,
        manifest_path=manifest_file,
        repo_root=root,
        participant_id=participant,
        phase=phase,
        session_block=session_block,
        validation_block_id=validation_block_id,
        validation_plan_sha256=validation_plan_sha256,
        trials=trials,
        screen_metrics=observed,
        git_state=current_git,
    )


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _assert_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not _SESSION_RE.fullmatch(session_id):
        raise ResponseGUIBlockedError(
            "session_id는 개인 정보가 아닌 영문·숫자·밑줄·하이픈 코드여야 합니다."
        )


def build_trial_record(
    verified: VerifiedTrial,
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    stimulus_manifest_sha256: str,
    session_id: str,
    phase: str,
    validation_plan_sha256: str | None = None,
    onset_utc: str,
    onset_monotonic_ns: int,
    render_time_ms: float,
    response_position: int | None = None,
    response_time_ms: float | None = None,
    abort_reason: str | None = None,
) -> dict[str, Any]:
    """GUI 상태에서 스키마 완전한 completed/aborted 시행 레코드를 만든다."""

    _assert_session_id(session_id)
    row = verified.row
    if row.get("config_sha256") != config_sha256:
        raise ResponseGUIBlockedError("시행의 config hash가 실행 config와 다릅니다.")
    if (
        not isinstance(stimulus_manifest_sha256, str)
        or not _SHA256_RE.fullmatch(stimulus_manifest_sha256)
    ):
        raise ResponseGUIBlockedError("시행에는 검증된 stimulus manifest SHA-256이 필요합니다.")
    if phase not in {"practice", "calibration", "validation"}:
        raise ResponseGUIBlockedError("지원하지 않는 사람 시행 phase입니다.")
    if phase == "validation":
        if (
            not isinstance(validation_plan_sha256, str)
            or not _SHA256_RE.fullmatch(validation_plan_sha256)
            or row.get("validation_plan_sha256") != validation_plan_sha256
        ):
            raise ResponseGUIBlockedError(
                "validation 시행에는 row와 일치하는 동결 plan SHA-256이 필요합니다."
            )
    elif validation_plan_sha256 is not None:
        raise ResponseGUIBlockedError("practice/calibration에는 validation plan hash를 넣지 않습니다.")
    if abort_reason is None:
        if response_position not in {1, 2, 3, 4}:
            raise ResponseGUIBlockedError("완료 시행에는 1~4 응답이 필요합니다.")
        if response_time_ms is None or response_time_ms < 0:
            raise ResponseGUIBlockedError("완료 시행에는 0 이상의 반응시간이 필요합니다.")
        status = "completed"
        is_correct: bool | None = response_position == row["correct_position"]
    else:
        if not isinstance(abort_reason, str) or not abort_reason.strip():
            raise ResponseGUIBlockedError("중단 사유는 빈 문자열일 수 없습니다.")
        status = "aborted"
        response_position = None
        response_time_ms = None
        is_correct = None
    if (
        isinstance(onset_monotonic_ns, bool)
        or not isinstance(onset_monotonic_ns, int)
        or onset_monotonic_ns < 0
    ):
        raise ResponseGUIBlockedError("onset monotonic 시각이 잘못됐습니다.")
    if not isinstance(render_time_ms, (int, float)) or render_time_ms < 0:
        raise ResponseGUIBlockedError("render_time_ms는 0 이상이어야 합니다.")

    labels = list(row["position_colors"])
    rgb_a = list(row["displayed_rgb_a"])
    rgb_b = list(row["displayed_rgb_b"])
    record = {
        "schema_version": "1.0",
        "data_origin": "human",
        "config_version": row["config_version"],
        "config_sha256": config_sha256,
        "stimulus_manifest_sha256": stimulus_manifest_sha256,
        "validation_plan_sha256": validation_plan_sha256,
        "git_commit": row["git_commit"],
        "stimulus_family_id": row["stimulus_family_id"],
        "renderer_id": RENDERER_ID,
        "stimulus_png_sha256": verified.png_sha256,
        "stimulus_decoded_rgb_sha256": verified.decoded_rgb_sha256,
        "stimulus_pixel_metadata_path": verified.metadata_repository_path,
        "participant_id": row["participant_id"],
        "session_id": session_id,
        "session_block": row["session_block"] if phase == "calibration" else None,
        "validation_block_id": (
            row["validation_block_id"] if phase == "validation" else None
        ),
        "trial_id": row["trial_id"],
        "phase": phase,
        "trial_status": status,
        "abort_reason": abort_reason,
        "timestamp_utc": onset_utc,
        "stimulus_onset_monotonic_ns": onset_monotonic_ns,
        "width_index": row.get("width_index") if phase == "calibration" else None,
        "stroke_width_px": row["stroke_width_px"],
        "stroke_width_deg": row["stroke_width_deg"],
        "delta_e00_level_index": (
            row.get("delta_e00_level_index") if phase == "calibration" else None
        ),
        "validation_level_index": (
            row["validation_level_index"] if phase == "validation" else None
        ),
        "target_delta_e00_simulated": row["target_delta_e00_simulated"],
        "displayed_rgb_a": rgb_a,
        "displayed_rgb_b": rgb_b,
        "displayed_lab_a": list(row["displayed_lab_a"]),
        "displayed_lab_b": list(row["displayed_lab_b"]),
        "simulated_rgb_a": list(row["simulated_rgb_a"]),
        "simulated_rgb_b": list(row["simulated_rgb_b"]),
        "simulated_lab_a": list(row["simulated_lab_a"]),
        "simulated_lab_b": list(row["simulated_lab_b"]),
        "delta_e00_original": row["delta_e00_original"],
        "delta_e00_simulated": row["delta_e00_simulated"],
        "correct_position": row["correct_position"],
        "position_colors": labels,
        "position_rgbs": [rgb_a if label == "a" else rgb_b for label in labels],
        "odd_color": row["odd_color"],
        "response_position": response_position,
        "is_correct": is_correct,
        "response_time_ms": response_time_ms,
        "render_time_ms": float(render_time_ms),
        "cvd_type": row["cvd_type"],
        "machado_severity": row["machado_severity"],
        "severity_source": row["severity_source"],
        "severity_session_id": row["severity_session_id"],
        "fixed_lapse": row["fixed_lapse"],
        "lapse_source": row["lapse_source"],
        "lapse_session_id": row["lapse_session_id"],
        "random_seed": row["random_seed"],
    }
    _validate_schema(record, TRIAL_SCHEMA_PATH, "human trial record")
    return record


def remaining_trials(plan: CollectionPlan, journal: TrialJournal) -> tuple[VerifiedTrial, ...]:
    terminal = journal.terminal_keys
    return tuple(
        trial
        for trial in plan.trials
        if (plan.participant_id, plan.config_sha256, trial.row["trial_id"])
        not in terminal
    )


def session_history_start_utc(
    plan: CollectionPlan,
    journal: TrialJournal,
    session_id: str,
) -> datetime | None:
    """Validate one physical-session identity and return its first recorded onset.

    A process restart may recover an interrupted trial, but it must not reset the
    preregistered session-duration clock or silently splice one block across two
    nominal sessions.  UTC is used only across process restarts; the monotonic
    clock remains authoritative within the current process.
    """

    _assert_session_id(session_id)
    try:
        frozen_session_id = expected_collection_session_id(
            plan.config,
            plan.phase,
            session_block=plan.session_block,
            validation_block_id=plan.validation_block_id,
        )
    except CollectionBlockedError as error:
        raise ResponseGUIBlockedError(str(error)) from error
    if session_id != frozen_session_id:
        raise ResponseGUIBlockedError(
            "session_id가 config에 동결된 현재 block 값과 다릅니다."
        )
    relevant = [
        record
        for record in journal.records
        if record.get("participant_id") == plan.participant_id
        and record.get("config_sha256") == plan.config_sha256
        and record.get("phase") == plan.phase
    ]
    starts: list[datetime] = []
    for record in relevant:
        record_session_block = record.get("session_block")
        record_validation_block = record.get("validation_block_id")
        try:
            record_expected_session = expected_collection_session_id(
                plan.config,
                plan.phase,
                session_block=record_session_block,
                validation_block_id=record_validation_block,
            )
        except CollectionBlockedError as error:
            raise ResponseGUIBlockedError(
                "기존 저널 record의 phase/block 식별자가 동결 계약과 맞지 않습니다."
            ) from error
        if record.get("session_id") != record_expected_session:
            raise ResponseGUIBlockedError(
                "기존 journal에서 session_id가 다른 block에 재사용되었거나 "
                "동결 block 값과 다릅니다."
            )
        current_block = (
            plan.phase == "practice"
            or plan.phase == "calibration"
            and record_session_block == plan.session_block
            or plan.phase == "validation"
            and record_validation_block == plan.validation_block_id
        )
        if not current_block:
            continue
        raw = record.get("timestamp_utc")
        if not isinstance(raw, str) or not raw.endswith("Z"):
            raise ResponseGUIBlockedError(
                "기존 세션 기록의 timestamp_utc가 없어 누적 세션 시간을 검증할 수 없습니다."
            )
        try:
            parsed = datetime.fromisoformat(raw[:-1] + "+00:00")
        except ValueError as error:
            raise ResponseGUIBlockedError(
                "기존 세션 기록의 timestamp_utc 형식이 잘못됐습니다."
            ) from error
        if parsed.tzinfo is None:
            raise ResponseGUIBlockedError(
                "기존 세션 기록의 timestamp_utc에 시간대가 없습니다."
            )
        starts.append(parsed.astimezone(timezone.utc))
    return min(starts) if starts else None


def _assert_journal_location(path: Path, plan: CollectionPlan) -> None:
    try:
        expected = canonical_journal_path(plan.config, plan.phase, plan.repo_root)
    except CollectionBlockedError as error:
        raise ResponseGUIBlockedError(str(error)) from error
    if os.path.normcase(str(path.resolve())) != os.path.normcase(str(expected)):
        raise ResponseGUIBlockedError(
            "--journal은 config에 동결된 현재 phase canonical 경로와 정확히 같아야 합니다: "
            f"{expected}"
        )
    relative = expected.relative_to(plan.repo_root.resolve())
    check = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=plan.repo_root,
        capture_output=True,
        timeout=5,
    )
    if check.returncode != 0:
        raise ResponseGUIBlockedError(
            "사람 시행 저널을 저장소 안에 둘 때는 반드시 .gitignore 대상이어야 합니다."
        )


def _verify_one_again(trial: VerifiedTrial, config: Mapping[str, Any]) -> None:
    result = verify_stimulus_artifact(
        trial.png_path, trial.metadata_path, config=config, trial=trial.row
    )
    if (
        result["png_sha256"] != trial.png_sha256
        or result["decoded_rgb_sha256"] != trial.decoded_rgb_sha256
    ):
        raise ResponseGUIBlockedError("시행 직전 자극 해시가 사전 검증값과 달라졌습니다.")


class ExperimentWindow:
    """Tk 전체화면 상태기계. 생성은 명시적인 실제 실행에서만 한다."""

    def __init__(
        self,
        root: Any,
        *,
        plan: CollectionPlan,
        journal: TrialJournal,
        session_id: str,
    ):
        import tkinter as tk

        self.tk = tk
        self.root = root
        self.plan = plan
        self.journal = journal
        self.session_id = session_id
        self.queue = list(remaining_trials(plan, journal))
        self.index = 0
        self.active: VerifiedTrial | None = None
        self.photo: Any = None
        self.onset_utc: str | None = None
        self.onset_ns: int | None = None
        self.render_time_ms: float | None = None
        self.deadline_after_id: str | None = None
        self.break_after_id: str | None = None
        self.session_start_ns: int | None = None
        self.session_start_utc = session_history_start_utc(
            plan, journal, session_id
        )
        self.completed_this_run = 0
        self.halted_for_session_limit = False
        background = "#%02x%02x%02x" % tuple(plan.config["display"]["background_rgb"])
        self.panel = tk.Label(
            root,
            background=background,
            foreground="white",
            borderwidth=0,
            highlightthickness=0,
            font=("Arial", 28),
        )
        self.panel.pack(fill="both", expand=True)
        root.configure(background=background)
        root.attributes("-fullscreen", True)
        root.bind("<KeyPress-1>", lambda _event: self._respond(1))
        root.bind("<KeyPress-2>", lambda _event: self._respond(2))
        root.bind("<KeyPress-3>", lambda _event: self._respond(3))
        root.bind("<KeyPress-4>", lambda _event: self._respond(4))
        root.bind("<Escape>", self._escape)
        root.bind("<space>", self._start)
        root.protocol("WM_DELETE_WINDOW", self._escape)
        self._show_intro()

    def _show_intro(self) -> None:
        done = len(self.plan.trials) - len(self.queue)
        stop_instruction = (
            "중단: Esc (이 연습 시행은 다시 시작할 수 있음)"
            if self.plan.phase == "practice"
            else "중단: Esc (활성 자극 중 누르면 참가자 자료의 확증 분석이 무효화됨)"
        )
        self.panel.configure(
            image="",
            text=(
                f"{self.plan.phase} 세션\n"
                f"남은 시행 {len(self.queue)}개 (기록 완료 {done}개)\n\n"
                "화면에서 다른 하나의 위치를 1~4로 누르세요.\n"
                f"시작: Space   {stop_instruction}"
            ),
        )

    def _start(self, _event: Any = None) -> None:
        self.root.unbind("<space>")
        if self.session_start_ns is None:
            self.session_start_ns = time.perf_counter_ns()
        if self.session_start_utc is None:
            self.session_start_utc = datetime.now(timezone.utc)
        self.root.update_idletasks()
        try:
            _validate_tk_window_dpi(
                self.root, self.plan.config["display"]["os_scale_percent"]
            )
        except ResponseGUIBlockedError as error:
            self._fatal(str(error))
            return
        observed = (self.root.winfo_width(), self.root.winfo_height())
        expected = (
            self.plan.config["display"]["resolution_width_px"],
            self.plan.config["display"]["resolution_height_px"],
        )
        if observed != expected:
            self._fatal(f"전체화면 크기가 동결 해상도와 다릅니다: {observed} != {expected}")
            return
        if not self.queue:
            self._finish()
            return
        self._fixation()

    def _fixation(self) -> None:
        if self.index >= len(self.queue):
            self._finish()
            return
        if self._session_limit_reached():
            self._show_session_limit()
            return
        self.active = None
        self.photo = None
        self.panel.configure(image="", text="+")
        delay = int(round(float(self.plan.config["timing"]["fixation_ms"])))
        self.root.after(delay, self._present)

    def _present(self) -> None:
        if self.index >= len(self.queue):
            self._finish()
            return
        trial = self.queue[self.index]
        try:
            _verify_one_again(trial, self.plan.config)
            photo = self.tk.PhotoImage(file=str(trial.png_path))
            expected = (
                self.plan.config["display"]["resolution_width_px"],
                self.plan.config["display"]["resolution_height_px"],
            )
            if (photo.width(), photo.height()) != expected:
                raise ResponseGUIBlockedError("PNG 크기가 동결 화면 해상도와 다릅니다.")
            render_start = time.perf_counter_ns()
            self.photo = photo
            self.panel.configure(image=photo, text="")
            self.root.update_idletasks()
            self.render_time_ms = (time.perf_counter_ns() - render_start) / 1_000_000.0
            self.onset_ns = time.perf_counter_ns()
            self.onset_utc = _utc_now_z()
            self.active = trial
            timing = self.plan.config["timing"]
            if timing["response_deadline_policy"] == "fixed":
                self.deadline_after_id = self.root.after(
                    int(round(float(timing["response_deadline_ms"]))), self._timeout
                )
        except (OSError, RendererValidationError, ResponseGUIBlockedError) as error:
            self._fatal(str(error))

    def _cancel_deadline(self) -> None:
        if self.deadline_after_id is not None:
            self.root.after_cancel(self.deadline_after_id)
            self.deadline_after_id = None

    def _respond(self, position: int) -> None:
        if self.active is None or self.onset_ns is None or self.onset_utc is None:
            return
        response_ns = time.perf_counter_ns()
        self._cancel_deadline()
        trial = self.active
        try:
            record = build_trial_record(
                trial,
                config=self.plan.config,
                config_sha256=self.plan.config_sha256,
                stimulus_manifest_sha256=self.plan.manifest_sha256,
                session_id=self.session_id,
                phase=self.plan.phase,
                validation_plan_sha256=self.plan.validation_plan_sha256,
                onset_utc=self.onset_utc,
                onset_monotonic_ns=self.onset_ns,
                render_time_ms=float(self.render_time_ms),
                response_position=position,
                response_time_ms=(response_ns - self.onset_ns) / 1_000_000.0,
            )
            self.journal.append(record)
        except (JournalError, ResponseGUIBlockedError, OSError) as error:
            self._fatal(str(error))
            return
        self.index += 1
        self.completed_this_run += 1
        self.active = None
        feedback = (
            self.plan.phase == "practice"
            and self.plan.config["task"]["practice_feedback"] is True
        ) or (
            self.plan.phase == "calibration"
            and self.plan.config["task"]["main_feedback"] is True
        )
        self.photo = None
        if feedback:
            self.panel.configure(image="", text="정답" if record["is_correct"] else "오답")
        else:
            self.panel.configure(image="", text="")
        delay = int(round(float(self.plan.config["timing"]["inter_trial_ms"])))
        break_interval = int(self.plan.config["timing"]["planned_break_interval_trials"])
        if (
            self.index < len(self.queue)
            and self.completed_this_run % break_interval == 0
        ):
            self.root.after(delay, self._begin_break)
        else:
            self.root.after(delay, self._fixation)

    def _session_limit_reached(self) -> bool:
        if self.session_start_ns is None:
            return False
        maximum_minutes = float(self.plan.config["timing"]["max_session_duration_min"])
        monotonic_minutes = (
            time.perf_counter_ns() - self.session_start_ns
        ) / 60_000_000_000.0
        wall_minutes = 0.0
        if self.session_start_utc is not None:
            wall_minutes = max(
                0.0,
                (datetime.now(timezone.utc) - self.session_start_utc).total_seconds()
                / 60.0,
            )
        return max(monotonic_minutes, wall_minutes) >= maximum_minutes

    def _show_session_limit(self) -> None:
        self.halted_for_session_limit = True
        self.active = None
        self.photo = None
        self.panel.configure(
            image="",
            text=(
                "동결한 세션 최대시간에 도달했습니다.\n"
                "이 block을 더 수집하거나 다른 세션으로 이어서 채우지 마세요.\n"
                "사전등록 정책에 따라 이 참가자 자료는 확증 분석에 사용할 수 없습니다.\n"
                "Esc로 종료하세요.\n"
                "최초 기록 시각부터 누적되므로 앱 재시작으로 시간을 초기화할 수 없습니다."
            ),
        )

    def _begin_break(self) -> None:
        if self._session_limit_reached():
            self._show_session_limit()
            return
        self.active = None
        self.photo = None
        duration_min = float(self.plan.config["timing"]["planned_break_duration_min"])
        self.panel.configure(
            image="",
            text=f"정기 휴식 {duration_min:g}분\n중단하려면 Esc를 누르세요.",
        )
        self.root.unbind("<space>")
        self.break_after_id = self.root.after(
            int(round(duration_min * 60_000.0)), self._break_finished
        )

    def _break_finished(self) -> None:
        self.break_after_id = None
        if self._session_limit_reached():
            self._show_session_limit()
            return
        self.panel.configure(image="", text="휴식 완료\n계속: Space   중단: Esc")
        self.root.bind("<space>", self._resume_after_break)

    def _resume_after_break(self, _event: Any = None) -> None:
        self.root.unbind("<space>")
        self._fixation()

    def _record_abort(self, reason: str) -> None:
        if self.active is None or self.onset_ns is None or self.onset_utc is None:
            return
        record = build_trial_record(
            self.active,
            config=self.plan.config,
            config_sha256=self.plan.config_sha256,
            stimulus_manifest_sha256=self.plan.manifest_sha256,
            session_id=self.session_id,
            phase=self.plan.phase,
            validation_plan_sha256=self.plan.validation_plan_sha256,
            onset_utc=self.onset_utc,
            onset_monotonic_ns=self.onset_ns,
            render_time_ms=float(self.render_time_ms),
            abort_reason=reason,
        )
        self.journal.append(record)
        self.active = None

    def _timeout(self) -> None:
        self.deadline_after_id = None
        try:
            self._record_abort("response_deadline_exceeded")
        except (JournalError, ResponseGUIBlockedError, OSError) as error:
            self._fatal(str(error))
            return
        if self.plan.phase != "practice":
            self._fatal(
                "본 시행의 활성 자극이 중단되어 이 참가자 자료를 확증 분석에 "
                "사용할 수 없으며 이 시행을 재제시할 수 없습니다."
            )
            return
        self.index += 1
        self.panel.configure(image="", text="시간 초과")
        delay = int(round(float(self.plan.config["timing"]["inter_trial_ms"])))
        self.root.after(delay, self._fixation)

    def _escape(self, _event: Any = None) -> None:
        self._cancel_deadline()
        if self.break_after_id is not None:
            self.root.after_cancel(self.break_after_id)
            self.break_after_id = None
        try:
            self._record_abort("participant_escape")
        except (JournalError, ResponseGUIBlockedError, OSError) as error:
            self._fatal(str(error))
            return
        self.root.destroy()

    def _fatal(self, message: str) -> None:
        self._cancel_deadline()
        if self.break_after_id is not None:
            self.root.after_cancel(self.break_after_id)
            self.break_after_id = None
        self.active = None
        self.photo = None
        self.panel.configure(
            image="", text=f"자료 수집 차단\n{message}\n\nEsc로 종료", foreground="#ff8080"
        )

    def _finish(self) -> None:
        self.active = None
        self.photo = None
        unfinished = remaining_trials(self.plan, self.journal)
        if unfinished:
            if self.plan.phase == "practice":
                message = (
                    "연습 1차 진행 완료\n"
                    f"미완료 {len(unfinished)}개\n"
                    "Esc로 종료한 뒤 같은 journal로 연습만 다시 실행할 수 있습니다."
                )
            else:
                message = (
                    "본 시행 block 미완료\n"
                    f"미완료 {len(unfinished)}개\n"
                    "재개·재시도하지 마세요. 이 참가자 자료는 확증 분석에 사용할 수 없습니다."
                )
        else:
            message = "세션 완료\nEsc로 종료"
        self.panel.configure(image="", text=message)


def run_gui(plan: CollectionPlan, *, journal_path: str | Path, session_id: str) -> int:
    _assert_session_id(session_id)
    path = Path(journal_path).resolve()
    _assert_journal_location(path, plan)
    try:
        expected_session = expected_collection_session_id(
            plan.config,
            plan.phase,
            session_block=plan.session_block,
            validation_block_id=plan.validation_block_id,
        )
    except CollectionBlockedError as error:
        raise ResponseGUIBlockedError(str(error)) from error
    if session_id != expected_session:
        raise ResponseGUIBlockedError(
            "--session-id가 config에 동결된 현재 block 값과 다릅니다."
        )
    journal = TrialJournal(path)
    # 이 canonical 파일 하나에는 한 가명 참가자·한 동결 config·한 phase만 둔다.
    for record in journal.records:
        if (
            record.get("participant_id") != plan.participant_id
            or record.get("config_sha256") != plan.config_sha256
            or record.get("phase") != plan.phase
        ):
            raise ResponseGUIBlockedError(
                "기존 canonical 저널이 다른 참가자·config·phase 레코드를 포함합니다."
            )
    if plan.phase in {"calibration", "validation"} and any(
        record.get("trial_status") == "aborted" for record in journal.records
    ):
        raise ResponseGUIBlockedError(
            "활성 자극 중 중단 기록이 있어 사전등록 정책상 이 참가자 자료는 "
            "확증 분석에 사용할 수 없고 같은 시행을 재시도할 수 없습니다."
        )
    # Tk 생성 전에 현재 block/session 결박과 다른 block의 session 재사용을 검사한다.
    session_history_start_utc(plan, journal, session_id)
    _set_windows_dpi_awareness()
    try:
        import tkinter as tk

        root = tk.Tk()
    except (ImportError, RuntimeError, OSError) as error:
        raise ResponseGUIBlockedError(f"Tk 전체화면을 시작하지 못했습니다: {error}") from error
    ExperimentWindow(root, plan=plan, journal=journal, session_id=session_id)
    root.mainloop()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="검증된 4AFC 자극의 dry-run 점검 또는 명시적 전체화면 응답 수집"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phase", choices=("practice", "calibration", "validation"), required=True)
    parser.add_argument("--session-block", type=int)
    parser.add_argument(
        "--validation-block",
        choices=("V01", "V02", "V03"),
        help="validation 실행에서 선택할 독립 64시행 블록",
    )
    parser.add_argument(
        "--calibration-manifest",
        help="practice trial_id가 본 시행과 겹치지 않는지 확인할 전체 calibration manifest",
    )
    parser.add_argument(
        "--validation-plan",
        help="held-out 응답 전에 동결한 validation plan JSON",
    )
    parser.add_argument(
        "--calibration-model-fits",
        help="validation plan이 참조하는 calibration model fits JSON",
    )
    parser.add_argument("--journal", help="실제 GUI 실행 시 append-only JSONL 경로")
    parser.add_argument("--session-id", help="개인정보가 아닌 세션 코드")
    parser.add_argument(
        "--run-gui",
        action="store_true",
        help="명시한 경우에만 전체화면 사람 응답 수집을 시작",
    )
    parser.add_argument(
        "--confirm-consent",
        action="store_true",
        help="해당 세션 참가자의 동의가 확인됐음을 운영자가 명시",
    )
    return parser


def _assert_repository_stage_ready(phase: str, repo_root: Path) -> None:
    """Require the full repository gate before opening a human-session GUI."""

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "src" / "research_readiness.py"),
            "--root",
            str(repo_root),
            "--format",
            "json",
        ],
        cwd=repo_root / "src",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        raise ResponseGUIBlockedError(
            "전체 연구 준비도 검사를 실행하지 못했습니다: "
            + (completed.stderr.strip() or f"exit={completed.returncode}")
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ResponseGUIBlockedError(
            "전체 연구 준비도 검사 결과가 유효한 JSON이 아닙니다."
        ) from error
    readiness_key = (
        "ready_for_validation_collection"
        if phase == "validation"
        else "ready_for_calibration_collection"
    )
    if report.get(readiness_key) is True:
        return
    blocked_group = (
        "validation_collection" if phase == "validation" else "calibration_collection"
    )
    blocked = report.get("blocked", {}).get(blocked_group, [])
    reasons = "; ".join(
        f"{item.get('check_id', 'unknown')}: {item.get('message', '')}"
        for item in blocked
        if isinstance(item, Mapping)
    )
    raise ResponseGUIBlockedError(
        "전체 연구 준비도 게이트가 이 사람 세션을 허용하지 않습니다"
        + (f": {reasons}" if reasons else ".")
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = prepare_collection(
            config_path=args.config,
            manifest_path=args.manifest,
            phase=args.phase,
            session_block=args.session_block,
            validation_block_id=args.validation_block,
            calibration_manifest_path=args.calibration_manifest,
            validation_plan_path=args.validation_plan,
            calibration_model_fits_path=args.calibration_model_fits,
        )
        technical_ready = True
        blockers: list[str] = []
    except (ResponseGUIBlockedError, RendererValidationError, OSError, ValueError) as error:
        plan = None
        technical_ready = False
        blockers = [str(error)]

    if not args.confirm_consent:
        blockers.append("이 세션의 참가자 동의를 --confirm-consent로 확인하지 않았습니다.")
    allowed = technical_ready and args.confirm_consent
    report = {
        "technical_readiness_passed": technical_ready,
        "ethics_approved_in_frozen_config": bool(
            plan and plan.config["privacy"].get("approved") is True
        ),
        "consent_confirmed_for_session": args.confirm_consent,
        "human_data_collection_allowed": allowed,
        "gui_requested": args.run_gui,
        "phase": args.phase,
        "session_block": args.session_block,
        "validation_block_id": args.validation_block,
        "participant_id": plan.participant_id if plan else None,
        "selected_trial_count": len(plan.trials) if plan else 0,
        "config_sha256": plan.config_sha256 if plan else None,
        "manifest_sha256": plan.manifest_sha256 if plan else None,
        "validation_plan_sha256": plan.validation_plan_sha256 if plan else None,
        "renderer_implementation_sha256": (
            renderer_implementation_sha256() if plan else None
        ),
        "verified_screen_environment": (
            {
                "width_px": plan.screen_metrics.width_px,
                "height_px": plan.screen_metrics.height_px,
                "os_scale_percent": plan.screen_metrics.os_scale_percent,
                "dpi_awareness_enabled": plan.screen_metrics.dpi_awareness_enabled,
                "operating_system": plan.screen_metrics.operating_system,
                "native_gui_runtime": plan.screen_metrics.native_gui_runtime,
                "source": plan.screen_metrics.source,
            }
            if plan
            else None
        ),
        "blockers": blockers,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.run_gui:
        return 0 if technical_ready else 2
    if not allowed:
        return 2
    if not args.journal or not args.session_id:
        print("자료 수집 차단: --journal과 --session-id가 필요합니다.", file=sys.stderr)
        return 2
    try:
        _assert_repository_stage_ready(args.phase, plan.repo_root)
        return run_gui(plan, journal_path=args.journal, session_id=args.session_id)
    except (ResponseGUIBlockedError, JournalError, OSError) as error:
        print(f"자료 수집 차단: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CollectionPlan",
    "ExperimentWindow",
    "GitState",
    "ResponseGUIBlockedError",
    "ScreenMetrics",
    "VerifiedTrial",
    "build_trial_record",
    "inspect_git_state",
    "inspect_screen_metrics",
    "main",
    "prepare_collection",
    "remaining_trials",
    "run_gui",
    "validate_screen_metrics",
]
