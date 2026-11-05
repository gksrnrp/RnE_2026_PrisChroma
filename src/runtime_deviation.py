"""동결 후 Windows 빌드가 바뀐 경우의 좁은 교정 수집 예외.

config를 고치면 이미 수집한 블록과의 hash 결합이 끊기므로 config는 그대로 둔다.
비공개 승인 기록이 config hash, 동결/관측 OS 문자열, 자극 commit, 실행 commit에
모두 정확히 결합될 때만 OS 불일치를 허용한다. 해상도·배율·Python/Tk·PNG·동의
검사는 면제하지 않으며, 실제 OS와 실행 commit을 세션마다 로그에 남긴다.
이 예외는 OS 차이가 결과에 영향이 없다는 근거가 아니다.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

RECORD_PATH = "data/experiment/private/runtime-os-deviation.json"
LOG_PATH = "data/experiment/private/runtime-os-deviation-log.jsonl"
ARTIFACT_TYPE = "runtime_os_build_deviation_v1"
# 교정 준비도 게이트가 연습 자극도 함께 검증하므로 practice를 포함한다.
ALLOWED_PHASES = frozenset({"practice", "calibration"})
# 자극·심리측정·스키마 파일이 바뀌면 이 예외로 실행할 수 없다.
ALLOWED_CHANGED_FILES = frozenset(
    {
        "src/response_gui.py",
        "src/runtime_deviation.py",
        "src/test_runtime_deviation.py",
        "docs/code-review.md",
    }
)


class RuntimeDeviationError(Exception):
    """승인 기록이 있지만 현재 실행 상태와 결합되지 않을 때 발생한다."""


def git_changed_files(root: Path, base: str, head: str) -> Sequence[str]:
    """base가 head의 조상일 때만 둘 사이에서 바뀐 파일을 돌려준다."""

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=8
        )

    try:
        ancestor = run("merge-base", "--is-ancestor", base, head)
        diff = run("diff", "--name-only", base, head)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeDeviationError(f"Git 비교 실패: {error}") from error
    if ancestor.returncode != 0 or diff.returncode != 0:
        raise RuntimeDeviationError("실행 commit이 자극 commit에서 이어지지 않습니다.")
    return [line for line in diff.stdout.splitlines() if line]


def load_os_deviation(
    root: str | Path,
    *,
    phase: str,
    config_sha256: str,
    frozen_operating_system: Any,
    observed_operating_system: str,
    head_commit: str,
    changed_files: Callable[[Path, str, str], Sequence[str]] = git_changed_files,
) -> dict[str, Any] | None:
    """유효한 승인 기록을 돌려준다. OS가 같거나 기록이 없으면 None이다."""
    if observed_operating_system == frozen_operating_system:
        return None
    base = Path(root)
    path = base / RECORD_PATH
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeDeviationError(f"OS 예외 기록을 읽지 못했습니다: {error}") from error
    if not isinstance(record, Mapping) or record.get("artifact_type") != ARTIFACT_TYPE:
        raise RuntimeDeviationError("OS 예외 기록 형식이 아닙니다.")
    if record.get("approved") is not True or not str(record.get("approved_by") or "").strip():
        raise RuntimeDeviationError("OS 예외 기록에 승인자가 없습니다.")
    if phase not in ALLOWED_PHASES or phase not in (record.get("allowed_phases") or []):
        raise RuntimeDeviationError(f"OS 예외는 {phase} 단계에 적용되지 않습니다.")
    expected = {
        "config_sha256": config_sha256,
        "frozen_operating_system": frozen_operating_system,
        "observed_operating_system": observed_operating_system,
        "execution_commit": head_commit,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise RuntimeDeviationError(
                f"OS 예외 기록의 {field}가 현재 상태와 다릅니다: "
                f"기록={record.get(field)!r}, 현재={value!r}"
            )
    base_commit = record.get("base_commit")
    if not isinstance(base_commit, str) or not base_commit:
        raise RuntimeDeviationError("OS 예외 기록에 base_commit이 없습니다.")
    extra = sorted(set(changed_files(base, base_commit, head_commit)) - ALLOWED_CHANGED_FILES)
    if extra:
        raise RuntimeDeviationError(f"허용 범위 밖 파일이 바뀌었습니다: {extra}")
    return dict(record)


def append_session_log(
    root: str | Path,
    record: Mapping[str, Any],
    *,
    session_id: str,
    phase: str,
    session_block: int | None,
) -> None:
    """시행 저장 전에 이 세션이 예외로 실행됐음을 남긴다. 실패하면 수집을 막는다."""
    base = Path(root)
    row = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "session_id": session_id,
        "phase": phase,
        "session_block": session_block,
        "frozen_operating_system": record["frozen_operating_system"],
        "observed_operating_system": record["observed_operating_system"],
        "stimulus_commit": record["base_commit"],
        "execution_commit": record["execution_commit"],
        "config_sha256": record["config_sha256"],
        "record_sha256": sha256((base / RECORD_PATH).read_bytes()).hexdigest(),
    }
    with (base / LOG_PATH).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
