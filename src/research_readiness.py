"""사람 대상 실험과 연구 결론의 준비 상태를 보수적으로 점검한다.

이 검사는 연구를 대신 수행하지 않는다. 저장소에 남은 초안, 가짜 자료,
출처 및 분석 provenance 누락을 찾아서 실수로 사람 자료를 받거나 예제 결과를
연구 결과로 보고하는 일을 막는 정적 안전장치다.

저장소 루트에서 다음처럼 실행한다.

    python src/research_readiness.py --format markdown
    python src/research_readiness.py --format json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError

from calibration_design import block_count_from_config
from recovery_coverage import validate_coverage_payload, summarise_coverage, MODELS, PROFILES
from implementation_provenance import (
    ImplementationProvenanceError,
    verify_implementation_hashes,
)


DEFAULT_MINIMUM_REAL_IMAGES = 20
MINIMUM_HUMAN_BOOTSTRAP_REPLICATES = 1000
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PLACEHOLDER_RE = re.compile(
    r"(?:\bTBD\b|\bTODO\b|\bEXAMPLE\b|\bPLACEHOLDER\b|"
    r"NOT[_ -]?(?:FROZEN|FOR[_ -]?HUMAN)|미정|예시|가짜)",
    re.IGNORECASE,
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
HUMAN_SCHEMA_NAMES = (
    "experiment_config.schema.json",
    "trial.schema.json",
    "participant_thresholds.schema.json",
)


@dataclass(frozen=True)
class CheckResult:
    """한 준비 상태 검사의 기계 판정과 사람용 설명."""

    check_id: str
    title: str
    status: str
    message: str
    evidence: list[str]
    blocks_human_data: bool
    blocks_research_claims: bool


def _check(
    check_id: str,
    title: str,
    passed: bool,
    passed_message: str,
    blocked_message: str,
    *,
    evidence: Iterable[str] = (),
    blocks_human_data: bool,
    blocks_research_claims: bool,
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        title=title,
        status="pass" if passed else "blocked",
        message=passed_message if passed else blocked_message,
        evidence=list(evidence),
        blocks_human_data=blocks_human_data,
        blocks_research_claims=blocks_research_claims,
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _human_schema_path(root: Path, name: str) -> Path:
    return root / "schemas" / "human" / name


def _schema_instance_problems(
    root: Path,
    schema_name: str,
    payload: Any,
    label: str,
) -> list[str]:
    """사람용 1.0 schema가 있을 때 실제 artifact를 그 schema로 검사한다."""

    path = _human_schema_path(root, schema_name)
    if not path.is_file():
        # 스키마 자체 누락은 human_contract_schemas 검사에서 한 번만 보고한다.
        return []
    try:
        schema = _read_json(path)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    except (OSError, json.JSONDecodeError, SchemaError, TypeError) as error:
        return [f"{label}: 사람용 schema를 적용할 수 없음 ({error})"]
    problems: list[str] = []
    for error in errors[:5]:
        location = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error.path
        )
        problems.append(f"{label}: 1.0 schema 오류 {location}: {error.message}")
    if len(errors) > 5:
        problems.append(f"{label}: 그 밖의 schema 오류 {len(errors) - 5}개")
    return problems


def _json_records(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path.name}:{line_number}: {error}") from error
        return records
    return [_read_json(path)]


def _placeholder_paths(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_placeholder_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_placeholder_paths(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and (not value.strip() or PLACEHOLDER_RE.search(value)):
        found.append(prefix)
    return found


def _find_non_example_json(root: Path, stem: str) -> list[Path]:
    bases = [root / "data" / "experiment", root / "data" / "contracts", root / "results"]
    found: set[Path] = set()
    for base in bases:
        if not base.exists():
            continue
        for path in base.rglob(f"*{stem}*.json"):
            lowered = path.name.lower()
            if "example" not in lowered and "synthetic" not in lowered:
                found.add(path)
    root_candidate = root / f"{stem}.json"
    if root_candidate.is_file():
        found.add(root_candidate)
    return sorted(found)


def _canonical_experiment_config(root: Path) -> Path | None:
    """Return the one active participant config, never archived/peer configs."""

    for path in (
        root / "data" / "experiment" / "private" / "experiment_config.json",
        root / "data" / "experiment" / "experiment_config.json",
    ):
        if path.is_file():
            return path
    return None


def _check_experiment_config(root: Path) -> CheckResult:
    # 준비도는 현재 실행할 한 참가자의 canonical 설정만 검사한다. 이전 참가자
    # archive까지 재귀 수집하면 서로 다른 config/hash를 현재 plan에 잘못 결합한다.
    canonical = _canonical_experiment_config(root)
    candidates = [canonical] if canonical is not None else []
    evidence = [_relative(path, root) for path in candidates]
    if not candidates:
        return _check(
            "experiment_config_final",
            "사람 실험용 설정 동결",
            False,
            "",
            "실제 사람 실험용 experiment_config가 없습니다. 예제 설정으로 사람 자료를 받으면 안 됩니다.",
            evidence=["data/contracts/experiment_config.example.json은 예제일 뿐입니다."],
            blocks_human_data=True,
            blocks_research_claims=True,
        )

    problems: list[str] = []
    for path in candidates:
        label = _relative(path, root)
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"{label}: JSON을 읽을 수 없음 ({error})")
            continue
        if not isinstance(payload, dict):
            problems.append(f"{label}: 최상위 값이 객체가 아님")
            continue
        problems.extend(
            _schema_instance_problems(
                root,
                "experiment_config.schema.json",
                payload,
                label,
            )
        )
        version = str(payload.get("schema_version", ""))
        if not version or "draft" in version.lower():
            problems.append(f"{label}: schema_version이 초안임")
        if payload.get("status") not in {"frozen", "final", "registered", "preregistered"}:
            problems.append(f"{label}: status가 동결 상태가 아님")
        if payload.get("human_data_collection_allowed") is not True:
            problems.append(f"{label}: human_data_collection_allowed가 true가 아님")
        placeholders = _placeholder_paths(payload)
        if placeholders:
            problems.append(
                f"{label}: 미정/예제 값이 남음 ({', '.join(placeholders[:5])})"
            )

    return _check(
        "experiment_config_final",
        "사람 실험용 설정 동결",
        not problems,
        "초안과 placeholder가 없는 동결 설정이며 사람 자료 수집이 명시적으로 허용되어 있습니다.",
        "사람 실험 설정이 아직 동결되지 않았습니다: " + "; ".join(problems),
        evidence=evidence,
        blocks_human_data=True,
        blocks_research_claims=True,
    )


def _runtime_collection_gate(config: dict[str, Any]) -> dict[str, Any]:
    """실행기의 분리된 config/renderer/응답 GUI 게이트를 그대로 사용한다."""

    from experiment_runner import assess_experiment_gates

    return assess_experiment_gates(config)


def _find_compiled_manifest(
    root: Path, *, artifact_type: str, config_sha256: str
) -> Path:
    relative_by_type = {
        "compiled_pixel_stimulus_manifest": Path("calibration/calibration-manifest.json"),
        "compiled_practice_pixel_stimulus_manifest": Path("practice/practice-manifest.json"),
        "compiled_validation_pixel_stimulus_manifest": Path("validation/validation-manifest.json"),
    }
    relative = relative_by_type.get(artifact_type)
    if relative is None:
        raise ValueError(f"지원하지 않는 manifest artifact_type입니다: {artifact_type}")
    candidate = root / "data" / "experiment" / relative
    try:
        payload = _read_json(candidate)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"canonical {artifact_type} manifest를 읽을 수 없습니다: {candidate}: {error}"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("artifact_type") != artifact_type
        or payload.get("schedule_config_sha256") != config_sha256
    ):
        raise ValueError(
            f"canonical {artifact_type} manifest가 현재 config와 결합되지 않았습니다."
        )
    return candidate


def _runtime_artifact_gate(
    root: Path,
    config_path: Path,
    *,
    phase: str,
) -> tuple[dict[str, Any], list[Path]]:
    """response GUI의 실제 manifest·화면·Git dry-run 게이트를 실행한다."""

    from response_gui import ResponseGUIBlockedError, prepare_collection

    config_sha256 = _sha256(config_path)
    evidence: list[Path] = []
    try:
        if phase == "calibration":
            config = _read_json(config_path)
            calibration = _find_compiled_manifest(
                root,
                artifact_type="compiled_pixel_stimulus_manifest",
                config_sha256=config_sha256,
            )
            practice = _find_compiled_manifest(
                root,
                artifact_type="compiled_practice_pixel_stimulus_manifest",
                config_sha256=config_sha256,
            )
            evidence.extend((calibration, practice))
            prepare_collection(
                config_path=config_path,
                manifest_path=practice,
                phase="practice",
                session_block=None,
                repo_root=root,
                calibration_manifest_path=calibration,
            )
            for block in range(1, block_count_from_config(config) + 1):
                prepare_collection(
                    config_path=config_path,
                    manifest_path=calibration,
                    phase="calibration",
                    session_block=block,
                    repo_root=root,
                )
        elif phase == "validation":
            manifest = _find_compiled_manifest(
                root,
                artifact_type="compiled_validation_pixel_stimulus_manifest",
                config_sha256=config_sha256,
            )
            plan = root / "data" / "experiment" / "validation" / "validation-plan.json"
            calibration_fits = root / "results" / "calibration_model_fits.json"
            evidence.extend((manifest, plan, calibration_fits))
            for block in ("V01", "V02", "V03"):
                prepare_collection(
                    config_path=config_path,
                    manifest_path=manifest,
                    phase="validation",
                    session_block=None,
                    validation_block_id=block,
                    repo_root=root,
                    validation_plan_path=plan,
                    calibration_model_fits_path=calibration_fits,
                )
        else:
            raise ValueError(f"지원하지 않는 runtime phase입니다: {phase}")
    except (OSError, ResponseGUIBlockedError, ValueError) as error:
        return {
            "human_data_collection_allowed": False,
            "blockers": [{"gate": f"{phase}_runtime", "message": str(error)}],
        }, evidence
    return {
        "human_data_collection_allowed": True,
        "blockers": [],
    }, evidence


def _check_experiment_runtime(root: Path) -> CheckResult:
    canonical = _canonical_experiment_config(root)
    candidates = [canonical] if canonical is not None else []
    evidence = [_relative(path, root) for path in candidates]
    problems: list[str] = []
    if not candidates:
        problems.append("실제 사람 실험용 experiment_config가 없음")
    for path in candidates:
        label = _relative(path, root)
        try:
            payload = _read_json(path)
            if not isinstance(payload, dict):
                problems.append(f"{label}: 최상위 객체가 아님")
                continue
            report = _runtime_collection_gate(payload)
            if (
                report.get("human_data_collection_allowed") is not True
                and report.get("config_gate_passed") is True
                and report.get("renderer_verified") is True
                and report.get("human_response_collection_implemented") is True
            ):
                report, artifact_evidence = _runtime_artifact_gate(
                    root, path, phase="calibration"
                )
                evidence.extend(_relative(item, root) for item in artifact_evidence)
        except (OSError, json.JSONDecodeError, ImportError, ValueError) as error:
            problems.append(f"{label}: 실행 게이트 확인 실패 ({error})")
            continue
        if report.get("human_data_collection_allowed") is not True:
            blockers = report.get("blockers")
            if isinstance(blockers, list):
                messages = [
                    f"{item.get('gate', 'unknown')}: {item.get('message', '차단')}"
                    for item in blockers
                    if isinstance(item, dict)
                ]
            else:
                messages = []
            problems.append(
                f"{label}: " + ("; ".join(messages) or "실행 게이트가 수집을 허용하지 않음")
            )

    return _check(
        "experiment_runtime_gates",
        "자극 렌더러·응답 수집 실행 게이트",
        not problems,
        "설정, 픽셀 렌더러, 응답·중단·저장 GUI 게이트가 모두 통과했습니다.",
        "설정 파일만으로 사람 자료를 받을 수 없습니다: " + "; ".join(problems),
        evidence=evidence + ["src/experiment_runner.py"],
        blocks_human_data=True,
        blocks_research_claims=True,
    )


def _check_validation_runtime(root: Path) -> CheckResult:
    canonical = _canonical_experiment_config(root)
    candidates = [canonical] if canonical is not None else []
    evidence = [_relative(path, root) for path in candidates]
    problems: list[str] = []
    if not candidates:
        problems.append("실제 사람 실험용 experiment_config가 없음")
    for path in candidates:
        label = _relative(path, root)
        try:
            payload = _read_json(path)
            if not isinstance(payload, dict):
                problems.append(f"{label}: 최상위 객체가 아님")
                continue
            config_report = _runtime_collection_gate(payload)
            if config_report.get("human_data_collection_allowed") is True:
                # 테스트 fixture 또는 외부 통합 게이트가 전체 허가를 명시한 경우.
                continue
            if not (
                config_report.get("config_gate_passed") is True
                and config_report.get("renderer_verified") is True
                and config_report.get("human_response_collection_implemented") is True
            ):
                blockers = config_report.get("blockers", [])
                messages = [
                    str(item.get("message", "차단"))
                    for item in blockers
                    if isinstance(item, dict)
                ]
                problems.append(f"{label}: " + "; ".join(messages))
                continue
            report, artifact_evidence = _runtime_artifact_gate(
                root, path, phase="validation"
            )
            evidence.extend(_relative(item, root) for item in artifact_evidence)
            if report.get("human_data_collection_allowed") is not True:
                messages = [
                    str(item.get("message", "차단"))
                    for item in report.get("blockers", [])
                    if isinstance(item, dict)
                ]
                problems.append(f"{label}: " + ("; ".join(messages) or "검증 runtime 차단"))
        except (OSError, json.JSONDecodeError, ImportError, ValueError) as error:
            problems.append(f"{label}: 검증 실행 게이트 확인 실패 ({error})")

    return _check(
        "validation_runtime_gates",
        "validation PNG·응답 GUI 실행 게이트",
        not problems,
        "세 validation 블록의 manifest, 픽셀, 화면, Git, 응답 GUI dry-run이 통과했습니다.",
        "validation 자료를 받을 수 없습니다: " + "; ".join(problems),
        evidence=evidence + ["src/response_gui.py"],
        blocks_human_data=False,
        blocks_research_claims=True,
    )


def _check_contract_examples(root: Path) -> CheckResult:
    contract_dir = root / "data" / "contracts"
    examples = sorted(contract_dir.glob("*.example.json")) + sorted(
        contract_dir.glob("*.example.jsonl")
    )
    problems: list[str] = []
    evidence = [_relative(path, root) for path in examples]
    if not examples:
        problems.append("계약 예제가 없음")
    for path in examples:
        try:
            records = _json_records(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            problems.append(f"{path.name}: 읽기 실패 ({error})")
            continue
        if not records:
            problems.append(f"{path.name}: 빈 예제")
        for index, payload in enumerate(records, start=1):
            if not isinstance(payload, dict):
                problems.append(f"{path.name}#{index}: 객체가 아님")
                continue
            explicitly_safe = payload.get("data_origin") == "synthetic" or (
                payload.get("human_data_collection_allowed") is False
            )
            if not explicitly_safe:
                problems.append(
                    f"{path.name}#{index}: synthetic 또는 human_data_collection_allowed=false 표시가 없음"
                )

    return _check(
        "contract_examples_synthetic",
        "계약 예제의 가짜 자료 표시",
        not problems,
        "모든 계약 예제가 기계 판독 가능한 방식으로 가짜 자료임을 표시합니다.",
        "실제 자료와 예제가 섞일 위험이 있습니다: " + "; ".join(problems),
        evidence=evidence,
        blocks_human_data=True,
        blocks_research_claims=True,
    )


def _walk_mapping(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_mapping(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_mapping(child, path + (str(index),))


def _check_human_contract_schemas(root: Path) -> CheckResult:
    # 0.1-draft 합성 계약과 실제 사람 자료 계약을 같은 폴더에서 훑으면,
    # 의도적으로 synthetic/false로 잠근 draft가 사람 계약을 영구 차단한다.
    # 사람 자료 가능성은 오직 이 전용 1.0 디렉터리만 근거로 판단한다.
    schema_dir = root / "schemas" / "human"
    required = set(HUMAN_SCHEMA_NAMES)
    present = {path.name for path in schema_dir.glob("*.schema.json")}
    problems = [f"필수 스키마 없음: {name}" for name in sorted(required - present)]
    evidence: list[str] = []
    for name in HUMAN_SCHEMA_NAMES:
        path = schema_dir / name
        if not path.is_file():
            continue
        label = _relative(path, root)
        evidence.append(label)
        try:
            payload = _read_json(path)
            Draft202012Validator.check_schema(payload)
        except (OSError, json.JSONDecodeError, SchemaError, TypeError, ValueError) as error:
            problems.append(f"{label}: 읽기 실패 ({error})")
            continue
        if not isinstance(payload, dict):
            problems.append(f"{label}: 최상위 schema가 객체가 아님")
            continue
        if payload.get("type") != "object" or payload.get("additionalProperties") is not False:
            problems.append(f"{label}: 최상위 객체가 additionalProperties=false로 닫혀 있지 않음")
        version_rule = payload.get("properties", {}).get("schema_version", {})
        if version_rule.get("const") != "1.0":
            problems.append(f"{label}: schema_version=1.0을 강제하지 않음")

        properties = payload.get("properties", {})
        if name == "experiment_config.schema.json":
            allowed_rule = properties.get("human_data_collection_allowed", {})
            if allowed_rule.get("type") != "boolean":
                problems.append(
                    f"{label}: human_data_collection_allowed가 명시적 boolean이 아님"
                )
        else:
            origin_rule = properties.get("data_origin", {})
            if origin_rule.get("const") != "human":
                problems.append(f"{label}: data_origin=human을 강제하지 않음")

        for json_path, value in _walk_mapping(payload):
            dotted = ".".join(json_path)
            if json_path and json_path[-1] == "const" and isinstance(value, str):
                if "draft" in value.lower():
                    problems.append(f"{label}: 초안 버전 {value!r}을 강제함")
                if value == "synthetic" and "data_origin" in json_path:
                    problems.append(f"{label}: data_origin=synthetic만 허용함")
            if (
                isinstance(value, dict)
                and value.get("type") == "object"
                and "properties" in value
                and value.get("additionalProperties") is not False
            ):
                problems.append(f"{label}: {dotted or '$'} 객체가 엄격히 닫혀 있지 않음")

        if name == "participant_thresholds.schema.json":
            participant_rule = payload.get("$defs", {}).get("participant", {})
            confirmed_rule = participant_rule.get("properties", {}).get("confirmed", {})
            if confirmed_rule.get("type") != "boolean":
                problems.append(f"{label}: confirmed가 명시적 boolean이 아님")

    # 같은 원인이 스키마 여러 곳에서 반복될 수 있으므로 보고서를 짧게 만든다.
    problems = list(dict.fromkeys(problems))
    return _check(
        "human_contract_schemas",
        "실제 자료용 계약 스키마",
        not problems,
        "schemas/human의 세 핵심 스키마가 유효한 1.0 계약이며 엄격한 객체로 닫혀 있습니다.",
        "계약이 아직 실제 사람 자료용이 아닙니다: " + "; ".join(problems),
        evidence=evidence,
        blocks_human_data=True,
        blocks_research_claims=True,
    )


def _check_participant_thresholds(root: Path) -> CheckResult:
    # 현재 참가자의 canonical 결과 한 개만 검사한다. results/participants 아래의
    # 과거 비식별 archive는 해당 참가자의 전체 private provenance bundle 없이
    # 현재 config/plan에 대조하면 항상 거짓 실패가 되므로 여기서 제외한다.
    canonical_candidates = (
        root / "results" / "participant_thresholds.json",
        root / "data" / "experiment" / "participant_thresholds.json",
    )
    candidates = [path for path in canonical_candidates if path.is_file()][:1]
    evidence = [_relative(path, root) for path in candidates]
    problems: list[str] = []
    config_files = [
        path
        for path in (
            root / "data" / "experiment" / "private" / "experiment_config.json",
            root / "data" / "experiment" / "experiment_config.json",
        )
        if path.is_file()
    ][:1]
    config_by_hash: dict[str, tuple[Path, dict[str, Any]]] = {}
    for config_path in config_files:
        try:
            config_payload = _read_json(config_path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(config_payload, dict):
            config_by_hash[_sha256(config_path)] = (config_path, config_payload)
    plan_path = root / "data" / "experiment" / "validation" / "validation-plan.json"
    plan_payload: dict[str, Any] | None = None
    if plan_path.is_file():
        try:
            loaded_plan = _read_json(plan_path)
            if isinstance(loaded_plan, dict):
                plan_payload = loaded_plan
        except (OSError, json.JSONDecodeError):
            pass
    raw_trial_hashes = {
        _sha256(path): path
        for path in (root / "data" / "experiment").rglob("*.jsonl")
        if path.is_file() and "example" not in path.name.lower()
    } if (root / "data" / "experiment").is_dir() else {}
    validation_manifest_hashes: dict[str, Path] = {}
    experiment_dir = root / "data" / "experiment"
    if experiment_dir.is_dir():
        for manifest_path in experiment_dir.rglob("*.json"):
            try:
                manifest_payload = _read_json(manifest_path)
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(manifest_payload, dict)
                and manifest_payload.get("artifact_type")
                == "compiled_validation_pixel_stimulus_manifest"
            ):
                validation_manifest_hashes[_sha256(manifest_path)] = manifest_path
    analysis_path = root / "src" / "analyze_validation.py"
    current_analysis_sha256 = _sha256(analysis_path) if analysis_path.is_file() else None
    if not candidates:
        problems.append("실제 participant_thresholds 파일이 없음")
    participant_count = 0
    for path in candidates:
        label = _relative(path, root)
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"{label}: 읽기 실패 ({error})")
            continue
        if not isinstance(payload, dict):
            problems.append(f"{label}: 최상위 객체가 아님")
            continue
        problems.extend(
            _schema_instance_problems(
                root,
                "participant_thresholds.schema.json",
                payload,
                label,
            )
        )
        if payload.get("data_origin") != "human":
            problems.append(f"{label}: data_origin이 실제 관측 자료로 표시되지 않음")
        if "draft" in str(payload.get("schema_version", "")).lower():
            problems.append(f"{label}: 초안 계약임")
        config_binding = config_by_hash.get(payload.get("config_sha256"))
        if config_binding is None:
            problems.append(f"{label}: config_sha256과 일치하는 실제 설정 파일이 없음")
        else:
            config_path, config_payload = config_binding
            evidence.append(_relative(config_path, root))
            if payload.get("config_version") != config_payload.get("config_version"):
                problems.append(f"{label}: config_version이 원 설정과 다름")
        if (
            plan_payload is None
            or payload.get("source_validation_plan_sha256")
            != plan_payload.get("plan_sha256")
        ):
            problems.append(f"{label}: source validation plan hash가 현재 동결 plan과 다름")
        else:
            evidence.append(_relative(plan_path, root))
        source_trials = payload.get("source_trials_sha256")
        if source_trials not in raw_trial_hashes:
            problems.append(f"{label}: source_trials_sha256과 일치하는 원 JSONL이 없음")
        else:
            evidence.append(_relative(raw_trial_hashes[source_trials], root))
        source_manifest = payload.get("source_stimulus_manifest_sha256")
        if source_manifest not in validation_manifest_hashes:
            problems.append(f"{label}: source stimulus manifest hash와 일치하는 파일이 없음")
        else:
            evidence.append(_relative(validation_manifest_hashes[source_manifest], root))
        if (
            current_analysis_sha256 is None
            or payload.get("analysis_script_sha256") != current_analysis_sha256
        ):
            problems.append(f"{label}: analyze_validation.py 구현 hash가 현재 코드와 다름")
        else:
            evidence.append(_relative(analysis_path, root))
        try:
            verify_implementation_hashes(
                payload.get("implementation_hashes"),
                "validation_analysis",
                project_root=root,
                label=label,
            )
        except ImplementationProvenanceError as error:
            problems.append(str(error))
        if (
            plan_payload is None
            or payload.get("git_commit") != plan_payload.get("source_git_commit")
        ):
            problems.append(f"{label}: 결과의 자료 수집 git_commit이 동결 plan과 다름")
        participants = payload.get("participants")
        if not isinstance(participants, list) or not participants:
            problems.append(f"{label}: 참가자 결과가 없음")
            continue
        participant_count += len(participants)
        for participant in participants:
            participant_id = (
                participant.get("participant_id", "ID 없음")
                if isinstance(participant, dict)
                else "형식 오류"
            )
            if not isinstance(participant, dict) or participant.get("confirmed") is not True:
                problems.append(f"{label}: {participant_id}의 confirmed가 true가 아님")
                continue
            validation = participant.get("validation")
            if not isinstance(validation, dict) or validation.get("stable") is not True:
                problems.append(f"{label}: {participant_id}의 validation이 안정하지 않음")
            validation_blocks = (
                validation.get("blocks") if isinstance(validation, dict) else None
            )
            validation_bootstrap_problem = (
                not isinstance(validation_blocks, list) or not validation_blocks
            )
            if isinstance(validation_blocks, list):
                for block in validation_blocks:
                    repetitions = (
                        block.get("bootstrap_replicates")
                        if isinstance(block, dict)
                        else None
                    )
                    if (
                        isinstance(repetitions, bool)
                        or not isinstance(repetitions, int)
                        or repetitions < MINIMUM_HUMAN_BOOTSTRAP_REPLICATES
                    ):
                        validation_bootstrap_problem = True
            if validation_bootstrap_problem:
                problems.append(
                    f"{label}: {participant_id}의 validation block bootstrap 반복 수는 "
                    "각각 1000회 이상이어야 함"
                )
            comparisons = participant.get("model_comparisons")
            delta_loss = (
                comparisons.get("primary_delta_loss")
                if isinstance(comparisons, dict)
                else None
            )
            if (
                not isinstance(delta_loss, dict)
                or delta_loss.get("status") == "unresolvable"
            ):
                problems.append(
                    f"{label}: {participant_id}의 교정·검증 bootstrap Δloss를 해석할 수 없음"
                )
            predictions = participant.get("predictions")
            prediction_problem = not isinstance(predictions, list) or not predictions
            if isinstance(predictions, list):
                for prediction in predictions:
                    try:
                        repetitions = (
                            prediction.get("bootstrap_replicates")
                            if isinstance(prediction, dict)
                            else None
                        )
                        converged = (
                            isinstance(prediction, dict)
                            and prediction.get("fit_converged") is True
                            and float(prediction.get("bootstrap_convergence_rate", -1)) >= 0.90
                            and not isinstance(repetitions, bool)
                            and isinstance(repetitions, int)
                            and repetitions >= MINIMUM_HUMAN_BOOTSTRAP_REPLICATES
                        )
                    except (TypeError, ValueError):
                        converged = False
                    prediction_problem = prediction_problem or not converged
            if prediction_problem:
                problems.append(
                    f"{label}: {participant_id}의 예측 적합/bootstrap 수렴 근거가 부족함"
                    "(각 예측 1000회 이상 필요)"
                )

    return _check(
        "participant_thresholds_confirmed",
        "실제 참가자 역치 확정",
        not problems,
        f"실제 참가자 {participant_count}명의 확정된 역치 자료가 있습니다.",
        "실제 참가자 역치가 연구 결과로 확정되지 않았습니다: " + "; ".join(problems),
        evidence=evidence,
        blocks_human_data=False,
        blocks_research_claims=True,
    )


def _normalise_source_name(name: str) -> str:
    return name.strip().replace("\\", "/").lstrip("./")


def _check_real_images_and_sources(root: Path, minimum_real_images: int) -> CheckResult:
    image_dir = root / "data" / "real" / "images"
    images = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ) if image_dir.exists() else []
    image_names = {_normalise_source_name(path.relative_to(image_dir).as_posix()) for path in images}
    sources_path = root / "data" / "real" / "sources.csv"
    problems: list[str] = []
    rows: list[dict[str, str]] = []
    if len(images) < minimum_real_images:
        problems.append(
            f"실제 이미지가 {len(images)}장뿐임(사전 목표 최소 {minimum_real_images}장)"
        )
    if not sources_path.is_file():
        problems.append("data/real/sources.csv가 없음")
    else:
        try:
            with sources_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error) as error:
            problems.append(f"sources.csv 읽기 실패 ({error})")

    source_names: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        filename = _normalise_source_name(row.get("filename", ""))
        if not filename:
            problems.append(f"sources.csv {row_number}행: filename이 비어 있음")
            continue
        source_names.append(filename)
        for field in ("source_url", "license"):
            value = (row.get(field) or "").strip()
            if not value:
                problems.append(f"sources.csv {row_number}행: {field}가 비어 있음")
            elif PLACEHOLDER_RE.search(value) or "example.org" in value.lower():
                problems.append(f"sources.csv {row_number}행: {field}가 placeholder임")
        if any(
            PLACEHOLDER_RE.search((row.get(field) or ""))
            for field in ("filename", "notes", "author")
        ):
            problems.append(f"sources.csv {row_number}행: 예시/미정 표시가 남아 있음")

    duplicates = sorted({name for name in source_names if source_names.count(name) > 1})
    if duplicates:
        problems.append("sources.csv filename 중복: " + ", ".join(duplicates))
    source_set = set(source_names)
    missing_rows = sorted(image_names - source_set)
    missing_images = sorted(source_set - image_names)
    if missing_rows:
        problems.append("출처 행이 없는 이미지: " + ", ".join(missing_rows))
    if missing_images:
        problems.append("실제 파일이 없는 출처 행: " + ", ".join(missing_images))

    evidence = [f"실제 이미지 {len(images)}장", f"sources.csv 행 {len(rows)}개"]
    return _check(
        "real_images_sources",
        "실제 이미지와 출처 일치",
        not problems,
        "실제 이미지 목록과 출처·라이선스 표가 빠짐없이 일치합니다.",
        "실제 이미지 자료를 연구 근거로 사용할 수 없습니다: " + "; ".join(problems),
        evidence=evidence,
        blocks_human_data=False,
        # 실제 이미지 사례를 논문 범위에 넣을 때만 필요한 확장 근거다.
        # 통제 자극 C80 사례연구의 핵심 결론을 자동으로 막지는 않는다.
        blocks_research_claims=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_plan_digest(payload: dict[str, Any]) -> str:
    preimage = json.loads(json.dumps(payload, ensure_ascii=False))
    preimage.pop("plan_sha256", None)
    schedule = preimage.get("schedule")
    if isinstance(schedule, list):
        for row in schedule:
            if isinstance(row, dict):
                row.pop("validation_plan_sha256", None)
    encoded = json.dumps(
        preimage,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check_validation_plan(root: Path) -> CheckResult:
    plan_path = root / "data" / "experiment" / "validation" / "validation-plan.json"
    calibration_path = root / "results" / "calibration_model_fits.json"
    recovery_path = root / "results" / "validation_recovery.json"
    recovery_script = root / "src" / "validation_recovery.py"
    canonical_config = _canonical_experiment_config(root)
    config_hashes = {_sha256(canonical_config)} if canonical_config is not None else set()
    problems: list[str] = []
    calibration_payload: dict[str, Any] | None = None
    if not calibration_path.is_file():
        problems.append("교정 분석 artifact results/calibration_model_fits.json이 없음")
    else:
        try:
            loaded_calibration = _read_json(calibration_path)
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"교정 분석 artifact 읽기 실패 ({error})")
        else:
            if not isinstance(loaded_calibration, dict):
                problems.append("교정 분석 artifact 최상위가 객체가 아님")
            else:
                calibration_payload = loaded_calibration
                source = calibration_payload.get("source")
                source = source if isinstance(source, dict) else {}
                if calibration_payload.get("artifact_type") != "calibration_model_fits":
                    problems.append("교정 분석 artifact_type 오류")
                if calibration_payload.get("data_origin") != "human":
                    problems.append("validation 계획용 교정 분석이 실제 사람 자료가 아님")
                if source.get("config_sha256") not in config_hashes:
                    problems.append("교정 분석의 config_sha256이 원 설정 파일과 다름")
                analysis_path = root / "src" / "analyze_calibration.py"
                if (
                    not analysis_path.is_file()
                    or calibration_payload.get("analysis_script_sha256")
                    != _sha256(analysis_path)
                ):
                    problems.append("교정 분석 artifact가 현재 analyze_calibration.py와 다름")
                try:
                    verify_implementation_hashes(
                        calibration_payload.get("implementation_hashes"),
                        "calibration_analysis",
                        project_root=root,
                        label="교정 분석 artifact",
                    )
                except ImplementationProvenanceError as error:
                    problems.append(str(error))

                trial_sha = source.get("trial_file_sha256")
                trial_matches = []
                experiment_dir = root / "data" / "experiment"
                if isinstance(trial_sha, str) and SHA256_RE.fullmatch(trial_sha):
                    trial_matches = [
                        path
                        for path in experiment_dir.rglob("*.jsonl")
                        if path.is_file() and _sha256(path) == trial_sha
                    ] if experiment_dir.is_dir() else []
                if len(trial_matches) != 1:
                    problems.append("교정 분석 원 JSONL hash와 일치하는 파일이 정확히 하나가 아님")

                manifest_sha = source.get("stimulus_manifest_sha256")
                manifest_matches: list[Path] = []
                if isinstance(manifest_sha, str) and SHA256_RE.fullmatch(manifest_sha):
                    manifest_matches = [
                        path
                        for path in experiment_dir.rglob("*.json")
                        if path.is_file() and _sha256(path) == manifest_sha
                    ] if experiment_dir.is_dir() else []
                if len(manifest_matches) != 1:
                    problems.append("교정 분석 원 stimulus manifest hash와 일치하는 파일이 정확히 하나가 아님")
                else:
                    try:
                        calibration_manifest = _read_json(manifest_matches[0])
                    except (OSError, json.JSONDecodeError):
                        problems.append("교정 stimulus manifest를 읽을 수 없음")
                    else:
                        if (
                            not isinstance(calibration_manifest, dict)
                            or calibration_manifest.get("artifact_type")
                            != "compiled_pixel_stimulus_manifest"
                        ):
                            problems.append("교정 stimulus manifest artifact_type 오류")

                participants = calibration_payload.get("participants")
                if (
                    not isinstance(participants, list)
                    or len(participants) != 1
                    or not isinstance(participants[0], dict)
                    or participants[0].get("analysis_status") != "complete"
                ):
                    problems.append("교정 모델 적합이 참가자 1명의 complete 상태가 아님")
                analysis_specification = calibration_payload.get("analysis_specification")
                calibration_bootstrap_replicates = (
                    analysis_specification.get("bootstrap_replicates")
                    if isinstance(analysis_specification, dict)
                    else None
                )
                if (
                    isinstance(calibration_bootstrap_replicates, bool)
                    or not isinstance(calibration_bootstrap_replicates, int)
                    or calibration_bootstrap_replicates
                    < MINIMUM_HUMAN_BOOTSTRAP_REPLICATES
                ):
                    problems.append(
                        "사람 교정 분석 bootstrap 반복 수는 1000회 이상이어야 함"
                    )
    if not plan_path.is_file():
        problems.append("동결 validation plan이 없음")
    else:
        try:
            payload = _read_json(plan_path)
        except (OSError, json.JSONDecodeError) as error:
            payload = None
            problems.append(f"validation plan 읽기 실패 ({error})")
        if isinstance(payload, dict):
            if payload.get("artifact_type") != "validation_plan":
                problems.append("validation plan artifact_type 오류")
            if payload.get("schema_version") != "1.0":
                problems.append("validation plan schema_version이 1.0이 아님")
            if (
                not isinstance(payload.get("source_git_commit"), str)
                or not COMMIT_RE.fullmatch(payload["source_git_commit"])
            ):
                problems.append("validation plan의 source_git_commit 오류")
            if (
                calibration_payload is not None
                and payload.get("source_git_commit")
                != calibration_payload.get("source", {}).get("git_commit")
            ):
                problems.append("validation plan과 교정 분석의 자료 수집 commit이 다름")
            if payload.get("status") != "frozen_before_held_out":
                problems.append("validation plan이 held-out 응답 전에 동결된 상태가 아님")
            if payload.get("observed_held_out_used") is not False:
                problems.append("validation 수준 결정에 held-out 관측값이 사용됨")
            try:
                verify_implementation_hashes(
                    payload.get("implementation_hashes"),
                    "validation_plan",
                    project_root=root,
                    label="validation plan",
                )
            except ImplementationProvenanceError as error:
                problems.append(str(error))
            if payload.get("config_sha256") not in config_hashes:
                problems.append("validation plan의 config_sha256이 원 설정 파일과 다름")
            calibration_hash = payload.get("calibration_model_fits_sha256")
            if calibration_path.is_file() and calibration_hash != _sha256(calibration_path):
                problems.append("validation plan의 calibration artifact hash가 다름")
            plan_hash = payload.get("plan_sha256")
            if not isinstance(plan_hash, str) or not SHA256_RE.fullmatch(plan_hash):
                problems.append("validation plan 자체 hash가 없거나 형식 오류")
            elif plan_hash != _validation_plan_digest(payload):
                problems.append("validation plan 자체 hash가 canonical preimage와 다름")
            schedule = payload.get("schedule")
            if not isinstance(schedule, list) or len(schedule) != 192:
                problems.append("validation schedule이 3블록×64=192시행이 아님")
            elif any(
                not isinstance(row, dict)
                or row.get("validation_plan_sha256") != plan_hash
                for row in schedule
            ):
                problems.append("validation schedule 행이 plan_sha256에 결합되지 않음")
            if not recovery_path.is_file():
                problems.append("plan-bound validation recovery 결과가 없음")
            else:
                try:
                    recovery = _read_json(recovery_path)
                except (OSError, json.JSONDecodeError) as error:
                    recovery = None
                    problems.append(f"validation recovery 읽기 실패 ({error})")
                if isinstance(recovery, dict):
                    simulation = recovery.get("simulation", {})
                    if (
                        recovery.get("artifact_type") != "validation_parameter_recovery"
                        or recovery.get("data_origin") != "synthetic"
                        or recovery.get("research_conclusion_allowed") is not False
                        or recovery.get("passed") is not True
                        or recovery.get("must_rerun_with_frozen_validation_plan") is not False
                    ):
                        problems.append("validation recovery 상태/gate가 통과가 아님")
                    if (
                        recovery.get("plan_sha256") != plan_hash
                        or recovery.get("config_sha256") != payload.get("config_sha256")
                    ):
                        problems.append("validation recovery가 현재 plan/config hash에 묶이지 않음")
                    if (
                        not isinstance(simulation, dict)
                        or simulation.get("repetitions", 0) < 200
                        or simulation.get("bootstrap_replicates_per_block", 0) < 50
                    ):
                        problems.append("validation recovery는 200명/bootstrap 50회 이상이어야 함")
                    if (
                        recovery_script.is_file()
                        and recovery.get("implementation_sha256") != _sha256(recovery_script)
                    ):
                        problems.append("validation recovery 구현 hash가 현재 코드와 다름")
                    try:
                        verify_implementation_hashes(
                            recovery.get("implementation_hashes"),
                            "validation_recovery",
                            project_root=root,
                            label="validation recovery",
                        )
                    except ImplementationProvenanceError as error:
                        problems.append(str(error))
                elif recovery is not None:
                    problems.append("validation recovery 최상위가 객체가 아님")
        elif payload is not None:
            problems.append("validation plan 최상위가 객체가 아님")

    return _check(
        "validation_plan_ready",
        "교정 후 validation plan 동결",
        not problems,
        "원 config와 교정 artifact에 결합된 3×64 validation plan이 held-out 전에 동결됐습니다.",
        "validation 자료 수집을 시작할 수 없습니다: " + "; ".join(problems),
        evidence=[
            _relative(calibration_path, root),
            _relative(plan_path, root),
            _relative(recovery_path, root),
        ],
        blocks_human_data=False,
        blocks_research_claims=True,
    )


def _repo_path(root: Path, value: str) -> Path | None:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _csv_image_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return set()
        image_field = next(
            (name for name in ("image", "filename", "source_image") if name in reader.fieldnames),
            None,
        )
        if image_field is None:
            return set()
        return {
            Path(row[image_field]).stem
            for row in reader
            if row.get(image_field) and row[image_field].strip()
        }


def _check_maps_provenance(root: Path) -> CheckResult:
    maps_dir = root / "results" / "maps"
    regions_path = maps_dir / "map_regions.csv"
    pairs_path = maps_dir / "map_pairs.csv"
    manifest_candidates = [maps_dir / "provenance.json", maps_dir / "manifest.json"]
    manifest_path = next((path for path in manifest_candidates if path.is_file()), None)
    problems: list[str] = []
    for output in (regions_path, pairs_path):
        if not output.is_file():
            problems.append(f"{_relative(output, root)}가 없음")

    referenced_ids: set[str] = set()
    try:
        referenced_ids |= _csv_image_ids(regions_path)
        referenced_ids |= _csv_image_ids(pairs_path)
    except (OSError, csv.Error) as error:
        problems.append(f"지도 결과 CSV 읽기 실패 ({error})")
    if not referenced_ids:
        problems.append("지도 결과가 참조한 입력 이미지 ID를 확인할 수 없음")

    manifest: dict[str, Any] = {}
    if manifest_path is None:
        problems.append("results/maps/provenance.json이 없음")
    else:
        try:
            loaded = _read_json(manifest_path)
            if isinstance(loaded, dict):
                manifest = loaded
            else:
                problems.append("provenance 최상위 값이 객체가 아님")
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"provenance 읽기 실패 ({error})")

    input_ids: set[str] = set()
    input_files = manifest.get("input_files")
    if manifest and (not isinstance(input_files, list) or not input_files):
        problems.append("provenance.input_files가 없음")
    elif isinstance(input_files, list):
        for index, entry in enumerate(input_files):
            if not isinstance(entry, dict):
                problems.append(f"input_files[{index}]가 path/hash 객체가 아님")
                continue
            value = entry.get("path")
            expected_hash = entry.get("sha256")
            if not isinstance(value, str) or not value:
                problems.append(f"input_files[{index}].path가 없음")
                continue
            path = _repo_path(root, value)
            if path is None:
                problems.append(f"input_files[{index}]가 저장소 밖을 가리킴")
                continue
            input_ids.add(path.stem)
            if not path.is_file():
                problems.append(f"provenance 입력 파일이 없음: {value}")
            elif not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
                problems.append(f"provenance 입력 hash가 없거나 형식 오류: {value}")
            elif _sha256(path) != expected_hash.lower():
                problems.append(f"provenance 입력 hash 불일치: {value}")

    if referenced_ids and input_ids and referenced_ids != input_ids:
        missing = sorted(referenced_ids - input_ids)
        extra = sorted(input_ids - referenced_ids)
        if missing:
            problems.append("provenance에서 빠진 지도 입력: " + ", ".join(missing))
        if extra:
            problems.append("결과에서 참조하지 않은 provenance 입력: " + ", ".join(extra))

    if manifest:
        script_value = manifest.get("analysis_script")
        script_hash = manifest.get("analysis_script_sha256")
        script_path = _repo_path(root, script_value) if isinstance(script_value, str) else None
        if script_path is None or not script_path.is_file():
            problems.append("provenance.analysis_script가 없거나 존재하지 않음")
        elif not isinstance(script_hash, str) or not SHA256_RE.fullmatch(script_hash):
            problems.append("analysis_script_sha256가 없거나 형식 오류")
        elif _sha256(script_path) != script_hash.lower():
            problems.append("analysis_script_sha256가 현재 코드와 다름")
        if not COMMIT_RE.fullmatch(str(manifest.get("git_commit", ""))):
            problems.append("provenance.git_commit이 전체 40자리 commit이 아님")
        if not isinstance(manifest.get("parameters"), dict) or not manifest["parameters"]:
            problems.append("provenance.parameters가 비어 있음")
        if not str(manifest.get("generated_at", "")).strip():
            problems.append("provenance.generated_at이 없음")

    evidence = [f"참조 이미지 ID: {', '.join(sorted(referenced_ids)) or '없음'}"]
    if manifest_path:
        evidence.append(_relative(manifest_path, root))
    return _check(
        "maps_provenance",
        "지도 분석 입력·코드 provenance",
        not problems,
        "지도 결과의 모든 입력, 분석 코드, hash, commit과 파라미터를 추적할 수 있습니다.",
        "지도 분석 결과를 재현할 provenance가 부족합니다: " + "; ".join(problems),
        evidence=evidence,
        blocks_human_data=False,
        # 지도 분석은 현재 제목의 필수 종속변수가 아닌 선택적 응용 사례다.
        blocks_research_claims=False,
    )


def _source_files(root: Path, token: str) -> list[Path]:
    source_dir = root / "src"
    if not source_dir.exists():
        return []
    return sorted(
        path
        for path in source_dir.glob(f"*{token}*.py")
        if not path.name.startswith("test_") and path.name != "research_readiness.py"
    )


def _test_files(root: Path, token: str) -> list[Path]:
    source_dir = root / "src"
    return sorted(source_dir.glob(f"test_*{token}*.py")) if source_dir.exists() else []


def _check_psychometric_implementation(root: Path) -> CheckResult:
    implementations = _source_files(root, "psychometric")
    tests = _test_files(root, "psychometric")
    problems: list[str] = []
    if not implementations:
        problems.append("psychometric 적합 코드가 없음")
    if not tests:
        problems.append("psychometric 단위/회귀 테스트가 없음")
    evidence = [_relative(path, root) for path in implementations + tests]
    return _check(
        "psychometric_implementation",
        "C80 심리측정 구현",
        not problems,
        "C80 적합 코드와 자동 테스트가 모두 있습니다.",
        "C80을 재현 가능하게 계산할 준비가 안 됐습니다: " + "; ".join(problems),
        evidence=evidence,
        blocks_human_data=True,
        blocks_research_claims=True,
    )


def _explicit_recovery_pass(value: Any) -> bool:
    """전체 결과의 명시적 통과만 인정한다.

    하위 시나리오 하나의 ``passed=true``가 전체 실패를 가리는 일을 막기 위해
    최상위 또는 최상위 ``summary``의 판정만 읽는다.
    """

    if not isinstance(value, dict):
        return False
    containers = [value]
    if isinstance(value.get("summary"), dict):
        containers.append(value["summary"])
    for container in containers:
        for key in ("passed", "recovery_passed", "ready_for_human_data"):
            if container.get(key) is True:
                return True
        if str(container.get("status", "")).lower() in {
            "pass",
            "passed",
            "complete",
            "completed",
        }:
            return True
    return False


def _valid_parameter_recovery_integrity(
    payload: dict[str, Any],
    *,
    simulation_replicates: int,
    bootstrap_replicates: int,
    require_coverage_pass: bool = True,
) -> bool:
    """새 recovery artifact의 stress/coverage/paired gates를 다시 검사한다."""
    if (
        payload.get("artifact_type") != "synthetic_parameter_recovery"
        or payload.get("data_origin") != "synthetic"
        or payload.get("research_conclusion_allowed") is not False
        or (require_coverage_pass and payload.get("passed") is not True)
        or (require_coverage_pass and payload.get("reasons") != [])
    ):
        return False
    integrity = payload.get("recovery_integrity")
    required_integrity = {
        "minimum_simulation_replicates_met",
        "minimum_bootstrap_replicates_met",
        "all_profile_gates_passed",
        "representative_bootstrap_passed",
        "repeated_coverage_passed",
        "passed",
    }
    if not require_coverage_pass:
        required_integrity -= {"repeated_coverage_passed", "passed"}
        reasons=payload.get("reasons")
        if (payload.get("passed") is not False or not isinstance(reasons,list)
                or not reasons or any(not isinstance(r,str) or not r.endswith(
                    ": repeated coverage guard failed") for r in reasons)):
            return False
    if not isinstance(integrity, dict) or any(
        integrity.get(key) is not True for key in required_integrity
    ):
        return False

    monte_carlo = payload.get("monte_carlo")
    if not isinstance(monte_carlo, dict):
        return False
    if monte_carlo.get("simulation_replicates") != simulation_replicates:
        return False
    gate = monte_carlo.get("worst_case_and_upper_quantile_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True or gate.get("reasons") != []:
        return False
    expected_profiles = {"low-steep", "central", "high-shallow"}
    profile_counts = monte_carlo.get("profile_counts")
    profile_results = monte_carlo.get("profile_results")
    if (
        not isinstance(profile_counts, dict)
        or set(profile_counts) != expected_profiles
        or not all(isinstance(value, int) and value > 0 for value in profile_counts.values())
        or sum(profile_counts.values()) != simulation_replicates
        or not isinstance(profile_results, dict)
        or set(profile_results) != expected_profiles
    ):
        return False
    expected_models = {"B192", "B384", "C192", "D384"}
    for name in expected_profiles:
        result = profile_results[name]
        if (
            not isinstance(result, dict)
            or result.get("passed") is not True
            or result.get("reasons") != []
            or result.get("simulation_replicates") != profile_counts[name]
            or not isinstance(result.get("models"), dict)
            or set(result["models"]) != expected_models
        ):
            return False

    representative = payload.get("representative_bootstrap")
    if (
        not isinstance(representative, dict)
        or representative.get("passed") is not True
        or representative.get("reasons") != []
        or not isinstance(representative.get("models"), dict)
        or set(representative["models"]) != expected_models
    ):
        return False
    for model, diagnostic in representative["models"].items():
        coverage = diagnostic.get("truth_coverage") if isinstance(diagnostic, dict) else None
        required = coverage.get("required_parameters") if isinstance(coverage, dict) else None
        expected_parameters = (
            {"c80_ref", "beta"}
            if model == "D384"
            else {"c80_ref"}
            if model == "C192"
            else {"c80"}
        )
        if (
            not isinstance(diagnostic, dict)
            or diagnostic.get("passed") is not True
            or diagnostic.get("converged") is not True
            or diagnostic.get("requested_replicates") != bootstrap_replicates
            or not isinstance(diagnostic.get("successful_replicates"), int)
            or diagnostic["successful_replicates"] < math.ceil(0.80 * bootstrap_replicates)
            or not isinstance(coverage, dict)
            or coverage.get("diagnostic_only") is not True
            or not isinstance(coverage.get("all_required_truth_covered"), bool)
            or not isinstance(required, dict)
            or set(required) != expected_parameters
            or any(not isinstance(value, bool) for value in required.values())
        ):
            return False
    paired = representative.get("paired_reference_integrity")
    checks = paired.get("checks") if isinstance(paired, dict) else None
    if (
        not isinstance(paired, dict)
        or paired.get("passed") is not True
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        return False
    coverage=payload.get("repeated_coverage")
    if require_coverage_pass:
        return validate_coverage_payload(coverage)
    if not isinstance(coverage,dict):return False
    rows=coverage.get("outer_records");count=coverage.get("bootstrap_per_outer")
    if (not isinstance(rows,list) or len(rows)<60 or not isinstance(count,int)
            or isinstance(count,bool) or count<200):return False
    for i,row in enumerate(rows):
        if (not isinstance(row,dict) or row.get("outer_index")!=i
                or row.get("profile")!=PROFILES[i%3]
                or not isinstance(row.get("models"),dict)
                or set(row["models"])!=set(MODELS)):return False
        for model,entry in row["models"].items():
            keys={"c80","beta"} if model=="D384" else {"c80"}
            if (not isinstance(entry,dict) or not isinstance(entry.get("fit_converged"),bool)
                    or not isinstance(entry.get("bootstrap_successful"),int)
                    or isinstance(entry.get("bootstrap_successful"),bool)
                    or not 0<=entry["bootstrap_successful"]<=count
                    or not isinstance(entry.get("covered"),dict)
                    or set(entry["covered"])!=keys
                    or any(not isinstance(v,bool) for v in entry["covered"].values())):return False
    summary=summarise_coverage(rows,count)
    return (summary["passed"] is False
            and all(coverage.get(k)==v for k,v in summary.items())
            and payload["reasons"]==summary["reasons"])


def _collection_deviation_binding(root: Path, recovery_path: Path) -> str | None:
    """Explicit local operator decision; never turns failed recovery into passed."""
    record_path=root/"data/experiment/private/calibration-collection-deviation.json"
    config=_canonical_experiment_config(root)
    try:
        record=_read_json(record_path)
        if (config is None or not isinstance(record,dict)
                or record.get("artifact_type")!="calibration_collection_deviation_v1"
                or record.get("decision")!="collect_calibration_only_defer_inference"
                or any(record.get(k) is not True for k in (
                    "approved","required_consent_confirmed","technical_checks_not_waived"))
                or not isinstance(record.get("recorded_at_utc"),str)
                or not record["recorded_at_utc"].endswith("Z")
                or record.get("source_recovery_sha256")!=_sha256(recovery_path)
                or record.get("config_sha256")!=_sha256(config)):
            return None
        return record["config_sha256"]
    except (OSError,ValueError,TypeError):return None


def _check_parameter_recovery(root: Path) -> CheckResult:
    implementations = sorted(
        set(_source_files(root, "parameter_recovery"))
        | set(_source_files(root, "recovery"))
    )
    result_dir = root / "results"
    result_files = sorted(result_dir.rglob("*parameter*recovery*.json")) if result_dir.exists() else []
    if result_dir.exists():
        result_files += [
            path
            for path in result_dir.rglob("*recovery*.json")
            if path not in result_files
        ]
    problems: list[str] = []
    if not implementations:
        problems.append("parameter recovery 실행 코드가 없음")

    # 합성 후보 수준으로 통과한 recovery를 동결된 실제 config의 recovery로
    # 오인하지 않는다. 각 1.0 config의 원본 파일 hash와 12개 수준이 결과에
    # 정확히 결합되어 있어야 한다.
    config_bindings: dict[str, dict[str, Any]] = {}
    canonical_config = _canonical_experiment_config(root)
    for config_path in ([canonical_config] if canonical_config is not None else []):
        try:
            config = _read_json(config_path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(config, dict) or config.get("schema_version") != "1.0":
            continue
        levels = config.get("sampling", {}).get("delta_e00_levels")
        if isinstance(levels, list) and len(levels) == 12:
            try:
                widths = config.get("calibration", {}).get("widths")
                if not isinstance(widths, list) or len(widths) != 3:
                    continue
                widths_deg = tuple(float(row["width_deg"]) for row in widths)
                config_bindings[_sha256(config_path)] = {
                    "levels": tuple(float(value) for value in levels),
                    "widths_deg": widths_deg,
                    "reference_width_deg": widths_deg[1],
                    "trials_per_width": tuple(config["calibration"]["trials_per_width"]),
                    "fixed_lapse": float(config["observer_profile"]["fixed_lapse"]),
                    "direction_lab": tuple(
                        float(value) for value in config["color_design"]["direction_lab"]
                    ),
                    "source_probability_coordinate": float(
                        config["model_plan"]["szafir_probability_coordinate"]
                    ),
                }
            except (TypeError, ValueError):
                continue
    if not config_bindings:
        problems.append("동결된 1.0 experiment_config와 결합할 recovery 대상이 없음")

    passed_results: list[Path] = []
    deferred_results: list[Path] = []
    matched_config_hashes: set[str] = set()
    for path in result_files:
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        deviation_hash=_collection_deviation_binding(root,path)
        if (
            not isinstance(payload, dict)
            or _placeholder_paths(payload)
            or (not _explicit_recovery_pass(payload) and deviation_hash is None)
        ):
            continue
        plan = payload.get("sampling_plan") if isinstance(payload, dict) else None
        if not isinstance(plan, dict):
            continue
        if plan.get("must_rerun_with_frozen_human_config") is not False and deviation_hash is None:
            continue
        result_hash = deviation_hash or payload.get("config_sha256")
        legacy_hash = payload.get("experiment_config_sha256")
        if deviation_hash is None and legacy_hash is not None and legacy_hash != result_hash:
            continue
        result_levels = plan.get("delta_e00_levels")
        expected = config_bindings.get(result_hash)
        try:
            normalised_levels = tuple(float(value) for value in result_levels)
        except (TypeError, ValueError):
            continue
        if expected is None or normalised_levels != expected["levels"]:
            continue
        try:
            result_widths = tuple(float(value) for value in payload["widths_deg"])
            result_reference = float(payload["reference_width_deg"])
            result_trials = tuple(int(value) for value in payload["trials_per_width"])
            result_simulations = int(payload["simulation_replicates"])
            result_bootstrap = int(payload["bootstrap_replicates"])
            result_lapse = float(payload["fixed_lapse"])
            szafir = payload["szafir_relative_scenario"]
            result_direction = tuple(float(value) for value in szafir["direction_lab"])
            result_probability = float(szafir["source_probability_coordinate"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            result_widths != expected["widths_deg"]
            or result_reference != expected["reference_width_deg"]
            or result_trials != expected["trials_per_width"]
            or result_simulations < 200
            or result_bootstrap < 200
            or result_lapse != expected["fixed_lapse"]
            or result_direction != expected["direction_lab"]
            or result_probability != expected["source_probability_coordinate"]
            or payload.get("all_fits_converged") is not True
        ):
            continue
        implementation = payload.get("implementation_hashes")
        legacy_implementation = payload.get("implementation_sha256")
        if legacy_implementation != implementation:
            continue
        try:
            verify_implementation_hashes(
                implementation,
                "parameter_recovery",
                project_root=root,
                label="parameter recovery",
            )
        except ImplementationProvenanceError:
            continue
        if not _valid_parameter_recovery_integrity(
            payload,
            simulation_replicates=result_simulations,
            bootstrap_replicates=result_bootstrap,
            require_coverage_pass=deviation_hash is None,
        ):
            continue
        matched_config_hashes.add(result_hash)
        (deferred_results if deviation_hash else passed_results).append(path)
    missing_bindings = sorted(set(config_bindings) - matched_config_hashes)
    if missing_bindings:
        problems.append(
            "동결 config hash/12개 수준과 일치하고 재실행 완료로 표시된 recovery가 없음: "
            + ", ".join(value[:12] for value in missing_bindings)
        )
    evidence = [_relative(path, root) for path in implementations + result_files]
    conditional=not problems and bool(deferred_results) and not passed_results
    return _check(
        "parameter_recovery",
        "가상 참가자 parameter recovery",
        not problems and not conditional,
        "parameter recovery 코드와 통과한 결과가 있습니다.",
        ("신뢰구간 검증 미달을 기록하고 교정자료 수집만 승인했습니다. validation/확정 결론은 차단됩니다."
         if conditional else "본 시행 전에 필요한 parameter recovery가 끝나지 않았습니다: " + "; ".join(problems)),
        evidence=evidence,
        blocks_human_data=not conditional,
        blocks_research_claims=True,
    )


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit_run = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if commit_run.returncode != 0:
            return {"commit": None, "dirty": None, "error": commit_run.stderr.strip()}
        commit = commit_run.stdout.strip()
        status_run = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if status_run.returncode != 0:
            return {"commit": commit, "dirty": None, "error": status_run.stderr.strip()}
        return {
            "commit": commit,
            "dirty": bool(status_run.stdout.strip()),
            "error": None,
        }
    except OSError as error:
        return {"commit": None, "dirty": None, "error": str(error)}


def _check_git_commit(root: Path, state: dict[str, Any]) -> list[CheckResult]:
    commit = state.get("commit")
    valid_commit = isinstance(commit, str) and COMMIT_RE.fullmatch(commit) is not None
    commit_result = _check(
        "git_commit",
        "재현 가능한 Git commit",
        valid_commit,
        f"현재 commit을 기록할 수 있습니다: {commit}",
        "유효한 40자리 Git commit을 확인하지 못했습니다. 결과와 코드를 같은 버전으로 묶을 수 없습니다.",
        evidence=[str(state.get("error") or commit or "commit 없음")],
        blocks_human_data=True,
        blocks_research_claims=True,
    )
    clean = valid_commit and state.get("dirty") is False
    clean_result = _check(
        "git_clean_worktree",
        "동결 코드의 변경 여부",
        clean,
        "작업 폴더가 깨끗해 결과가 commit과 정확히 대응합니다.",
        "commit 이후 변경 사항이 남아 있거나 작업 폴더 상태를 확인하지 못했습니다. 실험 전 코드를 commit해야 합니다.",
        evidence=[f"dirty={state.get('dirty')!r}"],
        blocks_human_data=True,
        blocks_research_claims=True,
    )
    return [commit_result, clean_result]


def _legacy_marked(text: str) -> bool:
    heading = text[:800].lower()
    return any(token in heading for token in ("legacy", "archived", "이전안", "폐기", "과거안"))


def _has_actionable_062_rule(text: str) -> bool:
    pattern = re.compile(
        r"(?:환산\s*비율.{0,100}0\.62|0\.62.{0,100}(?:곱|환산))",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        context = text[max(0, match.start() - 100) : match.end() + 120]
        if not any(
            token in context
            for token in (
                "사용하지",
                "쓰지 않",
                "금지",
                "아니다",
                "폐기",
                "제거",
                "대체하지",
                "범위가 아니다",
            )
        ):
            return True
    return False


def _check_research_identity(root: Path) -> CheckResult:
    required = [
        root / "README.md",
        root / "docs" / "experiment-protocol.md",
        root / "docs" / "project-context.txt",
        root / "docs" / "research-plan.txt",
    ]
    texts: dict[Path, str] = {}
    problems: list[str] = []
    for path in required:
        if not path.is_file():
            problems.append(f"{_relative(path, root)}가 없음")
            continue
        try:
            texts[path] = path.read_text(encoding="utf-8-sig")
        except OSError as error:
            problems.append(f"{_relative(path, root)} 읽기 실패 ({error})")

    current_text = "\n".join(
        texts.get(path, "")
        for path in (root / "README.md", root / "docs" / "experiment-protocol.md")
    )
    current_line_study = "선 굵기" in current_text and "C80" in current_text
    if not current_line_study:
        problems.append("README와 실험 프로토콜에서 현재 선 굵기 C80 연구 정체성을 확인할 수 없음")

    old_identity = "개인화 선택적 AI 색보정"
    for path in (root / "docs" / "project-context.txt", root / "docs" / "research-plan.txt"):
        text = texts.get(path, "")
        if current_line_study and old_identity in text and not _legacy_marked(text):
            problems.append(
                f"{_relative(path, root)}가 이전 AI 연구 제목을 현재안처럼 유지함"
            )
        mandatory_ai = any(
            phrase in text
            for phrase in (
                "AI 기반 색보정",
                "AI 색보정 모델로 확장",
                "Phase 6. AI",
                "U-Net/Encoder-Decoder AI 색보정 모델",
            )
        )
        if current_line_study and mandatory_ai and not _legacy_marked(text):
            problems.append(
                f"{_relative(path, root)}가 AI 단계를 필수 연구 범위처럼 서술함"
            )

    conditions = root / "results" / "conditions.md"
    if conditions.is_file():
        text = conditions.read_text(encoding="utf-8-sig")
        actionable_062 = _has_actionable_062_rule(text)
        if actionable_062 and not _legacy_marked(text):
            problems.append(
                "results/conditions.md가 0.62 환산 규칙을 현재 절차처럼 남겨 두었음"
            )

    docs_dir = root / "docs"
    if docs_dir.is_dir():
        for path in sorted(
            item
            for item in docs_dir.rglob("*")
            if item.is_file() and item.suffix.lower() in {".md", ".txt"}
        ):
            try:
                text = path.read_text(encoding="utf-8-sig")
            except OSError as error:
                problems.append(f"{_relative(path, root)} 읽기 실패 ({error})")
                continue
            if _legacy_marked(text):
                continue
            if _has_actionable_062_rule(text):
                problems.append(
                    f"{_relative(path, root)}가 미표시 0.62 환산 절차를 현재 지시처럼 남김"
                )
            if any(
                phrase in text
                for phrase in (
                    "AI 기반 색보정",
                    "AI 색보정 모델로 확장",
                    "Phase 6. AI",
                    "U-Net/Encoder-Decoder AI 색보정 모델",
                )
            ):
                problems.append(
                    f"{_relative(path, root)}가 미표시 AI 단계를 필수 범위처럼 남김"
                )

    problems = list(dict.fromkeys(problems))

    return _check(
        "research_identity",
        "핵심 문서의 연구 정체성 일치",
        not problems,
        "핵심 문서가 선 굵기 C80 사례연구라는 같은 범위와 용어를 사용합니다.",
        "핵심 문서가 서로 다른 연구를 지시합니다: " + "; ".join(problems),
        evidence=[_relative(path, root) for path in required if path.is_file()],
        blocks_human_data=True,
        blocks_research_claims=True,
    )


def assess_research_readiness(
    root: str | Path,
    *,
    minimum_real_images: int = DEFAULT_MINIMUM_REAL_IMAGES,
) -> dict[str, Any]:
    """저장소의 준비 상태를 검사해 JSON 직렬화 가능한 보고서를 반환한다."""

    project_root = Path(root).resolve()
    if minimum_real_images < 1:
        raise ValueError("minimum_real_images는 1 이상이어야 합니다.")
    git_state = _git_state(project_root)
    checks = [
        _check_experiment_config(project_root),
        _check_experiment_runtime(project_root),
        _check_validation_runtime(project_root),
        _check_contract_examples(project_root),
        _check_human_contract_schemas(project_root),
        _check_participant_thresholds(project_root),
        _check_validation_plan(project_root),
        _check_real_images_and_sources(project_root, minimum_real_images),
        _check_maps_provenance(project_root),
        _check_psychometric_implementation(project_root),
        _check_parameter_recovery(project_root),
        *_check_git_commit(project_root, git_state),
        _check_research_identity(project_root),
    ]

    calibration_blocked = [
        item
        for item in checks
        if item.status == "blocked" and item.blocks_human_data
    ]
    validation_only_ids = {"validation_plan_ready", "validation_runtime_gates", "parameter_recovery"}
    validation_blocked = calibration_blocked + [
        item
        for item in checks
        if item.status == "blocked"
        and item.check_id in validation_only_ids
        and item not in calibration_blocked
    ]
    claims_blocked = [
        item
        for item in checks
        if item.status == "blocked" and item.blocks_research_claims
    ]
    report = {
        "tool": "research_readiness",
        "project_root": str(project_root),
        "git_commit": git_state.get("commit"),
        "ready_for_calibration_collection": not calibration_blocked,
        "ready_for_validation_collection": not validation_blocked,
        # 이전 소비자 호환 별칭. validation plan은 calibration 자료 뒤에만
        # 만들 수 있으므로 이 값은 calibration 시작 가능성을 뜻한다.
        "ready_for_human_data": not calibration_blocked,
        "ready_for_research_claims": not claims_blocked,
        "summary": {
            "passed": sum(item.status == "pass" for item in checks),
            "blocked": sum(item.status == "blocked" for item in checks),
            "total": len(checks),
        },
        "blocked": {
            "calibration_collection": [
                {"check_id": item.check_id, "message": item.message}
                for item in calibration_blocked
            ],
            "validation_collection": [
                {"check_id": item.check_id, "message": item.message}
                for item in validation_blocked
            ],
            "human_data": [
                {"check_id": item.check_id, "message": item.message}
                for item in calibration_blocked
            ],
            "research_claims": [
                {"check_id": item.check_id, "message": item.message}
                for item in claims_blocked
            ],
        },
        "checks": [asdict(item) for item in checks],
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """검사 보고서를 사람이 읽기 쉬운 Markdown으로 렌더링한다."""

    calibration = "가능" if report["ready_for_calibration_collection"] else "차단"
    validation = "가능" if report["ready_for_validation_collection"] else "차단"
    claims = "가능" if report["ready_for_research_claims"] else "차단"
    lines = [
        "# 연구 준비 상태 검사",
        "",
        f"- 교정 자료 수집: **{calibration}**",
        f"- validation 자료 수집: **{validation}**",
        f"- 사람 자료 수집: **{calibration}** (호환 별칭은 교정 단계 기준)",
        f"- 연구 결론 보고: **{claims}**",
        f"- Git commit: `{report.get('git_commit') or '확인 불가'}`",
        f"- 검사 결과: 통과 {report['summary']['passed']} / 차단 {report['summary']['blocked']}",
        "",
    ]
    for key, title in (
        ("calibration_collection", "교정 자료 수집을 막는 항목"),
        ("validation_collection", "validation 자료 수집을 막는 항목"),
        ("research_claims", "연구 결론을 막는 항목"),
    ):
        lines.extend([f"## {title}", ""])
        blocked = report["blocked"][key]
        if blocked:
            lines.extend(
                f"- `{item['check_id']}` — {item['message']}" for item in blocked
            )
        else:
            lines.append("- 없음")
        lines.append("")

    lines.extend(
        [
            "## 전체 검사",
            "",
            "| 상태 | 검사 | 사람 자료 기준 | 연구 결론 기준 | 설명 |",
            "|---|---|---:|---:|---|",
        ]
    )
    for item in report["checks"]:
        state = "통과" if item["status"] == "pass" else "차단"
        human_scope = "필수" if item["blocks_human_data"] else "-"
        claims_scope = "필수" if item["blocks_research_claims"] else "-"
        message = str(item["message"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {state} | `{item['check_id']}` {item['title']} | "
            f"{human_scope} | {claims_scope} | {message} |"
        )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="사람 대상 실험과 연구 결론의 준비 상태를 별도로 검사합니다."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="저장소 루트(기본값: 이 스크립트의 상위 저장소)",
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="markdown",
        help="출력 형식",
    )
    parser.add_argument("--output", help="지정하면 보고서를 UTF-8 파일로 저장합니다.")
    parser.add_argument(
        "--minimum-real-images",
        type=int,
        default=DEFAULT_MINIMUM_REAL_IMAGES,
        help=f"실제 자료 최소 이미지 수(기본 {DEFAULT_MINIMUM_REAL_IMAGES})",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="차단 항목이 있으면 종료 코드 2를 반환합니다.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    args = _parser().parse_args(argv)
    report = assess_research_readiness(
        args.root, minimum_real_images=args.minimum_real_images
    )
    rendered = (
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.format == "json"
        else render_markdown(report)
    )
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.strict_exit and (
        not report["ready_for_human_data"]
        or not report["ready_for_research_claims"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
