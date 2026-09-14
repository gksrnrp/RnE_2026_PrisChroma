"""Fail-closed provenance for research analysis implementations.

Research artifacts keep a project-relative ``implementation_hashes`` mapping rather
than only the hash of the command-line wrapper.  The named sets below are deliberately
explicit: changing a numerical dependency, an input/output contract, or the pinned
Python environment invalidates the artifact until that stage is rerun.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Keep paths project-relative and sorted.  These are not inferred from imports because
# dynamic/import-time dependencies and JSON contracts must be visible to reviewers.
CALIBRATION_ANALYSIS_FILES = (
    "requirements-lock.txt",
    "schemas/calibration_model_fits.schema.json",
    "schemas/human/experiment_config.schema.json",
    "schemas/human/trial.schema.json",
    "src/analyze_calibration.py",
    "src/color_spaces.py",
    "src/data_contracts.py",
    "src/distance.py",
    "src/experiment_design.py",
    "src/experiment_runner.py",
    "src/implementation_provenance.py",
    "src/machado.py",
    "src/psychometrics.py",
    "src/stimulus_design.py",
    "src/stimulus_renderer.py",
    "src/validation_psychometrics.py",
)

VALIDATION_PLAN_FILES = tuple(sorted(set(CALIBRATION_ANALYSIS_FILES) | {
    "src/validation_design.py",
}))

VALIDATION_ANALYSIS_FILES = tuple(sorted(set(VALIDATION_PLAN_FILES) | {
    "schemas/human/participant_thresholds.schema.json",
    "src/analyze_validation.py",
    "src/validation_psychometrics.py",
}))

VALIDATION_RECOVERY_FILES = (
    "requirements-lock.txt",
    "schemas/human/experiment_config.schema.json",
    "src/implementation_provenance.py",
    "src/psychometrics.py",
    "src/validation_design.py",
    "src/validation_psychometrics.py",
    "src/validation_recovery.py",
)

PARAMETER_RECOVERY_FILES = tuple(sorted(set(CALIBRATION_ANALYSIS_FILES) | {
    "src/parameter_recovery.py",
}))

PATH_CORRECTION_ANALYSIS_FILES = tuple(sorted(
    set(VALIDATION_ANALYSIS_FILES)
    | {
        # Synthetic development mode reaches this draft contract through
        # threshold_adapter -> data_contracts.  Human mode uses the schemas in
        # VALIDATION_ANALYSIS_FILES.
        "schemas/participant_thresholds.schema.json",
        "src/config.py",
        "src/path_correction.py",
        "src/run_path_correction_analysis.py",
        "src/threshold_adapter.py",
    }
))

IMPLEMENTATION_FILES = {
    "calibration_analysis": CALIBRATION_ANALYSIS_FILES,
    "parameter_recovery": PARAMETER_RECOVERY_FILES,
    "path_correction_analysis": PATH_CORRECTION_ANALYSIS_FILES,
    "validation_plan": VALIDATION_PLAN_FILES,
    "validation_analysis": VALIDATION_ANALYSIS_FILES,
    "validation_recovery": VALIDATION_RECOVERY_FILES,
}


class ImplementationProvenanceError(ValueError):
    """The recorded implementation set is missing, malformed, or stale."""


def _normalise_relative_path(value: str) -> str:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ImplementationProvenanceError(
            f"implementation path must stay inside the project: {value!r}"
        )
    normalised = relative.as_posix().lstrip("./")
    if not normalised:
        raise ImplementationProvenanceError("implementation path cannot be empty")
    return normalised


def implementation_hashes(
    scope: str,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, str]:
    """Return deterministic SHA-256 hashes for the declared implementation set."""
    try:
        declared = IMPLEMENTATION_FILES[scope]
    except KeyError as error:
        raise ImplementationProvenanceError(
            f"unknown implementation provenance scope: {scope!r}"
        ) from error
    root = Path(project_root).resolve()
    result: dict[str, str] = {}
    for value in declared:
        relative = _normalise_relative_path(value)
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
            raw = target.read_bytes()
        except ValueError as error:
            raise ImplementationProvenanceError(
                f"implementation path escaped the project: {relative}"
            ) from error
        except FileNotFoundError as error:
            raise ImplementationProvenanceError(
                f"required implementation file is missing: {relative}"
            ) from error
        except OSError as error:
            raise ImplementationProvenanceError(
                f"cannot read implementation file {relative}: {error}"
            ) from error
        result[relative] = hashlib.sha256(raw).hexdigest()
    return dict(sorted(result.items()))


def verify_implementation_hashes(
    recorded: object,
    scope: str,
    *,
    project_root: str | Path = PROJECT_ROOT,
    label: str = "artifact",
) -> None:
    """Fail unless ``recorded`` exactly matches the current declared file set."""
    if not isinstance(recorded, Mapping):
        raise ImplementationProvenanceError(
            f"{label}: implementation_hashes mapping is required"
        )
    malformed = [
        key
        for key, value in recorded.items()
        if not isinstance(key, str)
        or not isinstance(value, str)
        or SHA256_RE.fullmatch(value) is None
    ]
    if malformed:
        raise ImplementationProvenanceError(
            f"{label}: malformed implementation_hashes entries: {malformed!r}"
        )
    current = implementation_hashes(scope, project_root=project_root)
    recorded_keys = set(recorded)
    current_keys = set(current)
    missing = sorted(current_keys - recorded_keys)
    unexpected = sorted(recorded_keys - current_keys)
    mismatched = sorted(
        key for key in current_keys & recorded_keys if recorded[key] != current[key]
    )
    if missing or unexpected or mismatched:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        if mismatched:
            details.append("changed=" + ",".join(mismatched))
        raise ImplementationProvenanceError(
            f"{label}: implementation provenance mismatch ({'; '.join(details)})"
        )


def declared_implementation_files(scope: str) -> Sequence[str]:
    """Expose the immutable declared set for tests and audit reporting."""
    try:
        return IMPLEMENTATION_FILES[scope]
    except KeyError as error:
        raise ImplementationProvenanceError(
            f"unknown implementation provenance scope: {scope!r}"
        ) from error


__all__ = [
    "CALIBRATION_ANALYSIS_FILES",
    "IMPLEMENTATION_FILES",
    "ImplementationProvenanceError",
    "PARAMETER_RECOVERY_FILES",
    "PATH_CORRECTION_ANALYSIS_FILES",
    "VALIDATION_ANALYSIS_FILES",
    "VALIDATION_PLAN_FILES",
    "VALIDATION_RECOVERY_FILES",
    "declared_implementation_files",
    "implementation_hashes",
    "verify_implementation_hashes",
]
