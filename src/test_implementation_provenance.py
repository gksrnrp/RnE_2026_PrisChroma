from __future__ import annotations

import unittest

from implementation_provenance import (
    IMPLEMENTATION_FILES,
    ImplementationProvenanceError,
    VALIDATION_ANALYSIS_FILES,
    implementation_hashes,
    verify_implementation_hashes,
)


class ImplementationProvenanceTests(unittest.TestCase):
    def test_declared_scopes_cover_required_numerical_and_contract_files(self) -> None:
        required = {
            "calibration_analysis": {
                "src/analyze_calibration.py",
                "src/psychometrics.py",
                "src/validation_psychometrics.py",
                "schemas/calibration_model_fits.schema.json",
                "requirements-lock.txt",
            },
            "validation_plan": {
                "src/validation_design.py",
                "src/psychometrics.py",
                "schemas/calibration_model_fits.schema.json",
                "requirements-lock.txt",
            },
            "validation_analysis": {
                "src/analyze_validation.py",
                "src/validation_psychometrics.py",
                "src/psychometrics.py",
                "schemas/human/participant_thresholds.schema.json",
                "requirements-lock.txt",
            },
            "validation_recovery": {
                "src/validation_recovery.py",
                "src/validation_psychometrics.py",
                "src/psychometrics.py",
                "requirements-lock.txt",
            },
            "parameter_recovery": {
                "src/parameter_recovery.py",
                "src/analyze_calibration.py",
                "src/psychometrics.py",
                "schemas/calibration_model_fits.schema.json",
                "requirements-lock.txt",
            },
            "path_correction_analysis": {
                "src/run_path_correction_analysis.py",
                "src/path_correction.py",
                "src/threshold_adapter.py",
                "src/experiment_runner.py",
                "src/config.py",
                "src/data_contracts.py",
                "src/analyze_calibration.py",
                "src/analyze_validation.py",
                "src/psychometrics.py",
                "src/experiment_design.py",
                "src/stimulus_renderer.py",
                "schemas/calibration_model_fits.schema.json",
                "schemas/human/experiment_config.schema.json",
                "schemas/human/trial.schema.json",
                "schemas/human/participant_thresholds.schema.json",
                "schemas/participant_thresholds.schema.json",
                "requirements-lock.txt",
            },
        }
        for scope, paths in required.items():
            with self.subTest(scope=scope):
                self.assertTrue(paths.issubset(set(IMPLEMENTATION_FILES[scope])))

    def test_path_correction_inherits_the_validated_threshold_dependency_set(self) -> None:
        path_files = set(IMPLEMENTATION_FILES["path_correction_analysis"])
        self.assertTrue(set(VALIDATION_ANALYSIS_FILES).issubset(path_files))

    def test_current_maps_verify_exactly(self) -> None:
        for scope in IMPLEMENTATION_FILES:
            with self.subTest(scope=scope):
                recorded = implementation_hashes(scope)
                verify_implementation_hashes(recorded, scope)
                self.assertEqual(list(recorded), sorted(recorded))

    def test_changed_missing_and_unexpected_entries_fail_closed(self) -> None:
        scope = "validation_analysis"
        current = implementation_hashes(scope)

        changed = dict(current)
        changed["src/psychometrics.py"] = "0" * 64
        with self.assertRaisesRegex(
            ImplementationProvenanceError, "changed=src/psychometrics.py"
        ):
            verify_implementation_hashes(changed, scope)

        missing = dict(current)
        missing.pop("requirements-lock.txt")
        with self.assertRaisesRegex(
            ImplementationProvenanceError, "missing=requirements-lock.txt"
        ):
            verify_implementation_hashes(missing, scope)

        unexpected = dict(current)
        unexpected["src/not-a-dependency.py"] = "1" * 64
        with self.assertRaisesRegex(
            ImplementationProvenanceError, "unexpected=src/not-a-dependency.py"
        ):
            verify_implementation_hashes(unexpected, scope)

        malformed = dict(current)
        malformed["src/psychometrics.py"] = "not-a-sha256"
        with self.assertRaisesRegex(
            ImplementationProvenanceError, "malformed implementation_hashes"
        ):
            verify_implementation_hashes(malformed, scope)


if __name__ == "__main__":
    unittest.main()
