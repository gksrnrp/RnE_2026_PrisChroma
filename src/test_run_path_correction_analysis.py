import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_path_correction_analysis import (
    ANALYZE_VALIDATION_SCRIPT,
    MODELS,
    PathCorrectionAnalysisError,
    main,
    run_path_correction_analysis,
)
from test_human_contracts import _human_config, _human_thresholds
from test_validation_design import _calibration
from validation_design import build_validation_plan, compute_plan_sha256


def _write_json(path: Path, payload: dict) -> str:
    raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _replace_plan_hash(plan: dict) -> None:
    plan["plan_sha256"] = compute_plan_sha256(plan)
    for row in plan["schedule"]:
        row["validation_plan_sha256"] = plan["plan_sha256"]


def _threshold_payload(config: dict, config_sha: str, plan: dict) -> dict:
    payload = _human_thresholds()
    payload["config_version"] = config["config_version"]
    payload["config_sha256"] = config_sha
    payload["source_validation_plan_sha256"] = plan["plan_sha256"]
    payload["analysis_script_sha256"] = hashlib.sha256(
        ANALYZE_VALIDATION_SCRIPT.read_bytes()
    ).hexdigest()
    payload["git_commit"] = plan["source_git_commit"]
    participant = payload["participants"][0]
    profile = config["observer_profile"]
    participant.update(
        participant_id=profile["participant_id"],
        cvd_type=profile["cvd_type"],
        severity=profile["severity"],
        severity_source=profile["severity_source"],
        severity_session_id=profile["severity_session_id"],
        fixed_lapse=profile["fixed_lapse"],
        lapse_source=profile["lapse_source"],
        lapse_session_id=profile["lapse_session_id"],
    )
    participant["confirmation_review"].update(
        participant_id=profile["participant_id"],
        config_sha256=config_sha,
        source_trials_sha256=payload["source_trials_sha256"],
        validation_plan_sha256=plan["plan_sha256"],
        validation_manifest_sha256=payload["source_stimulus_manifest_sha256"],
    )
    width_px = plan["selected_width"]["stroke_width_px"]
    width_deg = plan["selected_width"]["stroke_width_deg"]
    template = participant["predictions"][0]
    model_values = {
        "B96": (2.1, 1.8, 2.4, 96),
        "B288": (2.3, 2.0, 2.6, 288),
        "C96": (2.8, 2.5, 3.1, 96),
        "D288": (2.9, 2.6, 3.2, 288),
    }
    participant["predictions"] = [
        {
            **template,
            "model": model,
            "stroke_width_px": width_px,
            "stroke_width_deg": width_deg,
            "c80_sim": values[0],
            "ci_low": values[1],
            "ci_high": values[2],
            "fit_trial_count": values[3],
        }
        for model, values in model_values.items()
    ]
    participant["validation"].update(
        stroke_width_px=width_px,
        stroke_width_deg=width_deg,
        pooled_c80=2.8,
        pooled_ci_low=2.5,
        pooled_ci_high=3.1,
    )
    participant["model_comparisons"].update(
        stroke_width_px=width_px,
        stroke_width_deg=width_deg,
        observed_pooled_c80=2.8,
        observed_ci_low=2.5,
        observed_ci_high=3.1,
    )
    return payload


def _fixture(directory: Path) -> tuple[Path, Path, Path, dict, dict]:
    config = _human_config()
    config["color_design"]["search_grid_steps"] = 256
    config["color_design"]["target_absolute_tolerance_delta_e00"] = 0.5
    config_path = directory / "config.json"
    config_sha = _write_json(config_path, config)

    calibration = _calibration(config_sha, config=config, human=True)
    calibration["source"].update(
        config_version=config["config_version"],
        stimulus_manifest_sha256="f" * 64,
        stimulus_family_id=config["color_design"]["stimulus_family_id"],
        renderer_id=config["line"]["renderer"],
    )
    calibration_participant = calibration["participants"][0]
    profile = config["observer_profile"]
    calibration_participant.update(
        participant_id=profile["participant_id"],
        cvd_type=profile["cvd_type"],
        machado_severity=profile["severity"],
        severity_source=profile["severity_source"],
        severity_session_id=profile["severity_session_id"],
        fixed_lapse=profile["fixed_lapse"],
        lapse_source=profile["lapse_source"],
        lapse_session_id=profile["lapse_session_id"],
    )
    calibration_path = directory / "calibration.json"
    _write_json(calibration_path, calibration)
    plan = build_validation_plan(config_path, calibration_path, seed=19)
    plan_path = directory / "validation-plan.json"
    _write_json(plan_path, plan)

    thresholds = _threshold_payload(config, config_sha, plan)
    thresholds_path = directory / "participant-thresholds.json"
    _write_json(thresholds_path, thresholds)
    return config_path, plan_path, thresholds_path, plan, thresholds


class PathCorrectionAnalysisTests(unittest.TestCase):
    def test_cli_human_mode_does_not_require_a_source_level(self) -> None:
        with patch(
            "run_path_correction_analysis.run_path_correction_analysis",
            return_value={"artifact_type": "path_constrained_correction_analysis"},
        ) as run, patch("run_path_correction_analysis._write_json_atomic"):
            status = main(
                [
                    "--config",
                    "config.json",
                    "--validation-plan",
                    "plan.json",
                    "--participant-thresholds",
                    "thresholds.json",
                    "--participant-id",
                    "P01",
                    "--output",
                    "output.json",
                ]
            )
        self.assertEqual(status, 0)
        self.assertIsNone(run.call_args.kwargs["source_level_index"])
        self.assertFalse(run.call_args.kwargs["development_mode"])

    def test_human_artifacts_are_bound_and_models_are_corrected_on_same_path(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            config_path, plan_path, thresholds_path, plan, _ = _fixture(Path(name))
            with patch(
                "run_path_correction_analysis._require_clean_tracked_implementation",
                return_value="a" * 40,
            ):
                result = run_path_correction_analysis(
                    config_path,
                    plan_path,
                    thresholds_path,
                    participant_id="P_TEST_01",
                    models=MODELS,
                )
        self.assertEqual(result["data_origin"], "human")
        self.assertFalse(result["development_mode"])
        self.assertFalse(result["research_conclusion_allowed"])
        self.assertTrue(
            result["claim_flags"]["computational_constraint_claim_allowed"]
        )
        self.assertFalse(
            result["claim_flags"]["human_readability_improvement_claim_allowed"]
        )
        self.assertEqual(
            result["provenance"]["validation_plan_canonical_sha256"],
            plan["plan_sha256"],
        )
        self.assertEqual(
            result["analysis_scope"]["source_levels"],
            "all_eight_frozen_validation_levels",
        )
        self.assertEqual(result["analysis_scope"]["level_count"], 8)
        self.assertFalse(result["analysis_scope"]["selection_after_validation_allowed"])
        self.assertTrue(result["claim_flags"]["all_frozen_validation_levels_reported"])
        self.assertTrue(
            result["claim_flags"]["d288_correction_computation_allowed"]
        )
        self.assertTrue(
            result["claim_flags"]["d288_beta_or_width_effect_claim_allowed"]
        )
        self.assertEqual(
            result["interpretation"]["calibration_width_slope_sensitivity"][
                "d288_beta_interpretation_eligible"
            ],
            True,
        )
        self.assertNotIn("source_level", result)
        self.assertNotIn("corrections", result)
        self.assertEqual(
            [row["source_level"]["validation_level_index"] for row in result["level_results"]],
            list(range(1, 9)),
        )
        for level in result["level_results"]:
            self.assertEqual(
                [row["model"] for row in level["corrections"]], list(MODELS)
            )
            for row in level["corrections"]:
                correction = row["correction"]
                self.assertNotIn("allowed_quantization_overshoot_delta_e00", row)
                self.assertNotIn(
                    "maximum_overshoot_delta_e00", correction["search"]
                )
                self.assertGreaterEqual(
                    correction["achieved_simulated_delta_e00"],
                    row["target_simulated_delta_e00"],
                )
                self.assertEqual(
                    correction["behavioral_scope"],
                    "same_lab_midpoint_direction_as_c80_measurement",
                )
        self.assertEqual(
            result["level_results"][0]["corrections"][0][
                "achieved_vs_observed_pooled_c80_95ci"
            ],
            "below_observed_pooled_c80_95ci",
        )
        self.assertEqual(
            result["level_results"][0]["corrections"][2][
                "achieved_vs_observed_pooled_c80_95ci"
            ],
            "within_observed_pooled_c80_95ci",
        )

    def test_ineligible_d288_beta_claim_does_not_block_d288_correction(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            config_path, plan_path, thresholds_path, _, thresholds = _fixture(directory)
            sensitivity = thresholds["participants"][0][
                "calibration_width_slope_sensitivity"
            ]
            sensitivity.update(
                free_w_aic=240.0,
                delta_aic_free_minus_shared=-10.0,
                comparison_resolved=True,
                free_w_preferred=True,
                interpretation="free_w_lower_aic_warning_not_selected",
                d288_beta_interpretation_eligible=False,
                d288_beta_interpretation_reason="free_w_lower_aic",
            )
            _write_json(thresholds_path, thresholds)
            with patch(
                "run_path_correction_analysis._require_clean_tracked_implementation",
                return_value="a" * 40,
            ):
                result = run_path_correction_analysis(
                    config_path,
                    plan_path,
                    thresholds_path,
                    participant_id="P_TEST_01",
                    models=MODELS,
                )
        self.assertTrue(result["claim_flags"]["d288_correction_computation_allowed"])
        self.assertFalse(
            result["claim_flags"]["d288_beta_or_width_effect_claim_allowed"]
        )
        self.assertTrue(
            all(
                any(row["model"] == "D288" for row in level["corrections"])
                for level in result["level_results"]
            )
        )

    def test_human_mode_rejects_partial_model_selection(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            config_path, plan_path, thresholds_path, _, _ = _fixture(Path(name))
            with self.assertRaisesRegex(PathCorrectionAnalysisError, "선택 보고"):
                run_path_correction_analysis(
                    config_path,
                    plan_path,
                    thresholds_path,
                    participant_id="P_TEST_01",
                    models=("B96", "C96"),
                )

    def test_human_mode_rejects_post_validation_source_level_selection(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            config_path, plan_path, thresholds_path, _, _ = _fixture(Path(name))
            with self.assertRaisesRegex(
                PathCorrectionAnalysisError, "8개 validation 수준을 모두"
            ):
                run_path_correction_analysis(
                    config_path,
                    plan_path,
                    thresholds_path,
                    participant_id="P_TEST_01",
                    source_level_index=1,
                    models=MODELS,
                )

    def test_thresholds_must_name_the_exact_frozen_plan(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            config_path, plan_path, thresholds_path, _, thresholds = _fixture(directory)
            thresholds["source_validation_plan_sha256"] = "0" * 64
            thresholds["participants"][0]["confirmation_review"][
                "validation_plan_sha256"
            ] = "0" * 64
            _write_json(thresholds_path, thresholds)
            with self.assertRaisesRegex(
                PathCorrectionAnalysisError, "validation plan canonical"
            ):
                run_path_correction_analysis(
                    config_path,
                    plan_path,
                    thresholds_path,
                    participant_id="P_TEST_01",
                    models=MODELS,
                )

    def test_stale_validation_dependency_blocks_path_correction(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            config_path, plan_path, thresholds_path, _, thresholds = _fixture(directory)
            thresholds["implementation_hashes"]["src/validation_psychometrics.py"] = (
                "0" * 64
            )
            _write_json(thresholds_path, thresholds)
            with self.assertRaisesRegex(
                PathCorrectionAnalysisError, "changed=src/validation_psychometrics.py"
            ):
                run_path_correction_analysis(
                    config_path,
                    plan_path,
                    thresholds_path,
                    participant_id="P_TEST_01",
                    models=MODELS,
                )

    def test_unconfirmed_human_result_is_rejected_before_correction(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            config_path, plan_path, thresholds_path, _, thresholds = _fixture(directory)
            participant = thresholds["participants"][0]
            participant["confirmed"] = False
            participant["confirmation_review"] = None
            _write_json(thresholds_path, thresholds)
            with self.assertRaisesRegex(PathCorrectionAnalysisError, "confirmed=false"):
                run_path_correction_analysis(
                    config_path,
                    plan_path,
                    thresholds_path,
                    participant_id="P_TEST_01",
                    models=MODELS,
                )

    def test_synthetic_analysis_requires_explicit_development_mode(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            config_path, plan_path, thresholds_path, plan, thresholds = _fixture(directory)
            plan["data_origin"] = "synthetic"
            _replace_plan_hash(plan)
            _write_json(plan_path, plan)
            participant = copy.deepcopy(thresholds["participants"][0])
            participant["confirmed"] = False
            participant["confirmation_review"] = None
            development = {
                "schema_version": "0.1-development",
                "artifact_type": "synthetic_validation_analysis",
                "data_origin": "synthetic",
                "research_conclusion_allowed": False,
                "coordinate": "post_machado_delta_e00",
                "formula_version": "direct-c80-4afc-v1",
                "config_version": thresholds["config_version"],
                "config_sha256": thresholds["config_sha256"],
                "source_validation_plan_sha256": plan["plan_sha256"],
                "participants": [participant],
            }
            _write_json(thresholds_path, development)
            with self.assertRaisesRegex(PathCorrectionAnalysisError, "data_origin=human"):
                run_path_correction_analysis(
                    config_path,
                    plan_path,
                    thresholds_path,
                    participant_id="P_TEST_01",
                    models=MODELS,
                )
            result = run_path_correction_analysis(
                config_path,
                plan_path,
                thresholds_path,
                participant_id="P_TEST_01",
                source_level_index=1,
                models=("B96",),
                development_mode=True,
            )
        self.assertEqual(result["data_origin"], "synthetic")
        self.assertTrue(result["development_mode"])
        self.assertFalse(result["research_conclusion_allowed"])
        self.assertEqual(result["analysis_scope"]["level_count"], 1)
        self.assertFalse(result["claim_flags"]["all_frozen_validation_levels_reported"])
        self.assertTrue(result["claim_flags"]["d288_correction_computation_allowed"])
        self.assertFalse(
            result["claim_flags"]["d288_beta_or_width_effect_claim_allowed"]
        )
        self.assertEqual(result["source_level"], result["level_results"][0]["source_level"])
        self.assertEqual(result["corrections"], result["level_results"][0]["corrections"])

    def test_synthetic_mode_requires_explicit_single_source_level(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            config_path, plan_path, thresholds_path, plan, thresholds = _fixture(directory)
            plan["data_origin"] = "synthetic"
            _replace_plan_hash(plan)
            _write_json(plan_path, plan)
            participant = copy.deepcopy(thresholds["participants"][0])
            participant["confirmed"] = False
            participant["confirmation_review"] = None
            development = {
                "schema_version": "0.1-development",
                "artifact_type": "synthetic_validation_analysis",
                "data_origin": "synthetic",
                "research_conclusion_allowed": False,
                "coordinate": "post_machado_delta_e00",
                "formula_version": "direct-c80-4afc-v1",
                "config_version": thresholds["config_version"],
                "config_sha256": thresholds["config_sha256"],
                "source_validation_plan_sha256": plan["plan_sha256"],
                "participants": [participant],
            }
            _write_json(thresholds_path, development)
            with self.assertRaisesRegex(PathCorrectionAnalysisError, "source_level_index"):
                run_path_correction_analysis(
                    config_path,
                    plan_path,
                    thresholds_path,
                    participant_id="P_TEST_01",
                    models=("B96",),
                    development_mode=True,
                )


if __name__ == "__main__":
    unittest.main()
