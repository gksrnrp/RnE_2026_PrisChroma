from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from implementation_provenance import implementation_hashes

from experiment_runner import CollectionBlockedError, assert_schedule_config
from test_experiment_runner import frozen_config


ROOT = Path(__file__).resolve().parents[1]
HUMAN_SCHEMAS = ROOT / "schemas" / "human"


def _validator(name: str) -> Draft202012Validator:
    schema = json.loads((HUMAN_SCHEMAS / name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _human_config() -> dict:
    payload = frozen_config()
    return payload


def _human_trial() -> dict:
    payload = json.loads(
        (ROOT / "data" / "contracts" / "trial.example.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    payload["schema_version"] = "1.0"
    payload["data_origin"] = "human"
    payload.pop("staircase_id")
    payload["stimulus_manifest_sha256"] = "9" * 64
    payload["validation_plan_sha256"] = None
    payload["stimulus_onset_monotonic_ns"] = 123456789
    payload["validation_block_id"] = None
    payload["validation_level_index"] = None
    return payload


def _human_thresholds() -> dict:
    provenance = implementation_hashes("validation_analysis")
    block = {
        "block_id": "VB1",
        "session_id": "VS1",
        "completed_trial_count": 64,
        "source_trials_sha256": "1" * 64,
        "c80_sim": 2.8,
        "c80_ci": {"ci_low": 2.4, "ci_high": 3.2},
        "presented_delta_e00_min": 0.5,
        "presented_delta_e00_max": 8.0,
        "range_bracketed": True,
        "w": 0.31,
        "w_ci": {"ci_low": 0.2, "ci_high": 0.45},
        "fixed_lapse": 0.02,
        "fit_converged": True,
        "bootstrap_replicates": 1000,
        "bootstrap_successful_replicates": 980,
        "bootstrap_convergence_rate": 0.98,
    }
    return {
        "schema_version": "1.0",
        "data_origin": "human",
        "coordinate": "post_machado_delta_e00",
        "formula_version": "direct-c80-4afc-v1",
        "config_version": "FROZEN_TEST_V1",
        "config_sha256": "2" * 64,
        "git_commit": "3" * 40,
        "source_trials_sha256": "4" * 64,
        "source_validation_plan_sha256": "8" * 64,
        "source_stimulus_manifest_sha256": "9" * 64,
        "analysis_script_sha256": provenance["src/analyze_validation.py"],
        "implementation_hashes": provenance,
        "generated_at_utc": "2026-09-14T00:00:00Z",
        "bootstrap_seed": 20260914,
        "participants": [
            {
                "participant_id": "P_TEST_01",
                "cvd_type": "deutan",
                "severity": 0.8,
                "severity_source": "test source",
                "severity_session_id": "SEV1",
                "fixed_lapse": 0.02,
                "lapse_source": "test source",
                "lapse_session_id": "LAP1",
                "confirmed": True,
                "confirmation_review": {
                    "review_record_sha256": "a" * 64,
                    "review_type": "human_validation_confirmation",
                    "reviewer_id": "R_AUDIT_01",
                    "reviewed_at_utc": "2026-09-14T01:00:00Z",
                    "review_note": "All frozen inputs, block notes, and exclusions reviewed.",
                    "approved_for_confirmation": True,
                    "participant_id": "P_TEST_01",
                    "config_sha256": "2" * 64,
                    "source_trials_sha256": "4" * 64,
                    "validation_plan_sha256": "8" * 64,
                    "validation_manifest_sha256": "9" * 64,
                },
                "calibration_width_slope_sensitivity": {
                    "criterion": "AIC",
                    "shared_w_parameter_count": 4,
                    "free_w_parameter_count": 6,
                    "shared_w_aic": 250.0,
                    "free_w_aic": 253.0,
                    "delta_aic_free_minus_shared": 3.0,
                    "shared_w_converged": True,
                    "free_w_converged": True,
                    "comparison_resolved": True,
                    "free_w_preferred": False,
                    "interpretation": "shared_w_retained",
                    "d384_beta_interpretation_eligible": True,
                    "d384_beta_interpretation_reason": (
                        "shared_w_retained_and_d384_numerically_complete"
                    ),
                    "primary_c192_vs_b192_affected": False,
                },
                "predictions": [
                    {
                        "model": "D384",
                        "stroke_width_px": 4,
                        "stroke_width_deg": 0.1,
                        "c80_sim": 2.8,
                        "ci_low": 2.3,
                        "ci_high": 3.4,
                        "value_type": "predicted",
                        "w": 0.3,
                        "lapse": 0.02,
                        "fit_trial_count": 384,
                        "fit_converged": True,
                        "bootstrap_replicates": 1000,
                        "bootstrap_successful_replicates": 1000,
                        "bootstrap_convergence_rate": 1.0,
                        "bootstrap_c80_samples": [2.7, 2.8, 2.9] * 333 + [2.8],
                        "bootstrap_pairing_id": None,
                    }
                ],
                "validation": {
                    "stroke_width_px": 4,
                    "stroke_width_deg": 0.1,
                    "blocks": [
                        block,
                        {**block, "block_id": "VB2", "session_id": "VS2", "source_trials_sha256": "6" * 64, "c80_sim": 2.9},
                        {**block, "block_id": "VB3", "session_id": "VS3", "source_trials_sha256": "7" * 64, "c80_sim": 2.7},
                    ],
                    "coefficient_of_variation": 0.1 / 2.8,
                    "intervals_compatible": True,
                    "range_bracketed": True,
                    "stable": True,
                    "pooled_c80": 2.8,
                    "pooled_ci_low": 2.5,
                    "pooled_ci_high": 3.1,
                },
                "model_comparisons": {
                    "stroke_width_px": 4,
                    "stroke_width_deg": 0.1,
                    "observed_pooled_c80": 2.8,
                    "observed_ci_low": 2.5,
                    "observed_ci_high": 3.1,
                    "primary": {
                        "first_model": "B192",
                        "first_prediction": 2.1,
                        "first_abs_error": 0.7,
                        "second_model": "C192",
                        "second_prediction": 2.75,
                        "second_abs_error": 0.05,
                    },
                    "secondary": {
                        "first_model": "D384",
                        "first_prediction": 2.8,
                        "first_abs_error": 0.0,
                        "second_model": "B384",
                        "second_prediction": 2.2,
                        "second_abs_error": 0.6,
                    },
                    "prediction_separation": 0.65,
                    "observed_ci_half_width": 0.3,
                    "resolution_check": True,
                    "primary_outcome": "C192_lower_error",
                    "primary_delta_loss": {
                        "definition": "abs(C192_prediction-observed)-abs(B192_prediction-observed)",
                        "point_estimate": -0.65,
                        "ci_low": -0.8,
                        "ci_high": -0.4,
                        "confidence": 0.95,
                        "successful_combined_replicates": 480,
                        "calibration_pairing_id": "B192-C192-reference-width-parametric-v1",
                        "calibration_prediction_uncertainty_included": True,
                        "validation_observation_uncertainty_included": True,
                        "status": "resolved_C192_lower_error",
                        "reason": "bootstrap_ci_below_negative_tolerance",
                    },
                    "shared_w_sensitivity": {
                        "shared_w_aic": 250.0,
                        "free_w_aic": 253.0,
                        "delta_aic_free_minus_shared": 3.0,
                        "shared_w_converged": True,
                        "free_w_converged": True,
                        "comparison_resolved": True,
                        "free_w_preferred": False,
                        "interpretation": "shared_w_retained",
                    },
                    "metadata": {
                        "comparison_unit": "absolute_delta_e00_error",
                        "resolution_rule": "paired_calibration_and_validation_bootstrap_delta_loss_95ci",
                        "error_tie_tolerance_delta_e00": 1e-9,
                        "inference_scope": "paired-bootstrap-uncertainty-propagation-not-population-inference",
                        "model_or_width_reselected_after_validation": False,
                    },
                },
            }
        ],
    }


class HumanSchemaTests(unittest.TestCase):
    def test_all_three_schemas_are_meta_valid_and_accept_complete_records(self):
        fixtures = {
            "experiment_config.schema.json": _human_config(),
            "trial.schema.json": _human_trial(),
            "participant_thresholds.schema.json": _human_thresholds(),
        }
        self.assertEqual(set(fixtures), {path.name for path in HUMAN_SCHEMAS.glob("*.schema.json")})
        for name, payload in fixtures.items():
            with self.subTest(schema=name):
                self.assertEqual(list(_validator(name).iter_errors(payload)), [])

    def test_config_does_not_turn_an_unapproved_template_into_permission(self):
        payload = _human_config()
        payload["status"] = "template"
        payload["human_data_collection_allowed"] = False
        payload["privacy"]["approved"] = False
        self.assertEqual(list(_validator("experiment_config.schema.json").iter_errors(payload)), [])
        payload["human_data_collection_allowed"] = True
        self.assertTrue(list(_validator("experiment_config.schema.json").iter_errors(payload)))

    def test_config_rejects_unknown_fields_and_incomplete_validation_blocks(self):
        payload = _human_config()
        payload["display"]["invented"] = 1
        self.assertTrue(list(_validator("experiment_config.schema.json").iter_errors(payload)))
        payload = _human_config()
        payload["validation"]["trials_per_block"] = 60
        self.assertTrue(list(_validator("experiment_config.schema.json").iter_errors(payload)))

    def test_config_requires_fail_closed_abort_and_time_limit_policies(self):
        for field in (
            "active_trial_abort_policy",
            "incomplete_block_time_limit_policy",
        ):
            with self.subTest(field=field):
                payload = _human_config()
                del payload["timing"][field]
                self.assertTrue(
                    list(_validator("experiment_config.schema.json").iter_errors(payload))
                )
                with self.assertRaises(CollectionBlockedError):
                    assert_schedule_config(payload)

    def test_journal_root_and_participant_bound_filenames_are_fail_closed(self):
        payload = _human_config()
        payload["journal"]["root_repository_relative"] = ".git"
        self.assertTrue(
            list(_validator("experiment_config.schema.json").iter_errors(payload))
        )
        with self.assertRaisesRegex(CollectionBlockedError, "data/experiment/raw"):
            assert_schedule_config(payload)

        payload = _human_config()
        payload["journal"]["phase_files"]["calibration"] = (
            "P_OTHER-calibration.jsonl"
        )
        with self.assertRaisesRegex(CollectionBlockedError, "참가자와 결박"):
            assert_schedule_config(payload)

    def test_trial_requires_raster_provenance_and_status_consistency(self):
        payload = _human_trial()
        del payload["stimulus_decoded_rgb_sha256"]
        self.assertTrue(list(_validator("trial.schema.json").iter_errors(payload)))
        payload = _human_trial()
        payload["trial_status"] = "aborted"
        self.assertTrue(list(_validator("trial.schema.json").iter_errors(payload)))

    def test_validation_trial_uses_fixed_block_fields_not_staircase(self):
        payload = _human_trial()
        payload.update(
            phase="validation",
            width_index=None,
            session_block=None,
            delta_e00_level_index=None,
            validation_level_index=1,
            validation_block_id="VB1",
            validation_plan_sha256="9" * 64,
        )
        self.assertEqual(list(_validator("trial.schema.json").iter_errors(payload)), [])
        payload["validation_block_id"] = None
        self.assertTrue(list(_validator("trial.schema.json").iter_errors(payload)))

    def test_unstable_threshold_validation_cannot_pool_blocks(self):
        payload = _human_thresholds()
        validation = payload["participants"][0]["validation"]
        validation["stable"] = False
        self.assertTrue(list(_validator("participant_thresholds.schema.json").iter_errors(payload)))
        validation["pooled_c80"] = None
        validation["pooled_ci_low"] = None
        validation["pooled_ci_high"] = None
        comparison = payload["participants"][0]["model_comparisons"]
        comparison["observed_pooled_c80"] = None
        comparison["observed_ci_low"] = None
        comparison["observed_ci_high"] = None
        comparison["observed_ci_half_width"] = None
        comparison["resolution_check"] = False
        comparison["primary_outcome"] = "validation_unstable"
        comparison["primary_delta_loss"].update(
            point_estimate=None,
            ci_low=None,
            ci_high=None,
            successful_combined_replicates=0,
            calibration_pairing_id=None,
            calibration_prediction_uncertainty_included=False,
            validation_observation_uncertainty_included=False,
            status="unresolvable",
            reason="validation_unstable",
        )
        for pair in (comparison["primary"], comparison["secondary"]):
            pair["first_abs_error"] = None
            pair["second_abs_error"] = None
        self.assertEqual(list(_validator("participant_thresholds.schema.json").iter_errors(payload)), [])

    def test_resolved_delta_loss_requires_ci_entirely_on_declared_side(self):
        payload = _human_thresholds()
        delta_loss = payload["participants"][0]["model_comparisons"]["primary_delta_loss"]
        delta_loss["ci_high"] = 0.1
        self.assertTrue(
            list(_validator("participant_thresholds.schema.json").iter_errors(payload))
        )

    def test_confirmed_threshold_requires_a_bound_review_record(self):
        payload = _human_thresholds()
        payload["participants"][0]["confirmation_review"] = None
        self.assertTrue(
            list(_validator("participant_thresholds.schema.json").iter_errors(payload))
        )
        payload["participants"][0]["confirmed"] = False
        self.assertEqual(
            list(_validator("participant_thresholds.schema.json").iter_errors(payload)),
            [],
        )

    def test_draft_example_has_runner_fields_but_remains_blocked(self):
        payload = json.loads(
            (ROOT / "data" / "contracts" / "experiment_config.example.json")
            .read_text(encoding="utf-8")
        )
        for path in (
            ("observer_profile", "fixed_lapse"),
            ("display", "color_profile"),
            ("line", "renderer"),
            ("coordinates", "positions_px"),
            ("timing", "fixation_ms"),
            ("sampling", "delta_e00_levels"),
            ("privacy", "approved"),
        ):
            self.assertIn(path[1], payload[path[0]])
        self.assertFalse(payload["human_data_collection_allowed"])
        self.assertFalse(payload["privacy"]["approved"])
        with self.assertRaises(CollectionBlockedError):
            assert_schedule_config(payload)

    def test_human_1_0_template_is_schema_valid_but_cannot_collect(self):
        payload = json.loads(
            (
                ROOT
                / "data"
                / "contracts"
                / "human_experiment_config.template.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            list(_validator("experiment_config.schema.json").iter_errors(payload)),
            [],
        )
        self.assertEqual(payload["status"], "template")
        self.assertFalse(payload["human_data_collection_allowed"])
        self.assertFalse(payload["privacy"]["approved"])
        self.assertIn("PLACEHOLDER", payload["config_version"])
        with self.assertRaises(CollectionBlockedError):
            assert_schedule_config(payload)


if __name__ == "__main__":
    unittest.main()
