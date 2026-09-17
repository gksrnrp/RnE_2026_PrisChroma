import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from experiment_runner import (
    CollectionBlockedError,
    JournalError,
    TrialJournal,
    assess_experiment_gates,
    assert_human_collection_allowed,
    assert_schedule_config,
    build_balanced_schedule,
    canonical_journal_path,
    expected_collection_session_id,
    main,
)
from stimulus_renderer import RENDERER_ID
from experiment_design import geometric_validation_widths, pixels_to_visual_angle_deg


def frozen_config():
    config = {
        "schema_version": "1.0",
        "config_version": "frozen-test-v1",
        "status": "preregistered",
        "human_data_collection_allowed": True,
        "task": {
            "type": "4afc_odd_one_out",
            "layout": "2x2",
            "choice_count": 4,
            "practice_feedback": True,
            "main_feedback": False,
        },
        "display": {
            "model": "Lab Display 23",
            "width_mm": 510.0,
            "resolution_width_px": 1920,
            "resolution_height_px": 1080,
            "os_scale_percent": 100,
            "app_scale_percent": 100,
            "viewing_distance_mm": 600,
            "background_rgb": [117, 117, 117],
            "color_profile": "sRGB IEC61966-2.1",
            "colorimetry_policy": "measured-display-colorimetry",
            "calibration_id": "CAL-2026-09-14",
            "brightness_percent": 50.0,
            "luminance_cd_m2": 120.0,
            "luminance_measurement_method": "calibrated colorimeter fixture",
            "width_measurement_method": "ruler fixture",
            "viewing_distance_control": "fixed chin-rest fixture",
            "operating_system": "Windows test fixture",
            "native_gui_runtime": "CPython/Tk test fixture",
            "pixel_mapping_verification": "DPI-aware 1:1 raster fixture",
            "ambient_light_condition": "fixed test-room lighting",
            "ambient_light_measurement_method": "lux-meter fixture",
            "night_mode_disabled": True,
        },
        "line": {
            "length_px": 300,
            "orientation_deg": 0.0,
            "cap": "butt",
            "antialiasing": "disabled",
            "renderer": RENDERER_ID,
            "pixel_snapping": "integer-centers",
        },
        "color_design": {
            "stimulus_family_id": "LINE_FAMILY_TEST_V1",
            "midpoint_lab": [60.0, 8.0, 8.0],
            "direction_lab": [0.0, 1.0, 0.35],
            "symmetric_about_midpoint": True,
            "gamut_policy": "reject",
            "rgb_quantization": "nearest_8bit",
            "target_absolute_tolerance_delta_e00": 0.07,
            "target_relative_tolerance": 0.03,
            "search_grid_steps": 1024,
        },
        "observer_profile": {
            "participant_id": "P_TEST_01",
            "cvd_type": "deutan",
            "severity": 0.8,
            "severity_source": "independent large-patch session",
            "severity_session_id": "SEVERITY_TEST_01",
            "fixed_lapse": 0.02,
            "lapse_source": "preregistered synthetic test value",
            "lapse_session_id": "LAPSE_TEST_01",
        },
        "coordinates": {
            "positions_px": [[500, 300], [800, 300], [500, 600], [800, 600]],
            "center_spacing_px": 300,
        },
        "timing": {
            "fixation_ms": 500,
            "inter_trial_ms": 350,
            "response_deadline_policy": "unlimited",
            "response_deadline_ms": None,
            "active_trial_abort_policy": "invalidate-analysis-no-retry",
            "max_session_duration_min": 30,
            "incomplete_block_time_limit_policy": "invalidate-participant-dataset-no-retry",
            "planned_break_interval_trials": 24,
            "planned_break_duration_min": 2,
            "stop_rules": "stop immediately for eye strain, headache, or participant request",
            "resume_policy": "append-only-retry-incomplete",
        },
        "sampling": {
            "delta_e00_levels": [0.5, 0.7, 0.9, 1.2, 1.6, 2.1, 2.8, 3.7, 4.9, 6.5, 8.6, 11.4],
            "randomization": "balanced-4-session-blocks-within-block-shuffle",
            "practice_trials": 6,
        },
        "calibration": {
            "widths": [
                {"width_px": width, "width_deg": pixels_to_visual_angle_deg(
                    width,
                    display_width_mm=510.0,
                    horizontal_resolution_px=1920,
                    viewing_distance_mm=600.0,
                )}
                for width in (2, 6, 12)
            ],
            "trials_per_width": 96,
        },
        "validation": {
            "candidates": [
                {
                    "geometric_target_px": target,
                    "rendered_width_px": width,
                    "width_deg": pixels_to_visual_angle_deg(
                        width,
                        display_width_mm=510.0,
                        horizontal_resolution_px=1920,
                        viewing_distance_mm=600.0,
                    ),
                }
                for target, width in zip(geometric_validation_widths((2, 6, 12)), (3, 8))
            ],
            "independent_blocks": 3,
            "delta_e00_level_count": 8,
            "trials_per_block": 64,
            "randomization": "balanced-position-and-odd-color-within-block",
        },
        "model_plan": {
            "models": ["B96", "B288", "C96", "D288"],
            "include_szafir_transfer": True,
            "szafir_probability_coordinate": 0.5,
        },
        "privacy": {
            "approved": True,
            "approval_id": "RNE-2026-001",
            "approved_at_utc": "2026-09-14T00:00:00Z",
            "consent_version": "v1",
            "pseudonymization": "random participant codes",
            "storage_policy": "encrypted school device, restricted access",
        },
        "journal": {
            "storage_policy": "repository-relative-gitignored",
            "root_repository_relative": "data/experiment/raw",
            "phase_files": {
                "practice": "P_TEST_01-practice.jsonl",
                "calibration": "P_TEST_01-calibration.jsonl",
                "validation": "P_TEST_01-validation.jsonl",
            },
            "session_ids": {
                "practice": "P_TEST_01-PRACTICE-01",
                "calibration": {
                    "1": "P_TEST_01-CAL-01",
                    "2": "P_TEST_01-CAL-02",
                    "3": "P_TEST_01-CAL-03",
                    "4": "P_TEST_01-CAL-04",
                },
                "validation": {
                    "V01": "P_TEST_01-VAL-01",
                    "V02": "P_TEST_01-VAL-02",
                    "V03": "P_TEST_01-VAL-03",
                },
            },
        },
    }
    return config


class HumanCollectionGateTests(unittest.TestCase):
    def test_static_gates_pass_but_session_runtime_still_blocks_humans(self):
        config = frozen_config()
        assert_schedule_config(config)
        report = assess_experiment_gates(config)
        self.assertTrue(report["config_gate_passed"])
        self.assertTrue(report["renderer_implemented"])
        self.assertTrue(report["renderer_verified"])
        self.assertTrue(report["human_response_collection_implemented"])
        self.assertFalse(report["human_data_collection_allowed"])
        self.assertEqual(
            [item["gate"] for item in report["blockers"]], ["session_runtime"]
        )
        with self.assertRaisesRegex(CollectionBlockedError, "session_runtime"):
            assert_human_collection_allowed(config)

    def test_current_draft_example_is_blocked(self):
        example = Path(__file__).resolve().parents[1] / "data" / "contracts" / "experiment_config.example.json"
        payload = json.loads(example.read_text(encoding="utf-8"))
        with self.assertRaises(CollectionBlockedError):
            assert_human_collection_allowed(payload)
        self.assertEqual(main(["--check-config", str(example)]), 2)

    def test_each_safety_family_is_required(self):
        mutations = (
            ("display", "color_profile"),
            ("line", "renderer"),
            ("coordinates", "positions_px"),
            ("timing", "fixation_ms"),
            ("sampling", "delta_e00_levels"),
            ("privacy", "approval_id"),
        )
        for section, field in mutations:
            with self.subTest(path=f"{section}.{field}"):
                config = frozen_config()
                del config[section][field]
                with self.assertRaises(CollectionBlockedError):
                    assert_human_collection_allowed(config)

    def test_placeholder_and_unapproved_privacy_are_blocked(self):
        placeholder = frozen_config()
        placeholder["line"]["renderer"] = "EXAMPLE_NOT_FROZEN"
        with self.assertRaises(CollectionBlockedError):
            assert_human_collection_allowed(placeholder)
        unapproved = frozen_config()
        unapproved["privacy"]["approved"] = False
        with self.assertRaises(CollectionBlockedError):
            assert_human_collection_allowed(unapproved)

    def test_nominal_srgb_policy_allows_explicitly_unmeasured_luminance(self):
        config = frozen_config()
        config["display"]["colorimetry_policy"] = "nominal-srgb-device-specific"
        config["display"]["luminance_cd_m2"] = None
        config["display"]["luminance_measurement_method"] = (
            "not_measured_no_luminance_meter"
        )
        assert_schedule_config(config)

    def test_unmeasured_luminance_requires_nominal_policy_and_exact_method(self):
        wrong_policy = frozen_config()
        wrong_policy["display"]["luminance_cd_m2"] = None
        wrong_policy["display"]["luminance_measurement_method"] = (
            "not_measured_no_luminance_meter"
        )
        with self.assertRaisesRegex(CollectionBlockedError, "nominal-srgb"):
            assert_schedule_config(wrong_policy)

        wrong_method = frozen_config()
        wrong_method["display"]["colorimetry_policy"] = (
            "nominal-srgb-device-specific"
        )
        wrong_method["display"]["luminance_cd_m2"] = None
        wrong_method["display"]["luminance_measurement_method"] = "estimated"
        with self.assertRaisesRegex(CollectionBlockedError, "not_measured"):
            assert_schedule_config(wrong_method)

    def test_brightness_percent_above_100_is_blocked(self):
        config = frozen_config()
        config["display"]["brightness_percent"] = 101.0
        with self.assertRaisesRegex(CollectionBlockedError, "100"):
            assert_schedule_config(config)

    def test_fixed_response_deadline_is_blocked_to_prevent_selective_retry_bias(self):
        config = frozen_config()
        config["timing"]["response_deadline_policy"] = "fixed"
        config["timing"]["response_deadline_ms"] = 3000
        with self.assertRaisesRegex(CollectionBlockedError, "낙관 편향"):
            assert_schedule_config(config)

    def test_journal_contract_has_one_safe_path_and_unique_frozen_sessions(self):
        config = frozen_config()
        with tempfile.TemporaryDirectory() as directory:
            expected = (
                Path(directory)
                / "data"
                / "experiment"
                / "raw"
                / "P_TEST_01-calibration.jsonl"
            ).resolve()
            self.assertEqual(
                canonical_journal_path(config, "calibration", directory), expected
            )
        self.assertEqual(
            expected_collection_session_id(
                config, "calibration", session_block=3
            ),
            "P_TEST_01-CAL-03",
        )
        config["journal"]["phase_files"]["validation"] = config["journal"][
            "phase_files"
        ]["calibration"]
        with self.assertRaisesRegex(CollectionBlockedError, "참가자와 결박"):
            assert_schedule_config(config)

        config = frozen_config()
        config["journal"]["session_ids"]["validation"]["V02"] = (
            config["journal"]["session_ids"]["calibration"]["2"]
        )
        with self.assertRaisesRegex(CollectionBlockedError, "session_id는 서로"):
            assert_schedule_config(config)


class BalancedScheduleTests(unittest.TestCase):
    def test_schedule_is_fully_balanced_and_has_unique_ids(self):
        schedule = build_balanced_schedule(
            frozen_config(), 77, git_commit="a" * 40
        )
        self.assertEqual(len(schedule), 288)
        self.assertEqual({row["trial_id"] for row in schedule}, set(range(1, 289)))
        self.assertEqual(Counter(row["stroke_width_px"] for row in schedule), {2: 96, 6: 96, 12: 96})
        self.assertEqual(Counter(row["session_block"] for row in schedule), {1: 72, 2: 72, 3: 72, 4: 72})
        self.assertEqual(
            set(
                Counter(
                    (row["session_block"], row["stroke_width_px"])
                    for row in schedule
                ).values()
            ),
            {24},
        )
        self.assertEqual(
            set(
                Counter(
                    (
                        row["session_block"],
                        row["stroke_width_px"],
                        row["target_delta_e00_simulated"],
                    )
                    for row in schedule
                ).values()
            ),
            {2},
        )
        self.assertEqual(
            [row["session_block"] for row in schedule],
            [1] * 72 + [2] * 72 + [3] * 72 + [4] * 72,
        )

        cells = Counter(
            (row["stroke_width_px"], row["target_delta_e00_simulated"])
            for row in schedule
        )
        self.assertEqual(set(cells.values()), {8})
        positions = Counter(
            (
                row["stroke_width_px"],
                row["target_delta_e00_simulated"],
                row["correct_position"],
            )
            for row in schedule
        )
        self.assertEqual(set(positions.values()), {2})
        odd_colors = Counter(
            (
                row["stroke_width_px"],
                row["target_delta_e00_simulated"],
                row["correct_position"],
                row["odd_color"],
            )
            for row in schedule
        )
        self.assertEqual(set(odd_colors.values()), {1})
        self.assertTrue(all(row["config_sha256"] == schedule[0]["config_sha256"] for row in schedule))
        self.assertTrue(all(row["git_commit"] == "a" * 40 for row in schedule))
        self.assertTrue(all(row["participant_id"] == "P_TEST_01" for row in schedule))
        self.assertTrue(all(row["stimulus_family_id"] == "LINE_FAMILY_TEST_V1" for row in schedule))
        self.assertTrue(all(row["cvd_type"] == "deutan" for row in schedule))
        self.assertTrue(all(row["machado_severity"] == 0.8 for row in schedule))

    def test_same_seed_is_reproducible_and_different_seed_changes_order(self):
        first = build_balanced_schedule(frozen_config(), 123, git_commit="b" * 40)
        second = build_balanced_schedule(frozen_config(), 123, git_commit="b" * 40)
        third = build_balanced_schedule(frozen_config(), 124, git_commit="b" * 40)
        self.assertEqual(first, second)
        first_conditions = [(r["stroke_width_px"], r["delta_e00_level_index"], r["correct_position"], r["odd_color"]) for r in first]
        third_conditions = [(r["stroke_width_px"], r["delta_e00_level_index"], r["correct_position"], r["odd_color"]) for r in third]
        self.assertNotEqual(first_conditions, third_conditions)

    def test_cli_writes_schedule_only_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            output_path = Path(directory) / "schedule.json"
            config_path.write_text(json.dumps(frozen_config()), encoding="utf-8")
            exit_code = main(
                [
                    "--make-schedule",
                    str(config_path),
                    "--seed",
                    "9",
                    "--git-commit",
                    "c" * 40,
                    "--output",
                    str(output_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact_type"], "balanced_schedule_only")
            self.assertTrue(payload["renderer_implemented"])
            self.assertTrue(payload["renderer_verified"])
            self.assertTrue(payload["human_response_collection_implemented"])
            self.assertFalse(payload["human_data_collection_allowed"])
            self.assertEqual(payload["trial_count"], 288)


class TrialJournalTests(unittest.TestCase):
    def test_append_duplicate_rejection_and_safe_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            first = TrialJournal(path)
            record = {
                "participant_id": "P01",
                "session_id": "S01",
                "config_sha256": "a" * 64,
                "trial_id": 1,
                "answer": 4,
            }
            first.append(record)
            self.assertEqual(first.records, (record,))
            with self.assertRaises(JournalError):
                first.append(record)

            resumed = TrialJournal(path)
            self.assertEqual(resumed.completed_keys, frozenset({("P01", "a" * 64, 1)}))
            self.assertEqual(resumed.records, (record,))
            with self.assertRaises(JournalError):
                resumed.append({**record, "answer": 2})
            resumed.append({**record, "trial_id": 2})
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[1])["trial_id"], 2)

    def test_session_rename_cannot_repeat_same_planned_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            journal = TrialJournal(path)
            first = {
                "participant_id": "P01",
                "session_id": "S01",
                "config_sha256": "a" * 64,
                "trial_id": 1,
            }
            journal.append(first)
            with self.assertRaisesRegex(JournalError, "중복"):
                journal.append({**first, "session_id": "S02"})
            journal.append(
                {**first, "session_id": "S02", "config_sha256": "b" * 64}
            )
            journal.append({**first, "participant_id": "P02"})
            self.assertEqual(len(journal.completed_keys), 3)

    def test_aborted_attempt_remains_retryable_until_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            base = {
                "participant_id": "P01",
                "session_id": "S01",
                "config_sha256": "a" * 64,
                "trial_id": 1,
                "phase": "practice",
            }
            journal = TrialJournal(path)
            journal.append({**base, "trial_status": "aborted"})
            self.assertEqual(journal.completed_keys, frozenset())
            self.assertEqual(
                journal.attempted_keys,
                frozenset({("P01", "a" * 64, 1)}),
            )

            resumed = TrialJournal(path)
            self.assertEqual(resumed.completed_keys, frozenset())
            resumed.append(
                {**base, "session_id": "S02", "trial_status": "completed"}
            )
            self.assertEqual(
                resumed.completed_keys,
                frozenset({("P01", "a" * 64, 1)}),
            )
            with self.assertRaisesRegex(JournalError, "terminal 시행"):
                resumed.append(
                    {**base, "session_id": "S03", "trial_status": "completed"}
                )

            reread = TrialJournal(path)
            self.assertEqual(
                reread.completed_keys,
                frozenset({("P01", "a" * 64, 1)}),
            )

    def test_aborted_main_trial_is_terminal_and_cannot_be_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            base = {
                "participant_id": "P01",
                "session_id": "P01-CAL-01",
                "config_sha256": "a" * 64,
                "trial_id": 1,
                "phase": "calibration",
            }
            journal = TrialJournal(path)
            journal.append({**base, "trial_status": "aborted"})
            self.assertEqual(journal.completed_keys, frozenset())
            self.assertEqual(
                journal.terminal_keys,
                frozenset({("P01", "a" * 64, 1)}),
            )
            with self.assertRaisesRegex(JournalError, "terminal 시행"):
                journal.append(
                    {**base, "session_id": "P01-CAL-02", "trial_status": "completed"}
                )
            self.assertEqual(TrialJournal(path).terminal_keys, journal.terminal_keys)

    def test_truncated_last_record_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            path.write_text('{"participant_id":"P01"}', encoding="utf-8")
            with self.assertRaises(JournalError):
                TrialJournal(path)


if __name__ == "__main__":
    unittest.main()
