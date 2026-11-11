from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment_runner import TrialJournal, build_balanced_schedule
from response_gui import (
    GitState,
    ResponseGUIBlockedError,
    ScreenMetrics,
    VerifiedTrial,
    _assert_repository_stage_ready,
    build_trial_record,
    prepare_collection,
    remaining_trials,
    run_gui,
    session_history_start_utc,
    validate_screen_metrics,
)
from stimulus_renderer import (
    RENDERER_ID,
    compile_practice_stimulus_manifest,
    compile_stimulus_manifest,
    compile_validation_stimulus_manifest,
    render_stimulus_png,
    verify_renderer_environment,
)
from test_validation_design import _calibration
from validation_design import build_validation_plan
from experiment_design import geometric_validation_widths, pixels_to_visual_angle_deg


def human_config() -> dict:
    display_width_mm = 40.0
    resolution_width_px = 80
    viewing_distance_mm = 600.0
    calibration_widths = (2, 4, 6)
    config = {
        "schema_version": "1.0",
        "config_version": "GUI_TEST_V1",
        "status": "preregistered",
        "human_data_collection_allowed": True,
        "task": {
            "type": "4afc_odd_one_out",
            "layout": "2x2",
            "choice_count": 4,
            "practice_feedback": True,
            "main_feedback": False,
        },
        "observer_profile": {
            "participant_id": "P_GUI_01",
            "cvd_type": "deutan",
            "severity": 0.8,
            "severity_source": "independent session",
            "severity_session_id": "SEV_GUI_01",
            "fixed_lapse": 0.02,
            "lapse_source": "preregistered value",
            "lapse_session_id": "LAP_GUI_01",
        },
        "display": {
            "model": "TEST_DISPLAY",
            "width_mm": display_width_mm,
            "resolution_width_px": resolution_width_px,
            "resolution_height_px": 80,
            "os_scale_percent": 100,
            "app_scale_percent": 100,
            "viewing_distance_mm": viewing_distance_mm,
            "background_rgb": [117, 117, 117],
            "color_profile": "sRGB IEC61966-2.1",
            "colorimetry_policy": "measured-display-colorimetry",
            "calibration_id": "CAL_GUI_01",
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
            "length_px": 16,
            "orientation_deg": 0.0,
            "cap": "butt",
            "antialiasing": "disabled",
            "renderer": RENDERER_ID,
            "pixel_snapping": "integer-centers",
        },
        "coordinates": {
            "positions_px": [[20, 20], [60, 20], [20, 60], [60, 60]],
            "center_spacing_px": 40,
        },
        "color_design": {
            "stimulus_family_id": "GUI_LINE_TEST",
            "midpoint_lab": [60.0, 8.0, 8.0],
            "direction_lab": [0.0, 1.0, 0.35],
            "symmetric_about_midpoint": True,
            "gamut_policy": "reject",
            "rgb_quantization": "nearest_8bit",
            "target_absolute_tolerance_delta_e00": 0.07,
            "target_relative_tolerance": 0.03,
            "search_grid_steps": 1024,
        },
        "calibration": {
            "widths": [
                {
                    "width_px": width,
                    "width_deg": pixels_to_visual_angle_deg(
                        width,
                        display_width_mm=display_width_mm,
                        horizontal_resolution_px=resolution_width_px,
                        viewing_distance_mm=viewing_distance_mm,
                    ),
                }
                for width in calibration_widths
            ],
            "trials_per_width": [96, 192, 96],
            "session_block_count": 4,
        },
        "validation": {
            "candidates": [
                {
                    "geometric_target_px": target,
                    "rendered_width_px": width,
                    "width_deg": pixels_to_visual_angle_deg(
                        width,
                        display_width_mm=display_width_mm,
                        horizontal_resolution_px=resolution_width_px,
                        viewing_distance_mm=viewing_distance_mm,
                    ),
                }
                for target, width in zip(
                    geometric_validation_widths(calibration_widths), (3, 5)
                )
            ],
            "independent_blocks": 3,
            "delta_e00_level_count": 8,
            "trials_per_block": 64,
            "randomization": "balanced-position-and-odd-color-within-block",
        },
        "sampling": {
            "delta_e00_levels": [0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 7.9, 11.1, 15.6],
            "randomization": "balanced-mixed-width-session-blocks-within-block-shuffle",
            "practice_trials": 6,
        },
        "model_plan": {
            "models": ["B192", "B384", "C192", "D384"],
            "include_szafir_transfer": True,
            "szafir_probability_coordinate": 0.5,
        },
        "privacy": {
            "approved": True,
            "approval_id": "RNE-GUI-TEST",
            "approved_at_utc": "2026-09-14T00:00:00Z",
            "consent_version": "v1",
            "pseudonymization": "random participant codes only",
            "storage_policy": "restricted test storage",
        },
        "journal": {
            "storage_policy": "repository-relative-gitignored",
            "root_repository_relative": "data/experiment/raw",
            "phase_files": {
                "practice": "P_GUI_01-practice.jsonl",
                "calibration": "P_GUI_01-calibration.jsonl",
                "validation": "P_GUI_01-validation.jsonl",
            },
            "session_ids": {
                "practice": "P_GUI_01-PRACTICE-01",
                "calibration": {
                    "1": "P_GUI_01-CAL-01",
                    "2": "P_GUI_01-CAL-02",
                    "3": "P_GUI_01-CAL-03",
                    "4": "P_GUI_01-CAL-04",
                },
                "validation": {
                    "V01": "P_GUI_01-VAL-01",
                    "V02": "P_GUI_01-VAL-02",
                    "V03": "P_GUI_01-VAL-03",
                },
            },
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
    }
    return config


def trial_row(trial_id: int, digest: str, commit: str, *, phase: str = "practice") -> dict:
    correct = trial_id % 4 + 1
    labels = ["a"] * 4
    labels[correct - 1] = "b"
    return {
        "trial_id": trial_id,
        "phase": phase,
        "session_block": 1,
        "width_index": 2,
        "stroke_width_px": 4,
        "stroke_width_deg": 0.10,
        "delta_e00_level_index": 6,
        "target_delta_e00_simulated": 2.0,
        "correct_position": correct,
        "odd_color": "b",
        "position_colors": labels,
        "config_version": "GUI_TEST_V1",
        "config_sha256": digest,
        "git_commit": commit,
        "random_seed": 7,
        "stimulus_family_id": "GUI_LINE_TEST",
        "participant_id": "P_GUI_01",
        "cvd_type": "deutan",
        "machado_severity": 0.8,
        "severity_source": "independent session",
        "severity_session_id": "SEV_GUI_01",
        "fixed_lapse": 0.02,
        "lapse_source": "preregistered value",
        "lapse_session_id": "LAP_GUI_01",
        "displayed_rgb_a": [220, 50, 50],
        "displayed_rgb_b": [60, 160, 60],
        "displayed_lab_a": [50.0, 40.0, 20.0],
        "displayed_lab_b": [55.0, -30.0, 20.0],
        "simulated_rgb_a": [120.0, 100.0, 50.0],
        "simulated_rgb_b": [122.0, 101.0, 50.0],
        "simulated_lab_a": [45.0, 3.0, 15.0],
        "simulated_lab_b": [46.0, 3.2, 15.0],
        "delta_e00_original": 42.0,
        "delta_e00_simulated": 2.05,
    }


def bare_verified(row: dict) -> VerifiedTrial:
    return VerifiedTrial(
        row=row,
        png_path=Path("trial.png"),
        metadata_path=Path("trial.pixels.json"),
        metadata_repository_path="stimuli/trial.pixels.json",
        png_sha256="1" * 64,
        decoded_rgb_sha256="2" * 64,
    )


class RepositoryReadinessGateTests(unittest.TestCase):
    def test_calibration_and_validation_use_their_own_stage_gate(self) -> None:
        report = {
            "ready_for_calibration_collection": True,
            "ready_for_validation_collection": False,
            "blocked": {
                "calibration_collection": [],
                "validation_collection": [
                    {"check_id": "validation_plan_ready", "message": "plan missing"}
                ],
            },
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(report), stderr=""
        )
        with patch("response_gui.subprocess.run", return_value=completed):
            _assert_repository_stage_ready("calibration", Path.cwd())
            _assert_repository_stage_ready("practice", Path.cwd())
            with self.assertRaisesRegex(
                ResponseGUIBlockedError, "validation_plan_ready"
            ):
                _assert_repository_stage_ready("validation", Path.cwd())

    def test_invalid_readiness_output_fails_closed(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not-json", stderr=""
        )
        with patch("response_gui.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(ResponseGUIBlockedError, "JSON"):
                _assert_repository_stage_ready("calibration", Path.cwd())


class ScreenGateTests(unittest.TestCase):
    def test_exact_physical_screen_and_scale_pass(self) -> None:
        validate_screen_metrics(
            human_config(), ScreenMetrics(
                80, 80, 100.0, True, "test", "Windows test fixture", "CPython/Tk test fixture"
            )
        )

    def test_exact_150_percent_scale_passes_without_rescaling_pixels(self) -> None:
        config = human_config()
        config["display"]["os_scale_percent"] = 150.0
        validate_screen_metrics(
            config,
            ScreenMetrics(
                80,
                80,
                150.0,
                True,
                "test",
                "Windows test fixture",
                "CPython/Tk test fixture",
            ),
        )

    def test_resolution_scale_and_dpi_mismatches_block(self) -> None:
        cases = [
            ScreenMetrics(81, 80, 100.0, True, "test", "Windows test fixture", "CPython/Tk test fixture"),
            ScreenMetrics(80, 80, 125.0, True, "test", "Windows test fixture", "CPython/Tk test fixture"),
            ScreenMetrics(80, 80, 100.0, False, "test", "Windows test fixture", "CPython/Tk test fixture"),
        ]
        for metrics in cases:
            with self.subTest(metrics=metrics), self.assertRaises(ResponseGUIBlockedError):
                validate_screen_metrics(human_config(), metrics)


class TrialRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.digest = "a" * 64
        self.row = trial_row(1001, self.digest, "b" * 40)
        self.verified = bare_verified(self.row)

    def test_completed_and_aborted_records_are_schema_complete(self) -> None:
        completed = build_trial_record(
            self.verified,
            config=human_config(),
            config_sha256=self.digest,
            stimulus_manifest_sha256="d" * 64,
            session_id="PRACTICE_01",
            phase="practice",
            onset_utc="2026-09-14T00:00:00.000000Z",
            onset_monotonic_ns=123456,
            render_time_ms=2.5,
            response_position=self.row["correct_position"],
            response_time_ms=480.0,
        )
        self.assertEqual(completed["trial_status"], "completed")
        self.assertTrue(completed["is_correct"])
        self.assertEqual(completed["stimulus_onset_monotonic_ns"], 123456)
        self.assertIsNone(completed["validation_plan_sha256"])

        aborted = build_trial_record(
            self.verified,
            config=human_config(),
            config_sha256=self.digest,
            stimulus_manifest_sha256="d" * 64,
            session_id="PRACTICE_01",
            phase="practice",
            onset_utc="2026-09-14T00:00:00.000000Z",
            onset_monotonic_ns=123456,
            render_time_ms=2.5,
            abort_reason="participant_escape",
        )
        self.assertEqual(aborted["trial_status"], "aborted")
        self.assertIsNone(aborted["response_position"])
        self.assertIsNone(aborted["response_time_ms"])

    def test_validation_record_is_bound_to_frozen_plan_hash(self) -> None:
        row = trial_row(385, self.digest, "b" * 40, phase="validation")
        row.update(
            validation_plan_sha256="c" * 64,
            validation_block_id="V01",
            validation_level_index=1,
            within_block_index=1,
        )
        record = build_trial_record(
            bare_verified(row),
            config=human_config(),
            config_sha256=self.digest,
            stimulus_manifest_sha256="d" * 64,
            session_id="VALIDATION_01",
            phase="validation",
            validation_plan_sha256="c" * 64,
            onset_utc="2026-09-14T00:00:00.000000Z",
            onset_monotonic_ns=123456,
            render_time_ms=1.0,
            response_position=1,
            response_time_ms=400.0,
        )
        self.assertEqual(record["validation_plan_sha256"], "c" * 64)
        self.assertEqual(record["validation_block_id"], "V01")
        self.assertEqual(record["validation_level_index"], 1)

    def test_journal_resume_uses_global_config_trial_identity(self) -> None:
        record = build_trial_record(
            self.verified,
            config=human_config(),
            config_sha256=self.digest,
            stimulus_manifest_sha256="d" * 64,
            session_id="PRACTICE_01",
            phase="practice",
            onset_utc="2026-09-14T00:00:00.000000Z",
            onset_monotonic_ns=123456,
            render_time_ms=1.0,
            response_position=1,
            response_time_ms=400.0,
        )
        with tempfile.TemporaryDirectory() as folder:
            journal = TrialJournal(Path(folder) / "trials.jsonl")
            journal.append(record)
            plan = type(
                "Plan",
                (),
                {
                    "participant_id": "P_GUI_01",
                    "config_sha256": self.digest,
                    "trials": (self.verified, bare_verified(trial_row(1002, self.digest, "b" * 40))),
                },
            )()
            left = remaining_trials(plan, TrialJournal(journal.path))
            self.assertEqual([item.row["trial_id"] for item in left], [1002])

    def test_remaining_trials_retries_only_aborted_practice(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            practice = TrialJournal(Path(folder) / "practice.jsonl")
            base = {
                "participant_id": "P_GUI_01",
                "session_id": "P_GUI_01-PRACTICE-01",
                "config_sha256": self.digest,
                "trial_id": 1001,
            }
            practice.append({**base, "phase": "practice", "trial_status": "aborted"})
            plan = type(
                "Plan",
                (),
                {
                    "participant_id": "P_GUI_01",
                    "config_sha256": self.digest,
                    "trials": (self.verified,),
                },
            )()
            self.assertEqual(len(remaining_trials(plan, practice)), 1)

            main = TrialJournal(Path(folder) / "calibration.jsonl")
            main.append(
                {
                    **base,
                    "session_id": "P_GUI_01-CAL-01",
                    "phase": "calibration",
                    "trial_status": "aborted",
                }
            )
            self.assertEqual(remaining_trials(plan, main), ())

    def test_session_restart_preserves_first_utc_and_rejects_id_splicing(self) -> None:
        record = build_trial_record(
            self.verified,
            config=human_config(),
            config_sha256=self.digest,
            stimulus_manifest_sha256="d" * 64,
            session_id="P_GUI_01-PRACTICE-01",
            phase="practice",
            onset_utc="2026-09-14T00:00:00.000000Z",
            onset_monotonic_ns=123456,
            render_time_ms=1.0,
            abort_reason="participant_escape",
        )
        plan = type(
            "Plan",
            (),
                {
                    "config": human_config(),
                    "participant_id": "P_GUI_01",
                    "config_sha256": self.digest,
                    "phase": "practice",
                    "session_block": None,
                    "validation_block_id": None,
                    "trials": (self.verified,),
                },
        )()
        with tempfile.TemporaryDirectory() as folder:
            journal = TrialJournal(Path(folder) / "trials.jsonl")
            journal.append(record)
            started = session_history_start_utc(
                plan, journal, "P_GUI_01-PRACTICE-01"
            )
            self.assertEqual(started.isoformat(), "2026-09-14T00:00:00+00:00")
            with self.assertRaisesRegex(ResponseGUIBlockedError, "동결된 현재 block"):
                session_history_start_utc(plan, journal, "P_GUI_01-PRACTICE-02")

    def test_session_id_reused_for_another_block_is_rejected_before_tk(self) -> None:
        config = human_config()
        plan = type(
            "Plan",
            (),
            {
                "config": config,
                "participant_id": "P_GUI_01",
                "config_sha256": self.digest,
                "phase": "calibration",
                "session_block": 2,
                "validation_block_id": None,
                "trials": (),
            },
        )()
        with tempfile.TemporaryDirectory() as folder:
            journal = TrialJournal(Path(folder) / "trials.jsonl")
            journal.append(
                {
                    "participant_id": "P_GUI_01",
                    "session_id": "P_GUI_01-CAL-02",
                    "config_sha256": self.digest,
                    "trial_id": 1,
                    "phase": "calibration",
                    "session_block": 1,
                    "validation_block_id": None,
                    "timestamp_utc": "2026-09-14T00:00:00Z",
                }
            )
            with self.assertRaisesRegex(ResponseGUIBlockedError, "다른 block"):
                session_history_start_utc(plan, journal, "P_GUI_01-CAL-02")

    def test_run_gui_rejects_replacement_journal_and_existing_main_abort_before_tk(self) -> None:
        config = human_config()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            canonical = (
                root
                / config["journal"]["root_repository_relative"]
                / config["journal"]["phase_files"]["calibration"]
            )
            plan = type(
                "Plan",
                (),
                {
                    "config": config,
                    "participant_id": "P_GUI_01",
                    "config_sha256": self.digest,
                    "phase": "calibration",
                    "session_block": 1,
                    "validation_block_id": None,
                    "trials": (),
                    "repo_root": root,
                },
            )()
            with self.assertRaisesRegex(ResponseGUIBlockedError, "canonical 경로"):
                run_gui(
                    plan,
                    journal_path=root / "replacement.jsonl",
                    session_id="P_GUI_01-CAL-01",
                )

            journal = TrialJournal(canonical)
            journal.append(
                {
                    "participant_id": "P_GUI_01",
                    "session_id": "P_GUI_01-CAL-01",
                    "config_sha256": self.digest,
                    "trial_id": 1,
                    "phase": "calibration",
                    "trial_status": "aborted",
                    "session_block": 1,
                    "validation_block_id": None,
                    "timestamp_utc": "2026-09-14T00:00:00Z",
                }
            )
            ignored = type("Result", (), {"returncode": 0})()
            with (
                patch("response_gui.subprocess.run", return_value=ignored),
                self.assertRaisesRegex(ResponseGUIBlockedError, "중단 기록"),
            ):
                run_gui(
                    plan,
                    journal_path=canonical,
                    session_id="P_GUI_01-CAL-01",
                )


class CollectionPreparationTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path, Path, GitState, ScreenMetrics]:
        config = human_config()
        # The tiny 80px fixture uses a coarse, deliberately fast colour grid.
        # Keep its known quantization tolerance wide enough that this test
        # exercises GUI manifest validation rather than colour-pair search.
        config["color_design"]["target_absolute_tolerance_delta_e00"] = 0.30
        config_path = root / "config.json"
        raw = (json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        config_path.write_bytes(raw)
        digest = __import__("hashlib").sha256(raw).hexdigest()
        commit = "b" * 40
        stimulus_dir = root / "stimuli"
        stimulus_dir.mkdir()
        schedule = build_balanced_schedule(
            config,
            7,
            config_sha256=digest,
            git_commit=commit,
        )
        calibration = compile_stimulus_manifest(config, schedule)
        # Practice collection uses this manifest only as an immutable source of
        # IDs/color pairs.  The dedicated calibration GUI test exercises its
        # raster files separately.
        calibration["png_files_written"] = True
        calibration_path = root / "calibration.manifest.json"
        calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
        practice = compile_practice_stimulus_manifest(
            config_path,
            calibration_path,
            output_dir=stimulus_dir,
        )
        practice_path = stimulus_dir / "practice.manifest.json"
        practice_path.write_text(json.dumps(practice), encoding="utf-8")
        return (
            config_path,
            practice_path,
            calibration_path,
            GitState(commit, True, True),
            ScreenMetrics(80, 80, 100.0, True, "test", "Windows test fixture", "CPython/Tk test fixture"),
        )

    def test_practice_requires_verified_artifacts_and_disjoint_calibration_ids(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, practice, calibration, git_state, screen = self._write_fixture(root)
            plan = prepare_collection(
                config_path=config,
                manifest_path=practice,
                phase="practice",
                session_block=None,
                screen_metrics=screen,
                git_state=git_state,
                repo_root=root,
                calibration_manifest_path=calibration,
            )
            self.assertEqual(len(plan.trials), 6)
            self.assertEqual(plan.phase, "practice")

            first_png = plan.trials[0].png_path
            first_png.write_bytes(first_png.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ResponseGUIBlockedError, "재검증 실패"):
                prepare_collection(
                    config_path=config,
                    manifest_path=practice,
                    phase="practice",
                    session_block=None,
                    screen_metrics=screen,
                    git_state=git_state,
                    repo_root=root,
                    calibration_manifest_path=calibration,
                )

    def test_validation_rejects_a_practice_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, practice, _calibration, git_state, screen = self._write_fixture(root)
            with self.assertRaisesRegex(ResponseGUIBlockedError, "validation"):
                prepare_collection(
                    config_path=config,
                    manifest_path=practice,
                    phase="validation",
                    session_block=None,
                    screen_metrics=screen,
                    git_state=git_state,
                    repo_root=root,
                )

    def test_validation_manifest_plan_and_calibration_hashes_are_all_verified(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = human_config()
            config["color_design"]["target_absolute_tolerance_delta_e00"] = 0.30
            config["color_design"]["target_relative_tolerance"] = 0.05
            config["color_design"]["search_grid_steps"] = 512
            config_path = root / "config.json"
            config_raw = (
                json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            config_path.write_bytes(config_raw)
            config_sha256 = hashlib.sha256(config_raw).hexdigest()

            calibration = _calibration(config_sha256, config=config)
            calibration["source"].update(
                config_version=config["config_version"],
                stimulus_family_id=config["color_design"]["stimulus_family_id"],
                renderer_id=RENDERER_ID,
                git_commit="b" * 40,
            )
            participant = calibration["participants"][0]
            observer = config["observer_profile"]
            participant.update(
                participant_id=observer["participant_id"],
                cvd_type=observer["cvd_type"],
                machado_severity=observer["severity"],
                severity_source=observer["severity_source"],
                severity_session_id=observer["severity_session_id"],
                fixed_lapse=observer["fixed_lapse"],
                lapse_source=observer["lapse_source"],
                lapse_session_id=observer["lapse_session_id"],
            )
            calibration_path = root / "calibration-fits.json"
            calibration_raw = (
                json.dumps(calibration, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            calibration_path.write_bytes(calibration_raw)
            plan = build_validation_plan(
                config_path,
                calibration_path,
                seed=17,
                first_trial_id=385,
                allow_synthetic=True,
            )
            plan_path = root / "validation-plan.json"
            plan_path.write_text(
                json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            raster_dir = root / "validation-pixels"
            manifest = compile_validation_stimulus_manifest(
                config_path,
                calibration_path,
                plan_path,
                output_dir=raster_dir,
            )
            manifest_path = raster_dir / "validation.manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            prepared = prepare_collection(
                config_path=config_path,
                manifest_path=manifest_path,
                phase="validation",
                session_block=None,
                validation_block_id="V01",
                validation_plan_path=plan_path,
                calibration_model_fits_path=calibration_path,
                screen_metrics=ScreenMetrics(
                    80, 80, 100.0, True, "test", "Windows test fixture", "CPython/Tk test fixture"
                ),
                git_state=GitState("b" * 40, True, True),
                repo_root=root,
            )
            self.assertEqual(len(prepared.trials), 64)
            self.assertEqual(prepared.validation_block_id, "V01")
            self.assertEqual(prepared.validation_plan_sha256, plan["plan_sha256"])

            plan_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ResponseGUIBlockedError, "validation plan"):
                prepare_collection(
                    config_path=config_path,
                    manifest_path=manifest_path,
                    phase="validation",
                    session_block=None,
                    validation_block_id="V01",
                    validation_plan_path=plan_path,
                    calibration_model_fits_path=calibration_path,
                    screen_metrics=ScreenMetrics(
                        80, 80, 100.0, True, "test", "Windows test fixture", "CPython/Tk test fixture"
                    ),
                    git_state=GitState("b" * 40, True, True),
                    repo_root=root,
                )


if __name__ == "__main__":
    unittest.main()
