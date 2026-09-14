import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from analyze_calibration import (
    AnalysisBlockedError,
    _canonical_sha256,
    _validate_calibration_manifest,
    analyze_calibration,
    main,
    validate_calibration_artifact,
)
from color_spaces import srgb_to_lab
from distance import delta_e_2000
from experiment_design import (
    geometric_validation_widths,
    pixels_to_visual_angle_deg,
)
from experiment_runner import build_balanced_schedule
from implementation_provenance import implementation_hashes
from machado import simulate_color_float
from psychometrics import direct_c80_probability
from response_gui import (
    GitState,
    ScreenMetrics,
    build_trial_record,
    prepare_collection,
)
from stimulus_renderer import compile_stimulus_manifest
from test_response_gui import human_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = PROJECT_ROOT / "data" / "contracts" / "experiment_config.example.json"


def _pair_metrics(offset: int, *, cvd_type: str, severity: float) -> dict:
    rgb_a = (128, 128, 128)
    rgb_b = (128 + offset, 128, 128)
    displayed_lab_a = srgb_to_lab(rgb_a)
    displayed_lab_b = srgb_to_lab(rgb_b)
    simulated_rgb_a = simulate_color_float(rgb_a, cvd_type, severity)
    simulated_rgb_b = simulate_color_float(rgb_b, cvd_type, severity)
    simulated_lab_a = srgb_to_lab(simulated_rgb_a)
    simulated_lab_b = srgb_to_lab(simulated_rgb_b)
    return {
        "rgb_a": list(rgb_a),
        "rgb_b": list(rgb_b),
        "displayed_lab_a": list(displayed_lab_a),
        "displayed_lab_b": list(displayed_lab_b),
        "simulated_rgb_a": list(simulated_rgb_a),
        "simulated_rgb_b": list(simulated_rgb_b),
        "simulated_lab_a": list(simulated_lab_a),
        "simulated_lab_b": list(simulated_lab_b),
        "delta_e00_original": float(delta_e_2000(displayed_lab_a, displayed_lab_b)),
        "delta_e00_simulated": float(delta_e_2000(simulated_lab_a, simulated_lab_b)),
    }


def _write_synthetic_fixture(
    directory: Path,
    *,
    drop_last: bool = False,
    corrupt_lapse_source: bool = False,
    viewing_distance_mm: float = 600.0,
) -> tuple[Path, Path]:
    config = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    config["display"]["viewing_distance_mm"] = viewing_distance_mm
    conversion = {
        "display_width_mm": config["display"]["width_mm"],
        "horizontal_resolution_px": config["display"]["resolution_width_px"],
        "viewing_distance_mm": viewing_distance_mm,
    }
    for row in config["calibration"]["widths"]:
        row["width_deg"] = pixels_to_visual_angle_deg(row["width_px"], **conversion)
    for row in config["validation"]["candidates"]:
        row["width_deg"] = pixels_to_visual_angle_deg(row["rendered_width_px"], **conversion)

    profile = config["observer_profile"]
    pairs = [
        _pair_metrics(offset, cvd_type=profile["cvd_type"], severity=profile["severity"])
        for offset in (2, 3, 4, 5, 7, 9, 12, 16, 22, 30, 40, 55)
    ]
    config["sampling"]["delta_e00_levels"] = [
        pair["delta_e00_simulated"] for pair in pairs
    ]
    config["color_design"]["target_absolute_tolerance_delta_e00"] = 1e-6
    config["color_design"]["target_relative_tolerance"] = 0.0
    config["line"]["renderer"] = "SYNTHETIC_RENDERER_FIXTURE"
    config_path = directory / "synthetic-config.json"
    config_bytes = (
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    config_path.write_bytes(config_bytes)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    records = []
    trial_id = 1
    reference_deg = config["calibration"]["widths"][1]["width_deg"]
    for width_index, width in enumerate(config["calibration"]["widths"], start=1):
        true_c80 = 2.45 * (width["width_deg"] / reference_deg) ** -0.42
        for level_index, pair in enumerate(pairs, start=1):
            probability = direct_c80_probability(
                pair["delta_e00_simulated"], true_c80, 0.32, profile["fixed_lapse"]
            )
            successes = int(round(probability * 8))
            for repeat in range(8):
                correct_position = repeat % 4 + 1
                odd_color = "a" if repeat < 4 else "b"
                base_color = "b" if odd_color == "a" else "a"
                position_colors = [base_color] * 4
                position_colors[correct_position - 1] = odd_color
                position_rgbs = [
                    pair["rgb_a"] if color == "a" else pair["rgb_b"]
                    for color in position_colors
                ]
                is_correct = repeat < successes
                response_position = correct_position if is_correct else correct_position % 4 + 1
                record = {
                    "schema_version": "0.1-draft",
                    "data_origin": "synthetic",
                    "config_version": config["config_version"],
                    "config_sha256": config_sha256,
                    "git_commit": "a" * 40,
                    "stimulus_family_id": config["color_design"]["stimulus_family_id"],
                    "renderer_id": config["line"]["renderer"],
                    "stimulus_png_sha256": hashlib.sha256(
                        f"synthetic-png-{trial_id}".encode("ascii")
                    ).hexdigest(),
                    "stimulus_decoded_rgb_sha256": hashlib.sha256(
                        f"synthetic-rgb-{trial_id}".encode("ascii")
                    ).hexdigest(),
                    "stimulus_pixel_metadata_path": (
                        f"data/synthetic/fixture/trial-{trial_id:04d}.pixels.json"
                    ),
                    "participant_id": profile["participant_id"],
                    "session_id": f"S_CALIBRATION_{repeat // 2 + 1}",
                    "session_block": repeat // 2 + 1,
                    "staircase_id": None,
                    "trial_id": trial_id,
                    "phase": "calibration",
                    "trial_status": "completed",
                    "abort_reason": None,
                    "timestamp_utc": "2026-09-14T00:00:00Z",
                    "width_index": width_index,
                    "stroke_width_px": width["width_px"],
                    "stroke_width_deg": width["width_deg"],
                    "delta_e00_level_index": level_index,
                    "target_delta_e00_simulated": pair["delta_e00_simulated"],
                    "displayed_rgb_a": pair["rgb_a"],
                    "displayed_rgb_b": pair["rgb_b"],
                    "displayed_lab_a": pair["displayed_lab_a"],
                    "displayed_lab_b": pair["displayed_lab_b"],
                    "simulated_rgb_a": pair["simulated_rgb_a"],
                    "simulated_rgb_b": pair["simulated_rgb_b"],
                    "simulated_lab_a": pair["simulated_lab_a"],
                    "simulated_lab_b": pair["simulated_lab_b"],
                    "delta_e00_original": pair["delta_e00_original"],
                    "delta_e00_simulated": pair["delta_e00_simulated"],
                    "correct_position": correct_position,
                    "position_colors": position_colors,
                    "position_rgbs": position_rgbs,
                    "odd_color": odd_color,
                    "response_position": response_position,
                    "is_correct": is_correct,
                    "response_time_ms": 750.0,
                    "render_time_ms": 2.0,
                    "cvd_type": profile["cvd_type"],
                    "machado_severity": profile["severity"],
                    "severity_source": profile["severity_source"],
                    "severity_session_id": profile["severity_session_id"],
                    "fixed_lapse": profile["fixed_lapse"],
                    "lapse_source": profile["lapse_source"],
                    "lapse_session_id": profile["lapse_session_id"],
                    "random_seed": 20260914,
                }
                records.append(record)
                trial_id += 1
    if drop_last:
        records.pop()
    if corrupt_lapse_source:
        records[120]["lapse_source"] = "DIFFERENT_SYNTHETIC_SOURCE"
    trials_path = directory / "synthetic-trials.jsonl"
    trials_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    return config_path, trials_path


def _write_mock_calibration_manifest_fixture(
    directory: Path,
) -> tuple[Path, dict, str, list[dict], dict]:
    """실제 파일 검증 호출 수/결박 로직만 빠르게 시험하는 288행 fixture."""

    config_path, trials_path = _write_synthetic_fixture(directory)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    records = [
        json.loads(line)
        for line in trials_path.read_text(encoding="utf-8").splitlines()
    ]
    png = directory / "shared.png"
    sidecar = directory / "shared.pixels.json"
    png.write_bytes(b"mock-png")
    sidecar.write_bytes(b'{"mock":true}\n')
    png_sha256 = hashlib.sha256(png.read_bytes()).hexdigest()
    decoded_sha256 = "b" * 64
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    renderer = {
        "renderer_verified": True,
        "renderer_id": config["line"]["renderer"],
        "implementation_version": "test-renderer-v1",
        "implementation_sha256": "c" * 64,
        "pillow_version": "test-pillow",
    }
    manifest_rows = []
    for record in records:
        record["stimulus_png_sha256"] = png_sha256
        record["stimulus_decoded_rgb_sha256"] = decoded_sha256
        record["stimulus_pixel_metadata_path"] = sidecar.name
        row = copy.deepcopy(record)
        row["raster_artifact"] = {
            "png": png.name,
            "pixel_metadata": sidecar.name,
            "png_sha256": png_sha256,
            "decoded_rgb_sha256": decoded_sha256,
            "pixel_metadata_sha256": sidecar_sha256,
        }
        manifest_rows.append(row)
    profile = config["observer_profile"]
    pair_fields = (
        "displayed_rgb_a",
        "displayed_rgb_b",
        "displayed_lab_a",
        "displayed_lab_b",
        "simulated_rgb_a",
        "simulated_rgb_b",
        "simulated_lab_a",
        "simulated_lab_b",
        "delta_e00_original",
        "delta_e00_simulated",
    )
    color_pairs = []
    for level_index, target in enumerate(
        config["sampling"]["delta_e00_levels"], start=1
    ):
        source = next(
            row
            for row in manifest_rows
            if row["delta_e00_level_index"] == level_index
        )
        color_pairs.append(
            {
                "target_delta_e00_simulated": target,
                **{field: copy.deepcopy(source[field]) for field in pair_fields},
            }
        )
    color_design = config["color_design"]
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "compiled_pixel_stimulus_manifest",
        "config_version": config["config_version"],
        "config_canonical_sha256": _canonical_sha256(config),
        "schedule_config_sha256": config_sha256,
        "git_commit": records[0]["git_commit"],
        "stimulus_family_id": config["color_design"]["stimulus_family_id"],
        "participant_id": profile["participant_id"],
        "cvd_type": profile["cvd_type"],
        "machado_severity": profile["severity"],
        "renderer": renderer,
        "human_response_collection_implemented": True,
        "color_generation": {
            "gamut_policy": color_design["gamut_policy"],
            "rgb_quantization": color_design["rgb_quantization"],
            "target_absolute_tolerance_delta_e00": color_design[
                "target_absolute_tolerance_delta_e00"
            ],
            "target_relative_tolerance": color_design["target_relative_tolerance"],
            "allowed_error_rule": "max(absolute_tolerance, relative_tolerance * target)",
            "search_grid_steps": color_design["search_grid_steps"],
            "unique_target_count": 12,
            "configured_target_count": 12,
            "development_partial_schedule": False,
        },
        "color_pairs": color_pairs,
        "trial_count": 288,
        "png_files_written": True,
        "trials": manifest_rows,
    }
    manifest_path = directory / "calibration-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for record in records:
        record["stimulus_manifest_sha256"] = manifest_sha256
    verification = {
        "png_sha256": png_sha256,
        "decoded_rgb_sha256": decoded_sha256,
    }
    return manifest_path, config, config_sha256, records, verification


def _write_human_production_fixture(
    directory: Path,
) -> tuple[Path, Path, Path, str, list[Path]]:
    """생성기→PNG→수집 레코드의 실제 production 경로를 통과한다."""

    config = human_config()
    config["display"]["width_mm"] = 40.0
    conversion = {
        "display_width_mm": config["display"]["width_mm"],
        "horizontal_resolution_px": config["display"]["resolution_width_px"],
        "viewing_distance_mm": config["display"]["viewing_distance_mm"],
    }
    calibration_widths = [2, 4, 6]
    config["calibration"]["widths"] = [
        {
            "width_px": width,
            "width_deg": pixels_to_visual_angle_deg(width, **conversion),
        }
        for width in calibration_widths
    ]
    geometric = geometric_validation_widths(calibration_widths)
    config["validation"]["candidates"] = [
        {
            "geometric_target_px": geometric[index],
            "rendered_width_px": width,
            "width_deg": pixels_to_visual_angle_deg(width, **conversion),
        }
        for index, width in enumerate((3, 5))
    ]
    config["sampling"]["delta_e00_levels"] = [
        0.5,
        0.7,
        0.9,
        1.2,
        1.6,
        2.1,
        2.8,
        3.7,
        4.9,
        6.5,
        8.6,
        11.4,
    ]
    config["color_design"]["target_absolute_tolerance_delta_e00"] = 0.30
    config["color_design"]["target_relative_tolerance"] = 0.05
    config["color_design"]["search_grid_steps"] = 512
    config_path = directory / "human-config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    git_commit = "d" * 40
    schedule = build_balanced_schedule(
        config,
        17,
        config_sha256=config_sha256,
        git_commit=git_commit,
    )
    stimulus_dir = directory / "stimuli"
    stimulus_dir.mkdir()
    manifest_payload = compile_stimulus_manifest(
        config,
        schedule,
        output_dir=stimulus_dir,
    )
    manifest_path = stimulus_dir / "calibration-manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    verified = []
    for block in range(1, 5):
        plan = prepare_collection(
            config_path=config_path,
            manifest_path=manifest_path,
            phase="calibration",
            session_block=block,
            screen_metrics=ScreenMetrics(
                80, 80, 100.0, True, "test", "Windows test fixture", "CPython/Tk test fixture"
            ),
            git_state=GitState(git_commit, True, True),
            repo_root=directory,
        )
        verified.extend(plan.trials)
    reference_deg = config["calibration"]["widths"][1]["width_deg"]
    cell_offsets: Counter[tuple[int, int]] = Counter()
    records = []
    for item in verified:
        row = item.row
        key = (int(row["stroke_width_px"]), int(row["delta_e00_level_index"]))
        true_c80 = 2.45 * (float(row["stroke_width_deg"]) / reference_deg) ** -0.42
        probability = direct_c80_probability(
            float(row["delta_e00_simulated"]),
            true_c80,
            0.32,
            config["observer_profile"]["fixed_lapse"],
        )
        successes = int(round(probability * 8))
        correct = cell_offsets[key] < successes
        cell_offsets[key] += 1
        correct_position = int(row["correct_position"])
        response_position = (
            correct_position if correct else correct_position % 4 + 1
        )
        records.append(
            build_trial_record(
                item,
                config=config,
                config_sha256=config_sha256,
                stimulus_manifest_sha256=manifest_sha256,
                session_id=config["journal"]["session_ids"]["calibration"][
                    str(int(row["session_block"]))
                ],
                phase="calibration",
                onset_utc="2026-09-14T00:00:00Z",
                onset_monotonic_ns=int(row["trial_id"]),
                render_time_ms=1.0,
                response_position=response_position,
                response_time_ms=500.0,
            )
        )
    trials_path = (
        directory
        / config["journal"]["root_repository_relative"]
        / config["journal"]["phase_files"]["calibration"]
    )
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    trials_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    png_paths = sorted(stimulus_dir.glob("*.png"))
    return config_path, manifest_path, trials_path, manifest_sha256, png_paths


class CalibrationAnalysisEndToEndTests(unittest.TestCase):
    def test_generated_human_288_pipeline_binds_manifest_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config, manifest, trials, manifest_sha, png_paths = (
                _write_human_production_fixture(directory)
            )
            original_manifest = manifest.read_bytes()
            with (
                patch(
                    "analyze_calibration._validate_calibration_manifest",
                    return_value=manifest_sha,
                ),
                patch("analyze_calibration._fit_all_models") as fit_models,
                self.assertRaisesRegex(AnalysisBlockedError, "1000"),
            ):
                analyze_calibration(
                    config,
                    trials,
                    calibration_manifest_path=manifest,
                    bootstrap_replicates=999,
                )
            fit_models.assert_not_called()
            # The test below audits the complete human raster/manifest path.
            # Keep bootstrap tiny here and test the production >=1000 guard
            # separately so this integrity test remains fast.
            with (
                patch("analyze_calibration.PROJECT_ROOT", directory),
                patch("analyze_calibration.MINIMUM_HUMAN_BOOTSTRAP_REPLICATES", 2),
            ):
                payload = analyze_calibration(
                    config,
                    trials,
                    calibration_manifest_path=manifest,
                    bootstrap_replicates=2,
                    bootstrap_seed=19,
                )
            self.assertEqual(payload["data_origin"], "human")
            self.assertEqual(
                payload["source"]["stimulus_manifest_sha256"], manifest_sha
            )
            self.assertEqual(
                payload["participants"][0]["calibration"]["completed_trial_count"],
                288,
            )

            replacement_trials = directory / "replacement-calibration.jsonl"
            replacement_trials.write_bytes(trials.read_bytes())
            with (
                patch("analyze_calibration.PROJECT_ROOT", directory),
                patch("analyze_calibration.MINIMUM_HUMAN_BOOTSTRAP_REPLICATES", 2),
                patch(
                    "analyze_calibration._validate_calibration_manifest",
                    return_value=manifest_sha,
                ),
                self.assertRaisesRegex(AnalysisBlockedError, "canonical journal"),
            ):
                analyze_calibration(
                    config,
                    replacement_trials,
                    calibration_manifest_path=manifest,
                    bootstrap_replicates=2,
                )

            manifest.write_bytes(original_manifest + b" ")
            with (
                patch("analyze_calibration.PROJECT_ROOT", directory),
                self.assertRaisesRegex(AnalysisBlockedError, "manifest 원문"),
            ):
                analyze_calibration(
                    config,
                    trials,
                    calibration_manifest_path=manifest,
                    bootstrap_replicates=2,
                )
            manifest.write_bytes(original_manifest)

            self.assertEqual(len(png_paths), 288)
            first_png = png_paths[0]
            original_png = first_png.read_bytes()
            first_png.write_bytes(original_png + b"tamper")
            with (
                patch("analyze_calibration.PROJECT_ROOT", directory),
                self.assertRaisesRegex(AnalysisBlockedError, "PNG/sidecar"),
            ):
                analyze_calibration(
                    config,
                    trials,
                    calibration_manifest_path=manifest,
                    bootstrap_replicates=2,
                )
            first_png.write_bytes(original_png)

            first_sidecar = first_png.with_name(
                first_png.stem + ".pixels.json"
            )
            first_sidecar.write_bytes(first_sidecar.read_bytes() + b" ")
            with (
                patch("analyze_calibration.PROJECT_ROOT", directory),
                self.assertRaisesRegex(AnalysisBlockedError, "sidecar raw hash"),
            ):
                analyze_calibration(
                    config,
                    trials,
                    calibration_manifest_path=manifest,
                    bootstrap_replicates=2,
                )

    def test_synthetic_fixture_runs_all_models_and_exports_only_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config, trials = _write_synthetic_fixture(directory)
            output = directory / "calibration-fits.json"
            exit_code = main(
                [
                    "--config",
                    str(config),
                    "--trials",
                    str(trials),
                    "--output",
                    str(output),
                    "--bootstrap-replicates",
                    "4",
                    "--bootstrap-seed",
                    "77",
                    "--allow-synthetic",
                ]
            )
            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact_type"], "calibration_model_fits")
            self.assertEqual(payload["analysis_scope"], "calibration_only_intermediate")
            self.assertFalse(payload["research_conclusion_allowed"])
            self.assertFalse(payload["final_thresholds_ready"])
            self.assertFalse(payload["contains_observed_held_out"])
            self.assertIsNone(payload["source"]["stimulus_manifest_sha256"])
            self.assertEqual(
                payload["implementation_hashes"],
                implementation_hashes("calibration_analysis"),
            )
            self.assertEqual(
                payload["analysis_script_sha256"],
                payload["implementation_hashes"]["src/analyze_calibration.py"],
            )
            participant = payload["participants"][0]
            self.assertEqual(participant["calibration"]["completed_trial_count"], 288)
            block_quality = participant["calibration"]["session_block_quality"]
            self.assertEqual(
                [row["session_block"] for row in block_quality], [1, 2, 3, 4]
            )
            self.assertEqual([row["completed_trials"] for row in block_quality], [72] * 4)
            self.assertEqual(len({row["session_id"] for row in block_quality}), 4)
            for row in block_quality:
                self.assertAlmostEqual(
                    row["accuracy"], row["correct_trials"] / row["completed_trials"]
                )
            sensitivity = participant["width_slope_sensitivity"]
            self.assertEqual(sensitivity["criterion"], "AIC")
            self.assertEqual(sensitivity["shared_w_parameter_count"], 4)
            self.assertEqual(sensitivity["free_w_parameter_count"], 6)
            self.assertAlmostEqual(
                sensitivity["delta_aic_free_minus_shared"],
                sensitivity["free_w_aic"] - sensitivity["shared_w_aic"],
            )
            self.assertEqual(
                sensitivity["comparison_resolved"],
                sensitivity["shared_w_converged"]
                and sensitivity["free_w_converged"],
            )
            self.assertFalse(sensitivity["primary_c96_vs_b96_affected"])
            self.assertEqual(
                [row["model"] for row in participant["models"]],
                ["B96", "B288", "C96", "D288"],
            )
            self.assertEqual(
                [row["training_trial_count"] for row in participant["models"]],
                [96, 288, 96, 288],
            )
            for model in participant["models"]:
                self.assertEqual(len(model["held_out_candidate_predictions"]), 2)
                self.assertEqual(
                    len(model["held_out_candidate_predictions"][0]["bootstrap_c80_samples"]),
                    model["bootstrap"]["successful_replicates"],
                )
                self.assertTrue(
                    all(
                        not prediction["observed_held_out_used"]
                        for prediction in model["held_out_candidate_predictions"]
                    )
                )
            b96 = participant["models"][0]
            c96 = participant["models"][2]
            pairing_id = "B96-C96-reference-width-parametric-v1"
            self.assertEqual(b96["bootstrap"]["replicate_pairing_id"], pairing_id)
            self.assertEqual(c96["bootstrap"]["replicate_pairing_id"], pairing_id)
            for b_prediction, c_prediction in zip(
                b96["held_out_candidate_predictions"],
                c96["held_out_candidate_predictions"],
                strict=True,
            ):
                self.assertEqual(b_prediction["bootstrap_pairing_id"], pairing_id)
                self.assertEqual(c_prediction["bootstrap_pairing_id"], pairing_id)
                point_ratio = c_prediction["c80_sim"] / b_prediction["c80_sim"]
                for b_sample, c_sample in zip(
                    b_prediction["bootstrap_c80_samples"],
                    c_prediction["bootstrap_c80_samples"],
                    strict=True,
                ):
                    self.assertAlmostEqual(c_sample, b_sample * point_ratio, places=8)

            tampered = copy.deepcopy(payload)
            tampered["participants"][0]["width_slope_sensitivity"][
                "delta_aic_free_minus_shared"
            ] += 1.0
            with self.assertRaisesRegex(AnalysisBlockedError, "delta AIC"):
                validate_calibration_artifact(tampered)

            tampered = copy.deepcopy(payload)
            sensitivity = tampered["participants"][0]["width_slope_sensitivity"]
            sensitivity["d288_beta_interpretation_eligible"] = not sensitivity[
                "d288_beta_interpretation_eligible"
            ]
            with self.assertRaisesRegex(AnalysisBlockedError, "D288 beta"):
                validate_calibration_artifact(tampered)

            tampered = copy.deepcopy(payload)
            tampered["participants"][0]["calibration"]["session_block_quality"][0][
                "accuracy"
            ] = 0.0
            with self.assertRaisesRegex(AnalysisBlockedError, "accuracy"):
                validate_calibration_artifact(tampered)

    def test_aborted_attempt_before_completed_retry_is_preserved_but_not_fitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config, trials = _write_synthetic_fixture(directory)
            records = [
                json.loads(line)
                for line in trials.read_text(encoding="utf-8").splitlines()
            ]
            aborted = copy.deepcopy(records[0])
            aborted.update(
                trial_status="aborted",
                abort_reason="participant_escape",
                response_position=None,
                is_correct=None,
                response_time_ms=None,
            )
            records.insert(0, aborted)
            trials.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                    for row in records
                ),
                encoding="utf-8",
                newline="\n",
            )
            payload = analyze_calibration(
                config,
                trials,
                bootstrap_replicates=2,
                allow_synthetic=True,
            )
            summary = payload["participants"][0]["calibration"]
            self.assertEqual(summary["completed_trial_count"], 288)
            self.assertEqual(summary["aborted_trial_count"], 1)
            self.assertEqual(payload["source"]["total_record_count"], 289)

    def test_synthetic_input_requires_explicit_opt_in_and_creates_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config, trials = _write_synthetic_fixture(directory)
            output = directory / "must-not-exist.json"
            self.assertEqual(
                main(
                    [
                        "--config",
                        str(config),
                        "--trials",
                        str(trials),
                        "--output",
                        str(output),
                    ]
                ),
                2,
            )
            self.assertFalse(output.exists())


class CalibrationAnalysisSafetyTests(unittest.TestCase):
    def test_human_input_requires_calibration_manifest_before_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config, trials = _write_synthetic_fixture(directory)
            records = [
                json.loads(line)
                for line in trials.read_text(encoding="utf-8").splitlines()
            ]
            for row in records:
                row["data_origin"] = "human"
            trials.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                    for row in records
                ),
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(AnalysisBlockedError, "calibration-manifest"):
                analyze_calibration(config, trials, bootstrap_replicates=2)

    def test_manifest_binds_all_288_rows_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            manifest, config, config_sha, records, verification = (
                _write_mock_calibration_manifest_fixture(directory)
            )
            renderer = json.loads(manifest.read_text(encoding="utf-8"))["renderer"]
            with (
                patch("analyze_calibration.PROJECT_ROOT", directory),
                patch(
                    "analyze_calibration.verify_renderer_environment",
                    return_value=renderer,
                ),
                patch(
                    "analyze_calibration.verify_stimulus_artifact",
                    return_value=verification,
                ) as artifact_check,
            ):
                actual = _validate_calibration_manifest(
                    manifest,
                    config=config,
                    config_sha256=config_sha,
                    records=records,
                    require_record_manifest_hash=True,
                )
            self.assertEqual(actual, hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(artifact_check.call_count, 288)

    def test_manifest_audits_aborted_attempt_before_completed_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            manifest, config, config_sha, records, verification = (
                _write_mock_calibration_manifest_fixture(directory)
            )
            renderer = json.loads(manifest.read_text(encoding="utf-8"))["renderer"]
            aborted = copy.deepcopy(records[0])
            aborted.update(
                trial_status="aborted",
                abort_reason="participant_escape",
                response_position=None,
                is_correct=None,
                response_time_ms=None,
            )
            records.insert(0, aborted)
            with (
                patch("analyze_calibration.PROJECT_ROOT", directory),
                patch(
                    "analyze_calibration.verify_renderer_environment",
                    return_value=renderer,
                ),
                patch(
                    "analyze_calibration.verify_stimulus_artifact",
                    return_value=verification,
                ) as artifact_check,
            ):
                actual = _validate_calibration_manifest(
                    manifest,
                    config=config,
                    config_sha256=config_sha,
                    records=records,
                    require_record_manifest_hash=True,
                )
            self.assertEqual(actual, hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(artifact_check.call_count, 289)

            aborted["stimulus_png_sha256"] = "0" * 64
            with (
                patch("analyze_calibration.PROJECT_ROOT", directory),
                patch(
                    "analyze_calibration.verify_renderer_environment",
                    return_value=renderer,
                ),
                patch(
                    "analyze_calibration.verify_stimulus_artifact",
                    return_value=verification,
                ),
                self.assertRaisesRegex(AnalysisBlockedError, "PNG 픽셀 hash"),
            ):
                _validate_calibration_manifest(
                    manifest,
                    config=config,
                    config_sha256=config_sha,
                    records=records,
                    require_record_manifest_hash=True,
                )

    def test_manifest_rejects_record_hash_and_immutable_field_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            manifest, config, config_sha, records, verification = (
                _write_mock_calibration_manifest_fixture(directory)
            )
            renderer = json.loads(manifest.read_text(encoding="utf-8"))["renderer"]
            records[0]["stimulus_manifest_sha256"] = "0" * 64
            with (
                patch("analyze_calibration.PROJECT_ROOT", directory),
                patch(
                    "analyze_calibration.verify_renderer_environment",
                    return_value=renderer,
                ),
                patch(
                    "analyze_calibration.verify_stimulus_artifact",
                    return_value=verification,
                ),
                self.assertRaisesRegex(AnalysisBlockedError, "manifest 원문"),
            ):
                _validate_calibration_manifest(
                    manifest,
                    config=config,
                    config_sha256=config_sha,
                    records=records,
                    require_record_manifest_hash=True,
                )

            records[0]["stimulus_manifest_sha256"] = hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest()
            records[0]["correct_position"] = records[0]["correct_position"] % 4 + 1
            with (
                patch("analyze_calibration.PROJECT_ROOT", directory),
                patch(
                    "analyze_calibration.verify_renderer_environment",
                    return_value=renderer,
                ),
                patch(
                    "analyze_calibration.verify_stimulus_artifact",
                    return_value=verification,
                ),
                self.assertRaisesRegex(AnalysisBlockedError, "correct_position"),
            ):
                _validate_calibration_manifest(
                    manifest,
                    config=config,
                    config_sha256=config_sha,
                    records=records,
                    require_record_manifest_hash=True,
                )

    def test_validation_row_is_rejected_before_calibration_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config, trials = _write_synthetic_fixture(directory)
            records = [
                json.loads(line)
                for line in trials.read_text(encoding="utf-8").splitlines()
            ]
            held_out = copy.deepcopy(records[0])
            held_out["trial_id"] = 289
            held_out["phase"] = "validation"
            records.append(held_out)
            trials.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in records
                ),
                encoding="utf-8",
                newline="\n",
            )

            with self.assertRaisesRegex(AnalysisBlockedError, "validation/held-out"):
                analyze_calibration(
                    config,
                    trials,
                    bootstrap_replicates=2,
                    allow_synthetic=True,
                )

    def test_trial_id_is_global_within_participant_and_config_not_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config, trials = _write_synthetic_fixture(directory)
            records = [
                json.loads(line)
                for line in trials.read_text(encoding="utf-8").splitlines()
            ]
            duplicate = copy.deepcopy(records[0])
            duplicate["session_id"] = "S_DIFFERENT_SESSION"
            records.append(duplicate)
            trials.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in records
                ),
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(AnalysisBlockedError, "participant/config"):
                analyze_calibration(
                    config,
                    trials,
                    bootstrap_replicates=2,
                    allow_synthetic=True,
                )

    def test_exact_three_by_ninety_six_completed_trials_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            config, trials = _write_synthetic_fixture(Path(directory_name), drop_last=True)
            with self.assertRaisesRegex(AnalysisBlockedError, "정확히 96"):
                analyze_calibration(
                    config,
                    trials,
                    bootstrap_replicates=2,
                    allow_synthetic=True,
                )

    def test_config_raw_hash_must_match_every_trial(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config, trials = _write_synthetic_fixture(directory)
            config.write_text(config.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(AnalysisBlockedError, "config_sha256"):
                analyze_calibration(
                    config,
                    trials,
                    bootstrap_replicates=2,
                    allow_synthetic=True,
                )

    def test_lapse_value_and_provenance_are_frozen_per_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            config, trials = _write_synthetic_fixture(
                Path(directory_name), corrupt_lapse_source=True
            )
            with self.assertRaisesRegex(AnalysisBlockedError, "lapse_source"):
                analyze_calibration(
                    config,
                    trials,
                    bootstrap_replicates=2,
                    allow_synthetic=True,
                )

    def test_c96_rejects_szafir_range_extrapolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            config, trials = _write_synthetic_fixture(
                Path(directory_name), viewing_distance_mm=400.0
            )
            with self.assertRaisesRegex(AnalysisBlockedError, "Szafir"):
                analyze_calibration(
                    config,
                    trials,
                    bootstrap_replicates=2,
                    allow_synthetic=True,
                )

    def test_empty_trial_file_cannot_produce_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config, _ = _write_synthetic_fixture(directory)
            empty = directory / "empty.jsonl"
            empty.write_bytes(b"")
            with self.assertRaisesRegex(AnalysisBlockedError, "비었습니다"):
                analyze_calibration(
                    config,
                    empty,
                    bootstrap_replicates=2,
                    allow_synthetic=True,
                )


if __name__ == "__main__":
    unittest.main()
