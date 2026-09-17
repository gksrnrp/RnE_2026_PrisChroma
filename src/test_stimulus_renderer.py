from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import stimulus_renderer
from stimulus_renderer import (
    ANTIALIASING_POLICY,
    CAP_POLICY,
    LOCKED_PILLOW_VERSION,
    PIXEL_SNAPPING_POLICY,
    RENDERER_ID,
    RendererValidationError,
    StimulusCompilationError,
    build_stimulus_spec,
    compile_practice_stimulus_manifest,
    compile_stimulus_manifest,
    compile_validation_stimulus_manifest,
    main,
    render_stimulus_png,
    renderer_implementation_sha256,
    verify_renderer_environment,
    verify_stimulus_artifact,
)
from test_validation_design import _calibration
from validation_design import build_validation_plan, compute_plan_sha256


def renderer_config() -> dict:
    return {
        "schema_version": "1.0",
        "config_version": "RENDERER_TEST_V1",
        "task": {
            "type": "4afc_odd_one_out",
            "layout": "2x2",
            "choice_count": 4,
            "practice_feedback": True,
            "main_feedback": False,
        },
        "display": {
            "resolution_width_px": 80,
            "resolution_height_px": 80,
            "background_rgb": [117, 117, 117],
        },
        "line": {
            "length_px": 16,
            "orientation_deg": 0.0,
            "cap": CAP_POLICY,
            "antialiasing": ANTIALIASING_POLICY,
            "renderer": RENDERER_ID,
            "pixel_snapping": PIXEL_SNAPPING_POLICY,
        },
        "coordinates": {
            "positions_px": [[20, 20], [60, 20], [20, 60], [60, 60]],
            "center_spacing_px": 40,
        },
        "color_design": {
            "stimulus_family_id": "TEST_LINE_FAMILY",
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
            "severity_source": "independent test fixture",
            "severity_session_id": "S_SEVERITY_TEST",
            "fixed_lapse": 0.02,
            "lapse_source": "preregistered test fixture",
            "lapse_session_id": "S_LAPSE_TEST",
        },
        "sampling": {
            "randomization": "balanced-mixed-width-session-blocks-within-block-shuffle",
            "delta_e00_levels": [
                0.35,
                0.5,
                0.7,
                1.0,
                1.05,
                1.2,
                1.6,
                2.1,
                2.8,
                4.0,
                6.5,
                8.6,
            ],
            "practice_trials": 6,
        },
        "calibration": {
            "widths": [
                {"width_px": 2, "width_deg": 0.05},
                {"width_px": 4, "width_deg": 0.10},
                {"width_px": 6, "width_deg": 0.15},
            ],
            "trials_per_width": [96, 192, 96],
            "session_block_count": 4
        },
        "validation": {
            "candidates": [
                {
                    "geometric_target_px": 4.0,
                    "rendered_width_px": 4,
                    "width_deg": 0.08,
                },
                {
                    "geometric_target_px": 12.0,
                    "rendered_width_px": 12,
                    "width_deg": 0.24,
                },
            ],
            "independent_blocks": 3,
            "delta_e00_level_count": 8,
            "trials_per_block": 64,
            "randomization": "independent-seeded-within-block-shuffle",
        },
    }


def rendered_trial(**updates) -> dict:
    payload = {
        "trial_id": 1,
        "stroke_width_px": 4,
        "correct_position": 2,
        "odd_color": "b",
        "position_colors": ["a", "b", "a", "a"],
        "displayed_rgb_a": [220, 50, 50],
        "displayed_rgb_b": [60, 160, 60],
        "target_delta_e00_simulated": 4.0,
        "delta_e00_simulated": 4.1,
        "stimulus_family_id": "TEST_LINE_FAMILY",
    }
    payload.update(updates)
    return payload


def schedule_row(trial_id: int, correct_position: int) -> dict:
    return {
        "trial_id": trial_id,
        "session_block": 1,
        "stroke_width_px": 4,
        "stroke_width_deg": 0.10,
        "target_delta_e00_simulated": 4.0,
        "correct_position": correct_position,
        "odd_color": "b" if trial_id % 2 else "a",
        "config_version": "RENDERER_TEST_V1",
        "config_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "random_seed": 7,
        "stimulus_family_id": "TEST_LINE_FAMILY",
        "participant_id": "P_TEST_01",
        "cvd_type": "deutan",
        "machado_severity": 0.8,
        "severity_source": "independent test fixture",
        "severity_session_id": "S_SEVERITY_TEST",
        "fixed_lapse": 0.02,
        "lapse_source": "preregistered test fixture",
        "lapse_session_id": "S_LAPSE_TEST",
    }


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


def _full_calibration_schedule(config: dict, config_sha256: str) -> list[dict]:
    settings = config["observer_profile"]
    rows: list[dict] = []
    trial_counts = config["calibration"]["trials_per_width"]
    for width_index, width in enumerate(config["calibration"]["widths"], start=1):
        repeats = trial_counts[width_index - 1] // 96
        for level_index, target in enumerate(
            config["sampling"]["delta_e00_levels"], start=1
        ):
            combinations = [
                (position, odd)
                for position in range(1, 5)
                for odd in ("a", "b")
            ]
            for repeat_index in range(repeats):
                for combination_index, (position, odd) in enumerate(combinations):
                    block = (
                        (level_index - 1) * 8
                        + combination_index
                        + repeat_index * 3
                    ) % 4 + 1
                    rows.append(
                        {
                            "trial_id": 0,
                            "session_block": block,
                            "width_index": width_index,
                            "stroke_width_px": width["width_px"],
                            "stroke_width_deg": width["width_deg"],
                            "delta_e00_level_index": level_index,
                            "target_delta_e00_simulated": target,
                            "correct_position": position,
                            "odd_color": odd,
                            "config_version": config["config_version"],
                            "config_sha256": config_sha256,
                            "git_commit": "b" * 40,
                            "random_seed": 7,
                            "stimulus_family_id": config["color_design"][
                                "stimulus_family_id"
                            ],
                            "participant_id": settings["participant_id"],
                            "cvd_type": settings["cvd_type"],
                            "machado_severity": settings["severity"],
                            "severity_source": settings["severity_source"],
                            "severity_session_id": settings["severity_session_id"],
                            "fixed_lapse": settings["fixed_lapse"],
                            "lapse_source": settings["lapse_source"],
                            "lapse_session_id": settings["lapse_session_id"],
                        }
                    )
    rows.sort(key=lambda row: row["session_block"])
    for trial_id, row in enumerate(rows, start=1):
        row["trial_id"] = trial_id
    return rows


def _validation_fixture(directory: Path) -> tuple[Path, Path, Path]:
    config = renderer_config()
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
    config_path = directory / "config.json"
    config_sha256 = _write_json(config_path, config)

    calibration = _calibration(config_sha256, config=config)
    calibration["source"].update(
        {
            "config_version": config["config_version"],
            "stimulus_family_id": config["color_design"]["stimulus_family_id"],
            "renderer_id": RENDERER_ID,
        }
    )
    participant = calibration["participants"][0]
    participant.update(
        {
            "participant_id": config["observer_profile"]["participant_id"],
            "cvd_type": config["observer_profile"]["cvd_type"],
            "machado_severity": config["observer_profile"]["severity"],
            "severity_source": config["observer_profile"]["severity_source"],
            "severity_session_id": config["observer_profile"][
                "severity_session_id"
            ],
            "fixed_lapse": config["observer_profile"]["fixed_lapse"],
            "lapse_source": config["observer_profile"]["lapse_source"],
            "lapse_session_id": config["observer_profile"]["lapse_session_id"],
        }
    )
    calibration_path = directory / "calibration-fits.json"
    _write_json(calibration_path, calibration)
    plan = build_validation_plan(
        config_path,
        calibration_path,
        seed=17,
        first_trial_id=385,
        allow_synthetic=True,
    )
    plan_path = directory / "validation-plan.json"
    _write_json(plan_path, plan)
    return config_path, calibration_path, plan_path


def _practice_fixture(directory: Path) -> tuple[Path, Path, dict]:
    config = renderer_config()
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
    config_path = directory / "config.json"
    config_sha256 = _write_json(config_path, config)
    calibration = compile_stimulus_manifest(
        config,
        _full_calibration_schedule(config, config_sha256),
    )
    calibration_path = directory / "calibration-manifest.json"
    _write_json(calibration_path, calibration)
    return config_path, calibration_path, calibration


class PixelRendererTests(unittest.TestCase):
    def test_lossless_png_has_exact_boxes_colors_counts_and_hashes(self) -> None:
        config = renderer_config()
        trial = rendered_trial()
        with tempfile.TemporaryDirectory() as folder:
            png = Path(folder) / "trial-0001.png"
            metadata_path = Path(folder) / "trial-0001.pixels.json"
            metadata = render_stimulus_png(
                config, trial, png, metadata_path=metadata_path
            )

            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue(metadata_path.is_file())
            self.assertEqual(metadata["renderer"]["pillow_version"], LOCKED_PILLOW_VERSION)
            self.assertEqual(metadata["renderer"]["cap"], "butt")
            self.assertEqual(metadata["renderer"]["antialiasing"], "disabled")
            self.assertEqual(metadata["renderer"]["resampling"], "none")
            self.assertEqual(
                metadata["renderer"]["implementation_sha256"],
                renderer_implementation_sha256(),
            )
            self.assertEqual(metadata["geometry"]["line_length_px"], 16)
            self.assertEqual(metadata["geometry"]["stroke_width_px"], 4)
            self.assertEqual(
                metadata["objects"][0]["bbox_xyxy_exclusive"], [12, 18, 28, 22]
            )
            self.assertEqual(metadata["objects"][0]["expected_pixel_count"], 64)
            self.assertEqual(
                metadata["objects"][0]["pixel_index_centroid"], [19.5, 19.5]
            )

            with Image.open(png) as image:
                image.load()
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (80, 80))
                self.assertEqual(set(image.get_flattened_data()), {(117, 117, 117), (220, 50, 50), (60, 160, 60)})
                self.assertTrue(
                    all(
                        pixel == (220, 50, 50)
                        for pixel in image.crop((12, 18, 28, 22)).get_flattened_data()
                    )
                )
                self.assertTrue(
                    all(
                        pixel == (60, 160, 60)
                        for pixel in image.crop((52, 18, 68, 22)).get_flattened_data()
                    )
                )

            counts = {
                tuple(item["rgb"]): item["count"]
                for item in metadata["pixel_verification"]["expected_counts_by_rgb"]
            }
            self.assertEqual(counts[(220, 50, 50)], 3 * 64)
            self.assertEqual(counts[(60, 160, 60)], 64)
            self.assertEqual(counts[(117, 117, 117)], 80 * 80 - 4 * 64)
            verified = verify_stimulus_artifact(
                png, metadata_path, config=config, trial=trial
            )
            self.assertTrue(verified["renderer_verified"])

    def test_same_inputs_produce_identical_png_and_pixel_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = render_stimulus_png(
                renderer_config(), rendered_trial(), root / "first.png"
            )
            second = render_stimulus_png(
                renderer_config(), rendered_trial(), root / "second.png"
            )
            self.assertEqual((root / "first.png").read_bytes(), (root / "second.png").read_bytes())
            self.assertEqual(
                first["pixel_verification"], second["pixel_verification"]
            )

    def test_pixel_or_metadata_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            png = root / "trial.png"
            metadata = root / "trial.pixels.json"
            render_stimulus_png(
                renderer_config(), rendered_trial(), png, metadata_path=metadata
            )
            with Image.open(png) as image:
                image.load()
                image.putpixel((0, 0), (0, 0, 0))
                image.save(png, format="PNG")
            with self.assertRaisesRegex(RendererValidationError, "SHA-256"):
                verify_stimulus_artifact(png, metadata)

            render_stimulus_png(
                renderer_config(), rendered_trial(), png, metadata_path=metadata
            )
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            payload["renderer"]["antialiasing"] = "enabled"
            metadata.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RendererValidationError, "antialiasing"):
                verify_stimulus_artifact(png, metadata)

            render_stimulus_png(
                renderer_config(), rendered_trial(), png, metadata_path=metadata
            )
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            payload["renderer"]["implementation_sha256"] = "0" * 64
            metadata.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RendererValidationError, "implementation_sha256"):
                verify_stimulus_artifact(png, metadata)

    def test_unsupported_or_ambiguous_geometry_is_rejected(self) -> None:
        cases = [
            ("antialiasing", "enabled", "antialiasing"),
            ("cap", "round", "cap"),
            ("renderer", "pygame-2.6.1", "renderer"),
            ("pixel_snapping", "subpixel", "pixel_snapping"),
            ("orientation_deg", 45.0, "0/90"),
        ]
        for field, value, message in cases:
            with self.subTest(field=field):
                config = renderer_config()
                config["line"][field] = value
                with self.assertRaisesRegex(RendererValidationError, message):
                    build_stimulus_spec(config, rendered_trial())

        clipped = renderer_config()
        clipped["coordinates"]["positions_px"] = [
            [1, 1],
            [41, 1],
            [1, 41],
            [41, 41],
        ]
        with self.assertRaisesRegex(RendererValidationError, "캔버스"):
            build_stimulus_spec(clipped, rendered_trial())
        overlap = renderer_config()
        overlap["coordinates"]["positions_px"] = [
            [25, 25],
            [55, 25],
            [25, 55],
            [55, 55],
        ]
        overlap["coordinates"]["center_spacing_px"] = 30
        overlap["line"]["length_px"] = 32
        with self.assertRaisesRegex(RendererValidationError, "겹칩니다"):
            build_stimulus_spec(overlap, rendered_trial())

    def test_trial_position_contract_and_color_visibility_are_enforced(self) -> None:
        with self.assertRaisesRegex(RendererValidationError, "position_colors"):
            build_stimulus_spec(
                renderer_config(), rendered_trial(position_colors=["b", "a", "a", "a"])
            )
        with self.assertRaisesRegex(RendererValidationError, "서로 달라야"):
            build_stimulus_spec(
                renderer_config(), rendered_trial(displayed_rgb_b=[220, 50, 50])
            )
        with self.assertRaisesRegex(RendererValidationError, "배경색"):
            build_stimulus_spec(
                renderer_config(), rendered_trial(displayed_rgb_b=[117, 117, 117])
            )

    def test_environment_gate_covers_calibration_and_validation_widths(self) -> None:
        report = verify_renderer_environment(renderer_config())
        self.assertTrue(report["renderer_verified"])
        self.assertEqual(
            report["implementation_sha256"], renderer_implementation_sha256()
        )
        self.assertEqual(report["verified_stroke_widths_px"], [2, 4, 6, 12])
        self.assertTrue(report["human_response_collection_implemented"])

    def test_vertical_orientation_swaps_length_and_width_axes(self) -> None:
        config = renderer_config()
        config["line"]["orientation_deg"] = 90
        spec = build_stimulus_spec(config, rendered_trial())
        self.assertEqual(spec["geometry"]["orientation_class"], "vertical")
        self.assertEqual(spec["objects"][0]["bbox_xyxy_exclusive"], [18, 12, 22, 28])


class StimulusCompilerTests(unittest.TestCase):
    def test_full_compile_requires_all_twelve_configured_targets(self) -> None:
        with self.assertRaisesRegex(StimulusCompilationError, "12 target"):
            compile_stimulus_manifest(renderer_config(), [schedule_row(1, 1)])

    def test_schedule_targets_compile_once_per_unique_target_and_render_pngs(self) -> None:
        config = renderer_config()
        schedule = [schedule_row(1, 1), schedule_row(2, 4)]
        with tempfile.TemporaryDirectory() as folder:
            with patch(
                "stimulus_renderer.generate_symmetric_pair",
                wraps=stimulus_renderer.generate_symmetric_pair,
            ) as generator:
                manifest = compile_stimulus_manifest(
                    config,
                    schedule,
                    output_dir=folder,
                    development_allow_partial_schedule=True,
                )
            self.assertEqual(generator.call_count, 1)
            self.assertEqual(manifest["trial_count"], 2)
            self.assertEqual(manifest["color_generation"]["unique_target_count"], 1)
            self.assertTrue(manifest["renderer"]["renderer_verified"])
            self.assertTrue(manifest["human_response_collection_implemented"])
            self.assertFalse(manifest["human_data_collection_allowed"])
            pair = manifest["color_pairs"][0]
            self.assertLessEqual(
                pair["target_error_delta_e00"],
                pair["allowed_target_error_delta_e00"],
            )
            self.assertAlmostEqual(
                pair["allowed_target_error_delta_e00"],
                max(0.07, 0.03 * 4.0),
            )
            self.assertAlmostEqual(
                pair["target_error_delta_e00"],
                abs(pair["delta_e00_simulated"] - 4.0),
                places=12,
            )
            for row in manifest["trials"]:
                self.assertEqual(row["displayed_rgb_a"], pair["displayed_rgb_a"])
                self.assertEqual(row["displayed_rgb_b"], pair["displayed_rgb_b"])
                self.assertIsNotNone(row["raster_artifact"])
                artifact = row["raster_artifact"]
                self.assertTrue((Path(folder) / artifact["png"]).is_file())
                self.assertTrue((Path(folder) / artifact["pixel_metadata"]).is_file())
                verify_stimulus_artifact(
                    Path(folder) / artifact["png"],
                    Path(folder) / artifact["pixel_metadata"],
                )

    def test_missing_observer_mismatch_and_unreachable_target_fail_closed(self) -> None:
        missing = renderer_config()
        del missing["observer_profile"]
        with self.assertRaisesRegex(StimulusCompilationError, "observer_profile"):
            compile_stimulus_manifest(
                missing,
                [schedule_row(1, 1)],
                development_allow_partial_schedule=True,
            )

        mismatch = [schedule_row(1, 1)]
        mismatch[0]["cvd_type"] = "protan"
        with self.assertRaisesRegex(StimulusCompilationError, "cvd_type"):
            compile_stimulus_manifest(
                renderer_config(),
                mismatch,
                development_allow_partial_schedule=True,
            )

        missing_tolerance = renderer_config()
        del missing_tolerance["color_design"][
            "target_absolute_tolerance_delta_e00"
        ]
        with self.assertRaisesRegex(
            StimulusCompilationError, "target_absolute_tolerance"
        ):
            compile_stimulus_manifest(
                missing_tolerance,
                [schedule_row(1, 1)],
                development_allow_partial_schedule=True,
            )

        impossible = renderer_config()
        impossible["color_design"]["target_absolute_tolerance_delta_e00"] = 0.01
        impossible["color_design"]["target_relative_tolerance"] = 0.01
        row = schedule_row(1, 1)
        row["target_delta_e00_simulated"] = 0.35
        with self.assertRaisesRegex(StimulusCompilationError, "만들지 못했습니다"):
            compile_stimulus_manifest(
                impossible,
                [row],
                development_allow_partial_schedule=True,
            )

    def test_quantized_pairs_must_be_unique_and_actual_levels_strictly_increase(self) -> None:
        first = schedule_row(1, 1)
        second = schedule_row(2, 2)
        first["target_delta_e00_simulated"] = 1.0
        second["target_delta_e00_simulated"] = 1.05

        duplicate_results = [
            {
                "rgb_a": (10, 20, 30),
                "rgb_b": (40, 50, 60),
                "simulated_delta_e00": 1.0,
                "target_error": 0.0,
            },
            {
                # A/B identity가 뒤집혀도 같은 무순서 자극 쌍이다.
                "rgb_a": (40, 50, 60),
                "rgb_b": (10, 20, 30),
                "simulated_delta_e00": 1.05,
                "target_error": 0.0,
            },
        ]
        with patch(
            "stimulus_renderer.generate_symmetric_pair",
            side_effect=duplicate_results,
        ):
            with self.assertRaisesRegex(StimulusCompilationError, "같은 8-bit RGB"):
                compile_stimulus_manifest(
                    renderer_config(),
                    [first, second],
                    development_allow_partial_schedule=True,
                )

        reversed_results = [
            {
                "rgb_a": (10, 20, 30),
                "rgb_b": (40, 50, 60),
                "simulated_delta_e00": 1.04,
                "target_error": 0.04,
            },
            {
                "rgb_a": (11, 20, 30),
                "rgb_b": (41, 50, 60),
                "simulated_delta_e00": 1.01,
                "target_error": 0.04,
            },
        ]
        with patch(
            "stimulus_renderer.generate_symmetric_pair",
            side_effect=reversed_results,
        ):
            with self.assertRaisesRegex(StimulusCompilationError, "엄격히 증가"):
                compile_stimulus_manifest(
                    renderer_config(),
                    [first, second],
                    development_allow_partial_schedule=True,
                )

    def test_cli_renders_without_claiming_response_gui(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path = root / "config.json"
            trial_path = root / "trial.json"
            png_path = root / "trial.png"
            config_path.write_text(json.dumps(renderer_config()), encoding="utf-8")
            trial_path.write_text(json.dumps(rendered_trial()), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "render",
                        "--config",
                        str(config_path),
                        "--trial",
                        str(trial_path),
                        "--output",
                        str(png_path),
                    ]
                ),
                0,
            )
            self.assertTrue(png_path.is_file())
            metadata = json.loads(
                (root / "trial.pixels.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["pixel_verification"]["status"], "verified")


class PhaseManifestCompilerTests(unittest.TestCase):
    def test_calibration_cli_rewrites_raster_paths_relative_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest_path = root / "manifests" / "calibration.json"
            raster_dir = root / "manifests" / "pixels"
            fake = {
                "trial_count": 1,
                "png_files_written": True,
                "trials": [
                    {
                        "raster_artifact": {
                            "png": "trial-0001.png",
                            "pixel_metadata": "trial-0001.pixels.json",
                        }
                    }
                ],
            }
            with patch(
                "stimulus_renderer._read_json_object",
                side_effect=[{}, {"trials": [{}]}],
            ), patch(
                "stimulus_renderer.compile_stimulus_manifest", return_value=fake
            ):
                self.assertEqual(
                    main(
                        [
                            "compile",
                            "--config",
                            "config.json",
                            "--schedule",
                            "schedule.json",
                            "--manifest",
                            str(manifest_path),
                            "--png-dir",
                            str(raster_dir),
                        ]
                    ),
                    0,
                )
            written = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact = written["trials"][0]["raster_artifact"]
            self.assertEqual(artifact["png"], "pixels/trial-0001.png")
            self.assertEqual(
                artifact["pixel_metadata"], "pixels/trial-0001.pixels.json"
            )

    def test_phase_specific_cli_routes_to_dedicated_compilers(self) -> None:
        cases = (
            (
                "compile-validation",
                "compile_validation_stimulus_manifest",
                "compiled_validation_pixel_stimulus_manifest",
                ["--calibration-fits", "fits.json", "--validation-plan", "plan.json"],
            ),
            (
                "compile-practice",
                "compile_practice_stimulus_manifest",
                "compiled_practice_pixel_stimulus_manifest",
                ["--calibration-manifest", "calibration.json"],
            ),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for action, function_name, artifact_type, extra in cases:
                with self.subTest(action=action):
                    output = root / f"{action}.json"
                    fake = {
                        "artifact_type": artifact_type,
                        "phase": "validation" if action.endswith("validation") else "practice",
                        "trial_count": 192 if action.endswith("validation") else 6,
                    }
                    with patch(
                        f"stimulus_renderer.{function_name}", return_value=fake
                    ) as compiler:
                        self.assertEqual(
                            main(
                                [
                                    action,
                                    "--config",
                                    "config.json",
                                    *extra,
                                    "--manifest",
                                    str(output),
                                    "--png-dir",
                                    str(root),
                                ]
                            ),
                            0,
                        )
                    self.assertEqual(
                        json.loads(output.read_text(encoding="utf-8")), fake
                    )
                    compiler.assert_called_once()

    def test_validation_plan_compiles_192_balanced_hashed_pixel_trials(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path, calibration_path, plan_path = _validation_fixture(root)
            raster_dir = root / "validation-pixels"
            manifest = compile_validation_stimulus_manifest(
                config_path,
                calibration_path,
                plan_path,
                output_dir=raster_dir,
            )

            self.assertEqual(
                manifest["artifact_type"],
                "compiled_validation_pixel_stimulus_manifest",
            )
            self.assertEqual(manifest["phase"], "validation")
            self.assertEqual(manifest["trial_count"], 192)
            self.assertEqual(
                manifest["validation_plan_raw_file_sha256"],
                hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["calibration_model_fits_sha256"],
                hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                {row["trial_id"] for row in manifest["trials"]},
                set(range(385, 577)),
            )
            counts = Counter(
                (
                    row["validation_block_id"],
                    row["validation_level_index"],
                    row["correct_position"],
                    row["odd_color"],
                )
                for row in manifest["trials"]
            )
            self.assertEqual(len(counts), 192)
            self.assertEqual(set(counts.values()), {1})
            actual = [
                row["delta_e00_simulated"] for row in manifest["color_pairs"]
            ]
            self.assertEqual(actual, sorted(actual))
            self.assertTrue(
                all(first < second for first, second in zip(actual, actual[1:]))
            )
            first = manifest["trials"][0]["raster_artifact"]
            self.assertEqual(
                first["pixel_metadata_sha256"],
                hashlib.sha256(
                    (raster_dir / first["pixel_metadata"]).read_bytes()
                ).hexdigest(),
            )
            verify_stimulus_artifact(
                raster_dir / first["png"], raster_dir / first["pixel_metadata"]
            )

            tampered = json.loads(plan_path.read_text(encoding="utf-8"))
            tampered["levels"][0]["actual_delta_e00_simulated"] += 0.01
            tampered["plan_sha256"] = compute_plan_sha256(tampered)
            for row in tampered["schedule"]:
                row["validation_plan_sha256"] = tampered["plan_sha256"]
            tampered_path = root / "tampered-plan.json"
            _write_json(tampered_path, tampered)
            with self.assertRaisesRegex(StimulusCompilationError, "재계산"):
                compile_validation_stimulus_manifest(
                    config_path,
                    calibration_path,
                    tampered_path,
                    output_dir=root / "must-not-render",
                )

    def test_practice_compiler_uses_fixed_levels_balances_and_rejects_id_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path, calibration_path, calibration = _practice_fixture(root)
            raster_dir = root / "practice-pixels"
            manifest = compile_practice_stimulus_manifest(
                config_path,
                calibration_path,
                output_dir=raster_dir,
            )

            self.assertEqual(
                manifest["artifact_type"],
                "compiled_practice_pixel_stimulus_manifest",
            )
            self.assertEqual(manifest["phase"], "practice")
            self.assertEqual(manifest["trial_count"], 6)
            self.assertEqual(
                manifest["source_calibration_manifest_sha256"],
                hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["practice_design"][
                    "representative_calibration_level_indices"
                ],
                [1, 6, 12],
            )
            self.assertEqual(
                {row["calibration_level_index"] for row in manifest["trials"]},
                {1, 6, 12},
            )
            self.assertEqual(
                {row["trial_id"] for row in manifest["trials"]},
                set(range(1_000_001, 1_000_007)),
            )
            self.assertTrue(all(row["feedback_enabled"] for row in manifest["trials"]))
            counts = manifest["practice_design"]["counts"]
            self.assertLessEqual(
                max(counts["correct_position"].values())
                - min(counts["correct_position"].values()),
                1,
            )
            self.assertLessEqual(
                max(counts["odd_color"].values())
                - min(counts["odd_color"].values()),
                1,
            )
            artifact = manifest["trials"][0]["raster_artifact"]
            self.assertEqual(
                artifact["pixel_metadata_sha256"],
                hashlib.sha256(
                    (raster_dir / artifact["pixel_metadata"]).read_bytes()
                ).hexdigest(),
            )

            duplicate = compile_practice_stimulus_manifest(
                config_path,
                calibration_path,
                output_dir=root / "practice-pixels-second",
            )
            self.assertEqual(manifest, duplicate)

            overlapping = copy.deepcopy(calibration)
            overlapping["trials"][0]["trial_id"] = 1_000_001
            overlap_path = root / "overlap-calibration.json"
            _write_json(overlap_path, overlapping)
            with self.assertRaisesRegex(StimulusCompilationError, "trial_id"):
                compile_practice_stimulus_manifest(
                    config_path,
                    overlap_path,
                    output_dir=root / "must-not-render",
                )


if __name__ == "__main__":
    unittest.main()
