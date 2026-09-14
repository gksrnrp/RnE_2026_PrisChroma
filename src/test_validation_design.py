import hashlib
import json
import math
import tempfile
import unittest
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from analyze_validation import (
    ValidationAnalysisError,
    _model_comparisons,
    _pool_blocks,
    _validate_confirmation_review,
    analyze_validation,
)
from psychometrics import direct_c80_probability
from validation_design import (
    CANDIDATE_POLICY,
    ValidationDesignError,
    _read_json,
    _select_candidate,
    build_validation_plan,
    compute_plan_sha256,
    validate_plan,
)
from validation_recovery import (
    ValidationRecoveryError,
    run_plan_bound_recovery,
    run_validation_recovery,
)
from implementation_provenance import implementation_hashes


def _write_json(path: Path, payload: dict) -> str:
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _config() -> dict:
    return {
        "config_version": "synthetic-validation-test-v1",
        "observer_profile": {
            "participant_id": "P_SYNTH",
            "cvd_type": "deutan",
            "severity": 0.7,
            "fixed_lapse": 0.02,
        },
        "color_design": {
            "midpoint_lab": [55.0, 5.0, 0.0],
            "direction_lab": [0.0, 1.0, 0.0],
            "target_absolute_tolerance_delta_e00": 0.05,
            "target_relative_tolerance": 0.02,
            "search_grid_steps": 64,
        },
        "validation": {
            "candidates": [
                {"geometric_target_px": 4.0, "rendered_width_px": 4, "width_deg": 0.08},
                {"geometric_target_px": 12.0, "rendered_width_px": 12, "width_deg": 0.24},
            ],
            "independent_blocks": 3,
            "delta_e00_level_count": 8,
            "trials_per_block": 64,
            "randomization": "independent-seeded-within-block-shuffle",
        },
    }


def _model(
    name: str,
    predictions: list[tuple[int, float, float]],
    *,
    bootstrap_replicates: int = 200,
    training_range_checks: list[dict] | None = None,
) -> dict:
    if name in {"B96", "B288"}:
        parameters = {"c80": 2.0, "w": 0.34}
    elif name == "C96":
        parameters = {"c80_ref": 2.0, "w": 0.36}
    else:
        parameters = {"c80_ref": 2.0, "beta": -0.35, "w": 0.35}
    intervals = [
        {"parameter": key, "ci_low": value * 0.8, "ci_high": value * 1.2}
        for key, value in parameters.items()
        if key != "beta"
    ]
    if "beta" in parameters:
        intervals.append({"parameter": "beta", "ci_low": -0.5, "ci_high": -0.2})
    pairing_id = (
        "B96-C96-reference-width-parametric-v1"
        if name in {"B96", "C96"}
        else None
    )
    return {
        "model": name,
        "training_trial_count": 96 if name in {"B96", "C96"} else 288,
        "fit": {
            "parameters": parameters,
            "fixed_lapse": 0.02,
            "negative_log_likelihood": 120.0,
            "converged": True,
            "iterations": 80,
            "message": "synthetic convergence",
        },
        "bootstrap": {
            "seed": 11,
            "confidence": 0.95,
            "requested_replicates": bootstrap_replicates,
            "successful_replicates": bootstrap_replicates,
            "convergence_rate": 1.0,
            "converged": True,
            "replicate_pairing_id": pairing_id,
            "parameter_intervals": intervals,
        },
        "held_out_candidate_predictions": [
            {
                "stroke_width_px": width,
                "stroke_width_deg": degree,
                "c80_sim": c80,
                "ci_low": c80,
                "ci_high": c80,
                "bootstrap_c80_samples": [c80] * bootstrap_replicates,
                "bootstrap_pairing_id": pairing_id,
                "value_type": "calibration_prediction",
                "observed_held_out_used": False,
            }
            for width, degree, c80 in predictions
        ],
        "training_range_checks": training_range_checks or [],
    }


def _calibration(
    config_sha: str,
    *,
    config: dict | None = None,
    human: bool = False,
) -> dict:
    calibration_widths = (
        [
            (int(row["width_px"]), float(row["width_deg"]))
            for row in config["calibration"]["widths"]
        ]
        if config is not None
        else [(2, 0.04), (8, 0.16), (16, 0.32)]
    )
    candidate_widths = (
        [
            (int(row["rendered_width_px"]), float(row["width_deg"]))
            for row in config["validation"]["candidates"]
        ]
        if config is not None
        else [(4, 0.08), (12, 0.24)]
    )
    candidate_c80 = {
        "B96": (2.0, 2.0),
        "B288": (2.3, 2.3),
        "C96": (3.4, 2.15),
        "D288": (3.1, 1.7),
    }
    predictions = {
        name: [
            (width_px, width_deg, candidate_c80[name][index])
            for index, (width_px, width_deg) in enumerate(candidate_widths)
        ]
        for name in ("B96", "B288", "C96", "D288")
    }
    reference_width_px, reference_width_deg = calibration_widths[1]
    presented_levels = (
        [float(value) for value in config["sampling"]["delta_e00_levels"]]
        if config is not None
        else [0.5, 11.4]
    )
    presented_low, presented_high = min(presented_levels), max(presented_levels)
    bootstrap_replicates = 1000 if human else 200

    def range_checks(name: str) -> list[dict]:
        widths = (
            [(reference_width_px, reference_width_deg)]
            if name in {"B96", "C96"}
            else calibration_widths
        )
        rows: list[dict] = []
        for _width_px, width_deg in widths:
            if name in {"B96", "B288"}:
                c80 = 2.0
            elif name == "C96":
                c80 = 2.0
            else:
                c80 = 2.0 * (width_deg / reference_width_deg) ** -0.35
            ci_low, ci_high = c80 * 0.8, c80 * 1.2
            bracketed = presented_low <= ci_low <= c80 <= ci_high <= presented_high
            rows.append(
                {
                    "stroke_width_deg": width_deg,
                    "presented_delta_e00_min": presented_low,
                    "presented_delta_e00_max": presented_high,
                    "c80_sim": c80,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "point_bracketed": presented_low <= c80 <= presented_high,
                    "ci_bracketed": presented_low <= ci_low <= ci_high <= presented_high,
                    "range_bracketed": bracketed,
                }
            )
        return rows

    profile = config.get("observer_profile", {}) if config is not None else {}
    color_design = config.get("color_design", {}) if config is not None else {}
    line = config.get("line", {}) if config is not None else {}
    return {
        "schema_version": "1.0",
        "artifact_type": "calibration_model_fits",
        "analysis_scope": "calibration_only_intermediate",
        "data_origin": "human" if human else "synthetic",
        "research_conclusion_allowed": False,
        "final_thresholds_ready": False,
        "contains_observed_held_out": False,
        "coordinate": "post_machado_delta_e00",
        "formula_version": "direct-c80-4afc-v1",
        "analysis_script_sha256": hashlib.sha256(
            Path(__file__).with_name("analyze_calibration.py").read_bytes()
        ).hexdigest(),
        "implementation_hashes": implementation_hashes("calibration_analysis"),
        "source": {
            "config_version": (
                config["config_version"] if config is not None else "synthetic-validation-test-v1"
            ),
            "config_sha256": config_sha,
            "trial_file_sha256": "1" * 64,
            "stimulus_manifest_sha256": "f" * 64 if human else None,
            "git_commit": "a" * 40,
            "stimulus_family_id": color_design.get("stimulus_family_id", "synthetic-family"),
            "renderer_id": line.get("renderer", "synthetic-renderer"),
            "total_record_count": 288,
        },
        "analysis_specification": {
            "models": ["B96", "B288", "C96", "D288"],
            "reference_width_policy": "middle_of_three_sorted_calibration_widths",
            "confidence": 0.95,
            "bootstrap_replicates": bootstrap_replicates,
            "minimum_bootstrap_convergence_rate": 0.9,
            "bootstrap_seed": 11,
            "szafir_direction_lab": [0.0, 1.0, 0.0],
            "szafir_probability_coordinate": 0.8,
            "szafir_extrapolation_allowed": False,
        },
        "participants": [
            {
                "participant_id": profile.get("participant_id", "P_SYNTH"),
                "cvd_type": profile.get("cvd_type", "deutan"),
                "machado_severity": profile.get("severity", 0.7),
                "severity_source": profile.get("severity_source", "synthetic source"),
                "severity_session_id": profile.get("severity_session_id", "S_SEVERITY_SYNTH"),
                "fixed_lapse": profile.get("fixed_lapse", 0.02),
                "lapse_source": profile.get("lapse_source", "synthetic source"),
                "lapse_session_id": profile.get("lapse_session_id", "S_LAPSE_SYNTH"),
                "calibration": {
                    "completed_trial_count": 288,
                    "aborted_trial_count": 0,
                    "ignored_noncalibration_record_count": 0,
                    "widths": [
                        {
                            "stroke_width_px": width_px,
                            "stroke_width_deg": width_deg,
                            "completed_trials": 96,
                        }
                        for width_px, width_deg in calibration_widths
                    ],
                    "session_block_quality": [
                        {
                            "session_block": block,
                            "session_id": f"CAL-S{block}",
                            "completed_trials": 72,
                            "correct_trials": 54,
                            "accuracy": 0.75,
                        }
                        for block in range(1, 5)
                    ],
                    "session_block_accuracy_use": (
                        "descriptive_quality_only_not_exclusion_or_inference"
                    ),
                    "reference_width_px": reference_width_px,
                    "reference_width_deg": reference_width_deg,
                },
                "width_slope_sensitivity": {
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
                    "d288_beta_interpretation_eligible": True,
                    "d288_beta_interpretation_reason": (
                        "shared_w_retained_and_d288_numerically_complete"
                    ),
                    "primary_c96_vs_b96_affected": False,
                },
                "analysis_status": "complete",
                "models": [
                    _model(
                        name,
                        values,
                        bootstrap_replicates=bootstrap_replicates,
                        training_range_checks=range_checks(name),
                    )
                    for name, values in predictions.items()
                ],
            }
        ],
    }


def _fake_pair_generator(midpoint, direction, target, cvd_type, severity, **kwargs):
    code = int(round(target * 10000))
    return {
        "rgb_a": (code % 200 + 1, (code // 200) % 200 + 1, 10),
        "rgb_b": (code % 200 + 2, (code // 200) % 200 + 1, 10),
        "displayed_lab_a": (50.0, -target / 2, 0.0),
        "displayed_lab_b": (50.0, target / 2, 0.0),
        "simulated_rgb_a": (100.0, 100.0, 100.0),
        "simulated_rgb_b": (101.0, 100.0, 100.0),
        "simulated_lab_a": (50.0, -target / 2, 0.0),
        "simulated_lab_b": (50.0, target / 2, 0.0),
        "original_delta_e00": target * 1.2,
        "simulated_delta_e00": target,
    }


def _fixture(directory: Path):
    config_path = directory / "config.json"
    config_sha = _write_json(config_path, _config())
    calibration_path = directory / "calibration.json"
    _write_json(calibration_path, _calibration(config_sha))
    return config_path, calibration_path


def _set_candidate_prediction(
    calibration: dict,
    model_name: str,
    width_px: int,
    *,
    c80: float,
    sample_low: float,
    sample_high: float,
) -> None:
    model = next(
        row
        for row in calibration["participants"][0]["models"]
        if row["model"] == model_name
    )
    prediction = next(
        row
        for row in model["held_out_candidate_predictions"]
        if row["stroke_width_px"] == width_px
    )
    count = int(model["bootstrap"]["successful_replicates"])
    samples = [sample_low] * (count // 2) + [sample_high] * (count - count // 2)
    alpha = (1.0 - float(model["bootstrap"]["confidence"])) / 2.0
    ci_low, ci_high = np.quantile(
        np.asarray(samples, dtype=float), (alpha, 1.0 - alpha)
    )
    prediction.update(
        c80_sim=c80,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        bootstrap_c80_samples=samples,
    )


def _rehash_plan(plan: dict) -> None:
    plan["plan_sha256"] = compute_plan_sha256(plan)
    for row in plan["schedule"]:
        row["validation_plan_sha256"] = plan["plan_sha256"]


def _write_validation_trials(path: Path, plan: dict, *, c80_by_block=(2.5, 2.5, 2.5)) -> None:
    per_level_seen = defaultdict(int)
    records = []
    for schedule in plan["schedule"]:
        block = int(schedule["validation_block_id"][1:])
        level_index = schedule["validation_level_index"]
        level = plan["levels"][level_index - 1]
        probability = direct_c80_probability(
            level["actual_delta_e00_simulated"], c80_by_block[block - 1], 0.35, 0.02
        )
        success_count = int(round(probability * 8))
        key = (block, level_index)
        correct = per_level_seen[key] < success_count
        per_level_seen[key] += 1
        correct_position = schedule["correct_position"]
        response_position = correct_position if correct else correct_position % 4 + 1
        records.append(
            {
                **schedule,
                "session_id": f"S_VALIDATION_{block:02d}",
                "trial_status": "completed",
                "is_correct": correct,
                "response_position": response_position,
                "fixed_lapse": 0.02,
                "delta_e00_simulated": level["actual_delta_e00_simulated"],
            }
        )
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
        newline="\n",
    )


class ValidationPlanTests(unittest.TestCase):
    def test_default_generator_rechecks_real_quantized_levels(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            config_payload = _config()
            config_payload["color_design"]["target_absolute_tolerance_delta_e00"] = 0.30
            config_payload["color_design"]["target_relative_tolerance"] = 0.05
            config_payload["color_design"]["search_grid_steps"] = 512
            config = directory / "config.json"
            config_sha = _write_json(config, config_payload)
            calibration = directory / "calibration.json"
            _write_json(calibration, _calibration(config_sha))
            plan = build_validation_plan(
                config, calibration, seed=17, allow_synthetic=True
            )
            rgb_pairs = {
                tuple(sorted((tuple(row["rgb_a"]), tuple(row["rgb_b"]))))
                for row in plan["levels"]
            }
            actual = [row["actual_delta_e00_simulated"] for row in plan["levels"]]
            self.assertEqual(len(rgb_pairs), 8)
            self.assertEqual(actual, sorted(actual))

    def test_plan_is_deterministic_balanced_and_bound_to_inputs(self):
        with tempfile.TemporaryDirectory() as name:
            config, calibration = _fixture(Path(name))
            first = build_validation_plan(
                config, calibration, seed=17, allow_synthetic=True,
                pair_generator=_fake_pair_generator,
            )
            second = build_validation_plan(
                config, calibration, seed=17, allow_synthetic=True,
                pair_generator=_fake_pair_generator,
            )
            self.assertEqual(first, second)
            self.assertEqual(first["selected_width"]["stroke_width_px"], 4)
            self.assertEqual(len(first["levels"]), 8)
            self.assertEqual(len(first["schedule"]), 192)
            self.assertEqual(
                sorted(row["trial_id"] for row in first["schedule"]),
                list(range(289, 481)),
            )
            self.assertEqual(first["plan_sha256"], compute_plan_sha256(first))
            validate_plan(first)
            self.assertTrue(
                all(
                    first["levels"][index]["actual_delta_e00_simulated"]
                    < first["levels"][index + 1]["actual_delta_e00_simulated"]
                    for index in range(7)
                )
            )

    def test_selection_prefers_conservative_ci_gap_before_larger_point_gap(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            config, calibration_path = _fixture(directory)
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            _set_candidate_prediction(
                calibration, "B96", 4, c80=1.0, sample_low=0.1, sample_high=10.0
            )
            _set_candidate_prediction(
                calibration, "C96", 4, c80=6.0, sample_low=0.6, sample_high=60.0
            )
            _set_candidate_prediction(
                calibration, "B96", 12, c80=1.0, sample_low=0.9, sample_high=1.1
            )
            _set_candidate_prediction(
                calibration, "C96", 12, c80=2.0, sample_low=1.8, sample_high=2.2
            )
            _write_json(calibration_path, calibration)

            plan = build_validation_plan(
                config,
                calibration_path,
                allow_synthetic=True,
                pair_generator=_fake_pair_generator,
            )
            by_width = {
                row["stroke_width_px"]: row for row in plan["candidate_predictions"]
            }
            self.assertGreater(
                by_width[4]["absolute_log_c80_ratio"],
                by_width[12]["absolute_log_c80_ratio"],
            )
            self.assertEqual(by_width[4]["conservative_95ci_log_gap"], 0.0)
            self.assertGreater(by_width[12]["conservative_95ci_log_gap"], 0.0)
            self.assertEqual(plan["selected_width"]["stroke_width_px"], 12)
            self.assertTrue(plan["selected_width"]["predicted_separability"])
            self.assertEqual(
                plan["selection_specification"]["candidate_policy"],
                CANDIDATE_POLICY,
            )

    def test_zero_ci_gaps_use_point_gap_but_mark_separability_unconfirmed(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            config, calibration_path = _fixture(directory)
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            _set_candidate_prediction(
                calibration, "B96", 4, c80=1.0, sample_low=0.5, sample_high=2.0
            )
            _set_candidate_prediction(
                calibration, "C96", 4, c80=2.0, sample_low=1.0, sample_high=4.0
            )
            _set_candidate_prediction(
                calibration, "B96", 12, c80=1.0, sample_low=0.3, sample_high=2.0
            )
            _set_candidate_prediction(
                calibration, "C96", 12, c80=4.0, sample_low=1.2, sample_high=8.0
            )
            _write_json(calibration_path, calibration)

            plan = build_validation_plan(
                config,
                calibration_path,
                allow_synthetic=True,
                pair_generator=_fake_pair_generator,
            )
            self.assertTrue(
                all(
                    row["conservative_95ci_log_gap"] == 0.0
                    for row in plan["candidate_predictions"]
                )
            )
            self.assertEqual(plan["selected_width"]["stroke_width_px"], 12)
            self.assertFalse(plan["selected_width"]["predicted_separability"])
            self.assertEqual(
                plan["selected_width"]["pre_validation_discriminability"],
                "not-established-before-held-out-collection",
            )

    def test_exact_score_tie_selects_lower_px(self):
        rows = [
            {
                "stroke_width_px": width,
                "conservative_95ci_log_gap": 0.0,
                "absolute_log_c80_ratio": 0.5,
            }
            for width in (12, 4)
        ]
        self.assertEqual(_select_candidate(rows)["stroke_width_px"], 4)

    def test_rehashed_selection_policy_and_scores_cannot_bypass_validator(self):
        with tempfile.TemporaryDirectory() as name:
            config, calibration = _fixture(Path(name))
            original = build_validation_plan(
                config,
                calibration,
                allow_synthetic=True,
                pair_generator=_fake_pair_generator,
            )
            mutations = (
                (
                    "candidate policy",
                    lambda plan: plan["selection_specification"].update(
                        candidate_policy="maximum-absolute-log-c80-ratio-then-lower-width-v1"
                    ),
                    "candidate_policy",
                ),
                (
                    "candidate score",
                    lambda plan: plan["candidate_predictions"][0].update(
                        absolute_log_c80_ratio=99.0
                    ),
                    "선택 점수",
                ),
                (
                    "candidate flag",
                    lambda plan: plan["candidate_predictions"][0].update(
                        predicted_separability=False
                    ),
                    "구분 가능성",
                ),
                (
                    "selected width",
                    lambda plan: plan["selected_width"].update(stroke_width_px=12),
                    "사전등록 후보 선택",
                ),
                (
                    "selected score",
                    lambda plan: plan["selected_width"].update(
                        selection_score_conservative_95ci_log_gap=99.0
                    ),
                    "사전등록 후보 선택",
                ),
                (
                    "separability flag",
                    lambda plan: plan["selected_width"].update(
                        predicted_separability=False
                    ),
                    "구분 가능성",
                ),
            )
            for label, mutate, message in mutations:
                with self.subTest(label=label):
                    plan = json.loads(json.dumps(original))
                    mutate(plan)
                    _rehash_plan(plan)
                    with self.assertRaisesRegex(ValidationDesignError, message):
                        validate_plan(plan)

    def test_rehashed_level_target_cannot_bypass_level_policy(self):
        with tempfile.TemporaryDirectory() as name:
            config, calibration = _fixture(Path(name))
            plan = build_validation_plan(
                config,
                calibration,
                allow_synthetic=True,
                pair_generator=_fake_pair_generator,
            )
            plan["levels"][0]["target_delta_e00_simulated"] *= 1.01
            _rehash_plan(plan)
            with self.assertRaisesRegex(ValidationDesignError, "level policy"):
                validate_plan(plan)

    def test_duplicate_json_keys_are_rejected_for_all_validation_inputs(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            for label in ("config", "calibration", "plan"):
                with self.subTest(label=label):
                    path = directory / f"{label}.json"
                    path.write_text(
                        '{"outer":{"frozen":true,"frozen":false}}\n',
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValidationDesignError, "중복 key"):
                        _read_json(path)

    def test_validation_trial_id_range_cannot_be_rebased(self):
        with tempfile.TemporaryDirectory() as name:
            config, calibration = _fixture(Path(name))
            with self.assertRaisesRegex(ValidationDesignError, "289"):
                build_validation_plan(
                    config,
                    calibration,
                    first_trial_id=290,
                    allow_synthetic=True,
                    pair_generator=_fake_pair_generator,
                )

    def test_synthetic_plan_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as name:
            config, calibration = _fixture(Path(name))
            with self.assertRaisesRegex(ValidationDesignError, "allow_synthetic"):
                build_validation_plan(config, calibration, pair_generator=_fake_pair_generator)

    def test_stale_calibration_analysis_code_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            config_path = directory / "config.json"
            config_sha = _write_json(config_path, _config())
            calibration_payload = _calibration(config_sha)
            calibration_payload["analysis_script_sha256"] = "0" * 64
            calibration_path = directory / "calibration.json"
            _write_json(calibration_path, calibration_payload)
            with self.assertRaisesRegex(
                ValidationDesignError, "현재 analyze_calibration.py"
            ):
                build_validation_plan(
                    config_path,
                    calibration_path,
                    allow_synthetic=True,
                    pair_generator=_fake_pair_generator,
                )

    def test_stale_calibration_dependency_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            config_path = directory / "config.json"
            config_sha = _write_json(config_path, _config())
            calibration_payload = _calibration(config_sha)
            calibration_payload["implementation_hashes"]["src/psychometrics.py"] = "0" * 64
            calibration_path = directory / "calibration.json"
            _write_json(calibration_path, calibration_payload)
            with self.assertRaisesRegex(
                ValidationDesignError, "changed=src/psychometrics.py"
            ):
                build_validation_plan(
                    config_path,
                    calibration_path,
                    allow_synthetic=True,
                    pair_generator=_fake_pair_generator,
                )

    def test_quantized_duplicate_is_rejected(self):
        def duplicate_generator(*args, **kwargs):
            result = _fake_pair_generator(*args, **kwargs)
            result["rgb_a"] = (10, 10, 10)
            result["rgb_b"] = (11, 10, 10)
            return result

        with tempfile.TemporaryDirectory() as name:
            config, calibration = _fixture(Path(name))
            with self.assertRaisesRegex(ValidationDesignError, "같은 8-bit RGB"):
                build_validation_plan(
                    config, calibration, allow_synthetic=True,
                    pair_generator=duplicate_generator,
                )

    def test_tampered_plan_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            config, calibration = _fixture(Path(name))
            plan = build_validation_plan(
                config, calibration, allow_synthetic=True,
                pair_generator=_fake_pair_generator,
            )
            plan["schedule"][0]["correct_position"] = 2
            with self.assertRaisesRegex(ValidationDesignError, "plan_sha256"):
                validate_plan(plan)

    def test_rehashed_plan_with_stale_dependency_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            config, calibration = _fixture(Path(name))
            plan = build_validation_plan(
                config, calibration, allow_synthetic=True,
                pair_generator=_fake_pair_generator,
            )
            plan["implementation_hashes"]["src/validation_design.py"] = "0" * 64
            plan["plan_sha256"] = compute_plan_sha256(plan)
            for row in plan["schedule"]:
                row["validation_plan_sha256"] = plan["plan_sha256"]
            with self.assertRaisesRegex(
                ValidationDesignError, "changed=src/validation_design.py"
            ):
                validate_plan(plan)


class ValidationAnalysisTests(unittest.TestCase):
    def _paths(self, directory: Path):
        config, calibration = _fixture(directory)
        plan = build_validation_plan(
            config, calibration, allow_synthetic=True,
            pair_generator=_fake_pair_generator,
        )
        plan_path = directory / "plan.json"
        _write_json(plan_path, plan)
        trials_path = directory / "validation.jsonl"
        return config, calibration, plan, plan_path, trials_path

    def test_review_record_is_hash_bound_post_collection_and_rejects_duplicate_keys(self):
        now = datetime.now(timezone.utc)
        latest = now - timedelta(hours=2)
        record = {
            "schema_version": "1.0",
            "review_type": "human_validation_confirmation",
            "reviewer_id": "R_AUDIT_01",
            "reviewed_at_utc": (now - timedelta(hours=1)).isoformat().replace(
                "+00:00", "Z"
            ),
            "review_note": "Checked all frozen inputs, exclusions, and block notes.",
            "approved_for_confirmation": True,
            "participant_id": "P_TEST_01",
            "config_sha256": "1" * 64,
            "source_trials_sha256": "2" * 64,
            "validation_plan_sha256": "3" * 64,
            "validation_manifest_sha256": "4" * 64,
        }
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "review.json"
            expected_sha = _write_json(path, record)
            bound = _validate_confirmation_review(
                path,
                participant_id="P_TEST_01",
                config_sha256="1" * 64,
                source_trials_sha256="2" * 64,
                validation_plan_sha256="3" * 64,
                validation_manifest_sha256="4" * 64,
                latest_completed_timestamp_utc=latest,
            )
            self.assertEqual(bound["review_record_sha256"], expected_sha)
            self.assertEqual(bound["reviewer_id"], "R_AUDIT_01")

            record["reviewed_at_utc"] = (latest - timedelta(seconds=1)).isoformat().replace(
                "+00:00", "Z"
            )
            _write_json(path, record)
            with self.assertRaisesRegex(ValidationAnalysisError, "마지막 완료"):
                _validate_confirmation_review(
                    path,
                    participant_id="P_TEST_01",
                    config_sha256="1" * 64,
                    source_trials_sha256="2" * 64,
                    validation_plan_sha256="3" * 64,
                    validation_manifest_sha256="4" * 64,
                    latest_completed_timestamp_utc=latest,
                )

            duplicate = json.dumps(record, separators=(",", ":"))[:-1]
            path.write_text(
                duplicate + ',"reviewer_id":"R_OTHER"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValidationAnalysisError, "중복 key"):
                _validate_confirmation_review(
                    path,
                    participant_id="P_TEST_01",
                    config_sha256="1" * 64,
                    source_trials_sha256="2" * 64,
                    validation_plan_sha256="3" * 64,
                    validation_manifest_sha256="4" * 64,
                    latest_completed_timestamp_utc=latest,
                )

    def test_review_record_cannot_be_future_dated(self):
        now = datetime.now(timezone.utc)
        record = {
            "schema_version": "1.0",
            "review_type": "human_validation_confirmation",
            "reviewer_id": "R_AUDIT_01",
            "reviewed_at_utc": (now + timedelta(days=1)).isoformat().replace(
                "+00:00", "Z"
            ),
            "review_note": "Checked all frozen inputs, exclusions, and block notes.",
            "approved_for_confirmation": True,
            "participant_id": "P_TEST_01",
            "config_sha256": "1" * 64,
            "source_trials_sha256": "2" * 64,
            "validation_plan_sha256": "3" * 64,
            "validation_manifest_sha256": "4" * 64,
        }
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "review.json"
            _write_json(path, record)
            with self.assertRaisesRegex(ValidationAnalysisError, "미래"):
                _validate_confirmation_review(
                    path,
                    participant_id="P_TEST_01",
                    config_sha256="1" * 64,
                    source_trials_sha256="2" * 64,
                    validation_plan_sha256="3" * 64,
                    validation_manifest_sha256="4" * 64,
                    latest_completed_timestamp_utc=now - timedelta(hours=1),
                )

    def test_three_stable_blocks_are_fit_and_pooled(self):
        with tempfile.TemporaryDirectory() as name:
            paths = self._paths(Path(name))
            config, calibration, plan, plan_path, trials = paths
            _write_validation_trials(trials, plan)
            result = analyze_validation(
                config, calibration, plan_path, trials,
                bootstrap_replicates=20, bootstrap_seed=91, allow_synthetic=True,
            )
            self.assertEqual(result["data_origin"], "synthetic")
            self.assertFalse(result["research_conclusion_allowed"])
            self.assertEqual(
                result["implementation_hashes"],
                implementation_hashes("validation_analysis"),
            )
            self.assertEqual(
                result["analysis_script_sha256"],
                result["implementation_hashes"]["src/analyze_validation.py"],
            )
            validation = result["participants"][0]["validation"]
            self.assertEqual(len(validation["blocks"]), 3)
            self.assertTrue(validation["stable"])
            self.assertIsNotNone(validation["pooled_c80"])
            self.assertEqual(result["source_validation_plan_sha256"], plan["plan_sha256"])
            comparison = result["participants"][0]["model_comparisons"]
            self.assertEqual(
                result["participants"][0]["calibration_width_slope_sensitivity"],
                json.loads(calibration.read_text(encoding="utf-8"))["participants"][0][
                    "width_slope_sensitivity"
                ],
            )
            self.assertEqual(comparison["primary"]["first_model"], "B96")
            self.assertEqual(comparison["primary"]["second_model"], "C96")
            self.assertEqual(comparison["secondary"]["first_model"], "D288")
            self.assertEqual(comparison["secondary"]["second_model"], "B288")
            self.assertIn(
                comparison["primary_outcome"],
                {"B96_lower_error", "C96_lower_error", "equal_within_tolerance", "model_indistinguishable"},
            )
            self.assertEqual(
                comparison["metadata"]["inference_scope"],
                "paired-bootstrap-uncertainty-propagation-not-population-inference",
            )
            delta_loss = comparison["primary_delta_loss"]
            self.assertTrue(delta_loss["calibration_prediction_uncertainty_included"])
            self.assertTrue(delta_loss["validation_observation_uncertainty_included"])
            self.assertGreaterEqual(delta_loss["successful_combined_replicates"], 2)
            self.assertIn(
                delta_loss["status"],
                {
                    "resolved_B96_lower_error",
                    "resolved_C96_lower_error",
                    "equivalent_within_tolerance",
                    "indeterminate",
                },
            )
            self.assertIn(
                comparison["shared_w_sensitivity"]["interpretation"],
                {
                    "shared_w_retained",
                    "free_w_lower_aic_warning_not_selected",
                    "unresolved_numerical_convergence",
                },
            )

    def test_aborted_attempt_before_completed_retry_is_not_counted(self):
        with tempfile.TemporaryDirectory() as name:
            config, calibration, plan, plan_path, trials = self._paths(Path(name))
            _write_validation_trials(trials, plan)
            records = [
                json.loads(line)
                for line in trials.read_text(encoding="utf-8").splitlines()
            ]
            aborted = dict(records[0])
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
            result = analyze_validation(
                config,
                calibration,
                plan_path,
                trials,
                bootstrap_replicates=10,
                bootstrap_seed=31,
                allow_synthetic=True,
            )
            self.assertEqual(
                sum(
                    block["completed_trial_count"]
                    for block in result["participants"][0]["validation"]["blocks"]
                ),
                192,
            )

    def test_incompatible_blocks_are_not_pooled(self):
        with tempfile.TemporaryDirectory() as name:
            config, calibration, plan, plan_path, trials = self._paths(Path(name))
            _write_validation_trials(trials, plan, c80_by_block=(0.8, 2.5, 7.0))
            result = analyze_validation(
                config, calibration, plan_path, trials,
                bootstrap_replicates=20, bootstrap_seed=42, allow_synthetic=True,
            )
            validation = result["participants"][0]["validation"]
            self.assertFalse(validation["stable"])
            self.assertIsNone(validation["pooled_c80"])
            comparison = result["participants"][0]["model_comparisons"]
            self.assertEqual(comparison["primary_outcome"], "validation_unstable")
            self.assertFalse(comparison["resolution_check"])
            self.assertIsNone(comparison["primary"]["first_abs_error"])
            self.assertEqual(comparison["primary_delta_loss"]["status"], "unresolvable")
            self.assertEqual(
                comparison["primary_delta_loss"]["reason"], "validation_unstable"
            )

    def test_cv_instability_does_not_falsely_mark_bracketing_as_failed(self):
        blocks = [
            {
                "c80_sim": value,
                "c80_ci": {"ci_low": value * 0.8, "ci_high": value * 1.2},
                "presented_delta_e00_min": 0.1,
                "presented_delta_e00_max": 10.0,
                "range_bracketed": True,
            }
            for value in (1.0, 2.0, 4.0)
        ]
        result = _pool_blocks(
            blocks,
            np.eye(3),
            np.asarray([[1.0, 2.0, 4.0], [1.1, 2.1, 4.1]]),
        )
        self.assertFalse(result[2])
        self.assertTrue(result[-1])

    def test_primary_point_advantage_is_indeterminate_when_delta_loss_ci_crosses_zero(self):
        pairing_id = "B96-C96-reference-width-parametric-v1"
        predictions = [
            {
                "model": "B96",
                "c80_sim": 1.0,
                "bootstrap_c80_samples": [2.9, 1.0, 2.9, 1.0],
                "bootstrap_pairing_id": pairing_id,
            },
            {"model": "B288", "c80_sim": 1.2},
            {
                "model": "C96",
                "c80_sim": 2.8,
                "bootstrap_c80_samples": [1.0, 2.9, 1.0, 2.9],
                "bootstrap_pairing_id": pairing_id,
            },
            {"model": "D288", "c80_sim": 2.7},
        ]
        validation = {
            "stroke_width_px": 4,
            "stroke_width_deg": 0.08,
            "pooled_c80": 3.0,
            "pooled_ci_low": 2.7,
            "pooled_ci_high": 3.3,
        }
        sensitivity = {"comparison_resolved": True, "free_w_preferred": False}
        comparison = _model_comparisons(
            predictions,
            validation,
            sensitivity,
            np.asarray([3.0, 3.0, 3.0, 3.0]),
        )
        self.assertLess(
            comparison["primary"]["second_abs_error"],
            comparison["primary"]["first_abs_error"],
        )
        self.assertEqual(comparison["primary_outcome"], "model_indistinguishable")
        self.assertFalse(comparison["resolution_check"])
        self.assertEqual(comparison["primary_delta_loss"]["status"], "indeterminate")
        self.assertLess(comparison["primary_delta_loss"]["ci_low"], 0)
        self.assertGreater(comparison["primary_delta_loss"]["ci_high"], 0)

    def test_missing_paired_prediction_samples_fails_closed(self):
        predictions = [
            {"model": "B96", "c80_sim": 2.0},
            {"model": "B288", "c80_sim": 2.2},
            {"model": "C96", "c80_sim": 2.8},
            {"model": "D288", "c80_sim": 2.9},
        ]
        validation = {
            "stroke_width_px": 4,
            "stroke_width_deg": 0.08,
            "pooled_c80": 2.8,
            "pooled_ci_low": 2.5,
            "pooled_ci_high": 3.1,
        }
        comparison = _model_comparisons(
            predictions,
            validation,
            {"comparison_resolved": True, "free_w_preferred": False},
            np.asarray([2.7, 2.9]),
        )
        self.assertEqual(comparison["primary_outcome"], "comparison_unresolvable")
        self.assertEqual(comparison["primary_delta_loss"]["status"], "unresolvable")
        self.assertEqual(
            comparison["primary_delta_loss"]["reason"],
            "paired_calibration_bootstrap_unavailable",
        )

    def test_unresolved_shared_w_sensitivity_makes_primary_inference_indeterminate(self):
        pairing_id = "B96-C96-reference-width-parametric-v1"
        predictions = [
            {
                "model": "B96",
                "c80_sim": 1.0,
                "bootstrap_c80_samples": [0.9, 1.0, 1.1, 1.0],
                "bootstrap_pairing_id": pairing_id,
            },
            {"model": "B288", "c80_sim": 1.2},
            {
                "model": "C96",
                "c80_sim": 3.0,
                "bootstrap_c80_samples": [2.9, 3.0, 3.1, 3.0],
                "bootstrap_pairing_id": pairing_id,
            },
            {"model": "D288", "c80_sim": 2.9},
        ]
        validation = {
            "stroke_width_px": 4,
            "stroke_width_deg": 0.08,
            "pooled_c80": 3.0,
            "pooled_ci_low": 2.8,
            "pooled_ci_high": 3.2,
        }
        comparison = _model_comparisons(
            predictions,
            validation,
            {"comparison_resolved": False, "free_w_preferred": False},
            np.asarray([2.9, 3.0, 3.1, 3.0]),
        )
        self.assertEqual(comparison["primary_outcome"], "model_indistinguishable")
        self.assertFalse(comparison["resolution_check"])
        self.assertEqual(comparison["primary_delta_loss"]["status"], "indeterminate")
        self.assertEqual(
            comparison["primary_delta_loss"]["reason"],
            "shared_w_sensitivity_unresolved",
        )

    def test_legacy_calibration_intervals_without_samples_remain_readable_but_unresolvable(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            config, calibration_path = _fixture(directory)
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            for model in calibration["participants"][0]["models"]:
                model["bootstrap"].pop("replicate_pairing_id", None)
                for prediction in model["held_out_candidate_predictions"]:
                    prediction.pop("bootstrap_c80_samples", None)
                    prediction.pop("bootstrap_pairing_id", None)
            _write_json(calibration_path, calibration)
            plan = build_validation_plan(
                config,
                calibration_path,
                allow_synthetic=True,
                pair_generator=_fake_pair_generator,
            )
            plan_path = directory / "legacy-plan.json"
            _write_json(plan_path, plan)
            trials = directory / "legacy-validation.jsonl"
            _write_validation_trials(trials, plan)
            result = analyze_validation(
                config,
                calibration_path,
                plan_path,
                trials,
                bootstrap_replicates=10,
                bootstrap_seed=73,
                allow_synthetic=True,
            )
            comparison = result["participants"][0]["model_comparisons"]
            self.assertEqual(comparison["primary_outcome"], "comparison_unresolvable")
            self.assertEqual(comparison["primary_delta_loss"]["status"], "unresolvable")

    def test_delta_loss_ci_resolves_c96_only_when_entire_interval_is_negative(self):
        pairing_id = "B96-C96-reference-width-parametric-v1"
        predictions = [
            {
                "model": "B96",
                "c80_sim": 1.0,
                "bootstrap_c80_samples": [0.9, 1.0, 1.1, 1.0],
                "bootstrap_pairing_id": pairing_id,
            },
            {"model": "B288", "c80_sim": 1.2},
            {
                "model": "C96",
                "c80_sim": 3.0,
                "bootstrap_c80_samples": [2.9, 3.0, 3.1, 3.0],
                "bootstrap_pairing_id": pairing_id,
            },
            {"model": "D288", "c80_sim": 2.9},
        ]
        validation = {
            "stroke_width_px": 4,
            "stroke_width_deg": 0.08,
            "pooled_c80": 3.0,
            "pooled_ci_low": 2.8,
            "pooled_ci_high": 3.2,
        }
        comparison = _model_comparisons(
            predictions,
            validation,
            {"comparison_resolved": True, "free_w_preferred": False},
            np.asarray([2.9, 3.0, 3.1, 3.0]),
        )
        self.assertEqual(comparison["primary_outcome"], "C96_lower_error")
        self.assertTrue(comparison["resolution_check"])
        self.assertEqual(
            comparison["primary_delta_loss"]["status"],
            "resolved_C96_lower_error",
        )
        self.assertLess(comparison["primary_delta_loss"]["ci_high"], 0)

    def test_trial_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            config, calibration, plan, plan_path, trials = self._paths(Path(name))
            _write_validation_trials(trials, plan)
            rows = [json.loads(line) for line in trials.read_text(encoding="utf-8").splitlines()]
            rows[0]["validation_level_index"] = 8
            trials.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8", newline="\n",
            )
            with self.assertRaisesRegex(ValidationAnalysisError, "동결 계획"):
                analyze_validation(
                    config, calibration, plan_path, trials,
                    bootstrap_replicates=5, allow_synthetic=True,
                )

    def test_synthetic_result_can_never_be_confirmed(self):
        with tempfile.TemporaryDirectory() as name:
            config, calibration, plan, plan_path, trials = self._paths(Path(name))
            _write_validation_trials(trials, plan)
            with self.assertRaisesRegex(ValidationAnalysisError, "합성 결과"):
                analyze_validation(
                    config, calibration, plan_path, trials,
                    bootstrap_replicates=5, allow_synthetic=True,
                    confirm_human_after_review=True,
                )


class ValidationRecoveryTests(unittest.TestCase):
    def test_recovery_is_explicitly_synthetic_and_reproducible(self):
        kwargs = {
            "fixed_lapse": 0.02,
            "repetitions": 6,
            "bootstrap_replicates": 6,
            "seed": 123,
        }
        levels = [0.6, 0.85, 1.2, 1.7, 2.4, 3.4, 4.8, 6.8]
        first = run_validation_recovery(levels, **kwargs)
        second = run_validation_recovery(levels, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["data_origin"], "synthetic")
        self.assertFalse(first["research_conclusion_allowed"])
        self.assertTrue(first["must_rerun_with_frozen_validation_plan"])
        self.assertEqual(first["simulation"]["repetitions"], 6)
        self.assertEqual(len(first["replicates"]), 6)

    def test_plan_bound_recovery_preserves_both_hashes(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            config, calibration = _fixture(directory)
            plan = build_validation_plan(
                config, calibration, allow_synthetic=True,
                pair_generator=_fake_pair_generator,
            )
            plan_path = directory / "plan.json"
            _write_json(plan_path, plan)
            result = run_plan_bound_recovery(
                config, plan_path, repetitions=3, bootstrap_replicates=4, seed=9
            )
            self.assertEqual(result["plan_sha256"], plan["plan_sha256"])
            self.assertEqual(result["config_sha256"], plan["config_sha256"])
            self.assertEqual(
                result["implementation_hashes"],
                implementation_hashes("validation_recovery"),
            )
            self.assertFalse(result["must_rerun_with_frozen_validation_plan"])

            changed = _config()
            changed["observer_profile"]["fixed_lapse"] = 0.03
            _write_json(config, changed)
            with self.assertRaisesRegex(ValidationRecoveryError, "SHA-256"):
                run_plan_bound_recovery(
                    config, plan_path, repetitions=2, bootstrap_replicates=2
                )


if __name__ == "__main__":
    unittest.main()
