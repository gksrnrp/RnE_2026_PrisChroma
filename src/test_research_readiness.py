from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from implementation_provenance import IMPLEMENTATION_FILES, implementation_hashes
from research_readiness import (
    _validation_plan_digest,
    assess_research_readiness,
    render_markdown,
)
from test_human_contracts import _human_config, _human_thresholds


FAKE_COMMIT = "a" * 40


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_git_state(_root: Path) -> dict[str, object]:
    return {"commit": FAKE_COMMIT, "dirty": False, "error": None}


def _ready_runtime_gate(_config: dict) -> dict[str, object]:
    return {
        "config_gate_passed": True,
        "renderer_verified": True,
        "human_response_collection_implemented": True,
        "human_data_collection_allowed": True,
        "blockers": [],
    }


def _make_human_ready_scaffold(root: Path) -> None:
    config = _human_config()
    config_path = root / "data" / "experiment" / "experiment_config.json"
    _write_json(
        config_path,
        config,
    )
    _write_json(
        root / "data" / "contracts" / "experiment_config.example.json",
        {
            "schema_version": "1.0",
            "config_version": "SYNTHETIC_FIXTURE",
            "human_data_collection_allowed": False,
        },
    )
    _write_text(
        root / "data" / "contracts" / "trial.example.jsonl",
        json.dumps({"schema_version": "1.0", "data_origin": "synthetic"}) + "\n",
    )
    _write_json(
        root / "data" / "contracts" / "application_conditions.example.json",
        {"schema_version": "1.0", "data_origin": "synthetic", "cases": []},
    )
    for name, properties in {
        "experiment_config.schema.json": {
            "schema_version": {"const": "1.0"},
            "human_data_collection_allowed": {"type": "boolean"},
        },
        "trial.schema.json": {
            "schema_version": {"const": "1.0"},
            "data_origin": {"enum": ["synthetic", "human"]},
        },
        "participant_thresholds.schema.json": {
            "schema_version": {"const": "1.0"},
            "data_origin": {"enum": ["synthetic", "human"]},
            "confirmed": {"type": "boolean"},
        },
    }.items():
        _write_json(
            root / "schemas" / name,
            {"type": "object", "properties": properties},
        )
    source_schema_dir = Path(__file__).resolve().parents[1] / "schemas" / "human"
    for name in (
        "experiment_config.schema.json",
        "trial.schema.json",
        "participant_thresholds.schema.json",
    ):
        _write_text(
            root / "schemas" / "human" / name,
            (source_schema_dir / name).read_text(encoding="utf-8"),
        )

    current = "# 적록색각이상자를 위한 선 굵기별 C80 예측 및 최소 색이동 보정\n"
    _write_text(root / "README.md", current)
    _write_text(root / "docs" / "experiment-protocol.md", current)
    _write_text(root / "docs" / "project-context.txt", current)
    _write_text(root / "docs" / "research-plan.txt", current)
    _write_text(root / "src" / "psychometrics.py", "def fit():\n    return None\n")
    _write_text(
        root / "src" / "test_psychometrics.py",
        "def test_placeholder_name_only():\n    assert True\n",
    )
    _write_text(root / "src" / "parameter_recovery.py", "def run():\n    return True\n")
    _write_text(root / "src" / "experiment_design.py", "def scale():\n    return 1.0\n")
    source_root = Path(__file__).resolve().parents[1]
    required_provenance_files = sorted(
        {relative for paths in IMPLEMENTATION_FILES.values() for relative in paths}
    )
    for relative in required_provenance_files:
        target = root / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source_root / relative).read_bytes())
    config_hash = _hash(config_path)
    widths_deg = [row["width_deg"] for row in config["calibration"]["widths"]]
    recovery_hashes = implementation_hashes("parameter_recovery", project_root=root)
    _write_json(
        root / "results" / "parameter_recovery.json",
        {
            "artifact_type": "synthetic_parameter_recovery",
            "data_origin": "synthetic",
            "research_conclusion_allowed": False,
            "status": "passed",
            "passed": True,
            "reasons": [],
            "config_sha256": config_hash,
            "experiment_config_sha256": config_hash,
            "widths_deg": widths_deg,
            "reference_width_deg": widths_deg[1],
            "trials_per_width": config["calibration"]["trials_per_width"],
            "simulation_replicates": 200,
            "bootstrap_replicates": 1000,
            "fixed_lapse": config["observer_profile"]["fixed_lapse"],
            "all_fits_converged": True,
            "implementation_hashes": recovery_hashes,
            "implementation_sha256": recovery_hashes,
            "recovery_integrity": {
                "minimum_simulation_replicates_met": True,
                "minimum_bootstrap_replicates_met": True,
                "all_profile_gates_passed": True,
                "representative_bootstrap_passed": True,
                "passed": True,
            },
            "monte_carlo": {
                "simulation_replicates": 200,
                "profile_counts": {
                    "low-steep": 67,
                    "central": 67,
                    "high-shallow": 66,
                },
                "profile_results": {
                    name: {
                        "simulation_replicates": count,
                        "models": {model: {} for model in ("B96", "B288", "C96", "D288")},
                        "passed": True,
                        "reasons": [],
                    }
                    for name, count in (
                        ("low-steep", 67),
                        ("central", 67),
                        ("high-shallow", 66),
                    )
                },
                "worst_case_and_upper_quantile_gate": {
                    "passed": True,
                    "reasons": [],
                },
            },
            "representative_bootstrap": {
                "models": {
                    model: {
                        "passed": True,
                        "converged": True,
                        "requested_replicates": 1000,
                        "successful_replicates": 1000,
                        "truth_coverage": {
                            "all_required_truth_covered": True,
                            "required_parameters": (
                                {"c80_ref": True, "beta": True}
                                if model == "D288"
                                else {"c80_ref": True}
                                if model == "C96"
                                else {"c80": True}
                            ),
                        },
                    }
                    for model in ("B96", "B288", "C96", "D288")
                },
                "paired_reference_integrity": {
                    "passed": True,
                    "checks": {"same_reference_width": True},
                },
                "passed": True,
                "reasons": [],
            },
            "sampling_plan": {
                "delta_e00_levels": config["sampling"]["delta_e00_levels"],
                "must_rerun_with_frozen_human_config": False,
            },
            "szafir_relative_scenario": {
                "direction_lab": config["color_design"]["direction_lab"],
                "source_probability_coordinate": config["model_plan"][
                    "szafir_probability_coordinate"
                ],
            },
        },
    )


def _add_claim_evidence(root: Path, *, placeholder_source: bool = False) -> None:
    thresholds = _human_thresholds()
    config_path = root / "data" / "experiment" / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    thresholds["config_version"] = config["config_version"]
    thresholds["config_sha256"] = _hash(config_path)
    calibration_trials_path = (
        root / "data" / "experiment" / "raw" / "calibration-trials.jsonl"
    )
    _write_text(calibration_trials_path, '{"phase":"calibration"}\n')
    calibration_manifest_path = (
        root
        / "data"
        / "experiment"
        / "calibration"
        / "calibration-manifest.json"
    )
    _write_json(
        calibration_manifest_path,
        {
            "artifact_type": "compiled_pixel_stimulus_manifest",
            "schedule_config_sha256": _hash(config_path),
        },
    )
    calibration_analysis_script = root / "src" / "analyze_calibration.py"
    _write_text(calibration_analysis_script, "def analyse():\n    return None\n")
    calibration_path = root / "results" / "calibration_model_fits.json"
    _write_json(
        calibration_path,
        {
            "artifact_type": "calibration_model_fits",
            "data_origin": "human",
            "analysis_script_sha256": _hash(calibration_analysis_script),
            "implementation_hashes": implementation_hashes(
                "calibration_analysis", project_root=root
            ),
            "source": {
                "config_sha256": _hash(config_path),
                "trial_file_sha256": _hash(calibration_trials_path),
                "stimulus_manifest_sha256": _hash(calibration_manifest_path),
                "git_commit": FAKE_COMMIT,
            },
            "participants": [{"analysis_status": "complete"}],
            "analysis_specification": {"bootstrap_replicates": 1000},
        },
    )
    plan = {
        "artifact_type": "validation_plan",
        "schema_version": "1.0",
        "status": "frozen_before_held_out",
        "observed_held_out_used": False,
        "implementation_hashes": implementation_hashes(
            "validation_plan", project_root=root
        ),
        "config_sha256": _hash(config_path),
        "calibration_model_fits_sha256": _hash(calibration_path),
        "source_git_commit": FAKE_COMMIT,
        "schedule": [{"trial_id": index} for index in range(289, 481)],
    }
    plan["plan_sha256"] = _validation_plan_digest(plan)
    for row in plan["schedule"]:
        row["validation_plan_sha256"] = plan["plan_sha256"]
    _write_json(
        root / "data" / "experiment" / "validation" / "validation-plan.json",
        plan,
    )
    validation_manifest_path = (
        root / "data" / "experiment" / "validation" / "validation-manifest.json"
    )
    _write_json(
        validation_manifest_path,
        {
            "artifact_type": "compiled_validation_pixel_stimulus_manifest",
            "schedule_config_sha256": _hash(config_path),
        },
    )
    validation_trials_path = root / "data" / "experiment" / "raw" / "validation-trials.jsonl"
    _write_text(validation_trials_path, "{}\n")
    analysis_script = root / "src" / "analyze_validation.py"
    _write_text(analysis_script, "def analyse():\n    return None\n")
    thresholds["source_validation_plan_sha256"] = plan["plan_sha256"]
    thresholds["source_stimulus_manifest_sha256"] = _hash(validation_manifest_path)
    thresholds["source_trials_sha256"] = _hash(validation_trials_path)
    thresholds["analysis_script_sha256"] = _hash(analysis_script)
    thresholds["implementation_hashes"] = implementation_hashes(
        "validation_analysis", project_root=root
    )
    thresholds["git_commit"] = FAKE_COMMIT
    _write_json(
        root / "data" / "experiment" / "participant_thresholds.json",
        thresholds,
    )
    recovery_script = root / "src" / "validation_recovery.py"
    _write_text(recovery_script, "# synthetic readiness fixture\n")
    _write_json(
        root / "results" / "validation_recovery.json",
        {
            "artifact_type": "validation_parameter_recovery",
            "data_origin": "synthetic",
            "research_conclusion_allowed": False,
            "passed": True,
            "must_rerun_with_frozen_validation_plan": False,
            "plan_sha256": plan["plan_sha256"],
            "config_sha256": plan["config_sha256"],
            "implementation_sha256": _hash(recovery_script),
            "implementation_hashes": implementation_hashes(
                "validation_recovery", project_root=root
            ),
            "simulation": {
                "repetitions": 200,
                "bootstrap_replicates_per_block": 50,
            },
        },
    )
    image = root / "data" / "real" / "images" / "map_one.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"fixture image bytes")
    sources = root / "data" / "real" / "sources.csv"
    sources.parent.mkdir(parents=True, exist_ok=True)
    with sources.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "source_url", "license", "author", "notes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": "map_one.png",
                "source_url": "https://example.org/map" if placeholder_source else "https://data.go.kr/map-one",
                "license": "CC-BY-4.0",
                "author": "public agency",
                "notes": "verified source",
            }
        )

    maps = root / "results" / "maps"
    _write_text(maps / "map_regions.csv", "image,region_id\nmap_one,1\n")
    _write_text(maps / "map_pairs.csv", "image,region_A,region_B\nmap_one,1,2\n")
    script = root / "src" / "map_analysis.py"
    _write_text(script, "def analyse():\n    return []\n")
    _write_json(
        maps / "provenance.json",
        {
            "generated_at": "2026-09-14T12:00:00+09:00",
            "git_commit": FAKE_COMMIT,
            "analysis_script": "src/map_analysis.py",
            "analysis_script_sha256": _hash(script),
            "input_files": [
                {
                    "path": "data/real/images/map_one.png",
                    "sha256": _hash(image),
                }
            ],
            "parameters": {"quantization_bits": 5},
        },
    )
    # _add_claim_evidence가 fake 분석 wrapper를 만든 뒤의 최종 구현 map에 recovery
    # fixture를 결합한다. 실제 workflow에서도 의존 코드가 바뀌면 recovery를
    # 다시 실행해야 한다.
    parameter_recovery_path = root / "results" / "parameter_recovery.json"
    parameter_recovery = json.loads(
        parameter_recovery_path.read_text(encoding="utf-8")
    )
    recovery_hashes = implementation_hashes(
        "parameter_recovery", project_root=root
    )
    parameter_recovery["implementation_hashes"] = recovery_hashes
    parameter_recovery["implementation_sha256"] = dict(recovery_hashes)
    _write_json(parameter_recovery_path, parameter_recovery)


class ResearchReadinessTests(unittest.TestCase):
    def test_empty_or_draft_repository_is_strictly_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_json(
                root / "data" / "contracts" / "experiment_config.example.json",
                {
                    "schema_version": "0.1-draft",
                    "config_version": "EXAMPLE_NOT_FOR_HUMAN_DATA",
                    "status": "draft",
                    "human_data_collection_allowed": False,
                },
            )
            with patch("research_readiness._git_state", _clean_git_state), patch(
                "research_readiness._runtime_collection_gate", _ready_runtime_gate
            ):
                report = assess_research_readiness(root, minimum_real_images=1)
            self.assertFalse(report["ready_for_human_data"])
            self.assertFalse(report["ready_for_research_claims"])
            human_ids = {item["check_id"] for item in report["blocked"]["human_data"]}
            self.assertIn("experiment_config_final", human_ids)
            self.assertIn("psychometric_implementation", human_ids)
            self.assertIn("parameter_recovery", human_ids)

    def test_human_collection_can_be_ready_before_claims_are_ready(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _make_human_ready_scaffold(root)
            with patch("research_readiness._git_state", _clean_git_state), patch(
                "research_readiness._runtime_collection_gate", _ready_runtime_gate
            ):
                report = assess_research_readiness(root, minimum_real_images=1)
            self.assertTrue(report["ready_for_human_data"])
            self.assertTrue(report["ready_for_calibration_collection"])
            self.assertFalse(report["ready_for_validation_collection"])
            self.assertFalse(report["ready_for_research_claims"])
            claim_ids = {item["check_id"] for item in report["blocked"]["research_claims"]}
            self.assertIn("participant_thresholds_confirmed", claim_ids)
            self.assertNotIn("real_images_sources", claim_ids)
            self.assertNotIn("maps_provenance", claim_ids)

    def test_human_collection_rejects_failed_recovery_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _make_human_ready_scaffold(root)
            recovery_path = root / "results" / "parameter_recovery.json"
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            recovery["representative_bootstrap"]["paired_reference_integrity"][
                "passed"
            ] = False
            _write_json(recovery_path, recovery)
            with patch("research_readiness._git_state", _clean_git_state), patch(
                "research_readiness._runtime_collection_gate", _ready_runtime_gate
            ):
                report = assess_research_readiness(root, minimum_real_images=1)
            self.assertFalse(report["ready_for_human_data"])
            check = next(
                row for row in report["checks"]
                if row["check_id"] == "parameter_recovery"
            )
            self.assertEqual(check["status"], "blocked")

    def test_human_collection_rejects_stale_full_recovery_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _make_human_ready_scaffold(root)
            recovery_path = root / "results" / "parameter_recovery.json"
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            recovery["implementation_hashes"]["src/psychometrics.py"] = "0" * 64
            recovery["implementation_sha256"] = dict(
                recovery["implementation_hashes"]
            )
            _write_json(recovery_path, recovery)
            with patch("research_readiness._git_state", _clean_git_state), patch(
                "research_readiness._runtime_collection_gate", _ready_runtime_gate
            ):
                report = assess_research_readiness(root, minimum_real_images=1)
            self.assertFalse(report["ready_for_human_data"])
            check = next(
                row for row in report["checks"]
                if row["check_id"] == "parameter_recovery"
            )
            self.assertEqual(check["status"], "blocked")

    def test_complete_fixture_is_ready_for_both_stages(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _make_human_ready_scaffold(root)
            _add_claim_evidence(root)
            with patch("research_readiness._git_state", _clean_git_state), patch(
                "research_readiness._runtime_collection_gate", _ready_runtime_gate
            ):
                report = assess_research_readiness(root, minimum_real_images=1)
            self.assertTrue(report["ready_for_human_data"])
            self.assertTrue(report["ready_for_calibration_collection"])
            self.assertTrue(report["ready_for_validation_collection"])
            self.assertTrue(report["ready_for_research_claims"])
            self.assertEqual(report["summary"]["blocked"], 0)
            markdown = render_markdown(report)
            self.assertIn("사람 자료 수집: **가능**", markdown)
            self.assertIn("validation 자료 수집: **가능**", markdown)
            self.assertIn("연구 결론 보고: **가능**", markdown)

    def test_validation_collection_requires_1000_calibration_bootstraps(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _make_human_ready_scaffold(root)
            _add_claim_evidence(root)
            calibration_path = root / "results" / "calibration_model_fits.json"
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibration["analysis_specification"]["bootstrap_replicates"] = 999
            _write_json(calibration_path, calibration)
            with patch("research_readiness._git_state", _clean_git_state), patch(
                "research_readiness._runtime_collection_gate", _ready_runtime_gate
            ):
                report = assess_research_readiness(root, minimum_real_images=1)
            self.assertFalse(report["ready_for_validation_collection"])
            check = next(
                row for row in report["checks"]
                if row["check_id"] == "validation_plan_ready"
            )
            self.assertIn("교정 분석 bootstrap 반복 수는 1000회 이상", check["message"])

    def test_claims_require_1000_validation_block_bootstraps(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _make_human_ready_scaffold(root)
            _add_claim_evidence(root)
            thresholds_path = root / "data" / "experiment" / "participant_thresholds.json"
            thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
            thresholds["participants"][0]["validation"]["blocks"][0][
                "bootstrap_replicates"
            ] = 999
            _write_json(thresholds_path, thresholds)
            with patch("research_readiness._git_state", _clean_git_state), patch(
                "research_readiness._runtime_collection_gate", _ready_runtime_gate
            ):
                report = assess_research_readiness(root, minimum_real_images=1)
            self.assertFalse(report["ready_for_research_claims"])
            check = next(
                row for row in report["checks"]
                if row["check_id"] == "participant_thresholds_confirmed"
            )
            self.assertIn("validation block bootstrap 반복 수", check["message"])

    def test_claims_require_1000_prediction_bootstraps(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _make_human_ready_scaffold(root)
            _add_claim_evidence(root)
            thresholds_path = root / "data" / "experiment" / "participant_thresholds.json"
            thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
            thresholds["participants"][0]["predictions"][0][
                "bootstrap_replicates"
            ] = 999
            _write_json(thresholds_path, thresholds)
            with patch("research_readiness._git_state", _clean_git_state), patch(
                "research_readiness._runtime_collection_gate", _ready_runtime_gate
            ):
                report = assess_research_readiness(root, minimum_real_images=1)
            self.assertFalse(report["ready_for_research_claims"])
            check = next(
                row for row in report["checks"]
                if row["check_id"] == "participant_thresholds_confirmed"
            )
            self.assertIn("각 예측 1000회 이상 필요", check["message"])

    def test_validation_collection_requires_current_plan_bound_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _make_human_ready_scaffold(root)
            _add_claim_evidence(root)
            recovery_path = root / "results" / "validation_recovery.json"
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            recovery["simulation"]["bootstrap_replicates_per_block"] = 49
            _write_json(recovery_path, recovery)
            with patch("research_readiness._git_state", _clean_git_state), patch(
                "research_readiness._runtime_collection_gate", _ready_runtime_gate
            ):
                report = assess_research_readiness(root, minimum_real_images=1)
            self.assertTrue(report["ready_for_calibration_collection"])
            self.assertFalse(report["ready_for_validation_collection"])
            check = next(
                row for row in report["checks"]
                if row["check_id"] == "validation_plan_ready"
            )
            self.assertIn("bootstrap 50", check["message"])

    def test_validation_collection_rejects_stale_recovery_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _make_human_ready_scaffold(root)
            _add_claim_evidence(root)
            recovery_path = root / "results" / "validation_recovery.json"
            recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
            recovery["implementation_hashes"]["src/validation_psychometrics.py"] = "0" * 64
            _write_json(recovery_path, recovery)
            with patch("research_readiness._git_state", _clean_git_state), patch(
                "research_readiness._runtime_collection_gate", _ready_runtime_gate
            ):
                report = assess_research_readiness(root, minimum_real_images=1)
            self.assertFalse(report["ready_for_validation_collection"])
            check = next(
                row for row in report["checks"]
                if row["check_id"] == "validation_plan_ready"
            )
            self.assertIn("changed=src/validation_psychometrics.py", check["message"])

    def test_validation_collection_rejects_stale_calibration_analysis_code(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _make_human_ready_scaffold(root)
            _add_claim_evidence(root)
            _write_text(
                root / "src" / "analyze_calibration.py",
                "def analyse_changed():\n    return 'new code'\n",
            )
            with patch("research_readiness._git_state", _clean_git_state), patch(
                "research_readiness._runtime_collection_gate", _ready_runtime_gate
            ):
                report = assess_research_readiness(root, minimum_real_images=1)
            # calibration analysis is part of the complete recovery dependency
            # map, so changing it invalidates both collection and validation.
            self.assertFalse(report["ready_for_calibration_collection"])
            self.assertFalse(report["ready_for_validation_collection"])
            recovery_check = next(
                row for row in report["checks"]
                if row["check_id"] == "parameter_recovery"
            )
            self.assertEqual(recovery_check["status"], "blocked")
            check = next(
                row
                for row in report["checks"]
                if row["check_id"] == "validation_plan_ready"
            )
            self.assertIn("현재 analyze_calibration.py", check["message"])

    def test_placeholder_source_blocks_only_the_optional_image_case(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _make_human_ready_scaffold(root)
            _add_claim_evidence(root, placeholder_source=True)
            with patch("research_readiness._git_state", _clean_git_state), patch(
                "research_readiness._runtime_collection_gate", _ready_runtime_gate
            ):
                report = assess_research_readiness(root, minimum_real_images=1)
            self.assertTrue(report["ready_for_human_data"])
            self.assertTrue(report["ready_for_validation_collection"])
            self.assertTrue(report["ready_for_research_claims"])
            check = next(
                item for item in report["checks"] if item["check_id"] == "real_images_sources"
            )
            self.assertEqual(check["status"], "blocked")
            self.assertIn("placeholder", check["message"])

    def test_unmarked_ai_identity_and_legacy_062_rule_block_human_start(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _make_human_ready_scaffold(root)
            _write_text(
                root / "docs" / "project-context.txt",
                "프로젝트명:\n적록색약자를 위한 개인화 선택적 AI 색보정\n"
                "Phase 6. AI 기반 색보정\n",
            )
            _write_text(
                root / "results" / "conditions.md",
                "환산 비율 = 실제 선 역치 / 0.62\n",
            )
            with patch("research_readiness._git_state", _clean_git_state):
                report = assess_research_readiness(root, minimum_real_images=1)
            identity = next(
                item for item in report["checks"] if item["check_id"] == "research_identity"
            )
            self.assertEqual(identity["status"], "blocked")
            self.assertIn("이전 AI 연구 제목", identity["message"])
            self.assertIn("0.62 환산 규칙", identity["message"])
            self.assertFalse(report["ready_for_human_data"])

    def test_missing_response_gui_blocks_calibration_even_with_valid_config(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _make_human_ready_scaffold(root)
            with patch("research_readiness._git_state", _clean_git_state):
                report = assess_research_readiness(root, minimum_real_images=1)
            self.assertFalse(report["ready_for_calibration_collection"])
            blocked = {
                item["check_id"] for item in report["blocked"]["calibration_collection"]
            }
            self.assertIn("experiment_runtime_gates", blocked)


if __name__ == "__main__":
    unittest.main()
