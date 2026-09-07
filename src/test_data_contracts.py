import copy
import json
import unittest

from data_contracts import (
    ContractError,
    EXAMPLE_PATHS,
    validate_all_examples,
    validate_jsonl,
    validate_payload,
)


def _example(name: str) -> dict:
    return json.loads(EXAMPLE_PATHS[name].read_text(encoding="utf-8"))


class DataContractTests(unittest.TestCase):
    def test_all_draft_examples_validate(self) -> None:
        self.assertEqual(
            validate_all_examples(),
            {
                "experiment_config": 1,
                "trial": 1,
                "application_conditions": 1,
                "validation_pairs": 1,
                "participant_thresholds": 1,
                "correction_results": 1,
            },
        )

    def test_trial_jsonl_validates(self) -> None:
        records = validate_jsonl(EXAMPLE_PATHS["trial"])
        self.assertEqual(len(records), 1)

    def test_draft_trial_rejects_human_data(self) -> None:
        trial = json.loads(
            EXAMPLE_PATHS["trial"].read_text(encoding="utf-8").splitlines()[0]
        )
        trial["data_origin"] = "human"
        with self.assertRaises(ContractError):
            validate_payload("trial", trial)

    def test_trial_correctness_must_match_positions(self) -> None:
        trial = json.loads(
            EXAMPLE_PATHS["trial"].read_text(encoding="utf-8").splitlines()[0]
        )
        trial["is_correct"] = False
        with self.assertRaisesRegex(ContractError, "정답 위치"):
            validate_payload("trial", trial)

    def test_config_recalculates_visual_angle(self) -> None:
        config = _example("experiment_config")
        config["calibration"]["widths"][0]["width_deg"] = 99.0
        with self.assertRaisesRegex(ContractError, "재계산"):
            validate_payload("experiment_config", config)

    def test_config_rejects_zero_color_direction(self) -> None:
        config = _example("experiment_config")
        config["color_design"]["direction_lab"] = [0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ContractError, "영벡터"):
            validate_payload("experiment_config", config)

    def test_application_ratio_is_checked(self) -> None:
        conditions = _example("application_conditions")
        conditions["cases"][0]["stroke_width_ratio"] = 0.5
        with self.assertRaisesRegex(ContractError, "화면 폭"):
            validate_payload("application_conditions", conditions)

    def test_detection_failure_must_be_excluded(self) -> None:
        conditions = _example("application_conditions")
        conditions["cases"][0]["measurement_method"] = "detection_failed"
        conditions["cases"][0]["quality_flag"] = "usable"
        with self.assertRaisesRegex(ContractError, "excluded"):
            validate_payload("application_conditions", conditions)

    def test_detection_failure_uses_null_instead_of_fake_width(self) -> None:
        conditions = _example("application_conditions")
        row = conditions["cases"][0]
        row["measurement_method"] = "detection_failed"
        row["quality_flag"] = "excluded"
        row["stroke_width_px"] = None
        row["stroke_width_ratio"] = None
        validate_payload("application_conditions", conditions)

    def test_validation_trial_requires_staircase_id(self) -> None:
        trial = json.loads(
            EXAMPLE_PATHS["trial"].read_text(encoding="utf-8").splitlines()[0]
        )
        trial["phase"] = "validation"
        with self.assertRaisesRegex(ContractError, "staircase_id"):
            validate_payload("trial", trial)

    def test_palette_must_not_repeat_target_pair(self) -> None:
        pairs = _example("validation_pairs")
        pairs["pairs"][0]["palette"].append(
            copy.deepcopy(pairs["pairs"][0]["rgb_a"])
        )
        with self.assertRaisesRegex(ContractError, "A와 B"):
            validate_payload("validation_pairs", pairs)

    def test_c80_must_be_inside_confidence_interval(self) -> None:
        thresholds = _example("participant_thresholds")
        thresholds["participants"][0]["predictions"][0]["ci_low"] = 3.0
        with self.assertRaisesRegex(ContractError, "신뢰구간"):
            validate_payload("participant_thresholds", thresholds)

    def test_unstable_validation_cannot_have_pooled_c80(self) -> None:
        thresholds = _example("participant_thresholds")
        validation = thresholds["participants"][0]["validation"]
        validation["stable"] = False
        with self.assertRaisesRegex(ContractError, "pooled_c80"):
            validate_payload("participant_thresholds", thresholds)

    def test_correction_summary_must_match_rows(self) -> None:
        results = _example("correction_results")
        results["summary"]["corrected_count"] = 2
        with self.assertRaisesRegex(ContractError, "corrected_count"):
            validate_payload("correction_results", results)

    def test_correction_target_policy_must_match_selected_target(self) -> None:
        results = _example("correction_results")
        results["threshold"]["target_policy"] = "ci_high"
        with self.assertRaisesRegex(ContractError, "selected_target"):
            validate_payload("correction_results", results)

    def test_unchanged_result_cannot_claim_palette_search(self) -> None:
        results = _example("correction_results")
        results["results"][0]["palette_constraint_used"] = True
        with self.assertRaisesRegex(ContractError, "탐색을 생략"):
            validate_payload("correction_results", results)


if __name__ == "__main__":
    unittest.main()
