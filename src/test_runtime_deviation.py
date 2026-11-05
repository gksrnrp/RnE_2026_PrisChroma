import json
import tempfile
import unittest
from pathlib import Path

from response_gui import (
    ResponseGUIBlockedError,
    ScreenMetrics,
    validate_screen_metrics,
)
from runtime_deviation import (
    LOG_PATH,
    RECORD_PATH,
    RuntimeDeviationError,
    append_session_log,
    load_os_deviation,
)

FROZEN = "Windows-11-10.0.26200-SP0"
OBSERVED = "Windows-11-10.0.26300-SP0"
RUNTIME = "CPython 3.13.15; Tk 8.6.15"


def record(**changes):
    base = {
        "artifact_type": "runtime_os_build_deviation_v1",
        "approved": True,
        "approved_by": "tester",
        "allowed_phases": ["practice", "calibration"],
        "config_sha256": "c" * 64,
        "frozen_operating_system": FROZEN,
        "observed_operating_system": OBSERVED,
        "base_commit": "a" * 40,
        "execution_commit": "b" * 40,
    }
    base.update(changes)
    return base


class LoadOsDeviationTests(unittest.TestCase):
    def load(self, payload, *, phase="calibration", observed=OBSERVED, head="b" * 40,
             changed=("src/response_gui.py",)):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            if payload is not None:
                path = root / RECORD_PATH
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(payload), encoding="utf-8")
            return load_os_deviation(
                root, phase=phase, config_sha256="c" * 64,
                frozen_operating_system=FROZEN, observed_operating_system=observed,
                head_commit=head, changed_files=lambda *_: changed,
            )

    def test_valid_record_is_returned(self):
        self.assertEqual(self.load(record())["base_commit"], "a" * 40)

    def test_no_record_or_same_os_gives_no_exception(self):
        self.assertIsNone(self.load(None))
        self.assertIsNone(self.load(record(), observed=FROZEN))

    def test_any_binding_mismatch_blocks(self):
        for changes in (
            {"approved": False},
            {"approved_by": " "},
            {"config_sha256": "d" * 64},
            {"frozen_operating_system": "other"},
            {"observed_operating_system": "Windows-11-10.0.99999-SP0"},
            {"execution_commit": "e" * 40},
            {"allowed_phases": ["practice"]},
            {"artifact_type": "other"},
        ):
            with self.subTest(changes), self.assertRaises(RuntimeDeviationError):
                self.load(record(**changes))

    def test_practice_is_covered_for_the_calibration_gate(self):
        self.assertIsNotNone(self.load(record(), phase="practice"))

    def test_validation_cannot_be_enabled_by_record(self):
        payload = record(allowed_phases=["practice", "calibration", "validation"])
        with self.assertRaises(RuntimeDeviationError):
            self.load(payload, phase="validation")

    def test_stimulus_or_analysis_change_blocks(self):
        for name in ("src/stimulus_renderer.py", "src/psychometrics.py",
                     "schemas/human/trial.schema.json"):
            with self.subTest(name), self.assertRaises(RuntimeDeviationError):
                self.load(record(), changed=("src/response_gui.py", name))


class ScreenGateTests(unittest.TestCase):
    config = {"display": {
        "operating_system": FROZEN, "native_gui_runtime": RUNTIME,
        "resolution_width_px": 3840, "resolution_height_px": 2160,
        "app_scale_percent": 100, "os_scale_percent": 150,
    }}

    def metrics(self, width=3840, runtime=RUNTIME):
        return ScreenMetrics(width, 2160, 150.0, True, "test", OBSERVED, runtime)

    def test_os_mismatch_still_blocks_without_approval(self):
        with self.assertRaises(ResponseGUIBlockedError):
            validate_screen_metrics(self.config, self.metrics())

    def test_only_the_approved_os_string_passes(self):
        validate_screen_metrics(self.config, self.metrics(), approved_observed_os=OBSERVED)
        with self.assertRaises(ResponseGUIBlockedError):
            validate_screen_metrics(self.config, self.metrics(), approved_observed_os="other")

    def test_other_display_gates_are_not_waived(self):
        for metrics in (self.metrics(width=2560), self.metrics(runtime="CPython 3.12; Tk 8.6")):
            with self.subTest(metrics), self.assertRaises(ResponseGUIBlockedError):
                validate_screen_metrics(self.config, metrics, approved_observed_os=OBSERVED)


class SessionLogTests(unittest.TestCase):
    def test_log_records_real_os_and_both_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / RECORD_PATH
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(record()), encoding="utf-8")
            for session in ("S2", "S3"):
                append_session_log(root, record(), session_id=session,
                                   phase="calibration", session_block=2)
            rows = [json.loads(line) for line in
                    (root / LOG_PATH).read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["session_id"] for row in rows], ["S2", "S3"])
        self.assertEqual(rows[0]["observed_operating_system"], OBSERVED)
        self.assertEqual(rows[0]["stimulus_commit"], "a" * 40)
        self.assertEqual(rows[0]["execution_commit"], "b" * 40)


if __name__ == "__main__":
    unittest.main()
