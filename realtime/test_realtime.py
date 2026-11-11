"""Synthetic regression checks; no participant profile or screen capture is read."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import common as c
from evaluate import corrected_color, json_safe
from gen_teacher_labels import check_resume
from pipeline import RealtimeCorrector, Settings
from scene_gen import make_scene
from student import Student, make_features


HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / 'models/student_v1/student.npz'


def profile(**overrides):
    fields = dict(participant_id='SYNTHETIC_ONLY', cvd_type='deutan', severity=0.5,
                  c80_ref=3.0, s_ref_deg=0.1, beta=-0.2, s_min_deg=0.05,
                  s_max_deg=0.3, original_min=10.0, source='SYNTHETIC_TEST_NOT_MEASURED')
    return c.UserProfile(**(fields | overrides))


class RealtimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.student = Student(WEIGHTS)
        cls.display = c.DisplayGeometry(500.0, 1920, 600.0)

    def test_original_numeric_model_is_preserved(self):
        self.assertEqual(hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
                         '16836208b210715681c48dee3782ff3ec2ead1ef4bb47b2e47735d5920c30b89')
        self.assertEqual([w.shape for w, _ in self.student.layers],
                         [(256, 12), (256, 256), (256, 256), (6, 256)])

    def test_success_is_rechecked_for_multiple_synthetic_pairs_and_types(self):
        rng = np.random.default_rng(20261111)
        a = rng.integers(20, 236, (64, 3))
        b = np.rint(np.clip(a + rng.normal(0, 15, (64, 3)), 0, 255))
        types = np.array(['protan', 'deutan'] * 32)
        severity = rng.uniform(0.3, 1.0, 64)
        target = rng.uniform(1.0, 8.0, 64)
        na, nb, ok, scales = self.student.correct(a, b, types, severity, target)
        self.assertTrue(ok.any())
        needs = np.array([c.simulated_delta_e(a[i], b[i], types[i], severity[i]) < target[i] for i in range(64)])
        self.assertTrue((needs & ok).any())
        self.assertTrue(np.equal(na, np.rint(na)).all())
        self.assertTrue(np.equal(nb, np.rint(nb)).all())
        self.assertTrue(((na >= 0) & (na <= 255)).all())
        for i in np.flatnonzero(ok):
            actual = float(c.simulated_delta_e(na[i], nb[i], types[i], severity[i]))
            self.assertGreaterEqual(actual, target[i])
        np.testing.assert_array_equal(na[~ok], a[~ok])
        np.testing.assert_array_equal(nb[~ok], b[~ok])
        self.assertTrue(np.isnan(scales[~ok]).all())

    def test_failed_zero_direction_keeps_original_colors(self):
        with tempfile.TemporaryDirectory(dir=HERE, prefix='_synthetic-') as folder:
            path = Path(folder) / 'zero.npz'
            np.savez(path, mean=np.zeros(12), std=np.ones(12), n_layers=1,
                     W0=np.zeros((6, 12)), b0=np.zeros(6))
            a, b = [[120, 120, 120]], [[121, 120, 120]]
            na, nb, ok, scale = Student(path).correct(a, b, 'deutan', 0.5, 10.0)
            self.assertFalse(ok[0])
            self.assertTrue(np.isnan(scale[0]))
            np.testing.assert_array_equal(na, a)
            np.testing.assert_array_equal(nb, b)

    def test_invalid_types_targets_and_colors_are_rejected(self):
        a, b = [[80, 100, 120]], [[90, 110, 130]]
        cases = [(a, b, 'unknown', 0.5, 3), (a, b, 'protan', np.nan, 3),
                 (a, b, 'deutan', 1.1, 3), (a, b, 'protan', 0.5, 0),
                 (a, b, 'protan', 0.5, np.nan), ([[256, 0, 0]], b, 'protan', 0.5, 3)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                make_features(*args)

    def test_scale_grid_must_have_a_defined_order(self):
        for scales in ([], [2, 1], [0, 1], [1, np.nan]):
            with self.subTest(scales=scales), self.assertRaises(ValueError):
                self.student.correct([[80, 100, 120]], [[90, 110, 130]], 'protan', 0.5, 3, scales)

    def test_invalid_model_normalization_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE, prefix='_synthetic-') as folder:
            path = Path(folder) / 'bad.npz'
            np.savez(path, mean=np.zeros(12), std=np.zeros(12), n_layers=1,
                     W0=np.zeros((6, 12)), b0=np.zeros(6))
            with self.assertRaises(ValueError):
                Student(path)

    def test_empty_profile_templates_cannot_run(self):
        with self.assertRaises(ValueError):
            c.UserProfile.load(HERE / 'profiles/user.template.json')
        with self.assertRaises(ValueError):
            c.DisplayGeometry(**json.loads((HERE / 'profiles/display.template.json').read_text()))

    def test_profile_requires_source_and_consistent_range(self):
        for fields in ({'source': ''}, {'severity': 2}, {'c80_ref': 0}, {'beta': np.nan},
                       {'s_min_deg': 0.2}, {'cvd_type': 'other'}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                profile(**fields)

    def test_threshold_clamps_without_extrapolation(self):
        p = profile()
        np.testing.assert_allclose(p.c80([0.001, 0.05, 0.3, 10]), p.c80([0.05, 0.05, 0.3, 0.3]))
        np.testing.assert_array_equal(p.out_of_range([0.001, 0.1, 10]), [True, False, True])

    def test_none_and_sparse_palette_preserve_pixels(self):
        rng = np.random.default_rng(31)
        frame = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        before = frame.copy()
        for mode in ('none', 'student'):
            rc = RealtimeCorrector(profile(), self.display, mode, self.student)
            out, _ = rc.process(frame, instant=True)
            np.testing.assert_array_equal(out, before)
        np.testing.assert_array_equal(frame, before)

    def test_lut_sampling_matches_pixel_application(self):
        colors = np.array([[40, 50, 60], [100, 130, 160]], dtype=np.uint8)
        rc = RealtimeCorrector(profile(), self.display, 'student', self.student)
        rc.delta[10, 12, 15] = [2, -3, 4]
        rc._version += 1
        frame = colors.reshape(1, 2, 3)
        out = rc.apply(frame)
        for i, color in enumerate(colors):
            np.testing.assert_array_equal(out[0, i], corrected_color(rc, color))
        np.testing.assert_array_equal(out[0, 1], colors[1])

    def test_multiple_synthetic_scenes_run_without_mutating_input(self):
        for seed in (771, 772, 773):
            frame, meta = make_scene(seed, profile(), self.display, height=240, width=426)
            before = frame.copy()
            rc = RealtimeCorrector(profile(), self.display, 'student', self.student,
                                   settings=Settings(aa_unmix=True))
            out, _ = rc.process(frame, instant=True)
            again, timing = rc.process(frame, instant=True)
            self.assertEqual(out.shape, frame.shape)
            self.assertEqual(out.dtype, np.uint8)
            np.testing.assert_array_equal(frame, before)
            np.testing.assert_array_equal(out, again)
            self.assertNotIn('analyzed', timing)

    def test_reverted_pair_moves_are_removed_from_final_palette(self):
        colors = np.array([[117, 117, 117], [80, 90, 100], [90, 100, 110], [100, 110, 120]])
        class FakeStudent:
            count = 0
            def correct(self, a, b, *args):
                self.count += 1
                new = colors[1] + 1 if self.count == 1 else colors[1]
                return new[None], b[None], np.array([True]), np.array([1.0])
        rc = RealtimeCorrector(profile(), self.display, 'student', FakeStudent())
        pairs = [dict(i=1, j=j, sim=j, need=200, confused=True) for j in (2, 3)]
        self.assertEqual(rc.correct(colors, pairs), {})

    def test_missing_metrics_serialize_as_standard_json_null(self):
        value = json_safe({'metrics': [float('nan'), float('inf'), 1.0]})
        self.assertEqual(json.loads(json.dumps(value, allow_nan=False)), {'metrics': [None, None, 1.0]})

    def test_label_resume_rejects_changed_settings_and_records(self):
        with tempfile.TemporaryDirectory(dir=HERE, prefix='_synthetic-') as folder:
            out = Path(folder)
            labels = out / 'labels.jsonl'
            tasks = [dict(id=0, a=[10, 20, 30], target=3.0)]
            self.assertEqual(check_resume(out, labels, tasks, {'seed': 1}), set())
            labels.write_text(json.dumps(tasks[0]) + '\n', encoding='utf-8')
            before = labels.read_bytes()
            self.assertEqual(check_resume(out, labels, tasks, {'seed': 1}), {0})
            with self.assertRaises(ValueError):
                check_resume(out, labels, tasks, {'seed': 2})
            self.assertEqual(labels.read_bytes(), before)
            labels.write_bytes(before + before)
            with self.assertRaises(ValueError):
                check_resume(out, labels, tasks, {'seed': 1})


if __name__ == '__main__':
    unittest.main()
