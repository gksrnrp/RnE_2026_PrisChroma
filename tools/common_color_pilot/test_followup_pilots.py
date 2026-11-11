"""Follow-up checks using public colors and synthetic settings only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import lum_pilot as lum
import pilot
import run_v5
import wide_pilot as wide


class FollowupPilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='_synthetic-', dir=pilot.ROOT)
        self.root = Path(self.temp.name)
        self.source = self.root / 'v3.json'
        conditions = []
        for color in pilot.load_design()['conditions'][:2]:
            pairs = []
            for level, fraction in enumerate(pilot.FRACTIONS[2:5], 3):
                t = fraction * color['t_max']
                pairs.append({
                    'source_level': level, 'fraction': fraction, 'separation_t': t,
                    'rgb_a': pilot.encode_rgb([b-t*d/2 for b, d in zip(color['base_linear'], color['direction'])]),
                    'rgb_b': pilot.encode_rgb([b+t*d/2 for b, d in zip(color['base_linear'], color['direction'])]),
                })
            conditions.append({'condition_id': color['cid'], 'pairs': pairs})
        self.source.write_text(json.dumps({'conditions': conditions}), encoding='utf-8')
        self.observer = self.root / 'synthetic-observer.json'
        self.model = {'cvd_type': 'protan', 'severity': 0.5, 'source': 'SYNTHETIC_TEST_ONLY'}
        self.observer.write_text(json.dumps(self.model), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_v5_schedule_rasters_resume_and_frozen_plan(self):
        with contextlib.redirect_stdout(io.StringIO()):
            wide.self_test()

    def test_v5_launcher_rejects_other_resolutions_before_start(self):
        root = SimpleNamespace(winfo_screenwidth=lambda: 1920, winfo_screenheight=lambda: 1080)
        app = SimpleNamespace(root=root, messagebox=Mock())
        with patch.object(wide.App, 'start') as start:
            run_v5.GuardedApp.start(app)
        app.messagebox.showerror.assert_called_once()
        start.assert_not_called()

    def test_v5_launcher_continues_at_required_resolution(self):
        app = object.__new__(run_v5.GuardedApp)
        app.root = SimpleNamespace(winfo_screenwidth=lambda: 3840, winfo_screenheight=lambda: 2160)
        with patch.object(wide.App, 'start') as start:
            app.start()
        start.assert_called_once()

    def test_v6_requires_explicit_valid_model_and_source(self):
        for value in ({}, {'cvd_type': 'protan'}, dict(self.model, severity=True),
                      dict(self.model, severity=-0.1), dict(self.model, severity=1.1),
                      dict(self.model, severity=float('nan')), dict(self.model, source=''),
                      dict(self.model, cvd_type='unknown')):
            with self.subTest(value=value):
                self.observer.write_text(json.dumps(value), encoding='utf-8')
                with self.assertRaises(ValueError):
                    lum.load_observer(self.observer)

    def test_v6_schedule_luminance_rasters_and_no_plan_overwrite(self):
        target = self.root / 'v6.json'
        with patch.object(lum, 'V3_PLAN_PATH', self.source), patch.object(lum, 'PLAN_PATH', target):
            with contextlib.redirect_stdout(io.StringIO()):
                lum.prepare(self.observer)
            digest = pilot.sha(target)
            self.assertEqual(lum.plan()['observer_model'], self.model)
            lum.check_schedule(lum.schedule())
            with contextlib.redirect_stdout(io.StringIO()):
                lum.prepare(self.observer)
            self.assertEqual(pilot.sha(target), digest)
            self.observer.write_text(json.dumps(dict(self.model, source='DIFFERENT_SOURCE')), encoding='utf-8')
            with self.assertRaises(ValueError):
                lum.prepare(self.observer)
            self.assertEqual(pilot.sha(target), digest)


if __name__ == '__main__':
    unittest.main()
