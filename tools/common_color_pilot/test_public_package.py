"""Synthetic-only checks for a clean checkout with no private study bundle."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import pilot
import repeat_pilot as repeat
import fit_width_prediction as prediction
import validate_width_pilot as validation
import analyze_repeat_pilot
import evaluate_validation_pilot

HERE=Path(__file__).resolve().parent

class PublicPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='_synthetic-',dir=HERE)
        self.root=Path(self.temp.name)
        self.old=[]
        for mod,name,value in [(pilot,'ROOT',self.root),(repeat,'ROOT',self.root),
            (repeat,'PLAN_PATH',self.root/'raw/_design_v3/design.json'),
            (prediction,'OUT',self.root/'raw/_analysis/width-prediction-pilot.json'),
            (validation,'ROOT',self.root),(validation,'DESIGN',self.root/'raw/_design_v4'),
            (validation,'PLAN',self.root/'raw/_design_v4/validation_plan.json'),
            (validation,'SNAPSHOT',self.root/'raw/_design_v4/model_snapshot.json')]:
            self.old.append((mod,name,getattr(mod,name)));setattr(mod,name,value)
        (self.root/'common_conditions.json').write_bytes((HERE/'common_conditions.json').read_bytes())

    def tearDown(self):
        for mod,name,old in reversed(self.old):setattr(mod,name,old)
        self.temp.cleanup()

    def write_json(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

    def synthetic_v3(self):
        source=self.root/'raw/v2/SYNTHETIC/responses.jsonl';source.parent.mkdir(parents=True)
        for t in pilot.schedule():
            # Deliberate synthetic contrast: only these public design paths vary
            # by separation. No real response or fitted personal parameter used.
            right=t['level']>=3 if t['condition_id'] in ['c10_mix000','c11_plane020'] else True
            answer=t['correct_position'] if right else t['correct_position']%4+1
            pilot.append(source,{'trial':t,'response_position':answer,'correct':right,'response_time_ms':1000,'utc':pilot.now()})
        self.write_json(self.root/'raw/_analysis/complete-v2-analysis.json',
            {'source_relative':source.relative_to(self.root).as_posix(),'source_sha256':pilot.sha(source)})
        repeat.prepare();repeat.self_test()
        source=self.root/'raw/v3/SYNTHETIC/responses.jsonl';source.parent.mkdir(parents=True)
        cp=source.parent/'config.json';self.write_json(cp,{'design_sha256':pilot.sha(repeat.PLAN_PATH),'synthetic':True})
        for t in repeat.schedule():
            right=t['level']>=2;answer=t['correct_position'] if right else t['correct_position']%4+1
            pilot.append(source,{'trial':t,'response_position':answer,'correct':right,'response_time_ms':1000,
                'utc':pilot.now(),'config_sha256':pilot.sha(cp)})
        analyze_repeat_pilot.main()
        return source

    def test_common_assets_are_original_design_not_responses(self):
        self.assertEqual(pilot.sha(HERE/'source_conditions_v1.json'),'f47393515811c02a6111a0f4d49642e97e102a2f3a98e5cfccca0f2b25495d6c')
        self.assertEqual(pilot.sha(HERE/'legacy_color_math.py'),'26a1612cb7560e0222c51479a48b01e5166421f63621dcca08bb31b1b2eb25eb')
        self.assertEqual(len(pilot.load_design()['conditions']),12)

    def test_v2_schedule_and_raster_and_resume_without_private_data(self):
        with contextlib.redirect_stdout(io.StringIO()):pilot.self_test()

    def test_grid_optimizer_and_range_guards(self):
        with contextlib.redirect_stdout(io.StringIO()):prediction.self_test()

    def test_complete_synthetic_v3_to_v4_no_refit_or_stale_conclusions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            source=self.synthetic_v3()
            report=json.loads((self.root/'raw/_analysis/repeat-v3-analysis.json').read_text(encoding='utf-8'))
            self.assertIsNone(report['self_report']['same_spectacles_throughout'])
            self.assertIsNone(report['self_report']['observation'])
            self.assertEqual(report['overall']['correct'],160)
            colors=sorted(c['condition_id'] for c in repeat.plan()['conditions'])
            models=[]
            for name in prediction.MODELS:
                models.append({'model':name,'full_fit':{'model':name,'alpha':[-.6,-.8],
                    'slope':2.,'gamma':0. if name=='constant' else 2.,'lapse':.04}})
            self.write_json(prediction.OUT,{'source_relative':source.relative_to(self.root).as_posix(),
                'source_sha256':pilot.sha(source),'colors':colors,'models':models})
            validation.prepare();validation.self_test();snapshot_before=pilot.sha(validation.SNAPSHOT)
            plan_before=pilot.sha(validation.PLAN)
            validation.prepare()
            self.assertEqual(plan_before,pilot.sha(validation.PLAN))
            val_source=self.root/'raw/v4/SYNTHETIC/responses.jsonl';val_source.parent.mkdir(parents=True)
            cp=val_source.parent/'config.json'
            self.write_json(cp,{'plan_sha256':plan_before,'model_snapshot_sha256':snapshot_before,'synthetic':True})
            for t in validation.schedule():
                right=t['level']>=3;answer=t['correct_position'] if right else t['correct_position']%4+1
                pilot.append(val_source,{'trial':t,'response_position':answer,'correct':right,
                    'response_time_ms':1000,'utc':pilot.now(),'config_sha256':pilot.sha(cp)})
            evaluate_validation_pilot.main()
            result=json.loads((self.root/'raw/_analysis/validation-v4-results.json').read_text(encoding='utf-8'))
            self.assertFalse(result['model_refit_performed'])
            self.assertTrue(result['all_responses_after_plan_lock'])
            self.assertEqual(result['overall']['correct'],72)
            self.assertEqual(result['model_scores']['constant']['n'],120)
            self.assertEqual(snapshot_before,pilot.sha(validation.SNAPSHOT))
            text=(self.root/'raw/_analysis/validation-v4-results.md').read_text(encoding='utf-8')
            self.assertIn('72/120',text)
            self.assertNotIn('약 89~91%',text)

    def test_explicit_annotations_only(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.synthetic_v3();analyze_repeat_pilot.main(True,True)
        data=json.loads((self.root/'raw/_analysis/repeat-v3-analysis.json').read_text(encoding='utf-8'))
        self.assertTrue(data['self_report']['same_spectacles_throughout'])
        self.assertIsNotNone(data['self_report']['observation'])

    def test_reject_corrupt_or_mismatched_journal(self):
        path=self.root/'bad.jsonl';path.write_text('{broken',encoding='utf-8')
        with self.assertRaises(json.JSONDecodeError):pilot.read_rows(path)
        t=pilot.schedule()[0]
        with self.assertRaises(ValueError):pilot.verify_rows([{'trial':t,'response_position':0,'correct':False}],pilot.schedule())

if __name__=='__main__':unittest.main()
