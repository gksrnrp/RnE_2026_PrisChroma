import copy
import unittest

from recovery_coverage import MODELS, PROFILES, summarise_coverage, validate_coverage_payload


def coverage_fixture(bootstrap_count=200):
    rows = []
    for i in range(60):
        rows.append({'outer_index':i, 'profile':PROFILES[i % 3],
            'data_seed':900001+11*i, 'bootstrap_seed':1900001+11*i,
            'models':{model:{'fit_converged':True,
                'bootstrap_successful':bootstrap_count,
                'covered':({'c80':True, 'beta':True} if model=='D384' else {'c80':True})}
                for model in MODELS}})
    return {**summarise_coverage(rows, bootstrap_count), 'outer_records':rows,
        'seed':1, 'seed_rule':'data=seed+900000+11*i; bootstrap=seed+1900000+11*i'}


class RepeatedCoverageTests(unittest.TestCase):
    def test_complete_repeated_coverage_passes(self):
        self.assertTrue(validate_coverage_payload(coverage_fixture()))

    def test_one_missed_interval_does_not_veto_repeated_coverage(self):
        payload=coverage_fixture()
        payload['outer_records'][0]['models']['D384']['covered']['beta']=False
        payload.update(summarise_coverage(payload['outer_records'],200))
        self.assertTrue(validate_coverage_payload(payload))

    def test_poor_single_profile_fails_despite_good_aggregate(self):
        payload=coverage_fixture()
        for i in range(0,15,3):
            payload['outer_records'][i]['models']['D384']['covered']['beta']=False
        payload.update(summarise_coverage(payload['outer_records'],200))
        self.assertGreater(payload['models']['D384']['beta']['observed_fraction'],.90)
        self.assertFalse(validate_coverage_payload(payload))

    def test_missing_or_forged_record_fails(self):
        payload=coverage_fixture()
        payload['outer_records'].pop()
        self.assertFalse(validate_coverage_payload(payload))
        payload=coverage_fixture()
        payload['models']['D384']['beta']['included']=59
        self.assertFalse(validate_coverage_payload(payload))

    def test_failed_fit_counts_as_a_miss(self):
        payload=coverage_fixture()
        for i in range(10):
            payload['outer_records'][i]['models']['D384']['fit_converged']=False
        payload.update(summarise_coverage(payload['outer_records'],200))
        self.assertEqual(payload['models']['D384']['beta']['included'],50)
        self.assertFalse(validate_coverage_payload(payload))

    def test_insufficient_bootstrap_cannot_pass(self):
        self.assertFalse(validate_coverage_payload(coverage_fixture(40)))


if __name__=='__main__':
    unittest.main()
