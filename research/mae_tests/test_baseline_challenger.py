import copy
import datetime as dt
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from baseline_challenger import Challenger, block_predict, conditional_action, feature_sets, summarize, online_guard, paired_uncertainty
from policy_state import NAMES, policy_states


def notice(day, amount, frozen=True, identity=None, tax=3.7):
    return {"id":identity or day,"url":"https://www.cpc.com.tw/News_Content.aspx?n=28&s=1",
        "publishedDate":day,"assumedAvailableAt":day+"T15:59:59Z",
        "absorptionGas":amount,"absorptionDiesel":amount,"taxReliefGas":tax,
        "taxReliefDiesel":tax,"theoreticalGas":None,"theoreticalDiesel":None,
        "bothFuelFrozen":frozen,"specialAbsorption":True,"holidayNoRise":False}


def states(items):
    row={"origin":"2026-08-31","cutoff":"2026-08-30T16:00:00Z"}
    return policy_states([row],{"schema":1,"notFuturePolicyAnswers":True,"notices":items})[0,0]


class PolicyStateTest(unittest.TestCase):
    def test_equal_amount_is_zero_change_not_change_since_last_distinct_amount(self):
        s=states([notice("2026-08-15",5.5),notice("2026-08-22",6),notice("2026-08-29",6)])
        self.assertEqual(s[NAMES.index("prior_week_absorption_change")],0)
        self.assertEqual(s[NAMES.index("change_known")],1)
        self.assertAlmostEqual(s[NAMES.index("net_cpc_absorption")],2.3)

    def test_change_remains_visible_when_current_week_is_not_frozen(self):
        s=states([notice("2026-08-22",6),notice("2026-08-29",9,False)])
        self.assertEqual(s[NAMES.index("prior_week_absorption_change")],3)
        self.assertEqual(s[NAMES.index("frozen")],0)

    def test_missing_week_is_masked_and_does_not_extend_freeze_run(self):
        s=states([notice("2026-08-15",6),notice("2026-08-29",9)])
        self.assertEqual(s[NAMES.index("change_known")],0)
        self.assertAlmostEqual(s[NAMES.index("consecutive_frozen_publication_weeks")],1/52)

    def test_same_week_duplicates_do_not_count_twice_or_replace_prior_week(self):
        s=states([notice("2026-08-22",6),notice("2026-08-28",8),notice("2026-08-29",6)])
        self.assertEqual(s[NAMES.index("prior_week_absorption_change")],0)
        self.assertAlmostEqual(s[NAMES.index("consecutive_frozen_publication_weeks")],2/52)

    def test_future_target_notice_never_enters_state(self):
        before=[notice("2026-08-22",6),notice("2026-08-29",9)]
        np.testing.assert_array_equal(states(before),states(before+[notice("2026-09-05",50)]))

    def test_missing_tax_does_not_become_known_zero_cpc_absorption(self):
        s=states([notice("2026-08-29",9,tax=None)])
        self.assertEqual(s[NAMES.index("net_known")],0)
        self.assertEqual(s[NAMES.index("tax_known")],0)

    def test_wrong_forecast_cutoff_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Monday 00:00'):
            policy_states([{'origin':'2026-08-31','cutoff':'2026-08-31T12:00:00Z'}],
                {'schema':1,'notFuturePolicyAnswers':True,'notices':[notice('2026-08-29',9)]})


class RelativeLossTest(unittest.TestCase):
    def test_conditional_median_includes_no_change_as_optimal_action(self):
        y=np.array([0.,0.,0.,3.])
        action,info=conditional_action(y,np.ones(4))
        self.assertEqual(action,0)
        mean=y.mean()
        self.assertLess(np.mean(abs(y-action)),np.mean(abs(y-mean)))

    def test_only_a_marginal_estimated_gain_is_suppressed_by_risk_margin(self):
        y=np.array([0.,1.]);weights=np.array([.49,.51])
        point,_=conditional_action(y,weights,0.)
        cautious,info=conditional_action(y,weights,1.)
        self.assertEqual(point,1)
        self.assertEqual(cautious,0)
        self.assertLess(info['estimatedGain'],info['gainStandardError'])

    def test_supported_movement_reduces_conditional_absolute_loss(self):
        y=np.array([0.,.4,.6]);weights=np.array([.1,.6,.3])
        action,_=conditional_action(y,weights)
        self.assertEqual(action,.4)
        self.assertLess(weights@abs(y-action),weights@abs(y))

    def test_future_training_labels_cannot_change_block_predictions(self):
        rng=np.random.default_rng(9)
        features={h:{'market':rng.normal(size=(150,5))} for h in range(2)}
        y=rng.choice([-.2,0.,.3],size=(150,2))
        spec=Challenger('test','kernel','market')
        expected,_=block_predict(features,y,120,130,spec)
        y[120:]=5.
        actual,_=block_predict(features,y,120,130,spec)
        np.testing.assert_array_equal(expected,actual)

    def test_feature_prefix_is_independent_of_future_source_rows(self):
        rng=np.random.default_rng(10)
        x=rng.normal(size=(100,4,26));base=np.full((100,4),30.)
        oil=np.zeros((100,7));policy=np.zeros((100,4,20))
        expected=feature_sets(x,base,oil,policy,'v7')
        x[50:]=200.;base[50:]=90.;oil[50:]=500.;policy[50:]=30.
        actual=feature_sets(x,base,oil,policy,'v7')
        for head in expected:
            for name in expected[head]:
                np.testing.assert_array_equal(expected[head][name][:50],actual[head][name][:50])

    def test_error_budget_explains_why_blind_positive_shrinkage_loses_to_flat(self):
        actual=np.zeros((13,4));actual[5,:]=.7
        positive=np.full((13,4),.1)
        summary=summarize(positive,actual)['92']
        self.assertAlmostEqual(summary['gainOnChangedWeeks'],.1)
        self.assertAlmostEqual(summary['costOnUnchangedWeeks'],1.2)
        self.assertLess(summary['netErrorReduction'],0)
        self.assertAlmostEqual(summary['gainRequiredFor0_01Mae'],.13)

    def test_online_guard_cannot_use_the_current_or_future_outcome(self):
        y=np.full((70,4),.3);p=np.full_like(y,.2)
        expected,flags=online_guard(y,p)
        self.assertTrue(flags[40].all())
        y[40:]=-10
        changed,_=online_guard(y,p)
        np.testing.assert_array_equal(expected[:41],changed[:41])
        self.assertFalse(flags[:26].any())

    def test_paired_uncertainty_keeps_zero_gain_at_zero(self):
        y=np.full((104,4),.3);pred=np.zeros_like(y)
        report=paired_uncertainty(y,pred,draws=20)
        self.assertEqual(report['allFourPositiveLowerQuantile'],0)
        for score in report['byFuel'].values():
            self.assertEqual(score['meanMaeReduction'],0)
            self.assertEqual(score['lower'],0)
            self.assertEqual(score['upper'],0)


if __name__=='__main__':unittest.main()
