"""No-future-policy test for the 0.3B-text-encoder's small time-series head."""
import datetime as dt
import unittest
import numpy as np
from research.train_semantic_teacher import prepare

TZ=dt.timezone(dt.timedelta(hours=8))


def period(iso):
    d=dt.date.fromisoformat(iso)
    cutoff=dt.datetime.combine(d,dt.time(0),tzinfo=TZ).astimezone(dt.timezone.utc)
    return {
        "origin":iso,"targetDate":(d+dt.timedelta(days=7)).isoformat(),
        "cutoff":cutoff.isoformat().replace("+00:00","Z"),
        "features":[[0.]*26 for _ in range(4)],
        "nextWeekDeltas":[0.]*4,"policyProxy":[0.]*4,
    }


class DeepPolicyTest(unittest.TestCase):
    def test_augmentation_is_causal_and_shape_stable(self):
        monday=dt.date(2023,1,2)
        weeks=[period((monday+dt.timedelta(weeks=i)).isoformat()) for i in range(120)]
        semantic=np.zeros((len(weeks),16),dtype=np.float32)
        last=dt.date(2023,4,9)
        available=dt.datetime.combine(last,dt.time(23,59,59),tzinfo=TZ).astimezone(dt.timezone.utc)
        notice={"id":"123","url":"https://www.cpc.com.tw/News_Content.aspx?n=28&s=123",
                "publishedDate":last.isoformat(),
                "assumedAvailableAt":available.isoformat().replace("+00:00","Z"),
                "theoreticalGas":1.,"theoreticalDiesel":.5,
                "absorptionGas":5.,"absorptionDiesel":6.,
                "taxReliefGas":2.2,"taxReliefDiesel":1.6,
                "bothFuelFrozen":True,"specialAbsorption":True,"holidayNoRise":False}
        policy={"schema":1,"notFuturePolicyAnswers":True,"notices":[notice]}
        normal,_,_,_,perfuel=prepare(weeks,semantic)
        augmented,_,_,_,policy_perfuel=prepare(weeks,semantic,policy)
        self.assertEqual(perfuel,30)
        self.assertEqual(policy_perfuel,42)
        self.assertEqual(augmented.shape[-1]-normal.shape[-1],48)
        self.assertTrue(np.isfinite(augmented).all())
        # A future Sunday article appears only in the next week's origin.
        # The immediately previous Monday must not incorporate it.
        # Input sequence of four weeks: x[10] ends on week t=13
        # (April 3), x[11] ends on week t=14 (April 10).
        # Features are normalized by train-only statistics, not 0/1.
        self.assertLess(augmented[10, -1, 30],augmented[11, -1, 30])


if __name__=="__main__":unittest.main()
