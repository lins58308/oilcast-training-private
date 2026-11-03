import datetime as dt
import unittest
from research.policy_features import feature_for_period, normalize_policy

TZ=dt.timezone(dt.timedelta(hours=8))

def origin(d):
    local=dt.datetime.fromisoformat(d+"T00:00:00").replace(tzinfo=TZ)
    return {"origin":d,"cutoff":local.astimezone(dt.timezone.utc).isoformat().replace("+00:00","Z")}

def notice(iso, id, total, theoretical, frozen=False, special=False):
    stamp=dt.datetime.fromisoformat(iso+"T23:59:59").replace(tzinfo=TZ)
    return {"id":str(id),"url":"https://www.cpc.com.tw/News_Content.aspx?n=28&s="+str(id),
      "publishedDate":iso,
      "assumedAvailableAt":stamp.astimezone(dt.timezone.utc).isoformat().replace("+00:00","Z"),
      "theoreticalGas":theoretical,"theoreticalDiesel":theoretical,
      "absorptionGas":total,"absorptionDiesel":total+1 if total is not None else None,
      "taxReliefGas":3.7,"taxReliefDiesel":2.1,
      "bothFuelFrozen":frozen,"specialAbsorption":special,
      "holidayNoRise":False}

class PolicyFeatureTest(unittest.TestCase):
    def test_known_policy_applies_only_after_publication(self):
        # Saturday announcement is available for following Monday, and
        # next Sunday target-week announcement must never leak backwards.
        earlier=notice("2026-10-31",100,11.6,7.9,True,True)
        future=notice("2026-11-08",101,30.0,20.0,False,False)
        payload={"schema":1,"notFuturePolicyAnswers":True,"notices":[future,earlier]}
        f,n=feature_for_period([origin("2026-10-26"),origin("2026-11-02"),
                                origin("2026-11-09")],payload)
        self.assertEqual(n,2)
        self.assertEqual(f[0][0], [0.]*12)
        self.assertEqual(f[1][0][0],1.)
        self.assertEqual(f[1][0][3],11.6)
        self.assertEqual(f[1][3][3],12.6)
        self.assertEqual(f[1][0][6],1.)
        self.assertEqual(f[1][0][7],1.)
        self.assertEqual(f[1][0][10],1/8)
        self.assertEqual(f[2][0][3],30.0)
        self.assertEqual(f[2][0][6],0.)
        self.assertEqual(f[1][0][1],7.9)

    def test_staleness_and_missingness(self):
        event=notice("2025-12-21",200,None,None,True,False)
        payload={"schema":1,"notFuturePolicyAnswers":True,"notices":[event]}
        f,n=feature_for_period([origin("2025-12-22"),origin("2026-01-19")],payload)
        self.assertEqual(n,1)
        self.assertEqual(f[0][0][0],1.)
        self.assertEqual(f[0][0][2],0.) # unknown theoretical != zero oil rise
        self.assertEqual(f[0][0][4],0.) # unknown absorption != no absorption
        self.assertEqual(f[1][0], [0.]*12)

    def test_reject_source_or_time_forgery(self):
        valid=notice("2026-10-31",100,11.6,7.9)
        with self.assertRaises(ValueError):
            feature_for_period([origin("2026-11-02")],
              {"schema":1,"notFuturePolicyAnswers":True,"notices":[dict(valid,url="https://example.com/fake")]})
        with self.assertRaises(ValueError):
            normalize_policy({"schema":1,"notFuturePolicyAnswers":True,"notices":[valid,valid]})
        with self.assertRaises(ValueError):
            feature_for_period([dict(origin("2026-11-02"),cutoff="2026-11-02T00:00:00Z")],
              {"schema":1,"notFuturePolicyAnswers":True,"notices":[valid]})

    def test_adjacent_week_changes_survive_equal_amounts_moves_and_missing_weeks(self):
        events=[notice('2026-10-17',1,5,None,True),notice('2026-10-24',2,6,None,True),notice('2026-10-31',3,6,None,True)]
        payload={'schema':1,'notFuturePolicyAnswers':True,'notices':events}
        f,_=feature_for_period([origin('2026-11-02')],payload)
        self.assertEqual(f[0][0][11],0.) # not 6 minus the older distinct 5
        events[-1]=notice('2026-10-31',3,9,None,False)
        f,_=feature_for_period([origin('2026-11-02')],payload)
        self.assertAlmostEqual(f[0][0][11],.3) # retained when price changes
        events.pop(1) # no observed policy in the immediately prior week
        f,_=feature_for_period([origin('2026-11-02')],payload)
        self.assertEqual(f[0][0][11],0.)

if __name__=="__main__":unittest.main()
