import datetime as dt
import unittest
from research.official_daily_oil import parse_moea_daily, point_in_time_weekly


def csv_example():
    head="日期,西德州(美元/桶),杜拜(美元/桶),布蘭特(美元/桶),備註\n"
    values=[]
    current=dt.date(2000,1,3)
    while current<dt.date(2024,1,1):
        if current.weekday()<5:
            serial=(current-dt.date(2000,1,3)).days
            values.append(f"{current:%Y%m%d},{20+serial*.006:.2f},{22+serial*.007:.2f},{24+serial*.008:.2f},")
        current+=dt.timedelta(days=1)
    return head+"\n".join(reversed(values))+"\n"


def asof(iso):
    v=[0.]*26
    v[12]=1.0
    return {"cutoff":iso+"T00:00:00Z","features":[v]*4}


class OfficialDailyOilTest(unittest.TestCase):
    def test_csv_and_weekly_origin(self):
        source=parse_moea_daily(csv_example().encode("utf-8-sig"))
        self.assertGreater(len(source),4000)
        a,b=point_in_time_weekly([asof("2000-01-03"),asof("2010-02-15")],source)
        self.assertEqual(b,1)
        self.assertEqual(a[0],(0.,)*7)
        self.assertEqual(a[1][0],1.)
        self.assertEqual(len(a[1]),7)
        self.assertTrue(a[1][1]>0)
        self.assertGreater(a[1][2],0)

    def test_latest_three_days_and_future_never_used(self):
        source=parse_moea_daily(csv_example())
        earlier,_=point_in_time_weekly([asof("2010-02-15")],source)
        # By Monday, the latest date permitted is preceding Friday (Feb 12).
        source[dt.date(2010,2,15)]=(25.,400.,400.)
        source[dt.date(2010,2,14)]=(25.,400.,400.)
        source[dt.date(2010,2,13)]=(25.,400.,400.)
        later,_=point_in_time_weekly([asof("2010-02-15")],source)
        self.assertEqual(earlier,later)

    def test_reject_bad_history(self):
        with self.assertRaises(ValueError):
            parse_moea_daily(csv_example().replace("布蘭特(美元/桶)","Brent"))
        with self.assertRaises(ValueError):
            parse_moea_daily(csv_example()+"20100215,30,30,30,\n")
        with self.assertRaises(ValueError):
            parse_moea_daily("日期,西德州(美元/桶),杜拜(美元/桶),布蘭特(美元/桶),備註\n20261030,110,120,130,\n")


if __name__=="__main__":
    unittest.main()
