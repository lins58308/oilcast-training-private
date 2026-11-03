import datetime as dt
import unittest
from research.dubai_proxy import month_end, parse_dubai_csv, lagged_dubai_features


def fixture():
    lines=["observation_date,POILDUBUSDM"]
    y,m=1990,1
    while (y,m)<(2016,1):
        lines.append(f"{y:04d}-{m:02d}-01,{10+(y-1990)*1.0+m/10:.2f}")
        m+=1
        if m==13:y,m=y+1,1
    return "\n".join(lines)


def week(d):
    v=[0.0]*26
    v[10]=2.0
    v[12]=3.0
    return {"cutoff":d+"T00:00:00Z","features":[v]*4}


class DubaiPointInTimeTest(unittest.TestCase):
    def test_release_and_month_end(self):
        self.assertEqual(month_end(dt.date(2000,2,1)),dt.date(2000,2,29))
        records=parse_dubai_csv(fixture())
        january=next(a for a in records if a[1]==dt.date(2000,1,1))
        self.assertEqual(january[0],dt.date(2000,4,14))
        sampled,n=lagged_dubai_features([week("2000-04-13"),week("2000-04-14"),
                                          week("2000-05-15")],records)
        self.assertEqual(n,3)
        # January's full monthly Dubai mean must not leak before +75d.
        self.assertLess(sampled[0][1],sampled[1][1])
        self.assertEqual(sampled[1][1],(10+10+.1)/100)
        self.assertGreater(sampled[2][1],sampled[1][1])

    def test_missing_or_duplicate(self):
        with self.assertRaises(ValueError):
            parse_dubai_csv(fixture().replace("observation_date","date"))
        with self.assertRaises(ValueError):
            parse_dubai_csv(fixture()+"\n2000-01-01,25.0")

    def test_no_future_and_missing_coverage(self):
        records=parse_dubai_csv(fixture())
        out,n=lagged_dubai_features([week("1989-01-01"),week("2002-01-01")],records)
        self.assertEqual(out[0],(0.,0.,0.,0.,0.))
        self.assertEqual(n,1)
        self.assertTrue(0<=out[1][4]<=1)


if __name__=="__main__":
    unittest.main()
