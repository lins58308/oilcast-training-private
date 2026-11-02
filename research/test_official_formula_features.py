"""CPC 7D3B proxy audit: no late monthly averages in old weekly origins."""
from datetime import date
import importlib.util
from pathlib import Path
import unittest

def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

formula = load("official_formula_features")

class DubaiChronologyTests(unittest.TestCase):
    def test_real_csv_shape_and_gaps(self):
        rows = formula.parse_monthly_dubai(
            "observation_date,POILDUBUSDM\n2019-01-01,50\n2019-02-01,55\n2019-03-01,.\n")
        self.assertEqual(len(rows), 2)
        with self.assertRaises(ValueError):
            formula.parse_monthly_dubai("DATE,INVALID\n2019-01-01,5")

    def test_62_day_month_end_embargo(self):
        rows = [(date(2020, 1, 1), 100.), (date(2020, 2, 1), 110.),
                (date(2020, 3, 1), 120.)]
        # Feb monthly average is not yet available by Feb/Mar and neither
        # is its eventual final value visible in early April under embargo.
        for asof in ["2020-02-24T00:00:00Z", "2020-03-16T00:00:00Z",
                     "2020-04-01T00:00:00Z"]:
            self.assertEqual(formula.one_origin_proxy(asof, 3., 1., rows), [0., 0., 0., 0.])
        a = formula.one_origin_proxy("2020-04-14T00:00:00Z", 3., 1., rows)
        self.assertEqual(a[3], 1.)
        self.assertAlmostEqual(a[0], 10.)
        self.assertAlmostEqual(a[1], 7.9)
        b = formula.one_origin_proxy("2020-05-01T00:00:00Z", 3., 1., rows)
        self.assertEqual(a, b)  # March cannot enter until May/June.

    def test_reject_unavailable_before_1999(self):
        data = {"examples": [
            {"cutoff": "1998-12-01T00:00:00Z", "features": [[0.] * 26 for _ in range(4)]},
            {"cutoff": "2020-04-20T00:00:00Z", "features": [[0.] * 26 for _ in range(4)]}
        ]}
        series = [(date(2020, 1, 1), 100.), (date(2020, 2, 1), 110.)]
        formula.attach_dubai_proxy(data, series)
        self.assertEqual(data["examples"][0]["policyProxy"], [0., 0., 0., 0.])
        self.assertEqual(data["examples"][1]["policyProxy"][-1], 1.)
        self.assertFalse(data["policyProxyProvenance"]["isHistoricalReleaseVintage"])

if __name__ == "__main__":
    unittest.main()
