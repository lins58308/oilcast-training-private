import copy
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mae_walk_forward import Spec, choose, fit_predict, load_frozen, matrices, metrics, round_prices


class MaeChronologyTest(unittest.TestCase):
    def test_future_labels_cannot_change_earlier_prediction_or_scaler(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(140, 4))
        y = .2*x[:, 0]+rng.normal(scale=.05, size=140)
        spec = Spec("test", "linear-mae", "market", .01)
        expected = fit_predict(x, y, 110, 120, spec)
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[120:] = 1e9
        changed_y[110:] = -1e9
        np.testing.assert_array_equal(expected, fit_predict(changed_x, changed_y, 110, 120, spec))

    def test_regime_and_temporal_features_only_see_current_and_past_origins(self):
        rng = np.random.default_rng(4)
        num = rng.normal(size=(100, 4, 26))
        base = np.ones((100, 4))*30
        oil, policy = np.zeros((100, 7)), np.zeros((100, 4, 12))
        expected = matrices(num, base, oil, policy, "v7")
        num[50:] = 999
        changed = matrices(num, base, oil, policy, "v7")
        for group in expected:
            for family in expected[group]:
                np.testing.assert_array_equal(expected[group][family][:50], changed[group][family][:50])

    def test_one_bad_gasoline_grade_cannot_hide_behind_pooled_average(self):
        from mae_walk_forward import SPECS
        baseline = [[dict(mae=.3, directionAccuracy=.5) for _ in range(4)] for _ in range(5)]
        good = dict(mae=.2, directionAccuracy=.6, falseMoveRate=.1, changedWeekRecall=.4)
        history = {s.name: [[dict(good) for _ in range(4)] for _ in range(5)] for s in SPECS}
        self.assertNotEqual(choose(history, baseline)[0]["selected"], "unchanged")
        for scores in history.values():
            for fold in scores:
                fold[2]["mae"] = .31
        choices = choose(history, baseline)
        self.assertEqual(choices[0]["selected"], "unchanged")
        self.assertNotEqual(choices[1]["selected"], "unchanged")

    def test_no_change_week_errors_and_unidentified_recall(self):
        score = metrics(np.zeros(4), np.array([0, .1, 0, -.1]))
        self.assertAlmostEqual(score["mae"], .05)
        self.assertEqual(score["falseMoveRate"], .5)
        self.assertIsNone(score["changedWeekRecall"])

    def test_matches_java_price_rounding_including_negative_delta(self):
        got = round_prices(np.array([.05, -.05, .15, -.15]), np.full(4, 20.))
        np.testing.assert_allclose(got, [.1, 0, .2, -.1], atol=1e-10)

    def test_frozen_input_hash_mismatch_stops_before_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            (root/"manifest.json").write_text(json.dumps({"variant":"v6", "sha256":{"dataset.json":"wrong"}}))
            (root/"dataset.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Changed frozen input"):
                load_frozen(root, "v6")


if __name__ == "__main__":
    unittest.main()
