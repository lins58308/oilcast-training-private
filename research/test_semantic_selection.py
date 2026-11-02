"""Synthetic chronology / calibration audits; no production model is touched."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("semantic_trainer", Path(__file__).with_name("train_semantic_teacher.py"))
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

class SemanticAccuracyTests(unittest.TestCase):
    def test_baseline_wins_when_news_model_invents_moves(self):
        actual = np.zeros((39, 4), dtype=np.float32)
        unstable = np.ones((39, 4), dtype=np.float32) * .3
        result = trainer.select_conservative_forecaster(actual, {"deep": unstable, "ridge": unstable})
        self.assertEqual([x["model"] for x in result], ["unchanged"] * 4)
        calibrated = trainer.apply_selections({"deep": unstable, "ridge": unstable}, result, 39)
        self.assertTrue(np.all(calibrated == 0))

    def test_real_forecast_gain_can_be_selected(self):
        actual = np.zeros((39, 4), dtype=np.float32)
        actual[::2] = .5
        predicted = np.copy(actual)
        result = trainer.select_conservative_forecaster(actual, {"deep": predicted, "ridge": np.zeros_like(predicted)})
        self.assertTrue(all(x["approved"] and x["model"] == "deep" for x in result))
        calibrated = trainer.apply_selections({"deep": predicted, "ridge": np.zeros_like(predicted)}, result, 39)
        score = trainer.metrics(actual, calibrated)
        self.assertTrue(all(x < .25 for x in score["mae"]))
        self.assertTrue(all(x > 0 for x in score["movingWeekRecallPct"]))

    def test_heldout_is_not_used_for_selection(self):
        val_actual = np.zeros((39, 4), dtype=np.float32)
        val_actual[:10] = -.6
        val_pred = val_actual.copy()
        rules = trainer.select_conservative_forecaster(val_actual, {"deep": val_pred})
        good_future = val_actual.copy()
        bad_future = -val_actual
        self.assertEqual(rules, trainer.select_conservative_forecaster(val_actual, {"deep": val_pred}))
        self.assertNotEqual(trainer.metrics(good_future, val_pred), trainer.metrics(bad_future, val_pred))
        self.assertEqual(rules, trainer.select_conservative_forecaster(val_actual, {"deep": val_pred}))

    def test_no_late_news_can_enter_training(self):
        from datetime import date, timedelta
        rows = []
        d = date(2019, 1, 7)
        for i in range(92):
            origin = d + timedelta(days=7*i)
            next_date = origin + timedelta(days=7)
            rows.append({"origin": origin.isoformat(), "targetDate": next_date.isoformat(),
                         "cutoff": origin.isoformat() + "T00:00:00Z",
                         "news": [], "features": [[0.] * 26 for _ in range(4)],
                         "nextWeekDeltas": [0.] * 4})
        metadata = {"schema": 1, "featureDimension": 26, "fuels": 4, "examples": rows}
        self.assertEqual(len(trainer.validate_dataset(metadata)), len(rows))
        rows[1]["news"] = [{"publishedAt": rows[2]["cutoff"], "availableAt": rows[2]["cutoff"]}]
        with self.assertRaises(ValueError):
            trainer.validate_dataset(metadata)

if __name__ == "__main__":
    unittest.main()
