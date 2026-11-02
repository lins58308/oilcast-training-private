import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpc_aware_calibration import (
    tie_gasoline, fit_calibration, apply_calibration, policy_prediction,
    indicators, POLICIES
)

class CpcAwareCalibrationTest(unittest.TestCase):
    def fixture(self, weeks, offset=0):
        target = np.zeros((weeks, 4), dtype=np.float32)
        for i in range(weeks):
            t = i + offset
            if t % 2 == 0:
                target[i, :3] = .3
            if t % 3 == 0:
                target[i, 3] = -.2
        pred = target.copy()*1.4
        pred[:, 0] += .08
        pred[:, 1] -= .04
        pred[:, 2] -= .04
        return target, pred

    def test_gasoline_grades_share_one_adjustment_without_mutating_input(self):
        raw = np.array([[.4, -.1, .25, -.3]], dtype=np.float32)
        old = raw.copy()
        tied = tie_gasoline(raw)
        np.testing.assert_allclose(raw, old)
        np.testing.assert_allclose(tied[0, :3], [tied[0, 0]]*3)
        self.assertEqual(tied[0, 3], -.3)

    def test_positive_signal_passes_disjoint_confirm(self):
        train, p_train = self.fixture(26)
        confirm, p_confirm = self.fixture(13, 26)
        chosen, reports = fit_calibration(train, p_train, confirm, p_confirm)
        self.assertGreater(chosen[0], 0)
        self.assertGreater(chosen[1], 0)
        selected = apply_calibration(p_confirm, chosen)
        self.assertLess(np.abs(selected-confirm).mean(), np.abs(confirm).mean())
        np.testing.assert_allclose(selected[:, 0], selected[:, 1])

    def test_confirmation_blocks_overfit_development(self):
        development, p_dev = self.fixture(26)
        confirmation = np.zeros((13, 4), dtype=np.float32)
        p_confirm = self.fixture(13)[1]
        chosen, reports = fit_calibration(development, p_dev, confirmation, p_confirm)
        self.assertEqual(chosen, [0, 0])
        self.assertFalse(reports[0]["confirmed"])
        self.assertFalse(reports[1]["confirmed"])

    def test_predeclared_thresholds_reject_tiny_fuel_movements(self):
        raw = np.array([[.049, -.049, .04, -.06]], dtype=np.float32)
        selected = policy_prediction(raw, 1, .1)
        np.testing.assert_array_equal(selected, np.zeros_like(raw))

    def test_invalid_arrays_are_rejected(self):
        with self.assertRaises(ValueError):
            tie_gasoline(np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            fit_calibration(np.zeros((9,4)), np.zeros((9,4)),
                            np.zeros((13,4)), np.zeros((13,4)))

    def test_four_grade_metrics_are_auditable(self):
        observed, pred = self.fixture(26)
        m = indicators(observed, tie_gasoline(pred))
        self.assertEqual(len(m["mae"]), 4)
        self.assertEqual(len(m["movementRecallPct"]), 4)
        self.assertTrue(np.isfinite(m["mae"]).all())

if __name__ == "__main__":
    unittest.main()
