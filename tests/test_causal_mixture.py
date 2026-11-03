"""CPC gasoline-shared, diesel-independent causal MAE regression tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import unittest
import numpy as np
from causal_mixture_benchmark import (
    CANDIDATE_SETS,train_pair,make_forecast,quantized_forecast,
    validate_gasoline_labels,rolling_folds,scoring
)

class MixtureTest(unittest.TestCase):
    def test_single_gasoline_prediction_and_separate_diesel(self):
        p={"gas":np.array([[.05,.10,.85],[.80,.15,.05],[.1,.8,.1]]),
           "diesel":np.array([[.1,.8,.1],[.05,.05,.9],[.9,.05,.05]])}
        size={"gas":{0:-.5,2:.7},"diesel":{0:-.3,2:.6}}
        output=make_forecast(p,size,.6)
        self.assertEqual(output.shape,(3,4))
        np.testing.assert_allclose(output[:,0],output[:,1])
        np.testing.assert_allclose(output[:,0],output[:,2])
        np.testing.assert_allclose(output[:,0],np.array([.7,-.5,0]))
        np.testing.assert_allclose(output[:,3],np.array([0,.6,-.3]))

    def test_causal_historical_training_and_future_label_mutation(self):
        x=np.zeros((70,4,58),dtype=float)
        y=np.zeros((70,4),dtype=float)
        p,s=train_pair(x,y,60,60,64,CANDIDATE_SETS["market"],104)
        before=make_forecast(p,s,.6)
        y[60:]=17.3
        p2,s2=train_pair(x,y,60,60,64,CANDIDATE_SETS["market"],104)
        after=make_forecast(p2,s2,.6)
        np.testing.assert_allclose(before,after)
        with self.assertRaises(ValueError):
            train_pair(x,y,61,60,64,CANDIDATE_SETS["market"],104)

    def test_reject_incompatible_fuel_deltas_and_holdout_separation(self):
        y=np.zeros((2,4))
        y[1,1]=.3
        self.assertEqual(validate_gasoline_labels(y),1)
        with self.assertRaises(ValueError):
            validate_gasoline_labels(np.zeros((3,3)))
        intervals,audit=rolling_folds(1412)
        self.assertEqual(audit,1399)
        self.assertEqual(len(intervals),5)
        for start,end in intervals:
            self.assertLessEqual(end,audit-52)
            self.assertEqual(end-start,13)

    def test_quantization_and_false_move_reporting(self):
        probabilities=np.array([[.15,.4,.45],[.04,.07,.89],[.7,.2,.1]])
        amounts={0:-.6,2:.7}
        np.testing.assert_allclose(
            quantized_forecast(probabilities,amounts,.6),
            np.array([0.,.7,-.6]))
        score=scoring(np.array([0.,.7,-.6]),np.array([0.,.7,-.6]))
        self.assertEqual(score["mae"],0.)
        self.assertEqual(score["changedWeekRecall"],1.)

if __name__=="__main__":
    unittest.main()
