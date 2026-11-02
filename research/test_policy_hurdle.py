import sys
from pathlib import Path
from datetime import date,timedelta
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_policy_hurdle import dataset_features,apply_hurdle,choose_policies

def synthetic(n=300):
    rows=[]
    start=date(2000,1,3)
    for t in range(n):
        features=np.zeros((4,26)).tolist()
        for f in range(4):features[f][9]=20+f
        delta=[0,0,0,0]
        if t%8==0:delta=[.3,.3,.3,-.2]
        rows.append({"features":features,"nextWeekDeltas":delta,"news":[],
                     "origin":str(start+timedelta(days=7*t))})
    return rows

class PolicyHurdleTest(unittest.TestCase):
    def test_future_labels_never_leak_into_origin_features(self):
        rows=synthetic()
        orig,_=dataset_features(rows)
        rows[100]["nextWeekDeltas"]=[5,5,5,5]
        altered,_=dataset_features(rows)
        np.testing.assert_allclose(orig[:101],altered[:101])
        self.assertFalse(np.allclose(orig[101],altered[101]))

    def test_no_news_is_not_invented(self):
        rows=synthetic()
        x,y=dataset_features(rows)
        self.assertEqual(x.shape[0],300)
        self.assertEqual(y.shape,(300,2))
        self.assertTrue(np.isfinite(x).all())

    def test_hurdle_zeros_low_change_probability(self):
        exp=np.array([[1.2,-.5],[.3,.3]],dtype=np.float32)
        prob=np.array([[.1,.9],[.9,.1]],dtype=np.float32)
        p=apply_hurdle(exp,prob,[(1.,.5),(1.,.5)])
        np.testing.assert_array_equal(p[0,:3],[0,0,0])
        self.assertAlmostEqual(float(p[0,3]),-.5)
        np.testing.assert_allclose(p[1,:3],[.3,.3,.3],rtol=1e-6)
        self.assertEqual(p[1,3],0.)

    def test_confirmation_blocks_development_only_signal(self):
        n=300
        expected=np.zeros((n,2),dtype=np.float32)
        probabilities=np.ones((n,2),dtype=np.float32)
        y=np.zeros((n,4),dtype=np.float32)
        # perfect "predictions" in development, but bad in confirmation
        y[170:222,:3]=.5
        y[170:222,3]=-.4
        expected[170:,:]=[.5,-.4]
        chosen,records=choose_policies(y,expected,probabilities,170,222,248)
        self.assertEqual(chosen,[(0.,1.),(0.,1.)])
        self.assertFalse(records[0]["approvedByConfirmation"])

if __name__=="__main__":
    unittest.main()
