import unittest,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import hybrid_shadow as h

class HybridShadowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X,cls.prices,cls.dates=h.source_features(ROOT)
    def test_cutoff(self):
        origin=len(self.X)-13
        ends,y=h.training_samples(self.X,self.prices,origin)
        self.assertEqual(ends.max()+1,origin-1)
        self.assertEqual(len(ends),len(y))
    def test_forward(self):
        h.set_seed(42)
        output=h.Hybrid()(torch.randn(3,h.WINDOW,h.F))
        self.assertEqual(tuple(output.shape),(3,4))
        self.assertTrue(torch.isfinite(output).all().item())
    def test_zero_not_missing(self):
        base=np.array([[30.,32.,34.,27.]],np.float32)
        predicted=base+.1
        actual=base-.1
        score=h.evaluate(predicted,actual,base)
        self.assertEqual(score['directionPct'],0.0)
        self.assertEqual(score['movingFuelWeeks'],4)
    def test_future_ignored(self):
        origin=len(self.X)-13
        net,mean,std,_=h.model_fit(self.X,self.prices,origin,epochs=3)
        left=h.predict(net,mean,std,self.X,self.prices,origin)
        changed_X=self.X.copy();changed_y=self.prices.copy()
        changed_X[origin:]+=99;changed_y[origin:]+=99
        right=h.predict(net,mean,std,changed_X,changed_y,origin)
        np.testing.assert_equal(left,right)
    def test_reproducible(self):
        origin=len(self.X)-13
        a,_,_,_=h.model_fit(self.X,self.prices,origin,epochs=3)
        b,_,_,_=h.model_fit(self.X,self.prices,origin,epochs=3)
        for key in a.state_dict():
            self.assertTrue(torch.equal(a.state_dict()[key],b.state_dict()[key]),key)
    def test_coverage(self):
        self.assertGreater(len(self.X),120)
        self.assertTrue(np.isfinite(self.X).all())
        self.assertTrue(np.all((self.X[:,11]>=0)&(self.X[:,11]<=1)))

if __name__=='__main__':unittest.main()
