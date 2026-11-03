"""A directional mixture with an explicit point mass at no price change.

All three class frequencies are learned without class balancing. Conditional
magnitudes use only settled labels and past feature neighborhoods. The action
is the mixture median, rather than a probability-weighted mean movement.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def weighted_quantile(values,weights,level):
    order=np.argsort(values,kind='stable');weight=weights[order]
    index=min(len(order)-1,int(np.searchsorted(np.cumsum(weight),level*weight.sum(),side='left')))
    return float(values[order[index]])


def mixture_median(labels,weights,probabilities):
    y=np.asarray(labels,dtype=float);w=np.asarray(weights,dtype=float);p=np.asarray(probabilities,dtype=float)
    if y.ndim!=1 or y.shape!=w.shape or not len(y) or p.shape!=(3,) or not all(np.isfinite(v).all() for v in (y,w,p)):
        raise ValueError('Invalid directional mixture inputs')
    if np.min(w)<0 or w.sum()<=0 or np.min(p)<0 or abs(p.sum()-1)>1e-8:
        raise ValueError('Invalid directional mixture weights')
    down=y<-.049;up=y>.049
    if p[0]>.5:
        if not down.any():raise ValueError('Negative probability has no past conditional observations')
        return weighted_quantile(y[down],w[down],.5/p[0])
    if p[0]+p[1]>=.5:return 0.
    if not up.any():raise ValueError('Positive probability has no past conditional observations')
    return weighted_quantile(y[up],w[up],(.5-p[0]-p[1])/p[2])


class DirectionalMixture:
    def fit(self,x,y,weights):
        self.labels=np.asarray(y).copy();self.age=np.asarray(weights).copy()
        self.mean=np.average(x,axis=0,weights=weights)
        self.scale=np.sqrt(np.average((x-self.mean)**2,axis=0,weights=weights));self.scale[self.scale<1e-6]=1.
        self.source=np.clip((x-self.mean)/self.scale,-4,4)
        classes=np.where(y<-.049,-1,np.where(y>.049,1,0))
        self.classes=np.unique(classes);self.model=None
        if len(self.classes)>1:
            self.model=HistGradientBoostingClassifier(max_iter=70,max_leaf_nodes=7,min_samples_leaf=16,
                l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=20261103)
            self.model.fit(x,classes,sample_weight=weights)
        return self

    def predict(self,x):
        probability=np.zeros((len(x),3))
        if self.model is None:probability[:,int(self.classes[0])+1]=1.
        else:
            predicted=self.model.predict_proba(x)
            for j,c in enumerate(self.model.classes_):probability[:,int(c)+1]=predicted[:,j]
        results=[]
        for row,p in zip(x,probability):
            query=np.clip((row-self.mean)/self.scale,-4,4)
            distance=np.mean((self.source-query)**2,axis=1)
            k=min(63,len(distance)-1);bandwidth=max(1e-8,float(np.partition(distance,k)[k]))
            weight=self.age*(.05+np.exp(np.clip(-distance/bandwidth,-50,0)))
            results.append(mixture_median(self.labels,weight,p))
        return np.asarray(results)
