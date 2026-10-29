#!/usr/bin/env python3
"""Isolated, deterministic, TFT-inspired + CNN/LSTM/GRU shadow model.
No private key and no production model publication are permitted here.
"""
import argparse, datetime as dt, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
WINDOW,HOLDOUT,VALIDATION=12,13,26
F=14

def set_seed(seed=58308):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def source_features(root):
    seed=json.loads((root/'data/seed.json').read_text())
    outside=json.loads((root/'data/multisource.json').read_text())
    rows=sorted((dt.date.fromisoformat(r['date']),r['prices'])
                for r in seed['rows'] if all(v is not None for v in r['prices']))
    if len(rows)<120:raise ValueError('not enough complete historical rows')
    market=sorted((dt.datetime.fromisoformat(m['availableAt'].replace('Z','+00:00')),
                   dt.date.fromisoformat(m['date']),m['values']) for m in outside['market'])
    news=sorted((dt.datetime.fromisoformat(n['availableAt'].replace('Z','+00:00')),
                 dt.datetime.fromisoformat(n['publishedAt'].replace('Z','+00:00')),
                 n.get('topics',[])) for n in outside['news'])
    X=[];prices=[];dates=[];last=np.zeros(3,dtype=np.float64)
    for i,(day,values) in enumerate(rows):
        # All external inputs must have existed by the corresponding Monday.
        cutoff=dt.datetime.combine(day,dt.time.min,
            tzinfo=dt.timezone(dt.timedelta(hours=8))).astimezone(dt.timezone.utc)
        candidates=[(date,obs) for available,date,obs in market
                    if available<=cutoff and date<=day and (day-date).days<=24]
        newest=[None]*3
        for date,obs in candidates:
            for j in range(3):
                if obs[j] is not None and (newest[j] is None or date>=newest[j][0]):
                    newest[j]=(date,float(obs[j]))
        observed=0
        for j in range(3):
            if newest[j] is not None:last[j]=newest[j][1];observed+=1
        last_news=[topics for available,published,topics in news
                   if available<=cutoff and published<=cutoff and
                   0<=(cutoff-published).total_seconds()<7*86400]
        risk=sum(1 for topics in last_news if 4 in topics)
        current=np.asarray(values,dtype=np.float64)
        delta=current-np.asarray(rows[i-1][1],dtype=np.float64) if i else np.zeros(4)
        vector=np.r_[current,delta,last,observed/3,
                     math.log1p(len(last_news)),risk/max(1,len(last_news))]
        X.append(vector);prices.append(current);dates.append(day.isoformat())
    return np.asarray(X,np.float32),np.asarray(prices,np.float32),dates

class Hybrid(nn.Module):
    """TFT-inspired variable selection/gates and attention, not the full TFT."""
    def __init__(self,features=F,hidden=8):
        super().__init__()
        self.selector=nn.Linear(features,features)
        self.conv=nn.Conv1d(features,hidden,3,padding=1)
        self.gru=nn.GRU(hidden,hidden,batch_first=True)
        self.lstm=nn.LSTM(hidden,hidden,batch_first=True)
        self.attention=nn.MultiheadAttention(hidden*2,2,batch_first=True)
        self.gate=nn.Linear(hidden*2,hidden*2)
        self.norm=nn.LayerNorm(hidden*2)
        self.out=nn.Sequential(nn.Linear(hidden*2,8),nn.Tanh(),nn.Linear(8,4))
    def forward(self,x):
        selected=x*torch.softmax(self.selector(x),-1)*x.shape[-1]
        local=torch.tanh(self.conv(selected.transpose(1,2)).transpose(1,2))
        g,_=self.gru(local);l,_=self.lstm(local)
        merged=torch.cat((g,l),dim=-1)
        attended,_=self.attention(merged,merged,merged,need_weights=False)
        return self.out(self.norm(merged+torch.sigmoid(self.gate(merged))*attended)[:,-1,:])

def training_samples(X,prices,origin):
    # Last training label is origin-1, never the week under prediction.
    ends=np.arange(WINDOW-1,origin-1,dtype=int)
    return ends,prices[ends+1]-prices[ends]

def model_fit(X,prices,origin,epochs=30,seed=58308):
    set_seed(seed)
    ends,delta=training_samples(X,prices,origin)
    if len(ends)<60:raise ValueError('insufficient historical data')
    history=X[:origin];mean=history.mean(axis=0)
    std=np.maximum(history.std(axis=0),.01)
    z=np.clip((X-mean)/std,-6,6)
    tensor=torch.tensor(np.stack([z[e-WINDOW+1:e+1] for e in ends]),dtype=torch.float32)
    y=torch.tensor(np.clip(delta,-5,5),dtype=torch.float32)
    net=Hybrid();optimizer=torch.optim.AdamW(net.parameters(),lr=.007,weight_decay=.02)
    hold=max(12,int(len(ends)*.15));n=len(ends)-hold
    best_loss=float('inf');best_state=None;bad=0
    for epoch in range(epochs):
        net.train()
        for batch in torch.randperm(n).split(64):
            loss=nn.functional.smooth_l1_loss(net(tensor[batch]),y[batch])
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.0);optimizer.step()
        net.eval()
        with torch.no_grad():value=nn.functional.l1_loss(net(tensor[n:]),y[n:]).item()
        if value<best_loss-1e-5:
            best_loss=value;best_state={k:v.detach().clone() for k,v in net.state_dict().items()};bad=0
        else:
            bad+=1
            if bad>=7:break
    net.load_state_dict(best_state);net.eval()
    return net,mean,std,epoch+1

def predict(net,mean,std,X,prices,origin):
    z=np.clip((X[origin-WINDOW:origin]-mean)/std,-6,6).astype(np.float32)
    with torch.no_grad():delta=net(torch.from_numpy(z[None])).numpy()[0]
    return np.round(np.clip(prices[origin-1]+np.clip(delta,-5,5),5,100)*10)/10

def evaluate(predicted,actual,baseline):
    error=np.abs(predicted-actual)
    pd=np.sign(np.round((predicted-baseline)*10)/10)
    ad=np.sign(np.round((actual-baseline)*10)/10)
    moves=ad!=0
    return {'mae':float(error.mean()),'directionPct':float(100*(pd==ad).mean()),
        'changeRecallPct':float(100*(pd[moves]==ad[moves]).mean()) if moves.any() else None,
        'falseMovePct':float(100*(pd[~moves]!=0).mean()) if (~moves).any() else None,
        'movingFuelWeeks':int(moves.sum())}

def run(root,output,epochs=30):
    X,prices,dates=source_features(root);n=len(X)
    origins=list(range(n-HOLDOUT-VALIDATION,n-HOLDOUT,4))
    baseline=prices[-HOLDOUT-1:-1];actual=prices[-HOLDOUT:]
    options=[]
    for cfg in ({'epochs':epochs,'seed':58308},
                {'epochs':epochs,'seed':20261029}):
        forecast=[]
        for t in origins:
            net,mean,std,_=model_fit(X,prices,t,**cfg)
            forecast.append(predict(net,mean,std,X,prices,t))
        forecast=np.asarray(forecast)
        base=prices[np.asarray(origins)-1];observed=prices[origins]
        for alpha in (.25,.5,.75,1.):
            blended=np.round((base+alpha*(forecast-base))*10)/10
            options.append({'config':cfg,'alpha':alpha,
                            'metrics':evaluate(blended,observed,base)})
    best=min(options,key=lambda x:(x['metrics']['mae'],-x['metrics']['directionPct']))
    origins=list(range(n-HOLDOUT,n))
    raw=[]
    for t in origins:
        net,mean,std,_=model_fit(X,prices,t,**best['config'])
        raw.append(predict(net,mean,std,X,prices,t))
    raw=np.asarray(raw)
    forecast=np.round((baseline+best['alpha']*(raw-baseline))*10)/10
    hybrid=evaluate(forecast,actual,baseline);flat=evaluate(baseline,actual,baseline)
    passed=hybrid['mae']+.002<flat['mae'] and hybrid['directionPct']>=flat['directionPct']
    result={'status':'shadow-only-not-promoted','architecture':
        'tft-inspired-variable-selection-cnn-lstm-gru-gated-attention',
        'isFullTft':False,'source':'CPC seed and as-of FRED/news',
        'dateRange':[dates[0],dates[-1]],'rows':n,'lookbackWeeks':WINDOW,
        'validationOrigins':len(range(n-HOLDOUT-VALIDATION,n-HOLDOUT,4)),
        'holdoutOrigins':HOLDOUT,'validationCandidates':options,
        'selectedConfig':best['config'],'selectedShrinkage':best['alpha'],
        'holdoutHybrid':hybrid,'holdoutUnchanged':flat,'holdoutPass':passed,
        'holdout':[{'date':dates[t],'base':baseline[i].tolist(),
            'predicted':forecast[i].tolist(),'actual':actual[i].tolist()}
            for i,t in enumerate(origins)],
        'note':'Not signed, not published. Historical availability is estimated; not live accuracy.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:result[k] for k in
        ('rows','holdoutHybrid','holdoutUnchanged','holdoutPass')},ensure_ascii=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=ROOT/'output/hybrid-shadow-report.json')
    p.add_argument('--epochs',type=int,default=30)
    a=p.parse_args();run(a.root,a.output,a.epochs)
