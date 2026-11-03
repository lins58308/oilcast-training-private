#!/usr/bin/env python3
"""OilCast shared-gasoline / independent-diesel MAE model selection.

Research-only.  CPC 95/98 follow the same weekly gas price adjustment as 92.
A classifier predicts no-change / increase / decrease; a conditional
moving-week median estimates the increment ONLY if the classifier is confident.
No score or hyperparameter from the final 13 weeks enters selection.

The five historical windows are different eras; results on the already
repeatedly-inspected 2026 holdout are *diagnostics*, not a new blind claim.
A future prospective forecasting log is required before production promotion.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from accuracy_benchmark import make_week_data,scoring
from dubai_proxy import parse_dubai_csv
from official_daily_oil import parse_moea_daily

SEED=20261103
FUELS=("92","95","98","diesel")
# All indices below are published before the forecast origin.
BASE=list(range(18))+list(range(26,34))
CANDIDATE_SETS={
    "market":BASE,
    "market-policy":BASE+list(range(46,58)),
    "market-policy-7d3b":BASE+list(range(39,58)),
    "market-policy-news":list(range(26))+list(range(26,34))+list(range(46,58)),
}
HISTORY_WEEKS=(104,260,520)
DIRECTION_THRESHOLDS=(0.45,0.5,0.6)
MIN_GAP=0.005


def class_labels(delta):
    z=np.asarray(delta)
    return np.where(z > .049,2,np.where(z < -.049,0,1)).astype(np.int32)


def validate_gasoline_labels(y):
    """True CPC policy should share gasoline adjustment, record exceptions."""
    if not (y.ndim==2 and y.shape[1]==4 and np.isfinite(y).all()):
        raise ValueError("Historical CPC deltas must be finite 4-fuel weeks")
    differences=np.max(np.abs(y[:,:3]-y[:,0,None]),axis=1)
    return int((differences>.051).sum())


def features_for_head(x,head,columns):
    if head not in ("gas","diesel"):
        raise ValueError("Unknown fuel head")
    fuel=0 if head=="gas" else 3
    chosen=np.asarray(columns,dtype=int)
    if len(chosen)==0 or min(chosen)<0 or max(chosen)>=x.shape[2]:
        raise ValueError("Research feature not available in as-of CPC dataset")
    z=x[:,fuel,chosen]
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite as-of inputs")
    return z


def median_size(labels,direction):
    sign=1 if direction==2 else -1
    relevant=np.abs(labels[np.where(class_labels(labels)==direction)])
    if relevant.size<3:
        # No evidence for a new price-movement size; abstain instead.
        return 0.
    # Use historical median, not an optimistic magnitude chosen on test.
    return float(np.clip(np.round(np.median(relevant)*10)/10,.1,1.5))*sign


def class_probability(train_x,train_y,query,train_end,horizon):
    lookback=max(0,train_end-horizon)
    source=train_x[lookback:train_end]
    categories=class_labels(train_y[lookback:train_end])
    if len(source)<52:
        raise ValueError("Too little historical data")
    distinct=np.unique(categories)
    result=np.zeros((len(query),3),dtype=float)
    if len(distinct)==1:
        result[:,distinct[0]]=1.
    else:
        model=CatBoostClassifier(
            loss_function="MultiClass",iterations=220,depth=3,
            learning_rate=.04,l2_leaf_reg=12,random_seed=SEED,
            thread_count=2,verbose=False,allow_writing_files=False)
        age=np.arange(len(source))[::-1]
        weight=.65+.35*np.exp(-age/52.)
        model.fit(source,categories,sample_weight=weight)
        probabilities=model.predict_proba(query)
        for j,v in enumerate(model.classes_):
            result[:,int(v)]=probabilities[:,j]
    sizes={0:median_size(train_y[lookback:train_end],0),
           2:median_size(train_y[lookback:train_end],2)}
    return result,sizes


def quantized_forecast(prob,size,threshold):
    """Choose a move only when its calibrated one-sided probability is high."""
    if not .4<=threshold<=.8:
        raise ValueError("Confidence outside calibrated candidate range")
    up=prob[:,2]
    down=prob[:,0]
    delta=np.zeros(len(prob),dtype=float)
    delta[(up>down)&(up>=threshold)]=size[2]
    delta[(down>up)&(down>=threshold)]=size[0]
    return np.round(delta*10)/10


def train_pair(x,y,train_end,start,end,feature_columns,horizon):
    if not 52<=train_end<=start<end<=len(y):
        raise ValueError("Data from after a forecast was used in training")
    labels={"gas":np.median(y[:,:3],axis=1),"diesel":y[:,3]}
    p={}
    sizes={}
    for head in labels:
        z=features_for_head(x,head,feature_columns)
        p[head],sizes[head]=class_probability(
            z,labels[head],z[start:end],train_end,horizon)
    return p,sizes


def make_forecast(prob,sizes,threshold):
    n=len(prob["gas"])
    gas=quantized_forecast(prob["gas"],sizes["gas"],threshold)
    diesel=quantized_forecast(prob["diesel"],sizes["diesel"],threshold)
    return np.stack((gas,gas,gas,diesel),axis=1).reshape((n,4))


def rolling_folds(n):
    audit=n-13
    intervals=[]
    for offset in (260,208,156,104,52):
        stop=audit-offset
        start=stop-13
        if start>=520:
            intervals.append((start,stop))
    if len(intervals)<4:
        raise ValueError("Too few historical validation folds")
    return intervals,audit


def evaluate_candidates(x,y,rows):
    n=len(y)
    exceptions=validate_gasoline_labels(y)
    folds,audit=rolling_folds(n)
    # The holdout is reserved even from classifier calibration and family
    # selection. Each earlier candidate sees only weeks before its own fold.
    candidates={}
    fold_dates=[{"start":rows[a]["targetDate"],"end":rows[b-1]["targetDate"]} for a,b in folds]
    for feature_name,columns in CANDIDATE_SETS.items():
        for horizon in HISTORY_WEEKS:
            fold_results=[]
            for start,end in folds:
                p,s=train_pair(x,y,start,start,end,columns,horizon)
                fold_results.append({t:make_forecast(p,s,t) for t in DIRECTION_THRESHOLDS})
            for threshold in DIRECTION_THRESHOLDS:
                key=f"{feature_name}/lookback{horizon}/confidence{threshold}"
                scores=[]
                for i,(start,end) in enumerate(folds):
                    pred=fold_results[i][threshold]
                    scores.append({
                        "period":fold_dates[i],
                        "fuels":{FUELS[f]:scoring(y[start:end,f],pred[:,f])
                                 for f in range(4)},
                    })
                candidates[key]=scores
    bases=[{FUELS[f]:scoring(y[a:b,f],np.zeros(b-a))
            for f in range(4)} for a,b in folds]
    choices={}
    for f in range(4):
        fuel=FUELS[f]
        baseline=float(np.mean([v[fuel]["mae"] for v in bases]))
        eligible=[]
        for key,score_list in candidates.items():
            maelist=[z["fuels"][fuel]["mae"] for z in score_list]
            mean=float(np.mean(maelist))
            wins=sum(maelist[k]+MIN_GAP<bases[k][fuel]["mae"]
                     for k in range(len(folds)))
            # Changed-week recall is a diagnostic, not a substitute for MAE.
            if wins>=3 and mean+MIN_GAP<baseline:
                eligible.append((mean,key,wins))
        eligible.sort()
        choices[fuel]={"candidate":eligible[0][1] if eligible else "unchanged",
                       "meanHistoricalMae":round(eligible[0][0],6) if eligible else baseline,
                       "baselineHistoricalMae":baseline,
                       "historicalWinningFolds":eligible[0][2] if eligible else 0,
                       "eligibleChoices":len(eligible)}
    # Only after choices are frozen may holdout diagnostics be calculated.
    audit_scores={}
    ranked_unique=sorted(set(c["candidate"] for c in choices.values())-{"unchanged"})
    predictions={}
    for key in ranked_unique:
        feature,horizon,threshold=key.split("/")
        horizon=int(horizon.replace("lookback",""))
        threshold=float(threshold.replace("confidence",""))
        p,s=train_pair(x,y,audit,audit,n,CANDIDATE_SETS[feature],horizon)
        predictions[key]=make_forecast(p,s,threshold)
    for f in range(4):
        name=FUELS[f]
        base=scoring(y[audit:,f],np.zeros(n-audit))
        which=choices[name]["candidate"]
        trained=scoring(y[audit:,f],predictions[which][:,f]) if which!="unchanged" else base
        audit_scores[name]={
          "baselineMae":base["mae"],"chosenMae":trained["mae"],
          "strictHoldoutImprovement":trained["mae"]+1e-8<base["mae"],
          "directionAccuracy":trained["directionAccuracy"],
          "changedWeekRecall":trained["changedWeekRecall"],
          "falseMoveRate":trained["falseMoveRate"],
          "chosenMethod":which,"changedWeeks":base["changedCount"]}
    return {"schema":1,"researchOnly":True,"autoPromote":False,
            "gasolineAdjustmentExceptions":exceptions,
            "trainingWindows":list(HISTORY_WEEKS),
            "confidenceThresholds":list(DIRECTION_THRESHOLDS),
            "metric":"MAE NTD per liter on observed 0.1 NTD increments",
            "historicalFolds":fold_dates,
            "candidateCount":len(candidates),
            "selectedBeforeAudit":choices,
            "audit2026WasPreviouslyInspected":True,
            "audit13Weeks":audit_scores,
            "allFourBeatBaselineInPreviouslyInspectedAudit":
                all(v["strictHoldoutImprovement"] for v in audit_scores.values()),
            "limitation":"An already examined historical holdout is NOT fresh prospective evidence. No price model is eligible for production until future pre-registered live forecasts establish improvement."
           }


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("--dataset",required=True)
    arg.add_argument("--dubai-csv",required=True)
    arg.add_argument("--official-daily",required=True)
    arg.add_argument("--cpc-policy",required=True)
    arg.add_argument("--variant",choices=("v6","v7"),required=True)
    arg.add_argument("--output",required=True)
    args=arg.parse_args()
    data=json.loads(Path(args.dataset).read_text())
    dubai=parse_dubai_csv(Path(args.dubai_csv).read_text())
    daily=parse_moea_daily(Path(args.official_daily).read_bytes())
    policy=json.loads(Path(args.cpc_policy).read_text())
    x,y,rows,*_=make_week_data(data,dubai,daily,policy,False)
    if x.shape[2]!=58:
        raise ValueError("Expected 26 source + fuel level + Dubai/MOEA + 12 policy features")
    result=evaluate_candidates(x,y,rows)
    result["variant"]=args.variant
    result["weeklyOrigins"]=len(rows)
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({"stage":"causal_mixture_report","variant":args.variant,
       "candidateCount":result["candidateCount"],
       "gasolineExceptions":result["gasolineAdjustmentExceptions"],
       "selectedBeforeAudit":result["selectedBeforeAudit"],
       "audit":result["audit13Weeks"],
       "allFourOnHistoricalAudit":result["allFourBeatBaselineInPreviouslyInspectedAudit"],
       "researchOnly":True},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
