#!/usr/bin/env python3
"""Historical rolling-year Brier calibration: each fold fits only prior weeks.

Audits are descriptive; never use any evaluated fold to tune prediction model
and never auto-promote Android's signed price predictor.
"""
import argparse,json
from pathlib import Path
import numpy as np
from train_policy_hurdle import dataset_features,probability_predictions

def score(prob, actual, start, end, prior):
    truth=(np.abs(actual[start:end])>=.049).astype(float)
    forecast=np.clip(prob[start:end],0,1)
    return {
        "weeks":end-start,
        "actualChangedPct":round(float(np.mean(truth)*100),2),
        "modelBrier":round(float(np.mean((forecast-truth)**2)),6),
        "pastOnlyBaseRateBrier":round(float(np.mean((prior-truth)**2)),6),
        "improved":bool(np.mean((forecast-truth)**2)+1e-8 < np.mean((prior-truth)**2))
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True);p.add_argument("--output",required=True)
    p.add_argument("--variant",choices=("v6","v7"),required=True)
    a=p.parse_args()
    data=json.loads(Path(a.dataset).read_text())
    rows=data["examples"]
    x,targets=dataset_features(rows)
    n=len(rows)
    if n<500:raise ValueError("27-year history not available")
    folds=[]
    # last four disjoint one-year test eras: no future samples in any fit.
    for offset in (3,2,1,0):
        end=n-52*offset
        begin=end-52
        expected,prob,_=probability_predictions(x,targets,begin)
        result={"trainWeeks":begin,"startOrigin":rows[begin]["origin"],
                "endOrigin":rows[end-1]["origin"],"gasoline":{},"diesel":{}}
        for f,group in enumerate(("gasoline","diesel")):
            prior=float(np.mean(np.abs(targets[:begin,f])>=.049))
            result[group]=score(prob[:,f],targets[:,f],begin,end,prior)
        folds.append(result)
    summary={}
    for group in ("gasoline","diesel"):
        summary[group]={
            "improvedFolds":sum(f[group]["improved"] for f in folds),
            "folds":len(folds),
            "averageModelBrier":round(float(np.mean([f[group]["modelBrier"] for f in folds])),6),
            "averagePriorBrier":round(float(np.mean([f[group]["pastOnlyBaseRateBrier"] for f in folds])),6)
        }
    report={"schema":1,"status":"RESEARCH_ONLY","variant":a.variant,
            "selectionUsesHoldout":False,"folds":folds,"summary":summary,
            "note":"Four disjoint 52-week time-forward audits, trained on only earlier data; non-causal news descriptors."}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"stage":"four_rolling_year_audits","variant":a.variant,
                      "summary":summary,"folds":folds}),flush=True)

if __name__=="__main__":main()
