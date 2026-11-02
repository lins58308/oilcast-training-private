#!/usr/bin/env python3
"""Read-only 27-year CPC movement-hurdle backtest with genuinely disjoint eras.

Uses only known-at-origin CPC history, contemporaneously available market/news
features, and the prior adjustment streak. First predict up/flat/down, then
estimate the conditional CPC amount. No locked holdout selection or Android
publication. Separate from the optional pretrained semantic teacher.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from cpc_aware_calibration import indicators

TARGET_GROUPS = ((0,1,2),(3,))
SCALES = (0.25,0.5,0.75,1.0)
MOVE_PROB = (0.45,0.55,0.65,0.75,0.85)

def dataset_features(rows):
    if len(rows)<250:
        raise ValueError("Need at least 250 CPC weekly samples")
    n=len(rows)
    numeric=np.asarray([r["features"] for r in rows],dtype=np.float64)
    label=np.asarray([r["nextWeekDeltas"] for r in rows],dtype=np.float64)
    if numeric.shape!=(n,4,26) or label.shape!=(n,4):
        raise ValueError("Incorrect four-fuel, 26-feature weekly dataset")
    if not np.isfinite(numeric).all() or not np.isfinite(label).all():
        raise ValueError("Nonfinite features or labels")
    # CPC 95/98 generally follow the 92 gasoline adjustment. Use the
    # historical median to tolerate rare official rounding anomalies.
    gas=np.median(label[:,:3],axis=1)
    target=np.column_stack((gas,label[:,3]))
    feats=[]
    for i,row in enumerate(rows):
        current=numeric[i]
        # Historic per-fuel price lags, crude/FX/news at today's cutoff,
        # all computed by the Java publisher's strict snapshot(t).
        vector=list(current[1,:10])+list(current[3,:10])+list(current[1,10:26])
        for group in range(2):
            # target[i] is the FUTURE Monday; use only labels < i.
            history=target[max(0,i-52):i,group]
            streak=0
            for val in history[::-1]:
                if abs(val)>=.049:break
                streak+=1
            last8=history[-8:]
            last26=history[-26:]
            vector.extend([min(streak,52)/52.0,
                float(np.mean(np.abs(last8)>=.049)) if len(last8) else 0.0,
                float(np.mean(np.abs(last26)>=.049)) if len(last26) else 0.0,
                float(np.mean(last8)) if len(last8) else 0.0,
                float(np.mean(np.abs(last26))) if len(last26) else 0.0])
        # Interaction is a Brent/FX proxy, NOT licensed daily Dubai/Brent 7D3B.
        brent,fx=current[1,10],current[1,12]
        vector.extend([(brent+fx+brent*fx/100.0) if current[1,17]>=2/3 else 0.0,
            float(current[1,17]>=2/3),float(bool(row["news"]))])
        # Official regime landmarks are dated and known historically; do NOT
        # insert government absorption announcements that occurred in future.
        date=row["origin"]
        vector.extend([float(date>="2007-01-01"),float(date>="2018-05-14"),
            float(date>="2025-12-01"),
            float(current[1,9]>=32.5),float(current[1,9]>=35.0)])
        feats.append(vector)
    return np.asarray(feats,dtype=np.float32),target.astype(np.float32)

def probability_predictions(x, labels, train_end):
    from sklearn.ensemble import HistGradientBoostingClassifier
    out=np.zeros((len(labels),2),dtype=np.float32)
    probabilities=np.zeros((len(labels),2),dtype=np.float32)
    metadata=[]
    for group in range(2):
        truth=labels[:train_end,group]
        class_y=np.where(truth>.049,2,np.where(truth<-.049,0,1))
        if len(np.unique(class_y))<2:
            metadata.append({"group":group,"fallback":"only one historical class"})
            continue
        # Low capacity prevents fitting noisy 27-year policy/regime changes.
        clf=HistGradientBoostingClassifier(
            max_iter=65,max_leaf_nodes=6,max_depth=2,min_samples_leaf=45,
            learning_rate=.04,l2_regularization=20,random_state=20261103,
            early_stopping=False)
        clf.fit(x[:train_end],class_y)
        prob=np.zeros((len(labels),3))
        prob[:,clf.classes_]=clf.predict_proba(x)
        negatives=truth[truth<-.049];positives=truth[truth>.049]
        mu_neg=float(np.clip(negatives.mean(),-2,-.1)) if len(negatives) else -.2
        mu_pos=float(np.clip(positives.mean(),.1,2)) if len(positives) else .2
        out[:,group]=prob[:,0]*mu_neg+prob[:,2]*mu_pos
        probabilities[:,group]=prob[:,0]+prob[:,2]
        metadata.append({"group":group,"positiveWeeks":len(positives),"negativeWeeks":len(negatives),
            "meanUp":mu_pos,"meanDown":mu_neg,"features":x.shape[1]})
    return out,probabilities,metadata

def apply_hurdle(expectation, movement_probability, policy):
    pred=np.zeros((len(expectation),4),dtype=np.float32)
    for g,columns in enumerate(TARGET_GROUPS):
        scale,threshold=policy[g]
        estimate=np.round(expectation[:,g]*scale,1)
        estimate[(movement_probability[:,g]<threshold)|(np.abs(estimate)<.1-1e-6)]=0
        for f in columns:pred[:,f]=estimate
    return pred

def choose_policies(y,expectation,probabilities,train_end,dev_end,confirm_end):
    options=[(0.0,1.0)]+[(scale,p) for scale in SCALES for p in MOVE_PROB]
    selected=[];details=[]
    base=np.zeros_like(y)
    dev_base=indicators(y[train_end:dev_end],base[train_end:dev_end])
    conf_base=indicators(y[dev_end:confirm_end],base[dev_end:confirm_end])
    for g,grades in enumerate(TARGET_GROUPS):
        dev_scores=[]
        for index,option in enumerate(options):
            policy=[(0.,1.),(0.,1.)];policy[g]=option
            p=apply_hurdle(expectation,probabilities,policy)
            metric=indicators(y[train_end:dev_end],p[train_end:dev_end])
            mae=np.mean([metric["mae"][f] for f in grades])
            base_mae=np.mean([dev_base["mae"][f] for f in grades])
            accuracy=np.mean([metric["directionAccuracyPct"][f] for f in grades])
            base_accuracy=np.mean([dev_base["directionAccuracyPct"][f] for f in grades])
            false=max(metric["falseMovePct"][f] for f in grades)
            if index==0 or (mae+.01 <base_mae and accuracy>=base_accuracy-5 and false<=25):
                dev_scores.append((mae,index))
        best=min(dev_scores)[1]
        option=options[best]
        policies=[(0.,1.),(0.,1.)];policies[g]=option
        p=apply_hurdle(expectation,probabilities,policies)
        metric=indicators(y[dev_end:confirm_end],p[dev_end:confirm_end])
        mae=np.mean([metric["mae"][f] for f in grades])
        bmae=np.mean([conf_base["mae"][f] for f in grades])
        accuracy=np.mean([metric["directionAccuracyPct"][f] for f in grades])
        bacc=np.mean([conf_base["directionAccuracyPct"][f] for f in grades])
        false=max(metric["falseMovePct"][f] for f in grades)
        approved=best>0 and mae+.01 < bmae and accuracy>=bacc-5 and false<=25
        selected.append(option if approved else (0.,1.))
        details.append({"group":g,"developmentChoice":option,"approvedByConfirmation":bool(approved),
             "confirmationCandidateMAE":float(mae),"confirmationBaselineMAE":float(bmae)})
    return selected,details

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--variant",choices=("v6","v7"),required=True)
    args=parser.parse_args()
    rows=json.loads(Path(args.dataset).read_text())["examples"]
    x,ys=dataset_features(rows)
    labels=np.column_stack((ys[:,0],ys[:,0],ys[:,0],ys[:,1]))
    n=len(rows)
    # 52 develop + 26 independent confirmation + 52 final locked audit.
    hold=n-52;confirm=hold-26;train=confirm-52
    if train<250:raise ValueError("Insufficient training era before locked audit")
    estimated,prob,metadata=probability_predictions(x,ys,train)
    policies,selection=choose_policies(labels,estimated,prob,train,confirm,hold)
    predicted=apply_hurdle(estimated,prob,policies)
    base=np.zeros_like(labels)
    report={"schema":1,"variant":args.variant,"model":"regularized CPC up/flat/down hurdle",
        "status":"RESEARCH_ONLY","weeks":n,"featureCount":x.shape[1],
        "trainingWeeks":train,"developmentWeeks":confirm-train,
        "independentConfirmationWeeks":hold-confirm,"lockedHoldoutWeeks":n-hold,
        "development":{"baseline":indicators(labels[train:confirm],base[train:confirm]),
           "candidate":indicators(labels[train:confirm],predicted[train:confirm])},
        "confirmation":{"baseline":indicators(labels[confirm:hold],base[confirm:hold]),
           "candidate":indicators(labels[confirm:hold],predicted[confirm:hold])},
        "lockedHoldout52Weeks":{"baseline":indicators(labels[hold:],base[hold:]),
           "candidate":indicators(labels[hold:],predicted[hold:])},
        "policy":selection,"groupModels":metadata,
        "automaticProductionPromotion":False,
        "provenance":"All price-streak features use targets strictly preceding each forecast origin; market/news snapshots pre-date target."
        }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"stage":"policy_hurdle_complete","variant":args.variant,
        "selectedPolicies":selection,"lockedHoldout52Weeks":report["lockedHoldout52Weeks"]}),flush=True)

if __name__=="__main__":main()
