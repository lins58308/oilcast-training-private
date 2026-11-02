#!/usr/bin/env python3
"""OilCast time-point accuracy laboratory (research only; no auto promotion).

All folds obey CPC weekly origins and originally available market/news timestamps.
The engine compares no-change, pooled CatBoost, news+market ablations and a
zero-inflated change/no-change hurdle. Latest 13 weeks are quarantined.
"""
import argparse
import json
import math
import pathlib
from datetime import date
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier

SEED = 20261103
FUEL_NAMES = ("92", "95", "98", "diesel")


def make_week_data(data):
    if data.get("schema") != 1 or data.get("featureDimension") != 26:
        raise ValueError("Unexpected research dataset schema")
    weeks = data["examples"]
    if len(weeks) < 300:
        raise ValueError("Insufficient dated CPC history")
    num = np.asarray([e["features"] for e in weeks], dtype=np.float64)
    base = np.asarray([e["basePrices"] for e in weeks], dtype=np.float64)
    y = np.asarray([e["nextWeekDeltas"] for e in weeks], dtype=np.float64)
    if num.shape != (len(weeks), 4, 26) or y.shape != (len(weeks), 4):
        raise ValueError("Unexpected per-week shape")
    if not np.isfinite(num).all() or not np.isfinite(y).all():
        raise ValueError("Invalid historical numeric feature")
    for j, row in enumerate(weeks):
        if j and row["origin"] <= weeks[j - 1]["origin"]:
            raise ValueError("Weekly inputs not strictly chronological")
        if row["targetDate"] <= row["origin"]:
            raise ValueError("Target is not next week")
        for story in row["news"]:
            if story["publishedAt"] > row["cutoff"] or story["availableAt"] > row["cutoff"]:
                raise ValueError("Leaked future news")
    # Historic CPC level/recently-unchanged regime are known BEFORE next Monday.
    n = len(weeks)
    flat_streak = np.zeros((n, 4), dtype=np.float64)
    for k in range(n):
        if k:
            flat_streak[k] = np.where(np.abs(y[k-1]) < .049,
                                     np.minimum(52, flat_streak[k-1]+1), 0)
    oil_trends = num[:, :, 10:18]
    news_trends = num[:, :, 18:26]
    features = []
    for k in range(n):
        line = []
        for f in range(4):
            # Preserves all dated market and news inputs, and four-fuel regimes.
            common = np.concatenate((num[k, f],
                [base[k, f], base[k, 1], base[k, 3], f,
                 flat_streak[k, f], flat_streak[k, 1],
                 int(date.fromisoformat(weeks[k]["origin"]).year >= 2007),
                 int(date.fromisoformat(weeks[k]["origin"]).year >= 2020)]))
            line.append(common)
        features.append(line)
    x = np.asarray(features, dtype=np.float32)
    return x, y, weeks


def scoring(actual, delta):
    actual = np.asarray(actual)
    delta = np.round(np.asarray(delta)*10)/10
    actual_dir = np.where(actual > .049, 1, np.where(actual < -.049, -1, 0))
    predicted_dir = np.where(delta > .049, 1, np.where(delta < -.049, -1, 0))
    changed = actual_dir != 0
    unchanged = ~changed
    return {
        "mae": float(np.abs(actual-delta).mean()),
        "rmse": float(np.sqrt(np.mean((actual-delta)**2))),
        "directionAccuracy": float((actual_dir == predicted_dir).mean()),
        "changedWeekRecall": float(((actual_dir == predicted_dir)&changed).sum() / max(1, changed.sum())),
        "falseMoveRate": float(((predicted_dir != 0)&unchanged).sum() / max(1, unchanged.sum())),
        "changedCount": int(changed.sum()),
        "predictionMovements": int((predicted_dir != 0).sum()),
        "n": int(actual.size),
    }


def train_predict(x, y, train_end, predict_start, predict_end, family, depth=4, iterations=350):
    # Weeks must be grouped; never shuffle partial fuel rows into future.
    assert 0 < train_end <= predict_start < predict_end <= len(x)
    source = x[:train_end].reshape(-1, x.shape[-1])
    labels = y[:train_end].reshape(-1)
    query = x[predict_start:predict_end].reshape(-1, x.shape[-1])
    # Feature ablations test whether news or quotes actually add signal.
    if family == "price":
        selected = list(range(10)) + list(range(26, 34))
    elif family == "market":
        selected = list(range(18)) + list(range(26, 34))
    elif family in ("all", "hurdle"):
        selected = list(range(34))
    else:
        raise ValueError("Unknown family")
    source, query = source[:,selected],query[:,selected]
    if family == "hurdle":
        active = (np.abs(labels) >= .049).astype(int)
        classifier = CatBoostClassifier(
            iterations=iterations,depth=depth,learning_rate=.035,
            loss_function="Logloss",verbose=False,random_seed=SEED,
            l2_leaf_reg=8,thread_count=2)
        if len(np.unique(active)) < 2:
            return np.zeros((predict_end-predict_start,4))
        classifier.fit(source, active)
        prob = classifier.predict_proba(query)[:,1]
        if active.sum()<50: return np.zeros((predict_end-predict_start,4))
        conditional = CatBoostRegressor(
            iterations=iterations,depth=depth,learning_rate=.035,
            loss_function="RMSE",verbose=False,random_seed=SEED,
            l2_leaf_reg=10,thread_count=2)
        conditional.fit(source[active==1],labels[active==1])
        # Conservative probability-weighted expected adjustment.
        output = np.where(prob > .5, prob * conditional.predict(query), 0)
    else:
        model=CatBoostRegressor(
            iterations=iterations,depth=depth,learning_rate=.035,
            loss_function="MAE",verbose=False,random_seed=SEED,
            l2_leaf_reg=10,thread_count=2)
        model.fit(source, labels)
        output = model.predict(query)
    return np.clip(output.reshape(-1,4),-3,3)


def predict_family(x,y,start,end,family):
    if "-gated" not in family:
        return train_predict(x,y,start,start,end,family)
    base, threshold = family.split("-gated")
    pred=train_predict(x,y,start,start,end,base)
    # The streak is a bounded feature of PAST realized price changes;
    # it is constructed before labels at the current origin are known.
    # Index 30 = flat-streak for same fuel in feature matrix.
    consecutive=x[start:end,:,30]
    return np.where(consecutive>=int(threshold), 0, pred)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--output",default="output/accuracy-lab")
    p.add_argument("--variant",choices=("v6","v7"),required=True)
    args=p.parse_args()
    data=json.loads(pathlib.Path(args.dataset).read_text())
    x,y,rows=make_week_data(data)
    n=len(x)
    audit=n-13
    # 3 pre-audit periods, spaced across different historical regimes.
    offsets=(156,104,52)
    folds=[]
    for gap in offsets:
        end=audit-gap
        start=end-13
        if start<156: continue
        folds.append((start,end))
    # Domain-constrained no-change regimes: only use already known
    # streaks, never any future CPC price nor the latest 13-week audit.
    # If a long run of unchanged prices has occurred, suppress small
    # ML changes rather than assume crude moves translate to retail.
    families=("price","market","all","hurdle",
              "price-gated4","price-gated8","market-gated4","market-gated8")
    history={}
    for family in families:
        vals=[]
        for start,end in folds:
            pred=predict_family(x,y,start,end,family)
            vals.append({"from":rows[start]["origin"],"to":rows[end-1]["targetDate"],
                         "scores":scoring(y[start:end],pred),
                         "byFuel":{FUEL_NAMES[f]:scoring(y[start:end,f],pred[:,f]) for f in range(4)}})
        history[family]=vals
    baseline_val=[{"scores":scoring(y[a:b],np.zeros((b-a,4))),
                   "from":rows[a]["origin"],"to":rows[b-1]["targetDate"]} for a,b in folds]
    # Candidate selection is exclusively on the earlier historical folds.
    base_mae=np.mean([z["scores"]["mae"] for z in baseline_val])
    mean_val={f:float(np.mean([z["scores"]["mae"] for z in history[f]]))
              for f in families}
    # Need a persistent improvement across at least two separate, pre-audit eras.
    improved={f:sum(p["scores"]["mae"]+.01<baseline_val[k]["scores"]["mae"]
                    for k,p in enumerate(history[f])) for f in families}
    ranked=[f for f in families if mean_val[f]+.01 < base_mae
            and improved[f]>=2]
    choice=min(ranked,key=lambda f:mean_val[f]) if ranked else "flat"
    # Holdout never participates in selection.
    holdout={}
    for f in families:
        predicted=predict_family(x,y,audit,n,f)
        holdout[f]=scoring(y[audit:],predicted)
    holdout["flat"]=scoring(y[audit:],np.zeros_like(y[audit:]))
    report={"schema":1,"variant":args.variant,"status":"research-only",
        "noAutomaticPromotion":True,"totalWeeklyExamples":n,
        "timePointGuarantee":"all stories are published and available at or before origin",
        "modelFamily":"pooled CatBoost; optional zero-inflated change/no-change",
        "historicalFolds":history,"historicalBaseline":baseline_val,
        "historicalMeanMAE":mean_val,"historicalBaselineMeanMAE":base_mae,
        "historicalBetterFoldCounts":improved,
        "selectedUsingEarlierFoldsOnly":choice,
        "lockedFinal13Weeks":{"start":rows[audit]["targetDate"],
            "end":rows[-1]["targetDate"],"scores":holdout},
        "warning":"News features are metadata; absence before archive coverage is NOT proof of no wars or no supply events. Historical FRED series are not point-in-time vintages."}
    o=pathlib.Path(args.output);o.mkdir(parents=True,exist_ok=True)
    (o/"accuracy-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"status":"complete","version":args.variant,"weeks":n,
        "baselineVal":base_mae,"candidateVal":mean_val,
        "improvedFolds":improved,"selected":choice,
        "locked13WeekMAE":{k:round(v["mae"],4) for k,v in holdout.items()},
        "output":str(o/"accuracy-report.json")},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
