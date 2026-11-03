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
from dubai_proxy import parse_dubai_csv, lagged_dubai_features
from official_daily_oil import parse_moea_daily, point_in_time_weekly
from policy_features import feature_for_period

SEED = 20261103
FUEL_NAMES = ("92", "95", "98", "diesel")


def make_week_data(data, dubai_monthly=None, official_daily=None, policy=None):
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
    dubai_added = 0
    dubai_vectors = None
    official_vectors = None
    official_count = 0
    policy_vectors = None
    policy_count = 0
    if dubai_monthly is not None:
        dubai_vectors, dubai_added = lagged_dubai_features(weeks, dubai_monthly)
    if official_daily is not None:
        if dubai_monthly is None:
            raise ValueError('Official daily benchmark requires fixed monthly baseline dimensions')
        official_vectors, official_count = point_in_time_weekly(weeks,official_daily)
    if policy is not None:
        if dubai_monthly is None or official_daily is None:
            raise ValueError("CPC policy experiments require stable 46-feature quote baseline")
        policy_vectors,policy_count=feature_for_period(weeks,policy)
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
            if dubai_vectors is not None:
                # Extends research features ONLY, never the signed Android
                # schema. Each month is embargoed until +75d after close.
                common = np.concatenate((common, dubai_vectors[k]))
            if official_vectors is not None:
                # Verified government daily Dubai/Brent, 70/30 signal.
                # All underlying quotes must precede current CPC origin.
                common = np.concatenate((common,official_vectors[k]))
            if policy_vectors is not None:
                common = np.concatenate((common,policy_vectors[k][f]))
            line.append(common)
        features.append(line)
    x = np.asarray(features, dtype=np.float32)
    return x, y, weeks, dubai_added, official_count, policy_count


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
    elif family == "market-dubai":
        selected = list(range(18)) + list(range(26, 39))
    elif family == "all-dubai":
        selected = list(range(39))
    elif family == "hurdle-dubai":
        selected = list(range(39))
    elif family == "moea-market":
        selected = list(range(18)) + list(range(26, 34)) + list(range(39,46))
    elif family == "moea-all":
        selected = list(range(34)) + list(range(39,46))
    elif family == "moea-hurdle":
        selected = list(range(34)) + list(range(39,46))
    elif family == "policy-market":
        selected = list(range(18)) + list(range(26,34)) + list(range(46,58))
    elif family == "policy-all":
        selected = list(range(34)) + list(range(39,58))
    elif family == "policy-moea":
        selected = list(range(18)) + list(range(26,34)) + list(range(39,58))
    elif family == "policy-hurdle":
        selected = list(range(18)) + list(range(26,34)) + list(range(46,58))
    else:
        raise ValueError("Unknown family")
    source, query = source[:,selected],query[:,selected]
    if family in ("hurdle", "hurdle-dubai","moea-hurdle","policy-hurdle"):
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
    if family.startswith("policy-gate"):
        if x.shape[-1]!=58:raise ValueError("Missing CPC policy features")
        min_streak=int(family[len("policy-gate"):])
        if min_streak not in (1,2,4):raise ValueError("Unsupported freeze gate")
        pred=train_predict(x,y,start,start,end,"market")
        recent=x[start:end]
        source=(recent[:,:,46]>.5)&(recent[:,:,52]>.5)&(recent[:,:,53]>.5)
        duration=recent[:,:,56]>=(min_streak/8-.001)
        prior_absorption=recent[:,:,49]>=2.
        return np.where(source&duration&prior_absorption,0,pred)
    if "-shrink" in family:
        base, shrink = family.split("-shrink")
        if shrink not in ("25", "50", "75"):
            raise ValueError("Unapproved forecast shrinkage")
        return train_predict(x,y,start,start,end,base) * (int(shrink)/100)
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
    p.add_argument("--dubai-csv",default=None,help="Verified FRED monthly Dubai CSV; research only")
    p.add_argument("--official-daily",default=None,help="MOEA government daily Dubai Brent CSV 6029; research only")
    p.add_argument("--cpc-policy",default=None,help="Archived CPC press releases embargoed until their publication day")
    p.add_argument("--output",default="output/accuracy-lab")
    p.add_argument("--variant",choices=("v6","v7"),required=True)
    args=p.parse_args()
    data=json.loads(pathlib.Path(args.dataset).read_text())
    dubai = parse_dubai_csv(pathlib.Path(args.dubai_csv).read_text()) if args.dubai_csv else None
    official=parse_moea_daily(pathlib.Path(args.official_daily).read_bytes()) if args.official_daily else None
    policy=json.loads(pathlib.Path(args.cpc_policy).read_text()) if args.cpc_policy else None
    x,y,rows,dubai_added,official_count,policy_count=make_week_data(data,dubai,official,policy)
    n=len(x)
    audit=n-13
    # 3 pre-audit periods, spaced across different historical regimes.
    offsets=(260,208,156,104,52)
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
              "price-gated4","price-gated8","market-gated4","market-gated8",
              "price-shrink25","price-shrink50","market-shrink25",
              "market-shrink50","all-shrink50")
    if dubai is not None:
        families += ("market-dubai", "all-dubai", "hurdle-dubai")
    if official is not None:
        families += ("moea-market", "moea-all", "moea-hurdle",
                     "moea-market-shrink50")
    if policy is not None:
        families += ("policy-market","policy-all","policy-moea",
                     "policy-hurdle","policy-gate1","policy-gate2","policy-gate4")
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
            and improved[f] >= max(2,math.ceil(0.6*len(folds)))]
    choice=min(ranked,key=lambda f:mean_val[f]) if ranked else "flat"
    # Holdout never participates in selection.
    holdout={}
    for f in families:
        predicted=predict_family(x,y,audit,n,f)
        holdout[f]=scoring(y[audit:],predicted)
    holdout["flat"]=scoring(y[audit:],np.zeros_like(y[audit:]))
    report={"schema":1,"variant":args.variant,"status":"research-only",
        "noAutomaticPromotion":True,"totalWeeklyExamples":n,
        "monthlyDubaiDataUsed": dubai is not None,
        "monthlyDubaiAvailableWeekCount": dubai_added,
        "officialDailyDubaiAvailableWeekCount": official_count,
        "cpcPolicyAnnouncementWeeks":policy_count,
        "cpcPolicyArticleCount":len(policy["notices"]) if policy else 0,
        "cpcPolicyHistoricalDataOnly":True,
        "officialDailySource": "https://data.gov.tw/dataset/6029" if official is not None else None,
        "officialDailyThreeDayQuoteDelay": True,
        "DubaiEmbargoDaysAfterMonthEnd": 75,
        "monthlyDubaiIsNOTOfficialCpc7D3B": True,
        "timePointGuarantee":"Historical news and CPC notice publication precede feature origin Monday; target-week Sunday announcements never available",
        "modelFamily":"pooled CatBoost; optional zero-inflated change/no-change",
        "historicalFolds":history,"historicalBaseline":baseline_val,
        "historicalMeanMAE":mean_val,"historicalBaselineMeanMAE":base_mae,
        "historicalBetterFoldCounts":improved,
        "selectedUsingEarlierFoldsOnly":choice,
        "lockedFinal13Weeks":{"start":rows[audit]["targetDate"],
            "end":rows[-1]["targetDate"],"scores":holdout},
        "warning":"News features are metadata; absence before archive coverage is NOT proof of no wars or no supply events. Historical IMF monthly Dubai and MOEA daily Dubai/Brent are retrospective compilations, not independently verified original market publication vintages; 75-day and 3-day quote delays are conservative assumptions."}
    o=pathlib.Path(args.output);o.mkdir(parents=True,exist_ok=True)
    (o/"accuracy-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"status":"complete","version":args.variant,"weeks":n,
        "baselineVal":base_mae,"candidateVal":mean_val,
        "dubaiAvailableWeeks":dubai_added, "officialDailyAvailableWeeks":official_count,
        "policyAvailableWeeks":policy_count,
        "earlierFolds":len(folds),
        "improvedFolds":improved,"selected":choice,
        "locked13WeekMAE":{k:round(v["mae"],4) for k,v in holdout.items()},
        "output":str(o/"accuracy-report.json")},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
