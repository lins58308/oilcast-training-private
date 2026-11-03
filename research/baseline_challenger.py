#!/usr/bin/env python3
"""Conditional-distribution corrections to a no-change forecast (research).

Actions minimize conditional absolute loss, including zero as an action.
An optional one-standard-error margin suppresses weak estimated gains. That
margin is a heuristic, NOT a confidence guarantee under financial dependence.
Every label used to fit a distribution predates its prediction block.
"""
import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from mae_walk_forward import FUELS, GROUPS, json_write, load_frozen, metrics, round_prices
from policy_state import NAMES, policy_states

SEED = 20261103


@dataclass(frozen=True)
class Challenger:
    name: str
    method: str
    inputs: str
    leaf: int = 16
    risk_margin: float = 0.


# Fixed before running this experiment; the known 2026 audit is diagnostic.
CANDIDATES = (
    Challenger("kernel-market", "kernel", "market"),
    Challenger("cdf-market", "forest", "market"),
    Challenger("cdf-policy", "forest", "policy"),
    Challenger("cdf-policy-wide", "forest", "policy", 32),
    Challenger("cdf-policy-cautious", "forest", "policy", 16, 1.),
    Challenger("cdf-news-cautious", "forest", "news", 32, 1.),
)


def conditional_action(labels, weights, risk_margin=0.):
    """Weighted conditional median minimizes expected MAE; zero always allowed."""
    y, w = np.asarray(labels, dtype=float), np.asarray(weights, dtype=float)
    if y.ndim!=1 or y.shape!=w.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(w).all() or np.min(w)<0 or w.sum()<=0:
        raise ValueError("Invalid empirical forecast distribution")
    if risk_margin not in (0., 1.):
        raise ValueError("Unregistered risk margin")
    w = w/w.sum()
    order = np.argsort(y, kind="stable")
    index = min(len(y)-1, np.searchsorted(np.cumsum(w[order]), .5, side="left"))
    action = float(np.floor(y[order[index]]*10+.5)/10)
    benefit = np.abs(y)-np.abs(y-action)
    gain = float(w@benefit)
    # Forest neighborhoods are estimated from the same training observations;
    # this is an abstention heuristic, not a calibrated coverage interval.
    standard_error = float(np.sqrt(np.sum(w*w*(benefit-gain)**2)))
    if gain <= max(0., risk_margin*standard_error)+1e-12:
        action = 0.
    return action, {"estimatedGain":gain,"gainStandardError":standard_error,
                    "effectiveObservations":float(1/(w@w)),
                    "upProbability":float(w@(y>.049)),"downProbability":float(w@(y<-.049))}


class Distribution:
    def __init__(self, spec):
        self.spec=spec

    def fit(self, x, y):
        self.labels=np.asarray(y, dtype=float).copy()
        if x.ndim!=2 or len(x)!=len(y) or len(y)<100 or not np.isfinite(x).all():
            raise ValueError("Invalid past-only distribution training data")
        self.age=.5+.5*np.exp(-np.arange(len(y))[::-1]/104.)
        if self.spec.method=="kernel":
            self.scaler=StandardScaler().fit(x,sample_weight=self.age)
            self.source=np.clip(self.scaler.transform(x),-6,6)
        elif self.spec.method=="forest":
            self.model=ExtraTreesRegressor(n_estimators=64,max_depth=7,
                min_samples_leaf=self.spec.leaf,max_features=.8,criterion="absolute_error",
                bootstrap=False,n_jobs=2,random_state=SEED)
            self.model.fit(x,self.labels,sample_weight=self.age)
            self.leaves=self.model.apply(x)
        else:
            raise ValueError("Unknown conditional distribution")
        return self

    def predict(self, x):
        if self.spec.method=="forest":
            query=self.model.apply(x)
        else:
            query=np.clip(self.scaler.transform(x),-6,6)
        values=[];evidence=[]
        for row in query:
            if self.spec.method=="forest":
                weight=np.zeros(len(self.labels))
                for tree,leaf in enumerate(row):
                    members=self.leaves[:,tree]==leaf
                    local=members*self.age
                    weight+=local/local.sum()/self.leaves.shape[1]
            else:
                distance=np.mean((self.source-row)**2,axis=1)
                bandwidth=max(1e-8,float(np.partition(distance,min(63,len(distance)-1))[min(63,len(distance)-1)]))
                weight=self.age*np.exp(np.clip(-distance/bandwidth,-50,0))
            action,why=conditional_action(self.labels,weight,self.spec.risk_margin)
            values.append(action);evidence.append(why)
        return np.asarray(values),evidence


def feature_sets(numeric, base, oil, state, variant):
    result={}
    for head,f in enumerate((0,3)):
        other=3 if f==0 else 0
        regime=[]
        for i in range(len(numeric)):
            history=numeric[max(0,i-51):i+1,f,0]
            run=0
            for a in history[::-1]:
                if abs(a)>=.049: break
                run+=1
            regime.append([min(run,52)/52.,float(np.mean(abs(history[-4:])>=.049)),
                           float(np.mean(abs(history[-13:])>=.049))])
        market=np.column_stack((numeric[:,f,[0,4,5,6,10,11,12,13,17]],
            numeric[:,other,0],base[:,f]/30.,regime,oil[:,[0,2,6]]))
        policy=np.column_stack((market,state[:,f]))
        news=np.column_stack((policy,numeric[:,f,18:26]))
        if variant=="v7":
            # Temporal detail is from past feature origins, never future labels.
            context=[]
            for lag in (1,2,4):
                z=np.zeros((len(numeric),3));z[lag:]=numeric[:-lag,f,:3];context.append(z)
            market=np.column_stack((market,*context))
            policy=np.column_stack((policy,*context));news=np.column_stack((news,*context))
        result[head]={"market":market,"policy":policy,"news":news}
    return result


def block_predict(features, targets, start, end, spec):
    if not 100<=start<end<=len(targets):
        raise ValueError("Training origin must precede forecast labels")
    prediction=np.zeros((end-start,4));evidence={}
    for head,members in enumerate(GROUPS):
        x=features[head][spec.inputs]
        model=Distribution(spec).fit(x[:start],targets[:start,head])
        values,why=model.predict(x[start:end]);evidence[str(head)]=why
        for f in members: prediction[:,f]=values
    return prediction,evidence


def summarize(pred, y):
    result={}
    for f,name in enumerate(FUELS):
        actual,prediction=y[:,f],pred[:,f]
        moving=np.abs(actual)>=.049
        gain=float(np.sum(np.abs(actual[moving])-np.abs(actual[moving]-prediction[moving])))
        false_cost=float(np.sum(np.abs(prediction[~moving])))
        result[name]={**metrics(actual,prediction),"gainOnChangedWeeks":gain,
            "costOnUnchangedWeeks":false_cost,"netErrorReduction":gain-false_cost,
            "gainRequiredFor0_01Mae":.01*len(actual)}
    return result


def choose(history, baseline, variant):
    choices=[]
    for members in GROUPS:
        gates={};eligible=[]
        for spec in CANDIDATES:
            details={}
            for f in members:
                name=FUELS[f]
                scores=[row[name] for row in history[spec.name]]
                bases=[row[name] for row in baseline]
                old_mean=float(np.mean([p["mae"] for p in scores[:5]]))
                old_base=float(np.mean([p["mae"] for p in bases[:5]]))
                wins=sum(p["mae"]+.01<b["mae"]-1e-9 for p,b in zip(scores[:5],bases[:5]))
                consistent=old_mean+.01<old_base-1e-9 and wins>=3
                recent=(all(p["mae"]+.015<b["mae"]-1e-9 for p,b in zip(scores[5:],bases[5:]))
                    if variant=="v6" else scores[-1]["mae"]+.01<bases[-1]["mae"]-1e-9)
                # Preserve the existing per-fuel movement/direction safeguards.
                last=scores[-1];last_base=bases[-1]
                directional=last["directionAccuracy"]>=last_base["directionAccuracy"]-.05
                false_ok=last["falseMoveRate"] is not None and last["falseMoveRate"]<=.25
                passed=bool(consistent and recent and directional and false_ok)
                details[name]={"olderMeanMae":old_mean,"olderBaselineMae":old_base,
                    "olderWinningFolds":wins,"olderConsistencyPassed":bool(consistent),
                    "recentOriginalMaeMarginsPassed":bool(recent),"directionPassed":bool(directional),
                    "falseMovePassed":bool(false_ok),"passed":passed}
            gates[spec.name]=details
            if all(d["passed"] for d in details.values()):
                eligible.append((np.mean([d["olderMeanMae"] for d in details.values()]),spec.name))
        eligible.sort()
        choices.append({"fuels":[FUELS[f] for f in members],"selected":eligible[0][1] if eligible else "unchanged",
                        "eligibleCandidates":[q[1] for q in eligible],"candidateGates":gates})
    return choices


def online_guard(actual, candidate, window=26):
    """At week i, decisions use ONLY settled outcomes strictly before i."""
    y,p=np.asarray(actual),np.asarray(candidate)
    if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=4 or window!=26:
        raise ValueError("Expected four-fuel trace and the registered 26-week guard")
    prediction=np.zeros_like(p);enabled=np.zeros_like(p,dtype=bool)
    for i in range(window,len(y)):
        for members in GROUPS:
            old_y=y[i-window:i][:,members];old_p=p[i-window:i][:,members]
            # Every grade in a shared gasoline head must show a past gain.
            if np.all(np.mean(abs(old_y-old_p),axis=0)+.005 < np.mean(abs(old_y),axis=0)-1e-9):
                for f in members:prediction[i,f]=p[i,f];enabled[i,f]=True
    return prediction,enabled


def paired_uncertainty(actual,prediction,draws=2000):
    """Descriptive paired 52-week block bootstrap, not fresh trial evidence."""
    benefit=abs(np.asarray(actual))-abs(np.asarray(actual)-np.asarray(prediction))
    if benefit.ndim!=2 or benefit.shape[1]!=4 or not len(benefit):
        raise ValueError("Expected four-fuel paired loss differences")
    n=len(benefit);block=52;sample=[];rng=np.random.default_rng(SEED)
    for _ in range(draws):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        ids=((starts[:,None]+np.arange(block))%n).reshape(-1)[:n]
        sample.append(benefit[ids].mean(axis=0))
    sample=np.asarray(sample);interval=np.quantile(sample,[.025,.975],axis=0)
    return {"method":"paired circular 52-week block bootstrap","draws":draws,
        "descriptiveOnlyPreviouslyExaminedHistory":True,
        "byFuel":{f:{"meanMaeReduction":float(benefit[:,i].mean()),
            "lower":float(interval[0,i]),"upper":float(interval[1,i])} for i,f in enumerate(FUELS)},
        "allFourPositiveLowerQuantile":float(np.quantile(sample.min(axis=1),.025))}


def comprehensive_replay(features,targets,y,base,rows,audit,variant,output,start=260):
    """Continuous 13-week frozen fits, all outcomes before the known audit."""
    if start!=260 or start>=audit:
        raise ValueError("Replay starts after the registered 260 training origins")
    spec=next(s for s in CANDIDATES if s.name=="cdf-policy-cautious")
    raw=[];ranges=[]
    for a in range(start,audit,13):
        b=min(a+13,audit)
        pred,_=block_predict(features,targets,a,b,spec)
        raw.append(round_prices(pred,base[a:b]));ranges.append((a,b))
        if len(raw)%15==0:
            print(json.dumps({"stage":"continuous-replay","variant":variant,"blocks":len(raw),"through":rows[b-1]['targetDate']}),flush=True)
    raw=np.concatenate(raw)
    actual=y[start:audit]
    guarded,enabled=online_guard(actual,raw)
    periods={"all":np.ones(len(actual),dtype=bool)}
    years=np.array([int(r['targetDate'][:4]) for r in rows[start:audit]])
    periods.update({"2004-2018":years<=2018,"2019-2025":(years>=2019)&(years<=2025),"2026-before-audit":years>=2026})
    scores={}
    for name,mask in periods.items():
        if not mask.any(): continue
        scores[name]={"weeks":int(mask.sum()),"baseline":summarize(np.zeros_like(actual[mask]),actual[mask]),
            "conditionalMedian":summarize(raw[mask],actual[mask]),"pastOnlyGuarded":summarize(guarded[mask],actual[mask]),
            "guardEnabledWeeks":{f:int(enabled[mask,i].sum()) for i,f in enumerate(FUELS)}}
    report={"schema":1,"variant":variant,"candidate":asdict(spec),"from":rows[start]['targetDate'],
        "through":rows[audit-1]['targetDate'],"weeklyOrigins":len(actual),"trainingStart":rows[0]['origin'],
        "fullExpandingTraining":True,"frozenFitBlockWeeks":13,"guardUsesOnlyEarlier26SettledWeeks":True,
        "knownFinal13WeeksExcluded":True,"retrospectiveDiagnosticsNotBlind":True,"scores":scores,
        "pairedUncertainty":{"conditionalMedian":paired_uncertainty(actual,raw),
                             "pastOnlyGuarded":paired_uncertainty(actual,guarded)},
        "automaticPromotionApproved":False}
    out=Path(output);json_write(out/'continuous-replay.json',report)
    json_write(out/'continuous-replay-predictions.json',[{"origin":rows[start+i]['origin'],
        "targetDate":rows[start+i]['targetDate'],"actualDeltas":actual[i].tolist(),
        "conditionalMedianDeltas":raw[i].tolist(),"pastOnlyGuardedDeltas":guarded[i].tolist(),
        "guardEnabled":enabled[i].tolist()} for i in range(len(actual))])
    print(json.dumps({"stage":"continuous-replay-complete","variant":variant,"weeks":len(actual),
        "mae":{name:{method:{f:round(m[f]['mae'],6) for f in FUELS} for method,m in data.items() if method in ('baseline','conditionalMedian','pastOnlyGuarded')} for name,data in scores.items()}}),flush=True)
    return report


def evaluate(root,variant,output,replay=False):
    rows,num,y,base,oil,legacy,manifest,coverage=load_frozen(root,variant)
    state=policy_states(rows,json.loads((Path(root)/"policy.json").read_text()))
    features=feature_sets(num,base,oil,state,variant)
    targets=np.column_stack((np.median(y[:,:3],axis=1),y[:,3]))
    audit=len(y)-13
    folds=[(audit-gap-13,audit-gap) for gap in (260,208,156,104,52)]
    folds.extend(((audit-78,audit-39),(audit-39,audit)))
    history={};baseline=[summarize(np.zeros((b-a,4)),y[a:b]) for a,b in folds]
    for spec in CANDIDATES:
        scores=[]
        for a,b in folds:
            pred,_=block_predict(features,targets,a,b,spec)
            scores.append(summarize(round_prices(pred,base[a:b]),y[a:b]))
        history[spec.name]=scores
        print(json.dumps({"stage":"pre-audit","variant":variant,"candidate":spec.name,
            "olderMae":{f:round(float(np.mean([s[f]["mae"] for s in scores[:5]])),6) for f in FUELS},
            "recentMae":{f:[round(s[f]["mae"],6) for s in scores[5:]] for f in FUELS}}),flush=True)
    decisions=choose(history,baseline,variant)
    selection={"schema":1,"variant":variant,"sourceManifest":manifest,
        "method":"zero-inclusive conditional median with optional relative-loss abstention",
        "candidateSpecs":[asdict(s) for s in CANDIDATES],"policyStateNames":list(NAMES),
        "coverage":coverage,"pastOnlyTraining":True,"forecastHorizonDays":7,
        "folds":[{"start":rows[a]["targetDate"],"end":rows[b-1]["targetDate"]} for a,b in folds],
        "preAuditCandidates":history,"preAuditBaseline":baseline,"selectedBeforeAudit":decisions,
        "auditPreviouslyInspected":True,"auditLabelsUsedForSelection":False,
        "riskMarginIsHeuristic":True,"automaticPromotionApproved":False}
    out=Path(output);json_write(out/"selection.json",selection)
    digest=hashlib.sha256((out/"selection.json").read_bytes()).hexdigest()
    prediction=np.zeros((len(y)-audit,4));why={}
    for head,members in enumerate(GROUPS):
        name=decisions[head]["selected"]
        if name=="unchanged": continue
        spec=next(s for s in CANDIDATES if s.name==name)
        pred,evidence=block_predict(features,targets,audit,len(y),spec)
        for f in members:prediction[:,f]=pred[:,f]
        why[str(head)]=evidence[str(head)]
    prediction=round_prices(prediction,base[audit:])
    audit_scores=summarize(prediction,y[audit:]);audit_base=summarize(np.zeros_like(prediction),y[audit:])
    all_four=all(audit_scores[f]["mae"]+.01<audit_base[f]["mae"]-1e-9 for f in FUELS)
    report={**selection,"selectionSha256":digest,"audit":{"start":rows[audit]["targetDate"],
        "end":rows[-1]["targetDate"],"candidate":audit_scores,"baseline":audit_base},
        "errorBudgetRule":"changed-week error reduction must exceed unchanged-week false-movement cost plus 0.01 times the number of scored weeks",
        "allFourHistoricalAuditGatesPassed":bool(all_four)}
    json_write(out/"report.json",report)
    json_write(out/"audit-predictions.json",[{"origin":r["origin"],"targetDate":r["targetDate"],
        "actualDeltas":y[audit+i].tolist(),"predictedDeltas":prediction[i].tolist()}
        for i,r in enumerate(rows[audit:])])
    print(json.dumps({"stage":"complete","variant":variant,
        "selected":[d["selected"] for d in decisions],"auditMae":{f:audit_scores[f]["mae"] for f in FUELS},
        "allFourPassed":all_four}),flush=True)
    if replay:
        comprehensive_replay(features,targets,y,base,rows,audit,variant,out)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--inputs",required=True)
    p.add_argument("--variant",choices=("v6","v7"),required=True);p.add_argument("--output",required=True)
    p.add_argument("--continuous-replay",action="store_true")
    a=p.parse_args();evaluate(a.inputs,a.variant,a.output,a.continuous_replay)
