#!/usr/bin/env python3
"""Actual weekly refits and separate stability criteria, not an average-only win.

Four fixed learners compare carry-only and enriched context. No model may
use a target's outcome when fitting that forecast. Audit results from 2026
were already inspected; passing retrospective diagnostics never constitutes
prospective proof or production authorization.
"""
import argparse
from dataclasses import dataclass,asdict
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from mae_walk_forward import load_frozen,round_prices,FUELS,GROUPS,json_write,metrics
from official_daily_oil import parse_moea_daily
from policy_state import policy_states
from stability_features import carry_features,auxiliary_features,source_coverage

SEED=20261103


@dataclass(frozen=True)
class Candidate:
    name:str
    learner:str
    enriched:bool


CANDIDATES=tuple(Candidate(f'{learner}-{source}',learner,source=='enriched')
                 for source in ('carry','enriched') for learner in ('robust','absolute','regime','policy'))


class Robust:
    def fit(self,x,y,weights):
        self.mean=np.average(x,axis=0,weights=weights)
        self.scale=np.sqrt(np.average((x-self.mean)**2,axis=0,weights=weights));self.scale[self.scale<1e-6]=1.
        z=np.column_stack((np.ones(len(x)),np.clip((x-self.mean)/self.scale,-4,4)))
        penalty=np.eye(z.shape[1])*3.;penalty[0,0]=0.
        solve=lambda w:np.linalg.solve(z.T@(w[:,None]*z)+penalty,z.T@(w*y))
        b=solve(weights)
        for _ in range(8):
            residual=z@b-y;w=weights*.1/np.sqrt(residual**2+.01)
            next=solve(w)
            if np.max(abs(next-b))<1e-6:
                b=next;break
            b=next
        self.beta=b;return self
    def predict(self,x):
        return np.column_stack((np.ones(len(x)),np.clip((x-self.mean)/self.scale,-4,4)))@self.beta


def booster():
    return HistGradientBoostingRegressor(loss='absolute_error',max_iter=70,max_leaf_nodes=7,
        min_samples_leaf=16,l2_regularization=10,learning_rate=.05,
        early_stopping=False,random_state=SEED)


def features(num,base,carry,policy,aux,variant):
    result={}
    for head,f in enumerate((0,3)):
        other=3 if f==0 else 0
        regime=[]
        for i in range(len(num)):
            seen=num[max(0,i-51):i+1,f,0];run=0
            for value in seen[::-1]:
                if abs(value)>=.049:break
                run+=1
            regime.append([min(run,52)/52.,np.mean(abs(seen[-4:])>=.049),np.mean(abs(seen[-13:])>=.049)])
        common=np.column_stack((num[:,f,:18],num[:,other,:4],base/30.,carry,regime))
        if variant=='v7':
            lagged=[]
            for lag in (1,2,4):
                a=np.zeros((len(num),3));a[lag:]=num[:-lag,f,:3];lagged.append(a)
            common=np.column_stack((common,*lagged))
        # Explicit interactions allow linear carry to change with known policy.
        interaction=np.column_stack((carry[:,4]*policy[:,f,13],carry[:,4]*policy[:,f,7],
            carry[:,4]*num[:,f,12]/100.,carry[:,4]*base[:,f]/100.))
        covered=policy[:,f,1]>.5
        result[head]={'base':common,'carry':np.column_stack((common,policy[:,f],interaction)),
            'enriched':np.column_stack((common,policy[:,f],interaction,aux[:,f],num[:,f,18:26])),
            'policyCovered':covered}
    return result


def one_origin(data,labels,origin,spec):
    if not 100<=origin<len(labels):raise ValueError('Past labels required before a forecast origin')
    output=np.zeros(4)
    age=.5+.5*np.exp(-np.arange(origin)[::-1]/104.)
    for head,members in enumerate(GROUPS):
        z=data[head];key='enriched' if spec.enriched else 'carry'
        x=z[key][:origin];query=z[key][origin:origin+1];y=labels[:origin,head]
        if spec.learner=='robust':
            model=Robust().fit(x,y,age);p=float(model.predict(query)[0])
        elif spec.learner=='absolute':
            model=booster().fit(x,y,sample_weight=age);p=float(model.predict(query)[0])
        elif spec.learner in ('regime','policy'):
            model=Robust().fit(z['base'][:origin],y,age)
            p=float(model.predict(z['base'][origin:origin+1])[0])
            available=z['policyCovered'][:origin]
            if z['policyCovered'][origin] and available.sum()>=52:
                # A within-origin residual learner, not a gate evaluated on
                # in-sample errors. Outer forecast errors remain out of time.
                if spec.learner=='policy':
                    # The global head keeps all older observations; a local
                    # expert uses all observations with known policy inputs.
                    local=booster().fit(x[available],y[available],sample_weight=age[available])
                    p=float(local.predict(query)[0])
                else:
                    base_train=model.predict(z['base'][:origin][available])
                    residual=y[available]-base_train
                    local=booster().fit(np.column_stack((x[available],base_train)),residual,sample_weight=age[available])
                    p+=float(local.predict(np.column_stack((query,[p])))[0])
        else:raise ValueError('Unregistered learner')
        for f in members:output[f]=p
    return np.clip(output,-3,3)


def score(y,p):
    return {f:metrics(y[:,i],p[:,i]) for i,f in enumerate(FUELS)}


def train_choices(history,baseline,variant):
    result=[]
    for group in GROUPS:
        eligible=[];gates={}
        for name,folds in history.items():
            detail={}
            for f in group:
                fuel=FUELS[f];s=[r[fuel] for r in folds];b=[r[fuel] for r in baseline]
                wins=[q['mae']+.01<ref['mae']-1e-9 for q,ref in zip(s[:5],b[:5])]
                old_mean=np.mean([q['mae'] for q in s[:5]]);base_mean=np.mean([q['mae'] for q in b[:5]])
                consistent=sum(wins)>=3 and sum(wins[:4])>=3 and old_mean+.01<base_mean-1e-9
                recent=(all(q['mae']+.015<ref['mae']-1e-9 for q,ref in zip(s[5:],b[5:]))
                    if variant=='v6' else s[-1]['mae']+.01<b[-1]['mae']-1e-9)
                # A stable claim additionally requires both recent eras and
                # no material regression in any independent older period.
                cross_era=all(q['mae']+.01<ref['mae']-1e-9 for q,ref in zip(s[5:],b[5:]))
                noninferior=all(q['mae']<=ref['mae']+.005+1e-9 for q,ref in zip(s[:4],b[:4]))
                direction=s[-1]['directionAccuracy']>=b[-1]['directionAccuracy']-.05
                false=s[-1]['falseMoveRate'] is not None and s[-1]['falseMoveRate']<=.25
                detail[fuel]={'olderMeanMae':float(old_mean),'olderBaselineMae':float(base_mean),
                    'olderImprovedFolds':int(sum(wins)),'independentOlderImprovedFolds':int(sum(wins[:4])),
                    'consistencyPassed':bool(consistent),'originalRecentMarginsPassed':bool(recent),
                    'bothRecentErasImproved':bool(cross_era),'independentOlderNoMaterialRegression':bool(noninferior),
                    'directionPassed':bool(direction),'falseMovePassed':bool(false),
                    'passed':bool(consistent and recent and cross_era and noninferior and direction and false)}
            gates[name]=detail
            if all(r['passed'] for r in detail.values()):eligible.append((np.mean([r['olderMeanMae'] for r in detail.values()]),name))
        eligible.sort();result.append({'fuels':[FUELS[f] for f in group],
            'selected':eligible[0][1] if eligible else 'unchanged','candidateGates':gates})
    return result


def evaluate(root,variant,out,context=None,only_carry=False):
    rows,num,y,base,oil,_,manifest,coverage=load_frozen(root,variant)
    observed=parse_moea_daily((Path(root)/'moea-daily.csv').read_bytes())
    carry=carry_features(rows,observed)
    policy=policy_states(rows,json.loads((Path(root)/'policy.json').read_text()))
    ctx=json.loads(Path(context).read_text()) if context else {}
    if context and ctx.get('probeOnly'):raise ValueError('A three-date probe is not training history')
    if context and not ctx.get('collectionComplete'):raise ValueError('Source collection is incomplete; resume acquisition first')
    aux=auxiliary_features(rows,ctx,base)
    data=features(num,base,carry,policy,aux,variant)
    labels=np.column_stack((np.median(y[:,:3],axis=1),y[:,3]))
    candidates=[s for s in CANDIDATES if not s.enriched or not only_carry and context]
    n=len(y);audit=n-13
    folds=[(audit-gap-13,audit-gap) for gap in (260,208,156,104,52)]
    folds.extend(((audit-78,audit-39),(audit-39,audit)))
    baseline=[score(y[a:b],np.zeros((b-a,4))) for a,b in folds]
    history={};trace=[];unique_predictions={}
    for spec in candidates:
        history[spec.name]=[]
        for k,(a,b) in enumerate(folds):
            pred=[]
            for t in range(a,b):pred.append(one_origin(data,labels,t,spec))
            p=round_prices(np.array(pred),base[a:b]);history[spec.name].append(score(y[a:b],p))
            for j,t in enumerate(range(a,b)):
                key=(spec.name,t)
                if key in unique_predictions and not np.allclose(unique_predictions[key],p[j],atol=1e-9):
                    raise ValueError('Same information origin produced a different forecast in an overlapping fold')
                unique_predictions[key]=p[j].copy()
                trace.append({'candidate':spec.name,'fold':k,'origin':rows[t]['origin'],
                    'targetDate':rows[t]['targetDate'],'actualDeltas':y[t].tolist(),'predictedDeltas':p[j].tolist()})
            print(json.dumps({'stage':'weekly-refit','variant':variant,'candidate':spec.name,'fold':k,
                'through':rows[b-1]['targetDate'],'mae':{f:round(history[spec.name][-1][f]['mae'],6) for f in FUELS}}),flush=True)
    choices=train_choices(history,baseline,variant)
    selection={'schema':1,'variant':variant,'sourceManifest':manifest,'weeklyRefitting':True,
        'candidates':[asdict(s) for s in candidates],'historicalNewsCoverage':source_coverage(rows),
        'auxiliaryCoverage':{'neighborWeeks':int((aux[:,0,0]>.5).sum()),'firstSeenRiskWeeks':int((aux[:,0,8]>.5).sum()),
            'historicalNeighborCoverageSufficient':bool(ctx.get('historicalNeighborCoverageSufficient',False))},
        'folds':[{'start':rows[a]['targetDate'],'end':rows[b-1]['targetDate']} for a,b in folds],
        'historicalCandidates':history,'historicalBaseline':baseline,'selectedBeforeAudit':choices,
        'overlapping2025SummerFoldNotCountedAsIndependentConfirmation':True,
        'auditPreviouslyExamined':True,'automaticPromotionApproved':False}
    out=Path(out);json_write(out/'selection.json',selection)
    json_write(out/'validation-predictions.json',trace)
    digest=hashlib.sha256((out/'selection.json').read_bytes()).hexdigest()
    p=np.zeros((13,4))
    for head,group in enumerate(GROUPS):
        name=choices[head]['selected']
        if name=='unchanged':continue
        spec=next(s for s in candidates if s.name==name)
        for i,t in enumerate(range(audit,n)):
            point=one_origin(data,labels,t,spec)
            for f in group:p[i,f]=point[f]
    p=round_prices(p,base[audit:]);s=score(y[audit:],p);b=score(y[audit:],np.zeros_like(p))
    passed=all(s[f]['mae']+.01<b[f]['mae']-1e-9 for f in FUELS)
    report={**selection,'selectionSha256':digest,'audit':{'candidate':s,'baseline':b},
        'allFourHistoricalAuditGatesPassed':bool(passed),'stableProspectivePerformanceEstablished':False,
        'sourceTimestampLimitations':['Retrospective MOEA quotes use a three-day lag assumption',
            'CPC neighbor HTML is not verified original-publication HTML',
            'GDELT Friday sampling is not full weekly news coverage']}
    json_write(out/'report.json',report)
    print(json.dumps({'stage':'weekly-stability-complete','variant':variant,
        'selected':[c['selected'] for c in choices],'auditMae':{f:s[f]['mae'] for f in FUELS},
        'allFourPassed':passed}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--variant',choices=('v6','v7'),required=True)
    p.add_argument('--output',required=True);p.add_argument('--context');p.add_argument('--only-carry',action='store_true')
    a=p.parse_args();evaluate(a.inputs,a.variant,a.output,a.context,a.only_carry)
