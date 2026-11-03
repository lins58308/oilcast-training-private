"""Causal end-of-week carry and dated auxiliary context for Monday forecasts."""
import bisect
import datetime as dt
import math
import numpy as np


def carry_features(rows, observed):
    days=sorted(observed);values=[]
    for row in rows:
        origin=dt.date.fromisoformat(row['origin'])
        if origin.weekday()!=0:raise ValueError('Expected CPC Monday origin')
        end=origin-dt.timedelta(days=3)
        quote=[observed[d] for d in days if end-dt.timedelta(days=7)<d<=end]
        if len(quote)<3:
            values.append([0.]*10);continue
        a=np.array(quote);mix=.7*a[:,1]+.3*a[:,2]
        mean=float(mix.mean());last=float(mix[-1]);first=float(mix[0])
        pct=lambda x,y:100*(x/y-1)
        # A constant-latest-spot forecast of next week's mean differs from
        # the completed week's mean. This is a persistence proxy, not CPC's
        # licensed Platts series or knowledge of next week's actual quotes.
        values.append([1.,min(len(quote),5)/5.,mean/100,last/100,
            pct(last,mean),pct(last,first),float(np.std(np.diff(mix)/mix[:-1]*100)),
            pct(float(a[-1,1]),float(a[:,1].mean())),
            pct(float(a[-1,2]),float(a[:,2].mean())),
            .8*pct(last,mean)])
    return np.array(values)


def auxiliary_features(rows, context, bases):
    """Published tables and first-seen samples only; absent sources are masked."""
    result=np.zeros((len(rows),4,22))
    neighbors=sorted(context.get('neighbors',[]),key=lambda r:r['availableAt'])
    gdelt=sorted(context.get('gdelt',[]),key=lambda r:r['availableAt'])
    stamp=lambda s:dt.datetime.fromisoformat(s.replace('Z','+00:00'))
    nt=[stamp(n['availableAt']) for n in neighbors];gt=[stamp(n['availableAt']) for n in gdelt]
    for i,row in enumerate(rows):
        cutoff=stamp(row['cutoff']);ni=bisect.bisect_right(nt,cutoff)-1;gi=bisect.bisect_right(gt,cutoff)-1
        for f in range(4):
            if ni>=0 and (cutoff-nt[ni]).total_seconds()<=18*86400:
                now=neighbors[ni];country={k:v[f] for k,v in now['countries'].items() if k!='台灣中油' and v[f] is not None}
                if country:
                    minimum=min(country.values());delta=0.;known=0.
                    if ni>0:
                        old=neighbors[ni-1];previous={k:v[f] for k,v in old['countries'].items() if k!='台灣中油' and v[f] is not None}
                        interval=(nt[ni]-nt[ni-1]).total_seconds()/86400
                        if country.keys()==previous.keys() and 5<=interval<=9:
                            delta=minimum-min(previous.values());known=1.
                    result[i,f,:8]=[1.,len(country)/4.,minimum/30.,(minimum-bases[i,f])/10.,
                        delta,known,(cutoff-nt[ni]).total_seconds()/86400/18.,float(now.get('historicalHtmlVintageVerified',False))]
            if gi>=0 and (cutoff-gt[gi]).total_seconds()<=10*86400:
                item=gdelt[gi];c=item['counts'];n=max(1,c['events']);me=max(1,c['middleEast']);eu=max(1,c['europeRisk'])
                tone=item.get('energyMeanTone')
                result[i,f,8:]=[1.,math.log1p(n)/15.,c['materialConflict']/n,c['verbalConflict']/n,
                    c['middleEast']/n,c['middleEastConflict']/me,c['europeRisk']/n,c['europeConflict']/eu,
                    c['energyUrl']/n,c['energyConflict']/max(1,c['energyUrl']),
                    (tone or 0)/100.,float(tone is not None),c['goldsteinKnown']/n,
                    (cutoff-gt[gi]).total_seconds()/86400/10.]
    return result


def source_coverage(rows):
    yearly={}
    for row in rows:
        year=row['origin'][:4];record=yearly.setdefault(year,{'weeks':0,'newsWeeks':0,'nonEiaWeeks':0})
        record['weeks']+=1;record['newsWeeks']+=bool(row['news'])
        record['nonEiaWeeks']+=any(n['source']!='EIA' for n in row['news'])
    return yearly
