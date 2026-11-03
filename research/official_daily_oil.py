"""MOEA Taiwan government daily Dubai/Brent/WTI benchmark research.

Open data 6029: https://www2.moeaea.gov.tw/oil111/csv/GlobalOilOpenData.csv
Dates are exchange observation dates, not guaranteed original first-publication
vintages. For every CPC Monday origin this module excludes observations in
the most recent 3 calendar days. This approximates conservative publication
lag; archived revisions remain a research limitation and are NEVER used to
auto-promote a signed Android model.
"""
import csv
import datetime as dt
import io
import math


def parse_moea_daily(raw):
    text=raw.decode("utf-8-sig") if isinstance(raw,bytes) else raw.lstrip("\ufeff")
    reader=csv.DictReader(io.StringIO(text))
    expected=["日期","西德州(美元/桶)","杜拜(美元/桶)","布蘭特(美元/桶)","備註"]
    if reader.fieldnames!=expected:
        raise ValueError("MOEA official daily crude CSV schema changed")
    data={}
    for row in reader:
        value=row["日期"].strip()
        if not len(value)==8 or not value.isdecimal():
            raise ValueError("Invalid MOEA YYYYMMDD date")
        day=dt.datetime.strptime(value,"%Y%m%d").date()
        if day in data:raise ValueError("Repeated MOEA crude observation date")
        prices=[]
        incomplete=False
        for index,key in enumerate(expected[1:4]):
            s=row[key].strip()
            if s.upper() in ("N/A","NA","-",""):
                incomplete=True
                break
            try: p=float(s)
            except ValueError: raise ValueError("Unrecognized government crude price")
            # Cushing WTI futures famously settled below zero in April
            # 2020. Do NOT discard true historical negative WTI quotes;
            # Dubai/Brent must remain strictly positive.
            minimum = -100 if index == 0 else 1
            if not math.isfinite(p) or not minimum<=p<=1000:
                raise ValueError("MOEA crude daily price outside valid range")
            prices.append(p)
        # Missing observations are not zero prices. Keep the gap and never
        # forward-fill a market quote as though it was published that day.
        if incomplete:continue
        data[day]=tuple(prices)
    if len(data)<4000 or min(data)>dt.date(2001,1,1):
        raise ValueError("MOEA series insufficient for historical oil training")
    return data


def point_in_time_weekly(weeks, observed):
    """Seven research-only features; never use the target week's prices.

    A CPC origin is Monday midnight Taipei. Latest permitted raw price
    date is Thursday/Friday of the preceding period (three calendar
    days before cutoff), which is no later than known as-of data.
    """
    values=[]
    covered=0
    observations=sorted(observed.items())
    for item in weeks:
        cutoff=dt.date.fromisoformat(item["cutoff"][:10])
        latest=cutoff-dt.timedelta(days=3)
        earlier=latest-dt.timedelta(days=7)
        oldest=earlier-dt.timedelta(days=7)
        current=[p for day,p in observations if earlier<day<=latest]
        prior=[p for day,p in observations if oldest<day<=earlier]
        if len(current)<3 or len(prior)<3:
            values.append((0.,0.,0.,0.,0.,0.,0.))
            continue
        def average(rows, index):
            return sum(p[index] for p in rows)/len(rows)
        dubai=average(current,1)
        brent=average(current,2)
        dubai_prior=average(prior,1)
        brent_prior=average(prior,2)
        mix=.7*dubai+.3*brent
        oldmix=.7*dubai_prior+.3*brent_prior
        pct=lambda n,o:100*(n/o-1)
        fx_pct=float(item["features"][0][12])
        if not math.isfinite(fx_pct) or abs(fx_pct)>100:
            raise ValueError("Invalid as-of exchange change")
        # Official 70/30 crude weighting; estimated 80% pass-through
        # and FX interaction, but NOT a reconstructed CPC posted price.
        step=pct(mix,oldmix)
        v=(1.,mix/100,step,pct(dubai,dubai_prior),
           pct(brent,brent_prior),
           (dubai-brent)/max(1,mix),
           .8*((1+step/100)*(1+fx_pct/100)-1)*100)
        values.append(v)
        covered+=1
    return values,covered
