#!/usr/bin/env python3
"""Bounded public-source context acquisition for stable out-of-time research.

GDELT v1 daily archives are sampled on Fridays, embargoed until +2d UTC
(or +36h after the latest DATEADDED timestamp). Occurrence date is NEVER
used as availability. No article bodies or URLs are exported from the rows.
CPC neighbor tables are publication-date reconstructions, not verified old
HTML vintages; the original frozen reference price must still agree.
"""
import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
from html.parser import HTMLParser
import io
import json
import math
from pathlib import Path
import re
import time
import urllib.request
import zipfile

UTC=dt.timezone.utc
MAX_ZIP=32_000_000
MAX_INFLATED=256_000_000
MIDDLE_EAST={"IRN","IRQ","SAU","ARE","QAT","KWT","OMN","YEM","ISR","SYR","TUR","EGY"}
EUROPE_RISK={"RUS","UKR"}
ENERGY_URL=re.compile(r"oil|crude|brent|hormuz|opec|tanker|refiner|shipping|suez|sanction",re.I)
COUNTRIES=("日本","韓國","香港","新加坡","台灣中油")


def download(url,limit):
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={"User-Agent":"OilCastResearch/1.0 (+https://github.com/lins58308/oilcast-training-private)"})
            with urllib.request.urlopen(request,timeout=35) as response:
                raw=response.read(limit+1)
                if len(raw)>limit:raise ValueError("Source exceeds bounded input size")
                return raw
        except Exception:
            if attempt==2:raise
            time.sleep(.5*(attempt+1))


def aggregate_gdelt(raw,day):
    if day<dt.date(2013,4,1):raise ValueError("Pre-2013 occurrence-date archives are not accepted")
    if len(raw)>MAX_ZIP:raise ValueError("Oversized archive")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        files=[f for f in archive.infolist() if not f.is_dir()]
        if len(files)!=1 or files[0].file_size>MAX_INFLATED:raise ValueError("Unexpected ZIP content or inflated size")
        counts={k:0 for k in ("events","materialConflict","verbalConflict","middleEast","middleEastConflict","europeRisk","europeConflict","energyUrl","energyConflict","negativeGoldstein","goldsteinKnown","toneKnown","quadKnown")}
        seen=set();tones=[];last_added=dt.datetime.combine(day,dt.time(),UTC)
        future=0
        with archive.open(files[0]) as source:
            for row in csv.reader(io.TextIOWrapper(source,encoding='utf-8'),delimiter='\t'):
                if len(row)!=58:raise ValueError("Expected 58-field GDELT v1 daily stream")
                added=row[56]
                if not added.isdecimal() or len(added) not in (8,14):raise ValueError("Missing DATEADDED provenance")
                added_at=dt.datetime.strptime(added,"%Y%m%d%H%M%S" if len(added)==14 else "%Y%m%d").replace(tzinfo=UTC)
                if added_at.date()>day:
                    future+=1;continue
                identity=row[0]
                if identity in seen:continue
                seen.add(identity)
                if len(seen)>1_000_000:raise ValueError("Unexpected event count")
                last_added=max(last_added,added_at)
                quad=int(row[29]) if row[29] else 0
                gold=float(row[30]) if row[30] else None
                tone=float(row[34]) if row[34] else None
                if quad not in (0,1,2,3,4) or gold is not None and not (-10<=gold<=10) or tone is not None and not math.isfinite(tone):raise ValueError("Invalid event numeric fields")
                country={row[7],row[17]}
                middle=bool(country&MIDDLE_EAST);europe=bool(country&EUROPE_RISK)
                energy=bool(ENERGY_URL.search(row[57]))
                flags={"events":True,"materialConflict":quad==4,"verbalConflict":quad==3,
                    "middleEast":middle,"middleEastConflict":middle and quad>=3,
                    "europeRisk":europe,"europeConflict":europe and quad>=3,
                    "energyUrl":energy,"energyConflict":energy and quad>=3,
                    "negativeGoldstein":gold is not None and gold<0,"goldsteinKnown":gold is not None,
                    "toneKnown":tone is not None,"quadKnown":quad!=0}
                for key,value in flags.items():counts[key]+=int(value)
                if energy and tone is not None:tones.append(max(-100.,min(100.,tone)))
        if counts['events']<100:raise ValueError("Too few valid sampled events")
    available=max(dt.datetime.combine(day+dt.timedelta(days=2),dt.time(),UTC),last_added+dt.timedelta(hours=36))
    return {"date":day.isoformat(),"availableAt":available.isoformat().replace('+00:00','Z'),
        "counts":counts,"energyMeanTone":sum(tones)/len(tones) if tones else None,
        "ignoredFutureDateAdded":future,"source":"https://data.gdeltproject.org/events/"+day.strftime('%Y%m%d')+'.export.CSV.zip',
        "sha256":hashlib.sha256(raw).hexdigest(),"compressedBytes":len(raw),
        "sampling":"Friday-only daily file; not full weekly news","availabilityIsConservativeAssumption":True}


class Tables(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True);self.stack=[];self.tables=[];self.text=[]
    def handle_starttag(self,tag,attrs):
        if tag=='table':self.stack.append({'rows':[],'row':None,'cell':None})
        elif self.stack and tag=='tr':self.stack[-1]['row']=[]
        elif self.stack and tag in ('td','th'):self.stack[-1]['cell']=[]
    def handle_data(self,data):
        self.text.append(data)
        if self.stack and self.stack[-1]['cell'] is not None:self.stack[-1]['cell'].append(data)
    def handle_endtag(self,tag):
        if not self.stack:return
        top=self.stack[-1]
        if tag in ('td','th') and top['cell'] is not None:
            if top['row'] is not None:top['row'].append(' '.join(''.join(top['cell']).split()))
            top['cell']=None
        elif tag=='tr' and top['row'] is not None:
            top['rows'].append(top['row']);top['row']=None
        elif tag=='table':self.tables.append(self.stack.pop()['rows'])


def cpc_neighbors(raw,notice):
    html=raw.decode('utf-8-sig');parser=Tables();parser.feed(html)
    text=' '.join(parser.text)
    match=re.search(r'刊登日期\s*[:：]\s*(\d{2,3})[-/](\d{1,2})[-/](\d{1,2})',text)
    if not match:raise ValueError('CPC original publication date missing')
    day=dt.date(1911+int(match[1]),int(match[2]),int(match[3]))
    if day.isoformat()!=notice['publishedDate']:raise ValueError('CPC historical publication changed')
    candidates=[]
    for table in parser.tables:
        header=next((r for r in table if all(any(k in c for c in r) for k in ('92','柴油'))),None)
        if header is None:continue
        cols=[]
        for key in ('92','95','98','柴油'):
            cols.append(next((i for i,c in enumerate(header) if key in c),None))
        countries={}
        for row in table:
            if not row:continue
            label=row[0].replace('臺','台')
            country=next((c for c in COUNTRIES if c in label),None)
            if country is None and ('台灣' in label and '台塑' not in label or label=='中油'):country='台灣中油'
            if country is None or max(c for c in cols if c is not None)>=len(row):continue
            prices=[]
            for col in cols:
                if col is None:
                    prices.append(None);continue
                m=re.match(r'^\s*(\d+(?:\.\d+)?)',row[col].replace(',',''))
                value=float(m[1]) if m else None
                if value is not None and not 1<=value<=400:raise ValueError('Neighbor price outside expected retail range')
                prices.append(value)
            countries[country]=prices
        if len(countries)>=3 and any(c!='台灣中油' for c in countries):candidates.append(countries)
    candidates=list({json.dumps(c,sort_keys=True,ensure_ascii=False):c for c in candidates}.values())
    if len(candidates)!=1:raise ValueError('Missing or ambiguous dated neighbor table')
    countries=candidates[0];taiwan=countries.get('台灣中油')
    reference=notice.get('gasoline92Reference')
    if taiwan and taiwan[0] is not None and reference is not None and abs(taiwan[0]-reference)>.051:
        raise ValueError('Retrieved CPC table disagrees with frozen historical CPC price')
    return {'id':notice['id'],'publishedDate':notice['publishedDate'],
        'availableAt':notice['assumedAvailableAt'],'countries':countries,'source':notice['url'],
        'sha256':hashlib.sha256(raw).hexdigest(),'historicalHtmlVintageVerified':False,
        'isRetailGapProxyNotOfficialPreTaxCeiling':True}


def collect(policy,output,probe=False,friday_start=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    payload=json.loads(Path(policy).read_text());notices=payload['notices']
    selected=([notices[0],notices[len(notices)//2],notices[-1]] if probe else notices)
    neighbor=[];errors=[]
    def fetch_notice(n):return cpc_neighbors(download(n['url'],600_000),n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(fetch_notice,n):n for n in selected}
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            n=futures[future]
            try:neighbor.append(future.result())
            except Exception as exc:errors.append({'id':n['id'],'error':str(exc)[:180]})
            if i%40==0:print(json.dumps({'stage':'neighbors','completed':i,'valid':len(neighbor),'errors':len(errors)}),flush=True)
    gdelt=[]
    days=([dt.date(2013,4,5),dt.date(2019,1,4),dt.date(2026,9,4)] if probe else [])
    if friday_start:
        start=dt.date.fromisoformat(friday_start)
        if start.weekday()!=4 or start<dt.date(2013,4,5):raise ValueError('Friday source stream starts no earlier than 2013-04-05')
        end=dt.date(2026,10,30)
        days=[start+dt.timedelta(days=7*i) for i in range((end-start).days//7+1)]
    def fetch_day(day):
        try:
            url='https://data.gdeltproject.org/events/'+day.strftime('%Y%m%d')+'.export.CSV.zip'
            return aggregate_gdelt(download(url,MAX_ZIP),day)
        finally:time.sleep(.5)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(fetch_day,day):day for day in days}
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            day=futures[future]
            try:gdelt.append(future.result())
            except Exception as exc:errors.append({'date':day.isoformat(),'error':str(exc)[:180]})
            if i%40==0:print(json.dumps({'stage':'gdelt-first-seen','completed':i,'valid':len(gdelt),'errors':len(errors)}),flush=True)
    result={'schema':1,'retrievedAt':dt.datetime.now(UTC).isoformat(),'probeOnly':probe,
        'policySourceSha256':hashlib.sha256(Path(policy).read_bytes()).hexdigest(),
        'neighbors':sorted(neighbor,key=lambda r:r['publishedDate']),'gdelt':sorted(gdelt,key=lambda r:r['date']),'errors':errors,
        'noAutomaticPromotion':True,'noArticleBodiesRetained':True}
    (output/'context.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'stage':'source-context-complete','probe':probe,'neighbors':len(neighbor),'gdelt':len(gdelt),'errors':errors},ensure_ascii=False),flush=True)
    if friday_start and (len(neighbor)<100 or len(gdelt)<len(days)*.9):
        raise ValueError('Insufficient dated auxiliary coverage; errors preserved, not filled with zero')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',required=True);p.add_argument('--output',required=True)
    p.add_argument('--probe',action='store_true');p.add_argument('--friday-start')
    a=p.parse_args();collect(a.policy,a.output,a.probe,a.friday_start)
