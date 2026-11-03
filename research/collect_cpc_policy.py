#!/usr/bin/env python3
"""Collect dated CPC gasoline/diesel price policy notices for OilCast research.

Source: CPC public press releases (NOT future Sunday adjustment tables).
Only historical notices *already published by the input Monday* may be used.
Source dates lack exact server-first-published timestamps; conservatively
mark available at local day end (23:59:59 Asia/Taipei). Extract short numeric
fields and source URLs only; never reproduce long copyrighted article bodies.
This is separate from future target labels and must not overwrite Android.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta, timezone
import json
import re
import time as clock
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

BASE="https://www.cpc.com.tw"
LISTING=BASE+"/news2.aspx"
DETAIL=BASE+"/News_Content.aspx"
LOCAL=timezone(timedelta(hours=8))
MAX_HTML=600_000
FILTER=re.compile(r"汽[、，／及]?柴油|汽油|柴油|汽、柴|油價|調價")
EXCLUDE=re.compile(r"液化石油氣|桶裝瓦斯|天然氣|LPG")
DATE_PATTERN=re.compile(r"刊登日期[:：]\s*(\d{2,3})[-/](\d{1,2})[-/](\d{1,2})")
FUEL_PAIR=re.compile(r"汽[、，／及]?柴油.{0,45}?(?:應|原應)[^，。]{0,16}?調(漲|降)\s*(\d+(?:\.\d+)?)\s*元\s*(?:及|、)\s*(\d+(?:\.\d+)?)\s*元")
TWO_AMOUNTS=re.compile(r"(?:汽[、，／及]?柴油|汽、柴油)[^。]{0,70}?(?:各共吸收|合計[^。]{0,10}各共吸收|各吸收)\s*(\d+(?:\.\d+)?)\s*元\s*(?:及|、)\s*(\d+(?:\.\d+)?)\s*元")
SINGLE_PRICE=re.compile(r"92\s*無鉛汽油.{0,12}?(?:公升)?\s*(\d+(?:\.\d+)?)\s*元")
NONFUEL_TITLE=re.compile(r"職安|職業安全|事故|淨零|徵才|台灣中油舉辦|液化石油氣|天然氣|桶裝瓦斯")


def http_text(session,url,params=None):
    for attempt in range(3):
        try:
            response=session.get(url,params=params,timeout=(10,35))
            response.raise_for_status()
            if len(response.content)>MAX_HTML:
                raise ValueError("CPC public page unexpectedly large")
            response.encoding=response.apparent_encoding if response.encoding in (None,"ISO-8859-1") else response.encoding
            text=response.text
            if "新聞" not in text and "刊登日期" not in text:
                raise ValueError("Official site returned unexpected content")
            return text
        except (requests.RequestException,ValueError):
            if attempt==2:raise
            clock.sleep(.8*(attempt+1))


def listing_pages(pages):
    session=requests.Session()
    session.headers["User-Agent"]="OilCastPublicResearch/1.0 (+https://github.com/lins58308/oilcast-models)"
    urls={}
    errors=[]
    try:
        for page in range(1,pages+1):
            params={"PageSize":20,"n":28,"page":page,"sms":8920}
            try: html=http_text(session,LISTING,params)
            except Exception as exc:
                errors.append({"page":page,"error":str(exc)[:140]})
                continue
            soup=BeautifulSoup(html,"html.parser")
            seen=set()
            for link in soup.select('a[href*="News_Content.aspx"]'):
                href=link.get("href","")
                m=re.search(r"(?:[?&])n=28&s=(\d+)",href,re.I)
                if not m:continue
                key=int(m.group(1))
                if key in seen:continue
                seen.add(key)
                title=link.get_text(" ",strip=True)
                if not FILTER.search(title) or NONFUEL_TITLE.search(title):
                    continue
                if EXCLUDE.search(title) and not re.search(r"汽[、]?柴油|汽油|柴油",title):continue
                urls[key]=title
            if page%10==0:
                print(json.dumps({"stage":"list","pages":page,"candidateTitles":len(urls),"errors":len(errors)}),flush=True)
            clock.sleep(.10)
    finally:
        session.close()
    return urls,errors


def amount(n):
    x=float(n)
    if not (0<=x<=100):raise ValueError("Policy amount outside 0–100 NTD/L")
    return round(x,3)


def announcement(key,listing_title):
    session=requests.Session()
    session.headers["User-Agent"]="OilCastPublicResearch/1.0 (+https://github.com/lins58308/oilcast-models)"
    url=DETAIL+"?"+urlencode({"n":28,"s":key})
    try: html=http_text(session,url)
    finally:session.close()
    soup=BeautifulSoup(html,"html.parser")
    text=soup.get_text(" ",strip=True)
    dates=DATE_PATTERN.search(text)
    if not dates:return None
    year=1911+int(dates.group(1))
    d=date(year,int(dates.group(2)),int(dates.group(3)))
    if not date(2018,1,1)<=d<=date.today()+timedelta(days=2):
        return None
    article=text[dates.end():]
    for boundary in ("相關檔案","相關圖片","回上一頁"):
        if boundary in article:article=article.split(boundary,1)[0]
    article=article[:13500]
    if not re.search(r"92\s*無鉛汽油|超級柴油",article) or not FILTER.search(article[:300]):
        return None
    # Actual date available is unknown: mark conservatively at END of
    # the CPC date, rather than forging a Sunday-noon publication instant.
    available=datetime.combine(d,time(23,59,59),tzinfo=LOCAL).astimezone(timezone.utc)
    theoretical=FUEL_PAIR.search(article)
    sign=1 if theoretical and theoretical.group(1)=="漲" else -1
    theoreticalGas=sign*amount(theoretical.group(2)) if theoretical else None
    theoreticalDiesel=sign*amount(theoretical.group(3)) if theoretical else None
    absorptionGas=absorptionDiesel=None
    taxGas=taxDiesel=None
    # Unambiguous public CPC article HTML tables: header contains
    # "政府調降貨物稅"; cells labelled gasoline/diesel and last=total.
    for table in soup.find_all("table"):
        t=table.get_text(" ",strip=True)
        if not ("汽油" in t and "柴油" in t and ("政府調降貨物稅" in t or "吸收金額" in t)):
            continue
        for tr in table.find_all("tr"):
            columns=[c.get_text(" ",strip=True) for c in tr.find_all(["th","td"])]
            if len(columns)<3:continue
            label=columns[0]
            if not (label.strip()=="汽油" or label.strip()=="柴油"):continue
            numeric=[float(s.replace(",","")) for s in columns[1:] if re.fullmatch(r"\d+(?:\.\d+)?",s.replace(",","").strip())]
            if len(numeric)>=2 and all(0<=x<=100 for x in numeric):
                if label.strip()=="汽油":taxGas=round(numeric[0],3);absorptionGas=round(numeric[-1],3)
                else:taxDiesel=round(numeric[0],3);absorptionDiesel=round(numeric[-1],3)
        if absorptionGas is not None and absorptionDiesel is not None:break
    # If table is absent, prefer explicit sentence. Never infer one fuel
    # from the other, and never assume a missing field equals zero.
    if absorptionGas is None or absorptionDiesel is None:
        matches=list(TWO_AMOUNTS.finditer(article))
        if matches:
            pair=matches[-1]
            absorptionGas=amount(pair.group(1)) if absorptionGas is None else absorptionGas
            absorptionDiesel=amount(pair.group(2)) if absorptionDiesel is None else absorptionDiesel
    price=SINGLE_PRICE.search(article)
    gasolineReference=amount(price.group(1)) if price else None
    freezing=bool(re.search(r"汽[、，／及]?柴油.{0,45}?(?:維持不調整|皆不(?:予)?調整|均不(?:予)?調整|價格不調整)",listing_title+" "+article[:240]))
    holiday=bool(re.search(r"只跌不漲|上漲不調|春節期間",article[:2000]))
    emergency=bool(re.search(r"專案平穩|擴大吸收|加碼吸收|美伊戰爭|中東戰事",article[:2000]))
    if all(v is None for v in (theoreticalGas,absorptionGas,gasolineReference)) and not(freezing or holiday or emergency):
        return None
    return {
       "schema":1,"id":str(key),"url":url,"title":listing_title[:140],
       "publishedDate":d.isoformat(),"assumedAvailableAt":available.isoformat().replace("+00:00","Z"),
       "assumedAvailability":True,
       "theoreticalGas":theoreticalGas,"theoreticalDiesel":theoreticalDiesel,
       "absorptionGas":absorptionGas,"absorptionDiesel":absorptionDiesel,
       "taxReliefGas":taxGas,"taxReliefDiesel":taxDiesel,
       "gasoline92Reference":gasolineReference,
       "bothFuelFrozen":freezing,"specialAbsorption":emergency,"holidayNoRise":holiday
    }


def collect(pages,max_workers):
    candidates,errors=listing_pages(pages)
    out=[]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        tasks={pool.submit(announcement,key,title):key for key,title in candidates.items()}
        for number,task in enumerate(as_completed(tasks),1):
            try:
                data=task.result()
                if data:out.append(data)
            except Exception as exc:
                errors.append({"id":tasks[task],"error":str(exc)[:180]})
            if number%50==0:
                print(json.dumps({"stage":"details","done":number,"total":len(tasks),"accepted":len(out),"errors":len(errors)}),flush=True)
    out.sort(key=lambda x:(x["publishedDate"],int(x["id"])))
    if len(out)<80:raise ValueError("Too few verified historical CPC policy press releases")
    if out[-1]["publishedDate"]<"2026-09-01":
        raise ValueError("Latest CPC archive announcements missing")
    return {
       "schema":1,"source":"https://www.cpc.com.tw/news2.aspx?n=28",
       "availabilityRule":"CPC publication date at 23:59:59 Asia/Taipei; exact publication timestamp unknown",
       "notFuturePolicyAnswers":True,
       "listingPagesAttempted":pages,"candidates":len(candidates),
       "notices":out,"fetchErrors":errors,
       "coverage": {
           "earliest":out[0]["publishedDate"],"latest":out[-1]["publishedDate"],
           "total":len(out),
           "withTheoretical":sum(x["theoreticalGas"] is not None for x in out),
           "withAbsorption":sum(x["absorptionGas"] is not None for x in out),
           "special":sum(x["specialAbsorption"] for x in out)
       }
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--pages",type=int,default=44)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--output",default="output/cpc-policy-history.json")
    args=parser.parse_args()
    if not 3<=args.pages<=70 or not 1<=args.workers<=4:
        parser.error("Strict public-site rate-limit: pages 3..70, workers 1..4")
    import pathlib
    result=collect(args.pages,args.workers)
    destination=pathlib.Path(args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({"stage":"complete",**result["coverage"],"errors":len(result["fetchErrors"]),
                      "output":str(destination)},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
