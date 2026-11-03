"""Time-causal policy-absorption features derived from verified dated CPC notices.

The CPC article published by last Sunday belongs to the Monday at the
feature origin; its theoretical change, absorption and observed freeze are
PAST policy state when predicting the *following* Monday.
An article dated the future Sunday immediately preceding the target
would be label leakage and is never consulted.

All missing policy values are masked; no claim of zero subsidies in old eras.
Research-only features: not part of Android's signed 26-feature interface.
"""
import bisect
import datetime as dt
import math


def normalize_policy(payload):
    if payload.get("schema")!=1 or not payload.get("notFuturePolicyAnswers"):
        raise ValueError("Policy archive lacks public source/time contract")
    items=payload["notices"]
    times=[]
    sanitized=[]
    ids=set()
    for row in items:
        link=row.get("url","")
        if not link.startswith("https://www.cpc.com.tw/News_Content.aspx?n=28&s="):
            raise ValueError("Policy source outside official CPC domain")
        identity=row["id"]
        if identity in ids:raise ValueError("Duplicate CPC press release")
        ids.add(identity)
        published=dt.date.fromisoformat(row["publishedDate"])
        available=dt.datetime.fromisoformat(row["assumedAvailableAt"].replace("Z","+00:00"))
        if available.tzinfo is None:
            raise ValueError("Unknown official publication timezone")
        # Mark at the END of original publication day Taipei, never
        # mid-day or historic backdated as though first observed.
        local=available.astimezone(dt.timezone(dt.timedelta(hours=8))).date()
        if local!=published:
            raise ValueError("Policy source date and assumed availability differ")
        for key in ("theoreticalGas","theoreticalDiesel","absorptionGas",
                    "absorptionDiesel","taxReliefGas","taxReliefDiesel"):
            val=row.get(key)
            if val is not None and (not isinstance(val,(float,int)) or
                not math.isfinite(val) or abs(val)>100):
                raise ValueError("Nonfinite or absurd policy effect")
        times.append(available)
        sanitized.append(row)
    indices=sorted(range(len(times)),key=lambda i:times[i])
    return [times[i] for i in indices],[sanitized[i] for i in indices]


def publication_week_indices(weeks, index, limit=53):
    """Latest notice per consecutive publication week; stop at a missing week."""
    if not 0<=index<len(weeks) or limit<1:
        raise ValueError("Invalid policy publication history index")
    selected=[index]
    if limit==1:return selected
    for j in range(index-1,-1,-1):
        if weeks[j] != weeks[selected[-1]]:
            if weeks[selected[-1]]-weeks[j] != dt.timedelta(days=7):break
            selected.append(j)
            if len(selected)==limit:break
    return selected


def feature_for_period(rows, payload, max_age=18):
    if not 7<=max_age<=28:raise ValueError("Source stale limit must be 7–28d")
    dates,notices=normalize_policy(payload)
    published=[dt.date.fromisoformat(n['publishedDate']) for n in notices]
    weeks=[d-dt.timedelta(days=d.weekday()) for d in published]
    output=[]
    covered=0
    for item in rows:
        cutoff=dt.datetime.fromisoformat(item["cutoff"].replace("Z","+00:00"))
        local_origin=dt.date.fromisoformat(item["origin"])
        # Weekly training cutoff must be the exact Monday start Taipei.
        expected=dt.datetime.combine(local_origin,dt.time(0),tzinfo=dt.timezone(dt.timedelta(hours=8)))
        if local_origin.weekday()!=0 or cutoff!=expected:
            raise ValueError("Policy origin is not CPC Monday 00:00 Asia/Taipei")
        ix=bisect.bisect_right(dates,cutoff)-1
        selected=None
        if ix>=0 and (cutoff-dates[ix]).total_seconds()<=max_age*86400:
            selected=notices[ix]
        preceding=publication_week_indices(weeks,ix,8) if selected is not None else []
        per_fuel=[]
        for fuel in range(4):
            if selected is None:
                per_fuel.append([0.]*12)
                continue
            is_diesel=fuel==3
            name="Diesel" if is_diesel else "Gas"
            theoretical=selected.get("theoretical"+name)
            absorption=selected.get("absorption"+name)
            tax=selected.get("taxRelief"+name)
            freeze=bool(selected.get("bothFuelFrozen"))
            special=bool(selected.get("specialAbsorption"))
            holiday=bool(selected.get("holidayNoRise"))
            # Count unique consecutive publication weeks, not duplicate
            # notices or apparent freezes across an unobserved week.
            streak=0
            for j in preceding:
                older=notices[j]
                if older.get("bothFuelFrozen"):streak+=1
                else:break
            # A repeated amount means zero weekly change. A changed-price
            # week still has an absorption change; a missing week is masked.
            old_abs=notices[preceding[1]].get("absorption"+name) if len(preceding)>1 else None
            absorption_change=0.
            if absorption is not None and old_abs is not None:
                absorption_change=max(-30.,min(30.,absorption-old_abs))
            age_days=(cutoff-dates[ix]).total_seconds()/86400
            per_fuel.append([
                1.,                                          # verified report available
                float(theoretical or 0), float(theoretical is not None),
                float(absorption or 0), float(absorption is not None),
                float(tax or 0),                              # tax relief is separate
                float(freeze),float(special),float(holiday),
                min(1.,age_days/18.),
                float(streak)/8.,
                absorption_change/10.,
            ])
        output.append(per_fuel)
        if selected is not None:covered+=1
    return output,covered
