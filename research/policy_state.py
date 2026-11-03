"""Publication-week-aware CPC state; never reads the target announcement.

Unlike the legacy twelve-feature view, absorption differences compare the
immediately preceding publication WEEK, including weeks with a price change.
A repeated amount has a zero difference; a missing week has no difference.
Total absorption and tax relief are kept separate and individually masked.
"""
import bisect
import datetime as dt
import numpy as np
from policy_features import normalize_policy, publication_week_indices

NAMES = ("notice_available", "notice_fresh", "age_fraction_of_18_days", "total_absorption",
         "absorption_known", "tax_relief", "tax_known", "net_cpc_absorption",
         "net_known", "prior_week_absorption_change", "change_known",
         "prior_change_acceleration", "acceleration_known", "frozen",
         "consecutive_frozen_publication_weeks", "special", "holiday",
         "theoretical_change", "theoretical_known", "previous_week_available")


def monday(day):
    return day-dt.timedelta(days=day.weekday())


def policy_states(rows, payload):
    times, notices = normalize_policy(payload)
    source_weeks = [monday(dt.date.fromisoformat(p["publishedDate"])) for p in notices]
    result = np.zeros((len(rows), 4, len(NAMES)), dtype=float)
    for k, row in enumerate(rows):
        cutoff = dt.datetime.fromisoformat(row["cutoff"].replace("Z", "+00:00"))
        origin=dt.date.fromisoformat(row['origin'])
        expected=dt.datetime.combine(origin,dt.time(),tzinfo=dt.timezone(dt.timedelta(hours=8)))
        if origin.weekday()!=0 or cutoff!=expected:
            raise ValueError('Policy state must use Monday 00:00 Asia/Taipei')
        ix = bisect.bisect_right(times, cutoff)-1
        if ix < 0 or (cutoff-times[ix]).total_seconds() > 18*86400:
            continue
        # Retain the latest already-known notice within each publication week.
        selected = publication_week_indices(source_weeks,ix,53)
        now = notices[ix]
        previous = notices[selected[1]] if len(selected)>1 else None
        older = notices[selected[2]] if len(selected)>2 else None
        run = 0
        for j in selected:
            if notices[j].get("bothFuelFrozen") is not True:
                break
            run += 1
        age = (cutoff-times[ix]).total_seconds()/86400
        for f in range(4):
            suffix = "Diesel" if f == 3 else "Gas"
            amount = now.get("absorption"+suffix)
            tax = now.get("taxRelief"+suffix)
            before = previous.get("absorption"+suffix) if previous else None
            earliest = older.get("absorption"+suffix) if older else None
            delta_ok = amount is not None and before is not None
            acceleration_ok = delta_ok and earliest is not None
            net_ok = amount is not None and tax is not None and amount+1e-8 >= tax
            theoretical = now.get("theoretical"+suffix)
            result[k,f] = [1., float(age<=7), age/18., amount or 0., float(amount is not None),
                tax or 0., float(tax is not None), amount-tax if net_ok else 0., float(net_ok),
                amount-before if delta_ok else 0., float(delta_ok),
                amount-2*before+earliest if acceleration_ok else 0., float(acceleration_ok),
                float(now.get("bothFuelFrozen") is True), min(run,52)/52.,
                float(now.get("specialAbsorption") is True),float(now.get("holidayNoRise") is True),
                theoretical or 0.,float(theoretical is not None),float(previous is not None)]
    return result
