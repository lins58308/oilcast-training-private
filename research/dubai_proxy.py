"""Conservative lagged Dubai crude monthly benchmark (RESEARCH ONLY).

Source: IMF monthly Dubai crude price POILDUBUSDM, hosted by FRED.
This is not the original weekly CPC 70% Dubai + 30% Brent Platts
indicator, and revised historical series are not contemporary vintages.
A 75-day delay after month end prevents accidentally using a month
before it could plausibly have been reported, but does NOT establish
actual original first-publication timestamps. It cannot auto-promote.
"""
import bisect
import csv
import datetime as dt
import math


def month_end(month):
    if month.day != 1:
        raise ValueError("Dubai observation must be a first-of-month date")
    following = dt.date(month.year + (month.month == 12),
                        1 if month.month == 12 else month.month + 1, 1)
    return following - dt.timedelta(days=1)


def parse_dubai_csv(raw):
    records = []
    seen = set()
    parser = csv.DictReader(raw.splitlines())
    if parser.fieldnames != ["observation_date", "POILDUBUSDM"]:
        raise ValueError("Unexpected monthly IMF Dubai CSV schema")
    for row in parser:
        date_text = row["observation_date"]
        month = dt.date.fromisoformat(date_text)
        if month.day != 1 or month in seen:
            raise ValueError("Dubai month not unique or not monthly")
        seen.add(month)
        value = row["POILDUBUSDM"].strip()
        if value in ("", "."):
            continue
        price = float(value)
        if not math.isfinite(price) or not (1 <= price <= 500):
            raise ValueError("Invalid Dubai US-dollar crude price")
        # Historical monthly averages may be finalized/revised after
        # month close. Use a conservative 75-day latency embargo.
        available = month_end(month) + dt.timedelta(days=75)
        records.append((available, month, price))
    records.sort()
    if len(records) < 240 or records[0][1] > dt.date(1998, 1, 1):
        raise ValueError("Dubai monthly file does not cover historic CPC data")
    return records


def lagged_dubai_features(weeks, monthly):
    dates = [row[0] for row in monthly]
    output = []
    available_rows = 0
    for item in weeks:
        cutoff = dt.date.fromisoformat(item["cutoff"][:10])
        ix = bisect.bisect_right(dates, cutoff) - 1
        if ix < 1:
            output.append((0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        available, month, price = monthly[ix]
        # Do not compute month-over-month changes across missing months.
        _, earlier_month, previous = monthly[ix-1]
        gap = (month.year-earlier_month.year)*12+month.month-earlier_month.month
        pct = 100*(price/previous-1) if gap == 1 else 0.0
        # The observed weekly Brent/FX moves are already censored by the
        # independent strict as-of snapshot in ContextData.
        world = item["features"][0]
        brent_pct, fx_pct = float(world[10]), float(world[12])
        # Approximate 70/30 commodity-mix change under 80% transfer;
        # monthly Dubai is highly delayed, NOT a weekly CPC 7D3B value.
        mix = 0.8 * (0.7*pct + 0.3*brent_pct + fx_pct)
        age = min(365.0, float((cutoff-available).days))
        output.append((1.0, price/100, pct, mix, age/365))
        available_rows += 1
    return output, available_rows
