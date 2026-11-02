#!/usr/bin/env python3
"""Auditable 7D3B proxy inputs using lagged monthly Dubai observations.

CPC's actual benchmark is 70% Dubai + 30% Brent *weekly* Platts index.
The publicly accessible IMF/FRED POILDUBUSDM is MONTHLY, NOT the same data.
Never fill an earlier origin with its eventual full-month average, and never
advertise this as an official CPC formula reproduction. A 62-day embargo is
a conservative approximation, not verified as a historical publication vintage.
"""
import argparse
import csv
from datetime import date, datetime, timedelta
import json
from pathlib import Path


def parse_monthly_dubai(csv_string):
    reader = csv.DictReader(csv_string.lstrip("\ufeff").splitlines())
    if not reader.fieldnames or list(reader.fieldnames) != ["observation_date", "POILDUBUSDM"]:
        raise ValueError("Unexpected FRED Dubai CSV series schema")
    rows = []
    for entry in reader:
        month = date.fromisoformat(entry["observation_date"])
        if month.day != 1:
            raise ValueError("Dubai observations must be monthly")
        raw = entry["POILDUBUSDM"].strip()
        if raw in ("", "."):
            continue
        value = float(raw)
        if not (5 <= value <= 300):
            raise ValueError("Dubai price outside plausible source range")
        rows.append((month, value))
    rows.sort()
    if len(set(x[0] for x in rows)) != len(rows):
        raise ValueError("Duplicate monthly dates")
    return rows


def one_origin_proxy(origin_iso, brent_weekly_pct, fx_weekly_pct, monthly, embargo_days=62):
    from bisect import bisect_right
    timestamp = datetime.fromisoformat(origin_iso.replace("Z", "+00:00")).date()
    admissible = [(m, value) for m, value in monthly if m + timedelta(days=embargo_days) <= timestamp]
    if len(admissible) < 2:
        return [0., 0., 0., 0.]
    (prior_month, prior), (last_month, last) = admissible[-2:]
    # Do not treat gaps in monthly observations as continuous monthly change.
    distance = (last_month.year - prior_month.year) * 12 + last_month.month - prior_month.month
    if distance != 1:
        return [0., 0., 0., 0.]
    dubai_month_pct = 100 * (last / prior - 1)
    # This 7D3B proxy mixes monthly Dubai with weekly Brent and is not an
    # operational CPC price formula; evaluated strictly as an experimental input.
    blend_pct = .70 * dubai_month_pct + .30 * brent_weekly_pct
    return [
        float(max(-100, min(100, dubai_month_pct))),
        float(max(-100, min(100, blend_pct))),
        float(max(-100, min(100, blend_pct * (1 + fx_weekly_pct / 100)))),
        1.,
    ]


def attach_dubai_proxy(dataset, rows):
    for example in dataset["examples"]:
        feature = example["features"][0]
        if len(feature) != 26:
            raise ValueError("Unexpected baseline numeric feature dimension")
        example["policyProxy"] = one_origin_proxy(
            example["cutoff"], float(feature[10]), float(feature[12]), rows
        )
    dataset["policyProxyProvenance"] = {
        "series": "FRED POILDUBUSDM monthly IMF Dubai prices",
        "url": "https://fred.stlouisfed.org/series/POILDUBUSDM",
        "cpcFormula": "Not the original 7D3B Platts weekly index",
        "conservativeEmbargoDays": 62,
        "isHistoricalReleaseVintage": False,
        "isResearchOnly": True,
    }
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dubai-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    series = parse_monthly_dubai(Path(args.dubai_csv).read_text())
    if len(series) < 200:
        raise ValueError("Insufficient genuine monthly Dubai observations")
    data = json.loads(Path(args.dataset).read_text())
    attach_dubai_proxy(data, series)
    valid = sum(e["policyProxy"][-1] > 0 for e in data["examples"])
    print(json.dumps({"dubaiMonths": len(series), "supervisedWeeks": len(data["examples"]),
                      "weeksWithCausallyEmbargoedDubai": valid, "lastMonth": series[-1][0].isoformat()}))
    Path(args.output).write_text(json.dumps(data, separators=(",", ":")))


if __name__ == "__main__":
    main()
