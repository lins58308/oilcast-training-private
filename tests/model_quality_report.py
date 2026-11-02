#!/usr/bin/env python3
"""Recompute holdout diagnostics from a historical model package; never promotes it."""
import json
import math
import pathlib
import sys

def direction(change):
    return (change > 0.049) - (change < -0.049)

def report(path):
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    model = payload["model"]
    assert len(model["fuels"]) == 4, "missing a fuel"
    fuels = []
    for fuel in model["fuels"]:
        # All known weekly labels after the required 12-week lag window
        # must enter the final fit. Holdout applies to model selection only.
        full_history_samples = model["weeks"] - 13
        assert full_history_samples > 156, "fixture cannot verify expanding history"
        assert fuel["trainingSamples"] == full_history_samples, (
            "final fit silently discarded historical CPC weeks"
        )
        if fuel["family"] != 0:
            assert fuel["fit"]["samples"] == full_history_samples, (
                "active fitted model did not use the whole history"
            )
        points = fuel["backtest"]
        assert len(points) == fuel["testCount"] == 13, "unexpected locked audit size"
        errors = [abs(p["predicted"] - p["actual"]) for p in points]
        mae = sum(errors) / len(errors)
        assert math.isfinite(mae) and abs(mae - fuel["mae"]) < 1e-7, "MAE disagrees with backtest"
        direction_hits = 0
        movement_count = 0
        movement_hits = 0
        false_moves = 0
        no_change = 0
        for p in points:
            actual = direction(p["actual"] - p["base"])
            predicted = direction(p["predicted"] - p["base"])
            direction_hits += actual == predicted
            if actual:
                movement_count += 1
                movement_hits += actual == predicted
            else:
                no_change += 1
                false_moves += predicted != 0
        baseline = fuel["baselineMae"]
        assert math.isfinite(baseline) and baseline >= 0
        fuels.append({
            "fuel": fuel["fuel"], "activeFamily": fuel["family"],
            "holdoutCount": len(points), "maeTWD": round(mae, 4),
            "rmseTWD": round(math.sqrt(sum(e*e for e in errors)/len(errors)), 4),
            "baselineMaeTWD": round(baseline, 4),
            "directionAccuracyPct": round(100*direction_hits/len(points), 2),
            "changedWeeks": movement_count,
            "changedPriceRecallPct": round(100*movement_hits/movement_count, 2) if movement_count else None,
            "falseChangeRatePct": round(100*false_moves/no_change, 2) if no_change else None,
            "holdoutBeatsFlatBy0_01TWD": mae+0.01 < baseline,
        })
    result = {
        "algorithm": model["algorithm"], "trainedThrough": model["trainedThrough"],
        "targetDate": model["targetDate"], "holdoutStart": model["testStart"],
        "holdoutEnd": model["testEnd"], "historicalOnly": True,
        "notLiveAccuracy": True, "automaticPromotionApproved": False,
        "note": ("Holdout is only a diagnostic. v6 already uses this period as an activation gate, "
                 "so its holdout is NOT an unbiased final test; v7 selects using earlier validation. "
                 "Check realized, signed cloudAccuracy outcomes before any production switch."),
        "fuels": fuels,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 tests/model_quality_report.py output/model.json")
    report(sys.argv[1])
