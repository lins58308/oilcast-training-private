#!/usr/bin/env python3
"""Frozen, per-fuel MAE experiment. No publisher or Android schema changes.

Choose models on five earlier dated folds. Write the choice to disk BEFORE
calculating the already-inspected 2026 audit. All estimators see the full
available training history; recency weights have a 0.5 floor. Absolute-error
learners are compared with a squared-error ridge reference and no change.
"""
import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.preprocessing import StandardScaler

from official_daily_oil import parse_moea_daily, point_in_time_weekly
from policy_features import feature_for_period

FUELS = ("92", "95", "98", "diesel")
GROUPS = ((0, 1, 2), (3,))
SEED = 20261103
MIN_IMPROVEMENT = 0.01
AUDIT_WEEKS = 13


@dataclass(frozen=True)
class Spec:
    name: str
    learner: str
    inputs: str
    complexity: float


# Predeclared once for both source versions; no audit-specific thresholds.
SPECS = (
    Spec("ridge-market", "ridge", "market", 30.0),
    Spec("linear-mae-market", "linear-mae", "market", 0.01),
    Spec("linear-mae-policy", "linear-mae", "policy", 0.01),
    Spec("hist-mae-market", "hist-mae", "market", 7),
    Spec("hist-mae-policy", "hist-mae", "policy", 7),
    Spec("hist-mae-all", "hist-mae", "all", 7),
    Spec("forest-mae-market", "forest-mae", "market", 6),
    Spec("forest-mae-policy", "forest-mae", "policy", 6),
)


def json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def direction(values):
    z = np.asarray(values)
    return np.where(z > .049, 1, np.where(z < -.049, -1, 0))


def round_prices(delta, base):
    """Use the signed Java price rounding, including negative-half behavior."""
    d, b = np.asarray(delta), np.asarray(base)
    if d.shape != b.shape or not np.isfinite(d).all() or not np.isfinite(b).all():
        raise ValueError("Prediction and base must be finite matching arrays")
    price = np.clip(b + np.clip(d, -5, 5), 5, 100)
    return np.floor(price * 10 + .5) / 10 - b


def metrics(actual, predicted):
    a, p = np.asarray(actual), np.asarray(predicted)
    if a.shape != p.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Invalid scored observations")
    ad, pd = direction(a), direction(p)
    moved = ad != 0
    flat = ~moved
    return {
        "mae": float(np.abs(a-p).mean()),
        "rmse": float(np.sqrt(np.mean((a-p)**2))),
        "directionAccuracy": float(np.mean(ad == pd)),
        "changedWeekRecall": float(np.sum((ad == pd) & moved) / moved.sum()) if moved.any() else None,
        "falseMoveRate": float(np.sum((pd != 0) & flat) / flat.sum()) if flat.any() else None,
        "changedWeeks": int(moved.sum()),
        "falseMoves": int(np.sum((pd != 0) & flat)),
        "weeks": int(a.size),
    }


def load_frozen(root, variant):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["variant"] != variant:
        raise ValueError("Wrong version's frozen input artifact")
    for name, expected in manifest["sha256"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe frozen input filename")
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Changed frozen input: " + name)
    data = json.loads((root / "dataset.json").read_text())
    if data.get("schema") != 1 or data.get("featureDimension") != 26:
        raise ValueError("Unexpected input schema")
    rows = data["examples"]
    if len(rows) < 1300:
        raise ValueError("Full dated 1999-present history required")
    numeric = np.asarray([r["features"] for r in rows], dtype=float)
    y = np.asarray([r["nextWeekDeltas"] for r in rows], dtype=float)
    base = np.asarray([r["basePrices"] for r in rows], dtype=float)
    if numeric.shape != (len(rows), 4, 26) or y.shape != base.shape or y.shape != (len(rows), 4):
        raise ValueError("Malformed four-fuel data")
    if not all(np.isfinite(a).all() for a in (numeric, y, base)):
        raise ValueError("Nonfinite observations")
    taipei = timezone(timedelta(hours=8))
    for i, row in enumerate(rows):
        origin = date.fromisoformat(row["origin"])
        target = date.fromisoformat(row["targetDate"])
        cutoff = datetime.fromisoformat(row["cutoff"].replace("Z", "+00:00"))
        if origin.weekday() != 0 or target != origin + timedelta(days=7):
            raise ValueError("Not consecutive weekly CPC targets")
        if cutoff != datetime.combine(origin, datetime.min.time(), taipei):
            raise ValueError("Invalid Monday Taipei cutoff")
        if i and (row["origin"] != rows[i-1]["targetDate"] or not np.allclose(base[i], base[i-1]+y[i-1], atol=1e-7)):
            raise ValueError("Missing or inconsistent official price week")
        for item in row["news"]:
            if any(datetime.fromisoformat(item[key].replace("Z", "+00:00")) > cutoff
                   for key in ("publishedAt", "availableAt")):
                raise ValueError("Future headline in historical origin")
    daily = parse_moea_daily((root / "moea-daily.csv").read_bytes())
    oil, oil_count = point_in_time_weekly(rows, daily)
    policy, policy_count = feature_for_period(rows, json.loads((root / "policy.json").read_text()))
    return rows, numeric, y, base, np.asarray(oil), np.asarray(policy), manifest, {
        "dailyQuoteWeeks": oil_count, "policyWeeks": policy_count,
        "newsWeeks": data["weeksWithNews"], "weeklyExamples": len(rows)}


def matrices(numeric, base, oil, policy, variant):
    result = {}
    for head, f in enumerate((0, 3)):
        other = 3 if f == 0 else 0
        # Current-origin price changes are already official, whereas the
        # target label is next Monday. Describe past freeze duration and
        # volatility explicitly rather than infer them from a 12-week mean.
        regime = []
        for i in range(len(numeric)):
            history = numeric[max(0, i-51):i+1, f, 0]
            streak = 0
            for value in history[::-1]:
                if abs(value) >= .049:
                    break
                streak += 1
            row = [min(streak, 52)/52., min(1., (i+1)/26.)]
            for window in (4, 13, 26):
                seen = history[-window:]
                row.extend([float(np.mean(np.abs(seen) >= .049)),
                            float(np.mean(seen)), float(np.mean(np.abs(seen)))])
            regime.append(row)
        known = np.column_stack((numeric[:, f, :18], numeric[:, other, :10], base, regime))
        result[head] = {
            "market": known,
            "policy": np.column_stack((known, oil, policy[:, f])),
            "all": np.column_stack((known, numeric[:, f, 18:], oil, policy[:, f])),
        }
        if variant == "v7":
            # Temporal context contains only already observed origins. No
            # backfilling earlier dates with future news or policy values.
            for name, x in list(result[head].items()):
                lagged = []
                for lag in (1, 2, 4):
                    z = np.zeros((len(x), 6))
                    z[lag:] = numeric[:-lag, f, :6]
                    lagged.append(z)
                result[head][name] = np.column_stack((x, *lagged))
    return result


def fit_predict(x, labels, start, end, spec):
    if not 100 <= start < end <= len(x) or len(labels) != len(x):
        raise ValueError("Training must end before the forecast block")
    # All older examples retain at least half of the newest base weight.
    weights = .5 + .5*np.exp(-np.arange(start)[::-1]/104.)
    source, query = x[:start], x[start:end]
    if spec.learner in ("ridge", "linear-mae"):
        scaler = StandardScaler().fit(source, sample_weight=weights)
        source, query = scaler.transform(source), scaler.transform(query)
    if spec.learner == "ridge":
        model = Ridge(alpha=spec.complexity)
    elif spec.learner == "linear-mae":
        model = QuantileRegressor(quantile=.5, alpha=spec.complexity, solver="highs")
    elif spec.learner == "hist-mae":
        model = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=100, learning_rate=.05,
            max_leaf_nodes=int(spec.complexity), min_samples_leaf=26,
            l2_regularization=10, early_stopping=False, random_state=SEED)
    elif spec.learner == "forest-mae":
        model = ExtraTreesRegressor(
            criterion="absolute_error", n_estimators=48,
            max_depth=int(spec.complexity), min_samples_leaf=20,
            max_features=.8, n_jobs=2, random_state=SEED)
    else:
        raise ValueError("Unregistered learner")
    model.fit(source, labels[:start], sample_weight=weights)
    if spec.learner == "forest-mae":
        # Median of the learned conditional leaf medians targets MAE.
        return np.median([tree.predict(query) for tree in model.estimators_], axis=0)
    return model.predict(query)


def fold_ranges(n):
    audit = n - AUDIT_WEEKS
    folds = [(audit-gap-13, audit-gap) for gap in (260, 208, 156, 104, 52)]
    if folds[0][0] < 520:
        raise ValueError("Insufficient history for the registered folds")
    return folds, audit


def choose(history, baselines):
    """Independent per-fuel gates; gasoline must qualify for ALL three grades."""
    decisions = []
    for group in GROUPS:
        eligible = []
        diagnostics = {}
        for spec in SPECS:
            scores = history[spec.name]
            per_fuel = []
            detail = {}
            for f in group:
                mae = np.mean([r[f]["mae"] for r in scores])
                bmae = np.mean([r[f]["mae"] for r in baselines])
                wins = sum(r[f]["mae"] + MIN_IMPROVEMENT < b[f]["mae"] - 1e-10
                           for r, b in zip(scores, baselines))
                direction_ok = np.mean([r[f]["directionAccuracy"] for r in scores]) >= np.mean([r[f]["directionAccuracy"] for r in baselines]) - .05
                false_values = [r[f]["falseMoveRate"] for r in scores if r[f]["falseMoveRate"] is not None]
                recall_values = [r[f]["changedWeekRecall"] for r in scores if r[f]["changedWeekRecall"] is not None]
                passed = (mae+MIN_IMPROVEMENT < bmae-1e-10 and wins >= 3 and direction_ok
                          and (not false_values or np.mean(false_values) <= .25)
                          and bool(recall_values) and np.mean(recall_values) > 0)
                per_fuel.append(bool(passed))
                detail[FUELS[f]] = {
                    "meanMae": float(mae), "baselineMeanMae": float(bmae),
                    "winningFolds": int(wins),
                    "maeGate": bool(mae+MIN_IMPROVEMENT < bmae-1e-10),
                    "consistencyGate": bool(wins >= 3), "directionGate": bool(direction_ok),
                    "falseMoveRate": float(np.mean(false_values)) if false_values else None,
                    "falseMoveGate": bool(not false_values or np.mean(false_values) <= .25),
                    "changedWeekRecall": float(np.mean(recall_values)) if recall_values else None,
                    "passed": bool(passed),
                }
            diagnostics[spec.name] = detail
            if all(per_fuel):
                eligible.append((np.mean([r[f]["mae"] for r in scores for f in group]), spec.name))
        eligible.sort()
        decisions.append({"fuels": [FUELS[f] for f in group],
                          "selected": eligible[0][1] if eligible else "unchanged",
                          "eligibleCandidates": [name for _, name in eligible],
                          "candidateGates": diagnostics})
    return decisions


def evaluate(root, variant, output):
    rows, num, y, base, oil, policy, manifest, coverage = load_frozen(root, variant)
    x = matrices(num, base, oil, policy, variant)
    targets = np.column_stack((np.median(y[:, :3], axis=1), y[:, 3]))
    folds, audit = fold_ranges(len(y))
    baselines = [[metrics(y[a:b, f], np.zeros(b-a)) for f in range(4)] for a, b in folds]
    history = {}
    for spec in SPECS:
        history[spec.name] = []
        for a, b in folds:
            pred = np.zeros((b-a, 4))
            for head, members in enumerate(GROUPS):
                p = fit_predict(x[head][spec.inputs], targets[:, head], a, b, spec)
                for f in members:
                    pred[:, f] = p
            pred = round_prices(pred, base[a:b])
            history[spec.name].append([metrics(y[a:b, f], pred[:, f]) for f in range(4)])
        print(json.dumps({"stage": "pre-audit", "variant": variant, "candidate": spec.name,
              "mae": [round(np.mean([r[f]["mae"] for r in history[spec.name]]), 6) for f in range(4)]}), flush=True)
    decisions = choose(history, baselines)
    selection = {
        "schema": 1, "variant": variant, "candidateSpecs": [asdict(s) for s in SPECS],
        "minimumMaeImprovement": MIN_IMPROVEMENT, "minimumWinningFolds": 3,
        "selectionFolds": [{"start": rows[a]["targetDate"], "end": rows[b-1]["targetDate"]} for a, b in folds],
        "sourceManifest": manifest, "coverage": coverage,
        "preAuditBaseline": baselines, "preAuditCandidates": history,
        "selectedBeforeAudit": decisions, "auditLabelsUsedForSelection": False,
    }
    out = Path(output)
    json_write(out / "selection.json", selection)
    selection_hash = hashlib.sha256((out / "selection.json").read_bytes()).hexdigest()
    # Audit calculation begins only after persisting the frozen decision.
    prediction = np.zeros((len(y)-audit, 4))
    for head, members in enumerate(GROUPS):
        name = decisions[head]["selected"]
        if name == "unchanged":
            continue
        spec = next(s for s in SPECS if s.name == name)
        p = fit_predict(x[head][spec.inputs], targets[:, head], audit, len(y), spec)
        for f in members:
            prediction[:, f] = p
    prediction = round_prices(prediction, base[audit:])
    audit_metrics = {}
    for f, name in enumerate(FUELS):
        bm = metrics(y[audit:, f], np.zeros(len(y)-audit))
        cm = metrics(y[audit:, f], prediction[:, f])
        audit_metrics[name] = {"baseline": bm, "selected": cm,
            "beatsBaselineBy0_01": bool(cm["mae"] + MIN_IMPROVEMENT < bm["mae"]-1e-10)}
    report = {
        **selection, "selectionSha256": selection_hash,
        "auditPreviouslyInspected": True,
        "audit": {"start": rows[audit]["targetDate"], "end": rows[-1]["targetDate"], "fuels": audit_metrics},
        "allFourHistoricalAuditGatesPassed": all(m["beatsBaselineBy0_01"] for m in audit_metrics.values()),
        "automaticPromotionApproved": False,
        "deploymentBlockers": ["Previously inspected historical audit is not a prospective trial",
                               "Research estimators require compatible Android inference and device validation"],
    }
    json_write(out / "report.json", report)
    json_write(out / "audit-predictions.json", [{
        "origin": rows[i]["origin"], "targetDate": rows[i]["targetDate"],
        "actualDeltas": y[i].tolist(), "predictedDeltas": prediction[i-audit].tolist()
    } for i in range(audit, len(y))])
    print(json.dumps({"stage": "complete", "variant": variant,
                      "decisions": [{k: v for k, v in d.items() if k != "candidateGates"} for d in decisions],
                      "audit": audit_metrics, "allFourPassed": report["allFourHistoricalAuditGatesPassed"]}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--variant", choices=("v6", "v7"), required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    evaluate(a.inputs, a.variant, a.output)


if __name__ == "__main__":
    main()
