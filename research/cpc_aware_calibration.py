"""CPC-aware, non-leaky calibration for next-Monday retail fuel adjustments.

The three unleaded gasoline grades use the same CPC adjustment in normal
weeks. Use a shared learned gasoline delta, not three contradictory prices.
Tune a small, predeclared set of threshold/shrinkage policies on the earlier
validation segment; a disjoint confirmation segment must agree. The final
13-week audit is never used to select a policy.
"""
import numpy as np

POLICIES = ((0.0, 0.0), (0.25, 0.1), (0.5, 0.1), (0.75, 0.1),
            (1.0, 0.1), (0.5, 0.2), (0.75, 0.2), (1.0, 0.2))

def tie_gasoline(prediction):
    p = np.asarray(prediction, dtype=np.float32).copy()
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all():
        raise ValueError("Expected four finite CPC fuel deltas")
    gas = np.mean(p[:, :3], axis=1)
    p[:, :3] = gas[:, None]
    return p

def policy_prediction(teacher, scale, min_change):
    if scale == 0:
        return np.zeros_like(teacher)
    adjusted = np.round(np.asarray(teacher) * scale, 1)
    adjusted[np.abs(adjusted) < min_change - 1e-6] = 0.0
    return adjusted.astype(np.float32)

def indicators(actual, predicted):
    if actual.shape != predicted.shape:
        raise ValueError("Mismatched prediction and CPC label arrays")
    diff = np.asarray(actual) - np.asarray(predicted)
    a = np.where(actual > .049, 1, np.where(actual < -.049, -1, 0))
    p = np.where(predicted > .049, 1, np.where(predicted < -.049, -1, 0))
    moving = a != 0
    stopped = ~moving
    return {
        "mae": np.abs(diff).mean(axis=0).round(6).tolist(),
        "directionAccuracyPct": ((a == p).mean(axis=0) * 100).round(2).tolist(),
        "movementRecallPct": (((a == p) & moving).sum(axis=0)
                              / np.maximum(1, moving.sum(axis=0)) * 100).round(2).tolist(),
        "falseMovePct": (((p != 0) & stopped).sum(axis=0)
                         / np.maximum(1, stopped.sum(axis=0)) * 100).round(2).tolist()
    }

def fit_calibration(dev_labels, dev_teacher, confirm_labels, confirm_teacher):
    """Preselect on development weeks, confirm on NEXT 13 weeks, never audit."""
    dev_labels, confirm_labels = np.asarray(dev_labels), np.asarray(confirm_labels)
    dev_teacher, confirm_teacher = tie_gasoline(dev_teacher), tie_gasoline(confirm_teacher)
    if (dev_labels.shape[1] != 4 or confirm_labels.shape[1] != 4
        or len(dev_labels) < 20 or len(confirm_labels) < 10):
        raise ValueError("Need disjoint chronological validation and confirmation")
    if not np.isfinite(dev_labels).all() or not np.isfinite(confirm_labels).all():
        raise ValueError("Nonfinite target labels")
    chosen = []
    per_group = []
    # 95/98 track 92's ADJUSTMENT amount; calibrate them as one gasoline group.
    for fuels in ([0,1,2], [3]):
        base_dev = indicators(dev_labels, np.zeros_like(dev_labels))
        base_conf = indicators(confirm_labels, np.zeros_like(confirm_labels))
        scores = []
        for i, (alpha, threshold) in enumerate(POLICIES):
            candidate = policy_prediction(dev_teacher, alpha, threshold)
            q = indicators(dev_labels, candidate)
            mae = float(np.mean([q["mae"][f] for f in fuels]))
            baseline_mae = float(np.mean([base_dev["mae"][f] for f in fuels]))
            accuracy = float(np.mean([q["directionAccuracyPct"][f] for f in fuels]))
            baseline_accuracy = float(np.mean([base_dev["directionAccuracyPct"][f] for f in fuels]))
            false = max(q["falseMovePct"][f] for f in fuels)
            # Candidate must improve MAE and not destroy calibration / false alarms.
            if i > 0 and (mae + .012 > baseline_mae
                           or accuracy < baseline_accuracy - 5 or false > 25):
                continue
            scores.append((mae, i))
        best = min(scores)[1]
        alpha, threshold = POLICIES[best]
        confirmed = policy_prediction(confirm_teacher, alpha, threshold)
        check = indicators(confirm_labels, confirmed)
        confirmed_mae = float(np.mean([check["mae"][f] for f in fuels]))
        baseline_conf = float(np.mean([base_conf["mae"][f] for f in fuels]))
        confirmed_false = max(check["falseMovePct"][f] for f in fuels)
        confirmed_accuracy = float(np.mean([check["directionAccuracyPct"][f] for f in fuels]))
        baseline_conf_accuracy = float(np.mean([base_conf["directionAccuracyPct"][f] for f in fuels]))
        # Confirmed only if it independently beats no-adjustment and stays conservative.
        accepted = best > 0 and (confirmed_mae + .01 < baseline_conf
              and confirmed_false <= 25 and confirmed_accuracy >= baseline_conf_accuracy - 5)
        chosen.append(best if accepted else 0)
        per_group.append({
            "fuels": fuels, "chosenOnDevelopment": POLICIES[best],
            "confirmed": bool(accepted),
            "validationMaE": min(scores)[0], "confirmationMaE": confirmed_mae,
            "confirmationBaselineMaE": baseline_conf
        })
    return chosen, per_group

def apply_calibration(raw_teacher, chosen):
    tied = tie_gasoline(raw_teacher)
    if len(chosen) != 2:
        raise ValueError("Gasoline and diesel calibration required")
    gas = policy_prediction(tied, *POLICIES[chosen[0]])
    diesel = policy_prediction(tied, *POLICIES[chosen[1]])
    gas[:,3] = diesel[:,3]
    return gas
