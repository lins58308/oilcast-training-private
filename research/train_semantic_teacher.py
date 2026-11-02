#!/usr/bin/env python3
"""Offline GPU/CPU semantic OilCast teacher; never auto-publishes Android models.

Requires the strictly time-indexed JSON produced by SemanticResearchDataset.
A frozen ~0.3B parameter multilingual E5 text encoder is combined with a
small, regularized sequence forecaster. Train/validation/holdout are strictly
ordered. The 13-week holdout is never used to select/promote a candidate.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np

def seed_all(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

def validate_dataset(data):
    if data.get("schema") != 1 or data.get("featureDimension") != 26 or data.get("fuels") != 4:
        raise ValueError("Unsupported point-in-time teacher dataset")
    rows = data["examples"]
    if len(rows) < 90:
        raise ValueError("Need >=90 chronologically aligned CPC target weeks")
    for i, row in enumerate(rows):
        if i and rows[i - 1]["origin"] >= row["origin"]:
            raise ValueError("Unsorted weekly origins")
        if row["origin"] >= row["targetDate"] or row["cutoff"][:10] > row["origin"]:
            raise ValueError("A weekly origin must precede the actual CPC target")
        if len(row["features"]) != 4 or len(row["nextWeekDeltas"]) != 4:
            raise ValueError("Four fuel types are required")
        if any(len(x) != 26 for x in row["features"]):
            raise ValueError("26 input features per fuel are required")
        for item in row["news"]:
            if item["publishedAt"] > row["cutoff"] or item["availableAt"] > row["cutoff"]:
                raise ValueError("Future or late-ingested news has leaked into training")
    return rows

def encode_all(rows, model_id, revision, device, batch_size):
    import torch
    from transformers import AutoModel, AutoTokenizer
    # Never execute custom remote model code. Model weights must be safetensors.
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    encoder = AutoModel.from_pretrained(model_id, revision=revision,
                                        trust_remote_code=False, use_safetensors=True)
    encoder = encoder.to(device).eval()
    all_news = {}
    for row in rows:
        for article in row["news"]:
            all_news.setdefault(article["id"], article["title"])
    identifiers = sorted(all_news)
    embeddings = {}
    with torch.inference_mode():
        for start in range(0, len(identifiers), batch_size):
            keys = identifiers[start:start + batch_size]
            titles = ["passage: " + all_news[k] for k in keys]
            tokens = tokenizer(titles, max_length=128, truncation=True,
                               padding=True, return_tensors="pt").to(device)
            hidden = encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            for key, vector in zip(keys, pooled.cpu().numpy()):
                embeddings[key] = vector.astype(np.float32)
            if (start // batch_size) % 10 == 0:
                print(json.dumps({"stage": "encode", "done": start + len(keys),
                                  "total": len(identifiers)}), flush=True)
    dim = encoder.config.hidden_size
    checkpoint = getattr(encoder.config, "_commit_hash", None)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embeddings, dim, checkpoint

def weekly_embeddings(rows, vectors, dimension):
    from datetime import datetime, timezone
    aggregated = np.zeros((len(rows), dimension), dtype=np.float32)
    for t, row in enumerate(rows):
        grouped = {}
        cutoff = datetime.fromisoformat(row["cutoff"].replace("Z", "+00:00"))
        for a in row["news"]:
            vec = vectors.get(a["id"])
            if vec is None:
                raise ValueError("Missing headline embedding")
            stamp = datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00"))
            age_days = max(0, (cutoff - stamp).total_seconds() / 86400)
            weight = max(0.25, math.exp(-age_days / 4))
            source = a.get("source", "")
            grouped.setdefault(source, []).append((vec, weight))
        if grouped:
            per_source = []
            for values in grouped.values():
                total = sum(weight for _, weight in values)
                per_source.append(sum(vec * weight for vec, weight in values) / total)
            aggregated[t] = np.mean(per_source, axis=0)
    return aggregated

def train_only_pca(news, end, components=32):
    # PCA bases and all normalizers are fit on the TRAIN fold, never validation/test.
    center = news[:end].mean(axis=0, keepdims=True)
    centered = news[:end] - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(components, len(vt), news.shape[1])
    scores = (news - center) @ vt[:k].T
    if k < components:
        scores = np.pad(scores, ((0, 0), (0, components - k)))
    return scores.astype(np.float32)

def prepare(rows, semantic, sequence=4):
    numeric = np.array([r["features"] for r in rows], dtype=np.float32)
    # CPC's 70% Dubai + 30% Brent benchmark is a major omitted driver.
    # Append only deliberately lagged monthly Dubai proxies, never current
    # full-month values unavailable at that historical prediction origin.
    dubai = np.array([r.get("policyProxy", [0., 0., 0., 0.]) for r in rows], dtype=np.float32)
    if dubai.shape != (len(rows), 4) or not np.isfinite(dubai).all():
        raise ValueError("Invalid lagged Dubai policy proxies")
    numeric = np.concatenate((numeric, np.repeat(dubai[:, None, :], 4, axis=1)), axis=2)
    targets = np.array([r["nextWeekDeltas"] for r in rows], dtype=np.float32)
    if not np.isfinite(numeric).all() or not np.isfinite(targets).all():
        raise ValueError("Nonfinite CPC, market, or news features")
    n = len(rows)
    test_start = n - 13
    val_start = test_start - 39
    if val_start - sequence < 36:
        raise ValueError("Too few weeks for independent chronological train and validation")
    # Numeric normalization is based solely on the training segment.
    mean = numeric[:val_start].mean(axis=0)
    std = numeric[:val_start].std(axis=0)
    normal = np.clip((numeric - mean) / np.maximum(std, .01), -6, 6)
    compressed = train_only_pca(semantic, val_start)
    sem_mean = compressed[:val_start].mean(axis=0)
    sem_std = compressed[:val_start].std(axis=0)
    compressed = np.clip((compressed - sem_mean) / np.maximum(sem_std, .05), -6, 6)
    features = np.concatenate((normal.reshape(n, -1), compressed), axis=1)
    # Example t gets t-3..t, never input data from t+1.
    x = np.stack([features[i-sequence+1:i+1] for i in range(sequence-1, n)])
    y = targets[sequence-1:]
    return x, y, val_start - sequence + 1, test_start - sequence + 1, normal.shape[-1]

def metrics(actual, predicted):
    err = np.abs(actual - predicted)
    direction = lambda z: np.where(z > .049, 1, np.where(z < -.049, -1, 0))
    a, p = direction(actual), direction(predicted)
    moving = a != 0
    still = ~moving
    return {
        "mae": np.round(err.mean(0), 5).tolist(),
        "rmse": np.round(np.sqrt(np.mean((actual - predicted) ** 2, axis=0)), 5).tolist(),
        "directionAccuracyPct": np.round((a == p).mean(0) * 100, 2).tolist(),
        "movingWeekRecallPct": np.round(((a == p) & moving).sum(0) / np.maximum(1, moving.sum(0)) * 100, 2).tolist(),
        "falseMovementRatePct": np.round(((p != 0) & still).sum(0) / np.maximum(1, still.sum(0)) * 100, 2).tolist(),
        "movingWeeks": moving.sum(0).tolist(),
        "totalWeeks": len(actual)
    }

def fit_ridge_chronologically(x, y, train_end, tune_end):
    """Simple small-data challenger; lambda chosen on earlier-only stop period."""
    design = x[:, -1, :].astype(np.float64)
    mean = design[:train_end].mean(axis=0)
    scale = np.maximum(design[:train_end].std(axis=0), .1)
    z = np.clip((design - mean) / scale, -6, 6)
    z = np.concatenate([np.ones((len(z), 1)), z], axis=1)
    # Give all 27 years a nonzero vote while accommodating evolving CPC
    # stabilization and tax policy; never discard early years.
    age = train_end - 1 - np.arange(train_end)
    recency = np.maximum(.5, np.exp(-age / 156.0))
    gram = z[:train_end].T @ (recency[:, None] * z[:train_end])
    rhs = z[:train_end].T @ (recency[:, None] * y[:train_end].astype(np.float64))
    penalize = np.eye(gram.shape[0])
    penalize[0, 0] = 0
    best_mae, best_lambda, best_prediction = float("inf"), None, None
    for lam in (10, 100, 1000, 10000):
        coef = np.linalg.solve(gram + lam * penalize, rhs)
        pred = (z @ coef).astype(np.float32)
        mae = float(np.abs(pred[train_end:tune_end] - y[train_end:tune_end]).mean())
        if mae < best_mae:
            best_mae, best_lambda, best_prediction = mae, lam, pred
    return best_prediction, best_lambda, best_mae

def pool_gasoline_adjustments(predictions):
    """CPC generally adjusts 92/95/98 by the same TWD/L amount.

    A physics/price-policy challenger, not an unconditional rule for early
    historical weeks or for unverified announced prices.
    """
    out = np.asarray(predictions, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] != 4:
        raise ValueError("Four gasoline/diesel outputs required")
    out[:, :3] = np.mean(out[:, :3], axis=1, keepdims=True)
    return out

def select_conservative_forecaster(actual_validation, raw_candidates):
    """Select per-fuel abstention and regularization on validation, NEVER locked test.

    No-change baseline is always allowed. Reject false-change-heavy candidates
    even when they appear to lower total absolute error on a tiny period.
    """
    actual = np.asarray(actual_validation, dtype=np.float32)
    baseline = metrics(actual, np.zeros_like(actual))
    all_predictions = {"unchanged": np.zeros_like(actual)}
    selections = []
    options = [{"model": "unchanged", "scale": 0.0, "deadZone": 0.0}
               for _ in range(4)]
    for name, raw in raw_candidates.items():
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != actual.shape or not np.isfinite(raw).all():
            raise ValueError("Invalid predicted delta shape in validation")
        all_predictions[name] = raw
    for fuel in range(4):
        base_mae = float(baseline["mae"][fuel])
        best_mae = base_mae
        selected = options[fuel]
        for name, predictions in all_predictions.items():
            if name == "unchanged":
                continue
            for scale in (0.15, 0.30, 0.50, 0.75, 1.0):
                for cutoff in (0.05, 0.10, 0.20, 0.30):
                    prediction = predictions[:, fuel] * scale
                    prediction = np.where(np.abs(prediction) < cutoff, 0.0, prediction)
                    m = metrics(actual[:, [fuel]], prediction.reshape(-1, 1))
                    mae = float(m["mae"][0])
                    if (mae + 0.01 < best_mae
                        and m["directionAccuracyPct"][0] >=
                            baseline["directionAccuracyPct"][fuel] - 5
                        and m["falseMovementRatePct"][0] <= 25
                        and m["movingWeekRecallPct"][0] > 0):
                        best_mae = mae
                        selected = {"model": name, "scale": scale, "deadZone": cutoff}
        selections.append({"fuel": fuel, **selected,
                           "validationMae": round(best_mae, 5),
                           "baselineValidationMae": base_mae,
                           "approved": selected["model"] != "unchanged"})
    return selections

def apply_selections(raw_candidates, selections, length):
    out = np.zeros((length, 4), dtype=np.float32)
    for entry in selections:
        fuel = entry["fuel"]
        if entry["model"] == "unchanged":
            continue
        raw = np.asarray(raw_candidates[entry["model"]])[:, fuel]
        delta = raw * entry["scale"]
        out[:, fuel] = np.where(np.abs(delta) < entry["deadZone"], 0, delta)
    return out

def make_model(variant, features, torch):
    nn = torch.nn
    class MlpTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(nn.Flatten(), nn.Linear(features*4, 128),
                nn.GELU(), nn.Dropout(.2), nn.Linear(128, 64),
                nn.GELU(), nn.Dropout(.15), nn.Linear(64, 4))
        def forward(self, x):
            return self.network(x)
    class TemporalTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(features, 96)
            self.positions = nn.Parameter(torch.zeros(1, 4, 96))
            layer = nn.TransformerEncoderLayer(96, 4, 192, dropout=.15,
                                               batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=2,
                                                enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 48),
                                      nn.GELU(), nn.Dropout(.15), nn.Linear(48, 4))
        def forward(self, x):
            return self.head(self.encoder(self.project(x) + self.positions)[:, -1])
    return MlpTeacher() if variant == "v6" else TemporalTeacher()

def fit_and_report(data, args):
    import torch
    from safetensors.torch import save_file
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA training but no GPU was detected")
    seed_all(args.seed, torch)
    rows = validate_dataset(data)
    started = time.monotonic()
    vectors, dim, commit = encode_all(rows, args.encoder, args.encoder_revision, device, args.batch_size)
    print(json.dumps({"stage": "encoder_ready", "paramsClass": "frozen pretrained multilingual model",
                      "embeddingDimension": dim, "checkpoint": commit, "device": str(device)}), flush=True)
    semantic = weekly_embeddings(rows, vectors, dim)
    x, y, val_start, test_start, _ = prepare(rows, semantic)
    # Strict chronological nested split: core fit / early stopping / selection
    # / final 13-week blind audit. Never reuse early-stop weeks to select
    # the model family or output threshold.
    stop_start = val_start - 52
    if stop_start < 60:
        raise ValueError("Need at least 60 training weeks before 52-week stopping window")
    ridge_prediction, ridge_lambda, ridge_stop_mae = fit_ridge_chronologically(
        x, y, stop_start, val_start)
    tensors = torch.from_numpy(x).to(device)
    expected = torch.from_numpy(y).to(device)
    model = make_model(args.variant, x.shape[-1], torch).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.02)
    best_state, best_loss, stale, chosen = None, float("inf"), 0, 0
    curves = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(stop_start, device=device)
        losses = []
        for indices in order.split(args.batch_size):
            prediction = model(tensors[indices])
            label = expected[indices]
            smooth = torch.nn.functional.smooth_l1_loss(prediction, label, reduction="none", beta=.25)
            # Lower aggressive 'changed week' oversampling; reduce false alarms.
            weight = torch.where(label.abs() >= .049, 1.15, 1.0)
            age = (stop_start - 1 - indices).to(torch.float32)
            historical_weight = torch.maximum(torch.full_like(age, .5), torch.exp(-age / 156.0))
            loss = (smooth * weight * historical_weight.unsqueeze(-1)).sum() / (
                historical_weight.sum() * label.shape[-1])
            if not torch.isfinite(loss):
                raise ValueError("Numerical instability in deep model")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            valid_pred = model(tensors[stop_start:val_start]).cpu().numpy()
        valid_loss = float(np.abs(valid_pred - y[stop_start:val_start]).mean())
        curves.append({"epoch": epoch, "trainLoss": float(np.mean(losses)),
                       "validationMae": valid_loss})
        if valid_loss < best_loss - 1e-5:
            best_loss, chosen, stale = valid_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"stage": "train", **curves[-1]}), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        val_prediction = model(tensors[val_start:test_start]).cpu().numpy()
        held_prediction = model(tensors[test_start:]).cpu().numpy()
    val_baseline = metrics(y[val_start:test_start], np.zeros_like(y[val_start:test_start]))
    val_teacher = metrics(y[val_start:test_start], val_prediction)
    val_ridge = metrics(y[val_start:test_start], ridge_prediction[val_start:test_start])
    held_baseline = metrics(y[test_start:], np.zeros_like(y[test_start:]))
    held_teacher = metrics(y[test_start:], held_prediction)
    held_ridge = metrics(y[test_start:], ridge_prediction[test_start:])
    # Compare the legal gasoline-equal-adjustment candidate as well as
    # independent fuel heads; selection uses only the earlier 39 weeks.
    val_candidates = {
        "deep": val_prediction,
        "deepGasolineShared": pool_gasoline_adjustments(val_prediction),
        "ridge": ridge_prediction[val_start:test_start],
        "ridgeGasolineShared": pool_gasoline_adjustments(ridge_prediction[val_start:test_start]),
    }
    held_candidates = {
        "deep": held_prediction,
        "deepGasolineShared": pool_gasoline_adjustments(held_prediction),
        "ridge": ridge_prediction[test_start:],
        "ridgeGasolineShared": pool_gasoline_adjustments(ridge_prediction[test_start:]),
    }
    choices = select_conservative_forecaster(y[val_start:test_start], val_candidates)
    calibrated_validation = apply_selections(val_candidates, choices, test_start - val_start)
    calibrated_heldout = apply_selections(held_candidates, choices, len(y) - test_start)
    val_selected = metrics(y[val_start:test_start], calibrated_validation)
    held_selected = metrics(y[test_start:], calibrated_heldout)
    eligible = [choice["approved"] for choice in choices]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in best_state.items()}, str(output / "semantic-teacher.safetensors"))
    report = {
        "schema": 1, "variant": args.variant, "status": "RESEARCH_ONLY_NOT_ANDROID_COMPATIBLE",
        "pretrainedEncoder": args.encoder, "encoderRevision": commit,
        "encoderFrozen": True, "device": str(device),
        "weeklyExamples": len(rows), "uniqueNews": len(vectors),
        "numericalFeaturesPerFuel": 30, "originalAndroidFeatureDimension": 26,
        "dubaiProxy": data.get("policyProxyProvenance", {"missing": True}),
        "dubaiWeeks": int(sum(r.get("policyProxy", [0., 0., 0., 0.])[-1] > 0 for r in rows)),
        "newsPcaDimensions": 32,
        "temporalWeeks": 4, "predicts": "next-Monday CPC delta for each of 4 fuels",
        "forecasterParameters": params, "bestEpoch": chosen,
        "lastEpoch": len(curves), "elapsedSeconds": round(time.monotonic() - started, 2),
        "split": {"trainCoreWeeks": stop_start, "earlyStopWeeks": val_start-stop_start,
                  "selectionWeeks": test_start-val_start, "blindHoldoutWeeks": len(y)-test_start},
        "historicalSampleWeight": "max(0.5, exp(-weeklyAge/156)); all 27 years used",
        "modelSelection": choices,
        "simpleRidge": {"chosenLambdaOnEarlyStop": ridge_lambda,
                        "earlyStopMae": round(ridge_stop_mae, 5)},
        "validation": {"baseline": val_baseline, "deepRaw": val_teacher,
                       "ridge": val_ridge, "selectedCalibrated": val_selected},
        "lockedHoldout13Weeks": {"baseline": held_baseline, "deepRaw": held_teacher,
                                "ridge": held_ridge, "selectedCalibrated": held_selected},
        "validationApprovedByFuel": eligible,
        "automaticProductionPromotion": False,
        "timingWarning": "Historical news uses its first known availability; older RSS cannot be backdated.",
        "dataSHA256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (output / "training-curves.json").write_text(json.dumps(curves, indent=2))
    # Compact metrics in immutable GitHub logs, so results can be inspected
    # without downloading a binary artifact through a separate workflow.
    print(json.dumps({"stage": "audit", "variant": args.variant,
                      "chronologicalSplit": report["split"],
                      "selectedExperts": choices,
                      "validationBaselineMae": val_baseline["mae"],
                      "validationCalibratedMae": val_selected["mae"],
                      "lockedBaselineMae": held_baseline["mae"],
                      "lockedDeepRawMae": held_teacher["mae"],
                      "lockedRidgeMae": held_ridge["mae"],
                      "lockedCalibratedMae": held_selected["mae"],
                      "lockedChangedWeekRecall": held_selected["movingWeekRecallPct"],
                      "lockedFalseMoveRate": held_selected["falseMovementRatePct"]}), flush=True)
    print(json.dumps({"stage": "complete", "variant": args.variant,
                      "eligible": eligible, "report": str(output / "report.json"),
                      "elapsedSeconds": report["elapsedSeconds"]}), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="output/semantic")
    parser.add_argument("--variant", choices=("v6", "v7"), required=True)
    parser.add_argument("--encoder", default="intfloat/multilingual-e5-base")
    parser.add_argument("--encoder-revision", default="main")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    if not (1 <= args.epochs <= 300 and 1 <= args.batch_size <= 128
            and 2 <= args.patience <= 50 and 0 < args.lr <= .01):
        parser.error("Unsafe training hyperparameters")
    fit_and_report(json.loads(Path(args.dataset).read_text()), args)

if __name__ == "__main__":
    main()
