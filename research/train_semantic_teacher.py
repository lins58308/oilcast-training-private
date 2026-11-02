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
    tensors = torch.from_numpy(x).to(device)
    expected = torch.from_numpy(y).to(device)
    model = make_model(args.variant, x.shape[-1], torch).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.02)
    best_state, best_loss, stale, chosen = None, float("inf"), 0, 0
    curves = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(val_start, device=device)
        losses = []
        for indices in order.split(args.batch_size):
            prediction = model(tensors[indices])
            label = expected[indices]
            smooth = torch.nn.functional.smooth_l1_loss(prediction, label, reduction="none", beta=.25)
            weight = torch.where(label.abs() >= .049, 2.0, 1.0)
            loss = (smooth * weight).mean()
            if not torch.isfinite(loss):
                raise ValueError("Numerical instability in deep model")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            valid_pred = model(tensors[val_start:test_start]).cpu().numpy()
        valid_loss = float(np.abs(valid_pred - y[val_start:test_start]).mean())
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
    held_baseline = metrics(y[test_start:], np.zeros_like(y[test_start:]))
    held_teacher = metrics(y[test_start:], held_prediction)
    # Promotions are per fuel, and use VALIDATION only. Holdout stays blind.
    eligible = []
    for f in range(4):
        eligible.append(bool(
            val_teacher["mae"][f] + .01 < val_baseline["mae"][f]
            and val_teacher["directionAccuracyPct"][f] >= val_baseline["directionAccuracyPct"][f] - 5
            and val_teacher["falseMovementRatePct"][f] <= 25))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in best_state.items()}, str(output / "semantic-teacher.safetensors"))
    report = {
        "schema": 1, "variant": args.variant, "status": "RESEARCH_ONLY_NOT_ANDROID_COMPATIBLE",
        "pretrainedEncoder": args.encoder, "encoderRevision": commit,
        "encoderFrozen": True, "device": str(device),
        "weeklyExamples": len(rows), "uniqueNews": len(vectors),
        "numericalFeaturesPerFuel": 26, "newsPcaDimensions": 32,
        "temporalWeeks": 4, "predicts": "next-Monday CPC delta for each of 4 fuels",
        "forecasterParameters": params, "bestEpoch": chosen,
        "lastEpoch": len(curves), "elapsedSeconds": round(time.monotonic() - started, 2),
        "validation": {"baseline": val_baseline, "teacher": val_teacher},
        "lockedHoldout13Weeks": {"baseline": held_baseline, "teacher": held_teacher},
        "validationApprovedByFuel": eligible,
        "automaticProductionPromotion": False,
        "timingWarning": "Historical news uses its first known availability; older RSS cannot be backdated.",
        "dataSHA256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (output / "training-curves.json").write_text(json.dumps(curves, indent=2))
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
