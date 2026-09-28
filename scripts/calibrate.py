"""Calibrate the model's confidence and choose when the app should say "not sure".

Why: the raw model was over-confident (average confidence 69% while right 48% of the time).
For a child, a confident wrong answer ("that's a sad face!" when they were smiling) is worse
than an honest "I'm not sure". This script

  1. fits a single temperature T on the *validation* split (minimising log-loss),
  2. picks the "not sure" threshold on validation: the lowest confidence at which answered
     predictions are right at least --target of the time,
  3. reports calibration (ECE) and selective accuracy on the untouched *test* split,
  4. saves T and the threshold into the model file, plus plots in reports/.

Usage:  python scripts/calibrate.py --model models/emotion_model.npz --data data/split
        python scripts/calibrate.py --model models/emotion_model.keras --data data/split
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from emotion_model import NumpyModel  # noqa: E402
from train_numpy import LABELS, load_split, to_rgb  # noqa: E402


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def probs(logits, t):  # logits: (views, N, C) -> mean over views
    return softmax(logits / t).mean(0)


def nll(p, y):
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def ece(p, y, n_bins=10):
    conf, correct = p.max(1), p.argmax(1) == y
    total = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        k = (conf > lo) & (conf <= hi)
        if k.any():
            total += k.mean() * abs(correct[k].mean() - conf[k].mean())
    return float(total)


def selective(p, y, thr):
    keep = p.max(1) >= thr
    return float(keep.mean()), float((p.argmax(1) == y)[keep].mean()) if keep.any() else float("nan")


def model_logits(model, grays, batch=32):
    return np.concatenate([model.logits(to_rgb(grays[i:i + batch])) for i in range(0, len(grays), batch)], axis=1)


def keras_logits(path, grays, batch=32):
    """A softmax model's log-probabilities are valid logits up to an additive constant."""
    from tensorflow import keras
    model = keras.models.load_model(path, compile=False)
    chunks = []
    for i in range(0, len(grays), batch):
        p = np.asarray(model(to_rgb(grays[i:i + batch]), training=False))
        chunks.append(np.log(np.clip(p, 1e-8, 1.0)))
    return np.concatenate(chunks, axis=0)[None, ...]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/emotion_model.npz")
    ap.add_argument("--data", default="data/split")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--target", type=float, default=0.60, help="wanted accuracy on answered predictions")
    args = ap.parse_args()
    data, reports = Path(args.data), Path(args.reports)

    model_path = Path(args.model)
    xva, yva, _, _ = load_split(data / "val")
    xte, yte, _, _ = load_split(data / "test")
    print(f"Computing logits for {len(yva)} val and {len(yte)} test images…")
    if model_path.suffix == ".npz":
        model = NumpyModel(model_path)
        lva, lte = model_logits(model, xva), model_logits(model, xte)
    else:
        lva, lte = keras_logits(model_path, xva), keras_logits(model_path, xte)

    temps = np.round(np.arange(0.5, 5.01, 0.05), 2)
    t_best = float(min(temps, key=lambda t: nll(probs(lva, t), yva)))
    pva, pte, pte_raw = probs(lva, t_best), probs(lte, t_best), probs(lte, 1.0)

    thresholds = np.round(np.arange(0.20, 0.901, 0.05), 2)
    ok = [t for t in thresholds if selective(pva, yva, t)[1] >= args.target]
    thr = float(ok[0]) if ok else 0.0
    cov, acc = selective(pte, yte, thr)

    result = {
        "temperature": t_best,
        "unsure_below": thr,
        "target_answered_accuracy": args.target,
        "test": {
            "accuracy_all": round(float((pte.argmax(1) == yte).mean()), 4),
            "ece_before": round(ece(pte_raw, yte), 4), "ece_after": round(ece(pte, yte), 4),
            "nll_before": round(nll(pte_raw, yte), 4), "nll_after": round(nll(pte, yte), 4),
            "mean_confidence_before": round(float(pte_raw.max(1).mean()), 4),
            "mean_confidence_after": round(float(pte.max(1).mean()), 4),
            "answers_given": round(cov, 4), "accuracy_when_answering": round(acc, 4),
        },
        "selective_curve_test": [{"threshold": float(t), "coverage": round(selective(pte, yte, t)[0], 4),
                                  "accuracy": round(selective(pte, yte, t)[1], 4)} for t in thresholds],
    }
    print(json.dumps({k: v for k, v in result.items() if k != "selective_curve_test"}, indent=2))
    (reports / "calibration.json").write_text(json.dumps(result, indent=2))
    plot(pte_raw, pte, yte, thresholds, thr, reports / "calibration.png")

    # Save temperature + threshold beside/in the model, depending on its format.
    if model_path.suffix == ".npz":
        d = dict(np.load(model_path))
        meta = json.loads(str(d["meta"]))
        meta.update(temperature=t_best, unsure_below=thr)
        d["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(model_path, **d)
    else:
        labels_path = model_path.with_name("labels.json")
        meta = json.loads(labels_path.read_text()) if labels_path.exists() else {"labels": LABELS, "image_size": 224}
        meta.update(temperature=t_best, unsure_below=thr)
        labels_path.write_text(json.dumps(meta, indent=2))
    print(f"Updated {args.model}: temperature={t_best}, unsure_below={thr}")


def plot(p_raw, p_cal, y, thresholds, thr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    bins = np.linspace(0, 1, 11)
    for p, name, col in [(p_raw, "before (raw)", "#c9855b"), (p_cal, "after temperature scaling", "#2f7d6d")]:
        conf, correct = p.max(1), p.argmax(1) == y
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            k = (conf > lo) & (conf <= hi)
            if k.sum() >= 5:
                xs.append(conf[k].mean()); ys.append(correct[k].mean())
        a.plot(xs, ys, "o-", color=col, label=f"{name}, ECE {ece(p, y):.2f}")
    a.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfectly calibrated")
    a.set_xlabel("How sure the model says it is"); a.set_ylabel("How often it is actually right")
    a.set_title("Reliability (test set)"); a.legend(fontsize=8); a.set_xlim(0, 1); a.set_ylim(0, 1)

    cov = [selective(p_cal, y, t)[0] for t in thresholds]
    acc = [selective(p_cal, y, t)[1] for t in thresholds]
    b.plot(cov, acc, "o-", color="#2f7d6d")
    c0, a0 = selective(p_cal, y, thr)
    b.scatter([c0], [a0], s=120, color="#efb92f", zorder=3, label=f"app setting: answers {c0:.0%}, right {a0:.0%}")
    b.axhline((p_cal.argmax(1) == y).mean(), color="#999", ls="--", lw=1, label="always answering")
    b.set_xlabel("Share of faces the model answers (coverage)"); b.set_ylabel("Accuracy on answered faces")
    b.set_title("Saying “not sure” trades coverage for accuracy"); b.legend(fontsize=8)
    b.set_xlim(0, 1.02); b.set_ylim(0.4, 1)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
