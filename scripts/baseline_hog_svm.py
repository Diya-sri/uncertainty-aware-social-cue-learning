"""Classical baseline: HOG features + linear SVM, evaluated on the leak-free test split.

No deep-learning libraries needed. It gives a floor the CNN must beat, and a sanity
check that the split is sensible.

Usage:
    python scripts/baseline_hog_svm.py --data data/split --out reports/baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

LABELS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
SIZE = 64
HOG = cv2.HOGDescriptor((SIZE, SIZE), (16, 16), (8, 8), (8, 8), 9)


def features(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (SIZE, SIZE))
    img = cv2.equalizeHist(img)
    return HOG.compute(img).ravel()


def load(split_dir: Path, augment: bool = False):
    X, y, src = [], [], []
    for i, label in enumerate(LABELS):
        for p in sorted((split_dir / label).glob("*")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            X.append(features(img)); y.append(i); src.append(p.name.split("__")[0])
            if augment:
                X.append(features(cv2.flip(img, 1))); y.append(i); src.append(src[-1])
    return np.array(X), np.array(y), np.array(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/split")
    ap.add_argument("--out", default="reports/baseline")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, _ = load(data / "train", augment=True)
    Xva, yva, _ = load(data / "val")
    Xte, yte, ste = load(data / "test")

    best = None
    for C in [0.0005, 0.001, 0.005, 0.01, 0.05]:
        clf = make_pipeline(StandardScaler(), LinearSVC(C=C, class_weight="balanced", max_iter=20000))
        clf.fit(Xtr, ytr)
        f1 = f1_score(yva, clf.predict(Xva), average="macro")
        print(f"C={C:<7} val macro-F1={f1:.3f}")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)

    _, C, clf = best
    pred = clf.predict(Xte)
    metrics = {
        "model": "HOG + LinearSVC",
        "chosen_C": C,
        "test_accuracy": round(float(accuracy_score(yte, pred)), 4),
        "test_macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
        "test_images": int(len(yte)),
        "per_source_accuracy": {s: round(float((pred[ste == s] == yte[ste == s]).mean()), 4)
                                for s in sorted(set(ste))},
        "confusion_matrix": confusion_matrix(yte, pred, labels=range(len(LABELS))).tolist(),
        "labels": LABELS,
    }
    report = classification_report(yte, pred, labels=range(len(LABELS)), target_names=LABELS, zero_division=0)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "classification_report.txt").write_text(report)
    print(report)
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
