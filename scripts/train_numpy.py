"""Train the emotion model without TensorFlow: frozen ImageNet MobileNetV2 features
(NumPy implementation) + a logistic-regression head, with augmentation.

This is what produced the shipped models/emotion_model.npz. scripts/train.py (TensorFlow,
full fine-tuning) is the heavier alternative if you have a GPU.

Model selection (feature set, regularisation C) uses the validation split only. The test
split is touched exactly once, at the end.

Usage:
  python scripts/convert_mobilenetv2.py mobilenet_v2_..._no_top.h5 models/mobilenetv2_imagenet.npz
  python scripts/train_numpy.py --data data/split --backbone models/mobilenetv2_imagenet.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from emotion_model import crop_face_like_app  # noqa: E402
from mobilenet_np import MobileNetV2  # noqa: E402

LABELS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
SIZE = 224


def load_split(split_dir: Path):
    imgs, ys, srcs, names = [], [], [], []
    for i, label in enumerate(LABELS):
        for p in sorted((split_dir / label).glob("*")):
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            gray = cv2.cvtColor(crop_face_like_app(bgr), cv2.COLOR_BGR2GRAY)
            imgs.append(cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
            ys.append(i); srcs.append(p.name.split("__")[0]); names.append(p.name)
    return np.stack(imgs), np.array(ys), np.array(srcs), names


def augment(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random flip, rotation, zoom, shift, brightness/contrast and occasional blur,
    roughly the variation a webcam produces."""
    if rng.random() < 0.5:
        gray = gray[:, ::-1]
    angle = rng.uniform(-10, 10)
    scale = rng.uniform(0.9, 1.12)
    tx, ty = rng.uniform(-0.06, 0.06, 2) * SIZE
    m = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), angle, scale)
    m[:, 2] += (tx, ty)
    out = cv2.warpAffine(np.ascontiguousarray(gray), m, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)
    out = out.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25)
    if rng.random() < 0.2:
        out = cv2.GaussianBlur(out, (5, 5), 0)
    return np.clip(out, 0, 255)


def to_rgb(batch_gray: np.ndarray) -> np.ndarray:
    return np.repeat(batch_gray[..., None].astype(np.float32), 3, axis=-1)


def extract(net, grays, feature_set, views=1, rng=None, flip_tta=False, batch=32):
    """Features for every image; with views>1, the first view is the original and the rest are augmented."""
    feats = []
    for v in range(views):
        imgs = grays if v == 0 else np.stack([augment(g, rng) for g in grays])
        feats.append(net.features_batched(to_rgb(imgs), feature_set, batch))
    if flip_tta:
        feats.append(net.features_batched(to_rgb(grays[:, :, ::-1]), feature_set, batch))
    return np.stack(feats)  # (views, N, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/split")
    ap.add_argument("--backbone", default="models/mobilenetv2_imagenet.npz")
    ap.add_argument("--out", default="models/emotion_model.npz")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--views", type=int, default=6, help="training views per image (1 original + augmented)")
    ap.add_argument("--feature-sets", default="gap,multi")
    ap.add_argument("--cache", default="data/feature_cache")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(args.seed)
    data, reports, cache = Path(args.data), Path(args.reports), Path(args.cache)
    reports.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    bb = dict(np.load(args.backbone))
    net = MobileNetV2(bb)

    splits = {s: load_split(data / s) for s in ("train", "val", "test")}
    for s, (x, y, _, _) in splits.items():
        print(f"{s}: {len(y)} images")

    feats = {}
    for fs in args.feature_sets.split(","):
        path = cache / f"{fs}_v{args.views}_s{args.seed}.npz"
        if path.exists():
            d = np.load(path); feats[fs] = {k: d[k] for k in d.files}
            print(f"[{fs}] loaded cached features {path}")
            continue
        t0 = time.time()
        f = {
            "train": extract(net, splits["train"][0], fs, args.views, rng),
            "val": extract(net, splits["val"][0], fs, 1, flip_tta=True),
            "test": extract(net, splits["test"][0], fs, 1, flip_tta=True),
        }
        np.savez(path, **f)
        feats[fs] = f
        print(f"[{fs}] features {f['train'].shape[-1]}-d extracted in {time.time() - t0:.0f}s")

    ytr, yva, yte = splits["train"][1], splits["val"][1], splits["test"][1]

    def fit(fs, C):
        Xtr = feats[fs]["train"].reshape(-1, feats[fs]["train"].shape[-1])
        Ytr = np.tile(ytr, feats[fs]["train"].shape[0])
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(scaler.transform(Xtr), Ytr)
        return scaler, clf

    def predict_proba(scaler, clf, F):  # average over TTA views
        return np.mean([clf.predict_proba(scaler.transform(v)) for v in F], axis=0)

    results = []
    for fs in feats:
        for C in [0.0003, 0.001, 0.003, 0.01, 0.03]:
            scaler, clf = fit(fs, C)
            pv = predict_proba(scaler, clf, feats[fs]["val"]).argmax(1)
            f1, acc = f1_score(yva, pv, average="macro"), accuracy_score(yva, pv)
            results.append({"feature_set": fs, "C": C, "val_macro_f1": round(f1, 4), "val_acc": round(acc, 4)})
            print(f"  {fs:5s} C={C:<7} val acc={acc:.3f}  macro-F1={f1:.3f}")
    best = max(results, key=lambda r: r["val_macro_f1"])
    print("Selected on validation:", best)

    # Final model: refit on train only (keeps val a fair record of the selection), evaluate test once.
    scaler, clf = fit(best["feature_set"], best["C"])
    probs = predict_proba(scaler, clf, feats[best["feature_set"]]["test"])
    pred = probs.argmax(1)
    src = splits["test"][2]
    cm = confusion_matrix(yte, pred, labels=range(len(LABELS)))
    metrics = {
        "model": f"MobileNetV2 (ImageNet, frozen, NumPy) + logistic regression [{best['feature_set']} features]",
        "test_accuracy": round(float(accuracy_score(yte, pred)), 4),
        "test_macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
        "test_images": int(len(yte)),
        "per_source_accuracy": {s: round(float((pred[src == s] == yte[src == s]).mean()), 4) for s in sorted(set(src))},
        "selected": best,
        "validation_search": results,
        "train_views_per_image": args.views,
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
    }
    report = classification_report(yte, pred, labels=range(len(LABELS)), target_names=LABELS, zero_division=0)
    print(report)
    print(json.dumps({k: metrics[k] for k in ("test_accuracy", "test_macro_f1", "per_source_accuracy")}, indent=2))
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports / "classification_report.txt").write_text(report)
    plot_confusion(cm, reports / "confusion_matrix.png",
                   f"Test set (n={len(yte)}): accuracy {metrics['test_accuracy']:.1%}, macro-F1 {metrics['test_macro_f1']:.2f}")

    # Fold the scaler into one linear layer: logits = feats @ W + b
    W = (clf.coef_ / scaler.scale_).T.astype(np.float32)
    b = (clf.intercept_ - (clf.coef_ * scaler.mean_ / scaler.scale_).sum(1)).astype(np.float32)
    out = {k: v.astype(np.float16) for k, v in bb.items() if k.startswith("bb/")}
    out.update({"head/W": W, "head/b": b,
                "meta": np.array(json.dumps({"labels": LABELS, "image_size": SIZE, "feature_set": best["feature_set"],
                                             "flip_tta": True, "backbone": "mobilenetv2_imagenet"}))})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"Saved {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


def plot_confusion(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=40, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "#23303b")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of true class")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
