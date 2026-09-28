"""Train the facial-emotion classifier on the leak-free split made by split_dataset.py.

What changed vs. the original notebook (and why):
  * Leak-free train / val / test split; the final number is reported on the *test*
    set, which is never used for training or model selection.
  * EfficientNetB0 (or MobileNetV3) + GlobalAveragePooling instead of
    InceptionV3 + Flatten + Dense(512): ~5M instead of ~50M parameters, so the saved
    model is ~20 MB instead of ~409 MB and runs faster in real time.
  * Data augmentation that is actually applied (the old notebook overwrote it).
  * Two phases: train the new head with the backbone frozen, then fine-tune the top
    of the backbone with a low learning rate.
  * EarlyStopping + ReduceLROnPlateau and keeping the best weights, instead of a
    fixed 100 epochs (half of which were empty because of steps_per_epoch).
  * Class weights, because "happiness" has ~3x more images than other classes.
  * Saves the model without optimizer state, plus labels.json, metrics, a
    confusion matrix and training curves.

Usage (GPU strongly recommended; free Colab works, see notebooks/train_colab.ipynb):
    python scripts/train.py --data data/split --out models --reports reports

For the strongest route, first pre-train on a larger seven-class dataset such as
FER-2013 (arranged as <root>/train/<label> and <root>/val/<label>):
    python scripts/train.py --data data/split --pretrain-data data/fer2013
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

LABELS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
IMG_SIZE = 224


def make_datasets(data_dir: Path, batch: int, seed: int):
    common = dict(
        labels="inferred", label_mode="categorical", class_names=LABELS,
        color_mode="grayscale", image_size=(IMG_SIZE, IMG_SIZE), batch_size=batch,
    )
    train = keras.utils.image_dataset_from_directory(data_dir / "train", shuffle=True, seed=seed, **common)
    val = keras.utils.image_dataset_from_directory(data_dir / "val", shuffle=False, **common)
    test = keras.utils.image_dataset_from_directory(data_dir / "test", shuffle=False, **common)
    test_files = list(test.file_paths)

    # Grayscale -> 3 channels. Everything is grayscale so that colour photos (kids set)
    # and grayscale photos (general set) look the same to the model, and so the webcam
    # input at inference matches training. Pixel range stays 0..255: the backbones below
    # contain their own preprocessing layers.
    to_rgb = lambda x, y: (tf.image.grayscale_to_rgb(x), y)  # noqa: E731

    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.12),
        layers.RandomTranslation(0.06, 0.06),
        layers.RandomContrast(0.25),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ], name="augment")

    AUTOTUNE = tf.data.AUTOTUNE
    train = (train.map(to_rgb, num_parallel_calls=AUTOTUNE)
                  .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
                  .prefetch(AUTOTUNE))
    val = val.map(to_rgb, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    test = test.map(to_rgb, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    return train, val, test, test_files


def class_weights(data_dir: Path) -> dict[int, float]:
    counts = np.array([len(list((data_dir / "train" / l).glob("*"))) for l in LABELS], dtype=float)
    weights = counts.sum() / (len(LABELS) * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}


def make_pretrain_datasets(data_dir: Path, batch: int, seed: int):
    """Load a larger source dataset without ever mixing it into the target test set."""
    common = dict(
        labels="inferred", label_mode="categorical", class_names=LABELS,
        color_mode="grayscale", image_size=(IMG_SIZE, IMG_SIZE), batch_size=batch,
    )
    train = keras.utils.image_dataset_from_directory(data_dir / "train", shuffle=True, seed=seed, **common)
    val = keras.utils.image_dataset_from_directory(data_dir / "val", shuffle=False, **common)
    to_rgb = lambda x, y: (tf.image.grayscale_to_rgb(x), y)  # noqa: E731
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"), layers.RandomRotation(0.06), layers.RandomZoom(0.12),
        layers.RandomTranslation(0.06, 0.06), layers.RandomContrast(0.25),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ], name="pretrain_augment")
    autotune = tf.data.AUTOTUNE
    train = (train.map(to_rgb, num_parallel_calls=autotune)
                  .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=autotune)
                  .prefetch(autotune))
    val = val.map(to_rgb, num_parallel_calls=autotune).cache().prefetch(autotune)
    return train, val


def build_model(backbone_name: str, weights: str | None):
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")  # expects pixel values 0..255
    if backbone_name == "efficientnetb0":
        base = keras.applications.EfficientNetB0(include_top=False, weights=weights, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif backbone_name == "mobilenetv3":
        base = keras.applications.MobileNetV3Large(include_top=False, weights=weights, input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                   include_preprocessing=True)
    else:
        raise ValueError(backbone_name)
    base.trainable = False
    x = base(inputs, training=False)  # keep BatchNorm in inference mode, also while fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(LABELS), activation="softmax", name="emotion")(x)
    return keras.Model(inputs, outputs, name=f"emotion_{backbone_name}"), base


def callbacks(patience: int):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(2, patience // 2), min_lr=1e-7),
    ]


def plot_curves(histories, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = {k: sum((h.history.get(k, []) for h in histories), []) for k in ["loss", "val_loss", "accuracy", "val_accuracy"]}
    split_at = len(histories[0].history["loss"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(hist[metric], label=f"train {metric}")
        ax.plot(hist[f"val_{metric}"], label=f"val {metric}")
        ax.axvline(split_at - 0.5, color="grey", ls="--", lw=1)
        ax.text(split_at - 0.3, ax.get_ylim()[1] * 0.95, "fine-tune →", fontsize=8, color="grey")
        ax.set_xlabel("epoch"); ax.set_title(metric); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_confusion(cm: np.ndarray, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/split")
    ap.add_argument("--out", default="models")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--backbone", default="efficientnetb0", choices=["efficientnetb0", "mobilenetv3"])
    ap.add_argument("--weights", default="imagenet", help="'imagenet', a path to a weights file, or 'none'")
    ap.add_argument("--pretrain-data", help="optional large source dataset with train/ and val/ label folders")
    ap.add_argument("--pretrain-epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--head-epochs", type=int, default=20)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--unfreeze", type=int, default=60, help="number of backbone layers to fine-tune")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)
    data, out, reports = Path(args.data), Path(args.out), Path(args.reports)
    out.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)

    train, val, test, test_files = make_datasets(data, args.batch, args.seed)
    cw = class_weights(data)
    print("Class weights:", {LABELS[i]: round(w, 2) for i, w in cw.items()})

    model, base = build_model(args.backbone, None if args.weights == "none" else args.weights)

    histories = []
    if args.pretrain_data:
        source = Path(args.pretrain_data)
        pretrain, preval = make_pretrain_datasets(source, args.batch, args.seed)
        print(f"Pre-training the emotion head on {source} before target-domain fine-tuning")
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05), metrics=["accuracy"],
        )
        histories.append(model.fit(
            pretrain, validation_data=preval, epochs=args.pretrain_epochs,
            class_weight=class_weights(source), callbacks=callbacks(4),
        ))

    # Phase 1: adapt the classifier head to the small target dataset.
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                  metrics=["accuracy"])
    h1 = model.fit(train, validation_data=val, epochs=args.head_epochs, class_weight=cw, callbacks=callbacks(5))
    histories.append(h1)

    # Phase 2: fine-tune the top of the backbone (BatchNorm layers stay frozen).
    base.trainable = True
    for layer in base.layers[:-args.unfreeze]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                  metrics=["accuracy"])
    h2 = model.fit(train, validation_data=val, epochs=args.finetune_epochs, class_weight=cw, callbacks=callbacks(7))
    histories.append(h2)

    # ---- Final evaluation on the untouched test set ----
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in test])
    probs = model.predict(test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    sources = np.array([Path(f).name.split("__")[0] for f in test_files])

    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    metrics = {
        "model": f"{args.backbone} + GAP (fine-tuned)",
        "test_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "test_macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "test_images": int(len(y_true)),
        "per_source_accuracy": {s: round(float((y_pred[sources == s] == y_true[sources == s]).mean()), 4)
                                for s in sorted(set(sources))},
        "best_val_accuracy": round(float(max(h1.history["val_accuracy"] + h2.history["val_accuracy"])), 4),
        "epochs_run": {"head": len(h1.history["loss"]), "finetune": len(h2.history["loss"])},
        "source_pretraining": str(args.pretrain_data) if args.pretrain_data else None,
        "params": int(model.count_params()),
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
    }
    report = classification_report(y_true, y_pred, labels=range(len(LABELS)), target_names=LABELS, zero_division=0)
    print(report)
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))

    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports / "classification_report.txt").write_text(report)
    plot_confusion(cm, reports / "confusion_matrix.png", f"Test set confusion matrix (acc {metrics['test_accuracy']:.1%})")
    # Plot target adaptation only so the domain switch does not make the graph misleading.
    plot_curves([h1, h2], reports / "training_curves.png")

    # Save without optimizer state: an uncompiled copy with the trained weights.
    export = keras.Model(model.inputs, model.outputs, name=model.name)
    model_path = out / "emotion_model.keras"
    export.save(model_path)
    (out / "labels.json").write_text(json.dumps({
        "labels": LABELS, "image_size": IMG_SIZE, "color": "grayscale_as_rgb", "pixel_range": [0, 255],
        "backbone": args.backbone,
    }, indent=2))
    print(f"\nSaved {model_path} ({model_path.stat().st_size / 1e6:.1f} MB) and {out / 'labels.json'}")


if __name__ == "__main__":
    main()
