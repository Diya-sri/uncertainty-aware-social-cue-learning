"""Build a leak-free train / val / test split from one or more raw face datasets.

Why this exists: the original project's `validation/` folder was an exact copy of
images that were also in `train/`, so the reported 93% "validation accuracy" was
measured on images the model had already seen. This script:

  1. Pools every image from each source (ignoring the old train/validation folders).
  2. Maps each source's folder names onto one shared set of 7 labels.
  3. Removes exact duplicates (same bytes) and near-duplicates (same perceptual hash),
     and drops any duplicate group whose copies disagree on the label.
  4. Makes a stratified split (per source and label) into train / val / test.
  5. Writes the split to disk plus a manifest.csv and a split_report.json.

Usage:
    python scripts/split_dataset.py \
        --source general=data/raw/dataset \
        --source kids=data/raw/dataset1 \
        --out data/split

Optional: --quiz-faces 6 copies 6 *test* images per emotion into static/quiz_faces/
for the quiz game (only do this if the dataset's license allows it).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

LABELS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

# Folder-name aliases found in the two original datasets -> shared label.
LABEL_ALIASES = {
    "anger": "anger", "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happiness", "happy": "happiness", "joy": "happiness",
    "neutral": "neutral", "neutrality": "neutral", "natural": "neutral",
    "sadness": "sadness", "sad": "sadness",
    "surprise": "surprise", "surprised": "surprise",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dhash(path: Path, size: int = 8) -> str:
    """Difference hash: robust to resizing / re-encoding, so it catches near-duplicates."""
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        px = list(im.tobytes())
    bits = []
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits.append("1" if left > right else "0")
    return f"{int(''.join(bits), 2):0{size * size // 4}x}"


def collect(source_name: str, root: Path) -> list[dict]:
    items = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
            continue
        label = LABEL_ALIASES.get(path.parent.name.lower())
        if label is None:
            print(f"  skipping (unknown label folder '{path.parent.name}'): {path}")
            continue
        items.append({
            "source": source_name,
            "label": label,
            "path": path,
            "md5": hashlib.md5(path.read_bytes()).hexdigest(),
        })
    return items


def dedupe(items: list[dict]) -> tuple[list[dict], dict]:
    stats = Counter()
    # 1) exact duplicates
    by_md5 = defaultdict(list)
    for it in items:
        by_md5[it["md5"]].append(it)
    unique = []
    for group in by_md5.values():
        labels = {g["label"] for g in group}
        if len(labels) > 1:
            stats["exact_dup_groups_conflicting_labels_dropped"] += 1
            continue
        stats["exact_duplicates_removed"] += len(group) - 1
        unique.append(group[0])

    # 2) near duplicates (same perceptual hash)
    by_hash = defaultdict(list)
    for it in unique:
        it["dhash"] = dhash(it["path"])
        by_hash[it["dhash"]].append(it)
    kept = []
    for group in by_hash.values():
        labels = {g["label"] for g in group}
        if len(labels) > 1:
            stats["near_dup_groups_conflicting_labels_dropped"] += 1
            stats["images_dropped_for_label_conflict"] += len(group)
            continue
        stats["near_duplicates_removed"] += len(group) - 1
        kept.append(group[0])
    return kept, dict(stats)


def stratified_split(items, val_frac, test_frac, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for it in items:
        buckets[(it["source"], it["label"])].append(it)
    split = {"train": [], "val": [], "test": []}
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda x: x["md5"])
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * test_frac)) if n >= 3 else 0
        n_val = max(1, round(n * val_frac)) if n >= 3 else 0
        split["test"] += group[:n_test]
        split["val"] += group[n_test:n_test + n_val]
        split["train"] += group[n_test + n_val:]
    return split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True,
                    help="name=path to a raw dataset folder (repeatable)")
    ap.add_argument("--out", default="data/split")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiz-faces", type=int, default=0,
                    help="copy N test images per emotion to static/quiz_faces (check the dataset license first)")
    args = ap.parse_args()

    all_items = []
    for spec in args.source:
        name, _, path = spec.partition("=")
        root = Path(path)
        if not root.is_dir():
            raise SystemExit(f"Source folder not found: {root}")
        found = collect(name, root)
        print(f"{name}: {len(found)} image files in {root}")
        all_items += found

    kept, stats = dedupe(all_items)
    print(f"After de-duplication: {len(kept)} unique images  {stats}")

    split = stratified_split(kept, args.val, args.test, args.seed)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    rows = []
    for split_name, its in split.items():
        for it in its:
            dest = out / split_name / it["label"] / f"{it['source']}__{it['path'].name}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(it["path"], dest)
            rows.append({"split": split_name, "source": it["source"], "label": it["label"],
                         "file": str(dest.relative_to(out)), "original": str(it["path"]), "md5": it["md5"]})

    # Sanity check: no image content appears in more than one split.
    seen = defaultdict(set)
    for r in rows:
        seen[r["md5"]].add(r["split"])
    leaks = [m for m, s in seen.items() if len(s) > 1]
    assert not leaks, f"{len(leaks)} images appear in more than one split"

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    counts = {s: {src: dict(Counter(it["label"] for it in its if it["source"] == src))
                  for src in sorted({it["source"] for it in kept})}
              for s, its in split.items()}
    report = {"total_files_found": len(all_items), "unique_images_kept": len(kept),
              "dedup_stats": stats, "counts": counts, "seed": args.seed,
              "cross_split_leaks": len(leaks)}
    (out / "split_report.json").write_text(json.dumps(report, indent=2))

    print("\nImages per split:")
    for s, its in split.items():
        c = Counter(it["label"] for it in its)
        print(f"  {s:5s} {len(its):5d}  " + "  ".join(f"{l}={c.get(l, 0)}" for l in LABELS))
    print(f"\nNo image appears in more than one split. Wrote {out}/")

    if args.quiz_faces:
        qdir = Path(__file__).resolve().parent.parent / "static" / "quiz_faces"
        rng = random.Random(args.seed)
        for label in LABELS:
            pool = [it for it in split["test"] if it["label"] == label]
            rng.shuffle(pool)
            for it in pool[:args.quiz_faces]:
                d = qdir / label
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(it["path"], d / f"{it['source']}__{it['path'].name}")
        print(f"Copied quiz faces to {qdir}")


if __name__ == "__main__":
    main()
