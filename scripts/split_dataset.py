#!/usr/bin/env python3
"""
split_dataset.py — Create stratified train/val split with oversampling.

Splits data/v3/ into train/val sets using symlinks, preserving the original
data in place. The split is stratified by class to ensure rare classes
(doubles, triples, bulls) appear proportionally in both sets.

Oversampling duplicates images containing underrepresented classes so every
class gets roughly equal exposure during training. Without this, singles
(300-500 samples) dominate training while doubles (20-40 samples) are
barely seen.

Usage:
    uv run python scripts/split_dataset.py              # 80/20 split + oversample
    uv run python scripts/split_dataset.py --no-oversample  # split only
    uv run python scripts/split_dataset.py --val 0.15   # 85/15 split
    uv run python scripts/split_dataset.py --seed 123   # reproducible split
    uv run python scripts/split_dataset.py --stats      # just print current stats
"""

import argparse
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from dartscorer import config
from dartscorer.classes_v2 import CLASS_NAMES, ID_TO_CLASS, NUM_CLASSES


def load_labels(labels_dir):
    """Load class IDs from all label files. Returns {stem: set of class IDs}."""
    file_classes = {}
    for lp in sorted(labels_dir.glob("*.txt")):
        classes = set()
        for line in lp.read_text().strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5:
                classes.add(int(parts[0]))
        if classes:
            file_classes[lp.stem] = classes
    return file_classes


def stratified_split(file_classes, val_fraction=0.2, seed=42):
    """Split files into train/val, ensuring rare classes appear in both sets.

    Strategy: process classes from rarest to most common. For each class,
    ensure at least some of its files end up in val. Files that contain
    multiple classes get assigned once (first assignment wins).
    """
    rng = random.Random(seed)

    # Count class frequency across all files
    class_freq = Counter()
    class_files = defaultdict(list)  # class_id -> [stems]
    for stem, classes in file_classes.items():
        for c in classes:
            class_freq[c] += 1
            class_files[c].append(stem)

    # Sort classes by frequency (rarest first)
    sorted_classes = sorted(class_freq.keys(), key=lambda c: class_freq[c])

    assigned = {}  # stem -> "train" or "val"

    for cls_id in sorted_classes:
        stems = class_files[cls_id]
        # Only consider unassigned files
        unassigned = [s for s in stems if s not in assigned]
        if not unassigned:
            continue

        rng.shuffle(unassigned)

        # How many of this class's files should go to val?
        total_for_class = len(stems)
        target_val = max(1, round(total_for_class * val_fraction))

        # Count how many are already assigned to val
        already_val = sum(1 for s in stems if assigned.get(s) == "val")
        need_val = max(0, target_val - already_val)

        # Assign unassigned files
        for i, stem in enumerate(unassigned):
            if i < need_val:
                assigned[stem] = "val"
            else:
                assigned[stem] = "train"

    # Any remaining unassigned (shouldn't happen, but safety)
    for stem in file_classes:
        if stem not in assigned:
            assigned[stem] = "train"

    train = sorted(s for s, split in assigned.items() if split == "train")
    val = sorted(s for s, split in assigned.items() if split == "val")
    return train, val


def oversample_train(train_stems, file_classes, target_percentile=75):
    """Duplicate images containing rare classes to balance training exposure.

    For each class, counts how many training images contain it. Classes below
    the target count get their images duplicated until they reach it.

    The target is the given percentile of class counts — this avoids trying to
    match the absolute max (singles at 300+) which would create a huge dataset,
    while still giving rare classes much more exposure.

    Returns a new list of stems (with duplicates as "stem__dupN" suffixes).
    """
    # Count per-class representation in training set
    train_set = set(train_stems)
    class_stems = defaultdict(list)  # class_id -> [stems in train]
    for stem in train_stems:
        if stem in file_classes:
            for c in file_classes[stem]:
                class_stems[c].append(stem)

    class_counts = {c: len(stems) for c, stems in class_stems.items()}
    counts_list = sorted(class_counts.values())

    if not counts_list:
        return train_stems

    # Target: percentile of class counts
    idx = min(len(counts_list) - 1, int(len(counts_list) * target_percentile / 100))
    target = counts_list[idx]

    print(f"\nOversampling: target={target} (p{target_percentile} of class counts)")
    print(f"  Class count range before: {counts_list[0]}–{counts_list[-1]}")

    # Build oversampled list
    extra_stems = []
    classes_boosted = 0

    for c in sorted(class_stems.keys()):
        current = class_counts[c]
        if current >= target:
            continue

        classes_boosted += 1
        stems = class_stems[c]
        needed = target - current

        # Cycle through this class's images to fill the gap
        dup_num = 0
        while needed > 0:
            for stem in stems:
                if needed <= 0:
                    break
                extra_stems.append(f"{stem}__dup{dup_num}")
                dup_num += 1
                needed -= 1

    result = list(train_stems) + extra_stems
    n_unique = len(train_stems)
    n_duped = len(extra_stems)
    print(f"  Boosted {classes_boosted} classes, added {n_duped} duplicate images")
    print(f"  Training set: {n_unique} unique + {n_duped} oversampled = {len(result)} total")

    return result


def create_split_dirs(data_dir, train_stems, val_stems):
    """Create train/ and val/ directories with symlinks to original files.

    Handles oversampled stems (with __dupN suffix) by creating numbered symlinks
    pointing to the same original file.
    """
    img_dir = data_dir / "images"
    lbl_dir = data_dir / "labels"

    for split_name, stems in [("train", train_stems), ("val", val_stems)]:
        split_img = data_dir / split_name / "images"
        split_lbl = data_dir / split_name / "labels"

        # Clean previous split
        if (data_dir / split_name).exists():
            shutil.rmtree(data_dir / split_name)
        split_img.mkdir(parents=True)
        split_lbl.mkdir(parents=True)

        for stem in stems:
            # Parse out the duplicate suffix if present
            if "__dup" in stem:
                orig_stem = stem.split("__dup")[0]
                link_stem = stem  # use full name with dup suffix for unique link
            else:
                orig_stem = stem
                link_stem = stem

            # Find image (could be .png or .jpg)
            for ext in (".png", ".jpg"):
                src_img = img_dir / f"{orig_stem}{ext}"
                if src_img.exists():
                    dst = split_img / f"{link_stem}{ext}"
                    if not dst.exists():
                        os.symlink(src_img.resolve(), dst)
                    break

            src_lbl = lbl_dir / f"{orig_stem}.txt"
            if src_lbl.exists():
                dst = split_lbl / f"{link_stem}.txt"
                if not dst.exists():
                    os.symlink(src_lbl.resolve(), dst)


def print_split_stats(data_dir, file_classes, train_stems, val_stems):
    """Print distribution stats for the split."""
    print(f"\nSplit: {len(train_stems)} train / {len(val_stems)} val")

    # Count per-class, resolving dup stems to originals
    train_counts = Counter()
    val_counts = Counter()

    for stem in train_stems:
        orig = stem.split("__dup")[0] if "__dup" in stem else stem
        if orig in file_classes:
            for c in file_classes[orig]:
                train_counts[c] += 1

    for stem in val_stems:
        if stem in file_classes:
            for c in file_classes[stem]:
                val_counts[c] += 1

    print(f"\n{'Class':>8}  {'Train':>6}  {'Val':>5}")
    print("-" * 28)

    for ring_label, class_range in [
        ("Singles", [(i * 3, CLASS_NAMES[i * 3]) for i in range(20)]),
        ("Doubles", [(i * 3 + 1, CLASS_NAMES[i * 3 + 1]) for i in range(20)]),
        ("Triples", [(i * 3 + 2, CLASS_NAMES[i * 3 + 2]) for i in range(20)]),
        ("Bulls", [(60, "S_BULL"), (61, "D_BULL")]),
    ]:
        ring_train = sum(train_counts[cid] for cid, _ in class_range)
        ring_val = sum(val_counts[cid] for cid, _ in class_range)
        print(f"{ring_label:>8}  {ring_train:6d}  {ring_val:5d}")

    # Show weakest classes
    print(f"\nWeakest 10 classes in training set:")
    weakest = sorted(range(NUM_CLASSES), key=lambda c: train_counts[c])
    for c in weakest[:10]:
        name = ID_TO_CLASS[c]
        print(f"  {name:>8}: {train_counts[c]:3d} train / {val_counts[c]:3d} val")


def update_dataset_yaml(data_dir):
    """Update dataset.yaml to point to train/val split directories."""
    yaml_path = data_dir / "dataset.yaml"
    abs_path = str(data_dir.resolve())

    lines = [
        f"# Dartscorer YOLO dataset config — {NUM_CLASSES} classes",
        f"# Auto-generated by split_dataset.py",
        "",
        f"path: {abs_path}",
        "train: train/images",
        "val: val/images",
        "",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")

    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nUpdated {yaml_path}")


def print_current_stats(data_dir):
    """Print stats about the current dataset without modifying anything."""
    file_classes = load_labels(data_dir / "labels")
    counts = Counter()
    for classes in file_classes.values():
        for c in classes:
            counts[c] += 1

    print(f"Dataset: {len(file_classes)} labeled images, "
          f"{sum(counts.values())} annotations")
    print(f"Classes covered: {len(counts)}/{NUM_CLASSES}")
    print(f"\nPer class: min={min(counts.values())}  "
          f"avg={sum(counts.values()) / NUM_CLASSES:.0f}  "
          f"max={max(counts.values())}")

    # Check existing split
    train_dir = data_dir / "train" / "images"
    val_dir = data_dir / "val" / "images"
    if train_dir.exists() and val_dir.exists():
        n_train = len(list(train_dir.iterdir()))
        n_val = len(list(val_dir.iterdir()))
        print(f"\nExisting split: {n_train} train / {n_val} val")
    else:
        print("\nNo train/val split exists yet.")


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified train/val split with oversampling")
    parser.add_argument("--val", type=float, default=0.2,
                        help="Validation fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no-oversample", action="store_true",
                        help="Disable oversampling of rare classes")
    parser.add_argument("--target-percentile", type=int, default=75,
                        help="Oversample target as percentile of class counts (default: 75)")
    parser.add_argument("--stats", action="store_true",
                        help="Print current dataset stats without modifying")
    args = parser.parse_args()

    data_dir = config.DATASET_DIR

    if args.stats:
        print_current_stats(data_dir)
        return

    print(f"Creating {1 - args.val:.0%}/{args.val:.0%} train/val split "
          f"(seed={args.seed})")

    file_classes = load_labels(data_dir / "labels")
    print(f"Found {len(file_classes)} labeled images")

    train_stems, val_stems = stratified_split(
        file_classes, val_fraction=args.val, seed=args.seed)

    if not args.no_oversample:
        train_stems = oversample_train(
            train_stems, file_classes,
            target_percentile=args.target_percentile)

    create_split_dirs(data_dir, train_stems, val_stems)
    print_split_stats(data_dir, file_classes, train_stems, val_stems)
    update_dataset_yaml(data_dir)

    print("\nDone. Run `uv run train` to train with the new split.")


if __name__ == "__main__":
    main()
