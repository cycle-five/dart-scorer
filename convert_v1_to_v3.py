#!/usr/bin/env python3
"""
convert_v1_to_v3.py — Convert v1 training data (189-class) to v3 (63-class).

V1 had 189 classes = 3 ordinals × 63 segments.
V3 drops ordinals: new_class_id = old_class_id % 63.
Bounding boxes and tip coordinates are preserved as-is.

Usage:
    uv run python convert_v1_to_v3.py              # Dry run (stats)
    uv run python convert_v1_to_v3.py --convert     # Execute conversion
    uv run python convert_v1_to_v3.py --convert --skip-bad-frames  # Skip frames 0-600
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

import config
from classes import ID_TO_CLASS as V1_ID_TO_CLASS, NUM_CLASSES as V1_NUM_CLASSES
from classes import parse_class_name as v1_parse
from classes_v2 import (
    CLASS_NAMES as V3_CLASS_NAMES,
    CLASS_TO_ID as V3_CLASS_TO_ID,
    NUM_CLASSES as V3_NUM_CLASSES,
    SEGMENTS,
)

V1_DIR = config.PROJECT_ROOT / "data" / "training"
V3_DIR = config.PROJECT_ROOT / "data" / "training_v3"

# Number of segments (same in v1 and v3)
N_SEGMENTS = len(SEGMENTS)  # 63


def build_class_map():
    """Build v1_id → v3_id mapping.

    V1 classes are ordered: d1_S1, d1_D1, ..., d1_MISS, d2_S1, ..., d3_MISS
    V3 classes are ordered: S1, D1, ..., MISS
    So: v3_id = v1_id % 63
    """
    mapping = {}
    for v1_id, v1_name in V1_ID_TO_CLASS.items():
        info = v1_parse(v1_name)
        segment = info["segment"]
        v3_id = V3_CLASS_TO_ID.get(segment)
        if v3_id is not None:
            mapping[v1_id] = v3_id
    return mapping


def dry_run(skip_bad_frames=False):
    """Show stats about what conversion would do."""
    v1_labels = V1_DIR / "labels"
    v1_images = V1_DIR / "images"

    if not v1_labels.exists():
        print("ERROR: No v1 labels found at", v1_labels)
        return

    label_files = sorted(v1_labels.glob("*.txt"))
    image_files = sorted(v1_images.glob("*.png"))
    class_map = build_class_map()

    print(f"=== V1 → V3 Conversion (Dry Run) ===")
    print(f"  V1 labels:  {len(label_files)}")
    print(f"  V1 images:  {len(image_files)}")
    print(f"  V1 classes: {V1_NUM_CLASSES} → V3 classes: {V3_NUM_CLASSES}")
    print(f"  Mapping:    new_id = old_id % {N_SEGMENTS}")

    # Count annotations per v3 class
    v3_counts = Counter()
    total = 0
    skipped_frames = 0

    for lf in label_files:
        if skip_bad_frames:
            frame_num = int(lf.stem.split("_")[-1])
            if frame_num < 600:
                skipped_frames += 1
                continue

        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                v1_id = int(parts[0])
                v3_id = class_map.get(v1_id)
                if v3_id is not None:
                    v3_counts[V3_CLASS_NAMES[v3_id]] += 1
                    total += 1

    print(f"\n  Total annotations: {total}")
    if skip_bad_frames:
        print(f"  Skipped frames (0-599): {skipped_frames}")
    print(f"  Classes covered: {len(v3_counts)} / {V3_NUM_CLASSES}")

    # Per-class stats
    print(f"\n  Per-class distribution:")
    for seg in SEGMENTS:
        c = v3_counts.get(seg, 0)
        bar = "#" * min(c, 50)
        print(f"    {seg:>8s}: {c:4d}  {bar}")

    avg = total / V3_NUM_CLASSES if V3_NUM_CLASSES > 0 else 0
    print(f"\n  Average per class: {avg:.1f} (was {total / V1_NUM_CLASSES:.1f} with 189 classes)")
    print(f"\n  Output would go to: {V3_DIR}")
    print(f"  Run with --convert to execute.")


def convert(skip_bad_frames=False):
    """Execute the conversion."""
    v1_labels = V1_DIR / "labels"
    v1_images = V1_DIR / "images"

    if not v1_labels.exists():
        print("ERROR: No v1 labels found at", v1_labels)
        return

    v3_images = V3_DIR / "images"
    v3_labels = V3_DIR / "labels"
    v3_images.mkdir(parents=True, exist_ok=True)
    v3_labels.mkdir(parents=True, exist_ok=True)

    class_map = build_class_map()
    label_files = sorted(v1_labels.glob("*.txt"))

    converted = 0
    skipped = 0
    total_annotations = 0
    v3_counts = Counter()

    for lf in label_files:
        stem = lf.stem

        if skip_bad_frames:
            frame_num = int(stem.split("_")[-1])
            if frame_num < 600:
                skipped += 1
                continue

        # Find matching image
        img_src = v1_images / f"{stem}.png"
        if not img_src.exists():
            skipped += 1
            continue

        # Convert labels
        v3_lines = []
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                v1_id = int(parts[0])
                v3_id = class_map.get(v1_id)
                if v3_id is None:
                    continue
                # Keep bbox as-is (cx, cy, w, h normalized)
                v3_lines.append(f"{v3_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
                v3_counts[V3_CLASS_NAMES[v3_id]] += 1
                total_annotations += 1

        if v3_lines:
            # Symlink image to save disk space
            img_dst = v3_images / f"{stem}.png"
            if not img_dst.exists():
                img_dst.symlink_to(img_src.resolve())

            # Write v3 label
            with open(v3_labels / f"{stem}.txt", "w") as f:
                f.write("\n".join(v3_lines) + "\n")

            converted += 1

    # Generate dataset.yaml
    yaml_path = V3_DIR / "dataset.yaml"
    lines = [
        f"# Dartscorer V3 — {V3_NUM_CLASSES} classes (no ordinals)",
        f"# Converted from v1 ({V1_NUM_CLASSES} classes)",
        "",
        f"path: {V3_DIR.resolve()}",
        "train: images",
        "val: images",
        "",
        "names:",
    ]
    for i, name in enumerate(V3_CLASS_NAMES):
        lines.append(f"  {i}: {name}")
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"=== V1 → V3 Conversion Complete ===")
    print(f"  Converted:    {converted} frames")
    print(f"  Skipped:      {skipped} frames")
    print(f"  Annotations:  {total_annotations}")
    print(f"  Classes:      {len(v3_counts)} / {V3_NUM_CLASSES} covered")
    print(f"  Output:       {V3_DIR}")
    print(f"  Dataset YAML: {yaml_path}")

    avg = total_annotations / V3_NUM_CLASSES if V3_NUM_CLASSES > 0 else 0
    print(f"\n  Average per class: {avg:.1f} (was {total_annotations / V1_NUM_CLASSES:.1f} with v1)")

    # Show lowest-count classes
    print(f"\n  Lowest-count classes:")
    for seg, count in v3_counts.most_common()[-10:]:
        print(f"    {seg:>8s}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert v1 (189-class) to v3 (63-class)")
    parser.add_argument("--convert", action="store_true", help="Execute conversion")
    parser.add_argument("--skip-bad-frames", action="store_true",
                        help="Skip frames 0-599 (pre-recalibration)")
    args = parser.parse_args()

    if args.convert:
        convert(skip_bad_frames=args.skip_bad_frames)
    else:
        dry_run(skip_bad_frames=args.skip_bad_frames)
