#!/usr/bin/env python3
"""
migrate_data.py — Migrate training data to the v3 directory structure.

Consolidates data from:
  - data/training/       (v1, 189-class labels)
  - data/training_v2/    (v2, 1-class labels)
  - data/training_v3/    (v3, 63-class labels, old flat structure)

Into the new structure:
  data/v3/
  ├── images/
  │   ├── YYYYMMDD_HHMMSS_fff_{undistort|raw}_{W}x{H}.png
  │   └── ...
  ├── labels/
  │   ├── YYYYMMDD_HHMMSS_fff_{undistort|raw}_{W}x{H}.txt
  │   └── ...
  ├── annotations.jsonl
  └── dataset.yaml

V1 labels (189-class) are remapped to 63-class (id % 63).
V2 labels (1-class) are skipped — they use a different model architecture.
V3 labels (63-class) are copied as-is.

Usage:
    uv run python migrate_data.py              # Dry run
    uv run python migrate_data.py --migrate    # Execute migration
"""

import argparse
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2

import config
from classes import ID_TO_CLASS as V1_ID_TO_CLASS
from classes import parse_class_name as v1_parse
from classes_v2 import CLASS_TO_ID as V3_CLASS_TO_ID, CLASS_NAMES, NUM_CLASSES


def build_v1_to_v3_map():
    """Map v1 class IDs (189) to v3 class IDs (63)."""
    mapping = {}
    for v1_id, v1_name in V1_ID_TO_CLASS.items():
        info = v1_parse(v1_name)
        segment = info["segment"]
        v3_id = V3_CLASS_TO_ID.get(segment)
        if v3_id is not None:
            mapping[v1_id] = v3_id
    return mapping


def stem_from_image(img_path, fallback_time):
    """Generate new-format stem from an image file.

    Uses file mtime as timestamp. Reads image for resolution.
    Assumes raw (not undistorted) since we can't tell from the file alone.
    """
    # Use file modification time as the timestamp
    mtime_ms = img_path.stat().st_mtime * 1000

    # Read image for dimensions
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    h, w = img.shape[:2]

    # We can't reliably tell if undistortion was applied from the file alone.
    # Use "raw" as default — the metadata in annotations.jsonl can clarify.
    stem = config.training_frame_stem(mtime_ms, undistorted=False, width=w, height=h)
    return stem, img


def scan_source(src_dir, label_remap=None, skip_bad_frames=False):
    """Scan a source directory and return migration plan.

    Args:
        src_dir: Path to source data directory (has images/ and labels/).
        label_remap: Optional dict mapping old class IDs to new class IDs.
        skip_bad_frames: Skip frames 0-599 (known bad calibration).

    Returns:
        List of dicts: {src_img, src_label, new_stem, remapped_lines}
    """
    img_dir = src_dir / "images"
    label_dir = src_dir / "labels"

    if not img_dir.exists():
        return []

    plan = []
    for img_path in sorted(img_dir.glob("*.png")):
        stem = img_path.stem

        if skip_bad_frames:
            try:
                frame_num = int(stem.split("_")[-1])
                if frame_num < 600:
                    continue
            except ValueError:
                pass

        label_path = label_dir / f"{stem}.txt"
        if not label_path.exists():
            continue

        # Read and remap labels
        lines = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                old_id = int(parts[0])
                if label_remap is not None:
                    new_id = label_remap.get(old_id)
                    if new_id is None:
                        continue
                else:
                    new_id = old_id
                lines.append(f"{new_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

        if not lines:
            continue

        # Resolve actual image path (follow symlinks)
        real_img = img_path.resolve()

        plan.append({
            "src_img": real_img,
            "src_label": label_path,
            "old_stem": stem,
            "remapped_lines": lines,
        })

    return plan


def deduplicate_plan(plans):
    """Deduplicate across sources — same real image shouldn't appear twice.

    Returns merged plan with unique images. If the same image appears in
    multiple sources, prefer the one with v3 labels (63-class).
    """
    seen = {}  # real_img_path -> plan entry
    for source_name, entries in plans:
        for entry in entries:
            key = str(entry["src_img"])
            if key not in seen:
                entry["source"] = source_name
                seen[key] = entry
            # If already seen, keep the existing (first source wins — order matters)

    return list(seen.values())


def assign_stems(entries):
    """Assign new-format stems to deduplicated entries.

    Uses file mtime for timestamp. Adds small offsets if timestamps collide.
    """
    used_stems = set()
    for entry in entries:
        img_path = entry["src_img"]
        mtime_ms = img_path.stat().st_mtime * 1000

        img = cv2.imread(str(img_path))
        if img is None:
            entry["new_stem"] = None
            continue
        h, w = img.shape[:2]

        stem = config.training_frame_stem(mtime_ms, undistorted=False, width=w, height=h)

        # Handle collisions (same-second captures)
        base_stem = stem
        offset = 0
        while stem in used_stems:
            offset += 1
            stem = config.training_frame_stem(mtime_ms + offset, undistorted=False, width=w, height=h)

        used_stems.add(stem)
        entry["new_stem"] = stem


def dry_run():
    """Show migration plan without executing."""
    v1_map = build_v1_to_v3_map()

    sources = []

    # V3 converted data (already 63-class, highest priority)
    v3_old = config.PROJECT_ROOT / "data" / "training_v3"
    if v3_old.exists():
        entries = scan_source(v3_old)
        sources.append(("v3_converted", entries))
        print(f"V3 (converted): {len(entries)} frames")

    # V1 raw data (189-class, remap to 63)
    v1_dir = config.PROJECT_ROOT / "data" / "training"
    if v1_dir.exists():
        entries = scan_source(v1_dir, label_remap=v1_map, skip_bad_frames=True)
        sources.append(("v1_remapped", entries))
        print(f"V1 (remapped, skip bad): {len(entries)} frames")

    # Deduplicate
    merged = deduplicate_plan(sources)
    print(f"\nAfter deduplication: {merged_count(merged)} unique frames")

    # Check class distribution
    class_counts = Counter()
    for entry in merged:
        for line in entry["remapped_lines"]:
            cid = int(line.split()[0])
            class_counts[cid] += 1

    total = sum(class_counts.values())
    covered = sum(1 for c in range(NUM_CLASSES) if class_counts[c] > 0)
    print(f"Annotations: {total}")
    print(f"Classes covered: {covered}/{NUM_CLASSES}")
    print(f"Average per class: {total / NUM_CLASSES:.1f}")

    print(f"\nTarget: {config.DATASET_DIR}")
    print(f"Run with --migrate to execute.")


def merged_count(merged):
    return len([e for e in merged if e.get("src_img")])


def migrate():
    """Execute the migration."""
    v1_map = build_v1_to_v3_map()

    sources = []

    v3_old = config.PROJECT_ROOT / "data" / "training_v3"
    if v3_old.exists():
        entries = scan_source(v3_old)
        sources.append(("v3_converted", entries))
        print(f"V3 source: {len(entries)} frames")

    v1_dir = config.PROJECT_ROOT / "data" / "training"
    if v1_dir.exists():
        entries = scan_source(v1_dir, label_remap=v1_map, skip_bad_frames=True)
        sources.append(("v1_remapped", entries))
        print(f"V1 source: {len(entries)} frames")

    merged = deduplicate_plan(sources)
    print(f"Merged: {merged_count(merged)} unique frames")

    # Assign new stems
    print("Assigning timestamps from file mtimes...")
    assign_stems(merged)

    # Create output directories
    dst_images = config.DATASET_IMAGES_DIR
    dst_labels = config.DATASET_LABELS_DIR
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    # Copy files
    copied = 0
    skipped = 0
    annotations = []

    for entry in merged:
        stem = entry.get("new_stem")
        if stem is None:
            skipped += 1
            continue

        src_img = entry["src_img"]
        dst_img = dst_images / f"{stem}.png"
        dst_lbl = dst_labels / f"{stem}.txt"

        # Don't overwrite if already migrated
        if dst_img.exists():
            skipped += 1
            continue

        # Copy image (actual copy, not symlink — clean break)
        shutil.copy2(str(src_img), str(dst_img))

        # Write remapped label
        with open(dst_lbl, "w") as f:
            f.write("\n".join(entry["remapped_lines"]) + "\n")

        # Annotation record
        annotations.append({
            "filename": f"{stem}.png",
            "source": entry.get("source", "unknown"),
            "original": entry.get("old_stem", ""),
            "n_darts": len(entry["remapped_lines"]),
            "migrated_at": time.time(),
        })

        copied += 1

    # Write annotations
    ann_path = config.DATASET_ANNOTATIONS_PATH
    with open(ann_path, "a") as f:
        for rec in annotations:
            f.write(json.dumps(rec) + "\n")

    # Generate dataset.yaml
    yaml_path = config.DATASET_YAML_PATH
    lines = [
        f"# Dartscorer {config.DATASET_VERSION} — {NUM_CLASSES} classes",
        f"# Migrated from v1/v3 data",
        "",
        f"path: {config.DATASET_DIR.resolve()}",
        "train: images",
        "val: images",
        "",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Stats
    class_counts = Counter()
    for entry in merged:
        if entry.get("new_stem") is None:
            continue
        for line in entry["remapped_lines"]:
            cid = int(line.split()[0])
            class_counts[cid] += 1

    total = sum(class_counts.values())
    covered = sum(1 for c in range(NUM_CLASSES) if class_counts[c] > 0)

    print(f"\n=== Migration Complete ===")
    print(f"  Copied:  {copied} frames")
    print(f"  Skipped: {skipped} (already exists or unreadable)")
    print(f"  Output:  {config.DATASET_DIR}")
    print(f"  Annotations: {total}")
    print(f"  Classes: {covered}/{NUM_CLASSES}")
    print(f"  Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate training data to v3 structure")
    parser.add_argument("--migrate", action="store_true", help="Execute migration")
    args = parser.parse_args()

    if args.migrate:
        migrate()
    else:
        dry_run()
