#!/usr/bin/env python3
"""
analyze_data.py — Analyze training data distribution.

Generates heatmaps and stats showing how many examples exist per class.

Usage:
    python analyze_data.py                    # Print stats + show heatmap
    python analyze_data.py --save             # Save heatmap to file
"""

import argparse
import os
from pathlib import Path
from collections import Counter

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
from dartscorer.classes_v2 import ID_TO_CLASS, CLASS_NAMES, SEGMENTS, NUM_CLASSES, parse_class_name
from dartscorer import config


def load_class_counts(outdir=None):
    """Count instances of each class across all label files."""
    outdir = Path(outdir) if outdir else config.DATASET_DIR
    label_dir = outdir / "labels"
    counts = Counter()

    for label_path in label_dir.glob("*.txt"):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cls_name = ID_TO_CLASS.get(cls_id, f"unknown_{cls_id}")
                    counts[cls_name] += 1

    return counts


def print_stats(counts):
    """Print summary statistics."""
    total = sum(counts.values())
    n_classes_seen = sum(1 for seg in CLASS_NAMES if counts.get(seg, 0) > 0)

    print(f"\n{'='*60}")
    print(f"TRAINING DATA DISTRIBUTION — {total} annotations, {n_classes_seen}/{NUM_CLASSES} classes seen")
    print(f"{'='*60}")

    # Per-ring totals
    ring_counts = Counter()
    for seg, count in counts.items():
        info = parse_class_name(seg)
        ring_counts[info["ring"]] += count
    for ring in ("single", "double", "triple", "S-BULL", "D-BULL"):
        print(f"  {ring:>8s}: {ring_counts.get(ring, 0)} annotations")

    # Per-segment totals
    print(f"\nPer-segment counts:")
    for seg in SEGMENTS:
        c = counts.get(seg, 0)
        bar = "#" * min(c, 40)
        print(f"  {seg:>8s}: {c:4d}  {bar}")

    # Missing classes
    missing = [seg for seg in CLASS_NAMES if counts.get(seg, 0) == 0]
    if missing:
        print(f"\nMissing classes ({len(missing)}):")
        for seg in missing:
            print(f"  {seg}")


def render_heatmap(counts, save_path=None):
    """Render a visual heatmap: one row per segment with count bar."""
    n_segs = len(SEGMENTS)
    seg_counts = [counts.get(seg, 0) for seg in SEGMENTS]
    max_count = max(max(seg_counts), 1)

    # Render
    cell_w = 200
    cell_h = 18
    label_w = 80
    header_h = 30
    w = label_w + cell_w + 60
    h = header_h + n_segs * cell_h + 40

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)

    # Header
    cv2.putText(img, "Segment", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(img, "Count", (label_w + 5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Rows
    for si, seg in enumerate(SEGMENTS):
        y = header_h + si * cell_h
        count = seg_counts[si]

        # Row label
        cv2.putText(img, seg, (5, y + cell_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Color bar
        bar_w = int(cell_w * count / max_count) if max_count > 0 else 0
        if count == 0:
            color = (40, 40, 40)
            text_color = (80, 80, 80)
        else:
            intensity = min(count / max(max_count * 0.5, 1), 1.0)
            if intensity < 0.5:
                g = int(100 + 155 * (intensity * 2))
                color = (0, g, 0)
            else:
                r = int(255 * ((intensity - 0.5) * 2))
                color = (0, 255, r)
            text_color = (255, 255, 255)

        cv2.rectangle(img, (label_w, y + 1), (label_w + bar_w, y + cell_h - 1), color, -1)
        cv2.putText(img, str(count), (label_w + bar_w + 5, y + cell_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)

    # Total at bottom
    total = sum(seg_counts)
    n_seen = sum(1 for c in seg_counts if c > 0)
    cv2.putText(img, f"Total: {total}  |  Classes seen: {n_seen}/{NUM_CLASSES}",
                (5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if save_path:
        cv2.imwrite(str(save_path), img)
        print(f"Heatmap saved to {save_path}")

    return img


def main():
    parser = argparse.ArgumentParser(description="Analyze training data distribution")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--save", action="store_true", help="Save heatmap image")
    args = parser.parse_args()

    counts = load_class_counts(args.outdir)
    print_stats(counts)

    save_path = Path(args.outdir) / "class_heatmap.png" if args.save else None
    img = render_heatmap(counts, save_path)

    if not args.save:
        # Show interactively
        cv2.namedWindow("Class Density Heatmap", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Class Density Heatmap", img.shape[1], img.shape[0])
        cv2.imshow("Class Density Heatmap", img)
        print("\nPress any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
