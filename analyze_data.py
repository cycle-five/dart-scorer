#!/usr/bin/env python3
"""
analyze_data.py — Analyze training data distribution.

Generates heatmaps and stats showing how many examples exist per class,
per segment, per ordinal, etc.

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
from classes import ID_TO_CLASS, SEGMENTS, ORDINALS, parse_class_name, NUM_CLASSES
import config


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
    n_classes_seen = sum(1 for c in counts.values() if c > 0)

    print(f"\n{'='*60}")
    print(f"TRAINING DATA DISTRIBUTION — {total} annotations, {n_classes_seen}/{NUM_CLASSES} classes seen")
    print(f"{'='*60}")

    # Per-ordinal totals
    for d in ORDINALS:
        d_total = sum(v for k, v in counts.items() if k.startswith(f"d{d}_"))
        print(f"  Dart {d}: {d_total} annotations")

    # Per-segment totals (across all ordinals)
    print(f"\nSegment totals (all ordinals combined):")
    seg_counts = Counter()
    for cls_name, count in counts.items():
        info = parse_class_name(cls_name)
        seg_counts[info["segment"]] += count

    for seg in SEGMENTS:
        c = seg_counts.get(seg, 0)
        bar = "#" * min(c, 40)
        print(f"  {seg:>8s}: {c:4d}  {bar}")

    # Missing classes
    missing = [cls for cls in [f"d{d}_{s}" for d in ORDINALS for s in SEGMENTS]
               if counts.get(cls, 0) == 0]
    if missing:
        print(f"\nMissing classes ({len(missing)}):")
        # Group by segment
        missing_segs = Counter()
        for m in missing:
            info = parse_class_name(m)
            missing_segs[info["segment"]] += 1
        for seg, n in sorted(missing_segs.items(), key=lambda x: -x[1]):
            ords = [str(d) for d in ORDINALS if f"d{d}_{seg}" in missing]
            print(f"  {seg}: missing ordinals {', '.join(ords)}")


def render_heatmap(counts, save_path=None):
    """Render a visual heatmap: segments (rows) × ordinals (columns)."""
    # Build matrix: segments × ordinals
    n_segs = len(SEGMENTS)
    n_ords = len(ORDINALS)
    matrix = np.zeros((n_segs, n_ords), dtype=np.int32)

    for si, seg in enumerate(SEGMENTS):
        for oi, d in enumerate(ORDINALS):
            cls_name = f"d{d}_{seg}"
            matrix[si, oi] = counts.get(cls_name, 0)

    max_count = max(matrix.max(), 1)

    # Render
    cell_w, cell_h = 80, 18
    label_w = 100
    header_h = 30
    w = label_w + n_ords * cell_w + 20
    h = header_h + n_segs * cell_h + 40

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)

    # Header
    for oi, d in enumerate(ORDINALS):
        x = label_w + oi * cell_w + cell_w // 2 - 15
        cv2.putText(img, f"Dart {d}", (x, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Rows
    for si, seg in enumerate(SEGMENTS):
        y = header_h + si * cell_h
        # Row label
        cv2.putText(img, seg, (5, y + cell_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        for oi in range(n_ords):
            x = label_w + oi * cell_w
            count = matrix[si, oi]

            # Color: black (0) → green (some) → yellow (many)
            if count == 0:
                color = (40, 40, 40)
                text_color = (80, 80, 80)
            else:
                intensity = min(count / max(max_count * 0.5, 1), 1.0)
                if intensity < 0.5:
                    # dark to green
                    g = int(100 + 155 * (intensity * 2))
                    color = (0, g, 0)
                else:
                    # green to yellow
                    r = int(255 * ((intensity - 0.5) * 2))
                    color = (0, 255, r)
                text_color = (255, 255, 255)

            cv2.rectangle(img, (x + 1, y + 1), (x + cell_w - 1, y + cell_h - 1), color, -1)
            cv2.putText(img, str(count), (x + cell_w // 2 - 8, y + cell_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)

    # Total at bottom
    total = sum(counts.values())
    cv2.putText(img, f"Total: {total}  |  Classes seen: {sum(1 for r in matrix.flatten() if r > 0)}/{NUM_CLASSES}",
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
