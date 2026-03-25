#!/usr/bin/env python3
"""
convert_v1_data.py — Convert v1 training data (189-class, point annotations)
to v2 format (1-class detection, auto-expanded bounding boxes).

V1 format:  class_id center_x center_y width height  (189 classes, fixed tiny box)
V2 format:  0 center_x center_y width height         (1 class "dart", larger box)

The tip coordinates from v1 are preserved as metadata. Bounding boxes are
expanded along the radial direction (away from board center) to approximate
the full dart (tip + shaft).

Also validates geometry-based classification against human labels from v1.

Usage:
    uv run python convert_v1_data.py                    # Dry run (stats only)
    uv run python convert_v1_data.py --convert          # Convert dataset
    uv run python convert_v1_data.py --geometry-check   # Validate geometry vs labels
"""

import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import cv2
import numpy as np

import config
from board import apply_homography, pixel_to_polar, get_sector, get_ring
from classes import ID_TO_CLASS, parse_class_name

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

V1_IMAGES = config.PROJECT_ROOT / "data" / "training" / "images"
V1_LABELS = config.PROJECT_ROOT / "data" / "training" / "labels"
V2_DIR = config.PROJECT_ROOT / "data" / "training_v2"
V2_IMAGES = V2_DIR / "images"
V2_LABELS = V2_DIR / "labels"
V2_META = V2_DIR / "metadata"

# Auto-expand parameters — resolved at call time from config fractions.
# Module-level aliases kept for backward compat with existing calls;
# auto_expand_bbox() recomputes per-image below.
EXPAND_AWAY_PX = 80    # fallback, overridden per-image
EXPAND_TOWARD_PX = 15
EXPAND_LATERAL_PX = 25
MIN_BOX_PX = 60


def load_homography():
    """Load board homography if available."""
    path = config.BOARD_HOMOGRAPHY_PATH
    if not path.exists():
        return None
    data = np.load(str(path))
    return data["homography"]


def get_board_center_pixel(H):
    """Get the board center in camera pixel space (inverse homography)."""
    H_inv = np.linalg.inv(H)
    cx, cy = config.CANONICAL_CENTER
    pts = np.array([[[cx, cy]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pts, H_inv)
    return float(transformed[0][0][0]), float(transformed[0][0][1])


def auto_expand_bbox(tip_x, tip_y, img_w, img_h, board_center=None):
    """Expand a point annotation to a full-dart bounding box.

    Uses the radial direction from tip toward board center to estimate
    dart orientation. Expands more away from center (shaft) than toward.

    Args:
        tip_x, tip_y: Tip pixel coordinates (absolute).
        img_w, img_h: Image dimensions.
        board_center: (cx, cy) of board center in pixel space, or None.

    Returns:
        (cx, cy, w, h) normalized bounding box [0, 1].
    """
    # Resolve expansion constants from resolution-relative fractions
    expand_away = int(round(config.BBOX_EXPAND_AWAY_FRAC * img_w))
    expand_toward = int(round(config.BBOX_EXPAND_TOWARD_FRAC * img_w))
    expand_lateral = int(round(config.BBOX_EXPAND_LATERAL_FRAC * img_h))
    min_box = int(round(config.BBOX_MIN_SIZE_FRAC * img_w))

    if board_center is not None:
        bcx, bcy = board_center
        # Direction from tip toward center (the way the dart points)
        dx = bcx - tip_x
        dy = bcy - tip_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 1:
            dx /= dist
            dy /= dist
        else:
            dx, dy = 0, -1  # default: point up
    else:
        # No homography — use image center as rough guess
        dx = img_w / 2 - tip_x
        dy = img_h / 2 - tip_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 1:
            dx /= dist
            dy /= dist
        else:
            dx, dy = 0, -1

    # Radial unit vector: (dx, dy) points toward center
    # Perpendicular: (-dy, dx)
    perp_x, perp_y = -dy, dx

    # Compute box corners by extending in each direction from tip
    # "away" = opposite of toward-center = shaft direction
    corners_x = []
    corners_y = []

    # Toward center (past tip, small margin)
    corners_x.append(tip_x + dx * expand_toward)
    corners_y.append(tip_y + dy * expand_toward)

    # Away from center (shaft + flights)
    corners_x.append(tip_x - dx * expand_away)
    corners_y.append(tip_y - dy * expand_away)

    # Lateral (perpendicular, both sides)
    corners_x.append(tip_x + perp_x * expand_lateral)
    corners_y.append(tip_y + perp_y * expand_lateral)
    corners_x.append(tip_x - perp_x * expand_lateral)
    corners_y.append(tip_y - perp_y * expand_lateral)

    # Axis-aligned bounding box from corners
    x1 = max(0, min(corners_x))
    y1 = max(0, min(corners_y))
    x2 = min(img_w, max(corners_x))
    y2 = min(img_h, max(corners_y))

    # Enforce minimum size
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < min_box:
        pad = (min_box - box_w) / 2
        x1 = max(0, x1 - pad)
        x2 = min(img_w, x2 + pad)
    if box_h < min_box:
        pad = (min_box - box_h) / 2
        y1 = max(0, y1 - pad)
        y2 = min(img_h, y2 + pad)

    # Convert to YOLO normalized format
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    return cx, cy, w, h


def geometry_classify(tip_x, tip_y, H):
    """Classify a dart tip using geometry (homography → polar → segment).

    Returns:
        Dict with sector, ring, segment name, or None if homography fails.
    """
    try:
        can_x, can_y = apply_homography((tip_x, tip_y), H)
        r, theta = pixel_to_polar(can_x, can_y)
        ring_name, multiplier = get_ring(r)

        if ring_name == "D-BULL":
            return {"sector": 25, "ring": "D_BULL", "segment": "D_BULL",
                    "r": r, "theta": theta}
        elif ring_name == "S-BULL":
            return {"sector": 25, "ring": "S_BULL", "segment": "S_BULL",
                    "r": r, "theta": theta}
        elif ring_name == "miss":
            return {"sector": 0, "ring": "MISS", "segment": "MISS",
                    "r": r, "theta": theta}
        else:
            sector = get_sector(theta)
            ring_code = {"single": "S", "double": "D", "triple": "T"}[ring_name]
            segment = f"{ring_code}{sector}"
            return {"sector": sector, "ring": ring_name, "segment": segment,
                    "r": r, "theta": theta}
    except Exception:
        return None


def sector_distance(s1, s2):
    """Minimum hops between two sectors on the dartboard (circular)."""
    if s1 == s2:
        return 0
    order = config.SECTOR_ORDER
    try:
        i1 = order.index(s1)
        i2 = order.index(s2)
    except ValueError:
        return 99  # bulls or miss
    d = abs(i1 - i2)
    return min(d, config.NUM_SECTORS - d)


# ---------------------------------------------------------------------------
# Geometry validation
# ---------------------------------------------------------------------------

def geometry_check():
    """Compare geometry classification to human v1 labels."""
    H = load_homography()
    if H is None:
        print("ERROR: No homography found. Run calibrate.py --board first.")
        return

    # Load crop ROI to offset coordinates
    crop_offset = (0, 0)
    if config.CROP_ROI_PATH.exists():
        roi = np.load(str(config.CROP_ROI_PATH))
        crop_data = roi["crop_roi"]
        crop_offset = (int(crop_data[0]), int(crop_data[1]))

    total = 0
    segment_match = 0
    sector_match = 0
    ring_match = 0
    sector_off_by_1 = 0
    mismatches = []

    for lf in sorted(V1_LABELS.glob("*.txt")):
        img_path = V1_IMAGES / f"{lf.stem}.png"
        if not img_path.exists():
            continue

        # Read image dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                cx_norm, cy_norm = float(parts[1]), float(parts[2])

                # Absolute tip coordinates in crop space
                tip_x = cx_norm * img_w
                tip_y = cy_norm * img_h

                # Add crop offset for homography (calibrated in full-frame space)
                tip_full_x = tip_x + crop_offset[0]
                tip_full_y = tip_y + crop_offset[1]

                # Human label
                v1_name = ID_TO_CLASS.get(cls_id)
                if v1_name is None:
                    continue
                info = parse_class_name(v1_name)
                human_segment = info["segment"]
                human_ring = info["ring"]
                human_sector = info.get("sector", 0)

                # Geometry guess
                geo = geometry_classify(tip_full_x, tip_full_y, H)
                if geo is None:
                    continue

                total += 1
                geo_segment = geo["segment"]

                # Compare
                if geo_segment == human_segment:
                    segment_match += 1
                    sector_match += 1
                    ring_match += 1
                else:
                    # Check ring
                    geo_ring_code = geo["ring"]
                    # Normalize ring names for comparison
                    human_ring_norm = human_segment
                    if human_segment not in ("MISS", "S_BULL", "D_BULL"):
                        human_ring_code = human_segment[0]
                    else:
                        human_ring_code = human_segment

                    if human_segment in ("MISS", "S_BULL", "D_BULL"):
                        human_sector_num = 0
                    else:
                        human_sector_num = int(human_segment[1:])

                    # Ring match check
                    ring_map = {"S": "single", "D": "double", "T": "triple",
                                "S_BULL": "S_BULL", "D_BULL": "D_BULL", "MISS": "MISS"}
                    if ring_map.get(human_ring_code) == geo["ring"] or human_ring_code == geo_ring_code:
                        ring_match += 1

                    # Sector match check
                    geo_sector = geo["sector"]
                    if geo_sector == human_sector_num:
                        sector_match += 1
                    elif sector_distance(geo_sector, human_sector_num) <= 1:
                        sector_off_by_1 += 1

                    if sector_distance(geo_sector, human_sector_num) > 1:
                        mismatches.append({
                            "file": lf.stem,
                            "human": human_segment,
                            "geo": geo_segment,
                            "r": geo["r"],
                            "theta": geo["theta"],
                            "sector_dist": sector_distance(geo_sector, human_sector_num),
                        })

    print(f"=== Geometry vs Human Labels ({total} annotations) ===")
    print()
    print(f"  Exact segment match:  {segment_match:4d} / {total}  ({segment_match/total*100:.1f}%)")
    print(f"  Sector correct:       {sector_match:4d} / {total}  ({sector_match/total*100:.1f}%)")
    print(f"  Sector within ±1:     {sector_match + sector_off_by_1:4d} / {total}  ({(sector_match + sector_off_by_1)/total*100:.1f}%)")
    print(f"  Ring correct:         {ring_match:4d} / {total}  ({ring_match/total*100:.1f}%)")
    print()

    if mismatches:
        print(f"  Sector off by >1: {len(mismatches)} annotations")
        print()
        # Show worst offenders
        by_human = Counter(m["human"] for m in mismatches)
        print("  Most confused segments (human label → count of sector misses >1):")
        for seg, count in by_human.most_common(10):
            print(f"    {seg:10s}: {count}")
        print()
        print("  Sample mismatches:")
        for m in mismatches[:10]:
            print(f"    {m['file']:20s} human={m['human']:8s} geo={m['geo']:8s} "
                  f"r={m['r']:.1f}mm θ={m['theta']:.1f}° dist={m['sector_dist']}")


# ---------------------------------------------------------------------------
# Data conversion
# ---------------------------------------------------------------------------

def label_trust_level(human_segment, geo, H):
    """Determine trust level of a v1 label by comparing to geometry.

    Returns:
        (trust, best_segment) where trust is one of:
        - "exact":      geometry and human agree exactly
        - "near_wire":  same ring, sector within ±1 (near wire ambiguity)
        - "ring_ambig": same sector, different ring (near ring boundary)
        - "suspect":    sector disagree by >1, or special segment mismatch
    """
    if geo is None:
        return "no_geo", human_segment

    geo_seg = geo["segment"]
    if geo_seg == human_segment:
        return "exact", human_segment

    # Special segments: must match exactly
    if human_segment in ("MISS", "S_BULL", "D_BULL") or geo_seg in ("MISS", "S_BULL", "D_BULL"):
        return "suspect", human_segment

    human_ring = human_segment[0]
    geo_ring = geo_seg[0]
    human_sector = int(human_segment[1:])
    geo_sector = int(geo_seg[1:])
    sd = sector_distance(human_sector, geo_sector)

    if sd <= 1 and human_ring == geo_ring:
        # Near wire — trust human (they clicked the tip, closer to truth)
        return "near_wire", human_segment
    elif sd == 0 and human_ring != geo_ring:
        # Same sector, different ring — trust human ring judgment
        return "ring_ambig", human_segment
    else:
        return "suspect", human_segment


def convert_dataset():
    """Convert v1 dataset to v2 format.

    All annotations are converted for 1-class detection (position is always valid).
    Metadata records the trust level and best-guess segment label for each dart,
    which can be used to filter training data for future classifiers.
    """
    H = load_homography()
    board_center = get_board_center_pixel(H) if H is not None else None

    # Load crop offset
    crop_offset = (0, 0)
    if config.CROP_ROI_PATH.exists():
        roi = np.load(str(config.CROP_ROI_PATH))
        crop_data = roi["crop_roi"]
        crop_offset = (int(crop_data[0]), int(crop_data[1]))

    # Adjust board center to crop space
    if board_center is not None:
        board_center = (board_center[0] - crop_offset[0],
                        board_center[1] - crop_offset[1])

    # Create output directories
    V2_IMAGES.mkdir(parents=True, exist_ok=True)
    V2_LABELS.mkdir(parents=True, exist_ok=True)
    V2_META.mkdir(parents=True, exist_ok=True)

    label_files = sorted(V1_LABELS.glob("*.txt"))
    converted = 0
    skipped = 0
    trust_counts = Counter()
    meta_records = []

    for lf in label_files:
        stem = lf.stem
        img_src = V1_IMAGES / f"{stem}.png"
        if not img_src.exists():
            skipped += 1
            continue

        # Read image for dimensions
        img = cv2.imread(str(img_src))
        if img is None:
            skipped += 1
            continue
        img_h, img_w = img.shape[:2]

        v2_lines = []
        frame_meta = {"filename": f"{stem}.png", "darts": []}

        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                cx_norm, cy_norm = float(parts[1]), float(parts[2])

                # Original v1 class info (for metadata)
                v1_name = ID_TO_CLASS.get(cls_id)
                if v1_name is None:
                    continue
                info = parse_class_name(v1_name)

                # Tip in absolute pixels
                tip_x = cx_norm * img_w
                tip_y = cy_norm * img_h

                # Auto-expand to full-dart bbox
                new_cx, new_cy, new_w, new_h = auto_expand_bbox(
                    tip_x, tip_y, img_w, img_h, board_center
                )

                # V2 label: class 0 ("dart"), expanded bbox
                # ALL annotations are kept for detection — position is always valid
                v2_lines.append(f"0 {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")

                # Geometry classification and trust assessment
                geo = None
                if H is not None:
                    tip_full_x = tip_x + crop_offset[0]
                    tip_full_y = tip_y + crop_offset[1]
                    geo = geometry_classify(tip_full_x, tip_full_y, H)

                trust, best_segment = label_trust_level(info["segment"], geo, H)
                trust_counts[trust] += 1

                frame_meta["darts"].append({
                    "tip_x": round(tip_x, 1),
                    "tip_y": round(tip_y, 1),
                    "tip_x_norm": round(cx_norm, 6),
                    "tip_y_norm": round(cy_norm, 6),
                    "v1_class": v1_name,
                    "v1_segment": info["segment"],
                    "v1_ordinal": info["ordinal"],
                    "geo_segment": geo["segment"] if geo else None,
                    "geo_r_mm": round(geo["r"], 1) if geo else None,
                    "geo_theta": round(geo["theta"], 1) if geo else None,
                    "best_segment": best_segment,
                    "trust": trust,
                })

        if v2_lines:
            # Copy image (symlink to save space)
            img_dst = V2_IMAGES / f"{stem}.png"
            if not img_dst.exists():
                os.symlink(img_src.resolve(), img_dst)

            # Write v2 label
            with open(V2_LABELS / f"{stem}.txt", "w") as f:
                f.write("\n".join(v2_lines) + "\n")

            meta_records.append(frame_meta)
            converted += 1

    # Write metadata
    with open(V2_META / "annotations.jsonl", "w") as f:
        for rec in meta_records:
            f.write(json.dumps(rec) + "\n")

    # Write dataset.yaml for YOLO
    dataset_yaml = V2_DIR / "dataset.yaml"
    with open(dataset_yaml, "w") as f:
        f.write(f"path: {V2_DIR.resolve()}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("names:\n")
        f.write("  0: dart\n")

    total_annotations = sum(trust_counts.values())
    trusted = trust_counts["exact"] + trust_counts["near_wire"] + trust_counts["ring_ambig"]

    print(f"=== V1 → V2 Conversion Complete ===")
    print(f"  Converted: {converted} frames")
    print(f"  Skipped:   {skipped} frames (missing image)")
    print(f"  Output:    {V2_DIR}")
    print(f"  Labels:    {V2_LABELS} (1-class 'dart', expanded bbox)")
    print(f"  Metadata:  {V2_META}/annotations.jsonl")
    print(f"  YOLO config: {dataset_yaml}")
    print()
    print(f"  === Label Trust Summary ===")
    print(f"  Total annotations:  {total_annotations}")
    print(f"  For detection:      {total_annotations} (all usable — position is valid)")
    print(f"  For classification: {trusted} trusted ({trusted/total_annotations*100:.1f}%)")
    for level in ["exact", "near_wire", "ring_ambig", "suspect", "no_geo"]:
        c = trust_counts.get(level, 0)
        if c:
            print(f"    {level:12s}: {c:5d}  ({c/total_annotations*100:.1f}%)")
    print()

    # Summary stats
    box_widths = []
    box_heights = []
    for lf in V2_LABELS.glob("*.txt"):
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    box_widths.append(float(parts[3]))
                    box_heights.append(float(parts[4]))

    if box_widths:
        avg_w_px = sum(box_widths) / len(box_widths) * 1080
        avg_h_px = sum(box_heights) / len(box_heights) * 1080
        print(f"  Avg bbox (norm): {sum(box_widths)/len(box_widths):.4f} x {sum(box_heights)/len(box_heights):.4f}")
        print(f"  Avg bbox (px@1080): ~{avg_w_px:.0f} x {avg_h_px:.0f}")
        print(f"  vs v1 fixed box:   ~22 x 28")


# ---------------------------------------------------------------------------
# Dry run stats
# ---------------------------------------------------------------------------

def dry_run():
    """Show stats about what conversion would do."""
    print("=== V1 Dataset Stats (Dry Run) ===")
    print()

    label_files = sorted(V1_LABELS.glob("*.txt"))
    image_files = sorted(V1_IMAGES.glob("*.png"))
    print(f"  Label files: {len(label_files)}")
    print(f"  Image files: {len(image_files)}")

    # Class distribution (by segment, collapsing ordinals)
    seg_counts = Counter()
    total = 0
    for lf in label_files:
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    name = ID_TO_CLASS.get(cls_id)
                    if name:
                        info = parse_class_name(name)
                        seg_counts[info["segment"]] += 1
                        total += 1

    print(f"  Total annotations: {total}")
    print(f"  Unique segments: {len(seg_counts)} / 63")
    print(f"  → All become class 0 ('dart') in v2")
    print()

    # Ring distribution
    ring_counts = Counter()
    for seg, count in seg_counts.items():
        if seg == "MISS":
            ring_counts["MISS"] += count
        elif seg in ("S_BULL", "D_BULL"):
            ring_counts[seg] += count
        else:
            ring_counts[seg[0]] += count

    print("  Ring distribution (for future ring classifier):")
    for ring in ["S", "D", "T", "S_BULL", "D_BULL", "MISS"]:
        c = ring_counts.get(ring, 0)
        print(f"    {ring:8s}: {c:5d}  ({c/total*100:.1f}%)")

    print()
    print(f"  V2 output would go to: {V2_DIR}")
    print(f"  Run with --convert to execute, or --geometry-check to validate geometry.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert v1 training data to v2 format")
    parser.add_argument("--convert", action="store_true", help="Execute conversion")
    parser.add_argument("--geometry-check", action="store_true",
                        help="Validate geometry classification vs human labels")
    args = parser.parse_args()

    if args.geometry_check:
        geometry_check()
    elif args.convert:
        convert_dataset()
    else:
        dry_run()
