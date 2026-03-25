#!/usr/bin/env python3
"""
analyze_resolution.py — Analyze image quality and resolution tradeoffs.

Compares sharpness, frequency content, and effective YOLO input between
different camera resolutions to validate the resolution choice.

Also analyzes bounding box expansion and confidence model calibration
against actual training data.

Usage:
    python analyze_resolution.py                # Full analysis
    python analyze_resolution.py --bbox         # Bounding box analysis only
    python analyze_resolution.py --confidence   # Confidence model analysis only
    python analyze_resolution.py --sharpness    # Sharpness/resolution only
"""

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

import config


# ---------------------------------------------------------------------------
# Sharpness / resolution analysis
# ---------------------------------------------------------------------------

def laplacian_sharpness(gray):
    """Laplacian variance — higher means sharper edges."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def load_images_by_resolution(img_dir, max_per_res=20):
    """Load sample images grouped by resolution.

    Returns:
        Dict of (w, h) -> list of (path, image) tuples.
    """
    by_res = {}
    for p in sorted(img_dir.glob("*.png")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        key = (w, h)
        if key not in by_res:
            by_res[key] = []
        if len(by_res[key]) < max_per_res:
            by_res[key].append((p, img))
    return by_res


def analyze_sharpness(v1_dir=None):
    """Compare sharpness across resolutions to detect sensor upscaling."""
    if v1_dir is None:
        v1_dir = config.PROJECT_ROOT / "data" / "training" / "images"

    if not v1_dir.exists():
        print("No training images found.")
        return

    by_res = load_images_by_resolution(v1_dir)

    print("=" * 70)
    print("RESOLUTION & SHARPNESS ANALYSIS")
    print("=" * 70)
    print(f"\nImage directory: {v1_dir}")

    for (w, h), imgs in sorted(by_res.items(), key=lambda x: -x[0][0]):
        print(f"\n  {w}x{h}: {len(imgs)} sample images")

    # Compute sharpness at native resolution
    print("\nLaplacian variance (higher = sharper edges):")
    native_results = {}
    for (w, h), imgs in sorted(by_res.items(), key=lambda x: -x[0][0]):
        vals = [laplacian_sharpness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                for _, img in imgs]
        mean_s = np.mean(vals)
        native_results[(w, h)] = mean_s
        print(f"  {w}x{h} native:       {mean_s:>8.1f}")

    # If we have multiple resolutions, downscale the largest to each smaller
    # to see if the larger resolution contains real detail
    resolutions = sorted(by_res.keys(), key=lambda x: -x[0])
    if len(resolutions) >= 2:
        largest = resolutions[0]
        print(f"\nDownscale test (from {largest[0]}x{largest[1]}):")

        for target_res in resolutions[1:]:
            tw, th = target_res
            down_vals = []
            for _, img in by_res[largest]:
                down = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(down, cv2.COLOR_BGR2GRAY)
                down_vals.append(laplacian_sharpness(gray))
            mean_down = np.mean(down_vals)
            ratio = native_results[largest] / mean_down if mean_down > 0 else 0
            print(f"  {largest[0]}x{largest[1]} → {tw}x{th}:  {mean_down:>8.1f}  "
                  f"(ratio: {ratio:.2f}x)")

            native_at_target = native_results.get(target_res)
            if native_at_target is not None:
                print(f"  {tw}x{th} native:          {native_at_target:>8.1f}")

        print(f"\n  Interpretation:")
        ratio = native_results[largest] / mean_down if mean_down > 0 else 0
        if ratio < 1.0:
            print(f"    Ratio {ratio:.2f}x < 1.0 → larger resolution is SOFTER than downscaled.")
            print(f"    This indicates sensor upscaling — the extra pixels are interpolated,")
            print(f"    not real optical detail. The lower resolution is sufficient.")
        elif ratio < 1.5:
            print(f"    Ratio {ratio:.2f}x ≈ 1.0 → minimal real detail at higher resolution.")
            print(f"    Marginal benefit from the larger frame.")
        else:
            print(f"    Ratio {ratio:.2f}x > 1.5 → higher resolution captures real detail.")
            print(f"    Consider using the larger resolution if latency permits.")

    # YOLO effective input
    print(f"\n{'=' * 70}")
    print("YOLO EFFECTIVE INPUT")
    print("=" * 70)

    yolo_imgsz = 640
    print(f"\nYOLO imgsz={yolo_imgsz}")

    # Estimate px/mm at board surface (crop covers ~400mm of board)
    board_visible_mm = 400  # rough estimate including margins

    for (w, h) in resolutions:
        scale = yolo_imgsz / max(w, h)
        yw, yh = int(w * scale), int(h * scale)
        padding_pct = (yolo_imgsz ** 2 - yw * yh) / yolo_imgsz ** 2 * 100
        ppmm = w / board_visible_mm
        tip_px = 2 * ppmm  # 2mm dart tip diameter
        tip_yolo = tip_px * scale

        print(f"\n  {w}x{h}:")
        print(f"    Scale factor: {scale:.3f} → {yw}x{yh} in {yolo_imgsz}x{yolo_imgsz}")
        print(f"    Letterbox padding: {padding_pct:.0f}%")
        print(f"    Board px/mm: {ppmm:.1f}")
        print(f"    2mm dart tip: {tip_px:.1f}px native → {tip_yolo:.1f}px in YOLO")

    if len(resolutions) >= 2:
        # Compare tip sizes in YOLO input
        tips = []
        for (w, h) in resolutions:
            scale = yolo_imgsz / max(w, h)
            ppmm = w / board_visible_mm
            tips.append(2 * ppmm * scale)

        print(f"\n  YOLO tip size difference: {abs(tips[0] - tips[-1]):.1f}px "
              f"({abs(tips[0] - tips[-1]) / tips[0] * 100:.0f}%)")
        if abs(tips[0] - tips[-1]) < 0.5:
            print(f"  → Negligible difference. YOLO receives essentially identical input.")


# ---------------------------------------------------------------------------
# Bounding box analysis
# ---------------------------------------------------------------------------

def analyze_bboxes():
    """Analyze bounding box sizes and expansion constant calibration."""
    v2_labels = config.PROJECT_ROOT / "data" / "training_v2" / "labels"
    v2_images = config.PROJECT_ROOT / "data" / "training_v2" / "images"
    v1_images = config.PROJECT_ROOT / "data" / "training" / "images"

    if not v2_labels.exists():
        print("No v2 training labels found.")
        return

    # Detect image resolutions
    by_res = load_images_by_resolution(v1_images, max_per_res=5)
    resolutions = sorted(by_res.keys(), key=lambda x: -x[0])

    print("=" * 70)
    print("BOUNDING BOX EXPANSION ANALYSIS")
    print("=" * 70)

    # Load v2 bbox dimensions
    widths_norm, heights_norm = [], []
    for lf in v2_labels.glob("*.txt"):
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    widths_norm.append(float(parts[3]))
                    heights_norm.append(float(parts[4]))

    if not widths_norm:
        print("No v2 labels found.")
        return

    print(f"\nV2 labels: {len(widths_norm)} annotations")

    # Show bbox sizes at each known resolution
    print(f"\nResolution-relative fractions (config):")
    print(f"  BBOX_EXPAND_AWAY_FRAC   = {config.BBOX_EXPAND_AWAY_FRAC}")
    print(f"  BBOX_EXPAND_TOWARD_FRAC = {config.BBOX_EXPAND_TOWARD_FRAC}")
    print(f"  BBOX_EXPAND_LATERAL_FRAC = {config.BBOX_EXPAND_LATERAL_FRAC}")
    print(f"  BBOX_MIN_SIZE_FRAC      = {config.BBOX_MIN_SIZE_FRAC}")

    print(f"\nEffective pixel values at each resolution:")
    print(f"  {'Resolution':<15s} {'Away':>6s} {'Toward':>7s} {'Lateral':>8s} {'MinBox':>7s}")
    print(f"  {'-'*45}")
    for (w, h) in resolutions:
        away = int(round(config.BBOX_EXPAND_AWAY_FRAC * w))
        toward = int(round(config.BBOX_EXPAND_TOWARD_FRAC * w))
        lateral = int(round(config.BBOX_EXPAND_LATERAL_FRAC * h))
        minbox = int(round(config.BBOX_MIN_SIZE_FRAC * w))
        print(f"  {w}x{h:<10d} {away:>6d} {toward:>7d} {lateral:>8d} {minbox:>7d}")

    # Actual v2 bbox dimensions
    w_arr = np.array(widths_norm)
    h_arr = np.array(heights_norm)

    print(f"\nActual v2 bbox dimensions (normalized):")
    print(f"  Width:  min={w_arr.min():.4f}  median={np.median(w_arr):.4f}  "
          f"p90={np.percentile(w_arr, 90):.4f}  max={w_arr.max():.4f}")
    print(f"  Height: min={h_arr.min():.4f}  median={np.median(h_arr):.4f}  "
          f"p90={np.percentile(h_arr, 90):.4f}  max={h_arr.max():.4f}")

    # At each resolution
    for (w, h) in resolutions:
        wp = w_arr * w
        hp = h_arr * h
        minbox = int(round(config.BBOX_MIN_SIZE_FRAC * w))
        small_w = (wp <= minbox + 2).sum()
        small_h = (hp <= minbox + 2).sum()
        print(f"\n  At {w}x{h} (minbox={minbox}px):")
        print(f"    Width (px):  median={np.median(wp):.0f}  p90={np.percentile(wp, 90):.0f}")
        print(f"    Height (px): median={np.median(hp):.0f}  p90={np.percentile(hp, 90):.0f}")
        print(f"    At/near minbox: w={small_w} ({small_w / len(wp) * 100:.0f}%)  "
              f"h={small_h} ({small_h / len(hp) * 100:.0f}%)")


# ---------------------------------------------------------------------------
# Confidence model analysis
# ---------------------------------------------------------------------------

def analyze_confidence():
    """Analyze the wire proximity confidence model against real data."""
    meta_path = (config.PROJECT_ROOT / "data" / "training_v2"
                 / "metadata" / "annotations.jsonl")
    if not meta_path.exists():
        print("No v2 metadata found.")
        return

    darts = []
    with open(meta_path) as f:
        for line in f:
            rec = json.loads(line)
            for d in rec["darts"]:
                if d.get("geo_r_mm") is not None and d.get("geo_theta") is not None:
                    darts.append(d)

    if not darts:
        print("No geometry data in metadata.")
        return

    r_values = np.array([d["geo_r_mm"] for d in darts])
    theta_values = np.array([d["geo_theta"] for d in darts])

    ring_boundaries = [
        config.INNER_BULL_RADIUS, config.OUTER_BULL_RADIUS,
        config.TRIPLE_INNER_RADIUS, config.TRIPLE_OUTER_RADIUS,
        config.DOUBLE_INNER_RADIUS, config.DOUBLE_OUTER_RADIUS,
    ]

    print("=" * 70)
    print("CONFIDENCE MODEL ANALYSIS (arc-length based)")
    print("=" * 70)

    print(f"\nDarts with geometry: {len(darts)}")
    print(f"Radius (mm): min={r_values.min():.1f}  median={np.median(r_values):.1f}  "
          f"max={r_values.max():.1f}")

    # Ring wire distances
    ring_dists = np.array([min(abs(r - b) for b in ring_boundaries) for r in r_values])

    print(f"\nRing wire distance (mm):")
    print(f"  min={ring_dists.min():.1f}  p10={np.percentile(ring_dists, 10):.1f}  "
          f"median={np.median(ring_dists):.1f}  p90={np.percentile(ring_dists, 90):.1f}")
    print(f"  Within RING_MARGIN ({config.RING_MARGIN_MM}mm): "
          f"{(ring_dists < config.RING_MARGIN_MM).sum()} "
          f"({(ring_dists < config.RING_MARGIN_MM).sum() / len(ring_dists) * 100:.1f}%)")

    # Sector wire distances (angular + arc-length)
    sector_deg = []
    sector_arc = []
    for r, theta in zip(r_values, theta_values):
        offset = (theta + config.SECTOR_BOUNDARY_OFFSET) % config.SECTOR_SPAN_DEG
        swd = min(offset, config.SECTOR_SPAN_DEG - offset)
        sector_deg.append(swd)
        sector_arc.append(r * math.radians(swd))

    sector_deg = np.array(sector_deg)
    sector_arc = np.array(sector_arc)

    print(f"\nSector wire distance:")
    print(f"  Angular (°):  min={sector_deg.min():.1f}  median={np.median(sector_deg):.1f}  "
          f"p90={np.percentile(sector_deg, 90):.1f}")
    print(f"  Arc-len (mm): min={sector_arc.min():.1f}  median={np.median(sector_arc):.1f}  "
          f"p90={np.percentile(sector_arc, 90):.1f}")
    print(f"  Arc within WIRE_AMBIGUITY ({config.WIRE_AMBIGUITY_THRESHOLD_MM}mm): "
          f"{(sector_arc < config.WIRE_AMBIGUITY_THRESHOLD_MM).sum()} "
          f"({(sector_arc < config.WIRE_AMBIGUITY_THRESHOLD_MM).sum() / len(sector_arc) * 100:.1f}%)")

    # Confidence computation
    sector_conf = np.minimum(sector_arc / config.WIRE_CONF_DIVISOR_MM, 1.0)
    ring_conf = np.minimum(ring_dists / config.WIRE_CONF_DIVISOR_MM, 1.0)
    combined = np.minimum(sector_conf, ring_conf)

    print(f"\nCombined confidence:")
    print(f"  min={combined.min():.2f}  p10={np.percentile(combined, 10):.2f}  "
          f"median={np.median(combined):.2f}  p90={np.percentile(combined, 90):.2f}")
    print(f"  Below {config.GEO_CONF_MODERATE} [?]: "
          f"{(combined < config.GEO_CONF_MODERATE).sum()} "
          f"({(combined < config.GEO_CONF_MODERATE).sum() / len(combined) * 100:.1f}%)")
    mid_mask = (combined >= config.GEO_CONF_MODERATE) & (combined < config.GEO_CONF_HIGH)
    print(f"  {config.GEO_CONF_MODERATE}-{config.GEO_CONF_HIGH} [~]: "
          f"{mid_mask.sum()} ({mid_mask.sum() / len(combined) * 100:.1f}%)")
    print(f"  Above {config.GEO_CONF_HIGH} (OK): "
          f"{(combined >= config.GEO_CONF_HIGH).sum()} "
          f"({(combined >= config.GEO_CONF_HIGH).sum() / len(combined) * 100:.1f}%)")

    # Bottleneck
    sector_bottleneck = (sector_conf < ring_conf).sum()
    ring_bottleneck = (ring_conf < sector_conf).sum()
    tied = (sector_conf == ring_conf).sum()
    print(f"\n  Bottleneck: sector={sector_bottleneck} ({sector_bottleneck / len(combined) * 100:.0f}%)  "
          f"ring={ring_bottleneck} ({ring_bottleneck / len(combined) * 100:.0f}%)  "
          f"tied={tied} ({tied / len(combined) * 100:.0f}%)")

    # Physical context
    print(f"\n  Wire config: WIRE_CONF_DIVISOR_MM={config.WIRE_CONF_DIVISOR_MM}  "
          f"WIRE_AMBIGUITY_THRESHOLD_MM={config.WIRE_AMBIGUITY_THRESHOLD_MM}")
    print(f"  Standard wire width: ~1.6mm (SWB)")

    # Confidence vs label trust
    trust_buckets = {"low": Counter(), "mid": Counter(), "high": Counter()}
    for i, d in enumerate(darts):
        trust = d.get("trust", "?")
        c = combined[i]
        if c < config.GEO_CONF_MODERATE:
            trust_buckets["low"][trust] += 1
        elif c < config.GEO_CONF_HIGH:
            trust_buckets["mid"][trust] += 1
        else:
            trust_buckets["high"][trust] += 1

    print(f"\n  Confidence vs label trust (validation):")
    for level in ["low", "mid", "high"]:
        total = sum(trust_buckets[level].values())
        if total == 0:
            continue
        exact = trust_buckets[level].get("exact", 0)
        suspect = trust_buckets[level].get("suspect", 0)
        print(f"    {level:>4s} (n={total:4d}): exact={exact / total * 100:.0f}%  "
              f"suspect={suspect / total * 100:.0f}%")

    # Arc-length by ring region (shows radius-dependent behavior)
    print(f"\n  Arc-length by ring region:")
    regions = [
        ("Bull", 0, config.OUTER_BULL_RADIUS),
        ("Inner single", config.OUTER_BULL_RADIUS, config.TRIPLE_INNER_RADIUS),
        ("Triple", config.TRIPLE_INNER_RADIUS, config.TRIPLE_OUTER_RADIUS),
        ("Outer single", config.TRIPLE_OUTER_RADIUS, config.DOUBLE_INNER_RADIUS),
        ("Double", config.DOUBLE_INNER_RADIUS, config.DOUBLE_OUTER_RADIUS),
        ("Miss", config.DOUBLE_OUTER_RADIUS, 999),
    ]
    for name, r_lo, r_hi in regions:
        mask = (r_values >= r_lo) & (r_values < r_hi)
        if mask.sum() == 0:
            continue
        arcs = sector_arc[mask]
        confs = sector_conf[mask]
        deg = sector_deg[mask]
        print(f"    {name:<14s} (n={mask.sum():4d}): "
              f"median arc={np.median(arcs):>5.1f}mm  "
              f"at 1° from wire: arc={np.median(r_values[mask]) * math.radians(1):.1f}mm  "
              f"conf={np.median(confs):.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze resolution quality, bbox calibration, and confidence model")
    parser.add_argument("--sharpness", action="store_true",
                        help="Sharpness/resolution analysis only")
    parser.add_argument("--bbox", action="store_true",
                        help="Bounding box expansion analysis only")
    parser.add_argument("--confidence", action="store_true",
                        help="Confidence model analysis only")
    args = parser.parse_args()

    run_all = not (args.sharpness or args.bbox or args.confidence)

    if run_all or args.sharpness:
        analyze_sharpness()
        print()

    if run_all or args.bbox:
        analyze_bboxes()
        print()

    if run_all or args.confidence:
        analyze_confidence()


if __name__ == "__main__":
    main()
