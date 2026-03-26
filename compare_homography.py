#!/usr/bin/env python3
"""
compare_homography.py — Compare homography predictions against YOLO labels.

Takes labeled training data (images + YOLO labels) and compares what the
homography would predict for each dart tip vs what the human labeled it as.

Modes:
    --report    Print summary of matches/misses
    --browse    Interactive browser showing each dart with overlay

Usage:
    python compare_homography.py --report
    python compare_homography.py --browse
    python compare_homography.py --report --browse   # both
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import config
import board
from classes import ID_TO_CLASS, parse_class_name
from window_manager import create_window, save_window_sizes


def _draw_homography_overlay(display, homography, crop_offset=(0, 0), alpha=0.4):
    """Draw the board geometry overlay on camera image using inverse homography.

    Transforms ring circles, sector lines, and number labels from canonical
    board space back to camera pixel space so you can see where the homography
    thinks each segment is.
    """
    import math

    H_inv = np.linalg.inv(homography)
    cx, cy = config.CANONICAL_CENTER
    ox, oy = crop_offset

    def canon_to_cam(px, py):
        """Transform canonical board coords to cropped camera coords."""
        pts = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, H_inv)
        cam_x = float(transformed[0][0][0]) - ox
        cam_y = float(transformed[0][0][1]) - oy
        return int(round(cam_x)), int(round(cam_y))

    # Draw ring circles as polygons (circles warp to ellipses under perspective)
    ring_defs = [
        (config.INNER_BULL_RADIUS,   (0, 255, 0),   1),
        (config.OUTER_BULL_RADIUS,   (0, 255, 0),   1),
        (config.TRIPLE_INNER_RADIUS, (0, 150, 255), 1),
        (config.TRIPLE_OUTER_RADIUS, (0, 150, 255), 1),
        (config.DOUBLE_INNER_RADIUS, (0, 0, 255),   1),
        (config.DOUBLE_OUTER_RADIUS, (0, 0, 255),   2),
    ]

    for radius, color, thickness in ring_defs:
        pts = []
        for deg in range(0, 360, 3):
            rad = math.radians(deg)
            bx = cx + radius * math.sin(rad)
            by = cy - radius * math.cos(rad)
            pts.append(canon_to_cam(bx, by))
        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(display, [pts], isClosed=True, color=color, thickness=thickness)

    # Sector dividing lines
    for i in range(20):
        boundary_deg = (i * 18 - 9) % 360
        rad = math.radians(boundary_deg)
        # Line from bull to outer double
        inner_r = config.OUTER_BULL_RADIUS
        outer_r = config.DOUBLE_OUTER_RADIUS
        ix = cx + inner_r * math.sin(rad)
        iy = cy - inner_r * math.cos(rad)
        ex = cx + outer_r * math.sin(rad)
        ey = cy - outer_r * math.cos(rad)
        p1 = canon_to_cam(ix, iy)
        p2 = canon_to_cam(ex, ey)
        cv2.line(display, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)

    # Sector number labels
    label_r = config.DOUBLE_OUTER_RADIUS + 12
    for i, sector_num in enumerate(config.SECTOR_ORDER):
        angle_deg = i * 18
        rad = math.radians(angle_deg)
        lx = cx + label_r * math.sin(rad)
        ly = cy - label_r * math.cos(rad)
        pt = canon_to_cam(lx, ly)
        cv2.putText(display, str(sector_num), (pt[0] - 8, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Bullseye center marker
    center = canon_to_cam(cx, cy)
    cv2.drawMarker(display, center, (0, 255, 0), cv2.MARKER_CROSS, 15, 2)


def _homography_predict(x, y, homography, crop_offset=(0, 0)):
    """Predict segment from pixel coords using homography.

    Returns segment string (e.g. "T20", "S5", "D_BULL") or "MISS".
    """
    full_x = x + crop_offset[0]
    full_y = y + crop_offset[1]
    try:
        score_info = board.score_from_camera((full_x, full_y), homography)
        ring = score_info["ring"]
        sector = score_info["sector"]
        if ring == "D-BULL":
            return "D_BULL"
        elif ring == "S-BULL":
            return "S_BULL"
        elif ring == "miss":
            return "MISS"
        else:
            ring_code = {"single": "S", "double": "D", "triple": "T"}[ring]
            return f"{ring_code}{sector}"
    except Exception:
        return "ERROR"


def load_labeled_darts(outdir):
    """Load all labeled darts from YOLO label files.

    Returns list of dicts:
        {filename, image_path, darts: [{x, y, class_id, class_name, segment, ordinal}]}
    """
    outdir = Path(outdir)
    img_dir = outdir / "images"
    label_dir = outdir / "labels"

    results = []
    for label_path in sorted(label_dir.glob("*.txt")):
        img_path = img_dir / (label_path.stem + ".png")
        if not img_path.exists():
            continue

        # Read image dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        darts = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy = float(parts[1]) * w, float(parts[2]) * h

                class_name = ID_TO_CLASS.get(cls_id, f"unknown_{cls_id}")
                info = parse_class_name(class_name)

                darts.append({
                    "x": int(cx),
                    "y": int(cy),
                    "class_id": cls_id,
                    "class_name": class_name,
                    "segment": info["segment"],
                    "ordinal": info["ordinal"],
                    "score": info["score"],
                    "label": info["label"],
                })

        if darts:
            results.append({
                "filename": img_path.name,
                "image_path": str(img_path),
                "img_w": w,
                "img_h": h,
                "darts": darts,
            })

    return results


def compare_all(labeled_data, homography, crop_offset=(0, 0)):
    """Compare homography predictions against labels.

    Returns list of comparison dicts with match/mismatch info.
    """
    comparisons = []
    for frame_data in labeled_data:
        for dart in frame_data["darts"]:
            predicted = _homography_predict(dart["x"], dart["y"], homography, crop_offset)
            actual = dart["segment"]
            match = predicted == actual

            # Check if sector matches even if ring is wrong
            sector_match = False
            if predicted not in ("MISS", "ERROR") and actual not in ("S_BULL", "D_BULL"):
                try:
                    pred_sector = int(predicted[1:]) if predicted[0] in "SDT" else None
                    actual_sector = int(actual[1:]) if actual[0] in "SDT" else None
                    sector_match = pred_sector == actual_sector
                except (ValueError, IndexError):
                    pass

            comparisons.append({
                "filename": frame_data["filename"],
                "image_path": frame_data["image_path"],
                "x": dart["x"],
                "y": dart["y"],
                "actual": actual,
                "predicted": predicted,
                "match": match,
                "sector_match": sector_match,
                "ordinal": dart["ordinal"],
                "actual_label": dart["label"],
            })

    return comparisons


def print_report(comparisons):
    """Print a summary report of homography accuracy."""
    total = len(comparisons)
    if total == 0:
        print("No labeled darts found.")
        return

    exact = sum(1 for c in comparisons if c["match"])
    sector_only = sum(1 for c in comparisons if c["sector_match"] and not c["match"])
    misses = sum(1 for c in comparisons if not c["match"] and not c["sector_match"])

    print(f"\n{'='*60}")
    print(f"HOMOGRAPHY vs LABELS — {total} darts")
    print(f"{'='*60}")
    print(f"  Exact match (segment):  {exact:4d} / {total}  ({100*exact/total:.1f}%)")
    print(f"  Sector match only:      {sector_only:4d} / {total}  ({100*sector_only/total:.1f}%)")
    print(f"  Complete miss:          {misses:4d} / {total}  ({100*misses/total:.1f}%)")
    print()

    # Show mismatches
    wrong = [c for c in comparisons if not c["match"]]
    if wrong:
        print(f"Mismatches ({len(wrong)}):")
        print(f"  {'File':<20s} {'Pos':>12s}  {'Actual':<10s} {'Predicted':<10s} {'Sector?'}")
        print(f"  {'-'*18}  {'-'*12}  {'-'*10} {'-'*10} {'-'*7}")
        for c in wrong:
            sector_ok = "yes" if c["sector_match"] else "NO"
            print(f"  {c['filename']:<20s} ({c['x']:4d},{c['y']:4d})  "
                  f"{c['actual']:<10s} {c['predicted']:<10s} {sector_ok}")

    # Confusion matrix by sector
    print(f"\nSector confusion (predicted → actual):")
    sector_errors = {}
    for c in wrong:
        key = (c["predicted"], c["actual"])
        sector_errors[key] = sector_errors.get(key, 0) + 1
    for (pred, actual), count in sorted(sector_errors.items(), key=lambda x: -x[1]):
        print(f"  {pred:>10s} → {actual:<10s}  ({count}x)")


def browse_comparisons(comparisons, labeled_data, homography=None, crop_offset=(0, 0)):
    """Interactive browser to view each dart comparison on the image."""
    if not comparisons:
        print("No comparisons to browse.")
        return

    win = "Homography Comparison"
    panel_win = "Details"
    create_window(win, default_width=960, default_height=540)
    create_window(panel_win, default_width=400, default_height=300)

    # Group by file
    by_file = {}
    for c in comparisons:
        by_file.setdefault(c["filename"], []).append(c)

    file_list = sorted(by_file.keys())

    # Filter options
    SHOW_ALL = 0
    SHOW_MISSES = 1
    show_mode = SHOW_ALL
    mode_names = ["ALL", "MISSES ONLY"]
    show_overlay = True  # board geometry overlay

    file_idx = 0
    dart_idx = 0

    print("\nBrowse: LEFT/RIGHT=file  UP/DOWN=dart  M=misses  O=overlay  Q=quit\n")

    def get_filtered_files():
        if show_mode == SHOW_MISSES:
            return [f for f in file_list if any(not c["match"] for c in by_file[f])]
        return file_list

    while True:
        filtered = get_filtered_files()
        if not filtered:
            # No files match filter
            cp = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.putText(cp, "No mismatches found!", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(panel_win, cp)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                show_mode = (show_mode + 1) % 2
            continue

        file_idx = file_idx % len(filtered)
        fname = filtered[file_idx]
        file_comps = by_file[fname]
        dart_idx = dart_idx % len(file_comps)
        current = file_comps[dart_idx]

        # Load image
        img = cv2.imread(current["image_path"])
        if img is None:
            file_idx += 1
            continue

        display = img.copy()
        h, w = display.shape[:2]

        # Draw board geometry overlay
        if show_overlay and homography is not None:
            _draw_homography_overlay(display, homography, crop_offset)

        # Draw all darts in this file
        for i, c in enumerate(file_comps):
            color = (0, 255, 0) if c["match"] else (0, 0, 255)
            thickness = 3 if i == dart_idx else 1
            cv2.circle(display, (c["x"], c["y"]), 12, color, thickness)

            # Label
            label = f"d{c['ordinal']} {c['actual']}"
            cv2.putText(display, label, (c["x"] + 15, c["y"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if not c["match"]:
                # Show prediction
                cv2.putText(display, f"pred: {c['predicted']}", (c["x"] + 15, c["y"] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

        # File info
        cv2.putText(display, f"{fname}  |  {file_idx+1}/{len(filtered)}  |  Mode: {mode_names[show_mode]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(win, display)

        # Detail panel
        cp = np.zeros((300, 400, 3), dtype=np.uint8)
        cp[:] = (30, 30, 30)
        cp_y = 25

        cv2.putText(cp, f"Dart {dart_idx+1}/{len(file_comps)}", (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cp_y += 30

        match_text = "MATCH" if current["match"] else "MISMATCH"
        match_color = (0, 255, 0) if current["match"] else (0, 0, 255)
        cv2.putText(cp, match_text, (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, match_color, 2)
        cp_y += 35

        cv2.putText(cp, f"Actual:    d{current['ordinal']} {current['actual']}", (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cp_y += 25
        cv2.putText(cp, f"Predicted: d{current['ordinal']} {current['predicted']}", (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if current["match"] else (0, 0, 255), 1)
        cp_y += 25
        cv2.putText(cp, f"Position: ({current['x']}, {current['y']})", (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cp_y += 25

        if current["sector_match"] and not current["match"]:
            cv2.putText(cp, "Sector correct, ring wrong", (10, cp_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cp_y += 25

        cp_y += 15
        overlay_text = "ON" if show_overlay else "OFF"
        cv2.putText(cp, f"Overlay: {overlay_text}", (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cp_y += 20
        cv2.putText(cp, "LEFT/RIGHT=file  UP/DOWN=dart  M=filter  O=overlay  Q=quit", (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)
        cv2.imshow(panel_win, cp)

        # Keys
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == 81 or key == 2:  # LEFT arrow
            file_idx = (file_idx - 1) % len(filtered)
            dart_idx = 0
        elif key == 83 or key == 3:  # RIGHT arrow
            file_idx = (file_idx + 1) % len(filtered)
            dart_idx = 0
        elif key == 82 or key == 0:  # UP arrow
            dart_idx = (dart_idx - 1) % len(file_comps)
        elif key == 84 or key == 1:  # DOWN arrow
            dart_idx = (dart_idx + 1) % len(file_comps)
        elif key == ord('m'):
            show_mode = (show_mode + 1) % 2
            file_idx = 0
            dart_idx = 0
            print(f"  Filter: {mode_names[show_mode]}")
        elif key == ord('o'):
            show_overlay = not show_overlay
            print(f"  Overlay: {'ON' if show_overlay else 'OFF'}")

    save_window_sizes([win, panel_win])
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Compare homography vs YOLO labels")
    parser.add_argument("--outdir", default="data/training",
                        help="Training data directory")
    parser.add_argument("--report", action="store_true",
                        help="Print accuracy report")
    parser.add_argument("--browse", action="store_true",
                        help="Interactive comparison browser")
    args = parser.parse_args()

    if not args.report and not args.browse:
        args.report = True
        args.browse = True

    # Load homography
    if not config.BOARD_HOMOGRAPHY_PATH.exists():
        print(f"ERROR: No homography found at {config.BOARD_HOMOGRAPHY_PATH}")
        print("Run 'python calibrate.py --board' first.")
        sys.exit(1)

    hom_data = np.load(str(config.BOARD_HOMOGRAPHY_PATH), allow_pickle=False)
    homography = hom_data['homography']

    # Crop offset: only needed if homography was calibrated on uncropped frames.
    # If calibrated with --no-undistort (matching collection pipeline), the
    # homography is already in cropped space and no offset is needed.
    crop_offset = (0, 0)
    print(f"Crop offset: {crop_offset} (homography in cropped space)")

    # Load labeled data
    labeled = load_labeled_darts(args.outdir)
    total_darts = sum(len(f["darts"]) for f in labeled)
    print(f"Loaded {len(labeled)} frames with {total_darts} labeled darts")

    if total_darts == 0:
        print("No labeled data found. Run collection + labeling first.")
        sys.exit(0)

    # Compare
    comparisons = compare_all(labeled, homography, crop_offset)

    if args.report:
        print_report(comparisons)

    if args.browse:
        browse_comparisons(comparisons, labeled, homography, crop_offset)


if __name__ == "__main__":
    main()
