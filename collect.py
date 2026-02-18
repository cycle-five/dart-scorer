#!/usr/bin/env python3
"""
collect.py — Training data collection for dart tip detection.

Captures camera frames and lets the user click to annotate dart tip positions.
Saves images and annotations for later use in training ML models (YOLO, etc.)
or for Bayesian parameter optimization of the classical CV pipeline.

Usage:
    python collect.py                  # Start collecting
    python collect.py --outdir data/training   # Custom output directory
    python collect.py --no-undistort   # Skip lens undistortion

Controls:
    SPACE   Capture/freeze current frame for annotation
    Click   Mark a dart tip (while frozen)
    BACKSPACE / Z   Undo last click
    ENTER   Save annotated frame and resume live feed
    ESC     Discard current capture and resume live feed
    B       Save current frame as background (no darts)
    Q       Quit
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import config


# ---------------------------------------------------------------------------
# Mouse callback state
# ---------------------------------------------------------------------------

_click_points = []
_active = False


def _mouse_callback(event, x, y, flags, param):
    global _click_points
    if _active and event == cv2.EVENT_LBUTTONDOWN:
        _click_points.append((x, y))


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_data(outdir="data/training", use_undistort=True):
    global _click_points, _active

    outdir = Path(outdir)
    img_dir = outdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = outdir / "annotations.jsonl"

    # Load existing count to continue numbering
    existing = 0
    if annotations_path.exists():
        with open(annotations_path) as f:
            existing = sum(1 for _ in f)
    frame_counter = existing

    # Camera setup
    from calibrate import open_camera, load_lens_params, undistort_frame
    cap = open_camera()

    lens_params = None
    if use_undistort and config.LENS_PARAMS_PATH.exists():
        lens_params = load_lens_params()
        print("Lens undistortion enabled")
    elif use_undistort:
        print("WARNING: No lens params found, running without undistortion")

    win = "Collect Training Data"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _mouse_callback)

    print("\n=== Training Data Collection ===")
    print(f"Output: {outdir.resolve()}")
    print(f"Continuing from frame {frame_counter}")
    print()
    print("Controls:")
    print("  SPACE     Capture frame for annotation")
    print("  Click     Mark dart tip (while frozen)")
    print("  BKSP/Z    Undo last click")
    print("  ENTER     Save and resume")
    print("  ESC       Discard and resume")
    print("  B         Save as background (0 darts)")
    print("  Q         Quit")
    print()

    frozen_frame = None
    _active = False

    while True:
        if frozen_frame is None:
            # Live feed
            ret, raw = cap.read()
            if not ret:
                print("Camera read failed")
                break

            frame = raw
            if lens_params is not None:
                frame = undistort_frame(raw, *lens_params)

            display = frame.copy()
            h, w = display.shape[:2]
            cv2.putText(display, f"LIVE  |  Saved: {frame_counter}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, "SPACE=capture  B=background  Q=quit",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow(win, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Freeze frame for annotation
                frozen_frame = frame.copy()
                _click_points = []
                _active = True
                print(f"Frame captured — click dart tips, ENTER to save, ESC to discard")
            elif key == ord('b'):
                # Save as background (no darts)
                fname = f"frame_{frame_counter:05d}.png"
                cv2.imwrite(str(img_dir / fname), frame)
                entry = {
                    "filename": fname,
                    "tips": [],
                    "n_darts": 0,
                    "is_background": True,
                    "timestamp": time.time(),
                }
                with open(annotations_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                frame_counter += 1
                print(f"  Saved background: {fname}")
        else:
            # Frozen: annotating
            display = frozen_frame.copy()
            h, w = display.shape[:2]

            # Draw existing clicks
            for i, (px, py) in enumerate(_click_points):
                cv2.circle(display, (px, py), 7, (0, 0, 255), -1)
                cv2.circle(display, (px, py), 9, (255, 255, 255), 2)
                cv2.putText(display, str(i + 1), (px + 12, py + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            n = len(_click_points)
            cv2.putText(display, f"FROZEN  |  Tips marked: {n}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display, "Click=mark tip  BKSP/Z=undo  ENTER=save  ESC=discard",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow(win, display)

            key = cv2.waitKey(30) & 0xFF
            if key == 13:  # ENTER — save
                fname = f"frame_{frame_counter:05d}.png"
                cv2.imwrite(str(img_dir / fname), frozen_frame)
                entry = {
                    "filename": fname,
                    "tips": _click_points[:],
                    "n_darts": len(_click_points),
                    "is_background": False,
                    "timestamp": time.time(),
                }
                with open(annotations_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                frame_counter += 1
                print(f"  Saved: {fname} with {len(_click_points)} tip(s)")
                frozen_frame = None
                _active = False
                _click_points = []
            elif key == 27:  # ESC — discard
                print("  Discarded")
                frozen_frame = None
                _active = False
                _click_points = []
            elif key in (8, ord('z')):  # BACKSPACE or Z — undo
                if _click_points:
                    removed = _click_points.pop()
                    print(f"  Undo: removed ({removed[0]}, {removed[1]})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {frame_counter} total frames saved to {outdir.resolve()}")


# ---------------------------------------------------------------------------
# Convert annotations to YOLO format
# ---------------------------------------------------------------------------

def convert_to_yolo(outdir="data/training", box_size=30):
    """Convert annotations.jsonl to YOLO-format label files.

    Creates a labels/ directory with one .txt per image. Each dart tip
    becomes a bounding box of `box_size` x `box_size` pixels centered on the
    tip, normalized to image dimensions. Class 0 = dart_tip.

    Args:
        outdir: Training data directory containing annotations.jsonl and images/
        box_size: Width/height of the bounding box in pixels around each tip.
    """
    outdir = Path(outdir)
    annotations_path = outdir / "annotations.jsonl"
    label_dir = outdir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    if not annotations_path.exists():
        print(f"No annotations found at {annotations_path}")
        return

    count = 0
    with open(annotations_path) as f:
        for line in f:
            entry = json.loads(line)
            fname = entry["filename"]
            tips = entry["tips"]

            # Read image to get dimensions
            img_path = outdir / "images" / fname
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Write YOLO label file
            label_name = Path(fname).stem + ".txt"
            with open(label_dir / label_name, "w") as lf:
                for (tx, ty) in tips:
                    # YOLO format: class center_x center_y width height (all normalized)
                    cx = tx / w
                    cy = ty / h
                    bw = box_size / w
                    bh = box_size / h
                    lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            count += 1

    print(f"Converted {count} annotations to YOLO format in {label_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect dart training data")
    parser.add_argument("--outdir", default="data/training",
                        help="Output directory (default: data/training)")
    parser.add_argument("--no-undistort", action="store_true",
                        help="Skip lens undistortion")
    parser.add_argument("--to-yolo", action="store_true",
                        help="Convert existing annotations to YOLO format")
    parser.add_argument("--box-size", type=int, default=30,
                        help="Bounding box size in pixels for YOLO (default: 30)")
    args = parser.parse_args()

    if args.to_yolo:
        convert_to_yolo(args.outdir, args.box_size)
    else:
        collect_data(args.outdir, use_undistort=not args.no_undistort)
