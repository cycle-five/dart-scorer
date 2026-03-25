#!/usr/bin/env python3
"""
scorer.py — V2 dart scoring: YOLO detection + geometry classification.

Uses a 1-class YOLO model to find darts, then board homography to
classify each dart by segment (sector + ring). No ordinal prediction —
darts are numbered by order of appearance.

Usage:
    python scorer.py                    # Raw detection mode
    python scorer.py --game 501         # Track a 501 game
    python scorer.py --game 301         # Track a 301 game
    python scorer.py --debug            # Show detection details
    python scorer.py --conf 0.3         # Custom confidence threshold
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from enum import Enum, auto

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import config
from yolo_detector import YOLODartDetector


class State(Enum):
    WAITING = auto()             # No new darts, waiting for throw
    WAITING_FOR_REMOVAL = auto() # Dart(s) on board, waiting for removal


def init_csv_log():
    """Initialize the CSV log file with headers if it doesn't exist."""
    log_path = config.LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'tip_x', 'tip_y',
                           'segment', 'yolo_confidence', 'geo_confidence',
                           'score', 'label',
                           'ordinal', 'game_mode', 'remaining'])


def log_detection(det, game_mode=None, remaining=None):
    """Append a detection to the CSV log."""
    with open(config.LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            det["tip"][0], det["tip"][1],
            det["segment"],
            f"{det['yolo_confidence']:.3f}",
            f"{det['confidence']:.3f}",
            det["score"], det["label"],
            det.get("ordinal", ""),
            game_mode or '',
            remaining if remaining is not None else '',
        ])


class GameTracker:
    """Tracks score for 301/501 games."""

    def __init__(self, starting_score):
        self.starting_score = starting_score
        self.remaining = starting_score
        self.darts_thrown = 0
        self.history = []

    def add_score(self, points):
        """Add a dart score. Returns (remaining, bust)."""
        self.darts_thrown += 1
        self.history.append(points)

        new_remaining = self.remaining - points
        if new_remaining < 0:
            print(f"  BUST! {self.remaining} - {points} = {new_remaining}")
            return self.remaining, True

        self.remaining = new_remaining
        return self.remaining, False

    def is_finished(self):
        return self.remaining == 0

    def reset(self):
        self.remaining = self.starting_score
        self.darts_thrown = 0
        self.history = []

    def get_display_text(self):
        return f"Remaining: {self.remaining}  |  Darts: {self.darts_thrown}"


def main():
    parser = argparse.ArgumentParser(description="V2 dart scoring: YOLO + geometry")
    parser.add_argument("--game", type=int, choices=config.SUPPORTED_GAMES,
                       help="Game mode (301 or 501)")
    parser.add_argument("--debug", action="store_true",
                       help="Show detection details and bboxes")
    parser.add_argument("--conf", type=float, default=config.YOLO_DEFAULT_CONF,
                       help=f"YOLO confidence threshold (default: {config.YOLO_DEFAULT_CONF})")
    parser.add_argument("--weights", type=str, default=None,
                       help="Path to YOLO weights file")
    parser.add_argument("--no-undistort", action="store_true",
                       help="Skip lens undistortion")
    args = parser.parse_args()

    # --- Load homography ---
    homography = None
    if config.BOARD_HOMOGRAPHY_PATH.exists():
        data = np.load(str(config.BOARD_HOMOGRAPHY_PATH))
        homography = data["homography"]
        print("Board homography loaded.")
    else:
        print("WARNING: No board homography found. Scoring will not work.")
        print("Run: uv run python calibrate.py --board")

    # --- Crop ROI ---
    from calibrate import open_camera, load_crop_roi, apply_crop
    crop_roi = load_crop_roi()
    crop_offset = (0, 0)
    if crop_roi is not None:
        x, y, w, h = crop_roi
        crop_offset = (x, y)
        print(f"Crop ROI: ({x}, {y}) {w}x{h}")

    # --- Load YOLO model (resolution-aware) ---
    weights = args.weights
    if weights is None:
        # Auto-select best model for current crop resolution
        from yolo_detector import find_best_weights
        crop_w = crop_roi[2] if crop_roi is not None else config.CAMERA_WIDTH
        crop_h = crop_roi[3] if crop_roi is not None else config.CAMERA_HEIGHT
        weights = find_best_weights(crop_w, crop_h)
        if weights:
            print(f"Auto-selected model: {weights.parent.parent.name}")

    try:
        detector = YOLODartDetector(
            weights=weights,
            conf=args.conf,
            homography=homography,
            crop_offset=crop_offset,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("YOLO v2 detector loaded (1-class + geometry).")

    # --- Optional lens undistortion ---
    lens_params = None
    if not args.no_undistort and config.LENS_PARAMS_PATH.exists():
        from calibrate import load_lens_params, undistort_frame
        lens_params = load_lens_params()
        print("Lens undistortion enabled.")

    # --- Setup ---
    cap = open_camera()
    init_csv_log()

    game = None
    game_mode = args.game
    if game_mode:
        game = GameTracker(game_mode)
        print(f"\n=== {game_mode} Game Started ===")
        print(f"Remaining: {game.remaining}\n")

    state = State.WAITING
    last_score_label = None
    empty_frames = 0
    REMOVAL_THRESHOLD = config.EMPTY_FRAME_REMOVAL_THRESHOLD

    window = "Dart Scorer v2"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print("\nScoring active. Controls:")
    print("  q = quit")
    print("  r = reset game")
    print("  d = toggle debug")
    print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame.")
                break

            # Optional undistortion + crop
            if lens_params is not None:
                frame = undistort_frame(frame, *lens_params)
            frame = apply_crop(frame, crop_roi)

            display = frame.copy()

            # Run YOLO detection + geometry classification
            detections = detector.process_frame(frame)

            if state == State.WAITING:
                new_darts = detector.get_new_darts(detections)

                for det in new_darts:
                    detector.confirm_dart(det)
                    ordinal = det["ordinal"]
                    segment = det["segment"]
                    score = det["score"]
                    label = det["label"]
                    geo_conf = det["confidence"]

                    # Confidence indicator
                    conf_marker = ""
                    if geo_conf < config.GEO_CONF_MODERATE:
                        conf_marker = " [?]"
                    elif geo_conf < config.GEO_CONF_HIGH:
                        conf_marker = " [~]"

                    last_score_label = f"dart {ordinal}: {label}{conf_marker}"

                    print(f">>> DART {ordinal}: {label}"
                          f"  (yolo={det['yolo_confidence']:.0%}"
                          f", geo={geo_conf:.0%}){conf_marker}")

                    if game:
                        remaining, bust = game.add_score(score)
                        if bust:
                            print(f"    (Bust)")
                        else:
                            print(f"    {game.get_display_text()}")
                        if game.is_finished():
                            print(f"\n*** GAME OVER! {game_mode} in "
                                  f"{game.darts_thrown} darts! ***\n")

                    log_detection(det, game_mode,
                                 game.remaining if game else None)

                if detector.round_dart_count >= 3:
                    state = State.WAITING_FOR_REMOVAL
                    empty_frames = 0
                elif detector.round_dart_count > 0 and len(detections) == 0:
                    empty_frames += 1
                    if empty_frames >= REMOVAL_THRESHOLD:
                        print("  (Darts removed — new round)\n")
                        detector.reset()
                        state = State.WAITING
                        last_score_label = None
                        empty_frames = 0
                else:
                    empty_frames = 0

            elif state == State.WAITING_FOR_REMOVAL:
                if len(detections) == 0:
                    empty_frames += 1
                    if empty_frames >= REMOVAL_THRESHOLD:
                        print("  (Darts removed — ready for next round)\n")
                        detector.reset()
                        state = State.WAITING
                        last_score_label = None
                        empty_frames = 0
                else:
                    empty_frames = 0

            # --- Draw HUD ---
            if args.debug:
                display = detector.get_debug_frame(display, detections)

            if last_score_label:
                cv2.putText(display, last_score_label, (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            if game:
                cv2.putText(display, game.get_display_text(), (10, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if game.is_finished():
                    cv2.putText(display, "GAME OVER!", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            # State + dart count
            state_text = f"{state.name} | Darts: {detector.round_dart_count}/3"
            h_disp = display.shape[0]
            cv2.putText(display, state_text, (10, h_disp - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Draw confirmed dart markers (when not in debug mode)
            if not args.debug:
                for ordinal, dart in detector.confirmed_darts.items():
                    tx, ty = dart["tip"]
                    color = config.DART_ORDINAL_COLORS.get(ordinal, (255, 255, 255))
                    cv2.circle(display, (tx, ty), 8, color, 2)
                    cv2.circle(display, (tx, ty), 2, color, -1)

            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting.")
                break
            elif key == ord('r'):
                if game:
                    game.reset()
                    print(f"\n=== Game Reset — {game_mode} ===")
                    print(f"Remaining: {game.remaining}\n")
                detector.reset()
                state = State.WAITING
                last_score_label = None
                print("Round reset.\n")
            elif key == ord('d'):
                args.debug = not args.debug
                print(f"Debug: {'ON' if args.debug else 'OFF'}")

    except KeyboardInterrupt:
        print("\nInterrupted — exiting.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
