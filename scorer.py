#!/usr/bin/env python3
"""
scorer.py — Main dart scoring loop using YOLO detection.

Runs YOLO inference on camera frames to detect darts, classify their
board segment, and optionally track a game (301 or 501).

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
    WAITING_FOR_REMOVAL = auto() # Dart(s) on board, waiting for player to remove


def init_csv_log():
    """Initialize the CSV log file with headers if it doesn't exist."""
    log_path = config.LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'tip_x', 'tip_y',
                           'class_name', 'confidence',
                           'sector', 'ring', 'multiplier', 'score', 'label',
                           'ordinal', 'game_mode', 'remaining'])


def log_detection(det, game_mode=None, remaining=None):
    """Append a detection to the CSV log."""
    info = det["score_info"]
    with open(config.LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            det["tip"][0], det["tip"][1],
            det["class_name"], f"{det['confidence']:.3f}",
            info["sector"], info["ring"],
            info["multiplier"], info["score"], info["label"],
            info["ordinal"],
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
    parser = argparse.ArgumentParser(description="YOLO dart scoring system")
    parser.add_argument("--game", type=int, choices=config.SUPPORTED_GAMES,
                       help="Game mode (301 or 501)")
    parser.add_argument("--debug", action="store_true",
                       help="Show detection details")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--weights", type=str, default=None,
                       help="Path to YOLO weights file")
    args = parser.parse_args()

    # --- Load YOLO model ---
    try:
        detector = YOLODartDetector(
            weights=args.weights,
            conf=args.conf,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("YOLO model loaded.")

    # --- Optional lens undistortion ---
    lens_params = None
    if config.LENS_PARAMS_PATH.exists():
        from calibrate import load_lens_params, undistort_frame
        lens_params = load_lens_params()
        print("Lens undistortion enabled")

    # --- Crop ROI ---
    from calibrate import open_camera, load_crop_roi, apply_crop
    crop_roi = load_crop_roi()
    if crop_roi is not None:
        x, y, w, h = crop_roi
        print(f"Crop ROI: ({x}, {y}) {w}x{h}")

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
    # Track how many consecutive frames we see zero darts (for removal detection)
    empty_frames = 0
    REMOVAL_THRESHOLD = 10  # frames with 0 detections to confirm darts pulled

    window = "Dart Scorer (YOLO)"
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

            # Run YOLO
            detections = detector.process_frame(frame)

            if state == State.WAITING:
                new_darts = detector.get_new_darts(detections)

                for det in new_darts:
                    detector.confirm_dart(det)
                    info = det["score_info"]
                    last_score_label = f"d{info['ordinal']} {info['label']}"

                    print(f">>> DART {info['ordinal']}: {info['label']} "
                          f"(conf={det['confidence']:.2f})")

                    if game:
                        remaining, bust = game.add_score(info["score"])
                        if bust:
                            print(f"    (Bust)")
                        else:
                            print(f"    {game.get_display_text()}")
                        if game.is_finished():
                            print(f"\n*** GAME OVER! {game_mode} in {game.darts_thrown} darts! ***\n")

                    log_detection(det, game_mode,
                                 game.remaining if game else None)

                if detector.round_dart_count >= 3:
                    state = State.WAITING_FOR_REMOVAL
                    empty_frames = 0
                elif detector.round_dart_count > 0 and len(detections) == 0:
                    # Darts may have been pulled mid-round
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
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            if game:
                cv2.putText(display, game.get_display_text(), (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if game.is_finished():
                    cv2.putText(display, "GAME OVER!", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            # State + dart count
            state_text = f"{state.name} | Darts: {detector.round_dart_count}/3"
            h_disp = display.shape[0]
            cv2.putText(display, state_text, (10, h_disp - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Draw confirmed dart markers
            for ordinal, dart in detector.confirmed_darts.items():
                tx, ty = dart["tip"]
                colors = {1: (0, 255, 0), 2: (0, 255, 255), 3: (0, 0, 255)}
                color = colors.get(ordinal, (255, 255, 255))
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
