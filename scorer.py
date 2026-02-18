#!/usr/bin/env python3
"""
scorer.py — Main dart scoring loop.

Loads calibration data, runs the dart detector, scores detected darts,
and optionally tracks a game (301 or 501).

Usage:
    python scorer.py                    # Raw detection mode
    python scorer.py --game 501         # Track a 501 game
    python scorer.py --game 301         # Track a 301 game
    python scorer.py --debug            # Show intermediate CV steps
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from enum import Enum, auto

# Suppress Qt/Wayland warnings from pip-installed OpenCV on Linux
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import config
import board
from calibrate import open_camera, load_lens_params, undistort_frame, calibrate_board
from detector import DartDetector


class State(Enum):
    WAITING = auto()             # No new darts, waiting for throw
    DART_DETECTED = auto()       # New dart detected, scoring it
    WAITING_FOR_REMOVAL = auto() # Dart scored, waiting for player to remove darts


def init_csv_log():
    """Initialize the CSV log file with headers if it doesn't exist."""
    log_path = config.LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'cam_x', 'cam_y', 'canonical_x', 'canonical_y',
                           'sector', 'ring', 'multiplier', 'score', 'label',
                           'game_mode', 'remaining'])

def log_detection(score_info, cam_tip, game_mode=None, remaining=None):
    """Append a detection to the CSV log.

    Args:
        score_info: Score dict from board.score_from_camera (canonical coords).
        cam_tip: (x, y) dart tip in camera pixel space.
        game_mode: Game variant (301/501) or None.
        remaining: Remaining score or None.
    """
    with open(config.LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            cam_tip[0], cam_tip[1],
            score_info['x'], score_info['y'],
            score_info['sector'], score_info['ring'],
            score_info['multiplier'], score_info['score'],
            score_info['label'],
            game_mode or '',
            remaining if remaining is not None else '',
        ])


class GameTracker:
    """Tracks score for 301/501 games."""

    def __init__(self, starting_score):
        self.starting_score = starting_score
        self.remaining = starting_score
        self.darts_thrown = 0
        self.round_scores = []  # scores in current round (max 3 per round)
        self.history = []       # list of all scored darts

    def add_score(self, points):
        """Add a dart score. Returns (remaining, bust) tuple.

        In standard rules, if the score would go below 0 or to exactly 1,
        it's a bust and the round is void. For simplicity, we just prevent
        going below 0.
        """
        self.darts_thrown += 1
        self.history.append(points)

        new_remaining = self.remaining - points
        if new_remaining < 0:
            # Bust — score doesn't count
            print(f"  BUST! {self.remaining} - {points} = {new_remaining} (below zero)")
            return self.remaining, True

        self.remaining = new_remaining
        self.round_scores.append(points)
        return self.remaining, False

    def is_finished(self):
        return self.remaining == 0

    def reset(self):
        self.remaining = self.starting_score
        self.darts_thrown = 0
        self.round_scores = []
        self.history = []

    def get_display_text(self):
        return f"Remaining: {self.remaining}  |  Darts: {self.darts_thrown}"


def main():
    parser = argparse.ArgumentParser(description="Dart scoring system")
    parser.add_argument("--game", type=int, choices=config.SUPPORTED_GAMES,
                       help="Game mode (301 or 501)")
    parser.add_argument("--debug", action="store_true",
                       help="Show intermediate CV steps")
    args = parser.parse_args()

    # --- Load calibration ---
    try:
        cam_mtx, dist_coeffs, new_cam_mtx, roi = load_lens_params()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run 'python calibrate.py --lens' first.")
        sys.exit(1)

    homography_path = config.BOARD_HOMOGRAPHY_PATH
    if not homography_path.exists():
        print(f"ERROR: Board homography not found at {homography_path}")
        print("Run 'python calibrate.py --board' first.")
        sys.exit(1)

    hom_data = np.load(str(homography_path))
    homography = hom_data['homography']
    ellipse_center = tuple(hom_data['ellipse_center'])

    print("Calibration loaded successfully.")

    # --- Setup ---
    cap = open_camera()
    detector = DartDetector(board_center=(int(ellipse_center[0]), int(ellipse_center[1])),
                           debug=args.debug)
    init_csv_log()

    game = None
    game_mode = args.game
    if game_mode:
        game = GameTracker(game_mode)
        print(f"\n=== {game_mode} Game Started ===")
        print(f"Remaining: {game.remaining}\n")

    state = State.WAITING
    last_score_info = None

    window = "Dart Scorer"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print("\nScoring active. Controls:")
    print("  q = quit")
    print("  r = reset game")
    print("  c = recalibrate board")
    print("  d = toggle debug")
    print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame.")
                break

            # Undistort
            undistorted = undistort_frame(frame, cam_mtx, dist_coeffs, new_cam_mtx, roi)
            display = undistorted.copy()

            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

            if state == State.WAITING:
                # Look for new darts
                detections = detector.process_frame(undistorted)

                if detections:
                    state = State.DART_DETECTED
                    for det in detections:
                        tip = det['tip']
                        # Score the dart
                        score_info = board.score_from_camera(tip, homography)
                        last_score_info = score_info

                        print(f">>> DART: {score_info['label']}")

                        remaining = None
                        if game:
                            remaining, bust = game.add_score(score_info['score'])
                            if bust:
                                print(f"    (Bust — score voided)")
                            else:
                                print(f"    {game.get_display_text()}")
                            if game.is_finished():
                                print(f"\n*** GAME OVER! {game_mode} completed in {game.darts_thrown} darts! ***\n")

                        log_detection(score_info, tip, game_mode,
                                     game.remaining if game else None)

                    state = State.WAITING_FOR_REMOVAL

            elif state == State.WAITING_FOR_REMOVAL:
                # Still run detection to track existing darts
                detections = detector.process_frame(undistorted)

                # Also check for new darts while waiting
                if detections:
                    for det in detections:
                        tip = det['tip']
                        score_info = board.score_from_camera(tip, homography)
                        last_score_info = score_info

                        print(f">>> DART: {score_info['label']}")

                        if game:
                            remaining, bust = game.add_score(score_info['score'])
                            if bust:
                                print(f"    (Bust — score voided)")
                            else:
                                print(f"    {game.get_display_text()}")
                            if game.is_finished():
                                print(f"\n*** GAME OVER! {game_mode} completed in {game.darts_thrown} darts! ***\n")

                        log_detection(score_info, tip, game_mode,
                                     game.remaining if game else None)

                # Check if all darts removed
                removed = detector.dart_removed(gray)
                if removed and len(detector.confirmed_darts) == 0:
                    print("  (Darts removed — ready for next throw)\n")
                    state = State.WAITING
                    last_score_info = None

            # --- Draw HUD overlay ---

            # Show last score
            if last_score_info:
                label = last_score_info['label']
                cv2.putText(display, label, (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            # Show game info
            if game:
                game_text = game.get_display_text()
                cv2.putText(display, game_text, (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if game.is_finished():
                    cv2.putText(display, "GAME OVER!", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            # Show state
            state_text = f"State: {state.name}"
            h_disp = display.shape[0]
            cv2.putText(display, state_text, (10, h_disp - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Draw confirmed dart tips
            for blob_id, dart in detector.confirmed_darts.items():
                tx, ty = dart['tip']
                cv2.circle(display, (tx, ty), 8, (0, 0, 255), 2)
                cv2.circle(display, (tx, ty), 2, (0, 0, 255), -1)

            # Debug overlay
            if args.debug:
                display = detector.get_debug_frame(display, [])

            cv2.imshow(window, display)

            # --- Key handling ---
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
                last_score_info = None
                print("Detector reset.\n")
            elif key == ord('c'):
                print("Recalibrating board...")
                if calibrate_board(cap, debug=args.debug):
                    hom_data = np.load(str(homography_path))
                    homography = hom_data['homography']
                    ellipse_center = tuple(hom_data['ellipse_center'])
                    detector.board_center = (int(ellipse_center[0]), int(ellipse_center[1]))
                    detector.reset()
                    state = State.WAITING
                    last_score_info = None
                    print("Board recalibrated successfully.\n")
                else:
                    print("Board recalibration failed.\n")
            elif key == ord('d'):
                args.debug = not args.debug
                detector.debug = args.debug
                print(f"Debug: {'ON' if args.debug else 'OFF'}")
                if not args.debug:
                    cv2.destroyWindow("Diff Mask")

    except KeyboardInterrupt:
        print("\nInterrupted — exiting.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
