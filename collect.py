#!/usr/bin/env python3
"""
collect.py — Training data collection for YOLO dart detection + scoring.

Captures camera frames and lets the user annotate dart tips with their
board segment (e.g. T20, S5, D_BULL). Tracks dart ordinal (1st/2nd/3rd)
automatically within each round. Saves images and YOLO-format labels.

Trigger modes for auto-capture:
  audio  — ML classifier on camera mic detects dart impact sound (default)
  video  — frame differencing detects visual change (dart appearing on board)
  manual — SPACE key only

Board homography (if calibrated) provides a segment guess when you click a
tip, so you can just press ENTER to accept or type a correction.

Usage:
    python collect.py                          # Audio trigger (default)
    python collect.py --trigger video          # Video trigger (frame diff)
    python collect.py --trigger manual         # SPACE only
    python collect.py --audio-device 9         # Specific mic device index
    python collect.py --audio-threshold 0.7    # Adjust audio prob threshold
    python collect.py --video-threshold 5.0    # Adjust video diff threshold

Controls:
    (auto)  Frame freezes on dart impact / visual change
    SPACE   Manual capture/freeze (always available)
    Click   Mark a dart tip (while frozen) — auto-guesses segment if calibrated
    ENTER   Accept guess (or type correction first, then ENTER)
    BACKSPACE / Z   Undo last annotation (when not typing)
    ENTER   Save annotated frame (when no pending click)
    ESC     Discard current capture and resume live feed
    R       Reset round (back to dart 1)
    +/-     Adjust trigger threshold up/down
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
import board
from classes import (
    CLASS_TO_ID, SEGMENTS, make_class_name, segment_shorthand,
    parse_class_name,
)


from audio_trigger import DartAudioTrigger
from window_manager import create_window, save_window_sizes


# ---------------------------------------------------------------------------
# Video trigger (frame differencing)
# ---------------------------------------------------------------------------

class VideoTrigger:
    """Detects darts by frame differencing on heavily downsampled frames.

    Downsampling averages out per-pixel sensor noise while preserving
    dart-sized changes. We count how many coarse cells changed significantly
    rather than taking the mean diff — a dart changes a handful of cells
    by a lot, while sensor noise changes many cells by tiny amounts.
    """

    THUMB_SIZE = (80, 60)  # aggressive downsample — each cell ~24x18px at 1080p
    CELL_THRESHOLD = 12    # per-cell intensity change to count as "changed"

    def __init__(self, threshold=8, cooldown=1.5, warmup=None):
        """
        Args:
            threshold: Number of changed cells above baseline to trigger.
            cooldown: Seconds between triggers.
            warmup: Seconds before triggering is enabled (default: 5s).
        """
        self.threshold = threshold
        self.cooldown = cooldown
        self._warmup_seconds = warmup if warmup is not None else 5.0
        self._background = None
        self._frame_count = 0
        self._start_time = time.monotonic()
        self._last_trigger_time = 0.0
        self._current_diff = 0.0  # changed cells above baseline
        self._baseline_diff = 0.0
        self._raw_changed = 0     # raw count of changed cells
        self._diff_history = []
        self._history_max = 200
        self.triggered = False
        # Suppression: skip triggers while diff is elevated
        self._suppressed = False
        self._suppress_message = ""

    def _to_thumb(self, frame):
        """Convert frame to small grayscale thumbnail."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, self.THUMB_SIZE, interpolation=cv2.INTER_AREA)

    def update(self, frame):
        """Feed a new frame."""
        thumb = self._to_thumb(frame)
        self._frame_count += 1

        if self._background is None:
            self._background = thumb.astype(np.float32)
            return

        # Slow-adapting background
        cv2.accumulateWeighted(thumb, self._background, 0.02)
        bg = self._background.astype(np.uint8)

        # Count cells that changed more than CELL_THRESHOLD
        diff = cv2.absdiff(thumb, bg)
        changed_cells = int(np.sum(diff > self.CELL_THRESHOLD))
        self._raw_changed = changed_cells

        # Track baseline (noise floor of changed cell count)
        if changed_cells < self._baseline_diff * 3 + 2:
            self._baseline_diff = 0.95 * self._baseline_diff + 0.05 * changed_cells

        spike = max(0, changed_cells - self._baseline_diff)
        self._current_diff = spike

        self._diff_history.append(spike)
        if len(self._diff_history) > self._history_max:
            self._diff_history.pop(0)

        now = time.monotonic()
        elapsed = now - self._start_time

        # During warmup, don't trigger
        if elapsed < self._warmup_seconds:
            return

        # If suppressed, wait until diff drops back to near baseline
        if self._suppressed:
            if spike < self.threshold * 0.5:
                self._suppressed = False
                self._suppress_message = ""
            return

        # Trigger on spike
        if (spike > self.threshold
                and now - self._last_trigger_time > self.cooldown):
            self.triggered = True
            self._last_trigger_time = now

    def suppress_next(self, message="Suppressing trigger..."):
        """Suppress triggers until diff drops back to baseline.

        Use after 3rd dart to skip the removal event.
        """
        self._suppressed = True
        self._suppress_message = message

    @property
    def is_suppressed(self):
        return self._suppressed

    @property
    def suppress_message(self):
        return self._suppress_message

    @property
    def warmup_remaining(self):
        """Seconds of warmup remaining, or 0 if done."""
        return max(0, self._warmup_seconds - (time.monotonic() - self._start_time))

    def absorb(self, frame):
        """Reset background to current frame."""
        thumb = self._to_thumb(frame)
        self._background = thumb.astype(np.float32)

    def check_and_reset(self):
        if self.triggered:
            self.triggered = False
            return True
        return False

    @property
    def current_diff(self):
        return self._current_diff


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_click_point = None  # pending click waiting for label
_annotations = []    # list of (x, y, class_name) for current frame
_text_input = ""     # current text being typed for segment label
_guess_segment = ""  # homography-based segment guess (pre-filled on click)
_active = False      # whether we're in annotation mode
_dart_ordinal = 1    # current dart number in round (1, 2, or 3)


def _guess_segment_from_homography(x, y, homography):
    """Use board homography to guess which segment a click is in.

    Returns a segment shorthand string (e.g. "T20", "S5", "D_BULL") or None.
    """
    if homography is None:
        return None
    try:
        score_info = board.score_from_camera((x, y), homography)
        ring = score_info["ring"]
        sector = score_info["sector"]

        if ring == "D-BULL":
            return "D_BULL"
        elif ring == "S-BULL":
            return "S_BULL"
        elif ring == "miss":
            return None
        else:
            ring_code = {"single": "S", "double": "D", "triple": "T"}[ring]
            return f"{ring_code}{sector}"
    except Exception:
        return None


def _mouse_callback(event, x, y, flags, param):
    global _click_point, _text_input, _guess_segment
    if _active and event == cv2.EVENT_LBUTTONDOWN:
        _click_point = (x, y)
        # Auto-guess segment from homography
        homography = param  # passed via cv2.setMouseCallback(..., param=homography)
        guess = _guess_segment_from_homography(x, y, homography)
        if guess:
            _guess_segment = guess
            _text_input = guess.lower()
        else:
            _guess_segment = ""
            _text_input = ""


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_data(outdir="data/training", use_undistort=True, box_size=30,
                 trigger_mode="audio", audio_device=None, audio_threshold=0.7,
                 video_threshold=5.0, settle_delay=0.7):
    global _click_point, _annotations, _text_input, _guess_segment, _active, _dart_ordinal

    outdir = Path(outdir)
    img_dir = outdir / "images"
    label_dir = outdir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = outdir / "annotations.jsonl"

    # Load existing count to continue numbering
    existing = 0
    if annotations_path.exists():
        with open(annotations_path) as f:
            existing = sum(1 for _ in f)
    frame_counter = existing

    # Camera setup
    from calibrate import open_camera, load_lens_params, undistort_frame, load_crop_roi, apply_crop
    cap = open_camera()

    lens_params = None
    if use_undistort and config.LENS_PARAMS_PATH.exists():
        lens_params = load_lens_params()
        print("Lens undistortion enabled")
    elif use_undistort:
        print("WARNING: No lens params found, running without undistortion")

    crop_roi = load_crop_roi()
    if crop_roi is not None:
        x, y, w, h = crop_roi
        print(f"Crop ROI: ({x}, {y}) {w}x{h}")
    else:
        print("No crop ROI — using full frame")

    # Load board homography for segment guessing (optional)
    homography = None
    if config.BOARD_HOMOGRAPHY_PATH.exists():
        hom_data = np.load(str(config.BOARD_HOMOGRAPHY_PATH))
        homography = hom_data['homography']
        print("Board homography loaded — segment guess enabled")
    else:
        print("No board homography — you'll type segments manually")

    # Trigger setup
    audio = None
    video = None
    if trigger_mode == "audio":
        audio = DartAudioTrigger(device=audio_device, prob_threshold=audio_threshold)
        if not audio.start():
            audio = None
            print("Audio trigger failed — falling back to manual (SPACE)")
    elif trigger_mode == "video":
        video = VideoTrigger(threshold=video_threshold)
        print(f"Video trigger: ACTIVE (threshold={video_threshold:.1f})")

    win = "Collect Training Data"
    panel_win = "Control Panel"
    create_window(win, default_width=960, default_height=540)
    create_window(panel_win, default_width=400, default_height=350)
    cv2.setMouseCallback(win, _mouse_callback, param=homography)

    print("\n=== YOLO Training Data Collection (186 classes) ===")
    print(f"Output: {outdir.resolve()}")
    print(f"Continuing from frame {frame_counter}")
    print(f"Box size: {box_size}x{box_size}px")
    if audio:
        print(f"Trigger: AUDIO (prob_threshold={audio.prob_threshold:.2f})")
    elif video:
        print(f"Trigger: VIDEO (diff_threshold={video.threshold:.1f})")
    else:
        print("Trigger: MANUAL (use SPACE to capture)")
    print()
    print("Controls:")
    if audio or video:
        trigger_desc = "dart impact sound" if audio else "visual change"
        print(f"  (auto)    Frame freezes on {trigger_desc}")
    print("  SPACE     Manual capture")
    print("  Click     Mark a dart tip (while frozen)")
    print("  Type      Segment shorthand (t20, s5, dbull) then ENTER")
    print("  BKSP/Z    Undo last annotation (when not typing)")
    print("  ENTER     Save frame (when all darts labeled)")
    print("  ESC       Discard and resume")
    print("  R         Reset round to dart 1")
    if audio or video:
        print("  +/-       Adjust trigger threshold")
    print("  Q         Quit")
    print()

    frozen_frame = None
    _active = False
    _dart_ordinal = 1

    # Pre-capture delay: after trigger, keep reading frames for settle_delay
    # then freeze the LATEST frame (not the blurry trigger frame)
    pre_capture_active = False
    pre_capture_start = 0.0
    pre_capture_source = ""

    # Post-capture settle: after unfreezing, suppress triggers briefly
    settle_active = False
    settle_start_time = 0.0

    try:
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
                frame = apply_crop(frame, crop_roi)

                # Check triggers
                audio_fired = False
                video_fired = False
                # Always update video trigger
                if video:
                    video.update(frame)

                if not settle_active and not pre_capture_active:
                    if audio and audio.check_and_reset():
                        audio_fired = True
                    if video and video.check_and_reset():
                        video_fired = True

                display = frame.copy()
                h, w = display.shape[:2]

                # --- Control Panel (separate window) ---
                cp_w, cp_h = 400, 350
                cp = np.zeros((cp_h, cp_w, 3), dtype=np.uint8)
                cp[:] = (30, 30, 30)
                cp_y = 15

                # Title
                trigger_name = "VIDEO" if video else ("AUDIO" if audio else "MANUAL")
                cv2.putText(cp, f"Trigger: {trigger_name}", (10, cp_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cp_y += 25

                # Trigger waveform
                wave_x, wave_w, wave_h = 10, cp_w - 20, 100
                wave_y = cp_y
                cv2.rectangle(cp, (wave_x, wave_y), (wave_x + wave_w, wave_y + wave_h),
                              (50, 50, 50), -1)

                if video:
                    diff_history = video._diff_history
                    y_scale = max(video.threshold * 2, max(diff_history) * 1.2 if diff_history else 1.0, 1.0)

                    # Threshold line
                    ty = wave_y + wave_h - int((video.threshold / y_scale) * wave_h)
                    ty = max(wave_y + 1, min(wave_y + wave_h - 1, ty))
                    cv2.line(cp, (wave_x, ty), (wave_x + wave_w, ty), (0, 255, 255), 1)

                    if len(diff_history) > 1:
                        hmax = video._history_max
                        for i in range(1, len(diff_history)):
                            x1 = wave_x + int((i - 1) / hmax * wave_w)
                            x2 = wave_x + int(i / hmax * wave_w)
                            y1 = wave_y + wave_h - int((diff_history[i-1] / y_scale) * wave_h)
                            y2 = wave_y + wave_h - int((diff_history[i] / y_scale) * wave_h)
                            y1 = max(wave_y + 1, min(wave_y + wave_h - 1, y1))
                            y2 = max(wave_y + 1, min(wave_y + wave_h - 1, y2))
                            color = (0, 0, 255) if diff_history[i] > video.threshold else (0, 180, 0)
                            cv2.line(cp, (x1, y1), (x2, y2), color, 1)

                    cp_y = wave_y + wave_h + 5
                    cv2.putText(cp, f"cells={video.current_diff:.0f}  base={video._baseline_diff:.0f}  "
                                f"raw={video._raw_changed}  thresh={video.threshold}",
                                (wave_x, cp_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
                    cp_y += 25

                elif audio:
                    rms_history = audio._rms_history
                    prob_history = audio._prob_history
                    rms = audio.current_rms
                    prob = audio.current_prob
                    peak = audio._peak_rms

                    # Top: RMS
                    rms_h = wave_h // 2 - 3
                    y_scale = max(peak * 1.2, 0.05)
                    if len(rms_history) > 1:
                        hmax = audio._history_max
                        for i in range(1, len(rms_history)):
                            x1 = wave_x + int((i - 1) / hmax * wave_w)
                            x2 = wave_x + int(i / hmax * wave_w)
                            y1 = wave_y + rms_h - int((rms_history[i-1] / y_scale) * rms_h)
                            y2 = wave_y + rms_h - int((rms_history[i] / y_scale) * rms_h)
                            y1 = max(wave_y + 1, min(wave_y + rms_h, y1))
                            y2 = max(wave_y + 1, min(wave_y + rms_h, y2))
                            cv2.line(cp, (x1, y1), (x2, y2), (0, 200, 0), 1)
                    cv2.putText(cp, f"RMS={rms:.4f}", (wave_x + 2, wave_y + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1)

                    # Bottom: probability
                    prob_y = wave_y + rms_h + 6
                    prob_h = rms_h
                    ty = prob_y + prob_h - int(audio.prob_threshold * prob_h)
                    ty = max(prob_y, min(prob_y + prob_h, ty))
                    cv2.line(cp, (wave_x, ty), (wave_x + wave_w, ty), (0, 255, 255), 1)

                    if len(prob_history) > 1:
                        hmax = audio._history_max
                        for i in range(1, len(prob_history)):
                            x1 = wave_x + int((i - 1) / hmax * wave_w)
                            x2 = wave_x + int(i / hmax * wave_w)
                            y1 = prob_y + prob_h - int(prob_history[i-1] * prob_h)
                            y2 = prob_y + prob_h - int(prob_history[i] * prob_h)
                            y1 = max(prob_y, min(prob_y + prob_h, y1))
                            y2 = max(prob_y, min(prob_y + prob_h, y2))
                            color = (0, 0, 255) if prob_history[i] > audio.prob_threshold else (200, 100, 0)
                            cv2.line(cp, (x1, y1), (x2, y2), color, 1)

                    prob_color = (0, 0, 255) if prob > audio.prob_threshold else (200, 100, 0)
                    cv2.putText(cp, f"P(dart)={prob:.2f}", (wave_x + 2, prob_y + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, prob_color, 1)

                    cp_y = wave_y + wave_h + 5
                    cv2.putText(cp, f"thresh={audio.prob_threshold:.2f}",
                                (wave_x, cp_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
                    cp_y += 25
                else:
                    cp_y = wave_y + wave_h + 5

                # Status
                cp_y += 10
                # Determine state text
                warmup_left = video.warmup_remaining if video else 0
                if warmup_left > 0:
                    state_text = f"WARMING UP... {warmup_left:.1f}s"
                    state_color = (100, 100, 100)
                elif video and video.is_suppressed:
                    state_text = f"SUPPRESSED — {video.suppress_message}"
                    state_color = (0, 140, 255)
                elif pre_capture_active:
                    remaining = max(0, settle_delay - (time.monotonic() - pre_capture_start))
                    state_text = f"DART DETECTED — settling {remaining:.1f}s"
                    state_color = (0, 200, 255)
                elif settle_active:
                    state_text = "COOLDOWN"
                    state_color = (100, 100, 100)
                else:
                    state_text = "LISTENING"
                    state_color = (0, 255, 0)
                cv2.putText(cp, state_text, (10, cp_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
                cp_y += 25

                cv2.putText(cp, f"Dart: {_dart_ordinal}/3    Saved: {frame_counter}",
                            (10, cp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cp_y += 30

                # Controls help
                cv2.putText(cp, "SPACE  manual capture", (10, cp_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                cp_y += 18
                cv2.putText(cp, "R  reset round     Q  quit", (10, cp_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                cp_y += 18
                cv2.putText(cp, "+/-  adjust threshold", (10, cp_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

                cv2.imshow(panel_win, cp)

                # --- Camera display (clean) ---
                warmup_left = video.warmup_remaining if video else 0
                if warmup_left > 0:
                    cv2.putText(display, f"WARMING UP... {warmup_left:.1f}s",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                elif video and video.is_suppressed:
                    cv2.putText(display, f"PULL DARTS  |  Dart: {_dart_ordinal}/3",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
                elif pre_capture_active:
                    elapsed = time.monotonic() - pre_capture_start
                    remaining = max(0, settle_delay - elapsed)
                    cv2.putText(display, f"DART! Settling... {remaining:.1f}s",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                elif settle_active:
                    cv2.putText(display, f"COOLDOWN  |  Dart: {_dart_ordinal}/3",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                else:
                    cv2.putText(display, f"LIVE  |  Saved: {frame_counter}  |  Dart: {_dart_ordinal}/3",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # Minimal bottom status on camera view
                cv2.putText(display, f"Saved: {frame_counter}  |  Dart: {_dart_ordinal}/3",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                cv2.imshow(win, display)

                # Start pre-capture delay on trigger
                if (audio_fired or video_fired) and not pre_capture_active:
                    pre_capture_active = True
                    pre_capture_start = time.monotonic()
                    pre_capture_source = "AUDIO" if audio_fired else "VIDEO"
                    print(f"\n  [{pre_capture_source}] Dart detected — waiting {settle_delay}s...")

                # Check if pre-capture delay is done → freeze NOW
                should_capture = False
                if pre_capture_active and time.monotonic() - pre_capture_start >= settle_delay:
                    should_capture = True
                    pre_capture_active = False

                # Check if post-capture settle is done
                if settle_active and time.monotonic() - settle_start_time >= settle_delay:
                    settle_active = False

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    should_capture = True
                    pre_capture_active = False
                    pre_capture_source = ""
                elif key == ord('r'):
                    _dart_ordinal = 1
                    print("Round reset — dart 1")
                elif key == ord('+') or key == ord('='):
                    if audio:
                        audio.prob_threshold = min(audio.prob_threshold + 0.05, 0.99)
                        print(f"  Dart prob threshold: {audio.prob_threshold:.2f}")
                    elif video:
                        video.threshold = min(video.threshold + 2, 200)
                        print(f"  Video cell threshold: {video.threshold}")
                elif key == ord('-'):
                    if audio:
                        audio.prob_threshold = max(audio.prob_threshold - 0.05, 0.1)
                        print(f"  Dart prob threshold: {audio.prob_threshold:.2f}")
                    elif video:
                        video.threshold = max(video.threshold - 2, 1)
                        print(f"  Video cell threshold: {video.threshold}")

                if should_capture:
                    frozen_frame = frame.copy()
                    _annotations = []
                    _click_point = None
                    _text_input = ""
                    _guess_segment = ""
                    _active = True
                    trigger_label = pre_capture_source if pre_capture_source else "MANUAL"
                    print(f"[{trigger_label}] Frame captured — annotating dart {_dart_ordinal}")
                    print(f"  Click tip, type segment (e.g. t20, s5, dbull), press ENTER to confirm")
                    pre_capture_source = ""

            else:
                # Frozen: annotating
                display = frozen_frame.copy()
                h, w = display.shape[:2]

                # Update control panel while frozen
                cp_w, cp_h = 400, 350
                cp = np.zeros((cp_h, cp_w, 3), dtype=np.uint8)
                cp[:] = (30, 30, 30)
                cp_y = 15
                cv2.putText(cp, "ANNOTATING", (10, cp_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cp_y += 35
                cv2.putText(cp, f"Dart: {_dart_ordinal}/3    Saved: {frame_counter}",
                            (10, cp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cp_y += 25
                cv2.putText(cp, f"Annotated: {len(_annotations)}",
                            (10, cp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cp_y += 35
                if _click_point is not None:
                    if _guess_segment and _text_input == _guess_segment.lower():
                        cv2.putText(cp, f"Guess: [{_guess_segment}]", (10, cp_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cp_y += 22
                        cv2.putText(cp, "ENTER=accept  or type correction", (10, cp_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                    else:
                        cv2.putText(cp, f"Input: {_text_input}_", (10, cp_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        cp_y += 22
                        cv2.putText(cp, "Type segment (t20, s5, dbull) + ENTER", (10, cp_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                else:
                    cv2.putText(cp, "Click a dart tip", (10, cp_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                cp_y += 30
                cv2.putText(cp, "ENTER=save  ESC=discard  Z=undo  R=reset", (10, cp_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                cv2.imshow(panel_win, cp)

                # Draw existing annotations
                colors = {1: (0, 255, 0), 2: (0, 255, 255), 3: (0, 0, 255)}
                for (ax, ay, cls_name) in _annotations:
                    info = parse_class_name(cls_name)
                    color = colors.get(info["ordinal"], (255, 255, 255))
                    half = box_size // 2
                    cv2.rectangle(display, (ax - half, ay - half), (ax + half, ay + half), color, 2)
                    cv2.circle(display, (ax, ay), 3, color, -1)
                    label = f"d{info['ordinal']} {info['segment']}"
                    cv2.putText(display, label, (ax + half + 4, ay + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw pending click point with guess
                if _click_point is not None:
                    px, py = _click_point
                    cv2.circle(display, (px, py), 7, (255, 0, 255), 2)
                    cv2.circle(display, (px, py), 2, (255, 0, 255), -1)
                    if _guess_segment and _text_input == _guess_segment.lower():
                        # Showing the auto-guess — highlight it
                        input_text = f"Dart {_dart_ordinal} > [{_guess_segment}]  ENTER=accept / type to correct"
                        cv2.putText(display, input_text, (px + 12, py + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    else:
                        input_text = f"Dart {_dart_ordinal} > {_text_input}_"
                        cv2.putText(display, input_text, (px + 12, py + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                n = len(_annotations)
                status = f"FROZEN  |  Dart: {_dart_ordinal}/3  |  Annotated: {n}"
                cv2.putText(display, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if _click_point is not None:
                    if _guess_segment and _text_input == _guess_segment.lower():
                        cv2.putText(display, f"ENTER=accept [{_guess_segment}]  or type correction  ESC=cancel click",
                                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                    else:
                        cv2.putText(display, f"Type segment (e.g. t20, s5, dbull) then ENTER  ESC=cancel",
                                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 1)
                else:
                    cv2.putText(display, "Click=mark tip  ENTER=save  ESC=discard  Z=undo",
                                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.imshow(win, display)

                key = cv2.waitKey(30) & 0xFF

                if _click_point is not None:
                    # Text input mode — typing segment label
                    if key == 13:  # ENTER — confirm segment
                        seg = segment_shorthand(_text_input)
                        if seg is None:
                            print(f"  Invalid segment: '{_text_input}' — try again (e.g. t20, s5, dbull)")
                            _text_input = ""
                        else:
                            cls_name = make_class_name(_dart_ordinal, seg)
                            if cls_name not in CLASS_TO_ID:
                                print(f"  Unknown class: {cls_name}")
                                _text_input = ""
                            else:
                                px, py = _click_point
                                _annotations.append((px, py, cls_name))
                                info = parse_class_name(cls_name)
                                print(f"  Annotated: d{_dart_ordinal} {seg} at ({px}, {py}) — {info['label']}")
                                _click_point = None
                                _text_input = ""
                                _guess_segment = ""

                                # Auto-advance ordinal
                                if _dart_ordinal < 3:
                                    _dart_ordinal += 1

                    elif key == 27:  # ESC — cancel this click
                        _click_point = None
                        _text_input = ""
                        _guess_segment = ""
                        print("  Click cancelled")
                    elif key == 8:  # BACKSPACE
                        if _text_input:
                            _text_input = _text_input[:-1]
                            _guess_segment = ""  # user is editing, clear guess state
                    elif 32 <= key < 127:  # printable character
                        # If guess is showing and user starts typing, replace it
                        if _guess_segment and _text_input == _guess_segment.lower():
                            _text_input = ""
                            _guess_segment = ""
                        _text_input += chr(key)
                else:
                    # Not typing — handle frame-level keys
                    if key == 13:  # ENTER — save frame
                        if not _annotations:
                            print("  No annotations — click tips first, or ESC to discard")
                            continue

                        fname = f"frame_{frame_counter:05d}.png"
                        cv2.imwrite(str(img_dir / fname), frozen_frame)

                        # Write YOLO label file
                        label_name = f"frame_{frame_counter:05d}.txt"
                        with open(label_dir / label_name, "w") as lf:
                            for (ax, ay, cls_name) in _annotations:
                                cls_id = CLASS_TO_ID[cls_name]
                                cx = ax / w
                                cy = ay / h
                                bw = box_size / w
                                bh = box_size / h
                                lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                        # Also save to JSONL for bookkeeping
                        entry = {
                            "filename": fname,
                            "darts": [
                                {"x": ax, "y": ay, "class": cls_name}
                                for (ax, ay, cls_name) in _annotations
                            ],
                            "n_darts": len(_annotations),
                            "timestamp": time.time(),
                        }
                        with open(annotations_path, "a") as f:
                            f.write(json.dumps(entry) + "\n")

                        frame_counter += 1
                        print(f"  Saved: {fname} with {len(_annotations)} dart(s)")

                        # Update video trigger background so it sees next dart as new
                        if video:
                            video.absorb(frozen_frame)

                        # After 3rd dart, suppress trigger for dart removal
                        if _dart_ordinal >= 3:
                            if video:
                                video.suppress_next("Pull darts — trigger suppressed")
                            _dart_ordinal = 1
                            print("  Round complete — pull darts (trigger suppressed)")

                        frozen_frame = None
                        _active = False
                        _annotations = []
                        # Start post-capture settle
                        settle_active = True
                        settle_start_time = time.monotonic()

                    elif key == 27:  # ESC — discard
                        print("  Discarded")
                        frozen_frame = None
                        _active = False
                        _annotations = []
                        _click_point = None
                        _text_input = ""
                        _guess_segment = ""
                        # Start settle on discard too
                        settle_active = True
                        settle_start_time = time.monotonic()

                    elif key in (ord('z'),) and _annotations:
                        removed = _annotations.pop()
                        # Roll back ordinal
                        _dart_ordinal = max(1, _dart_ordinal - 1)
                        print(f"  Undo: removed {removed[2]} at ({removed[0]}, {removed[1]}), dart {_dart_ordinal}")

                    elif key == ord('r'):
                        _dart_ordinal = 1
                        _annotations = []
                        _click_point = None
                        _text_input = ""
                        _guess_segment = ""
                        print("  Round reset — dart 1")

    finally:
        save_window_sizes([win, panel_win])
        if audio:
            audio.stop()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone. {frame_counter} total frames saved to {outdir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect YOLO dart training data")
    parser.add_argument("--outdir", default="data/training",
                        help="Output directory (default: data/training)")
    parser.add_argument("--no-undistort", action="store_true",
                        help="Skip lens undistortion")
    parser.add_argument("--box-size", type=int, default=30,
                        help="Bounding box size in pixels (default: 30)")
    parser.add_argument("--trigger", choices=["audio", "video", "manual"],
                        default="audio",
                        help="Trigger mode: audio (ML dart sound), video (frame diff), manual (SPACE only)")
    parser.add_argument("--audio-device", type=int, default=None,
                        help="Audio input device index (default: auto-detect eMeet)")
    parser.add_argument("--audio-threshold", type=float, default=0.7,
                        help="Dart probability threshold for audio trigger (default: 0.7)")
    parser.add_argument("--video-threshold", type=float, default=5.0,
                        help="Mean pixel diff threshold for video trigger (default: 5.0)")
    parser.add_argument("--settle-delay", type=float, default=0.7,
                        help="Seconds to wait after trigger before freezing (default: 0.7)")
    args = parser.parse_args()

    collect_data(
        args.outdir,
        use_undistort=not args.no_undistort,
        box_size=args.box_size,
        trigger_mode=args.trigger,
        audio_device=args.audio_device,
        audio_threshold=args.audio_threshold,
        video_threshold=args.video_threshold,
        settle_delay=args.settle_delay,
    )
