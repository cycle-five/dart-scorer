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
    R       Reset round (back to dart 1, suppress trigger for dart removal)
    +/-     Adjust trigger threshold up/down
    Q       Quit
"""

import argparse
import json
import os
import time
from enum import Enum, auto
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import config
import board
from classes import (
    CLASS_TO_ID, make_class_name, segment_shorthand, parse_class_name,
)
from audio_trigger import DartAudioTrigger
from window_manager import create_window, save_window_sizes


# ---------------------------------------------------------------------------
# UI State Machine
# ---------------------------------------------------------------------------

class UIState(Enum):
    WARMUP = auto()       # video trigger warming up, triggers disabled
    LISTENING = auto()    # live feed, triggers armed
    PRE_CAPTURE = auto()  # trigger fired, settling before freeze
    ANNOTATING = auto()   # frame frozen, user annotating
    COOLDOWN = auto()     # post-save/discard, brief trigger suppression
    PULL_DARTS = auto()   # user pulling darts, suppress until board stable


# ---------------------------------------------------------------------------
# Video trigger
# ---------------------------------------------------------------------------

class VideoTrigger:
    """Detects darts by frame differencing on heavily downsampled frames."""

    THUMB_SIZE = (80, 60)
    CELL_THRESHOLD = 12
    SUPPRESS_CALM_REQUIRED = 15  # consecutive calm frames to clear suppression

    def __init__(self, threshold=8, cooldown=1.5, warmup_seconds=5.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self.warmup_seconds = warmup_seconds
        self._background = None
        self._frame_count = 0
        self._start_time = time.monotonic()
        self._last_trigger_time = 0.0
        self._current_diff = 0.0
        self._baseline_diff = 0.0
        self._raw_changed = 0
        self._diff_history = []
        self._history_max = 200
        self.triggered = False
        self._suppress_calm_count = 0

    def _to_thumb(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, self.THUMB_SIZE, interpolation=cv2.INTER_AREA)

    def update(self, frame):
        """Feed a new frame. Always call this, regardless of UI state."""
        thumb = self._to_thumb(frame)
        self._frame_count += 1

        if self._background is None:
            self._background = thumb.astype(np.float32)
            return

        cv2.accumulateWeighted(thumb, self._background, 0.02)
        bg = self._background.astype(np.uint8)

        diff = cv2.absdiff(thumb, bg)
        changed_cells = int(np.sum(diff > self.CELL_THRESHOLD))
        self._raw_changed = changed_cells

        if changed_cells < self._baseline_diff * 3 + 2:
            self._baseline_diff = 0.95 * self._baseline_diff + 0.05 * changed_cells

        spike = max(0, changed_cells - self._baseline_diff)
        self._current_diff = spike

        self._diff_history.append(spike)
        if len(self._diff_history) > self._history_max:
            self._diff_history.pop(0)

    def check_trigger(self):
        """Check if trigger should fire. Only call in LISTENING state."""
        now = time.monotonic()
        if now - self._start_time < self.warmup_seconds:
            return False
        if (self._current_diff > self.threshold
                and now - self._last_trigger_time > self.cooldown):
            self._last_trigger_time = now
            return True
        return False

    def is_calm(self):
        """Check if the scene has been calm long enough to exit PULL_DARTS."""
        if self._current_diff < self.threshold * 0.5:
            self._suppress_calm_count += 1
        else:
            self._suppress_calm_count = 0
        return self._suppress_calm_count >= self.SUPPRESS_CALM_REQUIRED

    def reset_calm_counter(self):
        """Reset the calm counter (call when entering PULL_DARTS)."""
        self._suppress_calm_count = 0

    def absorb(self, frame):
        """Reset background to current frame."""
        thumb = self._to_thumb(frame)
        self._background = thumb.astype(np.float32)

    @property
    def warmup_remaining(self):
        return max(0, self.warmup_seconds - (time.monotonic() - self._start_time))

    @property
    def current_diff(self):
        return self._current_diff


# ---------------------------------------------------------------------------
# Annotation session
# ---------------------------------------------------------------------------

class AnnotationSession:
    """Encapsulates annotation state for a single frozen frame."""

    def __init__(self, homography=None, previous_annotations=None, start_ordinal=1,
                 crop_offset=(0, 0)):
        self.homography = homography
        self.crop_offset = crop_offset
        self.annotations = list(previous_annotations or [])
        self.dart_ordinal = start_ordinal
        self.click_point = None
        self.text_input = ""
        self.guess_segment = ""

    @property
    def num_new(self):
        """Number of annotations added in this session (not carried forward)."""
        return len(self.annotations) - self._carry_count

    def _init_carry_count(self, n):
        self._carry_count = n

    def handle_click(self, x, y):
        """Process a mouse click on the image."""
        self.click_point = (x, y)
        guess = _guess_segment_from_homography(x, y, self.homography, self.crop_offset)
        if guess:
            self.guess_segment = guess
            self.text_input = guess.lower()
        else:
            self.guess_segment = ""
            self.text_input = ""

    def confirm_segment(self):
        """Try to confirm the current text input as a segment label.
        Returns (success, message) tuple."""
        seg = segment_shorthand(self.text_input)
        if seg is None:
            msg = f"Invalid segment: '{self.text_input}'"
            self.text_input = ""
            return False, msg

        cls_name = make_class_name(self.dart_ordinal, seg)
        if cls_name not in CLASS_TO_ID:
            msg = f"Unknown class: {cls_name}"
            self.text_input = ""
            return False, msg

        px, py = self.click_point
        self.annotations.append((px, py, cls_name))
        info = parse_class_name(cls_name)
        msg = f"Annotated: d{self.dart_ordinal} {seg} at ({px}, {py}) — {info['label']}"

        self.click_point = None
        self.text_input = ""
        self.guess_segment = ""

        # Auto-advance ordinal
        if self.dart_ordinal < 3:
            self.dart_ordinal += 1

        return True, msg

    def cancel_click(self):
        self.click_point = None
        self.text_input = ""
        self.guess_segment = ""

    def undo(self):
        """Undo last annotation. Returns removed annotation or None."""
        if not self.annotations:
            return None
        removed = self.annotations.pop()
        self.dart_ordinal = max(1, self.dart_ordinal - 1)
        return removed

    def handle_key(self, key):
        """Handle a keypress during text input. Returns True if consumed."""
        if key == 8:  # BACKSPACE
            if self.text_input:
                self.text_input = self.text_input[:-1]
                self.guess_segment = ""
            return True
        elif 32 <= key < 127:
            if self.guess_segment and self.text_input == self.guess_segment.lower():
                self.text_input = ""
                self.guess_segment = ""
            self.text_input += chr(key)
            return True
        return False


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _guess_segment_from_homography(x, y, homography, crop_offset=(0, 0)):
    """Use board homography to guess which segment a click is in.

    Args:
        x, y: Click coordinates in the (possibly cropped) image.
        homography: 3x3 homography matrix (computed on full-frame coords).
        crop_offset: (x_offset, y_offset) to map cropped coords back to full frame.
    """
    if homography is None:
        return None
    try:
        # Map cropped coords back to full-frame coords for homography
        full_x = x + crop_offset[0]
        full_y = y + crop_offset[1]
        score_info = board.score_from_camera((full_x, full_y), homography)
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


def _draw_waveform(canvas, x, y, w, h, history, threshold, history_max):
    """Draw a scrolling waveform with threshold line."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (50, 50, 50), -1)
    y_scale = max(threshold * 2, max(history) * 1.2 if history else 1.0, 1.0)

    # Threshold line
    ty = y + h - int((threshold / y_scale) * h)
    ty = max(y + 1, min(y + h - 1, ty))
    cv2.line(canvas, (x, ty), (x + w, ty), (0, 255, 255), 1)

    if len(history) > 1:
        for i in range(1, len(history)):
            x1 = x + int((i - 1) / history_max * w)
            x2 = x + int(i / history_max * w)
            y1 = y + h - int((history[i-1] / y_scale) * h)
            y2 = y + h - int((history[i] / y_scale) * h)
            y1 = max(y + 1, min(y + h - 1, y1))
            y2 = max(y + 1, min(y + h - 1, y2))
            color = (0, 0, 255) if history[i] > threshold else (0, 180, 0)
            cv2.line(canvas, (x1, y1), (x2, y2), color, 1)


def render_control_panel(state, video, audio, session, frame_counter, settle_remaining=0):
    """Render the control panel for any UI state."""
    cp_w, cp_h = 400, 350
    cp = np.zeros((cp_h, cp_w, 3), dtype=np.uint8)
    cp[:] = (30, 30, 30)
    cp_y = 15

    # Title
    trigger_name = "VIDEO" if video else ("AUDIO" if audio else "MANUAL")
    cv2.putText(cp, f"Trigger: {trigger_name}", (10, cp_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cp_y += 25

    # Waveform
    wave_x, wave_w, wave_h = 10, cp_w - 20, 100
    wave_y = cp_y

    if video:
        _draw_waveform(cp, wave_x, wave_y, wave_w, wave_h,
                       video._diff_history, video.threshold, video._history_max)
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

        cv2.rectangle(cp, (wave_x, wave_y), (wave_x + wave_w, wave_y + wave_h),
                      (50, 50, 50), -1)
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

    # State
    cp_y += 10
    state_labels = {
        UIState.WARMUP: ("WARMING UP...", (100, 100, 100)),
        UIState.LISTENING: ("LISTENING", (0, 255, 0)),
        UIState.PRE_CAPTURE: (f"DART DETECTED — settling {settle_remaining:.1f}s", (0, 200, 255)),
        UIState.ANNOTATING: ("ANNOTATING", (0, 0, 255)),
        UIState.COOLDOWN: ("COOLDOWN", (100, 100, 100)),
        UIState.PULL_DARTS: ("PULL DARTS — suppressed", (0, 140, 255)),
    }
    state_text, state_color = state_labels.get(state, ("???", (255, 255, 255)))
    cv2.putText(cp, state_text, (10, cp_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
    cp_y += 25

    # Session info
    dart_ord = session.dart_ordinal if session else 1
    cv2.putText(cp, f"Dart: {dart_ord}/3    Saved: {frame_counter}",
                (10, cp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cp_y += 25

    if state == UIState.ANNOTATING and session:
        cv2.putText(cp, f"Annotated: {len(session.annotations)}",
                    (10, cp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cp_y += 30
        if session.click_point is not None:
            if session.guess_segment and session.text_input == session.guess_segment.lower():
                cv2.putText(cp, f"Guess: [{session.guess_segment}]", (10, cp_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cp_y += 22
                cv2.putText(cp, "ENTER=accept  or type correction", (10, cp_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            else:
                cv2.putText(cp, f"Input: {session.text_input}_", (10, cp_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cp_y += 22
                cv2.putText(cp, "Type segment (t20, s5, dbull) + ENTER", (10, cp_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        else:
            cv2.putText(cp, "Click new dart tip", (10, cp_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cp_y += 25

    # Controls
    cv2.putText(cp, "SPACE=capture  R=reset  +/-=threshold  Q=quit", (10, cp_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    return cp


def render_camera_hud(display, state, session, frame_counter, settle_remaining=0):
    """Draw minimal HUD overlay on camera view."""
    h, w = display.shape[:2]
    dart_ord = session.dart_ordinal if session else 1

    # Top status bar
    labels = {
        UIState.WARMUP: (f"WARMING UP...", (100, 100, 100)),
        UIState.LISTENING: (f"LIVE  |  Saved: {frame_counter}  |  Dart: {dart_ord}/3", (0, 255, 0)),
        UIState.PRE_CAPTURE: (f"DART! Settling... {settle_remaining:.1f}s", (0, 200, 255)),
        UIState.ANNOTATING: (f"FROZEN  |  Dart: {dart_ord}/3  |  Annotated: {len(session.annotations) if session else 0}", (0, 0, 255)),
        UIState.COOLDOWN: (f"COOLDOWN  |  Dart: {dart_ord}/3", (100, 100, 100)),
        UIState.PULL_DARTS: (f"PULL DARTS  |  Dart: {dart_ord}/3", (0, 140, 255)),
    }
    text, color = labels.get(state, ("", (255, 255, 255)))
    cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Bottom status
    cv2.putText(display, f"Saved: {frame_counter}  |  Dart: {dart_ord}/3",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


def render_annotations(display, session, box_size):
    """Draw annotations and pending click on the camera view."""
    if session is None:
        return

    colors = {1: (0, 255, 0), 2: (0, 255, 255), 3: (0, 0, 255)}
    half = box_size // 2

    for (ax, ay, cls_name) in session.annotations:
        info = parse_class_name(cls_name)
        color = colors.get(info["ordinal"], (255, 255, 255))
        cv2.rectangle(display, (ax - half, ay - half), (ax + half, ay + half), color, 2)
        cv2.circle(display, (ax, ay), 3, color, -1)
        label = f"d{info['ordinal']} {info['segment']}"
        cv2.putText(display, label, (ax + half + 4, ay + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if session.click_point is not None:
        px, py = session.click_point
        cv2.circle(display, (px, py), 7, (255, 0, 255), 2)
        cv2.circle(display, (px, py), 2, (255, 0, 255), -1)
        if session.guess_segment and session.text_input == session.guess_segment.lower():
            text = f"Dart {session.dart_ordinal} > [{session.guess_segment}]  ENTER=accept / type to correct"
            cv2.putText(display, text, (px + 12, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            text = f"Dart {session.dart_ordinal} > {session.text_input}_"
            cv2.putText(display, text, (px + 12, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    h = display.shape[0]
    if session.click_point is not None:
        if session.guess_segment and session.text_input == session.guess_segment.lower():
            cv2.putText(display, f"ENTER=accept [{session.guess_segment}]  or type correction  ESC=cancel",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        else:
            cv2.putText(display, f"Type segment (e.g. t20, s5, dbull) then ENTER  ESC=cancel",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 1)
    else:
        cv2.putText(display, "Click=mark tip  ENTER=save  ESC=discard  Z=undo  R=reset",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ---------------------------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------------------------

_session_ref = [None]  # mutable container for mouse callback access


def _mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        session = _session_ref[0]
        if session is not None:
            session.handle_click(x, y)


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_data(outdir="data/training", use_undistort=True, box_size=30,
                 trigger_mode="audio", audio_device=None, audio_threshold=0.7,
                 video_threshold=5.0, settle_delay=0.7):

    outdir = Path(outdir)
    img_dir = outdir / "images"
    label_dir = outdir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = outdir / "annotations.jsonl"

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
    crop_offset = (0, 0)
    if crop_roi is not None:
        x, y, w, h = crop_roi
        crop_offset = (x, y)
        print(f"Crop ROI: ({x}, {y}) {w}x{h}")
    else:
        print("No crop ROI — using full frame")

    # Board homography for segment guessing
    homography = None
    if config.BOARD_HOMOGRAPHY_PATH.exists():
        hom_data = np.load(str(config.BOARD_HOMOGRAPHY_PATH))
        homography = hom_data['homography']
        print("Board homography loaded — segment guess enabled")
    else:
        print("No board homography — type segments manually")

    # Triggers
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

    # Windows
    win = "Collect Training Data"
    panel_win = "Control Panel"
    create_window(win, default_width=960, default_height=540)
    create_window(panel_win, default_width=400, default_height=350)
    cv2.setMouseCallback(win, _mouse_callback)

    # Print controls
    print(f"\n=== YOLO Training Data Collection (186 classes) ===")
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

    # --- State machine ---
    state = UIState.WARMUP if video else UIState.LISTENING
    frozen_frame = None
    session = None           # AnnotationSession, set when ANNOTATING
    _session_ref[0] = None
    previous_annotations = []
    dart_ordinal = 1
    pre_capture_start = 0.0
    pre_capture_source = ""
    cooldown_start = 0.0

    # Dummy session for rendering when not annotating
    class _DummySession:
        def __init__(self):
            self.dart_ordinal = 1
            self.annotations = []
            self.click_point = None

    dummy = _DummySession()

    try:
        while True:
            # Read frame
            ret, raw = cap.read()
            if not ret:
                print("Camera read failed")
                break

            frame = raw
            if lens_params is not None:
                frame = undistort_frame(raw, *lens_params)
            frame = apply_crop(frame, crop_roi)

            # Always update video trigger signal
            if video:
                video.update(frame)

            # --- State transitions ---
            settle_remaining = 0

            if state == UIState.WARMUP:
                if video and video.warmup_remaining <= 0:
                    state = UIState.LISTENING
                    print("Warmup complete — listening")
                elif not video:
                    state = UIState.LISTENING

            elif state == UIState.LISTENING:
                # Check triggers
                triggered = False
                source = ""
                if audio and audio.check_and_reset():
                    triggered = True
                    source = "AUDIO"
                elif video and video.check_trigger():
                    triggered = True
                    source = "VIDEO"

                if triggered:
                    state = UIState.PRE_CAPTURE
                    pre_capture_start = time.monotonic()
                    pre_capture_source = source
                    print(f"\n  [{source}] Dart detected — settling {settle_delay}s...")

            elif state == UIState.PRE_CAPTURE:
                elapsed = time.monotonic() - pre_capture_start
                settle_remaining = max(0, settle_delay - elapsed)
                if elapsed >= settle_delay:
                    # Freeze the current (settled) frame
                    frozen_frame = frame.copy()
                    session = AnnotationSession(
                        homography=homography,
                        previous_annotations=previous_annotations,
                        start_ordinal=dart_ordinal,
                        crop_offset=crop_offset,
                    )
                    session._init_carry_count(len(previous_annotations))
                    _session_ref[0] = session
                    state = UIState.ANNOTATING
                    if previous_annotations:
                        print(f"[{pre_capture_source}] Frame captured — {len(previous_annotations)} dart(s) carried forward, annotate dart {dart_ordinal}")
                    else:
                        print(f"[{pre_capture_source}] Frame captured — annotating dart {dart_ordinal}")

            elif state == UIState.COOLDOWN:
                elapsed = time.monotonic() - cooldown_start
                if elapsed >= settle_delay:
                    state = UIState.LISTENING

            elif state == UIState.PULL_DARTS:
                if video:
                    if video.is_calm():
                        state = UIState.LISTENING
                        print("  Board stable — listening")
                else:
                    # No video trigger, just wait a fixed time
                    if time.monotonic() - cooldown_start >= 3.0:
                        state = UIState.LISTENING

            # ANNOTATING state is handled in the key processing below

            # --- Render ---
            display = (frozen_frame if frozen_frame is not None else frame).copy()

            render_session = session if state == UIState.ANNOTATING else dummy
            dummy.dart_ordinal = dart_ordinal

            cp = render_control_panel(state, video, audio, render_session,
                                      frame_counter, settle_remaining)
            cv2.imshow(panel_win, cp)

            render_camera_hud(display, state, render_session, frame_counter, settle_remaining)

            if state == UIState.ANNOTATING and session:
                render_annotations(display, session, box_size)

            cv2.imshow(win, display)

            # --- Key handling ---
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' ') and state in (UIState.LISTENING, UIState.WARMUP,
                                                UIState.COOLDOWN, UIState.PULL_DARTS):
                # Manual capture — immediate freeze
                frozen_frame = frame.copy()
                session = AnnotationSession(
                    homography=homography,
                    previous_annotations=previous_annotations,
                    start_ordinal=dart_ordinal,
                    crop_offset=crop_offset,
                )
                session._init_carry_count(len(previous_annotations))
                _session_ref[0] = session
                state = UIState.ANNOTATING
                print(f"[MANUAL] Frame captured — annotating dart {dart_ordinal}")

            elif key == ord('r') and state != UIState.ANNOTATING:
                # Round reset from live states
                dart_ordinal = 1
                previous_annotations = []
                if video:
                    video.reset_calm_counter()
                    state = UIState.PULL_DARTS
                    cooldown_start = time.monotonic()
                    print("Round reset — pull darts (trigger suppressed)")
                else:
                    state = UIState.LISTENING
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

            elif state == UIState.ANNOTATING and session:
                if session.click_point is not None:
                    # Text input mode
                    if key == 13:  # ENTER — confirm segment
                        ok, msg = session.confirm_segment()
                        print(f"  {msg}")
                        if not ok:
                            print("  Try again (e.g. t20, s5, dbull)")
                    elif key == 27:  # ESC — cancel click
                        session.cancel_click()
                        print("  Click cancelled")
                    else:
                        session.handle_key(key)
                else:
                    # Frame-level keys
                    if key == 13:  # ENTER — save
                        if not session.annotations:
                            print("  No annotations — click tips first, or ESC to discard")
                            continue

                        h_img, w_img = frozen_frame.shape[:2]
                        fname = f"frame_{frame_counter:05d}.png"
                        cv2.imwrite(str(img_dir / fname), frozen_frame)

                        label_name = f"frame_{frame_counter:05d}.txt"
                        with open(label_dir / label_name, "w") as lf:
                            for (ax, ay, cls_name) in session.annotations:
                                cls_id = CLASS_TO_ID[cls_name]
                                cx = ax / w_img
                                cy = ay / h_img
                                bw = box_size / w_img
                                bh = box_size / h_img
                                lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                        entry = {
                            "filename": fname,
                            "darts": [{"x": ax, "y": ay, "class": cn}
                                      for (ax, ay, cn) in session.annotations],
                            "n_darts": len(session.annotations),
                            "timestamp": time.time(),
                        }
                        with open(annotations_path, "a") as f:
                            f.write(json.dumps(entry) + "\n")

                        frame_counter += 1
                        print(f"  Saved: {fname} with {len(session.annotations)} dart(s)")

                        # Carry forward
                        previous_annotations = list(session.annotations)
                        dart_ordinal = session.dart_ordinal

                        # Update video background
                        if video:
                            video.absorb(frozen_frame)

                        # Transition
                        frozen_frame = None
                        _session_ref[0] = None
                        session = None
                        cooldown_start = time.monotonic()
                        state = UIState.COOLDOWN

                    elif key == 27:  # ESC — discard
                        print("  Discarded")
                        frozen_frame = None
                        _session_ref[0] = None
                        session = None
                        cooldown_start = time.monotonic()
                        state = UIState.COOLDOWN

                    elif key == ord('z'):
                        removed = session.undo()
                        if removed:
                            print(f"  Undo: removed {removed[2]} at ({removed[0]}, {removed[1]}), dart {session.dart_ordinal}")

                    elif key == ord('r'):
                        # Round reset while annotating — discard and go to PULL_DARTS
                        dart_ordinal = 1
                        previous_annotations = []
                        frozen_frame = None
                        _session_ref[0] = None
                        session = None
                        if video:
                            video.reset_calm_counter()
                            state = UIState.PULL_DARTS
                            cooldown_start = time.monotonic()
                        else:
                            state = UIState.LISTENING
                        print("  Round reset — pull darts")

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
