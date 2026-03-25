#!/usr/bin/env python3
"""
collect.py — Training data collection for YOLO dart detection + scoring.

Batch collection workflow:
  1. Throw up to 3 darts — each impact is auto-captured
  2. After timeout or 3 darts, annotate each frame in sequence
  3. Previous darts carry forward automatically
  4. Press R to reset round, pull darts, throw again

Trigger modes:
  audio  — ML classifier on camera mic detects dart impact sound (default)
  video  — frame differencing detects visual change
  manual — SPACE key only

Usage:
    python collect.py                          # Audio trigger (default)
    python collect.py --trigger video          # Video trigger
    python collect.py --trigger manual         # SPACE only
    python collect.py --no-undistort           # Skip lens undistortion
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
from classes_v2 import (
    CLASS_TO_ID,
    CLASS_NAMES,
    NUM_CLASSES,
    make_class_name,
    segment_shorthand,
    parse_class_name,
    ring_from_segment,
    sector_from_segment,
)
from audio_trigger import DartAudioTrigger
from window_manager import create_window, save_window_sizes


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def auto_expand_bbox(tip_x, tip_y, img_w, img_h, board_center=None):
    """Expand a tip click to a full-dart bounding box.

    Extends along the radial direction (away from board center) to capture
    the dart shaft. Returns (cx, cy, w, h) in normalized [0,1] coordinates.
    """
    import math

    EXPAND_AWAY = int(round(config.BBOX_EXPAND_AWAY_FRAC * img_w))
    EXPAND_TOWARD = int(round(config.BBOX_EXPAND_TOWARD_FRAC * img_w))
    EXPAND_LATERAL = int(round(config.BBOX_EXPAND_LATERAL_FRAC * img_h))
    MIN_BOX = int(round(config.BBOX_MIN_SIZE_FRAC * img_w))

    if board_center is not None:
        bcx, bcy = board_center
        dx = bcx - tip_x
        dy = bcy - tip_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 1:
            dx /= dist; dy /= dist
        else:
            dx, dy = 0, -1
    else:
        dx = img_w / 2 - tip_x
        dy = img_h / 2 - tip_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 1:
            dx /= dist; dy /= dist
        else:
            dx, dy = 0, -1

    perp_x, perp_y = -dy, dx

    corners_x = [
        tip_x + dx * EXPAND_TOWARD,
        tip_x - dx * EXPAND_AWAY,
        tip_x + perp_x * EXPAND_LATERAL,
        tip_x - perp_x * EXPAND_LATERAL,
    ]
    corners_y = [
        tip_y + dy * EXPAND_TOWARD,
        tip_y - dy * EXPAND_AWAY,
        tip_y + perp_y * EXPAND_LATERAL,
        tip_y - perp_y * EXPAND_LATERAL,
    ]

    x1 = max(0, min(corners_x))
    y1 = max(0, min(corners_y))
    x2 = min(img_w, max(corners_x))
    y2 = min(img_h, max(corners_y))

    if x2 - x1 < MIN_BOX:
        pad = (MIN_BOX - (x2 - x1)) / 2
        x1 = max(0, x1 - pad); x2 = min(img_w, x2 + pad)
    if y2 - y1 < MIN_BOX:
        pad = (MIN_BOX - (y2 - y1)) / 2
        y1 = max(0, y1 - pad); y2 = min(img_h, y2 + pad)

    return (x1 + x2) / (2 * img_w), (y1 + y2) / (2 * img_h), (x2 - x1) / img_w, (y2 - y1) / img_h


def geometry_classify(tip_x, tip_y, homography, crop_offset=(0, 0)):
    """Classify a dart tip using board geometry.

    Returns segment name (e.g., "T20", "S_BULL", "MISS") or None on failure.
    """
    try:
        full_x = tip_x + crop_offset[0]
        full_y = tip_y + crop_offset[1]
        can_x, can_y = board.apply_homography((full_x, full_y), homography)
        r, theta = board.pixel_to_polar(can_x, can_y)
        ring_name, _ = board.get_ring(r)

        if ring_name == "D-BULL":
            return "D_BULL"
        if ring_name == "S-BULL":
            return "S_BULL"
        if ring_name == "miss":
            return "MISS"

        sector = board.get_sector(theta)
        ring_code = {"single": "S", "double": "D", "triple": "T"}[ring_name]
        return f"{ring_code}{sector}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# UI State Machine
# ---------------------------------------------------------------------------


class UIState(Enum):
    WARMUP = auto()  # video trigger warming up
    LISTENING = auto()  # waiting for first dart of round
    COLLECTING = auto()  # capturing darts (up to 3), waiting for next or timeout
    SETTLING = auto()  # dart detected, waiting for frame to settle
    ANNOTATING = auto()  # frozen on a frame, user clicking tips
    REVIEW = auto()  # all tips clicked, showing labels, wait for pull or X to edit
    PULL_DARTS = auto()  # round done, suppress until board stable
    PAUSED = auto()  # collection paused, live feed still shows


# ---------------------------------------------------------------------------
# Video trigger
# ---------------------------------------------------------------------------


class VideoTrigger:
    """Detects darts by frame differencing on heavily downsampled frames."""

    THUMB_SIZE = config.VIDEO_TRIGGER_THUMB_SIZE
    CELL_THRESHOLD = config.VIDEO_TRIGGER_CELL_THRESHOLD
    SUPPRESS_CALM_REQUIRED = config.VIDEO_TRIGGER_SUPPRESS_CALM

    def __init__(self, threshold=5, cooldown=config.VIDEO_TRIGGER_COOLDOWN,
                 warmup_seconds=config.VIDEO_TRIGGER_WARMUP):
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
        self._history_max = config.VIDEO_TRIGGER_HISTORY_MAX
        self._suppress_calm_count = 0
        self._saw_pull_disturbance = False

    def _to_thumb(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, self.THUMB_SIZE, interpolation=cv2.INTER_AREA)

    def update(self, frame):
        """Feed a new frame. Always call this."""
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
        """Check if trigger should fire. Only call when triggers are armed."""
        now = time.monotonic()
        if now - self._start_time < self.warmup_seconds:
            return False
        if (
            self._current_diff > self.threshold
            and now - self._last_trigger_time > self.cooldown
        ):
            self._last_trigger_time = now
            return True
        return False

    def is_calm(self):
        """Check if scene has been calm long enough (for PULL_DARTS exit)."""
        if self._current_diff < self.threshold * 0.5:
            self._suppress_calm_count += 1
        else:
            self._suppress_calm_count = 0
        return self._suppress_calm_count >= self.SUPPRESS_CALM_REQUIRED

    def saw_disturbance(self):
        """Check if a significant disturbance occurred (hand reaching in)."""
        return self._current_diff > self.threshold

    def reset_calm_counter(self):
        self._suppress_calm_count = 0
        self._saw_pull_disturbance = False

    @property
    def pull_disturbance_seen(self):
        return self._saw_pull_disturbance

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
    """Annotation state for a single frozen frame."""

    def __init__(
        self,
        homography=None,
        previous_annotations=None,
        crop_offset=(0, 0),
    ):
        self.homography = homography
        self.crop_offset = crop_offset
        self.annotations = list(previous_annotations or [])
        self.click_point = None
        self.text_input = ""
        self.guess_segment = ""
        self.last_auto_msg = ""
        self._carry_count = len(previous_annotations or [])

    @property
    def dart_ordinal(self):
        """1-based index of next dart to annotate (for HUD display)."""
        return len(self.annotations) + 1

    def handle_click(self, x, y):
        """Process a click. Auto-confirms if homography provides a guess."""
        guess = _guess_segment_from_homography(x, y, self.homography, self.crop_offset)
        if guess and guess in CLASS_TO_ID:
            # Auto-confirm: add annotation directly, no ENTER needed
            dart_num = len(self.annotations) + 1
            self.annotations.append((x, y, guess))
            info = parse_class_name(guess)
            self.last_auto_msg = f"dart {dart_num} {guess} — {info['label']}"
            self.click_point = None
            self.text_input = ""
            self.guess_segment = ""
            return True  # auto-confirmed
        # No guess or invalid — fall back to manual input
        self.click_point = (x, y)
        self.guess_segment = guess or ""
        self.text_input = guess.lower() if guess else ""
        return False  # needs manual input

    def confirm_segment(self):
        """Try to confirm current input. Returns (success, message)."""
        seg = segment_shorthand(self.text_input)
        if seg is None:
            msg = f"Invalid segment: '{self.text_input}'"
            self.text_input = ""
            return False, msg
        if seg not in CLASS_TO_ID:
            msg = f"Unknown class: {seg}"
            self.text_input = ""
            return False, msg
        px, py = self.click_point
        dart_num = len(self.annotations) + 1
        self.annotations.append((px, py, seg))
        info = parse_class_name(seg)
        msg = f"dart {dart_num} {seg} at ({px}, {py}) — {info['label']}"
        self.click_point = None
        self.text_input = ""
        self.guess_segment = ""
        return True, msg

    def cancel_click(self):
        self.click_point = None
        self.text_input = ""
        self.guess_segment = ""

    def undo(self):
        if not self.annotations:
            return None
        return self.annotations.pop()

    def handle_key(self, key):
        """Handle keypress during text input. Returns True if consumed."""
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
# Rendering
# ---------------------------------------------------------------------------


def _guess_segment_from_homography(x, y, homography, crop_offset=(0, 0)):
    if homography is None:
        return None
    try:
        full_x = x + crop_offset[0]
        full_y = y + crop_offset[1]
        canonical = board.apply_homography((full_x, full_y), homography)
        r, theta = board.pixel_to_polar(canonical[0], canonical[1])

        # Use angle to determine sector (always works, even if r is off)
        sector = board.get_sector(theta)

        # Use radius to determine ring, with generous tolerance
        # (clicking precision at the outer wire is inherently imprecise)
        if r < config.INNER_BULL_RADIUS + config.GUESS_BULL_TOLERANCE:
            return "D_BULL"
        elif r < config.OUTER_BULL_RADIUS + config.GUESS_BULL_TOLERANCE:
            return "S_BULL"
        elif config.TRIPLE_INNER_RADIUS - config.GUESS_TRIPLE_TOLERANCE < r < config.TRIPLE_OUTER_RADIUS + config.GUESS_TRIPLE_TOLERANCE:
            return f"T{sector}"
        elif config.DOUBLE_INNER_RADIUS - config.GUESS_DOUBLE_TOLERANCE_INNER < r < config.DOUBLE_OUTER_RADIUS + config.GUESS_DOUBLE_TOLERANCE_OUTER:
            return f"D{sector}"
        elif r > config.DOUBLE_OUTER_RADIUS + config.GUESS_MISS_TOLERANCE:
            return "MISS"
        else:
            return f"S{sector}"
    except Exception:
        return None


def _draw_waveform(canvas, x, y, w, h, history, threshold, history_max):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (50, 50, 50), -1)
    y_scale = max(threshold * 2, max(history) * 1.2 if history else 1.0, 1.0)
    ty = y + h - int((threshold / y_scale) * h)
    ty = max(y + 1, min(y + h - 1, ty))
    cv2.line(canvas, (x, ty), (x + w, ty), (0, 255, 255), 1)
    if len(history) > 1:
        for i in range(1, len(history)):
            x1 = x + int((i - 1) / history_max * w)
            x2 = x + int(i / history_max * w)
            y1 = y + h - int((history[i - 1] / y_scale) * h)
            y2 = y + h - int((history[i] / y_scale) * h)
            y1 = max(y + 1, min(y + h - 1, y1))
            y2 = max(y + 1, min(y + h - 1, y2))
            color = (0, 0, 255) if history[i] > threshold else (0, 180, 0)
            cv2.line(canvas, (x1, y1), (x2, y2), color, 1)


def render_control_panel(
    state,
    video,
    audio,
    session,
    frame_counter,
    batch_frames,
    batch_index,
    dart_ordinal,
    settle_remaining=0,
    collect_remaining=0,
    edit_dart=None,
    edit_text="",
    prediction_confidences=None,
):
    cp_w, cp_h = 400, 380
    cp = np.zeros((cp_h, cp_w, 3), dtype=np.uint8)
    cp[:] = (30, 30, 30)
    cp_y = 15

    trigger_name = "VIDEO" if video else ("AUDIO" if audio else "MANUAL")
    cv2.putText(
        cp,
        f"Trigger: {trigger_name}",
        (10, cp_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )
    cp_y += 25

    # Waveform
    wave_x, wave_w, wave_h = 10, cp_w - 20, 80
    wave_y = cp_y
    if video:
        _draw_waveform(
            cp,
            wave_x,
            wave_y,
            wave_w,
            wave_h,
            video._diff_history,
            video.threshold,
            video._history_max,
        )
        cp_y = wave_y + wave_h + 5
        cv2.putText(
            cp,
            f"cells={video.current_diff:.0f}  base={video._baseline_diff:.0f}  thresh={video.threshold}",
            (wave_x, cp_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 200, 200),
            1,
        )
        cp_y += 22
    elif audio:
        cv2.rectangle(
            cp, (wave_x, wave_y), (wave_x + wave_w, wave_y + wave_h), (50, 50, 50), -1
        )
        rms_history = audio._rms_history
        prob_history = audio._prob_history
        rms_h = wave_h // 2 - 3
        y_scale = max(audio._peak_rms * 1.2, 0.05)
        if len(rms_history) > 1:
            hmax = audio._history_max
            for i in range(1, len(rms_history)):
                x1 = wave_x + int((i - 1) / hmax * wave_w)
                x2 = wave_x + int(i / hmax * wave_w)
                y1 = wave_y + rms_h - int((rms_history[i - 1] / y_scale) * rms_h)
                y2 = wave_y + rms_h - int((rms_history[i] / y_scale) * rms_h)
                y1 = max(wave_y + 1, min(wave_y + rms_h, y1))
                y2 = max(wave_y + 1, min(wave_y + rms_h, y2))
                cv2.line(cp, (x1, y1), (x2, y2), (0, 200, 0), 1)
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
                y1 = prob_y + prob_h - int(prob_history[i - 1] * prob_h)
                y2 = prob_y + prob_h - int(prob_history[i] * prob_h)
                y1 = max(prob_y, min(prob_y + prob_h, y1))
                y2 = max(prob_y, min(prob_y + prob_h, y2))
                color = (
                    (0, 0, 255)
                    if prob_history[i] > audio.prob_threshold
                    else (200, 100, 0)
                )
                cv2.line(cp, (x1, y1), (x2, y2), color, 1)
        cp_y = wave_y + wave_h + 5
        cv2.putText(
            cp,
            f"P(dart)={audio.current_prob:.2f}  thresh={audio.prob_threshold:.2f}",
            (wave_x, cp_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 200, 200),
            1,
        )
        cp_y += 22
    else:
        cp_y = wave_y + wave_h + 5

    # State + batch info
    cp_y += 8
    state_info = {
        UIState.WARMUP: ("WARMING UP...", (100, 100, 100)),
        UIState.LISTENING: ("LISTENING — throw dart 1", (0, 255, 0)),
        UIState.SETTLING: (f"DART! Settling {settle_remaining:.1f}s", (0, 200, 255)),
        UIState.COLLECTING: (
            f"COLLECTING — {len(batch_frames)}/3 captured ({collect_remaining:.0f}s)",
            (0, 200, 255),
        ),
        UIState.ANNOTATING: (
            f"Click dart tips — {len(session.annotations) - session._carry_count if session else 0}/{len(batch_frames)} done",
            (0, 0, 255),
        ),
        UIState.REVIEW: ("DONE — pull darts or X to edit", (0, 255, 0)),
        UIState.PULL_DARTS: (
            "PULL DARTS — "
            + (
                "waiting for pull"
                if video and not video._saw_pull_disturbance
                else "stabilizing..."
            )
            if video
            else "PULL DARTS",
            (0, 140, 255),
        ),
        UIState.PAUSED: ("PAUSED — press P to resume", (128, 128, 128)),
    }
    text, color = state_info.get(state, ("", (255, 255, 255)))
    cv2.putText(cp, text, (10, cp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cp_y += 25

    cv2.putText(
        cp,
        f"Dart: {dart_ordinal}/3    Saved: {frame_counter}",
        (10, cp_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
    )
    cp_y += 25

    # Show labels prominently in REVIEW
    if state == UIState.REVIEW and session:
        confs = prediction_confidences or {}
        dart_colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]
        for i, (ax, ay, seg_name) in enumerate(session.annotations):
            color_d = dart_colors[i % len(dart_colors)]
            info = parse_class_name(seg_name)
            conf_str = f" ({confs[seg_name]:.0%})" if seg_name in confs else ""
            cv2.putText(
                cp,
                f"  dart {i + 1}: {info['label']}{conf_str}",
                (10, cp_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color_d,
                2,
            )
            cp_y += 22
        cp_y += 10
        if edit_dart is not None and edit_dart >= 1:
            cv2.putText(cp, f"Editing dart {edit_dart}: {edit_text}_", (10, cp_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
            cp_y += 22
        elif edit_dart == 0:
            cv2.putText(cp, "Press 1, 2, or 3 to edit  |  ESC=cancel", (10, cp_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
            cp_y += 22
        cv2.putText(
            cp,
            "Pull darts to continue  |  X=edit  |  R=discard",
            (10, cp_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (150, 150, 150),
            1,
        )

    # Annotation info
    elif state == UIState.ANNOTATING and session:
        cv2.putText(
            cp,
            f"Annotations: {len(session.annotations)}",
            (10, cp_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cp_y += 25
        if session.click_point is not None:
            if (
                session.guess_segment
                and session.text_input == session.guess_segment.lower()
            ):
                cv2.putText(
                    cp,
                    f"Guess: [{session.guess_segment}]",
                    (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cp_y += 20
                cv2.putText(
                    cp,
                    "ENTER=accept  or type correction",
                    (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (150, 150, 150),
                    1,
                )
            else:
                cv2.putText(
                    cp,
                    f"Input: {session.text_input}_",
                    (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                )
                cp_y += 20
                cv2.putText(
                    cp,
                    "Type segment + ENTER",
                    (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (150, 150, 150),
                    1,
                )
        else:
            cv2.putText(
                cp,
                "Click new dart tip",
                (10, cp_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150, 150, 150),
                1,
            )
    cp_y += 25

    # Controls
    cv2.putText(
        cp,
        "SPACE=capture  R=reset  P=pause  +/-=threshold  Q=quit",
        (10, cp_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (150, 150, 150),
        1,
    )

    return cp


def render_camera_hud(
    display,
    state,
    dart_ordinal,
    frame_counter,
    n_annotations=0,
    batch_count=0,
    batch_index=0,
    settle_remaining=0,
    collect_remaining=0,
):
    h, w = display.shape[:2]
    labels = {
        UIState.WARMUP: (f"WARMING UP...", (100, 100, 100)),
        UIState.LISTENING: (f"LIVE  |  Saved: {frame_counter}", (0, 255, 0)),
        UIState.SETTLING: (
            f"DART! Settling {settle_remaining:.1f}s  |  Captured: {batch_count}/3",
            (0, 200, 255),
        ),
        UIState.COLLECTING: (
            f"COLLECTING  |  Captured: {batch_count}/3  ({collect_remaining:.0f}s)",
            (0, 200, 255),
        ),
        UIState.ANNOTATING: (
            f"ANNOTATE  |  Frame {batch_index + 1}/{batch_count}  |  Dart: {dart_ordinal}/3  |  Labels: {n_annotations}",
            (0, 0, 255),
        ),
        UIState.REVIEW: (f"LABELED  |  Pull darts or X to edit", (0, 255, 0)),
        UIState.PULL_DARTS: (f"PULL DARTS", (0, 140, 255)),
        UIState.PAUSED: (f"PAUSED  |  Saved: {frame_counter}", (128, 128, 128)),
    }
    text, color = labels.get(state, ("", (255, 255, 255)))
    cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(
        display,
        f"Saved: {frame_counter}",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
    )


def render_annotations(display, session, box_size, prediction_confidences=None):
    if session is None:
        return
    confs = prediction_confidences or {}
    dart_colors = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]
    half = box_size // 2
    for i, (ax, ay, seg_name) in enumerate(session.annotations):
        color = dart_colors[i % len(dart_colors)]
        cv2.rectangle(display, (ax - half, ay - half), (ax + half, ay + half), color, 2)
        cv2.circle(display, (ax, ay), 3, color, -1)
        conf_str = f" {confs[seg_name]:.0%}" if seg_name in confs else ""
        label = f"dart {i + 1} {seg_name}{conf_str}"
        cv2.putText(
            display,
            label,
            (ax + half + 4, ay + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    if session.click_point is not None:
        px, py = session.click_point
        cv2.circle(display, (px, py), 7, (255, 0, 255), 2)
        cv2.circle(display, (px, py), 2, (255, 0, 255), -1)
        dart_num = len(session.annotations) + 1
        if (
            session.guess_segment
            and session.text_input == session.guess_segment.lower()
        ):
            text = f"Dart {dart_num} > [{session.guess_segment}]  ENTER=accept"
            cv2.putText(
                display,
                text,
                (px + 12, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
        else:
            text = f"Dart {dart_num} > {session.text_input}_"
            cv2.putText(
                display,
                text,
                (px + 12, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )
    h = display.shape[0]
    if session.click_point is not None:
        hint = "ENTER=confirm  ESC=cancel click"
        cv2.putText(
            display,
            hint,
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
    else:
        cv2.putText(
            display,
            "Click=tip  ENTER=save  ESC=discard  Z=undo",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )


# ---------------------------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------------------------

_session_ref = [None]
_auto_confirmed = [False]  # signal that a click auto-confirmed


def _predict_annotations(yolo_model, batch_frames, homography=None, crop_offset=(0, 0), conf=0.25):
    """Run 1-class YOLO on batch frames and build annotation lists with carry-forward.

    Detects dart positions (class 0), classifies each tip via geometry.
    Annotations are (x, y, segment_name) tuples (same as manual).
    Also returns a confidence dict keyed by segment_name for display.
    Returns (annotations_list_per_frame, confidence_dict) or (None, None).
    """
    if yolo_model is None or not batch_frames:
        return None, None

    all_frame_annotations = []
    previous = []
    confidences = {}  # seg_name -> confidence

    for fi, frame in enumerate(batch_frames):
        results = yolo_model.predict(frame, conf=conf, verbose=False)
        if not results or len(results) == 0 or results[0].boxes is None:
            return None, None

        boxes = results[0].boxes
        frame_annotations = list(previous)

        for box in sorted(boxes, key=lambda b: float(b.conf[0]), reverse=True):
            cls_conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Skip if close to existing annotation
            is_duplicate = False
            for (ax, ay, _) in frame_annotations:
                if abs(cx - ax) < 30 and abs(cy - ay) < 30:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            # Geometry classify the tip position
            if homography is not None:
                seg_name = geometry_classify(cx, cy, homography, crop_offset)
            else:
                seg_name = "MISS"  # fallback if no homography

            if seg_name is None or seg_name not in CLASS_TO_ID:
                continue

            frame_annotations.append((cx, cy, seg_name))
            confidences[seg_name] = cls_conf
            break

        if len(frame_annotations) <= len(previous):
            return None, None

        all_frame_annotations.append(frame_annotations)
        previous = list(frame_annotations)

    return all_frame_annotations, confidences


def _mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        session = _session_ref[0]
        if session is not None:
            was_auto = session.handle_click(x, y)
            if was_auto:
                _auto_confirmed[0] = True


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------


def collect_data(
    outdir="data/training_v2",
    use_undistort=True,
    box_size=30,
    trigger_mode="audio",
    audio_device=None,
    audio_threshold=0.7,
    video_threshold=5.0,
    settle_delay=0.7,
    collect_timeout=10.0,
):
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

    from calibrate import (
        open_camera,
        load_lens_params,
        undistort_frame,
        load_crop_roi,
        apply_crop,
    )

    cap = open_camera()

    lens_params = None
    if use_undistort and config.LENS_PARAMS_PATH.exists():
        lens_params = load_lens_params()
        print("Lens undistortion enabled")
    elif use_undistort:
        print("WARNING: No lens params found, running without undistortion")

    crop_roi = load_crop_roi()
    crop_offset = (0, 0)  # homography calibrated in cropped space, no offset needed
    if crop_roi is not None:
        x, y, w, h = crop_roi
        print(f"Crop ROI: ({x}, {y}) {w}x{h}")

    homography = None
    board_center = None  # board center in crop-pixel space for bbox expansion
    if config.BOARD_HOMOGRAPHY_PATH.exists():
        hom_data = np.load(str(config.BOARD_HOMOGRAPHY_PATH))
        homography = hom_data["homography"]
        print("Board homography loaded — segment guess enabled")
        # Compute board center in crop-pixel space (inverse homography from canonical center)
        try:
            H_inv = np.linalg.inv(homography)
            cx_can, cy_can = config.CANONICAL_CENTER
            pts = np.array([[[float(cx_can), float(cy_can)]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pts, H_inv)
            bc_full_x = float(transformed[0][0][0])
            bc_full_y = float(transformed[0][0][1])
            # Subtract crop offset to get crop-space coordinates
            if crop_roi is not None:
                cx_off, cy_off, _, _ = crop_roi
                board_center = (bc_full_x - cx_off, bc_full_y - cy_off)
            else:
                board_center = (bc_full_x, bc_full_y)
        except Exception as e:
            print(f"WARNING: Could not compute board center: {e}")

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

    # Load YOLO model for prediction-assisted labeling (optional)
    yolo_model = None
    model_weights = config.PROJECT_ROOT / "runs" / "detect" / "dartscorer" / "weights" / "best.pt"
    if model_weights.exists():
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(str(model_weights))
            print(f"YOLO model loaded for assisted labeling")
        except Exception as e:
            print(f"WARNING: Could not load YOLO model: {e}")

    win = "Collect Training Data"
    panel_win = "Control Panel"
    create_window(win, default_width=960, default_height=540)
    create_window(panel_win, default_width=400, default_height=380)
    cv2.setMouseCallback(win, _mouse_callback)

    print(f"\n=== YOLO Training Data Collection (v2, 1-class detection) ===")
    print(f"Output: {outdir.resolve()}")
    print(f"Continuing from frame {frame_counter}")
    print(f"Batch mode: throw up to 3 darts, then annotate all")
    print(f"Collect timeout: {collect_timeout}s after first dart")
    if yolo_model:
        print(f"Model-assisted labeling: ON")
    print()

    # --- State ---
    state = UIState.WARMUP if video else UIState.LISTENING
    paused_from = None  # state to resume to when unpausing
    session = None
    _session_ref[0] = None
    previous_annotations = []

    # Batch: list of captured frames during a round
    batch_frames = []  # list of frame arrays
    batch_index = 0  # which frame we're annotating

    # Timing
    settle_start = 0.0
    collect_start = 0.0  # when first dart of batch was captured
    cooldown_start = 0.0
    edit_dart = None  # REVIEW edit state: None=not editing, 0=picking dart#, 1-3=typing
    edit_text = ""
    prediction_confidences = {}  # cls_name -> confidence for display
    review_enter_time = 0.0  # when REVIEW state was entered

    try:
        while True:
            ret, raw = cap.read()
            if not ret:
                print("Camera read failed")
                break

            frame = raw
            if lens_params is not None:
                frame = undistort_frame(raw, *lens_params)
            frame = apply_crop(frame, crop_roi)

            if video:
                video.update(frame)

            settle_remaining = 0
            collect_remaining = 0

            # --- State transitions ---
            if state == UIState.WARMUP:
                if video and video.warmup_remaining <= 0:
                    state = UIState.LISTENING
                    print("Warmup complete — listening")
                elif not video:
                    state = UIState.LISTENING

            elif state == UIState.LISTENING:
                triggered = False
                if audio and audio.check_and_reset():
                    triggered = True
                elif video and video.check_trigger():
                    triggered = True

                if triggered:
                    settle_start = time.monotonic()
                    state = UIState.SETTLING
                    print(
                        f"\n  Dart {len(batch_frames) + 1} detected — settling {settle_delay}s..."
                    )

            elif state == UIState.SETTLING:
                elapsed = time.monotonic() - settle_start
                settle_remaining = max(0, settle_delay - elapsed)
                if elapsed >= settle_delay:
                    # Capture this settled frame
                    batch_frames.append(frame.copy())
                    n = len(batch_frames)
                    print(f"  Captured frame {n}/3")

                    # Absorb so next dart is detected against current state
                    if video:
                        video.absorb(frame)

                    if n == 1:
                        collect_start = time.monotonic()

                    if n >= 3:
                        # All 3 darts captured — try model prediction first
                        predicted, pred_confs = _predict_annotations(
                            yolo_model, batch_frames, homography, crop_offset)
                        if predicted:
                            prediction_confidences = pred_confs or {}
                            # Model predicted all darts — save and go to REVIEW
                            for fi, anns in enumerate(predicted):
                                h_img, w_img = batch_frames[fi].shape[:2]
                                fname = f"frame_{frame_counter:05d}.png"
                                cv2.imwrite(str(img_dir / fname), batch_frames[fi])
                                label_name = f"frame_{frame_counter:05d}.txt"
                                with open(label_dir / label_name, "w") as lf:
                                    for (ax, ay, seg) in anns:
                                        bcx, bcy, bw, bh = auto_expand_bbox(ax, ay, w_img, h_img, board_center)
                                        lf.write(f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f}\n")
                                entry = {"filename": fname,
                                         "darts": [{"tip_x": ax, "tip_y": ay, "segment": seg} for (ax, ay, seg) in anns],
                                         "n_darts": len(anns), "timestamp": time.time()}
                                with open(annotations_path, "a") as f:
                                    f.write(json.dumps(entry) + "\n")
                                frame_counter += 1
                                print(f"  Auto-saved: {fname} with {len(anns)} dart(s)")
                            previous_annotations = list(predicted[-1])
                            session = AnnotationSession(homography=homography,
                                                        previous_annotations=previous_annotations,
                                                        crop_offset=crop_offset)
                            _session_ref[0] = session
                            state = UIState.REVIEW
                            review_enter_time = time.monotonic()
                            for i, (_, _, seg) in enumerate(predicted[-1]):
                                info = parse_class_name(seg)
                                print(f"    dart {i+1}: {info['label']}")
                            print("  Model predicted — pull darts or X to edit")
                        else:
                            # Fall back to manual
                            state = UIState.ANNOTATING
                            batch_index = 0
                            previous_annotations = []
                            session = AnnotationSession(homography=homography,
                                                        previous_annotations=[],
                                                        crop_offset=crop_offset)
                            _session_ref[0] = session
                            print(f"  3 darts captured — click tips to annotate")
                    else:
                        state = UIState.COLLECTING

            elif state == UIState.COLLECTING:
                collect_remaining = max(
                    0, collect_timeout - (time.monotonic() - collect_start)
                )

                # Check for another dart
                triggered = False
                if audio and audio.check_and_reset():
                    triggered = True
                elif video and video.check_trigger():
                    triggered = True

                if triggered:
                    settle_start = time.monotonic()
                    state = UIState.SETTLING
                    print(
                        f"\n  Dart {len(batch_frames) + 1} detected — settling {settle_delay}s..."
                    )

                elif collect_remaining <= 0:
                    # Timeout — try model prediction first
                    predicted, pred_confs = _predict_annotations(
                        yolo_model, batch_frames, homography, crop_offset)
                    if predicted:
                        prediction_confidences = pred_confs or {}
                        for fi, anns in enumerate(predicted):
                            h_img, w_img = batch_frames[fi].shape[:2]
                            fname = f"frame_{frame_counter:05d}.png"
                            cv2.imwrite(str(img_dir / fname), batch_frames[fi])
                            label_name = f"frame_{frame_counter:05d}.txt"
                            with open(label_dir / label_name, "w") as lf:
                                for (ax, ay, seg) in anns:
                                    bcx, bcy, bw, bh = auto_expand_bbox(ax, ay, w_img, h_img, board_center)
                                    lf.write(f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f}\n")
                            entry = {"filename": fname,
                                     "darts": [{"tip_x": ax, "tip_y": ay, "segment": seg} for (ax, ay, seg) in anns],
                                     "n_darts": len(anns), "timestamp": time.time()}
                            with open(annotations_path, "a") as f:
                                f.write(json.dumps(entry) + "\n")
                            frame_counter += 1
                        previous_annotations = list(predicted[-1])
                        session = AnnotationSession(homography=homography,
                                                    previous_annotations=previous_annotations,
                                                    crop_offset=crop_offset)
                        _session_ref[0] = session
                        state = UIState.REVIEW
                        review_enter_time = time.monotonic()
                        print(f"  Timeout — model predicted {len(batch_frames)} frame(s), pull darts or X to edit")
                    else:
                        state = UIState.ANNOTATING
                        batch_index = 0
                        previous_annotations = []
                        session = AnnotationSession(homography=homography,
                                                    previous_annotations=[],
                                                    crop_offset=crop_offset)
                        _session_ref[0] = session
                        print(f"  Timeout — {len(batch_frames)} frame(s), click tips to annotate")

            elif state == UIState.PULL_DARTS:
                if video:
                    # Phase 1: wait for the hand/darts disturbance
                    if not video._saw_pull_disturbance:
                        if video.saw_disturbance():
                            video._saw_pull_disturbance = True
                            video._suppress_calm_count = 0
                            print("  Darts being pulled...")
                    else:
                        # Phase 2: wait for board to stabilize after pull
                        if video.is_calm():
                            # Save background frame (empty board)
                            bg_fname = f"frame_{frame_counter:05d}.png"
                            cv2.imwrite(str(img_dir / bg_fname), frame)
                            with open(label_dir / f"frame_{frame_counter:05d}.txt", "w") as lf:
                                pass  # empty label = background
                            frame_counter += 1
                            print(f"  Background saved: {bg_fname}")
                            state = UIState.LISTENING
                            print("  Board stable — listening")
                else:
                    if time.monotonic() - cooldown_start >= 3.0:
                        state = UIState.LISTENING

            # --- Render ---
            if state in (UIState.ANNOTATING, UIState.REVIEW) and batch_frames:
                # Show last frame (all darts visible) during annotating/review
                show_idx = min(batch_index, len(batch_frames) - 1)
                display = batch_frames[show_idx].copy()
            else:
                display = frame.copy()

            n_ann = len(session.annotations) if session else 0
            dart_ordinal = session.dart_ordinal if session else 1
            render_camera_hud(
                display,
                state,
                dart_ordinal,
                frame_counter,
                n_annotations=n_ann,
                batch_count=len(batch_frames),
                batch_index=batch_index,
                settle_remaining=settle_remaining,
                collect_remaining=collect_remaining,
            )

            if state in (UIState.ANNOTATING, UIState.REVIEW) and session:
                render_annotations(display, session, box_size, prediction_confidences)

            cv2.imshow(win, display)

            cp = render_control_panel(
                state,
                video,
                audio,
                session,
                frame_counter,
                batch_frames,
                batch_index,
                dart_ordinal,
                settle_remaining,
                collect_remaining,
                edit_dart,
                edit_text,
                prediction_confidences,
            )
            cv2.imshow(panel_win, cp)

            # --- Key handling ---
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break

            elif key == ord(" ") and state in (
                UIState.LISTENING,
                UIState.COLLECTING,
                UIState.WARMUP,
                UIState.PULL_DARTS,
            ):
                # Manual capture — add frame to batch
                batch_frames.append(frame.copy())
                n = len(batch_frames)
                print(f"  [MANUAL] Captured frame {n}/3")
                if video:
                    video.absorb(frame)
                if n == 1:
                    collect_start = time.monotonic()
                if n >= 3:
                    state = UIState.ANNOTATING
                    batch_index = 0
                    previous_annotations = []
                    session = AnnotationSession(
                        homography=homography, previous_annotations=[], crop_offset=crop_offset
                    )
                    _session_ref[0] = session
                    print(f"  3 frames captured — annotate frame 1")
                else:
                    state = UIState.COLLECTING

            elif key == ord("r") and state != UIState.ANNOTATING:
                previous_annotations = []
                batch_frames = []
                batch_index = 0
                if video:
                    video.reset_calm_counter()
                    state = UIState.PULL_DARTS
                    cooldown_start = time.monotonic()
                    print("Round reset — pull darts")
                else:
                    state = UIState.LISTENING
                    print("Round reset — dart 1")

            elif key == ord("n") and state == UIState.COLLECTING:
                # Skip waiting, go annotate now
                state = UIState.ANNOTATING
                batch_index = 0
                previous_annotations = []
                session = AnnotationSession(
                    homography=homography, previous_annotations=[], crop_offset=crop_offset
                )
                _session_ref[0] = session
                print(f"  Skipped — annotate {len(batch_frames)} frame(s)")

            elif key == ord("p") and state != UIState.ANNOTATING:
                if state == UIState.PAUSED:
                    # Resume
                    state = paused_from if paused_from else UIState.LISTENING
                    paused_from = None
                    print("Resumed")
                else:
                    # Pause
                    paused_from = state
                    state = UIState.PAUSED
                    print("Paused — press P to resume")

            elif key == ord("+") or key == ord("="):
                if audio:
                    audio.prob_threshold = min(audio.prob_threshold + 0.05, 0.99)
                    print(f"  Dart prob threshold: {audio.prob_threshold:.2f}")
                elif video:
                    video.threshold = min(video.threshold + 2, 200)
                    print(f"  Video cell threshold: {video.threshold}")

            elif key == ord("-"):
                if audio:
                    audio.prob_threshold = max(audio.prob_threshold - 0.05, 0.1)
                    print(f"  Dart prob threshold: {audio.prob_threshold:.2f}")
                elif video:
                    video.threshold = max(video.threshold - 2, 1)
                    print(f"  Video cell threshold: {video.threshold}")

            elif state == UIState.ANNOTATING and session:
                # Check if a click auto-confirmed
                if _auto_confirmed[0]:
                    _auto_confirmed[0] = False
                    print(f"  {session.last_auto_msg}")

                    # Save current frame immediately
                    h_img, w_img = batch_frames[batch_index].shape[:2]
                    fname = f"frame_{frame_counter:05d}.png"
                    cv2.imwrite(str(img_dir / fname), batch_frames[batch_index])
                    label_name = f"frame_{frame_counter:05d}.txt"
                    with open(label_dir / label_name, "w") as lf:
                        for ax, ay, seg in session.annotations:
                            bcx, bcy, bw, bh = auto_expand_bbox(ax, ay, w_img, h_img, board_center)
                            lf.write(f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f}\n")
                    entry = {
                        "filename": fname,
                        "darts": [
                            {"tip_x": ax, "tip_y": ay, "segment": seg}
                            for (ax, ay, seg) in session.annotations
                        ],
                        "n_darts": len(session.annotations),
                        "timestamp": time.time(),
                    }
                    with open(annotations_path, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                    frame_counter += 1
                    print(f"  Saved: {fname} with {len(session.annotations)} dart(s)")

                    # Carry forward and advance
                    previous_annotations = list(session.annotations)
                    next_dart_num = len(previous_annotations) + 1
                    batch_index += 1

                    if batch_index < len(batch_frames):
                        # Next frame — carry forward, show it
                        session = AnnotationSession(
                            homography=homography,
                            previous_annotations=previous_annotations,
                            crop_offset=crop_offset,
                        )
                        _session_ref[0] = session
                        print(
                            f"  → Frame {batch_index + 1}/{len(batch_frames)} — click dart {next_dart_num} tip"
                        )
                    else:
                        # All frames done — go to REVIEW
                        state = UIState.REVIEW
                        review_enter_time = time.monotonic()
                        print("  All labeled — pull darts or X to edit")

                if state == UIState.ANNOTATING and session:
                    # Manual input for clicks without auto-guess
                    if session.click_point is not None:
                        if key == 13:  # ENTER — confirm typed segment
                            ok, msg = session.confirm_segment()
                            print(f"  {msg}")
                            if not ok:
                                print("  Try again (e.g. t20, s5, dbull)")
                        elif key == 27:
                            session.cancel_click()
                        else:
                            session.handle_key(key)
                    else:
                        if key == ord("z"):
                            removed = session.undo()
                            if removed:
                                print(f"  Undo: {removed[2]}")
                        elif key == ord("r"):
                            previous_annotations = []
                            batch_frames = []
                            batch_index = 0
                            session = None
                            _session_ref[0] = None
                            if video:
                                video.reset_calm_counter()
                                state = UIState.PULL_DARTS
                                cooldown_start = time.monotonic()
                            else:
                                state = UIState.LISTENING
                            print("  Round reset — pull darts")
                        elif key == 27:  # ESC — discard batch
                            batch_frames = []
                            batch_index = 0
                            session = None
                            _session_ref[0] = None
                            state = UIState.LISTENING
                            print("  Discarded — listening")

            elif state == UIState.REVIEW:
                # Show last frame with all annotations
                # edit_dart: None=not editing, 0=waiting for dart#, 1-3=typing new label
                if key == ord("x") and edit_dart is None:
                    edit_dart = 0
                    edit_text = ""
                    print("  Edit: press 1, 2, or 3 to re-label that dart")

                elif edit_dart == 0:
                    # Waiting for dart number (1-based index into annotations list)
                    if key in (ord("1"), ord("2"), ord("3")):
                        dart_num = key - ord("0")
                        dart_idx = dart_num - 1  # 0-based index
                        if dart_idx < len(session.annotations):
                            edit_dart = dart_num
                            edit_text = ""
                            cur = session.annotations[dart_idx][2]  # segment name
                            print(f"  Editing dart {dart_num} ({cur}) — type new label + ENTER")
                        else:
                            print(f"  No dart {dart_num} in this round")
                            edit_dart = None
                    elif key == 27:
                        edit_dart = None
                        print("  Edit cancelled")

                elif edit_dart is not None and edit_dart >= 1:
                    # Typing new label for a specific dart (edit_dart is 1-based)
                    if key == 13:  # ENTER — confirm
                        seg = segment_shorthand(edit_text)
                        if seg is None:
                            print(f"  Invalid: '{edit_text}' — try again")
                            edit_text = ""
                        elif seg not in CLASS_TO_ID:
                            print(f"  Unknown class: {seg}")
                            edit_text = ""
                        else:
                            dart_idx = edit_dart - 1  # 0-based
                            ax, ay, _ = session.annotations[dart_idx]
                            session.annotations[dart_idx] = (ax, ay, seg)
                            info = parse_class_name(seg)
                            print(f"  dart {edit_dart} → {info['label']}")
                            # Re-save affected label files
                            # Each frame N has annotations [d0..dN-1]
                            for fi in range(len(batch_frames)):
                                frame_anns = session.annotations[:fi + 1]
                                h_img, w_img = batch_frames[fi].shape[:2]
                                saved_idx = frame_counter - len(batch_frames) + fi
                                label_name = f"frame_{saved_idx:05d}.txt"
                                # Only rewrite if this frame includes the edited dart
                                if fi + 1 >= edit_dart:
                                    with open(label_dir / label_name, "w") as lf:
                                        for (aax, aay, sseg) in frame_anns:
                                            bcx, bcy, bw, bh = auto_expand_bbox(aax, aay, w_img, h_img, board_center)
                                            lf.write(f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f}\n")
                                    print(f"  Updated: {label_name}")
                            # Update carry-forward
                            previous_annotations = list(session.annotations)
                            edit_dart = None
                            edit_text = ""
                    elif key == 27:
                        edit_dart = None
                        edit_text = ""
                        print("  Edit cancelled")
                    elif key == 8:  # BACKSPACE
                        if edit_text:
                            edit_text = edit_text[:-1]
                    elif 32 <= key < 127:
                        edit_text += chr(key)

                elif key == ord("r") or (edit_dart is None and video and video.saw_disturbance()
                                         and time.monotonic() - review_enter_time > 2.0):
                    # Accept and move on
                    edit_dart = None
                    edit_text = ""
                    prediction_confidences = {}
                    previous_annotations = []
                    batch_frames = []
                    batch_index = 0
                    session = None
                    _session_ref[0] = None
                    if video:
                        video.reset_calm_counter()
                        state = UIState.PULL_DARTS
                        cooldown_start = time.monotonic()
                        print("  Pull darts...")
                    else:
                        state = UIState.LISTENING
                        print("  Ready for next round")

    finally:
        save_window_sizes([win, panel_win])
        if audio:
            audio.stop()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone. {frame_counter} total frames saved to {outdir.resolve()}")


# ---------------------------------------------------------------------------
# Capture-only mode — just record frames, no labeling
# ---------------------------------------------------------------------------


def capture_only(
    outdir="data/training",
    use_undistort=True,
    trigger_mode="video",
    audio_device=None,
    audio_threshold=0.7,
    video_threshold=5.0,
    settle_delay=0.7,
):
    """Record frames on dart impacts. No annotation, no freezing.

    Just throw darts. Each impact auto-captures a settled frame.
    Press R between rounds. Images saved without labels for later annotation.
    """
    outdir = Path(outdir)
    img_dir = outdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_log = outdir / "unlabeled.jsonl"

    # Count existing images to continue numbering
    existing = len(list(img_dir.glob("*.png")))
    frame_counter = existing

    from calibrate import (
        open_camera,
        load_lens_params,
        undistort_frame,
        load_crop_roi,
        apply_crop,
    )

    cap = open_camera()

    lens_params = None
    if use_undistort and config.LENS_PARAMS_PATH.exists():
        lens_params = load_lens_params()

    crop_roi = load_crop_roi()
    crop_offset = (0, 0)  # homography calibrated in cropped space

    audio = None
    video = None
    if trigger_mode == "audio":
        audio = DartAudioTrigger(device=audio_device, prob_threshold=audio_threshold)
        if not audio.start():
            audio = None
    elif trigger_mode == "video":
        video = VideoTrigger(threshold=video_threshold)

    win = "Capture Mode"
    panel_win = "Control Panel"
    create_window(win, default_width=960, default_height=540)
    create_window(panel_win, default_width=400, default_height=250)

    print(f"\n=== CAPTURE-ONLY MODE ===")
    print(f"Just throw darts. Frames auto-saved on impact.")
    print(f"R=reset round  P=pause  SPACE=manual  Q=quit")
    print(f"Starting from frame {frame_counter}\n")

    state = UIState.WARMUP if video else UIState.LISTENING
    paused_from = None
    settle_start = 0.0
    cooldown_start = 0.0
    darts_this_round = 0

    try:
        while True:
            ret, raw = cap.read()
            if not ret:
                break

            frame = raw
            if lens_params is not None:
                frame = undistort_frame(raw, *lens_params)
            frame = apply_crop(frame, crop_roi)

            if video:
                video.update(frame)

            settle_remaining = 0

            # State machine
            if state == UIState.WARMUP:
                if (video and video.warmup_remaining <= 0) or not video:
                    state = UIState.LISTENING
                    print("Listening...")

            elif state == UIState.LISTENING:
                triggered = False
                if audio and audio.check_and_reset():
                    triggered = True
                elif video and video.check_trigger():
                    triggered = True
                if triggered:
                    settle_start = time.monotonic()
                    state = UIState.SETTLING
                    darts_this_round += 1

            elif state == UIState.SETTLING:
                elapsed = time.monotonic() - settle_start
                settle_remaining = max(0, settle_delay - elapsed)
                if elapsed >= settle_delay:
                    # Save frame
                    fname = f"frame_{frame_counter:05d}.png"
                    cv2.imwrite(str(img_dir / fname), frame)
                    entry = {
                        "filename": fname,
                        "dart_in_round": darts_this_round,
                        "timestamp": time.time(),
                    }
                    with open(unlabeled_log, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                    frame_counter += 1
                    print(f"  Captured: {fname} (dart {darts_this_round}/3)")

                    if video:
                        video.absorb(frame)

                    if darts_this_round >= 3:
                        state = UIState.LISTENING  # will wait for R to reset
                    else:
                        state = UIState.LISTENING

            elif state == UIState.PULL_DARTS:
                if video:
                    if not video._saw_pull_disturbance:
                        if video.saw_disturbance():
                            video._saw_pull_disturbance = True
                            video._suppress_calm_count = 0
                    else:
                        if video.is_calm():
                            state = UIState.LISTENING
                            print("  Board stable — listening")
                else:
                    if time.monotonic() - cooldown_start >= 3.0:
                        state = UIState.LISTENING

            # Render
            display = frame.copy()
            h, w = display.shape[:2]

            # Camera HUD
            if state == UIState.WARMUP:
                text = f"WARMING UP... {video.warmup_remaining:.1f}s"
                color = (100, 100, 100)
            elif state == UIState.SETTLING:
                text = f"DART! Settling {settle_remaining:.1f}s"
                color = (0, 200, 255)
            elif state == UIState.PULL_DARTS:
                text = "PULL DARTS"
                color = (0, 140, 255)
            elif state == UIState.PAUSED:
                text = "PAUSED"
                color = (128, 128, 128)
            else:
                text = f"LIVE  |  Saved: {frame_counter}  |  Round dart: {darts_this_round}/3"
                color = (0, 255, 0)
            cv2.putText(
                display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            cv2.imshow(win, display)

            # Control panel
            cp_w, cp_h = 400, 250
            cp = np.zeros((cp_h, cp_w, 3), dtype=np.uint8)
            cp[:] = (30, 30, 30)
            cp_y = 20
            cv2.putText(
                cp,
                "CAPTURE-ONLY MODE",
                (10, cp_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )
            cp_y += 30
            if video:
                wave_x, wave_w, wave_h = 10, cp_w - 20, 70
                _draw_waveform(
                    cp,
                    wave_x,
                    cp_y,
                    wave_w,
                    wave_h,
                    video._diff_history,
                    video.threshold,
                    video._history_max,
                )
                cp_y += wave_h + 15
                cv2.putText(
                    cp,
                    f"cells={video.current_diff:.0f}  thresh={video.threshold}",
                    (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (200, 200, 200),
                    1,
                )
                cp_y += 22
            cv2.putText(
                cp,
                f"Saved: {frame_counter}  |  Round: {darts_this_round}/3",
                (10, cp_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )
            cp_y += 30
            cv2.putText(
                cp,
                "R=reset  P=pause  SPACE=manual  +/-=thresh  Q=quit",
                (10, cp_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (150, 150, 150),
                1,
            )
            cv2.imshow(panel_win, cp)

            # Keys
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                darts_this_round = 0
                if video:
                    video.reset_calm_counter()
                    state = UIState.PULL_DARTS
                    cooldown_start = time.monotonic()
                    print("Round reset — pull darts")
                else:
                    print("Round reset")
            elif key == ord(" ") and state in (UIState.LISTENING, UIState.PAUSED):
                fname = f"frame_{frame_counter:05d}.png"
                cv2.imwrite(str(img_dir / fname), frame)
                darts_this_round += 1
                entry = {
                    "filename": fname,
                    "dart_in_round": darts_this_round,
                    "timestamp": time.time(),
                }
                with open(unlabeled_log, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                frame_counter += 1
                if video:
                    video.absorb(frame)
                print(f"  [MANUAL] {fname} (dart {darts_this_round}/3)")
            elif key == ord("p"):
                if state == UIState.PAUSED:
                    state = paused_from or UIState.LISTENING
                    paused_from = None
                    print("Resumed")
                else:
                    paused_from = state
                    state = UIState.PAUSED
                    print("Paused")
            elif key == ord("+") or key == ord("="):
                if video:
                    video.threshold = min(video.threshold + 2, 200)
                    print(f"  Threshold: {video.threshold}")
                elif audio:
                    audio.prob_threshold = min(audio.prob_threshold + 0.05, 0.99)
            elif key == ord("-"):
                if video:
                    video.threshold = max(video.threshold - 2, 1)
                    print(f"  Threshold: {video.threshold}")
                elif audio:
                    audio.prob_threshold = max(audio.prob_threshold - 0.05, 0.1)

    finally:
        save_window_sizes([win, panel_win])
        if audio:
            audio.stop()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone. {frame_counter} frames saved to {img_dir}")


# ---------------------------------------------------------------------------
# Label mode — annotate unlabeled frames offline
# ---------------------------------------------------------------------------


def label_offline(outdir="data/training", box_size=30):
    """Browse and label previously captured frames.

    Shows each unlabeled image, lets you click dart tips and type labels.
    Skips frames that already have label files.
    """
    outdir = Path(outdir)
    img_dir = outdir / "images"
    label_dir = outdir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = outdir / "annotations.jsonl"

    # Find unlabeled images (have .png but no matching .txt in labels/)
    all_images = sorted(img_dir.glob("*.png"))
    unlabeled = [p for p in all_images if not (label_dir / (p.stem + ".txt")).exists()]

    if not unlabeled:
        print("No unlabeled images found.")
        return

    print(f"\n=== LABEL MODE ===")
    print(f"Found {len(unlabeled)} unlabeled frames out of {len(all_images)} total")
    print(
        f"Controls: Click=tip  type label+ENTER  ENTER=save  ESC=skip  Z=undo  Q=quit\n"
    )

    # Load homography for guessing
    homography = None
    if config.BOARD_HOMOGRAPHY_PATH.exists():
        hom_data = np.load(str(config.BOARD_HOMOGRAPHY_PATH))
        homography = hom_data["homography"]
        print("Board homography loaded — segment guess enabled")

    crop_offset = (0, 0)  # homography calibrated in cropped space

    win = "Label Frames"
    panel_win = "Control Panel"
    create_window(win, default_width=960, default_height=540)
    create_window(panel_win, default_width=400, default_height=300)

    session = None
    _session_ref[0] = None
    cv2.setMouseCallback(win, _mouse_callback)

    img_index = 0
    previous_annotations = []
    labeled_count = 0

    # Group images into rounds by looking at unlabeled.jsonl dart_in_round
    # or just let the user manage annotations with R
    round_frame = 0  # which frame in the current round (0, 1, 2)

    while img_index < len(unlabeled):
        img_path = unlabeled[img_index]
        frame = cv2.imread(str(img_path))
        if frame is None:
            img_index += 1
            continue

        if session is None:
            session = AnnotationSession(
                homography=homography,
                previous_annotations=previous_annotations,
                crop_offset=crop_offset,
            )
            _session_ref[0] = session

        dart_ordinal = session.dart_ordinal

        # Render
        display = frame.copy()
        h, w = display.shape[:2]

        render_annotations(display, session, box_size)

        cv2.putText(
            display,
            f"{img_path.name}  |  {img_index + 1}/{len(unlabeled)}  |  Dart: {dart_ordinal}/3",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            display,
            f"Labeled: {labeled_count}  |  Remaining: {len(unlabeled) - img_index}",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
        )
        cv2.imshow(win, display)

        # Control panel
        cp_w, cp_h = 400, 300
        cp = np.zeros((cp_h, cp_w, 3), dtype=np.uint8)
        cp[:] = (30, 30, 30)
        cp_y = 20
        cv2.putText(
            cp, "LABEL MODE", (10, cp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        cp_y += 30
        cv2.putText(
            cp,
            f"File: {img_path.name}",
            (10, cp_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )
        cp_y += 22
        cv2.putText(
            cp,
            f"Frame {img_index + 1}/{len(unlabeled)}  |  Dart: {dart_ordinal}/3",
            (10, cp_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )
        cp_y += 22
        cv2.putText(
            cp,
            f"Annotations: {len(session.annotations)}",
            (10, cp_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cp_y += 30
        if session.click_point is not None:
            if (
                session.guess_segment
                and session.text_input == session.guess_segment.lower()
            ):
                cv2.putText(
                    cp,
                    f"Guess: [{session.guess_segment}]",
                    (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cp_y += 20
                cv2.putText(
                    cp,
                    "ENTER=accept  or type correction",
                    (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (150, 150, 150),
                    1,
                )
            else:
                cv2.putText(
                    cp,
                    f"Input: {session.text_input}_",
                    (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                )
                cp_y += 20
                cv2.putText(
                    cp,
                    "Type segment + ENTER",
                    (10, cp_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (150, 150, 150),
                    1,
                )
        else:
            cv2.putText(
                cp,
                "Click a dart tip",
                (10, cp_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150, 150, 150),
                1,
            )
        cp_y += 30
        cv2.putText(
            cp,
            "ENTER=save  ESC=skip  Z=undo  R=reset round  Q=quit",
            (10, cp_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (150, 150, 150),
            1,
        )
        cv2.imshow(panel_win, cp)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        elif session.click_point is not None:
            if key == 13:
                ok, msg = session.confirm_segment()
                print(f"  {msg}")
                if not ok:
                    print("  Try again")
            elif key == 27:
                session.cancel_click()
            else:
                session.handle_key(key)
        else:
            if key == 13:  # ENTER — save
                if not session.annotations:
                    print("  No annotations — click tips first, ESC to skip")
                    continue

                label_name = img_path.stem + ".txt"
                with open(label_dir / label_name, "w") as lf:
                    for ax, ay, seg in session.annotations:
                        bcx, bcy, bw, bh = auto_expand_bbox(ax, ay, w, h, board_center=None)
                        lf.write(f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f}\n")

                entry = {
                    "filename": img_path.name,
                    "darts": [
                        {"tip_x": ax, "tip_y": ay, "segment": seg}
                        for (ax, ay, seg) in session.annotations
                    ],
                    "n_darts": len(session.annotations),
                    "timestamp": time.time(),
                }
                with open(annotations_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

                labeled_count += 1
                print(f"  Saved: {label_name} with {len(session.annotations)} dart(s)")

                previous_annotations = list(session.annotations)
                session = None
                _session_ref[0] = None
                img_index += 1
                round_frame += 1

            elif key == 27:  # ESC — skip
                print(f"  Skipped {img_path.name}")
                session = None
                _session_ref[0] = None
                img_index += 1

            elif key == ord("z"):
                removed = session.undo()
                if removed:
                    print(f"  Undo: {removed[2]}")

            elif key == ord("r"):
                previous_annotations = []
                round_frame = 0
                session = None
                _session_ref[0] = None
                print("  Round reset — dart 1")

    save_window_sizes([win, panel_win])
    cv2.destroyAllWindows()
    print(f"\nDone. Labeled {labeled_count} frames.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect YOLO dart training data")
    sub = parser.add_subparsers(dest="mode", help="Mode")

    # Default: full batch collection (capture + annotate)
    p_collect = sub.add_parser("collect", help="Batch capture + annotate (default)")
    p_collect.add_argument("--outdir", default="data/training_v2")
    p_collect.add_argument("--no-undistort", action="store_true")
    p_collect.add_argument("--box-size", type=int, default=30)
    p_collect.add_argument(
        "--trigger", choices=["audio", "video", "manual"], default="audio"
    )
    p_collect.add_argument("--audio-device", type=int, default=None)
    p_collect.add_argument("--audio-threshold", type=float, default=0.7)
    p_collect.add_argument("--video-threshold", type=float, default=5.0)
    p_collect.add_argument("--settle-delay", type=float, default=0.7)
    p_collect.add_argument("--collect-timeout", type=float, default=10.0)

    # Capture-only: just record frames
    p_capture = sub.add_parser("capture", help="Capture frames only, no labeling")
    p_capture.add_argument("--outdir", default="data/training")
    p_capture.add_argument("--no-undistort", action="store_true")
    p_capture.add_argument(
        "--trigger", choices=["audio", "video", "manual"], default="video"
    )
    p_capture.add_argument("--audio-device", type=int, default=None)
    p_capture.add_argument("--audio-threshold", type=float, default=0.7)
    p_capture.add_argument("--video-threshold", type=float, default=5.0)
    p_capture.add_argument("--settle-delay", type=float, default=0.7)

    # Label: annotate unlabeled frames offline
    p_label = sub.add_parser("label", help="Label previously captured frames")
    p_label.add_argument("--outdir", default="data/training")
    p_label.add_argument("--box-size", type=int, default=30)

    args = parser.parse_args()

    if args.mode == "capture":
        capture_only(
            args.outdir,
            use_undistort=not args.no_undistort,
            trigger_mode=args.trigger,
            audio_device=args.audio_device,
            audio_threshold=args.audio_threshold,
            video_threshold=args.video_threshold,
            settle_delay=args.settle_delay,
        )
    elif args.mode == "label":
        label_offline(args.outdir, box_size=args.box_size)
    else:
        # Default to collect if no subcommand or "collect"
        if not hasattr(args, "outdir"):
            # No subcommand given — run collect with defaults
            collect_data()
        else:
            collect_data(
                args.outdir,
                use_undistort=not args.no_undistort,
                box_size=args.box_size,
                trigger_mode=args.trigger,
                audio_device=args.audio_device,
                audio_threshold=args.audio_threshold,
                video_threshold=args.video_threshold,
                settle_delay=args.settle_delay,
                collect_timeout=args.collect_timeout,
            )
