#!/usr/bin/env python3
"""
collect.py — Training data collection for YOLO dart detection + scoring.

Captures camera frames and lets the user annotate dart tips with their
board segment (e.g. T20, S5, D_BULL). Tracks dart ordinal (1st/2nd/3rd)
automatically within each round. Saves images and YOLO-format labels.

Audio trigger: listens to the camera mic for the "thud" of a dart hitting
the board and auto-freezes the frame for annotation. Falls back to manual
SPACE capture if --no-audio is set or the mic isn't available.

Usage:
    python collect.py                          # Start collecting (audio trigger on)
    python collect.py --no-audio               # Manual SPACE capture only
    python collect.py --audio-device 9         # Specific mic device index
    python collect.py --audio-threshold 0.15   # Adjust trigger sensitivity

Controls:
    (auto)  Frame freezes on dart impact sound
    SPACE   Manual capture/freeze (always available)
    Click   Mark a dart tip (while frozen)
    Then type segment shorthand (e.g. t20, s5, dbull) + ENTER to confirm
    BACKSPACE / Z   Undo last annotation
    ENTER   Save annotated frame (when no text input active)
    ESC     Discard current capture and resume live feed
    R       Reset round (back to dart 1)
    +/-     Adjust audio threshold up/down
    Q       Quit
"""

import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import config
from classes import (
    CLASS_TO_ID, SEGMENTS, make_class_name, segment_shorthand,
    parse_class_name,
)


# ---------------------------------------------------------------------------
# Audio trigger
# ---------------------------------------------------------------------------

class AudioTrigger:
    """Monitors a microphone for sudden amplitude spikes (dart impacts)."""

    def __init__(self, device=None, threshold=0.15, cooldown=1.0, samplerate=44100, blocksize=1024):
        """
        Args:
            device: Audio input device index. None = auto-detect eMeet C950.
            threshold: RMS amplitude threshold to trigger (0.0-1.0).
            cooldown: Seconds to wait after a trigger before allowing another.
            samplerate: Audio sample rate.
            blocksize: Samples per audio callback block.
        """
        self.threshold = threshold
        self.cooldown = cooldown
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.triggered = False
        self._last_trigger_time = 0.0
        self._stream = None
        self._current_rms = 0.0
        self._baseline_rms = 0.01  # running baseline noise level
        self._device = device

    def _find_emeet_device(self):
        """Auto-detect the eMeet C950 mic device index."""
        import sounddevice as sd
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0 and 'eMeet' in d.get('name', ''):
                return i
        return None

    def start(self):
        """Start listening. Returns True if mic was found and started."""
        try:
            import sounddevice as sd
        except ImportError:
            print("WARNING: sounddevice not installed — audio trigger disabled")
            return False

        device = self._device
        if device is None:
            device = self._find_emeet_device()
            if device is None:
                print("WARNING: eMeet C950 mic not found — audio trigger disabled")
                print("  Use --audio-device N to specify a device, or --no-audio to skip")
                return False

        device_info = sd.query_devices(device)
        print(f"Audio trigger: {device_info['name']} (device {device})")
        print(f"  Threshold: {self.threshold:.2f}, cooldown: {self.cooldown}s")

        def callback(indata, frames, time_info, status):
            rms = float(np.sqrt(np.mean(indata ** 2)))
            self._current_rms = rms

            # Update baseline with slow-moving average (ignore spikes)
            if rms < self._baseline_rms * 3:
                self._baseline_rms = 0.99 * self._baseline_rms + 0.01 * rms

            # Trigger if RMS exceeds threshold AND we're past cooldown
            now = time.monotonic()
            if (rms > self.threshold
                    and rms > self._baseline_rms * 5
                    and now - self._last_trigger_time > self.cooldown):
                self.triggered = True
                self._last_trigger_time = now

        try:
            self._stream = sd.InputStream(
                device=device,
                channels=1,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                callback=callback,
            )
            self._stream.start()
            return True
        except Exception as e:
            print(f"WARNING: Could not open audio device: {e}")
            return False

    def check_and_reset(self):
        """Check if a trigger occurred. Resets the flag."""
        if self.triggered:
            self.triggered = False
            return True
        return False

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def current_rms(self):
        return self._current_rms


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_click_point = None  # pending click waiting for label
_annotations = []    # list of (x, y, class_name) for current frame
_text_input = ""     # current text being typed for segment label
_active = False      # whether we're in annotation mode
_dart_ordinal = 1    # current dart number in round (1, 2, or 3)


def _mouse_callback(event, x, y, flags, param):
    global _click_point
    if _active and event == cv2.EVENT_LBUTTONDOWN:
        _click_point = (x, y)


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect_data(outdir="data/training", use_undistort=True, box_size=30,
                 use_audio=True, audio_device=None, audio_threshold=0.15):
    global _click_point, _annotations, _text_input, _active, _dart_ordinal

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
    from calibrate import open_camera, load_lens_params, undistort_frame
    cap = open_camera()

    lens_params = None
    if use_undistort and config.LENS_PARAMS_PATH.exists():
        lens_params = load_lens_params()
        print("Lens undistortion enabled")
    elif use_undistort:
        print("WARNING: No lens params found, running without undistortion")

    # Audio trigger setup
    audio = None
    if use_audio:
        audio = AudioTrigger(device=audio_device, threshold=audio_threshold)
        if not audio.start():
            audio = None

    win = "Collect Training Data"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _mouse_callback)

    print("\n=== YOLO Training Data Collection (186 classes) ===")
    print(f"Output: {outdir.resolve()}")
    print(f"Continuing from frame {frame_counter}")
    print(f"Box size: {box_size}x{box_size}px")
    if audio:
        print(f"Audio trigger: ACTIVE (threshold={audio.threshold:.2f})")
    else:
        print("Audio trigger: OFF (use SPACE to capture)")
    print()
    print("Controls:")
    if audio:
        print("  (auto)    Frame freezes on dart impact sound")
    print("  SPACE     Manual capture")
    print("  Click     Mark a dart tip (while frozen)")
    print("  Type      Segment shorthand (t20, s5, dbull) then ENTER")
    print("  BKSP/Z    Undo last annotation (when not typing)")
    print("  ENTER     Save frame (when all darts labeled)")
    print("  ESC       Discard and resume")
    print("  R         Reset round to dart 1")
    if audio:
        print("  +/-       Adjust audio threshold")
    print("  Q         Quit")
    print()

    frozen_frame = None
    _active = False
    _dart_ordinal = 1

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

                # Check audio trigger
                audio_fired = False
                if audio and audio.check_and_reset():
                    audio_fired = True

                display = frame.copy()
                h, w = display.shape[:2]

                # Audio level meter
                if audio:
                    rms = audio.current_rms
                    meter_w = int(min(rms / 0.5, 1.0) * 200)
                    meter_color = (0, 0, 255) if rms > audio.threshold else (0, 255, 0)
                    cv2.rectangle(display, (w - 220, 10), (w - 220 + meter_w, 25), meter_color, -1)
                    cv2.rectangle(display, (w - 220, 10), (w - 20, 25), (100, 100, 100), 1)
                    # Threshold marker
                    thresh_x = w - 220 + int(min(audio.threshold / 0.5, 1.0) * 200)
                    cv2.line(display, (thresh_x, 8), (thresh_x, 27), (0, 255, 255), 2)

                cv2.putText(display, f"LIVE  |  Saved: {frame_counter}  |  Dart: {_dart_ordinal}/3",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                trigger_hint = "AUTO+SPACE=capture" if audio else "SPACE=capture"
                cv2.putText(display, f"{trigger_hint}  R=reset round  Q=quit",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.imshow(win, display)

                should_capture = audio_fired

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    should_capture = True
                elif key == ord('r'):
                    _dart_ordinal = 1
                    print("Round reset — dart 1")
                elif key == ord('+') or key == ord('='):
                    if audio:
                        audio.threshold = min(audio.threshold + 0.02, 1.0)
                        print(f"  Audio threshold: {audio.threshold:.2f}")
                elif key == ord('-'):
                    if audio:
                        audio.threshold = max(audio.threshold - 0.02, 0.01)
                        print(f"  Audio threshold: {audio.threshold:.2f}")

                if should_capture:
                    frozen_frame = frame.copy()
                    _annotations = []
                    _click_point = None
                    _text_input = ""
                    _active = True
                    trigger_src = "AUDIO" if audio_fired else "MANUAL"
                    print(f"\n[{trigger_src}] Frame captured — annotating dart {_dart_ordinal}")
                    print(f"  Click tip, type segment (e.g. t20, s5, dbull), press ENTER to confirm")

            else:
                # Frozen: annotating
                display = frozen_frame.copy()
                h, w = display.shape[:2]

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

                # Draw pending click point
                if _click_point is not None:
                    px, py = _click_point
                    cv2.circle(display, (px, py), 7, (255, 0, 255), 2)
                    cv2.circle(display, (px, py), 2, (255, 0, 255), -1)
                    input_text = f"Dart {_dart_ordinal} > {_text_input}_"
                    cv2.putText(display, input_text, (px + 12, py + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                n = len(_annotations)
                status = f"FROZEN  |  Dart: {_dart_ordinal}/3  |  Annotated: {n}"
                cv2.putText(display, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if _click_point is not None:
                    cv2.putText(display, f"Type segment (e.g. t20, s5, dbull) then ENTER",
                                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
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

                                # Auto-advance ordinal
                                if _dart_ordinal < 3:
                                    _dart_ordinal += 1

                    elif key == 27:  # ESC — cancel this click
                        _click_point = None
                        _text_input = ""
                        print("  Click cancelled")
                    elif key == 8:  # BACKSPACE
                        if _text_input:
                            _text_input = _text_input[:-1]
                    elif 32 <= key < 127:  # printable character
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

                        frozen_frame = None
                        _active = False
                        _annotations = []

                    elif key == 27:  # ESC — discard
                        print("  Discarded")
                        frozen_frame = None
                        _active = False
                        _annotations = []
                        _click_point = None
                        _text_input = ""

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
                        print("  Round reset — dart 1")

    finally:
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
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable audio trigger, use SPACE only")
    parser.add_argument("--audio-device", type=int, default=None,
                        help="Audio input device index (default: auto-detect eMeet)")
    parser.add_argument("--audio-threshold", type=float, default=0.15,
                        help="Audio RMS trigger threshold (default: 0.15)")
    args = parser.parse_args()

    collect_data(
        args.outdir,
        use_undistort=not args.no_undistort,
        box_size=args.box_size,
        use_audio=not args.no_audio,
        audio_device=args.audio_device,
        audio_threshold=args.audio_threshold,
    )
