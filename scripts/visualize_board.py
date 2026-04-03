#!/usr/bin/env python3
"""
visualize_board.py — Diagnostic visualizer for dartboard geometry.

Shows the ring boundaries the program uses for scoring, with labels
for each radius in mm. Lets you compare standard vs parallax-corrected
radii side-by-side, and interactively adjust the miss cutoff.

Controls:
    p          Toggle parallax correction on/off
    +/-        Adjust miss cutoff (DOUBLE_OUTER_RADIUS) by 1mm
    Shift+/-   Adjust by 5mm
    r          Reset to standard radii
    s          Print current radii to console
    q/ESC      Quit

Usage:
    uv run python scripts/visualize_board.py
    uv run python scripts/visualize_board.py --camera   # Overlay on live camera feed
"""

import argparse
import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

from dartscorer import config
from dartscorer import board


# Working copies of radii — start from standard
_radii = {
    "inner_bull": config.INNER_BULL_RADIUS_STANDARD,
    "outer_bull": config.OUTER_BULL_RADIUS_STANDARD,
    "triple_inner": config.TRIPLE_INNER_RADIUS_STANDARD,
    "triple_outer": config.TRIPLE_OUTER_RADIUS_STANDARD,
    "double_inner": config.DOUBLE_INNER_RADIUS_STANDARD,
    "double_outer": config.DOUBLE_OUTER_RADIUS_STANDARD,
}

# Standard values for reference lines
_standard = dict(_radii)

_parallax_shift = 0.0
_parallax_on = False
_miss_adjust = 0.0  # mm adjustment to double_outer


def _load_parallax_shift():
    """Compute the parallax shift from the saved homography."""
    if not config.BOARD_HOMOGRAPHY_PATH.exists():
        return 0.0
    data = np.load(str(config.BOARD_HOMOGRAPHY_PATH), allow_pickle=False)
    H = data["homography"]

    # Replicate the tilt estimation from config.apply_parallax_correction
    cx, cy = config.CANONICAL_CENTER
    r = config.DOUBLE_OUTER_RADIUS_STANDARD
    pts_can = np.array([
        [cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy],
    ], dtype=np.float32).reshape(-1, 1, 2)

    H_inv = np.linalg.inv(H)
    pts_cam = cv2.perspectiveTransform(pts_can, H_inv)

    top = pts_cam[0][0]
    right = pts_cam[1][0]
    bottom = pts_cam[2][0]
    left = pts_cam[3][0]

    top_edge = np.linalg.norm(right - top)
    bottom_edge = np.linalg.norm(bottom - left)
    left_edge = np.linalg.norm(top - left)
    right_edge = np.linalg.norm(right - bottom)

    vert_ratio = min(top_edge, bottom_edge) / max(top_edge, bottom_edge)
    horiz_ratio = min(left_edge, right_edge) / max(left_edge, right_edge)
    ratio = min(vert_ratio, horiz_ratio)
    tilt_angle = math.acos(min(1.0, ratio))

    return config.WIRE_HEIGHT_MM * math.tan(tilt_angle)


def _update_radii():
    """Recalculate working radii from toggles."""
    global _radii
    shift = _parallax_shift if _parallax_on else 0.0
    _radii = {
        "inner_bull": config.INNER_BULL_RADIUS_STANDARD + shift,
        "outer_bull": config.OUTER_BULL_RADIUS_STANDARD + shift,
        "triple_inner": config.TRIPLE_INNER_RADIUS_STANDARD + shift,
        "triple_outer": config.TRIPLE_OUTER_RADIUS_STANDARD + shift,
        "double_inner": config.DOUBLE_INNER_RADIUS_STANDARD + shift,
        "double_outer": config.DOUBLE_OUTER_RADIUS_STANDARD + shift + _miss_adjust,
    }


def _draw_diagnostic(bg_img=None):
    """Draw the board geometry with labeled radii.

    Args:
        bg_img: Optional background image (canonical space). If None,
                draws on a dark background.

    Returns:
        Annotated image (numpy array).
    """
    # Scale everything up for a larger display window.
    # Canonical space: 340x340 at 1px=1mm, center at (170,170).
    # We add margin for labels, then scale up for visibility.
    display_scale = 1.5  # multiplier for window size
    margin = int(40 * display_scale)
    can_px = int(config.CANONICAL_DIAMETER * display_scale)
    size = can_px + 2 * margin
    cx, cy = size // 2, size // 2
    scale = display_scale  # mm -> display pixels

    if bg_img is not None:
        # Scale canonical image and place centered with margin
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)
        scaled_bg = cv2.resize(bg_img, (can_px, can_px))
        img[margin:margin + can_px, margin:margin + can_px] = scaled_bg
    else:
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)

    # Draw filled ring bands (subtle) for context
    overlay = img.copy()
    # Double ring
    cv2.circle(overlay, (cx, cy), int(_radii["double_outer"] * scale), (0, 0, 100), -1)
    cv2.circle(overlay, (cx, cy), int(_radii["double_inner"] * scale), (30, 30, 30), -1)
    # Triple ring
    cv2.circle(overlay, (cx, cy), int(_radii["triple_outer"] * scale), (0, 0, 100), -1)
    cv2.circle(overlay, (cx, cy), int(_radii["triple_inner"] * scale), (30, 30, 30), -1)
    # Outer bull
    cv2.circle(overlay, (cx, cy), int(_radii["outer_bull"] * scale), (0, 80, 0), -1)
    # Inner bull
    cv2.circle(overlay, (cx, cy), int(_radii["inner_bull"] * scale), (0, 0, 140), -1)

    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    # Sector lines
    for i in range(config.NUM_SECTORS):
        boundary_deg = (i * config.SECTOR_SPAN_DEG - config.SECTOR_BOUNDARY_OFFSET) % 360
        boundary_rad = math.radians(boundary_deg)
        r_outer = _radii["double_outer"] * scale
        end_x = int(cx + r_outer * math.sin(boundary_rad))
        end_y = int(cy - r_outer * math.cos(boundary_rad))
        cv2.line(img, (cx, cy), (end_x, end_y), (60, 60, 60), 1)

    # Sector number labels
    label_r = (_radii["double_outer"] + 12) * scale
    for i, sector_num in enumerate(config.SECTOR_ORDER):
        angle_deg = i * config.SECTOR_SPAN_DEG
        angle_rad = math.radians(angle_deg)
        lx = int(cx + label_r * math.sin(angle_rad))
        ly = int(cy - label_r * math.cos(angle_rad))
        cv2.putText(img, str(sector_num), (lx - 8, ly + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Ring boundary circles with labels
    ring_info = [
        ("inner_bull",   "D-BULL inner", (0, 255, 0)),
        ("outer_bull",   "S-BULL outer", (0, 200, 0)),
        ("triple_inner", "Triple inner",  (0, 180, 255)),
        ("triple_outer", "Triple outer",  (0, 140, 255)),
        ("double_inner", "Double inner",  (80, 80, 255)),
        ("double_outer", "Miss cutoff",   (0, 0, 255)),
    ]

    for key, label, color in ring_info:
        r_mm = _radii[key]
        r_px = int(r_mm * scale)
        r_std = _standard[key]

        # Draw current boundary
        cv2.circle(img, (cx, cy), r_px, color, 2)

        # If parallax is on, also draw standard as dashed (dotted via small segments)
        if _parallax_on and key != "double_outer":
            r_std_px = int(r_std * scale)
            _draw_dotted_circle(img, (cx, cy), r_std_px, (80, 80, 80), gap=8)

        # If miss cutoff adjusted, show original
        if key == "double_outer" and _miss_adjust != 0:
            r_orig = (config.DOUBLE_OUTER_RADIUS_STANDARD +
                      (_parallax_shift if _parallax_on else 0.0))
            r_orig_px = int(r_orig * scale)
            _draw_dotted_circle(img, (cx, cy), r_orig_px, (80, 80, 80), gap=8)

        # Label at 45° (upper-right quadrant for readability)
        # Stagger labels to avoid overlap
        label_angles = {
            "inner_bull": 30, "outer_bull": 60,
            "triple_inner": 120, "triple_outer": 150,
            "double_inner": 210, "double_outer": 315,
        }
        angle = math.radians(label_angles[key])
        lx = int(cx + r_px * math.sin(angle))
        ly = int(cy - r_px * math.cos(angle))

        text = f"{label}: {r_mm:.1f}mm"
        if _parallax_on and abs(r_mm - r_std) > 0.01:
            text += f" (std: {r_std:.1f})"

        # Background rect for readability
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        tx, ty = lx + 4, ly - 2
        cv2.rectangle(img, (tx - 1, ty - th - 2), (tx + tw + 1, ty + 2), (0, 0, 0), -1)
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Status text at top
    status_lines = []
    shift = _parallax_shift if _parallax_on else 0.0
    if _parallax_on:
        status_lines.append(f"Parallax: ON (shift={_parallax_shift:.2f}mm, "
                           f"tilt~{math.degrees(math.atan2(_parallax_shift, config.WIRE_HEIGHT_MM)):.1f}deg)")
    else:
        status_lines.append(f"Parallax: OFF (available shift={_parallax_shift:.2f}mm)")

    if _miss_adjust != 0:
        status_lines.append(f"Miss cutoff adjusted: {_miss_adjust:+.1f}mm")

    status_lines.append("[P] parallax  [+/-] miss cutoff  [R] reset  [S] print  [Q] quit")

    for i, line in enumerate(status_lines):
        cv2.putText(img, line, (10, 18 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return img


def _draw_dotted_circle(img, center, radius, color, gap=6):
    """Draw a dotted circle by drawing short arcs."""
    cx, cy = center
    circumference = 2 * math.pi * radius
    n_dots = max(int(circumference / gap), 20)
    for i in range(0, n_dots, 2):
        angle1 = 2 * math.pi * i / n_dots
        angle2 = 2 * math.pi * (i + 1) / n_dots
        pt1 = (int(cx + radius * math.cos(angle1)),
               int(cy + radius * math.sin(angle1)))
        pt2 = (int(cx + radius * math.cos(angle2)),
               int(cy + radius * math.sin(angle2)))
        cv2.line(img, pt1, pt2, color, 1)


def _print_radii():
    """Print current radii to console."""
    shift = _parallax_shift if _parallax_on else 0.0
    print(f"\n{'='*50}")
    print(f"Board Geometry — parallax {'ON' if _parallax_on else 'OFF'}"
          f" (shift={shift:.2f}mm)")
    print(f"{'='*50}")
    for key in ("inner_bull", "outer_bull", "triple_inner",
                "triple_outer", "double_inner", "double_outer"):
        std = _standard[key]
        cur = _radii[key]
        delta = cur - std
        extra = ""
        if key == "double_outer" and _miss_adjust != 0:
            extra = f"  (miss adjust: {_miss_adjust:+.1f}mm)"
        if abs(delta) > 0.01:
            print(f"  {key:>14s}: {cur:6.1f}mm  (standard: {std:.1f}, delta: {delta:+.2f}){extra}")
        else:
            print(f"  {key:>14s}: {cur:6.1f}mm{extra}")
    print(f"\n  Miss boundary = everything beyond {_radii['double_outer']:.1f}mm from center")
    miss_inches = _radii['double_outer'] / 25.4
    print(f"                = {miss_inches:.2f}\" radius = {miss_inches*2:.2f}\" diameter")
    print()


def main():
    parser = argparse.ArgumentParser(description="Dartboard geometry visualizer")
    parser.add_argument("--camera", action="store_true",
                       help="Overlay on live canonical camera feed")
    args = parser.parse_args()

    global _parallax_shift, _parallax_on, _miss_adjust

    # Load parallax data if available
    _parallax_shift = _load_parallax_shift()
    _update_radii()

    # Camera mode: load homography and show canonical view
    homography = None
    cap = None
    if args.camera:
        if config.BOARD_HOMOGRAPHY_PATH.exists():
            data = np.load(str(config.BOARD_HOMOGRAPHY_PATH), allow_pickle=False)
            homography = data["homography"]
        else:
            print("WARNING: No homography found, camera mode won't show canonical view")

        from dartscorer.calibrate import open_camera, load_crop_roi, load_crop_rotation, apply_crop
        cap = open_camera()
        crop_roi = load_crop_roi()
        crop_rotation = load_crop_rotation()

    window = "Board Geometry"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 630, 630)

    print("Board Geometry Visualizer")
    print(f"  Parallax shift available: {_parallax_shift:.2f}mm")
    _print_radii()

    while True:
        bg = None
        if cap is not None and homography is not None:
            ret, frame = cap.read()
            if ret:
                frame = apply_crop(frame, crop_roi, crop_rotation)
                # Warp to canonical
                size = config.CANONICAL_DIAMETER
                bg = cv2.warpPerspective(frame, homography, (size, size))

        display = _draw_diagnostic(bg)
        cv2.imshow(window, display)

        key = cv2.waitKey(30 if cap else 100) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('p'):
            _parallax_on = not _parallax_on
            _update_radii()
            state = "ON" if _parallax_on else "OFF"
            print(f"Parallax: {state} (shift={_parallax_shift if _parallax_on else 0:.2f}mm)")
        elif key in (ord('+'), ord('='), 171):  # + or = or numpad+
            _miss_adjust += 1.0
            _update_radii()
            print(f"Miss cutoff: {_radii['double_outer']:.1f}mm ({_miss_adjust:+.1f}mm)")
        elif key in (ord('-'), 173):  # - or numpad-
            _miss_adjust -= 1.0
            _update_radii()
            print(f"Miss cutoff: {_radii['double_outer']:.1f}mm ({_miss_adjust:+.1f}mm)")
        elif key == ord('r'):
            _parallax_on = False
            _miss_adjust = 0.0
            _update_radii()
            print("Reset to standard radii.")
        elif key == ord('s'):
            _print_radii()

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    # If adjustments were made, offer to show the config change
    if _miss_adjust != 0:
        new_outer = config.DOUBLE_OUTER_RADIUS_STANDARD + _miss_adjust
        print(f"\nTo apply your miss cutoff adjustment, update config.py:")
        print(f"  DOUBLE_OUTER_RADIUS_STANDARD = {new_outer:.1f}  # was {config.DOUBLE_OUTER_RADIUS_STANDARD}")


if __name__ == "__main__":
    main()
