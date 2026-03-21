#!/usr/bin/env python3
"""
calibrate.py — Camera lens calibration and dartboard homography setup.

Usage:
    python calibrate.py --lens           # Lens distortion calibration (Phase 1)
    python calibrate.py --board          # Board homography calibration (Phase 2)
    python calibrate.py --lens --board   # Both in sequence
    python calibrate.py --debug          # Enable debug visualization (combine with above)
"""

import argparse
import math
import os
import sys
import time

# Suppress Qt/Wayland warnings from pip-installed OpenCV on Linux
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
import config
import board


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def open_camera():
    """Open the configured camera device and set resolution.

    Uses the V4L2 backend explicitly to ensure resolution control works
    on Linux systems.
    """
    # Use V4L2 backend for reliable resolution setting on Linux
    cap = cv2.VideoCapture(config.CAMERA_DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback to default backend
        cap = cv2.VideoCapture(config.CAMERA_DEVICE)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera at {config.CAMERA_DEVICE}")
        print("Check that the device exists and is not in use by another process.")
        sys.exit(1)
    # Set FOURCC to MJPG first — many USB cameras only support high res via MJPEG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w != config.CAMERA_WIDTH or actual_h != config.CAMERA_HEIGHT:
        print(f"WARNING: Requested {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT} "
              f"but got {actual_w}x{actual_h}")
    print(f"Camera opened: {config.CAMERA_DEVICE} at {actual_w}x{actual_h}")
    return cap


# ---------------------------------------------------------------------------
# Lens parameter I/O
# ---------------------------------------------------------------------------

def load_lens_params():
    """Load and return (camera_matrix, dist_coeffs, new_camera_matrix, roi)."""
    path = config.LENS_PARAMS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Lens parameters not found at {path}.\n"
            "Run 'python calibrate.py --lens' to generate them."
        )
    data = np.load(str(path))
    return (
        data["camera_matrix"],
        data["dist_coeffs"],
        data["new_camera_matrix"],
        data["roi"],
    )


def undistort_frame(frame, camera_matrix, dist_coeffs, new_camera_matrix, roi):
    """Undistort a frame using the given lens parameters.

    Applies cv2.undistort and optionally crops to the valid ROI.
    Returns the undistorted (and cropped) frame.
    """
    dst = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = [int(v) for v in roi]
    if w > 0 and h > 0:
        dst = dst[y:y + h, x:x + w]
    return dst


# ---------------------------------------------------------------------------
# Phase 1 — Lens calibration
# ---------------------------------------------------------------------------

def calibrate_lens(cap, debug=False):
    """Collect checkerboard frames and compute lens distortion parameters."""

    # Object points: 3-D coordinates of each inner corner (Z=0 plane).
    # Scaled by the physical square size so units are in millimetres.
    cols, rows = config.CHECKERBOARD_SIZE
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= config.CHECKERBOARD_SQUARE_SIZE

    obj_points = []   # 3-D points in world space
    img_points = []   # Corresponding 2-D points in image space

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    last_capture_time = 0.0
    captured = 0
    target = config.MIN_CALIBRATION_FRAMES
    gray = None  # will be set on first frame

    print("\n=== Lens Calibration ===")
    print(f"Hold the {cols}x{rows} checkerboard in front of the camera.")
    print(f"Need {target} frames. Auto-capture every {config.CALIBRATION_DELAY}s when detected.")
    print("Controls: SPACE = force capture  |  q = abort\n")

    window = "Lens Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while captured < target:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from camera.")
                break

            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            found, corners = cv2.findChessboardCorners(gray, config.CHECKERBOARD_SIZE, None)

            now = time.monotonic()
            do_capture = False

            if found:
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                if debug:
                    cv2.drawChessboardCorners(display, config.CHECKERBOARD_SIZE, corners_refined, found)

                # Auto-capture when enough time has elapsed
                if now - last_capture_time >= config.CALIBRATION_DELAY:
                    do_capture = True

                status_text = "Checkerboard detected"
                status_color = (0, 200, 0)
            else:
                corners_refined = None
                status_text = "No checkerboard detected"
                status_color = (0, 0, 220)

            # Overlay HUD
            h_frame, w_frame = display.shape[:2]
            overlay_y = 30
            cv2.putText(display, status_text, (10, overlay_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            overlay_y += 35
            cv2.putText(display, f"Captured: {captured} / {target}", (10, overlay_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            overlay_y += 35
            cv2.putText(display,
                        "Hold checkerboard steady - press SPACE to capture, or auto-capture when detected",
                        (10, overlay_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Aborted by user.")
                cv2.destroyWindow(window)
                return False
            elif key == ord(' ') and found and corners_refined is not None:
                # Manual capture via SPACE
                do_capture = True

            if do_capture and found and corners_refined is not None:
                obj_points.append(objp)
                img_points.append(corners_refined)
                last_capture_time = now
                captured += 1
                print(f"  Captured frame {captured}/{target}")

    except KeyboardInterrupt:
        print("\nCalibration interrupted.")
        cv2.destroyWindow(window)
        return False

    cv2.destroyWindow(window)

    if captured < target:
        print(f"ERROR: Only captured {captured}/{target} frames. Calibration aborted.")
        return False

    # ---------------------------------------------------------------------------
    # Compute calibration
    # ---------------------------------------------------------------------------
    print(f"\nComputing calibration from {captured} frames...")
    h_img, w_img = gray.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w_img, h_img), None, None
    )

    print(f"Reprojection error: {ret:.4f} px")
    if ret > 1.0:
        print("WARNING: Reprojection error > 1.0 px — calibration may be poor.")
        print("Try again with more varied checkerboard poses and better lighting.")

    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w_img, h_img), 1, (w_img, h_img))

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    save_path = config.LENS_PARAMS_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(save_path),
        camera_matrix=mtx,
        dist_coeffs=dist,
        new_camera_matrix=new_mtx,
        roi=roi,
    )
    print(f"Lens parameters saved to: {save_path}")

    # ---------------------------------------------------------------------------
    # Debug: side-by-side comparison on last frame
    # ---------------------------------------------------------------------------
    if debug:
        # Re-read one more frame to show undistorted preview
        ok, sample = cap.read()
        if ok:
            undist = undistort_frame(sample, mtx, dist, new_mtx, roi)
            # Resize both to the same height for side-by-side display
            h_s = min(sample.shape[0], undist.shape[0])
            def _resize_h(img, target_h):
                scale = target_h / img.shape[0]
                return cv2.resize(img, (int(img.shape[1] * scale), target_h))

            left = _resize_h(sample, h_s)
            right = _resize_h(undist, h_s)

            # Label each side
            cv2.putText(left,  "Original",    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(right, "Undistorted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            combined = np.hstack([left, right])
            cmp_win = "Distortion comparison (any key to close)"
            cv2.namedWindow(cmp_win, cv2.WINDOW_NORMAL)
            cv2.imshow(cmp_win, combined)
            print("Showing distortion comparison — press any key to close.")
            cv2.waitKey(0)
            cv2.destroyWindow(cmp_win)

    print("Lens calibration complete.\n")
    return True


# ---------------------------------------------------------------------------
# Phase 2 — Board calibration (manual 21-point click)
# ---------------------------------------------------------------------------


def _compute_canonical_destinations(scale_factor=1.0):
    """Return 21 (x, y) float tuples for the canonical board coordinate system.

    Point 0: bullseye center at (170, 170).
    Points 1-20: wire intersections at the outer double ring — where sector
      boundary wires cross the outer wire.  These are physical features
      visible on the board, easier to click than sector midpoints.
      angle = i * 18 - 9 degrees, measured clockwise from 12 o'clock.
    """
    cx, cy = config.CANONICAL_CENTER   # (170, 170)
    r = config.DOUBLE_OUTER_RADIUS     # 170
    destinations = [(float(cx), float(cy))]  # bullseye center
    for i in range(20):
        angle = i * 18.0 - 9.0  # sector boundary, not midpoint
        x = cx + (r * scale_factor) * math.sin(math.radians(angle))
        y = cy - (r * scale_factor) * math.cos(math.radians(angle))
        destinations.append((x, y))
    return destinations


def _make_click_colors(n):
    """Return n visually distinct BGR color tuples using evenly-spaced HSV hues."""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)  # OpenCV hue range 0-179
        bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in bgr))
    return colors


# Mouse callback state for point collection
_click_points = []
_click_frame = None


def _mouse_callback(event, x, y, flags, param):
    """Mouse callback for collecting calibration clicks."""
    global _click_points, _click_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_points.append((x, y))


def calibrate_board(cap, debug=False):
    """Phase 2: Manual 21-point board calibration.

    The user clicks 21 known landmarks on the dartboard to establish both
    the board's shape (perspective) and rotational orientation (where 20 is).

    Points to click (in order):
    1. Bullseye center
    2-21. Wire intersections at the outer double ring — where each sector
          boundary wire crosses the outer wire, clockwise from the 5/20
          boundary (just left of 12 o'clock).

    Wire intersections are physically visible crossings, much easier to
    click precisely than sector midpoints.
    """
    global _click_points, _click_frame

    # --- Load lens calibration ---
    try:
        cam_mtx, dist, new_mtx, roi = load_lens_params()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return False

    print("\n=== Board Calibration (Manual 21-Point) ===")
    print("Position camera so the full dartboard is visible.")
    print("Press SPACE to capture a frame, then click 21 points.\n")

    window = "Board Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    captured_frame = None

    # --- Step 1: Capture a frame ---
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from camera.")
                return False

            undistorted = undistort_frame(frame, cam_mtx, dist, new_mtx, roi)
            display = undistorted.copy()

            cv2.putText(display, "Press SPACE to capture frame for calibration",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "Ensure full dartboard is visible | q = abort",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Aborted by user.")
                cv2.destroyWindow(window)
                return False
            elif key == ord(' '):
                captured_frame = undistorted.copy()
                print("Frame captured.\n")
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
        cv2.destroyWindow(window)
        return False

    # --- Step 2: Collect 21 clicks ---
    _click_points = []
    _click_frame = captured_frame.copy()
    cv2.setMouseCallback(window, _mouse_callback)

    # Build point labels — each outer click is a wire intersection between
    # two adjacent sectors.  Boundary i (0-based) sits between
    # SECTOR_ORDER[(i-1) % 20] and SECTOR_ORDER[i].
    SO = config.SECTOR_ORDER
    point_labels = ["1/21: Click the BULLSEYE (center of board)"]
    click_short_labels = ["Bull"]
    for i in range(20):
        left_sector = SO[(i - 1) % 20]
        right_sector = SO[i]
        hint = ""
        if i == 0:
            hint = " (near top, start here)"
        elif i == 10:
            hint = " (near bottom)"
        point_labels.append(
            f"{i+2}/21: Wire crossing — {left_sector}/{right_sector}{hint}")
        click_short_labels.append(f"{left_sector}|{right_sector}")

    colors = _make_click_colors(21)

    print("Click 21 points on the board in order:")
    print(f"  {point_labels[0]}")
    for i in range(1, 21):
        print(f"  {point_labels[i]}")
    print("Press 'u' to undo last click, 'q' to abort.\n")

    try:
        while len(_click_points) < 21:
            display = captured_frame.copy()

            # Draw instruction for the next point
            idx = len(_click_points)
            cv2.putText(display, point_labels[idx],
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "u = undo last | q = abort",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Draw already-clicked points with labels
            for i, pt in enumerate(_click_points):
                cv2.circle(display, pt, 6, colors[i], 2)
                cv2.circle(display, pt, 2, colors[i], -1)
                cv2.putText(display, click_short_labels[i], (pt[0] + 8, pt[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)

            # Draw spokes from center (click 0) to each subsequent click
            if len(_click_points) >= 2:
                for i in range(1, len(_click_points)):
                    cv2.line(display, _click_points[0], _click_points[i],
                             colors[i], 1, cv2.LINE_AA)

            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("Aborted by user.")
                cv2.destroyWindow(window)
                return False
            elif key == ord('u') and _click_points:
                removed = _click_points.pop()
                print(f"  Undid click at ({removed[0]}, {removed[1]})")
    except KeyboardInterrupt:
        print("\nInterrupted.")
        cv2.destroyWindow(window)
        return False

    # --- Step 3: Diagnostics + Compute homography ---
    center_px = _click_points[0]
    radii = []
    for i in range(1, 21):
        dx = _click_points[i][0] - center_px[0]
        dy = _click_points[i][1] - center_px[1]
        radii.append((dx**2 + dy**2)**0.5)
    print(f"\nCollected 21 points:")
    print(f"  Bullseye: ({center_px[0]}, {center_px[1]})")
    print(f"  Outer point radii: min={min(radii):.1f}px, max={max(radii):.1f}px, "
          f"mean={np.mean(radii):.1f}px")

    dst_positions = _compute_canonical_destinations(1.0)
    src_pts = np.array(_click_points, dtype=np.float32)
    dst_pts = np.array(dst_positions, dtype=np.float32)

    H, mask = cv2.findHomography(src_pts, dst_pts, 0)

    if H is None:
        print("ERROR: Failed to compute homography.")
        cv2.destroyWindow(window)
        return False

    print(f"\n--- Homography verification (21 points) ---")
    orig = np.array(_click_points, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(orig, H)
    errors = []
    for i in range(21):
        wx, wy = warped[i][0]
        ex, ey = dst_positions[i]
        err = ((wx - ex)**2 + (wy - ey)**2) ** 0.5
        errors.append(err)
        label = click_short_labels[i]
        print(f"    {label}: clicked ({_click_points[i][0]}, {_click_points[i][1]}) "
              f"→ ({wx:.1f}, {wy:.1f}), expected ({ex:.0f}, {ey:.0f}), err={err:.1f}px")
    print(f"  Mean error: {np.mean(errors):.2f}px, max: {np.max(errors):.2f}px")
    print(f"---")

    # --- Step 4: Interactive scale adjustment ---
    scale_factor = 1.0
    scale_step = 0.005  # 0.5% per keypress

    canonical_size = config.CANONICAL_DIAMETER
    canon_win = "Canonical View — [ ] to scale, ENTER to accept, q to abort"
    cv2.namedWindow(canon_win, cv2.WINDOW_NORMAL)

    # Draw the clicked points on the original frame for the debug view
    debug_frame = captured_frame.copy()
    for i, pt in enumerate(_click_points):
        cv2.circle(debug_frame, pt, 6, colors[i], 2)
        cv2.circle(debug_frame, pt, 2, colors[i], -1)
    for i in range(1, 21):
        cv2.line(debug_frame, _click_points[0], _click_points[i],
                 colors[i], 1, cv2.LINE_AA)
    cv2.imshow(window, debug_frame)

    print("\nVerify the canonical view: ring bands should align with the board.")
    print("Controls:")
    print("  ] = grow rings (if overlay is too small)")
    print("  [ = shrink rings (if overlay is too large)")
    print("  ENTER = accept and save")
    print("  q = abort and redo\n")

    while True:
        scaled_dst = np.array(_compute_canonical_destinations(scale_factor), dtype=np.float32)
        H_scaled, _ = cv2.findHomography(src_pts, scaled_dst, 0)

        if H_scaled is None:
            print("ERROR: Homography failed at this scale.")
            break

        warped_frame = cv2.warpPerspective(captured_frame, H_scaled,
                                           (canonical_size, canonical_size))
        warped_overlay = board.draw_board_overlay(warped_frame, alpha=0.4)

        scale_text = f"Scale: {scale_factor:.3f}  [ ] to adjust, ENTER to accept"
        cv2.putText(warped_overlay, scale_text, (5, canonical_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow(canon_win, warped_overlay)

        key = cv2.waitKey(30) & 0xFF
        if key == ord(']'):
            scale_factor += scale_step
            print(f"  Scale: {scale_factor:.3f} (growing)")
        elif key == ord('['):
            scale_factor -= scale_step
            if scale_factor < 0.5:
                scale_factor = 0.5
            print(f"  Scale: {scale_factor:.3f} (shrinking)")
        elif key == 13 or key == 10:  # ENTER
            H = H_scaled
            break
        elif key == ord('q'):
            print("Calibration aborted.")
            cv2.destroyWindow(window)
            cv2.destroyWindow(canon_win)
            return False

    cv2.destroyWindow(window)
    cv2.destroyWindow(canon_win)

    if scale_factor != 1.0:
        print(f"Applied scale adjustment: {scale_factor:.3f}")

    # --- Step 5: ROI computation + save ---
    all_clicks = np.array(_click_points)
    x_min = int(np.min(all_clicks[:, 0])) - config.ROI_PADDING
    y_min = int(np.min(all_clicks[:, 1])) - config.ROI_PADDING
    x_max = int(np.max(all_clicks[:, 0])) + config.ROI_PADDING
    y_max = int(np.max(all_clicks[:, 1])) + config.ROI_PADDING

    # Clamp to frame bounds
    h_frame, w_frame = captured_frame.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w_frame, x_max)
    y_max = min(h_frame, y_max)

    board_roi = np.array([x_min, y_min, x_max, y_max], dtype=np.int32)
    print(f"Board ROI: ({x_min}, {y_min}) to ({x_max}, {y_max}) = {x_max-x_min}x{y_max-y_min}px")

    save_path = config.BOARD_HOMOGRAPHY_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    center_pt = _click_points[0]
    np.savez(str(save_path), homography=H,
             ellipse_center=np.array(center_pt, dtype=np.float64),
             click_points=np.array(_click_points, dtype=np.float64),
             scale_factor=scale_factor,
             board_roi=board_roi)
    print(f"Board homography saved to: {save_path}")

    print("Board calibration complete.\n")
    return True


# ---------------------------------------------------------------------------
# Phase 3 — Crop ROI selection
# ---------------------------------------------------------------------------

def load_crop_roi():
    """Load and return the crop ROI (x, y, w, h) or None if not set."""
    path = config.CROP_ROI_PATH
    if not path.exists():
        return None
    data = np.load(str(path))
    roi = data["crop_roi"]
    return tuple(int(v) for v in roi)


def apply_crop(frame, crop_roi):
    """Crop a frame to the saved ROI. Returns cropped frame."""
    if crop_roi is None:
        return frame
    x, y, w, h = crop_roi
    return frame[y:y+h, x:x+w].copy()


_crop_dragging = False
_crop_start = None
_crop_rect = None


def _crop_mouse_callback(event, x, y, flags, param):
    global _crop_dragging, _crop_start, _crop_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        _crop_dragging = True
        _crop_start = (x, y)
        _crop_rect = None
    elif event == cv2.EVENT_MOUSEMOVE and _crop_dragging:
        _crop_rect = (_crop_start[0], _crop_start[1], x, y)
    elif event == cv2.EVENT_LBUTTONUP and _crop_dragging:
        _crop_dragging = False
        _crop_rect = (_crop_start[0], _crop_start[1], x, y)


def calibrate_crop(cap, debug=False):
    """Interactive crop ROI selection.

    Draw a rectangle on the live camera feed to select the region of
    interest (just the dartboard + some margin). The crop is applied
    before all other processing in the pipeline.
    """
    global _crop_dragging, _crop_start, _crop_rect

    print("\n=== Crop ROI Selection ===")
    print("Draw a rectangle around the dartboard area.")
    print("Controls:")
    print("  Click+drag  Draw crop rectangle")
    print("  ENTER       Accept and save")
    print("  R           Reset rectangle")
    print("  Q           Abort")
    print()

    window = "Select Crop Region"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _crop_mouse_callback)

    _crop_rect = None
    _crop_dragging = False
    _crop_start = None

    # Load existing crop for reference
    existing_roi = load_crop_roi()
    if existing_roi is not None:
        x, y, w, h = existing_roi
        _crop_rect = (x, y, x + w, y + h)
        print(f"Existing crop: ({x}, {y}) {w}x{h} — adjust or ENTER to keep")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame.")
                return False

            display = frame.copy()
            h_frame, w_frame = display.shape[:2]

            # Draw current rectangle
            if _crop_rect is not None:
                x1, y1, x2, y2 = _crop_rect
                # Normalize coordinates
                rx1, rx2 = min(x1, x2), max(x1, x2)
                ry1, ry2 = min(y1, y2), max(y1, y2)
                rx1 = max(0, rx1)
                ry1 = max(0, ry1)
                rx2 = min(w_frame, rx2)
                ry2 = min(h_frame, ry2)

                # Dim outside the rectangle
                overlay = display.copy()
                overlay[:ry1, :] = overlay[:ry1, :] // 3
                overlay[ry2:, :] = overlay[ry2:, :] // 3
                overlay[ry1:ry2, :rx1] = overlay[ry1:ry2, :rx1] // 3
                overlay[ry1:ry2, rx2:] = overlay[ry1:ry2, rx2:] // 3
                display = overlay

                cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
                size_text = f"{rx2 - rx1}x{ry2 - ry1}"
                cv2.putText(display, size_text, (rx1, ry1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(display, "Draw rectangle around dartboard | ENTER=save  R=reset  Q=abort",
                        (10, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                print("Aborted.")
                cv2.destroyWindow(window)
                return False

            elif key == ord('r'):
                _crop_rect = None
                print("  Rectangle reset")

            elif key in (13, 10):  # ENTER
                if _crop_rect is None:
                    print("  No rectangle drawn — draw one first")
                    continue

                x1, y1, x2, y2 = _crop_rect
                rx1, rx2 = min(x1, x2), max(x1, x2)
                ry1, ry2 = min(y1, y2), max(y1, y2)
                rx1 = max(0, rx1)
                ry1 = max(0, ry1)
                rx2 = min(w_frame, rx2)
                ry2 = min(h_frame, ry2)

                crop_w = rx2 - rx1
                crop_h = ry2 - ry1
                if crop_w < 100 or crop_h < 100:
                    print(f"  Rectangle too small ({crop_w}x{crop_h}) — draw a larger one")
                    continue

                roi = np.array([rx1, ry1, crop_w, crop_h], dtype=np.int32)
                save_path = config.CROP_ROI_PATH
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(str(save_path), crop_roi=roi)
                print(f"Crop ROI saved: ({rx1}, {ry1}) {crop_w}x{crop_h}")
                print(f"  Saved to {save_path}")

                cv2.destroyWindow(window)
                return True

    except KeyboardInterrupt:
        print("\nInterrupted.")
        cv2.destroyWindow(window)
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dartboard calibration tool")
    parser.add_argument("--lens",  action="store_true", help="Run lens distortion calibration")
    parser.add_argument("--board", action="store_true", help="Run board homography calibration")
    parser.add_argument("--crop",  action="store_true", help="Select crop ROI (dartboard region)")
    parser.add_argument("--debug", action="store_true", help="Show debug visualizations")
    args = parser.parse_args()

    if not args.lens and not args.board and not args.crop:
        parser.print_help()
        sys.exit(1)

    cap = open_camera()

    try:
        if args.lens:
            if not calibrate_lens(cap, debug=args.debug):
                print("Lens calibration failed. Aborting.")
                return
        if args.crop:
            if not calibrate_crop(cap, debug=args.debug):
                print("Crop ROI selection failed.")
                return
        if args.board:
            calibrate_board(cap, debug=args.debug)
    except KeyboardInterrupt:
        print("\nInterrupted — exiting cleanly.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
